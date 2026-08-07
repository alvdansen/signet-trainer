"""Hook-based transformer block CPU<->GPU weight swapping for VRAM reduction.

================================================================================
⚠️  PEFT-LoRA-aware (OFFL-03) — re-validated for FREEZE-MECHANICS training only  ⚠️
================================================================================
Harvest provenance (OFFL-01): this module originated VERBATIM from
``alvdansen/flimmer-trainer-dev`` (``flimmer/training/wan/block_swap.py``) — a
**musubi-tuner-derived**, **Wan-validated, NOT LTX-validated** offloader. No
block-swap offloader had ever been confirmed working on LTX-2.3.

It carried the **BSWP-TRAINING-01** training bug, which 07-12 run 4 ROOT-CAUSED on
LTX-2.3 (the definitive ``mat2 is on cpu`` device mismatch): ``_get_linear_weights``
matched the PEFT ``lora.Linear`` wrapper (whose ``.weight`` proxies
``base_layer.weight``) and then SKIPPED all of its descendants, so ``base_layer``
and every adapter's ``lora_A``/``lora_B`` sub-Linear went untracked. With TWO active
adapters (frozen + default), a freshly ``load_adapter``-ed frozen LoRA sub-linear ran
on CPU while its block ran on GPU. The ``weights[0]``-only forward-pre-hook guard
never noticed.

FIX (07-13, under the operator's explicit OFFL-03 authorization — "block swapping never
worked and broke all of flimmer; fix it now"): this file is **no longer verbatim**.
``_get_linear_weights`` is now PEFT-LoRA-aware — it descends into PEFT wrappers and
yields the real weight-owning leaves (``base_layer`` + every active adapter's
``lora_A``/``lora_B``). Only the large frozen ``base_layer`` (+ plain) weights are
swapped to CPU (:func:`_get_swappable_weights`); the tiny lora sub-Linears are kept
GPU-RESIDENT so adamw8bit's trainable state stays CUDA and a late-loaded frozen lora
never strands. The device guard (:func:`_block_needs_gpu`) now scans ALL tracked
weights and re-homes any off-device weight, so the crash is closed.

VALIDATION SCOPE: re-validated as a FREEZE-MECHANICS run (two-adapter forward /
backward / optimizer-step + frozen-excluded checkpoint) on LTX-2.3, single A100-80GB,
blocks_to_swap 16 (07-13 Task 2). This is NOT a long-run adapter-QUALITY validation —
weight-corruption over many steps remains unproven; treat as a scoped warning, not
"blanket trusted".

**Baseline-first (D-OFF-1):** ``blocks_to_swap=0`` still makes the offloader **inert**
(constructor early-return below → zero hooks, zero behavior change).

The Wan-topology prose (the 40-block sizing notes) is kept for context; ``num_blocks``
is read from the passed ``blocks`` ModuleList, NOT a carried Wan constant.
================================================================================

This module provides block-level CPU<->GPU weight swapping for training VRAM
reduction. Based on the musubi-tuner Offloader algorithm but implemented via
PyTorch hooks for compatibility with standard diffusers models.

How it works:
    Wan 2.2 has 40 transformer blocks, each ~175MB at fp8 (~350MB at bf16).
    Block swapping offloads inactive blocks to CPU during forward/backward
    passes, keeping only the currently-executing block and a configurable
    number of "resident" blocks on GPU at all times.

    A secondary CUDA stream handles async weight transfers so that the GPU
    can compute on one block while transferring the next. Pinned (page-locked)
    CPU memory enables direct DMA access by the GPU, roughly 2-3x faster
    than pageable memory transfers.

Algorithm (forward pass, blocks 0..N-1):
    1. forward_pre_hook(block[i]): Synchronize -- wait for any pending async
       transfer of block[i] to complete on the secondary stream.
    2. Block[i] executes forward pass on GPU.
    3. forward_hook(block[i]): If block[i] is in the swappable range, submit
       async transfer: move block[i] Linear weights to CPU pinned buffers,
       and prefetch the next block from CPU to GPU.

Algorithm (backward pass, blocks N-1..0):
    1. full_backward_hook(block[i]): After backward through block[i], submit
       async offload of block[i] to CPU. Prefetch block[i-1] if it needs
       to come from CPU.

Only plain Linear module weights are swapped: the ``endswith("Linear")`` class-name predicate
matches ``nn.Linear`` (and the PEFT ``lora.Linear`` wrapper, which is then skipped via
``base_layer``); it does NOT match bitsandbytes ``Linear8bitLt`` / ``Linear4bit`` (their class
names end in ``Lt`` / ``4bit``). The LTX-2.3 base transformer is plain bf16 ``nn.Linear``, so this
is plain-Linear-only by design. Buffers (e.g., LayerNorm parameters) stay on GPU.

PEFT-aware weight enumerators live in :mod:`signet_trainer.lora.peft` (single source of truth,
D-9-DEDUP); this module imports them and keeps only the torch.device guard helpers local.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

# Single-source PEFT-aware weight enumerators (promoted 09-01 from this module into lora/peft.py so
# there is exactly ONE implementation of the 07-13 BSWP-TRAINING-01 fix — D-9-DEDUP). The private
# aliases keep every call site below (_block_needs_gpu, the movers) unchanged.
from signet_trainer.lora.peft import (
    _is_lora_weight_name,
    get_linear_weights as _get_linear_weights,
    get_swappable_weights as _get_swappable_weights,
)

logger = logging.getLogger(__name__)

# Threshold for logging a warning about extreme swap counts.
# Above this, training speed is significantly impacted.
_EXTREME_SWAP_THRESHOLD = 35

# Re-export the promoted enumerators + the lora-name predicate so existing importers of these private
# names from this module keep resolving (they are the offloader's contract with lora/peft.py).
__all__ = [
    "BlockSwapOffloader",
    "_block_needs_gpu",
    "_get_linear_weights",
    "_get_swappable_weights",
    "_is_lora_weight_name",
    "_same_device",
]


def _same_device(a: torch.device, b: torch.device) -> bool:
    """True if two ``torch.device``s address the SAME physical device (index-normalized).

    WR-03: raw ``==`` on ``torch.device`` is index-SENSITIVE — ``torch.device("cuda") !=
    torch.device("cuda:0")`` even though a weight loaded on ``"cuda"`` materializes on ``cuda:0``.
    The offloader is constructed with ``device=torch.device("cuda")`` (no index, fns.py:365-369), so
    an index-sensitive compare fires on EVERY GPU-resident weight — making :func:`_block_needs_gpu`
    always-true on GPU (redundant re-home + stream sync per block per forward) and non-discriminating
    (it can never legitimately return False). Canonicalize the index before comparing: a ``None`` cuda
    index resolves to the current device; a ``None`` cpu index is 0. CPU-safe (no CUDA required — a
    ``None`` cuda index falls back to 0 when CUDA is unavailable, so the predicate is unit-testable).
    """
    if a.type != b.type:
        return False

    def _index(d: torch.device) -> int:
        if d.index is not None:
            return d.index
        if d.type == "cuda" and torch.cuda.is_available():
            return torch.cuda.current_device()
        return 0

    return _index(a) == _index(b)


def _block_needs_gpu(block: nn.Module, device: torch.device) -> bool:
    """True if ANY tracked weight of ``block`` is off ``device`` (the all-weights guard predicate).

    Replaces the harvested ``weights[0]``-only check that only ever inspected the base weight and
    thus MISSED an off-device lora sub-linear (BSWP-TRAINING-01). Re-walks :func:`_get_linear_weights`
    fresh each call, so it discovers late-loaded ``lora_A.frozen``/``lora_B.frozen`` submodules that
    did not exist when the offloader was constructed. CPU-unit-testable (no CUDA stream needed).

    Device comparison is index-NORMALIZED via :func:`_same_device` (WR-03): the offloader is built
    with an index-less ``torch.device("cuda")`` while GPU weights live on ``cuda:0``, so a raw ``!=``
    was always-true and the predicate never discriminated.
    """
    for module, _ in _get_linear_weights(block):
        if not _same_device(module.weight.data.device, device):
            return True
    return False


class BlockSwapOffloader:
    """Hook-based transformer block offloader for VRAM reduction.

    Registers forward_pre_hook, forward_hook, and full_backward_hook on
    transformer blocks to move their Linear weights between GPU and CPU
    during training. Uses a secondary CUDA stream and pinned memory for
    efficient async transfers.

    Args:
        blocks: The transformer's nn.ModuleList of blocks.
        blocks_to_swap: Number of blocks to offload to CPU. Clamped to
            num_blocks - 1 (at least one block must remain on GPU).
        device: The GPU device for training (e.g., torch.device("cuda:0")).

    Usage:
        offloader = BlockSwapOffloader(model.blocks, blocks_to_swap=20, device=device)
        # ... training loop runs; hooks handle weight transfers automatically ...
        offloader.remove_hooks()  # cleanup when done
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        blocks_to_swap: int,
        device: torch.device,
    ) -> None:
        self.num_blocks: int = len(blocks)
        self.device: torch.device = device
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._pinned_buffers: dict[int, list[torch.Tensor]] = {}

        # Clamp to num_blocks - 1 (can't swap ALL blocks)
        self.blocks_to_swap: int = min(blocks_to_swap, self.num_blocks - 1)

        if self.blocks_to_swap <= 0:
            # No swapping -- nothing to do
            self.blocks_to_swap = 0
            return

        if self.blocks_to_swap != blocks_to_swap:
            logger.warning(
                "blocks_to_swap=%d exceeds maximum (%d). Clamped to %d.",
                blocks_to_swap,
                self.num_blocks - 1,
                self.blocks_to_swap,
            )

        if self.blocks_to_swap > _EXTREME_SWAP_THRESHOLD:
            logger.warning(
                "blocks_to_swap=%d is very high (>%d). Expect significant "
                "training speed reduction.",
                self.blocks_to_swap,
                _EXTREME_SWAP_THRESHOLD,
            )

        # Number of blocks that permanently stay on GPU
        self._num_resident: int = self.num_blocks - self.blocks_to_swap

        # Create secondary CUDA stream for async weight transfers
        self._stream: torch.cuda.Stream = torch.cuda.Stream(device=device)

        # Allocate pinned memory buffers for swappable blocks (one-time cost)
        for i in range(self._num_resident, self.num_blocks):
            self._create_pinned_buffers(blocks[i], i)

        # Apply initial device placement
        self._prepare_block_devices(blocks)

        # Register hooks on all blocks
        self._register_hooks(blocks)

        logger.info(
            "Block swap: %d/%d blocks offloaded, %d resident on GPU",
            self.blocks_to_swap,
            self.num_blocks,
            self._num_resident,
        )

    def _create_pinned_buffers(self, block: nn.Module, block_idx: int) -> None:
        """Allocate pinned CPU tensors for each Linear weight in a block.

        Pinned memory allows direct DMA access by the GPU, making
        CPU<->GPU transfers 2-3x faster than pageable memory. These
        buffers are allocated once and reused across all training steps.
        """
        buffers = []
        for module, _ in _get_swappable_weights(block):
            buf = torch.empty_like(module.weight, device="cpu")
            try:
                buf = buf.pin_memory(device=self.device)
            except Exception:
                # pin_memory may fail on some systems (e.g., CPU-only).
                # Fall back to unpinned memory.
                pass
            buffers.append(buf)
        self._pinned_buffers[block_idx] = buffers

    def _prepare_block_devices(self, blocks: nn.ModuleList) -> None:
        """Set initial device placement for all blocks.

        Resident blocks (indices 0..num_resident-1) stay on GPU.
        Swappable blocks (indices num_resident..num_blocks-1) have their
        Linear weights moved to CPU. Non-Linear params (LayerNorm, etc.)
        stay on GPU so they're always ready.

        This method can be called after state_dict swap (expert switch)
        to re-prepare block placement.
        """
        for i in range(self._num_resident, self.num_blocks):
            self._move_block_to_cpu(blocks[i], i)

    def prepare_block_devices(self, blocks: nn.ModuleList) -> None:
        """Re-prepare initial device placement (public API).

        Called after state_dict swap during expert switching to ensure
        pinned buffers and device placement are consistent with the
        new weights.
        """
        if self.blocks_to_swap <= 0:
            return
        self._prepare_block_devices(blocks)

    def _move_block_to_cpu(self, block: nn.Module, block_idx: int) -> None:
        """Move only the SWAPPABLE (base_layer + plain) Linear weights of a block to CPU buffers.

        LoRA sub-Linears are excluded (kept GPU-resident) — see :func:`_get_swappable_weights`.
        Non-Linear params (LayerNorm, biases of non-Linear modules) stay on their current device.
        """
        linear_weights = _get_swappable_weights(block)
        pinned_bufs = self._pinned_buffers.get(block_idx, [])
        # WR-04: strict=True turns the prose "pinned-buffer count STABLE" invariant into an ENFORCED
        # one. If _get_swappable_weights ever returns a different count/order than at
        # _create_pinned_buffers time (module-graph mutation after construction — the late-load
        # scenario this fix caters to), zip would SILENTLY truncate/mispair and corrupt weights; in an
        # offloader with a corruption history (BSWP-TRAINING-01) that must fail loud, not silently.
        for (module, _), pinned_buf in zip(linear_weights, pinned_bufs, strict=True):
            pinned_buf.copy_(module.weight.data)
            module.weight.data = pinned_buf

    def _move_block_to_gpu(self, blocks: nn.ModuleList, block_idx: int) -> None:
        """Move a block's tracked Linear weights to GPU using the async stream.

        Uses the FULL tracked set (:func:`_get_linear_weights`) so this doubles as the re-home
        for late-loaded lora sub-Linears (the frozen adapter is stacked AFTER construction and may
        land on CPU): it pulls base_layer AND every adapter's lora_A/lora_B onto the GPU. Weights
        already on-device return the same tensor (``.to`` is a no-op), so the resident lora is
        homed once and then costs nothing. Uses non_blocking=True within the secondary CUDA stream.
        """
        block = blocks[block_idx]
        with torch.cuda.stream(self._stream):
            for module, _ in _get_linear_weights(block):
                module.weight.data = module.weight.data.to(
                    self.device, non_blocking=True
                )

    def _submit_move_to_cpu(self, blocks: nn.ModuleList, block_idx: int) -> None:
        """Async move of a block's SWAPPABLE (base_layer + plain) Linear weights to CPU buffers."""
        block = blocks[block_idx]
        linear_weights = _get_swappable_weights(block)
        pinned_bufs = self._pinned_buffers.get(block_idx, [])
        with torch.cuda.stream(self._stream):
            # WR-04: strict=True enforces the pinned-buffer alignment invariant (see _move_block_to_cpu)
            # — a count/order drift fails loud rather than silently mispairing weights into buffers.
            for (module, _), pinned_buf in zip(linear_weights, pinned_bufs, strict=True):
                pinned_buf.copy_(module.weight.data, non_blocking=True)
                module.weight.data = pinned_buf

    def _wait_for_stream(self) -> None:
        """Synchronize: make main stream wait for secondary stream transfers."""
        torch.cuda.current_stream().wait_stream(self._stream)

    def _register_hooks(self, blocks: nn.ModuleList) -> None:
        """Register forward_pre_hook, forward_hook, and backward hooks.

        Forward pre-hooks are registered on ALL blocks to synchronize
        the async stream before each block executes. Forward hooks and
        backward hooks are registered for the swapping logic.
        """
        for i in range(self.num_blocks):
            # Forward pre-hook: synchronize stream before block executes
            h = blocks[i].register_forward_pre_hook(
                self._make_forward_pre_hook(blocks, i)
            )
            self._handles.append(h)

            # Forward hook: after block forward, swap weights if needed
            fwd_hook = self._make_forward_hook(blocks, i)
            if fwd_hook is not None:
                h = blocks[i].register_forward_hook(fwd_hook)
                self._handles.append(h)

            # Backward hook: reverse-order weight swapping
            bwd_hook = self._make_backward_hook(blocks, i)
            if bwd_hook is not None:
                h = blocks[i].register_full_backward_hook(bwd_hook)
                self._handles.append(h)

    def _make_forward_pre_hook(
        self, blocks: nn.ModuleList, block_idx: int
    ) -> Any:
        """Create hook that ensures block weights are on GPU before forward.

        First waits for any pending async transfer to complete, then
        checks if the block's weights are actually on GPU. If not
        (e.g., during gradient-checkpoint backward recomputation where
        traversal order differs from forward), moves them synchronously.

        This makes block swap compatible with gradient checkpointing,
        which re-runs forward passes in reverse during backward.
        """

        def hook(module: nn.Module, args: Any) -> None:
            self._wait_for_stream()
            # For swappable blocks, ensure weights are on GPU.
            # During backward recomputation (gradient checkpointing),
            # blocks may be accessed out-of-order and the forward-pass
            # prefetch logic won't have moved them.
            if block_idx >= self._num_resident:
                # Scan ALL tracked weights (base_layer + every active adapter's lora_A/lora_B),
                # not just weights[0]: a late-loaded frozen lora sub-linear left on CPU is exactly
                # the BSWP-TRAINING-01 crash the weights[0]-only check missed. _move_block_to_gpu
                # re-homes base + resident lora together.
                if _block_needs_gpu(module, self.device):
                    self._move_block_to_gpu(blocks, block_idx)
                    self._wait_for_stream()

        return hook

    def _make_forward_hook(
        self, blocks: nn.ModuleList, block_idx: int
    ) -> Any | None:
        """Create hook that swaps weights after block forward completes.

        For blocks in the swappable range (after the resident blocks):
        - Move this block's Linear weights to CPU (pinned buffers)
        - Prefetch the next block that needs to come from CPU

        Returns None for resident blocks (no swap needed).
        """
        # During forward, blocks execute 0..N-1 in order.
        # After block[i] finishes forward:
        #   - If i >= num_resident, block[i] was brought from CPU, send it back
        #   - Prefetch block[i + 1] if it's a swappable block
        if block_idx < self._num_resident:
            # Resident block. But we may need to prefetch the first
            # swappable block when the last resident block finishes.
            if block_idx == self._num_resident - 1 and self.blocks_to_swap > 0:
                next_idx = self._num_resident

                def hook(module: nn.Module, args: Any, output: Any) -> None:
                    self._move_block_to_gpu(blocks, next_idx)

                return hook
            return None

        # Swappable block: offload to CPU, prefetch next if exists
        next_swap_idx = block_idx + 1 if block_idx + 1 < self.num_blocks else None

        def hook(module: nn.Module, args: Any, output: Any) -> None:
            self._submit_move_to_cpu(blocks, block_idx)
            if next_swap_idx is not None:
                self._move_block_to_gpu(blocks, next_swap_idx)

        return hook

    def _make_backward_hook(
        self, blocks: nn.ModuleList, block_idx: int
    ) -> Any | None:
        """Create hook for backward-pass weight swapping.

        Backward traverses blocks N-1..0 (reverse order). After backward
        through block[i], if block[i] was a swappable block, offload it
        to CPU and prefetch block[i-1] if it needs to come from CPU.

        Returns None for blocks that don't need backward swapping.
        """
        if block_idx < self._num_resident:
            return None

        # After backward through this swappable block, send it to CPU
        # and prefetch the previous block if it's also swappable
        prev_idx = block_idx - 1 if block_idx - 1 >= self._num_resident else None

        def hook(
            module: nn.Module, grad_input: Any, grad_output: Any
        ) -> None:
            self._submit_move_to_cpu(blocks, block_idx)
            if prev_idx is not None:
                self._move_block_to_gpu(blocks, prev_idx)

        return hook

    def remove_hooks(self) -> None:
        """Remove all registered hook handles.

        Call this when block swapping is no longer needed (e.g., at the
        end of training or before re-registering hooks after a disk
        reload expert switch).
        """
        for h in self._handles:
            h.remove()
        self._handles.clear()
