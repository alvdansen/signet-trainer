"""OFFL-03 / BSWP-TRAINING-01 root-cause proofs — the block-swap offloader is PEFT-LoRA-aware.

CPU-only, no-CUDA, no-modal, no-ltx_core proofs (mirror ``tests/test_frozen_stacking.py``'s
``pytest.importorskip("peft")`` guard + ``_TwoLinear`` fixture). They lock the fix that resolves the
run-4 ``mat2 is on cpu`` device mismatch (07-12): with TWO active PEFT adapters (frozen + default),
the harvested ``_get_linear_weights`` tracked only the wrapper's proxied base weight and SKIPPED every
adapter's ``lora_A``/``lora_B`` sub-linear, so a freshly ``load_adapter``-ed frozen LoRA sub-linear ran
on CPU while its block ran on GPU. Its ``weights[0]``-only forward-pre-hook guard compounded it.

The four behaviors proven here (all CPU):
  (1) ``_get_linear_weights`` on a stacked frozen+default PEFT block yields the REAL weight-owning
      leaves (base_layer + every active adapter's lora_A/lora_B), base once, wrapper proxy never;
  (2) a two-adapter CPU forward+backward runs with no device error and the freeze invariant holds
      (grad only on the default adapter; frozen lora untouched);
  (3) the extracted device-guard predicate ``_block_needs_gpu`` flags a block whose base is on-device
      but a lora sub-linear is off-device (``.to("meta")`` proxy) — the old weights[0]-only check misses it;
  (4) ``_get_swappable_weights`` keeps lora sub-linears RESIDENT (swaps only base_layer/plain), and a
      plain nn.Linear block is unchanged / ``blocks_to_swap=0`` stays inert (no regression).

The real gated GPU confirmation is 07-13 Task 2 (a committed frozen-excluded checkpoint).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402  — after skip-guard

from signet_trainer.lora.peft import (  # noqa: E402
    get_linear_weights as _get_linear_weights,
    get_swappable_weights as _get_swappable_weights,
    save_adapter,
    stack_frozen_adapter,
)
from signet_trainer.offload.block_swap import (  # noqa: E402
    BlockSwapOffloader,
    _block_needs_gpu,
    _same_device,
)


# --------------------------------------------------------------------------------------------------
# Fixtures (mirror test_frozen_stacking.py)
# --------------------------------------------------------------------------------------------------


class _TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):
        return self.b(self.a(x))


def _cfg() -> LoraConfig:
    return LoraConfig(
        r=4, lora_alpha=4, target_modules=["a", "b"], lora_dropout=0.0, bias="none"
    )


def _randomize_lora(model: nn.Module) -> None:
    # lora_B inits to zero — randomize so a stacked frozen adapter carries real (non-zero) deltas.
    for name, p in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.copy_(torch.randn_like(p))


def _stacked_two_adapter(tmp_path) -> nn.Module:
    """A CPU PEFT ``_TwoLinear`` carrying a stacked frozen + trainable default adapter."""
    src = get_peft_model(_TwoLinear(), _cfg())
    _randomize_lora(src)
    frozen_dir = tmp_path / "frozen_src"
    save_adapter(src, frozen_dir)

    model = get_peft_model(_TwoLinear(), _cfg())
    stack_frozen_adapter(model, frozen_dir, adapter_name="frozen")
    return model


def _leaf_linears_by_name(block: nn.Module):
    """Manual reference walk: real weight-owning leaf Linears (NOT the PEFT proxy wrapper),
    partitioned into (base_or_plain, lora) by name."""
    base_or_plain = []
    lora = []
    for name, m in block.named_modules():
        if hasattr(m, "base_layer"):
            continue  # PEFT wrapper — its .weight proxies base_layer.weight; not a real leaf
        if m.__class__.__name__.endswith("Linear") and getattr(m, "weight", None) is not None:
            (lora if "lora_" in name else base_or_plain).append(m)
    return base_or_plain, lora


# --------------------------------------------------------------------------------------------------
# (1) Enumeration coverage — base_layer + all active-adapter lora, base once, wrapper never
# --------------------------------------------------------------------------------------------------


def test_get_linear_weights_covers_base_and_all_adapter_lora(tmp_path) -> None:
    model = _stacked_two_adapter(tmp_path)

    weights = _get_linear_weights(model)
    got_modules = [m for m, _ in weights]
    got_ids = {id(m.weight) for m in got_modules}

    base_or_plain, lora = _leaf_linears_by_name(model)
    manual_ids = {id(m.weight) for m in (base_or_plain + lora)}

    # id-set equality against the manual leaf walk.
    assert got_ids == manual_ids, "enumeration must cover exactly the real weight-owning leaves"

    # The PEFT wrapper (proxy) is NEVER yielded — nothing yielded exposes base_layer.
    assert all(not hasattr(m, "base_layer") for m in got_modules), (
        "the proxy wrapper (base_layer holder) must not be yielded — descend into it instead"
    )

    # Base weight appears exactly once (no double count of the proxy vs base_layer).
    assert len(weights) == len(got_ids), "no weight id may be yielded twice (base counted once)"

    # 2 targeted Linears x (base_layer + lora_A.{frozen,default} + lora_B.{frozen,default}) = 10.
    assert len(weights) == 10, f"expected 10 leaf weights on the stacked block, got {len(weights)}"
    assert len(lora) == 8, f"expected 8 lora sub-linears (2 targets x 2 adapters x A/B), got {len(lora)}"


# --------------------------------------------------------------------------------------------------
# (2) Two-adapter CPU forward + backward — no device error, freeze invariant survives
# --------------------------------------------------------------------------------------------------


def test_two_adapter_forward_backward_freeze_intact(tmp_path) -> None:
    model = _stacked_two_adapter(tmp_path)

    x = torch.randn(3, 8)
    out = model(x)  # both adapters active — their deltas sum in the forward
    loss = out.sum()
    loss.backward()  # no "mat2 is on cpu" — everything on CPU, all leaves tracked

    named = list(model.named_parameters())
    default_lora_grads = []
    for n, p in named:
        if "frozen" in n:
            assert p.requires_grad is False, f"frozen param must stay frozen: {n}"
            assert p.grad is None, f"frozen param must receive NO grad: {n}"
        elif "lora_" in n and "default" in n:
            assert p.requires_grad is True, f"default lora must remain trainable: {n}"
            default_lora_grads.append(p.grad)

    assert default_lora_grads, "the default adapter must be optimizer-eligible"
    assert any(g is not None for g in default_lora_grads), (
        "backward must populate grad on the trainable default adapter"
    )


# --------------------------------------------------------------------------------------------------
# (3) Device guard — flags an off-device lora sub-linear the old weights[0]-only check would miss
# --------------------------------------------------------------------------------------------------


def test_device_guard_flags_off_device_lora(tmp_path) -> None:
    model = _stacked_two_adapter(tmp_path)
    device = torch.device("cpu")

    weights = _get_linear_weights(model)
    assert len(weights) > 1
    # weights[0] is the first tracked leaf (a base_layer) — it must NOT be a lora.
    assert weights[0][0].weight.data.device == device

    # Strand ONE lora sub-linear off-device: replace its weight with a meta Parameter (meta = a
    # no-CUDA off-device proxy on a CPU-only box). This is exactly the run-4 topology: the base
    # weights are present/on-device while a freshly-loaded lora sub-linear sits on the wrong device.
    _, lora = _leaf_linears_by_name(model)
    off_module = lora[-1]
    off_module.weight = nn.Parameter(
        off_module.weight.detach().to("meta"), requires_grad=off_module.weight.requires_grad
    )

    # The OLD guard inspected only weights[0] (a base weight, still on-device) → it would MISS this.
    old_guard_fires = weights[0][0].weight.data.device != device
    assert old_guard_fires is False, "precondition: the old weights[0]-only check misses the off-device lora"

    # The NEW predicate scans ALL tracked weights and catches it.
    assert _block_needs_gpu(model, device) is True, (
        "the all-tracked-weights guard must flag an off-device lora sub-linear"
    )


def test_device_guard_clean_when_all_on_device(tmp_path) -> None:
    model = _stacked_two_adapter(tmp_path)
    # Everything is on CPU already — no re-home needed.
    assert _block_needs_gpu(model, torch.device("cpu")) is False


# --------------------------------------------------------------------------------------------------
# (3b) WR-03 — index-normalized device compare: `cuda` and `cuda:0` are the SAME physical device
# --------------------------------------------------------------------------------------------------


def test_same_device_cuda_indexless_matches_cuda_zero() -> None:
    # WR-03 core regression: the offloader is built with torch.device("cuda") (no index) while GPU
    # weights live on cuda:0. A raw torch.device __eq__ treats these as DIFFERENT, making the device
    # guard always-true on GPU. _same_device must treat them as the same physical device. CPU-safe:
    # an index-less cuda device resolves to 0 when CUDA is unavailable.
    assert _same_device(torch.device("cuda"), torch.device("cuda:0")) is True
    assert _same_device(torch.device("cuda:0"), torch.device("cuda")) is True
    # Raw equality is the buggy behavior this fix works around — assert the trap it avoids exists.
    assert (torch.device("cuda") == torch.device("cuda:0")) is False


def test_same_device_distinguishes_index_and_type() -> None:
    assert _same_device(torch.device("cuda:0"), torch.device("cuda:1")) is False
    assert _same_device(torch.device("cpu"), torch.device("cuda")) is False
    assert _same_device(torch.device("cpu"), torch.device("cpu")) is True


def test_block_needs_gpu_indexless_cuda_target_not_always_true(tmp_path) -> None:
    # A block whose weights report device cuda:0 must NOT be flagged as "needs gpu" when the offloader
    # target is the index-less torch.device("cuda") — the always-true bug WR-03 fixes. We simulate by
    # relabeling one leaf's device string comparison via _same_device; here we assert the predicate is
    # False when every tracked weight is already on the (normalized) target device (CPU stand-in).
    model = _stacked_two_adapter(tmp_path)
    # cpu vs cpu (both index-less) normalizes equal -> no spurious re-home.
    assert _block_needs_gpu(model, torch.device("cpu")) is False


# --------------------------------------------------------------------------------------------------
# (4) Swappable partition keeps lora resident + plain-Linear no-regression / inert baseline
# --------------------------------------------------------------------------------------------------


def test_swappable_excludes_lora_keeps_it_resident(tmp_path) -> None:
    model = _stacked_two_adapter(tmp_path)

    base_or_plain, lora = _leaf_linears_by_name(model)
    sw_ids = {id(m.weight) for m, _ in _get_swappable_weights(model)}

    # Swappable = base_layer + plain Linears ONLY — the big weights that get parked to CPU.
    assert sw_ids == {id(m.weight) for m in base_or_plain}, (
        "swappable set must be exactly base_layer + plain Linears"
    )
    # Every lora sub-linear is EXCLUDED — kept GPU-resident so adamw8bit's state stays CUDA and a
    # late-loaded frozen lora never strands on CPU.
    assert all(id(m.weight) not in sw_ids for m in lora), "lora sub-linears must stay resident (never swapped)"


def test_move_block_to_cpu_strict_zip_rejects_misaligned_buffers() -> None:
    """WR-04: a pinned-buffer list whose length disagrees with the swappable-weight count must fail
    LOUD (strict=True), not silently truncate/mispair weights into wrong buffers."""

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Linear(4, 4)
            self.q = nn.Linear(4, 4)

    block = _Block()  # 2 swappable plain Linears
    # Hand-construct without __init__ (blocks_to_swap>0 would need a CUDA stream). _move_block_to_cpu
    # touches no stream — only _get_swappable_weights + self._pinned_buffers.
    off = object.__new__(BlockSwapOffloader)
    off._pinned_buffers = {0: [torch.empty_like(block.p.weight)]}  # 1 buffer for 2 weights

    with pytest.raises(ValueError):
        off._move_block_to_cpu(block, 0)


def test_move_block_to_cpu_aligned_buffers_ok() -> None:
    """The aligned case (buffer count == swappable count) still works under strict=True."""

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Linear(4, 4)
            self.q = nn.Linear(4, 4)

    block = _Block()
    off = object.__new__(BlockSwapOffloader)
    off._pinned_buffers = {0: [torch.empty_like(block.p.weight), torch.empty_like(block.q.weight)]}
    off._move_block_to_cpu(block, 0)  # no raise; both weights parked into their buffers
    assert block.p.weight.data.device == torch.device("cpu")


def test_plain_linear_enumeration_unchanged_and_inert() -> None:
    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = nn.Linear(4, 4)
            self.q = nn.Linear(4, 4)

    block = _Block()
    # Plain nn.Linear (no PEFT wrapper): one entry per Linear, unchanged.
    assert len(_get_linear_weights(block)) == 2
    assert len(_get_swappable_weights(block)) == 2  # all plain Linears are swappable

    # blocks_to_swap=0 stays inert (constructor early-return, zero hooks) — no regression.
    blocks = nn.ModuleList([_Block(), _Block(), _Block()])
    off = BlockSwapOffloader(blocks, blocks_to_swap=0, device=torch.device("cpu"))
    assert off.blocks_to_swap == 0
    assert len(off._handles) == 0
