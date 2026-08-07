"""OFFL-02 structural gate — the suspend seam (``remove_hooks``) exists and is callable.

The OFFL-02 requirement is that a suspend seam exists around the (deferred) sampler
call-site so block-swap hooks can be removed and later re-registered. In Phase 3 this is
proven STRUCTURALLY (the seam is present and callable), not behaviorally — the offloader
runs inert at ``blocks_to_swap=0`` and the sampler call-site lands later.

This CPU, no-GPU test asserts:
  (a) ``BlockSwapOffloader.remove_hooks`` exists and is callable; and
  (b) calling it on a baseline (inert) offloader is a safe no-op that leaves the handle
      list empty — the documented suspend-seam contract.

The re-register counterpart (``_register_hooks``) is also asserted present so the
suspend → resume round-trip seam is structurally complete for OFFL-03.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from signet_trainer.offload.block_swap import BlockSwapOffloader


def test_remove_hooks_exists_and_callable() -> None:
    """(a) The OFFL-02 suspend seam: remove_hooks is present and callable."""
    assert hasattr(BlockSwapOffloader, "remove_hooks")
    assert callable(BlockSwapOffloader.remove_hooks)
    # The re-register counterpart must also exist for the suspend → resume round-trip.
    assert hasattr(BlockSwapOffloader, "_register_hooks")
    assert callable(BlockSwapOffloader._register_hooks)


def test_remove_hooks_is_safe_noop_on_baseline() -> None:
    """(b) remove_hooks() on the inert baseline is a safe no-op (handles stay empty)."""
    blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    offloader = BlockSwapOffloader(blocks, blocks_to_swap=0, device=torch.device("cpu"))

    # Inert baseline has no hooks; the suspend seam must not raise.
    offloader.remove_hooks()
    assert offloader._handles == []
