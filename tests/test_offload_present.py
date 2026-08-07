"""OFFL-01 structural gate — the BlockSwapOffloader is harvested, importable, and INERT
at the baseline-first ``blocks_to_swap=0`` setting.

This is a CPU, no-GPU, no-modal, no-ltx_core structural test (mirrors
``tests/test_loop_purity.py``'s confinement style). It proves THREE things WITHOUT
ever activating the unvalidated swap path:

  (a) the offloader was harvested into the repo and imports cleanly on CPU (OFFL-01);
  (b) the source records its harvest provenance AND the BSWP-TRAINING-01 root-cause + the
      07-13 PEFT-LoRA-aware / OFFL-03 fix (reconciled 07-13: the blanket "UNVALIDATED" label
      was replaced when ``_get_linear_weights`` was made PEFT-aware — the banner is now a
      scoped freeze-mechanics warning, not a blanket "do not trust"); and
  (c) ``BlockSwapOffloader(<tiny ModuleList>, blocks_to_swap=0, device=cpu)`` registers
      ZERO hooks — the constructor early-return (block_swap.py) makes it inert, which is
      the baseline-first seam (D-OFF-1) the first metered run relies on.

It does NOT validate the swap behavior end-to-end — the gated GPU confirmation is OFFL-03
(07-13 Task 2), deliberately out of scope here.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn

# The harvested module SOURCE (used for the label text-scan; also imported normally below).
_BLOCK_SWAP_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "signet_trainer"
    / "offload"
    / "block_swap.py"
)


def test_offloader_importable_on_cpu() -> None:
    """(a) The offloader imports cleanly on CPU (OFFL-01 harvested, no GPU/modal/ltx needed)."""
    from signet_trainer.offload.block_swap import BlockSwapOffloader  # noqa: F401

    assert BlockSwapOffloader is not None


def test_offloader_labeled_unvalidated() -> None:
    """(b) The source records BSWP-TRAINING-01 + the 07-13 PEFT-LoRA-aware / OFFL-03 fix.

    Reconciled 07-13: the banner no longer carries a blanket "UNVALIDATED" label — the offloader
    was made PEFT-LoRA-aware, so it now records the root cause AND the scoped freeze-mechanics
    re-validation. The BSWP-TRAINING-01 citation stays (root cause); the OFFL-03 / PEFT-aware
    markers replace the old blanket label.
    """
    src = _BLOCK_SWAP_SRC.read_text(encoding="utf-8")
    assert "BSWP-TRAINING-01" in src, (
        "block_swap.py must cite the BSWP-TRAINING-01 training bug (now root-caused + fixed)"
    )
    assert "PEFT-LoRA-aware" in src, (
        "block_swap.py must record the 07-13 PEFT-LoRA-aware fix"
    )
    assert re.search(r"OFFL-03", src), (
        "block_swap.py must cite the OFFL-03 re-validation scope"
    )


def test_baseline_blocks_to_swap_zero_is_inert() -> None:
    """(c) blocks_to_swap=0 → constructor early-returns → ZERO hooks (the baseline seam)."""
    from signet_trainer.offload.block_swap import BlockSwapOffloader

    # A tiny stand-in transformer block stack: each block has one Linear weight.
    blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)])

    offloader = BlockSwapOffloader(blocks, blocks_to_swap=0, device=torch.device("cpu"))

    # Inert: the early-return path clamps to 0 and registers no hooks.
    assert offloader.blocks_to_swap == 0
    assert offloader.num_blocks == 3
    assert len(offloader._handles) == 0, (
        "baseline blocks_to_swap=0 must register ZERO hooks (inert, zero behavior change)"
    )
