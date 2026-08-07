"""REF-LOAD back-port wiring proof for ``fns.py::sample`` (Plan 08-02, D-8-REFLOAD, Pitfall 5).

The single_frame and multi_frame sample branches used to render on the RAW BASE transformer, so any
convergence/quality verdict off their grids validated the base model, NOT the trained LoRA. Plan
08-02 back-ports the ic_lora branch's trained-adapter load (``CheckpointManager.find_latest()`` ->
``load_lora_onto_transformer``) into both branches, with a loud fall-back-to-base warning when no
adapter exists. This test LOCKS that back-port and guards the ic_lora branch against regression.

Two zero-GPU layers (mirroring ``test_multi_frame_sample_wiring.py``):

  * MODULE-TOP IMPORT — ``import signet_trainer.modal.fns`` must succeed off-Modal (the module top
    imports ONLY from ``modal.app``; every heavy ltx/torch import in ``sample`` is function-local,
    Anti-Pattern 6). No GPU, no spend.

  * TEXT SCAN — the single_frame and multi_frame branch SEGMENTS each reference ``CheckpointManager``,
    ``find_latest``, and ``load_lora_onto_transformer``, and print BOTH the "loaded trained adapter"
    line and the loud "no trained adapter" / "BASE transformer" fall-back (the observable Pitfall-5
    warning-sign). Assertions are attributed to the CORRECT branch by segmenting the source on the
    branch marker comments, not merely "anywhere in the file". A scoped-change guard asserts the
    ic_lora branch's adapter-load block is still present (unchanged).

The source is segmented on the RAW marker comments FIRST; comments + docstrings are stripped WITHIN
each segment before the identifier/string assertions so prose mentions don't false-positive.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"

# Branch marker comments (present in the RAW source; used to attribute assertions to a branch).
_SINGLE_FRAME_MARKER = "SINGLE-FRAME reference-control branch"
_MULTI_FRAME_MARKER = "MULTI-FRAME reference-control branch"
_IC_LORA_MARKER = "IC-LoRA official-adapter"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _segment(raw: str, start_marker: str, end_marker: str | None) -> str:
    """Return the RAW source from ``start_marker`` up to ``end_marker`` (or EOF), comments stripped.

    Segmenting on the raw marker comments FIRST (before stripping) keeps each assertion attributed to
    the correct branch; stripping AFTER means the identifier/string assertions ignore prose.
    """
    start = raw.index(start_marker)
    end = raw.index(end_marker) if end_marker is not None else len(raw)
    assert start < end, f"marker order broken: {start_marker!r} must precede {end_marker!r}"
    return _strip_comments_and_docstrings(raw[start:end])


def test_fns_imports_off_modal() -> None:
    """Module-top import must succeed off-Modal — every heavy import in sample() is function-local."""
    import signet_trainer.modal.fns as fns  # noqa: PLC0415

    assert hasattr(fns, "sample"), "modal.fns must expose the sample stage function"


def test_single_frame_branch_loads_trained_adapter() -> None:
    """Plan 08-02: the single_frame branch resolves + loads the latest trained adapter before render."""
    raw = _FNS.read_text(encoding="utf-8")
    seg = _segment(raw, _SINGLE_FRAME_MARKER, _MULTI_FRAME_MARKER)

    assert "CheckpointManager" in seg, (
        "single_frame branch must resolve the checkpoint via CheckpointManager (no hand-rolled glob)"
    )
    assert "find_latest" in seg, (
        "single_frame branch must use CheckpointManager.find_latest() — the single source of 'latest "
        "step' (per-STEP dirs, not a flat file)"
    )
    assert "load_lora_onto_transformer" in seg, (
        "single_frame branch must PEFT-wrap the transformer via load_lora_onto_transformer"
    )
    assert "loaded trained adapter" in seg, (
        "single_frame branch must announce it loaded the trained adapter"
    )


def test_single_frame_branch_has_loud_base_fallback() -> None:
    """The single_frame branch keeps the honest fall-back-to-base warning (Pitfall-5 warning-sign)."""
    raw = _FNS.read_text(encoding="utf-8")
    seg = _segment(raw, _SINGLE_FRAME_MARKER, _MULTI_FRAME_MARKER)

    assert "no trained adapter" in seg, (
        "single_frame branch must warn when no trained adapter exists (wiring mode)"
    )
    assert "on the BASE transformer" in seg, (
        "single_frame branch must say it is rendering on the BASE transformer in the fall-back"
    )
    # The single_frame two_stage fail-fast guard must survive the edit (VRAM landmine, T-08-02-VRAM).
    assert "two_stage_upscale is not supported for single_frame" in seg, (
        "single_frame branch must preserve the two_stage fail-fast guard"
    )


def test_multi_frame_branch_loads_trained_adapter() -> None:
    """Plan 08-02: the multi_frame branch resolves + loads the latest trained adapter before render."""
    raw = _FNS.read_text(encoding="utf-8")
    seg = _segment(raw, _MULTI_FRAME_MARKER, _IC_LORA_MARKER)

    assert "CheckpointManager" in seg, (
        "multi_frame branch must resolve the checkpoint via CheckpointManager (no hand-rolled glob)"
    )
    assert "find_latest" in seg, (
        "multi_frame branch must use CheckpointManager.find_latest() — the single source of latest step"
    )
    assert "load_lora_onto_transformer" in seg, (
        "multi_frame branch must PEFT-wrap the transformer via load_lora_onto_transformer"
    )
    assert "loaded trained adapter" in seg, (
        "multi_frame branch must announce it loaded the trained adapter"
    )


def test_multi_frame_branch_has_loud_base_fallback() -> None:
    """The multi_frame branch keeps the honest fall-back-to-base warning (Pitfall-5 warning-sign)."""
    raw = _FNS.read_text(encoding="utf-8")
    seg = _segment(raw, _MULTI_FRAME_MARKER, _IC_LORA_MARKER)

    assert "no trained adapter" in seg, (
        "multi_frame branch must warn when no trained adapter exists (wiring mode)"
    )
    assert "on the BASE transformer" in seg, (
        "multi_frame branch must say it is rendering on the BASE transformer in the fall-back"
    )


def test_ic_lora_adapter_load_block_still_present() -> None:
    """Scoped-change guard: the ic_lora branch's adapter-load block is untouched (no regression)."""
    raw = _FNS.read_text(encoding="utf-8")
    seg = _segment(raw, _IC_LORA_MARKER, None)

    assert "find_latest" in seg, "ic_lora branch must still resolve the checkpoint via find_latest()"
    assert "load_lora_onto_transformer" in seg, (
        "ic_lora branch must still PEFT-wrap via load_lora_onto_transformer"
    )
    assert "reskin_model" in seg, (
        "ic_lora branch must still bind its trained adapter to reskin_model (block unchanged)"
    )
