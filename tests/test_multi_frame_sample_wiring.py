"""Structural + column-plan wiring proof for ``fns.py::sample``'s multi_frame branch (Plan 06-08).

Two zero-GPU layers (mirroring ``test_preprocess_wiring.py`` + ``test_multi_frame_strategy.py``):

  * MODULE-TOP IMPORT — ``import signet_trainer.modal.fns`` must succeed off-Modal (the module top
    imports ONLY from ``modal.app``; every heavy ltx/torch import in ``sample`` is function-local,
    Anti-Pattern 6). This proves the branch wiring never pulls the ltx stack onto Windows/CI.

  * TEXT SCAN — the multi_frame branch dispatches on ``config.conditioning.mode == "multi_frame"``,
    renders each column via ``run_multi_condition_sampler`` at ``config.validation.seed``, writes
    ``write_multi_frame_gallery``, and commits (commit-or-vanish). The existing single_frame /
    base-vs-LoRA branches are asserted still present (untouched).

  * COLUMN PLAN — the PURE ``plan_multi_frame_columns`` (imported directly) yields the ordered SC#3
    columns (N=2 / N=3 / strength-lo / strength-hi / no-reference control) off the example config,
    with the no-reference control carrying an EMPTY item list. The GPU render itself is proven by
    the gated run (Plan 06-09), so this covers the structural + column-plan surface ONLY.

Comments + docstrings are stripped before the text scan so prose mentions don't false-positive.
"""

from __future__ import annotations

import re
from pathlib import Path

from signet_trainer.config.load import load_config
from signet_trainer.inference.multi_condition import plan_multi_frame_columns

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"
_MULTI_FRAME_CFG = _REPO_ROOT / "configs" / "ltx23_multi_frame.example.yaml"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def test_fns_imports_off_modal() -> None:
    """Module-top import must succeed off-Modal — every heavy import in sample() is function-local."""
    import signet_trainer.modal.fns as fns  # noqa: PLC0415

    assert hasattr(fns, "sample"), "modal.fns must expose the sample stage function"
    assert hasattr(fns, "preprocess"), "modal.fns must expose the preprocess stage function"


def test_sample_has_multi_frame_branch() -> None:
    """Plan 06-08: sample() branches on mode == 'multi_frame' and wires the multi-condition sampler."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))

    assert re.search(
        r'config\.conditioning\.mode\s*==\s*["\']multi_frame["\']', code
    ), "sample() must branch on config.conditioning.mode == 'multi_frame'"
    assert "run_multi_condition_sampler" in code, (
        "the multi_frame branch must render via run_multi_condition_sampler (Plan 06-06)"
    )
    assert "write_multi_frame_gallery" in code, (
        "the multi_frame branch must write the write_multi_frame_gallery grid (Plan 06-03)"
    )
    assert "plan_multi_frame_columns" in code, (
        "the multi_frame branch must derive columns via plan_multi_frame_columns (Plan 06-06)"
    )
    # Rendered at the config seed (42), never a hardcoded literal.
    assert re.search(r"seed\s*=\s*config\.validation\.seed", code), (
        "sample() must render at config.validation.seed (42) — no hardcoded seed"
    )
    # Cold-path probe entry for the VideoConditionByLatentIndex surface (fail loudly before spend).
    assert "ltx_core.conditioning.types.latent_cond" in code, (
        "the cold-path import probe must include ltx_core.conditioning.types.latent_cond (T-06-SC)"
    )
    # WR-01: the branch must thread the planner's CONFIG-DRIVEN column label into the gallery row
    # (strength_lo_label/strength_hi_label) — the grid never hardcodes the sweep strengths.
    assert re.search(r'row\[f?["\']\{?stem\}?_label["\']\]\s*=\s*column\.label', code), (
        "the multi_frame branch must set row[f'{stem}_label'] = column.label (config-driven "
        "sweep labels, WR-01)"
    )


def test_multi_frame_branch_commits_and_is_scoped() -> None:
    """The branch commits (commit-or-vanish) and the single_frame / base-vs-LoRA paths are untouched."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))

    assert "checkpoints_vol.commit()" in code, "commit-or-vanish on the checkpoints Volume"
    # Sibling branches must still exist (this plan only ADDS a branch — it does not rewrite them).
    assert re.search(r'config\.conditioning\.mode\s*==\s*["\']single_frame["\']', code), (
        "the Phase-5 single_frame branch must remain present (scoped change)"
    )
    assert "write_comparison_gallery" in code, (
        "the base-vs-LoRA comparison path must remain present (scoped change)"
    )


def test_train_fails_fast_on_sample_only_conditioning_items() -> None:
    """train() must reject a multi_frame config carrying conditioning_items (WR-04).

    Training is self-conditioning (D-6-CONDSOURCE 'self'): MultiFrameStrategy samples its own
    keyframe positions/strengths, so keyframe items in a TRAIN config would be silently ignored —
    the silently-ignored-config-block class the schema field-split doctrine forbids. The guard
    must raise before any model load / GPU spend. Since gap-dryrun-ltx-0 the predicate lives in
    ONE CPU-pure home (config/mode_gate.py) shared with the free gates — the container calls the
    shared validator instead of carrying its own copy.
    """
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))

    assert re.search(
        r'validate_mode_config\(\s*config\s*,\s*["\']train["\']\s*\)',
        code,
    ), "train() must call the shared mode gate (WR-04 fail-fast, one home: config/mode_gate.py)"
    # The guard must fire BEFORE the component load (no GPU spend on a doomed config).
    guard_idx = code.index('validate_mode_config(config, "train")')
    load_idx = code.index("load_ltxv_components(")
    assert guard_idx < load_idx, "the WR-04 guard must precede load_ltxv_components in train()"

    # The predicate + rationale live in the ONE shared home, not in a private fns.py copy.
    gate_src = (_FNS.parents[1] / "config" / "mode_gate.py").read_text(encoding="utf-8")
    assert re.search(
        r'cfg\.conditioning\.mode\s*==\s*["\']multi_frame["\']', gate_src
    ), "mode_gate.py must carry the multi_frame predicate (one home)"
    assert "sample-only" in gate_src, (
        "the WR-04 guard must explain that conditioning_items are sample-only in Phase 6"
    )


def test_column_plan_yields_ordered_sc3_columns() -> None:
    """PURE column plan: N=2 / N=3 / strength-lo / strength-hi / no-reference-control off the example."""
    config = load_config(_MULTI_FRAME_CFG)
    columns = plan_multi_frame_columns(config)

    # Five ordered columns, keyed exactly to the write_multi_frame_gallery row-dict keys.
    row_keys = [c.row_key for c in columns]
    assert row_keys == [
        "n2_mp4",
        "n3_mp4",
        "strength_lo_mp4",
        "strength_hi_mp4",
        "no_ref_mp4",
    ], f"unexpected column order/keys: {row_keys}"

    by_key = {c.row_key: c for c in columns}
    assert len(by_key["n2_mp4"].items) == 2, "N=2 column must carry the first 2 keyframes"
    assert len(by_key["n3_mp4"].items) == 3, "N=3 column must carry the first 3 keyframes"

    # The no-reference control is the empty-item base render (the SC#3 divergence baseline).
    assert by_key["no_ref_mp4"].items == [], "no-reference control must have an EMPTY item list"

    # The strength sweep endpoints flow from config.conditioning.conditioning_strength_range.
    lo, hi = config.conditioning.conditioning_strength_range
    mid = len(by_key["strength_lo_mp4"].items) // 2
    assert by_key["strength_lo_mp4"].items[mid][1] == lo, "strength-lo must force the mid item to lo"
    assert by_key["strength_hi_mp4"].items[mid][1] == hi, "strength-hi must force the mid item to hi"
