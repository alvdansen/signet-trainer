"""The parallel watcher must verify renders against the FAMILY's real samples path.

Why this file exists (the blocker, 2026-08-06): ``scripts/watch_parallel_inference.py`` hardcoded the
LTX answer — it listed ``{output_dir}/samples`` and matched UTC wall-clock stamps. ``h3_sample``
writes ``{output_dir}/samples_h3/<render key>/``. Run against an H3 config the watcher would dispatch
a metered render, never see it land, mark it FAILED and re-dispatch on the next poll. Because
``append_spend`` fires BEFORE the dispatch and has no refund path, that books the full per-render
estimate every 240 s against renders that may have SUCCEEDED — the phantom-spend shape
(``KNOWLEDGE.md`` ``watcher`` ``phantom-spend``), and the thing that would trip ``session_cap_usd``
and halt a healthy campaign.

Both families are pinned, in the same file, deliberately: the LTX path must stay byte-identically
behaved, and a future "just make it H3" edit has to fail HERE rather than on a metered Volume.

Zero GPU / zero Modal / zero spend — pure string functions and a source scan.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from signet_trainer.inference.render_key import h3_render_key
from signet_trainer.inference.samples_layout import (
    expected_h3_render_key,
    landed_render_ids,
    samples_root,
    samples_subdir,
)

REPO = Path(__file__).resolve().parents[1]
WATCHER = REPO / "scripts" / "watch_parallel_inference.py"


# ---- the layout table itself ----------------------------------------------------------------


def test_ltx_samples_subdir_is_unchanged():
    # The historical LTX answer. If this ever moves, every prior campaign's watcher reseed breaks.
    assert samples_subdir("ltx") == "samples"


def test_h3_samples_subdir_is_samples_h3():
    # Transcribed from modal/fns.py: CHECKPOINTS_DIR / output_dir / "samples_h3" / h3_render_key(...)
    assert samples_subdir("h3") == "samples_h3"


def test_samples_root_composes_per_family():
    assert samples_root("outputs/embe_r1", "ltx") == "outputs/embe_r1/samples"
    assert samples_root("outputs/h3_embe_r1", "h3") == "outputs/h3_embe_r1/samples_h3"


def test_unknown_family_raises_never_defaults_to_ltx():
    # Fail-loud is the money-safe direction: a silent fallback to "samples" is precisely how this
    # hole would be re-opened for the next family, and the symptom is spend, not an exception.
    with pytest.raises(ValueError, match="unknown model family"):
        samples_subdir("wan")


# ---- landed-render detection, per family ------------------------------------------------------

_LTX_LISTING = (
    "outputs/embe_r1/samples/20260805T154357Z\n"
    "outputs/embe_r1/samples/20260805T184725Z\n"
)
_H3_LISTING = (
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f22_A-029\n"
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f22_B-029\n"
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f124_A-029\n"
)


def test_ltx_landed_ids_are_the_utc_stamps():
    assert landed_render_ids(_LTX_LISTING, "ltx") == ["20260805T154357Z", "20260805T184725Z"]


def test_h3_landed_ids_are_identity_keys_not_stamps():
    ids = landed_render_ids(_H3_LISTING, "h3")
    assert len(ids) == 3
    assert "checkpoint-step-00250-loss-0.1016_s42_f22_A-029" in ids
    # An H3 listing carries NO wall-clock stamp; the old regex would have returned [] here, which is
    # exactly the "render never landed" mis-read that caused the re-dispatch loop.
    assert not re.search(r"\d{8}T\d{6}Z", _H3_LISTING)


def test_h3_configs_differing_only_in_reference_are_distinct_renders():
    # carry-forward of aaaee62. The five sample configs share output_dir, seed and prompt set, so
    # they write identical clip FILENAMES — only the render-dir identity separates them. If the
    # watcher keyed on anything coarser, the A+029 render would be accepted as proof that the B+029
    # render landed, and the grid would be labelled for a reference condition it does not contain.
    a = expected_h3_render_key(checkpoint="checkpoint-step-00250-loss-0.1016", seed=42,
                               frame_count=22, subject_ids=["A", "029"])
    b = expected_h3_render_key(checkpoint="checkpoint-step-00250-loss-0.1016", seed=42,
                               frame_count=22, subject_ids=["B", "029"])
    long_a = expected_h3_render_key(checkpoint="checkpoint-step-00250-loss-0.1016", seed=42,
                                    frame_count=124, subject_ids=["A", "029"])
    assert a != b, "reference condition must be part of the render identity"
    assert a != long_a, "frame count must be part of the render identity"
    ids = landed_render_ids(_H3_LISTING, "h3")
    assert a in ids and b in ids and long_a in ids


def test_expected_key_delegates_to_the_renders_own_function():
    # Never a re-implementation: the watcher's expectation and the render's directory name must be
    # produced by the SAME function or they drift silently.
    kwargs = dict(checkpoint="checkpoint-step-03000-loss-0.1933", seed=42, frame_count=56,
                  subject_ids=["C", "018"])
    assert expected_h3_render_key(**kwargs) == h3_render_key(**kwargs)


def test_h3_reference_order_is_not_collapsed():
    # D-10-REFORDER: a reordered reference set is a genuinely different request (it fixes the
    # <Picture i> labels AND advances the shared rotary clock), so it must not collapse to one dir.
    fwd = expected_h3_render_key(checkpoint="c", seed=42, frame_count=22, subject_ids=["A", "029"])
    rev = expected_h3_render_key(checkpoint="c", seed=42, frame_count=22, subject_ids=["029", "A"])
    assert fwd != rev


# ---- the watcher actually USES it (source scan — the watcher drives metered renders) ----------


def _watcher_src() -> str:
    return WATCHER.read_text(encoding="utf-8")


def test_watcher_imports_the_family_aware_layout():
    src = _watcher_src()
    assert "from signet_trainer.inference.samples_layout import" in src
    assert "samples_root(" in src


def test_watcher_has_no_hardcoded_samples_path_in_the_landed_check():
    # The regression itself: the render-root used by the landed-check must come from SAMPLES_ROOT,
    # never from an f-string pinning `/samples`. (refresh_grid keeps its LTX-only legacy staging
    # branch, which is guarded by `if FAMILY == "h3": return` above it.)
    src = _watcher_src()
    assert "SAMPLES_ROOT = samples_root(OUTPUT_DIR, FAMILY)" in src
    assert 'FAMILY = _cfg.model.family' in src
    body = src.split("def committed_render_stamps")[1].split("def latest_checkpoint_name")[0]
    assert "SAMPLES_ROOT" in body
    assert '{OUTPUT_DIR}/samples"' not in body, "the landed-check must not hardcode the LTX subdir"


def test_watcher_render_landed_is_family_aware():
    src = _watcher_src()
    assert "def render_landed(" in src
    body = src.split("def render_landed")[1].split("\ndef ")[0]
    assert "expected_h3_render_key(" in body
    assert 'FAMILY != "h3"' in body, "the LTX branch must keep its any-new-stamp behaviour"


def test_watcher_grid_refresh_uses_gridwatch_driver_for_h3():
    # finetune-gridwatch is the ONLY sanctioned grid builder; the H3 branch delegates to the
    # first-party incremental driver rather than re-staging by hand.
    body = _watcher_src().split("def refresh_grid")[1].split("\ndef ")[0]
    assert "_h3_grid_serve.py" in body
    assert '"--rows", "step"' in body, "the checkpoint step must be the row axis during training"


def test_watcher_cadence_is_configurable_and_not_every_checkpoint():
    src = _watcher_src()
    assert "RENDER_EVERY" in src
    assert "s % RENDER_EVERY == 0 or s >= MAX_STEPS" in src, (
        "renders fire on cadence boundaries + the final step — never once per checkpoint"
    )
