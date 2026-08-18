"""The parallel watcher must verify renders against the FAMILY's real samples path.

Why this file exists (the blocker, 2026-08-06): ``scripts/watch_parallel_inference.py`` hardcoded the
LTX answer — it listed ``{output_dir}/samples`` and matched UTC wall-clock stamps. ``h3_sample``
writes ``{output_dir}/samples_h3/<render key>/``. Run against an H3 config the watcher would dispatch
a metered render, never see it land, mark it FAILED and re-dispatch on the next poll. At the time
the watcher's OWN ``append_spend`` fired BEFORE the dispatch with no refund path, booking the full
per-render estimate every 240 s against renders that may have SUCCEEDED — the phantom-spend shape
(``KNOWLEDGE.md`` ``watcher`` ``phantom-spend``), and the thing that would trip ``session_cap_usd``
and halt a healthy campaign. (Issue #37 findings 1/2 later moved the booking itself into the
entrypoint gate, once per successful ``.spawn()`` — see ``test_watcher_no_longer_double_books_the_
ledger_entry`` below; the family-aware landed-check this file exists for is unaffected.)

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
    committed_clip_names,
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
#: #22 finding 5 widened h3_render_key with width/height/num_inference_steps; every listing/key
#: literal below carries them so the fixtures stay valid render-dir names under the current regex.
_GEOM = dict(width=1344, height=768, num_inference_steps=25)
_H3_LISTING = (
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f22_w1344_h768_n25_A-029\n"
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f22_w1344_h768_n25_B-029\n"
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f124_w1344_h768_n25_A-029\n"
)


def test_ltx_landed_ids_are_the_utc_stamps():
    assert landed_render_ids(_LTX_LISTING, "ltx") == ["20260805T154357Z", "20260805T184725Z"]


def test_h3_landed_ids_are_identity_keys_not_stamps():
    ids = landed_render_ids(_H3_LISTING, "h3")
    assert len(ids) == 3
    assert "checkpoint-step-00250-loss-0.1016_s42_f22_w1344_h768_n25_A-029" in ids
    # An H3 listing carries NO wall-clock stamp; the old regex would have returned [] here, which is
    # exactly the "render never landed" mis-read that caused the re-dispatch loop.
    assert not re.search(r"\d{8}T\d{6}Z", _H3_LISTING)


def test_h3_configs_differing_only_in_reference_are_distinct_renders():
    # carry-forward of aaaee62. The five sample configs share output_dir, seed and prompt set, so
    # they write identical clip FILENAMES — only the render-dir identity separates them. If the
    # watcher keyed on anything coarser, the A+029 render would be accepted as proof that the B+029
    # render landed, and the grid would be labelled for a reference condition it does not contain.
    a = expected_h3_render_key(checkpoint="checkpoint-step-00250-loss-0.1016", seed=42,
                               frame_count=22, **_GEOM, subject_ids=["A", "029"])
    b = expected_h3_render_key(checkpoint="checkpoint-step-00250-loss-0.1016", seed=42,
                               frame_count=22, **_GEOM, subject_ids=["B", "029"])
    long_a = expected_h3_render_key(checkpoint="checkpoint-step-00250-loss-0.1016", seed=42,
                                    frame_count=124, **_GEOM, subject_ids=["A", "029"])
    assert a != b, "reference condition must be part of the render identity"
    assert a != long_a, "frame count must be part of the render identity"
    ids = landed_render_ids(_H3_LISTING, "h3")
    assert a in ids and b in ids and long_a in ids


def test_expected_key_delegates_to_the_renders_own_function():
    # Never a re-implementation: the watcher's expectation and the render's directory name must be
    # produced by the SAME function or they drift silently.
    kwargs = dict(checkpoint="checkpoint-step-03000-loss-0.1933", seed=42, frame_count=56,
                  **_GEOM, subject_ids=["C", "018"])
    assert expected_h3_render_key(**kwargs) == h3_render_key(**kwargs)


def test_h3_reference_order_is_not_collapsed():
    # D-10-REFORDER: a reordered reference set is a genuinely different request (it fixes the
    # <Picture i> labels AND advances the shared rotary clock), so it must not collapse to one dir.
    fwd = expected_h3_render_key(checkpoint="c", seed=42, frame_count=22, **_GEOM,
                                 subject_ids=["A", "029"])
    rev = expected_h3_render_key(checkpoint="c", seed=42, frame_count=22, **_GEOM,
                                 subject_ids=["029", "A"])
    assert fwd != rev


# ---- fine-grained clip progress (issue #45 PR-1 must-fix #1) ---------------------------------

_RENDER_DIR_BASE_LISTING_TWO_CLIPS = (
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f22_A-029/base/prompt_one_s42.mp4\n"
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f22_A-029/base/prompt_two_s42.mp4\n"
)


def test_committed_clip_names_reads_mp4_filenames_only():
    # The progress probe cares about CLIPS, not the render's own index.html/delta.json — those two
    # never grow mid-render at the cadence a clip does, and delta.json's own absence is what "not
    # yet landed" MEANS; treating it as progress would defeat the point of the finer signal.
    listing = _RENDER_DIR_BASE_LISTING_TWO_CLIPS + (
        "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f22_A-029/base/index.html\n"
    )
    names = committed_clip_names(listing)
    assert names == ["prompt_one_s42.mp4", "prompt_two_s42.mp4"]
    assert "index.html" not in names


def test_committed_clip_names_grows_as_clips_land():
    # The exact defect the stall clock needs a signal for: render_landed's coarse identity check
    # cannot distinguish "one clip in" from "eleven clips in" — this probe can, and it must actually
    # CHANGE as more clips commit, or resetting pending_since on it would be a no-op.
    one_clip = committed_clip_names(
        "outputs/x/samples_h3/key/base/prompt_one_s42.mp4\n"
    )
    two_clips = committed_clip_names(_RENDER_DIR_BASE_LISTING_TWO_CLIPS)
    assert len(two_clips) > len(one_clip)


def test_committed_clip_names_empty_listing_is_empty():
    assert committed_clip_names("") == []
    assert committed_clip_names("\n") == []


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


def test_watcher_render_landed_requires_explicit_checkpoint():
    # issue #45 PR-1 must-fix #2: the checkpoint identity used to verify a PENDING render must be the
    # value captured at dispatch time, not re-derived here every poll (a re-derivation could drift
    # onto a checkpoint newer than the one actually dispatched and never find the real render). Kill
    # the checkpoint=None match-anything path: render_landed takes checkpoint as a parameter and
    # refuses None rather than silently resolving one internally.
    src = _watcher_src()
    assert "def render_landed(step: int, checkpoint: str | None) -> bool:" in src
    body = src.split("def render_landed")[1].split("\ndef ")[0]
    assert "ckpt = latest_checkpoint_name()" not in body, (
        "render_landed must not re-derive the checkpoint itself — it must be handed the dispatch-"
        "time value by the caller, not resolve a fresh (possibly drifted) one internally"
    )
    assert "checkpoint is None" in body
    assert "raise ValueError" in body


def test_watcher_landed_check_carries_the_widened_geometry_axes():
    # #22 finding 5: the landed-check must pass the SAME axes h3_sample now keys on, or a
    # resolution/step-count probe would never be recognised as landed (it would key on the old,
    # narrower identity and poll forever against a directory the render never writes).
    body = _watcher_src().split("def render_landed")[1].split("\ndef ")[0]
    for axis in ("width=", "height=", "num_inference_steps="):
        assert axis in body, f"render_landed's expected_h3_render_key call is missing {axis!r}"


def test_watcher_progress_probe_exists_and_is_finer_than_landed():
    # issue #45 PR-1 must-fix #1: a distinct progress probe, reading one level deeper than
    # render_landed's coarse identity check, is what lets the stall clock refresh on evidence of
    # life instead of firing on every multi-hour H3 render.
    src = _watcher_src()
    assert "def render_progress_artifacts(" in src
    body = src.split("def render_progress_artifacts")[1].split("\ndef ")[0]
    assert "committed_clip_names(" in body
    assert '"base"' in body and '"lora"' in body


def test_watcher_progress_probe_carries_the_widened_geometry_axes():
    # Restack reconciliation (fix/h3-sample-base-dedup onto fix/watcher-stall-clock): the progress
    # probe keys the LORA half on expected_h3_render_key exactly like render_landed does, so it must
    # carry the same widened geometry axes — a probe still keyed on the pre-#22-finding-5 signature
    # would silently watch the wrong (stale-geometry) directory for new clips.
    body = _watcher_src().split("def render_progress_artifacts")[1].split("\ndef ")[0]
    for axis in ("width=", "height=", "num_inference_steps="):
        assert axis in body, f"render_progress_artifacts's expected_h3_render_key call is missing {axis!r}"


def test_watcher_progress_probe_composes_the_base_path_outside_the_render_dir():
    # Sibling-collision reconciliation (verify_c.json MAJOR finding): the base column now lives at
    # SAMPLES_ROOT/base/<base key>/, a SIBLING of the checkpoint-keyed render dir (#12) — the probe
    # must read that shared location for "base", never descend into f"{render_dir}/base" (which the
    # base-dedup relocation left permanently empty). The "lora" half stays checkpoint-scoped.
    body = _watcher_src().split("def render_progress_artifacts")[1].split("\ndef ")[0]
    assert 'f"{render_dir}/base"' not in body, (
        "the base column no longer lives under the checkpoint-scoped render_dir — reading it there "
        "silently sees zero progress for the whole base phase of a fresh-geometry render"
    )
    assert "expected_h3_base_render_key(" in body
    assert 'f"{SAMPLES_ROOT}/base/{base_key}"' in body, (
        "the base descent must be composed from SAMPLES_ROOT (a sibling of the render dir), not the "
        "checkpoint-scoped render_dir"
    )
    assert 'f"{render_dir}/lora"' in body, "the lora column stays checkpoint-scoped"


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


# ---- issue #37 findings 1/2: the entrypoint gate books the ledger now, the watcher must not ----


def test_watcher_no_longer_double_books_the_ledger_entry():
    """The watcher must not append its own ledger entry — the entrypoint subprocess it dispatches
    via `modal run ... --approve` now books that entry itself (issue #37 finding 1). A watcher
    that ALSO appended would double-count every parallel-venue render against the cumulative cap.
    """
    src = _watcher_src()
    assert "append_spend as _append_spend" not in src, (
        "the watcher must not import append_spend at all — booking is the entrypoint's job now"
    )
    assert "def append_spend(" not in src, "the append_spend(step) wrapper must be gone, not just unused"
    main_body = src.split("\ndef main()")[1]
    assert "append_spend(" not in main_body, (
        "main() must not call append_spend anywhere — double-booking against the entrypoint's own "
        "post-dispatch append"
    )


def test_watcher_still_runs_its_own_pre_dispatch_cap_check():
    """The watcher's OWN session_cap_check before dispatch stays — it is a cheap local pre-check
    that avoids shelling out to a dispatch the entrypoint's own cap gate would refuse anyway, and
    is independent of (not a substitute for, and not redundant with) the entrypoint's booking."""
    src = _watcher_src()
    assert "from signet_trainer.modal.session_cap import" in src
    assert "read_ledger" in src and "session_cap_check" in src
    main_body = src.split("\ndef main()")[1]
    assert "session_cap_check(" in main_body
