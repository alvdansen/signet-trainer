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

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from signet_trainer.inference.render_key import h3_render_key
from signet_trainer.inference.samples_layout import (
    STEP_KEYED_LTX_MODES,
    committed_clip_names,
    expected_h3_render_key,
    landed_render_ids,
    layout_mode,
    samples_root,
    samples_subdir,
)

REPO = Path(__file__).resolve().parents[1]
WATCHER = REPO / "scripts" / "watch_parallel_inference.py"
SAMPLE_CONFIG = REPO / "configs" / "sample.yaml"


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


# ---- LTX is MODE-aware too, not just family-aware (issue #45 PR-4) ----------------------------


def test_ltx_default_mode_is_unchanged_without_a_mode_argument():
    # Every call site that predates PR-4 (and every non-LTX family, which has no mode axis) omits
    # `mode` entirely — it must keep resolving to the historical "samples" answer.
    assert samples_subdir("ltx") == "samples"
    assert samples_subdir("ltx", None) == "samples"
    assert samples_subdir("ltx", "none") == "samples"
    assert samples_root("outputs/embe_r1", "ltx") == "outputs/embe_r1/samples"


def test_ltx_non_default_modes_resolve_their_own_root():
    # Transcribed from modal/fns.py's per-mode sample() branches — the whole point of PR-4: family
    # alone picks "samples" for ALL of these, which is the wrong directory for every one of them.
    assert samples_subdir("ltx", "inpaint") == "samples_inpaint"
    assert samples_subdir("ltx", "audio_to_video") == "samples_a2v"
    assert samples_subdir("ltx", "single_frame") == "samples_single_frame"
    assert samples_subdir("ltx", "multi_frame") == "samples_multi_frame"
    assert samples_subdir("ltx", "ic_lora") == "samples_ic_lora"
    assert samples_subdir("ltx", "ic_lora_baseline") == "samples_ic_lora_baseline"
    assert samples_root("outputs/x", "ltx", "inpaint") == "outputs/x/samples_inpaint"


def test_mode_is_ignored_for_non_ltx_families():
    # H3/qwen_edit have no mode axis; passing one (e.g. a stray default) must never redirect them.
    assert samples_subdir("h3", "inpaint") == "samples_h3"
    assert samples_subdir("qwen_edit", "inpaint") == "samples_qwen_edit"


def test_layout_mode_splits_ic_lora_on_two_stage_upscale():
    # modal/fns.py's own dispatch: conditioning.mode == "ic_lora" + two_stage_upscale picks the
    # SEPARATELY-GATED ic_lora_baseline branch, not the single-stage ic_lora one. layout_mode must
    # reproduce this exactly or the watcher and the render key on different identities.
    assert layout_mode(conditioning_mode="ic_lora", two_stage_upscale=False) == "ic_lora"
    assert layout_mode(conditioning_mode="ic_lora", two_stage_upscale=True) == "ic_lora_baseline"
    # every other mode is untouched by two_stage_upscale (single_frame/multi_frame ban it outright;
    # mode "none" doesn't gate on it here).
    assert layout_mode(conditioning_mode="single_frame", two_stage_upscale=True) == "single_frame"
    assert layout_mode(conditioning_mode="none", two_stage_upscale=False) == "none"


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


# ---- STEP-KEYED LTX modes: inpaint / audio_to_video (issue #45 PR-4) ---------------------------
#
# modal/fns.py's inpaint/a2v branches write a STABLE `<stem>` dir (named after the held-out test
# clip) that every dispatched checkpoint adds ONE `step_<N>.mp4` file into — never a fresh
# UTC-stamped dir per render. A stem-only id would make step 600 and step 1200 the identical
# render the moment the FIRST one committed (the verifier's finding against the original fix);
# `landed_render_ids` must return the STEP-BEARING id `"<stem>/step_<N>"` instead.

_LTX_INPAINT_LISTING = (
    "outputs/embe_inpaint_r1/samples_inpaint/clip_a\n"
    "outputs/embe_inpaint_r1/samples_inpaint/clip_b\n"
    "outputs/embe_inpaint_r1/samples_inpaint/clip_a/input.mp4\n"
    "outputs/embe_inpaint_r1/samples_inpaint/clip_a/step_0.mp4\n"
    "outputs/embe_inpaint_r1/samples_inpaint/clip_a/step_600.mp4\n"
    "outputs/embe_inpaint_r1/samples_inpaint/clip_b/step_600.mp4\n"
)


def test_ltx_step_keyed_modes_land_as_stem_dirs_not_stamps():
    ids = landed_render_ids(_LTX_INPAINT_LISTING, "ltx", "inpaint")
    assert ids == ["clip_a/step_0", "clip_a/step_600", "clip_b/step_600"]
    # STEM-anchored, never wall-clock-anchored — the family/mode contrast this test is named for.
    assert not any(re.search(r"\d{8}T\d{6}Z", i) for i in ids)
    # the staged raw input clip (never a render) must never be mistaken for a landed step.
    assert not any("input" in i for i in ids)


def test_ltx_step_keyed_deep_listing_resolves_to_the_stem():
    # `modal volume ls` has no recursive flag (confirmed against the real CLI) — SAMPLES_ROOT's own
    # one-level listing shows only the bare STEM dir, carrying no render identity by itself.
    stems_only = "outputs/x/samples_a2v/interview_clip\n"
    assert landed_render_ids(stems_only, "ltx", "audio_to_video") == []
    # once the caller folds in that stem's OWN listing (committed_render_stamps' per-stem probe),
    # the deeper path resolves DOWN to stem + step regardless of how many segments came before it.
    deep = stems_only + "outputs/x/samples_a2v/interview_clip/step_1200.mp4\n"
    assert landed_render_ids(deep, "ltx", "audio_to_video") == ["interview_clip/step_1200"]


def test_ltx_step_keyed_ids_distinguish_cadence_boundaries():
    # The exact bug a stem-only id let through: two renders at DIFFERENT steps into the SAME stem
    # dirs must never collapse to one identity, or a step-1200 render would read as already landed
    # the moment step-600 committed.
    at_step_600 = (
        "outputs/x/samples_inpaint/clip_a/step_600.mp4\n"
        "outputs/x/samples_inpaint/clip_b/step_600.mp4\n"
    )
    at_step_1200 = (
        "outputs/x/samples_inpaint/clip_a/step_1200.mp4\n"
        "outputs/x/samples_inpaint/clip_b/step_1200.mp4\n"
    )
    ids_600 = set(landed_render_ids(at_step_600, "ltx", "inpaint"))
    ids_1200 = set(landed_render_ids(at_step_1200, "ltx", "inpaint"))
    assert ids_600, "fixture must actually produce ids for this to be a real test"
    assert ids_600.isdisjoint(ids_1200), (
        "renders at different steps into the identical stem set must never share an id"
    )


def test_step_keyed_modes_are_the_documented_subset():
    # Pins the registered subset itself — a mode added to SAMPLES_SUBDIR_BY_LTX_MODE without also
    # being added here (if it is truly step-keyed) would silently fall through to the stamp regex
    # and return [] forever, the exact "render never landed" mis-read this module exists to close.
    assert STEP_KEYED_LTX_MODES == frozenset({"inpaint", "audio_to_video"})


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
    # issue #45 PR-4: SAMPLES_ROOT is now MODE-aware too, not just family-aware (family alone picks
    # the wrong root for 5 of 6 non-default LTX conditioning.mode branches) — RENDER_MODE is the
    # third argument, resolved via layout_mode() rather than hardcoded.
    assert "SAMPLES_ROOT = samples_root(OUTPUT_DIR, FAMILY, RENDER_MODE)" in src
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

# ---- render_landed's stale-id fix (issue #45 PR-4) — behavioral, not a source scan -------------
#
# PR-2's verifier already flagged pure source-scans as test theater for a decision this exact
# shaped (a semantic inversion of one comparison surviving every test in
# tests/test_watcher_hardening.py). render_landed's stale-id branch is the same shape: it decides
# `bool(set(ids) - _pending_baseline_ids)`, and a source scan cannot tell that apart from the old,
# buggy `bool(ids)` if someone "fixed" the diff back out while leaving comments untouched. These
# tests import the REAL watcher module (mirroring tests/test_watcher_pending_clock.py's
# `_load_watcher()` pattern) and drive `render_landed` directly, with only the Volume-shelling seam
# (`committed_render_stamps`) monkeypatched — zero Modal, zero spend.


def _load_watcher_module():
    old_argv = sys.argv
    sys.argv = ["watch_parallel_inference.py", str(SAMPLE_CONFIG)]
    try:
        spec = importlib.util.spec_from_file_location(
            "watch_parallel_inference_under_test_samples_path", WATCHER
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = old_argv


def test_stale_id_present_before_dispatch_does_not_land_the_render(monkeypatch):
    # configs/sample.yaml is LTX, conditioning.mode "none" -> the STAMPED (non-step-keyed) branch.
    mod = _load_watcher_module()
    assert mod.FAMILY == "ltx" and mod.RENDER_MODE not in mod.STEP_KEYED_LTX_MODES
    stale_stamp = "20260805T100000Z"
    # This stamp already existed on the Volume BEFORE the pending render was ever dispatched — the
    # baseline main() would have captured at dispatch time.
    monkeypatch.setattr(mod, "_pending_baseline_ids", frozenset({stale_stamp}))
    monkeypatch.setattr(mod, "committed_render_stamps", lambda: [stale_stamp])
    assert mod.render_landed(50, None) is False, (
        "a stamp that already existed before this render was dispatched must never count as proof "
        "THIS render landed — bool(ids) alone (the pre-PR-4 behaviour) would have returned True here"
    )


def test_new_id_beyond_the_baseline_does_land_the_render(monkeypatch):
    mod = _load_watcher_module()
    stale_stamp = "20260805T100000Z"
    fresh_stamp = "20260805T110000Z"
    monkeypatch.setattr(mod, "_pending_baseline_ids", frozenset({stale_stamp}))
    monkeypatch.setattr(mod, "committed_render_stamps", lambda: [stale_stamp, fresh_stamp])
    assert mod.render_landed(50, None) is True, (
        "a stamp that appeared AFTER the dispatch-time baseline is exactly what 'landed' must mean "
        "on the stamped LTX path"
    )


def test_baseline_only_gates_the_stamped_path_h3_stays_identity_keyed():
    # The h3 arm's landed criterion (identity-dir check with the captured checkpoint) must be
    # UNCHANGED by this fix — it never reads _pending_baseline_ids at all.
    body = _watcher_src().split("def render_landed")[1].split("\ndef ")[0]
    assert "_pending_baseline_ids" not in body.split('if checkpoint is None:')[1], (
        "the h3 branch must not read the LTX stale-id baseline"
    )
    assert "want in ids" in body, "h3 stays a pure identity-membership check"
