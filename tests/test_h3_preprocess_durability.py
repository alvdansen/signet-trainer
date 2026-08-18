"""Plan 10-14 — h3_preprocess durability: atomic writes, per-sample resume, commit-before-judging.

The defect this pins (audit finding, PR-3): the H3 pre-encode wrote cache files with a bare
``torch.save``, re-encoded every row unconditionally, and committed the dataset Volume exactly once
— at the very bottom, AFTER the raise-bearing guards. Any raise past PHASE A therefore discarded the
entire Qwen3-VL-32B pass (the documented five-containers-died loss), a re-dispatch started at row 0,
and (once commits become periodic / background) a container killed inside ``torch.save`` could
publish a truncated ``.pt`` at the canonical name for ``PrecomputedDataset`` to pair.

Three fixes, each tested here:

  1. ``write_h3_precomputed`` stages every payload and ``replace``s it into place
     (``prep/h3_encode._atomic_save``, the ``data/mask_encode.py`` idiom) — behavioural tests;
  2. both phase loops skip a sample whose FOUR sources are already on disk
     (``h3_precomputed_complete``) and PHASE B commits every ``h3.preprocess_commit_every``
     samples — behavioural tests for the predicate, source scans for the wiring;
  3. the loud-failure guards run AFTER ``dataset_vol.commit()`` — they refuse the SUCCESS REPORT,
     never the committed encode — source scan.

The ``modal/fns.py`` half follows the house wiring-scan convention (``test_h3_preprocess_wiring``):
scanned as TEXT with comments/docstrings stripped, never imported (importing it builds the Modal app
graph). CPU-only, zero GPU, zero Modal spend.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"#.*", "", src)


def _preprocess_body() -> str:
    """The stripped source of ``h3_preprocess`` alone, so assertions cannot match other stages."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    match = re.search(r"^def h3_preprocess\(", code, re.M)
    assert match, "h3_preprocess() not found in modal/fns.py"
    tail = re.search(r"^(?:def |class |@)", code[match.end() :], re.M)
    end = match.end() + tail.start() if tail else len(code)
    return code[match.start() : end]


# ==================================================================================================
# Fix 1 — the write is staged + renamed, never a bare torch.save at the canonical name
# ==================================================================================================


def test_a_write_that_dies_mid_save_publishes_nothing_at_the_canonical_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A container killed inside torch.save must not leave a truncated .pt where the resume guard
    (or PrecomputedDataset) would count it as a finished sample."""
    torch = pytest.importorskip("torch")
    from signet_trainer.prep import h3_encode

    payload = torch.zeros(3, 4)

    def dying_save(data: object, destination: object) -> None:
        # Simulate the kill: bytes land at whatever path the writer handed us, then the process
        # "dies". Pre-fix the writer handed the CANONICAL path, so the junk was published.
        Path(destination).write_bytes(b"truncated")
        raise RuntimeError("container killed mid-save")

    monkeypatch.setattr(h3_encode.torch, "save", dying_save)
    canonical = tmp_path / h3_encode.H3_CONDITIONS_DIR / "a" / "b.pt"
    with pytest.raises(RuntimeError, match="container killed"):
        h3_encode.write_h3_precomputed(tmp_path, "a/b.pt", text=payload)
    assert not canonical.exists(), (
        "the interrupted write published a file at the canonical name — write_h3_precomputed must "
        "stage (tmp + replace) so a kill mid-torch.save leaves only a staging file behind"
    )


def test_a_completed_write_lands_whole_with_no_staging_leftovers(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from signet_trainer.prep import h3_encode

    written = h3_encode.write_h3_precomputed(tmp_path, "a/b.pt", text=torch.zeros(3, 4))
    destination = written[h3_encode.H3_CONDITIONS_DIR]
    assert destination.is_file()
    siblings = sorted(p.name for p in destination.parent.iterdir())
    assert siblings == ["b.pt"], f"staging leftovers next to the canonical file: {siblings}"
    # Overwrite still works through the staged path (the atomic idiom must not break re-encodes).
    h3_encode.write_h3_precomputed(tmp_path, "a/b.pt", text=torch.ones(3, 4))
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)
    assert bool(reloaded.eq(1).all()), "the second (overwrite) write did not replace the payload"


# ==================================================================================================
# Fix 2 — the resume predicate: complete means ALL FOUR sources, nothing less
# ==================================================================================================


def test_h3_precomputed_complete_requires_the_whole_quartet(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from signet_trainer.prep import h3_encode

    rel = "scene/clip.pt"
    assert not h3_encode.h3_precomputed_complete(tmp_path, rel)

    # PHASE A alone (the died-in-PHASE-B shape) must NOT count as complete.
    h3_encode.write_h3_precomputed(tmp_path, rel, text=torch.zeros(2, 3))
    assert not h3_encode.h3_precomputed_complete(tmp_path, rel)

    # Fill the remaining three sources one at a time; only the full quartet flips the predicate.
    for dir_name in (
        h3_encode.H3_VIDEO_LATENTS_DIR,
        h3_encode.H3_REFERENCE_LATENTS_DIR,
        h3_encode.H3_AUDIO_LATENTS_DIR,
    ):
        assert not h3_encode.h3_precomputed_complete(tmp_path, rel)
        target = tmp_path / dir_name / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"stub": True}, target)
    assert h3_encode.h3_precomputed_complete(tmp_path, rel)

    # The media-suffix form names the same sample (the writer's own .pt normalization).
    assert h3_encode.h3_precomputed_complete(tmp_path, "scene/clip.mp4")


def test_the_predicate_and_the_writer_share_one_source_dir_tuple() -> None:
    """A fifth source added to the writer but not the predicate would resume-skip partial samples."""
    pytest.importorskip("torch")
    from signet_trainer.prep import h3_encode

    assert h3_encode.H3_PRECOMPUTED_SOURCE_DIRS == (
        h3_encode.H3_VIDEO_LATENTS_DIR,
        h3_encode.H3_CONDITIONS_DIR,
        h3_encode.H3_REFERENCE_LATENTS_DIR,
        h3_encode.H3_AUDIO_LATENTS_DIR,
    )
    with pytest.raises(ValueError, match="relative"):
        # Path.cwd() is absolute on every OS; a bare "/abs" string is NOT absolute on Windows.
        h3_encode.h3_precomputed_complete("root", Path.cwd() / "clip.pt")


# ==================================================================================================
# Fix 2/3 wiring — scans of h3_preprocess (never imported: importing fns.py builds the app graph)
# ==================================================================================================


def test_both_phase_loops_carry_the_resume_guard() -> None:
    body = _preprocess_body()
    assert "h3_precomputed_complete" in body, (
        "h3_preprocess must build its resume set through prep/h3_encode.h3_precomputed_complete — "
        "without it a re-dispatch re-pays the whole Qwen3-VL-32B pass from row 0"
    )
    assert "preprocess_overwrite" in body, (
        "the skip must be gated on the config's preprocess_overwrite (config-first), never "
        "unconditional — a recipe change needs a forced re-encode"
    )
    assert body.count("in resumed") >= 2, (
        "BOTH phase loops must consult the shared resume set: a sample skipped in PHASE A has no "
        "per_sample entry, so PHASE B skipping it too is what keeps the two loops consistent"
    )


def test_phase_b_commits_periodically_at_the_configured_interval() -> None:
    body = _preprocess_body()
    assert "preprocess_commit_every" in body, (
        "the commit cadence must be the threaded config value (h3.preprocess_commit_every), "
        "never a literal"
    )
    assert body.count("dataset_vol.commit()") >= 2, (
        "h3_preprocess needs BOTH the periodic in-loop commit (bounds the loss window) and the "
        "final commit (persists the tail shorter than one window)"
    )
    assert re.search(r">=\s*preprocess_commit_every", body), (
        "the periodic commit must fire on a counter reaching the configured interval"
    )


def test_the_loud_failure_guards_run_after_the_final_commit() -> None:
    """The guards refuse a FALSE SUCCESS REPORT — sat before the commit they also destroyed the
    successful encode they were judging (a with_audio corpus with zero streams lost everything)."""
    body = _preprocess_body()
    last_commit = body.rindex("dataset_vol.commit()")
    first_refusal = body.index("Refusing to report")
    assert last_commit < first_refusal, (
        "every dataset_vol.commit() must precede the 'Refusing to report' guards: a guard that "
        "raises before the commit throws the committed-nothing encode away with the report"
    )


def test_the_realized_ceiling_check_precedes_the_phase_b_write() -> None:
    """Post-write, an over-budget sample lands on disk as a complete quartet — which the resume
    guard would then trust on the next dispatch, shipping the training-container OOM after all."""
    body = _preprocess_body()
    assert body.index("realized > max_packed_rows") < body.index("video=video_payload"), (
        "the realized packed-row refusal must fire BEFORE write_h3_precomputed caches the sample"
    )


# ==================================================================================================
# Config-first — the two durability knobs are schema fields with documented defaults
# ==================================================================================================


def test_the_durability_knobs_are_documented_schema_fields() -> None:
    from signet_trainer.config.schema import H3Config

    h3 = H3Config()
    assert h3.preprocess_commit_every == 8
    assert h3.preprocess_overwrite is False
    for name in ("preprocess_commit_every", "preprocess_overwrite"):
        # In model_fields => the SignetConfig reverse guard enumerates them automatically, so a
        # non-default value under an LTX family is already a config-load error, not a silent no-op.
        assert H3Config.model_fields[name].description, f"{name} must carry a description"

    with pytest.raises(ValueError):
        H3Config(preprocess_commit_every=0)  # ge=1: "never commit" must not be expressible
