"""Phase 10 — the per-config reference selector (``validation.reference_subject_ids``).

``h3_sample`` used to hardcode ``rows[0]``, so every eval prompt at every length conditioned on ONE
reference pair. For a phase whose headline value is ref2v that is the thinnest possible axis: the
mixed-length rungs vary LENGTH and nothing else, and nothing varied the reference.

The selector names the reference CONDITION by ``subject_id`` — the config's own declared vocabulary
— rather than a row index or a clip name. Both alternatives were rejected for concrete reasons this
file asserts:

  * an INDEX silently re-points if the manifest is ever re-staged in a different order
    (``test_a_restage_moves_an_index_but_not_a_condition``);
  * a clip / media name is client property and must never enter a tracked config, so the selector
    must also never PRINT one (``test_the_failure_message_carries_no_paths_or_filenames``).

Everything here runs on a SYNTHETIC manifest — no client property, no Volume, no GPU, zero spend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DATA_ROOT = Path("/dataset/h3_synthetic")

#: Three character subjects + two environment illustrations, using the config's declared
#: subject-id vocabulary shape (letters for characters, digits for environments). Paths are
#: invented placeholders — real reference filenames live only on the dataset Volume.
_POOL = [
    {"path": "refs/char_a.png", "subject_id": "A"},
    {"path": "refs/char_b.png", "subject_id": "B"},
    {"path": "refs/char_c.png", "subject_id": "C"},
]


def _row(name: str, *, environment: str | None = None, pool: list | None = None) -> dict:
    row: dict = {
        "media_path": f"clips/{name}.mp4",
        "caption": f"a synthetic caption for {name}, silent footage",
        "character_references": list(pool if pool is not None else _POOL),
    }
    if environment:
        row["environment_reference"] = {
            "path": f"refs/env_{environment}.png",
            "subject_id": environment,
        }
    return row


def _manifest() -> list[dict]:
    """Row 0 carries an environment ref; the 'bed' row is PINNED to a single-entry pool."""
    return [
        _row("scene_0", environment="029"),
        _row("scene_1", environment="029"),
        _row("scene_2"),
        _row("scene_3"),
        # The pinned pair: a one-entry character pool removes the D-10-PAIRSEED rotation's freedom,
        # so this row resolves to C + 018 at ANY index — which is what makes the condition durable.
        _row("bed", environment="018", pool=[_POOL[2]]),
        _row("scene_5"),
    ]


def _select(subject_ids, rows=None, *, references_per_sample: int = 2):
    import signet_trainer.modal.fns as fns  # noqa: PLC0415 — importing fns builds the app graph

    return fns._h3_select_reference_row(
        rows if rows is not None else _manifest(),
        DATA_ROOT,
        subject_ids=subject_ids,
        references_per_sample=references_per_sample,
        reference_pair_seed=42,
        environment_ref_last=True,
    )


def _ids(references) -> list[str]:
    return [r["subject_id"] for r in references]


# ==================================================================================================
# Backwards compatibility — the default must be the OLD behaviour, exactly
# ==================================================================================================


@pytest.mark.parametrize("empty", [None, [], ()])
def test_an_unset_selector_is_manifest_row_zero(empty) -> None:
    """Every config predating this field must render byte-identically. No new default behaviour."""
    import signet_trainer.modal.fns as fns  # noqa: PLC0415

    rows = _manifest()
    index, references = _select(empty, rows)
    assert index == 0
    assert references == fns._h3_resolve_references(
        rows[0],
        0,
        DATA_ROOT,
        references_per_sample=2,
        reference_pair_seed=42,
        environment_ref_last=True,
    )


# ==================================================================================================
# Selecting by identity
# ==================================================================================================


def test_the_selector_returns_exactly_the_named_condition() -> None:
    """The point of the field: two sibling renders differing ONLY in the reference condition."""
    default_index, default_refs = _select([])
    assert _ids(default_refs) == ["A", "029"]

    index, references = _select(["B", "029"])
    assert _ids(references) == ["B", "029"]
    assert index != default_index, (
        "a different character against the SAME environment must come from a different row — "
        "otherwise the selector is not varying anything"
    )

    bed_index, bed = _select(["C", "018"])
    assert _ids(bed) == ["C", "018"]
    assert bed_index == 4


def test_the_selector_does_not_reorder_or_re_roll_anything() -> None:
    """It chooses WHICH row supplies the slots — D-10-REFORDER and D-10-PAIRSEED stay untouched."""
    import signet_trainer.modal.fns as fns  # noqa: PLC0415

    rows = _manifest()
    index, references = _select(["C", "018"], rows)
    assert references == fns._h3_resolve_references(
        rows[index],
        index,
        DATA_ROOT,
        references_per_sample=2,
        reference_pair_seed=42,
        environment_ref_last=True,
    ), "the selected slots must be exactly what the REAL resolver produces for that row"
    assert _ids(references)[-1] == "018", (
        "D-10-REFORDER puts the environment reference LAST and the selector must not disturb it"
    )


def test_the_match_is_order_exact_and_says_so() -> None:
    """A reordered reference set is a genuinely different request on the shared rotary clock."""
    with pytest.raises(RuntimeError) as excinfo:
        _select(["018", "C"])
    message = str(excinfo.value)
    assert "D-10-REFORDER" in message
    assert "['C', '018']" in message, (
        "the refusal must point at the order that DOES exist — otherwise the operator is told only "
        "that they are wrong, not how"
    )


# ==================================================================================================
# The durability claim — an index re-points on a re-stage, a condition does not
# ==================================================================================================


def test_a_restage_moves_an_index_but_not_a_condition() -> None:
    """THE reason this is not ``rows[N]``.

    Re-staging the corpus in a different order changes which row lives at any given index — and,
    because the D-10-PAIRSEED rotation is a function of the row index, it also changes which pair a
    given clip is assigned. A row index therefore names a moving target. The reference CONDITION
    does not move: ``["C", "018"]`` still selects the pinned pair, and still returns those slots.
    """
    original = _manifest()
    restaged = [*original[3:], *original[:3]]  # a re-stage: same rows, different order

    before_index, before_refs = _select(["C", "018"], original)
    after_index, after_refs = _select(["C", "018"], restaged)

    assert _ids(before_refs) == _ids(after_refs) == ["C", "018"], (
        "the selected CONDITION must survive a re-stage — that is the whole durability claim"
    )
    assert before_index != after_index, (
        "the row carrying it moved, so a config naming an index would now point somewhere else. If "
        "it did not move, this test cannot distinguish an index from a condition"
    )

    # And the negative half: an index-based selector genuinely would have re-pointed.
    _, before_row0 = _select([], original)
    _, after_row0 = _select([], restaged)
    assert _ids(before_row0) != _ids(after_row0), (
        "row 0 must resolve to a DIFFERENT condition after the re-stage — this is the silent "
        "re-pointing a bare rows[N] index would have delivered"
    )


# ==================================================================================================
# Failing loud
# ==================================================================================================


def test_an_unmatched_condition_raises_and_never_falls_back() -> None:
    """Falling back to row 0 would render a different probe under this config's label."""
    with pytest.raises(RuntimeError) as excinfo:
        _select(["A", "999"])
    message = str(excinfo.value)
    assert "matches no manifest row" in message
    assert "Available reference conditions" in message, (
        "the refusal must LIST what is available — an operator who mistyped a subject id should not "
        "have to go read the manifest off a Volume to find the right spelling"
    )
    for expected in ("['A', '029']", "['B', '029']", "['C', '018']"):
        assert expected in message, f"the available-choices list is missing {expected}"


def test_a_slot_count_mismatch_raises() -> None:
    """Three references per sample was never priced by the packed-sequence budget."""
    with pytest.raises(RuntimeError, match="references_per_sample"):
        _select(["A", "B", "029"])


def test_an_empty_manifest_raises() -> None:
    with pytest.raises(RuntimeError, match="no rows"):
        _select([], [])


def test_the_failure_message_carries_no_paths_or_filenames() -> None:
    """⛔ Client-property hygiene: the selector reports subject_ids, never media or reference names.

    The available-choices list is the one place this function prints anything derived from the
    manifest, and the manifest is where the real filenames live. It must stay label-only — a
    refusal that helpfully printed ``refs/<real file>.png`` would leak into a log, a paste, or a
    SUMMARY.
    """
    with pytest.raises(RuntimeError) as excinfo:
        _select(["A", "999"])
    message = str(excinfo.value)
    for leak in ("refs/", "clips/", ".png", ".mp4", "media_path", str(DATA_ROOT)):
        assert leak not in message, f"the refusal leaked {leak!r}: {message}"


# ==================================================================================================
# Schema — the field is on validation, defaults empty, and every shipped config still loads
# ==================================================================================================


def test_the_field_lives_on_validation_and_defaults_to_empty() -> None:
    from signet_trainer.config.schema import ValidationConfig  # noqa: PLC0415

    field = ValidationConfig.model_fields["reference_subject_ids"]
    assert field.default_factory is list, (
        "the default must be EMPTY, so adding the field changes no existing config's behaviour"
    )
    assert ValidationConfig().reference_subject_ids == []


def test_the_embe_eval_is_a_clean_split_one_axis_per_comparison() -> None:
    """The eval's whole interpretability rests on ONE variable moving per comparison.

    Three LENGTH rungs sharing one reference pair, plus two REFERENCE variants at the cheapest
    length sharing everything else. If a rung ever drifts on a second field, a difference in its
    grid stops being attributable and the comparison is worth nothing — which is exactly the kind
    of drift a config copy invites, so it is pinned here rather than in a comment.
    """
    from signet_trainer.config.load import load_config  # noqa: PLC0415

    repo = Path(__file__).resolve().parents[1]
    load = lambda name: load_config(str(repo / "configs" / f"{name}.yaml"))  # noqa: E731

    rungs = {
        22: load("h3_embe_r1_sample_short"),
        56: load("h3_embe_r1_sample_mid"),
        124: load("h3_embe_r1_sample"),
    }
    variants = {
        "b029": load("h3_embe_r1_sample_ref_b029"),
        "c018": load("h3_embe_r1_sample_ref_c018"),
    }
    everything = {**rungs, **variants}

    # --- shared by all five: the adapter, the geometry and the prompt set ---
    for label, config in everything.items():
        assert config.model.family == "h3", label
        assert config.output_dir == rungs[22].output_dir, (
            f"{label}: every config in the matrix must resolve the SAME adapter — find_latest() "
            f"reads CHECKPOINTS_DIR/<output_dir>, so a different one renders a different run"
        )
        assert config.training.max_steps == rungs[22].training.max_steps, (
            f"{label}: max_steps is part of the checkpoint dir name find_latest resolves"
        )
        assert list(config.validation.prompts) == list(rungs[22].validation.prompts), (
            f"{label}: the eval prompt set must be identical across the whole matrix, or a grid "
            f"difference could be a prompt difference"
        )
        assert config.validation.seed == rungs[22].validation.seed, label
        assert config.validation.num_inference_steps == rungs[22].validation.num_inference_steps
        assert config.h3.reference_pair_seed == rungs[22].h3.reference_pair_seed, label
        assert config.h3.environment_ref_last is True, label
        assert config.h3.reference_image_short_edge == rungs[22].h3.reference_image_short_edge

    # --- axis 1: LENGTH varies, the reference condition does NOT ---
    for frames, config in rungs.items():
        assert config.validation.frame_count == frames
        assert list(config.validation.reference_subject_ids) == ["A", "029"], (
            f"the {frames} f rung must share the length rungs' reference pair — otherwise the "
            f"mixed-length comparison confounds length with reference"
        )

    # --- axis 2: the REFERENCE condition varies, everything else does NOT ---
    for label, config in variants.items():
        assert config.validation.frame_count == 22, (
            f"{label}: a reference variant runs at the TRAINED length, so a difference is "
            f"attributable to the reference rather than to length extrapolation"
        )
        assert list(config.validation.reference_subject_ids) != ["A", "029"], (
            f"{label}: a variant that declares the length rungs' pair varies nothing"
        )
    assert list(variants["b029"].validation.reference_subject_ids) == ["B", "029"]
    assert list(variants["c018"].validation.reference_subject_ids) == ["C", "018"]
    assert (
        variants["b029"].validation.reference_subject_ids[1]
        == rungs[22].validation.reference_subject_ids[1]
    ), (
        "the b029 variant must hold the ENVIRONMENT slot fixed against the length rungs — that is "
        "what makes it a probe of the character reference specifically"
    )

    # --- 5 configs x 6 prompts x 2 columns = 60 clips ---
    assert len(everything) == 5
    assert sum(len(c.validation.prompts) * 2 for c in everything.values()) == 60


#: The D-10-AUDIO control tag, VERBATIM. Every staged corpus caption ends in it, so prompts that
#: carry it are in the caption grammar and prompt 6 (prompt 5 with it removed) isolates the lever.
_AUDIO_TAG = ", silent footage"

#: In-house Phase 10 eval matrix — 0 files on a published clone (CONTRIBUTING.md documents this
#: file as non-signal there). Hoisted to a module constant per the loud-guard pattern already used
#: at ``test_h3_pipeline_local_source.py:316-320``: emptiness must surface as a guard-test failure
#: plus a skip-with-reason on the parametrized test below, never as a test that iterates over
#: nothing and still reports PASSED (issue #41 finding 2).
_EMBE_SAMPLE_CONFIGS = sorted(
    Path(__file__).resolve().parents[1].glob("configs/h3_embe_r1_sample*.yaml")
)


def test_the_embe_sample_configs_exist() -> None:
    """Guard for ``_EMBE_SAMPLE_CONFIGS``. On a published clone this FAILS loudly instead of
    letting ``test_the_audio_control_tag_probe_isolates_exactly_one_lever`` pass over zero cases."""
    assert len(_EMBE_SAMPLE_CONFIGS) == 5


@pytest.mark.parametrize("path", _EMBE_SAMPLE_CONFIGS, ids=lambda p: p.name)
def test_the_audio_control_tag_probe_isolates_exactly_one_lever(path: Path) -> None:
    """D-10-AUDIO: prompt 6 must be prompt 5 with the tag REMOVED and nothing else changed.

    H3 is guidance-distilled with no negative branch, so NEGATION may reinforce rather than
    suppress — OMISSION is the only reliable lever. The pair only isolates it if the two prompts
    differ by exactly the tag, byte for byte.
    """
    from signet_trainer.config.load import load_config  # noqa: PLC0415

    prompts = list(load_config(str(path)).validation.prompts)
    assert len(prompts) == 6, f"{path.name}: the eval set is 6 prompts"
    assert prompts[4] == prompts[5] + _AUDIO_TAG, (
        f"{path.name}: prompt 6 must be prompt 5 with `{_AUDIO_TAG}` removed and NOTHING else "
        f"changed, or the omission probe measures a prompt difference instead of the lever"
    )
    assert all(p.endswith(_AUDIO_TAG) for p in prompts[:5]), (
        f"{path.name}: prompts 1-5 must all carry the control tag verbatim — a named "
        f"association is steerable, an inconsistent one is not"
    )


def test_the_eval_set_is_ranging_weighted_not_caption_dominated() -> None:
    """House rule ``eval-prompts-never-copy-captions``: a caption-dominated set tests memorization.

    A couple of verbatim dataset captions are fine (in-distribution recall) but the set must stay
    weighted toward RANGING probes. This pins the SHAPE — 2 in-distribution, 4 ranging — without
    pinning the captions themselves, which live on the dataset Volume and are client property.
    """
    from signet_trainer.config.load import load_config  # noqa: PLC0415

    repo = Path(__file__).resolve().parents[1]
    prompts = list(
        load_config(str(repo / "configs" / "h3_embe_r1_sample_short.yaml")).validation.prompts
    )
    in_distribution = prompts[:2]
    ranging = prompts[2:]
    assert len(ranging) > len(in_distribution), (
        "the set must stay weighted toward ranging probes — a caption-dominated eval only tells "
        "you whether the adapter memorized its training set"
    )
    assert len(set(prompts)) == len(prompts), (
        "every prompt must be distinct — prompts 5 and 6 differ by the audio tag, so a true "
        "duplicate means a prompt was pasted twice and a metered grid row renders nothing new"
    )
    # The anchor is the natural name, and appearance is deliberately never described — the REFERENCE
    # IMAGES carry likeness, and a prompt that described the face would let the model pass on text
    # alone (which would tell us nothing about the reference pathway, the point of the phase).
    for prompt in prompts:
        assert "Embe" in prompt, "every eval prompt must carry the anchor"


def test_every_shipped_h3_config_still_loads_and_declares_its_slots() -> None:
    """A config that names a condition must also DECLARE those subjects' sizes, or it is unpriced.

    ``exercised`` guards against the unconditional-``continue`` failure mode (issue #41 finding 2):
    if every shipped ``h3_*.yaml`` skips (as the sole clone-shipped ``h3_t2va_noref.example.yaml``
    does today — it deliberately names no condition), the loop body never runs an assertion and the
    test would otherwise report PASSED having checked nothing.
    """
    from signet_trainer.config.load import load_config  # noqa: PLC0415

    repo = Path(__file__).resolve().parents[1]
    exercised = False
    for path in sorted((repo / "configs").glob("h3_*.yaml")):
        config = load_config(str(path))
        wanted = list(config.validation.reference_subject_ids)
        if not wanted:
            continue
        exercised = True
        declared = {
            str(entry[2])
            for entry in [
                *config.h3.character_reference_sizes,
                *config.h3.environment_reference_sizes,
            ]
        }
        missing = [s for s in wanted if s not in declared]
        assert not missing, (
            f"{path.name} conditions on {wanted} but does not declare {missing} in "
            f"h3.character_reference_sizes / h3.environment_reference_sizes. An undeclared subject "
            f"is a reference the worst-case packed-row budget never priced."
        )
    assert exercised, "no shipped h3 config names a condition — this check ran on nothing"
