"""H3-04 Wave-1 gate — the ``17n+5`` frame law and the packed-``seq_len`` VRAM BUDGET refusal.

Two things break the LTX analogy outright and are asserted here:

1. **The frame law has a different modulus AND a different offset.** LTX is ``(F-1) % 8``, H3 is
   ``(F-5) % 17``. The LTX campaign's ``{25, 49, 81}`` buckets are NOT valid H3 counts, and a
   carried-over campaign config is exactly the mistake this validator catches.
2. **A brand-new CLASS of validator with no LTX analog** — a packed-sequence VRAM budget refusal.
   ``P10-1-MEASURED.md`` section 8.5 on the OOMs it exists to prevent: *"This run would have been
   caught locally."*

⛔ The budget check must price the WORST pair in the real pairing domain, not one nominal pair —
``test_worst_case_pair_not_nominal_pair`` is the named regression for that blocker.

CPU-only, zero spend. Every validator here drives the shared ``h3_geometry`` helpers; the
source-scan guard at the bottom proves the law and the budget are never re-derived inline.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from signet_trainer.config.validators import (
    H3_CANVAS_MULTIPLE,
    H3_VALID_FRAME_COUNTS,
    h3_latent_frames,
    h3_packed_seq_len,
    max_packed_rows_for_budget,
    validate_h3_frames,
    validate_h3_reference_budget,
    validate_h3_resolution_bucket,
    validate_h3_resolution_buckets,
    validate_h3_seq_len_budget,
)
from signet_trainer.conditioning.strategy import HEIGHT_SCALE, WIDTH_SCALE

_VALIDATORS_SRC = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "config" / "validators.py"

# ------------------------------------------------------------------------------------------------
# The operator's REAL seven-image reference set, post-D-10-CROP. Defined ONCE and reused by every
# budget test -- a budget assertion driven by a hand-picked convenient pair proves nothing.
#
# Every sample carries EXACTLY 2 reference slots (operator ruling): a non-environment segment gets
# two rotating character refs, an environment-bearing segment SUBSTITUTES the environment ref for
# its second character slot. D-10-ASYM is honored -- the reference REGIME varies, the COUNT does not.
# ------------------------------------------------------------------------------------------------
CHARACTER_REFS = [(832, 1248, "A"), (2048, 2048, "B"), (1024, 1536, "C")]
ENVIRONMENT_REFS = [(1344, 768, "029"), (1024, 1024, "000"), (1440, 800, "008"), (1344, 768, "023")]

NOMINAL_PROMPT_TOKENS = 96
NOMINAL_ASPECT = (16, 9)
NOMINAL_PAIR = [(832, 1248), (2048, 2048)]  # A+B, the pair P10-1 actually MEASURED

# MEASURED on a real A100-80GB (P10-1-MEASURED section 4), gradient checkpointing ON.
A100_USABLE_GIB = 79.25
RESIDENT_GIB = 62.97
MIB_PER_ROW = 1.21


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments and triple-quoted strings (repo convention, 5+ test files).

    The module DOCUMENTS the law and the budget in prose; we must scan only real code, so a
    docstring that legitimately explains ``17n+5`` cannot false-positive the guard.
    """
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


# ------------------------------------------------------------------------------------------------
# 1. The 17n+5 frame law.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("frames", H3_VALID_FRAME_COUNTS)
def test_every_valid_h3_count_passes_unchanged(frames: int) -> None:
    assert validate_h3_frames(frames) == frames


@pytest.mark.parametrize("frames", [141, 158])
def test_the_law_is_not_capped_at_the_enumerated_tuple(frames: int) -> None:
    """``H3_VALID_FRAME_COUNTS`` makes messages actionable; it is not an upper bound on the law."""
    assert validate_h3_frames(frames) == frames


def test_twenty_five_is_rejected_naming_the_law_and_the_nearest_counts() -> None:
    """25 is the LTX campaign bucket. It must never reach a metered container."""
    with pytest.raises(ValueError) as exc:
        validate_h3_frames(25)
    message = str(exc.value)
    assert "17n+5" in message
    assert "25" in message
    assert "22" in message and "39" in message


@pytest.mark.parametrize("ltx_bucket", [25, 49, 81])
def test_the_ltx_campaign_buckets_are_all_rejected(ltx_bucket: int) -> None:
    """A carried-over LTX multi-F bucket set is the exact mistake H3-04 exists to catch."""
    with pytest.raises(ValueError) as exc:
        validate_h3_frames(ltx_bucket)
    assert "17n+5" in str(exc.value)


@pytest.mark.parametrize("frames", [0, -12])
def test_non_positive_frame_counts_raise_on_positivity_first(frames: int) -> None:
    """Positivity is checked FIRST and its message DIFFERS from the modulo message (CR-01 shape).

    ``-12`` satisfies the modulo outright (``(-12 - 5) % 17 == 0``), so without a floor it would
    pass the law and only blow up later as a degenerate tensor shape.
    """
    with pytest.raises(ValueError) as exc:
        validate_h3_frames(frames)
    positivity_message = str(exc.value)
    assert "positive" in positivity_message

    with pytest.raises(ValueError) as modulo_exc:
        validate_h3_frames(25)
    assert positivity_message != str(modulo_exc.value)
    assert "remainder" in str(modulo_exc.value)
    assert "remainder" not in positivity_message


def test_validate_h3_frames_defers_to_the_shared_geometry_helper() -> None:
    """Guard against a future rewrite that re-derives ``% 17`` inline instead of delegating.

    Any count the shared helper accepts, the validator accepts -- and vice versa. This documents
    that the validator owns the POSITIVITY floor and the config framing, while ``h3_geometry``
    owns the LAW.
    """
    for frames in (*H3_VALID_FRAME_COUNTS, 141, 158):
        assert validate_h3_frames(frames) == frames
        h3_latent_frames(frames)  # must not raise either
    for frames in (25, 49, 81, 33):
        with pytest.raises(ValueError):
            validate_h3_frames(frames)
        with pytest.raises(ValueError):
            h3_latent_frames(frames)


# ------------------------------------------------------------------------------------------------
# 2. Resolution buckets -- H3 reuses the %32 spatial rules and swaps ONLY the frame law.
# ------------------------------------------------------------------------------------------------


def test_h3_canvas_multiple_matches_the_reused_spatial_scales() -> None:
    """The reuse of ``validate_width`` / ``validate_height`` is only honest while these agree."""
    assert H3_CANVAS_MULTIPLE == HEIGHT_SCALE == WIDTH_SCALE


def test_valid_h3_bucket_parses_to_a_tuple() -> None:
    assert validate_h3_resolution_bucket("1344x768x22") == (1344, 768, 22)


@pytest.mark.parametrize("bucket", ["1344x768x22", "768x1344x56", "1024x1024x124"])
def test_h3_buckets_round_trip(bucket: str) -> None:
    width, height, frames = validate_h3_resolution_bucket(bucket)
    assert f"{width}x{height}x{frames}" == bucket


def test_h3_bucket_with_an_ltx_frame_count_raises() -> None:
    with pytest.raises(ValueError) as exc:
        validate_h3_resolution_bucket("1344x768x25")
    assert "17n+5" in str(exc.value)


def test_h3_bucket_with_a_non_multiple_of_32_height_raises() -> None:
    with pytest.raises(ValueError) as exc:
        validate_h3_resolution_bucket("1344x760x22")
    assert "760" in str(exc.value)


@pytest.mark.parametrize("bucket", ["1344x768", "1344x768x22x5", "1344xAx22", ""])
def test_malformed_h3_bucket_strings_raise(bucket: str) -> None:
    with pytest.raises(ValueError):
        validate_h3_resolution_bucket(bucket)


def test_h3_bucket_list_returns_unchanged_and_names_the_offending_index() -> None:
    buckets = ["1344x768x22", "1344x768x56"]
    assert validate_h3_resolution_buckets(buckets) is buckets
    with pytest.raises(ValueError) as exc:
        validate_h3_resolution_buckets(["1344x768x22", "1344x768x25"])
    assert "resolution_buckets[1]" in str(exc.value)


# ------------------------------------------------------------------------------------------------
# 3. The packed-seq_len budget refusal -- a NEW class of validator with no LTX analog.
# ------------------------------------------------------------------------------------------------


def test_a_sequence_within_the_ceiling_passes_unchanged() -> None:
    assert validate_h3_seq_len_budget(12394, 13777) == 12394


def test_a_sequence_above_the_ceiling_is_refused_naming_both_numbers() -> None:
    with pytest.raises(ValueError) as exc:
        validate_h3_seq_len_budget(14026, 13777)
    message = str(exc.value)
    assert "14026" in message, "the computed packed sequence length must be named"
    assert "13777" in message, "the declared ceiling must be named"
    assert str(14026 - 13777) in message, "the overage must be named"


def test_the_refusal_carries_an_actionable_remedy() -> None:
    with pytest.raises(ValueError) as exc:
        validate_h3_seq_len_budget(14026, 13777, label="C+008")
    message = str(exc.value)
    assert "C+008" in message, "the refusal must NAME the offending reference pair"
    # The two levers that actually move.
    assert "frame" in message and "reference_image_short_edge" in message
    # Trimming references does not rescue campaign length: the no-reference t2v baseline also OOMs.
    assert "37806" in message
    assert "H200" in message


def test_the_ceiling_is_exactly_at_the_boundary() -> None:
    assert validate_h3_seq_len_budget(13777, 13777) == 13777
    with pytest.raises(ValueError):
        validate_h3_seq_len_budget(13778, 13777)


@pytest.mark.parametrize(("packed", "ceiling"), [(0, 13777), (-1, 13777), (12394, 0)])
def test_non_positive_budget_arguments_raise(packed: int, ceiling: int) -> None:
    with pytest.raises(ValueError):
        validate_h3_seq_len_budget(packed, ceiling)


# ------------------------------------------------------------------------------------------------
# 4. The caller-facing budget check -- and the NAMED REGRESSION it exists for.
# ------------------------------------------------------------------------------------------------


def _reference_budget(ref_short_edge: int):
    return validate_h3_reference_budget(
        22,
        NOMINAL_ASPECT,
        CHARACTER_REFS,
        ENVIRONMENT_REFS,
        NOMINAL_PROMPT_TOKENS,
        ref_short_edge,
        A100_USABLE_GIB,
        RESIDENT_GIB,
        MIB_PER_ROW,
    )


def test_worst_case_pair_not_nominal_pair() -> None:
    """⛔ NAMED REGRESSION: the budget must price the WORST pair, never one nominal pair.

    Driven from the full seven-image reference set:

      * ``ref_short_edge=896``  -> worst pair 12,394 rows, PASSES (within 0.3% of the 12,362-row
        configuration MEASURED passing on a real A100 at 76.36 GiB, so 896 sits on measured ground)
      * ``ref_short_edge=1024`` -> worst pair 14,026 rows, REFUSED, and the message names ``C+008``

    The blocker this guards: the SAME 1024 config priced on the nominal ``A+B`` pair alone reports
    12,362 rows and passes config load -- then OOMs on the first character-by-environment segment.
    """
    ceiling = max_packed_rows_for_budget(A100_USABLE_GIB, RESIDENT_GIB, MIB_PER_ROW)

    passing = _reference_budget(896)
    assert passing.total == 12394
    assert passing.total <= ceiling

    with pytest.raises(ValueError) as exc:
        _reference_budget(1024)
    message = str(exc.value)
    assert "C+008" in message, "the refusal must name the offending pair, not just 'some pair'"
    # The COMPUTED value the validator refused on -- asserted positionally so a stray 14026
    # elsewhere in the remedy prose cannot satisfy this.
    reported = re.search(r"reference pair C\+008: (\d+) rows exceeds", message)
    assert reported is not None, f"refusal did not report a computed row count: {message}"
    assert int(reported.group(1)) == 14026

    # The nominal-pair answer the naive check would have produced -- and passed on.
    nominal = h3_packed_seq_len(22, NOMINAL_ASPECT, NOMINAL_PAIR, NOMINAL_PROMPT_TOKENS, 1024)
    assert nominal.total == 12362
    assert nominal.total <= ceiling, "the naive nominal-pair check would have PASSED this config"
    assert int(reported.group(1)) != nominal.total, (
        "the worst-case answer must DIFFER from the nominal-pair answer -- that difference is the "
        "entire blocker H3-04 guards"
    )


def test_reference_budget_returns_the_worst_layout_for_the_dry_run_banner() -> None:
    layout = _reference_budget(896)
    assert layout.n_target_video == 7056
    assert layout.n_target_audio == 74
    assert str(layout.total) in layout.describe()


def test_reference_budget_rejects_an_illegal_frame_count_before_pricing() -> None:
    with pytest.raises(ValueError) as exc:
        validate_h3_reference_budget(
            25,
            NOMINAL_ASPECT,
            CHARACTER_REFS,
            ENVIRONMENT_REFS,
            NOMINAL_PROMPT_TOKENS,
            896,
            A100_USABLE_GIB,
            RESIDENT_GIB,
            MIB_PER_ROW,
        )
    assert "17n+5" in str(exc.value)


def test_reference_budget_refuses_when_the_weights_alone_exceed_the_gpu() -> None:
    with pytest.raises(ValueError) as exc:
        validate_h3_reference_budget(
            22,
            NOMINAL_ASPECT,
            CHARACTER_REFS,
            ENVIRONMENT_REFS,
            NOMINAL_PROMPT_TOKENS,
            896,
            40.0,
            RESIDENT_GIB,
            MIB_PER_ROW,
        )
    assert "budget" in str(exc.value).lower()


def test_an_environment_free_corpus_is_still_priced() -> None:
    layout = validate_h3_reference_budget(
        22,
        NOMINAL_ASPECT,
        CHARACTER_REFS,
        [],
        NOMINAL_PROMPT_TOKENS,
        896,
        A100_USABLE_GIB,
        RESIDENT_GIB,
        MIB_PER_ROW,
    )
    assert layout.total == 11946


def test_the_campaign_length_is_refused_even_with_zero_references() -> None:
    """Trimming references cannot rescue 124f: the no-reference t2v baseline is already over."""
    with pytest.raises(ValueError) as exc:
        validate_h3_reference_budget(
            124,
            NOMINAL_ASPECT,
            CHARACTER_REFS,
            [],
            NOMINAL_PROMPT_TOKENS,
            896,
            A100_USABLE_GIB,
            RESIDENT_GIB,
            MIB_PER_ROW,
        )
    assert "13777" in str(exc.value)


# ------------------------------------------------------------------------------------------------
# 5. Defer-to-the-shared-validator guard: the law and the budget are NEVER re-derived in validators.
# ------------------------------------------------------------------------------------------------


def test_validators_source_re_exports_h3_geometry_rather_than_redefining_it() -> None:
    code = _strip_comments_and_docstrings(_VALIDATORS_SRC.read_text(encoding="utf-8"))
    assert re.search(r"^\s*from\s+signet_trainer\.conditioning\.h3_geometry\s+import", code, re.M), (
        "config/validators.py must RE-EXPORT the H3 geometry (the compute_seq_len precedent), "
        "never redefine it"
    )


def test_validators_source_carries_no_inline_h3_arithmetic() -> None:
    """The law and the budget must be reachable ONLY through the re-exported geometry helpers."""
    code = _strip_comments_and_docstrings(_VALIDATORS_SRC.read_text(encoding="utf-8"))
    offenders = {
        "inline 17n+5 modulo": r"%\s*17\b",
        "literal H3 frame-count tuple": r"\b5\s*,\s*22\s*,\s*39\b",
        # A bare 4+ digit number. The lookarounds exclude dates (``HANDOFF-2026-06-30``) and the
        # illustrative bucket strings (``'1344x768x22'``) -- neither is a budget figure -- while
        # still catching a hardcoded 12362 / 13777 / 14026 / 37806.
        "literal packed-row / ceiling number": r"(?<![\w.\-])\d{4,}(?![\w.\-])",
        "literal measured VRAM constant": r"\b(?:79\.25|62\.97|1\.21|16\.28|76\.36)\b",
    }
    hits = [name for name, pattern in offenders.items() if re.search(pattern, code)]
    assert not hits, (
        f"config/validators.py re-derives H3 arithmetic inline: {hits}. The 17n+5 law and every "
        f"measured budget number live in conditioning/h3_geometry.py and are re-exported — a "
        f"duplicated copy drifts silently and the only symptom is an OOM in a paid container."
    )


def test_existing_ltx_validators_are_untouched() -> None:
    """Every H3 addition is ADDITIVE: the LTX validators keep their own law and their own names."""
    from signet_trainer.config.validators import (
        ALLOWED_CONDITIONING_MODES,
        A2V_CROSS_MODAL_ATTN_TARGETS,
        validate_frames,
        validate_resolution_bucket,
    )

    assert validate_frames(49) == 49, "LTX's 8k+1 law still accepts 49"
    assert validate_resolution_bucket("768x352x49") == "768x352x49", "LTX buckets return the string"
    with pytest.raises(ValueError):
        validate_frames(22)  # a VALID H3 count is NOT a valid LTX count -- the laws are distinct
    assert "audio_to_video" in ALLOWED_CONDITIONING_MODES
    assert len(A2V_CROSS_MODAL_ATTN_TARGETS) == 4


def test_every_new_h3_validator_is_exported() -> None:
    from signet_trainer.config import validators

    for name in (
        "validate_h3_frames",
        "validate_h3_resolution_bucket",
        "validate_h3_resolution_buckets",
        "validate_h3_seq_len_budget",
        "validate_h3_reference_budget",
        "h3_latent_frames",
        "h3_packed_seq_len",
        "max_packed_rows_for_budget",
        "H3_CANVAS_MULTIPLE",
        "H3_VALID_FRAME_COUNTS",
    ):
        assert name in validators.__all__, f"{name} missing from config.validators.__all__"
