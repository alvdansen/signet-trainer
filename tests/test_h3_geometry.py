"""H3-04 Wave-1 gate — ``conditioning/h3_geometry.py`` reproduces the P10-1 MEASURED numbers.

Every expected integer in this file was **measured** on live weights / a real A100 and recorded in
``.planning/phases/10-.../P10-1-MEASURED.md`` sections 4-5, or transcribed from diffusers at the
pinned SHA by ``scripts/_h3_probe_modal.py``. Nothing here is a round number chosen for convenience:
a regression in the arithmetic changes an integer that was paid for with GPU time.

CPU-only, zero spend, zero third-party imports beyond pytest.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from signet_trainer.conditioning.h3_geometry import (
    H3_CANVAS_MULTIPLE,
    H3_FRAMES_PER_CHUNK,
    H3_LATENTS_PER_CHUNK,
    H3_VALID_FRAME_COUNTS,
    H3_VISION_TOKENS_EQUAL_LATENT_ROWS,
    H3PackedLayout,
    h3_audio_rows,
    h3_latent_frames,
    h3_packed_seq_len,
    h3_reference_pairing_domain,
    h3_worst_case_packed_seq_len,
    max_packed_rows_for_budget,
    reference_image_size,
    resolve_canvas_size,
    rows_of,
)

REPO = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------------
# The operator's REAL corpus geometry, post-D-10-CROP (P10-1-MEASURED section 5 + the 10-03 plan's
# pairing-domain block). Every sample carries EXACTLY 2 reference slots: a non-environment segment
# gets two rotating character refs, an environment-bearing segment swaps its SECOND character slot
# for the environment ref -- never appends a third.
# ------------------------------------------------------------------------------------------------
CHARACTER_REFS = [(832, 1248, "A"), (2048, 2048, "B"), (1024, 1536, "C")]
ENVIRONMENT_REFS = [(1344, 768, "029"), (1024, 1024, "000"), (1440, 800, "008"), (1344, 768, "023")]

NOMINAL_PROMPT_TOKENS = 96
NOMINAL_ASPECT = (16, 9)


# ------------------------------------------------------------------------------------------------
# 1. The frame law -- 17n+5 pixel -> 5n+2 latent (DIFFERENT modulus AND offset from LTX's (F-1)%8).
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pixel_frames", "latent_frames"),
    [(5, 2), (22, 7), (39, 12), (56, 17), (73, 22), (90, 27), (107, 32), (124, 37)],
)
def test_latent_frames_reproduces_the_measured_law(pixel_frames: int, latent_frames: int) -> None:
    assert h3_latent_frames(pixel_frames) == latent_frames


def test_every_enumerated_valid_count_satisfies_the_law() -> None:
    for frames in H3_VALID_FRAME_COUNTS:
        assert (frames - H3_LATENTS_PER_CHUNK) % H3_FRAMES_PER_CHUNK == 0
        h3_latent_frames(frames)  # must not raise


@pytest.mark.parametrize("bad_frames", [25, 49, 81, 33, 121])
def test_non_conforming_frame_counts_raise_naming_the_law(bad_frames: int) -> None:
    """The LTX campaign buckets {25, 49, 81} are NOT valid H3 counts (P10-1-MEASURED section 5)."""
    with pytest.raises(ValueError) as exc:
        h3_latent_frames(bad_frames)
    assert "17n+5" in str(exc.value)
    assert str(bad_frames) in str(exc.value)


def test_frame_law_names_the_nearest_valid_counts() -> None:
    with pytest.raises(ValueError) as exc:
        h3_latent_frames(25)
    message = str(exc.value)
    assert "22" in message and "39" in message


@pytest.mark.parametrize("bad_frames", [0, -12, 4])
def test_below_the_floor_frame_counts_raise(bad_frames: int) -> None:
    with pytest.raises(ValueError):
        h3_latent_frames(bad_frames)


# ------------------------------------------------------------------------------------------------
# 2. Target canvas -- short edge 768, area cap 768*1344, each axis round()-ed to 32.
# ------------------------------------------------------------------------------------------------


def test_sixteen_by_nine_canvas_is_1344x768() -> None:
    """P10-1-MEASURED section 5: 16:9 canvas -> 1344x768 = 1,008 rows per latent frame."""
    assert resolve_canvas_size(16, 9) == (768, 1344)  # (height, width)
    assert rows_of(768, 1344) == 1008


def test_square_canvas_hits_the_short_edge_not_the_area_cap() -> None:
    assert resolve_canvas_size(1, 1) == (768, 768)


def test_canvas_axes_are_always_multiples_of_32() -> None:
    for aspect in [(16, 9), (9, 16), (4, 3), (3, 4), (1, 1), (21, 9)]:
        height, width = resolve_canvas_size(*aspect)
        assert height % H3_CANVAS_MULTIPLE == 0
        assert width % H3_CANVAS_MULTIPLE == 0


@pytest.mark.parametrize("aspect", [(5, 1), (1, 5)])
def test_canvas_aspect_outside_the_clamp_raises(aspect: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        resolve_canvas_size(*aspect)


# ------------------------------------------------------------------------------------------------
# 3. Reference images -- short edge is a CONFIG value, upscaling ON, NO area cap, per-axis round-32.
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "short_edge", "expected"),
    [
        # P10-1-MEASURED section 5 table, at the true 2048 spec short edge.
        (832, 1248, 2048, (3072, 2048)),
        (2048, 2048, 2048, (2048, 2048)),
        (1024, 1536, 2048, (3072, 2048)),
        (1344, 768, 2048, (2048, 3584)),
        (1440, 800, 2048, (2048, 3680)),
        # The Phase-10 short edges. Note A (2:3) and C (2:3) always encode identically.
        (832, 1248, 1024, (1536, 1024)),
        (2048, 2048, 1024, (1024, 1024)),
        (1024, 1536, 1024, (1536, 1024)),
        (832, 1248, 896, (1344, 896)),
        (1440, 800, 896, (896, 1600)),
    ],
)
def test_reference_image_size_matches_the_measured_table(
    width: int, height: int, short_edge: int, expected: tuple[int, int]
) -> None:
    assert reference_image_size(width, height, short_edge=short_edge) == expected


def test_reference_images_have_no_area_cap() -> None:
    """A 4:1 reference is encoded at 8192x2048 -- the target's area cap does NOT apply here."""
    assert reference_image_size(2048, 512, short_edge=2048) == (2048, 8192)


@pytest.mark.parametrize(("width", "height"), [(4096, 512), (512, 4096), (5000, 1000)])
def test_reference_aspect_outside_one_to_four_raises(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        reference_image_size(width, height, short_edge=1024)


def test_every_image_reference_bills_twice() -> None:
    """encoders.py:509 -- vision tokens == the ref's own latent rows. The biggest ref-cost driver."""
    assert H3_VISION_TOKENS_EQUAL_LATENT_ROWS is True
    layout = h3_packed_seq_len(22, NOMINAL_ASPECT, [(832, 1248)], NOMINAL_PROMPT_TOKENS, 1024)
    assert layout.n_vision == layout.n_cond_video == 1536


# ------------------------------------------------------------------------------------------------
# 4. Audio rows -- 32000/800 = 40 latents/s/channel, stereo, at 24 fps (a small but real term).
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("frames", "rows"), [(22, 74), (56, 186), (124, 414)])
def test_audio_rows_match_the_measured_table(frames: int, rows: int) -> None:
    assert h3_audio_rows(frames) == rows


# ------------------------------------------------------------------------------------------------
# 5. The packed sequence -- the three numbers P10-1 MEASURED on a real A100.
# ------------------------------------------------------------------------------------------------


def test_the_measured_passing_configuration_is_12362_rows() -> None:
    """22f target, 2 refs @1024 -- MEASURED 12,362 rows / 76.36 GiB / 16.6 s/it, PASSED."""
    layout = h3_packed_seq_len(
        22, NOMINAL_ASPECT, [(832, 1248), (2048, 2048)], NOMINAL_PROMPT_TOKENS, 1024
    )
    assert layout.total == 12362
    # The component breakdown, so a regression names WHICH term drifted.
    assert layout.n_text == 2672
    assert layout.n_vision == 2560
    assert layout.n_cond_video == 2560
    assert layout.n_target_video == 7056  # 7 latent frames x 1008 rows
    assert layout.n_target_audio == 74
    assert layout.n_cond_audio == 0


def test_the_true_2048_spec_configuration_is_27722_rows() -> None:
    """Same geometry at the TRUE spec short edge -- MEASURED OOM on one A100-80GB."""
    layout = h3_packed_seq_len(
        22, NOMINAL_ASPECT, [(832, 1248), (2048, 2048)], NOMINAL_PROMPT_TOKENS, 2048
    )
    assert layout.total == 27722


def test_the_no_reference_t2v_baseline_is_37806_rows() -> None:
    """The load-bearing row: plain t2v at 124f already OOMs, so trimming refs cannot rescue it."""
    layout = h3_packed_seq_len(124, NOMINAL_ASPECT, [], NOMINAL_PROMPT_TOKENS, 1024)
    assert layout.total == 37806
    assert layout.n_cond_video == 0
    assert layout.n_vision == 0


def test_layout_total_is_the_sum_of_its_parts() -> None:
    layout = h3_packed_seq_len(
        56, NOMINAL_ASPECT, [(832, 1248), (1440, 800)], NOMINAL_PROMPT_TOKENS, 896, n_cond_audio=64
    )
    assert layout.total == (
        layout.n_text
        + layout.n_cond_video
        + layout.n_cond_audio
        + layout.n_target_audio
        + layout.n_target_video
    )
    assert isinstance(layout, H3PackedLayout)
    assert str(layout.total) in layout.describe()


def test_layout_is_frozen() -> None:
    layout = h3_packed_seq_len(22, NOMINAL_ASPECT, [], NOMINAL_PROMPT_TOKENS, 896)
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.total = 1  # type: ignore[misc]


# ------------------------------------------------------------------------------------------------
# 6. The WORST CASE over the real pairing domain -- the whole point of H3-04.
# ------------------------------------------------------------------------------------------------


def test_the_pairing_domain_is_exactly_fifteen_two_slot_pairs() -> None:
    domain = h3_reference_pairing_domain(CHARACTER_REFS, ENVIRONMENT_REFS)
    assert len(domain) == 15, "3 character pairs + 12 character-by-environment pairs"
    for label, pair in domain:
        assert len(pair) == 2, (
            f"pair {label!r} has {len(pair)} references -- the environment ref SUBSTITUTES for the "
            f"second character slot, it is NEVER appended"
        )
    labels = [label for label, _ in domain]
    assert labels[:3] == ["A+B", "A+C", "B+C"]
    assert "C+008" in labels
    assert len(set(labels)) == 15


def test_worst_case_at_896_passes_and_at_1024_is_over_the_ceiling() -> None:
    ceiling = max_packed_rows_for_budget(79.25, 62.97, 1.21)
    layout_896, label_896 = h3_worst_case_packed_seq_len(
        22, NOMINAL_ASPECT, CHARACTER_REFS, ENVIRONMENT_REFS, NOMINAL_PROMPT_TOKENS, 896
    )
    layout_1024, label_1024 = h3_worst_case_packed_seq_len(
        22, NOMINAL_ASPECT, CHARACTER_REFS, ENVIRONMENT_REFS, NOMINAL_PROMPT_TOKENS, 1024
    )
    assert layout_896.total == 12394
    assert layout_1024.total == 14026
    assert label_896 == "C+008"
    assert label_1024 == "C+008"
    assert layout_896.total <= ceiling
    assert layout_1024.total > ceiling


@pytest.mark.parametrize(
    ("short_edge", "worst_two_char", "worst_char_env"),
    [
        (640, 9642, 9882),
        (768, 10698, 11034),
        (896, 11946, 12394),
        (960, 12642, 13182),
        (1024, 13386, 14026),
    ],
)
def test_the_full_short_edge_sweep_reproduces(
    short_edge: int, worst_two_char: int, worst_char_env: int
) -> None:
    """The 10-03 plan's sweep table. 896 is the adopted Phase-10 short edge."""
    two_char, _ = h3_worst_case_packed_seq_len(
        22, NOMINAL_ASPECT, CHARACTER_REFS, [], NOMINAL_PROMPT_TOKENS, short_edge
    )
    overall, label = h3_worst_case_packed_seq_len(
        22, NOMINAL_ASPECT, CHARACTER_REFS, ENVIRONMENT_REFS, NOMINAL_PROMPT_TOKENS, short_edge
    )
    assert two_char.total == worst_two_char
    assert overall.total == worst_char_env
    assert label == "C+008", "the worst pair is C+008 at every short edge on the sweep"


def test_an_environment_free_corpus_degenerates_to_the_character_pairs() -> None:
    layout, label = h3_worst_case_packed_seq_len(
        22, NOMINAL_ASPECT, CHARACTER_REFS, [], NOMINAL_PROMPT_TOKENS, 896
    )
    assert layout.total == 11946
    assert label == "A+C"
    assert len(h3_reference_pairing_domain(CHARACTER_REFS, [])) == 3


def test_the_worst_case_differs_from_the_nominal_pair() -> None:
    """The blocker H3-04 exists to prevent: pricing A+B alone passes, then a costlier pair OOMs."""
    nominal = h3_packed_seq_len(
        22, NOMINAL_ASPECT, [(832, 1248), (2048, 2048)], NOMINAL_PROMPT_TOKENS, 1024
    )
    worst, _ = h3_worst_case_packed_seq_len(
        22, NOMINAL_ASPECT, CHARACTER_REFS, ENVIRONMENT_REFS, NOMINAL_PROMPT_TOKENS, 1024
    )
    assert nominal.total == 12362
    assert worst.total > nominal.total
    assert (worst.total - nominal.total) / nominal.total > 0.10, "the domain spans >10% in row cost"


def test_six_pairs_are_over_the_ceiling_at_1024() -> None:
    ceiling = max_packed_rows_for_budget(79.25, 62.97, 1.21)
    over = [
        label
        for label, pair in h3_reference_pairing_domain(CHARACTER_REFS, ENVIRONMENT_REFS)
        if h3_packed_seq_len(22, NOMINAL_ASPECT, pair, NOMINAL_PROMPT_TOKENS, 1024).total > ceiling
    ]
    assert sorted(over) == ["A+008", "A+023", "A+029", "C+008", "C+023", "C+029"]


def test_no_pair_is_over_the_ceiling_at_896() -> None:
    ceiling = max_packed_rows_for_budget(79.25, 62.97, 1.21)
    over = [
        label
        for label, pair in h3_reference_pairing_domain(CHARACTER_REFS, ENVIRONMENT_REFS)
        if h3_packed_seq_len(22, NOMINAL_ASPECT, pair, NOMINAL_PROMPT_TOKENS, 896).total > ceiling
    ]
    assert over == []


def test_unlabelled_reference_tuples_still_work() -> None:
    """The (w, h) form the acceptance criteria use -- labels default to A/B/C and E1..En."""
    layout, label = h3_worst_case_packed_seq_len(
        22,
        NOMINAL_ASPECT,
        [(832, 1248), (2048, 2048), (1024, 1536)],
        [(1344, 768), (1024, 1024), (1440, 800), (1344, 768)],
        NOMINAL_PROMPT_TOKENS,
        896,
    )
    assert layout.total == 12394
    assert label == "C+E3"


def test_environment_leg_is_skipped_below_two_slots() -> None:
    """#39 finding 1 / step 2: an environment reference SUBSTITUTES for the LAST character slot
    (h3_ref.py's own rule), so it needs >= 2 total slots to have a character slot to substitute
    for. Before this fix, at references_per_sample == 1, itertools.combinations(characters, 0)
    yielded the single empty combo, so `pair = (env,)` had length 1 -- which EQUALS
    references_per_sample and sailed past the length invariant even though no such sample is
    resolvable (resolve_reference_slots / H3RefStrategy._resolve_slots refuse it by construction).
    """
    domain = h3_reference_pairing_domain(CHARACTER_REFS, ENVIRONMENT_REFS, references_per_sample=1)
    # Character-only leg: C(3, 1) = 3 single-character entries. NO environment-bearing entry.
    assert len(domain) == 3
    environment_labels = {env[2] for env in ENVIRONMENT_REFS}
    for label, pair in domain:
        assert len(pair) == 1
        assert pair[0].label not in environment_labels, (
            f"pair {label!r} carries an environment-only reference at 1 slot -- exactly the "
            f"phantom pair the runtime can never resolve"
        )


@pytest.mark.parametrize("references_per_sample", [0, -1, 4])
def test_an_out_of_range_slot_count_is_a_caller_bug(references_per_sample: int) -> None:
    """The slot count is FIXED at 2 by operator ruling; 0/negative or more slots than characters
    exist is a caller bug, not a config the corpus can satisfy."""
    with pytest.raises(ValueError):
        h3_worst_case_packed_seq_len(
            22,
            NOMINAL_ASPECT,
            CHARACTER_REFS,
            ENVIRONMENT_REFS,
            NOMINAL_PROMPT_TOKENS,
            896,
            references_per_sample=references_per_sample,
        )


# ------------------------------------------------------------------------------------------------
# 7. The VRAM ceiling -- COMPUTED from the measurement, never hardcoded (P10-1 rounded to "~13,800").
# ------------------------------------------------------------------------------------------------


def test_max_packed_rows_reproduces_the_measured_ceiling() -> None:
    assert max_packed_rows_for_budget(79.25, 62.97, 1.21) == 13777


def test_h200_class_budget_reaches_campaign_geometry() -> None:
    """P10-1-MEASURED section 4 extrapolation: H200-141GB clears the 58,302-row campaign."""
    assert max_packed_rows_for_budget(140.0, 62.97, 1.21) > 58302


@pytest.mark.parametrize(("usable", "resident"), [(62.97, 62.97), (40.0, 62.97)])
def test_a_non_positive_budget_raises(usable: float, resident: float) -> None:
    with pytest.raises(ValueError):
        max_packed_rows_for_budget(usable, resident, 1.21)


def test_a_non_positive_row_cost_raises() -> None:
    with pytest.raises(ValueError):
        max_packed_rows_for_budget(79.25, 62.97, 0.0)


# ------------------------------------------------------------------------------------------------
# 8. Import confinement (Anti-Pattern 6) -- STRICTER than conditioning/strategy.py: no torch either.
# ------------------------------------------------------------------------------------------------


def test_h3_geometry_imports_zero_third_party_roots() -> None:
    """Self-deriving closure gate: re-derive the answer in a clean subprocess, never hardcode it.

    ``config/validators.py`` re-exports this module, and ``validators`` sits inside the
    ``download_image`` import closure (``modal/app.py``). A ``torch`` import added here would widen
    that closure -- exactly the BK-01 landmine, whose only other symptom is a ModuleNotFoundError in
    a PAID container after every local gate has already passed.
    """
    probe = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import signet_trainer.conditioning.h3_geometry\n"
        "roots = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "print(sorted(roots - set(sys.stdlib_module_names) - {'signet_trainer'}))\n"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONUTF8": "1"},
        cwd=str(REPO),
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", (
        f"h3_geometry pulled third-party root(s) {out.stdout.strip()} -- it is the CONFINED tier "
        f"(stdlib + dataclasses ONLY, no torch) because config/validators.py re-exports it."
    )
