"""Phase 10 (D-10-DEF-4) — PHASE A and PHASE B must describe the SAME reference, proven on CPU.

The first metered H3 dispatch died in PHASE A because the pre-encode had **two** independent
opinions about what a reference image is:

  1. ``build_h3_presentation`` emitted vision blocks **pre-expanded** to N ``<|image_pad|>`` tokens,
     while ``Qwen3VLProcessor.__call__`` expects **exactly one** pad per image and performs the
     expansion itself — so its per-image replacement was exhausted after 2 of 2,576 pads;
  2. PHASE A handed the processor the **raw** file while counting vision tokens off the
     **896-resized** geometry, so reference ``A`` (832x1248) gridded at **1,014** tokens against the
     **1,176** the spans, the AdaLN modality tags and the packed budget all assumed.

⛔ **The crash was luck.** The image's ``transformers`` range resolved to a version whose
replacement path raises; a version whose loop merely stops would have left 2,574 pads un-expanded
and conditioned the model at a plausible-but-wrong shape, silently. So these tests do not check that
something raises in a particular library — they check the two properties that make the disagreement
impossible in the first place, and then check that the arithmetic guard bites when it is violated:

  * **One resize site.** ``prepare_h3_reference_images`` is the only place a reference's encode
    geometry is decided, and it returns the resized pixels and their vision-token count as ONE
    object. Both phases consume that object.
  * **Multiples of 32 on both axes.** That is what makes Qwen3-VL's ``smart_resize`` a no-op and its
    grid exactly ``rows_of(h, w)`` — the agreement is by construction, not by assertion.
  * **The realized expansion is checked anyway**, in both directions, against a stub processor that
    reproduces the real ``Qwen2VLImageProcessor.smart_resize`` / ``Qwen3VLProcessor.__call__``
    arithmetic — including the exact 1,014-vs-1,176 numbers the live failure produced.

Every one of the EIGHT staged references is exercised, read from the REAL campaign config through
the REAL Pydantic validators. Zero GPU, zero Modal spend, no Pillow, no transformers.

**Second subject (D-10-DEF-7): the M-RoPE modality mask.** The stub processor built here is the only
CPU fixture in the repo that can drive ``build_h3_processor_inputs`` end to end, so the behavioural
half of ``_assert_mm_token_type_ids`` lives here too — the arithmetic a source scan cannot prove.
Its counterpart, key-set parity against the REAL ``Qwen3VLProcessor.__call__`` at the pinned
``transformers``, is ``tests/test_h3_processor_output_parity.py``; a stub can never close THAT
question, because a stub only reproduces what we already believe ``__call__`` does.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from signet_trainer.conditioning.h3_geometry import (  # noqa: E402
    H3_CANVAS_MULTIPLE,
    reference_image_size,
    rows_of,
)
from signet_trainer.conditioning.h3_packing import h3_latent_grid_of_reference  # noqa: E402
from signet_trainer.models.h3_loader import (  # noqa: E402
    EXPECTED_H3_IN_CHANNELS,
    EXPECTED_H3_PATCH_SIZE,
)
from signet_trainer.prep.h3_encode import (  # noqa: E402
    H3_IMAGE_PAD_TOKEN,
    H3_VISION_END_TOKEN,
    H3_VISION_START_TOKEN,
    H3PreparedReference,
    build_h3_presentation,
    build_h3_processor_inputs,
    encode_h3_reference_latents,
    prepare_h3_reference_images,
    presentation_refs_from_prepared,
)
from signet_trainer.prep.h3_vae_contract import H3VideoVaeContractStub  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

#: The campaign config is the authority on which references exist and at what size. Reading it
#: through the real loader (rather than restating sizes here) is what makes this a test of the EIGHT
#: STAGED references rather than of eight numbers a test author picked.
_CAMPAIGN_CONFIG = REPO / "configs" / "h3_embe_r1.yaml"

#: ⚠ This file used to declare ``_VAE_SPATIAL_FACTOR = 32`` and hand-roll a ``_StubVideoVae`` that
#: divided by it, calling that "the real spatial factor". **It is not.** 32 is ``H3_CANVAS_MULTIPLE``
#: — the VAE's spatial compression (16) TIMES the patchifier's width (2). The stub therefore emitted
#: latent grids at half the real resolution on each axis, and the assertion below
#: (``latent_w * latent_h == latent_rows``) held only because of that error. The real relation has
#: the patchify in it, and it is now derived from ``h3_packing.h3_latent_grid_of_reference`` — the
#: same function ``H3PositionIdsBuilder`` derives RoPE geometry from — so the two cannot disagree.
_VAE_SPATIAL_FACTOR = H3_CANVAS_MULTIPLE // EXPECTED_H3_PATCH_SIZE[2]
_SHORT_EDGE = 896

# Qwen3-VL vision geometry, transcribed from the mounted ``minimax-h3/processor``'s
# ``preprocessor_config.json`` (patch_size 16 / merge_size 2 / shortest_edge 65536 /
# longest_edge 16777216). They live in the STUB, never in shipped code: shipped code reads
# ``merge_size`` off the processor it was handed.
_PATCH_SIZE = 16
_MERGE_SIZE = 2
_MIN_PIXELS = 65536
_MAX_PIXELS = 16777216


def _staged_references() -> list[tuple[int, int, str]]:
    """The 8 references the campaign declares: 3 character + 5 environment, with their labels."""
    from signet_trainer.config.load import load_config

    h3 = load_config(str(_CAMPAIGN_CONFIG)).h3
    refs = [
        (int(w), int(h), str(label))
        for w, h, label in [*h3.character_reference_sizes, *h3.environment_reference_sizes]
    ]
    assert len(refs) == 8, (
        f"expected the 8 staged references the D-10-CROP preflight verified, got {len(refs)}. If "
        f"the campaign's reference set genuinely changed, that is a RE-ENCODE trigger, not a test "
        f"to relax."
    )
    return refs


# ==================================================================================================
# CPU stubs — no Pillow, no transformers, no GPU
# ==================================================================================================


class _StubImage:
    """The three things the encode path uses: ``.size``, ``.convert``, ``.resize``."""

    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)

    def convert(self, _mode: str) -> "_StubImage":
        return self

    def resize(self, size: tuple[int, int], _resample: object = None) -> "_StubImage":
        return _StubImage(*size)

    def __array__(self, dtype: object = None, copy: object = None) -> object:  # noqa: ARG002
        width, height = self.size
        array = np.full((height, width, 3), 128, dtype=np.uint8)
        return array if dtype is None else array.astype(dtype)


_SPECIAL_IDS = {
    H3_VISION_START_TOKEN: 1,
    H3_VISION_END_TOKEN: 2,
    H3_IMAGE_PAD_TOKEN: 3,
}


class _StubTokenizer:
    """Character-level outside the sentinels, one id per sentinel.

    Character-level is deliberate: it is trivially concatenation-consistent, so the per-segment
    lengths ``build_h3_presentation`` measures its spans with are guaranteed to sum to the length of
    the whole-string tokenization. A merging tokenizer would make a span drift a property of the
    STUB rather than of the code under test.
    """

    def _ids(self, text: str) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            for token, token_id in _SPECIAL_IDS.items():
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    break
            else:
                ids.append(10 + (ord(text[index]) % 100))
                index += 1
        return ids

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:  # noqa: ARG002, FBT
        return self._ids(text)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return _SPECIAL_IDS.get(token)

    def __call__(
        self,
        text: list[str],
        add_special_tokens: bool = True,  # noqa: ARG002, FBT
        return_tensors: str | None = None,  # noqa: ARG002
    ) -> dict:
        ids = [self._ids(one) for one in text]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), len(ids[0]), dtype=torch.long),
        }


def _smart_resize(height: int, width: int) -> tuple[int, int]:
    """``Qwen2VLImageProcessor.smart_resize`` at ``factor = patch_size * merge_size`` (= 32).

    Transcribed from the installed ``transformers/models/qwen2_vl/image_processing_qwen2_vl.py``.
    A size already on the 32 grid and inside the pixel band passes through UNCHANGED — which is the
    property ``prepare_h3_reference_images`` guarantees and this whole file exists to prove.
    """
    factor = _PATCH_SIZE * _MERGE_SIZE
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > _MAX_PIXELS:
        beta = math.sqrt((height * width) / _MAX_PIXELS)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < _MIN_PIXELS:
        beta = math.sqrt(_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class _StubImageProcessor:
    merge_size = _MERGE_SIZE
    patch_size = _PATCH_SIZE

    def __call__(self, images: list, return_tensors: str | None = None) -> dict:  # noqa: ARG002
        grid = []
        for image in images:
            width, height = image.size
            h_bar, w_bar = _smart_resize(height, width)
            grid.append([1, h_bar // _PATCH_SIZE, w_bar // _PATCH_SIZE])
        return {
            "image_grid_thw": torch.tensor(grid, dtype=torch.long),
            "pixel_values": torch.zeros(len(images), 1),
        }


class _StubProcessor:
    """Faithful enough to ``Qwen3VLProcessor`` that its ``__call__`` reproduces the LIVE failure."""

    image_token_id = _SPECIAL_IDS[H3_IMAGE_PAD_TOKEN]
    vision_start_token_id = _SPECIAL_IDS[H3_VISION_START_TOKEN]
    vision_end_token_id = _SPECIAL_IDS[H3_VISION_END_TOKEN]
    image_token = H3_IMAGE_PAD_TOKEN

    def __init__(self) -> None:
        self.tokenizer = _StubTokenizer()
        self.image_processor = _StubImageProcessor()

    def create_mm_token_type_ids(self, input_ids: object) -> list[list[int]]:
        """Transcribed from ``transformers`` 5.14.1 ``ProcessorMixin.create_mm_token_type_ids``.

        Text 0, image 1, video 2, audio 3, membership decided against ``image_token_ids`` /
        ``video_token_ids`` / ``audio_token_ids``. This stub presents images only, so the video and
        audio id lists are empty and only the image branch can mark anything.

        ⚠ Returns a LIST OF LISTS, exactly as the real method does — that is what makes the
        ``torch.tensor(...)`` conversion in ``_mm_token_type_ids`` a real requirement here rather
        than an unexercised line.
        """
        rows = []
        for row in input_ids:
            ids = row if isinstance(row, list) else row.tolist()
            rows.append([1 if token_id == self.image_token_id else 0 for token_id in ids])
        return rows

    def __call__(self, text: list[str], images: list, return_tensors: str | None = None) -> dict:
        """``Qwen3VLProcessor.__call__``'s expansion loop, transcribed. THE PATH WE NO LONGER USE."""
        image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
        grid = image_inputs["image_grid_thw"]
        merge_length = self.image_processor.merge_size**2
        text = list(text)
        index = 0
        for i in range(len(text)):
            while self.image_token in text[i]:
                num_image_tokens = int(grid[index].prod()) // merge_length  # IndexError past N
                text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                index += 1
            text[i] = text[i].replace("<|placeholder|>", self.image_token)
        return {**self.tokenizer(text, return_tensors=return_tensors), **image_inputs}


_LATENT_STATS = (torch.zeros(EXPECTED_H3_IN_CHANNELS), torch.ones(EXPECTED_H3_IN_CHANNELS))


def _descriptor(index: int, label: str) -> dict:
    kind = "environment" if label.isdigit() else "character"
    return {"path": f"refs/slot_{index}.png", "kind": kind, "subject_id": label}


# ==================================================================================================
# The core claim — one geometry, both phases, all EIGHT staged references
# ==================================================================================================


def test_every_staged_reference_prepares_onto_the_32_grid() -> None:
    """Multiples of 32 on both axes is WHY ``smart_resize`` is a no-op — assert it, don't assume it."""
    for width, height, label in _staged_references():
        (prepared,) = prepare_h3_reference_images([_StubImage(width, height)], _SHORT_EDGE)
        encoded_width, encoded_height = prepared.encoded_wh
        assert encoded_width % H3_CANVAS_MULTIPLE == 0, f"{label}: width {encoded_width} off-grid"
        assert encoded_height % H3_CANVAS_MULTIPLE == 0, f"{label}: height {encoded_height} off-grid"
        assert prepared.source_wh == (width, height)
        assert prepared.short_edge == _SHORT_EDGE
        assert min(encoded_width, encoded_height) == _SHORT_EDGE, (
            f"{label}: the short edge must land on the configured {_SHORT_EDGE}"
        )


def test_phase_a_and_phase_b_bill_identical_vision_counts_for_all_eight() -> None:
    """The load-bearing equality, end to end, on CPU: what is PRESENTED == what is ENCODED.

    PHASE A's count is ``H3PreparedReference.vision_tokens`` (what the presentation expands to and
    what the packed budget was priced on). PHASE B's is the ``latent_rows`` written into the cached
    payload. They must be the same integer for every staged reference — 162 rows of disagreement per
    reference is what D-10-DEF-4 was.
    """
    references = _staged_references()
    for index, (width, height, label) in enumerate(references):
        image = _StubImage(width, height)

        # PHASE A
        (prepared,) = prepare_h3_reference_images([image], _SHORT_EDGE)
        (presentation_ref,) = presentation_refs_from_prepared([prepared])
        phase_a = int(presentation_ref.vision_token_counts[0])

        # PHASE B — a fresh open of the SAME source file, exactly as the pre-encode does it.
        payload = encode_h3_reference_latents(
            H3VideoVaeContractStub(),
            [_StubImage(width, height)],
            _SHORT_EDGE,
            *_LATENT_STATS,
            descriptors=[_descriptor(index, label)],
            references_per_sample=1,
        )
        slot = payload["references"][0]
        phase_b = int(slot["latent_rows"])

        expected = rows_of(*reference_image_size(width, height, short_edge=_SHORT_EDGE))
        assert phase_a == phase_b == expected, (
            f"{label} ({width}x{height}): PHASE A presents {phase_a}, PHASE B encodes {phase_b}, "
            f"geometry says {expected}"
        )
        # The latent grid the VAE produced must also match — the payload's rows are not just a
        # number carried alongside the tensor, they describe it. Derived through
        # ``h3_latent_grid_of_reference``, which is what ``H3PositionIdsBuilder`` derives RoPE
        # geometry from, so the cached grid and the coordinates the model gets cannot disagree.
        _, latent_height, latent_width = h3_latent_grid_of_reference(
            width, height, _SHORT_EDGE, EXPECTED_H3_PATCH_SIZE
        )
        assert (int(slot["width"]), int(slot["height"])) == (latent_width, latent_height), (
            f"{label}: the encoded latent grid is "
            f"{(int(slot['width']), int(slot['height']))} but h3_latent_grid_of_reference derives "
            f"{(latent_width, latent_height)} from the same short edge"
        )
        assert (int(slot["width"]), int(slot["height"])) == (
            slot["encoded_wh"][0] // _VAE_SPATIAL_FACTOR,
            slot["encoded_wh"][1] // _VAE_SPATIAL_FACTOR,
        )
        # ⚠ NOT ``latent_w * latent_h == rows``. The patchifier folds (patch_h, patch_w) latent
        # cells into ONE packed row, so the grid area is patch_h * patch_w times the row count. The
        # old form of this assertion passed only because the retired stub divided by 32 (the canvas
        # multiple) instead of by the VAE's own 16 — the two errors cancelled exactly.
        _, patch_h, patch_w = (int(p) for p in EXPECTED_H3_PATCH_SIZE)
        assert (latent_height // patch_h) * (latent_width // patch_w) == phase_b, (
            f"{label}: the patchified latent grid gives "
            f"{(latent_height // patch_h) * (latent_width // patch_w)} packed row(s), but the "
            f"payload declares {phase_b}"
        )


def test_the_processor_grid_equals_the_prepared_count_for_all_eight() -> None:
    """Against the transcribed ``smart_resize``: prepared -> agreement; RAW -> the live divergence.

    The second half is the actual bug reproduced, and it pins the exact numbers the failing run
    produced for reference ``A``: 1,014 from the raw 832x1248, 1,176 from the 896-resized 896x1344.
    """
    divergent = 0
    for width, height, label in _staged_references():
        (prepared,) = prepare_h3_reference_images([_StubImage(width, height)], _SHORT_EDGE)
        prepared_grid = _StubImageProcessor()(images=[prepared.image])["image_grid_thw"]
        assert int(prepared_grid[0].prod()) // (_MERGE_SIZE**2) == prepared.vision_tokens, (
            f"{label}: the processor grids the PREPARED image differently from rows_of() — the "
            f"multiple-of-32 no-op property does not hold, which is the whole structural claim"
        )

        raw_grid = _StubImageProcessor()(images=[_StubImage(width, height)])["image_grid_thw"]
        raw_tokens = int(raw_grid[0].prod()) // (_MERGE_SIZE**2)
        if raw_tokens != prepared.vision_tokens:
            divergent += 1

    assert divergent, (
        "not one staged reference grids differently from its prepared form — then this test cannot "
        "distinguish the fix from the bug and the stub processor is wrong"
    )


def test_reference_a_reproduces_the_measured_1014_vs_1176_divergence() -> None:
    """The concrete number from the live failure, pinned so a refactor cannot quietly restore it."""
    raw = _StubImage(832, 1248)
    (prepared,) = prepare_h3_reference_images([_StubImage(832, 1248)], _SHORT_EDGE)
    assert prepared.encoded_wh == (896, 1344)
    assert prepared.vision_tokens == 1176
    raw_grid = _StubImageProcessor()(images=[raw])["image_grid_thw"]
    assert int(raw_grid[0].prod()) // (_MERGE_SIZE**2) == 1014, (
        "the stub must reproduce the 1,014 tokens the raw 832x1248 reference gridded at on the real "
        "processor — otherwise the 162-row divergence this file guards is not being measured"
    )


# ==================================================================================================
# The text path — the double expansion, caught by arithmetic rather than by a library exception
# ==================================================================================================


def _presentation_for(sizes: list[tuple[int, int]]) -> tuple[str, list, list]:
    prepared = prepare_h3_reference_images([_StubImage(w, h) for w, h in sizes], _SHORT_EDGE)
    presentation, spans = build_h3_presentation(
        "a caption, silent footage",
        presentation_refs_from_prepared(prepared),
        tokenizer=_StubTokenizer(),
    )
    return presentation, spans, prepared


def test_the_stub_processor_reproduces_the_live_double_expansion_failure() -> None:
    """Proof the stub is faithful: the OLD call form dies on the pre-expanded presentation.

    This is the assertion that would have caught defect (1) for free, on CPU, before $0.15 of A100.
    """
    presentation, _spans, prepared = _presentation_for([(1024, 1536), (1440, 800)])
    assert presentation.count(H3_IMAGE_PAD_TOKEN) == sum(r.vision_tokens for r in prepared)
    with pytest.raises(IndexError):
        _StubProcessor()(text=[presentation], images=[r.image for r in prepared])


def test_the_replacement_path_agrees_and_returns_both_halves() -> None:
    """``build_h3_processor_inputs`` produces the same merged dict ``__call__`` would, and checks it."""
    presentation, spans, prepared = _presentation_for([(1024, 1536), (1440, 800)])
    inputs = build_h3_processor_inputs(
        _StubProcessor(), presentation, prepared, vision_spans=spans
    )
    assert set(inputs) >= {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        # D-10-DEF-7: ``ProcessorMixin.__call__`` puts this into ``text_inputs`` before merging, and
        # ``compute_3d_position_ids`` RAISES without it whenever image_grid_thw is present.
        "mm_token_type_ids",
    }
    mask = inputs["mm_token_type_ids"][0].tolist()
    assert sum(1 for value in mask if value == 1) == sum(r.vision_tokens for r in prepared), (
        "the modality mask must mark exactly the vision tokens the grids demand — an all-zeros "
        "mask positions every vision row as text and raises NOWHERE"
    )
    pads = int((inputs["input_ids"] == _SPECIAL_IDS[H3_IMAGE_PAD_TOKEN]).sum())
    assert pads == sum(r.vision_tokens for r in prepared)
    assert len(spans) == 2
    ids = inputs["input_ids"][0]
    for start, stop in spans:
        assert int(ids[start]) == _SPECIAL_IDS[H3_VISION_START_TOKEN]
        assert int(ids[stop - 1]) == _SPECIAL_IDS[H3_VISION_END_TOKEN]


def test_a_raw_image_is_refused_by_type_not_silently_gridded() -> None:
    """The cheapest form of the guard: a raw image can never reach a consumer again."""
    presentation, spans, _prepared = _presentation_for([(1024, 1536), (1440, 800)])
    with pytest.raises(TypeError, match="prepare_h3_reference_images"):
        build_h3_processor_inputs(
            _StubProcessor(),
            presentation,
            [_StubImage(1024, 1536), _StubImage(1440, 800)],
            vision_spans=spans,
        )


def test_a_grid_that_disagrees_with_the_prepared_count_raises() -> None:
    """Hand-built divergence: the pixels of one reference carrying another's declared count."""
    presentation, spans, prepared = _presentation_for([(1024, 1536), (1440, 800)])
    tampered = [
        H3PreparedReference(
            image=prepared[0].image,
            source_wh=prepared[0].source_wh,
            encoded_wh=prepared[0].encoded_wh,
            short_edge=prepared[0].short_edge,
            vision_tokens=prepared[0].vision_tokens - 1,
        ),
        prepared[1],
    ]
    with pytest.raises(RuntimeError, match="realized vision grid disagrees"):
        build_h3_processor_inputs(_StubProcessor(), presentation, tampered, vision_spans=spans)


def test_a_single_pad_presentation_is_refused_as_a_pad_count_mismatch() -> None:
    """The double-expansion defect from the OTHER side — and NOT via a library exception.

    A presentation carrying one pad per image (what ``Qwen3VLProcessor.__call__`` wants) is exactly
    what a "let the processor expand it" fix would produce. Under this code the vision spans and the
    modality tags would then be measured over a stream that no longer matches, so it is refused by
    arithmetic — no transformers version required.
    """
    presentation, spans, prepared = _presentation_for([(1024, 1536), (1440, 800)])
    collapsed = presentation
    for reference in prepared:
        collapsed = collapsed.replace(
            H3_IMAGE_PAD_TOKEN * reference.vision_tokens, H3_IMAGE_PAD_TOKEN, 1
        )
    assert collapsed.count(H3_IMAGE_PAD_TOKEN) == 2
    with pytest.raises(RuntimeError, match="double-expansion"):
        build_h3_processor_inputs(_StubProcessor(), collapsed, prepared, vision_spans=spans)


def test_a_shifted_vision_span_is_refused() -> None:
    """The spans drive ``h3_token_tags``; an off-by-one mis-tags conditioning rows, silently."""
    presentation, spans, prepared = _presentation_for([(1024, 1536), (1440, 800)])
    shifted = [(start + 1, stop + 1) for start, stop in spans]
    with pytest.raises(RuntimeError, match="does not bracket"):
        build_h3_processor_inputs(_StubProcessor(), presentation, prepared, vision_spans=shifted)


def test_a_short_edge_mismatch_between_the_phases_raises() -> None:
    """Re-using PHASE A's prepared references at a different short edge is a RE-ENCODE, not a resize."""
    prepared = prepare_h3_reference_images([_StubImage(1024, 1536)], _SHORT_EDGE)
    with pytest.raises(ValueError, match="RE-ENCODE"):
        prepare_h3_reference_images(prepared, 1024)


def test_preparing_an_already_prepared_reference_is_a_no_op() -> None:
    """PHASE B may pass PHASE A's list straight through — same object, no second resample."""
    prepared = prepare_h3_reference_images([_StubImage(1024, 1536)], _SHORT_EDGE)
    again = prepare_h3_reference_images(prepared, _SHORT_EDGE)
    assert again[0] is prepared[0]


def test_the_text_only_path_needs_no_references() -> None:
    """An encode with no vision blocks must still work — and must not fabricate an image input."""
    inputs = build_h3_processor_inputs(_StubProcessor(), "just a caption", (), vision_spans=())
    assert "image_grid_thw" not in inputs
    assert int(inputs["input_ids"].shape[-1]) == len("just a caption")
    # ⛔ ``__call__`` adds mm_token_type_ids whenever TEXT is given, not when IMAGES are — so the
    # text-only branch carries it too, all zeros. Gating it on "has references" would make our
    # output key set diverge from __call__'s here and nowhere else (D-10-DEF-7 / GAP-3).
    assert "mm_token_type_ids" in inputs
    assert set(inputs["mm_token_type_ids"][0].tolist()) == {0}


# ==================================================================================================
# D-10-DEF-7 — the M-RoPE modality mask. Both failure modes below raise NOWHERE in the library.
#
# `compute_3d_position_ids` raises only when `mm_token_type_ids` is MISSING (that is how D-10-DEF-7
# surfaced, on a metered container). A mask that is PRESENT and WRONG passes straight through, and
# the guard evaporates entirely on the `inputs_embeds` path. `get_rope_index` then pulls ONE grid
# per CONTIGUOUS run and builds that run's positions from the GRID rather than from the run's
# measured length, checking only the TOTAL sequence length at the end — so both mutations below
# produce a correctly-shaped `position_ids` describing conditioning the model was never shown.
# ==================================================================================================


class _AllZerosMaskProcessor(_StubProcessor):
    """``create_mm_token_type_ids`` marking NOTHING — the real library's silent fallback.

    ``ProcessorMixin.image_token_ids`` falls back to ``[getattr(self, "image_token_id", None)]``,
    and ``np.isin(ids, [None])`` marks nothing. The result is one all-text run: the grid iterators
    are never consumed, the concatenated position length still equals ``seq_len``, and the
    assignment succeeds. Nothing raises anywhere.
    """

    def create_mm_token_type_ids(self, input_ids: object) -> list[list[int]]:
        return [[0] * len(row if isinstance(row, list) else row.tolist()) for row in input_ids]


class _MergedRunMaskProcessor(_StubProcessor):
    """A mask that marks the whole span from the FIRST pad to the LAST as one image run.

    Total marked tokens still exceed nothing obvious, the key is present, the dtype is right — and
    two references collapse into ONE contiguous run, so ``get_rope_index`` consumes ONE grid and
    positions the second reference with the first one's lattice.
    """

    def create_mm_token_type_ids(self, input_ids: object) -> list[list[int]]:
        rows = []
        for row in input_ids:
            ids = row if isinstance(row, list) else row.tolist()
            positions = [i for i, token in enumerate(ids) if token == self.image_token_id]
            mask = [0] * len(ids)
            for index in range(positions[0], positions[-1] + 1):
                mask[index] = 1
            rows.append(mask)
        return rows


class _VideoTypedMaskProcessor(_StubProcessor):
    """A mask claiming VIDEO tokens the image-only builder never produced a grid for.

    ``get_rope_index``'s ``grid_iters[2]`` is ``None`` when no ``video_grid_thw`` was passed, so this
    is a ``TypeError`` deep inside the model — after the encode has already started.
    """

    def create_mm_token_type_ids(self, input_ids: object) -> list[list[int]]:
        # A TEXT token is retyped, deliberately: the IMAGE runs stay exactly right, so this reaches
        # the video/audio refusal instead of tripping the per-run check on its way there.
        rows = _StubProcessor.create_mm_token_type_ids(self, input_ids)
        for mask in rows:
            mask[0] = 2
        return rows


def test_an_all_zeros_modality_mask_is_refused() -> None:
    """The most dangerous mutation in this phase: present, well-shaped, and marking nothing."""
    presentation, spans, prepared = _presentation_for([(1024, 1536), (1440, 800)])
    with pytest.raises(RuntimeError, match="marks NO image tokens"):
        build_h3_processor_inputs(
            _AllZerosMaskProcessor(), presentation, prepared, vision_spans=spans
        )


def test_a_merged_modality_run_is_refused() -> None:
    """Two vision blocks in ONE run consume ONE grid — the second reference gets the wrong lattice."""
    presentation, spans, prepared = _presentation_for([(1024, 1536), (1440, 800)])
    with pytest.raises(RuntimeError, match="image runs"):
        build_h3_processor_inputs(
            _MergedRunMaskProcessor(), presentation, prepared, vision_spans=spans
        )


def test_a_video_typed_token_with_no_video_grid_is_refused() -> None:
    """A VIDEO reference is a D-10-DEF-4-sized re-derivation, not a widening of this path."""
    presentation, spans, prepared = _presentation_for([(1024, 1536), (1440, 800)])
    with pytest.raises(RuntimeError, match="video token"):
        build_h3_processor_inputs(
            _VideoTypedMaskProcessor(), presentation, prepared, vision_spans=spans
        )


def test_the_run_check_is_order_sensitive_not_a_count() -> None:
    """⛔ The distinction that matters: per-run mismatches CANCEL in the total ``get_rope_index`` checks.

    Two references of 1,176 and 1,372 vision tokens, marked in the WRONG order, still sum to 2,548.
    ``get_rope_index`` verifies only that total, so a count-based assertion passes on exactly the
    swap that hands reference ``A`` the environment reference's H x W lattice. Driven directly
    because reaching this end-to-end would trip the earlier grid-vs-declared check first — which is
    the point: this is the residue that check cannot see.
    """
    from signet_trainer.prep.h3_encode import _assert_mm_token_type_ids

    correct = torch.tensor([[0] * 3 + [1] * 1176 + [0] * 4 + [1] * 1372 + [0] * 2], dtype=torch.int)
    swapped = torch.tensor([[0] * 3 + [1] * 1372 + [0] * 4 + [1] * 1176 + [0] * 2], dtype=torch.int)

    _assert_mm_token_type_ids(correct, [1176, 1372])  # must not raise
    assert sum([1372, 1176]) == sum([1176, 1372]), "the premise: the totals are identical"
    with pytest.raises(RuntimeError, match=r"image runs \[1372, 1176\]"):
        _assert_mm_token_type_ids(swapped, [1176, 1372])


def test_the_mask_survives_the_real_builder_on_both_branches() -> None:
    """Non-vacuity for the three refusals above: the CORRECT mask must pass, end to end."""
    presentation, spans, prepared = _presentation_for([(1024, 1536), (1440, 800)])
    inputs = build_h3_processor_inputs(_StubProcessor(), presentation, prepared, vision_spans=spans)
    mask = inputs["mm_token_type_ids"]
    assert mask.shape == inputs["input_ids"].shape, (
        "the mask must be one value per token — get_rope_index indexes them together"
    )
    runs: list[list[int]] = []
    for value in mask[0].tolist():
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    assert [length for value, length in runs if value == 1] == [
        r.vision_tokens for r in prepared
    ], "the realized runs must equal the per-reference vision counts, in presentation order"
