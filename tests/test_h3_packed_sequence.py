"""CPU contract tests for the MiniMax-H3 packed-sequence batch (H3-05, Plan 10-06).

Every assertion here is a contract on ``src/signet_trainer/train/h3_step.py``, which is a HARVEST of
``scripts/_h3_probe_modal.py::_synthetic_batch`` — the batch that ran a real forward + backward +
optimizer step on an A100-80GB. The probe is authoritative; these tests pin the harvest so a later
"simplification" cannot quietly change the wire format.

Four things here are load-bearing and easy to get wrong:

  1. **The three index tensors partition the sequence exactly once each.** A row that is in two
     segments, or in none, is a silent corruption — the model reads it under the wrong head.
  2. **Qwen vision rows live in the TEXT span but carry modality tag 0 (video)**, sentinels
     included (P10-0e section 1). The tags index the AdaLN table via
     ``timestep_indices * H3_MODALITY_NUM + token_tags``, so they are a CHECKPOINT CONTRACT — a
     trainer that tags vision rows 1 silently modulates the wrong AdaLN rows and nothing raises.
  3. **The loss mask drops precisely the conditioning rows** — the transformer returns them
     UNMASKED and says so (``transformer_minimax_h3.py`` L44-50: masking them is the caller's job).
  4. **Target audio rows stay PRESENT and NOISED** with ``audio_in_loss=False`` (D-10-AUDIO):
     not-targeting audio is not the same as training silence.

All tensors are CPU float32 and tiny. Nothing here imports ``diffusers``, ``modal`` or touches CUDA.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from signet_trainer.conditioning.h3_geometry import (
    H3_A100_80GB_USABLE_GIB,
    H3_MIB_PER_PACKED_ROW,
    H3_NOMINAL_PROMPT_TOKENS,
    H3_RESIDENT_GIB_RANK64,
    h3_packed_seq_len,
    h3_worst_case_packed_seq_len,
    max_packed_rows_for_budget,
)
from signet_trainer.models.h3_loader import (
    EXPECTED_H3_AUDIO_IN_CHANNELS,
    EXPECTED_H3_IN_CHANNELS,
    EXPECTED_H3_MODALITY_NUM,
    EXPECTED_H3_PATCH_SIZE,
    EXPECTED_H3_TEXT_DIM,
)
from signet_trainer.train.h3_step import (
    H3_AUDIO_CONDITION_PIN,
    H3_AUDIO_SIGMA_SHIFT,
    H3_AUDIO_TAG,
    H3_MODALITY_NUM,
    H3_PACKED_BATCH_KEYS,
    H3_TEXT_TAG,
    H3_VIDEO_SIGMA_SHIFT,
    H3_VIDEO_TAG,
    H3_VISUAL_CONDITION_PIN,
    H3PackedBatch,
    build_h3_packed_batch,
    h3_indices,
    h3_layout_row_counts,
    h3_loss_mask,
    h3_row_timesteps,
    h3_segment_offsets,
    h3_token_tags,
)

_H3_STEP_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "train" / "h3_step.py"
)

# The Phase-10 reference corpus, identical to tests/test_h3_geometry.py so the two files price the
# same domain. Kept here as data, never as a derived row count.
CHARACTER_REFS = [(832, 1248, "A"), (2048, 2048, "B"), (1024, 1536, "C")]
ENVIRONMENT_REFS = [
    (1344, 768, "029"),
    (1024, 1024, "000"),
    (1440, 800, "008"),
    (1344, 768, "023"),
]
CAMPAIGN_ASPECT = (16.0, 9.0)

# Small synthetic geometry for the shape/tag/timestep contracts (the plan's suggested sizes).
N_TEXT = 40
N_COND_VIDEO = 12
N_TARGET_VIDEO = 20
N_AUDIO = 8

# Deliberately tiny feature widths — the contracts under test are about ROWS, not channels.
PATCH_DIM = 6
AUDIO_IN_CHANNELS = 4
TEXT_DIM = 5

T_VIDEO = 0.5
T_AUDIO = 0.4


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments and triple-quoted blocks (tests/test_no_warm_gpu.py discipline)."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _make_batch(
    *,
    n_text: int = N_TEXT,
    n_cond_video: int = N_COND_VIDEO,
    n_target_video: int = N_TARGET_VIDEO,
    n_cond_audio: int = 0,
    n_target_audio: int = N_AUDIO,
    vision_spans: object = (),
    t_video: float = T_VIDEO,
    t_audio: float = T_AUDIO,
    audio_in_loss: bool = False,
    patch_dim: int = PATCH_DIM,
    audio_in_channels: int = AUDIO_IN_CHANNELS,
    text_dim: int = TEXT_DIM,
    max_packed_rows: int | None = None,
    t_visual_cond: float | None = None,
    position_ids: torch.Tensor | None = None,
    expected_layout: object | None = None,
) -> H3PackedBatch:
    """Build one small CPU packed batch, defaulting to the plan's synthetic geometry."""
    n_video = n_cond_video + n_target_video
    n_audio = n_cond_audio + n_target_audio
    seq = n_text + n_video + n_audio

    generator = torch.Generator().manual_seed(42)
    video_latents = torch.randn(1, n_video, patch_dim, generator=generator)
    audio_latents = torch.randn(1, n_audio, audio_in_channels, generator=generator)
    text_embeds = torch.randn(1, n_text, text_dim, generator=generator)
    if position_ids is None:
        position_ids = torch.zeros(seq, 3, dtype=torch.float32)
        position_ids[:, 0] = torch.arange(seq, dtype=torch.float32)

    kwargs: dict[str, object] = {}
    if t_visual_cond is not None:
        kwargs["t_visual_cond"] = t_visual_cond
    if expected_layout is not None:
        kwargs["expected_layout"] = expected_layout

    return build_h3_packed_batch(
        video_latents,
        audio_latents,
        text_embeds,
        n_cond_video,
        n_cond_audio,
        position_ids,
        vision_spans,
        t_video,
        t_audio,
        audio_in_loss=audio_in_loss,
        patch_dim=patch_dim,
        audio_in_channels=audio_in_channels,
        text_dim=text_dim,
        max_packed_rows=max_packed_rows,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------------------------------
# 1. Constants — the checkpoint contract
# --------------------------------------------------------------------------------------------------


def test_modality_tags_are_the_checkpoint_contract_values() -> None:
    """``0 = video, 1 = text, 2 = audio`` — stated three independent times (P10-0e section 1)."""
    assert (H3_VIDEO_TAG, H3_TEXT_TAG, H3_AUDIO_TAG) == (0, 1, 2)
    assert H3_MODALITY_NUM == 3


def test_modality_num_is_the_loader_constant_not_a_restatement() -> None:
    """H3-01: ``models/h3_loader`` is the SINGLE source of every H3 architecture number."""
    assert H3_MODALITY_NUM is EXPECTED_H3_MODALITY_NUM or H3_MODALITY_NUM == EXPECTED_H3_MODALITY_NUM
    src = _strip_comments_and_docstrings(_H3_STEP_SRC.read_text(encoding="utf-8"))
    assert "EXPECTED_H3_MODALITY_NUM" in src, (
        "H3_MODALITY_NUM must be re-exported from models/h3_loader, never restated as a literal"
    )


def test_the_two_condition_pins_carry_their_measured_values() -> None:
    """0.999 (visual, ``modular_pipeline.py:287-295``) and exactly 1.0 (audio, never noised)."""
    assert H3_VISUAL_CONDITION_PIN == 0.999
    assert H3_AUDIO_CONDITION_PIN == 1.0


def test_the_two_sigma_shifts_are_named_constants() -> None:
    """Video shift 12.0 / audio shift 3.0 — triple-confirmed (ai-toolkit, ComfyUI, diffusers)."""
    assert H3_VIDEO_SIGMA_SHIFT == 12.0
    assert H3_AUDIO_SIGMA_SHIFT == 3.0


def test_patch_dim_derives_from_the_loader_constants() -> None:
    """``patch_dim = in_channels * prod(patch_size)`` = 24 * 1 * 2 * 2 = 96 (never a literal)."""
    t, h, w = EXPECTED_H3_PATCH_SIZE
    assert EXPECTED_H3_IN_CHANNELS * t * h * w == 96


# --------------------------------------------------------------------------------------------------
# 2. Segment offsets + indices (scripts/_h3_probe_modal.py:488-499)
# --------------------------------------------------------------------------------------------------


def test_segment_offsets_reproduce_the_probe_arithmetic() -> None:
    cond_start, audio_start, video_start, seq = h3_segment_offsets(
        N_TEXT, N_COND_VIDEO, N_AUDIO, N_TARGET_VIDEO
    )
    assert cond_start == N_TEXT
    assert audio_start == N_TEXT + N_COND_VIDEO
    assert video_start == N_TEXT + N_COND_VIDEO + N_AUDIO
    assert seq == N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO


def test_segment_offsets_reject_negative_counts() -> None:
    with pytest.raises(ValueError, match="n_cond_video"):
        h3_segment_offsets(N_TEXT, -1, N_AUDIO, N_TARGET_VIDEO)


def test_indices_partition_the_sequence_exactly_once_each() -> None:
    """The load-bearing invariant: sort(cat(all three)) == arange(seq)."""
    video_indices, audio_indices, text_indices = h3_indices(
        N_TEXT, N_COND_VIDEO, N_AUDIO, N_TARGET_VIDEO
    )
    seq = N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO
    union = torch.sort(torch.cat([video_indices, audio_indices, text_indices])).values
    assert torch.equal(union, torch.arange(seq))


def test_video_indices_are_reference_rows_then_target_rows() -> None:
    video_indices, _, _ = h3_indices(N_TEXT, N_COND_VIDEO, N_AUDIO, N_TARGET_VIDEO)
    cond_start, _, video_start, seq = h3_segment_offsets(
        N_TEXT, N_COND_VIDEO, N_AUDIO, N_TARGET_VIDEO
    )
    assert torch.equal(
        video_indices[:N_COND_VIDEO], torch.arange(cond_start, cond_start + N_COND_VIDEO)
    )
    assert torch.equal(video_indices[N_COND_VIDEO:], torch.arange(video_start, seq))


def test_index_tensors_are_long_dtype() -> None:
    for tensor in h3_indices(N_TEXT, N_COND_VIDEO, N_AUDIO, N_TARGET_VIDEO):
        assert tensor.dtype == torch.long


# --------------------------------------------------------------------------------------------------
# 3. Modality tags — including the counter-intuitive vision-rows-tagged-video case
# --------------------------------------------------------------------------------------------------


def test_token_tags_take_only_the_three_contract_values() -> None:
    video_indices, audio_indices, _ = h3_indices(N_TEXT, N_COND_VIDEO, N_AUDIO, N_TARGET_VIDEO)
    seq = N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO
    tags = h3_token_tags(seq, video_indices, audio_indices, ())
    assert set(tags.tolist()) <= {H3_VIDEO_TAG, H3_TEXT_TAG, H3_AUDIO_TAG}
    assert torch.all(tags[video_indices] == H3_VIDEO_TAG)
    assert torch.all(tags[audio_indices] == H3_AUDIO_TAG)
    assert torch.all(tags[:N_TEXT] == H3_TEXT_TAG)


def test_vision_spans_are_tagged_video_including_their_sentinels() -> None:
    """P10-0e section 1: the ``<|vision_start|>``/``<|vision_end|>`` sentinels are tag 0 TOO."""
    video_indices, audio_indices, _ = h3_indices(N_TEXT, N_COND_VIDEO, N_AUDIO, N_TARGET_VIDEO)
    seq = N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO
    spans = [(6, 18), (22, 30)]
    tags = h3_token_tags(seq, video_indices, audio_indices, spans)

    for start, stop in spans:
        assert torch.all(tags[start:stop] == H3_VIDEO_TAG), (
            f"vision span [{start}, {stop}) must be tagged video (0), sentinels included"
        )
    # everything else in the text span stays text
    assert torch.all(tags[0:6] == H3_TEXT_TAG)
    assert torch.all(tags[18:22] == H3_TEXT_TAG)
    assert torch.all(tags[30:N_TEXT] == H3_TEXT_TAG)


def test_vision_span_outside_the_text_span_raises() -> None:
    """A span that reaches into the conditioning/audio rows would corrupt real modality tags."""
    video_indices, audio_indices, _ = h3_indices(N_TEXT, N_COND_VIDEO, N_AUDIO, N_TARGET_VIDEO)
    seq = N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO
    with pytest.raises(ValueError, match="text span"):
        h3_token_tags(seq, video_indices, audio_indices, [(N_TEXT - 2, N_TEXT + 4)])


def test_vision_span_bounds_are_validated() -> None:
    video_indices, audio_indices, _ = h3_indices(N_TEXT, N_COND_VIDEO, N_AUDIO, N_TARGET_VIDEO)
    seq = N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO
    with pytest.raises(ValueError, match="vision span"):
        h3_token_tags(seq, video_indices, audio_indices, [(10, 4)])


def test_batch_token_tags_cover_the_vision_span() -> None:
    batch = _make_batch(vision_spans=[(4, 16)])
    tags = batch.kwargs["token_tags"]
    assert torch.all(tags[4:16] == H3_VIDEO_TAG)
    assert torch.all(tags[0:4] == H3_TEXT_TAG)


# --------------------------------------------------------------------------------------------------
# 4. Row timesteps (before_denoise.py:1228-1244, probe lines 509-517)
# --------------------------------------------------------------------------------------------------


def test_text_rows_inherit_the_video_timestep() -> None:
    batch = _make_batch()
    row_t = batch.kwargs["timestep"][batch.kwargs["timestep_indices"]]
    assert torch.allclose(row_t[:N_TEXT], torch.full((N_TEXT,), T_VIDEO))


def test_visual_conditioning_rows_pin_at_max_of_t_and_the_pin() -> None:
    batch = _make_batch()
    row_t = batch.kwargs["timestep"][batch.kwargs["timestep_indices"]]
    video_indices = batch.kwargs["video_indices"]
    cond_rows = row_t[video_indices[:N_COND_VIDEO]]
    assert torch.allclose(cond_rows, torch.full_like(cond_rows, H3_VISUAL_CONDITION_PIN))
    target_rows = row_t[video_indices[N_COND_VIDEO:]]
    assert torch.allclose(target_rows, torch.full_like(target_rows, T_VIDEO))


def test_the_visual_pin_is_a_max_not_a_constant() -> None:
    """With ``t_video`` above the pin the code follows the SCHEDULE up (P10-0d section 3.5)."""
    t_high = 0.9995
    batch = _make_batch(t_video=t_high)
    row_t = batch.kwargs["timestep"][batch.kwargs["timestep_indices"]]
    cond_rows = row_t[batch.kwargs["video_indices"][:N_COND_VIDEO]]
    assert torch.allclose(cond_rows, torch.full_like(cond_rows, t_high))


def test_t_visual_cond_is_parameterized_for_training() -> None:
    """D-10-REFPIN: 0.999 is an INFERENCE convention — training passes its own value."""
    batch = _make_batch(t_visual_cond=0.97)
    row_t = batch.kwargs["timestep"][batch.kwargs["timestep_indices"]]
    cond_rows = row_t[batch.kwargs["video_indices"][:N_COND_VIDEO]]
    assert torch.allclose(cond_rows, torch.full_like(cond_rows, 0.97))


def test_reference_audio_pins_at_exactly_one_and_target_audio_keeps_its_own_t() -> None:
    n_cond_audio = 3
    batch = _make_batch(n_cond_audio=n_cond_audio, n_target_audio=5)
    row_t = batch.kwargs["timestep"][batch.kwargs["timestep_indices"]]
    audio_indices = batch.kwargs["audio_indices"]
    ref_audio = row_t[audio_indices[:n_cond_audio]]
    tgt_audio = row_t[audio_indices[n_cond_audio:]]
    assert torch.allclose(ref_audio, torch.full_like(ref_audio, H3_AUDIO_CONDITION_PIN))
    assert torch.allclose(tgt_audio, torch.full_like(tgt_audio, T_AUDIO))


def test_timestep_is_sorted_unique_and_round_trips_through_timestep_indices() -> None:
    batch = _make_batch(n_cond_audio=3, n_target_audio=5)
    timestep = batch.kwargs["timestep"]
    timestep_indices = batch.kwargs["timestep_indices"]

    assert torch.equal(timestep, torch.sort(timestep).values)
    assert timestep.numel() == torch.unique(timestep).numel()
    assert timestep_indices.shape == (batch.seq_len,)
    assert timestep_indices.dtype == torch.long

    video_indices = batch.kwargs["video_indices"]
    audio_indices = batch.kwargs["audio_indices"]
    row_t, rebuilt_timestep, rebuilt_indices = h3_row_timesteps(
        batch.seq_len,
        video_indices,
        audio_indices,
        batch.n_cond_video,
        batch.n_cond_audio,
        T_VIDEO,
        T_AUDIO,
    )
    assert torch.equal(rebuilt_timestep, timestep)
    assert torch.equal(rebuilt_indices, timestep_indices)
    assert torch.allclose(timestep[timestep_indices], row_t)


# --------------------------------------------------------------------------------------------------
# 5. Loss masking (transformer_minimax_h3.py L44-50 + D-10-AUDIO)
# --------------------------------------------------------------------------------------------------


def test_loss_mask_drops_exactly_the_conditioning_rows_and_nothing_else() -> None:
    video_mask, _ = h3_loss_mask(N_COND_VIDEO, N_TARGET_VIDEO, 0, N_AUDIO, audio_in_loss=False)
    assert video_mask.shape == (1, N_COND_VIDEO + N_TARGET_VIDEO)
    assert video_mask.dtype == torch.bool
    assert not video_mask[0, :N_COND_VIDEO].any()
    assert video_mask[0, N_COND_VIDEO:].all()
    assert int(video_mask.sum()) == N_TARGET_VIDEO


def test_audio_mask_is_all_false_when_audio_is_not_a_target() -> None:
    _, audio_mask = h3_loss_mask(N_COND_VIDEO, N_TARGET_VIDEO, 2, 6, audio_in_loss=False)
    assert audio_mask.shape == (1, 8)
    assert audio_mask.dtype == torch.bool
    assert not audio_mask.any()


def test_audio_mask_drops_only_reference_audio_when_audio_is_a_target() -> None:
    _, audio_mask = h3_loss_mask(N_COND_VIDEO, N_TARGET_VIDEO, 2, 6, audio_in_loss=True)
    assert not audio_mask[0, :2].any()
    assert audio_mask[0, 2:].all()


def test_batch_exposes_the_two_masks() -> None:
    batch = _make_batch(n_cond_audio=2, n_target_audio=6)
    assert batch.video_loss_mask.shape == (1, N_COND_VIDEO + N_TARGET_VIDEO)
    assert batch.audio_loss_mask.shape == (1, 8)
    assert not batch.audio_loss_mask.any()


def test_target_audio_rows_are_present_and_noised_with_audio_in_loss_false() -> None:
    """D-10-AUDIO: not-targeting audio is NOT training silence — the rows stay in the sequence."""
    batch = _make_batch(audio_in_loss=False)
    audio = batch.kwargs["audio_hidden_states"]
    assert audio.shape == (1, N_AUDIO, AUDIO_IN_CHANNELS)
    assert torch.count_nonzero(audio) > 0
    assert batch.kwargs["audio_indices"].numel() == N_AUDIO


# --------------------------------------------------------------------------------------------------
# 6. The ten-kwarg packed batch
# --------------------------------------------------------------------------------------------------


def test_kwargs_keys_are_exactly_the_ten_measured_keys() -> None:
    batch = _make_batch()
    expected = {
        "hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "timestep",
        "timestep_indices",
        "token_tags",
        "position_ids",
        "video_indices",
        "audio_indices",
        "text_indices",
    }
    assert set(batch.kwargs) == expected
    assert set(H3_PACKED_BATCH_KEYS) == expected
    assert len(H3_PACKED_BATCH_KEYS) == 10


def test_hidden_states_shape_is_video_rows_by_patch_dim() -> None:
    batch = _make_batch()
    assert batch.kwargs["hidden_states"].shape == (1, N_COND_VIDEO + N_TARGET_VIDEO, PATCH_DIM)
    assert batch.kwargs["encoder_hidden_states"].shape == (1, N_TEXT, TEXT_DIM)
    assert batch.kwargs["audio_hidden_states"].shape == (1, N_AUDIO, AUDIO_IN_CHANNELS)


def test_hidden_states_is_reference_rows_then_target_rows() -> None:
    """The caller may hand the two halves separately; the builder concatenates ref-first."""
    cond = torch.full((1, N_COND_VIDEO, PATCH_DIM), 1.0)
    target = torch.full((1, N_TARGET_VIDEO, PATCH_DIM), 2.0)
    seq = N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO
    position_ids = torch.zeros(seq, 3)
    batch = build_h3_packed_batch(
        (cond, target),
        torch.randn(1, N_AUDIO, AUDIO_IN_CHANNELS),
        torch.randn(1, N_TEXT, TEXT_DIM),
        N_COND_VIDEO,
        0,
        position_ids,
        (),
        T_VIDEO,
        T_AUDIO,
        patch_dim=PATCH_DIM,
        audio_in_channels=AUDIO_IN_CHANNELS,
        text_dim=TEXT_DIM,
    )
    hidden = batch.kwargs["hidden_states"]
    assert torch.all(hidden[0, :N_COND_VIDEO] == 1.0)
    assert torch.all(hidden[0, N_COND_VIDEO:] == 2.0)


def test_batch_row_counts_are_reported() -> None:
    batch = _make_batch(n_cond_audio=2, n_target_audio=6)
    assert batch.n_text == N_TEXT
    assert batch.n_cond_video == N_COND_VIDEO
    assert batch.n_target_video == N_TARGET_VIDEO
    assert batch.n_cond_audio == 2
    assert batch.n_target_audio == 6
    assert batch.seq_len == N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + 8


def test_position_ids_pass_through_untouched() -> None:
    seq = N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO
    position_ids = torch.rand(seq, 3)
    batch = _make_batch(position_ids=position_ids)
    assert torch.equal(batch.kwargs["position_ids"], position_ids)


def test_indices_from_the_batch_partition_the_sequence() -> None:
    batch = _make_batch(n_cond_audio=3, n_target_audio=5)
    union = torch.sort(
        torch.cat(
            [
                batch.kwargs["video_indices"],
                batch.kwargs["audio_indices"],
                batch.kwargs["text_indices"],
            ]
        )
    ).values
    assert torch.equal(union, torch.arange(batch.seq_len))


# --------------------------------------------------------------------------------------------------
# 7. Shape / count validation — the fail-fast surface
# --------------------------------------------------------------------------------------------------


def _build_raw(
    *,
    n_video: int,
    n_audio: int,
    n_cond_video: int,
    n_cond_audio: int,
) -> H3PackedBatch:
    """Call the builder with row counts that need NOT agree with the tensors (the failure path)."""
    seq = N_TEXT + n_video + n_audio
    return build_h3_packed_batch(
        torch.randn(1, n_video, PATCH_DIM),
        torch.randn(1, n_audio, AUDIO_IN_CHANNELS),
        torch.randn(1, N_TEXT, TEXT_DIM),
        n_cond_video,
        n_cond_audio,
        torch.zeros(seq, 3),
        (),
        T_VIDEO,
        T_AUDIO,
        patch_dim=PATCH_DIM,
        audio_in_channels=AUDIO_IN_CHANNELS,
        text_dim=TEXT_DIM,
    )


def test_n_cond_video_over_the_row_count_raises() -> None:
    with pytest.raises(ValueError) as excinfo:
        _build_raw(n_video=32, n_audio=N_AUDIO, n_cond_video=33, n_cond_audio=0)
    message = str(excinfo.value)
    assert "n_cond_video" in message
    assert "33" in message and "32" in message


def test_negative_n_cond_video_raises() -> None:
    with pytest.raises(ValueError, match="n_cond_video"):
        _build_raw(n_video=32, n_audio=N_AUDIO, n_cond_video=-1, n_cond_audio=0)


def test_n_cond_audio_over_the_row_count_raises() -> None:
    with pytest.raises(ValueError, match="n_cond_audio"):
        _build_raw(n_video=32, n_audio=N_AUDIO, n_cond_video=12, n_cond_audio=99)


def test_patch_dim_mismatch_names_both_sides() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_h3_packed_batch(
            torch.randn(1, N_COND_VIDEO + N_TARGET_VIDEO, PATCH_DIM + 1),
            torch.randn(1, N_AUDIO, AUDIO_IN_CHANNELS),
            torch.randn(1, N_TEXT, TEXT_DIM),
            N_COND_VIDEO,
            0,
            torch.zeros(N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO, 3),
            (),
            T_VIDEO,
            T_AUDIO,
            patch_dim=PATCH_DIM,
            audio_in_channels=AUDIO_IN_CHANNELS,
            text_dim=TEXT_DIM,
        )
    message = str(excinfo.value)
    assert str(PATCH_DIM) in message and str(PATCH_DIM + 1) in message


def test_text_dim_mismatch_names_both_sides() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_h3_packed_batch(
            torch.randn(1, N_COND_VIDEO + N_TARGET_VIDEO, PATCH_DIM),
            torch.randn(1, N_AUDIO, AUDIO_IN_CHANNELS),
            torch.randn(1, N_TEXT, TEXT_DIM + 2),
            N_COND_VIDEO,
            0,
            torch.zeros(N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO, 3),
            (),
            T_VIDEO,
            T_AUDIO,
            patch_dim=PATCH_DIM,
            audio_in_channels=AUDIO_IN_CHANNELS,
            text_dim=TEXT_DIM,
        )
    message = str(excinfo.value)
    assert str(TEXT_DIM) in message and str(TEXT_DIM + 2) in message


def test_audio_channel_mismatch_names_both_sides() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_h3_packed_batch(
            torch.randn(1, N_COND_VIDEO + N_TARGET_VIDEO, PATCH_DIM),
            torch.randn(1, N_AUDIO, AUDIO_IN_CHANNELS + 3),
            torch.randn(1, N_TEXT, TEXT_DIM),
            N_COND_VIDEO,
            0,
            torch.zeros(N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO, 3),
            (),
            T_VIDEO,
            T_AUDIO,
            patch_dim=PATCH_DIM,
            audio_in_channels=AUDIO_IN_CHANNELS,
            text_dim=TEXT_DIM,
        )
    message = str(excinfo.value)
    assert str(AUDIO_IN_CHANNELS) in message and str(AUDIO_IN_CHANNELS + 3) in message


def test_position_ids_shape_mismatch_names_both_sides() -> None:
    seq = N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO
    with pytest.raises(ValueError) as excinfo:
        _make_batch(position_ids=torch.zeros(seq - 1, 3))
    message = str(excinfo.value)
    assert "position_ids" in message
    assert str(seq) in message and str(seq - 1) in message


def test_batch_dimension_other_than_one_raises() -> None:
    seq = N_TEXT + N_COND_VIDEO + N_TARGET_VIDEO + N_AUDIO
    with pytest.raises(ValueError) as excinfo:
        build_h3_packed_batch(
            torch.randn(2, N_COND_VIDEO + N_TARGET_VIDEO, PATCH_DIM),
            torch.randn(1, N_AUDIO, AUDIO_IN_CHANNELS),
            torch.randn(1, N_TEXT, TEXT_DIM),
            N_COND_VIDEO,
            0,
            torch.zeros(seq, 3),
            (),
            T_VIDEO,
            T_AUDIO,
            patch_dim=PATCH_DIM,
            audio_in_channels=AUDIO_IN_CHANNELS,
            text_dim=TEXT_DIM,
        )
    assert "batch" in str(excinfo.value).lower()


# --------------------------------------------------------------------------------------------------
# 8. The realized-seq_len ceiling assertion (T-10-06-D)
# --------------------------------------------------------------------------------------------------


def _worst_case_layout(short_edge: int) -> object:
    layout, _label = h3_worst_case_packed_seq_len(
        22,
        CAMPAIGN_ASPECT,
        CHARACTER_REFS,
        ENVIRONMENT_REFS,
        H3_NOMINAL_PROMPT_TOKENS,
        short_edge,
    )
    return layout


def _batch_from_layout(layout: object, *, max_packed_rows: int | None) -> H3PackedBatch:
    counts = h3_layout_row_counts(layout)
    return _make_batch(
        n_text=counts["n_text"],
        n_cond_video=counts["n_cond_video"],
        n_target_video=counts["n_target_video"],
        n_cond_audio=counts["n_cond_audio"],
        n_target_audio=counts["n_target_audio"],
        patch_dim=2,
        audio_in_channels=2,
        text_dim=2,
        max_packed_rows=max_packed_rows,
    )


def test_the_phase10_ceiling_and_worst_cases_come_from_the_geometry_module() -> None:
    """Pins the three numbers the ceiling test depends on, derived not hardcoded."""
    ceiling = max_packed_rows_for_budget(
        H3_A100_80GB_USABLE_GIB, H3_RESIDENT_GIB_RANK64, H3_MIB_PER_PACKED_ROW
    )
    assert ceiling == 13777
    assert _worst_case_layout(896).total == 12394  # type: ignore[attr-defined]
    assert _worst_case_layout(1024).total == 14026  # type: ignore[attr-defined]


def test_realized_ceiling_passes_at_the_adopted_short_edge() -> None:
    """896's worst pair (12,394 rows) is UNDER the 13,777 ceiling — the build must succeed."""
    ceiling = max_packed_rows_for_budget(
        H3_A100_80GB_USABLE_GIB, H3_RESIDENT_GIB_RANK64, H3_MIB_PER_PACKED_ROW
    )
    batch = _batch_from_layout(_worst_case_layout(896), max_packed_rows=ceiling)
    assert batch.seq_len == 12394


def test_realized_ceiling_raises_above_the_budget_naming_every_number() -> None:
    """1024's worst pair (14,026 rows) is OVER the ceiling — loud ValueError, never an OOM."""
    ceiling = max_packed_rows_for_budget(
        H3_A100_80GB_USABLE_GIB, H3_RESIDENT_GIB_RANK64, H3_MIB_PER_PACKED_ROW
    )
    with pytest.raises(ValueError) as excinfo:
        _batch_from_layout(_worst_case_layout(1024), max_packed_rows=ceiling)
    message = str(excinfo.value)
    assert "14026" in message, message
    assert "13777" in message, message
    assert str(14026 - 13777) in message, message
    assert re.search(r"reference set", message), message
    assert re.search(r"budget check", message), message


def test_none_disables_the_ceiling_assertion() -> None:
    """The CPU shape tests pass ``None``; only the Modal caller supplies a cfg-derived ceiling."""
    batch = _batch_from_layout(_worst_case_layout(1024), max_packed_rows=None)
    assert batch.seq_len == 14026


def test_layout_row_counts_are_a_pure_reprojection_of_the_geometry_layout() -> None:
    layout = h3_packed_seq_len(
        22, CAMPAIGN_ASPECT, [(832, 1248), (2048, 2048)], H3_NOMINAL_PROMPT_TOKENS, 896
    )
    counts = h3_layout_row_counts(layout)
    assert counts["n_text"] == layout.n_text
    assert counts["n_cond_video"] == layout.n_cond_video
    assert counts["n_target_video"] == layout.n_target_video
    assert counts["n_cond_audio"] == layout.n_cond_audio
    assert counts["n_target_audio"] == layout.n_target_audio
    assert counts["seq_len"] == layout.total
    assert sum(
        counts[k]
        for k in ("n_text", "n_cond_video", "n_target_video", "n_cond_audio", "n_target_audio")
    ) == counts["seq_len"]


def test_expected_layout_cross_check_catches_a_drifted_dataset() -> None:
    """A realized batch that disagrees with the PRICED layout is attributable, not mysterious."""
    layout = h3_packed_seq_len(
        22, CAMPAIGN_ASPECT, [(832, 1248), (2048, 2048)], H3_NOMINAL_PROMPT_TOKENS, 896
    )
    with pytest.raises(ValueError) as excinfo:
        _make_batch(expected_layout=layout)
    assert "expected_layout" in str(excinfo.value) or "priced" in str(excinfo.value)


def test_expected_layout_cross_check_passes_on_a_matching_batch() -> None:
    layout = h3_packed_seq_len(
        22, CAMPAIGN_ASPECT, [(832, 1248), (2048, 2048)], H3_NOMINAL_PROMPT_TOKENS, 896
    )
    counts = h3_layout_row_counts(layout)
    batch = _make_batch(
        n_text=counts["n_text"],
        n_cond_video=counts["n_cond_video"],
        n_target_video=counts["n_target_video"],
        n_cond_audio=counts["n_cond_audio"],
        n_target_audio=counts["n_target_audio"],
        patch_dim=2,
        audio_in_channels=2,
        text_dim=2,
        expected_layout=layout,
    )
    assert batch.seq_len == layout.total


# --------------------------------------------------------------------------------------------------
# 9. Import confinement (Anti-Pattern 6)
# --------------------------------------------------------------------------------------------------


def test_module_scope_imports_no_modal_and_no_diffusers() -> None:
    """The plan's literal acceptance scan, run against the RAW source (docstrings included)."""
    raw = _H3_STEP_SRC.read_text(encoding="utf-8")
    assert not re.search(r"^(import|from)\s+(modal|diffusers)\b", raw, re.MULTILINE)


def test_heavy_loader_import_is_function_local() -> None:
    """``build_h3_step_deps`` is the seam: its ``load_h3_transformer`` import must be INDENTED."""
    code = _strip_comments_and_docstrings(_H3_STEP_SRC.read_text(encoding="utf-8"))
    hits = [
        line
        for line in code.splitlines()
        if "load_h3_transformer" in line and line.lstrip().startswith(("import ", "from "))
    ]
    assert hits, "build_h3_step_deps must import load_h3_transformer"
    for line in hits:
        assert line.startswith((" ", "\t")), (
            f"heavy loader import must be function-local (indented), got: {line!r}"
        )


def test_importing_h3_step_does_not_pull_diffusers_or_modal() -> None:
    """Subprocess proof: importing the module leaves both heavy roots out of ``sys.modules``."""
    script = (
        "import sys\n"
        "import signet_trainer.train.h3_step as m\n"
        "assert 'diffusers' not in sys.modules, 'diffusers leaked'\n"
        "assert 'modal' not in sys.modules, 'modal leaked'\n"
        "print(m.H3_VIDEO_TAG, m.H3_TEXT_TAG, m.H3_AUDIO_TAG, m.H3_MODALITY_NUM,"
        " m.H3_VISUAL_CONDITION_PIN, m.H3_AUDIO_CONDITION_PIN)\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = {"PYTHONPATH": str(repo_root / "src"), "PYTHONIOENCODING": "utf-8"}
    import os

    merged = dict(os.environ)
    merged.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=merged,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0 1 2 3 0.999 1.0", proc.stdout


# --------------------------------------------------------------------------------------------------
# 10. The injected backend seam
# --------------------------------------------------------------------------------------------------


def test_h3_step_deps_is_a_frozen_bundle() -> None:
    from dataclasses import FrozenInstanceError, fields

    from signet_trainer.train.h3_step import H3StepDeps

    class _Cfg:
        in_channels = EXPECTED_H3_IN_CHANNELS
        patch_size = EXPECTED_H3_PATCH_SIZE
        audio_in_channels = EXPECTED_H3_AUDIO_IN_CHANNELS
        text_dim = EXPECTED_H3_TEXT_DIM

    deps = H3StepDeps(
        transformer=object(),
        config=_Cfg(),
        patch_dim=96,
        audio_in_channels=EXPECTED_H3_AUDIO_IN_CHANNELS,
        text_dim=EXPECTED_H3_TEXT_DIM,
    )
    names = {f.name for f in fields(deps)}
    assert {"transformer", "config", "patch_dim", "audio_in_channels", "text_dim"} <= names
    with pytest.raises(FrozenInstanceError):
        deps.patch_dim = 1  # type: ignore[misc]


def test_h3_step_deps_from_model_derives_patch_dim_from_the_live_config() -> None:
    from signet_trainer.train.h3_step import h3_step_deps_from_model

    class _Cfg:
        in_channels = EXPECTED_H3_IN_CHANNELS
        patch_size = EXPECTED_H3_PATCH_SIZE
        audio_in_channels = EXPECTED_H3_AUDIO_IN_CHANNELS
        text_dim = EXPECTED_H3_TEXT_DIM

    class _Model:
        config = _Cfg()

    model = _Model()
    deps = h3_step_deps_from_model(model)
    assert deps.transformer is model
    assert deps.patch_dim == 96
    assert deps.audio_in_channels == EXPECTED_H3_AUDIO_IN_CHANNELS
    assert deps.text_dim == EXPECTED_H3_TEXT_DIM


def test_h3_step_deps_from_model_can_drive_a_real_packed_batch() -> None:
    """The seam's dims flow straight into ``build_h3_packed_batch``'s validation arguments."""
    from signet_trainer.train.h3_step import h3_step_deps_from_model

    class _Cfg:
        in_channels = 2
        patch_size = (1, 1, 1)
        audio_in_channels = 3
        text_dim = 4

    class _Model:
        config = _Cfg()

    deps = h3_step_deps_from_model(_Model())
    batch = _make_batch(
        patch_dim=deps.patch_dim,
        audio_in_channels=deps.audio_in_channels,
        text_dim=deps.text_dim,
    )
    assert batch.kwargs["hidden_states"].shape[2] == deps.patch_dim
    assert batch.kwargs["audio_hidden_states"].shape[2] == deps.audio_in_channels
    assert batch.kwargs["encoder_hidden_states"].shape[2] == deps.text_dim


# --------------------------------------------------------------------------------------------------
# 11. The environment-bearing sample, pinned at the PACKED layer (plan 10-08)
#
# Deliberately independent of ``H3RefStrategy``: the strategy test proves the strategy, and this one
# proves that the packed batch a resolved environment-bearing sample produces really does carry
# EXACTLY 2 reference blocks with the environment last. A regression that broke both at once would
# otherwise be able to hide behind a single shared assertion.
# --------------------------------------------------------------------------------------------------

_REF_ROWS = 5


def _resolved_environment_sample(segment_index: int):
    """Resolve one environment-bearing segment through the 10-08 selection helpers."""
    from signet_trainer.conditioning.h3_ref import (
        H3Reference,
        order_reference_slots,
        resolve_reference_slots,
    )

    characters = [
        H3Reference(path=f"refs/char_{s.lower()}.png", kind="character", subject_id=s,
                    width=1024, height=1536)
        for s in ("A", "B", "C")
    ]
    environment = H3Reference(
        path="refs/env0.png", kind="environment", subject_id="env0", width=1344, height=768
    )
    slots = resolve_reference_slots(segment_index, characters, environment)
    return order_reference_slots(slots)


def test_an_environment_bearing_sample_packs_exactly_two_reference_blocks() -> None:
    for segment in range(88):
        slots = _resolved_environment_sample(segment)
        assert len(slots) == 2, segment
        assert slots[-1].kind == "environment", segment
        assert slots[0].kind == "character", segment

        batch = _make_batch(n_cond_video=len(slots) * _REF_ROWS)
        assert batch.n_cond_video == 2 * _REF_ROWS
        # the conditioning slice is the FIRST 2 blocks of video_indices, and nothing else
        assert int(batch.video_loss_mask.sum()) == batch.n_target_video
        assert not batch.video_loss_mask[0, : batch.n_cond_video].any()


def test_the_default_two_slot_regime_never_yields_three_blocks() -> None:
    """At the DEFAULT slot count, selection yields 2 blocks, environment or not.

    Was ``test_a_three_block_reference_set_is_unreachable_through_the_selection_helpers``. The
    3-slot case is no longer unreachable — it is reachable only by asking for it explicitly via
    ``references_per_sample=3`` (see the identity test below). This test pins the DEFAULT.
    """
    from signet_trainer.conditioning.h3_ref import H3Reference, resolve_reference_slots

    characters = [
        H3Reference(path=f"refs/char_{s.lower()}.png", kind="character", subject_id=s,
                    width=1024, height=1536)
        for s in ("A", "B", "C")
    ]
    environment = H3Reference(
        path="refs/env0.png", kind="environment", subject_id="env0", width=1344, height=768
    )
    counts = {
        len(resolve_reference_slots(i, characters, environment if i % 7 == 0 else None))
        for i in range(88)
    }
    assert counts == {2}


def test_three_of_three_selection_is_the_identity_permutation() -> None:
    """At references_per_sample=3 over a 3-reference pool, selection cannot rotate.

    combinations(range(3), 3) enumerates a SINGLE tuple, so every segment gets the same three
    references in manifest order. This is what makes 3 safe for explicit-manifest sequence tasks:
    order is load-bearing (opening / middle / closing) and the captions name the beats by index,
    so a rotation would contradict text that is baked into the conditioning at PHASE A.
    """
    from signet_trainer.conditioning.h3_ref import H3Reference, resolve_reference_slots

    refs = [
        H3Reference(path=f"control_{tag}/clip.png", kind="prop", subject_id=f"clip__{tag}",
                    width=1344, height=768)
        for tag in ("first", "mid", "last")
    ]
    expected = ["clip__first", "clip__mid", "clip__last"]
    for segment in range(64):
        slots = resolve_reference_slots(segment, refs, None, references_per_sample=3)
        assert [s.subject_id for s in slots] == expected, segment


def test_at_most_one_environment_reference_even_at_three_slots() -> None:
    """The environment cap is a property of SUBSTITUTION, not of the slot count.

    Regression guard (house audit, PR #51, MAJOR-2): the cap used to be computed as
    ``references_per_sample - 1``, which happened to equal 1 only while the count was pinned at 2.
    At 3 it admitted a SECOND environment reference that the call site then dropped silently,
    because only ``environments[0]`` is ever passed through to ``resolve_reference_slots``.

    ``resolve_reference_slots`` itself has NO environment-cap logic — it just takes the single
    ``environment_reference`` argument it is handed. The hardened cap lives one layer up, in
    ``H3RefStrategy._resolve_slots`` (``h3_ref.py:854``), which is what must be driven here. A
    version of this test that only calls the free function (as the original did) is vacuous: it
    cannot fail even if the cap in ``_resolve_slots`` is reverted, because it never reaches that
    line. Mutation-checked: reverting line 854 to ``references_per_sample - 1`` makes this test
    fail.
    """
    from signet_trainer.conditioning.h3_ref import H3Reference, H3RefStrategy

    strategy = H3RefStrategy(references_per_sample=3)
    characters = [
        H3Reference(path=f"refs/char_{s}.png", kind="character", subject_id=s,
                    width=1024, height=1536)
        for s in ("A", "B", "C")
    ]
    environments = [
        H3Reference(path=f"refs/env{i}.png", kind="environment", subject_id=f"env{i}",
                    width=1344, height=768)
        for i in range(2)
    ]
    pool = [(ref, torch.zeros(1, 1, 1)) for ref in (*characters, *environments)]
    with pytest.raises(ValueError, match="environment reference"):
        strategy._resolve_slots(0, pool)


# --------------------------------------------------------------------------------------------------
# 12. The H3 timestep draw (10-11) — H3's EXPONENTIAL shift, never LTX's logit-normal mean
# --------------------------------------------------------------------------------------------------


def test_h3_shifted_sigma_is_the_exponential_reparameterization() -> None:
    """``sigma' = s*sigma / (1 + (s-1)*sigma)`` — ``scheduling_minimax_h3.py`` L157."""
    from signet_trainer.train.h3_step import h3_shifted_sigma

    for shift in (12.0, 3.0, 1.0):
        for sigma in (0.0, 0.25, 0.5, 0.999, 1.0):
            assert h3_shifted_sigma(sigma, shift) == pytest.approx(
                shift * sigma / (1 + (shift - 1) * sigma)
            )
    # The two fixed points the grid relies on: 0 -> 0 and 1 -> 1.
    assert h3_shifted_sigma(0.0, 12.0) == 0.0
    assert h3_shifted_sigma(1.0, 12.0) == pytest.approx(1.0)
    # shift == 1 is the identity.
    assert h3_shifted_sigma(0.37, 1.0) == pytest.approx(0.37)


def test_h3_shifted_sigma_refuses_a_non_positive_shift() -> None:
    from signet_trainer.train.h3_step import h3_shifted_sigma

    with pytest.raises(ValueError, match="shift"):
        h3_shifted_sigma(0.5, 0.0)


def test_the_h3_shift_is_not_the_ltx_logit_normal_mean() -> None:
    """The confusion this helper exists to prevent, written as a failing case.

    ``train/flow_match.py`` uses ITS shift as the MEAN of a logit-normal draw. Feeding H3's 12.0
    into that formulation pins every sample at ~0.999994 — a silent schedule collapse that would
    look like an ordinary run.
    """
    import math

    from signet_trainer.train.h3_step import H3_VIDEO_SIGMA_SHIFT, h3_shifted_sigma

    ltx_style = 1.0 / (1.0 + math.exp(-H3_VIDEO_SIGMA_SHIFT))
    assert ltx_style > 0.99999, "the misuse really does collapse the schedule"
    assert h3_shifted_sigma(0.5, H3_VIDEO_SIGMA_SHIFT) == pytest.approx(12.0 / 13.0)


def test_h3_draw_timesteps_returns_the_inverted_convention_for_both_modalities() -> None:
    """``t = 1 - sigma`` with ``t = 1`` CLEAN, and ONE base draw feeding both shifts."""
    import numpy as np

    from signet_trainer.train.h3_step import (
        H3_AUDIO_SIGMA_SHIFT,
        H3_VIDEO_SIGMA_SHIFT,
        h3_draw_timesteps,
        h3_shifted_sigma,
    )

    for seed in range(25):
        rng = np.random.default_rng(seed)
        t_video, t_audio = h3_draw_timesteps(rng, uniform_prob=0.0)
        assert 0.0 < t_video < 1.0
        assert 0.0 < t_audio < 1.0
        # Recover the shared base sigma from the video leg, re-derive the audio leg from it.
        sigma_video = 1.0 - t_video
        base = sigma_video / (H3_VIDEO_SIGMA_SHIFT - (H3_VIDEO_SIGMA_SHIFT - 1) * sigma_video)
        assert t_audio == pytest.approx(1.0 - h3_shifted_sigma(base, H3_AUDIO_SIGMA_SHIFT))
        # The heavier video shift pushes sigma UP, i.e. t DOWN — the legs are never equal.
        assert t_video < t_audio


def test_h3_draw_timesteps_is_reproducible_from_the_generator() -> None:
    import numpy as np

    from signet_trainer.train.h3_step import h3_draw_timesteps

    drawn = [h3_draw_timesteps(np.random.default_rng(11), uniform_prob=0.0) for _ in range(3)]
    assert drawn[0] == drawn[1] == drawn[2]


def test_the_uniform_fallback_branch_is_reachable() -> None:
    """``uniform_prob`` is the mode-collapse guard; at 1.0 every draw takes that branch."""
    import numpy as np

    from signet_trainer.train.h3_step import h3_draw_timesteps

    always = {h3_draw_timesteps(np.random.default_rng(s), uniform_prob=1.0) for s in range(8)}
    never = {h3_draw_timesteps(np.random.default_rng(s), uniform_prob=0.0) for s in range(8)}
    assert always.isdisjoint(never)
