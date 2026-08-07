"""train.h3_step — the MiniMax-H3 packed-sequence batch + the injected backend seam (H3-05).

This module is a **HARVEST**, not a derivation. Every constant, offset, index expression and
timestep rule below is transcribed from ``scripts/_h3_probe_modal.py::_synthetic_batch``
(lines 477-535), the batch that ran a real forward + backward + optimizer step on a single
A100-80GB during P10-1. **The probe is authoritative on every number here.** Where this file and
the probe disagree, the probe is right and this file is a bug.

Why the H3 packed batch has no LTX analog
-----------------------------------------
``train/step.py`` drives an LTX ``Modality``-based forward. H3 is a **single-stream** DiT that
packs all three modalities into one sequence and addresses each segment by an index tensor::

    [ text (per-ref labels + vision blocks | prompt LAST) | ref video | target audio | target video ]

Four things DIVERGE hard enough that no LTX code transfers:

  1. **Geometry signature** — H3's row counts are multi-modal (``conditioning/h3_geometry.py``),
     not ``compute_seq_len(w, h, f)``.
  2. **Reference timestep pinning** — conditioning rows are held at their noise-augmentation level
     while the generated rows step down their own schedule.
  3. **Modality tags** — a CHECKPOINT CONTRACT (see below), with a counter-intuitive case.
  4. **Loss masking** — the transformer returns conditioning rows UNMASKED and says dropping them
     is the caller's job (``transformer_minimax_h3.py`` L44-50).

⚠ THE MODALITY TAGS ARE A CHECKPOINT CONTRACT, NOT A CONVENTION
---------------------------------------------------------------
``0 = video, 1 = text, 2 = audio``. H3 keeps one set of AdaLN modulation parameters per
``(timestep, modality)`` pair and addresses the table with::

    timestep_indices * H3_MODALITY_NUM + token_tags

so a wrong tag does not raise — it silently modulates the wrong AdaLN rows. The one that catches
trainers: **Qwen3-VL vision rows live inside the TEXT span but modulate as VIDEO (tag 0)**, and the
``<|vision_start|>`` / ``<|vision_end|>`` sentinels are tagged 0 as well, not just the pads
(``P10-0e-DIFFUSERS-H3.md`` section 1). A trainer that tags them 1 "because they sit in the text
stream" trains against the wrong modulation with no error anywhere.

CRITICAL — Anti-Pattern 6, the CONFINED tier
--------------------------------------------
Module scope is ``torch`` + stdlib + two signet siblings that are themselves confined
(``conditioning/h3_geometry`` is stdlib-only, ``models/h3_loader`` is torch + stdlib with its own
heavy import deferred). There is **no** module-scope ``modal`` and **no** module-scope
``diffusers`` import in this file — the heavy backend arrives ONLY through
``build_h3_step_deps``'s function-local import, exactly the ``train/step.py::build_step_deps``
precedent. That is what keeps ``build_h3_packed_batch`` CPU-testable on Windows/CI, and
``tests/test_h3_packed_sequence.py`` enforces it three ways (raw source scan, indentation scan,
subprocess ``sys.modules`` check).

⚠ D-10-REFPIN — 0.999 IS AN INFERENCE CONVENTION
------------------------------------------------
``H3_VISUAL_CONDITION_PIN = 0.999`` comes from ``modular_pipeline.py::keyframe_noise_aug``, which
is the **sampler's** anchor level. What the TRAINING-time reference augmentation should be is a
separate, deliberately parameterized decision: ``h3_row_timesteps`` and ``build_h3_packed_batch``
both take ``t_visual_cond`` explicitly, defaulting to the inference value only so the harvest is
faithful out of the box. **Do not hardcode the inference pin into the training path**, and do NOT
add latent-noise regularization here — per P10-0c the field trains sequence-concat references CLEAN
and regularizes in IMAGE space (jitter, dropout, cross-pair construction). That belongs to 10-08.

Note on ``position_ids``
------------------------
It is an ARGUMENT, never built here. H3 uses 3-axis float64 RoPE with h/w area-normalized to a
fixed 32-unit extent and ``t`` absolute starting at ``text_len``, each image reference consuming
1.0 ``t`` unit, applied partially over 96 of 128 head dims. **Zero LTX code is reusable for it.**
A monotone stand-in is sufficient for CPU shape tests; the real coordinate construction belongs
with the strategy/dataset that knows the reference geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

# Row counts come from the geometry module — this file never re-derives one (10-03 key link).
from signet_trainer.conditioning.h3_geometry import H3PackedLayout

# The SINGLE source of every H3 architecture number (10-01). Re-exported, never restated.
from signet_trainer.models.h3_loader import EXPECTED_H3_MODALITY_NUM

__all__ = [
    "H3_AUDIO_CONDITION_PIN",
    "H3_AUDIO_SIGMA_SHIFT",
    "H3_AUDIO_TAG",
    "H3_MODALITY_NUM",
    "H3_PACKED_BATCH_KEYS",
    "H3_TEXT_TAG",
    "H3_VIDEO_SIGMA_SHIFT",
    "H3_VIDEO_TAG",
    "H3_VISUAL_CONDITION_PIN",
    "H3PackedBatch",
    "H3StepDeps",
    "build_h3_packed_batch",
    "build_h3_step_deps",
    "h3_draw_timesteps",
    "h3_indices",
    "h3_layout_row_counts",
    "h3_loss_mask",
    "h3_row_timesteps",
    "h3_segment_offsets",
    "h3_shifted_sigma",
    "h3_step_deps_from_model",
    "h3_token_tags",
]


# --------------------------------------------------------------------------------------------------
# Constants — every one carries its source citation.
# --------------------------------------------------------------------------------------------------

#: AdaLN modality tags. ⚠ CHECKPOINT CONTRACT, not a convention: the table is addressed
#: ``timestep_indices * H3_MODALITY_NUM + token_tags`` with row layout
#: ``[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]``, so a wrong tag silently modulates the wrong rows
#: and nothing raises. Stated three independent times — ``modular_pipeline.py`` L22-25,
#: ``transformer_minimax_h3.py`` L34-36, ``P10-0e-DIFFUSERS-H3.md`` section 1.
H3_VIDEO_TAG: int = 0
H3_TEXT_TAG: int = 1
H3_AUDIO_TAG: int = 2

#: ``MINIMAX_H3_MODALITY_NUM`` — the AdaLN tag stride. Re-exported from ``models/h3_loader``
#: (measured on live weights), NEVER restated as a literal here.
H3_MODALITY_NUM: int = EXPECTED_H3_MODALITY_NUM

#: The ``t`` a VISUAL conditioning anchor is held at — ``modular_pipeline.py:287-295``
#: (``keyframe_noise_aug``), applied at ``before_denoise.py:948-955`` as
#: ``0.999 * ref_latent + 0.001 * noise``. ⚠ See D-10-REFPIN in the module docstring: this is the
#: INFERENCE convention and the training path takes it as an explicit, overridable argument.
#: ⚠ The time convention is INVERTED from the usual flow-match one: ``t = 1 - sigma``, ``t = 1`` is
#: CLEAN (``scheduling_minimax_h3.py:166-167``).
H3_VISUAL_CONDITION_PIN: float = 0.999

#: Reference AUDIO pins at exactly 1.0 and is **never noised** — only ``condition_latents`` (visual)
#: pass through ``scale_noise``; ``audio_condition_latents`` are concatenated raw
#: (``before_denoise.py:1099-1101``, pin passed at ``:1228-1244``). There is a prose bug at
#: ``before_denoise.py:1026-1028`` claiming ``t = 0``; the CODE passes 1.0 and is authoritative.
H3_AUDIO_CONDITION_PIN: float = 1.0

#: The two sigma schedules. Video 12.0 / audio 3.0, triple-confirmed across ai-toolkit
#: (``packing.py:47-48``), ComfyUI (``sampling_settings``) and the diffusers pipeline
#: (``P10-0a`` section 1, ``P10-0b`` section 5, ``P10-0d`` line 130). They are NOT used by the
#: packing code below — packing takes already-drawn ``t_video`` / ``t_audio`` — but they are named
#: here so the timestep sampler (``train/flow_match.py``) and the strategy import one constant each
#: instead of re-deriving them. They are also why audio rows land on a different ``temb`` row than
#: video rows: a different timestep VALUE, not a modality axis (``P10-0e`` section 2).
H3_VIDEO_SIGMA_SHIFT: float = 12.0
H3_AUDIO_SIGMA_SHIFT: float = 3.0

#: The exact ten kwargs the H3 transformer forward consumes, in probe order
#: (``scripts/_h3_probe_modal.py:524-535``). The set is CLOSED: ``build_h3_packed_batch`` produces
#: these keys and no others, so an extra kwarg cannot silently ride along into a metered forward.
H3_PACKED_BATCH_KEYS: tuple[str, ...] = (
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
)


# --------------------------------------------------------------------------------------------------
# The packed batch
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class H3PackedBatch:
    """One packed H3 training batch: the ten forward kwargs plus the row bookkeeping around them.

    ``kwargs`` is passed straight to the transformer (``model(**batch.kwargs)``). The row counts and
    the two loss masks are carried alongside because the caller needs them AFTER the forward, to
    slice the prediction — the model returns conditioning rows unmasked by contract.
    """

    kwargs: dict[str, Any]
    n_text: int
    n_cond_video: int
    n_target_video: int
    n_cond_audio: int
    n_target_audio: int
    seq_len: int
    video_loss_mask: torch.Tensor
    audio_loss_mask: torch.Tensor


def h3_segment_offsets(
    n_text: int,
    n_cond_video: int,
    n_audio: int,
    n_target_video: int,
) -> tuple[int, int, int, int]:
    """``(cond_start, audio_start, video_start, seq)`` — ``scripts/_h3_probe_modal.py:488-490``.

    The packed order is ``[ text | ref video | audio | target video ]``: reference video rows sit
    immediately after the text span, the whole audio block (reference soundtracks THEN target audio)
    follows, and the target video rows close the sequence.
    """
    for name, value in (
        ("n_text", n_text),
        ("n_cond_video", n_cond_video),
        ("n_audio", n_audio),
        ("n_target_video", n_target_video),
    ):
        if value < 0:
            raise ValueError(f"invalid {name} {value}: row counts must be >= 0.")

    cond_start = n_text
    audio_start = cond_start + n_cond_video
    video_start = audio_start + n_audio
    seq = video_start + n_target_video
    return cond_start, audio_start, video_start, seq


def h3_indices(
    n_text: int,
    n_cond_video: int,
    n_audio: int,
    n_target_video: int,
    *,
    device: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(video_indices, audio_indices, text_indices)`` exactly as the probe builds them.

    ``video_indices`` is **reference rows THEN target rows** — that order is what makes
    ``pred[:, :n_cond_video]`` the conditioning slice everywhere downstream (the loss mask, the
    inference write-mask at ``denoise.py`` L220-237, and the probe's own adapter-moves-model check).

    The three tensors PARTITION ``range(seq)``: every row belongs to exactly one segment, once.
    """
    cond_start, audio_start, video_start, seq = h3_segment_offsets(
        n_text, n_cond_video, n_audio, n_target_video
    )
    video_indices = torch.cat(
        [
            torch.arange(cond_start, cond_start + n_cond_video, device=device, dtype=torch.long),
            torch.arange(video_start, seq, device=device, dtype=torch.long),
        ]
    )
    audio_indices = torch.arange(audio_start, video_start, device=device, dtype=torch.long)
    text_indices = torch.arange(0, n_text, device=device, dtype=torch.long)
    return video_indices, audio_indices, text_indices


def h3_token_tags(
    seq: int,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    vision_spans: Sequence[tuple[int, int]] = (),
) -> torch.Tensor:
    """Per-row modality tags: start all-TEXT, set video then audio by index, then the vision spans.

    ``vision_spans`` are half-open ``(start, stop)`` ranges INSIDE the text span, one per reference
    vision block. Each span is overridden to ``H3_VIDEO_TAG`` **including its
    ``<|vision_start|>``/``<|vision_end|>`` sentinel positions** — that is deliberate, not an
    off-by-one: ``P10-0e-DIFFUSERS-H3.md`` section 1 records that the sentinels are tagged 0 too.
    Build the spans to cover ``[vision_start, ..pads.., vision_end]`` inclusive of both sentinels.

    A span that escapes the text span is refused: setting an audio row to tag 0 would corrupt a real
    modality assignment with no shape error to catch it.
    """
    if seq < 0:
        raise ValueError(f"invalid seq {seq}: must be >= 0.")

    tags = torch.full((seq,), H3_TEXT_TAG, dtype=torch.long, device=video_indices.device)
    tags[video_indices] = H3_VIDEO_TAG
    tags[audio_indices] = H3_AUDIO_TAG

    # The text span is everything before the first non-text row.
    starts = [int(t.min()) for t in (video_indices, audio_indices) if t.numel() > 0]
    text_span_end = min(starts) if starts else seq

    for span in vision_spans:
        start, stop = int(span[0]), int(span[1])
        if start < 0 or stop < start or stop > seq:
            raise ValueError(
                f"invalid vision span ({start}, {stop}): must satisfy "
                f"0 <= start <= stop <= seq ({seq})."
            )
        if stop > text_span_end:
            raise ValueError(
                f"invalid vision span ({start}, {stop}): it escapes the text span "
                f"[0, {text_span_end}). Qwen vision rows live INSIDE the text stream — a span "
                f"reaching into the conditioning/audio rows would overwrite a real modality tag."
            )
        tags[start:stop] = H3_VIDEO_TAG

    return tags


def h3_row_timesteps(
    seq: int,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    n_cond_video: int,
    n_cond_audio: int,
    t_video: float,
    t_audio: float,
    *,
    t_visual_cond: float = H3_VISUAL_CONDITION_PIN,
    t_audio_cond: float = H3_AUDIO_CONDITION_PIN,
    device: Any = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(row_t, timestep, timestep_indices)`` — ``before_denoise.py:1228-1244``, probe ``:509-517``.

    The plan, verbatim:

      * **text rows INHERIT the video timestep** (they are filled by the initial ``full``);
      * target video rows carry ``t_video``;
      * **visual conditioning rows pin at ``max(t_video, t_visual_cond)``** — a ``max``, NOT a
        constant. With ``shift = 12.0`` the video schedule's ``t`` stays below 0.999 in practice, so
        references sit at the pin every step, but the code follows the schedule UP if it ever
        exceeds it. That "in practice" is arithmetic on the shift formula, not a stated guarantee;
      * **reference audio pins at exactly ``t_audio_cond`` (1.0) and is never noised**;
      * target audio rows carry ``t_audio`` (its own schedule, shift 3.0 vs the video's 12.0).

    ``timestep`` / ``timestep_indices`` come from ``torch.unique(row_t, sorted=True,
    return_inverse=True)``, so ``timestep[timestep_indices]`` reconstructs ``row_t`` exactly. The
    transformer indexes its AdaLN table off ``timestep_indices``, which is why the de-duplication
    has to happen here rather than being faked with a per-row embedding.

    ``t_visual_cond`` is EXPLICIT (D-10-REFPIN): 0.999 is the inference convention and the training
    caller is expected to make its own decision. Nothing here hardcodes it.
    """
    row_t = torch.full((seq,), float(t_video), dtype=dtype, device=device)
    if n_cond_video:
        row_t[video_indices[:n_cond_video]] = max(float(t_video), float(t_visual_cond))
    row_t[audio_indices[n_cond_audio:]] = float(t_audio)
    if n_cond_audio:
        row_t[audio_indices[:n_cond_audio]] = float(t_audio_cond)

    timestep, timestep_indices = torch.unique(row_t, sorted=True, return_inverse=True)
    return row_t, timestep, timestep_indices


def h3_shifted_sigma(sigma: float, shift: float) -> float:
    """H3's EXPONENTIAL sigma shift: ``sigma' = s*sigma / (1 + (s-1)*sigma)``.

    Source: ``schedulers/scheduling_minimax_h3.py`` L157 at the pinned ``DIFFUSERS_SHA`` (the
    scheduler's own docstring names it "the exponential shift" at L30).

    ⚠ This is NOT the same kind of "shift" as ``train/flow_match.FlowMatchingSchedule``'s. LTX uses
    its shift as the MEAN of a logit-normal draw (``sigmoid(randn*std + shift)``, ``flow_match.py``
    L94-95), where the value lives in roughly ``[0.95, 2.05]``. H3's is a reparameterization of the
    sigma grid itself and is ``12.0`` / ``3.0``. Feeding 12.0 into the LTX formulation gives
    ``sigmoid(12) ~ 0.999994`` — every sample pinned at one end of the schedule, a silent collapse
    that would look like an ordinary run. The two must never be interchanged.
    """
    if shift <= 0:
        raise ValueError(f"invalid shift {shift}: must be positive.")
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def h3_draw_timesteps(
    rng: Any,
    *,
    uniform_prob: float = 0.0,
    std: float = 1.0,
    video_shift: float = H3_VIDEO_SIGMA_SHIFT,
    audio_shift: float = H3_AUDIO_SIGMA_SHIFT,
) -> tuple[float, float]:
    """``(t_video, t_audio)`` for ONE H3 training sample, in H3's INVERTED convention.

    Two things are combined here, and they come from different places on purpose:

      * the BASE distribution is the house recipe — logit-normal with a ``uniform_prob`` fallback,
        mirroring ``train/flow_match.sample_timesteps`` (which is where the fallback's purpose is
        documented: it prevents mode collapse around the schedule centre). The base is drawn
        UNSHIFTED, i.e. ``sigmoid(randn * std)`` with mean 0, because the shift is applied
        exponentially below rather than folded into the mean.
      * the SHIFTS are H3's own measured constants (12.0 video / 3.0 audio), applied through
        :func:`h3_shifted_sigma`.

    ONE base draw feeds BOTH shifts. That mirrors inference, where the two modality schedules are
    built from the same ``linspace(1, 0, num_inference_steps)`` grid and differ only by their shift
    (``scheduling_minimax_h3.py`` L34, L148-158) — so video and audio stay phase-aligned within a
    sample rather than being two unrelated draws.

    Returns ``t = 1 - sigma`` for each modality: H3's convention where **``t = 1`` is CLEAN**
    (``scheduling_minimax_h3.py`` L22-23). Values are kept inside ``[0.001, 0.999]`` before the
    inversion, so neither endpoint is ever exactly reached.
    """
    import math  # noqa: PLC0415 — stdlib, kept local so the module top stays torch + stdlib-lean

    if rng.random() < uniform_prob:
        sigma = float(rng.uniform(0.001, 0.999))
    else:
        sigma = 1.0 / (1.0 + math.exp(-float(rng.standard_normal()) * std))
    sigma = min(max(sigma, 0.001), 0.999)
    return (
        1.0 - h3_shifted_sigma(sigma, video_shift),
        1.0 - h3_shifted_sigma(sigma, audio_shift),
    )


def h3_loss_mask(
    n_cond_video: int,
    n_target_video: int,
    n_cond_audio: int,
    n_target_audio: int,
    audio_in_loss: bool = False,
    *,
    device: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(video_loss_mask, audio_loss_mask)``, both ``[1, n_rows]`` bool. ``True`` == in the loss.

    The transformer states its own contract: *"Conditioning rows are returned unmasked — masking
    them out before the scheduler step is the caller's job"* (``transformer_minimax_h3.py``
    L44-50). Inference discharges it by slicing ``[num_condition_video_rows:]``; TRAINING must
    discharge it here. Leaving conditioning rows in the loss is the dead-adapter bug: the objective
    is dominated by rows the model is not being asked to generate.

    D-10-AUDIO: ``audio_in_loss=False`` makes the audio mask all-``False`` — but the target audio
    ROWS stay in the packed sequence and stay NOISED. Not targeting audio is NOT the same as
    training on silence; dropping the rows would change the sequence the video rows attend to.
    """
    for name, value in (
        ("n_cond_video", n_cond_video),
        ("n_target_video", n_target_video),
        ("n_cond_audio", n_cond_audio),
        ("n_target_audio", n_target_audio),
    ):
        if value < 0:
            raise ValueError(f"invalid {name} {value}: row counts must be >= 0.")

    n_video = n_cond_video + n_target_video
    n_audio = n_cond_audio + n_target_audio

    video_mask = torch.zeros((1, n_video), dtype=torch.bool, device=device)
    video_mask[0, n_cond_video:] = True

    audio_mask = torch.zeros((1, n_audio), dtype=torch.bool, device=device)
    if audio_in_loss:
        audio_mask[0, n_cond_audio:] = True

    return video_mask, audio_mask


def h3_layout_row_counts(layout: H3PackedLayout) -> dict[str, int]:
    """Re-project a PRICED ``H3PackedLayout`` onto ``build_h3_packed_batch``'s row-count arguments.

    Pure re-projection — zero arithmetic. ``conditioning/h3_geometry.py`` owns the H3 packing math
    and this file never re-derives a row count; this is the adapter between the two.
    """
    return {
        "n_text": layout.n_text,
        "n_cond_video": layout.n_cond_video,
        "n_target_video": layout.n_target_video,
        "n_cond_audio": layout.n_cond_audio,
        "n_target_audio": layout.n_target_audio,
        "seq_len": layout.total,
    }


def _as_row_stack(value: Any, name: str, *, dim_name: str) -> torch.Tensor:
    """Coerce a ``[1, N, D]`` tensor, or a sequence of them to concatenate on the ROW axis.

    Accepting ``(cond, target)`` is what makes ``hidden_states`` literally
    ``[ reference rows | target rows ]`` in that order at the one place the order is decided.
    """
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
        if not parts:
            raise ValueError(f"invalid {name}: an empty sequence has no rows to pack.")
        for i, part in enumerate(parts):
            if not isinstance(part, torch.Tensor):
                raise TypeError(
                    f"invalid {name}[{i}]: expected a torch.Tensor, got {type(part).__name__}."
                )
            if part.ndim != 3:
                raise ValueError(
                    f"invalid {name}[{i}]: expected a 3-D [1, rows, {dim_name}] tensor, got "
                    f"{part.ndim}-D {tuple(part.shape)}."
                )
        tensor = torch.cat(parts, dim=1)
    else:
        raise TypeError(
            f"invalid {name}: expected a torch.Tensor or a sequence of them, got "
            f"{type(value).__name__}."
        )

    if tensor.ndim != 3:
        raise ValueError(
            f"invalid {name}: expected a 3-D [1, rows, {dim_name}] tensor, got "
            f"{tensor.ndim}-D {tuple(tensor.shape)}."
        )
    if tensor.shape[0] != 1:
        raise ValueError(
            f"invalid {name}: the leading batch dimension must be 1 (H3 packs ONE sample per "
            f"sequence), got {tensor.shape[0]} in {tuple(tensor.shape)}."
        )
    return tensor


def _check_feature_dim(tensor: torch.Tensor, expected: int, name: str, dim_name: str) -> None:
    if tensor.shape[2] != expected:
        raise ValueError(
            f"invalid {name}: expected {dim_name} {expected}, got {tensor.shape[2]} "
            f"(tensor shape {tuple(tensor.shape)}). The arch gate "
            f"(models/h3_loader.assert_h3_arch) and the pre-encode must agree on {dim_name}."
        )


def build_h3_packed_batch(
    video_latents: Any,
    audio_latents: Any,
    text_embeds: Any,
    n_cond_video: int,
    n_cond_audio: int,
    position_ids: torch.Tensor,
    vision_spans: Sequence[tuple[int, int]] = (),
    t_video: float = 0.0,
    t_audio: float = 0.0,
    *,
    t_visual_cond: float = H3_VISUAL_CONDITION_PIN,
    t_audio_cond: float = H3_AUDIO_CONDITION_PIN,
    audio_in_loss: bool = False,
    patch_dim: int,
    audio_in_channels: int,
    text_dim: int,
    max_packed_rows: int | None = None,
    expected_layout: H3PackedLayout | None = None,
) -> H3PackedBatch:
    """Assemble the exact ten-kwarg packed batch the H3 transformer forward consumes.

    Args:
        video_latents: ``[1, n_video, patch_dim]`` already ordered ``[ref rows | target rows]``, or
            a ``(cond, target)`` pair concatenated here on dim 1 in that order.
        audio_latents: ``[1, n_audio, audio_in_channels]`` ordered ``[ref audio | target audio]``.
            **Target audio rows must be PRESENT and NOISED even when ``audio_in_loss`` is False.**
        text_embeds: ``[1, n_text, text_dim]`` — the Qwen3-VL presentation, labels and vision blocks
            FIRST and the prompt LAST, verbatim with no chat template (P10-0d section 3.4).
        n_cond_video / n_cond_audio: the reference row counts inside those stacks.
        position_ids: ``[seq, 3]`` ``(t, h, w)`` RoPE coordinates — an ARGUMENT, never built here.
        vision_spans: half-open ``(start, stop)`` ranges of the vision blocks inside the text span,
            sentinels INCLUDED.
        t_video / t_audio: the two drawn timesteps (separate schedules, shift 12.0 / 3.0).
        t_visual_cond: the reference pin, applied as ``max(t_video, t_visual_cond)`` — D-10-REFPIN.
        t_audio_cond: the reference-audio pin. 1.0 == fully clean, never noised.
        audio_in_loss: whether target audio rows are a loss target (D-10-AUDIO).
        patch_dim / audio_in_channels / text_dim: the EXPECTED feature widths, validated against the
            tensors. ``patch_dim == in_channels * prod(patch_size)``; use ``H3StepDeps`` to get them
            from the live model config rather than restating them.
        max_packed_rows: the realized-``seq_len`` ceiling. ``None`` disables the check (CPU shape
            tests); the Modal training caller passes a ``cfg``-derived value.
        expected_layout: optional priced ``H3PackedLayout`` to cross-check the realized row counts
            against. Turns a dataset drift into an attributable error instead of a mystery.

    Returns:
        ``H3PackedBatch`` whose ``kwargs`` has exactly ``H3_PACKED_BATCH_KEYS`` and no other key.
    """
    video = _as_row_stack(video_latents, "video_latents", dim_name="patch_dim")
    audio = _as_row_stack(audio_latents, "audio_latents", dim_name="audio_in_channels")
    text = _as_row_stack(text_embeds, "text_embeds", dim_name="text_dim")

    _check_feature_dim(video, patch_dim, "video_latents", "patch_dim")
    _check_feature_dim(audio, audio_in_channels, "audio_latents", "audio_in_channels")
    _check_feature_dim(text, text_dim, "text_embeds", "text_dim")

    n_video = int(video.shape[1])
    n_audio = int(audio.shape[1])
    n_text = int(text.shape[1])

    if n_cond_video < 0 or n_cond_video > n_video:
        raise ValueError(
            f"invalid n_cond_video {n_cond_video}: video_latents carries {n_video} row(s), so the "
            f"reference prefix must satisfy 0 <= n_cond_video <= {n_video}."
        )
    if n_cond_audio < 0 or n_cond_audio > n_audio:
        raise ValueError(
            f"invalid n_cond_audio {n_cond_audio}: audio_latents carries {n_audio} row(s), so the "
            f"reference prefix must satisfy 0 <= n_cond_audio <= {n_audio}."
        )

    n_target_video = n_video - n_cond_video
    n_target_audio = n_audio - n_cond_audio
    _, _, _, seq = h3_segment_offsets(n_text, n_cond_video, n_audio, n_target_video)

    # ---- the realized-seq_len ceiling (T-10-06-D) ------------------------------------------------
    # Belt to the config-load braces. conditioning/h3_geometry prices the WORST reference pair at
    # config load; this fires on what the dataloader ACTUALLY served, so a dataset that drifted away
    # from the validated reference set surfaces as a loud, attributable error at batch-build time
    # rather than as a CUDA OOM 40 minutes into a metered run.
    if max_packed_rows is not None and seq > max_packed_rows:
        raise ValueError(
            f"realized packed sequence of {seq} rows exceeds the validated ceiling of "
            f"{max_packed_rows} rows (over by {seq - max_packed_rows}). The config-load budget "
            f"check (config/validators.validate_h3_reference_budget) passed, so the reference set "
            f"it enumerated no longer matches what the dataloader is serving — a reference image, "
            f"its short edge, the prompt length or the target frame count has drifted since the "
            f"config was validated. Re-price the CURRENT corpus with "
            f"conditioning/h3_geometry.h3_worst_case_packed_seq_len, or lower "
            f"reference_image_short_edge / target frames. Raised here deliberately: the alternative "
            f"is a CUDA OOM on a metered A100."
        )

    if expected_layout is not None:
        realized = {
            "n_text": n_text,
            "n_cond_video": n_cond_video,
            "n_target_video": n_target_video,
            "n_cond_audio": n_cond_audio,
            "n_target_audio": n_target_audio,
            "seq_len": seq,
        }
        priced = h3_layout_row_counts(expected_layout)
        drift = {k: (priced[k], realized[k]) for k in priced if priced[k] != realized[k]}
        if drift:
            detail = ", ".join(
                f"{k}: priced {priced_v} vs realized {realized_v}"
                for k, (priced_v, realized_v) in sorted(drift.items())
            )
            raise ValueError(
                f"expected_layout mismatch — the realized batch disagrees with the priced "
                f"H3PackedLayout ({detail}). The dataloader is serving a different reference set / "
                f"geometry than the one the budget check enumerated."
            )

    if position_ids.ndim != 2 or position_ids.shape[0] != seq or position_ids.shape[1] != 3:
        raise ValueError(
            f"invalid position_ids: expected shape ({seq}, 3) for a packed sequence of {seq} rows, "
            f"got {tuple(position_ids.shape)}. H3 RoPE is 3-axis (t, h, w), one coordinate per "
            f"packed row — build it for the FULL sequence, not just the video rows."
        )

    device = video.device
    # ⛔ position_ids arrives from `build_h3_ref2va_position_ids`, which builds it CPU-side
    # (`torch.zeros(seq, 3, dtype=float64)`) because the coordinates are pure geometry. EVERY other
    # structural tensor below is built ON `device`, and `MiniMaxH3RotaryPosEmbed.forward` does
    # `position_ids.to(torch.float32)` — a DTYPE cast with no device move — before multiplying by a
    # module buffer. The reference pipeline moves it in `before_denoise`; the training path has no
    # equivalent, so a CPU/CUDA split reaches the first forward, i.e. AFTER the arch gate, the
    # 61.7 GiB load and the LoRA inject. Closed BY CONSTRUCTION rather than by a test: a CPU-only
    # suite structurally cannot observe a device split.
    #
    # ⚠ float64 is PRESERVED. The rope casts to fp32 itself, and the area-normalized h/w grids carry
    # meaningful low bits until it does.
    position_ids = position_ids.to(device)
    video_indices, audio_indices, text_indices = h3_indices(
        n_text, n_cond_video, n_audio, n_target_video, device=device
    )
    token_tags = h3_token_tags(seq, video_indices, audio_indices, vision_spans)
    _, timestep, timestep_indices = h3_row_timesteps(
        seq,
        video_indices,
        audio_indices,
        n_cond_video,
        n_cond_audio,
        t_video,
        t_audio,
        t_visual_cond=t_visual_cond,
        t_audio_cond=t_audio_cond,
        device=device,
    )
    video_loss_mask, audio_loss_mask = h3_loss_mask(
        n_cond_video,
        n_target_video,
        n_cond_audio,
        n_target_audio,
        audio_in_loss,
        device=device,
    )

    kwargs: dict[str, Any] = {
        "hidden_states": video,
        "audio_hidden_states": audio,
        "encoder_hidden_states": text,
        "timestep": timestep,
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }
    # The kwarg set is CLOSED — assert it here so a future edit cannot leak an extra key into a
    # metered forward (the model would raise, but only after the container is already paid for).
    assert set(kwargs) == set(H3_PACKED_BATCH_KEYS), sorted(set(kwargs) ^ set(H3_PACKED_BATCH_KEYS))

    return H3PackedBatch(
        kwargs=kwargs,
        n_text=n_text,
        n_cond_video=n_cond_video,
        n_target_video=n_target_video,
        n_cond_audio=n_cond_audio,
        n_target_audio=n_target_audio,
        seq_len=seq,
        video_loss_mask=video_loss_mask,
        audio_loss_mask=audio_loss_mask,
    )


# --------------------------------------------------------------------------------------------------
# The injectable backend seam (function-local heavy import on Modal, stubbed on CPU tests)
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class H3StepDeps:
    """The loaded H3 backend, bundled so it injects as one unit — ``train/step.py::StepDeps`` shape.

    Deliberately MINIMAL. The probe proved the model can be called directly with the ten-kwarg dict,
    so there is no patchifier / coordinate abstraction here: H3 does not need one, and inventing one
    would be an LTX habit rather than an H3 requirement.

    ``patch_dim`` / ``audio_in_channels`` / ``text_dim`` are carried because they are exactly the
    validation arguments ``build_h3_packed_batch`` needs, derived from the LIVE config so the
    trainer never restates an architecture number.
    """

    transformer: Any
    config: Any
    patch_dim: int
    audio_in_channels: int
    text_dim: int


def _patch_dim_from_config(config: Any) -> int:
    """``in_channels * prod(patch_size)`` — ``scripts/_h3_probe_modal.py:486``. 24*1*2*2 = 96."""
    patch_size = tuple(int(p) for p in config.patch_size)
    if len(patch_size) != 3:
        raise ValueError(
            f"invalid patch_size {patch_size}: H3 patchifies over (t, h, w), expected 3 axes."
        )
    dim = int(config.in_channels)
    for axis in patch_size:
        dim *= axis
    return dim


def h3_step_deps_from_model(model: Any) -> H3StepDeps:
    """Bundle an ALREADY-LOADED transformer. No heavy import, so CPU tests can drive it with a stub.

    Split out from ``build_h3_step_deps`` so a caller that already holds the model (the arch gate
    loads it first, and re-loading 61.7 GiB of weights would be an expensive mistake) reuses it.
    """
    config = model.config
    return H3StepDeps(
        transformer=model,
        config=config,
        patch_dim=_patch_dim_from_config(config),
        audio_in_channels=int(config.audio_in_channels),
        text_dim=int(config.text_dim),
    )


def build_h3_step_deps(
    checkpoint_dir: Any,
    device: str = "cuda",
    dtype: Any = None,
) -> H3StepDeps:
    """Load the real H3 transformer and bundle it — FUNCTION-LOCAL heavy import (Modal-side only).

    Mirrors ``train/step.py::build_step_deps``. ``models/h3_loader`` owns the actual weight load and
    defers its own heavy import to inside ``load_h3_transformer``, so importing THIS module (or
    reading it for a structural scan) never requires the heavy dependency installed. The import is
    kept function-local here too so the seam is visible at the call site rather than implied.
    """
    from signet_trainer.models.h3_loader import load_h3_transformer  # noqa: PLC0415 — Modal-side

    model = load_h3_transformer(checkpoint_dir, device=device, dtype=dtype)
    return h3_step_deps_from_model(model)
