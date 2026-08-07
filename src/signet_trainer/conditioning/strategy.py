"""The conditioning/ Strategy seam — locked in Phase 1 (Plan 01-01, D-03).

This module is the spine every later phase builds on. It defines:

* ``Modality``     — the per-modality tensor bundle the LTX-2.3 transformer consumes.
* ``ModelInputs``  — the real training-batch contract (a dataclass wrapping ``Modality`` objects).
* ``TrainingStrategy`` — the ABC Phases 5-7 subclass to add single-frame / multi-frame / IC-LoRA
  reference control WITHOUT touching the loop, loader, or offloader.
* ``compute_seq_len`` — the verified video latent sequence-length helper.

Why a stub and not an import of ``ltx_core``:
    The real contract lives in ``ltx_core.model.transformer.modality.Modality`` and
    ``ltx_trainer.training_strategies.base_strategy.{ModelInputs, TrainingStrategy}``
    (RESEARCH.md Q4). We mirror its field NAMES and SHAPES exactly here so the CPU dry-run
    runs Windows/CI-free (zero GPU, zero Modal spend) and Phases 2-4 can drop in the real
    strategies with zero contract drift.

CRITICAL — Anti-Pattern 6 / Pitfall 4:
    This module imports ONLY stdlib + ``torch`` (for tensor type hints / shapes). It MUST NOT
    import ``modal``, ``ltx_core``, or ``ltx_trainer`` — that would break the transport-agnostic
    dry-run path on Windows.

CONTEXT.md alias names -> real fields (vocabulary bridge for future readers):
    latents            -> ModelInputs.video.latent          [1, seq_len, 128]
    targets            -> ModelInputs.video_targets          [1, seq_len, 128] (velocity = noise - clean)
    conditioning_mask  -> ~ModelInputs.video_loss_mask       [1, seq_len] bool
    timesteps          -> ModelInputs.video.timesteps        [1, seq_len]   (0 for conditioning tokens)
    video_coords       -> ModelInputs.video.positions        [1, 3, seq_len, 2] (RoPE patch bounds)
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

# --------------------------------------------------------------------------------------------------
# Verified LTX-2.3 constants (RESEARCH.md Q3, from ltx_core types.py + base_strategy.py).
# --------------------------------------------------------------------------------------------------

#: SpatioTemporalScaleFactors.default() = (time=8, height=32, width=32).
TIME_SCALE: int = 8
HEIGHT_SCALE: int = 32
WIDTH_SCALE: int = 32

#: Video latent channel count D (from base_strategy._get_video_positions, channels=128).
VIDEO_LATENT_CHANNELS: int = 128

#: RoPE positional dims: 3 for video (frame/height/width), 1 for audio.
VIDEO_N_POS_DIMS: int = 3


def compute_seq_len(width: int, height: int, frames: int) -> int:
    """Video latent token count for raw pixel dims [W, H, F] (patch_size=1 ⇒ one token/cell).

    ``seq_len = (H // 32) * (W // 32) * ((F - 1) // 8 + 1)`` — verified against ltx_core's
    ``VideoLatentShape.downscale`` (RESEARCH.md Q3). Example: [768, 512, 49] -> 2688.

    Note: this helper is the single home for the seq-len math used by both the dry-run synthetic
    batch and the config validators (config/validators.py in Plan 01-02 imports it from here —
    it is NOT duplicated there).
    """
    latent_frames = (frames - 1) // TIME_SCALE + 1
    latent_h = height // HEIGHT_SCALE
    latent_w = width // WIDTH_SCALE
    return latent_frames * latent_h * latent_w


# --------------------------------------------------------------------------------------------------
# The real training-batch contract (RESEARCH.md Q4). Mirrors ltx_core/ltx_trainer field names+shapes.
# --------------------------------------------------------------------------------------------------


@dataclass
class Modality:
    """Per-modality bundle the transformer consumes (mirrors ltx_core ... modality.Modality).

    Shapes (B = batch = 1, T = seq_len, D = 128 for video):
        latent     [B, T, D]              noised target; clean latents for conditioning tokens
        sigma      [B]                    sampled flow-matching sigma
        timesteps  [B, T]                 per-token; 0 for conditioning tokens, sampled sigma else
        positions  [B, n_pos_dims, T, 2]  n_pos_dims=3 video / 1 audio; RoPE patch bounds
        context    [B, n_text_tokens, ctx_dim]  text/prompt embeds
    """

    latent: torch.Tensor
    sigma: torch.Tensor
    timesteps: torch.Tensor
    positions: torch.Tensor
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


@dataclass
class ModelInputs:
    """The real LTX-2.3 training-batch contract (mirrors ltx_trainer base_strategy.ModelInputs).

    This is the spine. It is a dataclass wrapping ``Modality`` objects — NOT the older flat
    ``TrainingBatch{latents, targets, conditioning_mask, timesteps, video_coords}`` from
    CONTEXT.md/ARCHITECTURE.md (those names are aliases; see the module docstring mapping).

    Fields (B = 1, T = seq_len, D = 128):
        video             Modality            the video modality bundle
        audio             Modality | None     None for signet-trainer (video-only)
        video_targets     [B, T, D]           velocity target = noise - clean_latents
        audio_targets     [B, T, D] | None
        video_loss_mask   [B, T] bool         True = compute loss = ~conditioning_mask
        audio_loss_mask   [B, T] bool | None
        ref_seq_len       int | None          IC-LoRA (Phase 7) ONLY: length of the reference prefix
    """

    video: Modality
    audio: Modality | None
    video_targets: torch.Tensor
    audio_targets: torch.Tensor | None
    video_loss_mask: torch.Tensor
    audio_loss_mask: torch.Tensor | None
    ref_seq_len: int | None = None


# --------------------------------------------------------------------------------------------------
# The Strategy seam (ARCHITECTURE.md Pattern 1). Phases 5-7 subclass this; the loop/loader/offloader
# never change.
# --------------------------------------------------------------------------------------------------


class TrainingStrategy(abc.ABC):
    """Abstract seam for a training mode (text-to-video, IC-LoRA video-to-video, keyframe, ...).

    Phase 1 locks the ABC + the ``ModelInputs``/``Modality`` contract only. Concrete strategies
    (single-frame / multi-frame / IC-LoRA reference control) land in Phases 5-7 as subclasses that
    implement these methods WITHOUT touching the training loop, loader, or offloader (D-03).
    """

    @abc.abstractmethod
    def get_data_sources(self) -> Sequence[Any]:
        """Return the data sources (e.g. latent dirs) this strategy reads from.

        Per ARCHITECTURE.md Pattern 1: the strategy declares what it consumes so the loader can
        wire it up; reference-control strategies (Phase 7) add a reference-latents source here.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_training_inputs(self, batch: Any) -> ModelInputs:
        """Turn a raw dataloader ``batch`` into the ``ModelInputs`` the transformer consumes.

        This is where per-token conditioning masks, velocity targets, and (Phase 7) reference-prefix
        concatenation are built. Must return a fully-populated ``ModelInputs``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, model_inputs: ModelInputs, model_output: Any) -> torch.Tensor:
        """Compute the (masked) flow-matching loss given the inputs and the model output.

        Loss is taken only over ``model_inputs.video_loss_mask`` (target tokens), never over
        conditioning tokens.
        """
        raise NotImplementedError
