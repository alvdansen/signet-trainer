"""conditioning.mask_ops — shared N-region span + denoise-mask helpers for reference control.

Extracted from ``single_frame.py``'s four hard-coded first-frame mask lines so the validated
Phase-5 ``SingleFrameStrategy`` stays byte-identical (D-6-SIBLING / Pitfall 5) while
``MultiFrameStrategy`` (Phase 6) reuses the SAME primitives generalized to N latent-frame regions
at arbitrary offsets, each with a per-item strength.

The canonical strength semantics mirror ltx_core ``VideoConditionByLatentIndex.apply_to`` @ pinned
SHA d6053703 (RESEARCH §strength): ``denoise_mask[:, start:stop] = 1.0 - strength`` and inference
``timesteps = sigma * denoise_mask`` — training MIRRORS both here (``build_denoise_mask`` /
``per_token_sigma``).

CRITICAL — Anti-Pattern 6 / Pitfall 6 (import-confinement):
    Module top imports ``torch`` + stdlib ONLY. It MUST NOT import ``modal``/``ltx_core``/
    ``ltx_trainer`` so the CPU proof runs Windows/CI-free (zero GPU, zero Modal spend). The
    injected ``rng`` is a ``numpy.random.Generator``; we call its methods without importing numpy
    (its return values are coerced with ``int()``), keeping the module top torch + stdlib.

Reuses the verified LTX-2.3 scale constants from ``conditioning/strategy.py`` (never redefined).
"""

from __future__ import annotations

from typing import Any

import torch

from signet_trainer.conditioning.strategy import TIME_SCALE

__all__ = [
    "latent_frame_token_span",
    "build_denoise_mask",
    "per_token_sigma",
    "sample_conditioning_positions",
    "pixel_frame_for_latent_idx",
]


def latent_frame_token_span(latent_idx: int, lat_h: int, lat_w: int) -> tuple[int, int]:
    """Token ``[start, stop)`` for one latent frame at ``latent_idx`` (patchifier is frame-major).

    The video patchifier orders tokens frame-major, so latent frame ``i`` owns the contiguous span
    ``[i * lat_h * lat_w, (i + 1) * lat_h * lat_w)``. This is the generalization of single_frame's
    hard-coded ``first_frame_end_idx = lat_h * lat_w`` (== the span of latent frame 0).
    """
    per_frame = lat_h * lat_w
    start = latent_idx * per_frame
    return start, start + per_frame


def build_denoise_mask(
    batch: int,
    seq_len: int,
    regions: Any,
    device: Any,
) -> torch.Tensor:
    """Float ``[B, seq_len]`` denoise mask: ``1.0`` everywhere, ``1.0 - strength`` on each region.

    Generalizes single_frame's boolean first-frame ``cond_mask`` to N float regions. Canonical
    semantics (ltx_core ``VideoConditionByLatentIndex.apply_to``): a conditioning region gets
    ``denoise_mask = 1 - strength`` — strength ``1.0`` => ``0.0`` (hard clean, timestep 0),
    partial strength => a value in ``(0, 1)`` (partially noised, still IN the loss).

    Args:
        batch: batch size B.
        seq_len: total video token count T.
        regions: iterable of ``(start, stop, strength)`` — token span + per-item strength.
        device: tensor device.

    Returns:
        ``[B, seq_len]`` float tensor in ``[0, 1]``.
    """
    mask = torch.ones(batch, seq_len, dtype=torch.float32, device=device)
    for start, stop, strength in regions:
        mask[:, start:stop] = 1.0 - float(strength)
    return mask


def per_token_sigma(sampled_sigma: torch.Tensor, denoise_mask: torch.Tensor) -> torch.Tensor:
    """Per-token timestep = ``sampled_sigma * denoise_mask`` (mirror of inference ``sigma * mask``).

    Generalizes single_frame's ``video_timesteps = where(cond, 0, sigma)``: with a strength-1 region
    ``denoise_mask == 0`` so the per-token sigma is a hard ``0`` (byte-equal to the ``where``-to-zero
    form); a partial-strength region scales the sampled sigma by ``1 - strength``.

    Args:
        sampled_sigma: ``[B]`` per-sample flow-matching sigma.
        denoise_mask: ``[B, seq_len]`` float mask from :func:`build_denoise_mask`.

    Returns:
        ``[B, seq_len]`` per-token sigma.
    """
    return sampled_sigma.view(-1, 1) * denoise_mask


def sample_conditioning_positions(lat_frames: int, max_items: int, rng: Any) -> list[int]:
    """Draw K ~ Uniform{1..max_items} DISTINCT latent-frame indices in ``[0, lat_frames - 1]``.

    Implements the D-6-POSITIONS random fallback (canonical ltx has no multi-frame position policy).
    Returns ``[]`` when ``lat_frames == 1`` (single-latent-frame guard — mirrors single_frame's
    ``first_frame_end_idx < seq_len`` guard: a one-frame clip has no non-conditioning region).

    K is clamped to ``lat_frames`` so distinct sampling is always satisfiable.

    Args:
        lat_frames: number of latent frames F.
        max_items: upper bound on the number of conditioning frames.
        rng: a ``numpy.random.Generator``.

    Returns:
        A list of K distinct latent-frame indices (possibly length 0 when ``lat_frames == 1``).
    """
    if lat_frames <= 1:
        return []
    k = int(rng.integers(1, max_items + 1))  # Uniform{1..max_items}
    k = min(k, lat_frames)
    positions = rng.choice(lat_frames, size=k, replace=False)
    return [int(p) for p in positions]


def pixel_frame_for_latent_idx(latent_idx: int) -> int:
    """Latent frame index -> its pixel-frame start (``latent_idx * TIME_SCALE``).

    The latent<->pixel conversion boundary. The result is ALWAYS ``% TIME_SCALE == 0`` by
    construction — this is the SC#1 ``start_frame % 8 == 0`` invariant the strategy asserts: because
    training draws latent-native positions and every latent frame maps to pixel frame
    ``idx * TIME_SCALE``, conditioning starts are inherently %8-aligned.
    """
    return latent_idx * TIME_SCALE
