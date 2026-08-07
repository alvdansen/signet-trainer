"""train.flow_match — FlowMatchingSchedule: the TRAIN-02 LTX-2.3 flow-matching objective.

LTX-2.3 uses rectified flow matching, NOT DDPM. This module carries the objective math
and the shifted logit-normal timestep sampler:

    - shift  = lerp(min_shift, max_shift, clamp((seq_len - min_tokens) / (max_tokens - min_tokens), 0, 1))
    - noisy  = (1 - t) * clean + t * noise          (flow-match interpolation, NOT DDPM add)
    - target = noise - clean                        (LTX velocity, NOT Wan v-prediction)

Ported (near-verbatim) from enochiatron ``scripts/train/train.py::FlowMatchingSchedule``
(lines 312-421). The source ``__init__`` default ``uniform_prob=0.1`` is carried as-is; the
locked recipe value 0.30 is passed from config by ``train/loop.py`` (Plan 03-04), NOT
hardcoded here (PATTERNS.md:56).

CRITICAL — Anti-Pattern 6 (mirror ``data/precomputed.py`` discipline):
    This module imports ONLY ``torch`` + ``numpy`` + stdlib. It MUST NOT import ``modal``,
    ``ltx_core``, or ``ltx_trainer``, so ``tests/test_flow_match.py`` runs on Windows/CI
    with zero GPU and zero heavy deps.
"""

from __future__ import annotations

import numpy as np
import torch

# LTX-2.3 VAE compression factors (mirror enochiatron train.py:132-133) — used only by the
# optional ``compute_sequence_length`` helper; kept here so this module stays self-contained.
LTX_VAE_SPATIAL_COMPRESSION = 32
LTX_VAE_TEMPORAL_COMPRESSION = 8


class FlowMatchingSchedule:
    """Shifted logit-normal timestep sampling for LTX-2.3 flow matching.

    LTX-2.3 uses rectified flow matching, NOT DDPM.

    Key formulas:
        - shift  = lerp(min_shift, max_shift, clamp((seq_len - min_tokens) / (max_tokens - min_tokens), 0, 1))
        - noisy  = (1 - t) * clean + t * noise
        - target = noise - clean (velocity prediction)
    """

    def __init__(
        self,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        std: float = 1.0,
        uniform_prob: float = 0.1,
    ):
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.std = std
        self.uniform_prob = uniform_prob

    def compute_shift(self, seq_len: int) -> float:
        """Compute sequence-length-dependent shift for logit-normal sampling.

        Args:
            seq_len: Number of latent tokens (F * H * W in latent space)

        Returns:
            Shift value lerped between ``min_shift`` and ``max_shift``, clamped to
            ``[min_tokens, max_tokens]``.
        """
        t = max(0.0, min(1.0, (seq_len - self.min_tokens) / (self.max_tokens - self.min_tokens)))
        return self.min_shift + t * (self.max_shift - self.min_shift)

    def sample_timesteps(
        self, batch_size: int, seq_len: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample timesteps using a shifted logit-normal distribution.

        A ``uniform_prob`` fraction is drawn from a uniform fallback (prevents mode collapse
        near the shift center); the remainder is shifted logit-normal.

        Args:
            batch_size: Number of timesteps to sample.
            seq_len: Latent sequence length for the shift computation.
            rng: Numpy random generator for reproducibility (seed 42 in the loop).

        Returns:
            Array of timesteps clipped to ``[0.001, 0.999]``.
        """
        shift = self.compute_shift(seq_len)

        # uniform_prob fraction uses the uniform fallback branch.
        use_uniform = rng.random(batch_size) < self.uniform_prob

        # Logit-normal: N(shift, std^2) through sigmoid.
        normal_samples = rng.standard_normal(batch_size) * self.std + shift
        logit_normal = 1.0 / (1.0 + np.exp(-normal_samples))

        # Uniform fallback samples.
        uniform = rng.uniform(0.001, 0.999, batch_size)

        timesteps = np.where(use_uniform, uniform, logit_normal)
        return timesteps.clip(0.001, 0.999).astype(np.float64)

    @staticmethod
    def compute_noisy_latent(
        clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the noisy latent via flow-matching interpolation.

            noisy = (1 - t) * clean + t * noise

        This is NOT DDPM noise addition — it is a linear interpolation between clean data and
        noise along the probability-flow path. ``t`` may be a scalar or a broadcastable tensor.
        """
        return (1.0 - t) * clean + t * noise

    @staticmethod
    def compute_target(clean: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Compute the velocity-prediction target.

            target = noise - clean

        This is the velocity (derivative of the flow path) — the LTX objective, NOT the
        epsilon/v-prediction used by Wan/DDPM.
        """
        return noise - clean

    def compute_sequence_length(self, height: int, width: int, frames: int) -> int:
        """Compute the latent sequence length used by ``compute_shift``.

        Args:
            height: Pixel height (e.g. 352).
            width: Pixel width (e.g. 768).
            frames: Number of video frames (e.g. 25, 49, 81).

        Returns:
            ``lat_f * lat_h * lat_w`` latent-token count.
        """
        lat_h = height // LTX_VAE_SPATIAL_COMPRESSION
        lat_w = width // LTX_VAE_SPATIAL_COMPRESSION
        lat_f = (frames - 1) // LTX_VAE_TEMPORAL_COMPRESSION + 1
        return lat_f * lat_h * lat_w
