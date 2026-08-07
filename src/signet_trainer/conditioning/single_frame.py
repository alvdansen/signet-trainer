"""conditioning.single_frame — ``SingleFrameStrategy``: the first concrete ``TrainingStrategy``.

Ports ``ltx_trainer.training_strategies.text_to_video.TextToVideoStrategy`` +
``base_strategy`` (the per-token first-frame conditioning mask, per-token timesteps, clean-latent
substitution, and mask-fraction-normalized loss) onto **signet's** ``ModelInputs``/``Modality``
contract and ``FlowMatchingSchedule`` sigma sampler @ pinned SHA d6053703 (RESEARCH §Port Source).

This is the substrate multi-frame (Phase 6) and IC-LoRA (Phase 7) reuse. The objective math is a
strict generalization of ``train/step.py`` — the ONLY new mechanics over the Phase-3-validated step
are the first-frame mask, the ``torch.where`` clean substitution, per-token timesteps, and the
masked loss. With ``first_frame_conditioning_p=0`` the mask is all-target and the normalized-mean
loss is **byte-equivalent** to ``step.py``'s plain-mean MSE (the Phase-3 path).

CRITICAL — Anti-Pattern 6 / Pitfall 6 (import-confinement):
    Module top imports ``torch`` + stdlib ONLY. It MUST NOT import ``modal``/``ltx_core``/
    ``ltx_trainer`` — the heavy ltx_core symbols are injected via ``StepDeps`` (reused from
    ``train/step.py``, never duplicated) so the CPU proof runs Windows/CI-free (zero GPU, zero
    Modal spend). ``numpy`` is imported function-local only for the default-rng fallback.

Pitfall 2 (the 166.9 GiB OOM): latent f/h/w are derived from ``video_latent.shape``, NEVER from
pixel metadata — passing pixel dims explodes the RoPE grid by the 32×32×8 VAE compression.

Pitfall 1 (non-scalar loss): ``compute_loss`` reduces the ported ``[B,]`` per-sample loss to a
0-dim scalar; ``loop.py`` does ``(raw_loss / grad_accum).backward()`` on a scalar.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from signet_trainer.conditioning.strategy import Modality, ModelInputs, TrainingStrategy
from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import (
    DEFAULT_FPS,
    LTX_VIDEO_LATENT_CHANNELS,
    StepDeps,
    _meta_float,
)


class SingleFrameStrategy(TrainingStrategy):
    """First-frame reference-control training strategy (LTX-2.3 ``text_to_video`` port).

    Constructed with primitives (NOT the whole ``SignetConfig``) so both ``training_step`` (Plan
    05-04) and the CPU unit test can build it easily. The ltx_core seam is injected as ``deps``
    (a ``StepDeps`` from ``train/step.py::build_step_deps`` on Modal, a stub in tests).

    Args:
        deps: the injected ltx_core seam (patchifier / RoPE coords / ``Modality`` / shape cls).
        schedule: the ``FlowMatchingSchedule`` sigma sampler (Seam 3 — used, NEVER modified).
        first_frame_conditioning_p: P(condition on the first latent frame) per sample. Canonical
            LTX default is 0.1; signet ships 1.0 (always-condition) for the strongest anchoring
            proof (D-CONDP). At 0.0 the objective is byte-equivalent to the Phase-3 loss.
        first_frame_conditioning_strength: RESERVED (Phase 6). Inert on this hard-clean
            ``condition_image`` path (structurally strength ≡ 1.0); Phase 6 wires it into the
            ltx_core ``ConditioningItem`` API where ``denoise_mask = 1 - strength`` (D-STRENGTH).
        device / dtype: compute placement (cuda / bf16 on Modal; cpu / float32 in unit tests).
        fps: fallback frame-rate for the temporal RoPE coord (overridden by sample metadata).
    """

    def __init__(
        self,
        deps: StepDeps,
        schedule: FlowMatchingSchedule,
        *,
        first_frame_conditioning_p: float = 0.0,
        first_frame_conditioning_strength: float = 1.0,
        device: Any = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.deps = deps
        self.schedule = schedule
        self.first_frame_conditioning_p = first_frame_conditioning_p
        # Reserved for Phase 6 (multi-frame per-item strength); inert on the Phase-5 hard-clean path.
        self.first_frame_conditioning_strength = first_frame_conditioning_strength
        self.device = device
        self.dtype = dtype
        self.fps = fps

    def get_data_sources(self) -> Sequence[Any]:
        """The signet nested-sample sources this strategy reads (``latent_conditions`` /
        ``text_conditions``). Phase 7 (IC-LoRA) adds a reference-latents source here."""
        return ["latents", "conditions"]

    def prepare_training_inputs(self, batch: Any, rng: Any = None) -> ModelInputs:
        """Turn a signet ``PrecomputedDataset`` sample into the ``ModelInputs`` the transformer eats.

        Ports ``step.py``'s unwrap → patchify → sigma sample → flow-match interpolation → velocity
        target → RoPE coords → ``Modality`` build, plus the four new lines from RESEARCH §Port
        Source: the per-sample first-frame mask, the clean-latent ``torch.where`` substitution, the
        per-token timesteps (0 on conditioning tokens), and ``video_loss_mask = ~cond_mask``.

        Args:
            batch: a signet nested sample ``{"latent_conditions": {...}, "text_conditions": {...}}``.
            rng: a ``numpy.random.Generator`` for the timestep draw. ``None`` -> a fresh default
                generator (function-local numpy import — keeps module top torch + stdlib).
        """
        if rng is None:
            import numpy as np  # noqa: PLC0415 — function-local; module top stays torch + stdlib

            rng = np.random.default_rng()

        deps = self.deps
        device = self.device
        dtype = self.dtype

        latent_conditions = batch["latent_conditions"]
        text_conditions = batch["text_conditions"]

        # ── unwrap + normalize the video latent to [B, C, F, H, W] (Pitfall 2) ──────────────────
        video_latent = latent_conditions["latents"].to(device, dtype=dtype)
        if video_latent.dim() == 4:  # [C, F, H, W] -> [B=1, C, F, H, W]
            video_latent = video_latent.unsqueeze(0)
        batch_size, _channels, lat_f, lat_h, lat_w = video_latent.shape

        # RoPE grid dims come from the LATENT tensor's OWN shape (lat_f/lat_h/lat_w) — NOT a pixel
        # reconstruction. Trusting pixel metadata explodes the grid by the VAE compression → the
        # 166.9 GiB A100 OOM (step.py:174-181, Pitfall 2).
        fps = _meta_float(latent_conditions, "fps", self.fps)

        # ── patchify: [B, C, F, H, W] -> [B, seq_len, C] ────────────────────────────────────────
        video_latent_patched = deps.patchifier.patchify(video_latent)
        video_seq_len = video_latent_patched.shape[1]

        # ── noise + shifted-logit-normal sigmas (uniform_prob from config, via schedule) ─────────
        noise_patched = torch.randn_like(video_latent_patched)
        t_np = self.schedule.sample_timesteps(batch_size, video_seq_len, rng)
        sigmas = torch.tensor(t_np, device=device, dtype=dtype)  # [B]

        # ── THE LTX OBJECTIVE: flow-matching interpolation + velocity target (unchanged) ─────────
        sigmas_expanded = sigmas.view(-1, 1, 1)  # [B, 1, 1]
        noisy_video = (1 - sigmas_expanded) * video_latent_patched + sigmas_expanded * noise_patched
        video_target = noise_patched - video_latent_patched  # target = noise - latents

        # ── per-token timesteps: all tokens carry the sampled sigma (pre-mask) ───────────────────
        video_timesteps = sigmas.view(-1, 1).expand(-1, video_seq_len)  # [B, seq_len]

        # ── the FOUR new lines: first-frame conditioning mask (port of base_strategy) ────────────
        #   first_frame_end_idx = one latent frame of tokens (lat_h * lat_w — LATENT dims). The
        #   video patchifier orders tokens frame-major, so [:first_frame_end_idx] is the first frame.
        p = self.first_frame_conditioning_p
        first_frame_end_idx = lat_h * lat_w
        cond_mask = torch.zeros(batch_size, video_seq_len, dtype=torch.bool, device=device)
        if p > 0 and first_frame_end_idx < video_seq_len:
            per_sample = torch.rand(batch_size, device=device) < p  # per-sample Bernoulli draw
            cond_mask[per_sample, :first_frame_end_idx] = True
        # clean latents on conditioning tokens (they are the flow endpoint / timestep 0)
        noisy_video = torch.where(cond_mask.unsqueeze(-1), video_latent_patched, noisy_video)
        # timestep 0 on conditioning tokens; the sampled sigma everywhere else
        video_timesteps = torch.where(
            cond_mask, torch.zeros_like(video_timesteps), video_timesteps
        )
        video_loss_mask = ~cond_mask  # True == compute loss (target tokens)

        # ── RoPE pixel coords (temporal coord -> seconds) — LATENT dims into the shape cls ───────
        latent_coords = deps.patchifier.get_patch_grid_bounds(
            output_shape=deps.video_latent_shape_cls(
                frames=lat_f,
                height=lat_h,
                width=lat_w,
                batch=batch_size,
                channels=LTX_VIDEO_LATENT_CHANNELS,
            ),
            device=device,
        )
        pixel_coords = deps.get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=deps.scale_factors,
            causal_fix=True,
        ).to(dtype)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

        # ── text context: int attention mask -> additive (−inf) float mask ───────────────────────
        video_prompt_embeds = text_conditions["video_prompt_embeds"].to(device, dtype=dtype)
        prompt_attention_mask = text_conditions["prompt_attention_mask"].to(device)
        if video_prompt_embeds.dim() == 2:
            video_prompt_embeds = video_prompt_embeds.unsqueeze(0)
        if prompt_attention_mask.dim() == 1:
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0)
        additive_mask = torch.zeros_like(prompt_attention_mask, dtype=dtype)
        additive_mask[prompt_attention_mask == 0] = float("-inf")

        # ── build the Modality via the injected seam (real Modality on Modal, stub in tests) ─────
        video_modality = deps.modality_cls(
            enabled=True,
            latent=noisy_video,
            sigma=sigmas,
            timesteps=video_timesteps,
            positions=pixel_coords,
            context=video_prompt_embeds,
            context_mask=additive_mask,
        )

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=video_target,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
        )

    def compute_loss(self, model_inputs: ModelInputs, model_output: Any) -> torch.Tensor:
        """Mask-fraction-normalized velocity loss, reduced to a 0-dim scalar (Pitfall 1).

        Ports ``text_to_video.compute_loss``: squared-error, masked to ``video_loss_mask`` (target
        tokens only — conditioning tokens are masked off), normalized by the mask fraction so the
        loss scale is stable as the conditioned-token count varies → ``[B,]``; then ``.mean()`` to a
        scalar for ``loop.py``'s ``(raw_loss / grad_accum).backward()``.

        Backward-compat: with an all-target mask (``first_frame_conditioning_p=0``) the
        normalized-mean equals the plain-mean → byte-equivalent to the Phase-3 ``step.py`` loss.
        """
        loss = (model_output.float() - model_inputs.video_targets.float()).pow(2)  # [B, seq, D]
        m = model_inputs.video_loss_mask.unsqueeze(-1).float()  # [B, seq, 1]
        per_sample = loss.mul(m).mean(dim=[-2, -1]) / m.mean(dim=[-2, -1]).clamp(min=1e-8)  # [B,]
        return per_sample.mean()  # 0-dim scalar
