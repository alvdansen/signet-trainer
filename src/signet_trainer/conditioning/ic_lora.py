"""conditioning.ic_lora — ``ICLoraStrategy``: IC-LoRA in-context reference-control training (Phase 7).

The third concrete ``TrainingStrategy`` and a sibling of the validated ``SingleFrameStrategy``
(Phase 5) and ``MultiFrameStrategy`` (Phase 6). It is a verbatim port of the canonical
``VideoToVideoStrategy`` (Lightricks/LTX-2 @ pinned SHA d6053703,
``packages/ltx-trainer/src/ltx_trainer/training_strategies/video_to_video.py``) onto signet's
``ModelInputs``/``Modality``/``StepDeps`` contract (RESEARCH §Code Examples lines 282-355).

The ONLY structural change over the siblings — the five documented moves from 07-RESEARCH Pattern 1:

  (1) Patchify BOTH the target latent and the reference latent. ``ref_seq_len`` / ``target_seq_len``
      are the two patchified lengths; the combined sequence is ``ref_seq_len + target_seq_len`` long.
  (2) The reference prefix is CLEAN (never noised, per-token timestep 0 — it is pure conditioning).
      The target half reuses single_frame's first-frame conditioning mask + ``torch.where`` clean
      substitution + per-token timestep-0 (single_frame.py:142-157), applied to the TARGET tokens only.
  (3) ``video_targets = noise - target_latents`` — the TARGET-ONLY velocity target, length
      ``target_seq_len`` (NOT the combined sequence). This is the contract difference from single/multi.
  (4) ``combined_latents = cat([ref_latents_clean, noisy_target], dim=1)``. RoPE positions are built
      TWICE (once at reference latent dims, once at target latent dims) and concatenated on the
      sequence axis. The ``reference_downscale_factor != 1`` height/width-axis scaling branch is
      present but INERT at the default 1 (D-7-REF11).
  (5) ``video_loss_mask = cat([zeros(B, ref_seq_len, bool), ~target_conditioning_mask], dim=1)`` —
      the reference prefix is NEVER in the loss.

``compute_loss`` slices ``model_output[:, ref_seq_len:, :]`` before the masked MSE (L-3 guard):
loss on the reference prefix trains a dead/leaky adapter with no controllable behaviour. Because
``video_targets`` is target-only, forgetting the slice fails loud on shape (RESEARCH Pattern 2).

CRITICAL — Anti-Pattern 6 / Pitfall 6 (import-confinement):
    Module top imports ``torch`` + stdlib ONLY. It MUST NOT import ``modal``/``ltx_core``/
    ``ltx_trainer`` — the heavy ltx_core symbols inject via ``StepDeps`` from ``train/step.py``
    (reused, never duplicated) so the CPU proof runs Windows/CI-free (zero GPU, zero Modal spend).
    ``numpy`` is imported function-local only for the default-rng fallback.

Pitfall 2 (the 166.9 GiB OOM): latent f/h/w for BOTH halves are derived from the latent tensor's
own ``.shape``, NEVER from pixel metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from signet_trainer.conditioning.strategy import ModelInputs, TrainingStrategy
from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import (
    DEFAULT_FPS,
    LTX_VIDEO_LATENT_CHANNELS,
    StepDeps,
    _meta_float,
)


class ICLoraStrategy(TrainingStrategy):
    """IC-LoRA in-context reference-control training strategy (LTX-2.3 ``video_to_video`` port).

    Concatenates a CLEAN reference-latent prefix to the noised target, builds RoPE positions for
    each half separately and concatenates them on the sequence axis, and computes the loss ONLY on
    the target slice (``model_output[:, ref_seq_len:, :]``). It is the FIRST strategy to populate
    ``ModelInputs.ref_seq_len`` (the split index reserved on the contract at strategy.py:114/123) and
    the FIRST with a third data source (``reference_latents`` — single_frame.py:86 reserves exactly
    this).

    Constructed with primitives (NOT the whole ``SignetConfig``) so both ``training_step`` and the CPU
    unit test can build it easily. The ltx_core seam injects as ``deps`` (a ``StepDeps``).

    Args:
        deps: the injected ltx_core seam (patchifier / RoPE coords / ``Modality`` / shape cls).
        schedule: the ``FlowMatchingSchedule`` sigma sampler (Seam 3 — used, NEVER modified).
        first_frame_conditioning_p: P(condition on the first latent frame of the TARGET) per sample.
            Canonical ``video_to_video`` default is 0.1; signet ships 0.0 (the reference prefix already
            supplies the conditioning). Applied to the target half only — the reference prefix is
            unconditionally clean.
        reference_downscale_factor: RoPE height/width-axis scale for the reference positions when the
            reference latent is a downscaled view of the target (canonical ``ref0.5`` adapters use 2).
            At the default 1 (D-7-REF11) the scaling branch is inert and ``ref``/``target`` share dims.
        device / dtype: compute placement (cuda / bf16 on Modal; cpu / float32 in unit tests).
        fps: fallback frame-rate for the temporal RoPE coord (overridden by sample metadata).
    """

    def __init__(
        self,
        deps: StepDeps,
        schedule: FlowMatchingSchedule,
        *,
        first_frame_conditioning_p: float = 0.0,
        reference_downscale_factor: int = 1,
        device: Any = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.deps = deps
        self.schedule = schedule
        self.first_frame_conditioning_p = first_frame_conditioning_p
        self.reference_downscale_factor = reference_downscale_factor
        self.device = device
        self.dtype = dtype
        self.fps = fps

    def get_data_sources(self) -> Sequence[Any]:
        """The signet nested-sample sources this strategy reads.

        The FIRST strategy with a third source: ``reference_latents`` (in addition to the siblings'
        ``latents`` / ``conditions``). single_frame.py:86 reserves exactly this addition.
        """
        return ["latents", "conditions", "reference_latents"]

    def _unwrap_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Move a latent to (device, dtype) and normalize ``[C, F, H, W]`` -> ``[B=1, C, F, H, W]``."""
        latent = latent.to(self.device, dtype=self.dtype)
        if latent.dim() == 4:  # [C, F, H, W] -> [B=1, C, F, H, W]
            latent = latent.unsqueeze(0)
        return latent

    def _build_video_positions(
        self,
        *,
        frames: int,
        height: int,
        width: int,
        batch_size: int,
        fps: float,
    ) -> torch.Tensor:
        """Build the RoPE pixel coords for ONE latent-dim block (mirrors single_frame.py:159-175).

        RoPE grid dims come from the LATENT tensor's OWN shape (Pitfall 2). Called TWICE by
        ``prepare_training_inputs`` — once at reference dims, once at target dims — and the two
        results are concatenated on the sequence axis (dim=2).
        """
        deps = self.deps
        latent_coords = deps.patchifier.get_patch_grid_bounds(
            output_shape=deps.video_latent_shape_cls(
                frames=frames,
                height=height,
                width=width,
                batch=batch_size,
                channels=LTX_VIDEO_LATENT_CHANNELS,
            ),
            device=self.device,
        )
        pixel_coords = deps.get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=deps.scale_factors,
            causal_fix=True,
        ).to(self.dtype)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps  # temporal coord -> seconds
        return pixel_coords

    def prepare_training_inputs(self, batch: Any, rng: Any = None) -> ModelInputs:
        """Turn a signet ``PrecomputedDataset`` sample into the IC-LoRA ``ModelInputs`` (concat form).

        Ports ``video_to_video.prepare_training_inputs`` (RESEARCH §Code Examples): patchify target +
        reference, noise the TARGET only (reference is clean), build the two conditioning masks,
        concatenate ``[ref, target]`` latents / timesteps / positions / loss-mask, and return a
        ``ModelInputs`` whose ``video_targets`` is TARGET-ONLY and whose ``ref_seq_len`` marks the split.

        Args:
            batch: a signet nested sample carrying ``latent_conditions`` (target), ``ref_latent_conditions``
                (reference), and ``text_conditions``.
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
        ref_latent_conditions = batch["ref_latent_conditions"]
        text_conditions = batch["text_conditions"]

        # ── unwrap + normalize BOTH latents to [B, C, F, H, W] (Pitfall 2) ───────────────────────
        target_latent = self._unwrap_latent(latent_conditions["latents"])
        ref_latent = self._unwrap_latent(ref_latent_conditions["latents"])
        batch_size, _channels, tgt_f, tgt_h, tgt_w = target_latent.shape
        _rb, _rc, ref_f, ref_h, ref_w = ref_latent.shape

        fps = _meta_float(latent_conditions, "fps", self.fps)

        # ── (1) patchify BOTH: [B, C, F, H, W] -> [B, seq_len, C] ────────────────────────────────
        target_latent_patched = deps.patchifier.patchify(target_latent)  # [B, target_seq, C]
        ref_latent_patched = deps.patchifier.patchify(ref_latent)  # [B, ref_seq, C]
        target_seq_len = target_latent_patched.shape[1]
        ref_seq_len = ref_latent_patched.shape[1]

        # ── noise + shifted-logit-normal sigmas (TARGET length; reference is never noised) ────────
        noise_patched = torch.randn_like(target_latent_patched)
        t_np = self.schedule.sample_timesteps(batch_size, target_seq_len, rng)
        sigmas = torch.tensor(t_np, device=device, dtype=dtype)  # [B]

        # ── THE LTX OBJECTIVE on the TARGET: flow-match interpolation + velocity target ───────────
        sigmas_expanded = sigmas.view(-1, 1, 1)  # [B, 1, 1]
        noisy_target = (1 - sigmas_expanded) * target_latent_patched + sigmas_expanded * noise_patched
        # (3) TARGET-ONLY velocity target (length target_seq_len, NOT combined).
        video_target = noise_patched - target_latent_patched

        # ── (2) TARGET-half first-frame conditioning mask (port of single_frame.py:142-157) ──────
        target_timesteps = sigmas.view(-1, 1).expand(-1, target_seq_len)  # [B, target_seq]
        p = self.first_frame_conditioning_p
        first_frame_end_idx = tgt_h * tgt_w
        target_cond_mask = torch.zeros(
            batch_size, target_seq_len, dtype=torch.bool, device=device
        )
        if p > 0 and first_frame_end_idx < target_seq_len:
            per_sample = torch.rand(batch_size, device=device) < p  # per-sample Bernoulli draw
            target_cond_mask[per_sample, :first_frame_end_idx] = True
        # clean latents on conditioned target tokens (they are the flow endpoint / timestep 0)
        noisy_target = torch.where(
            target_cond_mask.unsqueeze(-1), target_latent_patched, noisy_target
        )
        target_timesteps = torch.where(
            target_cond_mask, torch.zeros_like(target_timesteps), target_timesteps
        )

        # ── reference prefix: ALWAYS conditioning (clean, per-token timestep 0) ───────────────────
        ref_timesteps = torch.zeros(batch_size, ref_seq_len, dtype=dtype, device=device)

        # ── (4) concatenate [ref, target] on the sequence axis ───────────────────────────────────
        combined_latents = torch.cat([ref_latent_patched, noisy_target], dim=1)
        combined_timesteps = torch.cat([ref_timesteps, target_timesteps], dim=1)

        # ── positions: build each half at its OWN latent dims, then concat on the sequence axis ───
        ref_positions = self._build_video_positions(
            frames=ref_f, height=ref_h, width=ref_w, batch_size=batch_size, fps=fps
        )
        if self.reference_downscale_factor != 1:  # INERT at the default 1 (D-7-REF11)
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= self.reference_downscale_factor  # height axis
            ref_positions[:, 2, ...] *= self.reference_downscale_factor  # width axis (time unchanged)
        target_positions = self._build_video_positions(
            frames=tgt_f, height=tgt_h, width=tgt_w, batch_size=batch_size, fps=fps
        )
        positions = torch.cat([ref_positions, target_positions], dim=2)  # [B, 3, ref+target, 2]

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
            latent=combined_latents,
            sigma=sigmas,
            timesteps=combined_timesteps,
            positions=positions,
            context=video_prompt_embeds,
            context_mask=additive_mask,
        )

        # ── (5) loss mask: reference prefix all-False; target half = ~first-frame-cond ───────────
        ref_loss_mask = torch.zeros(batch_size, ref_seq_len, dtype=torch.bool, device=device)
        target_loss_mask = ~target_cond_mask
        video_loss_mask = torch.cat([ref_loss_mask, target_loss_mask], dim=1)

        return ModelInputs(
            video=video_modality,
            audio=None,
            video_targets=video_target,  # TARGET-ONLY (length target_seq_len)
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=ref_seq_len,
        )

    def compute_loss(self, model_inputs: ModelInputs, model_output: Any) -> torch.Tensor:
        """Target-only mask-fraction-normalized velocity loss, reduced to a 0-dim scalar (L-3 guard).

        Slices the reference prefix off the model output (``model_output[:, ref_seq_len:, :]``) and
        the loss mask (``video_loss_mask[:, ref_seq_len:]``) BEFORE the masked MSE — loss on the
        reference region trains a dead/leaky adapter (L-3 / RESEARCH Pattern 2). ``video_targets`` is
        already target-only, so a missing slice fails loud on shape. Otherwise identical to the
        siblings: squared error, masked, normalized by the mask fraction -> ``[B,]``, then ``.mean()``
        to a scalar for ``loop.py``'s ``(raw_loss / grad_accum).backward()``.
        """
        target_pred = model_output[:, model_inputs.ref_seq_len :, :]  # DROP the reference prefix (L-3)
        target_mask = model_inputs.video_loss_mask[:, model_inputs.ref_seq_len :]
        loss = (target_pred.float() - model_inputs.video_targets.float()).pow(2)  # [B, tgt, D]
        m = target_mask.unsqueeze(-1).float()  # [B, tgt, 1]
        per_sample = loss.mul(m).mean(dim=[-2, -1]) / m.mean(dim=[-2, -1]).clamp(min=1e-8)  # [B,]
        return per_sample.mean()  # 0-dim scalar
