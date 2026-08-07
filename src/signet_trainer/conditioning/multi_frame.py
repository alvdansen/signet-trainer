"""conditioning.multi_frame — ``MultiFrameStrategy``: N-region reference-control training (Phase 6).

A sibling of ``SingleFrameStrategy`` (D-6-SIBLING) that generalizes the first-frame conditioning
span to N latent-frame regions at arbitrary offsets, each with a per-item strength drawn from a
config range. The mask/timestep primitives are extracted into ``conditioning/mask_ops.py`` so the
validated Phase-5 ``single_frame.py`` stays byte-identical — and this strategy is provably
byte-equivalent to it at ``count=1 / idx=0 / strength=1`` (the equivalence proof in
``tests/test_multi_frame_strategy.py``).

The ONLY math that differs from ``single_frame.py`` is the four mask lines: instead of a boolean
first-frame ``cond_mask`` + ``torch.where`` clean substitution + timestep-0, this uses the canonical
``denoise_mask = 1 - strength`` (ltx_core ``VideoConditionByLatentIndex.apply_to`` @ d6053703),
per-token ``sigma * denoise_mask`` noising, and an EXPLICIT ``video_loss_mask = denoise_mask > 0``.

D-6-LOSSMASK (why the loss mask is carried EXPLICITLY, not reconstructed from ``timesteps == 0``):
    A partial-strength conditioning token (strength ``s`` in ``(0, 1)``) has ``denoise_mask = 1 - s``
    in ``(0, 1)`` and therefore a NON-ZERO per-token sigma — yet it MUST participate in the loss
    (only hard-clean strength-1 tokens are excluded). Reconstructing the mask from ``timesteps == 0``
    (Pitfall 1) would wrongly KEEP partial-strength tokens (their timestep != 0) — correct — but the
    intent is fragile; a strength-1 token is the only excluded case. We carry ``denoise_mask > 0``
    explicitly so "in the loss" == "not hard-clean" is unambiguous and independent of the timestep.

SC#1 (``start_frame % 8 == 0`` asserted IN the strategy): training samples LATENT-native positions
    via ``sample_conditioning_positions``; every latent frame ``idx`` maps to pixel frame
    ``idx * TIME_SCALE`` (``pixel_frame_for_latent_idx``), which is inherently ``% TIME_SCALE == 0``.
    The strategy asserts this at the latent<->pixel boundary so the invariant is explicit even
    though no config-supplied pixel ``frame_index`` ever reaches the training path.

CRITICAL — Anti-Pattern 6 / Pitfall 6 (import-confinement):
    Module top imports ``torch`` + stdlib ONLY (heavy ltx_core symbols inject via ``StepDeps`` from
    ``train/step.py``). ``numpy`` is imported function-local only for the default-rng fallback.

Pitfall 2 (the 166.9 GiB OOM): latent f/h/w are derived from ``video_latent.shape``, NEVER from
pixel metadata.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from signet_trainer.conditioning.mask_ops import (
    build_denoise_mask,
    latent_frame_token_span,
    per_token_sigma,
    pixel_frame_for_latent_idx,
    sample_conditioning_positions,
)
from signet_trainer.conditioning.strategy import (
    TIME_SCALE,
    Modality,
    ModelInputs,
    TrainingStrategy,
)
from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import (
    DEFAULT_FPS,
    LTX_VIDEO_LATENT_CHANNELS,
    StepDeps,
    _meta_float,
)


class MultiFrameStrategy(TrainingStrategy):
    """Multi-frame reference-control training strategy (N latent-frame regions, per-item strength).

    A strict generalization of ``SingleFrameStrategy``: at ``max_conditioning_items=1`` with the
    single sampled position at latent frame 0 and strength 1.0, ``prepare_training_inputs`` is
    byte-equivalent to ``SingleFrameStrategy(first_frame_conditioning_p=1.0)`` (D-6-SIBLING).

    Constructed with primitives (NOT the whole ``SignetConfig``) so both the training loop and the
    CPU unit test can build it easily. The ltx_core seam injects as ``deps`` (a ``StepDeps``).

    Args:
        deps: the injected ltx_core seam (patchifier / RoPE coords / ``Modality`` / shape cls).
        schedule: the ``FlowMatchingSchedule`` sigma sampler (Seam 3 — used, NEVER modified).
        max_conditioning_items: upper bound on conditioning frames per sample; K ~ Uniform{1..this}
            distinct latent frames are conditioned each step (D-6-POSITIONS random fallback).
        conditioning_strength_range: ``(low, high)`` — each conditioning item draws its strength
            ``s ~ Uniform(low, high)``; ``denoise_mask = 1 - s`` (strength 1 => hard clean).
        device / dtype: compute placement (cuda / bf16 on Modal; cpu / float32 in unit tests).
        fps: fallback frame-rate for the temporal RoPE coord (overridden by sample metadata).
    """

    def __init__(
        self,
        deps: StepDeps,
        schedule: FlowMatchingSchedule,
        *,
        max_conditioning_items: int = 1,
        conditioning_strength_range: tuple[float, float] = (0.3, 1.0),
        device: Any = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        fps: float = DEFAULT_FPS,
    ) -> None:
        self.deps = deps
        self.schedule = schedule
        self.max_conditioning_items = max_conditioning_items
        self.conditioning_strength_range = conditioning_strength_range
        self.device = device
        self.dtype = dtype
        self.fps = fps

    def get_data_sources(self) -> Sequence[Any]:
        """The signet nested-sample sources this strategy reads (``latents`` / ``conditions``)."""
        return ["latents", "conditions"]

    def prepare_training_inputs(self, batch: Any, rng: Any = None) -> ModelInputs:
        """Turn a signet ``PrecomputedDataset`` sample into the ``ModelInputs`` the transformer eats.

        Identical to ``single_frame.prepare_training_inputs`` EXCEPT the four mask lines: N-region
        ``denoise_mask = 1 - strength``, per-token ``sigma * denoise_mask`` noising, and an explicit
        ``video_loss_mask = denoise_mask > 0`` (D-6-LOSSMASK — partial-strength tokens in the loss).

        Args:
            batch: a signet nested sample ``{"latent_conditions": {...}, "text_conditions": {...}}``.
            rng: a ``numpy.random.Generator``. ``None`` -> a fresh default generator (function-local
                numpy import — keeps module top torch + stdlib).
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

        # RoPE grid dims come from the LATENT tensor's OWN shape (Pitfall 2 — trusting pixel metadata
        # explodes the grid by the VAE compression -> the 166.9 GiB A100 OOM).
        fps = _meta_float(latent_conditions, "fps", self.fps)

        # ── patchify: [B, C, F, H, W] -> [B, seq_len, C] ────────────────────────────────────────
        video_latent_patched = deps.patchifier.patchify(video_latent)
        video_seq_len = video_latent_patched.shape[1]

        # ── noise + shifted-logit-normal sigmas (uniform_prob from config, via schedule) ─────────
        noise_patched = torch.randn_like(video_latent_patched)
        t_np = self.schedule.sample_timesteps(batch_size, video_seq_len, rng)
        sigmas = torch.tensor(t_np, device=device, dtype=dtype)  # [B]

        # ── velocity target (unchanged from single_frame / step.py) ──────────────────────────────
        video_target = noise_patched - video_latent_patched  # target = noise - latents

        # ── THE FOUR generalized mask lines: N-region denoise mask (canonical 1 - strength) ──────
        #   Positions are LATENT-native (frame indices). D-6-POSITIONS: canonical ltx has no
        #   multi-frame position policy, so we draw K ~ Uniform{1..max} distinct latent frames.
        positions = sample_conditioning_positions(lat_f, self.max_conditioning_items, rng)
        low, high = self.conditioning_strength_range
        regions: list[tuple[int, int, float]] = []
        for idx in positions:
            # SC#1: assert the latent<->pixel boundary is %8-aligned IN the strategy (holds by
            # construction — training samples latent-native positions; idx -> idx*TIME_SCALE).
            assert 0 <= idx <= lat_f - 1, f"conditioning latent_idx {idx} out of [0, {lat_f - 1}]"
            assert pixel_frame_for_latent_idx(idx) % TIME_SCALE == 0, (
                f"SC#1 violated: pixel frame for latent_idx {idx} is not % {TIME_SCALE} == 0"
            )
            s_i = float(rng.uniform(low, high))
            start, stop = latent_frame_token_span(idx, lat_h, lat_w)
            regions.append((start, stop, s_i))

        denoise_mask = build_denoise_mask(batch_size, video_seq_len, regions, device).to(dtype)

        # per-token timestep = sigma * denoise_mask (strength 1 => hard 0-clean; partial => scaled)
        t_tok = per_token_sigma(sigmas, denoise_mask)  # [B, seq_len]

        # per-token flow-matching interpolation (replaces single_frame's scalar-sigma + torch.where):
        #   noisy = (1 - t_tok) * clean + t_tok * noise  — on a strength-1 token t_tok == 0 => clean.
        t_tok_expanded = t_tok.unsqueeze(-1)  # [B, seq_len, 1]
        noisy_video = (1 - t_tok_expanded) * video_latent_patched + t_tok_expanded * noise_patched

        # EXPLICIT loss mask (D-6-LOSSMASK): partial-strength tokens (denoise > 0) participate in the
        # loss; ONLY hard-clean strength-1 tokens (denoise == 0) are excluded. NOT reconstructed from
        # ``timesteps == 0`` (Pitfall 1).
        video_loss_mask = denoise_mask > 0  # [B, seq_len] bool

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
            timesteps=t_tok,  # per-token (NOT the torch.where-to-zero form)
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

        Copied verbatim from ``single_frame.compute_loss`` — it already reads
        ``video_loss_mask`` (here the explicit ``denoise_mask > 0``): squared-error masked to the
        target tokens, normalized by the mask fraction so the loss scale is stable as the
        conditioned-token count varies -> ``[B,]``; then ``.mean()`` to a scalar for
        ``loop.py``'s ``(raw_loss / grad_accum).backward()``.
        """
        loss = (model_output.float() - model_inputs.video_targets.float()).pow(2)  # [B, seq, D]
        m = model_inputs.video_loss_mask.unsqueeze(-1).float()  # [B, seq, 1]
        per_sample = loss.mul(m).mean(dim=[-2, -1]) / m.mean(dim=[-2, -1]).clamp(min=1e-8)  # [B,]
        return per_sample.mean()  # 0-dim scalar
