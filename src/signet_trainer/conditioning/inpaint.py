"""conditioning.inpaint — ``InpaintStrategy``: per-token spatial denoise-mask (inpaint) training.

The fourth concrete ``TrainingStrategy`` and a sibling of the validated ``SingleFrameStrategy``
(Phase 5), ``MultiFrameStrategy`` (Phase 6), and ``ICLoraStrategy`` (Phase 7). It implements
the prior project's ``flexible``/``mask`` semantics (GATE-SPEC-inpaint-a2v rev 2): a per-token SPATIAL
denoise mask on the UNCHANGED transformer — no extra input channels, no sequence concat.

Semantic oracle — [canonical] upstream ``FlexibleStrategy`` (Lightricks/LTX-2 @ SHA
``780984275fd47128b02bef9b5c085404276866ee``, ``packages/ltx-trainer/src/ltx_trainer/
training_strategies/flexible.py`` — ``MaskConditionConfig`` + ``_apply_intrinsic_condition``).
NOTE: signet's own pinned SHA d6053703 PREDATES the upstream flexible/mask epoch — that is WHY
this strategy (and the mask encode) is signet-native; the upstream file is a READ-ONLY reference,
never imported. The ported semantics, verbatim from the oracle:

  * mask load: per-sample mask reshaped to ``[B, seq_len]``, binarized at ``> 0.5`` ("Binarize to
    match inference, which thresholds masks at load time").
  * per-sample Bernoulli apply: ``apply_per_sample = rand(B) < probability``; the mask is ZEROED
    for unselected samples (``mask = mask * apply_per_sample``) — an all-zero mask means every
    token is generated, i.e. the sample falls back to PLAIN t2v denoise on the full frame.
  * apply (binary mask ⇒ the oracle's lerp form == the contract's ``where`` form):
        noisy      = where(mask > 0.5, clean_latent, noisy)      # keep-tokens are the flow endpoint
        timesteps  = where(mask > 0.5, 0, sigma)                  # keep-tokens get timestep 0
        loss_mask  = mask <= 0.5                                  # keep-tokens excluded from loss

Mask polarity ON THE ENCODED TENSOR (GATE-SPEC / [precedent]):
    1.0 = KEEP/context  (clean latent, timestep 0, excluded from loss)
    0.0 = GENERATE      (noised, in loss)
(Source-video polarity: the region-to-generate is BLACK(0) in the mask mp4 — ffmpeg ``negate``
applied at mask-render time, precedent convention. That is upstream of the encode; this strategy
only ever sees the encoded ``[F_lat, H_lat, W_lat]`` tensor.)

Token order: the mask is flattened FRAME-MAJOR — ``token = f*H_lat*W_lat + h*W_lat + w`` — the
exact ordering of ``conditioning/mask_ops.py::latent_frame_token_span`` (patchifier is
frame-major at patch_size=1, one token per latent cell). ``reshape(B, F*H*W)`` on a
``[B, F, H, W]`` tensor IS that order; the CPU dry-run gate (dryrun/shapes.py) and
``tests/test_inpaint_strategy.py`` prove it against a hand-computed asymmetric pattern.

Dims: inpaint mode requires H%64==0, W%64==0, frames%8==1 (STRICTER than the video %32 rule —
GATE-SPEC "÷64 required for inpaint; validated by the smoke"). That rule is ENFORCED upstream of
training by ``config/validators.py`` (the inpaint-mode %64 assert) at config load — this module
does not own it; it fail-louds only on the mask↔latent LATENT-dim mismatch it can see per sample.

CRITICAL — Anti-Pattern 6 / Pitfall 6 (import-confinement):
    Module top imports ``torch`` + stdlib ONLY. It MUST NOT import ``modal``/``ltx_core``/
    ``ltx_trainer`` — the heavy ltx_core symbols inject via ``StepDeps`` from ``train/step.py``
    (reused, never duplicated) so the CPU proof runs Windows/CI-free (zero GPU, zero Modal
    spend). ``numpy`` is imported function-local only for the default-rng fallback.

Pitfall 2 (the 166.9 GiB OOM): latent f/h/w are derived from the latent tensor's own ``.shape``,
NEVER from pixel metadata — and the mask dims are VERIFIED against them (fail loud on mismatch).
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

#: The batch keys ``prepare_training_inputs`` accepts for the mask source, in lookup order.
#: ``video_mask_conditions`` mirrors the ``_PRECOMPUTED_SOURCE_OUTPUT_KEYS`` naming pattern
#: (``reference_latents`` -> ``ref_latent_conditions``); ``video_masks`` is the list-form
#: dir-name==output-key fallback (``PrecomputedDataset._normalize_data_sources``).
_MASK_BATCH_KEYS: tuple[str, ...] = ("video_mask_conditions", "video_masks")


class InpaintStrategy(TrainingStrategy):
    """Masked-region inpaint training strategy (prior-project ``flexible``/``mask`` semantics).

    Applies a per-sample, per-token binary spatial mask to the standard t2v flow-matching
    objective: mask-KEEP tokens (>0.5) carry the CLEAN latent at per-token timestep 0 and are
    excluded from the loss; mask-GENERATE tokens (<=0.5) are noised and carry the loss. The
    sequence length, positions, and model call are IDENTICAL to ``SingleFrameStrategy`` — no
    reference prefix, no concat (``ref_seq_len`` stays ``None``).

    Constructed with primitives (NOT the whole ``SignetConfig``) so both ``training_step`` and the
    CPU unit test can build it easily. The ltx_core seam injects as ``deps`` (a ``StepDeps``).

    Args:
        deps: the injected ltx_core seam (patchifier / RoPE coords / ``Modality`` / shape cls).
        schedule: the ``FlowMatchingSchedule`` sigma sampler (Seam 3 — used, NEVER modified).
        inpaint_mask_probability: P(apply the sample's mask) per sample — [canonical] upstream
            ``MaskConditionConfig.probability`` per-sample Bernoulli; [precedent] the prior project trains at
            1.0 (its inpaint config's mask prob 1.0 — the schema default). A sample that
            draws NO-mask falls back to PLAIN t2v denoise on the full frame (its effective mask is
            all zeros = everything generated, all tokens in the loss).
        device / dtype: compute placement (cuda / bf16 on Modal; cpu / float32 in unit tests).
        fps: fallback frame-rate for the temporal RoPE coord (overridden by sample metadata).
        mask_dir: sub-dir name under ``preprocessed_data_root`` holding the encoded mask tensors —
            mirrors ``config.conditioning.inpaint_mask_dir`` (schema.py). Defaulted to the schema's
            own default ``"video_masks"`` so every existing caller (train/step.py's per-step
            construction, which never needs ``get_data_sources()``) is unaffected. #21 finding 3:
            this was previously a bare literal here while ``inpaint_mask_dir`` was honored on the
            WRITE side (modal/entrypoint.py) — a non-default value validated clean and was then
            silently ignored on read. Threading it here closes that gap; the caller
            (``modal/fns.py``) is responsible for passing ``config.conditioning.inpaint_mask_dir``.
    """

    def __init__(
        self,
        deps: StepDeps,
        schedule: FlowMatchingSchedule,
        *,
        inpaint_mask_probability: float = 1.0,
        device: Any = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        fps: float = DEFAULT_FPS,
        mask_dir: str = "video_masks",
    ) -> None:
        self.deps = deps
        self.schedule = schedule
        self.inpaint_mask_probability = inpaint_mask_probability
        self.device = device
        self.dtype = dtype
        self.fps = fps
        self.mask_dir = mask_dir

    def get_data_sources(self) -> Sequence[Any]:
        """The signet nested-sample sources this strategy reads.

        Adds a third mask source (one ``.pt`` per sample, same rel path as latents, float32
        ``[F_lat, H_lat, W_lat]`` in {0., 1.}) under ``self.mask_dir`` — mirrors how
        ``ICLoraStrategy`` declares its third ``reference_latents`` source. #21 finding 3:
        ``self.mask_dir`` (config-driven, default ``"video_masks"``) replaces the old bare
        ``"video_masks"`` literal so a configured ``inpaint_mask_dir`` is actually read, not just
        written. The default itself deliberately does NOT contain the substring ``"latent"``, so
        ``PrecomputedDataset.__getitem__``'s legacy ``_normalize_video_latents`` branch cannot
        misfire on it — an operator-chosen ``mask_dir`` should follow the same convention.
        """
        return ["latents", "conditions", self.mask_dir]

    @staticmethod
    def _extract_mask(batch: Any) -> torch.Tensor:
        """Pull the raw mask tensor out of the batch — fail loud when absent or malformed.

        Accepts either payload form under the mask batch key:
          * a bare tensor (the signet-native mask-encode contract: one ``.pt`` per sample holding a
            float32 ``[F_lat, H_lat, W_lat]`` tensor), or
          * a ``{"mask": tensor}`` dict ([canonical] the upstream oracle reads
            ``batch[mask_dir]["mask"]`` — kept so an upstream-shaped cache also loads).
        """
        payload = None
        for key in _MASK_BATCH_KEYS:
            if key in batch:
                payload = batch[key]
                break
        if payload is None:
            raise KeyError(
                f"inpaint batch is missing the video mask: none of {list(_MASK_BATCH_KEYS)} "
                f"present in batch keys {sorted(k for k in batch)} — the dataset must be built "
                f"with the 'video_masks' source (get_data_sources)."
            )
        if isinstance(payload, dict):
            if "mask" not in payload:
                raise KeyError(
                    f"inpaint mask payload is a dict without a 'mask' key (keys: "
                    f"{sorted(payload)}) — expected a bare tensor or {{'mask': tensor}}."
                )
            payload = payload["mask"]
        if not isinstance(payload, torch.Tensor):
            raise TypeError(
                f"inpaint mask must be a torch.Tensor, got {type(payload).__name__}."
            )
        return payload

    def prepare_training_inputs(self, batch: Any, rng: Any = None) -> ModelInputs:
        """Turn a signet ``PrecomputedDataset`` sample into the masked-denoise ``ModelInputs``.

        Ports single_frame's unwrap → patchify → sigma sample → flow-match interpolation →
        velocity target → RoPE coords → ``Modality`` build, replacing the first-frame conditioning
        block with the mask application from the upstream oracle: load the ``[F_lat, H_lat,
        W_lat]`` mask, VERIFY its dims equal the sample's latent dims (fail loud), flatten
        frame-major to ``[seq_len]``, per-sample Bernoulli-gate it by ``inpaint_mask_probability``
        (no-mask draw ⇒ all-zero mask ⇒ plain full-frame t2v denoise), then apply the contract's
        ``where`` semantics onto the noisy latents / per-token timesteps / ``video_loss_mask``.

        Args:
            batch: a signet nested sample carrying ``latent_conditions`` (target latents),
                ``text_conditions``, and the ``video_masks`` source (see ``_MASK_BATCH_KEYS``).
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

        fps = _meta_float(latent_conditions, "fps", self.fps)

        # ── mask: load, dim-verify against THIS sample's latent dims (fail loud), flatten ────────
        mask = self._extract_mask(batch).to(device)
        if mask.dim() == 3:  # [F, H, W] -> [B=1, F, H, W] (mirror the latent unwrap)
            mask = mask.unsqueeze(0)
        if mask.dim() != 4:
            raise ValueError(
                f"inpaint mask must be [F_lat, H_lat, W_lat] or [B, F_lat, H_lat, W_lat]; got "
                f"shape {tuple(mask.shape)}."
            )
        if tuple(mask.shape[-3:]) != (lat_f, lat_h, lat_w) or mask.shape[0] != batch_size:
            raise ValueError(
                f"inpaint mask dims {tuple(mask.shape)} do not match the sample's latent dims "
                f"[B={batch_size}, F={lat_f}, H={lat_h}, W={lat_w}] — the mask must be encoded on "
                f"the SAME latent grid as the target (H//32, W//32, (F-1)//8+1). Re-encode the "
                f"mask; never train on a mismatched pair."
            )
        # Frame-major flatten: token = f*H_lat*W_lat + h*W_lat + w — EXACTLY the ordering of
        # mask_ops.latent_frame_token_span (the patchifier is frame-major at patch_size=1).
        mask_flat = mask.reshape(batch_size, lat_f * lat_h * lat_w).to(torch.float32)

        # ── patchify: [B, C, F, H, W] -> [B, seq_len, C] ────────────────────────────────────────
        video_latent_patched = deps.patchifier.patchify(video_latent)
        video_seq_len = video_latent_patched.shape[1]
        # Belt-and-braces: the flattened mask and the patchified sequence MUST agree (they are both
        # F*H*W at patch_size=1; a divergence means the patchifier seam changed under us).
        if mask_flat.shape[1] != video_seq_len:
            raise ValueError(
                f"flattened mask length {mask_flat.shape[1]} != patchified seq_len "
                f"{video_seq_len} — the frame-major mask flatten no longer matches the "
                f"patchifier's token order."
            )

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

        # ── per-sample Bernoulli mask gate ([canonical] oracle _apply_intrinsic_condition) ───────
        # A sample the draw does NOT select gets its mask zeroed — all tokens generated — i.e. it
        # falls back to PLAIN t2v denoise on the full frame (nothing kept, everything in the loss).
        p = self.inpaint_mask_probability
        apply_per_sample = torch.rand(batch_size, device=device) < p
        effective_mask = mask_flat * apply_per_sample.view(-1, 1).to(mask_flat.dtype)

        # ── the contract's where() semantics (binary mask ⇒ identical to the oracle's lerp) ──────
        keep = effective_mask > 0.5  # [B, seq_len] bool: True = KEEP/context token
        # clean latents on keep-tokens (they are the flow endpoint / timestep 0)
        noisy_video = torch.where(keep.unsqueeze(-1), video_latent_patched, noisy_video)
        # timestep 0 on keep-tokens; the sampled sigma everywhere else
        video_timesteps = torch.where(
            keep, torch.zeros_like(video_timesteps), video_timesteps
        )
        video_loss_mask = ~keep  # == (effective_mask <= 0.5): True = GENERATE = compute loss

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
            video_targets=video_target,  # full-length (no ref prefix; ref_seq_len stays None)
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
        )

    def compute_loss(self, model_inputs: ModelInputs, model_output: Any) -> torch.Tensor:
        """Mask-fraction-normalized velocity loss on GENERATE tokens only, 0-dim scalar (Pitfall 1).

        Identical math to ``SingleFrameStrategy.compute_loss`` (and to the oracle's
        ``_compute_modality_loss`` with a zero-length condition prefix): squared error, masked to
        ``video_loss_mask`` (GENERATE tokens — KEEP/context tokens are masked off), normalized by
        the mask fraction so the loss scale is stable as the masked-region size varies →
        ``[B,]``; then ``.mean()`` to a scalar for ``loop.py``'s
        ``(raw_loss / grad_accum).backward()``. A degenerate all-KEEP mask yields loss 0 (the
        ``clamp(min=1e-8)`` guards the 0/0) — nothing to generate, nothing to learn.
        """
        loss = (model_output.float() - model_inputs.video_targets.float()).pow(2)  # [B, seq, D]
        m = model_inputs.video_loss_mask.unsqueeze(-1).float()  # [B, seq, 1]
        per_sample = loss.mul(m).mean(dim=[-2, -1]) / m.mean(dim=[-2, -1]).clamp(min=1e-8)  # [B,]
        return per_sample.mean()  # 0-dim scalar
