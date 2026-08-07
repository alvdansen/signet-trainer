"""conditioning.a2v — ``A2VStrategy``: frozen-audio-driven video generation (audio-to-video).

The fifth concrete ``TrainingStrategy`` and a sibling of the validated ``SingleFrameStrategy``
(Phase 5), ``MultiFrameStrategy`` (Phase 6), ``ICLoraStrategy`` (Phase 7), and ``InpaintStrategy``
(Phase 9 inpaint). It implements audio-to-video training (GATE-SPEC-inpaint-a2v rev 2 item 4): the
VIDEO is generated (plain flow-matching t2v — every video token noised and in the loss) while the
driving AUDIO is FROZEN conditioning — clean audio latents at per-token timestep 0 / sigma 0,
EXCLUDED from the loss. The audio flows through the LTX-2.3 transformer's dedicated audio dual-stream
(``audio_transformer_blocks`` + the ``audio_to_video_attn`` cross-attention), so there is NO sequence
concat (unlike IC-LoRA) — the two modalities ride separate branches.

⚠ This is a plumbing-proof strategy (GATE-SPEC "~80% flip-switches"): the audio dual-stream / audio
patchifier contract is ported from the upstream ``base_strategy`` at signet's pinned SHA d6053703
(``_get_audio_positions``: audio latents patchify to ``[B, T, C*F] = [B, T, 128]`` with C=8 channels,
F=16 mel bins; positions are ``[B, 1, T, 2]`` — ONE positional dim, vs video's 3). The EXACT stored
``audio_latents/`` tensor layout and the audio-branch forward wiring want a live-GPU confirmation at
the gated a2v proof — the CPU proof here fixes the CONTRACT (frozen audio: timestep 0, no loss;
video: plain t2v) that the run must honor.

Frozen-audio semantics (the contract this strategy guarantees):
    * video Modality  — noised video (flow-matching interpolation), velocity target ``noise - clean``,
      ``video_loss_mask`` ALL True (every video token is generated and carries the loss).
    * audio Modality  — CLEAN audio latents, per-token ``timesteps = 0``, ``sigma = 0``; it is
      conditioning, never a training target — ``audio_targets`` and ``audio_loss_mask`` stay ``None``.

CRITICAL — Anti-Pattern 6 / Pitfall 6 (import-confinement):
    Module top imports ``torch`` + stdlib ONLY. It MUST NOT import ``modal``/``ltx_core``/
    ``ltx_trainer`` at module load — the heavy ltx_core VIDEO symbols inject via ``StepDeps`` from
    ``train/step.py`` (reused, never duplicated), and the AUDIO patchifier is either injected
    (CPU tests) or built FUNCTION-LOCAL (Modal-side) so the CPU proof runs Windows/CI-free.
    ``numpy`` is imported function-local only for the default-rng fallback.
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

#: Upstream audio-latent facts ([canonical] ltx-trainer ``base_strategy`` @ pinned SHA d6053703
#: ``_get_audio_positions``): audio latents are C=8 channels over F=16 mel bins, patchifying to
#: ``[B, T, C*F] = [B, T, 128]``; positions are ``[B, 1, T, 2]`` (ONE positional dim).
AUDIO_LATENT_CHANNELS: int = 8
AUDIO_MEL_BINS: int = 16

#: The batch keys ``prepare_training_inputs`` accepts for the audio-latent source, in lookup order.
#: ``audio_latent_conditions`` mirrors the ``_PRECOMPUTED_SOURCE_OUTPUT_KEYS`` naming pattern
#: (``reference_latents`` -> ``ref_latent_conditions``); ``audio_latents`` is the list-form
#: dir-name==output-key fallback (``PrecomputedDataset._normalize_data_sources``).
_AUDIO_BATCH_KEYS: tuple[str, ...] = ("audio_latent_conditions", "audio_latents")


class A2VStrategy(TrainingStrategy):
    """Audio-to-video training strategy (frozen driving audio -> generated video).

    The video objective is IDENTICAL to the unconditioned t2v path (``SingleFrameStrategy`` with
    ``first_frame_conditioning_p == 0``): full-frame flow-matching, every video token in the loss.
    The addition is the FROZEN audio Modality — clean audio latents at timestep 0, excluded from the
    loss — threaded through the SAME ``model(video=, audio=, perturbations=)`` call so the audio
    dual-stream + ``audio_to_video_attn`` cross-attention receive the conditioning.

    Constructed with primitives (NOT the whole ``SignetConfig``) so both ``training_step`` and the CPU
    unit test can build it easily. The ltx_core VIDEO seam injects as ``deps`` (a ``StepDeps``); the
    AUDIO patchifier / shape class are injected for tests or built function-local Modal-side.

    Args:
        deps: the injected ltx_core VIDEO seam (patchifier / RoPE coords / ``Modality`` / shape cls).
        schedule: the ``FlowMatchingSchedule`` sigma sampler (Seam 3 — used, NEVER modified).
        device / dtype: compute placement (cuda / bf16 on Modal; cpu / float32 in unit tests).
        fps: fallback frame-rate for the temporal RoPE coord (overridden by sample metadata).
        audio_patchifier: an object exposing ``.patchify`` + ``.get_patch_grid_bounds`` for audio
            (``ltx_core.components.patchifiers.AudioPatchifier(patch_size=1)`` Modal-side). ``None``
            -> built function-local from ltx_core at first use (Modal-side only). Injected in tests.
        audio_latent_shape_cls: the ``ltx_core.types.AudioLatentShape`` class (or a stub). ``None``
            -> imported function-local from ltx_core at first use. Injected in tests.
    """

    def __init__(
        self,
        deps: StepDeps,
        schedule: FlowMatchingSchedule,
        *,
        device: Any = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        fps: float = DEFAULT_FPS,
        audio_patchifier: Any = None,
        audio_latent_shape_cls: Any = None,
    ) -> None:
        self.deps = deps
        self.schedule = schedule
        self.device = device
        self.dtype = dtype
        self.fps = fps
        self._audio_patchifier = audio_patchifier
        self._audio_latent_shape_cls = audio_latent_shape_cls

    def get_data_sources(self) -> Sequence[Any]:
        """The signet nested-sample sources this strategy reads.

        Adds a third ``audio_latents`` source (one ``.pt`` per sample, same rel path as latents;
        the AUDIO-VAE latents ``process_dataset`` emits under ``audio_latents/`` when
        ``with_audio=True``) — mirrors how ``ICLoraStrategy`` declares its third ``reference_latents``
        source and ``InpaintStrategy`` its ``video_masks`` source. ``"audio_latents"`` is registered
        by the caller (``modal/fns.py::_PRECOMPUTED_SOURCE_OUTPUT_KEYS``) and is deliberately EXCLUDED
        from ``PrecomputedDataset._VIDEO_LATENT_SOURCE_DIRS`` so its tensors are never normalized
        through the video ``_normalize_video_latents`` path (GATE-SPEC rev 2 item 4).
        """
        return ["latents", "conditions", "audio_latents"]

    # ----------------------------------------------------------------------------------------------
    # audio dep resolution (injected in tests; function-local ltx_core build Modal-side)
    # ----------------------------------------------------------------------------------------------

    def _audio_deps(self) -> tuple[Any, Any]:
        """Return ``(audio_patchifier, audio_latent_shape_cls)`` — injected, or built from ltx_core.

        Function-local ltx_core import (Anti-Pattern 6 EXCEPTION, mirroring ``build_step_deps``):
        merely importing this module never requires ltx_core; the audio patchifier is materialized
        only when a real a2v batch is prepared Modal-side. Tests inject stubs via the constructor.
        """
        if self._audio_patchifier is None:
            from ltx_core.components.patchifiers import AudioPatchifier  # noqa: PLC0415

            self._audio_patchifier = AudioPatchifier(patch_size=1)
        if self._audio_latent_shape_cls is None:
            from ltx_core.types import AudioLatentShape  # noqa: PLC0415

            self._audio_latent_shape_cls = AudioLatentShape
        return self._audio_patchifier, self._audio_latent_shape_cls

    @staticmethod
    def _extract_audio_latent(batch: Any) -> torch.Tensor:
        """Pull the raw audio-latent tensor out of the batch — fail loud when absent or malformed.

        Accepts either payload form under the audio batch key:
          * a bare tensor (``[C, T, mel]`` or ``[B, C, T, mel]``), or
          * a ``{"latents": tensor, ...}`` dict (the video-latent-shaped payload
            ``process_dataset`` writes; the extra metadata keys are ignored here).
        """
        payload = None
        for key in _AUDIO_BATCH_KEYS:
            if key in batch:
                payload = batch[key]
                break
        if payload is None:
            raise KeyError(
                f"audio_to_video batch is missing the audio latents: none of "
                f"{list(_AUDIO_BATCH_KEYS)} present in batch keys "
                f"{sorted(k for k in batch)} — the dataset must be built with the 'audio_latents' "
                f"source (get_data_sources) and the encode must have run with with_audio=True."
            )
        if isinstance(payload, dict):
            if "latents" not in payload:
                raise KeyError(
                    f"audio latent payload is a dict without a 'latents' key (keys: "
                    f"{sorted(payload)}) — expected a bare tensor or {{'latents': tensor}}."
                )
            payload = payload["latents"]
        if not isinstance(payload, torch.Tensor):
            raise TypeError(
                f"audio latents must be a torch.Tensor, got {type(payload).__name__}."
            )
        return payload

    def _build_frozen_audio_modality(
        self, batch: Any, text_conditions: dict, batch_size: int
    ) -> Any:
        """Build the FROZEN audio Modality: clean latents, per-token timestep 0, sigma 0.

        Ports the audio-branch construction from the upstream ``base_strategy`` (``_get_audio_positions``
        + the Modality build): unwrap the ``[C, T, mel]`` audio latent, patchify to ``[B, T, 128]``,
        set every audio token's timestep and the modality sigma to 0 (nothing about the audio is
        denoised — it is the driving conditioning), and attach the audio positions ``[B, 1, T, 2]``.
        """
        audio_patchifier, audio_latent_shape_cls = self._audio_deps()
        device = self.device
        dtype = self.dtype

        audio_latent = self._extract_audio_latent(batch).to(device, dtype=dtype)
        if audio_latent.dim() == 3:  # [C, T, mel] -> [B=1, C, T, mel]
            audio_latent = audio_latent.unsqueeze(0)
        if audio_latent.dim() != 4:
            raise ValueError(
                f"audio latent must be [C, T, mel] or [B, C, T, mel]; got shape "
                f"{tuple(audio_latent.shape)}."
            )
        a_b, a_c, a_t, a_mel = audio_latent.shape
        if a_b != batch_size:
            raise ValueError(
                f"audio latent batch {a_b} != video batch {batch_size} — the a2v pair must be "
                f"per-sample aligned (one audio clip per video clip)."
            )

        # Patchify to [B, T, C*mel] (== [B, T, 128] at the LTX audio contract) — the audio branch's
        # token layout ([canonical] base_strategy: "[B, T, C*F] = [B, T, 128]", C=8, F=16).
        audio_patched = audio_patchifier.patchify(audio_latent)
        audio_seq_len = audio_patched.shape[1]

        # FROZEN: clean latents, sigma 0, per-token timestep 0 — the audio is conditioning, never
        # denoised (this is the whole "frozen audio" contract of a2v).
        audio_sigma = torch.zeros(batch_size, device=device, dtype=dtype)  # [B]
        audio_timesteps = torch.zeros(batch_size, audio_seq_len, device=device, dtype=dtype)

        # Audio positions [B, 1, T, 2] — get_patch_grid_bounds over the AudioLatentShape (ONE
        # positional dim; NOT the video get_pixel_coords path). [canonical] base_strategy.
        audio_positions = audio_patchifier.get_patch_grid_bounds(
            output_shape=audio_latent_shape_cls(
                frames=a_t,
                mel_bins=a_mel,
                batch=batch_size,
                channels=a_c,
            ),
            device=device,
        )
        if isinstance(audio_positions, torch.Tensor):
            audio_positions = audio_positions.to(dtype)

        # Audio context: prefer the audio prompt embeds (the audio branch cross-attends to them);
        # fall back to the video prompt embeds when the encode wrote only one context tensor.
        audio_ctx = text_conditions.get("audio_prompt_embeds")
        if audio_ctx is None:
            audio_ctx = text_conditions["video_prompt_embeds"]
        audio_ctx = audio_ctx.to(device, dtype=dtype)
        if audio_ctx.dim() == 2:
            audio_ctx = audio_ctx.unsqueeze(0)
        prompt_attention_mask = text_conditions["prompt_attention_mask"].to(device)
        if prompt_attention_mask.dim() == 1:
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0)
        audio_ctx_mask = torch.zeros_like(prompt_attention_mask, dtype=dtype)
        audio_ctx_mask[prompt_attention_mask == 0] = float("-inf")

        return self.deps.modality_cls(
            enabled=True,
            latent=audio_patched,
            sigma=audio_sigma,
            timesteps=audio_timesteps,
            positions=audio_positions,
            context=audio_ctx,
            context_mask=audio_ctx_mask,
        )

    def prepare_training_inputs(self, batch: Any, rng: Any = None) -> ModelInputs:
        """Turn a signet ``PrecomputedDataset`` sample into the a2v ``ModelInputs``.

        Builds the VIDEO modality as plain flow-matching t2v (every token noised + in the loss —
        the video is generated), then attaches the FROZEN audio modality (clean audio latents at
        timestep 0, excluded from the loss). ``audio_targets`` / ``audio_loss_mask`` stay ``None``:
        the audio is a driving input, never a training target.

        Args:
            batch: a signet nested sample carrying ``latent_conditions`` (target video latents),
                ``text_conditions``, and the ``audio_latents`` source (see ``_AUDIO_BATCH_KEYS``).
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

        # ── VIDEO: unwrap + normalize to [B, C, F, H, W] (Pitfall 2) ─────────────────────────────
        video_latent = latent_conditions["latents"].to(device, dtype=dtype)
        if video_latent.dim() == 4:  # [C, F, H, W] -> [B=1, C, F, H, W]
            video_latent = video_latent.unsqueeze(0)
        batch_size, _channels, lat_f, lat_h, lat_w = video_latent.shape
        fps = _meta_float(latent_conditions, "fps", self.fps)

        # ── VIDEO: patchify -> [B, seq_len, C] ────────────────────────────────────────────────────
        video_latent_patched = deps.patchifier.patchify(video_latent)
        video_seq_len = video_latent_patched.shape[1]

        # ── VIDEO: noise + shifted-logit-normal sigmas (uniform_prob from config, via schedule) ───
        noise_patched = torch.randn_like(video_latent_patched)
        t_np = self.schedule.sample_timesteps(batch_size, video_seq_len, rng)
        sigmas = torch.tensor(t_np, device=device, dtype=dtype)  # [B]

        # ── VIDEO: flow-matching interpolation + velocity target (plain t2v — nothing conditioned) ─
        sigmas_expanded = sigmas.view(-1, 1, 1)  # [B, 1, 1]
        noisy_video = (1 - sigmas_expanded) * video_latent_patched + sigmas_expanded * noise_patched
        video_target = noise_patched - video_latent_patched  # target = noise - latents
        video_timesteps = sigmas.view(-1, 1).expand(-1, video_seq_len)  # [B, seq_len]
        # Every video token is generated -> the WHOLE frame is in the loss (a2v generates the video).
        video_loss_mask = torch.ones(batch_size, video_seq_len, dtype=torch.bool, device=device)

        # ── VIDEO: RoPE pixel coords (temporal coord -> seconds) ─────────────────────────────────
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

        # ── VIDEO: text context (int attention mask -> additive −inf float mask) ─────────────────
        video_prompt_embeds = text_conditions["video_prompt_embeds"].to(device, dtype=dtype)
        prompt_attention_mask = text_conditions["prompt_attention_mask"].to(device)
        if video_prompt_embeds.dim() == 2:
            video_prompt_embeds = video_prompt_embeds.unsqueeze(0)
        if prompt_attention_mask.dim() == 1:
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0)
        additive_mask = torch.zeros_like(prompt_attention_mask, dtype=dtype)
        additive_mask[prompt_attention_mask == 0] = float("-inf")

        video_modality = deps.modality_cls(
            enabled=True,
            latent=noisy_video,
            sigma=sigmas,
            timesteps=video_timesteps,
            positions=pixel_coords,
            context=video_prompt_embeds,
            context_mask=additive_mask,
        )

        # ── AUDIO: the frozen driving-audio modality (clean, timestep 0, out of loss) ────────────
        audio_modality = self._build_frozen_audio_modality(
            batch, text_conditions, batch_size
        )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_target,
            audio_targets=None,  # audio is conditioning, never a training target
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,  # audio excluded from the loss (frozen)
        )

    def compute_loss(self, model_inputs: ModelInputs, model_output: Any) -> torch.Tensor:
        """Video-only mask-normalized velocity loss, 0-dim scalar (Pitfall 1).

        Only the VIDEO is generated in a2v, so the loss is the standard squared velocity error over
        ``video_loss_mask`` (all True here) — the frozen audio never contributes. Identical math to
        ``SingleFrameStrategy.compute_loss`` (mask-fraction normalized -> ``[B,]`` -> ``.mean()``).
        The model returns ``(video_pred, audio_pred)``; ``train/step.py`` unpacks only ``video_pred``,
        so ``model_output`` here is the video prediction.
        """
        loss = (model_output.float() - model_inputs.video_targets.float()).pow(2)  # [B, seq, D]
        m = model_inputs.video_loss_mask.unsqueeze(-1).float()  # [B, seq, 1]
        per_sample = loss.mul(m).mean(dim=[-2, -1]) / m.mean(dim=[-2, -1]).clamp(min=1e-8)  # [B,]
        return per_sample.mean()  # 0-dim scalar
