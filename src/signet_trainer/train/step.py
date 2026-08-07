"""train.step — the ltx_core ``StepDeps`` seam + ``training_step`` (the thin forward-driver).

Phase 5 (Plan 05-04) retrofit to **Option A-hybrid**: ``training_step`` is no longer the objective
itself — it is a thin driver that constructs a ``SingleFrameStrategy`` (conditioning/single_frame.py,
Plan 05-03) and routes ``prepare_training_inputs → model(...) → compute_loss``. The LTX velocity
flow-matching objective (shifted-logit-normal σ, ``noisy=(1−σ)·clean+σ·noise``, velocity target
``noise−clean``, RoPE coords, ``Modality`` build) now lives IN the strategy, which is a strict
generalization of the Phase-3 step — with ``first_frame_conditioning_p=0`` (the loop default) the
routed loss is **byte-equivalent** to the Phase-3-validated objective (backward compat is an
acceptance criterion). ``train/loop.py`` (D-03), the loader, the offloader, and ``train/flow_match.py``
(Seam 3) are all UNTOUCHED — the loop's call site does not pass the conditioning kwargs, so it
defaults to the unconditioned path.

This module retains the shared substrate the strategy imports (never duplicated):
  * ``StepDeps`` / ``build_step_deps()`` — the injectable ltx_core seam (function-local heavy import);
  * the nested-sample metadata helpers (``_as_scalar`` / ``_meta_int`` / ``_meta_float``);
  * the LTX compression / channel / fps constants.

CRITICAL — Anti-Pattern 6 (Modal-side EXCEPTION, mirror ``models/loader.py``):
    The heavy ltx_core symbols (``VideoLatentPatchifier``, ``VideoLatentShape``,
    ``get_pixel_coords``, ``SpatioTemporalScaleFactors``, ``Modality``) are imported
    FUNCTION-LOCAL inside ``build_step_deps()`` so this module stays OFF the dry-run/config
    import path AND ``tests/test_training_step.py`` can drive ``training_step`` on CPU with a
    stubbed ``StepDeps`` (no ltx_core, no CUDA). Module-load imports are torch + stdlib only.
    ``SingleFrameStrategy`` is likewise imported FUNCTION-LOCAL inside ``training_step`` — this
    breaks the train.step ↔ conditioning.single_frame import cycle (the strategy imports
    ``StepDeps``/``_meta_float``/constants from here) and keeps the module top torch + stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from signet_trainer.train.flow_match import FlowMatchingSchedule

# LTX-2.3 VAE compression factors (mirror enochiatron train.py:131-133) — used for the
# defensive pixel-dim reconstruction when the precomputed ``.pt`` lacks metadata (A3).
LTX_VAE_SPATIAL_COMPRESSION = 32
LTX_VAE_TEMPORAL_COMPRESSION = 8
#: The LTX-2.3 video latent channel count (``VideoLatentShape.channels``, train.py:1521).
LTX_VIDEO_LATENT_CHANNELS = 128
#: Fallback frame-rate used to convert the temporal RoPE coord to seconds when the sample
#: carries no ``fps`` metadata (the LTX-2 example configs train at 24fps).
DEFAULT_FPS = 24.0


# --------------------------------------------------------------------------------------------------
# The injectable ltx_core seam (function-local on Modal, stubbed on CPU tests)
# --------------------------------------------------------------------------------------------------


@dataclass
class StepDeps:
    """The ltx_core objects ``training_step`` needs, bundled so they inject as one unit.

    On Modal, ``build_step_deps()`` constructs these from ltx_core (function-local import).
    In CPU unit tests, a fake ``StepDeps`` (identity patchify + recording shape/Modality stubs)
    proves the objective math without ltx_core or CUDA.
    """

    patchifier: Any
    get_pixel_coords: Callable[..., torch.Tensor]
    scale_factors: Any
    video_latent_shape_cls: Callable[..., Any]
    modality_cls: Callable[..., Any]


def build_step_deps() -> StepDeps:
    """Construct the real ltx_core ``StepDeps`` — FUNCTION-LOCAL heavy import (Modal-side only).

    Mirrors ``models/loader.py``'s deliberate Anti-Pattern-6 exception: the ``ltx_core`` import
    lives inside the call so importing ``train.step`` (or reading it for a structural scan) never
    requires ltx_core installed. The import paths mirror enochiatron train.py:93-105.
    """
    from ltx_core.components.patchifiers import (  # noqa: PLC0415 — Modal-side, function-local
        VideoLatentPatchifier,
        get_pixel_coords,
    )
    from ltx_core.model.transformer.modality import Modality  # noqa: PLC0415
    from ltx_core.types import (  # noqa: PLC0415
        SpatioTemporalScaleFactors,
        VideoLatentShape,
    )

    return StepDeps(
        patchifier=VideoLatentPatchifier(patch_size=1),
        get_pixel_coords=get_pixel_coords,
        scale_factors=SpatioTemporalScaleFactors.default(),
        video_latent_shape_cls=VideoLatentShape,
        modality_cls=Modality,
    )


# --------------------------------------------------------------------------------------------------
# signet nested-sample unwrap helpers (Pitfall 2)
# --------------------------------------------------------------------------------------------------


def _as_scalar(value: Any) -> Any:
    """Coerce a (possibly batch-collated) metadata value to a python scalar.

    The DataLoader's default collate turns a sample's ``int``/``float`` metadata into a 1-element
    tensor; un-collated samples carry the raw scalar. This returns the underlying python number.
    """
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _meta_int(source: dict[str, Any], key: str, fallback: int) -> int:
    if key in source and source[key] is not None:
        return int(_as_scalar(source[key]))
    return int(fallback)


def _meta_float(source: dict[str, Any], key: str, fallback: float) -> float:
    if key in source and source[key] is not None:
        return float(_as_scalar(source[key]))
    return float(fallback)


# --------------------------------------------------------------------------------------------------
# The training step
# --------------------------------------------------------------------------------------------------


def training_step(
    model: Any,
    batch: dict[str, Any],
    schedule: FlowMatchingSchedule,
    rng: Any,
    *,
    device: Any = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    fps: float = DEFAULT_FPS,
    deps: StepDeps | None = None,
    conditioning_mode: str = "none",
    max_conditioning_items: int = 1,
    conditioning_strength_range: tuple[float, float] = (0.3, 1.0),
    first_frame_conditioning_p: float = 0.0,
    first_frame_conditioning_strength: float = 1.0,
    reference_downscale_factor: int = 1,
    inpaint_mask_probability: float = 1.0,
) -> torch.Tensor:
    """One LTX-2.3 flow-matching training step — the thin strategy-routed forward driver (05-04/06-05).

    Dispatches on ``conditioning_mode`` (Seam D, Plan 06-05) to build the conditioning strategy from
    the injected ltx_core seam, then drives ``prepare_training_inputs → model(...) → compute_loss``.
    The objective math lives IN the selected strategy. With ``conditioning_mode="none"`` (the loop
    default) the returned loss is byte-equivalent to the Phase-3-validated velocity loss.

      * ``"ic_lora"`` → ``ICLoraStrategy`` (Plan 07-01): concatenates a clean reference-latent
        prefix to the noised target (doubled sequence + doubled RoPE positions) and computes the
        loss only on the target slice. ``reference_downscale_factor`` scales the reference position
        axes (inert at the default 1, D-7-REF11).
      * ``"inpaint"`` → ``InpaintStrategy`` (Phase 9 gates): per-token spatial denoise mask
        (prior-project flexible/mask semantics) — mask-KEEP tokens clean at timestep 0 + out of loss,
        mask-GENERATE tokens noised + in loss; ``inpaint_mask_probability`` is the per-sample
        Bernoulli apply gate (a no-mask draw falls back to plain full-frame t2v denoise).
      * ``"multi_frame"`` → ``MultiFrameStrategy`` (Plan 06-02): N latent-frame regions with
        per-item strength drawn from ``conditioning_strength_range`` (strength-in-training).
      * ``"single_frame"`` → ``SingleFrameStrategy(first_frame_conditioning_p=...)`` (Plan 05-03).
      * ``"none"`` / default → ``SingleFrameStrategy(first_frame_conditioning_p=0.0)`` — the
        byte-equivalent unconditioned path (Phase-3/5 regression safety).

    Args:
        model: PEFT-wrapped LTX-2.3 model exposing ``model(video=Modality, audio=None,
            perturbations=None) -> (pred, _)`` (enochiatron train.py:1620-1622).
        batch: a signet ``PrecomputedDataset`` sample (or a bs=1 collated batch of one) —
            ``{"latent_conditions": {"latents": [C,F,H,W] | [B,C,F,H,W], …},
               "text_conditions": {"video_prompt_embeds": [.,4096], "prompt_attention_mask": […]}}``.
        schedule: the 03-01 ``FlowMatchingSchedule`` (already carrying ``uniform_prob`` from config).
        rng: a ``numpy.random.Generator`` (seed 42 in the loop) for the timestep draw.
        device / dtype: compute placement (cuda / bf16 Modal-side; cpu / float32 in unit tests).
        fps: fallback frame-rate for the temporal RoPE coord (overridden by sample metadata).
        deps: the ltx_core seam; ``None`` -> ``build_step_deps()`` (Modal-side). Tests inject a stub.
        conditioning_mode: reference-control mode selecting the strategy (see above). Default
            ``"none"`` → the unconditioned, byte-equivalent path.
        max_conditioning_items: upper bound on conditioned latent frames per step (multi_frame only).
        conditioning_strength_range: ``(low, high)`` per-item strength range (multi_frame only).
        first_frame_conditioning_p: P(condition on the first latent frame) per sample (single_frame).
        first_frame_conditioning_strength: RESERVED (Phase 6); inert on the hard-clean single path.
        reference_downscale_factor: IC-LoRA (ic_lora) only — RoPE height/width scale for the reference
            positions when the reference is a downscaled view of the target; inert at 1 (D-7-REF11).
        inpaint_mask_probability: inpaint only — P(apply the sample's spatial mask) per sample
            ([canonical] upstream flexible.py MaskConditionConfig.probability; [precedent] the
            prior project trains at 1.0). Unselected samples train as plain full-frame t2v.

    Returns:
        A 0-dim float32 scalar loss (Pitfall 1 — never a ``[B,]`` leak to ``loop.py``'s backward()).
    """
    if deps is None:
        deps = build_step_deps()

    # FUNCTION-LOCAL strategy imports (Anti-Pattern 6 / Pitfall 6): break the train.step ↔
    # conditioning.* import cycle (the strategies import StepDeps/_meta_float/constants from THIS
    # module) and keep step.py's module top torch + stdlib only.
    if conditioning_mode == "ic_lora":
        from signet_trainer.conditioning.ic_lora import ICLoraStrategy  # noqa: PLC0415

        # IC-LoRA (Phase 7): the concatenated [ref, target] sequence flows through the SAME
        # model(...) call below — the forward is agnostic to the doubled sequence length.
        strategy: Any = ICLoraStrategy(
            deps=deps,
            schedule=schedule,
            first_frame_conditioning_p=first_frame_conditioning_p,
            reference_downscale_factor=reference_downscale_factor,
            device=device,
            dtype=dtype,
            fps=fps,
        )
    elif conditioning_mode == "inpaint":
        from signet_trainer.conditioning.inpaint import InpaintStrategy  # noqa: PLC0415

        # Inpaint (Phase 9 gates): per-token spatial denoise mask on the UNCHANGED transformer —
        # same sequence length / positions / model(...) call as single_frame (no concat).
        strategy = InpaintStrategy(
            deps=deps,
            schedule=schedule,
            inpaint_mask_probability=inpaint_mask_probability,
            device=device,
            dtype=dtype,
            fps=fps,
        )
    elif conditioning_mode == "audio_to_video":
        from signet_trainer.conditioning.a2v import A2VStrategy  # noqa: PLC0415

        # Audio-to-video (Phase 9 gates): the VIDEO is generated (plain t2v — noised, all tokens in
        # loss) while the driving AUDIO is FROZEN conditioning (clean audio latents at sigma 0 /
        # per-token timestep 0, EXCLUDED from the loss). The frozen audio Modality rides through the
        # ``audio=`` argument of the SAME model(...) call below (threaded from ``inputs.audio`` — the
        # a2v-only change to the model call). audio_transformer_blocks is a real dual-stream branch,
        # so no sequence concat.
        strategy = A2VStrategy(
            deps=deps,
            schedule=schedule,
            device=device,
            dtype=dtype,
            fps=fps,
        )
    elif conditioning_mode == "multi_frame":
        from signet_trainer.conditioning.multi_frame import MultiFrameStrategy  # noqa: PLC0415

        strategy = MultiFrameStrategy(
            deps=deps,
            schedule=schedule,
            max_conditioning_items=max_conditioning_items,
            conditioning_strength_range=conditioning_strength_range,
            device=device,
            dtype=dtype,
            fps=fps,
        )
    else:
        from signet_trainer.conditioning.single_frame import SingleFrameStrategy  # noqa: PLC0415

        # single_frame → the configured p; none/default → the byte-equivalent unconditioned p=0 path.
        p = first_frame_conditioning_p if conditioning_mode == "single_frame" else 0.0
        strategy = SingleFrameStrategy(
            deps=deps,
            schedule=schedule,
            first_frame_conditioning_p=p,
            first_frame_conditioning_strength=first_frame_conditioning_strength,
            device=device,
            dtype=dtype,
            fps=fps,
        )
    inputs = strategy.prepare_training_inputs(batch, rng=rng)
    with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        # Thread the batch's audio Modality through (Phase 9 a2v item 5): for every non-a2v strategy
        # ``inputs.audio`` is ``None`` (byte-identical to the prior hardcoded ``audio=None``); for
        # ``audio_to_video`` it is the frozen driving-audio Modality the A2VStrategy built, so the
        # cross-modal audio→video attention actually receives the conditioning.
        video_pred, _ = model(video=inputs.video, audio=inputs.audio, perturbations=None)
        loss = strategy.compute_loss(inputs, video_pred)

    return loss
