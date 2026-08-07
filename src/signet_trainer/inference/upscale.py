"""inference.upscale — the two-stage spatial-upscaler render path (Phase 4, INFR-01; D-DEPTH-1).

A RUNNABLE toggle, DEFAULT OFF: single-stage (``inference.sampler.run_sampler``) stays the
base-vs-LoRA grid default (correctness-first). This module wraps LTX-2.3's native two-stage
text-to-video path — distilled base + 2x spatial upscaler — behind ``run_two_stage`` so
``fns.py::sample`` can branch on ``config.validation.two_stage_upscale``.

This is the correct TEXT-TO-VIDEO two-stage pipeline (``ltx_pipelines.ti2vid_two_stages.
TI2VidTwoStagesPipeline``), NOT ``ICLoraPipeline`` — the IC-LoRA / video-to-video path is
Phases 5-7 (RESEARCH Alternatives, line 88). Precedent: the prior project ran the two-stage path on an
**H100 with ``offload_mode="CPU"``** because distilled base + spatial upscaler + Gemma can
exceed A100-80GB (RESEARCH Pitfall 6). So gate any LIVE two-stage run on VRAM headroom
(H100 + CPU offload, small smoke resolution) — do NOT let it define the A100 baseline.

CRITICAL — Anti-Pattern 6 / import confinement (mirrors ``models/loader.py``):
    Module top imports stdlib + typing ONLY. ``ltx_pipelines`` / ``ltx_core`` / ``modal`` are
    NEVER imported at module top — the heavy
    ``from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline`` is deferred INSIDE
    ``run_two_stage``. This keeps the module CPU-importable so ``tests/test_upscale_toggle.py``
    can unit-test the toggle wiring WITHOUT ltx-pipelines installed (it is only added to
    ``gpu_image`` — Modal-side — by 04-05 Task 1).

DEFERRED live proof (D-RUN-1): the exact ``TI2VidTwoStagesPipeline`` constructor + call
signature is MEDIUM-confidence (README pipeline name verified at the pinned SHA; exact API
re-validated only when the toggle is exercised on a gated GPU run — RESEARCH line 463). The
structural wiring lands here; the metered upscaler run rides a future gated invocation. This
plan makes the path EXIST and CPU-verifies the toggle; it commits NO metered spend.

Port source:
    Lightricks/LTX-2 ``packages/ltx-pipelines`` @ pinned SHA d6053703 —
    ``ti2vid_two_stages.TI2VidTwoStagesPipeline`` (RESEARCH lines 81, 156).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only; never executed — keeps the module importable without ltx_pipelines.
    from ltx_trainer.validation_sampler import GenerationConfig


def run_two_stage(
    components: Any,
    transformer: Any,
    cfg: "GenerationConfig",
    *,
    distilled_ckpt_path: str,
    upscaler_ckpt_path: str,
    distilled_lora_path: str | None = None,
    device: str = "cuda",
    offload_mode: str = "CPU",
) -> Any:
    """Run the two-stage (distilled base + 2x spatial upscaler) render for one config (Modal-side).

    DEFAULT-OFF toggle path (``config.validation.two_stage_upscale``): constructs the
    ltx-pipelines ``TI2VidTwoStagesPipeline`` from the distilled base + spatial-upscaler
    weights (downloaded by ``download_weights(two_stage=True)`` — 04-05 Task 1) and the loaded
    ``components`` (video VAE decoder + Gemma text encoder / embeddings processor), then returns
    the upscaled video in the SAME ``(video, audio)`` shape as ``run_sampler`` so ``sample()``'s
    grid emission stays identical whichever branch runs.

    Heavy imports are function-local (Anti-Pattern 6): this only ever runs on the mounted-weights
    GPU container, never on Windows/CI. ``offload_mode="CPU"`` mirrors the prior project's H100 precedent
    (Pitfall 6) — the two-stage stack can exceed A100-80GB, so scope a live run to an H100 tier
    with CPU offload and a small smoke resolution.

    Args:
        components: The loaded ``LtxModelComponents`` (video VAE decoder ON, Gemma text encoder +
            ``embeddings_processor`` attached — the decoder-on inference bundle from 04-03).
        transformer: The (optionally PEFT-wrapped) transformer to render with — matches the
            base-vs-LoRA column ``sample()`` is currently emitting.
        cfg: The ``GenerationConfig`` built by ``inference.sampler.build_generation_config``
            (prompt / seed / canonical LTX params).
        distilled_ckpt_path: Path to ``ltx-2.3-22b-distilled-1.1.safetensors`` (WEIGHTS_DIR).
        upscaler_ckpt_path: Path to ``ltx-2.3-spatial-upscaler-x2-1.1.safetensors`` (WEIGHTS_DIR).
        distilled_lora_path: Optional path to ``ltx-2.3-22b-distilled-lora-384-1.1.safetensors``.
        device: Torch device string ("cuda" Modal-side).
        offload_mode: Pipeline CPU-offload mode ("CPU" per the Pitfall-6 precedent).

    Returns:
        ``(video_tensor, audio_or_None)`` — same unpack shape as ``run_sampler`` so ``sample()``
        can branch structurally without changing its grid-emission code.

    Note:
        The exact ``TI2VidTwoStagesPipeline`` constructor / call kwargs are MEDIUM-confidence and
        proven on the deferred gated run (D-RUN-1); see the module docstring. Heavy imports stay
        function-local here on purpose — do NOT hoist them to module top (import-confinement).
    """
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # noqa: PLC0415

    # Construct the distilled-base + spatial-upscaler two-stage pipeline from the loaded
    # components + the toggle-gated weights. The distilled base drives the low-res pass; the
    # spatial upscaler runs the 2x refine. Exact kwargs re-validated on the deferred gated run.
    pipeline = TI2VidTwoStagesPipeline(
        transformer=transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=components.video_vae_encoder,
        text_encoder=components.text_encoder,
        embeddings_processor=getattr(components, "embeddings_processor", None),
        distilled_checkpoint_path=distilled_ckpt_path,
        upscaler_checkpoint_path=upscaler_ckpt_path,
        distilled_lora_path=distilled_lora_path,
        offload_mode=offload_mode,
        device=device,
    )

    video = pipeline.generate(config=cfg, device=device)

    # Normalize to run_sampler's (video, audio) unpack shape (audio is out of scope — video-only).
    if isinstance(video, tuple):
        return video
    return video, None
