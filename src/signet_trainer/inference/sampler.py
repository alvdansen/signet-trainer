"""inference.sampler — the ported LTX-2.3 validation sampler (Phase 4, INFR-01).

Two layers, mirroring ``models/loader.py``'s heavy-import-local discipline:

  * ``build_generation_config(config, prompt, seed)`` — PURE/CPU except the one deferred
    ``GenerationConfig`` import. Maps ``SignetConfig.validation`` onto the RE-VALIDATED LTX
    canonical sampling kwargs (Euler + STG, guidance 3.0, frames 8k+1, W/H %32,
    generate_audio=False). The mapping itself is factored into ``_generation_kwargs`` — a
    fully importable helper returning a plain kwargs dict — so ``tests/test_gen_config.py``
    can assert the canonical values on Windows/CI WITHOUT ltx_trainer installed.

  * ``run_sampler(components, transformer, cfg, device)`` — Modal-side ONLY. Constructs the
    ltx-trainer ``ValidationSampler`` from loaded components and calls ``.generate(...)``.

Phase 9 (INPAINT, GATE-SPEC-inpaint-a2v rev 2) adds the MASK-CONDITION render branch in the
same two-layer shape: pure/CPU helpers (``plan_mask_condition`` / ``masked_render_latent_grid``
/ ``build_token_denoise_mask`` / ``pin_keep_tokens`` — the CPU-shape gates in
``tests/test_sampler_mask_condition.py``) + the Modal-side ``run_mask_condition_sampler``, an
ic_lora-pipeline-CLASS port over the ``ValidationSampler`` internals (the upstream sampler at
the pinned SHA exposes only ``condition_image``; its per-token denoise-mask machinery EXISTS
but has no config surface, so this branch drives the internals directly — the same documented
A3 pattern ``inference/multi_condition.py`` uses).

CRITICAL — Anti-Pattern 6 / Pitfall 3:
    Module top imports stdlib ONLY (``from __future__`` + typing). ``ltx_trainer`` /
    ``ltx_core`` / ``modal`` are NEVER imported at module top — the heavy
    ``from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler`` is
    deferred INSIDE the functions that actually run Modal-side. This keeps the module
    CPU-importable for the Wave-0 gates.

    There is NO ``decode_timestep`` in the mapped kwargs: ``GenerationConfig`` has no such
    field in the ValidationSampler port target (RESEARCH Pitfall 3). The canonical params
    are the LTX-re-validated set, NOT the Wan-tuned (UniPC / shift=4.0 / frames=33 /
    guidance=5 / steps=50) set.

Port source (read via ``gh``):
    enochiatron scripts/infer/generate.py + Lightricks/LTX-2 ltx-trainer
    ``validation_sampler`` at pinned SHA d6053703 (RESEARCH Pattern 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only; never executed — keeps the module importable without ltx_trainer.
    from ltx_trainer.validation_sampler import GenerationConfig

    from signet_trainer.config.schema import SignetConfig


def _generation_kwargs(
    config: "SignetConfig",
    prompt: str,
    seed: int,
    condition_image: "torch.Tensor | None" = None,
) -> dict[str, Any]:
    """Map ``SignetConfig.validation`` onto the LTX canonical ``GenerationConfig`` kwargs.

    Pure/CPU — no ltx_trainer, no torch, no filesystem. Returns a plain dict so the
    Wave-0 param-mapping gate (``tests/test_gen_config.py``) can assert every canonical
    value without instantiating the native ``GenerationConfig``.

    All defaults flow from ``ValidationConfig`` (Task 1), which carries the RE-VALIDATED
    LTX-2.3 params (D-PARAMS-1). ``num_frames`` rides ``validation.frame_count`` (an 8k+1
    value); ``width``/``height`` ride the validation sampling fields (both %32). Audio is
    OFF (video-only project); there is deliberately no ``decode_timestep`` / ``shift`` /
    scheduler key (RESEARCH Pitfall 3).

    ``condition_image`` is the optional single-frame reference (a ``[C,H,W]`` tensor in
    ``[0,1]``, or ``None`` — the ref-OFF control default; SC#2). It rides
    ``GenerationConfig.condition_image`` (CONTRADICTION #1: this is the ported
    ``ValidationSampler`` field with implicit first-frame index and hard-clean conditioning,
    NOT the diffusers first-frame condition classes named in CONTEXT/ROADMAP, which are absent
    from this stack). When ``None`` the kwargs reproduce the exact ref-OFF control set; the
    sampler VAE-encodes / resizes / center-crops the image internally, so no pre-encoding and
    no exact-size requirement here.
    """
    v = config.validation
    return {
        "prompt": prompt,
        "negative_prompt": "",
        "seed": seed,
        "num_inference_steps": v.num_inference_steps,
        "guidance_scale": v.guidance_scale,
        "stg_scale": v.stg_scale,
        "stg_blocks": list(v.stg_blocks),
        "stg_mode": v.stg_mode,
        "frame_rate": v.frame_rate,
        "num_frames": v.frame_count,
        "width": v.width,
        "height": v.height,
        "generate_audio": False,
        "condition_image": condition_image,
    }


def build_generation_config(
    config: "SignetConfig",
    prompt: str,
    seed: int,
    condition_image: "torch.Tensor | None" = None,
) -> "GenerationConfig":
    """Build the native ltx-trainer ``GenerationConfig`` from a ``SignetConfig``.

    Thin wrapper over ``_generation_kwargs`` — the only non-pure line is the function-local
    ``GenerationConfig`` import (deferred so the module stays CPU-importable; see docstring).
    Invoked Modal-side inside the sampling path. ``condition_image`` (default ``None`` =
    ref-OFF control) threads straight through to ``GenerationConfig.condition_image``.
    """
    from ltx_trainer.validation_sampler import GenerationConfig  # noqa: PLC0415

    return GenerationConfig(
        **_generation_kwargs(config, prompt=prompt, seed=seed, condition_image=condition_image)
    )


def build_validation_sampler(components: Any, transformer: Any) -> Any:
    """Construct the ltx-trainer ``ValidationSampler`` from loaded components (Modal-side ONLY).

    The single reuse point for BOTH the single-frame ``run_sampler`` and the multi-condition
    ``inference.multi_condition.run_multi_condition_sampler`` (Seam E): video-only (audio decoder
    and vocoder are ``None``), (optionally PEFT-wrapped) ``transformer``. Factored out so the
    multi-condition path reuses the exact same sampler construction WITHOUT re-specifying it and
    without changing ``run_sampler``'s behaviour.

    Heavy import is function-local (Anti-Pattern 6): runs on the mounted-weights GPU container only.
    """
    from ltx_trainer.validation_sampler import ValidationSampler  # noqa: PLC0415

    return ValidationSampler(
        transformer,
        vae_decoder=components.video_vae_decoder,
        vae_encoder=components.video_vae_encoder,
        text_encoder=components.text_encoder,
        embeddings_processor=getattr(components, "embeddings_processor", None),
        audio_decoder=None,
        vocoder=None,
    )


def run_sampler(
    components: Any,
    transformer: Any,
    cfg: "GenerationConfig",
    device: str = "cuda",
    cached_embeddings: Any | None = None,
) -> Any:
    """Run the ltx-trainer ``ValidationSampler`` for one ``GenerationConfig`` (Modal-side ONLY).

    Constructs the sampler via ``build_validation_sampler`` and returns
    ``sampler.generate(config=cfg, device=device)`` — a decoded video tensor. Ports RESEARCH
    Pattern 1 (generate.py:955-1014).

    Two 06-09 carry-forward VRAM guards (the A100-80GB two-phase discipline, now generalized from
    the multi_frame path to EVERY single-stage render — protects the single_frame + base-vs-LoRA
    branches that still called this raw):

      * ``cached_embeddings`` — when provided, assign it onto the ltx-native
        ``GenerationConfig.cached_embeddings`` field ("avoids loading Gemma" at the pinned SHA
        d6053703). ``ValidationSampler._get_prompt_embeddings`` then short-circuits without ever
        touching Gemma, so the caller can two-phase-load (pre-encode prompts, delete Gemma, load
        the 22B transformer) and the render loop never holds both. ``GenerationConfig`` is a plain
        dataclass, so post-build assignment is the minimal thread-through (mirrors
        ``run_multi_condition_sampler``; no ``_generation_kwargs`` schema change).

      * ``torch.no_grad()`` around ``generate`` — the ltx sampler internals (VAE encode of any
        ``condition_image``, the chained denoise transformer forwards, the decode) carry NO
        no_grad of their own; called raw each forward retains an autograd graph (06-09 run-7: the
        Gemma flavour cost ~20-25GB/encode; chained 22B denoise steps are worse). Inference never
        needs a gradient, so this is a pure-safety wrap that changes no output.

    Heavy imports are function-local (Anti-Pattern 6): this runs on the mounted-weights GPU
    container, never on Windows/CI.
    """
    import torch  # noqa: PLC0415 — GPU-side; function-local keeps the module CPU-importable

    if cached_embeddings is not None:
        cfg.cached_embeddings = cached_embeddings
    sampler = build_validation_sampler(components, transformer)
    with torch.no_grad():
        return sampler.generate(config=cfg, device=device)


# ==================================================================================================
# Phase 9 (INPAINT) — the mask-condition masked-render branch (GATE-SPEC-inpaint-a2v rev 2, item 6)
# ==================================================================================================
#
# Mask polarity ON THE ENCODED TENSOR (GATE-SPEC / [precedent] — the SAME contract
# conditioning/inpaint.py trains under):
#     1.0 = KEEP/context  (clean latent, timestep 0, excluded from loss)
#     0.0 = GENERATE      (noised / denoised from noise)
# The sampler-side denoise mask is the INVERSE view of the same bits: upstream
# ``ValidationSampler`` semantics are ``denoise_mask = 1`` -> denoise, ``0`` -> keep clean
# (validation_sampler.py:377-383 ``_apply_image_conditioning`` writes 0.0 on conditioned tokens;
# :571-577 the per-step copy-back ``denoised * denoise_mask + clean_latent * (1 - denoise_mask)``;
# the ``GaussianNoiser`` noises ``noise * denoise_mask + latent * (1 - denoise_mask)``). So
# ``denoise_mask = 1.0 - keep`` reproduces training's where-semantics EXACTLY:
#     noisy     = where(mask > 0.5, clean_latent, noisy)   <->  noiser at denoise_mask = 1 - keep
#     timesteps = where(mask > 0.5, 0, sigma)              <->  sigma * denoise_mask
# Token order: frame-major flatten, ``token = f*H_lat*W_lat + h*W_lat + w`` — the exact ordering
# of ``conditioning/mask_ops.py::latent_frame_token_span`` (patch_size=1, one token per latent
# cell). ``reshape(F*H*W)`` on a row-major ``[F, H, W]`` tensor IS that order; the CPU gate
# proves it against hand-computed indices.


def plan_mask_condition(condition: Any) -> tuple[str, str]:
    """Normalize one ``mask`` validation-sample condition to ``(video_path, mask_path)`` (pure/CPU).

    Accepts BOTH the real ``config.schema.MaskCondition`` sub-model (attribute access) and a plain
    ``dict`` (test / dry-run shape) — the same dual-shape convention as
    ``multi_condition._item_field``. Fail-fast (D-NOHARDCODE / lean-guard doctrine): a wrong
    ``type`` discriminator or a missing/empty ``video``/``mask`` path raises ``ValueError`` naming
    the offending field BEFORE any metered work, never a silent mis-render.

    Both returned paths are the config's Volume-relative strings, untouched — resolution under the
    Volume mount (and the traversal guard) happens Modal-side in ``run_mask_condition_sampler``.
    """

    def _field(name: str) -> Any:
        if isinstance(condition, dict):
            return condition.get(name)
        return getattr(condition, name, None)

    cond_type = _field("type")
    if cond_type is not None and cond_type != "mask":
        raise ValueError(
            f"mask condition has type={cond_type!r}: this branch renders ONLY the 'mask' condition "
            f"kind (MaskCondition.type — future kinds widen the discriminated union, they do not "
            f"reuse this branch)."
        )
    video = _field("video")
    mask = _field("mask")
    if not isinstance(video, str) or not video:
        raise ValueError(
            "mask condition is missing a non-empty 'video' path: expected the MaskCondition shape "
            "{type: 'mask', video: <Volume-relative clip>, mask: <Volume-relative mask>} "
            f"(got video={video!r})."
        )
    if not isinstance(mask, str) or not mask:
        raise ValueError(
            "mask condition is missing a non-empty 'mask' path: expected the MaskCondition shape "
            "{type: 'mask', video: <Volume-relative clip>, mask: <Volume-relative mask>} "
            f"(got mask={mask!r})."
        )
    return video, mask


def masked_render_latent_grid(height: int, width: int, num_frames: int) -> tuple[int, int, int]:
    """Render-geometry gate + latent grid for a masked render: ``(lat_f, lat_h, lat_w)`` (pure/CPU).

    Enforces the INPAINT dims rule — ``H % 64 == 0``, ``W % 64 == 0``, ``frames % 8 == 1`` —
    STRICTER than the sampler-wide %32 rule, inpaint-mode only ([precedent] prior-project
    HANDOFF-2026-06-30-NIGHT "÷64 required for inpaint; validated by the smoke"; GATE-SPEC rev 2).
    Then derives the latent grid the encoded mask must match: ``H_lat = H // 32``,
    ``W_lat = W // 32``, ``F_lat = (F - 1) // 8 + 1`` (causal temporal — [canonical] upstream
    ``VideoLatentShape.from_pixel_shape`` @ pinned SHA d6053703).

    Pure ints (no torch, no filesystem) so the CPU gate can assert the rule and the grid math on
    Windows/CI, and so a %64 violation dies BEFORE any decode/VAE work on a metered container.
    """
    problems: list[str] = []
    if height % 64 != 0:
        problems.append(f"height={height} (need height % 64 == 0)")
    if width % 64 != 0:
        problems.append(f"width={width} (need width % 64 == 0)")
    if num_frames % 8 != 1:
        problems.append(f"num_frames={num_frames} (need num_frames % 8 == 1)")
    if problems:
        raise ValueError(
            f"masked render dims violate the inpaint ÷64 rule: {'; '.join(problems)}. Inpaint-mode "
            f"renders require H%64==0, W%64==0, frames%8==1 — STRICTER than the video-wide %32 rule "
            f"([precedent]: all validated inpaint dims are ÷64; 1280x704 and 512x384 are "
            f"clean)."
        )
    return (num_frames - 1) // 8 + 1, height // 32, width // 32


def build_token_denoise_mask(latent_mask: Any, *, lat_f: int, lat_h: int, lat_w: int) -> Any:
    """Encoded ``[F_lat, H_lat, W_lat]`` KEEP-mask -> per-token sampler denoise mask ``[1, seq, 1]``.

    Input: the signet-native encoded mask tensor (one ``.pt`` per sample — float32, values
    ``{0., 1.}``, 1.0 = KEEP/context, 0.0 = GENERATE; thresholded at 0.5 at ENCODE time). Output:
    the upstream ``ValidationSampler`` denoise mask — float32 ``[1, seq_len, 1]`` (the exact shape
    of ``_generate_with_reference``'s ``ref_denoise_mask``, validation_sampler.py:279), where
    ``0.0`` pins a token clean (timestep 0, no noise) and ``1.0`` denoises it from noise. The
    polarity INVERSION (``denoise = mask <= 0.5``) is the whole point — see the section banner.

    Binarizes at ``> 0.5`` (defence-in-depth restatement of the encode-time threshold — the same
    "binarize to match inference" doctrine ``conditioning/inpaint.py`` ports from the upstream
    ``FlexibleStrategy`` oracle) and flattens FRAME-MAJOR (row-major ``reshape`` on ``[F, H, W]``
    == ``token = f*H_lat*W_lat + h*W_lat + w`` == ``mask_ops.latent_frame_token_span`` order).

    Fail-fast on any grid mismatch: a mask encoded for different dims silently mis-paints WRONG
    tokens (the #1 GATE-SPEC risk — "mask->token alignment + polarity"), so the shape check names
    both grids and dies before any GPU work. Pure tensor methods only (no imports) — CPU-testable.
    """
    if latent_mask.dim() != 3:
        raise ValueError(
            f"encoded mask must be a 3-dim [F_lat, H_lat, W_lat] tensor, got "
            f"{latent_mask.dim()}-dim shape {tuple(latent_mask.shape)} (the per-sample .pt contract "
            f"is float32 [F_lat, H_lat, W_lat] in {{0., 1.}})."
        )
    if tuple(latent_mask.shape) != (lat_f, lat_h, lat_w):
        raise ValueError(
            f"encoded mask grid {tuple(latent_mask.shape)} does not match the render's latent grid "
            f"({lat_f}, {lat_h}, {lat_w}) (= ((F-1)//8+1, H//32, W//32) for the requested render "
            f"dims): a mismatched mask would silently pin WRONG tokens (GATE-SPEC top risk — "
            f"mask->token alignment). Re-encode the mask at the render dims."
        )
    # KEEP (mask > 0.5) -> denoise 0.0; GENERATE (mask <= 0.5) -> denoise 1.0. `(mask <= 0.5)` is
    # the same comparison training uses for the loss side (`video_loss_mask = mask <= 0.5`).
    denoise = (latent_mask <= 0.5).float()  # float32
    return denoise.reshape(1, lat_f * lat_h * lat_w, 1)


def pin_keep_tokens(denoised: Any, clean_latent: Any, denoise_mask: Any) -> Any:
    """The per-step keep-token pinning (copy-back) formula, stated CPU-verifiably (pure math).

    ``denoised * denoise_mask + clean_latent * (1 - denoise_mask)`` — VERBATIM the conditioning
    copy-back inside the upstream ``ValidationSampler._run_denoising`` at the pinned SHA
    (validation_sampler.py:571-577, "Apply conditioning mask (keep conditioned tokens clean)").
    With the binary ``denoise_mask = 1 - keep`` from ``build_token_denoise_mask`` this is
    EXACTLY training's ``noisy = where(mask > 0.5, clean_latent, noisy)`` where-substitution
    (``conditioning/inpaint.py`` / GATE-SPEC strategy semantics) — keep-tokens stay pinned to the
    clean latents every step while generate-tokens denoise from noise.

    The LIVE path does NOT call this helper: ``run_mask_condition_sampler`` delegates to
    ``sampler._run_denoising``, which applies this same formula internally each step (reuse, not
    replicate). It exists so the CPU gate (``tests/test_sampler_mask_condition.py``) can PROVE the
    inversion + pinning math equals the training where-semantics on synthetic tensors — the
    executable statement of the contract, same factoring rationale as ``_generation_kwargs``.
    """
    return denoised * denoise_mask + clean_latent * (1 - denoise_mask)


def _load_latent_mask(resolved_mask_path: str, *, lat_f: int, lat_h: int, lat_w: int) -> Any:
    """Load one mask source to a float32 ``[F_lat, H_lat, W_lat]`` encoded tensor (Modal-side).

    ``.pt`` (the canonical signet-native encoded form — ``video_masks/`` contract: float32
    ``[F_lat, H_lat, W_lat]``, values ``{0., 1.}``) loads directly (``weights_only=True`` — the
    mask file is data, never code); the grid params are validated downstream by
    ``build_token_denoise_mask``. Any other source (a ``<stem>_mask.mp4`` polarity render / a
    PNG dir / a single image) routes through the signet-native mask encoder ``data/mask_encode.py``
    (``read_mask_frames`` -> ``encode_mask_pixels`` at the render's latent grid) — IMPORTED,
    never duplicated (the GATE-SPEC canonical-encoder-exception doctrine: ONE encode
    implementation — downsample /32, causal (F-1)//8+1, threshold 0.5).
    """
    import torch  # noqa: PLC0415 — GPU-side; function-local keeps the module CPU-importable

    if resolved_mask_path.endswith(".pt"):
        mask = torch.load(resolved_mask_path, map_location="cpu", weights_only=True)
        if not isinstance(mask, torch.Tensor):
            raise ValueError(
                f"encoded mask {resolved_mask_path!r} did not load as a tensor (got "
                f"{type(mask).__name__}): the video_masks contract is one float32 "
                f"[F_lat, H_lat, W_lat] tensor per .pt."
            )
        return mask.to(torch.float32)
    # Not a pre-encoded .pt -> the signet-native mask encoder owns pixel->latent-grid semantics.
    from signet_trainer.data.mask_encode import (  # noqa: PLC0415
        encode_mask_pixels,
        read_mask_frames,
    )

    pixel_f = (lat_f - 1) * 8 + 1  # the target pixel frame count for this latent grid
    mask_pixels = read_mask_frames(resolved_mask_path, expected_frames=pixel_f)
    return encode_mask_pixels(mask_pixels, lat_f, lat_h, lat_w)


def run_mask_condition_sampler(
    components: Any,
    transformer: Any,
    config: "SignetConfig",
    video_path: str,
    mask_path: str,
    device: str = "cuda",
    cached_embeddings: Any | None = None,
    prompt: str | None = None,
) -> Any:
    """Render one MASKED test video through the (LoRA-wrapped) transformer (Modal-side ONLY).

    The Phase-9 masked-render validation branch (GATE-SPEC rev 2 item 6): VAE-encode the held-out
    input clip to CLEAN latents, pin the mask-KEEP tokens to them (timestep 0, excluded from
    denoising — the copy-back mirrors training every step), and denoise the mask-GENERATE region
    from noise, so a trained inpaint LoRA regenerates ONLY the masked region in context.

    This is the ic_lora-pipeline-CLASS port the pre-build audit called for: the upstream
    ``ValidationSampler`` at the pinned SHA (d6053703) exposes only ``condition_image`` — its
    per-token denoise-mask machinery EXISTS (the ``[B, seq, 1]`` denoise_mask through
    ``GaussianNoiser`` -> ``_run_denoising`` -> copy-back) but has NO config surface. So this
    branch drives the internals directly, the documented A3 pattern ``multi_condition.py``
    established: ``_get_prompt_embeddings`` FIRST (sequential-VRAM ordering), then tools + state,
    conditioning, ``GaussianNoiser``, ``_run_denoising``, ``clear_conditioning`` -> ``unpatchify``
    -> ``_decode_video``. The state construction mirrors ``_generate_with_reference``'s combined-
    state build (validation_sampler.py:278-287) with the per-token spatial mask in place of the
    ref-prefix zeros — same machinery, no sequence concat (the mask condition is in-place).

    Inference landmines ([precedent] prior-project 07-01/07-02, GATE-SPEC rev 2 — likeness-critical):

      * SINGLE-PASS at the requested res — NO two-stage upscale. ``validation.two_stage_upscale``
        must be False (fail-fast below): the upstream two-pass path hard-codes stage-2
        ``loras=()`` (the adapter would be silently dropped) and ``skip_stage_2`` decodes at
        half-res. The decode below is the plain single-stage ``_decode_video`` at cfg dims.
      * Dev-family base ONLY (the fused In-Outpainting dev base), NEVER distilled — base-variant
        selection is the loader/fns seam (D-7-BASEVAR); documented here, enforced there.
      * NO resize/crop of the input clip: the mask is aligned 1:1 to it (same F/H/W). A center
        crop would silently misalign mask and pixels, so a dims mismatch FAILS instead (unlike
        the reference-video path, which may resize because nothing is aligned to it).

    Two 06-09 carry-forward VRAM guards, identical to ``run_sampler``:
      * ``cached_embeddings`` -> ``GenerationConfig.cached_embeddings`` ("avoids loading Gemma");
        ``_get_prompt_embeddings`` short-circuits so the render loop never holds Gemma + the 22B.
      * ``torch.no_grad()`` around EVERY forward (VAE encode, denoise steps, decode) — the ltx
        sampler internals carry no no_grad of their own.

    Args:
        components / transformer: loaded components + the (PEFT-wrapped) transformer, as for
            ``run_sampler``. Requires ``components.video_vae_encoder`` (the input clip is encoded).
        config: the full ``SignetConfig`` (canonical sampling params ride ``validation``).
        video_path / mask_path: ONE mask condition's Volume-relative paths (the
            ``plan_mask_condition`` output for a ``validation.samples[*].conditions[*]`` entry) —
            resolved under the checkpoints mount here, traversal-guarded first (T-07-05-01 class).
        prompt: this render's prompt (``ValidationSample.prompt``). ``None`` falls back to
            ``validation.prompts[0]`` (the siblings' convention).

    Returns:
        ``(video[C, F, H, W] in [0, 1], None)`` — the same 2-tuple unpack shape as
        ``run_sampler`` / ``run_reference_video_sampler`` (video-only; audio is out of scope).
    """
    from dataclasses import replace  # noqa: PLC0415 — stdlib; grouped with the GPU-side imports

    import torch  # noqa: PLC0415 — GPU-side; function-local keeps the module CPU-importable

    from ltx_core.components.noisers import GaussianNoiser  # noqa: PLC0415

    from signet_trainer.config.validators import validate_volume_relative_path  # noqa: PLC0415
    from signet_trainer.inference.reference_video import _decode_reference_video  # noqa: PLC0415

    # Fail-fast: the masked render is SINGLE-PASS at the requested res ([precedent] prior-project
    # 07-01/07-02 — two-pass drops the LoRA on stage 2 and skip_stage_2 halves the res; both are
    # silent likeness killers on an inpaint validation render).
    if config.validation.two_stage_upscale:
        raise ValueError(
            "validation.two_stage_upscale is True but the masked render is SINGLE-PASS ONLY "
            "(GATE-SPEC rev 2 inference landmines: likeness-critical output decodes once at the "
            "requested res; the upstream two-pass path hard-codes stage-2 loras=() and "
            "skip_stage_2 decodes at half-res). Set validation.two_stage_upscale: false."
        )
    if getattr(components, "video_vae_encoder", None) is None:
        raise ValueError(
            "mask-condition rendering requires components.video_vae_encoder (the held-out clip is "
            "VAE-encoded to clean latents) — load with the encoder, as the ic_lora sample branch "
            "does."
        )

    # T-07-05-01 class (Tampering — operator-authored path traversal): reject absolute or
    # '..'-escaping paths BEFORE any read, then resolve under the checkpoints mount — the exact
    # convention of run_reference_video_sampler.
    validate_volume_relative_path(video_path, field="mask condition video path")
    validate_volume_relative_path(mask_path, field="mask condition mask path")
    from signet_trainer.modal.app import CHECKPOINTS_DIR  # noqa: PLC0415

    resolved_video_path = str(CHECKPOINTS_DIR / video_path)
    resolved_mask_path = str(CHECKPOINTS_DIR / mask_path)

    # Normalize device the same way ValidationSampler.generate does (str -> torch.device).
    device = torch.device(device) if isinstance(device, str) else device

    # The canonical GenerationConfig (Euler + STG, guidance from config) — the SAME re-validated
    # params every sibling branch uses. condition_image stays None: the spatial mask carries the
    # conditioning, not the sampler's single first-frame path.
    v = config.validation
    render_prompt = prompt if prompt is not None else (v.prompts[0] if v.prompts else "")
    cfg = build_generation_config(
        config, prompt=render_prompt, seed=v.seed, condition_image=None
    )
    # Two-phase VRAM (06-09, generalized): post-build assignment of the ltx-native cached
    # embeddings — _get_prompt_embeddings then short-circuits without touching Gemma.
    if cached_embeddings is not None:
        cfg.cached_embeddings = cached_embeddings

    # CPU geometry gate BEFORE any decode/VAE forward: inpaint ÷64 dims + the latent grid the
    # encoded mask must match.
    lat_f, lat_h, lat_w = masked_render_latent_grid(cfg.height, cfg.width, cfg.num_frames)

    # Decode the held-out clip ([F, C, H, W] in [0, 1] — PyAV, same reader as the V2V branch) and
    # REQUIRE exact render dims: the mask is aligned 1:1 to this clip, so any resize/crop would
    # silently misalign them (see landmines above) — mismatch fails loud instead.
    clip = _decode_reference_video(resolved_video_path)
    clip_f, _clip_c, clip_h, clip_w = clip.shape
    if (clip_f, clip_h, clip_w) != (cfg.num_frames, cfg.height, cfg.width):
        raise ValueError(
            f"masked-render input clip {video_path!r} decodes to F={clip_f}, H={clip_h}, "
            f"W={clip_w} but the render requests F={cfg.num_frames}, H={cfg.height}, "
            f"W={cfg.width}: the mask is aligned 1:1 to the clip, so the clip must be staged at "
            f"EXACTLY the render dims (no resize/crop — it would silently misalign the mask)."
        )

    # Load the encoded mask and build the per-token denoise mask (polarity inversion + frame-major
    # flatten — the CPU-gated math). Grid mismatch dies inside build_token_denoise_mask.
    latent_mask = _load_latent_mask(resolved_mask_path, lat_f=lat_f, lat_h=lat_h, lat_w=lat_w)
    token_denoise_mask = build_token_denoise_mask(
        latent_mask.to(torch.float32), lat_f=lat_f, lat_h=lat_h, lat_w=lat_w
    ).to(device)

    sampler = build_validation_sampler(components, transformer)

    # no_grad wraps EVERY inference forward (06-09 run-7 landmine, generalized — same rationale
    # as multi_condition.py): VAE encode, chained denoise forwards, decode.
    with torch.no_grad():
        # PROMPT EMBEDDINGS FIRST — the source's sequential-VRAM ordering (_generate_standard runs
        # _get_prompt_embeddings BEFORE any latent/VAE work). With cached_embeddings set this is a
        # pure cache read; without a cache it falls back to the live Gemma encode — only safe if
        # the caller manages VRAM residency.
        v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = sampler._get_prompt_embeddings(cfg, device)

        # Target tools + initial state (positions + shapes for the render grid). `dtype` is
        # REQUIRED at the pinned SHA — bfloat16 is the LTX-2.3 compute dtype every call site uses
        # (validation_sampler.py:195; confirmed at the 06-09 GPU run).
        video_tools = sampler._create_video_latent_tools(cfg)
        initial_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)

        # VAE-encode the input clip to CLEAN patchified latents — reuse the sampler's own
        # _encode_video ([B, C, F, H, W] in [-1, 1] -> [1, seq, C] bf16; encoder is moved
        # GPU->CPU internally, the sampler's own VRAM discipline). Positions are discarded: the
        # clip occupies EXACTLY the target grid, so initial_state's own positions are correct
        # (unlike the ref-prefix path, which builds a second position block to concat).
        clip_m11 = clip.permute(1, 0, 2, 3).unsqueeze(0) * 2.0 - 1.0  # [1, C, F, H, W] in [-1, 1]
        clean_patched, _ref_positions = sampler._encode_video(clip_m11, cfg.frame_rate, device)
        if clean_patched.shape[1] != lat_f * lat_h * lat_w:
            raise ValueError(
                f"VAE-encoded clip patchified to seq_len={clean_patched.shape[1]} but the render "
                f"grid expects {lat_f * lat_h * lat_w} (= {lat_f}*{lat_h}*{lat_w}): encode/grid "
                f"drift — the mask->token alignment contract would be violated."
            )

        # The conditioned clean state — the in-place analogue of _generate_with_reference's
        # combined-state build (validation_sampler.py:278-287): clean video latents everywhere
        # (latent AND clean_latent, as _apply_image_conditioning writes both), the per-token
        # spatial denoise mask in place of the all-ones default. KEEP tokens: denoise_mask 0.0 ->
        # never noised (GaussianNoiser), timestep 0 (sigma * denoise_mask), pinned to clean every
        # step (the :571-577 copy-back). GENERATE tokens: denoise_mask 1.0 -> pure noise at
        # sigma[0], denoised normally.
        clean_state = replace(
            initial_state,
            latent=clean_patched.to(initial_state.latent.dtype),
            clean_latent=clean_patched.to(initial_state.clean_latent.dtype),
            denoise_mask=token_denoise_mask,
        )

        # Noise (mask-scaled: generate-tokens only), denoise, decode — the standard-branch tail.
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        noiser = GaussianNoiser(generator=generator)
        video_state = noiser(latent_state=clean_state, noise_scale=1.0)

        video_state, _audio_state = sampler._run_denoising(
            config=cfg,
            video_state=video_state,
            audio_state=None,
            video_clean_state=clean_state,
            audio_clean_state=None,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            device=device,
        )
        # Decode SINGLE-PASS at the requested res (landmine #1). clear_conditioning is a no-op
        # slice here (no extra tokens were appended — the mask condition is in-place), kept for
        # byte-parity with the _generate_standard tail.
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        video_output = sampler._decode_video(video_state, device, cfg.tiled_decoding)
    return video_output, None


# ==================================================================================================
# Phase 9 (AUDIO-TO-VIDEO) — the driving-audio validation render branch (GATE-SPEC rev 2 item 7)
# ==================================================================================================
#
# a2v renders VIDEO driven by an input ``.wav``: the driving audio is VAE-encoded to CLEAN audio
# latents at sample time and fed as FROZEN conditioning (the same contract A2VStrategy trains under —
# audio at per-token timestep 0, excluded from any denoise), while the video is generated from noise.
# ``generate_audio`` stays False (``_generation_kwargs`` L93): a2v emits VIDEO only. Dev base only
# (D-7-BASEVAR: the distilled two-stage path has no input-audio surface).


def plan_audio_condition(condition: Any) -> str:
    """Normalize one ``audio`` validation-sample condition to its ``wav_path`` (pure/CPU).

    Accepts BOTH the real ``config.schema.AudioCondition`` sub-model (attribute access) and a plain
    ``dict`` (test / dry-run shape) — the same dual-shape convention as ``plan_mask_condition`` /
    ``multi_condition._item_field``. Fail-fast (D-NOHARDCODE / lean-guard doctrine): a wrong ``type``
    discriminator or a missing/empty ``audio`` path raises ``ValueError`` naming the offending field
    BEFORE any metered work, never a silent mis-render. The returned path is the config's
    Volume-relative string, untouched — resolution under the Volume mount (and the traversal guard)
    happens Modal-side in ``run_audio_condition_sampler``.
    """

    def _field(name: str) -> Any:
        if isinstance(condition, dict):
            return condition.get(name)
        return getattr(condition, name, None)

    cond_type = _field("type")
    if cond_type is not None and cond_type != "audio":
        raise ValueError(
            f"audio condition has type={cond_type!r}: this branch renders ONLY the 'audio' "
            f"condition kind (AudioCondition.type — the 'mask' kind routes through "
            f"plan_mask_condition, not this branch)."
        )
    audio = _field("audio")
    if not isinstance(audio, str) or not audio:
        raise ValueError(
            "audio condition is missing a non-empty 'audio' path: expected the AudioCondition shape "
            "{type: 'audio', audio: <Volume-relative .wav>} "
            f"(got audio={audio!r})."
        )
    return audio


def run_audio_condition_sampler(
    components: Any,
    transformer: Any,
    config: "SignetConfig",
    audio_path: str,
    device: str = "cuda",
    cached_embeddings: Any | None = None,
    prompt: str | None = None,
) -> Any:
    """Render one a2v test video: VIDEO generated, driven by an input ``.wav`` (Modal-side ONLY).

    The Phase-9 a2v validation branch (GATE-SPEC rev 2 item 7): VAE-encode the driving audio to
    CLEAN audio latents (via ``components.audio_vae_encoder`` — loaded by
    ``models/loader.load_ltxv_components(with_audio_vae_encoder=True)``), feed it as FROZEN
    conditioning (timestep 0, never denoised — the SAME contract ``conditioning/a2v.py`` trains
    under), and generate the video from noise so a trained a2v LoRA produces video the audio steers.

    Inference discipline (carried from the sibling render branches):
      * Dev-family base ONLY, NEVER distilled (D-7-BASEVAR: the distilled two-stage path has no
        input-audio surface). ``validation.two_stage_upscale`` must be False (fail-fast below).
      * ``generate_audio=False`` (``_generation_kwargs``): a2v emits VIDEO only.
      * Two-phase VRAM (06-09): ``cached_embeddings`` -> ``GenerationConfig.cached_embeddings`` so
        ``_get_prompt_embeddings`` never live-loads Gemma alongside the 22B; ``torch.no_grad()``
        wraps every forward (audio VAE encode, denoise steps, decode).

    ⚠ LIVE-GPU VALIDATION REQUIRED (GATE-SPEC "~80% flip-switches" + a2v is DATA-BLOCKED): signet's
    pinned SHA d6053703 ships NO audio *training* strategy (only text_to_video / video_to_video), but
    its INFERENCE ``ValidationSampler`` DOES ship a full audio path — ``_run_denoising`` threads an
    audio ``LatentState`` into an audio ``Modality`` and calls ``x0_model(video=, audio=)`` every step
    (validation_sampler.py:511-599), copying conditioned audio tokens back clean at their per-token
    ``denoise_mask`` (:575-578). So the frozen-audio render is a faithful PORT of that path, not an
    invention: ``_encode_driving_audio`` mirrors ``process_videos.encode_audio`` (the exact latent
    payload training uses) and ``_render_video_with_frozen_audio`` builds the audio state with an
    ALL-ZERO ``denoise_mask`` — the sampler's own ``sigma * denoise_mask`` (→ per-token timestep 0) and
    clean copy-back then reproduce ``conditioning/a2v.py``'s frozen contract (sigma 0, clean, no
    denoise) EXACTLY. What still wants a real A100 to settle: (a) the audio VAE encoder dtype (the pin
    encodes at float32; signet loads it at the components dtype), (b) audio-latent seq alignment through
    the audio patchifier, (c) VRAM on the pure-dev-base a2v render (the offloader does not track
    ``audio_transformer_blocks``). Do not trust a verdict from this path until that live run confirms
    it.

    Args:
        components / transformer: loaded components (REQUIRES ``components.audio_vae_encoder``) + the
            (PEFT-wrapped) transformer.
        config: the full ``SignetConfig`` (canonical sampling params ride ``validation``).
        audio_path: ONE audio condition's Volume-relative ``.wav`` path (the ``plan_audio_condition``
            output) — resolved under the checkpoints mount here, traversal-guarded first.
        prompt: this render's prompt (``ValidationSample.prompt``). ``None`` falls back to
            ``validation.prompts[0]`` (the siblings' convention).

    Returns:
        ``(video[C, F, H, W] in [0, 1], None)`` — the same 2-tuple unpack shape as the sibling
        render fns (video-only; a2v generates no audio output).
    """
    import torch  # noqa: PLC0415 — GPU-side; function-local keeps the module CPU-importable

    from signet_trainer.config.validators import validate_volume_relative_path  # noqa: PLC0415

    # Fail-fast: a2v is single-pass on the dev base (the distilled two-stage path has no input-audio
    # surface, D-7-BASEVAR).
    if config.validation.two_stage_upscale:
        raise ValueError(
            "validation.two_stage_upscale is True but a2v renders SINGLE-PASS on the dev base only "
            "(D-7-BASEVAR: the distilled two-stage path has no input-audio surface). Set "
            "validation.two_stage_upscale: false."
        )
    if getattr(components, "audio_vae_encoder", None) is None:
        raise ValueError(
            "a2v rendering requires components.audio_vae_encoder (the driving .wav is VAE-encoded to "
            "audio latents) — load with load_ltxv_components(with_audio_vae_encoder=True)."
        )

    # Path-traversal guard (T-07-05-01 class) BEFORE any read, then resolve under the mount.
    validate_volume_relative_path(audio_path, field="audio condition wav path")
    from signet_trainer.modal.app import CHECKPOINTS_DIR  # noqa: PLC0415

    resolved_audio_path = str(CHECKPOINTS_DIR / audio_path)

    device = torch.device(device) if isinstance(device, str) else device

    v = config.validation
    render_prompt = prompt if prompt is not None else (v.prompts[0] if v.prompts else "")
    cfg = build_generation_config(
        config, prompt=render_prompt, seed=v.seed, condition_image=None
    )
    if cached_embeddings is not None:
        cfg.cached_embeddings = cached_embeddings

    # Encode the driving audio to CLEAN audio latents (the frozen a2v conditioning), then render the
    # video conditioned on them. The encode mirrors process_videos.encode_audio (identical latent
    # payload); the render ports the pin's own audio-capable ValidationSampler denoise path. Both wrap
    # under no_grad — inference never needs a gradient (06-09 run-7 landmine).
    with torch.no_grad():
        audio_latents = _encode_driving_audio(
            components.audio_vae_encoder, resolved_audio_path, cfg, device
        )
        video_output, _audio = _render_video_with_frozen_audio(
            components, transformer, cfg, audio_latents, device
        )
    return video_output, None


# --------------------------------------------------------------------------------------------------
# a2v pure/CPU helpers (no ltx_core / no torchaudio) — the executable statements of the audio-latent
# and frozen-state contracts, so tests/test_sampler_a2v_condition.py can assert them on Windows/CI.
# --------------------------------------------------------------------------------------------------


def _fit_waveform_to_duration(waveform: Any, sample_rate: int, target_duration: float) -> Any:
    """Trim / zero-pad a ``[channels, samples]`` waveform to ``target_duration`` (pure/CPU).

    VERBATIM the trim/pad of the pin's ``process_videos.VideoDataset._extract_audio`` (:194-205):
    ``target_samples = int(target_duration * sample_rate)``; slice on the sample axis when longer,
    right zero-pad when shorter. The render side trims the driving audio to the VIDEO's duration
    (``cfg.num_frames / cfg.frame_rate``) so its temporal RoPE positions line up with the video the
    same way training's clip-duration extraction does — train and inference then "mean the same
    thing". Torch-only (function-local import) so the module top stays stdlib (Anti-Pattern 6).
    """
    import torch  # noqa: PLC0415 — torch is a hard dep everywhere; keeps the module top stdlib-only

    target_samples = int(target_duration * sample_rate)
    current_samples = waveform.shape[-1]
    if current_samples > target_samples:
        return waveform[..., :target_samples]
    if current_samples < target_samples:
        return torch.nn.functional.pad(waveform, (0, target_samples - current_samples))
    return waveform


def _audio_latent_payload(latents: Any, duration: float) -> dict[str, Any]:
    """Package an audio-VAE encoder output ``[B, C, T, F]`` into the training-parity payload (pure).

    Returns EXACTLY the dict ``process_videos.encode_audio`` writes per sample (:884-889) and that
    ``conditioning/a2v.py::A2VStrategy._extract_audio_latent`` reads: ``latents`` is the batch-stripped
    ``[C, T, F]`` (= ``[8, T, 16]`` at the LTX audio contract), plus the ``num_time_steps`` /
    ``frequency_bins`` / ``duration`` shape metadata. Fail-fast on a non-4-dim encoder output so a
    shape drift dies here, not deep in the render. Pure tensor ops — CPU-gated with a synthetic tensor.
    """
    if latents.dim() != 4:
        raise ValueError(
            f"audio VAE encoder output must be [B, C, T, F] (4-dim), got {latents.dim()}-dim shape "
            f"{tuple(latents.shape)} — encode_audio squeezes the batch dim to yield the [C, T, F] "
            f"per-sample payload."
        )
    _b, _c, time_steps, freq_bins = latents.shape
    return {
        "latents": latents.squeeze(0),  # [C, T, F] — batch dim removed, as encode_audio does
        "num_time_steps": int(time_steps),
        "frequency_bins": int(freq_bins),
        "duration": float(duration),
    }


def build_frozen_audio_latent_state(
    audio_latents: Any,
    *,
    audio_patchifier: Any,
    audio_latent_shape_cls: Any,
    latent_state_cls: Any,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Build the FROZEN driving-audio ``LatentState`` for the sampler denoise loop (deps injected).

    The render-side mirror of ``conditioning/a2v.py::_build_frozen_audio_modality``: unwrap the
    ``[C, T, mel]`` audio latent (payload dict OR bare tensor, the same dual shape the training side
    accepts), patchify to ``[B, T, C*mel]`` (== ``[B, T, 128]`` at the LTX audio contract), and attach
    the audio positions ``[B, 1, T, 2]`` from ``get_patch_grid_bounds`` — ONE positional dim, the audio
    branch's own coord space.

    The FROZEN contract is encoded as an ALL-ZERO ``denoise_mask`` ``[B, seq, 1]`` (the upstream
    ``LatentState.denoise_mask`` shape; ic_lora's ``ref_denoise_mask`` is the same
    ``[1, seq, 1]``, validation_sampler.py:279). The sampler's ``_run_denoising`` then does the rest
    with NO a2v-specific code: ``timesteps = sigma * denoise_mask`` → per-token timestep 0, the
    ``GaussianNoiser`` leaves ``latent*(1-denoise_mask)`` = the clean latent untouched, and the
    per-step copy-back ``denoised*denoise_mask + clean*(1-denoise_mask)`` pins every audio token to the
    clean driving latent every step (validation_sampler.py:516-546, 575-578). That is EXACTLY
    ``A2VStrategy``'s training semantics (sigma 0 / timestep 0 / clean / never a loss target).

    ``latent`` and ``clean_latent`` are BOTH the clean patchified driving latents (the copy-back reads
    ``audio_clean_state.latent``). Deps (``audio_patchifier`` / ``audio_latent_shape_cls`` /
    ``latent_state_cls``) are INJECTED so the CPU gate drives this with the same stubs
    ``tests/test_a2v_strategy.py`` uses; Modal-side ``_render_video_with_frozen_audio`` passes the
    sampler's real ``_audio_patchifier`` + the ltx_core ``AudioLatentShape`` / ``LatentState``.
    """
    import torch  # noqa: PLC0415 — torch is a hard dep everywhere; keeps the module top stdlib-only

    latent = audio_latents["latents"] if isinstance(audio_latents, dict) else audio_latents
    if device is not None or dtype is not None:
        latent = latent.to(device=device, dtype=dtype)
    if latent.dim() == 3:  # [C, T, mel] -> [B=1, C, T, mel]
        latent = latent.unsqueeze(0)
    if latent.dim() != 4:
        raise ValueError(
            f"audio latent must be [C, T, mel] or [B, C, T, mel]; got shape {tuple(latent.shape)}."
        )
    a_b, a_c, a_t, a_mel = latent.shape

    # Patchify to [B, T, C*mel] and derive the audio positions from the SAME latent's actual T (the
    # audio is a separate dual-stream, cross-attended — its seq length is independent of the video's,
    # so building positions from the encoded T can never drift out of alignment with the latent).
    audio_patched = audio_patchifier.patchify(latent)
    audio_seq_len = audio_patched.shape[1]
    audio_positions = audio_patchifier.get_patch_grid_bounds(
        output_shape=audio_latent_shape_cls(
            frames=a_t,
            mel_bins=a_mel,
            batch=a_b,
            channels=a_c,
        ),
        device=device,
    )

    # FROZEN: denoise_mask 0 everywhere -> per-token timestep 0, never noised, pinned clean each step.
    denoise_mask = torch.zeros(a_b, audio_seq_len, 1, dtype=torch.float32)
    if device is not None:
        denoise_mask = denoise_mask.to(device)

    return latent_state_cls(
        latent=audio_patched,
        denoise_mask=denoise_mask,
        positions=audio_positions,
        clean_latent=audio_patched,
    )


# --------------------------------------------------------------------------------------------------
# a2v Modal-side seams — ported from the pin (process_videos.encode_audio + ValidationSampler audio
# path). Heavy imports (ltx_core / torchaudio) are function-local (Anti-Pattern 6).
# --------------------------------------------------------------------------------------------------


def _encode_driving_audio(
    audio_vae_encoder: Any, resolved_audio_path: str, cfg: Any, device: Any
) -> Any:
    """Read a driving ``.wav`` and VAE-encode it to CLEAN audio latents (Modal-side ONLY).

    A faithful port of the pin's audio pipeline (``process_videos.encode_audio`` @ d6053703): load the
    waveform, trim/pad it to the render's VIDEO duration exactly like ``_extract_audio``, run it through
    the ltx_core ``AudioProcessor`` (resample → log-mel spectrogram) built from the encoder's own
    ``sample_rate/mel_bins/mel_hop_length/n_fft``, then encode with ``components.audio_vae_encoder``.
    Returns the IDENTICAL payload the training data uses — ``{latents[C,T,F], num_time_steps,
    frequency_bins, duration}`` (see ``_audio_latent_payload``).

    Two deliberate, documented parity choices:
      * ``target_duration = cfg.num_frames / cfg.frame_rate`` — the render's video span. Training trims
        to the *clip's* duration; here the video span is the analogue, so a longer/shorter driving wav
        is trimmed/padded to the exact frames being rendered (the task's "equivalent muxed clip").
      * the mel is computed in **float32** (correct precision for the STFT), then cast to the encoder's
        loaded dtype right before the forward — mirroring ``encode_audio``'s ``mel.to(dtype)`` (:876)
        while avoiding a bf16-FFT failure if signet loaded the audio VAE at bf16 (the pin loads it
        float32; see the ⚠ note on ``run_audio_condition_sampler``). The latent FORMAT is unchanged.

    Heavy imports (``torchaudio`` / ltx_core audio) are function-local — this runs on the mounted-
    weights GPU container only.
    """
    import torch  # noqa: PLC0415 — GPU-side; function-local keeps the module CPU-importable
    import torchaudio  # noqa: PLC0415 — the SAME backend the pin's _extract_audio reads wavs with

    from ltx_core.model.audio_vae import AudioProcessor  # noqa: PLC0415
    from ltx_core.types import Audio  # noqa: PLC0415

    # Load the driving wav -> [channels, samples] float in [-1, 1] (torchaudio.load, as _extract_audio
    # does). Keep native channel count / rate; the AudioProcessor resamples to the encoder's rate.
    waveform, sample_rate = torchaudio.load(resolved_audio_path)

    # Trim/pad to the render's video duration (the _extract_audio contract).
    target_duration = cfg.num_frames / cfg.frame_rate
    waveform = _fit_waveform_to_duration(waveform, sample_rate, target_duration)

    # The encoder's own device/dtype (encode_audio reads them off the params). Build the AudioProcessor
    # from the encoder's mel config so the spectrogram matches what training encoded under.
    enc_device = next(audio_vae_encoder.parameters()).device
    enc_dtype = next(audio_vae_encoder.parameters()).dtype
    audio_processor = AudioProcessor(
        target_sample_rate=audio_vae_encoder.sample_rate,
        mel_bins=audio_vae_encoder.mel_bins,
        mel_hop_length=audio_vae_encoder.mel_hop_length,
        n_fft=audio_vae_encoder.n_fft,
    ).to(device=enc_device)

    # [channels, samples] -> [batch=1, channels, samples] (encode_audio:867-869). Duration is the
    # trimmed sample count over the ORIGINAL rate (encode_audio:872 — before the internal resample).
    waveform = waveform.to(device=enc_device, dtype=torch.float32)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    duration = waveform.shape[-1] / sample_rate

    # Waveform -> log-mel [B, C, T, n_mels] at float32, then cast to the encoder dtype for the forward.
    mel_spectrogram = audio_processor.waveform_to_mel(
        Audio(waveform=waveform, sampling_rate=sample_rate)
    )
    latents = audio_vae_encoder(mel_spectrogram.to(dtype=enc_dtype))  # [B, C, T, F] = [1, 8, T, 16]
    return _audio_latent_payload(latents, duration)


def _render_video_with_frozen_audio(
    components: Any, transformer: Any, cfg: Any, audio_latents: Any, device: Any
) -> Any:
    """Generate video conditioned on the frozen driving-audio latents (Modal-side ONLY).

    Ports the pin's own audio-capable ``ValidationSampler`` denoise (validation_sampler.py:182-236,
    477-599): plain t2v on the VIDEO side (fully noised from the prompt — no first-frame / mask
    conditioning), with the FROZEN driving audio threaded as an audio ``LatentState`` whose all-zero
    ``denoise_mask`` reproduces ``conditioning/a2v.py``'s clean/timestep-0/no-denoise contract. NO
    a2v-specific denoise code: ``sampler._run_denoising`` already builds the audio ``Modality`` from
    the audio state and calls ``x0_model(video=, audio=)`` every step, cross-attending the audio into
    the video branch. ``generate_audio`` stays False (video-only output — no audio decode/vocoder).

    Reuse-not-replicate: video state via the sampler's ``_create_video_latent_tools`` +
    ``create_initial_state`` + ``GaussianNoiser`` (the ``_generate_standard`` tail), audio state via
    ``build_frozen_audio_latent_state`` with the sampler's real ``_audio_patchifier`` and the ltx_core
    ``AudioLatentShape`` / ``LatentState``. Heavy imports are function-local (Anti-Pattern 6).
    """
    import torch  # noqa: PLC0415 — GPU-side; function-local keeps the module CPU-importable

    from ltx_core.components.noisers import GaussianNoiser  # noqa: PLC0415
    from ltx_core.types import AudioLatentShape, LatentState  # noqa: PLC0415

    sampler = build_validation_sampler(components, transformer)

    # Prompt embeddings FIRST (sequential-VRAM ordering); cached_embeddings (set on cfg by the caller)
    # short-circuits Gemma. Returns the audio context too — the audio branch cross-attends to a_ctx_*.
    v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = sampler._get_prompt_embeddings(cfg, device)

    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    noiser = GaussianNoiser(generator=generator)

    # VIDEO: plain t2v — full initial state, fully noised (denoise_mask all ones; no conditioning).
    video_tools = sampler._create_video_latent_tools(cfg)
    video_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
    video_state = noiser(latent_state=video_clean_state, noise_scale=1.0)

    # AUDIO: the FROZEN driving-audio state (clean latents, denoise_mask 0). Noising is inert at
    # denoise_mask 0 (noise*0 + latent*1 = clean) — run it through the SAME noiser to mirror
    # _generate_standard byte-for-byte. bf16 to match the video compute dtype / the transformer.
    audio_clean_state = build_frozen_audio_latent_state(
        audio_latents,
        audio_patchifier=sampler._audio_patchifier,
        audio_latent_shape_cls=AudioLatentShape,
        latent_state_cls=LatentState,
        device=device,
        dtype=torch.bfloat16,
    )
    audio_state = noiser(latent_state=audio_clean_state, noise_scale=1.0)

    # Denoise with the frozen audio threaded through (the sampler cross-attends it into the video).
    video_state, _audio_state = sampler._run_denoising(
        config=cfg,
        video_state=video_state,
        audio_state=audio_state,
        video_clean_state=video_clean_state,
        audio_clean_state=audio_clean_state,
        v_ctx_pos=v_ctx_pos,
        a_ctx_pos=a_ctx_pos,
        v_ctx_neg=v_ctx_neg,
        a_ctx_neg=a_ctx_neg,
        device=device,
    )

    # Decode SINGLE-PASS, video-only (generate_audio False — no audio decoder/vocoder loaded).
    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    video_output = sampler._decode_video(video_state, device, cfg.tiled_decoding)
    return video_output, None
