"""inference.multi_condition — the net-new multi-condition inference path (Phase 6, REF-02, Seam E).

LTX supports multi-condition natively via ``ltx_core`` ``VideoConditionByLatentIndex`` (in-place
replace at a latent-frame index, ``denoise_mask = 1 - strength``), but the ported
``ValidationSampler`` ``GenerationConfig`` only exposes a SINGLE hard-clean ``condition_image``.
This module generalizes the sampler's first-frame ``_apply_image_conditioning`` to a LIST of
conditioning items, single-stage, no new weights (NOT the heavy ``KeyframeInterpolationPipeline``).

Two layers, mirroring ``inference/sampler.py``'s heavy-import-local discipline:

  * ``plan_conditioning_items`` / ``plan_multi_frame_columns`` — PURE/CPU (stdlib + typing only, no
    torch, no ltx_core, no filesystem). The item planner maps config
    ``{image, frame_index, strength}`` keyframes onto ``(latent_idx, strength)`` tuples with the
    ``frame_index % 8 == 0`` defence-in-depth assert (Pitfall 3); the column planner derives the
    ordered SC#3 grid columns (N=2, N=3, strength-sweep-lo/hi, no-reference) the Modal ``sample``
    branch (Plan 06-08) renders at seed 42. Both are CPU-testable on Windows/CI.

  * ``run_multi_condition_sampler`` — Modal-side ONLY. Applies N ``VideoConditionByLatentIndex``
    items to the clean latent state, then noises + denoises + decodes. ALL heavy imports
    (``ltx_core`` / ``ltx_trainer`` / ``torch`` / ``torchvision``) are function-local
    (Anti-Pattern 6), so importing this module on CPU/CI never pulls the ltx stack.

No Wan params (T-06-05): the multi-condition path reuses the RE-VALIDATED canonical
``GenerationConfig`` via ``inference.sampler.build_generation_config`` (Euler + STG, guidance from
config, frames 8k+1) and introduces NO Wan-tuned token. ``tests/test_no_wan_params.py`` scans this
module for the banned set.

``frame_index -> latent_idx`` mapping (Pitfall 3):
    ``latent_idx = frame_index // 8`` given ``frame_index % 8 == 0`` (LTX temporal compression 8,
    causal first frame: latent frame 0 <-> pixel frame 0). ``VideoConditionByLatentIndex`` handles
    the ``get_token_count`` start-token offset and the latent/clean/denoise_mask writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only; never executed — keeps the module importable without ltx_core/ltx_trainer.
    from signet_trainer.config.schema import SignetConfig

__all__ = [
    "MultiFrameColumn",
    "plan_conditioning_items",
    "plan_multi_frame_columns",
    "run_multi_condition_sampler",
]

# LTX temporal compression factor (causal first frame). A config ``frame_index`` is a PIXEL index;
# the latent-space index is ``frame_index // TIME_SCALE``. Kept as a named constant, not a literal.
TIME_SCALE = 8


def _item_field(item: Any, name: str) -> Any:
    """Read ``name`` from a conditioning item that is either a mapping or an attribute object.

    ``plan_conditioning_items`` accepts both the real ``ConditioningItem`` Pydantic sub-model
    (attribute access) and a plain ``dict`` (test / dry-run shape) so it stays pure and easy to
    unit-test without constructing the schema.
    """
    if isinstance(item, dict):
        return item[name]
    return getattr(item, name)


def plan_conditioning_items(items: "list[Any]") -> list[tuple[int, float]]:
    """Map config conditioning items onto ``(latent_idx, strength)`` tuples (pure/CPU).

    For each item (mapping OR ``ConditioningItem`` object) reads ``frame_index`` + ``strength`` and
    returns ``(frame_index // TIME_SCALE, strength)``. Raises ``ValueError`` if any ``frame_index``
    is not a multiple of ``TIME_SCALE`` — a defence-in-depth restatement of the config-load
    ``validate_conditioning_items`` check (Pitfall 3), so a mis-wired caller can never place a
    keyframe latent at the wrong temporal region.
    """
    planned: list[tuple[int, float]] = []
    for i, item in enumerate(items):
        frame_index = int(_item_field(item, "frame_index"))
        strength = float(_item_field(item, "strength"))
        if frame_index % TIME_SCALE != 0:
            lower = frame_index - (frame_index % TIME_SCALE)
            upper = lower + TIME_SCALE
            raise ValueError(
                f"conditioning item {i}: frame_index={frame_index} is not a multiple of "
                f"{TIME_SCALE} (LTX temporal compression). Nearest valid indices: {lower} / {upper}."
            )
        planned.append((frame_index // TIME_SCALE, strength))
    return planned


@dataclass(frozen=True)
class MultiFrameColumn:
    """One column of the SC#3 multi-frame grid the Modal ``sample`` branch (Plan 06-08) renders.

    ``label``   — human-readable column label (for logs / debugging).
    ``row_key`` — the ``write_multi_frame_gallery`` row-dict key this column's mp4 fills
                  (``n2_mp4`` / ``n3_mp4`` / ``strength_lo_mp4`` / ``strength_hi_mp4`` /
                  ``no_ref_mp4`` — see ``inference/grid.py::_multi_frame_block``).
    ``items``   — the ordered ``(latent_idx, strength)`` tuples to apply for this column
                  (empty for the no-reference control). Feeds ``run_multi_condition_sampler``.
    """

    label: str
    row_key: str
    items: list[tuple[int, float]] = field(default_factory=list)


def plan_multi_frame_columns(config: "SignetConfig") -> list[MultiFrameColumn]:
    """Derive the ordered SC#3 grid columns from ``config.conditioning`` (pure/CPU).

    Columns, in render order (each at seed 42, single-stage base model):

      * ``N=2``      — the first 2 keyframes, original strengths.
      * ``N=3``      — the first 3 keyframes, original strengths.
      * strength-lo  — the full keyframe set with the MIDDLE item's strength forced to the LOW
                       end of ``conditioning.conditioning_strength_range`` (the strength dial down).
      * strength-hi  — same set with the middle item forced to the HIGH end of the range (dial up).
      * no-reference — the empty control (``items == []``): the base model with no conditioning.

    The strength-sweep endpoints come from ``conditioning_strength_range`` (config-driven, NOT
    hardcoded); the "middle item" is ``planned[len // 2]``. All strengths flow from config — this
    planner never invents a sampling value.

    Raises ``ValueError`` when fewer than 3 keyframes are configured (WR-07): the N=2/N=3 columns
    are keyframe SUBSETS (``planned[:2]`` / ``planned[:3]``), so with 1-2 items the N=3 column
    would render a byte-identical video to N=2 under a different label — inviting a wrong visual
    PASS/FAIL judgment ("N=3 matches N=2 — conditioning saturates?") from an artifact defect, not
    a model behavior. Fail fast before any render spend instead.
    """
    cond = config.conditioning
    planned = plan_conditioning_items(cond.conditioning_items)
    if len(planned) < 3:
        raise ValueError(
            f"multi_frame grid needs >= 3 conditioning_items (got {len(planned)}): the SC#3 "
            f"columns N=2 / N=3 / strength-sweep are keyframe subsets — with fewer items the "
            f"N=3 column would silently duplicate N=2 (same video, different label) and the "
            f"sweep would sweep a boundary item. List at least 3 keyframes in "
            f"conditioning.conditioning_items."
        )
    lo, hi = cond.conditioning_strength_range

    def _sweep(strength: float) -> list[tuple[int, float]]:
        mid = len(planned) // 2
        swept = list(planned)
        swept[mid] = (swept[mid][0], float(strength))
        return swept

    return [
        MultiFrameColumn(label="N=2", row_key="n2_mp4", items=planned[:2]),
        MultiFrameColumn(label="N=3", row_key="n3_mp4", items=planned[:3]),
        MultiFrameColumn(label=f"mid strength {lo}", row_key="strength_lo_mp4", items=_sweep(lo)),
        MultiFrameColumn(label=f"mid strength {hi}", row_key="strength_hi_mp4", items=_sweep(hi)),
        MultiFrameColumn(label="no reference (control)", row_key="no_ref_mp4", items=[]),
    ]


def run_multi_condition_sampler(
    components: Any,
    transformer: Any,
    config: "SignetConfig",
    device: str = "cuda",
    cached_embeddings: Any | None = None,
) -> Any:
    """Apply N ``VideoConditionByLatentIndex`` items on the single-stage sampler (Modal-side ONLY).

    Generalizes ``ValidationSampler._apply_image_conditioning`` (first-frame special case) to the
    list of ``config.conditioning.conditioning_items``: build the clean video latent state, then for
    each ``(latent_idx, strength)`` (from ``plan_conditioning_items``) read the keyframe image,
    normalize it to ``[3,H,W]`` in ``[0,1]`` via ``inference.reference.to_condition_image``,
    VAE-encode it to a single-latent-frame tensor, and
    ``state = VideoConditionByLatentIndex(latent, strength, latent_idx).apply_to(state, tools)``
    (in-place replace + ``denoise_mask = 1 - strength``). Then noise + denoise + decode, reusing the
    SAME canonical ``GenerationConfig`` (via ``build_generation_config``) — no Wan param.

    Reuse-vs-replicate (Assumption A3): reuses the ``ValidationSampler`` internals
    (``build_validation_sampler`` accessor + ``_create_video_latent_tools`` /
    ``_encode_conditioning_image`` / ``_generate_standard`` with a pre-conditioned
    ``video_clean_state``). All building blocks (``VideoLatentTools``, ``GaussianNoiser``,
    ``_run_denoising``, decode) are installed at the pinned SHA (d6053703); if
    ``_generate_standard`` does not accept the pre-built state at that SHA, replicate its ~30-line
    tail (Gaussian noising -> ``_run_denoising`` honouring ``denoise_mask`` -> decode). The exact
    private surface is confirmed at the FIRST GPU exercise of this path — the gated base-model
    sample (Plan 06-09), NOT on CPU.

    Heavy imports are function-local (Anti-Pattern 6): this runs on the mounted-weights GPU
    container, never on Windows/CI.
    """
    # Function-local heavy imports (never executed on CPU/CI).
    import torch  # noqa: PLC0415

    from ltx_core.components.noisers import GaussianNoiser  # noqa: PLC0415
    from ltx_core.conditioning.types.latent_cond import (  # noqa: PLC0415
        VideoConditionByLatentIndex,
    )
    from torchvision.io import read_image  # noqa: PLC0415

    from signet_trainer.inference.reference import to_condition_image  # noqa: PLC0415
    from signet_trainer.inference.sampler import (  # noqa: PLC0415
        build_generation_config,
        build_validation_sampler,
    )

    # Normalize device the same way ValidationSampler.generate does (str -> torch.device) so the
    # replicated tail below matches the source's device handling exactly (validation_sampler.py:174).
    device = torch.device(device) if isinstance(device, str) else device

    cond = config.conditioning
    planned = plan_conditioning_items(cond.conditioning_items)

    # One shared canonical GenerationConfig (Euler + STG, guidance from config) — the SAME
    # re-validated params the single-frame path uses. condition_image stays None: the N items carry
    # the conditioning, not the sampler's single first-frame path.
    prompt = config.validation.prompts[0] if config.validation.prompts else ""
    cfg = build_generation_config(
        config, prompt=prompt, seed=config.validation.seed, condition_image=None
    )
    # SEQUENTIAL-VRAM (06-09 OOM, runs 3-4): the A100-80GB cannot hold the 22B transformer +
    # Gemma-12B + Gemma's forward activations at once (77.29/79.25 GiB resident at the first
    # Gemma forward). Callers pre-encode prompts ONCE with the transformer parked on CPU and pass
    # the ltx-native ``GenerationConfig.cached_embeddings`` ("avoids loading Gemma" at the pinned
    # SHA) — ``_get_prompt_embeddings`` then short-circuits without ever touching Gemma.
    # ``GenerationConfig`` is a plain dataclass, so post-build assignment is the minimal thread-
    # through (no ``_generation_kwargs`` schema change; the CPU param-mapping gate stays intact).
    if cached_embeddings is not None:
        cfg.cached_embeddings = cached_embeddings

    sampler = build_validation_sampler(components, transformer)

    # no_grad wraps EVERY inference forward below (06-09 run-7 landmine, generalized): the ltx
    # sampler internals (_encode_conditioning_image VAE forward, _run_denoising transformer steps,
    # _decode_video) carry no no_grad of their own — called raw, each forward would retain an
    # autograd graph (the Gemma flavor of this cost 20-25GB per encode; 30 chained denoise steps
    # of a 22B transformer would be far worse). Inference-only path — no gradient is ever needed.
    with torch.no_grad():
        # PROMPT EMBEDDINGS FIRST — the source's sequential-VRAM ordering (_generate_standard
        # line 185 runs _get_prompt_embeddings BEFORE any latent/VAE work). With cached_embeddings
        # set this is a pure cache read (tensors .to(device)); without a cache it falls back to the
        # live Gemma encode — only safe if the caller manages VRAM residency.
        v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = sampler._get_prompt_embeddings(cfg, device)

        # Build video latent tools + the clean initial state (reuse sampler internals; A3).
        # `create_initial_state` REQUIRES `dtype` at the pinned SHA (d6053703) — bfloat16 is the
        # LTX-2.3 compute dtype every ValidationSampler call site uses (validation_sampler.py:195;
        # RESEARCH.md:300), not a tunable. Confirmed at the 06-09 GPU run (the earlier device-only
        # call raised TypeError).
        video_tools = sampler._create_video_latent_tools(cfg)
        video_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)

        # Apply one VideoConditionByLatentIndex per config keyframe (in-place replace + denoise_mask).
        for item, (latent_idx, strength) in zip(cond.conditioning_items, planned):
            frame = to_condition_image(read_image(_item_field(item, "image")))
            latent = sampler._encode_conditioning_image(frame, cfg.height, cfg.width, device)
            video_clean_state = VideoConditionByLatentIndex(latent, strength, latent_idx).apply_to(
                video_clean_state, video_tools
            )

        # Noise + denoise + decode from the pre-conditioned state. At the pinned SHA (d6053703)
        # `_generate_standard(self, config, device)` builds its OWN clean state and accepts no
        # external one, so the documented A3 fallback applies: replicate its noise ->
        # _run_denoising -> decode tail (validation_sampler.py:184-236) with our N-item
        # `video_clean_state` in place of the single first-frame `_apply_image_conditioning`.
        # Video-only: audio_state stays None throughout.
        generator = torch.Generator(device=device).manual_seed(cfg.seed)
        noiser = GaussianNoiser(generator=generator)
        video_state = noiser(latent_state=video_clean_state, noise_scale=1.0)
        video_state, _audio_state = sampler._run_denoising(
            config=cfg,
            video_state=video_state,
            audio_state=None,
            video_clean_state=video_clean_state,
            audio_clean_state=None,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            device=device,
        )
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        video_output = sampler._decode_video(video_state, device, cfg.tiled_decoding)
    return video_output, None
