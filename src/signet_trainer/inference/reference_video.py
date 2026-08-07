"""inference.reference_video — the dev single-stage V2V render path (Phase 7, REF-03, D-7-BASEVAR).

signet's OWN trained IC-LoRA adapter renders a re-skinned video-to-video output on the dev
single-stage substrate: the ported ``ValidationSampler`` conditioned by a full CLEAN reference
prefix (the seg-map video), not a single first-frame index. It is the substrate the 07-11 re-skin
grid consumes. The official-adapter distilled ``ICLoraPipeline`` baseline is a SEPARATE path
(07-10) — the two streams never cross.

Port, don't invent (canonical-refs mandate): the ltx-trainer ``ValidationSampler`` at the pinned
SHA (d6053703) ALREADY exposes the whole reference-video path — ``generate`` dispatches to
``_generate_with_reference`` whenever ``GenerationConfig.reference_video`` is set, which internally
``_preprocess_reference_video`` (resize/crop/[-1,1]) -> ``_encode_video`` (VAE encode to the clean
reference latent + doubled/scaled positions) -> reference-conditioned denoise -> decode. So this
module does NOT hand-roll the VAE encode or the noise/denoise/decode tail (Open Question 1, resolved
empirically at the first gated GPU exercise via ``scripts/_probe_encode_surface.py``): it decodes
the seg-map clip to pixels, sets it on the config, and delegates to the SAME
``sampler.generate(config, device)`` the single-frame / control paths use (``inference.sampler``).

The ONE load-bearing difference from ``run_sampler``: this path sets ``config.reference_video`` (the
decoded seg-map clip, ``[F, C, H, W]`` in ``[0, 1]``) + ``config.reference_downscale_factor``, which
flips ``generate`` onto its reference branch. Everything else — the two-phase VRAM ``cached_embeddings``
short-circuit, the ``no_grad`` wrap, the canonical ``GenerationConfig`` (Euler + STG, guidance from
config, frames 8k+1) — is byte-identical to ``run_sampler``.

All heavy imports (``ltx_trainer`` / ``torch`` / ``av`` (PyAV)) are function-local (Anti-Pattern 6):
importing this module on CPU/CI never pulls the ltx stack. ``tests/test_no_wan_params.py`` scans it
for the banned Wan set — this path introduces NO Wan token (no UniPC, no shift=4.0, no frames=33,
no guidance=5.0/steps=50); it reuses the RE-VALIDATED canonical ``GenerationConfig`` via
``inference.sampler.build_generation_config``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type-only; never executed — keeps the module importable without ltx_core/ltx_trainer.
    from signet_trainer.config.schema import SignetConfig

__all__ = [
    "run_reference_video_sampler",
]


def _decode_reference_video(reference_video_path: str) -> Any:
    """Decode a seg-map reference video to a ``[F, C, H, W]`` float clip in ``[0, 1]`` (Modal-side).

    The exact tensor ``GenerationConfig.reference_video`` wants (ltx-trainer
    ``_preprocess_reference_video`` reads ``config.reference_video  # [F, C, H, W] in [0, 1]`` then
    resizes/crops/[-1,1]-normalizes + VAE-encodes it itself). This helper ONLY decodes + normalizes
    pixels — no VAE, no resize (the sampler owns those), so it stays a thin, malformed-frame-guarded
    reader.

    ``torchvision.io.read_video`` was REMOVED from ``torchvision.io`` in the container's pinned
    ``torchvision~=0.22`` (the deprecated PyAV-backed reader is gone -> ImportError). Decode with
    PyAV directly — the SAME backend ``ltx_trainer.video_utils.save_video`` writes with, confirmed
    present by the sample cold-path import probe (fns.py:558; RESEARCH A7). Each frame decodes to an
    ``[H, W, C]`` uint8 ndarray (rgb24); ``to_condition_image`` normalizes each to ``[3, H, W]`` in
    ``[0, 1]`` (the malformed-input guard, threat T-05-03 carried), then stack on the FRAME axis
    (dim 0) to ``[F, 3, H, W]`` = ``[F, C, H, W]`` — the layout ``_preprocess_reference_video`` reads.

    Heavy imports are function-local (Anti-Pattern 6): runs on the mounted-weights GPU container.
    """
    import av  # noqa: PLC0415  (PyAV — the SAME mp4 backend ltx_trainer.video_utils.save_video writes)
    import torch  # noqa: PLC0415

    from signet_trainer.inference.reference import to_condition_image  # noqa: PLC0415

    with av.open(reference_video_path) as container:
        frames = [
            torch.from_numpy(frame.to_ndarray(format="rgb24"))
            for frame in container.decode(video=0)
        ]
    if not frames:
        raise ValueError(
            f"reference video decoded to zero frames: {reference_video_path!r} (empty or unreadable)."
        )
    # Each frame -> [3, H, W] in [0, 1] (per-frame malformed-input guard), stacked on the FRAME axis
    # to [F, 3, H, W] = [F, C, H, W] — the exact layout GenerationConfig.reference_video expects.
    normalized = [to_condition_image(frame) for frame in frames]
    return torch.stack(normalized, dim=0)  # [F, C, H, W] in [0, 1]


def run_reference_video_sampler(
    components: Any,
    transformer: Any,
    config: "SignetConfig",
    reference_video_path: str,
    device: str = "cuda",
    cached_embeddings: Any | None = None,
) -> Any:
    """Render one re-skinned V2V output from a seg-map reference video (Modal-side ONLY).

    Mirrors ``inference.sampler.run_sampler`` EXACTLY, plus two reference fields on the config:

      * ``config.reference_video`` — the decoded seg-map clip (``[F, C, H, W]`` in ``[0, 1]``). Setting
        it is what flips ``ValidationSampler.generate`` onto ``_generate_with_reference`` (the sampler
        then VAE-encodes the clip to the clean reference latent and prepends it as the doubled
        ``ref ⧺ target`` sequence with pre-scaled positions — the canonical IC-LoRA inference path).
      * ``config.reference_downscale_factor`` — 1:1 for the first proof (D-7-REF11); the sampler scales
        the reference positions by this factor internally.

    Reuse-not-replicate (canonical-refs mandate, Open Question 1 resolved on GPU): NO hand-rolled VAE
    encode, NO hand-rolled noise/denoise/decode tail. ``generate`` returns ``(video, audio)`` — the
    same 2-tuple shape ``run_sampler`` returns, so the ``ic_lora`` sample branch unpacks it unchanged.

    Two 06-09 carry-forward VRAM guards, identical to ``run_sampler``:
      * ``cached_embeddings`` -> ``GenerationConfig.cached_embeddings`` ("avoids loading Gemma");
        ``_get_prompt_embeddings`` short-circuits so the render loop never holds Gemma + the 22B.
      * ``torch.no_grad()`` around ``generate`` — the ltx sampler internals carry no no_grad; a raw
        forward would retain an autograd graph over the doubled sequence.

    Heavy imports are function-local (Anti-Pattern 6): this runs on the mounted-weights GPU
    container, never on Windows/CI.
    """
    import torch  # noqa: PLC0415

    from signet_trainer.config.validators import validate_volume_relative_path  # noqa: PLC0415
    from signet_trainer.inference.sampler import (  # noqa: PLC0415
        build_generation_config,
        build_validation_sampler,
    )

    # T-07-05-01 (Tampering — reference-video path traversal): the seg-map video path is
    # operator/dataset-authored, then joined under the Volume mount Modal-side. Reject absolute or
    # ``..``-escaping paths BEFORE any read (pure string parsing, no filesystem touch).
    validate_volume_relative_path(reference_video_path, field="reference video path")

    # Resolve the validated Volume-relative seg-map path under the checkpoints mount — the schema
    # contract ("joined as CHECKPOINTS_DIR / value Modal-side", validators.py:218). fns.py hands the
    # Volume-relative string (the SAME value it copies via ``CHECKPOINTS_DIR / seg_rel`` for grid
    # column 2, fns.py:1202); the V2V render must decode the SAME resolved file, never a CWD-relative
    # miss from the container's /root. Function-local import keeps this module CPU-importable
    # (Anti-Pattern 6): ``modal.app`` pulls ``import modal``, only wanted on the GPU container where
    # this call already runs.
    from signet_trainer.modal.app import CHECKPOINTS_DIR  # noqa: PLC0415

    resolved_reference_path = str(CHECKPOINTS_DIR / reference_video_path)

    # Normalize device the same way ValidationSampler.generate does (str -> torch.device).
    device = torch.device(device) if isinstance(device, str) else device

    # The canonical GenerationConfig (Euler + STG, guidance from config) — the SAME re-validated
    # params the single-frame / control paths use. condition_image stays None: the reference PREFIX
    # carries the conditioning, not the sampler's single first-frame path.
    prompt = config.validation.prompts[0] if config.validation.prompts else ""
    cfg = build_generation_config(
        config, prompt=prompt, seed=config.validation.seed, condition_image=None
    )

    # Two-phase VRAM (06-09, generalized): the caller pre-encoded prompts ONCE with the transformer
    # parked / Gemma freed and passes the ltx-native cached embeddings; ``_get_prompt_embeddings``
    # short-circuits without touching Gemma. ``GenerationConfig`` is a plain dataclass, so
    # post-build assignment is the minimal thread-through (mirrors ``run_sampler``).
    if cached_embeddings is not None:
        cfg.cached_embeddings = cached_embeddings

    # The reference fields: decode the seg-map clip to pixels and set it on the config, which flips
    # ``generate`` onto ``_generate_with_reference``. The sampler VAE-encodes it internally — do NOT
    # pre-encode here (Open Question 1 resolved: ``_encode_conditioning_image`` is single-frame-only;
    # the full-clip encode is the sampler's own ``_encode_video``, driven by this field).
    cfg.reference_video = _decode_reference_video(resolved_reference_path)
    cfg.reference_downscale_factor = config.conditioning.reference_downscale_factor

    sampler = build_validation_sampler(components, transformer)
    with torch.no_grad():
        # Returns (video[C, F, H, W] in [0, 1], audio | None) — the same 2-tuple shape run_sampler
        # returns; the ic_lora sample branch unpacks ``reskin_video, _audio``.
        return sampler.generate(config=cfg, device=device)
