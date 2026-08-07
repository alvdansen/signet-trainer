"""inference.ic_lora_pipeline — the tier-2 official-adapter distilled two-stage baseline (Phase 7, REF-03).

D-7-LADDER tier 2 / D-7-BASELINE: run ONE known-good OFFICIAL Lightricks IC-LoRA
(``LTX-2.3-22b-IC-LoRA-Union-Control`` — the closest structural-control match) through the ported
distilled two-stage ``ltx_pipelines.ic_lora.ICLoraPipeline`` to validate the ported inference
SURFACE against a KNOWN-good adapter BEFORE we train ours (kills pipeline-vs-adapter ambiguity).

This is a DIFFERENT substrate from signet's own dev single-stage PEFT V2V (``inference/reference_video.py``,
07-05/07-09) — do NOT cross the streams (RESEARCH Pitfall 4):

  * The official 2.3-22b IC-LoRAs are **comfy-key single-file safetensors**, NOT PEFT dirs — they are
    handed to ``ICLoraPipeline(loras=[...])``, NEVER loaded through signet's PEFT loader
    (``inference/lora_load.py``). Loading a comfy adapter as a PEFT dir produces garbage misread as a
    pipeline bug (Pitfall 4a). The format-mismatch guard is: this module's ONLY adapter-load surface is
    ``ICLoraPipeline``'s ``loras=`` param.
  * ``ICLoraPipeline`` is the **distilled two-stage** path (distilled base + 2x spatial upscaler), NOT
    the dev single-stage ``ValidationSampler`` — it needs the distilled + spatial-upscaler weights
    (Pitfall 4c). The distilled base variant pairs with ``validation.two_stage_upscale: true`` (the
    D-7-BASEVAR schema fail-fast); the ``fns.py`` baseline branch keys on that flag.
  * ``Union-Control`` is trained ``ref0.5`` (``reference_downscale_factor=2``), so the baseline supplies
    a HALF-RES reference and the pipeline reads ``downscale=2`` from the adapter's safetensors metadata
    (Pitfall 4b). ``read_reference_downscale_factor`` surfaces that value from metadata; do NOT conflate
    it with signet's own ``downscale=1`` dev training (D-7-REF11).

CRITICAL — Anti-Pattern 6 / import confinement (mirrors ``inference/upscale.py``):
    Module top imports stdlib + typing ONLY. ``ltx_pipelines`` / ``ltx_core`` / ``torch`` /
    ``safetensors`` are NEVER imported at module top — the heavy
    ``from ltx_pipelines.ic_lora import ICLoraPipeline`` is deferred INSIDE ``run_ic_lora_baseline``,
    and the metadata read is deferred inside ``read_reference_downscale_factor``. This keeps the
    module CPU-importable so ``tests/test_ic_lora_baseline_wiring.py`` can unit-test the wiring
    WITHOUT ltx-pipelines installed (it is only added to ``gpu_image`` — Modal-side).

DEFERRED live proof (D-7-BLANKET tier 2, 07-10 Task 2): the exact ``ICLoraPipeline`` constructor +
call signature (and the ``LoraPathStrengthAndSDOps`` / ``OffloadMode`` import paths) are
MEDIUM-confidence — VERIFIED from ``ltx_pipelines/ic_lora.py`` @ pinned SHA d6053703 (RESEARCH lines
393-402) but re-validated only at the FIRST gated GPU exercise (Open Question 1 —
``VideoConditionByReferenceLatent.apply_to`` and the pipeline call surface). The structural wiring
lands here; the metered baseline run rides the gated Task-2 invocation. This plan makes the path
EXIST and CPU-verifies the wiring; it commits NO metered spend.

Port source:
    Lightricks/LTX-2 ``packages/ltx-pipelines`` @ pinned SHA d6053703 —
    ``ic_lora.ICLoraPipeline`` (+ ``LoraPathStrengthAndSDOps`` / ``OffloadMode``; RESEARCH lines 393-402).
"""

from __future__ import annotations

from typing import Any

__all__ = ["read_reference_downscale_factor", "run_ic_lora_baseline"]

# The canonical metadata keys an official Lightricks IC-LoRA carries its ref-downscale under. Union-Control
# is ``ref0.5`` -> ``reference_downscale_factor = 2`` (RESEARCH line 410). Read from the adapter's
# safetensors metadata (Pitfall 4b) — NOT hardcoded, NOT guessed from the repo name.
_REF_DOWNSCALE_METADATA_KEYS = ("reference_downscale_factor", "ref_downscale_factor", "ref_downscale")


def read_reference_downscale_factor(adapter_path: str, default: int = 1) -> int:
    """Read the official adapter's ``reference_downscale_factor`` from its safetensors metadata (Pitfall 4b).

    The official comfy-format IC-LoRA safetensors carry their training ``reference_downscale_factor``
    in the file's metadata header (``Union-Control`` is ``ref0.5`` -> ``2``, so the baseline must supply
    a HALF-RES reference and the pipeline reads ``downscale=2`` from here). Reads ONLY the metadata
    header (no tensor load) via ``safetensors.safe_open(...).metadata()`` — a cheap header parse.

    ``safetensors`` is imported FUNCTION-LOCAL (import confinement) so this module stays CPU-importable
    without it. Returns ``default`` (1) when no known key is present rather than guessing — a wrong
    downscale silently mis-scales the reference (Pitfall 4b), so a missing key is surfaced by the
    caller's log, not invented here.

    Args:
        adapter_path: Path to the official comfy-format single-file adapter safetensors (WEIGHTS_DIR).
        default: The factor to return when metadata carries no known downscale key (1 = no downscale).

    Returns:
        The integer reference-downscale factor read from metadata (or ``default`` when absent).
    """
    from safetensors import safe_open  # noqa: PLC0415 — import-confined (CPU-optional dep)

    with safe_open(adapter_path, framework="pt") as f:
        meta = f.metadata() or {}
    for key in _REF_DOWNSCALE_METADATA_KEYS:
        if key in meta:
            # Metadata values are strings; coerce via float first so "2" and "2.0" both parse.
            return int(float(meta[key]))
    return default


def run_ic_lora_baseline(
    *,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    distilled_checkpoint_path: str,
    spatial_upsampler_path: str,
    gemma_root: str,
    official_adapter_path: str,
    reference_video_path: str,
    reference_downscale_factor: int,
    reference_strength: float = 1.0,
    lora_strength: float = 1.0,
    offload_mode: str = "CPU",
    device: str = "cuda",
) -> Any:
    """Run the official-adapter IC-LoRA baseline through the distilled two-stage ``ICLoraPipeline`` (Modal-side).

    Constructs ``ltx_pipelines.ic_lora.ICLoraPipeline`` from the distilled base + spatial-upscaler
    weights + the Gemma root, loads the OFFICIAL comfy-format adapter through the pipeline's ``loras=``
    param (``LoraPathStrengthAndSDOps`` — NEVER signet's PEFT loader, the Pitfall-4 format-mismatch
    guard), and renders one clip conditioned on the seg-map reference via ``video_conditioning=``. The
    reference enters at ``num_frames=25`` / ``frame_rate=25.0`` (D-7-25F), seed 42 (house A/B rule), on
    ``OffloadMode.CPU`` (the distilled base + upscaler + Gemma can exceed A100-80GB — the Pitfall-6 /
    prior-project CPU-offload precedent).

    ``reference_downscale_factor`` is READ FROM THE ADAPTER'S METADATA by the caller
    (``read_reference_downscale_factor``) and passed here for the log + as the authoritative supply-side
    contract: for ``Union-Control`` (``ref0.5`` = 2) the caller supplies a half-res reference. The
    pipeline ALSO reads the factor from the adapter metadata internally (RESEARCH line 398) — this arg
    keeps signet's supply side consistent with what the pipeline expects; the pipeline stays authoritative.

    Heavy imports are FUNCTION-LOCAL (Anti-Pattern 6): this only ever runs on the mounted-weights GPU
    container, never on Windows/CI. The exact ``ICLoraPipeline`` constructor / call kwargs and the
    ``LoraPathStrengthAndSDOps`` / ``OffloadMode`` import surface are MEDIUM-confidence (VERIFIED @
    d6053703, RESEARCH lines 393-402) and re-validated on the gated Task-2 run — do NOT hoist them to
    module top (import-confinement) and do NOT invent kwargs beyond the verified set.

    Args:
        prompt: The re-skin prompt for this render.
        seed: Sampling seed (42 — house A/B rule; passed through, never a literal here).
        height / width: Target render geometry (the demo 832x480 bucket; %32-valid).
        num_frames: Target frame count (25 — D-7-25F; the reference enters at the same length).
        frame_rate: Target fps (25.0 — LTX-native, NOT a Wan carry-forward).
        distilled_checkpoint_path: Path to ``ltx-2.3-22b-distilled-1.1.safetensors`` (WEIGHTS_DIR).
        spatial_upsampler_path: Path to ``ltx-2.3-spatial-upscaler-x2-1.1.safetensors`` (WEIGHTS_DIR).
        gemma_root: Path to the Gemma 3 12B text-encoder dir (WEIGHTS_DIR/gemma-3-12b-it).
        official_adapter_path: Path to the OFFICIAL comfy-format Union-Control adapter safetensors.
        reference_video_path: Path to the seg-map reference clip the adapter steers on.
        reference_downscale_factor: The metadata-read ref-downscale (2 for Union-Control ref0.5).
        reference_strength: Conditioning strength for the ``video_conditioning`` reference (default 1.0).
        lora_strength: Strength the official adapter is applied at (default 1.0).
        offload_mode: Pipeline CPU-offload mode ("CPU" per the Pitfall-6 precedent).
        device: Torch device string ("cuda" Modal-side).

    Returns:
        ``(video_tensor, audio_or_None)`` — same unpack shape as ``run_sampler`` / ``run_two_stage``
        so the ``fns.py`` baseline branch's grid-emission code stays identical.
    """
    # Function-local heavy imports (never executed on CPU/CI). Import surface RE-VALIDATED live at the
    # 07-11 gated GPU exercise against the pinned-SHA source (Open Question 1): ``OffloadMode`` /
    # ``LoraPathStrengthAndSDOps`` live in ``ltx_pipelines.ic_lora`` (re-exports), but the comfy
    # state-dict-ops mapping ``LTXV_LORA_COMFY_RENAMING_MAP`` comes from ``ltx_core.loader`` — the
    # canonical idiom every IC-LoRA caller uses (hdr_ic_lora.py:244, utils/args.py:111, README:449).
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP  # noqa: PLC0415
    from ltx_pipelines.ic_lora import (  # noqa: PLC0415
        ICLoraPipeline,
        LoraPathStrengthAndSDOps,
        OffloadMode,
    )

    # Construct the distilled two-stage IC-LoRA pipeline. The OFFICIAL comfy adapter is loaded through
    # the pipeline's ``loras=`` param ONLY (LoraPathStrengthAndSDOps) — the Pitfall-4 format-mismatch
    # guard: a comfy single-file adapter is NEVER routed through signet's PEFT loader. The THIRD tuple
    # field ``sd_ops`` is REQUIRED (07-11 live resolution of Open Question 1 — the 07-10 MEDIUM-confidence
    # wiring omitted it and crashed): for a comfy-format single-file IC-LoRA it MUST be
    # ``LTXV_LORA_COMFY_RENAMING_MAP`` (strips the ``diffusion_model.`` prefix), the exact idiom
    # ``hdr_ic_lora.py`` / ``utils/args.py`` use for official adapters. ``ref_downscale`` is read from the
    # adapter metadata by the pipeline itself (ic_lora.py __init__), so the full-res reference is passed.
    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=distilled_checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=[
            LoraPathStrengthAndSDOps(
                official_adapter_path, lora_strength, LTXV_LORA_COMFY_RENAMING_MAP
            )
        ],
        offload_mode=OffloadMode.CPU if offload_mode.upper() == "CPU" else OffloadMode.NONE,
    )

    # The reference enters via ``video_conditioning`` (a list of (reference_clip, strength) pairs);
    # ``images=[]`` — no first-frame image condition (the seg-map reference carries the steering).
    result = pipeline(
        prompt=prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=[],
        video_conditioning=[(reference_video_path, reference_strength)],
    )

    # Normalize to the (video, audio) unpack shape (audio is out of scope — video-only).
    if isinstance(result, tuple):
        return result
    return result, None
