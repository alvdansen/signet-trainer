"""models.ltx25_loader — load_ltx25_components(): the LTX-2.5 split-layout loader (issue #53 Stage 1).

Sibling of ``models/loader.py`` (LTX-2.3) and ``models/h3_loader.py`` / ``models/qwen_edit_loader.py``
(families #2/#3) — a SEPARATE module rather than a branch inside ``load_ltxv_components``, so the
byte-identity guarantee for LTX-2.3 (LTX25_STAGE1_DESIGN.md §11 #2) is structural, not just tested:
nothing in this file is ever imported by a ``ltx_generation == '2.3'`` call path.

Invoked Modal-side ONLY (it imports ``ltx_trainer``/``safetensors`` + needs GPU VRAM for the actual
load). The heavy ``ltx_trainer`` import is deferred to INSIDE ``load_ltx25_components`` — merely
importing this module (to call ``read_ltx25_checkpoint_metadata`` from a CPU test, or to read
nothing at all) must not require ``ltx_trainer`` to be installed (Anti-Pattern 6 EXCEPTION, same as
``models/loader.py``'s own module docstring).

⚠⚠ HONESTY (D3 is still open — issue #53's own precondition): no real LTX-2.5 / Gemma-4 checkpoint
has been downloaded or read by this repo. Every function below that reads a checkpoint's embedded
metadata does so through a mechanism transcribed from the upstream diff read
(LTX25_UPSTREAM_DIFF.md §C: ``SpatioTemporalScaleFactors.from_model_config`` /
``LTXModelConfigurator.from_metadata``) — it RECORDS observed values and asserts only
SELF-consistency (a declared count matching an introspected count; an existing (8,32,32) fact
staying true) and never asserts against an invented ``EXPECTED_*_25`` constant. Re-verify this
mechanism the day a real 2.5 checkpoint's metadata is actually read (D3's own precondition).

Port source (read via ``gh`` against ``Lightricks/LTX-2`` at the ``v1.2.0`` tag,
``d151147788a9284cca791edc6ce898007e727fe6``):
    ``packages/ltx-trainer/src/ltx_trainer/model_loader.py::load_model`` — gained OPTIONAL
    ``video_vae_path=``/``audio_vae_path=`` kwargs for the split-pack case (LTX25_UPSTREAM_DIFF.md
    §A row 1); everything else about the call shape is byte-identical to the 2.3 pin.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import torch

from signet_trainer.models.loader import (
    EXPECTED_VAE_SPATIAL_COMPRESSION,
    EXPECTED_VAE_TEMPORAL_COMPRESSION,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ltx_trainer.model_loader import LtxModelComponents

#: Same block-index convention as ``models/loader.py::_BLOCK_IDX_RE`` / ``train/validate_gate.py``'s
#: header-only introspection — matches "transformer_blocks.12.…" / "blocks.0.…".
_BLOCK_IDX_RE = re.compile(r"(?:^|\.)(?:transformer_)?blocks\.(\d+)(?:\.|$)")


def load_ltx25_components(
    checkpoint_path: str,
    text_encoder_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    video_vae_path: str | None = None,
    audio_vae_path: str | None = None,
    with_video_vae_decoder: bool = False,
    with_text_encoder: bool = True,
    with_audio_vae_encoder: bool = False,
) -> "LtxModelComponents":
    """Load LTX-2.5 components. Train-only (Stage 1): both decoder-adjacent flags raise.

    Args:
        checkpoint_path: Path to the LTX-2.5 checkpoint — a monolithic ``.safetensors`` (same
            shape as 2.3) or, when ``ltx25.checkpoint_layout == 'split'``, the TRANSFORMER-ONLY
            file (``video_vae_path``/``audio_vae_path`` name the sibling VAE files).
        text_encoder_path: The Gemma-4 root under ``WEIGHTS_DIR`` (``model.text_encoder_id`` on a
            2.5 config — no new schema field needed; see LTX25_STAGE1_DESIGN.md §1).
        video_vae_path: Split layout only. ``None`` (monolith default) makes ``load_model`` treat
            ``checkpoint_path`` as carrying its own VAE; on a SPLIT transformer-only checkpoint
            with no override, upstream's ``resolve_video_vae_path`` raises ``ValueError`` — moved
            off this metered container by ``SignetConfig._cross_field_checks``'s fail-fast at
            config load (a split layout with no ``video_vae_path`` never reaches this call).
        audio_vae_path: Split layout only, forward-declared for Stage 2/3 — Stage 1 threads it
            through unchanged but does not otherwise read it (a2v-on-2.5 is out of scope, §9).
        with_video_vae_decoder: Must stay False for Stage 1. The decoder path also needs
            ``load_embeddings_processor(gemma_model_path=...)``, which feeds
            ``inference/sampler.py``'s ``ValidationSampler``-shaped code —
            ``ltx_trainer.validation_sampler.{GenerationConfig,ValidationSampler}`` is REMOVED at
            v1.2.0, replaced by ``validation_runner.ValidationRunner``'s self-managed load/unload
            lifecycle (LTX25_UPSTREAM_DIFF.md §A). Raising here (rather than silently degrading)
            closes that gap before a caller ever reaches the missing surface.
        with_audio_vae_encoder: Must stay False for Stage 1 (a2v out of scope, §9) AND is a real
            silent-failure risk on a split checkpoint regardless of scope: the standalone
            ``ltx_trainer.model_loader.load_audio_vae_encoder`` was never extended with split-path
            resolution upstream — handed a split transformer-only file, it silently builds a
            **meta-device** encoder (warning-logged, not raised — ``SingleGPUModelBuilder.
            _return_model``, LTX25_UPSTREAM_DIFF.md §A row 2). Raising here fences that landmine
            for whichever future stage first turns this flag on, rather than leaving the guard
            absent until someone is burned by it.

    Returns:
        ``LtxModelComponents`` — same shape as ``load_ltxv_components``'s return (video-only,
        decoder/audio off for the training path).

    Note:
        ``ltx_trainer`` is imported here (function-local) on purpose — this call runs Modal-side
        ONLY, inside ``ltx25_gpu_image``'s container, never from the dry-run or config-load path.
    """
    if with_video_vae_decoder:
        raise NotImplementedError(
            "load_ltx25_components(with_video_vae_decoder=True) is Stage-2 scope (issue #53 D1) — "
            "the decoder path needs load_embeddings_processor(gemma_model_path=...), which feeds "
            "inference/sampler.py's ValidationSampler-shaped code, and "
            "ltx_trainer.validation_sampler.{GenerationConfig,ValidationSampler} is REMOVED at "
            "v1.2.0, replaced by validation_runner.ValidationRunner's self-managed load/unload "
            "lifecycle (LTX25_UPSTREAM_DIFF.md §A). Do not attempt a partial port."
        )

    if with_audio_vae_encoder:
        raise NotImplementedError(
            "load_ltx25_components(with_audio_vae_encoder=True) is out of Stage-1 scope "
            "(a2v-on-2.5, issue #53 §9) AND a confirmed silent-failure risk on a split checkpoint: "
            "ltx_trainer.model_loader.load_audio_vae_encoder was never extended with split-path "
            "resolution upstream — handed a split transformer-only file it silently builds a "
            "meta-device encoder (warning-logged, not raised; LTX25_UPSTREAM_DIFF.md §A row 2). "
            "Refusing outright rather than risking that landmine."
        )

    from ltx_trainer.model_loader import load_model  # noqa: PLC0415 — Modal-side only

    logger.info(
        "Loading LTX-2.5 components from %s (text_encoder=%s, video_vae_path=%r, "
        "audio_vae_path=%r)",
        checkpoint_path,
        text_encoder_path,
        video_vae_path,
        audio_vae_path,
    )

    components = load_model(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=device,
        dtype=dtype,
        with_video_vae_encoder=True,
        with_video_vae_decoder=False,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=with_text_encoder,
        # [v1.2.0 verified] the ONLY signature delta on load_model itself vs the 2.3 pin — optional,
        # default None (monolith). None on a split transformer-only checkpoint raises inside
        # resolve_video_vae_path (config-load already fail-fasts this — SignetConfig.
        # _cross_field_checks) — that is the intended failure surface, not this call site.
        video_vae_path=video_vae_path,
        audio_vae_path=audio_vae_path,
    )
    return components


def _parse_embedded_config(raw_metadata: dict[str, str] | None) -> dict[str, Any] | None:
    """Parse the ``"config"`` entry of a safetensors ``__metadata__`` dict (JSON text) to a dict.

    safetensors metadata values are always strings; upstream's ``LTXModelConfigurator.from_metadata``
    convention is ``metadata["config"]`` holding the JSON-encoded model config (LTX25_UPSTREAM_DIFF.md
    §C). Returns ``None`` if the key is absent or does not parse — callers treat that as "nothing to
    record", never as a silent pass on the presence check that matters (see
    ``read_ltx25_checkpoint_metadata``, which raises on a wholly-absent metadata header BEFORE this
    is even called).
    """
    if not raw_metadata:
        return None
    raw_config = raw_metadata.get("config")
    if raw_config is None:
        return None
    try:
        parsed = json.loads(raw_config)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def read_ltx25_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any]:
    """Read + RECORD the checkpoint's own embedded arch facts. Asserts self-consistency ONLY.

    No hardcoded ``EXPECTED_*_25`` constant exists anywhere in this repo (D3) — this function never
    compares against one. It:

      1. Opens the checkpoint with ``safetensors.safe_open`` and reads both the embedded
         ``f.metadata()`` dict and the tensor-name/shape introspection this repo already has
         (block indices, mirroring ``models/loader.py``'s ``_BLOCK_IDX_RE`` / ``train/
         validate_gate.py``'s header-only read).
      2. Raises loudly if the checkpoint carries no embedded metadata header at all — every other
         fact this gate needs comes from that dict, so its total absence is itself a hard failure.
      3. Asserts SELF-consistency only: a declared ``num_layers`` (if present in the embedded
         config) must match the max transformer-block index found by tensor-name introspection,
         plus one. Mismatched -> raise, naming BOTH values.
      4. Records (never asserts) ``ff_bias`` if present — the gate report confirms LTX-2.5/gemma4
         sets it ``False``, but this function does not hardcode that as a requirement (a
         dev/distilled-style variant split could exist for 2.5 too, per the D-7-BASEVAR precedent,
         and nothing here has verified it).
      5. Returns a structured summary — the artifact an operator reads to eventually write real
         ``EXPECTED_*_25`` constants, once D3's gated checkpoint download happens. This function
         produces that read; it does not consume it.
    """
    from safetensors import safe_open  # noqa: PLC0415 — optional dep, function-local

    with safe_open(checkpoint_path, framework="pt") as f:
        raw_metadata = f.metadata()
        tensor_names = list(f.keys())

    if not raw_metadata:
        raise ValueError(
            f"{checkpoint_path} carries no embedded __metadata__ header at all. Every arch fact "
            "this gate needs (declared num_layers, ff_bias, the VAE block list) comes from that "
            "dict — its total absence means this is not a metadata-carrying LTX checkpoint (or is "
            "the wrong file). Refusing rather than falling back to a guessed architecture."
        )

    config = _parse_embedded_config(raw_metadata)

    declared_num_layers = (config or {}).get("num_layers")

    block_indices: set[int] = set()
    for name in tensor_names:
        match = _BLOCK_IDX_RE.search(name)
        if match:
            block_indices.add(int(match.group(1)))
    introspected_num_layers = max(block_indices) + 1 if block_indices else None

    if (
        declared_num_layers is not None
        and introspected_num_layers is not None
        and declared_num_layers != introspected_num_layers
    ):
        raise ValueError(
            f"{checkpoint_path}: declared num_layers ({declared_num_layers}, from the embedded "
            f"metadata config) does not match the introspected transformer block count "
            f"({introspected_num_layers}, from tensor-name indices) — self-consistency check "
            "FAILED. This is a fact about the checkpoint's own two descriptions of itself, not a "
            "comparison to any assumed 2.5 architecture."
        )

    summary: dict[str, Any] = {
        "checkpoint_path": checkpoint_path,
        "declared_num_layers": declared_num_layers,
        "introspected_num_layers": introspected_num_layers,
        "ff_bias": (config or {}).get("ff_bias"),
        "tensor_count": len(tensor_names),
        "metadata_config_present": config is not None,
    }
    # RECORD, don't assert (D3): this print is the artifact the operator reads before ever writing
    # a real EXPECTED_*_25 constant.
    print(
        f"[ltx25 arch gate] observed checkpoint metadata (RECORDED, not asserted against any "
        f"EXPECTED_*_25 constant -- D3 open, no real 2.5 checkpoint has been read by this repo): "
        f"{summary}"
    )
    return summary


def compute_vae_scale_factors_from_metadata(
    vae_config: dict[str, Any] | None,
) -> tuple[int, int, int]:
    """``(time, height, width)`` VAE compression factors, derived from an encoder-block list.

    Mirrors the mechanism ``ltx_core.types.SpatioTemporalScaleFactors.from_model_config`` uses
    upstream (LTX25_UPSTREAM_DIFF.md §C): count ``compress_time*`` / ``compress_space*`` /
    ``compress_all*`` block-name prefixes in ``vae_config["encoder_blocks"]`` (a list of
    ``(block_name, params)`` entries, or bare names). Falls back to the EXISTING
    ``(8, 32, 32)`` default — the same fallback the upstream mechanism itself uses — only when no
    block list is present at all (e.g. an audio-only checkpoint's metadata).

    HONESTY: this reconstructs a documented MECHANISM, not a verified 2.5 VALUE (D3 open) — it has
    not been run against a real checkpoint's embedded metadata.
    """
    if not vae_config:
        return (
            EXPECTED_VAE_TEMPORAL_COMPRESSION,
            EXPECTED_VAE_SPATIAL_COMPRESSION,
            EXPECTED_VAE_SPATIAL_COMPRESSION,
        )
    blocks = vae_config.get("encoder_blocks")
    if not blocks:
        return (
            EXPECTED_VAE_TEMPORAL_COMPRESSION,
            EXPECTED_VAE_SPATIAL_COMPRESSION,
            EXPECTED_VAE_SPATIAL_COMPRESSION,
        )

    time_compression = 1
    space_compression = 1
    for entry in blocks:
        name = entry[0] if isinstance(entry, (list, tuple)) else entry
        if not isinstance(name, str):
            continue
        if name.startswith("compress_all"):
            time_compression *= 2
            space_compression *= 2
        elif name.startswith("compress_time"):
            time_compression *= 2
        elif name.startswith("compress_space"):
            space_compression *= 2
    return (time_compression, space_compression, space_compression)


def assert_ltx25_vae_compression(
    checkpoint_path: str, video_vae_path: str | None = None
) -> dict[str, int]:
    """Read the real VAE compression off the mounted checkpoint and assert it matches (8, 32, 32).

    Raises loudly on a mismatch — does NOT silently accept a resolution bucket validated against
    the wrong assumption, and does NOT invent an ``EXPECTED_*_25`` constant (D3): the assertion
    target is signet's EXISTING ``conditioning/strategy.py`` ``(TIME_SCALE, HEIGHT_SCALE,
    WIDTH_SCALE) == (8, 32, 32)``, read here as a fact to CONFIRM against the checkpoint's own
    embedded metadata, never a number chosen for 2.5.

    Cheap (header-only, no tensor materialization) — mirrors the existing
    ``_extract_safetensors_ground_truth`` header-only pattern (``train/validate_gate.py``).
    """
    from safetensors import safe_open  # noqa: PLC0415 — optional dep, function-local

    # A split checkpoint's VAE lives in its OWN file (video_vae_path); a monolith carries it inside
    # checkpoint_path itself.
    target_path = video_vae_path or checkpoint_path
    with safe_open(target_path, framework="pt") as f:
        raw_metadata = f.metadata()

    config = _parse_embedded_config(raw_metadata)
    vae_config = (config or {}).get("vae")
    time_c, height_c, width_c = compute_vae_scale_factors_from_metadata(vae_config)

    observed = {
        "vae_temporal_compression": time_c,
        "vae_spatial_compression_h": height_c,
        "vae_spatial_compression_w": width_c,
    }
    if (
        time_c != EXPECTED_VAE_TEMPORAL_COMPRESSION
        or height_c != EXPECTED_VAE_SPATIAL_COMPRESSION
        or width_c != EXPECTED_VAE_SPATIAL_COMPRESSION
    ):
        raise ValueError(
            f"{target_path}: observed VAE compression (time={time_c}, height={height_c}, "
            f"width={width_c}) does NOT match conditioning/strategy.py's existing assumption "
            f"(time={EXPECTED_VAE_TEMPORAL_COMPRESSION}, height={EXPECTED_VAE_SPATIAL_COMPRESSION}, "
            f"width={EXPECTED_VAE_SPATIAL_COMPRESSION}). Every resolution bucket this run validated "
            "at config load was validated against the WRONG compression factor — the frame-count "
            "law and the RoPE grid would silently mis-shape. This is a config/checkpoint mismatch, "
            "not a code bug; do not patch the constants without re-deriving the frame law for the "
            "real factors."
        )
    return observed
