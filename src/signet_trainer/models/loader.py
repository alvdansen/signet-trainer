"""models — load_ltxv_components(): the ported LTX-2.3 component loader (Phase 2).

Wraps ltx-trainer model_loader.load_model() video-only: transformer + video_vae_encoder
+ text_encoder (Gemma) + scheduler (LTX2Scheduler()); audio VAE / vocoder OFF.

Invoked Modal-side ONLY (it imports ltx_core / ltx_trainer + needs GPU VRAM). It is
NOT in the dry-run/import-confined path — keep it out of any module that the
dry-run or config load imports (the deliberate Anti-Pattern-6 EXCEPTION: unlike the
import-confined conditioning/data modules, this loader MAY import ltx_core/ltx_trainer
precisely because it only ever runs Modal-side at call time, never on Windows/CI).

The heavy ``import ltx_trainer`` is deferred to INSIDE ``load_ltxv_components`` so that
merely importing this module to read the EXPECTED_* constants (Task 2's CPU
safetensors pre-check, Plan-01's oracle script, CI) does NOT require ltx_trainer to be
installed.

Port source (read via ``gh``):
    Lightricks/LTX-2 packages/ltx-trainer/src/ltx_trainer/model_loader.py::load_model
    + LtxModelComponents  (RESEARCH.md Code Examples lines 318-336). The canonical loader
    encapsulates the comfy key-renaming + dtype/device placement via ltx-core's
    SingleGPUModelBuilder — we delegate to it rather than calling ltx_core directly
    (RESEARCH.md Pitfall 6: the ltx-core API is underdocumented; port the canonical loader).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

logger = logging.getLogger(__name__)

#: Matches a transformer block index in a ``named_modules`` key, e.g.
#: "transformer_blocks.12.attn1.to_q" / "blocks.0.ff.net.0.proj". Mirrors the
#: safetensors block-key convention in tests/test_ground_truth_read.py so the live
#: introspection and the header-only pre-check agree on what a "block" is.
_BLOCK_IDX_RE = re.compile(r"(?:^|\.)(?:transformer_)?blocks\.(\d+)(?:\.|$)")

if TYPE_CHECKING:
    # Import-only for type hints; never executed at runtime (keeps constants readable
    # without ltx_trainer installed — see module docstring).
    from ltx_trainer.model_loader import LtxModelComponents

# --------------------------------------------------------------------------------------------------
# enochiatron ground-truth constants — the architecture facts the smoke-test + the CPU safetensors
# pre-check assert against.
# Source: enochiatron validate_ltx23.py FLIMMER_CONSTANTS (RESEARCH.md Code Examples lines 339-345).
# These are the single source of truth: tests/test_ground_truth_read.py and
# scripts/compare_to_oracle.py import them from here (do NOT duplicate the values).
# --------------------------------------------------------------------------------------------------

EXPECTED_NUM_BLOCKS: int = 48
#: The text CONTEXT / connector dim the transformer cross-attends over (Gemma 3840 → connector →
#: 4096). This is what the precomputed video_prompt_embeds are projected to and what the forward
#: feeds as ``context`` (validate_ltx23.py video_connector_inner_dim == 4096). NOT the video inner dim.
EXPECTED_HIDDEN_DIM: int = 4096
#: The VIDEO transformer's attention inner dim, read from the VIDEO ``blocks.0.attn1.to_q.weight[0]``
#: (num_heads*head_dim = 32*128 = 4096). Confirmed 4096 on the real A100 preflight once the
#: safetensors read excludes the AUDIO dual-stream (``audio_transformer_blocks...attn1.to_q`` = 2048,
#: which a loose substring match wrongly picked up first). Same value as EXPECTED_HIDDEN_DIM here.
EXPECTED_VIDEO_INNER_DIM: int = 4096
EXPECTED_NUM_HEADS: int = 32
EXPECTED_HEAD_DIM: int = 128
EXPECTED_VAE_TEMPORAL_COMPRESSION: int = 8
EXPECTED_VAE_SPATIAL_COMPRESSION: int = 32
EXPECTED_T2V_IN_CHANNELS: int = 128


# --------------------------------------------------------------------------------------------------
# D-7-BASEVAR (REF-03) — base-variant selection.
#
# ``ModelConfig.model_id`` switches the base checkpoint by filename: ``ltx-2.3-22b-dev.safetensors``
# (the DEFAULT / proof base — the seg overfit + grid stay on dev) vs
# ``ltx-2.3-22b-distilled-*.safetensors`` (the distilled two-stage base — a supported config option,
# fail-fast paired with ``validation.two_stage_upscale`` in ``config/schema.py``).
#
# Both variants are the SAME LTX-2.3 22B transformer architecture: distillation changes the
# *inference schedule* (fewer denoise steps + spatial upscaler), NOT the block stack / attention
# dims / VAE compression. So both variants assert against the SAME ``EXPECTED_*`` constants above.
# ``expected_arch()`` is the single seam that states this explicitly (rather than silently letting an
# arbitrary checkpoint pass) — and the single place a future distilled-specific arch fact would hang
# off, WITHOUT duplicating the ``EXPECTED_*`` literals anywhere.
# --------------------------------------------------------------------------------------------------

#: The base checkpoint variants signet-trainer supports (D-7-BASEVAR). ``dev`` stays the default.
SUPPORTED_BASE_VARIANTS: tuple[str, ...] = ("dev", "distilled")


def base_variant_of(checkpoint_path: str) -> str:
    """Classify a checkpoint filename as the ``dev`` (default/proof) or ``distilled`` base variant.

    D-7-BASEVAR: ``config.model.model_id`` already selects dev/distilled by filename. This returns
    which variant so the loader can log it and so ``expected_arch`` can key off it — a distilled
    checkpoint is recognised by ``"distilled"`` in the filename; everything else is the dev base.
    """
    name = Path(checkpoint_path).name.lower()
    return "distilled" if "distilled" in name else "dev"


def expected_arch(checkpoint_path: str | None = None) -> dict[str, int]:
    """Return the ``EXPECTED_*`` arch constants for a base variant (dev == distilled on LTX-2.3 22B).

    Single source of the arch facts: both supported variants map to the SAME values because the
    distilled base is the dev transformer with a distilled inference schedule, not a smaller net.
    The ``checkpoint_path`` argument makes the variant-awareness explicit at the call site (and is
    the seam to diverge should a future variant ever change block/dim counts) — today both branches
    read the module-level ``EXPECTED_*`` constants, so there are no duplicated literals.
    """
    variant = base_variant_of(checkpoint_path) if checkpoint_path is not None else "dev"
    # dev and distilled share the 22B transformer arch; no per-variant divergence today. Keeping the
    # branch explicit documents the equality as a checked fact rather than an accident.
    if variant not in SUPPORTED_BASE_VARIANTS:  # pragma: no cover — defensive, base_variant_of is total
        raise ValueError(f"unsupported base variant {variant!r} (supported: {SUPPORTED_BASE_VARIANTS})")
    return {
        "num_blocks": EXPECTED_NUM_BLOCKS,
        "hidden_dim": EXPECTED_HIDDEN_DIM,
        "video_inner_dim": EXPECTED_VIDEO_INNER_DIM,
        "num_heads": EXPECTED_NUM_HEADS,
        "head_dim": EXPECTED_HEAD_DIM,
        "vae_temporal_compression": EXPECTED_VAE_TEMPORAL_COMPRESSION,
        "vae_spatial_compression": EXPECTED_VAE_SPATIAL_COMPRESSION,
        "t2v_in_channels": EXPECTED_T2V_IN_CHANNELS,
    }


def load_ltxv_components(
    checkpoint_path: str,
    text_encoder_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    with_video_vae_decoder: bool = False,
    with_text_encoder: bool = True,
    with_audio_vae_encoder: bool = False,
) -> "LtxModelComponents":
    """Load the LTX-2.3 components signet-trainer needs (video-first; audio ENCODER opt-in for a2v).

    Ports ltx-trainer's ``model_loader.load_model`` for the video-only case:
        * with_video_vae_encoder=True            — needed for pre-encoding (Phase 2/3)
        * with_video_vae_decoder=<param, def False> — training OFF (Phase 2/3);
          inference flips it True (Phase 4) so the ValidationSampler can decode latents
          back to pixels. The DEFAULT stays False so every existing Phase-2/3 training
          call site (which never passes the flag) is byte-for-byte unchanged.
        * with_audio_vae_decoder=False           — a2v generates VIDEO only (no audio OUTPUT);
          the audio decoder + vocoder stay OFF (GATE-SPEC rev 2 item 6: decoder/vocoder OFF).
        * with_vocoder=False                     — never emit an audio waveform
        * with_text_encoder=True                 — Gemma
        * with_audio_vae_encoder=<param, def False> — a2v ONLY (GATE-SPEC rev 2 item 6): the audio
          VAE ENCODER turns a driving ``.wav`` into audio latents at sample time. Default False so
          every existing (video-only) call site is byte-identical.

    ⚠ AT-PIN DIVERGENCE (verified via ``gh`` against Lightricks/LTX-2 @ d6053703): the upstream
    ``load_model`` does NOT expose a ``with_audio_vae_encoder`` kwarg (its audio knob is
    ``with_audio_vae_decoder``, and ``LtxModelComponents`` carries no ``audio_vae_encoder`` field).
    The pin DOES expose the standalone ``model_loader.load_audio_vae_encoder(checkpoint_path,
    device, dtype) -> AudioEncoder``. So the encoder is loaded via that dedicated function and
    ATTACHED as a dynamic ``components.audio_vae_encoder`` attribute (mirroring how
    ``embeddings_processor`` is attached below) rather than threaded as a ``load_model`` kwarg. The
    FULL base transformer (which carries the audio dual-stream weights) is loaded regardless — the
    weights are never stripped (BUGLOG#6 meta-device crash; signet already complies).

    On the inference path (``with_video_vae_decoder=True``) this also loads the Gemma
    ``embeddings_processor`` (ltx-trainer ``load_embeddings_processor``) and attaches it
    to the returned bundle as ``.embeddings_processor`` so ``inference.sampler.run_sampler``
    can pass it into the ``ValidationSampler`` (the training path never reads it, so the
    attribute is only set when the decoder is on).

    Args:
        checkpoint_path: Path to the LTX-2.3 ``.safetensors`` checkpoint (e.g.
            ``ltx-2.3-22b-dev.safetensors`` under WEIGHTS_DIR).
        text_encoder_path: Path to the Gemma text-encoder DIRECTORY containing
            ``model*.safetensors`` (``gemma-3-12b-it`` per decision A1 — NOT the
            ``-qat`` variant).
        device: Torch device string; "cuda" Modal-side (the 22B checkpoint + Gemma need VRAM).
        dtype: Component dtype (bfloat16, matching enochiatron).
        with_video_vae_decoder: Load the video VAE DECODER (Phase-4 inference). Default
            False — training/pre-encoding never needs the decoder, so training callers are
            unaffected. ``fns.py::sample`` passes True.
        with_text_encoder: Load the Gemma text encoder. Default True — every existing
            call site is byte-identical. ``fns.py::sample`` passes False for the
            multi_frame two-phase load (06-09 OOM fix): prompts are pre-encoded in a
            Gemma-only PHASE A before the 22B transformer is ever loaded, so the
            A100-80GB never holds both (runs 3-5 proved they cannot coexist with
            Gemma's forward activations, and SingleGPUModelBuilder's ``assign=True``
            loader-owned storages cannot be freed by in-place ``.to("cpu")`` parking).

    Returns:
        ``LtxModelComponents`` exposing ``.transformer / .video_vae_encoder /
        .text_encoder / .scheduler`` (``scheduler`` is a stateless ``LTX2Scheduler()``);
        ``.audio_vae_decoder / .vocoder`` are ``None`` (video-only). On the inference path
        ``.video_vae_decoder`` is populated and ``.embeddings_processor`` is attached; on
        the training path (default) ``.video_vae_decoder`` is ``None`` and no
        ``embeddings_processor`` is set.

    Note:
        ``ltx_trainer`` is imported here (function-local) on purpose — see the module
        docstring. This call runs Modal-side ONLY; do not invoke it from the dry-run or
        config-load path.
    """
    from ltx_trainer.model_loader import load_model

    # D-7-BASEVAR: honor the dev (default/proof) vs distilled base selected via ModelConfig.model_id.
    # Both variants are the same 22B architecture (expected_arch() is dev==distilled), so no arch
    # branch is needed here — the checkpoint_path drives the load; we log the resolved variant so the
    # metered run records which substrate loaded (distilled must pair with the two-stage inference
    # path, fail-fast enforced at config load in config/schema.py).
    variant = base_variant_of(checkpoint_path)
    logger.info("Loading LTX-2.3 base variant %r from %s", variant, checkpoint_path)

    components = load_model(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=device,
        dtype=dtype,
        with_video_vae_encoder=True,
        with_video_vae_decoder=with_video_vae_decoder,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=with_text_encoder,
    )

    if with_audio_vae_encoder:
        # a2v ONLY (GATE-SPEC rev 2 item 6): load the audio VAE ENCODER via the pin's dedicated
        # ``load_audio_vae_encoder`` (the ONLY audio-encoder surface at d6053703 — load_model has
        # no with_audio_vae_encoder kwarg, verified via gh) and attach it as a dynamic
        # ``components.audio_vae_encoder``. Loaded on the compute device (a driving .wav is encoded
        # at sample time). Decoder + vocoder stay OFF — a2v emits video only. Signature verified
        # against the pinned source: load_audio_vae_encoder(checkpoint_path, device, dtype).
        from ltx_trainer.model_loader import load_audio_vae_encoder

        # DTYPE PARITY (render-port risk #1, 2026-07-15): the pin encodes TRAINING audio latents at
        # float32 ("Audio VAE needs float32 for quality", process_videos.py:532). Loading the
        # inference-side encoder at the components dtype (bf16) would encode the DRIVING audio in a
        # different numeric regime than every latent the adapter trained on — a silent train/infer
        # distribution mismatch. float32 here restores exact parity; the encoder is tiny, VRAM
        # impact negligible.
        import torch  # noqa: PLC0415

        components.audio_vae_encoder = load_audio_vae_encoder(
            checkpoint_path,
            device,
            torch.float32,
        )
        logger.info(
            "Loaded audio VAE encoder at float32 (train/infer parity, a2v driving-audio path) — "
            "decoder/vocoder stay OFF."
        )

    if with_video_vae_decoder:
        # Inference path only: the ValidationSampler needs Gemma's embeddings processor
        # to turn prompts into the transformer's cross-attention context. Loaded here
        # (function-local heavy import, mirroring load_model) and attached to the bundle
        # so run_sampler can read it via getattr(components, "embeddings_processor", None).
        # The live GPU proof of this arg set rides the next gated run (D-RUN-1 part 2).
        from ltx_trainer.model_loader import load_embeddings_processor

        # Signature verified against the port source (enochiatron scripts/infer/compare.py:423,
        # generate.py:1062): load_embeddings_processor takes checkpoint_path (the LTX MODEL
        # checkpoint — same value load_model receives), NOT text_encoder_path (that kwarg belongs
        # to load_model). enochiatron loads it on CPU (device='cpu') for its sequential-VRAM
        # discipline; the ValidationSampler moves it as needed.
        components.embeddings_processor = load_embeddings_processor(
            checkpoint_path=checkpoint_path,
            device="cpu",
            dtype=dtype,
        )

    return components


def _count_blocks(transformer: Any) -> int | None:
    """Count transformer blocks by direct introspection (landmine #3 fix).

    The real LTX-2.3 transformer has no ``num_blocks``/``num_layers`` scalar attr, so
    ``summarize_components``'s ``_probe`` chain returns None. Resolve the count directly:

      1. ``len(transformer.transformer_blocks)`` — the canonical diffusers container name;
      2. ``len(transformer.blocks)`` — the alternate (Wan-style) container name;
      3. count distinct ``(?:transformer_)?blocks.<i>.`` indices over
         ``transformer.named_modules()`` keys → ``max(idx) + 1`` (the same block-key
         convention as tests/test_ground_truth_read.py's safetensors pre-check).

    Returns the resolved int, or None if no block container/keys are found.
    """
    if transformer is None:
        return None

    for attr in ("transformer_blocks", "blocks"):
        container = getattr(transformer, attr, None)
        if container is not None:
            try:
                n = len(container)
            except TypeError:
                n = None
            if n:
                return n

    named_modules = getattr(transformer, "named_modules", None)
    if callable(named_modules):
        block_indices: set[int] = set()
        for name, _ in named_modules():
            match = _BLOCK_IDX_RE.search(name)
            if match:
                block_indices.add(int(match.group(1)))
        if block_indices:
            return max(block_indices) + 1

    return None


def summarize_components(components: Any) -> dict[str, Any]:
    """Best-effort shape/arch summary of loaded components for the GPU smoke-test (D-04).

    Returns a dict of the architecture facts the smoke-test prints + asserts against the
    EXPECTED_* constants. Tolerant of attribute-name drift in the ltx-core models (Pitfall 6:
    the API is underdocumented) — it probes a small set of likely config locations and falls
    back to ``None`` rather than raising, so the smoke-test can print what it found and assert
    only on what is present.

    Invoked Modal-side ONLY (operates on a loaded ``LtxModelComponents``).
    """
    transformer = getattr(components, "transformer", None)

    def _probe(obj: Any, *names: str) -> Any:
        for name in names:
            # direct attribute
            if hasattr(obj, name):
                return getattr(obj, name)
            # nested under a `.config` object
            cfg = getattr(obj, "config", None)
            if cfg is not None and hasattr(cfg, name):
                return getattr(cfg, name)
        return None

    # Landmine #3 fix (D-9-P2-CARRY): the real LTX-2.3 transformer exposes the block stack
    # as a CONTAINER with no num_blocks/num_layers scalar attr, so `_count_blocks` is the
    # AUTHORITATIVE source — call it UNCONDITIONALLY (not merely as a `_probe` fallback).
    # On the real arch `_probe` returns None anyway, but keying off `_count_blocks` first
    # means the true container count can never be masked by a stray/incorrect scalar attr.
    # Fall back to the scalar `_probe` chain only when NO block container/keys are found
    # (i.e. `_count_blocks` returns None), so a non-LTX transformer that only exposes a
    # scalar depth attr still resolves.
    num_blocks = _count_blocks(transformer)
    if num_blocks is None:
        num_blocks = _probe(transformer, "num_blocks", "num_layers", "depth", "n_layers")
    hidden_dim = _probe(transformer, "hidden_dim", "hidden_size", "dim", "inner_dim")
    num_heads = _probe(transformer, "num_heads", "num_attention_heads", "n_heads")
    in_channels = _probe(transformer, "in_channels", "in_dim")

    return {
        "num_blocks": num_blocks,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "in_channels": in_channels,
        "has_video_vae_encoder": getattr(components, "video_vae_encoder", None) is not None,
        "has_text_encoder": getattr(components, "text_encoder", None) is not None,
        "has_scheduler": getattr(components, "scheduler", None) is not None,
    }
