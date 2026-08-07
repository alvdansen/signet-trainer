"""models — the MiniMax-H3 (Ref2VA) architecture contract + Modal-side transformer loader (H3-01).

The H3 sibling of ``models/loader.py``. It is the SINGLE SOURCE of every H3 architecture number in
this repo: nothing else in Phase 10 (LoRA targets, packed-sequence geometry, pre-encode, dry-run
budget, train gate) may re-derive one — they import from here.

Invoked Modal-side ONLY for the actual load (it imports ``diffusers`` + needs GPU VRAM). It is NOT
in the dry-run/import-confined path — keep it out of any module that the dry-run or config load
imports (the deliberate Anti-Pattern-6 EXCEPTION: unlike the import-confined conditioning/data
modules, this loader MAY import ``diffusers`` precisely because it only ever runs Modal-side at call
time, never on Windows/CI).

The heavy ``import diffusers`` is deferred to INSIDE ``load_h3_transformer`` so that merely importing
this module to read the ``EXPECTED_H3_*`` constants (``tests/test_h3_arch_constants.py``, the CPU
dry-run budget check, CI) does NOT require ``diffusers`` to be installed. ``tests/
test_h3_arch_constants.py`` enforces that two ways — a source scan and a subprocess import with
``diffusers`` blocked.

Provenance — these constants were MEASURED, not derived:
    ``scripts/_h3_probe_modal.py`` (``h3_arch_probe``) loaded the real
    ``MiniMaxAI/MiniMax-H3`` ``transformer_ref/`` on a single A100-80GB (diffusers pinned at
    ``9f169d98d0bce392a889c3b6524d0d97734dfc0e``, 2026-08-05) and confirmed 10/10 fields against
    ``model.config`` plus the live ``named_modules`` (``transformer_blocks == 50``,
    ``token_refiner.refiner_blocks == 2``, ``adaln_proj.linear.weight == [96768, 2688]``).
    Recorded in ``.planning/phases/10-.../P10-1-MEASURED.md`` §1 and pinned here against the
    committed ``tests/fixtures/h3_transformer_ref_config.json``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# MEASURED H3 ground-truth constants — the architecture facts the arch gate asserts against.
# Source: scripts/_h3_probe_modal.py:262-273 (live A100 probe) == P10-1-MEASURED.md §1, pinned to
# tests/fixtures/h3_transformer_ref_config.json (the shipped diffusers transformer_ref/config.json).
# These are the single source of truth: tests + every Phase-10 module import them from here
# (do NOT duplicate the values).
# --------------------------------------------------------------------------------------------------

#: DiT depth — live ``len(model.transformer_blocks)`` == 50.
EXPECTED_H3_NUM_LAYERS: int = 50
#: Single-stream model dim (H3 has NO separate audio-branch inner dim — see the note below).
EXPECTED_H3_HIDDEN_SIZE: int = 5376
EXPECTED_H3_NUM_ATTENTION_HEADS: int = 56
EXPECTED_H3_ATTENTION_HEAD_DIM: int = 128
EXPECTED_H3_FFN_DIM: int = 14336
#: Video VAE latent channels (patch_dim = in_channels * prod(patch_size) = 24 * 1 * 2 * 2 = 96).
EXPECTED_H3_IN_CHANNELS: int = 24
EXPECTED_H3_AUDIO_IN_CHANNELS: int = 32
#: Qwen3-VL conditioning width the ``context_embedder`` projects 5120 -> 5376 from.
EXPECTED_H3_TEXT_DIM: int = 5120
EXPECTED_H3_TIME_EMBED_DIM: int = 2688
#: ``token_refiner.refiner_blocks`` — TEXT-stream refiner blocks, NOT part of the generative stack.
#: (They are why a plain LoRA suffix list over-matches on H3; the target form is a path regex.)
EXPECTED_H3_NUM_REFINER_LAYERS: int = 2

#: ``(t, h, w)`` patch size. Kept as a tuple (the diffusers config stores a JSON list).
EXPECTED_H3_PATCH_SIZE: tuple[int, int, int] = (1, 2, 2)

#: ``transformer_blocks[0].adaln_proj.linear.weight`` on the diffusers path, bf16 — the FULL
#: projection, confirmed on live weights (P10-1-MEASURED.md §1, resolving the P10-0e discrepancy).
#: ⚠ ``[96768, 8]`` is the ComfyUI **pruned baked-bottleneck** form and does NOT occur on the
#: diffusers path — so no bottleneck handling exists in this repo BY DESIGN, and arbitrary-timestep
#: training is addressable normally (the adaln_t table is continuous + lerped). If a load ever
#: reports ``[96768, 8]``, the wrong weight set is mounted; the gate refuses it.
EXPECTED_H3_ADALN_PROJ_SHAPE: tuple[int, int] = (96768, 2688)

#: adaln modality tags: ``0=video 1=text 2=audio``, addressed ``timestep_indices * 3 + token_tags``.
#: (Qwen vision rows live in the TEXT stream but modulate as VIDEO, tag 0 — P10-1-MEASURED.md §7.)
EXPECTED_H3_MODALITY_NUM: int = 3

#: ⚠⚠ MANDATORY WARNING — this is the TEXT-ENCODER hidden-state index, NOT the DiT depth.
#: H3 reads Qwen3-VL ``hidden_states[50]`` out of its **64** layers
#: (``modular_pipeline.py::text_encoder_layer``): the final layer is post-norm and is NOT the
#: conditioning the released weights were trained against. A pre-encode that takes Qwen3-VL's FINAL
#: hidden state is **silently wrong** — it produces valid-shaped ``(B, L, 5120)`` conditioning that
#: trains against off-distribution text features, with no shape error to catch it. That is why this
#: is an ASSERTED arch constant and not a comment (P10-1-MEASURED.md §6).
#: Its equality with ``EXPECTED_H3_NUM_LAYERS = 50`` is a NUMERICAL COINCIDENCE (P10-0d §2). The two
#: must never be conflated or derived from each other.
EXPECTED_H3_TEXT_ENCODER_LAYER: int = 50

# NOTE (deliberate omission, 10-PATTERNS §1 DIVERGE): there is intentionally NO
# ``EXPECTED_H3_VIDEO_INNER_DIM``-style audio-vs-video disambiguation here. ``models/loader.py``
# needs one because LTX-2.3 is an asymmetric DUAL-STREAM DiT with a separate audio branch dim;
# H3 is SINGLE-STREAM — video, text and audio rows are packed into one sequence at
# ``hidden_size`` and differ only by modality tag. Adding such a constant would invent a
# distinction the architecture does not have.


def expected_h3_arch() -> dict[str, int]:
    """Return the measured H3 arch constants keyed by their DIFFUSERS config field names.

    Keyed to match ``MiniMaxH3Transformer3DModel.config`` exactly so a caller can diff this dict
    straight against ``model.config`` (or against the committed
    ``tests/fixtures/h3_transformer_ref_config.json``) with no name translation.

    Unlike ``models/loader.py::expected_arch`` there is no variant argument: H3 ships one
    ``transformer_ref`` partition for Ref2VA. A LoRA trained on Ref2VA is architecturally
    interchangeable with an FL2VA one (byte-identical config) but NOT semantically transferable
    (different base weights) — that is a weights-selection concern, not an arch-constant one.

    No literals live in this body — it reads the module-level constants, which are the single
    source of truth.
    """
    return {
        "num_layers": EXPECTED_H3_NUM_LAYERS,
        "hidden_size": EXPECTED_H3_HIDDEN_SIZE,
        "num_attention_heads": EXPECTED_H3_NUM_ATTENTION_HEADS,
        "attention_head_dim": EXPECTED_H3_ATTENTION_HEAD_DIM,
        "ffn_dim": EXPECTED_H3_FFN_DIM,
        "in_channels": EXPECTED_H3_IN_CHANNELS,
        "audio_in_channels": EXPECTED_H3_AUDIO_IN_CHANNELS,
        "text_dim": EXPECTED_H3_TEXT_DIM,
        "time_embed_dim": EXPECTED_H3_TIME_EMBED_DIM,
        "num_refiner_layers": EXPECTED_H3_NUM_REFINER_LAYERS,
    }


def _probe(obj: Any, *names: str) -> Any:
    """Attribute-drift-tolerant read: direct attribute, THEN ``.config``, else ``None``.

    Copied from ``models/loader.py::summarize_components``'s inner ``_probe`` — the diffusers H3
    module exposes its arch facts on ``.config``, but a PEFT-wrapped or otherwise proxied model may
    surface some of them directly. Returning ``None`` rather than raising is what lets
    ``assert_h3_arch`` assert only on what it actually has ground truth for.
    """
    for name in names:
        # direct attribute
        if hasattr(obj, name):
            return getattr(obj, name)
        # nested under a `.config` object
        cfg = getattr(obj, "config", None)
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
    return None


def load_h3_transformer(
    checkpoint_dir: str,
    device: str = "cuda",
    dtype: Any = None,
) -> Any:
    """Load the MiniMax-H3 ``transformer_ref`` partition (Modal-side ONLY).

    Mirrors the working load in ``scripts/_h3_probe_modal.py`` (measured: ~61.7 GiB resident on a
    single A100-80GB, bf16). The heavy ``diffusers`` import is function-local on purpose — see the
    module docstring; ``tests/test_h3_arch_constants.py`` enforces it.

    Args:
        checkpoint_dir: Directory holding the diffusers-format shards + ``config.json`` — the
            **``transformer_ref/``** partition (Ref2VA), e.g. ``<WEIGHTS_DIR>/minimax-h3/
            transformer_ref``. NOT a ``.safetensors`` FILE (H3's model IDs are directories, which
            is why ``loader.py::base_variant_of``'s filename sniff does not transfer).
        device: Torch device string; ``"cuda"`` Modal-side (the 61.7 GiB partition needs VRAM).
        dtype: Component dtype. ``None`` (default) resolves to ``torch.bfloat16``, the dtype the
            arch probe measured clean.

    Returns:
        The loaded ``MiniMaxH3Transformer3DModel``, moved to ``device``. Feed it to
        ``summarize_h3_transformer`` + ``assert_h3_arch`` BEFORE any sustained spend.

    Note:
        ``diffusers`` is imported here (function-local) on purpose. This call runs Modal-side ONLY;
        do not invoke it from the dry-run or config-load path.
    """
    from diffusers import MiniMaxH3Transformer3DModel  # noqa: PLC0415 — deliberate function-local

    resolved_dtype = torch.bfloat16 if dtype is None else dtype
    logger.info("Loading MiniMax-H3 transformer from %s (dtype=%s)", checkpoint_dir, resolved_dtype)

    model = MiniMaxH3Transformer3DModel.from_pretrained(checkpoint_dir, torch_dtype=resolved_dtype)
    return model.to(device)


def summarize_h3_transformer(model: Any) -> dict[str, Any]:
    """Best-effort arch summary of a loaded H3 transformer, for ``assert_h3_arch``.

    Reads every ``expected_h3_arch()`` field through ``_probe`` (attribute-drift tolerant), plus
    three LIVE introspections that the config alone cannot prove:

        * ``live_transformer_blocks`` — ``len(model.transformer_blocks)``
        * ``live_refiner_blocks``     — ``len(model.token_refiner.refiner_blocks)``
        * ``adaln_proj_shape``        — ``transformer_blocks[0].adaln_proj.linear.weight.shape``

    Every probe yields ``None`` rather than raising when the attribute is absent, so the gate can
    print what it found and assert only on what is present (the ``load_ltxv_smoke`` tolerance).
    """
    summary: dict[str, Any] = {field: _probe(model, field) for field in expected_h3_arch()}

    patch_size = _probe(model, "patch_size")
    summary["patch_size"] = None if patch_size is None else tuple(patch_size)

    summary["live_transformer_blocks"] = _live(lambda: len(model.transformer_blocks))
    summary["live_refiner_blocks"] = _live(lambda: len(model.token_refiner.refiner_blocks))
    summary["adaln_proj_shape"] = _live(
        lambda: tuple(model.transformer_blocks[0].adaln_proj.linear.weight.shape)
    )
    return summary


def _live(read: Any) -> Any:
    """Run a live introspection, returning ``None`` on ANY failure (missing attr, empty container).

    Deliberately broad: this runs on a freshly-loaded 61.7 GiB model Modal-side, and a probe that
    raises would abort the gate BEFORE it could report the fields it did manage to read.
    """
    try:
        return read()
    except Exception:  # noqa: BLE001 — a probe must never be the thing that fails the gate
        return None


def assert_h3_arch(summary: dict[str, Any]) -> None:
    """Raise ``RuntimeError`` naming EVERY mismatched arch field. The abort-before-spend gate.

    Compares a ``summarize_h3_transformer`` summary against ``expected_h3_arch()`` plus the three
    live facts (``live_transformer_blocks``, ``live_refiner_blocks``, ``adaln_proj_shape``). Fields
    the summary reports as ``None`` (or omits) are SKIPPED — we assert only what we have ground
    truth for — but they are named in the failure message so a partially-blind gate is never
    mistaken for a clean one.

    Every offending field is reported, not just the first: enochiatron's equivalent gate caught 6
    arch mismatches in one ~$1.40 run precisely because it did not stop at the first.
    """
    expected: dict[str, Any] = dict(expected_h3_arch())
    expected["live_transformer_blocks"] = EXPECTED_H3_NUM_LAYERS
    expected["live_refiner_blocks"] = EXPECTED_H3_NUM_REFINER_LAYERS
    expected["adaln_proj_shape"] = EXPECTED_H3_ADALN_PROJ_SHAPE

    mismatches: list[str] = []
    skipped: list[str] = []
    for field, want in expected.items():
        got = summary.get(field)
        if got is None:
            skipped.append(field)
            continue
        if isinstance(want, tuple):
            got = tuple(got)
        if got != want:
            mismatches.append(f"{field}: expected {want}, got {got}")

    if not mismatches:
        return

    note = f" (not asserted — probe returned None: {', '.join(skipped)})" if skipped else ""
    raise RuntimeError(
        "[h3-arch-gate] ARCH MISMATCH — "
        + "; ".join(mismatches)
        + note
        + ". The mounted weights are not the measured MiniMax-H3 transformer_ref partition "
        "(P10-1-MEASURED.md §1). Aborting BEFORE any spend."
    )
