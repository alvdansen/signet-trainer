"""models — the Qwen-Image-Edit-2511 architecture contract + Modal-side component loaders (family #3).

The ``qwen_edit`` sibling of ``models/h3_loader.py`` and ``models/loader.py``. It owns the
architecture facts that only the WEIGHTS can settle — head dim, the ``txt_in`` width, the presence
of the Qwen2.5-VL vision tower — and it owns the three component loads plus the qfloat8 pass.

It owns NO geometry. Every packing/latent number (``60`` blocks, hidden ``3072``, the ``64``-wide
packed row, ``16`` latent channels, the ``2x2`` patch) is IMPORTED from
``conditioning/qwen_edit_geometry.py``, and the LoRA target contract is IMPORTED from
``config/validators.py``. That direction is deliberate and matches ``h3_ref.py:64``: a second copy
of a number is worse than a wrong one, because no reader ever sees the two side by side. The only
literal architecture number declared here is the attention head dim, which no other module needs.

Invoked Modal-side ONLY for the actual loads (they import ``diffusers`` / ``transformers`` /
``optimum.quanto`` and need VRAM). The heavy imports are FUNCTION-LOCAL for the same reason
``h3_loader``'s are: merely importing this module to read the ``EXPECTED_QWEN_EDIT_*`` constants,
or to run the four CPU-pure assertions below, must work on Windows/CI with none of those installed.
Module scope is ``logging`` + ``typing`` + ``torch`` + two import-light signet modules.

Provenance — these constants were MEASURED, not derived
-------------------------------------------------------
Read off the live checkpoint HEADERS on this box (safetensors 8-byte length + JSON header; no
tensor body was ever materialised), 2026-08-08::

    F:/AI-Models/ComfyUI-models/diffusion_models/qwen_image_edit_2511_bf16.safetensors
      40,861,031,560 B  1934 tensors  transformer_blocks 0..59 (60)
      img_in.weight   [3072, 64]     txt_in.weight  [3072, 3584]
      proj_out.weight [64, 3072]     norm_out.linear.weight [6144, 3072]
      transformer_blocks.0.attn.norm_q.weight [128]        -> attention_head_dim
      transformer_blocks.0.img_mod.1.weight   [18432, 3072]
      top-level: __index_timestep_zero__, img_in, norm_out, proj_out, time_text_embed,
                 transformer_blocks, txt_in, txt_norm
      the 14-leaf path regex fullmatched 840 module names, 14 distinct leaves x 60, and matched
      ZERO of the 240 ``attn.norm_{q,k,added_q,added_k}`` RMSNorms

    F:/AI-Models/ComfyUI-models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
      9,384,670,680 B  1446 tensors  714 of them under ``visual.``
      top-level: lm_head, model, scaled_fp8, visual

    F:/AI-Models/ComfyUI-models/text_encoders/_MISLABELED_was-qwen2.5-7b-instruct-not-VL.safetensors
      15,231,272,152 B  339 tensors  **0** under ``visual.``   top-level: lm_head, model
      ^ the artifact of the day this house lost. See ``assert_qwen_edit_text_encoder_vision``.

    F:/AI-Models/ComfyUI-models/vae/qwen-image/qwen_image_vae.safetensors
      253,806,246 B  194 tensors  top-level: conv1, conv2, decoder, encoder

⚠ ORDER — quantize, THEN inject LoRA. Never the reverse.
--------------------------------------------------------
ai-toolkit quantizes inside ``load_model()`` (``qwen_image.py:127`` transformer,
``qwen_image.py:166-169`` text encoder), and ``load_model()`` is called at
``jobs/process/BaseSDTrainProcess.py:1619`` — a hundred and ninety lines BEFORE the adapter is
attached at ``:1770`` / ``:1809``. Every proven house chain therefore trained a qfloat8 base with a
bf16 adapter on top. :func:`assert_qwen_edit_not_peft_wrapped` makes the reverse order a named
refusal rather than a thing to remember.

The full intended sequence, and the reason for each step's position::

    load  ->  assert_qwen_edit_arch  ->  assert_qwen_edit_targets  ->  quantize_qwen_edit
                                                                   ->  lora.peft.inject_lora

Both assertions run BEFORE the quantization because they are nearly free and quantization is not:
``optimum.quanto`` rewrites 840+ ``nn.Linear`` leaves in place, and an arch mismatch discovered
after that is a mismatch discovered for the second time, on a metered container. (Module NAMES
survive quantization — ``_quantize_submodule`` does a ``setattr`` on the parent under the same
attribute — so the target survey would still be correct afterwards. It is ordered first for cost,
not for correctness.)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_BLOCK_COUNT,
    QWEN_EDIT_HIDDEN_DIM,
    QWEN_EDIT_LATENT_CHANNELS,
    QWEN_EDIT_PATCH,
    QWEN_EDIT_PATCH_DIM,
)
from signet_trainer.config.validators import (
    QWEN_EDIT_LORA_LEAVES,
    QWEN_EDIT_LORA_TARGET_REGEX,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EXPECTED_QWEN_EDIT_ATTENTION_HEAD_DIM",
    "EXPECTED_QWEN_EDIT_GUIDANCE_EMBEDS",
    "EXPECTED_QWEN_EDIT_IMG_IN_SHAPE",
    "EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM",
    "EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT",
    "EXPECTED_QWEN_EDIT_NUM_ATTENTION_HEADS",
    "EXPECTED_QWEN_EDIT_PROJ_OUT_SHAPE",
    "EXPECTED_QWEN_EDIT_TRANSFORMER_TENSOR_COUNT",
    "EXPECTED_QWEN_EDIT_TXT_IN_SHAPE",
    "EXPECTED_QWEN_VL_TENSOR_COUNT",
    "EXPECTED_QWEN_VL_VISION_TENSOR_COUNT",
    "QWEN_EDIT_BLOCK_LIST_ATTR",
    "QWEN_EDIT_QUANT_QTYPE",
    "QWEN_VL_VISION_PREFIX",
    "assert_qwen_edit_arch",
    "assert_qwen_edit_not_peft_wrapped",
    "assert_qwen_edit_targets",
    "assert_qwen_edit_text_encoder_vision",
    "expected_qwen_edit_arch",
    "load_qwen_edit_text_encoder",
    "load_qwen_edit_transformer",
    "load_qwen_edit_vae",
    "quantize_qwen_edit",
    "qwen_vl_vision_census",
    "summarize_qwen_edit_transformer",
]

# --------------------------------------------------------------------------------------------------
# MEASURED ground truth. See the module docstring for the header read that produced each one.
# Anything derivable from ``conditioning/qwen_edit_geometry`` is DERIVED here, never retyped.
# --------------------------------------------------------------------------------------------------

#: The ONLY literal architecture number this module declares. MEASURED:
#: ``transformer_blocks.0.attn.norm_q.weight`` is ``[128]`` — an RMSNorm over one attention head, so
#: its single dimension IS the head dim. It lives here rather than in ``qwen_edit_geometry`` because
#: nothing in the packing/budget arithmetic needs it: geometry prices SEQUENCES, this prices HEADS.
EXPECTED_QWEN_EDIT_ATTENTION_HEAD_DIM: int = 128

#: DERIVED, not measured separately: ``hidden // head_dim`` == 3072 // 128 == 24. Deriving is the
#: point — a hand-typed 24 could disagree with the two numbers it is supposed to be the quotient of,
#: and nothing about the resulting model would look wrong until attention silently reshaped.
EXPECTED_QWEN_EDIT_NUM_ATTENTION_HEADS: int = (
    QWEN_EDIT_HIDDEN_DIM // EXPECTED_QWEN_EDIT_ATTENTION_HEAD_DIM
)

#: The Qwen2.5-VL conditioning width ``txt_in`` projects 3584 -> 3072 from. MEASURED:
#: ``txt_in.weight == [3072, 3584]``.
#:
#: ⚠ This number DISCHARGES a written IOU. ``config/schema.py::QwenEditConfig.text_embed_dim``
#: carries ``[UNVERIFIED]`` and instructs: *"The weight-loading pass MUST assert it against the real
#: txt_in and correct it here if it differs; nothing downstream may treat it as measured until
#: then."* This is that pass, and the value the config declared (3584) is CONFIRMED. The assertion
#: itself is :func:`assert_qwen_edit_arch`'s ``config_text_embed_dim`` argument — a config that
#: drifts from the live weight is now a named refusal, not a shape error 60 blocks later.
EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM: int = 3584

#: ``guidance_embeds`` is FALSE for this checkpoint, established by ABSENCE: the only
#: ``time_text_embed`` tensors present are ``timestep_embedder.linear_{1,2}.{weight,bias}`` — there
#: is no guidance embedder to load. A ``True`` config would build one and then leave it at random
#: init, which trains but is not this model.
EXPECTED_QWEN_EDIT_GUIDANCE_EMBEDS: bool = False

#: ``(out_features, in_features)`` of the three projections that bracket the block stack. Each is
#: DERIVED from a geometry constant, so they are a cross-check of geometry against live weights
#: rather than a second declaration of it:
#:   ``img_in``   packed row (64)   -> hidden (3072)
#:   ``txt_in``   VL width  (3584)  -> hidden (3072)
#:   ``proj_out`` hidden    (3072)  -> packed row (64) == patch**2 * latent_channels
EXPECTED_QWEN_EDIT_IMG_IN_SHAPE: tuple[int, int] = (QWEN_EDIT_HIDDEN_DIM, QWEN_EDIT_PATCH_DIM)
EXPECTED_QWEN_EDIT_TXT_IN_SHAPE: tuple[int, int] = (
    QWEN_EDIT_HIDDEN_DIM,
    EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM,
)
EXPECTED_QWEN_EDIT_PROJ_OUT_SHAPE: tuple[int, int] = (QWEN_EDIT_PATCH_DIM, QWEN_EDIT_HIDDEN_DIM)

#: Tensor count in the single-file checkpoint. A weak check on its own (it cannot say WHICH tensors)
#: but a very cheap "is this the 2511 release at all" tell for a header-only pre-flight.
EXPECTED_QWEN_EDIT_TRANSFORMER_TENSOR_COUNT: int = 1934

#: 14 leaves x 60 blocks. DERIVED from the two single sources — add a leaf to
#: ``QWEN_EDIT_LORA_LEAVES`` and this updates for free, which is the whole reason it is not ``840``.
EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT: int = QWEN_EDIT_BLOCK_COUNT * len(QWEN_EDIT_LORA_LEAVES)

#: ``nn.ModuleList`` attribute holding the DiT stack. ai-toolkit names the same thing at
#: ``qwen_image.py:398-399`` (``get_transformer_block_names() -> ["transformer_blocks"]``), which is
#: what its block-at-a-time quantization walks; :func:`quantize_qwen_edit` walks the same attribute
#: for the same reason.
QWEN_EDIT_BLOCK_LIST_ATTR: str = "transformer_blocks"

#: Key prefix of the Qwen2.5-VL VISION tower. The live encoder exposes it as a top-level ``visual.``
#: group in the single file, and as ``text_encoder.model.visual`` on the loaded module — which is
#: exactly the attribute ai-toolkit deletes when it does NOT need vision
#: (``qwen_image.py:150-152``) and deliberately KEEPS for edit-plus
#: (``qwen_image_edit_plus.py:46`` sets ``_qwen_image_keep_visual = True``).
QWEN_VL_VISION_PREFIX: str = "visual."

#: MEASURED on ``qwen_2.5_vl_7b_fp8_scaled.safetensors``: 714 vision tensors out of 1446 total.
#: The mislabeled text-only file measured 339 / **0**.
EXPECTED_QWEN_VL_VISION_TENSOR_COUNT: int = 714
EXPECTED_QWEN_VL_TENSOR_COUNT: int = 1446

#: The house recipe's quantization type, as ``optimum.quanto`` spells it. ai-toolkit's default is
#: the same string (``toolkit/config_modules.py:642-643``: ``qtype`` and ``qtype_te`` both default
#: to ``"qfloat8"``), and ``optimum.quanto.qfloat8`` resolves to ``quanto.qfloat8_e4m3fn``.
#: Carried as a STRING, not the qtype object, so this module imports without ``optimum`` installed.
QWEN_EDIT_QUANT_QTYPE: str = "qfloat8"


def expected_qwen_edit_arch() -> dict[str, Any]:
    """Return the measured arch constants keyed by their DIFFUSERS config field names.

    Keyed to match ``QwenImageTransformer2DModel.config`` exactly (``diffusers``
    ``models/transformers/transformer_qwenimage.py:525-535``) so a caller can diff this dict
    straight against ``model.config`` with no name translation — the ``expected_h3_arch`` contract.

    ``patch_size``, ``in_channels`` and ``out_channels`` are the geometry module's constants under
    diffusers' names, not new numbers: diffusers builds ``proj_out = Linear(inner_dim, patch_size *
    patch_size * out_channels)`` (``transformer_qwenimage.py:562``), and ``2**2 * 16 == 64`` is
    precisely ``QWEN_EDIT_PATCH_DIM``. Asserting them is therefore a live cross-check that the
    checkpoint agrees with the packing arithmetic every other qwen module is priced by.

    No literals live in this body — it reads module-level constants and imported geometry, which are
    the single sources of truth.
    """
    return {
        "patch_size": QWEN_EDIT_PATCH,
        "in_channels": QWEN_EDIT_PATCH_DIM,
        "out_channels": QWEN_EDIT_LATENT_CHANNELS,
        "num_layers": QWEN_EDIT_BLOCK_COUNT,
        "attention_head_dim": EXPECTED_QWEN_EDIT_ATTENTION_HEAD_DIM,
        "num_attention_heads": EXPECTED_QWEN_EDIT_NUM_ATTENTION_HEADS,
        "joint_attention_dim": EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM,
        "guidance_embeds": EXPECTED_QWEN_EDIT_GUIDANCE_EMBEDS,
    }


def _probe(obj: Any, *names: str) -> Any:
    """Attribute-drift-tolerant read: direct attribute, THEN ``.config``, else ``None``.

    Byte-identical in behaviour to ``h3_loader._probe`` (itself copied from
    ``models/loader.py::summarize_components``). Duplicated rather than imported so that
    ``models/h3_loader`` and ``models/qwen_edit_loader`` stay independent of each other — the two
    families share no arch facts, and a cross-import would make an H3 refactor able to change what
    a Qwen gate reads. Returning ``None`` rather than raising is what lets the assert below check
    only the fields it actually has ground truth for.
    """
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        cfg = getattr(obj, "config", None)
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
    return None


def _live(read: Any) -> Any:
    """Run a live introspection, returning ``None`` on ANY failure (missing attr, empty container).

    Deliberately broad, for ``h3_loader._live``'s reason: this runs against a freshly-loaded ~40 GiB
    transformer Modal-side, and a probe that raises would abort the gate BEFORE it could report the
    fields it did manage to read. A probe must never be the thing that fails the gate.
    """
    try:
        return read()
    except Exception:  # noqa: BLE001 — a probe must never be the thing that fails the gate
        return None


def _shape_of(module: Any, path: str) -> Any:
    """``tuple(model.<path>.weight.shape)`` or ``None``. ``path`` is dotted, e.g. ``"txt_in"``."""
    target = module
    for part in path.split("."):
        target = getattr(target, part)
    return tuple(target.weight.shape)


def summarize_qwen_edit_transformer(model: Any) -> dict[str, Any]:
    """Best-effort arch summary of a loaded Qwen-Image-Edit transformer, for :func:`assert_qwen_edit_arch`.

    Reads every :func:`expected_qwen_edit_arch` field through :func:`_probe` (attribute-drift
    tolerant), plus four LIVE introspections the config alone cannot prove:

        * ``live_transformer_blocks`` — ``len(model.transformer_blocks)``
        * ``img_in_shape``            — ``model.img_in.weight.shape``   (packed row width -> hidden)
        * ``txt_in_shape``            — ``model.txt_in.weight.shape``   (the VL width, the IOU above)
        * ``proj_out_shape``          — ``model.proj_out.weight.shape`` (hidden -> packed row width)

    The three weight shapes are the ones that matter most, and they are here rather than only in the
    config for a specific reason: ``num_layers`` and ``joint_attention_dim`` are what the model was
    CONFIGURED with, while these are what it was LOADED with. A single-file load against a wrong
    config builds a wrong-shaped module and then fills what fits — the config would read perfect and
    the weights would not be the ones on disk.

    Every probe yields ``None`` rather than raising when the attribute is absent, so the gate can
    print what it found and assert only on what is present.
    """
    summary: dict[str, Any] = {field: _probe(model, field) for field in expected_qwen_edit_arch()}
    summary["live_transformer_blocks"] = _live(
        lambda: len(getattr(model, QWEN_EDIT_BLOCK_LIST_ATTR))
    )
    summary["img_in_shape"] = _live(lambda: _shape_of(model, "img_in"))
    summary["txt_in_shape"] = _live(lambda: _shape_of(model, "txt_in"))
    summary["proj_out_shape"] = _live(lambda: _shape_of(model, "proj_out"))
    return summary


def assert_qwen_edit_arch(summary: dict[str, Any], *, config_text_embed_dim: int | None = None) -> None:
    """Raise ``RuntimeError`` naming EVERY mismatched arch field. The abort-before-spend gate.

    Compares a :func:`summarize_qwen_edit_transformer` summary against
    :func:`expected_qwen_edit_arch` plus the four live facts. Fields the summary reports as ``None``
    (or omits) are SKIPPED — we assert only what we have ground truth for — but they are NAMED in
    the failure message, so a partially-blind gate is never mistaken for a clean one.

    Every offending field is reported, not just the first, for ``assert_h3_arch``'s recorded reason:
    enochiatron's equivalent gate caught 6 arch mismatches in one ~$1.40 run precisely because it
    did not stop at the first.

    Args:
        summary: the dict from :func:`summarize_qwen_edit_transformer`.
        config_text_embed_dim: when given, ``QwenEditConfig.text_embed_dim`` from the run config.
            It is checked against the LIVE ``txt_in`` in-features, which discharges the ``[UNVERIFIED]``
            IOU that field carries (``config/schema.py``: *"The weight-loading pass MUST assert it
            against the real txt_in"*). Passing ``None`` skips the check — the transformer gate is
            still complete without it; this argument exists so a caller that HAS the config does not
            have to re-derive the comparison.
    """
    expected: dict[str, Any] = dict(expected_qwen_edit_arch())
    expected["live_transformer_blocks"] = QWEN_EDIT_BLOCK_COUNT
    expected["img_in_shape"] = EXPECTED_QWEN_EDIT_IMG_IN_SHAPE
    expected["txt_in_shape"] = EXPECTED_QWEN_EDIT_TXT_IN_SHAPE
    expected["proj_out_shape"] = EXPECTED_QWEN_EDIT_PROJ_OUT_SHAPE

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

    # The config-vs-weights cross-check. Compared against the LIVE shape when we have one, so the
    # answer comes from the loaded tensor rather than from a config field echoing another config.
    if config_text_embed_dim is not None:
        live_txt_in = summary.get("txt_in_shape")
        live_width = (
            int(tuple(live_txt_in)[1])
            if live_txt_in is not None
            else EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM
        )
        if int(config_text_embed_dim) != live_width:
            mismatches.append(
                f"qwen_edit.text_embed_dim: config declares {int(config_text_embed_dim)}, the live "
                f"txt_in projects from {live_width}"
            )

    if not mismatches:
        return

    note = f" (not asserted — probe returned None: {', '.join(skipped)})" if skipped else ""
    raise RuntimeError(
        "[qwen-edit-arch-gate] ARCH MISMATCH — "
        + "; ".join(mismatches)
        + note
        + ". The mounted weights are not the measured Qwen-Image-Edit-2511 transformer "
        "(40,861,031,560 B / 1934 tensors / 60 blocks / hidden 3072 / txt_in [3072, 3584]). "
        "A single-file load against the WRONG config is the likeliest cause: diffusers builds the "
        "module from the config and then fills whatever fits, so a wrong config yields a "
        "wrong-shaped model with a clean-looking config and no load error. Aborting BEFORE any spend."
    )


def assert_qwen_edit_targets(model_or_names: Any) -> dict[str, Any]:
    """Assert the 14-leaf path regex resolves EXACTLY 840 modules, per leaf. CPU-pure.

    ``model_or_names`` is an ``nn.Module`` (its ``named_modules()`` names are used) or a plain
    iterable of module-name strings — the ``check_lora_targets_regex`` duck-typing, so this runs in
    a CPU unit test against names read out of a safetensors HEADER with no weights loaded at all.
    That is how it is verified: ``tests/test_qwen_edit_loader.py`` feeds it the real key names.

    Counting is DELEGATED to ``lora/peft.py::check_lora_targets_regex``, which mirrors PEFT exactly
    (``re.fullmatch``, not ``re.search`` — a search-based probe would "match"
    ``...attn.to_q.base_layer``, which PEFT does not inject). The pattern and the leaf list are
    IMPORTED from ``config/validators.py``, today's single source for both. Nothing here re-derives
    either, so this cannot drift from what the config layer already refuses at load.

    The refusal is PER LEAF and not on the grand total alone, for the reason recorded at
    ``modal/fns.py:2717-2746``: a grand total hides a per-leaf ZERO, and a per-leaf zero is exactly
    the "silently trains the wrong thing" failure the regex target form exists to prevent. On this
    family a per-leaf zero has a specific shape — Qwen-Image-Edit is a dual-stream MMDiT, so the
    leaves that would vanish first are the four ``txt_*`` attention projections and the two
    ``*_mod.1`` modulation projections, i.e. exactly the six a ported six-leaf LTX intuition drops.

    Returns:
        The ``check_lora_targets_regex`` survey dict (``pattern`` / ``total`` / ``main`` /
        ``collateral`` / ``collateral_names`` / ``per_leaf``), so a caller can print it.
    """
    # Function-local: ``lora/peft.py`` imports ``peft`` + ``safetensors``, and this module must
    # import on a machine that has neither (the ``h3_loader`` import-confinement contract).
    from signet_trainer.lora.peft import check_lora_targets_regex  # noqa: PLC0415

    survey = check_lora_targets_regex(
        model_or_names,
        QWEN_EDIT_LORA_TARGET_REGEX,
        collateral_markers=(),
        leaves=list(QWEN_EDIT_LORA_LEAVES),
    )

    zero_leaves = [leaf for leaf, count in survey["per_leaf"].items() if int(count) == 0]
    short_leaves = [
        f"{leaf}={count}"
        for leaf, count in survey["per_leaf"].items()
        if int(count) not in (0, QWEN_EDIT_BLOCK_COUNT)
    ]
    total = int(survey["total"])

    if zero_leaves or short_leaves or total != EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT:
        raise RuntimeError(
            f"[qwen-edit-arch-gate] LoRA TARGET MISMATCH — the 14-leaf path regex matched "
            f"total={total}, expected {EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT} "
            f"({QWEN_EDIT_BLOCK_COUNT} blocks x {len(QWEN_EDIT_LORA_LEAVES)} leaves). "
            + (f"ZERO matches for: {zero_leaves}. " if zero_leaves else "")
            + (f"Wrong per-leaf count for: {short_leaves}. " if short_leaves else "")
            + "Measured on the live checkpoint, every one of the fourteen leaves matches exactly "
            f"{QWEN_EDIT_BLOCK_COUNT} modules and the 240 attn.norm_{{q,k,added_q,added_k}} "
            "RMSNorms match none. A short count means the adapter trains fewer modules than the "
            "config priced and cannot warm-start from a 14-leaf primer. Aborting BEFORE any spend."
        )

    if int(survey["collateral"]) != 0:  # pragma: no cover — no marker tuple is passed today
        raise RuntimeError(
            f"[qwen-edit-arch-gate] LoRA TARGET COLLATERAL — {survey['collateral']} matched module(s) "
            f"live off the generative stack: {survey['collateral_names']}."
        )
    return survey


def qwen_vl_vision_census(names: Any) -> dict[str, Any]:
    """Count Qwen2.5-VL VISION tensors/modules among ``names``. CPU-pure, no weights.

    ``names`` is any iterable of dotted names — safetensors KEYS read from a header, or
    ``named_parameters()`` names off a loaded encoder. A name counts as vision when
    :data:`QWEN_VL_VISION_PREFIX` starts it or appears as a dotted segment inside it, which covers
    both spellings that occur in practice: the single file stores ``visual.blocks.0.attn.proj.weight``
    at top level, while the loaded transformers module nests it as
    ``model.visual.blocks.0.attn.proj.weight``.

    Returns ``{"total": n, "vision": n_vision, "examples": [...up to 3...]}``.
    """
    listed = list(names)
    vision = [
        n
        for n in listed
        if n.startswith(QWEN_VL_VISION_PREFIX) or f".{QWEN_VL_VISION_PREFIX}" in n
    ]
    return {"total": len(listed), "vision": len(vision), "examples": sorted(vision)[:3]}


def assert_qwen_edit_text_encoder_vision(model_or_names: Any, *, what: str = "text encoder") -> dict[str, Any]:
    """Refuse a text encoder with no VISION tower. The day this house lost, made impossible to repeat.

    ``model_or_names`` is a loaded encoder (its ``named_parameters()`` names are used, falling back
    to ``named_modules()``) or a plain iterable of names — so this is verifiable on CPU against
    safetensors header keys, which is exactly how the test drives it.

    ⚠ WHY THIS IS AN ASSERTION AND NOT A COMMENT. Qwen-Image-Edit-Plus feeds each control image
    through the Qwen2.5-VL **vision tower** as part of the TEXT stream: the pipeline builds
    ``"Picture {i}: <|vision_start|><|image_pad|><|vision_end|>"`` per control image and calls the
    encoder with ``pixel_values`` + ``image_grid_thw``
    (``pipeline_qwenimage_edit_plus.py:240-268``). ai-toolkit keeps the tower alive specifically for
    this arch — ``qwen_image.py:150-152`` deletes ``text_encoder.model.visual`` only when
    ``_qwen_image_keep_visual`` is false, and ``qwen_image_edit_plus.py:46`` sets it TRUE.

    A plain Qwen2.5 text LLM has the same family name, loads without error, produces embeddings of
    the same rank, and then fails deep in the forward with::

        mat1 and mat2 shapes cannot be multiplied (5376x1280 and 3840x1280)

    — a matmul error that names no file, no component and no missing feature. This house already
    hit that and spent a day on it. The evidence is still on disk: the mislabeled checkpoint sits
    beside the correct one as ``_MISLABELED_was-qwen2.5-7b-instruct-not-VL.safetensors``, and a
    header read tells the two apart instantly — 339 tensors and **0** under ``visual.``, against
    1446 tensors and 714 under ``visual.``. This assertion is that header read, promoted to a gate.

    Returns the :func:`qwen_vl_vision_census` dict on success.
    """
    if hasattr(model_or_names, "named_parameters"):
        names: Any = [name for name, _ in model_or_names.named_parameters()]
    elif hasattr(model_or_names, "named_modules"):
        names = [name for name, _ in model_or_names.named_modules()]
    else:
        names = model_or_names

    census = qwen_vl_vision_census(names)
    if census["vision"] > 0:
        return census

    raise RuntimeError(
        f"[qwen-edit-arch-gate] {what}: NO VISION TENSORS — {census['total']} name(s) inspected, "
        f"ZERO under {QWEN_VL_VISION_PREFIX!r}. This is a text-only Qwen2.5 LLM, not Qwen2.5-VL. "
        "Qwen-Image-Edit-Plus routes every CONTROL IMAGE through the VL vision tower as part of the "
        "text stream (pipeline_qwenimage_edit_plus.py:240-268 builds "
        "'Picture i: <|vision_start|><|image_pad|><|vision_end|>' and passes pixel_values + "
        "image_grid_thw), and ai-toolkit keeps that tower alive for exactly this arch "
        "(qwen_image_edit_plus.py:46 sets _qwen_image_keep_visual=True). A text-only encoder loads "
        "cleanly, embeds at the right rank, and then dies deep in the forward with "
        "'mat1 and mat2 shapes cannot be multiplied (5376x1280 and 3840x1280)' — an error that "
        "names neither the file nor the missing half. Expected the VL encoder: "
        f"{EXPECTED_QWEN_VL_TENSOR_COUNT} tensors of which {EXPECTED_QWEN_VL_VISION_TENSOR_COUNT} "
        "are vision. Mount the Qwen2.5-VL weights and re-dispatch. Aborting BEFORE any spend."
    )


def assert_qwen_edit_not_peft_wrapped(model_or_names: Any, *, what: str = "transformer") -> None:
    """Refuse to quantize a model that already carries a PEFT adapter. Enforces the recipe ORDER.

    CPU-pure and name-based: an injected model exposes ``...lora_A.default`` /  ``...lora_B.default``
    sub-Linears and a ``...base_layer`` under every targeted leaf, so the names alone settle it.
    A plain iterable of names is accepted for the same reason as everywhere else in this module.

    The order it enforces is EVIDENCED, not preferred. ai-toolkit quantizes inside ``load_model()``
    (``extensions_built_in/diffusion_models/qwen_image/qwen_image.py:127`` for the transformer,
    ``:166-169`` for the text encoder), and ``load_model()`` is called at
    ``jobs/process/BaseSDTrainProcess.py:1619`` — before the adapter is constructed at ``:1770`` and
    attached at ``:1809``. Every proven house edit chain therefore trained a qfloat8 base carrying a
    bf16 adapter.

    Quantizing afterwards is not a slower route to the same model. ``optimum.quanto`` would walk
    INTO the PEFT wrappers and convert ``lora_A`` / ``lora_B`` — the rank-42 tensors that are the
    entire trainable surface and the entire published artifact — to qfloat8, which is both a
    different optimization problem and an adapter no other round can load.
    """
    if hasattr(model_or_names, "named_modules"):
        names: Any = [name for name, _ in model_or_names.named_modules()]
    else:
        names = list(model_or_names)

    hits = [n for n in names if "lora_" in n or n.endswith("base_layer")]
    if not hits:
        return
    raise RuntimeError(
        f"[qwen-edit-quantize] {what} is ALREADY PEFT-wrapped ({len(hits)} adapter/base_layer "
        f"module(s), e.g. {hits[:3]}). The house order is quantize -> inject, evidenced by "
        "ai-toolkit: quantization happens inside load_model (qwen_image.py:127, :166-169), called "
        "at BaseSDTrainProcess.py:1619, while the adapter is attached at :1809. Quantizing now "
        "would convert lora_A/lora_B — the rank-42 trainable surface and the published artifact — "
        "to qfloat8, producing an adapter no other round of the chain can load. Call "
        "quantize_qwen_edit BEFORE lora.peft.inject_lora."
    )


# --------------------------------------------------------------------------------------------------
# Weight-touching loads. Modal-side ONLY — every backend import below is function-local.
# The CALLER composes every path from ``modal/app.py::WEIGHTS_DIR``, so no path literal lives here
# (D-NOHARDCODE; the ``run_h3_arch_gate`` docstring states the same law at ``fns.py:2620-2622``).
# --------------------------------------------------------------------------------------------------


def load_qwen_edit_transformer(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: Any = None,
    *,
    config_source: str | None = None,
    subfolder: str | None = None,
    local_files_only: bool = True,
) -> Any:
    """Load the Qwen-Image-Edit-2511 transformer (Modal-side ONLY).

    Handles BOTH shapes ai-toolkit handles, sniffed the way ai-toolkit sniffs them
    (``qwen_image.py:96``, ``if model_path.endswith(".safetensors")``):

      * a bare ``.safetensors`` FILE  -> ``QwenImageTransformer2DModel.from_single_file``
      * a diffusers DIRECTORY        -> ``QwenImageTransformer2DModel.from_pretrained``

    Both are live possibilities: ``config/schema.py::ModelConfig.family``'s own docstring records
    that "Qwen-Image-Edit's is a bare .safetensors filename indistinguishable in SHAPE from LTX's",
    the shipped example config sets ``model_id: qwen_image_edit_2511_bf16.safetensors``, and
    ai-toolkit's own default path is the diffusers repo directory.

    ⚠ ``config_source`` IS REQUIRED ON THE SINGLE-FILE PATH, and this is not a style preference.
    ``from_single_file`` with no ``config=`` calls ``fetch_diffusers_config(checkpoint)``
    (``diffusers/loaders/single_file_model.py:385``), which calls ``infer_diffusers_model_type``
    (``single_file_utils.py:566``) — and that function has **no Qwen branch at all** on the pinned
    diffusers. It falls through to ``model_type = "v1"``, whose default path is
    ``{"pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5"}``
    (``single_file_utils.py:160``). diffusers would then reach the HUB for a Stable Diffusion v1.5
    ``transformer/`` config, inside a metered container, and fail with an error about SD1.5.
    ai-toolkit works around exactly this by hard-coding ``config="Qwen/Qwen-Image"``
    (``qwen_image.py:99``) — a HUB repo id. Signet takes the same value from the caller instead, so
    an air-gapped Volume-local config directory is expressible and ``local_files_only`` can stay on.

    Args:
        checkpoint_path: ``.safetensors`` file OR diffusers directory on the mounted weights Volume.
            The CALLER composes it (``WEIGHTS_DIR / cfg.model.model_id``).
        device: torch device string; ``"cuda"`` Modal-side.
        dtype: component dtype. ``None`` resolves to ``torch.bfloat16`` — the dtype the checkpoint
            is stored in (every measured tensor read ``BF16``).
        config_source: a diffusers config location — a local directory on the Volume, or a hub repo
            id. REQUIRED for the single-file path (see above); optional for the directory path,
            where the config travels with the weights.
        subfolder: passed through to diffusers. ``"transformer"`` when ``config_source`` /
            ``checkpoint_path`` points at a full pipeline root rather than the component directory.
        local_files_only: default ``True`` — a load inside a metered container should not silently
            depend on hub reachability. Set ``False`` only when ``config_source`` is deliberately a
            hub repo id and the container is expected to fetch it.

    Returns:
        The loaded ``QwenImageTransformer2DModel`` on ``device``. Feed it to
        :func:`summarize_qwen_edit_transformer` + :func:`assert_qwen_edit_arch` and
        :func:`assert_qwen_edit_targets` BEFORE any sustained spend, and only then to
        :func:`quantize_qwen_edit`.

    Note:
        ``diffusers`` is imported function-local on purpose — see the module docstring. This call
        runs Modal-side ONLY; do not invoke it from the dry-run or config-load path.
    """
    from diffusers import QwenImageTransformer2DModel  # noqa: PLC0415 — deliberate function-local

    resolved_dtype = torch.bfloat16 if dtype is None else dtype
    is_single_file = checkpoint_path.endswith(".safetensors")

    if is_single_file and config_source is None:
        raise RuntimeError(
            f"[qwen-edit-loader] config_source is REQUIRED to load the single-file checkpoint "
            f"{checkpoint_path!r}. diffusers' from_single_file infers the config from the checkpoint "
            "keys (single_file_model.py:385 -> single_file_utils.py:566), and the pinned diffusers "
            "has NO Qwen branch in infer_diffusers_model_type: it falls through to model_type='v1' "
            "and would try to load a Stable Diffusion v1.5 transformer config from the HUB "
            "(single_file_utils.py:160), inside this container, at cost. Pass a Volume-local "
            "diffusers config directory (preferred — keeps local_files_only=True honest) or the "
            "hub repo id ai-toolkit hard-codes for this arch, 'Qwen/Qwen-Image' "
            "(qwen_image.py:99), with local_files_only=False."
        )

    logger.info(
        "Loading Qwen-Image-Edit transformer from %s (%s, dtype=%s, config_source=%s)",
        checkpoint_path,
        "single file" if is_single_file else "diffusers dir",
        resolved_dtype,
        config_source,
    )

    if is_single_file:
        model = QwenImageTransformer2DModel.from_single_file(
            checkpoint_path,
            config=config_source,
            subfolder=subfolder or "",  # None is NOT accepted: transformers does os.path.join(subfolder, path)
            torch_dtype=resolved_dtype,
            local_files_only=local_files_only,
        )
    else:
        model = QwenImageTransformer2DModel.from_pretrained(
            checkpoint_path,
            subfolder=subfolder or "",  # None is NOT accepted: transformers does os.path.join(subfolder, path)
            torch_dtype=resolved_dtype,
            local_files_only=local_files_only,
        )
    return model.to(device)


def load_qwen_edit_text_encoder(
    encoder_path: str,
    device: str = "cuda",
    dtype: Any = None,
    *,
    subfolder: str | None = None,
    local_files_only: bool = True,
) -> Any:
    """Load the Qwen2.5-VL text encoder WITH its vision tower, frozen (Modal-side ONLY).

    Mirrors ai-toolkit's load (``qwen_image.py:145-147``:
    ``Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_path, subfolder="text_encoder")``)
    and then does the two things ai-toolkit's training loop does NOT need but signet's pre-encode
    does:

    1. **Keeps the vision tower and asserts it is there.** ai-toolkit reaches the same end by not
       executing the delete at ``qwen_image.py:151-152``; signet asserts instead, because "we did
       not delete it" is only true if it was ever present. See
       :func:`assert_qwen_edit_text_encoder_vision` for the day this cost.
    2. **Freezes it at the LOAD site** via ``prep/h3_grad_contract.freeze_h3_component`` — ``.eval()``
       AND ``requires_grad_(False)``, verified rather than assumed. ``.eval()`` alone leaves
       ``requires_grad=True`` on every parameter of a freshly-``from_pretrained``-ed component, and
       an encode outside a no-grad context then retains a graph for a backward that never comes.
       That is D-10-DEF-10, which killed an 80 GiB A100 on a 544 MiB allocation on the H3 leg. The
       helper carries an ``h3_`` name because that is the module the contract was WRITTEN in, not
       because the contract is H3-specific; reusing it is D-9-DEDUP, and copying it here to get a
       nicer prefix would be the exact duplication that lets one copy drift.

    Args:
        encoder_path: directory on the mounted weights Volume — the CALLER composes it
            (``WEIGHTS_DIR / cfg.model.text_encoder_id``). A directory, not a file: transformers'
            ``from_pretrained`` needs ``config.json`` beside the weights, so a bare ComfyUI-style
            single ``.safetensors`` is not loadable by this path.
        device: torch device string.
        dtype: ``None`` resolves to ``torch.bfloat16``.
        subfolder: ``"text_encoder"`` when ``encoder_path`` is a full pipeline root.
        local_files_only: default ``True``; see :func:`load_qwen_edit_transformer`.

    Returns:
        The frozen ``Qwen2_5_VLForConditionalGeneration`` on ``device``, vision tower asserted.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415 — function-local

    from signet_trainer.prep.h3_grad_contract import freeze_h3_component  # noqa: PLC0415

    resolved_dtype = torch.bfloat16 if dtype is None else dtype
    logger.info(
        "Loading Qwen2.5-VL text encoder from %s (subfolder=%s, dtype=%s)",
        encoder_path,
        subfolder,
        resolved_dtype,
    )
    encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        encoder_path,
        subfolder=subfolder or "",  # None is NOT accepted: transformers does os.path.join(subfolder, path)
        torch_dtype=resolved_dtype,
        local_files_only=local_files_only,
    )
    census = assert_qwen_edit_text_encoder_vision(
        encoder, what="Qwen2.5-VL text encoder"
    )
    logger.info(
        "Qwen2.5-VL vision tower present: %d/%d parameter names under %r",
        census["vision"],
        census["total"],
        QWEN_VL_VISION_PREFIX,
    )
    return freeze_h3_component(encoder.to(device), what="qwen_edit Qwen2.5-VL text encoder")


def load_qwen_edit_vae(
    vae_path: str,
    device: str = "cuda",
    dtype: Any = None,
    *,
    subfolder: str | None = None,
    local_files_only: bool = True,
) -> Any:
    """Load the Qwen-Image ``AutoencoderKLQwenImage``, frozen (Modal-side ONLY).

    Mirrors ai-toolkit ``qwen_image.py:172-174``. Frozen at the load site for
    :func:`load_qwen_edit_text_encoder`'s reason (D-10-DEF-10) — the VAE is the component that
    failure was actually measured on.

    ⚠ This is a 3D/causal VIDEO VAE used on stills. It requires 5-D ``(B, C, F, H, W)`` input
    (``autoencoder_kl_qwenimage.py:783`` unpacks five dims) and has **no** ``scaling_factor`` /
    ``shift_factor``: per-channel ``latents_mean`` / ``latents_std`` 16-vectors replace them
    (``:687-688``). The encode-side handling of both facts belongs to ``prep/qwen_edit_encode.py``,
    not here — this function's contract ends at "a frozen VAE on the right device". They are named
    here only so the next reader does not discover them from a rank error.

    Args:
        vae_path: file or directory on the mounted weights Volume — the CALLER composes it
            (``WEIGHTS_DIR / cfg.model.vae_id``).
        device: torch device string.
        dtype: ``None`` resolves to ``torch.bfloat16``.
        subfolder: ``"vae"`` when ``vae_path`` is a full pipeline root.
        local_files_only: default ``True``; see :func:`load_qwen_edit_transformer`.
    """
    from diffusers import AutoencoderKLQwenImage  # noqa: PLC0415 — deliberate function-local

    from signet_trainer.prep.h3_grad_contract import freeze_h3_component  # noqa: PLC0415

    resolved_dtype = torch.bfloat16 if dtype is None else dtype
    logger.info("Loading Qwen-Image VAE from %s (subfolder=%s, dtype=%s)", vae_path, subfolder, resolved_dtype)
    vae = AutoencoderKLQwenImage.from_pretrained(
        vae_path,
        subfolder=subfolder or "",  # None is NOT accepted: transformers does os.path.join(subfolder, path)
        torch_dtype=resolved_dtype,
        local_files_only=local_files_only,
    )
    return freeze_h3_component(vae.to(device), what="qwen_edit Qwen-Image VAE")


#: Class names ``optimum.quanto`` has ALREADY converted. Byte-identical to ai-toolkit's ``Q_MODULES``
#: (``toolkit/util/quantize.py:25-33``). Matched by CLASS NAME rather than ``isinstance`` for
#: ai-toolkit's implicit reason and one explicit one: ``QLinear`` SUBCLASSES ``nn.Linear``, so an
#: ``isinstance(m, nn.Linear)`` test cannot tell a converted leaf from an unconverted one, and that
#: is exactly the distinction the second pass depends on.
_QUANTO_ALREADY_CONVERTED: frozenset[str] = frozenset(
    {
        "QLinear",
        "QConv2d",
        "QEmbedding",
        "QBatchNorm2d",
        "QLayerNorm",
        "QConvTranspose2d",
        "QEmbeddingBag",
    }
)


def _quantize_leaves(module: Any, weights: Any, *, what: str) -> int:
    """Convert every not-yet-converted quantizable leaf under ``module``. Returns how many.

    A faithful port of ai-toolkit's corrected ``quantize`` loop
    (``toolkit/util/quantize.py:99-119``) minus its exception swallow — see
    :func:`quantize_qwen_edit`'s docstring for the measured crash this exists to avoid and for the
    one deliberate divergence.

    ``named_modules()`` is materialised into a list before iterating because
    ``_quantize_submodule`` REPLACES modules on their parent via ``set_module_by_name`` while the
    walk is in progress; iterating the live generator would be mutation during traversal.
    """
    from optimum.quanto.quantize import _quantize_submodule  # noqa: PLC0415 — function-local

    converted = 0
    for name, leaf in list(module.named_modules()):
        if leaf.__class__.__name__ in _QUANTO_ALREADY_CONVERTED:
            continue
        before = leaf.__class__.__name__
        try:
            _quantize_submodule(module, name, leaf, weights=weights)
        except Exception as exc:  # noqa: BLE001 — re-raised immediately, named
            raise RuntimeError(
                f"[qwen-edit-quantize] {what}: failed to quantize {name!r} "
                f"({leaf.__class__.__name__}): {exc}. ai-toolkit swallows this and prints "
                "'Failed to quantize {name}', leaving an unknown subset of the component in bf16 "
                "while the run continues — a silent recipe change inside a metered container. "
                "Signet refuses instead."
            ) from exc
        # ``_quantize_submodule`` is a no-op for a leaf with no quantized counterpart; count only
        # the ones it actually replaced, so the log line reports work done rather than modules seen.
        replaced = module.get_submodule(name) if name else module
        if replaced.__class__.__name__ != before:
            converted += 1
    return converted


def quantize_qwen_edit(
    model: Any,
    *,
    what: str,
    qtype: str = QWEN_EDIT_QUANT_QTYPE,
    stage_device: str | None = None,
) -> Any:
    """qfloat8-quantize a Qwen-Image-Edit component in place, ai-toolkit's way (Modal-side ONLY).

    The house recipe locks ``qfloat8`` on BOTH the transformer and the text encoder, and
    ``QwenEditConfig``'s docstring records that this is deliberately NOT a config field: *"The house
    recipe locks the quantization (qfloat8 model + text encoder) the same way it locks the
    optimizer."* ``qtype`` is a parameter here only so the value has one name; do not wire it to a
    config knob.

    **Transcribed from ai-toolkit** (``toolkit/util/quantize.py::quantize_model``, the
    no-accuracy-recovery-adapter branch at ``:290-315``), which does two passes and not one:

      1. **Block by block.** Walk ``get_transformer_block_names()`` — ``["transformer_blocks"]`` for
         this family (``qwen_image.py:398-399``) — and ``quantize(block)`` + ``freeze(block)`` each
         in turn. Doing the stack a block at a time is what keeps peak memory near one block rather
         than near the whole model, since ``freeze`` materialises the quantized weight and releases
         the source.
      2. **Then the whole module.** Catches "the extras" — ``img_in``, ``txt_in``, ``proj_out``,
         ``norm_out.linear`` and the two ``time_text_embed`` projections — which pass 1 never reached.

    A component with no such attribute (the text encoder) simply takes pass 2, which is exactly what
    ai-toolkit does for it: ``quantize(text_encoder, weights=get_qtype(qtype_te)); freeze(...)``
    (``qwen_image.py:166-169``).

    ⛔ WHY THIS DOES NOT CALL ``optimum.quanto.quantize`` — MEASURED, not assumed.
    Pass 2 necessarily re-walks the blocks pass 1 already converted, and quanto's own ``quantize``
    has no guard against re-quantizing a ``QLinear``: ``_quantize_submodule`` calls
    ``quantize_module`` unconditionally, ``QLinear`` subclasses ``nn.Linear`` so the dispatch table
    matches it, and ``QModuleMixin.from_module`` then does ``qmodule.weight.copy_(module.weight)``
    on an already-quantized weight. An earlier draft of this function did exactly that and was run
    on CPU against a 4-block synthetic module under optimum-quanto 0.2.4::

        [qwen-edit-quantize] toy transformer: quantizing 4 transformer_blocks to qfloat8
        [qwen-edit-quantize] toy transformer: quantizing the remaining modules to qfloat8
          File ".../optimum/quanto/tensor/qbytes_ops.py", line 121, in copy_
            assert dest.qtype == src.qtype
        AttributeError: 'Parameter' object has no attribute 'qtype'

    :func:`_quantize_leaves` is therefore a faithful port of ai-toolkit's loop
    (``toolkit/util/quantize.py:99-119``), whose ``if m.__class__.__name__ in Q_MODULES: continue``
    guard is what makes the two-pass recipe work at all. That guard is UNDOCUMENTED in ai-toolkit —
    the comment at ``:24`` explains only the include/exclude bug — and it is the load-bearing half.
    Two independent reasons not to call quanto's ``quantize`` here, then:

      * the already-quantized crash above (fatal, and only reachable on the second pass);
      * ``quantize.py:0.2.4`` still ships ``include = [include] if isinstance(include, str) else
        exclude``, which silently substitutes ``exclude`` for a list-valued ``include``. This port
        takes no filter at all, so it cannot be hit here — but it is the reason the whole loop, and
        not just the guard, is carried locally.

    The one place this port DELIBERATELY diverges from ai-toolkit: ai-toolkit wraps each module in
    ``try/except`` and prints ``f"Failed to quantize {name}: {e}"`` with ``raise e`` commented out
    (``quantize.py:118-119``). Swallowing a per-module failure inside a metered container would leave
    an unknown subset of the model in bf16 with the run continuing — a silent recipe change. This
    port raises, naming the module.

    ⚠ ``optimum.quanto.freeze`` is NOT ``requires_grad_(False)``. It materialises quantized weights
    and drops the originals. Freezing FOR AUTOGRAD is a separate act, done at the load site by
    ``freeze_h3_component``. Two operations, one English word; conflating them leaves a component
    trainable that everyone believes is frozen.

    Args:
        model: the loaded component, NOT yet PEFT-wrapped (refused if it is).
        what: named in every message and log line — the transformer and the text encoder both come
            through here and a failure must say which.
        qtype: ``optimum.quanto`` qtype name; the recipe's ``"qfloat8"``.
        stage_device: when set, mirror ai-toolkit's low-VRAM staging — move each block to this
            device, quantize + freeze it, and park it back on CPU (``quantize.py:304-308``). ``None``
            (the default) quantizes in place on whatever device the component already sits on, which
            is the right choice when the whole component already fits, and is one fewer thing to get
            wrong. It is a parameter rather than a config field because it is a MEMORY strategy, not
            a recipe term: it changes where bytes are, never what the resulting weights are.

    Returns:
        The same component, quantized in place (quanto mutates; the return is for chaining).
    """
    # Function-local: ``optimum.quanto`` is a Modal-runtime dependency and is NOT installed in the
    # CPU/CI environment where this module's constants and assertions are unit-tested.
    from optimum.quanto import freeze, qtypes  # noqa: PLC0415 — deliberate function-local

    assert_qwen_edit_not_peft_wrapped(model, what=what)

    if qtype not in qtypes:
        raise ValueError(
            f"[qwen-edit-quantize] unknown optimum.quanto qtype {qtype!r} for {what}. The house "
            f"recipe is {QWEN_EDIT_QUANT_QTYPE!r} (ai-toolkit's default for both the transformer "
            f"and the text encoder, config_modules.py:642-643). Known: {sorted(qtypes)}."
        )
    weights = qtypes[qtype]

    blocks = getattr(model, QWEN_EDIT_BLOCK_LIST_ATTR, None)
    if blocks is not None:
        block_list = list(blocks)
        logger.info(
            "[qwen-edit-quantize] %s: quantizing %d %s to %s%s",
            what,
            len(block_list),
            QWEN_EDIT_BLOCK_LIST_ATTR,
            qtype,
            f" (staged via {stage_device})" if stage_device else "",
        )
        for index, block in enumerate(block_list):
            if stage_device is not None:
                block.to(stage_device, non_blocking=True)
            _quantize_leaves(block, weights, what=f"{what} {QWEN_EDIT_BLOCK_LIST_ATTR}[{index}]")
            freeze(block)
            if stage_device is not None:
                block.to("cpu", non_blocking=True)

    # Pass 2 — "the extras": img_in / txt_in / proj_out / norm_out.linear / time_text_embed, or, for
    # a component with no block list, the entire module. The blocks pass 1 already converted are
    # re-walked here and SKIPPED by the ``Q_MODULES`` guard — see the docstring for what happens
    # without it.
    converted = _quantize_leaves(model, weights, what=what)
    logger.info(
        "[qwen-edit-quantize] %s: quantized %d remaining module(s) to %s", what, converted, qtype
    )
    freeze(model)
    return model
