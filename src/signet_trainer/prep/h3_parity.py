"""D-10-DEF-7 — the CLASS-closing guard for the ``Qwen3VLProcessor.__call__`` bypass.

``prep/h3_encode.build_h3_processor_inputs`` deliberately does NOT go through
``processor(text=..., images=...)``. That bypass is correct and must stay: ``build_h3_presentation``
emits each vision block **already expanded**, because its half-open spans are only meaningful over
the expanded stream and ``train/h3_step.h3_token_tags`` consumes them to tag conditioning rows.
``__call__`` wants exactly ONE ``<|image_pad|>`` per image and expands it itself, so handing it the
pre-expanded string exhausts its per-image replacement iterator.

⛔ **But every bypass inherits the obligation to reproduce EVERY output that call makes, and the
gaps surface ONE AT A TIME, each costing a metered container.** Two have been paid for already:

  | attempt | gap | cost |
  |---|---|---|
  | 1 (D-10-DEF-4 defect 1) | the pre-expanded presentation reaching ``__call__`` at all | ~$0.15 |
  | 2 (D-10-DEF-7) | ``mm_token_type_ids``, which ``__call__`` adds and the model REQUIRES | ~$0.15 |

Patching the second one alone would simply buy the third gap at the same price. So this module
closes the CLASS instead: it runs the **REAL** ``processor.__call__`` and our replacement over the
same tiny synthetic input and diffs the **OUTPUT KEY SETS**. A key ``__call__`` produces and we do
not — including one nobody has ever heard of — fails by name.

Two callers, ONE implementation, deliberately:

  1. ``tests/test_h3_processor_output_parity.py`` — the CI gate, run at the PINNED transformers
     version (``modal/app.py::TRANSFORMERS_VERSION``). Zero GPU, zero Modal spend, no model weights.
  2. ``modal/fns.py::h3_preprocess`` — the runtime backstop, run against the REAL mounted processor
     immediately after ``AutoProcessor.from_pretrained`` and BEFORE the Qwen3-VL load, so a gap that
     ever escapes CI costs the processor mount rather than a weights load plus an encode.

They must not drift, which is why the diff lives here and neither caller reimplements it.

⚠ **Version sensitivity is the whole point.** ``return_mm_token_type_ids`` defaults to ``True`` at
the pinned 5.14.1 and to ``False`` at the 5.2.0 that happens to be installed on the authoring box,
where the mask is built inline in the Qwen3-VL processor instead of in the shared ``ProcessorMixin``
(which does not even define ``create_mm_token_type_ids`` at 5.2.0). A parity check executed against
5.2.0 would pass while the real dispatch failed — worse than no check at all, because it
manufactures confidence. The CI gate therefore asserts the running version IS the pin and FAILS
(never skips) when it is not.

Modal-side EXCEPTION tier: ``transformers`` / ``PIL`` imports are FUNCTION-LOCAL, so importing this
module to read its constants needs no backend installed.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------------------------------
# The probe geometry.
# --------------------------------------------------------------------------------------------------

#: Short edge the synthetic probe images are prepared at. SMALL on purpose — the question this
#: module asks ("which keys does ``__call__`` return") is independent of the image size, and a small
#: probe keeps the in-container backstop free. It must still clear the image processor's MINIMUM
#: pixel band (``shortest_edge`` = 65,536 px on the mounted ``preprocessor_config.json``), or
#: ``smart_resize`` would UPSCALE the probe and the realized grid would stop matching
#: ``rows_of(h, w)`` — tripping ``build_h3_processor_inputs``'s geometry assertion on a probe defect
#: rather than a real one. 288x288 = 82,944 px, comfortably inside the band and on the 32 grid.
H3_PARITY_PROBE_SHORT_EDGE: int = 288

#: Reference slots the probe presents. TWO, matching the campaign's ``references_per_sample``: one
#: reference cannot distinguish "one grid per contiguous run" from "one grid per token stream",
#: which is exactly the ``get_rope_index`` invariant ``build_h3_processor_inputs`` now asserts.
H3_PARITY_PROBE_REFERENCES: int = 2

#: The probe caption. Deliberately banal and property-free — it is never trained on, and it is
#: printed into a container log.
H3_PARITY_PROBE_PROMPT: str = "a parity probe caption"

#: ⛔ Keys allowed to differ between ``processor.__call__`` and ``build_h3_processor_inputs``,
#: mapped to the REASON they are allowed to. It is **EMPTY**, and that is the correct state: our
#: builder is documented as reproducing ``BatchFeature(data={**text_inputs, **image_inputs})``
#: exactly, so every key ``__call__`` emits is a key the model may require.
#:
#: An entry here is a deliberate, reasoned exception — never a way to make a red test green. An
#: empty-but-honest allowlist is the point; an over-broad one swallows exactly the gaps this module
#: exists to surface, and would have swallowed ``mm_token_type_ids`` for a third metered container.
H3_PROCESSOR_PARITY_ALLOWLIST: dict[str, str] = {}


# --------------------------------------------------------------------------------------------------
# The probe.
# --------------------------------------------------------------------------------------------------


def build_h3_parity_probe(
    processor: Any,
    *,
    short_edge: int = H3_PARITY_PROBE_SHORT_EDGE,
    n_references: int = H3_PARITY_PROBE_REFERENCES,
    prompt: str = H3_PARITY_PROBE_PROMPT,
) -> dict[str, Any]:
    """One synthetic sample, presented BOTH ways — pre-expanded (ours) and one-pad-per-image.

    Both presentations come from the SAME ``build_h3_presentation`` call form, differing only in the
    ``vision_token_counts`` handed to it: the reference's real count for our path, ``1`` for the
    path ``__call__`` expects and expands itself. Reusing the production presentation builder rather
    than hand-writing the ``__call__`` string is load-bearing — a hand-written probe string would be
    a second source of truth for the label and sentinel layout, and a probe that disagrees with the
    production presentation proves nothing about the production presentation.

    The tokenizer is taken off the processor so OUR spans are EXACT (``build_h3_presentation``
    bills nominal label lengths without one, and nominal spans would fail
    ``_assert_vision_spans_land_on_sentinels`` on the probe rather than on a defect).

    Returns ``images`` (raw PIL), ``prepared`` (``H3PreparedReference`` values), ``presentation`` +
    ``spans`` (OUR pre-expanded form) and ``call_text`` (the one-pad-per-image form).
    """
    from PIL import Image as PILImage  # noqa: PLC0415 — Modal-side EXCEPTION tier

    from signet_trainer.prep.h3_encode import (  # noqa: PLC0415
        H3PresentationRef,
        build_h3_presentation,
        prepare_h3_reference_images,
    )

    if int(n_references) < 1:
        raise ValueError(f"the parity probe needs at least one reference, got {n_references}.")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "[h3-parity] the processor exposes no .tokenizer, so the probe cannot measure EXACT "
            "vision spans and the parity result would describe a broken probe rather than the "
            "builder under test."
        )

    # Flat colours, one distinguishable value per slot, so a probe that silently collapsed two
    # references into one image would not look identical to a correct one under a debugger.
    images = [
        PILImage.new("RGB", (short_edge, short_edge), (96 + 24 * index, 128, 160))
        for index in range(int(n_references))
    ]
    prepared = prepare_h3_reference_images(images, short_edge)

    presentation, spans = build_h3_presentation(
        prompt,
        [
            H3PresentationRef(kind="image", vision_token_counts=(int(ref.vision_tokens),))
            for ref in prepared
        ],
        tokenizer=tokenizer,
    )
    call_text, _ = build_h3_presentation(
        prompt,
        [H3PresentationRef(kind="image", vision_token_counts=(1,)) for _ in prepared],
        tokenizer=tokenizer,
    )
    return {
        "images": images,
        "prepared": prepared,
        "presentation": presentation,
        "spans": spans,
        "call_text": call_text,
    }


def h3_processor_output_key_diff(
    processor: Any,
    *,
    short_edge: int = H3_PARITY_PROBE_SHORT_EDGE,
    n_references: int = H3_PARITY_PROBE_REFERENCES,
) -> dict[str, Any]:
    """Run BOTH paths over one synthetic sample and report their output KEY SETS.

    ⛔ This is the only place in the repo that calls ``processor(text=..., images=...)`` on purpose.
    Everywhere else that call is forbidden (it double-expands the vision blocks); HERE it is the
    ORACLE, and it is fed the one-pad-per-image presentation it expects so it is exercised on its
    own terms.

    Returns ``{"call_keys": [...], "our_keys": [...], "missing": [...], "extra": [...],
    "input_ids_match": bool}`` — sorted lists, so the result is JSON-serializable and can cross a
    subprocess boundary (the CI gate runs the pinned interpreter out of process).

    ``missing`` is the D-10-DEF-7 direction: a key ``__call__`` produces and we do not. ``extra`` is
    the opposite and matters just as much — an unexpected kwarg reaching ``text_encoder(**inputs)``
    is a ``TypeError`` inside a metered container. Both are reported net of
    ``H3_PROCESSOR_PARITY_ALLOWLIST``.
    """
    from signet_trainer.prep.h3_encode import build_h3_processor_inputs  # noqa: PLC0415

    probe = build_h3_parity_probe(
        processor, short_edge=short_edge, n_references=n_references
    )

    reference = processor(
        text=[probe["call_text"]], images=probe["images"], return_tensors="pt"
    )
    ours = build_h3_processor_inputs(
        processor,
        probe["presentation"],
        probe["prepared"],
        vision_spans=probe["spans"],
    )

    call_keys = set(reference.keys())
    our_keys = set(ours.keys())
    allowed = set(H3_PROCESSOR_PARITY_ALLOWLIST)
    return {
        "call_keys": sorted(call_keys),
        "our_keys": sorted(our_keys),
        "missing": sorted(call_keys - our_keys - allowed),
        "extra": sorted(our_keys - call_keys - allowed),
        "allowlisted": sorted(allowed & (call_keys ^ our_keys)),
        # Bonus, and free: our pre-expansion and the processor's own expansion of the SAME
        # presentation must produce the SAME token stream. A False here is D-10-DEF-4's defect (1)
        # in its silent form — the shape stays plausible while the pads stop describing the images.
        "input_ids_match": _same_input_ids(reference.get("input_ids"), ours.get("input_ids")),
        "n_references": int(n_references),
        "short_edge": int(short_edge),
    }


def _same_input_ids(left: Any, right: Any) -> bool:
    """Tensor-or-list equality that does not require both sides to be tensors."""
    if left is None or right is None:
        return False
    to_list = getattr(left, "tolist", None)
    left_ids = to_list() if to_list is not None else list(left)
    to_list = getattr(right, "tolist", None)
    right_ids = to_list() if to_list is not None else list(right)
    return left_ids == right_ids


def assert_h3_processor_output_parity(
    processor: Any,
    *,
    short_edge: int = H3_PARITY_PROBE_SHORT_EDGE,
    n_references: int = H3_PARITY_PROBE_REFERENCES,
) -> str:
    """RAISE unless our builder's output keys match ``processor.__call__``'s. Returns a log line.

    Called from ``h3_preprocess`` right after the processor is mounted and BEFORE Qwen3-VL is
    loaded, so the next gap in this class is a preflight failure measured in seconds rather than a
    weights load plus an encode.
    """
    diff = h3_processor_output_key_diff(
        processor, short_edge=short_edge, n_references=n_references
    )
    if diff["missing"] or diff["extra"]:
        raise RuntimeError(
            f"[h3-parity] build_h3_processor_inputs does not reproduce "
            f"{type(processor).__name__}.__call__'s outputs. MISSING (it produces, we do not): "
            f"{diff['missing']}; EXTRA (we produce, it does not): {diff['extra']}. Its keys were "
            f"{diff['call_keys']}, ours were {diff['our_keys']}.\n"
            f"The bypass of __call__ is deliberate (the pre-expanded presentation is what makes the "
            f"vision spans meaningful to h3_token_tags), but it inherits ALL of that call's "
            f"outputs. A MISSING key is D-10-DEF-7 again: add it to build_h3_processor_inputs using "
            f"the processor's own PUBLIC method for it — never a reimplementation, which would "
            f"re-derive a vocabulary the processor already owns. An EXTRA key becomes an unexpected "
            f"kwarg to text_encoder(**inputs). Only widen H3_PROCESSOR_PARITY_ALLOWLIST with a "
            f"written reason; an allowlist entry added to make this green is how the third metered "
            f"container gets bought."
        )
    if not diff["input_ids_match"]:
        raise RuntimeError(
            f"[h3-parity] our PRE-EXPANDED presentation and {type(processor).__name__}.__call__'s "
            f"own expansion of the same one-pad-per-image presentation produced DIFFERENT token "
            f"streams. That is D-10-DEF-4 defect (1) in its silent form: the shape stays plausible "
            f"while the pads stop describing the images, so the vision spans, the modality tags and "
            f"the packed budget all describe something the model was not shown."
        )
    return (
        f"[h3-parity] OK — {len(diff['call_keys'])} output key(s) match "
        f"{type(processor).__name__}.__call__ exactly ({', '.join(diff['call_keys'])}); "
        f"token streams identical over {diff['n_references']} synthetic reference(s) at "
        f"{diff['short_edge']}px."
    )
