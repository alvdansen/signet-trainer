"""prep.h3_text_payload — the SELF-DESCRIBING ``h3_conditions/`` payload, written once, read once.

⛔ **This module exists because two intra-repo promises were never kept.** Both were found by
reading the pinned diffusers/transformers tree against this repo, not by a container:

**(1) The vision spans were computed, logged, and then dropped.** ``h3_ref.py`` read
``batch.get("vision_spans", ())`` and **nothing at train time ever populated it**. ``modal/fns.py``
said in a comment that "the train side recomputes them rather than depending on a span sidecar this
stage does not write" — and no train-side code performed that recomputation. So every training step
ran with ``vision_spans=()``. At short edge 896 that is ~1,000-1,600 vision rows per reference
against ~100 prompt tokens, so **more than 90% of the text stream modulated with the TEXT AdaLN row
instead of the VIDEO one** (``timestep_indices * 3 + token_tags``). No shape error, an entirely
ordinary loss curve, and the wrong modulation on every step. The dry-run did not catch it because
``dryrun/shapes.py`` calls ``build_h3_packed_batch`` DIRECTLY with its own derived nominal spans —
correct tags in the dry-run batch, empty ones in the real one.

**(2) A "dropped" reference step was not reference-free.** D-10-REFDROP removed the reference LATENT
rows but the cached text state still carried both references' Qwen vision blocks, so the dropped
regime matched no inference regime that exists — at the pin, a no-reference request's presentation
contains no vision blocks at all. Identity leaked into 20% of steps while the model was being told
there were no references.

**(3) The regime that wrote a cache was never recorded IN the cache (#31 finding 2).**
``build_h3_text_payload`` enforced ``has_references`` as an ARGUMENT at write time and then dropped
it on the floor — the returned dict carried no field a reader could check it against. A cache
written at ``references_per_sample: 0`` (prompt-only text, empty spans) and one written at
``references_per_sample: 2`` (Ref2VA text, real spans) were then indistinguishable BY SHAPE to
``read_h3_text_state``: same keys, same dtypes, only the SPAN CONTENT differs, and an empty-spans
reference-bearing payload is nonsensical only from the writer's side, not the reader's. An operator
who re-runs ``--mode preprocess`` at 0 slots over an existing Ref2VA-cache root overwrites
``h3_conditions/`` in place with NO-REFERENCE text while ``h3_reference_latents/`` is untouched —
and a later train that flips back to the original slot count reads a same-version, wrong-regime
cache with no raise: real reference-latent rows packed against a text presentation that describes
none of them. The fix is the same shape as (1) and (2): the writer already knows the regime, so it
WRITES IT DOWN, and the reader refuses a ``has_references=False`` cache when the run's own
``references_per_sample`` is ``>= 1``. The MIRROR direction is deliberately NOT refused — a
``references_per_sample == 0`` run always selects the prompt-only state regardless of what the
cache's OTHER key holds, which is how a no-reference train stays able to reuse an existing Ref2VA
cache without a re-encode (see ``H3RefStrategy.prepare_training_inputs``'s ``no_reference`` branch).

The fix for all three is the same shape: **PHASE A already knows these things, so PHASE A writes them
down.** The payload carries the reference-bearing hidden state WITH its spans, a second
**prompt-only** hidden state whose spans are empty BY CONSTRUCTION, and the ``has_references`` flag
that says which regime the SAMPLE (not the per-step dropout draw) was encoded under. ``H3RefStrategy``
selects between the two states by the dropout draw, and separately checks the recorded regime
against its own ``references_per_sample``.

Versioning, and why a stale cache must be REFUSED
-------------------------------------------------
Version 1 was a bare ``(L, 5120)`` tensor. It is indistinguishable-by-duck-typing from the tensor
inside version 2, so a reader that merely "accepts a tensor or a dict" would consume a v1 cache
silently — with no spans and no prompt-only state, i.e. exactly defects (1) and (2) above, restored.
There are **88 v1 payloads on the dataset Volume right now** (the partial cache the D-10-DEF-9 run
committed before failing). They are NOT deleted — never auto-delete an intermediate — so the reader
is what has to refuse them, by shape, with the re-encode named. Version 3 adds ``has_references``
(defect 3) — a version-2 cache has every OTHER field a version-3 reader wants, which is exactly why
it must still be refused by version number rather than read for the fields that happen to match: the
one field that would catch a same-version regime mismatch is the field a v2 cache does not have.

⚠ **This costs a PHASE A re-encode**: two Qwen3-VL passes per sample instead of one. That is
expected and was decided deliberately — defect (1) needs the re-encode regardless, so (2) and (3)
ride along in the same pass over the corpus rather than buying a second or third one.

Import tier: ``torch`` + stdlib only. Both the WRITER (``prep/h3_encode`` via ``modal/fns``) and the
READER (``conditioning/h3_ref``) import this module, which is the whole point — one format, one
place, no second opinion about what a cached text condition is.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "H3_TEXT_PAYLOAD_HAS_REFERENCES_KEY",
    "H3_TEXT_PAYLOAD_PROMPT_ONLY_KEY",
    "H3_TEXT_PAYLOAD_PROMPT_ONLY_SPANS_KEY",
    "H3_TEXT_PAYLOAD_SPANS_KEY",
    "H3_TEXT_PAYLOAD_STATE_KEY",
    "H3_TEXT_PAYLOAD_VERSION",
    "H3_TEXT_PAYLOAD_VERSION_KEY",
    "build_h3_text_payload",
    "read_h3_text_state",
]

#: Bumped whenever the payload gains or loses a field a reader depends on. A reader that finds a
#: version it does not know REFUSES — it never falls back to "read what I recognize", which is how a
#: stale cache gets consumed as a fresh one.
#:
#: v2 -> v3 (#31 finding 2): added ``has_references`` (below) — the regime discriminator was
#: enforced at WRITE time and never persisted, so a v2 cache written under one
#: ``references_per_sample`` regime was indistinguishable BY SHAPE from one written under another.
#: Bumping the version (rather than just adding the key) means every pre-fix v2 cache is refused by
#: the existing version check above, instead of silently reading as "has_references missing".
H3_TEXT_PAYLOAD_VERSION: int = 3
H3_TEXT_PAYLOAD_VERSION_KEY: str = "h3_text_payload_version"

#: The reference-BEARING state and the spans of its vision blocks, half-open, in emission order.
H3_TEXT_PAYLOAD_STATE_KEY: str = "hidden_states"
H3_TEXT_PAYLOAD_SPANS_KEY: str = "vision_spans"

#: The PROMPT-ONLY state — the same caption presented with NO references at all — and its spans,
#: which are empty by construction. Selected on a D-10-REFDROP step so that a dropped step is a
#: genuinely reference-free request rather than "the reference latents are gone but the text still
#: describes them".
H3_TEXT_PAYLOAD_PROMPT_ONLY_KEY: str = "prompt_only_hidden_states"
H3_TEXT_PAYLOAD_PROMPT_ONLY_SPANS_KEY: str = "prompt_only_vision_spans"

#: (#31 finding 2, v3) Whether the SAMPLE (not the per-step dropout draw) was encoded under a
#: reference-bearing regime. Recorded so a reader can refuse a cache whose recorded regime
#: disagrees with the run's OWN ``references_per_sample`` — the mismatch that let a Ref2VA train
#: silently pack reference latents against prompt-only text (or vice versa) after a preprocess
#: re-run at a different slot count over the same output root.
H3_TEXT_PAYLOAD_HAS_REFERENCES_KEY: str = "has_references"


def _normalize_spans(spans: Any, what: str) -> tuple[tuple[int, int], ...]:
    """Coerce to a tuple of half-open ``(start, stop)`` int pairs, refusing anything malformed."""
    normalized: list[tuple[int, int]] = []
    for index, span in enumerate(tuple(spans)):
        pair = tuple(int(v) for v in span)
        if len(pair) != 2:
            raise ValueError(
                f"[h3-text-payload] {what} span {index} is {pair}, expected a half-open "
                f"(start, stop) pair."
            )
        start, stop = pair
        if start < 0 or stop <= start:
            raise ValueError(
                f"[h3-text-payload] {what} span {index} is {pair}: spans are half-open and "
                f"non-empty, so 0 <= start < stop must hold. A degenerate span silently tags no "
                f"rows, which is the defect this payload exists to close."
            )
        normalized.append((start, stop))
    return tuple(normalized)


def build_h3_text_payload(
    *,
    hidden_states: torch.Tensor,
    vision_spans: Any,
    prompt_only_hidden_states: torch.Tensor,
    has_references: bool,
) -> dict[str, Any]:
    """Assemble the version-2 ``h3_conditions/`` payload, refusing the two ways it can lie.

    Args:
        hidden_states: The selected Qwen3-VL hidden state for the REFERENCE-BEARING presentation.
        vision_spans: That presentation's vision-block spans, from ``build_h3_presentation``.
        prompt_only_hidden_states: The hidden state for the SAME caption presented with no
            references — what a D-10-REFDROP step conditions on.
        has_references: Whether the reference-bearing presentation actually carried any. Required
            rather than inferred from ``vision_spans`` being non-empty, because "no spans" is
            precisely the bug being closed and inferring it would make the guard self-fulfilling.

    Returns:
        The payload dict ``write_h3_precomputed`` stores under ``h3_conditions/``.
    """
    spans = _normalize_spans(vision_spans, "reference-bearing")
    if has_references and not spans:
        raise ValueError(
            "[h3-text-payload] the reference-bearing state carries NO vision spans, but the "
            "presentation had references. Empty spans are exactly the defect this payload closes: "
            "train/h3_step.h3_token_tags would then tag every Qwen vision row TEXT, so >90% of the "
            "text stream would modulate with the wrong AdaLN row — silently, at the correct shape."
        )
    if not has_references and spans:
        raise ValueError(
            f"[h3-text-payload] the presentation had no references but {len(spans)} vision span(s) "
            f"were supplied. A reference-free presentation contains no vision blocks at all."
        )
    for name, tensor in (
        (H3_TEXT_PAYLOAD_STATE_KEY, hidden_states),
        (H3_TEXT_PAYLOAD_PROMPT_ONLY_KEY, prompt_only_hidden_states),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"[h3-text-payload] {name} must be a torch.Tensor, got {type(tensor).__name__}."
            )
    return {
        H3_TEXT_PAYLOAD_VERSION_KEY: H3_TEXT_PAYLOAD_VERSION,
        H3_TEXT_PAYLOAD_STATE_KEY: hidden_states,
        H3_TEXT_PAYLOAD_SPANS_KEY: spans,
        H3_TEXT_PAYLOAD_PROMPT_ONLY_KEY: prompt_only_hidden_states,
        # Empty BY CONSTRUCTION and stored anyway, so the reader's contract is symmetric and a
        # future edit that starts putting spans here fails the round-trip check rather than the
        # first metered forward.
        H3_TEXT_PAYLOAD_PROMPT_ONLY_SPANS_KEY: (),
        # (#31 finding 2) THE DISCRIMINATOR, persisted — previously enforced only as an argument
        # at write time and then dropped; see the module docstring's defect (3).
        H3_TEXT_PAYLOAD_HAS_REFERENCES_KEY: bool(has_references),
    }


def read_h3_text_state(
    payload: Any, *, reference_dropped: bool, references_per_sample: int
) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
    """Select the hidden state this step conditions on, WITH its vision spans.

    ⛔ A bare tensor is the **version-1** format and is REFUSED. It is not a compatibility shim
    opportunity: a v1 payload has no spans (so every vision row would be tagged TEXT) and no
    prompt-only state (so a dropped step would keep describing references it no longer has). Both
    failures are silent and both were live. The 88 payloads sitting on the dataset Volume are v1.

    Args:
        payload: What ``PrecomputedDataset`` returned for the ``h3_conditions`` source.
        reference_dropped: The D-10-REFDROP draw for this ``(segment, step)``. ``True`` selects the
            PROMPT-ONLY state, which is what makes a dropped step a real no-reference request.
        references_per_sample: The CURRENT run's ``h3.references_per_sample``. Compared against the
            payload's persisted ``has_references`` (#31 finding 2), ONE-DIRECTIONALLY: a
            ``has_references=False`` cache read at ``references_per_sample >= 1`` is REFUSED — the
            run would resolve and pack real reference-latent rows against a text presentation that
            describes none of them (e.g. a Ref2VA root re-preprocessed at 0 slots and then resumed
            at 2 without a re-encode). The mirror (``has_references=True`` read at
            ``references_per_sample == 0``) is NOT an error — it is how a no-reference train stays
            able to reuse an existing Ref2VA cache (see the module docstring).

    Returns:
        ``(hidden_states, vision_spans)``. ``vision_spans`` is ``()`` on a dropped step.
    """
    if isinstance(payload, torch.Tensor):
        raise ValueError(
            f"[h3-text-payload] the h3_conditions payload is a bare "
            f"{tuple(int(v) for v in payload.shape)} tensor — the STALE version-1 format. It "
            f"carries no vision spans (so every Qwen vision row would be tagged TEXT and modulate "
            f"with the wrong AdaLN row) and no prompt-only state (so a D-10-REFDROP step would drop "
            f"the reference LATENTS while the text kept describing them, a regime that occurs "
            f"nowhere at inference). Both are silent at a perfectly valid shape. This cache must be "
            f"RE-ENCODED — re-dispatch `--mode preprocess`; PHASE A now writes version "
            f"{H3_TEXT_PAYLOAD_VERSION}. The stale tree is NOT deleted, it is refused."
        )
    if not isinstance(payload, dict):
        raise TypeError(
            f"[h3-text-payload] expected the version-{H3_TEXT_PAYLOAD_VERSION} h3_conditions dict, "
            f"got {type(payload).__name__}."
        )
    version = payload.get(H3_TEXT_PAYLOAD_VERSION_KEY)
    if version is None:
        raise ValueError(
            f"[h3-text-payload] the h3_conditions payload declares no "
            f"{H3_TEXT_PAYLOAD_VERSION_KEY!r} (keys: {sorted(payload)}). An undeclared format is a "
            f"pre-versioning cache; re-encode rather than guessing which fields it has."
        )
    if int(version) != H3_TEXT_PAYLOAD_VERSION:
        raise ValueError(
            f"[h3-text-payload] the h3_conditions cache is version {version}; this trainer reads "
            f"version {H3_TEXT_PAYLOAD_VERSION}. Re-encode — a text-conditioning format is not "
            f"forward- or backward-compatible by accident, and reading the fields that happen to "
            f"match is how a stale cache trains a run that reports success."
        )

    # (#31 finding 2) THE REGIME CROSS-CHECK — persisted at write time, checked here against the
    # CURRENT run's slot count. Independent of `reference_dropped` (which only selects a per-STEP
    # presentation): this catches a cache whose SAMPLE was encoded under a different
    # `references_per_sample` regime than the one training now, e.g. a Ref2VA root
    # re-preprocessed at 0 slots and then resumed at 2 without a re-encode.
    #
    # ONE-DIRECTIONAL, deliberately: refuse ``has_references=False`` when ``references_per_sample
    # >= 1`` (the run WILL resolve and pack real reference-latent rows against a text presentation
    # that describes none — the measured failure above). The MIRROR is NOT an error — a
    # ``references_per_sample == 0`` run always forces ``reference_dropped=True`` (see
    # ``H3RefStrategy.prepare_training_inputs``'s ``dropped or no_reference``) and therefore always
    # selects the PROMPT-ONLY key regardless of ``has_references``; reading only the prompt-only
    # half of an otherwise reference-bearing cache is the documented, tested way a NO-REFERENCE
    # train stays able to reuse an existing Ref2VA cache (module docstring, "keeps a REF-BEARING
    # cache consumable by a no-reference train") — no reference-latent row is ever packed at 0
    # slots, so there is no regime for that direction to clash with.
    has_references = payload.get(H3_TEXT_PAYLOAD_HAS_REFERENCES_KEY)
    if not isinstance(has_references, bool):
        raise ValueError(
            f"[h3-text-payload] the h3_conditions payload declares no boolean "
            f"{H3_TEXT_PAYLOAD_HAS_REFERENCES_KEY!r} (keys: {sorted(payload)}), despite declaring "
            f"version {H3_TEXT_PAYLOAD_VERSION}. Every version-{H3_TEXT_PAYLOAD_VERSION} payload "
            f"`build_h3_text_payload` writes carries this field; a version-3 payload without it is "
            f"malformed, not merely stale — re-encode."
        )
    if references_per_sample >= 1 and not has_references:
        raise ValueError(
            f"[h3-text-payload] regime mismatch: this cache was encoded with "
            f"{H3_TEXT_PAYLOAD_HAS_REFERENCES_KEY}=False (a NO-REFERENCE sample — no vision "
            f"blocks describing any reference), but the current run declares "
            f"references_per_sample={references_per_sample}, which resolves and packs "
            f"{references_per_sample} real reference-latent row(s) per sample. Reading through "
            f"this mismatch would pack reference latents against a text presentation that "
            f"describes none of them — no shape error, an ordinary loss curve, and a regime that "
            f"exists nowhere at inference. Re-run `--mode preprocess` against this output root at "
            f"the current references_per_sample before training."
        )

    key = H3_TEXT_PAYLOAD_PROMPT_ONLY_KEY if reference_dropped else H3_TEXT_PAYLOAD_STATE_KEY
    spans_key = (
        H3_TEXT_PAYLOAD_PROMPT_ONLY_SPANS_KEY if reference_dropped else H3_TEXT_PAYLOAD_SPANS_KEY
    )
    hidden = payload.get(key)
    if not isinstance(hidden, torch.Tensor):
        raise ValueError(
            f"[h3-text-payload] the payload has no {key!r} tensor (keys: {sorted(payload)}). "
            f"{'A dropped step needs the PROMPT-ONLY state' if reference_dropped else 'A normal step needs the reference-bearing state'} "
            f"— falling back to the other one would condition on a presentation this step is not "
            f"supposed to be shown."
        )
    spans = _normalize_spans(payload.get(spans_key, ()), key)
    if reference_dropped and spans:
        raise ValueError(
            f"[h3-text-payload] the PROMPT-ONLY state carries {len(spans)} vision span(s). It is a "
            f"reference-free presentation, so it contains no vision blocks and its spans are empty "
            f"by construction — spans here mean the two states were swapped when the cache was "
            f"written, which would tag prompt rows VIDEO and reference rows TEXT."
        )
    return hidden, spans
