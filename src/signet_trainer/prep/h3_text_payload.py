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

The fix for both is the same shape: **PHASE A already knows these things, so PHASE A writes them
down.** The payload carries the reference-bearing hidden state WITH its spans, and a second
**prompt-only** hidden state whose spans are empty BY CONSTRUCTION. ``H3RefStrategy`` selects
between them by the dropout draw.

Versioning, and why a stale cache must be REFUSED
-------------------------------------------------
Version 1 was a bare ``(L, 5120)`` tensor. It is indistinguishable-by-duck-typing from the tensor
inside version 2, so a reader that merely "accepts a tensor or a dict" would consume a v1 cache
silently — with no spans and no prompt-only state, i.e. exactly the two defects above, restored.
There are **88 v1 payloads on the dataset Volume right now** (the partial cache the D-10-DEF-9 run
committed before failing). They are NOT deleted — never auto-delete an intermediate — so the reader
is what has to refuse them, by shape, with the re-encode named.

⚠ **This costs a PHASE A re-encode**: two Qwen3-VL passes per sample instead of one. That is
expected and was decided deliberately — defect (1) needs the re-encode regardless, so (2) rides
along in the same pass over the corpus rather than buying a second one.

Import tier: ``torch`` + stdlib only. Both the WRITER (``prep/h3_encode`` via ``modal/fns``) and the
READER (``conditioning/h3_ref``) import this module, which is the whole point — one format, one
place, no second opinion about what a cached text condition is.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
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
H3_TEXT_PAYLOAD_VERSION: int = 2
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
    }


def read_h3_text_state(
    payload: Any, *, reference_dropped: bool
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
