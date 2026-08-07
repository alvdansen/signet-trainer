"""The versioned ``h3_conditions`` payload — and the two silent defects its shape exists to close.

Both were live, both were silent, and neither had a shape error:

  1. **the vision spans were never persisted.** ``h3_ref`` read ``batch.get("vision_spans", ())``
     and nothing populated it, so ``h3_token_tags`` tagged every Qwen vision row TEXT — at 896 short
     edge, >90% of the text stream modulating with the wrong AdaLN row on every step;
  2. **a D-10-REFDROP step was not reference-free.** The reference LATENT rows were dropped while
     the cached text state kept describing both references, a regime that occurs nowhere at
     inference.

The fix is a payload shape, so the tests are about the shape: it round-trips through ``torch.save``,
it refuses the stale format by inspection rather than by duck-typing, and it cannot be assembled in
either of the two ways that would restore a defect.

CPU-only, no GPU, no Modal spend.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from signet_trainer.prep.h3_text_payload import (  # noqa: E402
    H3_TEXT_PAYLOAD_PROMPT_ONLY_KEY,
    H3_TEXT_PAYLOAD_PROMPT_ONLY_SPANS_KEY,
    H3_TEXT_PAYLOAD_SPANS_KEY,
    H3_TEXT_PAYLOAD_STATE_KEY,
    H3_TEXT_PAYLOAD_VERSION,
    H3_TEXT_PAYLOAD_VERSION_KEY,
    build_h3_text_payload,
    read_h3_text_state,
)

_SPANS = ((1, 4), (5, 9))


def _payload(**overrides: object) -> dict:
    kwargs: dict = {
        # DIFFERENT lengths on purpose: a reference-free presentation has no label blocks and no
        # vision blocks, so equal lengths would let a swap of the two states pass every shape check.
        "hidden_states": torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4),
        "vision_spans": _SPANS,
        "prompt_only_hidden_states": torch.zeros(5, 4),
        "has_references": True,
    }
    kwargs.update(overrides)
    return build_h3_text_payload(**kwargs)  # type: ignore[arg-type]


# ==================================================================================================
# The round trip
# ==================================================================================================


def test_the_reference_bearing_state_comes_back_with_its_spans() -> None:
    hidden, spans = read_h3_text_state(_payload(), reference_dropped=False)
    assert spans == _SPANS, "the spans PHASE A measured must survive verbatim — they ARE the fix"
    assert hidden.shape == (12, 4)


def test_a_dropped_step_gets_the_prompt_only_state_and_NO_spans() -> None:
    hidden, spans = read_h3_text_state(_payload(), reference_dropped=True)
    assert hidden.shape == (5, 4), (
        "a dropped step must condition on the PROMPT-ONLY state; reading the reference-bearing one "
        "is the defect — the latents are gone but the text still describes them"
    )
    assert spans == (), "a reference-free presentation contains no vision blocks at all"


def test_the_payload_survives_torch_save_and_load(tmp_path) -> None:
    """``PrecomputedDataset`` loads with ``weights_only=True``; a tuple-of-tuples must survive it."""
    path = tmp_path / "sample.pt"
    torch.save(_payload(), path)
    loaded = torch.load(path, weights_only=True)
    hidden, spans = read_h3_text_state(loaded, reference_dropped=False)
    assert spans == _SPANS
    assert torch.equal(hidden, _payload()[H3_TEXT_PAYLOAD_STATE_KEY])


def test_the_payload_declares_its_version_and_both_states() -> None:
    payload = _payload()
    assert payload[H3_TEXT_PAYLOAD_VERSION_KEY] == H3_TEXT_PAYLOAD_VERSION
    for key in (
        H3_TEXT_PAYLOAD_STATE_KEY,
        H3_TEXT_PAYLOAD_SPANS_KEY,
        H3_TEXT_PAYLOAD_PROMPT_ONLY_KEY,
        H3_TEXT_PAYLOAD_PROMPT_ONLY_SPANS_KEY,
    ):
        assert key in payload, f"the payload must be SELF-DESCRIBING; {key} is missing"
    assert payload[H3_TEXT_PAYLOAD_PROMPT_ONLY_SPANS_KEY] == (), (
        "the prompt-only spans are empty BY CONSTRUCTION, and stored anyway so the reader's "
        "contract is symmetric"
    )


# ==================================================================================================
# The refusals — each one is a defect that would otherwise be silent
# ==================================================================================================


def test_the_STALE_v1_bare_tensor_is_refused_and_names_the_re_encode() -> None:
    """The 88 payloads on the dataset Volume are exactly this. They are refused, not deleted."""
    with pytest.raises(ValueError, match="version-1"):
        read_h3_text_state(torch.zeros(12, 4), reference_dropped=False)
    with pytest.raises(ValueError, match="RE-ENCODED"):
        read_h3_text_state(torch.zeros(12, 4), reference_dropped=False)


def test_an_undeclared_or_unknown_version_is_refused() -> None:
    with pytest.raises(ValueError, match="declares no"):
        read_h3_text_state({H3_TEXT_PAYLOAD_STATE_KEY: torch.zeros(3, 4)}, reference_dropped=False)
    payload = _payload()
    payload[H3_TEXT_PAYLOAD_VERSION_KEY] = H3_TEXT_PAYLOAD_VERSION + 1
    with pytest.raises(ValueError, match="version"):
        read_h3_text_state(payload, reference_dropped=False)


def test_a_reference_bearing_state_with_NO_spans_is_refused_at_WRITE_time() -> None:
    """Empty spans ARE defect (1). Refusing at write time is cheaper than at read time.

    ``has_references`` is a separate argument rather than inferred from ``vision_spans`` being
    non-empty, precisely so this guard cannot become self-fulfilling.
    """
    with pytest.raises(ValueError, match="NO vision spans"):
        _payload(vision_spans=())


def test_a_reference_FREE_presentation_carrying_spans_is_refused() -> None:
    with pytest.raises(ValueError, match="no references"):
        _payload(has_references=False)


def test_a_prompt_only_state_carrying_spans_is_refused_at_READ_time() -> None:
    """Spans on the prompt-only state mean the two states were SWAPPED when the cache was written.

    That would tag prompt rows VIDEO and reference rows TEXT — the defect, inverted, and just as
    silent.
    """
    payload = _payload()
    payload[H3_TEXT_PAYLOAD_PROMPT_ONLY_SPANS_KEY] = ((0, 2),)
    with pytest.raises(ValueError, match="PROMPT-ONLY state carries"):
        read_h3_text_state(payload, reference_dropped=True)


def test_a_missing_state_is_refused_rather_than_falling_back_to_the_other_one() -> None:
    payload = _payload()
    del payload[H3_TEXT_PAYLOAD_PROMPT_ONLY_KEY]
    with pytest.raises(ValueError, match="prompt_only_hidden_states"):
        read_h3_text_state(payload, reference_dropped=True)


def test_a_degenerate_span_is_refused() -> None:
    """A ``(4, 4)`` span tags NO rows — present in the payload, absent in effect."""
    with pytest.raises(ValueError, match="half-open"):
        _payload(vision_spans=((4, 4),))
    with pytest.raises(ValueError, match="half-open"):
        _payload(vision_spans=((5, 2),))
