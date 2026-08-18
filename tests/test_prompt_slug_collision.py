"""#22 finding 5 (cheap slice) — validation.prompts slug-collision refusal at config load.

inference/grid.py's ``slug()`` is the SOLE per-row clip filename key for validation.prompts
renders (``f"{slug(prompt)}_s{seed}.mp4"`` in modal/fns.py's sample() / h3_sample()) and it
truncates every prompt to its first ``_SLUG_MAX_LEN`` (60) alnum-ish characters. Two prompts
sharing that prefix collide on the SAME filename:

  * before resume this was a silent overwrite;
  * under resume (h3_sample's ``_render``, fns.py) it is a silently SKIPPED render — the second
    prompt never gets its own file because ``_render`` finds the first prompt's file already
    non-empty and prints "resume — already rendered", so the colliding row shows the FIRST
    prompt's pixels under the SECOND prompt's banner with zero GPU work to reveal it.

The house eval convention (a couple of in-distribution prompts plus ranging probes built on the
same anchor words) actively produces near-duplicate 60-char prefixes, so this is refused for
free at config load (CPU, zero spend) rather than discovered inside a metered container — the
same "cheapest first" doctrine as the qwen_edit caption_dropout refusal already on `main`.

CPU only — no ltx_core / modal import, no filesystem touch (ValidationConfig / SignetConfig
validators never touch disk).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from signet_trainer.config.schema import ValidationConfig
from signet_trainer.inference.grid import slug

# --------------------------------------------------------------------------------------------------
# Deterministic collision fixtures — built FROM the real `slug()` so the test can never drift out
# of sync with _SLUG_MAX_LEN or the alnum-collapse rule.
# --------------------------------------------------------------------------------------------------

# All-alnum, so slug() performs no character collapsing — only the length truncation applies.
# Both share the exact same first-60-char prefix; the suffixes differ, so they are NOT the same
# prompt text (a real distinct-prompts scenario, not a trivial exact-duplicate check).
_PROMPT_A = "x" * 60 + "_variant_one_wide_shot"
_PROMPT_B = "x" * 60 + "_variant_two_close_up"
assert slug(_PROMPT_A) == slug(_PROMPT_B) == "x" * 60, (
    "test fixture assumption broken: _PROMPT_A/_PROMPT_B must collide under slug() — if this "
    "fails, _SLUG_MAX_LEN or slug()'s collapse rule changed and the fixture needs updating."
)


def test_distinct_prompts_with_different_slugs_load_fine() -> None:
    v = ValidationConfig(prompts=["a wide establishing shot", "a tight close-up portrait"])
    assert list(v.prompts) == ["a wide establishing shot", "a tight close-up portrait"]


def test_empty_prompts_list_loads_fine() -> None:
    v = ValidationConfig(prompts=[])
    assert v.prompts == []


def test_single_prompt_never_collides_with_itself() -> None:
    v = ValidationConfig(prompts=[_PROMPT_A])
    assert v.prompts == [_PROMPT_A]


def test_slug_colliding_prompt_pair_is_rejected() -> None:
    with pytest.raises(ValidationError, match="slug collision"):
        ValidationConfig(prompts=[_PROMPT_A, _PROMPT_B])


def test_slug_collision_error_names_the_colliding_prompts() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ValidationConfig(prompts=[_PROMPT_A, _PROMPT_B])
    message = str(excinfo.value)
    assert _PROMPT_A in message
    assert _PROMPT_B in message
    assert "#22 finding 5" in message


def test_slug_collision_error_cites_resume_skip_mechanism() -> None:
    # Pin the mechanism, not just the existence of a collision — a message that only said
    # "collision" without explaining WHY it matters (silently skipped render under resume) would
    # be too weak a guard against a future edit trimming the wrong sentence.
    with pytest.raises(ValidationError, match="silently SKIPPED"):
        ValidationConfig(prompts=[_PROMPT_A, _PROMPT_B])


def test_three_prompts_two_colliding_one_distinct_still_rejected() -> None:
    with pytest.raises(ValidationError, match="slug collision"):
        ValidationConfig(prompts=[_PROMPT_A, _PROMPT_B, "a totally different establishing shot"])


def test_three_prompts_all_distinct_loads_fine() -> None:
    v = ValidationConfig(
        prompts=[
            "a wide establishing shot of a lighthouse at dawn",
            "a tight close-up portrait under warm tungsten light",
            "a tracking shot following a cyclist through the city",
        ]
    )
    assert len(v.prompts) == 3


def test_in_loop_sampling_still_works_with_a_single_valid_prompt() -> None:
    # Regression guard: the new collision check must not interfere with the pre-existing
    # in_loop_sampling + prompts validator (D-9-OFFL02-CLOSE) on the ordinary non-colliding path.
    v = ValidationConfig(in_loop_sampling=True, prompts=["a photoreal driving clip"])
    assert v.in_loop_sampling is True
