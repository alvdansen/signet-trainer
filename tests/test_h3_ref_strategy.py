"""CPU contract tests for H3 reference selection + ``H3RefStrategy`` (H3-05/H3-06, Plan 10-08).

Every assertion here is a contract on ``src/signet_trainer/conditioning/h3_ref.py`` — the ref2v
conditioning layer. Six things are load-bearing and each one is a silent failure if it drifts:

  1. **EXACTLY 2 reference slots, always.** The environment reference SUBSTITUTES for the second
     character slot; it is NEVER appended. A 3-slot sample was never priced by the packed-sequence
     VRAM budget (10-03), so it OOMs a metered A100 rather than failing locally.
  2. **The 2-of-3 rotation is balanced.** Feeding all three character refs every time makes
     conditioning CONSTANT across the corpus, so the model cannot learn what varies and is free to
     copy the reference wholesale — the ``[precedent]`` copy-collapse failure this house already
     paid for once on LTX.
  3. **Pair assignment is deterministic across PROCESSES** (D-10-PAIRSEED). Python's built-in
     ``hash()`` is salted per process, so a ``hash()``-based assignment reproduces perfectly in one
     interpreter and silently differs inside a Modal container. The subprocess test is what catches
     it; the source scan is the belt to that brace.
  4. **The environment reference is ALWAYS the final slot** (D-10-REFORDER). H3 imposes no slot
     semantics, but order is load-bearing twice over — it fixes the ``<Picture i>`` labels and it
     advances the shared rotary clock — so a different order is a different request, and the house
     order must be applied consistently between training and inference.
  5. **The loss drops the reference prefix and nothing else.** Loss over conditioning rows is the
     dead-adapter bug. The garbage-prefix-vs-zero-prefix equality assertion is the CPU proof that it
     cannot reach a metered run.
  6. **Target audio rows stay present and out of the loss** (D-10-AUDIO). Not targeting audio is not
     the same as training on silence.

All tensors are CPU float32 and tiny. Nothing here imports ``diffusers``, ``modal`` or touches CUDA.
Reference file names are SYNTHETIC — client-property hygiene keeps real reference filenames out of
tracked files (T-10-08-I).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from signet_trainer.conditioning.h3_ref import (
    REFERENCE_PAIRS,
    H3ModelInputs,
    H3Reference,
    H3RefStrategy,
    assign_character_ref,
    assign_reference_pair,
    order_reference_slots,
    reference_dropout_for,
    resolve_reference_slots,
)
from signet_trainer.conditioning.strategy import ModelInputs, TrainingStrategy
from signet_trainer.prep.h3_text_payload import build_h3_text_payload
from signet_trainer.train.h3_step import (
    H3_PACKED_BATCH_KEYS,
    H3_TEXT_TAG,
    H3_VIDEO_TAG,
    h3_step_deps_from_model,
)

_H3_REF_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "conditioning" / "h3_ref.py"
)

#: The corpus size the rotation is balanced over: 44 clips -> 88 segments (10-CONTEXT).
N_SEGMENTS = 88

#: ~12 of the 88 segments carry an environment reference (D-10-ASYM). The COUNT is still 2.
N_ENVIRONMENT_SEGMENTS = 12


# --------------------------------------------------------------------------------------------------
# helpers — synthetic references only (no real filenames in tracked files)
# --------------------------------------------------------------------------------------------------


def _character(subject: str, *, index: int = 0, width: int = 1024, height: int = 1536) -> H3Reference:
    return H3Reference(
        path=f"refs/char_{subject.lower()}_{index}.png",
        kind="character",
        subject_id=subject,
        width=width,
        height=height,
    )


def _environment(name: str = "env0", *, width: int = 1344, height: int = 768) -> H3Reference:
    return H3Reference(
        path=f"refs/{name}.png",
        kind="environment",
        subject_id=name,
        width=width,
        height=height,
    )


def _prop(subject: str, *, width: int = 768, height: int = 768) -> H3Reference:
    return H3Reference(
        path=f"refs/prop_{subject.lower()}.png",
        kind="prop",
        subject_id=subject,
        width=width,
        height=height,
    )


CHARACTERS = [_character("A"), _character("B"), _character("C")]


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments and triple-quoted blocks (tests/test_no_warm_gpu.py discipline)."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _run_in_subprocess(script: str) -> str:
    """Run ``script`` in a FRESH interpreter with ``PYTHONPATH=src`` and return its stdout."""
    repo_root = Path(__file__).resolve().parents[1]
    merged = dict(os.environ)
    merged.update({"PYTHONPATH": str(repo_root / "src"), "PYTHONIOENCODING": "utf-8"})
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=merged,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


# --------------------------------------------------------------------------------------------------
# 1. The 2-of-3 rotation
# --------------------------------------------------------------------------------------------------


def test_reference_pairs_is_the_three_two_of_three_combinations() -> None:
    assert REFERENCE_PAIRS == ((0, 1), (0, 2), (1, 2))


def test_assign_reference_pair_returns_two_distinct_indices() -> None:
    for segment in range(N_SEGMENTS):
        pair = assign_reference_pair(segment)
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert pair[0] != pair[1]
        assert all(0 <= i < 3 for i in pair)


def test_rotation_is_balanced_at_twenty_nine_or_thirty_over_eighty_eight_segments() -> None:
    """The plan's headline number: 29/29/30 over the 88 segments, and NO fourth pair."""
    counts = Counter(assign_reference_pair(i) for i in range(N_SEGMENTS))
    assert set(counts) == {(0, 1), (0, 2), (1, 2)}
    assert all(29 <= v <= 30 for v in counts.values()), dict(counts)
    assert sum(counts.values()) == N_SEGMENTS


def test_every_returned_pair_is_one_of_the_three_canonical_pairs() -> None:
    assert {assign_reference_pair(i) for i in range(500)} <= set(REFERENCE_PAIRS)


def test_assign_character_ref_is_a_single_balanced_index() -> None:
    """The environment-segment path: ONE rotating character ref, balanced within +/- 2 of 29."""
    counts = Counter(assign_character_ref(i) for i in range(N_SEGMENTS))
    assert set(counts) == {0, 1, 2}
    assert all(27 <= v <= 31 for v in counts.values()), dict(counts)
    assert sum(counts.values()) == N_SEGMENTS


def test_assign_reference_pair_honors_a_smaller_reference_pool() -> None:
    """A 2-character pool has exactly one pair; the helper must not assume three."""
    assert {assign_reference_pair(i, n_refs=2) for i in range(20)} == {(0, 1)}


def test_assign_reference_pair_refuses_a_pool_too_small_to_pair() -> None:
    with pytest.raises(ValueError, match="n_refs"):
        assign_reference_pair(0, n_refs=1)


# --------------------------------------------------------------------------------------------------
# 2. Determinism (D-10-PAIRSEED) — the salted-hash trap
# --------------------------------------------------------------------------------------------------


def test_assignment_is_deterministic_within_a_process() -> None:
    assert [assign_reference_pair(i) for i in range(50)] == [
        assign_reference_pair(i) for i in range(50)
    ]
    assert [assign_character_ref(i) for i in range(50)] == [
        assign_character_ref(i) for i in range(50)
    ]


def test_assignment_is_deterministic_ACROSS_PROCESSES() -> None:
    """The assertion that catches a salted ``hash()``: a fresh interpreter must agree.

    ``PYTHONHASHSEED`` is randomized per process by default, so a ``hash()``-based assignment
    reproduces perfectly inside one test session and silently differs inside a Modal container.
    """
    script = (
        "from signet_trainer.conditioning.h3_ref import assign_reference_pair\n"
        "print(assign_reference_pair(7, seed=42))\n"
    )
    first = _run_in_subprocess(script)
    second = _run_in_subprocess(script)
    assert first == second, "two fresh interpreters disagreed — the assignment is not stable"
    assert first == str(assign_reference_pair(7, seed=42))


def test_cross_process_determinism_holds_for_a_whole_schedule() -> None:
    script = (
        "from signet_trainer.conditioning.h3_ref import assign_reference_pair, assign_character_ref\n"
        "print([assign_reference_pair(i, seed=42) for i in range(88)])\n"
        "print([assign_character_ref(i, seed=42) for i in range(88)])\n"
    )
    lines = _run_in_subprocess(script).splitlines()
    assert lines[0] == str([assign_reference_pair(i, seed=42) for i in range(88)])
    assert lines[1] == str([assign_character_ref(i, seed=42) for i in range(88)])


def test_a_different_seed_changes_the_assignment_somewhere() -> None:
    baseline = [assign_reference_pair(i, seed=42) for i in range(N_SEGMENTS)]
    other = [assign_reference_pair(i, seed=1234) for i in range(N_SEGMENTS)]
    assert baseline != other


def test_a_different_seed_still_produces_a_balanced_rotation() -> None:
    """Seeding may not buy determinism at the cost of the balance the rotation exists for."""
    for seed in (0, 1, 7, 1234, 20260806):
        counts = Counter(assign_reference_pair(i, seed=seed) for i in range(N_SEGMENTS))
        assert set(counts) == set(REFERENCE_PAIRS), (seed, dict(counts))
        assert all(29 <= v <= 30 for v in counts.values()), (seed, dict(counts))


def test_source_never_uses_the_salted_builtin_hash() -> None:
    """Belt to the subprocess brace — ``hash(`` must not appear in the module at all."""
    code = _strip_comments_and_docstrings(_H3_REF_SRC.read_text(encoding="utf-8"))
    assert not re.search(r"(?<![\w.])hash\s*\(", code), (
        "Python's built-in hash() is salted per process (PYTHONHASHSEED) and would break "
        "D-10-PAIRSEED reproducibility across containers."
    )


def test_the_module_banner_states_the_live_union_not_the_superseded_fixed_count() -> None:
    """#31 finding 3 — the module the repo designates as the normative decision record must not
    lead with a headline asserting the slot count is fixed at 2, inside the same file that
    implements the reference-count union (h3_ref.py:789/904). A maintainer who reads a stale ⛔
    headline and then writes or reviews on that premise is the failure mode this closes.

    PR #51 (explicit-manifest sequence tasks, merged after this bundle was authored) widened the
    live union from {0, 1, 2} to {0, 1, 2, 3} — the banner correctly states the WIDER union now;
    this test pins the union not being stale, not any one specific literal set."""
    source = _H3_REF_SRC.read_text(encoding="utf-8")
    assert "EXACTLY 2 SLOTS, ALWAYS" not in source, (
        "the banner still asserts a fixed slot count the live union superseded"
    )
    assert "{0, 1, 2, 3}" in source, "the banner must state the live union (widened by PR #51)"
    assert "Fixed slot count (2)" not in source, (
        "resolve_reference_slots' docstring opener is still the pre-union phrase"
    )


# --------------------------------------------------------------------------------------------------
# 3. resolve_reference_slots — EXACTLY 2, environment SUBSTITUTES
# --------------------------------------------------------------------------------------------------


def test_resolve_returns_exactly_two_slots_without_an_environment_reference() -> None:
    for segment in range(N_SEGMENTS):
        slots = resolve_reference_slots(segment, CHARACTERS, None)
        assert len(slots) == 2, (segment, slots)
        assert all(s.kind == "character" for s in slots)
        assert slots[0].subject_id != slots[1].subject_id


def test_resolve_returns_exactly_two_slots_WITH_an_environment_reference() -> None:
    environment = _environment()
    for segment in range(N_SEGMENTS):
        slots = resolve_reference_slots(segment, CHARACTERS, environment)
        assert len(slots) == 2, (segment, slots)
        assert slots[-1] is environment
        assert slots[0].kind == "character"


def test_the_environment_reference_SUBSTITUTES_it_never_appends() -> None:
    """The operator ruling, as an assertion: three references never happen."""
    environment = _environment()
    for segment in range(N_SEGMENTS):
        with_env = resolve_reference_slots(segment, CHARACTERS, environment)
        without_env = resolve_reference_slots(segment, CHARACTERS, None)
        assert len(with_env) == len(without_env) == 2
        # exactly ONE character survives on the environment path
        assert sum(1 for s in with_env if s.kind == "character") == 1


def test_resolve_uses_the_single_character_rotation_on_the_environment_path() -> None:
    environment = _environment()
    picked = [
        resolve_reference_slots(i, CHARACTERS, environment)[0].subject_id
        for i in range(N_SEGMENTS)
    ]
    counts = Counter(picked)
    assert set(counts) == {"A", "B", "C"}
    assert all(27 <= v <= 31 for v in counts.values()), dict(counts)


def test_resolve_is_deterministic_for_a_fixed_segment_and_seed() -> None:
    environment = _environment()
    a = resolve_reference_slots(13, CHARACTERS, environment, seed=42)
    b = resolve_reference_slots(13, CHARACTERS, environment, seed=42)
    assert [s.path for s in a] == [s.path for s in b]


def test_resolve_refuses_a_slot_count_it_cannot_fill() -> None:
    with pytest.raises(ValueError):
        resolve_reference_slots(0, [_character("A")], None)


def test_resolve_asserts_the_returned_length_naming_the_substitution_rule() -> None:
    """A 3-slot return is the blocker this function exists to prevent — it must be named."""
    source = _H3_REF_SRC.read_text(encoding="utf-8")
    assert "SUBSTITUT" in source.upper()
    assert "never appended" in source or "never appends" in source


@pytest.mark.parametrize("n_environment_segments", [N_ENVIRONMENT_SEGMENTS])
def test_the_asymmetric_regime_varies_while_the_count_does_not(n_environment_segments: int) -> None:
    """D-10-ASYM: the REGIME varies across the corpus; the COUNT never does."""
    environment = _environment()
    regimes = set()
    for segment in range(N_SEGMENTS):
        env = environment if segment < n_environment_segments else None
        slots = resolve_reference_slots(segment, CHARACTERS, env)
        assert len(slots) == 2
        regimes.add(tuple(s.kind for s in slots))
    assert regimes == {("character", "character"), ("character", "environment")}


# --------------------------------------------------------------------------------------------------
# 4. order_reference_slots — D-10-REFORDER
# --------------------------------------------------------------------------------------------------


def test_order_puts_the_environment_reference_last() -> None:
    environment = _environment()
    ordered = order_reference_slots([environment, _character("A")])
    assert ordered[-1] is environment
    assert ordered[0].subject_id == "A"


def test_order_returns_just_the_characters_when_no_environment_is_present() -> None:
    refs = [_character("A"), _character("B")]
    assert order_reference_slots(refs) == refs


def test_order_preserves_the_given_scene_composition_order() -> None:
    """The given order IS the house order (rightmost subject first); ordering must be STABLE."""
    refs = [_character("C"), _character("A"), _character("B")]
    assert [r.subject_id for r in order_reference_slots(refs)] == ["C", "A", "B"]


def test_order_groups_multiple_images_of_one_subject_together() -> None:
    refs = [
        _character("A", index=0),
        _character("B", index=0),
        _character("A", index=1),
    ]
    ordered = order_reference_slots(refs)
    assert [r.subject_id for r in ordered] == ["A", "A", "B"]
    # grouping must not reorder WITHIN a subject
    assert [r.path for r in ordered[:2]] == [refs[0].path, refs[2].path]


def test_order_keeps_props_intermixed_with_characters_before_the_environment() -> None:
    environment = _environment()
    refs = [_character("A"), environment, _prop("sword"), _character("B")]
    ordered = order_reference_slots(refs)
    assert [r.kind for r in ordered] == ["character", "prop", "character", "environment"]


def test_order_is_idempotent() -> None:
    refs = [_character("A"), _environment(), _character("B")]
    once = order_reference_slots(refs)
    assert order_reference_slots(once) == once


def test_multiple_environment_references_all_land_at_the_end_in_order() -> None:
    e1, e2 = _environment("env1"), _environment("env2")
    ordered = order_reference_slots([e1, _character("A"), e2])
    assert [r.subject_id for r in ordered] == ["A", "env1", "env2"]


# --------------------------------------------------------------------------------------------------
# 5. reference_dropout_for — D-10-REFDROP
# --------------------------------------------------------------------------------------------------


def test_dropout_hits_about_twenty_percent_over_two_thousand_draws() -> None:
    draws = [reference_dropout_for(i % N_SEGMENTS, i) for i in range(2000)]
    rate = sum(draws) / len(draws)
    assert 0.17 <= rate <= 0.23, rate


def test_dropout_is_deterministic_for_a_fixed_input() -> None:
    for segment, step in ((0, 0), (7, 13), (87, 999)):
        assert reference_dropout_for(segment, step) == reference_dropout_for(segment, step)


def test_dropout_is_deterministic_across_processes() -> None:
    script = (
        "from signet_trainer.conditioning.h3_ref import reference_dropout_for\n"
        "print([reference_dropout_for(i, i * 3) for i in range(40)])\n"
    )
    assert _run_in_subprocess(script) == str(
        [reference_dropout_for(i, i * 3) for i in range(40)]
    )


def test_dropout_zero_never_fires() -> None:
    assert not any(
        reference_dropout_for(i % N_SEGMENTS, i, dropout=0.0) for i in range(2000)
    )


def test_dropout_one_always_fires() -> None:
    assert all(reference_dropout_for(i % N_SEGMENTS, i, dropout=1.0) for i in range(500))


def test_dropout_varies_with_the_step_not_just_the_segment() -> None:
    """Dropout is a per-(segment, step) draw — a per-segment-only draw would freeze the regime."""
    per_step = [reference_dropout_for(3, step) for step in range(200)]
    assert len(set(per_step)) == 2, "segment 3 never changed its dropout decision across steps"


def test_a_different_dropout_seed_changes_the_draw_somewhere() -> None:
    a = [reference_dropout_for(i, i) for i in range(200)]
    b = [reference_dropout_for(i, i, seed=1234) for i in range(200)]
    assert a != b


def test_dropout_rejects_a_probability_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError):
        reference_dropout_for(0, 0, dropout=1.5)
    with pytest.raises(ValueError):
        reference_dropout_for(0, 0, dropout=-0.1)


# --------------------------------------------------------------------------------------------------
# 6. Import confinement (Anti-Pattern 6) + the decision record
# --------------------------------------------------------------------------------------------------


def test_module_scope_imports_no_modal_and_no_diffusers() -> None:
    raw = _H3_REF_SRC.read_text(encoding="utf-8")
    assert not re.search(r"^(import|from)\s+(modal|diffusers)\b", raw, re.MULTILINE)


def test_importing_h3_ref_pulls_neither_diffusers_nor_modal() -> None:
    script = (
        "import sys\n"
        "import signet_trainer.conditioning.h3_ref as m\n"
        "assert 'diffusers' not in sys.modules, 'diffusers leaked'\n"
        "assert 'modal' not in sys.modules, 'modal leaked'\n"
        "print(m.REFERENCE_PAIRS)\n"
    )
    assert _run_in_subprocess(script) == str(REFERENCE_PAIRS)


def test_the_module_does_not_import_the_config_schema() -> None:
    """Injected-seam discipline: the Modal-side caller constructs the strategy FROM config."""
    raw = _H3_REF_SRC.read_text(encoding="utf-8")
    assert not re.search(r"^(import|from)\s+signet_trainer\.config\b", raw, re.MULTILINE)


def test_the_locked_decisions_are_recorded_in_the_source() -> None:
    """The rationale must survive in code — the next reader cannot re-derive an operator ruling."""
    source = _H3_REF_SRC.read_text(encoding="utf-8")
    for decision in ("D-10-REFORDER", "D-10-REFDROP", "D-10-PAIRSEED", "D-10-ASYM", "D-10-NOMINE"):
        assert decision in source, f"{decision} is not recorded in h3_ref.py"


def test_no_environment_reference_mining_was_added() -> None:
    """D-10-NOMINE: the four mapped illustrations are it — no discovery/globbing of a ref dir."""
    code = _strip_comments_and_docstrings(_H3_REF_SRC.read_text(encoding="utf-8"))
    for banned in ("glob", "iterdir", "listdir", "rglob"):
        assert banned not in code, f"{banned} suggests environment-reference mining (D-10-NOMINE)"


# ==================================================================================================
# Task 2 — H3RefStrategy: packed inputs and the reference-prefix-excluded loss
# ==================================================================================================

# Deliberately tiny feature widths — every contract below is about ROWS, not channels.
PATCH_DIM = 6
AUDIO_IN_CHANNELS = 4
TEXT_DIM = 5

REF_ROWS = 4
N_TEXT = 12
N_TARGET_VIDEO = 9
N_TARGET_AUDIO = 3

T_VIDEO = 0.5
T_AUDIO = 0.4

#: The prompt-only state is a DIFFERENT length from the reference-bearing one — a reference-free
#: presentation has no label blocks and no vision blocks — so the fixture makes them differ. Equal
#: lengths would let a swap of the two states pass every shape assertion.
N_PROMPT_ONLY_TEXT = 5

#: Two vision blocks inside the text stream, half-open, the shape PHASE A writes. Non-empty on
#: purpose: `()` is the defect (>90% of the text stream tagged TEXT instead of VIDEO), so a fixture
#: that omitted them would be testing the bug.
VISION_SPANS: tuple[tuple[int, int], ...] = ((1, 5), (6, 10))

#: Sentinel for "the caller did not override the audio payload" — `None` is a meaningful value here
#: (it is what an absent source looks like), so it cannot double as the default.
_UNSET = object()


def _deps() -> object:
    """A stub ``H3StepDeps`` — the injected backend seam, CPU-only (``train/step.py`` precedent)."""

    class _Cfg:
        in_channels = PATCH_DIM
        patch_size = (1, 1, 1)
        audio_in_channels = AUDIO_IN_CHANNELS
        text_dim = TEXT_DIM

    class _Model:
        config = _Cfg()

    return h3_step_deps_from_model(_Model())


def _monotone_position_ids(*, seq_len: int, **_: object) -> torch.Tensor:
    """A monotone stand-in for H3's 3-axis float64 RoPE — sufficient for CPU shape contracts."""
    ids = torch.zeros(seq_len, 3, dtype=torch.float64)
    ids[:, 0] = torch.arange(seq_len, dtype=torch.float64)
    return ids


def _reference_entry(reference: H3Reference, *, rows: int = REF_ROWS) -> dict:
    """One entry of the reference pool: the descriptor fields plus its ROW-MAJOR latents."""
    fill = float(ord(reference.subject_id[0]))
    return {
        "path": reference.path,
        "kind": reference.kind,
        "subject_id": reference.subject_id,
        "width": reference.width,
        "height": reference.height,
        "latents": torch.full((1, rows, PATCH_DIM), fill),
    }


def _batch(
    *,
    segment_index: int = 0,
    step: int = 0,
    with_environment: bool = False,
    environments: list[H3Reference] | None = None,
    characters: list[H3Reference] | None = None,
    video_noise: torch.Tensor | None = None,
    n_target_video: int = N_TARGET_VIDEO,
    vision_spans: object = None,
    audio: object = _UNSET,
) -> dict:
    chars = list(CHARACTERS if characters is None else characters)
    if environments is None:
        envs = [_environment()] if with_environment else []
    else:
        envs = list(environments)
    pool = [_reference_entry(c) for c in chars] + [_reference_entry(e) for e in envs]

    generator = torch.Generator().manual_seed(7)
    batch: dict = {
        "segment_index": segment_index,
        "step": step,
        "h3_latent_conditions": {
            "latents": torch.randn(1, n_target_video, PATCH_DIM, generator=generator)
        },
        # ⛔ The versioned payload, not a bare tensor. A bare tensor is the STALE v1 format the
        # reader now refuses: it carries no vision spans (so every Qwen vision row would be tagged
        # TEXT) and no prompt-only state (so a dropped step would keep describing references it no
        # longer has). Both were live and both are silent, so the fixture goes through the real
        # writer-side builder rather than restating its shape.
        "h3_text_conditions": build_h3_text_payload(
            hidden_states=torch.randn(1, N_TEXT, TEXT_DIM, generator=generator),
            vision_spans=vision_spans if vision_spans is not None else VISION_SPANS,
            prompt_only_hidden_states=torch.randn(
                1, N_PROMPT_ONLY_TEXT, TEXT_DIM, generator=generator
            ),
            has_references=True,
        ),
        "h3_ref_latent_conditions": {"references": pool},
        "h3_audio_latent_conditions": (
            {"latents": torch.randn(1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS, generator=generator)}
            if audio is _UNSET
            else audio
        ),
        "t_video": T_VIDEO,
        "t_audio": T_AUDIO,
    }
    if video_noise is not None:
        batch["video_noise"] = video_noise
    return batch


def _strategy(**overrides: object) -> H3RefStrategy:
    kwargs: dict = {
        "references_per_sample": 2,
        "reference_dropout": 0.0,
        "reference_pair_seed": 42,
        "environment_ref_last": True,
        "audio_in_loss": False,
        "t_visual_cond": 0.999,
        "deps": _deps(),
        "position_ids_fn": _monotone_position_ids,
    }
    kwargs.update(overrides)
    return H3RefStrategy(**kwargs)


def _row_timesteps(inputs: H3ModelInputs) -> torch.Tensor:
    """Reconstruct the per-row ``t`` exactly: ``timestep[timestep_indices]`` (see h3_step)."""
    kwargs = inputs.packed.kwargs
    return kwargs["timestep"][kwargs["timestep_indices"]]


# --------------------------------------------------------------------------------------------------
# 7. The ABC contract + data sources
# --------------------------------------------------------------------------------------------------


def test_h3_ref_strategy_implements_the_training_strategy_abc() -> None:
    assert issubclass(H3RefStrategy, TrainingStrategy)


def test_get_data_sources_declares_the_four_h3_sources() -> None:
    strategy = H3RefStrategy(
        references_per_sample=2,
        reference_dropout=0.2,
        reference_pair_seed=42,
        environment_ref_last=True,
        audio_in_loss=False,
        t_visual_cond=0.999,
    )
    assert list(strategy.get_data_sources()) == [
        "h3_latents",
        "h3_conditions",
        "h3_reference_latents",
        "h3_audio_latents",
    ]


def test_the_constructor_mirrors_the_h3_config_field_names() -> None:
    """10-05's ``H3Config`` field names, so the Modal caller is a straight field-for-field read."""
    import inspect

    parameters = set(inspect.signature(H3RefStrategy.__init__).parameters)
    for field in (
        "references_per_sample",
        "reference_dropout",
        "reference_pair_seed",
        "environment_ref_last",
        "audio_in_loss",
        "t_visual_cond",
    ):
        assert field in parameters, field


def test_prepare_returns_a_model_inputs_subclass_carrying_the_packed_batch() -> None:
    inputs = _strategy().prepare_training_inputs(_batch())
    assert isinstance(inputs, ModelInputs)
    assert set(inputs.packed.kwargs) == set(H3_PACKED_BATCH_KEYS)


# --------------------------------------------------------------------------------------------------
# 7b. #52 interim mitigation — the cached-pool-size-vs-slot-count startup log
# --------------------------------------------------------------------------------------------------


def _pool(*refs: H3Reference) -> list:
    """A minimal ``(H3Reference, tensor)`` pool — ``_resolve_slots`` never reads the tensor."""
    return [(ref, torch.zeros(1, 1, 1)) for ref in refs]


def test_resolve_slots_logs_the_cached_pool_size_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Issue #52's interim mitigation: no cache provenance exists yet to tell a genuinely rotating
    pool apart from a stale/re-configured one, so the cheap half is to make the mismatch VISIBLE —
    log the cached pool size against the configured slot count once, at the first resolved segment.

    Logged ONCE per strategy instance, not once per step: a per-segment log would drown a real run
    in noise the operator has to scroll past on every single sample.
    """
    import logging

    strategy = H3RefStrategy(references_per_sample=2)
    characters = [
        H3Reference(path=f"refs/char_{s}.png", kind="character", subject_id=s,
                    width=1024, height=1536)
        for s in ("A", "B", "C")
    ]
    pool = _pool(*characters)

    with caplog.at_level(logging.INFO):
        strategy._resolve_slots(0, pool)
        strategy._resolve_slots(1, pool)

    pool_size_logs = [r for r in caplog.records if "cached reference pool" in r.message]
    assert len(pool_size_logs) == 1, (
        "the pool-size-vs-slot-count log must fire exactly ONCE per strategy instance"
    )
    assert "3" in pool_size_logs[0].message  # pool size
    assert "references_per_sample=2" in pool_size_logs[0].message


# --------------------------------------------------------------------------------------------------
# 8. The reference prefix: length, mask, and D-10-REFPIN
# --------------------------------------------------------------------------------------------------


def test_ref_seq_len_equals_the_reference_latent_rows_for_that_sample() -> None:
    inputs = _strategy().prepare_training_inputs(_batch())
    assert inputs.ref_seq_len == 2 * REF_ROWS
    assert inputs.packed.n_cond_video == 2 * REF_ROWS


def test_video_loss_mask_is_false_on_the_prefix_and_true_after_it() -> None:
    inputs = _strategy().prepare_training_inputs(_batch())
    mask = inputs.video_loss_mask
    assert not mask[:, : inputs.ref_seq_len].any()
    assert mask[:, inputs.ref_seq_len :].all()
    assert int(mask.sum()) == N_TARGET_VIDEO


def test_video_targets_are_TARGET_ONLY_not_the_combined_length() -> None:
    """A missing loss slice must fail on SHAPE, not train silently on the reference prefix."""
    inputs = _strategy().prepare_training_inputs(_batch())
    assert tuple(inputs.video_targets.shape) == (1, N_TARGET_VIDEO, PATCH_DIM)
    combined = inputs.packed.kwargs["hidden_states"].shape[1]
    assert combined != inputs.video_targets.shape[1], (
        "the combined length must differ from the target length, or the shape guard is inert"
    )


def test_the_reference_prefix_is_the_anchor_noise_aug_the_release_documents() -> None:
    """D-10-REFPIN: anchors enter at ``t_visual_cond``, matching ``keyframe_noise_aug``.

    ⚠ This test previously asserted the prefix was the reference latents VERBATIM, on the reading
    that "train references CLEAN". The pinned ``modular_pipeline.py::keyframe_noise_aug`` states the
    convention outright — *"the released model was trained with its anchors very slightly noised"* —
    and applies ``scale_noise(ref, 0.999, noise)`` while tagging those rows 0.999. Entering them
    verbatim tagged rows at a noise level their contents did not have.

    The load-bearing half of the old claim is unchanged and asserted below: this is a
    **0.1%-amplitude** pin, NOT latent-space regularization. Reference regularization stays an
    IMAGE-space decision.
    """
    batch = _batch()
    strategy = _strategy()
    inputs = strategy.prepare_training_inputs(batch)
    prefix = inputs.packed.kwargs["hidden_states"][:, : inputs.ref_seq_len, :]
    pool = {e["subject_id"]: e["latents"] for e in batch["h3_ref_latent_conditions"]["references"]}
    clean = torch.cat([pool[r.subject_id] for r in inputs.references], dim=1)

    assert not torch.equal(prefix, clean), (
        "the anchors entered verbatim: the release documents them as very slightly noised, and the "
        "rows are tagged at t_visual_cond either way"
    )
    # The deviation is exactly (1 - t) in amplitude — a pin, not regularization.
    residual = (prefix - strategy.t_visual_cond * clean).abs().max().item()
    assert residual <= (1.0 - strategy.t_visual_cond) * 6.0, (
        f"the reference prefix deviates from t_visual_cond * clean by {residual}, far more than the "
        f"{1.0 - strategy.t_visual_cond:.4f}-amplitude noise term — this is latent-space "
        f"regularization, which D-10-REFPIN forbids"
    )


def test_t_visual_cond_of_one_reproduces_verbatim_anchors_exactly() -> None:
    """The pin is a PARAMETER, so its identity element must be exact — no epsilon left behind."""
    batch = _batch()
    inputs = _strategy(t_visual_cond=1.0).prepare_training_inputs(batch)
    prefix = inputs.packed.kwargs["hidden_states"][:, : inputs.ref_seq_len, :]
    pool = {e["subject_id"]: e["latents"] for e in batch["h3_ref_latent_conditions"]["references"]}
    expected = torch.cat([pool[r.subject_id] for r in inputs.references], dim=1)
    assert torch.equal(prefix, expected)


def test_the_anchor_noise_is_reproducible_from_the_batch() -> None:
    """A supplied ``reference_noise`` is used verbatim, so the pin is testable and replayable."""
    batch = _batch()
    noise = torch.full((1, REF_ROWS * 2, PATCH_DIM), 2.0)
    batch["reference_noise"] = noise
    inputs = _strategy().prepare_training_inputs(batch)
    prefix = inputs.packed.kwargs["hidden_states"][:, : inputs.ref_seq_len, :]
    pool = {e["subject_id"]: e["latents"] for e in batch["h3_ref_latent_conditions"]["references"]}
    clean = torch.cat([pool[r.subject_id] for r in inputs.references], dim=1)
    expected = 0.999 * clean + 0.001 * noise
    assert torch.allclose(prefix, expected)


def test_t_visual_cond_is_passed_through_and_never_hardcoded() -> None:
    inputs = _strategy(t_visual_cond=0.97).prepare_training_inputs(_batch())
    row_t = _row_timesteps(inputs)
    video_indices = inputs.packed.kwargs["video_indices"]
    reference_rows = video_indices[: inputs.ref_seq_len]
    assert torch.allclose(row_t[reference_rows], torch.full((inputs.ref_seq_len,), 0.97))


def test_the_visual_pin_follows_the_schedule_up_because_it_is_a_max() -> None:
    batch = _batch()
    batch["t_video"] = 0.9995
    inputs = _strategy(t_visual_cond=0.999).prepare_training_inputs(batch)
    row_t = _row_timesteps(inputs)
    reference_rows = inputs.packed.kwargs["video_indices"][: inputs.ref_seq_len]
    assert torch.allclose(row_t[reference_rows], torch.full((inputs.ref_seq_len,), 0.9995))


# --------------------------------------------------------------------------------------------------
# 9. EXACTLY 2 blocks, environment LAST — at the STRATEGY layer
# --------------------------------------------------------------------------------------------------


def test_a_non_environment_sample_carries_two_character_blocks() -> None:
    inputs = _strategy().prepare_training_inputs(_batch())
    assert len(inputs.references) == 2
    assert [r.kind for r in inputs.references] == ["character", "character"]


def test_an_environment_bearing_sample_carries_exactly_two_blocks_environment_last() -> None:
    inputs = _strategy().prepare_training_inputs(_batch(with_environment=True))
    assert len(inputs.references) == 2
    assert [r.kind for r in inputs.references] == ["character", "environment"]
    assert inputs.ref_seq_len == 2 * REF_ROWS


def test_the_environment_regime_never_produces_a_third_block_across_the_corpus() -> None:
    strategy = _strategy()
    for segment in range(N_SEGMENTS):
        with_env = strategy.prepare_training_inputs(
            _batch(segment_index=segment, with_environment=True)
        )
        assert len(with_env.references) == 2, segment
        assert with_env.ref_seq_len == 2 * REF_ROWS, segment


def test_too_many_environment_references_raises() -> None:
    strategy = _strategy()
    batch = _batch(environments=[_environment("env1"), _environment("env2")])
    with pytest.raises(ValueError, match="environment"):
        strategy.prepare_training_inputs(batch)


def test_a_dropped_reference_sample_has_a_zero_prefix_and_a_fully_true_mask() -> None:
    inputs = _strategy(reference_dropout=1.0).prepare_training_inputs(_batch())
    assert inputs.ref_seq_len == 0
    assert inputs.references == ()
    assert inputs.reference_dropped is True
    assert inputs.video_loss_mask.all()
    assert inputs.video_loss_mask.shape[1] == N_TARGET_VIDEO


def test_dropout_is_driven_by_the_segment_and_step_not_by_chance() -> None:
    strategy = _strategy(reference_dropout=0.2)
    first = strategy.prepare_training_inputs(_batch(segment_index=5, step=11))
    second = strategy.prepare_training_inputs(_batch(segment_index=5, step=11))
    assert first.reference_dropped == second.reference_dropped
    assert [r.path for r in first.references] == [r.path for r in second.references]


# --------------------------------------------------------------------------------------------------
# 10. Audio: present, noised, and out of the loss (D-10-AUDIO)
# --------------------------------------------------------------------------------------------------


def test_audio_modality_is_populated_and_masked_out_when_audio_is_not_a_target() -> None:
    inputs = _strategy().prepare_training_inputs(_batch())
    assert inputs.audio is not None
    assert inputs.audio_loss_mask is not None
    assert not inputs.audio_loss_mask.any()
    assert tuple(inputs.audio_targets.shape) == (1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS)


def test_target_audio_rows_stay_present_and_noised() -> None:
    """Not targeting audio is NOT the same as training on silence."""
    inputs = _strategy().prepare_training_inputs(_batch())
    audio_rows = inputs.packed.kwargs["audio_hidden_states"]
    assert tuple(audio_rows.shape) == (1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS)
    assert audio_rows.abs().sum() > 0.0


def test_flipping_audio_in_loss_is_a_mask_change_not_a_restructure() -> None:
    off = _strategy(audio_in_loss=False).prepare_training_inputs(_batch())
    on = _strategy(audio_in_loss=True).prepare_training_inputs(_batch())
    assert off.audio_targets.shape == on.audio_targets.shape
    assert off.audio_loss_mask.shape == on.audio_loss_mask.shape
    assert not off.audio_loss_mask.any()
    assert on.audio_loss_mask.all()


# --------------------------------------------------------------------------------------------------
# 11. compute_loss — THE prefix-exclusion proof
# --------------------------------------------------------------------------------------------------


def _model_output(inputs: H3ModelInputs, *, prefix_fill: float, seed: int = 3) -> torch.Tensor:
    """A synthetic video output: a controllable reference prefix + a FIXED target half."""
    generator = torch.Generator().manual_seed(seed)
    target = torch.randn(1, inputs.packed.n_target_video, PATCH_DIM, generator=generator)
    prefix = torch.full((1, inputs.ref_seq_len, PATCH_DIM), prefix_fill)
    return torch.cat([prefix, target], dim=1)


def test_compute_loss_returns_a_zero_dim_scalar() -> None:
    strategy = _strategy()
    inputs = strategy.prepare_training_inputs(_batch())
    loss = strategy.compute_loss(inputs, _model_output(inputs, prefix_fill=0.0))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_a_GARBAGE_reference_prefix_and_a_ZERO_prefix_give_the_SAME_loss() -> None:
    """The single most important assertion in this plan — the CPU proof of the L-3 guard.

    Loss over the reference prefix is the dead-adapter bug: the objective is dominated by rows the
    model is not being asked to generate. If this ever fails, the prefix is leaking into the loss.
    """
    strategy = _strategy()
    inputs = strategy.prepare_training_inputs(_batch())
    zeros = strategy.compute_loss(inputs, _model_output(inputs, prefix_fill=0.0))
    garbage = strategy.compute_loss(inputs, _model_output(inputs, prefix_fill=1.0e6))
    assert torch.equal(zeros, garbage), (float(zeros), float(garbage))


def test_the_prefix_exclusion_holds_on_an_environment_bearing_sample_too() -> None:
    strategy = _strategy()
    inputs = strategy.prepare_training_inputs(_batch(with_environment=True))
    zeros = strategy.compute_loss(inputs, _model_output(inputs, prefix_fill=0.0))
    garbage = strategy.compute_loss(inputs, _model_output(inputs, prefix_fill=-9.9e5))
    assert torch.equal(zeros, garbage)


def test_compute_loss_accepts_the_transformers_video_audio_two_tuple() -> None:
    strategy = _strategy()
    inputs = strategy.prepare_training_inputs(_batch())
    video_out = _model_output(inputs, prefix_fill=0.0)
    audio_out = torch.randn(1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS)
    tupled = strategy.compute_loss(inputs, (video_out, audio_out))
    bare = strategy.compute_loss(inputs, video_out)
    assert torch.equal(tupled, bare), "audio must contribute nothing with audio_in_loss=False"


def test_audio_enters_the_loss_only_when_audio_in_loss_is_true() -> None:
    strategy = _strategy(audio_in_loss=True)
    inputs = strategy.prepare_training_inputs(_batch())
    video_out = _model_output(inputs, prefix_fill=0.0)
    quiet = strategy.compute_loss(inputs, (video_out, torch.zeros(1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS)))
    loud = strategy.compute_loss(inputs, (video_out, torch.full((1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS), 5.0)))
    assert not torch.equal(quiet, loud)


# --------------------------------------------------------------------------------------------------
# 12. The flow-match objective + the fail-loud seams
# --------------------------------------------------------------------------------------------------


def test_target_rows_are_noised_at_h3s_INVERTED_time_convention() -> None:
    """``t = 1`` is CLEAN in H3 (``t = 1 - sigma``), so ``x_t = t * x0 + (1 - t) * eps``."""
    noise = torch.full((1, N_TARGET_VIDEO, PATCH_DIM), 2.0)
    batch = _batch(video_noise=noise)
    inputs = _strategy().prepare_training_inputs(batch)
    clean = batch["h3_latent_conditions"]["latents"]
    noisy = inputs.packed.kwargs["hidden_states"][:, inputs.ref_seq_len :, :]
    assert torch.allclose(noisy, T_VIDEO * clean + (1.0 - T_VIDEO) * noise)


def test_video_targets_are_the_DATA_ward_clean_minus_noise_velocity() -> None:
    """⛔ DIVERGE #5 — H3 predicts a DATA-ward velocity, so the target is ``clean - noise``.

    This assertion was INVERTED when the strategy landed (10-08), carrying the LTX ``noise - clean``
    convention that every other strategy in this repo uses. The authority is
    ``diffusers`` at the pinned ``DIFFUSERS_SHA``, ``schedulers/scheduling_minimax_h3.py``:

      * L20-21 — "MiniMax-H3's transformer predicts a *data-ward* velocity, so ``x0 = x_t + sigma *
        v`` instead of diffusers' ``x0 = x_t - sigma * v``";
      * ``step`` L269 — ``denoised = sample + sigma_from_timestep * model_output``, with the
        docstring calling out "note the ``+``, the opposite of the usual flow-match convention".

    With ``x_t = t*x0 + (1-t)*eps`` (``scale_noise`` L200) and ``sigma = 1 - t``, solving
    ``x0 = x_t + sigma*v`` gives ``v = x0 - eps``. Training against the negation costs nothing
    observable — same shapes, same loss magnitude, an ordinary curve — and produces an adapter that
    pushes every render the wrong way.
    """
    noise = torch.full((1, N_TARGET_VIDEO, PATCH_DIM), 2.0)
    batch = _batch(video_noise=noise)
    inputs = _strategy().prepare_training_inputs(batch)
    clean = batch["h3_latent_conditions"]["latents"]
    assert torch.allclose(inputs.video_targets, clean - noise)
    assert not torch.allclose(inputs.video_targets, noise - clean), (
        "the LTX noise-ward convention must NOT be what H3 trains against"
    )


def test_the_velocity_target_inverts_the_scheduler_forward_process_exactly() -> None:
    """End-to-end algebra check: ``x_t + (1 - t) * v`` must recover the CLEAN latent.

    Independent of how the target is spelled — it reconstructs ``x0`` through the scheduler's own
    ``step`` arithmetic, so it fails for any sign or scale error in the objective.
    """
    noise = torch.randn(1, N_TARGET_VIDEO, PATCH_DIM, generator=torch.Generator().manual_seed(3))
    batch = _batch(video_noise=noise)
    inputs = _strategy().prepare_training_inputs(batch)
    clean = batch["h3_latent_conditions"]["latents"]
    x_t = inputs.packed.kwargs["hidden_states"][:, inputs.ref_seq_len :, :]
    recovered = x_t + (1.0 - T_VIDEO) * inputs.video_targets  # scheduler step L269
    assert torch.allclose(recovered, clean, atol=1e-5)


def test_audio_targets_carry_the_same_data_ward_sign() -> None:
    inputs = _strategy(audio_in_loss=True).prepare_training_inputs(_batch())
    assert inputs.audio_targets is not None
    # Rebuilt from the packed rows the same way the video check above does.
    assert inputs.audio_targets.shape == (1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS)


def test_a_four_dimensional_latent_raises_when_no_patch_size_is_supplied() -> None:
    """Opt-in by design: with no ``patch_size`` the strategy still REFUSES to guess the order."""
    batch = _batch()
    batch["h3_latent_conditions"] = {"latents": torch.randn(24, 1, 8, 8)}
    with pytest.raises(ValueError, match="patchif"):
        _strategy().prepare_training_inputs(batch)


def test_a_four_dimensional_latent_is_patchified_when_patch_size_is_supplied() -> None:
    """SEAM 1 closed: with the LIVE model's patch size the transcribed transform is applied.

    The synthetic widths here are tiny (``PATCH_DIM == 6``), so the patch is ``(1, 1, 1)`` to keep
    ``patch_dim == in_channels``; the ORDER contract itself is proved at the real ``(1, 2, 2)`` in
    ``tests/test_h3_packing.py`` against an independent nested-loop gather.
    """
    from signet_trainer.conditioning.h3_packing import h3_video_rows

    patch = (1, 1, 1)
    latents = torch.randn(PATCH_DIM, 2, 3, 4, generator=torch.Generator().manual_seed(5))
    batch = _batch()
    batch["h3_latent_conditions"] = {"latents": latents}
    inputs = _strategy(patch_size=patch).prepare_training_inputs(batch)
    expected_rows = 2 * 3 * 4
    assert inputs.packed.n_target_video == expected_rows
    # The rows fed in are exactly what the transcribed patchify produces — no re-derivation.
    rows = h3_video_rows(latents, patch)
    assert rows.shape == (1, expected_rows, PATCH_DIM)
    # And the scheduler identity still recovers the patchified CLEAN rows (DIVERGE #5).
    x_t = inputs.packed.kwargs["hidden_states"][:, inputs.ref_seq_len :, :]
    assert torch.allclose(x_t + (1.0 - T_VIDEO) * inputs.video_targets, rows, atol=1e-5)


def test_absent_position_ids_with_no_builder_raises() -> None:
    strategy = _strategy(position_ids_fn=None)
    with pytest.raises(ValueError, match="position_ids"):
        strategy.prepare_training_inputs(_batch())


def test_batch_supplied_position_ids_are_used_verbatim() -> None:
    batch = _batch()
    seq = N_TEXT + 2 * REF_ROWS + N_TARGET_AUDIO + N_TARGET_VIDEO
    supplied = torch.full((seq, 3), 4.0, dtype=torch.float64)
    batch["position_ids"] = supplied
    inputs = _strategy(position_ids_fn=None).prepare_training_inputs(batch)
    assert torch.equal(inputs.packed.kwargs["position_ids"], supplied)


def test_a_reference_pool_too_small_for_the_slot_count_raises() -> None:
    with pytest.raises(ValueError):
        _strategy().prepare_training_inputs(_batch(characters=[_character("A")]))


def test_the_max_packed_rows_ceiling_is_threaded_through() -> None:
    strategy = _strategy(max_packed_rows=4)
    with pytest.raises(ValueError, match="exceeds the validated ceiling"):
        strategy.prepare_training_inputs(_batch())


# --------------------------------------------------------------------------------------------------
# 13. The four mandatory DIVERGE notes
# --------------------------------------------------------------------------------------------------


def test_the_four_diverge_notes_are_recorded_in_the_source() -> None:
    """Each one is a place a reasonable reader would copy the LTX analog and be WRONG."""
    source = _H3_REF_SRC.read_text(encoding="utf-8")
    assert source.count("DIVERGE") >= 4, "all four DIVERGE notes must be present"
    # 1 — the reference prefix is not clean-at-timestep-0; the pin is a training PARAMETER
    assert "D-10-REFPIN" in source
    assert "t_visual_cond" in source
    # 2 — video-only means loss-MASKING; the a2v analog is INVERTED
    assert "a2v" in source and "INVERTED" in source
    # 3 — RoPE: 3-axis float64, area-normalized, 96 of 128 head dims, pattern port only
    assert "96 of 128" in source
    # 4 — modality tags are a checkpoint contract with no LTX analog
    assert "checkpoint contract" in source.lower()


def test_the_audio_drift_eval_axis_is_recorded_not_assumed_away() -> None:
    """All 50 blocks are shared, so a video-only-loss LoRA still moves audio-path weights."""
    source = _H3_REF_SRC.read_text(encoding="utf-8")
    assert "eval axis" in source.lower()
    assert "0 of 44" in source


# --------------------------------------------------------------------------------------------------
# 14. The persisted vision spans, the prompt-only drop state, and the stale-cache refusal
#
# All three are ONE defect class: PHASE A computed something and the train side either read it from
# a batch key nothing populated, or inherited a state describing references the step no longer had.
# Neither has a shape error, and the dry-run sees neither — it calls `build_h3_packed_batch`
# directly with its own derived nominal spans, so it was CORRECT while the real path was not.
# --------------------------------------------------------------------------------------------------


def test_the_cached_vision_spans_reach_the_modality_tags() -> None:
    """The spans were computed in PHASE A and DROPPED; `h3_ref` read a batch key nothing set.

    At 896 short edge that is ~1,000-1,600 vision rows per reference against ~100 prompt tokens, so
    >90% of the text stream modulated with the TEXT AdaLN row instead of the VIDEO one
    (`timestep_indices * H3_MODALITY_NUM + token_tags`). Asserted on the REALIZED tags, never on the
    presence of a keyword.
    """
    inputs = _strategy().prepare_training_inputs(_batch())
    tags = inputs.packed.kwargs["token_tags"]
    for start, stop in VISION_SPANS:
        assert set(tags[start:stop].tolist()) == {H3_VIDEO_TAG}, (
            f"vision span ({start}, {stop}) is tagged {sorted(set(tags[start:stop].tolist()))}, not "
            f"VIDEO. Qwen vision rows live in the TEXT stream but modulate as VIDEO — a checkpoint "
            f"contract, not a convention."
        )
    # Non-vacuity: the rest of the text stream must still be TEXT, or the loop above would pass on a
    # batch where every row happened to be tagged video.
    covered = {i for start, stop in VISION_SPANS for i in range(start, stop)}
    text_only = [i for i in range(N_TEXT) if i not in covered]
    assert text_only and set(tags[text_only].tolist()) == {H3_TEXT_TAG}


def test_a_top_level_batch_vision_spans_key_is_IGNORED() -> None:
    """The spans are a property of the CACHED text state, not something a caller can inject.

    `batch["vision_spans"]` is what the old code read and nothing ever wrote. Honouring it now would
    keep the dead seam alive and let a caller tag rows that do not describe the payload.
    """
    batch = _batch()
    batch["vision_spans"] = ((0, N_TEXT),)  # would tag the WHOLE text stream video if honoured
    inputs = _strategy().prepare_training_inputs(batch)
    tags = inputs.packed.kwargs["token_tags"]
    assert int(tags[0]) == H3_TEXT_TAG, (
        "a top-level batch['vision_spans'] was honoured — the spans must come from the payload the "
        "pre-encode wrote, which is the only thing that knows where the vision blocks actually are"
    )


def test_a_dropped_step_uses_the_PROMPT_ONLY_state_with_no_vision_spans() -> None:
    """A 'dropped' step used to keep describing both references in its text state.

    D-10-REFDROP removed the reference LATENT rows only, so the regime matched no inference regime
    that exists (a no-reference request's presentation contains no vision blocks at all) and identity
    leaked into ~20% of steps.
    """
    inputs = _strategy(reference_dropout=1.0).prepare_training_inputs(_batch())
    assert inputs.reference_dropped
    assert inputs.ref_seq_len == 0, "a dropped step carries no reference latent rows"

    text_rows = inputs.packed.n_text
    assert text_rows == N_PROMPT_ONLY_TEXT, (
        f"a dropped step conditioned on {text_rows} text row(s); the PROMPT-ONLY state has "
        f"{N_PROMPT_ONLY_TEXT}. Reading the reference-bearing state on a dropped step is the defect."
    )
    tags = inputs.packed.kwargs["token_tags"][:text_rows]
    assert set(tags.tolist()) == {H3_TEXT_TAG}, (
        "every row of a reference-free presentation is TEXT — it contains no vision blocks"
    )


def test_the_stale_version_one_cache_is_REFUSED_by_shape() -> None:
    """The 88 payloads on the Volume are bare tensors; consuming them restores both defects."""
    batch = _batch()
    batch["h3_text_conditions"] = torch.randn(1, N_TEXT, TEXT_DIM)
    with pytest.raises(ValueError, match="version-1|RE-ENCODED"):
        _strategy().prepare_training_inputs(batch)


def test_an_unknown_payload_version_is_REFUSED_rather_than_partially_read() -> None:
    batch = _batch()
    payload = dict(batch["h3_text_conditions"])
    payload["h3_text_payload_version"] = 99
    batch["h3_text_conditions"] = payload
    with pytest.raises(ValueError, match="version 99"):
        _strategy().prepare_training_inputs(batch)


# --------------------------------------------------------------------------------------------------
# 15. The absent-audio block — rows PRESENT, pure noise, entirely out of the loss
# --------------------------------------------------------------------------------------------------


def _absent_audio_batch(**kwargs: object) -> dict:
    return _batch(audio={"latents": None, "present": False, "reason": "no stream"}, **kwargs)


def test_an_audio_less_sample_still_carries_its_priced_target_audio_rows() -> None:
    """Emitting ZERO rows made the training path structurally unrunnable on this corpus.

    `h3_packed_seq_len` and `H3PositionIdsBuilder` both price `h3_audio_rows(frames)`
    UNCONDITIONALLY and the builder's `n_target_audio` check is a hard equality — so a 0-row audio
    block fails the FIRST batch. It also contradicted `h3_loss_mask`'s own docstring ("target audio
    ROWS stay in the packed sequence and stay NOISED").
    """
    inputs = _strategy(target_audio_rows=N_TARGET_AUDIO).prepare_training_inputs(
        _absent_audio_batch()
    )
    assert inputs.packed.n_target_audio == N_TARGET_AUDIO
    assert inputs.packed.n_cond_audio == 0
    assert inputs.packed.kwargs["audio_hidden_states"].shape[1] == N_TARGET_AUDIO


def test_the_absent_audio_block_is_PURE_noise_not_interpolated_toward_silence() -> None:
    """Interpolating toward a zero 'clean' target would present SILENCE as the thing generated.

    The operator rejected exactly that: it would teach the model to be silent for every prompt, a
    behavioural change that would have to be untrained later. There is no x0 here, so the rows carry
    the unit-variance draw at every ``t``.
    """
    noise = torch.full((1, N_TARGET_AUDIO, AUDIO_IN_CHANNELS), 3.0)
    batch = _absent_audio_batch()
    batch["audio_noise"] = noise
    inputs = _strategy(target_audio_rows=N_TARGET_AUDIO).prepare_training_inputs(batch)
    rows = inputs.packed.kwargs["audio_hidden_states"]
    assert torch.equal(rows, noise), (
        f"the absent-audio block is {rows.flatten()[:3].tolist()}, not the noise draw. "
        f"`(1 - t) * eps` would shrink toward zero as t -> 1, i.e. converge on silence."
    )


def test_the_absent_audio_block_is_entirely_out_of_the_loss() -> None:
    inputs = _strategy(target_audio_rows=N_TARGET_AUDIO).prepare_training_inputs(
        _absent_audio_batch()
    )
    assert not bool(inputs.audio_loss_mask.any()), (
        "synthesized noise must never enter the loss — there is no ground truth behind it"
    )


def test_audio_in_loss_with_an_audio_less_sample_is_REFUSED() -> None:
    """Silently masking a flag the config turned ON is how a misconfiguration becomes invisible."""
    with pytest.raises(ValueError, match="audio_in_loss=True"):
        _strategy(audio_in_loss=True, target_audio_rows=N_TARGET_AUDIO).prepare_training_inputs(
            _absent_audio_batch()
        )


def test_an_absent_audio_row_count_that_cannot_be_derived_is_REFUSED_not_zeroed() -> None:
    """Defaulting to 0 IS the defect; the count is a function of the campaign frame count."""
    with pytest.raises(ValueError, match="target_audio_rows"):
        _strategy().prepare_training_inputs(_absent_audio_batch())


# --------------------------------------------------------------------------------------------------
# 11. issue #52 — reference-pool provenance: refuse ONLY a positively-marked explicit-manifest
#     mismatch; a flag-less (legacy) or pool-derived payload must be UNAFFECTED.
#
# ``_batch()``'s default pool (CHARACTERS = A, B, C — three entries) against ``_strategy()``'s
# default ``references_per_sample=2`` is already the Embe corpus regime this fix must never touch:
# a pool DELIBERATELY larger than the slot count, rotating across segments. Every test in this file
# that calls plain ``_batch()`` already exercises that combination without a flag key at all, so
# this section adds the DEDICATED, adversarial-audit-cited coverage on top.
# --------------------------------------------------------------------------------------------------


def test_a_flagless_pool_larger_than_slot_count_is_not_refused() -> None:
    """The Embe corpus regime: a 3-entry pool, 2 slots, NO 'explicit_manifest' key at all.

    This is what ``_batch()`` already builds by default (``h3_ref_latent_conditions`` carries no
    provenance key), so this test names the property directly rather than leaning on it being an
    accident of every other test's fixture.
    """
    batch = _batch()
    assert "explicit_manifest" not in batch["h3_ref_latent_conditions"]
    inputs = _strategy().prepare_training_inputs(batch)
    assert len(inputs.references) == 2


def test_explicit_manifest_true_with_a_mismatched_pool_is_refused() -> None:
    """The hard half of issue #52: a row positively marked explicit whose cached pool disagrees.

    The scenario named in the issue: a 3-reference cache exists (references_per_sample was 3 when
    it was written, and the row's references came from an explicit 'reference_paths' list), and a
    'references_per_sample: 2' config now points at it — the realistic path in is a VRAM retreat
    against a cache already paid to encode. That must be LOUD, not a silent 2-of-3 rotation.
    """
    batch = _batch()  # CHARACTERS pool has 3 entries
    batch["h3_ref_latent_conditions"]["explicit_manifest"] = True
    with pytest.raises(ValueError, match="explicit-manifest"):
        _strategy(references_per_sample=2).prepare_training_inputs(batch)


def test_explicit_manifest_true_with_a_matching_pool_is_allowed() -> None:
    """A positively-marked explicit row whose pool size DOES agree trains normally."""
    batch = _batch(characters=[_character("A"), _character("B")])
    batch["h3_ref_latent_conditions"]["explicit_manifest"] = True
    inputs = _strategy(references_per_sample=2).prepare_training_inputs(batch)
    assert len(inputs.references) == 2


def test_explicit_manifest_mismatch_is_refused_on_a_dropped_step_too() -> None:
    """The cache contract is validated on EVERY step that touches it, dropped or not (D-10-REFDROP
    only removes the reference ROWS from a step's inputs — it never skips validating the cache the
    step still reads).
    """
    batch = _batch()  # CHARACTERS pool has 3 entries
    batch["h3_ref_latent_conditions"]["explicit_manifest"] = True
    with pytest.raises(ValueError, match="explicit-manifest"):
        _strategy(references_per_sample=2, reference_dropout=1.0).prepare_training_inputs(batch)


def test_a_flagless_legacy_payload_trains_byte_identically_to_an_explicit_false_payload() -> None:
    """BINDING adversarial-audit constraint: a MISSING flag must behave as pool-derived —
    byte-identical to today (i.e. to a payload that positively records ``explicit_manifest=False``,
    the only other value that skips the new refusal). Every cache on the Volume predates this field,
    so this is the regression the fix may never introduce.
    """
    video_noise = torch.full((1, N_TARGET_VIDEO, PATCH_DIM), 0.25)
    reference_noise = torch.full((1, 2 * REF_ROWS, PATCH_DIM), -0.5)

    flagless = _batch(video_noise=video_noise)
    flagless["reference_noise"] = reference_noise
    assert "explicit_manifest" not in flagless["h3_ref_latent_conditions"]

    flagged_false = _batch(video_noise=video_noise)
    flagged_false["reference_noise"] = reference_noise
    flagged_false["h3_ref_latent_conditions"]["explicit_manifest"] = False

    strategy_a = _strategy()
    strategy_b = _strategy()
    inputs_a = strategy_a.prepare_training_inputs(flagless)
    inputs_b = strategy_b.prepare_training_inputs(flagged_false)

    assert [r.path for r in inputs_a.references] == [r.path for r in inputs_b.references]
    assert torch.equal(
        inputs_a.packed.kwargs["hidden_states"], inputs_b.packed.kwargs["hidden_states"]
    )
    assert torch.equal(inputs_a.video_targets, inputs_b.video_targets)
