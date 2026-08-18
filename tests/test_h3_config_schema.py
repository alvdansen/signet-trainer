"""Phase 10 (H3-02 / H3-04) — the config layer becomes FAMILY-AWARE.

Three failure modes this suite exists to prevent, all of which would otherwise reach a metered A100:

1. **The silent LoRA-target failure.** ``lora.target_modules`` used to hard-default to the ten LTX
   suffixes. On H3 that default is not loud: ``ff.net.0.proj`` / ``ff.net.2`` are byte-identical
   across the two families, so the LTX default still matches **104** H3 modules (100 main + 4
   ``token_refiner.refiner_blocks``). ``train/loop.py``'s ``"No trainable parameters found. Is LoRA
   applied?"`` guard only fires on an EMPTY trainable set, so it never fires — and the run proceeds
   on an attn-blind, refiner-polluted, ~1/3-capacity adapter that produces plausible-but-wrong
   output. Pinned upstream by ``tests/test_h3_lora_targets.py::
   test_ltx_default_on_h3_fails_silently_not_loud``. The family-SELECTED default is therefore a
   **correctness requirement with no runtime backstop behind it**.

2. **The nominal-pair budget hole.** ``validate_h3_reference_budget`` must be called, never
   ``h3_packed_seq_len`` on one nominal pair: at reference short edge 1024 the nominal ``A+B`` pair
   prices at 12,362 rows and PASSES, while six of the twelve character-by-environment pairs are over
   the 13,777-row ceiling and the first such segment OOMs. That is why the reference set is declared
   as SPLIT character/environment lists — a single flat list cannot express which images pair with
   which, so the real 15-pair domain could not be enumerated.

3. **The silently-ignored config block.** An H3 tunable set while ``model.family`` is ``ltx`` is
   REJECTED (the bidirectional lean field-split REVERSE guard), never quietly dropped.

Everything here is CPU-only and builds configs from dicts, so the whole file runs free on Windows/CI
before any GPU is provisioned.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from signet_trainer.config.schema import (
    LTX_DEFAULT_LORA_TARGETS,
    H3Config,
    LoraConfig,
    ModelConfig,
    SignetConfig,
)

# The operator's real Phase-10 reference corpus, AFTER D-10-CROP (crop never pad, and the crop must
# save the face). Labels are load-bearing: they are what a budget refusal names, and a refusal that
# says "some pair is over budget" is not actionable at 3am.
CHARACTER_REFS = [
    (832, 1248, "A"),  # IMG_3659
    (2048, 2048, "B"),  # IMG_3725
    (1024, 1536, "C"),  # tmppvclbw6u, 1037 -> 1024 wide
]
ENVIRONMENT_REFS = [
    (1344, 768, "029"),
    (1024, 1024, "000"),
    (1440, 800, "008"),  # 1456x816 -> 1440x800
    (1344, 768, "023"),
]


def _ltx_payload(**over) -> dict:
    """A minimal VALID LTX config — the byte-identical-load control for every H3 change."""
    payload = {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "training": {"max_steps": 100},
    }
    payload.update(over)
    return payload


def _h3_payload(**over) -> dict:
    """A minimal VALID H3 config: 22 frames (17n+5), 1344x768 canvas, the full reference set."""
    payload = {
        "training_dims": [1344, 768, 22],
        "data": {
            "preprocessed_data_root": "/data/h3_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x22"],
        },
        "model": {"family": "h3"},
        "training": {"max_steps": 100},
        "h3": {
            "character_reference_sizes": list(CHARACTER_REFS),
            "environment_reference_sizes": list(ENVIRONMENT_REFS),
        },
        # #20 (checklist item 1): the render triple (validation.width/height/frame_count) is now
        # family-validated at load — the shared default frame_count=49 is an LTX clip length and
        # is OUTSIDE MiniMax-H3's renderable band, so an h3 payload must declare a legal one.
        "validation": {"frame_count": 124},
    }
    payload.update(over)
    return payload


# ======================================================================================
# Task 1 — ModelConfig.family + the H3Config block
# ======================================================================================


def test_model_config_defaults_to_ltx_family() -> None:
    """Every existing config loads unchanged: the discriminator defaults to ``ltx``."""
    assert ModelConfig().family == "ltx"


def test_model_config_accepts_h3_family() -> None:
    assert ModelConfig(family="h3").family == "h3"


def test_model_config_rejects_unknown_family() -> None:
    """The discriminator is an allowlist — a typo'd or aspirational family dies at load."""
    with pytest.raises(ValidationError):
        ModelConfig(family="wan")


def test_model_config_h3_only_ids_default_to_none() -> None:
    """``vae_id`` / ``audio_vae_id`` are H3-only; both are None for LTX (no behaviour change)."""
    m = ModelConfig()
    assert m.vae_id is None
    assert m.audio_vae_id is None


def test_family_is_an_explicit_discriminator_not_a_filename_sniff() -> None:
    """``base_variant_of()``'s filename sniff does NOT transfer — H3 model IDs are DIRECTORIES.

    ``ltx-2.3-22b-dev.safetensors`` is a FILE; H3's are dirs under ``WEIGHTS_DIR``
    (``minimax-h3/transformer_ref`` etc.), so there is no suffix to sniff. Setting an H3-shaped
    ``model_id`` must NOT flip the family by itself — only the explicit discriminator does.
    """
    m = ModelConfig(model_id="minimax-h3/transformer_ref")
    assert m.family == "ltx", "family must never be inferred from the model_id string"


def test_h3_config_carries_the_locked_d10_defaults() -> None:
    h3 = H3Config()
    assert h3.reference_image_short_edge == 896  # Phase 10 VRAM decision (spec is 2048)
    assert h3.reference_dropout == pytest.approx(0.2)  # D-10-REFDROP
    assert h3.reference_pair_seed == 42  # D-10-PAIRSEED
    assert h3.references_per_sample == 2  # Ref2VA default; 1 is legal for single-control tasks
    assert h3.environment_ref_last is True  # D-10-REFORDER
    assert h3.prompt_tokens_estimate == 96
    assert h3.text_encoder_layer == 50  # Qwen3-VL hidden_states[50] of 64
    assert h3.audio_in_loss is False  # D-10-AUDIO — loss-MASKING, not arch-skipping
    assert h3.gpu_usable_gib == pytest.approx(79.25)
    assert h3.resident_gib == pytest.approx(62.97)
    assert h3.mib_per_packed_row == pytest.approx(1.21)
    assert tuple(h3.target_aspect) == (16, 9)
    assert h3.character_reference_sizes == []
    assert h3.environment_reference_sizes == []


def test_every_h3_field_is_documented() -> None:
    """A tunable with no description is an undocumented hardcode wearing a config field's clothes."""
    undocumented = [n for n, f in H3Config.model_fields.items() if not f.description]
    assert not undocumented, f"H3Config fields missing a description: {undocumented}"


def test_h3_block_forbids_unknown_keys() -> None:
    """``extra='forbid'`` is inherited — a typo'd key in an ``h3:`` YAML block dies at load."""
    with pytest.raises(ValidationError):
        H3Config(refrence_dropout=0.2)  # codespell:ignore


def test_no_arch_smoke_only_mode_switch_in_the_config_block() -> None:
    """Deliberately NOT a field: ``run_h3_arch_gate`` fires unconditionally at the front of BOTH
    ``h3_preprocess`` and ``h3_train``, so abort-before-spend is guaranteed by every real dispatch.
    A config knob here would let a domain/recipe block change a Modal function's OPERATIONAL mode.
    """
    assert "arch_smoke_only" not in H3Config.model_fields


def test_references_per_sample_is_pinned_at_two() -> None:
    """The exactly-2-slot rule: the environment ref SUBSTITUTES for the second character slot."""
    with pytest.raises(ValidationError) as exc:
        H3Config(references_per_sample=3)
    msg = str(exc.value)
    assert "2" in msg
    assert "substitut" in msg.lower(), "the refusal must name the substitution rule, not just a bound"


def test_reference_short_edge_must_be_a_multiple_of_32() -> None:
    with pytest.raises(ValidationError):
        H3Config(reference_image_short_edge=900)
    assert H3Config(reference_image_short_edge=1024).reference_image_short_edge == 1024


def test_reference_short_edge_has_a_floor() -> None:
    with pytest.raises(ValidationError):
        H3Config(reference_image_short_edge=128)


def test_reference_sizes_are_split_lists_and_accept_labels() -> None:
    """SPLIT, not one flat list: the budget validator must enumerate the real pairing domain
    (character pairs PLUS character-by-environment pairs). A flat list cannot express which images
    can pair with which. Labels are optional but flow into refusal messages.
    """
    h3 = H3Config(
        character_reference_sizes=list(CHARACTER_REFS),
        environment_reference_sizes=[(1344, 768)],
    )
    assert len(h3.character_reference_sizes) == 3
    assert len(h3.environment_reference_sizes) == 1
    assert tuple(h3.character_reference_sizes[2]) == (1024, 1536, "C")


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_image_short_edge", 1024),
        ("reference_dropout", 0.1),
        ("reference_pair_seed", 7),
        ("environment_ref_last", False),
        ("prompt_tokens_estimate", 128),
        ("text_encoder_layer", 63),
        ("audio_in_loss", True),
        ("gpu_usable_gib", 139.0),
        ("resident_gib", 60.0),
        ("mib_per_packed_row", 1.5),
        ("target_aspect", (9, 16)),
        ("character_reference_sizes", [(832, 1248)]),
        ("environment_reference_sizes", [(1344, 768)]),
    ],
)
def test_h3_field_under_a_non_h3_family_is_rejected_not_silently_ignored(field, value) -> None:
    """The bidirectional lean field-split REVERSE guard (T-10-05-T)."""
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(_ltx_payload(h3={field: value}))
    msg = str(exc.value)
    assert field in msg, "the refusal must NAME the offending field"
    assert "ltx" in msg, "the refusal must NAME the family"


@pytest.mark.parametrize("field,value", [("vae_id", "minimax-h3/vae"), ("audio_vae_id", "x")])
def test_h3_only_model_ids_under_a_non_h3_family_are_rejected(field, value) -> None:
    """Same doctrine, same raise: ``vae_id`` / ``audio_vae_id`` are H3-only fields."""
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(_ltx_payload(model={field: value}))
    assert field in str(exc.value)


def test_all_default_h3_block_loads_fine_under_the_ltx_family() -> None:
    """The reverse guard must fire on non-DEFAULT values only — an untouched block is invisible."""
    cfg = SignetConfig.model_validate(_ltx_payload())
    assert cfg.model.family == "ltx"
    assert cfg.h3.reference_image_short_edge == 896  # present, defaulted, unused


def test_existing_ltx_config_loads_byte_identically() -> None:
    """Every new field is additive + defaulted, so a pre-Phase-10 payload is unaffected."""
    cfg = SignetConfig.model_validate(_ltx_payload())
    assert cfg.training_dims == (768, 512, 49)
    assert cfg.lora.rank == 64
    assert cfg.conditioning.mode == "none"
    assert "attn1.to_q" in cfg.lora.target_modules
    assert len(cfg.lora.target_modules) == 10


# ======================================================================================
# Task 2 — family-selected LoRA default + the H3 cross-field frame-law / budget checks
# ======================================================================================


def test_config_layer_h3_regex_is_byte_identical_to_the_peft_copy() -> None:
    """Single source of truth enforced by TEST, not by a cross-import that would drag ``peft`` into
    the config closure (the ``A2V_CROSS_MODAL_ATTN_TARGETS`` precedent).
    """
    from signet_trainer.config.validators import H3_LORA_TARGET_REGEX as config_copy
    from signet_trainer.lora.peft import H3_LORA_TARGET_REGEX as peft_copy

    assert config_copy == peft_copy


def test_config_layer_visual_pin_is_identical_to_the_h3_step_copy() -> None:
    """SEAM 3 (10-08 carry-forward): ``t_visual_cond`` had no ``H3Config`` field.

    D-10-REFPIN calls the TRAINING pin "a separate, deliberately parameterized decision" and
    D-NOHARDCODE puts tunables in the YAML, but 10-05 shipped five of the six H3 field names and
    missed this one — leaving the training pin reachable only as a Python default. It is a field now,
    and its default is enforced identical to ``train/h3_step.H3_VISUAL_CONDITION_PIN`` by TEST
    rather than by a cross-import (the ``H3_LORA_TARGET_REGEX`` precedent directly above: importing
    ``train.h3_step`` would widen ``download_image``'s closure for one float).
    """
    from signet_trainer.config.validators import H3_VISUAL_CONDITION_PIN as config_copy
    from signet_trainer.train.h3_step import H3_VISUAL_CONDITION_PIN as step_copy

    assert config_copy == step_copy
    assert SignetConfig.model_validate(_h3_payload()).h3.t_visual_cond == step_copy


def test_t_visual_cond_is_a_settable_field_in_the_inverted_time_convention() -> None:
    """It must be tunable from YAML, and it is a ``t`` in ``[0, 1]`` where 1 is CLEAN."""
    payload = _h3_payload()
    payload["h3"] = {**payload.get("h3", {}), "t_visual_cond": 0.97}
    assert SignetConfig.model_validate(payload).h3.t_visual_cond == pytest.approx(0.97)

    for bad in (-0.01, 1.01):
        payload["h3"] = {**payload.get("h3", {}), "t_visual_cond": bad}
        with pytest.raises(ValidationError):
            SignetConfig.model_validate(payload)


def test_schema_does_not_import_peft_or_torch() -> None:
    """T-10-05-SC: ``schema.py`` stays pydantic + stdlib. A heavy import here widens
    ``download_image``'s closure, whose only symptom is a ModuleNotFoundError in a PAID container.
    """
    from pathlib import Path

    src = Path(__import__("signet_trainer.config.schema", fromlist=["x"]).__file__)
    code = src.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:from|import)\s+(?:torch|peft)\b", code, re.MULTILINE)
    assert not re.search(r"^\s*from\s+signet_trainer\.lora\b", code, re.MULTILINE)


def test_h3_family_resolves_the_h3_path_regex_without_an_override() -> None:
    """H3-02. The LTX-shaped default would NOT die loud on H3 — it matches 104 modules and trains a
    wrong adapter silently. Family selection is the only thing standing there.
    """
    from signet_trainer.lora.peft import H3_LORA_TARGET_REGEX

    cfg = SignetConfig.model_validate(_h3_payload())
    assert cfg.resolved_lora_targets() == H3_LORA_TARGET_REGEX
    assert isinstance(cfg.resolved_lora_targets(), str)


def test_ltx_family_keeps_the_exact_ten_suffixes() -> None:
    cfg = SignetConfig.model_validate(_ltx_payload())
    assert list(cfg.resolved_lora_targets()) == list(LTX_DEFAULT_LORA_TARGETS)
    assert len(LTX_DEFAULT_LORA_TARGETS) == 10
    # The ff.net inclusion invariant — attn-only underfits identity capacity.
    assert "ff.net.0.proj" in LTX_DEFAULT_LORA_TARGETS
    assert "ff.net.2" in LTX_DEFAULT_LORA_TARGETS


def test_explicit_target_modules_beats_the_family_default() -> None:
    explicit = ["attn.to_q", "ff.net.2"]
    cfg = SignetConfig.model_validate(_h3_payload(lora={"target_modules": explicit}))
    assert cfg.resolved_lora_targets() == explicit


def test_resolved_targets_are_written_back_so_every_consumer_is_family_correct() -> None:
    """``modal/fns.py`` and ``train/validate_gate.py`` read ``cfg.lora.target_modules`` directly.
    The resolved value is written back in-place so a consumer that never learns about
    ``resolved_lora_targets()`` still cannot inject an LTX-shaped adapter into an H3 run.
    """
    from signet_trainer.lora.peft import H3_LORA_TARGET_REGEX

    h3 = SignetConfig.model_validate(_h3_payload())
    assert h3.lora.target_modules == H3_LORA_TARGET_REGEX

    ltx = SignetConfig.model_validate(_ltx_payload())
    assert list(ltx.lora.target_modules) == list(LTX_DEFAULT_LORA_TARGETS)


@pytest.mark.parametrize("payload_fn", [_ltx_payload, _h3_payload])
def test_an_explicitly_empty_target_override_is_refused(payload_fn) -> None:
    """Both consumers spell the read ``cfg.lora.target_modules or P1_FF_LORA_TARGETS``
    (``modal/fns.py``, ``train/validate_gate.py``), so an EMPTY override falls through to the LTX
    list — which on H3 matches 104 modules and never trips the trainable-param guard. Refuse it.
    """
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload_fn(lora={"target_modules": []}))
    assert "EMPTY" in str(exc.value)


def test_a2v_guard_refuses_a_regex_target_form_instead_of_char_exploding() -> None:
    """``list("abc")`` == ``['a','b','c']`` — the char-explosion class that was live in
    ``build_lora_config`` until Plan 10-02. The a2v suffix guard must refuse a regex, not guess.
    """
    from signet_trainer.config.validators import H3_LORA_TARGET_REGEX, validate_a2v_lora_targets

    with pytest.raises(ValueError, match="REGEX target form"):
        validate_a2v_lora_targets(H3_LORA_TARGET_REGEX)


def test_a_bare_lora_config_leaves_the_override_unset() -> None:
    """The family default is resolved on ``SignetConfig`` (where the family lives), not on the
    sub-model — so a standalone ``LoraConfig`` honestly reports "no override".
    """
    assert LoraConfig().target_modules is None


def test_h3_frame_law_rejects_an_ltx_frame_count() -> None:
    """25 is a VALID LTX count and an INVALID H3 count — the exact carry-forward trap."""
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(
            _h3_payload(
                training_dims=[1344, 768, 25],
                data={
                    "preprocessed_data_root": "/data/h3_preprocessed",
                    "batch_size": 1,
                    "resolution_buckets": ["1344x768x22"],
                },
            )
        )
    assert "17n+5" in str(exc.value)


def test_h3_resolution_buckets_obey_the_h3_frame_law() -> None:
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(
            _h3_payload(
                data={
                    "preprocessed_data_root": "/data/h3_preprocessed",
                    "batch_size": 1,
                    "resolution_buckets": ["1344x768x49"],  # an LTX bucket
                }
            )
        )
    assert "17n+5" in str(exc.value)


def test_h3_config_at_short_edge_896_loads() -> None:
    """Worst pair ``C+008`` prices at 12,394 rows — under the 13,777 ceiling, and within 0.3% of the
    12,362-row configuration measured PASSING on a real A100 at 76.36 GiB.
    """
    cfg = SignetConfig.model_validate(_h3_payload())
    assert cfg.h3.reference_image_short_edge == 896


def test_h3_config_at_short_edge_1024_is_refused_naming_the_worst_pair() -> None:
    """⛔ The named regression. A nominal-pair check would PASS here (``A+B`` = 12,362 rows) and then
    OOM on the first environment-bearing segment. The worst pair is ``C+008`` at 14,026.
    """
    payload = _h3_payload()
    payload["h3"]["reference_image_short_edge"] = 1024
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload)
    msg = str(exc.value)
    assert "14026" in msg, "the refusal must report the WORST-case row count"
    assert "C+008" in msg, "the refusal must NAME the offending reference pair"
    assert "13777" in msg, "the refusal must name the ceiling it was measured against"
    # Extract the REPORTED value positionally rather than by substring: the refusal legitimately
    # cites 12,362 as the measured-passing anchor in its remedy prose, so a naive `"12362" not in
    # msg` would fail for the wrong reason, and a stray "14026" in prose could satisfy a naive
    # `in` check. This pins that the validator's own computed answer is the WORST pair, not the
    # nominal one — which is the entire hole this check closes.
    assert _extract_reported_rows(msg) == "14026"


def _extract_reported_rows(message: str) -> str:
    m = re.search(r"packed sequence too long(?: for reference pair \S+)?: (\d+) rows exceeds", message)
    return m.group(1) if m else ""


def test_a_reference_free_h3_config_is_still_budget_checked() -> None:
    """The 124f no-reference t2v baseline is 37,806 rows against a 13,777 ceiling — so this is NOT a
    reference-cost problem, and an H3 config that declares no references must not skip the budget.
    """
    payload = _h3_payload(
        training_dims=[1344, 768, 124],
        data={
            "preprocessed_data_root": "/data/h3_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x124"],
        },
    )
    payload["h3"] = {}  # no references declared at all
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload)
    # Positional again: 37,806 also appears in every refusal's remedy prose as the t2v-baseline
    # advisory, so a substring check would pass even if the budget had never been computed here.
    assert _extract_reported_rows(str(exc.value)) == "37806"


def test_ltx_frame_law_is_still_enforced_for_the_ltx_family() -> None:
    """Widening the field-level pre-screen must not create a hole: an H3 frame count under the LTX
    family is still refused, with the LTX message.
    """
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(_ltx_payload(training_dims=[768, 512, 22]))
    msg = str(exc.value)
    assert "(frames - 1) % 8 == 0" in msg, "the LTX family must still get the LTX frame message"
    assert "17n+5" not in msg, "an LTX config must never be handed the H3 law"


def test_ltx_bucket_law_is_still_enforced_for_the_ltx_family() -> None:
    with pytest.raises(ValidationError):
        SignetConfig.model_validate(
            _ltx_payload(
                data={
                    "preprocessed_data_root": "/data/preprocessed",
                    "batch_size": 1,
                    "resolution_buckets": ["1344x768x22"],  # an H3 bucket
                }
            )
        )


def test_dims_invalid_under_both_families_still_die_at_the_field_level() -> None:
    """32 satisfies neither ``(F-1)%8`` nor ``17n+5`` — the pre-screen still rejects it outright."""
    with pytest.raises(ValidationError):
        SignetConfig.model_validate(_ltx_payload(training_dims=[768, 512, 32]))


# ── references_per_sample is 1 OR 2 — and 3 is still refused ─────────────────────────────────────
#
# The original ruling fixed the count at 2 because Phase 10's corpus is Ref2VA. It was never a
# claim that the architecture requires two, and a SINGLE-CONTROL task (one image in, one image
# out) is a legitimate shape the same stack already handles end to end.


def test_a_single_control_task_may_declare_one_reference_slot() -> None:
    assert H3Config(references_per_sample=1).references_per_sample == 1
    assert H3Config(references_per_sample=2).references_per_sample == 2


def test_three_reference_slots_are_still_refused_and_the_refusal_says_why() -> None:
    """The bound that carries real weight: three slots were never priced by the row budget."""
    with pytest.raises(ValidationError) as excinfo:
        H3Config(references_per_sample=3)
    message = str(excinfo.value)
    # An environment ref SUBSTITUTES for the second character slot rather than being appended,
    # so a 3-reference case does not exist to be priced in the first place.
    assert "SUBSTITUTES" in message
    assert "OOM" in message


def test_zero_reference_slots_are_alpha_no_reference() -> None:
    """SUPERSEDED premise (this test once refused 0): 0 is now NO-REFERENCE training (ALPHA,
    2026-08-11) — text-only is a supported workflow on the same packing, with the ref-only
    fields reverse-guarded. The bound that still carries weight is 3 (never priced)."""
    assert H3Config(references_per_sample=0).references_per_sample == 0
    with pytest.raises(ValidationError):
        H3Config(references_per_sample=3)
