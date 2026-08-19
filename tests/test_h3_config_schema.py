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
    ModalConfig,
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
    """The discriminator is an allowlist — a typo'd or aspirational family dies at load.

    ⚠ AMENDED, DELIBERATELY (multi-source slice A). The example used to be ``family="wan"``, and
    ``tests/test_multisource_verifier_gaps.py`` armed a tripwire so that changing it could not be
    accidental. ``"wan"`` is now a REAL family — the musubi-tuner runner, the only one that consumes
    ``data.sources`` — with its own dims law (``validators.validate_wan_training_dims``) and its own
    arm in ``SignetConfig._cross_field_checks``, so it is no longer an example of anything rejected.

    The CLAIM this test makes is unchanged and still covered: the field is an ALLOWLIST, not a free
    string. ``"svd"`` replaces ``"wan"`` as the aspirational-family example, and
    ``test_model_config_accepts_wan_family`` below pins the positive half so the allowlist's real
    membership is asserted rather than implied.
    """
    with pytest.raises(ValidationError):
        ModelConfig(family="svd")
    with pytest.raises(ValidationError):
        ModelConfig(family="LTX")  # case matters — the literal is exact


def test_model_config_accepts_wan_family() -> None:
    """Family #4: the musubi-tuner RUNNER family (multi-source slice A)."""
    assert ModelConfig(family="wan").family == "wan"


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
    assert h3.modal_gpu == "A100-80GB"  # the booked card the budget triple was measured ON
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


def test_references_per_sample_accepts_three_for_explicit_manifest_tasks() -> None:
    """3 is permitted for sequence tasks whose rows name their own ``reference_paths``.

    Was ``test_references_per_sample_is_pinned_at_two``. The pin was lifted deliberately: the
    refusal rested on the POOL/rotation branch, which a row carrying ``reference_paths`` never
    enters, and on three slots being unpriced, which they no longer are (13,402 packed rows at
    short_edge 768 against a computed 13,777 ceiling).
    """
    assert H3Config(references_per_sample=3).references_per_sample == 3


def test_references_per_sample_still_refuses_four() -> None:
    """The bound moved; it did not disappear. 4 slots are unpriced at any usable reference size."""
    with pytest.raises(ValidationError) as exc:
        H3Config(references_per_sample=4)
    msg = str(exc.value)
    assert "substitut" in msg.lower(), "the refusal must still name the substitution rule"
    assert "priced" in msg.lower(), "the refusal must still name the budget as the binding reason"


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
        ("modal_gpu", "H200"),
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

    ``references_per_sample: 0`` (explicit, not the bare ``{}`` this test used pre-#31) is what
    makes "no references declared" a VALID reference-free config rather than the mirror-direction
    refusal (#31 finding 1): the block's default ``references_per_sample`` is 2 (Ref2VA), so an
    empty ``h3: {}`` with no size lists is now the exact un-priced-corpus shape that guard exists
    to reject, not a reference-free t2v declaration.
    """
    payload = _h3_payload(
        training_dims=[1344, 768, 124],
        data={
            "preprocessed_data_root": "/data/h3_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x124"],
        },
    )
    payload["h3"] = {"references_per_sample": 0}  # no references declared at all, EXPLICITLY
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload)
    # Positional again: 37,806 also appears in every refusal's remedy prose as the t2v-baseline
    # advisory, so a substring check would pass even if the budget had never been computed here.
    assert _extract_reported_rows(str(exc.value)) == "37806"


# ======================================================================================
# The GPU/budget/rate COHERENCE guards — the escalation lever must move the HARDWARE and the
# COST BASIS, not just the safety rail (audit finding config-coherence-0, bundle PR-5 rework).
# ======================================================================================


def test_a_raised_budget_on_the_default_a100_booking_is_refused() -> None:
    """⛔ NAMED REGRESSION: ``gpu_usable_gib`` was documented as THE H200-escalation lever while
    every H3 stage booted an A100-80GB regardless. Editing the budget alone loaded clean, widened
    the packed-row ceiling from 13,777 to 64,342 rows, and thereby switched OFF the only local OOM
    refusal — the container then paid the 61.7 GiB load and CUDA-OOM'd. The escalation is honest
    only as BOTH edits together, so the triple edited alone must die at config load.
    """
    payload = _h3_payload()
    payload["h3"]["gpu_usable_gib"] = 139.0  # H200-class usable VRAM, A100-80GB still booked
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload)
    msg = str(exc.value)
    assert "modal_gpu" in msg, "the refusal must NAME the missing half of the escalation"
    assert "A100-80GB" in msg, "the refusal must NAME the card actually booked"
    assert "139.0" in msg and "79.25" in msg, "both numbers must be named (typed state)"
    assert "hourly_rate_usd" in msg, "the remedy must keep the cost print honest too"


def test_the_audited_scenario_a_widened_ceiling_never_passes_the_37806_row_config() -> None:
    """The exact audited failure: training_dims [1344, 768, 124] (37,806 packed rows) with
    ``gpu_usable_gib: 139.0`` used to load clean and dry-run green — 37,806 < the widened 64,342
    ceiling — then OOM on the booked A100. The coherence guard must fire FIRST, before the budget
    prices anything against a ceiling the booked card cannot honor.
    """
    payload = _h3_payload(
        training_dims=[1344, 768, 124],
        data={
            "preprocessed_data_root": "/data/h3_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x124"],
        },
    )
    payload["h3"] = {"references_per_sample": 0, "gpu_usable_gib": 139.0}  # no-ref t2v baseline
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload)
    assert "modal_gpu" in str(exc.value), (
        "the refusal must be the COHERENCE guard (booking vs budget), not a budget pass — a "
        "widened ceiling passing this config is exactly the audited OOM-in-a-paid-container path"
    )


def test_an_escalated_booking_accepts_the_matching_budget_and_rate() -> None:
    """The honest escalation — ``modal_gpu``, the triple, AND ``hourly_rate_usd`` together — must
    load. 139.0 usable on an H200 clears the 37,806-row campaign geometry (P10-1-MEASURED section
    4 extrapolation).
    """
    payload = _h3_payload(
        training_dims=[1344, 768, 124],
        data={
            "preprocessed_data_root": "/data/h3_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x124"],
        },
    )
    payload["h3"] = {
        "references_per_sample": 0,
        "modal_gpu": "H200",
        "gpu_usable_gib": 139.0,
    }
    payload["modal"] = {"hourly_rate_usd": 3.5}  # a re-measured H200 rate, NOT the A100 default
    cfg = SignetConfig.model_validate(payload)
    assert cfg.h3.modal_gpu == "H200"
    assert cfg.h3.gpu_usable_gib == pytest.approx(139.0)
    assert cfg.modal.hourly_rate_usd == pytest.approx(3.5)


def test_a_lowered_budget_on_the_default_booking_still_loads() -> None:
    """Coherence bounds the budget ABOVE only: declaring LESS usable VRAM than measured is a
    conservative operator choice (a tighter ceiling), never a lie about the hardware.
    78.0 GiB keeps the ceiling above the worst declared pair in the default h3 payload, so the
    budget check itself still passes — this isolates the coherence guard's direction.
    """
    payload = _h3_payload()
    payload["h3"]["gpu_usable_gib"] = 78.0
    cfg = SignetConfig.model_validate(payload)
    assert cfg.h3.gpu_usable_gib == pytest.approx(78.0)


def test_modal_gpu_default_matches_the_measured_budget_card() -> None:
    """The field default and the constant the coherence guards compare against are ONE fact — the
    card the P10-1b triple was measured on.
    """
    from signet_trainer.config.validators import H3_DEFAULT_MODAL_GPU

    assert H3Config().modal_gpu == H3_DEFAULT_MODAL_GPU == "A100-80GB"


def test_the_budget_coherence_guard_is_importable_and_returns_the_value_unchanged() -> None:
    """The Pydantic-validator shape contract, same as every other validator in the module."""
    from signet_trainer.config.validators import validate_h3_gpu_budget_coherence

    assert validate_h3_gpu_budget_coherence("A100-80GB", 79.25) == pytest.approx(79.25)
    assert validate_h3_gpu_budget_coherence("H200", 139.0) == pytest.approx(139.0)
    with pytest.raises(ValueError):
        validate_h3_gpu_budget_coherence("A100-80GB", 79.26)


# --- must-fix 1: normalization (Modal uppercases the GPU string before booking) ------------------


@pytest.mark.parametrize("spelling", ["a100-80gb", " A100-80GB ", "A100-80gb", "\tA100-80GB\n"])
def test_a_differently_spelled_default_booking_is_still_recognized_as_the_default(
    spelling: str,
) -> None:
    """⛔ NAMED REGRESSION (must-fix 1): Modal uppercases the GPU string itself before booking, so
    a lowercase/whitespace-padded spelling of the default is the SAME booking to Modal but used to
    be a DIFFERENT string to a bare ``==`` coherence check — reopening the audited escalation trap
    just by spelling. ``modal_gpu`` must be stored canonicalized, so this raises exactly like the
    canonical spelling does.
    """
    payload = _h3_payload()
    payload["h3"]["modal_gpu"] = spelling
    payload["h3"]["gpu_usable_gib"] = 139.0
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload)
    assert "modal_gpu" in str(exc.value)


def test_modal_gpu_is_stored_canonicalized() -> None:
    """The STORED value is canonical (strip + upper), not the operator's raw spelling — every
    downstream reader (coherence guards, entrypoint dispatch) then compares what Modal actually
    books.
    """
    assert H3Config(modal_gpu=" h200 ").modal_gpu == "H200"
    assert H3Config(modal_gpu="a100-80gb").modal_gpu == "A100-80GB"


# --- must-fix 2: single-GPU house rule (no ':<count>' suffix / ',' fallback list) -----------------


@pytest.mark.parametrize(
    "modal_gpu",
    ["A100-80GB:2", "A100-80GB:8", "H100,A100-80GB", "H200,H100"],
)
def test_multi_gpu_and_fallback_list_spellings_are_refused(modal_gpu: str) -> None:
    """⛔ NAMED REGRESSION (must-fix 2): Modal reads a ':<count>' suffix as a GPU COUNT (N-fold
    real spend against a cost print that assumes one card) and a ',' -separated string as a
    FALLBACK LIST (the booked card becomes non-deterministic at dispatch time). Both must be
    refused at config load, naming the single-GPU house rule — this hole did not exist before the
    ``modal_gpu`` field did.
    """
    with pytest.raises(ValidationError) as exc:
        H3Config(modal_gpu=modal_gpu)
    msg = str(exc.value)
    assert "single-GPU" in msg or "single GPU" in msg


# --- must-fix 3: cost basis coupled to the booking (mirrors WR-04's cpu_hourly_rate_usd split) ----


def test_an_escalated_booking_with_the_untouched_a100_rate_is_refused() -> None:
    """⛔ NAMED REGRESSION (must-fix 3): ``modal.hourly_rate_usd`` prices the pre-approval cost
    print AND the ``$50`` guardrail. Escalating ``h3.modal_gpu`` while leaving the rate at its
    A100-80GB default prints A100 money for hardware that is not an A100.
    """
    payload = _h3_payload()
    payload["h3"]["modal_gpu"] = "H200"
    # modal.hourly_rate_usd intentionally left at the schema default — the untouched A100 rate.
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload)
    msg = str(exc.value)
    assert "hourly_rate_usd" in msg
    assert "H200" in msg


def test_the_default_booking_never_trips_the_rate_guard_regardless_of_rate() -> None:
    """The rate guard only fires on an ESCALATED booking — the default A100-80GB booking is exempt
    even if the operator has, for unrelated reasons, changed ``hourly_rate_usd``.
    """
    payload = _h3_payload()
    payload["modal"] = {"hourly_rate_usd": 1.64}  # byte-identical to the untouched default
    cfg = SignetConfig.model_validate(payload)
    assert cfg.h3.modal_gpu == "A100-80GB"
    payload["modal"] = {"hourly_rate_usd": 9.99}  # any rate at all, unrelated to h3
    cfg = SignetConfig.model_validate(payload)
    assert cfg.modal.hourly_rate_usd == pytest.approx(9.99)


def test_the_rate_coherence_guard_reads_its_default_off_modalconfig_not_a_second_literal() -> None:
    """D-NOHARDCODE: the guard's notion of 'the untouched A100 rate' must be ONE fact, read off
    ``ModalConfig``'s own field default — never a second hardcoded copy that could drift.
    """
    from signet_trainer.config.validators import validate_h3_gpu_rate_coherence

    default_rate = ModalConfig.model_fields["hourly_rate_usd"].default
    assert validate_h3_gpu_rate_coherence("A100-80GB", default_rate, default_rate) == pytest.approx(
        default_rate
    )
    assert validate_h3_gpu_rate_coherence("H200", 3.5, default_rate) == pytest.approx(3.5)
    with pytest.raises(ValueError):
        validate_h3_gpu_rate_coherence("H200", default_rate, default_rate)


# --- must-fix 4: shape validation beyond min_length=1, refused PRE-APPROVAL at config load --------


@pytest.mark.parametrize("modal_gpu", ["NOTAGPU", " ", "\t", "RTX4090"])
def test_an_unrecognized_or_blank_modal_gpu_is_refused_at_config_load(modal_gpu: str) -> None:
    """⛔ NAMED REGRESSION (must-fix 4): a typo'd or blank ``modal_gpu`` used to load clean and
    fail only inside the metered ``.spawn()`` call — AFTER the operator's approval, the exact
    burned-gate class this entrypoint guards against elsewhere. It must die at config load
    instead, before the cost print.
    """
    with pytest.raises(ValidationError):
        H3Config(modal_gpu=modal_gpu)


@pytest.mark.parametrize("modal_gpu", ["A100-80GB", "A100-40GB", "H100", "H200", "B200", "L40S"])
def test_every_known_modal_gpu_type_loads_clean(modal_gpu: str) -> None:
    """The allowlist must not be so narrow it blocks legitimate Modal GPU types."""
    assert H3Config(modal_gpu=modal_gpu).modal_gpu == modal_gpu


def test_normalize_h3_modal_gpu_is_importable_and_returns_the_canonical_value() -> None:
    """The Pydantic-validator shape contract, same as every other validator in the module."""
    from signet_trainer.config.validators import normalize_h3_modal_gpu

    assert normalize_h3_modal_gpu("a100-80gb") == "A100-80GB"
    assert normalize_h3_modal_gpu(" H200 ") == "H200"
    with pytest.raises(ValueError):
        normalize_h3_modal_gpu("")
    with pytest.raises(ValueError):
        normalize_h3_modal_gpu("A100-80GB:2")
    with pytest.raises(ValueError):
        normalize_h3_modal_gpu("NOTAGPU")


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


def test_four_reference_slots_are_refused_and_the_refusal_says_why() -> None:
    """SUPERSEDED premise (this test once refused 3, as ``..._three_reference_slots_...``).

    3 is now permitted for explicit-manifest sequence tasks: the refusal rested on the POOL /
    rotation branch, which a row carrying ``reference_paths`` never enters, and on three slots
    being unpriced, which they no longer are. The bound moved to 4 and still carries weight —
    4 slots are unpriced at any usable reference size.
    """
    with pytest.raises(ValidationError) as excinfo:
        H3Config(references_per_sample=4)
    message = str(excinfo.value)
    # An environment ref SUBSTITUTES for the last character slot rather than being appended.
    assert "SUBSTITUTES" in message
    assert "OOM" in message


def test_zero_reference_slots_are_alpha_no_reference() -> None:
    """SUPERSEDED premise (this test once refused 0): 0 is now NO-REFERENCE training (ALPHA,
    2026-08-11) — text-only is a supported workflow on the same packing, with the ref-only
    fields reverse-guarded. The bound that still carries weight is 4 (never priced)."""
    assert H3Config(references_per_sample=0).references_per_sample == 0
    with pytest.raises(ValidationError):
        H3Config(references_per_sample=4)


# ======================================================================================
# Issue #13 step 2 — training.timestep_std. Two 2026-08-11 operator rulings forbid changing the
# shipped H3 sampler; the ONLY sanctioned action is exposing the knob with a byte-identical
# default, sharing the field across families (H3-only threading was refuted as a new
# validated-but-ignored knob on LTX — the #20 defect class), and refusing it under qwen_edit,
# whose step path never consumes a std-bearing schedule at all.
# ======================================================================================


def _qwen_edit_payload(**over) -> dict:
    """A minimal VALID ``qwen_edit`` config — mirrors ``test_qwen_edit_config.py::_qwen_payload``."""
    payload: dict = {
        "training_dims": [1024, 1024, 1],
        "data": {
            "preprocessed_data_root": "/data/qwen_edit",
            "batch_size": 1,
            "resolution_buckets": ["1024x1024x1"],
        },
        "model": {"family": "qwen_edit"},
        "training": {"max_steps": 100},
        "lora": {"rank": 42, "alpha": 42},
        "validation": {"frame_count": 1},
        "conditioning": {"mode": "none"},
    }
    payload.update(over)
    return payload


def test_timestep_std_defaults_to_one_under_every_family() -> None:
    """default=1.0 == the pre-#13 hardcoded ``h3_draw_timesteps``/``FlowMatchingSchedule`` default
    — every existing config (LTX or H3) must load and train byte-identically."""
    from signet_trainer.config.schema import TrainingConfig

    assert TrainingConfig(max_steps=100).timestep_std == 1.0
    assert SignetConfig.model_validate(_ltx_payload()).training.timestep_std == 1.0
    assert SignetConfig.model_validate(_h3_payload()).training.timestep_std == 1.0


def test_timestep_std_is_documented() -> None:
    """A tunable with no description is an undocumented hardcode wearing a config field's clothes
    (the same doctrine ``test_every_h3_field_is_documented`` holds the h3 block to)."""
    from signet_trainer.config.schema import TrainingConfig

    assert TrainingConfig.model_fields["timestep_std"].description


def test_timestep_std_must_be_positive() -> None:
    """``std <= 0`` is not a shape a logit-normal draw can take — reject at config load."""
    with pytest.raises(ValidationError):
        SignetConfig.model_validate(_h3_payload(training={"max_steps": 100, "timestep_std": 0.0}))


def test_nondefault_timestep_std_is_accepted_under_h3_and_ltx() -> None:
    """The derived A/B value from issue #13 step 3 must load under both std-consuming families."""
    cfg_h3 = SignetConfig.model_validate(
        _h3_payload(training={"max_steps": 100, "timestep_std": 1.7})
    )
    assert cfg_h3.training.timestep_std == pytest.approx(1.7)
    cfg_ltx = SignetConfig.model_validate(
        _ltx_payload(training={"max_steps": 100, "timestep_std": 1.7})
    )
    assert cfg_ltx.training.timestep_std == pytest.approx(1.7)


def test_nondefault_timestep_std_under_qwen_edit_is_rejected_not_silently_ignored() -> None:
    """``qwen_edit_step.py`` draws from ai-toolkit's discrete uniform grid and never reads
    ``FlowMatchingSchedule``'s ``std`` (the module's own DIVERGE note) — a non-default value under
    this family would train a run that believes it set a knob that did nothing."""
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(
            _qwen_edit_payload(training={"max_steps": 100, "timestep_std": 1.7})
        )
    msg = str(exc.value)
    assert "timestep_std" in msg
    assert "qwen_edit" in msg


def test_default_timestep_std_loads_fine_under_qwen_edit() -> None:
    """The reverse guard must fire on non-DEFAULT values only (T-10-05-T doctrine) — an untouched
    field is invisible, same shape as ``test_all_default_h3_block_loads_fine_under_the_ltx_family``."""
    cfg = SignetConfig.model_validate(_qwen_edit_payload())
    assert cfg.model.family == "qwen_edit"
    assert cfg.training.timestep_std == 1.0


# ======================================================================================
# #31 finding 1 / #39 finding 1 — the H3 schema guards this bundle adds
# ======================================================================================
#
# #31 finding 1 (mirror direction): H3Config._check_no_reference_fields only ever guarded the
# references_per_sample == 0 direction. A 1- or 2-slot config with BOTH size lists left empty
# passed load clean, and the SignetConfig cross-field pricing arm keys on the size lists (not the
# slot count) — so it took the "no reference corpus declared" branch and certified the layout as
# if references_per_sample were 0, while the metered container still resolves real slots per
# sample. This guard is asserted at the SignetConfig level (never on a bare H3Config()) because
# H3Config's OWN default combination — references_per_sample=2, both size lists empty — is exactly
# this shape, and that combination must stay legal on an LTX/qwen_edit config's inert, all-default
# h3 block (test_all_default_h3_block_loads_fine_under_the_ltx_family above). A sub-model cannot
# see model.family, so the mirror guard lives where the family is actually known to be h3.


@pytest.mark.parametrize("slots", [1, 2])
def test_mirror_direction_refuses_declared_slots_with_no_reference_corpus(slots: int) -> None:
    payload = _h3_payload(h3={"references_per_sample": slots})
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload)
    msg = str(exc.value)
    assert f"references_per_sample is {slots}" in msg
    assert "neither" in msg and "declared" in msg, (
        "the refusal must name that NEITHER size list was declared, not just that the config failed"
    )


def test_mirror_direction_does_not_fire_on_the_all_default_h3_block_under_ltx() -> None:
    """The negative control for the guard above: family stays ltx, so the all-default h3 block
    (references_per_sample=2, empty size lists) must NOT trip the mirror guard."""
    cfg = SignetConfig.model_validate(_ltx_payload())
    assert cfg.model.family == "ltx"
    assert cfg.h3.references_per_sample == 2
    assert cfg.h3.character_reference_sizes == []


def test_single_control_one_slot_needs_only_a_character_reference() -> None:
    """A 1-slot config declaring ONLY character_reference_sizes is a legitimate single-control
    task and must load — the mirror guard only refuses BOTH lists empty, never one of them."""
    payload = _h3_payload(
        h3={
            "references_per_sample": 1,
            "character_reference_sizes": [CHARACTER_REFS[0]],
            "environment_reference_sizes": [],
        }
    )
    cfg = SignetConfig.model_validate(payload)
    assert cfg.h3.references_per_sample == 1


# #39 finding 1: an environment reference SUBSTITUTES for the LAST character slot (h3_ref.py's own
# rule), so it needs >= 2 total slots to have a character slot to substitute for.
# `resolve_reference_slots` / `H3RefStrategy._resolve_slots` both refuse any environment-bearing
# sample at 1 slot by construction; this guard closes the matching config-load hole so the priced
# domain (h3_reference_pairing_domain) can never be asked about a pair the runtime cannot produce.


def test_environment_references_below_two_slots_are_refused() -> None:
    with pytest.raises(ValidationError) as exc:
        H3Config(references_per_sample=1, environment_reference_sizes=[(1344, 768)])
    msg = str(exc.value)
    assert "invalid references_per_sample 1" in msg
    assert "SUBSTITUTES" in msg and "at least 2" in msg, (
        "the message must reuse resolve_reference_slots' own wording (conditioning/h3_ref.py) so "
        "the local and runtime refusals read identically"
    )


def test_environment_references_at_two_slots_are_fine() -> None:
    h3 = H3Config(
        references_per_sample=2,
        character_reference_sizes=list(CHARACTER_REFS),
        environment_reference_sizes=[(1344, 768)],
    )
    assert h3.references_per_sample == 2
    assert len(h3.environment_reference_sizes) == 1


# ======================================================================================
# FRAME-COUNT BUCKETING — the three back-compat guards ``_cross_field_checks`` enforces
# once ``data.resolution_buckets`` becomes LOAD-BEARING on the H3 path.
# ======================================================================================


def test_a_valid_multi_bucket_h3_config_loads() -> None:
    """The positive control: two buckets, same canvas, training_dims F declared among them."""
    cfg = SignetConfig.model_validate(
        _h3_payload(
            data={
                "preprocessed_data_root": "/data/h3_preprocessed",
                "batch_size": 1,
                "resolution_buckets": ["1344x768x22", "1344x768x5"],
            },
        )
    )
    assert cfg.data.resolution_buckets == ["1344x768x22", "1344x768x5"]


def test_a_bucket_declaring_a_different_canvas_is_refused() -> None:
    """Aspect bucketing is out of scope (issue #1) — every bucket must assert the RUN's own canvas."""
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(
            _h3_payload(
                data={
                    "preprocessed_data_root": "/data/h3_preprocessed",
                    "batch_size": 1,
                    # 1152x768 is a legal H3 canvas (a different aspect), not h3.target_aspect's
                    # 16:9 -> 1344x768.
                    "resolution_buckets": ["1344x768x22", "1152x768x5"],
                },
            )
        )
    msg = str(exc.value)
    assert "1152x768" in msg
    assert "1344x768" in msg


def test_a_single_bucket_disagreeing_with_training_dims_f_is_refused() -> None:
    """Back-compat: these strings used to be decorative. One declared bucket must now AGREE."""
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(
            _h3_payload(
                training_dims=[1344, 768, 22],
                data={
                    "preprocessed_data_root": "/data/h3_preprocessed",
                    "batch_size": 1,
                    "resolution_buckets": ["1344x768x5"],
                },
            )
        )
    msg = str(exc.value)
    assert "F=5" in msg
    assert "training_dims F=22" in msg


def test_training_dims_f_must_be_among_the_declared_buckets() -> None:
    """Under >1 bucket, training_dims F is the SINGLE-bucket default — it must still be legal."""
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(
            _h3_payload(
                # 39 is a VALID H3 count (17*2+5) and satisfies the frame law, so this exercises
                # the bucket-membership guard specifically rather than the frame-law pre-screen.
                training_dims=[1344, 768, 39],
                data={
                    "preprocessed_data_root": "/data/h3_preprocessed",
                    "batch_size": 1,
                    "resolution_buckets": ["1344x768x22", "1344x768x5"],
                },
            )
        )
    msg = str(exc.value)
    assert "training_dims F=39" in msg
    assert "[5, 22]" in msg


def test_multi_bucket_budget_prices_the_largest_declared_bucket() -> None:
    """VRAM peak is set by the LONGEST target any sample may carry, not the default bucket.

    A short default bucket (F=5) alongside a long one (F=124) must still be priced at 124 — a
    reference-free config makes this unambiguous because there is exactly one layout per frame
    count, so the refusal's row count pins WHICH bucket the gate actually priced.
    """
    payload = _h3_payload(
        training_dims=[1344, 768, 5],
        data={
            "preprocessed_data_root": "/data/h3_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x5", "1344x768x124"],
        },
    )
    # #31 finding 1's mirror-direction guard (already merged) refuses references_per_sample != 0
    # with BOTH size lists empty — declare 0 explicitly (NO-REFERENCE) rather than relying on the
    # h3 block's all-default references_per_sample=2, so this config exercises ONLY the bucketing
    # budget path under test, not the unrelated reference-corpus guard. h3_packed_seq_len's
    # reference-free branch below takes an empty tuple either way, so the priced row count (and
    # this test's point — the largest bucket is priced) is unchanged.
    payload["h3"] = {"references_per_sample": 0}
    with pytest.raises(ValidationError) as exc:
        SignetConfig.model_validate(payload)
    msg = str(exc.value)
    assert "37806" in msg, "the refusal must price the F=124 bucket, not the F=5 default"
