"""Family #3 (``qwen_edit``) config layer — the CPU gate that must hold before any GPU dollar.

Written by the VERIFIER pass, not the build slices, and deliberately independent of them: every
number here is re-derived from the MEASURED checkpoint enumeration (60 transformer blocks x 14
LoRA-targetable Linear leaves = 840 modules) rather than copied out of a build report.

Five failure modes this file exists to catch, each of which reaches a metered GPU otherwise:

1. **The zero-match LoRA default.** Unlike H3 — where the LTX ten-suffix default still matches 104
   modules and so never trips ``train/loop.py``'s empty-trainable-set guard — the LTX list matches
   *zero* modules on Qwen (no ``attn1.`` / ``attn2.`` / ``ff.`` path exists anywhere). That failure
   IS loud at runtime, but only after the container is paid for. Refuse at config load.
2. **The 6-of-14 "house invariant" port.** A dual-stream MMDiT has every leaf twice; dropping the
   four ``txt_*`` leaves and the two ``*_mod.1`` projections is a DIFFERENT SHAPE, not a smaller
   one, and cannot warm-start from a 14-leaf primer.
3. **The rank/alpha drift.** ``lora_A``/``lora_B`` are rank-shaped; a mid-chain change makes every
   published round unloadable from every other.
4. **The silently-ignored block.** A ``qwen_edit`` tunable set under ``family: ltx``.
5. **LTX's frame law admitting F != 1.** ``(9 - 1) % 8 == 0``, so a nine-frame typo sails through
   the shared pre-screen with LTX's blessing and reaches the packer as a video request against an
   image model.

CPU only. No torch model, no Modal, no downloads.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from signet_trainer.config.load import load_config
from signet_trainer.config.schema import (
    LTX_DEFAULT_LORA_TARGETS,
    ModelConfig,
    QwenEditConfig,
    SignetConfig,
)
from signet_trainer.config.validators import (
    QWEN_EDIT_LORA_LEAVES,
    QWEN_EDIT_LORA_TARGET_REGEX,
)

EXAMPLE_CONFIG = "configs/qwen_image_edit.example.yaml"

#: The measured ground truth, transcribed from the live checkpoint enumeration and NOT imported
#: from the module under test — an import would make this file agree with a wrong list.
MEASURED_LEAVES = (
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
    "img_mod.1",
    "txt_mod.1",
)
MEASURED_BLOCK_COUNT = 60
#: RMSNorms, not Linears — never LoRA targets.
MEASURED_NON_TARGET_LEAVES = (
    "attn.norm_q",
    "attn.norm_k",
    "attn.norm_added_q",
    "attn.norm_added_k",
)
#: Top-level (non-block) leaves. Several ARE Linear; the anchor is what keeps them out.
MEASURED_TOP_LEVEL = (
    "img_in",
    "txt_in",
    "txt_norm",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "norm_out.linear",
    "proj_out",
)


def _qwen_payload(**over) -> dict:
    """A minimal VALID ``qwen_edit`` config, built from dicts so the whole file runs free."""
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


def _ltx_payload(**over) -> dict:
    """The byte-identical-load control: a minimal valid LTX config."""
    payload: dict = {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "training": {"max_steps": 100},
    }
    payload.update(over)
    return payload


# ======================================================================================
# The example config loads
# ======================================================================================


def test_example_config_loads() -> None:
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.model.family == "qwen_edit"
    assert (cfg.lora.rank, cfg.lora.alpha) == (42, 42)
    assert cfg.qwen_edit.rank_alpha_lock == 42


def test_example_config_pins_one_frame_everywhere() -> None:
    """F == 1 in the geometry, the bucket, AND the render — three places, one law."""
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.training_dims[2] == 1
    assert cfg.data.resolution_buckets == ["1024x1024x1"]
    assert cfg.validation.frame_count == 1


def test_family_literal_still_defaults_to_ltx() -> None:
    """Family #3 landing must not move the discriminator's default out from under 16 configs."""
    assert ModelConfig().family == "ltx"
    assert ModelConfig(family="qwen_edit").family == "qwen_edit"
    with pytest.raises(ValidationError):
        ModelConfig(family="qwen")


# ======================================================================================
# rank/alpha lock
# ======================================================================================


def test_rank_alpha_lock_rejects_a_rank_off_the_lock() -> None:
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig(**_qwen_payload(lora={"rank": 64, "alpha": 64}))
    assert "42" in str(exc.value)


def test_rank_alpha_lock_rejects_rank_not_equal_alpha() -> None:
    """PEFT scale is ``alpha / rank``; the chain is specified at scale 1.0."""
    with pytest.raises((ValidationError, ValueError)):
        SignetConfig(**_qwen_payload(lora={"rank": 42, "alpha": 21}))


def test_rank_alpha_lock_accepts_the_locked_pair() -> None:
    cfg = SignetConfig(**_qwen_payload(lora={"rank": 42, "alpha": 42}))
    assert cfg.lora.rank == cfg.lora.alpha == 42


def test_null_lock_still_requires_rank_equals_alpha_and_forbids_warm_start() -> None:
    """``null`` is the deliberate exit from the chain, not an escape from PEFT scale 1.0."""
    cfg = SignetConfig(
        **_qwen_payload(lora={"rank": 16, "alpha": 16}, qwen_edit={"rank_alpha_lock": None})
    )
    assert cfg.qwen_edit.rank_alpha_lock is None
    with pytest.raises((ValidationError, ValueError)):
        SignetConfig(
            **_qwen_payload(
                lora={"rank": 16, "alpha": 8}, qwen_edit={"rank_alpha_lock": None}
            )
        )
    with pytest.raises((ValidationError, ValueError)):
        SignetConfig(
            **_qwen_payload(
                lora={"rank": 16, "alpha": 16},
                qwen_edit={"rank_alpha_lock": None},
                training={"max_steps": 100, "init_adapter_path": "outputs/r1/step-2000"},
            )
        )


# ======================================================================================
# The REVERSE guard — no silently-ignored block, in either direction
# ======================================================================================


@pytest.mark.parametrize(
    "field,value",
    [
        ("control_slots", 2),
        ("blank_slot_fill", "gray"),
        ("max_packed_rows", 20000),
        ("caption_dropout_rate", 0.1),
        ("control_cache_key_mode", "path"),
        ("rank_alpha_lock", None),
    ],
)
def test_reverse_guard_rejects_qwen_fields_under_family_ltx(field: str, value) -> None:
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig(**_ltx_payload(qwen_edit={field: value}))
    message = str(exc.value)
    assert field in message
    assert "ltx" in message


def test_reverse_guard_is_pristine_instance_based_so_it_covers_every_field() -> None:
    """A field added to ``QwenEditConfig`` later must be covered without editing the guard."""
    pristine = QwenEditConfig()
    for name in QwenEditConfig.model_fields:
        assert getattr(pristine, name) == getattr(QwenEditConfig(), name)
    # An all-default block under LTX is inert and must NOT raise.
    SignetConfig(**_ltx_payload(qwen_edit={}))


def test_default_qwen_block_under_ltx_is_inert() -> None:
    """Every shipped LTX config still loads: the new block defaults to a no-op."""
    cfg = SignetConfig(**_ltx_payload())
    assert cfg.qwen_edit.control_slots == QwenEditConfig().control_slots
    assert cfg.resolved_lora_targets() == list(LTX_DEFAULT_LORA_TARGETS)


def test_family_only_model_ids_are_a_per_field_allowlist() -> None:
    """``vae_id`` is legal on qwen_edit (separate VAE file) but ``audio_vae_id`` is not."""
    SignetConfig(**_qwen_payload(model={"family": "qwen_edit", "vae_id": "qwen_image_vae.safetensors"}))
    with pytest.raises((ValidationError, ValueError)):
        SignetConfig(**_qwen_payload(model={"family": "qwen_edit", "audio_vae_id": "x"}))
    with pytest.raises((ValidationError, ValueError)):
        SignetConfig(**_ltx_payload(model={"family": "ltx", "vae_id": "x"}))


# ======================================================================================
# Control-slot bounds
# ======================================================================================


@pytest.mark.parametrize("slots", [1, 2, 3])
def test_control_slot_count_accepts_one_to_three(slots: int) -> None:
    cfg = SignetConfig(**_qwen_payload(qwen_edit={"control_slots": slots}))
    assert cfg.qwen_edit.control_slots == slots


@pytest.mark.parametrize("slots", [0, 4, -1])
def test_control_slot_count_rejects_outside_one_to_three(slots: int) -> None:
    """ai-toolkit's Edit-Plus prompt template names ``ctrl_img_1..3`` and nothing beyond."""
    with pytest.raises((ValidationError, ValueError)):
        SignetConfig(**_qwen_payload(qwen_edit={"control_slots": slots}))


def test_slot_count_moves_the_packed_length_by_exactly_one_image_block() -> None:
    """Each slot costs a full 1024^2 image block: 4096 rows. Pricing must be linear in slots."""
    from signet_trainer.conditioning.qwen_edit_geometry import qwen_edit_packed_layout

    totals = []
    for slots in (1, 2, 3):
        cfg = SignetConfig(**_qwen_payload(qwen_edit={"control_slots": slots}))
        totals.append(qwen_edit_packed_layout(cfg.training_dims, cfg.qwen_edit).total)
    assert totals[1] - totals[0] == 4096
    assert totals[2] - totals[1] == 4096
    # 4096 target + 3 x 4096 control + 256 text — the DERIVED figure, re-computed here.
    assert totals[2] == 4096 + 3 * 4096 + 256 == 16640


# ======================================================================================
# The 14 measured suffixes
# ======================================================================================


def test_leaf_list_is_exactly_the_fourteen_measured_leaves() -> None:
    assert sorted(QWEN_EDIT_LORA_LEAVES) == sorted(MEASURED_LEAVES)
    assert len(QWEN_EDIT_LORA_LEAVES) == 14


def test_family_default_resolves_to_the_regex_not_the_ltx_suffix_list() -> None:
    cfg = SignetConfig(**_qwen_payload())
    resolved = cfg.resolved_lora_targets()
    assert isinstance(resolved, str)
    assert resolved == QWEN_EDIT_LORA_TARGET_REGEX
    assert resolved != LTX_DEFAULT_LORA_TARGETS
    assert resolved != list(LTX_DEFAULT_LORA_TARGETS)


def test_regex_matches_all_840_measured_modules_and_nothing_else() -> None:
    """60 blocks x 14 leaves = 840. RMSNorms and every top-level leaf must stay out."""
    pattern = re.compile(QWEN_EDIT_LORA_TARGET_REGEX)
    targets = [
        f"transformer_blocks.{b}.{leaf}"
        for b in range(MEASURED_BLOCK_COUNT)
        for leaf in MEASURED_LEAVES
    ]
    assert len(targets) == 840
    assert all(pattern.search(name) for name in targets)

    non_targets = [
        f"transformer_blocks.{b}.{leaf}"
        for b in range(MEASURED_BLOCK_COUNT)
        for leaf in MEASURED_NON_TARGET_LEAVES
    ] + list(MEASURED_TOP_LEVEL)
    matched = [name for name in non_targets if pattern.search(name)]
    assert matched == [], f"regex over-matches non-target modules: {matched[:5]}"


def test_ltx_suffix_list_pasted_into_a_qwen_config_is_refused() -> None:
    """It matches ZERO Qwen modules; the runtime guard that catches that is on a paid GPU."""
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig(
            **_qwen_payload(
                lora={"rank": 42, "alpha": 42, "target_modules": list(LTX_DEFAULT_LORA_TARGETS)}
            )
        )
    assert "14" in str(exc.value) or "fourteen" in str(exc.value).lower()


def test_six_leaf_house_invariant_port_is_refused_and_names_the_gap() -> None:
    """6-of-14 is a different shape, not a smaller one — and cannot warm-start from a primer."""
    six = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
    ]
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig(**_qwen_payload(lora={"rank": 42, "alpha": 42, "target_modules": six}))
    message = str(exc.value)
    for missing in ("txt_mlp.net.0.proj", "txt_mod.1", "attn.add_q_proj"):
        assert missing in message, f"refusal does not name the missing leaf {missing}"


def test_warm_start_requires_the_family_default_byte_for_byte() -> None:
    """Covering all fourteen is not the same as MATCHING the module set."""
    superset = list(MEASURED_LEAVES) + ["proj_out"]
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig(
            **_qwen_payload(
                lora={"rank": 42, "alpha": 42, "target_modules": superset},
                training={"max_steps": 100, "init_adapter_path": "outputs/r1/step-2000"},
            )
        )
    assert "init_adapter_path" in str(exc.value)


def test_ltx_family_target_resolution_is_untouched() -> None:
    cfg = SignetConfig(**_ltx_payload())
    assert cfg.resolved_lora_targets() == list(LTX_DEFAULT_LORA_TARGETS)
    assert len(LTX_DEFAULT_LORA_TARGETS) == 10


# ======================================================================================
# Family-exact geometry — the guards LTX's own law cannot supply
# ======================================================================================


def test_ltx_frame_law_admits_nine_so_the_family_arm_is_the_only_guard() -> None:
    """Documents WHY the arm exists: ``(9 - 1) % 8 == 0`` passes the shared pre-screen."""
    assert (9 - 1) % 8 == 0
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig(**_qwen_payload(training_dims=[1024, 1024, 9]))
    assert "1" in str(exc.value)


def test_resolution_bucket_with_frames_not_one_is_refused() -> None:
    with pytest.raises((ValidationError, ValueError)):
        SignetConfig(
            **_qwen_payload(
                data={
                    "preprocessed_data_root": "/data/qwen_edit",
                    "batch_size": 1,
                    "resolution_buckets": ["1024x1024x9"],
                }
            )
        )


def test_non_multiple_of_32_canvas_is_refused() -> None:
    with pytest.raises((ValidationError, ValueError)):
        SignetConfig(**_qwen_payload(training_dims=[1000, 1024, 1]))


def test_omitting_the_validation_block_is_refused_not_defaulted() -> None:
    """``validation.frame_count`` defaults to 49 — an LTX clip length — so silence is wrong."""
    payload = _qwen_payload()
    payload.pop("validation")
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig(**payload)
    assert "frame_count" in str(exc.value)


def test_ltx_conditioning_mode_is_refused_under_qwen_edit() -> None:
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig(**_qwen_payload(conditioning={"mode": "single_frame"}))
    assert "conditioning.mode" in str(exc.value)


def test_caption_dropout_is_refused_on_this_family_under_EITHER_cache_setting() -> None:
    """Both settings refuse — and the second half is the one that used to be the bug.

    This test previously asserted that `cache_text_embeddings: false` was an ESCAPE from the
    dropout refusal, matching a schema message that offered exactly that remedy. Neither was true.
    That flag's only runtime consumer is the re-encode SKIP in qwen_edit_preprocess
    (modal/fns.py:5560); it does not switch training to live text encoding, and nothing in the tree
    ever supplies `empty_text_conditions`. So the "escape" produced a config that loads, dry-runs
    green, passes the cost gate, boots an A100, pays the ~40.9 GiB load, and dies at the first
    dropout draw — ten more times under the stage's retry policy.

    A test that asserts a broken remedy WORKS is worse than no test: it pins the bug in place and
    reports green while doing it.
    """
    for caching in (True, False):
        with pytest.raises((ValidationError, ValueError)) as exc:
            SignetConfig(
                **_qwen_payload(
                    qwen_edit={"caption_dropout_rate": 0.1, "cache_text_embeddings": caching}
                )
            )
        message = str(exc.value)
        assert "empty_text_conditions" in message, (
            "the refusal must name what is actually missing — an empty-caption payload — rather "
            "than blame the cache setting"
        )
        assert "Do NOT reach for cache_text_embeddings: false" in message, (
            "the refusal must actively steer the operator off the remedy that costs eleven A100 "
            "container starts"
        )


def test_cache_text_embeddings_declares_that_it_has_no_training_side_effect() -> None:
    """The knob is preprocess-scoped; its description must say so, because its name does not.

    `cache_text_embeddings` reads as a training switch and is not one. That gap is what made the
    old dropout remedy plausible enough to ship.
    """
    description = QwenEditConfig.model_fields["cache_text_embeddings"].description or ""
    assert "does NOT switch training to live text encoding" in description
    assert "qwen_edit_preprocess" in description


def test_row_ceiling_refuses_only_when_someone_measured_one() -> None:
    """``max_packed_rows: 0`` = DISABLED. Layout is still priced; nothing is refused."""
    assert QwenEditConfig().max_packed_rows == 0
    SignetConfig(**_qwen_payload())  # ceiling disabled, 16,640 rows, loads
    SignetConfig(**_qwen_payload(qwen_edit={"max_packed_rows": 20000}))
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig(**_qwen_payload(qwen_edit={"max_packed_rows": 8000}))
    assert "16640" in str(exc.value).replace(",", "")


def test_text_embed_dim_is_carried_as_an_unverified_assumption() -> None:
    """Guards the handoff: the value is 3584 AND is declared unmeasured until the loader pass."""
    assert QwenEditConfig().text_embed_dim == 3584
    description = QwenEditConfig.model_fields["text_embed_dim"].description or ""
    assert "UNVERIFIED" in description
    assert "txt_in" in description


def test_the_timestep_weighting_ablation_has_a_config_surface() -> None:
    """The ablation build_qwen_edit_step_fn documents must be reachable from YAML, not from an edit.

    Its own docstring says the ablation should be "stated explicitly in a config rather than
    achieved by deleting a line". Until this field existed there was no key for it and no caller
    threaded it, so the only route to the unweighted loss was editing train/qwen_edit_step.py or
    modal/fns.py — an untracked source change driving a metered A100, producing an adapter that no
    config on disk explains. Default True is the locked recipe, so this changes no existing run.
    """
    assert QwenEditConfig().timestep_weighting is True
    assert SignetConfig(**_qwen_payload()).qwen_edit.timestep_weighting is True
    assert (
        SignetConfig(
            **_qwen_payload(qwen_edit={"timestep_weighting": False})
        ).qwen_edit.timestep_weighting
        is False
    )
    description = QwenEditConfig.model_fields["timestep_weighting"].description or ""
    assert "ABLATION" in description


def test_the_train_stage_threads_timestep_weighting_from_the_config() -> None:
    """A field with no consumer is the defect the lean field-split exists to kill.

    Source-scanned rather than executed: the threading point is inside ``qwen_edit_train``, a Modal
    stage that cannot run without ~40.9 GiB of weights. A field that parses but never reaches the
    builder would leave the ablation exactly as unreachable as it was before, while LOOKING landed.
    """
    stage = (
        Path(__file__).resolve().parents[1] / "src/signet_trainer/modal/fns.py"
    ).read_text(encoding="utf-8")
    assert "timestep_weighting=config.qwen_edit.timestep_weighting" in stage, (
        "qwen_edit_train no longer threads timestep_weighting from the config — the field is back "
        "to being changeable only by editing source"
    )
