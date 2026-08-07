"""Phase 09.1 — the four infra-hardening tunables are documented, defaulted config fields.

Config-first / D-NOHARDCODE (AUDIT #5 / #1 / #11): every new timeout / margin / watchdog
threshold that the Wave-2 durability fixes read is a Pydantic ``Field`` with a documented
default on ``ModalConfig`` / ``TrainingConfig`` — NEVER a code literal. This suite is the
interface contract the Wave-2 consumer plans (09.1-03/04/05) build on.

Asserts, CPU-only (no modal / ltx-core / GPU import):
  * the four fields EXIST with the documented defaults on a freshly-composed
    ModalConfig() / TrainingConfig(max_steps=1);
  * a minimal SignetConfig that sets NONE of them loads and the defaults apply
    (additive fields — existing YAMLs are byte-identical);
  * the fields carry a ``description`` (documented, not a bare literal);
  * the gt=0.0 constraints fire (a non-positive value is rejected pre-approval);
  * setting checkpoint_expected_minutes leaves the other tunables at their defaults.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from signet_trainer.config.schema import ModalConfig, SignetConfig, TrainingConfig


def _minimal_payload() -> dict:
    # A minimal SignetConfig that sets NONE of the four new fields — the defaults must apply
    # and the config must still load (additive / backward-compat, matching the composing schema).
    return {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "training": {"max_steps": 100},
    }


# ---- field existence + documented defaults on the sub-configs ----


def test_modal_config_has_infra_tunables_with_defaults():
    m = ModalConfig()
    assert m.timeout_margin == pytest.approx(1.5)
    assert m.render_stall_minutes == pytest.approx(120.0)


def test_training_config_has_watchdog_tunables_with_defaults():
    t = TrainingConfig(max_steps=1)
    # None-default watchdog OFF (matches the keep_checkpoints None-default precedent).
    assert t.checkpoint_expected_minutes is None
    assert t.checkpoint_stall_multiplier == pytest.approx(2.5)


def test_new_fields_are_documented():
    # D-NOHARDCODE: each new tunable carries a Field(description=...) so it is config-visible,
    # not an anonymous literal.
    for name in ("timeout_margin", "render_stall_minutes"):
        assert ModalConfig.model_fields[name].description
    for name in ("checkpoint_expected_minutes", "checkpoint_stall_multiplier"):
        assert TrainingConfig.model_fields[name].description


# ---- additive: a config that sets none of them still loads with defaults applied ----


def test_minimal_config_loads_and_defaults_apply():
    cfg = SignetConfig.model_validate(_minimal_payload())
    assert cfg.modal.timeout_margin == pytest.approx(1.5)
    assert cfg.modal.render_stall_minutes == pytest.approx(120.0)
    assert cfg.training.checkpoint_expected_minutes is None
    assert cfg.training.checkpoint_stall_multiplier == pytest.approx(2.5)


# ---- gt=0.0 constraints fire pre-approval ----


@pytest.mark.parametrize("field", ["timeout_margin", "render_stall_minutes"])
def test_modal_tunables_reject_non_positive(field):
    with pytest.raises(ValidationError):
        ModalConfig(**{field: 0.0})


def test_checkpoint_stall_multiplier_rejects_non_positive():
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=1, checkpoint_stall_multiplier=0.0)


def test_checkpoint_expected_minutes_rejects_non_positive_when_set():
    # gt=0.0 applies only when the Optional field is set; None (default) stays legal.
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=1, checkpoint_expected_minutes=0.0)


# ---- setting the watchdog on leaves the other tunables untouched ----


def test_setting_checkpoint_expected_minutes_leaves_others_default():
    payload = _minimal_payload()
    payload["training"] = {"max_steps": 100, "checkpoint_expected_minutes": 30.0}
    cfg = SignetConfig.model_validate(payload)
    assert cfg.training.checkpoint_expected_minutes == pytest.approx(30.0)
    # Watchdog K default and the modal-side tunables are unaffected.
    assert cfg.training.checkpoint_stall_multiplier == pytest.approx(2.5)
    assert cfg.modal.timeout_margin == pytest.approx(1.5)
    assert cfg.modal.render_stall_minutes == pytest.approx(120.0)
