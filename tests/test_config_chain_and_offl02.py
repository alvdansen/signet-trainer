"""Phase 9 (09-02) — config surface for the two Wave-2 wiring seams (config-first, D-NOHARDCODE):

    training.init_adapter_path   (D-9-CHAINED)   — chained warm-restart source, Volume-relative
    training.keep_checkpoints    (D-9-RECIPE)    — bounded checkpoint retention (CheckpointManager keep_n)
    validation.in_loop_sampling  (D-9-OFFL02)    — OFFL-02 in-loop-sampling / decoder knob

Every new run knob is validated fail-fast AT CONFIG LOAD, before any GPU/approval (WR-04): a bad value
must die pre-approval, never after a burned metered dispatch. The chained-round path is Volume-escape-safe
(T-07-02 precedent). CPU only — no ltx_core / modal import, no filesystem touch.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from signet_trainer.config.schema import TrainingConfig, ValidationConfig


# --------------------------------------------------------------------------------------------------
# (a) defaults preserve current behavior — every new field is a no-op by default.
# --------------------------------------------------------------------------------------------------


def test_training_defaults_are_no_op():
    t = TrainingConfig(max_steps=1)
    assert t.init_adapter_path is None
    assert t.keep_checkpoints is None


def test_validation_default_in_loop_sampling_off():
    v = ValidationConfig()
    assert v.in_loop_sampling is False


# --------------------------------------------------------------------------------------------------
# (b) in_loop_sampling True + empty prompts raises; (c) True + non-empty prompts loads.
# --------------------------------------------------------------------------------------------------


def test_in_loop_sampling_empty_prompts_rejected():
    with pytest.raises(ValidationError):
        ValidationConfig(in_loop_sampling=True)  # prompts defaults to []


def test_in_loop_sampling_with_prompts_accepted():
    v = ValidationConfig(in_loop_sampling=True, prompts=["a photoreal driving clip"])
    assert v.in_loop_sampling is True
    assert v.prompts == ["a photoreal driving clip"]


def test_in_loop_sampling_off_with_empty_prompts_ok():
    # Default-off must still load with no prompts (the current-behavior baseline).
    v = ValidationConfig(in_loop_sampling=False)
    assert v.in_loop_sampling is False
    assert v.prompts == []


# --------------------------------------------------------------------------------------------------
# (d) init_adapter_path — absolute / '..' rejected, Volume-relative accepted (T-07-02).
# --------------------------------------------------------------------------------------------------


def test_init_adapter_path_default_none():
    assert TrainingConfig(max_steps=1).init_adapter_path is None


def test_init_adapter_path_relative_accepted():
    t = TrainingConfig(max_steps=1, init_adapter_path="campaign_r1/checkpoint-step-00200")
    assert t.init_adapter_path == "campaign_r1/checkpoint-step-00200"


def test_init_adapter_path_absolute_posix_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=1, init_adapter_path="/abs/escape")


def test_init_adapter_path_absolute_windows_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=1, init_adapter_path="C:/abs/escape")


def test_init_adapter_path_dotdot_escape_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=1, init_adapter_path="../escape")


def test_init_adapter_path_dotdot_windows_escape_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=1, init_adapter_path="..\\escape")


# --------------------------------------------------------------------------------------------------
# (e) keep_checkpoints — 25 accepted, 0 rejected (ge=1), None (unbounded) accepted.
# --------------------------------------------------------------------------------------------------


def test_keep_checkpoints_default_none():
    assert TrainingConfig(max_steps=1).keep_checkpoints is None


def test_keep_checkpoints_25_accepted():
    assert TrainingConfig(max_steps=1, keep_checkpoints=25).keep_checkpoints == 25


def test_keep_checkpoints_zero_rejected():
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=1, keep_checkpoints=0)
