"""CONF-01 — YAML loads into a Pydantic SignetConfig that COMPOSES (not subclasses).

signet carries LTX fields as plain DATA + signet blocks (modal, conditioning, dry_run,
validation, training_dims). It does NOT instantiate the native LtxTrainerConfig locally
(its model_path / images / reference_videos validators call Path(...).exists() at load
time — RESEARCH.md Pitfall 1 — which breaks the Windows/zero-GPU dry-run).

This suite asserts:
  * a valid YAML loads via Pydantic model_validate (no hand-rolled setattr parser);
  * SignetConfig exposes the reserved signet blocks;
  * LTX fields are carried as data;
  * local load NEVER touches the filesystem (no Path.exists call) and NEVER imports
    modal / ltx_core.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from signet_trainer.config.load import load_config
from signet_trainer.config.schema import SignetConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_YAML = REPO_ROOT / "configs" / "ltx23_lora.example.yaml"
BAD_YAML = REPO_ROOT / "configs" / "bad_frames.example.yaml"


def _valid_payload() -> dict:
    return {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "lora": {"rank": 64, "alpha": 64},
        # training.max_steps is the one required run-specific knob (Plan 03-01 / D-RUN-1).
        "training": {"max_steps": 100},
        "modal": {"hourly_rate_usd": 1.64, "cost_guardrail_usd": 50.0, "est_hours": 2.0},
        "validation": {"interval": 250, "prompts": ["a cat"], "frame_count": 49},
    }


def test_loads_via_pydantic_model_validate():
    cfg = SignetConfig.model_validate(_valid_payload())
    assert isinstance(cfg, SignetConfig)


def test_exposes_reserved_signet_blocks():
    cfg = SignetConfig.model_validate(_valid_payload())
    # The reserved blocks must exist from day one (D-14 reserves validation/sampling).
    assert cfg.modal is not None
    assert cfg.conditioning is not None
    assert cfg.dry_run is not None
    assert cfg.validation is not None
    # Conditioning mode defaults to standard/"none" (reserved for Phases 5-7).
    assert cfg.conditioning.mode == "none"


def test_lora_block_carries_native_defaults():
    cfg = SignetConfig.model_validate(_valid_payload())
    assert cfg.lora.rank == 64
    assert cfg.lora.alpha == 64
    # target_modules use full module paths (attn1/attn2 + the ff.net scaffold-bug fix, Plan 03-01).
    assert "attn1.to_q" in cfg.lora.target_modules
    assert "ff.net.0.proj" in cfg.lora.target_modules


def test_modal_block_carries_cost_fields():
    cfg = SignetConfig.model_validate(_valid_payload())
    assert cfg.modal.hourly_rate_usd == pytest.approx(1.64)
    assert cfg.modal.cost_guardrail_usd == pytest.approx(50.0)


def test_modal_block_carries_configdriven_secret_names_with_account_defaults():
    # Secret NAMES are config-driven (each account uses its own). When omitted from YAML they
    # default to the maintainer account's actual secret names so the live proof unblocks; the
    # app graph (app.py) reads the same defaults via env overrides at module-import time.
    cfg = SignetConfig.model_validate(_valid_payload())
    assert cfg.modal.huggingface_secret_name == "my-huggingface-secret"
    assert cfg.modal.wandb_secret_name == "my-wandb-secret"


def test_modal_secret_names_are_overridable():
    payload = _valid_payload()
    payload["modal"] = {
        **payload["modal"],
        "huggingface_secret_name": "team-hf",
        "wandb_secret_name": "team-wandb",
    }
    cfg = SignetConfig.model_validate(payload)
    assert cfg.modal.huggingface_secret_name == "team-hf"
    assert cfg.modal.wandb_secret_name == "team-wandb"


def test_validation_block_reserves_sampling_fields():
    cfg = SignetConfig.model_validate(_valid_payload())
    # D-14: reserve interval, prompts, frame_count, guidance_scale, num_samples, seed.
    v = cfg.validation
    for field in ("interval", "prompts", "frame_count", "guidance_scale", "num_samples", "seed"):
        assert hasattr(v, field), f"validation block missing reserved field {field!r}"


def test_local_load_does_not_touch_filesystem():
    # Pitfall 1: composing-not-subclassing means we never call Path.exists during load.
    with mock.patch.object(Path, "exists", side_effect=AssertionError("Path.exists called during local load")):
        cfg = SignetConfig.model_validate(_valid_payload())
        assert cfg is not None


def test_load_config_reads_valid_example_yaml():
    cfg = load_config(VALID_YAML)
    assert isinstance(cfg, SignetConfig)
    assert tuple(cfg.training_dims) == (768, 512, 49)


def test_load_config_rejects_bad_frames_yaml():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="frame"):
        load_config(BAD_YAML)


def test_load_does_not_import_modal_or_ltx_core():
    # After a full load, neither heavy module should have been imported (Anti-Pattern 6).
    load_config(VALID_YAML)
    assert "modal" not in sys.modules
    assert "ltx_core" not in sys.modules
