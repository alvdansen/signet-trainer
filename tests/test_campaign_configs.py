"""Phase 9 (09-06) — the example per-round campaign configs load through the REAL SignetConfig
loader and carry the LOCKED production character recipe (D-9-RECIPE), the D-9-CHAINED cold→warm chain shape,
and the OFFL-02 live posture.

These are the config-translation guardrails for the gated campaign (Wave 3-4): a bad
recipe value or a broken chain seam must die HERE (CPU, zero-spend), never after a metered
dispatch. No ltx_core / modal import, no filesystem touch beyond reading the YAMLs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signet_trainer.config.load import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_R1 = str(REPO_ROOT / "configs" / "campaign_r1.example.yaml")
CAMPAIGN_R2 = str(REPO_ROOT / "configs" / "campaign_r2.example.yaml")


# --------------------------------------------------------------------------------------------------
# Both rounds load through the real loader.
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def r1():
    return load_config(CAMPAIGN_R1)


@pytest.fixture(scope="module")
def r2():
    return load_config(CAMPAIGN_R2)


# --------------------------------------------------------------------------------------------------
# (a) LOCKED recipe values — the production character recipe, translated to signet's nested schema.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_name", ["r1", "r2"])
def test_locked_recipe_values(cfg_name, request):
    c = request.getfixturevalue(cfg_name)
    # DEVIATION — precedent rank/alpha 42 (signet default 64).
    assert c.lora.rank == 42
    assert c.lora.alpha == 42
    assert c.lora.dropout == 0.0
    # Optimizer + LR + regularization (precedent locked; NEVER Prodigy).
    assert c.training.optimizer == "adamw8bit"
    assert c.training.learning_rate == 5e-5
    assert c.training.weight_decay == 0.01
    assert c.training.max_grad_norm == 1.0
    assert c.training.gradient_accumulation_steps == 1
    assert c.training.uniform_prob == 0.30
    assert c.training.mixed_precision == "bf16"
    assert c.training.checkpoint_every == 200
    assert c.training.keep_checkpoints == 25  # gap-fill (09-02 field), precedent used 25
    assert c.training.seed == 42
    assert c.seed == 42
    assert c.data.batch_size == 1  # on data, not training


@pytest.mark.parametrize("cfg_name", ["r1", "r2"])
def test_ff_net_targets_present(cfg_name, request):
    # ff.net inclusion is the identity-capacity invariant (attn-only underfits likeness).
    c = request.getfixturevalue(cfg_name)
    assert "ff.net.0.proj" in c.lora.target_modules
    assert "ff.net.2" in c.lora.target_modules
    # attn1 + attn2 + ff.net = the default 10-target set.
    assert len(c.lora.target_modules) == 10


# --------------------------------------------------------------------------------------------------
# (b) OFFL-02 live posture — in_loop_sampling on AND block-swap active.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_name", ["r1", "r2"])
def test_offl02_posture(cfg_name, request):
    c = request.getfixturevalue(cfg_name)
    assert c.validation.in_loop_sampling is True
    assert c.offload.blocks_to_swap > 0  # active block-swap (OFFL-02 live proof)
    assert c.validation.prompts, "in_loop_sampling requires a non-empty prompt"
    # the operator-locked eval spec (07-11) — 40 steps; superseded the precedent sample_steps-10 deviation.
    assert c.validation.num_inference_steps == 40
    assert c.validation.two_stage_upscale is False  # precedent F5 — single-stage only


# --------------------------------------------------------------------------------------------------
# (c) The chain shape — r1 cold start (no init_adapter_path), r2 warm restart (init_adapter_path set).
# --------------------------------------------------------------------------------------------------


def test_r1_is_cold_start(r1):
    assert r1.training.init_adapter_path is None


def test_r2_is_warm_restart(r2):
    assert r2.training.init_adapter_path is not None
    # points at r1's checkpoint dir (Volume-relative), the D-9-CHAINED seam.
    assert "campaign_r1" in r2.training.init_adapter_path


def test_output_dirs_are_per_round(r1, r2):
    # _mf suffix — rounds re-pointed to multi-F native buckets mid-campaign (07-11 correction).
    assert r1.output_dir == "outputs/campaign_r1_mf"
    assert r2.output_dir == "outputs/campaign_r2_mf"


# --------------------------------------------------------------------------------------------------
# (d) A100 house-rule rate + no secret literals in the configs (T-09-06-02).
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_name", ["r1", "r2"])
def test_a100_rate(cfg_name, request):
    c = request.getfixturevalue(cfg_name)
    assert c.modal.hourly_rate_usd == 1.64  # A100 (single-A100 house rule), not the precedent's 4.0
