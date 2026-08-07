"""Chained warm-restart cold-start branch + keep_checkpoints wiring (D-9-CHAINED / D-9-RECIPE).

Three zero-GPU layers (mirroring ``test_frozen_stacking.py`` + ``test_multi_frame_sample_wiring.py``):

  * PURE HELPER — ``should_warm_start`` truth table: warm-start iff ``init_adapter_path`` is set AND
    there is NO in-dir checkpoint (an in-dir checkpoint means same-dir ``resume()`` wins).

  * TINY-PEFT WARM-START — save a randomized adapter to a dir, ``load_adapter_into`` a second
    fresh-injected model, and assert the adapter weights transferred (a real cold-start warm restart
    on CPU — no ltx_core / no CUDA / no modal).

  * TRAIN-SEAM TEXT SCAN — ``fns.py::train`` constructs ``CheckpointManager(keep_n=
    config.training.keep_checkpoints)``, guards the cold-start branch on ``should_warm_start(...)``,
    calls ``load_adapter_into(model, ...)`` from ``config.training.init_adapter_path``, and builds
    the optimizer AFTER the warm-start branch (fresh optimizer over warm-started weights).

The LIVE GPU proof (warm-restart across rounds) rides the gated campaign run (Wave 4).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from signet_trainer.train.loop import should_warm_start

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402  — after skip-guard

from signet_trainer.lora.peft import load_adapter_into, save_adapter  # noqa: E402
from signet_trainer.train.checkpoint import CHECKPOINT_PREFIX, CheckpointManager  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


class _TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):  # pragma: no cover — not exercised
        return self.b(self.a(x))


def _fresh_peft() -> tuple[nn.Module, LoraConfig]:
    cfg = LoraConfig(
        r=4, lora_alpha=4, target_modules=["a", "b"], lora_dropout=0.0, bias="none"
    )
    return get_peft_model(_TwoLinear(), cfg), cfg


def _randomize_lora_b(model: nn.Module) -> None:
    # lora_B inits to zero; randomize so the source adapter carries real (non-zero) weights.
    for name, p in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.copy_(torch.randn_like(p))


# --------------------------------------------------------------------------------------------------
# (a) should_warm_start truth table
# --------------------------------------------------------------------------------------------------


def test_should_warm_start_truth_table() -> None:
    """Warm-start iff init_adapter_path is set AND no in-dir checkpoint (resume precedence)."""
    # cold start (no checkpoint) + init path set -> warm-start.
    assert should_warm_start(False, "prev_round/checkpoint-step-00500-loss-0.1000") is True
    # in-dir checkpoint present -> same-dir resume wins, init_adapter_path ignored.
    assert should_warm_start(True, "prev_round/checkpoint-step-00500-loss-0.1000") is False
    # no init path -> never warm-start (plain cold start or plain resume).
    assert should_warm_start(False, None) is False
    assert should_warm_start(True, None) is False


# --------------------------------------------------------------------------------------------------
# (b) tiny-PEFT warm-start — adapter weights transfer into a fresh-injected model
# --------------------------------------------------------------------------------------------------


def test_warm_start_transfers_adapter_weights(tmp_path) -> None:
    """load_adapter_into a fresh model transfers the prior round's adapter weights (warm start)."""
    src, cfg = _fresh_peft()
    _randomize_lora_b(src)
    prev_round = tmp_path / "prev_round"
    save_adapter(src, prev_round)  # writes adapter_config.json + adapter_model.safetensors

    # Fresh cold-start model: lora_B is zero (differs from the randomized source).
    dst = get_peft_model(_TwoLinear(), cfg)
    dst_before = {n: p.detach().clone() for n, p in dst.named_parameters() if "lora_B" in n}

    load_adapter_into(dst, prev_round)  # the warm-restart primitive (D-9-CHAINED cold-start branch)

    src_lora = {n: p for n, p in src.named_parameters() if "lora" in n}
    dst_lora = {n: p for n, p in dst.named_parameters() if "lora" in n}
    assert set(src_lora) == set(dst_lora)
    for n in src_lora:
        assert torch.allclose(src_lora[n], dst_lora[n]), f"adapter weight not transferred: {n}"
    # Prove a REAL transfer happened: at least one lora_B moved off its zero init.
    moved = any(
        not torch.allclose(dst_before[n], dict(dst.named_parameters())[n]) for n in dst_before
    )
    assert moved, "warm-start did not change any lora_B weight (no real transfer)"


# --------------------------------------------------------------------------------------------------
# (c) precedence via CheckpointManager.find_latest() — an in-dir checkpoint disables warm-start
# --------------------------------------------------------------------------------------------------


def test_warm_start_precedence_with_checkpoint_manager(tmp_path) -> None:
    """The exact fns.py predicate: should_warm_start(find_latest() is not None, init_path)."""
    out = tmp_path / "run"
    init_path = "prev_round/checkpoint-step-00500-loss-0.1000"

    # Cold start: no in-dir checkpoint yet -> find_latest() None -> warm-start eligible.
    cm = CheckpointManager(out)
    assert cm.find_latest() is None
    assert should_warm_start(cm.find_latest() is not None, init_path) is True

    # A half-written in-dir checkpoint (dir name only, audit #3) is NOT a resume target ->
    # find_latest() still None -> warm-start stays eligible.
    incomplete = out / f"{CHECKPOINT_PREFIX}00005-loss-0.2000"
    incomplete.mkdir(parents=True)
    assert CheckpointManager(out).find_latest() is None

    # A COMPLETE in-dir checkpoint appears -> find_latest() not None -> resume wins,
    # warm-start disabled.
    ckpt = out / f"{CHECKPOINT_PREFIX}00010-loss-0.1000"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_bytes(b"")
    (ckpt / "training_state.pt").write_bytes(b"")
    cm2 = CheckpointManager(out)
    assert cm2.find_latest() is not None
    assert should_warm_start(cm2.find_latest() is not None, init_path) is False


# --------------------------------------------------------------------------------------------------
# Train-seam text scan (the D-9-CHAINED cold-start branch + keep_n wiring in fns.py::train)
# --------------------------------------------------------------------------------------------------


def test_fns_wires_keep_n_and_warm_start_branch() -> None:
    """keep_checkpoints retention + should_warm_start-gated load_adapter_into from init_adapter_path."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    assert "keep_n=config.training.keep_checkpoints" in code, (
        "CheckpointManager must be constructed with keep_n=config.training.keep_checkpoints"
    )
    assert "should_warm_start(" in code, "the cold-start branch must gate on should_warm_start(...)"
    assert "load_adapter_into(model" in code, (
        "the warm-start branch must call load_adapter_into(model, init_dir)"
    )
    assert "config.training.init_adapter_path" in code, (
        "the branch must read config.training.init_adapter_path"
    )


def test_fns_optimizer_built_after_warm_start() -> None:
    """Fresh optimizer over warm-started weights: build_optimizer appears AFTER the warm-start call."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    warm_idx = code.index("load_adapter_into(model")
    opt_idx = code.index("build_optimizer(model")
    assert warm_idx < opt_idx, (
        "build_optimizer(model, ...) must be constructed AFTER the warm-start branch so the "
        "warm-started weights get a FRESH optimizer (no optimizer-state carry, D-9-CHAINED)"
    )
