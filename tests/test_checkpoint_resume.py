"""TRAIN-07 — ``CheckpointManager`` save / find_latest + the landmine #1 adapter-reload proof.

CPU unit tests for ``train/checkpoint.py`` on a tiny PEFT model. The load-bearing test is
``test_resume_reinjects_adapter``: it proves a FRESH model (zero/random adapter) + a fresh optimizer
end up with the SAVED (non-zero) adapter tensors after ``resume`` — not a zero-init adapter against a
stale optimizer. That is the SC#4 correctness gate that the source's resume() silently failed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import (  # noqa: E402 — after skip-guard
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

from signet_trainer.train.checkpoint import (  # noqa: E402
    ADAPTER_FILENAME,
    TRAINING_STATE_FILENAME,
    CheckpointManager,
)


class _TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):  # pragma: no cover — not exercised
        return self.b(self.a(x))


def _wrapped() -> nn.Module:
    cfg = LoraConfig(
        r=4, lora_alpha=4, target_modules=["a", "b"], lora_dropout=0.0, bias="none"
    )
    return get_peft_model(_TwoLinear(), cfg)


def _randomize_adapter(model: nn.Module) -> None:
    # lora_B inits to zero; randomize it so the saved adapter carries real (non-zero) weights.
    for name, p in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.copy_(torch.randn_like(p))


def _opt_and_sched(model: nn.Module):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _step: 1.0)
    return opt, sched


def _train_one_step(model: nn.Module, opt) -> None:
    out = model(torch.randn(2, 8))
    out.sum().backward()
    opt.step()
    opt.zero_grad()


# --------------------------------------------------------------------------------------------------
# save + find_latest
# --------------------------------------------------------------------------------------------------


def test_save_writes_adapter_and_state_and_find_latest(tmp_path) -> None:
    model = _wrapped()
    _randomize_adapter(model)
    opt, sched = _opt_and_sched(model)

    mgr = CheckpointManager(tmp_path)
    ckpt_dir = mgr.save(model, opt, sched, step=200, loss=0.1234)

    assert ckpt_dir.name == "checkpoint-step-00200-loss-0.1234"
    assert (ckpt_dir / ADAPTER_FILENAME).exists()
    assert (ckpt_dir / TRAINING_STATE_FILENAME).exists()
    assert mgr.find_latest() == ckpt_dir


def test_find_latest_picks_highest_step(tmp_path) -> None:
    model = _wrapped()
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model, opt, sched, step=50, loss=1.0)
    high = mgr.save(model, opt, sched, step=400, loss=0.5)
    mgr.save(model, opt, sched, step=200, loss=0.7)
    assert mgr.find_latest() == high


# --------------------------------------------------------------------------------------------------
# THE landmine #1 proof
# --------------------------------------------------------------------------------------------------


def test_resume_reinjects_adapter(tmp_path) -> None:
    # Model A: train one step so the adapter is NON-zero, then save.
    model_a = _wrapped()
    _randomize_adapter(model_a)
    opt_a, sched_a = _opt_and_sched(model_a)
    _train_one_step(model_a, opt_a)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model_a, opt_a, sched_a, step=200, loss=0.42)
    saved_adapter = get_peft_model_state_dict(model_a)

    # Model B: FRESH (zero/random adapter) + fresh optimizer; resume must restore A's adapter.
    model_b = _wrapped()  # lora_B == 0 here
    opt_b, sched_b = _opt_and_sched(model_b)

    before = get_peft_model_state_dict(model_b)
    # Sanity: B's adapter differs from A's saved adapter before resume.
    assert any(not torch.allclose(before[k], saved_adapter[k]) for k in saved_adapter)

    step = mgr.resume(model_b, opt_b, sched_b)

    assert step == 200  # step restored
    after = get_peft_model_state_dict(model_b)
    assert set(after) == set(saved_adapter)
    for k in saved_adapter:
        assert torch.allclose(after[k], saved_adapter[k]), f"adapter {k!r} not re-injected"

    # optimizer state restored (non-empty after a step was taken on A).
    assert opt_b.state_dict()["state"], "optimizer state not restored"


def test_resume_empty_dir_is_cold_start(tmp_path) -> None:
    model = _wrapped()
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path / "does-not-exist-yet")
    assert mgr.resume(model, opt, sched) == 0


def test_no_hf_upload_path() -> None:
    import signet_trainer.train.checkpoint as ckpt

    src = open(ckpt.__file__, encoding="utf-8").read()
    assert "upload_best_to_hf" not in src
    assert "HfApi" not in src
