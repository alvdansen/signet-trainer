"""AUDIT-#3 — atomic save + completeness-checked ``find_latest`` walk-back.

CPU unit proofs that a container killed mid-``save`` can never poison ``resume``/``sample``:
  * atomic save leaves NO ``.tmp-*`` staging dir and a COMPLETE final dir (both files present);
  * ``find_latest`` skips a highest-step dir missing the adapter and walks back to the lower complete one;
  * ``find_latest`` skips a highest-step dir missing ``training_state.pt`` likewise;
  * ``find_latest`` returns ``None`` when every candidate is incomplete;
  * ``resume`` ignores a poisoned highest-step dir and resumes from the complete lower checkpoint.

Reuses the ``test_checkpoint_resume.py`` fixtures (tiny PEFT model, CPU-only, peft importorskip).
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
    CHECKPOINT_PREFIX,
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


def _poison(output_dir, step: int, *, drop: str) -> None:
    """Create a name-valid highest-step dir that is INCOMPLETE (missing one required file).

    ``drop`` selects which required file to omit: ``"adapter"`` or ``"state"``.
    """
    d = output_dir / f"{CHECKPOINT_PREFIX}{step:05d}-loss-0.0100"
    d.mkdir(parents=True, exist_ok=True)
    if drop != "adapter":
        (d / ADAPTER_FILENAME).write_bytes(b"x")
    if drop != "state":
        (d / TRAINING_STATE_FILENAME).write_bytes(b"x")


# --------------------------------------------------------------------------------------------------
# atomic save leaves a complete final dir and no staging dir
# --------------------------------------------------------------------------------------------------


def test_atomic_save_leaves_no_staging_and_complete_final(tmp_path) -> None:
    model = _wrapped()
    _randomize_adapter(model)
    opt, sched = _opt_and_sched(model)

    mgr = CheckpointManager(tmp_path)
    ckpt_dir = mgr.save(model, opt, sched, step=200, loss=0.1234)

    # Final dir is complete.
    assert ckpt_dir.name == "checkpoint-step-00200-loss-0.1234"
    assert (ckpt_dir / ADAPTER_FILENAME).exists()
    assert (ckpt_dir / TRAINING_STATE_FILENAME).exists()

    # No leftover staging dir (leading-dot .tmp-*).
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == [], f"staging dir not renamed away: {leftovers}"


# --------------------------------------------------------------------------------------------------
# find_latest walk-back past an incomplete highest-step dir
# --------------------------------------------------------------------------------------------------


def test_find_latest_skips_missing_adapter(tmp_path) -> None:
    model = _wrapped()
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)
    good = mgr.save(model, opt, sched, step=200, loss=0.5)  # complete lower dir
    _poison(tmp_path, 400, drop="adapter")  # incomplete higher dir

    assert mgr.find_latest() == good


def test_find_latest_skips_missing_training_state(tmp_path) -> None:
    model = _wrapped()
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)
    good = mgr.save(model, opt, sched, step=200, loss=0.5)
    _poison(tmp_path, 400, drop="state")

    assert mgr.find_latest() == good


def test_find_latest_none_when_all_incomplete(tmp_path) -> None:
    mgr = CheckpointManager(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    _poison(tmp_path, 100, drop="adapter")
    _poison(tmp_path, 200, drop="state")

    assert mgr.find_latest() is None


# --------------------------------------------------------------------------------------------------
# resume is structurally immune to a poisoned highest-step dir (the audit-#3 integration proof)
# --------------------------------------------------------------------------------------------------


def test_resume_ignores_poisoned_highest_step(tmp_path) -> None:
    # Complete lower checkpoint at step 200.
    model_a = _wrapped()
    _randomize_adapter(model_a)
    opt_a, sched_a = _opt_and_sched(model_a)
    _train_one_step(model_a, opt_a)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model_a, opt_a, sched_a, step=200, loss=0.42)
    saved_adapter = get_peft_model_state_dict(model_a)

    # A half-written HIGHER dir (killed mid-save) that would poison a name-trusting resume.
    _poison(tmp_path, 400, drop="state")

    model_b = _wrapped()
    opt_b, sched_b = _opt_and_sched(model_b)
    step = mgr.resume(model_b, opt_b, sched_b)

    assert step == 200  # resumed from the COMPLETE lower dir, not the poison 400
    after = get_peft_model_state_dict(model_b)
    assert set(after) == set(saved_adapter)
    for k in saved_adapter:
        assert torch.allclose(after[k], saved_adapter[k]), f"adapter {k!r} not re-injected"
