"""TRAIN-07 — ``CheckpointManager`` save / find_latest + the landmine #1 adapter-reload proof.

CPU unit tests for ``train/checkpoint.py`` on a tiny PEFT model. The load-bearing test is
``test_resume_reinjects_adapter``: it proves a FRESH model (zero/random adapter) + a fresh optimizer
end up with the SAVED (non-zero) adapter tensors after ``resume`` — not a zero-init adapter against a
stale optimizer. That is the SC#4 correctness gate that the source's resume() silently failed.
"""

from __future__ import annotations
from pathlib import Path

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


# ==================================================================================================
# LANDMINE #1b — resume must work on a QUANTIZED model (measured failure, 2026-08-09)
# ==================================================================================================


def test_resume_on_a_quantized_model_bypasses_load_state_dict(tmp_path) -> None:
    """A quanto-quantized base must not route adapter weights through ``load_state_dict``.

    THE MEASURED FAILURE. Phase 1 of the embe chain trained to step 500, the container was
    interrupted, Modal retried it (``retries=modal.Retries(max_retries=10)``), and every retry died:

        KeyError: 'base_model.model.time_text_embed.timestep_embedder.linear_1.weight._data'
        optimum/quanto/tensor/qbytes.py:90 load_from_state_dict
        peft/utils/save_and_load.py:941 set_peft_model_state_dict

    ``set_peft_model_state_dict`` ends in ``model.load_state_dict(..., strict=False)``, which walks
    EVERY submodule — not only those the incoming dict names. quanto's ``QModuleMixin`` overrides
    ``_load_from_state_dict`` and pops its own ``_data`` keys UNCONDITIONALLY, never consulting
    ``strict``. So under quantization the trainer had no crash resume at all: an interruption was
    permanent, which inverts the documented property.

    This fake reproduces the shape without needing quanto or a GPU: a module whose
    ``_load_from_state_dict`` raises exactly as quanto's does. If ``resume`` ever routes a quantized
    model through ``load_state_dict`` again, this raises.
    """
    import torch.nn as nn

    class _FakeQuantoLinear(nn.Module):
        """Stands in for optimum.quanto's QLinear: pops its own key, ignores ``strict``."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # noqa: ANN001
            state_dict.pop(prefix + "weight._data")  # KeyError when absent — quanto's exact shape

    module = _FakeQuantoLinear()
    with pytest.raises(KeyError):
        module._load_from_state_dict({}, "base.", None, True, [], [], [])


def test_a_rank_change_between_rounds_is_refused_not_silently_partial() -> None:
    """A shape mismatch on resume must name the rank lock, not load a half-adapter.

    rank == alpha is a HARD LOCK across a chain precisely because changing either re-shapes
    lora_A/lora_B. Resuming across such a change cannot work, and the failure a partial load
    produces is a run continuing from a half-restored state at a plausible loss — the worst
    available outcome, because nothing looks wrong.

    ⚠ READS lora/peft.py, NOT checkpoint.py. The refusals moved when the quantized-adapter load was
    consolidated: resume() had the fix and load_adapter_into did not, so the §8 band render walked
    into the same KeyError from the other side. ``resume`` now delegates, and this assertion follows
    the implementation rather than the caller — pinning the caller would have gone quietly vacuous
    the moment the logic moved, which is the failure mode a source-text test exists to avoid.
    """
    source = (
        Path(__file__).resolve().parents[1] / "src/signet_trainer/lora/peft.py"
    ).read_text(encoding="utf-8")
    # Short, contiguous phrases only: an f-string wraps across lines, so a long literal
    # asserts the FORMATTING as much as the content.
    assert "HARD LOCK across a chain" in source, (
        "the shape-mismatch refusal no longer explains WHY it cannot resume"
    )
    assert "PARTIAL adapter is worse than refusing" in source, (
        "the partial-load refusal was removed — a partial resume is silent corruption"
    )


def test_the_resume_decision_is_printed_not_only_logged() -> None:
    """"Continue or restart?" must reach stdout, not just the logger.

    Modal shows container stdout at the default WARNING level, so a ``logger.info`` report leaves
    NO trace in a metered run. That is not cosmetic here: a silent restart burns the entire budget
    from step 0 and is indistinguishable from progress in the log. This was found the expensive
    way — after the quantized-resume fix, the relaunched 5000-step phase-1 run gave no answer to
    "did it resume from 500?" at all, because every branch reported at INFO.

    Both branches are asserted, because only reporting the happy path is the same defect: an
    unreported COLD START is exactly the case you need to see.
    """
    source = (
        Path(__file__).resolve().parents[1] / "src/signet_trainer/train/checkpoint.py"
    ).read_text(encoding="utf-8")
    assert "COLD START at step 0" in source, "the cold-start branch must announce itself on stdout"
    assert "RESUMED from" in source, "the resume branch must announce itself on stdout"
    assert source.count("print(banner)") >= 2, (
        "both resume branches must print; a logger-only report is invisible at Modal's default level"
    )
    assert "it does not restart" in source, (
        "the resume banner must state the CONSEQUENCE, not just the step number — the reader is "
        "checking whether they are about to pay for a silent restart"
    )


def test_resume_ACTUALLY_restores_the_adapter_on_a_quantized_model(tmp_path) -> None:
    """End-to-end through the quantized branch: save, wipe, resume, compare weights.

    The gap this closes. ``test_resume_on_a_quantized_model_bypasses_load_state_dict`` asserts only
    that the fake QModule raises the way quanto does — it never drives ``resume``. So the assertion
    that mattered ("resume works on a quantized model") was never actually made, and the failure it
    was written after was found on a metered container rather than here.

    That gap had consequences twice. The fix landed in ``resume`` and NOT in ``load_adapter_into``,
    so the §8 band render hit the identical KeyError from the other side; the logic is now shared in
    ``lora/peft.load_adapter_state_into`` and this drives the caller that has already broken once.

    The model is PEFT-wrapped AND carries a quanto-shaped module, so ``is_quanto_quantized`` returns
    True and ``resume`` must take the direct-assignment path. If it ever routes back through
    ``load_state_dict``, the fake's unconditional ``pop`` raises KeyError and this fails loudly.
    """
    from signet_trainer.lora.peft import is_quanto_quantized

    class QFakeLinear(nn.Module):
        __module__ = "optimum.quanto.nn.qmodule"

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # noqa: ANN001
            state_dict.pop(prefix + "weight._data")

    model = _wrapped()
    _randomize_adapter(model)
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model, opt, sched, step=200, loss=0.1234)
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    assert any(t.abs().sum() > 0 for t in saved.values()), "fixture wrote an all-zero adapter"

    # A fresh wrapper (lora_B back to zero) that is ALSO quantized, exactly as the trainer's is.
    revived = _wrapped()
    revived.quantized_stub = QFakeLinear()
    assert is_quanto_quantized(revived), "the fixture is not detected as quantized — test is vacuous"
    for name, p in revived.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.zero_()

    step = mgr.resume(revived)

    assert step == 200
    restored = {n: p.detach() for n, p in revived.named_parameters() if "lora_B" in n}
    assert set(restored) == set(saved), "the restored parameter set does not match what was saved"
    for name, want in saved.items():
        assert torch.allclose(restored[name], want), (
            f"{name} was not restored — a resume that silently leaves lora_B at zero continues "
            f"from a ZERO adapter against an optimizer pointing at step 200, which is landmine #1 "
            f"wearing the quantized branch as a disguise"
        )
