"""AUDIT #34 + ISSUE #30 FINDING 5 — the aside-then-in publish swap and the orphan-staging sweep.

CPU unit proofs for the two ``CheckpointManager.save()`` findings that had to be designed
together (both rewrite the same publish sequence):

  * #34 — a crash INSIDE the publish swap must never destroy the checkpoint being replaced; the
    old dir survives under a ``.old-`` alias and self-heals on the next ``save()``/``find_latest()``;
  * #30 f5 — a same-step retry at a different loss reclaims its own (step-scoped) staging dir, and
    a genuinely orphaned ``.tmp-`` dir for an already-superseded step gets swept, so neither leaks
    disk on every crash-and-resume cycle;
  * the normal, non-colliding save path is unchanged: same final dir name, same two files, no
    leftover ``.tmp-``/``.old-`` dirs.

Reuses the ``test_checkpoint_resume.py`` fixtures (tiny PEFT model, CPU-only, peft importorskip).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402 — after skip-guard

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
    cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["a", "b"], lora_dropout=0.0, bias="none")
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


def _dir_names(root) -> list[str]:
    return sorted(p.name for p in root.iterdir())


# --------------------------------------------------------------------------------------------------
# #34 — a crash INSIDE the publish swap never destroys the checkpoint being replaced
# --------------------------------------------------------------------------------------------------


def test_crash_between_aside_rename_and_publish_leaves_a_recoverable_dir(tmp_path, monkeypatch) -> None:
    """A crash between the two ``os.rename`` calls must leave the old checkpoint ON DISK.

    Reproduces issue #34's exact trigger: the unconditional tail save re-saves the SAME step at
    the SAME loss, so ``ckpt_dir`` already exists and the publish must replace a durable dir.
    The old code did ``rmtree(ckpt_dir)`` then ``os.rename(staging, ckpt_dir)`` — a crash in that
    window left NEITHER name on disk (reproduced upstream to ``find_latest() -> None``). The new
    swap must never reach a state with neither the live name nor the ``.old-`` alias present.
    """
    model = _wrapped()
    _randomize_adapter(model)
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)

    # First save: the in-loop save at step 800.
    first = mgr.save(model, opt, sched, step=800, loss=1.0)
    saved_adapter_bytes = (first / ADAPTER_FILENAME).read_bytes()

    # Second save: the guaranteed unconditional tail save at the SAME step+loss (loop.py:431).
    # Let the FIRST os.rename (ckpt_dir -> .old- alias) succeed, then crash on the SECOND
    # (staging -> ckpt_dir) — the exact window audit #34 flags.
    real_rename = os.rename
    calls = {"n": 0}

    def _flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated crash mid-publish")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", _flaky_rename)
    with pytest.raises(OSError, match="simulated crash mid-publish"):
        mgr.save(model, opt, sched, step=800, loss=1.0)
    monkeypatch.setattr(os, "rename", real_rename)

    # Never a window with neither present: the committed checkpoint survives under its aside
    # alias — not deleted outright, which is the #34 defect this swap replaces.
    old_alias = tmp_path / ".old-checkpoint-step-00800-loss-1.0000"
    assert old_alias.exists(), "the committed checkpoint was destroyed, not preserved, mid-swap"
    assert (old_alias / ADAPTER_FILENAME).read_bytes() == saved_adapter_bytes
    assert (old_alias / TRAINING_STATE_FILENAME).exists()

    # The final name is transiently gone, but self-heals on the very next read — this is what
    # makes the surviving dir "recoverable" rather than merely present-but-invisible.
    healed = mgr.find_latest()
    assert healed is not None, "a crash-orphaned .old- alias must self-heal, not stay invisible"
    assert healed.name == "checkpoint-step-00800-loss-1.0000"
    assert (healed / ADAPTER_FILENAME).read_bytes() == saved_adapter_bytes
    assert not old_alias.exists(), "the alias must be consumed by the heal, not left duplicated"


def test_recover_stale_swaps_discards_alias_when_final_already_won(tmp_path) -> None:
    """A ``.old-`` alias whose final name ALREADY exists is a stale duplicate, not data to restore.

    Covers the other half of the crash window: the crash lands AFTER the publish rename succeeds
    but BEFORE the aside cleanup runs. The fresh copy under the regex-matching name is the one
    that must survive; the alias is safe to discard.
    """
    model = _wrapped()
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)
    current = mgr.save(model, opt, sched, step=200, loss=0.5)

    # Simulate the leftover alias from a crash landing after the publish rename won.
    alias = tmp_path / ".old-checkpoint-step-00200-loss-0.5000"
    alias.mkdir()
    (alias / "leftover.txt").write_bytes(b"stale")

    assert mgr.find_latest() == current
    assert not alias.exists(), "a stale .old- alias whose final name already won must be swept"


# --------------------------------------------------------------------------------------------------
# #30 finding 5 — orphaned staging dirs are swept, live ones are not
# --------------------------------------------------------------------------------------------------


def test_orphan_tmp_dir_for_a_superseded_step_is_swept(tmp_path) -> None:
    """A crash-orphaned ``.tmp-`` dir for a step already superseded by the new save must be deleted.

    Names it STEP-scoped (the new, fixed naming), simulating the case the old loss-suffixed name
    could never self-clean: the crash landed on a step that is never retried at that exact step
    again (e.g. it was the tail save and training already ended).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    orphan = tmp_path / f".tmp-{CHECKPOINT_PREFIX}00200"
    orphan.mkdir(parents=True)
    (orphan / ADAPTER_FILENAME).write_bytes(b"x")
    (orphan / TRAINING_STATE_FILENAME).write_bytes(b"x")

    model = _wrapped()
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model, opt, sched, step=400, loss=0.5)

    assert not orphan.exists(), "an orphaned staging dir for an already-superseded step must be swept"


def test_orphan_sweep_does_not_touch_a_higher_step_staging_dir(tmp_path) -> None:
    """The sweep must be scoped to steps <= the one being committed, never a step still ahead."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    future = tmp_path / f".tmp-{CHECKPOINT_PREFIX}00900"
    future.mkdir(parents=True)

    model = _wrapped()
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model, opt, sched, step=400, loss=0.5)

    assert future.exists(), "the sweep must not delete a staging dir for a step beyond the current save"


def test_crash_mid_write_orphan_is_reclaimed_by_a_same_step_retry_at_a_different_loss(tmp_path) -> None:
    """The exact issue #30 f5 trigger: a crash mid-write, then resume lands on a DIFFERENT loss.

    Under the OLD loss-suffixed staging name this orphaned a full adapter+optimizer-state copy
    forever: the retry's staging dir got a DIFFERENT name (embedding the new loss), so neither
    ``find_latest`` nor ``prune`` could ever see it (both filter on ``_STEP_RE``, which a leading
    ``.`` never matches). The step-scoped name fixes it structurally — the retry's own
    staging-dir-exists check at the top of ``save()`` reclaims the identical directory.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    crashed_staging = tmp_path / f".tmp-{CHECKPOINT_PREFIX}00200"
    crashed_staging.mkdir(parents=True)
    (crashed_staging / ADAPTER_FILENAME).write_bytes(b"partial")

    model = _wrapped()
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model, opt, sched, step=200, loss=0.5678)  # resumed retry lands on a different loss

    assert not crashed_staging.exists(), "the crash-orphaned staging dir was not reclaimed"
    assert _dir_names(tmp_path) == ["checkpoint-step-00200-loss-0.5678"]


def test_same_step_different_loss_resave_leaves_no_staging_leftover(tmp_path) -> None:
    """A clean (no-crash) same-step retry at a different loss must not leave a staging dir behind.

    Both saves land on the same step-scoped staging name; each is consumed by its own publish, so
    two distinct final dirs (one per loss) are expected — pruning across them is ``keep_n``'s job,
    not this test's concern.
    """
    model = _wrapped()
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)
    mgr.save(model, opt, sched, step=200, loss=0.1234)
    mgr.save(model, opt, sched, step=200, loss=0.5678)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == [], f"leftover staging dir(s) after a same-step resave: {leftovers}"
    finals = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(CHECKPOINT_PREFIX))
    assert finals == [
        "checkpoint-step-00200-loss-0.1234",
        "checkpoint-step-00200-loss-0.5678",
    ], finals


# --------------------------------------------------------------------------------------------------
# normal save path is byte-identical in artifact layout
# --------------------------------------------------------------------------------------------------


def test_normal_save_layout_is_unchanged_no_tmp_or_old_leftovers(tmp_path) -> None:
    """A plain, non-colliding save must leave exactly the final dir — same name, same two files.

    Pins that the #34/#30 rewrite did not change the shipped artifact shape: no ``.old-`` alias
    (nothing existed to swap aside), no ``.tmp-`` staging dir, and the final name is unchanged.
    """
    model = _wrapped()
    _randomize_adapter(model)
    opt, sched = _opt_and_sched(model)
    mgr = CheckpointManager(tmp_path)

    ckpt_dir = mgr.save(model, opt, sched, step=200, loss=0.1234)

    assert ckpt_dir.name == "checkpoint-step-00200-loss-0.1234"
    assert (ckpt_dir / ADAPTER_FILENAME).exists()
    assert (ckpt_dir / TRAINING_STATE_FILENAME).exists()
    assert _dir_names(tmp_path) == ["checkpoint-step-00200-loss-0.1234"], (
        "a non-colliding save must leave no .tmp-/.old- leftovers alongside the final dir"
    )
