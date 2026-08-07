"""BK-01 — backup.plan is a pure, Windows-testable sync-planning core.

CPU proofs (stdlib tmp-dir fixtures; no torch/modal/GPU):
  * a half-written dir (missing training_state.pt) is EXCLUDED by list_complete_checkpoints;
  * select_for_backup(what='all') returns EVERY complete dir (mirror-all house default, D-BK-2);
  * select_for_backup(what='final') returns only the max-step dir;
  * select_for_backup(what='final+intervals', interval=1000) returns final + the 1000-boundary dirs;
  * plan_uploads skips a dir already in remote_names and is idempotent (second pass returns []);
  * a DRIFT-GUARD (importorskip torch/peft) asserts plan.py's mirror constants equal the canonical
    train.checkpoint + lora.peft constants — the mirror can never silently drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signet_trainer.backup import plan


def _make_checkpoint(root: Path, step: int, *, complete: bool = True) -> Path:
    """Create a ``checkpoint-step-{step:05d}-loss-*`` dir; complete = both required files present."""
    d = root / f"{plan.CHECKPOINT_PREFIX}{step:05d}-loss-0.1000"
    d.mkdir(parents=True, exist_ok=True)
    (d / plan.MIRRORED_ADAPTER_FILENAME).write_bytes(b"x")
    if complete:
        (d / plan.TRAINING_STATE_FILENAME).write_bytes(b"x")
    return d


# ---- pure-module discipline: no heavy imports leak into plan.py ----


def test_plan_module_is_pure_no_heavy_imports():
    # Scan the AST import NODES (not a substring match against the docstring, which legitimately
    # names the forbidden modules to document the discipline).
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(plan))
    imported: set[str] = set()  # full dotted module paths
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    top_level = {m.split(".")[0] for m in imported}
    forbidden_top = {"modal", "torch", "peft", "huggingface_hub"}
    assert not (forbidden_top & top_level), (
        f"backup.plan must not import {forbidden_top & top_level} (pure-module discipline)"
    )
    # And it must not import the torch-pulling canonical modules (train.checkpoint / lora.peft).
    forbidden_paths = {"signet_trainer.train.checkpoint", "signet_trainer.lora.peft"}
    assert not (forbidden_paths & imported), (
        f"backup.plan must not import {forbidden_paths & imported} (they pull torch)"
    )


# ---- is_complete / list_complete_checkpoints: a half-written dir is excluded ----


def test_half_written_dir_excluded(tmp_path):
    _make_checkpoint(tmp_path, 1000, complete=True)
    half = _make_checkpoint(tmp_path, 2000, complete=False)  # missing training_state.pt
    assert plan.is_complete(half) is False

    complete = plan.list_complete_checkpoints(tmp_path)
    names = [d.name for d in complete]
    assert half.name not in names
    assert len(complete) == 1
    assert complete[0].name.startswith(f"{plan.CHECKPOINT_PREFIX}01000")


def test_list_complete_checkpoints_sorted_ascending(tmp_path):
    _make_checkpoint(tmp_path, 3000)
    _make_checkpoint(tmp_path, 1000)
    _make_checkpoint(tmp_path, 2000)
    complete = plan.list_complete_checkpoints(tmp_path)
    steps = [plan._step_of(d) for d in complete]
    assert steps == [1000, 2000, 3000]


def test_list_complete_checkpoints_missing_dir_returns_empty(tmp_path):
    assert plan.list_complete_checkpoints(tmp_path / "nope") == []


# ---- select_for_backup scope selector (D-BK-2) ----


def test_select_all_returns_every_complete_dir(tmp_path):
    for step in (1000, 2000, 3000, 5000):
        _make_checkpoint(tmp_path, step)
    complete = plan.list_complete_checkpoints(tmp_path)
    selected = plan.select_for_backup(complete, what="all", interval=None)
    assert len(selected) == len(complete) == 4  # mirror-all, NOT final-only


def test_select_final_returns_only_max_step(tmp_path):
    for step in (1000, 2000, 3000, 5000):
        _make_checkpoint(tmp_path, step)
    complete = plan.list_complete_checkpoints(tmp_path)
    selected = plan.select_for_backup(complete, what="final", interval=None)
    assert len(selected) == 1
    assert plan._step_of(selected[0]) == 5000


def test_select_final_plus_intervals(tmp_path):
    for step in (500, 1000, 1500, 2000, 3500):
        _make_checkpoint(tmp_path, step)
    complete = plan.list_complete_checkpoints(tmp_path)
    selected = plan.select_for_backup(complete, what="final+intervals", interval=1000)
    steps = sorted(plan._step_of(d) for d in selected)
    # 1000 and 2000 are on the boundary; 3500 (final) is included even though not on a boundary.
    assert steps == [1000, 2000, 3500]


def test_select_final_plus_intervals_requires_interval(tmp_path):
    _make_checkpoint(tmp_path, 1000)
    complete = plan.list_complete_checkpoints(tmp_path)
    with pytest.raises(ValueError, match="requires a positive interval"):
        plan.select_for_backup(complete, what="final+intervals", interval=None)


def test_select_empty_returns_empty(tmp_path):
    assert plan.select_for_backup([], what="all", interval=None) == []


# ---- plan_uploads: idempotent skip ----


def test_plan_uploads_skips_already_backed_up(tmp_path):
    for step in (1000, 2000, 3000):
        _make_checkpoint(tmp_path, step)
    complete = plan.list_complete_checkpoints(tmp_path)
    selected = plan.select_for_backup(complete, what="all", interval=None)

    already = {complete[0].name}  # the step-1000 dir is already remote
    first = plan.plan_uploads(selected, already)
    assert complete[0].name not in [d.name for d in first]
    assert len(first) == 2

    # Idempotent: add the first pass to remote, re-plan -> nothing left to upload.
    remote = already | {d.name for d in first}
    second = plan.plan_uploads(selected, remote)
    assert second == []


# ---- WR-01: idempotency keys on DESTINATION completeness, not the bare dir name ----


def test_complete_remote_names_only_returns_fully_mirrored_dirs():
    # A dir with BOTH required files is complete; a dir missing training_state.pt (an interrupted
    # upload) is NOT — so plan_uploads re-selects it for repair rather than trusting its name.
    remote_dir_files = {
        "checkpoint-step-01000": {plan.MIRRORED_ADAPTER_FILENAME, plan.TRAINING_STATE_FILENAME},
        "checkpoint-step-02000": {plan.MIRRORED_ADAPTER_FILENAME},  # half-uploaded — missing state
        "checkpoint-step-03000": set(),  # bare dir, no files landed yet
    }
    complete = plan.complete_remote_names(remote_dir_files)
    assert complete == {"checkpoint-step-01000"}


def test_partial_remote_dir_is_replanned_for_upload(tmp_path):
    for step in (1000, 2000):
        _make_checkpoint(tmp_path, step)
    complete = plan.list_complete_checkpoints(tmp_path)
    selected = plan.select_for_backup(complete, what="all", interval=None)
    names = [d.name for d in selected]

    # step-1000 is fully mirrored; step-2000 was interrupted mid-upload (adapter only).
    remote_dir_files = {
        names[0]: {plan.MIRRORED_ADAPTER_FILENAME, plan.TRAINING_STATE_FILENAME},
        names[1]: {plan.MIRRORED_ADAPTER_FILENAME},
    }
    remote_names = plan.complete_remote_names(remote_dir_files)
    to_upload = [d.name for d in plan.plan_uploads(selected, remote_names)]
    # the complete dir is skipped; the partial dir is RE-selected (repaired), never masqueraded as done.
    assert names[0] not in to_upload
    assert names[1] in to_upload


# ---- DRIFT-GUARD: plan.py mirror constants equal the canonical train.checkpoint + lora.peft ----


def test_mirror_constants_match_canonical():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from signet_trainer.lora.peft import ADAPTER_FILENAME as canon_adapter
    from signet_trainer.train.checkpoint import (
        CHECKPOINT_PREFIX,
        TRAINING_STATE_FILENAME,
    )

    assert plan.CHECKPOINT_PREFIX == CHECKPOINT_PREFIX
    assert plan.TRAINING_STATE_FILENAME == TRAINING_STATE_FILENAME
    assert plan.MIRRORED_ADAPTER_FILENAME == canon_adapter


# ---- D-10-DEF-18: the mirror's ONE file-set subtraction, and the guard that keeps it safe ----


def test_mirror_excluded_files_is_disjoint_from_required_backup_files():
    """THE load-bearing invariant: the mirror may never omit a completeness-critical file.

    ``complete_remote_names`` treats a destination dir as backed-up only when its files are a
    SUPERSET of ``REQUIRED_BACKUP_FILES``. If an excluded basename ever intersected that set, every
    upload would be planned, uploaded, and then judged INCOMPLETE forever — an infinite re-upload
    loop that also silently destroys the durability guarantee. Pin them disjoint.
    """
    overlap = plan.MIRROR_EXCLUDED_FILES & plan.REQUIRED_BACKUP_FILES
    assert not overlap, (
        f"MIRROR_EXCLUDED_FILES must never exclude a completeness-critical file; got {overlap}. "
        "Excluding one would make every mirrored dir permanently 'incomplete'."
    )


def test_excluded_files_are_the_documented_peft_model_card_only():
    """The subtraction stays EXACTLY the PEFT-autogenerated model card (D-10-DEF-18), not a bucket.

    Deliberately exact: widening this set is a durability decision (what a restore will NOT bring
    back), so it must be an explicit edit accompanied by a rationale — never an incidental one.
    """
    assert plan.MIRROR_EXCLUDED_FILES == frozenset({"README.md"})


def test_dir_stays_complete_after_the_excluded_files_are_removed(tmp_path):
    """Behavioural form of the invariant: strip every excluded file, ``is_complete`` still holds."""
    d = _make_checkpoint(tmp_path, 50)
    for name in plan.MIRROR_EXCLUDED_FILES:
        (d / name).write_text("front matter the Hub rejects", encoding="utf-8")
    assert plan.is_complete(d)

    for name in plan.MIRROR_EXCLUDED_FILES:  # what the destination copy will look like
        (d / name).unlink()
    assert plan.is_complete(d), (
        "a mirrored dir missing only the excluded files must still count as COMPLETE"
    )
