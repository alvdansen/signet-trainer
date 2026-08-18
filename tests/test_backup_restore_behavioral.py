"""Behavioral tests for backup_sync / restore (issue #23 findings 2+3, issue #33 finding 1).

Unlike ``test_backup_restore_fns.py`` (pure static source-scan, zero modal import), these tests
DRIVE the real function bodies via Modal's ``.local()`` escape hatch — it calls the raw function
directly in-process (no container boot, no network dispatch) — with ``huggingface_hub`` and the
checkpoints Volume monkeypatched out. Source-scanning cannot exercise the actual CONTROL FLOW these
three findings live in (the narrowed ``except``, the enabled-gate no-op, restore's before/after
delta), so these tests call the functions for real.

IMPORT DISCIPLINE (Anti-Pattern 6): ``signet_trainer.modal.fns`` (and therefore ``modal`` /
``huggingface_hub``) is imported lazily inside each test, never at module top, so pytest COLLECTION
never pulls ``modal`` into ``sys.modules``. This file's name sorts BEFORE ``test_dryrun_*.py``
(alphabetically ``test_ba`` < ``test_dr``), and those files assert ``"modal" not in sys.modules``
GLOBALLY — so the ``_restore_modal_purity`` autouse fixture below undoes each test's own ``modal``
import afterward (mirrors ``test_dispatch_is_spawned.py::
test_spawn_is_async_and_remote_is_sync_in_the_installed_sdk``), keeping that guard meaningful for
files that run later instead of quietly weakening it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _restore_modal_purity():
    preexisting = set(sys.modules)
    yield
    for name in [n for n in sys.modules if n == "modal" or n.startswith("modal.")]:
        if name not in preexisting:
            del sys.modules[name]

_BASE_CONFIG = """
training_dims: [768, 352, 25]
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
backup:
  enabled: {enabled}
  destination: hf
  repo_id: owner/repo
"""


def _config_text(*, enabled: bool) -> str:
    return _BASE_CONFIG.format(enabled=str(enabled).lower())


class _FakeVolume:
    """Stand-in for the Modal checkpoints Volume — ``.reload()``/``.commit()`` are local no-ops."""

    def reload(self) -> None:
        return None

    def commit(self) -> None:
        return None


def _write_checkpoint(root: Path, output_dir: str, step: int) -> Path:
    # A COMPLETE checkpoint dir per backup/plan.py's contract (adapter + training_state.pt) — the
    # only shape list_complete_checkpoints will select for backup.
    from signet_trainer.backup.plan import MIRRORED_ADAPTER_FILENAME, TRAINING_STATE_FILENAME

    d = root / output_dir / f"checkpoint-step-{step}"
    d.mkdir(parents=True, exist_ok=True)
    (d / MIRRORED_ADAPTER_FILENAME).write_bytes(b"x")
    (d / TRAINING_STATE_FILENAME).write_bytes(b"x")
    return d


# ---- issue #33 finding 1: backup.enabled is a $0 no-op backstop in BOTH fns ----


def test_backup_sync_noop_when_disabled(tmp_path, monkeypatch) -> None:
    from signet_trainer.modal import fns

    monkeypatch.setattr(fns, "checkpoints_vol", _FakeVolume())
    monkeypatch.setattr(fns, "CHECKPOINTS_DIR", tmp_path)
    # A checkpoint present on disk proves the fn returns BEFORE even planning a backup — a disabled
    # block must no-op regardless of what it would otherwise have selected.
    _write_checkpoint(tmp_path, "outputs", 100)

    result = fns.backup_sync.local(_config_text(enabled=False))

    assert "no-op" in result
    assert "enabled=False" in result


def test_restore_noop_when_disabled(tmp_path, monkeypatch) -> None:
    from signet_trainer.modal import fns

    monkeypatch.setattr(fns, "checkpoints_vol", _FakeVolume())
    monkeypatch.setattr(fns, "CHECKPOINTS_DIR", tmp_path)

    result = fns.restore.local(_config_text(enabled=False))

    assert "no-op" in result
    assert "enabled=False" in result


# ---- issue #23 finding 3: the list_repo_tree except is narrowed to the fresh-repo signal ----


def test_backup_sync_narrowed_except_propagates_unrelated_errors(tmp_path, monkeypatch) -> None:
    """A rate-limit / auth / 5xx failure must PROPAGATE, never be rendered as a known-empty remote."""
    from huggingface_hub import HfApi

    from signet_trainer.modal import fns

    monkeypatch.setattr(fns, "checkpoints_vol", _FakeVolume())
    monkeypatch.setattr(fns, "CHECKPOINTS_DIR", tmp_path)
    _write_checkpoint(tmp_path, "outputs", 100)

    monkeypatch.setattr(HfApi, "create_repo", lambda self, **kw: None)
    monkeypatch.setattr(HfApi, "repo_info", lambda self, **kw: SimpleNamespace(private=True))

    def _boom(self, **kw):
        raise RuntimeError("simulated Hub 5xx / rate limit")

    monkeypatch.setattr(HfApi, "list_repo_tree", _boom)

    with pytest.raises(RuntimeError, match="simulated Hub 5xx"):
        fns.backup_sync.local(_config_text(enabled=True))


def test_backup_sync_still_treats_fresh_repo_signal_as_empty(tmp_path, monkeypatch) -> None:
    """The ONE signal the narrowed except still swallows: a fresh/empty repo has no tree yet."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError

    from signet_trainer.modal import fns

    monkeypatch.setattr(fns, "checkpoints_vol", _FakeVolume())
    monkeypatch.setattr(fns, "CHECKPOINTS_DIR", tmp_path)
    _write_checkpoint(tmp_path, "outputs", 100)

    monkeypatch.setattr(HfApi, "create_repo", lambda self, **kw: None)
    monkeypatch.setattr(HfApi, "repo_info", lambda self, **kw: SimpleNamespace(private=True))

    def _fresh(self, **kw):
        raise EntryNotFoundError("no tree at this prefix yet")

    monkeypatch.setattr(HfApi, "list_repo_tree", _fresh)
    monkeypatch.setattr(HfApi, "upload_folder", lambda self, **kw: None)

    result = fns.backup_sync.local(_config_text(enabled=True))

    assert result == "[backup_sync] uploaded 1 dir(s) (hf); 0 already backed up."


# ---- issue #23 finding 2: restore reports the before/after DELTA, not a post-copy listing ----


def test_restore_reports_zero_new_when_nothing_matched(tmp_path, monkeypatch) -> None:
    """The exact repro from the issue: a restore that matches/downloads NOTHING must not report the
    Volume's PRE-EXISTING dirs as freshly 'restored'."""
    from signet_trainer.modal import fns

    monkeypatch.setattr(fns, "checkpoints_vol", _FakeVolume())
    monkeypatch.setattr(fns, "CHECKPOINTS_DIR", tmp_path)
    # 12 checkpoint dirs ALREADY on the "Volume" for this output_dir before restore runs.
    steps = list(range(100, 220, 10))
    assert len(steps) == 12
    for step in steps:
        _write_checkpoint(tmp_path, "outputs", step)

    # snapshot_download matches nothing (allow_patterns misses) and downloads nothing.
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: None)

    result = fns.restore.local(_config_text(enabled=True))

    assert "restored 0 dir(s) (new)" in result
    assert "12 already present" in result


def test_restore_reports_only_newly_downloaded_dirs_as_new(tmp_path, monkeypatch) -> None:
    from signet_trainer.modal import fns

    monkeypatch.setattr(fns, "checkpoints_vol", _FakeVolume())
    monkeypatch.setattr(fns, "CHECKPOINTS_DIR", tmp_path)
    _write_checkpoint(tmp_path, "outputs", 100)  # 1 dir already present before restore

    def _fake_snapshot_download(**kw):
        _write_checkpoint(tmp_path, "outputs", 200)  # simulates a NEW dir landing from the download

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)

    result = fns.restore.local(_config_text(enabled=True))

    assert "restored 1 dir(s) (new)" in result
    assert "1 already present" in result
