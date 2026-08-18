"""Issue #33 findings 1 + 2: the backup/restore pre-approval refusal, and ``-d`` detach detection.

Two independent behavioral checks against the real ``entrypoint.py``:

  * ``--mode backup`` / ``--mode restore`` with ``backup.enabled=False`` (the shipped default) must
    abort BEFORE the dry-run gate / cost print / approval pause — a $0 refusal, never a metered CPU
    container that crashes inside ``api.create_repo(repo_id=None, ...)``.
  * ``_detach_requested()`` must recognize modal's documented ``-d`` short alias for ``--detach``,
    not just the long form — otherwise a genuinely detached run is told (falsely) that it is LOST.

IMPORT DISCIPLINE (Anti-Pattern 6): ``signet_trainer.modal.entrypoint`` (and therefore ``modal``) is
imported lazily inside each test body, mirroring ``test_entrypoint_secret_guard.py`` /
``test_entrypoint_gate_behavioral.py`` — both of those files also sort AFTER ``test_dryrun_*.py``
alphabetically, so (like this file) they need no ``sys.modules`` cleanup to keep the dry-run purity
guard (``"modal" not in sys.modules``) meaningful for files that run later.
"""

from __future__ import annotations

import builtins

import pytest

_DISABLED_BACKUP_CONFIG = """
training_dims: [768, 352, 25]
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
"""

_ENABLED_BACKUP_CONFIG = """
training_dims: [768, 352, 25]
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
backup:
  enabled: true
  destination: hf
  repo_id: owner/repo
"""


class _RecordingCall:
    object_id = "fc-test-stub"

    def get(self, timeout: float | None = None) -> None:
        return None


class _RecordingFn:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def spawn(self, *args, **kwargs) -> _RecordingCall:
        self.calls.append((args, kwargs))
        return _RecordingCall()


def _write_config(tmp_path, text: str) -> str:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    return str(cfg_path)


def _raw_main():
    from signet_trainer.modal import entrypoint  # lazy: keep modal out of collection-time sys.modules

    return entrypoint, entrypoint.main.info.raw_f


def _stub_backup_fns(monkeypatch) -> tuple[_RecordingFn, _RecordingFn]:
    from signet_trainer.modal import fns

    backup_rec = _RecordingFn()
    restore_rec = _RecordingFn()
    monkeypatch.setattr(fns, "backup_sync", backup_rec)
    monkeypatch.setattr(fns, "restore", restore_rec)
    return backup_rec, restore_rec


# ---- issue #33 finding 1: disabled backup/restore refuses pre-approval, at $0 ----


@pytest.mark.parametrize("mode", ["backup", "restore"])
def test_disabled_backup_mode_aborts_before_dry_run(tmp_path, monkeypatch, mode) -> None:
    entrypoint, raw_main = _raw_main()
    backup_rec, restore_rec = _stub_backup_fns(monkeypatch)
    # If the guard did not fire we would reach the dry-run gate — make that loud so the test cannot
    # pass for the wrong reason (e.g. a refusal that happens to also raise SystemExit later, post-cost).
    monkeypatch.setattr(
        entrypoint, "run_dryrun", lambda cfg: pytest.fail("reached dry-run — 1c guard did not fire")
    )

    with pytest.raises(SystemExit) as exc:
        raw_main(
            config=_write_config(tmp_path, _DISABLED_BACKUP_CONFIG), approve=True, mode=mode
        )

    assert f"--mode {mode}" in str(exc.value)
    assert "backup.enabled=True" in str(exc.value)
    assert backup_rec.calls == [] and restore_rec.calls == [], (
        "a disabled backup block must dispatch NOTHING — no metered CPU container may boot"
    )


def test_enabled_backup_mode_passes_the_gate_and_reaches_dry_run(tmp_path, monkeypatch) -> None:
    # The companion positive case: an ENABLED block must NOT be refused by the new 1c guard — it
    # must reach the dry-run gate exactly like every other mode (proving the guard is enabled-gated,
    # not a blanket refusal of backup/restore).
    entrypoint, raw_main = _raw_main()
    _stub_backup_fns(monkeypatch)
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg: 7)  # sentinel -> SystemExit(7)

    with pytest.raises(SystemExit) as exc:
        raw_main(config=_write_config(tmp_path, _ENABLED_BACKUP_CONFIG), approve=True, mode="backup")

    assert exc.value.code == 7, "an enabled backup block must pass the 1c guard and reach dry-run"


def test_disabled_backup_mode_never_prompts_for_approval(tmp_path, monkeypatch) -> None:
    # The refusal must precede _require_approval too — no interactive prompt for a run that never
    # had a chance of dispatching.
    entrypoint, raw_main = _raw_main()
    _stub_backup_fns(monkeypatch)
    monkeypatch.setattr(
        builtins, "input", lambda *a, **k: pytest.fail("approval prompt reached — 1c guard did not fire")
    )

    with pytest.raises(SystemExit):
        raw_main(config=_write_config(tmp_path, _DISABLED_BACKUP_CONFIG), approve=False, mode="restore")


def test_disabled_backup_does_not_affect_train_mode(tmp_path, monkeypatch) -> None:
    # The guard is mode-scoped (backup/restore only) — a disabled backup block must not block an
    # unrelated --mode train run (the shipped default carries backup.enabled=False everywhere).
    entrypoint, raw_main = _raw_main()
    from signet_trainer.modal import fns

    train_rec = _RecordingFn()
    monkeypatch.setattr(fns, "train", train_rec)
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg: 0)

    raw_main(config=_write_config(tmp_path, _DISABLED_BACKUP_CONFIG), approve=True, mode="train")

    assert len(train_rec.calls) == 1, "backup.enabled=False must never block --mode train"


# ---- issue #33 finding 2: -d is a documented alias for --detach ----


def test_detach_requested_recognizes_the_long_form(monkeypatch) -> None:
    entrypoint, _ = _raw_main()
    monkeypatch.setattr(
        "sys.argv", ["modal", "run", "--detach", "-m", "signet_trainer.modal.entrypoint"]
    )
    assert entrypoint._detach_requested() is True


def test_detach_requested_recognizes_the_short_alias(monkeypatch) -> None:
    # This is the exact defect: modal's own CLI accepts `-d` as `--detach`
    # (`@click.option("-d", "--detach", ...)` in modal/cli/run.py); a plain
    # `"--detach" in sys.argv` string match misses it entirely.
    entrypoint, _ = _raw_main()
    monkeypatch.setattr(
        "sys.argv", ["modal", "run", "-d", "-m", "signet_trainer.modal.entrypoint"]
    )
    assert entrypoint._detach_requested() is True


def test_detach_requested_false_when_absent(monkeypatch) -> None:
    entrypoint, _ = _raw_main()
    monkeypatch.setattr("sys.argv", ["modal", "run", "-m", "signet_trainer.modal.entrypoint"])
    assert entrypoint._detach_requested() is False


def test_detach_requested_does_not_false_positive_on_a_long_option():
    # A long option's substring must never falsely trip the short-cluster branch (e.g. --dry-run
    # carries a "d" but is not a detach flag).
    entrypoint, _ = _raw_main()
    import sys as _sys

    old = _sys.argv
    try:
        _sys.argv = ["modal", "run", "--dry-run", "-m", "signet_trainer.modal.entrypoint"]
        assert entrypoint._detach_requested() is False
    finally:
        _sys.argv = old


def test_watch_dispatch_honors_the_short_alias_not_just_the_long_form(monkeypatch) -> None:
    # Behavioral proof at the _watch_dispatch level (not just the helper in isolation): with only
    # `-d` on argv, the run must be treated as DETACHED — no false "--detach was NOT passed" warning,
    # and on expiry the "still RUNNING, not cancelled" message rather than "treat this run as LOST".
    entrypoint, _ = _raw_main()
    monkeypatch.setattr(
        "sys.argv", ["modal", "run", "-d", "-m", "signet_trainer.modal.entrypoint"]
    )

    class _ExpiringCall:
        object_id = "fc-detach-test"

        def get(self, timeout: float | None = None) -> None:
            raise TimeoutError

    printed: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))

    entrypoint._watch_dispatch(_ExpiringCall(), 0.01, "backup_sync")

    joined = "\n".join(printed)
    assert "NOT passed" not in joined, "-d must be recognized as --detach in the pre-watch warning"
    assert "LOST" not in joined, "-d must be recognized as --detach in the post-timeout message"
    assert "NOT cancelled" in joined
