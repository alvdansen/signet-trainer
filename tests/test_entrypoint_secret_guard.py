"""Secret-name guard test (WR-01) — the step-1b guard fail-fasts PRE-approval on a name mismatch.

app.py builds the Modal app graph at MODULE-IMPORT time and captures the SIGNET_*_SECRET_NAME values
THEN, so the entrypoint's in-main ``os.environ[...] = cfg.modal.*_secret_name`` export was dead code
(it cannot re-bind an already-built graph). The replacement is a fail-fast guard: if the names app.py
captured differ from the loaded config, main() aborts at step 1b — before dry-run, cost, approval, or
dispatch — with an actionable message telling the operator to export the env var before ``modal run``.

Invokes the real ``main`` body (``main.info.raw_f``) with ``fns.train`` stubbed so no metered
``.spawn()`` can run. IMPORT DISCIPLINE (Anti-Pattern 6): modal-touching imports are done INSIDE the
test bodies, never at module top, so pytest COLLECTION never pulls ``modal`` into ``sys.modules`` and
the dry-run purity tests (test_dryrun_*.py) stay green.
"""

from __future__ import annotations

import pytest

_MATCHING_CONFIG = """
training_dims: [768, 352, 25]
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
"""

_MISMATCH_CONFIG = """
training_dims: [768, 352, 25]
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
modal:
  huggingface_secret_name: "some-other-account-hf-secret"
"""

_HF_GATED_MISMATCH_CONFIG = """
training_dims: [768, 352, 25]
data:
  preprocessed_data_root: "precomputed"
training:
  max_steps: 10
modal:
  hf_gated_secret_name: "some-other-account-gated-secret"
"""


class _RecordingCall:
    """Stand-in for the ``modal.FunctionCall`` ``.spawn()`` returns (D-10-DEF-17)."""

    object_id = "fc-test-stub"

    def get(self, timeout: float | None = None) -> None:
        return None


class _RecordingFn:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def spawn(self, *args, **kwargs) -> _RecordingCall:
        self.calls.append((args, kwargs))
        return _RecordingCall()


def _raw_main():
    from signet_trainer.modal import entrypoint  # lazy: keep modal out of collection-time sys.modules

    return entrypoint, entrypoint.main.info.raw_f


def _stub_train(monkeypatch) -> _RecordingFn:
    from signet_trainer.modal import fns  # lazy import (see module docstring)

    rec = _RecordingFn()
    monkeypatch.setattr(fns, "train", rec)
    return rec


def _write_config(tmp_path, text: str) -> str:
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text(text, encoding="utf-8")
    return str(cfg_path)


def test_secret_name_mismatch_aborts_before_dispatch(tmp_path, monkeypatch) -> None:
    # app.py captured the DEFAULT names at import; a config naming a different secret must fail fast
    # at step 1b (before dry-run / cost / approval / dispatch), not silently proceed (WR-01).
    entrypoint, raw_main = _raw_main()
    rec = _stub_train(monkeypatch)
    # If the guard did NOT fire we would reach the dry-run — make that loud so the test can't pass
    # for the wrong reason.
    monkeypatch.setattr(
        entrypoint, "run_dryrun", lambda cfg: pytest.fail("reached dry-run — 1b guard did not fire")
    )

    with pytest.raises(SystemExit) as exc:
        raw_main(config=_write_config(tmp_path, _MISMATCH_CONFIG), approve=True, mode="train")

    assert "secret-name mismatch" in str(exc.value)
    assert rec.calls == [], "a secret-name mismatch must abort before any dispatch"


def test_hf_gated_secret_name_mismatch_aborts_before_dispatch(tmp_path, monkeypatch) -> None:
    # Issue #33 finding 3: hf_gated_secret_name was the ONE secret name with no config field and no
    # step-1b coverage — app.py's app graph resolves it for EVERY mode's dispatch (not "--mode fuse
    # only" as the README used to claim), so a mismatch must abort here exactly like the other two
    # secret names, before dry-run/cost/approval/dispatch.
    entrypoint, raw_main = _raw_main()
    rec = _stub_train(monkeypatch)
    monkeypatch.setattr(
        entrypoint, "run_dryrun", lambda cfg: pytest.fail("reached dry-run — 1b guard did not fire")
    )

    with pytest.raises(SystemExit) as exc:
        raw_main(config=_write_config(tmp_path, _HF_GATED_MISMATCH_CONFIG), approve=True, mode="train")

    assert "secret-name mismatch" in str(exc.value)
    assert "SIGNET_HF_GATED_SECRET_NAME" in str(exc.value)
    assert rec.calls == [], "a secret-name mismatch must abort before any dispatch"


def test_matching_secret_names_pass_the_1b_guard(tmp_path, monkeypatch) -> None:
    # The default config matches app.py's captured names -> the guard passes and control reaches the
    # dry-run gate (which we stub to abort with a sentinel rc, proving we got PAST 1b).
    entrypoint, raw_main = _raw_main()
    _stub_train(monkeypatch)
    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg: 7)  # sentinel non-zero -> SystemExit(7)

    with pytest.raises(SystemExit) as exc:
        raw_main(config=_write_config(tmp_path, _MATCHING_CONFIG), approve=True, mode="train")

    assert exc.value.code == 7, "matching secret names must pass 1b and reach the dry-run gate"
