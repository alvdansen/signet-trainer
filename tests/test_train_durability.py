"""09.1-05 — durability proofs for ``train_loop``: commit-before-render (#12) + liveness watchdog (#11).

Two audit findings, both CPU-testable without modal / ltx_core / GPU:

  * **#12 (commit-before-render).** The checkpoint boundary used to save → render (~8-13 min) →
    commit, so a preemption DURING the render vanished the just-saved checkpoint (F9 resumes ~200
    steps back, ~$1/boundary). The fix commits IMMEDIATELY after ``ckpt_manager.save`` — before the
    ``on_checkpoint`` render — then commits AGAIN for the rendered mp4. Proven here by a source-order
    scan (comments/docstrings stripped) of ``train_loop``.
  * **#11 (liveness watchdog).** A wedged/slow cadence (no committed checkpoint for
    > ``checkpoint_expected_minutes * K``) had no in-code gate and burned to the 24h ceiling
    (~$34 overnight). The pure ``checkpoint_watchdog_exceeded`` helper raises instead → an F9 in-dir
    resume. Config-gated and OFF by default (``checkpoint_expected_minutes is None`` → False every
    step → byte-identical to today). Proven by a unit table + a source scan of the ``train_loop``
    wiring (reads the config knobs, raises via the helper, refreshes ``last_commit`` at commit).

Pure source-scan style mirrors the existing pure-helper tests (no ltx_core / modal / CUDA).
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

from signet_trainer.train.loop import checkpoint_watchdog_exceeded

_LOOP_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "train" / "loop.py"
).read_text(encoding="utf-8")


def _train_loop_name_seq() -> list[str]:
    """Return the ordered NAME tokens of ``train_loop`` with comments AND string literals stripped.

    Stripping STRING tokens drops docstrings (which mention ``commit`` / ``on_checkpoint`` in prose),
    so the sequence reflects only executable references — the load-bearing call order.
    """
    tree = ast.parse(_LOOP_SRC)
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_loop"
    )
    segment = ast.get_source_segment(_LOOP_SRC, fn)
    assert segment is not None
    names: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(segment).readline):
        if tok.type == tokenize.NAME:
            names.append(tok.string)
    return names


# --------------------------------------------------------------------------------------------------
# #12 — commit BEFORE the in-loop render (two commits per checkpoint boundary)
# --------------------------------------------------------------------------------------------------


def test_commit_before_and_after_render_order():
    names = _train_loop_name_seq()
    commit_idxs = [i for i, n in enumerate(names) if n == "commit"]
    # Two boundary commits + one final commit = exactly three commit() calls in train_loop.
    assert len(commit_idxs) == 3, f"expected 3 commit() calls, found {len(commit_idxs)}"

    # The FIRST on_checkpoint CALL is the render. In the name stream it is the second `on_checkpoint`
    # occurrence (the first is the `if on_checkpoint is not None` guard).
    on_ckpt_idxs = [i for i, n in enumerate(names) if n == "on_checkpoint"]
    assert len(on_ckpt_idxs) >= 2, "expected the on_checkpoint guard + the render call"
    render_call_idx = on_ckpt_idxs[1]

    c0, c1, c2 = commit_idxs
    # #12: a commit lands BEFORE the render (durability first) and another AFTER (the mp4).
    assert c0 < render_call_idx, "checkpoint must be committed BEFORE the on_checkpoint render (#12)"
    assert render_call_idx < c1, "a second commit must follow the render (lands the rendered mp4)"
    # The third commit is the post-loop final save — after the boundary block.
    assert c1 < c2, "the final save commit must come last"


def test_on_checkpoint_render_still_wrapped_in_try_except():
    # A failed render must not kill a metered round — the try/except wrapper is preserved.
    names = _train_loop_name_seq()
    assert "on_checkpoint" in names
    render_idx = names.index("on_checkpoint")
    # An `Exception` handler follows the callback region (swallowed render failure).
    assert "Exception" in names[render_idx:], "on_checkpoint render must stay inside try/except"


# --------------------------------------------------------------------------------------------------
# #11 — pure liveness-watchdog helper (config-gated, OFF by default)
# --------------------------------------------------------------------------------------------------


def test_watchdog_off_when_expected_none():
    # None expected → watchdog OFF → False at ANY elapsed (byte-identical current behavior).
    for elapsed in (0.0, 1.0, 10_000.0):
        assert checkpoint_watchdog_exceeded(elapsed, None, 2.5) is False


def test_watchdog_below_threshold_is_false():
    # (expected*K - 1) is still under the deadline → no raise.
    expected, k = 10.0, 2.5  # threshold = 25 min
    assert checkpoint_watchdog_exceeded(expected * k - 1, expected, k) is False


def test_watchdog_above_threshold_is_true():
    # (expected*K + 1) is past the deadline → wedged cadence → raise.
    expected, k = 10.0, 2.5  # threshold = 25 min
    assert checkpoint_watchdog_exceeded(expected * k + 1, expected, k) is True


def test_watchdog_exactly_at_threshold_is_false():
    # Strict '>' — exactly at the threshold is NOT yet exceeded.
    expected, k = 10.0, 2.5
    assert checkpoint_watchdog_exceeded(expected * k, expected, k) is False


def test_watchdog_multiplier_scales_threshold():
    # A larger K pushes the deadline out: 30 min trips at K=2 but not at K=4 (expected 10).
    assert checkpoint_watchdog_exceeded(30.0, 10.0, 2.0) is True
    assert checkpoint_watchdog_exceeded(30.0, 10.0, 4.0) is False


# --------------------------------------------------------------------------------------------------
# #11 — train_loop wiring: config-driven, no hardcoded literal, last_commit refreshed at commit
# --------------------------------------------------------------------------------------------------


def test_train_loop_wires_watchdog_from_config():
    names = _train_loop_name_seq()
    # Reads BOTH knobs from config (D-NOHARDCODE — no literal minute/K in the loop).
    assert "checkpoint_expected_minutes" in names, "train_loop must read the config threshold"
    assert "checkpoint_stall_multiplier" in names, "train_loop must read the config multiplier K"
    # Raises via the pure helper (not an inline comparison).
    assert "checkpoint_watchdog_exceeded" in names, "train_loop must gate on the pure helper"


def test_train_loop_refreshes_last_commit():
    names = _train_loop_name_seq()
    # last_commit is initialized once and refreshed at each commit — appears multiple times.
    assert names.count("last_commit") >= 3, "last_commit must be refreshed at each commit"
    # time.monotonic is the clock source.
    assert "monotonic" in names, "watchdog clock must use time.monotonic"


# --------------------------------------------------------------------------------------------------
# Issue #30 finding #3 — the watchdog's trip message must not promise a Modal retry off-Modal
# --------------------------------------------------------------------------------------------------


def _watchdog_trip_harness(monkeypatch: pytest.MonkeyPatch):
    """A CPU/stub ``train_loop`` call that deterministically trips the watchdog on step 1.

    ``time.monotonic`` is monkeypatched (rather than using a tiny ``expected_minutes`` and hoping
    real wall-clock elapses enough) because on this machine's clock resolution two calls
    microseconds apart can read back byte-identical, making a real-clock trip flaky.
    ``loop.py``'s ``time.monotonic`` is called exactly twice on the untripped step-1 path: once to
    seed ``last_commit`` before the loop, once inside the watchdog check -- so a 2-value sequence
    (0.0, then a huge jump) trips it every time with zero timing dependence.
    """
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    from signet_trainer.train import loop as loop_mod
    from signet_trainer.train.flow_match import FlowMatchingSchedule

    clock = iter([0.0, 10_000.0])
    monkeypatch.setattr(loop_mod.time, "monotonic", lambda: next(clock))

    param = nn.Parameter(torch.zeros(1))

    class _Model:
        def parameters(self):
            return iter([param])

    class _CkptManager:
        def resume(self, model, optimizer, scheduler):  # noqa: ANN001
            return 0

        def save(self, *a, **k):  # noqa: ANN002, ANN003
            pass

    class _Optim:
        def step(self):
            pass

        def zero_grad(self):
            pass

    class _Sched:
        def step(self):
            pass

    def _step_fn(model, batch, schedule, rng, *, device, dtype):  # noqa: ANN001
        return (param * 2).sum()  # a real graph so .backward() works

    config = SimpleNamespace(
        training=SimpleNamespace(
            mixed_precision="fp32", seed=42, max_steps=1,
            gradient_accumulation_steps=1, max_grad_norm=1.0, checkpoint_every=1000,
            checkpoint_expected_minutes=1.0, checkpoint_stall_multiplier=1.0,
        ),
    )
    dataset = [torch.zeros(1)]
    return dict(
        model=_Model(), dataset=dataset, optimizer=_Optim(), scheduler=_Sched(),
        schedule=FlowMatchingSchedule(uniform_prob=0.30), ckpt_manager=_CkptManager(),
        config=config, step_fn=_step_fn,
    )


def test_watchdog_trip_promises_modal_retry_only_when_checkpoints_vol_present(monkeypatch):
    # Issue #30 finding #3 check: on Modal (checkpoints_vol is not None) the F9 retry really
    # exists — the trip message may promise it.
    from signet_trainer.train.loop import train_loop

    h = _watchdog_trip_harness(monkeypatch)
    with pytest.raises(RuntimeError, match=r"the train\(\) retry resumes in-dir"):
        train_loop(
            h["model"], h["dataset"], h["optimizer"], h["scheduler"], h["schedule"],
            h["ckpt_manager"], h["config"], checkpoints_vol=object(), step_fn=h["step_fn"],
        )


def test_watchdog_trip_off_modal_tells_operator_to_rerun_not_a_fake_retry(monkeypatch):
    # Issue #30 finding #3 defect: off-Modal (checkpoints_vol=None, the local runner) there is NO
    # retry — the old message unconditionally promised "the train() retry resumes in-dir", which
    # is false off-Modal. The local-runner path must say to re-run manually instead, and must NOT
    # repeat the Modal-only promise.
    from signet_trainer.train.loop import train_loop

    h = _watchdog_trip_harness(monkeypatch)
    with pytest.raises(RuntimeError, match="local runner does not retry") as excinfo:
        train_loop(
            h["model"], h["dataset"], h["optimizer"], h["scheduler"], h["schedule"],
            h["ckpt_manager"], h["config"], checkpoints_vol=None, step_fn=h["step_fn"],
        )
    assert "the train() retry resumes in-dir" not in str(excinfo.value), (
        "the off-Modal trip message must not promise a retry that does not exist locally"
    )
