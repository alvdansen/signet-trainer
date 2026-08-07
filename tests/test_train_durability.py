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
