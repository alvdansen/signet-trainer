"""AUDIT #34 direction 7 — the acceptance-failed marker's KEY (train/acceptance_marker.py).

Stdlib-only, imported directly (no ``modal`` in ``sys.modules`` — see the module's own docstring
for why this logic does not live in ``modal/fns.py``). ``h3_train``'s wiring (marker checked BEFORE
the 61.7 GiB load, written BEFORE the zero-delta raise, Volume commit/reload around both) is a
separate structural concern pinned in ``tests/test_h3_acceptance_marker_wiring.py`` via a source
scan that never imports ``modal/fns.py`` either.

The property this file exists to prove: the marker blocks a genuine retry of the SAME deterministic
failure, and does NOT block a later legitimate run — a different output_dir, or the same
output_dir after real further training moved the checkpoint past the recorded step.
"""

from __future__ import annotations

from signet_trainer.train.acceptance_marker import (
    ACCEPTANCE_FAILED_MARKER_NAME,
    acceptance_marker_blocks,
    acceptance_marker_is_current,
    clear_acceptance_failed_marker,
    marker_path,
    read_acceptance_failed_marker,
    write_acceptance_failed_marker,
)
from signet_trainer.train.checkpoint import ADAPTER_FILENAME, TRAINING_STATE_FILENAME, CheckpointManager


# --------------------------------------------------------------------------------------------------
# write / read / marker_path
# --------------------------------------------------------------------------------------------------


def test_marker_path_is_a_dot_file_never_matching_the_checkpoint_prefix(tmp_path) -> None:
    """Must never look like ``checkpoint-step-*`` — CheckpointManager's own regex would otherwise
    try to parse it as a checkpoint dir (or worse, prune()/rmtree it)."""
    path = marker_path(tmp_path)
    assert path.name == ACCEPTANCE_FAILED_MARKER_NAME
    assert not path.name.startswith("checkpoint-step-")


def test_write_then_read_round_trips_step_and_delta(tmp_path) -> None:
    write_acceptance_failed_marker(tmp_path, step=250, delta=0.0)
    recorded = read_acceptance_failed_marker(tmp_path)
    assert recorded == {"step": 250, "delta": 0.0}


def test_read_returns_none_when_no_marker_written(tmp_path) -> None:
    assert read_acceptance_failed_marker(tmp_path) is None


def test_write_creates_the_output_dir_if_missing(tmp_path) -> None:
    """``h3_train`` may call this before anything else has created the run's checkpoint dir on a
    from-scratch config — a container-boot-cheap re-raise must not depend on that dir pre-existing."""
    fresh = tmp_path / "not-yet-created" / "nested"
    write_acceptance_failed_marker(fresh, step=1, delta=0.0)
    assert marker_path(fresh).exists()


def test_clear_is_a_no_op_when_no_marker_exists(tmp_path) -> None:
    clear_acceptance_failed_marker(tmp_path)  # must not raise
    assert read_acceptance_failed_marker(tmp_path) is None


def test_clear_removes_a_written_marker(tmp_path) -> None:
    write_acceptance_failed_marker(tmp_path, step=10, delta=0.0)
    clear_acceptance_failed_marker(tmp_path)
    assert read_acceptance_failed_marker(tmp_path) is None


# --------------------------------------------------------------------------------------------------
# acceptance_marker_is_current — the STEP half of the key
# --------------------------------------------------------------------------------------------------


def test_marker_is_current_when_latest_step_matches_recorded() -> None:
    """The real-retry case: a Modal-driven retry boots fresh but the Volume's checkpoint dir is
    untouched, so the latest step is still exactly what it was when the marker was written."""
    assert acceptance_marker_is_current(recorded_step=800, latest_step=800) is True


def test_marker_is_stale_when_latest_step_has_advanced() -> None:
    """A legitimate new attempt: real training happened since (e.g. an operator config fix
    followed by a continued run), so the checkpoint moved past the recorded step."""
    assert acceptance_marker_is_current(recorded_step=800, latest_step=1000) is False


def test_marker_is_stale_when_no_checkpoint_exists_at_all() -> None:
    """``latest_step=None`` (cold start) must never equal an int — a marker from a run whose
    checkpoint dir was reset/deleted must not silently re-raise against a run that has nothing."""
    assert acceptance_marker_is_current(recorded_step=800, latest_step=None) is False


def test_marker_is_stale_when_latest_step_is_lower_than_recorded() -> None:
    """Defensive: a lower latest step (dir reset to an earlier checkpoint) is also "not current" —
    only an EXACT match is honoured, never a >= or <= comparison that could mask a real change."""
    assert acceptance_marker_is_current(recorded_step=800, latest_step=799) is False


# --------------------------------------------------------------------------------------------------
# acceptance_marker_blocks — the PLAN half of the key (current AND recorded_step >= max_steps)
# --------------------------------------------------------------------------------------------------


def test_bumped_max_steps_is_not_blocked_even_though_the_marker_is_current() -> None:
    """The operator bumps training.max_steps and re-dispatches into the SAME output_dir: nothing
    has trained yet (the checkpoint is still at 800, so the marker is step-current), but this
    dispatch has more training work to do (max_steps=1600 > recorded step 800) than the dispatch
    that plateaued. Must NOT block — this is the bug the marker's docstring warned about."""
    assert acceptance_marker_blocks(recorded_step=800, latest_step=800, max_steps=1600) is False


def test_marker_at_max_steps_is_still_blocked() -> None:
    """The real retry case: the marker's recorded step already equals THIS dispatch's max_steps,
    so there is no further training this dispatch could ever do — the zero-delta verdict still
    applies and the cheap re-raise must fire."""
    assert acceptance_marker_blocks(recorded_step=1600, latest_step=1600, max_steps=1600) is True


def test_marker_past_max_steps_is_still_blocked() -> None:
    """Defensive: recorded_step > max_steps (e.g. max_steps lowered) is also blocked — the ">="
    comparison, not "==", is what the conjunct uses."""
    assert acceptance_marker_blocks(recorded_step=1600, latest_step=1600, max_steps=1000) is True


def test_stale_marker_is_not_blocked_regardless_of_max_steps() -> None:
    """If real training already moved the checkpoint past the recorded step, the marker is stale
    on the STEP half alone — the max_steps conjunct must never rescue a stale marker into blocking."""
    assert acceptance_marker_blocks(recorded_step=800, latest_step=1000, max_steps=800) is False


# --------------------------------------------------------------------------------------------------
# End-to-end against a REAL CheckpointManager — the two behaviours that matter operationally
# --------------------------------------------------------------------------------------------------


def test_end_to_end_retry_of_the_identical_failure_is_honoured(tmp_path) -> None:
    """h3_train life 1: trains to step 800 (final_step == the checkpoint just saved), measures
    delta == 0.0, writes the marker at step=800. h3_train life 2 (a Modal retry, fresh container,
    same Volume): resume() finds the SAME step-800 checkpoint (no new training happened), and the
    marker's step still matches latest_step() — the retry must be honoured (re-raise, no reload)."""
    mgr = CheckpointManager(tmp_path)
    (tmp_path / "checkpoint-step-00800-loss-1.0000").mkdir(parents=True)
    for fname in (ADAPTER_FILENAME, TRAINING_STATE_FILENAME):
        (tmp_path / "checkpoint-step-00800-loss-1.0000" / fname).touch()

    write_acceptance_failed_marker(tmp_path, step=800, delta=0.0)

    recorded = read_acceptance_failed_marker(tmp_path)
    assert acceptance_marker_is_current(recorded["step"], mgr.latest_step()) is True


def test_end_to_end_a_later_run_that_actually_trained_further_is_never_blocked(tmp_path) -> None:
    """Life 1 failed and left the marker at step 800. The operator fixes the LoRA target config and
    relaunches the SAME output_dir; this time training genuinely proceeds to step 1200. The next
    entry check must see the marker as stale (never re-raise against a run that has moved on)."""
    mgr = CheckpointManager(tmp_path)
    (tmp_path / "checkpoint-step-00800-loss-1.0000").mkdir(parents=True)
    for fname in (ADAPTER_FILENAME, TRAINING_STATE_FILENAME):
        (tmp_path / "checkpoint-step-00800-loss-1.0000" / fname).touch()
    write_acceptance_failed_marker(tmp_path, step=800, delta=0.0)

    # Real further training landed a new, higher-step checkpoint.
    (tmp_path / "checkpoint-step-01200-loss-0.9000").mkdir(parents=True)
    for fname in (ADAPTER_FILENAME, TRAINING_STATE_FILENAME):
        (tmp_path / "checkpoint-step-01200-loss-0.9000" / fname).touch()

    recorded = read_acceptance_failed_marker(tmp_path)
    assert acceptance_marker_is_current(recorded["step"], mgr.latest_step()) is False


def test_a_different_output_dir_never_sees_another_runs_marker(tmp_path) -> None:
    """The RUN-scoping half of the key: the marker lives inside ITS run's checkpoint dir, so a
    sibling run under a different output_dir is structurally unable to read it."""
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    write_acceptance_failed_marker(run_a, step=800, delta=0.0)

    assert read_acceptance_failed_marker(run_b) is None
