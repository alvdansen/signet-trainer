"""Phase 09.1 — parallel-watcher durability hardening (AUDIT #1 / #2 / #4), CPU source-scan.

Zero GPU / zero modal / zero spend: these tests READ the two watcher scripts' source and assert
the structural money-safety invariants the audit demands, without importing them (importing would
run their module-level ``load_config`` + ``sys.exit`` usage). The watchers drive METERED renders,
so their correctness is verified by static structure, not by execution.

Scope (updated audit 2026-08-11 — Wave 2 #6 has LANDED):
  * #1 (detached dispatch + artifact-freshness gate) — BOTH watchers now. The generalized watcher's
    original attached-dispatch scope guard was written against the pre-D-10-DEF-17 synchronous
    ``.remote()`` dispatch; after the ``.spawn()`` change an attached watcher re-dispatched and
    re-billed the same render every poll. Detach + freshness landed together, as the coupling rule
    demands.
  * #2 (success-gated bookkeeping) + #4 (pre-dispatch session-cap gate) — BOTH watchers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORK = REPO / "scripts" / "_watch_campaign_parallel_inference.py"
GENERALIZED = REPO / "scripts" / "watch_parallel_inference.py"
SUPERVISOR = REPO / "scripts" / "watcher_supervisor.ps1"


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    # Comment-stripped source-order scan (the plan's #2 acceptance): drop ``#`` comments so a token
    # mentioned in a comment can't be mistaken for real code. The watcher main() bodies carry no
    # ``#`` inside string literals, so a line-wise strip is exact here.
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())


def _func_body(path: Path, name: str) -> str:
    """Comment-stripped source of the named top-level function ONLY."""
    src = _src(path)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node) or ""
            return _strip_comments(seg)
    raise AssertionError(f"no {name}() function found in {path}")


def _main_body(path: Path) -> str:
    """Comment-stripped source of the script's ``main()`` function ONLY.

    Scoping to ``main()`` keeps the ``def dispatch_render``/``def append_spend`` definitions (which
    live above main) out of the order scan — inside main these names appear only as CALLS.
    """
    return _func_body(path, "main")


# ---- AUDIT #1 (Task 1) — CAMPAIGN FORK: detached dispatch + artifact-freshness gate ----


def test_fork_dispatches_detached():
    # #1a: the live campaign-fork watcher dispatches `modal run --detach` so a local client death cannot
    # tear down the render app mid-run.
    assert "--detach" in _src(FORK)


def test_generalized_dispatches_detached_with_freshness():
    # Wave 2 #6 LANDED (audit 2026-08-11): the generalized watcher now carries the SAME coupled
    # detach+freshness pair as the fork. The old scope guard asserted --detach ABSENT — that pin was
    # written when dispatch was attached-synchronous .remote(); after D-10-DEF-17 moved the
    # entrypoint to .spawn() + a dispatch_watch_seconds disengage, an attached watcher re-dispatched
    # and re-billed the same render every poll (the app teardown killed each spawned render). Detach
    # lands together with its freshness gate, never alone:
    s = _src(GENERALIZED)
    assert "--detach" in _func_body(GENERALIZED, "dispatch_render")
    assert "render_stall_minutes" in s, "freshness threshold must be config-derived (D-NOHARDCODE)"
    assert "RENDER_STALL_MIN" in s
    assert "render_landed(" in _main_body(GENERALIZED), "completion must be ARTIFACT-gated in main()"
    assert "pending_step" in _main_body(GENERALIZED), "single-flight pending state must exist"


def test_fork_detach_removes_client_kill_timeout():
    # #1a / OBS-01 T-09.1-10-T: the render client-kill root cause stays gone. The RENDER_TIMEOUT_MIN
    # constant is absent file-wide; and dispatch_render's BODY (the only place a render client-kill
    # could live) carries NO TimeoutExpired and NO render-timeout — it OPTS OUT of sh()'s read-only
    # volume-op timeout via an explicit `timeout=None`. (OBS-01 legitimately adds TimeoutExpired to
    # sh() for READ-ONLY volume ops, so the assertion is scoped to the render dispatch, not the file.)
    s = _strip_comments(_src(FORK))
    assert "RENDER_TIMEOUT_MIN" not in s
    dr = _func_body(FORK, "dispatch_render")
    assert "TimeoutExpired" not in dr, "no client-kill timeout branch may live in the render dispatch"
    assert "RENDER_TIMEOUT" not in dr
    assert "timeout=None" in dr, "the render dispatch must EXPLICITLY opt out of the volume-op timeout"


def test_fork_freshness_from_config():
    # #1c: the freshness threshold is config-derived (D-NOHARDCODE), read from
    # cfg.modal.render_stall_minutes — never a new hardcoded render-timeout literal.
    s = _src(FORK)
    assert "render_stall_minutes" in s
    assert "RENDER_STALL_MIN" in s


def test_fork_stall_message_is_artifact_freshness():
    # #1b: a hung DETACHED render is caught by Volume artifact-freshness and flagged with a
    # [watcher][STALL] line, then re-opened for re-dispatch (not by killing a client).
    s = _src(FORK)
    assert "[watcher][STALL]" in s
    assert "no new committed render artifact" in s


# ---- AUDIT #2 (Task 2) — success-gated bookkeeping (BOTH watchers) ----

# issue #19 item 1 — FORK is not published in this repo. A hard `(FORK, GENERALIZED)` tuple meant
# every `for path in _WATCHERS: _main_body(path)` loop below raised FileNotFoundError on iteration
# 1 (FORK) and never reached iteration 2 — so not one test_both_* assertion about the money-safety
# invariants was ever evaluated against scripts/watch_parallel_inference.py, the ONLY watcher this
# repo actually ships. Filter to files that exist: a missing fork degrades to "fork not checked"
# instead of "nothing checked", and the shipped watcher's own compliance is asserted regardless of
# whether the fork ever lands. The generalized watcher itself must never be the one that goes
# missing — that would silently zero out this entire suite the same way the fork's absence did.
assert GENERALIZED.is_file(), "the shipped generalized watcher must always be present"
_WATCHERS = tuple(p for p in (FORK, GENERALIZED) if p.is_file())


def _first(body: str, needle: str) -> int:
    i = body.find(needle)
    assert i != -1, f"expected {needle!r} in main() body"
    return i


def test_both_success_gate_rendered_after_dispatch():
    # #2: the `dispatch_render(` CALL must precede any `rendered.update`/`rendered.add` mutation —
    # i.e. mark rendered on SUCCESS, never rendered-before-dispatch. (`rendered.discard`, the
    # freshness re-open, is intentionally NOT a forbidden mutation.)
    for path in _WATCHERS:
        body = _main_body(path)
        di = _first(body, "dispatch_render(")
        for mut in ("rendered.update(", "rendered.add("):
            j = body.find(mut)
            if j != -1:
                assert di < j, f"{path.name}: {mut} must come AFTER the dispatch_render() call"


def test_both_final_done_guarded_by_success():
    # #2: the "final checkpoint rendered — done." exit must be guarded by actual dispatch success,
    # never printed unconditionally after a failed/queued final render.
    gen = _main_body(GENERALIZED)
    assert "if ok and step >= MAX_STEPS" in gen
    fork = _main_body(FORK)
    # The fork's final-done fires only inside the artifact-freshness SUCCESS branch (a new committed
    # artifact was detected), gated on the outstanding render being the final one.
    assert "was_final = outstanding_step >= MAX_STEPS" in fork
    assert "if was_final:" in fork


# ---- AUDIT #4 (Task 2) — pre-dispatch session-cap gate (BOTH watchers) ----


def test_both_import_session_cap_api():
    for path in _WATCHERS:
        s = _src(path)
        assert "from signet_trainer.modal.session_cap import" in s
        assert "read_ledger" in s
        assert "session_cap_check" in s


def test_both_cap_checked_before_dispatch():
    # #4: read_ledger + session_cap_check must fire BEFORE the dispatch_render() call.
    for path in _WATCHERS:
        body = _main_body(path)
        assert "read_ledger(" in body
        assert _first(body, "session_cap_check(") < _first(body, "dispatch_render(")


def test_both_append_spend_every_dispatch_not_only_on_success():
    # #2 ORIGINAL SHAPE: append_spend used to fire on the dispatch path (before the `if
    # dispatch_render(...)` success branch), so a killed render's spend was still ledgered — not
    # solely inside the success block.
    #
    # Issue #37 finding 1/2 (single-source accounting): GENERALIZED no longer books its own spend —
    # its local `append_spend(step)` wrapper was deliberately removed, and every dispatch it makes
    # now gets booked INSIDE the entrypoint-gate subprocess (entrypoint.py's `_watch_dispatch`,
    # covered by test_entrypoint_session_cap.py), unconditionally once `.spawn()` succeeds — i.e.
    # still "not only on success" of the RENDER, just realized in a different process. For a watcher
    # that still owns its own booking (e.g. a reinstated campaign fork), the original ordering
    # invariant still applies and is asserted below.
    for path in _WATCHERS:
        body = _main_body(path)
        if "append_spend(" not in body:
            continue
        assert _first(body, "append_spend(") < _first(body, "dispatch_render(")


# ---- issue #45 PR-1 must-fix #2 — checkpoint captured BEFORE dispatch, refuse on None -----------
# (GENERALIZED only: the campaign FORK does not exist in this repo and is not in this fix's scope.)


def test_generalized_captures_checkpoint_before_booking_spend():
    # The regression: `ckpt_at_dispatch = latest_checkpoint_name()` must run BEFORE the render is
    # dispatched, never after — a failed capture (None, on a modal volume ls timeout) must be caught
    # before anything is booked or dispatched, not discovered afterward with the spend already
    # ledgered.
    #
    # Issue #37 finding 1/2 moved the actual ledger write server-side (GENERALIZED no longer calls
    # `append_spend` at all — see test_both_append_spend_every_dispatch_not_only_on_success), so the
    # gate this test now pins is `ckpt_at_dispatch` preceding `dispatch_render(` — dispatch_render()
    # is what triggers the entrypoint subprocess that books the spend, so ordering the checkpoint
    # capture before IT preserves the original CAPTURE-BEFORE-BOOK property end to end.
    body = _main_body(GENERALIZED)
    assert "ckpt_at_dispatch = latest_checkpoint_name()" in body
    assert "append_spend(" not in body, (
        "GENERALIZED must not reintroduce its own ledger write (issue #37 finding 1/2 — booking is "
        "single-sourced inside the entrypoint gate subprocess dispatch_render() triggers)"
    )
    assert _first(body, "ckpt_at_dispatch = latest_checkpoint_name()") < _first(body, "dispatch_render(")


def test_generalized_refuses_dispatch_on_unresolved_checkpoint():
    # A None capture must REFUSE the dispatch — no append_spend, no dispatch_render — rather than
    # booking spend for (and dispatching) a render this watcher could never verify as landed.
    body = _main_body(GENERALIZED)
    guard_i = _first(body, "ckpt_at_dispatch is None")
    # Isolate the guard's own if-branch: everything up to the following `else:` at the same
    # dispatch-block indentation (the branch that DOES proceed to append_spend/dispatch_render).
    refusal_branch = body[guard_i:body.find("\n            else:", guard_i)]
    assert "append_spend(" not in refusal_branch
    assert "dispatch_render(" not in refusal_branch
    assert "REFUSING" in refusal_branch


def test_generalized_render_landed_receives_the_captured_checkpoint():
    # render_landed must be called with the DISPATCH-TIME identity (`pending_checkpoint`), never a
    # fresh lookup — closing the drift hole must-fix #2 exists to close.
    body = _main_body(GENERALIZED)
    assert "render_landed(pending_step, pending_checkpoint)" in body


# ---- issue #45 PR-1 must-fix #1 — the stall clock refreshes on progress ----------------------
# (GENERALIZED only: the campaign FORK already has its own artifact-freshness gate, out of scope.)


def test_generalized_progress_probe_runs_before_staleness_is_measured():
    # The clock must be given the chance to reset BEFORE age_min is computed against it this same
    # iteration — reversing the order would measure staleness against a clock that could have just
    # been refreshed, one poll too late.
    body = _main_body(GENERALIZED)
    assert _first(body, "render_progress_artifacts(") < _first(body, "age_min = ")


def test_generalized_progress_refresh_resets_pending_since():
    # The regression itself: pending_since used to be assigned exactly ONCE in main() (at dispatch,
    # `pending_step, pending_since = step, time.time()`) and never touched again. A SECOND
    # reassignment — inside the progress-changed branch — is what makes the clock refresh on
    # evidence of life instead of counting straight from dispatch for the render's whole life.
    body = _main_body(GENERALIZED)
    reassignments = re.findall(r"pending_since[,\s]*=.*time\.time\(\)", body)
    assert len(reassignments) >= 2, (
        f"pending_since must be reassigned on progress in addition to the dispatch-time assignment "
        f"(found {len(reassignments)} reassignment(s), expected >= 2)"
    )


def test_generalized_stall_path_still_fires_without_progress():
    # A render is STALLED only when NO new artifact has appeared for render_stall_minutes — the
    # genuinely-hung-render path must survive the progress-refresh addition unchanged.
    body = _main_body(GENERALIZED)
    assert "elif age_min > RENDER_STALL_MIN:" in body
    stall_i = _first(body, "elif age_min > RENDER_STALL_MIN:")
    stall_branch = body[stall_i:body.find("\n            else:", stall_i)]
    assert "declaring STALLED" in stall_branch
    assert "pending_step, pending_checkpoint, pending_progress = None, None, frozenset()" in stall_branch


# ---- OBS-01 Task 1 — the four watcher-supervision tunables are documented config fields ----


def test_obs_tunables_exist_with_documented_defaults():
    # Task 1: the four config-first contracts the watchers + supervisor READ (D-NOHARDCODE — never a
    # code literal). Import here (not at module top) to keep the watcher-source scans import-free.
    from signet_trainer.config.schema import ModalConfig

    m = ModalConfig()
    assert m.volume_op_timeout_s == 120.0
    assert m.watcher_heartbeat_stall_minutes == 15.0
    assert m.watcher_relaunch_backoff_seconds == 30.0
    assert m.watcher_relaunch_cap == 20
    for name in ("volume_op_timeout_s", "watcher_heartbeat_stall_minutes",
                 "watcher_relaunch_backoff_seconds", "watcher_relaunch_cap"):
        assert ModalConfig.model_fields[name].description, f"{name} must be documented"


# ---- OBS-01 Task 2 (D-OBS-4) — config-derived sh() timeout on READ-ONLY volume ops (BOTH) ----


def test_both_sh_timeout_config_derived_and_caught():
    # D-OBS-4: sh() runs volume ops under a config-derived timeout sourced from
    # cfg.modal.volume_op_timeout_s and CATCHES subprocess.TimeoutExpired so a hung `modal volume`
    # call can't wedge the single-threaded loop.
    for path in _WATCHERS:
        s = _src(path)
        assert "volume_op_timeout_s" in s, f"{path.name}: timeout must source from volume_op_timeout_s"
        assert "VOLUME_OP_TIMEOUT_S" in s
        sh = _func_body(path, "sh")
        assert "timeout=" in sh, f"{path.name}: sh() must pass a timeout"
        assert "except subprocess.TimeoutExpired" in sh, f"{path.name}: sh() must catch TimeoutExpired"


def test_both_render_dispatch_exempt_from_timeout():
    # D-OBS-4 scope guard (T-09.1-10-T): the render dispatch (detached fork / attached generalized)
    # MUST opt out of the volume-op timeout (timeout=None) — a render client-kill is forbidden.
    for path in _WATCHERS:
        dr = _func_body(path, "dispatch_render")
        assert "timeout=None" in dr, f"{path.name}: render dispatch must opt out of the volume-op timeout"
        assert "TimeoutExpired" not in dr, f"{path.name}: no client-kill branch in the render dispatch"


# ---- OBS-01 Task 2 (D-OBS-3) — heartbeat touched each poll iteration (BOTH) ----


def test_both_touch_heartbeat_in_loop():
    for path in _WATCHERS:
        s = _src(path)
        assert "HEARTBEAT_FILE" in s, f"{path.name}: must define a heartbeat file"
        body = _main_body(path)
        assert "HEARTBEAT_FILE.touch()" in body, f"{path.name}: must touch the heartbeat in the loop"


# ---- issue #19 item 5 — the heartbeat must reflect the PROCESS, not the poll boundary -----------
# (GENERALIZED only: the FORK's own long call, if it has one, is out of this fix's scope.)


def test_generalized_starts_a_background_heartbeat_thread_before_the_poll_loop():
    # The regression: HEARTBEAT_FILE.touch() fired only once per poll ITERATION, so a single
    # blocking refresh_grid() call (timeout=None — a Volume fetch + grid rebuild) past
    # watcher_heartbeat_stall_minutes reads as "wedged" to the supervisor while the watcher is
    # healthy. A daemon thread touching the SAME file independently of the poll loop closes the gap
    # for the FIRST long call too — it must start BEFORE the `while time.time() < deadline:` loop.
    body = _main_body(GENERALIZED)
    assert "threading.Thread(" in body, "GENERALIZED must start a background heartbeat thread"
    assert "target=_heartbeat_loop" in body
    assert "daemon=True" in body, "the heartbeat thread must be a daemon (no explicit stop needed)"
    thread_start_i = _first(body, "threading.Thread(")
    loop_i = _first(body, "while time.time() < deadline:")
    assert thread_start_i < loop_i, (
        "the heartbeat thread must start BEFORE the poll loop, so the FIRST refresh_grid() call is "
        "already covered"
    )


def test_heartbeat_loop_touches_repeatedly_and_stops_on_signal(tmp_path):
    """Runtime behavioral proof (AST-extracted, no modal import — mirrors
    test_entrypoint_preprocess_mode.py's convention for a small pure helper): ``_heartbeat_loop``
    actually touches the file more than once at a short interval, and returns promptly once its
    ``stop`` Event is set — it does not busy-loop or leak past shutdown.
    """
    import threading
    import time as _time

    src = _src(GENERALIZED)
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_heartbeat_loop"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, "<extracted>", "exec"), namespace)  # noqa: S102 — test-local exec
    heartbeat_loop = namespace["_heartbeat_loop"]

    hb_file = tmp_path / "_watcher_heartbeat_test"
    stop = threading.Event()
    t = threading.Thread(target=heartbeat_loop, args=(stop, hb_file, 0.02), daemon=True)
    t.start()
    _time.sleep(0.09)
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive(), "_heartbeat_loop must return promptly once `stop` is set"
    assert hb_file.exists(), "_heartbeat_loop must have touched the heartbeat file at least once"


# ---- OBS-01 Task 2 (D-OBS-1) — Volume-derived `rendered` seed (crash-resumable) ----


def test_generalized_seeds_rendered_from_volume():
    # D-OBS-1: the generalized watcher no longer starts `rendered` from an empty set() alone — it is
    # seeded from committed Volume render artifacts so a relaunch is idempotent.
    body = _main_body(GENERALIZED)
    assert "rendered: set[int] = set()" not in body, "must not start rendered from an empty set alone"
    assert "committed_render_stamps()" in body or "_seed_steps" in body
    assert "list_checkpoint_steps()" in body


def test_fork_retains_volume_derived_seed():
    # D-OBS-1: the campaign fork keeps its existing Volume-derived reseed (idempotent relaunch).
    body = _main_body(FORK)
    assert "_seed_steps = list_checkpoint_steps()" in body
    assert "rendered: set[int] = {" in body


# ---- OBS-01 Task 2 (D-OBS-1/D-OBS-2) — distinct terminal exit codes + run-complete sentinel ----


def test_both_define_distinct_cap_stop_exit_code():
    for path in _WATCHERS:
        s = _src(path)
        assert "CAP_STOP_EXIT = 3" in s, f"{path.name}: must define the distinct cap-stop exit code"


def test_both_cap_stop_exits_distinct_not_zero():
    # cap-stop must exit with the DISTINCT CAP_STOP_EXIT (not 0, not a plain relaunchable crash) so the
    # supervisor treats it as "do NOT relaunch".
    for path in _WATCHERS:
        body = _main_body(path)
        cap_i = _first(body, "session cap reached")
        exit_i = body.find("sys.exit(CAP_STOP_EXIT)", cap_i)
        assert exit_i != -1, f"{path.name}: cap-stop path must sys.exit(CAP_STOP_EXIT)"


def test_both_run_complete_writes_sentinel_and_exits_zero():
    for path in _WATCHERS:
        body = _main_body(path)
        assert "SENTINEL_FILE.write_text(" in body, f"{path.name}: run-complete must write the sentinel"
        assert "sys.exit(0)" in body, f"{path.name}: run-complete must exit 0"


# ---- OBS-01 Task 3 (D-OBS-2) — watcher_supervisor.ps1 STATIC source-scans (never executed) ----
# The supervisor drives a METERED dispatcher, so it is verified by static structure, never run here.


def test_supervisor_exists_and_launches_detached():
    assert SUPERVISOR.exists(), "watcher_supervisor.ps1 must exist"
    s = SUPERVISOR.read_text(encoding="utf-8")
    # Detached launch idiom (copied from serve_gridwatch.ps1) — survives Claude-session death.
    assert "Start-Process" in s


def test_supervisor_relaunch_is_bounded():
    # D-OBS-1/threat-model E: a bounded relaunch cap + counter so a metered dispatcher can NEVER be
    # respawned unboundedly (fork-bomb / crash-loop guard).
    s = SUPERVISOR.read_text(encoding="utf-8")
    assert "watcher_relaunch_cap" in s
    assert "$RelaunchCap" in s
    assert "$relaunches" in s
    assert "relaunch cap reached" in s


def test_supervisor_stop_conditions_sentinel_and_cap_stop():
    # Stops on the run-complete sentinel AND on the DISTINCT cap-stop exit code (reuses session_cap via
    # the watcher's rc), and relaunches other crashes with backoff.
    s = SUPERVISOR.read_text(encoding="utf-8")
    assert "SentinelFile" in s and "sentinelPath" in s
    assert "$CapStopExit" in s
    assert "= 3" in s, "the cap-stop exit-code contract (3) must be present"
    assert "NOT relaunching" in s
    assert "backing off" in s  # crash path backs off then relaunches


def test_supervisor_heartbeat_stall_distinct_from_train_render():
    # D-OBS-3: a heartbeat-age kill+relaunch path, logged as WATCHER-stall (distinct from
    # training/render stalls).
    s = SUPERVISOR.read_text(encoding="utf-8")
    assert "WATCHER-stall" in s
    assert "Stop-Process" in s  # kill on stall
    assert "LastWriteTime" in s  # heartbeat AGE gate


def test_supervisor_thresholds_are_config_first():
    # D-NOHARDCODE: the three thresholds are SOURCED from the signet config (load_config), never
    # hardcoded literals in the .ps1.
    s = SUPERVISOR.read_text(encoding="utf-8")
    assert "load_config" in s
    for field in ("watcher_heartbeat_stall_minutes", "watcher_relaunch_backoff_seconds",
                  "watcher_relaunch_cap"):
        assert field in s, f"{field} must be sourced config-first in the supervisor"


def test_supervisor_is_ascii_only():
    # Windows PowerShell 5.1 without a BOM reads .ps1 as ANSI; a UTF-8 em-dash then decodes to a stray
    # quote and breaks string parsing. Keep the supervisor pure-ASCII (matches serve_gridwatch.ps1).
    raw = SUPERVISOR.read_bytes()
    assert all(b < 128 for b in raw), "watcher_supervisor.ps1 must be pure ASCII (no em-dashes/curly quotes)"


# ---- issue #19 item 3 (D-NOHARDCODE) — no script under scripts/ restates the checkpoints Volume
# name as a literal; every read must route through cfg.modal.checkpoints_volume_name ----------


def test_no_hardcoded_checkpoints_volume_literal_under_scripts():
    # Anti-regression, pinned exactly as the issue's Proposed Direction #3 asks: this suite already
    # source-scans these files, so a future edit that reintroduces the literal (instead of routing
    # through cfg.modal.checkpoints_volume_name / CKPT_VOL) is caught here. Scoped to CODE, not the
    # comment on watch_parallel_inference.py that documents the historical bug by name.
    for path in sorted(REPO.glob("scripts/*.py")):
        code = _strip_comments(path.read_text(encoding="utf-8"))
        assert "signe-trainer-" not in code, (
            f"{path.name}: must not hardcode a \"signe-trainer-*\" Volume/App literal — read "
            "cfg.modal.checkpoints_volume_name (or .app_name) instead (D-NOHARDCODE)"
        )
