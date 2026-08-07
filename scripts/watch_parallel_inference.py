"""Parallel-inference watcher — OPT-IN venue, generalized (operator 2026-07-11).

Usage:
    python scripts/watch_parallel_inference.py <sample_config.yaml>

All run facts are read FROM the config (config-first, D-NOHARDCODE): output_dir locates the
checkpoints to watch, training.max_steps the finish line, modal.est_hours*hourly_rate the
per-render ledger estimate. Pass a RENDER config (est_hours = honest per-render figure), not
the training config.

Prior-campaign style: while training runs on its own A100, this LOCAL watcher polls the
checkpoints Volume and, for every NEW checkpoint-step-* dir, dispatches ONE gated
``--mode sample`` render (a second, separate single-A100 container — the single-A100-per-job
house rule holds within each job; the parallel venue is the operator's explicit call). Every dispatch
goes through the canonical entrypoint gate (cost print + --approve under the yolo session
pre-authorization) and is appended to the SESSION-STATE spend ledger. After each render the
newest samples dir is fetched and the local finetune-gridwatch grid is rebuilt, so
grid-output/index.html updates live.

Exits when the final checkpoint (step == MAX_STEPS) has been rendered, or on WATCH_DEADLINE.
Zero bespoke encode/launch paths — modal CLI + the single gate only.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from signet_trainer.config.load import load_config  # noqa: E402
from signet_trainer.inference.samples_layout import (  # noqa: E402
    expected_h3_render_key,
    landed_render_ids,
    samples_root,
)
from signet_trainer.modal.session_cap import (  # noqa: E402
    append_spend as _append_spend,
    read_ledger,
    session_cap_check,
)

SAMPLE_CONFIG = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: watch_parallel_inference.py <sample_config.yaml>")
# Cadence (optional 2nd arg): dispatch a render every N checkpoint steps, not every checkpoint.
# Default 500. The checkpoint cadence is a TRAINING knob (checkpoint_every) and the render cadence is
# an ECONOMIC one, so they are deliberately separate: at checkpoint_every 50 a render-per-checkpoint
# would be 60 metered renders per 3000-step round. It stays a CLI argument rather than a config key
# because `schema.py::_Base` sets `extra="forbid"` and schema.py is zero-edit — the same reason
# `--render-tier` is a dispatch argument (training-review §11).
RENDER_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 500
_cfg = load_config(SAMPLE_CONFIG)
OUTPUT_DIR = _cfg.output_dir
MAX_STEPS = _cfg.training.max_steps
# ⛔ FAMILY-AWARE RENDER ROOT — the fix for the blocker that made this watcher unusable on H3.
# `h3_sample` commits to `<output_dir>/samples_h3/<render key>/`; the LTX `sample` commits to
# `<output_dir>/samples/<UTC stamp>/`. This watcher previously hardcoded the LTX answer, so against
# an H3 config it would dispatch renders, never see them land, mark every one FAILED and re-dispatch
# on the next poll — booking the full estimate each time with no refund path (KNOWLEDGE.md
# `watcher` `phantom-spend`). Resolved from `model.family`, never hardcoded per-family here.
FAMILY = _cfg.model.family
SAMPLES_ROOT = samples_root(OUTPUT_DIR, FAMILY)
POLL_SECONDS = 240
RENDER_EST_USD = round(_cfg.modal.est_hours * _cfg.modal.hourly_rate_usd, 2)
DEADLINE_HOURS = 12.0
# AUDIT #4 — cumulative session-cap ledger (WR-02 authoritative chain, config-first). The dispatch
# loop reads this before EVERY render and stops when the cap would be breached.
LEDGER_PATH = _cfg.modal.session_spend_ledger_path
# OBS-01 (D-OBS-4, config-first): READ-ONLY volume-op timeout so a hung `modal volume ls/get` can't
# wedge the single-threaded loop. sh() defaults to this + CATCHES TimeoutExpired; dispatch_render
# OPTS OUT (timeout=None) — this watcher's render is ATTACHED and BLOCKS for the full ~60-90 min
# render, so it must NEVER be timed out (that would be a client-kill, the AUDIT-#1 regression).
VOLUME_OP_TIMEOUT_S = _cfg.modal.volume_op_timeout_s
# OBS-01 exit-code PROTOCOL (contract, not a tunable): DISTINCT cap-stop code the supervisor treats
# as "session cap reached — do NOT relaunch"; a clean complete exits 0 (+ sentinel).
CAP_STOP_EXIT = 3


def _resolve_cap(ledger_path: str, default_cap: float) -> float:
    """Cap source of truth (WR-02): the SESSION-STATE ledger's top-level ``session_cap_usd`` when the
    setup gate wrote one, else the config house default. Never a divergent literal."""
    try:
        data = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default_cap
    if isinstance(data, dict) and isinstance(data.get("session_cap_usd"), (int, float)):
        return float(data["session_cap_usd"])
    return default_cap


GRID = REPO / "_tools" / "finetune-gridwatch" / ".venv" / "Scripts" / "grid.exe"
_slug = OUTPUT_DIR.rstrip("/").split("/")[-1]
LOCAL_SAMPLES = REPO / f"_samples_{_slug}"
GRID_SRC = REPO / f"_grid_{_slug}"
# OBS-01 (D-OBS-3 / D-OBS-1) local-convention paths (derived from output_dir, not config thresholds).
# HEARTBEAT_FILE is touched each poll iteration (supervisor reads its AGE); SENTINEL_FILE is written
# on a clean run-complete so the supervisor stops without relaunching.
HEARTBEAT_FILE = REPO / f"_watcher_heartbeat_{_slug}"
SENTINEL_FILE = REPO / f"_watcher_complete_{_slug}"
# WR-03: cap-stop gets the SAME dual signal as run-complete — a DISTINCT sentinel written on the
# session-cap stop, IN ADDITION to sys.exit(CAP_STOP_EXIT), so a null/unreliable child ExitCode can
# never relaunch the dispatcher back into the same cap (the supervisor checks it before the exit code).
CAP_STOP_SENTINEL_FILE = REPO / f"_watcher_capstop_{_slug}"


def sh(args: list[str], *, timeout: float | None = VOLUME_OP_TIMEOUT_S, **kw) -> subprocess.CompletedProcess:
    # OBS-01 (D-OBS-4): READ-ONLY volume ops run under timeout=VOLUME_OP_TIMEOUT_S so a hung `modal
    # volume ls/get` can't wedge the single-threaded loop; a TimeoutExpired is CAUGHT and returned as
    # a failed result (empty stdout, rc 124) so the poll just sees "no data" and continues.
    # dispatch_render passes timeout=None to OPT OUT — the ATTACHED render blocks for the full render
    # and must never be client-killed (the AUDIT-#1 regression).
    try:
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=str(REPO), timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        print(f"[watcher] volume op timed out after {timeout}s: {' '.join(args)} — skipping this "
              "poll (loop continues).", flush=True)
        return subprocess.CompletedProcess(args, returncode=124, stdout="",
                                           stderr=f"[sh] volume op timed out after {timeout}s")


def committed_render_stamps() -> list[str]:
    # OBS-01 (D-OBS-1) crash-resumable seed source: the committed parallel-render artifacts on the
    # checkpoints Volume. Any id means at least one render has committed — the signal used to seed
    # `rendered` on startup. FAMILY-AWARE: LTX renders are UTC stamps under `samples/`, H3 renders
    # are identity keys under `samples_h3/`. Both the directory AND the naming scheme differ, so
    # re-pointing only the directory would still mis-verify every H3 render.
    r = sh(["modal", "volume", "ls", "signe-trainer-checkpoints", SAMPLES_ROOT],
           timeout=VOLUME_OP_TIMEOUT_S)
    return landed_render_ids(r.stdout or "", FAMILY)


def latest_checkpoint_name() -> str | None:
    """The newest checkpoint DIR NAME (not just its step) — H3 render keys embed it verbatim."""
    r = sh(["modal", "volume", "ls", "signe-trainer-checkpoints", OUTPUT_DIR])
    names = re.findall(r"(checkpoint-step-\d+-loss-[\d.]+)", r.stdout or "")
    return max(names, key=lambda n: int(re.search(r"step-(\d+)", n).group(1))) if names else None


def render_landed(step: int) -> bool:
    """Did the render for `step` actually commit artifacts?  (carry-forward of aaaee62)

    The render key now carries checkpoint + seed + frame_count + ORDERED reference condition, so the
    landed-check keys on the SAME identity. All five H3 sample configs share `output_dir`, `seed` and
    their prompt set and therefore write identical clip FILENAMES — only the render-dir identity
    separates them. A coarser check (checkpoint alone) would accept the `A+029` render as proof the
    `B+029` render landed, and the grid would come out labelled for a reference condition it does not
    contain. On LTX there is no per-render identity, so ANY new committed stamp is the signal — the
    historical behaviour, unchanged.
    """
    ids = committed_render_stamps()
    if FAMILY != "h3":
        return bool(ids)
    ckpt = latest_checkpoint_name()
    if ckpt is None:
        return False
    want = expected_h3_render_key(
        checkpoint=ckpt,
        seed=int(_cfg.validation.seed),
        frame_count=int(_cfg.validation.frame_count),
        subject_ids=list(_cfg.validation.reference_subject_ids or []),
    )
    return want in ids


def list_checkpoint_steps() -> list[int]:
    r = sh(["modal", "volume", "ls", "signe-trainer-checkpoints", OUTPUT_DIR])
    return sorted({int(m.group(1)) for m in
                   re.finditer(r"checkpoint-step-(\d+)-loss-", r.stdout or "")})


def append_spend(step: int) -> None:
    # In-process ledger write (AUDIT-#4 fix, CR-01): call session_cap.append_spend directly rather
    # than shelling out to `sys.executable -c ...` with a discarded exit code. Under the documented
    # bare invocation (no PYTHONPATH=src) the subprocess silently ModuleNotFoundError'd and the
    # ledger never advanced, so the cumulative cap could not trip. A ValueError/OSError now
    # PROPAGATES out of main() and halts the watcher BEFORE dispatch — the money-safe direction.
    _append_spend(LEDGER_PATH, RENDER_EST_USD,
                  run_ref=f"parallel render @ step {step} ({OUTPUT_DIR})")


def dispatch_render(step: int) -> bool:
    print(f"[watcher] new checkpoint step {step} -> dispatching gated render", flush=True)
    # timeout=None (OBS-01): the ATTACHED render dispatch is EXPLICITLY EXEMPT from sh()'s read-only
    # volume-op timeout. This `modal run` (attached, no detach flag) blocks for the full render;
    # timing it out would client-kill a healthy render — the forbidden AUDIT-#1 regression.
    r = sh(["modal", "run", "-m", "signet_trainer.modal.entrypoint",
            "--config", SAMPLE_CONFIG, "--mode", "sample", "--approve"], timeout=None)
    ok = r.returncode == 0
    print(f"[watcher] render step {step}: {'OK' if ok else 'FAILED'}", flush=True)
    if not ok:
        print((r.stdout or "")[-2000:], flush=True)
        print((r.stderr or "")[-2000:], flush=True)
    return ok


def refresh_grid() -> None:
    # H3 delegates to the first-party incremental grid driver, which fetches ONLY clips it does not
    # already hold, restages, rebuilds with finetune-gridwatch and leaves the live `grid watch`
    # server running (it re-scans and pushes over SSE, so an already-served page picks up new cells
    # without a restart — the grid GROWS as renders land instead of being rebuilt at the end).
    # `--rows step` makes the CHECKPOINT STEP the row axis, which is the axis that varies during
    # training. Never hand-rolled: finetune-gridwatch is the only sanctioned grid builder.
    if FAMILY == "h3":
        r = sh([sys.executable, str(REPO / "scripts" / "_h3_grid_serve.py"),
                "--config", SAMPLE_CONFIG, "--rows", "step", "--checkpoint", "all"], timeout=None)
        print((r.stdout or "")[-1500:], flush=True)
        return
    r = sh(["modal", "volume", "ls", "signe-trainer-checkpoints", f"{OUTPUT_DIR}/samples"])
    stamps = sorted(re.findall(r"samples/(\d{8}T\d{6}Z)", r.stdout or ""))
    if not stamps:
        return
    latest = stamps[-1]
    dest = LOCAL_SAMPLES / latest
    if not list(dest.rglob("*.mp4")):
        dest.mkdir(parents=True, exist_ok=True)  # PRE-CREATE (Windows modal-get quirk: target must exist)
        sh(["modal", "volume", "get", "signe-trainer-checkpoints",
            f"{OUTPUT_DIR}/samples/{latest}/", str(dest) + "/", "--force"])
    # modal volume get nests the remote dir NAME under the target -> descend if present.
    inner = dest / latest
    src_root = inner if inner.exists() else dest
    # Restage into gridwatch convention: <prompt>/step_<N>.mp4 (base=0, lora=<latest step>).
    GRID_SRC.mkdir(exist_ok=True)
    steps_seen = list_checkpoint_steps()
    lora_step = steps_seen[-1] if steps_seen else 0
    for col, stepname in (("base", 0), ("lora", lora_step)):
        for mp4 in (src_root / col).glob("*.mp4"):
            prompt_dir = GRID_SRC / mp4.stem.rsplit("_s", 1)[0][:60]
            prompt_dir.mkdir(exist_ok=True)
            target = prompt_dir / f"step_{stepname}.mp4"
            target.write_bytes(mp4.read_bytes())
    if GRID.exists():
        sh([str(GRID), "build", str(GRID_SRC), "--no-open",
            "--template", "{prompt}/step_{step}.mp4"])
        print("[watcher] grid rebuilt -> grid-output/index.html", flush=True)


def main() -> None:
    # OBS-01 (D-OBS-1) crash-resumable seed: derive `rendered` from committed Volume render artifacts
    # so a supervisor relaunch is idempotent (never re-dispatches an already-rendered boundary). If any
    # parallel-render artifact has committed, every checkpoint BELOW the latest is treated as already
    # rendered (this watcher renders the newest and moves forward); the newest still renders. Empty
    # (fresh start, matching the prior behaviour) when no render has committed yet.
    _seed_steps = list_checkpoint_steps()
    _seed_latest = max(_seed_steps, default=0)
    rendered: set[int] = ({s for s in _seed_steps if s < _seed_latest}
                          if committed_render_stamps() else set())
    if rendered:
        print(f"[watcher] reseed from Volume: treating steps {sorted(rendered)} as already rendered "
              f"(latest={_seed_latest} still renders)", flush=True)
    deadline = time.time() + DEADLINE_HOURS * 3600
    print(f"[watcher] watching {OUTPUT_DIR} (poll {POLL_SECONDS}s, deadline {DEADLINE_HOURS}h)",
          flush=True)
    while time.time() < deadline:
        # OBS-01 (D-OBS-3): touch the heartbeat file EVERY iteration so the supervisor can detect a
        # frozen loop by heartbeat AGE (watcher_heartbeat_stall_minutes) and kill+relaunch (WATCHER-stall).
        HEARTBEAT_FILE.touch()
        steps = list_checkpoint_steps()
        # CADENCE (RENDER_EVERY): renders fire on step boundaries + the final step, never on every
        # checkpoint. `checkpoint_every` is a TRAINING knob; how often a metered second container is
        # worth spinning up is an ECONOMIC one. The final step always renders regardless of cadence.
        new = [s for s in steps
               if s not in rendered and (s % RENDER_EVERY == 0 or s >= MAX_STEPS)]
        if new:
            step = max(new)          # render the NEWEST; skip intermediates if we fell behind
            # AUDIT #4 — cumulative session-cap gate BEFORE dispatch. Refuse to dispatch when the
            # next render would breach the cap (SESSION-STATE override else config house default):
            # no dispatch, no ledger, no rendered mark — drop to ask-first.
            spent = read_ledger(LEDGER_PATH)
            cap = _resolve_cap(LEDGER_PATH, _cfg.modal.session_cap_usd)
            if not session_cap_check(RENDER_EST_USD, spent, cap).allowed:
                # OBS-01 cap-stop: exit with the DISTINCT CAP_STOP_EXIT so the supervisor treats this
                # as "session cap reached — do NOT relaunch" (reuses session_cap's decision, never a
                # parallel accounting). No dispatch, no ledger, no rendered mark. WR-03: ALSO write the
                # cap-stop sentinel (dual signal) BEFORE exiting so a missed exit code can't relaunch
                # the dispatcher back into the same cap.
                print(f"[watcher] session cap reached (spent ${spent:.2f} + projected "
                      f"${RENDER_EST_USD:.2f} > cap ${cap:.2f}) — halting dispatch.", flush=True)
                CAP_STOP_SENTINEL_FILE.write_text(
                    "cap-stop: session cap reached — do NOT relaunch\n", encoding="utf-8")
                sys.exit(CAP_STOP_EXIT)
            # AUDIT #2 — ledger EVERY dispatch (this watcher is ATTACHED: a client death mid-render
            # still burned A100 time), then success-gate `rendered`: mark rendered ONLY when the
            # render actually COMPLETED (dispatch_render True), so a killed render re-dispatches next
            # poll instead of being silently skipped. (find_latest renders the newest anyway.)
            append_spend(step)
            ok = dispatch_render(step)
            # ARTIFACT-VERIFIED, not exit-code-verified (`processes lie, artifacts don't`). The modal
            # CLI can exit non-zero on a `charmap` failure while printing its own success tick AFTER
            # the bytes are committed, and it can exit 0 on a render that wrote nothing. The render is
            # done when its identity-keyed dir is ON the Volume — which is also what makes the H3
            # samples-path fix load-bearing rather than cosmetic: this check reads that path.
            landed = render_landed(step)
            if ok and not landed:
                print(f"[watcher] render step {step}: exit 0 but NO committed artifact at "
                      f"{SAMPLES_ROOT} — treating as NOT rendered (re-dispatches next poll).",
                      flush=True)
            if not ok and landed:
                print(f"[watcher] render step {step}: non-zero exit but the artifact IS committed at "
                      f"{SAMPLES_ROOT} — counting it rendered (the CLI can die on its own success "
                      "tick; never trust the exit code alone).", flush=True)
            ok = landed
            if ok:
                rendered.update(new)
                refresh_grid()
            if ok and step >= MAX_STEPS:
                # OBS-01 run-complete: success-gated — write the sentinel + exit 0 so the supervisor
                # STOPS (does not relaunch).
                print("[watcher] final checkpoint rendered — done.", flush=True)
                SENTINEL_FILE.write_text("run-complete: final checkpoint rendered\n", encoding="utf-8")
                sys.exit(0)
        time.sleep(POLL_SECONDS)
    # OBS-01 run-complete (deadline): a clean, expected end — write the sentinel + exit 0 so the
    # supervisor STOPS rather than relaunching into an already-finished run.
    print("[watcher] deadline reached — exiting.", flush=True)
    SENTINEL_FILE.write_text("run-complete: watch deadline reached\n", encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
