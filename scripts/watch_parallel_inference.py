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
    committed_clip_names,
    expected_h3_base_render_key,
    expected_h3_render_key,
    landed_render_ids,
    samples_root,
)
from signet_trainer.inference.stall_clock import next_pending_since  # noqa: E402
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
# Wave 2 #6 LANDED (audit 2026-08-11): the coupled detach+freshness pair, ported from the campaign
# fork. Dispatch is DETACHED (`modal run --detach` — a client death or the entrypoint's normal
# `dispatch_watch_seconds` disengage can no longer tear down the render app, the who-holds-the-client
# rule), so the `modal run` subprocess returning is DISPATCH-ACCEPTED, never render-complete.
# Completion is ARTIFACT-gated: a dispatched render stays "pending" until its identity-keyed dir
# commits to the Volume, or until it has produced nothing for render_stall_minutes — only then is it
# declared stalled and re-dispatch (still cap-gated, still ledgered) becomes eligible. This is what
# kills the re-dispatch-and-re-bill-every-poll failure the attached assumption caused once the
# entrypoint moved to .spawn() (D-10-DEF-17).
#
# ⛔ issue #45 PR-1 must-fix #1 — THE CLOCK MUST REFRESH ON PROGRESS, NOT ONLY START AT DISPATCH.
# `pending_since` used to be set once when the render was dispatched and never touched again. A
# single H3 render identity can legitimately keep committing clips for hours (`h3_sample` commits
# per clip, `modal/fns.py`), so a clock that never refreshes declares every healthy long-running
# render STALLED at `render_stall_minutes`, releases single-flight, and lets a second metered A100
# dispatch OVER the still-running first — repeating until the session cap trips. `render_landed()`'s
# own check is coarse (it saturates true-or-false the moment the FIRST clip commits) and cannot
# supply that refresh signal by itself; `render_progress_artifacts()` below reads one level deeper
# for exactly that reason, and main()'s loop resets `pending_since` whenever it changes.
RENDER_STALL_MIN = float(_cfg.modal.render_stall_minutes)
# AUDIT #4 — cumulative session-cap ledger (WR-02 authoritative chain, config-first). The dispatch
# loop reads this before EVERY render and stops when the cap would be breached.
LEDGER_PATH = _cfg.modal.session_spend_ledger_path
# OBS-01 (D-OBS-4, config-first): READ-ONLY volume-op timeout so a hung `modal volume ls/get` can't
# wedge the single-threaded loop. sh() defaults to this + CATCHES TimeoutExpired; dispatch_render
# still OPTS OUT (timeout=None) — with --detach the dispatch returns at the entrypoint's
# dispatch_watch_seconds window, and the no-client-kill invariant (AUDIT-#1) says the dispatch
# subprocess is never timed out from here regardless.
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
    # dispatch_render passes timeout=None to OPT OUT — the dispatch subprocess (now detached, Wave 2
    # #6) is never client-killed from here (the AUDIT-#1 regression stays gone).
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


def render_landed(step: int, checkpoint: str | None) -> bool:
    """Did the render for `step` actually commit artifacts?  (carry-forward of aaaee62)

    The render key now carries checkpoint + seed + frame_count + ORDERED reference condition, so the
    landed-check keys on the SAME identity. All five H3 sample configs share `output_dir`, `seed` and
    their prompt set and therefore write identical clip FILENAMES — only the render-dir identity
    separates them. A coarser check (checkpoint alone) would accept the `A+029` render as proof the
    `B+029` render landed, and the grid would come out labelled for a reference condition it does not
    contain. On LTX there is no per-render identity, so ANY new committed stamp is the signal — the
    historical behaviour, unchanged.

    `checkpoint` MUST be the value main() captured AT DISPATCH TIME (issue #45 PR-1 must-fix #2),
    never re-derived here via a fresh `latest_checkpoint_name()` call. Re-resolving "latest" on every
    poll could drift onto a NEWER checkpoint that landed while THIS render was still pending, key
    `expected_h3_render_key` on the wrong identity, and never find the render actually dispatched —
    silently degrading into a permanent false STALL for a render that in fact landed.
    """
    ids = committed_render_stamps()
    if FAMILY != "h3":
        return bool(ids)
    if checkpoint is None:
        # Kill the match-anything path: a caller with no resolved checkpoint must be REFUSED, never
        # silently satisfied by whatever the identity-independent branch above would have returned.
        # main() already refuses to dispatch (and to book spend) when this resolves to None at
        # dispatch time — a None reaching here means that guard was bypassed, which is the bug, not
        # a signal to paper over by matching any render under this output_dir.
        raise ValueError(
            "render_landed(family='h3') called with checkpoint=None. An unresolved checkpoint must "
            "refuse verification, never match any committed render under this identity."
        )
    want = expected_h3_render_key(
        checkpoint=checkpoint,
        seed=int(_cfg.validation.seed),
        frame_count=int(_cfg.validation.frame_count),
        # #22 finding 5 / #12: the key now carries the geometry axes too — a resolution or
        # step-count change is a genuinely different render, not a resume of the old one.
        width=int(_cfg.validation.width),
        height=int(_cfg.validation.height),
        num_inference_steps=int(_cfg.validation.num_inference_steps),
        subject_ids=list(_cfg.validation.reference_subject_ids or []),
    )
    return want in ids


def render_progress_artifacts(checkpoint: str | None) -> frozenset[str]:
    """Every artifact committed so far for the PENDING render — the fine-grained PROGRESS signal the
    stall clock needs (issue #45 PR-1 must-fix #1).

    `render_landed` answers one coarse question — "has this render's identity directory appeared at
    all?" — which saturates the moment the FIRST clip commits and stays true for the rest of an
    up-to-6h render; it is useless as a freshness signal on its own. This reads one level deeper via
    `committed_clip_names` (`inference/samples_layout.py`) — so a newly-committed clip shows up here
    long before the whole render is judged landed, and a live render can be told apart from a hung
    one.

    ⚠ RESTACK RECONCILIATION (#12 base-render dedup, verify_c.json MAJOR finding): the two columns no
    longer share one directory. The lora half stays under the checkpoint-scoped render dir (the
    per-checkpoint adapter half); the base half lives under SAMPLES_ROOT's own base subdirectory,
    keyed by `expected_h3_base_render_key` and shared across every checkpoint that renders the same
    (seed, frame_count, geometry, references) request — a SIBLING of every checkpoint-scoped render
    dir, never nested back under one. Descending into the checkpoint-scoped render dir for the base
    half (the pre-dedup layout) would silently see zero progress for the ENTIRE base phase of a
    fresh-geometry render (h3_sample renders the base column first), which is exactly the
    false-stall risk this probe exists to prevent. Both key calls carry the SAME widened geometry
    axes (#22 finding 5) as `render_landed`'s — a probe keyed on the old, narrower signature would
    watch the wrong directory.

    On LTX, and while the checkpoint identity is not yet known, there is no per-render directory to
    descend into ahead of time — falls back to the same top-level `committed_render_stamps()` set
    `render_landed` itself reads on that family; a new stamp appearing is progress there too.
    """
    if FAMILY != "h3" or checkpoint is None:
        return frozenset(committed_render_stamps())
    lora_key = expected_h3_render_key(
        checkpoint=checkpoint,
        seed=int(_cfg.validation.seed),
        frame_count=int(_cfg.validation.frame_count),
        width=int(_cfg.validation.width),
        height=int(_cfg.validation.height),
        num_inference_steps=int(_cfg.validation.num_inference_steps),
        subject_ids=list(_cfg.validation.reference_subject_ids or []),
    )
    base_key = expected_h3_base_render_key(
        seed=int(_cfg.validation.seed),
        frame_count=int(_cfg.validation.frame_count),
        width=int(_cfg.validation.width),
        height=int(_cfg.validation.height),
        num_inference_steps=int(_cfg.validation.num_inference_steps),
        subject_ids=list(_cfg.validation.reference_subject_ids or []),
    )
    render_dir = f"{SAMPLES_ROOT}/{lora_key}"
    # "lora" stays checkpoint-scoped (under render_dir); "base" is the dedup's shared sibling
    # directory under SAMPLES_ROOT — never nested back under render_dir (#12).
    found: set[str] = set()
    r = sh(["modal", "volume", "ls", "signe-trainer-checkpoints", f"{render_dir}/lora"])
    found.update(f"lora/{name}" for name in committed_clip_names(r.stdout or ""))
    r = sh(["modal", "volume", "ls", "signe-trainer-checkpoints", f"{SAMPLES_ROOT}/base/{base_key}"])
    found.update(f"base/{name}" for name in committed_clip_names(r.stdout or ""))
    return frozenset(found)


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
    print(f"[watcher] new checkpoint step {step} -> dispatching gated render (detached)", flush=True)
    # DETACHED dispatch (Wave 2 #6): --detach keeps the ephemeral app alive after this client exits,
    # so the .spawn()'d render survives the entrypoint's dispatch_watch_seconds disengage. The
    # subprocess therefore returns ~minutes after launch and its exit code means DISPATCH ACCEPTED,
    # never render-complete — completion is judged by the artifact gate in main(). timeout=None
    # (OBS-01): the dispatch subprocess is still never timed out from here (no client-kill).
    r = sh(["modal", "run", "--detach", "-m", "signet_trainer.modal.entrypoint",
            "--config", SAMPLE_CONFIG, "--mode", "sample", "--approve"], timeout=None)
    ok = r.returncode == 0
    print(f"[watcher] render step {step}: {'DISPATCHED' if ok else 'DISPATCH FAILED'}", flush=True)
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
    # PIN REFUSAL (audit 2026-08-11): a pinned h3.render_checkpoint_name renders the SAME checkpoint
    # regardless of training progress, and this watcher's expected render key derives from the
    # LATEST checkpoint — so under a pin, per-step renders can never be verified (every cadence
    # boundary would re-render one artifact and the landed-check would key on the wrong identity).
    # A pinned render is a ONE-SHOT: dispatch it directly through the entrypoint instead.
    if FAMILY == "h3" and str(getattr(_cfg.h3, "render_checkpoint_name", "") or ""):
        print(
            f"[watcher] REFUSING to start: h3.render_checkpoint_name is pinned "
            f"({_cfg.h3.render_checkpoint_name!r}) in {SAMPLE_CONFIG}. A step-cadence watcher "
            "cannot verify pinned renders (its landed-check keys on the LATEST checkpoint). "
            "Clear the pin for watcher use, or dispatch the pinned render one-shot:\n"
            "  modal run --detach -m signet_trainer.modal.entrypoint --config <cfg> --mode sample --approve",
            flush=True,
        )
        sys.exit(2)
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
    # Wave 2 #6 single-flight state: at most ONE render in flight. `pending_step` is the step whose
    # detached render has been dispatched (and ledgered) but whose artifact has not yet committed;
    # `pending_since` anchors the render_stall_minutes freshness clock (issue #45 PR-1 must-fix #1:
    # REFRESHED on progress, not just set once). `pending_checkpoint` is the identity captured AT
    # DISPATCH TIME (must-fix #2) that `render_landed`/`render_progress_artifacts` verify against for
    # the life of this pending render — never re-derived mid-flight. `pending_progress` is the last
    # observed artifact snapshot, compared each poll to decide whether the clock resets.
    pending_step: int | None = None
    pending_since = 0.0
    pending_checkpoint: str | None = None
    pending_progress: frozenset[str] = frozenset()
    print(f"[watcher] watching {OUTPUT_DIR} (poll {POLL_SECONDS}s, deadline {DEADLINE_HOURS}h, "
          f"render stall gate {RENDER_STALL_MIN:g} min)", flush=True)
    while time.time() < deadline:
        # OBS-01 (D-OBS-3): touch the heartbeat file EVERY iteration so the supervisor can detect a
        # frozen loop by heartbeat AGE (watcher_heartbeat_stall_minutes) and kill+relaunch (WATCHER-stall).
        HEARTBEAT_FILE.touch()
        steps = list_checkpoint_steps()
        # CADENCE (RENDER_EVERY): renders fire on step boundaries + the final step, never on every
        # checkpoint. `checkpoint_every` is a TRAINING knob; how often a metered second container is
        # worth spinning up is an ECONOMIC one. The final step always renders regardless of cadence.
        new = [s for s in steps
               if s not in rendered and s != pending_step
               and (s % RENDER_EVERY == 0 or s >= MAX_STEPS)]
        if new and pending_step is None:  # SINGLE-FLIGHT: never dispatch over a pending render
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
            # CAPTURE-BEFORE-BOOK (issue #45 PR-1 must-fix #2): resolve the checkpoint identity the
            # render will key on BEFORE append_spend, never after. `latest_checkpoint_name()` returns
            # None on a `modal volume ls` timeout (sh()'s VOLUME_OP_TIMEOUT_S catch) or an empty
            # listing; booking spend and dispatching anyway would create a render this watcher could
            # never verify, and at MAX_STEPS could write the run-complete sentinel having verified
            # nothing. LTX has no checkpoint-keyed render identity to resolve, so it is exempt.
            ckpt_at_dispatch = latest_checkpoint_name() if FAMILY == "h3" else None
            if FAMILY == "h3" and ckpt_at_dispatch is None:
                print(f"[watcher] step {step}: could not resolve the latest checkpoint name (modal "
                      "volume ls timeout or empty listing) — REFUSING to book spend/dispatch for a "
                      "render this watcher could never verify. Retrying next poll.", flush=True)
            else:
                # AUDIT #2 — ledger EVERY dispatch (a dispatched render is booked A100 time whether
                # or not it lands), then hand completion to the ARTIFACT gate below. Spend books once
                # per DISPATCH, and single-flight + the stall gate mean a dispatch happens at most
                # once per render identity per render_stall_minutes window — never once per poll.
                append_spend(step)
                ok = dispatch_render(step)
                if ok:
                    pending_step, pending_since = step, time.time()
                    pending_checkpoint, pending_progress = ckpt_at_dispatch, frozenset()
                else:
                    print(f"[watcher] dispatch for step {step} REFUSED/FAILED before spawn — "
                          "eligible again next poll (cap-gated).", flush=True)
        if pending_step is not None:
            # ARTIFACT-VERIFIED completion (`processes lie, artifacts don't`): the render is done
            # when its identity-keyed dir is ON the Volume — never when a subprocess exits. The
            # freshness gate (render_stall_minutes, config-first) is the ONLY path that gives up on
            # a pending render; only then does re-dispatch (with a fresh, honest booking) become
            # eligible, and the pre-dispatch session-cap gate above still bounds the total.
            landed = render_landed(pending_step, pending_checkpoint)
            if not landed:
                # PROGRESS REFRESH (issue #45 PR-1 must-fix #1): reset the clock on evidence of life
                # BEFORE measuring staleness, not after — see the RENDER_STALL_MIN comment above for
                # why render_landed's coarse check cannot supply this signal on its own. The
                # DECISION itself is delegated to next_pending_since() (inference/stall_clock.py) — a
                # pure, importable function, not an inline comparison — because a PR-2 verifier
                # caught a semantic inversion of this exact comparison (`!=` -> `==`) surviving every
                # test in tests/test_watcher_hardening.py; those tests only scan main()'s SOURCE and
                # cannot exercise a decision that lives only inline. See
                # tests/test_watcher_pending_clock.py for the behavioral coverage that closes it.
                progress = render_progress_artifacts(pending_checkpoint)
                if next_pending_since(progress, pending_progress, pending_since, time.time()) != pending_since:
                    print(f"[watcher] render step {pending_step}: new committed artifact(s) "
                          f"({len(progress)} total) — resetting the stall clock.", flush=True)
                    pending_progress = progress
                    pending_since = time.time()
            age_min = (time.time() - pending_since) / 60.0
            if landed:
                ok, step = True, pending_step
                rendered.update(s for s in steps
                                if s <= step and (s % RENDER_EVERY == 0 or s >= MAX_STEPS))
                refresh_grid()
                pending_step, pending_checkpoint, pending_progress = None, None, frozenset()
                if ok and step >= MAX_STEPS:
                    # OBS-01 run-complete: success-gated — write the sentinel + exit 0 so the
                    # supervisor STOPS (does not relaunch).
                    print("[watcher] final checkpoint rendered — done.", flush=True)
                    SENTINEL_FILE.write_text("run-complete: final checkpoint rendered\n",
                                             encoding="utf-8")
                    sys.exit(0)
            elif age_min > RENDER_STALL_MIN:
                # A render is STALLED only when NO new artifact has appeared for render_stall_minutes
                # (issue #45 PR-1 must-fix #1) — a genuinely hung render still fires this, unchanged.
                print(f"[watcher] render step {pending_step}: no new committed render artifact "
                      f"after {age_min:.0f} min (> render_stall_minutes {RENDER_STALL_MIN:g}) — "
                      "declaring STALLED; re-dispatch eligible next poll (cap-gated).", flush=True)
                pending_step, pending_checkpoint, pending_progress = None, None, frozenset()
            else:
                print(f"[watcher] render step {pending_step}: awaiting artifact "
                      f"({age_min:.0f}/{RENDER_STALL_MIN:g} min)", flush=True)
        time.sleep(POLL_SECONDS)
    # OBS-01 run-complete (deadline): a clean, expected end — write the sentinel + exit 0 so the
    # supervisor STOPS rather than relaunching into an already-finished run.
    print("[watcher] deadline reached — exiting.", flush=True)
    SENTINEL_FILE.write_text("run-complete: watch deadline reached\n", encoding="utf-8")
    sys.exit(0)


if __name__ == "__main__":
    main()
