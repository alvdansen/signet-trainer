#!/usr/bin/env python3
"""Live RUN-STATUS view for a gated signet-trainer run — one self-contained HTML page.

    PYTHONPATH=src python scripts/_h3_run_status.py --config configs/<your-h3-run>.yaml

Re-run the SAME command to refresh. The page is regenerated from scratch every time, so there is
no incremental state to corrupt and no "stale half-page" failure mode.

WHY THIS EXISTS
---------------
A 13.8 h metered run is an object the operator cannot see. `modal app logs` is a firehose, the
Volume is a directory listing, and the spend ledger is a JSON file — three surfaces, none of which
answers "is it alive and healthy?" at a glance. This collapses them into ONE artifact.

⛔ ZERO METERED SPEND, BY CONSTRUCTION. Every source below is a READ-ONLY Modal CLI call
(`modal volume ls`, `modal app list`, `modal app logs`) or a local file. This script contains no
`modal run`, no `.remote()`, and no `.spawn()`, and it must never grow one — the single-gate law
(CLAUDE.md) says a metered dispatch goes through the entrypoint with a cost print and an approval,
and a *monitoring* tool is exactly the place where that would be easiest to violate by accident.

LIVENESS IS MEASURED BY ARTIFACT FRESHNESS, NEVER BY PROCESS STATE (house rule
`stall-detection-time-gates`: "processes lie, artifacts don't"). A container can be alive, billing,
and wedged. The authoritative liveness signal here is the **mtime of the newest committed
checkpoint on the Volume** — a number that can only advance if real work landed. The Modal app
record is read too, but only to date the run and to attribute it; it never overrides the artifact.

WHAT IT READS, AND WHY EACH SOURCE IS THE RIGHT ONE
--------------------------------------------------
* `modal volume ls <vol> <output_dir> --json` — the checkpoint dirs. `CheckpointManager` names them
  ``checkpoint-step-{step:05d}-loss-{loss:.4f}``, so ONE listing yields the step counter, the whole
  loss trend, and per-checkpoint mtimes. That is why the loss chart needs no log parsing and no
  wandb: the trend is already encoded in the artifact names.
* `modal app list --json` — dates the run (elapsed vs the config's `est_hours`) and names the app.
* `modal app logs <app> --tail N` — the ONLY source for the VRAM gauge (commit 7df4fd9 logs
  `torch.cuda.max_memory_allocated()` in the checkpoint branch). The Volume cannot carry it.
* `SESSION-STATE.json` — the SINGLE numeric source of truth for spend/cap (typed-state house rule:
  prose artifacts never hand-copy its figures). Read through `session_cap.read_ledger` when the
  package imports, so this page can never disagree with the gate about what has been spent.
* the campaign card `*.state.yaml` — `expected_peak_gib` / `expected_s_per_it`: the tripwires the
  live gauges are compared AGAINST. A gauge with no expectation is a number, not a signal.

PROPERTY HYGIENE. Nothing client-bearing reaches the page: no prompts, no reference filenames, no
staging paths. Reference conditions appear as subject_ids only. Raw log tails are OFF by default
(`--include-log-tail` opts in) and every surfaced line goes through `_redact()`.

DEGRADES, NEVER CRASHES. No Volume dir, no app, no checkpoints, no logs, Modal CLI absent, a
different profile — each renders as an explicit empty state with the reason, because a monitoring
tool that dies when the thing it monitors has not started yet is the one you cannot trust at 3am.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# A legacy Windows console defaults to cp1252, whose charmap cannot encode an em dash — so a
# diagnostic containing one raises UnicodeEncodeError and takes the monitor down INSTEAD of printing
# the diagnostic. Reconfigure both streams up front; guarded, because a captured pipe or an older
# wrapper may not expose `reconfigure`, and this must never fail on import.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — never fail on import
            pass

# ``CheckpointManager`` save format: ``checkpoint-step-{step:05d}-loss-{loss:.4f}``. Parsing the
# NAME (not a sidecar) is deliberate — the name is written atomically with the checkpoint, so a
# half-written checkpoint cannot contribute a phantom point to the loss trend.
_CKPT_RE = re.compile(r"checkpoint-step-(\d+)-loss-([0-9]+\.[0-9]+)$")

# The VRAM gauge line from train/loop.py (7df4fd9). The ``.{0,8}`` absorbs the em dash without
# depending on the console encoding that produced it.
_VRAM_RE = re.compile(
    r"step\s+(\d+).{0,8}VRAM peak\s+([0-9.]+)\s*GiB allocated\s*/\s*([0-9.]+)\s*GiB reserved"
)

# Log lines worth surfacing without asking anyone to read a firehose.
_ALERT_RE = re.compile(
    r"(CUDA out of memory|OutOfMemoryError|watchdog tripped|Traceback|RuntimeError|"
    r"does NOT change the model|Killed|preempt)",
    re.IGNORECASE,
)

# Client-property redaction for any log text that reaches the page (defence in depth: the page is
# generated into a gitignored dir, but "gitignored" is not the same as "safe to paste").
_REDACT_PATTERNS = (
    (re.compile(r"(?:/[\w.\-]+)*/refs/[^\s'\"]+"), "refs/<redacted>"),
    (re.compile(r"[\w.\-]+\.(?:jpg|jpeg|png|webp|mp4|mov|webm)", re.IGNORECASE), "<media>"),
)

_MODAL_ENV = {
    # Belt-and-braces for the Git-Bash/MSYS path-mangling landmine: an absolute Volume path such as
    # ``/checkpoints/...`` gets silently rewritten to a Windows path by MSYS, and the modal CLI then
    # REPORTS SUCCESS against the mangled target. subprocess without a shell is not subject to it,
    # but this script's own printed commands ARE meant to be copy-pasteable into Git Bash.
    "MSYS_NO_PATHCONV": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


# ──────────────────────────────────────────────────────────────────────────────────────────────
# process + parsing helpers
# ──────────────────────────────────────────────────────────────────────────────────────────────


def _redact(text: str) -> str:
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _clean_cli_noise(text: str) -> str:
    """Strip the modal CLI's rich error box so a surfaced reason reads as a sentence, not ASCII art.

    Box-drawing lives in U+2500-U+257F, which a naive `[|+-]{2,}` filter does not touch — the reason
    then arrives as 60 characters of `─│┌└` with the actual message pushed past the truncation.
    """
    stripped = re.sub(r"[─-╿|+]+", " ", text)
    return re.sub(r"\s{2,}", " ", stripped).strip()


def _run(cmd: list[str], timeout: float) -> tuple[int, str, str]:
    """Run a command, never raise. Returns ``(returncode, stdout, stderr)``.

    A missing binary, a timeout and a non-zero exit all collapse to the same shape so every caller
    can treat "no data" uniformly — this is what makes the whole page degrade instead of crash.
    """
    env = {**os.environ, **_MODAL_ENV}
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=env, cwd=str(REPO_ROOT),
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:.0f}s"
    except Exception as exc:  # noqa: BLE001 — a monitor must never die on its own probe
        return 1, "", f"{type(exc).__name__}: {exc}"


def _modal_json(args: list[str], timeout: float) -> tuple[object | None, str]:
    """`modal <args> --json` → parsed JSON, or ``(None, reason)``.

    The modal CLI prints a rich-boxed error to stdout for a missing path and STILL exits in a way
    that is awkward to branch on, so the parse itself is the test: if it is not JSON, there is no
    data, and the reason is carried forward to the page rather than swallowed.
    """
    rc, out, err = _run(["modal", *args, "--json"], timeout)
    text = out.strip()
    if not text or not text.lstrip().startswith(("[", "{")):
        reason = _clean_cli_noise(err or out)
        return None, (reason[:200] or f"modal {' '.join(args)} returned no JSON (rc={rc})")
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as exc:
        return None, f"unparseable JSON from `modal {' '.join(args)}`: {exc}"


def _parse_volume_mtime(raw: str) -> datetime | None:
    """``"2026-08-05 08:42 Pacific Daylight Time"`` → an aware local datetime.

    The CLI renders the timestamp in the LOCAL zone with a spelled-out zone name that
    ``datetime`` cannot parse, so the leading ``YYYY-MM-DD HH:MM`` is taken and stamped with the
    local offset. Resolution is one minute, which is two orders of magnitude finer than the
    60-minute staleness gate this feeds — precision is not the binding constraint here.
    """
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", raw or "")
    if not match:
        return None
    try:
        naive = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    return naive.astimezone()


def _fmt_hm(hours: float | None) -> str:
    if hours is None:
        return "—"
    total = int(round(hours * 60))
    sign = "-" if total < 0 else ""
    total = abs(total)
    return f"{sign}{total // 60}h {total % 60:02d}m"


def _fmt_age(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    if minutes < 90:
        return f"{minutes:.0f} min"
    return _fmt_hm(minutes / 60.0)


# ──────────────────────────────────────────────────────────────────────────────────────────────
# sources
# ──────────────────────────────────────────────────────────────────────────────────────────────


def load_recipe(config_path: Path) -> dict:
    """Recipe numbers from the training config — VALIDATED through the real schema when it imports.

    `load_config` is preferred over a bare `yaml.safe_load` so the page reports the values the
    trainer would actually run with (schema defaults applied, cross-field validators fired) rather
    than only what the YAML happens to spell out. It falls back to raw YAML so the page still
    renders on a machine without the package installed.
    """
    recipe: dict = {"config_path": str(config_path), "config_source": "schema"}
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from signet_trainer.config.load import load_config  # noqa: PLC0415

        cfg = load_config(str(config_path))
        recipe.update(
            output_dir=cfg.output_dir,
            max_steps=cfg.training.max_steps,
            checkpoint_every=cfg.training.checkpoint_every,
            checkpoint_expected_minutes=cfg.training.checkpoint_expected_minutes,
            checkpoint_stall_multiplier=cfg.training.checkpoint_stall_multiplier,
            keep_checkpoints=cfg.training.keep_checkpoints,
            est_hours=cfg.modal.est_hours,
            hourly_rate_usd=cfg.modal.hourly_rate_usd,
            cost_guardrail_usd=cfg.modal.cost_guardrail_usd,
            family=cfg.model.family,
            gpu_usable_gib=getattr(getattr(cfg, "h3", None), "gpu_usable_gib", None),
            # issue #19 item 3 (D-NOHARDCODE): the checkpoints Volume / App name this run actually
            # reads/dispatches under, so main() can default --volume/--app-name from them instead of
            # a hardcoded literal.
            checkpoints_volume_name=cfg.modal.checkpoints_volume_name,
            app_name=cfg.modal.app_name,
        )
        return recipe
    except Exception as exc:  # noqa: BLE001 — fall back rather than fail the whole page
        recipe["config_source"] = f"raw-yaml ({type(exc).__name__})"

    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    training, modal_cfg = raw.get("training", {}) or {}, raw.get("modal", {}) or {}
    recipe.update(
        output_dir=raw.get("output_dir"),
        max_steps=training.get("max_steps"),
        checkpoint_every=training.get("checkpoint_every"),
        checkpoint_expected_minutes=training.get("checkpoint_expected_minutes"),
        checkpoint_stall_multiplier=training.get("checkpoint_stall_multiplier", 2.5),
        keep_checkpoints=training.get("keep_checkpoints"),
        est_hours=modal_cfg.get("est_hours"),
        hourly_rate_usd=modal_cfg.get("hourly_rate_usd"),
        cost_guardrail_usd=modal_cfg.get("cost_guardrail_usd"),
        family=(raw.get("model", {}) or {}).get("family"),
        gpu_usable_gib=(raw.get("h3", {}) or {}).get("gpu_usable_gib"),
        checkpoints_volume_name=modal_cfg.get("checkpoints_volume_name"),
        app_name=modal_cfg.get("app_name"),
    )
    return recipe


def load_card(card_path: Path) -> dict:
    """`expected_peak_gib` / `expected_s_per_it` / `est_cost_usd` — the tripwires to compare against."""
    if not card_path.exists():
        return {"card_path": str(card_path), "available": False}
    try:
        import yaml  # noqa: PLC0415

        data = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return {"card_path": str(card_path), "available": False, "error": str(exc)[:160]}
    budget = data.get("budget", {}) or {}
    campaign = data.get("campaign", {}) or {}
    return {
        "card_path": str(card_path),
        "available": True,
        "slug": campaign.get("slug"),
        "expected_peak_gib": budget.get("expected_peak_gib"),
        "expected_s_per_it": budget.get("expected_s_per_it"),
        "est_cost_usd": budget.get("est_cost_usd"),
        "gpu": budget.get("gpu"),
        "max_packed_rows": budget.get("max_packed_rows"),
    }


def load_ledger(session_state_path: Path) -> dict:
    """Spend + cap. `read_ledger` is the canonical reader — never a second summation of the same list.

    Typed-state house rule: SESSION-STATE.json is the SINGLE numeric source of truth for spend/cap,
    and the $258.92-vs-$266.30 drift is exactly what a hand-rolled re-sum produces. The manual sum
    below is a FALLBACK for a machine without the package, and it is labelled as such on the page.
    """
    out: dict = {"path": str(session_state_path), "available": session_state_path.exists()}
    if not out["available"]:
        return out
    try:
        data = json.loads(session_state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        out["available"] = False
        out["error"] = str(exc)[:160]
        return out
    out["session_cap_usd"] = data.get("session_cap_usd")
    out["triage_mode"] = data.get("triage_mode")
    out["optimization_target"] = data.get("optimization_target")
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from signet_trainer.modal.session_cap import read_ledger  # noqa: PLC0415

        out["spent_usd"] = float(read_ledger(session_state_path))
        out["spend_source"] = "session_cap.read_ledger"
    except Exception:  # noqa: BLE001
        out["spent_usd"] = float(
            sum(float(e.get("est_usd", 0.0) or 0.0) for e in data.get("spend", []) or [])
        )
        out["spend_source"] = "local sum (package not importable)"
    blankets = data.get("blankets", []) or []
    out["blanket_caps"] = [b.get("cap_usd") for b in blankets if isinstance(b, dict)]
    return out


def discover_checkpoints(volume: str, output_dir: str, timeout: float) -> dict:
    """Every committed checkpoint on the Volume: step, loss, mtime — the run's whole spine.

    This is the ONLY source consulted for progress and for the loss trend, and it is deliberately
    the artifact rather than the log: a checkpoint dir exists only because `ckpt_manager.save()`
    completed AND `checkpoints_vol.commit()` landed it (commit-or-vanish). A log line proves an
    intention; a directory proves the work.
    """
    payload, reason = _modal_json(["volume", "ls", volume, output_dir], timeout)
    if payload is None:
        return {"available": False, "reason": reason, "checkpoints": []}
    rows = payload if isinstance(payload, list) else []
    checkpoints = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = Path(str(row.get("filename", ""))).name
        match = _CKPT_RE.search(name)
        if not match:
            continue
        written = _parse_volume_mtime(str(row.get("created_modified", "")))
        checkpoints.append(
            {
                "name": name,
                "step": int(match.group(1)),
                "loss": float(match.group(2)),
                "written_at": written.isoformat() if written else None,
                "_written": written,
            }
        )
    checkpoints.sort(key=lambda c: c["step"])
    other = [
        Path(str(r.get("filename", ""))).name
        for r in rows
        if isinstance(r, dict) and not _CKPT_RE.search(Path(str(r.get("filename", ""))).name)
    ]
    return {"available": True, "reason": "", "checkpoints": checkpoints, "other_entries": other}


def discover_app(app_name: str, app_id: str | None, timeout: float) -> dict:
    """The newest non-stopped Modal app with this name — used ONLY to date and attribute the run.

    Never used to decide liveness: a running container is not evidence of progress (the wedge case
    is precisely a container that is alive, billing, and doing nothing). Artifact freshness decides.
    """
    payload, reason = _modal_json(["app", "list"], timeout)
    if payload is None:
        return {"available": False, "reason": reason}
    apps = [a for a in payload if isinstance(a, dict)] if isinstance(payload, list) else []
    if app_id:
        match = [a for a in apps if a.get("app_id") == app_id]
        if not match:
            return {"available": False, "reason": f"app_id {app_id} not found in `modal app list`"}
        chosen = match[0]
    else:
        live = [
            a for a in apps
            if a.get("description") == app_name and not a.get("stopped_at")
        ]
        pool = live or [a for a in apps if a.get("description") == app_name]
        if not pool:
            return {"available": False, "reason": f"no Modal app named {app_name!r}"}
        chosen = max(pool, key=lambda a: str(a.get("created_at") or ""))

    started = None
    try:
        started = datetime.fromisoformat(str(chosen.get("created_at")))
    except (TypeError, ValueError):
        started = None
    stopped = None
    try:
        if chosen.get("stopped_at"):
            stopped = datetime.fromisoformat(str(chosen.get("stopped_at")))
    except (TypeError, ValueError):
        stopped = None

    end = stopped or datetime.now(timezone.utc).astimezone()
    elapsed_h = ((end - started).total_seconds() / 3600.0) if started else None
    return {
        "available": True,
        "reason": "",
        "app_id": chosen.get("app_id"),
        "state": chosen.get("state"),
        "tasks": chosen.get("tasks"),
        "started_at": started.isoformat() if started else None,
        "stopped_at": stopped.isoformat() if stopped else None,
        "elapsed_hours": elapsed_h,
        "running": not chosen.get("stopped_at"),
    }


def app_matches_run(app: dict, checkpoints: list[dict]) -> bool:
    """Is this Modal app plausibly the app that produced THESE checkpoints?

    APP ATTRIBUTION IS EARNED, NOT ASSUMED. ``modal app list`` returns every app of a given name on
    the account, and the newest is frequently a DIFFERENT dispatch — another campaign, another
    agent, a stray render. One comparison disqualifies a wrong match: a checkpoint written BEFORE an
    app started cannot have been written BY it.

    Getting this wrong is not cosmetic. Unmatched, a finished 13.8 h round inherits a stranger's
    six-minute elapsed and reports ``$0.17`` against a $40 guardrail, and — worse — the VRAM gauge
    would be scraped from that stranger's logs and reported against THIS campaign's 76.36 GiB
    tripwire. A wrong number under a right label is the failure this whole page exists to prevent.
    """
    if not app.get("available"):
        return False
    if not checkpoints:
        return True  # nothing to contradict it — the warming case
    try:
        started = datetime.fromisoformat(str(app.get("started_at")))
    except (TypeError, ValueError):
        return True  # no start time to compare against; do not manufacture a mismatch
    newest = checkpoints[-1].get("_written")
    return newest is None or newest >= started


def fetch_log_signals(app_id: str, tail: int, timeout: float, keep_tail: bool) -> dict:
    """The VRAM gauge + alert lines, from a BOUNDED log fetch (never `-f`, which never returns).

    `modal app logs` without `-f` fetches the last N entries and exits, so this cannot hang the
    refresh. The VRAM gauge is the only campaign tripwire that exists nowhere but the log: the card
    carries `expected_peak_gib: 76.36` against a 79.25 GiB usable ceiling, so ~2.9 GiB of headroom
    is the entire margin and the first sign of budget drift would otherwise be an OOM at hour N.
    """
    rc, out, err = _run(["modal", "app", "logs", app_id, "--tail", str(tail)], timeout)
    if rc != 0 and not out:
        return {"available": False, "reason": (err or "log fetch failed")[:200]}
    vram = [
        {"step": int(m.group(1)), "allocated_gib": float(m.group(2)), "reserved_gib": float(m.group(3))}
        for m in _VRAM_RE.finditer(out)
    ]
    alerts, seen = [], set()
    for line in out.splitlines():
        if _ALERT_RE.search(line):
            clean = _redact(line.strip())[:220]
            if clean not in seen:
                seen.add(clean)
                alerts.append(clean)
    payload = {
        "available": True,
        "reason": "",
        "lines_scanned": out.count("\n") + 1,
        "vram": vram[-40:],
        "alerts": alerts[-6:],
    }
    if keep_tail:
        payload["tail"] = [_redact(x)[:220] for x in out.splitlines()[-30:]]
    return payload


# ──────────────────────────────────────────────────────────────────────────────────────────────
# status assembly (typed slots — this dict IS the artifact; the HTML only renders it)
# ──────────────────────────────────────────────────────────────────────────────────────────────


def build_status(
    *, recipe: dict, card: dict, ledger: dict, volume: str, ckpts: dict, app: dict,
    logs: dict, stale_minutes: float,
) -> dict:
    now = datetime.now(timezone.utc).astimezone()
    checkpoints = ckpts.get("checkpoints", [])
    max_steps = recipe.get("max_steps")
    ckpt_every = recipe.get("checkpoint_every") or 0

    current_step = checkpoints[-1]["step"] if checkpoints else 0
    progress = (current_step / max_steps) if (max_steps and max_steps > 0) else None

    newest = checkpoints[-1]["_written"] if checkpoints else None
    age_min = ((now - newest).total_seconds() / 60.0) if newest else None
    if age_min is not None and age_min < 0:
        age_min = 0.0  # clock/zone skew: never report a checkpoint from the future

    # MEASURED throughput from consecutive checkpoint mtimes — the median interval divided by the
    # checkpoint stride. Median, not mean: one slow first interval (cold Volume shard reads) would
    # drag a mean and misreport the steady state the ETA depends on.
    intervals = [
        (b["_written"] - a["_written"]).total_seconds()
        for a, b in zip(checkpoints, checkpoints[1:])
        if a["_written"] and b["_written"] and b["_written"] >= a["_written"]
    ]
    measured_s_per_it = (
        statistics.median(intervals) / ckpt_every if intervals and ckpt_every else None
    )
    expected_s_per_it = card.get("expected_s_per_it")

    remaining = (max_steps - current_step) if (max_steps and current_step < max_steps) else 0
    rate = measured_s_per_it or expected_s_per_it
    eta_hours = (remaining * rate / 3600.0) if (rate and remaining) else (0.0 if max_steps else None)

    matched = app_matches_run(app, checkpoints)
    elapsed_hours = app.get("elapsed_hours") if matched else None
    elapsed_source = "modal app record" if elapsed_hours is not None else None
    if elapsed_hours is None and len(checkpoints) >= 2:
        first, last = checkpoints[0]["_written"], checkpoints[-1]["_written"]
        if first and last:
            # Floor, not the truth: it misses the pre-first-checkpoint model load and the tail after
            # the last one. Labelled as a floor on the page rather than quietly presented as elapsed.
            elapsed_hours = (last - first).total_seconds() / 3600.0
            elapsed_source = "checkpoint span (floor — excludes load + tail)"

    # Accrued cost is DERIVED (elapsed x rate) and is NOT the ledger. Both are shown, labelled: the
    # ledger is authoritative for the session, this figure is what the run is spending right now.
    hourly = recipe.get("hourly_rate_usd") or 0.0
    run_cost = (elapsed_hours * hourly) if (elapsed_hours and hourly) else None

    peak = max((v["allocated_gib"] for v in logs.get("vram", [])), default=None)
    reserved_peak = max((v["reserved_gib"] for v in logs.get("vram", [])), default=None)
    expected_peak = card.get("expected_peak_gib")
    usable = recipe.get("gpu_usable_gib")

    # The in-container liveness deadline, restated from the config so the page and the watchdog
    # cannot disagree about what "too long" means (D-NOHARDCODE).
    watchdog_minutes = None
    if recipe.get("checkpoint_expected_minutes"):
        watchdog_minutes = (
            float(recipe["checkpoint_expected_minutes"])
            * float(recipe.get("checkpoint_stall_multiplier") or 1.0)
        )

    if not ckpts.get("available"):
        state, headline = "no-data", "No output directory on the Volume yet"
        detail = (
            "Nothing has been committed under this run's output_dir. Expected before the first "
            "checkpoint lands (~"
            + (f"{recipe.get('checkpoint_expected_minutes')} min" if recipe.get("checkpoint_expected_minutes") else "one checkpoint interval")
            + " into training), or the run has not been dispatched. Reason: "
            + (ckpts.get("reason") or "unknown")
        )
    elif not checkpoints:
        state, headline = "warming", "Volume dir exists — no checkpoint committed yet"
        detail = (
            "The run has written something under output_dir but no `checkpoint-step-*` dir has "
            "committed. The first checkpoint lands at step "
            f"{ckpt_every or '?'}."
        )
    elif max_steps and current_step >= max_steps:
        state, headline = "complete", f"Run complete — step {current_step} / {max_steps}"
        detail = "Every planned step has a committed checkpoint. Next: the gated `--mode sample` render."
    elif age_min is not None and age_min > stale_minutes:
        state, headline = "stale", f"STALE — newest checkpoint is {_fmt_age(age_min)} old"
        detail = (
            f"The freshness gate is {stale_minutes:.0f} min and the expected cadence is "
            f"{recipe.get('checkpoint_expected_minutes') or '?'} min. Artifacts, not processes: "
            "check `modal app logs` before assuming the container is fine."
        )
    else:
        state, headline = "live", f"Live — step {current_step} / {max_steps or '?'}"
        detail = f"Newest checkpoint committed {_fmt_age(age_min)} ago."

    return {
        "generated_at": now.isoformat(),
        "state": state,
        "headline": headline,
        "detail": detail,
        "volume": volume,
        "output_dir": recipe.get("output_dir"),
        "recipe": recipe,
        "card": card,
        "current_step": current_step,
        "max_steps": max_steps,
        "progress_frac": progress,
        "checkpoints_written": len(checkpoints),
        "checkpoints_expected": (max_steps // ckpt_every) if (max_steps and ckpt_every) else None,
        "keep_checkpoints": recipe.get("keep_checkpoints"),
        "newest_checkpoint": checkpoints[-1]["name"] if checkpoints else None,
        "newest_checkpoint_age_min": age_min,
        "stale_gate_min": stale_minutes,
        "watchdog_deadline_min": watchdog_minutes,
        "latest_loss": checkpoints[-1]["loss"] if checkpoints else None,
        "loss_series": [{"step": c["step"], "loss": c["loss"]} for c in checkpoints],
        "measured_s_per_it": measured_s_per_it,
        "expected_s_per_it": expected_s_per_it,
        "elapsed_hours": elapsed_hours,
        "elapsed_source": elapsed_source,
        "app_matches_run": matched,
        "est_hours": recipe.get("est_hours"),
        "eta_hours": eta_hours,
        "peak_gib": peak,
        "reserved_peak_gib": reserved_peak,
        "expected_peak_gib": expected_peak,
        "gpu_usable_gib": usable,
        "vram_headroom_gib": (usable - peak) if (usable and peak) else None,
        "run_cost_usd_est": run_cost,
        "cost_guardrail_usd": recipe.get("cost_guardrail_usd"),
        "session_spent_usd": ledger.get("spent_usd"),
        "session_cap_usd": ledger.get("session_cap_usd"),
        "spend_source": ledger.get("spend_source"),
        "app": app,
        "logs": {k: v for k, v in logs.items() if k != "vram"},
        "checkpoint_rows": [
            {k: v for k, v in c.items() if not k.startswith("_")} for c in checkpoints[-14:]
        ],
        "volume_reason": ckpts.get("reason", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────────────────────
# rendering — ONE self-contained file: inline CSS, inline SVG, ~20 lines of inline JS, no CDN
# ──────────────────────────────────────────────────────────────────────────────────────────────

_CSS = """
:root{color-scheme:light dark;
  --bg:#fbfbfc;--panel:#fff;--ink:#14161a;--muted:#5d6470;--line:#e3e6ea;
  --accent:#3b5bdb;--ok:#2f9e44;--warn:#e8a317;--bad:#e03131;--idle:#868e96;
  --track:#e9ecef;--shadow:0 1px 2px rgba(16,20,28,.06),0 8px 24px rgba(16,20,28,.05);}
@media (prefers-color-scheme:dark){:root{
  --bg:#0e0f11;--panel:#16181c;--ink:#e8eaed;--muted:#9aa2ad;--line:#272b31;
  --accent:#7d92f5;--ok:#51cf66;--warn:#ffd43b;--bad:#ff6b6b;--idle:#7a828c;
  --track:#23262b;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);}}
html[data-theme=light]{color-scheme:light;
  --bg:#fbfbfc;--panel:#fff;--ink:#14161a;--muted:#5d6470;--line:#e3e6ea;
  --accent:#3b5bdb;--ok:#2f9e44;--warn:#e8a317;--bad:#e03131;--idle:#868e96;
  --track:#e9ecef;--shadow:0 1px 2px rgba(16,20,28,.06),0 8px 24px rgba(16,20,28,.05);}
html[data-theme=dark]{color-scheme:dark;
  --bg:#0e0f11;--panel:#16181c;--ink:#e8eaed;--muted:#9aa2ad;--line:#272b31;
  --accent:#7d92f5;--ok:#51cf66;--warn:#ffd43b;--bad:#ff6b6b;--idle:#7a828c;
  --track:#23262b;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 64px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between}
h1{font-size:19px;font-weight:650;margin:0;letter-spacing:-.01em}
h2{font-size:12px;font-weight:650;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin:30px 0 10px}
.sub{color:var(--muted);font-size:12.5px;margin:4px 0 0}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace}
button.theme{background:var(--panel);color:var(--muted);border:1px solid var(--line);
  border-radius:7px;padding:5px 11px;font-size:12px;cursor:pointer}
button.theme:hover{color:var(--ink)}
.banner{margin:18px 0 0;padding:15px 17px;border-radius:11px;border:1px solid var(--line);
  background:var(--panel);box-shadow:var(--shadow);border-left-width:5px}
.banner .hl{font-size:16px;font-weight:640}
.banner .dt{color:var(--muted);font-size:12.5px;margin-top:5px}
.banner.live{border-left-color:var(--ok)} .banner.stale{border-left-color:var(--bad)}
.banner.complete{border-left-color:var(--accent)} .banner.warming{border-left-color:var(--warn)}
.banner.no-data{border-left-color:var(--idle)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;vertical-align:1px}
.live .dot{background:var(--ok)} .stale .dot{background:var(--bad)}
.complete .dot{background:var(--accent)} .warming .dot{background:var(--warn)}
.no-data .dot{background:var(--idle)}
.cards{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(238px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px 15px;
  box-shadow:var(--shadow)}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.card .v{font-size:22px;font-weight:640;margin:7px 0 2px;letter-spacing:-.02em}
.card .v small{font-size:13px;font-weight:500;color:var(--muted);letter-spacing:0}
.card .n{font-size:12px;color:var(--muted);margin-top:5px}
.bar{height:7px;border-radius:4px;background:var(--track);overflow:hidden;margin-top:9px}
.bar>i{display:block;height:100%;border-radius:4px;background:var(--accent)}
.bar>i.ok{background:var(--ok)} .bar>i.warn{background:var(--warn)} .bar>i.bad{background:var(--bad)}
.tag{display:inline-block;font-size:11px;padding:2px 7px;border-radius:5px;border:1px solid var(--line);
  color:var(--muted);margin-left:6px}
.tag.ok{color:var(--ok);border-color:var(--ok)} .tag.bad{color:var(--bad);border-color:var(--bad)}
.tag.warn{color:var(--warn);border-color:var(--warn)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px;
  box-shadow:var(--shadow)}
svg{display:block;width:100%;height:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  font-weight:600;padding:0 10px 8px 0;border-bottom:1px solid var(--line)}
td{padding:7px 10px 7px 0;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:0}
.empty{color:var(--muted);font-size:13px;padding:10px 0}
ul.notes{margin:10px 0 0;padding-left:18px;color:var(--muted);font-size:12.5px}
ul.notes li{margin:5px 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;
  background:var(--track);padding:1.5px 5px;border-radius:4px}
pre{margin:0;overflow-x:auto;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11.5px;color:var(--muted);white-space:pre-wrap;word-break:break-word}
"""

_JS = """
(function(){
  var k='h3statustheme', r=document.documentElement, b=document.getElementById('themebtn');
  function set(v){ if(v){r.setAttribute('data-theme',v);} else {r.removeAttribute('data-theme');}
    b.textContent = v==='dark' ? 'Light' : (v==='light' ? 'Auto' : 'Dark'); }
  var cur=null; try{cur=localStorage.getItem(k);}catch(e){}
  set(cur);
  b.addEventListener('click',function(){
    var next = cur===null ? 'dark' : (cur==='dark' ? 'light' : null);
    cur=next; try{ next?localStorage.setItem(k,next):localStorage.removeItem(k); }catch(e){}
    set(next);
  });
})();
"""


def _esc(value: object) -> str:
    return html.escape("—" if value is None else str(value))


def _bar(frac: float | None, tone: str = "") -> str:
    if frac is None:
        return ""
    pct = max(0.0, min(1.0, frac)) * 100
    return f'<div class="bar"><i class="{tone}" style="width:{pct:.1f}%"></i></div>'


def _card(key: str, value: str, note: str = "", bar: str = "") -> str:
    note_html = f'<div class="n">{note}</div>' if note else ""
    return (
        f'<div class="card"><div class="k">{_esc(key)}</div>'
        f'<div class="v">{value}</div>{bar}{note_html}</div>'
    )


def _loss_svg(series: list[dict], width: int = 1000, height: int = 230) -> str:
    """Loss trend as a hand-built inline SVG — raw points + a moving average, no chart library.

    A library would need a CDN, and a CDN would break the "self-contained artifact" contract the
    moment the operator opens this offline or in a sandboxed viewer.
    """
    if len(series) < 2:
        return (
            '<div class="empty">Not enough checkpoints for a trend yet — the loss series is built '
            "from committed <code>checkpoint-step-*-loss-*</code> directory names, so it needs at "
            "least two.</div>"
        )
    pad_l, pad_r, pad_t, pad_b = 52, 14, 16, 30
    steps = [p["step"] for p in series]
    losses = [p["loss"] for p in series]
    x0, x1 = min(steps), max(steps)
    y0, y1 = min(losses), max(losses)
    span = (y1 - y0) or 1.0
    y0, y1 = y0 - span * 0.12, y1 + span * 0.12
    span = y1 - y0

    def px(step: int) -> float:
        return pad_l + (step - x0) / max(1, (x1 - x0)) * (width - pad_l - pad_r)

    def py(loss: float) -> float:
        return pad_t + (1 - (loss - y0) / span) * (height - pad_t - pad_b)

    raw = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(steps, losses))

    window = max(2, min(9, len(losses) // 4 or 2))
    smooth_pts = []
    for i in range(len(losses)):
        lo = max(0, i - window + 1)
        smooth_pts.append((steps[i], sum(losses[lo : i + 1]) / (i - lo + 1)))
    smooth = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in smooth_pts)

    grid_lines, y_labels = [], []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = y1 - frac * span
        y = pad_t + frac * (height - pad_t - pad_b)
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            'stroke="currentColor" stroke-opacity=".12"/>'
        )
        y_labels.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="currentColor" fill-opacity=".55">{value:.3f}</text>'
        )
    x_labels = "".join(
        f'<text x="{px(s):.1f}" y="{height - 8}" text-anchor="middle" font-size="11" '
        f'fill="currentColor" fill-opacity=".55">{s}</text>'
        for s in (x0, steps[len(steps) // 2], x1)
    )
    dots = "".join(
        f'<circle cx="{px(s):.1f}" cy="{py(v):.1f}" r="2.1" fill="var(--accent)" '
        f'fill-opacity=".55"><title>step {s} · loss {v:.4f}</title></circle>'
        for s, v in zip(steps, losses)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Loss per committed checkpoint, step {x0} to {x1}">'
        + "".join(grid_lines)
        + "".join(y_labels)
        + x_labels
        + f'<polyline points="{raw}" fill="none" stroke="var(--accent)" stroke-opacity=".32" '
        'stroke-width="1.4"/>'
        + f'<polyline points="{smooth}" fill="none" stroke="var(--accent)" stroke-width="2.4" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
        + dots
        + "</svg>"
    )


def render_html(status: dict, command: str) -> str:
    state = status["state"]
    recipe, card = status["recipe"], status["card"]

    # ── progress ──────────────────────────────────────────────────────────────────────────────
    cards = [
        _card(
            "Progress",
            f'{status["current_step"]} <small>/ {_esc(status["max_steps"])} steps</small>',
            (
                f'{status["progress_frac"] * 100:.1f}% complete'
                if status["progress_frac"] is not None
                else "no committed checkpoint yet"
            ),
            _bar(status["progress_frac"], "ok" if state in ("live", "complete") else ""),
        )
    ]

    # ── wall clock ────────────────────────────────────────────────────────────────────────────
    est_h = status["est_hours"]
    elapsed = status["elapsed_hours"]
    eta_note = (
        f"ETA {_fmt_hm(status['eta_hours'])} remaining"
        if status["eta_hours"] is not None
        else "ETA unavailable"
    )
    cards.append(
        _card(
            "Elapsed",
            f'{_fmt_hm(elapsed)} <small>/ {_fmt_hm(est_h)} est</small>',
            f"{eta_note} · source: {_esc(status.get('elapsed_source'))}",
            _bar((elapsed / est_h) if (elapsed and est_h) else None,
                 "warn" if (elapsed and est_h and elapsed > est_h) else ""),
        )
    )

    # ── liveness (ARTIFACT freshness, the house signal) ────────────────────────────────────────
    age = status["newest_checkpoint_age_min"]
    gate = status["stale_gate_min"]
    tone = "bad" if (age is not None and age > gate) else "ok"
    watchdog = status["watchdog_deadline_min"]
    cards.append(
        _card(
            "Newest checkpoint",
            f'{_fmt_age(age)} <small>ago</small>',
            (
                f'gate {gate:.0f} min'
                + (f' · in-container watchdog {watchdog:.0f} min' if watchdog else "")
                + f' · {_esc(status["newest_checkpoint"])}'
            ),
            _bar((age / gate) if age is not None else None, tone),
        )
    )

    # ── checkpoints written ───────────────────────────────────────────────────────────────────
    expected_ckpts = status["checkpoints_expected"]
    keep = status["keep_checkpoints"]
    cards.append(
        _card(
            "Checkpoints",
            f'{status["checkpoints_written"]} <small>/ {_esc(expected_ckpts)}</small>',
            (
                f'every {_esc(recipe.get("checkpoint_every"))} steps · keep_checkpoints '
                + ("null — nothing is pruned" if keep is None else f"{keep} — PRUNING IS ON")
            ),
            _bar(
                (status["checkpoints_written"] / expected_ckpts) if expected_ckpts else None,
                "ok",
            ),
        )
    )

    # ── VRAM against the campaign tripwire ────────────────────────────────────────────────────
    peak, expected_peak = status["peak_gib"], status["expected_peak_gib"]
    usable = status["gpu_usable_gib"]
    if peak is None:
        # Say WHICH kind of nothing this is. "no gauge yet" and "logs not attributed" look identical
        # on the page but mean opposite things — one is normal, the other is a wrong-app read averted.
        vram_note = (
            "gauge logged in the checkpoint branch — appears with the first checkpoint"
            if status["logs"].get("available")
            else _esc(status["logs"].get("reason") or "no log source")
        )
        if expected_peak:
            vram_note += f" · expected {expected_peak}"
        vram_tag, vram_bar = "", ""
    else:
        drift = (peak - expected_peak) if expected_peak else None
        over = drift is not None and drift > 0.5
        vram_tag = (
            f'<span class="tag {"bad" if over else "ok"}">'
            f'{"+" if (drift or 0) >= 0 else ""}{drift:.2f} GiB vs expected</span>'
            if drift is not None else ""
        )
        vram_note = (
            f'expected {expected_peak} · usable {usable} · headroom '
            f'{status["vram_headroom_gib"]:.2f} GiB'
            if (expected_peak and usable and status["vram_headroom_gib"] is not None)
            else f"expected {_esc(expected_peak)}"
        )
        vram_bar = _bar((peak / usable) if usable else None, "bad" if over else "ok")
    cards.append(
        _card(
            "Peak VRAM",
            (f"{peak:.2f} <small>GiB</small>" if peak is not None else "—") + vram_tag,
            vram_note,
            vram_bar,
        )
    )

    # ── throughput ────────────────────────────────────────────────────────────────────────────
    measured, expected_it = status["measured_s_per_it"], status["expected_s_per_it"]
    thr_note = (
        f"expected {expected_it} s/it (measured on a real A100)"
        if expected_it else "no expectation on the card"
    )
    thr_tag = ""
    if measured and expected_it:
        delta = measured - float(expected_it)
        thr_tag = (
            f'<span class="tag {"warn" if delta > float(expected_it) * 0.15 else "ok"}">'
            f'{"+" if delta >= 0 else ""}{delta:.1f} s</span>'
        )
    cards.append(
        _card(
            "Throughput",
            (f"{measured:.1f} <small>s/it</small>" if measured else "—") + thr_tag,
            thr_note + " · derived from checkpoint mtimes (median interval / stride)",
        )
    )

    # ── spend ─────────────────────────────────────────────────────────────────────────────────
    run_cost, guard = status["run_cost_usd_est"], status["cost_guardrail_usd"]
    cards.append(
        _card(
            "This run (accrued est.)",
            (f"${run_cost:,.2f}" if run_cost is not None else "—")
            + (f' <small>/ ${guard:,.0f} guardrail</small>' if guard else ""),
            f'elapsed x ${_esc(recipe.get("hourly_rate_usd"))}/h — DERIVED, not the ledger'
            + (f' · plan estimate ${card.get("est_cost_usd")}' if card.get("est_cost_usd") else ""),
            _bar((run_cost / guard) if (run_cost and guard) else None,
                 "bad" if (run_cost and guard and run_cost > guard) else "ok"),
        )
    )
    spent, cap = status["session_spent_usd"], status["session_cap_usd"]
    cards.append(
        _card(
            "Session ledger",
            (f"${spent:,.2f}" if spent is not None else "—")
            + (f' <small>/ ${cap:,.0f} cap</small>' if cap else ""),
            f'authoritative · {_esc(status["spend_source"])}',
            _bar((spent / cap) if (spent and cap) else None,
                 "bad" if (spent and cap and spent > cap * 0.9) else "ok"),
        )
    )

    # ── checkpoint table ──────────────────────────────────────────────────────────────────────
    rows = status["checkpoint_rows"]
    if rows:
        body = "".join(
            "<tr>"
            f'<td class="mono">{r["step"]}</td>'
            f'<td class="mono">{r["loss"]:.4f}</td>'
            f'<td class="mono">{_esc((r.get("written_at") or "")[:16].replace("T", " "))}</td>'
            f'<td class="mono">{_esc(r["name"])}</td>'
            "</tr>"
            for r in reversed(rows)
        )
        table = (
            "<table><thead><tr><th>Step</th><th>Loss</th><th>Committed</th><th>Directory</th>"
            f"</tr></thead><tbody>{body}</tbody></table>"
        )
    else:
        table = (
            '<div class="empty">No <code>checkpoint-step-*</code> directory on '
            f'<code>{_esc(status["volume"])}</code> under <code>{_esc(status["output_dir"])}</code> '
            f'yet. {_esc(status["volume_reason"])}</div>'
        )

    alerts = status["logs"].get("alerts") or []
    alerts_html = (
        '<div class="panel" style="margin-top:10px"><pre>'
        + html.escape("\n".join(alerts))
        + "</pre></div>"
        if alerts
        else ""
    )
    tail = status["logs"].get("tail")
    tail_html = (
        '<h2>Log tail</h2><div class="panel"><pre>' + html.escape("\n".join(tail)) + "</pre></div>"
        if tail
        else ""
    )

    app = status["app"]
    if not app.get("available"):
        app_line = f'no app record ({_esc(app.get("reason"))})'
    else:
        app_line = (
            f'{_esc(app.get("app_id"))} · state {_esc(app.get("state"))} · tasks '
            f'{_esc(app.get("tasks"))} · '
            + (
                '<span class="tag ok">attributed to this run</span>'
                if status.get("app_matches_run")
                else '<span class="tag warn">NOT this run — started after the newest '
                     "checkpoint; pin with --app-id</span>"
            )
        )

    generated = status["generated_at"][:19].replace("T", " ")
    title = f'{card.get("slug") or Path(str(recipe.get("config_path"))).stem} — run status'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>{_esc(title)}</h1>
      <p class="sub mono">{_esc(recipe.get("config_path"))} &middot; {_esc(status["volume"])}:{_esc(status["output_dir"])}</p>
    </div>
    <button class="theme" id="themebtn" type="button">Dark</button>
  </header>

  <div class="banner {state}">
    <div class="hl"><span class="dot"></span>{_esc(status["headline"])}</div>
    <div class="dt">{_esc(status["detail"])}</div>
  </div>

  <h2>At a glance</h2>
  <div class="cards">{"".join(cards)}</div>

  <h2>Loss per committed checkpoint</h2>
  <div class="panel">
    {_loss_svg(status["loss_series"])}
    <ul class="notes">
      <li><strong>Loss is sanity-only.</strong> Watch it for NaN or no-descent; a descending loss is
      never a success verdict (D-8-CONVERGENCE). The evidence hierarchy is the rendered clips.</li>
      <li>Faint line = raw per-checkpoint loss (one noisy minibatch each). Solid line = trailing
      mean, which is the only part worth reading as a trend.</li>
    </ul>
  </div>

  <h2>Committed checkpoints{" (latest 14)" if len(status["loss_series"]) > 14 else ""}</h2>
  <div class="panel">{table}</div>

  <h2>Signals</h2>
  <div class="panel">
    <ul class="notes" style="padding-left:16px">
      <li>Modal app: <span class="mono">{app_line}</span></li>
      <li>Log window: {_esc(status["logs"].get("lines_scanned", "not fetched"))} lines scanned
      {"— " + _esc(status["logs"].get("reason")) if not status["logs"].get("available") else ""}</li>
      <li>Alerts matched: {len(alerts)}</li>
    </ul>
    {alerts_html}
  </div>
  {tail_html}

  <h2>Regenerate</h2>
  <div class="panel">
    <p class="sub" style="margin:0 0 8px">One command. Zero metered spend — read-only Modal CLI only.</p>
    <pre>{_esc(command)}</pre>
  </div>

  <h2>Scope</h2>
  <div class="panel">
    <ul class="notes" style="padding-left:16px">
      <li><strong>Liveness is artifact freshness, not process state.</strong> A container can be
      alive, billing, and wedged; only a newly committed checkpoint proves work landed.</li>
      <li><strong>This run's cost is derived</strong> (elapsed x hourly rate) and is not a ledger
      entry. <code>SESSION-STATE.json</code> stays the single numeric source of truth for spend.</li>
      <li><strong>D-10-SCOPEGUARD.</strong> This phase grades whether the LOOP works, never adapter
      quality. Nothing on this page is a quality verdict.</li>
      <li>Generated {_esc(generated)} (local time).</li>
    </ul>
  </div>
</div>
<script>{_JS}</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from signet_trainer.harness_state import session_ledger_path  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Self-contained HTML run-status page for a gated signet-trainer run "
                    "(read-only Modal CLI; zero metered spend).",
    )
    parser.add_argument("--config", default="configs/<your-h3-run>.yaml",
                        help="REQUIRED in practice — pass your own training config. Supplies — supplies output_dir/max_steps/est_hours/guardrail.")
    parser.add_argument("--card", default="",
                        help="Campaign card *.state.yaml (expected_peak_gib / expected_s_per_it). "
                             "Default: derived from the config stem.")
    parser.add_argument("--session-state", default=session_ledger_path(),
                        help="Spend ledger — the single numeric source of truth for spend/cap.")
    parser.add_argument("--volume", default=None,
                        help="Modal Volume holding <output_dir>/checkpoint-step-*. Default: the "
                             "loaded config's modal.checkpoints_volume_name (D-NOHARDCODE) — falls "
                             "back to the schema default only when the config never set it.")
    parser.add_argument("--out", default="_h3_status",
                        help="Output directory for index.html + status.json (gitignored).")
    # ⛔ PRE-RENAME DEFAULT ON PURPOSE (matches SignetConfig.modal.app_name / app.py APP_NAME):
    # this is the name of an EXISTING Modal App that in-flight runs are dispatched under. Renaming
    # it to "signet-trainer" would silently attribute nothing and report a live run as missing.
    # issue #19 item 3 (D-NOHARDCODE): default resolved from the loaded config's modal.app_name
    # below (falling back to the schema default), never a literal repeated here.
    parser.add_argument("--app-name", default=None, help="Modal app name to attribute. Default: "
                         "the loaded config's modal.app_name.")
    parser.add_argument("--app-id", default="", help="Pin a specific Modal app id.")
    parser.add_argument("--stale-minutes", type=float, default=60.0,
                        help="Flag the run STALE when the newest checkpoint is older than this.")
    parser.add_argument("--log-tail", type=int, default=1500,
                        help="How many log entries to fetch for the VRAM gauge (bounded, never -f).")
    parser.add_argument("--no-logs", action="store_true",
                        help="Skip the log fetch entirely (no VRAM gauge, faster refresh).")
    parser.add_argument("--include-log-tail", action="store_true",
                        help="Embed the last 30 redacted log lines. OFF by default (property hygiene).")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-CLI-call timeout, seconds.")
    parser.add_argument("--open", action="store_true", help="Open the page in a browser when done.")
    args = parser.parse_args(argv)
    # issue #19 item 3 — remember whether --volume was passed explicitly BEFORE defaulting it from
    # the config below, so the reproduction command only echoes --volume when the operator actually
    # overrode it (not merely because the config's own checkpoints_volume_name differs from the
    # schema default — that case is already implied by --config).
    volume_explicit = args.volume is not None

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        print(f"[status] config not found: {config_path}", file=sys.stderr)
        return 2

    if shutil.which("modal") is None:
        print("[status] WARNING: `modal` is not on PATH — every remote panel will be empty.",
              file=sys.stderr)

    recipe = load_recipe(config_path)
    output_dir = recipe.get("output_dir")
    if not output_dir:
        print(f"[status] {config_path} has no output_dir — nothing to point at.", file=sys.stderr)
        return 2
    if args.volume is None or args.app_name is None:
        # issue #19 item 3 (D-NOHARDCODE) — config-first: the loaded config's own
        # modal.checkpoints_volume_name / modal.app_name win; only a config that never set a field
        # falls through to the schema's own default (never a literal repeated here, so the two
        # can't drift).
        from signet_trainer.config.schema import ModalConfig  # noqa: PLC0415
        if args.volume is None:
            args.volume = recipe.get("checkpoints_volume_name") or ModalConfig().checkpoints_volume_name
        if args.app_name is None:
            args.app_name = recipe.get("app_name") or ModalConfig().app_name

    card_path = Path(args.card) if args.card else (
        REPO_ROOT / ".planning/harness/cards"
        / f"TRAINING-CARD-{config_path.stem.replace('_', '-').removesuffix('-r1')}.state.yaml"
    )
    if not card_path.is_absolute():
        card_path = REPO_ROOT / card_path
    card = load_card(card_path)

    session_path = Path(args.session_state)
    if not session_path.is_absolute():
        session_path = REPO_ROOT / session_path
    ledger = load_ledger(session_path)

    ckpts = discover_checkpoints(args.volume, output_dir, args.timeout)
    app = discover_app(args.app_name, args.app_id or None, args.timeout)

    # The log fetch is gated on ATTRIBUTION, not merely on an app existing. Scraping the VRAM gauge
    # out of an app that did not write these checkpoints would report another dispatch's peak
    # against this campaign's 76.36 GiB tripwire — a wrong number under a right label.
    matched = app_matches_run(app, ckpts.get("checkpoints", []))
    logs: dict = {"available": False, "reason": "log fetch skipped (--no-logs)"}
    if args.no_logs:
        pass
    elif not app.get("available"):
        logs = {"available": False, "reason": f'no Modal app to read logs from '
                                              f'({app.get("reason", "")})'.strip()}
    elif not matched:
        logs = {
            "available": False,
            "reason": f'newest app {app.get("app_id")} started after the newest checkpoint was '
                      "written, so it is a DIFFERENT dispatch — logs not attributed. Pin the right "
                      "one with --app-id.",
        }
    else:
        logs = fetch_log_signals(
            str(app["app_id"]), args.log_tail, args.timeout, args.include_log_tail
        )

    status = build_status(
        recipe=recipe, card=card, ledger=ledger, volume=args.volume, ckpts=ckpts,
        app=app, logs=logs, stale_minutes=args.stale_minutes,
    )
    status["vram_series"] = logs.get("vram", [])

    command = (
        f"MSYS_NO_PATHCONV=1 PYTHONPATH=src python scripts/_h3_run_status.py "
        f"--config {args.config}"
        + (f" --volume {args.volume}" if volume_explicit else "")
        + (" --no-logs" if args.no_logs else "")
    )

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.html"
    index_path.write_text(render_html(status, command), encoding="utf-8")
    (out_dir / "status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")

    # Typed-state stdout line: name: value slots, quotable verbatim into a handoff or a card.
    print(
        "[status] "
        f"state: {status['state']} | "
        f"step: {status['current_step']}/{status['max_steps']} | "
        f"checkpoints: {status['checkpoints_written']}/{status['checkpoints_expected']} | "
        f"newest_age_min: {status['newest_checkpoint_age_min'] if status['newest_checkpoint_age_min'] is None else round(status['newest_checkpoint_age_min'])} | "
        f"loss: {status['latest_loss']} | "
        f"peak_gib: {status['peak_gib']} (expected {status['expected_peak_gib']}) | "
        f"s_per_it: {None if status['measured_s_per_it'] is None else round(status['measured_s_per_it'], 1)} | "
        f"elapsed_h: {None if status['elapsed_hours'] is None else round(status['elapsed_hours'], 2)}/{status['est_hours']} | "
        f"run_usd_est: {None if status['run_cost_usd_est'] is None else round(status['run_cost_usd_est'], 2)}/{status['cost_guardrail_usd']} | "
        f"session_usd: {status['session_spent_usd']}/{status['session_cap_usd']}"
    )
    print(f"[status] wrote {index_path}")
    print(f"[status] wrote {out_dir / 'status.json'}")
    if args.open:
        import webbrowser  # noqa: PLC0415

        webbrowser.open(index_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
