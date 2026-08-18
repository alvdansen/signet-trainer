#!/usr/bin/env python3
"""Fetch the H3 eval renders, build the finetune-gridwatch grid, serve it live, print a URL.

    PYTHONPATH=src python scripts/_h3_grid_serve.py

Idempotent and re-runnable. Safe to run while the renders are still trickling in: it fetches only
clips it does not already hold, restages, and — because `grid watch` re-scans and pushes over SSE —
an already-running server picks the new cells up without a restart.

⛔ finetune-gridwatch IS THE GRID. It is the operator's own first-party tool
(github.com/alvdansen/finetune-gridwatch, pinned + installed by scripts/setup_gridwatch.sh into an
isolated venv) and it is the ONLY sanctioned grid builder. This script stages files and calls
`grid`; it does not render a single cell itself. A hand-rolled grid is forbidden — the in-Modal
`index.html` that `h3_sample` writes is the artifact INDEX, explicitly not a replacement for this.

⛔ CLIPS, NEVER A FIRST FRAME (training-review §4, Pitfall 6). A still hides temporal corruption and
warp, which on a video LoRA is most of what there is to see. gridwatch renders `.mp4` cells as real
`<video>` elements; nothing here extracts a thumbnail.

⛔ ZERO METERED SPEND. `modal volume ls` / `modal volume get` only — both free. No `modal run`, no
`.remote()`. When there is nothing to fetch this prints the gated dispatch commands for a human to
run; it never dispatches one itself (the single-gate law).

THE LAYOUT PROBLEM, AND WHY IT IS THE WHOLE JOB
----------------------------------------------
gridwatch's lattice is fixed: rows are an INTEGER ``step``, columns are a STRING ``prompt``. The
eval is 5 configs x 6 prompts x 2 columns = 60 clips over TWO axes that must stay legible —
LENGTH (22 / 56 / 124 f, all on reference pair A+029) and REFERENCE (A+029 / B+029 / C+018, all at
22 f). That clean split is the entire design: if the grid does not preserve which axis varies, it
defeats it.

The mapping (``--rows frames``, the default):

  * ROW    = the frame count. 22 / 56 / 124 descend the rows, so LENGTH is a spatial axis and the
             row header reads as the frame count itself (gridwatch renders the bare row value).
  * COLUMN = ``{refs} · {NN} {role} · {prompt}``, REFERENCE-MAJOR. Sorting therefore lays the grid
             out as three blocks — A+029 (populated at all three lengths), then B+029, then C+018
             (each only at 22 f). The unrendered combinations become clean dashed RECTANGLES rather
             than a scattered checkerboard, which is the clean split made visible instead of hidden.
  * base and lora are ADJACENT columns inside each prompt group, so base-vs-adapter is a
    side-by-side read at identical seed, identical everything.

``--layout length`` pins one reference and drops the reference axis (dense 3 x 12); ``--layout
reference`` pins one length and drops the length axis (dense 1 x 36, prompt-major so the three
references sit next to each other per prompt); ``--layout all`` builds and serves all three.
``--rows step`` swaps the row axis to the checkpoint step (base = 0), which is the convergence
ladder view once more than one checkpoint has been rendered.

PROPERTY HYGIENE. Reference conditions are named by SUBJECT ID only (``A+029``), never by reference
filename, and no staging path reaches a label. The prompts are the operator's own eval set, read
from tracked configs.

⛔ THE BASE COLUMN IS FETCHED FROM A SHARED, CHECKPOINT-INDEPENDENT DIRECTORY (#12). ``h3_sample``
keys the base clips on ``(seed, frames, width, height, steps, refs)`` alone — never the checkpoint —
so every checkpoint sampled at the same geometry resolves ONE base directory, not one each.
``fetch_renders`` fetches that shared directory once and joins it onto every checkpoint-keyed render
via ``base_key_for``; staging never assumes a ``base/`` subdirectory nested under a checkpoint key.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# A legacy Windows console defaults to cp1252, whose charmap cannot encode an em dash or a warning
# glyph — so a diagnostic containing one raises UnicodeEncodeError and takes the whole tool down
# INSTEAD of printing the warning. Exactly the failure mode that makes the modal CLI die on its own
# success tick (see `_MODAL_ENV`). Reconfigure both streams up front; guarded, because a captured
# pipe or an older wrapper may not expose `reconfigure`, and this must never fail on import.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — never fail on import
            pass

# ``h3_render_key``: ``<checkpoint>_s<seed>_f<frames>_w<width>_h<height>_n<steps>_<ids>``. The
# checkpoint segment is a sanitized Volume dir name (``checkpoint-step-03000-loss-0.1933``) that
# keeps ``-`` and ``.`` but has no underscore, so a greedy head plus the anchored
# ``_s<d>_f<d>_w<d>_h<d>_n<d>_`` tail cannot mis-split. Widened by #22 finding 5 to carry the
# geometry axes the ``pipe(...)`` call reads exactly as literally as ``frame_count`` — a resolution
# or step-count probe used to resume the PREVIOUS geometry's clips under the old, narrower key.
_KEY_RE = re.compile(
    r"^(?P<ckpt>.+)_s(?P<seed>\d+)_f(?P<frames>\d+)_w(?P<width>\d+)_h(?P<height>\d+)_"
    r"n(?P<steps>\d+)_(?P<ids>.+)$"
)
# ``h3_base_render_key``: every axis above MINUS the checkpoint (#12) — no ``<checkpoint>_`` head,
# so this never collides with ``_KEY_RE``. Lives one level down, under the fixed ``base/`` dir name
# (``inference.samples_layout``'s layout), never mixed into the same listing as the checkpoint-keyed
# dirs, so the two regexes need not disambiguate a shared parent directory's rows against each other.
_BASE_KEY_RE = re.compile(
    r"^s(?P<seed>\d+)_f(?P<frames>\d+)_w(?P<width>\d+)_h(?P<height>\d+)_n(?P<steps>\d+)_(?P<ids>.+)$"
)
#: The fixed subdirectory name the shared base column lives under — never a render-key match itself.
_BASE_DIRNAME = "base"
_STEP_RE = re.compile(r"checkpoint-step-(\d+)")

# Windows-illegal filename characters + anything that would break the gridwatch template's
# path-segment-bounded capture.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_LABEL_MAX = 110  # keeps <repo>/_grid_.../<label>/step_NNN.mp4 clear of MAX_PATH

_MODAL_ENV = {
    # MSYS_NO_PATHCONV: Git Bash rewrites a leading-slash argument into a Windows path, and the
    # modal CLI then writes to the mangled target AND REPORTS SUCCESS. subprocess without a shell is
    # not subject to the rewrite, but this is set anyway because the commands this script PRINTS are
    # meant to be pasted into Git Bash, and because every verification below assumes it.
    "MSYS_NO_PATHCONV": "1",
    # PYTHONUTF8 / PYTHONIOENCODING: without these the modal CLI raises
    # `'charmap' codec can't encode character '✓'` when printing its success tick to a cp1252
    # console — AFTER the bytes are safely on disk. Measured on this machine: same download, exit 1
    # with the traceback vs exit 0 with UTF-8 set. Never trust the exit code alone; every fetch below
    # is verified by stat-ing the file it was supposed to produce.
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


# ──────────────────────────────────────────────────────────────────────────────────────────────
# process helpers
# ──────────────────────────────────────────────────────────────────────────────────────────────


def _run(cmd: list[str], timeout: float = 600.0) -> tuple[int, str, str]:
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
    except Exception as exc:  # noqa: BLE001
        return 1, "", f"{type(exc).__name__}: {exc}"


def _spawn(cmd: list[str], log_path: Path) -> int:
    """Start a long-lived server DETACHED, so it outlives this script. Returns the pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "ab")  # noqa: SIM115 — deliberately outlives this frame
    kwargs: dict = {"stdout": handle, "stderr": subprocess.STDOUT, "cwd": str(REPO_ROOT),
                    "env": {**os.environ, **_MODAL_ENV}}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs).pid  # noqa: S603


def _port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _clean_cli_noise(text: str) -> str:
    """Strip the modal CLI's rich error box so the reason reads as a sentence, not as ASCII art.

    Box-drawing lives in U+2500-U+257F, which a naive `[|+-]{2,}` filter does not touch — the
    surfaced reason then arrives as 60 characters of `─│┌└` and the actual message scrolls off.
    """
    stripped = re.sub(r"[─-╿|+]+", " ", text)
    return re.sub(r"\s{2,}", " ", stripped).strip()


def _modal_ls(volume: str, path: str) -> tuple[list[dict], str]:
    """`modal volume ls --json` → rows, or `([], reason)`. A missing path is data, not an error."""
    rc, out, err = _run(["modal", "volume", "ls", volume, path, "--json"], timeout=180)
    text = out.strip()
    if not text.lstrip().startswith("["):
        return [], (_clean_cli_noise(err or out)[:180] or f"no listing for {path} (rc={rc})")
    try:
        return [r for r in json.loads(text) if isinstance(r, dict)], ""
    except json.JSONDecodeError as exc:
        return [], f"unparseable listing for {path}: {exc}"


def _modal_get(volume: str, remote: str, local: Path) -> tuple[bool, str]:
    """Fetch ONE file and VERIFY it landed. Download-then-rename, so a kill leaves no half file.

    The exit code is deliberately not the test. The modal CLI can fail on its own success message
    (see `_MODAL_ENV`), and the MSYS landmine makes a mangled destination look like a clean success.
    The only thing that proves a fetch worked is a non-empty file at the path we asked for.
    """
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_suffix(local.suffix + ".part")
    part.unlink(missing_ok=True)
    rc, _out, err = _run(["modal", "volume", "get", volume, remote, str(part), "--force"])
    if not part.exists() or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        return False, f"rc={rc} — nothing landed at {part.name}. {err.strip()[:160]}"
    part.replace(local)
    return True, ""


# ──────────────────────────────────────────────────────────────────────────────────────────────
# the eval matrix, read from the tracked configs (never re-derived, never hardcoded)
# ──────────────────────────────────────────────────────────────────────────────────────────────


def _slugger():
    """The REAL `slug()` the render used — never a local re-implementation.

    The clip filenames on the Volume are `slug(prompt)_s{seed}.mp4` with a 60-char cap. Re-deriving
    that rule here would work until the day it drifts, and the failure would be silent: prompts
    would stop matching and cells would land in the wrong column with no error.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from signet_trainer.inference.grid import slug  # noqa: PLC0415

    return slug


def _base_render_key_fn():
    """The REAL `h3_base_render_key()` — the same never-re-implement rule as `_slugger`.

    A checkpoint-keyed render's shared base directory NAME must be computed by the exact function
    the render itself used, or a drift between this script's join logic and `h3_sample`'s would
    silently show the wrong base clips (or none) next to the right adapter clips.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from signet_trainer.inference.render_key import h3_base_render_key  # noqa: PLC0415

    return h3_base_render_key


def load_eval_matrix(config_paths: list[Path]) -> dict:
    """Read the sibling `--mode sample` configs into one prompt/axis map.

    The configs are the source of truth for BOTH axes: `validation.frame_count` is the length rung
    and `validation.reference_subject_ids` is the reference condition. Reading them (rather than
    parsing the render-key dir names alone) is what lets the grid label a cell with the question the
    config asked, and lets a missing render show up as a MISSING cell instead of silently vanishing.
    """
    import yaml  # noqa: PLC0415

    slug = _slugger()
    prompts: list[str] = []
    variants: list[dict] = []
    output_dirs, seeds = set(), set()
    for path in config_paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        validation = data.get("validation", {}) or {}
        cfg_prompts = list(validation.get("prompts") or [])
        if not cfg_prompts:
            continue
        if not prompts:
            prompts = cfg_prompts
        elif cfg_prompts != prompts:
            print(f"[grid] WARNING: {path.name} has a DIFFERENT prompt set — columns will not align "
                  "across configs. The clean-split eval requires one shared prompt set.")
        if data.get("output_dir"):
            output_dirs.add(str(data["output_dir"]))
        seeds.add(validation.get("seed", data.get("seed")))
        variants.append(
            {
                "config": path.name,
                "frames": int(validation.get("frame_count") or 0),
                "subject_ids": [str(s) for s in (validation.get("reference_subject_ids") or [])],
                "seed": validation.get("seed", data.get("seed")),
            }
        )
    if len(output_dirs) > 1:
        print(f"[grid] WARNING: configs disagree on output_dir {sorted(map(str, output_dirs))} — "
              "using the first. All sibling sample configs must share it (find_latest resolves the "
              "adapter under it).")
    # ⛔ SLUG COLLISION CHECK — a real defect detector, not a formality.
    #
    # `h3_sample` names every clip `{slug(prompt)}_s{seed}.mp4`, and `slug()` truncates at 60
    # characters. Two prompts that agree on their first 60 characters therefore produce ONE
    # filename, and `_render`'s resume rule ("an existing non-empty mp4 is work this render already
    # did") then SKIPS the second prompt entirely — so its cell shows the FIRST prompt's pixels
    # under the second prompt's label. Silent, at a valid shape, on the axis being measured.
    #
    # This is not hypothetical for the H3 eval set: prompts 5 and 6 are the audio-tag probe and the
    # SAME sentence with the tag omitted, which is precisely a pair that agrees on its first 60
    # characters — so the one comparison the pair exists to make is the one that cannot happen.
    slug_to_index: dict[str, int] = {}
    collisions: dict[str, list[int]] = {}
    for index, prompt in enumerate(prompts):
        token = slug(prompt)
        collisions.setdefault(token, []).append(index)
        slug_to_index.setdefault(token, index)
    colliding = {t: idx for t, idx in collisions.items() if len(idx) > 1}
    for token, indexes in colliding.items():
        human = ", ".join(str(i + 1) for i in indexes)
        print(f"[grid] ⛔ SLUG COLLISION: prompts {human} both render to {token!r}_s*.mp4.")
        print(f"[grid]    They agree on their first 60 characters, so h3_sample writes ONE file and "
              f"its resume-skip drops the later prompt(s). Only prompt {indexes[0] + 1} can appear "
              "in the grid; the others are NOT separately rendered.")
        print("[grid]    Fix belongs in the eval configs (differentiate the prompts inside the "
              "first 60 characters), not here — this grid cannot invent a clip that was skipped.")

    return {
        "prompts": prompts,
        "slug_to_index": slug_to_index,
        "collisions": colliding,
        "variants": variants,
        "output_dir": next(iter(output_dirs), None) if output_dirs else None,
        "seeds": sorted(str(s) for s in seeds),
    }


def parse_render_key(name: str) -> dict | None:
    match = _KEY_RE.match(name)
    if not match:
        return None
    step_match = _STEP_RE.search(match.group("ckpt"))
    return {
        "key": name,
        "checkpoint": match.group("ckpt"),
        "step": int(step_match.group(1)) if step_match else None,
        "seed": int(match.group("seed")),
        "frames": int(match.group("frames")),
        "width": int(match.group("width")),
        "height": int(match.group("height")),
        "steps": int(match.group("steps")),
        "subject_ids": match.group("ids").split("-"),
    }


def parse_base_render_key(name: str) -> dict | None:
    """Parse a shared BASE render-dir name (#12) — the same five axes minus the checkpoint."""
    match = _BASE_KEY_RE.match(name)
    if not match:
        return None
    return {
        "key": name,
        "seed": int(match.group("seed")),
        "frames": int(match.group("frames")),
        "width": int(match.group("width")),
        "height": int(match.group("height")),
        "steps": int(match.group("steps")),
        "subject_ids": match.group("ids").split("-"),
    }


def base_key_for(meta: dict) -> str:
    """The shared base-column key a checkpoint-keyed render (`parse_render_key`'s dict) resolves to.

    Delegates to `h3_base_render_key` — never a re-implementation (see `_base_render_key_fn`).
    """
    return _base_render_key_fn()(
        seed=meta["seed"],
        frame_count=meta["frames"],
        width=meta["width"],
        height=meta["height"],
        num_inference_steps=meta["steps"],
        subject_ids=meta["subject_ids"],
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────
# fetch
# ──────────────────────────────────────────────────────────────────────────────────────────────


def _fetch_column(
    volume: str, remote_dir: str, local_dir: Path, refetch: bool
) -> tuple[list[str], int, int, list[str]]:
    """Fetch every clip in ONE remote dir. Returns `(filenames, fetched, skipped, failed)`."""
    files, _why = _modal_ls(volume, remote_dir)
    names, fetched, skipped, failed = [], 0, 0, []
    for entry in files:
        if entry.get("type") != "file":
            continue
        filename = Path(str(entry.get("filename", ""))).name
        if not filename.lower().endswith((".mp4", ".webm", ".mov")):
            continue
        target = local_dir / filename
        if not refetch and target.exists() and target.stat().st_size > 0:
            skipped += 1
            names.append(filename)
            continue
        ok, why = _modal_get(volume, f"{remote_dir}/{filename}", target)
        if ok:
            fetched += 1
            names.append(filename)
        else:
            failed.append(f"{remote_dir}/{filename}: {why}")
    return names, fetched, skipped, failed


def fetch_renders(volume: str, samples_root: str, local_root: Path, refetch: bool) -> dict:
    """Mirror `<output_dir>/samples_h3` locally, one clip at a time, skipping what we already hold.

    Per-FILE rather than per-tree: a render trickles in over hours, so a whole-tree `--force` pull
    would re-download every finished clip on every refresh. The skip rule is `h3_sample`'s own —
    non-empty, not merely present — because a container killed mid-encode leaves a 0-byte file and
    skipping THAT would put a corrupt clip in the grid rather than re-fetch it.

    ⛔ #12 — THE BASE COLUMN IS FETCHED ONCE PER GEOMETRY, NOT ONCE PER CHECKPOINT. `h3_sample` no
    longer nests a `base/` dir under each checkpoint-keyed render; it writes ONE shared `base/<base
    key>/` per (seed, frames, width, height, steps, refs) and every checkpoint's render resolves the
    same key (`base_key_for`). Fetching it under the checkpoint-keyed dir the way the `lora/` column
    is fetched would just silently show an empty base column on every row — the directory nesting
    this function used to assume no longer exists on the Volume.
    """
    rows, reason = _modal_ls(volume, samples_root)
    if not rows:
        return {"available": False, "reason": reason, "renders": []}

    # ── the shared base column: one fetch pass, keyed by base_key, reused across every checkpoint ──
    base_local_by_key: dict[str, Path] = {}
    base_clips_by_key: dict[str, list[str]] = {}
    base_fetched = base_skipped = 0
    base_failed: list[str] = []
    base_rows, _why = _modal_ls(volume, f"{samples_root}/{_BASE_DIRNAME}")
    for row in base_rows:
        if row.get("type") != "dir":
            continue
        base_key = Path(str(row.get("filename", ""))).name
        if parse_base_render_key(base_key) is None:
            print(f"[grid] skipping unrecognised base dir {base_key!r} (not an h3_base_render_key name)")
            continue
        local_base_dir = local_root / _BASE_DIRNAME / base_key
        names, f, s, failed = _fetch_column(
            volume, f"{samples_root}/{_BASE_DIRNAME}/{base_key}", local_base_dir, refetch
        )
        base_local_by_key[base_key] = local_base_dir
        base_clips_by_key[base_key] = names
        base_fetched += f
        base_skipped += s
        base_failed.extend(failed)

    renders, fetched, skipped, failed = [], base_fetched, base_skipped, list(base_failed)
    for row in rows:
        if row.get("type") != "dir":
            continue
        key = Path(str(row.get("filename", ""))).name
        if key == _BASE_DIRNAME:
            continue  # handled above — the shared base column, not a checkpoint-keyed render
        meta = parse_render_key(key)
        if meta is None:
            print(f"[grid] skipping unrecognised render dir {key!r} (not an h3_render_key name)")
            continue
        local_key = local_root / key
        lora_names, f, s, lora_failed = _fetch_column(
            volume, f"{samples_root}/{key}/lora", local_key / "lora", refetch
        )
        fetched += f
        skipped += s
        failed.extend(lora_failed)
        # delta.json is the render's own provenance record — the measured base-vs-adapter delta, the
        # find_latest-resolved checkpoint, and the reference subject_ids. It is a few hundred bytes
        # and it is AUTHORITATIVE where the directory name is merely descriptive.
        delta_local = local_key / "delta.json"
        if refetch or not delta_local.exists():
            _modal_get(volume, f"{samples_root}/{key}/delta.json", delta_local)
        if delta_local.exists():
            try:
                meta["delta"] = json.loads(delta_local.read_text(encoding="utf-8"))
                declared = meta["delta"].get("references")
                if declared:
                    meta["subject_ids"] = [str(s) for s in declared]
            except Exception:  # noqa: BLE001 — provenance is a bonus, never a blocker
                pass
        # The shared base column this render joins against — resolved from THIS render's own axes
        # (post delta.json override, so an authoritative subject_ids correction is honoured), never
        # from a nested `base/` dir that no longer exists.
        this_base_key = base_key_for(meta)
        meta["base_key"] = this_base_key
        meta["clips"] = {"base": base_clips_by_key.get(this_base_key, []), "lora": lora_names}
        meta["local_by_column"] = {
            "base": base_local_by_key.get(this_base_key, local_root / _BASE_DIRNAME / this_base_key),
            "lora": local_key / "lora",
        }
        renders.append(meta)

    renders.sort(key=lambda r: (r["step"] or 0, r["frames"], "-".join(r["subject_ids"])))
    return {
        "available": True, "reason": "", "renders": renders,
        "fetched": fetched, "skipped": skipped, "failed": failed,
    }


# ──────────────────────────────────────────────────────────────────────────────────────────────
# staging — turn the eval matrix into gridwatch's (int step) x (str prompt) lattice
# ──────────────────────────────────────────────────────────────────────────────────────────────


def _safe_label(text: str) -> str:
    """A column label that is also a legal directory name on NTFS AND on POSIX.

    Windows rejects ``<>:"/\\|?*`` and control characters outright, and silently mangles names with
    a trailing dot or space — which would make the staged directory not the one gridwatch scans.
    """
    cleaned = re.sub(r"\s+", " ", _ILLEGAL.sub("", text)).strip().rstrip(". ")
    return cleaned[:_LABEL_MAX].strip().rstrip(". ") or "unlabelled"


# Staging artefacts that survive a restage: the built page + its cached assets (so a rebuild does
# not re-copy hundreds of MB), the detached watcher's log, and the prompt legend.
_STAGE_KEEP = {"grid-output", "watch.log", "legend.txt"}


def _clear_staged_columns(grid_dir: Path) -> None:
    """Remove the previously staged column dirs, and NOTHING else.

    Deterministic restaging matters: a layout/checkpoint switch must not leave a stale column
    standing. But `shutil.rmtree(grid_dir)` would also destroy `grid-output/assets`, forcing a full
    re-copy every refresh. Guarded twice — the target must live under this repo AND under a `_`
    scratch root — because a blanket delete driven by a CLI flag is exactly how a tool eats a repo.
    """
    grid_dir = grid_dir.resolve()
    if not grid_dir.exists():
        return
    if REPO_ROOT.resolve() not in grid_dir.parents or not any(
        part.startswith("_") for part in grid_dir.relative_to(REPO_ROOT.resolve()).parts
    ):
        raise RuntimeError(f"refusing to clear {grid_dir}: not a `_`-prefixed scratch dir in the repo")
    for child in grid_dir.iterdir():
        if child.name in _STAGE_KEEP:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _link_or_copy(src: Path, dest: Path) -> None:
    """Hardlink the clip into the grid tree; copy only when the filesystem refuses.

    60 clips at 124 f are hundreds of MB and gridwatch copies them AGAIN into `grid-output/assets`,
    so a staging copy would triple the footprint for no benefit — the staged file is a pure alias.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            if dest.stat().st_size == src.stat().st_size:
                return
        except OSError:
            pass
        dest.unlink(missing_ok=True)
    try:
        os.link(src, dest)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dest)


def stage_grid(
    *, renders: list[dict], matrix: dict, grid_dir: Path, layout: str, rows: str,
    pin_reference: str | None, pin_frames: int | None, prompt_chars: int,
) -> dict:
    """Stage `<grid_dir>/<column label>/step_<row>.mp4` for one layout. Returns a stage report."""
    slug_to_index = matrix["slug_to_index"]
    prompts = matrix["prompts"]

    grid_dir.mkdir(parents=True, exist_ok=True)
    _clear_staged_columns(grid_dir)

    staged, unmatched, columns, row_values = 0, [], set(), set()
    for render in renders:
        refs = "+".join(render["subject_ids"])
        frames = render["frames"]
        step = render["step"] or 0
        if layout == "length" and pin_reference and refs != pin_reference:
            continue
        if layout == "reference" and pin_frames and frames != pin_frames:
            continue
        for column, filenames in render["clips"].items():
            for filename in filenames:
                stem = Path(filename).stem
                clip_slug = re.sub(r"_s\d+$", "", stem)
                index = slug_to_index.get(clip_slug)
                if index is None:
                    unmatched.append(filename)
                    continue
                # Truncated to a FIXED width so every column header is the same shape and the
                # natural-key sort is decided by the index/refs/role prefix, never by prompt length.
                prompt_snippet = prompts[index][:prompt_chars].rstrip()

                if rows == "step":
                    # Convergence-ladder view: base is step 0, the adapter is its checkpoint step,
                    # so successive checkpoints stack as rows and everything else moves to columns.
                    row_value = 0 if column == "base" else step
                    label = f"{refs} · f{frames} · {index + 1:02d} · {prompt_snippet}"
                elif layout == "length":
                    row_value = frames
                    label = f"{index + 1:02d} {column} · {prompt_snippet}"
                elif layout == "reference":
                    # Prompt-major: the three reference conditions sit next to each other for the
                    # same prompt, which is the comparison this layout exists to make.
                    row_value = frames
                    label = f"{index + 1:02d} · {refs} {column} · {prompt_snippet}"
                else:  # matrix — REFERENCE-MAJOR, so unrendered combos form clean rectangles
                    row_value = frames
                    label = f"{refs} · {index + 1:02d} {column} · {prompt_snippet}"

                label = _safe_label(label)
                # ⛔ #12: `base` and `lora` no longer share one local root — the base column is
                # fetched once per geometry and joined onto every checkpoint's render (`fetch_renders`
                # / `--skip-fetch`'s local-mirror mirror of it), so its local dir lives OUTSIDE this
                # render's own `local_key`. `local_by_column` carries the per-column root explicitly
                # rather than this staging loop re-deriving it.
                src = render["local_by_column"][column] / filename
                if not src.exists():
                    continue
                _link_or_copy(src, grid_dir / label / f"step_{row_value}{src.suffix}")
                staged += 1
                columns.add(label)
                row_values.add(row_value)

    legend = "\n".join(f"{i + 1:02d}  {p}" for i, p in enumerate(prompts))
    (grid_dir / "legend.txt").write_text(
        f"Prompt index legend for {grid_dir.name}\n"
        f"(rows={rows}, layout={layout})\n\n{legend}\n",
        encoding="utf-8",
    )
    return {
        "staged": staged, "columns": len(columns), "rows": sorted(row_values),
        "unmatched": unmatched,
    }


# ──────────────────────────────────────────────────────────────────────────────────────────────
# gridwatch + serve + tunnel
# ──────────────────────────────────────────────────────────────────────────────────────────────


def gridwatch_binary(explicit: str | None) -> Path | None:
    """The `grid` entry point from the ISOLATED venv scripts/setup_gridwatch.sh creates.

    Never the trainer environment: gridwatch's fastapi/uvicorn/jinja2 pins and the trainer's
    torch/peft/bitsandbytes pins are deliberately disjoint dependency worlds.
    """
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.exists() else None
    base = REPO_ROOT / "_tools" / "finetune-gridwatch" / ".venv"
    for relative in ("Scripts/grid.exe", "bin/grid", "Scripts/grid"):
        candidate = base / relative
        if candidate.exists():
            return candidate
    found = shutil.which("grid")
    return Path(found) if found else None


def build_grid(binary: Path, grid_dir: Path, cell_size: int) -> tuple[bool, str]:
    # ⛔ `-o` IS REQUIRED. `grid build <folder>` defaults `--output` to `.`, i.e. the CWD — so
    # without this the page lands in the REPO ROOT as `grid-output/index.html`, one shared file that
    # every layout then overwrites in turn. It still prints "Wrote grid-output/index.html" and exits
    # 0, so the only thing that catches it is checking for the file where we asked for it.
    rc, out, err = _run(
        [str(binary), "build", str(grid_dir), "-o", str(grid_dir), "--no-open",
         "--cell-size", str(cell_size), "--template", "{prompt}/step_{step}.mp4"],
        timeout=900,
    )
    index = grid_dir / "grid-output" / "index.html"
    if not index.exists():
        return False, f"rc={rc} {(err or out).strip()[:200]}"
    return True, str(index)


def serve_grid(binary: Path, grid_dir: Path, port: int, cell_size: int) -> tuple[int, str]:
    """Start `grid watch` on `port` unless something is ALREADY serving it (idempotency).

    gridwatch binds 127.0.0.1 ONLY, by design — never patch that. Reaching it from another device
    is the tunnel's job, below.
    """
    if _port_open("127.0.0.1", port):
        return port, "already serving (re-used; grid watch re-scans and pushes over SSE)"
    _spawn(
        [str(binary), "watch", str(grid_dir), "-o", str(grid_dir), "--no-open", "--port", str(port),
         "--cell-size", str(cell_size), "--template", "{prompt}/step_{step}.mp4"],
        grid_dir / "watch.log",
    )
    for _ in range(60):
        if _port_open("127.0.0.1", port):
            return port, "started"
        time.sleep(0.5)
    return port, f"did NOT come up within 30s — see {grid_dir / 'watch.log'}"


def tunnel(ports: list[int]) -> list[str]:
    """Expose the localhost grids to the operator's own devices. Tailscale, per the house rule.

    Preference order, and why: `tailscale serve` gives an HTTPS MagicDNS URL but requires HTTPS
    certs enabled on the tailnet (a one-time admin click). Until that lands, `_tailnet_relay.py`
    bridges the localhost-only servers onto the Tailscale IP ONLY — 100.x, WireGuard-encrypted,
    tailnet-only, NEVER 0.0.0.0. Neither path patches gridwatch's localhost binding.
    """
    rc, out, _err = _run(["tailscale", "status", "--json"], timeout=30)
    if rc != 0:
        return [f"http://127.0.0.1:{p}/  (tailscale not available — localhost only)" for p in ports]
    try:
        status = json.loads(out)
    except json.JSONDecodeError:
        return [f"http://127.0.0.1:{p}/  (tailscale status unparseable — localhost only)"
                for p in ports]

    ips = [ip for ip in (status.get("TailscaleIPs") or []) if ip.startswith("100.")]
    dns = (status.get("Self", {}) or {}).get("DNSName", "").rstrip(".")
    if status.get("CertDomains") and dns and len(ports) == 1:
        rc, _out, err = _run(["tailscale", "serve", "--bg", str(ports[0])], timeout=60)
        if rc == 0:
            return [f"https://{dns}/  ->  127.0.0.1:{ports[0]}"]
        print(f"[grid] `tailscale serve` unavailable ({err.strip()[:120]}) — using the relay.")

    if not ips:
        return [f"http://127.0.0.1:{p}/  (no 100.x tailnet IP — localhost only)" for p in ports]

    ts_ip = ips[0]
    needed = [p for p in ports if not _port_open(ts_ip, p)]
    if needed:
        relay = REPO_ROOT / "scripts" / "_tailnet_relay.py"
        _spawn([sys.executable, str(relay), ts_ip, *[str(p) for p in needed]],
               REPO_ROOT / "_grid_relay.log")
        time.sleep(1.5)
    return [f"http://{ts_ip}:{p}/" for p in ports]


# ──────────────────────────────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────────────────────────────

_LAYOUTS = ("matrix", "length", "reference")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch H3 eval renders, build the finetune-gridwatch grid, serve it live, "
                    "print a tunnel URL. Idempotent. Zero metered spend.",
    )
    parser.add_argument("--config", action="append", default=[],
                        help="A --mode sample config. Repeatable. Default: the glob below.")
    parser.add_argument("--config-glob", default="configs/<your-h3-run>_sample*.yaml",
                        help="Glob used when no --config is given.")
    parser.add_argument("--volume", default="signe-trainer-checkpoints",
                        help="Modal Volume holding <output_dir>/samples_h3.")
    parser.add_argument("--samples-subdir", default="samples_h3",
                        help="Render root under output_dir (h3_sample writes samples_h3, NOT samples).")
    parser.add_argument("--local", default="_samples_h3",
                        help="Local mirror of the render tree (gitignored).")
    parser.add_argument("--grid-dir", default="_grid_h3",
                        help="Grid staging root (gitignored). One subdir per layout.")
    parser.add_argument("--layout", default="matrix", choices=(*_LAYOUTS, "all"),
                        help="matrix = both axes (default) | length | reference | all.")
    parser.add_argument("--rows", default="frames", choices=("frames", "step"),
                        help="Row axis: frame count (default) or checkpoint step (base = 0).")
    parser.add_argument("--checkpoint", default="latest",
                        help="'latest' (default), 'all', or a substring of the checkpoint dir name.")
    parser.add_argument("--pin-reference", default="A+029",
                        help="Reference condition kept by --layout length.")
    parser.add_argument("--pin-frames", type=int, default=22,
                        help="Frame count kept by --layout reference.")
    parser.add_argument("--port", type=int, default=8020, help="First localhost port for grid watch.")
    parser.add_argument("--cell-size", type=int, default=260, help="gridwatch cell width in px.")
    parser.add_argument("--prompt-chars", type=int, default=60,
                        help="Prompt characters kept in a column header (fixed-width, for sorting).")
    parser.add_argument("--grid-bin", default="", help="Path to the gridwatch `grid` entry point.")
    parser.add_argument("--refetch", action="store_true",
                        help="Re-download every clip even if a non-empty local copy exists.")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Rebuild from the existing local mirror without touching Modal.")
    parser.add_argument("--no-serve", action="store_true", help="Build only; do not serve or tunnel.")
    args = parser.parse_args(argv)

    # ── configs ───────────────────────────────────────────────────────────────────────────────
    if args.config:
        config_paths = [Path(c) if Path(c).is_absolute() else REPO_ROOT / c for c in args.config]
    else:
        config_paths = sorted(Path(p) for p in globmod.glob(str(REPO_ROOT / args.config_glob)))
    config_paths = [p for p in config_paths if p.exists()]
    if not config_paths:
        print(f"[grid] no sample configs matched {args.config_glob!r}", file=sys.stderr)
        return 2
    matrix = load_eval_matrix(config_paths)
    output_dir = matrix["output_dir"]
    if not output_dir:
        print("[grid] the sample configs carry no output_dir", file=sys.stderr)
        return 2
    samples_root = f"{output_dir}/{args.samples_subdir}"
    print(f"[grid] eval matrix: {len(config_paths)} config(s), {len(matrix['prompts'])} prompt(s), "
          f"{len(matrix['variants'])} variant(s) -> {args.volume}:{samples_root}")
    for variant in matrix["variants"]:
        print(f"[grid]   {variant['config']}: {variant['frames']}f  "
              f"refs {'+'.join(variant['subject_ids']) or '(unset)'}")

    # ── fetch ─────────────────────────────────────────────────────────────────────────────────
    local_root = Path(args.local) if Path(args.local).is_absolute() else REPO_ROOT / args.local
    local_root.mkdir(parents=True, exist_ok=True)
    if args.skip_fetch:
        # #12: the shared base column lives under `local_root/base/<base key>/`, a SIBLING of the
        # checkpoint-keyed dirs, not nested inside each one — read it once, the same way
        # `fetch_renders` does, rather than re-deriving base clips per checkpoint dir.
        base_local_dir = local_root / _BASE_DIRNAME
        base_clips_by_key = {
            d.name: sorted(f.name for f in d.glob("*.mp4"))
            for d in (base_local_dir.iterdir() if base_local_dir.is_dir() else [])
            if d.is_dir() and parse_base_render_key(d.name) is not None
        }
        renders = []
        for key_dir in sorted(p for p in local_root.iterdir() if p.is_dir()):
            if key_dir.name == _BASE_DIRNAME:
                continue
            meta = parse_render_key(key_dir.name)
            if meta is None:
                continue
            this_base_key = base_key_for(meta)
            meta["base_key"] = this_base_key
            meta["clips"] = {
                "base": base_clips_by_key.get(this_base_key, []),
                "lora": sorted(f.name for f in (key_dir / "lora").glob("*.mp4")),
            }
            meta["local_by_column"] = {
                "base": base_local_dir / this_base_key,
                "lora": key_dir / "lora",
            }
            renders.append(meta)
        fetch = {"available": bool(renders), "reason": "local mirror only (--skip-fetch)",
                 "renders": renders, "fetched": 0, "skipped": 0, "failed": []}
    else:
        if shutil.which("modal") is None:
            print("[grid] `modal` is not on PATH — cannot fetch. Use --skip-fetch to rebuild "
                  "from the local mirror.", file=sys.stderr)
            return 2
        fetch = fetch_renders(args.volume, samples_root, local_root, args.refetch)

    renders = fetch["renders"]
    if not renders:
        # NOT an error, and NOT a reason to dispatch anything. The gated commands are printed for a
        # human to run through the entrypoint; this script never launches a metered render.
        print(f"\n[grid] NO SAMPLES YET under {args.volume}:{samples_root}")
        print(f"[grid] reason: {fetch.get('reason') or 'directory empty'}")
        print("[grid] Nothing to build. When the round finishes, the five gated renders are:")
        for path in config_paths:
            print(f"    PYTHONPATH=src PYTHONUTF8=1 modal run -m signet_trainer.modal.entrypoint "
                  f"--config {path.relative_to(REPO_ROOT).as_posix()} --mode sample --approve")
        print("[grid] Then re-run this script — it is idempotent and picks up whatever exists.")
        return 0

    print(f"[grid] renders: {len(renders)} dir(s) | fetched {fetch['fetched']} clip(s), "
          f"skipped {fetch['skipped']} already local")
    for failure in fetch["failed"][:8]:
        print(f"[grid]   FETCH FAILED {failure}")

    # ── checkpoint selection ──────────────────────────────────────────────────────────────────
    if args.checkpoint == "latest":
        best = max((r["step"] or 0) for r in renders)
        selected = [r for r in renders if (r["step"] or 0) == best]
        print(f"[grid] checkpoint: latest = step {best} ({len(selected)} render dir(s)). "
              "Use --checkpoint all to stack every checkpoint.")
    elif args.checkpoint == "all":
        selected = renders
    else:
        selected = [r for r in renders if args.checkpoint in r["checkpoint"]]
        if not selected:
            print(f"[grid] no render dir matches --checkpoint {args.checkpoint!r}. Available: "
                  + ", ".join(sorted({r['checkpoint'] for r in renders})), file=sys.stderr)
            return 2

    for render in selected:
        delta = (render.get("delta") or {}).get("max_abs_delta_velocity")
        print(f"[grid]   {render['key']}  base={len(render['clips']['base'])} "
              f"lora={len(render['clips']['lora'])}"
              + (f"  max|delta velocity|={delta:.3e}" if isinstance(delta, (int, float)) else ""))

    # ── stage + build ─────────────────────────────────────────────────────────────────────────
    binary = gridwatch_binary(args.grid_bin or None)
    if binary is None:
        print("[grid] finetune-gridwatch is not installed. Run `bash scripts/setup_gridwatch.sh` "
              "(pinned SHA, isolated venv). A hand-rolled grid is not an option.", file=sys.stderr)
        return 2
    print(f"[grid] gridwatch: {binary}")

    grid_root = Path(args.grid_dir) if Path(args.grid_dir).is_absolute() else REPO_ROOT / args.grid_dir
    layouts = list(_LAYOUTS) if args.layout == "all" else [args.layout]
    built: list[tuple[str, Path, dict]] = []
    for layout in layouts:
        grid_dir = grid_root / layout
        report = stage_grid(
            renders=selected, matrix=matrix, grid_dir=grid_dir, layout=layout, rows=args.rows,
            pin_reference=args.pin_reference, pin_frames=args.pin_frames,
            prompt_chars=args.prompt_chars,
        )
        if report["staged"] == 0:
            print(f"[grid] layout {layout}: nothing staged "
                  f"(pin-reference={args.pin_reference}, pin-frames={args.pin_frames}) — skipped.")
            continue
        for name in list(dict.fromkeys(report["unmatched"]))[:5]:
            print(f"[grid]   WARNING unmatched clip {name!r} — its slug is in no config's prompt set")
        ok, detail = build_grid(binary, grid_dir, args.cell_size)
        print(f"[grid] layout {layout}: {report['staged']} cell(s), {report['columns']} column(s), "
              f"rows {report['rows']} -> {detail if ok else 'BUILD FAILED: ' + detail}")
        if ok:
            built.append((layout, grid_dir, report))

    if not built:
        print("[grid] no grid was built.", file=sys.stderr)
        return 1

    if args.no_serve:
        for layout, grid_dir, _report in built:
            print(f"[grid] {layout}: {grid_dir / 'grid-output' / 'index.html'}")
        return 0

    # ── serve + tunnel (house rule: never hand back a file path) ──────────────────────────────
    ports: list[int] = []
    for offset, (layout, grid_dir, _report) in enumerate(built):
        port, note = serve_grid(binary, grid_dir, args.port + offset, args.cell_size)
        ports.append(port)
        print(f"[grid] serving {layout} on 127.0.0.1:{port} ({note})")
    urls = tunnel(ports)

    print("\n" + "=" * 78)
    for (layout, _grid_dir, report), url in zip(built, urls):
        print(f"  {layout.upper():<10} {url}")
        print(f"             {report['staged']} clips | {report['columns']} columns | "
              f"rows {report['rows']} | base vs adapter side by side | CLIPS, not stills")
    print("=" * 78)
    print("[grid] Re-run this exact command to pick up new clips; the live view refreshes itself.")
    print("[grid] Scope: D-10-SCOPEGUARD — this phase grades whether the LOOP works, never adapter "
          "quality. The render is SINGLE-STAGE (eval honesty: an upscale stage can show detail the "
          "adapter never learned). H3 is guidance-distilled: no guidance scale, no negative branch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
