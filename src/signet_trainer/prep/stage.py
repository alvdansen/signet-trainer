"""prep.stage — inpaint dataset staging (D-13): mux clips + masks, trim to 8n+1, emit metadata.jsonl.

Formalizes ``scripts/_stage_campaign_inpaint_dataset.py`` into the ``prep`` package with three D-13
corrections vs the r1 harvest source:

  1. STEM RESOLUTION IS SINGLE-SOURCED. The r1 script carried a byte-duplicate of the
     stem-resolution helpers (its L64-90); this module imports them from :mod:`prep.resolve` instead
     (the r1 divergent-copy drift is the failure mode ``prep.resolve`` closes).
  2. THE POLARITY RENDER GOES THROUGH THE COMMITTED SEAM. The r1 script inlined an ffmpeg
     ``-vf`` polarity filter (its L240-244); this module calls the committed, tested,
     Windows-portable :func:`signet_trainer.data.mask_encode.render_mask_video` seam instead
     (``[canonical]``-oracle-mirrored — absorb, never duplicate; D-13 / RESEARCH §Don't Hand-Roll).
     That seam does NOT trim, so the per-frame PNG input is trimmed to the largest-8n+1 count
     BEFORE the render (via :func:`largest_8n1`, trim only, NEVER pad) so the emitted mask mp4 has
     exactly ``n_target`` frames matching the muxed clip.
  3. THE ÷64 SPATIAL + 8n+1 FRAME PREFLIGHT IS SURFACED AT PREP TIME. Each clip runs through
     :func:`signet_trainer.prep.preflight.check_inpaint_rules` (reading the divisor + frame period
     FROM the loaded MASK-SPEC, D-03) and a violation RAISES here with the exact crop/trim
     guidance — a non-÷64 clip fails loudly at prep, not deep in the canonical encode
     (closes RESEARCH Pitfall 5). Frames are trimmed automatically (house rule: trim, never pad);
     the spatial rule cannot be auto-fixed (cropping changes content) so it hard-raises.

Captions are the honest placeholder ``[PENDING: operator]`` when no caption file supplies one — never
invented (house rule; Klippbok is the caption gold standard).

Import confinement (Anti-Pattern 6): module top imports stdlib only. ``cv2`` (mask/clip probing) and
:func:`render_mask_video` (which drags torch via ``data.mask_encode``) are FUNCTION-LOCAL, so
``import signet_trainer.prep.stage`` never pulls torch/cv2/modal and unit-tests parse free on CI.

NO metered dispatch, NO Modal import — pure local file prep (ffmpeg + ffprobe required to stage).
Never deletes anything; existing staged outputs are overwritten only with ``force=True``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from signet_trainer.prep.preflight import check_inpaint_rules
from signet_trainer.prep.resolve import (
    load_manifest_map,
    resolve_clip,
    sanitize_stem,  # noqa: F401 — re-exported for the CLI + parity with the r1 helper surface
    split_mask_stem,
)
from signet_trainer.prep.spec import load_mask_spec

#: The honest caption placeholder when no caption file supplies one — never invent captions
#: (house rule; Klippbok is the caption gold standard). PENDING: operator.
CAPTION_PLACEHOLDER = "[PENDING: operator]"


def largest_8n1(n: int) -> int:
    """Largest F with ``F % 8 == 1`` and ``F <= n``. GATE-SPEC inpaint frame rule; trim only, NEVER pad.

    Kept verbatim from the r1 harvest source (``_stage_campaign_inpaint_dataset.py`` L118-120).
    """
    return ((n - 1) // 8) * 8 + 1


def assert_stage_dims(width: int, height: int, frames: int, spec: dict, *, stem: str = "") -> int:
    """Preflight the STAGED dims (÷64 spatial + 8n+1 frames) and return the trimmed frame count.

    Computes ``n_target = largest_8n1(frames)`` (frames are auto-trimmed — house rule: trim, never
    pad) then runs :func:`check_inpaint_rules` on ``(width, height, n_target)``. The frame rule is
    satisfied by construction; the ÷64 spatial rule CANNOT be auto-fixed (cropping changes content),
    so a violation RAISES here with the exact crop/trim guidance — surfacing the rules at prep time
    instead of deep in the canonical encode (RESEARCH Pitfall 5).
    """
    n_target = largest_8n1(frames)
    report = check_inpaint_rules(width, height, n_target, spec)
    if not report["passes"]:
        where = f" [{stem}]" if stem else ""
        raise ValueError(
            f"[stage-inpaint]{where} clip {width}x{height} {frames}f violates the inpaint dims "
            f"gate ({report['rule']}): " + "; ".join(report["needed_fix"]) + ". Fix the source clip "
            "before staging (trim is applied automatically; spatial dims must be cropped by hand — "
            "cropping changes content, so it is never done silently)."
        )
    return n_target


def ffprobe_stream(path: Path) -> dict:
    """width / height / fps / nframes via ffprobe (cv2 fallback for nframes if metadata is absent)."""
    r = subprocess.run(
        ["ffprobe", "-v", "0", "-select_streams", "v:0", "-of", "json",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames", str(path)],
        capture_output=True, text=True,
    )
    try:
        st = json.loads(r.stdout)["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        raise RuntimeError(f"[stage-inpaint] ffprobe failed on {path}: {r.stderr.strip()[:200]}") from exc
    num, _, den = st.get("r_frame_rate", "16/1").partition("/")
    fps = float(num) / float(den or 1)
    nframes = int(st["nb_frames"]) if str(st.get("nb_frames", "N/A")).isdigit() else None
    if nframes is None:
        try:
            import cv2  # noqa: PLC0415 — function-local: probing degrades gracefully without it
        except ImportError as exc:
            raise RuntimeError(
                f"[stage-inpaint] no nb_frames metadata for {path} and cv2 unavailable"
            ) from exc
        cap = cv2.VideoCapture(str(path))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return {"w": int(st["width"]), "h": int(st["height"]), "fps": fps, "n": nframes}


def png_dims(path: Path) -> tuple[int, int] | None:
    """(width, height) of a PNG via cv2, or ``None`` when cv2 is unavailable (probing degrades)."""
    try:
        import cv2  # noqa: PLC0415 — function-local
    except ImportError:
        return None  # ffmpeg staging still hard-fails on a real dims mismatch
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (im.shape[1], im.shape[0]) if im is not None else None


def build_rows(
    *,
    mask_manifest: Path,
    stem_manifest: Path,
    masks_video_dir: Path,
    clips_dir: Path,
    seeds_dir: Path,
    captions: Path | None,
) -> list[dict]:
    """One planned row per mask-manifest entry (one row per MASK, not per clip — r1 behavior).

    Each row carries everything the self-check + stage need: the resolved source clip, its probed
    dims (when it resolves), the largest-8n+1 target frame count, the propagated-mask PNG dir, the
    seed PNG, and the caption (or the ``[PENDING: operator]`` placeholder).
    """
    manifest = json.loads(Path(mask_manifest).read_text(encoding="utf-8"))
    stem_map = load_manifest_map(Path(stem_manifest))
    caption_map: dict[str, str] = {}
    if captions is not None and Path(captions).exists():
        caption_map = json.loads(Path(captions).read_text(encoding="utf-8"))

    rows: list[dict] = []
    for mask_stem in sorted(manifest):
        entry = manifest[mask_stem]
        clip_part, mask_type = split_mask_stem(mask_stem)
        clip = resolve_clip(clip_part, stem_map, Path(clips_dir))
        row = {
            "mask_stem": mask_stem,
            "type": entry.get("type") if isinstance(entry, dict) else mask_type,
            "clip": clip,
            "seed_png": Path(seeds_dir) / f"{mask_stem}.png",
            "masks_dir": Path(masks_video_dir) / mask_stem,
            "caption": caption_map.get(mask_stem, CAPTION_PLACEHOLDER),
        }
        if clip is not None:
            probe = ffprobe_stream(clip)
            row.update(probe)
            row["n_target"] = largest_8n1(probe["n"])
        rows.append(row)
    return rows


def self_check(rows: list[dict], spec: dict, *, staging: bool) -> tuple[int, int]:
    """Verify stems resolve, masks pair, dims agree, and each clip passes the ÷64/8n+1 preflight.

    Returns ``(hard_failures, pending)``. Reports every row (does not raise) so ``--dry-run`` can
    surface the full picture; the ÷64/8n+1 violation is collected here as a hard failure and RAISED
    only in the staging path (:func:`stage`).
    """
    fails, pending = 0, 0
    print(f"[stage-inpaint] self-check: {len(rows)} mask row(s)")
    for row in rows:
        stem = row["mask_stem"]
        problems, notes = [], []
        if row["clip"] is None:
            problems.append("clip stem does NOT resolve")
        else:
            notes.append(f"clip={row['clip'].name} {row['w']}x{row['h']} {row['n']}f @{row['fps']:g}fps")
            n_t = row["n_target"]
            notes.append(f"TRIM {row['n']}->{n_t} (8n+1)" if n_t != row["n"] else f"frames ok ({row['n']}%8==1)")
            report = check_inpaint_rules(row["w"], row["h"], n_t, spec)
            if not report["passes"]:
                problems.append(f"dims gate: {'; '.join(report['needed_fix'])}")
        if not row["seed_png"].exists():
            problems.append(f"seed PNG missing: {row['seed_png'].name}")
        elif row["clip"] is not None:
            dims = png_dims(row["seed_png"])
            if dims is not None and dims != (row["w"], row["h"]):
                problems.append(f"seed dims {dims} != clip {(row['w'], row['h'])}")
        prop_pngs = sorted(row["masks_dir"].glob("*.png")) if row["masks_dir"].exists() else []
        if not prop_pngs:
            pending += 1
            notes.append("propagated masks PENDING (run propagate first)")
            if staging:
                problems.append("cannot stage without propagated masks")
        elif row["clip"] is not None:
            if len(prop_pngs) < row["n_target"]:
                problems.append(f"only {len(prop_pngs)} propagated PNGs < target {row['n_target']}")
            dims = png_dims(prop_pngs[0])
            if dims is not None and dims != (row["w"], row["h"]):
                problems.append(f"propagated mask dims {dims} != clip {(row['w'], row['h'])}")
            else:
                notes.append(f"propagated {len(prop_pngs)} PNGs, dims ok")
        notes.append("caption ok" if row["caption"] != CAPTION_PLACEHOLDER else f"caption {CAPTION_PLACEHOLDER}")
        status = "FAIL" if problems else "ok"
        fails += bool(problems)
        print(f"  {status:4s} {stem}: " + " | ".join(notes + problems))
    return fails, pending


def _run_ffmpeg(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"[stage-inpaint] ffmpeg failed ({what}):\n{r.stderr[-800:]}")


def _stage_one(row: dict, spec: dict, muxed_dir: Path, masks_dir: Path, *, force: bool) -> None:
    """Stage a single row: mux the (trimmed) clip + render the polarity mask mp4 to ``n_target`` frames."""
    import shutil  # noqa: PLC0415 — function-local (stdlib; grouped with the render seam)
    import tempfile  # noqa: PLC0415
    from signet_trainer.data.mask_encode import render_mask_video  # noqa: PLC0415 — drags torch; confine it

    stem = row["mask_stem"]
    # Preflight the STAGED dims and get the trimmed frame count (raises on a ÷64 violation).
    n_t = assert_stage_dims(row["w"], row["h"], row["n"], spec, stem=stem)
    dst_v = muxed_dir / f"{stem}.mp4"
    dst_m = masks_dir / f"{stem}.mp4"

    if dst_v.exists() and dst_m.exists() and not force:
        print(f"[stage-inpaint]   SKIP {stem}: already staged (force=True to redo)")
        return

    if n_t == row["n"]:
        shutil.copy2(row["clip"], dst_v)  # bit-exact, frames already 8n+1
        print(f"[stage-inpaint]   COPY {stem}: {row['n']}f (already 8n+1)")
    else:
        # Trim = decode-bounded re-encode (near-lossless crf 10; audio dropped — inpaint is
        # video-only). NEVER pads. [house] frames%8==1 inpaint rule.
        _run_ffmpeg(
            ["ffmpeg", "-y", "-i", str(row["clip"]), "-frames:v", str(n_t),
             "-c:v", "libx264", "-crf", "10", "-preset", "medium",
             "-pix_fmt", "yuv420p", "-an", str(dst_v)],
            f"trim {stem}",
        )
        print(f"[stage-inpaint]   TRIM {stem}: {row['n']}f -> {n_t}f (largest 8n+1; crf10, audio dropped)")

    # Polarity mask mp4 via the committed render_mask_video seam. It does NOT trim, so the per-frame
    # PNG input is trimmed to n_t frames FIRST (staged into a temp dir) — the emitted mp4 then has
    # exactly n_t frames matching the muxed clip. render_mask_video sorts *.png in the dir.
    prop_pngs = sorted(Path(row["masks_dir"]).glob("*.png"))[:n_t]
    if len(prop_pngs) < n_t:
        raise RuntimeError(
            f"[stage-inpaint] {stem}: only {len(prop_pngs)} propagated PNGs < target {n_t} — "
            "re-run propagation so every target frame has a mask."
        )
    with tempfile.TemporaryDirectory() as trimmed:
        trimmed_dir = Path(trimmed)
        for i, png in enumerate(prop_pngs):
            shutil.copy(png, trimmed_dir / f"{i:06d}.png")
        render_mask_video(trimmed_dir, dst_m, fps=int(round(row["fps"])))
    print(f"[stage-inpaint]   MASK {stem}: rendered {n_t}f polarity mask -> {dst_m.name}")


def stage_inpaint_dataset(
    *,
    mask_manifest: str | Path,
    stem_manifest: str | Path,
    masks_video_dir: str | Path,
    clips_dir: str | Path,
    dest: str | Path,
    spec: str | Path | dict,
    seeds_dir: str | Path,
    captions: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Stage the inpaint dataset (mux + polarity mask + metadata.jsonl), or self-check with ``dry_run``.

    Muxes each source clip (trimmed to the largest 8n+1, never padded) + its polarity mask mp4
    (rendered via the committed :func:`render_mask_video` seam), surfaces the ÷64/8n+1 preflight per
    clip, and emits ``dest/metadata.jsonl`` rows
    ``{media_path: "muxed/<stem>.mp4", video_mask: "masks/<stem>.mp4", caption: <caption or PENDING>}``
    — one row per MASK. With ``dry_run=True`` nothing is written: it resolves every pairing and prints
    the trim/preflight/caption plan.

    Returns a report dict: ``{rows, fails, pending, manifest}`` (``manifest`` is ``None`` on dry-run).
    """
    spec_dict = spec if isinstance(spec, dict) else load_mask_spec(spec)
    if not Path(mask_manifest).exists():
        raise FileNotFoundError(f"[stage-inpaint] mask manifest not found: {mask_manifest}")

    rows = build_rows(
        mask_manifest=Path(mask_manifest),
        stem_manifest=Path(stem_manifest),
        masks_video_dir=Path(masks_video_dir),
        clips_dir=Path(clips_dir),
        seeds_dir=Path(seeds_dir),
        captions=Path(captions) if captions is not None else None,
    )
    fails, pending = self_check(rows, spec_dict, staging=not dry_run)

    if dry_run:
        print(
            f"\n[stage-inpaint] dry-run: {len(rows)} row(s); {fails} hard failure(s); "
            f"{pending} awaiting propagation. Nothing written."
        )
        return {"rows": rows, "fails": fails, "pending": pending, "manifest": None}

    if fails:
        raise RuntimeError(f"[stage-inpaint] {fails} row(s) failed the self-check — fix before staging.")

    dest = Path(dest)
    muxed_dir = dest / "muxed"
    masks_dir = dest / "masks"
    muxed_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for row in rows:
        _stage_one(row, spec_dict, muxed_dir, masks_dir, force=force)
        stem = row["mask_stem"]
        manifest_rows.append(
            {"media_path": f"muxed/{stem}.mp4", "video_mask": f"masks/{stem}.mp4", "caption": row["caption"]}
        )

    manifest = dest / "metadata.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in manifest_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pending = sum(1 for r in rows if r["caption"] == CAPTION_PLACEHOLDER)
    print(f"\n[stage-inpaint] STAGED {len(rows)} row(s) -> {dest}; wrote {manifest}")
    if n_pending:
        print(f"[stage-inpaint] NOTE: {n_pending} caption(s) are '{CAPTION_PLACEHOLDER}' — resolve with the operator before encode.")
    return {"rows": rows, "fails": 0, "pending": pending, "manifest": manifest}
