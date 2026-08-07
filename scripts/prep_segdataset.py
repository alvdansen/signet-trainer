"""Assemble a SEGMENTATION dataset from propagated masks — two export layouts.

Consumes what ``prep_textseed.py`` -> ``prep_inpaint_propagate.py`` produce (per-frame binary mask
PNGs under ``<masks-root>/<mask_stem>/NNNNN.png``, region WHITE(255)). Two independent, additive
output modes selected by which ``--out`` flag you pass (you may pass both):

1. ``--out <dir>`` — IMAGE/MASK PAIR layout (trainer-agnostic), one pair per kept frame:

    <out>/images/<clip_stem>_<frame:05d>.png
    <out>/masks/<clip_stem>_<frame:05d>.png     # strictly binary {0,255}, same dims as the image
    <out>/manifest.jsonl                        # one JSON object per pair + per-clip seed metadata

   Every pair is VERIFIED before it lands: the mask must be strictly two-valued ({0,255}) and
   dimension-identical to its image, otherwise the pair is rejected and counted (a silently
   mis-sized or grey-valued mask is the failure a segmentation trainer cannot detect). Empty-mask
   frames are KEPT and flagged (``mask_px: 0``) by default (``--skip-empty`` drops them).

2. ``--paired-out <dir>`` — PAIRED control/train ``.mov`` layout (for an EXTERNAL control trainer),
   one numbered clip per KEPT source clip:

    <paired-out>/control_data/NNN.mov   # ProRes white-silhouette-on-black mask VIDEO (from the PNGs)
    <paired-out>/train_data/NNN.mov     # byte-copy of the source clip (NO re-encode, no quality loss)
    <paired-out>/train_data/NNN.txt     # the caption for that clip (if resolvable)
    <paired-out>/MAPPING.local.txt      # "NNN <- <clip_stem>" crosswalk (local-only, .local = git-ignored)

   Clips are numbered 001, 002, ... in sorted order over the kept clips. The control mask video's
   frame count MUST equal the source clip's frame count — a mismatch FAILS that clip loudly (a
   control/train frame mismatch silently corrupts training). The ``<paired-out>`` dir MUST be a
   local / git-ignored path (e.g. under ``_<codename>_prep/``): it contains a byte-copy of the source
   clips and a real-name crosswalk. Never point it inside a tracked repo dir.

NO Modal import, NO metered dispatch, NO model load — pure local decode + write. Never deletes.

Usage:
  # image/mask pairs
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_segdataset.py \
      --masks-root <dir> --clips-dir <dir> --manifest <manifest.txt> --records <records.json> --out <dir>
  # paired control/train .mov
  PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_segdataset.py \
      --masks-root <dir> --clips-dir <dir> --manifest <manifest.txt> \
      --paired-out <local/git-ignored dir> --captions-dir <dir> --keep 01,02,03
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from signet_trainer.prep.resolve import load_manifest_map, resolve_clip, split_mask_stem  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks-root", required=True, help="Root of per-clip mask dirs (<mask_stem>/NNNNN.png).")
    ap.add_argument("--clips-dir", required=True, help="Directory of source clips.")
    ap.add_argument("--manifest", required=True, help="propagate manifest.txt (stem -> clip resolution).")
    ap.add_argument("--records", default=None, help="textseed_records.json (per-clip seed metadata).")
    ap.add_argument("--out", default=None, help="IMAGE/MASK-PAIR output root (images/ masks/ manifest.jsonl).")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame (pair mode; default 1 = all).")
    ap.add_argument("--skip-empty", action="store_true", help="Drop frames whose mask is empty (pair mode).")
    # --- PAIRED control/train .mov mode (additive; independent of the pair mode) ---
    ap.add_argument("--paired-out", default=None,
                    help="PAIRED control/train .mov output root (control_data/ train_data/ MAPPING.local.txt). "
                         "MUST be a local / git-ignored path — it byte-copies source clips + a real-name crosswalk.")
    ap.add_argument("--captions-dir", default=None,
                    help="Source dir of final caption .txt files (paired mode). Omitted -> fall back to a .txt "
                         "beside the source clip, then WARN + skip.")
    ap.add_argument("--keep", default=None,
                    help="Optional comma-separated clip selector (paired mode): clip_stem parts OR their numbers "
                         "(e.g. '01,02,03'). Kept clips are renumbered 001.. in sorted order. Omitted = keep all.")
    ap.add_argument("--fps", type=int, default=None,
                    help="Control .mov framerate override (paired mode). Default: probe the source clip, else 24.")
    return ap


# ==================================================================================================
# PAIRED control/train .mov mode — pure helpers (unit-testable with NO ffmpeg / NO cv2)
# ==================================================================================================

def parse_keep(keep_arg: str | None) -> set[str] | None:
    """'01, 02 ,foo' -> {'01','02','foo'}. None (or empty) -> None = keep all."""
    if not keep_arg:
        return None
    toks = {t.strip() for t in keep_arg.split(",") if t.strip()}
    return toks or None


def clip_matches_keep(clip_part: str, keep: set[str] | None) -> bool:
    """True if ``clip_part`` is selected by the ``--keep`` set (None = keep all).

    Accepts (per the CLI contract) EITHER a whole clip_stem / one of its '_'-delimited parts, OR a
    number that matches a numeric component of the stem (leading ``NN_`` index, trailing ``_NN``, or
    any numeric ``_``-part) — compared by integer value so '01' matches '1' and '001'.
    """
    if keep is None:
        return True
    if clip_part in keep:
        return True
    parts = clip_part.split("_")
    if any(p in keep for p in parts):
        return True
    nums = {str(int(p)) for p in parts if p.isdigit()}
    m = re.search(r"(\d+)$", clip_part)
    if m:
        nums.add(str(int(m.group(1))))
    return any(t.isdigit() and str(int(t)) in nums for t in keep)


def renumber(clip_parts: list[str], keep: set[str] | None) -> list[tuple[str, str]]:
    """Filter ``clip_parts`` (already in sorted order) by ``keep``, then number the survivors
    001, 002, ... contiguously. Returns [(clip_part, 'NNN'), ...]."""
    kept = [c for c in clip_parts if clip_matches_keep(c, keep)]
    return [(c, f"{i:03d}") for i, c in enumerate(kept, start=1)]


def resolve_caption(clip_part: str, clip_path: Path, captions_dir: Path | None) -> Path | None:
    """Resolve a caption .txt for a clip. Prefer ``<captions-dir>/<clip_part>.txt`` (then the source
    clip's own stem), else a .txt beside the source clip; None if nothing exists."""
    if captions_dir is not None:
        for name in dict.fromkeys([clip_part, Path(clip_path).stem]):
            cand = Path(captions_dir) / f"{name}.txt"
            if cand.is_file():
                return cand
    beside = Path(clip_path).with_suffix(".txt")
    return beside if beside.is_file() else None


def build_control_cmd(mask_dir: Path, out_path: Path, fps: int) -> list[str]:
    """The PROVEN ProRes encode of a mask PNG sequence -> white-silhouette-on-black .mov.

    prores_ks + profile 3 + yuv422p10le is verified to preserve the binary {0,255} edges (0 mid-tone
    leakage). Kept as a pure function so a test can assert the flags without invoking ffmpeg.
    """
    return [
        "ffmpeg", "-y", "-v", "error",
        "-framerate", str(int(fps)),
        "-start_number", "0",
        "-i", str(Path(mask_dir) / "%05d.png"),
        "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
        str(out_path),
    ]


def count_mask_frames(mask_dir: Path) -> int:
    """Count the NNNNN.png frames in a mask dir (strict 5-digit names)."""
    return sum(1 for p in Path(mask_dir).glob("*.png") if re.fullmatch(r"\d{5}", p.stem))


def probe_frame_count(clip: Path) -> int:
    """Source clip frame count: ffprobe nb_frames, else an exact cv2 decode count (-1 on failure)."""
    import subprocess  # noqa: PLC0415

    try:
        r = subprocess.run(
            ["ffprobe", "-v", "0", "-select_streams", "v:0", "-of",
             "default=nokey=1:noprint_wrappers=1", "-show_entries", "stream=nb_frames", str(clip)],
            capture_output=True, text=True,
        )
        s = r.stdout.strip()
        if s.isdigit():
            return int(s)
    except FileNotFoundError:
        pass
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return -1
    cap = cv2.VideoCapture(str(clip))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def probe_fps(clip: Path, default: int = 24) -> int:
    """Source clip fps (rounded int) via ffprobe r_frame_rate; ``default`` if unresolved."""
    import subprocess  # noqa: PLC0415

    try:
        r = subprocess.run(
            ["ffprobe", "-v", "0", "-select_streams", "v:0", "-of",
             "default=nokey=1:noprint_wrappers=1", "-show_entries", "stream=r_frame_rate", str(clip)],
            capture_output=True, text=True,
        )
        s = r.stdout.strip()
        if "/" in s:
            num, _, den = s.partition("/")
            d = float(den) or 1.0
            val = float(num) / d
            if val > 0:
                return int(round(val))
        elif s.replace(".", "", 1).isdigit():
            val = float(s)
            if val > 0:
                return int(round(val))
    except (FileNotFoundError, ValueError):
        pass
    return default


def encode_control_mov(mask_dir: Path, out_path: Path, fps: int) -> tuple[bool, str]:
    """Run the ProRes encode. Returns (ok, stderr). Never emits a broken file silently — the caller
    treats ok=False as a hard per-clip failure."""
    import subprocess  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(build_control_cmd(mask_dir, out_path, fps), capture_output=True, text=True)
    except FileNotFoundError:
        return False, "ffmpeg not found on PATH"
    return (r.returncode == 0), r.stderr.strip()


def run_paired(args) -> int:
    """Emit the PAIRED control/train .mov layout. Returns a process exit code (nonzero on any
    failure: unresolved source, missing masks, frame-count mismatch, ffmpeg failure)."""
    import shutil  # noqa: PLC0415

    masks_root = Path(args.masks_root)
    clips_dir = Path(args.clips_dir)
    out = Path(args.paired_out)
    captions_dir = Path(args.captions_dir) if args.captions_dir else None
    keep = parse_keep(args.keep)

    manifest_map = load_manifest_map(Path(args.manifest))

    # Build the ordered (mask_dir, clip_part, clip_path) items over the sorted mask dirs.
    items: list[tuple[Path, str, Path | None]] = []
    for md in sorted(p for p in masks_root.iterdir() if p.is_dir()):
        try:
            clip_part, _mask_type = split_mask_stem(md.name)
        except ValueError:
            print(f"[segdataset-paired] SKIP {md.name}: not a '<stem>__<type>' mask dir", flush=True)
            continue
        clip = resolve_clip(clip_part, manifest_map, clips_dir)
        items.append((md, clip_part, clip))

    # Filter + number the KEPT clips 001.. (numbering is deterministic from the mask-dir set + keep).
    numbering = {cp: num for cp, num in renumber([cp for _md, cp, _clip in items], keep)}
    kept = [(md, cp, clip, numbering[cp]) for md, cp, clip in items if cp in numbering]

    control_dir, train_dir = out / "control_data", out / "train_data"
    control_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    mapping_lines: list[str] = []
    n_done, fails, warns = 0, 0, 0
    for md, clip_part, clip, num in kept:
        if clip is None:
            print(f"[segdataset-paired] FAIL {num} ({clip_part}): source clip unresolved", flush=True)
            fails += 1
            continue
        n_mask = count_mask_frames(md)
        if n_mask == 0:
            print(f"[segdataset-paired] FAIL {num} ({clip_part}): no NNNNN.png mask frames in {md}", flush=True)
            fails += 1
            continue
        n_src = probe_frame_count(clip)
        if n_mask != n_src:
            print(f"[segdataset-paired] FAIL {num} ({clip_part}): FRAME-COUNT MISMATCH "
                  f"mask={n_mask} src={n_src} — control/train would desync; not encoding.", flush=True)
            fails += 1
            continue
        fps = args.fps if args.fps else probe_fps(clip)
        ok, err = encode_control_mov(md, control_dir / f"{num}.mov", fps)
        if not ok:
            print(f"[segdataset-paired] FAIL {num} ({clip_part}): ffmpeg prores_ks encode failed:\n{err}", flush=True)
            fails += 1
            continue
        shutil.copy2(clip, train_dir / f"{num}.mov")
        cap = resolve_caption(clip_part, clip, captions_dir)
        if cap is not None:
            shutil.copy2(cap, train_dir / f"{num}.txt")
            cap_note = cap.name
        else:
            print(f"[segdataset-paired] WARN {num} ({clip_part}): no caption found "
                  f"(captions-dir + beside-source both empty) — skipping .txt", flush=True)
            cap_note = "MISSING"
            warns += 1
        mapping_lines.append(f"{num} <- {clip_part}")
        n_done += 1
        print(f"[segdataset-paired] {num} <- {clip_part}: frames={n_mask} @{fps}fps caption={cap_note}", flush=True)

    (out / "MAPPING.local.txt").write_text("\n".join(mapping_lines) + ("\n" if mapping_lines else ""), encoding="utf-8")
    print(f"[segdataset-paired] DONE clips={n_done} warns={warns} fails={fails} -> {out}")
    print(f"[segdataset-paired] mapping -> {out / 'MAPPING.local.txt'} (local-only; keep <paired-out> git-ignored)")
    print("PAIRED_DONE", flush=True)
    return 1 if fails else 0


# ==================================================================================================
# IMAGE/MASK-PAIR mode (original behavior — unchanged)
# ==================================================================================================

def run_pairs(args) -> int:
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    out = Path(args.out)
    img_dir, msk_dir = out / "images", out / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    manifest_map = load_manifest_map(Path(args.manifest))
    records = {}
    if args.records and Path(args.records).is_file():
        for r in json.loads(Path(args.records).read_text(encoding="utf-8")):
            records[r["clip_stem"]] = r

    mask_dirs = sorted(p for p in Path(args.masks_root).iterdir() if p.is_dir())
    lines, n_pairs, n_bad, n_empty = [], 0, 0, 0
    per_clip = []

    for md in mask_dirs:
        stem = md.name
        clip_part, mask_type = split_mask_stem(stem)
        clip = resolve_clip(clip_part, manifest_map, Path(args.clips_dir))
        if clip is None:
            print(f"[segdataset] SKIP {stem}: clip unresolved")
            continue
        rec = records.get(clip_part, {})

        cap = cv2.VideoCapture(str(clip))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()

        kept, empt, bad, cov_sum = 0, 0, 0, 0.0
        for i, fr in enumerate(frames):
            if i % max(1, args.stride):
                continue
            mp = md / f"{i:05d}.png"
            if not mp.is_file():
                bad += 1
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None or m.shape[:2] != fr.shape[:2]:
                bad += 1
                continue
            vals = set(np.unique(m).tolist())
            if not vals.issubset({0, 255}):
                bad += 1
                continue
            px = int((m > 0).sum())
            if px == 0:
                empt += 1
                if args.skip_empty:
                    continue
            name = f"{clip_part}_{i:05d}.png"
            cv2.imwrite(str(img_dir / name), fr)
            cv2.imwrite(str(msk_dir / name), m)
            h, w = m.shape[:2]
            cov = px / float(w * h)
            cov_sum += cov
            lines.append(json.dumps({
                "image": f"images/{name}",
                "mask": f"masks/{name}",
                "clip_stem": clip_part,
                "mask_type": mask_type,
                "frame": i,
                "width": w,
                "height": h,
                "mask_px": px,
                "coverage": round(cov, 6),
                "seed_frame_idx": rec.get("seed_frame_idx"),
                "direction": rec.get("direction"),
                "seed_score": rec.get("score"),
            }))
            kept += 1
        n_pairs += kept
        n_bad += bad
        n_empty += empt
        per_clip.append({
            "clip_stem": clip_part, "pairs": kept, "empty": empt, "rejected": bad,
            "avg_coverage": round(cov_sum / max(1, kept), 5),
            "seed_frame_idx": rec.get("seed_frame_idx"), "direction": rec.get("direction"),
            "seed_score": rec.get("score"),
        })
        print(f"[segdataset] {clip_part}: pairs={kept} empty={empt} rejected={bad} "
              f"avg_cover={cov_sum / max(1, kept) * 100:.2f}%", flush=True)

    (out / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({"pairs": n_pairs, "clips": len(per_clip), "empty_masks": n_empty,
                    "rejected": n_bad, "per_clip": per_clip}, indent=2), encoding="utf-8")
    print(f"[segdataset] DONE clips={len(per_clip)} pairs={n_pairs} empty={n_empty} rejected={n_bad}")
    print(f"[segdataset] -> {out / 'manifest.jsonl'}")
    print("SEGDATASET_DONE", flush=True)
    return 1 if n_bad else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.out and not args.paired_out:
        build_parser().error("one of --out (image/mask pairs) or --paired-out (control/train .mov) is required")
    rc = 0
    if args.paired_out:
        rc = run_paired(args) or rc
    if args.out:
        rc = run_pairs(args) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
