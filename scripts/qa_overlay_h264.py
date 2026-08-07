"""Transcode QA overlay videos to BROWSER-PLAYABLE H.264 and lay them out for finetune-gridwatch.

LANDMINE THIS EXISTS TO CLOSE: ``cv2.VideoWriter`` writes mp4v (MPEG-4 Part 2) by default, which
NO browser can decode — a gridwatch page full of mp4v overlays renders as 34 black rectangles and
looks like a pipeline failure. Every video handed to the operator must be H.264 yuv420p with a relocated
moov atom (``-movflags +faststart``), which is exactly what this script produces.

Input:  ``<in-dir>/<stem>_overlay.mp4``      (whatever codec cv2 emitted)
Output: ``<out-dir>/<stem>/step_0.mp4``      (H.264 — gridwatch's ``{prompt}/step_{step}.mp4`` layout)

Then serve with:
    grid watch <out-dir> -o <grid-out> --no-open --port <port>
    python scripts/_tailnet_relay.py <tailnet-ip> <port>

Pure local ffmpeg. NO Modal, NO GPU, NO metered dispatch. Never deletes an input.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def probe_codec(path: Path) -> str:
    """Return the video stream's codec name via ffprobe ('?' when ffprobe is unavailable)."""
    if not shutil.which("ffprobe"):
        return "?"
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        return json.loads(r.stdout)["streams"][0]["codec_name"]
    except Exception:
        return "?"


def transcode(src: Path, dst: Path, max_width: int, crf: int) -> bool:
    """Transcode one video to browser-playable H.264. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    # yuv420p + even dims are REQUIRED for browser decode; scale only when wider than max_width.
    vf = f"scale='min({max_width},iw)':-2"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[qa-h264] FAIL {src.name}: {r.stderr.strip()[:300]}")
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="Directory of overlay videos to transcode.")
    ap.add_argument("--out-dir", required=True, help="Gridwatch source root (<stem>/step_0.mp4 per clip).")
    ap.add_argument("--suffix", default="_overlay", help="Strip this suffix from the stem (default _overlay).")
    ap.add_argument("--max-width", type=int, default=640, help="Downscale wider inputs (default 640).")
    ap.add_argument("--crf", type=int, default=20, help="libx264 CRF (default 20).")
    ap.add_argument("--step", type=int, default=0, help="Gridwatch step index (default 0).")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not shutil.which("ffmpeg"):
        raise SystemExit("[qa-h264] ffmpeg not found on PATH — required for H.264 (browser-playable) output")

    srcs = sorted(p for p in Path(args.in_dir).glob("*.mp4"))
    if not srcs:
        raise SystemExit(f"[qa-h264] no .mp4 inputs in {args.in_dir}")
    ok, bad = 0, 0
    for src in srcs:
        stem = src.stem
        if args.suffix and stem.endswith(args.suffix):
            stem = stem[: -len(args.suffix)]
        dst = Path(args.out_dir) / stem / f"step_{args.step}.mp4"
        src_codec = probe_codec(src)
        if transcode(src, dst, args.max_width, args.crf):
            out_codec = probe_codec(dst)
            if out_codec not in ("h264", "?"):
                print(f"[qa-h264] WARN {stem}: output codec is {out_codec}, expected h264")
                bad += 1
                continue
            ok += 1
            print(f"[qa-h264] {stem}: {src_codec} -> {out_codec} ({dst})", flush=True)
        else:
            bad += 1
    print(f"[qa-h264] DONE ok={ok} failed={bad} -> {args.out_dir}")
    print("QA_H264_DONE", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
