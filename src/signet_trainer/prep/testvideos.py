"""prep.testvideos — held-out inpaint TEST-video prep (D-13, GATE-SPEC-inpaint-a2v).

Formalizes ``scripts/_prep_inpaint_testvideos.py`` into the ``prep`` package. Operator directive: "pull 3 videos
from the earlier training run of different prompts" — pulls held-out grid renders
(``<grid-dir>/<prompt>/step_<N>.mp4``), then per video:

  1. Copies each to ``<videos-out>/<prompt>.mp4`` (clean stems — the propagate step matches
     ``<stem>__<type>.png`` seeds to ``<stem>.mp4`` verbatim).
  2. ffprobes each (dims / frames / fps) and runs the inpaint dims preflight — ÷64 spatial + 8n+1
     frames — through :func:`signet_trainer.prep.preflight.check_inpaint_rules` (D-13: single-sourced
     with the spec, NOT the r1 inline copy; the divisor + frame period come FROM ``MASK-SPEC.yaml``).
  3. Detects the face on frame 0 (mediapipe FaceDetection when available, else the cv2 haar cascade
     crop-confirmed by mediapipe — both local/free) and writes a FRAME-0 SEED as a filled ellipse
     over the detected box inflated ``FACE_INFLATE`` x -> ``<seeds-out>/<prompt>__face.png``
     (binary, region=WHITE(255) — the same shape :mod:`prep.propagate` consumes), plus a
     ``<prompt>__face_overlay.png`` QA image for the operator's eyeball pass.
  4. Emits ``<videos-out>/testvideos_manifest.json`` with per-video probe results, the spec-driven
     rule check, detector used, bbox, and source provenance.

The seeds are ready for the SAME propagate step (``scripts/prep_inpaint_propagate.py`` /
``prep.propagate``) — point ``--masks-dir`` at ``<seeds-out>`` and ``--clips-dir`` at ``<videos-out>``.

Import-confined (Anti-Pattern 6): module top imports stdlib + the import-confined
:mod:`prep.preflight` / :mod:`prep.spec` only. ``cv2`` / ``mediapipe`` / ``numpy`` are FUNCTION-LOCAL,
so ``import signet_trainer.prep.testvideos`` never pulls them and unit-tests parse free on CI. NO
metered dispatch, NO Modal import. Never deletes anything; ``force`` re-writes existing outputs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from signet_trainer.prep.preflight import check_inpaint_rules
from signet_trainer.prep.spec import load_mask_spec

REPO_ROOT = Path(__file__).resolve().parents[3]
PREP_DIR = REPO_ROOT / "_inpaint_prep"

FACE_INFLATE = 1.4  # ellipse over the detected face box inflated 1.4x (task spec)

MP_TASKS_MODEL = PREP_DIR / "blaze_face_short_range.tflite"
MP_TASKS_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


def ffprobe_stream(path: Path) -> dict:
    """ffprobe a clip's first video stream -> {w, h, fps, n}. Requires ffprobe on PATH."""
    r = subprocess.run(
        ["ffprobe", "-v", "0", "-select_streams", "v:0", "-of", "json",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames", str(path)],
        capture_output=True, text=True,
    )
    st = json.loads(r.stdout)["streams"][0]
    num, _, den = st.get("r_frame_rate", "24/1").partition("/")
    return {
        "w": int(st["width"]), "h": int(st["height"]),
        "fps": float(num) / float(den or 1),
        "n": int(st["nb_frames"]) if st.get("nb_frames", "N/A").isdigit() else -1,
    }


# --------------------------------------------------------------------------------------------------
# Frame-0 face detection: mediapipe (preferred) -> cv2 haar cascade (largest face) fallback.
# --------------------------------------------------------------------------------------------------

def detect_face_mediapipe(frame_bgr, _quiet: bool = False):
    """mediapipe face detect: legacy solutions API (<=0.10.1x) OR Tasks API (0.10.2x+, needs a
    small .tflite model — auto-downloaded once to _inpaint_prep). Returns (bbox, tag) or None."""
    try:
        import mediapipe as mp  # noqa: PLC0415
    except ImportError:
        return None
    import cv2  # noqa: PLC0415
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if hasattr(mp, "solutions"):  # legacy solutions API
        # model_selection=1 = full-range model (subjects further from camera — fits film stills).
        with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as fd:
            res = fd.process(rgb)
        if not res.detections:
            return None
        best = max(res.detections, key=lambda d: d.location_data.relative_bounding_box.width
                   * d.location_data.relative_bounding_box.height)
        rb = best.location_data.relative_bounding_box
        return (int(rb.xmin * w), int(rb.ymin * h), int(rb.width * w), int(rb.height * h)), \
            f"mediapipe solutions (score {best.score[0]:.2f})"

    try:  # Tasks API (mediapipe >= 0.10.2x removed mp.solutions)
        from mediapipe.tasks import python as mp_python  # noqa: PLC0415
        from mediapipe.tasks.python import vision  # noqa: PLC0415
        if not MP_TASKS_MODEL.exists():
            import urllib.request  # noqa: PLC0415
            print(f"[prep-testvideos]   downloading mediapipe face model -> {MP_TASKS_MODEL.name} (one-time, ~230KB)")
            urllib.request.urlretrieve(MP_TASKS_MODEL_URL, MP_TASKS_MODEL)
        detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MP_TASKS_MODEL)),
            min_detection_confidence=0.3,
        ))
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.detections:
            return None
        best = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb = best.bounding_box
        score = best.categories[0].score if best.categories else float("nan")
        return (int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)), \
            f"mediapipe tasks (score {score:.2f})"
    except Exception as exc:  # any Tasks/model failure -> haar fallback
        if not _quiet:
            print(f"[prep-testvideos]   mediapipe tasks unavailable ({type(exc).__name__}: {exc}) — falling back to haar")
        return None


def _mediapipe_confirm(frame_bgr, box: tuple[int, int, int, int]):
    """Verify a candidate box by running mediapipe on an UPSCALED crop around it (small/distant
    faces defeat both full-frame blazeface and haar precision — haar alone false-positives on
    texture, e.g. trousers). Returns the refined full-frame bbox + score, or None."""
    import cv2  # noqa: PLC0415
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2.0, y + bh / 2.0
    half = max(bw, bh) * 1.25  # crop = candidate inflated 2.5x
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    scale = max(1.0, 512.0 / max(crop.shape[:2]))
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    det = detect_face_mediapipe(up, _quiet=True)
    if det is None:
        return None
    (ux, uy, ubw, ubh), tag = det
    score = float(tag.rsplit(" ", 1)[-1].rstrip(")")) if "score" in tag else 0.0
    return (int(x0 + ux / scale), int(y0 + uy / scale), int(ubw / scale), int(ubh / scale)), score


def detect_face_haar(frame_bgr):
    """Haar candidates (frontal default + alt2 + profile), each VERIFIED via mediapipe on an
    upscaled crop. Returns (bbox, detector_tag, verified). Unverified largest-box is last resort."""
    import cv2  # noqa: PLC0415
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    candidates = []
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml",
                 "haarcascade_profileface.xml"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        for f in cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(24, 24)):
            candidates.append(tuple(int(v) for v in f))
    if not candidates:
        return None
    best = None
    for cand in sorted(candidates, key=lambda f: f[2] * f[3], reverse=True):
        confirmed = _mediapipe_confirm(frame_bgr, cand)
        if confirmed and (best is None or confirmed[1] > best[1]):
            best = confirmed
    if best is not None:
        bbox, score = best
        return bbox, f"haar + mediapipe crop-confirm (score {score:.2f})", True
    x, y, bw, bh = max(candidates, key=lambda f: f[2] * f[3])
    return (x, y, bw, bh), "haar UNVERIFIED largest box — CHECK THE OVERLAY", False


def make_seed(frame_bgr, bbox: tuple[int, int, int, int]):
    """Filled ellipse over the bbox inflated FACE_INFLATE x -> binary seed (white=region) + overlay."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    ax, ay = int(bw * FACE_INFLATE / 2.0), int(bh * FACE_INFLATE / 2.0)
    seed = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(seed, (int(cx), int(cy)), (ax, ay), 0, 0, 360, 255, -1)
    overlay = frame_bgr.copy()
    green = np.zeros_like(frame_bgr)
    green[:, :] = (0, 200, 0)
    m = seed.astype(bool)
    overlay[m] = (0.55 * frame_bgr[m] + 0.45 * green[m]).astype(np.uint8)
    cv2.ellipse(overlay, (int(cx), int(cy)), (ax, ay), 0, 0, 360, (0, 255, 255), 2)
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (255, 128, 0), 1)  # raw detector box for QA
    return seed, overlay


def first_frame(path: Path):
    """Read frame 0 of a clip (BGR). Raises on an unreadable clip."""
    import cv2  # noqa: PLC0415
    cap = cv2.VideoCapture(str(path))
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        raise SystemExit(f"[prep-testvideos] cannot read frame 0 of {path}")
    return fr


def prep_testvideos(args) -> int:
    """Copy + probe + seed the held-out test videos. Returns a process exit code.

    ``args`` carries: ``grid_dir``, ``step``, ``prompt`` (list | None), ``prompts`` (default set),
    ``videos_out``, ``seeds_out``, ``spec``, ``force``, ``dry_run``. The ÷64/8n+1 preflight is
    single-sourced through :func:`prep.preflight.check_inpaint_rules` reading the loaded spec (D-13).
    """
    prompts = args.prompt or args.prompts
    spec = load_mask_spec(args.spec)  # fail-loud; the dims divisor + frame rule come from here (D-03)
    grid_dir = Path(args.grid_dir)
    videos_out = Path(args.videos_out)
    seeds_out = Path(args.seeds_out)

    manifest: dict[str, dict] = {}
    fails = 0
    for prompt in prompts:
        src = grid_dir / prompt / f"step_{args.step}.mp4"
        if not src.exists():
            print(f"[prep-testvideos] FAIL {prompt}: source render missing: {src}")
            fails += 1
            continue
        probe = ffprobe_stream(src)
        rules = check_inpaint_rules(probe["w"], probe["h"], probe["n"], spec)
        rule_note = rules["rule"] if rules["passes"] else f"NEEDS FIX: {', '.join(rules['needed_fix'])}"
        state = "passes" if rules["passes"] else rule_note
        print(f"[prep-testvideos] {prompt}: {probe['w']}x{probe['h']} {probe['n']}f "
              f"@{probe['fps']:g}fps | {state}")

        entry = {"source": str(src), "step": args.step, **probe, "rules": rules}
        if not args.dry_run:
            videos_out.mkdir(parents=True, exist_ok=True)
            seeds_out.mkdir(parents=True, exist_ok=True)
            dst = videos_out / f"{prompt}.mp4"
            if not dst.exists() or args.force:
                shutil.copy2(src, dst)
            entry["video"] = str(dst)

            frame0 = first_frame(dst)
            det = detect_face_mediapipe(frame0)
            if det is not None:
                bbox, detector = det
                verified = True
            else:
                haar = detect_face_haar(frame0)
                bbox, detector, verified = haar if haar is not None else (None, None, False)
            if bbox is None:
                print(f"[prep-testvideos]   FAIL {prompt}: NO face detected on frame 0 — paint a manual seed")
                entry["seed"] = None
                fails += 1
            else:
                seed, overlay = make_seed(frame0, bbox)
                import cv2  # noqa: PLC0415
                seed_path = seeds_out / f"{prompt}__face.png"
                overlay_path = seeds_out / f"{prompt}__face_overlay.png"
                if not seed_path.exists() or args.force:
                    cv2.imwrite(str(seed_path), seed)
                    cv2.imwrite(str(overlay_path), overlay)
                cov = float((seed > 0).mean()) * 100.0
                print(f"[prep-testvideos]   seed {seed_path.name}: {detector}, bbox={bbox}, "
                      f"inflate {FACE_INFLATE}x, coverage {cov:.1f}%")
                if not verified:
                    print(f"[prep-testvideos]   WARN {prompt}: seed is UNVERIFIED — eyeball "
                          f"{overlay_path.name} before propagating")
                    fails += 1
                entry["seed"] = {"path": str(seed_path), "detector": detector, "bbox": list(bbox),
                                 "inflate": FACE_INFLATE, "coverage_pct": round(cov, 2),
                                 "verified": verified}
        manifest[prompt] = entry

    if not args.dry_run and manifest:
        videos_out.mkdir(parents=True, exist_ok=True)
        mpath = videos_out / "testvideos_manifest.json"
        mpath.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"[prep-testvideos] wrote {mpath}")
        print("[prep-testvideos] seeds are ready for the SAME propagate step "
              "(scripts/prep_inpaint_propagate.py --masks-dir <seeds-out> --clips-dir <videos-out>).")
    print(f"[prep-testvideos] done: {len(manifest)} video(s), {fails} problem(s).")
    return 1 if fails else 0
