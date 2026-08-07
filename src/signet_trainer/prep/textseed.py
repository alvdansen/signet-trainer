"""prep.textseed — SAM 3 TEXT-PROMPT auto-seeding (replaces hand-drawn frame-0 seeds).

Where :mod:`prep.chroma` extracts a HAND-PAINTED seed and :mod:`prep.propagate` tracks it across a
clip, this module produces that seed AUTOMATICALLY from a natural-language concept prompt
("a boy with curly red hair") using SAM 3's concept-segmentation head (``Sam3Model`` +
``Sam3Processor``, ``text=`` argument). The output is byte-compatible with
``propagate.load_seed`` — a binary PNG with region = WHITE(255) named
``<clip_stem>__<mask_type>.png`` — so the ENTIRE downstream propagation seam (backends, the D-02
dilate hook, the D-08 fwd/rev backward-seed, the per-frame PNG writer) is reused verbatim. Nothing
here re-implements propagation.

THE DIRECTION DECISION (D-08 reuse). ``propagate.process_job`` can only seed the frame it feeds the
predictor first — physically frame 0 (``fwd``) or frame n-1 (``rev``, via ``frame_order(n, True)``).
Many clips do not contain the subject at frame 0 (a clip may open on an empty room or on a different
character), so this module PROBES several evenly-spaced frames — always including both endpoints —
scores the concept detection on each, and then:

  * prefers ``fwd`` (seed frame 0) whenever frame 0 carries a valid detection, because forward
    propagation from the first frame is the well-trodden path;
  * switches to ``rev`` (seed frame n-1) when the last frame is the only valid endpoint, or when it
    beats frame 0 by more than ``seed.start_margin``;
  * falls back to a FLAGGED weak endpoint detection (``seed.weak_score``) when neither endpoint
    clears ``min_score`` but the mid-clip probes prove the subject is present;
  * reports the clip as NOT SEEDABLE (never silently emits an empty seed) when no endpoint detects
    anything at all.

The mid-clip probes are never seeded from; they are EVIDENCE (recorded as ``probes`` in the per-clip
record) so a "subject absent at both endpoints" clip is visibly distinguishable from "subject absent
from the whole clip".

Every threshold (scores, coverage bands, probe count, the start margin, the model id/dtype) is read
from ``configs/prep_textseed.yaml`` — NEVER hardcoded here (house config-first rule).

Import-confined (Anti-Pattern 6): stdlib + ``yaml`` at module top; ``torch`` / ``transformers`` /
``cv2`` / ``numpy`` are all function-local, so this module PARSES and its pure index/decision logic
unit-tests on a CPU/CI box with no GPU env. NO Modal import, NO metered dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "prep_textseed.yaml"

#: Direction tokens — the SAME vocabulary ``propagate.process_job`` uses (``rev`` set membership).
FWD = "fwd"
REV = "rev"

#: Reason codes recorded per clip (stable strings; the QA report groups on them).
REASON_START = "start-endpoint"
REASON_END = "end-endpoint"
REASON_END_BEATS_START = "end-beats-start"
REASON_WEAK = "weak-endpoint"
REASON_NOT_FOUND = "subject-not-found"

_REQUIRED_KEYS = ("model", "detect", "seed")
_REQUIRED_SEED_KEYS = (
    "n_probes",
    "min_score",
    "weak_score",
    "min_coverage",
    "max_coverage",
    "start_margin",
    "mask_type",
)

INSTALL_HINTS = {
    "torch": "pip install torch --index-url https://download.pytorch.org/whl/cu128  (CUDA build)",
    "cv2": "pip install opencv-python",
    "numpy": "pip install numpy",
    "transformers": "pip install 'transformers>=5.14'  (Sam3Model/Sam3Processor concept segmentation)",
}


def _need(module: str):
    """Import a heavy dep lazily with a clear install hint (module must PARSE without them)."""
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            f"[textseed] missing dependency '{module}': {exc}\n"
            f"    install hint: {INSTALL_HINTS.get(module, f'pip install {module}')}"
        ) from exc


# --------------------------------------------------------------------------------------------------
# Config (fail-LOUD, config-first). Every knob below comes from YAML, never from a literal in code.
# --------------------------------------------------------------------------------------------------

def load_textseed_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load + shape-check ``prep_textseed.yaml``. Fails LOUD on absence or a half-written config."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"textseed config not found: {path} (thresholds are config-sourced, never hardcoded)"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"textseed config {path}: top level must be a mapping, got {type(data).__name__}")
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"textseed config {path}: missing required blocks {missing}; a half-written config must "
            f"never render a plausible-looking default"
        )
    seed = data["seed"]
    if not isinstance(seed, dict):
        raise ValueError(f"textseed config {path}: 'seed' must be a mapping")
    missing_seed = [k for k in _REQUIRED_SEED_KEYS if k not in seed]
    if missing_seed:
        raise ValueError(f"textseed config {path}: seed block missing {missing_seed}")
    for key in ("threshold", "mask_threshold"):
        if key not in data["detect"]:
            raise ValueError(f"textseed config {path}: detect block missing {key!r}")
    return data


# --------------------------------------------------------------------------------------------------
# PURE probe/selection/decision logic (no torch, no cv2 — fully unit-testable).
# --------------------------------------------------------------------------------------------------

def probe_indices(n_frames: int, n_probes: int) -> list[int]:
    """Frame indices to probe: ALWAYS both endpoints, plus evenly-spaced interior probes.

    The endpoints are mandatory because they are the only two frames the propagation seam can seed
    (``fwd`` -> frame 0, ``rev`` -> frame n-1). Returns a sorted, de-duplicated list.
    """
    if n_frames <= 0:
        return []
    if n_frames == 1:
        return [0]
    k = max(2, int(n_probes))
    idx = {0, n_frames - 1}
    for j in range(k):
        idx.add(round(j * (n_frames - 1) / (k - 1)))
    return sorted(i for i in idx if 0 <= i < n_frames)


def select_instance(scores, coverages, cfg: dict) -> int | None:
    """Pick the best SAM 3 instance for a frame: highest score inside the coverage band.

    ``coverages`` are fractions of frame area. Instances outside
    ``[seed.min_coverage, seed.max_coverage]`` are rejected outright (a speck or a runaway
    whole-frame mask is never the subject), then the highest-scoring survivor wins. Ties break to the
    LOWER index (SAM 3 returns instances score-ordered). Returns ``None`` when nothing qualifies.
    """
    lo = float(cfg["seed"]["min_coverage"])
    hi = float(cfg["seed"]["max_coverage"])
    best, best_score = None, None
    for i, (s, c) in enumerate(zip(scores, coverages)):
        if not (lo <= float(c) <= hi):
            continue
        if best_score is None or float(s) > best_score:
            best, best_score = i, float(s)
    return best


def _valid(cand: dict | None, floor: float) -> bool:
    return bool(cand) and float(cand.get("score", 0.0)) >= floor


def decide_seed(probes: dict, n_frames: int, cfg: dict) -> dict:
    """Choose the seed frame + propagation direction from the per-frame probe results.

    Args:
        probes: ``{frame_idx: {'score': float, 'coverage': float} | None}`` — ``None`` (or a missing
            key) means SAM 3 returned no qualifying instance on that frame.
        n_frames: total frames in the clip (frame ``n_frames - 1`` is the ``rev`` seed frame).
        cfg: a loaded textseed config; ``min_score`` / ``weak_score`` / ``start_margin`` come FROM
            here (config-first), never from a literal.

    Returns a record ``{'found', 'seed_frame_idx', 'direction', 'score', 'coverage', 'reason',
    'weak'}``. ``found=False`` (direction ``None``) means the clip is NOT seedable — the caller must
    report it, never write an empty seed.
    """
    s = cfg["seed"]
    min_score = float(s["min_score"])
    weak_score = float(s["weak_score"])
    margin = float(s["start_margin"])
    last = max(0, int(n_frames) - 1)

    first_c = probes.get(0)
    last_c = probes.get(last)

    def _rec(idx, direction, cand, reason, weak=False):
        return {
            "found": True,
            "seed_frame_idx": int(idx),
            "direction": direction,
            "score": round(float(cand["score"]), 4),
            "coverage": round(float(cand["coverage"]), 5),
            "reason": reason,
            "weak": bool(weak),
        }

    first_ok = _valid(first_c, min_score)
    last_ok = _valid(last_c, min_score)

    if first_ok and last_ok:
        if float(last_c["score"]) > float(first_c["score"]) + margin:
            return _rec(last, REV, last_c, REASON_END_BEATS_START)
        return _rec(0, FWD, first_c, REASON_START)
    if first_ok:
        return _rec(0, FWD, first_c, REASON_START)
    if last_ok:
        return _rec(last, REV, last_c, REASON_END)

    # Neither endpoint cleared min_score. Fall back to the better WEAK endpoint (flagged), so a clip
    # whose subject is small/occluded at both ends still gets tracked rather than silently dropped.
    weak_first = _valid(first_c, weak_score)
    weak_last = _valid(last_c, weak_score)
    if weak_first and weak_last:
        if float(last_c["score"]) > float(first_c["score"]):
            return _rec(last, REV, last_c, REASON_WEAK, weak=True)
        return _rec(0, FWD, first_c, REASON_WEAK, weak=True)
    if weak_first:
        return _rec(0, FWD, first_c, REASON_WEAK, weak=True)
    if weak_last:
        return _rec(last, REV, last_c, REASON_WEAK, weak=True)

    return {
        "found": False,
        "seed_frame_idx": None,
        "direction": None,
        "score": 0.0,
        "coverage": 0.0,
        "reason": REASON_NOT_FOUND,
        "weak": False,
    }


def mask_stem(clip_stem: str, mask_type: str) -> str:
    """``('segtest_a_01', 'full_body') -> 'segtest_a_01__full_body'`` (propagate's stem contract)."""
    return f"{clip_stem}__{mask_type}"


def rev_stems(records) -> list[str]:
    """Mask stems whose direction is ``rev`` — fed straight into ``propagate``'s ``--rev`` set."""
    return [r["mask_stem"] for r in records if r.get("direction") == REV]


def summarize(records) -> dict:
    """Aggregate counts for the run report (pure — used by both the CLI and the tests)."""
    found = [r for r in records if r.get("found")]
    return {
        "clips": len(records),
        "seeded": len(found),
        "fwd": sum(1 for r in found if r["direction"] == FWD),
        "rev": sum(1 for r in found if r["direction"] == REV),
        "weak": sum(1 for r in found if r.get("weak")),
        "not_found": [r["clip_stem"] for r in records if not r.get("found")],
    }


# --------------------------------------------------------------------------------------------------
# SAM 3 concept detection (heavy deps — function-local).
# --------------------------------------------------------------------------------------------------

def load_detector(cfg: dict):
    """Load ``Sam3Model`` + ``Sam3Processor`` per the config's model block. Returns (model, proc, dev)."""
    torch = _need("torch")
    _need("transformers")
    from transformers import Sam3Model, Sam3Processor  # noqa: PLC0415

    m = cfg["model"]
    device = m.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"[textseed] model.device=cuda but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}). Re-run on a CUDA box or set model.device: cpu."
        )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        str(m.get("dtype", "bfloat16"))
    ]
    model_id = m["id"]
    proc = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, proc, device


def detect_frame(model, proc, device, frame_bgr, prompt: str, cfg: dict):
    """Run one concept detection on a BGR frame. Returns ``(bool mask HxW | None, record | None)``.

    Mirrors the validated probe: ``post_process_instance_segmentation(threshold, mask_threshold,
    target_sizes=inputs['original_sizes'])`` -> ``masks`` + ``scores``; the winning instance is chosen
    by :func:`select_instance` (coverage-banded, highest score).
    """
    torch = _need("torch")
    cv2 = _need("cv2")
    np = _need("numpy")
    d = cfg["detect"]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(images=rgb, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(
        out,
        threshold=float(d["threshold"]),
        mask_threshold=float(d["mask_threshold"]),
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    scores = [float(s) for s in res["scores"]]
    if not scores:
        return None, None
    masks = res["masks"]
    covs = [float(np.asarray(masks[i].cpu()).astype(bool).mean()) for i in range(len(scores))]
    pick = select_instance(scores, covs, cfg)
    if pick is None:
        return None, None
    mask = np.asarray(masks[pick].cpu()).astype(bool)
    while mask.ndim > 2:
        mask = mask[0]
    return mask, {"score": scores[pick], "coverage": covs[pick], "instances": len(scores)}


def read_frame(clip: Path, frame_idx: int):
    """Read a single frame (BGR) from a clip by index, plus ``(n_frames, w, h, fps)``."""
    cv2 = _need("cv2")
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    idx = frame_idx if frame_idx >= 0 else n + frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None, (n, 0, 0, fps)
    h, w = fr.shape[:2]
    return fr, (n, w, h, fps)


def count_frames(clip: Path) -> tuple[int, int, int, float]:
    """Decode-count a clip's frames (ProRes container metadata is not always trustworthy)."""
    cv2 = _need("cv2")
    cap = cv2.VideoCapture(str(clip))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    n, w, h = 0, 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n == 0:
            h, w = fr.shape[:2]
        n += 1
    cap.release()
    return n, w, h, fps


def seed_from_text(clip: Path, prompt: str, cfg: dict, detector, seeds_out: Path) -> dict:
    """Probe a clip, decide fwd/rev, and write the WHITE(255) seed PNG. Returns the per-clip record.

    The record carries ``{clip_stem, mask_stem, clip, n_frames, seed_frame_idx, direction, score,
    coverage, reason, weak, found, probes}`` — the ``probes`` map is the evidence trail showing where
    in the clip the subject was actually detected.
    """
    cv2 = _need("cv2")
    np = _need("numpy")
    model, proc, device = detector
    clip_stem = clip.stem
    stem = mask_stem(clip_stem, cfg["seed"]["mask_type"])

    n, w, h, fps = count_frames(clip)
    rec = {
        "clip_stem": clip_stem,
        "mask_stem": stem,
        "clip": str(clip),
        "n_frames": n,
        "width": w,
        "height": h,
        "fps": fps,
        "prompt_used": True,  # the prompt itself is campaign material — never written to disk here
    }
    if n <= 0:
        rec.update({"found": False, "direction": None, "reason": "no-frames", "probes": {}})
        return rec

    probes: dict[int, dict | None] = {}
    masks: dict[int, object] = {}
    for idx in probe_indices(n, cfg["seed"]["n_probes"]):
        fr, _ = read_frame(clip, idx)
        if fr is None:
            probes[idx] = None
            continue
        mask, info = detect_frame(model, proc, device, fr, prompt, cfg)
        probes[idx] = info
        if mask is not None:
            masks[idx] = mask

    decision = decide_seed(probes, n, cfg)
    rec.update(decision)
    rec["probes"] = {
        str(k): (None if v is None else {"score": round(v["score"], 4), "coverage": round(v["coverage"], 5)})
        for k, v in probes.items()
    }
    rec["probe_hits"] = sorted(int(k) for k, v in probes.items() if v is not None)

    if not decision["found"]:
        return rec

    seed_mask = masks.get(decision["seed_frame_idx"])
    if seed_mask is None:  # defensive: decision pointed at a frame with no mask
        rec.update({"found": False, "direction": None, "reason": REASON_NOT_FOUND})
        return rec

    seeds_out.mkdir(parents=True, exist_ok=True)
    seed_png = seeds_out / f"{stem}.png"
    cv2.imwrite(str(seed_png), np.asarray(seed_mask, dtype="uint8") * 255)
    rec["seed_png"] = str(seed_png)
    rec["seed_px"] = int(np.asarray(seed_mask).sum())
    return rec


def write_records(records, out_json: Path) -> Path:
    """Write the per-clip seed records as JSON (dir created; never deletes an existing file)."""
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return out_json


def write_manifest(records, out_txt: Path) -> Path:
    """Emit propagate's ``NN | <clip_stem>.png | <clip path>`` manifest for stem->clip resolution."""
    out_txt = Path(out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{i:02d} | {r['clip_stem']}.png | {r['clip']}"
        for i, r in enumerate(records, start=1)
        if r.get("found")
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_txt


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------

def discover_clips(clips_dir: Path, exts=(".mov", ".mp4", ".mkv")) -> list[Path]:
    """Sorted clip files in a directory (extension set is explicit — ProRes .mov included)."""
    clips = [p for p in sorted(Path(clips_dir).iterdir()) if p.suffix.lower() in exts]
    return clips


def dry_run(clips, cfg: dict, prompt: str | None) -> int:
    """Resolve everything that does NOT need the model: clips, frame counts, config shape."""
    print(f"[textseed] DRY-RUN: {len(clips)} clip(s)")
    print(f"[textseed] model={cfg['model']['id']} dtype={cfg['model']['dtype']} device={cfg['model']['device']}")
    print(
        f"[textseed] detect threshold={cfg['detect']['threshold']} mask_threshold={cfg['detect']['mask_threshold']} "
        f"| seed min_score={cfg['seed']['min_score']} weak={cfg['seed']['weak_score']} "
        f"margin={cfg['seed']['start_margin']} n_probes={cfg['seed']['n_probes']}"
    )
    print(f"[textseed] prompt: {'SET' if prompt else 'MISSING (--prompt required)'}")
    bad = 0 if prompt else 1
    for clip in clips:
        n, w, h, fps = count_frames(clip)
        idx = probe_indices(n, cfg["seed"]["n_probes"])
        status = "ok" if n > 0 else "FAIL"
        if n <= 0:
            bad += 1
        print(f"  {status:4s} {clip.name}: {w}x{h} {n}f @{fps:g}fps | probes={idx}")
    print(f"[textseed] DRY-RUN done: {len(clips) - max(0, bad)} ok, {bad} problem(s). No model loaded.")
    return 1 if bad else 0


def run(args) -> int:
    """Auto-seed every clip in ``--clips-dir``. Returns a process exit code."""
    cfg = load_textseed_config(args.config)
    prompt = args.prompt or cfg["seed"].get("prompt")
    for key, val in (("min_score", getattr(args, "min_score", None)),
                     ("n_probes", getattr(args, "n_probes", None))):
        if val is not None:
            cfg["seed"][key] = val
    if getattr(args, "mask_type", None):
        cfg["seed"]["mask_type"] = args.mask_type
    if getattr(args, "device", None):
        cfg["model"]["device"] = args.device

    clips = discover_clips(Path(args.clips_dir))
    if not clips:
        raise SystemExit(f"[textseed] no clips found in {args.clips_dir}")
    if getattr(args, "only", None):
        only = set(args.only)
        clips = [c for c in clips if c.stem in only]

    if args.dry_run:
        return dry_run(clips, cfg, prompt)
    if not prompt:
        raise SystemExit("[textseed] --prompt is required (the concept prompt is campaign material)")

    detector = load_detector(cfg)
    print(f"[textseed] detector loaded ({cfg['model']['id']}) — seeding {len(clips)} clip(s)", flush=True)
    seeds_out = Path(args.seeds_out)
    records = []
    for i, clip in enumerate(clips, start=1):
        try:
            rec = seed_from_text(clip, prompt, cfg, detector, seeds_out)
        except Exception as exc:  # keep going; never delete partial outputs
            print(f"[textseed] {i}/{len(clips)} {clip.name}: ERROR {exc}", flush=True)
            records.append({"clip_stem": clip.stem, "clip": str(clip), "found": False,
                            "direction": None, "reason": f"error: {exc}"})
            continue
        records.append(rec)
        if rec.get("found"):
            print(
                f"[textseed] {i}/{len(clips)} {rec['clip_stem']}: seed@{rec['seed_frame_idx']}/"
                f"{rec['n_frames'] - 1} dir={rec['direction']} score={rec['score']:.3f} "
                f"cov={rec['coverage'] * 100:.2f}% reason={rec['reason']}"
                f"{' WEAK' if rec.get('weak') else ''}",
                flush=True,
            )
        else:
            print(
                f"[textseed] {i}/{len(clips)} {rec['clip_stem']}: NOT SEEDABLE ({rec['reason']}) "
                f"probe_hits={rec.get('probe_hits', [])}",
                flush=True,
            )

    write_records(records, Path(args.records_out))
    write_manifest(records, Path(args.manifest_out))
    s = summarize(records)
    print(
        f"[textseed] SUMMARY clips={s['clips']} seeded={s['seeded']} fwd={s['fwd']} rev={s['rev']} "
        f"weak={s['weak']} not_found={s['not_found']}",
        flush=True,
    )
    print(f"[textseed] rev stems: {rev_stems(records)}", flush=True)
    print("TEXTSEED_DONE", flush=True)
    return 0
