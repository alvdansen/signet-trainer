"""prep.propagate — SAM mask propagation for the inpaint gate (D-13/D-14/D-08/D-02).

Formalizes ``scripts/_sam_propagate_masks.py`` into the ``prep`` package. The operator's frame-0 binary
seeds (teal paintovers extracted by :mod:`prep.chroma`, or app-drawn masks) seed
``frame_idx=0`` and propagate across the matching source clip via SAM, writing:

  * per-frame binary mask PNGs -> ``<masks-out>/<mask_stem>/NNNNN.png`` (contiguous %05d,
    region=WHITE(255) — the D-05 polarity chain's first link; staging negates downstream);
  * a green-overlay QA mp4 -> ``<overlays-out>/<mask_stem>_overlay.mp4`` (operator eyeballs for drift
    BEFORE any encode — the D-04 hard gate) + an ``avg_cover`` per clip (the D-02 coverage feed).

D-13 corrections vs the r1 harvest source:
  1. STEM RESOLUTION IS SINGLE-SOURCED. The r1 script carried inline copies of the
     stem-resolution helpers; this module imports them from :mod:`prep.resolve` (closing the
     divergent-copy drift). The seed binarize threshold comes FROM ``MASK-SPEC.yaml``
     (``thresholds.seed_load`` = 127) rather than a hardcoded literal (D-03).
  2. THE D-02 AUTO-EXTEND IS WIRED IN. After the backend returns the exact per-frame masks, they
     are DILATED per subject class toward the spec's coverage target via
     :func:`prep.dilate.dilate_to_target` — BEFORE the WHITE PNGs, BEFORE the green-overlay QA mp4,
     and BEFORE ``avg_cover`` — so the QA gate + coverage number judge the EXTENDED mask that won in
     r1, never the raw exact-SAM mask that HURT likeness (D-02, LOCKED).

DUAL BACKEND (``--backend``), stable seam ``run(jpeg_dir, seed_bool_hw, n_frames) -> {idx: bool}``:
  * ``sam3`` (DEFAULT) — SAM 3. Auto-resolves the install Plan 02 landed: prefers the standalone
    ``facebookresearch/sam3`` API (Linux; ``build_sam3_video_model`` + ``add_new_mask`` /
    ``propagate_in_video``, r1 verbatim) and falls back to the ``transformers>=5.5`` native
    tracker-video API (``Sam3TrackerVideoModel`` / ``Sam3TrackerVideoProcessor`` /
    ``propagate_in_video_iterator``, seeding via ``add_inputs_to_inference_session(input_masks=)``).
    Keeping the SAM3 install choice behind this seam is what localizes it (RESEARCH Pitfall 1).
    Plan 02 resolved ``SAM3_BACKEND_RESOLVED: transformers  mask_seed=yes`` on Windows.
  * ``sam21`` — SAM 2.1 (``sam2`` package, sam2.1_hiera_large). The LAST-DITCH fallback ONLY
    (D-14); it was validated on Wan, may NOT track LTX content as well, and requires ``--checkpoint``.

REV set (D-08): pass a mask stem to ``rev`` to seed the LAST frame and propagate BACKWARD — done
precedent-style by reversing the physical frame order fed to the predictor and remapping indices
(backend-uniform; re-seeds clips that drifted forward). ``load_directions`` reads the SAME set from
textseed's ``textseed_records.json`` (``--directions``, #36 finding 3) so the fwd/rev decision
reaches this module as structured data, not a printed line the operator retypes. Every ``--rev`` /
``--only`` / ``--directions`` stem is validated against the discovered seed PNGs in
``propagate_masks`` — an unmatched stem is a fatal error, never a silent no-op.

RUNNABLE-LATER: NO heavy imports at module top — torch/cv2/numpy/sam/transformers imports are
function-local and fail with a clear install hint, so this module PARSES + unit-tests on a CPU/CI
box. NO metered dispatch, NO Modal import — local/CPU-or-local-GPU only. Never deletes outputs.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from signet_trainer import harness_data
from signet_trainer.prep import dilate
from signet_trainer.prep.resolve import (
    load_manifest_map,
    resolve_clip,
    sanitize_stem,  # noqa: F401  (re-exported for callers that resolved stems via this module)
    split_mask_stem,
)
from signet_trainer.prep.spec import load_mask_spec

REPO_ROOT = Path(__file__).resolve().parents[3]
PREP_DIR = REPO_ROOT / "_inpaint_prep"

DEFAULT_MASKS_DIR = PREP_DIR / "masks_frame0"
DEFAULT_CLIPS_DIR = PREP_DIR
DEFAULT_MANIFEST = PREP_DIR / "manifest.txt"  # "NN | first_frame.png | source clip.mp4" lines
DEFAULT_MASKS_OUT = PREP_DIR / "masks_video"
DEFAULT_OVERLAYS_OUT = PREP_DIR / "overlays_video"


def default_spec() -> Path:
    """The PACKAGED MASK-SPEC.yaml (``signet_trainer.harness_data``), honouring the override env var.

    Resolved through ``importlib.resources`` at CALL time, not a ``.planning/harness/`` path built
    from ``__file__``: the spec now ships inside the package, so this default resolves for a fresh
    ``pip install`` that has no ``.planning/`` at all (and still works from a zip import). Pass
    ``--spec`` to point at any other file; set ``SIGNET_HARNESS_DATA_DIR`` to override every packaged
    spec at once.
    """
    return harness_data.spec_path(harness_data.MASK_SPEC)

SAM3_HF_REPO = "facebook/sam3"  # gated — Plan 02 resolved access with the local token
SAM21_CFG_DEFAULT = "configs/sam2.1/sam2.1_hiera_l.yaml"  # [precedent] prior project's SAM paint-seed script

# Plan 02 resolved: the transformers SAM3 path DOES accept a binary mask seed
# (processor.add_inputs_to_inference_session(input_masks=...)). If a future backend cannot seed with
# a mask, propagate.py raises loudly rather than silently degrading to a point prompt (T-09.3-05-D).
SAM3_TRANSFORMERS_MASK_SEED = True

INSTALL_HINTS = {
    "torch": "pip install torch --index-url https://download.pytorch.org/whl/cu128  (CUDA build)",
    "cv2": "pip install opencv-python",
    "numpy": "pip install numpy",
    "sam2": "pip install 'git+https://github.com/facebookresearch/sam2.git'  + download sam2.1_hiera_large.pt",
    # D-14 fix-it-first SAM3 path (Plan 02): grant HF access -> install transformers -> weights auto-pull.
    "sam3": (
        "SAM3 (D-14 fix-it-first, in order): "
        "1) request + confirm HF gated access to facebook/sam3; "
        "2) install the transformers backend into .venv-sam: pip install 'transformers>=5.5'; "
        "3) weights auto-download once into the HF cache on first run "
        "(standalone facebookresearch/sam3 is Linux-only — its EDT kernel hard-imports triton)"
    ),
}


def _need(module: str):
    """Import a heavy dep lazily with a clear install hint (module must PARSE without them)."""
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            f"[sam-propagate] missing dependency '{module}': {exc}\n"
            f"    install hint: {INSTALL_HINTS.get(module, f'pip install {module}')}"
        ) from exc


# --------------------------------------------------------------------------------------------------
# Job discovery: one job per seed PNG. Stem resolution is single-sourced via prep.resolve (D-13).
# --------------------------------------------------------------------------------------------------

def load_directions(records_path: str | Path) -> set[str]:
    """Read textseed's structured JSON handoff (``--directions``) and return its REV mask-stem set.

    Replaces copy-pasting the ``[textseed] rev stems: [...]`` print into ``--rev`` by hand (#36
    finding 3): this reads the SAME ``direction`` field, from the SAME ``textseed_records.json``
    that print was only summarizing, so a file is the transport, not an operator's clipboard.
    """
    import json  # noqa: PLC0415  (function-local, import-confined)

    from signet_trainer.prep.textseed import rev_stems  # noqa: PLC0415

    records = json.loads(Path(records_path).read_text(encoding="utf-8"))
    return set(rev_stems(records))


def discover_jobs(args) -> list[dict]:
    """One job per seed PNG: {'mask_stem','type','seed_png','clip'} (clip=None if unresolved)."""
    manifest_map = load_manifest_map(Path(args.manifest))
    jobs = []
    for seed_png in sorted(Path(args.masks_dir).glob("*.png")):
        mask_stem = seed_png.stem
        if args.only and mask_stem not in args.only:
            continue
        try:
            _clip_part, mask_type = split_mask_stem(mask_stem)
        except ValueError as exc:
            print(f"[sam-propagate] SKIP {seed_png.name}: {exc}")
            continue
        clip = resolve_clip(_clip_part, manifest_map, Path(args.clips_dir))
        jobs.append({"mask_stem": mask_stem, "type": mask_type, "seed_png": seed_png, "clip": clip})
    return jobs


# --------------------------------------------------------------------------------------------------
# Frame IO + seed loading (cv2/numpy, lazy).
# --------------------------------------------------------------------------------------------------

def read_frames(clip: Path):
    """Return (frames BGR list, fps) for a clip. [precedent] the prior project reads all frames up-front."""
    cv2 = _need("cv2")
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, float(fps)


def load_seed(seed_png: Path, w: int, h: int, spec: dict):
    """Load a binary seed PNG -> bool HxW at the clip's resolution (nearest resize if needed).

    The binarize threshold is read FROM the spec (``thresholds.seed_load`` = 127) rather than a
    hardcoded literal (D-03/config-first) — the value equals the r1 ``>127`` rule but is now sourced.
    """
    cv2 = _need("cv2")
    np = _need("numpy")
    thr = int(spec["thresholds"]["seed_load"])
    im = cv2.imread(str(seed_png), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise SystemExit(f"[sam-propagate] cannot read seed PNG: {seed_png}")
    if (im.shape[1], im.shape[0]) != (w, h):
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
    return (im > thr).astype(np.bool_)


# --------------------------------------------------------------------------------------------------
# Backends. Each returns a runner: (jpeg_dir, seed_bool_hw, n_frames) -> {temp_idx: bool HxW}.
# The caller remaps temp indices through the (possibly reversed) frame order.
# --------------------------------------------------------------------------------------------------

def _mask_to_bool_2d(mask, np, torch=None):
    """Squeeze a (possibly tensor) mask down to a 2D numpy bool array."""
    if torch is not None and isinstance(mask, torch.Tensor):
        arr = mask.detach().to(torch.float32).cpu().numpy()
    else:
        arr = np.asarray(mask, dtype="float32")
    arr = np.squeeze(arr)
    while arr.ndim > 2:
        arr = arr[0]
    return (arr > 0.5).astype(bool)


def _make_sam21_runner(args):
    """SAM 2.1 last-ditch backend (D-14) — [precedent] prior project's SAM paint-seed script, needs --checkpoint."""
    torch = _need("torch")
    np = _need("numpy")
    if not args.checkpoint:
        raise SystemExit(
            "[sam-propagate] --backend sam21 requires --checkpoint /path/to/sam2.1_hiera_large.pt\n"
            f"    install hint: {INSTALL_HINTS['sam2']}"
        )
    try:
        from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            f"[sam-propagate] missing dependency 'sam2': {exc}\n    install hint: {INSTALL_HINTS['sam2']}"
        ) from exc
    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device=args.device)

    def run(jpeg_dir: str, seed, n_frames: int):
        out = {}
        with _autocast_ctx(torch, args.device):
            state = predictor.init_state(
                video_path=jpeg_dir, offload_video_to_cpu=True, offload_state_to_cpu=True
            )
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(seed))
            for jidx, _ids, logits in predictor.propagate_in_video(state):
                out[int(jidx)] = _mask_to_bool_2d(logits[0] > 0.0, np, torch)
        return out

    return run


def _make_sam3_standalone_runner(args):
    """Standalone facebookresearch/sam3 (r1 API verbatim — Linux boxes where triton installs)."""
    torch = _need("torch")
    np = _need("numpy")
    from sam3.model_builder import build_sam3_video_model  # noqa: PLC0415

    model = build_sam3_video_model(
        checkpoint_path=args.checkpoint or None,  # None -> load_from_HF (facebook/sam3, GATED)
        device=args.device,
    )
    predictor = model.tracker
    predictor.backbone = model.detector.backbone  # [canonical] required wiring per the example nb

    def run(jpeg_dir: str, seed, n_frames: int):
        out = {}
        with _autocast_ctx(torch, args.device):
            state = predictor.init_state(
                video_path=jpeg_dir, offload_video_to_cpu=True, offload_state_to_cpu=True
            )
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(seed).float())
            for frame_idx, _ids, _low, video_res_masks, _scores in predictor.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=n_frames, reverse=False,
                propagate_preflight=True,
            ):
                out[int(frame_idx)] = _mask_to_bool_2d(video_res_masks[0] > 0.0, np, torch)
        return out

    return run


def _make_sam3_transformers_runner(args):
    """transformers>=5.5 native tracker-video backend (Plan 02 resolved path on Windows).

    Seeds with the binary mask via ``processor.add_inputs_to_inference_session(input_masks=...)``
    (Plan 02: ``mask_seed=yes``) — if that contract were ever unavailable the standalone install is
    required and this raises loudly rather than silently seeding with a point prompt (T-09.3-05-D).
    """
    if not SAM3_TRANSFORMERS_MASK_SEED:
        raise RuntimeError(
            "[sam-propagate] the transformers SAM3 path cannot seed with a binary mask on this "
            "install; the paint-seed workflow REQUIRES the standalone facebookresearch/sam3 backend. "
            "Install standalone sam3 (Linux) — do NOT fall back to a point prompt."
        )
    torch = _need("torch")
    np = _need("numpy")
    try:
        from transformers import (  # noqa: PLC0415
            Sam3TrackerVideoModel,
            Sam3TrackerVideoProcessor,
        )
    except ImportError as exc:
        raise SystemExit(
            f"[sam-propagate] SAM3 backend unavailable: {exc}\n    {INSTALL_HINTS['sam3']}"
        ) from exc

    device = args.device
    model = Sam3TrackerVideoModel.from_pretrained(SAM3_HF_REPO).to(device).eval()
    processor = Sam3TrackerVideoProcessor.from_pretrained(SAM3_HF_REPO)

    def run(jpeg_dir: str, seed, n_frames: int):
        cv2 = _need("cv2")
        # Read the (physically ordered) temp jpegs back into an RGB video array. cv2 gives BGR;
        # the transformers processor expects RGB (matches the Plan 02 probe's synthetic RGB frames).
        paths = sorted(Path(jpeg_dir).glob("*.jpg"))
        frames_rgb = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
        video = np.stack(frames_rgb, axis=0)  # (T, H, W, 3) uint8

        session = _init_transformers_session(processor, video, device, args)
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=1,
            input_masks=seed.astype(np.uint8),  # bool HxW -> 0/1
        )

        out = {}
        with torch.inference_mode():
            for res in model.propagate_in_video_iterator(session, start_frame_idx=0):
                masks = processor.post_process_masks(
                    [res.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=True,
                )[0]
                out[int(res.frame_idx)] = _mask_to_bool_2d(masks[0], np, torch)
        return out

    return run


def _init_transformers_session(processor, video, device, args):
    """Open a tracker-video session, offloading video+state to CPU per the r1/Plan-02 discipline.

    The offload kwargs (``video_storage_device`` / ``inference_state_device`` = cpu) keep real
    production-resolution clips (1280x704, ~81f) off the GPU (Plan 02 SUMMARY). A transformers build that
    does not accept them degrades to the default (VRAM-heavier) session with a warning — never a
    seeding change.
    """
    storage = getattr(args, "storage_device", "cpu")
    state = getattr(args, "state_device", "cpu")
    try:
        return processor.init_video_session(
            video=video,
            inference_device=device,
            video_storage_device=storage,
            inference_state_device=state,
        )
    except TypeError:
        print(
            "[sam-propagate] WARN: this transformers build does not accept the CPU-offload session "
            "kwargs; falling back to the default (VRAM-heavier) session.",
            flush=True,
        )
        return processor.init_video_session(video=video, inference_device=device)


def _autocast_ctx(torch, device: str):
    """inference_mode (+ bf16 autocast on cuda) — [precedent]."""
    if device == "cuda":
        import contextlib  # noqa: PLC0415

        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
        return stack
    return torch.inference_mode()


def make_backend(args):
    """Resolve the propagation backend. ``sam3`` (DEFAULT) auto-resolves standalone->transformers;
    ``sam21`` is the last-ditch fallback (D-14).
    """
    torch = _need("torch")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "[sam-propagate] --device cuda but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}). Re-run on a CUDA box or pass --device cpu.\n"
            f"    install hint: {INSTALL_HINTS['torch']}"
        )

    if args.backend == "sam21":
        return _make_sam21_runner(args)

    # sam3 (default): prefer the standalone API (zero backend churn), fall back to transformers
    # (the Plan-02-resolved Windows path). Keeping the choice behind this seam localizes the install.
    try:
        import sam3  # noqa: F401,PLC0415
        from sam3.model_builder import build_sam3_video_model  # noqa: F401,PLC0415

        print("[sam-propagate] SAM3 backend: standalone facebookresearch/sam3", flush=True)
        return _make_sam3_standalone_runner(args)
    except Exception:
        print("[sam-propagate] SAM3 backend: transformers native tracker-video (Plan 02)", flush=True)
        return _make_sam3_transformers_runner(args)


# --------------------------------------------------------------------------------------------------
# --rev backward-seed (D-08): pure index helpers (physical frame reversal + remap), unit-testable.
# --------------------------------------------------------------------------------------------------

def frame_order(n_frames: int, reverse: bool) -> list[int]:
    """Physical frame indices fed to the predictor. ``reverse`` (D-08) feeds them last->first so a
    drifted clip can be re-seeded on its LAST frame and propagated backward (backend-uniform)."""
    order = list(range(n_frames))
    return order[::-1] if reverse else order


def remap_segmentation(seg_temp: dict, order: list[int]) -> dict:
    """Map temp-order predictor indices back to original frame indices (the D-08 REV remap).

    ``seg_temp[j]`` is the mask the predictor emitted for the j-th frame it saw; ``order[j]`` is that
    frame's ORIGINAL index. Forward: identity. Reverse: ``order = [n-1, n-2, ..., 0]`` so temp-0 maps
    back to frame n-1, restoring physical order.
    """
    return {order[j]: m for j, m in seg_temp.items()}


# --------------------------------------------------------------------------------------------------
# Per-clip pipeline: seed -> propagate -> D-02 dilate -> per-frame PNGs + green-overlay QA mp4.
# --------------------------------------------------------------------------------------------------

def process_job(job: dict, runner, args, spec: dict) -> str:
    cv2 = _need("cv2")
    np = _need("numpy")
    mask_stem = job["mask_stem"]
    frames, fps = read_frames(job["clip"])
    if not frames:
        return f"{mask_stem}: NO FRAMES in {job['clip']}"
    n = len(frames)
    h, w = frames[0].shape[:2]
    seed = load_seed(job["seed_png"], w, h, spec)

    rev = set(getattr(args, "rev", ()) or ())
    mode = "rev" if mask_stem in rev else "fwd"
    order = frame_order(n, reverse=(mode == "rev"))

    mask_dir = Path(args.masks_out) / mask_stem
    overlay_path = Path(args.overlays_out) / f"{mask_stem}_overlay.mp4"
    if mask_dir.exists() and any(mask_dir.glob("*.png")) and not args.force:
        return f"{mask_stem}: SKIP (already propagated at {mask_dir} — use --force to redo)"

    td = tempfile.mkdtemp(prefix="sam_prop_")
    try:
        for j, oi in enumerate(order):
            cv2.imwrite(str(Path(td) / f"{j:05d}.jpg"), frames[oi])
        seg_temp = runner(td, seed, n)
    finally:
        shutil.rmtree(td, ignore_errors=True)

    # Remap temp-order indices back to original frame indices ([precedent] REV remap, D-08).
    seg = remap_segmentation(seg_temp, order)

    # D-02 AUTO-EXTEND (LOCKED): grow the exact-SAM masks toward the per-class coverage target
    # BEFORE the PNG write, the QA overlay, and avg_cover — so the gate + coverage judge the
    # EXTENDED mask that won in r1, never the raw exact-SAM mask that HURT likeness.
    # Capture the pre-dilation (exact-SAM) coverage FIRST so the QA gate can read both numbers
    # per clip (r1: exact-SAM ~4.8% HURT likeness; extended ~14% won — the gate judges the delta).
    subject_class = job["type"]
    pre_cover = sum(float(m.mean()) for m in seg.values()) / max(1, n)  # exact-SAM, pre-dilation
    seg = dilate.dilate_to_target(seg, subject_class, spec)

    mask_dir.mkdir(parents=True, exist_ok=True)
    Path(args.overlays_out).mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cover, missing = 0.0, 0
    for i, fr in enumerate(frames):
        m = seg.get(i)
        if m is None:  # contiguous %05d sequence is REQUIRED for the staging ffmpeg render
            m = np.zeros((h, w), dtype=bool)
            missing += 1
        mm = m.astype(np.uint8)
        cv2.imwrite(str(mask_dir / f"{i:05d}.png"), mm * 255)
        green = np.zeros_like(fr)
        green[:, :] = (0, 200, 0)
        out = np.where(mm[..., None].astype(bool), (0.55 * fr + 0.45 * green).astype(np.uint8), fr)
        cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
        cover += float(mm.mean())
        vw.write(out)
    vw.release()
    note = f" MISSING-SEG on {missing} frame(s) (written as empty)" if missing else ""
    return (
        f"{mask_stem}: {n}f mode={mode} class={subject_class} seed_px={int(seed.sum())} "
        f"pre_cover={pre_cover:.3f} avg_cover={cover / max(1, n):.3f} fps={fps:g}{note} -> {mask_dir}"
    )


def dry_run(jobs: list[dict], args) -> int:
    """Resolve + verify everything that does NOT need the model: stems, clips, seed dims."""
    cv2 = _need("cv2")
    bad = 0
    print(f"[sam-propagate] DRY-RUN: {len(jobs)} seed(s) in {args.masks_dir}")
    for job in jobs:
        stem = job["mask_stem"]
        if job["clip"] is None:
            print(f"  FAIL {stem}: clip stem does not resolve (clips-dir={args.clips_dir})")
            bad += 1
            continue
        cap = cv2.VideoCapture(str(job["clip"]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        im = cv2.imread(str(job["seed_png"]), cv2.IMREAD_GRAYSCALE)
        seed_ok = im is not None and (im > 127).sum() >= 20
        dims_ok = im is not None and (im.shape[1], im.shape[0]) == (w, h)
        status = "ok" if (seed_ok and dims_ok) else "WARN"
        if not seed_ok:
            bad += 1
        seed_note = (
            "match" if dims_ok
            else f"{im.shape[1]}x{im.shape[0]} (will nearest-resize)" if im is not None
            else "UNREADABLE"
        )
        print(
            f"  {status:4s} {stem}: clip={job['clip'].name} {w}x{h} {n}f @{fps:g}fps | "
            f"seed dims {seed_note}{'' if seed_ok else ' | SEED EMPTY/UNREADABLE'}"
        )
    print(f"[sam-propagate] DRY-RUN done: {len(jobs) - bad} ok, {bad} problem(s). No model loaded.")
    return 1 if bad else 0


def propagate_masks(args) -> int:
    """Orchestrate propagation over every discovered seed. Returns a process exit code.

    Loads the mask spec once (D-03: the seed threshold + per-class dilation targets are spec-sourced),
    discovers jobs, and either dry-runs (no model) or propagates + dilates + writes QA. Prints the
    ``SAM_PROPAGATE_DONE`` stdout marker on a real run.
    """
    args.rev = set(getattr(args, "rev", []) or [])
    args.only = set(getattr(args, "only", []) or [])

    directions = getattr(args, "directions", None)
    if directions:
        # Structured handoff (#36 finding 3): union textseed's decided REV set into --rev rather
        # than requiring the operator to copy it off stdout by hand.
        args.rev |= load_directions(directions)

    jobs = discover_jobs(args)
    if not jobs:
        raise SystemExit(f"[sam-propagate] no seed PNGs found in {args.masks_dir}")

    # Typo-fatal (#36 finding 3): a --rev/--only/--directions stem that matches no discovered seed
    # PNG must abort loudly, never silently no-op — a dropped/mistyped --rev seeds the wrong frame
    # with nothing downstream able to see the mistake.
    discovered = sorted({j["mask_stem"] for j in jobs})
    for flag, stems in (("--rev", args.rev), ("--only", args.only)):
        unknown = sorted(s for s in stems if s not in discovered)
        if unknown:
            raise SystemExit(
                f"[sam-propagate] unknown {flag} stem(s) {unknown}; discovered: {discovered}"
            )

    unresolved = [j["mask_stem"] for j in jobs if j["clip"] is None]

    if args.dry_run:
        return dry_run(jobs, args)

    if unresolved:
        raise SystemExit(f"[sam-propagate] unresolved clip stems (fix before running): {unresolved}")

    spec = load_mask_spec(args.spec)  # fail-loud; the seed threshold + dilation targets come from here
    runner = make_backend(args)
    print(f"[sam-propagate] backend={args.backend} device={args.device} jobs={len(jobs)}")
    for job in jobs:
        try:
            print("[sam-propagate]", process_job(job, runner, args, spec), flush=True)
        except Exception as exc:  # keep going, precedent-style; never delete partial outputs
            print(f"[sam-propagate] {job['mask_stem']}: ERROR {exc}", flush=True)
    print("SAM_PROPAGATE_DONE", flush=True)
    return 0
