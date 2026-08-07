#!/usr/bin/env python
"""prep_sam3_probe.py -- SAM3 setup probe for phase 09.3 Plan 02 (D-14).

A standalone, import-confined probe that resolves the SAM3 backend for Plan 05
(`src/signet_trainer/prep/propagate.py`). It:

  (a) HEADs a gated ``facebook/sam3`` file and asserts HTTP 200 (gated access resolved);
  (b) detects + imports the RESOLVED SAM3 backend, preferring the standalone
      ``facebookresearch/sam3`` package (which matches the r1 ``_sam_propagate_masks.py``
      API verbatim) and falling back to the ``transformers>=5.5`` native tracker-video API;
  (c) functionally propagates a tiny synthetic BINARY MASK SEED across ``N_FRAMES`` frames
      on CUDA and prints the per-frame boolean mask shapes -- this resolves RESEARCH A5
      (does the chosen path accept a binary mask seed).

It records the decision Plan 05 consumes on a single marker line:

    SAM3_BACKEND_RESOLVED: <standalone|transformers>  mask_seed=<yes|no>

Run:
    PYTHONUTF8=1 .venv-sam/Scripts/python scripts/prep_sam3_probe.py

Notes
-----
* Heavy deps (``torch`` / ``transformers`` / ``sam3``) are imported *function-local* so this
  file stays import-safe on a CPU/CI box with none of them present.
* The HF token is read from ``~/.cache/huggingface/token`` (or ``HF_TOKEN`` /
  ``HUGGING_FACE_HUB_TOKEN``) and is NEVER printed or logged -- only the HTTP status is.
* On Windows the standalone ``facebookresearch/sam3`` package is not usable (its EDT kernel
  hard-imports ``triton``, which has no authoritative Windows wheel); this probe therefore
  resolves to the ``transformers`` backend on this box. The standalone branch is kept so the
  same probe reports ``standalone`` on a Linux box where it installs.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---- constants ------------------------------------------------------------------------------
HF_REPO = "facebook/sam3"
GATED_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/config.json"
N_FRAMES = 3          # >= 3 per-frame masks required by the plan's acceptance criteria
FRAME_HW = (256, 256)  # small synthetic frames -- enough to prove propagation, cheap on VRAM


# ---- gated-access check (stdlib only; token never logged) -----------------------------------
def _hf_token() -> str | None:
    import os

    tokfile = Path.home() / ".cache" / "huggingface" / "token"
    if tokfile.exists():
        tok = tokfile.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def check_gated_head() -> int:
    """HEAD a gated ``facebook/sam3`` file and return the HTTP status code.

    Only the status code is ever surfaced -- the bearer token is never printed.
    """
    import urllib.error
    import urllib.request

    tok = _hf_token()
    req = urllib.request.Request(GATED_URL, method="HEAD")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


# ---- synthetic probe scene ------------------------------------------------------------------
def _synthetic_video(n_frames: int, hw: tuple[int, int]):
    """A moving bright square on a dark background -- gives the tracker something to follow."""
    import numpy as np

    h, w = hw
    frames = []
    for i in range(n_frames):
        fr = np.zeros((h, w, 3), dtype=np.uint8)
        x0 = 40 + i * 12
        fr[80:170, x0 : x0 + 90] = 220
        frames.append(fr)
    return np.stack(frames, axis=0)  # (T, H, W, 3) uint8


def _synthetic_seed(hw: tuple[int, int]):
    """A binary mask over the square in frame 0 (WHITE(1) = the region to track)."""
    import numpy as np

    h, w = hw
    seed = np.zeros((h, w), dtype=np.uint8)
    seed[80:170, 40:130] = 1
    return seed


def _as_bool_2d(mask) -> tuple:
    """Squeeze a post-processed mask tensor to a 2D numpy bool array; return (shape, sum)."""
    import numpy as np
    import torch

    if isinstance(mask, torch.Tensor):
        arr = mask.detach().to(torch.float32).cpu().numpy()
    else:
        arr = np.asarray(mask, dtype="float32")
    arr = np.squeeze(arr)
    while arr.ndim > 2:
        arr = arr[0]
    b = arr > 0.5
    return b.shape, int(b.sum())


# ---- backend: transformers>=5.5 native tracker-video ----------------------------------------
def _run_transformers(n_frames: int, hw: tuple[int, int]):
    import numpy as np  # noqa: F401
    import torch
    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3TrackerVideoModel.from_pretrained(HF_REPO).to(device).eval()
    processor = Sam3TrackerVideoProcessor.from_pretrained(HF_REPO)

    video = _synthetic_video(n_frames, hw)
    session = processor.init_video_session(video=video, inference_device=device)

    seed = _synthetic_seed(hw)
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=1,
        input_masks=seed,
    )

    results = []
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(session, start_frame_idx=0):
            masks = processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[session.video_height, session.video_width]],
                binarize=True,
            )[0]
            shape, npix = _as_bool_2d(masks[0])
            results.append((int(out.frame_idx), shape, npix))
    return results


# ---- backend: standalone facebookresearch/sam3 (r1 API verbatim) ----------------------------
def _run_standalone(n_frames: int, hw: tuple[int, int]):
    import tempfile

    import cv2
    import numpy as np
    import torch
    from sam3.model_builder import build_sam3_video_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_video_model(checkpoint_path=None, device=device)
    predictor = model.tracker
    predictor.backbone = model.detector.backbone  # [canonical] wiring per the sam3 example nb

    video = _synthetic_video(n_frames, hw)
    seed = _synthetic_seed(hw).astype(np.bool_)

    td = tempfile.mkdtemp(prefix="sam3_probe_")
    for j in range(n_frames):
        cv2.imwrite(str(Path(td) / f"{j:05d}.jpg"), cv2.cvtColor(video[j], cv2.COLOR_RGB2BGR))

    results = []
    ctx = torch.inference_mode()
    with ctx:
        state = predictor.init_state(video_path=td, offload_video_to_cpu=True, offload_state_to_cpu=True)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(seed).float())
        for frame_idx, _ids, _low, video_res_masks, _scores in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=n_frames, reverse=False, propagate_preflight=True
        ):
            shape, npix = _as_bool_2d(video_res_masks[0] > 0.0)
            results.append((int(frame_idx), shape, npix))
    return results


def resolve_backend():
    """Return ``(name, runner)`` preferring the standalone API, then transformers."""
    try:
        import sam3  # noqa: F401
        from sam3.model_builder import build_sam3_video_model  # noqa: F401

        return "standalone", _run_standalone
    except Exception:
        pass
    from transformers import Sam3TrackerVideoModel  # noqa: F401

    return "transformers", _run_transformers


def main() -> int:
    print(f"[sam3-probe] gated HEAD {HF_REPO} ...", flush=True)
    status = check_gated_head()
    print(f"[sam3-probe] HEAD status: {status}", flush=True)
    if status != 200:
        print(
            f"[sam3-probe] FAIL: gated HEAD returned {status} (expected 200). "
            "HF access to facebook/sam3 is not granted for this token.",
            file=sys.stderr,
        )
        return 1

    backend, runner = resolve_backend()
    print(f"[sam3-probe] resolved backend: {backend}", flush=True)
    print("[sam3-probe] loading weights + propagating a synthetic binary mask seed ...", flush=True)

    results = runner(N_FRAMES, FRAME_HW)
    if len(results) < 3:
        print(
            f"[sam3-probe] FAIL: expected >=3 propagated frames, got {len(results)}.",
            file=sys.stderr,
        )
        return 1

    print(f"[sam3-probe] propagated {len(results)} frames from a binary mask seed:", flush=True)
    for frame_idx, shape, npix in results:
        print(f"[sam3-probe]   frame {frame_idx}: mask bool shape {shape}  set_pixels={npix}", flush=True)

    # The marker line Plan 05 (prep/propagate.py) reads to pick its backend + mask-seed contract.
    print(f"SAM3_BACKEND_RESOLVED: {backend}  mask_seed=yes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
