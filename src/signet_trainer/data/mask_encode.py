"""data.mask_encode — signet-native latent-grid mask encode for inpaint training (Phase 9, GATE-SPEC rev 2).

signet's pinned upstream SHA ``d6053703`` PREDATES the upstream flexible/mask epoch, so the canonical
``process_dataset.py`` at our pin CANNOT emit ``video_masks/`` — this module is the signet-native
replacement (the logged canonical-encoder-doctrine exception; the pin itself NEVER changes). It turns a
per-sample mask VIDEO (or per-frame PNG dir, or a single tiled image) into the latent-grid binary mask
the ``InpaintStrategy`` consumes:

    float32 ``[F_lat, H_lat, W_lat]`` with values in {0., 1.}, saved as a BARE tensor, one ``.pt`` per
    sample under ``video_masks/`` mirroring the ``latents/`` rel-path layout.

    H_lat = H // 32, W_lat = W // 32, F_lat = (F - 1) // 8 + 1 (causal temporal).

Mask semantics ON THE ENCODED TENSOR (GATE-SPEC rev 2): ``1.0`` = KEEP/context (clean latent, timestep
0, excluded from loss); ``0.0`` = GENERATE (noised, in loss). Source-video polarity: the region to
GENERATE is BLACK(0) in the mask mp4 — the ffmpeg ``negate`` render (:func:`render_mask_video`,
[precedent] prior-project ``_render_inv_mask``) produces exactly that from white-on-black SAM PNG masks.

Semantic oracle — [canonical] ``Lightricks/LTX-2`` @ ``780984275fd47128b02bef9b5c085404276866ee``
(gh-fetched read-only 2026-07-14; ``packages/ltx-trainer/scripts/process_videos.py::compute_video_masks``
lines 905-988 + constants lines 60-61). Verified there and MIRRORED here:

  * ``VAE_SPATIAL_FACTOR = 32`` / ``VAE_TEMPORAL_FACTOR = 8`` (process_videos.py:60-61).
  * target grid read from the SAVED LATENT metadata (``num_frames``/``height``/``width`` are LATENT
    dims; pixel dims derived as ``h*32``, ``w*32``, ``(f-1)*8+1`` — process_videos.py:949-954).
  * grayscale = channel MEAN (``frames.mean(dim=1)`` — process_videos.py:963).
  * spatial-mismatch guard: nearest-resize to the pixel grid (process_videos.py:964-966).
  * spatial downsample = area/mean pooling ``avg_pool2d(kernel_size=32)`` (process_videos.py:970).
  * single-image masks tile across all pixel frames (process_videos.py:957-960).
  * binarize ``(x > 0.5).float()`` ONCE, after all pooling (process_videos.py:982).
  * atomic ``.pt`` save + rel-path naming by the main media column (process_videos.py:937-938, 985).

DELIBERATE deviations from that oracle (both fixed by the shared integration contract in
``.planning/harness/GATE-SPEC-inpaint-a2v.md`` rev 2 — the binding spec every inpaint agent codes to):

  1. TEMPORAL GROUPING IS CAUSAL, AGGREGATED BY MEAN. The oracle pads F to a multiple of 8 and
     ``amax``-pools plain groups ``{0..7}, {8..15}, ...`` (process_videos.py:973-979 — NON-causal).
     signet groups CAUSALLY — frame 0 alone -> latent frame 0, then each consecutive 8 -> one latent
     frame — matching the causal VAE's actual frame->latent mapping at our pin (F = 8k+1 encodes to
     k+1 latent frames with frame 0 owning latent frame 0), and aggregates each group by MEAN before
     the single 0.5 threshold (contract: "F_lat=(F-1)//8+1 (causal temporal)", "thresholded at 0.5 at
     ENCODE time"). Both schemes yield F_lat=(F-1)//8+1; only group membership + aggregation differ.
  2. The saved payload is a BARE float32 tensor, not the oracle's ``{"mask": tensor}`` dict — the
     contract fixes "tensor float32 [F_lat, H_lat, W_lat]" so ``PrecomputedDataset.__getitem__``
     returns the tensor directly (``conditioning/inpaint.py::_extract_mask`` accepts both forms).

Import confinement (the ``data/seg_palette.py`` discipline): module top imports torch + numpy + stdlib
ONLY. Video/image decode backends (``av`` -> ``cv2`` -> ``imageio``; ``PIL`` -> ``cv2`` -> ``imageio``)
and the ffmpeg subprocess are FUNCTION-LOCAL — importing this module never pulls a decode stack, modal,
ltx_core, or ltx_trainer, so it stays CPU/CI-cheap on Windows (pure CPU, no VAE, no GPU dollar).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from glob import glob as _stdlib_glob
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "VAE_SPATIAL_FACTOR",
    "VAE_TEMPORAL_FACTOR",
    "latent_grid_for_pixels",
    "encode_mask_pixels",
    "read_mask_frames",
    "encode_mask_dataset",
    "render_mask_video",
]

#: [canonical] Lightricks/LTX-2 @ 7809842 process_videos.py:60 — pixel->latent spatial factor.
VAE_SPATIAL_FACTOR = 32
#: [canonical] Lightricks/LTX-2 @ 7809842 process_videos.py:61 — pixel->latent temporal factor.
VAE_TEMPORAL_FACTOR = 8

#: [canonical] Lightricks/LTX-2 @ 7809842 process_videos.py:900-902 — extensions that select the
#: single-image (tile-across-frames) branch vs the video decode branch.
IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def latent_grid_for_pixels(frames: int, height: int, width: int) -> tuple[int, int, int]:
    """Pixel dims -> the (F_lat, H_lat, W_lat) latent grid (contract: //32 spatial, causal-8 temporal).

    ``F_lat = (F - 1) // 8 + 1`` (causal: frame 0 alone, then 8s), ``H_lat = H // 32``,
    ``W_lat = W // 32``. Pure arithmetic — the CPU alignment gate's single source of truth.
    """
    return (
        (frames - 1) // VAE_TEMPORAL_FACTOR + 1,
        height // VAE_SPATIAL_FACTOR,
        width // VAE_SPATIAL_FACTOR,
    )


def encode_mask_pixels(
    mask_pixels: Tensor,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> Tensor:
    """Encode a pixel-space mask clip to the latent-grid binary mask (pure CPU, pure torch).

    Pipeline (each step's provenance in the module docstring):

        1. validate/truncate to ``pixel_f = (latent_frames - 1) * 8 + 1`` frames (fail loud when the
           clip is SHORTER — a short mask would silently misalign the whole run).
        2. nearest-resize to ``(latent_height * 32, latent_width * 32)`` when the spatial dims
           mismatch ([canonical] mirror — normally a no-op: the mask mp4 is rendered 1:1 to the target).
        3. spatial area/mean pool: ``avg_pool2d(kernel_size=32)`` ([canonical] mirror).
        4. CAUSAL temporal grouping, MEAN aggregate: frame 0 -> latent frame 0; frames
           ``[8(k-1)+1 .. 8k]`` -> latent frame k (contract; deviation #1 in the module docstring).
        5. single threshold ``(x > 0.5).float()`` ([canonical] mirror — white/1.0 stays 1.0 = KEEP).

    Args:
        mask_pixels: ``[F, H, W]`` float tensor in ``[0, 1]`` (grayscale mask frames).
        latent_frames: target LATENT frame count (from the sample's saved latent metadata).
        latent_height: target LATENT height (== pixel H // 32).
        latent_width: target LATENT width (== pixel W // 32).

    Returns:
        float32 ``[latent_frames, latent_height, latent_width]`` tensor with values in {0., 1.}.
    """
    if mask_pixels.dim() != 3:
        raise ValueError(f"mask_pixels must be [F, H, W], got shape {tuple(mask_pixels.shape)}")
    pixel_f = (latent_frames - 1) * VAE_TEMPORAL_FACTOR + 1
    pixel_h = latent_height * VAE_SPATIAL_FACTOR
    pixel_w = latent_width * VAE_SPATIAL_FACTOR

    if mask_pixels.shape[0] < pixel_f:
        raise ValueError(
            f"mask clip has {mask_pixels.shape[0]} frames but the target needs pixel_f={pixel_f} "
            f"((latent_frames={latent_frames} - 1) * {VAE_TEMPORAL_FACTOR} + 1) — a short mask "
            "would silently misalign every latent frame after it. Re-render the mask 1:1 to the "
            "target clip (same F/H/W, F % 8 == 1)."
        )
    mask_pixels = mask_pixels[:pixel_f].to(torch.float32)  # [canonical] truncate mirror (:962)

    # [canonical] spatial-mismatch guard (:964-966): nearest-resize to the pixel grid. Normally a
    # no-op — the polarity-rendered mask mp4 is aligned 1:1 to the target (same F/H/W).
    if mask_pixels.shape[1] != pixel_h or mask_pixels.shape[2] != pixel_w:
        mask_pixels = torch.nn.functional.interpolate(
            mask_pixels.unsqueeze(1), size=(pixel_h, pixel_w), mode="nearest"
        ).squeeze(1)

    # [canonical] spatial area/mean pooling (:970): avg_pool2d over 32x32 pixel blocks.
    pooled = torch.nn.functional.avg_pool2d(
        mask_pixels.unsqueeze(1), kernel_size=VAE_SPATIAL_FACTOR
    ).squeeze(1)  # [pixel_f, H_lat, W_lat]

    # CAUSAL temporal grouping + MEAN aggregate (contract; deviation #1 — the oracle amax-pools
    # NON-causal groups-of-8). Frame 0 alone -> latent frame 0; then each consecutive 8 -> one
    # latent frame: latent k (k>=1) = mean(frames[8(k-1)+1 : 8k+1]).
    first = pooled[:1]  # latent frame 0 == pixel frame 0 (the causal VAE's anchor frame)
    if latent_frames > 1:
        rest = pooled[1:].reshape(
            latent_frames - 1, VAE_TEMPORAL_FACTOR, latent_height, latent_width
        ).mean(dim=1)
        grouped = torch.cat([first, rest], dim=0)
    else:
        grouped = first

    # [canonical] single binarize AFTER all pooling (:982): (x > 0.5).float(). White (1.0) stays
    # 1.0 = KEEP/context; black (0.0) stays 0.0 = GENERATE; a 0.5 tie falls to GENERATE (in loss).
    return (grouped > 0.5).to(torch.float32)


# --------------------------------------------------------------------------------------------------
# Readers — decode backends are FUNCTION-LOCAL (import confinement). Order of preference: ``av``
# (the ltx-trainer container backend, cold-path probed in modal/fns.py) -> ``cv2`` (the operator-box
# staging dep) -> ``imageio``. Grayscale is ALWAYS the channel MEAN ([canonical] :963 mirror —
# channel order (RGB vs BGR) is therefore irrelevant).
# --------------------------------------------------------------------------------------------------


def _frames_to_gray_tensor(frames: list[np.ndarray]) -> Tensor:
    """[H, W, C] / [H, W] uint8 frames -> ``[F, H, W]`` float32 in [0, 1] via channel MEAN."""
    grays = []
    for frame in frames:
        arr = np.asarray(frame, dtype=np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)  # [canonical] grayscale = channel mean (:963)
        grays.append(arr)
    return torch.from_numpy(np.stack(grays, axis=0)).to(torch.float32)


def _read_video_gray(path: Path, max_frames: int | None = None) -> Tensor:
    """Decode a mask video to ``[F, H, W]`` float32 in [0, 1] (av -> cv2 -> imageio, function-local)."""
    frames: list[np.ndarray] = []
    try:
        import av  # noqa: PLC0415 — container backend (ltx-trainer dep; same as video_utils.read_video)

        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if max_frames is not None and len(frames) >= max_frames:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
    except ImportError:
        try:
            import cv2  # noqa: PLC0415 — operator-box backend (staging scripts precedent)

            cap = cv2.VideoCapture(str(path))
            try:
                while max_frames is None or len(frames) < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)  # BGR — irrelevant under channel-mean grayscale
            finally:
                cap.release()
        except ImportError:
            try:
                import imageio  # noqa: PLC0415 — last-resort backend

                reader = imageio.get_reader(str(path))
                for frame in reader:
                    if max_frames is not None and len(frames) >= max_frames:
                        break
                    frames.append(np.asarray(frame))
            except ImportError as exc:
                raise RuntimeError(
                    "no video decode backend available for the mask encode — install one of "
                    "'av' (the container backend), 'cv2' (opencv-python), or 'imageio' "
                    "(+ffmpeg plugin)."
                ) from exc
    if not frames:
        raise ValueError(f"mask video decoded to zero frames: {path} (empty or unreadable).")
    return _frames_to_gray_tensor(frames)


def _read_image_gray(path: Path) -> Tensor:
    """Decode one image to ``[H, W]`` float32 in [0, 1] (PIL -> cv2 -> imageio, function-local)."""
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"))
    except ImportError:
        try:
            import cv2  # noqa: PLC0415

            arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError(f"cv2 could not read mask image: {path}")
        except ImportError:
            try:
                import imageio  # noqa: PLC0415

                arr = np.asarray(imageio.imread(str(path)))
            except ImportError as exc:
                raise RuntimeError(
                    "no image decode backend available for the mask encode — install one of "
                    "'PIL' (pillow), 'cv2' (opencv-python), or 'imageio'."
                ) from exc
    return _frames_to_gray_tensor([arr])[0]  # [H, W]


def read_mask_frames(mask_path: str | Path, expected_frames: int | None = None) -> Tensor:
    """Read a mask source to ``[F, H, W]`` float32 grayscale in [0, 1].

    Three source forms:

      * a DIRECTORY of per-frame PNGs (sorted by filename — the SAM-propagation output layout),
      * a single IMAGE file (tiled across ``expected_frames`` — [canonical] mirror :957-960), or
      * a mask VIDEO file (the polarity-rendered ``<stem>_mask.mp4``; decode capped at
        ``expected_frames`` when given — [canonical] ``max_frames=pixel_f`` mirror :962).

    Args:
        mask_path: the mask source (dir / image / video).
        expected_frames: the target pixel frame count (``(F_lat - 1) * 8 + 1``); required for the
            single-image branch, a decode cap for the video branch, ignored for PNG dirs
            (:func:`encode_mask_pixels` truncates + fail-fasts on short clips).
    """
    path = Path(mask_path)
    if path.is_dir():
        pngs = sorted(path.glob("*.png"))
        if not pngs:
            raise ValueError(f"mask PNG dir contains no *.png frames: {path}")
        return torch.stack([_read_image_gray(p) for p in pngs], dim=0)
    if not path.exists():
        raise FileNotFoundError(f"mask source does not exist: {path}")
    if path.suffix.lower() in IMAGE_FILE_EXTENSIONS:
        if expected_frames is None:
            raise ValueError(
                f"single-image mask {path} needs expected_frames to tile across the clip "
                "([canonical] image masks tile across all pixel frames)."
            )
        img = _read_image_gray(path)  # [H, W]
        return img.unsqueeze(0).expand(expected_frames, -1, -1).clone()
    return _read_video_gray(path, max_frames=expected_frames)


# --------------------------------------------------------------------------------------------------
# Dataset-level driver — mirrors [canonical] compute_video_masks' manifest walk (:926-988) against
# signet's JSONL manifests, reading the target grid from each sample's SAVED latent metadata so the
# mask is always encoded on the exact grid the canonical encoder produced.
# --------------------------------------------------------------------------------------------------


def _atomic_save(data: object, out: Path) -> None:
    """Atomic ``torch.save`` via per-PID temp + replace ([canonical] process_videos.py:958-966)."""
    tmp = out.with_suffix(f"{out.suffix}.tmp.{os.getpid()}")
    torch.save(data, tmp)
    tmp.replace(out)


def _output_relative(path: Path) -> Path:
    """Manifest media path -> the rel path that names the sample's cached outputs.

    signet manifests carry ``media_path`` RELATIVE to the manifest's parent (``data/dataset_file.py``),
    which is already the layout the canonical encoder mirrors under ``latents/``. Absolute paths keep
    their structure minus the anchor ([canonical] ``_output_relative`` fallback :1193-1204).
    """
    return Path(*path.parts[1:]) if path.is_absolute() else path


def encode_mask_dataset(
    dataset_file: str | Path,
    mask_column: str,
    latents_dir: str | Path,
    output_dir: str | Path,
    media_column: str = "media_path",
    overwrite: bool = False,
) -> int:
    """Encode every manifest sample's mask into ``output_dir/`` mirroring the ``latents/`` layout.

    For each JSONL row: resolve the mask source (relative to the manifest's parent), read the
    target LATENT grid from ``latents_dir/<rel>.pt`` metadata (``num_frames``/``height``/``width``
    are LATENT dims — [canonical] :949-951), encode via :func:`encode_mask_pixels`, and atomically
    save the BARE float32 ``[F_lat, H_lat, W_lat]`` tensor to ``output_dir/<rel>.pt`` — the same
    rel path as the sample's latent, so ``PrecomputedDataset`` pairs the ``video_masks`` source by
    rel path with zero loader changes (the ``reference_latents`` precedent).

    Samples whose target latent is missing are SKIPPED with a warning ([canonical] :941-943 mirror
    — the canonical encoder owns which samples survived bucketing); existing outputs are skipped
    unless ``overwrite``. Pure CPU — no VAE, no GPU, no metered anything.

    Args:
        dataset_file: the JSONL manifest ({caption, media_path, <mask_column>} rows).
        mask_column: manifest column naming each sample's mask source (dir / image / mp4).
        latents_dir: the canonical encoder's ``latents/`` output dir (grid metadata source).
        output_dir: where the encoded masks land (``<root>/video_masks`` for the inpaint contract).
        media_column: the main-media column whose rel path names outputs ([canonical]
            ``main_media_column`` naming :931 — must match the latents/ layout).
        overwrite: re-encode existing outputs (default False, [canonical] mirror).

    Returns:
        The number of masks written this call.
    """
    dataset_path = Path(dataset_file)
    data_root = dataset_path.parent
    latents_path = Path(latents_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    written = 0
    skipped_no_latent = 0
    skipped_existing = 0
    for idx, row in enumerate(rows):
        if mask_column not in row:
            raise KeyError(
                f"manifest row {idx} has no {mask_column!r} column (keys: {sorted(row)}) — every "
                "inpaint sample needs a mask source; fix the manifest before encoding."
            )
        if media_column not in row:
            raise KeyError(
                f"manifest row {idx} has no {media_column!r} column (keys: {sorted(row)}) — the "
                "media column names each sample's cached outputs (must match the latents/ layout)."
            )
        rel_path = _output_relative(Path(row[media_column]))
        latent_file = latents_path / rel_path.with_suffix(".pt")
        out_file = output_path / rel_path.with_suffix(".pt")

        if not latent_file.exists():
            # [canonical] :941-943 — the canonical encoder owns which samples survived; skip + warn.
            logger.warning("No target latent at %s, skipping mask %s", latent_file, row[mask_column])
            skipped_no_latent += 1
            continue
        if not overwrite and out_file.is_file():
            # #35 finding 1 / step 2: this is the hardcoded no-op — `overwrite` defaults False and
            # (before #35 step 3) had no caller that ever passed True, so a repainted mask silently
            # never re-encoded. Counted, not just skipped, so the loud summary below can name it —
            # a bare "wrote 0" line reads as "already up to date", which is exactly the failure shape.
            skipped_existing += 1
            continue

        meta = torch.load(latent_file, map_location="cpu", weights_only=True)
        latent_f = int(meta["num_frames"])  # LATENT dims ([canonical] :949-951)
        latent_h = int(meta["height"])
        latent_w = int(meta["width"])
        pixel_f = (latent_f - 1) * VAE_TEMPORAL_FACTOR + 1

        mask_source = data_root / row[mask_column]
        mask_pixels = read_mask_frames(mask_source, expected_frames=pixel_f)
        mask = encode_mask_pixels(mask_pixels, latent_f, latent_h, latent_w)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        _atomic_save(mask, out_file)  # BARE tensor — the GATE-SPEC contract payload (deviation #2)
        written += 1

    # #35 step 2 — make the skip LOUD (stdout): the package configures no logging handler (same gap
    # #21 names for PrecomputedDataset), so `logger.info`/`logger.warning` alone never reach a Modal
    # log. `print()` is what the Modal stages already use for anything an operator must see.
    print(
        f"[encode_mask_dataset] {written} mask(s) written, {skipped_existing} skipped (existing, "
        f"overwrite={overwrite}) — pass overwrite=True to re-encode, "
        f"{skipped_no_latent} skipped (no target latent) -> {output_path}."
    )
    if skipped_existing > 0 and not overwrite:
        logger.warning(
            "Mask encode: %d mask(s) written, %d skipped as already-existing (overwrite=False) — "
            "pass overwrite=True to re-encode a repainted mask.",
            written,
            skipped_existing,
        )
    else:
        logger.info("Mask encode complete: %d masks written to %s", written, output_path)
    return written


# --------------------------------------------------------------------------------------------------
# Polarity render — [precedent] prior-project ``_render_inv_mask``
# (``<prior-project>/_train/modal/`` Modal trainer :245-249): SAM per-frame PNGs (region=WHITE)
# -> mp4 INVERTED via ffmpeg ``negate`` (region=BLACK=0=GENERATE, context=WHITE=1=KEEP).
# --------------------------------------------------------------------------------------------------


def render_mask_video(png_source: str | Path, dst: str | Path, fps: int = 24) -> Path:
    """Render per-frame mask PNGs to the INVERTED mask mp4 (region=BLACK=generate, precedent polarity).

    Port of the prior project's ``_render_inv_mask`` [precedent]: ``ffmpeg ... -vf negate -pix_fmt yuv420p``.
    The precedent fed ffmpeg a ``-pattern_type glob`` input, which Windows ffmpeg builds lack — behaviorally
    identical here: the sorted frame list is staged to a sequentially-numbered temp dir and fed via
    the portable ``%06d`` sequence pattern (same frames, same order, same filters).

    Args:
        png_source: a directory of per-frame PNGs (sorted by filename) or a glob pattern.
        dst: output mp4 path (parent dirs created).
        fps: output frame rate ([precedent] prior-project default 24).

    Returns:
        The written mp4 ``Path``.
    """
    src = Path(png_source)
    if src.is_dir():
        frames = sorted(src.glob("*.png"))
    else:
        frames = sorted(Path(p) for p in _stdlib_glob(str(png_source)))
    if not frames:
        raise ValueError(f"no PNG frames matched mask source {png_source!r} — nothing to render.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH — the polarity render needs it (region=BLACK=generate is "
            "produced by ffmpeg's negate filter; precedent convention)."
        )

    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        for i, frame in enumerate(frames):
            shutil.copy(frame, Path(staging) / f"frame_{i:06d}.png")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(Path(staging) / "frame_%06d.png"),
                "-vf",
                "negate",  # [precedent]: face/hair=WHITE PNGs -> BLACK(0)=generate in the mp4
                "-pix_fmt",
                "yuv420p",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
    return out
