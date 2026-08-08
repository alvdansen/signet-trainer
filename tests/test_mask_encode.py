"""Behavior tests for the signet-native inpaint mask encode (Phase 9, GATE-SPEC rev 2).

Covers ``data/mask_encode.py`` (the latent-grid mask encode replacing upstream's
``compute_video_masks`` — impossible at signet's pinned SHA), the ``video_masks``
pass-through-un-normalized regression on ``PrecomputedDataset``, and the ``modal/fns.py``
preprocess-arm wiring (source-text scan, mirroring ``test_preprocess_wiring.py`` — never imports
the ``modal``-decorated module).

All CPU, zero Modal spend. The known-pattern test is designed to DISCRIMINATE causal grouping
(frame 0 alone -> latent frame 0) from the upstream oracle's non-causal groups-of-8 — a port that
silently regressed to upstream's grouping would fail it.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

# ----------------------------------------------------------------------------------------------
# The known 17-frame 64x128 pattern -> exact expected [3, 2, 4] latent-grid mask.
#   F=17, H=64, W=128  =>  F_lat=(17-1)//8+1=3, H_lat=64//32=2, W_lat=128//32=4.
#   frame 0        : LEFT half white              -> latent frame 0 = [[1,1,0,0],[1,1,0,0]]
#   frames 1..8    : TOP half white               -> latent frame 1 = [[1,1,1,1],[0,0,0,0]]
#   frames 9..16   : top half white in 4/8 frames (mean 0.5, NOT > 0.5 -> 0);
#                    bottom half white in 5/8 frames (mean 0.625 > 0.5 -> 1)
#                                                 -> latent frame 2 = [[0,0,0,0],[1,1,1,1]]
# Discrimination: under upstream's NON-causal {0..7} grouping, group 0 would mix frame 0's
# left-half with frames 1-7's top-half (mean >= 0.875 on the whole top row -> [[1,1,1,1],...]),
# which differs from the causal latent frame 0 above. The 4/8-vs-5/8 split also kills an
# amax-instead-of-mean regression (amax would give 1 everywhere either half was ever white).
# ----------------------------------------------------------------------------------------------


def _known_pattern() -> torch.Tensor:
    """The 17x64x128 float mask clip described above (values in {0., 1.})."""
    clip = torch.zeros(17, 64, 128, dtype=torch.float32)
    clip[0, :, :64] = 1.0  # frame 0: left half
    clip[1:9, :32, :] = 1.0  # frames 1..8: top half
    clip[9:13, :32, :] = 1.0  # frames 9..12: top half white in 4/8 -> mean 0.5 -> 0
    clip[9:14, 32:, :] = 1.0  # frames 9..13: bottom half white in 5/8 -> mean 0.625 -> 1
    return clip


_EXPECTED = torch.tensor(
    [
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
    ],
    dtype=torch.float32,
)


def test_import_confinement_no_heavy_or_decode_modules() -> None:
    """Module import pulls NO modal/ltx/decode stack (the seg_palette.py discipline).

    Snapshot before popping so we can RESTORE afterwards, matching
    ``test_grid_html.py:80-92``. Popping without restoring poisons every LATER test in the
    session, and ``PIL`` is the sharpest case: evicting the ``"PIL"`` key alone leaves
    ``sys.modules["PIL.Image"]`` cached, so a subsequent ``import PIL.Image`` re-executes
    ``PIL/__init__`` into a NEW module object while the submodule import short-circuits — the
    ``Image`` attribute is never rebound. Anything later that evaluates the annotation
    ``list[PIL.Image.Image]`` at import time then dies with ``AttributeError: module 'PIL' has
    no attribute 'Image'`` — which is what ``diffusers/utils/export_utils.py:27`` does, so
    ``pytest.importorskip("diffusers")`` FAILS instead of skipping (importorskip catches
    ImportError, not AttributeError). Restoring keeps the confinement assertion exactly as
    strict while making it hermetic.
    """
    forbidden = ("modal", "ltx_core", "ltx_trainer", "av", "cv2", "PIL", "imageio")
    saved = {mod: sys.modules.pop(mod, None) for mod in forbidden}
    try:
        import signet_trainer.data.mask_encode  # noqa: F401

        for mod in forbidden:
            assert mod not in sys.modules, (
                f"import confinement violation: mask_encode imported {mod}"
            )
    finally:
        for mod, value in saved.items():
            if value is not None:
                sys.modules[mod] = value


def test_latent_grid_for_pixels_contract_dims() -> None:
    """(F-1)//8+1 causal temporal, //32 spatial — the contract grid arithmetic."""
    from signet_trainer.data.mask_encode import latent_grid_for_pixels

    assert latent_grid_for_pixels(17, 64, 128) == (3, 2, 4)
    assert latent_grid_for_pixels(1, 32, 32) == (1, 1, 1)
    assert latent_grid_for_pixels(81, 704, 1280) == (11, 22, 40)  # the production inpaint res family


def test_encode_known_pattern_exact_causal_grouping_and_threshold() -> None:
    """The load-bearing exactness test: causal grouping + MEAN aggregate + strict >0.5 threshold."""
    from signet_trainer.data.mask_encode import encode_mask_pixels

    out = encode_mask_pixels(_known_pattern(), latent_frames=3, latent_height=2, latent_width=4)

    assert out.shape == (3, 2, 4)
    assert out.dtype == torch.float32
    assert torch.equal(out, _EXPECTED), f"encoded mask diverged:\n{out}\nexpected:\n{_EXPECTED}"


def test_encode_dims_and_binary_values_random_input() -> None:
    """17f 64x128 random input -> [3, 2, 4] float32 with values only in {0., 1.}."""
    from signet_trainer.data.mask_encode import encode_mask_pixels

    rng = torch.Generator().manual_seed(42)
    clip = torch.rand(17, 64, 128, generator=rng)
    out = encode_mask_pixels(clip, latent_frames=3, latent_height=2, latent_width=4)

    assert out.shape == (3, 2, 4)
    assert out.dtype == torch.float32
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_polarity_white_stays_one_black_stays_zero() -> None:
    """Contract polarity on the ENCODED tensor: white(1)=KEEP stays 1; black(0)=GENERATE stays 0."""
    from signet_trainer.data.mask_encode import encode_mask_pixels

    white = encode_mask_pixels(torch.ones(17, 64, 128), 3, 2, 4)
    black = encode_mask_pixels(torch.zeros(17, 64, 128), 3, 2, 4)
    assert torch.equal(white, torch.ones(3, 2, 4))
    assert torch.equal(black, torch.zeros(3, 2, 4))


def test_encode_truncates_long_clips_and_rejects_short_ones() -> None:
    """Extra frames are truncated at pixel_f ([canonical] mirror); short clips fail LOUD."""
    from signet_trainer.data.mask_encode import encode_mask_pixels

    # 20 frames for a 3-latent-frame target (pixel_f=17): frames 17..19 must be ignored.
    clip = torch.cat([_known_pattern(), torch.ones(3, 64, 128)], dim=0)
    out = encode_mask_pixels(clip, 3, 2, 4)
    assert torch.equal(out, _EXPECTED)

    # 16 frames < pixel_f=17: a short mask would silently misalign — must raise.
    with pytest.raises(ValueError, match="16 frames.*pixel_f=17"):
        encode_mask_pixels(torch.ones(16, 64, 128), 3, 2, 4)


def test_encode_resizes_mismatched_spatial_dims() -> None:
    """A mask at the wrong resolution nearest-resizes to the pixel grid ([canonical] mirror)."""
    from signet_trainer.data.mask_encode import encode_mask_pixels

    # Half-res input (32x64) for a (64,128)-pixel target: left half white -> [[1,1,0,0],...] x3.
    clip = torch.zeros(17, 32, 64)
    clip[:, :, :32] = 1.0
    out = encode_mask_pixels(clip, 3, 2, 4)
    expected = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]).expand(3, 2, 4)
    assert torch.equal(out, expected)


def _write_pattern_pngs(dst: Path) -> None:
    """Write the known pattern as per-frame PNGs (lossless — exact assertions survive)."""
    PIL = pytest.importorskip("PIL")  # noqa: F841 — decode backend for the PNG-dir path
    from PIL import Image

    dst.mkdir(parents=True, exist_ok=True)
    clip = (_known_pattern().numpy() * 255).astype("uint8")
    for i in range(clip.shape[0]):
        Image.fromarray(clip[i]).save(dst / f"frame_{i:04d}.png")


def test_png_dir_roundtrip_exact(tmp_path: Path) -> None:
    """PNG-dir mask source -> read_mask_frames -> encode == the exact expected tensor."""
    from signet_trainer.data.mask_encode import encode_mask_pixels, read_mask_frames

    png_dir = tmp_path / "mask_frames"
    _write_pattern_pngs(png_dir)

    frames = read_mask_frames(png_dir)
    assert frames.shape == (17, 64, 128)
    out = encode_mask_pixels(frames, 3, 2, 4)
    assert torch.equal(out, _EXPECTED)


def _write_latent_meta(path: Path, num_frames: int, height: int, width: int) -> None:
    """A latents/<rel>.pt with the pinned-SHA metadata contract (LATENT dims)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.randn(128, num_frames, height, width),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": 24.0,
        },
        path,
    )


def test_encode_mask_dataset_end_to_end(tmp_path: Path) -> None:
    """Manifest -> encode_mask_dataset writes the exact BARE tensor mirroring latents/ rel paths."""
    import json

    from signet_trainer.data.mask_encode import encode_mask_dataset

    root = tmp_path
    _write_pattern_pngs(root / "masks" / "clipA")
    _write_latent_meta(root / ".precomputed" / "latents" / "clips" / "a.pt", 3, 2, 4)

    manifest = root / "metadata.jsonl"
    rows = [
        # Sample with a latent -> encoded.
        {"caption": "x", "media_path": "clips/a.mp4", "video_mask": "masks/clipA"},
        # Sample WITHOUT a latent (didn't survive bucketing) -> skipped with a warning, not fatal.
        {"caption": "y", "media_path": "clips/missing.mp4", "video_mask": "masks/clipA"},
    ]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    written = encode_mask_dataset(
        dataset_file=manifest,
        mask_column="video_mask",
        latents_dir=root / ".precomputed" / "latents",
        output_dir=root / ".precomputed" / "video_masks",
        media_column="media_path",
    )
    assert written == 1

    out_file = root / ".precomputed" / "video_masks" / "clips" / "a.pt"
    assert out_file.is_file(), "mask must mirror the latents/ rel-path layout"
    payload = torch.load(out_file, map_location="cpu", weights_only=True)
    assert isinstance(payload, torch.Tensor), "contract payload is a BARE tensor (not a dict)"
    assert payload.dtype == torch.float32
    assert torch.equal(payload, _EXPECTED)

    # Idempotent skip: a second run rewrites nothing (overwrite=False default).
    assert (
        encode_mask_dataset(
            dataset_file=manifest,
            mask_column="video_mask",
            latents_dir=root / ".precomputed" / "latents",
            output_dir=root / ".precomputed" / "video_masks",
        )
        == 0
    )


def test_encode_mask_dataset_missing_column_fails_loud(tmp_path: Path) -> None:
    """A manifest row without the mask column raises with an actionable message."""
    import json

    from signet_trainer.data.mask_encode import encode_mask_dataset

    _write_latent_meta(tmp_path / "lat" / "a.pt", 3, 2, 4)
    manifest = tmp_path / "metadata.jsonl"
    manifest.write_text(json.dumps({"caption": "x", "media_path": "a.mp4"}) + "\n", encoding="utf-8")

    with pytest.raises(KeyError, match="video_mask"):
        encode_mask_dataset(
            dataset_file=manifest,
            mask_column="video_mask",
            latents_dir=tmp_path / "lat",
            output_dir=tmp_path / "out",
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_render_mask_video_negate_polarity_roundtrip(tmp_path: Path) -> None:
    """[precedent] prior-project _render_inv_mask port: WHITE SAM PNGs -> negate -> BLACK(0)=GENERATE mp4.

    Also exercises the video-decode branch of ``read_mask_frames`` (av -> cv2 -> imageio): the
    rendered mp4 reads back as (near-)black and encodes to the all-zero GENERATE mask.
    """
    from PIL import Image

    from signet_trainer.data.mask_encode import (
        encode_mask_pixels,
        read_mask_frames,
        render_mask_video,
    )

    png_dir = tmp_path / "sam_masks"
    png_dir.mkdir()
    for i in range(17):
        Image.fromarray(np.full((64, 128), 255, np.uint8)).save(png_dir / f"m_{i:03d}.png")

    dst = render_mask_video(png_dir, tmp_path / "clip_mask.mp4", fps=24)
    assert dst.is_file()

    frames = read_mask_frames(dst, expected_frames=17)
    assert frames.shape[0] == 17
    assert float(frames.max()) < 0.1, "negate must flip white SAM regions to (near-)black"
    out = encode_mask_pixels(frames, 3, 2, 4)
    assert torch.equal(out, torch.zeros(3, 2, 4)), "region-to-generate must encode to 0.0"


# ----------------------------------------------------------------------------------------------
# PrecomputedDataset regression: the video_masks source pairs by rel path and passes through
# UN-normalized (the dir name contains no "latent" substring — the a2v audio_latents trip-wire
# class, GATE-SPEC rev 2 top-risks).
# ----------------------------------------------------------------------------------------------

_INPAINT_3_SOURCE = {
    "latents": "latent_conditions",
    "conditions": "text_conditions",
    "video_masks": "video_mask_conditions",
}


def _write_condition(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video_prompt_embeds": torch.randn(4, 4096),
            "prompt_attention_mask": torch.ones(4, dtype=torch.bool),
        },
        path,
    )


def test_video_masks_source_passes_through_unnormalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """video_masks pairs by rel path; its bare tensor is returned UN-normalized, byte-identical."""
    from signet_trainer.data.precomputed import PrecomputedDataset

    # The structural guard the whole registration rests on: no "latent" substring in the dir name.
    assert "latent" not in "video_masks".lower()

    root = tmp_path / ".precomputed"
    _write_latent_meta(root / "latents" / "a.pt", 3, 2, 4)
    _write_condition(root / "conditions" / "a.pt")
    (root / "video_masks").mkdir(parents=True)
    torch.save(_EXPECTED.clone(), root / "video_masks" / "a.pt")  # BARE tensor payload

    # Spy on the normalizer to prove the mask source is NEVER routed through it.
    normalized_calls: list[tuple[int, ...]] = []
    orig = PrecomputedDataset._normalize_video_latents

    def _spy(data: dict) -> dict:
        normalized_calls.append(tuple(data["latents"].shape))
        return orig(data)

    monkeypatch.setattr(PrecomputedDataset, "_normalize_video_latents", staticmethod(_spy))

    ds = PrecomputedDataset(str(root), data_sources=_INPAINT_3_SOURCE)
    assert len(ds) == 1
    sample = ds[0]

    assert "video_mask_conditions" in sample
    mask = sample["video_mask_conditions"]
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.float32
    assert torch.equal(mask, _EXPECTED), "mask must pass through UN-normalized, byte-identical"

    # The normalizer ran ONLY for the "latents" source — never for video_masks (whose bare-tensor
    # payload would crash the dict-shaped normalizer if the branch ever fired).
    assert normalized_calls == [(128, 3, 2, 4)], normalized_calls


def test_video_masks_unpaired_sample_drops_out(tmp_path: Path) -> None:
    """A latent without its video_masks twin is excluded (rel-path pairing, reference_latents rule)."""
    from signet_trainer.data.precomputed import PrecomputedDataset

    root = tmp_path / ".precomputed"
    _write_latent_meta(root / "latents" / "a.pt", 3, 2, 4)
    _write_latent_meta(root / "latents" / "b.pt", 3, 2, 4)
    _write_condition(root / "conditions" / "a.pt")
    _write_condition(root / "conditions" / "b.pt")
    (root / "video_masks").mkdir(parents=True)
    torch.save(_EXPECTED.clone(), root / "video_masks" / "a.pt")  # b.pt has NO mask

    ds = PrecomputedDataset(str(root), data_sources=_INPAINT_3_SOURCE)
    assert len(ds) == 1  # only the fully-paired sample survives


# ----------------------------------------------------------------------------------------------
# modal/fns.py preprocess-arm wiring — source-text scan (mirrors test_preprocess_wiring.py; the
# modal-decorated module is never imported).
# ----------------------------------------------------------------------------------------------

_FNS = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def test_fns_preprocess_mask_arm_wiring() -> None:
    """The preprocess arm carries the mask params, calls the signet-native encode BEFORE commit."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))

    # Backward-compatible signature: mask_column defaults None (non-inpaint runs byte-identical);
    # the output dir name is a param (config-driven from conditioning.inpaint_mask_dir upstream).
    assert re.search(r"mask_column:\s*str\s*\|\s*None\s*=\s*None", code), (
        "preprocess must accept mask_column: str | None = None (default = no mask encode)"
    )
    assert re.search(r'mask_output_dir_name:\s*str\s*=\s*["\']video_masks["\']', code), (
        "preprocess must accept mask_output_dir_name defaulting to 'video_masks' (contract dir)"
    )

    # The arm is gated on mask_column and calls the signet-native encoder with threaded params.
    assert re.search(r"if\s+mask_column\s+is\s+not\s+None\s*:", code), (
        "the mask encode must be gated behind `if mask_column is not None:`"
    )
    assert "encode_mask_dataset" in code, "preprocess must call data.mask_encode.encode_mask_dataset"
    assert re.search(r"mask_column\s*=\s*mask_column", code), "mask_column must thread through"
    assert re.search(r"output_dir\s*\)\s*/\s*mask_output_dir_name", code), (
        "the mask output dir must be <output_dir>/<mask_output_dir_name> (config-driven, no hardcode)"
    )

    # Ordering: the mask encode rides the SAME dataset_vol.commit() as the canonical encode —
    # encode first, commit after (commit-or-vanish).
    assert code.index("encode_mask_dataset") < code.index("dataset_vol.commit()"), (
        "the mask encode must run BEFORE dataset_vol.commit() so masks ride the same commit"
    )


def test_fns_registers_video_masks_source_output_key() -> None:
    """_PRECOMPUTED_SOURCE_OUTPUT_KEYS maps video_masks -> video_mask_conditions (inpaint contract)."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    assert re.search(r'["\']video_masks["\']\s*:\s*["\']video_mask_conditions["\']', code), (
        "_PRECOMPUTED_SOURCE_OUTPUT_KEYS must register 'video_masks': 'video_mask_conditions'"
    )
