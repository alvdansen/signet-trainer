"""prep.stage polarity-chain proof (D-05) + staging preflight (D-13). Pure CPU — no modal, no GPU.

Extends the encode-side polarity fixture (``tests/test_inpaint_strategy.py`` proves the
``>0.5 -> 1.0 = KEEP`` law on the strategy) by asserting the FULL D-05 stage chain end to end:

    paint region = WHITE(255)  ->  render_mask_video (polarity render)  ->  mask mp4 region = BLACK(0)
    ->  encode_mask_pixels (>0.5)  ->  tensor 1.0 = KEEP / 0.0 = GENERATE  ->  InpaintStrategy

plus the staging dims preflight (÷64 spatial hard-raise; 8n+1 frames auto-trimmed, never padded) and
the thin-CLI hygiene (no Modal, dry-run, completion marker). The ffmpeg leg skips gracefully when
ffmpeg is unavailable; the encode-polarity + preflight legs assert UNCONDITIONALLY.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from signet_trainer.data.mask_encode import (
    encode_mask_pixels,
    read_mask_frames,
    render_mask_video,
)
from signet_trainer.prep.stage import assert_stage_dims, largest_8n1

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "prep_inpaint_stage.py"

# Div64-clean tiny grid: W=64,H=64,F=9 -> latent [F_lat=2, H_lat=2, W_lat=2]. The LEFT half
# (cols 0-31 -> latent col 0) is the REGION to generate; the RIGHT half (cols 32-63 -> latent col 1)
# is context.
_W, _H, _F = 64, 64, 9
_F_LAT, _H_LAT, _W_LAT = 2, 2, 2
_SPEC = {"dims": {"spatial_divisor": 64, "frame_rule": "8n+1"}}


def _write_white_region_pngs(dst: Path) -> None:
    """Write F PNGs: LEFT half WHITE(255) = region-to-generate, RIGHT half BLACK(0) = context."""
    import numpy as _np  # local, mirrors decode-backend confinement
    from PIL import Image

    dst.mkdir(parents=True, exist_ok=True)
    for i in range(_F):
        arr = _np.zeros((_H, _W), dtype=_np.uint8)
        arr[:, : _W // 2] = 255  # paint the region WHITE (pre-negate)
        Image.fromarray(arr).save(dst / f"{i:06d}.png")


# --------------------------------------------------------------------------------------------------
# D-05 polarity chain: WHITE region PNGs -> render_mask_video -> decoded region is BLACK
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required for the polarity render leg")
def test_polarity_chain_white_region_renders_to_black(tmp_path: Path) -> None:
    pngs = tmp_path / "frames"
    _write_white_region_pngs(pngs)
    mp4 = tmp_path / "mask.mp4"
    render_mask_video(pngs, mp4, fps=24)

    # Decode: the painted (LEFT) region must be BLACK(~0), the context (RIGHT) WHITE(~1) — the
    # polarity render inverted the source. A skipped/flipped render fails this (RESEARCH Pitfall 4).
    frames = read_mask_frames(mp4, expected_frames=_F)  # [F, H, W] in [0, 1]
    region_mean = float(frames[:, :, : _W // 2].mean())
    context_mean = float(frames[:, :, _W // 2 :].mean())
    assert region_mean < 0.2, f"region should be BLACK after the polarity render, got {region_mean:.3f}"
    assert context_mean > 0.8, f"context should be WHITE after the polarity render, got {context_mean:.3f}"

    # Encode -> latent-grid tensor: region 0.0 = GENERATE, context 1.0 = KEEP (D-05 encode step).
    mask = encode_mask_pixels(frames, _F_LAT, _H_LAT, _W_LAT)  # [2, 2, 2]
    assert torch.all(mask[:, :, 0] == 0.0), "region (latent col 0) must encode to 0.0 = GENERATE"
    assert torch.all(mask[:, :, 1] == 1.0), "context (latent col 1) must encode to 1.0 = KEEP"


# --------------------------------------------------------------------------------------------------
# encode-side polarity (unconditional — no ffmpeg): a negated-mask tensor -> 1.0=KEEP / 0.0=GENERATE,
# threaded through the REAL InpaintStrategy (extends test_inpaint_strategy's law)
# --------------------------------------------------------------------------------------------------


def _synthetic_negated_frames() -> torch.Tensor:
    """The post-negate pixel mask: region (LEFT) BLACK(0), context (RIGHT) WHITE(1) — [F, H, W]."""
    frames = torch.zeros(_F, _H, _W, dtype=torch.float32)
    frames[:, :, _W // 2 :] = 1.0  # context WHITE
    return frames


def test_encode_yields_keep_context_generate_region() -> None:
    mask = encode_mask_pixels(_synthetic_negated_frames(), _F_LAT, _H_LAT, _W_LAT)
    assert mask.shape == (_F_LAT, _H_LAT, _W_LAT)
    assert torch.all(mask[:, :, 0] == 0.0)  # GENERATE
    assert torch.all(mask[:, :, 1] == 1.0)  # KEEP


def _make_stub_deps():
    """Compact copy of the test_inpaint_strategy ltx_core stub seam (frame-major identity patchify)."""
    from signet_trainer.train.step import StepDeps

    def _identity_patchify(latent_5d: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = latent_5d.shape
        return latent_5d.reshape(b, c, f * h * w).permute(0, 2, 1).contiguous()

    class _RecordingShape:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.seq = kwargs["frames"] * kwargs["height"] * kwargs["width"]

    class _StubPatchifier:
        def patchify(self, latent_5d: torch.Tensor) -> torch.Tensor:
            return _identity_patchify(latent_5d)

        def get_patch_grid_bounds(self, output_shape, device):  # noqa: ANN001
            return output_shape

    def _get_pixel_coords(latent_coords, scale_factors, causal_fix):  # noqa: ANN001
        return torch.zeros(1, 3, latent_coords.seq, 2)

    return StepDeps(
        patchifier=_StubPatchifier(),
        get_pixel_coords=_get_pixel_coords,
        scale_factors=None,
        video_latent_shape_cls=_RecordingShape,
        modality_cls=lambda **kw: SimpleNamespace(**kw),
    )


class _FixedSchedule:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def sample_timesteps(self, batch_size, seq_len, rng):  # noqa: ANN001
        return np.full(batch_size, self.sigma, dtype=np.float64)


def test_encoded_mask_drives_strategy_keep_generate() -> None:
    """The encoded mask, fed to the REAL InpaintStrategy, keeps context tokens + generates region."""
    from signet_trainer.conditioning.inpaint import InpaintStrategy

    torch.manual_seed(0)
    mask = encode_mask_pixels(_synthetic_negated_frames(), _F_LAT, _H_LAT, _W_LAT)  # [2,2,2]
    latents = torch.randn(2, _F_LAT, _H_LAT, _W_LAT)  # [C, F, H, W] -> seq_len 8, frame-major

    batch = {
        "latent_conditions": {"latents": latents},
        "text_conditions": {
            "video_prompt_embeds": torch.randn(4, 8),
            "prompt_attention_mask": torch.ones(4, dtype=torch.long),
        },
        "idx": 0,
        "video_mask_conditions": mask,
    }
    strat = InpaintStrategy(
        deps=_make_stub_deps(),
        schedule=_FixedSchedule(0.5),
        inpaint_mask_probability=1.0,
        device="cpu",
        dtype=torch.float32,
    )
    inputs = strat.prepare_training_inputs(batch, rng=np.random.default_rng(0))

    # KEEP tokens (timestep 0, out of loss) are exactly the context cells (latent w=1): frame-major
    # token = f*(H*W) + h*W + w = f*4 + h*2 + 1 -> {1, 3, 5, 7}.
    keep = (inputs.video.timesteps == 0).nonzero()[:, 1].tolist()
    assert keep == [1, 3, 5, 7], f"KEEP tokens {keep} != context cells {{1,3,5,7}}"
    expected = torch.zeros(1, 8, dtype=torch.bool)
    expected[:, [1, 3, 5, 7]] = True
    assert torch.equal(inputs.video_loss_mask, ~expected)  # GENERATE cells carry the loss


# --------------------------------------------------------------------------------------------------
# staging dims preflight: ÷64 spatial hard-raises; 8n+1 frames auto-trim (never pad)
# --------------------------------------------------------------------------------------------------


def test_stage_preflight_raises_non_div64() -> None:
    # 736 is %32-clean (23*32) but NOT %64 — legal for plain video, ILLEGAL for inpaint. The spatial
    # rule cannot be auto-fixed (cropping changes content), so it hard-raises at prep.
    with pytest.raises(ValueError, match=r"÷64|crop W"):
        assert_stage_dims(736, 512, _F, _SPEC, stem="bad_w")


def test_stage_preflight_surfaces_8n1_but_trims_never_pads() -> None:
    # A non-8n+1 frame count is AUTO-TRIMMED to the largest 8n+1 (house rule: trim, never pad), so a
    # ÷64-clean clip does NOT raise — it returns the trimmed target. The 8n+1 rule is still surfaced
    # by the preflight report (see test_prep_preflight); staging fixes it by trimming, never padding.
    n_target = assert_stage_dims(768, 512, 80, _SPEC)  # 80 -> largest 8n+1 <= 80 = 73
    assert n_target == largest_8n1(80) == 73
    assert (n_target - 1) % 8 == 0  # the staged clip is 8n+1 by construction
    assert n_target <= 80  # trimmed DOWN, never padded up


# --------------------------------------------------------------------------------------------------
# thin CLI hygiene: exists, no Modal, no metered dispatch, completion marker, --help
# --------------------------------------------------------------------------------------------------


def test_cli_exists_and_has_no_modal_dispatch() -> None:
    assert CLI_PATH.is_file(), f"staging CLI missing: {CLI_PATH}"
    src = CLI_PATH.read_text(encoding="utf-8")
    assert "import modal" not in src, "the staging CLI must NOT import modal (no metered dispatch)"
    assert ".remote(" not in src, "the staging CLI must NOT call .remote( (no metered dispatch)"
    assert "STAGE_DONE" in src, "the staging CLI must print a completion marker"
    assert "--dry-run" in src, "the staging CLI must expose a --dry-run self-check"


def test_cli_help_exits_zero() -> None:
    r = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONUTF8": "1"},
    )
    assert r.returncode == 0, f"--help exited {r.returncode}: {r.stderr[-400:]}"
