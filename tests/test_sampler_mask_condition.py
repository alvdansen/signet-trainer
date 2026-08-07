"""Phase 9 (INPAINT) — CPU-shape gates for the mask-condition masked-render branch.

Covers the pure/CPU layer of ``inference/sampler.py``'s Phase-9 additions (GATE-SPEC-inpaint-a2v
rev 2, item 6 — the ic_lora-pipeline-class port) WITHOUT a model, a VAE, or the ltx stack:

  * ``build_token_denoise_mask`` — per-token denoise-mask construction from a tiny synthetic
    ``[F, H, W]`` encoded mask: FRAME-MAJOR token order (proved against hand-computed
    ``f*H_lat*W_lat + h*W_lat + w`` indices AND against ``mask_ops.latent_frame_token_span``),
    polarity inversion (KEEP 1.0 -> denoise 0.0), the 0.5 threshold, and the fail-loud grid
    mismatch (the GATE-SPEC top risk: mask->token alignment + polarity — both CPU-gated here,
    zero GPU exposure).
  * ``pin_keep_tokens`` — the per-step copy-back formula equals training's
    ``where(mask > 0.5, clean, noisy)`` substitution exactly on synthetic tensors, and
    ``sigma * denoise_mask`` (via the SHARED ``mask_ops.per_token_sigma`` primitive) equals
    ``where(mask > 0.5, 0, sigma)`` — the timestep-0 semantics.
  * ``masked_render_latent_grid`` — the inpaint ÷64 dims rule (STRICTER than %32) + the causal
    latent-grid math ((F-1)//8+1, H//32, W//32).
  * ``plan_mask_condition`` — config-shape acceptance of a mask condition as a plain dict AND as
    the real ``config.schema.MaskCondition`` sub-model; fail-fast on wrong/missing fields.
  * ``_load_latent_mask`` — the ``.pt`` branch round-trips a tensor (tmp_path; no Modal, no VAE).

Import-confinement is asserted the same way ``test_gen_config.py`` does: importing the sampler
module (with the new branch) must never pull modal / ltx_core / ltx_trainer on CPU/CI.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from signet_trainer.conditioning.mask_ops import latent_frame_token_span, per_token_sigma
from signet_trainer.config.schema import MaskCondition
from signet_trainer.inference.sampler import (
    _load_latent_mask,
    build_token_denoise_mask,
    masked_render_latent_grid,
    pin_keep_tokens,
    plan_mask_condition,
)

# A tiny synthetic latent grid — small enough to hand-compute every token index.
LAT_F, LAT_H, LAT_W = 2, 3, 4
SEQ_LEN = LAT_F * LAT_H * LAT_W  # 24


def _keep_mask(keep_cells: list[tuple[int, int, int]]) -> torch.Tensor:
    """Build a [F, H, W] encoded KEEP-mask (1.0 = KEEP) with 1.0 at the given (f, h, w) cells."""
    mask = torch.zeros(LAT_F, LAT_H, LAT_W, dtype=torch.float32)
    for f, h, w in keep_cells:
        mask[f, h, w] = 1.0
    return mask


# --------------------------------------------------------------------------------------------------
# build_token_denoise_mask — frame-major token order + polarity + threshold + fail-loud grid
# --------------------------------------------------------------------------------------------------


def test_token_mask_shape_and_dtype() -> None:
    """Output is the upstream denoise-mask shape [1, seq_len, 1], float32."""
    dm = build_token_denoise_mask(_keep_mask([]), lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)
    assert dm.shape == (1, SEQ_LEN, 1)
    assert dm.dtype == torch.float32


def test_token_mask_polarity_inversion() -> None:
    """KEEP (1.0 on the encoded tensor) -> denoise 0.0; GENERATE (0.0) -> denoise 1.0."""
    mask = _keep_mask([(0, 0, 0)])
    dm = build_token_denoise_mask(mask, lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)
    assert dm[0, 0, 0].item() == 0.0  # KEEP token: pinned clean (never denoised)
    assert dm[0, 1, 0].item() == 1.0  # GENERATE token: denoised from noise


def test_token_mask_single_keep_cell_is_frame_major() -> None:
    """One keep cell at (f=1, h=2, w=3) -> token index f*H*W + h*W + w = 23 (frame-major)."""
    dm = build_token_denoise_mask(
        _keep_mask([(1, 2, 3)]), lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W
    )
    expected_token = 1 * LAT_H * LAT_W + 2 * LAT_W + 3  # 23
    zeros = (dm[0, :, 0] == 0.0).nonzero().flatten().tolist()
    assert zeros == [expected_token]


def test_token_mask_asymmetric_pattern_matches_hand_computed_indices() -> None:
    """An asymmetric keep pattern lands EXACTLY on the hand-computed token set (contract order)."""
    keep_cells = [(0, 0, 1), (0, 2, 0), (1, 0, 3), (1, 1, 2)]
    dm = build_token_denoise_mask(
        _keep_mask(keep_cells), lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W
    )
    expected = sorted(f * LAT_H * LAT_W + h * LAT_W + w for f, h, w in keep_cells)
    zeros = (dm[0, :, 0] == 0.0).nonzero().flatten().tolist()
    assert zeros == expected
    # Every other token is a generate token (denoise 1.0) — no stray pins.
    assert int((dm[0, :, 0] == 1.0).sum()) == SEQ_LEN - len(keep_cells)


def test_token_mask_full_frame_matches_latent_frame_token_span() -> None:
    """Keeping ALL of latent frame 1 pins exactly mask_ops.latent_frame_token_span(1, H, W)."""
    mask = torch.zeros(LAT_F, LAT_H, LAT_W)
    mask[1] = 1.0  # keep the whole second latent frame
    dm = build_token_denoise_mask(mask, lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)
    start, stop = latent_frame_token_span(1, LAT_H, LAT_W)
    zeros = (dm[0, :, 0] == 0.0).nonzero().flatten().tolist()
    assert zeros == list(range(start, stop))


def test_token_mask_threshold_is_strictly_above_half() -> None:
    """Binarize at > 0.5: 0.6 -> KEEP (0.0 denoise); 0.4 and exactly 0.5 -> GENERATE (1.0)."""
    mask = torch.zeros(LAT_F, LAT_H, LAT_W)
    mask[0, 0, 0] = 0.6
    mask[0, 0, 1] = 0.4
    mask[0, 0, 2] = 0.5
    dm = build_token_denoise_mask(mask, lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)
    assert dm[0, 0, 0].item() == 0.0  # 0.6 > 0.5 -> KEEP
    assert dm[0, 1, 0].item() == 1.0  # 0.4 -> GENERATE
    assert dm[0, 2, 0].item() == 1.0  # exactly 0.5 -> GENERATE (mask <= 0.5)


def test_token_mask_rejects_grid_mismatch() -> None:
    """A mask encoded for different dims fails loud, naming both grids (GATE-SPEC top risk)."""
    with pytest.raises(ValueError, match="latent grid"):
        build_token_denoise_mask(
            _keep_mask([]), lat_f=LAT_F + 1, lat_h=LAT_H, lat_w=LAT_W
        )


def test_token_mask_rejects_wrong_ndim() -> None:
    """A non-3-dim tensor is not the per-sample [F_lat, H_lat, W_lat] contract."""
    with pytest.raises(ValueError, match="3-dim"):
        build_token_denoise_mask(
            torch.zeros(LAT_H, LAT_W), lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W
        )


# --------------------------------------------------------------------------------------------------
# pin_keep_tokens + per-token timesteps — where-semantics identical to training
# --------------------------------------------------------------------------------------------------


def test_pin_keep_tokens_equals_training_where_semantics() -> None:
    """denoised*dm + clean*(1-dm) == where(mask > 0.5, clean, denoised) — exactly, per token."""
    channels = 5
    mask = _keep_mask([(0, 0, 1), (1, 2, 3), (1, 0, 0)])
    dm = build_token_denoise_mask(mask, lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)
    keep_tokens = mask.reshape(1, SEQ_LEN, 1) > 0.5  # the training-side comparison

    denoised = torch.randn(1, SEQ_LEN, channels)
    clean = torch.randn(1, SEQ_LEN, channels)

    pinned = pin_keep_tokens(denoised, clean, dm)
    expected = torch.where(keep_tokens, clean, denoised)  # training: where(mask>0.5, clean, noisy)
    assert torch.equal(pinned, expected)  # binary mask -> the two forms are bit-identical


def test_pin_keep_tokens_all_keep_returns_clean() -> None:
    dm = torch.zeros(1, SEQ_LEN, 1)  # denoise 0.0 everywhere == all-KEEP
    denoised = torch.randn(1, SEQ_LEN, 3)
    clean = torch.randn(1, SEQ_LEN, 3)
    assert torch.equal(pin_keep_tokens(denoised, clean, dm), clean)


def test_pin_keep_tokens_all_generate_returns_denoised() -> None:
    dm = torch.ones(1, SEQ_LEN, 1)  # denoise 1.0 everywhere == all-GENERATE
    denoised = torch.randn(1, SEQ_LEN, 3)
    clean = torch.randn(1, SEQ_LEN, 3)
    assert torch.equal(pin_keep_tokens(denoised, clean, dm), denoised)


def test_per_token_timesteps_zero_on_keep_tokens() -> None:
    """sigma * denoise_mask (the shared mask_ops primitive) == where(mask > 0.5, 0, sigma)."""
    mask = _keep_mask([(0, 1, 1), (1, 2, 0)])
    dm = build_token_denoise_mask(mask, lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)
    sigma = torch.tensor([0.7])
    timesteps = per_token_sigma(sigma, dm[:, :, 0])  # [1, seq]
    keep_tokens = mask.reshape(1, SEQ_LEN) > 0.5
    expected = torch.where(keep_tokens, torch.zeros(1), torch.full((1,), 0.7))
    assert torch.equal(timesteps, expected)


# --------------------------------------------------------------------------------------------------
# masked_render_latent_grid — inpaint ÷64 dims rule + causal latent-grid math
# --------------------------------------------------------------------------------------------------


def test_latent_grid_primary_bucket() -> None:
    """1280x704 @ 81f (the res-matched primary bucket) -> ((81-1)//8+1, 704//32, 1280//32)."""
    assert masked_render_latent_grid(704, 1280, 81) == (11, 22, 40)


def test_latent_grid_cheap_smoke_bucket() -> None:
    """512x384 @ 49f (the cheap smoke bucket) -> (7, 12, 16)."""
    assert masked_render_latent_grid(384, 512, 49) == (7, 12, 16)


def test_latent_grid_rejects_non_div64_height() -> None:
    """352 is %32-clean but NOT %64 — the inpaint rule is STRICTER than the video-wide one."""
    assert 352 % 32 == 0  # sanity: the plain video rule would accept this
    with pytest.raises(ValueError, match="height=352"):
        masked_render_latent_grid(352, 768, 49)


def test_latent_grid_rejects_non_div64_width() -> None:
    with pytest.raises(ValueError, match="width=736"):
        masked_render_latent_grid(704, 736, 49)


def test_latent_grid_rejects_non_8k1_frames() -> None:
    with pytest.raises(ValueError, match="num_frames=48"):
        masked_render_latent_grid(704, 1280, 48)


# --------------------------------------------------------------------------------------------------
# plan_mask_condition — config-shape acceptance (dict AND the real MaskCondition sub-model)
# --------------------------------------------------------------------------------------------------


def test_plan_mask_condition_accepts_plain_dict() -> None:
    video, mask = plan_mask_condition(
        {"video": "clips/test1.mp4", "mask": "video_masks/test1_mask.pt"}
    )
    assert (video, mask) == ("clips/test1.mp4", "video_masks/test1_mask.pt")


def test_plan_mask_condition_accepts_type_discriminator() -> None:
    video, mask = plan_mask_condition(
        {"type": "mask", "video": "clips/a.mp4", "mask": "video_masks/a.pt"}
    )
    assert (video, mask) == ("clips/a.mp4", "video_masks/a.pt")


def test_plan_mask_condition_accepts_the_real_schema_model() -> None:
    """The landed config surface (MaskCondition) normalizes identically to the dict form."""
    cond = MaskCondition(video="clips/test2.mp4", mask="video_masks/test2_mask.pt")
    assert plan_mask_condition(cond) == ("clips/test2.mp4", "video_masks/test2_mask.pt")


def test_plan_mask_condition_accepts_attribute_objects() -> None:
    cond = SimpleNamespace(type="mask", video="clips/b.mp4", mask="video_masks/b.pt")
    assert plan_mask_condition(cond) == ("clips/b.mp4", "video_masks/b.pt")


def test_plan_mask_condition_rejects_wrong_type() -> None:
    with pytest.raises(ValueError, match="type='depth'"):
        plan_mask_condition({"type": "depth", "video": "a.mp4", "mask": "a.pt"})


@pytest.mark.parametrize("missing", ["video", "mask"])
def test_plan_mask_condition_rejects_missing_field(missing: str) -> None:
    cond = {"video": "clips/a.mp4", "mask": "video_masks/a.pt"}
    del cond[missing]
    with pytest.raises(ValueError, match=missing):
        plan_mask_condition(cond)


@pytest.mark.parametrize("empty", ["video", "mask"])
def test_plan_mask_condition_rejects_empty_field(empty: str) -> None:
    cond = {"video": "clips/a.mp4", "mask": "video_masks/a.pt", empty: ""}
    with pytest.raises(ValueError, match=empty):
        plan_mask_condition(cond)


# --------------------------------------------------------------------------------------------------
# _load_latent_mask — the .pt branch (tmp_path; no Modal, no VAE, no mask_encode import)
# --------------------------------------------------------------------------------------------------


def test_load_latent_mask_pt_roundtrip(tmp_path) -> None:
    """A saved float32 {0,1} [F,H,W] tensor loads back equal (and float32)."""
    mask = _keep_mask([(0, 1, 2), (1, 0, 0)])
    path = tmp_path / "test1_mask.pt"
    torch.save(mask, str(path))
    loaded = _load_latent_mask(str(path), lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)
    assert loaded.dtype == torch.float32
    assert torch.equal(loaded, mask)


def test_load_latent_mask_pt_rejects_non_tensor(tmp_path) -> None:
    """A .pt that is not a bare tensor violates the video_masks per-sample contract."""
    path = tmp_path / "bad_mask.pt"
    torch.save({"mask": _keep_mask([])}, str(path))
    with pytest.raises(ValueError, match="tensor"):
        _load_latent_mask(str(path), lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)


def test_load_latent_mask_non_pt_routes_through_mask_encode(monkeypatch) -> None:
    """A non-.pt source routes through data/mask_encode (read_mask_frames -> encode_mask_pixels).

    The GATE-SPEC canonical-encoder-exception doctrine: ONE mask-encode implementation. The two
    encoder functions are stubbed (no decode backend on CI) to assert the sampler branch composes
    them with the RIGHT arguments — expected_frames = (F_lat-1)*8+1 pixel frames, encode at the
    render's latent grid — and returns the encode verbatim. A rename in data/mask_encode.py fails
    HERE, on CPU, not on a metered render.
    """
    import signet_trainer.data.mask_encode as mask_encode

    pixel_f = (LAT_F - 1) * 8 + 1
    pixels_sentinel = torch.zeros(pixel_f, LAT_H * 32, LAT_W * 32)
    encoded_sentinel = _keep_mask([(0, 0, 0)])
    calls: dict[str, tuple] = {}

    def fake_read(path, expected_frames=None):
        calls["read"] = (str(path), expected_frames)
        return pixels_sentinel

    def fake_encode(mask_pixels, lat_f, lat_h, lat_w):
        calls["encode"] = (mask_pixels is pixels_sentinel, lat_f, lat_h, lat_w)
        return encoded_sentinel

    monkeypatch.setattr(mask_encode, "read_mask_frames", fake_read)
    monkeypatch.setattr(mask_encode, "encode_mask_pixels", fake_encode)

    out = _load_latent_mask("clips/test1_mask.mp4", lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)
    assert calls["read"] == ("clips/test1_mask.mp4", pixel_f)
    assert calls["encode"] == (True, LAT_F, LAT_H, LAT_W)
    assert torch.equal(out, encoded_sentinel)


def test_real_mask_encode_polarity_survives_the_sampler_inversion() -> None:
    """Cross-agent contract lock: the REAL encode_mask_pixels output inverts to the right pins.

    A white (KEEP) 32x32 pixel block at (h=0, w=0) on every frame encodes to latent cell
    (f, 0, 0) = 1.0 (data/mask_encode.py), and the sampler-side denoise mask then pins EXACTLY
    that cell's token in every latent frame — polarity holds END-TO-END across the two modules.
    """
    from signet_trainer.data.mask_encode import encode_mask_pixels

    pixel_f, pixel_h, pixel_w = (LAT_F - 1) * 8 + 1, LAT_H * 32, LAT_W * 32
    pixels = torch.zeros(pixel_f, pixel_h, pixel_w)
    pixels[:, :32, :32] = 1.0  # one white (KEEP) block == latent cell (h=0, w=0), all frames
    encoded = encode_mask_pixels(pixels, LAT_F, LAT_H, LAT_W)
    assert encoded.shape == (LAT_F, LAT_H, LAT_W)
    assert encoded[0, 0, 0].item() == 1.0  # KEEP survives the encode
    assert encoded[0, 1, 1].item() == 0.0  # GENERATE stays 0

    dm = build_token_denoise_mask(encoded, lat_f=LAT_F, lat_h=LAT_H, lat_w=LAT_W)
    keep_tokens = (dm[0, :, 0] == 0.0).nonzero().flatten().tolist()
    # Exactly the (h=0, w=0) token of every latent frame — frame-major stride H_lat*W_lat.
    assert keep_tokens == [f * LAT_H * LAT_W for f in range(LAT_F)]


# --------------------------------------------------------------------------------------------------
# import confinement — the new branch must not un-confine the sampler module
# --------------------------------------------------------------------------------------------------


def test_mask_branch_keeps_sampler_import_confined_on_cpu() -> None:
    """Importing the sampler (with the mask branch) must NOT pull ltx/modal on CPU/CI."""
    for mod in ("modal", "ltx_core", "ltx_trainer"):
        sys.modules.pop(mod, None)

    from signet_trainer.inference.sampler import run_mask_condition_sampler  # noqa: F401

    for mod in ("modal", "ltx_core", "ltx_trainer"):
        assert mod not in sys.modules, (
            f"import-confinement violation: the mask branch transitively imported {mod!r}"
        )
