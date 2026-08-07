"""CPU/Windows tests for the pure single-frame condition-image normalizer (05-05 Task 2, SC#2).

``to_condition_image`` is the malformed-image guard (threat T-05-03): whatever a decoded frame
looks like — uint8 or float, HWC or CHW, RGB / RGBA / grayscale, in-range or out-of-range — it
must come out as a ``[3, H, W]`` float tensor clamped to ``[0, 1]``, ready to feed straight into
``build_generation_config(condition_image=...)``. Torch-only, no GPU, no ltx_trainer — runs in CI.
"""

from __future__ import annotations

import torch

from signet_trainer.inference.reference import to_condition_image


def test_uint8_hwc_scaled_and_transposed() -> None:
    # An HxWx3 uint8 frame (the torchvision.io.read_image / decoded-array shape).
    h, w = 4, 5
    img = torch.zeros((h, w, 3), dtype=torch.uint8)
    img[..., 0] = 255  # full-red channel
    out = to_condition_image(img)
    assert out.shape == (3, h, w), out.shape
    assert out.dtype.is_floating_point
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    # 255 -> 1.0 on channel 0; 0 -> 0.0 elsewhere.
    assert torch.allclose(out[0], torch.ones(h, w))
    assert torch.allclose(out[1], torch.zeros(h, w))


def test_uint8_chw_passthrough_layout_scaled() -> None:
    # An already channel-first uint8 frame stays CHW; values scale by 1/255.
    img = torch.full((3, 2, 2), 128, dtype=torch.uint8)
    out = to_condition_image(img)
    assert out.shape == (3, 2, 2)
    assert abs(float(out[0, 0, 0]) - (128 / 255)) < 1e-6


def test_float_out_of_range_is_clamped() -> None:
    # Defensive: an out-of-range float input is clamped into [0, 1], not rescaled.
    img = torch.tensor([[[2.0, -1.0], [0.5, 0.25]]] * 3)  # [3,2,2]
    out = to_condition_image(img)
    assert out.shape == (3, 2, 2)
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0
    assert float(out[0, 0, 0]) == 1.0   # 2.0 -> 1.0
    assert float(out[0, 0, 1]) == 0.0   # -1.0 -> 0.0
    assert abs(float(out[0, 1, 0]) - 0.5) < 1e-6  # in-range value untouched


def test_float_in_range_passthrough_values() -> None:
    img = torch.full((3, 3, 3), 0.5)
    out = to_condition_image(img)
    assert torch.allclose(out, torch.full((3, 3, 3), 0.5))


def test_rgba_alpha_dropped_chw() -> None:
    img = torch.zeros((4, 2, 2), dtype=torch.uint8)
    out = to_condition_image(img)
    assert out.shape == (3, 2, 2)


def test_rgba_alpha_dropped_hwc() -> None:
    img = torch.zeros((2, 2, 4), dtype=torch.uint8)
    out = to_condition_image(img)
    assert out.shape == (3, 2, 2)


def test_grayscale_2d_expanded_to_rgb() -> None:
    img = torch.full((6, 7), 255, dtype=torch.uint8)  # [H,W], no channel
    out = to_condition_image(img)
    assert out.shape == (3, 6, 7)
    assert torch.allclose(out, torch.ones(3, 6, 7))


def test_grayscale_single_channel_expanded_to_rgb() -> None:
    img = torch.full((1, 6, 7), 128, dtype=torch.uint8)  # [1,H,W]
    out = to_condition_image(img)
    assert out.shape == (3, 6, 7)
    # all three channels identical (expanded from the one grey channel)
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])


def test_output_is_contiguous_float() -> None:
    img = torch.zeros((5, 4, 3), dtype=torch.uint8)  # HWC -> permute makes it non-contiguous
    out = to_condition_image(img)
    assert out.is_contiguous()
    assert out.dtype.is_floating_point
