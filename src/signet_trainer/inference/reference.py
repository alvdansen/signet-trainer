"""inference.reference — the pure single-frame condition-image normalizer (05-05, SC#2).

One function, ``to_condition_image``, that turns whatever a decoded frame looks like into the
exact tensor the sampler's first-frame conditioning path wants: a ``[3, H, W]`` float tensor with
every value in ``[0, 1]``. It is the malformed-image guard (threat T-05-03 Tampering/DoS): a
uint8/float, HWC/CHW, RGB/RGBA/grayscale, in-range or wildly-out-of-range input can never corrupt
the tensor handed to ``build_generation_config(condition_image=...)`` — it is coerced to shape and
clamped to range here.

Keeping it pure (torch + stdlib only at module top; no ltx_trainer / no filesystem / no GPU) makes
the ``[0,1]`` clamp CPU-testable on Windows/CI — the same import-confinement discipline as
``inference/sampler.py`` (Anti-Pattern 6). The Modal ``sample`` branch feeds a decoded frame
through this before ref-ON rendering.

CONTRADICTION #1 (carried): the normalized tensor rides ``GenerationConfig.condition_image`` (the
ported ``ValidationSampler`` field, implicit first-frame index, hard-clean), NOT the diffusers
first-frame condition classes. The sampler VAE-encodes / resizes / center-crops internally, so this
helper only fixes dtype, layout, channel count, and range — no resize, no VAE, no pre-encoding.
"""

from __future__ import annotations

import torch

__all__ = ["to_condition_image"]

# Plausible channel counts for the channel axis (grayscale / RGB / RGBA).
_CHANNEL_COUNTS = frozenset({1, 3, 4})


def to_condition_image(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a decoded frame to a ``[3, H, W]`` float tensor clamped to ``[0, 1]``.

    Accepts:
      * dtype — integer (e.g. uint8 ``[0,255]``, scaled by ``1/255``) or float (used as-is).
      * layout — ``[H, W]`` (grayscale), ``[C, H, W]`` (channel-first), or ``[H, W, C]``
        (channel-last / HWC, as ``torchvision.io`` decode or a numpy frame gives); HWC is
        permuted to CHW, CHW passes through.
      * channels — grayscale (1) is expanded to RGB, RGB (3) passes through, RGBA (4) has its
        alpha dropped.

    Always returns a contiguous ``[3, H, W]`` floating-point tensor with values clamped to
    ``[0, 1]`` (the defensive guard: an out-of-range float is clamped, not rescaled).

    Raises:
        ValueError: if the input is not 2D/3D, or a 3D input's channel axis cannot be identified.
    """
    if not isinstance(tensor, torch.Tensor):  # defensive — the sampler path only ever passes tensors.
        raise ValueError(f"to_condition_image expects a torch.Tensor, got {type(tensor)!r}")

    was_integer = not tensor.dtype.is_floating_point

    # ── layout → CHW ────────────────────────────────────────────────────────────────────────────
    if tensor.ndim == 2:
        # [H, W] grayscale — add a single channel axis.
        chw = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        c_first = int(tensor.shape[0]) in _CHANNEL_COUNTS
        c_last = int(tensor.shape[-1]) in _CHANNEL_COUNTS
        if c_last:
            # HWC → CHW. The last axis is treated as the channel axis whenever it is a valid
            # channel count — channel-last is the common decoded-frame layout (cv2 / numpy /
            # torchvision-array), and this is the tie-break for ambiguous shapes like [4,H,3].
            chw = tensor.permute(2, 0, 1)
        elif c_first:
            # CHW (channel-first) — the last axis is not a plausible channel count.
            chw = tensor
        else:
            raise ValueError(
                f"to_condition_image: cannot identify a channel axis in shape {tuple(tensor.shape)} "
                "(no dimension is 1, 3, or 4)."
            )
    else:
        raise ValueError(
            f"to_condition_image: expected a 2D [H,W] or 3D [C,H,W]/[H,W,C] tensor, got "
            f"{tensor.ndim}D shape {tuple(tensor.shape)}."
        )

    # ── channels → exactly 3 (RGB) ──────────────────────────────────────────────────────────────
    c = int(chw.shape[0])
    if c == 1:
        chw = chw.expand(3, *chw.shape[1:])   # grayscale → RGB
    elif c == 4:
        chw = chw[:3]                          # drop alpha
    elif c != 3:
        raise ValueError(f"to_condition_image: unexpected channel count {c} (want 1, 3, or 4).")

    # ── dtype → float, scale integers, clamp to [0, 1] ──────────────────────────────────────────
    out = chw.to(dtype=torch.float32)
    if was_integer:
        out = out / 255.0
    out = out.clamp(0.0, 1.0)

    return out.contiguous()
