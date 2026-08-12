"""The 2x2 patch pack for Qwen-Image-Edit — THE single transcription of that transform.

``QwenEditStrategy`` refuses a latent-form payload unless this module's :func:`qwen_edit_image_rows`
is injected as its ``pack_fn``, and the refusal message says why: *"the intra-patch element ORDER is
a checkpoint contract and a wrong order corrupts training silently at the correct shape."* Every
wrong ordering of the six-axis permute below yields a tensor of the RIGHT shape holding the WRONG
values. Nothing raises. The loss is finite and descends. The adapter is simply wrong, and stays
wrong through every later round of a chain that warm-starts from it.

So this transform is transcribed ONCE, here, and every consumer imports it.

THE TRANSFORM, and it is identical in both upstreams — checked, not assumed:

  diffusers  ``pipeline_qwenimage_edit_plus.py:389-394``   (``_pack_latents``)
  ai-toolkit ``qwen_image_edit_plus.py:223-229``

      latents.view(B, C, H // 2, 2, W // 2, 2)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B, (H // 2) * (W // 2), C * 4)

The two agree byte-for-byte. That is worth stating explicitly because the OTHER geometry decision
on this arch does NOT agree — ai-toolkit computes its area-budget aspect ratio as ``H/W`` where
diffusers uses ``W/H``, transposing every non-square control image
(``conditioning/qwen_edit_geometry.py`` records that divergence and this project follows diffusers).
The pack is common ground; the ratio is not. Do not let the second fact cast doubt on the first.

READING THE PERMUTE. Axes after the view are ``(B, C, hb, ph, wb, pw)`` — batch, channel, the
half-resolution block grid, and the 2x2 intra-patch offsets. ``permute(0, 2, 4, 1, 3, 5)`` reorders
them to ``(B, hb, wb, C, ph, pw)``: block-major on the outside, and INSIDE each block the 64 values
run channel-slowest, then row, then column. The final reshape flattens ``(C, ph, pw) -> C*4 = 64``.
Two orderings a reimplementation reaches for naturally — ``(ph, pw, C)`` (pixel-major) and a
``permute(0, 2, 4, 3, 5, 1)`` — both produce ``[B, rows, 64]`` and both are wrong.

Everything here is pure torch on whatever device it is handed. No modal, no diffusers, no
transformers: this module is imported by the CPU dry-run.
"""

from __future__ import annotations

from typing import Any

from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_LATENT_CHANNELS,
    QWEN_EDIT_PATCH,
    QWEN_EDIT_PATCH_DIM,
)

__all__ = ["qwen_edit_image_rows", "qwen_edit_rows_to_latent"]


def _as_bchw(tensor: Any, name: str) -> tuple[Any, int]:
    """Normalise the accepted latent forms to ``[B, C, H, W]``, returning the batch size.

    Three forms reach this function and all three are legitimate:

      ``[C, H, W]``      a single latent, unbatched
      ``[C, F, H, W]``   signet's ON-DISK form. ``prep/qwen_edit_encode`` stores latents in the
                         same ``[C, F, H, W]`` dict shape the H3/LTX precomputed sources use, which
                         is what lets ``PrecomputedDataset`` pair them by rel path with no special
                         case. F is pinned to 1 on this family — it is an IMAGE model — and an F > 1
                         here is a video latent that wandered in, so it is refused rather than
                         silently squeezed.
      ``[B, C, H, W]``   a collated batch.

    ``[C, F, H, W]`` and ``[B, C, H, W]`` are both 4-D and cannot be told apart by rank alone. They
    are disambiguated by CHANNEL COUNT: the Qwen-Image VAE emits exactly
    ``QWEN_EDIT_LATENT_CHANNELS`` (16) channels, so a 4-D tensor whose axis 0 is 16 is
    ``[C, F, H, W]``, and anything else is a batch. An ambiguous ``[16, 16, H, W]`` — a batch of 16
    — is refused rather than guessed, because guessing wrong here silently transposes batch and
    channel and produces, once more, a correctly-shaped wrong answer.
    """
    if tensor.ndim == 3:
        return tensor.unsqueeze(0), 1

    if tensor.ndim == 4:
        axis0, axis1 = int(tensor.shape[0]), int(tensor.shape[1])
        looks_cfhw = axis0 == QWEN_EDIT_LATENT_CHANNELS
        looks_bchw = axis1 == QWEN_EDIT_LATENT_CHANNELS
        if looks_cfhw and looks_bchw:
            raise ValueError(
                f"{name}: shape {tuple(tensor.shape)} is ambiguous — axis 0 and axis 1 are both "
                f"{QWEN_EDIT_LATENT_CHANNELS}, so it reads equally as [C, F, H, W] with F="
                f"{QWEN_EDIT_LATENT_CHANNELS} and as [B, C, H, W] with B="
                f"{QWEN_EDIT_LATENT_CHANNELS}. Refusing rather than guessing: the two differ by a "
                f"transpose of batch and channel, which packs to the correct shape and the wrong "
                f"values. Pass [C, H, W] per sample, or squeeze F before collating."
            )
        if looks_cfhw:
            frames = axis1
            if frames != 1:
                raise ValueError(
                    f"{name}: [C, F, H, W] latent carries F={frames}. qwen_edit is an IMAGE family "
                    f"and F is pinned to 1 everywhere; an F > 1 here is a video latent reaching an "
                    f"image packer. Refused rather than squeezed — a squeeze would silently train "
                    f"on frame 0 of a clip nobody meant to send."
                )
            return tensor.permute(1, 0, 2, 3), 1  # [C, 1, H, W] -> [1, C, H, W]
        if looks_bchw:
            return tensor, axis0
        raise ValueError(
            f"{name}: 4-D latent {tuple(tensor.shape)} has {QWEN_EDIT_LATENT_CHANNELS} channels on "
            f"neither axis 0 ([C, F, H, W]) nor axis 1 ([B, C, H, W]). The Qwen-Image VAE emits "
            f"exactly {QWEN_EDIT_LATENT_CHANNELS} latent channels, so this is not a Qwen-Image "
            f"latent."
        )

    raise ValueError(
        f"{name}: expected a 3-D [C, H, W], 4-D [C, F, H, W] or 4-D [B, C, H, W] latent, got "
        f"{tensor.ndim}-D {tuple(tensor.shape)}."
    )


def qwen_edit_image_rows(latent: Any, *, name: str = "latent") -> Any:
    """Pack a Qwen-Image latent into the transformer's row form ``[B, rows, 64]``.

    Args:
        latent: ``[C, H, W]``, ``[C, F=1, H, W]`` or ``[B, C, H, W]``, channels =
            ``QWEN_EDIT_LATENT_CHANNELS``.
        name: what to call this tensor in an error message. Callers pass the slot's own label
            (``"control slot 1"``) so a failure names the offending input rather than "latent".

    Returns:
        ``[B, (H//2) * (W//2), QWEN_EDIT_PATCH_DIM]`` — contiguous, same dtype and device as input.

    Raises:
        ValueError: on any rank/channel/parity violation. Every check is a refusal rather than a
            fix-up, for the reason at the top of this module: the failure mode is a correctly-shaped
            wrong answer, so anything this function repairs silently is a bug it has hidden.
    """
    bchw, batch = _as_bchw(latent, name)
    channels, height, width = int(bchw.shape[1]), int(bchw.shape[2]), int(bchw.shape[3])

    if channels != QWEN_EDIT_LATENT_CHANNELS:
        raise ValueError(
            f"{name}: {channels} latent channel(s), expected {QWEN_EDIT_LATENT_CHANNELS}. The "
            f"packed feature width is channels x {QWEN_EDIT_PATCH}^2, so a wrong channel count "
            f"yields rows the transformer's img_in ({QWEN_EDIT_PATCH_DIM} -> 3072) cannot consume."
        )

    if height % QWEN_EDIT_PATCH or width % QWEN_EDIT_PATCH:
        raise ValueError(
            f"{name}: latent grid {height}x{width} is not divisible by {QWEN_EDIT_PATCH}. The 2x2 "
            f"pack has no remainder handling and neither does the model — this is why every canvas "
            f"is snapped to a multiple of 32 pixels upstream (32 = VAE scale 8 x patch "
            f"{QWEN_EDIT_PATCH} x 2), so an odd latent edge means the canvas skipped that snap."
        )

    # ── THE TRANSFORM. diffusers pipeline_qwenimage_edit_plus.py:389-394,
    #    ai-toolkit qwen_image_edit_plus.py:223-229. Do not "simplify" this.
    rows = bchw.view(
        batch, channels, height // QWEN_EDIT_PATCH, QWEN_EDIT_PATCH, width // QWEN_EDIT_PATCH, QWEN_EDIT_PATCH
    )
    rows = rows.permute(0, 2, 4, 1, 3, 5)
    rows = rows.reshape(
        batch,
        (height // QWEN_EDIT_PATCH) * (width // QWEN_EDIT_PATCH),
        channels * QWEN_EDIT_PATCH * QWEN_EDIT_PATCH,
    )

    if int(rows.shape[-1]) != QWEN_EDIT_PATCH_DIM:
        raise ValueError(
            f"{name}: packed feature width {int(rows.shape[-1])} != {QWEN_EDIT_PATCH_DIM}. "
            f"Unreachable via the checks above; if it fires, the geometry constants and this "
            f"transform have drifted apart."
        )
    return rows.contiguous()


def qwen_edit_rows_to_latent(rows: Any, height: int, width: int, *, name: str = "rows") -> Any:
    """Exact inverse of :func:`qwen_edit_image_rows`, returning ``[B, C, H, W]``.

    Transcribed from diffusers ``_unpack_latents``
    (``pipeline_qwenimage_edit_plus.py:398-410``), minus its pixel-space rescaling — this takes the
    LATENT grid directly, because its only caller is the round-trip test that proves the forward
    transform is order-correct.

    That test is the point of shipping an inverse at all. Asserting the forward pack's SHAPE proves
    nothing; the wrong permutes produce that shape too. Round-tripping to bit-equality is what
    distinguishes the correct ordering from the plausible ones.
    """
    if rows.ndim != 3:
        raise ValueError(f"{name}: expected 3-D [B, rows, features], got {tuple(rows.shape)}")
    batch, n_rows, features = (int(x) for x in rows.shape)
    if features != QWEN_EDIT_PATCH_DIM:
        raise ValueError(f"{name}: feature width {features} != {QWEN_EDIT_PATCH_DIM}")
    if height % QWEN_EDIT_PATCH or width % QWEN_EDIT_PATCH:
        raise ValueError(f"{name}: latent grid {height}x{width} is not divisible by {QWEN_EDIT_PATCH}")

    expected = (height // QWEN_EDIT_PATCH) * (width // QWEN_EDIT_PATCH)
    if n_rows != expected:
        raise ValueError(
            f"{name}: {n_rows} row(s) do not match latent grid {height}x{width} "
            f"(expected {expected} = {height // QWEN_EDIT_PATCH} x {width // QWEN_EDIT_PATCH})"
        )

    latent = rows.view(
        batch,
        height // QWEN_EDIT_PATCH,
        width // QWEN_EDIT_PATCH,
        features // (QWEN_EDIT_PATCH * QWEN_EDIT_PATCH),
        QWEN_EDIT_PATCH,
        QWEN_EDIT_PATCH,
    )
    latent = latent.permute(0, 3, 1, 4, 2, 5)
    return latent.reshape(
        batch, features // (QWEN_EDIT_PATCH * QWEN_EDIT_PATCH), height, width
    ).contiguous()
