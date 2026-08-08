"""The 2x2 pack, proved by ROUND TRIP rather than by shape.

Shape proves nothing here. Every wrong ordering of the six-axis permute produces
``[B, rows, 64]`` exactly like the right one — that is the entire hazard the packing module exists
to contain. So the load-bearing test is bit-equality through pack -> unpack, plus explicit
demonstrations that the near-miss orderings a reimplementation would reach for are DIFFERENT.

CPU only, synthetic tensors, no weights.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from signet_trainer.conditioning.qwen_edit_geometry import (  # noqa: E402
    QWEN_EDIT_LATENT_CHANNELS,
    QWEN_EDIT_PATCH_DIM,
    qwen_edit_rows_of,
)
from signet_trainer.conditioning.qwen_edit_packing import (  # noqa: E402
    qwen_edit_image_rows,
    qwen_edit_rows_to_latent,
)

C = QWEN_EDIT_LATENT_CHANNELS


def _latent(h: int, w: int, b: int = 1) -> "torch.Tensor":
    """Arange, not randn — every element distinct, so a permutation error cannot cancel."""
    return torch.arange(b * C * h * w, dtype=torch.float32).reshape(b, C, h, w)


# ==================================================================================================
# The proof.
# ==================================================================================================


@pytest.mark.parametrize(("h", "w"), [(128, 128), (128, 96), (96, 128), (176, 92), (32, 32)])
def test_pack_unpack_round_trips_bit_exactly(h: int, w: int) -> None:
    """pack -> unpack returns the original latent EXACTLY. This is the ordering proof."""
    latent = _latent(h, w)
    rows = qwen_edit_image_rows(latent)
    assert rows.shape == (1, (h // 2) * (w // 2), QWEN_EDIT_PATCH_DIM)
    assert torch.equal(qwen_edit_rows_to_latent(rows, h, w), latent)


def test_the_row_count_agrees_with_the_geometry_module() -> None:
    """The packer and the row-budget arithmetic must not drift apart.

    ``qwen_edit_rows_of`` prices a run at config load; this function produces the rows that price
    was meant to describe. If they disagree, a config is priced for one sequence and trains another.
    """
    for pixel_h, pixel_w in ((1024, 1024), (768, 1376), (1408, 736), (832, 1248)):
        latent_h, latent_w = pixel_h // 8, pixel_w // 8
        rows = qwen_edit_image_rows(_latent(latent_h, latent_w))
        assert int(rows.shape[1]) == qwen_edit_rows_of(pixel_w, pixel_h), (
            f"{pixel_w}x{pixel_h}: packer produced {int(rows.shape[1])} rows, geometry priced "
            f"{qwen_edit_rows_of(pixel_w, pixel_h)}"
        )


def test_the_near_miss_orderings_are_genuinely_different() -> None:
    """The two plausible wrong permutes give the SAME SHAPE and DIFFERENT VALUES.

    This is the test that justifies the module's existence. Without it, "it returns [B, rows, 64]"
    reads like sufficient evidence of correctness, and it is not.
    """
    h = w = 64
    latent = _latent(h, w)
    correct = qwen_edit_image_rows(latent)

    staged = latent.view(1, C, h // 2, 2, w // 2, 2)
    pixel_major = staged.permute(0, 2, 4, 3, 5, 1).reshape(1, (h // 2) * (w // 2), C * 4)
    channel_last = staged.permute(0, 2, 4, 1, 5, 3).reshape(1, (h // 2) * (w // 2), C * 4)

    for label, wrong in (("pixel-major (ph,pw,C)", pixel_major), ("row/col swapped", channel_last)):
        assert wrong.shape == correct.shape, f"{label}: shapes must match — that is the hazard"
        assert not torch.equal(wrong, correct), (
            f"{label} produced the SAME values as the correct pack; the round-trip test would not "
            f"discriminate and this test is no longer meaningful"
        )


def test_the_intra_patch_block_is_channel_slowest() -> None:
    """Spot-check the documented ordering directly: row 0 is block (0,0) read channel-slowest."""
    h = w = 4
    latent = _latent(h, w)
    rows = qwen_edit_image_rows(latent)
    expected = torch.stack([latent[0, c, :2, :2].reshape(-1) for c in range(C)]).reshape(-1)
    assert torch.equal(rows[0, 0], expected)


# ==================================================================================================
# Input forms and refusals.
# ==================================================================================================


def test_the_on_disk_cfhw_form_is_accepted() -> None:
    """``[C, F=1, H, W]`` is what prep writes; it must pack identically to ``[B=1, C, H, W]``."""
    latent = _latent(32, 32)
    cfhw = latent.squeeze(0).unsqueeze(1)
    assert cfhw.shape == (C, 1, 32, 32)
    assert torch.equal(qwen_edit_image_rows(cfhw), qwen_edit_image_rows(latent))


def test_an_unbatched_chw_is_accepted() -> None:
    assert torch.equal(
        qwen_edit_image_rows(_latent(32, 32).squeeze(0)), qwen_edit_image_rows(_latent(32, 32))
    )


def test_a_real_batch_packs_per_item() -> None:
    batched = _latent(32, 32, b=3)
    rows = qwen_edit_image_rows(batched)
    assert rows.shape == (3, 256, QWEN_EDIT_PATCH_DIM)
    for i in range(3):
        assert torch.equal(rows[i : i + 1], qwen_edit_image_rows(batched[i : i + 1]))


def test_a_video_latent_is_refused_not_squeezed() -> None:
    """F > 1 on an IMAGE family is a video latent that wandered in."""
    with pytest.raises(ValueError, match="F=4"):
        qwen_edit_image_rows(torch.zeros(C, 4, 32, 32))


def test_the_ambiguous_shape_is_refused_rather_than_guessed() -> None:
    """``[16, 16, H, W]`` reads as [C,F,H,W] and [B,C,H,W] equally; guessing transposes B and C."""
    with pytest.raises(ValueError, match="ambiguous"):
        qwen_edit_image_rows(torch.zeros(C, C, 32, 32))


def test_an_odd_latent_edge_is_refused() -> None:
    with pytest.raises(ValueError, match="divisible"):
        qwen_edit_image_rows(torch.zeros(1, C, 33, 32))


def test_a_wrong_channel_count_is_refused() -> None:
    with pytest.raises(ValueError, match="latent channel"):
        qwen_edit_image_rows(torch.zeros(1, 8, 32, 32))


def test_dtype_and_device_survive() -> None:
    latent = _latent(32, 32).to(torch.bfloat16)
    assert qwen_edit_image_rows(latent).dtype is torch.bfloat16


def test_the_error_names_the_caller_supplied_label() -> None:
    """Slot-level labels make a failure name the offending input, not just 'latent'."""
    with pytest.raises(ValueError, match="control slot 1"):
        qwen_edit_image_rows(torch.zeros(1, 8, 32, 32), name="control slot 1")


def test_the_module_imports_without_the_heavy_stack() -> None:
    """The dry-run imports this; it must not drag in modal/diffusers/transformers/ltx_core.

    Measured in a FRESH SUBPROCESS, not by inspecting this process's ``sys.modules``. An in-process
    check answers "what has the whole suite imported by now", which is a different question and a
    changing one — it passes when this file runs alone and fails when a sibling has already pulled
    diffusers in. That is an order-dependent test, which is the exact defect
    ``tests/test_mask_encode.py`` was fixed for earlier today. A subprocess makes the question
    hermetic: import ONLY this module, then look.
    """
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")
    probe = (
        "import sys; "
        "import signet_trainer.conditioning.qwen_edit_packing as m; "
        "bad=[n for n in ('modal','diffusers','transformers','ltx_core') if n in sys.modules]; "
        "print('LEAKED:'+','.join(bad) if bad else 'CLEAN'); "
        "print('HAS_TORCH:'+str('torch' in sys.modules))"
    )
    import os

    # INHERIT the environment and override only PYTHONPATH. A hand-built minimal env omits
    # SYSTEMROOT, without which CPython on Windows dies at startup with
    # "Fatal Python error: _Py_HashRandomization_Init: failed to get random numbers" — a probe
    # failure that looks nothing like the thing being probed.
    env = dict(os.environ)
    env["PYTHONPATH"] = src
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, f"probe failed: {result.stderr[-800:]}"
    assert "CLEAN" in result.stdout, (
        f"qwen_edit_packing imported a forbidden dependency — {result.stdout.strip()}. This module "
        f"is on the free CPU dry-run path; pulling in modal/diffusers/transformers/ltx_core would "
        f"make that check require the GPU image."
    )
    # And it does not import torch EITHER, which is stronger than required and worth pinning.
    # The module annotates tensors as ``Any`` and calls ``.view``/``.permute``/``.reshape`` on
    # whatever it is handed, so importing it costs nothing at all. Anyone adding a top-level
    # ``import torch`` for a type annotation should have to see this test say so first — it would
    # put a multi-hundred-MB import on the config-load path for a cosmetic gain.
    assert "HAS_TORCH:False" in result.stdout, (
        "qwen_edit_packing now imports torch at module level. That is not fatal, but it was free "
        "before: the transform is duck-typed and needs no import. Prefer `if TYPE_CHECKING:` for "
        "annotations, or update this test deliberately."
    )
