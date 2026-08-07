"""07-07 tier-1 CPU proof — paired-reference preflight + downscale-factor inference (SC#2).

``assert_paired_reference`` is the cheap OOM guard Modal (07-09) runs BEFORE spend: a mismatched
reference/target bucket (Pitfall 2, the doubled-sequence OOM) must be rejected on CPU, not on a
metered GPU. ``infer_reference_downscale_factor`` is the CPU-pure mirror of the canonical
``VideoToVideoStrategy._infer_reference_downscale_factor`` (RESEARCH.md 357-367): 1 on the 1:1 path
(D-7-REF11), a uniform integer factor otherwise, and a clear ValueError on any mismatch.

Latent shapes are ``[C, F, H, W]`` (channels, frames, height, width). CPU-pure — no ltx_core / modal.
"""

from __future__ import annotations

import sys

import pytest

from signet_trainer.dryrun.shapes import (
    assert_paired_reference,
    infer_reference_downscale_factor,
)

# A concrete [C, F, H, W] latent shape for the 832x480x25 demo bucket:
#   lat_f = (25-1)//8+1 = 4, lat_h = 480//32 = 15, lat_w = 832//32 = 26.
REF_1TO1 = (128, 4, 15, 26)
TGT_1TO1 = (128, 4, 15, 26)


# -- infer_reference_downscale_factor -----------------------------------------------------------


def test_infer_factor_1_for_matched_pair():
    # D-7-REF11: identical dims -> factor 1.
    assert infer_reference_downscale_factor((15, 26), (15, 26)) == 1


def test_infer_factor_2_for_uniform_double():
    # Uniform 2x integer multiple -> factor 2.
    assert infer_reference_downscale_factor((15, 26), (30, 52)) == 2


def test_infer_raises_on_non_integer_multiple():
    with pytest.raises(ValueError, match="not an exact integer multiple"):
        infer_reference_downscale_factor((15, 26), (16, 26))


def test_infer_raises_on_non_uniform_scale():
    # H doubles, W unchanged -> non-uniform.
    with pytest.raises(ValueError, match="uniform"):
        infer_reference_downscale_factor((15, 26), (30, 26))


# -- assert_paired_reference --------------------------------------------------------------------


def test_matched_1to1_pair_passes_and_infers_factor_1():
    # A matched 1:1 pair passes with downscale_factor 1 (D-7-REF11) and infers factor 1.
    assert_paired_reference(REF_1TO1, TGT_1TO1, downscale_factor=1)
    assert infer_reference_downscale_factor(REF_1TO1[-2:], TGT_1TO1[-2:]) == 1


def test_uniform_downscale_2_pair_passes():
    # A ref-half-res pair passes when the declared factor matches the inferred uniform 2x.
    ref = (128, 4, 15, 26)
    tgt = (128, 4, 30, 52)
    assert_paired_reference(ref, tgt, downscale_factor=2)


def test_resolution_mismatch_raises():
    # Height differs by a non-integer relation -> rejected before spend.
    ref = (128, 4, 15, 26)
    tgt = (128, 4, 16, 26)
    with pytest.raises(ValueError):
        assert_paired_reference(ref, tgt, downscale_factor=1)


def test_length_mismatch_raises():
    # Frame counts differ -> the reference is not temporally aligned; rejected.
    ref = (128, 4, 15, 26)
    tgt = (128, 7, 15, 26)
    with pytest.raises(ValueError, match="length mismatch"):
        assert_paired_reference(ref, tgt, downscale_factor=1)


def test_non_uniform_scale_raises():
    # H scale 2, W scale 1 -> non-uniform; rejected.
    ref = (128, 4, 15, 26)
    tgt = (128, 4, 30, 26)
    with pytest.raises(ValueError, match="uniform"):
        assert_paired_reference(ref, tgt, downscale_factor=2)


def test_declared_factor_mismatch_raises():
    # A 1:1 pair declared as downscale=2 (or a 2x pair declared as 1) is a config/data disagreement.
    with pytest.raises(ValueError, match="declared reference_downscale_factor"):
        assert_paired_reference(REF_1TO1, TGT_1TO1, downscale_factor=2)


def test_preflight_helpers_are_pure_no_heavy_imports():
    # The preflight path must stay CPU-pure (Anti-Pattern 6 / Pitfall 4).
    assert_paired_reference(REF_1TO1, TGT_1TO1, downscale_factor=1)
    infer_reference_downscale_factor((15, 26), (15, 26))
    assert "modal" not in sys.modules
    assert "ltx_core" not in sys.modules
    assert "ltx_trainer" not in sys.modules
