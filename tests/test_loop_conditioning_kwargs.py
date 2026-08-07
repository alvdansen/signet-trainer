"""CR-01 — ``build_conditioning_kwargs`` threads each mode's knobs into ``training_step`` (CPU only).

The training loop builds the per-step conditioning kwargs ONCE from the static conditioning config
(Seam D). CR-01: the ``ic_lora`` branch was missing, so a valid ``reference_downscale_factor: 2``
config silently trained at the signature default 1 (unscaled reference RoPE positions) on a metered
A100 with no error anywhere. These proofs lock the mode-dispatched threading — including the
deliberate NON-threading of ``first_frame_conditioning_p`` in ic_lora mode (its schema default 1.0
would flip ic_lora to always-condition-first-frame; ic_lora keeps the strategy's 0.0).

CPU only — the real ``ConditioningConfig`` builds without ltx_core / modal / CUDA.
"""

from __future__ import annotations

from signet_trainer.config.schema import ConditioningConfig
from signet_trainer.train.loop import build_conditioning_kwargs


def test_none_mode_threads_only_mode():
    kwargs = build_conditioning_kwargs(ConditioningConfig(mode="none"))
    assert kwargs == {"conditioning_mode": "none"}


def test_single_frame_threads_first_frame_p():
    c = ConditioningConfig(mode="single_frame", first_frame_conditioning_p=0.5)
    kwargs = build_conditioning_kwargs(c)
    assert kwargs["conditioning_mode"] == "single_frame"
    assert kwargs["first_frame_conditioning_p"] == 0.5
    assert "reference_downscale_factor" not in kwargs


def test_multi_frame_threads_strength_primitives():
    c = ConditioningConfig(
        mode="multi_frame",
        max_conditioning_items=3,
        conditioning_strength_range=(0.2, 0.9),
    )
    kwargs = build_conditioning_kwargs(c)
    assert kwargs["conditioning_mode"] == "multi_frame"
    assert kwargs["max_conditioning_items"] == 3
    assert kwargs["conditioning_strength_range"] == (0.2, 0.9)
    assert "first_frame_conditioning_p" not in kwargs


def test_ic_lora_threads_reference_downscale_factor():
    # CR-01: a non-default reference_downscale_factor MUST reach training_step (not silently drop
    # to the signature default 1).
    c = ConditioningConfig(mode="ic_lora", reference_downscale_factor=2)
    kwargs = build_conditioning_kwargs(c)
    assert kwargs["conditioning_mode"] == "ic_lora"
    assert kwargs["reference_downscale_factor"] == 2


def test_ic_lora_does_not_thread_first_frame_p():
    # ic_lora must NOT thread first_frame_conditioning_p — its schema default (1.0) would flip
    # ic_lora training to always-condition-first-frame. ic_lora keeps the strategy's 0.0.
    c = ConditioningConfig(mode="ic_lora", reference_downscale_factor=1)
    kwargs = build_conditioning_kwargs(c)
    assert "first_frame_conditioning_p" not in kwargs
    assert "first_frame_conditioning_strength" not in kwargs
