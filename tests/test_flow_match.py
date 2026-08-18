"""TRAIN-02 — the LTX-2.3 flow-matching objective + shift-lerp timestep sampler.

CPU unit tests for ``train/flow_match.py::FlowMatchingSchedule`` (ported verbatim from
enochiatron ``scripts/train/train.py`` lines 312-421). Proves the LTX velocity objective
``target = noise - clean`` (NOT Wan v-prediction) and ``noisy = (1-t)*clean + t*noise``
(flow-match interpolation, NOT DDPM add), the seq-len shift lerp 0.95->2.05 over
[1024, 4096], and the deterministic 30/70 uniform/logit-normal sampler.

Anti-Pattern 6: importing ``train.flow_match`` must NOT pull modal / ltx_core / ltx_trainer.
"""

from __future__ import annotations

import numpy as np
import torch

from signet_trainer.train.flow_match import FlowMatchingSchedule


# --------------------------------------------------------------------------------------------------
# objective statics — the LTX velocity target + flow-match interpolation
# --------------------------------------------------------------------------------------------------


def test_compute_noisy_latent_endpoints():
    # noisy = (1 - t) * clean + t * noise -> t=0 is clean, t=1 is noise.
    clean = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)

    at_zero = FlowMatchingSchedule.compute_noisy_latent(clean, noise, 0.0)
    at_one = FlowMatchingSchedule.compute_noisy_latent(clean, noise, 1.0)

    assert torch.equal(at_zero, clean)
    assert torch.equal(at_one, noise)


def test_compute_noisy_latent_midpoint():
    clean = torch.zeros(4)
    noise = torch.ones(4)
    mid = FlowMatchingSchedule.compute_noisy_latent(clean, noise, 0.5)
    assert torch.allclose(mid, torch.full((4,), 0.5))


def test_compute_target_is_velocity_not_vpred():
    # The LTX objective is exactly noise - clean (velocity), NOT Wan v-prediction.
    clean = torch.randn(3, 5)
    noise = torch.randn(3, 5)
    target = FlowMatchingSchedule.compute_target(clean, noise)
    assert torch.equal(target, noise - clean)


# --------------------------------------------------------------------------------------------------
# compute_shift — the seq-len-dependent lerp, clamped to [min_tokens, max_tokens]
# --------------------------------------------------------------------------------------------------


def test_compute_shift_endpoints_and_midpoint():
    s = FlowMatchingSchedule()
    assert s.compute_shift(1024) == np.float64(0.95)
    assert s.compute_shift(4096) == np.float64(2.05)
    # Lerp midpoint: (4096 + 1024) / 2 = 2560 -> halfway between 0.95 and 2.05 = 1.50.
    assert abs(s.compute_shift(2560) - 1.50) < 1e-9


def test_compute_shift_is_clamped_outside_range():
    s = FlowMatchingSchedule()
    assert s.compute_shift(0) == np.float64(0.95)       # below min_tokens -> clamp low
    assert s.compute_shift(512) == np.float64(0.95)
    assert s.compute_shift(8192) == np.float64(2.05)    # above max_tokens -> clamp high


# --------------------------------------------------------------------------------------------------
# sample_timesteps — range, determinism, and the configurable uniform fraction
# --------------------------------------------------------------------------------------------------


def test_sample_timesteps_in_range():
    s = FlowMatchingSchedule()
    ts = s.sample_timesteps(10000, 2688, np.random.default_rng(42))
    assert ts.min() >= 0.001
    assert ts.max() <= 0.999


def test_sample_timesteps_is_deterministic_under_seed():
    s = FlowMatchingSchedule()
    a = s.sample_timesteps(5000, 2688, np.random.default_rng(42))
    b = s.sample_timesteps(5000, 2688, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_uniform_branch_fraction_matches_uniform_prob():
    # Make the two branches separable: a huge shift + tiny std drives every logit-normal
    # sample to sigmoid(~12) -> clipped to exactly 0.999, while the uniform branch is
    # continuous in [0.001, 0.999). So the fraction of samples STRICTLY below 0.999 is the
    # uniform-branch fraction, which must track uniform_prob.
    uniform_prob = 0.30
    s = FlowMatchingSchedule(
        min_shift=12.0, max_shift=12.0, std=1e-3, uniform_prob=uniform_prob
    )
    ts = s.sample_timesteps(10000, 2688, np.random.default_rng(42))
    frac_uniform = float((ts < 0.999).mean())
    assert abs(frac_uniform - uniform_prob) < 0.03


def test_constructor_default_uniform_prob_is_source_value():
    # The source __init__ default is 0.1; the locked 0.30 is passed from config in loop.py,
    # NOT hardcoded here (deviation note, PATTERNS.md:56).
    assert FlowMatchingSchedule().uniform_prob == 0.1


# --------------------------------------------------------------------------------------------------
# Issue #13 step 2 — training.timestep_std threading. default=1.0 must be byte-identical to today's
# behavior (two 2026-08-11 operator rulings forbid changing the shipped default); a non-default
# value must actually move the draw, or the knob is a no-op wearing a config field's clothes.
# --------------------------------------------------------------------------------------------------


def test_default_std_is_byte_identical_to_the_unthreaded_constructor():
    """Passing ``std=1.0`` explicitly must reproduce the pre-#13 (no-``std``-arg) draw exactly."""
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    unthreaded = FlowMatchingSchedule(uniform_prob=0.30).sample_timesteps(256, 2688, rng_a)
    threaded_default = FlowMatchingSchedule(uniform_prob=0.30, std=1.0).sample_timesteps(
        256, 2688, rng_b
    )
    assert np.array_equal(unthreaded, threaded_default)


def test_nondefault_std_actually_changes_the_draw():
    """The derived A/B value (``std=1.7``, issue #13 step 3) must produce a DIFFERENT draw."""
    baseline = FlowMatchingSchedule(uniform_prob=0.30, std=1.0).sample_timesteps(
        2000, 2688, np.random.default_rng(42)
    )
    wider = FlowMatchingSchedule(uniform_prob=0.30, std=1.7).sample_timesteps(
        2000, 2688, np.random.default_rng(42)
    )
    assert not np.array_equal(baseline, wider)
