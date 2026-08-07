"""SC#1 CPU proof for ``MultiFrameStrategy`` — the N-region reference-control training core.

Drives the REAL ``MultiFrameStrategy.prepare_training_inputs`` + ``compute_loss`` on a synthetic
batch with a stubbed ``StepDeps`` (mirroring ``test_single_frame_strategy.py``) — zero ltx_core,
zero CUDA, Windows/CI-free. Proves the REF-02 / SC#1 invariants:

    - mask_ops helpers (``-k mask_ops`` selects these): span math, ``denoise_mask = 1 - strength``,
      per-token ``sigma * denoise_mask``, single-latent-frame guard, latent->pixel %8 boundary.
    - N-item mask: each region spans exactly ``lat_h*lat_w`` tokens; ``denoise_mask`` in [0, 1].
    - Per-token sigma == ``sigma * denoise_mask`` (strength 1 => hard 0-clean).
    - Explicit ``video_loss_mask == (denoise_mask > 0)`` — partial-strength tokens participate in
      loss (Pitfall 1 break-guard: a ``timesteps != 0`` token is still ``True`` in the mask).
    - EQUIVALENCE: ``MultiFrameStrategy(count=1, idx=0, strength=1)`` is byte-equal (atol 1e-6) to
      ``SingleFrameStrategy(first_frame_conditioning_p=1.0)`` — D-6-SIBLING, single_frame.py untouched.
    - ``compute_loss`` returns a 0-dim scalar; RoPE grid dims come from the latent tensor shape.
    - SC#1 %8: every strategy-sampled position maps to a pixel frame that is ``% TIME_SCALE == 0``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from signet_trainer.conditioning.multi_frame import MultiFrameStrategy
from signet_trainer.conditioning.single_frame import SingleFrameStrategy
from signet_trainer.conditioning.strategy import TIME_SCALE, ModelInputs
from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import StepDeps


# --------------------------------------------------------------------------------------------------
# CPU stubs for the ltx_core seam (copied from test_single_frame_strategy.py — same harness)
# --------------------------------------------------------------------------------------------------


def _identity_patchify(latent_5d: torch.Tensor) -> torch.Tensor:
    """A faithful ``patch_size=1`` patchify: [B, C, F, H, W] -> [B, F*H*W, C]."""
    b, c, f, h, w = latent_5d.shape
    return latent_5d.reshape(b, c, f * h * w).permute(0, 2, 1).contiguous()


class _RecordingShape:
    """Stub ``VideoLatentShape`` that records the (latent) dims it is built with."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        _RecordingShape.last_kwargs = kwargs
        self.kwargs = kwargs


class _StubPatchifier:
    def patchify(self, latent_5d: torch.Tensor) -> torch.Tensor:
        return _identity_patchify(latent_5d)

    def get_patch_grid_bounds(self, output_shape, device):  # noqa: ANN001
        return torch.zeros(1)


def _make_deps(seq_len_holder: dict) -> StepDeps:
    def _get_pixel_coords(latent_coords, scale_factors, causal_fix):  # noqa: ANN001
        seq = seq_len_holder["seq_len"]
        return torch.zeros(1, 3, seq, 2)

    return StepDeps(
        patchifier=_StubPatchifier(),
        get_pixel_coords=_get_pixel_coords,
        scale_factors=None,
        video_latent_shape_cls=_RecordingShape,
        modality_cls=lambda **kw: SimpleNamespace(**kw),
    )


class _IdentityModel:
    """Returns ``video.latent`` unchanged (pred == noisy) — the Phase-3 regression net."""

    def __init__(self) -> None:
        self.seen = None

    def __call__(self, *, video, audio, perturbations):  # noqa: ANN001
        self.seen = video
        return video.latent, None


class _FixedSchedule:
    """A schedule stub that returns a constant sigma for every sample, ignoring ``rng``.

    Used to hold sigma fixed across strategies so equivalence/mask assertions compare pure mask
    math (numpy rng is then consumed ONLY by the multi-frame position/strength draws, not sigma).
    """

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def sample_timesteps(self, batch_size, seq_len, rng):  # noqa: ANN001
        return np.full(batch_size, self.sigma, dtype=np.float64)


class _FakeRNG:
    """Deterministic rng for forcing exact conditioning positions/strength in mask tests.

    Only implements the surface ``sample_conditioning_positions`` + the strength draw touch
    (``integers`` -> K, ``choice`` -> positions, ``uniform`` -> strength). It is paired with a
    ``_FixedSchedule`` so the schedule never calls into it.
    """

    def __init__(self, *, k: int, positions, strength: float) -> None:
        self._k = k
        self._positions = list(positions)
        self._strength = strength

    def integers(self, low, high):  # noqa: ANN001
        return self._k

    def choice(self, a, size=None, replace=True):  # noqa: ANN001
        n = size if size is not None else 1
        return np.array(self._positions[:n])

    def uniform(self, low, high):  # noqa: ANN001
        return self._strength


# --------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------


def _sample(latents: torch.Tensor, *, n_text: int = 4, ctx_dim: int = 8, mask=None) -> dict:
    if mask is None:
        mask = torch.ones(n_text, dtype=torch.long)
    return {
        "latent_conditions": {"latents": latents},
        "text_conditions": {
            "video_prompt_embeds": torch.randn(n_text, ctx_dim),
            "prompt_attention_mask": mask,
        },
        "idx": 0,
    }


def _make_multi(
    latents: torch.Tensor,
    *,
    max_items: int = 1,
    strength_range=(0.3, 1.0),
    schedule=None,
) -> MultiFrameStrategy:
    f, h, w = latents.shape[-3:]
    holder = {"seq_len": f * h * w}
    deps = _make_deps(holder)
    if schedule is None:
        schedule = FlowMatchingSchedule(uniform_prob=0.30)
    return MultiFrameStrategy(
        deps,
        schedule,
        max_conditioning_items=max_items,
        conditioning_strength_range=strength_range,
        device="cpu",
        dtype=torch.float32,
    )


def _make_single(latents: torch.Tensor, *, p: float, schedule=None) -> SingleFrameStrategy:
    f, h, w = latents.shape[-3:]
    holder = {"seq_len": f * h * w}
    deps = _make_deps(holder)
    if schedule is None:
        schedule = FlowMatchingSchedule(uniform_prob=0.30)
    return SingleFrameStrategy(
        deps,
        schedule,
        first_frame_conditioning_p=p,
        device="cpu",
        dtype=torch.float32,
    )


# --------------------------------------------------------------------------------------------------
# mask_ops helper unit tests (marked so ``-k mask_ops`` selects them)
# --------------------------------------------------------------------------------------------------


def test_mask_ops_latent_frame_token_span() -> None:
    from signet_trainer.conditioning.mask_ops import latent_frame_token_span

    assert latent_frame_token_span(0, 2, 3) == (0, 6)
    assert latent_frame_token_span(2, 2, 3) == (12, 18)
    assert latent_frame_token_span(1, 4, 4) == (16, 32)


def test_mask_ops_build_denoise_mask() -> None:
    from signet_trainer.conditioning.mask_ops import build_denoise_mask

    # strength 1.0 => 1 - 1 = 0 on the region, 1.0 elsewhere.
    m = build_denoise_mask(1, 12, [(0, 4, 1.0)], device="cpu")
    assert m.shape == (1, 12)
    assert torch.allclose(m[0, :4], torch.zeros(4))
    assert torch.allclose(m[0, 4:], torch.ones(8))

    # partial strength 0.5 => 0.5 on the region.
    m2 = build_denoise_mask(2, 12, [(0, 4, 0.5)], device="cpu")
    assert torch.allclose(m2[:, :4], torch.full((2, 4), 0.5))
    assert torch.allclose(m2[:, 4:], torch.ones(2, 8))
    assert bool((m2 >= 0).all() and (m2 <= 1).all())  # in [0, 1]

    # multiple regions.
    m3 = build_denoise_mask(1, 12, [(0, 3, 1.0), (6, 9, 0.25)], device="cpu")
    assert torch.allclose(m3[0, :3], torch.zeros(3))
    assert torch.allclose(m3[0, 6:9], torch.full((3,), 0.75))
    assert torch.allclose(m3[0, 3:6], torch.ones(3))
    assert torch.allclose(m3[0, 9:], torch.ones(3))


def test_mask_ops_per_token_sigma() -> None:
    from signet_trainer.conditioning.mask_ops import build_denoise_mask, per_token_sigma

    denoise = build_denoise_mask(2, 6, [(0, 3, 1.0)], device="cpu")  # 0 on [0:3]
    sigma = torch.tensor([0.4, 0.7])
    t = per_token_sigma(sigma, denoise)
    assert t.shape == (2, 6)
    assert torch.allclose(t[:, :3], torch.zeros(2, 3))  # strength 1 => 0
    assert torch.allclose(t[0, 3:], torch.full((3,), 0.4))
    assert torch.allclose(t[1, 3:], torch.full((3,), 0.7))


def test_mask_ops_sample_conditioning_positions_single_frame_guard() -> None:
    from signet_trainer.conditioning.mask_ops import sample_conditioning_positions

    rng = np.random.default_rng(0)
    assert sample_conditioning_positions(1, 3, rng) == []


def test_mask_ops_sample_conditioning_positions_bounds() -> None:
    from signet_trainer.conditioning.mask_ops import sample_conditioning_positions

    rng = np.random.default_rng(0)
    for _ in range(100):
        pos = sample_conditioning_positions(5, 3, rng)
        assert 1 <= len(pos) <= 3
        assert len(set(pos)) == len(pos)  # distinct
        assert all(0 <= p <= 4 for p in pos)


def test_mask_ops_sample_conditioning_positions_clamps_to_available_frames() -> None:
    from signet_trainer.conditioning.mask_ops import sample_conditioning_positions

    rng = np.random.default_rng(1)
    for _ in range(100):
        pos = sample_conditioning_positions(2, 5, rng)  # max_items > lat_frames
        assert 1 <= len(pos) <= 2
        assert len(set(pos)) == len(pos)


def test_mask_ops_pixel_frame_for_latent_idx() -> None:
    from signet_trainer.conditioning.mask_ops import pixel_frame_for_latent_idx

    for idx in range(10):
        pf = pixel_frame_for_latent_idx(idx)
        assert pf == idx * TIME_SCALE
        assert pf % TIME_SCALE == 0


def test_mask_ops_is_import_confined() -> None:
    """mask_ops must import torch + stdlib only (Pitfall 6) — no modal/ltx at module top."""
    import signet_trainer.conditioning.mask_ops as mo

    src = mo.__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    # crude module-top scan: no top-level modal/ltx imports (function-local numpy is allowed)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
            assert "modal" not in stripped
            assert "ltx" not in stripped


# --------------------------------------------------------------------------------------------------
# N-item mask + per-token sigma + explicit loss mask
# --------------------------------------------------------------------------------------------------


def test_n_item_mask_two_regions_partial_strength_in_loss() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 4, 2, 3)  # C=2, F=4, lat_h=2, lat_w=3 -> seq=24, per-frame=6
    strat = _make_multi(
        latents, max_items=2, strength_range=(0.5, 0.5), schedule=_FixedSchedule(0.6)
    )
    rng = _FakeRNG(k=2, positions=[1, 3], strength=0.5)
    inputs = strat.prepare_training_inputs(_sample(latents), rng=rng)

    # frames 1 and 3 -> tokens [6:12] and [18:24]; each region spans exactly lat_h*lat_w = 6.
    region = torch.zeros(24, dtype=torch.bool)
    region[6:12] = True
    region[18:24] = True
    assert int(region.sum()) == 12  # 2 regions * (lat_h*lat_w)

    ts = inputs.video.timesteps  # per-token sigma = sigma * denoise_mask
    # region: sigma(0.6) * denoise(1 - 0.5 = 0.5) = 0.3 ; elsewhere sigma * 1.0 = 0.6
    assert torch.allclose(ts[:, region], torch.full((2, 12), 0.3), atol=1e-6)
    assert torch.allclose(ts[:, ~region], torch.full((2, 12), 0.6), atol=1e-6)

    # explicit loss mask: partial-strength tokens (denoise 0.5 > 0) PARTICIPATE in loss.
    assert inputs.video_loss_mask.dtype == torch.bool
    assert bool(inputs.video_loss_mask.all())


def test_partial_strength_token_has_nonzero_timestep_yet_in_loss_mask() -> None:
    """Pitfall-1 break guard: a token with ``timesteps != 0`` must still be ``True`` in the mask."""
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 2, 2)  # seq=12, per-frame=4
    strat = _make_multi(
        latents, max_items=1, strength_range=(0.5, 0.5), schedule=_FixedSchedule(0.8)
    )
    rng = _FakeRNG(k=1, positions=[1], strength=0.5)
    inputs = strat.prepare_training_inputs(_sample(latents), rng=rng)

    ts = inputs.video.timesteps
    lm = inputs.video_loss_mask
    region = slice(4, 8)  # frame 1
    assert torch.all(ts[:, region] != 0)  # partial strength => nonzero per-token sigma
    assert bool(lm[:, region].all())  # yet still IN the loss (denoise 0.5 > 0)
    assert bool(lm.all())  # nothing is hard-clean here


def test_loss_mask_excludes_only_hard_clean_strength1() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 2, 2)  # seq=12, per-frame=4
    strat = _make_multi(
        latents, max_items=1, strength_range=(1.0, 1.0), schedule=_FixedSchedule(0.7)
    )
    rng = _FakeRNG(k=1, positions=[1], strength=1.0)
    inputs = strat.prepare_training_inputs(_sample(latents), rng=rng)

    lm = inputs.video_loss_mask
    region = torch.zeros(12, dtype=torch.bool)
    region[4:8] = True
    assert not bool(lm[:, region].any())  # hard-clean strength-1 tokens excluded
    assert bool(lm[:, ~region].all())  # everything else in loss
    # and video_loss_mask == (denoise_mask > 0): timesteps==0 exactly on the excluded region.
    assert torch.equal(inputs.video_loss_mask, inputs.video.timesteps > 0)


def test_single_latent_frame_gets_no_conditioning() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 1, 2, 2)  # F=1 -> sample_conditioning_positions returns []
    strat = _make_multi(latents, max_items=3, schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(latents), rng=np.random.default_rng(0))
    # no regions -> denoise_mask all 1 -> per-token sigma == sigma everywhere -> all in loss.
    assert bool(inputs.video_loss_mask.all())
    assert torch.allclose(inputs.video.timesteps, torch.full_like(inputs.video.timesteps, 0.5))


# --------------------------------------------------------------------------------------------------
# EQUIVALENCE — multi(count=1, idx=0, strength=1) == single(p=1.0), single_frame.py untouched
# --------------------------------------------------------------------------------------------------


def test_equivalence_to_single_frame_at_count1_idx0_strength1() -> None:
    latents = torch.randn(2, 3, 2, 2)  # C=2, F=3, H=2, W=2 -> seq=12, first-frame span=4
    sigma_val = 0.35

    # SingleFrameStrategy(p=1.0) — hard-clean first-frame conditioning.
    sf = _make_single(latents, p=1.0, schedule=_FixedSchedule(sigma_val))
    torch.manual_seed(123)
    sf_inputs = sf.prepare_training_inputs(_sample(latents), rng=np.random.default_rng(7))

    # MultiFrameStrategy(count=1, idx=0, strength=1) — same clean first-frame region.
    mf = _make_multi(
        latents, max_items=1, strength_range=(1.0, 1.0), schedule=_FixedSchedule(sigma_val)
    )
    torch.manual_seed(123)  # identical torch seed => identical noise (first randn_like)
    mf_inputs = mf.prepare_training_inputs(
        _sample(latents), rng=_FakeRNG(k=1, positions=[0], strength=1.0)
    )

    assert torch.allclose(sf_inputs.video.latent, mf_inputs.video.latent, atol=1e-6)
    assert torch.allclose(sf_inputs.video.timesteps, mf_inputs.video.timesteps, atol=1e-6)
    assert torch.equal(sf_inputs.video_loss_mask, mf_inputs.video_loss_mask)
    assert torch.allclose(sf_inputs.video_targets, mf_inputs.video_targets, atol=1e-6)

    pred = torch.randn_like(sf_inputs.video_targets)
    loss_sf = sf.compute_loss(sf_inputs, pred)
    loss_mf = mf.compute_loss(mf_inputs, pred)
    assert torch.allclose(loss_sf, loss_mf, atol=1e-6)
    assert loss_mf.dim() == 0


# --------------------------------------------------------------------------------------------------
# compute_loss scalar + RoPE grid dims from latent shape (the OOM guard)
# --------------------------------------------------------------------------------------------------


def test_compute_loss_scalar_and_grid_dims_from_latent_shape() -> None:
    torch.manual_seed(0)
    lat_f, lat_h, lat_w = 3, 3, 4
    latents = torch.randn(2, lat_f, lat_h, lat_w)  # [C, F, H, W]
    strat = _make_multi(latents, max_items=2)
    inputs = strat.prepare_training_inputs(_sample(latents), rng=np.random.default_rng(1))

    pred = torch.randn_like(inputs.video_targets)
    loss = strat.compute_loss(inputs, pred)
    assert loss.dim() == 0  # 0-dim scalar

    kw = _RecordingShape.last_kwargs
    assert kw is not None
    assert (kw["frames"], kw["height"], kw["width"]) == (lat_f, lat_h, lat_w)


def test_prepare_returns_video_only_model_inputs() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 2, 2)
    strat = _make_multi(latents, max_items=2)
    inputs = strat.prepare_training_inputs(_sample(latents), rng=np.random.default_rng(0))

    assert isinstance(inputs, ModelInputs)
    assert inputs.audio is None
    assert inputs.audio_targets is None
    assert inputs.audio_loss_mask is None
    assert strat.get_data_sources() == ["latents", "conditions"]


# --------------------------------------------------------------------------------------------------
# SC#1 %8 — every strategy-sampled position maps to a %8-aligned pixel frame (asserted in-strategy)
# --------------------------------------------------------------------------------------------------


def test_sc1_sampled_positions_map_to_8_aligned_pixel_frames() -> None:
    from signet_trainer.conditioning.mask_ops import (
        pixel_frame_for_latent_idx,
        sample_conditioning_positions,
    )

    # direct: every sampled latent position -> pixel frame % TIME_SCALE == 0, across many seeds.
    for seed in range(50):
        rng = np.random.default_rng(seed)
        for idx in sample_conditioning_positions(6, 3, rng):
            assert pixel_frame_for_latent_idx(idx) % TIME_SCALE == 0

    # in-strategy: driving the strategy across seeds never trips its internal %8 alignment assert.
    torch.manual_seed(0)
    latents = torch.randn(2, 6, 2, 2)  # 6 latent frames
    for seed in range(50):
        strat = _make_multi(latents, max_items=3)
        strat.prepare_training_inputs(_sample(latents), rng=np.random.default_rng(seed))
