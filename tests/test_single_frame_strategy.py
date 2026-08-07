"""SC#1 CPU proof for ``SingleFrameStrategy`` — the first concrete ``TrainingStrategy``.

Drives the REAL ``SingleFrameStrategy.prepare_training_inputs`` + ``compute_loss`` on a synthetic
batch with a stubbed ``StepDeps`` (mirroring ``test_training_step.py``) — zero ltx_core, zero CUDA,
Windows/CI-free. Proves the eight §Seam-8 invariants:

    (a) conditioning region == first (H//32)*(W//32) tokens at p=1.0
    (b) timesteps[cond] == 0; non-cond tokens carry the sampled sigma
    (c) noisy[cond] == clean[cond] (clean-latent substitution via torch.where)
    (d) video_loss_mask == ~cond_mask (bool)
    (e) compute_loss invariant to model_output values on conditioning tokens (masked off)
    (f) p=0 reproduces the Phase-3 velocity-objective loss within 1e-6 (byte-equivalence)
    (g) RoPE grid dims come from the LATENT tensor shape, never pixel metadata (the OOM guard)
    (h) compute_loss returns a 0-dim scalar
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from signet_trainer.conditioning.single_frame import SingleFrameStrategy
from signet_trainer.conditioning.strategy import ModelInputs
from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import StepDeps


# --------------------------------------------------------------------------------------------------
# CPU stubs for the ltx_core seam (copied from test_training_step.py — same stubbed-StepDeps harness)
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


def _make_strategy(latents: torch.Tensor, *, p: float, strength: float = 1.0) -> SingleFrameStrategy:
    f, h, w = latents.shape[-3:]
    holder = {"seq_len": f * h * w}
    deps = _make_deps(holder)
    schedule = FlowMatchingSchedule(uniform_prob=0.30)
    return SingleFrameStrategy(
        deps,
        schedule,
        first_frame_conditioning_p=p,
        first_frame_conditioning_strength=strength,
        device="cpu",
        dtype=torch.float32,
    )


def _prepare(latents: torch.Tensor, *, p: float, seed: int = 42, sample=None):
    strat = _make_strategy(latents, p=p)
    if sample is None:
        sample = _sample(latents)
    inputs = strat.prepare_training_inputs(sample, rng=np.random.default_rng(seed))
    return strat, inputs


# --------------------------------------------------------------------------------------------------
# (a) conditioning region == first (H//32)*(W//32) tokens at p=1.0
# --------------------------------------------------------------------------------------------------


def test_conditioning_region_is_first_latent_frame_at_p1() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 2, 2)  # [C=2, F=3, lat_h=2, lat_w=2] -> seq=12, cond=4
    _, inputs = _prepare(latents, p=1.0)

    cond_mask = inputs.video.timesteps == 0
    lat_h, lat_w = 2, 2
    first_frame_end_idx = lat_h * lat_w
    assert int(cond_mask.sum()) == first_frame_end_idx == 4
    # Exactly the FIRST latent-frame tokens are conditioning; the rest are targets.
    assert bool(cond_mask[:, :first_frame_end_idx].all())
    assert not bool(cond_mask[:, first_frame_end_idx:].any())


def test_no_conditioning_when_region_spans_whole_sequence() -> None:
    """Single-latent-frame clips (first_frame_end_idx == seq_len) get NO conditioning (guard)."""
    torch.manual_seed(0)
    latents = torch.randn(2, 1, 2, 2)  # F=1 -> first_frame_end_idx == seq_len -> guard trips
    _, inputs = _prepare(latents, p=1.0)
    cond_mask = inputs.video.timesteps == 0
    assert int(cond_mask.sum()) == 0


# --------------------------------------------------------------------------------------------------
# (b) timesteps[cond]==0; non-cond tokens carry the sampled sigma
# --------------------------------------------------------------------------------------------------


def test_timesteps_zero_on_cond_sigma_elsewhere() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 2, 2)
    _, inputs = _prepare(latents, p=1.0)

    ts = inputs.video.timesteps
    sigma = inputs.video.sigma  # [B]
    cond_mask = ts == 0
    assert torch.all(ts[cond_mask] == 0)

    non_cond = ~cond_mask
    sigma_broadcast = sigma.view(-1, 1).expand_as(ts)
    assert torch.allclose(ts[non_cond], sigma_broadcast[non_cond])


# --------------------------------------------------------------------------------------------------
# (c) noisy[cond] == clean[cond] (clean-latent substitution)
# --------------------------------------------------------------------------------------------------


def test_clean_latent_substitution_on_cond_tokens() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 2, 2)
    _, inputs = _prepare(latents, p=1.0)

    clean = _identity_patchify(latents.unsqueeze(0))  # [B, seq, C]
    noisy = inputs.video.latent
    cond = (inputs.video.timesteps == 0).unsqueeze(-1).expand_as(noisy)
    assert torch.allclose(noisy[cond], clean[cond])


# --------------------------------------------------------------------------------------------------
# (d) video_loss_mask == ~cond_mask
# --------------------------------------------------------------------------------------------------


def test_loss_mask_is_inverse_of_cond_mask() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 2, 2)
    _, inputs = _prepare(latents, p=1.0)

    cond_mask = inputs.video.timesteps == 0
    assert inputs.video_loss_mask.dtype == torch.bool
    assert torch.equal(inputs.video_loss_mask, ~cond_mask)


# --------------------------------------------------------------------------------------------------
# (e) compute_loss invariant to model_output on conditioning tokens + (h) scalar
# --------------------------------------------------------------------------------------------------


def test_compute_loss_ignores_conditioning_token_predictions() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 2, 2)
    strat, inputs = _prepare(latents, p=1.0)

    pred = torch.randn_like(inputs.video_targets)
    loss_a = strat.compute_loss(inputs, pred)

    cond_mask = inputs.video.timesteps == 0
    pred_perturbed = pred.clone()
    pred_perturbed[cond_mask] += 1000.0  # smash predictions on conditioning tokens only
    loss_b = strat.compute_loss(inputs, pred_perturbed)

    assert torch.allclose(loss_a, loss_b)
    assert loss_a.dim() == 0  # (h) scalar for loop.py backward()


# --------------------------------------------------------------------------------------------------
# (f) p=0 byte-equivalence to the Phase-3 velocity objective (within 1e-6)
# --------------------------------------------------------------------------------------------------


def test_p0_byte_equivalent_to_velocity_objective() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 1, 2, 2)  # [C=2, F=1, H=2, W=2] — mirrors test_training_step.py
    strat, inputs = _prepare(latents, p=0.0)

    model = _IdentityModel()
    video_pred, _ = model(video=inputs.video, audio=None, perturbations=None)
    loss = strat.compute_loss(inputs, video_pred)

    assert loss.dtype == torch.float32
    assert loss.dim() == 0

    # Recompute the expected velocity-objective loss exactly as test_training_step.py:127-135
    # (invert the flow-matching interpolation from the captured noisy latent).
    noisy = inputs.video.latent  # pred == noisy (identity model)
    sigma = inputs.video.sigma.view(-1, 1, 1)
    clean = _identity_patchify(latents.unsqueeze(0))
    noise = (noisy - (1 - sigma) * clean) / sigma
    target = noise - clean
    expected = (noisy.float() - target.float()).pow(2).mean()

    assert torch.allclose(loss, expected, atol=1e-6)


# --------------------------------------------------------------------------------------------------
# (g) RoPE grid dims come from the LATENT tensor shape (the 166.9 GiB OOM guard)
# --------------------------------------------------------------------------------------------------


def test_grid_dims_are_latent_shape() -> None:
    torch.manual_seed(0)
    lat_f, lat_h, lat_w = 2, 3, 4
    latents = torch.randn(2, lat_f, lat_h, lat_w)  # [C, F, H, W]
    _prepare(latents, p=1.0)

    kw = _RecordingShape.last_kwargs
    assert kw is not None
    assert (kw["frames"], kw["height"], kw["width"]) == (lat_f, lat_h, lat_w)


def test_grid_dims_ignore_pixel_metadata() -> None:
    """Even with pixel-space metadata present, the grid dims come from the LATENT tensor shape."""
    torch.manual_seed(0)
    lat_f, lat_h, lat_w = 2, 2, 2
    latents = torch.randn(2, lat_f, lat_h, lat_w)
    sample = _sample(latents)
    sample["latent_conditions"].update({"num_frames": 81, "height": 352, "width": 768, "fps": 24})

    _prepare(latents, p=1.0, sample=sample)

    kw = _RecordingShape.last_kwargs
    assert (kw["frames"], kw["height"], kw["width"]) == (lat_f, lat_h, lat_w)


# --------------------------------------------------------------------------------------------------
# ABC contract: prepare_training_inputs returns a fully-populated (video-only) ModelInputs
# --------------------------------------------------------------------------------------------------


def test_prepare_returns_video_only_model_inputs() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 2, 2)
    strat, inputs = _prepare(latents, p=1.0)

    assert isinstance(inputs, ModelInputs)
    assert inputs.audio is None
    assert inputs.audio_targets is None
    assert inputs.audio_loss_mask is None
    assert strat.get_data_sources() == ["latents", "conditions"]
