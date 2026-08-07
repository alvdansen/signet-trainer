"""TRAIN-04 — the flow-matching ``training_step`` objective + signet nested-sample unwrap.

CPU unit tests for ``train/step.py``. The heavy ltx_core seam (patchify / RoPE coords /
``Modality``) is injected as a stub ``StepDeps`` so the objective math + the nested-sample unwrap
(Pitfall 2) + the defensive pixel-dim reconstruction (A3) are proven with zero ltx_core and zero
CUDA. With an identity-stub model (``forward`` returns ``video.latent`` unchanged) the returned
loss is exactly ``mean((noisy − (noise − clean))²)`` — recomputed from the captured noisy latent
+ σ + the known patched clean.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import StepDeps, training_step


# --------------------------------------------------------------------------------------------------
# CPU stubs for the ltx_core seam
# --------------------------------------------------------------------------------------------------


def _identity_patchify(latent_5d: torch.Tensor) -> torch.Tensor:
    """A faithful ``patch_size=1`` patchify: [B, C, F, H, W] -> [B, F*H*W, C]."""
    b, c, f, h, w = latent_5d.shape
    return latent_5d.reshape(b, c, f * h * w).permute(0, 2, 1).contiguous()


class _RecordingShape:
    """Stub ``VideoLatentShape`` that records the reconstructed pixel dims it is built with."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        _RecordingShape.last_kwargs = kwargs
        self.kwargs = kwargs


class _StubPatchifier:
    def __init__(self) -> None:
        self.grid_seq_len: int | None = None

    def patchify(self, latent_5d: torch.Tensor) -> torch.Tensor:
        return _identity_patchify(latent_5d)

    def get_patch_grid_bounds(self, output_shape, device):  # noqa: ANN001
        # Return a small placeholder; pixel-coord shape is driven by get_pixel_coords below.
        return torch.zeros(1)


def _make_deps(seq_len_holder: dict) -> StepDeps:
    def _get_pixel_coords(latent_coords, scale_factors, causal_fix):  # noqa: ANN001
        # Shape [B, 3, seq, 2] so the `pixel_coords[:, 0, ...]` temporal-scale write works.
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
    """Captures the Modality it receives and returns ``video.latent`` unchanged (pred = noisy)."""

    def __init__(self) -> None:
        self.seen = None

    def __call__(self, *, video, audio, perturbations):  # noqa: ANN001
        self.seen = video
        return video.latent, None


class _CondPerturbModel:
    """Identity model that additively PERTURBS its prediction on the first ``n_cond`` tokens.

    Used to prove the p=1 loss is invariant to predictions on the conditioning region: those
    tokens are masked out of the loss (``video_loss_mask == ~cond_mask``), so a huge perturbation
    there must leave the loss unchanged.
    """

    def __init__(self, n_cond: int) -> None:
        self.n_cond = n_cond
        self.seen = None

    def __call__(self, *, video, audio, perturbations):  # noqa: ANN001
        self.seen = video
        out = video.latent.clone()
        out[:, : self.n_cond, :] += 1000.0  # blow up the conditioning-region prediction
        return out, None


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


def _run(sample: dict) -> tuple[torch.Tensor, _IdentityModel, StepDeps]:
    # seq_len = F*H*W of the (4D->5D) latent; the stub get_pixel_coords needs it up front.
    lat = sample["latent_conditions"]["latents"]
    f, h, w = lat.shape[-3:]
    holder = {"seq_len": f * h * w}
    deps = _make_deps(holder)
    model = _IdentityModel()
    schedule = FlowMatchingSchedule(uniform_prob=0.30)
    rng = np.random.default_rng(42)
    loss = training_step(
        model, sample, schedule, rng, device="cpu", dtype=torch.float32, deps=deps
    )
    return loss, model, deps


def _run_with_model(
    sample: dict, model: Any, *, p: float = 0.0, torch_seed: int = 0, rng_seed: int = 42
) -> torch.Tensor:
    """Drive ``training_step`` with an explicit model + conditioning-p, re-seeding torch + numpy so
    two invocations that differ ONLY in the model are RNG-identical (noise/sigma/Bernoulli match).

    Selects ``conditioning_mode="single_frame"`` when ``p > 0`` so ``first_frame_conditioning_p``
    actually takes effect under the 06-05 dispatch (``mode="none"`` hard-forces p=0.0 — threat
    T-06-01); ``p == 0`` stays on the byte-equivalent ``none`` path.
    """
    lat = sample["latent_conditions"]["latents"]
    f, h, w = lat.shape[-3:]
    holder = {"seq_len": f * h * w}
    deps = _make_deps(holder)
    schedule = FlowMatchingSchedule(uniform_prob=0.30)
    torch.manual_seed(torch_seed)
    rng = np.random.default_rng(rng_seed)
    return training_step(
        model,
        sample,
        schedule,
        rng,
        device="cpu",
        dtype=torch.float32,
        deps=deps,
        conditioning_mode="single_frame" if p > 0 else "none",
        first_frame_conditioning_p=p,
    )


# --------------------------------------------------------------------------------------------------
# the objective: identity stub ⇒ loss = mean((noisy − (noise − clean))²)
# --------------------------------------------------------------------------------------------------


def test_identity_stub_loss_matches_velocity_objective() -> None:
    torch.manual_seed(0)
    latents = torch.randn(2, 1, 2, 2)  # [C=2, F=1, H=2, W=2]
    loss, model, _ = _run(_sample(latents))

    assert loss.dtype == torch.float32
    assert loss.dim() == 0

    # Recompute the expected loss from the captured noisy latent + σ + the known patched clean.
    noisy = model.seen.latent  # pred == noisy (identity model)
    sigma = model.seen.sigma.view(-1, 1, 1)
    clean = _identity_patchify(latents.unsqueeze(0))  # [B, seq, C]
    noise = (noisy - (1 - sigma) * clean) / sigma  # invert the interpolation
    target = noise - clean
    expected = (noisy.float() - target.float()).pow(2).mean()

    assert torch.allclose(loss, expected, atol=1e-5)


# --------------------------------------------------------------------------------------------------
# Pitfall 2: nested-sample unwrap with no KeyError
# --------------------------------------------------------------------------------------------------


def test_nested_sample_unwrap_no_keyerror() -> None:
    latents = torch.randn(3, 2, 2, 2)  # [C, F, H, W]
    loss, _, _ = _run(_sample(latents, n_text=5, ctx_dim=16))
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------------------------------
# RoPE grid dims: LATENT space (must match video_seq_len), NOT pixel space (the 166.9 GiB OOM bug)
# --------------------------------------------------------------------------------------------------


def test_grid_dims_are_latent_shape() -> None:
    """The RoPE coord grid is built in LATENT space (matching video_seq_len = patchify(latent)).

    Passing pixel dims (lat*compression) explodes the grid by the VAE compression (32x32x8) → a
    ~21.9M-position tensor → 166.9 GiB OOM on the A100 (gate check #4). VideoLatentShape must get the
    latent tensor's own F/H/W — mirroring enochiatron (num_frames/height/width ← video_tensor.shape).
    """
    lat_f, lat_h, lat_w = 2, 3, 4
    latents = torch.randn(2, lat_f, lat_h, lat_w)  # [C, F, H, W]
    _run(_sample(latents))

    kw = _RecordingShape.last_kwargs
    assert kw is not None
    assert (kw["frames"], kw["height"], kw["width"]) == (lat_f, lat_h, lat_w)


def test_grid_dims_ignore_pixel_metadata() -> None:
    """Even with pixel-space num_frames/height/width metadata present, the grid dims come from the
    LATENT tensor shape. Trusting that metadata (pixel-vs-latent ambiguity) is what OOM'd the A100.
    """
    lat_f, lat_h, lat_w = 2, 2, 2
    latents = torch.randn(2, lat_f, lat_h, lat_w)
    sample = _sample(latents)
    sample["latent_conditions"].update({"num_frames": 81, "height": 352, "width": 768, "fps": 24})

    holder = {"seq_len": lat_f * lat_h * lat_w}
    deps = _make_deps(holder)
    schedule = FlowMatchingSchedule(uniform_prob=0.30)
    training_step(
        _IdentityModel(), sample, schedule, np.random.default_rng(42),
        device="cpu", dtype=torch.float32, deps=deps,
    )
    kw = _RecordingShape.last_kwargs
    assert (kw["frames"], kw["height"], kw["width"]) == (lat_f, lat_h, lat_w)


# --------------------------------------------------------------------------------------------------
# additive text mask: −inf where attention==0, else 0
# --------------------------------------------------------------------------------------------------


def test_additive_mask_is_minus_inf_where_attn_zero() -> None:
    latents = torch.randn(2, 1, 2, 2)
    mask = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    _, model, _ = _run(_sample(latents, n_text=4, mask=mask))

    additive = model.seen.context_mask
    assert additive.shape == (1, 4)
    assert additive[0, 0].item() == 0.0
    assert additive[0, 1].item() == 0.0
    assert additive[0, 2].item() == float("-inf")
    assert additive[0, 3].item() == float("-inf")


# --------------------------------------------------------------------------------------------------
# Option A-hybrid retrofit: p=0 is byte-equivalent; p=1 masks the conditioning region
# --------------------------------------------------------------------------------------------------


def test_p0_byte_equivalent_to_velocity_objective() -> None:
    """With ``first_frame_conditioning_p=0.0`` (the loop default) the strategy-routed driver returns
    the SAME analytic velocity loss as the Phase-3 step — the byte-equivalence backward-compat guard.
    """
    torch.manual_seed(0)
    latents = torch.randn(2, 1, 2, 2)  # [C=2, F=1, H=2, W=2]
    lat = latents
    f, h, w = lat.shape[-3:]
    holder = {"seq_len": f * h * w}
    deps = _make_deps(holder)
    model = _IdentityModel()
    schedule = FlowMatchingSchedule(uniform_prob=0.30)
    rng = np.random.default_rng(42)
    loss = training_step(
        model,
        _sample(latents),
        schedule,
        rng,
        device="cpu",
        dtype=torch.float32,
        deps=deps,
        first_frame_conditioning_p=0.0,
    )

    assert loss.dtype == torch.float32
    assert loss.dim() == 0

    noisy = model.seen.latent  # pred == noisy (identity model)
    sigma = model.seen.sigma.view(-1, 1, 1)
    clean = _identity_patchify(latents.unsqueeze(0))
    noise = (noisy - (1 - sigma) * clean) / sigma
    target = noise - clean
    expected = (noisy.float() - target.float()).pow(2).mean()

    assert torch.allclose(loss, expected, atol=1e-5)


def test_p1_loss_invariant_to_conditioning_region_predictions() -> None:
    """With ``first_frame_conditioning_p=1.0`` the first ``lat_h*lat_w`` tokens are conditioning
    tokens excluded from the loss. Perturbing the model output ONLY on that region must not change
    the loss (the masking is real, not cosmetic). RNG is re-seeded so the two runs differ ONLY in
    the model's conditioning-region prediction.
    """
    latents = torch.randn(2, 2, 2, 2)  # [C=2, F=2, H=2, W=2] -> seq_len=8, first-frame=4 tokens
    sample = _sample(latents)
    n_cond = 2 * 2  # lat_h * lat_w

    loss_clean = _run_with_model(sample, _IdentityModel(), p=1.0)
    loss_perturbed = _run_with_model(sample, _CondPerturbModel(n_cond), p=1.0)

    assert loss_clean.dim() == 0
    assert torch.allclose(loss_clean, loss_perturbed, atol=1e-5), (
        "loss must be invariant to predictions on the masked-out first-frame conditioning region"
    )


# --------------------------------------------------------------------------------------------------
# Seam D (06-05): conditioning_mode dispatch — multi_frame → MultiFrameStrategy; none stays byte-equiv
# --------------------------------------------------------------------------------------------------


def test_multi_frame_dispatch_returns_scalar() -> None:
    """``training_step(..., conditioning_mode="multi_frame")`` drives the Plan-06-02
    ``MultiFrameStrategy`` on the stubbed ltx_core seam and returns a 0-dim scalar — the wiring the
    gated multi-frame overfit depends on, proven CPU-only with zero ltx_core / zero CUDA.
    """
    latents = torch.randn(2, 2, 2, 2)  # [C=2, F=2, H=2, W=2] -> seq_len=8
    sample = _sample(latents)
    f, h, w = latents.shape[-3:]
    holder = {"seq_len": f * h * w}
    deps = _make_deps(holder)
    schedule = FlowMatchingSchedule(uniform_prob=0.30)
    rng = np.random.default_rng(42)
    loss = training_step(
        _IdentityModel(),
        sample,
        schedule,
        rng,
        device="cpu",
        dtype=torch.float32,
        deps=deps,
        conditioning_mode="multi_frame",
        max_conditioning_items=1,
        conditioning_strength_range=(0.3, 1.0),
    )
    assert loss.dtype == torch.float32
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_conditioning_mode_none_byte_equivalent_to_p0() -> None:
    """``conditioning_mode="none"`` (the default) routes ``SingleFrameStrategy(p=0)`` — the returned
    loss is byte-equivalent to the Phase-3 velocity objective (the unconditioned regression guard).
    """
    torch.manual_seed(0)
    latents = torch.randn(2, 1, 2, 2)  # [C=2, F=1, H=2, W=2]
    f, h, w = latents.shape[-3:]
    holder = {"seq_len": f * h * w}
    deps = _make_deps(holder)
    model = _IdentityModel()
    schedule = FlowMatchingSchedule(uniform_prob=0.30)
    rng = np.random.default_rng(42)
    loss = training_step(
        model,
        _sample(latents),
        schedule,
        rng,
        device="cpu",
        dtype=torch.float32,
        deps=deps,
        conditioning_mode="none",
    )

    assert loss.dim() == 0
    noisy = model.seen.latent  # pred == noisy (identity model)
    sigma = model.seen.sigma.view(-1, 1, 1)
    clean = _identity_patchify(latents.unsqueeze(0))
    noise = (noisy - (1 - sigma) * clean) / sigma
    target = noise - clean
    expected = (noisy.float() - target.float()).pow(2).mean()

    assert torch.allclose(loss, expected, atol=1e-5)


# --------------------------------------------------------------------------------------------------
# Seam D (06-05 Task 2): train_loop threads config.conditioning into the training_step call
# --------------------------------------------------------------------------------------------------


def test_train_loop_threads_multi_frame_conditioning_kwargs(monkeypatch) -> None:
    """``train_loop`` threads ``config.conditioning`` into the ``training_step`` call for a
    ``multi_frame`` config — a CPU/stub loop-level smoke that captures the kwargs reaching
    ``training_step`` (mode + max_conditioning_items + conditioning_strength_range). Everything heavy
    (the ltx_core seam + the real training_step) is monkeypatched, so this runs with zero CUDA.
    """
    import torch.nn as nn

    from signet_trainer.train import loop as loop_mod
    from signet_trainer.train import step as step_mod

    captured: dict = {}
    param = nn.Parameter(torch.zeros(1))

    def _fake_training_step(model, batch, schedule, rng, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return (param * 2).sum()  # a real graph so loop.py's .backward() works

    monkeypatch.setattr(step_mod, "training_step", _fake_training_step)
    monkeypatch.setattr(step_mod, "build_step_deps", lambda: None)

    class _Model:
        def parameters(self):
            return iter([param])

    class _CkptManager:
        def resume(self, model, optimizer, scheduler):  # noqa: ANN001
            return 0

        def save(self, *a, **k):  # noqa: ANN002, ANN003
            pass

    class _Optim:
        def step(self):
            pass

        def zero_grad(self):
            pass

    class _Sched:
        def step(self):
            pass

    config = SimpleNamespace(
        training=SimpleNamespace(
            mixed_precision="fp32", seed=42, max_steps=1,
            gradient_accumulation_steps=1, max_grad_norm=1.0, checkpoint_every=1000,
            checkpoint_expected_minutes=None, checkpoint_stall_multiplier=2.5,
        ),
        conditioning=SimpleNamespace(
            mode="multi_frame", max_conditioning_items=3,
            conditioning_strength_range=[0.3, 0.9], first_frame_conditioning_p=1.0,
        ),
    )
    dataset = [_sample(torch.randn(2, 1, 2, 2))]

    loop_mod.train_loop(
        _Model(), dataset, _Optim(), _Sched(),
        FlowMatchingSchedule(uniform_prob=0.30), _CkptManager(), config, checkpoints_vol=None,
    )

    assert captured["conditioning_mode"] == "multi_frame"
    assert captured["max_conditioning_items"] == 3
    assert captured["conditioning_strength_range"] == (0.3, 0.9)
