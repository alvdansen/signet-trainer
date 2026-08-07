"""SC#1 CPU proof for ``ICLoraStrategy`` — the IC-LoRA in-context reference-control core (Phase 7).

Drives the REAL ``ICLoraStrategy.prepare_training_inputs`` + ``compute_loss`` on a synthetic batch
with a stubbed ``StepDeps`` (mirroring ``test_single_frame_strategy.py`` /
``test_multi_frame_strategy.py``) — zero ltx_core, zero CUDA, Windows/CI-free. Proves the REF-03 /
SC#1 invariants of the concat + doubled-position + target-only-loss construction:

    - combined latent length == ref_seq_len + target_seq_len; ref_seq_len == reference patchified len.
    - ``video_targets`` length == target_seq_len (NOT the combined sequence).
    - positions tensor shape [B, 3, ref_seq_len + target_seq_len, 2] (built per-half, cat on dim=2).
    - ``video_loss_mask[:, :ref_seq_len]`` is all-False (the reference prefix is never in the loss).
    - the reference prefix of the combined latent is CLEAN (== the patchified ref, never noised) and
      its per-token timesteps are 0.
    - ``compute_loss`` slices ``model_output[:, ref_seq_len:, :]`` and returns a 0-dim scalar.
    - L-3 guard: ``video_targets`` is target-only, so an UNSLICED subtract against the combined-length
      model output fails loud on shape.
    - regression: with a degenerate ref of length 0 the target-half math matches single_frame's
      mask/loss shape.
    - D-7-REF11: ``reference_downscale_factor != 1`` scales the reference height/width position axes
      (inert at the default 1).
    - import-confinement: module top is torch + stdlib only (no modal/ltx_core/ltx_trainer).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from signet_trainer.conditioning.ic_lora import ICLoraStrategy
from signet_trainer.conditioning.single_frame import SingleFrameStrategy
from signet_trainer.conditioning.strategy import ModelInputs
from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import StepDeps


# --------------------------------------------------------------------------------------------------
# CPU stubs for the ltx_core seam — a TWO-CALL variant (ref dims + target dims) of the sibling stubs.
# --------------------------------------------------------------------------------------------------


def _identity_patchify(latent_5d: torch.Tensor) -> torch.Tensor:
    """A faithful ``patch_size=1`` patchify: [B, C, F, H, W] -> [B, F*H*W, C]."""
    b, c, f, h, w = latent_5d.shape
    return latent_5d.reshape(b, c, f * h * w).permute(0, 2, 1).contiguous()


class _RecordingShape:
    """Stub ``VideoLatentShape`` that records EVERY (latent) dim-set it is built with.

    IC-LoRA builds the shape TWICE (ref half, target half), so we append to a class-level list
    (unlike the siblings' single-shot ``last_kwargs``) and expose ``.seq`` so the paired
    ``get_pixel_coords`` stub can size each half's position tensor independently.
    """

    all_kwargs: list[dict] = []

    def __init__(self, **kwargs) -> None:
        _RecordingShape.all_kwargs.append(kwargs)
        self.kwargs = kwargs
        self.seq = kwargs["frames"] * kwargs["height"] * kwargs["width"]


class _StubPatchifier:
    def patchify(self, latent_5d: torch.Tensor) -> torch.Tensor:
        return _identity_patchify(latent_5d)

    def get_patch_grid_bounds(self, output_shape, device):  # noqa: ANN001
        # Pass the shape object through so the paired get_pixel_coords stub can read its .seq.
        return output_shape


def _make_deps(*, pos_fill: float = 0.0) -> StepDeps:
    """A stubbed ``StepDeps`` whose ``get_pixel_coords`` sizes [1, 3, seq, 2] from the shape's seq.

    ``pos_fill`` seeds every position coord to a constant so the downscale-factor test can observe
    the height/width-axis scaling (zeros would hide a multiply).
    """

    def _get_pixel_coords(latent_coords, scale_factors, causal_fix):  # noqa: ANN001
        seq = latent_coords.seq
        return torch.full((1, 3, seq, 2), float(pos_fill))

    return StepDeps(
        patchifier=_StubPatchifier(),
        get_pixel_coords=_get_pixel_coords,
        scale_factors=None,
        video_latent_shape_cls=_RecordingShape,
        modality_cls=lambda **kw: SimpleNamespace(**kw),
    )


class _FixedSchedule:
    """Schedule stub returning a constant sigma for every sample (ignores ``rng``)."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def sample_timesteps(self, batch_size, seq_len, rng):  # noqa: ANN001
        return np.full(batch_size, self.sigma, dtype=np.float64)


# --------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------


def _sample(
    target_latents: torch.Tensor,
    ref_latents: torch.Tensor,
    *,
    n_text: int = 4,
    ctx_dim: int = 8,
    mask=None,
) -> dict:
    if mask is None:
        mask = torch.ones(n_text, dtype=torch.long)
    return {
        "latent_conditions": {"latents": target_latents},
        "ref_latent_conditions": {"latents": ref_latents},
        "text_conditions": {
            "video_prompt_embeds": torch.randn(n_text, ctx_dim),
            "prompt_attention_mask": mask,
        },
        "idx": 0,
    }


def _make_ic(
    *,
    first_frame_conditioning_p: float = 0.0,
    reference_downscale_factor: int = 1,
    schedule=None,
    deps=None,
) -> ICLoraStrategy:
    if schedule is None:
        schedule = FlowMatchingSchedule(uniform_prob=0.30)
    if deps is None:
        deps = _make_deps()
    return ICLoraStrategy(
        deps=deps,
        schedule=schedule,
        first_frame_conditioning_p=first_frame_conditioning_p,
        reference_downscale_factor=reference_downscale_factor,
        device="cpu",
        dtype=torch.float32,
    )


def _make_single(latents: torch.Tensor, *, p: float, schedule=None) -> SingleFrameStrategy:
    if schedule is None:
        schedule = FlowMatchingSchedule(uniform_prob=0.30)
    return SingleFrameStrategy(
        _make_deps(),
        schedule,
        first_frame_conditioning_p=p,
        device="cpu",
        dtype=torch.float32,
    )


def setup_function(_func) -> None:
    _RecordingShape.all_kwargs = []


# --------------------------------------------------------------------------------------------------
# concat: combined length == ref + target ; ref_seq_len == reference patchified length
# --------------------------------------------------------------------------------------------------


def test_combined_length_is_ref_plus_target() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)  # [C, F=3, H=2, W=2] -> target_seq = 12
    ref = torch.randn(2, 2, 2, 2)  # [C, F=2, H=2, W=2] -> ref_seq = 8
    strat = _make_ic(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    assert inputs.ref_seq_len == 8  # reference patchified length
    combined = inputs.video.latent
    assert combined.shape[1] == 8 + 12  # ref_seq_len + target_seq_len


# --------------------------------------------------------------------------------------------------
# video_targets is TARGET-ONLY (not combined)
# --------------------------------------------------------------------------------------------------


def test_video_targets_is_target_only() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)  # target_seq = 12
    ref = torch.randn(2, 2, 2, 2)  # ref_seq = 8
    strat = _make_ic(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    assert inputs.video_targets.shape[1] == 12  # target_seq_len, NOT 20
    assert inputs.video.latent.shape[1] == 20  # combined


# --------------------------------------------------------------------------------------------------
# positions shape [B, 3, ref+target, 2] (built per-half, concatenated on dim=2)
# --------------------------------------------------------------------------------------------------


def test_positions_shape_is_combined() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)  # target_seq = 12
    ref = torch.randn(2, 2, 2, 2)  # ref_seq = 8
    strat = _make_ic(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    assert inputs.video.positions.shape == (1, 3, 8 + 12, 2)

    # the shape cls was built TWICE — once at ref dims, once at target dims.
    assert len(_RecordingShape.all_kwargs) == 2
    ref_kw, target_kw = _RecordingShape.all_kwargs
    assert (ref_kw["frames"], ref_kw["height"], ref_kw["width"]) == (2, 2, 2)
    assert (target_kw["frames"], target_kw["height"], target_kw["width"]) == (3, 2, 2)


# --------------------------------------------------------------------------------------------------
# loss mask: reference prefix all-False (L-3 — the ref half is never in the loss)
# --------------------------------------------------------------------------------------------------


def test_loss_mask_ref_prefix_all_false() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)
    ref = torch.randn(2, 2, 2, 2)
    strat = _make_ic(first_frame_conditioning_p=1.0, schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    ref_len = inputs.ref_seq_len
    assert inputs.video_loss_mask.dtype == torch.bool
    assert not bool(inputs.video_loss_mask[:, :ref_len].any())  # ref prefix never in loss


# --------------------------------------------------------------------------------------------------
# reference prefix is CLEAN (== patchified ref, never noised) + ref timesteps == 0
# --------------------------------------------------------------------------------------------------


def test_reference_prefix_is_clean_and_timestep_zero() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)
    ref = torch.randn(2, 2, 2, 2)
    strat = _make_ic(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    ref_len = inputs.ref_seq_len
    ref_clean = _identity_patchify(ref.unsqueeze(0))  # [1, ref_seq, C]
    assert torch.allclose(inputs.video.latent[:, :ref_len, :], ref_clean, atol=1e-6)
    # per-token timesteps on the reference prefix are all 0 (conditioning tokens).
    assert torch.allclose(
        inputs.video.timesteps[:, :ref_len], torch.zeros(1, ref_len), atol=1e-6
    )


# --------------------------------------------------------------------------------------------------
# compute_loss slices off the reference prefix and returns a 0-dim scalar
# --------------------------------------------------------------------------------------------------


def test_compute_loss_slices_ref_and_returns_scalar() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)
    ref = torch.randn(2, 2, 2, 2)
    strat = _make_ic(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    combined_len = inputs.video.latent.shape[1]
    model_output = torch.randn(1, combined_len, target.shape[0])  # COMBINED-length pred
    loss = strat.compute_loss(inputs, model_output)
    assert loss.dim() == 0  # 0-dim scalar

    # loss is invariant to the reference-prefix predictions (they are sliced off): blowing up the
    # ref half of the model output must not change the loss.
    perturbed = model_output.clone()
    perturbed[:, : inputs.ref_seq_len, :] += 1000.0
    loss_perturbed = strat.compute_loss(inputs, perturbed)
    assert torch.allclose(loss, loss_perturbed, atol=1e-5)


def test_target_only_targets_fail_loud_without_slice() -> None:
    """L-3 guard: ``video_targets`` is target-only, so subtracting it from the COMBINED-length model
    output (i.e. forgetting the ``[:, ref_seq_len:, :]`` slice) raises a shape error — the mistake
    fails loud rather than silently training a dead adapter on the reference region.
    """
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)  # target_seq = 12
    ref = torch.randn(2, 2, 2, 2)  # ref_seq = 8 -> combined 20
    strat = _make_ic(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    combined_len = inputs.video.latent.shape[1]
    model_output = torch.randn(1, combined_len, target.shape[0])
    assert inputs.video_targets.shape[1] != combined_len  # target-only vs combined
    with pytest.raises(RuntimeError):
        _ = (model_output - inputs.video_targets)  # broadcast fails on the seq axis


# --------------------------------------------------------------------------------------------------
# degenerate zero-length reference -> target-half math matches single_frame shape
# --------------------------------------------------------------------------------------------------


def test_degenerate_zero_ref_matches_single_frame_shape() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)  # target_seq = 12
    ref = torch.randn(2, 0, 2, 2)  # F=0 -> ref_seq = 0 (degenerate)
    strat = _make_ic(first_frame_conditioning_p=0.0, schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    assert inputs.ref_seq_len == 0
    assert inputs.video.latent.shape[1] == 12  # combined == target only
    assert inputs.video_targets.shape[1] == 12
    assert inputs.video_loss_mask.shape[1] == 12

    # p=0 -> the whole target is in the loss, matching single_frame(p=0)'s all-target mask shape.
    sf = _make_single(target, p=0.0, schedule=_FixedSchedule(0.5))
    sf_inputs = sf.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))
    assert inputs.video_loss_mask.shape == sf_inputs.video_loss_mask.shape
    assert bool(inputs.video_loss_mask.all())  # nothing conditioned at p=0, ref len 0


# --------------------------------------------------------------------------------------------------
# D-7-REF11: reference_downscale_factor scales the ref height/width position axes (inert at 1)
# --------------------------------------------------------------------------------------------------


def test_reference_downscale_factor_scales_ref_position_axes() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)
    ref = torch.randn(2, 2, 2, 2)
    # pos_fill=1.0 so a multiply is observable; factor 2 doubles ref axes 1 (H) and 2 (W).
    deps = _make_deps(pos_fill=1.0)
    strat = _make_ic(reference_downscale_factor=2, schedule=_FixedSchedule(0.5), deps=deps)
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    ref_len = inputs.ref_seq_len
    pos = inputs.video.positions
    # ref half: height (axis 1) and width (axis 2) coords doubled; time (axis 0) untouched by scaling.
    assert torch.allclose(pos[:, 1, :ref_len, :], torch.full((1, ref_len, 2), 2.0), atol=1e-6)
    assert torch.allclose(pos[:, 2, :ref_len, :], torch.full((1, ref_len, 2), 2.0), atol=1e-6)
    # target half: unscaled (still 1.0 on axes 1/2).
    assert torch.allclose(pos[:, 1, ref_len:, :], torch.full((1, 12, 2), 1.0), atol=1e-6)


def test_reference_downscale_factor_1_is_inert() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)
    ref = torch.randn(2, 2, 2, 2)
    deps = _make_deps(pos_fill=1.0)
    strat = _make_ic(reference_downscale_factor=1, schedule=_FixedSchedule(0.5), deps=deps)
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    ref_len = inputs.ref_seq_len
    pos = inputs.video.positions
    # factor 1 -> no scaling: ref axes 1/2 stay at the un-scaled fill value.
    assert torch.allclose(pos[:, 1, :ref_len, :], torch.full((1, ref_len, 2), 1.0), atol=1e-6)
    assert torch.allclose(pos[:, 2, :ref_len, :], torch.full((1, ref_len, 2), 1.0), atol=1e-6)


# --------------------------------------------------------------------------------------------------
# ModelInputs shape (video-only) + data sources declare the third reference source
# --------------------------------------------------------------------------------------------------


def test_prepare_returns_video_only_model_inputs() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 2)
    ref = torch.randn(2, 2, 2, 2)
    strat = _make_ic(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(_sample(target, ref), rng=np.random.default_rng(0))

    assert isinstance(inputs, ModelInputs)
    assert inputs.audio is None
    assert inputs.audio_targets is None
    assert inputs.audio_loss_mask is None


def test_get_data_sources_includes_reference() -> None:
    strat = _make_ic()
    assert strat.get_data_sources() == ["latents", "conditions", "reference_latents"]


# --------------------------------------------------------------------------------------------------
# import-confinement (Anti-Pattern 6 / Pitfall 6): torch + stdlib only at module top
# --------------------------------------------------------------------------------------------------


def test_module_is_import_confined() -> None:
    import signet_trainer.conditioning.ic_lora as ic

    with open(ic.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
            assert "modal" not in stripped
            assert "ltx_core" not in stripped
            assert "ltx_trainer" not in stripped
