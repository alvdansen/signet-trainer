"""CPU proof for ``InpaintStrategy`` — prior-project flexible/mask semantics (Phase 9 inpaint gate).

Drives the REAL ``InpaintStrategy.prepare_training_inputs`` + ``compute_loss`` on a synthetic
batch with a stubbed ``StepDeps`` (mirroring ``test_ic_lora_strategy.py`` /
``test_single_frame_strategy.py``) — zero ltx_core, zero CUDA, Windows/CI-free. Proves the
GATE-SPEC invariants of the per-token spatial denoise-mask construction:

    - FLATTEN ORDER: the ``[F_lat, H_lat, W_lat]`` mask flattens FRAME-MAJOR
      (``token = f*H*W + h*W + w``) — verified against a HAND-COMPUTED 2x2x3 example AND against
      ``mask_ops.latent_frame_token_span``.
    - POLARITY: mask>0.5 (KEEP/context) tokens carry the CLEAN latent, per-token timestep 0, and
      are EXCLUDED from the loss; mask<=0.5 (GENERATE) tokens are noised, carry the sampled sigma,
      and are IN the loss (``video_loss_mask = mask <= 0.5``).
    - ``inpaint_mask_probability``: p=1.0 always applies the mask; p=0.0 never does — the sample
      falls back to PLAIN t2v denoise on the full frame (all tokens generated, all in the loss).
    - DIM MISMATCH fails loud: a mask whose dims don't match the sample's latent dims raises.
    - ``compute_loss`` is a 0-dim scalar, invariant to predictions on KEEP tokens.
    - dry-run gate: ``dryrun/shapes.py`` covers mode="inpaint" — flatten-order token indices,
      polarity, and the %64 / 8n+1 dims verification (rule OWNED by config/validators.py).
    - import-confinement: module top is torch + stdlib only (no modal/ltx_core/ltx_trainer).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from signet_trainer.conditioning.inpaint import InpaintStrategy
from signet_trainer.conditioning.mask_ops import latent_frame_token_span
from signet_trainer.conditioning.strategy import ModelInputs
from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import StepDeps

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_YAML = str(REPO_ROOT / "configs" / "ltx23_lora.example.yaml")


# --------------------------------------------------------------------------------------------------
# CPU stubs for the ltx_core seam (mirrors the sibling strategy tests)
# --------------------------------------------------------------------------------------------------


def _identity_patchify(latent_5d: torch.Tensor) -> torch.Tensor:
    """A faithful ``patch_size=1`` patchify: [B, C, F, H, W] -> [B, F*H*W, C] (frame-major)."""
    b, c, f, h, w = latent_5d.shape
    return latent_5d.reshape(b, c, f * h * w).permute(0, 2, 1).contiguous()


class _RecordingShape:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.seq = kwargs["frames"] * kwargs["height"] * kwargs["width"]


class _StubPatchifier:
    def patchify(self, latent_5d: torch.Tensor) -> torch.Tensor:
        return _identity_patchify(latent_5d)

    def get_patch_grid_bounds(self, output_shape, device):  # noqa: ANN001
        return output_shape


def _make_deps() -> StepDeps:
    def _get_pixel_coords(latent_coords, scale_factors, causal_fix):  # noqa: ANN001
        return torch.zeros(1, 3, latent_coords.seq, 2)

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
    latents: torch.Tensor,
    mask,
    *,
    mask_key: str = "video_mask_conditions",
    n_text: int = 4,
    ctx_dim: int = 8,
) -> dict:
    batch = {
        "latent_conditions": {"latents": latents},
        "text_conditions": {
            "video_prompt_embeds": torch.randn(n_text, ctx_dim),
            "prompt_attention_mask": torch.ones(n_text, dtype=torch.long),
        },
        "idx": 0,
    }
    if mask is not None:
        batch[mask_key] = mask
    return batch


def _make_inpaint(
    *,
    inpaint_mask_probability: float = 1.0,
    schedule=None,
    deps=None,
) -> InpaintStrategy:
    if schedule is None:
        schedule = FlowMatchingSchedule(uniform_prob=0.30)
    if deps is None:
        deps = _make_deps()
    return InpaintStrategy(
        deps=deps,
        schedule=schedule,
        inpaint_mask_probability=inpaint_mask_probability,
        device="cpu",
        dtype=torch.float32,
    )


#: The hand-computed 2x2x3 example (F_lat=2, H_lat=2, W_lat=3 -> seq_len=12).
#: KEEP cells (f, h, w) -> expected frame-major flat index f*(2*3) + h*3 + w:
#:   (0, 0, 1) -> 0*6 + 0*3 + 1 = 1
#:   (0, 1, 2) -> 0*6 + 1*3 + 2 = 5
#:   (1, 1, 0) -> 1*6 + 1*3 + 0 = 9
_KEEP_CELLS = [(0, 0, 1), (0, 1, 2), (1, 1, 0)]
_EXPECTED_FLAT = [1, 5, 9]


def _mask_2x2x3() -> torch.Tensor:
    mask = torch.zeros(2, 2, 3, dtype=torch.float32)
    for f, h, w in _KEEP_CELLS:
        mask[f, h, w] = 1.0
    return mask


def _latents_2x2x3(channels: int = 2) -> torch.Tensor:
    return torch.randn(channels, 2, 2, 3)  # [C, F=2, H=2, W=3] -> seq_len = 12


# --------------------------------------------------------------------------------------------------
# FLATTEN ORDER: hand-computed 2x2x3 example + latent_frame_token_span cross-check
# --------------------------------------------------------------------------------------------------


def test_flatten_order_matches_hand_computed_2x2x3() -> None:
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_latents_2x2x3(), _mask_2x2x3()), rng=np.random.default_rng(0)
    )

    # KEEP tokens are exactly the hand-computed frame-major indices {1, 5, 9} — nothing else.
    keep = inputs.video.timesteps == 0
    expected = torch.zeros(1, 12, dtype=torch.bool)
    expected[:, _EXPECTED_FLAT] = True
    assert torch.equal(keep, expected), (
        f"KEEP token indices {keep.nonzero()[:, 1].tolist()} != hand-computed {_EXPECTED_FLAT}"
    )


def test_flatten_order_matches_latent_frame_token_span() -> None:
    # The same indices recomputed via mask_ops.latent_frame_token_span (the ORDER oracle): each
    # KEEP cell (f, h, w) must land at span_start(f) + h*W + w.
    lat_h, lat_w = 2, 3
    recomputed = []
    for f, h, w in _KEEP_CELLS:
        start, stop = latent_frame_token_span(f, lat_h, lat_w)
        idx = start + h * lat_w + w
        assert start <= idx < stop
        recomputed.append(idx)
    assert recomputed == _EXPECTED_FLAT

    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_latents_2x2x3(), _mask_2x2x3()), rng=np.random.default_rng(0)
    )
    keep_indices = (inputs.video.timesteps == 0).nonzero()[:, 1].tolist()
    assert keep_indices == recomputed


# --------------------------------------------------------------------------------------------------
# POLARITY: keep = clean latent + timestep 0 + OUT of loss; generate = sigma + IN loss
# --------------------------------------------------------------------------------------------------


def test_polarity_keep_tokens_clean_timestep0_out_of_loss() -> None:
    torch.manual_seed(0)
    latents = _latents_2x2x3()
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(latents, _mask_2x2x3()), rng=np.random.default_rng(0)
    )

    clean_patched = _identity_patchify(latents.unsqueeze(0))  # [1, 12, C]
    keep = torch.zeros(1, 12, dtype=torch.bool)
    keep[:, _EXPECTED_FLAT] = True

    # KEEP: clean latent (the where(mask>0.5, clean, noisy) substitution), timestep 0, NOT in loss.
    assert torch.allclose(inputs.video.latent[keep], clean_patched[keep], atol=1e-6)
    assert torch.all(inputs.video.timesteps[keep] == 0)
    assert not bool(inputs.video_loss_mask[keep].any())

    # GENERATE: sampled sigma, IN loss, latent is the noised mix (differs from clean w.p. 1).
    assert torch.all(inputs.video.timesteps[~keep] == 0.5)
    assert bool(inputs.video_loss_mask[~keep].all())
    assert not torch.allclose(inputs.video.latent[~keep], clean_patched[~keep], atol=1e-6)

    # The explicit contract form: video_loss_mask == (mask <= 0.5) == ~keep.
    assert torch.equal(inputs.video_loss_mask, ~keep)


def test_targets_are_full_length_velocity_no_ref_prefix() -> None:
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_latents_2x2x3(), _mask_2x2x3()), rng=np.random.default_rng(0)
    )
    assert isinstance(inputs, ModelInputs)
    assert inputs.video_targets.shape[1] == 12  # full sequence — no concat, no prefix
    assert inputs.video.latent.shape[1] == 12
    assert inputs.ref_seq_len is None
    assert inputs.audio is None and inputs.audio_targets is None and inputs.audio_loss_mask is None


# --------------------------------------------------------------------------------------------------
# inpaint_mask_probability: 1.0 always applies; 0.0 falls back to plain full-frame t2v
# --------------------------------------------------------------------------------------------------


def test_probability_one_always_applies_mask() -> None:
    for seed in range(5):  # any torch rng state: rand() in [0, 1) is always < 1.0
        torch.manual_seed(seed)
        strat = _make_inpaint(inpaint_mask_probability=1.0, schedule=_FixedSchedule(0.5))
        inputs = strat.prepare_training_inputs(
            _sample(_latents_2x2x3(), _mask_2x2x3()), rng=np.random.default_rng(seed)
        )
        assert int((inputs.video.timesteps == 0).sum()) == len(_EXPECTED_FLAT)
        assert int((~inputs.video_loss_mask).sum()) == len(_EXPECTED_FLAT)


def test_probability_zero_falls_back_to_plain_t2v() -> None:
    # A no-mask draw zeroes the effective mask: ALL tokens generated (mask of all zeros = all
    # generated), all in the loss, no timestep-0 tokens — plain t2v denoise on the full frame.
    for seed in range(5):
        torch.manual_seed(seed)
        strat = _make_inpaint(inpaint_mask_probability=0.0, schedule=_FixedSchedule(0.5))
        inputs = strat.prepare_training_inputs(
            _sample(_latents_2x2x3(), _mask_2x2x3()), rng=np.random.default_rng(seed)
        )
        assert bool(inputs.video_loss_mask.all())  # every token carries the loss
        assert torch.all(inputs.video.timesteps == 0.5)  # every token at the sampled sigma
        assert not bool((inputs.video.timesteps == 0).any())


# --------------------------------------------------------------------------------------------------
# DIM MISMATCH + missing/malformed mask: fail loud
# --------------------------------------------------------------------------------------------------


def test_dim_mismatch_fails_loud() -> None:
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    bad_mask = torch.zeros(3, 2, 3)  # F=3 != latent F=2
    with pytest.raises(ValueError, match="do not match the sample's latent dims"):
        strat.prepare_training_inputs(
            _sample(_latents_2x2x3(), bad_mask), rng=np.random.default_rng(0)
        )


def test_spatial_dim_mismatch_fails_loud() -> None:
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    bad_mask = torch.zeros(2, 3, 2)  # H/W transposed vs latent [F=2, H=2, W=3]
    with pytest.raises(ValueError, match="do not match the sample's latent dims"):
        strat.prepare_training_inputs(
            _sample(_latents_2x2x3(), bad_mask), rng=np.random.default_rng(0)
        )


def test_missing_mask_source_fails_loud() -> None:
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    with pytest.raises(KeyError, match="video_masks"):
        strat.prepare_training_inputs(
            _sample(_latents_2x2x3(), None), rng=np.random.default_rng(0)
        )


def test_wrong_rank_mask_fails_loud() -> None:
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    with pytest.raises(ValueError, match=r"\[F_lat, H_lat, W_lat\]"):
        strat.prepare_training_inputs(
            _sample(_latents_2x2x3(), torch.zeros(12)), rng=np.random.default_rng(0)
        )


# --------------------------------------------------------------------------------------------------
# mask payload forms: bare tensor (signet contract), {"mask": tensor} dict (upstream oracle), and
# the dir-name fallback batch key
# --------------------------------------------------------------------------------------------------


def test_dict_payload_with_mask_key_accepted() -> None:
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_latents_2x2x3(), {"mask": _mask_2x2x3()}), rng=np.random.default_rng(0)
    )
    keep_indices = (inputs.video.timesteps == 0).nonzero()[:, 1].tolist()
    assert keep_indices == _EXPECTED_FLAT


def test_video_masks_fallback_batch_key_accepted() -> None:
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_latents_2x2x3(), _mask_2x2x3(), mask_key="video_masks"),
        rng=np.random.default_rng(0),
    )
    keep_indices = (inputs.video.timesteps == 0).nonzero()[:, 1].tolist()
    assert keep_indices == _EXPECTED_FLAT


def test_batched_mask_accepted() -> None:
    # DataLoader default-collate form: [B=1, F, H, W].
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_latents_2x2x3(), _mask_2x2x3().unsqueeze(0)), rng=np.random.default_rng(0)
    )
    keep_indices = (inputs.video.timesteps == 0).nonzero()[:, 1].tolist()
    assert keep_indices == _EXPECTED_FLAT


# --------------------------------------------------------------------------------------------------
# data sources + compute_loss
# --------------------------------------------------------------------------------------------------


def test_get_data_sources_declares_video_masks() -> None:
    strat = _make_inpaint()
    assert strat.get_data_sources() == ["latents", "conditions", "video_masks"]


def test_get_data_sources_honors_configured_mask_dir() -> None:
    """#21 finding 3: a non-default inpaint_mask_dir must be READ here, not just written on the
    Modal preprocess side (modal/entrypoint.py::_mask_encode_params). Before this fix,
    ``get_data_sources()`` returned the bare literal ``"video_masks"`` unconditionally, so a
    config-driven ``inpaint_mask_dir: "custom_masks"`` validated clean at load and was then
    silently ignored on the read side.
    """
    strat = InpaintStrategy(deps=None, schedule=None, mask_dir="custom_masks")
    assert strat.get_data_sources() == ["latents", "conditions", "custom_masks"]


def test_get_data_sources_mask_dir_defaults_to_video_masks() -> None:
    """The default matches the schema default (``ConditioningConfig.inpaint_mask_dir``) exactly —
    every caller that never sets ``inpaint_mask_dir`` (or constructs the strategy directly, as
    ``train/step.py`` does) is unaffected by #21 finding 3's fix.
    """
    strat = InpaintStrategy(deps=None, schedule=None)
    assert strat.get_data_sources() == ["latents", "conditions", "video_masks"]


def test_compute_loss_scalar_and_invariant_to_keep_predictions() -> None:
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_latents_2x2x3(), _mask_2x2x3()), rng=np.random.default_rng(0)
    )

    model_output = torch.randn(1, 12, 2)
    loss = strat.compute_loss(inputs, model_output)
    assert loss.dim() == 0  # 0-dim scalar (Pitfall 1)

    # KEEP tokens are masked out of the loss: blowing up predictions there changes nothing.
    perturbed = model_output.clone()
    perturbed[:, _EXPECTED_FLAT, :] += 1000.0
    assert torch.allclose(loss, strat.compute_loss(inputs, perturbed), atol=1e-5)

    # GENERATE tokens DO carry the loss: perturbing one of them changes the loss.
    perturbed2 = model_output.clone()
    perturbed2[:, 0, :] += 1000.0  # token 0 is a GENERATE token (not in _EXPECTED_FLAT)
    assert not torch.allclose(loss, strat.compute_loss(inputs, perturbed2), atol=1e-5)


def test_all_keep_mask_yields_zero_loss() -> None:
    # Degenerate all-KEEP mask: nothing to generate -> empty loss mask -> loss 0 (clamp guards 0/0).
    torch.manual_seed(0)
    strat = _make_inpaint(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_latents_2x2x3(), torch.ones(2, 2, 3)), rng=np.random.default_rng(0)
    )
    assert not bool(inputs.video_loss_mask.any())
    loss = strat.compute_loss(inputs, torch.randn(1, 12, 2))
    assert loss.dim() == 0
    assert float(loss) == 0.0


# --------------------------------------------------------------------------------------------------
# CPU DRY-RUN GATE (dryrun/shapes.py inpaint coverage)
# --------------------------------------------------------------------------------------------------


def _load_inpaint_cfg(training_dims=None):
    """A validated example config flipped to mode='inpaint' via model_copy (no re-validation) —
    lets the dry-run gate run BEFORE the schema's 'inpaint' allowlist widening lands (that change
    is config-territory; see integration notes)."""
    from signet_trainer.config.load import load_config

    cfg = load_config(VALID_YAML)
    update = {"conditioning": cfg.conditioning.model_copy(update={"mode": "inpaint"})}
    # model_copy skips the schema's inpaint cross-field checks, so mirror what a REAL loaded
    # inpaint config carries: %64 buckets (validate_inpaint_resolution_buckets fires at load).
    # The dry-run gate now builds + asserts at EVERY bucket (gap-dryrun-ltx-1), so the example's
    # default %32-only buckets would be a fixture artifact, not a gate defect.
    update["data"] = cfg.data.model_copy(update={"resolution_buckets": ["768x512x49"]})
    if training_dims is not None:
        update["training_dims"] = training_dims
    return cfg.model_copy(update=update)


def test_dryrun_inpaint_mode_passes() -> None:
    from signet_trainer.dryrun.shapes import run_dryrun

    # 768x512x49 is ÷64-clean + 8n+1 — the inpaint gate must pass end-to-end.
    assert run_dryrun(_load_inpaint_cfg()) == 0


def test_dryrun_inpaint_flatten_order_and_polarity() -> None:
    from signet_trainer.dryrun.shapes import (
        _INPAINT_CLEAN,
        _inpaint_expected_keep_indices,
        build_dryrun_inputs,
    )

    cfg = _load_inpaint_cfg()
    mi = build_dryrun_inputs(cfg)

    width, height, frames = cfg.training_dims
    lat_f, lat_h, lat_w = (frames - 1) // 8 + 1, height // 32, width // 32
    expected = _inpaint_expected_keep_indices(lat_f, lat_h, lat_w)

    # Flatten order: the KEEP (timestep-0) token indices are EXACTLY the frame-major
    # latent_frame_token_span indices of the asymmetric pattern.
    keep_indices = (mi.video.timesteps == 0).nonzero()[:, 1].tolist()
    assert keep_indices == expected

    # Polarity: KEEP -> clean sentinel + out of loss; GENERATE -> in loss.
    keep = torch.zeros(1, mi.video.timesteps.shape[1], dtype=torch.bool)
    keep[:, expected] = True
    assert torch.equal(mi.video_loss_mask, ~keep)
    assert torch.all(mi.video.latent[keep] == _INPAINT_CLEAN)


def test_dryrun_inpaint_rejects_non_div64_width() -> None:
    from signet_trainer.dryrun.shapes import run_dryrun

    # 736 is %32-clean (23*32) but NOT %64 — legal for plain video, ILLEGAL for inpaint.
    # The %64 rule is OWNED by config/validators.py at load; the gate VERIFIES it fired.
    assert run_dryrun(_load_inpaint_cfg(training_dims=(736, 512, 49))) != 0


def test_dryrun_inpaint_rejects_non_8n1_frames() -> None:
    from signet_trainer.dryrun.shapes import run_dryrun

    assert run_dryrun(_load_inpaint_cfg(training_dims=(768, 512, 48))) != 0


def test_dryrun_inpaint_does_not_regress_other_modes() -> None:
    from signet_trainer.config.load import load_config
    from signet_trainer.dryrun.shapes import run_dryrun

    assert run_dryrun(load_config(VALID_YAML)) == 0  # mode as shipped in the example config


# --------------------------------------------------------------------------------------------------
# strategy selection (train/step.py mode-string dispatch) + loop kwargs threading
# --------------------------------------------------------------------------------------------------


def test_training_step_dispatches_inpaint_strategy() -> None:
    # Drive the REAL training_step with a stub model: mode="inpaint" must route through
    # InpaintStrategy (masked posture observable via the loss's keep-token invariance).
    from signet_trainer.train.step import training_step

    torch.manual_seed(0)
    batch = _sample(_latents_2x2x3(), _mask_2x2x3())

    captured: dict = {}

    def _model(video=None, audio=None, perturbations=None):  # noqa: ANN001
        captured["timesteps"] = video.timesteps
        return torch.zeros(1, video.latent.shape[1], 2), None

    loss = training_step(
        _model,
        batch,
        _FixedSchedule(0.5),
        np.random.default_rng(0),
        device="cpu",
        dtype=torch.float32,
        deps=_make_deps(),
        conditioning_mode="inpaint",
        inpaint_mask_probability=1.0,
    )
    assert loss.dim() == 0
    keep_indices = (captured["timesteps"] == 0).nonzero()[:, 1].tolist()
    assert keep_indices == _EXPECTED_FLAT  # the mask reached the model via the strategy


def test_build_conditioning_kwargs_threads_inpaint_probability() -> None:
    from signet_trainer.train.loop import build_conditioning_kwargs

    cond = SimpleNamespace(mode="inpaint", inpaint_mask_probability=0.9)
    kwargs = build_conditioning_kwargs(cond)
    assert kwargs == {"conditioning_mode": "inpaint", "inpaint_mask_probability": 0.9}

    # Other modes do NOT thread the inpaint knob (no silently-ignored kwargs).
    cond_none = SimpleNamespace(mode="none")
    assert "inpaint_mask_probability" not in build_conditioning_kwargs(cond_none)


# --------------------------------------------------------------------------------------------------
# import-confinement (Anti-Pattern 6 / Pitfall 6): torch + stdlib only at module top
# --------------------------------------------------------------------------------------------------


def test_module_is_import_confined() -> None:
    import signet_trainer.conditioning.inpaint as inp

    with open(inp.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
            assert "modal" not in stripped
            assert "ltx_core" not in stripped
            assert "ltx_trainer" not in stripped
