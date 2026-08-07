"""Contract tests for the conditioning/ Strategy seam (Plan 01-01, D-03).

Locks the REAL LTX-2.3 training-batch contract — ``ModelInputs`` wrapping ``Modality`` objects
(RESEARCH.md Q4) — NOT the older flat ``TrainingBatch`` named in CONTEXT.md/ARCHITECTURE.md.
If these shapes drift, Phases 2-4 hit contract mismatch when the real strategies drop in.
"""

from __future__ import annotations

import abc

import pytest
import torch

from signet_trainer.conditioning.strategy import (
    Modality,
    ModelInputs,
    TrainingStrategy,
    compute_seq_len,
)

from .conftest import (
    SAMPLE_SEQ_LEN,
    SAMPLE_WHF,
    VIDEO_CHANNELS,
    build_synthetic_model_inputs,
    build_synthetic_modality,
)


def test_seq_len_formula() -> None:
    """seq_len = (H//32)*(W//32)*((F-1)//8+1); verified [768,512,49] -> 2688."""
    w, h, f = SAMPLE_WHF
    assert compute_seq_len(w, h, f) == SAMPLE_SEQ_LEN == 2688


def test_video_latent_shape() -> None:
    """A synthetic Modality for [768,512,49] has latent.shape == (1, 2688, 128)."""
    modality = build_synthetic_modality(*SAMPLE_WHF)
    assert modality.latent.shape == (1, SAMPLE_SEQ_LEN, VIDEO_CHANNELS)
    assert modality.latent.shape == (1, 2688, 128)


def test_modality_positions_and_timesteps_shapes() -> None:
    """positions == (1, 3, seq_len, 2) (n_pos_dims=3 video); timesteps == (1, seq_len)."""
    modality = build_synthetic_modality(*SAMPLE_WHF)
    assert modality.positions.shape == (1, 3, SAMPLE_SEQ_LEN, 2)
    assert modality.timesteps.shape == (1, SAMPLE_SEQ_LEN)
    assert modality.sigma.shape == (1,)


def test_loss_mask_is_inverse_of_conditioning_mask() -> None:
    """ModelInputs.video_loss_mask == ~conditioning_mask (bool, shape (1, seq_len))."""
    inputs = build_synthetic_model_inputs(*SAMPLE_WHF, cond_first_frame=True)
    assert inputs.video_loss_mask.dtype == torch.bool
    assert inputs.video_loss_mask.shape == (1, SAMPLE_SEQ_LEN)

    # Reconstruct the conditioning mask from timestep-0 tokens and assert the inverse relation.
    conditioning_mask = inputs.video.timesteps == 0
    assert torch.equal(inputs.video_loss_mask, ~conditioning_mask)
    # And conditioning tokens must carry timestep 0.
    assert torch.all(inputs.video.timesteps[conditioning_mask] == 0)


def test_first_frame_conditioning_region_size() -> None:
    """First-frame conditioning marks exactly (H//32)*(W//32) tokens."""
    w, h, f = SAMPLE_WHF
    inputs = build_synthetic_model_inputs(w, h, f, cond_first_frame=True)
    conditioning_mask = inputs.video.timesteps == 0
    expected = (h // 32) * (w // 32)  # 16 * 24 = 384
    assert int(conditioning_mask.sum()) == expected == 384


def test_model_inputs_ref_seq_len_defaults_none() -> None:
    """ModelInputs exposes an optional ref_seq_len defaulting to None (IC-LoRA room, Phase 7)."""
    inputs = build_synthetic_model_inputs(*SAMPLE_WHF)
    assert inputs.ref_seq_len is None

    # And it is settable for the Phase-7 IC-LoRA path.
    inputs.ref_seq_len = 128
    assert inputs.ref_seq_len == 128


def test_modality_optional_fields_default() -> None:
    """Modality.enabled defaults True; context_mask/attention_mask default None."""
    bare = Modality(
        latent=torch.zeros(1, 4, VIDEO_CHANNELS),
        sigma=torch.tensor([0.5]),
        timesteps=torch.zeros(1, 4),
        positions=torch.zeros(1, 3, 4, 2),
        context=torch.zeros(1, 2, 16),
    )
    assert bare.enabled is True
    assert bare.context_mask is None
    assert bare.attention_mask is None


def test_model_inputs_audio_is_optional() -> None:
    """The audio modality/targets/mask are optional (signet-trainer is video-only)."""
    inputs = build_synthetic_model_inputs(*SAMPLE_WHF)
    assert inputs.audio is None
    assert inputs.audio_targets is None
    assert inputs.audio_loss_mask is None


def test_training_strategy_is_abstract() -> None:
    """TrainingStrategy is an ABC — instantiating it directly raises TypeError."""
    assert issubclass(TrainingStrategy, abc.ABC)
    with pytest.raises(TypeError):
        TrainingStrategy()  # type: ignore[abstract]


def test_training_strategy_abstract_methods() -> None:
    """The seam exposes the abstract methods Phases 5-7 subclass."""
    expected = {"prepare_training_inputs", "compute_loss", "get_data_sources"}
    assert expected.issubset(TrainingStrategy.__abstractmethods__)


def test_training_strategy_concrete_subclass_instantiable() -> None:
    """A subclass implementing all abstract methods CAN be instantiated and returns ModelInputs."""

    class _Concrete(TrainingStrategy):
        def get_data_sources(self):  # type: ignore[override]
            return []

        def prepare_training_inputs(self, batch):  # type: ignore[override]
            return build_synthetic_model_inputs(*SAMPLE_WHF)

        def compute_loss(self, model_inputs, model_output):  # type: ignore[override]
            return torch.tensor(0.0)

    strategy = _Concrete()
    out = strategy.prepare_training_inputs(None)
    assert isinstance(out, ModelInputs)
    assert out.video.latent.shape == (1, SAMPLE_SEQ_LEN, VIDEO_CHANNELS)
