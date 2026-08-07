"""CPU proof for ``A2VStrategy`` — frozen-audio-driven video generation (Phase 9 a2v gate).

Drives the REAL ``A2VStrategy.prepare_training_inputs`` + ``compute_loss`` on a synthetic batch with
a stubbed video ``StepDeps`` AND stubbed audio patchifier/shape (mirroring
``test_inpaint_strategy.py``) — zero ltx_core, zero CUDA, Windows/CI-free. Proves the a2v contract:

    - VIDEO is GENERATED: every video token noised, at the sampled sigma, ALL in the loss
      (``video_loss_mask`` all True) — plain t2v (no first-frame / mask conditioning).
    - AUDIO is FROZEN conditioning: clean latents, per-token timestep 0, sigma 0, and EXCLUDED from
      the loss (``audio_targets`` / ``audio_loss_mask`` stay None). Positions are ``[B, 1, T, 2]``.
    - AUDIO latent patchifies to ``[B, T, C*mel]`` (the LTX audio token dim); dim mismatch fails loud.
    - MISSING audio source fails loud.
    - ``compute_loss`` is a 0-dim scalar, video-only.
    - train/step.py dispatches mode="audio_to_video" -> A2VStrategy, and threads ``inputs.audio`` into
      the model call.
    - dryrun/shapes.py covers mode="audio_to_video".
    - import-confinement: module top is torch + stdlib only (no modal/ltx_core/ltx_trainer).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from signet_trainer.conditioning.a2v import A2VStrategy
from signet_trainer.conditioning.strategy import ModelInputs
from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import StepDeps

_AUDIO_C = 8
_AUDIO_MEL = 16
_AUDIO_T = 5  # a small synthetic audio time-step count


# --------------------------------------------------------------------------------------------------
# CPU stubs for the ltx_core seams (video StepDeps + the audio patchifier/shape)
# --------------------------------------------------------------------------------------------------


def _identity_patchify(latent_5d: torch.Tensor) -> torch.Tensor:
    """[B, C, F, H, W] -> [B, F*H*W, C] (frame-major, patch_size=1)."""
    b, c, f, h, w = latent_5d.shape
    return latent_5d.reshape(b, c, f * h * w).permute(0, 2, 1).contiguous()


class _RecordingShape:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.seq = kwargs["frames"] * kwargs["height"] * kwargs["width"]


class _StubVideoPatchifier:
    def patchify(self, latent_5d: torch.Tensor) -> torch.Tensor:
        return _identity_patchify(latent_5d)

    def get_patch_grid_bounds(self, output_shape, device):  # noqa: ANN001
        return output_shape


class _StubAudioPatchifier:
    """Audio patchifier stub: [B, C, T, mel] -> [B, T, C*mel]; positions -> [B, 1, T, 2]."""

    def patchify(self, audio_4d: torch.Tensor) -> torch.Tensor:
        b, c, t, mel = audio_4d.shape
        # move to [B, T, C, mel] then flatten channel*mel -> [B, T, C*mel]
        return audio_4d.permute(0, 2, 1, 3).reshape(b, t, c * mel).contiguous()

    def get_patch_grid_bounds(self, output_shape, device):  # noqa: ANN001
        b = output_shape.batch
        t = output_shape.frames
        return torch.zeros(b, 1, t, 2)


class _StubAudioShape:
    def __init__(self, *, frames, mel_bins, batch, channels) -> None:  # noqa: ANN001
        self.frames = frames
        self.mel_bins = mel_bins
        self.batch = batch
        self.channels = channels


def _make_deps() -> StepDeps:
    def _get_pixel_coords(latent_coords, scale_factors, causal_fix):  # noqa: ANN001
        return torch.zeros(1, 3, latent_coords.seq, 2)

    return StepDeps(
        patchifier=_StubVideoPatchifier(),
        get_pixel_coords=_get_pixel_coords,
        scale_factors=None,
        video_latent_shape_cls=_RecordingShape,
        modality_cls=lambda **kw: SimpleNamespace(**kw),
    )


class _FixedSchedule:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def sample_timesteps(self, batch_size, seq_len, rng):  # noqa: ANN001
        return np.full(batch_size, self.sigma, dtype=np.float64)


def _make_a2v(*, schedule=None, deps=None) -> A2VStrategy:
    if schedule is None:
        schedule = FlowMatchingSchedule(uniform_prob=0.30)
    if deps is None:
        deps = _make_deps()
    return A2VStrategy(
        deps=deps,
        schedule=schedule,
        device="cpu",
        dtype=torch.float32,
        audio_patchifier=_StubAudioPatchifier(),
        audio_latent_shape_cls=_StubAudioShape,
    )


def _video_latents(channels: int = 2) -> torch.Tensor:
    return torch.randn(channels, 2, 2, 3)  # [C, F=2, H=2, W=3] -> seq_len = 12


def _audio_latents() -> torch.Tensor:
    return torch.randn(_AUDIO_C, _AUDIO_T, _AUDIO_MEL)  # [C=8, T, mel=16]


def _sample(video, audio, *, audio_key="audio_latent_conditions", n_text=4, ctx_dim=8) -> dict:
    batch = {
        "latent_conditions": {"latents": video},
        "text_conditions": {
            "video_prompt_embeds": torch.randn(n_text, ctx_dim),
            "prompt_attention_mask": torch.ones(n_text, dtype=torch.long),
        },
        "idx": 0,
    }
    if audio is not None:
        batch[audio_key] = audio
    return batch


# --------------------------------------------------------------------------------------------------
# VIDEO generated in full + AUDIO frozen conditioning
# --------------------------------------------------------------------------------------------------


def test_video_fully_generated_all_in_loss() -> None:
    torch.manual_seed(0)
    strat = _make_a2v(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_video_latents(), _audio_latents()), rng=np.random.default_rng(0)
    )
    assert isinstance(inputs, ModelInputs)
    # every video token generated at the sampled sigma, all in the loss.
    assert torch.all(inputs.video.timesteps == 0.5)
    assert bool(inputs.video_loss_mask.all())
    assert inputs.video.latent.shape[1] == 12
    assert inputs.video_targets.shape[1] == 12
    assert inputs.ref_seq_len is None


def test_audio_is_frozen_conditioning() -> None:
    torch.manual_seed(0)
    strat = _make_a2v(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_video_latents(), _audio_latents()), rng=np.random.default_rng(0)
    )
    assert inputs.audio is not None
    # frozen: clean latents at timestep 0, sigma 0.
    assert torch.all(inputs.audio.timesteps == 0)
    assert torch.all(inputs.audio.sigma == 0)
    # audio latent patchified to [B, T, C*mel] = [1, T, 128]; ONE positional dim.
    assert tuple(inputs.audio.latent.shape) == (1, _AUDIO_T, _AUDIO_C * _AUDIO_MEL)
    assert tuple(inputs.audio.positions.shape) == (1, 1, _AUDIO_T, 2)
    # excluded from the loss — never a training target.
    assert inputs.audio_targets is None
    assert inputs.audio_loss_mask is None


def test_get_data_sources_declares_audio_latents() -> None:
    strat = _make_a2v()
    assert strat.get_data_sources() == ["latents", "conditions", "audio_latents"]


def test_audio_dict_payload_accepted() -> None:
    torch.manual_seed(0)
    strat = _make_a2v(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_video_latents(), {"latents": _audio_latents()}), rng=np.random.default_rng(0)
    )
    assert tuple(inputs.audio.latent.shape) == (1, _AUDIO_T, _AUDIO_C * _AUDIO_MEL)


def test_audio_latents_fallback_key_accepted() -> None:
    torch.manual_seed(0)
    strat = _make_a2v(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_video_latents(), _audio_latents(), audio_key="audio_latents"),
        rng=np.random.default_rng(0),
    )
    assert inputs.audio is not None


def test_missing_audio_source_fails_loud() -> None:
    torch.manual_seed(0)
    strat = _make_a2v(schedule=_FixedSchedule(0.5))
    with pytest.raises(KeyError, match="audio_latents"):
        strat.prepare_training_inputs(
            _sample(_video_latents(), None), rng=np.random.default_rng(0)
        )


def test_audio_batch_mismatch_fails_loud() -> None:
    torch.manual_seed(0)
    strat = _make_a2v(schedule=_FixedSchedule(0.5))
    bad_audio = torch.randn(2, _AUDIO_C, _AUDIO_T, _AUDIO_MEL)  # [B=2, C, T, mel]
    with pytest.raises(ValueError, match="audio latent batch"):
        strat.prepare_training_inputs(
            _sample(_video_latents(), bad_audio), rng=np.random.default_rng(0)
        )


def test_compute_loss_scalar_video_only() -> None:
    torch.manual_seed(0)
    strat = _make_a2v(schedule=_FixedSchedule(0.5))
    inputs = strat.prepare_training_inputs(
        _sample(_video_latents(), _audio_latents()), rng=np.random.default_rng(0)
    )
    loss = strat.compute_loss(inputs, torch.randn(1, 12, 2))
    assert loss.dim() == 0
    # perturbing any video token changes the loss (all are in it).
    perturbed = torch.randn(1, 12, 2)
    assert not torch.allclose(loss, strat.compute_loss(inputs, perturbed), atol=1e-6)


# --------------------------------------------------------------------------------------------------
# train/step.py dispatch + audio threading
# --------------------------------------------------------------------------------------------------


def test_training_step_dispatches_a2v_and_threads_audio() -> None:
    from signet_trainer.train.step import training_step

    torch.manual_seed(0)
    batch = _sample(_video_latents(), _audio_latents())
    captured: dict = {}

    def _model(video=None, audio=None, perturbations=None):  # noqa: ANN001
        captured["audio"] = audio
        captured["video_timesteps"] = video.timesteps
        return torch.zeros(1, video.latent.shape[1], 2), None

    # inject the a2v audio stubs onto the strategy via a patched dispatch: training_step builds the
    # strategy internally, so patch A2VStrategy's default audio deps by monkeypatching is overkill —
    # instead drive with a schedule + deps and rely on the strategy building audio deps function-local
    # would import ltx_core. So we assert the dispatch reaches the strategy by giving it the stubs via
    # the module-level construction path: training_step constructs A2VStrategy(deps, schedule, ...)
    # WITHOUT the stubs, so its _audio_deps would import ltx_core. Guard: expect that import to be the
    # only failure, proving the a2v branch was selected.
    import signet_trainer.conditioning.a2v as a2v_mod

    orig = a2v_mod.A2VStrategy

    class _Stubbed(orig):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            k.setdefault("audio_patchifier", _StubAudioPatchifier())
            k.setdefault("audio_latent_shape_cls", _StubAudioShape)
            super().__init__(*a, **k)

    a2v_mod.A2VStrategy = _Stubbed
    try:
        loss = training_step(
            _model,
            batch,
            _FixedSchedule(0.5),
            np.random.default_rng(0),
            device="cpu",
            dtype=torch.float32,
            deps=_make_deps(),
            conditioning_mode="audio_to_video",
        )
    finally:
        a2v_mod.A2VStrategy = orig

    assert loss.dim() == 0
    # the frozen audio Modality reached the model call (item 5: audio=inputs.audio).
    assert captured["audio"] is not None
    assert torch.all(captured["audio"].timesteps == 0)
    # the video was fully generated (no conditioning tokens).
    assert torch.all(captured["video_timesteps"] == 0.5)


def test_build_conditioning_kwargs_a2v_threads_only_mode() -> None:
    from signet_trainer.train.loop import build_conditioning_kwargs

    cond = SimpleNamespace(mode="audio_to_video")
    # a2v needs no extra per-step knob — only the mode is threaded (no silently-ignored kwargs).
    assert build_conditioning_kwargs(cond) == {"conditioning_mode": "audio_to_video"}


# --------------------------------------------------------------------------------------------------
# CPU dry-run gate coverage (dryrun/shapes.py audio_to_video)
# --------------------------------------------------------------------------------------------------


def _load_a2v_cfg():
    """A validated example config flipped to a fully-legal a2v posture via model_copy."""
    from pathlib import Path

    from signet_trainer.config.load import load_config

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(str(repo / "configs" / "ltx23_lora.example.yaml"))
    a2v_targets = list(cfg.lora.target_modules) + [
        "audio_to_video_attn.to_q",
        "audio_to_video_attn.to_k",
        "audio_to_video_attn.to_v",
        "audio_to_video_attn.to_out.0",
    ]
    return cfg.model_copy(
        update={
            "conditioning": cfg.conditioning.model_copy(update={"mode": "audio_to_video"}),
            "lora": cfg.lora.model_copy(update={"target_modules": a2v_targets}),
            "audio": cfg.audio.model_copy(update={"with_audio": True}),
        }
    )


def test_dryrun_a2v_mode_passes() -> None:
    from signet_trainer.dryrun.shapes import run_dryrun

    assert run_dryrun(_load_a2v_cfg()) == 0


def test_dryrun_a2v_audio_frozen_and_video_generated() -> None:
    from signet_trainer.dryrun.shapes import _A2V_AUDIO_TIME_STEPS, build_dryrun_inputs

    mi = build_dryrun_inputs(_load_a2v_cfg())
    # video: all generated.
    assert bool(mi.video_loss_mask.all())
    # audio: frozen (timestep 0), one positional dim, excluded from loss.
    assert mi.audio is not None
    assert torch.all(mi.audio.timesteps == 0)
    assert tuple(mi.audio.positions.shape)[:2] == (1, 1)
    assert mi.audio.positions.shape[2] == _A2V_AUDIO_TIME_STEPS
    assert mi.audio_targets is None and mi.audio_loss_mask is None


def test_dryrun_a2v_does_not_regress_other_modes() -> None:
    from pathlib import Path

    from signet_trainer.config.load import load_config
    from signet_trainer.dryrun.shapes import run_dryrun

    repo = Path(__file__).resolve().parents[1]
    assert run_dryrun(load_config(str(repo / "configs" / "ltx23_lora.example.yaml"))) == 0


# --------------------------------------------------------------------------------------------------
# import-confinement (Anti-Pattern 6 / Pitfall 6)
# --------------------------------------------------------------------------------------------------


def test_module_is_import_confined() -> None:
    import signet_trainer.conditioning.a2v as a2v_mod

    with open(a2v_mod.__file__, encoding="utf-8") as fh:
        text = fh.read()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
            assert "modal" not in stripped
            assert "ltx_core" not in stripped
            assert "ltx_trainer" not in stripped
