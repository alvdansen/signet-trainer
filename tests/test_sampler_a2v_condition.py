"""Phase 9 (AUDIO-TO-VIDEO) — CPU gates for the driving-audio validation-render branch.

Covers the pure/CPU + injectable layer of ``inference/sampler.py``'s a2v additions
(GATE-SPEC-inpaint-a2v rev 2, item 7) WITHOUT the audio VAE, torchaudio, or the ltx stack:

  * ``_fit_waveform_to_duration`` — the trim/zero-pad-to-video-duration contract, VERBATIM the pin's
    ``process_videos._extract_audio`` (trim on the sample axis / right zero-pad).
  * ``_audio_latent_payload`` — the encoder-output ``[B,C,T,F]`` → training-parity payload dict
    (``latents[C,T,F]`` + ``num_time_steps`` / ``frequency_bins`` / ``duration``), incl. fail-loud on
    a non-4-dim encoder output. This is the format ``conditioning/a2v.py`` reads — kept identical.
  * ``build_frozen_audio_latent_state`` — the FROZEN audio state: all-zero ``denoise_mask`` (the
    sampler's ``sigma * denoise_mask`` → per-token timestep 0 + clean copy-back == training's frozen
    contract), positions ``[B,1,T,2]``, latent ``[B,T,C*mel]``, ``clean_latent == latent``. Driven
    with the SAME stubs ``test_a2v_strategy.py`` uses (audio patchifier / shape).
  * ``plan_audio_condition`` — dict AND real ``AudioCondition`` sub-model; fail-fast on wrong/missing.
  * a wiring test: ``_render_video_with_frozen_audio`` reaches ``sampler._run_denoising`` with
    ``audio_state`` NON-None and frozen (denoise_mask all zeros) — the model-call boundary, proved
    with a fake sampler + injected fake ltx_core modules (no GPU, no ltx install).

Import-confinement is asserted the same way ``test_sampler_mask_condition.py`` does.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from signet_trainer.config.schema import AudioCondition
from signet_trainer.inference.sampler import (
    _audio_latent_payload,
    _fit_waveform_to_duration,
    build_frozen_audio_latent_state,
    plan_audio_condition,
)

_AUDIO_C = 8
_AUDIO_MEL = 16
_AUDIO_T = 5  # a small synthetic audio time-step count


# --------------------------------------------------------------------------------------------------
# CPU stubs (reused from test_a2v_strategy.py — the SAME contract the training side proves under)
# --------------------------------------------------------------------------------------------------


class _StubAudioPatchifier:
    """Audio patchifier stub: [B, C, T, mel] -> [B, T, C*mel]; positions -> [B, 1, T, 2]."""

    def patchify(self, audio_4d: torch.Tensor) -> torch.Tensor:
        b, c, t, mel = audio_4d.shape
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


def _audio_latents_tensor() -> torch.Tensor:
    return torch.randn(_AUDIO_C, _AUDIO_T, _AUDIO_MEL)  # [C=8, T, mel=16]


# --------------------------------------------------------------------------------------------------
# _fit_waveform_to_duration — the _extract_audio trim/pad contract
# --------------------------------------------------------------------------------------------------


def test_fit_waveform_trims_when_longer() -> None:
    wav = torch.arange(200, dtype=torch.float32).unsqueeze(0)  # [1, 200], sr=100 -> 2.0s
    out = _fit_waveform_to_duration(wav, sample_rate=100, target_duration=1.0)
    assert out.shape == (1, 100)
    assert torch.equal(out, wav[:, :100])  # trims on the sample axis, keeps the head


def test_fit_waveform_pads_when_shorter() -> None:
    wav = torch.ones(2, 50)  # [channels=2, 50], sr=100 -> 0.5s
    out = _fit_waveform_to_duration(wav, sample_rate=100, target_duration=1.0)
    assert out.shape == (2, 100)
    assert torch.all(out[:, :50] == 1.0)  # original head preserved
    assert torch.all(out[:, 50:] == 0.0)  # right zero-pad


def test_fit_waveform_exact_is_identity() -> None:
    wav = torch.randn(2, 100)
    out = _fit_waveform_to_duration(wav, sample_rate=100, target_duration=1.0)
    assert torch.equal(out, wav)


# --------------------------------------------------------------------------------------------------
# _audio_latent_payload — training-parity payload (encode_audio format)
# --------------------------------------------------------------------------------------------------


def test_audio_payload_keys_and_shapes() -> None:
    latents = torch.randn(1, _AUDIO_C, _AUDIO_T, _AUDIO_MEL)  # [B, C, T, F]
    payload = _audio_latent_payload(latents, duration=2.5)
    assert set(payload) == {"latents", "num_time_steps", "frequency_bins", "duration"}
    # batch dim stripped -> [C, T, F], the exact per-sample shape conditioning/a2v.py reads.
    assert tuple(payload["latents"].shape) == (_AUDIO_C, _AUDIO_T, _AUDIO_MEL)
    assert payload["num_time_steps"] == _AUDIO_T
    assert payload["frequency_bins"] == _AUDIO_MEL
    assert payload["duration"] == pytest.approx(2.5)
    assert isinstance(payload["num_time_steps"], int)
    assert isinstance(payload["duration"], float)


def test_audio_payload_roundtrips_into_a2v_extractor() -> None:
    """The payload this render produces is exactly what A2VStrategy._extract_audio_latent accepts."""
    from signet_trainer.conditioning.a2v import A2VStrategy

    payload = _audio_latent_payload(torch.randn(1, _AUDIO_C, _AUDIO_T, _AUDIO_MEL), duration=1.0)
    # The training-side extractor accepts the {"latents": ...} dict form and returns the tensor.
    extracted = A2VStrategy._extract_audio_latent({"audio_latents": payload})
    assert tuple(extracted.shape) == (_AUDIO_C, _AUDIO_T, _AUDIO_MEL)


def test_audio_payload_rejects_non_4dim() -> None:
    with pytest.raises(ValueError, match="4-dim"):
        _audio_latent_payload(torch.randn(_AUDIO_C, _AUDIO_T, _AUDIO_MEL), duration=1.0)


# --------------------------------------------------------------------------------------------------
# build_frozen_audio_latent_state — frozen contract (timestep 0 via denoise_mask 0)
# --------------------------------------------------------------------------------------------------


def _build_state(audio_latents):  # noqa: ANN001
    return build_frozen_audio_latent_state(
        audio_latents,
        audio_patchifier=_StubAudioPatchifier(),
        audio_latent_shape_cls=_StubAudioShape,
        latent_state_cls=lambda **kw: SimpleNamespace(**kw),
    )


def test_frozen_state_denoise_mask_all_zero() -> None:
    """denoise_mask 0 everywhere -> the sampler pins every audio token clean at timestep 0 (frozen)."""
    state = _build_state(_audio_latents_tensor())
    assert tuple(state.denoise_mask.shape) == (1, _AUDIO_T, 1)  # upstream [B, seq, 1] shape
    assert torch.all(state.denoise_mask == 0.0)
    assert state.denoise_mask.dtype == torch.float32


def test_frozen_state_patchified_latent_and_positions() -> None:
    state = _build_state(_audio_latents_tensor())
    # latent patchified to [B, T, C*mel] = [1, T, 128] (the LTX audio token dim).
    assert tuple(state.latent.shape) == (1, _AUDIO_T, _AUDIO_C * _AUDIO_MEL)
    # ONE positional dim: [B, 1, T, 2] (NOT the video 3-dim coord path).
    assert tuple(state.positions.shape) == (1, 1, _AUDIO_T, 2)


def test_frozen_state_clean_latent_equals_latent() -> None:
    """The per-step copy-back reads clean_latent -> it must be the clean driving latent (== latent)."""
    state = _build_state(_audio_latents_tensor())
    assert torch.equal(state.latent, state.clean_latent)


def test_frozen_state_accepts_payload_dict() -> None:
    payload = _audio_latent_payload(torch.randn(1, _AUDIO_C, _AUDIO_T, _AUDIO_MEL), duration=1.0)
    state = _build_state(payload)  # dict form (the _encode_driving_audio output)
    assert tuple(state.latent.shape) == (1, _AUDIO_T, _AUDIO_C * _AUDIO_MEL)


def test_frozen_state_accepts_prebatched_tensor() -> None:
    state = _build_state(torch.randn(1, _AUDIO_C, _AUDIO_T, _AUDIO_MEL))  # already [B, C, T, mel]
    assert tuple(state.latent.shape) == (1, _AUDIO_T, _AUDIO_C * _AUDIO_MEL)


def test_frozen_state_rejects_bad_ndim() -> None:
    with pytest.raises(ValueError, match=r"\[C, T, mel\]"):
        _build_state(torch.randn(_AUDIO_T, _AUDIO_MEL))  # 2-dim, not an audio latent


# --------------------------------------------------------------------------------------------------
# plan_audio_condition — config-shape acceptance (dict AND the real AudioCondition sub-model)
# --------------------------------------------------------------------------------------------------


def test_plan_audio_condition_accepts_plain_dict() -> None:
    assert plan_audio_condition({"audio": "a2v/clip_01.wav"}) == "a2v/clip_01.wav"


def test_plan_audio_condition_accepts_type_discriminator() -> None:
    assert plan_audio_condition({"type": "audio", "audio": "a2v/x.wav"}) == "a2v/x.wav"


def test_plan_audio_condition_accepts_the_real_schema_model() -> None:
    cond = AudioCondition(audio="a2v/clip_02.wav")
    assert plan_audio_condition(cond) == "a2v/clip_02.wav"


def test_plan_audio_condition_accepts_attribute_objects() -> None:
    cond = SimpleNamespace(type="audio", audio="a2v/y.wav")
    assert plan_audio_condition(cond) == "a2v/y.wav"


def test_plan_audio_condition_rejects_wrong_type() -> None:
    with pytest.raises(ValueError, match="type='mask'"):
        plan_audio_condition({"type": "mask", "audio": "a2v/x.wav"})


def test_plan_audio_condition_rejects_missing_audio() -> None:
    with pytest.raises(ValueError, match="audio"):
        plan_audio_condition({"type": "audio"})


def test_plan_audio_condition_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="audio"):
        plan_audio_condition({"audio": ""})


# --------------------------------------------------------------------------------------------------
# wiring — _render_video_with_frozen_audio reaches _run_denoising with a FROZEN audio state
# --------------------------------------------------------------------------------------------------


class _FakeNoiser:
    """GaussianNoiser stub: frozen audio has denoise_mask 0, so noising is inert (identity)."""

    def __init__(self, generator=None) -> None:  # noqa: ANN001
        self.generator = generator

    def __call__(self, latent_state, noise_scale=1.0):  # noqa: ANN001
        return latent_state


class _FakeVideoTools:
    def create_initial_state(self, device=None, dtype=None):  # noqa: ANN001
        return SimpleNamespace(latent=torch.zeros(1, 4, 8), denoise_mask=torch.ones(1, 4, 1))

    def clear_conditioning(self, state):  # noqa: ANN001
        return state

    def unpatchify(self, state):  # noqa: ANN001
        return state


class _FakeSampler:
    def __init__(self) -> None:
        self._audio_patchifier = _StubAudioPatchifier()
        self.denoise_calls: list[dict] = []

    def _get_prompt_embeddings(self, cfg, device):  # noqa: ANN001
        return ("v_pos", "a_pos", "v_neg", "a_neg")

    def _create_video_latent_tools(self, cfg):  # noqa: ANN001
        return _FakeVideoTools()

    def _run_denoising(self, **kwargs):  # noqa: ANN003
        self.denoise_calls.append(kwargs)
        return kwargs["video_state"], None

    def _decode_video(self, video_state, device, tiled):  # noqa: ANN001
        return "DECODED_VIDEO"


def test_render_reaches_denoising_with_frozen_audio(monkeypatch) -> None:
    """The frozen audio state reaches the transformer-call boundary (_run_denoising), non-None."""
    import signet_trainer.inference.sampler as sampler_mod

    # Inject fake ltx_core modules so the render fn's function-local imports resolve on CI (no ltx
    # install). monkeypatch.setitem auto-restores, so import-confinement for OTHER tests is preserved.
    ltx_core = ModuleType("ltx_core")
    ltx_components = ModuleType("ltx_core.components")
    ltx_noisers = ModuleType("ltx_core.components.noisers")
    ltx_noisers.GaussianNoiser = _FakeNoiser
    ltx_types = ModuleType("ltx_core.types")
    ltx_types.AudioLatentShape = _StubAudioShape
    ltx_types.LatentState = lambda **kw: SimpleNamespace(**kw)
    for name, mod in {
        "ltx_core": ltx_core,
        "ltx_core.components": ltx_components,
        "ltx_core.components.noisers": ltx_noisers,
        "ltx_core.types": ltx_types,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    fake_sampler = _FakeSampler()
    monkeypatch.setattr(sampler_mod, "build_validation_sampler", lambda c, t: fake_sampler)

    cfg = SimpleNamespace(seed=42, num_frames=49, frame_rate=24.0, tiled_decoding=None)
    payload = _audio_latent_payload(torch.randn(1, _AUDIO_C, _AUDIO_T, _AUDIO_MEL), duration=1.0)

    video_out, audio_out = sampler_mod._render_video_with_frozen_audio(
        components=SimpleNamespace(), transformer=SimpleNamespace(), cfg=cfg,
        audio_latents=payload, device="cpu",
    )

    assert video_out == "DECODED_VIDEO"
    assert audio_out is None  # video-only output (generate_audio False)
    assert len(fake_sampler.denoise_calls) == 1
    call = fake_sampler.denoise_calls[0]
    # the audio state reached the model-call boundary, non-None ...
    assert call["audio_state"] is not None
    assert call["audio_clean_state"] is not None
    # ... and it is FROZEN: denoise_mask all zeros (per-token timestep 0, pinned clean each step).
    assert torch.all(call["audio_state"].denoise_mask == 0.0)
    assert tuple(call["audio_state"].positions.shape) == (1, 1, _AUDIO_T, 2)
    # the audio context is threaded so the audio branch can cross-attend.
    assert call["a_ctx_pos"] == "a_pos"


# --------------------------------------------------------------------------------------------------
# import confinement — the a2v branch must not un-confine the sampler module on CPU/CI
# --------------------------------------------------------------------------------------------------


def test_a2v_branch_keeps_sampler_import_confined_on_cpu() -> None:
    for mod in ("modal", "ltx_core", "ltx_trainer", "torchaudio"):
        sys.modules.pop(mod, None)

    from signet_trainer.inference.sampler import run_audio_condition_sampler  # noqa: F401

    for mod in ("modal", "ltx_core", "ltx_trainer", "torchaudio"):
        assert mod not in sys.modules, (
            f"import-confinement violation: the a2v branch transitively imported {mod!r}"
        )
