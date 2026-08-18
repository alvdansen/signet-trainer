"""tests/test_h3_audio_latents_layout.py — #21 finding 2: the audio-latent WIRE FORMAT.

The defect
----------
``encode_h3_audio_latents`` (``prep/h3_encode.py:996-1013``, pre-fix) returned the raw audio-VAE
posterior verbatim — audio-channel axis at dim 0 (batch; stereo is ``batch_size=2``, never two
channels on axis 1 — ``h3_vae_contract.assert_h3_audio_vae_encode_input`` names this), the VAE's
own ``EXPECTED_H3_AUDIO_IN_CHANNELS``-wide latent axis at dim 1, time last. Every consumer instead
wants CHANNEL-MAJOR ROWS — one row per ``(audio_channel, time)`` pair, audio-channel SLOWEST — the
exact order ``conditioning/h3_packing.py::_fill_h3_audio_positions`` assumes
(``time.repeat(audio_channels)``) and the count ``conditioning/h3_geometry.h3_audio_rows`` prices.
Nothing performed that transpose before writing the cache, so the whole audio arm (``audio_in_loss``,
``n_cond_audio``, the mode-vs-sample posterior policy) was built on a wire format the encoder never
produced. Marked unverified/unexercised in the source issue (0 of 44 corpus clips carry an audio
stream) — these are CPU-only structural proofs, zero GPU, zero real weights.

What is pinned here
--------------------
1. The returned tensor is reshaped to ``[audio_channels * num_audio_latents,
   EXPECTED_H3_AUDIO_IN_CHANNELS]`` with audio_channel the SLOWEST-varying axis (not merely the
   right total row count — a channel-minor flatten would pass a shape check and still be wrong).
2. An optional ``pixel_frames`` kwarg cross-checks the reshaped row count against
   ``h3_audio_rows(pixel_frames)`` and fails loud on disagreement — mirroring
   ``encode_h3_video_latents``'s own ``pixel_frames`` cross-check (the same shape-of-contract
   pattern, not a new one).
3. Omitting ``pixel_frames`` (every existing caller before this fix, and the two other pre-existing
   behavioural tests) is unaffected — the cross-check is opt-in, never a default-on refusal.

House doctrine reminder: no hand-rolled VAE stub (D-10-DEF-9's lesson) — every test here drives the
SANCTIONED ``H3AudioVaeContractStub``, monkeypatching only its already-validated ``.encode()``
return value to inject a per-``(channel, time)`` DISTINGUISHABLE payload so row ORDER (not just
row count) is provably checked.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from signet_trainer.conditioning.h3_geometry import h3_audio_rows
from signet_trainer.models.h3_loader import EXPECTED_H3_AUDIO_IN_CHANNELS
from signet_trainer.prep.h3_encode import encode_h3_audio_latents
from signet_trainer.prep.h3_vae_contract import H3_AUDIO_VAE_HOP_LENGTH, H3AudioVaeContractStub


class _DistinguishablePosterior:
    """The same two-method surface real/stub posteriors expose, holding a KNOWN tensor."""

    def __init__(self, moments: torch.Tensor) -> None:
        self._moments = moments

    def sample(self, generator: Any = None) -> torch.Tensor:  # noqa: ARG002 — signature parity
        return self._moments

    def mode(self) -> torch.Tensor:
        return self._moments


def _stub_with_distinguishable_payload(
    monkeypatch: pytest.MonkeyPatch, num_channels: int, num_time: int
) -> Any:
    """The SANCTIONED ``H3AudioVaeContractStub``, its ``.encode()`` wrapped to inject a payload
    where ``moments[b, :, t] == b * 1000 + t`` — every latent-channel column at (b, t) shares one
    value, so the post-transform row order is checkable without depending on the real 32-wide
    latent-channel content (never meaningful for this defect; only the (channel, time) axes are).
    """
    audio_vae = H3AudioVaeContractStub()
    orig_encode = audio_vae.encode  # the sanctioned stub's OWN input/device validation still runs

    def _encode(sample: Any, return_dict: bool = True) -> tuple:  # noqa: FBT001, FBT002
        (posterior,) = orig_encode(sample, return_dict=False)
        batch, latent_channels, time = posterior.mode().shape
        assert (batch, time) == (num_channels, num_time), (
            f"the stub's own ceil(samples/hop) math produced ({batch}, {time}), expected "
            f"({num_channels}, {num_time}) — the waveform sample count picked in this test no "
            "longer matches H3_AUDIO_VAE_HOP_LENGTH."
        )
        moments = torch.zeros(batch, latent_channels, time)
        for b in range(batch):
            for t in range(time):
                moments[b, :, t] = b * 1000 + t
        return (_DistinguishablePosterior(moments),)

    monkeypatch.setattr(audio_vae, "encode", _encode)
    return audio_vae


def test_channel_major_row_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """#21 finding 2: rows are (audio_channel, time) with audio_channel SLOWEST, not channel-minor."""
    num_channels, num_time = 2, 3
    audio_vae = _stub_with_distinguishable_payload(monkeypatch, num_channels, num_time)
    waveform = torch.zeros(num_channels, 1, H3_AUDIO_VAE_HOP_LENGTH * num_time, dtype=torch.float32)

    latents = encode_h3_audio_latents(audio_vae, waveform, is_reference=False)

    assert latents is not None
    assert latents.shape == (num_channels * num_time, EXPECTED_H3_AUDIO_IN_CHANNELS)
    for b in range(num_channels):
        for t in range(num_time):
            row = b * num_time + t  # audio_channel SLOWEST: block b holds rows [b*T, (b+1)*T)
            expected = torch.full((EXPECTED_H3_AUDIO_IN_CHANNELS,), float(b * 1000 + t))
            assert torch.equal(latents[row], expected), (
                f"row {row} must carry the (audio_channel={b}, time={t}) block's value — a "
                "channel-minor (time-slowest) flatten would put this value at a different row."
            )


def test_row_count_matches_h3_audio_rows_when_pixel_frames_agrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pixel_frames=2 encodes h3_audio_rows(2) == 6 rows for this stub's 3-latent, 2-channel VAE."""
    num_channels, num_time = 2, 3
    assert h3_audio_rows(2) == num_channels * num_time, (
        "this test's pixel_frames fixture must agree with the stub's (channels, time) — recompute "
        "if H3_FPS / H3_AUDIO_LATENTS_PER_SECOND / H3_AUDIO_CHANNELS ever change."
    )
    audio_vae = _stub_with_distinguishable_payload(monkeypatch, num_channels, num_time)
    waveform = torch.zeros(num_channels, 1, H3_AUDIO_VAE_HOP_LENGTH * num_time, dtype=torch.float32)

    latents = encode_h3_audio_latents(
        audio_vae, waveform, is_reference=False, pixel_frames=2
    )

    assert latents is not None
    assert latents.shape[0] == 6


def test_pixel_frames_mismatch_raises_before_the_cache_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#21 finding 2: a disagreeing pixel_frames fails loud, naming both row counts."""
    num_channels, num_time = 2, 3
    bad_pixel_frames = 5
    assert h3_audio_rows(bad_pixel_frames) != num_channels * num_time, (
        "this test's bad_pixel_frames fixture must DISAGREE with the stub's row count — recompute "
        "if H3_FPS / H3_AUDIO_LATENTS_PER_SECOND / H3_AUDIO_CHANNELS ever change."
    )
    audio_vae = _stub_with_distinguishable_payload(monkeypatch, num_channels, num_time)
    waveform = torch.zeros(num_channels, 1, H3_AUDIO_VAE_HOP_LENGTH * num_time, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="channel-major audio row"):
        encode_h3_audio_latents(
            audio_vae, waveform, is_reference=False, pixel_frames=bad_pixel_frames
        )


def test_pixel_frames_omitted_skips_the_cross_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every pre-fix caller passed no pixel_frames — the cross-check must stay strictly opt-in."""
    num_channels, num_time = 2, 3
    audio_vae = _stub_with_distinguishable_payload(monkeypatch, num_channels, num_time)
    waveform = torch.zeros(num_channels, 1, H3_AUDIO_VAE_HOP_LENGTH * num_time, dtype=torch.float32)

    latents = encode_h3_audio_latents(audio_vae, waveform, is_reference=False)

    assert latents is not None  # no pixel_frames -> no cross-check -> no raise, regardless of count
