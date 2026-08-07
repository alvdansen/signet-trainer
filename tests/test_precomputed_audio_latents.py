"""Regression proof for the a2v ``audio_latents`` normalization gate (GATE-SPEC rev 2 item 4).

``PrecomputedDataset.__getitem__`` used to route ANY source dir whose name CONTAINS "latent" through
``_normalize_video_latents`` (the legacy patchified-video fixup). ``audio_latents`` contains that
substring but is an AUDIO-VAE latent (a bare tensor / a payload without a video ``"latents"`` key), so
the old substring test would trip the normalizer and raise. The fix replaced the substring test with
an explicit ``_VIDEO_LATENT_SOURCE_DIRS`` allowlist ({latents, reference_latents}); these tests prove
``audio_latents`` passes through UN-normalized while ``latents`` / ``reference_latents`` still get the
fixup.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from signet_trainer.data.precomputed import (  # noqa: E402 — importorskip gate must precede
    _VIDEO_LATENT_SOURCE_DIRS,
    PrecomputedDataset,
)


def _write_video_latent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"latents": torch.randn(128, 1, 2, 2), "num_frames": 1, "height": 2, "width": 2}, path
    )


def _write_condition(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video_prompt_embeds": torch.randn(4, 4096),
            "prompt_attention_mask": torch.ones(4, dtype=torch.bool),
        },
        path,
    )


def _write_audio_latent_bare(path: Path) -> None:
    # The a2v audio-latent contract: a bare tensor with NO "latents" key. Under the OLD substring
    # branch this would have gone into _normalize_video_latents and raised (no "latents" key).
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(8, 5, 16), path)  # [C=8, T=5, mel=16]


def test_allowlist_excludes_audio_latents() -> None:
    assert "audio_latents" not in _VIDEO_LATENT_SOURCE_DIRS
    assert "latents" in _VIDEO_LATENT_SOURCE_DIRS
    assert "reference_latents" in _VIDEO_LATENT_SOURCE_DIRS


def test_audio_latents_pass_through_unnormalized(tmp_path: Path) -> None:
    root = tmp_path / ".precomputed"
    _write_video_latent(root / "latents" / "a.pt")
    _write_condition(root / "conditions" / "a.pt")
    _write_audio_latent_bare(root / "audio_latents" / "a.pt")

    ds = PrecomputedDataset(
        str(root),
        data_sources={
            "latents": "latent_conditions",
            "conditions": "text_conditions",
            "audio_latents": "audio_latent_conditions",
        },
    )
    sample = ds[0]
    # audio latents load as the bare tensor, untouched (the old substring path would have raised).
    audio = sample["audio_latent_conditions"]
    assert isinstance(audio, torch.Tensor)
    assert tuple(audio.shape) == (8, 5, 16)
    # the VIDEO latents still get the [C,F,H,W] normalization (allowlisted).
    assert sample["latent_conditions"]["latents"].dim() == 4


def test_video_latents_still_normalized_when_patchified(tmp_path: Path) -> None:
    # A legacy patchified [seq_len, C] video latent still routes through the einops fixup -> [C,F,H,W].
    pytest.importorskip("einops")
    root = tmp_path / ".precomputed"
    (root / "latents").mkdir(parents=True)
    torch.save(
        {"latents": torch.randn(4, 128), "num_frames": 1, "height": 2, "width": 2},
        root / "latents" / "a.pt",
    )
    _write_condition(root / "conditions" / "a.pt")

    ds = PrecomputedDataset(str(root))
    latents = ds[0]["latent_conditions"]["latents"]
    assert latents.dim() == 4  # normalized from [seq_len, C] to [C, F, H, W]
