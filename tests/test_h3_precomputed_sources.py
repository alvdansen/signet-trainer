"""Phase 10 (H3-03) — the four H3 precomputed source dirs pair by rel path and normalize correctly.

``PrecomputedDataset`` is source-count-agnostic: a new source pairs by relative path with NO code
change. So registering H3 needs exactly ONE decision — which of its sources are VIDEO-VAE latents and
therefore belong in the ``_VIDEO_LATENT_SOURCE_DIRS`` normalization allowlist.

The H3-prefixed dir convention (``h3_latents`` / ``h3_conditions`` / ``h3_reference_latents`` /
``h3_audio_latents``) exists so the LTX path stays byte-identical and an H3 cache can never be paired
into an LTX run. Two of the four are allowlisted:

  * ``h3_latents``, ``h3_reference_latents`` -> video-VAE latent dicts ``{"latents": [C,F,H,W], ...}``
  * ``h3_audio_latents``                     -> AUDIO-VAE latents, NO ``"latents"`` key
  * ``h3_conditions``                        -> text embeddings, NO ``"latents"`` key

The last two must NOT be allowlisted: the allowlist replaced an old ``"latent" in dir_name`` SUBSTRING
test precisely because a name CONTAINING "latent" is not necessarily a video latent, and routing an
audio payload through ``_normalize_video_latents`` raises on the missing key.

CPU-only, tensors on CPU — zero GPU, zero Modal spend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from signet_trainer.data.precomputed import (  # noqa: E402 — importorskip gate must precede
    _VIDEO_LATENT_SOURCE_DIRS,
    PrecomputedDataset,
)

# H3's video VAE is in_channels=24 (NOT LTX's 128) — the H3 pre-encode is signet-native, so this dict
# shape is OUR choice, and this plan commits to the same {"latents": [C,F,H,W], ...} shape LTX uses.
H3_LATENT_CHANNELS = 24

#: The committed H3 source map: dir name -> output key.
H3_SOURCES = {
    "h3_latents": "latent_conditions",
    "h3_conditions": "text_conditions",
    "h3_reference_latents": "ref_latent_conditions",
    "h3_audio_latents": "audio_latent_conditions",
}


def _write_h3_video_latent(path: Path, *, frames: int = 2, hw: int = 3) -> None:
    """A non-patchified H3 video latent: {"latents": [C, F, H, W], ...}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.randn(H3_LATENT_CHANNELS, frames, hw, hw),
            "num_frames": frames,
            "height": hw,
            "width": hw,
        },
        path,
    )


def _write_h3_condition(path: Path) -> None:
    """H3 text conditioning: an embedding dict with NO "latents" key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompt_embeds": torch.randn(6, 5120),  # H3 text_dim = 5120
            "prompt_attention_mask": torch.ones(6, dtype=torch.bool),
        },
        path,
    )


def _write_h3_audio_latent(path: Path) -> None:
    """H3 audio latents: a BARE tensor. Routing this through _normalize_video_latents would raise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.randn(32, 40), path)  # [audio_in_channels=32, latent rows]


def _build_cache(root: Path, rel: str = "a/b.pt") -> None:
    """A complete four-source H3 cache; ``rel`` is deliberately NESTED to exercise rel-path pairing."""
    _write_h3_video_latent(root / "h3_latents" / rel)
    _write_h3_condition(root / "h3_conditions" / rel)
    _write_h3_video_latent(root / "h3_reference_latents" / rel)
    _write_h3_audio_latent(root / "h3_audio_latents" / rel)


# ------------------------------------------------------------------------------------------------
# 1. The allowlist itself
# ------------------------------------------------------------------------------------------------


def test_allowlist_contains_exactly_the_four_video_latent_sources() -> None:
    assert _VIDEO_LATENT_SOURCE_DIRS == {
        "latents",
        "reference_latents",
        "h3_latents",
        "h3_reference_latents",
    }


def test_ltx_allowlist_entries_are_unchanged() -> None:
    """Regression guard on the EXISTING path — registering H3 must not disturb LTX."""
    assert "latents" in _VIDEO_LATENT_SOURCE_DIRS
    assert "reference_latents" in _VIDEO_LATENT_SOURCE_DIRS


def test_h3_audio_and_conditions_are_deliberately_not_allowlisted() -> None:
    """``h3_audio_latents`` CONTAINS "latent" but is an audio payload — the substring trap."""
    assert "h3_audio_latents" not in _VIDEO_LATENT_SOURCE_DIRS
    assert "h3_conditions" not in _VIDEO_LATENT_SOURCE_DIRS


# ------------------------------------------------------------------------------------------------
# 2. Pairing — four sources, one sample, keyed by the declared output keys
# ------------------------------------------------------------------------------------------------


def test_four_h3_sources_pair_into_one_sample(tmp_path: Path) -> None:
    root = tmp_path / ".precomputed"
    _build_cache(root)

    ds = PrecomputedDataset(str(root), data_sources=H3_SOURCES)
    assert len(ds) == 1

    sample = ds[0]
    assert set(sample) == set(H3_SOURCES.values()) | {"idx"}
    assert sample["idx"] == 0

    # Each payload arrives as its own shape, unmixed.
    assert sample["latent_conditions"]["latents"].shape[0] == H3_LATENT_CHANNELS
    assert sample["ref_latent_conditions"]["latents"].shape[0] == H3_LATENT_CHANNELS
    assert sample["text_conditions"]["prompt_embeds"].shape == (6, 5120)
    assert isinstance(sample["audio_latent_conditions"], torch.Tensor)
    assert tuple(sample["audio_latent_conditions"].shape) == (32, 40)


def test_audio_and_text_sources_pass_through_unnormalized(tmp_path: Path) -> None:
    """Neither is allowlisted, so neither hits ``_normalize_video_latents`` (which would raise)."""
    root = tmp_path / ".precomputed"
    _build_cache(root)

    sample = PrecomputedDataset(str(root), data_sources=H3_SOURCES)[0]
    # A bare tensor survives untouched; a dict with no "latents" key survives untouched.
    assert isinstance(sample["audio_latent_conditions"], torch.Tensor)
    assert "latents" not in sample["text_conditions"]


def test_missing_pair_raises_the_existing_pairing_error(tmp_path: Path) -> None:
    """A source dir present but missing its counterpart file yields zero paired samples."""
    root = tmp_path / ".precomputed"
    _write_h3_video_latent(root / "h3_latents" / "a/b.pt")
    # Dir exists (so _setup_source_paths passes) but the file does not pair.
    _write_h3_condition(root / "h3_conditions" / "a/DIFFERENT.pt")

    with pytest.raises(ValueError, match="No valid samples found"):
        PrecomputedDataset(
            str(root),
            data_sources={"h3_latents": "latent_conditions", "h3_conditions": "text_conditions"},
        )


def test_h3_conditions_does_not_inherit_the_legacy_condition_rename(tmp_path: Path) -> None:
    """The legacy ``latent_*`` -> ``condition_*`` rename is gated on ``dir_name == "conditions"``.

    This is the payoff of the h3_ PREFIX: an H3 cache pairs purely by rel path, so a file named
    ``latent_x.pt`` stays ``latent_x.pt`` under ``h3_conditions/`` instead of being looked up as
    ``condition_x.pt``. The LTX path keeps its legacy behavior, byte-identical.
    """
    root = tmp_path / ".precomputed"
    _write_h3_video_latent(root / "h3_latents" / "latent_x.pt")
    _write_h3_condition(root / "h3_conditions" / "latent_x.pt")

    ds = PrecomputedDataset(
        str(root),
        data_sources={"h3_latents": "latent_conditions", "h3_conditions": "text_conditions"},
    )
    assert len(ds) == 1
    assert ds[0]["text_conditions"]["prompt_embeds"].shape == (6, 5120)


# ------------------------------------------------------------------------------------------------
# 3. Normalization routing — the actual behavior the allowlist controls
# ------------------------------------------------------------------------------------------------


def test_allowlisted_h3_sources_actually_reach_the_normalizer(tmp_path: Path) -> None:
    """einops-FREE routing discriminator (``einops`` is a lazy optional dep and usually absent).

    ``_normalize_video_latents`` reads ``data["latents"]`` unconditionally. So a payload WITHOUT that
    key raises iff its source dir is allowlisted — which distinguishes "routed" from "not routed"
    without needing the legacy patchified conversion to be installable. This is the exact landmine
    the allowlist exists for, run forwards.
    """
    root = tmp_path / ".precomputed"
    keyless = {"not_latents": torch.randn(2, 2)}

    (root / "h3_latents").mkdir(parents=True)
    torch.save(keyless, root / "h3_latents" / "a.pt")
    _write_h3_condition(root / "h3_conditions" / "a.pt")

    ds = PrecomputedDataset(
        str(root),
        data_sources={"h3_latents": "latent_conditions", "h3_conditions": "text_conditions"},
    )
    with pytest.raises(RuntimeError, match="Failed to load latent_conditions"):
        _ = ds[0]


def test_non_allowlisted_h3_sources_never_reach_the_normalizer(tmp_path: Path) -> None:
    """The mirror image: the SAME keyless payload under a NON-allowlisted source loads fine."""
    root = tmp_path / ".precomputed"
    keyless = {"not_latents": torch.randn(2, 2)}

    _write_h3_video_latent(root / "h3_latents" / "a.pt")
    (root / "h3_audio_latents").mkdir(parents=True)
    torch.save(keyless, root / "h3_audio_latents" / "a.pt")

    ds = PrecomputedDataset(
        str(root),
        data_sources={
            "h3_latents": "latent_conditions",
            "h3_audio_latents": "audio_latent_conditions",
        },
    )
    assert "not_latents" in ds[0]["audio_latent_conditions"]


def test_h3_video_sources_route_through_video_latent_normalization(tmp_path: Path) -> None:
    """Proof of ROUTING: a legacy patchified ``[seq_len, C]`` payload is converted to ``[C,F,H,W]``.

    Only an allowlisted source gets this fixup, so a still-2-dim tensor here means ``h3_latents`` /
    ``h3_reference_latents`` were NOT allowlisted.
    """
    pytest.importorskip("einops")
    root = tmp_path / ".precomputed"
    frames, hw = 2, 3
    seq_len = frames * hw * hw

    for dir_name in ("h3_latents", "h3_reference_latents"):
        path = root / dir_name / "a/b.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latents": torch.randn(seq_len, H3_LATENT_CHANNELS),  # patchified [seq_len, C]
                "num_frames": frames,
                "height": hw,
                "width": hw,
            },
            path,
        )
    _write_h3_condition(root / "h3_conditions" / "a/b.pt")
    _write_h3_audio_latent(root / "h3_audio_latents" / "a/b.pt")

    sample = PrecomputedDataset(str(root), data_sources=H3_SOURCES)[0]
    for key in ("latent_conditions", "ref_latent_conditions"):
        latents = sample[key]["latents"]
        assert latents.dim() == 4, f"{key} was not normalized — is its source dir allowlisted?"
        assert tuple(latents.shape) == (H3_LATENT_CHANNELS, frames, hw, hw)
