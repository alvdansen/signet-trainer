"""Shared pytest fixtures for the signet_trainer contract tests.

CPU-only synthetic-tensor helpers. Nothing here imports modal / ltx_core / CUDA, so the
contract suite runs free on Windows/CI (Anti-Pattern 6 / CONF-03 zero-spend).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from signet_trainer.conditioning.strategy import (
    Modality,
    ModelInputs,
    compute_seq_len,
)

# --------------------------------------------------------------------------- LIVE harness guard

#: The LIVE per-project harness state (DECISION-LOG.md, KNOWLEDGE.md, SESSION-STATE.json, the
#: campaign cards). It is a WORKING-REPO planning surface: operator-written, per-project, and
#: deliberately excluded from a release cut — unlike the SPEC/TEMPLATE data, which now ships as
#: package data under ``signet_trainer.harness_data``.
LIVE_HARNESS_DIR = Path(__file__).resolve().parents[1] / ".planning" / "harness"

_LIVE_HARNESS_SKIP_REASON = (
    "no LIVE harness state at .planning/harness/ — that directory is a working-repo planning "
    "surface (DECISION-LOG / KNOWLEDGE / SESSION-STATE / campaign cards) and is deliberately not "
    "shipped. These tests lint the maintainer's memory scaffold, not the package; the shipped "
    "SPEC/TEMPLATE data is covered by tests/test_harness_data.py instead."
)


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``requires_live_harness`` marker (no ``--strict-markers`` surprises)."""
    config.addinivalue_line(
        "markers",
        "requires_live_harness: test lints the LIVE .planning/harness/ scaffold; skipped when "
        "that working-repo planning surface is absent (e.g. in a release cut).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip LIVE-scaffold tests when there is no scaffold to lint.

    In the working repo ``.planning/harness/`` exists, so these RUN and hard-fail exactly as
    before — nothing is weakened where it matters. In a release tree the directory is absent BY
    DESIGN, and a red suite there would tell a fresh user "your install is broken" when it is not.
    Applied as a collection hook rather than a decorator import so test modules stay
    import-independent of conftest.
    """
    if LIVE_HARNESS_DIR.is_dir():
        return
    skip = pytest.mark.skip(reason=_LIVE_HARNESS_SKIP_REASON)
    for item in items:
        if "requires_live_harness" in item.keywords:
            item.add_marker(skip)

# A known [W, H, F] and its verified expected seq_len.
#   latent_frames = (49 - 1) // 8 + 1 = 7
#   latent_h      = 512 // 32         = 16
#   latent_w      = 768 // 32         = 24
#   seq_len       = 7 * 16 * 24       = 2688
SAMPLE_WHF: tuple[int, int, int] = (768, 512, 49)
SAMPLE_SEQ_LEN: int = 2688
VIDEO_CHANNELS: int = 128  # LTX-2.3 video latent channel count D


@pytest.fixture
def sample_whf() -> tuple[int, int, int]:
    return SAMPLE_WHF


@pytest.fixture
def sample_seq_len() -> int:
    return SAMPLE_SEQ_LEN


def build_synthetic_modality(
    w: int,
    h: int,
    f: int,
    *,
    n_text_tokens: int = 8,
    ctx_dim: int = 16,
    cond_first_frame: bool = False,
) -> Modality:
    """Build a CPU synthetic video ``Modality`` from raw [W, H, F] dims.

    Shapes mirror the real ltx_core ``Modality`` contract (RESEARCH.md Q4). No model, no VAE.
    """
    seq_len = compute_seq_len(w, h, f)
    latent = torch.zeros(1, seq_len, VIDEO_CHANNELS)

    cond_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    if cond_first_frame:
        # First-frame conditioning region = one latent frame = (H//32)*(W//32) tokens.
        first_frame_tokens = (h // 32) * (w // 32)
        cond_mask[:, :first_frame_tokens] = True

    # Conditioning tokens get timestep 0; target tokens get the sampled sigma (0.5 here).
    timesteps = torch.where(cond_mask, torch.zeros(1), torch.full((1,), 0.5))
    positions = torch.zeros(1, 3, seq_len, 2)
    context = torch.zeros(1, n_text_tokens, ctx_dim)

    return Modality(
        latent=latent,
        sigma=torch.tensor([0.5]),
        timesteps=timesteps,
        positions=positions,
        context=context,
        context_mask=torch.ones(1, n_text_tokens, dtype=torch.bool),
    )


def build_synthetic_model_inputs(
    w: int,
    h: int,
    f: int,
    *,
    cond_first_frame: bool = False,
) -> ModelInputs:
    """Build a CPU synthetic ``ModelInputs`` (video-only) from raw [W, H, F] dims."""
    video = build_synthetic_modality(w, h, f, cond_first_frame=cond_first_frame)
    seq_len = video.latent.shape[1]

    cond_mask = video.timesteps == 0  # timestep 0 marks conditioning tokens
    video_loss_mask = ~cond_mask
    video_targets = torch.randn(1, seq_len, VIDEO_CHANNELS)

    return ModelInputs(
        video=video,
        audio=None,
        video_targets=video_targets,
        audio_targets=None,
        video_loss_mask=video_loss_mask,
        audio_loss_mask=None,
    )


@pytest.fixture
def synthetic_model_inputs() -> ModelInputs:
    return build_synthetic_model_inputs(*SAMPLE_WHF, cond_first_frame=True)
