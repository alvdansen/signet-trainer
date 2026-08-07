"""Free CPU safetensors ground-truth pre-check — the resolved-A2 pre-spend gate (SC#1).

This is the genuinely zero-GPU sanity gate that runs BEFORE any metered Modal spend. It
reads ONLY the checkpoint header/metadata via ``safetensors.safe_open`` (no full tensor
load, no GPU, no ltx_core / modal) and asserts the derivable architecture facts against
``signet_trainer.models.loader``'s EXPECTED_* constants.

The 46GB LTX-2.3 checkpoint does NOT exist on Windows/CI, so the test resolves the path
from ``SIGNET_LTX_CHECKPOINT`` (env) and ``pytest.skip(...)`` cleanly when it is absent —
a no-op green on CI, a real gate when run against a mounted/downloaded checkpoint
(e.g. inside a Modal container or after ``download_weights``).

Mirrors tests/test_schema_compose.py's CPU, no-GPU, assert-against-known-constants style
(PATTERNS.md § tests/test_ground_truth_read.py). Does NOT import modal / ltx_core.

How the architecture facts are derived from the safetensors metadata (no values loaded):
    * num_blocks   — count of distinct transformer block indices in the tensor key names
                     (keys like ``transformer_blocks.<i>....`` / ``blocks.<i>....``).
    * hidden_dim   — the hidden axis of a per-block weight tensor's recorded shape.
    * in_channels  — the input-channel axis of the patch/proj-in tensor's recorded shape.
All three are read from ``f.get_slice(key).get_shape()`` (header-only) — never materialized.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from signet_trainer.models.loader import (
    EXPECTED_HIDDEN_DIM,
    EXPECTED_NUM_BLOCKS,
    EXPECTED_T2V_IN_CHANNELS,
    summarize_components,
)

#: Env var pointing at the local LTX-2.3 checkpoint (.safetensors). Absent on CI/Windows.
CHECKPOINT_ENV = "SIGNET_LTX_CHECKPOINT"

#: Matches a transformer block index in a tensor key, e.g.
#: "transformer_blocks.12.attn1.to_q.weight" / "blocks.0.ff.net.0.proj.weight".
_BLOCK_IDX_RE = re.compile(r"(?:^|\.)(?:transformer_)?blocks\.(\d+)\.")


def _resolve_checkpoint() -> Path | None:
    """Resolve the checkpoint path from the env var. Return None if unset/missing."""
    raw = os.environ.get(CHECKPOINT_ENV)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def test_safetensors_ground_truth() -> None:
    """Read checkpoint metadata (no load, no GPU) and assert it matches EXPECTED_*.

    Skips cleanly when the checkpoint is absent (the default on Windows/CI) so the suite
    stays green; becomes a real pre-spend gate when ``SIGNET_LTX_CHECKPOINT`` points at a
    mounted/downloaded LTX-2.3 ``.safetensors``.
    """
    safetensors = pytest.importorskip("safetensors")
    from safetensors import safe_open  # noqa: PLC0415 — optional dep, imported after skip-guard

    checkpoint = _resolve_checkpoint()
    if checkpoint is None:
        pytest.skip(
            f"no checkpoint at ${CHECKPOINT_ENV} — free pre-check is a no-op until a "
            f"local/mounted LTX-2.3 .safetensors is provided"
        )

    _ = safetensors  # keep the importorskip handle referenced for clarity

    # Header/metadata only: safe_open does NOT materialize tensors; we read keys + shapes.
    with safe_open(str(checkpoint), framework="pt") as f:
        keys = list(f.keys())

        # --- num_blocks: count distinct transformer block indices in the key names ---
        block_indices: set[int] = set()
        for key in keys:
            match = _BLOCK_IDX_RE.search(key)
            if match:
                block_indices.add(int(match.group(1)))
        assert block_indices, (
            "no transformer block keys found in checkpoint — key naming may differ from "
            f"the expected (transformer_)blocks.<i>. pattern; sample keys: {keys[:5]}"
        )
        num_blocks = max(block_indices) + 1
        assert num_blocks == EXPECTED_NUM_BLOCKS, (
            f"transformer block count {num_blocks} != EXPECTED_NUM_BLOCKS "
            f"{EXPECTED_NUM_BLOCKS}"
        )

        # --- hidden_dim: read the hidden axis from a per-block attention q-proj weight ---
        # to_q.weight is [hidden_dim, hidden_dim] for self-attention; either axis gives it.
        q_keys = [k for k in keys if "blocks.0." in k and "to_q" in k and k.endswith("weight")]
        if q_keys:
            q_shape = f.get_slice(q_keys[0]).get_shape()
            assert q_shape, f"empty shape for {q_keys[0]}"
            hidden_dim = q_shape[0]
            assert hidden_dim == EXPECTED_HIDDEN_DIM, (
                f"hidden dim {hidden_dim} (from {q_keys[0]} shape {q_shape}) != "
                f"EXPECTED_HIDDEN_DIM {EXPECTED_HIDDEN_DIM}"
            )

        # --- in_channels: read the input-channel axis from the patch/proj-in weight ---
        # A linear/conv proj-in maps in_channels -> hidden_dim; in_channels is the LAST axis.
        proj_in_keys = [
            k
            for k in keys
            if ("patchify_proj" in k or "proj_in" in k or "patch_embed" in k)
            and k.endswith("weight")
        ]
        if proj_in_keys:
            proj_shape = f.get_slice(proj_in_keys[0]).get_shape()
            assert proj_shape, f"empty shape for {proj_in_keys[0]}"
            in_channels = proj_shape[-1]
            assert in_channels == EXPECTED_T2V_IN_CHANNELS, (
                f"in_channels {in_channels} (from {proj_in_keys[0]} shape {proj_shape}) != "
                f"EXPECTED_T2V_IN_CHANNELS {EXPECTED_T2V_IN_CHANNELS}"
            )


# ------------------------------------------------------------------------------------------
# Landmine #3 — live num_blocks introspection (CPU, no checkpoint).
#
# The Phase-2 summarize_components left num_blocks=None on the real arch because the
# transformer exposes no num_blocks/num_layers scalar attr — the block stack is a
# CONTAINER. These tests prove the fixed introspection counts the container WITHOUT the
# 46GB checkpoint, so EXPECTED_NUM_BLOCKS=48 becomes assertable for the validation gate's
# dimension check (03-05). Mirrors the safetensors block-key regex above.
# ------------------------------------------------------------------------------------------


def _make_components(transformer: object) -> object:
    """Wrap a fake transformer in a minimal components stand-in (only .transformer is read)."""

    class _Components:
        pass

    components = _Components()
    components.transformer = transformer  # type: ignore[attr-defined]
    return components


def test_summarize_components_counts_transformer_blocks() -> None:
    """summarize_components introspects a 48-long transformer_blocks container → num_blocks=48."""
    import torch.nn as nn  # noqa: PLC0415 — torch only needed for this CPU introspection test

    class _FakeTransformer(nn.Module):
        def __init__(self, n_blocks: int) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList(nn.Linear(4, 4) for _ in range(n_blocks))

    summary = summarize_components(_make_components(_FakeTransformer(EXPECTED_NUM_BLOCKS)))
    assert summary["num_blocks"] == EXPECTED_NUM_BLOCKS, (
        f"introspected num_blocks {summary['num_blocks']!r} != EXPECTED_NUM_BLOCKS "
        f"{EXPECTED_NUM_BLOCKS} (landmine #3 — transformer_blocks container not counted)"
    )


def test_summarize_components_counts_named_modules_fallback() -> None:
    """When no container attr is exposed, count distinct block indices over named_modules keys."""

    class _NamedModulesOnlyTransformer:
        """A transformer that exposes ONLY named_modules() keys (no .transformer_blocks attr)."""

        config = None

        def named_modules(self):
            yield ("", self)
            for i in range(EXPECTED_NUM_BLOCKS):
                yield (f"transformer_blocks.{i}", object())
                yield (f"transformer_blocks.{i}.attn1.to_q", object())

    summary = summarize_components(_make_components(_NamedModulesOnlyTransformer()))
    assert summary["num_blocks"] == EXPECTED_NUM_BLOCKS, (
        f"named_modules fallback num_blocks {summary['num_blocks']!r} != "
        f"EXPECTED_NUM_BLOCKS {EXPECTED_NUM_BLOCKS}"
    )
