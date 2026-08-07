"""CPU proof that ``summarize_components`` surfaces ``_count_blocks`` output (D-9-P2-CARRY).

Pure CPU, no checkpoint, no Modal/GPU. Builds tiny stand-in transformers that expose a block
CONTAINER (``transformer_blocks`` / ``blocks``) of known length N — the exact real-LTX-2.3 shape
(no ``num_blocks``/``num_layers`` scalar attr) — and asserts
``summarize_components(...)["num_blocks"] == N``.

This proves the WIRING (``summarize_components`` -> ``_count_blocks``) unconditionally on an arch
that has NO scalar block attr. The live ``48``-on-real-arch assert (``EXPECTED_NUM_BLOCKS``) rides
the gated campaign load smoke in Wave 4, not here.
"""

from __future__ import annotations

import pytest

from signet_trainer.models.loader import summarize_components


def _components_with(transformer: object) -> object:
    """Wrap a stand-in transformer in a minimal components object (only ``.transformer`` is read)."""

    class _Components:
        pass

    components = _Components()
    components.transformer = transformer  # type: ignore[attr-defined]
    return components


@pytest.mark.parametrize("n_blocks", [1, 12, 48])
def test_summarize_components_surfaces_transformer_blocks_count(n_blocks: int) -> None:
    """A ``transformer_blocks`` container of length N (no scalar attr) -> ``num_blocks == N``."""
    import torch.nn as nn  # noqa: PLC0415 — torch only needed for this CPU introspection stand-in

    class _StandInTransformer(nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            # A block CONTAINER of known length, with NO num_blocks/num_layers scalar attr —
            # exactly why the Phase-2 _probe chain returned None on the real arch.
            self.transformer_blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(n))

    summary = summarize_components(_components_with(_StandInTransformer(n_blocks)))
    assert summary["num_blocks"] == n_blocks, (
        f"summarize_components surfaced num_blocks={summary['num_blocks']!r} != {n_blocks} "
        "(the _count_blocks wiring is not surfacing the transformer_blocks container count)"
    )


def test_summarize_components_surfaces_alt_blocks_container() -> None:
    """The alternate (Wan-style) ``blocks`` container name is also counted."""
    import torch.nn as nn  # noqa: PLC0415

    class _AltStandIn(nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(n))

    summary = summarize_components(_components_with(_AltStandIn(7)))
    assert summary["num_blocks"] == 7


def test_summarize_components_num_blocks_none_without_any_blocks() -> None:
    """No container, no block-indexed named_modules, no scalar attr -> honest ``None`` (not 0)."""

    class _Bare:
        config = None

        def named_modules(self):
            yield ("", self)

    summary = summarize_components(_components_with(_Bare()))
    assert summary["num_blocks"] is None, (
        f"expected num_blocks=None on a block-less stand-in, got {summary['num_blocks']!r}"
    )
