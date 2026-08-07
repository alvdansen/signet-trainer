"""TRAIN-05 — the locked LoRA target list + the PEFT ``base_layer`` skip-filter.

CPU unit tests for ``lora/peft.py`` (ported from enochiatron ``train.py`` PEFT inject
1814-1847 + ``P1_*_LORA_TARGETS`` 137-162, and the canonical skip-filter from
flimmer-trainer-dev ``block_swap.py::_get_linear_weights`` 50-76).

Proven here: the module constants lock the ``attn1+attn2+ff.net`` target set (the ff.net FIX),
and ``build_lora_config`` builds the locked r==alpha==64 config.

(The stale one-per-original-Linear skip-filter was removed in 09-01, D-9-DEDUP — the offloader's
PEFT-aware ``get_linear_weights`` / ``get_swappable_weights`` in ``lora.peft`` are now the single
enumeration source, comprehensively covered by ``tests/test_offload_peft_aware.py``.)

Anti-Pattern 6: importing ``lora.peft`` must NOT pull modal / ltx_core / ltx_trainer.
``peft`` itself is the boundary (a declared modal-runtime dep); skip cleanly when absent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

peft = pytest.importorskip("peft")

from signet_trainer.lora.peft import (  # noqa: E402
    P1_FF_LORA_TARGETS,
    P1_LORA_TARGETS,
    build_lora_config,
)

# The module under guard (path used for the text scan).
_PEFT_SRC = (
    Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "lora" / "peft.py"
)
_FORBIDDEN_MODULES = ("modal", "ltx_core", "ltx_trainer")


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


# --------------------------------------------------------------------------------------------------
# Anti-Pattern-6 import confinement
# --------------------------------------------------------------------------------------------------


def test_peft_module_imports_no_modal_or_ltx() -> None:
    for mod in _FORBIDDEN_MODULES:
        sys.modules.pop(mod, None)

    import signet_trainer.lora.peft  # noqa: F401

    for mod in _FORBIDDEN_MODULES:
        assert mod not in sys.modules, (
            f"import-confinement violation: lora.peft transitively imported {mod!r}"
        )


def test_peft_source_has_no_modal_or_ltx_import() -> None:
    code = _strip_comments_and_docstrings(_PEFT_SRC.read_text(encoding="utf-8"))
    offenders = [
        r"^\s*import\s+modal\b",
        r"^\s*from\s+modal\b",
        r"^\s*import\s+ltx_core\b",
        r"^\s*from\s+ltx_core\b",
        r"^\s*import\s+ltx_trainer\b",
        r"^\s*from\s+ltx_trainer\b",
    ]
    hits = [pat for pat in offenders if re.search(pat, code, re.MULTILINE)]
    assert not hits, f"import-confinement violation in lora/peft.py real code: {hits}"


# --------------------------------------------------------------------------------------------------
# the locked target list (the ff.net FIX)
# --------------------------------------------------------------------------------------------------


def test_attn_targets_are_the_eight_locked_entries() -> None:
    assert P1_LORA_TARGETS == [
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
    ]


def test_ff_targets_extend_attn_with_the_ffnet_fix() -> None:
    # The video-only set adds the two ff.net projections on top of the 8 attn entries.
    assert "ff.net.0.proj" in P1_FF_LORA_TARGETS
    assert "ff.net.2" in P1_FF_LORA_TARGETS
    assert all(t in P1_FF_LORA_TARGETS for t in P1_LORA_TARGETS)
    assert P1_FF_LORA_TARGETS == P1_LORA_TARGETS + ["ff.net.0.proj", "ff.net.2"]


# --------------------------------------------------------------------------------------------------
# build_lora_config — r==alpha==64, dropout 0.0, bias none, locked targets
# --------------------------------------------------------------------------------------------------


def test_build_lora_config_defaults() -> None:
    cfg = build_lora_config()
    assert cfg.r == 64
    assert cfg.lora_alpha == 64  # rank == alpha -> PEFT scale = 1.0 (clean convert)
    assert cfg.lora_dropout == 0.0
    assert cfg.bias == "none"
    assert set(cfg.target_modules) == set(P1_FF_LORA_TARGETS)


def test_build_lora_config_is_overridable() -> None:
    cfg = build_lora_config(rank=8, alpha=8, dropout=0.1, targets=["a", "b"])
    assert cfg.r == 8
    assert cfg.lora_alpha == 8
    assert cfg.lora_dropout == 0.1
    assert set(cfg.target_modules) == {"a", "b"}
