"""INFR-02 — the PEFT-native inference adapter load path (`inference/lora_load.py`).

CPU unit tests: a synthetic Phase-3 adapter state_dict is stripped, rank-extracted, and
turned into a bindable ``peft.LoraConfig`` with NO GPU. Proves the *inference* key
convention (strip ``base_model.model.`` → ``get_base_model()``), which is DISTINCT from the
ICLoraPipeline ``diffusion_model.`` convert (Phases 5-7, RESEARCH Pitfall 1).

Anti-Pattern 6: importing ``inference.lora_load`` must NOT pull modal / ltx_core / ltx_trainer.
``peft`` is the boundary (a declared modal-runtime dep); skip cleanly when absent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

peft = pytest.importorskip("peft")
from peft import LoraConfig  # noqa: E402  — after skip-guard

from signet_trainer.inference.lora_load import (  # noqa: E402
    _extract_lora_target_modules,
    _get_lora_rank,
    build_inference_lora_config,
)
from signet_trainer.lora.peft import _PEFT_PREFIX, strip_peft_prefix  # noqa: E402

# The module under guard (path used for the text scan).
_LORA_LOAD_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "signet_trainer"
    / "inference"
    / "lora_load.py"
)
_FORBIDDEN_MODULES = ("modal", "ltx_core", "ltx_trainer")


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _synthetic_prefixed_sd(
    rank: int = 64, in_features: int = 128, out_features: int = 256
) -> dict[str, torch.Tensor]:
    """A PEFT-native adapter state_dict (carries the ``base_model.model.`` prefix).

    ``lora_A`` is ``[rank, in]``; ``lora_B`` is ``[out, rank]``. Module names are flat
    (post-strip form) so the extracted target set is exactly ``{attn1.to_q, ...}``.
    """
    modules = ["attn1.to_q", "attn1.to_k", "ff.net.2"]
    sd: dict[str, torch.Tensor] = {}
    for m in modules:
        sd[f"{_PEFT_PREFIX}{m}.lora_A.weight"] = torch.zeros(rank, in_features)
        sd[f"{_PEFT_PREFIX}{m}.lora_B.weight"] = torch.zeros(out_features, rank)
    return sd


# --------------------------------------------------------------------------------------------------
# Anti-Pattern-6 import confinement
# --------------------------------------------------------------------------------------------------


def test_lora_load_imports_no_modal_or_ltx() -> None:
    for mod in _FORBIDDEN_MODULES:
        sys.modules.pop(mod, None)

    import signet_trainer.inference.lora_load  # noqa: F401

    for mod in _FORBIDDEN_MODULES:
        assert mod not in sys.modules, (
            f"import-confinement violation: inference.lora_load transitively imported {mod!r}"
        )


def test_lora_load_source_has_no_modal_or_ltx_import() -> None:
    code = _strip_comments_and_docstrings(_LORA_LOAD_SRC.read_text(encoding="utf-8"))
    offenders = [
        r"^\s*import\s+modal\b",
        r"^\s*from\s+modal\b",
        r"^\s*import\s+ltx_core\b",
        r"^\s*from\s+ltx_core\b",
        r"^\s*import\s+ltx_trainer\b",
        r"^\s*from\s+ltx_trainer\b",
    ]
    hits = [pat for pat in offenders if re.search(pat, code, re.MULTILINE)]
    assert not hits, f"import-confinement violation in inference/lora_load.py real code: {hits}"


def test_lora_load_does_not_use_iclora_convert_path() -> None:
    # RESEARCH Pitfall 1: the ValidationSampler path is PEFT-native. The `diffusion_model.`
    # ICLoraPipeline prefix convert is Phases 5-7 only and MUST NOT appear here. (The dead
    # PEFT->LTX prefix-convert helper was removed in 09-01, D-9-DEADCODE.)
    src = _LORA_LOAD_SRC.read_text(encoding="utf-8")
    assert "diffusion_model." not in src, "inference load must not use the `diffusion_model.` prefix"


# --------------------------------------------------------------------------------------------------
# rank + target-module extraction
# --------------------------------------------------------------------------------------------------


def test_get_lora_rank_reads_lora_a_shape0() -> None:
    sd = strip_peft_prefix(_synthetic_prefixed_sd(rank=64, in_features=128))
    assert _get_lora_rank(sd) == 64


def test_get_lora_rank_generalizes() -> None:
    sd = strip_peft_prefix(_synthetic_prefixed_sd(rank=16, in_features=64))
    assert _get_lora_rank(sd) == 16


def test_extract_target_modules_contains_attn_and_ff() -> None:
    sd = strip_peft_prefix(_synthetic_prefixed_sd())
    targets = _extract_lora_target_modules(sd)
    assert "attn1.to_q" in targets
    assert "attn1.to_k" in targets
    assert "ff.net.2" in targets
    # No `.lora_A` / `.lora_B` suffix leaks into a target name.
    assert all(".lora_A" not in t and ".lora_B" not in t for t in targets)


# --------------------------------------------------------------------------------------------------
# the reused strip helper
# --------------------------------------------------------------------------------------------------


def test_strip_peft_prefix_removes_prefix() -> None:
    prefixed = _synthetic_prefixed_sd()
    assert all(k.startswith(_PEFT_PREFIX) for k in prefixed)
    stripped = strip_peft_prefix(prefixed)
    assert all(not k.startswith(_PEFT_PREFIX) for k in stripped)
    assert set(stripped) == {k.replace(_PEFT_PREFIX, "", 1) for k in prefixed}


# --------------------------------------------------------------------------------------------------
# build_inference_lora_config — r==alpha==64, dropout 0.0 (scale 1.0)
# --------------------------------------------------------------------------------------------------


def test_build_inference_lora_config_rank64_scale1() -> None:
    sd = strip_peft_prefix(_synthetic_prefixed_sd(rank=64))
    cfg = build_inference_lora_config(sd, lora_scale=1.0)
    assert isinstance(cfg, LoraConfig)
    assert cfg.r == 64
    assert cfg.lora_alpha == 64  # rank == alpha -> PEFT scale 1.0, no rescale
    assert cfg.lora_dropout == 0.0
    assert "attn1.to_q" in set(cfg.target_modules)


def test_build_inference_lora_config_scale_multiplies_alpha() -> None:
    sd = strip_peft_prefix(_synthetic_prefixed_sd(rank=64))
    cfg = build_inference_lora_config(sd, lora_scale=2.0)
    assert cfg.r == 64
    assert cfg.lora_alpha == 128  # alpha == int(rank * scale)
