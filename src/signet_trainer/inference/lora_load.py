"""inference.lora_load — the PEFT-native inference adapter load path (INFR-02, LoRA column).

Ported from enochiatron ``scripts/infer/generate.py`` (955-1014, VERIFIED): load the Phase-3
``adapter_model.safetensors``, strip the ``base_model.model.`` prefix, extract rank + target
modules from the ``lora_A`` / ``lora_B`` keys, build a ``LoraConfig``, ``get_peft_model`` the
transformer, then ``set_peft_model_state_dict`` onto ``.get_base_model()``.

WARNING (RESEARCH Pitfall 1): this is the ValidationSampler path and is PEFT-native. Do NOT use
the ICLoraPipeline convert seam (the LTX-prefix rename in ``lora/peft.py``) here — that
conversion targets the official ICLoraPipeline loader (Phases 5-7) and will NOT match the keys
``get_peft_model`` expects. Because the Phase-3 adapter is ``rank == lora_alpha == 64`` the PEFT
scale is 1.0, so ``lora_scale=1.0`` applies no rescale.

Import-confinement (Anti-Pattern 6): module top imports ``torch`` + ``peft`` + ``safetensors``
+ the reused ``lora.peft`` helpers ONLY — NO ``modal`` / ``ltx_core`` / ``ltx_trainer`` — so the
rank/target/config builders stay CPU-unit-testable on Windows/CI. The transformer-binding step
(``load_lora_onto_transformer``) is Modal-side and defers ``get_peft_model`` /
``set_peft_model_state_dict`` / ``load_file`` to function-local imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig

# Reuse the existing PEFT prefix/constant helpers — do NOT redefine them (lora/peft.py owns them).
from signet_trainer.lora.peft import ADAPTER_FILENAME, _PEFT_PREFIX, strip_peft_prefix

__all__ = [
    "ADAPTER_FILENAME",
    "build_inference_lora_config",
    "load_lora_onto_transformer",
    "strip_peft_prefix",
]

# The adapter-key markers that identify a LoRA down/up projection.
_LORA_A_MARKER = ".lora_A"
_LORA_B_MARKER = ".lora_B"


def _get_lora_rank(state_dict: dict[str, torch.Tensor]) -> int:
    """Read the LoRA rank from the first ``lora_A`` tensor's ``shape[0]``.

    ``lora_A`` is ``[rank, in_features]`` (the down-projection), so ``shape[0]`` is the rank.
    Expects an already-stripped state_dict (no ``base_model.model.`` prefix required, but the
    marker match is prefix-agnostic).
    """
    for key, tensor in state_dict.items():
        if _LORA_A_MARKER in key:
            return int(tensor.shape[0])
    raise ValueError(
        "no `lora_A` tensor found in adapter state_dict — not a PEFT LoRA checkpoint"
    )


def _extract_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> set[str]:
    """Derive the target-module set from the ``lora_A`` / ``lora_B`` key paths.

    For a key ``<module_path>.lora_A.weight`` the module path (everything before ``.lora_A`` /
    ``.lora_B``) is the target. Returns e.g. ``{"attn1.to_q", "attn1.to_k", "ff.net.2"}``.
    """
    targets: set[str] = set()
    for key in state_dict:
        for marker in (_LORA_A_MARKER, _LORA_B_MARKER):
            idx = key.find(marker)
            if idx != -1:
                targets.add(key[:idx])
                break
    return targets


def build_inference_lora_config(
    state_dict: dict[str, torch.Tensor], lora_scale: float = 1.0
) -> LoraConfig:
    """Build the inference ``LoraConfig`` from a stripped adapter state_dict.

    ``r`` = extracted rank, ``lora_alpha`` = ``int(rank * lora_scale)`` (rank==alpha==64 →
    scale 1.0 → no rescale), ``lora_dropout`` = 0.0, ``target_modules`` = the extracted set.
    """
    rank = _get_lora_rank(state_dict)
    targets = _extract_lora_target_modules(state_dict)
    return LoraConfig(
        r=rank,
        lora_alpha=int(rank * lora_scale),
        target_modules=sorted(targets),
        lora_dropout=0.0,
    )


def load_lora_onto_transformer(
    transformer: Any, adapter_path: str | Path, lora_scale: float = 1.0
) -> Any:
    """Load a Phase-3 adapter onto ``transformer`` the PEFT-native way; return the wrapped model.

    Sequence (enochiatron generate.py:955-1014): load ``.safetensors`` → strip
    ``base_model.model.`` → build config → ``get_peft_model(transformer, cfg)`` →
    ``set_peft_model_state_dict(wrapped.get_base_model(), sd)``. The ``.get_base_model()`` target
    is load-bearing: the stripped keys address the base transformer modules, not the PEFT wrapper.

    ``adapter_path`` may be the ``adapter_model.safetensors`` file itself or its parent directory.
    """
    # Heavy / GPU-side imports stay function-local (Anti-Pattern 6): merely importing this module
    # for the CPU config builders must not require peft's runtime graph wiring or a real model.
    from peft import get_peft_model, set_peft_model_state_dict  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    path = Path(adapter_path)
    if path.is_dir():
        path = path / ADAPTER_FILENAME

    raw = load_file(str(path))
    state_dict = strip_peft_prefix(raw)

    lora_config = build_inference_lora_config(state_dict, lora_scale=lora_scale)
    wrapped = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(wrapped.get_base_model(), state_dict)
    return wrapped
