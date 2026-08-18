"""inference.lora_load — the PEFT-native inference adapter load path (INFR-02, LoRA column).

Ported from enochiatron ``scripts/infer/generate.py`` (955-1014, VERIFIED): load the Phase-3
``adapter_model.safetensors``, strip the ``base_model.model.`` prefix, extract rank + target
modules from the ``lora_A`` / ``lora_B`` keys, build a ``LoraConfig``, ``get_peft_model`` the
transformer, then ``set_peft_model_state_dict`` onto ``.get_base_model()``.

WARNING (RESEARCH Pitfall 1): this is the ValidationSampler path and is PEFT-native. Do NOT use
the ICLoraPipeline convert seam (the LTX-prefix rename in ``lora/peft.py``) here — that
conversion targets the official ICLoraPipeline loader (Phases 5-7) and will NOT match the keys
``get_peft_model`` expects.

ALPHA (#22 finding 4): ``CheckpointManager.save`` writes ``adapter_config.json`` (PEFT
``save_pretrained``) next to ``adapter_model.safetensors`` carrying the TRAINED ``r`` / ``lora_alpha``
— ``lora.alpha`` is an independently settable knob (``config/schema.py`` — rank and alpha need not
match), so re-deriving alpha as ``rank * lora_scale`` silently renders every adapter trained with
``alpha != rank`` at the wrong strength. :func:`build_inference_lora_config` now reads the trained
``lora_alpha`` from ``adapter_config.json`` when present and applies ``lora_scale`` as a multiplier
ON TOP of it. A bare ``adapter_model.safetensors`` with no sidecar (legacy / externally-produced
adapters that render fine today) is NEVER refused — it falls back to the old ``rank * lora_scale``
rebuild, loudly, via a printed notice. Refusal is reserved for the metadata being PRESENT and
DISAGREEING with the weights themselves (``adapter_config.json``'s ``r`` != the extracted
``lora_A`` rank) — that is a corrupted/mismatched pair, not an absence.

Import-confinement (Anti-Pattern 6): module top imports ``torch`` + ``peft`` + ``safetensors``
+ the reused ``lora.peft`` helpers ONLY — NO ``modal`` / ``ltx_core`` / ``ltx_trainer`` — so the
rank/target/config builders stay CPU-unit-testable on Windows/CI. The transformer-binding step
(``load_lora_onto_transformer``) is Modal-side and defers ``get_peft_model`` /
``set_peft_model_state_dict`` / ``load_file`` to function-local imports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig

# Reuse the existing PEFT prefix/constant helpers — do NOT redefine them (lora/peft.py owns them).
from signet_trainer.lora.peft import ADAPTER_FILENAME, _PEFT_PREFIX, strip_peft_prefix

__all__ = [
    "ADAPTER_CONFIG_FILENAME",
    "ADAPTER_FILENAME",
    "build_inference_lora_config",
    "load_lora_onto_transformer",
    "read_trained_alpha",
    "strip_peft_prefix",
]

#: The PEFT ``save_pretrained`` sidecar filename carrying the trained ``r`` / ``lora_alpha`` (#22
#: finding 4) — written next to :data:`ADAPTER_FILENAME` by ``CheckpointManager.save``.
ADAPTER_CONFIG_FILENAME = "adapter_config.json"

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


def read_trained_alpha(adapter_dir: Path, extracted_rank: int) -> int | None:
    """Read the trained ``lora_alpha`` from ``adapter_dir/adapter_config.json``, if present.

    Returns ``None`` when the sidecar is absent — a bare ``adapter_model.safetensors`` with no
    ``adapter_config.json`` (legacy checkpoints, or adapters produced outside this repo) renders
    fine today and MUST NOT be refused (#22 finding 4); the caller falls back to the
    ``rank * lora_scale`` rebuild.

    Raises ``ValueError`` only when the sidecar IS present and its declared ``r`` disagrees with
    ``extracted_rank`` (the rank read from the ``lora_A`` tensors themselves) — that is a
    corrupted or mismatched checkpoint pair, a real contradiction, not an absence. A sidecar with
    no ``lora_alpha`` key (present but incomplete) is treated the same as absent.
    """
    config_path = adapter_dir / ADAPTER_CONFIG_FILENAME
    if not config_path.is_file():
        return None
    metadata = json.loads(config_path.read_text(encoding="utf-8"))
    meta_rank = metadata.get("r")
    if meta_rank is not None and int(meta_rank) != int(extracted_rank):
        raise ValueError(
            f"{config_path} declares r={meta_rank} but the adapter_model.safetensors lora_A "
            f"tensors carry rank {extracted_rank} — the checkpoint's metadata and weights "
            "disagree (a corrupted or mismatched pair). Refusing rather than rendering at a "
            "possibly-wrong LoRA strength; regenerate a matching adapter_config.json/weights pair."
        )
    meta_alpha = metadata.get("lora_alpha")
    return int(meta_alpha) if meta_alpha is not None else None


def build_inference_lora_config(
    state_dict: dict[str, torch.Tensor],
    lora_scale: float = 1.0,
    adapter_dir: Path | None = None,
) -> LoraConfig:
    """Build the inference ``LoraConfig`` from a stripped adapter state_dict.

    ``r`` = extracted rank. ``lora_alpha`` prefers the TRAINED value (#22 finding 4): when
    ``adapter_dir`` carries an ``adapter_config.json`` sidecar, its ``lora_alpha`` is read and
    ``lora_scale`` is applied as a multiplier ON TOP of it (``lora_alpha = trained_alpha *
    lora_scale``) — ``lora.alpha`` is an independently settable knob (``config/schema.py``), so
    rank and alpha need not match at training time. When no sidecar is found (``adapter_dir`` is
    ``None``, or the directory has no ``adapter_config.json`` — legacy/external adapters that
    render fine today), this FALLS BACK to the old ``int(rank * lora_scale)`` rebuild (assumes
    rank == alpha) with a loud printed notice — it is never refused for a bare
    ``adapter_model.safetensors``. ``lora_dropout`` = 0.0, ``target_modules`` = the extracted set.
    """
    rank = _get_lora_rank(state_dict)
    targets = _extract_lora_target_modules(state_dict)

    trained_alpha = read_trained_alpha(adapter_dir, rank) if adapter_dir is not None else None
    if trained_alpha is not None:
        lora_alpha = int(round(trained_alpha * lora_scale))
        print(
            f"[lora_load] read trained lora_alpha={trained_alpha} from "
            f"{adapter_dir / ADAPTER_CONFIG_FILENAME} (r={rank}); applying lora_scale="
            f"{lora_scale} -> lora_alpha={lora_alpha}."
        )
    else:
        lora_alpha = int(rank * lora_scale)
        print(
            f"[lora_load] WARNING: no {ADAPTER_CONFIG_FILENAME} found"
            f"{f' under {adapter_dir}' if adapter_dir is not None else ''} — rebuilding "
            f"lora_alpha as rank * lora_scale = {rank} * {lora_scale} = {lora_alpha}. This "
            "assumes rank == alpha at training time; if the trained config set "
            "lora.alpha != lora.rank, this render is at the WRONG LoRA strength (#22 finding 4)."
        )

    return LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
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

    # adapter_dir = the resolved file's parent — adapter_config.json (#22 finding 4) is always a
    # sidecar of adapter_model.safetensors, never addressed independently.
    lora_config = build_inference_lora_config(
        state_dict, lora_scale=lora_scale, adapter_dir=path.parent
    )
    wrapped = get_peft_model(transformer, lora_config)
    set_peft_model_state_dict(wrapped.get_base_model(), state_dict)
    return wrapped
