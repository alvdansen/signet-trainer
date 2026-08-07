"""D-7-FREEZE proof-a — CPU structural proof of frozen-LoRA stacking.

Mirrors the ``test_lora_roundtrip`` fixture (a tiny two-``Linear`` ``torch.nn`` module wrapped with
PEFT, no ``ltx_core`` / no CUDA / no ``modal``). Proves the three structural invariants of
``stack_frozen_adapter`` (REF-03, RESEARCH §Pattern 4):

  (a) every param whose key contains ``"frozen"`` has ``requires_grad == False`` (re-frozen);
  (b) the optimizer-eligible set (``requires_grad == True``) is EXACTLY the new ``"default"`` adapter
      params — no frozen params, no base weights;
  (c) a save via the ``adapter_name="default"`` path writes NO ``"frozen"`` keys (frozen weights are
      excluded from the checkpoint).

The real gated stacking run (proof-b) is 07-12; this tier-1 test locks the freeze discipline,
optimizer filter set, and save exclusion on CPU/Windows/CI.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402  — after skip-guard
from safetensors.torch import load_file  # noqa: E402

from signet_trainer.lora.peft import (  # noqa: E402
    save_adapter,
    stack_frozen_adapter,
)


class _TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):  # pragma: no cover — not exercised
        return self.b(self.a(x))


def _fresh_peft() -> tuple[nn.Module, LoraConfig]:
    cfg = LoraConfig(
        r=4, lora_alpha=4, target_modules=["a", "b"], lora_dropout=0.0, bias="none"
    )
    return get_peft_model(_TwoLinear(), cfg), cfg


def _randomize_lora_b(model: nn.Module) -> None:
    # lora_B inits to zero; randomize so the frozen adapter carries real (non-zero) weights.
    for name, p in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.copy_(torch.randn_like(p))


# --------------------------------------------------------------------------------------------------
# D-7-FREEZE proof-a — one CPU test, three structural asserts
# --------------------------------------------------------------------------------------------------


def test_frozen_stacking_freeze_optimizer_and_save(tmp_path) -> None:
    # 1. Build a trainable "default" adapter, randomize it, save it as the FROZEN source adapter.
    src, cfg = _fresh_peft()
    _randomize_lora_b(src)
    frozen_dir = tmp_path / "frozen_src"
    save_adapter(src, frozen_dir)  # writes adapter_config.json + adapter_model.safetensors

    # 2. Fresh model with a fresh trainable "default" adapter, then stack the frozen one on top.
    model = get_peft_model(_TwoLinear(), cfg)
    stack_frozen_adapter(model, frozen_dir, adapter_name="frozen")

    named = list(model.named_parameters())

    # (a) every frozen-adapter param is re-frozen.
    frozen_params = [(n, p) for n, p in named if "frozen" in n]
    assert frozen_params, "the frozen adapter did not load (no 'frozen' params present)"
    assert all(p.requires_grad is False for _, p in frozen_params)

    # (b) the trainable (requires_grad) set is EXACTLY the new "default" adapter params.
    trainable = {n for n, p in named if p.requires_grad}
    assert trainable, "no trainable params — the 'default' adapter must remain optimizer-eligible"
    assert all("default" in n for n in trainable), trainable
    assert all("frozen" not in n for n in trainable), trainable
    assert all(("lora_A" in n or "lora_B" in n) for n in trainable), trainable
    # no base weight (non-LoRA param) is trainable.
    for n, p in named:
        if "lora_A" not in n and "lora_B" not in n:
            assert p.requires_grad is False, f"base param unexpectedly trainable: {n}"

    # (c) saving with adapter_name="default" EXCLUDES the frozen adapter's weights.
    out = tmp_path / "saved"
    save_adapter(model, out, adapter_name="default")
    sd = load_file(str(out / "adapter_model.safetensors"))
    assert not any("frozen" in k for k in sd), f"frozen weights leaked into the save: {sorted(sd)}"
    # 2 target Linears x (lora_A, lora_B) = 4 default-only tensors.
    assert len(sd) == 4, f"expected 4 default-only keys, got {len(sd)}: {sorted(sd)}"
