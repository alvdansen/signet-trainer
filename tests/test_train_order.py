"""TRAIN-06 — gradient-checkpointing is enabled BEFORE ``get_peft_model`` (order matters).

The enochiatron inject (``train.py`` 1814-1847) enables grad-checkpointing on the base
model FIRST, then wraps it with ``get_peft_model``. If LoRA wraps first, the GC enable can
land on the PEFT wrapper instead of the base transformer and silently no-op. These tests
pin the call order behaviourally: a spy records when ``set_gradient_checkpointing`` (or the
``gradient_checkpointing_enable`` fallback) fires relative to ``get_peft_model``.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

pytest.importorskip("peft")

from signet_trainer.lora import peft as lora_peft  # noqa: E402  — after skip-guard


def test_set_gradient_checkpointing_runs_before_get_peft_model(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def set_gradient_checkpointing(self, value: bool) -> None:
            events.append(("gc", value))

        def train(self, mode: bool = True):  # noqa: D401 — record-free passthrough
            return self

    def _fake_get_peft_model(model, cfg):
        events.append(("peft", cfg))
        return model

    monkeypatch.setattr(lora_peft, "get_peft_model", _fake_get_peft_model)

    model = _TinyModel()
    lora_peft.inject_lora(model, lora_peft.build_lora_config())

    kinds = [e[0] for e in events]
    assert kinds == ["gc", "peft"], (
        "GC must be enabled before get_peft_model wraps the model"
    )
    assert events[0][1] is True  # GC enabled with True


def test_gradient_checkpointing_enable_fallback_runs_before_peft(monkeypatch) -> None:
    # When the model lacks set_gradient_checkpointing, inject falls back to the HF-style
    # gradient_checkpointing_enable() — and it too must fire before get_peft_model.
    events: list[str] = []

    class _HFStyleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def gradient_checkpointing_enable(self) -> None:
            events.append("gce")

        def train(self, mode: bool = True):
            return self

    def _fake_get_peft_model(model, cfg):
        events.append("peft")
        return model

    monkeypatch.setattr(lora_peft, "get_peft_model", _fake_get_peft_model)

    model = _HFStyleModel()
    lora_peft.inject_lora(model, lora_peft.build_lora_config())

    assert events == ["gce", "peft"]


def test_inject_returns_the_peft_wrapped_model_in_train_mode(monkeypatch) -> None:
    seen = {}

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def set_gradient_checkpointing(self, value: bool) -> None:
            seen["gc"] = value

        def train(self, mode: bool = True):
            seen["train"] = mode
            return self

    sentinel = _TinyModel()

    monkeypatch.setattr(lora_peft, "get_peft_model", lambda m, c: sentinel)

    returned = lora_peft.inject_lora(_TinyModel(), lora_peft.build_lora_config())

    assert returned is sentinel
    assert seen.get("train") is True  # model.train() called on the wrapped model
