"""AUDIT #34 direction 2 — ``lora.dropout`` must reach the gate-built adapter.

``validate_gate.check_training_step`` (check #5) builds the ``LoraConfig`` that trains on the
shipped DEFAULT path: ``fns.py`` / ``local/runner.py`` both do ``model = gate_adapter`` (03-07
adapter reuse — see ``tests/test_gate_adapter_reuse.py``), so whatever dropout this call built is
what actually trains. Before the fix, the ``build_lora_config(...)`` call here threaded ``rank`` /
``alpha`` / ``targets`` but never ``dropout=``, so PEFT's default (``lora_dropout=0.0``) always
landed regardless of ``config.lora.dropout``.

This test does NOT exercise the real GPU check (``ltx_core`` / a live transformer / ``peft`` — that
stays 03-07-metered-preflight-only, same boundary ``tests/test_validate_gate.py`` documents).
Every heavy call (``build_lora_config``, ``inject_lora``, ``training_step``) is monkeypatched with
a spy/stub, mirroring ``tests/test_training_step.py``'s
``test_train_loop_threads_multi_frame_conditioning_kwargs`` injection pattern — so this runs
CPU-only, with zero ``ltx_core`` / CUDA dependency, and still proves the real wiring: which kwargs
``check_training_step`` actually passes to ``build_lora_config``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from signet_trainer.train import validate_gate as vg


def _stubbed_gate(monkeypatch, *, captured: dict) -> None:
    def _fake_build_lora_config(**kwargs):
        captured.update(kwargs)
        return object()  # opaque; inject_lora is stubbed below so this is never inspected by PEFT

    param = nn.Parameter(torch.zeros(1))

    class _FakeModel:
        def parameters(self):
            return iter([param])

    def _fake_inject_lora(transformer, lora_config, **kwargs):  # noqa: ANN001
        return _FakeModel()

    def _fake_training_step(model, batch, schedule, rng, **kwargs):  # noqa: ANN001
        return (param * 2).sum()  # a real graph so loss.backward() has something to do

    monkeypatch.setattr(vg, "build_lora_config", _fake_build_lora_config)
    monkeypatch.setattr(vg, "inject_lora", _fake_inject_lora)
    monkeypatch.setattr(vg, "training_step", _fake_training_step)


def test_check_training_step_threads_config_lora_dropout(monkeypatch) -> None:
    captured: dict = {}
    _stubbed_gate(monkeypatch, captured=captured)

    components = SimpleNamespace(transformer=object())
    config = SimpleNamespace(
        lora=SimpleNamespace(rank=64, alpha=64, dropout=0.15, target_modules=None),
        training=SimpleNamespace(uniform_prob=0.30),
    )

    result, model = vg.check_training_step(components, config, device="cpu", dtype=torch.float32)

    assert result.status == "PASS", result.message
    assert model is not None
    assert captured.get("dropout") == 0.15, (
        "check_training_step must thread config.lora.dropout into build_lora_config — the "
        f"kwargs it actually passed were {captured!r}. Before the AUDIT #34 fix this key was "
        "simply absent, so PEFT's lora_dropout=0.0 default always landed on the shipped default "
        "train path (fns.py / runner.py both reuse this exact adapter)."
    )


def test_check_training_step_dropout_defaults_to_zero_when_config_carries_none(monkeypatch) -> None:
    """No ``lora.dropout`` on the config (or no ``lora`` section at all) must fall back to the
    documented ``0.0`` — the SAME default ``build_lora_config`` itself carries — not raise."""
    captured: dict = {}
    _stubbed_gate(monkeypatch, captured=captured)

    components = SimpleNamespace(transformer=object())
    config = SimpleNamespace(training=SimpleNamespace(uniform_prob=0.30))  # no `lora` at all

    result, _ = vg.check_training_step(components, config, device="cpu", dtype=torch.float32)

    assert result.status == "PASS", result.message
    assert captured.get("dropout") == 0.0


def test_check_training_step_still_threads_rank_alpha_and_targets(monkeypatch) -> None:
    """Regression guard: adding dropout must not disturb the three knobs already threaded."""
    captured: dict = {}
    _stubbed_gate(monkeypatch, captured=captured)

    components = SimpleNamespace(transformer=object())
    config = SimpleNamespace(
        lora=SimpleNamespace(rank=8, alpha=16, dropout=0.2, target_modules=["a.b", "c.d"]),
        training=SimpleNamespace(uniform_prob=0.30),
    )

    vg.check_training_step(components, config, device="cpu", dtype=torch.float32)

    assert captured.get("rank") == 8
    assert captured.get("alpha") == 16
    assert captured.get("targets") == ["a.b", "c.d"]
    assert captured.get("dropout") == 0.2
