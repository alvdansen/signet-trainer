"""03-07 regression: the gated train body REUSES the validation gate's injected adapter.

The gate's check #5/#6 PEFT-inject + roundtrip-prove a LoRA adapter on ``components.transformer``.
On the Open-Q1 default path the ``train()`` body must REUSE that exact adapter, never call
``inject_lora`` on ``components.transformer`` a second time — re-wrapping an already-LoRA'd module
would error / corrupt the adapter on the real A100. Proven by (a) the ``run_validation_gate``
3-tuple contract that hands the adapter back, and (b) a CPU source-order scan of ``modal/fns.py``
— no Modal run, zero spend.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

_FNS = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip(src: str) -> str:
    """Drop triple-quoted blocks + line comments so prose mentions don't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"#.*", "", src)


def test_run_validation_gate_returns_injected_adapter() -> None:
    """The gate returns (all_passed, results, peft_model) so the caller can reuse the adapter."""
    from signet_trainer.train.validate_gate import run_validation_gate

    src = _strip(inspect.getsource(run_validation_gate))
    assert re.search(r"return\s+all_passed\s*,\s*results\s*,\s*peft_model", src), (
        "run_validation_gate must return the injected adapter as a third element (03-07 reuse)"
    )


def _train_body(code: str) -> str:
    """Slice the LTX ``train()`` body out of the stripped source.

    Scoping matters as of Phase 10: ``fns.py`` now carries a SECOND training stage (``h3_train``,
    plan 10-11), which legitimately calls ``inject_lora`` once of its own — the H3 arch gate is a
    cheap read-only assertion that deliberately does NOT inject, so there is no adapter to reuse and
    nothing to double-wrap. A file-wide count would read that as a regression in ``train()``. The
    assertion below is about the LTX gate/train handoff, so it is measured on the LTX body.
    """
    start = code.index("def train(config_yaml")
    tail = code[start:]
    nxt = tail.find("@app.function", 1)
    return tail if nxt == -1 else tail[:nxt]


def test_train_body_reuses_gate_adapter_no_double_inject() -> None:
    """The default path reuses the gate adapter; inject_lora() is confined to the use_builder branch."""
    code = _train_body(_strip(_FNS.read_text(encoding="utf-8")))

    # The gate unpack captures the adapter as the third element.
    unpack = re.search(r"(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*=\s*run_validation_gate", code)
    assert unpack, "train() must unpack the 3-tuple from run_validation_gate (adapter reuse)"
    adapter_var = unpack.group(3)

    # Open-Q1 default path reuses the adapter: `model = <adapter_var>`.
    assert re.search(rf"model\s*=\s*{adapter_var}\b", code), (
        "the Open-Q1 default path must reuse the gate adapter (model = gate adapter), not re-inject"
    )

    # inject_lora() is CALLED exactly once, and only inside the use_builder branch — never on the
    # default path (which would double-wrap components.transformer).
    builder_idx = code.find("if use_builder")
    assert builder_idx != -1, "train() must branch on use_builder"
    inject_calls = [m.start() for m in re.finditer(r"inject_lora\s*\(", code)]
    assert len(inject_calls) == 1, (
        f"expected exactly one inject_lora() call (use_builder branch only), found {len(inject_calls)}"
    )
    assert inject_calls[0] > builder_idx, (
        "inject_lora() must live inside the use_builder branch, not the Open-Q1 default path"
    )


def test_train_body_asserts_gate_adapter_dropout_matches_config_before_reuse() -> None:
    """AUDIT #34 direction 2 belt-and-braces — the gate's check #5 built ``gate_adapter``'s
    dropout from ``config.lora.dropout`` too (``validate_gate.check_training_step``); ``train()``
    must assert the two still agree BEFORE reusing the adapter as the training model, so a future
    drift between the gate's build and this reuse site aborts pre-spend instead of silently
    training at the wrong dropout — the exact #34 defect, guarded against recurring.
    """
    code = _train_body(_strip(_FNS.read_text(encoding="utf-8")))

    assert "gate_adapter" in code and "lora_dropout" in code and "config.lora.dropout" in code
    assert re.search(
        r"lora_dropout(?:.|\n)*?!=(?:.|\n)*?config\.lora\.dropout"
        r"|config\.lora\.dropout(?:.|\n)*?!=(?:.|\n)*?lora_dropout",
        code,
    ), "train() must compare gate_adapter's lora_dropout against config.lora.dropout"

    dropout_idx = code.index("lora_dropout")
    reuse_idx = code.index("model = gate_adapter")
    assert dropout_idx < reuse_idx, (
        "the dropout consistency assertion must run BEFORE `model = gate_adapter` reuses the "
        "adapter, not after"
    )
    assert "RuntimeError" in code[dropout_idx : reuse_idx + 50]
