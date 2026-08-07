"""INFR-03 — the PEFT save -> load roundtrip.

CPU unit tests for the save/load surface of ``lora/peft.py`` (ported from enochiatron
``infer/generate.py::load_lora_onto_transformer`` 955-1003, key-strip 984-996). The cheapest
sufficient INFR-03 proof is a roundtrip: ``save_pretrained`` -> ``load_file`` ->
``set_peft_model_state_dict`` into a freshly-wrapped identical model, asserting key-count +
tensor equality with no ``.alpha`` / serialization mismatch (rank==alpha==64 -> clean).

(The dead PEFT->LTX ``diffusion_model.``-prefix convert seam was removed in 09-01 (D-9-DEADCODE):
it was never wired to any real inference consumer — the PEFT-native ValidationSampler path in
``inference/lora_load.py`` deliberately does NOT convert prefixes.)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import (  # noqa: E402  — after skip-guard
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from safetensors.torch import load_file  # noqa: E402

from signet_trainer.lora.peft import (  # noqa: E402
    load_adapter_into,
    roundtrip_check,
    save_adapter,
)


class _TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x):  # pragma: no cover — not exercised
        return self.b(self.a(x))


def _wrapped() -> tuple[nn.Module, LoraConfig]:
    cfg = LoraConfig(
        r=4, lora_alpha=4, target_modules=["a", "b"], lora_dropout=0.0, bias="none"
    )
    return get_peft_model(_TwoLinear(), cfg), cfg


def _randomize_lora_b(model: nn.Module) -> None:
    # lora_B inits to zero; randomize it so the roundtrip moves real (non-zero) weights.
    for name, p in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.copy_(torch.randn_like(p))


# --------------------------------------------------------------------------------------------------
# the roundtrip into a FRESH identical model (the strongest INFR-03 proof)
# --------------------------------------------------------------------------------------------------


def test_save_load_roundtrip_into_fresh_model(tmp_path) -> None:
    src, cfg = _wrapped()
    _randomize_lora_b(src)
    saved = get_peft_model_state_dict(src)

    save_adapter(src, tmp_path)

    dst = get_peft_model(_TwoLinear(), cfg)
    result = load_adapter_into(dst, tmp_path)

    # An adapter-only file legitimately reports the frozen base_layer weights as "missing"
    # (they are not part of the adapter); what must be empty is UNEXPECTED keys.
    assert not getattr(result, "unexpected_keys", [])

    loaded = get_peft_model_state_dict(dst)
    assert set(loaded) == set(saved)  # key-count + identity, no missing/extra
    for k in saved:
        assert torch.allclose(saved[k], loaded[k])


def test_saved_keys_have_peft_prefix_and_no_alpha(tmp_path) -> None:
    model, _ = _wrapped()
    _randomize_lora_b(model)
    save_adapter(model, tmp_path)

    sd = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert all(k.startswith("base_model.model.") for k in sd)
    # rank==alpha==64 -> no serialized scale scalar -> no .alpha mismatch on load.
    assert not any(k.endswith(".alpha") for k in sd)

    # The generate.py:984 strip yields clean transformer-side keys, same count.
    stripped = {k.replace("base_model.model.", "", 1): v for k, v in sd.items()}
    assert all(not k.startswith("base_model.model.") for k in stripped)
    assert len(stripped) == len(sd)


def test_roundtrip_check_helper_reports_key_count(tmp_path) -> None:
    model, _ = _wrapped()
    _randomize_lora_b(model)
    report = roundtrip_check(model, tmp_path)
    # 2 target Linears x (lora_A, lora_B) = 4 adapter tensors.
    assert report["num_keys"] == 4


# --------------------------------------------------------------------------------------------------
# P7-IN-04 — a NON-default adapter round-trips via its <name>/ subdirectory
# --------------------------------------------------------------------------------------------------


def test_non_default_adapter_roundtrips_via_subdir(tmp_path) -> None:
    # save_pretrained(selected_adapters=["frozen"]) writes into tmp_path/frozen/, NOT the flat root.
    # load_adapter_into must resolve that subdir; a flat-root read would raise FileNotFound and the
    # non-default save/load path would be silently broken (P7-IN-04).
    src, cfg = _wrapped()
    src.add_adapter("frozen", cfg)
    _randomize_lora_b(src)  # randomizes lora_B across BOTH adapters
    saved = get_peft_model_state_dict(src, adapter_name="frozen")

    save_adapter(src, tmp_path, adapter_name="frozen")

    # layout: the non-default adapter lands under a <name>/ subdir, not the flat root.
    assert (tmp_path / "frozen" / "adapter_model.safetensors").exists()
    assert not (tmp_path / "adapter_model.safetensors").exists()

    dst = get_peft_model(_TwoLinear(), cfg)
    dst.add_adapter("frozen", cfg)
    result = load_adapter_into(dst, tmp_path, adapter_name="frozen")
    assert not getattr(result, "unexpected_keys", [])

    loaded = get_peft_model_state_dict(dst, adapter_name="frozen")
    assert set(loaded) == set(saved)
    for k in saved:
        assert torch.allclose(saved[k], loaded[k])
