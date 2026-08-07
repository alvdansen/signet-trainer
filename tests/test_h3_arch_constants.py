"""H3-01 CPU arch-contract test — the ``EXPECTED_H3_*`` constants ARE the committed config.

Pins ``src/signet_trainer/models/h3_loader.py``'s single-source constants to
``tests/fixtures/h3_transformer_ref_config.json`` (the diffusers-format ``transformer_ref/config.json``
read off ``MiniMaxAI/MiniMax-H3``). Every number is imported from the loader — this file duplicates
NO architecture literal, so a drifted constant fails here rather than on a metered A100.

CPU-only and import-confined: does not import ``diffusers`` / ``modal`` / CUDA, mirroring
``tests/test_ground_truth_read.py``. It additionally PROVES the loader's ``diffusers`` import is
function-local, two ways: a source scan (comments/docstrings stripped, same discipline as
``tests/test_preprocess_wiring.py``) and a subprocess import with ``diffusers`` hard-blocked.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from signet_trainer.models.h3_loader import (
    EXPECTED_H3_ADALN_PROJ_SHAPE,
    EXPECTED_H3_MODALITY_NUM,
    EXPECTED_H3_NUM_LAYERS,
    EXPECTED_H3_NUM_REFINER_LAYERS,
    EXPECTED_H3_PATCH_SIZE,
    EXPECTED_H3_TEXT_ENCODER_LAYER,
    assert_h3_arch,
    expected_h3_arch,
    summarize_h3_transformer,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "h3_transformer_ref_config.json"
_H3_LOADER = _REPO_ROOT / "src" / "signet_trainer" / "models" / "h3_loader.py"


def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _fixture_summary() -> dict[str, Any]:
    """A ``summarize_h3_transformer``-shaped dict built from the committed fixture (the PASS case)."""
    fx = _fixture()
    summary: dict[str, Any] = {key: fx[key] for key in expected_h3_arch()}
    summary["patch_size"] = tuple(fx["patch_size"])
    summary["live_transformer_blocks"] = fx["num_layers"]
    summary["live_refiner_blocks"] = fx["num_refiner_layers"]
    summary["adaln_proj_shape"] = EXPECTED_H3_ADALN_PROJ_SHAPE
    return summary


# ---------------------------------------------------------------------------------------------
# 1. The constants ARE the committed config
# ---------------------------------------------------------------------------------------------


def test_expected_h3_arch_matches_committed_fixture() -> None:
    """Every key ``expected_h3_arch()`` publishes equals the fixture's value for that field.

    Driven off ``expected_h3_arch()``'s OWN keys, so a constant added to the accessor tomorrow is
    auto-covered here (and fails loudly if the fixture has no such field).
    """
    fx = _fixture()
    arch = expected_h3_arch()
    assert arch, "expected_h3_arch() must not be empty"
    for field, value in arch.items():
        assert field in fx, f"{field!r} is in expected_h3_arch() but not in the committed fixture"
        assert value == fx[field], f"{field}: constant {value!r} != fixture {fx[field]!r}"


def test_expected_h3_arch_is_the_ten_measured_fields() -> None:
    """The accessor is keyed by DIFFUSERS config field names so a caller can diff ``model.config``."""
    assert set(expected_h3_arch()) == {
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "attention_head_dim",
        "ffn_dim",
        "in_channels",
        "audio_in_channels",
        "text_dim",
        "time_embed_dim",
        "num_refiner_layers",
    }
    assert len(expected_h3_arch()) == 10


def test_patch_size_matches_fixture() -> None:
    """``EXPECTED_H3_PATCH_SIZE`` is a tuple of the fixture's JSON list (patch_dim math depends on it)."""
    assert EXPECTED_H3_PATCH_SIZE == tuple(_fixture()["patch_size"])


def test_text_encoder_layer_is_the_qwen3vl_index_not_the_dit_depth() -> None:
    """H3 reads Qwen3-VL ``hidden_states[50]`` of its 64 layers — NOT the final (post-norm) layer.

    The 50 here is the TEXT-ENCODER hidden-state index (P10-1-MEASURED §6 / P10-0d §2). Its equality
    with ``num_layers = 50`` (the DiT depth) is a numerical COINCIDENCE — the two must never be
    conflated, which is why this asserts the literal rather than comparing the two constants.
    """
    assert EXPECTED_H3_TEXT_ENCODER_LAYER == 50


def test_modality_tag_count() -> None:
    """adaln modality tags are ``0=video 1=text 2=audio`` — three, addressed ``t_idx * 3 + tag``."""
    assert EXPECTED_H3_MODALITY_NUM == 3


# ---------------------------------------------------------------------------------------------
# 2. The gate
# ---------------------------------------------------------------------------------------------


def test_assert_h3_arch_passes_on_a_fixture_derived_summary() -> None:
    assert assert_h3_arch(_fixture_summary()) is None


def test_assert_h3_arch_raises_naming_the_mismatched_field() -> None:
    summary = _fixture_summary()
    summary["num_layers"] = 48  # the LTX block count — the exact confusion the gate exists to catch
    with pytest.raises(RuntimeError) as exc:
        assert_h3_arch(summary)
    message = str(exc.value)
    assert "ARCH MISMATCH" in message
    assert "num_layers" in message
    assert "48" in message


def test_assert_h3_arch_names_every_offending_field_not_just_the_first() -> None:
    summary = _fixture_summary()
    summary["num_layers"] = 48
    summary["hidden_size"] = 4096
    summary["live_refiner_blocks"] = 0
    with pytest.raises(RuntimeError) as exc:
        assert_h3_arch(summary)
    message = str(exc.value)
    for field in ("num_layers", "hidden_size", "live_refiner_blocks"):
        assert field in message, f"{field} missing from the mismatch report: {message}"


def test_assert_h3_arch_catches_the_comfy_pruned_adaln_bottleneck() -> None:
    """``[96768, 8]`` is the ComfyUI pruned baked-bottleneck form — it must never pass this gate."""
    summary = _fixture_summary()
    summary["adaln_proj_shape"] = (96768, 8)
    with pytest.raises(RuntimeError) as exc:
        assert_h3_arch(summary)
    assert "adaln_proj_shape" in str(exc.value)


def test_assert_h3_arch_skips_unknown_fields_but_says_so() -> None:
    """Assert only what we have ground truth for (the ``load_ltxv_smoke`` tolerance)."""
    summary = _fixture_summary()
    summary["adaln_proj_shape"] = None  # probe could not read it
    assert assert_h3_arch(summary) is None

    summary["num_layers"] = 48
    with pytest.raises(RuntimeError) as exc:
        assert_h3_arch(summary)
    assert "adaln_proj_shape" in str(exc.value), "skipped fields must be named in the message"


# ---------------------------------------------------------------------------------------------
# 3. The summarizer tolerates attribute drift
# ---------------------------------------------------------------------------------------------


class _FakeConfig:
    """A partial diffusers-style config: two fields present, the rest absent."""

    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


class _FakeModel:
    """A model exposing only ``.config`` — no ``transformer_blocks`` / ``token_refiner``."""

    def __init__(self, config: _FakeConfig) -> None:
        self.config = config


def test_summarize_returns_none_for_missing_fields_rather_than_raising() -> None:
    fx = _fixture()
    model = _FakeModel(_FakeConfig(num_layers=fx["num_layers"], hidden_size=fx["hidden_size"]))
    summary = summarize_h3_transformer(model)
    assert summary["num_layers"] == fx["num_layers"]
    assert summary["hidden_size"] == fx["hidden_size"]
    assert summary["ffn_dim"] is None
    assert summary["live_transformer_blocks"] is None
    assert summary["live_refiner_blocks"] is None
    assert summary["adaln_proj_shape"] is None


def test_summarize_reads_live_containers_when_present() -> None:
    fx = _fixture()

    class _Weight:
        shape = EXPECTED_H3_ADALN_PROJ_SHAPE

    class _Linear:
        weight = _Weight()

    class _AdaLN:
        linear = _Linear()

    class _Block:
        adaln_proj = _AdaLN()

    class _Refiner:
        refiner_blocks = [object()] * fx["num_refiner_layers"]

    model = _FakeModel(_FakeConfig(**{key: fx[key] for key in expected_h3_arch()}))
    model.transformer_blocks = [_Block() for _ in range(fx["num_layers"])]  # type: ignore[attr-defined]
    model.token_refiner = _Refiner()  # type: ignore[attr-defined]

    summary = summarize_h3_transformer(model)
    assert summary["live_transformer_blocks"] == EXPECTED_H3_NUM_LAYERS
    assert summary["live_refiner_blocks"] == EXPECTED_H3_NUM_REFINER_LAYERS
    assert summary["adaln_proj_shape"] == EXPECTED_H3_ADALN_PROJ_SHAPE
    # …and a live summary of the real arch passes the gate.
    assert assert_h3_arch(summary) is None


# ---------------------------------------------------------------------------------------------
# 4. Import confinement — the deliberate Anti-Pattern-6 EXCEPTION is function-local ONLY
# ---------------------------------------------------------------------------------------------


def test_diffusers_import_is_function_local() -> None:
    """No module-scope ``import diffusers`` — every diffusers import line must be indented."""
    code = _strip_comments_and_docstrings(_H3_LOADER.read_text(encoding="utf-8"))
    import_lines = [
        line
        for line in code.splitlines()
        if re.match(r"\s*(?:from\s+diffusers\b|import\s+diffusers\b)", line)
    ]
    assert import_lines, "expected the heavy diffusers import to exist (function-local)"
    for line in import_lines:
        assert line != line.lstrip(), f"module-scope diffusers import is forbidden: {line!r}"
    assert any("MiniMaxH3Transformer3DModel" in line for line in import_lines), (
        "load_h3_transformer must import diffusers.MiniMaxH3Transformer3DModel"
    )


def test_module_imports_with_diffusers_blocked() -> None:
    """Behavioral proof: reading the constants works on a machine with no ``diffusers`` installed."""
    script = textwrap.dedent(
        """
        import sys

        class _BlockDiffusers:
            def find_spec(self, name, path=None, target=None):
                if name == "diffusers" or name.startswith("diffusers."):
                    raise ImportError("diffusers is blocked for this test")
                return None

        sys.meta_path.insert(0, _BlockDiffusers())
        from signet_trainer.models.h3_loader import (
            EXPECTED_H3_NUM_LAYERS,
            EXPECTED_H3_TEXT_ENCODER_LAYER,
        )
        assert "diffusers" not in sys.modules, "importing h3_loader must not drag in diffusers"
        print(EXPECTED_H3_NUM_LAYERS, EXPECTED_H3_TEXT_ENCODER_LAYER)
        """
    )
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src")}
    proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"import failed with diffusers blocked:\n{proc.stderr}"
    assert proc.stdout.split() == ["50", "50"], proc.stdout
