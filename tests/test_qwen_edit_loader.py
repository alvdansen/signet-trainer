"""Family #3 (``qwen_edit``) LOADER layer — the arch gate, driven by the REAL checkpoint headers.

Every assertion here is fed module names that were read out of the live safetensors HEADERS (the
8-byte length prefix + its JSON), never invented and never taken from the build report. The headers
are committed as ``tests/fixtures/qwen_edit_header_facts.json`` so the gate is a real test on a CI
box with no weights, and the live files re-verify the fixture whenever the env vars below point at
them.

Three failure classes this file exists to catch, each of which otherwise reaches a metered GPU:

1. **A target set that resolves to the wrong module count.** 14 leaves x 60 blocks = 840. A short
   count trains fewer modules than the config priced and cannot warm-start from a 14-leaf primer;
   the four ``attn.norm_*`` RMSNorms per block are the nearest wrong answer (240 of them sit
   directly beside the targets and are NOT Linear).
2. **A text-only encoder mislabeled as Qwen2.5-VL.** It loads, it embeds at the right rank, and it
   dies deep in the forward with a matmul shape error naming no file. This house lost a day to it
   and the evidence is still on disk; the fixture carries that file's real 339 keys, zero of which
   are vision.
3. **An arch mismatch surviving the load.** diffusers builds the module from the CONFIG and fills
   whatever fits, so a single-file load against a wrong config yields a wrong-shaped model with a
   clean-looking config and no load error. Only the live weight SHAPES catch that.

CPU only. No torch model is built, no weights are loaded, no Modal, no downloads, no GPU.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from signet_trainer.config.schema import QwenEditConfig
from signet_trainer.config.validators import QWEN_EDIT_LORA_LEAVES
from signet_trainer.models.qwen_edit_loader import (
    EXPECTED_QWEN_EDIT_ATTENTION_HEAD_DIM,
    EXPECTED_QWEN_EDIT_IMG_IN_SHAPE,
    EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM,
    EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT,
    EXPECTED_QWEN_EDIT_NUM_ATTENTION_HEADS,
    EXPECTED_QWEN_EDIT_PROJ_OUT_SHAPE,
    EXPECTED_QWEN_EDIT_TRANSFORMER_TENSOR_COUNT,
    EXPECTED_QWEN_EDIT_TXT_IN_SHAPE,
    EXPECTED_QWEN_VL_TENSOR_COUNT,
    EXPECTED_QWEN_VL_VISION_TENSOR_COUNT,
    assert_qwen_edit_arch,
    assert_qwen_edit_not_peft_wrapped,
    assert_qwen_edit_targets,
    assert_qwen_edit_text_encoder_vision,
    expected_qwen_edit_arch,
    qwen_vl_vision_census,
    summarize_qwen_edit_transformer,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "qwen_edit_header_facts.json"
_LOADER = _REPO_ROOT / "src" / "signet_trainer" / "models" / "qwen_edit_loader.py"

#: Env vars pointing at the local checkpoints. Absent on CI — the fixture carries the same facts,
#: so only the "does the fixture still match the live file" tests skip.
CHECKPOINT_ENV = "SIGNET_QWEN_EDIT_CHECKPOINT"
TEXT_ENCODER_ENV = "SIGNET_QWEN_EDIT_TEXT_ENCODER"

#: Number of blocks the measured checkpoint carries — restated here on purpose. This file is the
#: VERIFIER of the loader module and must not import the constant it is checking.
MEASURED_BLOCKS = 60
MEASURED_LEAVES_PER_BLOCK = 14
MEASURED_HIDDEN = 3072
MEASURED_PACKED_ROW = 64


@pytest.fixture(scope="module")
def facts() -> dict[str, Any]:
    """The committed safetensors-header facts."""
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def transformer_modules(facts: dict[str, Any]) -> list[str]:
    """Every module name in the live transformer, derived from its real tensor keys."""
    return list(facts["transformer"]["module_names"])


def _header_keys(path: Path) -> list[str]:
    """Read a safetensors HEADER only — 8-byte LE length, then that many JSON bytes.

    No tensor body is ever touched: the read stops at the end of the header. This is what makes a
    40 GiB checkpoint a free pre-check.
    """
    with path.open("rb") as fh:
        (length,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(length))
    return sorted(k for k in header if k != "__metadata__")


def _module_names(keys: list[str]) -> list[str]:
    return sorted({k.rsplit(".", 1)[0] for k in keys if "." in k})


def _live(env_var: str) -> Path | None:
    raw = os.environ.get(env_var)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


# --------------------------------------------------------------------------------------------------
# The fixture is real data, not a convenience — assert its shape before anything trusts it.
# --------------------------------------------------------------------------------------------------


def test_the_fixture_carries_the_measured_checkpoint(facts: dict[str, Any]) -> None:
    """The committed header facts are the ones the family was specified from."""
    transformer = facts["transformer"]
    assert transformer["filename"] == "qwen_image_edit_2511_bf16.safetensors"
    assert transformer["bytes"] == 40_861_031_560
    assert transformer["tensor_count"] == EXPECTED_QWEN_EDIT_TRANSFORMER_TENSOR_COUNT == 1934

    encoder = facts["text_encoder_vl"]
    assert encoder["tensor_count"] == EXPECTED_QWEN_VL_TENSOR_COUNT == 1446
    assert encoder["vision_tensor_count"] == EXPECTED_QWEN_VL_VISION_TENSOR_COUNT == 714
    assert "visual" in encoder["top_level_groups"]

    text_only = facts["text_encoder_text_only"]
    assert text_only["vision_tensor_count"] == 0
    assert "visual" not in text_only["top_level_groups"]


def test_the_transformer_has_sixty_blocks_in_the_real_key_names(transformer_modules: list[str]) -> None:
    """Block count is COUNTED off the real names, not read from a config field."""
    indices = {
        int(m.group(1))
        for name in transformer_modules
        if (m := re.match(r"transformer_blocks\.(\d+)\.", name))
    }
    assert len(indices) == MEASURED_BLOCKS
    assert indices == set(range(MEASURED_BLOCKS))


# --------------------------------------------------------------------------------------------------
# THE HEADLINE GATE — 840 modules, per leaf, on real names.
# --------------------------------------------------------------------------------------------------


def test_the_target_regex_resolves_exactly_840_modules(transformer_modules: list[str]) -> None:
    """14 leaves x 60 blocks, every leaf at exactly 60, zero collateral — on the real key names."""
    survey = assert_qwen_edit_targets(transformer_modules)

    assert survey["total"] == MEASURED_BLOCKS * MEASURED_LEAVES_PER_BLOCK == 840
    assert survey["total"] == EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT
    assert survey["collateral"] == 0
    assert len(survey["per_leaf"]) == MEASURED_LEAVES_PER_BLOCK
    assert set(survey["per_leaf"].values()) == {MEASURED_BLOCKS}


def test_the_attn_rmsnorms_are_present_and_never_targeted(transformer_modules: list[str]) -> None:
    """240 ``attn.norm_{q,k,added_q,added_k}`` sit beside the targets; none may match.

    They are the nearest wrong answer on this checkpoint: same depth, same ``attn.`` parent, and a
    ``norm`` suffix a hand-written pattern could easily admit. They are RMSNorm ``[128]``, not
    Linear, and PEFT would refuse them at injection — after the container is paid for.
    """
    norms = [n for n in transformer_modules if re.search(r"attn\.norm_(q|k|added_q|added_k)$", n)]
    assert len(norms) == MEASURED_BLOCKS * 4 == 240

    survey = assert_qwen_edit_targets(transformer_modules)
    matched_norms = [n for n in norms if re.fullmatch(survey["pattern"], n)]
    assert matched_norms == []


def test_a_checkpoint_missing_the_txt_stream_leaves_is_refused(transformer_modules: list[str]) -> None:
    """The dual-stream half going missing is a REFUSAL, and the message names every dead leaf.

    This is the shape of the mistake the family's whole target contract exists to prevent: an
    ``img_*``-only intuition ported from LTX's single-stream block. Here it is simulated from the
    weights side — a checkpoint whose ``txt_*`` leaves are simply absent.
    """
    dropped = {"attn.add_q_proj", "attn.add_k_proj", "txt_mlp.net.2", "txt_mod.1"}
    thinned = [n for n in transformer_modules if n.split(".", 2)[-1] not in dropped]

    with pytest.raises(RuntimeError) as excinfo:
        assert_qwen_edit_targets(thinned)

    message = str(excinfo.value)
    assert "LoRA TARGET MISMATCH" in message
    assert "total=600" in message  # 840 - (4 leaves x 60)
    for leaf in dropped:
        assert leaf in message, f"the refusal must name the dead leaf {leaf!r}"
    assert "BEFORE any spend" in message


def test_a_partially_loaded_stack_is_refused(transformer_modules: list[str]) -> None:
    """A leaf present on SOME blocks but not all is refused too, and named as a wrong count.

    Distinct from the zero case: a per-leaf 41 is what a half-written shard or a truncated download
    looks like, and it is the one a grand-total check could hide behind a compensating over-count.
    """
    thinned = [
        n
        for n in transformer_modules
        if not (n.endswith(".attn.to_q") and int(n.split(".")[1]) >= 41)
    ]
    with pytest.raises(RuntimeError, match=r"attn\.to_q=41"):
        assert_qwen_edit_targets(thinned)


def test_every_declared_leaf_is_actually_present_on_the_checkpoint(
    transformer_modules: list[str],
) -> None:
    """Independent of the regex: each of the 14 leaf names exists 60x as a literal suffix."""
    for leaf in QWEN_EDIT_LORA_LEAVES:
        hits = [n for n in transformer_modules if n.endswith("." + leaf)]
        assert len(hits) == MEASURED_BLOCKS, f"{leaf} matched {len(hits)} modules, expected 60"


# --------------------------------------------------------------------------------------------------
# The arch constants and the summarize/assert pair.
# --------------------------------------------------------------------------------------------------


def test_expected_arch_is_the_eight_diffusers_config_fields() -> None:
    """Keyed to ``QwenImageTransformer2DModel.config``, so a caller can diff without translating."""
    assert set(expected_qwen_edit_arch()) == {
        "patch_size",
        "in_channels",
        "out_channels",
        "num_layers",
        "attention_head_dim",
        "num_attention_heads",
        "joint_attention_dim",
        "guidance_embeds",
    }


def _require(module: str) -> None:
    """Skip unless ``module`` is INSTALLABLE, resolved without touching ``sys.modules``.

    Deliberately not ``pytest.importorskip``. That helper decides availability by importing, and
    catches only ``ImportError`` — so any exception raised *during* a present package's import
    surfaces as a test FAILURE rather than a skip. The measured instance: an earlier test that
    evicts ``"PIL"`` from ``sys.modules`` without restoring it leaves ``sys.modules["PIL.Image"]``
    cached, so re-importing ``PIL`` builds a fresh parent module on which ``Image`` is never
    rebound; ``diffusers/utils/export_utils.py:27`` then evaluates the annotation
    ``list[PIL.Image.Image]`` and raises ``AttributeError``, which ``importorskip`` does not catch.
    That poisoning is fixed at source (``tests/test_mask_encode.py`` now restores what it pops,
    matching ``tests/test_grid_html.py:80-92``); this helper is the second line of defence, so a
    future evictor cannot turn an optional-dependency probe into a spurious failure again.

    ``PathFinder`` answers "is it on ``sys.path``" by locating the spec only — it neither imports
    the package nor mutates ``sys.modules``. Same idiom as ``tests/test_qwen_edit_prep.py:612-617``.
    """
    from importlib.machinery import PathFinder  # noqa: PLC0415

    root = module.split(".", 1)[0]
    if PathFinder().find_spec(root, sys.path) is None:
        pytest.skip(f"{module} is not installed in this interpreter")


def test_the_head_count_is_derived_not_typed() -> None:
    """3072 / 128 == 24 — and the derivation is what makes a typo impossible."""
    assert EXPECTED_QWEN_EDIT_ATTENTION_HEAD_DIM == 128
    assert EXPECTED_QWEN_EDIT_NUM_ATTENTION_HEADS == MEASURED_HIDDEN // 128 == 24


def test_expected_arch_matches_the_diffusers_class_defaults() -> None:
    """Independent oracle: diffusers' own ``__init__`` defaults ARE this architecture.

    Not circular — the constants here were measured off the checkpoint header, and this asserts the
    library that will BUILD the module agrees. A disagreement means the pinned diffusers would
    construct a differently-shaped model from the same config, which is precisely the single-file
    hazard the loader's ``config_source`` refusal is about.
    """
    _require("diffusers")
    import inspect  # noqa: PLC0415

    # Same reason as tests/test_qwen_edit_sampler.py's module-level guard: diffusers is a
    # [modal-runtime] extra kept out of the core install, so on a bare clone this is a SKIP,
    # not a hard failure that pollutes the documented red set.
    pytest.importorskip("diffusers")
    from diffusers import QwenImageTransformer2DModel  # noqa: PLC0415

    defaults = {
        name: param.default
        for name, param in inspect.signature(QwenImageTransformer2DModel.__init__).parameters.items()
    }
    for field, want in expected_qwen_edit_arch().items():
        assert defaults[field] == want, f"{field}: diffusers default {defaults[field]!r} != {want!r}"


def test_the_txt_in_width_discharges_the_unverified_config_field() -> None:
    """``QwenEditConfig.text_embed_dim`` was carried as ``[UNVERIFIED]``. The weights confirm 3584.

    The config field's own docstring instructs: *"The weight-loading pass MUST assert it against the
    real txt_in and correct it here if it differs."* The measured ``txt_in.weight`` is
    ``[3072, 3584]``, so the declared value stands — and the assertion now exists, which is the part
    that was missing.
    """
    assert EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM == 3584
    assert QwenEditConfig().text_embed_dim == EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM
    assert EXPECTED_QWEN_EDIT_TXT_IN_SHAPE == (MEASURED_HIDDEN, 3584)
    assert EXPECTED_QWEN_EDIT_IMG_IN_SHAPE == (MEASURED_HIDDEN, MEASURED_PACKED_ROW)
    assert EXPECTED_QWEN_EDIT_PROJ_OUT_SHAPE == (MEASURED_PACKED_ROW, MEASURED_HIDDEN)


class _FakeConfig:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class _FakeLinear:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.weight = _FakeTensor(shape)


class _FakeTensor:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape = shape


class _FakeTransformer:
    """A duck-typed stand-in shaped exactly like the live module's probe surface."""

    def __init__(self, **overrides: Any) -> None:
        fields = dict(expected_qwen_edit_arch())
        fields.update({k: v for k, v in overrides.items() if k in fields})
        self.config = _FakeConfig(**fields)
        self.transformer_blocks = [object()] * int(overrides.get("blocks", MEASURED_BLOCKS))
        self.img_in = _FakeLinear(overrides.get("img_in", EXPECTED_QWEN_EDIT_IMG_IN_SHAPE))
        self.txt_in = _FakeLinear(overrides.get("txt_in", EXPECTED_QWEN_EDIT_TXT_IN_SHAPE))
        self.proj_out = _FakeLinear(overrides.get("proj_out", EXPECTED_QWEN_EDIT_PROJ_OUT_SHAPE))


def test_a_correct_model_passes_the_arch_gate() -> None:
    assert_qwen_edit_arch(summarize_qwen_edit_transformer(_FakeTransformer()))


def test_the_arch_gate_names_every_offending_field_not_just_the_first() -> None:
    """enochiatron's gate caught 6 mismatches in one run because it did not stop at the first."""
    summary = summarize_qwen_edit_transformer(
        _FakeTransformer(
            num_layers=48,
            joint_attention_dim=4096,
            blocks=48,
            guidance_embeds=True,
            txt_in=(MEASURED_HIDDEN, 4096),
        )
    )
    with pytest.raises(RuntimeError) as excinfo:
        assert_qwen_edit_arch(summary)

    message = str(excinfo.value)
    for field in (
        "num_layers",
        "joint_attention_dim",
        "guidance_embeds",
        "live_transformer_blocks",
        "txt_in_shape",
    ):
        assert field in message, f"{field} missing from the refusal"
    assert message.count(";") == 4, f"expected all five fields reported: {message}"


def test_the_arch_gate_catches_a_wrong_config_that_the_config_itself_would_hide() -> None:
    """The single-file hazard: config reads perfect, the loaded WEIGHT shape does not.

    diffusers builds the module from the config and then fills whatever fits, so this is the only
    signal that separates "loaded the right weights" from "built the right shape".
    """
    summary = summarize_qwen_edit_transformer(_FakeTransformer(txt_in=(MEASURED_HIDDEN, 4096)))
    with pytest.raises(RuntimeError, match=r"txt_in_shape"):
        assert_qwen_edit_arch(summary)


def test_the_arch_gate_names_the_fields_it_could_not_read() -> None:
    """A partially-blind gate must never be mistaken for a clean one."""
    summary = summarize_qwen_edit_transformer(_FakeTransformer(num_layers=48))
    summary["joint_attention_dim"] = None
    with pytest.raises(RuntimeError, match=r"probe returned None: joint_attention_dim"):
        assert_qwen_edit_arch(summary)


def test_the_arch_gate_cross_checks_the_config_text_embed_dim() -> None:
    """A config declaring a width the live ``txt_in`` does not project from is refused."""
    summary = summarize_qwen_edit_transformer(_FakeTransformer())
    assert_qwen_edit_arch(summary, config_text_embed_dim=3584)
    with pytest.raises(RuntimeError, match=r"text_embed_dim: config declares 3072"):
        assert_qwen_edit_arch(summary, config_text_embed_dim=3072)


def test_summarize_returns_none_rather_than_raising_on_a_bare_object() -> None:
    """A probe must never be the thing that fails the gate."""
    summary = summarize_qwen_edit_transformer(object())
    assert set(summary) >= set(expected_qwen_edit_arch())
    assert all(value is None for value in summary.values())


# --------------------------------------------------------------------------------------------------
# The vision-tower gate — the day this house lost.
# --------------------------------------------------------------------------------------------------


def test_the_vl_encoder_passes_the_vision_gate(facts: dict[str, Any]) -> None:
    """Real ``visual.*`` keys off the Qwen2.5-VL header."""
    sample = facts["text_encoder_vl"]["vision_key_sample"]
    assert sample, "the fixture must carry real vision keys"
    census = assert_qwen_edit_text_encoder_vision(sample)
    assert census["vision"] == len(sample)


def test_the_nested_visual_spelling_also_counts() -> None:
    """A loaded encoder nests it as ``model.visual.*``; the single file stores it at top level."""
    census = qwen_vl_vision_census(
        ["model.visual.blocks.0.attn.proj.weight", "model.layers.0.mlp.up_proj.weight"]
    )
    assert census["vision"] == 1
    assert census["total"] == 2


def test_the_mislabeled_text_only_encoder_is_refused(facts: dict[str, Any]) -> None:
    """The actual file that cost a day, by its actual 339 keys, refused by name.

    ``_MISLABELED_was-qwen2.5-7b-instruct-not-VL.safetensors`` still sits on the box beside the
    correct encoder. The refusal must name the real downstream symptom, because that symptom —
    ``mat1 and mat2 shapes cannot be multiplied (5376x1280 and 3840x1280)`` — is what a future
    operator will actually be looking at when they search for this.
    """
    keys = facts["text_encoder_text_only"]["key_names"]
    assert len(keys) == 339

    with pytest.raises(RuntimeError) as excinfo:
        assert_qwen_edit_text_encoder_vision(keys, what="Qwen2.5-VL text encoder")

    message = str(excinfo.value)
    assert "NO VISION TENSORS" in message
    assert "339 name(s) inspected" in message
    assert "text-only Qwen2.5 LLM, not Qwen2.5-VL" in message
    assert "mat1 and mat2 shapes cannot be multiplied (5376x1280 and 3840x1280)" in message
    assert "714" in message and "1446" in message
    assert "BEFORE any spend" in message


# --------------------------------------------------------------------------------------------------
# Quantization ORDER.
# --------------------------------------------------------------------------------------------------


def test_quantizing_an_uninjected_model_is_allowed(transformer_modules: list[str]) -> None:
    assert assert_qwen_edit_not_peft_wrapped(transformer_modules, what="transformer") is None


def test_the_already_converted_guard_is_ai_toolkits_q_modules() -> None:
    """The guard set, restated from the source rather than imported from the module under test.

    ai-toolkit ``toolkit/util/quantize.py:25-33``. ``QLinear`` being in it is the load-bearing
    member: pass 2 of the recipe re-walks the blocks pass 1 already converted, and without this
    skip quanto re-quantizes a ``QLinear`` and dies in ``qbytes_ops.copy_``.
    """
    from signet_trainer.models.qwen_edit_loader import (  # noqa: PLC0415
        _QUANTO_ALREADY_CONVERTED,
    )

    assert set(_QUANTO_ALREADY_CONVERTED) == {
        "QLinear",
        "QConv2d",
        "QEmbedding",
        "QBatchNorm2d",
        "QLayerNorm",
        "QConvTranspose2d",
        "QEmbeddingBag",
    }


def test_the_two_pass_quantization_converts_blocks_then_extras_exactly_once() -> None:
    """Run the real thing on CPU: pass 1 the blocks, pass 2 only what pass 1 did not reach.

    Skips where ``optimum.quanto`` is not installed (the CPU/CI env). Where it IS installed this is
    the regression test for the measured crash recorded in ``quantize_qwen_edit``'s docstring: an
    earlier draft called ``optimum.quanto.quantize`` for both passes and died on the second with
    ``AttributeError: 'Parameter' object has no attribute 'qtype'``.
    """
    _require("optimum.quanto")
    import torch.nn as nn  # noqa: PLC0415

    # quantize_qwen_edit reaches optimum.quanto, which imports diffusers — so this test needs
    # the [modal-runtime] extra even though nothing in its own body names diffusers.
    pytest.importorskip("diffusers")
    from signet_trainer.models.qwen_edit_loader import quantize_qwen_edit  # noqa: PLC0415

    hidden = 32

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.ModuleDict({"to_q": nn.Linear(hidden, hidden)})
            self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.img_in = nn.Linear(4, hidden)
            self.txt_in = nn.Linear(8, hidden)
            self.transformer_blocks = nn.ModuleList(_Block() for _ in range(3))
            self.proj_out = nn.Linear(hidden, 4)

    model = _Toy()
    names_before = [n for n, _ in model.named_modules()]

    quantize_qwen_edit(model, what="toy")

    linears = {
        name: m.__class__.__name__
        for name, m in model.named_modules()
        if m.__class__.__name__.endswith(("Linear", "QLinear"))
    }
    assert set(linears.values()) == {"QLinear"}, linears
    assert len(linears) == 3 * 2 + 3  # 3 blocks x 2 leaves, plus img_in / txt_in / proj_out
    assert [n for n, _ in model.named_modules()] == names_before, "module names must survive"


def test_quantizing_a_peft_wrapped_model_is_refused() -> None:
    """quantize -> inject. The reverse would convert lora_A/lora_B, the published artifact."""
    injected = [
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_q.base_layer",
        "transformer_blocks.0.attn.to_q.lora_A.default",
        "transformer_blocks.0.attn.to_q.lora_B.default",
    ]
    with pytest.raises(RuntimeError) as excinfo:
        assert_qwen_edit_not_peft_wrapped(injected, what="transformer")

    message = str(excinfo.value)
    assert "ALREADY PEFT-wrapped" in message
    assert "BaseSDTrainProcess.py:1619" in message  # load_model, where quantization happens
    assert "quantize_qwen_edit BEFORE lora.peft.inject_lora" in message


# --------------------------------------------------------------------------------------------------
# Import confinement — the constants and the four assertions must work with no backend installed.
# --------------------------------------------------------------------------------------------------


def test_every_backend_import_is_function_local() -> None:
    """No module-scope ``diffusers`` / ``transformers`` / ``optimum`` import: all must be indented."""
    source = _LOADER.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\s*)(?:from|import)\s+(diffusers|transformers|optimum)\b", re.MULTILINE)
    matches = pattern.findall(source)
    assert matches, "expected the heavy backend imports to exist (function-local)"
    for indent, module in matches:
        assert indent, f"module-scope {module} import is forbidden in qwen_edit_loader.py"
    assert {module for _, module in matches} == {"diffusers", "transformers", "optimum"}


def test_module_imports_with_every_backend_blocked() -> None:
    """Behavioral proof: the gate runs on a CI box with no diffusers/transformers/quanto."""
    script = textwrap.dedent(
        """
        import sys

        BLOCKED = ("diffusers", "transformers", "optimum")

        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name in BLOCKED or any(name.startswith(b + ".") for b in BLOCKED):
                    raise ImportError(f"{name} is blocked for this test")
                return None

        sys.meta_path.insert(0, _Block())
        from signet_trainer.models.qwen_edit_loader import (
            EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT,
            EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM,
            assert_qwen_edit_text_encoder_vision,
        )
        for blocked in BLOCKED:
            assert blocked not in sys.modules, f"importing qwen_edit_loader dragged in {blocked}"
        census = assert_qwen_edit_text_encoder_vision(["visual.blocks.0.attn.proj.weight"])
        print(EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT, EXPECTED_QWEN_EDIT_JOINT_ATTENTION_DIM,
              census["vision"])
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
    assert proc.returncode == 0, f"import failed with backends blocked:\n{proc.stderr}"
    assert proc.stdout.split() == ["840", "3584", "1"], proc.stdout


# --------------------------------------------------------------------------------------------------
# Live re-verification — skipped without the weights, a real gate with them.
# --------------------------------------------------------------------------------------------------


def test_the_live_checkpoint_header_still_matches_the_fixture(facts: dict[str, Any]) -> None:
    """Re-read the real 40 GiB checkpoint's HEADER and re-run the gate against it.

    Header only: 8-byte length + JSON. The fixture cannot rot silently while this can run.
    """
    checkpoint = _live(CHECKPOINT_ENV)
    if checkpoint is None:
        pytest.skip(f"no checkpoint at ${CHECKPOINT_ENV} — the fixture carries the same facts")

    keys = _header_keys(checkpoint)
    assert len(keys) == facts["transformer"]["tensor_count"]
    assert checkpoint.stat().st_size == facts["transformer"]["bytes"]

    modules = _module_names(keys)
    assert modules == facts["transformer"]["module_names"]
    assert assert_qwen_edit_targets(modules)["total"] == EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT


def test_the_live_text_encoder_header_still_carries_its_vision_tower(facts: dict[str, Any]) -> None:
    """Re-read the real Qwen2.5-VL header and re-run the vision gate against every key."""
    encoder = _live(TEXT_ENCODER_ENV)
    if encoder is None:
        pytest.skip(f"no encoder at ${TEXT_ENCODER_ENV} — the fixture carries the same facts")

    keys = _header_keys(encoder)
    assert len(keys) == facts["text_encoder_vl"]["tensor_count"]
    census = assert_qwen_edit_text_encoder_vision(keys, what="Qwen2.5-VL text encoder")
    assert census["vision"] == EXPECTED_QWEN_VL_VISION_TENSOR_COUNT
