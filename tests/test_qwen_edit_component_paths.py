"""Family #3 (``qwen_edit``) COMPONENT-PATH contract — the CPU test that would have caught the
three live-dispatch failures of 2026-08-08.

Every failure this file pins was discovered on a metered A100, one dispatch at a time, because the
CPU dry-run validates CONFIG and SHAPES and cannot see a real checkpoint's directory layout or the
argument types a library's ``from_pretrained`` will accept. Each round cost an arch-gate load
(~38 GiB, ~1 min) to learn one fact. That is the wrong trade, and it is avoidable: none of these
three need weights to test, only the LAYOUT and the CALL.

The three, in the order they were hit:

  1. the PROCESSOR was loaded from ``<root>/text_encoder`` -> OSError, no preprocessor_config.json
  2. then from ``<root>`` -> ValueError: Unrecognized model. The root holds diffusers'
     ``model_index.json``, a PIPELINE index carrying no ``model_type`` for AutoProcessor to
     dispatch on. transformers then prints all ~400 known model types, which is how the failure
     announces itself.
  3. ``subfolder=None`` was passed to ``from_pretrained`` -> TypeError: expected str, bytes or
     os.PathLike object, not NoneType, raised inside ``os.path.join(subfolder, path)``
     (transformers/configuration_utils.py:709). transformers' own default is ``""``; it does not
     normalise None.

The house lesson these violate is already written down, in the H3 arch gate: *"caught 6 mismatches
in one ~$1.40 run precisely because it did not stop at the first."* Report everything at once,
and find it at the cheapest point that can find it. These tests are that cheapest point.

CPU only. No weights, no network, no Modal, no GPU.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOADER = _REPO_ROOT / "src" / "signet_trainer" / "models" / "qwen_edit_loader.py"
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"

#: The real Qwen-Image-Edit-2511 diffusers snapshot layout, read off the weights Volume after
#: ``download_qwen_edit_weights`` ran (31 files, 4m21s). This is EVIDENCE, not a convention — the
#: component a loader must address is decided by which directory holds which config file.
SNAPSHOT_LAYOUT: dict[str, tuple[str, ...]] = {
    "": ("model_index.json",),  # the ROOT: a diffusers PIPELINE index. No model_type. Not loadable
    #                              by AutoProcessor / AutoModel.
    "transformer": (
        "config.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00001-of-00005.safetensors",
        "diffusion_pytorch_model-00005-of-00005.safetensors",
    ),
    "vae": ("config.json", "diffusion_pytorch_model.safetensors"),
    "text_encoder": (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ),
    "processor": (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
    ),
    "tokenizer": ("tokenizer_config.json",),
    "scheduler": ("scheduler_config.json",),
}


# ==================================================================================================
# 1 — the LAYOUT facts. Which directory is addressable by which loader, and why.
# ==================================================================================================


def test_only_the_processor_subfolder_carries_a_preprocessor_config() -> None:
    """``preprocessor_config.json`` lives in ``processor/`` and NOWHERE else in the snapshot.

    This is the whole content of failure (1): the processor is not a sibling of the text encoder's
    weights, it is its own component directory. Asserting the negative matters as much as the
    positive — it is what makes "load the processor from text_encoder/" impossible to reintroduce.
    """
    carriers = [d for d, files in SNAPSHOT_LAYOUT.items() if "preprocessor_config.json" in files]
    assert carriers == ["processor"], (
        f"preprocessor_config.json is carried by {carriers}, expected exactly ['processor']"
    )
    assert "preprocessor_config.json" not in SNAPSHOT_LAYOUT["text_encoder"]
    assert "preprocessor_config.json" not in SNAPSHOT_LAYOUT[""]


def test_the_pipeline_root_carries_no_transformers_config() -> None:
    """The root has ``model_index.json`` and no ``config.json`` — failure (2), as a fact.

    ``AutoProcessor``/``AutoModel`` dispatch on ``config.json``'s ``model_type``. A diffusers
    pipeline root has neither, so pointing a transformers auto-class at it raises
    "Unrecognized model" and dumps every registered model type. The root is for diffusers'
    ``DiffusionPipeline.from_pretrained``, which is a different consumer.
    """
    root = SNAPSHOT_LAYOUT[""]
    assert "model_index.json" in root
    assert "config.json" not in root, (
        "the pipeline root must NOT be addressed by a transformers auto-class; if a config.json "
        "ever appears here, revisit this test rather than deleting it"
    )


@pytest.mark.parametrize(
    ("component", "required"),
    [
        ("transformer", "config.json"),
        ("vae", "config.json"),
        ("text_encoder", "config.json"),
        ("processor", "preprocessor_config.json"),
    ],
)
def test_each_component_directory_is_self_describing(component: str, required: str) -> None:
    """Every component a loader addresses carries its own config beside its weights.

    That is the property making the diffusers DIRECTORY layout the right choice over the ComfyUI
    single-file layout: ``from_single_file`` with no explicit ``config=`` calls
    ``infer_diffusers_model_type``, which has no Qwen branch on the pinned diffusers.
    """
    assert required in SNAPSHOT_LAYOUT[component]


# ==================================================================================================
# 2 — the CALL contract. subfolder must never reach a library as None.
# ==================================================================================================


def test_no_from_pretrained_call_passes_a_bare_subfolder_variable() -> None:
    """No ``subfolder=<name>,`` may reach a library call un-normalised — failure (3).

    ``load_qwen_edit_*`` all declare ``subfolder: str | None = None`` so a caller may omit it, but
    transformers does ``os.path.join(subfolder, pretrained_model_name_or_path)``
    (configuration_utils.py:709) with no guard and its own default is ``""``. Passing the None
    through raises TypeError AFTER the arch gate has already loaded 38 GiB.

    Scanned as SOURCE rather than exercised with a mock on purpose: the failure is a value flowing
    into third-party code, so the check that survives a refactor is "the value is normalised at
    every site", not "this one call was made with this one argument".
    """
    src = _LOADER.read_text(encoding="utf-8")
    offenders = [
        (i, line.strip())
        for i, line in enumerate(src.splitlines(), start=1)
        if re.search(r"subfolder\s*=\s*subfolder\s*,", line)
    ]
    assert not offenders, (
        "subfolder passed through un-normalised at "
        + "; ".join(f"{_LOADER.name}:{i} {text!r}" for i, text in offenders)
        + ". Use `subfolder=subfolder or \"\"` — None is not accepted by transformers."
    )


def test_every_loader_normalises_subfolder() -> None:
    """Positive form: each library call site uses the ``or ""`` normalisation."""
    src = _LOADER.read_text(encoding="utf-8")
    normalised = re.findall(r'subfolder\s*=\s*subfolder\s+or\s+""', src)
    assert len(normalised) >= 4, (
        f"expected >=4 normalised subfolder call sites (transformer x2, text encoder, VAE), "
        f"found {len(normalised)}"
    )


@pytest.mark.parametrize(
    "func_name",
    ["load_qwen_edit_transformer", "load_qwen_edit_text_encoder", "load_qwen_edit_vae"],
)
def test_loader_subfolder_default_is_none_and_that_is_why_normalisation_is_required(
    func_name: str,
) -> None:
    """The signatures genuinely default ``subfolder`` to None, so normalisation is load-bearing.

    If a future refactor changes the default to ``""`` this test fails LOUDLY rather than leaving
    the normalisation looking like dead defensive code someone later removes.
    """
    from signet_trainer.models import qwen_edit_loader

    func = getattr(qwen_edit_loader, func_name, None)
    if func is None:
        pytest.skip(f"{func_name} not present")
    param = inspect.signature(func).parameters.get("subfolder")
    assert param is not None, f"{func_name} has no subfolder parameter"
    assert param.default is None, (
        f"{func_name}'s subfolder default is {param.default!r}, not None — if this became \"\" the "
        f"`or \"\"` normalisation is redundant and this test should be updated deliberately"
    )


# ==================================================================================================
# 3 — the WIRING. The preprocess stage addresses each component at its own directory.
# ==================================================================================================


def test_preprocess_loads_the_processor_from_the_pipeline_root_processor_subfolder() -> None:
    """``qwen_edit_preprocess`` composes the processor path as ``<root>/processor``.

    Pinned as source because the alternatives are both silently plausible: ``<root>`` and
    ``<root>/text_encoder`` are each a real directory that a reader could believe is right, and
    both were tried on live hardware before this landed.
    """
    src = _FNS.read_text(encoding="utf-8")
    assert re.search(
        r'_qwen_edit_load_processor\(\s*str\(\s*WEIGHTS_DIR\s*/\s*pipeline_root_id\s*/\s*"processor"\s*\)',
        src,
    ), (
        "qwen_edit_preprocess must load the processor from WEIGHTS_DIR / pipeline_root_id / "
        '"processor". Loading it from the root raises "Unrecognized model"; loading it from '
        "text_encoder raises \"Can't load image processor\"."
    )


def test_preprocess_refuses_without_a_pipeline_root() -> None:
    """A missing ``pipeline_root_id`` is refused with the remedy in the message, not a TypeError."""
    src = _FNS.read_text(encoding="utf-8")
    assert "pipeline_root_id is unset" in src, (
        "qwen_edit_preprocess must refuse an unset pipeline_root_id explicitly — otherwise the "
        "path composition fails with an unattributed TypeError inside a metered container"
    )


def test_the_entrypoint_gap_check_catches_a_missing_root_before_dispatch() -> None:
    """The $0 form of the same refusal: the entrypoint names it before any GPU is provisioned."""
    from signet_trainer.config.load import load_config
    from signet_trainer.modal.entrypoint import _qwen_edit_config_gaps

    cfg = load_config(str(_REPO_ROOT / "configs" / "qwen_image_edit.example.yaml"))
    assert not _qwen_edit_config_gaps(cfg, mode="preprocess"), (
        "the shipped example config must be dispatchable — a shipped example that cannot run is "
        "what every new family arm gets copied from"
    )

    object.__setattr__(cfg.model, "pipeline_root_id", None)
    gaps = _qwen_edit_config_gaps(cfg, mode="preprocess")
    assert len(gaps) == 1 and "pipeline_root_id" in gaps[0]
    assert "processor" in gaps[0], "the gap must explain WHY the root is needed, not just that it is"


def test_the_shipped_example_config_declares_every_component() -> None:
    """The example names all four component ids, so a copy of it addresses real directories."""
    from signet_trainer.config.load import load_config

    cfg = load_config(str(_REPO_ROOT / "configs" / "qwen_image_edit.example.yaml"))
    assert cfg.model.pipeline_root_id, "pipeline_root_id must be declared"
    for field, expected_leaf in (
        ("model_id", "transformer"),
        ("vae_id", "vae"),
        ("text_encoder_id", "text_encoder"),
    ):
        value = str(getattr(cfg.model, field))
        assert value.endswith(expected_leaf), (
            f"model.{field} is {value!r}; the diffusers snapshot addresses this component at "
            f"<root>/{expected_leaf}"
        )
        assert value.startswith(f"{cfg.model.pipeline_root_id}/"), (
            f"model.{field} must live under the declared pipeline root "
            f"{cfg.model.pipeline_root_id!r} — a component from a different snapshot than the "
            f"processor is an arch mismatch that loads clean"
        )


# ── the fourth live-dispatch failure, 2026-08-10 ───────────────────────────────────────────────────
# ``qwen_edit_sample`` read ``assert_qwen_edit_text_encoder_vision(text_encoder)["summary"]``. The
# census has no such key, so the stage raised KeyError AFTER the arch gate, qfloat8, the adapter
# injection and a ~40.9 GiB load — the most expensive line in the stage at which to discover a typo,
# and one that can only ever fire on real weights. Same shape as the three above, same remedy: pin
# the CONTRACT at the cheapest point that can see it.

#: Every key ``qwen_vl_vision_census`` documents and returns. Anything a caller subscripts that is
#: not in here is a KeyError waiting for a metered container.
CENSUS_KEYS: frozenset[str] = frozenset({"total", "vision", "examples"})


def test_vision_census_returns_exactly_its_documented_keys() -> None:
    """The census contract, driven by tensor NAMES — no weights, no torch, no GPU."""
    import sys

    sys.path.insert(0, str(_REPO_ROOT / "src"))
    from signet_trainer.models.qwen_edit_loader import qwen_vl_vision_census

    census = qwen_vl_vision_census(
        ["visual.blocks.0.attn.qkv.weight", "model.layers.0.self_attn.q_proj.weight"]
    )
    assert set(census) == CENSUS_KEYS, (
        f"qwen_vl_vision_census returned keys {sorted(census)}; the documented contract is "
        f"{sorted(CENSUS_KEYS)}. Callers subscript this dict by literal key."
    )


def test_no_call_site_subscripts_an_undocumented_census_key() -> None:
    """No ``assert_qwen_edit_text_encoder_vision(...)[...]`` may read a key the census lacks.

    A source scan rather than a call, deliberately: the defect was at the CALL SITE, inside a Modal
    stage that cannot be executed without ~40.9 GiB of weights. The cheapest thing that can see it
    is the text.
    """
    source = _FNS.read_text(encoding="utf-8")
    subscripts = re.findall(
        r"assert_qwen_edit_text_encoder_vision\([^)]*\)\s*\[\s*[\"']([^\"']+)[\"']\s*\]", source
    )
    bad = sorted({key for key in subscripts if key not in CENSUS_KEYS})
    assert not bad, (
        f"modal/fns.py subscripts assert_qwen_edit_text_encoder_vision(...) with {bad}, which the "
        f"census never returns (it has {sorted(CENSUS_KEYS)}). This raises KeyError only on real "
        f"weights, after the arch gate and a ~40.9 GiB load."
    )
