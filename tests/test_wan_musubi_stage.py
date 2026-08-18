"""Slice B — the Wan/musubi Modal stage: the recipe, the gap fence, the gated arm, the batch line.

Everything here is CPU-only, filesystem-read-only and free. Nothing dispatches, nothing downloads,
nothing touches a GPU. The modal-touching imports are done INSIDE test bodies (never at module top)
so pytest COLLECTION does not pull the SDK into ``sys.modules`` and the dry-run purity guards in
``test_dryrun_*.py`` stay meaningful — the discipline ``test_entrypoint_gate_behavioral.py``
records.

WHAT IS BEING DEFENDED, in the order the money is at risk:

  1. the RECIPE is a transcription, so it is diffed against the oracle it was transcribed from
     (``docs/source-methods/musubi-wan21/train_kohya.py``) rather than restated in prose;
  2. the stage is THREADED, not config-by-value, because the musubi image carries pydantic 1.x —
     so the module it imports in-container must stay pydantic-free, asserted by import closure;
  3. the gap fence fires in BOTH directions, and the fully-declared direction is what proves the
     refusals are about the config rather than about the feature being unfinished;
  4. the approval pause still stops a dispatch that was never approved.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from signet_trainer.config.load import load_config
from signet_trainer.config.sources import SourceSpec
from signet_trainer.runners.wan_musubi import (
    WAN_COMPONENT_CONFIG_FIELDS,
    WAN_MUSUBI_RECIPE,
    WanComponents,
    wan_clips_per_media,
    wan_resolve_component_ids,
    wan_stage_argv,
    wan_train_network_argv,
)

_REPO = Path(__file__).resolve().parents[1]
_ORACLE = _REPO / "docs" / "source-methods" / "musubi-wan21" / "train_kohya.py"
_EXAMPLE = _REPO / "configs" / "wan21_kaboom.example.yaml"
_ENTRYPOINT = _REPO / "src" / "signet_trainer" / "modal" / "entrypoint.py"
_FNS = _REPO / "src" / "signet_trainer" / "modal" / "fns.py"
_WAN_MUSUBI = _REPO / "src" / "signet_trainer" / "runners" / "wan_musubi.py"

_COMPONENTS = WanComponents(
    dit="/weights/wan/dit.safetensors",
    vae="/weights/wan/vae.safetensors",
    t5="/weights/wan/umt5.pth",
    clip="/weights/wan/open-clip.pth",
)


def _oracle_flags() -> dict[str, str]:
    """Every ``"--flag", "value"`` pair in the oracle's training argv, read off its AST.

    AST rather than a regex over the text: the oracle is a real python file and its argv is a real
    list literal, so the pairs can be recovered exactly instead of approximately. Flags with no
    value (``--sdpa``) map to ``""``.
    """
    tree = ast.parse(_ORACLE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        items = [e.value for e in node.elts if isinstance(e, ast.Constant)]
        if "wan_train_network.py" in items:
            flags: dict[str, str] = {}
            for i, item in enumerate(items):
                if isinstance(item, str) and item.startswith("--"):
                    nxt = items[i + 1] if i + 1 < len(items) else "--"
                    flags[item] = "" if str(nxt).startswith("--") else str(nxt)
            return flags
    raise AssertionError(
        "no argv list containing 'wan_train_network.py' in the oracle — the transcription this "
        "whole file diffs against has moved, so every assertion below is vacuous until it is found."
    )


# ==================================================================================================
# (1) The recipe IS the transcription — diffed against the oracle, never restated
# ==================================================================================================


def test_the_training_argv_reproduces_every_oracle_flag() -> None:
    """Each ``--flag value`` in ``train_kohya.py``'s training call appears in the rendered argv.

    The five values signet legitimately re-points (the four component paths and the dataset TOML)
    are compared by FLAG PRESENCE only; everything else must match by value. Restating the flag list
    in this test would make it a copy of the thing it checks — reading it off the oracle means a
    transcription that silently drops ``--fp8_scaled`` fails here rather than producing a run
    nobody asked for at an ordinary-looking loss.
    """
    argv = wan_train_network_argv(
        dataset_config="/ckpt/run/dataset-config.toml",
        components=_COMPONENTS,
        output_dir="/ckpt/run",
        output_name="run",
        seed=42,
    )
    rendered: dict[str, str] = {}
    for i, token in enumerate(argv):
        if token.startswith("--"):
            nxt = argv[i + 1] if i + 1 < len(argv) else "--"
            rendered[token] = "" if nxt.startswith("--") else nxt

    #: Flags whose VALUE signet re-points on purpose. Everything else is compared exactly.
    repointed = {"--dit", "--vae", "--t5", "--clip", "--dataset_config", "--output_dir", "--output_name"}

    oracle = _oracle_flags()
    missing = sorted(set(oracle) - set(rendered))
    assert not missing, f"the rendered argv drops oracle flag(s) {missing}"
    drifted = {
        flag: (value, rendered[flag])
        for flag, value in oracle.items()
        if flag not in repointed and rendered[flag] != value
    }
    assert not drifted, (
        f"rendered argv diverges from the transcription at {drifted} (oracle, rendered). Every one "
        f"of these is a recipe term settled by the method, not a knob."
    )


def test_the_locked_shift_is_wans_own_value_and_reaches_the_argv() -> None:
    """7.0 — the one shift literal this family may pin, and it must actually be EMITTED.

    A recipe constant nobody threads into the command line is decoration. The value gate
    (``test_no_wan_params.py``) proves the file says only 7.0; this proves musubi is told.
    """
    assert WAN_MUSUBI_RECIPE.discrete_flow_shift == 7.0
    argv = wan_train_network_argv(
        dataset_config="/d.toml", components=_COMPONENTS, output_dir="/o", output_name="o", seed=1
    )
    assert argv[argv.index("--discrete_flow_shift") + 1] == "7.0"
    assert _oracle_flags()["--discrete_flow_shift"] == "7.0", (
        "the oracle no longer pins 7.0 — re-derive the recipe before trusting this constant"
    )


def test_mixed_precision_is_passed_twice_because_two_programs_read_it() -> None:
    """``accelerate launch`` takes one and ``wan_train_network.py`` takes its own. Dropping one changes the run."""
    argv = wan_train_network_argv(
        dataset_config="/d.toml", components=_COMPONENTS, output_dir="/o", output_name="o", seed=1
    )
    assert argv.count("--mixed_precision") == 2, (
        "one --mixed_precision is accelerate's and one is the training script's; a de-duplicating "
        "'cleanup' silently changes what runs"
    )
    assert argv[:2] == ["accelerate", "launch"]


def test_the_seed_is_threaded_not_the_oracles_literal() -> None:
    """``cfg.seed`` reaches ``--seed`` (D-NOHARDCODE) — a second round is expressible."""
    argv = wan_train_network_argv(
        dataset_config="/d.toml", components=_COMPONENTS, output_dir="/o", output_name="o", seed=1234
    )
    assert argv[argv.index("--seed") + 1] == "1234"


def test_the_three_stages_are_ordered_and_share_one_dataset_config() -> None:
    """Cache latents -> cache text encoder -> train, all pointed at the SAME rendered TOML.

    The order is load-bearing (the trainer reads caches the first two write, and musubi does not
    build them on demand) and so is the shared path: three passes reading two different files would
    train against a dataset the committed artifact does not describe.
    """
    stages = wan_stage_argv(
        dataset_config="/ckpt/run/dataset-config.toml",
        components=_COMPONENTS,
        output_dir="/ckpt/run",
        output_name="run",
        seed=42,
    )
    assert [label for label, _ in stages] == [
        "cache_latents",
        "cache_text_encoder_outputs",
        "train_network",
    ]
    for _label, argv in stages:
        assert argv[argv.index("--dataset_config") + 1] == "/ckpt/run/dataset-config.toml"
    # The cache passes take the encoders they actually use, and NOT the DiT — loading a 14B
    # transformer to cache latents would be paid-for work nobody asked for.
    assert "--clip" in stages[0][1] and "--vae" in stages[0][1] and "--dit" not in stages[0][1]
    assert "--t5" in stages[1][1] and "--dit" not in stages[1][1]


# ==================================================================================================
# (2) The threading is FORCED by the image — so the in-container module must stay pydantic-free
# ==================================================================================================


def test_the_wan_runner_module_imports_no_pydantic_torch_or_modal() -> None:
    """The hard requirement, DERIVED from a real import in a SUBPROCESS rather than asserted.

    In-process this would be order-dependent — pydantic is already in ``sys.modules`` by the time
    any test runs, so an accidental import would be invisible. A subprocess with a clean interpreter
    is the only honest way to ask "what does importing this module pull in", and it is the
    convention this repo already uses for every ``sys.modules``-inspecting check.

    This is not hygiene. ``wan_musubi_image`` carries musubi's ``pydantic==1.10.13``
    (train_kohya.py:37) while ``config/load`` needs pydantic>=2.10.4, so a pydantic import in this
    module's closure breaks ``wan_train`` inside a metered container — and breaks it on a laptop
    where both versions happen to resolve, i.e. invisibly.
    """
    probe = (
        "import sys; base=set(sys.modules);"
        "import signet_trainer.runners.wan_musubi;"
        "print(sorted({m.split('.')[0] for m in set(sys.modules)-base}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO,
        env={**os.environ, "PYTHONPATH": str(_REPO / "src"), "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"the probe itself failed:\n{result.stderr}"
    pulled = set(ast.literal_eval(result.stdout.strip()))
    assert pulled, "the probe reported an EMPTY import diff — it is broken, not the module"
    forbidden = pulled & {"pydantic", "torch", "modal", "yaml", "numpy"}
    assert not forbidden, (
        f"importing runners.wan_musubi pulls in {sorted(forbidden)}. That module is imported INSIDE "
        f"the musubi container, whose pydantic is 1.10.13; anything here that needs pydantic v2 "
        f"(or torch, or the modal SDK) makes wan_train abort at its first statement."
    )


def test_the_wan_stage_takes_threaded_params_and_every_one_is_required() -> None:
    """``wan_train`` must NOT take ``config_yaml``, and must default NOTHING.

    Both halves matter and they are the same lesson from two directions. A ``config_yaml``
    parameter would mean calling ``load_config_from_text`` on an interpreter that cannot, so its
    ABSENCE is the design. A defaulted parameter is how a threading gap goes silent: with no
    defaults a missing kwarg is a ``TypeError`` at dispatch, before a container is allocated.
    """
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "wan_train"
    )
    args = [a.arg for a in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]]
    assert "config_yaml" not in args, (
        "wan_train takes a config_yaml — but wan_musubi_image carries musubi's pydantic 1.10.13 and "
        "load_config_from_text needs pydantic>=2.10.4, so the stage cannot parse one in-container"
    )
    assert not fn.args.defaults and not [d for d in fn.args.kw_defaults if d is not None], (
        f"wan_train has defaulted parameter(s); every one must be REQUIRED so a threading gap is a "
        f"TypeError at dispatch rather than a silent wrong default inside a paid container "
        f"(params: {args})"
    )


def test_the_entrypoint_supplies_exactly_the_stages_required_params() -> None:
    """SELF-DERIVING: the helper's dict keys are diffed against the real signature. Neither is listed.

    The ``test_h3_entrypoint_gate.py`` guard, applied to family #4 — a hand-typed list of eight
    names rots the first time a parameter is added.
    """
    stage = next(
        node
        for node in ast.walk(ast.parse(_FNS.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "wan_train"
    )
    required = {a.arg for a in [*stage.args.posonlyargs, *stage.args.args, *stage.args.kwonlyargs]}
    helper = next(
        node
        for node in ast.walk(ast.parse(_ENTRYPOINT.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "_wan_train_params"
    )
    supplied = {
        key.value
        for node in ast.walk(helper)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert required - supplied == set(), f"_wan_train_params does not supply {sorted(required - supplied)}"
    assert supplied - required == set(), f"_wan_train_params supplies unknown {sorted(supplied - required)}"


def test_the_dataset_toml_is_rendered_at_dispatch_not_baked_into_the_image() -> None:
    """The feature's structural claim, asserted as structure.

    ``train_kohya.py:40-43`` does ``.add_local_file("wan21-dataset-config.toml", ...)`` — the TOML
    is an IMAGE input there, so changing the dataset is an image rebuild and "what did round 2 train
    on?" is unanswerable from the artifacts. signet renders it from the validated manifest in
    ``_wan_train_params`` and ships it by value.
    """
    entry_src = _ENTRYPOINT.read_text(encoding="utf-8")
    assert "render_from_config(cfg)" in entry_src, (
        "the entrypoint no longer renders the dataset TOML at dispatch time — that render is what "
        "makes 'only the dataset changed between rounds' a property of the artifacts"
    )
    app_src = (_REPO / "src" / "signet_trainer" / "modal" / "app.py").read_text(encoding="utf-8")
    stripped = re.sub(r"#.*", "", app_src)
    assert "add_local_file" not in stripped, (
        "a dataset file is being baked into an image (add_local_file) — that is the "
        "train_kohya.py:40-43 shape this stage exists to replace"
    )


def test_the_stage_checks_every_subprocess_return_code() -> None:
    """``check=True`` — the one deviation from the transcription that prevents a silent bad round.

    ``train_kohya.py:105,117,129`` calls ``subprocess.run`` with no return-code inspection, so a
    failed cache pass there proceeds to training against an empty cache and produces an adapter
    trained on nothing, at a perfectly ordinary loss curve.
    """
    calls = [
        node
        for node in ast.walk(ast.parse(_FNS.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert calls, "no subprocess.run call in fns.py — the wan stage's three passes vanished"
    for call in calls:
        checked = [k for k in call.keywords if k.arg == "check"]
        assert checked and getattr(checked[0].value, "value", False) is True, (
            f"subprocess.run at fns.py:{call.lineno} does not pass check=True; a failed musubi pass "
            f"would be swallowed and the next pass would run against its missing output"
        )


# ==================================================================================================
# (3) The gap fence — both directions
# ==================================================================================================


def _stub_model(**overrides):
    """A ``model:`` block with all four Wan components DECLARED. Not a ModelConfig.

    A SimpleNamespace, still, now for a different reason than when it was written. It used to be a
    workaround: ``ModelConfig`` had no ``clip_id`` field, so under ``extra="forbid"`` the
    fully-declared case was unexpressible as a real config. ``clip_id`` landed 2026-08-10 and
    ``test_a_real_model_config_can_now_declare_all_four`` covers that path with the genuine article.

    This stub stays because it exercises the OTHER half of the contract: every consumer reads the
    block by ``getattr`` with a default, so the resolver must work on anything block-shaped and must
    not quietly acquire a dependency on pydantic — ``runners/wan_musubi`` has to stay importable on
    an interpreter that has none (see the module docstring).
    """
    fields = {
        "model_id": "wan/wan2.1_t2v_14B_bf16.safetensors",
        "vae_id": "wan/wan_2.1_vae.safetensors",
        "text_encoder_id": "wan/models_t5_umt5-xxl-enc-bf16.pth",
        "clip_id": "wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    }
    fields.update(overrides)
    return SimpleNamespace(family="wan", **fields)


def test_a_fully_declared_model_block_resolves_all_four_components() -> None:
    """The POSITIVE direction. Without it every refusal below could be a feature that never works."""
    components = wan_resolve_component_ids(_stub_model())
    assert components.dit.endswith("wan2.1_t2v_14B_bf16.safetensors")
    assert components.clip.endswith("open-clip-xlm-roberta-large-vit-huge-14.pth")
    assert set(WAN_COMPONENT_CONFIG_FIELDS) == {"dit", "vae", "t5", "clip"}


def test_an_INHERITED_ltx_default_is_refused_as_firmly_as_a_missing_id() -> None:
    """The silent wrong-weights hole: ``text_encoder_id`` defaults to LTX's Gemma, not to None.

    A truthiness test alone RESOLVES on a wan config that declared no text encoder, and the stage
    would hand musubi ``--t5 /weights/gemma-3-12b-it`` — a Gemma directory where it expects a umT5
    checkpoint — after the latent cache pass has already run.
    """
    from signet_trainer.config.schema import ModelConfig

    gemma = ModelConfig.model_fields["text_encoder_id"].default
    assert gemma, "text_encoder_id has no schema default — this test's premise is gone"
    # Without the defaults it resolves. That is the hole, demonstrated rather than described.
    assert wan_resolve_component_ids(_stub_model(text_encoder_id=gemma)).t5 == gemma
    # With them it refuses, and says WHICH kind of gap it is.
    with pytest.raises(NotImplementedError, match="INHERITED FROM THE LTX DEFAULT"):
        wan_resolve_component_ids(
            _stub_model(text_encoder_id=gemma), schema_defaults={"text_encoder_id": gemma}
        )


@pytest.mark.parametrize("mode", ["sample", "preprocess", "fuse", "restore", "backup"])
def test_every_mode_but_train_is_refused_by_name(mode: str) -> None:
    """The FAMILY FENCE. Without it a wan config on ``--mode sample`` reaches the LTX sampler.

    That fall-through used to be blocked for free by the dry-run refusal, which failed every mode.
    The manifest gate now passes, so this fence is what replaced that protection and it is the one
    regression a reader of slice B should worry about.
    """
    from signet_trainer.modal.entrypoint import _wan_config_gaps

    gaps = _wan_config_gaps(load_config(_EXAMPLE), mode=mode)
    assert any(f"--mode {mode!r} has no wan arm" in gap for gap in gaps), (
        f"mode {mode!r} is not refused by name; the gaps were {gaps}"
    )


def test_train_mode_is_not_refused_for_being_train() -> None:
    """The negative control for the fence: ``train`` produces no MODE gap (only the real ones)."""
    from signet_trainer.modal.entrypoint import _wan_config_gaps

    gaps = _wan_config_gaps(load_config(_EXAMPLE), mode="train")
    assert not any("has no wan arm" in gap for gap in gaps)
    # The component gap is now CLOSED on the shipped config — download_wan_weights stages the four
    # and the config declares them — so asserting it FIRES here would pin the branch to its own
    # unfinished state. What must still hold is that an UNDECLARED config is refused; that is
    # test_the_component_gap_still_fires_on_an_undeclared_config below.
    assert not any("wan_resolve_component_ids" in gap for gap in gaps), (
        f"the shipped wan config no longer resolves its components: {gaps}"
    )


def test_the_component_gap_still_fires_on_an_undeclared_config() -> None:
    """The refusal that used to fire on the shipped config must still fire on an incomplete one.

    Closing a gap by declaring the ids is progress; closing it by weakening the check is not, and
    the two look identical from a green suite. This drives the seam with model_id/text_encoder_id
    left at their LTX defaults — the dangerous case, because an inherited default is not an empty
    value and a truthiness test would resolve it straight into `--t5 <weights>/gemma-3-12b-it`.
    """
    from signet_trainer.modal.entrypoint import _wan_config_gaps

    cfg = load_config(_EXAMPLE)
    bare = SimpleNamespace(
        data=cfg.data,
        model=SimpleNamespace(family="wan", model_id=None, vae_id=None, text_encoder_id=None, clip_id=None),
        training=cfg.training,
    )
    gaps = _wan_config_gaps(bare, mode="train")
    assert any("wan_resolve_component_ids" in gap for gap in gaps), gaps


def test_the_unpinned_musubi_checkout_gap_fires_ONLY_when_it_is_unpinned(monkeypatch) -> None:
    """Both directions, because the pin landing must not silently retire the guard that demands it.

    MUSUBI_TUNER_COMMIT_SHA was None while the SHA was a declared gap; it is now a literal, so the
    gap correctly stops firing. That is a behaviour change in a money-safe check, and deleting the
    assertion would have been the easy way to make this file green — leaving nothing to catch a
    later edit that sets it back to None or to a floating ref.
    """
    from signet_trainer.modal import app, entrypoint

    assert app.MUSUBI_TUNER_COMMIT_SHA and len(app.MUSUBI_TUNER_COMMIT_SHA) == 40, (
        "the musubi checkout is unpinned again — a floating clone means musubi's dataset-config "
        "schema can change under the renderer between two builds of the same image"
    )
    pinned = entrypoint._wan_config_gaps(load_config(_EXAMPLE), mode="train")
    assert not any("MUSUBI_TUNER_COMMIT_SHA" in gap for gap in pinned)

    monkeypatch.setattr(app, "MUSUBI_TUNER_COMMIT_SHA", None)
    unpinned = entrypoint._wan_config_gaps(load_config(_EXAMPLE), mode="train")
    assert any("MUSUBI_TUNER_COMMIT_SHA" in gap for gap in unpinned), (
        "un-pinning no longer produces a gap — the guard is decorative"
    )


def test_the_gap_list_is_EMPTY_when_everything_is_declared(monkeypatch) -> None:
    """THE DIRECTION THAT MATTERS MOST: the refusals are about the config, not about the feature.

    A gap check that can never go green is indistinguishable from a stage that does not work, and
    the reader has no way to tell which. This drives every gap to its satisfied state — sources
    declared, caches distinct, caption extension set, four components named, SHA pinned — and
    asserts nothing is left.
    """
    from signet_trainer.modal import app, entrypoint

    monkeypatch.setattr(app, "MUSUBI_TUNER_COMMIT_SHA", "0" * 40)
    cfg = load_config(_EXAMPLE)
    stub = SimpleNamespace(data=cfg.data, model=_stub_model(), training=cfg.training)
    assert entrypoint._wan_config_gaps(stub, mode="train") == []


def test_a_blank_caption_extension_is_refused_before_it_renders_empty_captions(monkeypatch) -> None:
    """``caption_extension = ""`` is valid TOML and trains every clip on an empty caption.

    Not a hypothetical: ``_toml_string("")`` renders cleanly, so the renderer would emit the line
    and musubi would look for a caption file at the bare media stem.
    """
    from signet_trainer.modal import app, entrypoint
    from signet_trainer.runners.musubi_toml import render_musubi_toml

    monkeypatch.setattr(app, "MUSUBI_TUNER_COMMIT_SHA", "0" * 40)
    cfg = load_config(_EXAMPLE)
    blank_data = SimpleNamespace(
        sources=cfg.data.sources,
        preprocessed_data_root=cfg.data.preprocessed_data_root,
        caption_extension="",
    )
    stub = SimpleNamespace(data=blank_data, model=_stub_model(), training=cfg.training)
    gaps = entrypoint._wan_config_gaps(stub, mode="train")
    assert len(gaps) == 1 and "caption_extension" in gaps[0]
    # The defect the refusal describes, demonstrated: the renderer really does emit it.
    rendered = render_musubi_toml(cfg.data.sources, data_root="/r", caption_extension="")
    assert 'caption_extension = ""' in rendered


def test_colliding_cache_roots_are_refused_at_the_dispatch_seam(monkeypatch) -> None:
    """Held a SECOND time here, and not for tidiness.

    musubi deletes cache files absent from the current dataset spec — correct when the spec changes
    between rounds, and safe only because every source has a unique cache directory. Two sources
    sharing one mutually destroy each other's latents inside a paid container.
    """
    from signet_trainer.modal import app, entrypoint

    monkeypatch.setattr(app, "MUSUBI_TUNER_COMMIT_SHA", "0" * 40)
    cfg = load_config(_EXAMPLE)
    collided = [
        SourceSpec(
            id="appearance",
            kind="video",
            directory="/d/V",
            resolution=(1280, 720),
            extraction="head",
            target_frames=[21],
            cache_root="/d/V/cache_1",
        ),
        SourceSpec(
            id="motion",
            kind="video",
            directory="/d/V",
            resolution=(640, 352),
            extraction="uniform",
            target_frames=[45],
            frame_sample=2,
            cache_root="/d/V/cache_1/",  # ONE directory, two spellings
        ),
    ]
    stub = SimpleNamespace(
        data=SimpleNamespace(
            sources=collided, preprocessed_data_root="/r", caption_extension=".txt"
        ),
        model=_stub_model(),
        training=cfg.training,
    )
    gaps = entrypoint._wan_config_gaps(stub, mode="train")
    assert len(gaps) == 1 and "cache collision" in gaps[0] and "mutually destroy" in gaps[0]


def test_an_absent_source_list_is_refused_at_the_dispatch_seam(monkeypatch) -> None:
    """The renderer runs at THIS seam, so its refusal is reported as a gap rather than a traceback."""
    from signet_trainer.modal import app, entrypoint

    monkeypatch.setattr(app, "MUSUBI_TUNER_COMMIT_SHA", "0" * 40)
    cfg = load_config(_EXAMPLE)
    stub = SimpleNamespace(
        data=SimpleNamespace(sources=None, preprocessed_data_root="/r", caption_extension=".txt"),
        model=_stub_model(),
        training=cfg.training,
    )
    gaps = entrypoint._wan_config_gaps(stub, mode="train")
    assert len(gaps) == 1 and "data.sources is absent or empty" in gaps[0]


def test_the_refusal_names_every_gap_at_once_and_costs_nothing() -> None:
    """One abort listing all gaps — fixing one and re-running to find the next is the expensive loop."""
    from signet_trainer.modal.entrypoint import _wan_refuse_on_gaps

    with pytest.raises(SystemExit) as excinfo:
        _wan_refuse_on_gaps(load_config(_EXAMPLE), mode="sample")
    message = str(excinfo.value)
    # ONE now. The list has shrunk twice on purpose: the musubi SHA gap closed when the
    # checkout was pinned, and the component gap closed when download_wan_weights landed and
    # the config declared the four ids. Only the MODE fence remains, because there is still no
    # Wan inference path. The count is asserted rather than ">= 1" because reporting EVERY gap
    # in one abort is the property, and a silently shrinking list is how it rots.
    assert "1 DECLARED gap(s)" in message, message
    assert "nothing was spent" in message


# ==================================================================================================
# (4) The batch line — honest about the multiplier it cannot supply
# ==================================================================================================


@pytest.mark.parametrize(
    ("extraction", "target_frames", "frame_sample", "expected"),
    [
        ("image", (), 1, 1),
        ("full", (), 1, 1),
        ("head", (1, 21), 1, 2),
        ("uniform", (45,), 2, 2),
        ("chunk", (21,), 1, None),
        ("slide", (21,), 1, None),
    ],
)
def test_clips_per_media_is_arithmetic_where_it_can_be_and_None_where_it_cannot(
    extraction: str, target_frames: tuple[int, ...], frame_sample: int, expected: int | None
) -> None:
    """``None`` for chunk/slide is the HONEST answer — both depend on video LENGTH, a Volume fact."""
    assert wan_clips_per_media(extraction, target_frames, frame_sample) == expected


def test_an_unknown_extraction_is_refused_rather_than_priced() -> None:
    """A seventh mode name would be priced as something musubi's schema rejects."""
    with pytest.raises(ValueError, match="unknown extraction"):
        wan_clips_per_media("tail", (21,), 1)


def test_the_batch_line_prices_the_real_example_and_names_what_it_cannot_read() -> None:
    """The Kaboom method, reported PER CORPUS — because the sources sit in different directories.

    The line used to end "5 clip instance(s) per media file across 3 source(s)", summing an IMAGE
    source and two VIDEO sources over DIFFERENT directories. Those addends count different files,
    so the operator's multiplication was wrong in both directions: with 100 stills and 10 videos
    the true clip-instance count is 100x1 + 10x2 + 10x2 = 140, while 5 x 110 = 550 (~4x over) and
    5 x 10 = 50 (3x under). The banner exists to be checkable by eye against a real corpus, and in
    that form it could not be.

    Now one multiplier per corpus: Images x1, Videos x4 — each multiplied by the file count of that
    directory, which is a number the operator can read off their own dataset. 100x1 + 10x4 = 140.
    """
    from signet_trainer.modal.entrypoint import _wan_batch_note

    line = _wan_batch_note(load_config(_EXAMPLE))
    assert "stills=image:1x1=1" in line
    assert "appearance=head:2x1=2" in line
    assert "motion=uniform:2x1=2" in line
    assert "/dataset/Kaboom/Images x1" in line
    assert "/dataset/Kaboom/Videos x4" in line, (
        "the two video sources over ONE directory must be summed together (2 + 2), while the image "
        "corpus stays separate"
    )
    assert "clip instance(s) per media file across" not in line, (
        "the cross-corpus total is back — it adds counts of different files"
    )
    # The two things an operator must not read off this line as if they were measured.
    assert "EPOCH-driven" in line and "--max_train_epochs 16" in line
    assert "training.max_steps=4000" in line and "NOT passed to the runner" in line
    assert "is not a measurement" in line


def test_a_corpus_dependent_source_makes_the_TOTAL_decline_to_exist() -> None:
    """None must PROPAGATE. A total that dropped the unknown source would silently under-count."""
    from signet_trainer.modal.cost import WanSourceView, format_wan_batch_line, wan_batch_estimate

    estimate = wan_batch_estimate(
        sources=[
            WanSourceView("a", "video", "head", (21,), 1, 1, "/dataset/K/Videos"),
            WanSourceView("b", "video", "chunk", (21,), 1, 3, "/dataset/K/Videos"),
        ],
        max_train_epochs=16,
        declared_max_steps=4000,
        est_hours=6.0,
    )
    assert estimate.instances_per_media_total is None
    assert "NOT SIZEABLE" in format_wan_batch_line(estimate)


def test_num_repeats_multiplies_the_instances() -> None:
    """On musubi ``num_repeats`` LENGTHENS the run — so it belongs in the priced count."""
    from signet_trainer.modal.cost import WanSourceView, wan_batch_estimate

    estimate = wan_batch_estimate(
        sources=[WanSourceView("a", "video", "head", (1, 21), 1, 3, "/dataset/K/Videos")],
        max_train_epochs=16,
        declared_max_steps=10,
        est_hours=1.0,
    )
    assert estimate.instances_per_media_total == 6


# ==================================================================================================
# (5) The gate itself — the approval pause still stops an unapproved dispatch
# ==================================================================================================


def test_a_wan_dispatch_without_approve_stops_at_the_approval_pause(monkeypatch, capsys) -> None:
    """MODL-02 on the new arm: no ``--approve``, non-interactive stdin -> abort, ZERO dispatch.

    Drives the REAL ``main`` body against the REAL example config. ``wan_train`` is replaced by a
    recorder so that even a regression that dispatched would spend nothing — and the recorder is
    what proves the abort, since an assertion on the exit code alone would pass for a main() that
    dispatched and then failed.
    """
    import builtins

    from signet_trainer.modal import entrypoint, fns

    calls: list[tuple] = []
    monkeypatch.setattr(
        fns, "wan_train", SimpleNamespace(with_options=lambda **_: SimpleNamespace(spawn=lambda **kw: calls.append(kw)))
    )
    monkeypatch.setattr(builtins, "input", lambda *_: (_ for _ in ()).throw(EOFError()))

    with pytest.raises(SystemExit) as excinfo:
        entrypoint.main.info.raw_f(config=str(_EXAMPLE), approve=False, mode="train")

    assert excinfo.value.code == 1
    assert calls == [], "a dispatch happened without approval — MODL-02 is absolute"
    out = capsys.readouterr().out
    # The gate ran in order: dry-run banner, then the wan batch line, then the cost line, then the
    # approval refusal. The batch line before the cost line is what lets the two be checked against
    # each other.
    assert out.index("wan config valid") < out.index("[signet-cost] wan batch:")
    assert out.index("[signet-cost] wan batch:") < out.index("[signet-cost] est $")
    assert "approval: DECLINED" in out


def test_a_wan_dispatch_WITH_approve_still_refuses_on_the_declared_gaps(monkeypatch, tmp_path) -> None:
    """Approval is not a bypass: a gap aborts AFTER the pause and BEFORE any spawn, at $0.

    ⚠ REWRITTEN 2026-08-12, and the reason is the point. This used to drive the SHIPPED config,
    which had two open gaps (unpinned musubi SHA, unresolvable components). Both are now closed —
    the checkout is pinned and download_wan_weights stages the four components the config declares —
    so the shipped config passes every gate and reaches `.spawn()`. That is the feature landing.

    It also means the shipped config can no longer prove this property, and quietly deleting the
    test would have removed the only check that `--approve` does not skip the gap refusal. So it now
    drives a config with a gap deliberately reopened (the components undeclared), which is the
    condition the property is about. The MVP closing a gap must not retire the guard that the gap
    was ever refused.
    """
    from signet_trainer.modal import entrypoint, fns

    raw = _EXAMPLE.read_text(encoding="utf-8")
    for line in ("  model_id:", "  vae_id:", "  text_encoder_id:", "  clip_id:"):
        raw = "\n".join(l for l in raw.split("\n") if not l.startswith(line))
    ungapped = tmp_path / "wan_undeclared.yaml"
    ungapped.write_text(raw, encoding="utf-8")

    calls: list[tuple] = []
    monkeypatch.setattr(
        fns, "wan_train", SimpleNamespace(with_options=lambda **_: SimpleNamespace(spawn=lambda **kw: calls.append(kw)))
    )
    with pytest.raises(SystemExit) as excinfo:
        entrypoint.main.info.raw_f(config=str(ungapped), approve=True, mode="train")

    assert "DECLARED gap(s)" in str(excinfo.value)
    assert "nothing was spent" in str(excinfo.value)
    assert calls == [], "a gap was reported and the dispatch happened anyway"


def test_the_wan_arm_lands_after_the_approval_gate_in_source_order() -> None:
    """MODL-02 as STRUCTURE for the new arm, on AST positions rather than text indices."""
    tree = ast.parse(_ENTRYPOINT.read_text(encoding="utf-8"))
    main = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    approvals = [
        (n.lineno, n.col_offset)
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_require_approval"
    ]
    assert approvals
    spawns = [
        (n.lineno, n.col_offset)
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "spawn"
        and "wan_train" in ast.unparse(n.func)
    ]
    assert spawns, "no wan_train .spawn( in main() — the arm is missing"
    assert min(spawns) > max(approvals), "the wan dispatch precedes the approval pause"


def test_the_stage_is_reachable_only_through_the_gate_and_declares_no_warm_gpu() -> None:
    """No second entry point, no warm container. Both are house invariants, checked on the new stage."""
    src = _FNS.read_text(encoding="utf-8")
    block = src[src.index("def wan_train("):]
    decorator = src[: src.index("def wan_train(")].rsplit("@app.function(", 1)[-1]
    assert "keep_warm" not in decorator and "min_containers" not in decorator
    assert "image=wan_musubi_image" in decorator
    assert "gpu=" in decorator
    assert "checkpoints_vol.commit()" in block, "commit-or-vanish (Pitfall 3) is missing"


def test_a_real_model_config_can_now_declare_all_four() -> None:
    """The gap the stub above used to stand in for, closed 2026-08-10 and pinned here.

    Two things had to change before a ``family: wan`` config could name its own weights, and both
    are the sort that fail SILENTLY if they regress: ``clip_id`` did not exist on ``ModelConfig``
    (Wan 2.1 is the first signet family with two text-side encoders), and ``vae_id`` was fenced to
    ``{"h3", "qwen_edit"}``, so declaring it under ``family: wan`` failed at config LOAD.

    Driven through a real ``ModelConfig`` rather than the SimpleNamespace stub, because
    ``extra="forbid"`` and ``_FAMILY_ONLY_MODEL_IDS`` are exactly what this asserts and a stub sees
    neither.
    """
    from signet_trainer.config.schema import ModelConfig

    cfg = ModelConfig(
        family="wan",
        model_id="wan/wan2.1_t2v_14B_bf16.safetensors",
        vae_id="wan/wan_2.1_vae.safetensors",
        text_encoder_id="wan/models_t5_umt5-xxl-enc-bf16.pth",
        clip_id="wan/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )
    components = wan_resolve_component_ids(
        cfg,
        schema_defaults={
            field: ModelConfig.model_fields[field].default
            for field in WAN_COMPONENT_CONFIG_FIELDS.values()
        },
    )
    assert components.clip.endswith("open-clip-xlm-roberta-large-vit-huge-14.pth")
    assert components.vae.endswith("wan_2.1_vae.safetensors")


def test_clip_id_is_fenced_to_wan_and_refused_elsewhere() -> None:
    """An unfenced knob is one a config can set while nothing reads it.

    ``clip_id`` has exactly one consumer (musubi's ``--clip``, wan only), so every other family must
    refuse it at config load rather than accept a value it will silently ignore — the same rule that
    put ``vae_id`` and ``pipeline_root_id`` in ``_FAMILY_ONLY_MODEL_IDS``.
    """
    from signet_trainer.config.schema import _FAMILY_ONLY_MODEL_IDS

    assert _FAMILY_ONLY_MODEL_IDS["clip_id"] == frozenset({"wan"})
    assert "wan" in _FAMILY_ONLY_MODEL_IDS["vae_id"]


# ==================================================================================================
# (6) The commit cadence — the audit finding that a timeout loses every epoch
# ==================================================================================================


def test_a_first_sighting_is_never_committed_only_a_settled_one_is(tmp_path) -> None:
    """A file that APPEARED is not a file that is WRITTEN.

    musubi saves an adapter every epoch straight into the mounted Volume, and a Volume commit is
    explicit — so committing once after the last subprocess returned meant a timeout at epoch 15 of
    16 landed NOTHING, with no `retries=` and no resume to recover from. Committing on first sight
    is the obvious fix and the wrong one: a half-flushed safetensors on the Volume under a
    finished-looking name is worse than a missing one, because the next round of the chain would
    warm-start from it.
    """
    from signet_trainer.modal.fns import wan_settled_adapters

    sizes: dict[str, int] = {}
    committed: set[str] = set()

    growing = tmp_path / "run-000001.safetensors"
    growing.write_bytes(b"x" * 100)
    assert wan_settled_adapters(tmp_path, sizes, committed) == [], (
        "committed on FIRST sighting — one poll cannot tell 'written' from 'still writing'"
    )

    growing.write_bytes(b"x" * 400)  # still growing
    assert wan_settled_adapters(tmp_path, sizes, committed) == []

    # size unchanged across two polls -> settled
    assert wan_settled_adapters(tmp_path, sizes, committed) == ["run-000001.safetensors"]


def test_a_settled_adapter_is_never_committed_twice(tmp_path) -> None:
    from signet_trainer.modal.fns import wan_settled_adapters

    sizes: dict[str, int] = {}
    committed: set[str] = set()
    (tmp_path / "a.safetensors").write_bytes(b"y" * 10)
    wan_settled_adapters(tmp_path, sizes, committed)
    landed = wan_settled_adapters(tmp_path, sizes, committed)
    assert landed == ["a.safetensors"]
    committed.update(landed)
    assert wan_settled_adapters(tmp_path, sizes, committed) == [], "committed the same adapter twice"


def test_a_zero_byte_adapter_is_never_settled(tmp_path) -> None:
    """A touched-but-empty file holds still at zero forever; size-equality alone would commit it."""
    from signet_trainer.modal.fns import wan_settled_adapters

    sizes: dict[str, int] = {}
    (tmp_path / "empty.safetensors").write_bytes(b"")
    for _ in range(3):
        assert wan_settled_adapters(tmp_path, sizes, set()) == []


def test_the_training_pass_commits_DURING_the_run_and_on_every_exit_path() -> None:
    """Structure, source-scanned: the stage cannot be executed without a GPU and musubi.

    Three properties, each of which was absent and each of which loses a metered round:
      * the training pass runs under Popen (so the Volume can be committed WHILE it runs);
      * the commit sits in a `finally` (so a timeout or CalledProcessError still lands what exists);
      * the poll interval is finite and named.
    """
    source = _FNS.read_text(encoding="utf-8")
    stage = source[source.index("def wan_train("):]
    stage = stage[: stage.index("\n@app.function") if "\n@app.function" in stage else len(stage)]
    assert "subprocess.Popen(" in stage, "the training pass is not run under Popen"
    assert "finally:" in stage, "the commit is not on every exit path"
    assert "WAN_ADAPTER_COMMIT_POLL_SECONDS" in stage
    assert "wan_settled_adapters(" in stage
    body = stage[stage.index("finally:"):]
    assert "checkpoints_vol.commit()" in body, "the finally block does not commit"
