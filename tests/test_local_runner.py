"""Local-training BETA — CPU-only tests for the refusal gate, path checks and banner honesty.

Zero GPU / zero Modal / zero heavy imports: ``local/runner.py`` deliberately keeps its module
top stdlib-light so these behavioral checks run everywhere (the heavy path is exactly the part
the BETA/UNTESTED label disclaims — see the package docstring and issue #25).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from signet_trainer.config.load import load_config
from signet_trainer.local import BETA_BANNER, ISSUES_URL, ROADMAP_ISSUE
from signet_trainer.local.runner import (
    EXIT_NOT_APPROVED,
    EXIT_REFUSED,
    LocalPaths,
    plan_text,
    refusals,
    resolve_paths,
    run,
)

REPO = Path(__file__).resolve().parents[1]
LTX_EXAMPLE = REPO / "configs" / "ltx23_lora.example.yaml"
# A SHIPPED config the local-beta matrix refuses (ic_lora mode) — real file, real load path.
REFUSED_EXAMPLE = REPO / "configs" / "ltx23_ic_lora.example.yaml"


def _ltx_config():
    return load_config(LTX_EXAMPLE)


# ---- the banner is loud, honest, and points at the ticket queue --------------------------------


def test_banner_declares_beta_untested_and_links_issues():
    for token in ("BETA", "UNTESTED", ISSUES_URL, ROADMAP_ISSUE):
        assert token in BETA_BANNER, f"banner must carry {token!r}"


def test_run_prints_banner_before_anything_else(capsys):
    # Even a refused config sees the banner FIRST — the tag is unconditional.
    rc = run(str(REFUSED_EXAMPLE))
    out = capsys.readouterr().out
    assert rc == EXIT_REFUSED
    assert out.index("UNTESTED") < out.index("REFUSED")


# ---- the refusal gate: unsupported surface fails LOUD, never half-runs -------------------------


def test_refuses_h3_family_with_roadmap_pointer():
    cfg = _ltx_config().model_copy(deep=True)
    object.__setattr__(cfg.model, "family", "h3")  # bypass validators: refusals() is object-level
    blockers = refusals(cfg)
    assert any("family" in b and ROADMAP_ISSUE in b for b in blockers)


def test_refuses_unsupported_conditioning_modes():
    cfg = _ltx_config()
    for mode in ("ic_lora", "inpaint", "audio_to_video"):
        patched = cfg.model_copy(deep=True)
        object.__setattr__(patched.conditioning, "mode", mode)
        blockers = refusals(patched)
        assert any("conditioning.mode" in b for b in blockers), mode


def test_refuses_in_loop_sampling():
    cfg = _ltx_config().model_copy(deep=True)
    object.__setattr__(cfg.validation, "in_loop_sampling", True)
    assert any("in_loop_sampling" in b for b in refusals(cfg))


def test_supported_ltx_example_has_no_refusals():
    assert refusals(_ltx_config()) == []


def test_refuses_multi_frame_training_with_conditioning_items():
    # WR-04 parity: the Modal arm raises on sample-only conditioning_items in a training config;
    # the local gate must refuse the same shape (items would be SILENTLY ignored by training).
    cfg = _ltx_config().model_copy(deep=True)
    object.__setattr__(cfg.conditioning, "mode", "multi_frame")
    object.__setattr__(cfg.conditioning, "conditioning_items", [{"frame": 0, "strength": 1.0}])
    assert any("sample-only" in b and "WR-04" in b for b in refusals(cfg))


def test_failed_dryrun_gate_refuses_and_never_claims_passed(monkeypatch, capsys):
    # Parity-review BLOCKER regression: run_dryrun returns non-zero and NEVER raises; a discarded
    # rc printed 'PASSED' over a failed gate. The gate must be the rc check.
    import signet_trainer.dryrun.shapes as shapes

    monkeypatch.setattr(shapes, "run_dryrun", lambda cfg: 1)
    from signet_trainer.local.runner import EXIT_REFUSED as rc_refused
    from signet_trainer.local.runner import run as run_fn

    cfg = _ltx_config()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        w = Path(td)
        (w / cfg.model.model_id).parent.mkdir(parents=True, exist_ok=True)
        (w / cfg.model.model_id).touch()
        (w / cfg.model.text_encoder_id).mkdir(parents=True, exist_ok=True)
        data = w / "data"; data.mkdir()
        import yaml
        doc = yaml.safe_load(LTX_EXAMPLE.read_text(encoding="utf-8"))
        doc["data"]["preprocessed_data_root"] = str(data)
        patched_cfg = w / "cfg.yaml"
        patched_cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")
        rc = run_fn(str(patched_cfg), weights_root=str(w), output_root=td)
    out = capsys.readouterr().out
    assert rc == rc_refused
    assert "shape gate FAILED" in out
    assert "PASSED" not in out


# ---- path resolution: existence-checked BEFORE any load, with actionable messages --------------


def test_resolve_paths_requires_weights_root_for_relative_ids():
    with pytest.raises(FileNotFoundError, match="--weights-root"):
        resolve_paths(_ltx_config(), weights_root=None, output_root=".")


def test_resolve_paths_checks_existence_and_names_the_missing_piece(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.model_id"):
        resolve_paths(_ltx_config(), weights_root=str(tmp_path), output_root=".")


def test_resolve_paths_happy_path(tmp_path):
    cfg = _ltx_config()
    (tmp_path / cfg.model.model_id).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / cfg.model.model_id).touch()
    (tmp_path / cfg.model.text_encoder_id).mkdir(parents=True, exist_ok=True)
    data = tmp_path / "data"
    data.mkdir()
    patched = cfg.model_copy(deep=True)
    object.__setattr__(patched.data, "preprocessed_data_root", str(data))
    paths = resolve_paths(patched, weights_root=str(tmp_path), output_root=str(tmp_path))
    assert paths.checkpoint_path.exists()
    assert paths.output_dir == tmp_path / cfg.output_dir


# ---- the plan print carries its numerics as name: value (typed-state discipline) ---------------


def test_plan_text_carries_typed_slots(tmp_path):
    cfg = _ltx_config()
    paths = LocalPaths(tmp_path / "m", tmp_path / "t", tmp_path / "d", tmp_path / "o")
    text = plan_text(cfg, paths, "vram: test")
    for slot in ("max_steps:", "checkpoint_every:", "keep_checkpoints:", "blocks_to_swap:",
                 "rank", "wall-clock: UNKNOWN"):
        assert slot in text, slot


# ---- structural: the local package must never touch Modal --------------------------------------


def test_local_package_never_imports_modal():
    pkg = REPO / "src" / "signet_trainer" / "local"
    for py in pkg.glob("*.py"):
        src = py.read_text(encoding="utf-8")
        stripped = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert not re.search(r"^\s*(import modal|from modal)", stripped, re.M), py.name
        assert ".spawn(" not in stripped and ".remote(" not in stripped, py.name


def test_refused_run_never_reaches_torch(tmp_path, capsys, monkeypatch):
    # A refused config must exit BEFORE the heavy-import boundary: simulate torch being broken —
    # a refusal path that imports it would blow up instead of returning EXIT_REFUSED.
    import builtins

    real_import = builtins.__import__

    def _no_torch(name, *a, **k):
        if name == "torch":
            raise AssertionError("refusal path must not import torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    assert run(str(REFUSED_EXAMPLE)) == EXIT_REFUSED
