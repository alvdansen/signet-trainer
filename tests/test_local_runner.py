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
    EXIT_CRASHED,
    EXIT_NOT_APPROVED,
    EXIT_REFUSED,
    LocalPaths,
    main,
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


def test_refuses_armed_checkpoint_expected_minutes():
    # Issue #30 finding #3: this deadline is Modal-calibrated -- an armed value deterministically
    # kills a healthy local run (no retry off-Modal) and must be a loud refusal, not a silent pass.
    cfg = _ltx_config().model_copy(deep=True)
    object.__setattr__(cfg.training, "checkpoint_expected_minutes", 45.0)
    blockers = refusals(cfg)
    assert any("checkpoint_expected_minutes" in b and "45.0" in b for b in blockers)


def test_unarmed_checkpoint_expected_minutes_has_no_refusal():
    # None (the default) is the byte-identical off-state -- must not be refused.
    cfg = _ltx_config()
    assert cfg.training.checkpoint_expected_minutes is None
    assert not any("checkpoint_expected_minutes" in b for b in refusals(cfg))


def test_refuses_backup_enabled():
    # Issue #30 finding #6: backup mirrors the Modal checkpoints Volume ONLY -- a local run's
    # checkpoints are never enumerated, so silently passing backup.enabled=true is the exact
    # silently-ignored-config-block class the multi_frame refusal already forbids.
    cfg = _ltx_config().model_copy(deep=True)
    object.__setattr__(cfg.backup, "enabled", True)
    blockers = refusals(cfg)
    assert any("backup.enabled" in b and "Modal checkpoints Volume" in b for b in blockers)


def test_backup_disabled_by_default_has_no_refusal():
    cfg = _ltx_config()
    assert cfg.backup.enabled is False
    assert not any("backup" in b for b in refusals(cfg))


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


def test_plan_text_prints_checkpoint_watchdog_and_backup_knobs(tmp_path):
    # Issue #30 findings #3 / #6: plan_text claims to print every load-bearing value -- these two
    # were previously silently omitted entirely.
    cfg = _ltx_config()
    paths = LocalPaths(tmp_path / "m", tmp_path / "t", tmp_path / "d", tmp_path / "o")
    text = plan_text(cfg, paths, "vram: test")
    assert "checkpoint_expected_minutes:" in text
    assert "backup.enabled:" in text


def _stage_runnable_config(tmp_path: Path) -> tuple[str, str]:
    """Write weights + a data dir for the LTX example config under ``tmp_path``.

    Returns ``(config_path, weights_root)`` -- both under ``tmp_path`` so ``output_root=tmp_path``
    works too. Shared by the ``--dry-run-only`` and post-preflight-crash tests below.
    """
    import yaml

    cfg = _ltx_config()
    (tmp_path / cfg.model.model_id).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / cfg.model.model_id).touch()
    (tmp_path / cfg.model.text_encoder_id).mkdir(parents=True, exist_ok=True)
    data = tmp_path / "data"
    data.mkdir()
    doc = yaml.safe_load(LTX_EXAMPLE.read_text(encoding="utf-8"))
    doc["data"]["preprocessed_data_root"] = str(data)
    patched_cfg = tmp_path / "cfg.yaml"
    patched_cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(patched_cfg), str(tmp_path)


# ---- --dry-run-only is the FREE preview: no GPU, no training deps required (finding #2) --------


def test_dry_run_only_never_probes_cuda(tmp_path, monkeypatch, capsys):
    # Issue #30 finding #2: --dry-run-only is documented (README + --help) as FREE -- refusals +
    # shape gate + plan print, nothing else. Before the fix, `torch.cuda.is_available()` was
    # probed BEFORE this check, so a laptop (or a GPU box without a CUDA context) got a FALSE
    # "REFUSED -- torch.cuda.is_available() is False" and never saw the plan. NOTE: the dry-run
    # shape gate itself (signet_trainer.dryrun.shapes) legitimately imports torch for CPU shape
    # math -- torch is a base dep repo-wide -- so this pins the actual defect (a CUDA probe, not
    # torch import) rather than the broader, wrong claim that dry-run-only needs zero torch.
    import torch

    def _no_cuda_probe(*a, **k):
        raise AssertionError("--dry-run-only must not probe torch.cuda")

    monkeypatch.setattr(torch.cuda, "is_available", _no_cuda_probe)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _no_cuda_probe)

    config_path, weights_root = _stage_runnable_config(tmp_path)
    rc = run(config_path, weights_root=weights_root, output_root=str(tmp_path), dry_run_only=True)
    out = capsys.readouterr().out

    assert rc == 0, "the FREE preview must succeed without ever probing CUDA"
    assert "not probed (--dry-run-only" in out
    assert "PLAN (nothing has run yet)" in out
    assert "--dry-run-only: stopping before the approval gate" in out
    assert "REFUSED" not in out


# ---- main() installs the process's only logging handler (finding #1) ---------------------------


def _reset_root_logger():
    import logging

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    for h in saved_handlers:
        root.removeHandler(h)
    return root, saved_handlers, saved_level


def _restore_root_logger(root, saved_handlers, saved_level):
    for h in root.handlers[:]:
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


def test_main_installs_a_logging_handler_that_carries_library_info_lines(monkeypatch, capsys):
    # Issue #30 finding #1: before the fix, `logging.getLogger().handlers == []`, so every
    # logger.info in train/loop.py (the VRAM gauge) and train/checkpoint.py (saved/resumed/pruned)
    # was silently dropped for the whole run. main() must install a handler that actually surfaces
    # them.
    monkeypatch.delenv("SIGNET_LOG_LEVEL", raising=False)
    root, saved_handlers, saved_level = _reset_root_logger()
    try:
        rc = main(["--config", str(REFUSED_EXAMPLE)])
        assert rc == EXIT_REFUSED
        assert root.handlers, "main() must install a logging handler"

        import signet_trainer.train.checkpoint as ck

        ck.logger.info("Saved checkpoint: %s", "checkpoint-step-00200-loss-0.1234")
        out = capsys.readouterr().out
        assert "Saved checkpoint: checkpoint-step-00200-loss-0.1234" in out
    finally:
        _restore_root_logger(root, saved_handlers, saved_level)


def test_main_honors_signet_log_level_env_override(monkeypatch):
    import logging

    monkeypatch.setenv("SIGNET_LOG_LEVEL", "DEBUG")
    root, saved_handlers, saved_level = _reset_root_logger()
    try:
        main(["--config", str(REFUSED_EXAMPLE)])
        assert root.level == logging.DEBUG
    finally:
        _restore_root_logger(root, saved_handlers, saved_level)


# ---- FileNotFoundError: REFUSED only pre-approval, a propagated crash after (finding #4) --------


def test_main_labels_pre_approval_filenotfound_as_refused_not_crashed():
    # REFUSED_EXAMPLE's ic_lora conditioning.mode is refused before any path is even touched --
    # a baseline that the pre-approval path still gets the friendly line + EXIT_REFUSED.
    rc = run(str(REFUSED_EXAMPLE))
    assert rc == EXIT_REFUSED


def test_run_reraises_filenotfound_once_preflight_is_done(tmp_path, monkeypatch):
    # Issue #30 finding #4 regression: drive run() all the way past the approval gate (approve=
    # True, faked CUDA + deps), then make the first post-approval loader raise FileNotFoundError.
    # Before the fix this was caught by the SAME handler as a bad --weights-root and relabelled
    # "REFUSED" (exit 2) with the traceback swallowed -- indistinguishable from "your config is
    # unsupported, nothing was loaded" even though 22B+Gemma had already loaded.
    import sys
    import types

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (80 * 2**30, 80 * 2**30))
    for dep in ("bitsandbytes", "ltx_trainer"):
        monkeypatch.setitem(sys.modules, dep, types.ModuleType(dep))

    import signet_trainer.models.loader as loader_mod

    def _boom(*a, **k):
        raise FileNotFoundError("dataset .pt vanished 6h into the run")

    monkeypatch.setattr(loader_mod, "load_ltxv_components", _boom)

    config_path, weights_root = _stage_runnable_config(tmp_path)
    with pytest.raises(FileNotFoundError, match="dataset .pt vanished"):
        run(config_path, weights_root=weights_root, output_root=str(tmp_path), approve=True)


def test_main_turns_post_preflight_filenotfound_into_exit_crashed_with_traceback(
    monkeypatch, capsys
):
    # main()'s own boundary: it must NOT relabel a FileNotFoundError re-raised by run() as
    # EXIT_REFUSED -- it prints the traceback (the bug report the beta asks for) and returns the
    # distinct EXIT_CRASHED code.
    import signet_trainer.local.runner as runner_mod

    def _fake_run(*a, **k):
        raise FileNotFoundError("simulated post-approval crash: dataset .pt vanished")

    monkeypatch.setattr(runner_mod, "run", _fake_run)
    rc = runner_mod.main(["--config", "whatever.yaml"])
    err = capsys.readouterr().err

    assert rc == EXIT_CRASHED
    assert rc != EXIT_REFUSED
    assert "simulated post-approval crash" in err
    assert "Traceback" in err, "the traceback must be printed, not swallowed"


# ---- structural: the local package must never touch Modal --------------------------------------


def test_local_package_never_imports_modal():
    pkg = REPO / "src" / "signet_trainer" / "local"
    for py in pkg.glob("*.py"):
        src = py.read_text(encoding="utf-8")
        stripped = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert not re.search(r"^\s*(import modal|from modal)", stripped, re.M), py.name
        assert ".spawn(" not in stripped and ".remote(" not in stripped, py.name


# ---- AUDIT #34 direction 2 belt-and-braces: gate_adapter.lora_dropout must match config ---------


def test_run_asserts_gate_adapter_dropout_matches_config_before_reuse():
    """Mirrors the Modal ``train()`` assertion in ``modal/fns.py``: on the Open-Q1 default path
    ``run()`` reuses ``gate_adapter`` (the gate's check #5 adapter) as the training model without
    re-injecting it, so a future drift between the gate's build and this reuse site must abort
    BEFORE the training loop starts rather than silently train at the wrong dropout — the exact
    #34 defect, guarded against recurring.
    """
    import inspect

    from signet_trainer.local import runner as runner_mod

    src = inspect.getsource(runner_mod.run)
    stripped = re.sub(r'"""(?:.|\n)*?"""', "", src)
    stripped = re.sub(r"#.*", "", stripped)

    assert "gate_adapter" in stripped and "lora_dropout" in stripped and "config.lora.dropout" in stripped
    assert re.search(
        r"lora_dropout(?:.|\n)*?!=(?:.|\n)*?config\.lora\.dropout"
        r"|config\.lora\.dropout(?:.|\n)*?!=(?:.|\n)*?lora_dropout",
        stripped,
    ), "run() must compare gate_adapter's lora_dropout against config.lora.dropout"
    dropout_idx = stripped.index("lora_dropout")
    reuse_idx = stripped.index("model = gate_adapter")
    assert dropout_idx < reuse_idx, (
        "the dropout consistency assertion must run BEFORE `model = gate_adapter` reuses the "
        "adapter, not after"
    )
    assert "RuntimeError" in stripped[dropout_idx : reuse_idx + 50]


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
