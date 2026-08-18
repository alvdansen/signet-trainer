"""issue #19 item 3 (D-NOHARDCODE) — scripts/_h3_grid_serve.py and scripts/_h3_run_status.py must
resolve the checkpoints Volume name (and, for _h3_run_status.py, the App name) from the loaded
config's ``modal.checkpoints_volume_name`` / ``modal.app_name`` — never restate the historical
"signe-trainer-checkpoints" / "signe-trainer" literals as the --volume/--app-name default.

Both scripts are standalone (not a package) — loaded via importlib per the test_qa_overlay_h264 /
test_prep_segdataset standalone-script precedent. Pure CPU: no modal, no GPU, no network. Only the
config-resolution HELPERS (``load_eval_matrix``, ``load_recipe``) are exercised directly — `main()`
itself shells out to `modal` and is covered by the source-scan pins in test_watcher_hardening.py
instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_CONFIG = REPO_ROOT / "configs" / "h3_t2va_noref.example.yaml"


def _load(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_config(tmp_path: Path, **modal_overrides) -> Path:
    """A copy of the shipped h3 example config with `modal.*` fields overridden."""
    data = yaml.safe_load(_EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    data.setdefault("modal", {}).update(modal_overrides)
    out = tmp_path / "cfg.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


# ---- scripts/_h3_grid_serve.py — load_eval_matrix() ----------------------------------------------


def test_grid_serve_resolves_checkpoints_volume_name_from_config(tmp_path):
    grid = _load("_h3_grid_serve")
    cfg_path = _write_config(tmp_path, checkpoints_volume_name="my-custom-checkpoints")
    matrix = grid.load_eval_matrix([cfg_path])
    assert matrix["checkpoints_volume_name"] == "my-custom-checkpoints"


def test_grid_serve_checkpoints_volume_name_none_when_config_never_sets_it(tmp_path):
    # The shipped example config does not override modal.checkpoints_volume_name — main() must fall
    # through to the schema default in this case, never a bare None reaching `modal volume ls`.
    grid = _load("_h3_grid_serve")
    matrix = grid.load_eval_matrix([_EXAMPLE_CONFIG])
    assert matrix["checkpoints_volume_name"] is None


def test_grid_serve_warns_on_disagreeing_volume_names_and_keeps_the_first(tmp_path, capsys):
    grid = _load("_h3_grid_serve")
    first_dir = tmp_path / "a"
    first_dir.mkdir()
    first = _write_config(first_dir, checkpoints_volume_name="vol-a")
    second_dir = tmp_path / "b"
    second_dir.mkdir()
    second = _write_config(second_dir, checkpoints_volume_name="vol-b")
    matrix = grid.load_eval_matrix([first, second])
    assert matrix["checkpoints_volume_name"] == "vol-a"
    assert "WARNING" in capsys.readouterr().out


def test_grid_serve_volume_argparse_default_is_none_not_a_literal():
    # The CLI default must be None (resolved from config inside main()), never the retired
    # "signe-trainer-checkpoints" literal (D-NOHARDCODE) — pinned as structure, not prose.
    grid = _load("_h3_grid_serve")
    import ast

    src = (REPO_ROOT / "scripts" / "_h3_grid_serve.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    defaults = [
        kw.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and arg.value == "--volume"
        for kw in node.keywords
        if kw.arg == "default"
    ]
    assert defaults, "no --volume argparse default found — argument definition changed shape"
    assert all(isinstance(d, ast.Constant) and d.value is None for d in defaults), (
        f"--volume's argparse default must be None (resolved from config in main()), got: "
        f"{[ast.unparse(d) for d in defaults]}"
    )


# ---- scripts/_h3_run_status.py — load_recipe() ----------------------------------------------------


def test_run_status_recipe_carries_volume_and_app_name_via_schema(tmp_path):
    status = _load("_h3_run_status")
    cfg_path = _write_config(
        tmp_path, checkpoints_volume_name="schema-vol", app_name="schema-app",
    )
    recipe = status.load_recipe(cfg_path)
    assert recipe["config_source"] == "schema"
    assert recipe["checkpoints_volume_name"] == "schema-vol"
    assert recipe["app_name"] == "schema-app"


def test_run_status_recipe_falls_back_to_schema_default_when_config_omits_it(tmp_path):
    status = _load("_h3_run_status")
    recipe = status.load_recipe(_EXAMPLE_CONFIG)
    from signet_trainer.config.schema import ModalConfig

    assert recipe["checkpoints_volume_name"] == ModalConfig().checkpoints_volume_name
    assert recipe["app_name"] == ModalConfig().app_name


def test_run_status_recipe_carries_volume_and_app_name_via_raw_yaml_fallback(tmp_path):
    # Force the raw-yaml fallback branch: an unknown top-level field trips extra="forbid" and
    # load_config raises, so load_recipe must still surface the fields from the raw YAML.
    data = yaml.safe_load(_EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    data["bogus_field_that_does_not_exist"] = True
    data.setdefault("modal", {}).update(
        checkpoints_volume_name="fallback-vol", app_name="fallback-app",
    )
    cfg_path = tmp_path / "invalid_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    status = _load("_h3_run_status")
    recipe = status.load_recipe(cfg_path)
    assert recipe["config_source"].startswith("raw-yaml"), (
        f"expected the raw-yaml fallback to trigger, got config_source={recipe['config_source']!r}"
    )
    assert recipe["checkpoints_volume_name"] == "fallback-vol"
    assert recipe["app_name"] == "fallback-app"


@pytest.mark.parametrize("flag", ["--volume", "--app-name"])
def test_run_status_argparse_defaults_are_none_not_literals(flag):
    import ast

    src = (REPO_ROOT / "scripts" / "_h3_run_status.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    defaults = [
        kw.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and arg.value == flag
        for kw in node.keywords
        if kw.arg == "default"
    ]
    assert defaults, f"no {flag} argparse default found — argument definition changed shape"
    assert all(isinstance(d, ast.Constant) and d.value is None for d in defaults), (
        f"{flag}'s argparse default must be None (resolved from config in main()), got: "
        f"{[ast.unparse(d) for d in defaults]}"
    )
