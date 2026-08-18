"""Phase 09.1 — the documented install extra carries no name-squattable ltx PyPI names (AUDIT #8).

T-09.1-01-SC (Tampering): the `pip install -e ".[modal-runtime]"` path pulls third-party code
across the PyPI boundary. `ltx-core` / `ltx-trainer` are NOT published on any index — the real
GPU image git-installs them from github.com/Lightricks/LTX-2 pinned at LTX2_COMMIT_SHA (app.py).
A bare `ltx-core` name in the extra resolves to a name-squatted package by an unverified publisher
(a live supply-chain vector). This test asserts the extra contains no bare `ltx-core`/`ltx-trainer`
entry — only absent, or (if the pin route was chosen) a git+https URL carrying a 40-hex commit SHA.

CPU-only: parses pyproject.toml with tomllib, no import of the package under test.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

_SQUATTABLE = ("ltx-core", "ltx-trainer")
# PEP 508 requirement: the project name is the leading run of name chars before any
# version specifier / URL marker / whitespace / extras.
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_SHA40_RE = re.compile(r"[0-9a-fA-F]{40}")


def _dependency_lists() -> dict[str, list[str]]:
    """Every requirement list pyproject.toml declares: the unconditional core deps plus every
    `optional-dependencies` group (`modal-runtime`, `dev`, `h3-parity`, and any future one added).

    Issue #41 finding 1: the original guard read only `[modal-runtime]`. `[project].dependencies`
    is the list a contributor is MOST likely to edit (it is what `pip install -e .` and every CI
    job install unconditionally), and it went unscanned.
    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    lists = {"project.dependencies": list(data["project"]["dependencies"])}
    for group, deps in data["project"]["optional-dependencies"].items():
        lists[f"optional-dependencies.{group}"] = list(deps)
    return lists


def _modal_runtime_deps() -> list[str]:
    return _dependency_lists()["optional-dependencies.modal-runtime"]


def _req_name(dep: str) -> str:
    m = _NAME_RE.match(dep)
    return m.group(1).lower().replace("_", "-") if m else ""


def _assert_no_bare_squatted_name(list_name: str, deps: list[str]) -> None:
    for dep in deps:
        name = _req_name(dep)
        if name in _SQUATTABLE:
            # An ltx-core/ltx-trainer entry is ONLY allowed as a git+https URL pinned to a 40-hex
            # SHA (the audit-sanctioned pin route). A bare index name is forbidden.
            assert "git+" in dep and _SHA40_RE.search(dep), (
                f"{name!r} appears in [{list_name}] as a bare/index name ({dep!r}): "
                "this pulls a name-squatted package on a documented install path (AUDIT #8). "
                "Remove it (git-pinned in app.py) or pin it as a git+https URL with a 40-hex SHA."
            )


def test_no_bare_ltx_core_or_trainer_in_any_dependency_list():
    """AUDIT #8 / issue #41 finding 1: scan EVERY requirement list, not just `[modal-runtime]` —
    a bare `ltx-core` in `[project].dependencies`, `[dev]`, or `[h3-parity]` is exactly as live a
    supply-chain vector and none of those three were scanned before this test existed."""
    for list_name, deps in _dependency_lists().items():
        _assert_no_bare_squatted_name(list_name, deps)


def test_the_guard_is_live_not_a_rubber_stamp_on_a_synthetic_bare_name():
    """Non-vacuity companion (issue #41 proposed direction, step 5).

    The shipped tree has never once tripped the `if name in _SQUATTABLE` branch, so a passing
    suite is consistent with the classifier being broken. Prove it independently: a synthetic bare
    name must be REJECTED and the sanctioned git+https/40-hex-SHA pin must be ACCEPTED.
    """
    with pytest.raises(AssertionError, match="bare/index name"):
        _assert_no_bare_squatted_name("synthetic", ["ltx-core==0.1"])
    with pytest.raises(AssertionError, match="bare/index name"):
        _assert_no_bare_squatted_name("synthetic", ["ltx_trainer @ git+https://example.com/x"])
    # The sanctioned pin route must clear the same guard it is meant to satisfy.
    _assert_no_bare_squatted_name(
        "synthetic", [f"ltx-core @ git+https://github.com/Lightricks/LTX-2@{'a' * 40}"]
    )


def test_core_lora_deps_still_present():
    # The genuine PyPI deps the extra legitimately provides must remain.
    names = {_req_name(d) for d in _modal_runtime_deps()}
    for expected in ("peft", "accelerate", "diffusers"):
        assert expected in names, f"{expected!r} missing from [modal-runtime]"
