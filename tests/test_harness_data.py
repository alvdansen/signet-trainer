"""Packaged harness data tests — the fence against dangling `.planning/harness/` defaults.

THE DEFECT THIS LOCKS OUT. `.planning/` is a working-repo planning surface that is deliberately not
shipped. Modules whose DEFAULT paths pointed into `.planning/harness/` therefore resolved only
inside the maintainer's checkout: a fresh `pip install` got defaults that pointed at a directory it
does not have. The fix moved the five SPEC/TEMPLATE files into ``signet_trainer.harness_data``
(package data, resolved via ``importlib.resources``). These tests keep it that way:

  1. every packaged spec RESOLVES and its loader accepts it;
  2. the package ships exactly the five specs (no live state smuggled in, none dropped);
  3. no runtime default anywhere in src/ or scripts/ points at a `.planning/harness/` SPEC again —
     while LIVE state (SESSION-STATE.json, DECISION-LOG.md, cards/) is still allowed to;
  4. ``harness_lint``'s duplicated resource literals still name the same file (import confinement
     forces it to hardcode them, so they need a drift fence);
  5. the packaged specs stay free of campaign/property references — they ship, so a leak here is a
     leak in every install.

Pure CPU — no modal, no GPU, no network.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from signet_trainer import harness_data, harness_lint
from signet_trainer.harness_state import load_house_spec, load_tier_taxonomy
from signet_trainer.prep.spec import load_mask_spec

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_SCRIPTS = _REPO_ROOT / "scripts"


# --------------------------------------------------------------------------- (1) it resolves


@pytest.mark.parametrize("name", harness_data.PACKAGED_SPECS)
def test_every_packaged_spec_resolves_and_is_readable(name: str) -> None:
    """``spec_path`` returns a real, non-empty file for every declared spec."""
    path = harness_data.spec_path(name)
    assert path.is_file(), f"{name} did not resolve to a real file: {path}"
    assert path.read_text(encoding="utf-8").strip(), f"{name} resolved to an EMPTY file: {path}"
    # spec_text must agree with the materialised file (the zip-safe read path).
    assert harness_data.spec_text(name) == path.read_text(encoding="utf-8")


def test_packaged_specs_pass_their_real_loaders() -> None:
    """Each spec is accepted by the fail-loud loader that consumes it in production."""
    mask = load_mask_spec(harness_data.spec_path(harness_data.MASK_SPEC))
    assert mask["hex"] and mask["coverage_bands"]
    seg = load_mask_spec(harness_data.spec_path(harness_data.MASK_SPEC_SEGMENTATION))
    assert seg["coverage_bands"]
    house = load_house_spec(harness_data.spec_path(harness_data.HOUSE_SPEC))
    assert house["verdict_spec"]["frame_count"]
    tax = load_tier_taxonomy(harness_data.spec_path(harness_data.TIER_TAXONOMY))
    assert tax["knobs"]
    template = json.loads(harness_data.spec_text(harness_data.SESSION_STATE_TEMPLATE))
    assert template["spend"] == [] and template["blankets"] == [], (
        "SESSION-STATE.template.json must ship EMPTY — it is a template, never a live ledger"
    )


def test_unknown_name_is_a_loud_keyerror() -> None:
    """A name outside the shipped set fails loudly, and the message names the live/packaged split."""
    with pytest.raises(KeyError, match="not packaged harness data"):
        harness_data.spec_path("SESSION-STATE.json")  # the LIVE ledger — deliberately not packaged


# --------------------------------------------------------------------------- (2) exact contents


def test_package_ships_exactly_the_five_specs_and_no_live_state() -> None:
    """The data dir holds the five SPEC/TEMPLATE files + ``__init__.py`` — nothing else.

    An extra file here is either dead weight in every wheel or, worse, LIVE per-project state
    (a ledger, a decision log, a campaign card) baked into the package.
    """
    data_dir = _SRC / "signet_trainer" / "harness_data"
    shipped = {p.name for p in data_dir.iterdir() if p.is_file() and p.name != "__init__.py"}
    assert shipped == set(harness_data.PACKAGED_SPECS), (
        f"packaged data drifted from the declared set: extra={sorted(shipped - set(harness_data.PACKAGED_SPECS))} "
        f"missing={sorted(set(harness_data.PACKAGED_SPECS) - shipped)}"
    )


_LIVE_STATE_NAMES = ("SESSION-STATE.json", "DECISION-LOG.md", "KNOWLEDGE.md", ".state.yaml")


def test_live_state_is_never_packaged() -> None:
    """LIVE per-project artifacts must not appear under the data package under any name."""
    data_dir = _SRC / "signet_trainer" / "harness_data"
    for path in data_dir.rglob("*"):
        for live in _LIVE_STATE_NAMES:
            assert not path.name.endswith(live), (
                f"{path.name} is LIVE per-project state and must stay in .planning/harness/, "
                f"never inside the shipped package"
            )


# --------------------------------------------------------------------------- (3) no regressions


# A `.planning/harness/...` string literal in shipped code. LIVE state may still be addressed this
# way (it is project-relative by design); a SPEC may not (it ships, so the default must resolve).
_PLANNING_LITERAL = re.compile(r"\.planning/harness/([A-Za-z0-9_.\-]+)")


def _shipped_python_files() -> list[Path]:
    files = [p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts]
    files += [p for p in _SCRIPTS.rglob("*.py") if "__pycache__" not in p.parts]
    return files


def test_no_shipped_module_defaults_to_a_planning_spec_path() -> None:
    """No src/ or scripts/ file may reference a PACKAGED spec through a `.planning/harness/` path.

    This is the regression fence for the whole defect class: the audit found five such defaults in
    src/ (schema.py, propagate.py, harness_state.py x5, harness_lint.py x2) plus five more in
    scripts/. Live-state references (SESSION-STATE.json, DECISION-LOG.md, cards/, KNOWLEDGE.md,
    COMPACTION-*, GATE-SPEC-*) stay legal — they are project-relative on purpose.
    """
    offenders: list[str] = []
    for path in _shipped_python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for hit in _PLANNING_LITERAL.finditer(line):
                if hit.group(1) in harness_data.PACKAGED_SPECS:
                    rel = path.relative_to(_REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "packaged specs must be resolved via signet_trainer.harness_data (importlib.resources), "
        "never a .planning/harness/ path — that path does not exist in an installed package:\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------- (4) lint duplication


def test_harness_lint_taxonomy_matches_harness_data() -> None:
    """``harness_lint``'s hardcoded resource literals still name the packaged taxonomy.

    ``test_harness_lint_is_import_confined`` forbids the lint from importing ``harness_data``, so it
    repeats the package name / filename / override env var as string literals. This is the fence
    that keeps that duplication from drifting.
    """
    assert harness_lint.HARNESS_DATA_PACKAGE == harness_data.PACKAGE
    assert harness_lint.TIER_TAXONOMY_NAME == harness_data.TIER_TAXONOMY
    assert harness_lint.HARNESS_DATA_OVERRIDE_ENV_VAR == harness_data.OVERRIDE_ENV_VAR
    assert harness_lint.tier_taxonomy_path().read_text(encoding="utf-8") == harness_data.spec_text(
        harness_data.TIER_TAXONOMY
    ), "the lint and the data package must resolve the SAME TIER-TAXONOMY.yaml bytes"


def test_override_env_var_redirects_and_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``SIGNET_HARNESS_DATA_DIR`` redirects resolution; a half-populated override fails loudly."""
    override = tmp_path / "specs"
    override.mkdir()
    (override / harness_data.MASK_SPEC).write_text("hex: '#000000'\n", encoding="utf-8")
    monkeypatch.setenv(harness_data.OVERRIDE_ENV_VAR, str(override))

    assert harness_data.spec_path(harness_data.MASK_SPEC) == override / harness_data.MASK_SPEC
    with pytest.raises(FileNotFoundError, match="does not contain"):
        harness_data.spec_path(harness_data.HOUSE_SPEC)  # present in the package, absent here


# --------------------------------------------------------------------------- (5) leak gate


# Campaign / client-property tokens. The packaged specs SHIP, so a reference here leaks into every
# install — the methodology is the point, the campaign it was used on is not.
#
# The token list is itself sensitive: client/property names must never live in a tracked file (they
# would be the leak this very test exists to prevent, and a repo-wide `grep -ilE '<names>'` gate
# would flag this module). So the list is sourced LOCAL-ONLY, never hardcoded:
#
#   * ``SIGNET_PROPERTY_TOKENS`` — comma-separated, for CI; or
#   * ``<repo>/.property-tokens`` — one token per line, gitignored (`#` comments allowed).
#
# With neither present the test SKIPS rather than passing vacuously — a green light from a scanner
# that had nothing to scan for is worse than no scanner.
_PROPERTY_TOKENS_ENV_VAR = "SIGNET_PROPERTY_TOKENS"
_PROPERTY_TOKENS_FILE = _REPO_ROOT / ".property-tokens"


def _property_tokens() -> list[str]:
    raw = os.environ.get(_PROPERTY_TOKENS_ENV_VAR, "")
    if raw.strip():
        return [t.strip().lower() for t in raw.split(",") if t.strip()]
    if _PROPERTY_TOKENS_FILE.is_file():
        return [
            line.strip().lower()
            for line in _PROPERTY_TOKENS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return []


@pytest.mark.parametrize("name", harness_data.PACKAGED_SPECS)
def test_packaged_specs_carry_no_property_references(name: str) -> None:
    tokens = _property_tokens()
    if not tokens:
        pytest.skip(
            f"no property-token list: set {_PROPERTY_TOKENS_ENV_VAR}=a,b,c or create "
            f"{_PROPERTY_TOKENS_FILE.name} (gitignored, one token per line)"
        )
    text = harness_data.spec_text(name).lower()
    hits = [token for token in tokens if token in text]
    assert not hits, f"{name} leaks campaign/property reference(s) {hits} into every install"


def test_property_token_scan_actually_works() -> None:
    """Positive control: the scan must be able to FIND a token, or its silence proves nothing."""
    tokens = _property_tokens()
    if not tokens:
        pytest.skip(f"no property-token list ({_PROPERTY_TOKENS_ENV_VAR} / .property-tokens)")
    probe = f"coverage band tuned on the {tokens[0]} campaign".lower()
    assert [t for t in tokens if t in probe] == [tokens[0]]
