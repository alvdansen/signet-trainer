"""VERIFIER-added coverage for the multi-source slice — the gaps the slice's own suite left.

``tests/test_musubi_toml_render.py`` proves the renderer reproduces Timothy's real runner config
(``docs/source-methods/musubi-wan21/wan21-dataset-config.toml``) byte-for-byte, and that proof was
independently reproduced. What it does NOT cover is the ground the feature sits on:

  * the oracle it asserts against is not in git (a CI landmine — the suite is green here and red on
    a fresh clone);
  * ``check_cache_collisions`` compares cache roots as STRINGS, so two spellings of one directory
    pass the one gate that exists to stop two extractions sharing a cache;
  * the ADDITIVE promise ("absence of ``sources:`` is the byte-identical path") is asserted in
    prose in the fixture header and nowhere in code;
  * ``SourceSpec`` is a type two lanes both wanted to declare, and a second declaration is silent
    corruption rather than a merge conflict;
  * the design's own worked geometry, 1280x720x21, is rejected by every family that exists — the
    ``wan`` family needs a THIRD dims branch, not a widened literal.

Each gap below is either a hard assertion (the ground still holds) or a ``strict`` xfail naming the
symbol that lands it (the ground does NOT hold, and the test flips loudly the moment it does).
Everything here is CPU-only, filesystem-read-only, and free.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from signet_trainer.config.sources import SourceSpec
from signet_trainer.runners.musubi_toml import check_cache_collisions, render_musubi_toml

_REPO = Path(__file__).resolve().parents[1]
_REAL_TOML = _REPO / "docs" / "source-methods" / "musubi-wan21" / "wan21-dataset-config.toml"
_FIXTURE = _REPO / "tests" / "fixtures" / "wan21_kaboom.example.yaml"


# ==================================================================================================
# The oracle's provenance
# ==================================================================================================


def test_the_acceptance_oracle_exists_on_disk() -> None:
    """Without this file the acceptance test does not fail — it ERRORS, which reads as infra noise."""
    assert _REAL_TOML.is_file(), (
        f"the acceptance oracle {_REAL_TOML} is missing; tests/test_musubi_toml_render.py cannot "
        f"prove anything without it"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "docs/source-methods/ is UNTRACKED (git status: '?? docs/source-methods/'). The acceptance "
        "test's oracle is therefore absent on a fresh clone and the suite goes red in CI for a "
        "reason unrelated to the code. Fix: `git add docs/source-methods/` in the commit that "
        "lands this slice. Remove this xfail in the same commit."
    ),
)
def test_the_acceptance_oracle_is_tracked_in_git() -> None:
    """The oracle is a TEST INPUT. An untracked test input is a test that only passes locally."""
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", _REAL_TOML.relative_to(_REPO).as_posix()],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{_REAL_TOML.relative_to(_REPO).as_posix()} is not tracked by git:\n{result.stderr.strip()}"
    )


# ==================================================================================================
# The cache gate — the one refusal that stands between two views and one corrupted cache
# ==================================================================================================


def test_explicit_and_derived_cache_roots_collide_correctly() -> None:
    """The mixed case DOES work: an explicit root equal to another's derived root is refused."""
    sources = [
        SourceSpec(id="a", kind="image", directory="/d", resolution=(64, 64), cache_root="/r/cache/b"),
        SourceSpec(id="b", kind="image", directory="/d", resolution=(64, 64)),
    ]
    assert check_cache_collisions(sources, data_root="/r") == [
        "cache collision: source 'b' and 'a' both resolve to '/r/cache/b'"
    ]


# FIXED 2026-08-09 — was xfail(strict). SourceSpec.resolve_cache_root now rstrips the EXPLICIT
# branch as well as the derived one, so the two spellings resolve to one string and the
# collision gate sees one directory. Kept as a live test rather than deleted: the failure it
# describes is silent corruption (two extractions latents into one cache, no warning), and this
# feature is the one that provokes it — hand-written near-identical cache paths off a single
# source directory is its defining move.
def test_trailing_slash_cache_roots_are_recognised_as_one_directory() -> None:
    sources = [
        SourceSpec(
            id="appearance",
            kind="video",
            directory="/d/V",
            resolution=(1280, 720),
            extraction="head",
            target_frames=[21],
            cache_root="/d/V/cache_1/",
        ),
        SourceSpec(
            id="motion",
            kind="video",
            directory="/d/V",
            resolution=(640, 352),
            extraction="uniform",
            target_frames=[45],
            frame_sample=2,
            cache_root="/d/V/cache_1",
        ),
    ]
    assert check_cache_collisions(sources, data_root="/r"), (
        "'/d/V/cache_1/' and '/d/V/cache_1' are ONE directory; the collision gate did not see it"
    )
    with pytest.raises(ValueError, match="cache collision"):
        render_musubi_toml(sources, data_root="/r", caption_extension=".txt")


# ==================================================================================================
# The ADDITIVE promise, asserted in code rather than in a fixture comment
# ==================================================================================================


def test_no_shipped_config_declares_sources_so_absence_is_the_identity_path() -> None:
    """The feature is opt-in BECAUSE every shipped config omits the key. Absence cannot regress."""
    offenders = [
        path.name
        for path in sorted((_REPO / "configs").glob("*.yaml"))
        if "sources" in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("data", {})
    ]
    assert offenders == [], (
        f"{offenders} declare data.sources; the byte-identical dry-run guarantee for existing "
        f"configs rests on the key being ABSENT everywhere it has not been opted into"
    )


def test_source_spec_is_declared_exactly_once() -> None:
    """Two lanes both wanted this type. A second declaration is silent divergence, not a conflict."""
    hits = [
        f"{path.relative_to(_REPO).as_posix()}:{n}"
        for path in sorted((_REPO / "src").rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.startswith("class SourceSpec")
    ]
    # FILE, not file:line. The earlier form pinned "sources.py:129" and broke the moment a
    # six-line comment was added above the class — a test that fails on a comment edit trains
    # people to edit the test rather than read it. The claim is "exactly one home", and the home
    # is the file.
    assert [h.split(":")[0] for h in hits] == ["src/signet_trainer/config/sources.py"], (
        f"SourceSpec is declared at {hits}; it must have exactly one home, and every consumer "
        f"(schema.py when `sources:` lands, and the runners) must IMPORT it rather than redeclare it"
    )


# ==================================================================================================
# The geometry blocker — why `wan` is not a one-word literal change
# ==================================================================================================


def test_the_fixtures_geometry_is_rejected_by_every_family_that_exists() -> None:
    """1280x720x21 fails BOTH axes of both existing laws, so `wan` needs its own dims branch.

    720 % 32 == 16 (LTX/H3 spatial step is 32; musubi's Wan step is 16) and 21 satisfies neither
    ``(F-1) % 8 == 0`` nor ``(F-5) % 17 == 0`` (musubi's law is ``F % 4 == 1``, which 21 meets).
    Landing ``"wan"`` in ``ModelConfig.family`` without a third arm in the ``training_dims``
    pre-screen produces a family that cannot be configured at all.
    """
    from signet_trainer.config import validators as v

    dims = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))["training_dims"]
    assert dims == [1280, 720, 21]
    with pytest.raises(ValueError, match=r"height 720"):
        v.validate_height(dims[1])
    with pytest.raises(ValueError, match=r"frame count 21"):
        v.validate_frames(dims[2])
    with pytest.raises(ValueError, match=r"frame count 21"):
        v.validate_h3_frames(dims[2])


def test_the_wan_family_tripwire_is_still_armed() -> None:
    """A pre-existing test uses "wan" as its example of a REJECTED family.

    ``tests/test_h3_config_schema.py::test_model_config_rejects_unknown_family`` asserts
    ``ModelConfig(family="wan")`` raises. Whoever adds ``"wan"`` to the literal must amend that
    test deliberately and say why — this assertion exists so the amendment cannot be accidental.
    """
    source = (_REPO / "tests" / "test_h3_config_schema.py").read_text(encoding="utf-8")
    assert 'ModelConfig(family="wan")' in source, (
        "the tripwire moved or was silently edited; confirm the family allowlist change was "
        "intentional and that an equivalent rejection case still covers unknown families"
    )
