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
from pydantic import ValidationError

from signet_trainer.config.sources import SourceSpec
from signet_trainer.runners.musubi_toml import check_cache_collisions, render_musubi_toml

_REPO = Path(__file__).resolve().parents[1]
_REAL_TOML = _REPO / "docs" / "source-methods" / "musubi-wan21" / "wan21-dataset-config.toml"
#: MOVED in slice A from ``tests/fixtures/`` to ``configs/``, exactly as its own header said it
#: would once ``DataConfig.sources`` and ``ModelConfig.family: wan`` landed. It is now a real,
#: LOADABLE config rather than a fixture describing one, which is what lets the assertions below be
#: about the schema instead of about a YAML file's shape.
_FIXTURE = _REPO / "configs" / "wan21_kaboom.example.yaml"
#: The one config permitted to declare ``data.sources``. See the test that reads it.
_SOURCES_OPT_IN = {"wan21_kaboom.example.yaml"}


# ==================================================================================================
# The oracle's provenance
# ==================================================================================================


def test_the_acceptance_oracle_exists_on_disk() -> None:
    """Without this file the acceptance test does not fail — it ERRORS, which reads as infra noise."""
    assert _REAL_TOML.is_file(), (
        f"the acceptance oracle {_REAL_TOML} is missing; tests/test_musubi_toml_render.py cannot "
        f"prove anything without it"
    )


# FIXED 2026-08-09 — was xfail(strict). The oracle and train_kohya.py are now git-tracked under
# docs/source-methods/musubi-wan21/. Kept live rather than deleted: the acceptance test
# compares the renderer against this file, so an untracked oracle means the suite is green on
# the author's box and RED on a fresh clone — the failure mode that makes a byte-exact proof
# worthless to everyone but the person who wrote it.
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


def test_only_the_opt_in_config_declares_sources_so_absence_is_the_identity_path() -> None:
    """The feature is opt-in BECAUSE every other shipped config omits the key.

    ⚠ AMENDED, DELIBERATELY (slice A). This asserted ``offenders == []`` while no config could
    declare ``sources`` at all. Now one does — ``configs/wan21_kaboom.example.yaml``, which is the
    feature's own worked example and the reason the field exists. The CLAIM is unchanged and is if
    anything sharper: the byte-identical dry-run path for every PRE-EXISTING config is the ABSENCE
    of the key, so the allowlist is pinned by name and a 20th config cannot join it by accident.

    Not a weakening, and this is checkable rather than a matter of opinion: the opt-in config is on
    ``family: wan``, which no other config uses, and the 18 pre-existing configs' dry-run output is
    byte-identical across this slice (verified by re-running the gate on a clean ``git archive
    HEAD`` extraction and diffing).
    """
    declaring = {
        path.name
        for path in sorted((_REPO / "configs").glob("*.yaml"))
        if "sources" in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("data", {})
    }
    assert declaring == _SOURCES_OPT_IN, (
        f"configs declaring data.sources are {sorted(declaring)}, expected "
        f"{sorted(_SOURCES_OPT_IN)}; the byte-identical dry-run guarantee for pre-existing configs "
        f"rests on the key being ABSENT everywhere it has not been deliberately opted into"
    )


def test_every_other_config_stays_on_a_native_family() -> None:
    """The opt-in is fenced by FAMILY too, not only by the ``sources`` key.

    ``data.sources`` is refused at config load on ltx/h3/qwen_edit, so the two fences agree: a
    config that does not say ``family: wan`` could not carry sources even if someone added the key.
    """
    families = {
        path.name: (yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        .get("model", {})
        .get("family", "ltx")
        for path in sorted((_REPO / "configs").glob("*.yaml"))
    }
    wan_configs = {name for name, family in families.items() if family == "wan"}
    assert wan_configs == _SOURCES_OPT_IN, (
        f"configs on family 'wan' are {sorted(wan_configs)}, expected {sorted(_SOURCES_OPT_IN)}"
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
# The geometry blocker — why `wan` was not a one-word literal change
# ==================================================================================================


def test_the_examples_geometry_is_rejected_by_every_NATIVE_family() -> None:
    """1280x720x21 fails BOTH axes of both native laws — which is why `wan` got its own branch.

    720 % 32 == 16 (LTX/H3 spatial step is 32; musubi's Wan step is 16) and 21 satisfies neither
    ``(F-1) % 8 == 0`` nor ``(F-5) % 17 == 0`` (musubi's law is ``F % 4 == 1``, which 21 meets).
    Landing ``"wan"`` in ``ModelConfig.family`` without a third arm in the ``training_dims``
    pre-screen would have produced a family that could not be configured at all.

    Kept live rather than deleted now that the branch exists: it is the STANDING reason the branch
    exists, and it fails loudly if someone ever tries to collapse the three laws into one.
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
    # ...and is accepted by the wan law, on both axes. Without this half the test above proves only
    # that the geometry is unusual, not that it is USABLE.
    assert v.validate_wan_training_dims(tuple(dims)) == (1280, 720, 21)


def test_the_wan_family_tripwire_fired_and_was_answered() -> None:
    """The successor to ``test_the_wan_family_tripwire_is_still_armed``.

    The armed form asserted that ``tests/test_h3_config_schema.py`` still contained
    ``ModelConfig(family="wan")`` as its example of a REJECTED family, so that adding ``"wan"`` to
    the literal could not be an accident. Slice A added it. The amendment was deliberate, and this
    is the assertion that the tripwire's actual REQUIREMENT — "an equivalent rejection case still
    covers unknown families" — was honoured rather than deleted along with the example.
    """
    from signet_trainer.config.schema import ModelConfig

    source = (_REPO / "tests" / "test_h3_config_schema.py").read_text(encoding="utf-8")
    assert "def test_model_config_rejects_unknown_family" in source, (
        "the unknown-family rejection case was removed rather than amended; the discriminator must "
        "stay an allowlist"
    )
    # The requirement itself, asserted BEHAVIOURALLY rather than by sniffing source text. The armed
    # form had to read a file because the thing it guarded was a literal in another test; the
    # successor guards the schema, which can just be called. (A source-substring check would also
    # have been wrong here: the amended file legitimately contains ``ModelConfig(family="wan")``
    # again, in the test that asserts the family is now ACCEPTED.)
    with pytest.raises(ValidationError):
        ModelConfig(family="svd")
    assert [ModelConfig(family=f).family for f in ("ltx", "h3", "qwen_edit", "wan")] == [
        "ltx",
        "h3",
        "qwen_edit",
        "wan",
    ]
