"""The musubi dataset-TOML renderer, proved against Timothy's REAL runner config.

THE ACCEPTANCE TEST is :func:`test_renders_the_kaboom_config_equivalently`: build the source list
from the signet-side YAML and assert the rendered TOML is equivalent to
``docs/source-methods/musubi-wan21/wan21-dataset-config.toml`` — the file that was actually baked
into a Modal image (``train_kohya.py:40-43``) and read by all three musubi stages (``:109``,
``:121``, ``:140``).

Equivalence is asserted in BOTH senses, semantic FIRST and byte second, and the order matters:

  * a byte-only assertion fails uselessly on a key reordering, reporting a diff of whitespace when
    the real question is whether the two files MEAN the same thing;
  * a semantic-only assertion lets formatting drift, so an operator diffing the rendered artifact
    against the reference reads noise instead of the one line that changed.

Everything here is CPU-only, filesystem-read-only, and free.
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
import pytest

try:  # 3.11+ stdlib
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - the bench env is 3.10, BELOW pyproject
    # requires-python >=3.11. A hard ImportError here is a COLLECTION ERROR, which reads as
    # a broken suite rather than an out-of-contract interpreter; tests/conftest.py
    # requires_live_harness sets the house precedent of skipping WITH A REASON instead.
    tomllib = pytest.importorskip(
        "tomli",
        reason=(
            "tomllib is stdlib only from 3.11 and this interpreter is older than the "
            "project floor (pyproject requires-python >=3.11); install tomli or run the "
            "TOML tests on a compliant interpreter"
        ),
    )
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from signet_trainer.config.load import load_config_from_text
from signet_trainer.config.sources import ExtractionMode, SourceSpec
from signet_trainer.runners.musubi_toml import (
    MUSUBI_DATASET_KEYS,
    check_cache_collisions,
    musubi_resolution_warnings,
    parse_musubi_toml,
    render_control_directories,
    render_from_config,
    render_musubi_toml,
)

_REPO = Path(__file__).resolve().parents[1]
#: Timothy's REAL musubi config — the oracle. Not a fixture written for this test.
_REAL_TOML = _REPO / "docs" / "source-methods" / "musubi-wan21" / "wan21-dataset-config.toml"
#: The signet-side expression of the same three-dataset method. MOVED in slice A from
#: ``tests/fixtures/`` to ``configs/`` — with ``DataConfig.sources`` and ``ModelConfig.family: wan``
#: landed it is a real, loadable run config, which is what its own header said would happen.
#: The oracle-equivalence fixture. NOT configs/wan21_kaboom.example.yaml — that config now
#: carries signet's /dataset mount, while the production TOML this file diffs against carries
#: train_kohya.py's /datasets. Byte-equivalence is a property of the RENDERER; making the
#: shipped config carry a foreign mount point to keep this test green was the tail wagging the
#: dog, and it put a config in configs/ that could not dispatch.
_FIXTURE = _REPO / "tests" / "fixtures" / "wan21_kaboom.oracle.yaml"


def _fixture_config():
    """Return the example as a VALIDATED ``SignetConfig``.

    This helper used to be ``yaml.safe_load(...)["data"]`` with a note saying it would become
    ``load_config_from_text(...)`` once ``DataConfig.sources`` landed. It has. Going through the
    real loader is the substantive upgrade in this file: every assertion below is now made about
    objects the schema BUILT and VALIDATED, so it exercises the ``SourceSpec`` coercion, the
    cross-source cache-collision gate and the wan family arm — rather than about a dict that merely
    happened to have the right keys.
    """
    return load_config_from_text(_FIXTURE.read_text(encoding="utf-8"))


def _fixture_data():
    return _fixture_config().data


def _fixture_sources() -> list[SourceSpec]:
    return list(_fixture_data().sources)


def _render_fixture() -> str:
    data = _fixture_data()
    return render_musubi_toml(
        _fixture_sources(),
        data_root=data.preprocessed_data_root,
        caption_extension=data.caption_extension,
        enable_bucket=data.enable_bucket,
        bucket_no_upscale=data.bucket_no_upscale,
    )


# ==================================================================================================
# THE ACCEPTANCE TEST
# ==================================================================================================


def test_renders_the_kaboom_config_equivalently() -> None:
    """signet YAML -> musubi TOML == the real file, semantically and then byte-for-byte."""
    rendered = _render_fixture()
    real = _REAL_TOML.read_text(encoding="utf-8")

    assert tomllib.loads(rendered) == tomllib.loads(real), (
        "rendered TOML does not PARSE to the same structure as "
        f"{_REAL_TOML.relative_to(_REPO).as_posix()}:\n"
        + "\n".join(
            difflib.unified_diff(
                real.splitlines(), rendered.splitlines(), "real", "rendered", lineterm=""
            )
        )
    )
    assert rendered == real, (
        "rendered TOML parses identically but differs BYTE-wise from "
        f"{_REAL_TOML.relative_to(_REPO).as_posix()} — the artifact an operator diffs round over "
        "round must be stable:\n"
        + "\n".join(
            difflib.unified_diff(
                real.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                "real",
                "rendered",
                lineterm="",
            )
        )
    )


def test_the_same_video_directory_appears_twice_with_different_caches() -> None:
    """The FEATURE, asserted directly on the oracle rather than left implicit in a byte compare."""
    blocks = tomllib.loads(_render_fixture())["datasets"]
    video_blocks = [b for b in blocks if "video_directory" in b]
    assert len({b["video_directory"] for b in video_blocks}) == 1, (
        "the two video views must read ONE directory — that is the resolution-vs-extent trade"
    )
    assert len({b["cache_directory"] for b in video_blocks}) == len(video_blocks), (
        "each view must own its cache directory; the cache dir IS the extraction's identity"
    )
    resolutions = [tuple(b["resolution"]) for b in video_blocks]
    extractions = [b["frame_extraction"] for b in video_blocks]
    assert resolutions == [(1280, 720), (640, 360)] and extractions == ["head", "uniform"]


def test_the_fixture_prices_the_largest_declared_view() -> None:
    """``training_dims`` is the most expensive view the run assembles, not a fourth geometry.

    Slice A promoted this from a property of the example to a LAW of the family: the wan arm of
    ``SignetConfig._cross_field_checks`` refuses any other value. Both halves are asserted — the
    example still satisfies it, and the schema still enforces it.
    """
    config = _fixture_config()
    views = [
        (s.resolution[0], s.resolution[1], max(s.target_frames, default=1))
        for s in _fixture_sources()
    ]
    largest = max(views, key=lambda v: v[0] * v[1] * v[2])
    assert list(config.training_dims) == list(largest), (
        f"training_dims {list(config.training_dims)} is not the largest declared view "
        f"{list(largest)} (views: {views}) — the gate would price something this run never "
        f"assembles"
    )

    raw = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))
    raw["training_dims"] = [640, 352, 45]  # the `motion` view: legal wan geometry, but not largest
    with pytest.raises(ValidationError, match="largest view declared in data.sources"):
        load_config_from_text(yaml.safe_dump(raw))


# ==================================================================================================
# Determinism
# ==================================================================================================


def test_render_is_deterministic() -> None:
    """100 renders of one input produce exactly one distinct output."""
    outputs = {_render_fixture() for _ in range(100)}
    assert len(outputs) == 1, f"renderer produced {len(outputs)} distinct outputs for one input"


def test_explicit_cache_roots_make_data_root_inert() -> None:
    """A source that NAMES its cache never reads ``data_root`` — proved, not assumed."""
    sources = _fixture_sources()
    caption = _fixture_data().caption_extension
    a = render_musubi_toml(sources, data_root="/one", caption_extension=caption)
    b = render_musubi_toml(sources, data_root="/completely/other", caption_extension=caption)
    assert a == b


def test_an_unnamed_cache_root_is_derived_from_the_id() -> None:
    """``cache_root: None`` mints ``<data_root>/cache/<id>`` — derived, never guessed."""
    source = SourceSpec(
        id="motion",
        kind="video",
        directory="/dataset/videos",
        resolution=(640, 352),
        extraction=ExtractionMode.UNIFORM,
        target_frames=[45],
        frame_sample=2,
    )
    assert source.resolve_cache_root("/dataset/.precomputed_wan_r1") == (
        "/dataset/.precomputed_wan_r1/cache/motion"
    )
    assert source.resolve_cache_root("/dataset/.precomputed_wan_r1/") == (
        "/dataset/.precomputed_wan_r1/cache/motion"
    )


# ==================================================================================================
# Refusals — every one of these is a config that would otherwise describe a run nobody performed
# ==================================================================================================


def test_cache_collision_is_refused_before_any_toml_exists() -> None:
    shared = [
        SourceSpec(id="a", kind="image", directory="/d", resolution=(64, 64), cache_root="/c/same"),
        SourceSpec(id="b", kind="image", directory="/d", resolution=(64, 64), cache_root="/c/same"),
    ]
    assert check_cache_collisions(shared, data_root="/root") == [
        "cache collision: source 'b' and 'a' both resolve to '/c/same'"
    ]
    with pytest.raises(ValueError, match="cache collision: source 'b' and 'a'"):
        render_musubi_toml(shared, data_root="/root", caption_extension=".txt")


def test_a_derived_collision_is_caught_too() -> None:
    """Two sources cannot collide by OMISSION either — the derivation is id-keyed, and ids differ."""
    sources = [
        SourceSpec(id="x", kind="image", directory="/d", resolution=(64, 64)),
        SourceSpec(id="y", kind="image", directory="/d", resolution=(64, 64)),
    ]
    assert check_cache_collisions(sources, data_root="/root") == []
    blocks = tomllib.loads(
        render_musubi_toml(sources, data_root="/root", caption_extension=".txt")
    )["datasets"]
    assert [b["cache_directory"] for b in blocks] == ["/root/cache/x", "/root/cache/y"]


def test_an_empty_source_list_is_refused() -> None:
    with pytest.raises(ValueError, match="no sources"):
        render_musubi_toml([], data_root="/root", caption_extension=".txt")


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(id="x", kind="image", directory="/d", resolution=(64, 64), extraction="head"),
            "kind 'image' requires extraction: image, got 'head'",
        ),
        (
            dict(
                id="x",
                kind="video",
                directory="/d",
                resolution=(64, 64),
                extraction="head",
                target_frames=[30],
            ),
            "target_frames [30] are not N*4+1",
        ),
        (
            dict(
                id="x",
                kind="video",
                directory="/d",
                resolution=(64, 64),
                extraction="uniform",
                target_frames=[45],
                frame_sample=1,
            ),
            "rewritten to 'head' by musubi",
        ),
        (
            dict(
                id="x",
                kind="video",
                directory="/d",
                resolution=(64, 64),
                extraction="full",
                target_frames=[45],
            ),
            "ignored entirely by musubi",
        ),
        (
            dict(
                id="x",
                kind="video",
                directory="/d",
                resolution=(64, 64),
                extraction="head",
                target_frames=[21],
                frame_stride=3,
            ),
            "read ONLY by slide",
        ),
        (
            dict(id="x", kind="video", directory="C:\\d", resolution=(64, 64), extraction="full"),
            "contains a backslash",
        ),
    ],
)
def test_musubi_silent_repairs_are_refused(kwargs: dict, expected: str) -> None:
    """musubi warns, at most, and trains something else. signet refuses at config load instead."""
    with pytest.raises(ValidationError, match=r"(?s)" + re.escape(expected)):
        SourceSpec(**kwargs)


def test_the_floored_value_is_named_in_the_message() -> None:
    """An actionable refusal: it prints the number musubi WOULD have used."""
    with pytest.raises(ValidationError) as excinfo:
        SourceSpec(
            id="x",
            kind="video",
            directory="/d",
            resolution=(64, 64),
            extraction="head",
            target_frames=[30, 48],
        )
    assert "[29, 45]" in str(excinfo.value)


def test_the_resolution_floor_warns_with_the_real_trained_size() -> None:
    """The ONE repair that is not refused — refusing it would refuse the method's own config."""
    warnings = musubi_resolution_warnings(_fixture_sources())
    assert warnings == [
        "source 'motion': resolution [640, 360] is not a multiple of 16 - musubi will train it at "
        "640x352. Accept the crop, or declare [640, 352]."
    ]
    assert _render_fixture().count("[640, 360]") == 1, "the warning must not rewrite the config"


# ==================================================================================================
# The vocabulary
# ==================================================================================================


_MODE_EXPECTATIONS: dict[str, tuple[set[str], set[str]]] = {
    # mode -> (keys that MUST be present, keys that MUST be absent). Written out by hand rather
    # than imported from the renderer's own table, which would make the test assert nothing.
    "head": ({"target_frames"}, {"frame_sample", "frame_stride", "max_frames"}),
    "chunk": ({"target_frames"}, {"frame_sample", "frame_stride", "max_frames"}),
    "slide": ({"target_frames", "frame_stride"}, {"frame_sample", "max_frames"}),
    "uniform": ({"target_frames", "frame_sample"}, {"frame_stride", "max_frames"}),
    "full": ({"max_frames"}, {"target_frames", "frame_sample", "frame_stride"}),
}


def _source_for_mode(mode: str) -> SourceSpec:
    kwargs: dict = dict(
        id=mode, kind="video", directory="/d/videos", resolution=(640, 352), extraction=mode
    )
    if mode != "full":
        kwargs["target_frames"] = [21]
    if mode == "uniform":
        kwargs["frame_sample"] = 2
    if mode == "slide":
        kwargs["frame_stride"] = 5
    return SourceSpec(**kwargs)


@pytest.mark.parametrize("mode", sorted(_MODE_EXPECTATIONS))
def test_every_extraction_mode_renders_only_the_keys_it_reads(mode: str) -> None:
    source = _source_for_mode(mode)
    block = tomllib.loads(
        render_musubi_toml([source], data_root="/root", caption_extension=".txt")
    )["datasets"][0]
    present, absent = _MODE_EXPECTATIONS[mode]
    assert block["frame_extraction"] == mode
    assert present <= set(block), f"{mode}: missing {sorted(present - set(block))}"
    assert not (absent & set(block)), (
        f"{mode}: emitted {sorted(absent & set(block))}, which musubi ignores under this mode"
    )


@pytest.mark.parametrize("mode", sorted(_MODE_EXPECTATIONS))
def test_every_extraction_mode_round_trips_through_signet(mode: str) -> None:
    """TOML -> signet -> TOML is byte-identical for every mode in the vocabulary."""
    first = render_musubi_toml([_source_for_mode(mode)], data_root="/root", caption_extension=".txt")
    parsed = parse_musubi_toml(first)
    second = render_musubi_toml(
        parsed.sources,
        data_root="/root",
        caption_extension=parsed.caption_extension,
        enable_bucket=parsed.enable_bucket,
        bucket_no_upscale=parsed.bucket_no_upscale,
    )
    assert second == first


def test_the_real_toml_round_trips_through_signet() -> None:
    """The oracle itself: real TOML -> SourceSpecs -> TOML, byte-for-byte unchanged."""
    real = _REAL_TOML.read_text(encoding="utf-8")
    parsed = parse_musubi_toml(real)
    assert [s.id for s in parsed.sources] == ["cache", "cache_1", "cache_2"], (
        "ids are PROVENANCE-ONLY on the parse direction (musubi has no id concept) and are derived "
        "from the cache-directory basename"
    )
    assert [s.extraction for s in parsed.sources] == [
        ExtractionMode.IMAGE,
        ExtractionMode.HEAD,
        ExtractionMode.UNIFORM,
    ]
    re_rendered = render_musubi_toml(
        parsed.sources,
        data_root="/unused-every-cache-root-is-explicit",
        caption_extension=parsed.caption_extension,
        enable_bucket=parsed.enable_bucket,
        bucket_no_upscale=parsed.bucket_no_upscale,
    )
    assert re_rendered == real


def test_a_hand_written_toml_carrying_a_silent_repair_is_refused_on_parse() -> None:
    """The parse direction runs the same validators — it is not a trusting reader."""
    text = (
        '[general]\ncaption_extension = ".txt"\n\n[[datasets]]\n'
        'video_directory = "/d"\ncache_directory = "/c"\nresolution = [640, 352]\n'
        'frame_extraction = "uniform"\ntarget_frames = [45]\nframe_sample = 1\n'
    )
    with pytest.raises(ValidationError, match="rewritten to 'head'"):
        parse_musubi_toml(text)


def test_rendered_keys_are_a_subset_of_the_upstream_vocabulary() -> None:
    """A renderer emitting a key musubi's schema REJECTS must die in CI, not in a metered container."""
    emitted: set[str] = set()
    for source in [*_fixture_sources(), *(_source_for_mode(m) for m in _MODE_EXPECTATIONS)]:
        blocks = tomllib.loads(
            render_musubi_toml([source], data_root="/root", caption_extension=".txt")
        )["datasets"]
        emitted |= set(blocks[0])
    unknown = emitted - MUSUBI_DATASET_KEYS
    assert not unknown, (
        f"rendered key(s) {sorted(unknown)} are not in the transcribed musubi vocabulary — "
        f"musubi's own schema validator would reject the file"
    )


def test_the_general_table_carries_exactly_the_three_shared_settings() -> None:
    general = tomllib.loads(_render_fixture())["general"]
    assert general == {"caption_extension": ".txt", "enable_bucket": True, "bucket_no_upscale": True}


# ==================================================================================================
# Declared gaps — named symbols, actionable messages
# ==================================================================================================


def test_render_from_config_renders_the_oracle_straight_from_a_loaded_config() -> None:
    """⚠ WAS a declared-gap test (``NotImplementedError``); the gap CLOSED in slice A.

    ``DataConfig.sources`` landed, so ``render_from_config`` has something to read and asserting it
    still raises would be asserting that the wiring did NOT happen. This is the same acceptance
    claim as :func:`test_renders_the_kaboom_config_equivalently`, made through the SEAM an operator
    actually uses: YAML on disk -> validated ``SignetConfig`` -> the real runner config, byte for
    byte. It is the end-to-end proof that the schema field, the loader and the renderer agree.
    """
    rendered = render_from_config(_fixture_config())
    assert rendered == _REAL_TOML.read_text(encoding="utf-8")
    assert rendered == _render_fixture(), (
        "render_from_config disagrees with the explicit render_musubi_toml call — the [general] "
        "kwargs it reads off DataConfig have drifted from the ones the test passes by hand"
    )


def test_render_from_config_refuses_a_sourceless_config_rather_than_inventing_a_block() -> None:
    """A config with no sources has no dataset. Deriving one block from the root would be a guess.

    Reached with a plain namespace rather than a ``SignetConfig``, because the schema itself refuses
    ``family: wan`` without sources — this guards the function's OWN contract, which is what a
    future caller holding a partially-built config would hit.
    """

    class _Data:
        sources = None
        preprocessed_data_root = "/datasets/whatever"
        caption_extension = ".txt"
        enable_bucket = True
        bucket_no_upscale = True

    class _Cfg:
        data = _Data()

    with pytest.raises(ValueError, match="data.sources is empty or absent"):
        render_from_config(_Cfg())


def test_render_control_directories_names_what_lands_it() -> None:
    with pytest.raises(NotImplementedError, match="control_directory"):
        render_control_directories([])


def test_an_unsupported_upstream_key_is_a_named_gap_not_a_silent_drop() -> None:
    text = (
        '[general]\ncaption_extension = ".txt"\n\n[[datasets]]\n'
        'video_directory = "/d"\ncontrol_directory = "/ctrl"\ncache_directory = "/c"\n'
        'resolution = [640, 352]\nframe_extraction = "head"\ntarget_frames = [21]\n'
    )
    with pytest.raises(NotImplementedError, match="render_control_directories"):
        parse_musubi_toml(text)


# ==================================================================================================
# Purity — asserted in a SUBPROCESS
# ==================================================================================================


_PURITY_PROBE = (
    "import sys;"
    "{extra}"
    "import signet_trainer.runners.musubi_toml as m;"
    "bad=[n for n in sys.modules if n in ('modal', 'torch') or"
    " n.startswith('modal.') or n.startswith('torch.')];"
    "print(sorted(set(n.split('.')[0] for n in bad)));"
    "sys.exit(1 if bad else 0)"
)


def _run_purity_probe(extra: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _PURITY_PROBE.format(extra=extra)],
        capture_output=True,
        text=True,
        # The real environment plus an explicit PYTHONPATH: pytest's `pythonpath = ["src"]` ini
        # setting applies to the pytest process, not to a child, and a stripped env cannot start
        # Python on Windows at all.
        env={**os.environ, "PYTHONPATH": str(_REPO / "src")},
    )


def test_the_renderer_imports_no_modal_and_no_torch() -> None:
    """``sys.modules`` inspection in a SUBPROCESS.

    An in-process check would be order-dependent — another test module having already imported
    torch makes it pass or fail for reasons having nothing to do with this module — and that
    order-dependence has bitten this repo twice.
    """
    result = _run_purity_probe()
    assert result.returncode == 0, (
        f"importing the renderer pulled in modal/torch: {result.stdout.strip()}\n{result.stderr}"
    )


def test_the_purity_probe_would_catch_a_violation() -> None:
    """RED self-check: the probe must FAIL when the module under test really does pull torch in.

    Without this, the probe passes vacuously in any environment where the forbidden package is
    simply not installed — which is exactly the environment a CPU/CI box tends to be.
    """
    result = _run_purity_probe(extra="import torch;")
    assert result.returncode == 1 and "torch" in result.stdout, (
        f"the purity probe failed to flag a deliberately-imported torch: {result.stdout.strip()}\n"
        f"{result.stderr}"
    )


def test_parse_defaults_are_MUSUBIS_defaults_not_signets() -> None:
    """An absent key must resolve the way musubi resolves it, or the ingest is not lossless.

    This direction exists so an operator's existing, known-good runner config can be read INTO the
    manifest vocabulary. Two defaults contradicted upstream:

      * `frame_extraction` defaulted to `image`, so a video block relying on musubi's `head` default
        was REJECTED with "kind 'video' cannot use extraction: image" — a complaint about a mode the
        operator's file never mentions;
      * `enable_bucket` defaulted to True, so a `[general]` that omitted it was recorded as
        bucketing ON for a config musubi ran with it OFF — and a re-render then EMITTED
        `enable_bucket = true`, a training-affecting flag flipped by an ingest advertising
        losslessness.
    """
    from signet_trainer.config.sources import ExtractionMode
    from signet_trainer.runners.musubi_toml import parse_musubi_toml

    minimal = "\n".join([
        "[general]",
        'caption_extension = ".txt"',
        "",
        "[[datasets]]",
        'video_directory = "/dataset/K/Videos"',
        'cache_directory = "/dataset/K/Videos/cache"',
        "resolution = [640, 352]",
        "target_frames = [45]",
        "batch_size = 1",
        "",
    ])
    parsed = parse_musubi_toml(minimal)
    assert parsed.enable_bucket is False, (
        "an omitted enable_bucket was recorded as True — re-rendering would turn bucketing ON for a "
        "run that had it OFF"
    )
    assert parsed.sources[0].extraction is ExtractionMode.HEAD, (
        "an omitted frame_extraction on a VIDEO block must resolve to musubi's `head`, not `image`"
    )


def test_an_omitted_frame_extraction_on_an_IMAGE_block_still_resolves_to_image() -> None:
    """The other half of the same default — kind-dependent, not a blanket swap."""
    from signet_trainer.config.sources import ExtractionMode
    from signet_trainer.runners.musubi_toml import parse_musubi_toml

    minimal = "\n".join([
        "[general]",
        'caption_extension = ".txt"',
        "",
        "[[datasets]]",
        'image_directory = "/dataset/K/Images"',
        'cache_directory = "/dataset/K/Images/cache"',
        "resolution = [1024, 1024]",
        "batch_size = 1",
        "",
    ])
    assert parse_musubi_toml(minimal).sources[0].extraction is ExtractionMode.IMAGE
