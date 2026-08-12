"""``DataConfig.sources`` — the multi-source wiring (slice A), asserted through the real loader.

``tests/test_musubi_toml_render.py`` proves the RENDERER reproduces Timothy's runner config. This
file proves the SCHEMA: that a source list survives YAML -> ``SignetConfig`` intact, that every
refusal fires at config load rather than in a metered container, and — the part that costs the most
if it is wrong — that the feature is genuinely opt-in.

Everything here is CPU-only, filesystem-read-only, and free.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from signet_trainer.config.load import load_config, load_config_from_text
from signet_trainer.config.schema import DataConfig, SignetConfig
from signet_trainer.config.sources import ExtractionMode, SourceSpec
from signet_trainer.data.multisource import NATIVE_FAMILIES, build_native_source_datasets

_REPO = Path(__file__).resolve().parents[1]
_EXAMPLE = _REPO / "configs" / "wan21_kaboom.example.yaml"


def _wan_payload(**overrides) -> dict:
    """The example config as a dict, with top-level overrides applied."""
    payload = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))
    payload.update(overrides)
    return payload


def _with_sources(*sources: dict, **data_overrides) -> dict:
    payload = _wan_payload()
    payload["data"]["sources"] = list(sources)
    payload["data"].update(data_overrides)
    return payload


_STILLS = {
    "id": "stills",
    "kind": "image",
    "directory": "/d/Images",
    "resolution": [1024, 1024],
    "extraction": "image",
}


# ==================================================================================================
# THE ADDITIVE PROMISE — absence is the identity path
# ==================================================================================================


def test_sources_defaults_to_none_and_none_means_behave_as_today() -> None:
    """The default is ``None``, NOT an empty list and NOT a one-element list.

    This is the whole additive guarantee in one assertion. An empty-list default would make
    ``if cfg.data.sources`` and ``if cfg.data.sources is not None`` disagree at every call site; a
    synthesised one-element default would put every pre-existing config on the new code path, where
    "byte-identical" would be a claim about a renderer rather than about an untouched branch.
    """
    assert DataConfig.model_fields["sources"].default is None
    assert DataConfig(preprocessed_data_root="/data").sources is None
    assert load_config(_REPO / "configs" / "ltx23_lora.example.yaml").data.sources is None


def test_the_general_trio_defaults_match_the_renderers_own_defaults() -> None:
    """A wan config that omits them renders the file it would have rendered by naming them.

    Asserted against the renderer's signature rather than against literals, so the two cannot drift:
    if someone changes a default on either side this fails instead of silently changing an artifact.
    """
    import inspect

    from signet_trainer.runners.musubi_toml import render_musubi_toml

    params = inspect.signature(render_musubi_toml).parameters
    assert DataConfig.model_fields["enable_bucket"].default == params["enable_bucket"].default
    assert (
        DataConfig.model_fields["bucket_no_upscale"].default
        == params["bucket_no_upscale"].default
    )


# ==================================================================================================
# The example config — the wiring, end to end
# ==================================================================================================


def test_the_example_loads_and_yields_real_source_specs() -> None:
    cfg = load_config(_EXAMPLE)
    assert cfg.model.family == "wan"
    assert [s.id for s in cfg.data.sources] == ["stills", "appearance", "motion"]
    assert all(isinstance(s, SourceSpec) for s in cfg.data.sources)
    assert [s.extraction for s in cfg.data.sources] == [
        ExtractionMode.IMAGE,
        ExtractionMode.HEAD,
        ExtractionMode.UNIFORM,
    ]


def test_the_two_video_views_read_one_directory_through_two_caches() -> None:
    """The FEATURE, asserted on the loaded config rather than only on the rendered TOML."""
    cfg = load_config(_EXAMPLE)
    videos = [s for s in cfg.data.sources if s.kind == "video"]
    assert len({s.directory for s in videos}) == 1, "the two views must read ONE corpus"
    roots = {s.resolve_cache_root(cfg.data.preprocessed_data_root) for s in videos}
    assert len(roots) == len(videos), "each view must own its cache — the cache IS the identity"


def test_preprocessed_data_root_is_the_cache_parent_for_an_unnamed_source() -> None:
    """The field's second role, exercised through the loader: derived, never guessed."""
    payload = _with_sources({**_STILLS, "resolution": [1280, 720]})
    # One image source -> exactly one view, [1280, 720, 1]; the wan arm requires training_dims to
    # restate it. That coupling is the point of the law, so the test states it rather than dodging.
    payload["training_dims"] = [1280, 720, 1]
    cfg = load_config_from_text(yaml.safe_dump(payload))
    (source,) = cfg.data.sources
    assert source.cache_root is None
    assert source.resolve_cache_root(cfg.data.preprocessed_data_root) == (
        f"{cfg.data.preprocessed_data_root}/cache/stills"
    )


# ==================================================================================================
# Native families — REFUSED, with the symbol that would land it
# ==================================================================================================


@pytest.mark.parametrize("family", NATIVE_FAMILIES)
def test_sources_is_refused_on_every_native_family(family: str) -> None:
    """Not "meaningless on an image family" — unconsumed on ALL of them, which is stronger.

    ``qwen_edit``'s frame pin does make the video extraction vocabulary meaningless there, and
    ``kind: image`` corpus balancing would be perfectly coherent. It is refused anyway, because
    signet's native loop reads ONE ``preprocessed_data_root`` and has no multi-corpus dataset to
    honour the list with. A key that loads and changes nothing is the defect.
    """
    payload = {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data", "sources": [_STILLS]},
        "training": {"max_steps": 100},
        "model": {"family": family},
    }
    with pytest.raises((ValidationError, ValueError)) as exc:
        SignetConfig.model_validate(payload)
    message = str(exc.value)
    assert "data.sources is set" in message
    assert "build_native_source_datasets" in message, (
        "the refusal must name the symbol that lands it, not merely say no"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("caption_extension", ".caption"), ("enable_bucket", False), ("bucket_no_upscale", False)],
)
def test_the_general_trio_is_refused_under_a_native_family(field: str, value: object) -> None:
    """Set-but-unread is the same defect as sources-but-unconsumed — same reverse guard."""
    payload = {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data", field: value},
        "training": {"max_steps": 100},
    }
    with pytest.raises((ValidationError, ValueError), match=field):
        SignetConfig.model_validate(payload)


def test_an_ltx_config_that_never_mentions_the_new_fields_is_untouched() -> None:
    """The guard fires on a DECLARED value only — silence is the norm and stays free."""
    cfg = SignetConfig.model_validate(
        {
            "training_dims": [768, 512, 49],
            "data": {"preprocessed_data_root": "/data"},
            "training": {"max_steps": 100},
        }
    )
    assert (cfg.data.caption_extension, cfg.data.enable_bucket, cfg.data.bucket_no_upscale) == (
        ".txt",
        True,
        True,
    )


def test_build_native_source_datasets_names_what_lands_it() -> None:
    with pytest.raises(NotImplementedError, match="weighted sampler|MIXTURE PROPORTION"):
        build_native_source_datasets([], family="qwen_edit")


# ==================================================================================================
# The wan family arm
# ==================================================================================================


def test_wan_without_sources_is_refused_rather_than_derived_from_the_root() -> None:
    payload = _wan_payload()
    payload["data"].pop("sources")
    with pytest.raises((ValidationError, ValueError), match="data.sources is absent"):
        SignetConfig.model_validate(payload)


def test_wan_refuses_a_declared_resolution_bucket_list() -> None:
    """musubi buckets from each source's own area budget; this list would be read by nobody."""
    payload = _wan_payload()
    payload["data"]["resolution_buckets"] = ["1280x720x21"]
    with pytest.raises((ValidationError, ValueError), match="resolution_buckets"):
        SignetConfig.model_validate(payload)


def test_wan_accepts_the_default_bucket_list_because_silence_is_not_a_declaration() -> None:
    """The reverse guard must not fire on a default the operator never wrote."""
    assert load_config(_EXAMPLE).data.resolution_buckets == (
        DataConfig.model_fields["resolution_buckets"].default_factory()
    )


def test_wan_dims_law_is_musubis_not_ltxs() -> None:
    """1280x720x21 loads on wan and on nothing else — the third dims branch, exercised."""
    assert list(load_config(_EXAMPLE).training_dims) == [1280, 720, 21]
    ltx = _wan_payload()
    ltx["model"] = {"family": "ltx"}
    ltx["data"].pop("sources")
    for key in ("caption_extension", "enable_bucket", "bucket_no_upscale"):
        ltx["data"].pop(key, None)
    with pytest.raises((ValidationError, ValueError), match="720|frame count 21"):
        SignetConfig.model_validate(ltx)


@pytest.mark.parametrize("frames", [20, 22, 24])
def test_wan_refuses_a_clip_length_musubi_would_silently_floor(frames: int) -> None:
    payload = _wan_payload()
    payload["training_dims"] = [1280, 720, frames]
    with pytest.raises((ValidationError, ValueError)):
        SignetConfig.model_validate(payload)


def test_a_full_mode_source_is_priced_at_max_frames_not_at_one() -> None:
    """``full`` ignores target_frames — its extent lives in ``max_frames``.

    Reading the view's F off ``max(target_frames, default=1)`` would price a 129-frame clip as a
    single frame, which is not a rounding error: it would silently let the largest-view law accept a
    training_dims that under-prices the run by two orders of magnitude. Asserted from both sides —
    the correct triple loads, and a 1-frame restatement of it is refused.
    """
    full = {
        "id": "whole",
        "kind": "video",
        "directory": "/d/Videos",
        "resolution": [640, 352],
        "extraction": "full",
        "max_frames": 129,
    }
    payload = _with_sources(full)
    payload["training_dims"] = [640, 352, 129]
    assert list(load_config_from_text(yaml.safe_dump(payload)).training_dims) == [640, 352, 129]

    payload["training_dims"] = [640, 352, 1]
    with pytest.raises((ValidationError, ValueError), match="largest view declared"):
        load_config_from_text(yaml.safe_dump(payload))


# ==================================================================================================
# Cross-source refusals — at CONFIG LOAD, delegated, not reimplemented
# ==================================================================================================


def test_a_cache_collision_dies_at_config_load_not_at_render_time() -> None:
    """The $0 position. Same function ``render_musubi_toml`` raises on, so they cannot disagree."""
    payload = _with_sources(
        {**_STILLS, "id": "a", "cache_root": "/d/shared"},
        {**_STILLS, "id": "b", "cache_root": "/d/shared/"},  # one directory, two spellings
    )
    with pytest.raises((ValidationError, ValueError), match="cache collision"):
        SignetConfig.model_validate(payload)


def test_duplicate_ids_are_refused_even_when_the_caches_differ() -> None:
    """Distinct caches make this pass the collision gate; the id is still the census label."""
    payload = _with_sources(
        {**_STILLS, "id": "twin", "cache_root": "/d/one"},
        {**_STILLS, "id": "twin", "cache_root": "/d/two"},
    )
    with pytest.raises((ValidationError, ValueError), match="duplicate source id"):
        SignetConfig.model_validate(payload)


def test_an_empty_source_list_is_refused() -> None:
    payload = _wan_payload()
    payload["data"]["sources"] = []
    with pytest.raises((ValidationError, ValueError), match="present but EMPTY"):
        SignetConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"kind": "image", "extraction": "head"}, "requires extraction: image"),
        ({"target_frames": [30]}, "are not N\\*4\\+1"),
        ({"extraction": "uniform", "target_frames": [45], "frame_sample": 1}, "rewritten to 'head'"),
        ({"num_repeats": 0}, "greater than or equal to 1"),
        ({"frame_stride": 3}, "read ONLY by slide"),
    ],
)
def test_per_source_rules_fire_THROUGH_config_load(mutation: dict, expected: str) -> None:
    """The per-source validators are not bypassed by the nesting — they ARE the config-load gate.

    Deliberately asserted from the OUTSIDE (whole YAML -> ``SignetConfig``) rather than by
    constructing a ``SourceSpec`` directly, which ``test_musubi_toml_render.py`` already covers.
    The question here is whether an operator editing a YAML file gets these refusals, and a nested
    model that silently coerced instead of validating would pass the direct test and fail this one.
    """
    base = {
        "id": "x",
        "kind": "video",
        "directory": "/d/Videos",
        "resolution": [640, 352],
        "extraction": "head",
        "target_frames": [21],
    }
    payload = _with_sources({**base, **mutation})
    payload["training_dims"] = [1280, 720, 21]
    with pytest.raises((ValidationError, ValueError), match=expected):
        SignetConfig.model_validate(payload)


def test_a_windows_path_in_a_source_dies_locally_rather_than_contributing_zero_clips() -> None:
    payload = _with_sources({**_STILLS, "directory": "C:\\datasets\\Images"})
    with pytest.raises((ValidationError, ValueError), match="contains a backslash"):
        SignetConfig.model_validate(payload)


def test_data_caption_extension_must_start_with_a_dot() -> None:
    """A missing dot loses EVERY caption at once, which reads as an empty-caption run."""
    payload = _wan_payload()
    payload["data"]["caption_extension"] = "txt"
    with pytest.raises((ValidationError, ValueError), match="must start with"):
        SignetConfig.model_validate(payload)


# ==================================================================================================
# The dispatch fence — a wan config is config-valid and dispatch-REFUSED
# ==================================================================================================


def test_the_dryrun_gate_never_prints_an_ltx_banner_for_a_wan_config(capsys) -> None:
    """The money-safe half of the slice, ASSERTED ON THE CLAIM rather than on the mechanism.

    ⚠ AMENDED, DELIBERATELY (slice B). The predecessor —
    ``test_the_dryrun_gate_refuses_wan_by_name_instead_of_printing_an_ltx_banner`` — asserted
    ``run_dryrun(...) == 1``, because in slice A there was no Wan Modal stage for the gate to gate
    and refusing everything was the correct money-safe answer. The stage has now landed, and
    ``run_dryrun`` routes this family to ``assert_wan_dryrun_manifest`` (the MANIFEST gate the old
    refusal's own docstring specified), so a valid wan config now legitimately returns 0.

    The CLAIM is unchanged and is what this asserts: a ``family: wan`` config must NEVER be
    described by the LTX synthetic-batch banner. The old test proved that by proving nothing passed;
    this proves it by checking WHICH banner was printed — a strictly narrower thing to be right
    about, and one that keeps holding after the stage exists. ``seq_len`` is the LTX banner's
    signature term and appears in no wan banner.

    The refusal itself is NOT retired: ``build_dryrun_inputs``'s wan branch still raises, which is
    what the next test asserts. Gate and refusal are complements — one checks the artifact that
    gates the run, the other names the batch that cannot be built.
    """
    from signet_trainer.dryrun.shapes import run_dryrun

    assert run_dryrun(load_config(_EXAMPLE)) == 0
    captured = capsys.readouterr()
    assert "seq_len" not in captured.out, (
        "the LTX synthetic-batch banner (its signature term is seq_len) described a wan config — "
        "compute_seq_len divides by 32 and assumes 128-channel LTX latents, so that number would be "
        "about a model this run does not use"
    )
    assert "wan config valid" in captured.out and "musubi dataset TOML rendered + parsed" in captured.out
    # The gate must report what it CHECKED, not merely that it passed: the manifest gate's whole
    # value is that the artifact the runner consumes was rendered and read back here, at $0.
    assert "3 [[datasets]] block(s)" in captured.out
    assert "distinct cache director" in captured.out
    # The declared-vs-trained crop is a WARNING, printed inline (never to stderr, where a
    # non-interactive dispatch would lose it) — [640, 360] trains at 640x352 and that is the
    # method's own working value, so it is reported rather than refused.
    assert "640x352" in captured.out


def test_the_wan_arm_is_reached_before_the_ltx_synthetic_batch() -> None:
    """Named-symbol form, so the gap is a callable rather than a branch nobody can point at.

    UNCHANGED by slice B, and that it still passes is the point: the manifest gate landing did not
    make the synthetic-batch refusal obsolete. ``build_dryrun_inputs`` is public, anything may call
    it, and its wan branch must keep saying "there is no synthetic Wan batch" rather than falling
    through to LTX's.
    """
    from signet_trainer.dryrun.shapes import assert_wan_dryrun_geometry, build_dryrun_inputs

    with pytest.raises(NotImplementedError, match="MANIFEST gate"):
        assert_wan_dryrun_geometry(load_config(_EXAMPLE))
    # ...and it is genuinely REACHED from the synthetic-batch builder, not merely defined beside it.
    with pytest.raises(NotImplementedError, match="MANIFEST gate"):
        build_dryrun_inputs(load_config(_EXAMPLE))


def test_the_largest_view_law_and_the_percent16_law_are_mutually_satisfiable() -> None:
    """A non-%16 source must not deadlock the config. The audit's exact round-2 case.

    Two laws run three lines apart on family 'wan': validate_wan_training_dims requires every
    training_dims edge to be a multiple of 16, and _cross_field_checks required training_dims to
    equal the largest DECLARED source view. Whenever the largest view was non-%16 — precisely the
    case musubi_resolution_warnings exists to PERMIT, with a warning — no value satisfied both:

        [640, 360, 45] -> "invalid height 360 ... declare 352"      (the %16 law)
        [640, 352, 45] -> "not the largest view ([640, 360, 45])"   (the largest-view law)

    The operator was handed two errors each demanding what the other forbids, and the only escape
    was editing the source resolution — the refusal the warning path was written not to impose.

    Fixed by comparing the FLOORED view, which is the view musubi actually builds, so the priced
    number stays honest. This test asserts BOTH directions: the floored value loads, and the
    declared non-%16 value is still refused. Asserting only the first would pass against a fix that
    simply deleted the %16 law.
    """
    import yaml

    from signet_trainer.config.load import load_config_from_text

    raw = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))
    raw["data"]["sources"] = [
        {
            "id": "stills", "kind": "image", "directory": "/dataset/K/Images",
            "cache_root": "/dataset/K/Images/cache", "resolution": [1024, 1024],
            "extraction": "image",
        },
        {
            "id": "motion", "kind": "video", "directory": "/dataset/K/Videos",
            "cache_root": "/dataset/K/Videos/c1", "resolution": [640, 360],
            "extraction": "uniform", "target_frames": [45], "frame_sample": 2,
        },
    ]

    raw["training_dims"] = [640, 352, 45]  # the FLOORED view — what musubi will build
    cfg = load_config_from_text(yaml.safe_dump(raw))
    assert list(cfg.training_dims) == [640, 352, 45]
    assert list(cfg.data.sources[1].resolution) == [640, 360], (
        "the source was silently rewritten — the warning path exists so the operator keeps their "
        "declared resolution and is TOLD about the crop, not edited around"
    )

    raw["training_dims"] = [640, 360, 45]  # the declared view — still non-%16, still refused
    with pytest.raises((ValidationError, ValueError)):
        load_config_from_text(yaml.safe_dump(raw))


def test_wan_refuses_the_knobs_the_locked_recipe_overrides() -> None:
    """The lean field-split, applied to the family that had been exempted from it.

    This PR already refuses `data.resolution_buckets` on wan because musubi never reads it — while
    leaving `lora.*` and the whole `training` block free, all hard-overridden by WAN_MUSUBI_RECIPE.

    The failure is the quiet kind: `lora: {rank: 128}` loads, dry-runs green, prints a cost line,
    dispatches — and musubi trains rank 32. Nothing errors. The config committed beside the adapter
    describes a run nobody performed, which corrupts the provenance record rather than failing.
    """
    import yaml

    from signet_trainer.config.load import load_config_from_text

    raw = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))
    assert load_config_from_text(yaml.safe_dump(raw)), "the shipped wan config must still load"

    for patch, expected in (
        ({"lora": {"rank": 128, "alpha": 128}}, "lora.rank"),
        ({"training": {"max_steps": 4000, "learning_rate": 1.0e-4}}, "training.learning_rate"),
        ({"training": {"max_steps": 4000, "optimizer": "adamw"}}, "training.optimizer"),
    ):
        candidate = dict(raw)
        candidate.update(patch)
        with pytest.raises((ValidationError, ValueError)) as exc:
            load_config_from_text(yaml.safe_dump(candidate))
        assert expected in str(exc.value)
        assert "WAN_MUSUBI_RECIPE" in str(exc.value), (
            "the refusal must name where the value actually lives, or the operator has nowhere to go"
        )


def test_wan_still_reads_max_steps_because_the_cost_line_prices_from_it() -> None:
    """The one documented exception. A blanket refusal would break the cost line's own basis."""
    import yaml

    from signet_trainer.config.load import load_config_from_text

    raw = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))
    raw["training"] = {"max_steps": 9999}
    cfg = load_config_from_text(yaml.safe_dump(raw))
    assert cfg.training.max_steps == 9999


def test_a_wan_config_is_never_stamped_with_the_LTX_target_list() -> None:
    """musubi builds its own network (`--network_module networks.lora_wan`) and never reads this.

    `_cross_field_checks` wrote `resolved_lora_targets()` back onto every config that left
    `lora.target_modules` unset — which on wan is the LTX ATTENTION SUFFIX LIST. That put
    `['attn1.to_k', …]` into the artifact shipped beside a lora_wan adapter, asserting a target set
    no part of the run ever used.
    """
    import yaml

    from signet_trainer.config.load import load_config_from_text

    raw = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))
    cfg = load_config_from_text(yaml.safe_dump(raw))
    assert cfg.lora.target_modules is None, (
        f"a wan config was stamped with {cfg.lora.target_modules!r} — musubi never reads it, so "
        f"this is a false claim in the provenance record"
    )
