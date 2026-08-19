"""Phase 10 (gap closure) — every config read in the H3 stages must resolve against the SCHEMA.

D-10-DEF-1 was ``config.data.frame_count``: ``frame_count`` is a ``ValidationConfig`` field, not a
``DataConfig`` one, so the attribute access raised. It passed every local gate — the config loads,
the dry-run never reaches the stage bodies, the entrypoint's five steps all succeed — and the only
discovery channel left was a paid container. On ``h3_train`` that costs cents (the CPU preflight
fires first). On ``h3_sample`` the site sits AFTER ``load_components`` + ``enable_auto_cpu_offload``
+ ``inject_lora``, so it costs most of a 61.7 GiB container.

A test asserting the literal string ``config.validation.frame_count`` would pin THIS defect and
nothing else. Instead this file resolves **every** ``config.<section>.<field>`` chain reachable in
the H3 stage bodies against the real Pydantic models, so any future mis-sectioned attribute goes red
— and it does the resolution statically, without importing ``modal/fns.py`` (importing it builds the
Modal app graph and eagerly resolves every ``Secret.from_name``).

The second half of the file pins what the resolver CANNOT see: ``config.validation.frame_count``
also resolves, and would have been silently WRONG at both sites. The H3 position-ids builder
re-derives the target latent grid and proves it against rows measured off the CACHED tensors, so it
must be handed the frame count the cache was encoded at — the very value
``entrypoint._h3_encode_params`` threads into ``h3_preprocess(target_frames=...)``. That agreement
is re-derived from both real sources and diffed, rather than restated as a literal.

CPU-only, zero GPU, zero Modal spend.
"""

from __future__ import annotations

import ast
import re
import types
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from signet_trainer.config.schema import SignetConfig

REPO = Path(__file__).resolve().parents[1]
_FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"
_ENTRYPOINT = REPO / "src" / "signet_trainer" / "modal" / "entrypoint.py"

#: The names a ``SignetConfig`` is bound to across the Modal tier. ``fns.py``'s by-value stages call
#: theirs ``config``; ``entrypoint.py`` calls its ``cfg``.
_CONFIG_ROOTS = ("config", "cfg")

#: The H3 call sites that read a config and must therefore resolve. Every one of them runs INSIDE a
#: metered container (or, for ``_h3_encode_params``, decides what one is handed).
_H3_READERS = (
    (_FNS, "h3_train"),
    (_FNS, "h3_sample"),
    (_ENTRYPOINT, "_h3_encode_params"),
)


# ==================================================================================================
# The resolver — chains out of the AST, fields out of the live Pydantic models
# ==================================================================================================


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {path.name}")


def _chain_of(node: ast.AST) -> tuple[str, tuple[str, ...]] | None:
    """``a.b.c`` -> ``("a", ("b", "c"))``. Anything not rooted at a bare Name -> ``None``."""
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or not parts:
        return None
    return current.id, tuple(reversed(parts))


def _config_chains(node: ast.AST) -> list[tuple[tuple[str, ...], int]]:
    """Every MAXIMAL config-rooted attribute chain in ``node``, aliases followed.

    Maximal, not every intermediate: ``ast.walk`` visits ``config.h3`` as well as
    ``config.h3.target_aspect``, and asserting on the prefix would be noise. An Attribute that is
    itself the ``.value`` of another Attribute is by definition not maximal.

    Aliases are followed one hop because ``h3_sample`` genuinely uses one (``v = config.validation``
    then ``v.frame_count``). Without it four real reads would be invisible to this guard, and an
    invisible read is exactly the hole D-10-DEF-1 slipped through.
    """
    aliases: dict[str, tuple[str, ...]] = {}
    for assign in ast.walk(node):
        if not isinstance(assign, ast.Assign) or len(assign.targets) != 1:
            continue
        target = assign.targets[0]
        chain = _chain_of(assign.value)
        if isinstance(target, ast.Name) and chain and chain[0] in _CONFIG_ROOTS:
            aliases[target.id] = chain[1]

    nested = {id(a.value) for a in ast.walk(node) if isinstance(a, ast.Attribute)}
    found: list[tuple[tuple[str, ...], int]] = []
    for attribute in ast.walk(node):
        if not isinstance(attribute, ast.Attribute) or id(attribute) in nested:
            continue
        chain = _chain_of(attribute)
        if chain is None:
            continue
        root, parts = chain
        if root in _CONFIG_ROOTS:
            found.append((parts, attribute.lineno))
        elif root in aliases:
            found.append((aliases[root] + parts, attribute.lineno))
    return found


def _attribute_holder(annotation: object) -> type:
    """The class an annotation's attributes should be looked up on.

    ``object`` is the deliberate "cannot check further" answer: once a chain leaves the Pydantic
    models (into a ``str``, a ``tuple``, a ``Literal``) this guard stops claiming anything rather
    than inventing a rule. The strictness that matters is retained — while the chain is still ON a
    ``BaseModel``, an unknown attribute is a failure.
    """
    if isinstance(annotation, type):
        return annotation
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        for arg in args:
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
        return _attribute_holder(args[0]) if args else object
    if isinstance(origin, type):
        return origin
    return object


def _unresolvable(chain: tuple[str, ...]) -> str | None:
    """``None`` if the whole chain resolves against ``SignetConfig``; otherwise WHY it does not."""
    current: type = SignetConfig
    walked: list[str] = ["config"]
    for part in chain:
        fields = getattr(current, "model_fields", None)
        if isinstance(fields, dict) and part in fields:
            current = _attribute_holder(fields[part].annotation)
            walked.append(part)
            continue
        if hasattr(current, part):
            # A method / property / ClassVar on the model (``config.resolved_lora_targets``), or an
            # attribute of whatever the chain descended into. Legal, but its return type is not
            # statically known, so nothing deeper can be checked.
            return None
        holder = current.__name__ if isinstance(current, type) else repr(current)
        known = sorted(fields) if isinstance(fields, dict) else []
        hint = ""
        if known:
            elsewhere = sorted(
                name
                for name, field in SignetConfig.model_fields.items()
                if isinstance(_attribute_holder(field.annotation), type)
                and issubclass(_attribute_holder(field.annotation), BaseModel)
                and part in _attribute_holder(field.annotation).model_fields
            )
            hint = f" {holder} declares {known}."
            if elsewhere:
                hint += f" {part!r} IS a field of: {elsewhere}."
        return f"{'.'.join([*walked, part])} does not exist —{hint}"
    return None


def _bad_reads(path: Path, function: str) -> list[str]:
    """Every config chain in ``function`` that the real schema cannot resolve."""
    return [
        f"{path.name}:{line} {why}"
        for chain, line in _config_chains(_function_node(path, function))
        if (why := _unresolvable(chain)) is not None
    ]


# ==================================================================================================
# The guard
# ==================================================================================================


@pytest.mark.parametrize(("path", "function"), _H3_READERS, ids=lambda v: getattr(v, "name", v))
def test_every_config_read_in_the_h3_stages_resolves_against_the_real_schema(
    path: Path, function: str
) -> None:
    """D-10-DEF-1's whole CLASS, not its one instance.

    A mis-sectioned config read type-checks nowhere, loads nowhere and dry-runs nowhere — it
    surfaces as an ``AttributeError`` inside a metered container. Resolving the chains statically
    against the same Pydantic models the container will use moves that to CI for free.
    """
    bad = _bad_reads(path, function)
    assert not bad, (
        f"{function}() reads config fields that do not exist:\n  " + "\n  ".join(bad) + "\n"
        "These raise AttributeError inside a metered container and nowhere else."
    )


@pytest.mark.parametrize(("path", "function"), _H3_READERS, ids=lambda v: getattr(v, "name", v))
def test_the_config_chain_collector_is_not_vacuous(path: Path, function: str) -> None:
    """A collector that silently found nothing would make the guard above pass forever."""
    chains = _config_chains(_function_node(path, function))
    assert len(chains) >= 8, (
        f"{function}() should read many config fields; the collector found {len(chains)}. A guard "
        "over an empty set is a guard over nothing."
    )


def test_aliased_config_reads_are_followed_not_skipped() -> None:
    """``v = config.validation`` then ``v.frame_count`` must still be checked.

    ``h3_sample`` really does this for the gallery params. If the collector stopped at bare
    ``config.`` roots, four real reads would be exempt from the guard — and an exempt read is where
    the next D-10-DEF-1 lives.
    """
    chains = {chain for chain, _ in _config_chains(_function_node(_FNS, "h3_sample"))}
    assert ("validation", "frame_count") in chains
    assert ("validation", "num_inference_steps") in chains


# --------------------------------------------------------------------------------------------------
# Negative controls — the guard is PROVEN to bite, not assumed to.
# --------------------------------------------------------------------------------------------------


def test_the_resolver_reports_the_exact_defect_it_was_written_for() -> None:
    """``config.data.frame_count`` — D-10-DEF-1 verbatim — must be reported, with the real section."""
    why = _unresolvable(("data", "frame_count"))
    assert why is not None, "the resolver must reject a field that lives in a different block"
    assert "frame_count" in why
    assert "validation" in why, (
        "the message must NAME the block that really declares the field — a refusal that only says "
        "'no such field' is not actionable at 3am"
    )


@pytest.mark.parametrize(
    "chain",
    [
        ("data", "frame_count"),          # D-10-DEF-1 itself
        ("h3", "reference_short_edge"),   # the real field is reference_image_short_edge
        ("training", "steps"),            # the real field is max_steps
        ("validation", "prompt"),         # the real field is prompts
        ("nosuchblock", "anything"),
    ],
)
def test_the_resolver_rejects_every_mis_sectioned_or_misspelled_read(chain: tuple[str, ...]) -> None:
    assert _unresolvable(chain) is not None, f"config.{'.'.join(chain)} must not resolve"


@pytest.mark.parametrize(
    "chain",
    [
        ("data", "preprocessed_data_root"),
        ("validation", "frame_count"),
        ("h3", "reference_image_short_edge"),
        ("training_dims",),
        ("resolved_lora_targets",),        # a METHOD on SignetConfig, not a field
        ("output_dir",),
    ],
)
def test_the_resolver_accepts_the_reads_that_are_genuinely_correct(chain: tuple[str, ...]) -> None:
    """The other half of non-vacuity: a resolver that rejected everything would also be green."""
    assert _unresolvable(chain) is None, f"config.{'.'.join(chain)} is real and must resolve"


def test_the_guard_would_go_red_on_a_mutated_stage_body() -> None:
    """Drive ``_bad_reads``' machinery over a MUTATED copy of the real ``h3_train`` body.

    The file-level guard reads the committed source, so it cannot demonstrate its own failure. This
    re-runs the identical resolution over the same function with one attribute re-sectioned, which
    is precisely the edit that shipped D-10-DEF-1.
    """
    source = _FNS.read_text(encoding="utf-8")
    # Re-targeted for FRAME-COUNT BUCKETING: h3_train's RoPE target_frames is now the declared-bucket
    # helper call ``_h3_frame_buckets(config)``, not a bare attribute chain, so the old
    # ``config.training_dims[2],`` anchor no longer appears there. The audio-rows default under
    # single-bucket mode still reads ``config.training_dims[2]`` directly — that is the anchor now.
    mutated = source.replace(
        "int(config.training_dims[2])", "int(config.data.frame_count)", 1
    )
    assert mutated != source, "the mutation anchor no longer matches — re-target it"

    tree = ast.parse(mutated)
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "h3_train"
    )
    failures = [
        why for chain, _ in _config_chains(node) if (why := _unresolvable(chain)) is not None
    ]
    assert failures, "the mutated body MUST be reported — otherwise the guard is decorative"
    assert any("frame_count" in f for f in failures)


# ==================================================================================================
# What the resolver cannot see: validation.frame_count RESOLVES, and is still the wrong value
# ==================================================================================================


def _position_ids_target_frames(path: Path, function: str) -> list[str]:
    """The expression each ``make_h3_position_ids_fn(target_frames=...)`` call is handed."""
    found: list[str] = []
    for node in ast.walk(_function_node(path, function)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        named = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if named != "make_h3_position_ids_fn":
            continue
        found.extend(
            ast.unparse(kw.value) for kw in node.keywords if kw.arg == "target_frames"
        )
    return found


def _pre_encode_frame_buckets() -> str:
    """The expression ``_h3_encode_params`` threads into ``h3_preprocess(frame_buckets=...)``.

    FRAME-COUNT BUCKETING (re-point, 2026-08): the pre-encode's single source of "what frame counts
    this run declares" is ``frame_buckets``, not the ``target_frames`` scalar — that scalar is only
    the SINGLE-bucket-mode default for a manifest row that names none of its own
    (``fns.py::_row_frames``). The RoPE builder in ``h3_train``/``h3_sample`` needs every declared
    bucket, so that is the seam this guard must follow.
    """
    node = _function_node(_ENTRYPOINT, "_h3_encode_params")
    for dict_node in ast.walk(node):
        if not isinstance(dict_node, ast.Dict):
            continue
        for key, value in zip(dict_node.keys, dict_node.values, strict=False):
            if isinstance(key, ast.Constant) and key.value == "frame_buckets":
                return ast.unparse(value)
    raise AssertionError("_h3_encode_params supplies no 'frame_buckets' — the seam moved")


def _frame_buckets_helper_body() -> str:
    """The expression ``fns.py::_h3_frame_buckets`` returns — the RoPE builder's own source."""
    node = _function_node(_FNS, "_h3_frame_buckets")
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            return ast.unparse(stmt.value)
    raise AssertionError("_h3_frame_buckets has no return — the seam moved")


def _without_root(expression: str) -> str:
    """``cfg.training_dims[2]`` and ``config.training_dims[2]`` are the SAME read."""
    return re.sub(rf"^(?:{'|'.join(_CONFIG_ROOTS)})\.", "", expression)


def _without_any_root(expression: str) -> str:
    """Like ``_without_root``, but strips ``cfg.``/``config.`` wherever they occur, not just at the
    start — needed to diff two INDEPENDENT re-derivations of the same formula (``entrypoint.py``'s
    inline computation and ``fns.py::_h3_frame_buckets``'s body), which necessarily bind the config
    to different local names and are not import-linked (``entrypoint.py`` must not import
    ``modal/fns.py`` — that eagerly builds the Modal app graph).
    """
    return re.sub(rf"\b(?:{'|'.join(_CONFIG_ROOTS)})\.", "", expression)


@pytest.mark.parametrize("function", ["h3_train", "h3_sample"])
def test_both_h3_stages_derive_target_frames_from_the_declared_bucket_helper(
    function: str,
) -> None:
    """Both H3 stages hand the RoPE builder the DECLARED BUCKET SET, via one named helper.

    Under frame-count bucketing ``H3PositionIdsBuilder`` no longer takes the single frame count the
    cache was encoded at — it takes every bucket the run declared and proves each sample's own frame
    count against it (``resolve_bucket``). ``_h3_frame_buckets(config)`` is that single helper; a
    stage that instead handed it ``config.training_dims[2]`` (the pre-bucketing shape) would price
    one bucket for every sample and make the builder raise on the first sample of any other bucket,
    mid-run on a metered container — the same class of defect as D-10-DEF-1, one call earlier.
    """
    supplied = _position_ids_target_frames(_FNS, function)
    assert supplied, f"{function}() no longer builds a position_ids fn — re-target this guard"
    for expression in supplied:
        assert expression == "_h3_frame_buckets(config)", (
            f"{function}() builds H3 RoPE for {expression!r}, not the declared-bucket helper "
            "'_h3_frame_buckets(config)'. Under bucketing the builder must be handed every declared "
            "bucket, never a single scalar."
        )


def test_the_declared_bucket_helper_and_the_pre_encode_derive_it_from_one_formula() -> None:
    """The helper's body and the pre-encode's inline copy must be the SAME formula.

    They cannot share one function call (``entrypoint.py`` does not import ``modal/fns.py``), so each
    re-derives ``sorted({validate_h3_resolution_bucket(b)[2] for b in <cfg>.data.resolution_buckets})``
    independently. Diffing the two bodies (root name stripped) is what proves they cannot silently
    drift apart into two different bucket sets for the same config.
    """
    helper = _without_any_root(_frame_buckets_helper_body())
    pre_encode = _without_any_root(_pre_encode_frame_buckets())
    assert helper == pre_encode, (
        f"fns.py::_h3_frame_buckets computes {helper!r} but entrypoint.py's frame_buckets= computes "
        f"{pre_encode!r}. Both must derive the declared bucket set from data.resolution_buckets by "
        "the SAME formula, or the pre-encode and the training-time RoPE builder can disagree about "
        "which buckets a run declared."
    )


def test_the_bucket_formula_agreement_check_would_report_a_divergence() -> None:
    """Negative control for the comparison above — a mismatch must not compare equal."""
    assert _without_any_root("cfg.data.resolution_buckets") == _without_any_root(
        "config.data.resolution_buckets"
    )
    assert _without_any_root("cfg.validation.frame_count") != _without_any_root(
        "config.data.resolution_buckets"
    )


def test_the_render_frame_count_stays_a_separate_read() -> None:
    """The RENDER count is a different decision from the training geometry — keep both.

    ``h3_sample`` legitimately reads ``validation.frame_count`` for the clips it renders, which stays
    a plain config chain this file's resolver sees directly. The adapter-delta geometry it builds out
    of the precomputed training cache is a SEPARATE decision — under bucketing it no longer comes from
    a bare ``training_dims`` chain (see ``test_both_h3_stages_derive_target_frames_from_the_declared_
    bucket_helper``); it comes from ``_h3_frame_buckets(config)``, a call this AST-chain resolver
    cannot see by design (it walks attribute chains, not calls). Collapsing the render count into the
    training geometry would be a plausible-looking "cleanup" that silently re-times the eval clips.
    """
    chains = {chain for chain, _ in _config_chains(_function_node(_FNS, "h3_sample"))}
    assert ("validation", "frame_count") in chains, "the render count must still come from validation"
    supplied = _position_ids_target_frames(_FNS, "h3_sample")
    assert supplied == ["_h3_frame_buckets(config)"], (
        "the delta batch's geometry must come from the declared-bucket helper, not a bare "
        "training_dims chain"
    )
