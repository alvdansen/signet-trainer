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
    mutated = source.replace("config.training_dims[2],", "config.data.frame_count,", 1)
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


def _pre_encode_target_frames() -> str:
    """The expression ``_h3_encode_params`` threads into ``h3_preprocess(target_frames=...)``."""
    node = _function_node(_ENTRYPOINT, "_h3_encode_params")
    for dict_node in ast.walk(node):
        if not isinstance(dict_node, ast.Dict):
            continue
        for key, value in zip(dict_node.keys, dict_node.values, strict=False):
            if isinstance(key, ast.Constant) and key.value == "target_frames":
                return ast.unparse(value)
    raise AssertionError("_h3_encode_params supplies no 'target_frames' — the seam moved")


def _without_root(expression: str) -> str:
    """``cfg.training_dims[2]`` and ``config.training_dims[2]`` are the SAME read."""
    return re.sub(rf"^(?:{'|'.join(_CONFIG_ROOTS)})\.", "", expression)


@pytest.mark.parametrize("function", ["h3_train", "h3_sample"])
def test_both_h3_stages_derive_target_frames_from_the_same_source_as_the_pre_encode(
    function: str,
) -> None:
    """The RoPE geometry must agree with the CACHE, and the cache's frame count has one home.

    ``H3PositionIdsBuilder`` re-derives the target latent grid and refuses any batch whose measured
    rows disagree, so a stage handed a different frame count than the pre-encode used cannot train —
    it aborts naming two numbers. ``config.validation.frame_count`` resolves perfectly well and is
    still wrong here: it is a RENDER field, ``configs/h3_embe_r1.yaml`` does not set it, and the
    schema default is 49 against a 22-frame cache.

    Both sides are re-derived from the real sources and diffed rather than restated, so this fails
    if EITHER moves.
    """
    expected = _without_root(_pre_encode_target_frames())
    supplied = _position_ids_target_frames(_FNS, function)
    assert supplied, f"{function}() no longer builds a position_ids fn — re-target this guard"
    for expression in supplied:
        assert _without_root(expression) == expected, (
            f"{function}() builds H3 RoPE for {expression!r} but the pre-encode encoded "
            f"{_pre_encode_target_frames()!r}. The position-ids builder proves its derivation "
            "against rows measured off the CACHED tensors, so these two must name one source."
        )


def test_the_target_frames_agreement_check_would_report_a_divergence() -> None:
    """Negative control for the comparison above — a mismatch must not compare equal."""
    assert _without_root("config.validation.frame_count") != _without_root("cfg.training_dims[2]")
    assert _without_root("config.training_dims[2]") == _without_root("cfg.training_dims[2]")


def test_the_render_frame_count_stays_a_separate_read() -> None:
    """The RENDER count is a different decision from the training geometry — keep both.

    ``h3_sample`` legitimately reads BOTH: ``training_dims[2]`` for the delta batch it builds out of
    the precomputed training cache, and ``validation.frame_count`` for the clips it renders.
    Collapsing them into one read would be a plausible-looking "cleanup" that silently re-times
    either the eval clips or the adapter-delta geometry.
    """
    chains = {chain for chain, _ in _config_chains(_function_node(_FNS, "h3_sample"))}
    assert ("validation", "frame_count") in chains, "the render count must still come from validation"
    assert ("training_dims",) in chains, "the delta batch's geometry must come from training_dims"
