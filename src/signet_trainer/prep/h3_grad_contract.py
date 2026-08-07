"""prep.h3_grad_contract — the RUNTIME IDIOM every H3 encode helper must run under.

⛔ **D-10-DEF-10, and the class it belongs to.**

The immediate defect was an absence: ``torch.no_grad`` appeared **exactly once** in the ~1,400 lines
of ``prep/h3_encode.py`` — around the Qwen3-VL forward — and the VAE path had none. ``.eval()``
changes norm/dropout behaviour and does **not** disable autograd, and a loaded VAE's parameters
default to ``requires_grad=True``, so ``vae.encode`` built and RETAINED a graph for a backward that
never comes. With ``use_tiling`` on (256px tiles, a 1344x768 canvas, a 22-frame clip padding to two
17-frame chunks) that is ~36-42 encoder forwards per sample, each holding full float32 activations.
The container died on a 544 MiB allocation with 541 MiB free.

Why nothing caught it, and why this module is a different SHAPE from its siblings
--------------------------------------------------------------------------------
Every mechanism this phase has built so far checks a **CONTRACT**:

  | mechanism | question it asks |
  |---|---|
  | ``prep/h3_parity.py`` | do our processor outputs have the same KEY SET as ``__call__``'s? |
  | ``prep/h3_vae_contract.py`` | does the VAE accept this RANK, and does the stub agree? |
  | ``tests/test_h3_real_class_parity.py`` | are our tensors BIT-EXACT against the reference's? |
  | the four-source count check | are the payload COUNTS equal? |

D-10-DEF-10 has correct shapes, correct values, correct keys, correct counts, correct everything —
**and still cannot run.** No contract check can see it, because nothing about the contract is wrong.
It is a **resource / runtime-idiom** defect: the right computation performed in the wrong ambient
mode. So the guard cannot be a diff; it has to be an assertion about the MODE the component runs in.

⚠ **The lesson already existed in this repo and the new leg walked past it.**
``modal/fns.py`` (the LTX Gemma path) carries, in prose::

    # no_grad is MANDATORY here (run-7 OOM): ltx-core GemmaTextEncoder.encode() runs the full
    # 12B 48-layer forward with autograd ON ... one encode's retained autograd graph is ~20-25GB

That was paid for on run 7. It was **remembered**, not **structural**, so the H3 leg inherited
nothing. This module is the attempt to make it structural: one named marker, applied at every encode
entry point, with the enumeration of those entry points computed rather than listed — so a helper
added tomorrow without the marker is a red test on the day it is written, not an OOM on the day it
is dispatched.

no_grad or inference_mode — the choice, and why
-----------------------------------------------
``torch.inference_mode()`` is strictly cheaper: it additionally skips version-counter bookkeeping and
view tracking, so it saves a little more memory and a little more time than ``torch.no_grad()``.
**It is not used here, deliberately.** Tensors created inside ``inference_mode`` are permanently
tagged as *inference tensors*: they may never be recorded by autograd afterwards, and using one in a
graph raises ``RuntimeError: Inference tensors cannot be saved for backward`` — **outside** the
context, arbitrarily far from the code that created them.

These are shared library helpers with no control over their callers. ``encode_video_latents`` is
already reached from three sites, and the reference-encode path is the one a render or an
on-the-fly-conditioning caller would reach first. The saving ``inference_mode`` buys over ``no_grad``
on this path is marginal — the ~78 GiB that was retained is the GRAPH, and ``no_grad`` does not build
it at all — while the failure it risks is a hard crash inside a metered container, at a call site
that did nothing wrong. The house precedent (``modal/fns.py``, the Gemma encode) is also ``no_grad``.

So: ``no_grad``, and the reason is written down rather than left to be re-litigated.

Import tier — **Modal-side EXCEPTION** (10-PATTERNS "Anti-Pattern 6", tier 3): ``torch`` is the only
module-scope third-party import, so a CPU box with no ``diffusers`` can import this module to read
the contract and enumerate the entry points.
"""

from __future__ import annotations

import ast
import functools
from typing import Any, Callable, TypeVar

import torch

__all__ = [
    "H3_ENCODE_ENTRY_POINT_PREFIX",
    "H3_FORWARD_VERBS",
    "H3_NO_GRAD_DECORATOR",
    "H3_NO_GRAD_MARKER_ATTR",
    "ModelForwardSite",
    "freeze_h3_component",
    "h3_encode_entry_points",
    "h3_no_grad",
    "model_forward_sites",
]

_F = TypeVar("_F", bound=Callable[..., Any])

#: The decorator name the entry-point scan looks for. A NAME rather than a bare ``with`` block on
#: purpose: a ``with torch.no_grad():`` somewhere inside a body is not a statement about the FUNCTION
#: (it can cover three lines of forty, and the component call can drift out of it in a later edit),
#: whereas a decorator is a claim about every path through the callable and is trivially enumerable.
H3_NO_GRAD_DECORATOR: str = "h3_no_grad"

#: Attribute stamped on the wrapper so the marker survives ``functools.wraps`` and can be checked on
#: the LIVE object as well as on the source. Two independent observations of the same claim: the AST
#: scan catches a helper that was never decorated, this catches an import-time re-binding that
#: replaced a decorated function with an undecorated one.
H3_NO_GRAD_MARKER_ATTR: str = "__h3_no_grad__"

#: What makes a function an "encode entry point". ``prep/h3_encode`` names every helper that drives a
#: heavyweight component ``encode_*`` and nothing else, so the prefix IS the population — the scan
#: does not need a hand-maintained list, which is the whole point (a list is what gets forgotten).
H3_ENCODE_ENTRY_POINT_PREFIX: str = "encode_"


def h3_no_grad(fn: _F) -> _F:
    """Run ``fn`` with autograd DISABLED. The marker every H3 encode entry point must carry.

    ⛔ **D-10-DEF-10.** Without this, a pre-encode helper builds and retains an autograd graph for a
    backward that never comes. ``.eval()`` does not do this — it switches norm/dropout behaviour and
    leaves ``requires_grad`` exactly as it was, and a freshly-loaded diffusers component has
    ``requires_grad=True`` on every parameter.

    It is applied to the OUTER helpers as well as the inner ones. The nesting is free (``no_grad``
    re-entry is a no-op) and it means every entry point states its own idiom rather than inheriting
    it from a callee that a refactor might inline, reorder or bypass.

    See the module docstring for why this is ``no_grad`` and not ``inference_mode``.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            return fn(*args, **kwargs)

    setattr(wrapper, H3_NO_GRAD_MARKER_ATTR, True)
    return wrapper  # type: ignore[return-value]


def freeze_h3_component(component: Any, *, what: str) -> Any:
    """``.eval()`` **and** ``requires_grad_(False)`` — the belt-and-braces half of D-10-DEF-10.

    ⚠ ``.eval()`` reads like "inference mode" and is not. It sets ``self.training = False``, which
    changes what dropout and the norm layers do, and touches autograd **not at all**. A component
    loaded by ``AutoModel.from_pretrained`` arrives with ``requires_grad=True`` on every parameter,
    so an encode outside a no-grad context retains a graph — which is exactly the ~78 GiB that killed
    attempt 4.

    The no-grad context on the encode helpers is the load-bearing guard; this is the second lock. It
    matters because the two fail in different directions: the context protects a call whose component
    was never frozen, and the freeze protects a call that somehow escapes the context. Freezing also
    makes the intent legible at the LOAD site, where a reader is deciding what the component is for.

    Verified rather than assumed: after the two calls, a parameter still requiring grad is a named
    refusal, because ``requires_grad_`` on a module that overrode it (or on a non-module the loader
    happened to return) can silently do nothing.

    Args:
        component: The loaded component (a ``torch.nn.Module``-shaped object).
        what: Named in every message — a freeze failure should say WHICH component.

    Returns:
        The same component, for use as ``return freeze_h3_component(x.to(device), what=...)``.
    """
    for method in ("eval", "requires_grad_"):
        if not callable(getattr(component, method, None)):
            raise TypeError(
                f"[h3-grad-contract] {what} is a {type(component).__name__}, which has no "
                f"{method}(). An encode component must be a torch.nn.Module-shaped object so it can "
                f"be put in eval mode AND frozen: .eval() alone leaves requires_grad=True on every "
                f"parameter, which is D-10-DEF-10 (the retained autograd graph that exhausted an "
                f"80 GiB A100 on a 544 MiB allocation)."
            )
    component.eval()
    component.requires_grad_(False)

    parameters = getattr(component, "parameters", None)
    if callable(parameters):
        thawed = sum(1 for p in parameters() if getattr(p, "requires_grad", False))
        if thawed:
            raise RuntimeError(
                f"[h3-grad-contract] {what}: {thawed} parameter(s) still require grad after "
                f"requires_grad_(False). The freeze did not take, so every encode through this "
                f"component would retain an autograd graph (D-10-DEF-10). Refusing to hand it to "
                f"the encode helpers."
            )
    return component


def h3_encode_entry_points(source: str) -> dict[str, tuple[str, ...]]:
    """Every PUBLIC top-level ``encode_*`` function in ``source`` -> the decorator names it carries.

    This is the generalization, and it is the reason this module exists rather than two more lines of
    ``with torch.no_grad():``. The population is COMPUTED from the source, so the guard covers
    helpers nobody has written yet: add ``encode_h3_whatever_latents`` tomorrow without the marker
    and the coverage test names it the first time the suite runs.

    Decorator names are flattened to their attribute/callee name (``h3_no_grad``,
    ``some.module.h3_no_grad`` and ``h3_no_grad()`` all report ``"h3_no_grad"``), so the check is
    about which decorator is applied and not about how it was imported.

    Args:
        source: The module source text (read, not imported — the scan must work on a box where the
            module's function-local backends are absent).

    Returns:
        ``{function_name: (decorator_name, ...)}``, in source order.
    """
    entry_points: dict[str, tuple[str, ...]] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith(H3_ENCODE_ENTRY_POINT_PREFIX) or node.name.startswith("_"):
            continue
        entry_points[node.name] = tuple(
            _decorator_name(decorator) for decorator in node.decorator_list
        )
    return entry_points


def _decorator_name(node: ast.expr) -> str:
    """``@f`` / ``@a.b.f`` / ``@f(...)`` -> ``"f"``; anything else -> its AST class name."""
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return type(node).__name__


# --------------------------------------------------------------------------------------------------
# The CLASS-granular scan: every forward of a loaded model, wherever it is written.
# --------------------------------------------------------------------------------------------------

#: Methods that RUN a model rather than describe one. ``decode`` is included deliberately even though
#: nothing on the pre-encode path decodes today — if a decode is ever added here it must make a
#: conscious appearance rather than slipping in unclassified. (Note that an independent H3
#: implementation leaves decode differentiable on purpose; "grad-free" is the right default for a
#: PRE-ENCODE, and a future decode that genuinely needs a graph is a decision to be written down.)
H3_FORWARD_VERBS: frozenset[str] = frozenset({"encode", "decode", "forward"})


class ModelForwardSite:
    """One place a loaded model is RUN, and whether autograd was provably off when it ran."""

    __slots__ = ("function", "grad_free", "lineno", "target", "verb")

    def __init__(
        self, *, function: str, target: str, verb: str | None, lineno: int, grad_free: bool
    ) -> None:
        self.function = function
        self.target = target
        self.verb = verb
        self.lineno = lineno
        self.grad_free = grad_free

    def __repr__(self) -> str:
        call = f"{self.target}.{self.verb}(...)" if self.verb else f"{self.target}(...)"
        return f"{self.function}:{self.lineno} {call}"


def model_forward_sites(source: str) -> list[ModelForwardSite]:
    """Find every **forward of a passed-in model** in ``source`` and say whether it is grad-free.

    ⛔ **This is the CLASS, and the reason it is not just "the encode helpers".** D-10-DEF-10 was
    not "two functions forgot ``no_grad``" — it was that the house had never encoded *"a forward of
    a loaded model outside a training step is grad-free"* as a checkable rule. The LTX leg learned it
    at one site and wrote it there in prose (``modal/fns.py``: *"no_grad is MANDATORY here (run-7
    OOM)"*); the H3 leg inherited the prose's SITE (its text-encoder forward is wrapped) and not its
    RULE, so the VAE surface — which the LTX leg never owned, because canonical ``process_dataset.py``
    encoded LTX latents — was uncovered. Lessons transfer where they are STRUCTURE and fail where
    they are prose at a site.

    **The detection rule is "a parameter you were handed is being invoked."** That is exactly what
    "a forward of a LOADED model" means in this codebase: ``prep/h3_encode``'s own docstrings say
    components are *"Passed in, never constructed here"*, and every helper's first parameter is the
    component. Keying on parameters rather than on names is what makes the scan precise — it cannot
    be fooled by a rename, and it does not fire on a tokenizer's ``encode`` (a tokenizer arrives via
    the processor, not as a parameter of the function that forwards a model).

    Two forms are recognized:

    * ``component(...)``            — a directly-callable module (``text_encoder(**inputs)``)
    * ``component.encode(...)``     — a verb from ``H3_FORWARD_VERBS``

    A site is ``grad_free`` when the enclosing function carries ``@h3_no_grad`` **or** the call is
    lexically inside a ``with torch.no_grad():`` / ``torch.inference_mode():`` block. Both are real
    guarantees; the decorator is preferred because it is a claim about every path through the
    callable, but an existing inner context is not a defect and must not be reported as one.

    Args:
        source: Module source text. Read, never imported — the scan must work on a box with none of
            the heavy backends installed, which is the whole point of the CPU gate.

    Returns:
        Every detected site, in source order.
    """
    tree = ast.parse(source)
    sites: list[ModelForwardSite] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters = _parameter_names(node)
        if not parameters:
            continue
        decorated = any(
            _decorator_name(decorator) == H3_NO_GRAD_DECORATOR
            for decorator in node.decorator_list
        )
        guarded_lines = _no_grad_guarded_lines(node)
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            target, verb = _forward_target(call, parameters)
            if target is None:
                continue
            sites.append(
                ModelForwardSite(
                    function=node.name,
                    target=target,
                    verb=verb,
                    lineno=call.lineno,
                    grad_free=decorated or call.lineno in guarded_lines,
                )
            )
    sites.sort(key=lambda site: site.lineno)
    return sites


def _parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return names


def _forward_target(call: ast.Call, parameters: set[str]) -> tuple[str | None, str | None]:
    """``(parameter_name, verb)`` if this call runs a passed-in model, else ``(None, None)``."""
    func = call.func
    if isinstance(func, ast.Name) and func.id in parameters:
        return func.id, None
    if (
        isinstance(func, ast.Attribute)
        and func.attr in H3_FORWARD_VERBS
        and isinstance(func.value, ast.Name)
        and func.value.id in parameters
    ):
        return func.value.id, func.attr
    return None, None


def _no_grad_guarded_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[int]:
    """Line numbers covered by a ``with torch.no_grad():`` / ``inference_mode():`` in this function.

    Line-range based rather than tree-descent based so a call nested in a comprehension, a helper
    expression or a nested ``with`` inside the guarded block still counts as covered.
    """
    covered: set[int] = set()
    for inner in ast.walk(node):
        if not isinstance(inner, ast.With):
            continue
        if not any(_is_no_grad_context(item.context_expr) for item in inner.items):
            continue
        end = getattr(inner, "end_lineno", None) or inner.lineno
        covered.update(range(inner.lineno, end + 1))
    return covered


def _is_no_grad_context(node: ast.expr) -> bool:
    call = node.func if isinstance(node, ast.Call) else node
    return isinstance(call, ast.Attribute) and call.attr in {"no_grad", "inference_mode"}
