"""D-10-DEF-10 — the CLASS closer for RESOURCE / RUNTIME-IDIOM defects.

⛔ **This defect is a different class from every closer built so far, and that is the point.**

Every mechanism this phase has built checks a CONTRACT — key sets (``prep/h3_parity.py``), ranks
(``prep/h3_vae_contract.py``), counts (the four-source check), bit-exact parity
(``tests/test_h3_real_class_parity.py``). D-10-DEF-10 has correct shapes, correct values, correct
keys, correct counts, **correct everything, and still cannot run**: ``vae.encode`` ran with autograd
ENABLED, retained a graph for a backward that never comes, and exhausted an 80 GiB A100 on a 544 MiB
allocation. No contract diff can see that, because nothing about the contract is wrong.

So the question this file asks is not "is the output right" but **"what MODE did the component run
in"**, and it asks it of every encode entry point that exists — including the ones that do not exist
yet.

Four mechanisms, all CPU, all free
----------------------------------
1. **COVERAGE, computed not listed.** ``h3_encode_entry_points`` enumerates every public
   ``encode_*`` function in ``prep/h3_encode.py`` off the AST; each must carry ``@h3_no_grad``. A
   helper added tomorrow without the marker is named by this test the first time the suite runs —
   which is the whole difference between a lesson that is *structural* and one that is *remembered*.
   The lesson WAS remembered, in prose, at ``modal/fns.py`` (*"no_grad is MANDATORY here (run-7
   OOM)"*), and the H3 leg inherited none of it.
2. **THE PROBE REGISTRY IS ALSO CHECKED FOR COVERAGE.** A decorator that a test never exercises is
   a decorator that can be a no-op. Every enumerated entry point must have a behavioural probe here;
   there is no allowlist and no "unprobed with a reason" escape hatch.
3. **BEHAVIOUR, against components whose parameters REQUIRE GRAD.** The sanctioned stubs hold a real
   ``torch.nn.Parameter`` and record ``torch.is_grad_enabled()`` at each ``encode``. The load-bearing
   assertion is the *observed grad mode*, not the output's ``requires_grad`` — a trailing
   ``.detach()`` produces a grad-free output while the graph was still built and retained, which is
   exactly the OOM. ``encode_h3_text_conditions`` really does end in ``.detach()``, so it is the live
   proof that the weaker assertion would have been vacuous.
4. **A MUTATION PROBE THAT LIVES IN THE TEST.** Every probe is run a second time with the decorators
   STRIPPED off the whole call chain (``__wrapped__``, restored by ``monkeypatch``), and must come
   back grad-ENABLED. Prior agents on this phase repeatedly found false greens; this makes the
   red-when-broken direction a permanent, self-executing assertion rather than a one-off manual
   check, so the mechanism cannot rot into a tautology.

Zero GPU, zero Modal spend, zero model weights, zero network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from signet_trainer.models.h3_loader import (  # noqa: E402
    EXPECTED_H3_IN_CHANNELS,
    EXPECTED_H3_TEXT_ENCODER_LAYER,
)
from signet_trainer.prep import h3_encode  # noqa: E402
from signet_trainer.prep.h3_grad_contract import (  # noqa: E402
    H3_NO_GRAD_DECORATOR,
    H3_NO_GRAD_MARKER_ATTR,
    freeze_h3_component,
    h3_encode_entry_points,
    h3_no_grad,
)
from signet_trainer.prep.h3_vae_contract import (  # noqa: E402
    H3_AUDIO_VAE_HOP_LENGTH,
    H3AudioVaeContractStub,
    H3VideoVaeContractStub,
)

REPO = Path(__file__).resolve().parents[1]
_H3_ENCODE = REPO / "src" / "signet_trainer" / "prep" / "h3_encode.py"

_LATENT_STATS = (torch.zeros(EXPECTED_H3_IN_CHANNELS), torch.ones(EXPECTED_H3_IN_CHANNELS))


def _entry_points() -> dict[str, tuple[str, ...]]:
    return h3_encode_entry_points(_H3_ENCODE.read_text(encoding="utf-8"))


def _entry_points_with_an_inner_no_grad_context() -> set[str]:
    """Entry points carrying their OWN ``with torch.no_grad():`` in addition to the decorator.

    ⚠ COMPUTED, not listed — and the distinction is load-bearing for the mutation probe below. Strip
    the decorator from a helper whose only guard it is and the component runs grad-ENABLED; strip it
    from one that also has an inner context and nothing changes, because the inner context was
    already doing the job.

    Today there is exactly one such helper: ``encode_h3_text_conditions``. That is not a footnote —
    **it is the whole of D-10-DEF-10.** The lesson had been written down at precisely one site, so
    the Qwen3-VL forward survived and every VAE forward did not. Deriving the split means a helper
    that gains or loses an inner context re-classifies itself, instead of a hand-written exception
    quietly going stale.
    """
    import ast

    def _is_no_grad(node: ast.expr) -> bool:
        call = node.func if isinstance(node, ast.Call) else node
        return isinstance(call, ast.Attribute) and call.attr == "no_grad"

    guarded: set[str] = set()
    for node in ast.parse(_H3_ENCODE.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.FunctionDef) or node.name not in _entry_points():
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.With) and any(
                _is_no_grad(item.context_expr) for item in inner.items
            ):
                guarded.add(node.name)
    return guarded


# ==================================================================================================
# Mechanism 1 — the population is COMPUTED, and every member carries the marker
# ==================================================================================================


def test_the_entry_point_scanner_reads_decorators_it_is_shown() -> None:
    """Non-vacuity for everything below: prove the scanner sees all four decorator spellings.

    A scan that silently found nothing would make the coverage test green on a module with no
    markers at all — the purest form of the manufactured confidence this whole file exists against.
    """
    found = h3_encode_entry_points(
        "@h3_no_grad\n"
        "def encode_plain(): ...\n"
        "@some.module.h3_no_grad\n"
        "def encode_dotted(): ...\n"
        "@h3_no_grad()\n"
        "def encode_called(): ...\n"
        "def encode_bare(): ...\n"
        "def _encode_private(): ...\n"
        "def not_an_encode(): ...\n"
    )
    assert found == {
        "encode_plain": ("h3_no_grad",),
        "encode_dotted": ("h3_no_grad",),
        "encode_called": ("h3_no_grad",),
        "encode_bare": (),
    }, f"the entry-point scanner reported {found}"


def test_every_encode_entry_point_runs_under_the_no_grad_marker() -> None:
    """⛔ THE COVERAGE CLOSER. The population is derived from the source, not maintained by hand.

    ``torch.no_grad`` appeared EXACTLY ONCE in this ~1,400-line module — around the Qwen3-VL forward
    — and the VAE path had none. Counting occurrences is how the gap was found; deriving the
    population is how the next one is prevented.
    """
    entry_points = _entry_points()
    assert len(entry_points) >= 5, (
        f"only {sorted(entry_points)} look like encode entry points in prep/h3_encode.py. The scan "
        f"must find the real population or the coverage claim is empty."
    )
    undecorated = sorted(
        name
        for name, decorators in entry_points.items()
        if H3_NO_GRAD_DECORATOR not in decorators
    )
    assert not undecorated, (
        f"encode entry point(s) {undecorated} do not carry @{H3_NO_GRAD_DECORATOR}. Every helper "
        f"that drives a heavyweight component must run with autograd DISABLED: .eval() does not do "
        f"it, and a diffusers component arrives with requires_grad=True on every parameter, so the "
        f"encode builds and RETAINS a graph for a backward that never comes (D-10-DEF-10 — ~36-42 "
        f"tiled encoder forwards per sample, an 80 GiB A100 dead on a 544 MiB allocation). Add the "
        f"decorator; do not relax this test."
    )


def test_the_marker_survives_onto_the_live_functions() -> None:
    """The AST says what the source claims; this says what the imported module actually is.

    Two independent observations of one property. The source scan catches a helper that was never
    decorated; this catches a decorated helper that a later re-binding replaced with a bare one.
    """
    for name in sorted(_entry_points()):
        function = getattr(h3_encode, name)
        assert getattr(function, H3_NO_GRAD_MARKER_ATTR, False), (
            f"prep.h3_encode.{name} is importable but carries no {H3_NO_GRAD_MARKER_ATTR} — the "
            f"live object is not the decorated one the source declares."
        )
        assert getattr(function, "__wrapped__", None) is not None, (
            f"{name} has the marker but no __wrapped__, so the decorator did not actually wrap it"
        )


# ==================================================================================================
# Mechanism 2/3 — the behavioural probes, and the registry that must cover every entry point
#
# ⛔ The video and audio components are the SANCTIONED stubs (prep/h3_vae_contract). They hold a real
# torch.nn.Parameter, so their outputs carry a grad_fn under an enabled autograd context exactly as
# the real classes' do, and they are diffed probe-for-probe against those real classes — including
# on that autograd property — by tests/test_h3_real_class_parity.py. A stub whose output never
# required grad could not fail when a caller forgets the context, which is D-10-DEF-9's mistake moved
# one dimension over.
# ==================================================================================================


class _StubImage:
    """The three things ``prepare_h3_reference_images`` / ``_reference_pixels`` use."""

    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)

    def convert(self, _mode: str) -> _StubImage:
        return self

    def resize(self, size: tuple[int, int], _resample: object = None) -> _StubImage:
        return _StubImage(*size)

    def __array__(self, dtype: object = None, copy: object = None) -> object:  # noqa: ARG002
        width, height = self.size
        array = np.full((height, width, 3), 128, dtype=np.uint8)
        return array if dtype is None else array.astype(dtype)


class _GradSpyTextEncoder:
    """A Qwen3-VL stand-in that records the grad mode and returns a grad-carrying hidden stack.

    ⚠ It is a GRAD SPY, not a fidelity oracle. Nothing here claims to reproduce Qwen3-VL's outputs —
    ``prep/h3_parity.py`` owns that question and answers it against the real ``__call__`` at the
    pinned version. What this must reproduce is the ONE property D-10-DEF-10 is about: a module with
    grad-requiring parameters, so an encode outside a no-grad context retains a graph.
    """

    device = None  # `_to_device` passes tensors straight through when this is None

    def __init__(self, text_dim: int = 8) -> None:
        self.weight = torch.nn.Parameter(torch.ones(text_dim, dtype=torch.float32))
        self.grad_enabled_calls: list[bool] = []

    @property
    def last_grad_enabled(self) -> bool | None:
        return self.grad_enabled_calls[-1] if self.grad_enabled_calls else None

    def __call__(self, **kwargs: Any) -> Any:
        self.grad_enabled_calls.append(bool(torch.is_grad_enabled()))
        length = int(kwargs["input_ids"].shape[-1])
        base = torch.zeros(1, length, self.weight.numel(), dtype=torch.float32)
        # Through the parameter, so the hidden states carry a graph when autograd is on.
        hidden = base + self.weight
        return type(
            "_Out",
            (),
            {"hidden_states": tuple(hidden for _ in range(EXPECTED_H3_TEXT_ENCODER_LAYER + 1))},
        )()


class _PlumbingProcessor:
    """Enough of a processor to reach the encoder call on the TEXT-ONLY branch. Not an oracle.

    ``build_h3_processor_inputs``'s no-reference branch needs a tokenizer and the modality-mask
    method, and then returns. Fidelity of either is guarded elsewhere and deliberately not
    re-litigated here — if this drifts, the probe fails loudly rather than passing quietly.
    """

    def __init__(self, length: int = 6) -> None:
        self.length = int(length)

    def tokenizer(self, texts: Any, **_kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        return {"input_ids": torch.zeros(1, self.length, dtype=torch.long)}

    def create_mm_token_type_ids(self, input_ids: Any) -> list[list[int]]:
        return [[0] * int(input_ids.shape[-1])]


def _probe_video(fn: Callable[..., Any], component: Any) -> Any:
    return fn(component, torch.zeros(3, 22, 32, 48), *_LATENT_STATS)


def _probe_reference(fn: Callable[..., Any], component: Any) -> Any:
    return fn(
        component,
        [_StubImage(1024, 1536)],
        896,
        *_LATENT_STATS,
        descriptors=[{"path": "refs/a.png", "kind": "character", "subject_id": "A"}],
        references_per_sample=1,
    )


def _probe_audio(fn: Callable[..., Any], component: Any) -> Any:
    return fn(
        component,
        torch.zeros(2, 1, H3_AUDIO_VAE_HOP_LENGTH * 3),
        is_reference=True,
    )


def _probe_text(fn: Callable[..., Any], component: Any) -> Any:
    return fn(component, _PlumbingProcessor(), "a probe caption", (), vision_spans=())


#: ``entry point -> (component factory, invocation)``. Asserted below to cover EXACTLY the computed
#: entry-point population — a new helper with no probe here is a named failure, which is what stops
#: the coverage scan degrading into "the decorator is present" with nobody checking it does anything.
_PROBES: dict[str, tuple[Callable[[], Any], Callable[..., Any]]] = {
    "encode_video_latents": (H3VideoVaeContractStub, _probe_video),
    "encode_h3_video_latents": (H3VideoVaeContractStub, _probe_video),
    "encode_h3_reference_latents": (H3VideoVaeContractStub, _probe_reference),
    "encode_h3_audio_latents": (H3AudioVaeContractStub, _probe_audio),
    "encode_h3_text_conditions": (_GradSpyTextEncoder, _probe_text),
}


def test_every_entry_point_has_a_behavioural_probe() -> None:
    """A marker nobody exercises is a marker that can be a no-op.

    There is deliberately NO allowlist here. This phase has already learned what an escape hatch on
    a class-closing gate is worth: ``H3_PROCESSOR_PARITY_ALLOWLIST`` is empty and documented as
    staying empty, because an entry added to make a red test green is how the next metered container
    gets bought.
    """
    entry_points = set(_entry_points())
    unprobed = sorted(entry_points - set(_PROBES))
    assert not unprobed, (
        f"encode entry point(s) {unprobed} carry the marker but have no behavioural probe in "
        f"_PROBES. Add one: drive the helper with a grad-requiring component and let the shared "
        f"assertions run. Without it, @{H3_NO_GRAD_DECORATOR} on that helper is an unverified claim."
    )
    stale = sorted(set(_PROBES) - entry_points)
    assert not stale, f"_PROBES names {stale}, which are not entry points in prep/h3_encode.py"


def test_most_entry_points_have_the_decorator_as_their_SOLE_grad_guard() -> None:
    """Non-vacuity for the mutation probe: its strict branch must actually have members.

    ⛔ The counts here ARE the defect, stated as an assertion. ``torch.no_grad`` appeared exactly
    once in ~1,400 lines — one helper had the lesson, four did not — and the four that did not are
    the ones that reached an OOM. If a future edit gave every helper an inner context, the mutation
    probe's strict branch would empty out and stop proving anything; this is what says so.
    """
    entry_points = set(_entry_points())
    inner_guarded = _entry_points_with_an_inner_no_grad_context()
    assert inner_guarded <= entry_points, f"stale classification: {sorted(inner_guarded)}"
    sole_guard = entry_points - inner_guarded
    assert len(sole_guard) >= 3, (
        f"only {sorted(sole_guard)} rely on @{H3_NO_GRAD_DECORATOR} alone, so the mutation probe's "
        f"strict branch barely exercises anything. Re-read it before relaxing this."
    )
    assert "encode_video_latents" in sole_guard, (
        "encode_video_latents is the chokepoint BOTH pixel producers route through and the exact "
        "site D-10-DEF-10 OOMed at. It must stay in the group the mutation probe drives red."
    )


def _tensors(value: Any) -> list[torch.Tensor]:
    """Every tensor reachable in a payload — helpers return tensors, dicts and nested lists."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for item in value.values() for t in _tensors(item)]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _tensors(item)]
    return []


@pytest.mark.parametrize("name", sorted(_PROBES))
def test_the_component_never_sees_an_enabled_autograd_context(name: str) -> None:
    """⛔ THE ASSERTION D-10-DEF-10 NEEDED. Not "is the output clean" — "was the graph ever built".

    The distinction is the whole defect. A retained graph costs its VRAM DURING the forward; whether
    the caller detaches afterwards is irrelevant to the allocator. ``encode_h3_text_conditions``
    literally ends in ``hidden.detach().to("cpu")``, so on that helper the output check passes with
    or without the fix — see the mutation probe below, which proves it.
    """
    factory, invoke = _PROBES[name]
    component = factory()
    result = invoke(getattr(h3_encode, name), component)

    assert component.grad_enabled_calls, (
        f"{name} never touched its component, so this probe proves nothing about {name}. Fix the "
        f"probe arguments — a vacuous probe is worse than no probe."
    )
    assert component.grad_enabled_calls == [False] * len(component.grad_enabled_calls), (
        f"{name} called its component with autograd ENABLED "
        f"({component.grad_enabled_calls}). That is D-10-DEF-10: the component's parameters require "
        f"grad (.eval() does not change that), so the forward builds and RETAINS activations for a "
        f"backward that never comes."
    )

    tensors = _tensors(result)
    assert tensors, f"{name} returned no tensors, so the requires_grad check below is vacuous"
    grad_bearing = [tuple(t.shape) for t in tensors if t.requires_grad]
    assert not grad_bearing, (
        f"{name} returned tensor(s) {grad_bearing} that still require grad. Anything written to the "
        f"precomputed cache must be graph-free — torch.save would otherwise carry autograd state "
        f"into a file the training run reads back."
    )


@pytest.mark.parametrize("name", sorted(_PROBES))
def test_MUTATION_stripping_the_marker_turns_each_probe_red(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⛔ The mutation probe, executed by the suite rather than performed by hand once.

    Every entry point in the module is re-bound to its ``__wrapped__`` — the exact pre-fix source —
    so the WHOLE call chain runs undecorated, not just the outermost frame. (The outer helpers reach
    the component through ``encode_video_latents``; leaving that one decorated would make them look
    protected by their own marker when they are not.) ``monkeypatch`` restores everything.

    The expected direction is COMPUTED per helper, and getting that right is the point rather than a
    detail: a helper whose only guard is the decorator must come back grad-ENABLED, and one that
    also carries its own ``with torch.no_grad():`` must come back unchanged. Asserting "everything
    goes red" would have been false — and papering over it with a hardcoded exception would have
    hidden exactly the fact D-10-DEF-10 IS: one helper had the lesson written down and the rest did
    not.

    If the strict branch ever stops firing, the sibling test above has become a tautology and the
    class is open again.
    """
    for entry_point in _entry_points():
        original = getattr(h3_encode, entry_point)
        monkeypatch.setattr(h3_encode, entry_point, original.__wrapped__)

    factory, invoke = _PROBES[name]
    component = factory()
    result = invoke(getattr(h3_encode, name), component)
    assert component.grad_enabled_calls, f"the {name} probe never reached its component"

    if name in _entry_points_with_an_inner_no_grad_context():
        assert component.grad_enabled_calls == [False] * len(component.grad_enabled_calls), (
            f"{name} carries its own `with torch.no_grad():` and must therefore be unaffected by "
            f"stripping the decorator. It is not — so the inner context no longer covers the "
            f"component call, and this helper has silently joined the group the decorator is the "
            f"sole guard for."
        )
        # ⚠ And here is why the observed grad MODE is the load-bearing check rather than the
        # output's `requires_grad`: this helper ends in `.detach()`, so its output is grad-free with
        # the graph fully built and retained. On the OOM path the graph is the cost; the detach is
        # bookkeeping that happens after the VRAM is already gone.
        assert not any(t.requires_grad for t in _tensors(result)), (
            "encode_h3_text_conditions no longer detaches its output. That is not itself a bug, but "
            "it removes the live demonstration that a grad-free RESULT does not imply a grad-free "
            "FORWARD — the confusion that makes this whole defect class invisible."
        )
        return

    assert component.grad_enabled_calls == [True] * len(component.grad_enabled_calls), (
        f"with @{H3_NO_GRAD_DECORATOR} stripped from the call chain, {name} STILL ran its component "
        f"under a disabled autograd context ({component.grad_enabled_calls}), and it carries no "
        f"inner no-grad context of its own. Either the probe does not reach the component or "
        f"something else is disabling grad — either way the passing test above is not evidence that "
        f"the decorator does anything."
    )


# ==================================================================================================
# THE CLASS GATE — every forward of a loaded model, everywhere, CLASSIFIED
#
# ⛔ The sharper framing, and it changes what this file is for. D-10-DEF-10 was NOT "two helpers
# forgot no_grad". H3 inherited nearly every house lesson it could — est_hours-as-kill-timer,
# keep_checkpoints null, single-A100, commit-or-vanish, the gated entrypoint, arch-gate-before-spend,
# the CPU preflight, expandable_segments, find_latest-per-stack, frame-law fail-fast — and improved
# on LTX twice. The pattern is narrower and much more useful than "a leg was sloppy":
#
#     lessons transferred where they were encoded as STRUCTURE, and failed exactly where they were
#     encoded as PROSE AT A SITE.
#
# The no_grad lesson WAS inherited: `encode_h3_text_conditions` wraps its Qwen3-VL forward, which is
# precisely the surface `modal/fns.py`'s "no_grad is MANDATORY here (run-7 OOM)" comment describes.
# It was missed on the VAE surface — which the LTX leg never owned, because canonical
# `process_dataset.py` encoded LTX latents, so the house never paid for a VAE-encode no_grad lesson
# and never wrote one down as a CLASS.
#
# So the gate below is class-granular: every place this repo RUNS a model is enumerated from the
# source and must be classified. A new forward added anywhere in the scanned roots is unclassified,
# and therefore RED, on the day it is written — including in a module nobody thought of as an
# "encode" module, which is exactly the blind spot that cost the container.
# ==================================================================================================

#: Roots scanned by the class gate. `train/` is IN, not excluded: a training step MUST build a graph,
#: and classifying those sites positively is what makes the boundary explicit rather than a silence.
_SCANNED_ROOTS = ("prep", "inference", "train", "modal/fns.py")

# Buckets. Every detected site must land in exactly one, with the reason written down once.
_GRAD_FREE_REQUIRED = "grad-free required"  # a model forward outside a training step
_GRAD_FREE_VIA_CALLER = "grad-free at the caller"  # guarded one frame up, caller named
_TRAINING_STEP = "training step"  # MUST build a graph
_GRAD_ON_BY_DESIGN = "grad deliberately ON"  # a probe measuring autograd behaviour
_NOT_A_MODEL_FORWARD = "not a model forward"  # tokenizer / processor / callback / factory

#: ``(module, function, target, verb) -> (bucket, reason)``.
#:
#: ⚠ This is a CLASSIFICATION registry, not an allowlist. An allowlist hides sites; this one forces
#: every site to be named and judged, and the judgement is then ENFORCED — `_GRAD_FREE_REQUIRED`
#: entries must be provably grad-free, `_TRAINING_STEP` entries must provably NOT be. The AST cannot
#: tell a VAE from a tokenizer and does not pretend to; a human says so once, in a checked list.
_FORWARD_SITE_REGISTRY: dict[tuple[str, str, str, str | None], tuple[str, str]] = {
    # ---- the pre-encode: the class's home ground -------------------------------------------------
    ("prep/h3_encode.py", "encode_video_latents", "vae", "encode"): (
        _GRAD_FREE_REQUIRED,
        "D-10-DEF-10 itself — the chokepoint both pixel producers route through.",
    ),
    ("prep/h3_encode.py", "encode_h3_audio_latents", "audio_vae", "encode"): (
        _GRAD_FREE_REQUIRED,
        "the same defect on the path that has never run (0 of 44 clips carry audio).",
    ),
    ("prep/h3_encode.py", "encode_h3_text_conditions", "text_encoder", None): (
        _GRAD_FREE_REQUIRED,
        "the Qwen3-VL forward — the ONE site that had the lesson before this fix.",
    ),
    ("prep/h3_encode.py", "build_h3_presentation", "tokenizer", "encode"): (
        _NOT_A_MODEL_FORWARD,
        "a TOKENIZER's encode: text -> ids, no parameters, no activations, no graph.",
    ),
    ("prep/h3_parity.py", "h3_processor_output_key_diff", "processor", None): (
        _NOT_A_MODEL_FORWARD,
        "Qwen3VLProcessor.__call__ — tokenizer + image-processor config, no model weights.",
    ),
    # ---- family #3 (qwen_edit) pre-encode: the SAME class, inherited as structure not as prose ----
    # This block is the point of the registry. The H3 leg inherited D-10-DEF-10's SITE and not its
    # RULE and paid for it again; family #3 arrives already classified, because the scan named these
    # three the first time the suite ran after prep/qwen_edit_encode.py existed.
    ("prep/qwen_edit_encode.py", "encode_qwen_edit_latents", "vae", "encode"): (
        _GRAD_FREE_REQUIRED,
        "AutoencoderKLQwenImage — the chokepoint every target and control image routes through.",
    ),
    ("prep/qwen_edit_encode.py", "encode_qwen_edit_text_conditions", "text_encoder", None): (
        _GRAD_FREE_REQUIRED,
        "the Qwen2.5-VL forward; its vision half runs too, so the retained graph would be larger.",
    ),
    ("prep/qwen_edit_encode.py", "encode_qwen_edit_text_conditions", "processor", None): (
        _NOT_A_MODEL_FORWARD,
        "Qwen2_5_VLProcessor.__call__ — tokenizer + image-processor, no model weights.",
    ),
    ("prep/h3_vae_contract.py", "_contract_report", "vae", "encode"): (
        _GRAD_FREE_REQUIRED,
        "the probe loop's production-form pass; it asserts its own result is graph-free.",
    ),
    ("prep/h3_vae_contract.py", "_requires_grad_when_enabled", "vae", "encode"): (
        _GRAD_ON_BY_DESIGN,
        "MEASURES autograd behaviour under torch.enable_grad — grad ON is the whole point.",
    ),
    ("prep/h3_grad_contract.py", "h3_no_grad", "fn", None): (
        _NOT_A_MODEL_FORWARD,
        "the decorator invoking the function it wraps; it IS the no-grad context.",
    ),
    ("prep/textseed.py", "detect_frame", "model", None): (
        _GRAD_FREE_REQUIRED,
        "the segmentation-prep detector forward; already inside its own no-grad context.",
    ),
    ("prep/textseed.py", "detect_frame", "proc", None): (
        _NOT_A_MODEL_FORWARD,
        "the detector's PROCESSOR (image -> tensors), not the detector.",
    ),
    ("prep/propagate.py", "process_job", "runner", None): (
        _NOT_A_MODEL_FORWARD,
        "a job runner callable, not a module forward.",
    ),
    # ---- inference: renders, never a backward ----------------------------------------------------
    ("inference/sampler.py", "_encode_driving_audio", "audio_vae_encoder", None): (
        _GRAD_FREE_VIA_CALLER,
        "guarded one frame up — `_render_a2v` wraps the call in torch.no_grad() (run-7 landmine).",
    ),
    ("inference/sampler.py", "build_frozen_audio_latent_state", "audio_latent_shape_cls", None): (
        _NOT_A_MODEL_FORWARD,
        "a CLASS being constructed (ltx-core's LatentShape), passed in to avoid a hard import.",
    ),
    ("inference/sampler.py", "build_frozen_audio_latent_state", "latent_state_cls", None): (
        _NOT_A_MODEL_FORWARD,
        "likewise — ltx-core's LatentState constructor, not a forward.",
    ),
    ("inference/qwen_edit_layout.py", "of", "cls", None): (
        _NOT_A_MODEL_FORWARD,
        "CheckpointBand.of's `cls(...)` — a frozen dataclass of checkpoint NAMES being constructed "
        "by its own classmethod factory. No weights, no tensors; qwen_edit_layout is stdlib-only "
        "and cannot import torch. Registered rather than rewritten to `CheckpointBand(...)`: the "
        "scan's `cls` heuristic is doing its job, and editing correct code to hide from a detector "
        "is how the population stops being the prefix.",
    ),
    # ---- training: a graph is REQUIRED here, and saying so is what defines the boundary -----------
    ("modal/fns.py", "_h3_step", "model_", None): (
        _TRAINING_STEP,
        "the H3 training forward — it MUST build a graph; loss.backward() follows.",
    ),
    ("train/step.py", "training_step", "model", None): (
        _TRAINING_STEP,
        "the LTX training forward — same reason.",
    ),
    ("train/qwen_edit_step.py", "_qwen_edit_step", "model_", None): (
        _TRAINING_STEP,
        "the qwen_edit training forward — it MUST build a graph; loop.py:353 calls backward() on "
        "what it returns. Deliberately BARE (no autocast, QWEN_EDIT_AUTOCAST=False) and deliberately "
        "not grad-free: the only no_grad in this family's step path is the caller's, if any.",
    ),
    ("train/loop.py", "train_loop", "step_fn", None): (
        _TRAINING_STEP,
        "the loop invoking whichever training step the family supplies.",
    ),
    ("train/loop.py", "train_loop", "on_checkpoint", None): (
        _NOT_A_MODEL_FORWARD,
        "the in-loop render CALLBACK; the render it triggers carries its own no-grad context.",
    ),
    ("train/loop.py", "_peak_gib", "probe", None): (
        _NOT_A_MODEL_FORWARD,
        "torch.cuda.max_memory_* — a measurement, passed in so the gauge is CPU-testable.",
    ),
    ("modal/fns.py", "h3_adapter_delta", "model", None): (
        _GRAD_FREE_REQUIRED,
        "the base-vs-adapter delta probe; a measurement forward, already grad-free.",
    ),
}

#: Bucket -> what `grad_free` must be for a site in it. `None` = the AST cannot see the guard (it is
#: at the caller), so the site's own verdict carries no information and a separate check applies.
_BUCKET_EXPECTS_GRAD_FREE = {
    _GRAD_FREE_REQUIRED: True,
    _GRAD_FREE_VIA_CALLER: None,
    _TRAINING_STEP: False,
    _GRAD_ON_BY_DESIGN: False,
    _NOT_A_MODEL_FORWARD: None,
}


def _detected_sites() -> dict[tuple[str, str, str, str | None], bool]:
    """``(module, function, target, verb) -> grad_free``, over every scanned root."""
    from signet_trainer.prep.h3_grad_contract import model_forward_sites

    source_root = REPO / "src" / "signet_trainer"
    detected: dict[tuple[str, str, str, str | None], bool] = {}
    for root in _SCANNED_ROOTS:
        path = source_root / root
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            if "__pycache__" in str(file):
                continue
            module = file.relative_to(source_root).as_posix()
            for site in model_forward_sites(file.read_text(encoding="utf-8")):
                key = (module, site.function, site.target, site.verb)
                # Same key twice in one function (a probe called on two models) collapses; a site is
                # only grad-free if EVERY occurrence of it is.
                detected[key] = detected.get(key, True) and site.grad_free
    return detected


def test_the_forward_detector_finds_the_sites_it_is_shown() -> None:
    """Non-vacuity for the gate: both recognized forms, and the guard forms that clear them."""
    from signet_trainer.prep.h3_grad_contract import model_forward_sites

    sites = model_forward_sites(
        "def bare(vae, x):\n"
        "    return vae.encode(x)\n"
        "@h3_no_grad\n"
        "def decorated(vae, x):\n"
        "    return vae.encode(x)\n"
        "def inner_context(text_encoder, x):\n"
        "    with torch.no_grad():\n"
        "        return text_encoder(x)\n"
        "def local_not_a_param(x):\n"
        "    vae = build()\n"
        "    return vae.encode(x)\n"
    )
    assert [(s.function, s.target, s.verb, s.grad_free) for s in sites] == [
        ("bare", "vae", "encode", False),
        ("decorated", "vae", "encode", True),
        ("inner_context", "text_encoder", None, True),
    ], f"the forward detector reported {sites}"


def test_every_model_forward_in_the_repo_is_CLASSIFIED() -> None:
    """⛔ THE CLASS GATE. A forward nobody has judged is a forward nobody has checked.

    This is the "red on day one" property. Add a model forward anywhere under the scanned roots —
    in a module nobody thinks of as an encode module, which is precisely where this defect hid — and
    the suite names it and refuses to pass until someone says which bucket it belongs in.
    """
    detected = set(_detected_sites())
    registered = set(_FORWARD_SITE_REGISTRY)
    assert detected, "the scan found no forward sites at all — the roots must be wrong"

    unclassified = sorted(detected - registered)
    assert not unclassified, (
        "UNCLASSIFIED model-forward site(s):\n"
        + "\n".join(f"  {module}::{fn} -> {target}{'.' + v if v else ''}(...)"
                    for module, fn, target, v in unclassified)
        + f"\n\nEvery place this repo runs a model must be classified in _FORWARD_SITE_REGISTRY as "
        f"one of: {_GRAD_FREE_REQUIRED!r} (a forward outside a training step — it must be provably "
        f"grad-free), {_GRAD_FREE_VIA_CALLER!r}, {_TRAINING_STEP!r} (a graph is required), "
        f"{_GRAD_ON_BY_DESIGN!r}, or {_NOT_A_MODEL_FORWARD!r}. Write the reason down — that is the "
        f"whole mechanism. D-10-DEF-10 reached a metered container because the rule existed as prose "
        f"at one site instead of as a rule over all of them."
    )
    stale = sorted(registered - detected)
    assert not stale, (
        f"_FORWARD_SITE_REGISTRY classifies site(s) {stale} that no longer exist. A stale registry "
        f"is how a classification survives the code it was a judgement about."
    )


def test_every_classified_forward_obeys_its_bucket() -> None:
    """The classification is ENFORCED, not decorative — otherwise it is just a comment."""
    detected = _detected_sites()
    violations: list[str] = []
    for key, grad_free in sorted(detected.items()):
        bucket, reason = _FORWARD_SITE_REGISTRY[key]
        expected = _BUCKET_EXPECTS_GRAD_FREE[bucket]
        if expected is not None and grad_free != expected:
            module, function, target, verb = key
            call = f"{target}.{verb}(...)" if verb else f"{target}(...)"
            violations.append(
                f"  {module}::{function} -> {call} is classified {bucket!r} ({reason}) but the "
                f"scan says grad_free={grad_free}, expected {expected}"
            )
    assert not violations, "\n".join(["model-forward classification violated:", *violations])

    # Non-vacuity, in BOTH directions: the gate must be enforcing real requirements on real sites,
    # and the training boundary must actually contain something. A registry of nothing-but-exempt
    # entries would satisfy every assertion above.
    buckets = [bucket for bucket, _ in _FORWARD_SITE_REGISTRY.values()]
    assert buckets.count(_GRAD_FREE_REQUIRED) >= 4, (
        f"only {buckets.count(_GRAD_FREE_REQUIRED)} site(s) are held to grad-free — the gate is "
        f"barely enforcing anything"
    )
    assert buckets.count(_TRAINING_STEP) >= 2, (
        "no training-step sites are classified, so the boundary between 'a graph is required' and "
        "'a graph is a bug' is not actually being drawn"
    )


def test_the_caller_guarded_site_really_is_guarded_by_its_caller() -> None:
    """``_GRAD_FREE_VIA_CALLER`` is the one bucket the AST cannot self-check. So check the caller.

    Without this the bucket would be an honour system — the exact shape of a comment that says a
    thing is handled while nothing verifies it (``modal/fns.py`` once promised "the train side
    recomputes them" about spans nothing recomputed, and that cost an entire silent defect).
    """
    import ast

    sampler = REPO / "src" / "signet_trainer" / "inference" / "sampler.py"
    tree = ast.parse(sampler.read_text(encoding="utf-8"))
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        contexts = [
            item.context_expr.func if isinstance(item.context_expr, ast.Call) else item.context_expr
            for item in node.items
        ]
        if not any(isinstance(c, ast.Attribute) and c.attr == "no_grad" for c in contexts):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "_encode_driving_audio"
            ):
                guarded = True
    assert guarded, (
        "inference/sampler.py::_encode_driving_audio runs an audio-VAE forward and is classified "
        "'grad-free at the caller', but no `with torch.no_grad():` block in that module calls it. "
        "Either the guard was removed — in which case this is D-10-DEF-10 on the LTX a2v render "
        "path — or the classification is now wrong. Do not relax this; go and look."
    )


# ==================================================================================================
# The load-site half — freeze_h3_component
# ==================================================================================================


class _TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


def test_freeze_h3_component_does_what_eval_does_NOT() -> None:
    """``.eval()`` reads like "inference mode" and leaves every parameter requiring grad."""
    module = _TinyModule()
    module.train()
    module.eval()
    assert all(p.requires_grad for p in module.parameters()), (
        "sanity: .eval() must NOT have frozen anything, or the point below is not being made"
    )

    returned = freeze_h3_component(module, what="the probe module")
    assert returned is module
    assert not module.training
    assert not any(p.requires_grad for p in module.parameters())


def test_freeze_h3_component_refuses_a_component_it_cannot_freeze() -> None:
    """A loader that returned something module-shaped-but-not is a silent unfrozen component."""
    with pytest.raises(TypeError, match="requires_grad_|eval"):
        freeze_h3_component(object(), what="a non-module")


def test_freeze_h3_component_verifies_the_freeze_TOOK() -> None:
    """``requires_grad_`` can be overridden to do nothing; the check is measured, not assumed."""

    class _Stubborn(_TinyModule):
        def requires_grad_(self, requires_grad: bool = True) -> _Stubborn:  # noqa: ARG002, FBT001, FBT002
            return self  # silently ignores the freeze

    with pytest.raises(RuntimeError, match="still require grad"):
        freeze_h3_component(_Stubborn(), what="a stubborn component")


def test_the_load_site_freezes_rather_than_merely_evaling() -> None:
    """``modal/fns.py`` must not go back to ``.to(device).eval()`` on an encode component.

    Structural, on the AST: ``.eval()`` is the thing that LOOKS sufficient, so the guard has to be
    about which call is made, not about a comment saying which one should be.
    """
    import ast

    fns = REPO / "src" / "signet_trainer" / "modal" / "fns.py"
    tree = ast.parse(fns.read_text(encoding="utf-8"))
    loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_h3_load_component"
    )
    called = {
        inner.func.attr if isinstance(inner.func, ast.Attribute) else getattr(inner.func, "id", "")
        for inner in ast.walk(loader)
        if isinstance(inner, ast.Call)
    }
    assert "freeze_h3_component" in called, (
        "_h3_load_component must route through prep.h3_grad_contract.freeze_h3_component. "
        "`.to(device).eval()` leaves requires_grad=True on every parameter — the load-site half of "
        "D-10-DEF-10."
    )


# ==================================================================================================
# The decorator itself
# ==================================================================================================


def test_h3_no_grad_disables_grad_and_restores_the_previous_mode() -> None:
    """It must not leak: a caller that had autograd on gets it back, exception or not."""
    observed: list[bool] = []

    @h3_no_grad
    def probe(*, boom: bool = False) -> torch.Tensor:
        observed.append(torch.is_grad_enabled())
        if boom:
            raise ValueError("boom")
        return torch.ones(2, requires_grad=False)

    assert torch.is_grad_enabled(), "sanity: the suite runs with autograd on by default"
    probe()
    assert observed == [False]
    assert torch.is_grad_enabled(), "the decorator leaked a disabled autograd context to its caller"

    with pytest.raises(ValueError, match="boom"):
        probe(boom=True)
    assert torch.is_grad_enabled(), "the decorator leaked a disabled context after an exception"


def test_h3_no_grad_is_not_inference_mode() -> None:
    """⛔ A deliberate choice, pinned so a "cheaper context" tidy-up has to argue with a test.

    ``torch.inference_mode`` saves marginally more than ``no_grad``, and permanently tags every
    tensor it creates: using one in a graph later raises ``Inference tensors cannot be saved for
    backward``, arbitrarily far from the code that made it. These are shared helpers whose callers
    they do not control, and the ~78 GiB D-10-DEF-10 retained was the GRAPH — which ``no_grad`` does
    not build at all. The saving is marginal; the failure it risks is a crash in a metered container.
    """

    @h3_no_grad
    def probe() -> torch.Tensor:
        return torch.ones(2)

    out = probe()
    assert not out.requires_grad
    assert not out.is_inference(), (
        "h3_no_grad produced an INFERENCE tensor. It must use torch.no_grad(): an inference tensor "
        "is poisoned for autograd for its whole lifetime, and these helpers do not control their "
        "callers. See prep/h3_grad_contract's module docstring for the full reasoning."
    )
    # And it must be usable in a graph afterwards — the property inference_mode would remove.
    (out * torch.ones(2, requires_grad=True)).sum().backward()
