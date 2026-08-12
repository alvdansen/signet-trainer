"""train.family_hooks — which ``step_fn`` / ``collate_fn`` ``train_loop`` gets, keyed by model FAMILY.

``train/loop.py`` is model-agnostic in everything that matters (resume, accumulate, clip, save ->
commit -> callback -> commit) and model-SPECIFIC in exactly two arguments: ``step_fn`` and
``collate_fn`` (``loop.py:251-252``, documented at ``:274-287``). Those two arguments ARE the family
seam. This module is the table that resolves them, and nothing else — ``train_loop`` itself is not
touched, and its cadence logic is not reachable from here.

Why a table with a LOUD raise, rather than a default
-----------------------------------------------------
``step_fn=None`` is not a neutral value: ``loop.py:300-321`` interprets it as *"build the LTX
forward"*, which imports ``ltx_core`` (``build_step_deps``) and assembles LTX ``Modality`` inputs. On
family #3 that import is the failure — and on a family whose batch merely happens to survive it, the
result would be an adapter trained through the wrong objective at a perfectly ordinary loss value.
So an unrecognised family RAISES here rather than falling through to the LTX default, the same
money-safe direction ``inference/samples_layout.samples_subdir`` takes for the same class of reason
(``samples_layout.py:20-22, 75-89``: a silent fallback is how one family's answer gets re-applied to
the next one).

Import tier: **stdlib + ``dataclasses`` only at module scope.** Every family's builder is imported
FUNCTION-LOCAL inside its own branch, so resolving ``"ltx"`` never imports the qwen strategy and
resolving ``"qwen_edit"`` never imports ``ltx_core``. That is what lets a config-load / structural
scan import this module for the table alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "LOOP_HOOKS_BY_FAMILY",
    "LoopHooks",
    "build_loop_hooks",
]

#: What each family's ``train_loop`` seam resolves to, as prose. A table of STRINGS rather than of
#: callables on purpose: importing three families' forwards to build a dispatch dict would drag
#: ``ltx_core`` and the qwen strategy into every consumer's import closure, which is the BK-01
#: landmine (``config/validators.py:69-72`` states it for the same reason).
LOOP_HOOKS_BY_FAMILY: dict[str, str] = {
    "ltx": "train/loop.py's own default — build_step_deps() + training_step (loop.py:300-321)",
    "h3": "modal/fns.py::h3_train's inline _h3_step closure (fns.py:4111-4141)",
    "qwen_edit": "train/qwen_edit_step.build_qwen_edit_step_fn + qwen_edit_collate_fn",
}


@dataclass(frozen=True)
class LoopHooks:
    """The two family-specific arguments ``train_loop`` takes, bundled so they travel together.

    ``None`` means *"pass nothing and let the loop keep its own default"* — which for ``step_fn`` is
    the LTX forward and for ``collate_fn`` is torch's default collate. They are bundled rather than
    returned as a pair because getting ONE of them right is the failure mode: an H3-or-qwen
    ``step_fn`` with torch's default ``collate_fn`` receives a per-slot control list restructured
    into batched leaves, which fails deep inside the strategy on a type, not a shape.
    """

    step_fn: Any | None
    collate_fn: Any | None


def build_loop_hooks(family: str, **kwargs: Any) -> LoopHooks:
    """Resolve ``(step_fn, collate_fn)`` for ``family``. Raises on anything unrecognised.

    Args:
        family: ``model.family`` — ``"ltx"`` | ``"h3"`` | ``"qwen_edit"``.
        **kwargs: the family's builder arguments. ``ltx`` accepts none (it has no builder to feed);
            ``qwen_edit`` requires ``strategy`` and ``seed`` and forwards the rest to
            :func:`train.qwen_edit_step.build_qwen_edit_step_fn`.

    Raises:
        ValueError: on an unknown family, or on kwargs handed to a family that has no builder to
            consume them — a silently-ignored builder argument is the same defect class as a
            silently-ignored config field (``config/schema.py``'s bidirectional lean field-split).
        NotImplementedError: on ``"h3"``, which today builds its closure inline where the loaded
            deps live. Named rather than absent, per the declared-stub discipline.
    """
    if family not in LOOP_HOOKS_BY_FAMILY:
        raise ValueError(
            f"unknown model family {family!r} — no train-loop hooks are registered for it. Known "
            f"families: {sorted(LOOP_HOOKS_BY_FAMILY)}. Register the family here rather than "
            f"letting the caller pass step_fn=None: the loop reads None as 'build the LTX forward' "
            f"(loop.py:300-321), which imports ltx_core and assembles LTX Modality inputs — on a "
            f"non-LTX family that is either an import error or, worse, a plausible loss curve from "
            f"the wrong objective."
        )

    if family == "ltx":
        if kwargs:
            raise ValueError(
                f"build_loop_hooks('ltx') takes no builder kwargs, got {sorted(kwargs)}. The LTX "
                f"forward is constructed INSIDE train_loop from the config it already holds "
                f"(loop.py:300-321); anything passed here would be silently dropped. Set the knobs "
                f"on config.conditioning — build_conditioning_kwargs (loop.py:96-126) threads them."
            )
        # Both None: train_loop builds build_step_deps() + training_step and keeps torch's default
        # collate. Byte-identical to every existing LTX call site, which passes neither argument.
        return LoopHooks(step_fn=None, collate_fn=None)

    if family == "h3":
        raise NotImplementedError(
            "H3's step closure has no factory yet: it is built INLINE in modal/fns.py::h3_train "
            "(fns.py:4111-4141) where the loaded transformer, the H3StepDeps bundle, the strategy "
            "and the config are all in scope, and it captures all four. Route H3 runs through "
            "h3_train (which already passes step_fn=_h3_step and collate_fn=lambda b: b[0]), or "
            "lift that closure into train/h3_step.py as build_h3_step_fn(strategy, ...) and "
            "register it in this branch — the qwen_edit branch below is the shape to copy. Do NOT "
            "fall back to LoopHooks(None, None): that is the LTX forward."
        )

    # family == "qwen_edit"
    from signet_trainer.train.qwen_edit_step import (  # noqa: PLC0415 — family-local, see module doc
        build_qwen_edit_step_fn,
        qwen_edit_collate_fn,
    )

    missing = [name for name in ("strategy", "seed") if name not in kwargs]
    if missing:
        raise ValueError(
            f"build_loop_hooks('qwen_edit') requires {missing}: the step needs an already-"
            f"constructed QwenEditStrategy (it carries pack_fn / blank_latent_fn / device / dtype) "
            f"and the run seed (it owns the torch.Generator driving the strategy's noise draw — "
            f"loop.py hands a numpy Generator, which torch.randn cannot consume)."
        )
    return LoopHooks(
        step_fn=build_qwen_edit_step_fn(**kwargs),
        collate_fn=qwen_edit_collate_fn,
    )
