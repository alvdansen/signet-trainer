"""inference.samples_layout — WHERE a render's artifacts land, keyed by model FAMILY.

The parallel-inference watcher dispatches a metered render and then asks the Volume "did it land?".
That question has a different answer per family, and getting it wrong is not a cosmetic bug — it is
the phantom-spend shape (``KNOWLEDGE.md`` ``watcher`` ``PYTHONPATH`` ``phantom-spend``):

  * ``append_spend`` fires BEFORE ``dispatch_render`` (deliberate — an ATTACHED render whose client
    dies still burned A100 time) and there is NO refund path.
  * ``rendered`` is success-gated.

So a watcher that looks for landed artifacts in the WRONG directory sees every render as FAILED,
never marks the step rendered, and re-dispatches on the next poll — booking the full per-render
estimate each time. On the H3 config that is $3.28 every 240 s (~$49/hr) against renders that may
have succeeded, which would trip ``session_cap_usd`` and halt a healthy campaign.

``h3_sample`` writes to ``<output_dir>/samples_h3/<render key>/`` while the LTX ``sample`` writes to
``<output_dir>/samples/<UTC stamp>/``. BOTH axes differ — the subdirectory AND the naming scheme —
so a fix that only re-pointed the subdirectory would still mis-verify every H3 render.

⛔ FAMILY-AWARE, NEVER H3-HARDCODED. ``samples_subdir`` RAISES on an unknown family rather than
falling back to ``"samples"``: a silent default is exactly how the LTX value would be re-applied to
a future family and re-open this same hole. The LTX answers are unchanged and pinned by test.

Import tier: **stdlib only, no package side effects** — the watcher imports this WITHOUT the modal
SDK loaded, and a test that dragged ``modal`` into ``sys.modules`` would break the dry-run gate's
Anti-Pattern-6 assertion for the whole session (the same reason ``inference/render_key.py`` exists).
"""

from __future__ import annotations

import re
from typing import Any

from signet_trainer.inference.render_key import h3_render_key

__all__ = [
    "SAMPLES_SUBDIR_BY_FAMILY",
    "expected_h3_render_key",
    "expected_qwen_edit_render_key",
    "landed_render_ids",
    "samples_root",
    "samples_subdir",
]

#: The render root each family's sample fn actually writes, relative to ``output_dir``.
#: Transcribed from ``modal/fns.py`` (``"samples_h3" / h3_render_key(...)`` for H3; the LTX
#: ``sample`` branch's ``samples*`` dirs) — NOT guessed. ``samples_text_to_video`` is never written
#: by anything (training-review §2 WR-08); do not add it here on the strength of a mode name.
#:
#: ``qwen_edit`` (family #3) is RESERVED here rather than transcribed: no Qwen sample fn exists yet.
#: Reserving it is the money-safe move and it is why this dict is not a "register when you build it"
#: table. ``samples_subdir`` raises on an unknown family, and the FIRST thing anyone wiring a Qwen
#: render will do is discover that raise and reach for the shortest fix — which, without an entry
#: here, is to point the watcher at ``"samples"`` and re-open the phantom-spend hole this module
#: exists to close. Claiming the name now costs one line and makes that shortcut unavailable. The
#: ``landed_render_ids`` arm below is a LOUD stub, so nothing can accidentally verify against it.
SAMPLES_SUBDIR_BY_FAMILY = {
    "ltx": "samples",
    "h3": "samples_h3",
    "qwen_edit": "samples_qwen_edit",
}

#: A LTX render dir is a UTC wall-clock stamp: ``samples/20260805T154357Z/``.
_LTX_STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")

#: An H3 render dir is an IDENTITY key: ``<checkpoint>_s<seed>_f<frames>_<ids>``. Anchored on the
#: ``_s<d>_f<d>_`` tail so a greedy checkpoint head cannot mis-split (checkpoint dir names keep
#: ``-`` and ``.`` but carry no underscore). Mirrors ``scripts/_h3_grid_serve.py::_KEY_RE``.
_H3_KEY_RE = re.compile(r"^(?P<ckpt>.+)_s(?P<seed>\d+)_f(?P<frames>\d+)_(?P<ids>.+)$")


def samples_subdir(family: str) -> str:
    """The render subdirectory ``family`` writes under ``output_dir``.

    Raises:
        ValueError: on an unrecognised family. Fail-loud is the money-safe direction — a default of
            ``"samples"`` would make a new family silently mis-verify every render and re-dispatch
            it, which is the phantom-spend failure this module exists to prevent.
    """
    try:
        return SAMPLES_SUBDIR_BY_FAMILY[family]
    except KeyError:
        raise ValueError(
            f"unknown model family {family!r} — no samples layout is registered for it. Known "
            f"families: {sorted(SAMPLES_SUBDIR_BY_FAMILY)}. Register the family's render root here "
            "rather than letting the watcher fall back to the LTX 'samples' dir: a watcher that "
            "verifies the wrong directory books the full per-render estimate on every poll while "
            "never marking the render done (KNOWLEDGE.md 'watcher phantom-spend')."
        ) from None


def samples_root(output_dir: str, family: str) -> str:
    """The Volume-relative render root, e.g. ``outputs/h3_embe_r1/samples_h3``."""
    return f"{output_dir.rstrip('/')}/{samples_subdir(family)}"


def expected_h3_render_key(
    *, checkpoint: str, seed: int, frame_count: int, subject_ids: Any
) -> str:
    """The directory ``h3_sample`` WILL write for this (checkpoint, seed, frames, refs) request.

    The watcher's "did this render land?" check must key on exactly what the render keys on, or it
    mis-attributes one sample config's output to another. All five H3 sample configs share
    ``output_dir``, ``seed`` and their prompt set, so they write identical clip FILENAMES; only the
    render-dir identity separates them. A watcher checking a coarser key (checkpoint alone) would
    accept the ``A+029`` render as proof the ``B+029`` render landed.

    Delegates to ``inference.render_key.h3_render_key`` — the render's own function, never a
    re-implementation, so the two cannot drift.
    """
    return h3_render_key(
        checkpoint=checkpoint, seed=seed, frame_count=frame_count, subject_ids=subject_ids
    )


def expected_qwen_edit_render_key(**kwargs: Any) -> str:
    """DECLARED STUB — the ``qwen_edit`` render identity does not exist yet.

    Raises unconditionally and on purpose. The CPU dry-run milestone ships no Qwen sampler, no grid
    builder and no render key, so there is nothing to transcribe here — and a *guess* would be worse
    than a gap, because the watcher's whole contract is that the key it checks for is byte-identical
    to the key the render writes. A plausible-looking guess is how a render that succeeded gets
    marked FAILED and re-dispatched at full price.

    ⚠ Two things must be settled TOGETHER when this lands, and settling only the first is the trap:
    an ``h3``-shaped key (``<ckpt>_s<seed>_f<frames>_<ids>``) carries ``_f<frames>``, which is
    meaningless for an image family where F is pinned to 1 — so the identity axes are the checkpoint,
    the seed, and *which control images were fed*. Whatever shape is chosen, ``_QWEN_EDIT_KEY_RE``
    and the ``landed_render_ids`` arm below must be updated in the SAME commit, or the watcher will
    parse zero landed renders out of a directory full of them.
    """
    raise NotImplementedError(
        "[qwen_edit] expected_qwen_edit_render_key is a DECLARED STUB (no Qwen sampler exists yet). "
        f"Called with {sorted(kwargs)}. The CPU dry-run milestone deliberately ships no render "
        "surface; this raise exists so the gap announces itself at the call site rather than "
        "degrading into a guessed key that makes every successful render look FAILED and get "
        "re-dispatched at full price. Land it in the sampler pass together with "
        "inference/render_key.py's qwen_edit key builder, the _QWEN_EDIT_KEY_RE below, the "
        "landed_render_ids arm, and their tests."
    )


#: A ``qwen_edit`` render dir has NO agreed shape yet — see ``expected_qwen_edit_render_key``.
#: Deliberately ``None`` rather than a speculative pattern: ``landed_render_ids`` checks it and
#: raises, so the absence is a loud stop instead of a regex that quietly matches nothing.
_QWEN_EDIT_KEY_RE = None


def landed_render_ids(listing: str, family: str) -> list[str]:
    """Every committed render identity visible in a ``modal volume ls`` listing, sorted.

    ``listing`` is raw CLI stdout (the watcher shells out; this stays a pure string function so it
    is testable with zero Modal and zero spend).

      * ``ltx`` -> the UTC stamps, the historical behaviour, byte-for-byte.
      * ``h3``  -> the identity keys, so two renders differing ONLY in reference condition or frame
        count are two distinct ids rather than one.
      * ``qwen_edit`` -> RAISES. There is no Qwen render key yet (see
        ``expected_qwen_edit_render_key``), and the H3 arm below is NOT a safe default for it: an
        H3-shaped listing parse against Qwen dir names returns ``[]`` for a directory full of
        successful renders, which reads to the watcher as "nothing landed" and re-dispatches every
        one of them. An explicit arm is required precisely because the fall-through is silent.
    """
    subdir = samples_subdir(family)  # validates the family first — unknown families raise here
    if family == "ltx":
        return sorted(set(_LTX_STAMP_RE.findall(listing or "")))
    if family == "qwen_edit":
        raise NotImplementedError(
            "[qwen_edit] landed_render_ids is a DECLARED STUB: the render-key shape for family "
            f"'qwen_edit' is not defined yet, so {subdir!r} cannot be parsed. Falling through to "
            "the H3 key regex would return [] for a directory of SUCCESSFUL renders — the watcher "
            "would read that as 'nothing landed', never mark the step rendered, and re-book the "
            "full per-render estimate on every poll (KNOWLEDGE.md 'watcher phantom-spend'). Define "
            "the key in expected_qwen_edit_render_key first, then _QWEN_EDIT_KEY_RE, then this arm."
        )

    found: set[str] = set()
    for raw in (listing or "").splitlines():
        # `modal volume ls` prints Volume-relative paths (`outputs/x/samples_h3/<key>`); take the
        # segment after the render root so a checkpoint name containing `/` can never leak through.
        name = raw.strip().rstrip("/").split(f"{subdir}/")[-1].split("/")[-1]
        if name and _H3_KEY_RE.match(name):
            found.add(name)
    return sorted(found)
