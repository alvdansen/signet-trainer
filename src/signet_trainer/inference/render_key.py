"""inference.render_key — the IDENTITY of one render, as a directory name.

``h3_sample`` renders 12 clips sequentially. Committing once at the END makes that a ~6 h loss unit
with no retries and no resume — the shape ``KNOWLEDGE.md``'s ``preemption no-resume`` landmine says
cannot survive, already paid for on LTX embe r1 (a 2.2 h render on a ~24-min preemption cycle could
never finish). The fix is a render directory keyed on WHAT IS BEING RENDERED rather than on the wall
clock, so a re-dispatch skips the clips already on the Volume instead of restarting from zero.

⛔ **The fix is more dangerous than the problem if the key is incomplete**, which is why this is its
own module with its own tests rather than three lines inside ``modal/fns.py``. All five H3 sample
configs share ``output_dir`` (they must — ``find_latest`` resolves the adapter under it), share
``seed: 42`` and share their prompt set by design, so every one of them writes the identical
filename ``{slug}_s42.mp4``. Two differ ONLY in ``validation.reference_subject_ids`` and two more
ONLY in ``validation.frame_count``. A key missing either axis lets one config's resume skip another
config's clips, and the gallery comes out labelled for a reference condition it does not contain —
silent, at a valid shape, on the exact axis the phase exists to measure.

Import tier: **stdlib only, and no package side effects**. It lives here rather than in
``modal/fns.py`` for the same reason ``train/loop.checkpoint_watchdog_exceeded`` does — a test that
had to ``import signet_trainer.modal.fns`` to reach it would drag ``modal`` into ``sys.modules`` and
break the dry-run gate's Anti-Pattern-6 assertion for the whole session.
"""

from __future__ import annotations

from typing import Any

__all__ = ["h3_render_key"]

#: Characters allowed through into a directory name. Everything else becomes ``_`` — a checkpoint
#: name is derived from a Volume path, and a separator or a ``..`` in it would escape the samples
#: directory rather than name a subdirectory of it.
_SAFE_EXTRA = "-_."


def h3_render_key(
    *, checkpoint: str, seed: int, frame_count: int, subject_ids: Any
) -> str:
    """The render's identity as ONE directory name: ``<checkpoint>_s<seed>_f<frames>_<ids>``.

    Every axis the sibling sample configs differ on appears here. That is the whole contract; see
    the module docstring for what a missing axis costs.

    ``subject_ids`` ORDER is preserved, never sorted: D-10-REFORDER makes a reordered reference set
    a genuinely different request (it fixes the ``<Picture i>`` labels AND advances the shared
    rotary clock), so ``A-029`` and ``029-A`` must not collapse into one directory.

    Deliberately human-readable rather than a hash — this name is what an operator reads off a
    ``modal volume ls``, and ``checkpoint-step-3000_s42_f22_A-029`` says what the grid is without
    opening it.

    Args:
        checkpoint: The resolved checkpoint directory NAME (``find_latest``'s per-step dir).
        seed: The render seed — both columns use it, so it identifies the pair.
        frame_count: ``validation.frame_count``; the LENGTH axis of the eval matrix.
        subject_ids: The reference condition in D-10-REFORDER order; the REFERENCE axis.

    Returns:
        A filesystem-safe directory name containing no path separator.
    """
    ids = "-".join(_sanitize(str(s)) for s in subject_ids) or "noref"
    return f"{_sanitize(str(checkpoint))}_s{int(seed)}_f{int(frame_count)}_{ids}"


def _sanitize(value: str) -> str:
    return "".join(c if (c.isalnum() or c in _SAFE_EXTRA) else "_" for c in value)
