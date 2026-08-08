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

__all__ = ["h3_render_key", "qwen_edit_render_key"]

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


def qwen_edit_render_key(*, checkpoint: str, seed: int, control_ids: Any) -> str:
    """The ``qwen_edit`` render's identity as ONE directory name: ``<checkpoint>_s<seed>_c<ids>``.

    Family #3's answer to the same question ``h3_render_key`` answers, with a DIFFERENT axis set —
    which is the whole reason this is a second function rather than a reused one.

    **``_f<frames>`` is deliberately absent.** Qwen-Image-Edit is an image family;
    ``QWEN_EDIT_FRAMES`` is pinned to EXACTLY 1 (``conditioning/qwen_edit_geometry.py:141``), so a
    frame-count segment would be a constant masquerading as an identity axis — it would make every
    key one token longer while distinguishing nothing, and it would invite a reader to believe the
    length axis is live here. The axes that DO vary are:

      * ``checkpoint`` — and on this family that is not "the winner", it is a BAND member. §8 of the
        house method ships THREE checkpoints (JPM: 4k / 4.2k / 5k) because eval showed only nuance
        differences with no single best. Every band member renders the same held-out set at the same
        seed, so without the checkpoint in the key the band collapses into one directory and the
        second member's renders are skipped as "already done" — a grid labelled for a band that
        contains one checkpoint's pixels three times.
      * ``seed`` — both the base and the adapter column use it, so it identifies the pair.
      * ``control_ids`` — WHICH control images were fed, in slot order. This is the qwen analog of
        H3's ``subject_ids``, and it is positional for a stronger reason: slot index IS the caption's
        ``ctrl_img_N`` addressing (``conditioning/qwen_edit.py:175-177``), so a reordered control set
        is a genuinely different request, exactly as D-10-REFORDER holds for H3 references. Order is
        PRESERVED, never sorted.

    **The A/B prompt mode is deliberately NOT in the key**, and this is the half of
    ``expected_qwen_edit_render_key``'s two-part warning that is easiest to get backwards. §8
    requires the SAME held-out input rendered under (A) a style-only prompt and (B) a content-named
    prompt **side by side** — they are one comparison, not two requests, and they are produced by one
    pass over the held-out set rather than by two sibling sample configs. So they live as SUBDIRS of
    a single render dir, exactly the way ``h3_sample`` puts ``base/`` and ``lora/`` under one key
    (``modal/fns.py:4693``). Promoting the mode into the directory name would split one grid row
    across two render dirs, and the watcher would then treat a row whose A half landed as a fully
    landed render.

    Deliberately human-readable rather than a hash, for the same reason as the H3 key: this is what
    an operator reads off a ``modal volume ls``.

    Args:
        checkpoint: The resolved checkpoint directory NAME — one BAND member, not "the best".
        seed: The render seed.
        control_ids: The control-image condition in slot order; the CONDITIONING axis. May be empty
            (an unconditioned probe), which keys as ``noctrl``.

    Returns:
        A filesystem-safe directory name containing no path separator.
    """
    ids = "-".join(_sanitize(str(s)) for s in control_ids) or "noctrl"
    return f"{_sanitize(str(checkpoint))}_s{int(seed)}_c{ids}"


def _sanitize(value: str) -> str:
    return "".join(c if (c.isalnum() or c in _SAFE_EXTRA) else "_" for c in value)
