"""inference.samples_layout — WHERE a render's artifacts land, keyed by model FAMILY.

The parallel-inference watcher dispatches a metered render and then asks the Volume "did it land?".
That question has a different answer per family, and getting it wrong is not a cosmetic bug — it is
the phantom-spend shape (``KNOWLEDGE.md`` ``watcher`` ``PYTHONPATH`` ``phantom-spend``):

  * the ledger booking (``append_spend``, now inside the entrypoint gate itself — issue #37 findings
    1/2) fires on every successful dispatch and there is NO refund path.
  * ``rendered`` is success-gated.

So a watcher that looks for landed artifacts in the WRONG directory sees every render as FAILED,
never marks the step rendered, and re-dispatches on the next poll — booking the full per-render
estimate each time. On the H3 config that is $3.28 every 240 s (~$49/hr) against renders that may
have succeeded, which would trip ``session_cap_usd`` and halt a healthy campaign.

``h3_sample`` writes to ``<output_dir>/samples_h3/<render key>/lora/`` while the LTX ``sample``
writes to ``<output_dir>/samples/<UTC stamp>/``. BOTH axes differ — the subdirectory AND the naming
scheme — so a fix that only re-pointed the subdirectory would still mis-verify every H3 render.

⚠ THE BASE COLUMN IS NOT UNDER THE CHECKPOINT-KEYED DIR (#12). It depends only on the pipe
parameters — never the checkpoint — so it lives as a SIBLING, ``<output_dir>/samples_h3/base/<base
render key>/``, shared across every checkpoint that renders the same request. A watcher's landed-
check still only asks about the checkpoint-keyed ``lora`` half (that IS the per-checkpoint signal);
the grid assembler is what joins the shared base column back onto each checkpoint's row — see
``expected_h3_base_render_key`` and ``scripts/_h3_grid_serve.py``.

⛔ FAMILY-AWARE, NEVER H3-HARDCODED. ``samples_subdir`` RAISES on an unknown family rather than
falling back to ``"samples"``: a silent default is exactly how the LTX value would be re-applied to
a future family and re-open this same hole. The LTX answers are unchanged and pinned by test.

⛔ issue #45 PR-4 — LTX IS ALSO MODE-AWARE, NOT JUST FAMILY-AWARE. ``family == "ltx"`` alone picks
the WRONG root for five of ``modal/fns.py``'s ``sample()`` branches: ``conditioning.mode`` selects
between ``samples`` (mode ``"none"``, the historical default), ``samples_inpaint``, ``samples_a2v``,
``samples_single_frame``, ``samples_multi_frame``, ``samples_ic_lora`` and ``samples_ic_lora_baseline``
(the last selected by ``mode == "ic_lora"`` + ``validation.two_stage_upscale`` together — see
``layout_mode``). A watcher bound to the family-only root lists an EMPTY or WRONG directory for any
non-default mode, which reads as "nothing landed" and re-dispatches forever — the same phantom-spend
shape, wearing a mode disguise instead of a checkpoint-vs-samples-dir one.

``inpaint`` and ``audio_to_video`` additionally write STEP-KEYED render dirs
(``samples_inpaint/<stem>/step_<N>.mp4``, ``samples_a2v/<stem>/step_<N>.mp4``) rather than a fresh
UTC-stamped dir per render: the ``<stem>`` directory is STABLE across every checkpoint dispatched
against this ``output_dir``/mode — only a new ``step_<N>.mp4`` FILE lands inside it per render. A
watcher that treated the stem's mere existence as "landed" would see step 600 and step 1200 as the
identical render the moment the FIRST one committed. ``landed_render_ids`` therefore returns
step-BEARING ids (``"<stem>/step_<N>"``) for these two modes, never the stem alone — see
``STEP_KEYED_LTX_MODES``.

Import tier: **stdlib only, no package side effects** — the watcher imports this WITHOUT the modal
SDK loaded, and a test that dragged ``modal`` into ``sys.modules`` would break the dry-run gate's
Anti-Pattern-6 assertion for the whole session (the same reason ``inference/render_key.py`` exists).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from signet_trainer.inference.render_key import h3_base_render_key, h3_render_key, qwen_edit_render_key

__all__ = [
    "SAMPLES_SUBDIR_BY_FAMILY",
    "SAMPLES_SUBDIR_BY_LTX_MODE",
    "STEP_KEYED_LTX_MODES",
    "committed_clip_names",
    "expected_h3_base_render_key",
    "expected_h3_render_key",
    "expected_qwen_edit_band_keys",
    "expected_qwen_edit_render_key",
    "landed_render_ids",
    "layout_mode",
    "samples_root",
    "samples_subdir",
]

#: The render root each family's sample fn actually writes, relative to ``output_dir``.
#: Transcribed from ``modal/fns.py`` (``"samples_h3" / h3_render_key(...)`` for H3; the LTX
#: ``sample`` branch's ``samples*`` dirs) — NOT guessed. ``samples_text_to_video`` is never written
#: by anything (training-review §2 WR-08); do not add it here on the strength of a mode name.
#:
#: ``qwen_edit`` (family #3) was RESERVED here before its render surface existed, and is now LIVE:
#: ``expected_qwen_edit_render_key`` and the ``landed_render_ids`` arm below are real. The reservation
#: did its job — ``samples_subdir`` raised on the unknown family, so nobody could reach for the
#: shortest fix (point the watcher at ``"samples"``) and re-open the phantom-spend hole this module
#: exists to close.
SAMPLES_SUBDIR_BY_FAMILY = {
    "ltx": "samples",
    "h3": "samples_h3",
    "qwen_edit": "samples_qwen_edit",
}

#: issue #45 PR-4 — every LTX ``conditioning.mode`` that writes somewhere OTHER than the family
#: default (``SAMPLES_SUBDIR_BY_FAMILY["ltx"]`` == ``"samples"``, mode ``"none"``). Transcribed from
#: ``modal/fns.py``'s per-mode ``sample()`` branches (never guessed): ``inpaint``, ``audio_to_video``,
#: ``single_frame``, ``multi_frame``, ``ic_lora`` and ``ic_lora_baseline`` each ``mkdir`` their own
#: named root. ``mode == "none"`` (and any mode not listed here) falls through to the family default
#: in ``samples_subdir`` — it is deliberately absent from this table rather than redundantly mapped.
SAMPLES_SUBDIR_BY_LTX_MODE = {
    "inpaint": "samples_inpaint",
    "audio_to_video": "samples_a2v",
    "single_frame": "samples_single_frame",
    "multi_frame": "samples_multi_frame",
    "ic_lora": "samples_ic_lora",
    "ic_lora_baseline": "samples_ic_lora_baseline",
}

#: issue #45 PR-4 — the subset of ``SAMPLES_SUBDIR_BY_LTX_MODE`` whose render dir is a STABLE
#: ``<stem>`` directory (named after the held-out test clip, ``modal/fns.py``'s ``slug(stem)``)
#: rather than a fresh UTC-stamped dir per render. Every checkpoint dispatched against the same
#: output_dir/mode writes into the SAME stem dir, adding one ``step_<N>.mp4`` — so "did THIS render
#: land" can never be answered by the stem's existence alone (see ``landed_render_ids``).
STEP_KEYED_LTX_MODES = frozenset({"inpaint", "audio_to_video"})

#: A LTX render dir is a UTC wall-clock stamp: ``samples/20260805T154357Z/``.
_LTX_STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")

#: A step-keyed LTX render FILE (``modal/fns.py``'s inpaint/a2v branches: ``f"step_{step}.mp4"``).
#: Matched against the LAST path segment only — ``landed_render_ids`` anchors the STEM to the first
#: segment after the render root, so this never needs to see the stem itself.
_LTX_STEP_FILE_RE = re.compile(r"^step_(\d+)\.mp4$")

#: An H3 render dir is an IDENTITY key:
#: ``<checkpoint>_s<seed>_f<frames>_w<width>_h<height>_n<steps>_<ids>``. Anchored on the
#: ``_s<d>_f<d>_w<d>_h<d>_n<d>_`` tail so a greedy checkpoint head cannot mis-split (checkpoint dir
#: names keep ``-`` and ``.`` but carry no underscore). Mirrors ``scripts/_h3_grid_serve.py::_KEY_RE``.
#: #22 finding 5 widened this from ``_s<d>_f<d>_`` alone — a resolution or step-count probe that
#: differed ONLY on an axis outside the OLD key would have resumed the previous geometry's clips
#: under the new banner.
#:
#: ⛔ The BASE key (``inference.render_key.h3_base_render_key``, no checkpoint segment) never matches
#: this pattern — it has no ``<checkpoint>_`` head, so ``landed_render_ids`` and ``parse_render_key``
#: (``scripts/_h3_grid_serve.py``) both skip it automatically rather than needing a special case.
_H3_KEY_RE = re.compile(
    r"^(?P<ckpt>.+)_s(?P<seed>\d+)_f(?P<frames>\d+)_w(?P<width>\d+)_h(?P<height>\d+)_"
    r"n(?P<steps>\d+)_(?P<ids>.+)$"
)


def samples_subdir(family: str, mode: str | None = None) -> str:
    """The render subdirectory ``family`` (and, for LTX, ``mode``) writes under ``output_dir``.

    ``mode`` is ``conditioning.mode`` (issue #45 PR-4) — only consulted when ``family == "ltx"``
    AND it names an entry in ``SAMPLES_SUBDIR_BY_LTX_MODE``. ``None`` (the default) and mode
    ``"none"`` both fall through to the family default, so every call site that predates PR-4 (and
    every non-LTX family, which has no mode axis at all) keeps its exact prior answer.

    Raises:
        ValueError: on an unrecognised family. Fail-loud is the money-safe direction — a default of
            ``"samples"`` would make a new family silently mis-verify every render and re-dispatch
            it, which is the phantom-spend failure this module exists to prevent.
    """
    if family == "ltx" and mode in SAMPLES_SUBDIR_BY_LTX_MODE:
        return SAMPLES_SUBDIR_BY_LTX_MODE[mode]
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


def samples_root(output_dir: str, family: str, mode: str | None = None) -> str:
    """The Volume-relative render root, e.g. ``outputs/h3_embe_r1/samples_h3``."""
    return f"{output_dir.rstrip('/')}/{samples_subdir(family, mode)}"


def layout_mode(*, conditioning_mode: str, two_stage_upscale: bool = False) -> str:
    """The LTX layout key a render will ACTUALLY use — mirrors ``modal/fns.py``'s own dispatch so
    the watcher and the render can never key differently (issue #45 PR-4).

    ``conditioning.mode == "ic_lora"`` splits into two DIFFERENT render roots depending on
    ``validation.two_stage_upscale``: the two-stage path is the separately-gated
    ``ic_lora_baseline`` branch (``samples_ic_lora_baseline``), the single-stage path is
    ``samples_ic_lora``. Every other mode maps to itself; there is nothing else to resolve.
    """
    if conditioning_mode == "ic_lora" and two_stage_upscale:
        return "ic_lora_baseline"
    return conditioning_mode


def expected_h3_render_key(
    *,
    checkpoint: str,
    seed: int,
    frame_count: int,
    width: int,
    height: int,
    num_inference_steps: int,
    subject_ids: Any,
) -> str:
    """The directory ``h3_sample`` WILL write for this (checkpoint, seed, frames, geometry, refs)
    request.

    The watcher's "did this render land?" check must key on exactly what the render keys on, or it
    mis-attributes one sample config's output to another. All five H3 sample configs share
    ``output_dir``, ``seed`` and their prompt set, so they write identical clip FILENAMES; only the
    render-dir identity separates them. A watcher checking a coarser key (checkpoint alone, or the
    pre-#22-finding-5 key that omitted ``width`` / ``height`` / ``num_inference_steps``) would accept
    the ``A+029`` render — or a render at a stale resolution — as proof the requested render landed.

    Delegates to ``inference.render_key.h3_render_key`` — the render's own function, never a
    re-implementation, so the two cannot drift.
    """
    return h3_render_key(
        checkpoint=checkpoint,
        seed=seed,
        frame_count=frame_count,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        subject_ids=subject_ids,
    )


def expected_h3_base_render_key(
    *,
    seed: int,
    frame_count: int,
    width: int,
    height: int,
    num_inference_steps: int,
    subject_ids: Any,
) -> str:
    """The directory ``h3_sample`` WILL write the BASE column to — the same request MINUS the
    checkpoint (#12).

    Delegates to ``inference.render_key.h3_base_render_key`` for the identical never-drift reason
    ``expected_h3_render_key`` delegates to ``h3_render_key``. A caller resolving "did the base clip
    for THIS request already land" (the grid assembler, merging one shared base column across every
    checkpoint that shares these five axes) reads this key, never ``expected_h3_render_key`` with the
    checkpoint segment discarded by hand.
    """
    return h3_base_render_key(
        seed=seed,
        frame_count=frame_count,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        subject_ids=subject_ids,
    )


def expected_qwen_edit_render_key(
    *, checkpoint: str, seed: int, control_ids: Any
) -> str:
    """The directory a ``qwen_edit`` render WILL write for this (checkpoint, seed, controls) request.

    Delegates to ``inference.render_key.qwen_edit_render_key`` — the render's own function, never a
    re-implementation, so the watcher's expectation and the directory the render creates cannot
    drift. Identical discipline to ``expected_h3_render_key``.

    The stub this replaces named two things that had to be settled TOGETHER; both are settled and
    both are recorded in ``qwen_edit_render_key``'s docstring:

      1. **No ``_f<frames>`` segment.** F is pinned to exactly 1 on this family, so a frame segment
         would be a constant wearing an identity axis's clothes.
      2. **The A/B prompt mode is NOT an identity axis.** §8's style-only / content-named pair is one
         side-by-side comparison produced by one pass, so the two modes are SUBDIRS of a single
         render dir (the ``base/`` + ``lora/`` shape ``h3_sample`` already uses), not two dirs. Had
         the mode gone into the key, a row whose A half landed would read to the watcher as a
         complete render.

    ``_QWEN_EDIT_KEY_RE`` and the ``landed_render_ids`` arm below were updated in the same commit as
    this function, which is what the stub required.
    """
    return qwen_edit_render_key(checkpoint=checkpoint, seed=seed, control_ids=control_ids)


def expected_qwen_edit_band_keys(
    *, checkpoints: Sequence[str], seed: int, control_ids: Any
) -> list[str]:
    """Every render key a checkpoint BAND produces, in band order — the band's landed-check unit.

    §8 of the house method: *"Checkpoint selection = a band, not a winner … the shipped deliverable
    was three checkpoints … A trainer should support exporting a band."* The watcher's question on
    this family is therefore **"did the whole band land?"**, and asking it one checkpoint at a time
    is how a partially-rendered band gets shipped: the three members share ``output_dir``, ``seed``,
    the held-out control set and their prompt pair, so they write identical FILENAMES and differ only
    in the render-dir identity. A caller that checked the first member alone would accept it as proof
    the other two rendered.

    Order is the BAND's order, preserved — ``["4000", "4200", "5000"]`` is how the band is read and
    handed over, and a sorted copy would silently reorder a band whose members are named by anything
    other than a zero-padded step.

    Returns one key per member; duplicates in ``checkpoints`` are refused rather than de-duplicated,
    because a repeated member means the band was assembled wrongly and a silent de-dupe would ship a
    two-checkpoint band under a three-checkpoint label.
    """
    members = list(checkpoints)
    if not members:
        raise ValueError(
            "expected_qwen_edit_band_keys got an EMPTY band. A band is 1..N checkpoints (§8 ships "
            "three); zero members means the caller resolved no checkpoint at all, and returning [] "
            "would read to the watcher as 'every render landed' — the inverse of the truth."
        )
    seen: dict[str, int] = {}
    for i, name in enumerate(members):
        if name in seen:
            raise ValueError(
                f"checkpoint {name!r} appears twice in the band (positions {seen[name]} and {i}). "
                "A repeated member means the band was assembled wrongly; de-duplicating it silently "
                "would ship a shorter band under its original label."
            )
        seen[name] = i
    return [
        expected_qwen_edit_render_key(checkpoint=name, seed=seed, control_ids=control_ids)
        for name in members
    ]


#: A ``qwen_edit`` render dir is an IDENTITY key: ``<checkpoint>_s<seed>_c<control ids>``. Anchored
#: on the ``_s<d>_c`` tail exactly as ``_H3_KEY_RE`` anchors on ``_s<d>_f<d>_``.
#:
#: ⚠ The ``_c`` marker is what keeps the two families' patterns MUTUALLY EXCLUSIVE, and that matters
#: because ``landed_render_ids`` takes the family from config: an H3 key carries ``_s<d>_f<d>_`` and
#: never ``_s<d>_c``, and a qwen key carries ``_s<d>_c`` and never ``_s<d>_f<d>_``. Pointing either
#: family's parse at the other family's listing therefore yields ``[]`` — visibly nothing landed —
#: rather than a plausible partial list. Pinned both directions by test.
#:
#: Like ``_H3_KEY_RE`` this is a RECOGNISER, not a parser: ``landed_render_ids`` adds the whole
#: directory NAME, never a capture group, so a control id that happened to contain ``_s12_c`` would
#: mis-split the groups harmlessly and still be recognised.
_QWEN_EDIT_KEY_RE = re.compile(r"^(?P<ckpt>.+)_s(?P<seed>\d+)_c(?P<ids>.+)$")


def landed_render_ids(listing: str, family: str, mode: str | None = None) -> list[str]:
    """Every committed render identity visible in a ``modal volume ls`` listing, sorted.

    ``listing`` is raw CLI stdout (the watcher shells out; this stays a pure string function so it
    is testable with zero Modal and zero spend).

      * ``ltx`` (``mode`` in ``STEP_KEYED_LTX_MODES``, issue #45 PR-4) -> STEP-BEARING stem ids
        (``"<stem>/step_<N>"``), never the stem alone — the stem dir is STABLE across every
        checkpoint dispatched against this output_dir/mode (``modal/fns.py``'s inpaint/a2v
        branches add one ``step_<N>.mp4`` per render into the SAME dir), so a bare stem id would
        make a step-600 render and a step-1200 render indistinguishable the moment the first
        landed. ``modal volume ls`` has no recursive listing (confirmed: no ``-r``/``--recursive``
        flag), so the caller is expected to fold in each stem's own one-level listing — this
        function tolerates that listing arriving at ANY depth past the render root, so long as
        each committed file's line ends in its ``step_<N>.mp4`` filename; a bare stem-only line (no
        step file visible yet) contributes no id, never a plausible-but-wrong stem-only one.
      * ``ltx`` (every other mode, including ``"none"``) -> the UTC stamps, the historical
        behaviour, byte-for-byte.
      * ``h3``  -> the identity keys, so two renders differing ONLY in reference condition or frame
        count are two distinct ids rather than one.
      * ``qwen_edit`` -> the identity keys, so two BAND members differing only in checkpoint are two
        distinct ids rather than one. The two families' patterns are mutually exclusive by
        construction (see ``_QWEN_EDIT_KEY_RE``): each family's parse returns ``[]`` against the
        other's listing — visibly nothing, never a plausible partial list.

    An unknown family raises via ``samples_subdir`` BEFORE any parse: a silent fall-through to the H3
    regex would return ``[]`` for a directory of SUCCESSFUL renders, which reads to the watcher as
    "nothing landed", never marks the step rendered, and re-books the full per-render estimate on
    every poll (KNOWLEDGE.md 'watcher phantom-spend').
    """
    subdir = samples_subdir(family, mode)  # validates the family first — unknown families raise here
    if family == "ltx" and mode in STEP_KEYED_LTX_MODES:
        found: set[str] = set()
        for raw in (listing or "").splitlines():
            line = raw.strip().rstrip("/")
            if not line:
                continue
            tail = line.split(f"{subdir}/")[-1]
            parts = [p for p in tail.split("/") if p]
            if len(parts) < 2:
                continue  # a bare stem dir, no step file visible yet — no render identity to report
            stem, fname = parts[0], parts[-1]
            m = _LTX_STEP_FILE_RE.match(fname)
            if m:
                found.add(f"{stem}/step_{m.group(1)}")
        return sorted(found)
    if family == "ltx":
        return sorted(set(_LTX_STAMP_RE.findall(listing or "")))

    key_re = _QWEN_EDIT_KEY_RE if family == "qwen_edit" else _H3_KEY_RE
    found: set[str] = set()
    for raw in (listing or "").splitlines():
        # `modal volume ls` prints Volume-relative paths (`outputs/x/samples_h3/<key>`); take the
        # segment after the render root so a checkpoint name containing `/` can never leak through.
        name = raw.strip().rstrip("/").split(f"{subdir}/")[-1].split("/")[-1]
        if name and key_re.match(name):
            found.add(name)
    return sorted(found)


def committed_clip_names(listing: str) -> list[str]:
    """Every ``*.mp4`` filename visible in a ``modal volume ls`` listing of ONE render's own column
    (``base/`` or ``lora/``) — the fine-grained PROGRESS signal a multi-hour render's stall clock
    needs (issue #45 PR-1 must-fix #1), distinct from ``landed_render_ids``'s coarse existence check.

    ``landed_render_ids`` answers "has this render's IDENTITY directory appeared at all?", which
    saturates the moment the render's first clip commits (``h3_sample`` commits per clip —
    ``checkpoints_vol.commit()`` inside ``_render()``, ``modal/fns.py``) and stays true for the whole
    rest of an up-to-6h render; it cannot tell "still writing clips" apart from "fully done", so a
    stall-freshness clock cannot safely reset on it alone. Reading one level deeper — the individual
    clip filenames inside the render's own directory — gives a signal that keeps changing for as long
    as the render keeps committing clips, which is exactly what that clock needs.

    Same string-only contract as ``landed_render_ids``: pure, zero Modal, zero spend, testable with a
    canned ``modal volume ls`` stdout.
    """
    found: set[str] = set()
    for raw in (listing or "").splitlines():
        name = raw.strip().rstrip("/").split("/")[-1]
        if name.endswith(".mp4"):
            found.add(name)
    return sorted(found)
