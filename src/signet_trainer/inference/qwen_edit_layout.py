"""inference.qwen_edit_layout — the ``qwen_edit`` RENDER SURFACE: A/B prompt modes, the checkpoint
BAND, and the ordered grid columns those two produce.

This is family #3's analog of ``inference/multi_condition.py``'s ``plan_multi_frame_columns``: a
PURE planner that derives the grid's column list from declared inputs, so the Modal render branch
iterates a list it did not invent and ``inference/grid.py`` renders a column set it did not guess.
``multi_condition`` itself cannot be reused — it is LTX-shaped end to end (``TIME_SCALE = 8``,
``VideoConditionByLatentIndex``, ``(latent_idx, strength)`` items) and Qwen-Image-Edit has no latent
frame index and no per-item strength dial.

Two requirements from §8 of ``QWEN-CHAINED-EDIT-METHOD.md`` drive every decision in this file, and
both are requirements about the LAYOUT rather than about the sampler:

**§8 A/B prompt modes.** *"A/B prompt modes on every held-out input: (A) style-only 'reimagine the
reference icon…' and (B) content-named 'reimagine the train icon…'. Same input, two prompts, side by
side every N steps. This is how the trace-vs-reinterpret behavior gets read."* The two modes are one
comparison, not two requests — so they are adjacent COLUMNS of one row, produced by one pass over
the held-out set, and (see ``render_key.qwen_edit_render_key``) they are SUBDIRS of one render
directory rather than two render identities.

**§8 checkpoint band.** *"Checkpoint selection = a band, not a winner. JPM's usable band was
~3600–4200 … and the shipped deliverable was three checkpoints (4k / 4.2k / 5k) handed to the
client's creative team, because eval showed only nuance differences with no single winner. A trainer
should support exporting a band."* :class:`CheckpointBand` therefore has no ``best`` field, no
``winner`` field and no ordering by loss — it is an ordered set of members, and every consumer that
wants "the deliverable" gets all of them. A layout that modelled a single best checkpoint would make
the band expressible only as three separate renders that nothing joins.

**§8 convergence check.** *"base-vs-LoRA divergence at ~200–250 steps, not a step threshold and not
loss. If the sample is essentially the base render, it isn't converging — intervene. Read samples,
not loss."* That is why ``plan_qwen_edit_columns`` emits a BASE column pair by default and marks it
as the ``.control`` divergence reference: the base render is not decoration here, it is the thing
the PASS/FAIL judgment is made against.

Import tier: **stdlib only, no package side effects** — same tier as ``render_key`` and
``samples_layout``, and for the same reason. The watcher and the CPU grid tests import this without
the modal SDK or torch loaded, and a test that dragged either into ``sys.modules`` would break the
dry-run gate's Anti-Pattern-6 assertion for the whole session.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "QWEN_EDIT_PROMPT_MODES",
    "BASE_COLUMN_TOKEN",
    "CheckpointBand",
    "QwenEditColumn",
    "QwenEditPromptMode",
    "band_render_keys",
    "plan_qwen_edit_columns",
    "render_qwen_edit_sample",
]

#: Reserved column token for the un-adaptered BASE render. A band member that sanitises to this
#: token is REFUSED (see :func:`plan_qwen_edit_columns`) rather than silently overwriting the base
#: column's row key — a checkpoint dir literally named ``base`` is unlikely, and a grid in which the
#: divergence reference has been replaced by an adapter render while still labelled BASE is exactly
#: the silent mislabel this module's sibling (``samples_layout``) exists to prevent.
BASE_COLUMN_TOKEN = "base"

#: Characters allowed through into a row key / column token. Mirrors ``render_key._SAFE_EXTRA``'s
#: intent (no path separator can survive) but is STRICTER: ``.`` is excluded because a row key is a
#: dict key read by ``grid.py``, and a dotted key invites an attribute-access reading of what is
#: really a mapping lookup. Kept local rather than imported so ``render_key``'s directory-name rule
#: and this module's dict-key rule can diverge without one silently re-defining the other.
_SAFE_TOKEN_EXTRA = "-_"


@dataclass(frozen=True)
class QwenEditPromptMode:
    """One of §8's two prompt modes, as a data record rather than a string literal at a call site.

    ``id``            — the stable token used in row keys and in the render dir's per-mode SUBDIR
                        name. Never rendered to the operator; never parsed back into meaning.
    ``label``         — what the grid column header says.
    ``row_prompt_key`` — the row-dict key carrying THIS mode's prompt text, so the gallery can print
                        the exact prompt each column was rendered under. §8's read is
                        trace-vs-reinterpret, and it is unreadable if the two prompts are not both
                        visible on the row.
    ``description``   — §8's own gloss, carried for the grid's criteria block.
    """

    id: str
    label: str
    row_prompt_key: str
    description: str


#: The two modes, in §8's order (A then B). A TUPLE, not a list, and not a config field: the pair is
#: the method's read protocol, not a tunable. A third mode would change what the grid means, and the
#: right way to add one is to change this tuple deliberately — with the tests that pin the column
#: order failing first.
QWEN_EDIT_PROMPT_MODES: tuple[QwenEditPromptMode, ...] = (
    QwenEditPromptMode(
        id="a_style",
        label="A · style-only",
        row_prompt_key="prompt_a",
        description="style-only prompt ('reimagine the reference icon…') — the content word is "
        "withheld, so the adapter must carry the subject from the control image",
    ),
    QwenEditPromptMode(
        id="b_content",
        label="B · content-named",
        row_prompt_key="prompt_b",
        description="content-named prompt ('reimagine the train icon…') — the subject is named in "
        "text as well as shown, which is the condition under which tracing is cheapest",
    ),
)


@dataclass(frozen=True)
class CheckpointBand:
    """An ordered BAND of checkpoints — §8's deliverable unit. **There is no winner field.**

    JPM shipped three checkpoints because eval showed only nuance differences between them; the band
    IS the answer, and the creative team picks per asset. A ``best: str`` attribute here would let
    every downstream consumer quietly collapse the band back to one render, which is the behaviour
    §8 explicitly rejects. If a caller genuinely needs one member it takes ``band.members[i]`` and
    says which index and why at the call site.

    ``members`` is ordered and order is PRESERVED (never sorted): a band is read and handed over in
    its own order, and members are not always zero-padded step numbers, so a sort could reorder
    ``["4000", "4200", "5000"]`` into something that reads as a different band.

    A single-member band is LEGAL — a mid-training grid renders one checkpoint — but it is still a
    band, so nothing downstream needs a special case for "the winner". An EMPTY band is refused: it
    would render a grid with a base column and nothing to compare it to, which passes every shape
    check and tells the operator nothing.
    """

    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError(
                "CheckpointBand is EMPTY. A band is 1..N checkpoints (§8 of "
                "QWEN-CHAINED-EDIT-METHOD.md ships three); zero members renders a grid whose only "
                "column is the base control — a valid-looking artifact that answers nothing. Pass "
                "the resolved checkpoint dir names."
            )
        seen: dict[str, int] = {}
        for i, name in enumerate(self.members):
            if not str(name).strip():
                raise ValueError(
                    f"CheckpointBand member at index {i} is empty/whitespace. Every member must be "
                    "a resolved checkpoint directory NAME."
                )
            if name in seen:
                raise ValueError(
                    f"CheckpointBand member {name!r} appears twice (indices {seen[name]} and {i}). "
                    "A repeated member means the band was assembled wrongly; de-duplicating it "
                    "silently would ship a shorter band under its original label."
                )
            seen[name] = i

    @classmethod
    def of(cls, checkpoints: Iterable[str]) -> "CheckpointBand":
        """Build a band from any iterable of checkpoint names, order preserved."""
        return cls(members=tuple(str(c) for c in checkpoints))

    @property
    def size(self) -> int:
        return len(self.members)

    def describe(self) -> str:
        """One-liner for the grid's params banner: ``band of 3: 4000 / 4200 / 5000``."""
        return f"band of {self.size}: " + " / ".join(self.members)


@dataclass(frozen=True)
class QwenEditColumn:
    """One output column of the qwen_edit grid — a (checkpoint, prompt mode) pair.

    ``label``       — the column header the operator reads.
    ``row_key``     — the row-dict key whose value is this cell's image path
                      (``grid.write_qwen_edit_gallery`` reads ``row.get(col.row_key, "")``,
                      mirroring ``_multi_frame_block``'s ``row.get("n2_mp4", "")``).
    ``checkpoint``  — the band member this column renders, or ``None`` for the BASE column.
    ``mode``        — which §8 prompt mode this column was rendered under.
    ``control``     — True for the BASE columns: they are the §8 convergence reference, and the grid
                      marks them so an operator cannot mistake the divergence baseline for a result.
    ``render_subdir`` — where this cell's file lives INSIDE the render dir keyed by
                      ``render_key.qwen_edit_render_key`` (``lora/<ckpt>/<mode id>`` or
                      ``base/<mode id>``). The mode is a subdir precisely because it is NOT part of
                      the render identity; see that function's docstring.
    """

    label: str
    row_key: str
    checkpoint: str | None
    mode: QwenEditPromptMode
    control: bool
    render_subdir: str

    @property
    def is_base(self) -> bool:
        return self.checkpoint is None


def _token(value: str) -> str:
    """Reduce a checkpoint name to a row-key token: only ``[A-Za-z0-9_-]`` survives."""
    return "".join(c if (c.isalnum() or c in _SAFE_TOKEN_EXTRA) else "_" for c in str(value))


def plan_qwen_edit_columns(
    band: CheckpointBand, *, include_base: bool = True
) -> list[QwenEditColumn]:
    """Derive the ordered qwen_edit grid columns from a BAND and §8's two prompt modes (pure/CPU).

    Column order, and why it is this order::

        [ BASE · A | BASE · B ] [ ckpt1 · A | ckpt1 · B ] [ ckpt2 · A | ckpt2 · B ] ...

    Grouped by CHECKPOINT with the mode inner, so **A sits immediately beside B for the same
    weights** — §8's read is "same input, two prompts, side by side", and it is the A-vs-B delta at a
    FIXED checkpoint that exposes trace-vs-reinterpret. Grouping the other way (all A columns, then
    all B) would put the two halves of every comparison at opposite ends of a row that is already
    ``2 * (band size + 1)`` cells wide, and the read the method asks for would become a scan.

    The BASE pair comes FIRST and carries ``control=True``: §8's convergence check is
    base-vs-LoRA divergence, so the baseline is the leftmost thing on the row and is marked as a
    control rather than presented as one more result. ``include_base=False`` exists for the shipped
    deliverable grid (the band handed to a creative team, where the base render is not part of the
    product); it is never the default, because for an in-training grid a missing baseline turns
    "isn't converging" into an unanswerable question.

    Raises:
        ValueError: if a band member's row-key token collides with the reserved ``base`` token, or
            if two members collide with each other after tokenisation. Both cases would silently
            overwrite one column's cell with another's — the same class of failure as a coarse render
            key, one layer up: a grid labelled for a checkpoint it does not contain.
    """
    columns: list[QwenEditColumn] = []

    if include_base:
        for mode in QWEN_EDIT_PROMPT_MODES:
            columns.append(
                QwenEditColumn(
                    label=f"BASE · {mode.label}",
                    row_key=f"{BASE_COLUMN_TOKEN}__{mode.id}_img",
                    checkpoint=None,
                    mode=mode,
                    control=True,
                    render_subdir=f"{BASE_COLUMN_TOKEN}/{mode.id}",
                )
            )

    seen_tokens: dict[str, str] = {}
    for checkpoint in band.members:
        token = _token(checkpoint)
        if token == BASE_COLUMN_TOKEN:
            raise ValueError(
                f"band member {checkpoint!r} tokenises to the reserved column token "
                f"{BASE_COLUMN_TOKEN!r}, which names the un-adaptered convergence reference. Its "
                "cells would overwrite the BASE column's and the grid would show an adapter render "
                "under a BASE label — the §8 divergence check read against itself. Rename the "
                "checkpoint directory."
            )
        if token in seen_tokens:
            raise ValueError(
                f"band members {seen_tokens[token]!r} and {checkpoint!r} both tokenise to "
                f"{token!r}, so they would share row keys and one member's cells would overwrite "
                "the other's — a grid labelled for a band member it does not contain. Give the "
                "checkpoints names that differ in more than punctuation."
            )
        seen_tokens[token] = checkpoint

        for mode in QWEN_EDIT_PROMPT_MODES:
            columns.append(
                QwenEditColumn(
                    label=f"{checkpoint} · {mode.label}",
                    row_key=f"{token}__{mode.id}_img",
                    checkpoint=checkpoint,
                    mode=mode,
                    control=False,
                    render_subdir=f"lora/{token}/{mode.id}",
                )
            )

    return columns


def render_qwen_edit_sample(**kwargs: Any) -> str:
    """DECLARED STUB — the qwen_edit GENERATE call. The layout around it is real; this is not.

    Everything this slice ships is the plumbing that a render fills: the render identity
    (``render_key.qwen_edit_render_key``), the landed-check (``samples_layout.landed_render_ids``),
    the band (:class:`CheckpointBand`), the column list (:func:`plan_qwen_edit_columns`) and the HTML
    surface (``grid.write_qwen_edit_gallery``). All of that is CPU-testable and tested. The
    generate call is not, and is not guessed.

    **What lands it**, in dependency order — each is itself a named symbol that raises today:

      1. ``models/qwen_edit_loader.load_qwen_edit_transformer`` + ``assert_qwen_edit_arch`` +
         ``quantize_qwen_edit`` — no transformer can be loaded or arch-gated yet.
      2. ``prep/qwen_edit_encode`` — does not exist, and is itself BLOCKED on a ruling: ai-toolkit
         computes its control-image aspect ratio as ``H/W`` where diffusers uses ``W/H``, so every
         non-square control is resized to a transposed canvas. Bug-compatible / diffusers-correct /
         square-only-gate is an open decision, not a detail to pick while writing a sampler.
      3. The §8 INFERENCE SETTINGS, which are a documented trap and not defaults:
         ``steps 30``, ``true_cfg 4.0`` with CFGNorm ON, **static shift 3.0**
         (``use_dynamic_shifting=False``), LoRA strength ``1.0``, and the reference fed to BOTH the
         positive and negative encode. ``QwenImageEditPlusModel.get_generation_pipeline()`` rebuilds
         a fresh default-shift scheduler, so ``pipeline.scheduler`` must be overridden AFTER the
         pipeline is built and passed into ``generate_images(pipeline=...)``. ai-toolkit's default
         dynamic shift (~1.6–2.5) is far weaker than the shipped ComfyUI ``ModelSamplingAuraFlow``
         shift, and §8 names the diagnosis tell explicitly: *training samples look fine but renders
         are muddy → it's inference settings, not the model*. A sampler written without these is the
         failure that looks like a bad adapter.

    ⛔ ITEM 3 CARRIES AN UNRESOLVED CONTRACT CONFLICT, and whoever lands this must settle it FIRST.
    ``tests/test_no_wan_params.py:31`` bans the bare token ``shift`` from executable code in every
    ``*.py`` under ``inference/`` — a guard written to keep the Wan-tuned sampling set out of **LTX**
    paths (its docstring, :6-8), whose directory-wide scope now also covers a family where that same
    setting is MANDATORY at a different value. This module and ``grid.py`` therefore name it in prose
    only (the guard's own ``test_a_wan_token_only_in_a_docstring_is_not_flagged`` sanctions exactly
    that), and ``grid.py``'s banner OMITS the field rather than weakening the guard or renaming the
    field to slip past the scanner. Narrowing the guard to LTX modules is the likely resolution, but
    it is a RULING on a shipped money-safe guard, not an edit to make in passing.

    Raising here rather than returning a placeholder path is the same money-safe direction
    ``samples_layout`` takes: a placeholder would let a grid build, look complete, and be judged.
    """
    raise NotImplementedError(
        "[qwen_edit] render_qwen_edit_sample is a DECLARED STUB — no Qwen generate call exists. "
        f"Called with {sorted(kwargs)}. The render SURFACE around it is real and tested (render key, "
        "landed-check, CheckpointBand, plan_qwen_edit_columns, write_qwen_edit_gallery); only the "
        "generate call is missing, and it needs, in order: (1) models/qwen_edit_loader's "
        "load_qwen_edit_transformer / assert_qwen_edit_arch / quantize_qwen_edit; (2) "
        "prep/qwen_edit_encode, itself blocked on the control-image aspect-ratio ruling "
        "(ai-toolkit computes H/W where diffusers computes W/H, so non-square controls transpose); "
        "(3) the METHOD §8 inference settings — steps 30, true_cfg 4.0 + CFGNorm, the STATIC "
        "scheduler reparameterisation value overridden onto pipeline.scheduler AFTER "
        "get_generation_pipeline() rebuilds it, LoRA strength 1.0, reference into both pos and neg "
        "encode. Item (3) is itself BLOCKED BY A CONTRACT CONFLICT with tests/test_no_wan_params.py "
        "— read this function's docstring before writing the sampler. Returning a placeholder path "
        "instead would let a grid build, look complete, and be judged."
    )


def band_render_keys(
    band: CheckpointBand, *, seed: int, control_ids: Sequence[str]
) -> list[str]:
    """Convenience: the render key of every band member, delegating to ``samples_layout``.

    Function-local import purely to keep the two modules' dependency direction one-way at module
    scope (``samples_layout`` imports ``render_key``; nothing imports ``qwen_edit_layout``), so that
    adding a planner can never widen the watcher's import closure.
    """
    from signet_trainer.inference.samples_layout import (  # noqa: PLC0415
        expected_qwen_edit_band_keys,
    )

    return expected_qwen_edit_band_keys(
        checkpoints=band.members, seed=seed, control_ids=control_ids
    )
