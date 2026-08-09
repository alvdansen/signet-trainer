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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "QWEN_EDIT_PROMPT_MODES",
    "BASE_COLUMN_TOKEN",
    "CheckpointBand",
    "QwenEditColumn",
    "QwenEditHeldOutInput",
    "QwenEditPromptMode",
    "band_render_keys",
    "plan_qwen_edit_columns",
    "plan_qwen_edit_rows",
    "qwen_edit_cell_relpath",
    "qwen_edit_control_ids",
    "qwen_edit_render_dir",
    "qwen_edit_sample_filename",
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


# --------------------------------------------------------------------------------------------------
# THE ROW SIDE: held-out inputs -> gallery rows, and THE ONE JOIN between a column and a file.
#
# ``plan_qwen_edit_columns`` names WHERE a cell's image goes (``QwenEditColumn.render_subdir``) and
# WHICH row-dict key the gallery reads it back from (``QwenEditColumn.row_key``). Until this block
# those two facts were connected by nothing: ``grid._qwen_edit_block`` reads ``row[column.row_key]``
# and the sampler would write under ``column.render_subdir``, and no code in the package made the two
# agree. ``tests/test_qwen_edit_render_surface.py`` filled the gap with a row-builder invented inside
# the test (``out/{name}__{ckpt}__{mode}.png``) — a path shape that matches no ``render_subdir``, so
# the gallery was proved against paths the render will never produce.
#
# That is the module's own documented failure class one layer up: a grid labelled for something it
# does not contain, arrived at through a valid shape. ``qwen_edit_cell_relpath`` is therefore the
# SINGLE transcription of the column->file join, and the sampler must compose its output path from it
# rather than from a second spelling of the same rule.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenEditHeldOutInput:
    """One HELD-OUT control input — the unit of one grid ROW.

    ``input_id``     — the stable id of this held-out item. It is used TWICE and both uses are load
                       bearing: it is one slot of the render key's ``control_ids`` (the conditioning
                       axis, in slot order) and it is the stem of every file this input renders. A
                       row and the directory identity it lives under therefore cannot disagree.
    ``prompts``      — mode id -> prompt text, for EVERY mode in :data:`QWEN_EDIT_PROMPT_MODES`.
                       Keyed by mode id rather than carried as two ``prompt_a`` / ``prompt_b`` fields
                       so the §8 mode tuple stays the single place a mode is declared; the mapping to
                       each mode's ``row_prompt_key`` happens in :func:`plan_qwen_edit_rows`.
    ``control_imgs`` — the control-slot thumbnail paths for this row, RELATIVE to the gallery, in
                       slot order. Staging them is the caller's job exactly as it is for H3's
                       reference thumbnails; this planner never invents an image path.
    ``label``        — the row header an operator reads. Defaults to ``input_id``.

    **A missing or blank prompt is REFUSED, not defaulted.** ``grid._qwen_edit_block`` prints an
    absent prompt as ``—`` so a hand-built row cannot silently lose one, but a row that reached the
    gallery missing half of §8's A/B pair has already lost the comparison the grid exists to make:
    the read is trace-vs-reinterpret between two prompts, and with one of them gone the A/B delta
    gets attributed to the adapter instead of to the empty prompt. Refusing here puts the error at
    the config, which is where it can be fixed.
    """

    input_id: str
    prompts: Mapping[str, str]
    control_imgs: tuple[str, ...] = field(default=())
    label: str | None = None

    def __post_init__(self) -> None:
        if not str(self.input_id).strip():
            raise ValueError(
                "QwenEditHeldOutInput.input_id is empty/whitespace. The id is the render key's "
                "conditioning slot AND the stem of every file this input renders, so an empty one "
                "collapses two held-out inputs onto one filename inside one render dir — the second "
                "overwrites the first and the grid shows one input's pixels twice under two labels."
            )
        # Normalise to a tuple so a caller's list cannot be mutated after the row plan is built.
        object.__setattr__(self, "control_imgs", tuple(self.control_imgs or ()))

        expected = {mode.id for mode in QWEN_EDIT_PROMPT_MODES}
        got = {str(k) for k in self.prompts}
        missing = sorted(expected - got)
        if missing:
            raise ValueError(
                f"QwenEditHeldOutInput {self.input_id!r} has no prompt for mode(s) {missing}. §8 of "
                "QWEN-CHAINED-EDIT-METHOD.md renders every held-out input under BOTH the style-only "
                "and the content-named prompt — the pair IS the measurement, and a row carrying one "
                "of them makes the A/B delta read as an adapter behaviour rather than as a missing "
                f"prompt. Supply a prompt for every mode in QWEN_EDIT_PROMPT_MODES ({sorted(expected)})."
            )
        unknown = sorted(got - expected)
        if unknown:
            raise ValueError(
                f"QwenEditHeldOutInput {self.input_id!r} carries prompt(s) for unknown mode(s) "
                f"{unknown}. Known modes: {sorted(expected)}. An unrecognised key is refused rather "
                "than ignored, because a mistyped mode id would leave the real mode's prompt absent "
                "while the config looked complete."
            )
        for mode_id in sorted(expected):
            if not str(self.prompts[mode_id]).strip():
                raise ValueError(
                    f"QwenEditHeldOutInput {self.input_id!r} has a blank prompt for mode "
                    f"{mode_id!r}. A blank prompt is not a style-only prompt — §8's mode A withholds "
                    "the CONTENT word ('reimagine the reference icon…'), it does not withhold the "
                    "prompt. Rendering one column unprompted and comparing it to a prompted one "
                    "measures the prompt, not the adapter."
                )

    @property
    def header(self) -> str:
        """The row header: the explicit ``label`` when given, else the id."""
        return str(self.label) if self.label else str(self.input_id)


def qwen_edit_control_ids(inputs: Sequence[QwenEditHeldOutInput]) -> tuple[str, ...]:
    """The render key's ``control_ids`` for a held-out set — slot order PRESERVED, never sorted.

    Slot index is the caption's ``ctrl_img_N`` addressing, so a reordered held-out set is a genuinely
    different request and must key to a different render directory (the same rule
    ``render_key.qwen_edit_render_key`` states for its ``control_ids`` argument, and D-10-REFORDER
    for H3 references).

    Derived from the inputs rather than accepted alongside them, so a caller cannot hand
    :func:`plan_qwen_edit_rows` a row set and a control-id list that disagree — which would produce
    rows whose paths point into a render directory built for a different held-out set.
    """
    ids = tuple(str(item.input_id) for item in inputs)
    seen: dict[str, int] = {}
    for i, name in enumerate(ids):
        if name in seen:
            raise ValueError(
                f"held-out input id {name!r} appears twice (positions {seen[name]} and {i}). The id "
                "is the file stem inside one render dir, so a repeat means the second input's "
                "renders overwrite the first's and the grid shows one input's pixels under two row "
                "labels — while every shape check passes."
            )
        seen[name] = i
    return ids


def qwen_edit_sample_filename(*, input_id: str, seed: int) -> str:
    """The file ONE cell renders to, inside its column's ``render_subdir``: ``<id>_s<seed>.png``.

    The H3 precedent is ``f"{slug(prompt)}_s{seed}.mp4"`` (``modal/fns.py:4853``) — the same shape
    with two deliberate changes:

      * **``.png``, not ``.mp4``.** ``QWEN_EDIT_FRAMES`` is pinned to 1; this family renders images,
        which is the whole reason ``grid`` grew an image tile.
      * **Keyed on the input id, not on the prompt.** H3 varies the PROMPT inside one render dir, so
        the prompt is what separates its clips. Here the prompt mode is a SUBDIR
        (``render_key.qwen_edit_render_key``'s second settled decision) and what varies inside a
        subdir is the held-out INPUT. Keying on the prompt instead would give the two modes' files
        different names in different subdirs — harmless — but would also make every filename change
        when an eval prompt is reworded, which silently defeats the resume that identity-keyed render
        dirs exist to provide.

    ``slug`` is reused from ``inference.grid`` rather than re-implemented: it is the house's
    filesystem-safe token function, and a second transcription of a path-safety rule is how the two
    spellings drift apart. Imported function-locally to keep this module's import tier (stdlib only,
    no package side effects) and its one-way dependency direction, exactly as ``band_render_keys``
    does for ``samples_layout``.
    """
    from signet_trainer.inference.grid import slug  # noqa: PLC0415

    return f"{slug(str(input_id))}_s{int(seed)}.png"


def qwen_edit_render_dir(
    column: QwenEditColumn, *, seed: int, control_ids: Sequence[str]
) -> str:
    """The identity-keyed render DIRECTORY one column writes into, relative to ``samples_root``.

    A band member keys on its own checkpoint, so the band's members are N sibling directories and the
    watcher's "did the whole band land?" question stays answerable per member
    (``samples_layout.expected_qwen_edit_band_keys``).

    **The BASE column keys on the reserved** ``base`` **token**, giving ``base_s<seed>_c<ids>`` — one
    directory for the whole band rather than one per member. Two reasons, and the first is money:
    the base render is the same pixels for every member (same seed, same controls, no adapter), so
    rendering it per member burns A100 time producing byte-identical files. The second is that it
    keeps the base render addressable as itself — §8 reads convergence as base-vs-LoRA divergence, so
    the reference has to be one thing the whole grid points at, not N copies that could silently
    differ. The token cannot collide with a real member: ``plan_qwen_edit_columns`` already refuses a
    band member that tokenises to ``base``.
    """
    from signet_trainer.inference.render_key import qwen_edit_render_key  # noqa: PLC0415

    checkpoint = BASE_COLUMN_TOKEN if column.is_base else str(column.checkpoint)
    return qwen_edit_render_key(checkpoint=checkpoint, seed=seed, control_ids=control_ids)


def qwen_edit_cell_relpath(
    column: QwenEditColumn,
    *,
    input_id: str,
    seed: int,
    control_ids: Sequence[str],
) -> str:
    """⛔ THE ONE JOIN: the path of one grid cell's image, relative to the gallery's directory.

    ``<render dir>/<column.render_subdir>/<input>_s<seed>.png``, e.g.::

        step-4000_s42_cbus-train/lora/step-4000/a_style/train_icon_s42.png
        base_s42_cbus-train/base/b_content/train_icon_s42.png

    **The sampler must compose its output path from this function**, not from a second spelling of
    the same rule. The gallery reads a cell back through ``row[column.row_key]``; this is what puts
    the value there. Two independent transcriptions of a path layout is precisely how a grid comes to
    reference files that exist under different names — every tile falls back to 'generation failed'
    on a render that actually succeeded, or worse, one column's file is found under another column's
    key and the band is judged on the wrong pixels.

    The gallery therefore belongs at ``samples_root``'s top level (``samples_qwen_edit/index.html``),
    ABOVE the per-band-member render dirs, because one page spans the whole band while each render
    directory holds one member.
    """
    return (
        f"{qwen_edit_render_dir(column, seed=seed, control_ids=control_ids)}"
        f"/{column.render_subdir}"
        f"/{qwen_edit_sample_filename(input_id=input_id, seed=seed)}"
    )


def plan_qwen_edit_rows(
    inputs: Sequence[QwenEditHeldOutInput],
    columns: Sequence[QwenEditColumn],
    *,
    seed: int,
    criteria_line: str | None = None,
) -> list[dict]:
    """Derive the gallery's ROWS from the held-out set and the planned columns (pure/CPU).

    One row per held-out input; one cell per column, at the path
    :func:`qwen_edit_cell_relpath` says that column writes. The result is exactly what
    ``grid.write_qwen_edit_gallery`` consumes, so the render loop assembles no dict of its own — it
    renders to ``qwen_edit_cell_relpath``'s answer and hands these rows to the writer.

    Every path is EXPECTED, not verified: this is a pure planner and never touches a filesystem. A
    cell whose file did not land renders as the gallery's honest 'generation failed' tile, which is
    the reporting behaviour §8 wants — a failed band member must appear as a column of failures, not
    vanish from the page.

    ``control_ids`` is derived from ``inputs`` via :func:`qwen_edit_control_ids` rather than accepted
    as an argument, so the row paths and the render-directory identity are guaranteed to describe the
    same held-out set.

    Args:
        inputs: the held-out control inputs, in SLOT ORDER (the conditioning axis).
        columns: ``plan_qwen_edit_columns(band)``'s result, passed through unchanged.
        seed: the render seed — shared by every cell, which is what makes base-vs-LoRA and A-vs-B
            differences attributable to the adapter and the prompt rather than to noise.
        criteria_line: optional per-grid override of the row's PASS line.

    Returns:
        One dict per input: ``input_id`` / ``input_label`` / ``seed`` / ``control_imgs``, each mode's
        prompt under that mode's ``row_prompt_key``, an optional ``criteria_line``, and one relative
        image path per ``column.row_key``.
    """
    if not inputs:
        raise ValueError(
            "plan_qwen_edit_rows got an EMPTY held-out set. §8 reads convergence as base-vs-LoRA "
            "divergence on HELD-OUT inputs; with none, the page renders a criteria block and a "
            "banner over zero rows — a complete-looking artifact that measures nothing."
        )
    if not columns:
        raise ValueError(
            "plan_qwen_edit_rows got an EMPTY column list. The columns are "
            "plan_qwen_edit_columns(band)'s result; with none, every row would carry its prompts and "
            "control thumbnails and no cell to judge them against."
        )

    control_ids = qwen_edit_control_ids(inputs)
    modes = {column.mode.id: column.mode for column in columns}

    rows: list[dict] = []
    for item in inputs:
        row: dict[str, Any] = {
            "input_id": str(item.input_id),
            "input_label": item.header,
            "seed": int(seed),
            "control_imgs": list(item.control_imgs),
        }
        if criteria_line is not None:
            row["criteria_line"] = criteria_line
        # Prompts are keyed off the COLUMNS' modes, so the text printed above a row can never
        # describe a mode the row has no columns for.
        for mode_id, mode in modes.items():
            row[mode.row_prompt_key] = str(item.prompts[mode_id])
        for column in columns:
            row[column.row_key] = qwen_edit_cell_relpath(
                column,
                input_id=item.input_id,
                seed=seed,
                control_ids=control_ids,
            )
        rows.append(row)
    return rows


def render_qwen_edit_sample(
    *,
    pipeline: Any,
    control_images: Sequence[Any],
    prompt: str,
    out_path: Any,
    seed: int,
    width: int,
    height: int,
    negative_prompt: str | None = None,
    steps: int | None = None,
    true_cfg: float | None = None,
    adapter: bool = True,
    **refused: Any,
) -> str:
    r"""Render ONE cell of the §8 grid — one control set, one prompt, one adapter state. Modal-side.

    LANDED 2026-08-09, replacing the declared stub. The declared entry point of the render surface,
    and a THIN DELEGATION to ``models/qwen_edit_pipeline.qwen_edit_generate``, which is where the
    behaviour, the §8 recipe, the two pre-render gates and every line of provenance live. Read that
    function's docstring before changing anything here.

    **Why the implementation is not in this file**, given that this is the symbol the whole surface
    was built around — two shipped guards, either of which alone would force the split:

      * ``tests/test_qwen_edit_render_surface.py`` scans this module's source for
        ``^\s*import\s+torch`` — the ``^\s*`` anchor matches an INDENTED, function-local import
        too, so this module may not touch torch even lazily. A generate call needs a seeded
        ``torch.Generator`` and a ``no_grad`` context.
      * ``tests/test_no_wan_params.py:31`` bans the bare scheduler token from executable code in
        every ``*.py`` under ``inference/`` — a guard written for LTX paths whose directory-wide
        scope now also covers a family where that setting is MANDATORY at a different value. §8's
        static reparameterisation cannot be spelled here.

    Neither guard is weakened to land this: the scan is non-recursive and covers ``inference/``
    only, so the construction lives in ``models/``, which it does not scan. That placement is
    additive and reversible — the day ``_WAN_TOKENS`` is narrowed to the LTX modules (a RULING on a
    shipped money-safe guard, and Timothy's to make, not a side effect of landing a sampler) nothing
    has to move.

    The split also puts the seam in the right place on its own merits. This module is a PURE PLANNER
    everywhere else — it decides which cells exist (:func:`plan_qwen_edit_columns` over a
    :class:`CheckpointBand` and the two :data:`QWEN_EDIT_PROMPT_MODES`) and never opens a file. The
    call that drives torch and diffusers belongs beside the pipeline it drives.

    The signature is restated rather than ``**kwargs``-forwarded so this entry point is
    self-documenting, and ``tests/test_qwen_edit_sampler.py`` asserts the two signatures are
    IDENTICAL so the restatement cannot drift.

    Args:
        pipeline: a ``QwenImageEditPlusPipeline`` from ``build_qwen_edit_pipeline`` — scheduler
            pinned, transformer PEFT-wrapped with the band member's adapter loaded.
        control_images: the control slots in order, as Pillow images (1..3), RAW and unresized.
        prompt: this cell's prompt text — one of the row's §8 A/B pair.
        out_path: destination image path; parent directories are created.
        seed: §8 renders a band at ONE seed so its members are comparable.
        width: target canvas width in pixels.
        height: target canvas height in pixels.
        negative_prompt: defaults to §8's. Must be non-None — a ``None`` here turns true-CFG off.
        steps: defaults to §8's 30 (the pipeline's own default is 50).
        true_cfg: defaults to §8's 4.0. Must exceed 1.0. This is ``true_cfg_scale``, NOT
            ``guidance_scale`` — the latter is refused by name, because this checkpoint is not
            guidance-distilled and passing it renders at effective CFG 1.0.
        adapter: ``False`` renders the §8 BASE control column from the SAME model under
            ``disable_adapter()``. Everything else about the call is identical, which is the point.
        **refused: forwarded, and refused by name with the failure each would cause.

    Returns:
        The written path, as a string.
    """
    from signet_trainer.models.qwen_edit_pipeline import (  # noqa: PLC0415 — Modal-side tier
        qwen_edit_generate,
    )

    return qwen_edit_generate(
        pipeline=pipeline,
        control_images=control_images,
        prompt=prompt,
        out_path=out_path,
        seed=seed,
        width=width,
        height=height,
        negative_prompt=negative_prompt,
        steps=steps,
        true_cfg=true_cfg,
        adapter=adapter,
        **refused,
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
