"""conditioning.qwen_edit — ``QwenEditStrategy``: Qwen-Image-Edit-2511 multi-slot control images.

The sixth concrete ``TrainingStrategy`` and the FIRST of model family #3 (``qwen_edit``). Siblings:
``SingleFrameStrategy`` (Phase 5), ``MultiFrameStrategy`` (Phase 6), ``ICLoraStrategy`` (Phase 7),
``InpaintStrategy`` / ``A2VStrategy`` (Phase 9) — all LTX — and ``H3RefStrategy`` (Phase 10, family
#2). Like ``h3_ref``, this is BUILD work rather than port work: the reference implementation is
ai-toolkit's ``QwenImageEditPlusModel``, which is a *sampler-side* wiring, not a signet strategy.

The module splits in two, deliberately, the way ``h3_ref`` does:

  * **Pure slot resolution** (top of the file) — no ``torch``, no tensors, no I/O. Which control
    image occupies which slot, whether a slot is a blank fill, and the 1:1 stem match against the
    target. It is a decision record: the slot contract expressed as code a CPU test can prove.
  * **``QwenEditStrategy``** (bottom) — the ``TrainingStrategy`` that turns a dataloader batch into
    the packed Qwen image sequence, with the control block out of the loss.

⛔ DIVERGE #1 — THE CONTROL BLOCK IS A **SUFFIX**, NOT A PREFIX. ``ref_seq_len`` STAYS ``None``.
------------------------------------------------------------------------------------------------
``ic_lora.py`` concatenates ``[reference, target]`` and slices the loss with
``model_output[:, ref_seq_len:, :]``. Qwen-Image-Edit is the **exact inversion**:
``qwen_image_edit_plus.py:314-316`` builds ``torch.cat([packed_latents_list[b], control], dim=1)``
— target FIRST, controls appended — and ``:346`` reads back
``noise_pred = noise_pred[:, : raw_packed_latents.size(1)]``, a **prefix** slice.

``ModelInputs.ref_seq_len`` is documented at ``strategy.py:114`` as *"length of the reference
prefix"*, and every consumer written against it slices ``[:, ref_seq_len:, :]``. Populating it here
would keep the CONTROL rows and drop the TARGET rows — identical dtype, plausible loss curve, an
adapter trained on the thing it was supposed to be conditioned by. So this strategy leaves
``ref_seq_len`` at ``None`` and carries the split as ``QwenEditModelInputs.target_seq_len`` /
``.control_seq_len`` instead, and ``compute_loss`` slices the PREFIX. The field is not "unused
because we forgot"; it is empty because its semantics are inverted here.

⛔ DIVERGE #2 — THE TEXT MASK IS AN **INT 0/1** MASK, NOT AN ADDITIVE ``-inf`` FLOAT MASK.
-----------------------------------------------------------------------------------------
``single_frame.py:184-185`` / ``ic_lora.py:248-249`` convert the int attention mask into an additive
float mask (``0`` / ``-inf``) because that is what ltx_core's attention consumes. Qwen's transformer
consumes the mask **twice, as integers**: ``qwen_image_edit_plus.py:326-328`` casts it to
``torch.int64`` and ``:329`` derives ``txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()`` from
it — a SUM, which an ``-inf``-filled float mask turns into ``-inf``. Carrying the LTX conversion here
does not fail; it produces a garbage sequence length and a silently mis-attended text stream.

⛔ DIVERGE #3 — CONTROL ENTERS THE MODEL **TWICE**, AND ONLY ONE HALF IS THIS STRATEGY'S.
----------------------------------------------------------------------------------------
``qwen_image_edit_plus.py:66`` sets ``encode_control_in_text_embeddings = True``: every control image
is ALSO rendered through the Qwen2.5-VL encoder at ``CONDITION_IMAGE_SIZE`` (``384*384``,
``diffusers .../pipeline_qwenimage_edit_plus.py:66``) and lands inside ``encoder_hidden_states``,
while the packed latent block this strategy assembles is the VAE channel at ``VAE_IMAGE_SIZE``
(``1024*1024``, ``:67``). Channel A (VL) is a PREP-time artifact and arrives here already baked into
the text payload; channel B (VAE) is assembled below. A control image dropped from one channel and
not the other is a regime that occurs nowhere at inference — which is why the slot resolver below
refuses a short list instead of shortening the sequence.

⛔ DIVERGE #4 — A MISSING CONTROL SLOT IS AN **ERROR**, NEVER A DROP.
--------------------------------------------------------------------
``toolkit/dataloader_mixins.py:981-989`` matches control images by TARGET STEM
(``file_name_no_ext + ext``) inside each configured ``control_path`` directory, and appends to
``found_control_images`` **only inside** ``if os.path.exists(...)`` (``:984-985``). With
``control_path = [dirA, dirB, dirC]`` and the stem missing from ``dirB``, the list does not merely
get shorter — ``dirC``'s image **slides into slot 1**, where ``dirB``'s belongs. Slot identity is
positional and the model was conditioned on that order, so the sample trains against a different
request than the one the caption describes, and nothing raises. ``resolve_control_slots`` below
refuses: a gap must be an explicit blank fill occupying its declared index, or the sample is an
error. This is the single deliberate behavioural divergence from ai-toolkit in this module.

⛔ DIVERGE #5 — the velocity sign is the **LTX** one, and that is a checked fact, not an assumption.
---------------------------------------------------------------------------------------------------
``h3_ref.py:702-713`` documents family #2 flipping to a data-ward velocity, so family #3 was checked
rather than assumed: ``extensions_built_in/diffusion_models/qwen_image/qwen_image.py:389-392``
(``get_loss_target`` → ``(noise - batch.latents).detach()``), inherited unchanged by
``QwenImageEditPlusModel(QwenImageModel)`` (``qwen_image_edit_plus.py:44``), and
``toolkit/samplers/custom_flowmatch_sampler.py:90-101`` (``add_noise`` →
``(1 - t_01) * original_samples + t_01 * noise``). Qwen is **noise-ward**, exactly like LTX:
``sigma = 0`` is clean, ``x_t = (1 - sigma) * x0 + sigma * eps``, ``v = eps - x0``.

CRITICAL — Anti-Pattern 6 / Pitfall 6 (import-confinement):
    Module top imports ``torch`` + stdlib + two signet siblings that are themselves confined
    (``conditioning/strategy``, ``conditioning/qwen_edit_geometry`` — the latter is stdlib +
    ``dataclasses`` only, so it adds no third-party root). There is no ``modal``, no ``ltx_core``,
    no ``diffusers``, no ``config/schema`` and — deliberately — no
    ``train/qwen_edit_step`` import: that module is a declared stub this pass, and a strategy that
    could not be constructed without it would make the whole family untestable until the forward
    lands. Every heavy seam (the 2×2 packing transcription, the blank-fill encode, the sigma draw)
    arrives INJECTED, the ``ic_lora.py`` / ``h3_ref.py`` precedent. That is what keeps this file
    CPU-unit-testable on Windows with zero GPU and zero Modal spend.

Pitfall 2 (the 166.9 GiB OOM): the latent grid is read from the LATENT TENSOR'S OWN SHAPE, never
reconstructed from the payload's pixel ``height``/``width``. Where a caller hands rows-form input
the grid must be declared explicitly and is cross-checked against the row count — it is never
inferred from pixels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

# Single home for the qwen_edit arch math (spec §1.1 deliverable D). Re-export, do NOT duplicate —
# the ``config/validators.py:43-45`` doctrine for ``h3_geometry``, and the reason ``h3_ref.py:64``
# imports ``H3PackedLayout`` rather than restating it. ``qwen_edit_geometry`` is the CONFINED tier
# (stdlib + ``dataclasses`` only), so this import adds ZERO third-party roots to this module.
from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_FRAMES,
    QWEN_EDIT_LATENT_CHANNELS,
    QWEN_EDIT_MAX_CONTROL_SLOTS,
    QWEN_EDIT_PATCH,
    QWEN_EDIT_PATCH_DIM,
)
from signet_trainer.conditioning.strategy import Modality, ModelInputs, TrainingStrategy

__all__ = [
    "QWEN_EDIT_BLANK_FILLS",
    "QWEN_EDIT_CONTROL_BATCH_KEYS",
    "QWEN_EDIT_DATA_SOURCES",
    "QWEN_EDIT_TARGET_BATCH_KEYS",
    "QWEN_EDIT_TEXT_BATCH_KEYS",
    "QwenControlSlot",
    "QwenEditModelInputs",
    "QwenEditStrategy",
    "resolve_control_slots",
]


# --------------------------------------------------------------------------------------------------
# Module-local constants. The ARCHITECTURE numbers are imported above and re-declared NOWHERE here.
# The packed row width (64) in particular was briefly declared in this file under a SECOND name
# (``QWEN_EDIT_ROW_DIM``, alongside the geometry module's ``QWEN_EDIT_PATCH_DIM``), which is worse
# than a plain duplicate: one quantity under two names drifts without either copy looking wrong,
# because no reader ever sees the two side by side.
# --------------------------------------------------------------------------------------------------

#: The synthetic images a gap may be filled with. Kept as a tuple (not a ``Literal``) so this module
#: needs no ``config/schema`` import; the config block re-declares the same three and
#: ``tests/test_qwen_edit_strategy.py`` pins the two equal — the ``H3_LORA_TARGET_REGEX`` precedent
#: (``config/validators.py:147-153``): duplication enforced by TEST, never by a cross-import.
QWEN_EDIT_BLANK_FILLS: tuple[str, ...] = ("black", "white", "gray")

#: The three precomputed sources a qwen_edit run reads. All are ``qwen_edit_``-prefixed for the same
#: reason H3's four are ``h3_``-prefixed (``h3_ref.py:778-780``): a different VAE and a different text
#: encoder produce tensors of the SAME rank as the LTX ones, so an unprefixed name could be paired
#: with an LTX-encoded source and be silently wrong at the correct shape.
#:
#: ⚠ ``qwen_edit_conditions`` carries Qwen2.5-VL text embeddings and NO ``"latents"`` key, so it must
#: NOT be added to ``data/precomputed.py``'s video-latent allowlist — the same trap as
#: ``h3_conditions``. The other two DO store ``{"latents": [C, F=1, H_lat, W_lat], ...}``, which is
#: exactly what makes allowlisting THEM correct.
QWEN_EDIT_DATA_SOURCES: tuple[str, ...] = (
    "qwen_edit_latents",
    "qwen_edit_conditions",
    "qwen_edit_control_latents",
)

#: Batch keys per source, in lookup order: the ``_PRECOMPUTED_SOURCE_OUTPUT_KEYS``-style output key
#: first, then the list-form dir-name==output-key fallback. Mirrors ``a2v.py:60`` and
#: ``h3_ref.py:449-452``.
QWEN_EDIT_TARGET_BATCH_KEYS: tuple[str, ...] = ("qwen_edit_latent_conditions", "qwen_edit_latents")
QWEN_EDIT_TEXT_BATCH_KEYS: tuple[str, ...] = ("qwen_edit_text_conditions", "qwen_edit_conditions")
QWEN_EDIT_CONTROL_BATCH_KEYS: tuple[str, ...] = (
    "qwen_edit_control_latent_conditions",
    "qwen_edit_control_latents",
)


# --------------------------------------------------------------------------------------------------
# Pure slot resolution — stdlib only, no tensors. This half is the DECISION RECORD.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenControlSlot:
    """One resolved control slot: which image, at which index, or an explicit blank fill.

    ``index`` is the slot's position in the packed control block AND its position in the caption's
    ``ctrl_img_N`` addressing — the two must agree, which is the whole reason a gap cannot be closed
    up (DIVERGE #4).

    ``stem`` is the target's file stem, carried on every slot including blanks. ai-toolkit matches
    control images to targets BY STEM (``dataloader_mixins.py:979-985``), so recording it here makes
    the 1:1 correspondence a checkable property of the resolved plan rather than a property of a
    directory listing that no longer exists by training time.
    """

    index: int
    stem: str
    path: str | None
    blank: bool
    fill: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"invalid control slot index {self.index}: must be >= 0.")
        if self.blank and self.path is not None:
            raise ValueError(
                f"control slot {self.index} is marked blank but carries path {self.path!r}: a blank "
                f"fill is a SYNTHETIC image, so a source path would make the record ambiguous about "
                f"what the model was actually shown."
            )
        if not self.blank and self.fill is not None:
            raise ValueError(
                f"control slot {self.index} carries fill {self.fill!r} but is not blank: the fill "
                f"names which synthetic image occupied a gap and is meaningless on a real control."
            )
        if self.blank and self.fill not in QWEN_EDIT_BLANK_FILLS:
            raise ValueError(
                f"control slot {self.index} is blank with fill {self.fill!r}: expected one of "
                f"{list(QWEN_EDIT_BLANK_FILLS)}."
            )


def resolve_control_slots(
    target_stem: str,
    entries: Sequence[Any],
    *,
    control_slots: int,
    blank_slot_fill: str,
) -> list[QwenControlSlot]:
    """Resolve the ordered ``control_slots``-long slot plan for one sample. Pure; no tensors.

    ``entries`` are the per-slot dicts carried by the ``qwen_edit_control_latents`` payload. Each may
    declare ``slot`` (its index), ``path``, ``stem``, and ``blank`` / ``fill``.

    The resolution rules, and why each exists:

    * **Explicit indices win.** When any entry declares ``slot``, ALL must — a mixed payload has no
      unambiguous reading and would be resolved by silently trusting list order, which is the
      ai-toolkit slide (DIVERGE #4).
    * **Positional order is accepted ONLY when the payload is complete.** ``len(entries) ==
      control_slots`` with no declared indices is unambiguous: entry *i* is slot *i*. A SHORT payload
      with no declared indices is refused, because "which slot is missing" is exactly the question
      list order cannot answer.
    * **Every entry's stem must equal the target's.** The control set is defined by stem match; a
      mismatched stem means the sample is conditioned on another sample's control image, which
      trains a caption/condition pair that describes neither.
    * **Gaps become explicit blank slots**, at their own index. The blank still costs rows — a black
      image has a real VAE latent, not a zero one — so it is present in the sequence, present in
      ``img_shapes``, and out of the loss like every other control row.

    Returns exactly ``control_slots`` slots in index order.
    """
    if not 1 <= control_slots <= QWEN_EDIT_MAX_CONTROL_SLOTS:
        raise ValueError(
            f"invalid control_slots {control_slots}: Qwen-Image-Edit-2511 addresses 1.."
            f"{QWEN_EDIT_MAX_CONTROL_SLOTS} control images (ai-toolkit's Edit-Plus prompt template "
            f"names ctrl_img_1..{QWEN_EDIT_MAX_CONTROL_SLOTS}); a slot beyond that is packed into "
            f"the sequence but can never be referred to by the caption."
        )
    if blank_slot_fill not in QWEN_EDIT_BLANK_FILLS:
        raise ValueError(
            f"invalid blank_slot_fill {blank_slot_fill!r}: expected one of "
            f"{list(QWEN_EDIT_BLANK_FILLS)}."
        )

    declared = [entry for entry in entries if _entry_get(entry, "slot") is not None]
    if declared and len(declared) != len(entries):
        undeclared = [i for i, entry in enumerate(entries) if _entry_get(entry, "slot") is None]
        raise ValueError(
            f"control payload mixes declared and undeclared slot indices: entries at positions "
            f"{undeclared} carry no 'slot'. A partially-indexed payload can only be resolved by "
            f"trusting list order for the rest, which is precisely the ai-toolkit slide this refuses "
            f"(dataloader_mixins.py:984-985 appends only on os.path.exists, so a missing stem moves "
            f"every later control one slot to the left). Declare 'slot' on every entry, or on none "
            f"and supply all {control_slots}."
        )
    if not declared and len(entries) not in (0, control_slots):
        raise ValueError(
            f"control payload has {len(entries)} entr(ies) for {control_slots} slot(s) and declares "
            f"no 'slot' index on any of them: list order cannot say WHICH slot is missing, and "
            f"guessing shifts every later control into the wrong ctrl_img_N addressing. Declare "
            f"'slot' on each entry (blank fills included), or supply all {control_slots}."
        )

    by_index: dict[int, QwenControlSlot] = {}
    for position, entry in enumerate(entries):
        raw_index = _entry_get(entry, "slot")
        index = position if raw_index is None else int(raw_index)
        if not 0 <= index < control_slots:
            raise ValueError(
                f"control entry at position {position} declares slot {index}, outside "
                f"0..{control_slots - 1}. Either the payload was written for a different "
                f"qwen_edit.control_slots than this config declares, or a slot index is off by one — "
                f"both change which image the caption's ctrl_img_N refers to."
            )
        if index in by_index:
            raise ValueError(
                f"control entry at position {position} declares slot {index}, which is already "
                f"occupied by {by_index[index].path or 'a blank fill'}. Two images cannot share a "
                f"slot; one of them would be dropped and the sample would train against a "
                f"conditioning set the caption does not describe."
            )
        stem = _entry_get(entry, "stem")
        if stem is not None and str(stem) != target_stem:
            raise ValueError(
                f"control entry at position {position} (slot {index}) has stem {str(stem)!r} but "
                f"the target's stem is {target_stem!r}. Control images are matched to targets BY "
                f"STEM (dataloader_mixins.py:979-985); a mismatch means this sample is conditioned "
                f"on another sample's control image."
            )
        blank = bool(_entry_get(entry, "blank") or False)
        path = _entry_get(entry, "path")
        by_index[index] = QwenControlSlot(
            index=index,
            stem=target_stem,
            path=None if blank else (None if path is None else str(path)),
            blank=blank,
            fill=(str(_entry_get(entry, "fill") or blank_slot_fill) if blank else None),
        )

    for index in range(control_slots):
        if index not in by_index:
            by_index[index] = QwenControlSlot(
                index=index, stem=target_stem, path=None, blank=True, fill=blank_slot_fill
            )
    return [by_index[index] for index in range(control_slots)]


def _entry_get(entry: Any, name: str) -> Any:
    """Read ``name`` off a slot entry that may be a mapping or a small object. Pure; stdlib only."""
    if isinstance(entry, dict):
        return entry.get(name)
    return getattr(entry, name, None)


# --------------------------------------------------------------------------------------------------
# The batch contract — ``ModelInputs`` plus what a Qwen edit batch has and an LTX one does not.
# --------------------------------------------------------------------------------------------------


@dataclass
class QwenEditModelInputs(ModelInputs):
    """``ModelInputs`` plus the Qwen wire format. ``isinstance(x, ModelInputs)`` still holds.

    ``ModelInputs`` stays the shared contract (``strategy.py:99-123`` is NOT edited by family #3 —
    the ``h3_ref.py:455-474`` precedent), so the loop, the dry-run and every existing consumer see
    the familiar shape. The additions exist because Qwen's real wire format is an eight-kwarg call
    whose positional information lives in ``img_shapes`` / ``txt_seq_lens``, not in a coordinate
    tensor:

        target_seq_len      packed rows of the TARGET image — the loss prefix
        control_seq_len     packed rows of ALL control slots — the suffix, out of the loss
        control_slots       the resolved plan, in index order (blank fills included)
        control_slot_rows   rows contributed by each slot, index-aligned with ``control_slots``
        img_shapes          ``[[(F, H2, W2), ...]]`` — target block first, then each control slot
        txt_seq_lens        per-batch-item unpadded text length
        caption_dropped     whether the caption-dropout draw fired for this sample

    ⚠ ``ref_seq_len`` is deliberately left at ``None``. See DIVERGE #1: the control block is a
    SUFFIX, and every consumer of ``ref_seq_len`` slices ``[:, ref_seq_len:, :]`` — which here would
    keep the controls and drop the target.
    """

    target_seq_len: int = 0
    control_seq_len: int = 0
    control_slots: tuple[QwenControlSlot, ...] = ()
    control_slot_rows: tuple[int, ...] = ()
    img_shapes: tuple[tuple[int, int, int], ...] = ()
    txt_seq_lens: tuple[int, ...] = ()
    caption_dropped: bool = False

    def transformer_kwargs(self) -> dict[str, Any]:
        """The exact ``QwenImageTransformer2DModel(**kwargs)`` call, assembled from fields above.

        This lives HERE rather than in ``train/qwen_edit_step.py`` for the reason ``h3_ref.py:463-469``
        gives for carrying ``packed.kwargs`` on the batch: if the strategy assembles the sequence and
        something else assembles the call, the wire format has two homes and drifts in one of them.
        Mirrors ``qwen_image_edit_plus.py:332-344`` argument for argument.

        ``timestep`` is the 0-1 sigma, NOT the 0-1000 scale: ai-toolkit divides at the call site
        (``:336``, ``timestep / 1000``) and its scheduler divides again for the interpolation
        (``custom_flowmatch_sampler.py:91``), so signet carries the already-normalized quantity and
        the division happens nowhere. ``guidance=None`` is not an omission — it is the value
        ai-toolkit passes (``:337``).
        """
        return {
            "hidden_states": self.video.latent,
            "timestep": self.video.sigma,
            "guidance": None,
            "encoder_hidden_states": self.video.context,
            "encoder_hidden_states_mask": self.video.context_mask,
            "img_shapes": [[tuple(shape) for shape in self.img_shapes]],
            "txt_seq_lens": list(self.txt_seq_lens),
            "return_dict": False,
        }


# --------------------------------------------------------------------------------------------------
# Private tensor helpers. Deliberately NOT imported from ``h3_ref`` — coupling family #3 to family
# #2's private helpers would make an H3 refactor a silent Qwen behaviour change.
# --------------------------------------------------------------------------------------------------


def _lookup(batch: Any, keys: Sequence[str], what: str, *, required: bool = True) -> Any:
    """First present key wins; a missing REQUIRED source names every key it looked for."""
    for key in keys:
        if key in batch:
            return batch[key]
    if not required:
        return None
    raise ValueError(
        f"batch is missing the {what} source: looked for {list(keys)}. The source dir -> output-key "
        f"wiring lives in modal/fns.py::_PRECOMPUTED_SOURCE_OUTPUT_KEYS and must cover every name "
        f"QwenEditStrategy.get_data_sources() declares — a source configured but not mapped silently "
        f"drops the whole sample from the index rather than raising."
    )


def _latent_grid(latent: torch.Tensor, name: str) -> tuple[int, int]:
    """Read ``(H_lat, W_lat)`` off a stored image latent's OWN shape. Refuses anything but ``F == 1``.

    Accepts ``[C, F, H, W]`` (what ``PrecomputedDataset`` returns for a ``qwen_edit_``-prefixed
    latent source) and ``[B=1, C, F, H, W]`` (what ``AutoencoderKLQwenImage.encode`` emits).

    ⚠ A 4-D tensor is read as ``[C, F, H, W]`` and its leading dim is CHECKED against
    ``QWEN_EDIT_LATENT_CHANNELS``. Without that check ``[B, C, H, W]`` — the other perfectly ordinary
    4-D image-latent layout — parses as ``C=1, F=16``, which then fails the ``F == 1`` gate with a
    confusing message instead of the right one.

    Pitfall 2: the grid comes from the tensor, never from the payload's pixel ``height``/``width``.
    """
    if latent.ndim == 5:
        if latent.shape[0] != 1:
            raise ValueError(
                f"invalid {name}: leading batch dimension must be 1 (one sample per sequence), got "
                f"{latent.shape[0]} in {tuple(latent.shape)}."
            )
        latent = latent[0]
    if latent.ndim != 4:
        raise ValueError(
            f"invalid {name}: expected a [C, F, H, W] or [1, C, F, H, W] image latent, got "
            f"{latent.ndim}-D {tuple(latent.shape)}."
        )
    channels, frames, height, width = latent.shape
    if channels != QWEN_EDIT_LATENT_CHANNELS:
        raise ValueError(
            f"invalid {name}: leading dimension is {channels}, expected "
            f"{QWEN_EDIT_LATENT_CHANNELS} channels in [C, F, H, W] order. A [B, C, H, W] tensor "
            f"reaches here looking identical in RANK and would be read as C={channels}, "
            f"F={frames} — re-shape it to [C, F, H, W] rather than letting the frame gate below "
            f"report the wrong cause."
        )
    if frames != QWEN_EDIT_FRAMES:
        raise ValueError(
            f"invalid {name}: F = {frames}, but qwen_edit is an IMAGE family — every latent carries "
            f"exactly {QWEN_EDIT_FRAMES} frame and img_shapes hardcodes it "
            f"(qwen_image_edit_plus.py:236 builds (1, h//2, w//2)). A multi-frame latent would pack "
            f"into rows that the transformer's RoPE prices as a single frame."
        )
    if height % QWEN_EDIT_PATCH or width % QWEN_EDIT_PATCH:
        raise ValueError(
            f"invalid {name}: latent grid {height}x{width} is not divisible by the 2x2 pack "
            f"(qwen_image_edit_plus.py:223-229). Pixel dims must be a multiple of 16 "
            f"(VAE scale 8 x pack 2)."
        )
    return int(height), int(width)


def _rows_and_grid(
    payload: Any,
    name: str,
    pack_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Coerce a payload to ``([1, rows, 64], (H2, W2))`` — the packed form the sequence consumes.

    Two input forms, and the asymmetry between them is deliberate:

    * A ``[C, F, H, W]`` LATENT carries its own grid, so ``(H2, W2)`` is derived and the 2×2 pack is
      applied — but ONLY through the injected ``pack_fn``. ``h3_ref.py:500-505`` states the reason
      and it holds identically here: the reshape is not the hard part, the intra-patch ELEMENT ORDER
      is (``view(B, C, H//2, 2, W//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(...)`` —
      ``qwen_image_edit_plus.py:223-229``), and it is a checkpoint contract. A wrong order corrupts
      training silently at the correct shape, so nothing here transcribes it a second time.
    * A ROWS-form ``[1, rows, 64]`` tensor has already been packed and has LOST its grid. It must
      therefore declare ``latent_hw``; the declaration is cross-checked against the row count, which
      is what makes the declaration trustworthy rather than merely present.
    """
    grid: tuple[int, int] | None = None
    tensor: Any = payload
    if isinstance(payload, dict):
        for key in ("latents", "rows", "hidden_states"):
            if payload.get(key) is not None:
                tensor = payload[key]
                break
        else:
            raise ValueError(
                f"invalid {name} payload: no 'latents'/'rows'/'hidden_states' tensor in "
                f"{sorted(payload)}."
            )
        declared = payload.get("latent_hw")
        if declared is not None:
            grid = (int(declared[0]), int(declared[1]))
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"invalid {name}: expected a torch.Tensor, got {type(tensor).__name__}.")

    if tensor.ndim >= 4:
        latent_h, latent_w = _latent_grid(tensor, name)
        if pack_fn is None:
            raise ValueError(
                f"invalid {name}: got a {tensor.ndim}-D {tuple(tensor.shape)} latent, but the packed "
                f"sequence consumes ROW-MAJOR [1, rows, {QWEN_EDIT_PATCH_DIM}] tensors. The 2x2 "
                f"pack (view/permute/reshape, qwen_image_edit_plus.py:223-229) is NOT applied "
                f"because no pack_fn was injected: the intra-patch element ORDER is a checkpoint "
                f"contract and a wrong order corrupts training silently at the correct shape. "
                f"Construct the strategy with "
                f"QwenEditStrategy(..., pack_fn=conditioning.qwen_edit_packing.qwen_edit_image_rows)"
                f" — the single transcription of that transform — or pack upstream."
            )
        rows = pack_fn(tensor)
        expected_grid = (latent_h // QWEN_EDIT_PATCH, latent_w // QWEN_EDIT_PATCH)
    else:
        if grid is None:
            raise ValueError(
                f"invalid {name}: rows-form input {tuple(tensor.shape)} has already been packed and "
                f"no longer carries its latent grid, but img_shapes needs (F, H//2, W//2) per block "
                f"(qwen_image_edit_plus.py:236,305). Declare it as payload['latent_hw'] = "
                f"(H_lat, W_lat) — the LATENT grid, never the pixel size: reconstructing latent dims "
                f"from pixel metadata is Pitfall 2, the 166.9 GiB OOM."
            )
        rows = tensor
        expected_grid = (grid[0] // QWEN_EDIT_PATCH, grid[1] // QWEN_EDIT_PATCH)

    if rows.ndim == 2:
        rows = rows.unsqueeze(0)
    if rows.ndim != 3:
        raise ValueError(
            f"invalid {name}: expected [1, rows, {QWEN_EDIT_PATCH_DIM}] or "
            f"[rows, {QWEN_EDIT_PATCH_DIM}] after packing, got {rows.ndim}-D {tuple(rows.shape)}."
        )
    if rows.shape[0] != 1:
        raise ValueError(
            f"invalid {name}: leading batch dimension must be 1 (one sample per sequence), got "
            f"{rows.shape[0]} in {tuple(rows.shape)}."
        )
    if rows.shape[2] != QWEN_EDIT_PATCH_DIM:
        raise ValueError(
            f"invalid {name}: packed row width is {rows.shape[2]}, expected "
            f"{QWEN_EDIT_PATCH_DIM} (= {QWEN_EDIT_LATENT_CHANNELS} channels x "
            f"{QWEN_EDIT_PATCH}x{QWEN_EDIT_PATCH} pack). This is img_in's input width; a different "
            f"one is a different checkpoint."
        )
    if rows.shape[1] != expected_grid[0] * expected_grid[1]:
        raise ValueError(
            f"invalid {name}: {rows.shape[1]} packed row(s) but the declared/derived latent grid "
            f"{expected_grid[0]}x{expected_grid[1]} prices "
            f"{expected_grid[0] * expected_grid[1]}. img_shapes and the row block would then "
            f"describe different images to the same RoPE clock."
        )
    return rows, expected_grid


def _text_payload(payload: Any, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read ``(encoder_hidden_states, encoder_hidden_states_mask)`` out of a text payload.

    The mask stays INTEGER (DIVERGE #2). It is not converted to an additive ``-inf`` float mask the
    way ``single_frame.py:184-185`` does, because ``qwen_image_edit_plus.py:329`` SUMS it to derive
    ``txt_seq_lens`` — an ``-inf``-filled mask sums to ``-inf`` and the text stream is silently
    mis-attended rather than loudly wrong.
    """
    if isinstance(payload, dict):
        embeds = None
        for key in ("prompt_embeds", "encoder_hidden_states", "embeddings", "text_embeds"):
            if payload.get(key) is not None:
                embeds = payload[key]
                break
        if embeds is None:
            raise ValueError(
                f"invalid {name} payload: no 'prompt_embeds'/'encoder_hidden_states'/'embeddings'/"
                f"'text_embeds' tensor in {sorted(payload)}."
            )
        mask = None
        for key in ("prompt_embeds_mask", "encoder_hidden_states_mask", "attention_mask"):
            if payload.get(key) is not None:
                mask = payload[key]
                break
    else:
        raise TypeError(
            f"invalid {name} payload: expected a dict carrying the Qwen2.5-VL embeddings and their "
            f"mask, got {type(payload).__name__}. The mask is not optional — txt_seq_lens is derived "
            f"from it (qwen_image_edit_plus.py:329) and a padded-length fallback would attend to "
            f"pad tokens."
        )
    if not isinstance(embeds, torch.Tensor):
        raise TypeError(f"invalid {name}: embeddings must be a torch.Tensor.")
    if mask is None:
        raise ValueError(
            f"invalid {name} payload: no 'prompt_embeds_mask'/'encoder_hidden_states_mask'/"
            f"'attention_mask'. txt_seq_lens = mask.sum(dim=1) (qwen_image_edit_plus.py:329), so "
            f"without the mask the model attends over the PADDED length — which for a short caption "
            f"is mostly pad tokens, silently."
        )
    if embeds.ndim == 2:
        embeds = embeds.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if embeds.ndim != 3 or mask.ndim != 2:
        raise ValueError(
            f"invalid {name}: expected embeddings [1, n_text, dim] and mask [1, n_text], got "
            f"{tuple(embeds.shape)} and {tuple(mask.shape)}."
        )
    if embeds.shape[1] != mask.shape[1]:
        raise ValueError(
            f"invalid {name}: embeddings carry {embeds.shape[1]} text token(s) but the mask covers "
            f"{mask.shape[1]}. txt_seq_lens is derived from the MASK, so the mismatch would price a "
            f"text stream of one length and attend over another."
        )
    return embeds, mask.to(torch.int64)


def _unwrap_model_output(model_output: Any) -> torch.Tensor:
    """The Qwen forward is called with ``return_dict=False`` and indexed ``[0]`` (``:344``)."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, Sequence) and len(model_output) >= 1:
        first = model_output[0]
        if isinstance(first, torch.Tensor):
            return first
    sample = getattr(model_output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    raise TypeError(
        f"invalid model_output: expected a tensor, a (sample, ...) sequence (the return_dict=False "
        f"form ai-toolkit indexes at qwen_image_edit_plus.py:344), or an object with a .sample "
        f"tensor; got {type(model_output).__name__}."
    )


def _masked_velocity_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Squared error, masked, normalized by the mask fraction -> ``[B,]`` (``ic_lora.py:289-291``)."""
    loss = (prediction.float() - target.float()).pow(2)
    m = mask.unsqueeze(-1).float()
    return loss.mul(m).mean(dim=[-2, -1]) / m.mean(dim=[-2, -1]).clamp(min=1e-8)


# --------------------------------------------------------------------------------------------------
# The strategy — dataloader batch -> packed Qwen image sequence, control block out of the loss.
# --------------------------------------------------------------------------------------------------


class QwenEditStrategy(TrainingStrategy):
    """Qwen-Image-Edit-2511 multi-slot control-image training strategy (model family #3).

    Turns one dataloader sample into the packed image sequence ``[target | ctrl_0 | ... | ctrl_n]``,
    noises the TARGET only, and computes the loss over the target PREFIX. Constructed from
    PRIMITIVES rather than from ``SignetConfig`` — the parameter names mirror the ``QwenEditConfig``
    fields so the Modal-side caller is a field-for-field read, while this module keeps the
    CPU-testable import tier (the injected-seam discipline, ``ic_lora.py`` / ``h3_ref.py`` precedent).

    The control block is CLEAN. Unlike ``h3_ref``'s ``t_visual_cond`` pin there is no documented
    noise-augmentation convention for Qwen's control latents — ai-toolkit encodes the control image
    and concatenates it verbatim (``qwen_image_edit_plus.py:280-316``), with no ``scale_noise`` and no
    per-row timestep. Inventing one here would be a training decision dressed as plumbing, so the
    rows go in as they came out of the VAE. Their per-row conditioning is expressed the way Qwen
    expresses it: they are outside the loss and outside the read-back slice.

    Args:
        control_slots: how many control slots every sample carries (1..3). FIXED across the corpus —
            a varying count changes the packed sequence length and therefore the row budget, and
            ``ctrl_img_N`` addressing in the caption is positional.
        blank_slot_fill: which synthetic image fills a gap. Recorded on the resolved slot and passed
            to ``blank_latent_fn``; this module never renders one.
        caption_dropout_rate: P(this sample trains against the EMPTY caption). Requires the empty
            caption's embeddings — see ``empty_text_conditions``. Zero (the default) is a no-op.
        pack_fn: the single transcription of the 2×2 pack, applied when a payload arrives as a
            ``[C, F, H, W]`` latent rather than as rows.
            ``conditioning/qwen_edit_packing.qwen_edit_image_rows`` is the real one; ``None`` (the
            default) REFUSES a latent-form payload rather than transcribing the transform twice.
        blank_latent_fn: builds the latent/rows for a slot the payload left empty, called as
            ``blank_latent_fn(slot_index=i, fill=...)``. ``None`` refuses a gap instead of
            synthesizing zeros — a black image's VAE latent is NOT the zero tensor, and zeros are a
            perfectly-shaped wrong answer.
        sigma_fn: draws the flow-matching sigma when the batch does not carry one, called as
            ``sigma_fn(batch_size)``. Never defaulted: a silent ``sigma = 0`` trains every step at
            zero noise, which converges to a loss of zero and an adapter that has learned nothing.
        empty_text_conditions: the empty-caption text payload used by ``caption_dropout_rate``.
        max_packed_rows: realized-sequence ceiling (belt to the config-load braces). ``None`` or 0
            disables it — which is the shipped default, because no OOM boundary has been MEASURED
            for this model on any card in this program and a synthesized one would read as safety.
        device / dtype: compute placement (cuda / bf16 on Modal; cpu / float32 in unit tests).
    """

    def __init__(
        self,
        *,
        control_slots: int = 3,
        blank_slot_fill: str = "black",
        caption_dropout_rate: float = 0.0,
        pack_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        blank_latent_fn: Callable[..., Any] | None = None,
        sigma_fn: Callable[[int], Any] | None = None,
        empty_text_conditions: Any = None,
        max_packed_rows: int | None = None,
        device: Any = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not 1 <= control_slots <= QWEN_EDIT_MAX_CONTROL_SLOTS:
            raise ValueError(
                f"invalid control_slots {control_slots}: expected 1.."
                f"{QWEN_EDIT_MAX_CONTROL_SLOTS} (ai-toolkit's Edit-Plus template addresses "
                f"ctrl_img_1..{QWEN_EDIT_MAX_CONTROL_SLOTS})."
            )
        if blank_slot_fill not in QWEN_EDIT_BLANK_FILLS:
            raise ValueError(
                f"invalid blank_slot_fill {blank_slot_fill!r}: expected one of "
                f"{list(QWEN_EDIT_BLANK_FILLS)}."
            )
        if not 0.0 <= caption_dropout_rate <= 1.0:
            raise ValueError(
                f"invalid caption_dropout_rate {caption_dropout_rate}: expected 0.0..1.0."
            )
        self.control_slots = control_slots
        self.blank_slot_fill = blank_slot_fill
        self.caption_dropout_rate = caption_dropout_rate
        self.pack_fn = pack_fn
        self.blank_latent_fn = blank_latent_fn
        self.sigma_fn = sigma_fn
        self.empty_text_conditions = empty_text_conditions
        self.max_packed_rows = max_packed_rows
        self.device = device
        self.dtype = dtype

    def get_data_sources(self) -> Sequence[Any]:
        """The three qwen_edit precomputed sources this strategy reads.

        One fewer than H3 (``h3_ref.py:773-782``) — there is no audio branch — and one more than the
        LTX siblings' ``["latents", "conditions"]``, the extra being the control block. See
        ``QWEN_EDIT_DATA_SOURCES`` for why every name carries the family prefix and why
        ``qwen_edit_conditions`` must stay OUT of the video-latent allowlist.
        """
        return list(QWEN_EDIT_DATA_SOURCES)

    def _resolve_sigma(self, batch: Any, batch_size: int) -> torch.Tensor:
        """Read the per-sample sigma, or draw it through the injected seam. Never defaulted to 0.

        ⚠ A deliberate divergence from ``h3_ref.py:883-884``, which spells
        ``float(batch.get("t_video", 0.0))``. On H3's INVERTED clock ``t = 0`` is pure noise, so that
        default is merely extreme; on Qwen's noise-ward clock ``sigma = 0`` is CLEAN — the model
        would be asked to denoise an unnoised image at every step, the loss would collapse toward
        zero, and the run would look healthy while training nothing.
        """
        for key in ("sigma", "sigmas", "t"):
            if key in batch and batch[key] is not None:
                sigma = batch[key]
                break
        else:
            if self.sigma_fn is None:
                raise ValueError(
                    "batch carries no 'sigma' and no sigma_fn was injected. The qwen_edit timestep "
                    "distribution is the trainer's decision (ai-toolkit trains this family under the "
                    "bsmntw bell curve, NOT a uniform draw), so it is supplied rather than assumed "
                    "here — and it is never defaulted, because sigma = 0 is CLEAN on Qwen's "
                    "noise-ward clock and would train every step against an unnoised target."
                )
            sigma = self.sigma_fn(batch_size)
        sigma = torch.as_tensor(sigma, device=self.device, dtype=self.dtype).reshape(-1)
        if sigma.numel() != batch_size:
            raise ValueError(
                f"sigma has {sigma.numel()} element(s) for batch size {batch_size}: Qwen takes one "
                f"scalar timestep PER BATCH ITEM (qwen_image_edit_plus.py:336), not per token."
            )
        return sigma

    def _control_rows(
        self, slots: Sequence[QwenControlSlot], entries: Sequence[Any]
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        """Gather the packed rows for each resolved slot, in index order.

        A slot the payload supplied is read from the payload. A slot the resolver marked blank is
        built through ``blank_latent_fn`` — and REFUSED outright when that seam is absent, rather
        than filled with zeros. The VAE latent of a black image is a specific tensor, not a zero one;
        zeros would pack to the right shape, price the right number of rows, sail through every
        guard here, and teach the adapter that "no control" looks like a thing the VAE never emits.
        """
        by_index: dict[int, Any] = {}
        for position, entry in enumerate(entries):
            raw_index = _entry_get(entry, "slot")
            by_index[position if raw_index is None else int(raw_index)] = entry

        rows: list[torch.Tensor] = []
        grids: list[tuple[int, int]] = []
        for slot in slots:
            entry = by_index.get(slot.index)
            if entry is None:
                if self.blank_latent_fn is None:
                    raise ValueError(
                        f"control slot {slot.index} has no entry in the payload and must be filled "
                        f"with the {slot.fill!r} blank, but no blank_latent_fn was injected. This is "
                        f"NOT filled with zeros: a {slot.fill!r} image has a real VAE latent, zeros "
                        f"are not it, and a zero block packs to the correct shape and prices the "
                        f"correct number of rows — it would pass every check here and teach the "
                        f"adapter that an empty slot looks like something the encoder never emits. "
                        f"Encode the blanks in prep (the payload then carries them, blank=True), or "
                        f"inject blank_latent_fn."
                    )
                entry = self.blank_latent_fn(slot_index=slot.index, fill=slot.fill)
            slot_rows, grid = _rows_and_grid(entry, f"control slot {slot.index}", self.pack_fn)
            rows.append(slot_rows.to(self.device, dtype=self.dtype))
            grids.append(grid)
        return rows, grids

    def _text(
        self, batch: Any, batch_size: int, generator: torch.Generator | None
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Read the text conditioning and apply the caption-dropout draw.

        ⚠ Caption dropout SWAPS the caption for the empty one; it never zeroes the embeddings. A
        zeroed ``encoder_hidden_states`` is not the unconditional embedding of any text encoder — the
        empty string encodes to a specific, non-zero sequence — so zeroing trains the adapter toward
        a point in embedding space the sampler will never visit.

        ⚠ ai-toolkit HARD-DISABLES caption dropout under text-embedding caching
        (``dataloader_mixins.py:402,417`` — taken on report from the config lane, not re-read here);
        signet refuses that combination at config load instead. This method therefore never sees a
        cached-and-dropping config, and the draw below is reachable only on the uncached path.
        """
        payload = _lookup(batch, QWEN_EDIT_TEXT_BATCH_KEYS, "text condition")
        dropped = False
        if self.caption_dropout_rate > 0.0:
            draw = torch.rand(1, generator=generator).item()
            dropped = draw < self.caption_dropout_rate
            if dropped:
                empty = batch.get("empty_text_conditions", self.empty_text_conditions)
                if empty is None:
                    raise ValueError(
                        f"caption_dropout_rate={self.caption_dropout_rate} fired for this sample but "
                        f"no empty-caption conditioning is available (batch key "
                        f"'empty_text_conditions', or the empty_text_conditions constructor "
                        f"argument). Dropout SWAPS the caption for the encoded empty string; it does "
                        f"not zero the embeddings, because zero is not the unconditional embedding "
                        f"of any text encoder and the sampler never visits it."
                    )
                payload = empty
        embeds, mask = _text_payload(payload, "text conditions")
        embeds = embeds.to(self.device, dtype=self.dtype)
        mask = mask.to(self.device)
        if embeds.shape[0] != batch_size:
            raise ValueError(
                f"text conditioning carries {embeds.shape[0]} item(s) but the latent batch is "
                f"{batch_size}."
            )
        return embeds, mask, dropped

    def prepare_training_inputs(
        self, batch: Any, *, generator: torch.Generator | None = None
    ) -> QwenEditModelInputs:
        """Turn one qwen_edit sample into the packed sequence, control block out of the loss.

        The assembly, in the order ``qwen_image_edit_plus.get_noise_prediction`` performs it:

          1. pack the TARGET latent -> ``[1, n_target, 64]`` and open ``img_shapes`` with its
             ``(1, H2, W2)`` (``:223-236``);
          2. noise the target at the sampled sigma, noise-ward (``custom_flowmatch_sampler.py:91``),
             and take the velocity target ``eps - x0`` (``qwen_image.py:389-392``);
          3. resolve the control slots, gather their rows, and APPEND each slot's ``(1, H2, W2)`` to
             ``img_shapes`` in slot order (``:290-306``);
          4. concatenate ``[noisy_target, *control]`` on the sequence axis — target FIRST
             (``:314-316``);
          5. build the loss mask ``True`` over the target prefix, ``False`` over the control suffix.

        Args:
            batch: a signet nested sample carrying the three ``qwen_edit_`` sources, plus ``sigma``
                (or an injected ``sigma_fn``) and optionally a fixed ``noise`` tensor.
            generator: RNG for the noise draw and the caption-dropout draw. ``None`` uses global
                state — acceptable in a smoke test, never in a reproducible run.
        """
        target_payload = _lookup(batch, QWEN_EDIT_TARGET_BATCH_KEYS, "target image latent")
        target_rows, target_grid = _rows_and_grid(target_payload, "target latents", self.pack_fn)
        target_rows = target_rows.to(self.device, dtype=self.dtype)
        batch_size, target_seq_len, _ = target_rows.shape

        # ── (2) noise the TARGET at the sampled sigma. Noise-ward: sigma = 0 is CLEAN. ────────────
        sigma = self._resolve_sigma(batch, batch_size)
        noise = batch.get("noise")
        noise = (
            torch.as_tensor(noise, device=self.device, dtype=self.dtype)
            if noise is not None
            else torch.randn(
                target_rows.shape, dtype=self.dtype, device=self.device, generator=generator
            )
        )
        if noise.shape != target_rows.shape:
            raise ValueError(
                f"supplied noise {tuple(noise.shape)} does not match the packed target "
                f"{tuple(target_rows.shape)}."
            )
        sigma_expanded = sigma.view(-1, 1, 1)
        noisy_target = (1.0 - sigma_expanded) * target_rows + sigma_expanded * noise
        # DIVERGE #5 — NOISE-ward velocity, the LTX convention, verified for this family rather than
        # inherited: qwen_image.py:389-392 returns (noise - batch.latents).
        video_targets = noise - target_rows

        # ── (3) control slots: resolve the plan, then gather rows in slot order ───────────────────
        control_payload = _lookup(
            batch, QWEN_EDIT_CONTROL_BATCH_KEYS, "control latent", required=False
        )
        entries = _control_entries(control_payload)
        slots = resolve_control_slots(
            _target_stem(batch),
            entries,
            control_slots=self.control_slots,
            blank_slot_fill=self.blank_slot_fill,
        )
        control_rows, control_grids = self._control_rows(slots, entries)
        control_slot_rows = tuple(int(rows.shape[1]) for rows in control_rows)
        control_seq_len = sum(control_slot_rows)

        # ── (1)+(3) img_shapes: the TARGET block first, then each control slot, in index order ────
        img_shapes = tuple(
            [(1, target_grid[0], target_grid[1])]
            + [(1, grid[0], grid[1]) for grid in control_grids]
        )

        # ── (4) the packed sequence. TARGET FIRST — the control block is a SUFFIX (DIVERGE #1) ────
        hidden_states = torch.cat([noisy_target, *control_rows], dim=1)
        seq_len = int(hidden_states.shape[1])
        if seq_len != target_seq_len + control_seq_len:
            raise RuntimeError(
                f"packed sequence is {seq_len} row(s) but the parts price "
                f"{target_seq_len} + {control_seq_len}. Re-checked because every downstream slice "
                f"(the loss prefix, the read-back at qwen_image_edit_plus.py:346) is positional."
            )
        if self.max_packed_rows and seq_len > self.max_packed_rows:
            raise ValueError(
                f"packed sequence is {seq_len} row(s), over the {self.max_packed_rows}-row ceiling "
                f"(target {target_seq_len} + control {control_seq_len} across "
                f"{self.control_slots} slot(s) {list(control_slot_rows)}). A blank fill is NOT free "
                f"— it encodes to a real latent and costs its rows like any other control."
            )

        embeds, mask, caption_dropped = self._text(batch, batch_size, generator)

        # ── (5) loss mask: True over the target PREFIX, False over the control SUFFIX ─────────────
        video_loss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        video_loss_mask[:, :target_seq_len] = True

        video_modality = Modality(
            enabled=True,
            latent=hidden_states,
            sigma=sigma,
            # Per-row timesteps do not exist in Qwen's wire format — the transformer takes ONE
            # scalar per batch item (:336) and derives per-row modulation from img_shapes. This view
            # broadcasts the scalar so the shared ModelInputs contract still reads sensibly; the
            # AUTHORITATIVE value is ``sigma`` / ``transformer_kwargs()["timestep"]``.
            timesteps=sigma.view(-1, 1).expand(-1, seq_len),
            # ⚠ NOT LTX's [B, 3, T, 2] RoPE patch bounds and NOT H3's [1, rows, 3] coordinates.
            # Qwen builds its RoPE INSIDE the transformer from img_shapes + txt_seq_lens, so the
            # positional specification is the (F, H2, W2) block list; this is a tensor VIEW of it,
            # one row per BLOCK, for contract compatibility only.
            positions=torch.tensor(img_shapes, dtype=torch.int64, device=self.device).unsqueeze(0),
            context=embeds,
            # INT 0/1, never additive -inf (DIVERGE #2) — txt_seq_lens is its SUM.
            context_mask=mask,
        )

        return QwenEditModelInputs(
            video=video_modality,
            audio=None,
            video_targets=video_targets,  # TARGET-ONLY: a missing loss slice fails on SHAPE
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=None,  # DIVERGE #1 — the control block is a SUFFIX; see the class docstring
            target_seq_len=target_seq_len,
            control_seq_len=control_seq_len,
            control_slots=tuple(slots),
            control_slot_rows=control_slot_rows,
            img_shapes=img_shapes,
            txt_seq_lens=tuple(int(n) for n in mask.sum(dim=1).tolist()),
            caption_dropped=caption_dropped,
        )

    def compute_loss(self, model_inputs: ModelInputs, model_output: Any) -> torch.Tensor:
        """Masked velocity loss over the target PREFIX — the L-3 guard, inverted for this family.

        ``ic_lora.py:287-288`` and ``h3_ref.py:1143-1144`` both slice
        ``model_output[:, ref_seq_len:, :]`` because their conditioning is a prefix. Qwen's is a
        suffix, so the slice is ``[:, :target_seq_len, :]`` — which is exactly what ai-toolkit reads
        back at ``qwen_image_edit_plus.py:346`` (``noise_pred[:, : raw_packed_latents.size(1)]``).
        Copying the sibling slice here would compute the loss over the CONTROL rows against the
        target's velocity: a shape error only if the two blocks happen to differ in length, and
        silence when they do not.

        The slice is a no-op on an already-read-back output (a caller that handed the sliced tensor
        gets the identity), so both call shapes are correct without a branch.

        ⚠ ``prediction.shape[1] >= target_seq_len`` is NOT a sufficient guard, and the first draft of
        this method shipped it: handed the CONTROL suffix (12,288 rows against a 4,096-row target)
        it passed the check and computed the loss over the first 4,096 control rows against the
        target's velocity — the precise silent failure DIVERGE #1 exists to prevent, at a perfectly
        ordinary loss value. The output length is therefore pinned to one of exactly two legal
        values: the FULL packed sequence, or the already-read-back target prefix.

        ⚠ Residual, and stated rather than papered over: when a single control slot happens to pack
        to exactly ``target_seq_len`` rows, the suffix and the read-back prefix are the same length
        and no shape check can separate them. Shape is the second line of defence here, never the
        first — the first is that the target block is written FIRST (see ``prepare_training_inputs``
        step 4) and read back FIRST, on both sides of the seam.
        """
        target_seq_len = getattr(model_inputs, "target_seq_len", None)
        if not target_seq_len:
            raise ValueError(
                "compute_loss needs QwenEditModelInputs.target_seq_len to find the loss prefix. A "
                "plain ModelInputs cannot carry it, and ref_seq_len is deliberately None for this "
                "family (the control block is a SUFFIX — slicing on ref_seq_len would keep the "
                "controls and drop the target)."
            )
        prediction = _unwrap_model_output(model_output)
        if prediction.ndim != 3:
            raise ValueError(
                f"invalid model_output: expected [B, rows, {QWEN_EDIT_PATCH_DIM}], got "
                f"{prediction.ndim}-D {tuple(prediction.shape)}."
            )
        control_seq_len = int(getattr(model_inputs, "control_seq_len", 0) or 0)
        packed_seq_len = target_seq_len + control_seq_len
        if prediction.shape[1] not in (target_seq_len, packed_seq_len):
            raise ValueError(
                f"model_output carries {prediction.shape[1]} row(s), which is neither the full "
                f"packed sequence ({packed_seq_len} = target {target_seq_len} + control "
                f"{control_seq_len}) nor the read-back target prefix ({target_seq_len}). The "
                f"{control_seq_len}-row case in particular is the suffix-slice inversion — the "
                f"control block kept and the target dropped — which a bare 'at least as long as the "
                f"target' check waves through and which then trains the adapter on its own "
                f"conditioning at an unremarkable loss value. qwen_image_edit_plus.py:346 reads back "
                f"the PREFIX (noise_pred[:, : raw_packed_latents.size(1)]), never the suffix."
            )
        target_prediction = prediction[:, :target_seq_len, :]  # DROP the control suffix (L-3)
        target_mask = model_inputs.video_loss_mask[:, :target_seq_len]
        return _masked_velocity_loss(
            target_prediction, model_inputs.video_targets, target_mask
        ).mean()


def _control_entries(payload: Any) -> list[Any]:
    """Read the per-slot entry list out of a control payload, or ``[]`` when the source is absent.

    ⚠ Mirrors ``h3_ref.py:557-590``'s warning for the same structural reason: the per-slot data lives
    under ``"controls"`` (or ``"references"``/``"slots"``), and a top-level ``"latents"`` key ALIASES
    slot 0 only — it exists so the payload satisfies ``data/precomputed.py``'s video-latent
    allowlist contract, whose ``_normalize_video_latents`` reads ``data["latents"]`` unconditionally.
    A consumer that reads only the top-level tensor sees ONE control where the sample carries three.

    An absent source returns ``[]``, which the resolver then turns into ``control_slots`` blank
    slots — and those are refused unless a ``blank_latent_fn`` exists. Absence is never silently a
    zero-slot sample: the sequence length, and therefore the row budget, is a function of the slot
    count.
    """
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("controls", "slots", "references"):
            entries = payload.get(key)
            if entries is not None:
                return list(entries)
        raise ValueError(
            f"invalid control payload: no 'controls'/'slots'/'references' list in {sorted(payload)}. "
            f"The per-slot data is a LIST; the top-level 'latents' key aliases slot 0 only (it "
            f"exists to satisfy the video-latent allowlist contract), so reading it would silently "
            f"see one control where the sample carries several."
        )
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        return list(payload)
    raise TypeError(
        f"invalid control payload: expected a dict with a 'controls' list or a sequence of slot "
        f"dicts, got {type(payload).__name__}."
    )


def _target_stem(batch: Any) -> str:
    """The target's file stem — the key the control set is matched on. Required, never inferred.

    ai-toolkit derives it from the target's path (``dataloader_mixins.py:979``,
    ``os.path.splitext(os.path.basename(img_path))[0]``) and then looks for that stem inside each
    configured control directory. By training time the directories are gone and only the precomputed
    payload remains, so the stem must have been carried forward — without it the 1:1 correspondence
    this strategy checks is unfalsifiable and the check would degrade into a no-op.
    """
    for key in ("stem", "target_stem", "sample_id"):
        value = batch.get(key)
        if value is not None:
            return str(value)

    # NESTED form — the one PrecomputedDataset actually produces. ``__getitem__`` stores each
    # source's whole payload dict under that source's OUTPUT KEY (``result[output_key] = data``,
    # data/precomputed.py:288), and prep writes ``payload["stem"]``
    # (prep/qwen_edit_encode.py:923), so the stem arrives nested rather than at the top level.
    #
    # The lookup uses the module's own ``QWEN_EDIT_*_BATCH_KEYS`` tuples rather than the directory
    # names. Those are NOT the same string: ``fns.py:5828`` builds ``data_sources`` as
    # ``{dir_name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[dir_name]}``, so the batch is keyed by the output
    # key (``qwen_edit_latent_conditions``) while the on-disk directory is ``qwen_edit_latents``.
    # Both spellings are already enumerated in those tuples, in lookup order, for exactly this
    # reason — and a hand-written list here would drift from them the moment either changes.
    #
    # TARGET first, control only as a fallback: the two agreeing is what this stem is used to
    # verify, so reading the control's copy preferentially would compare a value against itself and
    # pass unconditionally.
    for source in (*QWEN_EDIT_TARGET_BATCH_KEYS, *QWEN_EDIT_CONTROL_BATCH_KEYS):
        payload = batch.get(source)
        if isinstance(payload, dict):
            value = payload.get("stem")
            if value is not None:
                return str(value)

    path = batch.get("path")
    if path is not None:
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[0] if "." in name else name
    raise ValueError(
        "batch carries no 'stem'/'target_stem'/'sample_id'/'path', and neither "
        "batch['qwen_edit_latents'] nor batch['qwen_edit_control_latents'] carries a 'stem': "
        "control images are matched to targets BY STEM (dataloader_mixins.py:979-985), so without "
        "the target's stem the 1:1 check cannot run and would silently pass on a control set "
        "belonging to another sample. prep/qwen_edit_encode writes 'stem' into every payload it "
        "commits; a batch missing it was pre-encoded by something else, or by an older prep."
    )
