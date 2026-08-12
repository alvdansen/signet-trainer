"""conditioning.h3_ref — MiniMax-H3 ref2v conditioning: slot selection + ``H3RefStrategy``.

This is the headline value add of Phase 10. **Nobody has trained Ref2VA** (double-confirmed, P10-0a
/ P10-0d), so the reference pathway is BUILD work, not port work — the LTX strategies are an
analogy to reason from, never code to copy.

The module splits in two, deliberately:

  * **Pure selection helpers** (top of the file) — no ``torch``, no tensors, no I/O. Which references
    a segment gets, in what order, and whether they are dropped. All of it is a decision record: the
    operator's locked D-10-* rulings expressed as code that a test can prove.
  * **``H3RefStrategy``** (bottom) — the ``TrainingStrategy`` that turns a dataloader batch into the
    packed H3 batch, with the reference prefix out of the loss.

⛔ EXACTLY 2 SLOTS, ALWAYS — THE ENVIRONMENT REFERENCE SUBSTITUTES, IT NEVER APPENDS
------------------------------------------------------------------------------------
Operator ruling, verbatim: *"for the three reference just remove one character reference."*

So a non-environment segment (~76 of 88) carries **two rotating character refs** (AB / AC / BC), and
an environment-bearing segment (~12 of 88) carries **ONE rotating character ref plus the environment
ref**. This restores the operator's original SCOPE rule verbatim — *"use all 3 but mix and match
them, only using like 2 at a time to be thoughtful"* — which the environment axis had been layered
on top of rather than folded into.

**D-10-ASYM is unaffected and still honored:** the reference REGIME still varies across the corpus
(character+character vs character+environment), which is the point — *"we dont want to bias and lock
a particular reference behavior."* What no longer varies is the reference COUNT. A 3-slot sample was
never priced by the packed-sequence VRAM budget (``conditioning/h3_geometry``,
``validate_h3_reference_budget``), so it would not fail locally — it would OOM a metered A100.

Why the rotation exists at all
------------------------------
It is **not frugality**. Feeding all three references on every sample makes conditioning CONSTANT
across the corpus, so the model cannot learn what varies and is free to copy the reference wholesale
— the ``[precedent]`` copy-collapse failure this house already paid for once on LTX. Rotating pairs
make identity the INVARIANT. Dropout (D-10-REFDROP) and non-identity pairing are the other two
levers; see ``reference_dropout_for``.

CRITICAL — Anti-Pattern 6, the CONFINED tier
--------------------------------------------
Module scope is ``torch`` + stdlib + two signet siblings that are themselves confined
(``conditioning/strategy``, ``conditioning/h3_geometry``, ``train/h3_step``). There is **no**
``modal`` and **no** ``diffusers`` import, and deliberately **no** ``config/schema`` import either —
the Modal-side caller reads ``cfg.h3`` and constructs the strategy from primitives (the injected-seam
discipline, ``ic_lora.py`` precedent). That is what keeps the whole reference pathway CPU-unit-
testable on Windows with zero GPU and zero Modal spend.

D-10-NOMINE — no environment-reference mining
---------------------------------------------
The four mapped illustrations are it. Nothing here discovers, globs or walks a reference directory;
references arrive as arguments. The remaining 56 environment images stay unmined by decision.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from signet_trainer.conditioning.h3_geometry import H3PackedLayout
from signet_trainer.conditioning.h3_geometry import H3Reference as H3GeometryReference
from signet_trainer.conditioning.strategy import Modality, ModelInputs, TrainingStrategy

# The cached text-conditioning FORMAT — shared verbatim with the writer (``modal/fns.py`` PHASE A),
# so the spans this strategy tags rows with and the spans the pre-encode measured are the same
# object rather than two readings of one document. torch + stdlib only; no cycle (that module
# imports nothing from ``conditioning``).
from signet_trainer.prep.h3_text_payload import read_h3_text_state
from signet_trainer.train.h3_step import (
    H3_VISUAL_CONDITION_PIN,
    H3PackedBatch,
    H3StepDeps,
    build_h3_packed_batch,
)

__all__ = [
    "H3_AUDIO_BATCH_KEYS",
    "H3_DATA_SOURCES",
    "H3_REFERENCE_BATCH_KEYS",
    "H3_REFERENCE_KINDS",
    "H3_TARGET_BATCH_KEYS",
    "H3_TEXT_BATCH_KEYS",
    "H3ModelInputs",
    "H3Reference",
    "H3RefStrategy",
    "REFERENCE_PAIRS",
    "assign_character_ref",
    "assign_reference_pair",
    "order_reference_slots",
    "reference_dropout_for",
    "resolve_reference_slots",
]


# --------------------------------------------------------------------------------------------------
# Pure selection — stdlib only, no tensors. This half is the DECISION RECORD.
# --------------------------------------------------------------------------------------------------

#: The three 2-of-3 character pairings, in canonical order: AB / AC / BC.
#:
#: ⚠ This tuple is the copy-collapse mitigation, not an optimization. Feeding all three refs on every
#: sample makes conditioning CONSTANT across the corpus — the model cannot learn what varies and is
#: free to copy the reference wholesale. Rotating pairs make identity the INVARIANT.
REFERENCE_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))

#: The reference kinds the slot ordering understands. ``character`` and ``prop`` are ordered together
#: by scene composition; ``environment`` is always LAST (D-10-REFORDER).
H3_REFERENCE_KINDS: tuple[str, ...] = ("character", "environment", "prop")

#: Domain-separation tags so the pair stream and the single-character stream are INDEPENDENT draws
#: off the same seed. Without them, an environment segment's character pick would be correlated with
#: the pair a non-environment segment at the same index would have received.
_PAIR_DOMAIN = "h3-reference-pair"
_CHARACTER_DOMAIN = "h3-character-ref"
_DROPOUT_DOMAIN = "h3-reference-dropout"

#: blake2b digest width, and the divisor that maps a digest onto ``[0, 1)``.
_DIGEST_BYTES = 8
_DIGEST_SCALE = float(1 << (8 * _DIGEST_BYTES))


def _stable_digest_int(*parts: object) -> int:
    """A process-STABLE integer derived from ``parts`` — blake2b, never the built-in salted one.

    D-10-PAIRSEED demands assignments that reproduce across runs AND across containers. Python's
    built-in string hashing is salted per process (``PYTHONHASHSEED`` is random by default), so an
    assignment built on it reproduces perfectly inside one interpreter and silently differs inside a
    Modal container — every sample would get a different reference pair than the one the local dry
    run showed. blake2b is keyless, deterministic, and stdlib.
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x1f")  # unit separator — keeps ("a", "bc") distinct from ("ab", "c")
    return int.from_bytes(digest.digest(), "big")


def _stable_unit_interval(*parts: object) -> float:
    """``_stable_digest_int`` mapped onto ``[0.0, 1.0)`` — the deterministic uniform draw."""
    return _stable_digest_int(*parts) / _DIGEST_SCALE


def _seeded_permutation(n: int, *parts: object) -> list[int]:
    """A deterministic permutation of ``range(n)``, Fisher-Yates driven by ``_stable_digest_int``.

    Written out rather than delegated to ``random.Random`` so the ordering depends on nothing but
    blake2b — no reliance on the Mersenne-Twister stream layout staying stable across interpreter
    versions, which is the sort of assumption that only breaks in production.
    """
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = _stable_digest_int(*parts, "swap", i) % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def _assign_combination(
    segment_index: int,
    n_refs: int,
    k: int,
    *,
    seed: int,
    domain: str,
) -> tuple[int, ...]:
    """Round-robin over the ``k``-subsets of ``range(n_refs)``, block-permuted by ``seed``.

    The schedule is exactly balanced by construction: every complete block of ``len(combos)``
    segments contributes each subset exactly once, so over ``N`` segments each subset occurs either
    ``N // len(combos)`` or that plus one. At the Phase-10 numbers — 88 segments, 3 pairs — that is
    the 29/29/30 the plan names.

    The per-BLOCK permutation (rather than one global offset) is what makes the seed load-bearing:
    with a single offset there would be only ``len(combos)`` distinguishable seeds, so "a different
    seed gives a different schedule" would be false for most seed pairs.
    """
    if segment_index < 0:
        raise ValueError(f"invalid segment_index {segment_index}: must be >= 0.")
    if k < 1:
        raise ValueError(f"invalid subset size {k}: must be >= 1.")
    if n_refs < k:
        raise ValueError(
            f"invalid n_refs {n_refs}: cannot choose {k} distinct reference(s) from a pool of "
            f"{n_refs}. Every H3 sample carries a FIXED number of reference slots, so a pool too "
            f"small to fill them is a dataset error, not something to silently pad."
        )

    combos = tuple(itertools.combinations(range(n_refs), k))
    block, position = divmod(int(segment_index), len(combos))
    order = _seeded_permutation(len(combos), domain, seed, block)
    return combos[order[position]]


def assign_reference_pair(
    segment_index: int,
    *,
    n_refs: int = 3,
    seed: int = 42,
) -> tuple[int, int]:
    """The 2-of-3 rotation: which TWO character references segment ``segment_index`` gets.

    Returns a 2-tuple of DISTINCT indices into the character-reference list. At the default
    ``n_refs=3`` every returned pair is one of ``REFERENCE_PAIRS`` and, over the corpus's 88
    segments, the three pairs occur 29 / 29 / 30 times.

    **D-10-PAIRSEED — fixed, seeded, reproducible.** This is a deliberate choice with a known
    trade-off: a fresh random draw per epoch would give strictly more pair coverage, while a fixed
    schedule is reproducible and debuggable (the same segment always shows the same pair, so a bad
    sample can be reproduced from its index alone). **Fixed is the decision, and it is revisitable**
    — if the reference pathway underfits, more coverage is one of the levers.

    The assignment uses blake2b, never Python's built-in salted hashing: see ``_stable_digest_int``.
    """
    if n_refs < 2:
        raise ValueError(
            f"invalid n_refs {n_refs}: a 2-of-N rotation needs at least 2 character references."
        )
    pair = _assign_combination(segment_index, n_refs, 2, seed=seed, domain=_PAIR_DOMAIN)
    return (pair[0], pair[1])


def assign_character_ref(segment_index: int, *, n_refs: int = 3, seed: int = 42) -> int:
    """The ENVIRONMENT-segment path: which SINGLE character reference this segment gets.

    An environment-bearing segment carries one rotating character ref plus the environment ref — the
    environment SUBSTITUTES for the second character slot rather than being appended, so exactly one
    character index is needed. Balanced by the same block round-robin as ``assign_reference_pair``
    (29/29/30 over 88 segments), on an INDEPENDENT seed stream so the two schedules do not correlate.
    """
    if n_refs < 1:
        raise ValueError(f"invalid n_refs {n_refs}: need at least one character reference.")
    return _assign_combination(segment_index, n_refs, 1, seed=seed, domain=_CHARACTER_DOMAIN)[0]


@dataclass(frozen=True)
class H3Reference:
    """One reference the model will be shown: where it lives, what it IS, and how big it is.

    ⚠ **Not the same class as ``conditioning.h3_geometry.H3Reference``**, which is the *pricing*
    record (``width``/``height``/``label`` only — everything the VRAM budget needs and nothing more).
    This one adds the two facts selection needs and geometry does not care about: ``kind`` (which
    drives D-10-REFORDER) and ``subject_id`` (which groups multiple images of one character).
    ``to_geometry_reference`` is the one legal bridge between them, so a caller prices the references
    it actually resolved rather than a nominal pair.

    Fields:
        path: Volume-relative path to the reference image. Relative by contract — client-property
            hygiene keeps real reference filenames out of tracked files and out of logs.
        kind: ``"character"`` / ``"environment"`` / ``"prop"``.
        subject_id: Stable identity for the depicted subject. Two images of the same character share
            a ``subject_id`` and are therefore grouped adjacently by ``order_reference_slots``.
        width / height: SOURCE pixel size AFTER D-10-CROP (crop, never pad — padding injects a
            synthetic border the adapter would learn — and the crop must save the face).
    """

    path: str
    kind: Literal["character", "environment", "prop"]
    subject_id: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.kind not in H3_REFERENCE_KINDS:
            raise ValueError(
                f"invalid reference kind {self.kind!r} for {self.path!r}: expected one of "
                f"{H3_REFERENCE_KINDS}. The kind drives D-10-REFORDER (environment LAST), so an "
                f"unrecognized one would silently order as a character."
            )

    def to_geometry_reference(self) -> H3GeometryReference:
        """Project onto the ``h3_geometry`` pricing record so the resolved slots can be priced.

        ``conditioning/h3_geometry`` owns every packed-sequence number in this repo and must never be
        handed a foreign type. The label it carries is what a budget refusal names (``C+008``), so
        ``subject_id`` is what flows through.
        """
        return H3GeometryReference(width=self.width, height=self.height, label=self.subject_id)


def resolve_reference_slots(
    segment_index: int,
    character_references: Sequence[H3Reference],
    environment_reference: H3Reference | None,
    *,
    references_per_sample: int = 2,
    seed: int = 42,
) -> list[H3Reference]:
    """THE SUBSTITUTION RULE, and the single place in the repo it lives.

    Returns EXACTLY ``references_per_sample`` references for every input:

      * ``environment_reference is None`` -> the two rotating character refs from
        ``assign_reference_pair``;
      * otherwise -> ONE rotating character ref from ``assign_character_ref``, plus the environment
        reference. **The environment reference SUBSTITUTES for the second character slot; it is
        never appended.** Operator ruling, verbatim: *"for the three reference just remove one
        character reference."*

    D-10-ASYM is preserved through the REGIME variation (character+character vs
    character+environment), NOT through a variable count. The count is invariant; the kind mix is
    not. That is the asymmetry the operator called *"actually ideal"* — it stops the adapter binding
    to a fixed reference regime.

    Why the length check is a ``raise`` and not an ``assert``: a 3-slot return is the specific
    blocker this function exists to prevent, and it does not fail locally — it OOMs a metered A100,
    because ``validate_h3_reference_budget`` priced a 2-reference pairing domain. A bare ``assert``
    disappears under ``python -O``, and this guard must not.
    """
    if references_per_sample < 1:
        raise ValueError(
            f"invalid references_per_sample {references_per_sample}: must be >= 1."
        )

    characters = list(character_references)
    n_characters_needed = references_per_sample - (1 if environment_reference is not None else 0)
    if n_characters_needed < 1:
        raise ValueError(
            f"invalid references_per_sample {references_per_sample}: with an environment reference "
            f"present there is no character slot left. The environment reference SUBSTITUTES for "
            f"the LAST character slot — it is never appended — so at least 2 slots are needed for "
            f"an environment-bearing sample."
        )
    if len(characters) < n_characters_needed:
        raise ValueError(
            f"cannot fill {references_per_sample} reference slot(s) for segment {segment_index}: "
            f"the sample needs {n_characters_needed} character reference(s) but the pool holds "
            f"{len(characters)}. Every H3 sample carries a FIXED slot count; a short pool is a "
            f"dataset error, not something to pad."
        )

    if n_characters_needed == 1:
        picked: tuple[int, ...] = (
            assign_character_ref(segment_index, n_refs=len(characters), seed=seed),
        )
    elif n_characters_needed == 2:
        picked = assign_reference_pair(segment_index, n_refs=len(characters), seed=seed)
    else:  # pragma: no cover — Phase 10 fixes references_per_sample at 2
        picked = _assign_combination(
            segment_index, len(characters), n_characters_needed, seed=seed, domain=_PAIR_DOMAIN
        )

    slots = [characters[i] for i in picked]
    if environment_reference is not None:
        slots.append(environment_reference)

    if len(slots) != references_per_sample:
        raise RuntimeError(
            f"resolve_reference_slots produced {len(slots)} slot(s) for segment {segment_index}, "
            f"expected exactly {references_per_sample}. The environment reference SUBSTITUTES for "
            f"the last character slot and is never appended, so a 3-slot sample cannot be correct. "
            f"A wrong count was never priced by the packed-sequence budget check "
            f"(config/validators.validate_h3_reference_budget) and would surface as a CUDA OOM on a "
            f"metered A100 rather than as an error here."
        )
    return slots


def order_reference_slots(references: Sequence[H3Reference]) -> list[H3Reference]:
    """D-10-REFORDER: characters/props in scene-composition order, environment reference LAST.

    Characters and props keep the order they were given — that order IS the house convention
    (rightmost-intended subject first, then leftward, characters and props intermixed in that spatial
    order) — with all images sharing a ``subject_id`` pulled adjacent to the first one. Environment
    references then close the list, in their given order.

    ⚠ **Two facts about ordering, from P10-0d section 4.2, that a reader must not conflate:**

      1. **H3 imposes NO slot semantics.** Nothing in the architecture or the docs says "slot 1 is
         the subject" or "the last slot is the scene". The binding of MEANING happens in the PROMPT
         (MiniMax's ``<Subject N>`` / ``<Picture N>`` labels and its ``subject_definitions`` block),
         not by position. Relying on position alone would be relying on a convention the model was
         never taught.
      2. **Order is nonetheless load-bearing TWICE OVER.** It fixes the ``<Picture i>`` labels of the
         prompt presentation AND it advances the shared rotary clock (an image reference consumes one
         integer ``t`` unit), so *a different order is a different request*.

    Together those make the house order a REAL training variable rather than a cosmetic choice: it
    must be applied CONSISTENTLY between training and inference, or the adapter is being asked a
    different question at sample time than it was taught at train time.
    """
    refs = list(references)
    foreground = [r for r in refs if r.kind != "environment"]
    environment = [r for r in refs if r.kind == "environment"]

    ordered: list[H3Reference] = []
    seen: set[str] = set()
    for reference in foreground:
        if reference.subject_id in seen:
            continue  # already emitted with its subject group
        seen.add(reference.subject_id)
        ordered.extend(r for r in foreground if r.subject_id == reference.subject_id)

    return ordered + environment


def reference_dropout_for(
    segment_index: int,
    step: int,
    *,
    dropout: float = 0.2,
    seed: int = 42,
) -> bool:
    """D-10-REFDROP: does THIS ``(segment, step)`` sample train with its references dropped?

    A deterministic seeded draw — same inputs, same answer, in this process and in a container.

    **0.2 is deliberately more aggressive than the field.** Precedent is ~10%; ours is doubled
    because the corpus is small (88 segments) and a small corpus is exactly where a reference pathway
    can memorize instead of generalize. ⚠ **If the reference pathway UNDERFITS, this is the first
    suspect** — it is a config field (``h3.reference_dropout``), so lowering it is a YAML edit.

    Dropout is one of THREE copy-collapse levers, and they are complementary:

      1. this dropout — sometimes the model must generate with NO reference at all;
      2. the 2-of-3 rotation (``assign_reference_pair``) — which two refs vary across the corpus,
         so identity is the invariant rather than the pixels;
      3. non-identity pairs — by construction here, since the references are hand-drawn STILLS while
         the targets are video segments, so no sample is ever "reconstruct this exact image".

    The draw is per ``(segment, step)``, not per segment: a per-segment-only draw would freeze each
    segment into one regime for the whole run, which is the opposite of what dropout is for.
    """
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"invalid dropout {dropout}: must be a probability in [0.0, 1.0].")
    if dropout <= 0.0:
        return False
    if dropout >= 1.0:
        return True
    return _stable_unit_interval(_DROPOUT_DOMAIN, seed, segment_index, step) < dropout


# --------------------------------------------------------------------------------------------------
# The strategy — the dataloader batch -> packed H3 batch, reference prefix out of the loss
# --------------------------------------------------------------------------------------------------

#: The four precomputed sources an H3 ref2v run reads (10-04 committed the dir names).
H3_DATA_SOURCES: tuple[str, ...] = (
    "h3_latents",
    "h3_conditions",
    "h3_reference_latents",
    "h3_audio_latents",
)

#: Batch keys per source, in lookup order: the ``_PRECOMPUTED_SOURCE_OUTPUT_KEYS``-style output key
#: first, then the list-form dir-name==output-key fallback. Mirrors ``a2v.py::_AUDIO_BATCH_KEYS``.
H3_TARGET_BATCH_KEYS: tuple[str, ...] = ("h3_latent_conditions", "h3_latents")
H3_TEXT_BATCH_KEYS: tuple[str, ...] = ("h3_text_conditions", "h3_conditions")
H3_REFERENCE_BATCH_KEYS: tuple[str, ...] = ("h3_ref_latent_conditions", "h3_reference_latents")
H3_AUDIO_BATCH_KEYS: tuple[str, ...] = ("h3_audio_latent_conditions", "h3_audio_latents")


@dataclass
class H3ModelInputs(ModelInputs):
    """``ModelInputs`` plus the three things an H3 batch has and an LTX one does not.

    ``ModelInputs`` stays the shared contract (this IS one — ``isinstance`` holds), so the loop, the
    dry-run and every existing consumer see the familiar shape. The additions exist because H3's real
    wire format is the packed ten-kwarg dict, not a ``Modality`` pair:

        packed             the ``H3PackedBatch``; ``packed.kwargs`` is what ``model(**kwargs)`` takes
        references         the resolved slots, in final order — what the sample was actually shown
        reference_dropped  whether D-10-REFDROP fired for this ``(segment, step)``

    Without ``packed`` the kwargs this strategy just assembled would be unreachable and the training
    function would have to rebuild them, which is exactly the duplication that lets a wire format
    drift in two places.
    """

    packed: H3PackedBatch | None = None
    references: tuple[H3Reference, ...] = ()
    reference_dropped: bool = False


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
        f"H3RefStrategy.get_data_sources() declares — a source configured but not mapped silently "
        f"drops the whole sample from the index rather than raising."
    )


def _as_rows(value: Any, name: str, patch_size: tuple[int, int, int] | None = None) -> torch.Tensor:
    """Coerce to the ``[1, rows, feature]`` ROW-MAJOR form the packed batch consumes.

    A 4-D ``[C, F, H, W]`` (or 5-D batched) latent — what ``PrecomputedDataset`` returns for an
    ``h3_``-prefixed video-latent source — is patchified ONLY when ``patch_size`` is supplied, and
    then only through ``conditioning/h3_packing.h3_video_rows``, which is a verbatim transcription of
    diffusers' ``patchify_video_latents`` at the pinned ``DIFFUSERS_SHA``.

    ⚠ With ``patch_size=None`` a 4-D latent is still REFUSED. That is deliberate and it is the
    default: the reshape is not the hard part, the intra-patch ELEMENT ORDER is, and it is a
    checkpoint contract — a wrong order corrupts training silently at the correct shape. So the
    caller must say, explicitly and with the patch size read off the LIVE model config, that it
    wants the transcribed transform applied. Nothing here infers one.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"invalid {name}: expected a torch.Tensor, got {type(value).__name__}.")
    if value.ndim >= 4 and patch_size is not None:
        from signet_trainer.conditioning.h3_packing import h3_video_rows  # noqa: PLC0415 — see below

        # Function-local so ``h3_ref`` keeps its own module scope free of the numpy dependency
        # ``h3_packing`` needs for the exact float64 linspace, and so a caller that hands this
        # strategy rows never pays for the import at all.
        return h3_video_rows(value, patch_size)
    if value.ndim >= 4:
        raise ValueError(
            f"invalid {name}: got a {value.ndim}-D {tuple(value.shape)} latent, but the H3 packed "
            f"batch consumes ROW-MAJOR [1, rows, feature] tensors. The [C, F, H, W] -> rows "
            f"patchify (patch_dim = in_channels * prod(patch_size)) is NOT applied because no "
            f"patch_size was supplied: the intra-patch element ORDER is a checkpoint contract and a "
            f"wrong order corrupts training silently at the correct shape. Construct the strategy "
            f"with patch_size read off the LIVE model config "
            f"(H3RefStrategy(..., patch_size=tuple(model.config.patch_size))) — never a literal — "
            f"or patchify upstream with conditioning/h3_packing.h3_video_rows."
        )
    tensor = value.unsqueeze(0) if value.ndim == 2 else value
    if tensor.ndim != 3:
        raise ValueError(
            f"invalid {name}: expected [1, rows, feature] or [rows, feature], got "
            f"{value.ndim}-D {tuple(value.shape)}."
        )
    if tensor.shape[0] != 1:
        raise ValueError(
            f"invalid {name}: the leading batch dimension must be 1 (H3 packs ONE sample per "
            f"sequence), got {tensor.shape[0]} in {tuple(tensor.shape)}."
        )
    return tensor


def _payload_rows(
    payload: Any, name: str, patch_size: tuple[int, int, int] | None = None
) -> torch.Tensor:
    """Accept a bare tensor or a ``{"latents": ...}`` / ``{"embeddings": ...}`` payload dict."""
    if isinstance(payload, torch.Tensor):
        return _as_rows(payload, name, patch_size)
    if isinstance(payload, dict):
        for key in ("latents", "embeddings", "hidden_states"):
            if payload.get(key) is not None:
                return _as_rows(payload[key], name, patch_size)
        raise ValueError(
            f"invalid {name} payload: no 'latents'/'embeddings'/'hidden_states' tensor in "
            f"{sorted(payload)}."
        )
    raise TypeError(f"invalid {name} payload: expected a tensor or a dict, got {type(payload).__name__}.")


def _parse_reference_pool(
    payload: Any, patch_size: tuple[int, int, int] | None = None
) -> list[tuple[H3Reference, torch.Tensor]]:
    """Read the CANDIDATE POOL out of the ``h3_reference_latents`` payload, descriptors included.

    ⚠ The real per-slot data is ``payload["references"]`` — a LIST. The payload's top-level
    ``"latents"`` key ALIASES slot 0 only; it exists purely to satisfy the ``_VIDEO_LATENT_SOURCE_DIRS``
    allowlist contract (``_normalize_video_latents`` reads ``data["latents"]`` unconditionally), and
    the two slots cannot be one tensor because they encode at DIFFERENT sizes. **A consumer that
    reads only the top-level tensor silently sees one reference instead of two.**

    ⚠ ``H3Reference.width``/``height`` are SOURCE PIXEL sizes. A ``prep/h3_encode`` slot carries
    BOTH: ``source_wh`` (source pixels) and ``height``/``width`` (the LATENT grid, ~1/32 the size).
    ``source_wh`` therefore WINS when present — reading the latent dims into an ``H3Reference``
    would type-check, order correctly and then under-price the reference by three orders of
    magnitude the moment anything called ``to_geometry_reference()``. The ``width``/``height``
    fallback keeps hand-built pools (which carry source dims under those names) working.
    """
    if isinstance(payload, dict):
        entries = payload.get("references")
        if entries is None:
            raise ValueError(
                "invalid reference payload: no 'references' list. The per-slot data lives under "
                "'references'; the top-level 'latents' key ALIASES slot 0 only (it exists to "
                "satisfy the _VIDEO_LATENT_SOURCE_DIRS allowlist), so reading it would silently "
                "see one reference where the sample carries several."
            )
    elif isinstance(payload, Sequence) and not isinstance(payload, str | bytes):
        entries = list(payload)
    else:
        raise TypeError(
            f"invalid reference payload: expected a dict with a 'references' list or a sequence of "
            f"slot dicts, got {type(payload).__name__}."
        )

    pool: list[tuple[H3Reference, torch.Tensor]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(
                f"invalid reference slot {index}: expected a dict carrying the descriptor fields "
                f"plus 'latents', got {type(entry).__name__}."
            )
        reference = entry.get("reference")
        if not isinstance(reference, H3Reference):
            has_size = "source_wh" in entry or {"width", "height"} <= set(entry)
            missing = [k for k in ("path", "kind", "subject_id") if k not in entry]
            if not has_size:
                missing.append("source_wh (or width+height)")
            if missing:
                raise ValueError(
                    f"invalid reference slot {index}: missing {missing}. Selection needs 'kind' "
                    f"(D-10-REFORDER puts the environment LAST) and 'subject_id' (multiple images "
                    f"of one character are grouped), so a bare latent tensor is not enough. "
                    f"prep/h3_encode.encode_h3_reference_latents propagates all three from the "
                    f"manifest; a cache written before that did is unreadable and must be "
                    f"RE-ENCODED (nothing here can infer them, and guessing 'kind' reorders the "
                    f"references, which the shared rotary clock makes a different request)."
                )
            source = entry.get("source_wh")
            width, height = (
                (int(source[0]), int(source[1]))
                if source is not None
                else (int(entry["width"]), int(entry["height"]))
            )
            reference = H3Reference(
                path=str(entry["path"]),
                kind=entry["kind"],
                subject_id=str(entry["subject_id"]),
                width=width,
                height=height,
            )
        if "latents" not in entry:
            raise ValueError(f"invalid reference slot {index}: no 'latents' tensor.")
        pool.append(
            (reference, _as_rows(entry["latents"], f"reference slot {index} latents", patch_size))
        )
    return pool


def _split_model_output(model_output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The H3 forward returns ``(video_out, audio_out)`` with ``return_dict=False``; accept both."""
    if isinstance(model_output, torch.Tensor):
        return model_output, None
    if isinstance(model_output, Sequence) and len(model_output) == 2:
        return model_output[0], model_output[1]
    raise TypeError(
        f"invalid model_output: expected a video tensor or a (video, audio) 2-tuple, got "
        f"{type(model_output).__name__}."
    )


def _masked_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Squared error, masked, normalized by the mask fraction -> ``[B,]`` (``ic_lora.py:289-291``)."""
    loss = (prediction.float() - target.float()).pow(2)
    m = mask.unsqueeze(-1).float()
    return loss.mul(m).mean(dim=[-2, -1]) / m.mean(dim=[-2, -1]).clamp(min=1e-8)


class H3RefStrategy(TrainingStrategy):
    """MiniMax-H3 ref2v training strategy — the reference pathway, end to end (H3-05 / H3-06).

    Turns one dataloader sample into the packed ten-kwarg H3 batch: resolve the reference slots
    (EXACTLY ``references_per_sample``, environment SUBSTITUTING never appending), apply
    D-10-REFDROP, noise the target, and hand ``train/h3_step.build_h3_packed_batch`` the pieces. The
    loss then drops the reference prefix and nothing else.

    Constructed from PRIMITIVES, never from ``SignetConfig`` — the parameter names mirror 10-05's
    ``H3Config`` fields so the Modal-side caller is a field-for-field read, while this module keeps
    the CPU-testable import tier (the injected-seam discipline, ``ic_lora.py`` precedent).

    ⛔ DIVERGE #1 — the reference prefix is NOT clean-at-timestep-0.
        ``ic_lora.py:221-222`` (``ref_timesteps = torch.zeros(...)``) is the ONE line that must not
        be copied: H3 pins visual conditioning rows at ``max(t, t_visual_cond)`` instead. And per
        **D-10-REFPIN** the ``0.999`` value is an INFERENCE convention (``keyframe_noise_aug``), so
        the training value is a PARAMETER here, not a constant; regularization happens in IMAGE space
        (affine jitter, dropout, cross-pair construction) and latent-noise "poisoning" of the
        reference is explicitly rejected. The prefix latents go in VERBATIM.

    ⛔ DIVERGE #2 — video-only means loss-MASKING, and the a2v analog is INVERTED.
        ``a2v.py:20-24`` freezes audio CLEAN at per-token timestep 0 and leaves ``audio_targets`` /
        ``audio_loss_mask`` at ``None``. H3 is the mirror image: target audio rows are PRESENT and
        NOISED at the audio timestep (matching the inference regime) and merely excluded from the
        loss. Reuse the ``ModelInputs`` FIELDS, never a2v's semantics. Because all 50 blocks are
        shared, a video-only-loss LoRA still moves weights the audio path uses — **audio drift is an
        eval axis, not an assumption**. Measured: 0 of 44 corpus clips carry an audio stream.

    ⛔ DIVERGE #3 — RoPE positions.
        ``ic_lora.py:229-239`` builds pixel coords twice through ltx_core and concatenates on dim 2.
        H3 uses 3-axis float64 RoPE with h/w area-normalized to a fixed 32-unit extent and ``t``
        absolute starting at ``text_len``, each image reference consuming 1.0 ``t`` unit, applied
        partially over **96 of 128** head dims. Pattern port only, ZERO LTX code reuse — so
        ``position_ids`` arrives as an argument (from the batch, or from the injected
        ``position_ids_fn``), never fabricated here.

    ⛔ DIVERGE #4 — modality tags.
        No LTX analog at all. ``0 = video, 1 = text, 2 = audio``, addressed
        ``timestep_indices * H3_MODALITY_NUM + token_tags``: a **checkpoint contract**, not a
        convention, so a wrong tag silently modulates the wrong AdaLN rows. Qwen vision rows live in
        the TEXT stream but modulate as VIDEO. Owned by ``train/h3_step.h3_token_tags``; this
        strategy passes the ``vision_spans`` through.

    ⛔ DIVERGE #5 — the VELOCITY SIGN is reversed.
        Every LTX strategy in this repo trains ``target = noise - clean`` (``ic_lora.py:277-292``
        and siblings), the noise-ward flow-match convention where ``x0 = x_t - sigma * v``. **H3
        predicts a DATA-ward velocity**: ``scheduling_minimax_h3.py`` L20-21 states it
        (*"MiniMax-H3's transformer predicts a data-ward velocity, so ``x0 = x_t + sigma * v``
        instead of diffusers' ``x0 = x_t - sigma * v``"*) and ``MiniMaxH3Scheduler.step`` L269
        implements it as ``denoised = sample + sigma_from_timestep * model_output`` — *"note the
        ``+``, the opposite of the usual flow-match convention"*. Solving with
        ``x_t = t*x0 + (1-t)*eps`` and ``sigma = 1 - t`` gives ``v = x0 - eps``, so the training
        target is ``clean - noise``. Carrying LTX's ``noise - clean`` here trains the adapter to
        predict the NEGATION of what the sampler consumes: identical shapes, a perfectly ordinary
        loss curve, and an adapter that pushes every render the wrong way.

    Args:
        references_per_sample: Fixed slot count (2). The environment ref SUBSTITUTES for the second
            character slot rather than being appended. 0 = NO-REFERENCE training (ALPHA): the
            reference source is never read, the text conditions on the PROMPT-ONLY state, and the
            packed batch carries zero conditioning-video rows — through the same packer.
        reference_dropout: D-10-REFDROP probability, per ``(segment, step)``.
        reference_pair_seed: D-10-PAIRSEED. One seed feeds three domain-separated streams (pair,
            single-character, dropout), so the schedules are reproducible AND uncorrelated.
        environment_ref_last: D-10-REFORDER. Applies ``order_reference_slots`` to the resolved set.
        audio_in_loss: D-10-AUDIO. ``False`` keeps target audio present, noised, and unmasked-out.
        t_visual_cond: the reference pin handed to ``build_h3_packed_batch``, applied there as
            ``max(t_video, t_visual_cond)``. A training decision (D-10-REFPIN), never a hardcode.
        deps: the injected ``H3StepDeps`` supplying ``patch_dim`` / ``audio_in_channels`` /
            ``text_dim`` from the LIVE model config, so no architecture number is restated.
        position_ids_fn: builds ``[seq_len, 3]`` RoPE coordinates when the batch does not carry them.
            ``conditioning/h3_packing.make_h3_position_ids_fn`` is the real one.
        patch_size: the transformer's ``(t, h, w)`` patch, read off ``model.config.patch_size``.
            Supplying it lets the strategy patchify a ``[C, F, H, W]`` precomputed payload through
            the transcribed ``h3_packing.h3_video_rows``. ``None`` (the default) keeps the 10-08
            behaviour of REFUSING a 4-D latent — the transform is opt-in because a guessed
            intra-patch order corrupts training silently at the correct shape.
        max_packed_rows: the realized-``seq_len`` ceiling (belt to the config-load braces).
        expected_layout: optional priced ``H3PackedLayout`` to cross-check the realized row counts.
        target_audio_rows: ``h3_geometry.h3_audio_rows(target_frames)`` — how many target-audio rows
            a sample carries. Used ONLY when a sample has no audio stream, where the block is
            synthesized as pure noise rather than dropped (dropping it makes the packed sequence
            disagree with the budget and the RoPE builder, both of which price it unconditionally).
            Falls back to ``expected_layout`` / the ``H3PositionIdsBuilder``, and refuses if neither
            is available — it is never defaulted to 0.
    """

    def __init__(
        self,
        *,
        references_per_sample: int = 2,
        reference_dropout: float = 0.2,
        reference_pair_seed: int = 42,
        environment_ref_last: bool = True,
        audio_in_loss: bool = False,
        t_visual_cond: float = H3_VISUAL_CONDITION_PIN,
        deps: H3StepDeps | None = None,
        position_ids_fn: Callable[..., torch.Tensor] | None = None,
        patch_size: tuple[int, int, int] | None = None,
        max_packed_rows: int | None = None,
        expected_layout: H3PackedLayout | None = None,
        target_audio_rows: int | None = None,
    ) -> None:
        self.target_audio_rows = None if target_audio_rows is None else int(target_audio_rows)
        self.references_per_sample = references_per_sample
        self.reference_dropout = reference_dropout
        self.reference_pair_seed = reference_pair_seed
        self.environment_ref_last = environment_ref_last
        self.audio_in_loss = audio_in_loss
        self.t_visual_cond = t_visual_cond
        self.deps = deps
        self.position_ids_fn = position_ids_fn
        self.patch_size = None if patch_size is None else tuple(int(p) for p in patch_size)
        self.max_packed_rows = max_packed_rows
        self.expected_layout = expected_layout

    def get_data_sources(self) -> Sequence[Any]:
        """The H3 precomputed sources this strategy reads — four for Ref2VA, three at zero slots.

        Mirrors ``ic_lora.py:102-108``'s third-source declaration, one source further: H3 needs a
        FOURTH (``h3_audio_latents``) because target audio rows stay in the packed sequence even
        though they are out of the loss (D-10-AUDIO). All four are ``h3_``-prefixed so they can never
        be paired with the LTX-encoded sources of the same role — different VAE, different text
        encoder, silently wrong at the correct shape.

        NO-REFERENCE (ALPHA, ``references_per_sample == 0``): ``h3_reference_latents`` is NOT
        declared — ``PrecomputedDataset`` raises on a configured source whose dir does not exist, so
        declaring it would make a no-reference cache (which has no reference dir) unloadable. A
        REF-BEARING cache stays consumable: the dataset simply never reads its reference dir.
        """
        if self.references_per_sample == 0:
            return [name for name in H3_DATA_SOURCES if name != "h3_reference_latents"]
        return list(H3_DATA_SOURCES)

    def _absent_target_audio_rows(self) -> int:
        """How many NOISE rows an audio-less sample contributes — never guessed, never zero.

        Read in priority order from the two places that already know, and REFUSED if neither was
        supplied. ``h3_audio_rows(target_frames)`` is a function of the campaign's frame count, which
        this strategy is not otherwise told; deriving it from the target latents is impossible after
        the patchify, and defaulting it to 0 is precisely the defect (the packed budget and the RoPE
        builder both price the rows unconditionally).
        """
        if self.target_audio_rows is not None:
            return self.target_audio_rows
        if self.expected_layout is not None:
            return int(self.expected_layout.n_target_audio)
        builder_rows = getattr(self.position_ids_fn, "n_target_audio_rows", None)
        if builder_rows is not None:
            return int(builder_rows)
        raise ValueError(
            "this sample carries no audio stream, so its target-audio rows must be SYNTHESIZED as "
            "noise — but the row count is unknown. h3_geometry.h3_audio_rows(target_frames) is a "
            "function of the campaign frame count, and both h3_packed_seq_len and "
            "H3PositionIdsBuilder price those rows unconditionally, so emitting zero of them makes "
            "the packed sequence disagree with every consumer that priced it. Construct the "
            "strategy with target_audio_rows=h3_audio_rows(config.training_dims[2]), or with an "
            "expected_layout / a make_h3_position_ids_fn builder — never a literal."
        )

    def _require_deps(self) -> H3StepDeps:
        if self.deps is None:
            raise ValueError(
                "H3RefStrategy needs an H3StepDeps to build a batch: patch_dim / audio_in_channels "
                "/ text_dim are read from the LIVE model config so the trainer never restates an "
                "architecture number. Build one with train/h3_step.h3_step_deps_from_model(model) "
                "(the arch gate already holds the loaded transformer — re-loading 61.7 GiB would be "
                "an expensive mistake)."
            )
        return self.deps

    def _resolve_slots(
        self,
        segment_index: int,
        pool: Sequence[tuple[H3Reference, torch.Tensor]],
    ) -> list[H3Reference]:
        """Resolve THIS segment's reference set: substitution, then D-10-REFORDER."""
        characters = [reference for reference, _ in pool if reference.kind != "environment"]
        environments = [reference for reference, _ in pool if reference.kind == "environment"]

        allowed_environments = self.references_per_sample - 1
        if len(environments) > allowed_environments:
            raise ValueError(
                f"got {len(environments)} environment reference(s) but at most "
                f"{allowed_environments} can be carried with references_per_sample="
                f"{self.references_per_sample}: the environment reference SUBSTITUTES for the last "
                f"character slot, it is never appended, so it cannot displace every character. "
                f"A sample with no character reference was never priced by the packed-sequence "
                f"budget check."
            )

        slots = resolve_reference_slots(
            segment_index,
            characters,
            environments[0] if environments else None,
            references_per_sample=self.references_per_sample,
            seed=self.reference_pair_seed,
        )
        if self.environment_ref_last:
            slots = order_reference_slots(slots)
        if len(slots) != self.references_per_sample:
            raise RuntimeError(
                f"resolved {len(slots)} reference slot(s) for segment {segment_index}, expected "
                f"exactly {self.references_per_sample}. Re-checked here after ordering because a "
                f"wrong count OOMs a metered A100 rather than failing locally."
            )
        return slots

    def prepare_training_inputs(  # noqa: PLR0914 — one linear assembly, splitting it hides the order
        self,
        batch: Any,
        *,
        generator: torch.Generator | None = None,
    ) -> H3ModelInputs:
        """Turn one H3 sample into the packed batch, with the reference prefix out of the loss.

        The reference latents enter VERBATIM (D-10-REFPIN) and only the TARGET rows are noised, at
        H3's INVERTED time convention — ``t = 1 - sigma``, so ``t = 1`` is CLEAN and
        ``x_t = t * x0 + (1 - t) * eps``. That is the same convention the ``0.999`` visual pin lives
        in (``0.999 * ref + 0.001 * noise``), which is why the pin reads as "almost clean".

        The velocity target is ``clean - noise`` (DIVERGE #5), NOT the LTX ``noise - clean``: H3
        predicts a DATA-ward velocity.
        """
        deps = self._require_deps()
        if "segment_index" not in batch:
            raise ValueError(
                "batch is missing 'segment_index': the 2-of-3 rotation and D-10-REFDROP are both "
                "deterministic functions OF the segment index (D-10-PAIRSEED). Without it the "
                "assignment would silently become order-dependent and irreproducible."
            )
        segment_index = int(batch["segment_index"])
        step = int(batch.get("step", 0))
        t_video = float(batch.get("t_video", 0.0))
        t_audio = float(batch.get("t_audio", 0.0))

        # ── target video: rows in, noised at t_video, TARGET-ONLY velocity out ───────────────────
        target = _payload_rows(
            _lookup(batch, H3_TARGET_BATCH_KEYS, "target video latent"),
            "target video latents",
            self.patch_size,
        )

        # NO-REFERENCE (ALPHA): zero slots means no dropout draw at all — there is no reference
        # regime to drop out of, and the schema already pinned reference_dropout at its default.
        no_reference = self.references_per_sample == 0
        # ⛔ The D-10-REFDROP draw is made HERE, before the text is read, because it now selects
        # WHICH text state this step conditions on. Previously the drop removed the reference LATENT
        # rows while the cached text kept describing both references — a regime that occurs nowhere
        # at inference, leaking identity into ~20% of steps.
        dropped = not no_reference and reference_dropout_for(
            segment_index,
            step,
            dropout=self.reference_dropout,
            seed=self.reference_pair_seed,
        )
        # ⛔ The vision spans come from the PAYLOAD, never from the batch. `batch.get("vision_spans")`
        # was read here and populated by nothing, so `h3_token_tags` tagged every Qwen vision row
        # TEXT — >90% of the text stream modulating with the wrong AdaLN row, silently, on every
        # step. `read_h3_text_state` also REFUSES the stale version-1 cache (a bare tensor), which is
        # what is on the Volume today.
        #
        # A NO-REFERENCE step reads the PROMPT-ONLY state for the same reason a dropped step does:
        # it is the presentation with no vision blocks, which is what makes the request genuinely
        # reference-free — and it is what keeps a REF-BEARING cache consumable by a no-reference
        # train (the reference-bearing state would describe references this run never shows).
        text_state, vision_spans = read_h3_text_state(
            _lookup(batch, H3_TEXT_BATCH_KEYS, "text condition"),
            reference_dropped=dropped or no_reference,
        )
        text = _as_rows(text_state, "text conditions")

        video_noise = batch.get("video_noise")
        video_noise = (
            _as_rows(video_noise, "video_noise")
            if video_noise is not None
            else torch.randn(
                target.shape, dtype=target.dtype, device=target.device, generator=generator
            )
        )
        noisy_target = t_video * target + (1.0 - t_video) * video_noise
        # DIVERGE #5 — DATA-ward velocity (``x0 = x_t + sigma * v`` =>  ``v = x0 - eps``).
        # ``scheduling_minimax_h3.py`` L20-21 + ``step`` L269. NOT the LTX ``noise - clean``.
        video_targets = target - video_noise

        # ── references: resolve, order, gather. CLEAN — no latent noise here (D-10-REFPIN) ───────
        if no_reference:
            # NO-REFERENCE (ALPHA): the reference source is NEVER read — not looked up, not parsed.
            # That is the whole train-side contract: a ref-bearing cache stays consumable because
            # its h3_reference_latents dir is simply ignored, and a no-reference cache needs none.
            # Zero rows through the SAME packer below; h3_row_timesteps / h3_loss_mask both handle
            # n_cond_video == 0 (the packing math is reused, never forked).
            references: list[H3Reference] = []
            reference_rows = torch.zeros(
                (1, 0, deps.patch_dim), dtype=target.dtype, device=target.device
            )
        elif dropped:
            # The pool is still PARSED on a dropped step (the cache contract is validated every
            # step, exactly as before) — only its rows go unused this step.
            _parse_reference_pool(
                _lookup(batch, H3_REFERENCE_BATCH_KEYS, "reference latent"), self.patch_size
            )
            # A genuine conditioning-REGIME variation, which is the point (D-10-ASYM): varying the
            # number and kind of references across the corpus stops the adapter binding to a fixed
            # reference regime, and is one of the three copy-collapse levers.
            references = []
            reference_rows = torch.zeros(
                (1, 0, deps.patch_dim), dtype=target.dtype, device=target.device
            )
        else:
            pool = _parse_reference_pool(
                _lookup(batch, H3_REFERENCE_BATCH_KEYS, "reference latent"), self.patch_size
            )
            references = self._resolve_slots(segment_index, pool)
            by_path = {reference.path: rows for reference, rows in pool}
            reference_rows = torch.cat([by_path[r.path] for r in references], dim=1)
            # D-10-REFPIN, the other half. The anchors used to enter VERBATIM while their rows were
            # tagged at t_visual_cond — `modular_pipeline.py::keyframe_noise_aug` at the pin states
            # the convention outright ("the released model was trained with its anchors very
            # slightly noised") and applies `scale_noise(ref, 0.999, noise)`. A 0.1%-amplitude
            # difference, very likely under the encode's own fp16 rounding, but it is a documented
            # deviation from a stated convention of the release and the pin is already a
            # parameterized decision here rather than a constant — so it is applied, not assumed
            # negligible. `t_visual_cond = 1.0` reproduces the old verbatim behaviour exactly.
            reference_noise = batch.get("reference_noise")
            reference_noise = (
                _as_rows(reference_noise, "reference_noise")
                if reference_noise is not None
                else torch.randn(
                    reference_rows.shape,
                    dtype=reference_rows.dtype,
                    device=reference_rows.device,
                    generator=generator,
                )
            )
            reference_rows = (
                self.t_visual_cond * reference_rows
                + (1.0 - self.t_visual_cond) * reference_noise
            )

        # ── audio: reference rows stay clean (pin 1.0, never noised); target rows are NOISED ─────
        audio_payload = _lookup(batch, H3_AUDIO_BATCH_KEYS, "audio latent", required=False)
        n_cond_audio = 0
        audio_absent = audio_payload is None or (
            isinstance(audio_payload, dict) and audio_payload.get("present") is False
        )
        if audio_absent:
            # ⛔ D-10-AUDIO / the absent-audio block. This used to emit ZERO target-audio rows, which
            # made the training path structurally unrunnable on a 0-of-44-audio corpus: both
            # `h3_packed_seq_len` and `H3PositionIdsBuilder` price `h3_audio_rows(frames)` (74 at
            # 22f) UNCONDITIONALLY, and the builder's hard `n_target_audio` check fails on 0. It also
            # contradicted `h3_loss_mask`'s own docstring ("target audio ROWS stay in the packed
            # sequence and stay NOISED").
            #
            # The block is therefore materialized exactly as inference step 0 has it: PURE NOISE at
            # the sampled `t_audio`, present in the attention context the video rows attend to —
            # which is the whole reason it must exist — and out of the loss, because there is no
            # ground-truth x0 to learn toward. Two rejected alternatives, both deliberate:
            #   * encoding real silence as the TARGET would teach the model to generate silence for
            #     every prompt — a behavioural change that would have to be untrained later;
            #   * dropping the rows and re-deriving the geometry would diverge structurally from the
            #     inference layout and invalidate the measured 12,394-row budget.
            n_absent_audio = self._absent_target_audio_rows()
            audio_rows = torch.zeros(
                (1, n_absent_audio, deps.audio_in_channels),
                dtype=target.dtype,
                device=target.device,
            )
        else:
            audio_rows = _payload_rows(audio_payload, "audio latents")
            if isinstance(audio_payload, dict):
                n_cond_audio = int(audio_payload.get("n_cond_audio", 0))

        audio_cond = audio_rows[:, :n_cond_audio, :]
        audio_clean_target = audio_rows[:, n_cond_audio:, :]
        audio_noise = batch.get("audio_noise")
        audio_noise = (
            _as_rows(audio_noise, "audio_noise")
            if audio_noise is not None
            else torch.randn(
                audio_clean_target.shape,
                dtype=audio_clean_target.dtype,
                device=audio_clean_target.device,
                generator=generator,
            )
        )
        if audio_absent:
            # PURE noise, NOT `(1 - t) * eps`. Interpolating toward a zero "clean" target would make
            # the block converge on silence as t -> 1, i.e. it would present silence as the thing
            # being generated — the exact behaviour the operator rejected. There is no x0 here, so
            # the rows carry the unit-variance draw at every t and stay out of the loss below.
            noisy_audio_target = audio_noise
        else:
            noisy_audio_target = t_audio * audio_clean_target + (1.0 - t_audio) * audio_noise
        audio_stack = torch.cat([audio_cond, noisy_audio_target], dim=1)
        audio_targets = audio_clean_target - audio_noise  # DIVERGE #5 — data-ward, as above
        if audio_absent and self.audio_in_loss:
            raise ValueError(
                f"audio_in_loss=True but this sample carries NO audio stream, so its "
                f"{int(audio_stack.shape[1])} target-audio row(s) are synthesized noise with no "
                f"ground truth behind them. Training on them would fit the adapter to a random "
                f"draw. Either set h3.audio_in_loss=False (the Phase-10 campaign does — 0 of 44 "
                f"corpus clips carry audio, D-10-AUDIO) or exclude audio-less clips from the "
                f"corpus. Refusing rather than silently masking a flag the config turned on."
            )

        # ── position_ids: an ARGUMENT (DIVERGE #3), never fabricated ─────────────────────────────
        n_text = int(text.shape[1])
        n_cond_video = int(reference_rows.shape[1])
        n_target_video = int(target.shape[1])
        n_audio = int(audio_stack.shape[1])
        seq_len = n_text + n_cond_video + n_audio + n_target_video
        position_ids = batch.get("position_ids")
        if position_ids is None:
            if self.position_ids_fn is None:
                raise ValueError(
                    "no position_ids in the batch and no position_ids_fn injected. H3's RoPE is "
                    "3-axis float64 with h/w area-normalized to a fixed 32-unit extent and t "
                    "absolute from text_len (each image reference consuming 1.0 t unit), applied "
                    "over 96 of 128 head dims — ZERO LTX code is reusable for it, and a monotone "
                    "stand-in would train the adapter against the wrong geometry with nothing "
                    "raising. Supply the coordinates; they are not fabricated here."
                )
            position_ids = self.position_ids_fn(
                n_text=n_text,
                references=tuple(references),
                n_cond_video=n_cond_video,
                n_cond_audio=n_cond_audio,
                n_target_audio=n_audio - n_cond_audio,
                n_target_video=n_target_video,
                seq_len=seq_len,
            )

        packed = build_h3_packed_batch(
            (reference_rows, noisy_target),
            audio_stack,
            text,
            n_cond_video,
            n_cond_audio,
            position_ids,
            vision_spans,
            t_video,
            t_audio,
            t_visual_cond=self.t_visual_cond,
            audio_in_loss=self.audio_in_loss,
            patch_dim=deps.patch_dim,
            audio_in_channels=deps.audio_in_channels,
            text_dim=deps.text_dim,
            max_packed_rows=self.max_packed_rows,
            expected_layout=self.expected_layout,
        )

        row_t = packed.kwargs["timestep"][packed.kwargs["timestep_indices"]]
        video_modality = self._modality(packed, "video_indices", row_t, t_video, text)
        audio_modality = self._modality(packed, "audio_indices", row_t, t_audio, text)

        # The two masks are ``train/h3_step.h3_loss_mask``'s output, carried on the H3PackedBatch.
        # Re-calling h3_loss_mask here would create a SECOND source of truth for the same row
        # counts — the one thing 10-06 is explicit about not doing (row counts are re-projected from
        # the packed layer, never re-derived in the strategy).
        return H3ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,  # TARGET-ONLY: a missing loss slice fails on SHAPE
            audio_targets=audio_targets,
            video_loss_mask=packed.video_loss_mask,
            audio_loss_mask=packed.audio_loss_mask,
            ref_seq_len=packed.n_cond_video,
            packed=packed,
            references=tuple(references),
            reference_dropped=dropped,
        )

    @staticmethod
    def _modality(
        packed: H3PackedBatch,
        index_key: str,
        row_t: torch.Tensor,
        sigma: float,
        text: torch.Tensor,
    ) -> Modality:
        """A ``Modality`` VIEW onto the packed rows, for ``ModelInputs`` contract compatibility.

        ⚠ ``positions`` here carries H3's ``[1, rows, 3]`` ``(t, h, w)`` RoPE coordinates, NOT LTX's
        ``[B, 3, T, 2]`` patch bounds — a different quantity in a differently-shaped tensor. The
        AUTHORITATIVE wire format is ``H3ModelInputs.packed.kwargs``; these views exist so the shared
        ``ModelInputs`` contract (and every consumer written against it) still holds.
        """
        indices = packed.kwargs[index_key]
        latent_key = "hidden_states" if index_key == "video_indices" else "audio_hidden_states"
        return Modality(
            latent=packed.kwargs[latent_key],
            sigma=torch.tensor([sigma], dtype=torch.float32),
            timesteps=row_t[indices].unsqueeze(0),
            positions=packed.kwargs["position_ids"][indices].unsqueeze(0),
            context=text,
        )

    def compute_loss(self, model_inputs: ModelInputs, model_output: Any) -> torch.Tensor:
        """Masked velocity loss over the TARGET rows only — the L-3 guard, on H3's contract.

        The transformer states the contract itself: *"Conditioning rows are returned unmasked —
        masking them out before the scheduler step is the caller's job"*
        (``transformer_minimax_h3.py`` L44-50). Inference discharges it by slicing
        ``[num_condition_video_rows:]``; TRAINING discharges it HERE. Leaving the prefix in is the
        dead-adapter bug — the objective becomes dominated by rows the model is not being asked to
        generate, and the adapter learns nothing controllable.

        Shape is the second line of defence: ``video_targets`` is TARGET-ONLY, so a missing slice
        fails loud rather than training on the prefix (``ic_lora.py:277-292``, same shape).

        The audio term is added ONLY when ``audio_in_loss`` is True; otherwise the audio mask is
        all-``False`` and contributes nothing, while the rows stay in the sequence (D-10-AUDIO).
        """
        video_output, audio_output = _split_model_output(model_output)
        reference_rows = model_inputs.ref_seq_len or 0

        target_prediction = video_output[:, reference_rows:, :]  # DROP the reference prefix (L-3)
        target_mask = model_inputs.video_loss_mask[:, reference_rows:]
        total = _masked_velocity_loss(
            target_prediction, model_inputs.video_targets, target_mask
        ).mean()

        if (
            self.audio_in_loss
            and audio_output is not None
            and model_inputs.audio_targets is not None
            and model_inputs.audio_loss_mask is not None
        ):
            n_cond_audio = getattr(model_inputs, "packed", None)
            n_cond_audio = n_cond_audio.n_cond_audio if n_cond_audio is not None else 0
            audio_prediction = audio_output[:, n_cond_audio:, :]
            audio_mask = model_inputs.audio_loss_mask[:, n_cond_audio:]
            total = total + _masked_velocity_loss(
                audio_prediction, model_inputs.audio_targets, audio_mask
            ).mean()

        return total
