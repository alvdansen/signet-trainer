"""``SourceSpec`` — one member of a multi-source training set, and the extraction vocabulary.

THE THING BEING GENERALIZED. Timothy's Wan 2.1 method (``docs/source-methods/musubi-wan21/
wan21-dataset-config.toml``) puts the SAME video directory into a training run TWICE, with two
different cache directories, two resolutions and two extraction modes::

    [[datasets]] video_directory=.../Videos  resolution=[1280,720]  frame_extraction="head"
                 cache_directory=.../cache_1 target_frames=[1,21]
    [[datasets]] video_directory=.../Videos  resolution=[640,360]   frame_extraction="uniform"
                 cache_directory=.../cache_2 target_frames=[45]  frame_sample=2

That is a RESOLUTION-against-TEMPORAL-EXTENT trade inside one run: 45 frames at 720p is
unaffordable, so motion is learned cheap-and-wide at 640x360 while appearance is learned
expensive-and-narrow at 1280x720, and stills carry maximum detail at 1024x1024. The general object
underneath it is not "frame extraction" — it is:

    a SOURCE is a (corpus, geometry, sampling policy, repeat weight, cache identity) tuple,
    and a training set is a LIST of them.

Extraction is the VIDEO SPECIALIZATION of "sampling policy". Keeping that split is what lets an
image family (``qwen_edit``) use the source list for corpus balancing — 500 studio stills at
``num_repeats: 1`` beside 40 hero shots at ``num_repeats: 3``, which is upstream's own stated use
of the field ("useful to balance the multiple datasets with different sizes") — while refusing the
video vocabulary outright, because "the first N frames" names nothing on a family whose F is pinned
to 1 (``configs/qwen_image_edit.example.yaml:6-9``, ``schema.py`` qwen_edit frame law).

WHY THIS IS ITS OWN MODULE, and not fields typed inline in ``schema.py``:
  * It is imported by the MANIFEST layer (``DataConfig.sources``) *and* by a RUNNER translator
    (``runners/musubi_toml.py``), and a runner importing ``schema`` wholesale to reach one type
    would drag the whole 2268-line config surface behind it.
  * ``config -> runners`` is the wrong dependency direction, so the type cannot live in the
    runner either. It is a config concept: h3 and ltx consume the same list without musubi.

✅ WIRING LANDED (slice A). ``DataConfig.sources`` is exactly the optional sibling field this
docstring specified — ``sources: list[SourceSpec] | None = None``, importing this class rather than
redeclaring it — and ``None`` remains the default, so the ABSENCE of the key (not a re-expressed
one-element list) is the byte-identical path for all 18 pre-existing configs. The one config that
does declare it, ``configs/wan21_kaboom.example.yaml``, is new and opt-in;
``tests/test_multisource_verifier_gaps.py`` pins that allowlist so a 20th cannot appear by accident.

WHICH FAMILIES HONOUR IT. Only ``model.family == "wan"`` (the musubi-tuner RUNNER). ``sources`` is
REFUSED at config load on every signet-NATIVE family — see ``config/schema.py``'s wan arm and
``data/multisource.build_native_source_datasets``, the named symbol that would land native support.
The refusal is not squeamishness about video vocabulary on an image family: signet's native loop
reads ``DataConfig.preprocessed_data_root`` through ``data/precomputed.py`` and has NO consumer for
a source LIST at all, so accepting one would be accepting a config nothing honours — including the
corpus-balancing use (``num_repeats`` on ``kind: image``) that this module's own design notes
describe as the natural first native application.

CRITICAL — Anti-Pattern 6 / Pitfall 1: pydantic + stdlib only. No ``modal``, no ``torch``, no
filesystem touch during validation. ``directory`` and ``cache_root`` are carried as DATA and
checked Modal-side where the Volume is mounted, exactly as ``DataConfig.preprocessed_data_root``
is (``schema.py:128-132``).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: musubi floors every requested clip length to ``N*4+1`` (the Wan latent temporal stride). The
#: floor is SILENT, so signet refuses a non-conforming value rather than inheriting the repair —
#: an accepted ``30`` trains at ``29`` and the config then describes a run nobody performed.
MUSUBI_FRAME_MULTIPLE = 4

#: musubi floors each resolution edge to a multiple of this before bucketing (the Wan step). A
#: declared ``[640, 360]`` therefore TRAINS at 640x352.
#:
#: MOVED HERE from ``runners/musubi_toml.py`` in slice A, and the direction of the move is the
#: point: this is one of musubi's LAWS, and ``config/validators.validate_wan_training_dims`` (the
#: family's dims pre-screen) needs it at CONFIG-LOAD time. A ``config -> runners`` import is the
#: wrong dependency direction — the reason stated at the top of this module for why ``SourceSpec``
#: cannot live in the renderer either. The renderer re-exports it, so its own import path and every
#: message it prints are unchanged. Split of responsibility: this module owns musubi's LAWS, and
#: ``runners/musubi_toml.py`` owns musubi's SYNTAX.
MUSUBI_RESOLUTION_STEP = 16

#: musubi's own default for ``max_frames`` (``full`` mode only). Carried so a set-but-ignored
#: value is distinguishable from an unset one.
MUSUBI_DEFAULT_MAX_FRAMES = 129

#: A source ``id`` is minted into a filesystem path (the derived cache root) and printed in the
#: per-source census, so it is restricted to a safe single path segment.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def musubi_floor_frames(frames: int) -> int:
    """Return the clip length musubi would ACTUALLY use for ``frames``: ``(f-1)//4*4+1``.

    Used only to build refusal messages — signet never applies the floor itself, because silently
    training 29 frames for a declared 30 is the defect, not the fix.
    """
    return (frames - 1) // MUSUBI_FRAME_MULTIPLE * MUSUBI_FRAME_MULTIPLE + 1


class ExtractionMode(str, Enum):
    """How clips are drawn from one source. Five upstream modes + one signet-only member.

    THERE IS NO ``tail``. The upstream ``if/elif`` chain has exactly five branches; a sixth name
    invented here would render a ``frame_extraction`` string musubi's schema validator rejects,
    inside a metered container, after the weights are already resident.

    ==========  ====================================================  ==========================
    member      upstream behaviour                                    keys it reads
    ==========  ====================================================  ==========================
    ``head``    window from frame 0, one per element of target_frames  ``target_frames``
    ``chunk``   non-overlapping tiling from 0, remainder DROPPED       ``target_frames``
    ``slide``   overlapping windows at 0, stride, 2*stride, ...        ``target_frames``,
                                                                      ``frame_stride``
    ``uniform`` ``frame_sample`` windows, starts linspace(0, n-t)      ``target_frames``,
                                                                      ``frame_sample``
    ``full``    ONE clip, whole video, capped at ``max_frames``        ``max_frames``
                then floored to N*4+1. ``target_frames`` IGNORED.
    ``image``   signet-only; selects ``image_directory``. Never        --
                rendered as a ``frame_extraction`` value.
    ==========  ====================================================  ==========================

    ``chunk`` / ``slide`` / ``full`` are here from day one although the Kaboom corpus uses none of
    them: they are the same resolution-vs-extent trade at other sampling densities (``slide`` is
    the dense version of ``uniform``; ``chunk`` is exhaustive tiling; ``full`` is for short clips).
    Shipping only the two this corpus needs would encode THIS CORPUS in the vocabulary rather than
    THE METHOD, and re-adding one later would be a schema change instead of a config edit.

    ``image`` is a MEMBER rather than ``extraction: null`` because an Optional field admits
    ``kind: video, extraction: null``, which then has to be caught somewhere downstream. A total
    enum makes the coherence rule a two-line validator that fires at config load.
    """
    # ``(str, Enum)`` rather than ``StrEnum``: StrEnum is 3.11+, and while pyproject declares
    # requires-python >=3.11, the BENCH interpreter is the 3.10 conda env that every regression
    # baseline in this project was measured on. A module that cannot import there splits the suite
    # by interpreter — and this repo already has one such split (test_pyproject_supply_chain's
    # tomllib), which is exactly the thing that made "what does the suite report" ambiguous. The
    # mixin is equivalent for every use here and costs nothing.

    HEAD = "head"
    CHUNK = "chunk"
    SLIDE = "slide"
    UNIFORM = "uniform"
    FULL = "full"
    IMAGE = "image"


#: The video modes that draw their clip lengths from ``target_frames``. ``FULL`` is deliberately
#: absent: it reads ``max_frames`` and ignores ``target_frames`` entirely.
TARGET_FRAME_MODES: frozenset[ExtractionMode] = frozenset(
    {ExtractionMode.HEAD, ExtractionMode.CHUNK, ExtractionMode.SLIDE, ExtractionMode.UNIFORM}
)

#: Every mode that is legal on ``kind: video`` (i.e. everything except the signet-only ``image``).
VIDEO_MODES: frozenset[ExtractionMode] = frozenset(ExtractionMode) - {ExtractionMode.IMAGE}


class SourceSpec(BaseModel):
    """One (corpus, geometry, sampling policy, repeat weight, cache identity) member of a run.

    ``extra="forbid"`` mirrors ``schema.py::_Base`` — a hand-edited key typo must die at config
    load, not be silently ignored (T-01-CF1).

    EVERY VALIDATOR BELOW REFUSES A SILENT UPSTREAM REPAIR. musubi will happily floor a clip
    length, rewrite ``uniform`` to ``head``, or ignore a key that belongs to another mode, warning
    at most. Each of those turns an operator's declared config into a description of a run that
    never happened, which is the single defect class this project spends the most guards on. The
    one repair signet does NOT refuse is the resolution floor (see
    ``runners/musubi_toml.musubi_resolution_warnings``): refusing it would refuse Timothy's own
    working ``[640, 360]``, so it warns with the real trained number instead.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="LOAD-BEARING: the cache discriminator, the provenance tag on a sample, and "
        "the label on the per-source census. Minted into a path when cache_root is derived, so it "
        "must be a safe single path segment.",
    )
    kind: Literal["image", "video"] = Field(
        ...,
        description="Selects image_directory vs video_directory in the rendered musubi TOML.",
    )
    directory: str = Field(
        ...,
        description="Corpus directory. A STRING carried as DATA — NOT FS-validated locally; "
        "existence is checked Modal-side where the Volume is mounted (Pitfall 1).",
    )
    resolution: tuple[int, int] = Field(
        ...,
        description="[W, H]. An AREA BUDGET, not an exact size: musubi buckets from it and floors "
        "each edge to a multiple of 16, so [640, 360] actually trains at 640x352.",
    )
    extraction: ExtractionMode = Field(
        default=ExtractionMode.IMAGE,
        description="Sampling policy. The only legal value on kind: image is 'image'.",
    )
    target_frames: list[int] = Field(
        default_factory=list,
        description="A list of clip LENGTHS, each yielding its own clips. Every entry must be "
        "N*4+1. Read by head/chunk/slide/uniform; IGNORED by full.",
    )
    frame_sample: int = Field(
        default=1,
        ge=1,
        description="uniform ONLY: how many windows to draw. Must be >= 2 (1 is rewritten to head).",
    )
    frame_stride: int = Field(default=1, ge=1, description="slide ONLY: window start stride.")
    max_frames: int = Field(
        default=MUSUBI_DEFAULT_MAX_FRAMES, ge=1, description="full ONLY: clip length cap."
    )
    num_repeats: int = Field(
        default=1,
        ge=1,
        description="⚠ SEMANTICS DIFFER ACROSS THE RUNNER BOUNDARY. musubi is EPOCH-driven "
        "(--max_train_epochs 16), so num_repeats: 3 LENGTHENS the run threefold. signet's native "
        "loop is STEP-driven, so the same 3 changes MIXTURE PROPORTIONS at a fixed step budget and "
        "the wall clock is unmoved. An operator who reads one onto the other is wrong by 3x in "
        "either time or data balance.",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Per-source on musubi. Native families pin this to 1 (single-bucket law, "
        "validators.py::validate_batch_size), so a value != 1 is refused there, not here.",
    )
    cache_root: str | None = Field(
        default=None,
        description="Where this source's cache lands. None DERIVES <data_root>/cache/<id> — the "
        "default is derived, not named, so two views of one directory cannot collide by omission.",
    )
    caption_extension: str | None = Field(
        default=None,
        description="Per-source override; None inherits the run-level extension. Present because "
        "the TE cache is per-CLIP while captions are one .txt per video, so a caption naming a "
        "moment ('reaching toward each other') is false for a uniform window starting at frame 75.",
    )

    @model_validator(mode="after")
    def _check_coherence(self) -> SourceSpec:
        """Refuse every arrangement musubi would silently repair, ignore, or rewrite."""
        if not _ID_RE.match(self.id):
            raise ValueError(
                f"source id {self.id!r} is not a safe path segment - it is minted into a cache "
                f"path when cache_root is derived. Use [A-Za-z0-9][A-Za-z0-9._-]*."
            )
        for field_name in ("directory", "cache_root"):
            value = getattr(self, field_name)
            if value is not None and "\\" in value:
                raise ValueError(
                    f"source {self.id!r}: {field_name} {value!r} contains a backslash. These paths "
                    f"are carried verbatim into a TOML read inside a Linux container; a Windows "
                    f"path here resolves to nothing and the source contributes zero clips at a "
                    f"perfectly ordinary loss."
                )
        w, h = self.resolution
        if w < 1 or h < 1:
            raise ValueError(f"source {self.id!r}: resolution {list(self.resolution)} must be positive")

        # --- kind <-> extraction coherence -------------------------------------------------------
        if self.kind == "image" and self.extraction is not ExtractionMode.IMAGE:
            raise ValueError(
                f"source {self.id!r}: kind 'image' requires extraction: image, got "
                f"{self.extraction.value!r}"
            )
        if self.kind == "video" and self.extraction is ExtractionMode.IMAGE:
            raise ValueError(
                f"source {self.id!r}: kind 'video' cannot use extraction: image - that mode selects "
                f"image_directory. Choose one of "
                f"{sorted(m.value for m in VIDEO_MODES)}."
            )

        # --- target_frames: required where read, refused where ignored ---------------------------
        if self.extraction in TARGET_FRAME_MODES:
            if not self.target_frames:
                raise ValueError(
                    f"source {self.id!r}: extraction {self.extraction.value!r} reads target_frames "
                    f"and none were declared - musubi would produce no clips from this source."
                )
            bad = [f for f in self.target_frames if f < 1 or f % MUSUBI_FRAME_MULTIPLE != 1]
            if bad:
                raise ValueError(
                    f"source {self.id!r}: target_frames {bad} are not N*4+1 - musubi would SILENTLY "
                    f"floor them ((f-1)//4*4+1). Declare the floored value: "
                    f"{[musubi_floor_frames(f) for f in bad]}."
                )
        elif self.target_frames:
            ignored = "ignored entirely" if self.extraction is ExtractionMode.FULL else "not read"
            raise ValueError(
                f"source {self.id!r}: extraction {self.extraction.value!r} {ignored} by musubi, but "
                f"target_frames {self.target_frames} were declared. Remove the key, or choose a "
                f"mode that reads it ({sorted(m.value for m in TARGET_FRAME_MODES)})."
            )

        # --- per-mode keys: set-but-ignored is a config that describes a run nobody performed -----
        if self.extraction is ExtractionMode.UNIFORM:
            if self.frame_sample == 1:
                raise ValueError(
                    f"source {self.id!r}: uniform with frame_sample=1 is rewritten to 'head' by "
                    f"musubi with only a warning. Declare head, or frame_sample >= 2."
                )
        elif self.frame_sample != 1:
            raise ValueError(
                f"source {self.id!r}: frame_sample={self.frame_sample} is read ONLY by uniform; "
                f"under {self.extraction.value!r} it is silently ignored."
            )

        if self.extraction is not ExtractionMode.SLIDE and self.frame_stride != 1:
            raise ValueError(
                f"source {self.id!r}: frame_stride={self.frame_stride} is read ONLY by slide; "
                f"under {self.extraction.value!r} it is silently ignored."
            )

        if self.extraction is ExtractionMode.FULL:
            if self.max_frames % MUSUBI_FRAME_MULTIPLE != 1:
                raise ValueError(
                    f"source {self.id!r}: max_frames={self.max_frames} is not N*4+1 - musubi would "
                    f"SILENTLY floor it to {musubi_floor_frames(self.max_frames)}. Declare that."
                )
        elif self.max_frames != MUSUBI_DEFAULT_MAX_FRAMES:
            raise ValueError(
                f"source {self.id!r}: max_frames={self.max_frames} is read ONLY by full; under "
                f"{self.extraction.value!r} it is silently ignored."
            )

        if self.caption_extension is not None and not self.caption_extension.startswith("."):
            raise ValueError(
                f"source {self.id!r}: caption_extension {self.caption_extension!r} must start with "
                f"'.' - musubi appends it to the media stem verbatim."
            )
        return self

    def resolve_cache_root(self, data_root: str) -> str:
        """Return this source's cache directory: the declared ``cache_root``, else a DERIVED one.

        The derivation is ``<data_root>/cache/<id>``. Joined with ``/`` rather than
        ``os.path.join`` on purpose: this repo is developed on Windows and the joined string is a
        LINUX CONTAINER path, so ``os.path.join`` would emit ``\\`` and silently produce a
        directory the container never creates.

        ⛔ BOTH branches are NORMALISED, and the explicit one is why this matters. The trailing
        slash is not cosmetic here: ``check_cache_collisions`` compares resolved roots as strings,
        so ``/d/V/cache_1/`` and ``/d/V/cache_1`` — one directory — read as two, the collision gate
        reports clean, and musubi writes TWO different extractions' latents into ONE cache with no
        warning. That is the silent-corruption class this whole module exists to refuse, and THIS
        FEATURE PROVOKES IT: its defining move is hand-writing near-identical cache paths for a
        single source directory (``cache_1`` / ``cache_2`` off the same ``Videos/``), which is
        exactly when a stray slash gets typed. An earlier version normalised only the derived
        branch, leaving the hand-written path — the one an operator actually types — unguarded.
        """
        if self.cache_root is not None:
            return self.cache_root.rstrip("/")
        return f"{data_root.rstrip('/')}/cache/{self.id}"


# --------------------------------------------------------------------------------------------------
# Cross-source checks — pure, list-returning, $0
# --------------------------------------------------------------------------------------------------


def check_cache_collisions(sources: Sequence[SourceSpec], *, data_root: str) -> list[str]:
    """Return one message per pair of sources resolving to the SAME cache directory (empty = clean).

    Exposed as a pure list-returning check, not only as the renderer's exception, so the config-gap
    layer can report it BESIDE the other gaps at $0 rather than as a traceback out of a renderer
    the operator did not know was running. musubi raises on duplicate cache dirs anyway; sharing
    one would cross-claim, overwrite, and — with musubi's default cache pruning — mutually delete.

    MOVED HERE from ``runners/musubi_toml.py`` in slice A, unchanged line for line, for the reason
    :data:`MUSUBI_RESOLUTION_STEP` records: ``DataConfig`` must run this refusal at CONFIG LOAD and
    ``config -> runners`` is the wrong direction. It is the ONE declaration — the renderer imports
    it from here and keeps re-exporting it, so ``from signet_trainer.runners.musubi_toml import
    check_cache_collisions`` still resolves and both existing call sites are untouched. A second
    copy on the config side would be the ``SourceSpec``-declared-twice failure in miniature: two
    implementations of one refusal, diverging silently.
    """
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for source in sources:
        resolved = source.resolve_cache_root(data_root)
        first = seen.get(resolved)
        if first is None:
            seen[resolved] = source.id
            continue
        collisions.append(
            f"cache collision: source {source.id!r} and {first!r} both resolve to {resolved!r}"
        )
    return collisions


def check_duplicate_ids(sources: Sequence[SourceSpec]) -> list[str]:
    """Return one message per repeated source ``id`` (empty = clean).

    A SEPARATE check from :func:`check_cache_collisions` even though duplicate ids usually produce a
    cache collision too, because they do not ALWAYS: two sources sharing ``id: motion`` while naming
    DIFFERENT explicit ``cache_root``s pass the collision gate cleanly and then make the id — which
    ``SourceSpec.id`` documents as "the provenance tag on a sample, and the label on the per-source
    census" — ambiguous in every artifact that carries it. The census would report one line for two
    corpora and no gate would have said anything.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for source in sources:
        if source.id in seen:
            duplicates.append(
                f"duplicate source id {source.id!r}: ids are the provenance tag and the census "
                f"label, so two sources cannot share one"
            )
        seen.add(source.id)
    return duplicates
