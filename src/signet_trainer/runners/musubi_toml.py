"""The musubi-tuner dataset TOML — rendered from a signet source list, and parsed back.

WHAT THIS TRANSLATES, and why the translation is the whole feature. musubi's dataset config is a
``[general]`` table plus an ARRAY of ``[[datasets]]`` blocks, and the composition mechanism is that
the SAME directory may appear in more than one block with a DIFFERENT ``cache_directory``. That is
not incidental: the cache filename carries the SOURCE video's size, never the training resolution,
so two views of one corpus are two blocks precisely BECAUSE they are two caches. Timothy's Wan 2.1
config (``docs/source-methods/musubi-wan21/wan21-dataset-config.toml``) is exactly that — stills at
1024x1024, appearance at 1280x720 from frame 0, motion at 640x360 across the whole clip — and this
module's acceptance test renders it back byte-for-byte from a signet source list.

WHERE THE SEAM IS. The runner owns caching END TO END: ``wan_cache_latents.py`` and
``wan_cache_text_encoder_outputs.py`` read this TOML directly (``train_kohya.py:109``, ``:121``)
and ``wan_train_network.py`` reads it again (``:140``). signet owns the MANIFEST that produced it,
the gates, and the ledger. Three reasons the line falls there rather than one step earlier:

  1. The cache directory IS the extraction's identity (above). It is not a scratch location signet
     can route around.
  2. A signet-side Wan encoder would be a custom re-implementation of a canonical one — the
     failure this project names by hand elsewhere ("NOT a custom encoder").
  3. It would be thrown away: musubi caches on every run regardless, and a signet pre-encode writes
     a ``.precomputed/`` layout no musubi stage reads.

What that costs is signet's cheapest checkpoint — the separately-metered encode with its own cost
line. The mitigation is that these cache directories stay SIGNET's Volume paths, declared in the
manifest and rendered here, so the artifact remains inspectable and reusable across rounds; and
that the rendered file commits beside the adapter, which turns "only the dataset changed between
rounds" from a memory into a ``diff`` of two committed artifacts.

⚠ DO NOT pass ``--keep_cache`` to the runner. musubi deletes cache files not present in the
current dataset spec, which is CORRECT when the spec changes between rounds — and is only safe
because every source has a unique cache directory. The collision refusal below is therefore
MONEY-SAFETY, not tidiness: a shared directory turns that default delete into mutual destruction
between two datasets in the same run.

PURE: stdlib only (``tomllib`` for the parse direction). No ``modal``, no ``torch``, no filesystem,
no network. Deterministic by construction — fixed module-level key-order tuples, iteration over the
caller's declared source ORDER only, no dict iteration over user input, no timestamps, no CWD or
environment reads, and no third-party TOML writer (whose key order is not signet's contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from signet_trainer.config.sources import (
    ExtractionMode,
    SourceSpec,
)

# --------------------------------------------------------------------------------------------------
# The upstream key vocabulary
# --------------------------------------------------------------------------------------------------

#: Every key musubi's dataset schema accepts inside a ``[[datasets]]`` block.
#:
#: ⚠ PROVENANCE: this is a TRANSCRIPTION of upstream's ``docs/dataset_config.md``, not a value read
#: out of musubi at run time (this pass performs zero downloads). It exists so a renderer that
#: starts emitting a key upstream would REJECT dies in CI rather than inside a metered container
#: after the weights are resident. Re-check it against the doc when the pin moves.
MUSUBI_DATASET_KEYS: frozenset[str] = frozenset(
    {
        "image_directory",
        "image_jsonl_file",
        "video_directory",
        "video_jsonl_file",
        "cache_directory",
        "control_directory",
        "caption_extension",
        "resolution",
        "batch_size",
        "num_repeats",
        "enable_bucket",
        "bucket_no_upscale",
        "target_frames",
        "frame_extraction",
        "frame_stride",
        "frame_sample",
        "max_frames",
        "source_fps",
        "multiple_target",
    }
)

#: musubi floors each resolution edge to a multiple of this before bucketing (the Wan step). A
#: declared ``[640, 360]`` therefore TRAINS at 640x352.
MUSUBI_RESOLUTION_STEP = 16

# Fixed emission order. Determinism lives here: the renderer walks these tuples, never a dict built
# from user input. Order is not semantically meaningful to TOML — it is meaningful to an operator
# diffing this round's dataset against last round's, which is the artifact's whole purpose.
_GENERAL_KEY_ORDER: tuple[str, ...] = (
    "caption_extension",
    "enable_bucket",
    "bucket_no_upscale",
)
_IMAGE_KEY_ORDER: tuple[str, ...] = (
    "image_directory",
    "cache_directory",
    "caption_extension",
    "resolution",
    "batch_size",
    "num_repeats",
)
_VIDEO_KEY_ORDER: tuple[str, ...] = (
    "video_directory",
    "cache_directory",
    "caption_extension",
    "resolution",
    "batch_size",
    "num_repeats",
    "frame_extraction",
    "target_frames",
    "frame_sample",
    "frame_stride",
    "max_frames",
)

#: Which optional keys each mode actually READS. The renderer emits only these, so a rendered block
#: can never carry a key its own mode ignores — the same rule ``SourceSpec`` enforces at load, held
#: a second time at the point of emission where it is cheap.
_MODE_KEYS: dict[ExtractionMode, tuple[str, ...]] = {
    ExtractionMode.HEAD: ("target_frames",),
    ExtractionMode.CHUNK: ("target_frames",),
    ExtractionMode.SLIDE: ("target_frames", "frame_stride"),
    ExtractionMode.UNIFORM: ("target_frames", "frame_sample"),
    ExtractionMode.FULL: ("max_frames",),
}


# --------------------------------------------------------------------------------------------------
# $0 checks the gate layer hoists out of the renderer
# --------------------------------------------------------------------------------------------------


def check_cache_collisions(sources: Sequence[SourceSpec], *, data_root: str) -> list[str]:
    """Return one message per pair of sources resolving to the SAME cache directory (empty = clean).

    Exposed as a pure list-returning check, not only as the renderer's exception, so the config-gap
    layer can report it BESIDE the other gaps at $0 rather than as a traceback out of a renderer
    the operator did not know was running. musubi raises on duplicate cache dirs anyway; sharing
    one would cross-claim, overwrite, and — with musubi's default cache pruning — mutually delete.
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


def musubi_resolution_warnings(sources: Sequence[SourceSpec]) -> list[str]:
    """Return one warning per source whose declared resolution is not what will be TRAINED.

    A WARNING, deliberately, where every neighbouring silent-repair check is a refusal: musubi
    floors each edge to a multiple of 16, so Timothy's own working ``[640, 360]`` trains at
    640x352. Refusing it would refuse the method's own reference config. The line prints the real
    number so the operator can accept the crop or declare the floored value.
    """
    warnings: list[str] = []
    for source in sources:
        w, h = source.resolution
        fw = w - w % MUSUBI_RESOLUTION_STEP
        fh = h - h % MUSUBI_RESOLUTION_STEP
        if (fw, fh) != (w, h):
            warnings.append(
                f"source {source.id!r}: resolution [{w}, {h}] is not a multiple of "
                f"{MUSUBI_RESOLUTION_STEP} - musubi will train it at {fw}x{fh}. Accept the crop, "
                f"or declare [{fw}, {fh}]."
            )
    return warnings


# --------------------------------------------------------------------------------------------------
# Render: signet source list -> musubi TOML
# --------------------------------------------------------------------------------------------------


def _toml_string(value: str) -> str:
    """Emit a TOML basic string, refusing anything needing an escape we do not test.

    A backslash or control character in one of these values is a bug (they are Linux container
    paths and a caption suffix), and emitting a hand-rolled escape sequence that no test covers is
    how a renderer starts producing a file that parses to something other than what it says.
    """
    if "\\" in value or any(ord(c) < 0x20 for c in value) or '"' in value:
        raise ValueError(
            f"cannot render {value!r} as a TOML string: it contains a backslash, a quote, or a "
            f"control character. These values are container paths and a caption suffix; none of "
            f"them legitimately needs an escape."
        )
    return f'"{value}"'


def _toml_value(value: object) -> str:
    """Render one scalar/array TOML value. Bools BEFORE ints — ``bool`` is a subclass of ``int``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"unrenderable TOML value {value!r} of type {type(value).__name__}")


def _dataset_block(source: SourceSpec, *, data_root: str) -> dict[str, object]:
    """Build the key->value mapping for one ``[[datasets]]`` block (emission order applied later)."""
    directory_key = "image_directory" if source.kind == "image" else "video_directory"
    block: dict[str, object] = {
        directory_key: source.directory,
        "cache_directory": source.resolve_cache_root(data_root),
        "resolution": list(source.resolution),
        "batch_size": source.batch_size,
        "num_repeats": source.num_repeats,
    }
    if source.caption_extension is not None:
        block["caption_extension"] = source.caption_extension
    if source.extraction is not ExtractionMode.IMAGE:
        block["frame_extraction"] = source.extraction.value
        for key in _MODE_KEYS[source.extraction]:
            block[key] = getattr(source, key)
    return block


def render_musubi_toml(
    sources: Sequence[SourceSpec],
    *,
    data_root: str,
    caption_extension: str,
    enable_bucket: bool = True,
    bucket_no_upscale: bool = True,
) -> str:
    """Render a signet source list as a musubi-tuner dataset TOML. Deterministic; pure.

    ``data_root`` resolves the DERIVED cache root of any source that did not name one
    (``<data_root>/cache/<id>``); a source with an explicit ``cache_root`` never reads it.

    Raises ``ValueError`` before any string is built when the list is empty or two sources resolve
    to one cache directory — the collision is money-safety, not tidiness (see the module docstring
    on ``--keep_cache``).
    """
    if not sources:
        raise ValueError(
            "render_musubi_toml: no sources. musubi needs at least one [[datasets]] block; a run "
            "with none would cache nothing and train on nothing."
        )
    collisions = check_cache_collisions(sources, data_root=data_root)
    if collisions:
        raise ValueError("; ".join(collisions))

    lines: list[str] = ["[general]"]
    general: dict[str, object] = {
        "caption_extension": caption_extension,
        "enable_bucket": enable_bucket,
        "bucket_no_upscale": bucket_no_upscale,
    }
    for key in _GENERAL_KEY_ORDER:
        lines.append(f"{key} = {_toml_value(general[key])}")

    for source in sources:
        block = _dataset_block(source, data_root=data_root)
        order = _IMAGE_KEY_ORDER if source.kind == "image" else _VIDEO_KEY_ORDER
        unknown = set(block) - set(order)
        if unknown:
            # Unreachable via the public surface; it fires only if a key is added to _dataset_block
            # and NOT to the emission order, which would otherwise DROP it silently.
            raise ValueError(
                f"source {source.id!r}: keys {sorted(unknown)} have no emission order - add them "
                f"to _IMAGE_KEY_ORDER/_VIDEO_KEY_ORDER rather than letting them be dropped."
            )
        lines.append("")
        lines.append("[[datasets]]")
        for key in order:
            if key in block:
                lines.append(f"{key} = {_toml_value(block[key])}")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------------
# Parse: musubi TOML -> signet source list (the other direction of the round trip)
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedMusubiConfig:
    """A musubi dataset TOML read back into signet's manifest vocabulary.

    ⚠ ``id`` DOES NOT ROUND-TRIP. musubi has no concept of a source id, so the ids assigned here
    are PROVENANCE-ONLY, derived from each block's cache-directory basename. Feeding a parsed list
    back to :func:`render_musubi_toml` reproduces the input file byte-for-byte (the ids never reach
    the TOML), but it does not reconstruct the signet config that produced it.
    """

    caption_extension: str
    enable_bucket: bool
    bucket_no_upscale: bool
    sources: tuple[SourceSpec, ...]


def _provenance_id(cache_directory: str, index: int, taken: set[str]) -> str:
    """Mint a stable, unique id from a cache directory basename (``.../cache_2`` -> ``cache_2``)."""
    base = cache_directory.rstrip("/").rsplit("/", 1)[-1] or f"ds{index + 1}"
    candidate = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)
    if not candidate or not (candidate[0].isalnum()):
        candidate = f"ds{index + 1}"
    if candidate in taken:
        candidate = f"{candidate}_{index + 1}"
    return candidate


def parse_musubi_toml(text: str) -> ParsedMusubiConfig:
    """Read a musubi dataset TOML back into ``SourceSpec``s (the reverse of the renderer).

    This is what makes the equivalence claim two-directional: an operator's existing, known-good
    runner config can be read INTO the manifest vocabulary and re-emitted, and the assertion that
    the bytes are unchanged proves the vocabulary lost nothing on the way through.

    Every ``SourceSpec`` is constructed through the normal validators, so a hand-written TOML
    carrying a silent-repair arrangement (a 30-frame target, ``uniform`` with ``frame_sample = 1``)
    is refused here with the same message it would get at config load.
    """
    # Function-local, and this is the repo's Anti-Pattern-6 idiom rather than a style choice.
    # ``tomllib`` is stdlib only from 3.11. This module's JOB is to WRITE TOML — the reader is
    # needed by exactly one function, the parse direction used for round-trip verification. A
    # module-level import made the whole renderer unimportable on the 3.10 bench interpreter that
    # every regression baseline in this project was measured on, taking two test files down at
    # COLLECTION time. A reader needed by one function belongs in that function.
    try:
        import tomllib  # noqa: PLC0415 - deliberate: see above
    except ModuleNotFoundError:  # pragma: no cover - 3.10 bench env; tomli is its backport
        import tomli as tomllib  # noqa: PLC0415

    parsed = tomllib.loads(text)
    unknown_tables = set(parsed) - {"general", "datasets"}
    if unknown_tables:
        raise ValueError(f"unsupported top-level table(s) {sorted(unknown_tables)} in musubi TOML")
    general = parsed.get("general", {})
    blocks = parsed.get("datasets", [])
    if not blocks:
        raise ValueError("musubi TOML declares no [[datasets]] blocks")

    sources: list[SourceSpec] = []
    taken: set[str] = set()
    for index, block in enumerate(blocks):
        unknown = set(block) - MUSUBI_DATASET_KEYS
        if unknown:
            raise ValueError(
                f"[[datasets]] block {index + 1} carries key(s) {sorted(unknown)} that are not in "
                f"the transcribed musubi vocabulary"
            )
        unsupported = set(block) & {
            "image_jsonl_file",
            "video_jsonl_file",
            "control_directory",
            "multiple_target",
            "source_fps",
        }
        if unsupported:
            raise NotImplementedError(
                f"[[datasets]] block {index + 1} uses {sorted(unsupported)}, which signet's source "
                f"vocabulary does not carry yet. "
                f"signet_trainer.runners.musubi_toml.render_control_directories lands them."
            )
        cache_directory = block["cache_directory"]
        source_id = _provenance_id(cache_directory, index, taken)
        taken.add(source_id)
        kind = "image" if "image_directory" in block else "video"
        extraction = ExtractionMode(block.get("frame_extraction", ExtractionMode.IMAGE.value))
        fields: dict[str, object] = {
            "id": source_id,
            "kind": kind,
            "directory": block["image_directory"] if kind == "image" else block["video_directory"],
            "cache_root": cache_directory,
            "resolution": tuple(block["resolution"]),
            "extraction": extraction,
            "batch_size": block.get("batch_size", 1),
            "num_repeats": block.get("num_repeats", 1),
        }
        if "caption_extension" in block:
            fields["caption_extension"] = block["caption_extension"]
        for key in _MODE_KEYS.get(extraction, ()):
            if key in block:
                fields[key] = block[key]
        sources.append(SourceSpec.model_validate(fields))

    return ParsedMusubiConfig(
        caption_extension=general.get("caption_extension", ".txt"),
        enable_bucket=general.get("enable_bucket", True),
        bucket_no_upscale=general.get("bucket_no_upscale", False),
        sources=tuple(sources),
    )


# --------------------------------------------------------------------------------------------------
# Declared gaps — named symbols, actionable messages, never a silent no-op
# --------------------------------------------------------------------------------------------------


def render_from_config(cfg: object) -> str:
    """Render the TOML straight from a ``SignetConfig``. NOT LANDED — the schema field is missing.

    ``DataConfig`` has no ``sources`` field on this tree (``config/schema.py:125-183``), so there is
    nothing to read. What lands this, in one additive edit:

        ``sources: list[SourceSpec] | None = None`` on ``DataConfig``, importing ``SourceSpec``
        from ``signet_trainer.config.sources`` rather than redeclaring it, with ``None`` kept as
        the default so the ABSENCE of the key stays the byte-identical path for all 18 shipped
        configs; plus ``"wan"`` in ``ModelConfig.family`` (``schema.py:244``).

    This function then becomes::

        return render_musubi_toml(
            cfg.data.sources,
            data_root=cfg.data.preprocessed_data_root,
            caption_extension=cfg.data.caption_extension,
        )

    Until then, callers build the source list explicitly and call :func:`render_musubi_toml`.
    """
    raise NotImplementedError(
        "render_from_config: DataConfig has no `sources` field yet (config/schema.py:125-183). "
        "Land `sources: list[SourceSpec] | None = None` on DataConfig (importing SourceSpec from "
        "signet_trainer.config.sources) and `wan` in ModelConfig.family (schema.py:244); until "
        "then call render_musubi_toml(sources, data_root=..., caption_extension=...) directly."
    )


def render_control_directories(sources: Sequence[SourceSpec]) -> str:
    """Emit the upstream keys this pass does not carry. NOT LANDED — no signet vocabulary for them.

    ``control_directory``, ``image_jsonl_file``, ``video_jsonl_file``, ``multiple_target`` and
    ``source_fps`` are REAL musubi keys (they are in :data:`MUSUBI_DATASET_KEYS`), and each needs a
    corresponding ``SourceSpec`` field plus its own coherence rules before it can be rendered — a
    jsonl source and a directory source are mutually exclusive, and ``control_directory`` is a
    paired corpus, not a geometry knob. Landing them is a schema change, not a renderer change.
    """
    raise NotImplementedError(
        "render_control_directories: control_directory / image_jsonl_file / video_jsonl_file / "
        "multiple_target / source_fps are real musubi keys with no SourceSpec field to carry them. "
        f"Add the fields (and their exclusivity rules) to signet_trainer.config.sources.SourceSpec "
        f"first; {len(sources)} source(s) were offered."
    )
