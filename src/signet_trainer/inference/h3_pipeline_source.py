"""Where an H3 render's components come from — the MOUNTED VOLUME, never the Hub (D-10-DEF-14).

⛔ THE DEFECT THIS MODULE EXISTS FOR, and why the obvious fix was rejected.

``h3_sample`` used to build its pipeline with::

    ModularPipeline.from_pretrained(WEIGHTS_DIR / config.model.model_id, workflow="ref2va", ...)

``model.model_id`` is ``minimax-h3/transformer_ref`` — the TRANSFORMER PARTITION. That is the
correct meaning for ``h3_train`` and ``h3_loader`` (both proven across five dispatches) and the
WRONG one for a pipeline ROOT, which is one level up. One config field, two incompatible meanings.

**Re-pointing the field at the root was evaluated and REJECTED**, on the index read off the Volume
rather than on taste. ``minimax-h3/model_index.json`` records, for EVERY component::

    "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3"

so ``load_components()`` would resolve each one from the **HUB**, not from the Volume it is
standing on — ~134 GiB of egress inside a metered A100 container, and the ONLY symptom is the
bill. It also declares a ``transformer`` partition this Volume does not hold (we hold
``transformer_ref``, which is the partition ``ref2va`` runs against anyway).

So the location of every component is decided HERE, from the local root, and the recorded Hub id is
used for exactly one thing: naming, in a refusal, the egress that was refused.

⚠ A THIRD trap, found while fixing the first two and the reason this module reads the index rather
than trusting ``from_pretrained``: the file on the Volume is NAMED ``model_index.json`` but carries
the MODULAR shape (three-element values whose third element is a loading spec, plus
``_blocks_class_name``). ``ModularPipeline.from_pretrained`` looks for ``modular_model_index.json``
first, falls back to ``DiffusionPipeline.load_config`` for ``model_index.json``, and that fallback
branch only understands **two**-element ``(library, class_name)`` values. Every three-element entry
is therefore SKIPPED — no spec gets a source, ``load_components`` logs
"Skipping component: no pretrained model path specified" at INFO, and the pipeline ends up with
every component ``None``. Silent, at a perfectly valid shape, inside a paid container.

Stdlib only, on purpose: ``tests/`` must be able to drive this without importing ``modal.fns``
(which would drag ``modal`` into ``sys.modules`` and break the dry-run gate's Anti-Pattern-6
assertion for the whole session) and without diffusers or torch.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath

# The component name the ``ref2va`` workflow's denoise loop reads, transcribed from the pinned
# diffusers source (``modular_pipelines/minimax_h3/denoise.py``::``MiniMaxH3Ref2VALoopDenoiser``,
# which is ``MiniMaxH3LoopDenoiser(transformer_name="transformer_ref")``). ONE repository holds both
# checkpoint partitions under DIFFERENT COMPONENT NAMES — ``transformer`` for ``t2va``/``fl2va`` and
# ``transformer_ref`` for ``ref2va`` — so ``pipe.transformer`` on a ref2va pipeline is not "the
# transformer under another name", it is a component the workflow never declares and never loads.
H3_REF2VA_TRANSFORMER = "transformer_ref"

# Read in order. The modular name is the one ``ModularPipeline`` itself looks for; the plain name is
# what this checkpoint actually ships (see the module docstring's third trap).
H3_PIPELINE_INDEX_NAMES = ("modular_model_index.json", "model_index.json")

# Keys in the index that are pipeline metadata rather than components.
_INDEX_METADATA_PREFIX = "_"

# ── D-10-DEF-15 — the RENDER frame-count band, which is NOT the training frame law ───────────────
#
# Transcribed from the pinned diffusers source, never restated from a docs page:
#   ``minimax_h3/modular_pipeline.py``  ``min_duration`` = 5.0 s, ``max_duration`` = 15.0 s,
#                                       ``fps`` = ``MINIMAX_H3_FPS`` = 24
#   ``minimax_h3/before_encoder.py``    ``MiniMaxH3Ref2VASetupStep`` raises unless
#                                       ``min_duration <= align_num_frames(n)/fps <= max_duration``
#
# ⛔ THIS IS A DIFFERENT LAW FROM ``17n + 5``, and satisfying one says nothing about the other. 22
# frames is a perfectly legal ``17n + 5`` TRAINING bucket — it is the bucket this campaign's cache
# was encoded at — and MiniMax-H3 will not GENERATE it. The trained length and the renderable
# lengths are disjoint here, which is a fact about the checkpoint, not a config mistake.
#
# It fires inside ``MiniMaxH3Ref2VASetupStep``, i.e. after BOTH 61.7 GiB loads and the adapter
# injection, so discovering it in-container costs the whole warm-up for zero clips.
H3_MIN_DURATION_S = 5.0
H3_MAX_DURATION_S = 15.0
H3_FPS = 24
H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5


def h3_aligned_num_frames(num_frames: int) -> int:
    """Snap a frame count up to the next ``17n + 5`` the video VAE can encode.

    Transcribed from ``minimax_h3/modular_pipeline.align_num_frames``. The pipeline checks the
    duration of the ALIGNED count, not the requested one, so this has to be applied before the band.
    """
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
    aligned = int(num_frames)
    while aligned % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        aligned += 1
    return aligned


def h3_renderable_frame_bounds() -> tuple[int, int]:
    """The inclusive ``num_frames`` band MiniMax-H3 will generate: (120, 360) at the pin."""
    return int(H3_MIN_DURATION_S * H3_FPS), int(H3_MAX_DURATION_S * H3_FPS)


def assert_h3_frame_count_is_renderable(num_frames: int, *, where: str) -> str:
    """Refuse an unrenderable ``validation.frame_count`` BEFORE anything is loaded.

    ``where`` names the config field so the message is actionable. Returns a one-line confirmation
    when the count is legal, so the log records the band that was checked rather than its silence.
    """
    low, high = h3_renderable_frame_bounds()
    aligned = h3_aligned_num_frames(num_frames)
    if not low <= aligned <= high:
        legal = [n for n in range(low, high + 1) if n % H3_FRAMES_PER_CHUNK == H3_LATENTS_PER_CHUNK]
        raise RuntimeError(
            f"[h3] {where}={num_frames} rounds up to {aligned} frames = "
            f"{aligned / H3_FPS:.3f} s, outside the {H3_MIN_DURATION_S}-{H3_MAX_DURATION_S} s "
            f"MiniMax-H3 generates at {H3_FPS} fps (num_frames {low}-{high}). The pipeline refuses "
            "this inside MiniMaxH3Ref2VASetupStep — i.e. AFTER both 61.7 GiB loads and the adapter "
            "injection — so it is refused here instead, before any of that is paid for. "
            f"Renderable counts: {legal}. ⚠ This is NOT the 17n+5 training law: a legal TRAINING "
            "bucket (this campaign trained at 22) can be unrenderable, and choosing a render length "
            "is an eval-design decision, not a mechanical bump."
        )
    return (
        f"[h3] {where}={num_frames} -> {aligned} aligned frames = {aligned / H3_FPS:.3f} s, "
        f"inside MiniMax-H3's {H3_MIN_DURATION_S}-{H3_MAX_DURATION_S} s generation band."
    )


@dataclass(frozen=True)
class H3ComponentSource:
    """One component's LOCAL source: the directory on the weights Volume that holds its weights.

    Frozen so a resolved source cannot be re-pointed after it has been asserted local — the same
    reason ``H3PreparedReference`` is frozen (D-10-DEF-4).
    """

    name: str
    subfolder: str
    local_dir: Path
    # The Hub id the index recorded for this component. Carried ONLY so a refusal can name the
    # egress it refused; nothing ever loads from it.
    declared_hub_id: str | None = None


def read_h3_pipeline_index(root: Path | str) -> tuple[dict, Path]:
    """Read the pipeline index out of the mounted pipeline ROOT.

    Returns the parsed index and the path it came from. Raises — never returns a default — because
    every failure here is cheap and every failure downstream of here is a paid container.
    """
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError(
            f"[h3] the H3 pipeline root {root} is not a directory on the mounted weights Volume. "
            "This is the ROOT that holds the component partitions (transformer_ref/, text_encoder/, "
            "vae/, audio_vae/, scheduler/, audio_scheduler/, processor/, tokenizer/), NOT the "
            "transformer partition itself — `model.model_id` names the partition and is a different "
            "field with a different meaning (D-10-DEF-14)."
        )
    for name in H3_PIPELINE_INDEX_NAMES:
        candidate = root / name
        if candidate.is_file():
            try:
                index = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"[h3] {candidate} is not valid JSON ({exc}). The pipeline index decides which "
                    "class every component is loaded as; a damaged one cannot be guessed around."
                ) from exc
            if not isinstance(index, dict):
                raise RuntimeError(f"[h3] {candidate} does not hold a JSON object.")
            return index, candidate
    raise RuntimeError(
        f"[h3] neither {' nor '.join(H3_PIPELINE_INDEX_NAMES)} exists under the H3 pipeline root "
        f"{root}. Contents: {sorted(p.name for p in root.iterdir())[:20]}."
    )


def resolve_h3_component_sources(
    index: Mapping[str, object],
    root: Path | str,
    needed: Iterable[str],
) -> dict[str, H3ComponentSource]:
    """Map each NEEDED component name to its local directory under ``root``.

    ``needed`` is the workflow's own component list (``pipe.pretrained_component_names`` on a
    pipeline built from the ``ref2va`` blocks) — never a list restated here, because a restated one
    drifts from the blocks the moment diffusers moves.

    The index supplies the SUBFOLDER (transcribed, not assumed: a component's directory name is a
    checkpoint-layout fact). ``root`` supplies the location. The recorded Hub id supplies nothing.
    """
    root = Path(root)
    sources: dict[str, H3ComponentSource] = {}
    available = sorted(k for k in index if not str(k).startswith(_INDEX_METADATA_PREFIX))
    for name in needed:
        if name not in index:
            raise RuntimeError(
                f"[h3] the pipeline index under {root} declares no component named {name!r}, which "
                f"the ref2va workflow requires. It declares: {available}. A component the index "
                "does not name cannot be located, and guessing a directory would load whatever "
                "happened to be there."
            )
        entry = index[name]
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise RuntimeError(
                f"[h3] the index entry for {name!r} is {entry!r}, not the three-element "
                "(library, class_name, loading-spec) shape this checkpoint's index uses. A "
                "TWO-element entry means a plain (non-modular) model_index.json, which diffusers' "
                "own fallback would parse into specs pointing at the pipeline root — silently "
                "wrong for a repo whose partitions live in subfolders."
            )
        spec = entry[2]
        if not isinstance(spec, Mapping):
            raise RuntimeError(f"[h3] the loading spec for {name!r} is {spec!r}, not a mapping.")
        subfolder = spec.get("subfolder")
        if not subfolder or not isinstance(subfolder, str):
            raise RuntimeError(
                f"[h3] the index records subfolder={subfolder!r} for {name!r}. Every H3 component "
                "lives in its own partition directory under the pipeline root; an empty subfolder "
                "would resolve the component to the ROOT and load the wrong weights at a valid "
                "shape."
            )
        if PurePath(subfolder).is_absolute() or ".." in PurePath(subfolder).parts:
            raise RuntimeError(
                f"[h3] the index records subfolder={subfolder!r} for {name!r}, which escapes the "
                "pipeline root. Refusing to resolve a component outside the mounted root."
            )
        declared = spec.get("pretrained_model_name_or_path")
        sources[name] = H3ComponentSource(
            name=name,
            subfolder=subfolder,
            local_dir=root / subfolder,
            declared_hub_id=declared if isinstance(declared, str) else None,
        )
    return sources


def assert_h3_sources_are_local(
    sources: Mapping[str, H3ComponentSource],
    root: Path | str,
) -> str:
    """THE ZERO-HUB-EGRESS GUARD, before a single byte is loaded.

    Every component must resolve to an existing directory INSIDE the mounted pipeline root. A
    component that does not is refused HERE, naming the Hub id the index would otherwise have
    resolved it from — because the failure mode being guarded is not an exception, it is ~134 GiB of
    egress and a bill, with no exception anywhere.
    """
    root = Path(root)
    if not sources:
        # A vacuous guard is worse than none: `all(...)` over an empty mapping is True, so an empty
        # component set would "pass" this assertion and then render with nothing loaded.
        raise RuntimeError(
            "[h3] no components were resolved for the ref2va workflow. An empty component set "
            "passes every check below vacuously and renders with nothing loaded."
        )
    for name, source in sorted(sources.items()):
        local = source.local_dir
        if not local.is_absolute():
            raise RuntimeError(
                f"[h3] component {name!r} resolved to the relative path {local} — a relative source "
                "is resolved against the container's cwd, not the Volume mount."
            )
        try:
            local.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"[h3] component {name!r} resolved to {local}, which is outside the mounted "
                f"pipeline root {root}."
            ) from exc
        if not local.is_dir():
            raise RuntimeError(
                f"[h3] component {name!r} is not on the weights Volume: {local} does not exist. "
                f"The index would resolve it from the HUB instead "
                f"({source.declared_hub_id!r}, subfolder {source.subfolder!r}) — refusing, because "
                "a Hub fallback here is ~134 GiB of egress inside a metered A100 container and its "
                "only symptom is the bill. Download the partition into the weights Volume, or fix "
                "`model.pipeline_root_id`."
            )
    return (
        f"[h3] {len(sources)} component(s) resolved LOCALLY under {root}: "
        + ", ".join(f"{n}->{s.subfolder}" for n, s in sorted(sources.items()))
        + " (zero Hub egress — asserted, not assumed)."
    )


def assert_h3_components_loaded_locally(
    loaded_from: Mapping[str, object],
    root: Path | str,
) -> str:
    """The SECOND half of the guard: what the loaded components say they came from.

    ``loaded_from`` maps component name -> the ``pretrained_model_name_or_path`` recorded on the
    pipeline's spec AFTER loading (``ComponentSpec.load`` writes it, and ``update_components``
    re-derives the spec from the loaded object's ``_diffusers_load_id``). The pre-load guard proves
    what we ASKED for; this proves what the library actually DID — which is the half that catches a
    spec silently re-pointed by any code between the two.
    """
    root = Path(root)
    if not loaded_from:
        raise RuntimeError("[h3] no loaded components to verify — see assert_h3_sources_are_local.")
    for name, declared in sorted(loaded_from.items()):
        if declared is None:
            raise RuntimeError(
                f"[h3] component {name!r} is registered with NO source path. On the ref2va path "
                "that means it was never loaded — diffusers' `load_components` swallows a failed "
                "load into a warning and leaves the attribute None, which then crashes deep inside "
                "the denoise loop after the load has been paid for."
            )
        if not isinstance(declared, (str, Path)):
            raise RuntimeError(f"[h3] component {name!r} records a non-path source {declared!r}.")
        declared_path = Path(str(declared))
        if declared_path != root:
            raise RuntimeError(
                f"[h3] component {name!r} was loaded from {declared!r}, not from the mounted "
                f"pipeline root {root}. A Hub repo id here ('org/name') is the ~134 GiB egress "
                "this guard exists to make loud (D-10-DEF-14)."
            )
    return (
        f"[h3] all {len(loaded_from)} loaded component(s) report {root} as their source — "
        "no component resolved from the Hub."
    )
