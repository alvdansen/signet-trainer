"""data.cache_provenance — PROVENANCE.json for the LTX-2.5 precomputed cache (issue #53 Stage 1).

Closes the risk LTX25_UPSTREAM_DIFF.md §D names: ``load_embeddings_processor``'s new
``gemma_model_path`` argument means the feature-extractor/connector that produces
``video_prompt_embeds`` is dimensioned by the SPECIFIC Gemma root paired at encode time — a
Gemma-3-encoded ``conditions/*.pt`` cache cannot feed a Gemma-4-paired checkpoint's connector
(different ``hidden_size`` at minimum), and today nothing stops that mismatch from being
*attempted* (it would fail deep inside the connector's input projection, not at a checked
boundary). This module moves that failure to the earliest possible point: ``ltx25_train``'s
startup, before any GPU memory is committed to a doomed run — and, symmetrically,
``ltx25_preprocess``'s startup, before a mixed-generation cache can ever be produced.

Sibling of ``data/precomputed.py`` (kept SEPARATE — ``PrecomputedDataset`` must stay untouched,
byte-identity guarantee #10, LTX25_STAGE1_DESIGN.md §11), so the read side has one import and the
2.3 read path (``PrecomputedDataset`` itself) never has to know this module exists.

Anti-Pattern 6: stdlib only (``json``, ``pathlib``) — importable with zero heavy deps, on CPU, in
a test, exactly like ``data/precomputed.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: One file per precomputed root — a full ``output_dir`` is produced (or refused, see
#: ``assert_output_dir_ready_for_ltx25_encode`` below) by ONE ``ltx25_preprocess`` call with one
#: fixed ``(generation, gemma_root)`` pair, so root-level granularity is correct, not merely
#: convenient.
PROVENANCE_FILENAME = "PROVENANCE.json"


def _provenance_path(output_dir: str) -> Path:
    return Path(output_dir) / PROVENANCE_FILENAME


def read_provenance(output_dir: str) -> dict[str, Any] | None:
    """Read ``<output_dir>/PROVENANCE.json``. ``None`` if absent or unparseable (treated as
    "no provenance recorded" by every caller — never as a silent pass on a check that matters)."""
    path = _provenance_path(output_dir)
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return parsed if isinstance(parsed, dict) else None


def write_provenance(output_dir: str, *, ltx_generation: str, gemma_root: str) -> None:
    """Stamp ``<output_dir>/PROVENANCE.json`` at encode time.

    ⛔⛔ CALL ORDER (refuter MUST-CHANGE, HIGH — the commit-or-vanish pitfall applied to THIS file):
    call this BEFORE ``dataset_vol.commit()``, never after. ``modal/fns.py``'s own docstring names
    Pitfall 3 "commit-or-vanish": ``dataset_vol.commit()`` is the TERMINAL volume commit, and
    anything written to the Volume mount after the last commit is not durably persisted — a
    ``PROVENANCE.json`` written post-commit vanishes with the container, and
    ``assert_cache_provenance`` then treats the freshly Gemma-4-encoded cache as "missing
    provenance == legacy 2.3" and refuses it: fail-closed (no silent corruption), but the Stage-1
    workflow could never pass its own gate. ``modal/fns.py::ltx25_preprocess`` calls this
    immediately after the canonical encode returns and strictly BEFORE its own
    ``dataset_vol.commit()``.
    """
    path = _provenance_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ltx_generation": ltx_generation, "gemma_root": gemma_root}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def assert_output_dir_ready_for_ltx25_encode(
    output_dir: str, *, gemma_root: str, overwrite: bool
) -> None:
    """Refuse an ``ltx25_preprocess`` encode into a non-empty ``output_dir`` lacking MATCHING
    provenance — the mixed-cache case (refuter MUST-FIX, HIGH).

    ``preprocess()``'s canonical encode is INCREMENTAL by default (``overwrite=False`` skips
    already-encoded samples) and ``ltx25_preprocess`` mirrors it almost verbatim (§5) — so pointed
    at a dir left over from a 2.3/Gemma-3 encode (or a DIFFERENT Gemma root under the same
    generation), the encode would silently SKIP those stale items, and ``write_provenance`` would
    then stamp the WHOLE root ``ltx_generation: '2.5'`` — certifying a MIXED Gemma-3/Gemma-4 cache
    as clean. ``assert_cache_provenance`` would then PASS on exactly the mismatch it exists to
    catch: worse than no gate at all, because it actively vouches for the corrupted cache.

    Passes (encode may proceed) when:
      * ``output_dir`` does not exist, or exists and is empty — nothing to mix with;
      * ``overwrite=True`` — the canonical encoder is told to re-produce EVERY sample rather than
        skip already-encoded ones, so no stale item can hide underneath a freshly-stamped
        provenance file (a full re-encode, not an incremental one);
      * ``output_dir`` already carries MATCHING provenance (``ltx_generation == '2.5'`` AND the
        SAME ``gemma_root``) — an incremental ltx25-on-ltx25 re-run against its own prior output.

    Refuses otherwise: a non-empty dir with no provenance, or provenance recording a different
    generation/Gemma root, and ``overwrite`` was not explicitly set.
    """
    root = Path(output_dir)
    if not root.exists() or not any(root.iterdir()):
        return
    if overwrite:
        return
    recorded = read_provenance(output_dir)
    if (
        recorded is not None
        and recorded.get("ltx_generation") == "2.5"
        and recorded.get("gemma_root") == gemma_root
    ):
        return
    raise ValueError(
        f"{output_dir} is non-empty and its existing content does not carry MATCHING "
        f"PROVENANCE.json (ltx_generation='2.5', gemma_root={gemma_root!r}) — refusing to encode "
        "into it. The canonical encode is INCREMENTAL (it skips already-encoded samples unless "
        "overwrite=True): a dir left over from a 2.3/Gemma-3 encode, or a DIFFERENT Gemma root, "
        "would silently keep its stale conditions/*.pt items untouched while write_provenance "
        "stamped the WHOLE root '2.5' — certifying a mixed Gemma-3/Gemma-4 cache as clean (the "
        "exact silent failure this gate exists to prevent). Point output_dir at a fresh directory, "
        "or pass overwrite=True (cfg.data.preprocess_overwrite) for a full re-encode of every "
        "sample under the new generation/Gemma root."
    )


def assert_cache_provenance(
    output_dir: str, *, configured_generation: str, configured_gemma_root: str
) -> None:
    """Refuse a generation OR Gemma-root mismatch between a precomputed cache and a train run.

    Missing provenance == legacy '2.3' (every cache written before this Stage-1 change) — refused
    ONLY when ``configured_generation == '2.5'``, so every existing 2.3 workflow (whose
    ``.precomputed/`` dirs carry no ``PROVENANCE.json`` and never will, since ``preprocess()`` —
    the 2.3 fn — is not being changed to write one, and ``train()`` — also unmodified — never
    calls this function at all) stays byte-identical.

    Compares BOTH generation AND Gemma root (refuter MUST-FIX, HIGH — a generation-only check
    cannot catch a root swap: re-pointing ``model.text_encoder_id`` between two Gemma-4 variants,
    e.g. the codebase's own ``-it`` vs ``-qat-q4_0`` root-variant precedent, schema.py). The
    feature-extractor/connector that produces ``video_prompt_embeds`` is dimensioned by the
    SPECIFIC Gemma root paired at encode time, not merely by "which generation" — a same-generation
    root swap is exactly as real a mismatch as a generation swap (LTX25_UPSTREAM_DIFF.md §D).
    """
    recorded = read_provenance(output_dir)
    if recorded is None:
        if configured_generation == "2.5":
            raise ValueError(
                f"{output_dir} carries no PROVENANCE.json but this run is configured for "
                f"ltx_generation='2.5': missing provenance is treated as a LEGACY 2.3 cache (every "
                "cache written before this Stage-1 change never had a provenance file), which "
                "cannot feed a 2.5/Gemma-4-paired connector — the feature-extractor/connector that "
                "produces video_prompt_embeds is dimensioned by the specific Gemma root at encode "
                "time (LTX25_UPSTREAM_DIFF.md §D). Re-run ltx25_preprocess against this "
                "output_dir, or point data.preprocessed_data_root at an already-provenanced 2.5 "
                "cache."
            )
        return

    recorded_generation = recorded.get("ltx_generation")
    if recorded_generation != configured_generation:
        raise ValueError(
            f"{output_dir}'s PROVENANCE.json records ltx_generation={recorded_generation!r} but "
            f"this run is configured for ltx_generation={configured_generation!r} — a cache "
            "encoded under one generation cannot feed a train run configured for the other."
        )

    if configured_generation == "2.5":
        recorded_gemma_root = recorded.get("gemma_root")
        if recorded_gemma_root != configured_gemma_root:
            raise ValueError(
                f"{output_dir}'s PROVENANCE.json records gemma_root={recorded_gemma_root!r} but "
                f"this run is configured for model.text_encoder_id={configured_gemma_root!r} — "
                "the cached conditions/*.pt were encoded through a DIFFERENT Gemma root's "
                "feature-extractor/connector (dimensioned by that specific root's hidden_size / "
                "num_hidden_layers, LTX25_UPSTREAM_DIFF.md §D) and cannot feed this run's "
                "connector. Re-run ltx25_preprocess with model.text_encoder_id="
                f"{configured_gemma_root!r}, or point at a cache provenanced for it."
            )
