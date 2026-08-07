"""Structural SC#2 wiring proof (MODL-04) — the preprocess body runs the CANONICAL encoder.

Scans ``modal/fns.py`` as TEXT (does NOT import the ``modal``-decorated module — keeps this test
runnable on Windows/CI with zero Modal spend, mirroring ``test_no_warm_gpu.py``). Asserts that
``preprocess`` is wired to ltx-trainer's canonical ``process_dataset.preprocess_dataset`` video-only,
multi-bucket, ``batch_size=1``, ``device="cuda"``, with the commit-or-vanish ``dataset_vol.commit()``
— and, critically, that it does NOT port the ABANDONED custom encoders (``encode_video_vae`` /
``encode_text``, RESEARCH.md Pitfall 1 — the enochiatron landmine).

Comments + docstrings are stripped first so prose mentions of the anti-landmine names or "cuda"
don't false-positive (same discipline as ``test_no_warm_gpu.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

_FNS = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def test_preprocess_wires_canonical_encoder() -> None:
    """SC#2: preprocess calls the canonical preprocess_dataset video-only, multi-bucket, with commit."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))

    assert "preprocess_dataset" in code, "preprocess must call the canonical preprocess_dataset"
    # a2v (Phase 9): with_audio is now CONFIG-DRIVEN (default False), not a hardcoded literal — the
    # preprocess fn takes a with_audio param (default False, byte-identical for non-a2v) and threads
    # it into preprocess_dataset(with_audio=with_audio).
    assert re.search(r"with_audio\s*:\s*bool\s*=\s*False", code), (
        "with_audio must default to False on the preprocess signature (video-only default)"
    )
    assert "with_audio=with_audio" in code, "with_audio must be threaded into preprocess_dataset"
    assert "batch_size=1" in code, "multi-bucket requires batch_size=1 (Pitfall 4)"
    assert re.search(r'device\s*=\s*["\']cuda["\']', code), "encode runs device=cuda (RESEARCH A2)"
    assert "dataset_vol.commit()" in code, "Pitfall 3 commit-or-vanish on the dataset Volume"


def test_preprocess_uses_full_multibucket_set() -> None:
    """D-05: all three buckets 768x352x{25,49,81} are passed as (F,H,W) tuples from day one."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    for frames in (25, 49, 81):
        assert re.search(rf"\(\s*{frames}\s*,\s*352\s*,\s*768\s*\)", code), (
            f"missing (F,H,W) bucket ({frames}, 352, 768) — D-05 multi-bucket set"
        )


def test_preprocess_is_parameterized_backward_compatible() -> None:
    """Plan 06-08 / REF-02: preprocess exposes metadata_path/resolution_buckets/output_dir params.

    The signature must accept the three overridable params with the Phase-2 ``fresh`` defaults
    (backward-compat), and pass them THROUGH to the canonical ``preprocess_dataset`` — so a demo
    subset can be encoded to a distinct output dir (Plan 06-10) WITHOUT changing the default encode.
    """
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))

    # 1) the three new params exist with the Phase-2 fresh defaults (backward-compatible signature).
    assert re.search(
        r"def preprocess\(\s*metadata_path:\s*str\s*=\s*str\(\s*DATASET_DIR\s*/\s*"
        r'["\']fresh["\']\s*/\s*["\']metadata\.jsonl["\']\s*\)',
        code,
    ), "preprocess must accept metadata_path defaulting to the fresh metadata.jsonl (backward-compat)"
    assert re.search(r"resolution_buckets:\s*list\s*=\s*\[", code), (
        "preprocess must accept resolution_buckets defaulting to the D-05 multi-bucket list"
    )
    assert re.search(
        r'output_dir:\s*str\s*=\s*str\(\s*DATASET_DIR\s*/\s*["\']\.precomputed["\']\s*\)',
        code,
    ), "preprocess must accept output_dir defaulting to .precomputed (backward-compat)"

    # 2) the params are threaded THROUGH to the canonical encoder (not shadowed by the old literals).
    assert re.search(r"dataset_file\s*=\s*metadata_path", code), (
        "preprocess must pass metadata_path -> preprocess_dataset(dataset_file=...)"
    )
    assert re.search(r"resolution_buckets\s*=\s*resolution_buckets", code), (
        "preprocess must pass the resolution_buckets param through to the canonical encoder"
    )
    assert re.search(r"output_dir\s*=\s*output_dir", code), (
        "preprocess must pass the output_dir param through to the canonical encoder"
    )
    assert re.search(r"return\s+output_dir", code), "preprocess must return the (param) output_dir"


def test_no_abandoned_custom_encoders() -> None:
    """Anti-landmine guard (Pitfall 1): NEVER port enochiatron's abandoned custom encoders."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    assert "encode_video_vae" not in code, "must not port the abandoned custom VAE encoder (Pitfall 1)"
    assert "encode_text" not in code, "must not port the abandoned custom text encoder (Pitfall 1)"


def test_download_weights_pulls_dev_gemma_and_gates_two_stage() -> None:
    """D-07-W + D-DEPTH-1 (04-05): dev + Gemma by default; distilled/upscaler GATED, default OFF.

    Phase-2's D-07-W reserved the distilled/upscaler download for Phase 4; 04-05 lands it as a
    default-OFF ``two_stage`` gate (single-stage runs still fetch ONLY dev + Gemma). The two-stage
    filenames now appear in the source, but strictly behind ``if two_stage:`` and a
    ``two_stage: bool = False`` default (never fetched unless a run opts in).
    """
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    assert "Lightricks/LTX-2.3" in code, "download_weights must pull the LTX-2.3 dev checkpoint"
    assert "google/gemma-3-12b-it" in code, "download_weights must pull Gemma 3 12B"
    assert "weights_vol.commit()" in code, "Pitfall 3 commit-or-vanish on the weights Volume"
    # D-DEPTH-1: the two-stage download is GATED behind a default-OFF flag — single-stage runs
    # (two_stage=False) never touch the distilled/upscaler weights.
    assert re.search(r"def download_weights\(\s*two_stage:\s*bool\s*=\s*False", code), (
        "download_weights must gate the two-stage download behind two_stage: bool = False (default OFF)"
    )
    assert re.search(r"if\s+two_stage\s*:", code), (
        "the distilled/upscaler download must sit behind an `if two_stage:` guard (default OFF)"
    )


def test_smoke_asserts_against_expected_constants() -> None:
    """SC#1: load_ltxv_smoke imports + asserts against the EXPECTED_* single-source constants."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    assert "load_ltxv_components" in code, "smoke must call the ported load_ltxv_components"
    assert "EXPECTED_NUM_BLOCKS" in code, "smoke must assert block count vs EXPECTED_NUM_BLOCKS"
    assert "EXPECTED_HIDDEN_DIM" in code, "smoke must assert hidden_dim vs EXPECTED_HIDDEN_DIM"
