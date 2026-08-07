"""Structural wiring proof for ``fns.py``'s IC-LoRA seams (Plan 07-09, REF-03) — zero GPU/Modal.

Three zero-GPU layers (mirroring ``test_multi_frame_sample_wiring.py`` + ``test_preprocess_wiring.py``):

  * MODULE-TOP IMPORT — ``import signet_trainer.modal.fns`` must succeed off-Modal (the module top
    imports ONLY from ``modal.app``; every heavy ltx/torch import in ``preprocess`` / ``train`` /
    ``sample`` is function-local, Anti-Pattern 6). Proves the ic_lora wiring never pulls the ltx
    stack onto Windows/CI.

  * TRAIN + PREPROCESS TEXT SCAN — preprocess threads the two reference kwargs into the CANONICAL
    ``preprocess_dataset`` call; ``train`` builds the 3-source dataset from
    ``strategy.get_data_sources()`` (``reference_latents`` loads), stacks a frozen adapter so the
    ``requires_grad``-filtered optimizer trains only the new adapter (D-7-FREEZE), and runs
    ``assert_paired_reference`` in the ic_lora preflight (before spend).

  * SAMPLE TEXT SCAN — the ``ic_lora`` sample branch dispatches on
    ``config.conditioning.mode == "ic_lora"``, renders the re-skin column via
    ``run_reference_video_sampler`` at ``config.validation.seed`` and the control via the standard
    ``run_sampler`` with no reference, reuses the PHASE-A ``cached_by_prompt`` two-phase load, writes
    ``write_reskin_gallery``, and commits (commit-or-vanish). The multi_frame / single_frame /
    base-vs-LoRA branches are asserted still present (untouched — this plan only ADDS a branch).

Comments + docstrings are stripped before every text scan so prose mentions don't false-positive
(same discipline as ``test_multi_frame_sample_wiring.py``). The GPU render is proven by the
separately-gated 07-11 run — this covers the structural wiring surface ONLY.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _code() -> str:
    return _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))


def test_fns_imports_off_modal() -> None:
    """Module-top import must succeed off-Modal — every heavy import in the stages is function-local."""
    import signet_trainer.modal.fns as fns  # noqa: PLC0415

    assert hasattr(fns, "preprocess"), "modal.fns must expose the preprocess stage function"
    assert hasattr(fns, "train"), "modal.fns must expose the train stage function"
    assert hasattr(fns, "sample"), "modal.fns must expose the sample stage function"


# ── preprocess -------------------------------------------------------------------------------------


def test_preprocess_threads_reference_kwargs() -> None:
    """Task 1: preprocess threads reference_column + reference_downscale_factor into the canonical call."""
    code = _code()

    # Both params exist on the signature (backward-compatible None/1 defaults).
    assert re.search(r"reference_column:\s*str\s*\|\s*None\s*=\s*None", code), (
        "preprocess must accept reference_column: str | None = None (fresh/demo backward-compat)"
    )
    assert re.search(r"reference_downscale_factor:\s*int\s*=\s*1", code), (
        "preprocess must accept reference_downscale_factor: int = 1 (D-7-REF11)"
    )
    # Threaded THROUGH to the canonical preprocess_dataset encoder (NOT a custom encoder).
    assert re.search(r"reference_column\s*=\s*reference_column", code), (
        "preprocess must pass reference_column -> preprocess_dataset(reference_column=...)"
    )
    assert re.search(r"reference_downscale_factor\s*=\s*reference_downscale_factor", code), (
        "preprocess must pass reference_downscale_factor through to the canonical encoder"
    )
    # The canonical path is unchanged — still the ltx-trainer preprocess_dataset, no custom encoder.
    assert "preprocess_dataset" in code, "preprocess must still call the canonical preprocess_dataset"
    assert "encode_video_vae" not in code and "encode_text" not in code, (
        "must not port the abandoned custom encoders (RESEARCH Pitfall 1)"
    )


# ── train ------------------------------------------------------------------------------------------


def test_train_builds_three_source_dataset_from_strategy() -> None:
    """Task 1: train builds the 3-source PrecomputedDataset from strategy.get_data_sources() (Pitfall 3)."""
    code = _code()

    assert re.search(r'config\.conditioning\.mode\s*==\s*["\']ic_lora["\']', code), (
        "train must branch on config.conditioning.mode == 'ic_lora' for the 3-source dataset"
    )
    assert "get_data_sources()" in code, (
        "train must thread the source list the strategy DECLARES (get_data_sources()), not hardcode"
    )
    assert "reference_latents" in code, (
        "the 3-source map must include the reference_latents source (Pitfall 3)"
    )
    assert "ref_latent_conditions" in code, (
        "the reference_latents source must map to the ref_latent_conditions batch key"
    )
    assert re.search(r"PrecomputedDataset\(\s*[^)]*data_sources\s*=", code), (
        "train must pass the 3-source data_sources dict to PrecomputedDataset for ic_lora"
    )


def test_train_frozen_filter_and_optimizer() -> None:
    """Task 1: stack_frozen_adapter + requires_grad-filtered optimizer (D-7-FREEZE / T-07-09-02)."""
    code = _code()

    assert "stack_frozen_adapter" in code, (
        "train must call stack_frozen_adapter when conditioning.frozen_adapter_path is set"
    )
    assert re.search(r"config\.conditioning\.frozen_adapter_path", code), (
        "the frozen-stack must be gated on config.conditioning.frozen_adapter_path"
    )
    assert "requires_grad" in code, (
        "train must document/enforce the requires_grad optimizer filter (frozen params excluded)"
    )
    # The stack must precede build_optimizer so the frozen params (requires_grad=False) are excluded.
    stack_idx = code.index("stack_frozen_adapter(model")
    opt_idx = code.index("build_optimizer(model, config)")
    assert stack_idx < opt_idx, "stack_frozen_adapter must run BEFORE build_optimizer (T-07-09-02)"


def test_train_runs_paired_reference_preflight() -> None:
    """Task 1: assert_paired_reference runs in the ic_lora preflight, before any sustained spend."""
    code = _code()

    assert "assert_paired_reference" in code, (
        "train must call assert_paired_reference in the ic_lora preflight (T-07-09-01)"
    )
    # The preflight must precede the training loop dispatch (no sustained spend on a bad bucket).
    preflight_idx = code.index("assert_paired_reference(")
    loop_idx = code.index("train_loop(")
    assert preflight_idx < loop_idx, "assert_paired_reference must precede train_loop (before spend)"


# ── sample -----------------------------------------------------------------------------------------


def test_sample_has_ic_lora_branch() -> None:
    """Task 2: sample() branches on mode == 'ic_lora' and wires the V2V re-skin sampler + gallery."""
    code = _code()

    # The branch dispatches on the ic_lora mode.
    ic_lora_branches = re.findall(r'config\.conditioning\.mode\s*==\s*["\']ic_lora["\']', code)
    assert ic_lora_branches, "sample()/train() must branch on config.conditioning.mode == 'ic_lora'"

    assert "run_reference_video_sampler" in code, (
        "the ic_lora sample branch must render the re-skin column via run_reference_video_sampler"
    )
    assert "write_reskin_gallery" in code, (
        "the ic_lora sample branch must write the write_reskin_gallery grid (07-04)"
    )
    # Rendered at the config seed (42), never a hardcoded literal (house A/B rule).
    assert re.search(r"seed\s*=\s*config\.validation\.seed", code), (
        "sample() must render at config.validation.seed (42) — no hardcoded seed"
    )
    # The no-reference control column reuses the standard sampler (the SC#3 divergence baseline).
    assert "run_sampler" in code, (
        "the ic_lora branch must render the no-reference control via the standard run_sampler"
    )


def test_ic_lora_branch_reuses_two_phase_cached_embeddings() -> None:
    """Task 2: the ic_lora branch reuses the PHASE-A cached-embeddings two-phase VRAM load (MANDATORY)."""
    code = _code()

    # The branch must read the PHASE-A cache (cached_by_prompt) — Gemma is already deleted, so every
    # render MUST feed cached embeddings (never a live Gemma encode alongside the 22B transformer).
    assert "cached_by_prompt" in code, (
        "the ic_lora branch must reuse the PHASE-A cached_by_prompt two-phase load (06-09 OOM fix)"
    )
    assert re.search(r"cached_embeddings\s*=\s*cached", code), (
        "the ic_lora renders must feed cached_embeddings=cached (two-phase VRAM discipline)"
    )
    # Defensive fail-fast: a two_stage config skips PHASE A -> cached_by_prompt None -> raise.
    assert re.search(
        r'\[sample\]\[ic_lora\][^"]*PHASE-A cached embeddings missing', code
    ), "the ic_lora branch must fail fast when PHASE-A cached embeddings are missing"


def test_ic_lora_branch_loads_trained_adapter() -> None:
    """CR-02: the ic_lora sample branch loads the trained adapter (find_latest + load_lora) onto the
    transformer and renders BOTH col-3 and col-4 through it — never the raw base labeled as adapter."""
    code = _code()

    # The branch resolves the highest-step checkpoint via the same helper the base-vs-LoRA tail uses.
    assert "CheckpointManager" in code, (
        "the ic_lora branch must resolve the trained adapter via CheckpointManager.find_latest()"
    )
    assert "find_latest()" in code, "must resolve the highest-step checkpoint dir, not a flat file"
    assert "load_lora_onto_transformer" in code, (
        "the ic_lora branch must PEFT-wrap the transformer with the trained adapter (CR-02)"
    )
    # Both columns render through the resolved model (reskin_model), NOT components.transformer.
    assert "reskin_model" in code, (
        "the ic_lora renders must run through the trained-adapter model (reskin_model), not raw base"
    )
    # The banner lora_scale reflects reality (the resolved scale var), not a hardcoded 1.0.
    assert "reskin_lora_scale" in code, (
        "the banner lora_scale must reflect whether the adapter actually loaded (CR-02), not 1.0"
    )
    # Fail-loud when no adapter exists (wiring mode) rather than silently rendering the base as adapter.
    assert re.search(r"no trained adapter", code), (
        "the ic_lora branch must warn when no trained adapter is found (renders BASE, wiring mode)"
    )


def test_ic_lora_branch_commits_and_is_scoped() -> None:
    """The branch commits (commit-or-vanish) and the sibling branches are untouched (scoped ADD)."""
    code = _code()

    assert "checkpoints_vol.commit()" in code, "commit-or-vanish on the checkpoints Volume"
    # Sibling branches must still exist — this plan only ADDS a branch, it does not rewrite them.
    assert re.search(r'config\.conditioning\.mode\s*==\s*["\']multi_frame["\']', code), (
        "the Phase-6 multi_frame branch must remain present (scoped change)"
    )
    assert re.search(r'config\.conditioning\.mode\s*==\s*["\']single_frame["\']', code), (
        "the Phase-5 single_frame branch must remain present (scoped change)"
    )
    assert "write_multi_frame_gallery" in code, "the multi_frame gallery path must remain present"
    assert "write_comparison_gallery" in code, "the base-vs-LoRA comparison path must remain present"
