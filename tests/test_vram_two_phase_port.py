"""06-09 carry-forward VRAM port proof (Plan 06-10 prep): the proven two-phase A100-80GB pattern —
two-phase load + ``CachedPromptEmbeddings`` + EP-on-CPU + ``torch.no_grad()`` — generalized from the
multi_frame branch (06-09) to the single_frame + base-vs-LoRA ``sample`` branches AND the ``train``
loop (free Gemma before the transformer runs heavy compute).

All zero-GPU (source scan + signature introspection). The GPU behaviour is proven by the gated run;
this covers the structural surface so a regression that re-raws any render path is caught on CI.

Comments + docstrings are stripped before every text scan so prose mentions don't false-positive.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"
_SAMPLER = _REPO_ROOT / "src" / "signet_trainer" / "inference" / "sampler.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _function_body(code: str, name: str) -> str:
    """Slice the source of a top-level ``def name(...)`` up to the next top-level ``def``/decorator.

    Used to scope a scan to a single function (e.g. ``sample`` vs ``train``) after comment/docstring
    stripping — the top-level boundary is a newline followed by ``def ``/``@app`` at column 0.
    """
    start = re.search(rf"\ndef\s+{re.escape(name)}\s*\(", code)
    assert start is not None, f"could not find def {name}("
    rest = code[start.end():]
    end = re.search(r"\n(?:def\s|@app)", rest)
    return rest[: end.start()] if end else rest


def test_run_sampler_accepts_cached_embeddings_and_wraps_no_grad() -> None:
    """``run_sampler`` gains a ``cached_embeddings`` param and wraps ``generate`` in ``no_grad``.

    The enabling primitive for the single_frame + base-vs-LoRA two-phase port: with cached
    embeddings the ltx sampler short-circuits Gemma (``_get_prompt_embeddings``), and the no_grad
    wrap stops the raw denoise/decode forwards retaining autograd graphs (06-09 run-7 landmine).
    """
    from signet_trainer.inference.sampler import run_sampler

    params = inspect.signature(run_sampler).parameters
    assert "cached_embeddings" in params, "run_sampler must accept cached_embeddings (two-phase load)"
    assert params["cached_embeddings"].default is None, "cached_embeddings must default to None"

    code = _strip_comments_and_docstrings(_SAMPLER.read_text(encoding="utf-8"))
    # Assign cached embeddings onto the GenerationConfig (mirrors run_multi_condition_sampler).
    assert re.search(r"cfg\.cached_embeddings\s*=\s*cached_embeddings", code), (
        "run_sampler must thread cached_embeddings onto GenerationConfig.cached_embeddings"
    )
    # The generate call must run under torch.no_grad().
    assert "with torch.no_grad():" in code, "run_sampler must wrap generate() in torch.no_grad()"
    ng_idx = code.index("with torch.no_grad():")
    gen_idx = code.index("sampler.generate(")
    assert ng_idx < gen_idx, "sampler.generate() must be inside the torch.no_grad() block"


def test_sample_phase_a_is_generalized_to_all_single_stage_modes() -> None:
    """PHASE A pre-encode is no longer gated on multi_frame only — it runs for every single-stage mode.

    06-09 ran PHASE A only for ``mode == 'multi_frame'``; the 06-10 prep generalizes it so the
    single_frame + base-vs-LoRA branches (which called run_sampler raw) get the same two-phase load.
    The gate is now ``if not two_stage:`` (two_stage keeps its legacy Gemma-loaded flow).
    """
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))

    # The PHASE-A pre-encode gate must be the generalized `if not two_stage:` — NOT the old
    # multi_frame-only guard. (The multi_frame RENDER branch below still dispatches on mode, but the
    # PRE-ENCODE that builds cached_by_prompt must cover all single-stage modes.)
    assert re.search(r"cached_by_prompt:\s*dict\s*\|\s*None\s*=\s*None\s*\n\s*if\s+not\s+two_stage:", code), (
        "PHASE A must be gated on `if not two_stage:` (generalized from multi_frame-only)"
    )
    # two_stage must be read BEFORE PHASE A so the gate can see it.
    two_stage_idx = code.index("two_stage = config.validation.two_stage_upscale")
    phase_a_idx = code.index("if not two_stage:")
    assert two_stage_idx < phase_a_idx, "two_stage must be read before the PHASE-A gate"
    # two_stage must be assigned exactly once (the duplicate in section 3b was removed).
    assert code.count("two_stage = config.validation.two_stage_upscale") == 1, (
        "two_stage must be defined exactly once (hoisted above PHASE A; the 3b duplicate removed)"
    )


def test_single_frame_and_base_vs_lora_pass_cached_embeddings() -> None:
    """Every single-stage render in sample() feeds its PHASE-A cached embeddings.

    The single_frame ref-ON/ref-OFF renders and the base-vs-LoRA ``_render`` dispatch must pass
    ``cached_embeddings=`` / a ``cached`` arg so no render touches the (deleted) Gemma.
    """
    sample_body = _function_body(_strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8")), "sample")

    # Every run_sampler call in sample() carries cached_embeddings=cached (single_frame ON/OFF +
    # the base-vs-LoRA _render dispatch). The train() inert validation-sampler seam is a separate
    # decoder-off path and is intentionally out of this scope.
    run_sampler_calls = re.findall(r"run_sampler\((?:.|\n)*?\)", sample_body)
    assert run_sampler_calls, "sample() must still call run_sampler"
    assert all("cached_embeddings=cached" in c for c in run_sampler_calls), (
        "every run_sampler call in sample() must pass cached_embeddings=cached (two-phase load)"
    )
    # base-vs-LoRA: _render threads the cached arg into run_sampler.
    assert re.search(r"run_sampler\([^)]*cached_embeddings=cached[^)]*\)", sample_body), (
        "_render must forward cached embeddings into run_sampler for the base-vs-LoRA columns"
    )
    # single_frame + base + LoRA loops must each resolve the per-prompt cached lookup (two_stage=None).
    assert sample_body.count("cached_by_prompt[prompt] if cached_by_prompt is not None else None") >= 3, (
        "single_frame + base + LoRA loops must each resolve the per-prompt cached embeddings"
    )


def test_single_frame_branch_fails_fast_on_two_stage() -> None:
    """The single_frame branch must raise on two_stage_upscale (WR-02) — never skip PHASE A silently.

    With two_stage set, PHASE A is skipped (cached_by_prompt is None, with_text_encoder=True), so
    every single_frame run_sampler call would live-encode with Gemma alongside the loaded 22B —
    the proven 77-79 GiB A100-80GB OOM (06-09 runs 3-4). The branch also never routes through
    _render, so the toggle would otherwise be silently ignored. Mirror of the multi_frame branch's
    cached_by_prompt-is-None defensive raise.
    """
    sample_body = _function_body(
        _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8")), "sample"
    )

    sf_match = re.search(r'if\s+config\.conditioning\.mode\s*==\s*["\']single_frame["\']\s*:', sample_body)
    assert sf_match is not None, "sample() must still have the single_frame branch"
    mf_match = re.search(r'if\s+config\.conditioning\.mode\s*==\s*["\']multi_frame["\']\s*:', sample_body)
    assert mf_match is not None, "sample() must still have the multi_frame branch"

    sf_branch = sample_body[sf_match.end(): mf_match.start()]
    guard = re.search(r"if\s+two_stage\s*:\s*\n\s*raise\s+RuntimeError", sf_branch)
    assert guard is not None, (
        "the single_frame branch must fail fast (raise RuntimeError) when two_stage is set — "
        "otherwise PHASE A is skipped and Gemma coexists with the 22B (proven OOM, WR-02)"
    )
    # The guard must sit BEFORE the first render call of the branch.
    first_render = sf_branch.find("run_sampler(")
    assert first_render == -1 or guard.start() < first_render, (
        "the two_stage guard must precede any run_sampler call in the single_frame branch"
    )


def test_train_frees_gemma_after_gate_before_loop() -> None:
    """The train loop drops Gemma after the validation gate, before the heavy transformer compute.

    Two-phase VRAM discipline for training: Gemma is only needed by gate check #2 (a config read);
    training uses precomputed conditions. Freeing it (null attr + del local + gc + empty_cache)
    before the loop keeps it from coexisting with the 22B transformer + activations + optimizer.
    """
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))

    # The free must null the attribute (assign=True storage frees only on last-ref drop, 06-09 run-5).
    assert "components.text_encoder = None" in code, (
        "train() must null components.text_encoder to drop Gemma's last reference (assign=True fix)"
    )
    assert "torch.cuda.empty_cache()" in code, "train() must empty_cache after freeing Gemma"

    # Ordering: the free must sit AFTER run_validation_gate and BEFORE the train_loop dispatch.
    gate_idx = code.index("run_validation_gate(")
    free_idx = code.index("components.text_encoder = None")
    loop_idx = code.index("final_step = train_loop(")
    assert gate_idx < free_idx < loop_idx, (
        "Gemma free must be AFTER run_validation_gate and BEFORE train_loop "
        f"(gate@{gate_idx}, free@{free_idx}, loop@{loop_idx})"
    )


def test_train_dataset_uses_config_preprocessed_root_not_hardcoded() -> None:
    """train() builds PrecomputedDataset from config.data.preprocessed_data_root — not a hardcode.

    06-10 fix: the hardcoded ``DATASET_DIR / '.precomputed'`` would silently train the demo
    multi-frame overfit (preprocessed_data_root=/dataset/.precomputed_demo) on the wrong (fresh)
    latents, or fail if absent. The Phase-3 config carries ``/dataset/.precomputed`` so the change
    is byte-identical there.
    """
    body = _function_body(_strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8")), "train")

    assert "PrecomputedDataset(config.data.preprocessed_data_root)" in body, (
        "train() must build PrecomputedDataset from config.data.preprocessed_data_root"
    )
    assert 'PrecomputedDataset(str(DATASET_DIR / ".precomputed"))' not in body, (
        "the hardcoded .precomputed dataset path must be gone (06-10 demo-routing fix)"
    )
