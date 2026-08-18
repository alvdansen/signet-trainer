"""Structural wiring proof for the tier-2 official-adapter IC-LoRA baseline (Plan 07-10, REF-03).

D-7-LADDER tier 2 / D-7-BASELINE: run the OFFICIAL Lightricks Union-Control IC-LoRA through the ported
distilled two-stage ``ICLoraPipeline`` to validate the ported inference SURFACE against a KNOWN-good
adapter BEFORE we train ours. Everything here is proven on CPU (no GPU, no ltx-pipelines installed) —
the metered baseline render is the separately-gated 07-10 Task-2 exercise.

Four zero-GPU layers (mirroring ``test_upscale_toggle.py`` + ``test_ic_lora_wiring.py``):

  1. ``inference/ic_lora_pipeline.py`` exists, exposes ``ICLoraPipeline`` (referenced) +
     ``run_ic_lora_baseline`` + ``read_reference_downscale_factor``, and has NO MODULE-LEVEL
     ``ltx_pipelines`` / ``ltx_core`` / ``torch`` import (Anti-Pattern 6 — the heavy import is
     function-local). The module IMPORTS on CPU without ltx-pipelines installed.

  2. The ``fns.py`` ``ic_lora_baseline`` branch selects ``ICLoraPipeline`` (via
     ``run_ic_lora_baseline``) and loads the OFFICIAL adapter through the comfy-format ``loras=`` path —
     NOT signet's PEFT loader (``load_lora_onto_transformer`` — the Pitfall-4 format-mismatch guard).

  3. The baseline branch READS ``reference_downscale_factor`` from the adapter metadata
     (``read_reference_downscale_factor``) and is gated on the distilled two-stage flag.

  4. ``configs/ltx23_ic_lora_baseline.example.yaml`` names ``Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control``.

Comments + docstrings are stripped before every fns.py text scan so prose mentions don't false-positive
(same discipline as ``test_ic_lora_wiring.py``). The ``sys.modules`` snapshot/restore in the clean-import
test mirrors ``test_upscale_toggle.py`` (popping a shared module without restoring poisons later tests).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"
_PIPELINE_SRC = _REPO_ROOT / "src" / "signet_trainer" / "inference" / "ic_lora_pipeline.py"
_BASELINE_CONFIG = _REPO_ROOT / "configs" / "ltx23_ic_lora_baseline.example.yaml"
_FORBIDDEN_MODULES = ("ltx_pipelines", "ltx_core", "torch")


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _fns_code() -> str:
    return _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------------
# (1) the ported wrapper module exists + is import-confined + CPU-importable
# --------------------------------------------------------------------------------------------------


def test_pipeline_module_exists_and_references_iclorapipeline() -> None:
    assert _PIPELINE_SRC.exists(), "inference/ic_lora_pipeline.py must exist (the ported wrapper)"
    src = _PIPELINE_SRC.read_text(encoding="utf-8")
    assert "ICLoraPipeline" in src, (
        "ic_lora_pipeline.py must wrap ltx_pipelines.ic_lora.ICLoraPipeline (the distilled two-stage path)"
    )
    assert "run_ic_lora_baseline" in src, "the public baseline entrypoint must be run_ic_lora_baseline"
    assert "read_reference_downscale_factor" in src, (
        "the wrapper must expose read_reference_downscale_factor (Pitfall 4b metadata read)"
    )


def test_pipeline_source_has_no_module_level_heavy_import() -> None:
    # Column-0 anchored (``^`` not ``^\s*``): only MODULE-LEVEL imports match — indented
    # function-local + ``if TYPE_CHECKING:`` imports never execute on CPU/CI (Anti-Pattern 6). Mirrors
    # test_upscale_toggle.py's confinement scan.
    code = _strip_comments_and_docstrings(_PIPELINE_SRC.read_text(encoding="utf-8"))
    module_level_heavy_import = re.compile(
        r"^(?:import|from)\s+(?:ltx_pipelines|ltx_core|torch)\b", re.MULTILINE
    )
    hits = module_level_heavy_import.findall(code)
    assert not hits, (
        f"import-confinement violation: module-level heavy import in ic_lora_pipeline.py: {hits} "
        "(the ltx_pipelines/torch imports MUST be function-local)"
    )


def test_pipeline_module_imports_on_cpu_without_ltx_pipelines() -> None:
    saved = {mod: sys.modules.get(mod) for mod in _FORBIDDEN_MODULES}
    try:
        for mod in _FORBIDDEN_MODULES:
            sys.modules.pop(mod, None)

        import signet_trainer.inference.ic_lora_pipeline as icp  # noqa: PLC0415

        for mod in _FORBIDDEN_MODULES:
            assert mod not in sys.modules, (
                f"import-confinement violation: inference.ic_lora_pipeline transitively imported {mod!r}"
            )
        assert hasattr(icp, "run_ic_lora_baseline")
        assert hasattr(icp, "read_reference_downscale_factor")
    finally:
        for mod, prev in saved.items():
            if prev is not None:
                sys.modules[mod] = prev
            else:
                sys.modules.pop(mod, None)


# --------------------------------------------------------------------------------------------------
# (2) the fns.py baseline branch selects ICLoraPipeline + the comfy loader (NOT PEFT)
# --------------------------------------------------------------------------------------------------


def test_fns_has_ic_lora_baseline_branch() -> None:
    code = _fns_code()

    # The branch is gated on ic_lora mode AND the distilled two-stage flag (D-7-BASEVAR discriminator).
    assert re.search(
        r'config\.conditioning\.mode\s*==\s*["\']ic_lora["\']\s*and\s*two_stage', code
    ), "sample() must gate the baseline branch on mode == 'ic_lora' AND two_stage (distilled path)"
    # The branch name is discoverable (acceptance: grep 'ic_lora_baseline' in fns.py).
    assert "ic_lora_baseline" in code, "fns.py must carry the ic_lora_baseline branch"
    # It selects the ported ICLoraPipeline wrapper (via run_ic_lora_baseline), NOT the dev V2V sampler.
    assert "run_ic_lora_baseline" in code, (
        "the baseline branch must render via run_ic_lora_baseline (the distilled two-stage ICLoraPipeline)"
    )
    assert "from signet_trainer.inference.ic_lora_pipeline import" in code, (
        "the baseline branch must import from inference.ic_lora_pipeline (the ported wrapper)"
    )


def test_baseline_uses_comfy_loras_not_peft() -> None:
    """Pitfall 4: the official comfy adapter is loaded through ICLoraPipeline(loras=), NEVER the PEFT loader."""
    code = _fns_code()

    # The official adapter is fetched + passed to run_ic_lora_baseline(official_adapter_path=...).
    assert re.search(r"official_adapter_path\s*=", code), (
        "the baseline branch must resolve an official_adapter_path (the comfy single-file adapter)"
    )
    # The baseline branch must NOT route the official adapter through signet's PEFT loader. The PEFT
    # loader (load_lora_onto_transformer) is the base-vs-LoRA path; the format-mismatch guard is that the
    # official comfy adapter never touches it. Assert the load surface is run_ic_lora_baseline's loras=,
    # not a load_lora_onto_transformer(...) call on official_adapter_path.
    assert not re.search(r"load_lora_onto_transformer\([^)]*official_adapter", code), (
        "format-mismatch guard (Pitfall 4): the official comfy adapter must NOT be loaded via the "
        "PEFT loader (load_lora_onto_transformer) — only via ICLoraPipeline(loras=[...])"
    )


def test_baseline_reads_reference_downscale_from_metadata() -> None:
    code = _fns_code()

    assert "read_reference_downscale_factor" in code, (
        "the baseline branch must READ reference_downscale_factor from the adapter metadata (Pitfall 4b)"
    )
    # The read result is threaded into run_ic_lora_baseline (honored, not discarded).
    assert re.search(r"reference_downscale_factor\s*=\s*ref_downscale", code), (
        "the metadata-read downscale must be passed to run_ic_lora_baseline (honored)"
    )


# --------------------------------------------------------------------------------------------------
# (3) the baseline config names the official adapter repo
# --------------------------------------------------------------------------------------------------


def test_baseline_config_names_official_adapter() -> None:
    assert _BASELINE_CONFIG.exists(), (
        "configs/ltx23_ic_lora_baseline.example.yaml must exist (the tier-2 baseline run config)"
    )
    text = _BASELINE_CONFIG.read_text(encoding="utf-8")
    assert "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control" in text, (
        "the baseline config must name the official Union-Control adapter repo (D-7-BASELINE)"
    )


def test_official_adapter_repo_constant_matches_config() -> None:
    """The fns.py constant and the config comment name the SAME official adapter repo (single truth)."""
    code = _fns_code()
    assert 'OFFICIAL_IC_LORA_BASELINE_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"' in code, (
        "fns.py must define OFFICIAL_IC_LORA_BASELINE_REPO = the Union-Control repo (the download source)"
    )


# --------------------------------------------------------------------------------------------------
# (4) #22 finding 2 — the 22B + Gemma component load must be SKIPPED on this leg, not just unread
# --------------------------------------------------------------------------------------------------
#
# Before the fix: sample() called load_ltxv_components(..., with_text_encoder=cached_by_prompt is
# None) UNCONDITIONALLY at (3), before the ic_lora_baseline branch's `mode == "ic_lora" and
# two_stage` gate. Because two_stage=True skips PHASE A (cached_by_prompt stays None),
# with_text_encoder resolved True — dragging the full 22B transformer + Gemma-12B onto CUDA — and
# the baseline branch below never reads `components` at all; it builds its own ICLoraPipeline from
# raw weight paths and owns its own Gemma. Two 22B transformers + two Gemma-12Bs cannot coexist on
# one A100-80GB (the two-phase VRAM rule): guaranteed OOM before the first clip, for a container
# that goes on to do all its own loading anyway.


def test_component_load_is_skipped_for_ic_lora_baseline() -> None:
    """The load_ltxv_components() call at (3) must not fire when mode=='ic_lora' and two_stage."""
    code = _fns_code()

    # The `components = load_ltxv_components(...)` assignment must be conditioned on the SAME
    # predicate that gates the ic_lora_baseline branch (`mode == "ic_lora" and two_stage`), so the
    # 22B + Gemma load never executes on that leg.
    m = re.search(
        r'components\s*=\s*\(\s*None\s*\n\s*if\s*\(\s*config\.conditioning\.mode\s*==\s*'
        r'["\']ic_lora["\']\s*and\s*two_stage\s*\)\s*\n\s*else\s*load_ltxv_components\(',
        code,
    )
    assert m is not None, (
        "sample()'s (3) component load must be `None if (config.conditioning.mode == 'ic_lora' "
        "and two_stage) else load_ltxv_components(...)` — a bare unconditional "
        "`load_ltxv_components(...)` call reintroduces the guaranteed-OOM-before-first-clip bug "
        "(#22 finding 2): the ic_lora_baseline branch never reads `components`."
    )

    # The load call itself must sit strictly AFTER the guard predicate in source order (belt and
    # suspenders — proves the guard actually wraps the load, not merely co-occurs with it).
    guard_idx = code.index('if (config.conditioning.mode == "ic_lora" and two_stage)')
    load_idx = code.index("load_ltxv_components(", guard_idx)
    assert load_idx > guard_idx, (
        "the ic_lora/two_stage guard must precede the load_ltxv_components(...) call it protects"
    )

    # And the ic_lora_baseline branch itself must appear AFTER the (now-guarded) load call — i.e.
    # this is a gate on the load, not a relocation of the branch above it (either fix shape is
    # acceptable per the issue, but the guard-shape fix must not also duplicate the branch).
    baseline_branch_idx = code.index('if config.conditioning.mode == "ic_lora" and two_stage:')
    assert baseline_branch_idx > load_idx, (
        "the ic_lora_baseline branch should still appear once, after the guarded component load "
        "statement"
    )
