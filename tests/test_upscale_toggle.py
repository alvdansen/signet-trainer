"""INFR-01 / D-DEPTH-1 — the two-stage spatial-upscaler toggle (DEFAULT OFF) + import confinement.

CPU unit tests for the 04-05 two-stage path. Three things are proven here (all on CPU, no GPU,
no ltx-pipelines installed):

  1. ``config.validation.two_stage_upscale`` DEFAULTS to False — single-stage stays the grid
     default (correctness-first). A run must OPT IN to the upscaler.
  2. ``inference/upscale.py`` has NO module-level ``ltx_pipelines`` / ``ltx_core`` / ``modal``
     import (source scan, reusing the ``test_lora.py`` scanner) — the heavy
     ``TI2VidTwoStagesPipeline`` import is function-local (Anti-Pattern 6).
  3. The module IMPORTS on CPU without ltx-pipelines installed (it is only added to ``gpu_image``
     Modal-side by 04-05 Task 1), so the toggle wiring unit-tests off-GPU.

Import-confinement discipline: the clean-import test snapshots ``sys.modules`` and RESTORES it in a
``finally`` — popping ``torch`` (or any shared module) without restoring poisons later tests (the
bug just fixed in test_grid_html.py; do NOT reintroduce it).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from signet_trainer.config.schema import ValidationConfig

# The module under guard (path used for the text scan) — mirrors test_lora.py's _PEFT_SRC.
_UPSCALE_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "signet_trainer"
    / "inference"
    / "upscale.py"
)
_FORBIDDEN_MODULES = ("ltx_pipelines", "ltx_core", "modal")


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


# --------------------------------------------------------------------------------------------------
# (1) the toggle defaults OFF
# --------------------------------------------------------------------------------------------------


def test_two_stage_upscale_defaults_off() -> None:
    assert ValidationConfig().two_stage_upscale is False, (
        "two_stage_upscale must default False — single-stage is the base-vs-LoRA grid default "
        "(D-DEPTH-1, correctness-first). A run must opt into the upscaler."
    )


# --------------------------------------------------------------------------------------------------
# (2) Anti-Pattern-6 import confinement — no module-level heavy import in the source
# --------------------------------------------------------------------------------------------------


def test_upscale_source_has_no_ltx_pipelines_or_modal_import() -> None:
    # Column-0 anchored (``^`` not ``^\s*``): only MODULE-LEVEL imports match — indented
    # function-local + ``if TYPE_CHECKING:`` imports never execute on CPU/CI and are allowed
    # (upscale.py's ``TI2VidTwoStagesPipeline`` import is deliberately function-local). Mirrors
    # tests/test_no_wan_params.py::_MODULE_LEVEL_HEAVY_IMPORT.
    code = _strip_comments_and_docstrings(_UPSCALE_SRC.read_text(encoding="utf-8"))
    module_level_heavy_import = re.compile(
        r"^(?:import|from)\s+(?:ltx_pipelines|ltx_core|modal)\b", re.MULTILINE
    )
    hits = module_level_heavy_import.findall(code)
    assert not hits, f"import-confinement violation: module-level heavy import in upscale.py: {hits}"


def test_confinement_regex_would_catch_a_top_level_import() -> None:
    """RED self-check: column-0 heavy import flagged; indented function-local one is not."""
    rx = re.compile(r"^(?:import|from)\s+(?:ltx_pipelines|ltx_core|modal)\b", re.MULTILINE)
    assert rx.search("from ltx_pipelines.ti2vid_two_stages import X\n")
    assert not rx.search("    from ltx_pipelines.ti2vid_two_stages import X\n")


# --------------------------------------------------------------------------------------------------
# (3) the module imports on CPU without ltx-pipelines installed (snapshot + restore sys.modules)
# --------------------------------------------------------------------------------------------------


def test_upscale_module_imports_on_cpu_without_ltx_pipelines() -> None:
    # Snapshot the forbidden modules so we can prove upscale.py never pulled them in AND restore
    # any pre-existing entry afterward (popping a shared module without restoring poisons later
    # tests — do NOT reintroduce the test_grid_html.py bug).
    saved = {mod: sys.modules.get(mod) for mod in _FORBIDDEN_MODULES}
    try:
        for mod in _FORBIDDEN_MODULES:
            sys.modules.pop(mod, None)

        import signet_trainer.inference.upscale as upscale  # noqa: F401

        # The heavy imports are function-local, so merely importing the module must NOT have
        # pulled ltx_pipelines / ltx_core / modal into sys.modules.
        for mod in _FORBIDDEN_MODULES:
            assert mod not in sys.modules, (
                f"import-confinement violation: inference.upscale transitively imported {mod!r}"
            )
        # The public toggle entrypoint exists (wired into fns.py::sample).
        assert hasattr(upscale, "run_two_stage")
    finally:
        for mod, prev in saved.items():
            if prev is not None:
                sys.modules[mod] = prev
            else:
                sys.modules.pop(mod, None)
