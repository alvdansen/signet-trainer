"""local/runner.py LTX-2.5 refusal (issue #53 Stage 1) — CPU-only, mirrors test_local_runner.py's
own ``refusals()``-object-level style exactly.

Stage 1 is Modal-only (ltx25_gpu_image / ltx25_train). The local runner still imports the
UNMODIFIED, 2.3-only ``load_ltxv_components`` / ``run_validation_gate`` — without this refusal a
``model.ltx_generation == '2.5'`` config would sail past every other check (``family`` stays
``"ltx"`` for both generations) and crash cryptically deep inside a multi-GB load.
"""

from __future__ import annotations

from pathlib import Path

from signet_trainer.config.load import load_config
from signet_trainer.local import ROADMAP_ISSUE
from signet_trainer.local.runner import refusals

REPO = Path(__file__).resolve().parents[1]
LTX_EXAMPLE = REPO / "configs" / "ltx23_lora.example.yaml"


def _ltx_config():
    return load_config(LTX_EXAMPLE)


def test_refuses_ltx_generation_25_with_issue_53_pointer() -> None:
    cfg = _ltx_config().model_copy(deep=True)
    object.__setattr__(cfg.model, "ltx_generation", "2.5")
    blockers = refusals(cfg)
    assert any("ltx_generation" in b and "issue #53" in b for b in blockers)


def test_ltx_generation_23_has_no_new_refusal() -> None:
    """Byte-identity: an ordinary config (ltx_generation defaults to '2.3') is unaffected."""
    assert refusals(_ltx_config()) == []


def test_refusal_survives_alongside_family_refusal() -> None:
    """A doubly-wrong config (family AND generation) names BOTH — refusals() collects every
    applicable reason, it does not stop at the first."""
    cfg = _ltx_config().model_copy(deep=True)
    object.__setattr__(cfg.model, "family", "h3")
    object.__setattr__(cfg.model, "ltx_generation", "2.5")
    blockers = refusals(cfg)
    assert any("family" in b and ROADMAP_ISSUE in b for b in blockers)
    assert any("ltx_generation" in b for b in blockers)
