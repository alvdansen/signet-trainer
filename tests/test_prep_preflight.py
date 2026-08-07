"""prep.preflight tests (spec-driven inpaint dims gate). Pure CPU — NO modal, NO GPU, NO network.

Locks ``check_inpaint_rules``: it reads ``spatial_divisor`` + ``frame_rule`` FROM the loaded
MASK-SPEC dict (D-03 — never a hardcoded 64/8) and returns the exact crop/trim guidance for a
violating clip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from signet_trainer.prep.preflight import check_inpaint_rules

# The minimal spec slice the preflight actually reads.
_SPEC = {"dims": {"spatial_divisor": 64, "frame_rule": "8n+1"}}


def test_passes_div64_and_8n1() -> None:
    r = check_inpaint_rules(768, 512, 81, _SPEC)  # 81 == 8*10 + 1
    assert r["passes"] is True
    assert r["needed_fix"] == []


def test_flags_non_div64_spatial() -> None:
    r = check_inpaint_rules(770, 500, 81, _SPEC)
    assert r["passes"] is False
    assert not r["div_spatial_w"] and not r["div_spatial_h"]
    assert any("crop W" in f for f in r["needed_fix"])
    assert any("crop H" in f for f in r["needed_fix"])


def test_flags_non_8n1_frames() -> None:
    r = check_inpaint_rules(768, 512, 80, _SPEC)  # (80 - 1) % 8 != 0
    assert r["passes"] is False
    assert not r["frames_ok"]
    assert any("trim frames" in f for f in r["needed_fix"])


def test_dims_read_from_spec_not_hardcoded() -> None:
    """A spec with a DIFFERENT divisor/period proves the gate is spec-driven, not a literal 64/8."""
    spec = {"dims": {"spatial_divisor": 32, "frame_rule": "4n+1"}}
    # 96 is div32 but NOT div64; 33 is 4n+1 but NOT 8n+1 — passes only under the spec's own rules.
    r = check_inpaint_rules(96, 96, 33, spec)
    assert r["passes"] is True
    # And the same clip fails the div64/8n+1 spec.
    assert check_inpaint_rules(96, 96, 33, _SPEC)["passes"] is False


def test_malformed_frame_rule_raises() -> None:
    with pytest.raises(ValueError):
        check_inpaint_rules(768, 512, 81, {"dims": {"spatial_divisor": 64, "frame_rule": "bogus"}})


def test_preflight_is_import_confined() -> None:
    import signet_trainer.prep.preflight as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for banned in ("import torch", "import cv2", "import modal", "from torch ", "from modal "):
        assert banned not in src, f"{banned!r} leaked into prep.preflight (must stay stdlib only)"
