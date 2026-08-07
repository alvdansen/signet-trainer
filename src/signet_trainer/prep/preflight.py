"""Inpaint dims preflight — the ÷N spatial + Kn+1 frame gate, spec-driven (D-03).

Harvested from ``scripts/_prep_inpaint_testvideos.py`` L75-86 (``check_inpaint_rules``), but the
divisor and frame period are read FROM the loaded MASK-SPEC dict rather than hardcoded — the gate
is single-sourced with the spec, so a campaign that retunes the dims rules changes ONE file. Surfaces
BOTH the ÷64 spatial rule (previously enforced only at config load) and the 8n+1 frame rule
(previously only in staging) at prep time.

Import-confined: stdlib ``re`` only — no torch/cv2/modal.
"""

from __future__ import annotations

import re


def _frame_period(frame_rule: str) -> int:
    """Parse a '<period>n+1' frame rule -> the integer period. '8n+1' -> 8.

    Reading the period from the spec string (rather than a literal 8) keeps the gate spec-driven.
    """
    m = re.fullmatch(r"\s*(\d+)\s*n\s*\+\s*1\s*", str(frame_rule))
    if not m:
        raise ValueError(
            f"unsupported frame_rule {frame_rule!r}; expected the '<period>n+1' shape (e.g. '8n+1')"
        )
    period = int(m.group(1))
    if period <= 0:
        raise ValueError(f"frame_rule period must be positive, got {period}")
    return period


def check_inpaint_rules(width: int, height: int, frames: int, spec: dict) -> dict:
    """Inpaint dims gate: W/H divisible by ``spec['dims']['spatial_divisor']``, frames on the spec
    ``frame_rule`` (Kn+1).

    ``spec`` is a loaded MASK-SPEC dict (pass ``load_mask_spec(...)`` once and reuse). The divisor and
    period come FROM the spec (D-03 — never a hardcoded 64/8). Returns a report dict with the exact
    crop/trim guidance in ``needed_fix`` (mirrors the D-04 UI-SPEC dims error copy: "requires ÷64
    spatial and (frames-1) divisible by 8"). Trim, never pad.
    """
    dims = spec["dims"]
    divisor = int(dims["spatial_divisor"])
    period = _frame_period(dims["frame_rule"])

    ok_w = width % divisor == 0
    ok_h = height % divisor == 0
    ok_n = (frames - 1) % period == 0

    fixes: list[str] = []
    if not ok_w:
        fixes.append(f"crop W {width}->{(width // divisor) * divisor}")
    if not ok_h:
        fixes.append(f"crop H {height}->{(height // divisor) * divisor}")
    if not ok_n:
        fixes.append(
            f"trim frames {frames}->{((frames - 1) // period) * period + 1} "
            f"({period}n+1, never pad)"
        )

    return {
        "passes": ok_w and ok_h and ok_n,
        "div_spatial_w": ok_w,
        "div_spatial_h": ok_h,
        "frames_ok": ok_n,
        "needed_fix": fixes,
        "rule": f"requires ÷{divisor} spatial and (frames-1) divisible by {period}",
    }
