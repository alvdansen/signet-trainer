"""Run-gate structural invariants (HARN-03 / SC#3) — pure source-order text scans, zero spend.

Locks the two disciplines every metered dispatch depends on, for ALL SIX dispatches — the three LTX
stages (train / sample / preprocess) and their Phase-10 MiniMax-H3 counterparts (h3_train /
h3_sample / h3_preprocess), which ride the SAME three ``--mode`` values and route on
``cfg.model.family`` inside the same arms (H3-07). No exemption is granted to the H3 leg: a new
dispatch arm is either covered here or the gate silently stops describing reality. Text scans
against ``entrypoint.py`` source (no Modal, no GPU — runs in the CPU suite):

  * MODL-02 stop-at-gate: every ``*.spawn(`` dispatch appears AFTER the blocking
    ``_require_approval`` pause — no metered run can auto-launch ahead of approval.
  * MODL-03 cost-before-approval: the cost banner is printed BEFORE the approval pause — the
    operator always sees the estimate before authorizing.
  * The non-interactive approval path fails CLOSED: ``_require_approval`` treats ``EOFError``
    (piped/CI stdin, no ``--approve``) as declined (``answer = ""``) — never auto-spends.

Copies the ``_strip_comments_and_docstrings`` scan convention from test_gated_launch_order.py so
prose mentions of ``.spawn`` / ``approved`` don't false-positive.

D-10-DEF-17 swapped the dispatch VERB (``.remote`` -> ``.spawn``, SYNC -> ASYNC). The two
invariants asserted here are about ORDERING, not about the verb, and are unchanged by that.
"""

from __future__ import annotations

import re
from pathlib import Path

_ENTRYPOINT = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal" / "entrypoint.py"

#: Every gated stage the entrypoint dispatches under train / sample / preprocess. The three ``h3_*``
#: entries landed with Phase 10 (H3-07): they are NOT new modes — the existing arms route on
#: ``cfg.model.family`` — but they ARE new metered dispatches, so they owe the same two invariants.
_GATED_DISPATCHES = ("train", "sample", "preprocess", "h3_train", "h3_sample", "h3_preprocess")


def _dispatch_pattern(fn: str) -> str:
    """``fn[.with_options(...)].spawn(<something>``, anchored so the six checks are six DISTINCT claims.

    Two guards, both load-bearing, both found by mutation rather than by inspection:

    * ``(?<![\\w.])`` — ``h3_train.spawn(`` literally CONTAINS ``train.spawn(``, so the unanchored
      form would let the H3 dispatch satisfy the LTX assertion (and vice versa): three real checks
      wearing six names. The ``.`` is in the class for the same reason — ``obj.sample.spawn(`` must
      not satisfy a check for the module-level ``sample``.
    * ``\\(\\s*(?!\\))`` — the call must pass an ARGUMENT. ``_strip_comments_and_docstrings`` removes
      comments and triple-quoted blocks but NOT ordinary string literals, and every arm's APPROVED
      print contains the sentence ``Dispatching <fn>.spawn() (gated…)``. Without this lookahead,
      DELETING a dispatch leaves its own log message behind to satisfy the scan — a gate describing
      a call that is not there. Every real dispatch passes ``config_text`` or ``**h3_params``; only
      the prose writes empty parens.
    """
    return rf"(?<![\w.]){re.escape(fn)}(?:\.with_options\([^)]*\))?\.spawn\s*\(\s*(?!\))"


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop triple-quoted blocks + line comments so prose mentions don't trip the source-order scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _entrypoint_code() -> str:
    return _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))


def _last_approval_idx(code: str) -> int:
    approval_calls = list(re.finditer(r"_require_approval\s*\(", code))
    assert approval_calls, "entrypoint.py must call _require_approval (MODL-02)"
    # The LAST occurrence is the single gate CALL in main() (the earlier one is the `def`).
    return approval_calls[-1].start()


def test_all_six_dispatches_follow_the_approval_gate() -> None:
    """MODL-02: all SIX gated ``.spawn(`` calls dispatch AFTER the approval pause (no auto-launch).

    Six, not three: the three LTX stages plus their H3 counterparts, which share the same three
    ``--mode`` values and route on ``cfg.model.family``.
    """
    code = _entrypoint_code()
    approval_idx = _last_approval_idx(code)

    seen: dict[str, int] = {}
    for fn in _GATED_DISPATCHES:
        # sample/preprocess (and all three H3 arms) dispatch via
        # ``{fn}.with_options(timeout=...).spawn(`` (09.1-04, AUDIT-#5 config-derived timeout);
        # LTX train stays a bare ``train.spawn(``. The optional with_options segment must not defeat
        # the MODL-02 source-order scan.
        remote_match = re.search(_dispatch_pattern(fn), code)
        assert remote_match is not None, (
            f"entrypoint.py must dispatch {fn}.spawn() (all 6 gated dispatches)"
        )
        assert remote_match.start() > approval_idx, (
            f"{fn}.spawn() must appear AFTER the _require_approval gate (MODL-02: no auto-launch) — "
            f"approval@{approval_idx} vs {fn}.spawn@{remote_match.start()}"
        )
        seen[fn] = remote_match.start()

    assert len(set(seen.values())) == len(_GATED_DISPATCHES), (
        f"the six dispatch checks must land on six DISTINCT source positions, got {seen} — the "
        f"anchored pattern is matching a substring and the loop is not really six checks."
    )


def test_cost_line_is_printed_before_the_approval_gate() -> None:
    """MODL-03: the cost banner (format_cost_line) is printed BEFORE the approval pause."""
    code = _entrypoint_code()
    approval_idx = _last_approval_idx(code)

    cost_match = re.search(r"format_cost_line\s*\(", code)
    assert cost_match is not None, "entrypoint.py must print the cost banner via format_cost_line (MODL-03)"
    assert cost_match.start() < approval_idx, (
        "the cost line must be printed BEFORE _require_approval (MODL-03: cost-before-approval) — "
        f"cost@{cost_match.start()} vs approval@{approval_idx}"
    )


def test_guardrail_check_precedes_all_six_dispatches() -> None:
    """The cost/guardrail computation must precede every gated dispatch (MODL-03) — all six."""
    code = _entrypoint_code()
    cost_match = re.search(r"guardrail_check\s*\(", code)
    assert cost_match is not None, "entrypoint.py must run guardrail_check before dispatch (MODL-03)"
    for fn in _GATED_DISPATCHES:
        # Tolerate the optional ``.with_options(timeout=...)`` (09.1-04, AUDIT-#5; every H3 arm too).
        remote_match = re.search(_dispatch_pattern(fn), code)
        assert remote_match is not None and cost_match.start() < remote_match.start(), (
            f"guardrail_check must run before {fn}.spawn() (MODL-03)"
        )


def test_non_interactive_approval_declines_on_eoferror() -> None:
    """The approval gate fails CLOSED: EOFError (piped/CI stdin, no --approve) -> declined (answer="")."""
    code = _entrypoint_code()
    # The EOFError handler sets answer = "" so the .lower()=="approved" check below fails -> declined.
    assert re.search(r"except\s+EOFError\s*:", code), (
        "_require_approval must handle EOFError (non-interactive stdin) — never auto-spend"
    )
    assert re.search(r"answer\s*=\s*[\"']{2}", code), (
        "the EOFError branch must set answer = \"\" so a piped run is DECLINED, not authorized"
    )
