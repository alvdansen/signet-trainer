"""AUDIT #34 direction 7 — ``h3_train``'s marker wiring, pinned by a source scan.

The KEY (what makes the marker current vs. stale) is tested behaviourally in
``tests/test_acceptance_marker.py`` against ``train/acceptance_marker.py`` directly — a stdlib-only
module, safe to import. This file is the structural counterpart: it proves ``h3_train`` actually
CALLS that key at the right place in its own body, WITHOUT importing ``modal/fns.py`` (importing it
builds the Modal app graph and eagerly resolves every ``Secret.from_name`` — the same reason
``tests/test_h3_train_wiring.py`` gives for scanning it as text).

Two properties matter, and both fail EXPENSIVELY if wrong:
  * the marker check must run BEFORE ``run_h3_arch_gate`` (the 61.7 GiB load) — checking it AFTER
    defeats the entire point of the marker;
  * the marker write must run BEFORE the ``delta == 0.0`` raise — writing it after a raise never
    executes (an unreachable write is as useless as no marker at all).

CPU-only, zero GPU, zero Modal spend.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"#.*", "", src)


def _code() -> str:
    return _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))


def _function_source(code: str, name: str) -> str:
    match = re.search(rf"^def {re.escape(name)}\(", code, re.M)
    assert match, f"{name}() not found in modal/fns.py"
    tail = re.search(r"^(?:def |class |@)", code[match.end() :], re.M)
    end = match.end() + tail.start() if tail else len(code)
    return code[match.start() : end]


def _train_body() -> str:
    return _function_source(_code(), "h3_train")


def test_acceptance_marker_is_imported_from_the_stdlib_only_module() -> None:
    """The KEY must come from ``train/acceptance_marker.py``, not be re-derived inline in
    ``modal/fns.py`` — a second copy is exactly the class of drift AUDIT #19 named."""
    code = _code()
    assert "from signet_trainer.train.acceptance_marker import" in code, (
        "h3_train's marker helpers must import the key from train/acceptance_marker.py rather "
        "than reimplementing the step/run scoping logic inline."
    )


def test_marker_check_precedes_the_61_7_gib_load() -> None:
    """``_h3_check_acceptance_marker`` must run BEFORE ``run_h3_arch_gate`` — the whole point is a
    container-boot-cheap re-raise instead of paying for the load first."""
    body = _train_body()
    check_idx = body.index("_h3_check_acceptance_marker(")
    gate_idx = body.index("run_h3_arch_gate(")
    assert check_idx < gate_idx, (
        f"marker check at {check_idx} must precede the 61.7 GiB load (run_h3_arch_gate) at "
        f"{gate_idx} — checking after the load defeats the marker's purpose (AUDIT #34 dir. 7)."
    )


def test_marker_check_precedes_the_cpu_preflight_too() -> None:
    """Cheaper still: the marker check should short-circuit before even the CPU preflight builds
    a real batch off the dataset."""
    body = _train_body()
    check_idx = body.index("_h3_check_acceptance_marker(")
    # The CPU preflight's first real call — comments are stripped, so the section's prose
    # header ("── (3) CPU PREFLIGHT ──") is not visible here; this is the actual code.
    preflight_idx = body.index("prepare_training_inputs(")
    assert check_idx < preflight_idx


def test_marker_write_precedes_the_zero_delta_raise() -> None:
    """``_h3_write_acceptance_failed_marker`` must run BEFORE the ``delta == 0.0`` raise it exists
    to make cheap on retry — written after the raise would never execute."""
    body = _train_body()
    delta_check_idx = body.index("delta == 0.0")
    write_idx = body.index("_h3_write_acceptance_failed_marker(")
    raise_idx = body.index("RuntimeError", delta_check_idx)
    assert delta_check_idx < write_idx < raise_idx, (
        f"delta==0.0 at {delta_check_idx}, marker write at {write_idx}, raise at {raise_idx} — "
        "the marker must be written strictly between the zero-delta check and the raise."
    )


def test_the_raise_itself_is_unmodified_and_still_pinned() -> None:
    """Direction 7 changes ONLY the retry's cost, never the raise — test_a_zero_adapter_delta_raises
    in test_h3_train_wiring.py pins that deliberately; this test cross-checks the same property
    from this file so a future edit that guts the raise while 'fixing' the marker goes red here too.
    """
    body = _train_body()
    assert "h3_adapter_delta(" in body
    delta_check_idx = body.index("delta == 0.0")
    assert "RuntimeError" in body[delta_check_idx : delta_check_idx + 700]


def test_checkpoints_vol_is_reloaded_before_the_marker_check() -> None:
    """reload-before-read (house convention, see e.g. runner sample stages): the entry check must
    see everything the PRIOR life committed, not a stale mount view."""
    body = _train_body()
    reload_idx = body.index("checkpoints_vol.reload()")
    check_idx = body.index("_h3_check_acceptance_marker(")
    assert reload_idx < check_idx
