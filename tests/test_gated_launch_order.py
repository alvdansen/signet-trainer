"""MODL-02 gate-order for the Phase-4 ``sample`` dispatch — reuses the train gate, no auto-launch.

The 04-03 sample launch path reuses ``entrypoint.py::main``'s EXACT cost -> approval -> dispatch
gate (a ``--mode`` branch), so the metered base-vs-LoRA grid is gated identically to train. These
CPU source-order scans (no Modal run, zero spend) prove the ``sample.spawn(`` dispatch can never
precede the blocking ``_require_approval`` pause — mirroring ``tests/test_train_entrypoint_wiring.py``
and ``tests/test_no_warm_gpu.py``'s text-scan convention.
"""

from __future__ import annotations

import re
from pathlib import Path

_MODAL_DIR = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal"
_ENTRYPOINT = _MODAL_DIR / "entrypoint.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop triple-quoted blocks + line comments so prose mentions don't trip the source-order scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _entrypoint_code() -> str:
    return _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))


def test_sample_remote_dispatched_after_approval_gate() -> None:
    """The ``sample.spawn(`` dispatch must follow the ``_require_approval`` call in source order.

    Same invariant as train (MODL-02 / T-03-61): the sample run is metered, so its dispatch can
    never precede the blocking approval pause — no auto-launch of a metered grid. (D-10-DEF-17
    swapped the verb ``.remote`` -> ``.spawn``; the ORDERING invariant is unchanged.)
    """
    code = _entrypoint_code()

    # 09.1-04 (AUDIT-#5): sample dispatches via ``sample.with_options(timeout=...).spawn(``
    # (config-derived timeout). The optional with_options segment must not defeat the MODL-02 scan.
    remote_match = re.search(r"sample(?:\.with_options\([^)]*\))?\.spawn\s*\(", code)
    assert remote_match is not None, "entrypoint.py must dispatch sample.spawn() (04-03 wiring)"

    approval_calls = [m for m in re.finditer(r"_require_approval\s*\(", code)]
    assert approval_calls, "entrypoint.py must call _require_approval before dispatch (MODL-02)"
    # The last _require_approval occurrence is the single gate call in main() (the def is earlier).
    approval_call_idx = approval_calls[-1].start()

    assert approval_call_idx < remote_match.start(), (
        "sample.spawn() must appear AFTER the _require_approval gate (MODL-02: no auto-launch) — "
        f"approval@{approval_call_idx} vs sample.spawn@{remote_match.start()}"
    )


def test_sample_remote_passes_config_text() -> None:
    """The sample dispatch forwards the YAML text by value (the container re-parses + revalidates)."""
    code = _entrypoint_code()
    assert re.search(r"sample(?:\.with_options\([^)]*\))?\.spawn\s*\(\s*config_text\s*\)", code), (
        "sample.spawn(config_text) must forward the config YAML text so the container re-runs the "
        "gate without needing the file on its filesystem"
    )


def test_cost_and_dryrun_precede_sample_remote() -> None:
    """Both the cost estimate and the dry-run gate must run before the sample dispatch (MODL-03)."""
    code = _entrypoint_code()
    remote_match = re.search(r"sample(?:\.with_options\([^)]*\))?\.spawn\s*\(", code)
    assert remote_match is not None

    cost_match = re.search(r"guardrail_check\s*\(|estimate_cost\s*\(", code)
    dryrun_match = re.search(r"run_dryrun\s*\(|dryrun", code)
    assert cost_match is not None and cost_match.start() < remote_match.start(), (
        "the cost estimate must be computed before sample.spawn() (MODL-03)"
    )
    assert dryrun_match is not None and dryrun_match.start() < remote_match.start(), (
        "the dry-run gate must run before sample.spawn()"
    )


def test_sample_seam_in_train_is_not_a_bare_pass() -> None:
    """SC#3: the train in-loop sampler callback wires the sampler (no bare ``pass`` no-op).

    Since the 2026-07-11 mid-run-cadence fix the seam lives in the ``_in_loop_sample`` callback
    (invoked by train_loop's ``on_checkpoint`` at every checkpoint boundary), with the
    ``offloader_suspended`` wrapper around the render INSIDE the callback. Asserts the callback
    references build_generation_config + run_sampler + save_video + the checkpoint_every-coupled
    due-gate, and wraps the render in offloader_suspended.
    """
    fns = _MODAL_DIR / "fns.py"
    code = _strip_comments_and_docstrings(fns.read_text(encoding="utf-8"))

    seam = re.search(
        r"def _in_loop_sample\(step: int\) -> None:(.*?)dataset\s*=\s*PrecomputedDataset",
        code,
        re.DOTALL,
    )
    assert seam is not None, "could not locate the _in_loop_sample callback in train"
    body = seam.group(1)

    assert "pass" not in body.split(), "the seam must no longer be a bare `pass` no-op (SC#3)"
    for token in (
        "build_generation_config",
        "run_sampler",
        "save_video",
        "checkpoint_every",
        "with offloader_suspended(offloader, block_list):",
    ):
        assert token in body, f"the in-loop seam must reference {token!r} (SC#3 structural wiring)"
