"""Bound-the-metered-shell source scans (09.1-04 / AUDIT-#5 + #1) — pure CPU, zero spend.

Source-order + presence scans (no Modal, no GPU) proving the Wave-2 hardening of the gated
dispatch path:

  * AUDIT-#5 — ``sample``/``preprocess`` dispatch with a CONFIG-DERIVED Modal timeout
    (``est_hours * timeout_margin``) via ``.with_options(timeout=...)`` instead of the hardcoded
    24h decorator ceiling, so a wedged/CUDA-hung render is killed early (not burned to ~$40);
  * ``train`` KEEPS its 24h decorator — its dispatch stays a bare ``train.spawn(config_text)``
    with NO ``.with_options(timeout=...)`` override;
  * AUDIT-#1 (e) — no dispatch print claims ``(detached gated run)``. ⚠ The ORIGINAL justification
    ("``.remote()`` is not itself detached") is now FALSE: D-10-DEF-17 replaced the verb with
    ``.spawn()``, which IS an async dispatch. The ban SURVIVES for a DIFFERENT and stronger reason —
    the entrypoint CANNOT OBSERVE whether the run is detached, because ``--detach`` is a CLI-level
    property of the app SHELL that the entrypoint does not control. See
    ``test_no_false_detached_phrase_in_entrypoint`` for the full statement;
  * MODL-02 — the ``.with_options(...).spawn(`` dispatch is STILL strictly after ``_require_approval``;
  * AUDIT-#1 (d) — ``sample()`` carries ``retries=modal.Retries`` so a preempted detached render
    self-heals server-side (scanned against ``fns.py``).

Mirrors the ``_strip_comments_and_docstrings`` text-scan convention of test_entrypoint_gate.py /
test_gated_launch_order.py so prose mentions don't false-positive.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ENTRYPOINT = _ROOT / "src" / "signet_trainer" / "modal" / "entrypoint.py"
_FNS = _ROOT / "src" / "signet_trainer" / "modal" / "fns.py"

#: Anchor every ``{fn}.`` scan so an ``h3_``-prefixed dispatch cannot satisfy an LTX assertion.
#:
#: Phase 10 (H3-07) added ``h3_train`` / ``h3_sample`` / ``h3_preprocess`` alongside the LTX three,
#: and ``h3_train.with_options(timeout=`` literally CONTAINS ``train.with_options(timeout=`` — so an
#: unanchored scan for one dispatch form can spuriously match the OTHER function's line. (Historical
#: note: this bit an earlier version of ``test_train_dispatch_no_longer_stays_24h_no_with_options``
#: back when LTX ``train`` was still a bare ``train.spawn(config_text)`` and the H3 match was the
#: false positive; issue #45 PR-2 retired that exemption, but the anchor stays load-bearing for the
#: ``qwen_edit_train`` / ``wan_train`` scans below, which have the identical substring hazard.) Same
#: lookbehind the sibling gate scans use.
_ANCHOR = r"(?<![\w.])"


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop triple-quoted blocks + line comments so prose mentions don't trip the source scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _entrypoint_code() -> str:
    return _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))


def _fns_code() -> str:
    return _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# AUDIT-#5 — config-derived sample/preprocess timeout (Task 1)
# ---------------------------------------------------------------------------

def test_sample_dispatch_applies_with_options_timeout() -> None:
    """The sample dispatch must go through ``sample.with_options(timeout=...).spawn(`` (AUDIT-#5)."""
    code = _entrypoint_code()
    assert re.search(rf"{_ANCHOR}sample\.with_options\(\s*timeout\s*=", code), (
        "sample must dispatch via sample.with_options(timeout=...).spawn(...) — config-derived "
        "timeout, not the hardcoded 24h decorator ceiling (AUDIT-#5)"
    )


def test_preprocess_dispatch_applies_with_options_timeout() -> None:
    """The preprocess dispatch must go through ``preprocess.with_options(timeout=...).spawn(`` (AUDIT-#5)."""
    code = _entrypoint_code()
    assert re.search(rf"{_ANCHOR}preprocess\.with_options\(\s*timeout\s*=", code), (
        "preprocess must dispatch via preprocess.with_options(timeout=...).spawn(...) — config-derived "
        "timeout, not the hardcoded 24h decorator ceiling (AUDIT-#5)"
    )


def test_derived_timeout_reads_est_hours_and_timeout_margin() -> None:
    """The derived timeout must be computed from ``cfg.modal.est_hours * cfg.modal.timeout_margin`` (D-NOHARDCODE)."""
    code = _entrypoint_code()
    # A timeout local bound from the two config knobs (config-first, not a literal). Allow any local
    # name but require the est_hours * timeout_margin * 3600 shape.
    assert re.search(
        r"int\(\s*cfg\.modal\.est_hours\s*\*\s*cfg\.modal\.timeout_margin\s*\*\s*3600\s*\)",
        code,
    ), (
        "the sample/preprocess timeout must be derived as int(cfg.modal.est_hours * "
        "cfg.modal.timeout_margin * 3600) — config-first, no hardcoded literal (D-NOHARDCODE / AUDIT-#5)"
    )


def test_train_dispatch_no_longer_stays_24h_no_with_options() -> None:
    """issue #45 PR-2 RETIRED LTX train()'s 24h-decorator exemption (AUDIT-#5's original deviation).

    LTX train's dispatch used to stay a bare ``train.spawn(config_text)`` so a driver-level hang
    would burn to the fixed 24h decorator ceiling — the SAME gap the H3/qwen_edit/wan train arms
    already closed for themselves (see the two positive assertions below). PR-2 closed it for LTX
    train too, for a SECOND reason beyond the hang: the cost gate now prices the worst-case ceiling
    as ``rate * bounded_hours * lives``, and pricing a bound the dispatch never actually used would
    make the printed/guardrailed estimate dishonest. train's dispatch now carries the SAME
    config-derived timeout bound every other GPU arm uses.
    """
    code = _entrypoint_code()
    assert re.search(
        rf"{_ANCHOR}train\.with_options\(timeout=train_timeout_s\)\.spawn\s*\(\s*config_text\s*\)",
        code,
    ), (
        "train must dispatch via train.with_options(timeout=train_timeout_s).spawn(config_text) — "
        "the 24h-decorator exemption is retired (issue #45 PR-2)"
    )


def test_h3_train_and_qwen_edit_train_and_wan_train_bound_their_shell_with_a_config_derived_timeout() -> None:
    """The H3 / qwen_edit / wan train arms bound their metered shell at ``est_hours *
    timeout_margin`` — LTX train (above) now matches them uniformly (issue #45 PR-2 retired the
    deviation this test used to document); pinned here as a positive claim on the OTHER three arms
    so a future regression can't quietly re-exempt any of the four from the config-derived bound.
    Reason it matters beyond honesty in the cost print: a driver-level hang burning to a bare 24h
    decorator ceiling is ~$40+ of A100 for nothing.
    """
    code = _entrypoint_code()
    assert re.search(r"h3_train\.with_options\(\s*timeout\s*=[^)]*\)\.spawn\s*\(", code), (
        "the H3 train arm must dispatch via h3_train.with_options(timeout=...).spawn(...) — a "
        "config-derived bound on the metered shell, not the 24h decorator ceiling"
    )
    assert re.search(r"qwen_edit_train\.with_options\(\s*timeout\s*=[^)]*\)\.spawn\s*\(", code), (
        "the qwen_edit train arm must dispatch via qwen_edit_train.with_options(timeout=...)."
        "spawn(...) — a config-derived bound on the metered shell"
    )
    assert re.search(r"wan_train\.with_options\(\s*timeout\s*=[^)]*\)\.spawn\s*\(", code), (
        "the wan train arm must dispatch via wan_train.with_options(timeout=...).spawn(...) — a "
        "config-derived bound on the metered shell"
    )


def test_with_options_timeout_dispatch_follows_approval_gate() -> None:
    """MODL-02: the ``.with_options(...).spawn(`` dispatch stays strictly after ``_require_approval``."""
    code = _entrypoint_code()
    approval_calls = list(re.finditer(r"_require_approval\s*\(", code))
    assert approval_calls, "entrypoint.py must call _require_approval (MODL-02)"
    approval_idx = approval_calls[-1].start()

    for fn in ("sample", "preprocess", "h3_train", "h3_sample", "h3_preprocess"):
        dispatch = re.search(
            rf"{_ANCHOR}{fn}\.with_options\(\s*timeout\s*=[^)]*\)\.spawn\s*\(", code
        )
        assert dispatch is not None, f"{fn} must dispatch via with_options(timeout=...).spawn()"
        assert dispatch.start() > approval_idx, (
            f"{fn}.with_options(...).spawn() must appear AFTER _require_approval (MODL-02) — "
            f"approval@{approval_idx} vs dispatch@{dispatch.start()}"
        )


# ---------------------------------------------------------------------------
# AUDIT-#1 (e) — honest dispatch prints (Task 1)
# ---------------------------------------------------------------------------

def test_no_false_detached_phrase_in_entrypoint() -> None:
    """The ``(detached gated run)`` ban SURVIVES D-10-DEF-17 — with an INVERTED rationale.

    ⚠ The original justification for this assertion was *"``.remote()`` is not itself detached"*.
    That justification is now **FALSE**: the dispatch verb is ``.spawn()``, which really is an
    ASYNC dispatch the server does not cancel when its client dies.

    **The assertion nonetheless stands, for a DIFFERENT and stronger reason: the entrypoint CANNOT
    OBSERVE whether the run is detached.** ``--detach`` is a CLI-level property of the ephemeral app
    SHELL, not of the input: ``.spawn()`` makes the INPUT async, but a non-detached ephemeral app is
    still stopped at client exit — and the entrypoint does not control that flag. So the print must
    not claim detachment. It announces a **spawned (async) gated dispatch**, prints the
    **FunctionCall id** (the checkable fact), and prints an **advisory** when ``--detach`` is missing
    from ``sys.argv``.

    The ban therefore moves from *"the verb is wrong"* to *"the entrypoint may not assert a property
    it cannot see."* This is an OPEN, STATED choice: the alternative — permitting the phrase now that
    the verb is async — was considered and REJECTED, because it would let the entrypoint claim a
    survival property it cannot verify.
    """
    raw = _ENTRYPOINT.read_text(encoding="utf-8")
    assert "(detached gated run)" not in raw, (
        "entrypoint.py must not print '(detached gated run)'. The verb is now .spawn() (async), but "
        "detachment is a property of the APP SHELL (the CLI --detach flag), which this module cannot "
        "observe — so it may not assert it. Announce the SPAWNED gated dispatch, print the "
        "FunctionCall id, and warn when --detach is absent from sys.argv (D-10-DEF-17 / AUDIT-#1e)."
    )


def test_all_three_dispatch_prints_remain_gated() -> None:
    """The three dispatch prints still announce a gated dispatch (the corrected, honest wording)."""
    raw = _ENTRYPOINT.read_text(encoding="utf-8")
    for fn in ("sample", "preprocess", "train"):
        assert re.search(rf"Dispatching {fn}\.spawn\(\) \(gated\)", raw), (
            f"the {fn} dispatch print must announce 'Dispatching {fn}.spawn() (gated)' (honest wording)"
        )


# ---------------------------------------------------------------------------
# AUDIT-#1 (d) — sample() server-side retries (Task 2)
# ---------------------------------------------------------------------------

def _sample_decorator_block() -> str:
    """The ``@app.function(...)`` block immediately preceding ``def sample(`` in fns.py.

    Isolates the sample decorator so the retries assertion can't be satisfied by train()'s (or any
    other function's) retries elsewhere in the module.
    """
    code = _fns_code()
    def_idx = re.search(r"\ndef\s+sample\s*\(", code)
    assert def_idx is not None, "fns.py must define a sample() function"
    head = code[: def_idx.start()]
    # The nearest @app.function( opening before def sample( starts this function's decorator block.
    dec_idx = head.rfind("@app.function(")
    assert dec_idx != -1, "sample() must carry an @app.function(...) decorator"
    return head[dec_idx:]


def test_sample_decorator_carries_modal_retries() -> None:
    """AUDIT-#1 (d): the sample() @app.function decorator must carry ``retries=modal.Retries`` (like train())."""
    block = _sample_decorator_block()
    assert re.search(r"retries\s*=\s*modal\.Retries\s*\(", block), (
        "sample()'s @app.function decorator must set retries=modal.Retries(...) so a preempted "
        "detached render self-heals server-side (AUDIT-#1d), mirroring train()"
    )


# ---------------------------------------------------------------------------
# The PREEMPTION CONTRACT (2026-08-06) — structure, not prose
#
# The deliberate call, recorded here so it is a pinned property rather than a comment that erodes:
#   * LTX train()  RIDES ALONG   — a long RESUMABLE round (CheckpointManager.resume in-dir,
#                                  commit-per-save), the identical defect class to h3_train.
#   * LTX sample() IS EXCLUDED   — a RENDER is not resumable; a retry re-does the whole thing
#                                  rather than continuing it, so a bigger budget would multiply a
#                                  total-loss unit. (The 2026-08-05 LTX incident burned ~2.8 A100-h
#                                  for zero usable output on exactly this shape.)
# ---------------------------------------------------------------------------


def _train_decorator_block() -> str:
    """The ``@app.function(...)`` block immediately preceding ``def train(`` in fns.py.

    Twin of ``_sample_decorator_block`` — isolates LTX ``train``'s decorator so an assertion cannot
    be satisfied by ``h3_train``'s (or any other function's) kwargs elsewhere in the module.
    """
    code = _fns_code()
    def_idx = re.search(rf"\n{_ANCHOR}def\s+train\s*\(", code)
    assert def_idx is not None, "fns.py must define a train() function"
    head = code[: def_idx.start()]
    dec_idx = head.rfind("@app.function(")
    assert dec_idx != -1, "train() must carry an @app.function(...) decorator"
    return head[dec_idx:]


def test_single_use_containers_is_a_real_modal_sdk_kwarg() -> None:
    """The self-deriving half: ``single_use_containers`` must be a REAL parameter of the INSTALLED
    ``modal.App.function``.

    Derived from ``inspect.signature`` rather than a hand-copied literal, so a future SDK that
    renames or drops the kwarg fails LOUDLY here instead of letting a dead literal sit in the
    decorator doing nothing in production. Verified present at modal 1.5.0.
    """
    import inspect  # noqa: PLC0415 — local so the module import stays SDK-free

    import modal  # noqa: PLC0415

    assert "single_use_containers" in inspect.signature(modal.App.function).parameters, (
        "modal.App.function no longer accepts single_use_containers — the fresh-container-per-retry "
        "contract on train()/h3_train() is now a no-op literal and must be re-derived"
    )


#: Modal's SERVER-enforced ceiling on ``max_retries`` — measured 2026-08-06 on app
#: ``ap-8Gra2Yka1fs4pwMIh8AgLv``: a dispatch with ``max_retries=60`` is rejected at app init with
#: "Invalid function retries. Must specify number between 0 and 10". Zero containers, zero spend.
_MODAL_SERVER_MAX_RETRIES = 10


def test_the_retry_budget_is_within_modals_server_ceiling() -> None:
    """⚠ THE CLIENT DOES NOT VALIDATE THE REAL BOUND — so the bound is asserted here.

    ``modal/retries.py`` validates only ``max_retries >= 0``; ``modal.Retries(max_retries=60, ...)``
    constructs happily and then the DISPATCH is rejected server-side at app init. A constructor
    round-trip is therefore NOT evidence that a retry policy is dispatchable, which is the whole
    lesson of the 2026-08-06 rejection — so this test asserts BOTH halves.

    ``max_delay`` is capped at 60.0 s by the SDK and ``initial_delay`` is already 60.0, so
    ``backoff_coefficient=2.0`` is clamped from the very first retry: the budget adds a BOUNDED
    queue delay, never a doubling tail.
    """
    import modal  # noqa: PLC0415

    match = re.search(r"max_retries\s*=\s*(\d+)", _train_decorator_block())
    assert match is not None, "train() must declare modal.Retries(max_retries=<int>, ...)"
    budget = int(match.group(1))

    modal.Retries(max_retries=budget, initial_delay=60.0, backoff_coefficient=2.0)
    assert budget <= _MODAL_SERVER_MAX_RETRIES, (
        f"max_retries={budget} exceeds Modal's server ceiling of {_MODAL_SERVER_MAX_RETRIES}. The "
        "SDK constructor accepts it, so this fails only at `modal run` app init — killing the "
        "dispatch, not the suite. Pinned locally so it fails HERE instead."
    )


def test_train_mirrors_the_h3_preemption_contract() -> None:
    """LTX ``train`` carries the SAME two properties as ``h3_train`` — it rode along deliberately.

    Same safety argument (resumable in-dir, commit-per-save), zero metered cost to apply. Leaving
    it under-configured while fixing only the H3 twin would repeat verbatim the prose-not-structure
    failure the change exists to close.
    """
    block = _train_decorator_block()
    assert "single_use_containers=True" in block.replace(" ", ""), (
        "train()'s @app.function must set single_use_containers=True (fresh container per retry)"
    )
    budget = re.search(r"max_retries\s*=\s*(\d+)", block)
    assert budget is not None, "train() must declare modal.Retries(max_retries=<int>, ...)"
    assert int(budget.group(1)) == _MODAL_SERVER_MAX_RETRIES, (
        f"train()'s max_retries={budget.group(1)} must be exactly {_MODAL_SERVER_MAX_RETRIES} — "
        "Modal's server ceiling. The observed preemption cadence wants 44 container lives, which "
        "the platform cannot grant (D-10-DEF-16, open); take the ceiling and pin the shortfall."
    )


def test_sample_is_deliberately_excluded_from_single_use_containers() -> None:
    """LTX ``sample`` must NOT carry ``single_use_containers`` — the exclusion is STRUCTURAL.

    **A render is not resumable in-dir.** ``sample`` writes a FRESH timestamped samples dir per
    render, so a retry does not continue the previous attempt — it re-does the whole thing. A
    bigger retry budget therefore multiplies a total-loss unit rather than salvaging progress, the
    exact inverse of the training case. Its ``retries=3`` (AUDIT-#1d) stays as-is.

    Asserted rather than left as a comment because a prose-only exclusion is precisely the failure
    mode the preemption fix exists to close: the H3 side already encodes it structurally
    (``h3_sample`` carries no retries at all, pinned by
    ``test_h3_train_declares_retries_and_h3_sample_does_not``) and the LTX side must too.
    """
    assert "single_use_containers" not in _sample_decorator_block(), (
        "sample() must NOT declare single_use_containers — a render is not resumable, so a fresh "
        "container per retry buys nothing and a raised budget multiplies a total-loss unit"
    )


def test_preprocess_decorator_has_no_retries() -> None:
    """The retries addition is scoped to sample() ONLY — preprocess retries are Wave-2 finding #13, out of scope."""
    code = _fns_code()
    def_idx = re.search(r"\ndef\s+preprocess\s*\(", code)
    assert def_idx is not None, "fns.py must define a preprocess() function"
    head = code[: def_idx.start()]
    dec_idx = head.rfind("@app.function(")
    assert dec_idx != -1, "preprocess() must carry an @app.function(...) decorator"
    block = head[dec_idx:]
    assert not re.search(r"retries\s*=\s*modal\.Retries", block), (
        "preprocess() must NOT carry retries in this plan (finding #13 is Wave-2, out of scope for 09.1-04)"
    )
