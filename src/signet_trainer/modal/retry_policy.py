"""Per-arm server-side retry budgets (issue #45 PR-2) — the ONE table the cost gate prices from.

The ``@app.function`` decorators in ``fns.py`` grant Modal server-side retries: a preempted arm
re-runs on a FRESH container up to ``max_retries`` times, each life with its own timeout. The
launch therefore authorizes ``max_retries + 1`` container lives — and the pre-launch cost print /
guardrail / harness ledger must price THAT quantity, not one life (PR-2: an LTX ``train`` dispatch
behind a $3.28-shaped print authorized 11 x bounded-hours of metered A100).

CRITICAL — Anti-Pattern 6: this module imports NOTHING from ``modal`` (it lives under modal/ for
cohesion but is pure stdlib, so the Windows test suite and the entrypoint's cost math can read it
without hydrating the app graph). The decorators keep their literal ints — Modal function objects
expose no stable public surface to read a retry policy back off the resolved function — so this
table is welded to those literals by ``tests/test_retry_policy.py`` (an AST scan of every shipped
decorator): a decorator budget that drifts from this table fails the suite, never the cost print.
"""

from __future__ import annotations

__all__ = ["ARM_MAX_RETRIES", "resolve_arm"]

#: ``max_retries`` per dispatch arm, mirroring the shipped ``fns.py`` decorators (weld-tested).
#: Keyed by the ``@app.function`` NAME each entrypoint arm spawns. Arms with no ``retries=`` kwarg
#: carry 0 (one container life). Lives = value + 1.
ARM_MAX_RETRIES: dict[str, int] = {
    "preprocess": 0,
    "train": 10,  # Modal's server ceiling — the long RESUMABLE round self-heals.
    "sample": 3,  # AUDIT-#1 (d); renders resume in-dir (identity-keyed + per-clip commit, PR-2).
    "h3_preprocess": 0,
    "h3_train": 10,  # same ceiling, same resumable-round contract as train.
    "h3_sample": 0,  # deliberately no retries — re-dispatch resumes instead (D-10 resume).
    "qwen_edit_preprocess": 0,
    "qwen_edit_train": 10,  # same resumable-round contract as train / h3_train.
    "qwen_edit_sample": 0,  # deliberately no retries — same shape as h3_sample minus the retries.
    # wan_train carries NO retries, DELIBERATELY (fns.py's own decorator comment): musubi's resume
    # semantics are unverified here, so a retry over an unverified resume is not a safety feature.
    "wan_train": 0,
    "fuse": 0,
    "restore": 0,
    "backup_sync": 0,
}


def resolve_arm(mode: str, family: str) -> str:
    """The ``@app.function`` NAME the entrypoint dispatches for ``(mode, family)``.

    Mirrors the entrypoint's dispatch ladder exactly: ``fuse`` / ``restore`` / ``backup`` route to
    their single CPU arm regardless of family; the ``wan`` family serves exactly ONE dispatch arm
    (``wan_train`` — every other mode is refused before ``.spawn()`` by ``_wan_refuse_on_gaps``, but
    this resolver still needs an answer for the cost print that runs BEFORE that refusal); ``h3`` and
    ``qwen_edit`` each get their own ``sample``/``preprocess``/``train`` arm; ``ltx`` (the default
    family) falls through to the bare arm name. ``train`` is the default mode, matching the
    entrypoint's trailing ``else`` arm.
    """
    if mode == "fuse":
        return "fuse"
    if mode == "restore":
        return "restore"
    if mode == "backup":
        return "backup_sync"
    if family == "wan":
        # wan serves exactly one supported dispatch (mode == "train" -> wan_train); every other mode
        # is refused before any spawn, but the cost print above that refusal still needs a priceable
        # arm to resolve, so every wan mode maps to the one arm this family ever dispatches.
        return "wan_train"
    if mode in ("sample", "preprocess"):
        if family == "h3":
            return f"h3_{mode}"
        if family == "qwen_edit":
            return f"qwen_edit_{mode}"
        return mode
    if family == "h3":
        return "h3_train"
    if family == "qwen_edit":
        return "qwen_edit_train"
    return "train"
