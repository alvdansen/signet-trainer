"""mode=preprocess routes through the SINGLE validated gate (D-8-PREPROC).

Source-order + attribute-path scans (no Modal, no GPU) proving the Phase-8 preprocess arm:
  * is a first-class ``--mode`` alongside train/sample (mode guard includes "preprocess");
  * dispatches ``preprocess.spawn(`` STRICTLY after ``_require_approval`` (MODL-02 — no auto-launch);
  * reads the reference params from ``cfg.conditioning.*`` (NOT ``cfg.data.*`` — a wrong path is an
    AttributeError AFTER a burned approval; the runtime test proves where the fields actually live);
  * carries NO hardcoded ``HOURLY_RATE_USD`` (the retired scripts/_encode_* drift is gone — the arm
    reuses the shared ``cfg.modal`` cost line).
Mirrors the text-scan convention of test_gated_launch_order.py + test_preprocess_wiring.py.
"""

from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENTRYPOINT = _ROOT / "src" / "signet_trainer" / "modal" / "entrypoint.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop triple-quoted blocks + line comments so prose mentions don't trip the source-order scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _entrypoint_code() -> str:
    return _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))


def test_mode_guard_includes_preprocess() -> None:
    """The ``main`` mode guard must accept "preprocess" as a first-class gated mode (D-8-PREPROC)."""
    code = _entrypoint_code()
    guard = re.search(r"mode\s+not\s+in\s*\(([^)]*)\)", code)
    assert guard is not None, "entrypoint.py must guard `mode not in (...)`"
    modes = guard.group(1)
    for expected in ("train", "sample", "preprocess"):
        assert expected in modes, f"the mode guard must include {expected!r}, got: {modes}"


def test_preprocess_remote_dispatched_after_approval_gate() -> None:
    """``preprocess.spawn(`` must follow the ``_require_approval`` call in source order (MODL-02)."""
    code = _entrypoint_code()

    # 09.1-04 (AUDIT-#5): preprocess dispatches via ``preprocess.with_options(timeout=...).spawn(``
    # (config-derived timeout; verb swapped to the ASYNC .spawn by D-10-DEF-17). The optional
    # with_options segment must not defeat the MODL-02 scan.
    remote_match = re.search(r"preprocess(?:\.with_options\([^)]*\))?\.spawn\s*\(", code)
    assert remote_match is not None, "entrypoint.py must dispatch preprocess.spawn() (D-8-PREPROC)"

    approval_calls = list(re.finditer(r"_require_approval\s*\(", code))
    assert approval_calls, "entrypoint.py must call _require_approval before dispatch (MODL-02)"
    approval_call_idx = approval_calls[-1].start()

    assert approval_call_idx < remote_match.start(), (
        "preprocess.spawn() must appear AFTER the _require_approval gate (MODL-02: no auto-launch) — "
        f"approval@{approval_call_idx} vs preprocess.spawn@{remote_match.start()}"
    )


def test_preprocess_arm_reads_conditioning_reference_paths() -> None:
    """The reference params must come from ``cfg.conditioning.*`` — NOT ``cfg.data.*`` (attribute-path guard).

    ``reference_column`` / ``reference_downscale_factor`` live on ConditioningConfig; reading them off
    ``cfg.data`` would raise AttributeError AFTER _require_approval succeeds (a burned approval), since
    ``_Base`` is extra="forbid".
    """
    code = _entrypoint_code()
    assert "cfg.conditioning.reference_column" in code, (
        "preprocess arm must read cfg.conditioning.reference_column (correct attribute path)"
    )
    assert "cfg.conditioning.reference_downscale_factor" in code, (
        "preprocess arm must read cfg.conditioning.reference_downscale_factor (correct attribute path)"
    )
    assert "cfg.data.reference_column" not in code, (
        "cfg.data.reference_column is the WRONG path (ConditioningConfig owns it) — AttributeError trap"
    )
    assert "cfg.data.reference_downscale_factor" not in code, (
        "cfg.data.reference_downscale_factor is the WRONG path — AttributeError trap"
    )


def test_no_hardcoded_hourly_rate_literal() -> None:
    """No literal ``HOURLY_RATE_USD`` in entrypoint.py — the arm reuses cfg.modal (kills the drift)."""
    code = _entrypoint_code()
    assert "HOURLY_RATE_USD" not in code, (
        "entrypoint.py must not carry a hardcoded HOURLY_RATE_USD (read cfg.modal.hourly_rate_usd)"
    )


def test_preprocess_arm_reads_metadata_and_output_from_config() -> None:
    """Config-first (D-NOHARDCODE): the arm reads metadata_path / preprocessed_data_root from cfg."""
    code = _entrypoint_code()
    assert "cfg.data.metadata_path" in code, "preprocess arm must read cfg.data.metadata_path"
    assert "cfg.data.preprocessed_data_root" in code, (
        "preprocess arm must read cfg.data.preprocessed_data_root for the output dir"
    )


def test_preprocess_arm_threads_overwrite_knob() -> None:
    """#35 step 3: the re-encode knob must reach preprocess.spawn(overwrite=cfg.data.preprocess_overwrite)."""
    code = _entrypoint_code()
    assert re.search(r"overwrite\s*=\s*cfg\.data\.preprocess_overwrite", code), (
        "preprocess.spawn(...) must pass overwrite=cfg.data.preprocess_overwrite — before #35 "
        "step 3 this was hardcoded overwrite=False with no config knob at all"
    )


def test_runtime_reference_fields_live_on_conditioning_not_data() -> None:
    """Load a real example YAML and prove the reference fields resolve on conditioning, not data."""
    from signet_trainer.config.load import load_config_from_text

    cfg = load_config_from_text(
        (_ROOT / "configs" / "ltx23_lora.example.yaml").read_text(encoding="utf-8")
    )
    assert hasattr(cfg.conditioning, "reference_column")
    assert hasattr(cfg.conditioning, "reference_downscale_factor")
    assert not hasattr(cfg.data, "reference_column"), (
        "cfg.data must NOT carry reference_column (it lives on ConditioningConfig)"
    )
    # The new DataConfig.metadata_path field must exist and be defaulted (existing configs still load).
    assert hasattr(cfg.data, "metadata_path")


def test_existing_example_config_still_loads() -> None:
    """Adding DataConfig.metadata_path must not break existing train/sample configs (defaulted field)."""
    from signet_trainer.config.load import load_config_from_text

    cfg = load_config_from_text(
        (_ROOT / "configs" / "ltx23_lora.example.yaml").read_text(encoding="utf-8")
    )
    assert cfg.data.metadata_path  # defaulted, non-empty


def test_reference_params_gated_on_ic_lora_mode() -> None:
    """mode != ic_lora must dispatch reference_column=None (the burned campaign gate, 09-07 T3).

    ``ConditioningConfig.reference_column`` defaults to ``"reference_path"`` but is ic_lora-only;
    passing it through on a ``mode: none`` encode makes the canonical ``compute_latents`` raise
    ``Key 'reference_path' not found in JSONL entry`` AFTER approval. The dispatch must route
    through ``_reference_encode_params`` which returns (None, 1) for every non-ic_lora mode.
    Behavioral: the helper is AST-extracted and exec'd (no modal import — pure CPU).
    """
    import ast
    from types import SimpleNamespace

    src = _ENTRYPOINT.read_text(encoding="utf-8")

    # 1. The dispatch itself must consume the helper, not cfg.conditioning.* inline.
    code = _strip_comments_and_docstrings(src)
    # issue #19 item 2 — the naive pattern (no lookbehind, non-greedy tail stopping at the FIRST
    # inner `)`) matched the plain APPROVED print's own string literal — "Dispatching
    # preprocess.spawn() (gated); ..." — instead of the real call, and proved it by mutation: with
    # `reference_column=cfg.conditioning.reference_column` inlined into the real `.spawn(...)`, the
    # assertion below still passed. Two fixes, both load-bearing:
    #   * `(?<![\w.])` before `preprocess` — blocks matching `h3_preprocess.spawn(...)` (the H3 arm's
    #     own APPROVED print, `preprocess` as a substring of `h3_preprocess`) and
    #     `cfg.data.preprocessed_data_root` (`preprocess` as a substring after a literal `.`).
    #   * `\.spawn\s*\(\s*(?!\))` (test_no_warm_gpu.py's existing convention) — requires a
    #     NON-EMPTY arg list, which the print's literal `preprocess.spawn()` (empty parens) fails.
    # The tail `(?:.|\n)*?\n\s*\)` anchors the close on a `)` alone on its own line, so a real
    # multi-line kwarg call is captured whole rather than truncated at the first inner `)`.
    dispatch = re.search(
        r"(?<![\w.])preprocess(?:\.with_options\([^)]*\))?\.spawn\s*\(\s*(?!\))(?:.|\n)*?\n\s*\)",
        code,
    )
    assert dispatch is not None, "entrypoint.py must dispatch preprocess.spawn(...)"
    assert "cfg.conditioning.reference_column" not in dispatch.group(0), (
        "preprocess.spawn must not read cfg.conditioning.reference_column inline — "
        "route through _reference_encode_params (ic_lora-only gating)"
    )
    assert "_reference_encode_params" in code, "helper _reference_encode_params must exist"

    # 2. Behavioral check of the extracted helper (no modal import).
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_reference_encode_params"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    namespace: dict = {}
    exec(compile(module, "<extracted>", "exec"), namespace)  # noqa: S102 — test-local exec
    helper = namespace["_reference_encode_params"]

    plain = SimpleNamespace(
        conditioning=SimpleNamespace(mode="none", reference_column="reference_path", reference_downscale_factor=2)
    )
    assert helper(plain) == (None, 1)

    ic = SimpleNamespace(
        conditioning=SimpleNamespace(mode="ic_lora", reference_column="reference_path", reference_downscale_factor=2)
    )
    assert helper(ic) == ("reference_path", 2)


# ---- issue #19 item 2 — the dispatch-guard regex must match the CALL, never a print's string ----

_H3_PRINT_SNIPPET = '''
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching h3_preprocess.spawn() (gated, family: h3); the two-phase encode writes "
        )
        fc = h3_preprocess.with_options(timeout=h3_preprocess_timeout_s).spawn(**h3_params)
'''

_PLAIN_PRINT_SNIPPET = '''
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching preprocess.spawn() (gated); the canonical process_dataset.py "
            "encode writes {latents,conditions}/ to cfg.data.preprocessed_data_root."
        )
        fc = preprocess.with_options(timeout=preprocess_timeout_s).spawn(
            metadata_path=cfg.data.metadata_path,
            reference_column=reference_column,
        )
'''

_DISPATCH_GUARD_RE = (
    r"(?<![\w.])preprocess(?:\.with_options\([^)]*\))?\.spawn\s*\(\s*(?!\))(?:.|\n)*?\n\s*\)"
)


def test_dispatch_guard_regex_skips_h3_print_and_preprocessed_data_root() -> None:
    """The regex must not match `h3_preprocess.spawn()` (a print literal) nor treat the `.` before
    `preprocessed_data_root` as a word boundary — both are substrings of `preprocess`, not the
    dispatch call, and the OLD (no-lookbehind) pattern could not tell them apart from the real one.
    """
    match = re.search(_DISPATCH_GUARD_RE, _H3_PRINT_SNIPPET)
    assert match is None, (
        f"regex must not match the H3 print's own string literal, matched: {match.group(0)!r}"
    )


def test_dispatch_guard_regex_catches_inlined_reference_column_mutation() -> None:
    """Mutation proof (the exact regression the issue reports): with
    `reference_column=cfg.conditioning.reference_column` inlined into the REAL `.spawn(...)` call,
    the fixed regex's match must be the call itself — not the plain-preprocess print's empty-parens
    `preprocess.spawn()` — so the forbidden-inline assertion actually sees the mutation instead of
    silently passing against a vacuous match.
    """
    mutated = _PLAIN_PRINT_SNIPPET.replace(
        "reference_column=reference_column,",
        "reference_column=cfg.conditioning.reference_column,",
    )
    match = re.search(_DISPATCH_GUARD_RE, mutated)
    assert match is not None, "regex must still find the real .spawn(...) call"
    assert "cfg.conditioning.reference_column" in match.group(0), (
        "the match must be the real dispatch call (which now carries the inlined mutation), not "
        f"the print's vacuous 'preprocess.spawn()' literal — got: {match.group(0)!r}"
    )
