"""Structural doc-coverage proof (HARN-04) + skill<->code drift guard (D-9-TESTHARDEN).

Two layers, both pure CPU / zero Modal / zero GPU:

1. **Doc-coverage (HARN-04):** every ROADMAP-declared per-phase "Agent entry point" seam
   is referenced by at least one ``.claude/skills/*/SKILL.md`` (the ``ENTRY_POINT_TOKENS``
   substring scan — correct for prose seams like ``latents/`` / ``process_dataset``).

2. **Drift guard (D-9-TESTHARDEN):** the ``--mode`` VALUES and the ``--config`` / ``--approve``
   / ``--mode`` FLAGS the skills document must actually EXIST on the REAL entrypoint CLI
   surface — so a skill documenting a non-existent ``--mode`` (or a renamed flag) FAILS the
   test instead of silently passing a substring scan. modal's ``@app.local_entrypoint()``
   derives the CLI from ``main``'s signature (``config`` -> ``--config`` etc.) and the accepted
   ``--mode`` values are the string literals in ``main``'s ``mode not in (...)`` validation
   tuple. We recover both by **AST-parsing** ``entrypoint.py`` as text (never importing it —
   importing pulls the modal SDK and would break the pure-CPU invariant, akin to reading an
   argparse spec statically).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILLS_DIR = _REPO_ROOT / ".claude" / "skills"
_ENTRYPOINT_PY = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "entrypoint.py"

# The four lifecycle skills the coverage assertion scans (Plans 08-04 / 08-05).
_EXPECTED_SKILLS = frozenset(
    {
        "training-session-setup",
        "training-prep",
        "training-run",
        "training-review",
    }
)

# token -> (phase label, why this is the load-bearing seam for that phase).
# Derived from the ROADMAP "Agent entry points (Phase 8 harness)" lines on Phases 1-7
# (plus the read-only status seam the Phase-8 harness surfaces). A future added seam is
# a one-line addition; the phase label is what the failure message reports.
ENTRY_POINT_TOKENS: dict[str, tuple[str, str]] = {
    # --- Phase 1: dryrun subcommand · load_config loader · modal/fns.py boundary ---
    "dryrun": ("Phase 1", "the CPU dry-run gate subcommand"),
    "load_config": ("Phase 1", "the load_config(yaml) loader"),
    "signet_trainer.modal.entrypoint": (
        "Phase 1",
        "the Modal entrypoint boundary every gated run is driven through",
    ),
    # --- Phase 2: preprocess fn · load_ltxv_components smoke · latents/ + conditions/ ---
    "--mode preprocess": ("Phase 2", "the cheap gated pre-encode mode"),
    "process_dataset": (
        "Phase 2",
        "the canonical process_dataset.py encoder (NOT a custom encoder)",
    ),
    "latents/": ("Phase 2", "the precomputed latents artifact path on the Volume"),
    "conditions/": ("Phase 2", "the precomputed text-embed conditions artifact path"),
    # --- Phase 3: gated train entrypoint · validation-gate/approval · checkpoint/resume ---
    "_require_approval": (
        "Phase 3",
        "the gated-train approval pause seam (metered runs never auto-launch)",
    ),
    "find_latest": ("Phase 3", "checkpoint/resume newest-checkpoint resolution"),
    # --- Phase 4: sample entrypoint emitting the base-vs-LoRA grid ---
    "--mode sample": ("Phase 4", "the sample/inference entrypoint"),
    "base-vs-LoRA": ("Phase 4", "the base-vs-LoRA comparison grid the harness fetches"),
    "grid build": ("Phase 4", "the gridwatch grid build over the fetched samples"),
    # --- Read-only status seam (Phase 3 run-state / Phase 8 surface), no spend ---
    "modal app list": ("Phase 3/8 status", "the read-only run-state snapshot"),
    "modal volume ls": (
        "Phase 3/8 status",
        "the read-only checkpoint/dataset progress snapshot",
    ),
    # --- Phase 5: single_frame reference mode ---
    "single_frame": ("Phase 5", "the single-frame reference mode (conditioning.mode)"),
    # --- Phase 6: multi-condition config ---
    "multi_frame": ("Phase 6", "the multi-frame reference mode (conditioning.mode)"),
    # --- Phase 7: ic_lora mode + reference_latents/ preprocessing ---
    "ic_lora": ("Phase 7", "the IC-LoRA in-context mode (conditioning.mode)"),
    # --- Phase 9: inpaint gate — the gated CPU In-Outpainting scaffold fuse ---
    "--mode fuse": ("Phase 9", "the gated CPU In-Outpainting fuse job (inpaint scaffold base)"),
    "reference_latents": (
        "Phase 7",
        "the reference_latents/ preprocessing an ic_lora run needs",
    ),
}


def _skill_files() -> list[Path]:
    return sorted(_SKILLS_DIR.glob("*/SKILL.md"))


def _combined_skill_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in _skill_files())


def test_all_four_lifecycle_skills_present() -> None:
    """The four lifecycle skills the coverage scan asserts against must all exist."""
    names = {p.parent.name for p in _skill_files()}
    missing = _EXPECTED_SKILLS - names
    assert not missing, f"missing lifecycle skill(s): {sorted(missing)}"


@pytest.mark.parametrize("token", sorted(ENTRY_POINT_TOKENS))
def test_entry_point_token_referenced_by_a_skill(token: str) -> None:
    """Every declared per-phase entry point is referenced by at least one skill (HARN-04)."""
    text = _combined_skill_text().lower()
    phase, why = ENTRY_POINT_TOKENS[token]
    assert token.lower() in text, (
        f"{phase} entry point UNDOCUMENTED: no .claude/skills/*/SKILL.md references "
        f"'{token}' ({why}). Document the seam in a skill, or update ENTRY_POINT_TOKENS."
    )


def test_every_phase_1_through_7_has_at_least_one_token() -> None:
    """The map must cover at least one entry point per Phase 1-7 (a gap here is a hole)."""
    covered = {phase for phase, _ in ENTRY_POINT_TOKENS.values()}
    for n in range(1, 8):
        label = f"Phase {n}"
        assert any(c.startswith(label) for c in covered), (
            f"{label} has no entry-point token mapped in ENTRY_POINT_TOKENS"
        )


def test_module_imports_nothing_from_modal() -> None:
    """This coverage test is pure-CPU: it must not import the modal SDK (acceptance criterion)."""
    src = Path(__file__).read_text(encoding="utf-8")
    for lineno, line in enumerate(src.splitlines(), start=1):
        assert not re.match(r"\s*(import\s+modal\b|from\s+modal\b)", line), (
            f"line {lineno} imports modal — the coverage test must stay pure-CPU"
        )


# ==========================================================================================
# D-9-TESTHARDEN — validate documented --mode values / flags against the REAL entrypoint CLI
# surface (like inspecting an argparse spec), WITHOUT importing modal.
# ==========================================================================================


def _entrypoint_cli_surface() -> tuple[frozenset[str], frozenset[str]]:
    """Statically read the real entrypoint CLI surface (modes, flags) via AST — no modal import.

    modal's ``@app.local_entrypoint()`` generates the CLI from ``main``'s signature (each param
    ``x`` becomes ``--x``), and the accepted ``--mode`` values are the string literals in ``main``'s
    ``mode not in (...)`` validation tuple. AST-parsing the source recovers both without importing
    the module (which would pull the modal SDK and break the pure-CPU invariant) — the static
    equivalent of reading an argparse ``add_argument`` spec.
    """
    tree = ast.parse(_ENTRYPOINT_PY.read_text(encoding="utf-8"))
    main_fn = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ),
        None,
    )
    assert main_fn is not None, (
        "entrypoint.py no longer defines a main() @app.local_entrypoint — the CLI surface moved; "
        "update this drift guard."
    )
    # Flags = the local-entrypoint params (config/approve/mode -> --config/--approve/--mode).
    flags = frozenset(arg.arg for arg in main_fn.args.args)
    # Modes = the string literals in the `mode not in (...)` membership validation.
    modes: set[str] = set()
    for node in ast.walk(main_fn):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "mode"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.NotIn)
            and isinstance(node.comparators[0], (ast.Tuple, ast.List, ast.Set))
        ):
            for elt in node.comparators[0].elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    modes.add(elt.value)
    return frozenset(modes), flags


_MODE_DOC_RE = re.compile(r"--mode\s+(<[^>]*>|[A-Za-z_][A-Za-z0-9_]*)")
_MODE_TOKEN_RE = re.compile(r"^[a-z_]+$")


def _documented_mode_values(text: str) -> set[str]:
    """Extract the concrete ``--mode <value>`` tokens the skills document.

    Handles both the placeholder form ``--mode <train|sample|preprocess>`` (split on ``|``) and the
    bare form ``--mode preprocess``. Non-identifier placeholders (e.g. ``<...>``) are ignored.
    """
    values: set[str] = set()
    for raw in _MODE_DOC_RE.findall(text):
        if raw.startswith("<"):
            for part in raw.strip("<>").split("|"):
                part = part.strip()
                if _MODE_TOKEN_RE.match(part):
                    values.add(part)
        elif _MODE_TOKEN_RE.match(raw):
            values.add(raw)
    return values


def test_entrypoint_mode_surface_is_the_expected_triple() -> None:
    """Sanity: the static AST parse recovers the real gated-mode triple (guards a silent parse fail)."""
    modes, _flags = _entrypoint_cli_surface()
    assert modes == {"train", "sample", "preprocess", "fuse", "restore", "backup"}, (
        f"AST-parsed entrypoint --mode surface {sorted(modes)} != expected "
        "{'train','sample','preprocess','fuse','restore','backup'} — the static parse or the "
        "entrypoint surface changed (fuse landed with the Phase-9 inpaint gate; restore/backup "
        "landed with 09.1-08 BK-01)."
    )


def test_documented_modes_exist_on_the_real_entrypoint() -> None:
    """Every ``--mode <value>`` the skills document must EXIST on the real entrypoint (drift guard)."""
    modes, _flags = _entrypoint_cli_surface()
    documented = _documented_mode_values(_combined_skill_text())
    assert documented, (
        "no concrete --mode <value> is documented in any skill — coverage regressed "
        "(the skills should drive --mode preprocess/sample/train)."
    )
    unknown = documented - set(modes)
    assert not unknown, (
        f"skill docs reference --mode value(s) {sorted(unknown)} that do NOT exist on the real "
        f"entrypoint (accepts {sorted(modes)}) — skill<->code drift; an operator would run a "
        "non-existent mode."
    )


def test_real_entrypoint_modes_are_all_documented() -> None:
    """Every real gated ``--mode`` value must be documented by at least one skill (no silent seam)."""
    modes, _flags = _entrypoint_cli_surface()
    documented = _documented_mode_values(_combined_skill_text())
    missing = set(modes) - documented
    assert not missing, (
        f"entrypoint accepts --mode value(s) {sorted(missing)} that NO lifecycle skill documents — "
        "a gated mode must be documented in a skill (skill<->code drift)."
    )


def test_documented_flags_exist_on_the_real_entrypoint() -> None:
    """The ``--config`` / ``--approve`` / ``--mode`` flags the skills drive must exist on main()."""
    _modes, flags = _entrypoint_cli_surface()
    text = _combined_skill_text()
    for flag in ("config", "approve", "mode"):
        assert flag in flags, (
            f"entrypoint main() no longer exposes --{flag} (real flags: {sorted(flags)}), but the "
            "skills document it — skill<->code drift."
        )
        assert f"--{flag}" in text, (
            f"real entrypoint flag --{flag} is documented by NO lifecycle skill — coverage gap."
        )
