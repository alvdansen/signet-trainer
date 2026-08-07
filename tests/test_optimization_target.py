"""Phase 09.2 guardrail suite — the draft/pro optimization-target axis, made PERMANENT.

Every test here converts a decision that is currently true-by-inspection into a property that
FAILS CI when a future edit breaks it. Two halves:

  Task 1 — the ORTHOGONALITY FENCES. The three zero-edit modules (``cost.py`` / ``session_cap.py``
    / ``schema.py``) are the proof that the two axes are orthogonal: today that proof holds only
    because nobody wired the mode through the money path. These tests are what keep it holding
    (D-19). The verdict path (``assert_verdict_spec``) is likewise pinned posture-free (D-24).

  Task 2 — TAXONOMY INTEGRITY. The three-tier partition IS the safety mechanism (D-23). The
    dual-parser contract (D-30), the D-01..D-04 tier membership, the fail-to-``draft`` default
    (D-15), the house verdict spec (D-25/D-29), and KNOWLEDGE retrievability are pinned so a
    silent widening or a silent drift fails CI.

HOUSE TEST RULES (``tests/conftest.py`` / ``tests/test_harness_state.py``): pure-CPU, zero Modal,
zero CUDA, zero network. Tmp-file fixtures + read-only reads of the committed planning artifacts.
This module asserts its OWN import confinement (the house idiom, see
``test_module_imports_nothing_from_modal`` below).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

from signet_trainer import harness_data
from signet_trainer.harness_lint import load_tier_map
from signet_trainer.harness_state import (
    assert_verdict_spec,
    load_campaign_state,
    load_house_spec,
    render_state_block,
)

# ------------------------------------------------------------------------------------ constants

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "signet_trainer"
_MODAL = _SRC / "modal"
# LIVE per-project harness state stays under .planning/harness (never shipped).
_HARNESS = _REPO_ROOT / ".planning" / "harness"
_SKILLS = _REPO_ROOT / ".claude" / "skills"

# SPEC/TEMPLATE data ships as package data — resolve it the way production does, so this suite
# fails if a spec ever stops shipping inside the package.
_TAXONOMY = harness_data.spec_path(harness_data.TIER_TAXONOMY)
_HOUSE_SPEC = harness_data.spec_path(harness_data.HOUSE_SPEC)
_TEMPLATE = harness_data.spec_path(harness_data.SESSION_STATE_TEMPLATE)
_KNOWLEDGE = _HARNESS / "KNOWLEDGE.md"
_HARNESS_STATE_PY = _SRC / "harness_state.py"

# The posture tokens that must NEVER leak into the money path or the verdict path. ``mode`` /
# ``triage_mode`` guard against a posture threaded under another name (AST signature check).
_POSTURE_TOKEN = "optimization_target"


# The three zero-edit fences: module path -> (label, WHY the money/schema modules stay mode-blind).
# The map is in the ``ENTRY_POINT_TOKENS`` teaching-message style — a failure TEACHES the design
# rule, it does not merely assert. If any of these three ever references the posture, the two axes
# are no longer orthogonal and the whole phase's proof is void.
_ZERO_EDIT_MODULES: dict[str, tuple[Path, str]] = {
    "modal/cost.py": (
        _MODAL / "cost.py",
        "pure arithmetic on projected_usd; the axes are orthogonal precisely because the money "
        "modules never read a mode. If you are adding the posture here, the design is wrong — "
        "pro biases VALUES, never the ceiling (D-19).",
    ),
    "modal/session_cap.py": (
        _MODAL / "session_cap.py",
        "pure cap/ledger arithmetic; accounting stays entirely on the triage axis (D-19). pro "
        "hitting the cap is CORRECT behavior — it drops to ask-first (the backstop working), it "
        "never raises session_cap_usd / cost_guardrail_usd.",
    ),
    "config/schema.py": (
        _SRC / "config" / "schema.py",
        "optimization_target is a SESSION-STATE / card field, NEVER a Pydantic field (D-13/D-31). "
        "schema.py::_Base uses extra='forbid' — a render_tier: config key would be rejected too; "
        "the tier is a dispatch argument, not a schema field.",
    ),
}

# The posture-arg names the money modules may NEVER take as a parameter (a mode threaded under any
# of these names is the subtler break the text scan would miss — D-19).
_FORBIDDEN_PARAM_NAMES = frozenset({"optimization_target", "mode", "triage_mode"})


# ============================================================================================
# TASK 1 — the orthogonality fences (D-19, D-24)
# ============================================================================================


@pytest.mark.parametrize("module_key", sorted(_ZERO_EDIT_MODULES))
def test_zero_edit_module_never_references_the_posture(module_key: str) -> None:
    """cost.py / session_cap.py / schema.py contain the token ``optimization_target`` ZERO times.

    Read as TEXT — deliberately NOT imported (schema.py pulls pydantic; the fence must stay cheap
    and pure-CPU). This is the D-19 orthogonality proof made permanent.
    """
    path, why = _ZERO_EDIT_MODULES[module_key]
    src = path.read_text(encoding="utf-8")
    assert _POSTURE_TOKEN not in src, (
        f"{module_key} references '{_POSTURE_TOKEN}' — the D-19 orthogonality fence is BROKEN. {why}"
    )


@pytest.mark.parametrize("module_key", ["modal/cost.py", "modal/session_cap.py"])
def test_money_module_takes_no_posture_parameter(module_key: str) -> None:
    """No FunctionDef in the money modules takes an ``optimization_target`` / ``mode`` / ``triage_mode``
    parameter (AST-checked) — catches a posture threaded through under another name (D-19)."""
    path, why = _ZERO_EDIT_MODULES[module_key]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        arg_names = {
            arg.arg
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)
        }
        for extra in (a.vararg, a.kwarg):
            if extra is not None:
                arg_names.add(extra.arg)
        leaked = arg_names & _FORBIDDEN_PARAM_NAMES
        assert not leaked, (
            f"{module_key}::{node.name}() takes posture parameter(s) {sorted(leaked)} — the money "
            f"path must never receive a mode, even under a different name. {why}"
        )


def _function_source(module_path: Path, func_name: str) -> str:
    """Return the ``ast.unparse``-d source of a single top-level FunctionDef — the SCOPED scan.

    Per the plan's <constraints> warning: harness_state.py legitimately reads the posture elsewhere
    (the campaign-card slot, Plan 02). The D-24 test must therefore scope to the verdict function
    ONLY, never whole-file.
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name),
        None,
    )
    assert node is not None, f"{module_path.name} no longer defines {func_name}() — update this guard."
    return ast.unparse(node)


def test_verdict_path_is_posture_free() -> None:
    """``assert_verdict_spec``'s own source names no posture token (D-24) — AST-scoped, not whole-file.

    The verdict tier is LOCKED and is not a mode lever: no verdict path may read the posture. Scoped
    by AST to the verdict function so harness_state.py's legitimate campaign-card posture read
    (elsewhere in the module) does not defeat the scan. Word-boundary matching so ``pro`` does not
    false-match ``prompts`` / ``prompt count`` inside the function.
    """
    src = _function_source(_HARNESS_STATE_PY, "assert_verdict_spec")
    for token in ("optimization_target", "triage_mode", "draft", "pro"):
        assert not re.search(rf"\b{re.escape(token)}\b", src), (
            f"assert_verdict_spec names the posture token '{token}' — the verdict tier is locked and "
            "is NOT a mode lever (D-24). If the posture is needed in the verdict path, the design is "
            "wrong: a verdict render is asserted against the house spec, never against a mode."
        )


# --- skill-region fences: the posture reaches the direction-only read-site, never the arithmetic --


def _between(text: str, start_prefix: str, end_prefix: str | None = None) -> str:
    """Return the slice of ``text`` from the first line starting with ``start_prefix`` up to (but not
    including) the first later line starting with ``end_prefix`` (or EOF if ``end_prefix`` is None)."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(start_prefix)), None)
    assert start is not None, f"heading {start_prefix!r} not found — the skill structure moved."
    if end_prefix is None:
        return "\n".join(lines[start:])
    end = next(
        (j for j in range(start + 1, len(lines)) if lines[j].startswith(end_prefix)), len(lines)
    )
    return "\n".join(lines[start:end])


def test_training_run_dispatch_arithmetic_is_posture_free_but_3d_reads_it() -> None:
    """training-run §3a-§3c (the dispatch arithmetic) never names the posture; §3d does (D-19).

    Both halves matter: absence alone would ALSO pass if §3d had been deleted. The money call
    (``session_cap_check``) must stay INSIDE the arithmetic region, never in the direction-only §3d.
    """
    text = (_SKILLS / "training-run" / "SKILL.md").read_text(encoding="utf-8")
    arithmetic = _between(text, "### 3a.", "### 3d.")
    section_3d = _between(text, "### 3d.", "## 4.")

    assert _POSTURE_TOKEN not in arithmetic, (
        "training-run §3a-§3c (the BLANKET/STRICT/YOLO dispatch arithmetic) names "
        f"'{_POSTURE_TOKEN}' — the posture must never touch the cost gate / cap (D-19)."
    )
    assert "session_cap_check" in arithmetic, (
        "training-run §3a-§3c no longer calls session_cap_check — the money call must live in the "
        "arithmetic region (its presence here is half the D-19 fence)."
    )
    assert _POSTURE_TOKEN in section_3d, (
        "training-run §3d no longer reads optimization_target — the direction-only read-site is gone; "
        "absence in the arithmetic would then pass VACUOUSLY (T-09.2-31)."
    )
    assert "session_cap_check" not in section_3d, (
        "training-run §3d (direction only) names session_cap_check — the money call leaked out of the "
        "arithmetic into the posture read-site (D-19)."
    )


def test_training_review_verdict_sections_are_posture_free() -> None:
    """The review verdict/disclosure sections never READ the posture (D-24).

    §3 CONVERGENCE PROBE and §12 SINGLE-STAGE DISCLOSURE are fully posture-free (token absent). §11
    RENDER TIERS mentions the token ONLY to DECLARE it is never read ("No verdict path reads
    optimization_target") — a positive statement of the D-24 invariant, never a lever. Pinning §11
    that way matches the landed surface (Plan 06) rather than forcing a whole-token absence that the
    rule-declaration line would break.
    """
    text = (_SKILLS / "training-review" / "SKILL.md").read_text(encoding="utf-8")

    convergence = _between(text, "## 3.", "## 4.")
    disclosure = _between(text, "## 12.", None)
    assert _POSTURE_TOKEN not in convergence, (
        "training-review §3 CONVERGENCE PROBE names the posture — the ~200-step verdict probe is "
        "locked at the house spec and reads no mode (D-24/D-25)."
    )
    assert _POSTURE_TOKEN not in disclosure, (
        "training-review §12 SINGLE-STAGE DISCLOSURE names the posture — the disclosure is "
        "posture-independent (D-26); it is not a draft/pro lever."
    )

    render_tiers = _between(text, "## 11.", "## 12.")
    posture_lines = [ln for ln in render_tiers.splitlines() if _POSTURE_TOKEN in ln]
    assert posture_lines, (
        "training-review §11 RENDER TIERS no longer STATES the 'No verdict path reads "
        "optimization_target' rule (D-24) — the invariant declaration is gone."
    )
    assert all("no verdict path reads" in ln.lower() for ln in posture_lines), (
        "training-review §11 mentions optimization_target OUTSIDE the 'No verdict path reads' "
        "declaration — the verdict tier must only declare the posture OUT, never use it (D-24)."
    )


def test_this_module_imports_nothing_from_modal() -> None:
    """House idiom: this guardrail suite is pure-CPU and must not import the modal SDK."""
    src = Path(__file__).read_text(encoding="utf-8")
    for lineno, line in enumerate(src.splitlines(), start=1):
        assert not re.match(r"\s*(import\s+modal\b|from\s+modal\b)", line), (
            f"line {lineno} imports modal — the guardrail suite must stay pure-CPU."
        )


# ============================================================================================
# TASK 2 — taxonomy integrity, tier membership, the fail-safe (D-03..D-07, D-15, D-18, D-25, D-30)
# ============================================================================================

# The D-01..D-04 tier lists, pinned as literal frozensets. This is DELIBERATE DUPLICATION: the
# taxonomy YAML is the runtime source, and these frozensets are the REVIEW GATE proving a change to
# it was intentional. Changing either set requires a DECISION (a new D-NN), never a "fix" — a silent
# edit here to match a widened taxonomy defeats the gate. Mirrors CONTEXT D-01..D-04.
_TIER_1_KNOBS = frozenset(
    {
        "checkpoint_every",
        "validation.in_loop_sampling",
        "validation.num_samples",
        "sampling_plan.venue",
        "data.num_dataloader_workers",
        "training.gradient_checkpointing",
        "quicklook_render_fidelity",
        "gradient_accumulation_steps",  # D-02: Tier 1, NOT Tier 2 (operator ML call, overrides SCOPE §6 Q5)
    }
)
_TIER_2_KNOBS = frozenset(
    {
        "training_dims",
        "data.resolution_buckets",
        "resolution_buckets.frame_count",
        "training.max_steps",
        "lora.rank",
        "offload.blocks_to_swap",  # D-04: FULL Tier-2 carve-out, overriding SCOPE §3's "soft"
    }
)


def _taxonomy_data() -> dict:
    return yaml.safe_load(_TAXONOMY.read_text(encoding="utf-8"))


def _yaml_tier_map(data: dict) -> dict[str, int]:
    return {knob: entry["tier"] for knob, entry in data["knobs"].items()}


def test_dual_parser_contract_yaml_and_regex_agree() -> None:
    """``harness_lint.load_tier_map`` (stdlib regex) and ``yaml.safe_load`` extract the SAME knob->tier
    map for the real committed taxonomy (D-30 single-source, the dual-parser contract).

    Plan 01 constrained the taxonomy to a one-line flow-mapping shape precisely so both parsers
    agree. If they ever diverge, a knob could be Tier-2 to the emitter and INVISIBLE to the lint — a
    silent safety hole where a learning-affecting knob escapes the mode-pick hard-fail.
    """
    regex_map = load_tier_map(_TAXONOMY)
    yaml_map = _yaml_tier_map(_taxonomy_data())
    assert regex_map == yaml_map, (
        "the stdlib-regex parser (harness_lint.load_tier_map) and yaml.safe_load DISAGREE on "
        "TIER-TAXONOMY.yaml — the dual-parser contract (D-30) is broken. A knob visible to one "
        "parser and not the other is a silent safety hole. Diff: "
        f"regex-only={sorted(set(regex_map) - set(yaml_map))}, "
        f"yaml-only={sorted(set(yaml_map) - set(regex_map))}, "
        f"tier-mismatch={sorted(k for k in regex_map.keys() & yaml_map.keys() if regex_map[k] != yaml_map[k])}"
    )


def test_tier1_membership_equals_the_D01_D02_list() -> None:
    """The Tier-1 knob set equals exactly the D-01+D-02 list — a silent widening fails CI (T-09.2-32)."""
    tier1 = {knob for knob, tier in _yaml_tier_map(_taxonomy_data()).items() if tier == 1}
    assert tier1 == set(_TIER_1_KNOBS), (
        "Tier-1 membership drifted from CONTEXT D-01/D-02. Widening Tier 1 grants draft auto-pick "
        "authority over a knob with no review signal. This requires a DECISION (a new D-NN), not a "
        f"fix. added={sorted(tier1 - set(_TIER_1_KNOBS))}, removed={sorted(set(_TIER_1_KNOBS) - tier1)}"
    )


def test_tier2_membership_equals_the_D03_D04_list() -> None:
    """The Tier-2 carve-out set equals exactly the D-03+D-04 list, incl. ``offload.blocks_to_swap``
    (D-04 made it a FULL carve-out) — a silent narrowing/widening fails CI (T-09.2-32)."""
    tier2 = {knob for knob, tier in _yaml_tier_map(_taxonomy_data()).items() if tier == 2}
    assert tier2 == set(_TIER_2_KNOBS), (
        "Tier-2 (always-check-in carve-out) membership drifted from CONTEXT D-03/D-04. Moving a knob "
        "OUT of Tier 2 lets the mode auto-touch a learning-adjacent knob. Requires a DECISION, not a "
        f"fix. added={sorted(tier2 - set(_TIER_2_KNOBS))}, removed={sorted(set(_TIER_2_KNOBS) - tier2)}"
    )


def test_gradient_accumulation_steps_is_tier1() -> None:
    """``gradient_accumulation_steps`` is Tier 1 (D-02's override of SCOPE §6 Q5) — pinned explicitly
    so a future reader cannot 'correct' it back to Tier 2."""
    assert _yaml_tier_map(_taxonomy_data())["gradient_accumulation_steps"] == 1, (
        "gradient_accumulation_steps is not Tier 1 — D-02 (the operator's ML domain call) makes it auto; do "
        "NOT revert it to Tier 2."
    )


def test_target_modules_is_the_only_fenced_knob_and_tier3() -> None:
    """``target_modules`` carries ``fenced: true`` and is Tier 3 (D-07); it is the ONLY fenced knob."""
    knobs = _taxonomy_data()["knobs"]
    tm = knobs["target_modules"]
    assert tm.get("fenced") is True and tm["tier"] == 3, (
        "target_modules must be fenced:true AND Tier 3 (D-07 hard fence — dropping ff.net is the "
        "documented scaffold bug that corrupted the prior project's likeness)."
    )
    fenced = sorted(k for k, v in knobs.items() if v.get("fenced"))
    assert fenced == ["target_modules"], (
        f"exactly one knob may be hard-fenced (D-07); found {fenced}."
    )


def test_default_tier_is_three_failsafe() -> None:
    """``default_tier == 3`` (D-06): a new/absent knob is Tier 3 (ask), never Tier 1 (auto)."""
    assert _taxonomy_data()["default_tier"] == 3, (
        "default_tier must be 3 (D-06 fail-safe direction) — an absent knob defaulting to Tier 1 is a "
        "safety hole, not a convenience."
    )


def test_keep_checkpoints_and_two_stage_are_not_mode_levers() -> None:
    """``keep_checkpoints`` and ``validation.two_stage_upscale`` are ABSENT from ``knobs`` and PRESENT
    in ``not_a_mode_lever`` (D-08, D-26) — no tier is inferable for them."""
    data = _taxonomy_data()
    knobs = data["knobs"]
    not_lever_keys: set[str] = set()
    for item in data["not_a_mode_lever"]:
        not_lever_keys.update(item.keys())
    for off_axis, decision in (("keep_checkpoints", "D-08"), ("validation.two_stage_upscale", "D-26")):
        assert off_axis not in knobs, (
            f"{off_axis} appears under 'knobs' — it is off-axis ({decision}); a tier must not be "
            "inferable for it."
        )
        assert off_axis in not_lever_keys, (
            f"{off_axis} is missing from not_a_mode_lever ({decision})."
        )


def test_every_knob_carries_an_allowed_source_tag_and_no_prior() -> None:
    """Every knob entry carries a source tag from the four allowed classes; none carries ``[prior]``
    (D-8-SOURCETAG — an untagged model prior is the drift failure mode)."""
    allowed = ("[house]", "[precedent]", "[canonical]", "[community]")
    for knob, entry in _taxonomy_data()["knobs"].items():
        tags = entry.get("source_tags", "")
        assert "[prior]" not in tags, f"knob {knob} carries a banned [prior] tag: {tags!r}"
        assert any(tag in tags for tag in allowed), (
            f"knob {knob} carries no allowed source tag (one of {allowed}); got {tags!r}"
        )


def test_house_verdict_spec_carries_the_locked_params() -> None:
    """``HOUSE-SPEC.yaml``'s ``verdict_spec`` equals 81 / 40 / 4.0 / 42 / two_stage false, and
    ``min_prompt_count >= 2`` (D-25 floor)."""
    spec = load_house_spec(_HOUSE_SPEC)["verdict_spec"]
    assert spec["frame_count"] == 81
    assert spec["num_inference_steps"] == 40
    assert spec["guidance_scale"] == pytest.approx(4.0)
    assert spec["seed"] == 42
    assert spec["two_stage_upscale"] is False
    assert spec["min_prompt_count"] >= 2, (
        "min_prompt_count must be >= 2 (D-25) — one prompt is never a valid verdict basis."
    )


def test_house_config_satisfies_the_verdict_spec() -> None:
    """``configs/campaign_r5.example.yaml``'s ``validation`` block passes the verdict assertion — the house spec
    matches the real house config (a trimmed house config would fail loudly at verdict dispatch)."""
    cfg = yaml.safe_load((_REPO_ROOT / "configs" / "campaign_r5.example.yaml").read_text(encoding="utf-8"))
    validation = cfg["validation"]
    house_spec = load_house_spec(_HOUSE_SPEC)
    # Raises ValueError on any trimmed param; passing == the house config IS a valid verdict render.
    assert_verdict_spec(validation, "verdict", house_spec)


def test_unmigrated_card_fails_safe_to_draft(tmp_path) -> None:
    """A tmp card lacking ``campaign.optimization_target`` renders ``optimization_target: draft``
    (D-15 fail-safe) without raising — the un-migrated-card property CONTEXT deferred the campaign on."""
    card = tmp_path / "TRAINING-CARD-test.state.yaml"
    card.write_text(
        yaml.safe_dump(
            {
                "campaign": {"slug": "test-campaign", "phase": "test"},  # NO optimization_target
                "rounds": [
                    {
                        "round": "r1",
                        "config": "test.yaml",
                        "final_checkpoint": "outputs/test/checkpoint-step-00100",
                        "steps": 100,
                        "final_loss": 0.1,
                        "verdict": "PASS",
                    }
                ],
                "current": {"note": "in-flight"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    data = load_campaign_state(card)
    ledger = {
        "session_cap_usd": None,
        "spent_est_usd": 0.0,
        "entries": 0,
        "source": "tmp/SESSION-STATE.json",
    }
    block = render_state_block({"campaign": data, "ledger": ledger})
    assert re.search(r"^optimization_target:\s*draft\s*$", block, flags=re.MULTILINE), (
        "an un-migrated card (no campaign.optimization_target) must render 'optimization_target: "
        "draft' (D-15 fail-safe to least spend), never raise and never omit the slot."
    )


def test_session_state_template_defaults_to_draft() -> None:
    """``SESSION-STATE.template.json``'s ``optimization_target`` is ``"draft"`` (D-15)."""
    template = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    assert template["optimization_target"] == "draft", (
        "the SESSION-STATE template must default optimization_target to 'draft' (D-15 fail-safe)."
    )


# --- KNOWLEDGE retrievability (D-30 part 2 is useless if unfindable) ---


def _knowledge_text() -> str:
    return _KNOWLEDGE.read_text(encoding="utf-8")


def _sections_containing(text: str, keyword: str) -> str:
    """Return ALL ``#``-headed section blocks that contain ``keyword`` (case-insensitive), joined —
    mirrors ``test_memory_scaffold._section_containing`` so a bare keyword that also appears in the
    preamble index still reaches its real rule section."""
    kw = keyword.lower()
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return "\n".join(b for b in blocks if kw in b.lower())


@pytest.mark.parametrize("keyword", ["draft", "pro", "tier", "carve-out", "optimization_target"])
@pytest.mark.requires_live_harness
def test_tier_partition_is_keyword_retrievable(keyword: str) -> None:
    """KNOWLEDGE.md is retrievable by the bare keywords ``draft`` / ``pro`` / ``tier`` / ``carve-out``
    / ``optimization_target`` — each reaching the tier-partition section (D-30 part 2).

    ``pro`` is a substring of ``process`` / ``probe`` / ``prompt``, so section-membership (not a bare
    non-empty hit) is asserted: the tier-partition section (its 'three-tier law' marker) must be among
    the sections the keyword surfaces.
    """
    section = _sections_containing(_knowledge_text(), keyword)
    assert "three-tier law" in section.lower(), (
        f"keyword {keyword!r} does not reach the knob->tier partition section in KNOWLEDGE.md — the "
        "emitted tier table (D-30 part 2) is not retrievable, defeating retrieval-before-improvising."
    )


def test_training_prep_resolution_checkin_is_posture_independent() -> None:
    """``training-prep/SKILL.md`` states the resolution check-in is posture-independent / all four
    quadrants (D-18).

    Honest scope note: this is a DOC-SCAN guardrail, because the posture layer is agent-instruction
    PROSE (PATTERNS F-1) — there is no policy function to call. The genuine behavioral seam for the
    axis is the ``harness_lint`` mode-pick rule (tested in test_memory_scaffold.py, Plan 03).
    """
    text = (_SKILLS / "training-prep" / "SKILL.md").read_text(encoding="utf-8")
    assert re.search(r"resolution check-in is\s+POSTURE-INDEPENDENT", text, flags=re.IGNORECASE), (
        "training-prep no longer states the resolution check-in is POSTURE-INDEPENDENT (D-18) — the "
        "always-check-in must fire in all four quadrants, incl. pro+yolo (a VRAM/OOM safety gate, not "
        "a cost-direction lever)."
    )
    assert re.search(r"all four quadrants", text, flags=re.IGNORECASE), (
        "training-prep no longer names 'all four quadrants' for the resolution check-in (D-18)."
    )


def test_no_config_declares_a_render_tier_key() -> None:
    """No file under ``configs/`` contains a ``render_tier:`` key (D-31 / schema ``extra='forbid'``) —
    the tier is a dispatch argument, never a config key."""
    offenders = [
        p.name
        for p in sorted((_REPO_ROOT / "configs").glob("*.yaml"))
        if re.search(r"^\s*render_tier\s*:", p.read_text(encoding="utf-8"), flags=re.MULTILINE)
    ]
    assert not offenders, (
        f"config file(s) {offenders} declare a 'render_tier:' key — it is a DISPATCH ARGUMENT, not a "
        "schema field (D-31); schema.py::_Base extra='forbid' would reject it at config load."
    )
