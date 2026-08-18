"""Decision-memory scaffold tests (D-8-MEMORY). Pure CPU — NO modal, NO GPU.

Locks the forcing functions of the sans-Hermes two-layer memory scaffold:

1. ``KNOWLEDGE.md`` (semantic layer) is grep-retrievable — a bare keyword lands on the seeded
   house rule / landmine (retrieval-before-improvising).
2. ``DECISION-LOG.md`` (episodic layer) append is well-formed — every entry carries the four
   required fields incl. ``source-tags:`` (the D-8-SOURCETAG write-gate forcing function), and a
   missing ``source-tags:`` is detectable.
3. Every seeded ``KNOWLEDGE.md`` knowledge item that makes a recommendation carries at least one
   bracketed source tag (structural D-8-SOURCETAG guard against untagged model priors).
4. TS-01c typed-state lint (typed-state-beats-prose, adopted 2026-07-14 — see
   ``.planning/harness/COMPACTION-PROTOCOL.md``): every dated DECISION-LOG entry AFTER the
   adoption entry whose prose carries load-bearing numerics ($-figures, step counts, loss
   values, checkpoint-step paths) MUST carry a ``- state:`` block of simple indented
   ``name: value`` lines. Forward-only AND entry-ORDER-aware: entries before the adoption
   entry are exempt even when they share the 2026-07-14 date (several do).

The tests read the REAL committed files (seeded in Task 1) and are tolerant of prose wording —
they search for tokens/substrings, not exact sentences.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# TS-01c: the typed-state lint is single-sourced in ``signet_trainer.harness_lint`` (extracted from
# this module so it also runs as a standalone CLI / pre-commit hook). Import it here so these tests
# exercise the extracted module — mirrors how ``tests/test_harness_state.py`` imports
# ``signet_trainer.harness_state``.
from signet_trainer.harness_lint import (
    _ADOPTION_TITLE,
    _TYPED_STATE_CUTOFF,
    entry_class,
    entry_date,
    extract_state_block,
    find_adoption_index,
    has_load_bearing_numerics,
    has_mode_pick_prose,
    lint_typed_state_entry,
    lint_typed_state_log,
    load_tier_map,
    split_log_entries,
    state_slot_names,
    state_slot_values,
    strip_state_block,
)

# Repo root = two levels up from this test file (tests/ -> repo root).
_HARNESS_DIR = Path(__file__).resolve().parent.parent / ".planning" / "harness"
_KNOWLEDGE = _HARNESS_DIR / "KNOWLEDGE.md"
_DECISION_LOG = _HARNESS_DIR / "DECISION-LOG.md"

_SOURCE_TAGS = ("[house]", "[precedent]", "[canonical]", "[community]")
_REQUIRED_LOG_FIELDS = ("what:", "why:", "source-tags:", "run-refs:")


# --------------------------------------------------------------------------- helpers


def keyword_search(text: str, keyword: str) -> list[str]:
    """Return the lines of ``text`` that contain ``keyword`` (case-insensitive) — the grep seam."""
    kw = keyword.lower()
    return [line for line in text.splitlines() if kw in line.lower()]


def is_wellformed_entry(block: str) -> bool:
    """A well-formed log entry has a dated ``## `` heading and all four required fields."""
    lines = block.splitlines()
    if not lines or not lines[0].startswith("## "):
        return False
    return all(any(field in line for line in lines) for field in _REQUIRED_LOG_FIELDS)


# --------------------------------------------------------------------------- files exist


@pytest.mark.requires_live_harness
def test_scaffold_files_exist() -> None:
    assert _KNOWLEDGE.is_file(), f"missing seeded KNOWLEDGE.md at {_KNOWLEDGE}"
    assert _DECISION_LOG.is_file(), f"missing seeded DECISION-LOG.md at {_DECISION_LOG}"


# --------------------------------------------------------------------------- (1) retrieval


@pytest.mark.requires_live_harness
def test_prodigy_keyword_returns_never_line() -> None:
    """Searching ``Prodigy`` must yield a line telling the agent NEVER to use it (house rule)."""
    text = _KNOWLEDGE.read_text(encoding="utf-8")
    hits = keyword_search(text, "Prodigy")
    assert hits, "no line mentions Prodigy — the house 'never Prodigy' rule is not retrievable"
    assert any("never" in line.lower() for line in hits), (
        "Prodigy is mentioned but no line says 'never' — the house rule is not retrievable by keyword"
    )


@pytest.mark.requires_live_harness
def test_block_swap_keyword_returns_scope_limit() -> None:
    """Searching ``block-swap`` must surface the freeze-mechanics / OFFL-02 scope limit."""
    text = _KNOWLEDGE.read_text(encoding="utf-8")
    hits = keyword_search(text, "block-swap")
    assert hits, "no line mentions block-swap — the offloader scope landmine is not retrievable"
    joined = "\n".join(hits).lower()
    # The landmine keyword may sit on an adjacent line within the same section; fall back to section.
    if not ("freeze-mechanics" in joined or "offl-02" in joined):
        section = _section_containing(text, "block-swap").lower()
        assert "freeze-mechanics" in section or "offl-02" in section, (
            "block-swap entry does not carry the freeze-mechanics / OFFL-02 scope limit"
        )


@pytest.mark.requires_live_harness
def test_frames_keyword_returns_alignment_rule() -> None:
    """Searching ``frames`` must surface the (frames-1)%8 LTX alignment rule."""
    text = _KNOWLEDGE.read_text(encoding="utf-8")
    section = _section_containing(text, "frames")
    assert "(frames - 1) % 8" in section or "(frames-1)%8" in section or "(frames - 1)%8" in section, (
        "the (frames-1)%8 LTX frame-alignment rule is not retrievable under the 'frames' keyword"
    )


def _section_containing(text: str, keyword: str) -> str:
    """Return ALL ``#``-headed section blocks that contain ``keyword``, joined.

    Joining (rather than returning the first match) keeps the search robust: a bare keyword can
    appear in the preamble keyword-index AND in its real rule section, and the rule may wrap onto
    a line adjacent to the keyword. Concatenating every matching section captures the rule text.
    """
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
    return "\n".join(block for block in blocks if kw in block.lower())


# --------------------------------------------------------------------------- (2) log append


@pytest.mark.requires_live_harness
def test_seeded_log_entries_are_wellformed() -> None:
    """Every seeded dated entry carries a heading + the four required fields."""
    text = _DECISION_LOG.read_text(encoding="utf-8")
    dated = [e for e in split_log_entries(text) if e.startswith("## 20")]
    assert len(dated) >= 2, "expected at least two dated precedent entries in DECISION-LOG.md"
    for entry in dated:
        assert is_wellformed_entry(entry), f"malformed seeded entry:\n{entry}"


@pytest.mark.requires_live_harness
def test_append_roundtrips_a_wellformed_entry(tmp_path: Path) -> None:
    """A well-formed append round-trips with all four fields on a tmp copy of the log."""
    tmp_log = tmp_path / "DECISION-LOG.md"
    tmp_log.write_text(_DECISION_LOG.read_text(encoding="utf-8"), encoding="utf-8")

    new_entry = (
        "\n## 2026-07-10 - Test append roundtrip\n"
        "- what: sample decision written by the append convention\n"
        "- why: exercise the well-formedness contract\n"
        "- source-tags: [precedent]\n"
        "- run-refs: none\n"
    )
    with tmp_log.open("a", encoding="utf-8") as fh:
        fh.write(new_entry)

    entries = split_log_entries(tmp_log.read_text(encoding="utf-8"))
    appended = entries[-1]
    assert appended.startswith("## 2026-07-10 - Test append roundtrip")
    assert is_wellformed_entry(appended)
    for field in _REQUIRED_LOG_FIELDS:
        assert field in appended


def test_missing_source_tags_is_detectable() -> None:
    """The write-gate forcing function: an entry lacking ``source-tags:`` is NOT well-formed."""
    bad_entry = (
        "## 2026-07-10 - Missing source tags\n"
        "- what: a decision with no source class\n"
        "- why: to prove the forcing function catches it\n"
        "- run-refs: none\n"
    )
    assert not is_wellformed_entry(bad_entry), (
        "an entry missing 'source-tags:' must be detectable as malformed (untagged-prior guard)"
    )


# --------------------------------------------------------------------------- (3) source-tag coverage


@pytest.mark.requires_live_harness
def test_knowledge_recommendation_sections_are_source_tagged() -> None:
    """Every KNOWLEDGE.md section that makes a recommendation carries at least one source tag.

    Structural D-8-SOURCETAG guard: scan per-section (robust to prose formatting) rather than
    per-line. A section is a recommendation section if it contains an imperative/knowledge claim
    (a bullet). Skip the header/preamble and pure cross-reference sections.
    """
    text = _KNOWLEDGE.read_text(encoding="utf-8")

    # Build (heading, body) section blocks.
    sections: list[tuple[str, str]] = []
    heading = "<preamble>"
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            sections.append((heading, "\n".join(body)))
            heading = line
            body = []
        else:
            body.append(line)
    sections.append((heading, "\n".join(body)))

    # Sections that are legitimately tag-free (structure/navigation, not recommendations).
    exempt_substrings = ("cross-references", "keyword index", "search convention")

    checked = 0
    for heading, sect_body in sections:
        low_head = heading.lower()
        if any(sub in low_head for sub in exempt_substrings):
            continue
        # Only assert on sections that actually make a knowledge claim (have a '- ' bullet with prose).
        has_claim = any(l.strip().startswith("- ") and len(l.strip()) > 4 for l in sect_body.splitlines())
        if not has_claim:
            continue
        checked += 1
        assert any(tag in sect_body for tag in _SOURCE_TAGS), (
            f"KNOWLEDGE.md section {heading!r} makes a recommendation but carries no source tag "
            f"(untagged model prior is the documented drift failure mode)"
        )

    assert checked >= 5, "expected several source-tagged recommendation sections; found too few"


# --------------------------------------------------------------------------- (4) TS-01c typed-state lint


@pytest.mark.requires_live_harness
def test_adoption_entry_exists_and_is_its_own_first_exemplar() -> None:
    """The adoption entry is present, dated on the cutoff, and carries a non-empty state block."""
    entries = split_log_entries(_DECISION_LOG.read_text(encoding="utf-8"))
    idx = find_adoption_index(entries)
    assert idx is not None, f"adoption entry ({_ADOPTION_TITLE!r}) not found in DECISION-LOG.md"
    adoption = entries[idx]
    assert entry_date(adoption) == _TYPED_STATE_CUTOFF, (
        "the adoption entry must be dated on the forward-only cutoff (2026-07-14)"
    )
    state = extract_state_block(adoption)
    assert state, "the adoption entry must itself carry a non-empty '- state:' block (first exemplar)"


@pytest.mark.requires_live_harness
def test_post_adoption_entries_pass_typed_state_lint() -> None:
    """The real log lints clean, and the linted set covers the two known post-adoption entries."""
    text = _DECISION_LOG.read_text(encoding="utf-8")
    entries = split_log_entries(text)
    idx = find_adoption_index(entries)
    assert idx is not None
    linted = [
        e for e in entries[idx:]
        if (entry_date(e) or "") >= _TYPED_STATE_CUTOFF
    ]
    assert len(linted) >= 2, "expected at least the ADOPTED + F9 RECURRENCE post-adoption entries"
    headings = [e.splitlines()[0] for e in linted]
    assert any(_ADOPTION_TITLE in h for h in headings)
    assert any("F9 RECURRENCE" in h for h in headings), (
        "the F9 RECURRENCE entry should be in the linted (post-adoption) set"
    )
    violations = lint_typed_state_log(text)
    assert violations == [], "typed-state lint violations in the REAL log:\n" + "\n".join(violations)


@pytest.mark.requires_live_harness
def test_pre_adoption_entries_are_exempt_by_entry_order() -> None:
    """The cutoff is entry-ORDER aware, not just date-aware.

    Several entries dated 2026-07-14 (r5 verdict, precedent grounding, audit, cap raise, dispatch,
    burned gate, build) PRECEDE the adoption entry, carry load-bearing numerics in prose, and
    have NO state block — a date-only cutoff would flag them. They must be exempt, and the
    full-log lint must still report zero violations.
    """
    text = _DECISION_LOG.read_text(encoding="utf-8")
    entries = split_log_entries(text)
    idx = find_adoption_index(entries)
    assert idx is not None
    pre = [e for e in entries[:idx] if e.startswith("## 20")]
    would_violate = [
        e for e in pre
        if has_load_bearing_numerics(strip_state_block(e)) and extract_state_block(e) is None
    ]
    assert would_violate, (
        "expected pre-adoption entries with load-bearing numerics and no state block "
        "(the forward-only cutoff must be doing real work)"
    )
    assert any((entry_date(e) or "") >= _TYPED_STATE_CUTOFF for e in would_violate), (
        "expected pre-adoption entries ON the cutoff date itself — proves the cutoff is "
        "entry-order-aware, not merely date-aware"
    )
    assert lint_typed_state_log(text) == [], (
        "pre-adoption entries leaked into the lint (cutoff not order-aware)"
    )


# ------------------------------------------------ (4b) issue #37 finding 6: absent-adoption-entry
# handling — a downstream project's own DECISION-LOG must never be permanently red with a message
# naming no fix.


def test_absent_adoption_entry_with_nothing_post_cutoff_lints_clean() -> None:
    """No adoption entry AND nothing dated on/after the cutoff -> [] ('nothing post-cutoff to
    lint'), not the old single pseudo-violation that short-circuited every real rule."""
    log = (
        "## 2026-06-01 - Some pre-cutoff entry\n"
        "- what: routine note, no adoption title anywhere in this log.\n"
        "- why: prove an absent title alone is not fatal.\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )
    assert find_adoption_index(split_log_entries(log)) is None, "fixture must carry no adoption entry"
    assert lint_typed_state_log(log) == []


def test_absent_adoption_entry_still_catches_real_post_cutoff_violations() -> None:
    """No adoption entry, but a post-cutoff entry HAS load-bearing numerics with no state block ->
    the violation still reports (issue #37 finding 6: the date-based cutoff runs independently of
    the title anchor, so a missing title can never silently exempt the whole log)."""
    log = (
        "## 2026-07-20 - Synthetic: numerics with no adoption entry in this log at all\n"
        "- what: the run cost $12.34 at step 3000.\n"
        "- why: prove the absent-title path still catches real numerics.\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )
    violations = lint_typed_state_log(log)
    assert any("no '- state:' typed block" in v for v in violations), (
        f"the real violation must still be reported: {violations}"
    )
    assert any(_ADOPTION_TITLE in v and "NOTE" in v for v in violations), (
        f"an informational note naming the missing adoption entry must accompany real "
        f"violations: {violations}"
    )


@pytest.mark.requires_live_harness
def test_lint_catches_numerics_without_state_block(tmp_path: Path) -> None:
    """A synthetic violating entry FAILS the lint — proven on a tmp copy, never the real log."""
    tmp_log = tmp_path / "DECISION-LOG.md"
    tmp_log.write_text(_DECISION_LOG.read_text(encoding="utf-8"), encoding="utf-8")

    bad_entry = (
        "\n## 2026-07-15 - Synthetic: load-bearing numerics only in prose\n"
        "- what: the run cost $12.34, reached step 3000 with 8000 steps planned, settled at\n"
        "  loss 0.2156; final = checkpoint-step-03000-loss-0.2156. No typed slots anywhere.\n"
        "- why: prove the TS-01c lint bites on a paraphrase-prone entry.\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )
    with tmp_log.open("a", encoding="utf-8") as fh:
        fh.write(bad_entry)

    violations = lint_typed_state_log(tmp_log.read_text(encoding="utf-8"))
    assert len(violations) == 1, f"expected exactly the synthetic violation, got: {violations}"
    assert "Synthetic: load-bearing numerics only in prose" in violations[0]
    assert "no '- state:' typed block" in violations[0]


@pytest.mark.requires_live_harness
def test_lint_rejects_prose_sentences_and_empty_state_blocks(tmp_path: Path) -> None:
    """A state block must be simple ``name: value`` lines — prose inside it, or an empty block, fails."""
    tmp_log = tmp_path / "DECISION-LOG.md"
    tmp_log.write_text(_DECISION_LOG.read_text(encoding="utf-8"), encoding="utf-8")

    prose_in_block = (
        "\n## 2026-07-15 - Synthetic: prose sentence inside the state block\n"
        "- what: a run summary whose author narrated inside the typed block.\n"
        "- why: prove the name:value parse check bites.\n"
        "- state:\n"
        "    ledger_spent_usd: 280.10\n"
        "    the run went really well overall and cost a fair amount of money\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )
    empty_block = (
        "\n## 2026-07-15 - Synthetic: empty state block over numeric prose\n"
        "- what: cost $45.00 at step 5000 but the state block was left empty.\n"
        "- why: prove present-but-empty is insufficient.\n"
        "- state:\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )
    with tmp_log.open("a", encoding="utf-8") as fh:
        fh.write(prose_in_block)
        fh.write(empty_block)

    violations = lint_typed_state_log(tmp_log.read_text(encoding="utf-8"))
    assert any(
        "prose sentence inside the state block" in v and "not a simple indented 'name: value'" in v
        for v in violations
    ), f"prose-in-block synthetic not caught: {violations}"
    assert any(
        "empty state block over numeric prose" in v and "no 'name: value' lines" in v
        for v in violations
    ), f"empty-block synthetic not caught: {violations}"
    # And the real (untouched) entries above the synthetics still lint clean.
    assert all("Synthetic" in v for v in violations), f"real entries flagged unexpectedly: {violations}"


# ------------------------------------------------ (5) D-TS-2 mandatory-slot cores per entry class


def _run_class_entry(missing: str | None = None) -> str:
    """A synthetic post-cutoff ``run``-class entry; optionally omit one mandatory core slot."""
    slots = {
        "class": "run",
        "round": "r7",
        "config": "configs/campaign_r7.yaml",
        "init_adapter": "outputs/campaign_r6/checkpoint-step-03000-loss-0.1200",
        "final_checkpoint": "outputs/campaign_r7/checkpoint-step-03000-loss-0.1100",
        "steps": "3000",
        "final_loss": "0.1100",
    }
    if missing is not None:
        slots.pop(missing)
    state_lines = "\n".join(f"    {k}: {v}" for k, v in slots.items())
    return (
        "## 2026-07-15 - Synthetic: run-class entry\n"
        "- what: a warm-start round result recorded as a run-class entry.\n"
        "- why: exercise the D-TS-2 mandatory-core check.\n"
        "- state:\n"
        f"{state_lines}\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )


def test_run_class_entry_missing_mandatory_slot_violates() -> None:
    """A ``run``-class entry missing a mandatory core slot (final_loss) is a D-TS-2 violation."""
    violations = lint_typed_state_entry(_run_class_entry(missing="final_loss"))
    assert any("mandatory state slot" in v and "final_loss" in v for v in violations), (
        f"missing 'final_loss' on a run-class entry must violate D-TS-2: {violations}"
    )


def test_complete_run_class_entry_passes() -> None:
    """A complete ``run``-class entry (all six core slots present) produces no violation."""
    assert lint_typed_state_entry(_run_class_entry()) == []


def test_verdict_and_incident_class_cores_enforced() -> None:
    """The ``verdict`` and ``incident`` class cores are each enforced (missing key -> violation)."""
    verdict_missing = (
        "## 2026-07-15 - Synthetic: verdict-class missing checkpoint_judged\n"
        "- what: a review verdict recorded without the judged checkpoint slot.\n"
        "- why: exercise the verdict-class core.\n"
        "- state:\n"
        "    class: verdict\n"
        "    round: r7\n"
        "    verdict: CONTINUE\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )
    incident_missing = (
        "## 2026-07-15 - Synthetic: incident-class missing detected_via\n"
        "- what: an incident recorded without how it was detected.\n"
        "- why: exercise the incident-class core.\n"
        "- state:\n"
        "    class: incident\n"
        "    impact: watcher died mid-render\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )
    assert any(
        "mandatory state slot" in v and "checkpoint_judged" in v
        for v in lint_typed_state_entry(verdict_missing)
    )
    assert any(
        "mandatory state slot" in v and "detected_via" in v
        for v in lint_typed_state_entry(incident_missing)
    )


def test_untagged_entry_with_state_block_is_not_core_constrained() -> None:
    """An entry with a state block but NO ``class:`` slot keeps ONLY the governing rule.

    ``entry_class`` returns None, so the mandatory-core check never fires — forward-only, legacy
    entries are not over-constrained (D-TS-5).
    """
    untagged = (
        "## 2026-07-15 - Synthetic: untagged entry with a typed block\n"
        "- what: a decision at step 3000 with a partial typed block and no class tag.\n"
        "- why: prove the mandatory core does not apply to untagged entries.\n"
        "- state:\n"
        "    steps: 3000\n"
        "    note_ref: some-run\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )
    state_lines = extract_state_block(untagged)
    assert state_lines is not None
    assert entry_class(state_lines) is None, "an entry with no 'class:' slot has no declared class"
    # Governing rule is satisfied (a state block exists over the numerics), and no core check fires.
    assert lint_typed_state_entry(untagged) == []


def test_state_slot_names_extracts_keys() -> None:
    """``state_slot_names`` returns the bare key tokens from indented ``name: value`` lines."""
    names = state_slot_names(["    class: run", "    steps: 3000", "    final_loss: 0.11"])
    assert names == {"class", "steps", "final_loss"}


# ------------------------------------------------ (6) D-30 mode-pick tier enforcement (Plan 09.2-03)


def _mode_pick_entry(
    knob: str = "checkpoint_every",
    tier: str = "1",
    source_tag: str = "[house]",
    chosen: str = "every 1000 steps (coarser dev cadence)",
    optimization_target: str = "draft",
    omit: str | None = None,
) -> str:
    """A synthetic post-cutoff ``mode-pick`` entry; optionally omit one mandatory slot."""
    slots = {
        "class": "mode-pick",
        "knob": knob,
        "tier": tier,
        "chosen": chosen,
        "source_tag": source_tag,
        "optimization_target": optimization_target,
    }
    if omit is not None:
        slots.pop(omit)
    state_lines = "\n".join(f"    {k}: {v}" for k, v in slots.items())
    return (
        "## 2026-07-16 - Synthetic: mode-pick entry\n"
        "- what: a posture-driven pick recorded as a mode-pick entry.\n"
        "- why: exercise the D-30 Tier-2/3 enforcement.\n"
        "- state:\n"
        f"{state_lines}\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )


def test_load_tier_map_reads_taxonomy_by_regex() -> None:
    """``load_tier_map`` returns a non-empty {knob: tier} map from the real committed taxonomy."""
    tier_map = load_tier_map()
    assert tier_map, "tier map is empty — the stdlib regex parsed zero knobs"
    assert tier_map["checkpoint_every"] == 1
    assert tier_map["training_dims"] == 2
    assert tier_map["lr"] == 3
    assert tier_map["target_modules"] == 3
    # The excluded not_a_mode_lever entries must NOT leak in as knobs.
    assert "keep_checkpoints" not in tier_map


def test_load_tier_map_fails_loud_on_missing_and_empty(tmp_path: Path) -> None:
    """A missing taxonomy raises ``FileNotFoundError``; a zero-knob parse raises ``ValueError`` (D-23)."""
    with pytest.raises(FileNotFoundError):
        load_tier_map(tmp_path / "absent.yaml")
    empty = tmp_path / "empty.yaml"
    empty.write_text("version: 1\ndefault_tier: 3\nknobs:\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tier_map(empty)


def test_load_tier_map_fails_loud_on_partial_divergence(tmp_path: Path) -> None:
    """WR-02: a knob the one-line regex DROPS (block-style, or a hyphenated name) makes load_tier_map
    fail LOUD on the parsed-vs-declared count mismatch — not only the ZERO-knob case.

    Without the divergence canary the dropped knob is invisible to the lint's Tier-2/3 safety map
    (silently desyncing it from the emitted taxonomy), and a legitimately Tier-1 knob hard-fails as
    'absent'. The count reconciliation catches PARTIAL divergence at runtime.
    """
    bad = tmp_path / "TIER-TAXONOMY.yaml"
    bad.write_text(
        "version: 1\n"
        "default_tier: 3\n"
        "knobs:\n"
        '  checkpoint_every: {tier: 1, source_tags: "[house]"}\n'  # regex parses this one
        "  block_style_knob:\n"  # block-style — the one-line regex silently drops it
        "    tier: 2\n"
        '    source_tags: "[house]"\n'
        '  hyphen-knob: {tier: 1, source_tags: "[house]"}\n',  # hyphen — regex class misses it
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="declared under 'knobs:'"):
        load_tier_map(bad)


def test_load_tier_map_accepts_all_one_line_knobs(tmp_path: Path) -> None:
    """WR-02 canary must NOT false-positive: an all-one-line taxonomy parses with no count mismatch,
    including comment/blank lines interleaved under ``knobs:``."""
    good = tmp_path / "TIER-TAXONOMY.yaml"
    good.write_text(
        "version: 1\n"
        "default_tier: 3\n"
        "knobs:\n"
        "  # Tier 1 group\n"
        '  checkpoint_every: {tier: 1, source_tags: "[house]"}\n'
        "\n"
        '  lora.rank: {tier: 2, source_tags: "[house]"}\n'
        '  lr: {tier: 3, source_tags: "[house]"}\n'
        "not_a_mode_lever:\n"
        "  - keep_checkpoints: never a mode lever\n",
        encoding="utf-8",
    )
    tier_map = load_tier_map(good)
    assert tier_map == {"checkpoint_every": 1, "lora.rank": 2, "lr": 3}


def test_mode_pick_tier1_knob_passes() -> None:
    """A well-formed Tier-1 mode-pick (all five slots, house-tagged option) produces no violation."""
    assert lint_typed_state_entry(_mode_pick_entry()) == []


def test_mode_pick_missing_slot_violates() -> None:
    """A mode-pick missing a mandatory core slot names the missing slot (D-20 shape)."""
    violations = lint_typed_state_entry(_mode_pick_entry(omit="chosen"))
    assert any("mandatory state slot" in v and "chosen" in v for v in violations)


def test_mode_pick_tier2_knob_hard_fails() -> None:
    """draft auto-setting a Tier-2 knob (training_dims) HARD-FAILS naming the knob + tier 2 (D-30)."""
    violations = lint_typed_state_entry(_mode_pick_entry(knob="training_dims", tier="1"))
    assert any(
        "training_dims" in v and "tier 2" in v.lower() for v in violations
    ), f"Tier-2 mode-pick must hard-fail naming training_dims + tier 2: {violations}"


def test_mode_pick_tier3_knob_hard_fails() -> None:
    """A mode-pick on a Tier-3 knob (lr) HARD-FAILS naming tier 3 (house-first, D-05)."""
    violations = lint_typed_state_entry(_mode_pick_entry(knob="lr", tier="3"))
    assert any("lr" in v and "tier 3" in v.lower() for v in violations)


def test_mode_pick_unknown_knob_fails_safe_to_tier3() -> None:
    """A mode-pick on a knob ABSENT from the taxonomy HARD-FAILS (D-06 fail-safe to Tier 3)."""
    violations = lint_typed_state_entry(_mode_pick_entry(knob="some_new_knob", tier="1"))
    assert any("some_new_knob" in v and "absent" in v.lower() for v in violations)


def test_mode_pick_tier_disagreement_hard_fails() -> None:
    """An entry whose tier slot disagrees with the taxonomy names both values (no self-declared tier)."""
    # checkpoint_every is really Tier 1; declaring tier 2 is a disagreement.
    violations = lint_typed_state_entry(_mode_pick_entry(knob="checkpoint_every", tier="2"))
    assert any(
        "checkpoint_every" in v and "'2'" in v and "tier 1" in v.lower() for v in violations
    ), f"tier disagreement must name both values: {violations}"


def test_mode_pick_prior_source_tag_hard_fails() -> None:
    """A mode-pick whose source_tag is [prior] HARD-FAILS — the mode is never itself a tag (D-20)."""
    violations = lint_typed_state_entry(_mode_pick_entry(source_tag="[prior]"))
    assert any("source_tag" in v and "[prior]" in v for v in violations)


def test_mode_pick_mode_word_source_tag_hard_fails() -> None:
    """source_tag 'draft' (the mode as a tag) is rejected — it names the option's class, not the mode."""
    violations = lint_typed_state_entry(_mode_pick_entry(source_tag="draft"))
    assert any("source_tag" in v for v in violations)


def test_mode_pick_prose_without_state_block_hard_fails() -> None:
    """A post-cutoff entry whose prose says draft auto-selected a knob, no state block, HARD-FAILS."""
    entry = (
        "## 2026-07-16 - Synthetic: prose-only posture pick\n"
        "- what: under yolo, draft auto-selected a coarser checkpoint cadence for the dev pass.\n"
        "- why: prove the omission catcher bites when no typed block is present.\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n"
    )
    violations = lint_typed_state_entry(entry)
    assert any("no '- state:'" in v for v in violations), (
        f"prose-only posture pick must hard-fail: {violations}"
    )


def test_has_mode_pick_prose_is_tight_on_pro_substrings() -> None:
    """The 'pro' pattern must NOT fire on process/probe/proven/prompt (DoS guard, T-09.2-13)."""
    assert not has_mode_pick_prose("the process proved the probe prompt was fine")
    assert has_mode_pick_prose("draft picked the coarser cadence")
    # WR-03: a bare MENTION of the field name is narrative about the axis, not a posture-driven
    # pick — it no longer trips the hard-fail omission catcher.
    assert not has_mode_pick_prose("the optimization_target axis landed")


def test_has_mode_pick_prose_still_catches_recorded_optimization_target_pick() -> None:
    """WR-03: a RECORDED optimization_target selection (pick verb near the field, or an explicit
    posture assignment) still trips the omission catcher — the D-20 purpose is preserved."""
    assert has_mode_pick_prose("chose optimization_target pro for the dev pass")
    assert has_mode_pick_prose("optimization_target auto-set to draft this round")
    assert has_mode_pick_prose("optimization_target: pro")
    assert has_mode_pick_prose("optimization_target = draft")


def test_has_mode_pick_prose_is_case_insensitive_on_optimization_target() -> None:
    """WR-03: IGNORECASE parity with the sibling patterns — a capitalized field name with a pick
    verb no longer slips the catcher (the old bare arm lacked re.IGNORECASE)."""
    assert has_mode_pick_prose("Optimization_Target was auto-set this round")
    assert not has_mode_pick_prose("the Optimization_Target axis was discussed")


def test_state_slot_values_extracts_values() -> None:
    """``state_slot_values`` returns {key: value} from indented ``name: value`` lines."""
    values = state_slot_values(["    knob: lr", "    tier: 3"])
    assert values == {"knob": "lr", "tier": "3"}


def test_standalone_lint_cli_exit_codes(tmp_path: Path) -> None:
    """The ``python -m signet_trainer.harness_lint --log PATH`` CLI hard-fails on a violating log."""
    from signet_trainer.harness_lint import main as harness_lint_main

    # A minimal log with the adoption anchor + a clean post-cutoff entry -> exit 0.
    clean_log = tmp_path / "clean.md"
    clean_log.write_text(
        "## 2026-07-14 - ADOPTED: typed-state-beats-prose\n"
        "- what: adoption.\n"
        "- why: paper.\n"
        "- state:\n"
        "    cutoff: 2026-07-14\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n",
        encoding="utf-8",
    )
    assert harness_lint_main(["--log", str(clean_log)]) == 0

    # Same log plus a violating entry (numerics in prose, no state block) -> exit 1.
    bad_log = tmp_path / "bad.md"
    bad_log.write_text(
        clean_log.read_text(encoding="utf-8")
        + "\n## 2026-07-15 - Synthetic: numerics only in prose\n"
        "- what: the run cost $12.34 at step 3000.\n"
        "- why: prove the CLI bites.\n"
        "- source-tags: [house]\n"
        "- run-refs: none\n",
        encoding="utf-8",
    )
    assert harness_lint_main(["--log", str(bad_log)]) == 1

    # A missing log file is a distinct non-zero exit (2), not a crash.
    assert harness_lint_main(["--log", str(tmp_path / "absent.md")]) == 2


def test_harness_lint_is_import_confined() -> None:
    """harness_lint must stay pure stdlib (re/pathlib/argparse) — NO torch, modal, yaml, network."""
    src = (
        Path(__file__).resolve().parent.parent
        / "src" / "signet_trainer" / "harness_lint.py"
    ).read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in src.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    for stmt in import_lines:
        for banned in ("torch", "modal", "yaml", "requests", "httpx"):
            assert banned not in stmt, f"harness_lint leaked a heavy/network import: {stmt}"
    # No signet-internal imports at all — the lint is fully self-contained.
    assert not any("signet_trainer" in s for s in import_lines), (
        f"harness_lint must not import other signet_trainer modules: {import_lines}"
    )


def test_test_imports_no_modal() -> None:
    """This test module must not pull in modal (pure CPU scaffold test)."""
    src = Path(__file__).read_text(encoding="utf-8")
    import_lines = [
        line.strip() for line in src.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    assert not any(
        stmt.startswith("import modal") or stmt.startswith("from modal") for stmt in import_lines
    ), "memory-scaffold test must not import modal"
