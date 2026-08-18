"""Typed-state emitter tests (TS-01a + TS-01b). Pure CPU — NO modal, NO GPU, NO network.

Locks the emitter contracts from the typed-state refactor (KNOWLEDGE `typed-state`):

1. ``emit_into`` marker replacement is IDEMPOTENT (same block twice -> byte-identical file) and
   touches nothing outside the marker pair.
2. Missing markers are CREATED right after the registered anchor heading (prose preserved).
3. Malformed markers (unbalanced / duplicated / inverted / no anchor) FAIL LOUD without writing.
4. Ledger figures FLOW from a SESSION-STATE fixture through ``session_cap.read_ledger`` — the
   rendered block carries the computed spend verbatim, never a hand-copy.
5. The typed campaign core (.state.yaml) round-trips: source values survive into the emitted
   block's parsed YAML unchanged (the "never paraphrase a computed value" contract).
6. REAL-file smoke: the three registered repo artifacts each contain exactly one well-formed
   marker pair whose fenced content parses as YAML (the 2026-07-14 conversion holds).

House test rules honored: no metered dispatch, no modal CLI, no network — tmp-file fixtures plus
read-only reads of the committed planning artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from signet_trainer import harness_data
from signet_trainer.harness_state import (
    CAMPAIGN_ENV_VAR,
    CARD_STATE_PREFIX,
    CARD_STATE_SUFFIX,
    CARDS_DIR,
    MARKER_BEGIN,
    MARKER_END,
    MASK_SPEC_ARTIFACT,
    MASK_SPEC_MARKER_BEGIN,
    MASK_SPEC_MARKER_END,
    SESSION_LEDGER_PATH_ENV_VAR,
    SESSION_STATE_PATH,
    TAXONOMY_ARTIFACT,
    artifact_registry,
    assert_verdict_spec,
    campaign_card_md_path,
    check_stale,
    discover_campaign_state_path,
    discover_continue_here_artifacts,
    emit_all,
    emit_into,
    house_spec_path,
    load_campaign_state,
    load_house_spec,
    load_mask_spec,
    load_tier_taxonomy,
    main,
    mask_spec_path,
    read_ledger_figures,
    render_mask_spec_block,
    render_state_block,
    render_tier_table,
    session_ledger_path,
    tier_taxonomy_path,
)
from signet_trainer.modal.session_cap import read_ledger

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixture repos carry their campaign card HERE — discovery (glob over cards/) finds it; no
# campaign name is hardcoded in src or in this test file.
_FIXTURE_CARD_STATE = f"{CARDS_DIR}/{CARD_STATE_PREFIX}test-campaign{CARD_STATE_SUFFIX}"

# A tiny stand-in block — splice mechanics are independent of what render produces.
_SIMPLE_BLOCK = "```yaml\nfoo: 1\nbar: baz\n```"

# The D-8-SOURCETAG source classes — every emitted tier-table row must carry at least one.
_SOURCE_TAGS = ("[house]", "[precedent]", "[canonical]", "[community]")


# --------------------------------------------------------------------------- fixtures / helpers


def _campaign_dict() -> dict:
    """A minimal-but-complete typed campaign core (one completed round, one in-flight)."""
    return {
        "campaign": {"slug": "test-campaign", "phase": "round-2", "goal": "unit-test campaign"},
        "rounds": [
            {
                "round": "r1",
                "config": "configs/a.yaml",
                "init_adapter": None,
                "final_checkpoint": "outputs/r1/checkpoint-step-00100-loss-0.5000",
                "steps": 100,
                "final_loss": 0.5,
                "verdict": "CONTINUE",
                "notes": "clean round",
            },
            {
                "round": "r2",
                "config": "configs/b.yaml",
                "init_adapter": "outputs/r1/checkpoint-step-00100-loss-0.5000",
                "final_checkpoint": None,
                "steps": None,
                "final_loss": None,
                "verdict": None,  # null verdict == in-flight (schema convention)
                "notes": "in-flight",
            },
        ],
        "current": {
            "likeness_final": "outputs/r1/checkpoint-step-00100-loss-0.5000",
            "max_steps": 200,
        },
    }


def _write_session_state(path: Path, cap: float, est_amounts: list[float]) -> None:
    path.write_text(
        json.dumps(
            {
                "triage_mode": "yolo",
                "session_cap_usd": cap,
                "blankets": [],
                "spend": [
                    {"ts": f"2026-07-14T00:0{i}:00+00:00", "est_usd": amt, "run_ref": f"run-{i}"}
                    for i, amt in enumerate(est_amounts)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _parse_block(block: str) -> dict:
    """Parse the emitted code-fenced block back into a dict (the quotability contract)."""
    lines = block.split("\n")
    assert lines[0] == "```yaml", f"block must open with a yaml fence, got {lines[0]!r}"
    assert lines[-1] == "```", f"block must close its fence, got {lines[-1]!r}"
    return yaml.safe_load("\n".join(lines[1:-1]))


def _artifact_with_markers(tmp_path: Path, inner: str = "old: stale\n") -> Path:
    art = tmp_path / "artifact.md"
    art.write_text(
        "# Some card\n\n## TYPED STATE\n\n"
        f"{MARKER_BEGIN}\n```yaml\n{inner}```\n{MARKER_END}\n\n"
        "## Findings\nprose the emitter must never touch\n",
        encoding="utf-8",
    )
    return art


# --------------------------------------------------------------------------- (1) idempotency


def test_replace_is_idempotent(tmp_path: Path) -> None:
    art = _artifact_with_markers(tmp_path)
    assert emit_into(art, _SIMPLE_BLOCK, anchor="## TYPED STATE") == "replaced"
    first = art.read_bytes()
    assert emit_into(art, _SIMPLE_BLOCK, anchor="## TYPED STATE") == "unchanged"
    assert art.read_bytes() == first, "second emit of the same block must be a byte-level no-op"

    text = art.read_text(encoding="utf-8")
    assert text.count(MARKER_BEGIN) == 1 and text.count(MARKER_END) == 1
    assert "foo: 1" in text and "old: stale" not in text


def test_replace_touches_nothing_outside_markers(tmp_path: Path) -> None:
    art = _artifact_with_markers(tmp_path)
    before = art.read_text(encoding="utf-8")
    prefix = before.split(MARKER_BEGIN)[0]
    suffix = before.split(MARKER_END)[1]
    emit_into(art, _SIMPLE_BLOCK)
    after = art.read_text(encoding="utf-8")
    assert after.startswith(prefix), "prose before the marker section must be untouched"
    assert after.endswith(suffix), "prose after the marker section must be untouched"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    art = _artifact_with_markers(tmp_path)
    before = art.read_bytes()
    assert emit_into(art, _SIMPLE_BLOCK, dry_run=True) == "replaced"
    assert art.read_bytes() == before, "dry_run must not write"


# --------------------------------------------------------------------------- (2) creation


def test_missing_markers_created_after_anchor(tmp_path: Path) -> None:
    art = tmp_path / "artifact.md"
    art.write_text(
        "# Card\n\n## TYPED STATE (quotable slots)\nLast session: prose line\n\n## Later\nmore\n",
        encoding="utf-8",
    )
    assert emit_into(art, _SIMPLE_BLOCK, anchor="## TYPED STATE") == "created"

    text = art.read_text(encoding="utf-8")
    lines = text.split("\n")
    anchor_i = next(i for i, l in enumerate(lines) if l.startswith("## TYPED STATE"))
    begin_i = next(i for i, l in enumerate(lines) if l.startswith("<!-- TYPED-STATE:BEGIN"))
    prose_i = next(i for i, l in enumerate(lines) if l.startswith("Last session:"))
    assert anchor_i < begin_i < prose_i, "section must be created between the anchor and the prose"
    assert "Last session: prose line" in text and "## Later" in text, "prose must be preserved"

    # Creation then replacement converges: the same block again is a no-op.
    assert emit_into(art, _SIMPLE_BLOCK, anchor="## TYPED STATE") == "unchanged"


# --------------------------------------------------------------------------- (3) fail-loud


@pytest.mark.parametrize(
    "body, match",
    [
        (f"## H\n{MARKER_BEGIN}\nx\n", "unbalanced"),  # BEGIN without END
        (f"## H\nx\n{MARKER_END}\n", "unbalanced"),  # END without BEGIN
        (
            f"## H\n{MARKER_BEGIN}\nx\n{MARKER_END}\n{MARKER_BEGIN}\ny\n{MARKER_END}\n",
            "duplicated or nested",
        ),
        (f"## H\n{MARKER_END}\nx\n{MARKER_BEGIN}\n", "precedes"),  # inverted order
    ],
)
def test_malformed_markers_fail_loud(tmp_path: Path, body: str, match: str) -> None:
    art = tmp_path / "bad.md"
    art.write_text(body, encoding="utf-8")
    before = art.read_bytes()
    with pytest.raises(ValueError, match=match):
        emit_into(art, _SIMPLE_BLOCK, anchor="## H")
    assert art.read_bytes() == before, "a malformed artifact must never be rewritten"


def test_missing_markers_and_missing_anchor_fail_loud(tmp_path: Path) -> None:
    art = tmp_path / "no-anchor.md"
    art.write_text("# Card\nno typed-state section here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no anchor heading given"):
        emit_into(art, _SIMPLE_BLOCK)  # no markers AND no anchor
    with pytest.raises(ValueError, match="not found"):
        emit_into(art, _SIMPLE_BLOCK, anchor="## TYPED STATE")  # anchor absent from the file


# --------------------------------------------------------------------------- (4) ledger flow


def test_ledger_figures_flow_from_session_state_fixture(tmp_path: Path) -> None:
    """The rendered block's spend figure IS read_ledger's figure — no hand-copy in the chain."""
    session = tmp_path / "SESSION-STATE.json"
    _write_session_state(session, cap=123.45, est_amounts=[1.25, 2.5, 0.05])

    figures = read_ledger_figures(session)
    assert figures["spent_est_usd"] == pytest.approx(3.8)
    assert figures["spent_est_usd"] == pytest.approx(round(read_ledger(session), 2))
    assert figures["session_cap_usd"] == 123.45
    assert figures["entries"] == 3

    block = render_state_block({"campaign": _campaign_dict(), "ledger": figures})
    parsed = _parse_block(block)
    assert parsed["ledger"]["spent_est_usd"] == pytest.approx(3.8)
    assert parsed["ledger"]["session_cap_usd"] == 123.45
    assert parsed["ledger"]["entries"] == 3


def test_missing_session_state_is_fresh_session(tmp_path: Path) -> None:
    figures = read_ledger_figures(tmp_path / "absent.json")
    assert figures["spent_est_usd"] == 0.0
    assert figures["session_cap_usd"] is None
    assert figures["entries"] == 0


# --------------------------------------------------------------------------- (5) campaign round-trip


def test_campaign_yaml_roundtrips(tmp_path: Path) -> None:
    """Source values survive dump -> load -> render -> parse UNCHANGED (quotability contract)."""
    src = _campaign_dict()
    state_file = tmp_path / "card.state.yaml"
    state_file.write_text(yaml.safe_dump(src, sort_keys=False, allow_unicode=True), encoding="utf-8")

    loaded = load_campaign_state(state_file)
    assert loaded == src, "the typed campaign core must round-trip through yaml verbatim"

    session = tmp_path / "SESSION-STATE.json"
    _write_session_state(session, cap=10.0, est_amounts=[0.5])
    parsed = _parse_block(
        render_state_block({"campaign": loaded, "ledger": read_ledger_figures(session)})
    )

    assert parsed["campaign"] == "test-campaign"
    assert parsed["campaign_phase"] == "round-2"
    assert parsed["rounds_recorded"] == 2
    # latest completed round carries its computed values verbatim
    assert parsed["latest_round"]["round"] == "r1"
    assert parsed["latest_round"]["final_loss"] == 0.5
    assert parsed["latest_round"]["final_checkpoint"] == "outputs/r1/checkpoint-step-00100-loss-0.5000"
    # the in-flight (null-verdict) round is surfaced with its init
    assert parsed["in_flight"][0]["round"] == "r2"
    assert parsed["in_flight"][0]["init_adapter"] == "outputs/r1/checkpoint-step-00100-loss-0.5000"
    # current slots are passed through untouched
    assert parsed["current"] == src["current"]


# --------------------------------------------------------------------------- optimization_target (D-13/D-14/D-15)


def test_unmigrated_card_renders_optimization_target_draft(tmp_path: Path) -> None:
    """A card WITHOUT optimization_target loads and renders the D-15 fail-safe ``draft``.

    The emitter never hard-requires the slot (the un-migrated-card fail-safe): a card that predates
    the draft/pro axis must still render, defaulting the posture to ``draft``.
    """
    src = _campaign_dict()  # note: _campaign_dict has NO optimization_target
    assert "optimization_target" not in src["campaign"], "fixture must omit the slot for this test"
    state_file = tmp_path / "card.state.yaml"
    state_file.write_text(yaml.safe_dump(src, sort_keys=False, allow_unicode=True), encoding="utf-8")

    loaded = load_campaign_state(state_file)  # must NOT raise on the missing slot
    session = tmp_path / "SESSION-STATE.json"
    _write_session_state(session, cap=10.0, est_amounts=[0.5])
    parsed = _parse_block(
        render_state_block({"campaign": loaded, "ledger": read_ledger_figures(session)})
    )
    assert parsed["optimization_target"] == "draft", (
        "an un-migrated card must fail safe to draft, matching read_ledger_figures' missing-value idiom"
    )


def test_migrated_card_renders_declared_posture(tmp_path: Path) -> None:
    """A card WITH optimization_target: pro renders ``pro`` (the slot is honored when present)."""
    src = _campaign_dict()
    src["campaign"]["optimization_target"] = "pro"
    state_file = tmp_path / "card.state.yaml"
    state_file.write_text(yaml.safe_dump(src, sort_keys=False, allow_unicode=True), encoding="utf-8")

    loaded = load_campaign_state(state_file)
    session = tmp_path / "SESSION-STATE.json"
    _write_session_state(session, cap=10.0, est_amounts=[0.5])
    parsed = _parse_block(
        render_state_block({"campaign": loaded, "ledger": read_ledger_figures(session)})
    )
    assert parsed["optimization_target"] == "pro"


@pytest.mark.requires_live_harness
def test_real_card_renders_optimization_target() -> None:
    """The committed campaign card carries a VALID campaign-durable posture, and it renders into
    the block.

    Asserts the INVARIANT (present + one of the two legal postures + faithfully rendered), never a
    specific campaign's choice: the posture is an operator decision set per campaign at the setup
    gate, so pinning a literal here makes the suite fail the moment a new campaign legitimately
    picks the other one.
    """
    state = load_campaign_state(_REPO_ROOT / discover_campaign_state_path())
    posture = state["campaign"].get("optimization_target")
    assert posture in {"draft", "pro"}, (
        "the active campaign card must carry a legal posture under campaign: "
        f"(draft|pro); got {posture!r}"
    )
    session = _REPO_ROOT / ".planning" / "harness" / "SESSION-STATE.json"
    parsed = _parse_block(
        render_state_block({"campaign": state, "ledger": read_ledger_figures(session)})
    )
    assert parsed.get("optimization_target") == posture, (
        "the rendered block must carry the card's posture verbatim"
    )


def test_load_campaign_state_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_campaign_state(tmp_path / "absent.state.yaml")

    bad = tmp_path / "bad.state.yaml"
    bad.write_text(yaml.safe_dump({"campaign": {"slug": "x", "phase": "y"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="rounds"):
        load_campaign_state(bad)  # a half-written source must never render


# --------------------------------------------------------------------------- (6) real-file smoke


@pytest.mark.requires_live_harness
def test_real_artifacts_have_exactly_one_wellformed_marker_pair() -> None:
    """The 2026-07-14 conversion: each registered repo artifact carries ONE marker pair whose
    fenced content parses as YAML with the campaign + ledger slots present."""
    for rel_path, _anchor in artifact_registry():
        path = _REPO_ROOT / rel_path
        assert path.is_file(), f"registered artifact missing: {rel_path}"
        lines = path.read_text(encoding="utf-8").split("\n")

        begins = [i for i, l in enumerate(lines) if l.lstrip().startswith("<!-- TYPED-STATE:BEGIN")]
        ends = [i for i, l in enumerate(lines) if l.lstrip().startswith("<!-- TYPED-STATE:END")]
        assert len(begins) == 1, f"{rel_path}: expected exactly 1 BEGIN marker, found {len(begins)}"
        assert len(ends) == 1, f"{rel_path}: expected exactly 1 END marker, found {len(ends)}"
        assert begins[0] < ends[0], f"{rel_path}: BEGIN must precede END"

        inner = lines[begins[0] + 1 : ends[0]]
        assert inner and inner[0] == "```yaml" and inner[-1] == "```", (
            f"{rel_path}: marker section must hold a yaml code fence"
        )
        parsed = yaml.safe_load("\n".join(inner[1:-1]))
        assert isinstance(parsed, dict), f"{rel_path}: emitted block must parse as a mapping"
        for key in ("campaign", "campaign_phase", "current", "ledger"):
            assert key in parsed, f"{rel_path}: emitted block missing {key!r} slot"
        assert parsed["ledger"]["source"] == ".planning/harness/SESSION-STATE.json"


@pytest.mark.requires_live_harness
def test_real_campaign_state_loads_and_carries_live_slots() -> None:
    state_rel = discover_campaign_state_path()
    state = load_campaign_state(_REPO_ROOT / state_rel)
    # The card filename IS the campaign identity: TRAINING-CARD-<slug>.state.yaml must carry a
    # matching campaign.slug inside (the discovery invariant — no campaign name is hardcoded).
    expected_slug = Path(state_rel).name[len(CARD_STATE_PREFIX) : -len(CARD_STATE_SUFFIX)]
    assert state["campaign"]["slug"] == expected_slug
    # `current` is the live-slot mapping (TS-01a territory). Assert the INVARIANT — it exists, is a
    # mapping, and carries the one slot every campaign has regardless of training mode — rather than
    # a fixed slot list: the previous list pinned inpaint-campaign-specific slots (`inpaint_run`,
    # `fused_base`, `likeness_final`), which a vanilla text-to-video campaign has no reason to carry.
    # Modality/mode-specific slots are additive per campaign, not universally required.
    assert isinstance(state["current"], dict) and state["current"], (
        ".state.yaml must carry a non-empty 'current' mapping of live slots"
    )
    assert "data_root" in state["current"], (
        ".state.yaml current is missing the 'data_root' slot (universal across campaigns)"
    )
    # Spend/cap must NOT live in the campaign core — SESSION-STATE.json is the single source.
    assert "session_cap_usd" not in state["current"]
    assert not any("spent" in str(k) for k in state["current"]), (
        "spend figures must never be hand-carried in the campaign core"
    )


# --------------------------------------------------------------------------- import confinement


# --------------------------------------------------------------------------- (7) staleness check


# The fixture's CONTROLLED spec sources. TIER-TAXONOMY.yaml / MASK-SPEC.yaml ship as PACKAGE DATA
# (signet_trainer.harness_data), so writing them under a fixture root no longer makes emit_all read
# them — the fixture must inject them explicitly via the taxonomy_source / mask_spec_source seam.
_FIXTURE_TAXONOMY = "fixture-specs/TIER-TAXONOMY.yaml"
_FIXTURE_MASK_SPEC = "fixture-specs/MASK-SPEC.yaml"


def _fixture_specs(root: Path) -> dict[str, Path]:
    """emit_all/check_stale kwargs pinning the fixture's controlled specs (never the packaged ones)."""
    return {
        "taxonomy_source": root / _FIXTURE_TAXONOMY,
        "mask_spec_source": root / _FIXTURE_MASK_SPEC,
    }


def _build_fixture_repo(tmp_path: Path) -> Path:
    """Build a CONTROLLED tmp repo_root with the typed sources + registered artifacts.

    Deliberately NOT the live ``.planning/STATE.md`` — deferred-items.md records a known gsd-sdk
    fence-reflow hazard there, so the staleness test must run against an isolated fixture root.
    """
    root = tmp_path / "repo"
    # Typed sources. The card lands under cards/ where discovery (glob) will find it.
    state_yaml = root / _FIXTURE_CARD_STATE
    state_yaml.parent.mkdir(parents=True, exist_ok=True)
    state_yaml.write_text(
        yaml.safe_dump(_campaign_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_session_state(root / SESSION_STATE_PATH, cap=100.0, est_amounts=[1.0, 2.0])
    # Tier taxonomy source (Plan 02: emit_all also emits the knob->tier table). Minimal but
    # shape-valid — default_tier 3, non-empty knobs — so load_tier_taxonomy accepts it.
    tax = root / _FIXTURE_TAXONOMY
    tax.parent.mkdir(parents=True, exist_ok=True)
    tax.write_text(yaml.safe_dump(_taxonomy_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    # Mask spec source (Plan 09.3-01: emit_all also emits the mask-spec block into KNOWLEDGE.md).
    mask = root / _FIXTURE_MASK_SPEC
    mask.parent.mkdir(parents=True, exist_ok=True)
    mask.write_text(yaml.safe_dump(_mask_spec_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    # Registered prose artifacts + the KNOWLEDGE.md receiver. Group anchors BY FILE so KNOWLEDGE.md
    # (which receives BOTH the tier table AND the mask-spec block, at distinct anchors + distinct
    # marker types) carries BOTH anchors and emit can CREATE each block. artifact_registry(root)
    # DISCOVERS the fixture card written above — the registry is per-repo, never a constant.
    anchors_by_file: dict[str, list[str]] = {}
    for rel_path, anchor in [*artifact_registry(root), TAXONOMY_ARTIFACT, MASK_SPEC_ARTIFACT]:
        anchors_by_file.setdefault(rel_path, []).append(anchor)
    for rel_path, anchors in anchors_by_file.items():
        art = root / rel_path
        art.parent.mkdir(parents=True, exist_ok=True)
        body = "# Fixture artifact\n\n"
        for anchor in anchors:
            body += f"{anchor}\n\nprose the emitter must preserve\n\n"
        art.write_text(body, encoding="utf-8")
    return root


def test_check_stale_is_empty_right_after_emit(tmp_path: Path) -> None:
    """After emit_all writes fresh blocks, check_stale reports NO drift (every action unchanged)."""
    root = _build_fixture_repo(tmp_path)
    created = emit_all(dry_run=False, repo_root=root, **_fixture_specs(root))
    assert all(action in ("created", "replaced") for _rel, action in created), created
    assert check_stale(repo_root=root, **_fixture_specs(root)) == [], "freshly-emitted blocks must not be reported stale"


def test_check_stale_reports_drift_when_source_mutated(tmp_path: Path) -> None:
    """Mutating the typed SOURCE without re-emitting makes check_stale report the drifted artifacts."""
    root = _build_fixture_repo(tmp_path)
    emit_all(dry_run=False, repo_root=root, **_fixture_specs(root))
    assert check_stale(repo_root=root, **_fixture_specs(root)) == []

    # Bump a step in the typed .state.yaml source WITHOUT re-emitting -> the prose blocks stale.
    state_yaml = root / _FIXTURE_CARD_STATE
    data = yaml.safe_load(state_yaml.read_text(encoding="utf-8"))
    data["rounds"][0]["steps"] = 999  # was 100 in _campaign_dict()
    state_yaml.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    drifted = check_stale(repo_root=root, **_fixture_specs(root))
    assert drifted, "a source mutation without re-emit must be reported as drift"
    drifted_paths = {rel for rel, _action in drifted}
    assert drifted_paths == {rel for rel, _anchor in artifact_registry(root)}, (
        f"every registered artifact should drift after the shared source changed: {drifted}"
    )
    assert all(action != "unchanged" for _rel, action in drifted)


def test_discovery_fails_loud_on_empty_cards_dir(tmp_path: Path) -> None:
    """Zero campaign cards is never guessed around — discovery raises FileNotFoundError."""
    root = tmp_path / "repo"
    (root / CARDS_DIR).mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="no campaign card"):
        discover_campaign_state_path(root)


def test_discovery_multiple_cards_requires_env_pin(tmp_path: Path, monkeypatch) -> None:
    """Two cards without a SIGNET_CAMPAIGN pin fails LOUD; the pin selects; a bad pin fails LOUD."""
    root = tmp_path / "repo"
    cards = root / CARDS_DIR
    cards.mkdir(parents=True, exist_ok=True)
    for slug in ("campaign-a", "campaign-b"):
        (cards / f"{CARD_STATE_PREFIX}{slug}{CARD_STATE_SUFFIX}").write_text("campaign: {}\n", encoding="utf-8")

    monkeypatch.delenv(CAMPAIGN_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match=CAMPAIGN_ENV_VAR):
        discover_campaign_state_path(root)

    monkeypatch.setenv(CAMPAIGN_ENV_VAR, "campaign-b")
    rel = discover_campaign_state_path(root)
    assert rel == f"{CARDS_DIR}/{CARD_STATE_PREFIX}campaign-b{CARD_STATE_SUFFIX}"
    assert campaign_card_md_path(rel) == f"{CARDS_DIR}/{CARD_STATE_PREFIX}campaign-b.md"

    monkeypatch.setenv(CAMPAIGN_ENV_VAR, "no-such-campaign")
    with pytest.raises(FileNotFoundError, match="no-such-campaign"):
        discover_campaign_state_path(root)


def test_check_cli_exit_codes_on_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``check`` CLI exits 0 when fresh and non-zero after a source drifts (fixture root).

    The CLI has no spec-source flags, so it points at the fixture's controlled specs through the
    PRODUCTION override — ``SIGNET_HARNESS_DATA_DIR``. That makes this the test that proves the
    override env var actually redirects packaged-spec resolution end to end.
    """
    root = _build_fixture_repo(tmp_path)
    monkeypatch.setenv(harness_data.OVERRIDE_ENV_VAR, str(root / "fixture-specs"))
    emit_all(dry_run=False, repo_root=root, **_fixture_specs(root))
    assert main(["check", "--repo-root", str(root)]) == 0, "clean fixture must exit 0"

    state_yaml = root / _FIXTURE_CARD_STATE
    data = yaml.safe_load(state_yaml.read_text(encoding="utf-8"))
    data["current"]["max_steps"] = 12345
    state_yaml.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    assert main(["check", "--repo-root", str(root)]) == 1, "drifted fixture must exit non-zero"


# --------------------------------------------------------------------------- (8) house verdict spec


def _house_spec_dict() -> dict:
    """A minimal house verdict spec mirroring HOUSE-SPEC.yaml's verdict_spec (Plan 01)."""
    return {
        "verdict_spec": {
            "frame_count": 81,
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "seed": 42,
            "two_stage_upscale": False,
            "min_prompt_count": 2,
        },
        "render_tiers": ["quicklook", "verdict"],
    }


def _matching_validation() -> dict:
    """A render config validation block that matches the house verdict spec exactly."""
    return {
        "frame_count": 81,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "seed": 42,
        "two_stage_upscale": False,
        "prompts": ["prompt one", "prompt two"],
    }


def test_load_house_spec_accepts_real_committed_file() -> None:
    spec = load_house_spec(house_spec_path())
    assert isinstance(spec, dict) and isinstance(spec["verdict_spec"], dict)
    for key in ("frame_count", "num_inference_steps", "guidance_scale", "seed", "min_prompt_count"):
        assert key in spec["verdict_spec"], f"real HOUSE-SPEC.yaml verdict_spec missing {key!r}"


def test_load_house_spec_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_house_spec(tmp_path / "absent-house-spec.yaml")


def test_load_house_spec_malformed_names_file(tmp_path: Path) -> None:
    bad = tmp_path / "HOUSE-SPEC.yaml"
    bad.write_text(yaml.safe_dump({"verdict_spec": {"frame_count": 81}}), encoding="utf-8")
    with pytest.raises(ValueError, match=str(bad.name)):
        load_house_spec(bad)  # missing num_inference_steps/guidance_scale/seed/min_prompt_count


def test_assert_verdict_spec_quicklook_is_noop() -> None:
    # quicklook is mode-cheapenable: it returns without checking, even on garbage validation.
    assert assert_verdict_spec({"frame_count": 1}, "quicklook", _house_spec_dict()) is None


def test_assert_verdict_spec_verdict_passes_on_match() -> None:
    assert assert_verdict_spec(_matching_validation(), "verdict", _house_spec_dict()) is None


def test_assert_verdict_spec_trimmed_steps_names_param_and_values() -> None:
    val = _matching_validation()
    val["num_inference_steps"] = 10
    with pytest.raises(ValueError) as exc:
        assert_verdict_spec(val, "verdict", _house_spec_dict())
    msg = str(exc.value)
    assert "num_inference_steps" in msg and "10" in msg and "40" in msg


def test_assert_verdict_spec_frame_count_49_named() -> None:
    val = _matching_validation()
    val["frame_count"] = 49  # the schema.py default — the realistic trim
    with pytest.raises(ValueError, match="frame_count"):
        assert_verdict_spec(val, "verdict", _house_spec_dict())


def test_assert_verdict_spec_single_prompt_rejected() -> None:
    val = _matching_validation()
    val["prompts"] = ["only one"]
    with pytest.raises(ValueError, match="prompt count"):
        assert_verdict_spec(val, "verdict", _house_spec_dict())


def test_assert_verdict_spec_reports_all_mismatches_at_once() -> None:
    val = _matching_validation()
    val["num_inference_steps"] = 10
    val["frame_count"] = 49
    val["prompts"] = ["only one"]
    with pytest.raises(ValueError) as exc:
        assert_verdict_spec(val, "verdict", _house_spec_dict())
    msg = str(exc.value)
    # ALL three mismatches surface in a single message, not just the first.
    assert "num_inference_steps" in msg and "frame_count" in msg and "prompt count" in msg


def test_assert_verdict_spec_omitted_two_stage_upscale_passes() -> None:
    """WR-01: a verdict config that OMITS two_stage_upscale (relying on the schema default False)
    is compliant — omission of a key whose schema default equals the house value is not a mismatch."""
    val = _matching_validation()
    del val["two_stage_upscale"]  # rely on schema.py ValidationConfig's default (False)
    assert "two_stage_upscale" not in val
    assert assert_verdict_spec(val, "verdict", _house_spec_dict()) is None


def test_assert_verdict_spec_explicit_two_stage_upscale_true_still_fails() -> None:
    """WR-01: an EXPLICIT two_stage_upscale: true still fails loudly — the schema-default fallback
    only rescues OMISSION, never an explicit non-house value."""
    val = _matching_validation()
    val["two_stage_upscale"] = True
    with pytest.raises(ValueError) as exc:
        assert_verdict_spec(val, "verdict", _house_spec_dict())
    msg = str(exc.value)
    assert "two_stage_upscale" in msg and "True" in msg


def test_assert_verdict_spec_unknown_tier_lists_valid_tiers() -> None:
    with pytest.raises(ValueError) as exc:
        assert_verdict_spec(_matching_validation(), "cheap", _house_spec_dict())
    msg = str(exc.value)
    assert "cheap" in msg and "quicklook" in msg and "verdict" in msg


def test_assert_verdict_spec_source_never_reads_the_posture() -> None:
    """D-24: no verdict path reads optimization_target / the posture. Proven by AST here too."""
    import ast

    src = (_REPO_ROOT / "src" / "signet_trainer" / "harness_state.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "assert_verdict_spec"
    )
    unparsed = ast.unparse(fn)
    for banned in ("optimization_target", "draft", "triage_mode"):
        assert banned not in unparsed, f"verdict path references {banned!r} — D-24 violated"


def test_assert_verdict_spec_cli_passes_real_r5_config() -> None:
    """The r5 example config IS the house spec source — the real-file proof (default tier == verdict)."""
    assert main(["assert-verdict-spec", "--config", "configs/campaign_r5.example.yaml", "--render-tier", "verdict"]) == 0
    assert main(["assert-verdict-spec", "--config", "configs/campaign_r5.example.yaml"]) == 0  # default verdict


# --------------------------------------------------------------------------- (9) tier taxonomy


def _taxonomy_dict() -> dict:
    return {
        "version": 1,
        "default_tier": 3,
        "tiers": {1: "mode-auto", 2: "carve-out", 3: "house-first"},
        "knobs": {
            "checkpoint_every": {"tier": 1, "source_tags": "[house]", "note": "cadence, cost-only"},
            "lora.rank": {"tier": 2, "source_tags": "[house] [precedent]", "note": "capacity knob"},
            "lr": {"tier": 3, "source_tags": "[house]", "note": "learning rate, house 5e-5"},
            "target_modules": {"tier": 3, "fenced": True, "source_tags": "[house] [precedent]", "note": "hard-fenced"},
        },
        "not_a_mode_lever": [
            {"keep_checkpoints": "never a mode lever (D-08)"},
            {"validation.two_stage_upscale": "eval-honesty rule (D-26)"},
        ],
    }


def test_load_tier_taxonomy_accepts_real_committed_file() -> None:
    tax = load_tier_taxonomy(tier_taxonomy_path())
    assert tax["default_tier"] == 3 and isinstance(tax["knobs"], dict) and tax["knobs"]


def test_load_tier_taxonomy_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tier_taxonomy(tmp_path / "absent.yaml")


def test_load_tier_taxonomy_wrong_default_tier_is_a_safety_hole(tmp_path: Path) -> None:
    bad = tmp_path / "TIER-TAXONOMY.yaml"
    data = _taxonomy_dict()
    data["default_tier"] = 1  # fail-safe to auto == the safety hole D-06 forbids
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="default_tier"):
        load_tier_taxonomy(bad)


def test_load_tier_taxonomy_empty_knobs_raises(tmp_path: Path) -> None:
    bad = tmp_path / "TIER-TAXONOMY.yaml"
    bad.write_text(yaml.safe_dump({"default_tier": 3, "knobs": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="knobs"):
        load_tier_taxonomy(bad)


def test_load_tier_taxonomy_malformed_knob_value_raises(tmp_path: Path) -> None:
    """WR-02: a knob whose VALUE is a scalar (not a ``{tier: N, ...}`` mapping) fails LOUD at the
    loader, naming the offending knob + file — instead of crashing render_tier_table with an
    opaque AttributeError downstream."""
    bad = tmp_path / "TIER-TAXONOMY.yaml"
    data = _taxonomy_dict()
    data["knobs"]["foo"] = "bar"  # scalar value — not the required flow-mapping shape
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_tier_taxonomy(bad)
    msg = str(exc.value)
    assert "foo" in msg and bad.name in msg, f"error must name the knob and the file: {msg}"


def test_load_tier_taxonomy_non_int_tier_raises(tmp_path: Path) -> None:
    """WR-02: a knob mapping without an int ``tier`` in {1,2,3} (incl. a bool ``tier: true``) fails
    loud — ``True == 1`` must not sneak a knob in as Tier 1."""
    for bad_tier in ("high", True):
        bad = tmp_path / "TIER-TAXONOMY.yaml"
        data = _taxonomy_dict()
        data["knobs"]["weird"] = {"tier": bad_tier, "source_tags": "[house]"}
        bad.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match="weird"):
            load_tier_taxonomy(bad)


def test_render_tier_table_is_deterministic_and_sorted() -> None:
    tax = _taxonomy_dict()
    first = render_tier_table(tax)
    assert first == render_tier_table(tax), "re-rendering unchanged input must be byte-identical"
    # rows sorted by (tier, knob): tier-1 checkpoint_every precedes the tier-3 lr row.
    assert first.index("checkpoint_every") < first.index("| `lr`")


def test_render_tier_table_every_row_has_a_source_tag() -> None:
    table = render_tier_table(_taxonomy_dict())
    for row in table.splitlines():
        if row.startswith("| `") and "knob |" not in row:
            assert any(tag in row for tag in _SOURCE_TAGS), f"row lacks a source tag: {row}"


def test_render_tier_table_marks_fenced_knob() -> None:
    table = render_tier_table(_taxonomy_dict())
    fenced_row = next(l for l in table.splitlines() if l.startswith("| `target_modules`"))
    assert "HARD-FENCED" in fenced_row
    lr_row = next(l for l in table.splitlines() if l.startswith("| `lr`"))
    assert "HARD-FENCED" not in lr_row, "an ordinary tier-3 row must not read as hard-fenced"


def test_render_tier_table_not_a_mode_lever_has_no_tier() -> None:
    table = render_tier_table(_taxonomy_dict())
    # keep_checkpoints appears only in the labelled list, never as a table row with a tier column.
    assert "Not a mode lever" in table
    assert not any(
        l.startswith("| `keep_checkpoints`") for l in table.splitlines()
    ), "not_a_mode_lever entries must never render as tiered table rows"


def test_emitter_is_import_confined() -> None:
    """harness_state must stay yaml/json/stdlib + session_cap only — NO torch, NO modal."""
    src = (_REPO_ROOT / "src" / "signet_trainer" / "harness_state.py").read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in src.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    for stmt in import_lines:
        assert not stmt.startswith(("import torch", "from torch")), f"torch import leaked: {stmt}"
        assert not stmt.startswith("import modal"), f"modal import leaked: {stmt}"
        if stmt.startswith("from modal"):
            raise AssertionError(f"modal import leaked: {stmt}")
    # Exactly two signet-internal imports are allowed, and the allow-list is CLOSED (equality, not
    # membership) so a heavy dependency cannot sneak in transitively:
    #   * session_cap.read_ledger  — the pure-stdlib authoritative ledger reader;
    #   * harness_data             — the SPEC/TEMPLATE package data (stdlib-only: importlib.resources).
    # harness_data was added when the specs moved out of `.planning/harness/` (unshipped) into
    # package data; both modules are verified import-light by their own confinement tests.
    internal = [s for s in import_lines if "signet_trainer" in s]
    assert internal == [
        "from signet_trainer import harness_data",
        "from signet_trainer.modal.session_cap import read_ledger",
    ], f"unexpected internal imports (only harness_data + session_cap.read_ledger are allowed): {internal}"


# --------------------------------------------------------------------------- (10) mask spec block


def _mask_spec_dict() -> dict:
    """A minimal-but-complete mask spec mirroring MASK-SPEC.yaml (Plan 09.3-01)."""
    return {
        "hex": "#458070",
        "rgb": [69, 128, 112],
        "chroma": {"method": "channel_relationship", "alternate_method": "distance", "tolerance": 40},
        "thresholds": {"seed_load": 127, "encode_binarize": 0.5},
        "polarity": {"chain": ["paint_region=white_255", "stage_negate", "mask_mp4_region=black_0"]},
        "coverage_bands": {
            "face": {"target": 0.08, "warn_below": 0.04,
                     "dilation": {"max_margin_px": 12, "grow_to_target": True}},
            "face_hair": {"target": 0.14, "warn_below": 0.08,
                          "dilation": {"max_margin_px": 28, "grow_to_target": True}},
        },
        "dims": {"spatial_divisor": 64, "frame_rule": "8n+1"},
    }


def test_load_mask_spec_accepts_real_committed_file() -> None:
    spec = load_mask_spec(mask_spec_path())
    assert spec["hex"] == "#458070"
    for key in ("hex", "chroma", "thresholds", "polarity", "coverage_bands", "dims"):
        assert key in spec, f"real MASK-SPEC.yaml missing {key!r}"


def test_load_mask_spec_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mask_spec(tmp_path / "absent-mask-spec.yaml")


def test_load_mask_spec_missing_key_names_file(tmp_path: Path) -> None:
    bad = tmp_path / "MASK-SPEC.yaml"
    data = _mask_spec_dict()
    del data["hex"]
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=str(bad.name)):
        load_mask_spec(bad)


def test_render_mask_spec_block_is_deterministic_tagged_and_no_headings() -> None:
    spec = _mask_spec_dict()
    first = render_mask_spec_block(spec)
    assert first == render_mask_spec_block(spec), "re-rendering unchanged input must be byte-identical"
    assert "#458070" in first, "the canonical hex must render"
    assert any(tag in first for tag in _SOURCE_TAGS), "the mask-spec block must be source-tagged"
    assert not any(line.startswith("#") for line in first.splitlines()), (
        "NO #-prefixed lines (the KNOWLEDGE.md section scanner would mis-parse them as headings)"
    )
    # the D-02 per-class auto-extend dilation margin surfaces in the emitted table
    assert "face_hair" in first and "28" in first


def test_mask_spec_emitted_into_knowledge_and_drift_fenced(tmp_path: Path) -> None:
    """The mask-spec block is emitted into KNOWLEDGE.md, and a hand-edit inside its markers drifts."""
    root = _build_fixture_repo(tmp_path)
    emit_all(dry_run=False, repo_root=root, **_fixture_specs(root))
    assert check_stale(repo_root=root, **_fixture_specs(root)) == [], "freshly-emitted mask-spec block must not be stale"

    knowledge = root / MASK_SPEC_ARTIFACT[0]
    text = knowledge.read_text(encoding="utf-8")
    assert MASK_SPEC_MARKER_BEGIN in text and MASK_SPEC_MARKER_END in text
    corrupted = text.replace("#458070", "#000000", 1)  # hand-edit inside the emitted block
    assert corrupted != text, "the hex must appear in the emitted mask-spec block"
    knowledge.write_text(corrupted, encoding="utf-8")

    drifted = check_stale(repo_root=root, **_fixture_specs(root))
    assert any(rel == MASK_SPEC_ARTIFACT[0] for rel, _ in drifted), (
        "a hand-edit to the emitted mask-spec block must be caught by the drift fence"
    )
    emit_all(dry_run=False, repo_root=root, **_fixture_specs(root))  # re-emit restores freshness
    assert check_stale(repo_root=root, **_fixture_specs(root)) == []


def test_knowledge_holds_both_block_types_without_collision(tmp_path: Path) -> None:
    """KNOWLEDGE.md carries the tier TYPED-STATE block AND the mask-spec block — distinct markers."""
    root = _build_fixture_repo(tmp_path)
    emit_all(dry_run=False, repo_root=root, **_fixture_specs(root))
    text = (root / MASK_SPEC_ARTIFACT[0]).read_text(encoding="utf-8")
    assert text.count(MASK_SPEC_MARKER_BEGIN) == 1 and text.count(MASK_SPEC_MARKER_END) == 1
    assert "<!-- TYPED-STATE:BEGIN" in text and "<!-- TYPED-STATE:END" in text, (
        "the tier table's TYPED-STATE markers must coexist with the mask-spec markers"
    )


@pytest.mark.requires_live_harness
def test_real_knowledge_carries_mask_spec_section_and_is_clean() -> None:
    """The committed KNOWLEDGE.md holds the emitted mask-spec section and reports no drift."""
    text = (_REPO_ROOT / MASK_SPEC_ARTIFACT[0]).read_text(encoding="utf-8")
    assert MASK_SPEC_MARKER_BEGIN in text and "#458070" in text
    assert "<!-- TYPED-STATE:BEGIN" in text and "<!-- MASK-SPEC:BEGIN" in text
    assert check_stale() == [], "committed KNOWLEDGE.md must match MASK-SPEC.yaml + the typed sources"


# --------------------------------------------------------------------------- (11) issue #37 finding 3:
# single-source the ledger path (SESSION_LEDGER_PATH_ENV_VAR / session_ledger_path)


def test_session_ledger_path_matches_schema_default() -> None:
    """SESSION_STATE_PATH must stay byte-identical to ModalConfig.session_spend_ledger_path's
    default — the literal this module cannot import (test_emitter_is_import_confined), pinned
    here instead so the two can never silently diverge (issue #37 finding 3)."""
    from signet_trainer.config.schema import ModalConfig

    assert SESSION_STATE_PATH == ModalConfig.model_fields["session_spend_ledger_path"].default


def test_session_ledger_path_resolution_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """explicit > SIGNET_SESSION_LEDGER_PATH env var > the packaged SESSION_STATE_PATH default."""
    monkeypatch.delenv(SESSION_LEDGER_PATH_ENV_VAR, raising=False)
    assert session_ledger_path() == SESSION_STATE_PATH

    monkeypatch.setenv(SESSION_LEDGER_PATH_ENV_VAR, ".planning/harness/campaign-x/SESSION-STATE.json")
    assert session_ledger_path() == ".planning/harness/campaign-x/SESSION-STATE.json", (
        "an operator who set cfg.modal.session_spend_ledger_path off the default must be able to "
        "point this module at the SAME file via the env var (issue #37 finding 3)"
    )

    assert session_ledger_path("explicit/path.json") == "explicit/path.json", (
        "an explicit caller-supplied path must win over the env var"
    )


def test_missing_ledger_renders_a_loud_warning_not_an_authoritative_zero(tmp_path: Path) -> None:
    """A ledger path nothing writes must say so LOUDLY in the emitted block (issue #37 finding 3),
    not render a plausible-looking $0.00 that could be a wrong-path artifact instead of real state."""
    figures = read_ledger_figures(tmp_path / "absent.json")
    assert figures["available"] is False

    block = render_state_block({"campaign": _campaign_dict(), "ledger": figures})
    parsed = _parse_block(block)
    assert parsed["ledger"]["available"] is False
    assert "WARNING" in parsed["ledger"], "a missing ledger file must carry a loud warning slot"
    assert "not found" in parsed["ledger"]["WARNING"]

    # And the present-ledger case carries no such warning (no false alarms on a real, fresh ledger).
    session = tmp_path / "SESSION-STATE.json"
    _write_session_state(session, cap=10.0, est_amounts=[1.0])
    present_figures = read_ledger_figures(session)
    assert present_figures["available"] is True
    present_block = _parse_block(
        render_state_block({"campaign": _campaign_dict(), "ledger": present_figures})
    )
    assert "WARNING" not in present_block["ledger"]


# --------------------------------------------------------------------------- (12) issue #37 finding 5:
# STATIC_ARTIFACTS discovery + emit_into's missing-artifact guard


def test_discover_continue_here_artifacts_is_empty_when_no_phases_dir(tmp_path: Path) -> None:
    """A project with no '.planning/phases' at all (like this repo) gets zero entries, not an error —
    the old hardcoded phase-slug literal is gone; discovery replaces it (issue #37 finding 5)."""
    root = tmp_path / "repo"
    root.mkdir()
    assert discover_continue_here_artifacts(root) == []


def test_discover_continue_here_artifacts_finds_every_continue_here_md(tmp_path: Path) -> None:
    """Every '.planning/phases/*/.continue-here.md' is discovered, none hand-picked by name."""
    root = tmp_path / "repo"
    for slug in ("03-alpha", "07-beta"):
        phase_dir = root / ".planning" / "phases" / slug
        phase_dir.mkdir(parents=True)
        (phase_dir / ".continue-here.md").write_text("## TYPED STATE\n\nprose\n", encoding="utf-8")
    # A phase dir with no .continue-here.md must not be picked up as a false positive.
    (root / ".planning" / "phases" / "11-gamma").mkdir(parents=True)

    found = discover_continue_here_artifacts(root)
    assert {rel for rel, _anchor in found} == {
        ".planning/phases/03-alpha/.continue-here.md",
        ".planning/phases/07-beta/.continue-here.md",
    }
    assert all(anchor == "## TYPED STATE" for _rel, anchor in found)


def test_emit_into_raises_actionable_not_a_traceback_on_missing_artifact(tmp_path: Path) -> None:
    """A registered artifact that does not exist raises FileNotFoundError with an actionable
    message (issue #37 finding 5) — never the bare 7-frame traceback from an unguarded read."""
    missing = tmp_path / "STATE.md"
    with pytest.raises(FileNotFoundError, match="registered artifact not found"):
        emit_into(missing, _SIMPLE_BLOCK, anchor="## H")


def test_check_command_fails_actionably_not_with_a_bare_traceback_when_planning_is_absent(
    tmp_path: Path,
) -> None:
    """'harness_state check' on a project shipping no .planning/ at all (this repo, per issue #37
    finding 5) must exit 2 with an actionable message — never propagate a bare traceback, and
    never collide with exit 1 (reserved for genuine drift, per the proposed fix)."""
    empty_project = tmp_path / "empty-project"
    empty_project.mkdir()

    rc = main(["check", "--repo-root", str(empty_project)])

    assert rc == 2, "a missing project (no campaign card, no .planning/) must exit 2, not crash"
