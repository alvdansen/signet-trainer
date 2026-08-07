"""TS-01c typed-state lint — the standalone, importable + CLI home of the DECISION-LOG lint.

"Typed State Beats Prose" (KNOWLEDGE `typed-state`, operator-adopted 2026-07-14; protocol at
`.planning/harness/COMPACTION-PROTOCOL.md`): under cascaded compaction, prose summaries corrupted
36.4% of numeric probes vs 9.4% for typed ``name: value`` blocks. This module is the SINGLE source
of the DECISION-LOG lint that enforces the protocol's law #2 — every dated log entry AFTER the
adoption entry whose prose carries load-bearing numerics ($-figures, step counts, loss values,
checkpoint-step paths) MUST carry a ``- state:`` block of simple indented ``name: value`` lines.

The lint logic was first implemented inside ``tests/test_memory_scaffold.py`` (the TS-01c hard-fail
gate). TS-01c EXTRACTS it here verbatim so it (a) runs OUTSIDE pytest as a pre-commit / CI hook and
(b) is single-sourced — the test now imports these functions rather than duplicating them (mirrors
how ``tests/test_harness_state.py`` imports ``signet_trainer.harness_state``).

Two enforcement layers, both forward-only from the 2026-07-14 cutoff (D-TS-5):

  1. **Governing rule (D-TS-2 headline):** any load-bearing numeric in an entry's PROSE requires a
     ``- state:`` block; a present block must carry at least one line; every block line must parse
     as a simple indented ``name: value`` (no prose sentences inside the typed block).
  2. **Mandatory-slot core (D-TS-2):** an entry may DECLARE its class with a ``class:`` slot inside
     its ``- state:`` block — ``class: run`` / ``class: verdict`` / ``class: incident``. A declared
     class asserts its mandatory core slots are present:

        run     : round, config, init_adapter, final_checkpoint, steps, final_loss
        verdict : round, checkpoint_judged, verdict
        incident: impact, detected_via

     Entries with NO ``class:`` slot keep only the governing rule (forward-only — untagged
     legacy-style entries are never over-constrained). The entry-class tagging convention (a
     ``class:`` slot on the typed block) is documented in COMPACTION-PROTOCOL.md.

Severity is **hard-fail** (D-TS-3): any violation string fails the memory-scaffold suite and makes
the CLI exit non-zero.

Import-confined (Anti-Pattern 6, mirrors ``harness_state.py`` / ``session_cap.py``): re / pathlib /
argparse — pure stdlib, NO torch / modal / yaml / network. Runs free on Windows / CI.

CLI (module runs as ``python -m signet_trainer.harness_lint`` from the repo root)::

    PYTHONPATH=src PYTHONUTF8=1 python -m signet_trainer.harness_lint [--log PATH]

Default ``--log`` is ``.planning/harness/DECISION-LOG.md``. Prints every violation line and exits
non-zero when any entry fails; prints a clean line and exits 0 otherwise.
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import sys
from contextlib import ExitStack
from importlib import resources
from pathlib import Path

# --- LIVE state vs PACKAGED spec ---------------------------------------------------------------
#
# The DECISION-LOG is LIVE per-project state (operator-written, never shipped), so its default stays
# PROJECT-relative and is resolved against ``project_root()`` below — with an actionable message,
# not a stack trace, when there is no project. The tier taxonomy is a SPEC that ships as package
# data, so it is resolved through ``importlib.resources`` and works from an installed wheel.

PLANNING_HARNESS_DIR = ".planning/harness"
PROJECT_ROOT_ENV_VAR = "SIGNET_PROJECT_ROOT"
DEFAULT_LOG_PATH = f"{PLANNING_HARNESS_DIR}/DECISION-LOG.md"

# The checkout this module was imported from (src/signet_trainer/harness_lint.py -> parents[2]).
# Correct in the source repo, meaningless from site-packages — hence project_root().
REPO_ROOT = Path(__file__).resolve().parents[2]

# The knob->tier partition (the draft/pro safety mechanism, D-30 part 1). harness_lint reads it by
# a STDLIB REGEX rather than ``yaml.safe_load``: ``test_harness_lint_is_import_confined`` bans the
# yaml import AND every ``signet_trainer`` import, so the lint must stay self-contained. The
# taxonomy's one-line flow-mapping shape (``<knob>: {tier: N, ...}``, tier FIRST) exists precisely
# so this regex and ``yaml.safe_load`` extract byte-identical knob->tier maps (the dual-parser
# contract, Plan 01). Plan 07 makes that agreement a permanent cross-parser test.
#
# Import confinement is why the resource anchor + env var below are STRING LITERALS rather than an
# import of ``signet_trainer.harness_data``: the lint may not import a sibling signet_trainer module.
# ``resources.files()`` takes the anchor by name, so the data still resolves from a wheel (or a zip
# import) without a code dependency. ``test_harness_lint_taxonomy_matches_harness_data`` asserts
# these literals still name the same file the data package exposes, so the duplication cannot drift.
TIER_TAXONOMY_NAME = "TIER-TAXONOMY.yaml"
HARNESS_DATA_PACKAGE = "signet_trainer.harness_data"
HARNESS_DATA_OVERRIDE_ENV_VAR = "SIGNET_HARNESS_DATA_DIR"
# Back-compat alias for callers that imported the old constant. It now names the packaged FILE, not
# a ``.planning/`` path, and is not used for resolution (see tier_taxonomy_path()).
TIER_TAXONOMY_PATH = TIER_TAXONOMY_NAME


def project_root(explicit: Path | None = None) -> Path:
    """The project root that owns the LIVE harness state (the dir holding ``.planning/harness``).

    Mirrors ``harness_state.project_root`` deliberately (import confinement forbids sharing it):
    explicit arg -> ``SIGNET_PROJECT_ROOT`` -> nearest CWD ancestor carrying ``.planning/harness``
    -> this module's own checkout. Never raises; the caller names the missing artifact.
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(PROJECT_ROOT_ENV_VAR, "").strip()
    if env:
        return Path(env)
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / PLANNING_HARNESS_DIR).is_dir():
            return candidate
    return REPO_ROOT


# ``resources.as_file`` may extract a temp file (zip import) and deletes it when its context exits,
# so the returned Path would dangle. This ExitStack holds the materialised file open for the process
# and is closed at interpreter exit. For a normal directory install nothing is extracted.
_FILE_MANAGER = ExitStack()
atexit.register(_FILE_MANAGER.close)
_MATERIALIZED_TAXONOMY: list[Path] = []


def tier_taxonomy_path() -> Path:
    """Real path of the PACKAGED tier taxonomy, honouring ``SIGNET_HARNESS_DATA_DIR``. Fails LOUD."""
    override = os.environ.get(HARNESS_DATA_OVERRIDE_ENV_VAR, "").strip()
    if override:
        candidate = Path(override) / TIER_TAXONOMY_NAME
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{HARNESS_DATA_OVERRIDE_ENV_VAR}={override} does not contain "
                f"{TIER_TAXONOMY_NAME!r} ({candidate}) — an override directory must supply every "
                f"spec it overrides; unset it to use the packaged default."
            )
        return candidate
    if _MATERIALIZED_TAXONOMY and _MATERIALIZED_TAXONOMY[0].is_file():
        return _MATERIALIZED_TAXONOMY[0]
    ref = resources.files(HARNESS_DATA_PACKAGE).joinpath(TIER_TAXONOMY_NAME)
    if not ref.is_file():
        raise FileNotFoundError(
            f"packaged {TIER_TAXONOMY_NAME!r} is missing from {HARNESS_DATA_PACKAGE} — the install "
            f"is incomplete; reinstall signet-trainer or set {HARNESS_DATA_OVERRIDE_ENV_VAR}."
        )
    materialized = Path(_FILE_MANAGER.enter_context(resources.as_file(ref)))
    _MATERIALIZED_TAXONOMY.append(materialized)
    return materialized

# One knob per line, flow-mapping, tier FIRST. The ``tiers:`` legend lines (``  1: "..."``) carry
# no ``{tier:`` mapping and the ``not_a_mode_lever`` entries start with ``- `` — neither matches.
_TIER_KNOB_RE = re.compile(r"^\s{2,}([A-Za-z0-9_.]+):\s*\{\s*tier:\s*([123])\b")

# WR-02 divergence canary: a top-level entry directly under ``knobs:`` (a ``key:`` line at the
# block's child indentation). Counting these with the stdlib and comparing against the strict
# ``_TIER_KNOB_RE`` match count catches PARTIAL divergence — a block-style knob, or a hyphenated
# name the ``[A-Za-z0-9_.]+`` class misses — that the regex would SILENTLY DROP. The key class here
# deliberately INCLUDES the hyphen so such a knob is still counted as declared.
_KNOB_ENTRY_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]+\s*:")
_KNOBS_BLOCK_START_RE = re.compile(r"^knobs:\s*(#.*)?$")

# --- TS-01c typed-state lint constants (typed-state-beats-prose, the operator 2026-07-14) -----------
# Forward-only adoption cutoff (ISO date strings compare lexicographically — no datetime needed).
_TYPED_STATE_CUTOFF = "2026-07-14"
# The adoption entry's title substring — the entry-ORDER anchor. Only entries from this entry
# ONWARD are linted (the adoption entry is the first exemplar and lints itself); several earlier
# entries share the cutoff date but predate the adoption and are exempt.
_ADOPTION_TITLE = "ADOPTED: typed-state-beats-prose"

# Load-bearing-numeric prose patterns: any hit means the entry's numbers must live in a typed
# ``- state:`` block, not only in prose (prose corrupts 36.4% of numerics under cascaded
# compaction vs 9.4% typed — COMPACTION-PROTOCOL.md).
_LOAD_BEARING_PATTERNS = (
    re.compile(r"\$\d"),                                  # dollar figures: $29.52, $300
    re.compile(r"\bstep[ -]0*\d{3,}\b", re.IGNORECASE),   # 'step 3000' / 'step-03000'
    re.compile(r"\b\d{3,}\s+steps\b", re.IGNORECASE),     # '8000 steps'
    re.compile(r"\bloss\s+0\.\d+", re.IGNORECASE),        # 'loss 0.2156'
    re.compile(r"checkpoint-step-\d+"),                    # checkpoint dir paths
)

# A well-formed state line: indented, a bare key token, a colon, a value. Prose sentences
# ("    the run went well and cost a lot") do not parse — that is the point.
_STATE_LINE_RE = re.compile(r"^\s{2,}[A-Za-z0-9_][A-Za-z0-9_.\-]*:\s*\S")

# The source classes a mode-pick's OPTION may carry. The mode is a SELECTION POLICY over
# already-tagged options, never itself a tag (D-20): ``source_tag`` names the OPTION's own class;
# ``[prior]`` and the bare posture words (``draft``/``pro``) are banned — "because draft/pro" is a
# banned untagged model prior (the documented drift failure mode, D-8-SOURCETAG).
_ALLOWED_SOURCE_TAGS = ("[house]", "[precedent]", "[canonical]", "[community]")

# Mode-pick OMISSION catcher (D-20). The ``class: mode-pick`` core is opt-in — an entry can escape
# it by simply not declaring the class (harness_lint docstring, "forward-only" note). So a prose
# detector (mirroring ``_LOAD_BEARING_PATTERNS``) catches a post-cutoff entry that DESCRIBES a
# posture-driven pick but carries no typed block. Patterns are deliberately TIGHT: ``pro`` is a
# substring of process/probe/proven/prompt, so word boundaries + a pick-verb context are required,
# or the retro-lint over post-cutoff entries would fire on unrelated prose (T-09.2-13).
_MODE_PICK_VERB = r"(?:auto-?set|auto-?select(?:ed|s)?|picked|picks|chose|chosen|select(?:ed|s)?)"
_MODE_PICK_WORD = r"(?:draft|pro)"
_MODE_PICK_PROSE_PATTERNS = (
    re.compile(rf"\b{_MODE_PICK_WORD}\b[^.\n]{{0,40}}\b{_MODE_PICK_VERB}\b", re.IGNORECASE),
    re.compile(rf"\b{_MODE_PICK_VERB}\b[^.\n]{{0,40}}\b{_MODE_PICK_WORD}\b", re.IGNORECASE),
    # WR-03: the bare ``optimization_target`` token alone is a mere MENTION — narrative about the
    # axis itself ("the optimization_target axis landed") is discussion, not a posture-DRIVEN pick,
    # so it must not trip the hard-fail omission catcher and force a typed block onto a routine
    # retrospective note. Require the SAME pick-verb proximity the two sibling patterns use, OR an
    # explicit posture ASSIGNMENT in prose (``optimization_target: pro`` / ``= draft``) — i.e. a
    # RECORDED selection that carries no ``- state:`` block. All three carry IGNORECASE for parity
    # with the siblings, so ``Optimization_Target`` no longer slips through.
    re.compile(rf"\boptimization_target\b[^.\n]{{0,40}}\b{_MODE_PICK_VERB}\b", re.IGNORECASE),
    re.compile(rf"\b{_MODE_PICK_VERB}\b[^.\n]{{0,40}}\boptimization_target\b", re.IGNORECASE),
    re.compile(rf"\boptimization_target\b\s*[:=]\s*{_MODE_PICK_WORD}\b", re.IGNORECASE),
)

# D-TS-2 mandatory-slot cores per DECLARED entry class. An entry opts into a class with a
# ``class: <name>`` slot inside its ``- state:`` block; the core keys below must then be present.
_ENTRY_CLASS_CORES: dict[str, tuple[str, ...]] = {
    "run": ("round", "config", "init_adapter", "final_checkpoint", "steps", "final_loss"),
    "verdict": ("round", "checkpoint_judged", "verdict"),
    "incident": ("impact", "detected_via"),
    # D-20 typed shape of a posture-driven pick: the knob, its tier, what was chosen, the OPTION's
    # own source class, and the posture in force. The Tier-2/3 hard-fail rule (D-30 part 3) runs on
    # top of this core — see ``_lint_mode_pick``.
    "mode-pick": ("knob", "tier", "chosen", "source_tag", "optimization_target"),
}


# --------------------------------------------------------------------------- tier taxonomy


def _count_declared_knobs(text: str) -> int:
    """Count the top-level entries directly under ``knobs:`` with a stdlib scan (WR-02 canary).

    A ``key:`` line at the ``knobs:`` block's child indentation is one declared knob; deeper lines
    belong to a block-style knob's own value and are not counted separately. Stdlib-only, because
    the module-level parser confinement rule bars the structured loader here — so this cheap
    divergence canary is how partial regex-vs-structured-parse drift is caught at runtime.
    """
    in_knobs = False
    child_indent: int | None = None
    count = 0
    for line in text.splitlines():
        if not in_knobs:
            if _KNOBS_BLOCK_START_RE.match(line):
                in_knobs = True
            continue
        if line and not line[0].isspace():
            break  # a column-0 line (next top-level key or comment) ends the knobs block
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if child_indent is None:
            child_indent = indent
        if indent == child_indent and _KNOB_ENTRY_KEY_RE.match(stripped):
            count += 1
    return count


def load_tier_map(path: str | Path | None = None) -> dict[str, int]:
    """Return the ``{knob: tier_int}`` partition from TIER-TAXONOMY.yaml via a STDLIB regex.

    Read by regex, NOT ``yaml.safe_load`` — ``test_harness_lint_is_import_confined`` forbids the
    yaml dependency and every ``signet_trainer`` dependency, so this lint stays self-contained. The
    taxonomy's one-line flow-mapping shape (``<knob>: {tier: N, ...}``, tier FIRST) exists so this
    regex and ``yaml.safe_load`` extract byte-identical maps (the dual-parser contract, Plan 01).

    Fails LOUD — the partition IS the safety mechanism (D-23); its absence must never silently
    degrade to "everything is Tier 1":

      * a missing taxonomy file raises ``FileNotFoundError``;
      * a parse that matches ZERO knobs raises ``ValueError`` (a shape change that silently disabled
        the Tier-2/3 rule must fail loudly, not pass).
    """
    taxonomy_path = Path(path) if path is not None else tier_taxonomy_path()
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"tier taxonomy not found: {taxonomy_path}")
    text = taxonomy_path.read_text(encoding="utf-8")
    tier_map: dict[str, int] = {}
    for line in text.splitlines():
        match = _TIER_KNOB_RE.match(line)
        if match:
            tier_map[match.group(1)] = int(match.group(2))
    if not tier_map:
        raise ValueError(
            f"tier taxonomy {taxonomy_path} parsed to ZERO knobs — the one-line flow-mapping shape "
            f"may have changed; the Tier-2/3 mode-pick rule would silently no-op (D-23)"
        )
    # WR-02: the regex silently DROPS any knob not in the one-line flow-mapping shape (a block-style
    # entry, or a hyphenated name the ``[A-Za-z0-9_.]+`` class misses). The old check only fired on
    # ZERO knobs — a PARTIAL divergence (some knobs dropped) passed silently, desyncing the lint's
    # Tier-2/3 safety map from the emitted taxonomy and hard-failing a legitimately Tier-1 knob as
    # "absent". Reconcile the parsed count against the declared entry count at runtime so partial
    # divergence fails LOUD here, not only inside the dual-parser contract test.
    declared = _count_declared_knobs(text)
    if len(tier_map) < declared:
        raise ValueError(
            f"tier taxonomy {taxonomy_path}: the stdlib regex parsed {len(tier_map)} knob(s) but "
            f"{declared} entries are declared under 'knobs:' — a knob is not in the required one-line "
            f"flow-mapping shape ('<knob>: {{tier: N, ...}}', tier FIRST) or uses a name the regex "
            f"misses (e.g. a hyphen). The lint's Tier-2/3 safety map would silently drop it and "
            f"hard-fail it as absent (D-23); fix the taxonomy shape so both parsers agree."
        )
    return tier_map


# --------------------------------------------------------------------------- log splitting


def split_log_entries(text: str) -> list[str]:
    """Split a decision log into per-entry blocks keyed on ``## `` headings (skip the preamble)."""
    entries: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                entries.append("\n".join(current))
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        entries.append("\n".join(current))
    return entries


# --------------------------------------------------------------------------- TS-01c helpers


def entry_date(entry: str) -> str | None:
    """Return the ISO date of a dated ``## YYYY-MM-DD`` entry heading, or None."""
    match = re.match(r"^## (\d{4}-\d{2}-\d{2})", entry)
    return match.group(1) if match else None


def find_adoption_index(entries: list[str]) -> int | None:
    """Index of the typed-state adoption entry (the entry-order cutoff anchor), or None."""
    for idx, entry in enumerate(entries):
        heading = entry.splitlines()[0] if entry else ""
        if heading.startswith("## ") and _ADOPTION_TITLE in heading:
            return idx
    return None


def extract_state_block(entry: str) -> list[str] | None:
    """Return the non-empty indented lines of the first ``- state:`` block, or None if absent.

    The block runs from the ``- state:`` field bullet until the next top-level (column-0) line —
    the next ``- field:`` bullet ends it. An entry WITH a ``- state:`` line but no indented
    ``name: value`` lines returns ``[]`` (present-but-empty, which the lint rejects).
    """
    lines = entry.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("- state:"):
            block: list[str] = []
            for follower in lines[i + 1 :]:
                if follower and not follower[0].isspace():
                    break
                block.append(follower)
            return [ln for ln in block if ln.strip()]
    return None


def strip_state_block(entry: str) -> str:
    """Return the entry's PROSE — every line except the ``- state:`` block itself.

    Numerics inside the typed block must not count as 'numerics in prose'.
    """
    out: list[str] = []
    in_state = False
    for line in entry.splitlines():
        if line.startswith("- state:"):
            in_state = True
            continue
        if in_state:
            if line and not line[0].isspace():
                in_state = False  # column-0 line ends the block; fall through and keep it
            else:
                continue
        out.append(line)
    return "\n".join(out)


def has_load_bearing_numerics(prose: str) -> bool:
    """True if the prose carries any load-bearing numeric pattern ($/steps/loss/checkpoint)."""
    return any(pattern.search(prose) for pattern in _LOAD_BEARING_PATTERNS)


def has_mode_pick_prose(prose: str) -> bool:
    """True if prose describes a posture(draft/pro)-driven pick — the D-20 omission-catcher trigger.

    Kept tight (word boundaries + a pick-verb context, or an explicit ``optimization_target: pro``
    assignment) so it does not fire on ``process``/``probe``/``proven``/``prompt`` (T-09.2-13), nor
    on a bare MENTION of the ``optimization_target`` field in narrative prose (WR-03) — only a
    RECORDED selection trips it.
    """
    return any(pattern.search(prose) for pattern in _MODE_PICK_PROSE_PATTERNS)


def state_slot_names(state_lines: list[str]) -> set[str]:
    """Return the set of slot KEYS from the indented ``name: value`` lines of a state block."""
    names: set[str] = set()
    for line in state_lines:
        stripped = line.strip()
        if ":" in stripped:
            names.add(stripped.split(":", 1)[0].strip())
    return names


def state_slot_values(state_lines: list[str]) -> dict[str, str]:
    """Return ``{key: value}`` from the indented ``name: value`` lines (mirrors state_slot_names).

    The Tier-2/3 rule needs the slot VALUES (which knob was picked, its declared tier, the option's
    source tag), so this reads them with the same ``split(":", 1)`` idiom ``state_slot_names`` uses.
    """
    values: dict[str, str] = {}
    for line in state_lines:
        stripped = line.strip()
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def entry_class(state_lines: list[str]) -> str | None:
    """Return the declared entry class from a ``class: <name>`` slot, lower-cased, or None.

    The class tag is the D-TS-2 opt-in: ``class: run`` / ``class: verdict`` / ``class: incident``
    inside the ``- state:`` block. An unrecognized or absent value means "no declared class" — the
    governing rule still applies, but no mandatory-core check (forward-only, do not over-constrain).
    """
    for line in state_lines:
        stripped = line.strip()
        if stripped.startswith("class:"):
            value = stripped.split(":", 1)[1].strip().lower()
            return value or None
    return None


def _lint_mode_pick(
    heading: str, state_lines: list[str], tier_map: dict[str, int] | None
) -> list[str]:
    """The D-20 / D-30-part-3 rule: a mode-driven pick may only auto-select a Tier-1 knob.

    Runs for a ``class: mode-pick`` entry whose mandatory core is already checked. Resolves the
    entry's declared ``knob`` against ``TIER-TAXONOMY.yaml`` (never against the entry's own ``tier``
    slot) and hard-fails on:

      * a Tier-2 knob — the always-check-in carve-out (D-03/D-04), NEVER mode-auto-selected even
        under yolo+draft;
      * a Tier-3 knob — house-first (D-05); the mode may only reword the recommendation;
      * a knob ABSENT from the taxonomy — treated as Tier 3, the D-06 fail-safe direction;
      * a declared ``tier`` disagreeing with the taxonomy — the entry cannot self-declare a
        convenient tier (the taxonomy is the single source, D-30);
      * a ``source_tag`` of ``[prior]`` or anything outside the four allowed option classes — the
        mode is a selection policy over already-tagged options, never itself a tag (D-20).
    """
    if tier_map is None:
        tier_map = load_tier_map()
    values = state_slot_values(state_lines)
    knob = values.get("knob", "")
    declared_tier_raw = values.get("tier", "")
    source_tag = values.get("source_tag", "")
    violations: list[str] = []

    resolved_tier = tier_map.get(knob)
    if resolved_tier is None:
        violations.append(
            f"{heading}: mode-pick on knob '{knob}' which is ABSENT from TIER-TAXONOMY.yaml — "
            f"treated as Tier 3 (D-06 fail-safe): an unknown knob is house-first, never auto-picked "
            f"by the mode. Add it to the taxonomy with its real tier, or do not auto-pick it."
        )
    elif resolved_tier == 2:
        violations.append(
            f"{heading}: mode-pick on knob '{knob}' which is Tier 2 — the always-check-in carve-out "
            f"(D-03/D-04). A Tier-2 knob is NEVER mode-auto-selected, even under yolo+draft; it must "
            f"be checked in with the operator, house default named."
        )
    elif resolved_tier == 3:
        violations.append(
            f"{heading}: mode-pick on knob '{knob}' which is Tier 3 — house-first (D-05). The mode "
            f"may only reword the recommendation, never auto-pick a learns-side knob."
        )

    # The entry cannot self-declare a convenient tier: its ``tier:`` slot must equal the taxonomy's.
    if resolved_tier is not None:
        declared_tier = int(declared_tier_raw) if declared_tier_raw.isdigit() else None
        if declared_tier != resolved_tier:
            violations.append(
                f"{heading}: mode-pick on '{knob}' declares tier {declared_tier_raw!r} but "
                f"TIER-TAXONOMY.yaml says tier {resolved_tier} — the entry cannot self-declare a "
                f"convenient tier (the taxonomy is the single source, D-30)."
            )

    # ``source_tag`` names the OPTION's class, never the mode. ``[prior]`` and ``draft``/``pro`` are
    # banned — "because draft/pro" is a banned untagged prior (D-20).
    if source_tag not in _ALLOWED_SOURCE_TAGS:
        violations.append(
            f"{heading}: mode-pick source_tag {source_tag!r} is not one of "
            f"{'/'.join(_ALLOWED_SOURCE_TAGS)} — the mode is a selection policy over already-tagged "
            f"options, never itself a tag; 'because draft/pro' is a banned untagged [prior] (D-20)."
        )
    return violations


def lint_typed_state_entry(entry: str, tier_map: dict[str, int] | None = None) -> list[str]:
    """TS-01c lint for one post-adoption dated entry. Returns violation messages (empty = pass).

    Rules: (a) load-bearing numerics in prose require a ``- state:`` block; (a2) prose describing a
    draft/pro-driven pick with no ``- state:`` block hard-fails (the D-20 omission catcher — the
    class core is opt-in, so an undeclared mode-pick would otherwise escape); (b) a present block
    must carry at least one line; (c) every block line must parse as simple indented
    ``name: value`` — no prose sentences inside the block; (d) an entry DECLARING its class
    (``class: run|verdict|incident|mode-pick``) must carry that class's mandatory core slots
    (D-TS-2); (e) a ``mode-pick`` entry must additionally pass the Tier-2/3 rule (D-30 part 3) —
    resolved against ``TIER-TAXONOMY.yaml`` via ``tier_map`` (loaded lazily if not supplied).
    """
    heading = entry.splitlines()[0]
    violations: list[str] = []
    state_lines = extract_state_block(entry)
    prose = strip_state_block(entry)
    if has_load_bearing_numerics(prose) and state_lines is None:
        violations.append(
            f"{heading}: load-bearing numerics in prose but no '- state:' typed block "
            f"(typed-state-beats-prose — COMPACTION-PROTOCOL.md)"
        )
    if has_mode_pick_prose(prose) and state_lines is None:
        violations.append(
            f"{heading}: prose describes a draft/pro-driven pick but carries no '- state:' typed "
            f"block — every mode-driven pick lands as a typed 'class: mode-pick' entry (D-20)."
        )
    if state_lines is not None:
        if not state_lines:
            violations.append(f"{heading}: '- state:' block carries no 'name: value' lines")
        for line in state_lines:
            if not _STATE_LINE_RE.match(line):
                violations.append(
                    f"{heading}: state line is not a simple indented 'name: value': {line!r}"
                )
        # D-TS-2 mandatory-slot core — only for entries that DECLARE their class (forward-only).
        cls = entry_class(state_lines)
        if cls in _ENTRY_CLASS_CORES:
            present = state_slot_names(state_lines)
            missing = [key for key in _ENTRY_CLASS_CORES[cls] if key not in present]
            if missing:
                violations.append(
                    f"{heading}: '{cls}'-class entry missing mandatory state slot(s): "
                    f"{', '.join(missing)} (D-TS-2 mandatory core — COMPACTION-PROTOCOL.md)"
                )
            # D-30 part 3 — the phase's genuine behavioral seam: a mode-driven pick may only
            # auto-select a Tier-1 knob. Resolves the tier from the taxonomy, not the entry.
            if cls == "mode-pick":
                violations.extend(_lint_mode_pick(heading, state_lines, tier_map))
    return violations


def lint_typed_state_log(text: str, tier_map: dict[str, int] | None = None) -> list[str]:
    """Run the TS-01c lint over a whole DECISION-LOG text; entry-ORDER-aware forward-only cutoff.

    Lints the adoption entry itself (the first exemplar) and every dated entry after it whose
    date is >= the cutoff. Entries BEFORE the adoption entry are exempt even on the same date
    (the adoption landed mid-day 2026-07-14 — several earlier same-date entries legitimately
    have no state block). Older entries are never retrofitted (forward-only law).

    The tier map (for the ``mode-pick`` Tier-2/3 rule) is loaded ONCE here and threaded down —
    fail-loud on a missing/zero-knob taxonomy (D-23), so a whole-log lint can never silently skip
    the safety mechanism.
    """
    entries = split_log_entries(text)
    adoption_idx = find_adoption_index(entries)
    if adoption_idx is None:
        return [f"adoption entry ({_ADOPTION_TITLE!r}) not found in the decision log"]
    if tier_map is None:
        tier_map = load_tier_map()
    violations: list[str] = []
    for entry in entries[adoption_idx:]:
        date = entry_date(entry)
        if date is None or date < _TYPED_STATE_CUTOFF:
            continue
        violations.extend(lint_typed_state_entry(entry, tier_map))
    return violations


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m signet_trainer.harness_lint",
        description=(
            "Hard-fail TS-01c typed-state lint over a DECISION-LOG: post-adoption entries with "
            "load-bearing prose numerics must carry a typed '- state:' block; declared "
            "run/verdict/incident entries must carry their D-TS-2 mandatory core slots."
        ),
    )
    parser.add_argument(
        "--log",
        default=None,
        help=f"path to the decision log to lint (default: {DEFAULT_LOG_PATH} under the project "
        f"root — see {PROJECT_ROOT_ENV_VAR})",
    )
    args = parser.parse_args(argv)

    root = project_root()
    log_path = Path(args.log) if args.log else root / DEFAULT_LOG_PATH
    if not log_path.is_file():
        # LIVE per-project state: never shipped with the package, so an absent log means "no
        # project here", not "broken install". Say which, and how to fix it.
        print(
            f"[harness_lint] decision log not found: {log_path}\n"
            f"    resolved against project root {root} (the DECISION-LOG is LIVE per-project state "
            f"and is deliberately NOT shipped with the package).\n"
            f"    fix: run this from inside a signet-trainer project (a directory carrying "
            f"{PLANNING_HARNESS_DIR}/), set {PROJECT_ROOT_ENV_VAR}=/path/to/project, or pass "
            f"--log /path/to/DECISION-LOG.md.",
            file=sys.stderr,
        )
        return 2

    violations = lint_typed_state_log(log_path.read_text(encoding="utf-8"))
    if violations:
        print(f"[harness_lint] {len(violations)} typed-state violation(s) in {log_path}:")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print(f"[harness_lint] clean: no typed-state violations in {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
