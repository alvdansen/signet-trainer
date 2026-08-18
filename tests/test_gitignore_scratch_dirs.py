"""Issue #40 finding 1 — the scratch-dir ignore globs must cover the BARE names the skills fetch
into, not only a name with a trailing underscore.

**The defect.** `.gitignore` had `_samples_*/` and `_grid_*/` (and `_reskin_grid_*/`), each of which
requires a literal `_` before the glob star. Every shipped fetch command names the BARE directory —
`training-review/SKILL.md:104,161` and `training-run/SKILL.md:384` all fetch into `./_samples`, not
`./_samples_anything` — so the directory every playbook actually creates was git-TRACKABLE. What
lands there is not only media: `fns.py` writes `_samples/index.html` (every validation prompt,
verbatim) and `_samples/delta.json` (`subject_id`s), and neither is caught by the media-extension
block, so `git status` looked clean while the two files stayed trackable.

This test walks the LITERAL scratch paths the skills instruct (derived from the skill files
themselves, not hand-copied) plus the sibling dirs the audit swept as already-correct, and asserts
`git check-ignore` matches every one of them — so a future doc naming a new scratch dir fails here
instead of leaking.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SKILLS_DIR = _ROOT / ".claude" / "skills"

# Every shipped `modal volume get ... <local-dir>` fetch target in the skills. Extracted by
# pattern rather than hand-typed so a new skill naming a new scratch dir is picked up here too.
_FETCH_RE = re.compile(r"modal volume get\s+\S+\s+\S+\s+(\./\S+)")


def _skill_fetch_targets() -> set[str]:
    targets: set[str] = set()
    for md in _SKILLS_DIR.rglob("SKILL.md"):
        for m in _FETCH_RE.finditer(md.read_text(encoding="utf-8")):
            targets.add(m.group(1).rstrip("/`\"'"))
    return targets


#: Sibling scratch dirs the audit re-verified as already correctly ignored today (bundle #40,
#: finding 1's "sweep the already-correct ones into the same test").
_ALREADY_CORRECT = [
    "_inpaint_prep/x.txt",
    "_staging_test/x.txt",
    "_h3_status/x.txt",
    "_grid_relay.log",
    ".venv-smoke/x.txt",
    "_tools/x.txt",
]

#: Diverges BY DESIGN — its own explicit non-glob line, never the samples/grid family.
_DESIGN_DIVERGENCE = "scripts/_seg_preview/x.txt"


def _git_check_ignore(*paths: str) -> tuple[int, str]:
    p = subprocess.run(
        ["git", "check-ignore", "-v", *paths],
        cwd=_ROOT, capture_output=True, text=True,
    )
    return p.returncode, p.stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not on PATH")
def test_skill_fetch_targets_exist_and_are_named_bare() -> None:
    """Sanity on the extraction itself: the skills must still name `./_samples` (bare)."""
    targets = _skill_fetch_targets()
    assert targets, (
        "no 'modal volume get ... ./_...' fetch command found in any SKILL.md — the "
        "extraction regex or the skills moved; re-derive before trusting this test"
    )
    assert "./_samples" in targets, (
        f"expected the bare './_samples' fetch target among {sorted(targets)} — if the skills now "
        f"fetch into a different bare name, update this test's expectation, not the regex"
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not on PATH")
def test_every_skill_fetch_target_is_git_ignored() -> None:
    """Every literal dir a playbook fetches INTO must be un-trackable — a probe file inside it too."""
    targets = sorted(_skill_fetch_targets())
    probes = [f"{t.lstrip('./')}/x.txt" for t in targets]
    rc, out = _git_check_ignore(*probes)
    assert rc == 0, (
        f"the following skill fetch targets are NOT git-ignored: {probes}. A `git check-ignore` "
        f"miss here means a future `git add -A` will track validation prompts / subject_ids that "
        f"land in that directory (issue #40 finding 1). check-ignore output:\n{out}"
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not on PATH")
def test_bare_grid_and_reskin_grid_dirs_are_git_ignored() -> None:
    """`_grid/` and `_reskin_grid/` (no trailing underscore) — the same off-by-one as `_samples_*/`."""
    rc, out = _git_check_ignore("_grid/x.txt", "_reskin_grid/x.txt")
    assert rc == 0, f"bare _grid/ or _reskin_grid/ is trackable again: {out}"


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not on PATH")
def test_sibling_scratch_dirs_already_correct_stay_ignored() -> None:
    """Regression guard on the dirs the audit found already-correct, swept into this test."""
    rc, out = _git_check_ignore(*_ALREADY_CORRECT)
    assert rc == 0, f"a previously-correct scratch pattern regressed: {out}"


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not on PATH")
def test_seg_preview_ignored_by_its_own_explicit_line() -> None:
    """`scripts/_seg_preview/` diverges by design — pinned separately so nobody folds it into the
    samples/grid family and accidentally narrows it."""
    rc, out = _git_check_ignore(_DESIGN_DIVERGENCE)
    assert rc == 0, f"scripts/_seg_preview/ is no longer ignored: {out}"
    assert "scripts/_seg_preview/" in out, (
        f"scripts/_seg_preview/ is ignored by a DIFFERENT rule than its own explicit line; "
        f"got: {out}"
    )
