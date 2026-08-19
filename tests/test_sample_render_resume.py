"""LTX ``sample`` renders are RESUMABLE (issue #45 PR-2) — identity-keyed dirs, skip guards, per-clip
commits.

The pre-PR-2 shape: ``sample`` carried ``retries=3`` while keying five of its seven render branches
on the WALL CLOCK (``strftime``) and committing ONCE at the end of each branch. A server-side retry
of a preempted render therefore booted a fresh container, computed a NEW timestamp dir, found
nothing to skip, and re-rendered every clip from zero (re-paying both 22B loads) — on a preemption
cycle shorter than the render the run could NEVER finish, and the retry budget multiplied the burn
4x instead of self-healing it.

The fix ports ``h3_sample``'s resume shape (D-10): key the dir on WHAT is rendered
(``render_key.ltx_render_key``), skip clips already complete on the Volume, commit per clip. The two
step-keyed branches (inpaint / a2v) were already identity-stable but still re-rendered everything on
retry — they get the skip guard + per-clip commit too.

A verifier previously found the FIRST cut of this test file itself structurally hollow: nothing
called the resume predicate twice and asserted the second pass skips, and every save/guard weld was
an aggregate COUNT rather than tied to actual control flow (a count is gameable — a guard sitting
next to an UNRELATED save still satisfies it). This version fixes both:

  * ``clip_already_rendered`` (the resume predicate) is a real, stdlib-only, independently callable
    function — ``fns.py``'s ``_clip_done`` is a plain alias to it, never a re-implementation — so the
    tests below that write real files and call it twice are exercising the ACTUAL predicate every
    LTX render site guards its save on, not a parallel stand-in for it.
  * the AST checks below are scoped to each ``if _clip_done(...):`` node's own body/orelse, asserting
    the SPECIFIC control-flow shape (skip via ``continue``, or skip-vs-render via if/else with
    ``save_video`` in the render branch) rather than counting unrelated tokens across the whole
    function body.

Pure CPU tests — real file I/O in ``tmp_path``, plus stdlib ``ast``/``re`` source scans. NO modal
import (Anti-Pattern 6): importing ``fns`` would drag ``modal`` into ``sys.modules`` for the whole
pytest session and break the dry-run gate's purity assertion elsewhere.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from signet_trainer.inference.render_key import clip_already_rendered, ltx_render_key

_ROOT = Path(__file__).resolve().parents[1]
_FNS = _ROOT / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _sample_body() -> str:
    """The source of ``def sample(...)`` up to the next module-level def, comments stripped."""
    code = _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))
    start = re.search(r"\ndef\s+sample\s*\(", code)
    assert start is not None, "fns.py must define sample()"
    tail = code[start.end():]
    end = re.search(r"\n(?:@|def\s)", tail)  # next top-level decorator/def
    return tail[: end.start()] if end else tail


def _sample_ast() -> ast.FunctionDef:
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "sample"
    )
    return node


# ---------------------------------------------------------------------------
# ltx_render_key — the identity, as a directory name (real function calls)
# ---------------------------------------------------------------------------

_AXES = dict(
    checkpoint="checkpoint-step-3000-loss-0.36",
    seed=42,
    frame_count=81,
    width=768,
    height=512,
    num_inference_steps=30,
    guidance_scale=3.0,
    stg_scale=1.0,
    condition="ic_lora",
)


def test_ltx_render_key_carries_all_nine_axes() -> None:
    key = ltx_render_key(**_AXES)
    assert key == "checkpoint-step-3000-loss-0.36_s42_f81_w768_h512_n30_g3_st1_ic_lora"


def test_ltx_render_key_new_checkpoint_never_resumes_into_an_old_dir() -> None:
    """The checkpoint axis is load-bearing: a fresh checkpoint's grid gets a fresh directory."""
    old = ltx_render_key(**{**_AXES, "checkpoint": "checkpoint-step-1000"})
    new = ltx_render_key(**{**_AXES, "checkpoint": "checkpoint-step-2000"})
    assert old != new


def test_ltx_render_key_is_filesystem_safe() -> None:
    """A checkpoint name derived from a Volume path must not escape the samples directory."""
    key = ltx_render_key(**{**_AXES, "checkpoint": "../..//evil name"})
    assert "/" not in key and "\\" not in key
    assert key.startswith(".._..__evil_name_")


def test_ltx_render_key_every_geometry_and_sampling_axis_distinguishes_sibling_configs() -> None:
    """⛔ THE COLLISION GUARD a settings sweep depends on. A resolution/steps/guidance/stg probe at a
    fixed checkpoint+seed+frames must land in ITS OWN directory, or ``_clip_done`` would skip every
    clip against the OLD settings' output and the banner would claim the NEW settings for pixels it
    never rendered — silent, at a valid shape, on exactly the axis a sweep exists to vary."""
    base = ltx_render_key(**_AXES)
    for axis, bump in (
        ("frame_count", 40),
        ("width", 32),
        ("height", 32),
        ("num_inference_steps", 1),
        ("guidance_scale", 0.5),
        ("stg_scale", 0.5),
    ):
        changed = ltx_render_key(**{**_AXES, axis: _AXES[axis] + bump})
        assert changed != base, f"{axis} is not in the key — a probe on this axis would collide"
    changed_condition = ltx_render_key(**{**_AXES, "condition": "single_frame"})
    assert changed_condition != base, "condition is not in the key — sibling modes would collide"


# ---------------------------------------------------------------------------
# clip_already_rendered — the resume PREDICATE, exercised for real (behavioral, not structural)
# ---------------------------------------------------------------------------


def test_clip_already_rendered_is_false_for_a_missing_file(tmp_path) -> None:
    assert clip_already_rendered(tmp_path / "never_written.mp4") is False


def test_clip_already_rendered_rejects_a_zero_byte_file(tmp_path) -> None:
    """A container killed mid-``save_video`` leaves a 0-byte file — skipping it would ship a
    corrupt clip in the grid instead of re-rendering it."""
    path = tmp_path / "half_written.mp4"
    path.touch()
    assert path.stat().st_size == 0
    assert clip_already_rendered(path) is False


def test_clip_already_rendered_second_pass_RESUMES(tmp_path) -> None:
    """⛔ THE BEHAVIORAL REGRESSION TEST a prior verifier found missing: call the REAL resume
    predicate twice against the SAME path, simulating exactly what a server-side retry (or manual
    re-dispatch) does — land in the identical identity-keyed directory and ask "is this clip done?"

    Pass 1 (nothing rendered yet): False -> the render site would render + save + commit.
    Pass 2 (after that save): True -> the render site would SKIP, which is the entire point of the
    fix — a retry resumes instead of re-paying the render.
    """
    render_dir = tmp_path / ltx_render_key(**_AXES)
    render_dir.mkdir(parents=True)
    clip_path = render_dir / "a_prompt_s42.mp4"

    # Pass 1 — nothing on the "Volume" yet.
    assert clip_already_rendered(clip_path) is False

    # The render site's save_video(...) + checkpoints_vol.commit() equivalent: write real bytes.
    clip_path.write_bytes(b"\x00" * 4096)

    # Pass 2 — a retry (or re-dispatch) lands in the SAME dir and must see the clip as done.
    assert clip_already_rendered(clip_path) is True, (
        "the resume predicate must return True once the clip is non-empty on disk — a retry that "
        "still saw False here would re-render a clip that already exists (PR-2's whole point)"
    )


# ---------------------------------------------------------------------------
# sample() — resume structure, scoped to actual control flow (not aggregate counts)
# ---------------------------------------------------------------------------


def test_sample_never_keys_a_render_dir_on_the_wall_clock() -> None:
    """No ``strftime`` in the sample body — a wall-clock dir makes every retry a fresh restart."""
    body = _sample_body()
    assert "strftime" not in body, (
        "sample() must key render dirs on the render identity (ltx_render_key), never the wall "
        "clock — a retry into a new ts dir re-renders every clip and strands the previous "
        "attempt's output (PR-2)"
    )


def test_sample_keys_all_five_formerly_ts_branches_on_the_render_identity() -> None:
    """single_frame / multi_frame / ic_lora_baseline / ic_lora / base-vs-LoRA all use the key."""
    body = _sample_body()
    assert len(re.findall(r"ltx_render_key\(", body)) >= 5, (
        "all five formerly ts-keyed render branches must build their samples dir via "
        "ltx_render_key(...) (PR-2 resume)"
    )


def test_sample_guard_delegates_to_the_shared_predicate_not_a_reimplementation() -> None:
    """``_clip_done`` must be the SAME function the behavioral tests above exercise — a local
    re-implementation could silently drift from ``clip_already_rendered`` (e.g. drop the 0-byte
    check) without any test here catching it."""
    body = _sample_body()
    assert re.search(r"_clip_done\s*=\s*clip_already_rendered\b", body), (
        "sample()'s _clip_done must be an alias for render_key.clip_already_rendered, never a "
        "re-implementation of the same predicate"
    )


def _clip_done_if_nodes(fn: ast.FunctionDef) -> list[ast.If]:
    """Every ``ast.If`` in ``sample()`` whose test calls ``_clip_done(...)``."""
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "_clip_done"
    ]


def _contains_call(nodes: list[ast.stmt], func_name: str) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == func_name
        for outer in nodes
        for n in ast.walk(outer)
    )


def _contains_continue(nodes: list[ast.stmt]) -> bool:
    return any(isinstance(n, ast.Continue) for outer in nodes for n in ast.walk(outer))


def test_every_clip_done_guard_has_the_correct_skip_or_render_shape() -> None:
    """⛔ THE WELD a verifier found missing: not "does save_video appear somewhere in the function"
    but "does THIS SPECIFIC guard's own body/orelse do the right thing".

    Two accepted shapes, matching the two branch families in sample():
      * loop-style (inpaint/a2v): ``if _clip_done(...): print(...); continue`` — no orelse, body
        exits the loop iteration without rendering.
      * if/else style (single_frame/multi_frame/ic_lora*/base-vs-LoRA): the ``orelse`` branch (the
        NOT-done path) must itself call ``save_video`` — proving the render only happens when the
        guard is false.
    Anything else (an empty orelse with no continue, or an orelse with no save_video) means the
    guard exists but does not actually gate a render — exactly what a bare count could miss.
    """
    fn = _sample_ast()
    guards = _clip_done_if_nodes(fn)
    assert len(guards) == 10, f"expected the 10 known _clip_done guards, found {len(guards)}"
    for node in guards:
        if node.orelse:
            assert _contains_call(node.orelse, "save_video"), (
                f"line {node.lineno}: the guard's orelse (not-yet-rendered path) must call "
                f"save_video — an orelse that never renders makes the guard a no-op"
            )
        else:
            assert _contains_continue(node.body), (
                f"line {node.lineno}: a guard with no orelse must skip via `continue` in its body "
                f"— otherwise the loop falls through into the render code unconditionally"
            )


def test_every_save_video_call_sits_behind_exactly_one_clip_done_guard() -> None:
    """The inverse direction: every ``save_video`` call must be reachable only through a guard's
    orelse (or, for the two loop-style branches, gated by a preceding sibling guard's ``continue``).
    Counts must match 1:1 — a save_video with NO guard anywhere near it would re-render on every
    retry; a guard with no matching save_video (checked above) would be a guard over nothing.
    """
    body = _sample_body()
    saves = len(re.findall(r"save_video\(", body))
    assert saves == 10, f"expected the 10 known sample render sites, found {saves} — re-audit the weld"
    guards = len(re.findall(r"_clip_done\(", body))
    assert guards == 10, f"expected 10 _clip_done call sites, found {guards}"


def test_every_clip_done_guard_has_a_per_clip_commit_on_its_render_path() -> None:
    """commit-or-vanish, scoped per guard: the if/else guards' orelse (and the loop-style guards'
    surrounding body, past the ``continue``) must commit the checkpoints Volume — a save without a
    nearby commit is a clip that vanishes with the container on a preemption."""
    fn = _sample_ast()
    for node in _clip_done_if_nodes(fn):
        if node.orelse:
            assert any(
                isinstance(n, ast.Attribute) and n.attr == "commit"
                for stmt in node.orelse
                for n in ast.walk(stmt)
            ), f"line {node.lineno}: the render orelse must commit the checkpoints Volume per clip"


def test_per_clip_and_terminal_commits_total_the_expected_count() -> None:
    """10 per-clip commits (one per guarded save_video) + 7 branch-terminal commits (index.html /
    gallery durability across the seven return points: inpaint, a2v, single_frame, multi_frame,
    ic_lora_baseline, ic_lora, base-vs-LoRA tail)."""
    body = _sample_body()
    commits = len(re.findall(r"checkpoints_vol\.commit\(\)", body))
    assert commits >= 17, (
        f"expected >= 17 checkpoints_vol.commit() sites in sample() (10 per-clip + 7 terminal), "
        f"found {commits} — a branch lost its per-clip or terminal durability (PR-2)"
    )
