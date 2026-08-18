"""#12 + #22 finding 5 — the H3 base render is keyed OUT of the checkpoint scope, on the FULL pipe
parameter identity.

THE DEFECT. ``h3_sample`` renders base-vs-adapter from ONE transformer, but the base column runs
under ``disable_adapter()`` — it depends only on the pipe parameters (``seed``, ``frame_count``,
``width``, ``height``, ``num_inference_steps``, the reference condition), never on the checkpoint
being compared against it. Keying the base render's directory on the SAME identity as the adapter
column (which correctly DOES include the checkpoint) makes a byte-identical base clip re-render at
every sampled checkpoint — ~50% of H3 sampling spend at the keyframe campaign's cadence, and the
render backlog it produced once starved a training phase of GPU slots for two-plus hours (#12).

THE ADVERSARIAL REFUTATION. A naive fix keys the base render on ``(probe, seed, frame_count)`` and
stops there. #22's audit explicitly refuted that: the ``pipe(...)`` call also receives ``width`` /
``height`` / ``num_inference_steps``, so a key missing any of the three would silently resume a base
clip rendered at a DIFFERENT geometry (a resolution or step-count probe) under the new banner — the
same silent-at-a-valid-shape class the reference axis exists to prevent, just relocated to the base
column. Every test below that asserts "changing X changes the key" is pinning one axis of that
refutation; delete none of them without deleting the axis they guard from the render call itself.

Zero GPU, zero Modal, zero spend: pure string functions plus a source/regex scan of the render
directory this fix actually touches.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from signet_trainer.inference.render_key import h3_base_render_key, h3_render_key
from signet_trainer.inference.samples_layout import (
    _H3_KEY_RE,
    expected_h3_base_render_key,
    expected_h3_render_key,
)

REPO = Path(__file__).resolve().parents[1]
_FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"

#: A canonical (seed, frames, width, height, steps, subject_ids) request, and the six single-axis
#: perturbations #22 finding 5 requires the key to distinguish. "checkpoint" is included in the
#: perturbation set because ``h3_render_key`` (the ADAPTER key) must still vary on it — the base
#: key's whole point is that it must NOT.
_BASE = dict(seed=42, frame_count=22, width=1344, height=768, num_inference_steps=25,
             subject_ids=["A", "029"])


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# render_key.h3_base_render_key — the identity itself
# ══════════════════════════════════════════════════════════════════════════════════════════════════


def test_the_base_key_has_no_checkpoint_parameter_at_all() -> None:
    """Structural: a checkpoint kwarg would let a caller accidentally re-couple the two.

    The function must not even ACCEPT a checkpoint — the whole fix is that the base render's
    identity cannot vary on it, and a silently-ignored kwarg is worse than a missing one.
    """
    import inspect

    params = inspect.signature(h3_base_render_key).parameters
    assert "checkpoint" not in params, (
        "h3_base_render_key accepts a checkpoint kwarg. #12's fix is that the base render's "
        "identity does not depend on the checkpoint at all; an accepted-but-unused kwarg invites "
        "a caller to pass one back in and re-open the re-render bug."
    )


@pytest.mark.parametrize(
    "changed_kwarg,changed_value",
    [
        ("seed", 43),
        ("frame_count", 56),
        ("width", 1344 + 32),
        ("height", 768 + 32),
        ("num_inference_steps", 26),
        ("subject_ids", ["B", "029"]),
    ],
)
def test_every_pipe_parameter_changes_the_base_key(changed_kwarg: str, changed_value) -> None:
    """The adversarial refutation, made concrete: each of the SIX axes the pipe call reads must be
    load-bearing in the base key, or that axis's change silently reuses a stale base clip."""
    reference = h3_base_render_key(**_BASE)
    perturbed = h3_base_render_key(**{**_BASE, changed_kwarg: changed_value})
    assert reference != perturbed, (
        f"changing {changed_kwarg!r} did not change h3_base_render_key's output — a probe that "
        f"differs ONLY on this axis would silently resume the wrong base clip."
    )


def test_the_same_five_parameters_produce_the_same_base_key_every_time() -> None:
    """The flip side: dedup only works if identical requests are actually identical keys."""
    assert h3_base_render_key(**_BASE) == h3_base_render_key(**_BASE)


def test_the_base_key_never_collides_with_the_checkpoint_scoped_key() -> None:
    """The base key must never be mistaken for an adapter (checkpoint-scoped) render's key by the
    SAME regex the render-key module ships (``_H3_KEY_RE``) — no special-casing required downstream."""
    base_key = h3_base_render_key(**_BASE)
    assert not _H3_KEY_RE.match(base_key), (
        f"the base key {base_key!r} matches the checkpoint-scoped _H3_KEY_RE. A watcher or grid "
        f"script listing a directory of both kinds would then misparse the base dir as a "
        f"nonsensical checkpoint render."
    )


def test_the_base_key_is_a_safe_and_readable_directory_name() -> None:
    key = h3_base_render_key(**_BASE)
    assert key == "s42_f22_w1344_h768_n25_A-029", key
    assert "/" not in key and "\\" not in key and ".." not in key


def test_the_base_key_carries_every_axis_h3_render_key_does_except_checkpoint() -> None:
    """Both keys must agree on the shared five axes byte-for-byte (minus the checkpoint segment),
    or the two functions could describe requests that are not actually the same pipe call."""
    adapter_key = h3_render_key(checkpoint="checkpoint-step-3000", **_BASE)
    base_key = h3_base_render_key(**_BASE)
    assert adapter_key == f"checkpoint-step-3000_{base_key}", (
        f"adapter key {adapter_key!r} does not decompose as '<checkpoint>_' + the base key "
        f"{base_key!r} — the two functions have drifted apart on the axes they share."
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# samples_layout — the watcher/grid-facing wrappers, and the widened checkpoint-scoped regex
# ══════════════════════════════════════════════════════════════════════════════════════════════════


def test_expected_h3_base_render_key_delegates_never_reimplements() -> None:
    kwargs = dict(seed=42, frame_count=56, width=1024, height=576, num_inference_steps=30,
                  subject_ids=["C", "018"])
    assert expected_h3_base_render_key(**kwargs) == h3_base_render_key(**kwargs)


def test_expected_h3_render_key_still_delegates_after_widening() -> None:
    kwargs = dict(checkpoint="checkpoint-step-03000-loss-0.1933", seed=42, frame_count=56,
                  width=1024, height=576, num_inference_steps=30, subject_ids=["C", "018"])
    assert expected_h3_render_key(**kwargs) == h3_render_key(**kwargs)


@pytest.mark.parametrize(
    "changed_kwarg,changed_value",
    [("width", 1376), ("height", 800), ("num_inference_steps", 26)],
)
def test_the_checkpoint_scoped_key_also_gained_the_geometry_axes(changed_kwarg, changed_value) -> None:
    """The OTHER half of #22 finding 5: the adapter/checkpoint-scoped key widened too, not only the
    new base key — a stale-geometry resume is exactly as possible on the lora column."""
    base = dict(checkpoint="checkpoint-step-3000", seed=42, frame_count=22, width=1344, height=768,
                num_inference_steps=25, subject_ids=["A", "029"])
    reference = h3_render_key(**base)
    perturbed = h3_render_key(**{**base, changed_kwarg: changed_value})
    assert reference != perturbed, (
        f"changing {changed_kwarg!r} did not change h3_render_key's output"
    )


def test_h3_key_regex_requires_the_widened_tail() -> None:
    """The pre-fix key format (no w/h/n segments) must no longer be recognised as a valid render
    dir — a stale entry from before this fix should read as unrecognised, not silently accepted."""
    old_style = "checkpoint-step-3000_s42_f22_A-029"
    assert not _H3_KEY_RE.match(old_style), (
        "the widened _H3_KEY_RE still matches the pre-#22-finding-5 key shape — either the widening "
        "did not land or the regex is not anchored on the new tail."
    )
    new_style = h3_render_key(checkpoint="checkpoint-step-3000", seed=42, frame_count=22, width=1344,
                              height=768, num_inference_steps=25, subject_ids=["A", "029"])
    assert _H3_KEY_RE.match(new_style)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# fns.py source — the base dir must be composed OUTSIDE the checkpoint-scoped samples_root
# ══════════════════════════════════════════════════════════════════════════════════════════════════


def _h3_sample_source() -> str:
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "h3_sample"
    )
    return ast.get_source_segment(_FNS.read_text(encoding="utf-8"), node) or ""


def test_h3_sample_computes_a_base_key_via_the_shared_function() -> None:
    src = _h3_sample_source()
    assert "h3_base_render_key(" in src, (
        "h3_sample no longer calls h3_base_render_key — the base dir must be derived from the "
        "same stdlib function the tests above pin, never re-implemented inline."
    )


def test_h3_sample_never_nests_base_under_the_checkpoint_scoped_samples_root() -> None:
    """The actual #12 fix: ``base_dir`` must be composed from ``samples_h3_root`` (checkpoint-free),
    not from ``samples_root`` (checkpoint-keyed) — a base dir nested under samples_root is the exact
    bug this whole file exists to catch, re-armed with a wider key that no longer helps."""
    src = _h3_sample_source()
    base_dir_line = next(
        line for line in src.splitlines() if line.strip().startswith("base_dir = ")
    )
    assert "samples_root" not in base_dir_line or "samples_h3_root" in base_dir_line, (
        f"base_dir is composed from {base_dir_line.strip()!r} — it must be built on "
        f"samples_h3_root (checkpoint-independent), not samples_root (checkpoint-keyed)."
    )
    assert "samples_h3_root" in base_dir_line and '"base"' in base_dir_line, (
        f"base_dir line {base_dir_line.strip()!r} does not compose samples_h3_root / 'base' / "
        f"<base key> — the sibling layout #12 requires."
    )


def test_h3_sample_still_reuses_the_existing_resume_skip_check() -> None:
    """#12's proposed direction: reuse `_render`'s existing skip check, not a new one. The base
    column loop must still route through `_render`, which is what makes an already-shared base clip
    a no-op the second (and every subsequent) time a checkpoint's render resolves it."""
    src = _h3_sample_source()
    base_loop = src[src.index("with adapted.disable_adapter():"):src.index("lora_mp4s: dict")]
    assert "_render(prompt, base_dir" in base_loop, (
        "the base column no longer calls the shared _render helper — #12's fix must reuse the "
        "existing 'already rendered, skip' check, not add a second one."
    )


def test_h3_sample_base_mp4_paths_are_relative_to_samples_root_not_base_relative() -> None:
    """The montage HTML lives at samples_root/index.html; base clips now live OUTSIDE samples_root
    (sibling base/<key>/), so a bare 'base/<name>' path (correct before this fix) would 404."""
    src = _h3_sample_source()
    assign = next(
        line for line in src.splitlines() if "base_mp4s[prompt] = " in line
    )
    assert "../base/" in assign, (
        f"{assign.strip()!r} does not point out of samples_root at the sibling base/ dir — the "
        f"montage's base column would 404 against the new layout."
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# scripts/_h3_grid_serve.py — the grid assembler must resolve the shared base column, not a nested one
# ══════════════════════════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def grid_serve():
    """Import the script by file path — it is deliberately a script, not a package module, and has
    no import-time side effects (argparse/Modal calls live inside main())."""
    spec = importlib.util.spec_from_file_location(
        "_h3_grid_serve_under_test", REPO / "scripts" / "_h3_grid_serve.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_grid_serve_key_regex_matches_the_widened_render_key(grid_serve) -> None:
    name = h3_render_key(checkpoint="checkpoint-step-3000", **_BASE)
    assert grid_serve._KEY_RE.match(name), (
        "scripts/_h3_grid_serve.py's own _KEY_RE (duplicated from samples_layout for the same "
        "Anti-Pattern-6 import-isolation reason) was not widened alongside it."
    )


def test_grid_serve_parses_a_base_dir_name_and_never_a_checkpoint_key_as_one(grid_serve) -> None:
    base_name = h3_base_render_key(**_BASE)
    parsed = grid_serve.parse_base_render_key(base_name)
    assert parsed is not None and parsed["subject_ids"] == ["A", "029"]
    ckpt_name = h3_render_key(checkpoint="checkpoint-step-3000", **_BASE)
    assert grid_serve.parse_base_render_key(ckpt_name) is None, (
        "the base-key parser accepted a checkpoint-scoped render key — the two formats must stay "
        "mutually exclusive or a checkpoint dir could be mis-read as the shared base dir."
    )


def test_grid_serve_resolves_the_same_base_key_a_render_produces(grid_serve) -> None:
    """The join key: given a checkpoint-scoped render's parsed metadata, base_key_for must return
    EXACTLY the base directory h3_sample actually wrote — never a re-derived approximation."""
    ckpt_name = h3_render_key(checkpoint="checkpoint-step-3000", **_BASE)
    meta = grid_serve.parse_render_key(ckpt_name)
    assert grid_serve.base_key_for(meta) == h3_base_render_key(**_BASE)


def test_grid_serve_fetch_never_assumes_a_base_subdir_nested_under_a_checkpoint_key(grid_serve) -> None:
    """Source scan: fetch_renders must not compose a remote path of the shape
    '<samples_root>/<checkpoint key>/base/...' — that nesting no longer exists on the Volume."""
    src = (REPO / "scripts" / "_h3_grid_serve.py").read_text(encoding="utf-8")
    fetch_src = src[src.index("def fetch_renders"):src.index("def _safe_label")]
    assert '{samples_root}/{key}/base' not in fetch_src, (
        "fetch_renders still fetches a 'base' subdir nested under each checkpoint-keyed render dir "
        "— that layout no longer exists; the shared base/<base key>/ dir must be fetched separately."
    )
    assert "_BASE_DIRNAME" in fetch_src, (
        "fetch_renders no longer fetches the shared base/ directory at all."
    )
