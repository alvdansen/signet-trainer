"""``h3_sample`` is resumable — and the resume key cannot collide across the sibling configs.

A 12-clip sequential render that commits only at the END is a ~6 h loss unit with no retries and no
resume. That is the shape ``KNOWLEDGE.md``'s ``preemption no-resume`` landmine says cannot survive,
and it has already been paid for once on LTX embe r1 (a 2.2 h render on a ~24-min preemption cycle
could never finish). The fix is a render directory keyed on the render's IDENTITY plus a
skip-if-present check and a commit per clip.

⛔ **The danger the fix introduces is worse than the problem it solves, if the key is wrong.** All
five H3 sample configs share ``output_dir`` (they must — ``find_latest`` resolves the adapter under
it), share ``seed: 42`` and share their prompt set, so every one of them writes the identical
filename ``{slug}_s42.mp4``. Two differ ONLY in ``validation.reference_subject_ids``; two more ONLY
in ``validation.frame_count``. A key missing either axis makes one config's resume skip another
config's clips, and the grid comes out labelled for a reference condition it does not contain —
silent, at a perfectly valid shape, on the precise axis this phase exists to measure.

So this file drives the REAL configs through the REAL key function. CPU-only, zero spend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from signet_trainer.config.load import load_config  # noqa: E402

# ⚠ NOT `from signet_trainer.modal.fns import ...`. Importing that module pulls `modal` into
# sys.modules for the whole pytest session and breaks the dry-run gate's Anti-Pattern-6
# assertion (`"modal" not in sys.modules`) in an unrelated file — a test that only fails when
# the suite runs in full, which is the worst kind. The key lives in a stdlib-only module for
# exactly this reason; `h3_sample` imports it function-locally.
from signet_trainer.inference.render_key import h3_render_key  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

#: Every H3 sample config in the repo. Globbed, not listed — a sixth config added later is covered
#: the day it appears, which is the only way a collision guard stays true.
_SAMPLE_CONFIGS = sorted((REPO / "configs").glob("h3_embe_r1_sample*.yaml"))


def _key_for(path: Path) -> str:
    config = load_config(str(path))
    return h3_render_key(
        # The checkpoint name is the same for all of them (they render the same adapter) — which is
        # exactly why it cannot be the only component of the key.
        checkpoint="checkpoint-step-3000",
        seed=config.seed,
        frame_count=config.validation.frame_count,
        width=config.validation.width,
        height=config.validation.height,
        num_inference_steps=config.validation.num_inference_steps,
        subject_ids=list(config.validation.reference_subject_ids) or ["A", "029"],
    )


def test_there_are_sibling_sample_configs_to_collide() -> None:
    """Non-vacuity: the collision guard below is empty if there is only one config."""
    assert len(_SAMPLE_CONFIGS) >= 4, (
        f"found {[p.name for p in _SAMPLE_CONFIGS]} — the matrix is meant to be five configs "
        f"isolating length and reference condition"
    )


def test_the_sibling_sample_configs_share_everything_the_filename_uses() -> None:
    """⛔ The premise. If these ever stop colliding, re-read the key before trusting it.

    The whole hazard is that ``{slug(prompt)}_s{seed}.mp4`` is IDENTICAL across the matrix. This
    asserts that premise rather than assuming it, so the guard cannot quietly become about nothing.
    """
    configs = [load_config(str(p)) for p in _SAMPLE_CONFIGS]
    assert len({c.output_dir for c in configs}) == 1, (
        "the sample configs no longer share output_dir — they must, because find_latest resolves "
        "the adapter under it"
    )
    assert len({c.seed for c in configs}) == 1, "the sample configs no longer share a seed"
    prompt_sets = {tuple(c.validation.prompts) for c in configs}
    assert len(prompt_sets) == 1, (
        "the sample configs no longer share a prompt set — the matrix compares grids ACROSS "
        "configs, so a prompt difference would be indistinguishable from a length or reference one"
    )


def test_every_sample_config_gets_its_OWN_render_directory() -> None:
    """⛔ THE COLLISION GUARD. Five configs, five directories, or a grid is mislabelled."""
    keys = {path.name: _key_for(path) for path in _SAMPLE_CONFIGS}
    duplicates = sorted(
        {key for key in keys.values() if list(keys.values()).count(key) > 1}
    )
    assert not duplicates, (
        f"sample configs collide on render key(s) {duplicates}: {keys}. A collision means one "
        f"config's resume skips another config's clips and the gallery is labelled for a render it "
        f"does not contain. Widen h3_render_key to carry the axis they differ on — never remove "
        f"this test."
    )


#: Fixed geometry for the tests below that aren't exercising the geometry axes themselves — the
#: values do not matter for those tests, only that every call supplies them (the whole point of
#: #22 finding 5 is that there is no longer a default to omit).
_GEOM = dict(width=1344, height=768, num_inference_steps=25)


def test_the_key_separates_the_two_axes_the_matrix_isolates() -> None:
    """Named directly, because these are the two comparisons the whole eval rests on."""
    base = {"checkpoint": "checkpoint-step-3000", "seed": 42, **_GEOM}
    length_a = h3_render_key(**base, frame_count=22, subject_ids=["A", "029"])
    length_b = h3_render_key(**base, frame_count=56, subject_ids=["A", "029"])
    assert length_a != length_b, "frame_count is not in the key — the LENGTH axis would collide"

    reference_a = h3_render_key(**base, frame_count=22, subject_ids=["A", "029"])
    reference_b = h3_render_key(**base, frame_count=22, subject_ids=["B", "029"])
    assert reference_a != reference_b, (
        "the reference condition is not in the key — the REFERENCE axis would collide, which is the "
        "headline claim of the phase"
    )


def test_the_key_separates_every_geometry_axis() -> None:
    """#22 finding 5: width / height / num_inference_steps are pipe(...) inputs, not banner-only.

    A resolution or step-count probe dispatched mid-campaign must land in its OWN directory, or its
    render silently resumes the previous geometry's clips under the new banner — the render key's
    entire job is to make exactly this collision impossible.
    """
    base = dict(checkpoint="checkpoint-step-3000", seed=42, frame_count=22, subject_ids=["A", "029"])
    reference = h3_render_key(**base, **_GEOM)
    width_changed = h3_render_key(**{**base, **_GEOM, "width": _GEOM["width"] + 32})
    height_changed = h3_render_key(**{**base, **_GEOM, "height": _GEOM["height"] + 32})
    steps_changed = h3_render_key(**{**base, **_GEOM, "num_inference_steps": _GEOM["num_inference_steps"] + 1})
    assert reference != width_changed, "width is not in the key — a resolution probe would collide"
    assert reference != height_changed, "height is not in the key — a resolution probe would collide"
    assert reference != steps_changed, (
        "num_inference_steps is not in the key — a step-count probe would collide"
    )


def test_the_key_is_order_sensitive_because_D_10_REFORDER_is() -> None:
    """A reordered reference set is a genuinely DIFFERENT request, not a permutation of one.

    Order fixes the ``<Picture i>`` labels AND advances the shared rotary clock, so sorting the
    subject ids into the key would merge two renders that are not the same render.
    """
    forward = h3_render_key(checkpoint="c", seed=42, frame_count=22, **_GEOM, subject_ids=["A", "029"])
    reversed_ = h3_render_key(checkpoint="c", seed=42, frame_count=22, **_GEOM, subject_ids=["029", "A"])
    assert forward != reversed_, "the render key sorts or otherwise loses reference ORDER"


def test_the_key_is_a_safe_and_readable_directory_name() -> None:
    """It is read off a Volume listing by a human, and it becomes a path — both matter."""
    key = h3_render_key(
        checkpoint="checkpoint-step-3000", seed=42, frame_count=22, **_GEOM, subject_ids=["A", "029"]
    )
    assert key == "checkpoint-step-3000_s42_f22_w1344_h768_n25_A-029", key
    assert "/" not in key and "\\" not in key and ".." not in key
    nasty = h3_render_key(
        checkpoint="../../etc/passwd", seed=1, frame_count=1, **_GEOM, subject_ids=["A"]
    )
    assert "/" not in nasty and "\\" not in nasty, (
        f"a checkpoint name containing path separators produced {nasty!r}, which would escape the "
        f"samples directory"
    )


def test_the_render_helper_skips_and_commits_per_clip() -> None:
    """Structural: the resume check and the per-clip commit must both be inside ``_render``.

    On the AST, because the payoff is entirely about WHERE these calls sit — a commit outside the
    per-clip function is the single end-of-run commit this change exists to replace, and it would
    still satisfy any text scan for ``checkpoints_vol.commit()``.
    """
    import ast

    fns = REPO / "src" / "signet_trainer" / "modal" / "fns.py"
    tree = ast.parse(fns.read_text(encoding="utf-8"))
    # ⚠ Scoped to h3_sample. The LTX `sample` stage defines its own `_render`, and an unscoped
    # `ast.walk` finds THAT one first — which would have made this test report on a function it is
    # not about, in either direction.
    h3_sample = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "h3_sample"
    )
    render = next(
        (
            node
            for node in ast.walk(h3_sample)
            if isinstance(node, ast.FunctionDef) and node.name == "_render"
        ),
        None,
    )
    assert render is not None, "h3_sample no longer defines a per-clip _render"

    commits = [
        call
        for call in ast.walk(render)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "commit"
    ]
    assert commits, (
        "_render does not commit. Without a commit-per-clip the completed clips of a preempted "
        "render are not on the Volume, so commit-or-vanish says they did not happen."
    )
    source = ast.get_source_segment(fns.read_text(encoding="utf-8"), render) or ""
    assert "exists()" in source and "st_size" in source, (
        "_render must skip a clip that is ALREADY rendered, and must test for a NON-EMPTY file — a "
        "container killed mid-encode leaves a 0-byte mp4, and skipping that would put a corrupt "
        "clip in the grid instead of re-rendering it."
    )
