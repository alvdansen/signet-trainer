"""SLICE 2 — the qwen_edit ROW JOIN: held-out inputs -> gallery rows -> the one cell-path rule.

``plan_qwen_edit_columns`` states two facts about every grid cell: WHERE its image is written
(``QwenEditColumn.render_subdir``) and WHICH row-dict key the gallery reads it back from
(``QwenEditColumn.row_key``). Before this slice nothing in the package connected them.
``grid._qwen_edit_block`` reads ``row[column.row_key]``; the sampler writes under
``column.render_subdir``; and the only code that ever produced a row was a builder invented inside
``tests/test_qwen_edit_render_surface.py`` (``out/{name}__{ckpt}__{mode}.png``) — a path shape that
matches no ``render_subdir`` at all. The HTML surface was therefore proved against paths the render
will never produce.

That is this family's own documented failure class one layer up: an artifact labelled for something
it does not contain, reached through a completely valid shape. Every tile would fall back to
'generation failed' on renders that actually succeeded — or, worse, a path collision would show one
column's pixels under another column's header and the BAND would be judged on the wrong image.

So the join gets ONE transcription (``qwen_edit_cell_relpath``) which both the planner and the
sampler must use, and this file pins it: the path a row carries is byte-identical to the path the
sampler is told to write, the base render is one directory for the whole band rather than N
byte-identical copies, and the resulting rows drive the real writer to a page with ``<img>``, zero
``<video>``, a labelled base row and both §8 prompt modes on every input.

Zero GPU, zero Modal, zero spend: pure dataclass/string functions plus HTML written to ``tmp_path``.
"""

from __future__ import annotations

import posixpath
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from signet_trainer.inference.grid import slug, write_qwen_edit_gallery
from signet_trainer.inference.qwen_edit_layout import (
    BASE_COLUMN_TOKEN,
    QWEN_EDIT_PROMPT_MODES,
    CheckpointBand,
    QwenEditHeldOutInput,
    plan_qwen_edit_columns,
    plan_qwen_edit_rows,
    qwen_edit_cell_relpath,
    qwen_edit_control_ids,
    qwen_edit_render_dir,
    qwen_edit_sample_filename,
)
from signet_trainer.inference.render_key import qwen_edit_render_key
from signet_trainer.inference.samples_layout import landed_render_ids

REPO = Path(__file__).resolve().parents[1]
SEED = 42


def _band() -> CheckpointBand:
    return CheckpointBand.of(["step-4000", "step-4200", "step-5000"])


def _columns() -> list:
    return plan_qwen_edit_columns(_band())


def _inputs() -> list[QwenEditHeldOutInput]:
    return [
        QwenEditHeldOutInput(
            input_id="train_icon",
            label="held-out · train_icon",
            control_imgs=("control/train_icon_slot1.png",),
            prompts={
                "a_style": "reimagine the reference icon in the house style",
                "b_content": "reimagine the train icon in the house style",
            },
        ),
        QwenEditHeldOutInput(
            input_id="bus_icon",
            label="held-out · bus_icon",
            control_imgs=("control/bus_icon_slot1.png",),
            prompts={
                "a_style": "reimagine the reference icon in the house style",
                "b_content": "reimagine the bus icon in the house style",
            },
        ),
    ]


def _params() -> dict:
    return {
        "steps": 30,
        "true_cfg": 4.0,
        "cfg_norm": "on",
        "width": 1024,
        "height": 1024,
        "lora_scale": 1.0,
        "checkpoint_band": _band().describe(),
    }


# --------------------------------------------------------------------------------------------------
# (1) THE JOIN — one transcription, used by both sides.
# --------------------------------------------------------------------------------------------------


def test_every_row_cell_equals_the_cell_path_function() -> None:
    """The headline assertion: what the gallery reads == what the sampler is told to write."""
    cols = _columns()
    inputs = _inputs()
    ids = qwen_edit_control_ids(inputs)
    rows = plan_qwen_edit_rows(inputs, cols, seed=SEED)

    for row, item in zip(rows, inputs, strict=True):
        for col in cols:
            expected = qwen_edit_cell_relpath(
                col, input_id=item.input_id, seed=SEED, control_ids=ids
            )
            assert row[col.row_key] == expected, (
                f"row key {col.row_key!r} carries {row[col.row_key]!r} but the sampler writes "
                f"{expected!r} — two transcriptions of one path layout have drifted"
            )


def test_cell_path_is_render_dir_then_render_subdir_then_filename() -> None:
    cols = _columns()
    ids = ("train_icon", "bus_icon")
    by_key = {c.row_key: c for c in cols}

    lora = by_key["step-4000__a_style_img"]
    assert qwen_edit_cell_relpath(
        lora, input_id="train_icon", seed=SEED, control_ids=ids
    ) == "step-4000_s42_ctrain_icon-bus_icon/lora/step-4000/a_style/train_icon_s42.png"

    base = by_key[f"{BASE_COLUMN_TOKEN}__b_content_img"]
    assert qwen_edit_cell_relpath(
        base, input_id="train_icon", seed=SEED, control_ids=ids
    ) == "base_s42_ctrain_icon-bus_icon/base/b_content/train_icon_s42.png"


def test_cell_path_prefix_is_exactly_the_render_key_the_watcher_expects() -> None:
    """The dir segment must be ``qwen_edit_render_key``'s output, or the watcher verifies elsewhere."""
    cols = _columns()
    ids = qwen_edit_control_ids(_inputs())
    for col in cols:
        head = qwen_edit_cell_relpath(
            col, input_id="train_icon", seed=SEED, control_ids=ids
        ).split("/")[0]
        ckpt = BASE_COLUMN_TOKEN if col.is_base else col.checkpoint
        assert head == qwen_edit_render_key(checkpoint=ckpt, seed=SEED, control_ids=ids)


def test_render_dirs_are_recognised_by_the_landed_check() -> None:
    """A planned path's dir segment must parse as a landed render id, both directions of the seam."""
    cols = _columns()
    ids = qwen_edit_control_ids(_inputs())
    dirs = sorted({qwen_edit_render_dir(c, seed=SEED, control_ids=ids) for c in cols})
    listing = "\n".join(f"outputs/qwen_edit_embe_phase1/samples_qwen_edit/{d}" for d in dirs)
    assert landed_render_ids(listing, "qwen_edit") == dirs
    # ...and the H3 parse must find NOTHING in it (mutually exclusive patterns, both directions).
    assert landed_render_ids(listing, "h3") == []


def test_base_render_is_one_directory_for_the_whole_band_not_one_per_member() -> None:
    """§8's convergence reference is ONE thing the grid points at — and N copies cost A100 time."""
    cols = _columns()
    ids = qwen_edit_control_ids(_inputs())
    base_dirs = {
        qwen_edit_render_dir(c, seed=SEED, control_ids=ids) for c in cols if c.is_base
    }
    assert base_dirs == {f"{BASE_COLUMN_TOKEN}_s42_ctrain_icon-bus_icon"}
    lora_dirs = {
        qwen_edit_render_dir(c, seed=SEED, control_ids=ids) for c in cols if not c.is_base
    }
    assert len(lora_dirs) == 3  # one per band member — the watcher's per-member landed unit
    assert base_dirs.isdisjoint(lora_dirs)


def test_filename_is_input_keyed_not_prompt_keyed() -> None:
    """Rewording an eval prompt must not rename every file and silently defeat resume."""
    assert qwen_edit_sample_filename(input_id="train_icon", seed=42) == "train_icon_s42.png"
    assert qwen_edit_sample_filename(input_id="train icon/../x", seed=7) == f"{slug('train icon/../x')}_s7.png"
    # No path separator can survive into a filename (T-04-04 carry-forward via `slug`).
    assert "/" not in qwen_edit_sample_filename(input_id="a/b/c", seed=1)
    assert ".." not in qwen_edit_sample_filename(input_id="../../etc", seed=1)


def test_every_cell_path_in_a_full_plan_is_unique() -> None:
    """A collision here shows one input's pixels under another's header, at a valid shape."""
    cols = _columns()
    rows = plan_qwen_edit_rows(_inputs(), cols, seed=SEED)
    paths = [row[c.row_key] for row in rows for c in cols]
    assert len(paths) == len(set(paths)) == 2 * 8


# --------------------------------------------------------------------------------------------------
# (2) control_ids — the conditioning axis, order preserved.
# --------------------------------------------------------------------------------------------------


def test_control_ids_preserve_slot_order() -> None:
    a = qwen_edit_control_ids(_inputs())
    b = qwen_edit_control_ids(list(reversed(_inputs())))
    assert a == ("train_icon", "bus_icon")
    assert b == ("bus_icon", "train_icon")
    assert a != b  # a reordered held-out set is a different request, never the same directory


def test_duplicate_held_out_ids_are_refused() -> None:
    dupe = _inputs()[0]
    with pytest.raises(ValueError, match="appears twice"):
        qwen_edit_control_ids([dupe, dupe])


# --------------------------------------------------------------------------------------------------
# (3) The held-out input record — §8's A/B pair is mandatory, never defaulted.
# --------------------------------------------------------------------------------------------------


def test_missing_prompt_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="no prompt for mode"):
        QwenEditHeldOutInput(input_id="x", prompts={"a_style": "only A"})


def test_unknown_prompt_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        QwenEditHeldOutInput(
            input_id="x",
            prompts={"a_style": "A", "b_content": "B", "c_typo": "oops"},
        )


def test_blank_prompt_is_refused() -> None:
    with pytest.raises(ValueError, match="blank prompt"):
        QwenEditHeldOutInput(input_id="x", prompts={"a_style": "   ", "b_content": "B"})


def test_blank_input_id_is_refused() -> None:
    with pytest.raises(ValueError, match="input_id is empty"):
        QwenEditHeldOutInput(input_id="  ", prompts={"a_style": "A", "b_content": "B"})


def test_control_imgs_are_normalised_to_a_tuple() -> None:
    staged = ["control/a.png"]
    item = QwenEditHeldOutInput(
        input_id="x", prompts={"a_style": "A", "b_content": "B"}, control_imgs=staged
    )
    assert item.control_imgs == ("control/a.png",)
    staged.append("control/b.png")  # caller mutates their list afterwards
    assert item.control_imgs == ("control/a.png",)  # the plan is unaffected


def test_header_falls_back_to_the_id() -> None:
    item = QwenEditHeldOutInput(input_id="train_icon", prompts={"a_style": "A", "b_content": "B"})
    assert item.header == "train_icon"


# --------------------------------------------------------------------------------------------------
# (4) The rows themselves.
# --------------------------------------------------------------------------------------------------


def test_rows_carry_both_prompts_under_their_mode_row_keys() -> None:
    rows = plan_qwen_edit_rows(_inputs(), _columns(), seed=SEED)
    for row, item in zip(rows, _inputs(), strict=True):
        for mode in QWEN_EDIT_PROMPT_MODES:
            assert row[mode.row_prompt_key] == item.prompts[mode.id]
    assert rows[0]["prompt_a"] != rows[0]["prompt_b"]  # A withholds the content word


def test_rows_carry_label_seed_and_control_thumbnails() -> None:
    rows = plan_qwen_edit_rows(_inputs(), _columns(), seed=SEED)
    assert [r["input_label"] for r in rows] == ["held-out · train_icon", "held-out · bus_icon"]
    assert [r["input_id"] for r in rows] == ["train_icon", "bus_icon"]
    assert all(r["seed"] == SEED for r in rows)
    assert rows[0]["control_imgs"] == ["control/train_icon_slot1.png"]
    assert "criteria_line" not in rows[0]  # absent unless overridden — the writer owns the default


def test_criteria_line_override_reaches_every_row() -> None:
    rows = plan_qwen_edit_rows(_inputs(), _columns(), seed=SEED, criteria_line="custom PASS")
    assert all(r["criteria_line"] == "custom PASS" for r in rows)


def test_empty_held_out_set_is_refused() -> None:
    with pytest.raises(ValueError, match="EMPTY held-out set"):
        plan_qwen_edit_rows([], _columns(), seed=SEED)


def test_empty_column_list_is_refused() -> None:
    with pytest.raises(ValueError, match="EMPTY column list"):
        plan_qwen_edit_rows(_inputs(), [], seed=SEED)


def test_rows_follow_a_base_less_band_when_the_planner_omits_it() -> None:
    cols = plan_qwen_edit_columns(_band(), include_base=False)
    rows = plan_qwen_edit_rows(_inputs(), cols, seed=SEED)
    assert not any(k.startswith(f"{BASE_COLUMN_TOKEN}__") for k in rows[0])
    assert all(not p.startswith(f"{BASE_COLUMN_TOKEN}_s") for p in
               (rows[0][c.row_key] for c in cols))


# --------------------------------------------------------------------------------------------------
# (5) END TO END — planner output drives the REAL writer, with no hand-built row anywhere.
# --------------------------------------------------------------------------------------------------


def _write(tmp_path: Path) -> str:
    cols = _columns()
    rows = plan_qwen_edit_rows(_inputs(), cols, seed=SEED)
    out = write_qwen_edit_gallery(rows, tmp_path / "index.html", _params(), columns=cols)
    return out.read_text(encoding="utf-8")


def test_planned_grid_is_images_and_never_video(tmp_path: Path) -> None:
    html = _write(tmp_path)
    assert "<video" not in html
    assert "</video>" not in html
    # 2 rows x (1 control thumbnail + 8 output cells) = 18 <img>
    assert html.count("<img") == 18
    assert html.count("generation failed") == 0  # every planned cell has a path


def test_planned_grid_labels_the_base_row_as_the_measurement(tmp_path: Path) -> None:
    html = _write(tmp_path)
    assert '<div class="col-label">BASE · A · style-only' in html
    assert '<div class="col-label">BASE · B · content-named' in html
    assert html.count("CONTROL — no adapter") == 4  # 2 base columns x 2 rows
    assert html.count('class="cell control"') == 4


def test_planned_grid_shows_both_modes_for_every_band_member(tmp_path: Path) -> None:
    html = _write(tmp_path)
    for ckpt in ("step-4000", "step-4200", "step-5000"):
        assert f'<div class="col-label">{ckpt} · A · style-only</div>' in html
        assert f'<div class="col-label">{ckpt} · B · content-named</div>' in html
        for mode_id in ("a_style", "b_content"):
            assert f"{ckpt}_s42_ctrain_icon-bus_icon/lora/{ckpt}/{mode_id}/train_icon_s42.png" in html


def test_planned_grid_prints_both_prompts_on_every_row(tmp_path: Path) -> None:
    html = _write(tmp_path)
    assert html.count("A · style-only: reimagine the reference icon in the house style") == 2
    assert "B · content-named: reimagine the train icon in the house style" in html
    assert "B · content-named: reimagine the bus icon in the house style" in html
    assert "—" not in html.split('<div class="reskin-prompt">')[1][:200]  # no missing-prompt dash


def test_hostile_input_id_cannot_escape_the_render_dir(tmp_path: Path) -> None:
    """T-04-04. The invariant is COMPONENT-level, and the distinction is load-bearing.

    ``render_key._sanitize`` deliberately keeps ``.`` — it has to, because real checkpoint dir names
    carry it (``checkpoint-step-00250-loss-0.0283``). So a hostile id CAN leave ``..`` as a substring
    inside the render-dir segment (``base_s42_c.._..____script_``), and asserting ``".." not in path``
    would be asserting something both false and irrelevant. What actually matters is that no path
    COMPONENT is ``.`` or ``..``, which the key shape guarantees by construction: it is always
    ``<ckpt>_s<digits>_c<ids>``, so the middle ``_s0_c`` makes a bare ``..`` segment unreachable
    (worst case ``qwen_edit_render_key(checkpoint='..', control_ids=['..'])`` -> ``'.._s0_c..'``).
    """
    hostile = QwenEditHeldOutInput(
        input_id='../../"><script>bad()</script>',
        prompts={"a_style": "A", "b_content": "B"},
    )
    cols = _columns()
    rows = plan_qwen_edit_rows([hostile], cols, seed=SEED)
    for col in cols:
        path = rows[0][col.row_key]
        parts = PurePosixPath(path).parts
        assert not any(p in (".", "..") for p in parts), f"traversal component in {path!r}"
        assert posixpath.normpath(f"root/{path}").startswith("root/"), path
        assert "\\" not in path
        assert "<script>" not in path
    html = write_qwen_edit_gallery(rows, tmp_path / "index.html", _params(), columns=cols).read_text(
        encoding="utf-8"
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# --------------------------------------------------------------------------------------------------
# (6) Import tier — in a SUBPROCESS, because an in-process sys.modules check answers "what has the
# suite imported by now", not "what does this module pull in". The new code adds FUNCTION-LOCAL
# imports (`grid.slug`, `render_key`) which only execute when called, so the check must call them.
# --------------------------------------------------------------------------------------------------


def test_row_planner_pulls_in_no_torch_or_modal_in_a_clean_interpreter() -> None:
    script = (
        "import sys\n"
        "sys.path.insert(0, r'" + str(REPO / "src") + "')\n"
        "from signet_trainer.inference.qwen_edit_layout import (\n"
        "    CheckpointBand, QwenEditHeldOutInput, plan_qwen_edit_columns, plan_qwen_edit_rows)\n"
        "cols = plan_qwen_edit_columns(CheckpointBand.of(['step-4000']))\n"
        "items = [QwenEditHeldOutInput(input_id='t', prompts={'a_style':'A','b_content':'B'})]\n"
        "rows = plan_qwen_edit_rows(items, cols, seed=42)\n"
        "assert rows[0][cols[0].row_key]\n"
        "bad = [m for m in ('torch','modal','ltx_core','ltx_trainer') if m in sys.modules]\n"
        "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "CLEAN", proc.stdout
