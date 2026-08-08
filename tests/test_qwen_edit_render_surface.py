"""SLICE 5 — the qwen_edit IMAGE render surface + the A/B x checkpoint-BAND sampling seam.

Family #3 is the first IMAGE family in this repo, and it lands three things that every prior family
got for free from the video path:

  1. **An image OUTPUT column.** ``inference/grid.py``'s four shipped writers route every OUTPUT
     through ``_video_cell``; ``_image_cell`` existed only for REFERENCE thumbnails. A qwen_edit grid
     rendered through the video path would emit ``<video src="…png">`` — black rectangles with
     playback controls. ``write_qwen_edit_gallery`` must emit ``<img>`` and NO ``<video>``.
  2. **A render key.** ``expected_qwen_edit_render_key`` / ``landed_render_ids`` were LOUD stubs; a
     guessed key is how a successful render is marked FAILED and re-dispatched at full price
     (KNOWLEDGE.md 'watcher phantom-spend'). The two families' key patterns must be mutually
     exclusive so neither family's parse can produce a plausible partial list from the other's
     listing.
  3. **§8's two layout requirements** — the A/B prompt-mode pair rendered side by side on one row,
     and a CHECKPOINT BAND rather than a single winner.

Zero GPU, zero Modal, zero spend: pure string/dataclass functions plus HTML written to ``tmp_path``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from signet_trainer.inference.grid import (
    _image_cell,
    _image_tile,
    write_comparison_gallery,
    write_multi_frame_gallery,
    write_qwen_edit_gallery,
    write_reference_gallery,
    write_reskin_gallery,
)
from signet_trainer.inference.qwen_edit_layout import (
    BASE_COLUMN_TOKEN,
    QWEN_EDIT_PROMPT_MODES,
    CheckpointBand,
    band_render_keys,
    plan_qwen_edit_columns,
    render_qwen_edit_sample,
)
from signet_trainer.inference.render_key import h3_render_key, qwen_edit_render_key
from signet_trainer.inference.samples_layout import (
    _H3_KEY_RE,
    _QWEN_EDIT_KEY_RE,
    expected_qwen_edit_band_keys,
    expected_qwen_edit_render_key,
    landed_render_ids,
    samples_root,
    samples_subdir,
)

REPO = Path(__file__).resolve().parents[1]
_FORBIDDEN_MODULES = ("torch", "modal", "ltx_core", "ltx_trainer")


# --------------------------------------------------------------------------------------------------
# Anti-Pattern-6 import confinement — the watcher imports these WITHOUT the modal SDK loaded.
# --------------------------------------------------------------------------------------------------


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def test_qwen_edit_layout_imports_no_torch_modal_or_ltx() -> None:
    # Snapshot before popping so we can RESTORE afterwards (same rationale as test_grid_html.py:
    # evicting torch without restoring double-registers the triton TORCH_LIBRARY on re-import).
    saved = {mod: sys.modules.pop(mod, None) for mod in _FORBIDDEN_MODULES}
    try:
        import signet_trainer.inference.qwen_edit_layout  # noqa: F401

        for mod in _FORBIDDEN_MODULES:
            assert mod not in sys.modules, (
                f"import-confinement violation: qwen_edit_layout transitively imported {mod!r}"
            )
    finally:
        for mod, value in saved.items():
            if value is not None:
                sys.modules[mod] = value


def test_qwen_edit_layout_source_has_no_heavy_import() -> None:
    src = REPO / "src" / "signet_trainer" / "inference" / "qwen_edit_layout.py"
    code = _strip_comments_and_docstrings(src.read_text(encoding="utf-8"))
    offenders = [
        r"^\s*import\s+torch\b",
        r"^\s*from\s+torch\b",
        r"^\s*import\s+modal\b",
        r"^\s*from\s+modal\b",
        r"^\s*import\s+ltx_core\b",
        r"^\s*from\s+ltx_core\b",
        r"^\s*import\s+ltx_trainer\b",
        r"^\s*from\s+ltx_trainer\b",
    ]
    hits = [pat for pat in offenders if re.search(pat, code, re.MULTILINE)]
    assert not hits, f"import-confinement violation in qwen_edit_layout.py real code: {hits}"


# --------------------------------------------------------------------------------------------------
# (2) The render key — the identity axes, and the delegation contract.
# --------------------------------------------------------------------------------------------------


def test_expected_key_delegates_to_the_renders_own_function() -> None:
    # Never a re-implementation: the watcher's expectation and the render's directory name must come
    # from the SAME function or they drift silently. Mirrors the h3 pin.
    kwargs = dict(checkpoint="checkpoint-step-04000-loss-0.0912", seed=42,
                  control_ids=["train_icon", "blank", "blank"])
    assert expected_qwen_edit_render_key(**kwargs) == qwen_edit_render_key(**kwargs)


def test_band_members_are_distinct_renders() -> None:
    # §8 ships three checkpoints. They share output_dir, seed, the held-out control set and the
    # prompt pair, so they write identical FILENAMES — only the render-dir identity separates them.
    keys = [
        expected_qwen_edit_render_key(checkpoint=c, seed=42, control_ids=["train_icon"])
        for c in ("step-4000", "step-4200", "step-5000")
    ]
    assert len(set(keys)) == 3, "the checkpoint must be part of the render identity"


def test_seed_and_control_set_are_identity_axes() -> None:
    base = dict(checkpoint="step-4000", control_ids=["train_icon"])
    assert (
        expected_qwen_edit_render_key(seed=42, **base)
        != expected_qwen_edit_render_key(seed=7, **base)
    )
    assert expected_qwen_edit_render_key(
        checkpoint="step-4000", seed=42, control_ids=["train_icon"]
    ) != expected_qwen_edit_render_key(
        checkpoint="step-4000", seed=42, control_ids=["bus_icon"]
    )


def test_control_slot_order_is_not_collapsed() -> None:
    # Slot index IS the caption's ctrl_img_N addressing (conditioning/qwen_edit.py:175-177), so a
    # reordered control set is a genuinely different request — same law as H3's D-10-REFORDER.
    fwd = expected_qwen_edit_render_key(checkpoint="c", seed=42, control_ids=["A", "B"])
    rev = expected_qwen_edit_render_key(checkpoint="c", seed=42, control_ids=["B", "A"])
    assert fwd != rev


def test_key_carries_no_frame_segment() -> None:
    # F is pinned to EXACTLY 1 on this family; an _f<frames> segment would be a constant wearing an
    # identity axis's clothes (and is what the stub warned against copying from the h3 shape).
    key = expected_qwen_edit_render_key(checkpoint="step-4000", seed=42, control_ids=["A"])
    assert key == "step-4000_s42_cA"
    assert not re.search(r"_f\d+", key)


def test_empty_control_set_keys_as_noctrl() -> None:
    key = expected_qwen_edit_render_key(checkpoint="step-4000", seed=42, control_ids=[])
    assert key.endswith("_cnoctrl")


def test_key_sanitises_path_separators() -> None:
    key = expected_qwen_edit_render_key(
        checkpoint="../../escape", seed=42, control_ids=["a/b", "c\\d"]
    )
    assert "/" not in key and "\\" not in key


# --------------------------------------------------------------------------------------------------
# (2b) Mutual exclusion of the two families' key patterns — both directions.
# --------------------------------------------------------------------------------------------------

_QWEN_LISTING = (
    "outputs/qwen_edit_r1/samples_qwen_edit/checkpoint-step-04000-loss-0.0912_s42_ctrain_icon\n"
    "outputs/qwen_edit_r1/samples_qwen_edit/checkpoint-step-04200-loss-0.0904_s42_ctrain_icon\n"
    "outputs/qwen_edit_r1/samples_qwen_edit/checkpoint-step-05000-loss-0.0899_s42_ctrain_icon\n"
)
_H3_LISTING = (
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f22_A-029\n"
    "outputs/h3_embe_r1/samples_h3/checkpoint-step-00250-loss-0.1016_s42_f124_A-029\n"
)


def test_qwen_landed_ids_are_the_band_keys() -> None:
    ids = landed_render_ids(_QWEN_LISTING, "qwen_edit")
    assert len(ids) == 3
    assert "checkpoint-step-04000-loss-0.0912_s42_ctrain_icon" in ids


def test_h3_regex_never_matches_a_qwen_key() -> None:
    for line in _QWEN_LISTING.splitlines():
        name = line.strip().split("/")[-1]
        assert not _H3_KEY_RE.match(name), f"h3 pattern matched a qwen key: {name}"
    assert landed_render_ids(_QWEN_LISTING, "h3") == []


def test_qwen_regex_never_matches_an_h3_key() -> None:
    for line in _H3_LISTING.splitlines():
        name = line.strip().split("/")[-1]
        assert not _QWEN_EDIT_KEY_RE.match(name), f"qwen pattern matched an h3 key: {name}"
    assert landed_render_ids(_H3_LISTING, "qwen_edit") == []


def test_generated_keys_are_mutually_exclusive_across_families() -> None:
    # Property-ish sweep over adversarial ids/checkpoints, not just the two fixtures above: an id
    # that LOOKS like the other family's segment must not flip the recogniser.
    ckpts = ["step-4000", "checkpoint-step-00250-loss-0.1016", "f22", "s42"]
    ids = [["A"], ["f22", "A"], ["c1", "2"], ["s1", "c"], []]
    for ckpt in ckpts:
        for id_set in ids:
            q = qwen_edit_render_key(checkpoint=ckpt, seed=42, control_ids=id_set)
            h = h3_render_key(checkpoint=ckpt, seed=42, frame_count=22, subject_ids=id_set)
            assert _QWEN_EDIT_KEY_RE.match(q), f"qwen recogniser missed its own key {q}"
            assert _H3_KEY_RE.match(h), f"h3 recogniser missed its own key {h}"
            assert not _H3_KEY_RE.match(q), f"h3 pattern matched qwen key {q}"
            assert not _QWEN_EDIT_KEY_RE.match(h), f"qwen pattern matched h3 key {h}"


def test_prompt_mode_subdirs_are_not_mistaken_for_render_keys() -> None:
    # The A/B modes live as SUBDIRS of one render dir. A recursive listing must not report them as
    # landed renders — that would count one row's half as a complete render.
    listing = (
        "outputs/qwen_edit_r1/samples_qwen_edit/step-4000_s42_cA/base/a_style\n"
        "outputs/qwen_edit_r1/samples_qwen_edit/step-4000_s42_cA/lora/step-4000/b_content\n"
    )
    ids = landed_render_ids(listing, "qwen_edit")
    assert "a_style" not in ids and "b_content" not in ids


def test_qwen_samples_root_is_unchanged() -> None:
    assert samples_subdir("qwen_edit") == "samples_qwen_edit"
    assert samples_root("outputs/qwen_edit_r1", "qwen_edit") == (
        "outputs/qwen_edit_r1/samples_qwen_edit"
    )


# --------------------------------------------------------------------------------------------------
# (4) The checkpoint BAND — §8 ships a band, not a winner.
# --------------------------------------------------------------------------------------------------


def test_band_preserves_order_and_has_no_winner_field() -> None:
    band = CheckpointBand.of(["step-5000", "step-4000", "step-4200"])
    assert band.members == ("step-5000", "step-4000", "step-4200")  # never sorted
    assert band.size == 3
    for forbidden in ("best", "winner", "selected"):
        assert not hasattr(band, forbidden), (
            f"CheckpointBand grew a {forbidden!r} attribute — §8 is explicit that selection is a "
            "band, not a winner, and any such field lets consumers collapse it back to one render."
        )


def test_band_describe_names_every_member() -> None:
    assert CheckpointBand.of(["4000", "4200", "5000"]).describe() == "band of 3: 4000 / 4200 / 5000"


def test_single_member_band_is_legal() -> None:
    # A mid-training grid renders one checkpoint — still a band, so nothing needs a winner special case.
    assert CheckpointBand.of(["step-250"]).size == 1


def test_empty_band_is_refused() -> None:
    with pytest.raises(ValueError, match="EMPTY"):
        CheckpointBand.of([])


def test_duplicate_band_member_is_refused_not_deduped() -> None:
    with pytest.raises(ValueError, match="appears twice"):
        CheckpointBand.of(["step-4000", "step-4000"])


def test_band_render_keys_are_one_per_member_in_band_order() -> None:
    band = CheckpointBand.of(["step-5000", "step-4000"])
    keys = band_render_keys(band, seed=42, control_ids=["train_icon"])
    assert keys == [
        "step-5000_s42_ctrain_icon",
        "step-4000_s42_ctrain_icon",
    ]
    assert keys == expected_qwen_edit_band_keys(
        checkpoints=band.members, seed=42, control_ids=["train_icon"]
    )


def test_empty_band_key_request_is_refused() -> None:
    # Returning [] would read to the watcher as "every render landed" — the inverse of the truth.
    with pytest.raises(ValueError, match="EMPTY band"):
        expected_qwen_edit_band_keys(checkpoints=[], seed=42, control_ids=["A"])


# --------------------------------------------------------------------------------------------------
# (3) The A/B x band column planner.
# --------------------------------------------------------------------------------------------------


def test_prompt_modes_are_exactly_the_two_from_method_section_8() -> None:
    assert [m.id for m in QWEN_EDIT_PROMPT_MODES] == ["a_style", "b_content"]
    assert [m.row_prompt_key for m in QWEN_EDIT_PROMPT_MODES] == ["prompt_a", "prompt_b"]


def test_columns_group_by_checkpoint_with_ab_adjacent() -> None:
    # §8's read is "same input, two prompts, side by side" — A must sit immediately beside B for the
    # SAME weights, or the comparison the method asks for becomes a scan across the row.
    band = CheckpointBand.of(["step-4000", "step-4200"])
    cols = plan_qwen_edit_columns(band)
    assert [(c.checkpoint, c.mode.id) for c in cols] == [
        (None, "a_style"),
        (None, "b_content"),
        ("step-4000", "a_style"),
        ("step-4000", "b_content"),
        ("step-4200", "a_style"),
        ("step-4200", "b_content"),
    ]


def test_base_columns_come_first_and_are_marked_control() -> None:
    # §8's convergence check is base-vs-LoRA divergence; the baseline is leftmost and marked, never
    # presented as one more result.
    cols = plan_qwen_edit_columns(CheckpointBand.of(["step-4000"]))
    assert cols[0].is_base and cols[1].is_base
    assert all(c.control for c in cols if c.is_base)
    assert not any(c.control for c in cols if not c.is_base)


def test_include_base_false_drops_only_the_base_pair() -> None:
    band = CheckpointBand.of(["step-4000", "step-4200"])
    cols = plan_qwen_edit_columns(band, include_base=False)
    assert len(cols) == 4
    assert not any(c.is_base for c in cols)


def test_column_row_keys_are_unique_and_dict_safe() -> None:
    cols = plan_qwen_edit_columns(CheckpointBand.of(["step-4000", "step-4200", "step-5000"]))
    keys = [c.row_key for c in cols]
    assert len(set(keys)) == len(keys) == 8  # (3 band + 1 base) x 2 modes
    for key in keys:
        assert re.fullmatch(r"[A-Za-z0-9_-]+", key), key


def test_band_member_colliding_with_the_base_token_is_refused() -> None:
    with pytest.raises(ValueError, match="reserved column token"):
        plan_qwen_edit_columns(CheckpointBand.of([BASE_COLUMN_TOKEN]))


def test_band_members_colliding_after_tokenisation_are_refused() -> None:
    # "step 4000" and "step/4000" both tokenise to step_4000 — one member's cells would overwrite
    # the other's and the grid would be labelled for a checkpoint it does not contain.
    with pytest.raises(ValueError, match="tokenise to"):
        plan_qwen_edit_columns(CheckpointBand.of(["step 4000", "step/4000"]))


def test_render_subdir_puts_the_mode_inside_the_render_dir() -> None:
    # The mode is deliberately NOT part of the render identity, so it must appear as a SUBDIR.
    cols = plan_qwen_edit_columns(CheckpointBand.of(["step-4000"]))
    assert [c.render_subdir for c in cols] == [
        "base/a_style",
        "base/b_content",
        "lora/step-4000/a_style",
        "lora/step-4000/b_content",
    ]


# --------------------------------------------------------------------------------------------------
# (1) The image OUTPUT cell + the gallery. THE headline assertion: <img>, never <video>.
# --------------------------------------------------------------------------------------------------


def _band() -> CheckpointBand:
    return CheckpointBand.of(["step-4000", "step-4200", "step-5000"])


def _columns() -> list:
    return plan_qwen_edit_columns(_band())


def _qwen_params() -> dict:
    # §8's Qwen-Image-Edit-2511 inference settings, MINUS the static scheduler-reparameterisation
    # value — that field is blocked by test_no_wan_params.py's directory-wide token ban and its
    # resolution is an open ruling. See _qwen_edit_params_banner's docstring.
    return {
        "steps": 30,
        "true_cfg": 4.0,
        "cfg_norm": "on",
        "width": 1024,
        "height": 1024,
        "lora_scale": 1.0,
        "checkpoint_band": _band().describe(),
    }


def _qwen_rows() -> list[dict]:
    cols = _columns()
    rows = []
    for name in ("train_icon", "bus_icon"):
        row = {
            "input_label": f"held-out · {name}",
            "seed": 42,
            "control_imgs": [f"control/{name}_slot1.png"],
            "prompt_a": "reimagine the reference icon in the house style",
            "prompt_b": f"reimagine the {name.split('_')[0]} icon in the house style",
        }
        for col in cols:
            token = "base" if col.is_base else col.checkpoint
            row[col.row_key] = f"out/{name}__{token}__{col.mode.id}.png"
        rows.append(row)
    return rows


def test_image_tile_renders_escaped_img_with_no_loop_attribute() -> None:
    out = _image_tile("BASE · A · style-only", "out/x.png")
    assert "<img" in out
    assert 'src="out/x.png"' in out
    assert "loop" not in out  # `loop` is not a valid <img> attribute; not reproduced from _image_cell


def test_image_tile_fallback_is_caller_chosen_not_no_reference() -> None:
    # A missing OUTPUT is a failed render, not a missing reference — mislabelling it sends the
    # operator hunting for a dataset problem that does not exist.
    assert "generation failed" in _image_tile("ckpt · A", "").lower()
    assert "no reference" not in _image_tile("ckpt · A", "").lower()
    assert "control image missing" in _image_tile("control 1", "", fallback="control image missing")


def test_image_tile_escapes_hostile_label_and_src() -> None:
    out = _image_tile("<script>alert(1)</script>", '"><script>bad()</script>')
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_image_cell_is_byte_identical_after_the_addition() -> None:
    # ADDITIVE ONLY: the reference-thumbnail cell the three shipped galleries use is untouched,
    # `loop` artefact and all. _image_tile is a SIBLING, never a replacement.
    assert _image_cell("reference", "ref/frame0.png") == (
        '<div class="cell">'
        '<div class="col-label">reference</div>'
        '<img loop src="ref/frame0.png" alt="reference">'
        "</div>"
    )


def test_qwen_gallery_writes_index_and_returns_path(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "index.html"
    result = write_qwen_edit_gallery(_qwen_rows(), out, _qwen_params(), columns=_columns())
    assert result == out
    assert out.exists()  # parent dirs were created
    assert out.read_text(encoding="utf-8").strip() != ""


def test_qwen_gallery_renders_images_and_no_video(tmp_path: Path) -> None:
    """THE headline assertion of this slice: an image family renders <img>, never <video>."""
    out = write_qwen_edit_gallery(
        _qwen_rows(), tmp_path / "index.html", _qwen_params(), columns=_columns()
    )
    html = out.read_text(encoding="utf-8")
    assert "<video" not in html
    assert "</video>" not in html
    # 2 rows x (1 control thumb + 8 output cells) = 18 <img>
    assert html.count("<img") == 18


def test_qwen_gallery_renders_every_band_member_and_both_modes(tmp_path: Path) -> None:
    out = write_qwen_edit_gallery(
        _qwen_rows(), tmp_path / "index.html", _qwen_params(), columns=_columns()
    )
    html = out.read_text(encoding="utf-8")
    for ckpt in ("step-4000", "step-4200", "step-5000"):
        # The exact rendered label — not an `or` over plausible spellings, which would pass on the
        # wrong branch and stop pinning anything.
        assert f'<div class="col-label">{ckpt} · A · style-only</div>' in html
        assert f'<div class="col-label">{ckpt} · B · content-named</div>' in html
        assert f"out/train_icon__{ckpt}__a_style.png" in html
        assert f"out/train_icon__{ckpt}__b_content.png" in html
    assert "BASE" in html
    assert html.count('class="cell control"') == 4  # 2 base columns x 2 rows
    assert html.count('class="badge-control"') == 4


def test_qwen_gallery_prints_both_prompts_per_row(tmp_path: Path) -> None:
    # §8's read is a comparison BETWEEN two prompts; a row showing one of them invites the delta to
    # be attributed to the model rather than to the missing prompt.
    out = write_qwen_edit_gallery(
        _qwen_rows(), tmp_path / "index.html", _qwen_params(), columns=_columns()
    )
    html = out.read_text(encoding="utf-8")
    assert "reimagine the reference icon in the house style" in html
    assert "reimagine the train icon in the house style" in html
    assert html.count("style-only:") == 2  # one A prompt line per row
    assert html.count("content-named:") == 2


def test_qwen_gallery_missing_prompt_renders_em_dash_not_silence(tmp_path: Path) -> None:
    rows = _qwen_rows()
    del rows[0]["prompt_b"]
    out = write_qwen_edit_gallery(rows, tmp_path / "index.html", _qwen_params(), columns=_columns())
    html = out.read_text(encoding="utf-8")
    assert "content-named: —" in html


def test_qwen_gallery_missing_output_renders_failed_fallback(tmp_path: Path) -> None:
    cols = _columns()
    rows = _qwen_rows()
    rows[0][cols[2].row_key] = ""  # step-4000 · A render failed
    out = write_qwen_edit_gallery(rows, tmp_path / "index.html", _qwen_params(), columns=cols)
    html = out.read_text(encoding="utf-8")
    assert "generation failed" in html.lower()
    assert html.count("<img") == 17  # 18 minus the failed tile
    assert "<video" not in html


def test_qwen_gallery_shows_a_wholly_failed_band_member_as_a_column(tmp_path: Path) -> None:
    # The band is a DECLARATION: a member whose renders all failed must appear as a column of honest
    # 'generation failed' tiles, never vanish. This is why `columns` is passed, not derived.
    cols = _columns()
    rows = _qwen_rows()
    for row in rows:
        for col in cols:
            if col.checkpoint == "step-4200":
                row[col.row_key] = ""
    out = write_qwen_edit_gallery(rows, tmp_path / "index.html", _qwen_params(), columns=cols)
    html = out.read_text(encoding="utf-8")
    assert "step-4200" in html, "a fully-failed band member must still be labelled on the grid"
    assert html.count("generation failed") == 4  # 2 rows x 2 modes


def test_qwen_gallery_banner_carries_section_8_inference_settings_and_the_band(
    tmp_path: Path,
) -> None:
    # §8 names these a documented trap with an explicit diagnosis tell ("training samples look fine
    # but renders are muddy -> it's inference settings, not the model") — unreadable off the grid if
    # the grid does not print them.
    out = write_qwen_edit_gallery(
        _qwen_rows(), tmp_path / "index.html", _qwen_params(), columns=_columns()
    )
    html = out.read_text(encoding="utf-8")
    assert "steps=30" in html
    assert "true_cfg=4.0" in html
    assert "cfg_norm=on" in html
    assert "band of 3: step-4000 / step-4200 / step-5000" in html
    assert "stg=" not in html, "stg is an LTX-only knob and must not appear on a Qwen banner"
    assert "1024x1024x" not in html, "F is pinned to 1; the dims must not present it as a setting"
    assert html.count('class="params"') == 1


def test_banner_refuses_an_unknown_field_rather_than_dropping_it() -> None:
    """A dropped SAMPLING setting is invisible on the artifact an operator diagnoses from.

    This is also the seam where the OPEN RULING announces itself: the one §8 inference setting this
    banner cannot render — the static scheduler-reparameterisation value — is blocked by
    ``test_no_wan_params.py``'s directory-wide token ban, and a caller who plumbs it through must
    hit a loud, actionable error rather than watch it vanish.
    """
    with pytest.raises(ValueError) as excinfo:
        write_qwen_edit_gallery(
            _qwen_rows(),
            Path("unused.html"),
            dict(_qwen_params(), some_unrenderable_setting=3.0),
            columns=_columns(),
        )
    msg = str(excinfo.value)
    assert "unrecognised qwen_edit banner field" in msg
    assert "BLOCKED, not forgotten" in msg
    assert "test_no_wan_params.py" in msg, "the error must name the guard that blocks the field"


def test_inference_dir_stays_clean_of_the_banned_token() -> None:
    """The slice respects ``test_no_wan_params.py`` instead of dodging or weakening it.

    Prose (docstrings/comments) MAY name the banned token — the guard's own
    ``test_a_wan_token_only_in_a_docstring_is_not_flagged`` sanctions that, and both new modules use
    it to state the conflict in full. What must stay clean is EXECUTABLE code. Pinned here as well as
    in the guard so that a future edit which re-introduces the token has to confront this slice's
    written reasoning, not just a red scanner it might be tempted to narrow.
    """
    inference_dir = REPO / "src" / "signet_trainer" / "inference"
    for name in ("grid.py", "qwen_edit_layout.py"):
        code = _strip_comments_and_docstrings((inference_dir / name).read_text(encoding="utf-8"))
        assert "shift" not in code, (
            f"{name} re-introduced a token tests/test_no_wan_params.py bans across inference/. "
            "If the guard has since been narrowed to LTX modules, delete THIS test in the same "
            "commit and say so — do not simply let both drift."
        )
        # ...and the conflict must still be documented, not silently dropped along with the token.
        prose = (inference_dir / name).read_text(encoding="utf-8")
        assert "test_no_wan_params" in prose, (
            f"{name} must keep the written statement of the conflict; a silent omission is how the "
            "next reader concludes the §8 setting was simply forgotten."
        )


def test_qwen_gallery_has_the_pass_fail_criteria_block(tmp_path: Path) -> None:
    out = write_qwen_edit_gallery(
        _qwen_rows(), tmp_path / "index.html", _qwen_params(), columns=_columns()
    )
    html = out.read_text(encoding="utf-8")
    assert 'class="criteria"' in html
    assert "PASS =" in html and "FAIL =" in html
    assert 'class="grid-title"' in html
    assert html.count("PASS if:") == 2  # one per row


def test_qwen_gallery_escapes_hostile_input(tmp_path: Path) -> None:
    cols = _columns()
    row = {
        "input_label": "<script>alert(1)</script>",
        "seed": 42,
        "control_imgs": ['"><script>evil()</script>.png'],
        "prompt_a": "<script>a()</script>",
        "prompt_b": "ok",
    }
    for col in cols:
        row[col.row_key] = '"><script>bad()</script>.png'
    out = write_qwen_edit_gallery([row], tmp_path / "index.html", _qwen_params(), columns=cols)
    html = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "<script>evil()</script>" not in html
    assert "<script>bad()</script>" not in html
    assert "&lt;script&gt;" in html


def test_qwen_gallery_refuses_an_empty_column_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="EMPTY column list"):
        write_qwen_edit_gallery(_qwen_rows(), tmp_path / "index.html", _qwen_params(), columns=[])


def test_qwen_gallery_is_scoped_so_shipped_galleries_keep_their_rendering(tmp_path: Path) -> None:
    # The new CSS is appended and SCOPED under .qwen-edit-grid; a bare `img{...}` rule would resize
    # the reference thumbnails in three already-delivered galleries.
    out = write_qwen_edit_gallery(
        _qwen_rows(), tmp_path / "index.html", _qwen_params(), columns=_columns()
    )
    html = out.read_text(encoding="utf-8")
    assert ".qwen-edit-grid img{" in html
    assert 'class="qwen-edit-grid"' in html
    # Scan the actual <style> block for a bare `img{...}` / `.row{...}` / `.cell{...}` selector.
    # _STYLE is one newline-free string, so a `"\nimg{" not in html` check would be vacuous.
    style = html.split("<style>")[1].split("</style>")[0]
    selectors = [rule.split("{", 1)[0] for rule in style.split("}") if "{" in rule]
    assert "img" not in selectors, "an UNSCOPED img rule would resize three shipped galleries"
    for shared in (".row", ".cell"):
        assert selectors.count(shared) == 1, (
            f"{shared} must keep exactly its one shipped rule; the qwen overrides are scoped under "
            ".qwen-edit-grid"
        )


# --------------------------------------------------------------------------------------------------
# ADDITIVE-ONLY regression: the four shipped video galleries are untouched.
# --------------------------------------------------------------------------------------------------


def test_shipped_video_galleries_still_render_video(tmp_path: Path) -> None:
    params = {
        "steps": 30, "guidance": 3.0, "stg_scale": 1.0,
        "width": 768, "height": 352, "frames": 81, "lora_scale": 1.0,
    }
    rows = [{"prompt": "p", "seed": 42, "base_mp4": "b.mp4", "lora_mp4": "l.mp4"}]
    html = write_comparison_gallery(rows, tmp_path / "a.html", params).read_text(encoding="utf-8")
    assert html.count("<video") == 2 and "<img" not in html

    ref_params = dict(params, first_frame_conditioning="on")
    ref_rows = [{"prompt": "p", "seed": 42, "reference_img": "r.png",
                 "ref_on_mp4": "on.mp4", "ref_off_mp4": "off.mp4"}]
    html = write_reference_gallery(ref_rows, tmp_path / "b.html", ref_params).read_text("utf-8")
    assert html.count("<video") == 2 and html.count("<img") == 1
    assert "<img loop " in html  # the shipped attribute set is unchanged

    mf_rows = [{"prompt": "p", "seed": 42, "reference_imgs": ["r1.png", "r2.png"],
                "n2_mp4": "n2.mp4", "n3_mp4": "n3.mp4", "strength_lo_mp4": "lo.mp4",
                "strength_hi_mp4": "hi.mp4", "no_ref_mp4": "c.mp4"}]
    html = write_multi_frame_gallery(mf_rows, tmp_path / "c.html", params).read_text("utf-8")
    assert html.count("<video") == 5 and html.count("<img") == 2

    rs_params = dict(params, conditioning_mode="ic_lora", reference_downscale_factor=1)
    rs_rows = [{"row": "A", "row_type": "held-out", "clip_name": "c", "prompt": "p", "seed": 42,
                "steps": 30, "original_mp4": "o.mp4", "seg_map_mp4": "s.mp4",
                "ic_lora_mp4": "i.mp4", "base_no_ref_mp4": "n.mp4"}]
    html = write_reskin_gallery(rs_rows, tmp_path / "d.html", rs_params).read_text("utf-8")
    assert html.count("<video") == 4
    assert "CONTROL — no reference" in html  # the re-skin badge copy is unchanged


# --------------------------------------------------------------------------------------------------
# The one remaining gap is a NAMED symbol with an actionable message.
# --------------------------------------------------------------------------------------------------


def test_render_call_is_a_declared_stub_naming_what_lands_it() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        render_qwen_edit_sample(checkpoint="step-4000", seed=42)
    msg = str(excinfo.value)
    assert "DECLARED STUB" in msg
    for expected in (
        "load_qwen_edit_transformer",
        "qwen_edit_encode",
        "aspect-ratio",
        "true_cfg",
        "scheduler reparameterisation",
    ):
        assert expected in msg, f"the stub message must name {expected!r} as blocking work"
    # ...and it must surface the OPEN RULING, not just the missing code: a reader who starts writing
    # the sampler needs to know item (3) is blocked by a contract conflict before, not after.
    assert "CONTRACT CONFLICT" in msg
    assert "test_no_wan_params.py" in msg
