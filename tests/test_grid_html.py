"""INFR-02 — the base-vs-LoRA HTML montage writer (`inference/grid.py`).

CPU unit tests (torch-free, GPU-free): ``write_comparison_gallery`` emits an ``index.html`` with
one BASE ``<video>`` and one LoRA ``<video>`` per (prompt, seed), a params banner, HTML-escaped
captions (T-04-03: HTML-injection boundary), and a "generation failed" fallback for a missing mp4.
``slug`` keeps filenames alphanumeric (T-04-04: caption→path-traversal boundary).

Anti-Pattern 6: importing ``inference.grid`` must NOT pull torch / modal / ltx_core / ltx_trainer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from signet_trainer.inference.grid import (
    _image_cell,
    slug,
    write_comparison_gallery,
    write_multi_frame_gallery,
    write_reference_gallery,
    write_reskin_gallery,
)

_GRID_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "signet_trainer"
    / "inference"
    / "grid.py"
)
_FORBIDDEN_MODULES = ("torch", "modal", "ltx_core", "ltx_trainer")


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _params() -> dict:
    return {
        "steps": 30,
        "guidance": 3.0,
        "stg_scale": 1.0,
        "width": 768,
        "height": 352,
        "frames": 81,
        "lora_scale": 1.0,
    }


def _two_rows() -> list[dict]:
    return [
        {
            "prompt": "a cat riding a bicycle",
            "seed": 42,
            "base_mp4": "base/a_cat_riding_a_bicycle_s42.mp4",
            "lora_mp4": "lora/a_cat_riding_a_bicycle_s42.mp4",
        },
        {
            "prompt": "a dog surfing a wave",
            "seed": 42,
            "base_mp4": "base/a_dog_surfing_a_wave_s42.mp4",
            "lora_mp4": "lora/a_dog_surfing_a_wave_s42.mp4",
        },
    ]


# --------------------------------------------------------------------------------------------------
# Anti-Pattern-6 import confinement
# --------------------------------------------------------------------------------------------------


def test_grid_imports_no_torch_modal_or_ltx() -> None:
    # Snapshot before popping so we can RESTORE afterwards. Evicting torch (loaded by
    # conftest) without restoring poisons every later test: a subsequent `import torch`
    # re-runs torch/__init__ and double-registers the triton TORCH_LIBRARY, crashing
    # the process. Restore keeps the confinement assertion hermetic.
    saved = {mod: sys.modules.pop(mod, None) for mod in _FORBIDDEN_MODULES}
    try:
        import signet_trainer.inference.grid  # noqa: F401

        for mod in _FORBIDDEN_MODULES:
            assert mod not in sys.modules, (
                f"import-confinement violation: inference.grid transitively imported {mod!r}"
            )
    finally:
        for mod, value in saved.items():
            if value is not None:
                sys.modules[mod] = value


def test_grid_source_has_no_heavy_import() -> None:
    code = _strip_comments_and_docstrings(_GRID_SRC.read_text(encoding="utf-8"))
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
    assert not hits, f"import-confinement violation in inference/grid.py real code: {hits}"


# --------------------------------------------------------------------------------------------------
# writer I/O shape (mirrors dataset_file.write_manifest)
# --------------------------------------------------------------------------------------------------


def test_writes_index_html_and_returns_path(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "index.html"
    result = write_comparison_gallery(_two_rows(), out, _params())
    assert result == out
    assert out.exists()  # parent dirs were created
    assert out.read_text(encoding="utf-8").strip() != ""


# --------------------------------------------------------------------------------------------------
# two columns per row: exactly 2 base + 2 lora <video> tags for 2 rows
# --------------------------------------------------------------------------------------------------


def test_two_rows_render_four_video_tags(tmp_path: Path) -> None:
    out = write_comparison_gallery(_two_rows(), tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")

    assert html.count("<video") == 4  # 2 base + 2 lora
    # each given relative mp4 path is referenced
    for row in _two_rows():
        assert row["base_mp4"] in html
        assert row["lora_mp4"] in html


# --------------------------------------------------------------------------------------------------
# params banner (steps / guidance 3.0 / stg 1.0 / WxHxF / lora_scale) appears once
# --------------------------------------------------------------------------------------------------


def test_params_banner_present(tmp_path: Path) -> None:
    out = write_comparison_gallery(_two_rows(), tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")

    assert "768x352x81" in html  # WxHxF
    assert "3.0" in html  # guidance
    assert "1.0" in html  # stg / lora_scale
    assert "30" in html  # steps
    assert html.count('class="params"') == 1  # banner element appears exactly once


# --------------------------------------------------------------------------------------------------
# SECURITY: caption HTML-escaped (T-04-03)
# --------------------------------------------------------------------------------------------------


def test_caption_is_html_escaped(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "<script>alert(1)</script>",
            "seed": 42,
            "base_mp4": "base/x_s42.mp4",
            "lora_mp4": "lora/x_s42.mp4",
        }
    ]
    out = write_comparison_gallery(rows, tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in html  # never raw
    assert "&lt;script&gt;" in html  # escaped


# --------------------------------------------------------------------------------------------------
# missing mp4 -> "generation failed" fallback, not a broken <video>
# --------------------------------------------------------------------------------------------------


def test_missing_mp4_renders_failed_fallback(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "half rendered",
            "seed": 42,
            "base_mp4": "base/half_s42.mp4",
            "lora_mp4": "",  # LoRA render failed
        }
    ]
    out = write_comparison_gallery(rows, tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")

    assert html.count("<video") == 1  # only the base column has a video
    assert "generation failed" in html.lower()


# --------------------------------------------------------------------------------------------------
# slug: alphanumeric filenames (T-04-04 path-traversal mitigation)
# --------------------------------------------------------------------------------------------------


def test_slug_strips_path_traversal_and_specials() -> None:
    assert "/" not in slug("../../etc/passwd")
    assert "\\" not in slug("..\\..\\windows")
    assert slug("a cat riding!") == "a_cat_riding"
    # only alphanumerics, underscores, hyphens survive
    assert re.fullmatch(r"[A-Za-z0-9_-]+", slug("<script>alert(1)</script>"))


# --------------------------------------------------------------------------------------------------
# Phase-5 reference gallery: reference <img> cell + ref-ON/ref-OFF montage
# --------------------------------------------------------------------------------------------------


def _ref_params() -> dict:
    p = _params()
    p["first_frame_conditioning"] = "on"
    return p


def _ref_rows() -> list[dict]:
    return [
        {
            "prompt": "a sedan on a coastal road",
            "seed": 42,
            "reference_img": "ref/sedan_frame0.png",
            "ref_on_mp4": "on/a_sedan_s42.mp4",
            "ref_off_mp4": "off/a_sedan_s42.mp4",
        },
        {
            "prompt": "a sedan at dusk",
            "seed": 42,
            "reference_img": "ref/sedan_dusk_frame0.png",
            "ref_on_mp4": "on/a_sedan_dusk_s42.mp4",
            "ref_off_mp4": "off/a_sedan_dusk_s42.mp4",
        },
    ]


def test_image_cell_renders_escaped_img() -> None:
    out = _image_cell("reference", "ref/frame0.png")
    assert "<img" in out
    assert 'src="ref/frame0.png"' in out
    assert "reference" in out


def test_image_cell_escapes_malicious_src_and_label() -> None:
    out = _image_cell('<script>alert(1)</script>', '"><script>bad()</script>')
    assert "<script>" not in out  # never raw
    assert "&lt;script&gt;" in out  # escaped label
    assert "&quot;&gt;" in out or "&gt;&lt;script&gt;" in out  # src quote-escaped


def test_image_cell_empty_img_renders_no_reference_fallback() -> None:
    out = _image_cell("reference", "")
    assert "<img" not in out
    assert "no reference" in out.lower()


def test_reference_gallery_row_emits_three_cells(tmp_path: Path) -> None:
    out = write_reference_gallery(_ref_rows(), tmp_path / "index.html", _ref_params())
    html = out.read_text(encoding="utf-8")

    # per row: 1 reference <img> + 2 <video> (ref ON + ref OFF control) => 2 img, 4 video
    assert html.count("<img") == 2
    assert html.count("<video") == 4
    for row in _ref_rows():
        assert row["reference_img"] in html
        assert row["ref_on_mp4"] in html
        assert row["ref_off_mp4"] in html


def test_reference_gallery_labels_ref_on_and_off_control(tmp_path: Path) -> None:
    out = write_reference_gallery(_ref_rows(), tmp_path / "index.html", _ref_params())
    html = out.read_text(encoding="utf-8")
    assert "ref ON" in html
    assert "ref OFF" in html and "control" in html.lower()


def test_reference_gallery_banner_shows_first_frame_conditioning(tmp_path: Path) -> None:
    out = write_reference_gallery(_ref_rows(), tmp_path / "index.html", _ref_params())
    html = out.read_text(encoding="utf-8")
    assert "first_frame_conditioning" in html
    assert "on" in html


def test_reference_gallery_escapes_caption_and_img_src(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "<script>alert(1)</script>",
            "seed": 42,
            "reference_img": '"><script>evil()</script>.png',
            "ref_on_mp4": "on/x_s42.mp4",
            "ref_off_mp4": "off/x_s42.mp4",
        }
    ]
    out = write_reference_gallery(rows, tmp_path / "index.html", _ref_params())
    html = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html  # caption never raw
    assert "<script>evil()</script>" not in html  # img src never raw
    assert "&lt;script&gt;" in html


def test_reference_gallery_missing_video_renders_failed_fallback(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "half rendered",
            "seed": 42,
            "reference_img": "ref/frame0.png",
            "ref_on_mp4": "on/half_s42.mp4",
            "ref_off_mp4": "",  # ref-OFF control render failed
        }
    ]
    out = write_reference_gallery(rows, tmp_path / "index.html", _ref_params())
    html = out.read_text(encoding="utf-8")
    assert html.count("<video") == 1  # only ref-ON has a video
    assert "generation failed" in html.lower()


def test_write_comparison_gallery_still_intact(tmp_path: Path) -> None:
    """Phase-4 base-vs-LoRA path must remain unchanged alongside the new writer."""
    out = write_comparison_gallery(_two_rows(), tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")
    assert html.count("<video") == 4
    assert "<img" not in html  # comparison gallery has no reference img cell


# --------------------------------------------------------------------------------------------------
# Phase-6 multi-frame gallery: N reference thumbnails + N=2/N=3/strength-sweep/no-ref columns (SC#3)
# --------------------------------------------------------------------------------------------------


def _mf_rows() -> list[dict]:
    return [
        # row A: same-clip keyframes (frame0 + frame48) — an N=2 reference set
        {
            "prompt": "a sedan pulling out of a driveway",
            "seed": 42,
            "reference_imgs": [
                "ref/sedan_frame0.png",
                "ref/sedan_frame48.png",
            ],
            "n2_mp4": "n2/a_sedan_s42.mp4",
            "n3_mp4": "n3/a_sedan_s42.mp4",
            "strength_lo_mp4": "str/a_sedan_s42_lo.mp4",
            "strength_hi_mp4": "str/a_sedan_s42_hi.mp4",
            "no_ref_mp4": "control/a_sedan_s42.mp4",
            # Config-driven sweep labels (WR-01): fns.py threads plan_multi_frame_columns'
            # labels — the ACTUAL conditioning_strength_range endpoints (0.3/1.0 in the shipped
            # example config), never a hardcoded "0.5".
            "strength_lo_label": "mid strength 0.3",
            "strength_hi_label": "mid strength 1.0",
        },
        # row B: three external held-out references — an N=3 reference set
        {
            "prompt": "a sedan at dusk on a coastal road",
            "seed": 42,
            "reference_imgs": [
                "ref/held0.png",
                "ref/held1.png",
                "ref/held2.png",
            ],
            "n2_mp4": "n2/a_sedan_dusk_s42.mp4",
            "n3_mp4": "n3/a_sedan_dusk_s42.mp4",
            "strength_lo_mp4": "str/a_sedan_dusk_s42_lo.mp4",
            "strength_hi_mp4": "str/a_sedan_dusk_s42_hi.mp4",
            "no_ref_mp4": "control/a_sedan_dusk_s42.mp4",
            "strength_lo_label": "mid strength 0.3",
            "strength_hi_label": "mid strength 1.0",
        },
    ]


def test_multi_frame_gallery_writes_index_and_returns_path(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "index.html"
    result = write_multi_frame_gallery(_mf_rows(), out, _params())
    assert result == out
    assert out.exists()  # parent dirs were created
    assert out.read_text(encoding="utf-8").strip() != ""


def test_multi_frame_gallery_renders_all_columns(tmp_path: Path) -> None:
    out = write_multi_frame_gallery(_mf_rows(), tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")

    # row A (2 refs) + row B (3 refs) => 5 <img>; 5 output <video> columns per row => 10 <video>
    assert html.count("<img") == 5
    assert html.count("<video") == 10
    # every output-column label present — the sweep labels are CONFIG-DRIVEN (WR-01): they carry
    # the row's strength_lo_label/strength_hi_label (the actual conditioning_strength_range
    # endpoints, 0.3/1.0 for the shipped config), never the old hardcoded "mid strength 0.5".
    assert "N=2" in html
    assert "N=3" in html
    assert "mid strength 0.3" in html
    assert "mid strength 1.0" in html
    assert "mid strength 0.5" not in html  # the WR-01 hardcoded literal must be gone
    assert "no reference (control)" in html.lower() or "no reference" in html.lower()
    # every given relative path is referenced
    for row in _mf_rows():
        for img in row["reference_imgs"]:
            assert img in html
        for key in ("n2_mp4", "n3_mp4", "strength_lo_mp4", "strength_hi_mp4", "no_ref_mp4"):
            assert row[key] in html


def test_multi_frame_gallery_sweep_labels_fall_back_when_absent(tmp_path: Path) -> None:
    """A row without label keys gets the generic (lo)/(hi) fallbacks — never a strength literal.

    WR-01 guard: the writer must not invent a numeric strength; only the caller (fns.py, via
    plan_multi_frame_columns) knows the real config-driven sweep endpoints.
    """
    rows = [{k: v for k, v in _mf_rows()[0].items() if not k.endswith("_label")}]
    out = write_multi_frame_gallery(rows, tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")
    assert "mid strength (lo)" in html
    assert "mid strength (hi)" in html
    assert "mid strength 0.5" not in html  # no hardcoded strength literal, even as a fallback


def test_multi_frame_gallery_n3_row_renders_exactly_three_img_cells(tmp_path: Path) -> None:
    # a single N=3 row must render exactly 3 reference <img> cells
    row = _mf_rows()[1]
    out = write_multi_frame_gallery([row], tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")
    assert html.count("<img") == 3


def test_multi_frame_gallery_seed_appears_in_each_block(tmp_path: Path) -> None:
    out = write_multi_frame_gallery(_mf_rows(), tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")
    assert html.count("seed 42") == 2  # one seed banner per row


def test_multi_frame_gallery_escapes_malicious_caption_and_img_src(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "<script>alert(1)</script>",
            "seed": 42,
            "reference_imgs": ['"><script>evil()</script>.png'],
            "n2_mp4": "n2/x_s42.mp4",
            "n3_mp4": "n3/x_s42.mp4",
            "strength_lo_mp4": "str/x_s42_lo.mp4",
            "strength_hi_mp4": "str/x_s42_hi.mp4",
            "no_ref_mp4": "control/x_s42.mp4",
        }
    ]
    out = write_multi_frame_gallery(rows, tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html  # caption never raw
    assert "<script>evil()</script>" not in html  # img src never raw
    assert "&lt;script&gt;" in html  # escaped, not raw


def test_multi_frame_gallery_prompt_slug_is_path_safe() -> None:
    # a path-traversal-y prompt reduces to a filename token with only [A-Za-z0-9_-]
    token = slug("../../etc/passwd for a sedan")
    assert "/" not in token
    assert "\\" not in token
    assert re.fullmatch(r"[A-Za-z0-9_-]+", token)


def test_multi_frame_gallery_missing_mp4_renders_failed_fallback(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "half rendered multi-frame",
            "seed": 42,
            "reference_imgs": ["ref/frame0.png", "ref/frame48.png"],
            "n2_mp4": "n2/half_s42.mp4",
            "n3_mp4": "",  # N=3 render failed
            "strength_lo_mp4": "str/half_s42_lo.mp4",
            "strength_hi_mp4": "str/half_s42_hi.mp4",
            "no_ref_mp4": "control/half_s42.mp4",
        }
    ]
    out = write_multi_frame_gallery(rows, tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")
    assert html.count("<video") == 4  # 5 output columns minus the failed N=3
    assert "generation failed" in html.lower()


def test_multi_frame_gallery_empty_reference_renders_no_reference_fallback(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "prompt": "missing keyframe",
            "seed": 42,
            "reference_imgs": ["ref/frame0.png", ""],  # second keyframe missing
            "n2_mp4": "n2/x_s42.mp4",
            "n3_mp4": "n3/x_s42.mp4",
            "strength_lo_mp4": "str/x_s42_lo.mp4",
            "strength_hi_mp4": "str/x_s42_hi.mp4",
            "no_ref_mp4": "control/x_s42.mp4",
        }
    ]
    out = write_multi_frame_gallery(rows, tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")
    assert html.count("<img") == 1  # only the present keyframe renders an <img>
    assert "no reference" in html.lower()


def test_write_reference_gallery_still_intact_alongside_multi_frame(tmp_path: Path) -> None:
    """Phase-5 single-frame reference path must remain unchanged alongside the new writer."""
    out = write_reference_gallery(_ref_rows(), tmp_path / "index.html", _ref_params())
    html = out.read_text(encoding="utf-8")
    assert html.count("<img") == 2  # one reference per row, unchanged
    assert html.count("<video") == 4  # ref-ON + ref-OFF per row, unchanged


# --------------------------------------------------------------------------------------------------
# Phase-7 IC-LoRA re-skin gallery: 4-col D-7-GRIDCOL montage (REF-03 / SC#3) — the phase's single UI
# surface, per the approved 07-UI-SPEC. CPU-only (torch-free, GPU-free).
# --------------------------------------------------------------------------------------------------


def _reskin_params() -> dict:
    return {
        "steps": 30,
        "guidance": 3.0,
        "stg_scale": 1.0,
        "width": 768,
        "height": 512,
        "frames": 25,
        "lora_scale": 1.0,
        "conditioning_mode": "ic_lora",
        "reference_downscale_factor": 1,
    }


def _reskin_rows() -> list[dict]:
    return [
        # Row A: in-domain held-out sedan clip
        {
            "row": "A",
            "row_type": "held-out sedan",
            "clip_name": "sedan_driveway_01",
            "prompt": "a sedan made of chrome, cyberpunk night",
            "seed": 42,
            "steps": 30,
            "original_mp4": "orig/sedan_driveway_01_s42.mp4",
            "seg_map_mp4": "seg/sedan_driveway_01_s42.mp4",
            "ic_lora_mp4": "reskin/sedan_driveway_01_s42.mp4",
            "base_no_ref_mp4": "control/sedan_driveway_01_s42.mp4",
        },
        # Row B: generalization — one external video
        {
            "row": "B",
            "row_type": "external",
            "clip_name": "external_coastal_road",
            "prompt": "a vintage convertible on a coastal road at dusk",
            "seed": 42,
            "steps": 30,
            "original_mp4": "orig/external_coastal_road_s42.mp4",
            "seg_map_mp4": "seg/external_coastal_road_s42.mp4",
            "ic_lora_mp4": "reskin/external_coastal_road_s42.mp4",
            "base_no_ref_mp4": "control/external_coastal_road_s42.mp4",
        },
    ]


def test_reskin_gallery_writes_index_and_returns_path(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "index.html"
    result = write_reskin_gallery(_reskin_rows(), out, _reskin_params())
    assert result == out
    assert out.exists()  # parent dirs were created
    assert out.read_text(encoding="utf-8").strip() != ""


def test_reskin_gallery_renders_exactly_four_cells_per_row(tmp_path: Path) -> None:
    # (a) EXACTLY four cells left-to-right per row (D-7-GRIDCOL). Two full rows => 8 cells, 8 videos.
    out = write_reskin_gallery(_reskin_rows(), tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert html.count('<div class="cell') == 8  # matches both "cell" and "cell control"
    assert html.count("<video") == 8  # 4 motion-first videos per row, all tiles present
    # single-row check: exactly four cells in one .row
    single = write_reskin_gallery(_reskin_rows()[:1], tmp_path / "single.html", _reskin_params())
    assert single.read_text(encoding="utf-8").count('<div class="cell') == 4


def test_reskin_gallery_has_all_four_column_labels(tmp_path: Path) -> None:
    # (b) all four UI-SPEC column labels present, left-to-right
    out = write_reskin_gallery(_reskin_rows(), tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert "original footage" in html
    assert "semantic map (what the adapter sees)" in html
    assert "IC-LoRA re-skin (new prompt)" in html
    assert "no reference (control)" in html


def test_reskin_gallery_has_control_badge_and_marker(tmp_path: Path) -> None:
    # (c) accent CONTROL badge + .cell.control divergence marker on column 4
    out = write_reskin_gallery(_reskin_rows(), tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert "CONTROL — no reference" in html  # exact UI-SPEC badge copy
    assert 'class="badge-control"' in html
    assert 'class="cell control"' in html  # the .control marker cell
    # one control cell + one badge per row
    assert html.count('class="cell control"') == 2
    assert html.count('class="badge-control"') == 2


def test_reskin_gallery_has_pass_fail_criteria_block(tmp_path: Path) -> None:
    # (d) the mandatory 'what to look for' PASS/FAIL block + per-row PASS line (async self-review)
    out = write_reskin_gallery(_reskin_rows(), tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert 'class="criteria"' in html
    assert "PASS =" in html
    assert "FAIL =" in html
    assert 'class="grid-title"' in html  # page <h1> title present
    assert "does the seg map steer the output?" in html
    # per-row PASS line, one per row
    assert html.count("PASS if:") == 2


def test_reskin_gallery_escapes_hostile_caption(tmp_path: Path) -> None:
    # (e) a caption / clip-name with HTML-special chars is escaped — no raw injection (T-07-04-01)
    rows = [
        {
            "row": "A",
            "row_type": "held-out sedan",
            "clip_name": '"><script>evil()</script>',
            "prompt": "<script>alert(1)</script>",
            "seed": 42,
            "steps": 30,
            "original_mp4": "orig/x_s42.mp4",
            "seg_map_mp4": "seg/x_s42.mp4",
            "ic_lora_mp4": "reskin/x_s42.mp4",
            "base_no_ref_mp4": "control/x_s42.mp4",
        }
    ]
    out = write_reskin_gallery(rows, tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html  # prompt never raw
    assert "<script>evil()</script>" not in html  # clip-name never raw
    assert "&lt;script&gt;" in html  # escaped, not raw


def test_reskin_gallery_missing_seg_map_renders_unavailable_fallback(tmp_path: Path) -> None:
    # (f) a missing seg_map_mp4 renders the 'seg map unavailable' fallback, not a crash/broken tile
    row = dict(_reskin_rows()[0])
    row["seg_map_mp4"] = ""  # seg-map tile unavailable
    out = write_reskin_gallery([row], tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert "seg map unavailable" in html
    assert html.count("<video") == 3  # 4 columns minus the missing seg-map tile
    # the other fallbacks stay column-appropriate
    row2 = dict(_reskin_rows()[0])
    row2["base_no_ref_mp4"] = ""
    row2["ic_lora_mp4"] = ""
    html2 = write_reskin_gallery([row2], tmp_path / "b.html", _reskin_params()).read_text(
        encoding="utf-8"
    )
    assert "no reference" in html2  # control fallback
    assert "generation failed" in html2  # re-skin fallback


def test_reskin_gallery_col1_renders_original_mp4_when_present(tmp_path: Path) -> None:
    # 07-15 GAP-2: col-1 sources the real ORIGINAL footage from the row's original_mp4 (config-driven,
    # wired via conditioning.original_videos). Present -> the path renders as a <video src=...>.
    out = write_reskin_gallery(_reskin_rows(), tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert "orig/sedan_driveway_01_s42.mp4" in html  # Row A original present in col-1
    assert "orig/external_coastal_road_s42.mp4" in html  # Row B original present in col-1
    assert "original not staged" not in html  # no col-1 fallback when originals present


def test_reskin_gallery_missing_original_renders_not_staged_fallback(tmp_path: Path) -> None:
    # 07-15 GAP-2: a missing original_mp4 renders the honest 'original not staged' fallback (NOT the
    # misleading 'generation failed'), and the col-1 tile drops from the video count. Mirrors the
    # col-2 missing-seg-map fallback test.
    row = dict(_reskin_rows()[0])
    row["original_mp4"] = ""  # col-1 original unavailable
    out = write_reskin_gallery([row], tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert "original not staged" in html  # honest col-1 fallback label
    assert "generation failed" not in html  # NOT the old misleading label for col-1
    assert html.count("<video") == 3  # 4 columns minus the missing col-1 original tile


def test_reskin_gallery_params_banner_is_config_driven(tmp_path: Path) -> None:
    # conditioning_mode + reference_downscale_factor surfaced from params (config-first, WR-01)
    out = write_reskin_gallery(_reskin_rows(), tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert "conditioning_mode=ic_lora" in html
    assert "reference_downscale_factor=1" in html
    assert html.count('class="params"') == 1


def test_reskin_gallery_seg_map_still_image_uses_img_cell(tmp_path: Path) -> None:
    # forward-compat: a still seg_map_img renders via the polymorphic _image_cell path (<img>)
    row = dict(_reskin_rows()[0])
    del row["seg_map_mp4"]
    row["seg_map_img"] = "seg/sedan_driveway_01_frame0.png"
    out = write_reskin_gallery([row], tmp_path / "index.html", _reskin_params())
    html = out.read_text(encoding="utf-8")
    assert "<img" in html
    assert "seg/sedan_driveway_01_frame0.png" in html
    assert html.count("<video") == 3  # original + re-skin + control remain videos


def test_write_multi_frame_gallery_still_intact_alongside_reskin(tmp_path: Path) -> None:
    """Phase-6 multi-frame path must remain unchanged alongside the Phase-7 re-skin writer."""
    out = write_multi_frame_gallery(_mf_rows(), tmp_path / "index.html", _params())
    html = out.read_text(encoding="utf-8")
    assert html.count("<img") == 5  # unchanged
    assert html.count("<video") == 10  # unchanged
    # the shipped multi-frame video tiles keep their exact (non-motion-first) attribute set
    assert "autoplay loop muted playsinline controls" not in html
