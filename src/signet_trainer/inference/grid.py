"""inference.grid — the base-vs-LoRA HTML montage writer (INFR-02).

Ports enochiatron ``scripts/infer/compare.py::write_comparison_gallery`` (624-770, VERIFIED),
relabeling its two video columns BASE (left) and LoRA (right). For each (prompt, seed) row it
emits one ``.comparison`` block with two ``<video controls>`` columns at the same seed=42
(D-GRID-2), a params banner, and a "generation failed" fallback when an mp4 path is missing.

Mirrors ``data/dataset_file.py::write_manifest``'s I/O shape: pure Python, torch-free, GPU-free,
explicit ``encoding="utf-8"``, ``mkdir(parents=True)``, return the written ``Path`` — so
``tests/test_grid_html.py`` runs on Windows/CI without a GPU (Anti-Pattern 6).

SECURITY:
  - T-04-03 (HTML injection): every interpolated caption/prompt flows through ``html.escape``
    — the 6 clip captions are semi-trusted local text crossing into ``index.html``.
  - T-04-04 (path traversal): ``slug`` restricts a prompt-derived filename to alphanumerics,
    underscores and hyphens; the gallery references mp4s only by their given relative path.

Kept stdlib-only (``html`` + ``pathlib`` + ``re``) so it never breaks the import-confinement scan.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

__all__ = [
    "slug",
    "write_comparison_gallery",
    "write_multi_frame_gallery",
    "write_reference_gallery",
    "write_reskin_gallery",
]

# Max slug length — keeps prompt-derived filenames bounded (defensive, not correctness-critical).
_SLUG_MAX_LEN = 60


def slug(text: str) -> str:
    """Reduce arbitrary caption text to a filesystem-safe token (T-04-04 mitigation).

    Non-alphanumerics collapse to single underscores; leading/trailing separators are stripped;
    the result contains ONLY ``[A-Za-z0-9_-]`` so it can never encode a path separator (``/`` /
    ``\\``) or a ``..`` traversal. Returns ``"untitled"`` for an empty/all-special input.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_-")
    cleaned = cleaned[:_SLUG_MAX_LEN].strip("_-")
    return cleaned or "untitled"


def _video_cell(label: str, mp4: str) -> str:
    """One column: a ``<video>`` when ``mp4`` is truthy, else a 'generation failed' fallback."""
    safe_label = html.escape(label)
    if mp4:
        safe_src = html.escape(mp4, quote=True)
        media = f'<video controls loop muted src="{safe_src}"></video>'
    else:
        media = '<div class="failed">generation failed</div>'
    return f'<div class="cell"><div class="col-label">{safe_label}</div>{media}</div>'


def _image_cell(label: str, img: str) -> str:
    """One column: an ``<img>`` reference thumbnail when ``img`` is truthy, else a 'no reference'
    fallback. Mirrors ``_video_cell`` exactly — ``label`` through ``html.escape`` and ``src``
    through ``html.escape(..., quote=True)`` (T-05-02 / T-04-03 output-encoding control)."""
    safe_label = html.escape(label)
    if img:
        safe_src = html.escape(img, quote=True)
        media = f'<img loop src="{safe_src}" alt="{safe_label}">'
    else:
        media = '<div class="failed">no reference</div>'
    return f'<div class="cell"><div class="col-label">{safe_label}</div>{media}</div>'


def _reskin_video_cell(
    label: str,
    mp4: str,
    *,
    control: bool = False,
    fallback: str = "generation failed",
) -> str:
    """One re-skin column — a motion-first ``<video>`` when ``mp4`` is truthy, else a labeled
    ``.failed`` fallback (``fallback``: ``generation failed`` / ``no reference`` / ``seg map
    unavailable``).

    Emits the UI-SPEC motion-first attribute set ``autoplay loop muted playsinline controls`` so
    the montage animates on open for motion judgment while still letting the reviewer scrub — this
    is deliberately distinct from ``_video_cell`` (``controls loop muted``) so the shipped Phase
    4-6 galleries keep their exact playback attributes (UI-SPEC line 150: the re-skin writer emits
    its own attribute set). When ``control=True`` the cell carries the ``.control`` divergence
    marker class and the accent ``CONTROL — no reference`` badge. Every ``label``/``src``/``fallback``
    flows through ``html.escape`` (T-07-04-01)."""
    safe_label = html.escape(label)
    cell_class = "cell control" if control else "cell"
    badge = (
        '<span class="badge-control">CONTROL — no reference</span>' if control else ""
    )
    if mp4:
        safe_src = html.escape(mp4, quote=True)
        media = f'<video autoplay loop muted playsinline controls src="{safe_src}"></video>'
    else:
        media = f'<div class="failed">{html.escape(fallback)}</div>'
    return (
        f'<div class="{cell_class}">'
        f'<div class="col-label">{safe_label}{badge}</div>{media}</div>'
    )


def _reference_block(row: dict) -> str:
    """One ``.comparison`` block: caption + seed banner + reference / ref-ON / ref-OFF columns.

    Columns (same seed across all three — the house A/B rule, D-GRID):
    ``[ reference <img> | ref ON <video> | ref OFF (control) <video> ]``. The ref-OFF control
    is the SC#3 divergence signal (same prompt+seed, ``condition_image=None``)."""
    prompt = html.escape(str(row.get("prompt", "")))
    seed = html.escape(str(row.get("seed", "")))
    reference = _image_cell("reference", row.get("reference_img", ""))
    ref_on = _video_cell("ref ON", row.get("ref_on_mp4", ""))
    ref_off = _video_cell("ref OFF (control)", row.get("ref_off_mp4", ""))
    return (
        '<section class="comparison">'
        f'<h2 class="caption">{prompt}</h2>'
        f'<div class="seed">seed {seed}</div>'
        f'<div class="row">{reference}{ref_on}{ref_off}</div>'
        "</section>"
    )


def _multi_frame_block(row: dict) -> str:
    """One ``.comparison`` block for the multi-frame PASS/FAIL grid (SC#3 artifact).

    Renders, as an ORDERED list of cells at the SAME seed (D-6-GRIDROWS/GRIDN/STRENGTHCOL):
    one ``_image_cell`` per keyframe reference thumbnail (``row["reference_imgs"]``), then
    ``_video_cell`` columns for ``N=2``, ``N=3``, the strength-sweep pair and the no-reference
    control. The sweep labels are CONFIG-DRIVEN (D-NOHARDCODE / WR-01): they come from the row's
    ``strength_lo_label`` / ``strength_hi_label`` keys — the ``plan_multi_frame_columns`` labels
    carrying the actual ``conditioning_strength_range`` endpoints — never a hardcoded strength
    literal (the shipped config sweeps 0.3/1.0, not 0.5/1.0). A falsy reference/mp4 path routes
    through the existing 'no reference' / 'generation failed' fallback. Every label/prompt/seed
    and every img/mp4 ``src`` flows through ``html.escape`` / ``slug`` via the reused cell
    builders (T-06-03)."""
    prompt = html.escape(str(row.get("prompt", "")))
    seed = html.escape(str(row.get("seed", "")))

    reference_cells = "".join(
        _image_cell(f"ref {i}", img)
        for i, img in enumerate(row.get("reference_imgs", []), start=1)
    )
    n2 = _video_cell("N=2", row.get("n2_mp4", ""))
    n3 = _video_cell("N=3", row.get("n3_mp4", ""))
    # Config-driven sweep labels (WR-01): the row carries the planner's label (the real swept
    # strengths from conditioning_strength_range). The fallback names the endpoint, not a number.
    strength_lo = _video_cell(
        str(row.get("strength_lo_label", "mid strength (lo)")), row.get("strength_lo_mp4", "")
    )
    strength_hi = _video_cell(
        str(row.get("strength_hi_label", "mid strength (hi)")), row.get("strength_hi_mp4", "")
    )
    no_ref = _video_cell("no reference (control)", row.get("no_ref_mp4", ""))

    return (
        '<section class="comparison">'
        f'<h2 class="caption">{prompt}</h2>'
        f'<div class="seed">seed {seed}</div>'
        f'<div class="row">{reference_cells}{n2}{n3}{strength_lo}{strength_hi}{no_ref}</div>'
        "</section>"
    )


# The mandatory "what to look for" PASS/FAIL block copy (UI-SPEC line 109, verbatim). The literal
# ``&`` is emitted as ``&amp;`` since this is authored HTML, not user input.
_RESKIN_CRITERIA_PASS = (
    "column 3 (IC-LoRA re-skin) follows the layout &amp; motion of column 1 (original), as "
    "described by column 2 (seg map), while column 4 (same prompt, no reference) visibly diverges."
)
_RESKIN_CRITERIA_FAIL = (
    "column 3 looks like column 4 (the reference had no effect) or ignores the seg-map layout."
)
# Per-row PASS condition default (UI-SPEC line 110). Overridable per row via ``criteria_line``.
_RESKIN_ROW_PASS_LINE = (
    "col 3 keeps the road / sky / vehicle layout from col 2 and diverges from col 4."
)


def _reskin_block(row: dict) -> str:
    """One ``.comparison`` block for the IC-LoRA re-skin PASS/FAIL grid (REF-03 / SC#3 artifact).

    Renders, top-to-bottom, per the approved 07-UI-SPEC Grid Layout Contract:
    ``.row-label`` (``Row {A|B} · {row_type} · {clip_name}``) → ``.reskin-prompt``
    (``re-skin prompt: {prompt}``) → ``.seed`` (``seed 42 · steps {N}``) → ``.criteria-line``
    (accent ``PASS if:`` label + per-row condition) → ``.row`` (flex, ``nowrap``) with EXACTLY four
    cells left-to-right (D-7-GRIDCOL): col1 original footage, col2 semantic map (video, or an
    ``_image_cell`` when the row carries a still ``seg_map_img``), col3 IC-LoRA re-skin, col4 the
    no-reference control (``.control`` marker + accent ``CONTROL — no reference`` badge). Every
    interpolated caption/prompt/clip-name/path flows through ``html.escape`` via the escaped cell
    builders (T-07-04-01). Missing tiles fall back with the row-appropriate label
    (col1 ``original not staged`` / col3 ``generation failed`` / col4 ``no reference`` /
    col2 ``seg map unavailable``)."""
    row_id = html.escape(str(row.get("row", "")))
    row_type = html.escape(str(row.get("row_type", "")))
    clip_name = html.escape(str(row.get("clip_name", "")))
    prompt = html.escape(str(row.get("prompt", "")))
    seed = html.escape(str(row.get("seed", 42)))
    steps = html.escape(str(row.get("steps", "?")))
    criteria_line = html.escape(str(row.get("criteria_line", _RESKIN_ROW_PASS_LINE)))

    col1 = _reskin_video_cell(
        "original footage", row.get("original_mp4", ""), fallback="original not staged"
    )
    # Col 2 is polymorphic (UI-SPEC forward-compat): a still ``seg_map_img`` renders via the
    # existing escaped ``_image_cell`` (deferred image modality); otherwise the seg-map palette
    # video with the 'seg map unavailable' fallback.
    if row.get("seg_map_img"):
        col2 = _image_cell(
            "semantic map (what the adapter sees)", row.get("seg_map_img", "")
        )
    else:
        col2 = _reskin_video_cell(
            "semantic map (what the adapter sees)",
            row.get("seg_map_mp4", ""),
            fallback="seg map unavailable",
        )
    col3 = _reskin_video_cell(
        "IC-LoRA re-skin (new prompt)", row.get("ic_lora_mp4", ""), fallback="generation failed"
    )
    col4 = _reskin_video_cell(
        "no reference (control)",
        row.get("base_no_ref_mp4", ""),
        control=True,
        fallback="no reference",
    )

    return (
        '<section class="comparison">'
        f'<div class="row-label">Row {row_id} · {row_type} · {clip_name}</div>'
        f'<div class="reskin-prompt">re-skin prompt: {prompt}</div>'
        f'<div class="seed">seed {seed} · steps {steps}</div>'
        f'<div class="criteria-line"><b>PASS if:</b> {criteria_line}</div>'
        f'<div class="row">{col1}{col2}{col3}{col4}</div>'
        "</section>"
    )


def _comparison_block(row: dict) -> str:
    """One ``.comparison`` block: caption + seed banner + BASE / LoRA columns."""
    prompt = html.escape(str(row.get("prompt", "")))
    seed = html.escape(str(row.get("seed", "")))
    base = _video_cell("BASE", row.get("base_mp4", ""))
    lora = _video_cell("LoRA", row.get("lora_mp4", ""))
    return (
        '<section class="comparison">'
        f'<h2 class="caption">{prompt}</h2>'
        f'<div class="seed">seed {seed}</div>'
        f'<div class="row">{base}{lora}</div>'
        "</section>"
    )


def _params_banner(params: dict) -> str:
    """The single params banner: steps / guidance / stg / WxHxF / lora_scale (all escaped)."""
    width = params.get("width", "?")
    height = params.get("height", "?")
    frames = params.get("frames", "?")
    dims = html.escape(f"{width}x{height}x{frames}")
    steps = html.escape(str(params.get("steps", "?")))
    guidance = html.escape(str(params.get("guidance", "?")))
    stg = html.escape(str(params.get("stg_scale", "?")))
    lora_scale = html.escape(str(params.get("lora_scale", "?")))
    # The Phase-5 reference gallery passes first_frame_conditioning=on/off; the Phase-4
    # comparison path omits it, so its banner is unchanged (key absent → nothing appended).
    ffc_suffix = ""
    if "first_frame_conditioning" in params:
        ffc = html.escape(str(params.get("first_frame_conditioning", "?")))
        ffc_suffix = f" &middot; first_frame_conditioning={ffc}"
    # The Phase-7 re-skin gallery passes conditioning_mode + reference_downscale_factor (UI-SPEC
    # line 108); older galleries omit both keys, so their banners are unchanged (config-driven,
    # never hardcoded — absent key → nothing appended).
    cm_suffix = ""
    if "conditioning_mode" in params:
        cm = html.escape(str(params.get("conditioning_mode", "?")))
        cm_suffix = f" &middot; conditioning_mode={cm}"
    rdf_suffix = ""
    if "reference_downscale_factor" in params:
        rdf = html.escape(str(params.get("reference_downscale_factor", "?")))
        rdf_suffix = f" &middot; reference_downscale_factor={rdf}"
    return (
        '<div class="params">'
        f"steps={steps} &middot; guidance={guidance} &middot; stg={stg} "
        f"&middot; {dims} &middot; lora_scale={lora_scale}"
        f"{ffc_suffix}{cm_suffix}{rdf_suffix}"
        "</div>"
    )


_STYLE = (
    "body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:24px}"
    ".params{background:#222;padding:8px 12px;border-radius:6px;margin-bottom:24px;font-size:14px}"
    ".comparison{margin-bottom:32px;border-bottom:1px solid #333;padding-bottom:16px}"
    ".caption{font-size:16px;margin:0 0 4px}"
    ".seed{font-size:12px;color:#999;margin-bottom:8px}"
    ".row{display:flex;gap:16px}"
    ".cell{flex:1}.col-label{font-size:12px;color:#bbb;margin-bottom:4px}"
    "video{width:100%;background:#000;border-radius:4px}"
    ".failed{padding:40px;text-align:center;color:#c66;background:#1a1a1a;border-radius:4px}"
    # Phase-7 re-skin gallery additions (UI-SPEC lines 140-148) — appended, existing rules unchanged.
    ".grid-title{font-size:20px;font-weight:600;margin:0 0 8px}"
    ".criteria{background:#1a1a1a;border-left:3px solid #d9a441;padding:12px 16px;"
    "margin-bottom:24px;font-size:14px;line-height:1.5}"
    ".criteria b{color:#d9a441}"
    ".row-label{font-size:16px;font-weight:600;margin:0 0 4px}"
    ".reskin-prompt{font-size:14px;color:#ccc;margin:0 0 4px}"
    ".criteria-line{font-size:12px;color:#bbb;margin:4px 0 8px}"
    ".criteria-line b{color:#d9a441;font-weight:600}"
    ".badge-control{display:inline-block;color:#d9a441;border:1px solid #d9a441;"
    "border-radius:4px;padding:0 8px;font-size:12px;margin-left:8px}"
    ".cell.control{border-left:1px dashed #444;padding-left:16px}"
)


def write_comparison_gallery(
    rows: list[dict], out_path: str | Path, params: dict
) -> Path:
    """Write a base-vs-LoRA ``index.html`` montage and return its ``Path``.

    Args:
        rows: one dict per (prompt, seed) with ``prompt``, ``seed``, ``base_mp4``, ``lora_mp4``
            (mp4 paths RELATIVE to ``out_path``'s parent; a falsy mp4 → 'generation failed').
        out_path: destination ``index.html`` path (parent dirs are created).
        params: sampling params for the banner (``steps``, ``guidance``, ``stg_scale``,
            ``width``, ``height``, ``frames``, ``lora_scale``).

    Returns:
        The resolved ``Path`` written.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    blocks = "".join(_comparison_block(row) for row in rows)
    doc = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>base vs LoRA</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"{_params_banner(params)}"
        f"{blocks}"
        "</body></html>"
    )

    with out.open("w", encoding="utf-8") as f:  # utf-8 explicit (Phase-1 carry-forward)
        f.write(doc)
    return out


def write_multi_frame_gallery(
    rows: list[dict], out_path: str | Path, params: dict
) -> Path:
    """Write the multi-frame reference PASS/FAIL montage ``index.html`` and return its ``Path`` (SC#3).

    Each row renders, at the SAME seed=42 (D-6-GRIDROWS/GRIDN/STRENGTHCOL), an ordered list of
    cells: N keyframe reference thumbnails (``row["reference_imgs"]``) followed by the ``N=2``,
    ``N=3``, strength-sweep (labeled by the row's config-driven ``strength_lo_label`` /
    ``strength_hi_label`` — the ``conditioning_strength_range`` endpoints, WR-01) and
    no-reference-control output columns — the artifact the operator reads to judge that each keyframe
    influences its region and the strength dial works. Reuses
    ``_image_cell``/``_video_cell``/``_params_banner``/``_STYLE`` — every caption,
    seed, img ``src`` and mp4 ``src`` flows through ``html.escape`` (T-06-03). Stdlib-only,
    torch-free, CPU/Windows-testable (mirrors ``write_reference_gallery``).

    Args:
        rows: one dict per prompt with ``prompt``, ``seed``, ``reference_imgs`` (list of relative
            img paths; a falsy entry → 'no reference'), the output mp4 paths ``n2_mp4``,
            ``n3_mp4``, ``strength_lo_mp4``, ``strength_hi_mp4``, ``no_ref_mp4`` (relative; a falsy
            mp4 → 'generation failed'), and the config-driven sweep-column labels
            ``strength_lo_label`` / ``strength_hi_label`` (from ``plan_multi_frame_columns``;
            omitted → the generic 'mid strength (lo)'/'(hi)' fallbacks).
        out_path: destination ``index.html`` path (parent dirs are created).
        params: sampling params for the banner (``steps``, ``guidance``, ``stg_scale``, ``width``,
            ``height``, ``frames``, ``lora_scale``).

    Returns:
        The resolved ``Path`` written.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    blocks = "".join(_multi_frame_block(row) for row in rows)
    doc = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>multi-frame: N references &middot; N=2/N=3 &middot; strength sweep</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"{_params_banner(params)}"
        f"{blocks}"
        "</body></html>"
    )

    with out.open("w", encoding="utf-8") as f:  # utf-8 explicit (Phase-1 carry-forward)
        f.write(doc)
    return out


def write_reference_gallery(
    rows: list[dict], out_path: str | Path, params: dict
) -> Path:
    """Write the single-frame reference montage ``index.html`` and return its ``Path`` (SC#3).

    Each row renders three columns at the SAME seed — ``[ reference <img> | ref ON <video> |
    ref OFF (control) <video> ]`` (D-GRID). The ref-OFF control column is the divergence signal:
    same prompt+seed rendered with ``condition_image=None``. Reuses ``_video_cell``, ``_image_cell``,
    ``_params_banner`` and ``_STYLE`` — all captions/paths/img-src flow through ``html.escape``
    (T-05-02). Stdlib-only, torch-free, CPU/Windows-testable (mirrors ``write_comparison_gallery``).

    Args:
        rows: one dict per prompt with ``prompt``, ``seed``, ``reference_img`` (relative img path;
            falsy → 'no reference'), ``ref_on_mp4`` and ``ref_off_mp4`` (relative mp4 paths; a falsy
            mp4 → 'generation failed').
        out_path: destination ``index.html`` path (parent dirs are created).
        params: sampling params for the banner; include ``first_frame_conditioning`` (``"on"``/``"off"``)
            to surface the conditioning indicator.

    Returns:
        The resolved ``Path`` written.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    blocks = "".join(_reference_block(row) for row in rows)
    doc = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>reference: ref ON vs ref OFF</title>"
        f"<style>{_STYLE}</style></head><body>"
        f"{_params_banner(params)}"
        f"{blocks}"
        "</body></html>"
    )

    with out.open("w", encoding="utf-8") as f:  # utf-8 explicit (Phase-1 carry-forward)
        f.write(doc)
    return out


def write_reskin_gallery(
    rows: list[dict], out_path: str | Path, params: dict
) -> Path:
    """Write the IC-LoRA re-skin PASS/FAIL montage ``index.html`` and return its ``Path`` (REF-03 / SC#3).

    The single UI surface of Phase 7, per the approved 07-UI-SPEC. A static, self-contained page
    (no external network deps) with a page ``<h1>`` title, the mandatory "what to look for"
    PASS/FAIL criteria block, a ``_params_banner`` (extended with ``conditioning_mode`` /
    ``reference_downscale_factor``), and one ``.comparison`` block per row. Each row renders the
    D-7-GRIDCOL 4-column story left-to-right at the SAME seed 42 (house rule): original footage ·
    semantic map · IC-LoRA re-skin · no-reference control (with the accent ``CONTROL — no
    reference`` badge). Motion-first ``<video autoplay loop muted playsinline controls>`` tiles so
    the montage animates on open. Reuses ``_image_cell`` / ``_reskin_video_cell`` /
    ``_params_banner`` / ``_STYLE`` — every caption/prompt/clip-name/path flows through
    ``html.escape`` (T-07-04-01). Stdlib-only, torch-free, CPU/Windows-testable (mirrors
    ``write_multi_frame_gallery``). 07-11 populates it with trained-adapter outputs.

    Args:
        rows: one dict per row with ``row`` (``A``/``B``), ``row_type`` (in-domain vs
            generalization, D-7-GRIDROW), ``clip_name``, ``prompt``, ``seed`` (default 42),
            ``steps``, the four output mp4 paths ``original_mp4`` / ``seg_map_mp4`` (or a still
            ``seg_map_img``) / ``ic_lora_mp4`` / ``base_no_ref_mp4`` (RELATIVE to ``out_path``'s
            parent; a falsy tile → the row-appropriate ``generation failed`` / ``seg map
            unavailable`` / ``no reference`` fallback), and an optional per-row ``criteria_line``
            override.
        out_path: destination ``index.html`` path (parent dirs are created).
        params: sampling params for the banner (``steps``, ``guidance``, ``stg_scale``, ``width``,
            ``height``, ``frames``, ``lora_scale``, ``conditioning_mode``,
            ``reference_downscale_factor``).

    Returns:
        The resolved ``Path`` written.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    criteria = (
        '<div class="criteria">'
        f"<b>PASS =</b> {_RESKIN_CRITERIA_PASS} <b>FAIL =</b> {_RESKIN_CRITERIA_FAIL}"
        "</div>"
    )
    blocks = "".join(_reskin_block(row) for row in rows)
    doc = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>IC-LoRA re-skin: does the seg map steer the output?</title>"
        f"<style>{_STYLE}</style></head><body>"
        '<h1 class="grid-title">IC-LoRA re-skin — does the seg map steer the output?</h1>'
        f"{criteria}"
        f"{_params_banner(params)}"
        f"{blocks}"
        "</body></html>"
    )

    with out.open("w", encoding="utf-8") as f:  # utf-8 explicit (Phase-1 carry-forward)
        f.write(doc)
    return out
