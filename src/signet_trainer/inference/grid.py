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
    "write_qwen_edit_gallery",
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


def _image_tile(
    label: str,
    img: str,
    *,
    control: bool = False,
    fallback: str = "generation failed",
) -> str:
    """One IMAGE column with a caller-chosen fallback label and an optional ``.control`` marker.

    The image-family analog of ``_reskin_video_cell``, and a SEPARATE function from ``_image_cell``
    for the same reason ``_reskin_video_cell`` is separate from ``_video_cell``: the two differ in
    ways that are not cosmetic, and the shipped Phase 4-7 galleries must keep their exact markup.

    Two concrete differences from ``_image_cell``:

      * **The fallback is a parameter, not the literal 'no reference'.** ``_image_cell`` renders
        REFERENCE thumbnails, where a missing file honestly means "no reference was supplied". On an
        OUTPUT column a missing file means the render did not happen, and labelling that "no
        reference" would send an operator hunting for a dataset problem that does not exist — the
        same distinction that made the re-skin gallery's ``original not staged`` fallback worth
        having (see ``_reskin_block``).
      * **No ``loop`` attribute.** ``_image_cell`` emits ``<img loop …>``; ``loop`` is not a valid
        ``<img>`` attribute and is a carry-over from the video cell it was adapted from. It is left
        untouched there — three shipped galleries' HTML is pinned by test — and simply not
        reproduced here.

    ``control=True`` carries the ``.control`` divergence marker class and the accent badge, so a
    baseline column can never be read as a result. Every ``label`` / ``src`` / ``fallback`` flows
    through ``html.escape`` (T-04-03 output encoding), identically to the other cell builders.
    """
    safe_label = html.escape(label)
    cell_class = "cell control" if control else "cell"
    badge = '<span class="badge-control">CONTROL — no adapter</span>' if control else ""
    if img:
        safe_src = html.escape(img, quote=True)
        media = f'<img src="{safe_src}" alt="{safe_label}">'
    else:
        media = f'<div class="failed">{html.escape(fallback)}</div>'
    return (
        f'<div class="{cell_class}">'
        f'<div class="col-label">{safe_label}{badge}</div>{media}</div>'
    )


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


# The qwen_edit "what to look for" block. Both halves are §8 of QWEN-CHAINED-EDIT-METHOD.md, not
# house invention: the divergence half is the convergence check (*"base-vs-LoRA divergence at ~200–250
# steps, not a step threshold and not loss. If the sample is essentially the base render, it isn't
# converging — intervene"*), and the A/B half is what the two prompt modes exist to expose (*"this is
# how the trace-vs-reinterpret behavior gets read"*). The literal ``&`` is emitted as ``&amp;`` since
# this is authored HTML, not user input.
_QWEN_EDIT_CRITERIA_PASS = (
    "every adapter column visibly diverges from its BASE control at the same prompt, "
    "&amp; the A (style-only) and B (content-named) columns of one checkpoint read differently — "
    "that A/B delta is the trace-vs-reinterpret signal."
)
_QWEN_EDIT_CRITERIA_FAIL = (
    "an adapter column is essentially its BASE render (not converging — intervene, do not wait for "
    "a step threshold), or A and B are indistinguishable (the adapter is tracing the control image "
    "rather than reinterpreting it)."
)
#: Per-row PASS condition default. Overridable per row via ``criteria_line``.
_QWEN_EDIT_ROW_PASS_LINE = (
    "the adapter cells depart from BASE at the same prompt, and A differs from B within a checkpoint."
)


def _qwen_edit_block(row: dict, columns: "list") -> str:
    """One ``.comparison`` block for the qwen_edit A/B x checkpoint-BAND grid (§8).

    Renders, top-to-bottom: ``.row-label`` (the held-out input's name) → one ``.reskin-prompt`` line
    per §8 prompt mode, each printing the EXACT prompt that mode's columns were rendered under →
    ``.seed`` → ``.criteria-line`` → ``.row`` containing, left-to-right:

        [ control slot thumbnails ] [ BASE · A | BASE · B ] [ ckpt1 · A | ckpt1 · B ] ...

    The control thumbnails come from ``row["control_imgs"]`` (the held-out control image per slot —
    up to ``QWEN_EDIT_MAX_CONTROL_SLOTS``); the output cells come from ``columns``, which is the
    caller's ``plan_qwen_edit_columns`` result. The column list is PASSED, never derived from the
    row's keys: a band is a declaration, and deriving columns from whichever keys happen to be
    present would let a band member whose renders all failed vanish from the grid entirely instead of
    showing a row of honest 'generation failed' tiles.

    Both prompt lines are printed even when a mode's prompt is missing (rendered as ``—``): §8's read
    is a comparison between two prompts, and a row that silently shows one of them invites the delta
    to be attributed to the model rather than to the missing prompt.

    EVERY cell is an ``<img>``. This family renders images — a ``<video>`` element anywhere in this
    block is a bug, and ``tests/test_qwen_edit_render_surface.py`` asserts its absence.
    """
    input_label = html.escape(str(row.get("input_label", row.get("input_name", ""))))
    seed = html.escape(str(row.get("seed", "")))
    criteria_line = html.escape(str(row.get("criteria_line", _QWEN_EDIT_ROW_PASS_LINE)))

    prompt_lines = "".join(
        '<div class="reskin-prompt">'
        f"{html.escape(col_mode_label)}: {html.escape(str(row.get(prompt_key) or '—'))}"
        "</div>"
        for col_mode_label, prompt_key in _row_prompt_lines(columns)
    )

    control_cells = "".join(
        _image_tile(f"control {i}", img, fallback="control image missing")
        for i, img in enumerate(row.get("control_imgs", []), start=1)
    )
    output_cells = "".join(
        _image_tile(
            column.label,
            str(row.get(column.row_key, "") or ""),
            control=bool(column.control),
            fallback="generation failed",
        )
        for column in columns
    )

    return (
        '<section class="comparison">'
        f'<div class="row-label">{input_label}</div>'
        f"{prompt_lines}"
        f'<div class="seed">seed {seed}</div>'
        f'<div class="criteria-line"><b>PASS if:</b> {criteria_line}</div>'
        f'<div class="row">{control_cells}{output_cells}</div>'
        "</section>"
    )


def _row_prompt_lines(columns: "list") -> list[tuple[str, str]]:
    """The ordered ``(mode label, row prompt key)`` pairs to print above one qwen_edit row.

    Derived from the COLUMNS rather than from a hardcoded A/B pair, so the header lines can never
    describe a mode set the columns do not contain. De-duplicated in first-appearance order because
    each mode owns several columns (one per band member) but only one prompt.
    """
    seen: list[tuple[str, str]] = []
    keys: set[str] = set()
    for column in columns:
        mode = getattr(column, "mode", None)
        if mode is None:
            continue
        key = str(getattr(mode, "row_prompt_key", ""))
        if key and key not in keys:
            keys.add(key)
            seen.append((str(getattr(mode, "label", key)), key))
    return seen


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


#: Banner fields ``_qwen_edit_params_banner`` knows how to render. An allowlist rather than a
#: ``.get()`` free-for-all: an unrecognised key is REFUSED, so plumbing a §8 sampling setting that
#: this banner cannot render fails loudly at the call site instead of disappearing from the page.
#: ⛔ One §8 field is deliberately ABSENT — the static scheduler-reparameterisation value (3.0). It
#: is blocked by ``tests/test_no_wan_params.py``'s ``_WAN_TOKENS`` ban on the bare token ``shift``
#: across every ``*.py`` under ``inference/``, a guard written for LTX paths whose scope now also
#: covers a family where that setting is mandatory. Resolving that is a ruling; see the function's
#: docstring for the full statement of the conflict and why omission is the reversible side.
_QWEN_EDIT_BANNER_FIELDS = (
    "steps",
    "true_cfg",
    "cfg_norm",
    "width",
    "height",
    "lora_scale",
    "checkpoint_band",
)


def _qwen_edit_params_banner(params: dict) -> str:
    """The qwen_edit params banner — §8's inference settings + the checkpoint BAND.

    A SEPARATE function from ``_params_banner``, not an extension of it, because the two describe
    different instruments and the shipped four galleries' banners are pinned by test:

      * ``stg`` is an LTX-only concept (spatiotemporal guidance). Emitting ``stg=?`` on a Qwen grid
        would put a knob on the page that does not exist on this model, which is worse than omitting
        it — an operator diagnosing a bad render would reach for it.
      * The dims are ``WxH``, not ``WxHxF``. ``QWEN_EDIT_FRAMES`` is pinned to exactly 1
        (``conditioning/qwen_edit_geometry.py:141``); printing ``x1`` would present a constant as a
        setting.
      * ``true_cfg`` / ``cfg_norm`` are on the banner because §8 names its inference settings as a
        DOCUMENTED TRAP with an explicit diagnosis tell: *"training samples look fine but renders are
        muddy → it's inference settings, not the model."* The settings that produced a grid have to
        be readable off the grid, or that diagnosis cannot be made from the artifact.
      * ``checkpoint_band`` names every member of the rendered band (``CheckpointBand.describe()``),
        so the page says which three checkpoints it is showing rather than implying a winner.

    ⛔ ONE §8 SETTING IS MISSING FROM THIS BANNER, AND IT IS A CONTRACT CONFLICT, NOT AN OVERSIGHT.
    -----------------------------------------------------------------------------------------------
    §8's inference table for Qwen-Image-Edit-2511 reads: steps 30 · true_cfg 4.0 with CFGNorm on ·
    **static shift 3.0** (``use_dynamic_shifting=False``) · LoRA strength 1.0. That shift value is
    the single most trap-prone number in the table — §8 warns that
    ``QwenImageEditPlusModel.get_generation_pipeline()`` rebuilds a fresh default-shift scheduler, so
    it must be overridden onto ``pipeline.scheduler`` AFTER the pipeline is built, and that
    ai-toolkit's default dynamic shift (~1.6–2.5) is far weaker than the shipped ComfyUI
    ``ModelSamplingAuraFlow`` value.

    ``tests/test_no_wan_params.py:31`` bans the bare token ``shift`` from **executable code in every
    ``*.py`` under ``inference/``** (``_WAN_TOKENS``, scanned by ``test_no_wan_params_in_inference_code``
    over ``_INFERENCE_DIR.glob("*.py")``). That guard's stated purpose (:6-8) is to keep the
    Wan-tuned sampling set out of **LTX** code paths — its scope has simply outgrown its rationale
    now that a non-LTX family lives in this directory, where the same token names a MANDATORY
    setting rather than a forbidden one.

    Both cannot hold. This writer takes the reversible side: the field is OMITTED rather than the
    guard weakened, because narrowing a shipped money-safe guard is a ruling and quietly renaming the
    field to slip past a scanner would be worse than either — it would leave the guard falsely green
    while stripping §8's most trap-prone number out of the artifact an operator diagnoses from.
    ⚠ THE DAY HAS ARRIVED — FOR REVIEW, not silently resolved here. When this was written,
    ``render_qwen_edit_sample`` raised, so no real render existed whose scheduler setting needed
    reporting and omitting the field cost nothing. The sampler landed at e132b30 and real renders
    now exist, so the deferred ruling is live: this gallery is the artifact an operator diagnoses a
    muddy render from, and §8's most trap-prone number is the one it cannot show.

    It is not UNVERIFIED, which is why this is a reporting gap and not a safety one:
    ``assert_qwen_edit_scheduler_pinned`` re-checks the pin before every single render and its
    report goes into the per-cell log line — the value is proven, just not surfaced in the HTML.
    The two options remain what they were (narrow ``_WAN_TOKENS``' directory-wide scope, or move
    this writer out from under it), both are rulings on a shipped money-safe guard, and neither is
    the sort of thing to slip into an unrelated commit.

    Missing keys render ``?`` — the same honest-placeholder convention ``_params_banner`` uses. This
    writer never invents a sampling value. An UNRECOGNISED key is REFUSED rather than dropped, so a
    caller plumbing the blocked §8 setting through gets a loud, actionable error at the exact point
    of the conflict instead of watching its value vanish from the page.
    """
    unknown = sorted(k for k in params if k not in _QWEN_EDIT_BANNER_FIELDS)
    if unknown:
        raise ValueError(
            f"unrecognised qwen_edit banner field(s) {unknown}. Known fields: "
            f"{list(_QWEN_EDIT_BANNER_FIELDS)}. An unknown key is refused rather than silently "
            "dropped, because a dropped SAMPLING setting is invisible on the artifact an operator "
            "diagnoses from. If this is the static scheduler-reparameterisation value from METHOD "
            "§8's inference table (the 3.0 one, the setting §8 says must be overridden onto "
            "pipeline.scheduler AFTER get_generation_pipeline rebuilds it): it is BLOCKED, not "
            "forgotten — tests/test_no_wan_params.py bans that token from executable code anywhere "
            "under inference/, and un-banning it for the non-LTX family is a ruling, not an edit. "
            "See _qwen_edit_params_banner's docstring."
        )
    width = params.get("width", "?")
    height = params.get("height", "?")
    dims = html.escape(f"{width}x{height}")
    steps = html.escape(str(params.get("steps", "?")))
    true_cfg = html.escape(str(params.get("true_cfg", "?")))
    cfg_norm = html.escape(str(params.get("cfg_norm", "?")))
    lora_scale = html.escape(str(params.get("lora_scale", "?")))
    band = html.escape(str(params.get("checkpoint_band", "?")))
    return (
        '<div class="params">'
        f"steps={steps} &middot; true_cfg={true_cfg} &middot; cfg_norm={cfg_norm} "
        f"&middot; {dims} &middot; lora_scale={lora_scale}"
        f" &middot; {band}"
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
    # qwen_edit (family #3) image-grid additions — appended, and SCOPED under `.qwen-edit-grid` so
    # the four shipped video galleries render byte-identically. A bare `img{width:100%}` here would
    # resize the reference thumbnails in the Phase-5/6/7 galleries, which is a visual change to
    # already-delivered artifacts. The row also scrolls rather than squeezing: a band grid is
    # 2 x (band + 1) output cells wide, and flex-shrinking 8 images to fit a laptop makes the
    # base-vs-LoRA divergence judgment §8 asks for impossible to make.
    ".qwen-edit-grid img{width:100%;background:#000;border-radius:4px;display:block}"
    ".qwen-edit-grid .row{overflow-x:auto;flex-wrap:nowrap}"
    ".qwen-edit-grid .cell{flex:0 0 220px}"
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


def write_qwen_edit_gallery(
    rows: list[dict],
    out_path: str | Path,
    params: dict,
    *,
    columns: list,
) -> Path:
    """Write the qwen_edit A/B x checkpoint-BAND montage ``index.html`` and return its ``Path``.

    **The first IMAGE-output gallery in this module.** The four writers above render generated
    outputs as ``<video>``; family #3 is an image family (``QWEN_EDIT_FRAMES == 1``) and every
    OUTPUT cell here is an ``<img>``, via ``_image_tile``. Reusing ``write_comparison_gallery`` with
    ``.png`` paths would have emitted ``<video src="…png">`` — a page that renders as a row of black
    rectangles with playback controls, i.e. an artifact that looks broken rather than one that
    reports a broken render.

    Per row, left-to-right (§8 of ``QWEN-CHAINED-EDIT-METHOD.md``)::

        [ control 1..N ] [ BASE · A | BASE · B ] [ ckpt1 · A | ckpt1 · B ] [ ckpt2 · A | ... ] ...

    ``columns`` is REQUIRED and keyword-only — it is ``plan_qwen_edit_columns(band)``'s result. It is
    not derived from the rows and it is not defaulted, because the band is a DECLARATION: §8 ships a
    band of checkpoints rather than a winner, and a gallery that inferred its columns from whichever
    row keys were present would drop a band member whose renders all failed instead of showing it as
    a column of honest 'generation failed' tiles. A silently 2-wide band under a 3-checkpoint label
    is the grid-layer version of a coarse render key.

    Reuses ``_image_tile`` / ``_qwen_edit_params_banner`` / ``_STYLE`` — every input label, prompt,
    seed and image ``src`` flows through ``html.escape`` (T-04-03). Stdlib-only, torch-free,
    CPU/Windows-testable (mirrors ``write_reskin_gallery``). The generate call that populates it is
    ``inference/qwen_edit_layout.render_qwen_edit_sample``, which landed at e132b30 and delegates to
    ``models/qwen_edit_pipeline.qwen_edit_generate``.

    Args:
        rows: one dict per HELD-OUT INPUT with ``input_label`` (or ``input_name``), ``seed``,
            ``control_imgs`` (list of relative control-slot thumbnail paths; a falsy entry →
            'control image missing'), one prompt per §8 mode under that mode's ``row_prompt_key``
            (``prompt_a`` / ``prompt_b``; absent → printed as ``—``), an optional per-row
            ``criteria_line`` override, and one image path per column keyed by the column's
            ``row_key`` (relative to ``out_path``'s parent; a falsy path → 'generation failed').
        out_path: destination ``index.html`` path (parent dirs are created).
        params: banner values — EXACTLY the ``_QWEN_EDIT_BANNER_FIELDS`` allowlist: ``steps``,
            ``true_cfg``, ``cfg_norm``, ``width``, ``height``, ``lora_scale``, ``checkpoint_band``
            (``CheckpointBand.describe()``). ⚠ This list previously also advertised the static
            scheduler-reparameterisation field (the 3.0 one); that was a documentation bug, not a
            supported key — ``_qwen_edit_params_banner`` REFUSES it, so a caller following the old
            docstring got a ``ValueError`` rather than a banner. The field is blocked by
            ``tests/test_no_wan_params.py``'s directory-wide token ban and its restoration is a
            ruling; see ``_qwen_edit_params_banner``'s docstring for the full statement.
        columns: the ordered ``QwenEditColumn`` list from ``plan_qwen_edit_columns``.

    Returns:
        The resolved ``Path`` written.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not columns:
        raise ValueError(
            "write_qwen_edit_gallery got an EMPTY column list. The columns are "
            "plan_qwen_edit_columns(band)'s result and a band is 1..N checkpoints (§8 ships three); "
            "with no columns the page renders control thumbnails and nothing to judge them against "
            "— a complete-looking artifact that answers nothing. Pass the planner's output."
        )

    criteria = (
        '<div class="criteria">'
        f"<b>PASS =</b> {_QWEN_EDIT_CRITERIA_PASS} <b>FAIL =</b> {_QWEN_EDIT_CRITERIA_FAIL}"
        "</div>"
    )
    blocks = "".join(_qwen_edit_block(row, columns) for row in rows)
    doc = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>qwen_edit: A/B prompt modes across the checkpoint band</title>"
        f"<style>{_STYLE}</style></head><body>"
        '<div class="qwen-edit-grid">'
        '<h1 class="grid-title">qwen_edit — A/B prompt modes across the checkpoint band</h1>'
        f"{criteria}"
        f"{_qwen_edit_params_banner(params)}"
        f"{blocks}"
        "</div>"
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
