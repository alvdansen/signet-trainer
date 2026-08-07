"""CPU contract for ``conditioning/h3_packing.py`` — the two TRANSCRIBED MiniMax-H3 wire formats.

Both contracts this file pins are **checkpoint contracts that fail SILENTLY at the correct shape**:

  * the **patchify intra-patch element order** — a wrong order produces correctly-shaped, entirely
    plausible rows trained against scrambled features. Nothing raises, the loss curve looks normal,
    and the adapter is garbage.
  * the **RoPE coordinate construction** — a monotone stand-in trains the adapter against the wrong
    geometry with nothing raising.

So neither may be asserted against a re-statement of the implementation. Every structural assertion
below is written against an INDEPENDENT formulation:

  * the patchify order is proved by an explicit nested-loop gather (a different algorithm), not by a
    second ``reshape``/``permute`` pair;
  * the spatial grid is proved against ``numpy.linspace(..., endpoint=False)``, which is the exact
    primitive diffusers uses and which ``torch.linspace`` does NOT reproduce;
  * the temporal grid is proved against an explicit cumulative sum of the ``5/3 * (1, 4, 4, 4, 4)``
    spans.

Authority: ``diffusers`` at the pinned ``modal/app.py::DIFFUSERS_SHA``
(``9f169d98d0bce392a889c3b6524d0d97734dfc0e``),
``src/diffusers/modular_pipelines/minimax_h3/before_denoise.py`` — ``patchify_video_latents``
(L44-73), ``_spatial_position_grid`` (L76-86), ``_temporal_position_grid`` (L89-98),
``_frame_position_grid`` (L101-109), ``_fill_audio_positions`` (L112-133) and
``MiniMaxH3Ref2VAPackedSequenceStep.build_ref2va_packed_sequence`` (L559-722).

Zero spend: CPU tensors and source scans only.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from signet_trainer.conditioning.h3_packing import (
    H3_ROPE_FRAMES_PER_LATENT,
    H3_ROPE_FRAME_RESCALE,
    H3_ROPE_IMAGE_REFERENCE_SPAN,
    H3_ROPE_SPATIAL_SCALE,
    build_h3_ref2va_position_ids,
    h3_frame_position_grid,
    h3_latent_grid_of_reference,
    h3_spatial_position_grid,
    h3_temporal_position_grid,
    h3_video_rows,
    make_h3_position_ids_fn,
    patchify_h3_video_latents,
)

_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "signet_trainer"
    / "conditioning"
    / "h3_packing.py"
)

PATCH = (1, 2, 2)


# --------------------------------------------------------------------------------------------------
# 1. The transcribed constants
# --------------------------------------------------------------------------------------------------


def test_the_rope_constants_are_the_transcribed_values() -> None:
    assert H3_ROPE_FRAME_RESCALE == pytest.approx(5.0 / 3.0)
    assert H3_ROPE_FRAMES_PER_LATENT == (1, 4, 4, 4, 4)
    assert H3_ROPE_SPATIAL_SCALE == 32
    # An IMAGE reference takes ONE integer rotary slot, NOT a latent frame's 5/3 units
    # (before_denoise.py L646-647 — the comment is in the reference implementation).
    assert H3_ROPE_IMAGE_REFERENCE_SPAN == 1.0


def test_the_source_cites_the_pinned_diffusers_sha() -> None:
    """The transcription is only auditable if it names WHERE it was transcribed from."""
    from signet_trainer.modal.app import DIFFUSERS_SHA

    src = _SRC.read_text(encoding="utf-8")
    assert DIFFUSERS_SHA in src, (
        "h3_packing.py must cite the pinned DIFFUSERS_SHA it was transcribed at — a transcription "
        "with no provenance cannot be re-verified when diffusers moves."
    )
    assert "before_denoise.py" in src


# --------------------------------------------------------------------------------------------------
# 2. patchify — the INTRA-PATCH ELEMENT ORDER (the dangerous one)
# --------------------------------------------------------------------------------------------------


def _reference_patchify_by_gather(
    latents: torch.Tensor, patch: tuple[int, int, int]
) -> torch.Tensor:
    """An INDEPENDENT patchify: explicit nested loops, no reshape/permute anywhere.

    Row order is frame-block major, then height-block, then width-block. Inside a row the elements
    run ``(channel, patch_t, patch_h, patch_w)`` — channel SLOWEST. Written as a gather precisely so
    it cannot agree with the implementation by sharing its algorithm.
    """
    pt, ph, pw = patch
    b, c, f, h, w = latents.shape
    rows = []
    for bi in range(b):
        for fb in range(f // pt):
            for hb in range(h // ph):
                for wb in range(w // pw):
                    row = []
                    for ci in range(c):
                        for it in range(pt):
                            for ih in range(ph):
                                for iw in range(pw):
                                    row.append(
                                        latents[
                                            bi,
                                            ci,
                                            fb * pt + it,
                                            hb * ph + ih,
                                            wb * pw + iw,
                                        ]
                                    )
                    rows.append(torch.stack(row))
    return torch.stack(rows)


def test_patchify_matches_an_independent_nested_loop_gather() -> None:
    """THE assertion this module exists for. A different algorithm, element for element."""
    torch.manual_seed(0)
    latents = torch.randn(1, 5, 2, 6, 8)
    got = patchify_h3_video_latents(latents, PATCH)
    want = _reference_patchify_by_gather(latents, PATCH)
    assert got.shape == want.shape
    assert torch.equal(got, want)


def test_patchify_matches_the_gather_at_a_non_unit_temporal_patch() -> None:
    """Guard the ``patch_t`` axis too — H3 ships ``(1, 2, 2)`` but the transcription is general."""
    torch.manual_seed(1)
    latents = torch.randn(1, 3, 4, 4, 6)
    patch = (2, 2, 2)
    assert torch.equal(
        patchify_h3_video_latents(latents, patch),
        _reference_patchify_by_gather(latents, patch),
    )


def test_patchify_row_count_and_feature_width() -> None:
    latents = torch.zeros(1, 24, 2, 8, 10)
    rows = patchify_h3_video_latents(latents, PATCH)
    assert rows.shape == (2 * (8 // 2) * (10 // 2), 24 * 1 * 2 * 2)


def test_patchify_row_order_is_frame_major_then_row_major() -> None:
    """Row ``r`` must be the ``(frame_block, h_block, w_block)`` triple in that nesting order."""
    c, f, h, w = 1, 3, 4, 6
    # Encode the position of every element as a unique integer.
    flat = torch.arange(c * f * h * w, dtype=torch.float32).reshape(1, c, f, h, w)
    rows = patchify_h3_video_latents(flat, PATCH)
    hb, wb = h // 2, w // 2
    for fi in range(f):
        for hi in range(hb):
            for wi in range(wb):
                row_index = (fi * hb + hi) * wb + wi
                # First element of the row is (channel 0, patch_t 0, patch_h 0, patch_w 0).
                expected = flat[0, 0, fi, hi * 2, wi * 2]
                assert rows[row_index, 0] == expected


def test_patchify_refuses_an_indivisible_grid_naming_the_patch() -> None:
    latents = torch.zeros(1, 24, 1, 7, 8)  # 7 is not divisible by patch_h = 2
    with pytest.raises(ValueError, match=r"divisible"):
        patchify_h3_video_latents(latents, PATCH)


def test_h3_video_rows_accepts_the_precomputed_four_dimensional_payload() -> None:
    """``PrecomputedDataset`` hands ``[C, F, H, W]``; the collated form is ``[1, C, F, H, W]``."""
    torch.manual_seed(2)
    four_d = torch.randn(24, 1, 8, 10)
    five_d = four_d.unsqueeze(0)
    a = h3_video_rows(four_d, PATCH)
    b = h3_video_rows(five_d, PATCH)
    assert a.shape == (1, (8 // 2) * (10 // 2), 96)
    assert torch.equal(a, b)


def test_h3_video_rows_passes_an_already_row_major_tensor_through() -> None:
    rows = torch.randn(1, 12, 96)
    assert torch.equal(h3_video_rows(rows, PATCH), rows)


def test_h3_video_rows_refuses_a_batched_stack() -> None:
    with pytest.raises(ValueError, match="one sample"):
        h3_video_rows(torch.zeros(2, 24, 1, 8, 8), PATCH)


# --------------------------------------------------------------------------------------------------
# 3. The spatial rotary axis — numpy's linspace, NOT torch's
# --------------------------------------------------------------------------------------------------


def test_spatial_grid_matches_numpy_linspace_endpoint_false() -> None:
    """``np.linspace(..., endpoint=False)`` is ``start + arange(n) * (stop - start) / n``.

    ``torch.linspace`` computes something else, and the float64 grid has to be reproduced exactly —
    the reference implementation says so in a comment at ``before_denoise.py`` L83-85.
    """
    dim, patch, sqrt_area = 84, 2, float(np.sqrt(84 * 56))
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    want = torch.from_numpy(
        np.linspace(left, left + ratio, dim // patch, endpoint=False) * H3_ROPE_SPATIAL_SCALE
    ).to(torch.float64)
    got = h3_spatial_position_grid(dim, patch, sqrt_area)
    assert got.dtype is torch.float64
    assert torch.equal(got, want)


def test_a_square_canvas_spans_zero_to_thirty_two_exclusive() -> None:
    """The docstring contract: a square canvas spans ``[0, 32)`` — right endpoint EXCLUDED."""
    grid = h3_spatial_position_grid(64, 2, float(np.sqrt(64 * 64)))
    assert grid[0].item() == pytest.approx(0.0)
    assert grid[-1].item() < H3_ROPE_SPATIAL_SCALE
    assert grid[-1].item() == pytest.approx(H3_ROPE_SPATIAL_SCALE * (1 - 1 / (64 // 2)))


def test_the_frame_grid_is_height_major_meshgrid_of_the_two_axes() -> None:
    lh, lw, ph, pw = 6, 8, 2, 2
    grid, width_grid = h3_frame_position_grid(lh, lw, ph, pw)
    assert grid.shape == ((lh // ph) * (lw // pw), 2)
    assert grid.dtype is torch.float64
    sqrt_area = float(np.sqrt(lh * lw))
    h_axis = h3_spatial_position_grid(lh, ph, sqrt_area)
    w_axis = h3_spatial_position_grid(lw, pw, sqrt_area)
    assert torch.equal(width_grid, w_axis)
    # indexing="ij" then reshape(-1) == height slowest, width fastest.
    for hi in range(lh // ph):
        for wi in range(lw // pw):
            row = hi * (lw // pw) + wi
            assert grid[row, 0] == h_axis[hi]
            assert grid[row, 1] == w_axis[wi]


# --------------------------------------------------------------------------------------------------
# 4. The temporal rotary axis — non-uniform 5/3 * (1, 4, 4, 4, 4) spacing
# --------------------------------------------------------------------------------------------------


def test_temporal_grid_spacing_is_the_non_uniform_five_thirds_pattern() -> None:
    origin = 137.0
    n = 7
    got = h3_temporal_position_grid(n, origin)
    spans = [H3_ROPE_FRAME_RESCALE * H3_ROPE_FRAMES_PER_LATENT[i % 5] for i in range(n)]
    want = [origin]
    for s in spans[:-1]:
        want.append(want[-1] + s)
    assert got.dtype is torch.float64
    assert got.tolist() == pytest.approx(want)


def test_temporal_grid_starts_exactly_at_the_origin() -> None:
    assert h3_temporal_position_grid(4, 96.0)[0].item() == 96.0


# --------------------------------------------------------------------------------------------------
# 5. The ref2va packed layout — [text | reference blocks | target audio | target video]
# --------------------------------------------------------------------------------------------------

# Two IMAGE references at different latent grids (the measured Phase-10 case: the two slots encode
# at DIFFERENT sizes, 1344x896 and 896x1600 in pixels).
REF_GRIDS = ((1, 84, 56), (1, 56, 100))
TARGET_GRID = (2, 48, 84)
N_TEXT = 40
N_AUDIO_LATENTS = 3


def _rows(grid: tuple[int, int, int], patch: tuple[int, int, int] = PATCH) -> int:
    f, h, w = grid
    return f * (h // patch[1]) * (w // patch[2])


def test_position_ids_shape_and_dtype() -> None:
    ids = build_h3_ref2va_position_ids(
        N_TEXT, REF_GRIDS, TARGET_GRID, N_AUDIO_LATENTS, PATCH, audio_channels=2
    )
    seq = (
        N_TEXT
        + sum(_rows(g) for g in REF_GRIDS)
        + N_AUDIO_LATENTS * 2
        + _rows(TARGET_GRID)
    )
    assert ids.shape == (seq, 3)
    assert ids.dtype is torch.float64


def test_text_rows_are_arange_on_the_time_axis_and_zero_on_the_spatial_axes() -> None:
    ids = build_h3_ref2va_position_ids(
        N_TEXT, REF_GRIDS, TARGET_GRID, N_AUDIO_LATENTS, PATCH, audio_channels=2
    )
    assert torch.equal(ids[:N_TEXT, 0], torch.arange(N_TEXT, dtype=torch.float64))
    assert torch.all(ids[:N_TEXT, 1:] == 0.0)


def test_each_image_reference_block_is_constant_in_time_and_advances_the_clock_by_one() -> None:
    """``t`` is absolute from ``text_len``; an image reference consumes exactly 1.0 ``t`` unit."""
    ids = build_h3_ref2va_position_ids(
        N_TEXT, REF_GRIDS, TARGET_GRID, N_AUDIO_LATENTS, PATCH, audio_channels=2
    )
    cursor = N_TEXT
    for i, grid in enumerate(REF_GRIDS):
        n = _rows(grid)
        block = ids[cursor : cursor + n]
        assert torch.all(block[:, 0] == float(N_TEXT) + i * H3_ROPE_IMAGE_REFERENCE_SPAN)
        # The spatial coordinates are that reference's OWN frame grid (its own resolution).
        frame_grid, _ = h3_frame_position_grid(grid[1], grid[2], PATCH[1], PATCH[2])
        assert torch.equal(block[:, 1:], frame_grid)
        cursor += n


def test_the_target_video_clock_starts_after_every_reference_block() -> None:
    ids = build_h3_ref2va_position_ids(
        N_TEXT, REF_GRIDS, TARGET_GRID, N_AUDIO_LATENTS, PATCH, audio_channels=2
    )
    n_target = _rows(TARGET_GRID)
    tail = ids[-n_target:]
    rotary_time = float(N_TEXT) + len(REF_GRIDS) * H3_ROPE_IMAGE_REFERENCE_SPAN
    frame_time = h3_temporal_position_grid(TARGET_GRID[0], rotary_time)
    rows_per_frame = n_target // TARGET_GRID[0]
    assert torch.equal(tail[:, 0], frame_time.repeat_interleave(rows_per_frame))
    frame_grid, _ = h3_frame_position_grid(TARGET_GRID[1], TARGET_GRID[2], PATCH[1], PATCH[2])
    assert torch.equal(tail[:, 1:], frame_grid.repeat(TARGET_GRID[0], 1))


def test_target_audio_rows_are_channel_major_and_pinned_to_the_width_extremes() -> None:
    ids = build_h3_ref2va_position_ids(
        N_TEXT, REF_GRIDS, TARGET_GRID, N_AUDIO_LATENTS, PATCH, audio_channels=2
    )
    audio_start = N_TEXT + sum(_rows(g) for g in REF_GRIDS)
    audio = ids[audio_start : audio_start + N_AUDIO_LATENTS * 2]
    rotary_time = float(N_TEXT) + len(REF_GRIDS) * H3_ROPE_IMAGE_REFERENCE_SPAN
    time = rotary_time + torch.arange(N_AUDIO_LATENTS, dtype=torch.float64)
    assert torch.equal(audio[:, 0], time.repeat(2))
    # Audio rows carry NO height coordinate and sit at the two extremes of the TARGET width grid.
    _, width_grid = h3_frame_position_grid(TARGET_GRID[1], TARGET_GRID[2], PATCH[1], PATCH[2])
    assert torch.all(audio[:, 1] == 0.0)
    assert torch.all(audio[:N_AUDIO_LATENTS, 2] == width_grid[0])
    assert torch.all(audio[N_AUDIO_LATENTS:, 2] == width_grid[-1])


def test_a_reference_free_sample_is_a_valid_layout() -> None:
    """D-10-REFDROP fires ~20% of the time — the dropped-reference sample must still build."""
    ids = build_h3_ref2va_position_ids(N_TEXT, (), TARGET_GRID, 0, PATCH, audio_channels=2)
    assert ids.shape == (N_TEXT + _rows(TARGET_GRID), 3)
    assert ids[N_TEXT, 0] == float(N_TEXT)


def test_reordering_the_references_is_a_different_request() -> None:
    """D-10-REFORDER is load-bearing: order advances the shared rotary clock."""
    a = build_h3_ref2va_position_ids(N_TEXT, REF_GRIDS, TARGET_GRID, 0, PATCH, audio_channels=2)
    b = build_h3_ref2va_position_ids(
        N_TEXT, tuple(reversed(REF_GRIDS)), TARGET_GRID, 0, PATCH, audio_channels=2
    )
    assert a.shape == b.shape
    assert not torch.equal(a, b)


def test_an_audio_reference_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="audio"):
        build_h3_ref2va_position_ids(
            N_TEXT,
            REF_GRIDS,
            TARGET_GRID,
            N_AUDIO_LATENTS,
            PATCH,
            audio_channels=2,
            reference_kinds=("image", "audio"),
        )


def test_a_video_reference_uses_the_temporal_grid_and_advances_by_its_span() -> None:
    grids = ((3, 24, 32),)
    ids = build_h3_ref2va_position_ids(
        N_TEXT, grids, TARGET_GRID, 0, PATCH, audio_channels=2, reference_kinds=("video",)
    )
    n = _rows(grids[0])
    block = ids[N_TEXT : N_TEXT + n]
    frame_time = h3_temporal_position_grid(3, float(N_TEXT))
    rows_per_frame = n // 3
    assert torch.equal(block[:, 0], frame_time.repeat_interleave(rows_per_frame))
    span = sum(H3_ROPE_FRAME_RESCALE * H3_ROPE_FRAMES_PER_LATENT[i % 5] for i in range(3))
    assert ids[-_rows(TARGET_GRID), 0] == pytest.approx(float(N_TEXT) + span)


# --------------------------------------------------------------------------------------------------
# 6. The strategy-facing factory — every derived count is CROSS-CHECKED against a measured one
# --------------------------------------------------------------------------------------------------


class _Ref:
    """The two selection fields ``make_h3_position_ids_fn`` reads off an ``H3Reference``."""

    def __init__(self, width: int, height: int, kind: str = "character") -> None:
        self.width = width
        self.height = height
        self.kind = kind


def _fn(**overrides: object):
    kwargs: dict = {
        "target_frames": 22,
        "target_aspect": (16, 9),
        "reference_short_edge": 896,
        "patch_size": PATCH,
    }
    kwargs.update(overrides)
    return make_h3_position_ids_fn(**kwargs)


def test_the_factory_builds_coordinates_for_the_full_sequence() -> None:
    refs = (_Ref(1024, 1536), _Ref(1440, 800, kind="environment"))
    grids = [h3_latent_grid_of_reference(r.width, r.height, 896, PATCH) for r in refs]
    n_cond_video = sum(_rows(g) for g in grids)
    fn = _fn()
    # The target row counts come off the factory's own reported layout (derived from h3_geometry).
    n_target_video = fn.n_target_video_rows
    n_target_audio = fn.n_target_audio_rows
    seq = N_TEXT + n_cond_video + n_target_audio + n_target_video
    ids = fn(
        n_text=N_TEXT,
        references=refs,
        n_cond_video=n_cond_video,
        n_cond_audio=0,
        n_target_audio=n_target_audio,
        n_target_video=n_target_video,
        seq_len=seq,
    )
    assert ids.shape == (seq, 3)


def test_the_factory_refuses_a_reference_row_count_it_cannot_reproduce() -> None:
    """The self-check that makes the derivation safe: derived rows MUST equal measured rows."""
    fn = _fn()
    with pytest.raises(ValueError, match="reference"):
        fn(
            n_text=N_TEXT,
            references=(_Ref(1024, 1536),),
            n_cond_video=7,  # nothing like the real 1176
            n_cond_audio=0,
            n_target_audio=fn.n_target_audio_rows,
            n_target_video=fn.n_target_video_rows,
            seq_len=N_TEXT + 7 + fn.n_target_audio_rows + fn.n_target_video_rows,
        )


def test_the_factory_refuses_a_target_row_count_it_cannot_reproduce() -> None:
    fn = _fn()
    refs = (_Ref(1024, 1536), _Ref(1440, 800, kind="environment"))
    n_cond_video = sum(
        _rows(h3_latent_grid_of_reference(r.width, r.height, 896, PATCH)) for r in refs
    )
    with pytest.raises(ValueError, match="target"):
        fn(
            n_text=N_TEXT,
            references=refs,
            n_cond_video=n_cond_video,
            n_cond_audio=0,
            n_target_audio=fn.n_target_audio_rows,
            n_target_video=fn.n_target_video_rows + 1,
            seq_len=N_TEXT + n_cond_video + fn.n_target_audio_rows + fn.n_target_video_rows + 1,
        )


def test_the_factory_refuses_reference_audio_rows_it_does_not_build() -> None:
    fn = _fn()
    with pytest.raises(ValueError, match="n_cond_audio"):
        fn(
            n_text=N_TEXT,
            references=(),
            n_cond_video=0,
            n_cond_audio=4,
            n_target_audio=fn.n_target_audio_rows,
            n_target_video=fn.n_target_video_rows,
            seq_len=N_TEXT + 4 + fn.n_target_audio_rows + fn.n_target_video_rows,
        )


def test_the_factory_handles_the_dropped_reference_case() -> None:
    fn = _fn()
    seq = N_TEXT + fn.n_target_audio_rows + fn.n_target_video_rows
    ids = fn(
        n_text=N_TEXT,
        references=(),
        n_cond_video=0,
        n_cond_audio=0,
        n_target_audio=fn.n_target_audio_rows,
        n_target_video=fn.n_target_video_rows,
        seq_len=seq,
    )
    assert ids.shape == (seq, 3)


def test_the_factory_target_geometry_comes_from_h3_geometry_not_a_literal() -> None:
    """The canvas + frame law are owned by ``conditioning/h3_geometry``; nothing re-derives them."""
    from signet_trainer.conditioning.h3_geometry import (
        h3_audio_rows,
        h3_latent_frames,
        resolve_canvas_size,
        rows_of,
    )

    fn = _fn(target_frames=22, target_aspect=(16, 9))
    h, w = resolve_canvas_size(16, 9)
    assert fn.n_target_video_rows == h3_latent_frames(22) * rows_of(h, w)
    assert fn.n_target_audio_rows == h3_audio_rows(22)


def test_reference_latent_grid_agrees_with_the_geometry_row_count() -> None:
    """``h3_latent_grid_of_reference`` must reproduce ``rows_of(reference_image_size(...))``."""
    from signet_trainer.conditioning.h3_geometry import reference_image_size, rows_of

    for width, height in ((1024, 1536), (832, 1248), (2048, 2048), (1440, 800), (1344, 768)):
        grid = h3_latent_grid_of_reference(width, height, 896, PATCH)
        assert grid[0] == 1  # an image reference is exactly ONE latent frame
        assert _rows(grid) == rows_of(*reference_image_size(width, height, short_edge=896))


# --------------------------------------------------------------------------------------------------
# 7. Import confinement — Anti-Pattern 6, the CONFINED tier
# --------------------------------------------------------------------------------------------------


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"#.*", "", src)


def test_no_module_scope_modal_or_diffusers_import() -> None:
    code = _strip_comments_and_docstrings(_SRC.read_text(encoding="utf-8"))
    hits = re.findall(r"^(?:import|from)\s+(modal|diffusers)\b", code, re.MULTILINE)
    assert not hits, f"h3_packing.py must stay CPU-importable; found {hits}"


def test_importing_h3_packing_pulls_no_backend() -> None:
    """Subprocess proof: importing the module leaves both heavy roots out of ``sys.modules``."""
    import os

    script = (
        "import sys\n"
        "import signet_trainer.conditioning.h3_packing as m\n"
        "assert 'diffusers' not in sys.modules, 'diffusers leaked'\n"
        "assert 'modal' not in sys.modules, 'modal leaked'\n"
        "print(m.H3_ROPE_SPATIAL_SCALE, m.H3_ROPE_IMAGE_REFERENCE_SPAN)\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    merged = dict(os.environ)
    merged.update({"PYTHONPATH": str(repo_root / "src"), "PYTHONIOENCODING": "utf-8"})
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=merged,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "32 1.0", proc.stdout
