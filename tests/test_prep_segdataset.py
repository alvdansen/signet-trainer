"""scripts/prep_segdataset.py — PAIRED control/train .mov export mode (additive to image/mask pairs).

Covers the paired build's pure logic WITHOUT ffmpeg/cv2 (keep-list parsing, sorted renumbering,
caption resolution + fallback, the ProRes command flags, mask-frame counting), the frame-count
PARITY fail path (monkeypatched probe so no real decode is needed), and a full run_paired build with
the encode monkeypatched out — asserting control_data/train_data/NNN naming, --keep subsetting +
renumbering, MAPPING.local.txt contents, and caption resolution + fallback-warn. A single real
ProRes encode leg is guarded by skipif so the suite stays green where ffmpeg/prores is unavailable.

The script is standalone (not a package) — loaded via importlib per the test_qa_overlay_h264
standalone-script precedent. Fixtures are synthetic + generic (codename 'projx'); no property,
character, or franchise name appears. NO modal, NO metered dispatch, NO network.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seg = _load("prep_segdataset")


def _has_ffmpeg_prores() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
        return "prores_ks" in r.stdout
    except Exception:
        return False


# --------------------------------------------------------------------------------------------------
# Pure helpers — no ffmpeg, no cv2, no filesystem.
# --------------------------------------------------------------------------------------------------

def test_parse_keep_splits_and_none_means_all() -> None:
    assert seg.parse_keep("01, 02 ,foo") == {"01", "02", "foo"}
    assert seg.parse_keep("") is None
    assert seg.parse_keep(None) is None
    assert seg.parse_keep("  ,  ") is None


def test_clip_matches_keep_none_keeps_all() -> None:
    assert seg.clip_matches_keep("projx_07", None) is True


def test_clip_matches_keep_by_number_and_part_and_full_stem() -> None:
    # trailing number: '01' selects a clip whose trailing index is 1 (zero-pad insensitive)
    assert seg.clip_matches_keep("projx_01", {"01"}) is True
    assert seg.clip_matches_keep("projx_01", {"1"}) is True
    # leading NN_ index component also matches a numeric token
    assert seg.clip_matches_keep("07_projx", {"7"}) is True
    # a '_'-delimited word part matches
    assert seg.clip_matches_keep("projx_07", {"projx"}) is True
    # whole stem matches
    assert seg.clip_matches_keep("projx_07", {"projx_07"}) is True
    # non-member number is excluded
    assert seg.clip_matches_keep("projx_02", {"01", "03"}) is False


def test_renumber_keeps_all_in_sorted_order() -> None:
    parts = ["projx_01", "projx_02", "projx_03"]
    assert seg.renumber(parts, None) == [
        ("projx_01", "001"), ("projx_02", "002"), ("projx_03", "003"),
    ]


def test_renumber_subsets_and_renumbers_contiguously() -> None:
    parts = ["projx_01", "projx_02", "projx_03", "projx_13", "projx_14"]
    kept = seg.renumber(parts, seg.parse_keep("01,03,13"))
    assert kept == [("projx_01", "001"), ("projx_03", "002"), ("projx_13", "003")]


def test_build_control_cmd_carries_binary_preserving_prores_flags() -> None:
    cmd = seg.build_control_cmd(Path("masks/projx_01__full_body"), Path("out/001.mov"), 24)
    assert cmd[0] == "ffmpeg"
    assert "prores_ks" in cmd, "ProRes ks encoder is mandatory (verified 0 mid-tone leakage)"
    i = cmd.index("-profile:v")
    assert cmd[i + 1] == "3", "profile 3 is the proven mask-preserving profile"
    j = cmd.index("-pix_fmt")
    assert cmd[j + 1] == "yuv422p10le"
    k = cmd.index("-framerate")
    assert cmd[k + 1] == "24"
    assert any(str(a).endswith("%05d.png") for a in cmd), "input must be the NNNNN.png sequence"
    s = cmd.index("-start_number")
    assert cmd[s + 1] == "0", "PNG sequence starts at 00000.png"


def test_count_mask_frames_counts_five_digit_pngs_only(tmp_path: Path) -> None:
    md = tmp_path / "projx_01__full_body"
    md.mkdir()
    for i in range(5):
        (md / f"{i:05d}.png").write_bytes(b"x")
    (md / "overlay.png").write_bytes(b"x")  # non-frame -> ignored
    (md / "0001.png").write_bytes(b"x")     # 4-digit -> ignored
    assert seg.count_mask_frames(md) == 5


def test_resolve_caption_prefers_captions_dir_then_beside_source(tmp_path: Path) -> None:
    caps = tmp_path / "caps"
    caps.mkdir()
    (caps / "projx_01.txt").write_text("cap", encoding="utf-8")
    clip = tmp_path / "clips" / "projx_01.mov"
    clip.parent.mkdir()
    clip.write_bytes(b"x")
    # captions-dir hit
    assert seg.resolve_caption("projx_01", clip, caps) == caps / "projx_01.txt"
    # captions-dir miss for this clip -> beside-source fallback
    clip3 = tmp_path / "clips" / "projx_03.mov"
    clip3.write_bytes(b"x")
    (clip3.with_suffix(".txt")).write_text("beside", encoding="utf-8")
    assert seg.resolve_caption("projx_03", clip3, caps) == clip3.with_suffix(".txt")
    # nothing anywhere -> None
    bare = tmp_path / "clips" / "projx_bare.mov"
    bare.write_bytes(b"x")
    assert seg.resolve_caption("projx_bare", bare, caps) is None


# --------------------------------------------------------------------------------------------------
# run_paired integration — encode + parity probe monkeypatched (no ffmpeg, no cv2 decode).
# --------------------------------------------------------------------------------------------------

def _build_fixture(tmp_path: Path, stems: list[str], n_frames: int = 5) -> dict:
    """Synthetic masks_root + clips_dir + a manifest mapping stems -> the ProRes .mov sources
    (the realistic resolution path — resolve_clip's direct match is .mp4-only)."""
    masks_root = tmp_path / "masks_video"
    clips_dir = tmp_path / "source"
    masks_root.mkdir()
    clips_dir.mkdir()
    manifest_lines = []
    for n, stem in enumerate(stems, start=1):
        md = masks_root / f"{stem}__full_body"
        md.mkdir()
        for i in range(n_frames):
            (md / f"{i:05d}.png").write_bytes(b"x")
        clip = clips_dir / f"{stem}.mov"
        clip.write_bytes(b"clip-bytes")
        manifest_lines.append(f"{n:02d} | {stem}.png | {clip}")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return {"masks_root": masks_root, "clips_dir": clips_dir, "manifest": manifest}


def _paired_args(fx: dict, paired_out: Path, *, captions_dir: Path | None = None,
                 keep: str | None = None, fps: int | None = None):
    argv = [
        "--masks-root", str(fx["masks_root"]),
        "--clips-dir", str(fx["clips_dir"]),
        "--manifest", str(fx["manifest"]),
        "--paired-out", str(paired_out),
    ]
    if captions_dir is not None:
        argv += ["--captions-dir", str(captions_dir)]
    if keep is not None:
        argv += ["--keep", keep]
    if fps is not None:
        argv += ["--fps", str(fps)]
    return seg.build_parser().parse_args(argv)


def _stub_encode(monkeypatch: pytest.MonkeyPatch, *, frames: int = 5) -> None:
    """Bypass real decode/encode: every source clip reports `frames`, and encode just touches
    the target .mov so the layout/naming is what's under test."""
    monkeypatch.setattr(seg, "probe_frame_count", lambda clip: frames)

    def fake_encode(mask_dir, out_path, fps):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"prores")
        return True, ""

    monkeypatch.setattr(seg, "encode_control_mov", fake_encode)


def test_run_paired_naming_numbering_mapping_and_caption_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All 3 clips exported: control_data/train_data/NNN.mov numbered 001..003, MAPPING crosswalk,
    and the three caption outcomes (captions-dir hit / missing-warn / beside-source fallback)."""
    fx = _build_fixture(tmp_path, ["projx_01", "projx_02", "projx_03"], n_frames=5)
    caps = tmp_path / "caps"
    caps.mkdir()
    (caps / "projx_01.txt").write_text("Foreground: a subject walks. Emotion: calm.", encoding="utf-8")
    # projx_02 has NO caption anywhere -> should WARN + skip .txt.
    # projx_03 has only a beside-source caption -> fallback path.
    (fx["clips_dir"] / "projx_03.txt").write_text("Foreground: a subject turns.", encoding="utf-8")

    out = tmp_path / "_projx_prep" / "paired"
    _stub_encode(monkeypatch, frames=5)
    rc = seg.run_paired(_paired_args(fx, out, captions_dir=caps))
    assert rc == 0

    for num in ("001", "002", "003"):
        assert (out / "control_data" / f"{num}.mov").is_file()
        assert (out / "train_data" / f"{num}.mov").is_file()
    # train video is a byte-copy of the source clip (no re-encode)
    assert (out / "train_data" / "001.mov").read_bytes() == b"clip-bytes"
    # caption outcomes
    assert (out / "train_data" / "001.txt").is_file()       # captions-dir hit
    assert not (out / "train_data" / "002.txt").exists()     # missing -> warn + skip
    assert (out / "train_data" / "003.txt").is_file()        # beside-source fallback

    mapping = (out / "MAPPING.local.txt").read_text(encoding="utf-8").splitlines()
    assert mapping == ["001 <- projx_01", "002 <- projx_02", "003 <- projx_03"]


def test_run_paired_keep_subsets_and_renumbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--keep 01,03 keeps two of three clips, renumbered 001/002 in sorted order (no gap)."""
    fx = _build_fixture(tmp_path, ["projx_01", "projx_02", "projx_03"], n_frames=5)
    out = tmp_path / "paired"
    _stub_encode(monkeypatch, frames=5)
    rc = seg.run_paired(_paired_args(fx, out, keep="01,03"))
    assert rc == 0

    assert (out / "control_data" / "001.mov").is_file()
    assert (out / "control_data" / "002.mov").is_file()
    assert not (out / "control_data" / "003.mov").exists()
    mapping = (out / "MAPPING.local.txt").read_text(encoding="utf-8").splitlines()
    assert mapping == ["001 <- projx_01", "002 <- projx_03"]


def test_run_paired_frame_count_mismatch_fails_and_skips_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source frame count != mask frame count FAILS the clip (rc!=0) and never encodes a
    desynced control .mov."""
    fx = _build_fixture(tmp_path, ["projx_01"], n_frames=5)  # 5 mask PNGs
    out = tmp_path / "paired"
    # source reports 4 frames while the mask dir has 5 -> parity mismatch
    monkeypatch.setattr(seg, "probe_frame_count", lambda clip: 4)
    encoded = []
    monkeypatch.setattr(seg, "encode_control_mov",
                        lambda md, op, fps: (encoded.append(op), (True, ""))[1])

    rc = seg.run_paired(_paired_args(fx, out, fps=24))
    assert rc == 1, "a frame-count mismatch must return nonzero"
    assert encoded == [], "encode must NOT run when parity fails (no broken .mov emitted)"
    assert not (out / "control_data" / "001.mov").exists()


def test_run_paired_unresolved_source_clip_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mask dir whose source clip cannot be resolved is a broken pair -> nonzero exit."""
    fx = _build_fixture(tmp_path, ["projx_01"], n_frames=5)
    (fx["clips_dir"] / "projx_01.mov").unlink()  # remove the source so resolve_clip returns None
    out = tmp_path / "paired"
    _stub_encode(monkeypatch, frames=5)
    rc = seg.run_paired(_paired_args(fx, out))
    assert rc == 1
    assert not (out / "control_data" / "001.mov").exists()


def test_main_requires_an_output_mode() -> None:
    """Neither --out nor --paired-out -> argparse error (SystemExit)."""
    with pytest.raises(SystemExit):
        seg.main(["--masks-root", "m", "--clips-dir", "c", "--manifest", "x"])


# --------------------------------------------------------------------------------------------------
# Real ProRes encode — guarded so the suite stays green where ffmpeg/prores is unavailable.
# --------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not _has_ffmpeg_prores(), reason="ffmpeg with prores_ks not available")
def test_encode_control_mov_real_prores_roundtrip(tmp_path: Path) -> None:
    """A real ProRes encode of a 3-frame white-on-black mask sequence yields a decodable .mov whose
    frame count matches the input PNG count."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    md = tmp_path / "projx_01__full_body"
    md.mkdir()
    for i in range(3):
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[16:48, 16:48] = 255  # white silhouette on black
        cv2.imwrite(str(md / f"{i:05d}.png"), frame)

    out_mov = tmp_path / "control_data" / "001.mov"
    ok, err = seg.encode_control_mov(md, out_mov, 24)
    assert ok, f"prores encode failed: {err}"
    assert out_mov.is_file() and out_mov.stat().st_size > 0
    assert seg.probe_frame_count(out_mov) == 3
