"""Regression guard for the BROWSER-PLAYABLE overlay contract (scripts/qa_overlay_h264.py).

The landmine: ``cv2.VideoWriter`` emits mp4v (MPEG-4 Part 2), which no browser decodes — a grid of
mp4v overlays renders as black rectangles and reads as a pipeline failure. Every QA video handed to
a human must therefore be H.264 / yuv420p / +faststart. This test asserts the ffmpeg invocation
carries all three flags, so a future "simplification" of the command cannot silently reintroduce an
unplayable encode. ``subprocess.run`` is monkeypatched — NO ffmpeg, NO media, NO GPU needed.

Also covers scripts/prep_segdataset.py's parser contract (the dataset CLI is otherwise exercised by
its real run). Both scripts are standalone (not packages) — loaded via importlib, per the
tests/test_stage_demo_seg_dataset.py precedent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qa = _load("qa_overlay_h264")


class _R:
    returncode = 0
    stderr = ""
    stdout = ""


def test_transcode_command_is_browser_playable(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return _R()

    monkeypatch.setattr(qa.subprocess, "run", fake_run)
    ok = qa.transcode(tmp_path / "in.mp4", tmp_path / "out" / "step_0.mp4", 640, 20)
    assert ok
    cmd = seen["cmd"]
    assert cmd[0] == "ffmpeg"
    # the three flags that make the file decodable in a browser
    assert "libx264" in cmd, "H.264 is mandatory — mp4v/mpeg4 output is unplayable in browsers"
    i = cmd.index("-pix_fmt")
    assert cmd[i + 1] == "yuv420p", "yuv420p is mandatory for browser decode"
    j = cmd.index("-movflags")
    assert cmd[j + 1] == "+faststart", "moov atom must be relocated for progressive playback"
    # -2 height keeps dims even (odd dims break yuv420p encoding)
    assert any(str(a).endswith(":-2") for a in cmd), "scale filter must force even dimensions"


def test_transcode_reports_failure(monkeypatch, tmp_path):
    class _Bad:
        returncode = 1
        stderr = "boom"
        stdout = ""

    monkeypatch.setattr(qa.subprocess, "run", lambda cmd, **kw: _Bad())
    assert qa.transcode(tmp_path / "in.mp4", tmp_path / "out.mp4", 640, 20) is False


def test_qa_parser_defaults():
    args = qa.build_parser().parse_args(["--in-dir", "a", "--out-dir", "b"])
    assert args.max_width == 640 and args.step == 0 and args.suffix == "_overlay"


def test_segdataset_parser_contract():
    seg = _load("prep_segdataset")
    args = seg.build_parser().parse_args(
        ["--masks-root", "m", "--clips-dir", "c", "--manifest", "man.txt", "--out", "o"]
    )
    assert args.stride == 1 and args.skip_empty is False and args.records is None
