"""Guards ``_h3_manifest_rows``' per-row caption check (issue #3).

An empty/whitespace ``caption`` used to reach the Qwen3-VL-32B text encoder and die in an opaque
zero-length reshape (``cannot reshape tensor of 0 elements into shape [1, 0, -1, 128]``) — AFTER
the expensive PHASE A load. ``_h3_manifest_rows`` already walks every row before any model loads
and already refuses a reference-less row loudly there (``_h3_resolve_references``'s "Refusing to
encode a reference-less ref2v sample"); this pins the same-shape refusal for an empty caption,
firing during the manifest read itself — before PHASE A.

``_h3_manifest_rows`` is pure stdlib (``json`` + ``pathlib``, both function-local), so it is
exec'd out of the parsed source rather than imported — importing ``modal/fns.py`` builds the
Modal app graph and eagerly resolves every ``Secret.from_name`` (the ``test_h3_reference_
descriptors.py`` convention).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"


def _manifest_rows():
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    module = ast.Module(
        body=[
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "_h3_manifest_rows"
        ],
        type_ignores=[],
    )
    assert len(module.body) == 1, "_h3_manifest_rows is not a top-level def in fns.py"
    namespace: dict = {}
    exec(compile(ast.fix_missing_locations(module), "<fns-slice>", "exec"), namespace)  # noqa: S102
    return namespace["_h3_manifest_rows"]


def _write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _valid_row(**over) -> dict:
    row = {"media_path": "clips/segment_000.mp4", "caption": "a woman walks through a doorway"}
    row.update(over)
    return row


@pytest.mark.parametrize("bad_caption", ["", "   ", "\t\n"])
def test_empty_or_whitespace_captions_are_refused(tmp_path: Path, bad_caption: str) -> None:
    manifest = _write_manifest(
        tmp_path,
        [_valid_row(), _valid_row(media_path="clips/segment_001.mp4", caption=bad_caption)],
    )
    with pytest.raises(ValueError, match="empty caption") as exc:
        _manifest_rows()(str(manifest))
    msg = str(exc.value)
    # Names the offending row's INDEX and media_path — not a generic "some row" message.
    assert "row 1" in msg
    assert "clips/segment_001.mp4" in msg


def test_a_missing_caption_key_is_refused_the_same_way(tmp_path: Path) -> None:
    row = {"media_path": "clips/segment_000.mp4"}
    manifest = _write_manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="empty caption"):
        _manifest_rows()(str(manifest))


def test_valid_rows_are_unaffected(tmp_path: Path) -> None:
    rows = [_valid_row(), _valid_row(media_path="clips/segment_001.mp4", caption="a second row")]
    manifest = _write_manifest(tmp_path, rows)
    result = _manifest_rows()(str(manifest))
    assert len(result) == 2
    assert [row["caption"] for row in result] == [r["caption"] for r in rows]


def test_the_refusal_fires_pre_phase_a_in_h3_preprocess() -> None:
    """The caption check must live in the manifest-read pass — the sweep every stage runs before
    any model loads — not somewhere reachable only after Qwen3-VL-32B is already resident."""
    source = _FNS.read_text(encoding="utf-8")
    start = source.index("def _h3_manifest_rows(")
    end = source.index("\ndef _h3_output_rel(")
    body = source[start:end]
    assert "empty caption" in body
    assert "row {index}" in body or "row {index} (" in body
