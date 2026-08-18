"""Behavior tests for the loop-purity data path (TRAIN-03 / D-08, Plan 02-01 Task 1).

Covers the ported ``PrecomputedDataset`` (video-only) + the fresh-clip JSONL manifest
writer. The structural CI gate lives in ``tests/test_loop_purity.py`` (Task 3); these are
the behavior tests that drive the port.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _write_latent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "latents": torch.randn(128, 1, 2, 2),
            "num_frames": 1,
            "height": 2,
            "width": 2,
        },
        path,
    )


def _write_condition(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video_prompt_embeds": torch.randn(4, 4096),
            "prompt_attention_mask": torch.ones(4, dtype=torch.bool),
        },
        path,
    )


def test_imports_and_exposes_public_surface() -> None:
    """Test 1: the module imports and exposes PrecomputedDataset + PRECOMPUTED_DIR_NAME."""
    import signet_trainer.data.precomputed as pc

    assert pc.PRECOMPUTED_DIR_NAME == ".precomputed"
    assert hasattr(pc, "PrecomputedDataset")


def test_paired_sample_returns_both_condition_keys(tmp_path: Path) -> None:
    """Test 2: a paired latents/a.pt + conditions/a.pt yields latent_conditions + text_conditions."""
    from signet_trainer.data.precomputed import PrecomputedDataset

    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "a.pt")
    _write_condition(root / "conditions" / "a.pt")

    ds = PrecomputedDataset(str(root))
    assert len(ds) == 1
    sample = ds[0]
    assert "latent_conditions" in sample
    assert "text_conditions" in sample
    assert "latents" in sample["latent_conditions"]
    assert "video_prompt_embeds" in sample["text_conditions"]


_IC_LORA_3_SOURCE = {
    "latents": "latent_conditions",
    "conditions": "text_conditions",
    "reference_latents": "ref_latent_conditions",
}


def test_reference_latents_third_source_pairs_and_normalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """07-08 Pitfall 3: a 3rd reference_latents/ source pairs by rel path + gets latent normalization.

    Proves (a) the source-count-agnostic ``_discover_samples`` pairs reference_latents/a.pt by
    relative path alongside latents + conditions, and (b) the ``"latent" in dir_name.lower()`` check
    routes reference_latents through the SAME ``_normalize_video_latents`` path as latents (and NOT
    the text ``conditions`` source) — the [C, F, H, W] normalization is identical.
    """
    from signet_trainer.data.precomputed import PrecomputedDataset

    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "a.pt")
    _write_condition(root / "conditions" / "a.pt")
    _write_latent(root / "reference_latents" / "a.pt")  # ref latents share the VAE-encode shape

    # Spy on the shared normalizer to prove WHICH sources are routed through it.
    normalized_shapes: list[tuple[int, ...]] = []
    orig = PrecomputedDataset._normalize_video_latents  # already the plain fn (staticmethod access)

    def _spy(data: dict) -> dict:
        normalized_shapes.append(tuple(data["latents"].shape))
        return orig(data)

    monkeypatch.setattr(PrecomputedDataset, "_normalize_video_latents", staticmethod(_spy))

    ds = PrecomputedDataset(str(root), data_sources=_IC_LORA_3_SOURCE)
    assert len(ds) == 1
    sample = ds[0]

    # All three output keys are present and the reference source paired.
    assert "latent_conditions" in sample
    assert "text_conditions" in sample
    assert "ref_latent_conditions" in sample
    assert "latents" in sample["ref_latent_conditions"]

    # reference_latents gets the SAME [C, F, H, W] normalization as latents (shape preserved).
    assert sample["ref_latent_conditions"]["latents"].shape == (128, 1, 2, 2)
    assert sample["latent_conditions"]["latents"].shape == (128, 1, 2, 2)

    # The normalizer ran for exactly the two "latent"-named sources — latents + reference_latents —
    # and NOT for the text conditions source (which lacks a "latents" key).
    assert len(normalized_shapes) == 2, normalized_shapes
    assert all(shape == (128, 1, 2, 2) for shape in normalized_shapes)


def test_reference_latents_pairs_by_rel_path_not_legacy_condition_rename(tmp_path: Path) -> None:
    """07-08: reference_latents must pair by rel path even under the legacy ``latent_``/``condition_`` naming.

    With ``latent_x.pt``, ``_get_expected_file_path`` renames ONLY the ``conditions`` source to
    ``condition_x.pt`` (the legacy pairing). The reference_latents source must resolve by plain
    relative path (``reference_latents/latent_x.pt``) — no ``condition_`` misfire — or the third
    source would silently drop out and the sample would fail to pair.
    """
    from signet_trainer.data.precomputed import PrecomputedDataset

    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "latent_x.pt")  # legacy latent_ prefix
    _write_condition(root / "conditions" / "condition_x.pt")  # legacy rename target
    _write_latent(root / "reference_latents" / "latent_x.pt")  # rel-path match, NOT condition_x.pt

    ds = PrecomputedDataset(str(root), data_sources=_IC_LORA_3_SOURCE)
    assert len(ds) == 1
    sample = ds[0]
    assert "ref_latent_conditions" in sample
    assert sample["ref_latent_conditions"]["latents"].shape == (128, 1, 2, 2)


def test_mismatched_counts_raise_clear_error(tmp_path: Path) -> None:
    """Test 3: a latent with no paired condition produces no valid samples -> clear error."""
    from signet_trainer.data.precomputed import PrecomputedDataset

    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "a.pt")  # latent only, no matching condition
    (root / "conditions").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="(?i)no valid samples|matching"):
        PrecomputedDataset(str(root))


# --------------------------------------------------------------------------------------------------
# #21 finding 1 — silent partial-pairing drops must be LOUD (stdout) and OPT-IN fatal (strict=).
# --------------------------------------------------------------------------------------------------


def test_partial_pairing_prints_valid_total_skipped_counts_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#21 finding 1: valid/total/skipped counts print to STDOUT — logger.info has no handler."""
    from signet_trainer.data.precomputed import PrecomputedDataset

    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "a.pt")
    _write_condition(root / "conditions" / "a.pt")
    _write_latent(root / "latents" / "b.pt")  # b has NO matching condition -> partial pairing

    ds = PrecomputedDataset(str(root))  # default strict=False -> must NOT raise
    assert len(ds) == 1

    out = capsys.readouterr().out
    assert "1 valid sample" in out
    assert "2 total" in out
    assert "1 skipped" in out


def test_partial_pairing_default_strict_false_does_not_raise(tmp_path: Path) -> None:
    """#21 finding 1: strict defaults False — a legacy cache with strays trains unchanged."""
    from signet_trainer.data.precomputed import PrecomputedDataset

    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "a.pt")
    _write_condition(root / "conditions" / "a.pt")
    _write_latent(root / "latents" / "b.pt")  # the intentional stray

    ds = PrecomputedDataset(str(root))  # no strict= passed at all
    assert ds.strict is False
    assert len(ds) == 1


def test_partial_pairing_strict_true_refuses_with_named_counts(tmp_path: Path) -> None:
    """#21 finding 1: the OPT-IN strict knob turns skipped > 0 into a fatal, count-naming error."""
    from signet_trainer.data.precomputed import PrecomputedDataset

    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "a.pt")
    _write_condition(root / "conditions" / "a.pt")
    _write_latent(root / "latents" / "b.pt")
    _write_latent(root / "latents" / "c.pt")

    with pytest.raises(ValueError, match=r"strict=True") as excinfo:
        PrecomputedDataset(str(root), strict=True)
    message = str(excinfo.value)
    assert "2 of 3 sample(s)" in message
    assert "1 valid sample(s)" in message


def test_strict_true_still_raises_on_valid_count_zero(tmp_path: Path) -> None:
    """#21 finding 1: valid_count == 0 is fatal regardless of strict — no behavior change there."""
    from signet_trainer.data.precomputed import PrecomputedDataset

    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "a.pt")
    (root / "conditions").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="(?i)no valid samples|matching"):
        PrecomputedDataset(str(root), strict=True)


def test_import_pulls_no_encoder() -> None:
    """Test 4 (loop-purity headline): no modal/ltx_core/ltx_trainer after import."""
    for mod in ("modal", "ltx_core", "ltx_trainer"):
        sys.modules.pop(mod, None)
    import signet_trainer.data.precomputed  # noqa: F401

    for mod in ("modal", "ltx_core", "ltx_trainer"):
        assert mod not in sys.modules, f"loop-purity violation: data.precomputed imported {mod}"


def test_write_manifest_one_json_per_line_preserves_media_path(tmp_path: Path) -> None:
    """Test 5: write_manifest writes one JSON object per line, utf-8, media_path verbatim."""
    from signet_trainer.data.dataset_file import write_manifest

    rows = [
        {"caption": "a cat", "media_path": "clips/a.mp4"},
        {"caption": "a dog", "media_path": "clips/b.mp4"},
    ]
    out = write_manifest(rows, tmp_path / "metadata.jsonl")

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["media_path"] == "clips/a.mp4"
    assert parsed[1]["caption"] == "a dog"
