"""CPU contract tests for the In-Outpainting fuse module (modal/fuse.py, GATE-SPEC rev 2).

Covers the CPU-testable logic ONLY — no modal, no ltx_core, no GPU, no 44GB files:

  * key-prefix transformation (``diffusion_model.*`` -> ``model.diffusion_model.*``) on
    synthetic dicts ([precedent] the prior project's validated rename);
  * config-preserving metadata build + the [house] fail-fast on a config-less base
    ([precedent] prior-project BUGLOG #5 — the 3840-vs-2048/4096 connector crash);
  * ``verify_fused_metadata`` (the pre-metered-dispatch gate) against tiny synthetic
    safetensors files — config present, audio intact (BUGLOG #6 do-NOT-audio-strip),
    names/shapes/dtypes unchanged vs the base;
  * the fuse_keycheck-style diff logic (targets derived exactly like upstream
    ``_affected_weight_keys``; changed-tensor probe).

The heavy ``fuse_inoutpaint`` path (hf download + ltx_core ``apply_loras``) runs Modal-side
only and is exercised by the gated fuse job, not here (Anti-Pattern 6 / zero-spend).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from signet_trainer.modal.fuse import (
    FUSED_MARKER,
    FUSED_MODEL_VERSION,
    build_fused_metadata,
    diff_changed_keys,
    expected_safetensors_size,
    fuse_inoutpaint,
    fuse_keycheck,
    is_fused_base_filename,
    lora_target_weight_keys,
    prepend_model_prefix,
    read_safetensors_header,
    sha256_file,
    split_fuse_targets,
    verify_fused_metadata,
)

# A representative embedded model config (the REAL one is the full ltx-core model config JSON;
# only presence + verbatim preservation matter to the gate).
SYNTHETIC_CONFIG = '{"video_dim": 4096, "audio_dim": 2048}'


# --------------------------------------------------------------------------------------------------
# Key-prefix transformation (diffusion_model.* -> model.diffusion_model.*)
# --------------------------------------------------------------------------------------------------


def test_prepend_model_prefix_renames_diffusion_model_keys() -> None:
    lora = {
        "diffusion_model.blocks.0.attn1.to_q.lora_A.weight": 1,
        "diffusion_model.blocks.0.attn1.to_q.lora_B.weight": 2,
        "diffusion_model.blocks.47.ff.net.2.lora_A.weight": 3,
    }
    renamed = prepend_model_prefix(lora)
    assert set(renamed) == {
        "model.diffusion_model.blocks.0.attn1.to_q.lora_A.weight",
        "model.diffusion_model.blocks.0.attn1.to_q.lora_B.weight",
        "model.diffusion_model.blocks.47.ff.net.2.lora_A.weight",
    }
    # Values ride along untouched (identity, not copies-of).
    assert renamed["model.diffusion_model.blocks.0.attn1.to_q.lora_A.weight"] == 1
    assert renamed["model.diffusion_model.blocks.47.ff.net.2.lora_A.weight"] == 3


def test_prepend_model_prefix_is_idempotent_on_prefixed_keys() -> None:
    """A key already carrying ``model.`` must pass through — never double-prefix."""
    once = prepend_model_prefix({"diffusion_model.blocks.0.attn1.to_q.lora_A.weight": 1})
    twice = prepend_model_prefix(once)
    assert twice == once


def test_prepend_model_prefix_collision_raises() -> None:
    """Prefixed + unprefixed forms of one key must not silently collapse into one tensor."""
    lora = {
        "diffusion_model.blocks.0.attn1.to_q.lora_A.weight": 1,
        "model.diffusion_model.blocks.0.attn1.to_q.lora_A.weight": 2,
    }
    with pytest.raises(ValueError, match="collision"):
        prepend_model_prefix(lora)


# --------------------------------------------------------------------------------------------------
# Target derivation + hit/miss split (mirrors upstream _affected_weight_keys at the pin)
# --------------------------------------------------------------------------------------------------


def test_lora_target_weight_keys_derives_from_lora_a_only() -> None:
    keys = [
        "model.diffusion_model.blocks.0.attn1.to_q.lora_A.weight",
        "model.diffusion_model.blocks.0.attn1.to_q.lora_B.weight",  # B never derives a target
        "model.diffusion_model.blocks.0.attn1.to_q.alpha",  # non-lora key ignored
    ]
    assert lora_target_weight_keys(keys) == {
        "model.diffusion_model.blocks.0.attn1.to_q.weight"
    }


def test_split_fuse_targets_hits_and_misses() -> None:
    renamed = [
        "model.diffusion_model.blocks.0.attn1.to_q.lora_A.weight",
        "model.diffusion_model.blocks.0.attn1.to_q.lora_B.weight",
        "model.diffusion_model.blocks.0.nonexistent.lora_A.weight",
    ]
    base = [
        "model.diffusion_model.blocks.0.attn1.to_q.weight",
        "model.diffusion_model.blocks.0.attn1.to_k.weight",
    ]
    hits, misses = split_fuse_targets(renamed, base)
    assert hits == ["model.diffusion_model.blocks.0.attn1.to_q.weight"]
    assert misses == ["model.diffusion_model.blocks.0.nonexistent.weight"]


def test_split_fuse_targets_bad_rename_yields_zero_hits() -> None:
    """The un-renamed adapter (no ``model.`` prefix) hits NOTHING — the abort signature."""
    unrenamed = ["diffusion_model.blocks.0.attn1.to_q.lora_A.weight"]
    base = ["model.diffusion_model.blocks.0.attn1.to_q.weight"]
    hits, misses = split_fuse_targets(unrenamed, base)
    assert hits == []
    assert misses == ["diffusion_model.blocks.0.attn1.to_q.weight"]


# --------------------------------------------------------------------------------------------------
# Config-preserving metadata build (prior-project BUGLOG #5)
# --------------------------------------------------------------------------------------------------


def test_build_fused_metadata_preserves_config_and_markers() -> None:
    meta = build_fused_metadata({"config": SYNTHETIC_CONFIG, "other": "ignored"})
    assert meta["config"] == SYNTHETIC_CONFIG  # verbatim — the BUGLOG-#5 fix
    assert meta["fused"] == FUSED_MARKER
    assert meta["model_version"] == FUSED_MODEL_VERSION


@pytest.mark.parametrize("base_metadata", [None, {}, {"config": ""}])
def test_build_fused_metadata_missing_config_raises(base_metadata) -> None:
    """[house] fail-fast: a config-less base must never produce a 44GB unloadable artifact."""
    with pytest.raises(RuntimeError, match="config"):
        build_fused_metadata(base_metadata)


def test_build_fused_metadata_missing_config_allowed_when_opted_out() -> None:
    meta = build_fused_metadata(None, require_config=False)
    assert "config" not in meta
    assert meta["fused"] == FUSED_MARKER


# --------------------------------------------------------------------------------------------------
# Header read + verify_fused_metadata against tiny synthetic safetensors files
# --------------------------------------------------------------------------------------------------

pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402 — after the importorskip guard


def _write_checkpoint(
    path: Path,
    *,
    metadata: dict[str, str] | None,
    with_audio: bool = True,
    q_value: float = 1.0,
    q_shape: tuple[int, ...] = (4, 4),
) -> str:
    """Write a tiny synthetic 'base-shaped' checkpoint: one video weight + one audio weight."""
    tensors = {
        "model.diffusion_model.blocks.0.attn1.to_q.weight": torch.full(q_shape, q_value),
    }
    if with_audio:
        tensors["model.diffusion_model.audio_transformer_blocks.0.attn1.to_q.weight"] = (
            torch.zeros(2, 2)
        )
    save_file(tensors, str(path), metadata=metadata)
    return str(path)


def test_read_safetensors_header_roundtrip(tmp_path: Path) -> None:
    path = _write_checkpoint(tmp_path / "base.safetensors", metadata={"config": SYNTHETIC_CONFIG})
    metadata, tensors = read_safetensors_header(path)
    assert metadata["config"] == SYNTHETIC_CONFIG
    assert set(tensors) == {
        "model.diffusion_model.blocks.0.attn1.to_q.weight",
        "model.diffusion_model.audio_transformer_blocks.0.attn1.to_q.weight",
    }
    # Header-only shape facts (never materialized) — what the base-comparison gate reads.
    assert tensors["model.diffusion_model.blocks.0.attn1.to_q.weight"]["shape"] == [4, 4]


def test_read_safetensors_header_no_metadata(tmp_path: Path) -> None:
    path = _write_checkpoint(tmp_path / "bare.safetensors", metadata=None)
    metadata, tensors = read_safetensors_header(path)
    assert metadata == {}
    assert len(tensors) == 2


def test_verify_fused_metadata_passes_on_sound_file(tmp_path: Path) -> None:
    base = _write_checkpoint(
        tmp_path / "base.safetensors", metadata={"config": SYNTHETIC_CONFIG}, q_value=1.0
    )
    fused = _write_checkpoint(
        tmp_path / "fused.safetensors",
        metadata={"config": SYNTHETIC_CONFIG, "fused": FUSED_MARKER, "model_version": "2.3"},
        q_value=2.0,  # values changed, names/shapes/dtypes identical — the legal fuse delta
    )
    summary = verify_fused_metadata(fused, base_path=base)
    assert summary["n_keys"] == 2
    assert summary["n_audio_keys"] == 1
    assert summary["config_chars"] == len(SYNTHETIC_CONFIG)
    assert summary["fused_marker"] == FUSED_MARKER
    assert summary["base_compared"] is True


def test_verify_fused_metadata_missing_config_raises(tmp_path: Path) -> None:
    """The BUGLOG-#5 gate itself: a fused file without meta['config'] must be refused."""
    fused = _write_checkpoint(tmp_path / "fused.safetensors", metadata={"fused": FUSED_MARKER})
    with pytest.raises(RuntimeError, match="config"):
        verify_fused_metadata(fused)


def test_verify_fused_metadata_audio_stripped_raises(tmp_path: Path) -> None:
    """BUGLOG #6 do-NOT-audio-strip: zero audio-branch weights must be refused."""
    fused = _write_checkpoint(
        tmp_path / "fused.safetensors", metadata={"config": SYNTHETIC_CONFIG}, with_audio=False
    )
    with pytest.raises(RuntimeError, match="audio"):
        verify_fused_metadata(fused)


def test_verify_fused_metadata_audio_check_can_be_opted_out(tmp_path: Path) -> None:
    fused = _write_checkpoint(
        tmp_path / "vonly.safetensors",
        metadata={"config": SYNTHETIC_CONFIG, "fused": FUSED_MARKER},
        with_audio=False,
    )
    summary = verify_fused_metadata(fused, require_audio=False)
    assert summary["n_audio_keys"] == 0


def test_verify_fused_metadata_shape_change_vs_base_raises(tmp_path: Path) -> None:
    """Fusing is value-only — a shape drift vs the base invalidates the EXPECTED_* gate."""
    base = _write_checkpoint(
        tmp_path / "base.safetensors", metadata={"config": SYNTHETIC_CONFIG}, q_shape=(4, 4)
    )
    fused = _write_checkpoint(
        tmp_path / "fused.safetensors",
        metadata={"config": SYNTHETIC_CONFIG, "fused": FUSED_MARKER},
        q_shape=(8, 8),
    )
    with pytest.raises(RuntimeError, match="shape"):
        verify_fused_metadata(fused, base_path=base)


def test_verify_fused_metadata_key_set_drift_vs_base_raises(tmp_path: Path) -> None:
    base = _write_checkpoint(
        tmp_path / "base.safetensors", metadata={"config": SYNTHETIC_CONFIG}, with_audio=True
    )
    # Fused file DROPPED the audio key but still has >=1 'audio'-free key set difference.
    fused_tensors = {
        "model.diffusion_model.blocks.0.attn1.to_q.weight": torch.ones(4, 4),
        "model.diffusion_model.blocks.0.attn1.to_k.weight": torch.ones(4, 4),  # extra
        "model.diffusion_model.audio_transformer_blocks.0.attn1.to_q.weight": torch.zeros(2, 2),
    }
    fused = tmp_path / "fused.safetensors"
    save_file(
        fused_tensors, str(fused), metadata={"config": SYNTHETIC_CONFIG, "fused": FUSED_MARKER}
    )
    with pytest.raises(RuntimeError, match="NAME set"):
        verify_fused_metadata(str(fused), base_path=base)


def test_verify_fused_metadata_config_drift_vs_base_raises(tmp_path: Path) -> None:
    """The preserved config must match the base's byte-for-byte — not just be present."""
    base = _write_checkpoint(
        tmp_path / "base.safetensors", metadata={"config": SYNTHETIC_CONFIG}
    )
    fused = _write_checkpoint(
        tmp_path / "fused.safetensors",
        metadata={"config": '{"video_dim": 3840}', "fused": FUSED_MARKER},
        q_value=2.0,
    )
    with pytest.raises(RuntimeError, match="DIFFERS"):
        verify_fused_metadata(fused, base_path=base)


# --------------------------------------------------------------------------------------------------
# fuse_keycheck (header-only, runs on tiny files) + the changed-tensor diff probe
# --------------------------------------------------------------------------------------------------


def test_fuse_keycheck_hits_on_matching_rename(tmp_path: Path) -> None:
    base = _write_checkpoint(tmp_path / "base.safetensors", metadata={"config": SYNTHETIC_CONFIG})
    lora = tmp_path / "lora.safetensors"
    save_file(
        {
            "diffusion_model.blocks.0.attn1.to_q.lora_A.weight": torch.ones(2, 4),
            "diffusion_model.blocks.0.attn1.to_q.lora_B.weight": torch.ones(4, 2),
        },
        str(lora),
    )
    report = fuse_keycheck(base, str(lora))
    assert report["n_hits"] == 1
    assert report["hits"] == ["model.diffusion_model.blocks.0.attn1.to_q.weight"]
    assert report["misses"] == []


def test_fuse_keycheck_zero_hits_raises(tmp_path: Path) -> None:
    """A lora whose renamed targets miss every base weight must abort before any heavy load."""
    base = _write_checkpoint(tmp_path / "base.safetensors", metadata={"config": SYNTHETIC_CONFIG})
    lora = tmp_path / "lora.safetensors"
    save_file(
        {"diffusion_model.blocks.9.some_other_module.lora_A.weight": torch.ones(2, 4)}, str(lora)
    )
    with pytest.raises(RuntimeError, match="ZERO"):
        fuse_keycheck(base, str(lora))


def test_diff_changed_keys_detects_only_real_changes() -> None:
    base_sd = {"a.weight": torch.ones(2, 2), "b.weight": torch.ones(2, 2)}
    fused_sd = {"a.weight": torch.full((2, 2), 2.0), "b.weight": torch.ones(2, 2)}
    assert diff_changed_keys(base_sd, fused_sd, ["a.weight", "b.weight"]) == ["a.weight"]


def test_diff_changed_keys_skips_absent_candidates() -> None:
    base_sd = {"a.weight": torch.ones(2, 2)}
    fused_sd = {"a.weight": torch.full((2, 2), 2.0)}
    # Candidate absent from either dict is skipped, never a KeyError.
    assert diff_changed_keys(base_sd, fused_sd, ["a.weight", "ghost.weight"]) == ["a.weight"]


def test_diff_changed_keys_zero_changes_is_the_no_op_signature() -> None:
    base_sd = {"a.weight": torch.ones(2, 2)}
    fused_sd = {"a.weight": torch.ones(2, 2)}
    assert diff_changed_keys(base_sd, fused_sd, ["a.weight"]) == []


# --------------------------------------------------------------------------------------------------
# PR-6 audit fixes — truncation coverage (gap-fuse-0), overwrite guard (gap-fuse-2), and fuse
# provenance + marker assertion (gap-fuse-3). The truncation and unfused-base tests FAIL on the
# pre-fix verify_fused_metadata (which passed both).
# --------------------------------------------------------------------------------------------------


def _write_sound_fused_pair(tmp_path: Path) -> tuple[str, str]:
    base = _write_checkpoint(
        tmp_path / "base.safetensors", metadata={"config": SYNTHETIC_CONFIG}, q_value=1.0
    )
    fused = _write_checkpoint(
        tmp_path / "fused.safetensors",
        metadata={"config": SYNTHETIC_CONFIG, "fused": FUSED_MARKER, "model_version": "2.3"},
        q_value=2.0,
    )
    return base, fused


def test_expected_safetensors_size_matches_a_complete_file(tmp_path: Path) -> None:
    _, fused = _write_sound_fused_pair(tmp_path)
    assert expected_safetensors_size(fused) == Path(fused).stat().st_size


def test_verify_fused_metadata_truncated_payload_raises(tmp_path: Path) -> None:
    """gap-fuse-0 regression: safetensors writes the header FIRST, so a file cut mid-payload

    still carries a complete valid header and (pre-fix) passed every gate assertion while
    load_file raised on the real consumer. The payload-coverage check must refuse it.
    """
    base, fused = _write_sound_fused_pair(tmp_path)
    full = Path(fused).read_bytes()
    header_len = int.from_bytes(full[:8], "little")
    payload = len(full) - 8 - header_len
    Path(fused).write_bytes(full[: 8 + header_len + payload // 2])  # header intact, payload cut
    # Header-only reads still succeed on the truncated file — that is exactly the trap.
    metadata, _ = read_safetensors_header(fused)
    assert metadata["config"] == SYNTHETIC_CONFIG
    with pytest.raises(RuntimeError, match="TRUNCATED"):
        verify_fused_metadata(fused, base_path=base)


def test_verify_fused_metadata_rejects_the_unfused_base(tmp_path: Path) -> None:
    """gap-fuse-3 regression: the plain dev base (no 'fused' marker) must never verify as a

    sound fused scaffold — pre-fix it PASSED with fused_marker None, even with base comparison.
    """
    base, _ = _write_sound_fused_pair(tmp_path)
    with pytest.raises(RuntimeError, match="marker"):
        verify_fused_metadata(base, base_path=base)


def test_verify_fused_metadata_marker_assert_can_be_opted_out(tmp_path: Path) -> None:
    base, _ = _write_sound_fused_pair(tmp_path)
    summary = verify_fused_metadata(base, expected_marker=None)
    assert summary["fused_marker"] is None


def test_verify_fused_metadata_surfaces_fuse_strength(tmp_path: Path) -> None:
    fused = _write_checkpoint(
        tmp_path / "fused.safetensors",
        metadata={"config": SYNTHETIC_CONFIG, "fused": FUSED_MARKER, "fuse_strength": "1.0"},
    )
    summary = verify_fused_metadata(fused)
    assert summary["fuse_strength"] == "1.0"


def test_build_fused_metadata_records_provenance() -> None:
    """gap-fuse-3: strength + adapter identity ride INTO the artifact, as str->str metadata."""
    meta = build_fused_metadata(
        {"config": SYNTHETIC_CONFIG},
        strength=1.0,
        lora_repo="Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting",
        lora_filename="adapter.safetensors",
        lora_sha256="ab" * 32,
    )
    assert meta["fuse_strength"] == "1.0"
    assert meta["lora_repo"].endswith("In-Outpainting")
    assert meta["lora_filename"] == "adapter.safetensors"
    assert meta["lora_sha256"] == "ab" * 32
    assert all(isinstance(v, str) for v in meta.values())  # safetensors metadata is str->str


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    payload = b"adapter-bytes" * 100
    p = tmp_path / "adapter.bin"
    p.write_bytes(payload)
    assert sha256_file(str(p)) == hashlib.sha256(payload).hexdigest()


def test_is_fused_base_filename_naming_contract() -> None:
    assert is_fused_base_filename("ltx-2.3-22b-inoutpaint-fused.safetensors")
    assert is_fused_base_filename("weights/ltx-2.3-22b-inoutpaint-fused.safetensors")
    assert not is_fused_base_filename("ltx-2.3-22b-dev.safetensors")
    assert not is_fused_base_filename("ltx-2.3-22b-distilled.safetensors")
    # A directory component carrying '-fused' must not classify a non-fused file.
    assert not is_fused_base_filename("some-fused-dir/ltx-2.3-22b-dev.safetensors")


def test_fuse_inoutpaint_refuses_to_overwrite_existing_scaffold(tmp_path: Path) -> None:
    """gap-fuse-2 regression: an existing fused base is an artifact prior runs trained against —

    a re-dispatch must raise (house rule 6), BEFORE any heavy import/load, unless explicitly
    opted in. Pre-fix, save_file overwrote it in place with no warning.
    """
    base, fused = _write_sound_fused_pair(tmp_path)
    with pytest.raises(RuntimeError, match="overwrite"):
        fuse_inoutpaint(base_path=base, out_path=fused)
