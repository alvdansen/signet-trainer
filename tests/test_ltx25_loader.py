"""models.ltx25_loader (issue #53 Stage 1) — argument-threading + metadata-gate tests, CPU only.

The mocking boundary is exactly where ``models/ltx25_loader.py`` already draws it: ``ltx_trainer``
imports are function-local, so a test exercises the argument-threading contract by injecting a
FAKE ``ltx_trainer.model_loader`` module into ``sys.modules`` — never installing the real package.
``safetensors`` IS a real, installed dependency (unlike ``ltx_trainer``), so the metadata-gate
tests build real tiny ``.safetensors`` fixtures via ``safetensors.torch.save_file``.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch
from safetensors.torch import save_file

from signet_trainer.models.ltx25_loader import (
    assert_ltx25_vae_compression,
    compute_vae_scale_factors_from_metadata,
    load_ltx25_components,
    read_ltx25_checkpoint_metadata,
)

# ======================================================================================
# load_ltx25_components — argument threading + the two decoder-adjacent raises.
# ======================================================================================


def test_decoder_flag_raises_before_any_ltx_trainer_import(monkeypatch) -> None:
    """``with_video_vae_decoder=True`` raises NotImplementedError naming Stage 2, WITHOUT ever
    importing ltx_trainer (proves the raise is BEFORE the deferred import) — remove ltx_trainer
    from sys.modules entirely (it is never installed anyway) so an accidental early import would
    surface as ModuleNotFoundError, not silently succeed."""
    monkeypatch.delitem(sys.modules, "ltx_trainer", raising=False)
    with pytest.raises(NotImplementedError, match="Stage-2"):
        load_ltx25_components(
            "ckpt.safetensors", "gemma-4-root", with_video_vae_decoder=True
        )


def test_audio_encoder_flag_raises_before_any_ltx_trainer_import(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "ltx_trainer", raising=False)
    with pytest.raises(NotImplementedError, match="meta-device"):
        load_ltx25_components(
            "ckpt.safetensors", "gemma-4-root", with_audio_vae_encoder=True
        )


def _install_fake_ltx_trainer(monkeypatch) -> list[dict]:
    """Inject a fake ``ltx_trainer.model_loader.load_model`` that RECORDS its kwargs and returns a
    sentinel — never installing the real (uninstalled) ``ltx_trainer`` package."""
    calls: list[dict] = []

    def _fake_load_model(**kwargs):
        calls.append(kwargs)
        return "SENTINEL_COMPONENTS"

    fake_pkg = types.ModuleType("ltx_trainer")
    fake_model_loader = types.ModuleType("ltx_trainer.model_loader")
    fake_model_loader.load_model = _fake_load_model
    fake_pkg.model_loader = fake_model_loader

    monkeypatch.setitem(sys.modules, "ltx_trainer", fake_pkg)
    monkeypatch.setitem(sys.modules, "ltx_trainer.model_loader", fake_model_loader)
    return calls


def test_load_ltx25_threads_split_paths_unchanged(monkeypatch) -> None:
    calls = _install_fake_ltx_trainer(monkeypatch)

    result = load_ltx25_components(
        checkpoint_path="ltx25-transformer.safetensors",
        text_encoder_path="gemma-4-12b-it",
        device="cuda",
        dtype=torch.bfloat16,
        video_vae_path="ltx25-video-vae.safetensors",
        audio_vae_path="ltx25-audio-vae.safetensors",
    )

    assert result == "SENTINEL_COMPONENTS"
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["checkpoint_path"] == "ltx25-transformer.safetensors"
    assert kwargs["text_encoder_path"] == "gemma-4-12b-it"
    assert kwargs["video_vae_path"] == "ltx25-video-vae.safetensors"
    assert kwargs["audio_vae_path"] == "ltx25-audio-vae.safetensors"
    # Stage 1 is train-only: the decoder/audio-decoder/vocoder stay off regardless of caller intent
    # (the two raise-first flags are checked BEFORE this call is ever reached).
    assert kwargs["with_video_vae_encoder"] is True
    assert kwargs["with_video_vae_decoder"] is False
    assert kwargs["with_audio_vae_decoder"] is False
    assert kwargs["with_vocoder"] is False


def test_load_ltx25_monolith_defaults_pass_none_for_split_paths(monkeypatch) -> None:
    calls = _install_fake_ltx_trainer(monkeypatch)

    load_ltx25_components(
        checkpoint_path="ltx25-monolith.safetensors",
        text_encoder_path="gemma-4-12b-it",
    )

    assert calls[0]["video_vae_path"] is None
    assert calls[0]["audio_vae_path"] is None


# ======================================================================================
# read_ltx25_checkpoint_metadata — real safetensors fixtures (safetensors IS installed).
# ======================================================================================


def _write_fixture(path, tensors: dict, metadata: dict[str, str] | None) -> None:
    save_file(tensors, str(path), metadata=metadata)


def test_read_metadata_missing_header_raises(tmp_path) -> None:
    """A synthetic .safetensors with NO __metadata__ block raises loudly, naming what's missing."""
    path = tmp_path / "no_metadata.safetensors"
    _write_fixture(path, {"blocks.0.attn1.to_q.weight": torch.zeros(4, 4)}, metadata=None)

    with pytest.raises(ValueError, match="no embedded __metadata__ header"):
        read_ltx25_checkpoint_metadata(str(path))


def test_read_metadata_num_layers_self_consistency_passes(tmp_path) -> None:
    path = tmp_path / "consistent.safetensors"
    tensors = {
        f"transformer_blocks.{i}.attn1.to_q.weight": torch.zeros(4, 4) for i in range(3)
    }
    _write_fixture(path, tensors, metadata={"config": json.dumps({"num_layers": 3, "ff_bias": False})})

    summary = read_ltx25_checkpoint_metadata(str(path))
    assert summary["declared_num_layers"] == 3
    assert summary["introspected_num_layers"] == 3
    assert summary["ff_bias"] is False
    assert summary["metadata_config_present"] is True


def test_read_metadata_num_layers_mismatch_raises(tmp_path) -> None:
    path = tmp_path / "mismatch.safetensors"
    tensors = {
        f"transformer_blocks.{i}.attn1.to_q.weight": torch.zeros(4, 4) for i in range(3)
    }
    _write_fixture(path, tensors, metadata={"config": json.dumps({"num_layers": 5})})

    with pytest.raises(ValueError, match="does not match the introspected"):
        read_ltx25_checkpoint_metadata(str(path))


def test_read_metadata_records_without_asserting_against_any_hardcoded_25_constant(tmp_path) -> None:
    """D3 honesty: an UNKNOWN declared num_layers (absent from the embedded config) must still
    return a summary rather than raising — nothing here compares against an EXPECTED_*_25."""
    path = tmp_path / "no_declared_layers.safetensors"
    tensors = {"transformer_blocks.0.attn1.to_q.weight": torch.zeros(4, 4)}
    _write_fixture(path, tensors, metadata={"config": json.dumps({"ff_bias": False})})

    summary = read_ltx25_checkpoint_metadata(str(path))
    assert summary["declared_num_layers"] is None
    assert summary["introspected_num_layers"] == 1


# ======================================================================================
# assert_ltx25_vae_compression — the (8, 32, 32) self-consistency confirmation.
# ======================================================================================


def test_vae_compression_matches_default_when_no_vae_block_list_present(tmp_path) -> None:
    """No 'vae' key at all in the embedded config -> falls back to the EXISTING (8,32,32), same
    fallback the upstream mechanism itself uses (e.g. an audio-only checkpoint)."""
    path = tmp_path / "no_vae_config.safetensors"
    _write_fixture(path, {"x": torch.zeros(1)}, metadata={"config": json.dumps({})})

    observed = assert_ltx25_vae_compression(str(path))
    assert observed == {
        "vae_temporal_compression": 8,
        "vae_spatial_compression_h": 32,
        "vae_spatial_compression_w": 32,
    }


def test_vae_compression_matching_block_list_passes(tmp_path) -> None:
    """A synthetic encoder_blocks list resolving to (8,32,32) — 3x compress_time, 5x
    compress_space (2**3=8, 2**5=32) — passes."""
    path = tmp_path / "matching_vae.safetensors"
    vae_config = {
        "encoder_blocks": (
            [["compress_time_1", {}], ["compress_time_2", {}], ["compress_time_3", {}]]
            + [["compress_space_%d" % i, {}] for i in range(5)]
        )
    }
    _write_fixture(path, {"x": torch.zeros(1)}, metadata={"config": json.dumps({"vae": vae_config})})

    observed = assert_ltx25_vae_compression(str(path))
    assert observed == {
        "vae_temporal_compression": 8,
        "vae_spatial_compression_h": 32,
        "vae_spatial_compression_w": 32,
    }


def test_vae_compression_mismatch_raises_naming_both_values(tmp_path) -> None:
    """A synthetic metadata declaring a VAE block list that resolves to (4, 16, 8) instead of
    (8, 32, 32) raises, naming both values."""
    path = tmp_path / "mismatched_vae.safetensors"
    vae_config = {
        "encoder_blocks": (
            [["compress_time_1", {}], ["compress_time_2", {}]]  # 2**2 = 4
            + [["compress_space_%d" % i, {}] for i in range(3)]  # 2**3 = 8
        )
    }
    _write_fixture(path, {"x": torch.zeros(1)}, metadata={"config": json.dumps({"vae": vae_config})})

    with pytest.raises(ValueError, match="does NOT match"):
        assert_ltx25_vae_compression(str(path))


def test_vae_compression_reads_split_video_vae_path_when_given(tmp_path) -> None:
    """Split layout: the VAE metadata lives in the SEPARATE video_vae_path file, not the
    (checkpoint) transformer file."""
    transformer_path = tmp_path / "transformer_only.safetensors"
    _write_fixture(transformer_path, {"x": torch.zeros(1)}, metadata={"config": json.dumps({})})

    vae_path = tmp_path / "video_vae.safetensors"
    _write_fixture(vae_path, {"x": torch.zeros(1)}, metadata={"config": json.dumps({"vae": {}})})

    observed = assert_ltx25_vae_compression(str(transformer_path), str(vae_path))
    assert observed["vae_temporal_compression"] == 8  # empty vae dict -> the (8,32,32) fallback


def test_compute_vae_scale_factors_from_metadata_none_falls_back_to_default() -> None:
    assert compute_vae_scale_factors_from_metadata(None) == (8, 32, 32)
    assert compute_vae_scale_factors_from_metadata({}) == (8, 32, 32)
