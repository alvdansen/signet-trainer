"""dryrun/shapes.py LTX-2.5 coverage (issue #53 Stage 1, LTX25_STAGE1_DESIGN.md §8) — CPU only.

NOT a new dispatch arm: ``model.family`` stays ``"ltx"`` for both generations, so the existing LTX
branch of ``build_dryrun_inputs``/``_assert_contract`` already runs for a ``ltx_generation=='2.5'``
config unchanged. These tests prove that parity explicitly, plus the belt-and-braces
``_assert_ltx25_dryrun_contract`` re-check.
"""

from __future__ import annotations

import pytest

from signet_trainer.config.schema import SignetConfig
from signet_trainer.dryrun.shapes import build_dryrun_inputs, run_dryrun


def _payload(**over) -> dict:
    payload: dict = {
        "training_dims": [768, 352, 25],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "training": {"max_steps": 100},
    }
    payload.update(over)
    return payload


def test_dryrun_25_split_layout_produces_the_same_shape_as_23() -> None:
    """A ltx_generation=='2.5', checkpoint_layout='split' config produces the SAME ModelInputs
    shape as the equivalent 2.3 config — proving the LTX dryrun path is generation-agnostic (§8):
    it is the existing branch, not a new one."""
    cfg_23 = SignetConfig(**_payload())
    cfg_25 = SignetConfig(
        **_payload(
            model={"ltx_generation": "2.5"},
            ltx25={"checkpoint_layout": "split", "video_vae_path": "ltx25_vae.safetensors"},
        )
    )

    mi_23 = build_dryrun_inputs(cfg_23)
    mi_25 = build_dryrun_inputs(cfg_25)

    assert tuple(mi_23.video.latent.shape) == tuple(mi_25.video.latent.shape)
    assert tuple(mi_23.video.positions.shape) == tuple(mi_25.video.positions.shape)
    assert tuple(mi_23.video_targets.shape) == tuple(mi_25.video_targets.shape)


def test_dryrun_23_unaffected_byte_for_byte() -> None:
    """Regression gate: an existing (gen-2.3) config's dry-run output is untouched by this change."""
    from signet_trainer.conditioning.strategy import compute_seq_len

    cfg = SignetConfig(**_payload())
    mi = build_dryrun_inputs(cfg)
    assert tuple(mi.video.latent.shape) == (1, compute_seq_len(768, 352, 25), 128)


def test_dryrun_25_monolith_passes_without_ltx25_block_set() -> None:
    """The common case: ltx_generation='2.5' with the ltx25 block left at its monolith default."""
    cfg = SignetConfig(**_payload(model={"ltx_generation": "2.5"}))
    mi = build_dryrun_inputs(cfg)
    assert mi is not None


def test_dryrun_25_split_without_video_vae_path_is_refused_at_config_load() -> None:
    """The SignetConfig-level guard fires first — build_dryrun_inputs is never even reached with
    an illegal split-without-video_vae_path config (belt-and-braces has nothing to catch here,
    but the config-load guard must still hold)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="video_vae_path"):
        SignetConfig(**_payload(model={"ltx_generation": "2.5"}, ltx25={"checkpoint_layout": "split"}))


def test_run_dryrun_ok_for_a_25_config() -> None:
    cfg = SignetConfig(**_payload(model={"ltx_generation": "2.5"}))
    assert run_dryrun(cfg, mode="train") == 0
    assert run_dryrun(cfg, mode="preprocess") == 0


def test_run_dryrun_refuses_sample_mode_for_a_25_config(capsys) -> None:
    cfg = SignetConfig(**_payload(model={"ltx_generation": "2.5"}))
    rc = run_dryrun(cfg, mode="sample")
    assert rc != 0
    err = capsys.readouterr().err
    assert "Stage-2" in err


def test_run_dryrun_refuses_fuse_mode_for_a_25_config() -> None:
    cfg = SignetConfig(**_payload(model={"ltx_generation": "2.5"}))
    assert run_dryrun(cfg, mode="fuse") != 0


def test_run_dryrun_allows_restore_and_backup_for_a_25_config() -> None:
    """restore/backup are cleared as generation-agnostic — no refusal for either."""
    cfg = SignetConfig(**_payload(model={"ltx_generation": "2.5"}))
    assert run_dryrun(cfg, mode="restore") == 0
    assert run_dryrun(cfg, mode="backup") == 0
