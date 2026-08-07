"""CONF-03 / D-11/D-12/D-13 — the CPU dry-run hard gate.

``main(argv)`` (the ``signet-dryrun`` console-script target) must:
  * ``load_config`` the YAML (so a bad frame count exits non-zero HERE, before batch construction);
  * compute ``seq_len`` from ``training_dims`` via the shared helper;
  * build a synthetic ``ModelInputs``/``Modality`` on CPU (no weights, no VAE);
  * assert the REAL contract shapes/masks (RESEARCH.md Q4); and
  * return 0 on success, non-zero on ANY violation.

For [768, 512, 49]:  seq_len = (512//32)*(768//32)*((49-1)//8+1) = 16*24*7 = 2688.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from signet_trainer.dryrun.shapes import main

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_YAML = str(REPO_ROOT / "configs" / "ltx23_lora.example.yaml")
BAD_YAML = str(REPO_ROOT / "configs" / "bad_frames.example.yaml")
SINGLE_FRAME_YAML = str(REPO_ROOT / "configs" / "ltx23_single_frame.example.yaml")

EXPECTED_SEQ_LEN = 2688
VIDEO_CHANNELS = 128


def test_valid_config_exits_zero():
    assert main([VALID_YAML]) == 0


def test_bad_frames_config_exits_nonzero():
    # The config validator fires before batch construction -> non-zero, clear error.
    assert main([BAD_YAML]) != 0


def test_built_batch_has_real_contract_shapes(capsys):
    # Expose the built ModelInputs for shape assertions via the builder used by main().
    from signet_trainer.dryrun.shapes import build_dryrun_inputs
    from signet_trainer.config.load import load_config

    cfg = load_config(VALID_YAML)
    mi = build_dryrun_inputs(cfg)

    # video.latent [1, seq_len, 128]
    assert tuple(mi.video.latent.shape) == (1, EXPECTED_SEQ_LEN, VIDEO_CHANNELS)
    # positions [1, 3, seq_len, 2]
    assert tuple(mi.video.positions.shape) == (1, 3, EXPECTED_SEQ_LEN, 2)
    # video_targets [1, seq_len, 128]
    assert tuple(mi.video_targets.shape) == (1, EXPECTED_SEQ_LEN, VIDEO_CHANNELS)
    # sigma [1]
    assert tuple(mi.video.sigma.shape) == (1,)
    # timesteps [1, seq_len]
    assert tuple(mi.video.timesteps.shape) == (1, EXPECTED_SEQ_LEN)


def test_built_batch_mask_invariants():
    from signet_trainer.dryrun.shapes import build_dryrun_inputs
    from signet_trainer.config.load import load_config

    cfg = load_config(VALID_YAML)
    mi = build_dryrun_inputs(cfg)

    cond_mask = mi.video.timesteps == 0
    # video_loss_mask == ~conditioning_mask
    assert torch.equal(mi.video_loss_mask, ~cond_mask)
    # timesteps[conditioning_mask] == 0
    assert torch.all(mi.video.timesteps[cond_mask] == 0)
    # cond_mask shape is [1, seq_len]
    assert tuple(cond_mask.shape) == (1, EXPECTED_SEQ_LEN)


def test_dryrun_does_not_import_modal_or_ltx_core():
    # Running the gate must not pull in modal / ltx_core (Anti-Pattern 6 / Pitfall 4).
    main([VALID_YAML])
    assert "modal" not in sys.modules
    assert "ltx_core" not in sys.modules
    assert "ltx_trainer" not in sys.modules


# --------------------------------------------------------------------------------------------------
# 05-04: the gate exercises the real conditioning config field (single_frame, p=1.0)
# --------------------------------------------------------------------------------------------------


def test_single_frame_config_exits_zero():
    # A single_frame config (mode=single_frame, first_frame_conditioning_p=1.0) passes the contract.
    assert main([SINGLE_FRAME_YAML]) == 0


def test_conditioning_field_drives_region_without_dry_run_gate():
    # With the explicit dry_run knob OFF, the conditioning config field alone must still mark the
    # first-frame region — proving the field is wired into the gate (not the dry_run knob masking it).
    from signet_trainer.config.load import load_config
    from signet_trainer.dryrun.shapes import build_dryrun_inputs, run_dryrun

    cfg = load_config(SINGLE_FRAME_YAML)
    assert cfg.conditioning.mode == "single_frame"
    assert cfg.conditioning.first_frame_conditioning_p > 0
    cfg_no_gate = cfg.model_copy(
        update={"dry_run": cfg.dry_run.model_copy(update={"cond_first_frame": False})}
    )

    mi = build_dryrun_inputs(cfg_no_gate)
    cond_mask = mi.video.timesteps == 0
    assert cond_mask.any(), "conditioning field (single_frame, p>0) must mark the first-frame region"
    # The contract still holds and the gate returns 0.
    assert torch.equal(mi.video_loss_mask, ~cond_mask)
    assert run_dryrun(cfg_no_gate) == 0


def test_no_conditioning_path_marks_no_region():
    # Both gates OFF (mode=none + dry_run knob off) -> no conditioning tokens; the gate still passes.
    from signet_trainer.config.load import load_config
    from signet_trainer.dryrun.shapes import build_dryrun_inputs, run_dryrun

    cfg = load_config(VALID_YAML)
    cfg_off = cfg.model_copy(
        update={"dry_run": cfg.dry_run.model_copy(update={"cond_first_frame": False})}
    )
    assert cfg_off.conditioning.mode == "none"

    mi = build_dryrun_inputs(cfg_off)
    cond_mask = mi.video.timesteps == 0
    assert not cond_mask.any(), "no conditioning gate -> no first-frame region marked"
    assert run_dryrun(cfg_off) == 0


# --------------------------------------------------------------------------------------------------
# 06-04: the multi_frame branch — N-item strength-scaled mask against the EXPLICIT video_loss_mask
# --------------------------------------------------------------------------------------------------

# per-latent-frame token span for [768, 512, 49]: (512//32)*(768//32) = 16*24 = 384.
PER_FRAME_TOKENS = 384
SAMPLED_SIGMA = 0.5


def _multi_frame_payload(items: list[dict], max_items: int = 3) -> dict:
    """A minimal valid multi_frame SignetConfig payload (same LTX-2.3 substrate as the example YAMLs).

    Leaves the Phase-5 single-frame fields at their defaults (the schema's lean field-split rejects
    non-default first_frame_* / reference_images while mode == 'multi_frame').
    """
    return {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "training": {"max_steps": 100},
        "conditioning": {
            "mode": "multi_frame",
            "conditioning_items": items,
            "max_conditioning_items": max_items,
        },
    }


def test_multi_frame_config_exits_zero():
    # A multi_frame config (2 items at frame_index 0 and 48, strengths 1.0 and 0.5) passes the gate.
    from signet_trainer.config.schema import SignetConfig
    from signet_trainer.dryrun.shapes import run_dryrun

    cfg = SignetConfig.model_validate(
        _multi_frame_payload(
            [
                {"image": "kf0.png", "frame_index": 0, "strength": 1.0},
                {"image": "kf48.png", "frame_index": 48, "strength": 0.5},
            ]
        )
    )
    assert cfg.conditioning.mode == "multi_frame"
    assert run_dryrun(cfg) == 0


def test_multi_frame_partial_strength_token_in_loss_mask():
    # Pitfall 1: a partial-strength (0.5) token has timesteps != 0 yet MUST be True in the loss mask,
    # and the loss mask equals (reconstructed denoise_mask > 0) — NOT (timesteps == 0).
    from signet_trainer.config.schema import SignetConfig
    from signet_trainer.dryrun.shapes import build_dryrun_inputs

    cfg = SignetConfig.model_validate(
        _multi_frame_payload(
            [
                {"image": "kf0.png", "frame_index": 0, "strength": 1.0},   # hard-clean region
                {"image": "kf48.png", "frame_index": 48, "strength": 0.5}, # partial-strength region
            ]
        )
    )
    mi = build_dryrun_inputs(cfg)

    denoise_mask = mi.video.timesteps / SAMPLED_SIGMA
    # Loss mask is the EXPLICIT (denoise_mask > 0), not the timesteps == 0 reconstruction.
    assert torch.equal(mi.video_loss_mask, denoise_mask > 0)

    # frame_index 48 -> latent frame 6 -> token span [2304, 2688); strength 0.5 -> denoise 0.5.
    partial_start = 6 * PER_FRAME_TOKENS
    partial_stop = 7 * PER_FRAME_TOKENS
    partial_ts = mi.video.timesteps[:, partial_start:partial_stop]
    assert torch.all(partial_ts != 0), "partial-strength (0.5) tokens must carry a non-zero timestep"
    assert torch.all(
        mi.video_loss_mask[:, partial_start:partial_stop]
    ), "partial-strength tokens must be IN the loss mask despite timesteps != 0 (Pitfall 1)"

    # The hard-clean (strength 1.0) region at latent frame 0 is EXCLUDED from the loss.
    assert torch.all(mi.video.timesteps[:, :PER_FRAME_TOKENS] == 0)
    assert not mi.video_loss_mask[:, :PER_FRAME_TOKENS].any()


def test_multi_frame_count1_idx0_strength1_equals_single_frame():
    # A count=1 / idx=0 / strength=1 multi_frame config produces the SAME mask as single_frame at p=1.
    from signet_trainer.config.load import load_config
    from signet_trainer.config.schema import SignetConfig
    from signet_trainer.dryrun.shapes import build_dryrun_inputs

    sf_cfg = load_config(SINGLE_FRAME_YAML)
    sf_mi = build_dryrun_inputs(sf_cfg)

    mf_cfg = SignetConfig.model_validate(
        _multi_frame_payload(
            [{"image": "kf0.png", "frame_index": 0, "strength": 1.0}],
            max_items=1,
        )
    )
    mf_mi = build_dryrun_inputs(mf_cfg)

    # Same per-token timesteps and same loss mask (velocity targets are randn -> not compared).
    assert torch.allclose(mf_mi.video.timesteps, sf_mi.video.timesteps, atol=1e-6)
    assert torch.equal(mf_mi.video_loss_mask, sf_mi.video_loss_mask)


def test_multi_frame_self_conditioning_fallback_marks_regions():
    # Empty conditioning_items (self-conditioning) -> evenly-spaced regions at the strength upper bound;
    # the gate still passes and marks per-frame-sized conditioning regions.
    from signet_trainer.config.schema import SignetConfig
    from signet_trainer.dryrun.shapes import build_dryrun_inputs, run_dryrun

    cfg = SignetConfig.model_validate(_multi_frame_payload([], max_items=3))
    assert cfg.conditioning.conditioning_items == []

    mi = build_dryrun_inputs(cfg)
    denoise_mask = mi.video.timesteps / SAMPLED_SIGMA
    assert torch.equal(mi.video_loss_mask, denoise_mask > 0)
    # Some tokens are conditioning (denoise < 1) and some are targets (denoise == 1).
    assert (denoise_mask < 1).any(), "self-conditioning fallback must mark conditioning regions"
    assert (denoise_mask == 1).any(), "targets (denoise == 1) must remain"
    assert run_dryrun(cfg) == 0
