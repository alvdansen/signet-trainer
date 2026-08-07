"""07-07 tier-1 CPU gate — the dry-run ``ic_lora`` branch proves the L-3 invariant with zero GPU.

The IC-LoRA in-context strategy concatenates a CLEAN reference prefix to the noised target (a
DOUBLED sequence). The dead-adapter bug (Pitfall 1 / L-3) is: loss lands on the reference prefix, so
the adapter learns nothing controllable. This CPU gate makes that impossible to ship un-caught — it
asserts, on synthetic tensors, that:

  (a) the reference-prefix half of ``video_loss_mask`` is ALL False (``[:ref_seq_len]``);
  (b) ``video_targets`` is TARGET-ONLY (length ``combined - ref_seq_len``, NOT the combined seq);
  (c) the ``positions`` tensor is the DOUBLED ``(1, 3, combined, 2)`` shape.

For [832, 480, 25]: seq_len = (480//32)*(832//32)*((25-1)//8+1) = 15*26*4 = 1560; combined = 3120.
CPU only — no ltx_core / modal / CUDA import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from signet_trainer.config.load import load_config
from signet_trainer.conditioning.strategy import VIDEO_LATENT_CHANNELS, compute_seq_len
from signet_trainer.dryrun.shapes import build_dryrun_inputs, main, run_dryrun

REPO_ROOT = Path(__file__).resolve().parents[1]
IC_LORA_YAML = str(REPO_ROOT / "configs" / "ltx23_ic_lora.example.yaml")

# [832, 480, 25] -> 15 * 26 * 4 = 1560; the DOUBLED ref+target sequence = 3120.
EXPECTED_SEQ_LEN = 1560
EXPECTED_COMBINED = 3120


def test_ic_lora_example_config_exits_zero():
    # signet-dryrun on the ic_lora example config exits 0 (tier-1 phase gate).
    assert main([IC_LORA_YAML]) == 0


def test_ic_lora_run_dryrun_exits_zero():
    cfg = load_config(IC_LORA_YAML)
    assert cfg.conditioning.mode == "ic_lora"
    assert run_dryrun(cfg) == 0


def test_ic_lora_doubled_sequence_shapes():
    cfg = load_config(IC_LORA_YAML)
    width, height, frames = cfg.training_dims
    seq_len = compute_seq_len(width, height, frames)
    assert seq_len == EXPECTED_SEQ_LEN
    combined = 2 * seq_len
    assert combined == EXPECTED_COMBINED

    mi = build_dryrun_inputs(cfg)

    # ref_seq_len populated and 1:1 with the target (D-7-REF11).
    assert mi.ref_seq_len == seq_len
    # (c) DOUBLED positions / latent / timesteps / loss mask.
    assert tuple(mi.video.positions.shape) == (1, 3, combined, 2)
    assert tuple(mi.video.latent.shape) == (1, combined, VIDEO_LATENT_CHANNELS)
    assert tuple(mi.video.timesteps.shape) == (1, combined)
    assert tuple(mi.video_loss_mask.shape) == (1, combined)


def test_ic_lora_video_targets_are_target_only():
    # (b) video_targets length == the target slice length (combined - ref_seq_len == seq_len).
    cfg = load_config(IC_LORA_YAML)
    mi = build_dryrun_inputs(cfg)
    ref_seq_len = mi.ref_seq_len
    combined = tuple(mi.video.latent.shape)[1]
    assert tuple(mi.video_targets.shape) == (1, combined - ref_seq_len, VIDEO_LATENT_CHANNELS)
    assert tuple(mi.video_targets.shape)[1] == EXPECTED_SEQ_LEN


def test_ic_lora_reference_prefix_loss_mask_all_false():
    # (a) THE L-3 guard: loss NEVER lands on the reference prefix.
    cfg = load_config(IC_LORA_YAML)
    mi = build_dryrun_inputs(cfg)
    ref_seq_len = mi.ref_seq_len
    assert not mi.video_loss_mask[:, :ref_seq_len].any(), (
        "reference-prefix loss mask must be ALL False (L-3)"
    )
    # The reference prefix is clean conditioning -> per-token timestep 0.
    assert torch.all(mi.video.timesteps[:, :ref_seq_len] == 0)
    # Some target-half tokens ARE in the loss (the run is not degenerate).
    assert mi.video_loss_mask[:, ref_seq_len:].any(), "target half must contribute to the loss"


def test_ic_lora_target_half_loss_mask_matches_conditioning():
    # The target-half loss mask is exactly ~target_conditioning_mask (reconstructed via timesteps==0).
    cfg = load_config(IC_LORA_YAML)
    mi = build_dryrun_inputs(cfg)
    ref_seq_len = mi.ref_seq_len
    target_cond_mask = mi.video.timesteps[:, ref_seq_len:] == 0
    assert torch.equal(mi.video_loss_mask[:, ref_seq_len:], ~target_cond_mask)


def test_shapes_any_annotations_resolve():
    # WR-05: _fhw / assert_paired_reference annotate params as `Any`, previously used without an
    # import — masked only by `from __future__ import annotations`. get_type_hints() must resolve them
    # (pydantic / docs tooling / any runtime annotation introspection would otherwise raise NameError).
    import typing

    from signet_trainer.dryrun import shapes

    typing.get_type_hints(shapes._fhw)
    typing.get_type_hints(shapes.assert_paired_reference)
    assert shapes.Any is typing.Any


def test_ic_lora_dryrun_does_not_import_modal_or_ltx_core():
    # The gate must stay Windows/CI-free (Anti-Pattern 6 / Pitfall 4).
    main([IC_LORA_YAML])
    assert "modal" not in sys.modules
    assert "ltx_core" not in sys.modules
    assert "ltx_trainer" not in sys.modules
