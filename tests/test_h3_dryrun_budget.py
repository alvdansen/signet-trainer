"""Phase 10 (H3-04 / H3-07) — the H3 branch of the CPU dry-run HARD GATE.

The dry-run is the last free checkpoint before the cost print and the approval pause. For H3 it now
carries two jobs no LTX config ever needed:

1. **Prove on CPU that the reference prefix is excluded from the loss.** Leaving conditioning rows in
   the loss is the dead-adapter bug (L-3): the objective is dominated by rows the model is not being
   asked to generate, so the adapter learns nothing controllable. The synthetic batch is built by
   ``train/h3_step.build_h3_packed_batch`` — the SAME function training calls — so the gate cannot
   pass while training diverges.

2. **REFUSE a dispatch whose packed sequence exceeds the target GPU's measured activation ceiling.**
   This is a NEW CLASS of dry-run assertion: every existing LTX validator refuses on a divisibility
   rule, none on a VRAM budget. P10-1b's OOMs cost real money and would have been caught here for
   free.

The blocker this file exists to prevent is the NOMINAL-PAIR hole. At reference short edge 1024 the
nominal ``A+B`` pair prices at 12,362 rows and is UNDER the ceiling — a gate that prices one pair
passes, and then the first environment-bearing segment OOMs a metered A100. The gate must price the
WORST of the real 15-pair domain (``C+008``) and name it when it refuses.

Everything here is CPU-only and builds configs from dicts — no YAML, no weights, no CUDA, no Modal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from signet_trainer.conditioning.h3_geometry import (
    h3_latent_frames,
    h3_packed_seq_len,
    h3_worst_case_packed_seq_len,
    max_packed_rows_for_budget,
    resolve_canvas_size,
)
from signet_trainer.config.schema import SignetConfig
from signet_trainer.dryrun.shapes import (
    assert_h3_seq_len_budget,
    build_dryrun_inputs,
    build_h3_dryrun_batch,
    run_dryrun,
)
from signet_trainer.train.h3_step import (
    H3_AUDIO_TAG,
    H3_PACKED_BATCH_KEYS,
    H3_TEXT_TAG,
    H3_VIDEO_TAG,
    h3_layout_row_counts,
)

SHAPES_PY = (
    Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "dryrun" / "shapes.py"
)

# The operator's real Phase-10 reference corpus AFTER D-10-CROP (crop never pad; the crop must save
# the face). Labels are load-bearing — they are what a budget refusal names, and "some pair is over
# budget" is not actionable at 3am.
CHARACTER_REFS = [
    (832, 1248, "A"),  # IMG_3659
    (2048, 2048, "B"),  # IMG_3725
    (1024, 1536, "C"),  # tmppvclbw6u, 1037 -> 1024 wide
]
ENVIRONMENT_REFS = [
    (1344, 768, "029"),
    (1024, 1024, "000"),
    (1440, 800, "008"),  # 1456x816 -> 1440x800
    (1344, 768, "023"),
]

# Derived from h3_geometry, NEVER hardcoded (the 10-06 precedent): the gate's own numbers must be
# reproducible from the geometry module or they are not a budget, they are folklore.
CEILING_ROWS = max_packed_rows_for_budget(79.25, 62.97, 1.21)
WORST_896, WORST_LABEL_896 = h3_worst_case_packed_seq_len(
    22, (16, 9), CHARACTER_REFS, ENVIRONMENT_REFS, 96, 896
)
WORST_1024, WORST_LABEL_1024 = h3_worst_case_packed_seq_len(
    22, (16, 9), CHARACTER_REFS, ENVIRONMENT_REFS, 96, 1024
)
# The pair a nominal-pair gate would have priced — UNDER the ceiling at 1024, which is the whole
# reason this gate must enumerate the domain.
NOMINAL_1024 = h3_packed_seq_len(22, (16, 9), CHARACTER_REFS[:2], 96, 1024).total


def _h3_payload(**over) -> dict:
    """A minimal VALID H3 config: 22 frames (17n+5), 1344x768 canvas, the full reference set."""
    payload = {
        "training_dims": [1344, 768, 22],
        "data": {
            "preprocessed_data_root": "/data/h3_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x22"],
        },
        "model": {"family": "h3"},
        "training": {"max_steps": 100},
        "h3": {
            "character_reference_sizes": list(CHARACTER_REFS),
            "environment_reference_sizes": list(ENVIRONMENT_REFS),
        },
        # #20 (checklist item 1): the render triple is now family-validated at load — the shared
        # default frame_count=49 is an LTX clip length and is OUTSIDE the H3 renderable band.
        "validation": {"frame_count": 124},
    }
    payload.update(over)
    return payload


def _h3_cfg(**over) -> SignetConfig:
    return SignetConfig(**_h3_payload(**over))


def _ltx_cfg(**over) -> SignetConfig:
    """The byte-identical-load control for every H3 change."""
    payload = {
        "training_dims": [768, 512, 49],
        "data": {"preprocessed_data_root": "/data/preprocessed", "batch_size": 1},
        "training": {"max_steps": 100},
    }
    payload.update(over)
    return SignetConfig(**payload)


# ======================================================================================
# Task 1 — the H3 branch of build_dryrun_inputs + the H3 contract assertions
# ======================================================================================


def test_h3_branch_prices_the_worst_pair_not_a_nominal_one() -> None:
    """``ref_seq_len`` is the WORST pair's reference rows — the most expensive shape servable.

    A dry-run that synthesises the CHEAPEST (or a nominal) pair exercises a batch the dataloader will
    beat on the first environment-bearing segment, which is exactly the P10-1b OOM class.
    """
    mi = build_dryrun_inputs(_h3_cfg())
    assert mi.ref_seq_len == WORST_896.n_cond_video
    nominal = h3_packed_seq_len(22, (16, 9), CHARACTER_REFS[:2], 96, 896)
    assert mi.ref_seq_len != nominal.n_cond_video, (
        "the synthetic batch must be built from the WORST reference pair, not the nominal A+B pair"
    )


def test_h3_reference_prefix_is_out_of_the_loss() -> None:
    """THE L-3 / dead-adapter CPU proof: no loss ever lands on the reference rows."""
    mi = build_dryrun_inputs(_h3_cfg())
    assert mi.ref_seq_len > 0
    assert not mi.video_loss_mask[:, : mi.ref_seq_len].any(), (
        "reference-prefix loss mask must be ALL False — loss on the reference region trains a dead "
        "adapter"
    )
    assert bool(mi.video_loss_mask[:, mi.ref_seq_len :].all()), (
        "every TARGET video row must be IN the loss"
    )


def test_h3_video_targets_are_target_only() -> None:
    """``video_targets`` is TARGET-length, NOT the combined length — a missing slice fails LOUD."""
    mi = build_dryrun_inputs(_h3_cfg())
    assert tuple(mi.video_targets.shape)[1] == WORST_896.n_target_video
    assert mi.video_loss_mask.shape[1] == WORST_896.n_cond_video + WORST_896.n_target_video


def test_h3_audio_is_present_noised_and_out_of_the_loss() -> None:
    """D-10-AUDIO: video-only means loss-MASKING, never architecture-skipping."""
    cfg = _h3_cfg()
    assert cfg.h3.audio_in_loss is False
    mi = build_dryrun_inputs(cfg)
    assert mi.audio is not None, "the H3 audio span must be PRESENT even when out of the loss"
    assert mi.audio.latent.shape[1] == WORST_896.n_cond_audio + WORST_896.n_target_audio > 0
    assert mi.audio_loss_mask is not None
    assert not mi.audio_loss_mask.any(), "audio_in_loss=False -> the audio mask is all-False"
    assert bool((mi.audio.latent != 0).any()), (
        "target audio rows must be NOISED, not silent — dropping/zeroing them changes the sequence "
        "the video rows attend to"
    )


def test_h3_batch_is_built_by_the_training_packer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dry-run and training MUST share one packer, or the gate can pass while training diverges."""
    import signet_trainer.dryrun.shapes as shapes

    calls: list[int] = []
    real = shapes.build_h3_packed_batch

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(shapes, "build_h3_packed_batch", spy)
    build_dryrun_inputs(_h3_cfg())
    assert calls, "build_dryrun_inputs must call train.h3_step.build_h3_packed_batch"


def test_h3_shapes_source_imports_the_training_packer() -> None:
    """key_link: the import is the structural proof of the shared code path."""
    source = SHAPES_PY.read_text(encoding="utf-8")
    assert "from signet_trainer.train.h3_step import" in source


def test_h3_row_counts_come_from_the_priced_layout() -> None:
    """The batch's row counts are RE-PROJECTED from the priced layout, never re-derived here."""
    batch = build_h3_dryrun_batch(_h3_cfg())
    priced = h3_layout_row_counts(WORST_896)
    assert batch.n_text == priced["n_text"]
    assert batch.n_cond_video == priced["n_cond_video"]
    assert batch.n_target_video == priced["n_target_video"]
    assert batch.n_cond_audio == priced["n_cond_audio"]
    assert batch.n_target_audio == priced["n_target_audio"]
    assert batch.seq_len == priced["seq_len"] == WORST_896.total


def test_h3_packed_batch_keys_are_the_closed_ten() -> None:
    batch = build_h3_dryrun_batch(_h3_cfg())
    assert set(batch.kwargs) == set(H3_PACKED_BATCH_KEYS)


def test_h3_token_tags_take_only_the_three_contract_values() -> None:
    """A tag outside {0,1,2} indexes the AdaLN table off the end of its modality stride."""
    batch = build_h3_dryrun_batch(_h3_cfg())
    tags = batch.kwargs["token_tags"]
    assert set(int(v) for v in torch.unique(tags)) <= {
        H3_VIDEO_TAG,
        H3_TEXT_TAG,
        H3_AUDIO_TAG,
    }
    # All three modalities must actually be present — an all-text tag vector would trivially satisfy
    # the subset check while proving nothing.
    assert set(int(v) for v in torch.unique(tags)) == {H3_VIDEO_TAG, H3_TEXT_TAG, H3_AUDIO_TAG}


def test_h3_index_tensors_partition_the_sequence_exactly_once() -> None:
    batch = build_h3_dryrun_batch(_h3_cfg())
    joined = torch.cat(
        [
            batch.kwargs["video_indices"],
            batch.kwargs["audio_indices"],
            batch.kwargs["text_indices"],
        ]
    )
    assert torch.equal(torch.sort(joined).values, torch.arange(batch.seq_len))


def test_h3_timestep_indices_reconstruct_the_row_timesteps() -> None:
    """The transformer indexes its AdaLN table off ``timestep_indices`` — the round-trip must hold."""
    cfg = _h3_cfg()
    mi = build_dryrun_inputs(cfg)
    batch = build_h3_dryrun_batch(cfg)
    reconstructed = batch.kwargs["timestep"][batch.kwargs["timestep_indices"]]
    assert torch.equal(reconstructed, mi.video.timesteps[0]), (
        "timestep[timestep_indices] must reconstruct the per-row timesteps exactly"
    )


def test_h3_reference_rows_are_pinned_clean() -> None:
    """Visual conditioning rows pin at ``max(t_video, t_visual_cond)`` — never the target's t."""
    cfg = _h3_cfg()
    mi = build_dryrun_inputs(cfg)
    batch = build_h3_dryrun_batch(cfg)
    ref_rows = batch.kwargs["video_indices"][: batch.n_cond_video]
    target_rows = batch.kwargs["video_indices"][batch.n_cond_video :]
    row_t = mi.video.timesteps[0]
    assert torch.unique(row_t[ref_rows]).numel() == 1
    assert float(row_t[ref_rows][0]) > float(row_t[target_rows][0]), (
        "reference rows must sit CLEANER than the target rows (H3 inverts the convention: t=1 is "
        "clean)"
    )


def test_h3_canvas_and_latent_frames_come_from_the_config() -> None:
    """The canvas is resolved from ``h3.target_aspect``; frames from ``training_dims[2]``."""
    cfg = _h3_cfg()
    height, width = resolve_canvas_size(*cfg.h3.target_aspect)
    assert (height, width) == (768, 1344)
    assert h3_latent_frames(cfg.training_dims[2]) == 7


def test_h3_dryrun_is_cpu_pure() -> None:
    mi = build_dryrun_inputs(_h3_cfg())
    assert mi.video.latent.device.type == "cpu"
    assert mi.audio is not None and mi.audio.latent.device.type == "cpu"


def test_h3_dryrun_returns_zero_at_the_phase10_short_edge() -> None:
    assert run_dryrun(_h3_cfg()) == 0


# ======================================================================================
# Task 2 — the packed-seq_len budget refusal and the H3 OK banner
# ======================================================================================


def _over_budget_cfg() -> SignetConfig:
    """A config whose worst pair busts the ceiling.

    Built at the Phase-10 short edge (which LOADS) and then raised to 1024 by assignment: the
    config-load check already refuses 1024 outright, so a dict at 1024 cannot be constructed. That
    is precisely the drift this gate is the second line of defence against — a cfg object that was
    valid when loaded and is not valid any more.
    """
    cfg = _h3_cfg()
    cfg.h3.reference_image_short_edge = 1024
    return cfg


def test_budget_accepts_the_worst_pair_at_the_phase10_short_edge() -> None:
    budget = assert_h3_seq_len_budget(_h3_cfg())
    assert budget.layout.total == WORST_896.total
    assert budget.worst_pair_label == WORST_LABEL_896 == "C+008"
    assert budget.ceiling_rows == CEILING_ROWS
    assert budget.headroom_rows == CEILING_ROWS - WORST_896.total > 0


def test_budget_refuses_the_worst_pair_one_short_edge_up() -> None:
    """The refusal must be a single, complete, actionable message."""
    with pytest.raises(ValueError) as excinfo:
        assert_h3_seq_len_budget(_over_budget_cfg())
    message = str(excinfo.value)

    # BOTH sides of the comparison, the overage, and the offending PAIR.
    reported = re.search(r"reference pair C\+008: (\d+) rows exceeds", message)
    assert reported is not None, message
    assert int(reported.group(1)) == WORST_1024.total
    assert str(CEILING_ROWS) in message
    assert str(WORST_1024.total - CEILING_ROWS) in message

    # The two levers that actually move.
    assert "target frame count" in message
    assert "reference_image_short_edge" in message

    # Trimming references does NOT rescue campaign length — an H200-class decision, not a geometry
    # one. The number is COMPUTED by the validator, so derive the expectation the same way.
    t2v_baseline = h3_packed_seq_len(124, (16, 9), [], 96, 896).total
    assert str(t2v_baseline) in message


def test_the_nominal_pair_would_have_passed_and_the_gate_still_refuses() -> None:
    """THE named regression: pricing one nominal pair passes, then OOMs on the first costlier one."""
    assert NOMINAL_1024 < CEILING_ROWS, (
        "premise: the nominal A+B pair at short edge 1024 is UNDER the ceiling, so a nominal-pair "
        "gate would let this config through"
    )
    assert WORST_1024.total > CEILING_ROWS
    assert run_dryrun(_over_budget_cfg()) != 0


def test_run_dryrun_refuses_over_budget_and_names_both_numbers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_dryrun(_over_budget_cfg()) != 0
    captured = capsys.readouterr()
    assert str(WORST_1024.total) in captured.err
    assert "C+008" in captured.err
    assert str(CEILING_ROWS) in captured.err


def test_run_dryrun_refuses_before_building_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is a PREFLIGHT — it must not depend on allocating the over-budget batch."""
    import signet_trainer.dryrun.shapes as shapes

    calls: list[int] = []
    real = shapes.build_h3_packed_batch

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(shapes, "build_h3_packed_batch", spy)
    assert run_dryrun(_over_budget_cfg()) != 0
    assert not calls, "the budget refusal must fire BEFORE any packed batch is assembled"


def test_h3_ok_banner_reports_the_worst_pair_length_and_headroom(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run_dryrun(_h3_cfg()) == 0
    out = capsys.readouterr().out
    assert "family=h3" in out
    assert "1344x768" in out  # the resolved canvas
    assert "22 target frames" in out
    assert f"{h3_latent_frames(22)} latent frames" in out
    assert WORST_896.describe() in out  # the full packed breakdown
    assert "C+008" in out  # the PAIR label — what makes a near-ceiling number actionable
    assert f"packed seq_len={WORST_896.total}" in out
    assert f"ceiling={CEILING_ROWS}" in out
    assert f"headroom {CEILING_ROWS - WORST_896.total} rows" in out
    assert "Zero GPU, zero Modal spend." in out


def test_reference_free_h3_config_is_still_budget_checked() -> None:
    """A t2v-shaped H3 config busts the ceiling on target length alone — price it, do not skip it."""
    ok = _h3_cfg(h3={})
    assert assert_h3_seq_len_budget(ok).worst_pair_label == ""
    assert run_dryrun(ok) == 0

    # Campaign target length busts the ceiling with ZERO references, so trimming the reference set
    # cannot rescue it. Reached by assignment for the same reason as ``_over_budget_cfg``: config
    # load already refuses it outright, and this gate is the second line of defence against drift.
    campaign = _h3_cfg(h3={})
    campaign.training_dims = [1344, 768, 124]
    with pytest.raises(ValueError) as excinfo:
        assert_h3_seq_len_budget(campaign)
    assert str(h3_packed_seq_len(124, (16, 9), [], 96, 896).total) in str(excinfo.value)


def test_budget_gate_delegates_to_the_shared_worst_pair_validator() -> None:
    """key_link: the refusal message lives in ONE place — ``validate_h3_reference_budget``."""
    source = SHAPES_PY.read_text(encoding="utf-8")
    assert "validate_h3_reference_budget" in source


# ======================================================================================
# Task 3 (house audit, PR #51) — the 3-reference explicit-manifest budget pin
# ======================================================================================

#: Three IDENTICAL 1344x768 references — the explicit-manifest shape (every row supplies its OWN
#: ``reference_paths``, so there is no rotation domain and ``h3_worst_case_packed_seq_len`` enumerates
#: exactly one combination, C(3,3)=1). ``prompt_tokens_estimate=200`` matches the PRICING TABLE the
#: house audit verified reproduces exactly against this repo's own functions — see the comment beside
#: ``H3Config.references_per_sample`` in ``schema.py`` (the field's own nominal default of 96 prices
#: the same shape at 13,298, not 13,402).
_THREE_REF_SIZES = [(1344, 768, "r0"), (1344, 768, "r1"), (1344, 768, "r2")]


def _h3_three_ref_payload(short_edge: int, **over) -> dict:
    payload = {
        "training_dims": [1344, 768, 22],
        "data": {
            "preprocessed_data_root": "/data/h3_preprocessed",
            "batch_size": 1,
            "resolution_buckets": ["1344x768x22"],
        },
        "model": {"family": "h3"},
        "training": {"max_steps": 100},
        # The 3-ref campaign renders AT its trained length (22 frames, off the 5-15s band), so the
        # real config carries the waiver — without it the schema-level renderable-band guard (#20
        # item 1, landed after this fixture was written) refuses the inherited frame_count default.
        "validation": {"allow_offband_frame_count": True},
        "h3": {
            "references_per_sample": 3,
            "character_reference_sizes": list(_THREE_REF_SIZES),
            "prompt_tokens_estimate": 200,
            "reference_image_short_edge": short_edge,
        },
    }
    payload.update(over)
    return payload


def test_three_reference_config_loads_at_short_edge_768() -> None:
    """PIN: a 3-reference explicit-manifest config at short_edge 768 prices at 13,402 packed rows
    (prompt_tokens=200) against a computed ceiling of 13,777 — under, so config LOAD must succeed.
    Mirrors the 2-slot precedent (``test_budget_accepts_the_worst_pair_at_the_phase10_short_edge``).
    """
    layout = h3_packed_seq_len(22, (16, 9), _THREE_REF_SIZES, 200, 768)
    assert layout.total == 13402
    assert layout.total < CEILING_ROWS

    cfg = SignetConfig(**_h3_three_ref_payload(768))
    assert cfg.h3.references_per_sample == 3
    assert cfg.h3.reference_image_short_edge == 768


def test_three_reference_config_refused_at_short_edge_896() -> None:
    """Same 3-reference corpus at short_edge 896 prices at 15,586 rows — OVER the ceiling, so it
    must be refused at config LOAD (not merely at the separate dry-run gate): the mismatch between
    ``references_per_sample``'s allowlist and the packed-sequence budget is exactly what a 3-slot
    campaign that skips straight to ``--mode sample`` would otherwise discover on a metered A100.
    """
    layout = h3_packed_seq_len(22, (16, 9), _THREE_REF_SIZES, 200, 896)
    assert layout.total == 15586
    assert layout.total > CEILING_ROWS

    with pytest.raises(ValueError, match="exceeds the declared ceiling"):
        SignetConfig(**_h3_three_ref_payload(896))


# --------------------------------------------------------------------------------------------------
# LTX regression controls — the LTX dry-run path must be untouched.
# --------------------------------------------------------------------------------------------------


def test_ltx_ok_banner_is_pinned(capsys: pytest.CaptureFixture[str]) -> None:
    """The LTX banner is pinned — no H3/qwen_edit field leaks in, and the worst BUCKET is priced.

    training_dims [768, 512, 49] -> 2688, but the default buckets 768x352x{25,49,81} are what
    training actually shapes (gap-dryrun-ltx-1): worst = 768x352x81 -> 24*11*11 = 2904.
    """
    assert run_dryrun(_ltx_cfg()) == 0
    out = capsys.readouterr().out
    assert out == (
        "[signet-dryrun] OK — config valid, synthetic ModelInputs built: "
        "dims [W=768, H=512, F=49] -> seq_len=2688, "
        "video.latent (1, 2688, 128); worst bucket "
        "[W=768, H=352, F=81] -> seq_len=2904, "
        "video.latent (1, 2904, 128) (3 bucket(s) priced). Zero GPU, zero Modal spend.\n"
    )


def test_ltx_dryrun_inputs_are_unchanged() -> None:
    cfg = _ltx_cfg()
    assert cfg.model.family == "ltx"
    mi = build_dryrun_inputs(cfg)
    # [768, 512, 49] -> (512//32)*(768//32)*((49-1)//8+1) = 16*24*7 = 2688
    assert tuple(mi.video.latent.shape) == (1, 2688, 128)
    assert mi.ref_seq_len is None
    assert mi.audio is None


def test_ltx_dryrun_still_returns_zero() -> None:
    assert run_dryrun(_ltx_cfg()) == 0


def test_h3_and_ltx_paths_do_not_share_row_math() -> None:
    """A sanity control: the H3 packed length is nothing like the LTX seq_len for the same dims."""
    ltx_like = build_dryrun_inputs(_ltx_cfg())
    h3_like = build_dryrun_inputs(_h3_cfg())
    assert ltx_like.video.latent.shape[1] != h3_like.video.latent.shape[1]


def test_no_budget_literal_in_the_gate() -> None:
    """Every budget number arrives from config + ``h3_geometry`` — never a literal in the gate."""
    source = SHAPES_PY.read_text(encoding="utf-8")
    forbidden = re.compile(r"1\.21|13[,]?7[0-9][0-9]|12394|14026")
    assert not forbidden.search(source), (
        f"budget literal(s) found in {SHAPES_PY.name}: {forbidden.findall(source)}"
    )
