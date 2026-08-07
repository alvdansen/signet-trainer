"""LoRA-target probe + cross-modal-target proof for a2v (Phase 9, GATE-SPEC rev 2 item 3).

The ``check_lora_targets`` per-target instance-count probe makes VISIBLE that the short-suffix video
defaults COLLATERALLY match the ``audio_transformer_blocks`` attn modules (which affects video-only
runs too), and that the a2v cross-modal ``audio_to_video_attn.*`` targets actually hit modules.
Also asserts the two canonical copies of the cross-modal target list (config.validators + lora.peft)
stay byte-identical (single source of truth enforced by test, not a cross-import).
"""

from __future__ import annotations

from signet_trainer.config.validators import A2V_CROSS_MODAL_ATTN_TARGETS
from signet_trainer.lora.peft import (
    A2V_CROSS_MODAL_LORA_TARGETS,
    P1_A2V_LORA_TARGETS,
    P1_FF_LORA_TARGETS,
    check_lora_targets,
)

# A synthetic module-name list mirroring the LTX-2.3 dual-stream naming: a video transformer block
# stack + an AUDIO dual-stream stack (audio_transformer_blocks) carrying its own attn projections and
# the cross-modal audio->video attention (audio_to_video_attn).
_SYNTHETIC_NAMES = [
    # 2 video blocks
    "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.attn1.to_k",
    "transformer_blocks.0.attn1.to_v",
    "transformer_blocks.0.attn1.to_out.0",
    "transformer_blocks.0.ff.net.0.proj",
    "transformer_blocks.0.ff.net.2",
    "transformer_blocks.1.attn1.to_q",
    "transformer_blocks.1.attn1.to_k",
    "transformer_blocks.1.attn1.to_v",
    "transformer_blocks.1.attn1.to_out.0",
    "transformer_blocks.1.ff.net.0.proj",
    "transformer_blocks.1.ff.net.2",
    # 2 audio blocks (collateral matches for the short-suffix video targets) + cross-modal attn
    "audio_transformer_blocks.0.attn1.to_q",
    "audio_transformer_blocks.0.attn1.to_k",
    "audio_transformer_blocks.0.attn1.to_v",
    "audio_transformer_blocks.0.attn1.to_out.0",
    "audio_transformer_blocks.0.audio_to_video_attn.to_q",
    "audio_transformer_blocks.0.audio_to_video_attn.to_k",
    "audio_transformer_blocks.0.audio_to_video_attn.to_v",
    "audio_transformer_blocks.0.audio_to_video_attn.to_out.0",
    "audio_transformer_blocks.1.attn1.to_q",
    "audio_transformer_blocks.1.attn1.to_k",
    "audio_transformer_blocks.1.attn1.to_v",
    "audio_transformer_blocks.1.attn1.to_out.0",
    "audio_transformer_blocks.1.audio_to_video_attn.to_q",
    "audio_transformer_blocks.1.audio_to_video_attn.to_k",
    "audio_transformer_blocks.1.audio_to_video_attn.to_v",
    "audio_transformer_blocks.1.audio_to_video_attn.to_out.0",
]


def test_cross_modal_target_lists_are_in_sync() -> None:
    # The config-layer fail-fast copy and the peft-layer inject copy must be byte-identical.
    assert list(A2V_CROSS_MODAL_ATTN_TARGETS) == list(A2V_CROSS_MODAL_LORA_TARGETS)


def test_a2v_target_set_is_video_plus_cross_modal() -> None:
    assert P1_A2V_LORA_TARGETS == P1_FF_LORA_TARGETS + A2V_CROSS_MODAL_LORA_TARGETS


def test_short_suffix_video_targets_collaterally_match_audio_branch() -> None:
    # The #1 finding: attn1.to_q (a video default) ALSO matches audio_transformer_blocks — surfaced,
    # not silent. This affects video-only runs too.
    report = check_lora_targets(_SYNTHETIC_NAMES, ["attn1.to_q"])
    r = report["attn1.to_q"]
    assert r["video"] == 2  # the 2 video blocks
    assert r["audio"] == 2  # collateral: the 2 audio blocks
    assert r["total"] == 4
    assert any("audio_transformer_blocks" in n for n in r["audio_names"])


def test_ff_targets_do_not_collaterally_match_audio() -> None:
    # ff.net.* is video-only in the synthetic stack — no audio collateral.
    report = check_lora_targets(_SYNTHETIC_NAMES, ["ff.net.0.proj", "ff.net.2"])
    for t in ("ff.net.0.proj", "ff.net.2"):
        assert report[t]["audio"] == 0
        assert report[t]["video"] == 2


def test_cross_modal_targets_hit_audio_to_video_attn() -> None:
    # The a2v cross-modal targets MUST match modules (a zero count is "the #1 silent a2v failure").
    report = check_lora_targets(_SYNTHETIC_NAMES, list(A2V_CROSS_MODAL_LORA_TARGETS))
    for t in A2V_CROSS_MODAL_LORA_TARGETS:
        assert report[t]["total"] == 2, f"{t} matched {report[t]['total']} modules (expected 2)"
        # they live on the audio branch.
        assert report[t]["audio"] == 2


def test_video_default_set_reports_full_collateral_footprint() -> None:
    # A full video-default probe: every short attn suffix collaterally hits the audio branch,
    # while ff.net.* stays video-only — the visible footprint the probe is for.
    report = check_lora_targets(_SYNTHETIC_NAMES, P1_FF_LORA_TARGETS)
    attn_audio = sum(
        report[t]["audio"] for t in P1_FF_LORA_TARGETS if t.startswith("attn")
    )
    ff_audio = sum(report[t]["audio"] for t in P1_FF_LORA_TARGETS if t.startswith("ff"))
    assert attn_audio > 0  # attn suffixes collaterally hit the audio branch
    assert ff_audio == 0  # ff.net stays video-only
