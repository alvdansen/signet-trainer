"""The CPU dry-run hard gate (CONF-03, D-11/D-12/D-13).

``main(argv)`` is the ``signet-dryrun`` console-script target (wired in pyproject.toml) and the
Phase-8-harness ``dryrun`` subcommand seam. It:

  1. parses the config path arg and ``load_config(path)`` — this fires all CONF-02 validators,
     so a bad frame count exits non-zero HERE with a clear message (D-13 / SC#1);
  2. computes ``seq_len`` from ``training_dims`` via the shared helper;
  3. builds a synthetic ``ModelInputs``/``Modality`` on CPU (``torch.zeros``/``randn`` — no real
     data, no weights, no VAE — D-12) mirroring the REAL contract (RESEARCH.md Q4);
  4. asserts every shape/dtype AND the mask invariants;
  5. returns 0 on success, raises/returns non-zero on ANY violation.

CRITICAL — Pitfall 2: assert the REAL contract (``video.latent [1,seq,128]``,
``video.positions [1,3,seq,2]``, ``video_loss_mask == ~cond_mask``), NOT a flat 2-D
``video_coords``. Pitfall 4 / Windows: no ``modal`` / ``ltx_core`` / CUDA import — tensors are
CPU-only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import torch

from signet_trainer.conditioning.h3_geometry import (
    H3PackedLayout,
    H3Reference,
    h3_latent_frames,
    h3_packed_seq_len,
    h3_reference_pairing_domain,
    h3_worst_case_packed_seq_len,
    max_packed_rows_for_budget,
    reference_image_size,
    resolve_canvas_size,
    rows_of,
)
from signet_trainer.conditioning.mask_ops import (
    build_denoise_mask,
    latent_frame_token_span,
    per_token_sigma,
)
from signet_trainer.conditioning.qwen_edit import (
    QwenControlSlot,
    QwenEditModelInputs,
    resolve_control_slots,
)
from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_FRAMES,
    QWEN_EDIT_LATENT_CHANNELS,
    QWEN_EDIT_PATCH,
    QWEN_EDIT_PATCH_DIM,
    QwenEditPackedLayout,
    qwen_edit_area_budget_size,
    qwen_edit_latent_size,
    qwen_edit_packed_layout,
    qwen_edit_rows_of,
    qwen_edit_vae_latent_shape,
)
from signet_trainer.conditioning.strategy import (
    HEIGHT_SCALE,
    TIME_SCALE,
    VIDEO_LATENT_CHANNELS,
    WIDTH_SCALE,
    Modality,
    ModelInputs,
    compute_seq_len,
)
from signet_trainer.config.load import load_config
from signet_trainer.config.schema import SignetConfig
from signet_trainer.config.validators import (
    validate_h3_reference_budget,
    validate_h3_seq_len_budget,
    validate_qwen_edit_row_budget,
)
from signet_trainer.models.h3_loader import (
    EXPECTED_H3_AUDIO_IN_CHANNELS,
    EXPECTED_H3_IN_CHANNELS,
    EXPECTED_H3_PATCH_SIZE,
    EXPECTED_H3_TEXT_DIM,
)
from signet_trainer.train.h3_step import (
    H3_AUDIO_TAG,
    H3_TEXT_TAG,
    H3_VIDEO_TAG,
    H3PackedBatch,
    build_h3_packed_batch,
    h3_layout_row_counts,
)

#: The synthetic dry-run target sigma (WR-02: a NON-ZERO constant so timestep 0 stays the reserved
#: conditioning sentinel for the single_frame/none reconstruction). The multi_frame branch reuses it
#: for both the forward per-token noising AND the ``denoise_mask = timesteps / _SAMPLED_SIGMA``
#: reconstruction in ``_assert_contract`` — one source of truth so the two can never diverge.
_SAMPLED_SIGMA = 0.5


#: Inpaint dry-run clean/noisy latent sentinels — DISTINCT constants so the contract assert can
#: verify the ``where(mask>0.5, clean, noisy)`` substitution polarity on the latent itself
#: (all-zeros would hide a swapped branch).
_INPAINT_CLEAN = -1.0
_INPAINT_NOISY = 1.0

#: a2v (audio-to-video) synthetic-audio dims (GATE-SPEC rev 2 item 8). The audio token dim is
#: C*mel = 8*16 = 128 ([canonical] ltx-trainer base_strategy @ pinned SHA d6053703: audio latents
#: patchify to ``[B, T, C*F] = [B, T, 128]``). ``_A2V_AUDIO_TIME_STEPS`` is a small deterministic
#: synthetic T (real T rides the audio VAE compression of each clip's driving audio — the CPU gate
#: only needs a shape-correct frozen-audio slot, not real dims).
_A2V_AUDIO_TIME_STEPS = 8
_A2V_AUDIO_TOKEN_DIM = 128


# --------------------------------------------------------------------------------------------------
# MiniMax-H3 (Phase 10, H3-04 / H3-07) — the packed-sequence branch of the gate.
#
# Two jobs no LTX config ever needed: prove on CPU that the reference prefix is out of the loss (the
# dead-adapter bug can never reach a paid container), and REFUSE a dispatch whose computed packed
# sequence exceeds the target GPU's measured activation ceiling. Every budget number arrives from
# ``cfg.h3`` and ``conditioning/h3_geometry`` — this module owns NO budget arithmetic and carries no
# row-count or ceiling literal.
# --------------------------------------------------------------------------------------------------

#: The two synthetic H3 timesteps. ⚠ H3 INVERTS the usual flow-match convention: ``t = 1 - sigma``,
#: so ``t = 1`` is CLEAN (``scheduling_minimax_h3.py:166-167``). Video reuses this file's single
#: sampled-sigma source of truth; audio takes a DISTINCT value so the audio span is identifiable in
#: the ``timestep`` / ``timestep_indices`` round-trip (the two modalities run separate schedules,
#: shift 12.0 vs 3.0). Both are strictly below ``H3_VISUAL_CONDITION_PIN``, so the reference rows are
#: provably pinned CLEANER than the target rows rather than tying with them.
_H3_T_VIDEO = _SAMPLED_SIGMA
_H3_T_AUDIO = 0.25

#: Target-audio sentinel — a NON-ZERO constant, deliberately. D-10-AUDIO: with ``audio_in_loss``
#: False the target audio rows stay PRESENT and NOISED and merely stay OUT of the loss. Zeroing them
#: would make the gate assert against silence, which is a different (and wrong) training regime —
#: dropping or muting the rows changes the sequence the video rows attend to.
_H3_NOISED_AUDIO = 1.0

#: ``<|vision_start|>`` + ``<|vision_end|>``. A reference's vision span is tagged VIDEO **including**
#: both sentinel positions — deliberate, not an off-by-one (``P10-0e-DIFFUSERS-H3.md`` section 1, and
#: ``train/h3_step.h3_token_tags``'s own contract). Named here for the same reason
#: ``_A2V_AUDIO_TOKEN_DIM`` is: it is a cited arch fact, not geometry the gate may re-derive. The
#: per-reference LABEL block size is NOT restated — it is derived from the priced layout below.
_H3_VISION_SENTINELS = 2


@dataclass(frozen=True)
class H3DryrunBudget:
    """What the H3 preflight priced: the WORST reference pair, its label, and the row ceiling.

    ``layout`` is the ``H3PackedLayout`` of the MOST EXPENSIVE pair in the real pairing domain — never
    a nominal one. ``references`` is that pair itself, so the synthetic batch exercises the most
    expensive shape the dataloader can serve rather than the cheapest. ``worst_pair_label`` is what a
    refusal (and the OK banner) names; it is ``""`` only when no reference corpus is declared.
    """

    layout: H3PackedLayout
    worst_pair_label: str
    references: tuple[H3Reference, ...]
    ceiling_rows: int

    @property
    def headroom_rows(self) -> int:
        """Rows still available under the declared ceiling. Negative means over budget."""
        return self.ceiling_rows - self.layout.total


def _h3_patch_dim() -> int:
    """``in_channels * prod(patch_size)`` — the H3 video feature width, from the arch constants."""
    return EXPECTED_H3_IN_CHANNELS * prod(EXPECTED_H3_PATCH_SIZE)


def _h3_pair_for_label(cfg: SignetConfig, label: str) -> tuple[H3Reference, ...]:
    """Recover the reference tuple behind a worst-case pair LABEL (e.g. ``C+008``).

    ``h3_worst_case_packed_seq_len`` returns the winning layout and its label but not the pair, and
    the synthetic batch needs the pair's per-reference sizes to place the vision spans. Re-enumerate
    the SAME domain (``h3_geometry`` owns it) and take the LAST match, mirroring that function's
    documented tie-break — character refs A and C are both 2:3 and encode identically, so every
    ``A+x`` pair ties exactly with its ``C+x`` counterpart and last-wins keeps the answer aligned.
    """
    domain = h3_reference_pairing_domain(
        cfg.h3.character_reference_sizes,
        cfg.h3.environment_reference_sizes,
        cfg.h3.references_per_sample,
    )
    matches = [pair for pair_label, pair in domain if pair_label == label]
    if not matches:
        raise ValueError(
            f"worst-case reference pair {label!r} is not in the enumerated pairing domain "
            f"({[lbl for lbl, _ in domain]}). The reference corpus changed between pricing and "
            f"batch build."
        )
    return matches[-1]


def _h3_worst_case_budget(cfg: SignetConfig) -> H3DryrunBudget:
    """Price the WORST reference pair and the row ceiling. PRICING ONLY — raises no refusal.

    ⛔ Never prices one nominal pair. The 15 real pairs differ in row cost by up to 12%, so a nominal
    price passes at config load and then OOMs on the first costlier segment — the exact failure
    H3-04 exists to prevent. The refusal itself lives in ``assert_h3_seq_len_budget``, which delegates
    to ``config/validators.validate_h3_reference_budget`` so the message exists in exactly one place.

    CPU-pure: geometry ints only, no ``torch``, no ``ltx_core``, no ``modal``.
    """
    h3 = cfg.h3
    target_frames = cfg.training_dims[2]
    ceiling = max_packed_rows_for_budget(h3.gpu_usable_gib, h3.resident_gib, h3.mib_per_packed_row)

    if h3.character_reference_sizes or h3.environment_reference_sizes:
        layout, label = h3_worst_case_packed_seq_len(
            target_frames,
            h3.target_aspect,
            h3.character_reference_sizes,
            h3.environment_reference_sizes,
            h3.prompt_tokens_estimate,
            h3.reference_image_short_edge,
            references_per_sample=h3.references_per_sample,
        )
        references = _h3_pair_for_label(cfg, label)
    else:
        # No reference corpus declared -> exactly ONE possible layout, so worst == nominal TRIVIALLY.
        # This is NOT the nominal-pair hole above (there is no other pair to be wrong about), and it
        # still must be priced: a campaign-length NO-REFERENCE t2v baseline busts the ceiling with
        # zero references, so leaving this branch unpriced would hand that OOM to a metered A100.
        # Same reasoning, same shape as config/schema.py's reference-free arm.
        layout = h3_packed_seq_len(
            target_frames,
            h3.target_aspect,
            (),
            h3.prompt_tokens_estimate,
            h3.reference_image_short_edge,
        )
        label, references = "", ()

    return H3DryrunBudget(
        layout=layout,
        worst_pair_label=label,
        references=references,
        ceiling_rows=ceiling,
    )


def _h3_vision_spans(
    cfg: SignetConfig, budget: H3DryrunBudget
) -> tuple[tuple[int, int], ...]:
    """Half-open ``(start, stop)`` vision blocks inside the text span, SENTINELS INCLUDED.

    The text stream is ``[ per-ref label block | <|vision_start|> vision rows <|vision_end|> ]*``
    with the prompt LAST (``H3PackedLayout``'s packed order). The per-reference label-block size is
    DERIVED from the priced layout — ``n_text - prompt_tokens - n_vision`` is the whole text-side
    overhead, so the gate never restates ``h3_packed_seq_len``'s own ``+6``/``+2`` arithmetic.
    """
    references = budget.references
    if not references:
        return ()

    layout = budget.layout
    overhead = layout.n_text - cfg.h3.prompt_tokens_estimate - layout.n_vision
    per_ref_overhead, remainder = divmod(overhead, len(references))
    if remainder or per_ref_overhead < _H3_VISION_SENTINELS:
        raise ValueError(
            f"cannot place the H3 vision spans: the priced layout's text overhead ({overhead} rows "
            f"over {len(references)} reference(s)) does not divide into a per-reference label block "
            f"plus {_H3_VISION_SENTINELS} sentinels. conditioning/h3_geometry.h3_packed_seq_len's "
            f"text arithmetic changed — re-derive the spans, do not guess."
        )

    spans: list[tuple[int, int]] = []
    cursor = 0
    label_tokens = per_ref_overhead - _H3_VISION_SENTINELS
    for reference in references:
        height, width = reference_image_size(
            reference.width, reference.height, short_edge=cfg.h3.reference_image_short_edge
        )
        vision_tokens = rows_of(height, width)
        cursor += label_tokens
        spans.append((cursor, cursor + vision_tokens + _H3_VISION_SENTINELS))
        cursor += vision_tokens + _H3_VISION_SENTINELS
    return tuple(spans)


def build_h3_dryrun_batch(
    cfg: SignetConfig, budget: H3DryrunBudget | None = None
) -> H3PackedBatch:
    """Build the synthetic H3 packed batch with ``train/h3_step.build_h3_packed_batch``.

    Deliberately the SAME function the training path calls (T-10-09-T2): a gate that assembles its
    own kwargs can pass while training diverges, which is how a loss-masking bug reaches a metered
    container with every local check green.

    The tensors are ``torch.zeros`` at the REAL row counts and the REAL feature widths, so a shape
    bug still surfaces on CPU; only the target-audio rows carry a non-zero sentinel, because
    D-10-AUDIO's claim is that they are present and NOISED. CPU-pure — no CUDA, no ``ltx_core``,
    no ``modal``.
    """
    budget = _h3_worst_case_budget(cfg) if budget is None else budget
    counts = h3_layout_row_counts(budget.layout)

    patch_dim = _h3_patch_dim()
    n_video = counts["n_cond_video"] + counts["n_target_video"]
    n_audio = counts["n_cond_audio"] + counts["n_target_audio"]

    video_latents = torch.zeros(1, n_video, patch_dim)
    audio_latents = torch.zeros(1, n_audio, EXPECTED_H3_AUDIO_IN_CHANNELS)
    audio_latents[:, counts["n_cond_audio"] :, :] = _H3_NOISED_AUDIO  # present AND noised
    text_embeds = torch.zeros(1, counts["n_text"], EXPECTED_H3_TEXT_DIM)
    # H3 RoPE is 3-axis (t, h, w) in float64, ONE coordinate per PACKED row — not LTX's
    # [B, 3, T, 2] patch-bounds tensor. The real coordinates belong to the strategy that knows the
    # reference geometry; the gate only needs the shape to be right.
    position_ids = torch.zeros(counts["seq_len"], 3, dtype=torch.float64)

    return build_h3_packed_batch(
        video_latents=video_latents,
        audio_latents=audio_latents,
        text_embeds=text_embeds,
        n_cond_video=counts["n_cond_video"],
        n_cond_audio=counts["n_cond_audio"],
        position_ids=position_ids,
        vision_spans=_h3_vision_spans(cfg, budget),
        t_video=_H3_T_VIDEO,
        t_audio=_H3_T_AUDIO,
        audio_in_loss=cfg.h3.audio_in_loss,
        patch_dim=patch_dim,
        audio_in_channels=EXPECTED_H3_AUDIO_IN_CHANNELS,
        text_dim=EXPECTED_H3_TEXT_DIM,
        max_packed_rows=budget.ceiling_rows,
        expected_layout=budget.layout,
    )


def _build_h3_dryrun_inputs(cfg: SignetConfig) -> ModelInputs:
    """Wrap the synthetic H3 packed batch in the ``ModelInputs`` contract (H3 arm of the gate).

    NOTE — the LTX shape conventions do NOT apply on this arm, exactly as they do not on the
    ``ic_lora`` arm: ``video.latent`` is the PACKED ``[1, n_cond_video + n_target_video, patch_dim]``
    stack, ``video.timesteps`` is per-row over the WHOLE packed sequence, and ``video.positions``
    carries H3's ``[seq, 3]`` RoPE coordinates rather than LTX's ``[B, 3, T, 2]`` patch bounds. The
    H3 arm of ``_assert_contract`` asserts the H3 contract and returns before any generic assert.
    """
    batch = build_h3_dryrun_batch(cfg)
    kwargs = batch.kwargs
    # timestep[timestep_indices] IS the per-row timestep vector — reconstruct it rather than carrying
    # a second copy, so the two can never disagree.
    row_t = kwargs["timestep"][kwargs["timestep_indices"]]
    context_mask = torch.ones(1, batch.n_text, dtype=torch.bool)

    video = Modality(
        latent=kwargs["hidden_states"],
        sigma=torch.tensor([_H3_T_VIDEO]),
        timesteps=row_t.unsqueeze(0),
        positions=kwargs["position_ids"],
        context=kwargs["encoder_hidden_states"],
        context_mask=context_mask,
    )
    audio = Modality(
        latent=kwargs["audio_hidden_states"],
        sigma=torch.tensor([_H3_T_AUDIO]),
        timesteps=row_t[kwargs["audio_indices"]].unsqueeze(0),
        positions=kwargs["position_ids"][kwargs["audio_indices"]],
        context=kwargs["encoder_hidden_states"],
        context_mask=context_mask,
    )

    # TARGET-ONLY velocity target (length ``n_target_video``, NOT the packed length): a forgotten
    # ``[:, n_cond_video:]`` slice in the loss then fails LOUD on shape instead of quietly training
    # the adapter on its own reference rows.
    video_targets = torch.randn(1, batch.n_target_video, _h3_patch_dim())
    audio_targets = (
        torch.randn(1, batch.n_target_audio, EXPECTED_H3_AUDIO_IN_CHANNELS)
        if cfg.h3.audio_in_loss
        else None
    )

    return ModelInputs(
        video=video,
        audio=audio,
        video_targets=video_targets,
        audio_targets=audio_targets,
        video_loss_mask=batch.video_loss_mask,
        audio_loss_mask=batch.audio_loss_mask,
        ref_seq_len=batch.n_cond_video,
    )


def _assert_h3_contract(cfg: SignetConfig, mi: ModelInputs) -> None:
    """THE CPU proof for the H3 arm. Raises ``AssertionError`` on any violation.

    (a) the reference-prefix span of ``video_loss_mask`` is ALL False — loss over the reference rows
        is the dead-adapter bug (L-3): the objective gets dominated by rows the model is not being
        asked to generate, and the transformer explicitly hands conditioning rows back UNMASKED
        (``transformer_minimax_h3.py`` L44-50), so masking is the CALLER's job and this is where it
        is proven before any dispatch;
    (b) ``video_targets`` is TARGET-ONLY, so a missing slice fails on shape rather than silently;
    (c) ``token_tags`` take only the three checkpoint-contract values — the AdaLN table is addressed
        ``timestep_indices * H3_MODALITY_NUM + token_tags``, so a stray tag silently modulates the
        wrong rows and NOTHING raises;
    (d) the three index tensors partition ``range(seq_len)`` exactly once each;
    (e) ``timestep[timestep_indices]`` reconstructs the per-row timesteps exactly;
    (f) the audio span is non-empty, noised, and out of the loss when ``audio_in_loss`` is False
        (D-10-AUDIO — the rows are PRESENT, merely not a target).

    The batch is rebuilt here (the build is fully deterministic) and BRIDGED to ``mi`` by equality on
    every tensor the caller can see, so the H3-internal assertions below are provably about the same
    object ``build_dryrun_inputs`` returned.
    """
    budget = _h3_worst_case_budget(cfg)
    batch = build_h3_dryrun_batch(cfg, budget)
    kwargs = batch.kwargs
    layout = budget.layout

    # --- the bridge: the rebuild IS what build_dryrun_inputs returned -----------------------------
    assert mi.ref_seq_len == batch.n_cond_video, (
        f"ModelInputs.ref_seq_len {mi.ref_seq_len} != n_cond_video {batch.n_cond_video}"
    )
    assert torch.equal(mi.video_loss_mask, batch.video_loss_mask), (
        "video_loss_mask does not match the packed batch's own mask"
    )
    assert mi.audio_loss_mask is not None and torch.equal(
        mi.audio_loss_mask, batch.audio_loss_mask
    ), "audio_loss_mask does not match the packed batch's own mask"
    assert mi.video.latent.data_ptr() == kwargs["hidden_states"].data_ptr() or torch.equal(
        mi.video.latent, kwargs["hidden_states"]
    ), "video.latent is not the packed hidden_states"

    # --- the priced layout is what was actually built ----------------------------------------------
    priced = h3_layout_row_counts(layout)
    realized = {
        "n_text": batch.n_text,
        "n_cond_video": batch.n_cond_video,
        "n_target_video": batch.n_target_video,
        "n_cond_audio": batch.n_cond_audio,
        "n_target_audio": batch.n_target_audio,
        "seq_len": batch.seq_len,
    }
    assert priced == realized, f"realized row counts {realized} != priced layout {priced}"

    # --- (a) THE L-3 guard -------------------------------------------------------------------------
    assert mi.ref_seq_len > 0 or not budget.references, (
        "a declared reference corpus must produce a non-empty reference prefix"
    )
    assert not mi.video_loss_mask[:, : batch.n_cond_video].any(), (
        "reference-prefix loss mask must be ALL False across [:n_cond_video] (L-3): loss on the "
        "reference rows trains a dead adapter."
    )
    assert bool(mi.video_loss_mask[:, batch.n_cond_video :].all()), (
        "every TARGET video row must be IN the loss."
    )
    assert int(mi.video_loss_mask.sum()) == batch.n_target_video, (
        f"video loss mask covers {int(mi.video_loss_mask.sum())} rows != n_target_video "
        f"{batch.n_target_video}"
    )

    # --- (b) TARGET-ONLY targets -------------------------------------------------------------------
    assert tuple(mi.video_targets.shape) == (1, batch.n_target_video, _h3_patch_dim()), (
        f"video_targets shape {tuple(mi.video_targets.shape)} != "
        f"(1, {batch.n_target_video}, {_h3_patch_dim()}) — must be TARGET-ONLY, NOT the packed "
        f"length {batch.n_cond_video + batch.n_target_video} (L-3)."
    )

    # --- (c) modality tags -------------------------------------------------------------------------
    tags = kwargs["token_tags"]
    tag_values = {int(v) for v in torch.unique(tags)}
    assert tag_values <= {H3_VIDEO_TAG, H3_TEXT_TAG, H3_AUDIO_TAG}, (
        f"token_tags carry {sorted(tag_values)} — only "
        f"{{{H3_VIDEO_TAG}, {H3_TEXT_TAG}, {H3_AUDIO_TAG}}} address the AdaLN table "
        f"(timestep_indices * H3_MODALITY_NUM + token_tags); a stray tag modulates the wrong rows "
        f"with nothing raising."
    )
    assert tuple(tags.shape) == (batch.seq_len,), (
        f"token_tags shape {tuple(tags.shape)} != ({batch.seq_len},) — one tag per PACKED row."
    )

    # --- (d) the index partition -------------------------------------------------------------------
    joined = torch.cat([kwargs["video_indices"], kwargs["audio_indices"], kwargs["text_indices"]])
    assert torch.equal(torch.sort(joined).values, torch.arange(batch.seq_len)), (
        "video_indices / audio_indices / text_indices must PARTITION range(seq_len) — every packed "
        "row belongs to exactly one segment, exactly once."
    )

    # --- (e) the AdaLN timestep round-trip ---------------------------------------------------------
    assert torch.equal(kwargs["timestep"][kwargs["timestep_indices"]], mi.video.timesteps[0]), (
        "timestep[timestep_indices] must reconstruct the per-row timesteps exactly — the transformer "
        "indexes its AdaLN table off timestep_indices."
    )
    reference_rows = kwargs["video_indices"][: batch.n_cond_video]
    target_rows = kwargs["video_indices"][batch.n_cond_video :]
    if batch.n_cond_video and target_rows.numel():
        row_t = mi.video.timesteps[0]
        assert float(row_t[reference_rows].min()) > float(row_t[target_rows].max()), (
            "visual conditioning rows must be pinned CLEANER than the target rows "
            "(max(t_video, t_visual_cond); H3 inverts the convention — t = 1 is clean)."
        )

    # --- (f) D-10-AUDIO ----------------------------------------------------------------------------
    assert mi.audio is not None, "the H3 audio span must be PRESENT (D-10-AUDIO)"
    n_audio = batch.n_cond_audio + batch.n_target_audio
    assert n_audio > 0 and tuple(mi.audio.latent.shape) == (
        1,
        n_audio,
        EXPECTED_H3_AUDIO_IN_CHANNELS,
    ), (
        f"audio.latent shape {tuple(mi.audio.latent.shape)} != (1, {n_audio}, "
        f"{EXPECTED_H3_AUDIO_IN_CHANNELS}) — target audio rows stay in the packed sequence."
    )
    assert bool((mi.audio.latent[:, batch.n_cond_audio :] != 0).any()), (
        "target audio rows must be NOISED, not silent (D-10-AUDIO): dropping or muting them changes "
        "the sequence the video rows attend to. Do not teach the model to be silent."
    )
    if not cfg.h3.audio_in_loss:
        assert not mi.audio_loss_mask.any(), (
            "audio_in_loss is False -> the audio loss mask must be ALL False. Video-only means "
            "loss-MASKING, never architecture-skipping."
        )
        assert mi.audio_targets is None, (
            "audio is not a training target when audio_in_loss is False."
        )


# --------------------------------------------------------------------------------------------------
# Qwen-Image-Edit-2511 (family #3) — the multi-slot control-image branch of the gate.
#
# Three jobs no earlier arm needed, and each one is a failure that produces a plausible loss curve
# rather than a crash:
#
#   1. Prove the CONTROL BLOCK IS A SUFFIX and is out of the loss. Every other conditioning arm in
#      this gate puts its clean rows FIRST (``ic_lora``'s reference prefix, H3's ``n_cond_video``
#      prefix) and both exclude them by slicing the HEAD. ai-toolkit does the exact inverse —
#      ``torch.cat([packed_latents_list[b], control], dim=1)``
#      (``qwen_image_edit_plus.py:315-317``) — and reads the prediction back with a PREFIX slice
#      (``:346``). A ``[:, ref_seq_len:]`` habit carried from a sibling arm keeps the CONTROL rows
#      and drops the TARGET rows: same dtype, same shape family, an adapter trained on the thing it
#      was supposed to be conditioned by. ``ref_seq_len`` is therefore pinned ``None`` here and the
#      split travels as ``target_seq_len`` / ``control_seq_len``.
#   2. Prove the geometry is QWEN's, not LTX's. ``compute_seq_len`` divides by 32; Qwen packs 16
#      pixels per row edge, so ``compute_seq_len(1024, 1024, 1) == 1024`` against this family's real
#      4096. That is a silently-4x-wrong sequence length, not a shape error.
#   3. Prove the text mask is an INT 0/1 mask. ``qwen_image_edit_plus.py:326-329`` casts it to
#      ``torch.int64`` and derives ``txt_seq_lens`` by SUMMING it; the additive ``-inf`` float mask
#      that ``single_frame``/``ic_lora`` build for ltx_core would sum to ``-inf``.
#
# Every row count arrives from ``cfg.qwen_edit`` via ``conditioning/qwen_edit_geometry`` — this
# module owns NO Qwen geometry and carries no row-count, patch-dim or area literal.
# --------------------------------------------------------------------------------------------------

#: Control-row sentinel — a DISTINCT non-zero constant from the target sentinel, deliberately, so
#: the contract assert can verify the CONCAT POLARITY on the latent itself. Zeros on both halves
#: would let ``torch.cat([control, target])`` pass every shape and mask assertion in this file.
_QWEN_EDIT_CLEAN_CONTROL = -1.0

#: Target-row sentinel. Non-zero for the same reason ``_H3_NOISED_AUDIO`` is: the target rows are
#: PRESENT and NOISED, and an all-zero target block would make the gate assert against a regime
#: that never occurs.
_QWEN_EDIT_NOISED_TARGET = 1.0

#: The synthetic target's file stem. ai-toolkit matches control images to targets BY STEM
#: (``dataloader_mixins.py:979-985``), and ``resolve_control_slots`` records the stem on every slot
#: — including blank fills — so the 1:1 correspondence stays checkable. A fixed literal keeps the
#: gate deterministic; it names no real file and none is opened (D-12: no data, no weights, no VAE).
_QWEN_EDIT_DRYRUN_STEM = "dryrun-0001"


@dataclass(frozen=True)
class QwenEditDryrunBudget:
    """What the qwen_edit preflight priced: the packed layout and the DECLARED row ceiling.

    ``ceiling_rows`` of ``0`` means **DISABLED — nobody has measured this**, which is a different
    state from "measured and roomy" and the banner is required to render it differently. H3 can
    derive a ceiling from a measured ``mib_per_packed_row``; no equivalent measurement exists for
    Qwen-Image-Edit on any card in this program, so there is nothing here to derive from and this
    module refuses to invent one. ``headroom_rows`` is ``None`` in that state rather than a large
    number, because a large number is exactly what an operator would misread as safety.
    """

    layout: QwenEditPackedLayout
    ceiling_rows: int

    @property
    def ceiling_enabled(self) -> bool:
        """True when an operator declared a MEASURED ceiling. False == unmeasured, nothing refused."""
        return self.ceiling_rows > 0

    @property
    def headroom_rows(self) -> int | None:
        """Rows still available, or ``None`` when no ceiling was declared. Negative means over."""
        return self.ceiling_rows - self.layout.total if self.ceiling_enabled else None


def _qwen_edit_budget(cfg: SignetConfig) -> QwenEditDryrunBudget:
    """Price the packed layout and read the declared ceiling. PRICING ONLY — raises no refusal.

    Unlike H3 there is no worst-case ENUMERATION to do, and that is a property of the design rather
    than an omission: ``qwen_edit.control_slots`` is a FIXED count with blank-padding, and every
    slot is priced at the same ``control_area_px`` because an arbitrarily-sized control image is
    fitted to that budget before the VAE sees it. So there is exactly ONE possible layout for a
    given config and worst == nominal non-trivially. (H3's hole was that its pairing REGIME varies
    by design; nothing here varies.)

    CPU-pure: geometry ints only, no ``torch``, no ``ltx_core``, no ``modal``.
    """
    return QwenEditDryrunBudget(
        layout=qwen_edit_packed_layout(cfg.training_dims, cfg.qwen_edit),
        ceiling_rows=cfg.qwen_edit.max_packed_rows,
    )


def assert_qwen_edit_row_budget(cfg: SignetConfig) -> QwenEditDryrunBudget:
    """Preflight guard: refuse a qwen_edit packed sequence that exceeds a DECLARED row ceiling.

    Same posture as :func:`assert_h3_seq_len_budget` — raised BEFORE any GPU spend, with a message
    naming both sides — and the refusal is likewise DELEGATED, to
    ``config/validators.validate_qwen_edit_row_budget``, so it exists in exactly one place.

    ⚠ The honest difference from the H3 sibling: this check is **opt-in**, because its ceiling is an
    operator's measurement rather than a derived number. With ``qwen_edit.max_packed_rows == 0``
    nothing is refused and the layout is merely reported. That is deliberate — the alternative was
    to synthesise a ceiling from H3's measured A100 triple, which prices a different model at a
    different row width with different resident weights. A guess wearing a measurement's clothes is
    worse than a disabled check, because only one of the two announces itself in the banner.

    Returns the :class:`QwenEditDryrunBudget` for the OK banner.
    """
    budget = _qwen_edit_budget(cfg)
    if budget.ceiling_enabled:
        validate_qwen_edit_row_budget(
            budget.layout.total,
            budget.ceiling_rows,
            label=(
                f"{budget.layout.control_slots} control slot(s) at "
                f"{cfg.qwen_edit.control_area_px} px"
            ),
        )
    return budget


def _qwen_edit_control_plan(cfg: SignetConfig) -> list[QwenControlSlot]:
    """The deterministic slot plan the synthetic batch is built against.

    Resolved through ``conditioning/qwen_edit.resolve_control_slots`` — the strategy's OWN resolver,
    never a re-implementation (the ``build_h3_packed_batch`` discipline, T-10-09-T2): a gate that
    resolves its own slots can pass while the strategy resolves them differently.

    The posture is MIXED on purpose, not all-real and not all-blank: every slot but the last carries
    a real path with an EXPLICIT ``slot`` index, and the last is left absent so the resolver has to
    gap-fill it. That exercises the three properties that matter in one shape — real slots resolve
    to their declared index, a gap becomes an explicit blank AT ITS OWN INDEX instead of sliding the
    later controls left (the ai-toolkit slide, ``dataloader_mixins.py:984-985``), and a blank still
    COSTS ROWS because a black image has a real VAE latent rather than a zero one. At
    ``control_slots == 1`` it degenerates to a single blank, which is the correct edge answer.
    """
    entries = [
        {"slot": index, "stem": _QWEN_EDIT_DRYRUN_STEM, "path": f"controls/{index}/synthetic.png"}
        for index in range(cfg.qwen_edit.control_slots - 1)
    ]
    return resolve_control_slots(
        _QWEN_EDIT_DRYRUN_STEM,
        entries,
        control_slots=cfg.qwen_edit.control_slots,
        blank_slot_fill=cfg.qwen_edit.blank_slot_fill,
    )


def _qwen_edit_img_shapes(cfg: SignetConfig) -> tuple[tuple[int, int, int], ...]:
    """``[(F, H2, W2), ...]`` — the target block FIRST, then one entry per control slot.

    Transcribes ai-toolkit's own construction: ``img_shapes = [[(1, img_h2, img_w2)]]`` for the
    target (``qwen_image_edit_plus.py:236``, with ``img_h2, img_w2 = height // 2, width // 2`` over
    the LATENT dims at ``:234``) and one ``append((1, cl_height // 2, cl_width // 2))`` per control
    (``:302``). ``H2 * W2`` is that block's packed row count, which is what makes this tuple a
    checkable restatement of the layout rather than decoration — the contract assert multiplies it
    out and requires the product to equal the rows actually built.
    """
    width, height, _frames = cfg.training_dims
    target_lat_h, target_lat_w = qwen_edit_latent_size(width, height)
    shapes = [(QWEN_EDIT_FRAMES, target_lat_h // QWEN_EDIT_PATCH, target_lat_w // QWEN_EDIT_PATCH)]

    slot_width, slot_height = qwen_edit_area_budget_size(
        width, height, cfg.qwen_edit.control_area_px
    )
    slot_lat_h, slot_lat_w = qwen_edit_latent_size(slot_width, slot_height)
    slot_shape = (QWEN_EDIT_FRAMES, slot_lat_h // QWEN_EDIT_PATCH, slot_lat_w // QWEN_EDIT_PATCH)
    shapes.extend(slot_shape for _ in range(cfg.qwen_edit.control_slots))
    return tuple(shapes)


def _qwen_edit_vae_latents(cfg: SignetConfig) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """The UNPACKED VAE latents the packer folds into rows: ``[1, C, F, H_lat, W_lat]`` each.

    Built even though the gate's ``ModelInputs`` carries only the PACKED sequence, because the two
    shapes are pinned at different layers and a mismatch between them is invisible from either side
    alone. ``qwen_edit_vae_latent_shape`` is the STORAGE contract — rank 4 ``[C, F, H, W]`` with
    ``F == 1``, batched to rank 5 here — which is what lets ``qwen_edit_latents/`` ride
    ``data/precomputed.py``'s ``_normalize_video_latents`` einops path; the transformer boundary is
    rank 3. F is squeezed at the model boundary, never at rest. The contract assert checks the
    element-count identity ``C*F*H*W == rows * QWEN_EDIT_PATCH_DIM``, which is the invariant the 2x2
    pack must satisfy and the one an off-by-a-factor-of-2 patch size breaks first.
    """
    width, height, _frames = cfg.training_dims
    target = torch.zeros(1, *qwen_edit_vae_latent_shape(width, height))

    slot_width, slot_height = qwen_edit_area_budget_size(
        width, height, cfg.qwen_edit.control_area_px
    )
    slot_shape = qwen_edit_vae_latent_shape(slot_width, slot_height)
    controls = tuple(
        torch.full((1, *slot_shape), _QWEN_EDIT_CLEAN_CONTROL)
        for _ in range(cfg.qwen_edit.control_slots)
    )
    return target, controls


def build_qwen_edit_dryrun_inputs(
    cfg: SignetConfig, budget: QwenEditDryrunBudget | None = None
) -> QwenEditModelInputs:
    """Build the synthetic qwen_edit batch as the strategy's own ``QwenEditModelInputs``.

    Deliberately that subclass and not a bare ``ModelInputs`` (the ``h3_ref`` precedent): the fields
    this family adds — ``target_seq_len`` / ``control_seq_len`` / ``img_shapes`` / ``txt_seq_lens``
    — ARE the wire format, and a gate that asserted only the base contract would prove nothing about
    the half of the call that carries Qwen's positional information.

    Tensors are CPU-only at the REAL row counts and the REAL feature widths, so a shape bug still
    surfaces here (D-12: no data, no weights, no VAE). Only the two latent halves carry sentinels,
    and they carry DIFFERENT ones, because the concat polarity is the single most consequential
    thing this arm proves.

    ⚠ DECLARED GAP, so it is not mistaken for coverage: this builds the ``QwenEditModelInputs``
    directly rather than driving ``QwenEditStrategy.prepare_training_inputs``, because that entry
    point consumes a real dataloader batch and the qwen_edit prep pass (``qwen_edit_latents/``,
    ``qwen_edit_control_latents/``, ``qwen_edit_conditions/``) does not exist yet. The slot plan
    already goes through the strategy's own ``resolve_control_slots``. When the prep pass lands,
    re-point this at the strategy the way the H3 arm calls ``build_h3_packed_batch`` — until then
    this gate proves GEOMETRY, POLARITY and MASKING, and does not prove batch-key wiring.
    """
    budget = _qwen_edit_budget(cfg) if budget is None else budget
    layout = budget.layout
    qe = cfg.qwen_edit

    # --- the packed image stream: TARGET rows FIRST, control rows APPENDED ------------------------
    latent = torch.empty(1, layout.n_image_stream, QWEN_EDIT_PATCH_DIM)
    latent[:, : layout.n_target_image, :] = _QWEN_EDIT_NOISED_TARGET
    latent[:, layout.n_target_image :, :] = _QWEN_EDIT_CLEAN_CONTROL

    sampled_sigma = _SAMPLED_SIGMA
    # Control rows are CLEAN conditioning -> per-row timestep 0; target rows carry the sampled sigma.
    # WR-02's reserved-sentinel rule applies here exactly as it does on the LTX arms.
    timesteps = torch.full((1, layout.n_image_stream), sampled_sigma)
    timesteps[:, layout.n_target_image :] = 0.0

    # THE SUFFIX MASK. True on the target PREFIX, False on the control SUFFIX — the inverse of every
    # other arm in this file, which is precisely why it is built explicitly rather than reconstructed
    # from a sibling's idiom.
    video_loss_mask = torch.zeros(1, layout.n_image_stream, dtype=torch.bool)
    video_loss_mask[:, : layout.n_target_image] = True

    # Qwen carries NO coordinate tensor. Positional information travels as ``img_shapes`` /
    # ``txt_seq_lens`` inside ``transformer_kwargs()`` (``qwen_image_edit_plus.py:332-344`` passes no
    # positions argument at all). An EMPTY tensor rather than plausible zeros: a consumer that
    # indexes it gets an immediate IndexError instead of a silently-zero RoPE grid.
    positions = torch.zeros(0)

    text_embeds = torch.zeros(1, layout.n_text, qe.text_embed_dim)
    # ⛔ INT 0/1, NOT the additive -inf float mask the LTX arms build. The transformer casts this to
    # int64 (``qwen_image_edit_plus.py:326-328``) and SUMS it for ``txt_seq_lens`` (``:329``); an
    # -inf-filled float mask sums to -inf and yields a garbage sequence length rather than an error.
    context_mask = torch.ones(1, layout.n_text, dtype=torch.int64)

    video = Modality(
        latent=latent,
        sigma=torch.tensor([sampled_sigma]),
        timesteps=timesteps,
        positions=positions,
        context=text_embeds,
        context_mask=context_mask,
    )

    slots = _qwen_edit_control_plan(cfg)
    return QwenEditModelInputs(
        video=video,
        audio=None,
        # TARGET-ONLY velocity target, at the PREFIX length. A forgotten
        # ``[:, :target_seq_len]`` slice in the loss then fails LOUD on shape instead of quietly
        # training the adapter against its own control conditioning.
        video_targets=torch.randn(1, layout.n_target_image, QWEN_EDIT_PATCH_DIM),
        audio_targets=None,
        video_loss_mask=video_loss_mask,
        audio_loss_mask=None,
        # ⛔ STAYS None. The control block is a SUFFIX; ``ref_seq_len`` means "length of the
        # reference PREFIX" (``strategy.py:114``) and every consumer slices ``[:, ref_seq_len:, :]``.
        ref_seq_len=None,
        target_seq_len=layout.n_target_image,
        control_seq_len=layout.n_control,
        control_slots=tuple(slots),
        control_slot_rows=tuple(layout.per_slot for _ in slots),
        img_shapes=_qwen_edit_img_shapes(cfg),
        txt_seq_lens=(int(context_mask.sum()),),
        caption_dropped=False,
    )


def _assert_qwen_edit_contract(cfg: SignetConfig, mi: ModelInputs) -> None:
    """THE CPU proof for the qwen_edit arm. Raises ``AssertionError`` on any violation.

    (a) the CONTROL SUFFIX is out of the loss and the TARGET PREFIX is entirely in it — the L-3
        guard in its inverted form, and the one assertion that separates this family from every
        other arm in this file;
    (b) the concat POLARITY is right, checked on the latent's own sentinels, so a
        ``torch.cat([control, target])`` cannot pass by satisfying shapes and masks alone;
    (c) ``video_targets`` is TARGET-ONLY at the PREFIX length, so a missing slice fails on shape;
    (d) ``ref_seq_len`` is ``None`` — populating it would hand every ``[:, ref_seq_len:]`` consumer
        the controls and drop the targets;
    (e) the geometry is QWEN's, not LTX's: the packed rows must NOT equal ``compute_seq_len``, which
        under-counts this family by exactly 4x while returning a plausible integer;
    (f) the unpacked VAE latent and the packed rows agree by element count, which is the invariant a
        wrong patch size breaks first;
    (g) ``img_shapes`` multiplies out to exactly the rows built, target block first;
    (h) the text mask is INT 0/1 and ``txt_seq_lens`` is its SUM;
    (i) ``positions`` is EMPTY — this family has no coordinate tensor.

    The batch is rebuilt here (the build is fully deterministic) and BRIDGED to ``mi`` by equality on
    the tensors the caller can see, exactly as ``_assert_h3_contract`` does, so the assertions below
    are provably about the object ``build_dryrun_inputs`` returned.
    """
    budget = _qwen_edit_budget(cfg)
    rebuilt = build_qwen_edit_dryrun_inputs(cfg, budget)
    layout = budget.layout
    width, height, frames = cfg.training_dims

    # --- the bridge: the rebuild IS what build_dryrun_inputs returned -----------------------------
    assert isinstance(mi, QwenEditModelInputs), (
        f"the qwen_edit arm must return QwenEditModelInputs (got {type(mi).__name__}): "
        f"target_seq_len / control_seq_len / img_shapes / txt_seq_lens ARE the wire format, and a "
        f"bare ModelInputs proves nothing about the half of the call that carries Qwen's "
        f"positional information."
    )
    assert torch.equal(mi.video.latent, rebuilt.video.latent), (
        "video.latent does not match the deterministic rebuild"
    )
    assert torch.equal(mi.video_loss_mask, rebuilt.video_loss_mask), (
        "video_loss_mask does not match the deterministic rebuild"
    )
    assert (mi.target_seq_len, mi.control_seq_len) == (
        rebuilt.target_seq_len,
        rebuilt.control_seq_len,
    ), "the target/control split does not match the deterministic rebuild"

    # --- the priced layout is what was actually built ----------------------------------------------
    assert frames == QWEN_EDIT_FRAMES, (
        f"qwen_edit training_dims F is {frames}, must be EXACTLY {QWEN_EDIT_FRAMES} — enforced "
        f"upstream by config/validators.validate_qwen_edit_frames; a config reaching here "
        f"unvalidated bypassed the load-time gate (and LTX's law admits 1, so the shared pre-screen "
        f"never would have caught it)."
    )
    assert mi.target_seq_len == layout.n_target_image == qwen_edit_rows_of(width, height), (
        f"target_seq_len {mi.target_seq_len} != priced target rows {layout.n_target_image}"
    )
    assert mi.control_seq_len == layout.n_control == layout.per_slot * layout.control_slots, (
        f"control_seq_len {mi.control_seq_len} != priced control rows {layout.n_control}"
    )
    assert len(mi.control_slots) == cfg.qwen_edit.control_slots, (
        f"resolved {len(mi.control_slots)} control slot(s) != configured "
        f"{cfg.qwen_edit.control_slots} — the slot count is FIXED and blank-padded, so a short plan "
        f"means a gap was closed up rather than filled (dataloader_mixins.py:984-985's slide)."
    )
    assert tuple(slot.index for slot in mi.control_slots) == tuple(
        range(cfg.qwen_edit.control_slots)
    ), (
        f"control slots are not at consecutive indices 0..{cfg.qwen_edit.control_slots - 1}: "
        f"{[slot.index for slot in mi.control_slots]}. Slot index IS the caption's ctrl_img_N "
        f"addressing, so a shifted plan conditions the sample on the wrong image."
    )
    assert sum(mi.control_slot_rows) == layout.n_control, (
        f"per-slot rows {mi.control_slot_rows} sum to {sum(mi.control_slot_rows)} != "
        f"n_control {layout.n_control}"
    )

    # --- (a) THE SUFFIX loss guard -----------------------------------------------------------------
    assert tuple(mi.video_loss_mask.shape) == (1, layout.n_image_stream), (
        f"video_loss_mask shape {tuple(mi.video_loss_mask.shape)} != (1, {layout.n_image_stream})"
    )
    assert bool(mi.video_loss_mask[:, : mi.target_seq_len].all()), (
        "every TARGET image row must be IN the loss."
    )
    assert not mi.video_loss_mask[:, mi.target_seq_len :].any(), (
        "control-SUFFIX loss mask must be ALL False across [target_seq_len:]: loss on the control "
        "rows trains a dead adapter. ⚠ Note the direction — the control block is a SUFFIX here "
        "(qwen_image_edit_plus.py:315-317 appends it), the inverse of ic_lora's and H3's reference "
        "PREFIX. A [:ref_seq_len] guard copied from either sibling checks the wrong half."
    )
    assert int(mi.video_loss_mask.sum()) == mi.target_seq_len, (
        f"video loss mask covers {int(mi.video_loss_mask.sum())} rows != target_seq_len "
        f"{mi.target_seq_len}"
    )
    assert torch.all(mi.video.timesteps[:, : mi.target_seq_len] == _SAMPLED_SIGMA), (
        "target rows must carry the sampled sigma."
    )
    assert torch.all(mi.video.timesteps[:, mi.target_seq_len :] == 0), (
        "control rows must carry per-row timestep 0 (they are clean conditioning, never denoised)."
    )

    # --- (b) CONCAT POLARITY, on the latent itself -------------------------------------------------
    assert tuple(mi.video.latent.shape) == (1, layout.n_image_stream, QWEN_EDIT_PATCH_DIM), (
        f"video.latent shape {tuple(mi.video.latent.shape)} != (1, {layout.n_image_stream}, "
        f"{QWEN_EDIT_PATCH_DIM}) — [1, target rows + control rows, C * patch**2]."
    )
    assert torch.all(mi.video.latent[:, : mi.target_seq_len, :] == _QWEN_EDIT_NOISED_TARGET), (
        "concat polarity violation: the PREFIX must be the noised TARGET block. ai-toolkit builds "
        "torch.cat([packed_latents, control], dim=1) (qwen_image_edit_plus.py:315-317) and reads "
        "the prediction back with a PREFIX slice (:346) — target first, controls appended."
    )
    assert torch.all(mi.video.latent[:, mi.target_seq_len :, :] == _QWEN_EDIT_CLEAN_CONTROL), (
        "concat polarity violation: the SUFFIX must be the clean CONTROL block."
    )

    # --- (c) TARGET-ONLY targets -------------------------------------------------------------------
    assert tuple(mi.video_targets.shape) == (1, mi.target_seq_len, QWEN_EDIT_PATCH_DIM), (
        f"video_targets shape {tuple(mi.video_targets.shape)} != (1, {mi.target_seq_len}, "
        f"{QWEN_EDIT_PATCH_DIM}) — must be TARGET-ONLY, NOT the packed length "
        f"{layout.n_image_stream}."
    )

    # --- (d) ref_seq_len stays None ----------------------------------------------------------------
    assert mi.ref_seq_len is None, (
        f"ref_seq_len is {mi.ref_seq_len}, must be None on qwen_edit. It means 'length of the "
        f"reference PREFIX' (strategy.py:114) and every consumer slices [:, ref_seq_len:, :]; with "
        f"a SUFFIX control block that keeps the controls and drops the targets — same dtype, "
        f"plausible loss curve, an adapter trained on its own conditioning. The split travels as "
        f"target_seq_len / control_seq_len instead."
    )

    # --- (e) the geometry is QWEN's, not LTX's -----------------------------------------------------
    ltx_rows = compute_seq_len(width, height, frames)
    assert mi.target_seq_len != ltx_rows, (
        f"qwen_edit target rows ({mi.target_seq_len}) must NOT equal compute_seq_len "
        f"({ltx_rows}): LTX divides by 32 and Qwen packs {QWEN_EDIT_PATCH_DIM // QWEN_EDIT_LATENT_CHANNELS}"
        f" latent cells per row at 8x VAE compression, i.e. 16 pixels per row edge. Equality here "
        f"means the LTX helper leaked into this arm — a silently-4x-wrong sequence length, not a "
        f"shape error."
    )

    # --- (f) unpacked VAE latents agree with the packed rows by element count -----------------------
    target_latent, control_latents = _qwen_edit_vae_latents(cfg)
    assert tuple(target_latent.shape) == (1, QWEN_EDIT_LATENT_CHANNELS, QWEN_EDIT_FRAMES, *
                                          qwen_edit_latent_size(width, height)), (
        f"target VAE latent shape {tuple(target_latent.shape)} != (1, {QWEN_EDIT_LATENT_CHANNELS}, "
        f"{QWEN_EDIT_FRAMES}, {qwen_edit_latent_size(width, height)}) — the STORAGE contract is "
        f"rank-4 [C, F, H_lat, W_lat] with F == 1, batched to rank 5. F is squeezed at the model "
        f"boundary, never at rest."
    )
    assert target_latent.numel() == mi.target_seq_len * QWEN_EDIT_PATCH_DIM, (
        f"pack identity violated: the unpacked target latent holds {target_latent.numel()} "
        f"elements but the packed block is {mi.target_seq_len} rows x {QWEN_EDIT_PATCH_DIM} = "
        f"{mi.target_seq_len * QWEN_EDIT_PATCH_DIM}. The 2x2 pack is a reshape and must preserve "
        f"element count exactly (qwen_image_edit_plus.py:222-228) — this is the invariant a wrong "
        f"patch size breaks first."
    )
    assert len(control_latents) == cfg.qwen_edit.control_slots
    for index, control_latent in enumerate(control_latents):
        assert control_latent.numel() == mi.control_slot_rows[index] * QWEN_EDIT_PATCH_DIM, (
            f"pack identity violated for control slot {index}: {control_latent.numel()} elements "
            f"vs {mi.control_slot_rows[index]} rows x {QWEN_EDIT_PATCH_DIM}."
        )

    # --- (g) img_shapes multiplies out to the rows actually built -----------------------------------
    assert len(mi.img_shapes) == 1 + cfg.qwen_edit.control_slots, (
        f"img_shapes has {len(mi.img_shapes)} entries != 1 target + "
        f"{cfg.qwen_edit.control_slots} control slot(s). ai-toolkit builds exactly this list "
        f"(qwen_image_edit_plus.py:236 then :302 per control) and the transformer reads its "
        f"positional grid from it."
    )
    # ⚠ Known limit, stated so the next reader does not over-trust these three lines: they compare
    # ROW COUNTS, so they can only detect a mis-ORDERED img_shapes when the target block and the
    # control blocks have different geometries. At the house 1024x1024 target with
    # control_area_px = VAE_IMAGE_SIZE every block is 4096 rows, and any permutation of an
    # all-4096 list satisfies all three. Probed both ways: at [512, 512, 1] (target 1024 rows,
    # slots 4096) a rotated list is caught on the first assert. Ordering is therefore proven for
    # asymmetric geometries and merely consistent for symmetric ones — detecting it in the
    # symmetric case needs a per-block identity the row count does not carry.
    block_rows = [int(f) * int(h2) * int(w2) for f, h2, w2 in mi.img_shapes]
    assert block_rows[0] == mi.target_seq_len, (
        f"img_shapes[0] {mi.img_shapes[0]} multiplies to {block_rows[0]} rows != target_seq_len "
        f"{mi.target_seq_len} — the TARGET block must come first."
    )
    assert block_rows[1:] == list(mi.control_slot_rows), (
        f"img_shapes control entries multiply to {block_rows[1:]} != per-slot rows "
        f"{list(mi.control_slot_rows)}"
    )
    assert sum(block_rows) == layout.n_image_stream, (
        f"img_shapes accounts for {sum(block_rows)} rows != the packed image stream "
        f"{layout.n_image_stream}"
    )

    # --- (h) the INT text mask and its SUM -----------------------------------------------------------
    assert mi.video.context_mask is not None, "qwen_edit must carry a text attention mask"
    assert mi.video.context_mask.dtype == torch.int64, (
        f"context_mask dtype is {mi.video.context_mask.dtype}, must be torch.int64. Qwen consumes "
        f"this mask TWICE as integers — cast at qwen_image_edit_plus.py:326-328 and SUMMED for "
        f"txt_seq_lens at :329 — so the additive 0/-inf FLOAT mask that single_frame.py:184-185 and "
        f"ic_lora.py:248-249 build for ltx_core sums to -inf here. It does not raise; it produces a "
        f"garbage sequence length and a silently mis-attended text stream."
    )
    assert tuple(mi.video.context.shape) == (1, layout.n_text, cfg.qwen_edit.text_embed_dim), (
        f"video.context shape {tuple(mi.video.context.shape)} != (1, {layout.n_text}, "
        f"{cfg.qwen_edit.text_embed_dim})"
    )
    assert list(mi.txt_seq_lens) == [int(mi.video.context_mask.sum())], (
        f"txt_seq_lens {list(mi.txt_seq_lens)} != the mask's own sum "
        f"[{int(mi.video.context_mask.sum())}] — it is DERIVED from the mask (:329), never carried "
        f"independently."
    )

    # --- (i) no coordinate tensor -------------------------------------------------------------------
    assert mi.video.positions.numel() == 0, (
        f"video.positions holds {mi.video.positions.numel()} elements, must be EMPTY: Qwen carries "
        f"no coordinate tensor. Positional information travels as img_shapes / txt_seq_lens inside "
        f"transformer_kwargs() (qwen_image_edit_plus.py:332-344 passes no positions argument). "
        f"Plausible zeros here would be a silently-zero RoPE grid; an empty tensor makes any "
        f"consumer fail immediately."
    )

    # --- the budget split: what is shaped vs what is priced -----------------------------------------
    assert layout.total == layout.n_image_stream + layout.n_text, (
        f"layout total {layout.total} != image stream {layout.n_image_stream} + text "
        f"{layout.n_text}. The TOTAL is the attention sequence (the budget number); the image "
        f"stream is what hidden_states is shaped by. Dual-stream MMDiT keeps img_* and txt_* as "
        f"separate parameter sets joined only in attention — reporting one and shaping with the "
        f"other is the mistake this split exists to prevent."
    )


def _qwen_edit_ok_banner(cfg: SignetConfig, budget: QwenEditDryrunBudget) -> str:
    """The qwen_edit OK line: geometry, the packed breakdown, the slot plan, and the ceiling STATE.

    ⚠ The ceiling clause is required to read ``ceiling=DISABLED (unmeasured)`` when no ceiling was
    declared, and to print NO headroom number in that state. A headroom figure is read as reassurance
    and there is nothing here to be reassured by: no MiB-per-row measurement for Qwen-Image-Edit
    exists in this program. The banner's job is to make "nobody measured this" as visible as a
    number would have been.
    """
    width, height, frames = cfg.training_dims
    lat_h, lat_w = qwen_edit_latent_size(width, height)
    if budget.ceiling_enabled:
        ceiling = f"ceiling={budget.ceiling_rows} (headroom {budget.headroom_rows} rows)"
    else:
        ceiling = "ceiling=DISABLED (unmeasured — qwen_edit.max_packed_rows is 0, nothing refused)"
    return (
        f"[signet-dryrun] OK — qwen_edit config valid, synthetic packed batch built on CPU: "
        f"family={cfg.model.family}, {width}x{height} pixels x {frames} frame -> "
        f"{lat_w}x{lat_h} latent, {budget.layout.describe()}, "
        f"{budget.layout.control_slots} slot(s) at {cfg.qwen_edit.control_area_px} px "
        f"(blank fill '{cfg.qwen_edit.blank_slot_fill}'), rank/alpha "
        f"{cfg.lora.rank}/{cfg.lora.alpha} lock={cfg.qwen_edit.rank_alpha_lock}, "
        f"{ceiling}. Zero GPU, zero Modal spend."
    )


def _inpaint_dryrun_mask(lat_f: int, lat_h: int, lat_w: int) -> torch.Tensor:
    """Deterministic ASYMMETRIC ``[F_lat, H_lat, W_lat]`` KEEP mask for the inpaint dry-run.

    One KEEP cell per latent frame at ``(h, w) = (f % H_lat, (2*f + 1) % W_lat)`` — asymmetric in
    all three axes so a transposed / non-frame-major flatten CANNOT reproduce the same flat token
    indices. Shared by ``build_dryrun_inputs`` and ``_assert_contract`` (one source of truth, the
    ``_SAMPLED_SIGMA`` pattern) — the assert recomputes the EXPECTED flat indices independently via
    ``mask_ops.latent_frame_token_span`` and compares them against the built tensors.
    """
    mask = torch.zeros(lat_f, lat_h, lat_w, dtype=torch.float32)
    for f in range(lat_f):
        mask[f, f % lat_h, (2 * f + 1) % lat_w] = 1.0
    return mask


def _inpaint_expected_keep_indices(lat_f: int, lat_h: int, lat_w: int) -> list[int]:
    """The hand-computed flat token indices of :func:`_inpaint_dryrun_mask`'s KEEP cells.

    Computed via ``latent_frame_token_span`` (frame-major: ``token = f*H*W + h*W + w``) — the
    INDEPENDENT recomputation the contract assert checks the flattened mask against, proving the
    strategy-side ``reshape(B, F*H*W)`` flatten matches the patchifier's token order exactly.
    """
    indices: list[int] = []
    for f in range(lat_f):
        start, _stop = latent_frame_token_span(f, lat_h, lat_w)
        indices.append(start + (f % lat_h) * lat_w + ((2 * f + 1) % lat_w))
    return indices


def _multi_frame_regions(cfg: SignetConfig) -> tuple[list[tuple[int, int, float]], int]:
    """Deterministic N-item ``(start, stop, strength)`` region set for the multi_frame dry-run.

    Mirrors ``MultiFrameStrategy``'s latent-native region math (``conditioning/multi_frame.py``) but
    with DETERMINISTIC positions (no rng) so the CPU gate is reproducible:

      * ``conditioning_items`` non-empty  -> one region per item at ``latent_idx = frame_index // 8``
        (frame_index is %8-aligned by ``validate_conditioning_items``), carrying the item's strength;
      * ``conditioning_items`` empty (self-conditioning) -> up to ``max_conditioning_items``
        evenly-spaced DISTINCT latent indices at strength = the range's upper bound.

    Returns ``(regions, per_frame_tokens)`` where ``per_frame_tokens = (H//32)*(W//32)`` — the token
    span EACH latent-frame region occupies (the multi-frame generalization of single_frame's
    ``first_frame_tokens``).
    """
    width, height, frames = cfg.training_dims
    latent_frames = (frames - 1) // TIME_SCALE + 1
    lat_h = height // HEIGHT_SCALE
    lat_w = width // WIDTH_SCALE
    per_frame_tokens = lat_h * lat_w

    cond = cfg.conditioning
    positions_strengths: list[tuple[int, float]] = []
    if cond.conditioning_items:
        for item in cond.conditioning_items:
            latent_idx = item.frame_index // TIME_SCALE
            positions_strengths.append((latent_idx, float(item.strength)))
    else:
        # Self-conditioning fallback: evenly-spaced distinct latent frames (e.g. [0, mid, last] at
        # max=3) at strength = the range's upper bound (hi). K clamped to latent_frames.
        _lo, hi = cond.conditioning_strength_range
        k = min(cond.max_conditioning_items, latent_frames)
        if k <= 1:
            idxs = [0]
        else:
            idxs = [round(i * (latent_frames - 1) / (k - 1)) for i in range(k)]
        positions_strengths = [(idx, float(hi)) for idx in idxs]

    regions = [
        (*latent_frame_token_span(latent_idx, lat_h, lat_w), strength)
        for latent_idx, strength in positions_strengths
    ]
    return regions, per_frame_tokens


def build_dryrun_inputs(cfg: SignetConfig) -> ModelInputs:
    """Build a synthetic CPU ``ModelInputs`` from the validated config dims (D-12).

    No model, no VAE, no weights — just shape-correct tensors derived from ``training_dims``.
    Mirrors the per-token-conditioning facts from RESEARCH.md Q4:
        * conditioning tokens get timestep 0; targets get the sampled sigma;
        * first-frame conditioning region = (H//32)*(W//32) tokens;
        * video_targets = velocity (noise - clean); here synthetic randn;
        * video_loss_mask = ~conditioning_mask.

    The ``multi_frame`` branch (Plan 06-04) generalizes this to N latent-frame regions with per-item
    strength: it builds an EXPLICIT ``video_loss_mask = denoise_mask > 0`` (NOT the ``timesteps == 0``
    reconstruction) because a partial-strength token has ``denoise_mask = 1 - strength`` in ``(0, 1)``
    and therefore a NON-ZERO per-token sigma, yet must still participate in the loss (Pitfall 1).

    The ``h3`` branch (Plan 10-09) dispatches on ``cfg.model.family`` FIRST — before the
    ``conditioning.mode`` dispatch below — because MiniMax-H3's sequence is a PACKED multi-modal
    layout, not an LTX video-token count. ``compute_seq_len``'s ``(F-1)//8+1`` frame law and
    128-channel latents are both wrong for H3 and would silently produce a plausible number. Every
    LTX branch below is untouched.

    The ``qwen_edit`` branch (family #3) dispatches on the family for the SAME reason and one more.
    ``compute_seq_len`` is not merely inapplicable there, it is quietly WRONG: it divides by 32
    while Qwen packs 16 pixels per row edge, so it returns exactly a quarter of the real row count
    as a perfectly plausible integer (``compute_seq_len(1024, 1024, 1) == 1024`` against a real
    4096). And unlike H3, that family's frame count is legal under LTX's own law
    (``(1 - 1) % 8 == 0``), so nothing upstream would have diverted it. Its ``conditioning.mode`` is
    pinned to ``none`` at config load, so the mode dispatch below is unreachable for it either way.
    """
    if cfg.model.family == "h3":
        return _build_h3_dryrun_inputs(cfg)
    if cfg.model.family == "qwen_edit":
        return build_qwen_edit_dryrun_inputs(cfg)

    width, height, frames = cfg.training_dims
    seq_len = compute_seq_len(width, height, frames)

    latent = torch.zeros(1, seq_len, VIDEO_LATENT_CHANNELS)

    sampled_sigma = _SAMPLED_SIGMA
    # WR-02: timestep 0 is a RESERVED conditioning sentinel. Downstream (_assert_contract) and the
    # real loop reconstruct the conditioning mask as ``timesteps == 0``, so a target sigma of
    # exactly 0.0 (a legitimate flow-matching endpoint) would alias every target token as a
    # conditioning token and silently invert the loss mask while still passing the invariant. Guard
    # the sentinel explicitly: target sigmas must never collide with 0.
    assert sampled_sigma != 0, (
        "sampled_sigma must be non-zero: timestep 0 is the reserved conditioning sentinel "
        "(target sigma == 0 would alias targets as conditioning tokens — WR-02)."
    )

    if cfg.conditioning.mode == "ic_lora":
        # IC-LoRA DOUBLED sequence (Plan 07-07, mirrors ICLoraStrategy.prepare_training_inputs): a
        # CLEAN reference prefix concatenated to the noised target. ref is 1:1 with the target
        # (D-7-REF11) so ``ref_seq_len == seq_len`` and ``combined == 2 * seq_len``. This is the CPU
        # proof that L-3 (loss on the reference prefix — the dead-adapter bug, Pitfall 1) cannot
        # reach a metered run: the reference-prefix half of ``video_loss_mask`` is ALL False and
        # ``video_targets`` is TARGET-ONLY (length ``seq_len``, NOT the combined sequence).
        ref_seq_len = seq_len
        combined = ref_seq_len + seq_len

        # TARGET-half first-frame conditioning (deterministic port of single_frame.py:142-157 — no
        # rng so the CPU gate is reproducible). The reference prefix is unconditionally clean and is
        # NEVER in the loss, so conditioning only ever touches the target half.
        # WR-02: drive this ONLY off the explicit dry-run knob — the real ic_lora training path trains
        # at p=0.0 (loop.py threads only reference_downscale_factor, CR-01; the schema rejects a
        # non-default first_frame_conditioning_p in ic_lora mode). Reading first_frame_conditioning_p
        # here (default 1.0 > 0) made the gate exercise a first-frame-conditioned posture the real
        # ic_lora path never uses — the gate and the loop read different effective values of the knob.
        cond_enabled = cfg.dry_run.cond_first_frame
        target_cond_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        if cond_enabled:
            first_frame_tokens = (height // 32) * (width // 32)
            target_cond_mask[:, :first_frame_tokens] = True

        # timesteps: reference prefix ALL 0 (clean conditioning); target half = 0 on conditioned
        # tokens, sampled sigma elsewhere. Concatenate [ref, target] on the sequence axis.
        target_timesteps = torch.where(
            target_cond_mask, torch.zeros(1), torch.full((1,), sampled_sigma)
        )
        ref_timesteps = torch.zeros(1, ref_seq_len)
        combined_timesteps = torch.cat([ref_timesteps, target_timesteps], dim=1)

        # (5) loss mask: reference prefix ALL False; target half = ~target_cond_mask. The ref-prefix
        # zeros are the L-3 guard — loss on the reference region trains a dead/leaky adapter.
        video_loss_mask = torch.cat(
            [torch.zeros(1, ref_seq_len, dtype=torch.bool), ~target_cond_mask], dim=1
        )

        combined_latent = torch.zeros(1, combined, VIDEO_LATENT_CHANNELS)
        combined_positions = torch.zeros(1, 3, combined, 2)  # doubled RoPE positions [1,3,combined,2]
        context = torch.zeros(1, cfg.dry_run.n_text_tokens, cfg.dry_run.ctx_dim)
        video = Modality(
            latent=combined_latent,
            sigma=torch.tensor([sampled_sigma]),
            timesteps=combined_timesteps,
            positions=combined_positions,
            context=context,
            context_mask=torch.ones(1, cfg.dry_run.n_text_tokens, dtype=torch.bool),
        )
        # (3) TARGET-ONLY velocity target: length ``seq_len == combined - ref_seq_len`` (NOT combined).
        # A forgotten ``[:, ref_seq_len:]`` slice in compute_loss then fails LOUD on shape.
        video_targets = torch.randn(1, seq_len, VIDEO_LATENT_CHANNELS)
        return ModelInputs(
            video=video,
            audio=None,
            video_targets=video_targets,
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
            ref_seq_len=ref_seq_len,
        )

    if cfg.conditioning.mode == "inpaint":
        # Inpaint per-token spatial denoise mask (Phase 9 gates — mirrors
        # InpaintStrategy.prepare_training_inputs at its always-apply posture; the
        # inpaint_mask_probability Bernoulli gate is a strategy-side rng concern covered by
        # tests/test_inpaint_strategy.py — the CPU gate stays DETERMINISTIC like the siblings).
        # Build the known-asymmetric [F_lat, H_lat, W_lat] KEEP mask, flatten it FRAME-MAJOR
        # (token = f*H*W + h*W + w — exactly mask_ops.latent_frame_token_span order), and apply
        # the contract's where() semantics: KEEP tokens carry the CLEAN sentinel at timestep 0
        # and are excluded from the loss; GENERATE tokens carry the NOISY sentinel at the
        # sampled sigma and are in the loss.
        lat_f = (frames - 1) // TIME_SCALE + 1
        lat_h = height // HEIGHT_SCALE
        lat_w = width // WIDTH_SCALE
        mask = _inpaint_dryrun_mask(lat_f, lat_h, lat_w)
        mask_flat = mask.reshape(1, lat_f * lat_h * lat_w)  # frame-major flatten
        keep = mask_flat > 0.5  # [1, seq_len] bool: True = KEEP/context

        inpaint_latent = torch.where(
            keep.unsqueeze(-1),
            torch.full((1, seq_len, VIDEO_LATENT_CHANNELS), _INPAINT_CLEAN),
            torch.full((1, seq_len, VIDEO_LATENT_CHANNELS), _INPAINT_NOISY),
        )
        timesteps = torch.where(keep, torch.zeros(1), torch.full((1,), sampled_sigma))
        video_loss_mask = ~keep  # == (mask <= 0.5): GENERATE tokens carry the loss

        positions = torch.zeros(1, 3, seq_len, 2)
        context = torch.zeros(1, cfg.dry_run.n_text_tokens, cfg.dry_run.ctx_dim)
        video = Modality(
            latent=inpaint_latent,
            sigma=torch.tensor([sampled_sigma]),
            timesteps=timesteps,
            positions=positions,
            context=context,
            context_mask=torch.ones(1, cfg.dry_run.n_text_tokens, dtype=torch.bool),
        )
        return ModelInputs(
            video=video,
            audio=None,
            video_targets=torch.randn(1, seq_len, VIDEO_LATENT_CHANNELS),
            audio_targets=None,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,
        )

    if cfg.conditioning.mode == "audio_to_video":
        # a2v (Phase 9 gates — mirrors A2VStrategy.prepare_training_inputs): the VIDEO is generated
        # (plain t2v — every token noised, at the sampled sigma, ALL in the loss) and the driving
        # AUDIO is FROZEN conditioning (clean latents, per-token timestep 0, sigma 0, EXCLUDED from
        # the loss — audio_targets / audio_loss_mask stay None).
        video_timesteps = torch.full((1, seq_len), sampled_sigma)
        video_loss_mask = torch.ones(1, seq_len, dtype=torch.bool)  # whole frame generated
        positions = torch.zeros(1, 3, seq_len, 2)
        context = torch.zeros(1, cfg.dry_run.n_text_tokens, cfg.dry_run.ctx_dim)
        video = Modality(
            latent=latent,
            sigma=torch.tensor([sampled_sigma]),
            timesteps=video_timesteps,
            positions=positions,
            context=context,
            context_mask=torch.ones(1, cfg.dry_run.n_text_tokens, dtype=torch.bool),
        )
        # The FROZEN audio Modality: clean latents [1, T, 128], sigma 0, per-token timestep 0, and
        # audio positions [1, 1, T, 2] (ONE positional dim — vs video's 3).
        audio_t = _A2V_AUDIO_TIME_STEPS
        audio = Modality(
            latent=torch.zeros(1, audio_t, _A2V_AUDIO_TOKEN_DIM),
            sigma=torch.zeros(1),
            timesteps=torch.zeros(1, audio_t),
            positions=torch.zeros(1, 1, audio_t, 2),
            context=torch.zeros(1, cfg.dry_run.n_text_tokens, cfg.dry_run.ctx_dim),
            context_mask=torch.ones(1, cfg.dry_run.n_text_tokens, dtype=torch.bool),
        )
        return ModelInputs(
            video=video,
            audio=audio,
            video_targets=torch.randn(1, seq_len, VIDEO_LATENT_CHANNELS),
            audio_targets=None,  # audio is conditioning, never a training target
            video_loss_mask=video_loss_mask,
            audio_loss_mask=None,  # frozen audio excluded from the loss
        )

    if cfg.conditioning.mode == "multi_frame":
        # N-item denoise mask (canonical ``1 - strength``), per-token ``sigma * denoise_mask``
        # noising, and an EXPLICIT loss mask. This is the SAME math MultiFrameStrategy reads — the
        # CPU gate proves it against ``video_loss_mask``, not against ``timesteps == 0`` (Pitfall 1).
        regions, _per_frame_tokens = _multi_frame_regions(cfg)
        denoise_mask = build_denoise_mask(1, seq_len, regions, device="cpu")
        timesteps = per_token_sigma(torch.tensor([sampled_sigma]), denoise_mask)
        video_loss_mask = denoise_mask > 0  # partial-strength tokens (denoise>0) stay IN the loss
    else:
        # The synthetic first-frame conditioning region is populated when EITHER gate is on: the
        # explicit dry-run knob (``dry_run.cond_first_frame``), OR the real conditioning config the
        # training path uses (``conditioning.mode == "single_frame"`` with
        # ``first_frame_conditioning_p > 0``). Threading the config field here makes the CPU gate
        # exercise the SAME switch that ``training_step``'s SingleFrameStrategy reads — a bad/zero p
        # no longer silently bypasses the gate's mask invariants (SC#1 train-side proof, zero GPU).
        cond_enabled = cfg.dry_run.cond_first_frame or (
            cfg.conditioning.mode == "single_frame"
            and cfg.conditioning.first_frame_conditioning_p > 0
        )
        cond_mask = torch.zeros(1, seq_len, dtype=torch.bool)
        if cond_enabled:
            first_frame_tokens = (height // 32) * (width // 32)
            cond_mask[:, :first_frame_tokens] = True

        # Conditioning tokens get timestep 0; target tokens get the sampled sigma.
        timesteps = torch.where(
            cond_mask, torch.zeros(1), torch.full((1,), sampled_sigma)
        )
        video_loss_mask = ~cond_mask

    positions = torch.zeros(1, 3, seq_len, 2)
    context = torch.zeros(1, cfg.dry_run.n_text_tokens, cfg.dry_run.ctx_dim)

    video = Modality(
        latent=latent,
        sigma=torch.tensor([sampled_sigma]),
        timesteps=timesteps,
        positions=positions,
        context=context,
        context_mask=torch.ones(1, cfg.dry_run.n_text_tokens, dtype=torch.bool),
    )

    video_targets = torch.randn(1, seq_len, VIDEO_LATENT_CHANNELS)  # velocity target stand-in

    return ModelInputs(
        video=video,
        audio=None,
        video_targets=video_targets,
        audio_targets=None,
        video_loss_mask=video_loss_mask,
        audio_loss_mask=None,
    )


def _assert_contract(cfg: SignetConfig, mi: ModelInputs) -> None:
    """Assert the REAL ModelInputs/Modality contract (RESEARCH.md Q4). Raises on any violation."""
    # H3 first: the packed multi-modal layout has its own contract and none of the LTX seq_len shape
    # asserts below apply to it (same posture as the ic_lora arm, one level up).
    if cfg.model.family == "h3":
        _assert_h3_contract(cfg, mi)
        return
    # qwen_edit likewise: a packed image stream with a SUFFIX control block, an int text mask and no
    # coordinate tensor. None of the LTX seq_len/channel asserts below describe it — and unlike the
    # H3 arm they would not even fail loudly on it, because ``compute_seq_len`` returns a plausible
    # (4x too small) integer for an image geometry rather than raising.
    if cfg.model.family == "qwen_edit":
        _assert_qwen_edit_contract(cfg, mi)
        return

    width, height, frames = cfg.training_dims
    seq_len = compute_seq_len(width, height, frames)
    D = VIDEO_LATENT_CHANNELS

    if cfg.conditioning.mode == "ic_lora":
        # IC-LoRA DOUBLED-sequence contract (Plan 07-07). The generic seq_len shape asserts below do
        # NOT apply — every video-side tensor is ``combined = ref_seq_len + seq_len`` long, while
        # ``video_targets`` is TARGET-ONLY (length ``seq_len``). Assert the three L-3 invariants.
        ref_seq_len = seq_len  # D-7-REF11: reference is 1:1 with the target.
        combined = ref_seq_len + seq_len

        assert mi.ref_seq_len == ref_seq_len, (
            f"ModelInputs.ref_seq_len {mi.ref_seq_len} != {ref_seq_len}"
        )
        # (c) doubled positions / latent / timesteps / loss mask are all combined-length.
        assert tuple(mi.video.positions.shape) == (1, 3, combined, 2), (
            f"video.positions shape {tuple(mi.video.positions.shape)} != (1, 3, {combined}, 2)"
        )
        assert tuple(mi.video.latent.shape) == (1, combined, D), (
            f"video.latent shape {tuple(mi.video.latent.shape)} != (1, {combined}, {D})"
        )
        assert tuple(mi.video.timesteps.shape) == (1, combined), (
            f"video.timesteps shape {tuple(mi.video.timesteps.shape)} != (1, {combined})"
        )
        assert tuple(mi.video_loss_mask.shape) == (1, combined), (
            f"video_loss_mask shape {tuple(mi.video_loss_mask.shape)} != (1, {combined})"
        )
        # (b) video_targets is TARGET-ONLY: length == combined - ref_seq_len (== seq_len).
        assert tuple(mi.video_targets.shape) == (1, combined - ref_seq_len, D), (
            f"video_targets shape {tuple(mi.video_targets.shape)} != "
            f"(1, {combined - ref_seq_len}, {D}) — must be TARGET-ONLY (L-3)"
        )
        # (a) THE L-3 guard: the reference-prefix loss mask is ALL False — loss NEVER lands on the
        # reference region (the dead/leaky-adapter bug, Pitfall 1).
        assert not mi.video_loss_mask[:, :ref_seq_len].any(), (
            "reference-prefix loss mask must be ALL False across [:ref_seq_len] (L-3): loss on the "
            "reference region trains a dead adapter."
        )
        # The reference prefix is clean conditioning -> per-token timestep 0 across the prefix.
        assert torch.all(mi.video.timesteps[:, :ref_seq_len] == 0), (
            "reference-prefix timesteps must all be 0 (the reference is clean conditioning)."
        )
        # Target half: loss mask == ~target-conditioning mask (reconstructed via timesteps == 0 on
        # the target slice; sampled_sigma != 0 keeps 0 the reserved conditioning sentinel — WR-02).
        target_cond_mask = mi.video.timesteps[:, ref_seq_len:] == 0
        assert torch.equal(mi.video_loss_mask[:, ref_seq_len:], ~target_cond_mask), (
            "target-half loss mask must equal ~target_conditioning_mask"
        )
        return

    if cfg.conditioning.mode == "inpaint":
        # Inpaint dims rule (GATE-SPEC: "÷64 required for inpaint" + 8n+1 frames — STRICTER than
        # the video %32 rule). The rule is ENFORCED at config load by config/validators.py (the
        # inpaint-mode %64 assert — that module OWNS it, not this gate); this assert VERIFIES the
        # upstream enforcement fired before any synthetic-batch math trusts the dims.
        assert height % 64 == 0 and width % 64 == 0, (
            f"inpaint dims must be %64 (H={height}, W={width}) — enforced upstream by "
            f"config/validators.py (inpaint-mode %64 assert); a config reaching here unvalidated "
            f"bypassed the load-time gate."
        )
        assert frames % 8 == 1, (
            f"inpaint frames must be 8n+1 (F={frames}) — enforced upstream by "
            f"config/validators.py; a config reaching here unvalidated bypassed the load-time gate."
        )

        lat_f = (frames - 1) // TIME_SCALE + 1
        lat_h = height // HEIGHT_SCALE
        lat_w = width // WIDTH_SCALE

        # FLATTEN-ORDER PROOF: recompute the asymmetric pattern's KEEP token indices INDEPENDENTLY
        # via mask_ops.latent_frame_token_span (token = f*H*W + h*W + w) and require the built
        # tensors to mark EXACTLY those tokens. A transposed / non-frame-major flatten cannot
        # reproduce this index set (the pattern is asymmetric in all three axes).
        expected_keep = torch.zeros(1, seq_len, dtype=torch.bool)
        expected_keep[:, _inpaint_expected_keep_indices(lat_f, lat_h, lat_w)] = True

        keep_from_timesteps = mi.video.timesteps == 0  # sentinel-safe: sampled_sigma != 0 (WR-02)
        assert torch.equal(keep_from_timesteps, expected_keep), (
            "inpaint flatten-order violation: the timestep-0 (KEEP) token set does not match the "
            "frame-major latent_frame_token_span indices — the [F,H,W] -> [seq_len] flatten no "
            "longer matches the patchifier's token order."
        )

        # POLARITY: KEEP tokens (mask > 0.5) got the CLEAN latent + timestep 0 + are EXCLUDED from
        # the loss; GENERATE tokens (mask <= 0.5) got the noisy latent + the sampled sigma + are IN
        # the loss. (1.0 = KEEP/context, 0.0 = GENERATE — the encoded-tensor polarity.)
        assert torch.equal(mi.video_loss_mask, ~expected_keep), (
            "inpaint polarity violation: video_loss_mask must equal ~keep — KEEP/context tokens "
            "(mask > 0.5) must be EXCLUDED from the loss, GENERATE tokens (mask <= 0.5) IN it."
        )
        assert torch.all(mi.video.timesteps[expected_keep] == 0), (
            "inpaint polarity violation: KEEP tokens must carry per-token timestep 0."
        )
        assert torch.all(mi.video.timesteps[~expected_keep] == _SAMPLED_SIGMA), (
            "inpaint polarity violation: GENERATE tokens must carry the sampled sigma."
        )
        keep_latent = mi.video.latent[expected_keep.unsqueeze(-1).expand(-1, -1, D)]
        assert torch.all(keep_latent == _INPAINT_CLEAN), (
            "inpaint polarity violation: KEEP tokens must carry the CLEAN latent (the "
            "where(mask>0.5, clean, noisy) substitution branch is swapped)."
        )
        gen_latent = mi.video.latent[(~expected_keep).unsqueeze(-1).expand(-1, -1, D)]
        assert torch.all(gen_latent == _INPAINT_NOISY), (
            "inpaint polarity violation: GENERATE tokens must carry the noisy latent."
        )
        # Generic full-length contract shapes still hold (no concat — unlike ic_lora).
        assert tuple(mi.video.latent.shape) == (1, seq_len, D)
        assert tuple(mi.video_targets.shape) == (1, seq_len, D)
        assert tuple(mi.video_loss_mask.shape) == (1, seq_len)
        assert mi.ref_seq_len is None, "inpaint has no reference prefix — ref_seq_len must be None"
        return

    if cfg.conditioning.mode == "audio_to_video":
        # a2v contract (GATE-SPEC rev 2 item 8): VIDEO generated in full; AUDIO frozen conditioning.
        # VIDEO: standard full-length shapes, every token generated (loss mask all True) at the
        # sampled sigma (no conditioning tokens).
        assert tuple(mi.video.latent.shape) == (1, seq_len, D), (
            f"video.latent shape {tuple(mi.video.latent.shape)} != (1, {seq_len}, {D})"
        )
        assert tuple(mi.video.positions.shape) == (1, 3, seq_len, 2), (
            f"video.positions shape {tuple(mi.video.positions.shape)} != (1, 3, {seq_len}, 2)"
        )
        assert tuple(mi.video_targets.shape) == (1, seq_len, D), (
            f"video_targets shape {tuple(mi.video_targets.shape)} != (1, {seq_len}, {D})"
        )
        assert tuple(mi.video_loss_mask.shape) == (1, seq_len), (
            f"video_loss_mask shape {tuple(mi.video_loss_mask.shape)} != (1, {seq_len})"
        )
        assert bool(mi.video_loss_mask.all()), (
            "a2v generates the whole video — every video token must be IN the loss "
            "(video_loss_mask all True)."
        )
        assert torch.all(mi.video.timesteps == _SAMPLED_SIGMA), (
            "a2v video tokens are all generated — every video timestep must be the sampled sigma "
            "(no first-frame / mask conditioning on the video side)."
        )
        # AUDIO: FROZEN conditioning — present, clean, per-token timestep 0, sigma 0, EXCLUDED from
        # the loss (audio_targets / audio_loss_mask None). ONE positional dim (vs video's 3).
        assert mi.audio is not None, "a2v must carry a frozen audio Modality (mi.audio is None)"
        audio_t = _A2V_AUDIO_TIME_STEPS
        assert tuple(mi.audio.latent.shape) == (1, audio_t, _A2V_AUDIO_TOKEN_DIM), (
            f"audio.latent shape {tuple(mi.audio.latent.shape)} != (1, {audio_t}, "
            f"{_A2V_AUDIO_TOKEN_DIM})"
        )
        assert tuple(mi.audio.positions.shape) == (1, 1, audio_t, 2), (
            f"audio.positions shape {tuple(mi.audio.positions.shape)} != (1, 1, {audio_t}, 2) "
            f"(audio has ONE positional dim, not video's 3)."
        )
        assert torch.all(mi.audio.timesteps == 0), (
            "frozen audio must carry per-token timestep 0 (it is driving conditioning, never "
            "denoised)."
        )
        assert torch.all(mi.audio.sigma == 0), "frozen audio modality sigma must be 0."
        assert mi.audio_targets is None and mi.audio_loss_mask is None, (
            "a2v audio is conditioning, never a training target — audio_targets and audio_loss_mask "
            "must be None (the audio is EXCLUDED from the loss)."
        )
        assert mi.ref_seq_len is None, "a2v has no reference prefix — ref_seq_len must be None"
        return

    # Shapes.
    assert tuple(mi.video.latent.shape) == (1, seq_len, D), (
        f"video.latent shape {tuple(mi.video.latent.shape)} != (1, {seq_len}, {D})"
    )
    assert tuple(mi.video.positions.shape) == (1, 3, seq_len, 2), (
        f"video.positions shape {tuple(mi.video.positions.shape)} != (1, 3, {seq_len}, 2)"
    )
    assert tuple(mi.video_targets.shape) == (1, seq_len, D), (
        f"video_targets shape {tuple(mi.video_targets.shape)} != (1, {seq_len}, {D})"
    )
    assert tuple(mi.video.sigma.shape) == (1,), (
        f"video.sigma shape {tuple(mi.video.sigma.shape)} != (1,)"
    )
    assert tuple(mi.video.timesteps.shape) == (1, seq_len), (
        f"video.timesteps shape {tuple(mi.video.timesteps.shape)} != (1, {seq_len})"
    )
    assert tuple(mi.video_loss_mask.shape) == (1, seq_len), (
        f"video_loss_mask shape {tuple(mi.video_loss_mask.shape)} != (1, {seq_len})"
    )

    if cfg.conditioning.mode == "multi_frame":
        # Multi-frame Pitfall-1 guard: the ``timesteps == 0`` reconstruction is INVALID here — a
        # partial-strength token has ``denoise_mask = 1 - strength`` in ``(0, 1)`` and thus a NON-ZERO
        # per-token sigma, yet it MUST be IN the loss. So assert against the EXPLICIT ``video_loss_mask``
        # via the reconstructed denoise_mask (``timesteps / sampled_sigma``), NOT against timestep 0.
        regions, per_frame_tokens = _multi_frame_regions(cfg)
        denoise_mask = mi.video.timesteps / _SAMPLED_SIGMA  # [1, seq_len]
        assert torch.all((denoise_mask >= 0.0) & (denoise_mask <= 1.0)), (
            "reconstructed denoise_mask must be in [0, 1] (timesteps / sampled_sigma)"
        )
        assert torch.equal(mi.video_loss_mask, denoise_mask > 0), (
            "video_loss_mask must equal (denoise_mask > 0) — the EXPLICIT loss mask, NOT timesteps == 0"
        )
        for start, stop, strength in regions:
            # Each conditioning region spans exactly one latent frame = (H//32)*(W//32) tokens.
            assert stop - start == per_frame_tokens, (
                f"conditioning region [{start}, {stop}) spans {stop - start} tokens != "
                f"per-frame {per_frame_tokens}"
            )
            expected = 1.0 - strength  # canonical denoise_mask = 1 - strength
            region = denoise_mask[:, start:stop]
            assert torch.allclose(region, torch.full_like(region, expected), atol=1e-6), (
                f"region [{start}, {stop}) denoise_mask != 1 - strength ({expected})"
            )
        # Pitfall-1 explicit: any partial-strength token (0 < denoise < 1) has timesteps != 0 yet
        # stays True in the loss mask — the exact case a ``timesteps == 0`` reconstruction gets right
        # by luck but is fragile about; assert it directly.
        partial = (denoise_mask > 0) & (denoise_mask < 1)
        if partial.any():
            assert torch.all(mi.video.timesteps[partial] != 0), (
                "partial-strength conditioning tokens must carry a NON-ZERO per-token sigma"
            )
            assert torch.all(mi.video_loss_mask[partial]), (
                "partial-strength conditioning tokens must be IN the loss mask (Pitfall 1)"
            )
        return

    # Mask invariants (single_frame / none). NOTE (WR-02): reconstructing cond_mask from
    # ``timesteps == 0`` is only valid because timestep 0 is a reserved conditioning sentinel that
    # target sigmas never collide with (build_dryrun_inputs asserts ``sampled_sigma != 0``). The
    # multi_frame branch above deliberately does NOT use this reconstruction (partial strength breaks
    # it); it asserts against the explicit ``video_loss_mask`` instead.
    cond_mask = mi.video.timesteps == 0
    assert tuple(cond_mask.shape) == (1, seq_len), (
        f"conditioning_mask shape {tuple(cond_mask.shape)} != (1, {seq_len})"
    )
    assert torch.equal(mi.video_loss_mask, ~cond_mask), (
        "video_loss_mask must equal ~conditioning_mask"
    )
    assert torch.all(mi.video.timesteps[cond_mask] == 0), (
        "conditioning tokens must carry timestep 0"
    )


def _fhw(shape: Any) -> tuple[int, int, int]:
    """Extract the trailing ``(frames, height, width)`` latent dims from a shape tuple.

    Accepts ``[..., F, H, W]`` — e.g. patchified/latent tensors as ``[C, F, H, W]`` or
    ``[B, C, F, H, W]`` (or a bare ``(F, H, W)``). Pure stdlib: reads the last three entries so both
    the batched and unbatched latent layouts work without an ltx_core import.
    """
    dims = tuple(int(d) for d in shape)
    if len(dims) < 3:
        raise ValueError(f"shape {dims} must have at least 3 dims (..., F, H, W)")
    return dims[-3], dims[-2], dims[-1]


def infer_reference_downscale_factor(
    ref_hw: tuple[int, int], target_hw: tuple[int, int]
) -> int:
    """Infer the reference downscale factor from ``(H, W)`` pairs (mirror of the canonical helper).

    Ports ``VideoToVideoStrategy._infer_reference_downscale_factor`` (Lightricks/LTX-2 @ d6053703,
    RESEARCH.md lines 357-367): the target dims MUST be exact integer multiples of the reference dims
    and the H/W scale MUST be uniform. Returns 1 on the D-7-REF11 1:1 path; raises a clear
    ``ValueError`` on a non-integer or non-uniform relation. CPU-pure (int math only).
    """
    ref_h, ref_w = int(ref_hw[0]), int(ref_hw[1])
    tgt_h, tgt_w = int(target_hw[0]), int(target_hw[1])
    if tgt_h == ref_h and tgt_w == ref_w:
        return 1  # D-7-REF11: reference is 1:1 with the target.
    if ref_h <= 0 or ref_w <= 0:
        raise ValueError(f"reference dims must be positive, got (H={ref_h}, W={ref_w}).")
    if tgt_h % ref_h or tgt_w % ref_w:
        raise ValueError(
            f"reference/target resolution mismatch: target (H={tgt_h}, W={tgt_w}) is not an exact "
            f"integer multiple of reference (H={ref_h}, W={ref_w})."
        )
    scale_h, scale_w = tgt_h // ref_h, tgt_w // ref_w
    if scale_h != scale_w:
        raise ValueError(
            f"reference scale must be uniform: H scale {scale_h} != W scale {scale_w} "
            f"(ref (H={ref_h}, W={ref_w}) -> target (H={tgt_h}, W={tgt_w}))."
        )
    return scale_h


def assert_paired_reference(
    ref_shape: Any, target_shape: Any, downscale_factor: int
) -> None:
    """Preflight guard (SC#2): assert a paired (reference, target) latent pair is dim-compatible.

    Modal (07-09) calls this BEFORE any GPU spend so a mismatched reference/target bucket (Pitfall 2,
    the doubled-sequence OOM) is rejected cheaply rather than crashing a metered run. The reference is
    ALWAYS temporally frame-aligned to the target, so the frame (length) dims must be EQUAL; the H/W
    dims must be equal for ``downscale_factor == 1`` (D-7-REF11) or exact uniform integer multiples
    otherwise — verified via ``infer_reference_downscale_factor`` and checked against the declared
    factor. Raises a clear ``ValueError`` on any resolution/length mismatch. CPU-pure (shape ints
    only; no ltx_core / modal).
    """
    ref_f, ref_h, ref_w = _fhw(ref_shape)
    tgt_f, tgt_h, tgt_w = _fhw(target_shape)
    if ref_f != tgt_f:
        raise ValueError(
            f"reference/target length mismatch: reference frames {ref_f} != target frames {tgt_f} "
            f"(the reference must be temporally aligned 1:1 with the target)."
        )
    inferred = infer_reference_downscale_factor((ref_h, ref_w), (tgt_h, tgt_w))
    if inferred != downscale_factor:
        raise ValueError(
            f"declared reference_downscale_factor {downscale_factor} != inferred {inferred} from "
            f"reference (H={ref_h}, W={ref_w}) -> target (H={tgt_h}, W={tgt_w})."
        )


def assert_h3_seq_len_budget(cfg: SignetConfig) -> H3DryrunBudget:
    """Preflight guard (H3-07): refuse an H3 packed sequence that exceeds the GPU's row ceiling.

    The same posture as :func:`assert_paired_reference` one level up — raised BEFORE any GPU spend so
    an over-budget geometry is rejected cheaply rather than crashing a metered run, with a message
    naming BOTH sides. CPU-pure: geometry ints only, no ``ltx_core``, no ``modal``, no CUDA.

    This is a NEW CLASS of dry-run assertion. Every existing LTX validator refuses on a divisibility
    rule; none refuses on a VRAM budget. ``P10-1-MEASURED.md`` section 8.5, verbatim: **"This run
    would have been caught locally."**

    ⛔ The refusal itself is DELEGATED to ``config/validators.validate_h3_reference_budget``, so the
    worst-pair enumeration and the refusal message each exist in exactly ONE place. This function is
    the WIRING plus the banner inputs. It never prices a single nominal pair: at reference short edge
    1024 the nominal ``A+B`` pair reports 12,362 rows and PASSES, while six of the twelve
    character-by-environment pairs are over the ceiling — a nominal price passes here and then OOMs
    on the first environment-bearing segment.

    Returns the :class:`H3DryrunBudget` (worst layout, its pair label, the ceiling) for the OK banner.
    """
    budget = _h3_worst_case_budget(cfg)
    h3 = cfg.h3

    if budget.references:
        # THE refusal, in its single home. It re-enumerates the same domain from the same geometry
        # module; the coherence assert below turns that second call from a duplicate into a proof
        # that the banner's numbers and the refusal's numbers cannot diverge.
        priced = validate_h3_reference_budget(
            target_frames=cfg.training_dims[2],
            aspect=h3.target_aspect,
            character_references=h3.character_reference_sizes,
            environment_references=h3.environment_reference_sizes,
            prompt_tokens=h3.prompt_tokens_estimate,
            ref_short_edge=h3.reference_image_short_edge,
            gpu_usable_gib=h3.gpu_usable_gib,
            resident_gib=h3.resident_gib,
            mib_per_packed_row=h3.mib_per_packed_row,
            references_per_sample=h3.references_per_sample,
        )
        assert priced == budget.layout, (
            f"H3 budget incoherence: the shared worst-pair validator priced {priced} while the "
            f"dry-run priced {budget.layout}. Both call conditioning/h3_geometry — they cannot "
            f"legitimately disagree."
        )
    else:
        # Reference-free: exactly ONE possible layout, so worst == nominal trivially (this is not
        # the nominal-pair hole — there is no other pair to be wrong about). Still priced, because
        # campaign target length busts the ceiling with zero references.
        validate_h3_seq_len_budget(budget.layout.total, budget.ceiling_rows)

    return budget


def _h3_ok_banner(cfg: SignetConfig, budget: H3DryrunBudget) -> str:
    """The H3 OK line: geometry, the full packed breakdown, the PAIR label, ceiling and headroom.

    Printing the pair label is what makes the banner actionable when the number sits near the
    ceiling — the operator's next move is to change THAT pair's fidelity, and a bare "N rows of M"
    does not say which one to change.
    """
    frames = cfg.training_dims[2]
    canvas_height, canvas_width = resolve_canvas_size(*cfg.h3.target_aspect)
    aspect_w, aspect_h = cfg.h3.target_aspect
    pair = (
        f"worst pair {budget.worst_pair_label}"
        if budget.worst_pair_label
        else "no references declared"
    )
    return (
        f"[signet-dryrun] OK — H3 config valid, synthetic packed batch built on CPU: "
        f"family={cfg.model.family}, canvas {canvas_width}x{canvas_height} "
        f"(aspect {aspect_w}:{aspect_h}), {frames} target frames -> "
        f"{h3_latent_frames(frames)} latent frames, {budget.layout.describe()}, "
        f"{pair}: packed seq_len={budget.layout.total} / ceiling={budget.ceiling_rows} "
        f"(headroom {budget.headroom_rows} rows). Zero GPU, zero Modal spend."
    )


def run_dryrun(cfg: SignetConfig) -> int:
    """Run the dry-run hard gate on an ALREADY-validated config (CONF-03). Returns 0/non-zero.

    Split out from ``main`` (WR-03) so a caller that already holds a loaded ``SignetConfig`` — e.g.
    the Modal local_entrypoint, which loads once for the cost estimate — can gate the SAME object
    instead of forcing a second independent disk read + re-validate. Removes the TOCTOU gap where
    the cost banner and the gated config could diverge if the file changed between reads.

    Assumes ``cfg`` is already validated (the CONF-02 validators fired at load time). Only the
    synthetic-batch build + contract assertions happen here.

    For ``model.family == "h3"`` the gate additionally REFUSES an over-budget packed sequence
    (H3-07). That check runs FIRST, inside the same try/except, so the refusal becomes a non-zero
    return with the message on stderr — and so an over-budget geometry is rejected without ever
    allocating the batch it could not afford.

    ``model.family == "qwen_edit"`` prices its packed sequence in the same position, with one
    deliberate difference: the refusal is OPT-IN. ``qwen_edit.max_packed_rows`` defaults to 0 =
    ceiling DISABLED, because no MiB-per-row figure has been measured for Qwen-Image-Edit on any
    card in this program and H3's measured triple prices a different model at a different row width.
    The layout is computed and PRINTED regardless; the banner says ``ceiling=DISABLED (unmeasured)``
    so the gap is visible rather than implied by a comfortable-looking headroom number.
    """
    h3_budget: H3DryrunBudget | None = None
    qwen_edit_budget: QwenEditDryrunBudget | None = None
    try:
        if cfg.model.family == "h3":
            h3_budget = assert_h3_seq_len_budget(cfg)
        elif cfg.model.family == "qwen_edit":
            qwen_edit_budget = assert_qwen_edit_row_budget(cfg)
        mi = build_dryrun_inputs(cfg)
        _assert_contract(cfg, mi)
    except AssertionError as exc:
        print(f"[signet-dryrun] dry-run contract assertion FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[signet-dryrun] dry-run FAILED: {exc}", file=sys.stderr)
        return 1

    if h3_budget is not None:
        print(_h3_ok_banner(cfg, h3_budget))
        return 0

    if qwen_edit_budget is not None:
        print(_qwen_edit_ok_banner(cfg, qwen_edit_budget))
        return 0

    width, height, frames = cfg.training_dims
    seq_len = compute_seq_len(width, height, frames)
    print(
        f"[signet-dryrun] OK — config valid, synthetic ModelInputs built: "
        f"dims [W={width}, H={height}, F={frames}] -> seq_len={seq_len}, "
        f"video.latent {tuple(mi.video.latent.shape)}. Zero GPU, zero Modal spend."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the dry-run hard gate from a config PATH. Returns 0 on success, non-zero otherwise (D-13).

    Two invocation forms exit identically: the ``signet-dryrun`` console script and
    ``python -m signet_trainer.dryrun <config.yaml>``. This is the file-reading wrapper around
    ``run_dryrun``; callers that already hold a validated ``SignetConfig`` should call
    ``run_dryrun(cfg)`` directly (WR-03) to avoid a redundant/divergent second load.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: signet-dryrun <config.yaml>", file=sys.stderr)
        return 2

    config_path = Path(args[0])

    # (1) Load + validate. A bad frame count raises here -> caught -> non-zero exit (D-13).
    try:
        cfg = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 — the gate must turn ANY load error into a non-zero exit.
        print(f"[signet-dryrun] config validation FAILED: {exc}", file=sys.stderr)
        return 1

    # (2-4) Build the synthetic batch + assert the real contract on the loaded config.
    return run_dryrun(cfg)


if __name__ == "__main__":
    sys.exit(main())
