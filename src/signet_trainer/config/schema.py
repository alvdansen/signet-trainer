"""SignetConfig — the composing (not subclassing) Pydantic v2 run-config schema (CONF-01, D-04/D-05).

signet carries LTX fields as plain DATA + signet-owned blocks. It deliberately does NOT
instantiate the native ``ltx_trainer.config.LtxTrainerConfig`` locally — that class's
``model_path`` / ``images`` / ``reference_videos`` validators call ``Path(...).exists()``
at load time (RESEARCH.md Pitfall 1), which would break the Windows/zero-GPU dry-run.
The real ``LtxTrainerConfig`` is hydrated Modal-side at run time (Phase 2+), where the
weights actually exist on the mounted Volume.

Field NAMES mirror the native ``LtxTrainerConfig`` where signet carries them as data, so
Phase 2 can build the real object Modal-side without renaming.

CRITICAL — Anti-Pattern 6 / Pitfall 1/4:
    This module imports ONLY pydantic + the signet validators (stdlib + the shared seq-len
    helper). It MUST NOT import ``modal``, ``ltx_core``, or ``ltx_trainer``, and MUST NOT
    touch the filesystem during validation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from signet_trainer.config.validators import (
    H3_A100_80GB_USABLE_GIB,
    H3_CAMPAIGN_ASPECT,
    H3_CANVAS_MULTIPLE,
    H3_LORA_TARGET_REGEX,
    H3_MIB_PER_PACKED_ROW,
    H3_NOMINAL_PROMPT_TOKENS,
    H3_PHASE10_REFERENCE_SHORT_EDGE,
    H3_PHASE10_REFERENCES_PER_SAMPLE,
    H3_RESIDENT_GIB_RANK64,
    H3_VISUAL_CONDITION_PIN,
    QWEN_EDIT_CONDITION_IMAGE_SIZE,
    QWEN_EDIT_LORA_TARGET_REGEX,
    QWEN_EDIT_MAX_CONTROL_SLOTS,
    QWEN_EDIT_VAE_IMAGE_SIZE,
    h3_packed_seq_len,
    max_packed_rows_for_budget,
    qwen_edit_packed_layout,
    validate_a2v_lora_targets,
    validate_batch_size,
    validate_conditioning_items,
    validate_conditioning_mode,
    validate_h3_frames,
    validate_h3_reference_budget,
    validate_h3_resolution_buckets,
    validate_h3_seq_len_budget,
    validate_height,
    validate_inpaint_dims,
    validate_inpaint_resolution_buckets,
    validate_qwen_edit_frames,
    validate_qwen_edit_lora_coverage,
    validate_qwen_edit_rank_alpha_lock,
    validate_qwen_edit_resolution_buckets,
    validate_qwen_edit_row_budget,
    validate_resolution_buckets,
    validate_training_dims,
    validate_volume_relative_path,
    validate_width,
)

# The LTX-2.3 LoRA target set, preserved as a module constant so the value survives
# ``LoraConfig.target_modules`` becoming an OPTIONAL override (Phase 10, H3-02). attn1 + attn2 +
# ff.net, full module paths; mirrors ``lora/peft.py::P1_FF_LORA_TARGETS``.
#
# SCAFFOLD-BUG FIX (kept): the old attn-only default silently under-fit identity capacity
# (HANDOFF.md:19 — exactly the mistake that corrupted the prior project's likeness). ``ff.net`` is
# ~2/3 of block params and is where identity is encoded, so it MUST stay targeted.
LTX_DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)

# Which model FAMILIES each family-only ``ModelConfig`` ID is legal under (the lean field-split, per
# field rather than per block). Set one of these under a family that is not in its allowlist and the
# value would be SILENTLY IGNORED — the same class of defect the h3/qwen_edit block reverse guards
# kill, so it dies at config load the same way.
#
# ⚠ Why this replaced a flat ``("vae_id", "audio_vae_id")`` H3-only tuple: Qwen-Image-Edit needs a
# ``vae_id`` too (``qwen_image_vae.safetensors``), so "not h3 -> reject vae_id" became wrong the
# moment family #3 landed. Getting this wrong in the PERMISSIVE direction (widening to every family
# rather than to a named set) would silently hand LTX a no-op knob back, which is why this is an
# allowlist per field and not a family check per call site.
#
# Closed for free on the way past: ``pipeline_root_id`` was never in the old tuple at all, so an LTX
# config could set it with no error and no consumer (``modal/fns.py:4432`` reads it only on the h3
# sample path).
_FAMILY_ONLY_MODEL_IDS: dict[str, frozenset[str]] = {
    "vae_id": frozenset({"h3", "qwen_edit"}),
    "audio_vae_id": frozenset({"h3"}),
    # qwen_edit added 2026-08-08: the field now has a consumer on this family, which is the exact
    # condition the comment above records for inclusion. The Qwen2.5-VL PROCESSOR is a pipeline-root
    # component — a Qwen-Image-Edit-2511 snapshot puts preprocessor_config.json in `processor/`,
    # NOT in `text_encoder/` — so composing its path from text_encoder_id raises
    # "Can't load image processor for .../text_encoder". Measured on a live A100 run.
    "pipeline_root_id": frozenset({"h3", "qwen_edit"}),
}


class _Base(BaseModel):
    """Base with ``extra="forbid"`` so a malformed/hand-edited YAML is rejected (T-01-CF1)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------------
# LTX field blocks — carried as DATA (NOT a constructed LtxTrainerConfig). Names mirror the native
# config so Phase 2 hydrates the real object Modal-side without renaming.
# --------------------------------------------------------------------------------------------------


class DataConfig(_Base):
    """Mirrors native ``DataConfig`` (the only native sub-config without a default factory)."""

    preprocessed_data_root: str = Field(
        ...,
        description="Path to the preprocessed-latents root. A STRING here — NOT FS-validated "
        "locally; existence is checked Modal-side where the Volume is mounted (Pitfall 1).",
    )
    # D-8-PREPROC — the metadata.jsonl the gated `--mode preprocess` arm encodes. A STRING carried
    # as DATA (read Modal-side by fns.preprocess, NOT FS-checked here — Pitfall 1). Defaulted to the
    # Phase-2 fresh set (matching fns.preprocess's own default) so existing train/sample configs that
    # never set it still load unchanged.
    metadata_path: str = Field(
        default="dataset/fresh/metadata.jsonl",
        description="metadata.jsonl to encode via `--mode preprocess`; read Modal-side, not "
        "FS-checked here (Pitfall 1). Matches the fns.preprocess fresh default (backward-compat).",
    )
    num_dataloader_workers: int = Field(default=2, ge=0)
    batch_size: int = Field(
        default=1,
        description="Single-bucket; must be 1 (multi-bucket has per-sample shapes).",
    )
    resolution_buckets: list[str] = Field(
        # D-05: the full multi-bucket set 768x352x{25,49,81} from day one (no minimal-first
        # subset). WxHxF CLI strings — parsed Modal-side via parse_resolution_buckets() to
        # (F, H, W) tuples. WR-04: the STRINGS are now shape-validated HERE at config load
        # (W%32 / H%32 / (F-1)%8), not just carried as opaque DATA — a malformed/mis-ordered
        # bucket must die pre-approval, before any metered container.
        default_factory=lambda: ["768x352x25", "768x352x49", "768x352x81"],
        description="WxHxF bucket strings (enochiatron set); shape-validated at load (WR-04), "
        "parsed Modal-side to (F,H,W).",
    )

    @field_validator("batch_size")
    @classmethod
    def _check_batch_size(cls, v: int) -> int:
        return validate_batch_size(v)

    @field_validator("resolution_buckets")
    @classmethod
    def _check_resolution_buckets(cls, v: list[str]) -> list[str]:
        # WR-04: fail-fast at config load (step 1, pre-approval) — parse each WxHxF and enforce the
        # W%32 / H%32 / frame-law rules, so a burned-approval or garbage-bucket metered dispatch can
        # never happen. The entrypoint's _parse_resolution_buckets then operates on known-good strings.
        #
        # Phase 10: this is now a PRE-SCREEN that accepts a bucket valid under EITHER family's frame
        # law, because ``DataConfig`` is a sub-model and cannot see ``model.family``. The
        # FAMILY-EXACT law is re-asserted in ``SignetConfig._cross_field_checks``, which does know the
        # family — so nothing is weakened at the SignetConfig level (an LTX config with H3 buckets
        # still dies at load, with the LTX message). A bucket invalid under BOTH laws still dies
        # right here, unchanged, and the LTX message is what an operator sees.
        try:
            validate_resolution_buckets(v)
        except ValueError as ltx_error:
            try:
                validate_h3_resolution_buckets(v)
            except ValueError:
                raise ltx_error from None
        return v


class LoraConfig(_Base):
    """Mirrors native ``LoraConfig`` defaults (rank/alpha/target_modules)."""

    # rank == alpha -> PEFT scale 1.0 -> clean PEFT->LTX key conversion (HANDOFF.md). 64 is the operator's
    # choice for more identity capacity; the prior project's validated likeness value was 42 (03-CONTEXT D-OFF).
    rank: int = Field(default=64, ge=2)
    alpha: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    # Phase 10 (H3-02): an OPTIONAL OVERRIDE, no longer a hard LTX default. ``None`` means "use the
    # family default", which ``SignetConfig`` resolves (it is the only object that knows the family)
    # and writes back in place, so every downstream consumer of ``cfg.lora.target_modules`` stays
    # family-correct without having to learn a new accessor.
    #
    # ⚠ Why this is a CORRECTNESS change, not an ergonomic one: the LTX-shaped list does NOT fail
    # loud on H3. H3's ``ff.net.0.proj`` / ``ff.net.2`` leaf names are byte-identical to LTX's, so
    # the LTX default still matches 104 H3 modules (100 main + 4 token_refiner). ``train/loop.py``'s
    # "No trainable parameters found. Is LoRA applied?" guard only fires on an EMPTY trainable set,
    # so it never fires — and a metered A100 run proceeds on an attn-blind, refiner-polluted,
    # ~1/3-capacity adapter that produces plausible-but-wrong output. There is no runtime backstop
    # behind this default; family selection is what moves that failure to config load.
    # Pinned by ``tests/test_h3_lora_targets.py::test_ltx_default_on_h3_fails_silently_not_loud``.
    #
    # A bare ``str`` is a PEFT target REGEX (H3's form, matched with ``re.fullmatch``); a list is the
    # LTX suffix form. Both are legal operator overrides.
    target_modules: list[str] | str | None = Field(
        default=None,
        description="LoRA target override. None (the default) selects the FAMILY default: the H3 "
        "path regex when model.family == 'h3', else the ten LTX suffixes "
        "(LTX_DEFAULT_LORA_TARGETS). A list is the LTX suffix form; a bare string is a PEFT target "
        "regex. Resolved and written back by SignetConfig at config load.",
    )


class ModelConfig(_Base):
    """Mirrors native ``ModelConfig`` — model IDs carried as DATA (native object hydrated Modal-side).

    The IDs are string filenames/dir-names under the Modal ``WEIGHTS_DIR``; they are NOT
    FS-validated locally (Pitfall 1 — existence is checked Modal-side where the Volume mounts).
    """

    model_id: str = Field(
        default="ltx-2.3-22b-dev.safetensors",
        description="Checkpoint filename under WEIGHTS_DIR — read Modal-side, never FS-checked here.",
    )
    text_encoder_id: str = Field(
        # [A1] Pin the full `-it` variant to match enochiatron's validated loader and keep the
        # D-02 oracle a true numerical diff. CLAUDE.md lists the `-qat-q4_0-unquantized` variant;
        # A1 deliberately diverges from it — flag the discrepancy to the operator before any infer phase.
        default="gemma-3-12b-it",
        description="[A1] Gemma dir under WEIGHTS_DIR; CLAUDE.md lists the -qat-q4_0 variant.",
    )
    # Phase 10 (H3-02): the MODEL-FAMILY discriminator. An EXPLICIT field, deliberately NOT a
    # filename sniff — ``models/loader.py::base_variant_of()`` classifies the LTX dev/distilled
    # variants by inspecting the ``.safetensors`` FILENAME, and that technique does not transfer:
    # H3's model IDs are DIRECTORIES under WEIGHTS_DIR (``minimax-h3/transformer_ref``,
    # ``minimax-h3/text_encoder``, ``minimax-h3/vae``, ``minimax-h3/audio_vae``), so there is no
    # suffix to sniff and any heuristic would be a guess. Defaults to ``ltx`` so every pre-Phase-10
    # config loads byte-identically.
    family: Literal["ltx", "h3", "qwen_edit"] = Field(
        default="ltx",
        description="Model FAMILY discriminator ('ltx' | 'h3' | 'qwen_edit'). Selects the LoRA "
        "target form (LTX suffix list vs the H3 path regex vs the Qwen 14-leaf path regex) and "
        "which frame law / budget checks run at config load. Explicit by design: H3 model IDs are "
        "directories and Qwen-Image-Edit's is a bare .safetensors filename indistinguishable in "
        "SHAPE from LTX's, so there is nothing to sniff in either case.",
    )
    vae_id: str | None = Field(
        default=None,
        description="H3 / qwen_edit: VAE dir or file under WEIGHTS_DIR (e.g. 'minimax-h3/vae', "
        "'qwen_image_vae.safetensors'). None for LTX, whose VAE ships inside the single checkpoint. "
        "Legal families are declared in _FAMILY_ONLY_MODEL_IDS. DATA only — never FS-checked here "
        "(Pitfall 1).",
    )
    audio_vae_id: str | None = Field(
        default=None,
        description="H3-only: audio VAE dir under WEIGHTS_DIR (e.g. 'minimax-h3/audio_vae'). None "
        "for LTX. DATA only — never FS-checked here (Pitfall 1).",
    )
    # D-10-DEF-14. A DISTINCT field, deliberately NOT a second meaning for ``model_id`` and
    # deliberately NOT derived from it. ``model_id`` names the transformer PARTITION
    # (``minimax-h3/transformer_ref``) — the meaning ``h3_train`` and ``h3_loader`` depend on, proven
    # across five dispatches — and a pipeline needs the ROOT that holds every partition plus the
    # index. Deriving the root as ``Path(model_id).parent`` would work today and would be a
    # convention no config states and nothing checks; naming it is one line and cannot drift.
    pipeline_root_id: str | None = Field(
        default=None,
        description="h3 / qwen_edit: the diffusers pipeline ROOT dir under WEIGHTS_DIR (e.g. "
        "'minimax-h3', 'qwen-image-edit-2511') — the directory holding model_index.json and every "
        "component partition. NOT the same as model_id, which names the transformer partition "
        "INSIDE this root. On h3: required by `--mode sample`, unused by train/preprocess. On "
        "qwen_edit: required by preprocess and train, for TWO distinct reasons — (1) the "
        "Qwen2.5-VL PROCESSOR lives in the root's `processor/` subfolder, not beside the text "
        "encoder, and (2) it is the `config_source` a single-file transformer load needs, because "
        "diffusers' infer_diffusers_model_type has no Qwen branch on the pinned version. DATA "
        "only — never FS-checked here (Pitfall 1).",
    )


class H3Config(_Base):
    """MiniMax-H3 (Hailuo 3.0) tunables — Phase 10, every locked D-10-* decision as a FIELD.

    D-NOHARDCODE: none of these may ever be a literal in code. Each one is documented, defaulted to
    the decided value, and fail-fast validated BEFORE any GPU is touched. The measured budget triple
    (``gpu_usable_gib`` / ``resident_gib`` / ``mib_per_packed_row``) is config-first precisely so an
    H200 escalation is a YAML edit rather than a code change.

    Only meaningful when ``model.family == 'h3'``; ``SignetConfig`` carries the bidirectional lean
    field-split REVERSE guard that rejects a non-default value under any other family, so this block
    can never be silently ignored.

    Deliberately NOT a field here: any ``arch_smoke_only``-style mode switch. ``run_h3_arch_gate``
    fires unconditionally at the front of BOTH the H3 preprocess and train dispatches, so
    abort-before-spend is already guaranteed by every real dispatch — and P10-1 proved the
    architecture on live weights (10/10 constants, 300/300 targets against real ``named_modules``).
    A knob here would let a domain/recipe config block change a Modal function's OPERATIONAL mode.
    """

    reference_image_short_edge: int = Field(
        default=H3_PHASE10_REFERENCE_SHORT_EDGE,
        ge=256,
        description="Short edge each reference IMAGE is re-encoded at. The true H3 spec is 2048; "
        "Phase 10 runs 896 because the WORST reference pair (C+008) is 12,394 packed rows there — "
        "within 0.3% of the 12,362-row configuration measured PASSING on a real A100 at 76.36 GiB — "
        "while at 1024 it is 14,026 and six of the twelve character-by-environment pairs exceed the "
        "ceiling. VAE latents cannot be spatially downscaled after the fact, so a higher-fidelity "
        "campaign needs a full re-encode; budget for it. Must be a multiple of 32.",
    )
    reference_dropout: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="D-10-REFDROP: P(drop the reference conditioning for a training sample). Field "
        "precedent is ~10%; ours is deliberately more aggressive on a small corpus. If the reference "
        "pathway UNDERFITS, this is the first suspect.",
    )
    reference_pair_seed: int = Field(
        default=42,
        description="D-10-PAIRSEED: seed for the fixed round-robin reference-pair assignment, so "
        "which refs a sample gets is reproducible and debuggable across runs. Revisitable.",
    )
    references_per_sample: int = Field(
        default=H3_PHASE10_REFERENCES_PER_SAMPLE,
        description="The reference SLOT COUNT. 2 (the default) is the Ref2VA operator ruling: a "
        "non-environment segment gets two rotating character refs; an environment segment gets one "
        "rotating character ref plus the environment ref, which SUBSTITUTES for the second character "
        "slot rather than being appended. Feeding all three character refs every time would make "
        "conditioning CONSTANT across the corpus, so the model could not learn what varies and would "
        "be free to copy the reference wholesale (the [precedent] copy-collapse failure) — rotating "
        "pairs make identity the INVARIANT. D-10-ASYM is still honored: the reference REGIME "
        "(character+character vs character+environment) still varies; only the COUNT is fixed. "
        "0 = NO-REFERENCE training (ALPHA — smoke-tested only, no end-to-end run exists): no slot "
        "resolution, no ref latent rows, no Qwen vision blocks, loss over every target row; the "
        "ref-only knobs (reference_dropout etc.) must stay at their defaults (reverse guard below). "
        "PR #6 (single-control) adds 1 — the allowlist unions to {0, 1, 2} on merge.",
    )
    environment_ref_last: bool = Field(
        default=True,
        description="D-10-REFORDER: the environment reference always occupies the LAST slot. H3 "
        "imposes no slot semantics, but order IS load-bearing twice over — it fixes the <Picture i> "
        "labels and advances the shared rotary clock — so a different order is a different request. "
        "The house order must be applied CONSISTENTLY between training and inference.",
    )
    prompt_tokens_estimate: int = Field(
        default=H3_NOMINAL_PROMPT_TOKENS,
        ge=1,
        description="Token budget the packed-seq_len preflight assumes for the caption. An ESTIMATE "
        "used ONLY for the budget check — it is never the real tokenization, which happens Modal-side "
        "against the actual text encoder.",
    )
    text_encoder_layer: int = Field(
        default=50,
        ge=0,
        description="Qwen3-VL hidden_states[50] of its 64 layers is the text-conditioning source. "
        "The FINAL layer is post-norm and off-distribution — a final-layer pre-encode is silently "
        "wrong at the correct shape. NOTE: the 50 here is a numerical COINCIDENCE with the DiT's "
        "num_layers = 50; the two must never be conflated.",
    )
    t_visual_cond: float = Field(
        default=H3_VISUAL_CONDITION_PIN,
        ge=0.0,
        le=1.0,
        description="D-10-REFPIN: the TRAINING-time noise level the visual reference rows are "
        "pinned at, applied as max(t_video, t_visual_cond). NOTE H3's INVERTED time convention "
        "(t = 1 - sigma, so t = 1 is CLEAN and x_t = t*x0 + (1-t)*eps): 0.999 means 'almost clean', "
        "not 'almost pure noise'. The default is the imported inference constant "
        "(train/h3_step.H3_VISUAL_CONDITION_PIN, itself modular_pipeline.py::keyframe_noise_aug) so "
        "the harvest is faithful out of the box — but D-10-REFPIN is explicit that the SAMPLER's "
        "anchor level and the TRAINING augmentation are separate decisions, so this is a field "
        "rather than a constant. Lowering it augments the references more aggressively, which is a "
        "copy-collapse lever alongside reference_dropout; latent-noise 'poisoning' beyond this pin "
        "is deliberately NOT how this codebase regularizes references (P10-0c: the field "
        "regularizes in IMAGE space — jitter, dropout, cross-pair construction).",
    )
    audio_in_loss: bool = Field(
        default=False,
        description="D-10-AUDIO: whether target audio rows participate in the loss. Video-only means "
        "loss-MASKING, never architecture-skipping — H3 has no separate audio branch (audio is "
        "entangled through all 50 blocks). Target audio rows stay PRESENT and NOISED, matching the "
        "inference regime, and merely stay OUT of the loss. Do not teach the model to be silent.",
    )
    gpu_usable_gib: float = Field(
        default=H3_A100_80GB_USABLE_GIB,
        gt=0.0,
        description="MEASURED (P10-1b) usable VRAM on the target GPU. Config-first so an H200 "
        "escalation is a YAML edit, not a code change.",
    )
    resident_gib: float = Field(
        default=H3_RESIDENT_GIB_RANK64,
        gt=0.0,
        description="MEASURED (P10-1b) resident footprint: 61.73 GiB weights + the rank-64 LoRA "
        "inject. A different LoRA rank simply declares a different number.",
    )
    mib_per_packed_row: float = Field(
        default=H3_MIB_PER_PACKED_ROW,
        gt=0.0,
        description="MEASURED (P10-1b) marginal activation cost per packed row, gradient "
        "checkpointing ON. With the two fields above this yields the packed-row ceiling.",
    )
    target_aspect: tuple[int, int] = Field(
        default=H3_CAMPAIGN_ASPECT,
        description="Canvas aspect the packed-seq_len preflight resolves (short edge 768, area cap "
        "768*1344, each axis snapped to 32). 16:9 -> 1344x768 = 1008 rows per latent frame.",
    )
    character_reference_sizes: list[tuple[int, int] | tuple[int, int, str]] = Field(
        default_factory=list,
        description="(width, height[, label]) of each CHARACTER reference AFTER D-10-CROP. The "
        "operator's picks: IMG_3659 832x1248 (A), IMG_3725 2048x2048 (B), tmppvclbw6u 1037x1536 "
        "cropped to 1024 wide (C). Crop NEVER pad — padding injects a synthetic border the adapter "
        "would learn — and the crop must save the face. Labels flow into budget-refusal messages.",
    )
    environment_reference_sizes: list[tuple[int, int] | tuple[int, int, str]] = Field(
        default_factory=list,
        description="(width, height[, label]) of each ENVIRONMENT reference AFTER D-10-CROP: 029 "
        "1344x768, 000 1024x1024, 008 1456x816 cropped to 1440x800, 023 1344x768. SPLIT from the "
        "character list rather than one flat list because the budget validator must enumerate the "
        "REAL pairing domain (3 character pairs + 12 character-by-environment pairs = 15); a single "
        "flat list cannot express which images can pair with which, and pricing one nominal pair is "
        "exactly the failure H3-04 exists to prevent.",
    )

    # ── RENDER-side knobs (h3_sample only) ────────────────────────────────────────────────────────
    # These live on the h3 block rather than on `validation` on purpose. `validation` is shared with
    # the LTX family, and the LTX `sample` path consumes none of them — a field there would be
    # SILENTLY IGNORED under LTX, which is the exact anti-pattern the lean field-split exists to
    # prevent. Here the REVERSE guard on SignetConfig refuses a non-default value under any non-h3
    # family, so setting one on an LTX config is a config-load error rather than a no-op.
    render_merge_adapter: bool = Field(
        default=True,
        description="D-10-DEF-19: MERGE the LoRA delta into the base weights before the ADAPTER "
        "render column, so the adapted forward costs exactly what the base forward costs. MEASURED "
        "(render app ap-ENw0kWTvmlfm4WhoYtc7KT, 6/6 base clips written then 0/6 adapter clips): the "
        "adapter column died in transformer block 0 of denoise step 0 trying to allocate 6.88 GiB "
        "with 6.70 GiB free / 72.54 GiB in use and only 103 MiB reserved-but-unallocated (so not "
        "fragmentation), and the failing frame was `lora_B`'s own F.linear inside "
        "`result + lora_B(lora_A(dropout(x))) * scaling` (peft 0.20.0 lora/layer.py:1058) beneath "
        "the SwiGLU up-projection `ff.net.0.proj` (diffusers activations.py:144). INFERRED from that "
        "expression, not measured: the unmerged path needs roughly TWICE the failing allocation in "
        "extra headroom, since `lora_B`'s output, the `* scaling` temporary and the `+` result are "
        "all the shape of the base output — none of which the base path allocates at all. Merged, "
        "PEFT takes its `elif self.merged: result = self.base_layer(x)` branch, which is byte-for-"
        "byte the base column's allocation profile. Set False only on a card with well over ~14 GiB "
        "of spare headroom beyond the base column's measured peak, where the unmerged path's "
        "slightly higher arithmetic precision (the delta accumulates in activation space rather "
        "than through one bf16 weight-space add) is worth the VRAM.",
    )
    render_offload_reserve: str = Field(
        default="12GB",
        description="`memory_reserve_margin` handed to diffusers' "
        "`ComponentsManager.enable_auto_cpu_offload`. ⚠ VERIFIED SEMANTICS, because the obvious "
        "reading is wrong and cost a render: this is NOT a standing VRAM reservation. It is a "
        "one-shot EVICTION THRESHOLD consulted only inside `CustomOffloadHook.pre_forward`, and only "
        "when the component being called is not already on the execution device "
        "(components_manager.py:86-119). `AutoOffloadStrategy` subtracts it from `mem_get_info()[0]` "
        "to decide WHICH OTHER components to push to CPU (:189); once the component is resident, "
        "activations are free to consume every remaining byte. Raising it cannot rescue an "
        "activation-peak OOM — when the strategy already logs `no combination of models to offload "
        "to cpu is found, offloading all models` (:233) it has evicted everything there is to evict "
        "and its effect is maximal. Units follow huggingface `convert_file_size_to_int`: a `GB` "
        "suffix is 10^9 bytes, `GIB` is 2^30 — they are NOT the same number.",
    )
    render_checkpoint_name: str = Field(
        default="",
        description="PIN the checkpoint DIRECTORY NAME the render probes (e.g. "
        "'checkpoint-step-00300-loss-0.0890'), relative to CHECKPOINTS_DIR/<output_dir>. Empty "
        "(default) = `CheckpointManager.find_latest()`, the historical behaviour, byte-identically. "
        "Why this exists: the render directory is keyed on the render's IDENTITY, and the checkpoint "
        "name is part of that key — so while a training run is LIVE, `find_latest()` resolves a new "
        "checkpoint every `checkpoint_every` steps and every re-dispatch lands in a FRESH directory "
        "with nothing to resume. The identity-keyed resume is correct; it just cannot survive a "
        "moving adapter. Pinning makes a re-dispatch continue the render it started. There is "
        "deliberately no LTX equivalent yet — the LTX `sample` path still resolves find_latest only.",
    )

    @field_validator("render_offload_reserve")
    @classmethod
    def _check_render_offload_reserve(cls, v: str) -> str:
        # Parsed Modal-side by `convert_file_size_to_int`, which raises a bare ValueError deep inside
        # enable_auto_cpu_offload — i.e. AFTER eight components (~134 GiB of reads) have loaded. A
        # typo is a config error; make it one, here, for free.
        if not re.fullmatch(r"\d+(\.\d+)?\s*(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)", v.strip().upper()):
            raise ValueError(
                f"h3.render_offload_reserve={v!r} is not a size huggingface's "
                "convert_file_size_to_int can parse. Expected a number plus one of "
                "B/KB/MB/GB/TB/KIB/MIB/GIB/TIB (e.g. '12GB', '11.2GiB'). Note GB is 10^9 and GIB is "
                "2^30 — they are different numbers."
            )
        return v

    @field_validator("render_checkpoint_name")
    @classmethod
    def _check_render_checkpoint_name(cls, v: str) -> str:
        # A NAME, never a path. `ckpt_root / <value>` with an absolute value REPLACES the prefix and
        # a '..' escapes the mount (the WR-09 pathlib-join contract every operator-authored path in
        # this schema obeys). Empty is the documented "use find_latest" sentinel.
        if not v:
            return v
        if v != Path(v).name or v in {".", ".."}:
            raise ValueError(
                f"h3.render_checkpoint_name={v!r} must be a bare directory NAME under "
                "CHECKPOINTS_DIR/<output_dir> (e.g. 'checkpoint-step-00300-loss-0.0890'), never a "
                "path. pathlib join semantics let an absolute value replace the Volume prefix and "
                "'..' escape it."
            )
        return v

    @field_validator("reference_image_short_edge")
    @classmethod
    def _check_reference_short_edge(cls, v: int) -> int:
        # Both sides of the H3 arch snap every axis to 32 (vae_spatial_compression 16 * patch 2), so
        # a non-multiple silently re-rounds and the config's declared fidelity is a lie.
        if v % H3_CANVAS_MULTIPLE != 0:
            raise ValueError(
                f"invalid reference_image_short_edge {v}: must be a multiple of "
                f"{H3_CANVAS_MULTIPLE} (vae_spatial_compression 16 * patch_size 2). H3 rounds every "
                f"axis to {H3_CANVAS_MULTIPLE} internally, so a non-multiple would silently encode "
                f"at a different size than the config declares. Nearest valid: "
                f"{max(H3_CANVAS_MULTIPLE, round(v / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)}."
            )
        return v

    @field_validator("references_per_sample")
    @classmethod
    def _check_references_per_sample(cls, v: int) -> int:
        # THE UNION, LIVE (2026-08-11): #6 (single-control, adds 1) merged to main and the no-ref
        # ALPHA (adds 0) rebased onto it -- the documented one-token union {0, 1, 2}.
        # 1 -- single-control tasks (operator ruling 2026-08-07: hero keyframe model, one start
        #      extreme in / one end extreme out); everything downstream already handled 1 and a
        #      single reference is strictly CHEAPER (5,040 vs 7,848 packed rows at 16:9).
        # 0 -- NO-REFERENCE training (ALPHA -- smoke-tested only: one 50-step metered smoke, no
        #      end-to-end run; --mode sample refused at 0 pending the t2va render leg).
        # 3 remains refused, and that bound carries the real weight: three slots were never priced,
        # and an environment reference SUBSTITUTES for the last character slot rather than being
        # appended -- feeding all three character refs every time would make conditioning CONSTANT
        # across the corpus and invite copy-collapse.
        if v not in (0, 1, H3_PHASE10_REFERENCES_PER_SAMPLE):
            raise ValueError(
                f"invalid references_per_sample {v}: the slot count is 0 (NO-REFERENCE training, "
                f"ALPHA), 1 (single-control), or {H3_PHASE10_REFERENCES_PER_SAMPLE} (Ref2VA -- a "
                f"non-environment segment gets two rotating character refs; an environment segment "
                f"gets one rotating character ref plus the environment ref, which SUBSTITUTES for "
                f"the second character slot). There is no 3-reference case: three slots were never "
                f"priced by the packed-sequence budget and would OOM a metered container."
            )
        return v

    @model_validator(mode="after")
    def _check_no_reference_fields(self) -> "H3Config":
        # NO-REFERENCE (ALPHA) reverse guard — the same bidirectional lean field-split shape as the
        # ic_lora / inpaint / a2v guards on their blocks: a ref-only knob set while the slot count
        # is 0 would be SILENTLY ignored (there are no slots for it to act on), so it dies at
        # config load naming the offending field(s). The budget triple / prompt estimate / aspect /
        # audio_in_loss / text_encoder_layer still act at 0 and stay free. The three render_* knobs
        # (render_checkpoint_name / render_merge_adapter / render_offload_reserve) are DELIBERATELY
        # left free even though --mode sample is REFUSED at 0 slots today: they are inert until the
        # t2va render leg lands (the tracked no-ref follow-up), and rejecting them would force a
        # config edit the moment it does. Inert-but-free is a documented choice here, not drift.
        if self.references_per_sample != 0:
            return self
        # Defaults are read off the FIELD DEFINITIONS (not a pristine instance — instantiating one
        # inside its own model_validator would recurse), so a default change is covered automatically
        # and the guard cannot drift out of sync (the T-10-05-T doctrine, same as the family guard).
        ref_only = (
            "reference_image_short_edge",
            "reference_dropout",
            "reference_pair_seed",
            "environment_ref_last",
            "t_visual_cond",
            "character_reference_sizes",
            "environment_reference_sizes",
        )
        nondefault = [
            name
            for name in ref_only
            if getattr(self, name) != H3Config.model_fields[name].get_default(call_default_factory=True)
        ]
        if nondefault:
            raise ValueError(
                f"H3 reference field(s) {nondefault} set while references_per_sample is 0 "
                f"(NO-REFERENCE, ALPHA): with zero reference slots these knobs act on nothing and "
                f"would be silently ignored (lean field-split — no silently-ignored config block). "
                f"Remove them, or set references_per_sample: 2 for Ref2VA."
            )
        return self


class QwenEditRenderInput(_Base):
    """One HELD-OUT control input for the §8 render grid: its images and its A/B prompts travel TOGETHER.

    An OBJECT, for the ``ValidationSample`` / ``ConditioningItem`` reason (D-6-ITEMS) and one more
    that is specific to this grid. §8's read is the A-vs-B delta at a fixed checkpoint — (A) the
    subject withheld from the prompt, (B) the subject named — so a mis-pairing does not degrade the
    measurement, it INVERTS it: the trace-vs-reinterpret verdict gets attributed to the adapter when
    it belongs to the prompt. Parallel ``images`` / ``prompt_a`` / ``prompt_b`` lists can desync
    across a config edit and produce exactly that, at a grid that looks perfectly ordinary.

    The field names are the reader's, not this file's preference: ``modal/fns.py::qwen_edit_sample``
    and ``modal/entrypoint.py::_qwen_edit_config_gaps`` both reach for ``id`` / ``images`` /
    ``prompts`` / ``label`` by ATTRIBUTE (``getattr(entry, "images", ())``), and
    ``inference/qwen_edit_layout.QwenEditHeldOutInput`` consumes the same four. Nothing translates
    between the config and the planner, which is the property that keeps a rename from producing a
    config the entrypoint accepts and the container refuses.
    """

    id: str = Field(
        ...,
        min_length=1,
        description="Stable id for this held-out input. Load-bearing TWICE: it is one slot of the "
        "render key's control axis, and it is the stem of every file this input renders — so two "
        "inputs sharing an id overwrite each other inside one render directory and the grid shows "
        "one input's pixels twice under two labels.",
    )
    images: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="The control images for this input, one per control slot, IN SLOT ORDER. "
        "POSITIONAL: image i fills slot i, which is what the prompt's ctrl_img_{i+1} addresses. "
        "Relative paths resolve under the dataset Volume mount; an absolute path is accepted as-is "
        "(the h3_sample treatment of data.metadata_path). The count must equal "
        "qwen_edit.control_slots — enforced by QwenEditConfig._check_render_input_slot_count, where "
        "both values are visible, and mirrored by the entrypoint gap check and the container.",
    )
    prompts: dict[str, str] = Field(
        ...,
        description="Mode id -> prompt text, for EVERY mode in "
        "inference/qwen_edit_layout.QWEN_EDIT_PROMPT_MODES (a_style, b_content). Keyed by mode id "
        "rather than carried as two fields so that tuple stays the single place a mode is declared.",
    )
    label: str | None = Field(
        default=None,
        description="Optional row header an operator reads in the gallery. Defaults to the id.",
    )

    @field_validator("images")
    @classmethod
    def _check_images(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        # WR-09: a relative path is joined onto the dataset mount, so '..' escapes it. Absolute is
        # deliberately allowed (the reader tests is_absolute() and honours it) — silently resolving
        # an absolute path under the mount would 404 for a reason nobody could read.
        for raw in v:
            if not str(raw).strip():
                raise ValueError(
                    "qwen_edit.render_inputs[].images carries an empty path. Every control slot "
                    "names a real file; a blank entry would fill slot i with nothing and shift "
                    "every later image one slot left, rendering the WRONG request under the right "
                    "label."
                )
            if ".." in Path(str(raw)).parts:
                raise ValueError(
                    f"qwen_edit.render_inputs[].images entry {raw!r} contains '..'. Relative "
                    "control paths are joined onto the dataset Volume mount, and pathlib join "
                    "semantics let '..' escape it."
                )
        return v

    @field_validator("prompts")
    @classmethod
    def _check_prompts(cls, v: dict[str, str]) -> dict[str, str]:
        # Function-local: config must not take a module-scope dependency on the inference tier, and
        # this keeps QWEN_EDIT_PROMPT_MODES the single declaration site rather than re-typing the
        # ids here. qwen_edit_layout is stdlib-only at module scope, so this costs nothing.
        from signet_trainer.inference.qwen_edit_layout import (  # noqa: PLC0415
            QWEN_EDIT_PROMPT_MODES,
        )

        expected = {mode.id for mode in QWEN_EDIT_PROMPT_MODES}
        missing = sorted(expected - {str(k) for k in v})
        if missing:
            raise ValueError(
                f"qwen_edit.render_inputs[].prompts has no prompt for mode(s) {missing}. §8 renders "
                f"every held-out input under BOTH modes side by side and the A-vs-B delta IS the "
                f"measurement; with one half absent the delta gets read as an adapter property. "
                f"Expected keys: {sorted(expected)}."
            )
        unknown = sorted({str(k) for k in v} - expected)
        if unknown:
            raise ValueError(
                f"qwen_edit.render_inputs[].prompts carries unknown mode id(s) {unknown}. The modes "
                f"are declared once, in inference/qwen_edit_layout.QWEN_EDIT_PROMPT_MODES: "
                f"{sorted(expected)}. An unknown key is silently never rendered."
            )
        blank = sorted(k for k, text in v.items() if not str(text).strip())
        if blank:
            raise ValueError(
                f"qwen_edit.render_inputs[].prompts is blank for mode(s) {blank}. A blank cell is "
                f"not a render with less text — the grid prints it as an em dash and the A/B "
                f"comparison it belongs to cannot be read."
            )
        return v


class QwenEditConfig(_Base):
    """Qwen-Image-Edit-2511 tunables — family #3, the chained-edit family.

    Same doctrine as ``H3Config``: D-NOHARDCODE, so none of these may ever be a literal in code —
    each one is documented, defaulted to the decided value, and fail-fast validated on CPU BEFORE
    any GPU is touched. Only meaningful when ``model.family == 'qwen_edit'``; ``SignetConfig``
    carries the bidirectional lean field-split REVERSE guard that rejects a non-default value under
    any other family, so this block can never be silently ignored.

    ⚠ One structural difference from ``H3Config`` worth reading before trusting a number here. H3's
    budget triple (``gpu_usable_gib`` / ``resident_gib`` / ``mib_per_packed_row``) is MEASURED, so
    H3 can DERIVE a packed-row ceiling and refuse an over-budget geometry. **No equivalent
    measurement exists for Qwen-Image-Edit on any card in this program.** ``max_packed_rows``
    therefore defaults to ``0`` = ceiling DISABLED, the row layout is computed and reported but
    nothing is refused, and the dry-run banner is required to say ``ceiling=DISABLED (unmeasured)``
    in that state. Synthesising a plausible ceiling from H3's A100 numbers would be a different
    model with a different row width and different resident weights — i.e. a guess wearing a
    measurement's clothes. When someone measures the real OOM boundary they set the integer and the
    refusal turns on.

    Deliberately NOT fields here: any qfloat8 / quantization switch, and any "which stage runs"
    mode knob. The house recipe locks the quantization (``qfloat8`` model + text encoder) the same
    way it locks the optimizer, and a recipe block must never be able to change a Modal function's
    OPERATIONAL mode (the ``arch_smoke_only`` precedent on ``H3Config``).
    """

    control_slots: int = Field(
        default=QWEN_EDIT_MAX_CONTROL_SLOTS,
        ge=1,
        le=QWEN_EDIT_MAX_CONTROL_SLOTS,
        description="How many control-image slots every sample carries. The cap is ai-toolkit's: "
        "its prompt template exposes ctrl_img_1..3 and nothing beyond "
        "(qwen_image_edit_plus.py:105-122). The count is FIXED, not per-sample — a sample with "
        "fewer real control images is blank-padded to this many slots (see blank_slot_fill), so "
        "the packed sequence length is a constant of the config rather than a property of the row. "
        "A variable-length sequence would make the row budget a distribution instead of a number, "
        "and the entire point of pricing at config load is that the answer is one integer.",
    )
    control_dirs: tuple[str, ...] = Field(
        default=(),
        description="The ORDERED control-image directories, one per slot, relative to the "
        "manifest's parent. POSITIONAL and never inferred: directory i fills slot i, which is "
        "exactly what a caption's `ctrl_img_{i+1}` refers to. A guessed convention (alphabetical, "
        "say) would re-point every caption's references and train each sample against a request "
        "nobody wrote — silently, and at an ordinary-looking loss. So there is no default order; "
        "`modal/entrypoint.py::_qwen_edit_config_gaps` refuses the dispatch when this is empty. "
        "For the embe dataset the two entries are ('refs_a', 'refs_b'), matching the two "
        "`control_path` entries of the production ai-toolkit config that trained it.",
    )
    blank_slots: tuple[int, ...] = Field(
        default=(),
        description="Zero-based slot indices that carry NO source directory and are blank-filled "
        "on every sample (see blank_slot_fill). Declared rather than derived so that 'this slot is "
        "deliberately empty' is distinguishable from 'a directory is missing', which is the "
        "difference between a designed dataset and a broken one. Must not overlap the positions "
        "covered by control_dirs, and the two together must account for exactly control_slots.",
    )
    blank_slot_fill: Literal["black", "white", "gray"] = Field(
        default="black",
        description="Which synthetic image fills a control slot that has no real image for this "
        "sample. A DECLARED choice rather than an implicit zero-tensor, because the fill is a "
        "visual input the adapter sees on every padded row and 'black' vs 'gray' is not a "
        "cosmetic difference to a VAE.",
    )
    control_area_px: int = Field(
        default=QWEN_EDIT_VAE_IMAGE_SIZE,
        ge=1024,
        description="Pixel-AREA budget each control image is fitted to before the VAE encodes it "
        "(channel B). TRANSCRIBED, not chosen: diffusers' "
        "pipeline_qwenimage_edit_plus.py:67 VAE_IMAGE_SIZE = 1024*1024. An arbitrarily-sized "
        "control image is resized to this budget, so slot cost does not vary with source "
        "resolution — which is exactly why the packed length is knowable at config load.",
    )
    condition_area_px: int = Field(
        default=QWEN_EDIT_CONDITION_IMAGE_SIZE,
        ge=1024,
        description="Pixel-AREA budget each control image is fitted to for the Qwen2.5-VL text "
        "encoder (channel A). TRANSCRIBED: pipeline_qwenimage_edit_plus.py:66 "
        "CONDITION_IMAGE_SIZE = 384*384. ⚠ The SAME control image goes down BOTH channels at "
        "DIFFERENT sizes — ai-toolkit's encode_control_in_text_embeddings is True "
        "(qwen_image_edit_plus.py:66), so the VL encoder sees a 384x384-budget copy while the VAE "
        "sees a 1024x1024-budget copy. Conflating the two budgets is the easy mistake here.",
    )
    prompt_tokens_estimate: int = Field(
        default=256,
        ge=1,
        description="Token budget the packed-row layout assumes for the caption. An ESTIMATE used "
        "ONLY for the layout/budget arithmetic — it is never the real tokenization, which happens "
        "Modal-side against the actual Qwen2.5-VL encoder. Mirrors H3Config.prompt_tokens_estimate.",
    )
    text_embed_dim: int = Field(
        default=3584,
        ge=1,
        description="[UNVERIFIED] Width of a Qwen2.5-VL text embedding, used ONLY as a synthetic "
        "shape in the zero-GPU dry-run. The TRUE value is ``txt_in.in_features`` in the live "
        "checkpoint, and the measured ground truth this family was specified from enumerates the "
        "60 transformer blocks but NOT txt_in's shape — so this number is carried as a declared "
        "assumption, not a fact. The weight-loading pass MUST assert it against the real txt_in and "
        "correct it here if it differs; nothing downstream may treat it as measured until then.",
    )
    rank_alpha_lock: int | None = Field(
        default=42,
        ge=2,
        description="The rank == alpha value locked across every round of a chain "
        "(QWEN-CHAINED-EDIT-METHOD.md: only dataset / name / steps change between rounds). "
        "Machine-checked at config load rather than remembered, because lora_A/lora_B are "
        "RANK-SHAPED: change the rank mid-chain and no later round can warm-start from an earlier "
        "one, and a published primer becomes unloadable. ``null`` is the deliberate exit from the "
        "chain — it still requires rank == alpha (PEFT scale 1.0) and then FORBIDS "
        "training.init_adapter_path.",
    )
    max_packed_rows: int = Field(
        default=0,
        ge=0,
        description="Packed-row ceiling for the refusal check. ``0`` = ceiling DISABLED, which is "
        "the honest default: no MiB-per-row figure has been MEASURED for Qwen-Image-Edit on any "
        "card in this program, and H3's measured triple prices a different model at a different "
        "row width. With 0 the layout is still computed and printed — it is simply not used to "
        "refuse anything, and the banner must say so. Set the integer once the real OOM boundary "
        "is measured; the refusal turns on with it.",
    )
    caption_dropout_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="P(replace this sample's caption with the empty string for one step) — "
        "unconditional-branch regularization. ZERO FOOTPRINT in signet today (no consumer exists "
        "yet), declared here rather than on ``data``/``training`` because a shared block is exactly "
        "what gets SILENTLY IGNORED under a family that has no consumer, and LTX has none. If LTX "
        "ever gains one, this field PROMOTES to ``data`` in the same commit as its LTX consumer — "
        "never before. See the cache_text_embeddings cross-check: a non-zero rate under caching is "
        "refused rather than silently trained at 0.",
    )
    cache_text_embeddings: bool = Field(
        default=True,
        description="Pre-compute and cache the Qwen2.5-VL text embeddings instead of encoding live "
        "each step. Effectively REQUIRED at >= 2 control slots: ai-toolkit's multi-slot path reads "
        "only the cached batch.control_tensor (SDTrainer.py:1509-1510), so signet mirrors the "
        "requirement rather than inventing a live multi-slot encode it has never run. "
        "⚠ NARROWER THAN ITS NAME, and flagged rather than quietly left: its ONLY runtime consumer "
        "is the re-encode SKIP in qwen_edit_preprocess (modal/fns.py:5560). Setting it false does "
        "NOT switch training to live text encoding — training always reads the precomputed "
        "conditions — so it cannot be used to enable caption dropout. It is a preprocess knob "
        "wearing a training-shaped name; either give it a training-side consumer or retire it "
        "(deliberately not retired here: shipped configs set it, and removing a field under "
        "extra='forbid' breaks them at load).",
    )
    control_cache_key_mode: Literal["path", "content"] = Field(
        default="content",
        description="What the control-image cache key hashes. ⚠ 'path' reproduces ai-toolkit "
        "BIT-FOR-BIT and that is the ONLY reason it exists: ai-toolkit hashes the control PATH "
        "STRING (dataloader_mixins.py:1971-1984), so overwriting a control file IN PLACE reuses a "
        "STALE VL embedding — and overwriting control images in place is precisely the chained-edit "
        "workflow this family is for. 'content' (the default) hashes the file BYTES and is correct "
        "for any chain; choose 'path' only when bug-compatibility with ai-toolkit is the goal.",
    )

    timestep_weighting: bool = Field(
        default=True,
        description="TRUE (the default) = the locked recipe: the loss is weighted by the bsmntw "
        "bell curve that ai-toolkit actually trains under. FALSE is the unweighted-loss ABLATION, "
        "pinning the weight at 1.0. Exposed as a config field because "
        "build_qwen_edit_step_fn's own docstring says the ablation should be 'stated explicitly in "
        "a config rather than achieved by deleting a line' — and until this field existed there was "
        "no YAML key for it, so the only route was editing train/qwen_edit_step.py or modal/fns.py: "
        "an untracked source edit driving a metered A100, producing an adapter attributable to no "
        "config on disk. Changing this makes a run diverge from every proven house chain; the "
        "DECISION-LOG entry is the point of it being a field.",
    )

    # ── the §8 RENDER request. Read only under mode 'sample'; empty is legal for the other modes ──
    # These two are the config surface of a sampler that already landed (feat/qwen_edit e132b30):
    # ``modal/fns.py::qwen_edit_sample`` and ``modal/entrypoint.py::_qwen_edit_config_gaps`` both
    # read them through tolerant ``getattr(..., default)`` so the stage could land before its schema
    # did. Under ``extra="forbid"`` that tolerance is not a soft landing — until the fields exist
    # here, a YAML declaring them is REJECTED at load, so the render cannot be configured at all.
    # Defaulting empty (rather than required) is deliberate: preprocess and train configs are legal
    # without a render request, and ``_qwen_edit_config_gaps`` owns the per-MODE refusal.
    render_checkpoint_band: tuple[str, ...] = Field(
        default=(),
        description="The ordered band-member checkpoint DIRECTORY NAMES under <output_dir>/ — H3's "
        "render_checkpoint_name pin made PLURAL, because on this family the deliverable is a band, "
        "not a winner (§8: 'checkpoint selection = a band, not a winner'). Never a path, and never "
        "left to CheckpointManager.find_latest(): the render directory is keyed on the render's "
        "identity and the checkpoint name is part of that key, so against a live run every "
        "re-dispatch would resolve a different adapter, land in a fresh directory and resume "
        "nothing (H3's D-10-DEF-19). Empty is legal here and refused per-mode by the entrypoint.",
    )
    render_inputs: tuple[QwenEditRenderInput, ...] = Field(
        default=(),
        description="The held-out control inputs, each carrying its ordered per-slot images and its "
        "A/B prompt pair. Neither half can be borrowed from what already exists: control_dirs names "
        "the TRAINING controls (a different question than a held-out probe), and validation.prompts "
        "is a flat list that cannot say which entry is A, which is B, or which input either belongs "
        "to. Empty is legal here and refused per-mode by the entrypoint.",
    )

    @field_validator("render_checkpoint_band")
    @classmethod
    def _check_render_checkpoint_band(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        # A NAME per member, never a path — the WR-09 pathlib-join contract every operator-authored
        # path in this schema obeys: ``ckpt_root / <value>`` with an absolute value REPLACES the
        # Volume prefix and '..' escapes the mount. CheckpointBand.__post_init__ already refuses an
        # empty band and a repeated member; it does NOT inspect path shape, and it runs in the
        # container. This runs at config load, on a laptop, for free.
        for name in v:
            if not str(name).strip():
                raise ValueError(
                    "qwen_edit.render_checkpoint_band carries an empty member name. Each member is "
                    "a checkpoint directory under <output_dir>/ and is part of the render key."
                )
            if name != Path(name).name or name in {".", ".."}:
                raise ValueError(
                    f"qwen_edit.render_checkpoint_band member {name!r} must be a bare directory "
                    "NAME under CHECKPOINTS_DIR/<output_dir> (e.g. "
                    "'checkpoint-step-00250-loss-0.0194'), never a path. pathlib join semantics let "
                    "an absolute value replace the Volume prefix and '..' escape it."
                )
        return v

    @field_validator("render_inputs")
    @classmethod
    def _check_render_input_ids(
        cls, v: tuple[QwenEditRenderInput, ...]
    ) -> tuple[QwenEditRenderInput, ...]:
        # The one refusal NOTHING else can make. QwenEditHeldOutInput validates an input in
        # isolation and cannot see its siblings; _qwen_edit_config_gaps checks presence and image
        # COUNT but not identity; qwen_edit_sample mirrors those two. So a duplicated id survives
        # every existing gate — and because the id is the file stem AND a slot of the render key,
        # the duplicate does not collide loudly: the second input silently overwrites the first
        # inside one render directory, and the gallery shows one input's pixels under two labels.
        seen: dict[str, int] = {}
        for i, entry in enumerate(v):
            key = str(entry.id)
            if key in seen:
                raise ValueError(
                    f"qwen_edit.render_inputs[{i}].id is {key!r}, already used by entry "
                    f"[{seen[key]}]. The id is the stem of every file an input renders and one slot "
                    f"of the render key's control axis, so two inputs sharing it write the SAME "
                    f"filenames into the SAME render directory — the later silently overwrites the "
                    f"earlier and the grid shows one input twice under two labels."
                )
            seen[key] = i
        return v

    @model_validator(mode="after")
    def _check_render_input_slot_count(self) -> "QwenEditConfig":
        """Every held-out input must name EXACTLY ``control_slots`` images, in slot order.

        Separate from ``_check_control_slot_coverage`` rather than folded into it, because that one
        early-returns when neither ``control_dirs`` nor ``blank_slots`` is declared — which is
        precisely the shape of a ``sample`` config. Folding this in would make it dead code in the
        only mode that reads these fields.

        A short list does not render a smaller grid: the mapping is positional, image i fills slot
        i, so a missing entry re-points what every later ``ctrl_img_{i+1}`` addresses and the render
        answers a request nobody wrote — under the right label, at an ordinary-looking grid.
        """
        for i, entry in enumerate(self.render_inputs):
            if len(entry.images) != self.control_slots:
                raise ValueError(
                    f"qwen_edit.render_inputs[{i}] ({entry.id!r}) declares {len(entry.images)} "
                    f"control image(s) but qwen_edit.control_slots is {self.control_slots}. The "
                    f"mapping is POSITIONAL — image i fills slot i, which is what the prompt's "
                    f"ctrl_img_{{i+1}} addresses — so a short list does not render a smaller grid, "
                    f"it renders the WRONG request under the right label."
                )
        return self

    @model_validator(mode="after")
    def _check_control_slot_coverage(self) -> "QwenEditConfig":
        """``control_dirs`` + ``blank_slots`` must account for EXACTLY ``control_slots``, disjointly.

        Refused at config load rather than in the container, because every failure mode here is
        silent-and-plausible rather than loud. Too few directories and slot N is filled by whatever
        the resolver reaches for next; an overlap and one directory's images are discarded in favour
        of a blank fill; a duplicate index and a slot is written twice. All three produce a run that
        trains happily against control images the captions do not describe, at an ordinary loss, and
        the adapter is wrong in a way no metric shows.

        Empty ``control_dirs`` is allowed HERE — ``modal/entrypoint.py::_qwen_edit_config_gaps``
        owns that refusal, and it is per-MODE (a ``sample`` run needs no manifest-relative control
        directories). Splitting it this way keeps a dry-run honest for configs that are legal for
        one mode and not another.
        """
        if not self.control_dirs and not self.blank_slots:
            return self  # nothing declared yet; the per-mode gap check owns this case.

        overlap = sorted(set(self.blank_slots) & set(range(len(self.control_dirs))))
        if overlap:
            raise ValueError(
                f"qwen_edit.blank_slots {list(self.blank_slots)} overlaps the positions covered by "
                f"control_dirs (0..{len(self.control_dirs) - 1}) at {overlap}: slot(s) {overlap} "
                f"are declared BOTH as a real control directory and as deliberately blank. The "
                f"mapping is positional — directory i fills slot i — so this is ambiguous rather "
                f"than merely redundant."
            )

        if len(set(self.blank_slots)) != len(self.blank_slots):
            raise ValueError(
                f"qwen_edit.blank_slots {list(self.blank_slots)} contains a duplicate index. Each "
                f"slot is filled exactly once."
            )

        out_of_range = sorted(i for i in self.blank_slots if not 0 <= i < self.control_slots)
        if out_of_range:
            raise ValueError(
                f"qwen_edit.blank_slots {list(self.blank_slots)} carries index(es) {out_of_range} "
                f"outside 0..{self.control_slots - 1} (control_slots={self.control_slots})."
            )

        covered = len(self.control_dirs) + len(self.blank_slots)
        if covered != self.control_slots:
            raise ValueError(
                f"qwen_edit control slots are under/over-declared: {len(self.control_dirs)} "
                f"control_dirs + {len(self.blank_slots)} blank_slots = {covered}, but "
                f"control_slots is {self.control_slots}. Every slot must be accounted for exactly "
                f"once — the packed sequence length is a constant of the config "
                f"({self.control_slots} slots x its per-slot rows), so an unaccounted slot changes "
                f"the priced row budget without changing the price."
            )
        return self


# --------------------------------------------------------------------------------------------------
# signet-owned blocks (layered above the embedded LTX data).
# --------------------------------------------------------------------------------------------------


class TrainingConfig(_Base):
    """The locked LTX-2.3 LoRA recipe, carried as DATA (03-CONTEXT §Optimizer + LoRA config).

    The prior production project's validated enochiatron-ancestor recipe (``<prior-project>/_train/ltx/r1_config.yaml``)
    with the operator's two deliberate deviations — cosine-with-min-lr (vs the precedent's effective constant)
    and rank/alpha 64 (vs the precedent's 42). The cosine ``min_lr_ratio`` / ``warmup_steps`` come from the
    flimmer template. These are plain hyperparameters consumed Modal-side by ``train/loop.py``
    (Plan 03-04); NO ``import modal``/``ltx_*`` here (Anti-Pattern 6).
    """

    learning_rate: float = Field(default=5e-5, gt=0.0, description="5e-5 beat 1e-4 in the prior project's A/B.")
    weight_decay: float = Field(default=0.01, ge=0.0)
    # NEVER Prodigy — banned in the precedent config (breaks on differential-LR); LTX is single-stream.
    optimizer: str = Field(default="adamw8bit")
    betas: tuple[float, float] = Field(default=(0.9, 0.999))
    eps: float = Field(default=1e-8, gt=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    # Run-specific knob (no default): the first metered run is a tiny correctness overfit — a handful
    # of steps on 1-2 clips (D-RUN-1). Every real run must state its own step budget.
    max_steps: int = Field(..., ge=1, description="Run-specific total training steps (D-RUN-1).")
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    # 0.30 = the precedent's standing 'sharper detail' choice (0.10 was the less-preferred std chain).
    uniform_prob: float = Field(default=0.30, ge=0.0, le=1.0)
    scheduler: str = Field(default="cosine_with_min_lr")
    min_lr_ratio: float = Field(default=0.01, ge=0.0, le=1.0, description="flimmer template (source 0.1).")
    warmup_steps: int = Field(default=0, ge=0, description="flimmer template (source min(50, total//10)).")
    checkpoint_every: int = Field(default=200, ge=1, description="~200-step checkpoint + sample cadence.")
    mixed_precision: str = Field(default="bf16")
    gradient_checkpointing: bool = Field(default=True, description="GC ON, before LoRA inject (TRAIN-06).")
    seed: int = Field(default=42)
    # D-9-CHAINED — chained warm-restart source. Volume-relative like frozen_adapter_path (T-07-02).
    init_adapter_path: str | None = Field(
        default=None,
        description="[D-9-CHAINED] Volume-relative path to a PRIOR round's final adapter dir; at "
        "COLD START only (no in-dir checkpoint) the trainable adapter is warm-started from it with a "
        "FRESH optimizer at step 0. Routed through validate_volume_relative_path (T-07-02). None = no "
        "chaining. Ignored when an in-dir checkpoint exists (same-dir resume wins).",
    )
    # D-9-RECIPE gap-fill — bounded checkpoint retention (CheckpointManager keep_n); precedent used 25.
    keep_checkpoints: int | None = Field(
        default=None,
        ge=1,
        description="[D-9-RECIPE gap-fill] Prune to the N most-recent checkpoints after each save "
        "(CheckpointManager keep_n). None = unbounded (current behavior). The prior project used 25.",
    )
    # ---- Phase 09.1 infra-hardening — in-loop liveness watchdog (config-first; AUDIT #11).
    # D-NOHARDCODE: the watchdog threshold is a documented, defaulted config field, never a code
    # literal. Consumer logic lands in Wave-2 plan 09.1-05; these are the interface-first contracts.
    checkpoint_expected_minutes: float | None = Field(
        default=None,
        gt=0.0,
        description="[AUDIT-#11] Expected wall-clock minutes per checkpoint_every interval; enables "
        "the in-loop training liveness watchdog. None = watchdog OFF (byte-identical current "
        "behavior — matches the keep_checkpoints None-default precedent). Consumed by 09.1-05.",
    )
    checkpoint_stall_multiplier: float = Field(
        default=2.5,
        gt=0.0,
        description="[AUDIT-#11] Watchdog K — the stall threshold is "
        "checkpoint_expected_minutes * checkpoint_stall_multiplier; inert while "
        "checkpoint_expected_minutes is None. 2.5x the cadence per the time-gate house rule. "
        "Consumed by 09.1-05 (no consumer logic here).",
    )

    @model_validator(mode="after")
    def _check_chain_and_retention(self) -> "TrainingConfig":
        # WR-04 / T-07-02: enforce the Volume-relative contract on init_adapter_path AT CONFIG LOAD —
        # reject absolute paths and '..' escapes PRE-approval (same shape as frozen_adapter_path), so a
        # bad chained-round path can never reach a metered dispatch. Only when set (None = no chaining).
        if self.init_adapter_path is not None:
            validate_volume_relative_path(self.init_adapter_path, field="init_adapter_path")
        return self


class OffloadConfig(_Base):
    """OFFL-01/02 block-swap offloader field — baseline-first ``blocks_to_swap=0`` (D-OFF-1).

    The first LoRA trains with NO block-swap (enochiatron shipped its 22B LoRA on one
    A100-80GB without a confirmed offloader). The harvested ``offload/block_swap.py`` is
    UNVALIDATED reference code; ``0`` keeps it constructed-but-inert until VRAM forces A/B.
    """

    blocks_to_swap: int = Field(
        default=0,
        ge=0,
        description="0 = no block-swap (baseline; D-OFF-1). >0 harvested-but-UNVALIDATED on LTX.",
    )


class AudioConfig(_Base):
    """Audio modality block (Phase 9, GATE-SPEC-inpaint-a2v rev 2 item 2 — modality-extensible).

    signet is video-first but NOT video-ONLY (CLAUDE.md: "Do NOT hardcode video-only … the harness
    must stay modality-extensible"). This block carries the audio-modality knobs the audio-to-video
    (a2v) mode needs, config-first (D-NOHARDCODE): every field documented, and a lean field-split on
    ``SignetConfig`` rejects a non-default value under any non-a2v mode (no silently-ignored block).

    In a2v the audio is the DRIVING input (frozen conditioning), NOT a generated output — hence
    ``is_generated: false`` and ``generate_audio: false``. Both are reserved defaults today (the
    generated-audio path is out of scope); they exist so the modality's role is config-visible rather
    than an implicit assumption, and so a future generated-audio mode has a declared surface.
    """

    is_generated: bool = Field(
        default=False,
        description="Whether audio is a GENERATED modality. a2v drives video FROM audio, so audio is "
        "the INPUT (frozen conditioning), never generated -> False. Reserved default (a "
        "generated-audio mode is out of scope); flipping it is only meaningful under a future "
        "audio-generating mode (lean field-split rejects a non-default value under any current mode).",
    )
    with_audio: bool = Field(
        default=False,
        description="Whether the canonical `--mode preprocess` encode extracts + encodes audio to "
        "``audio_latents/`` (upstream process_dataset with_audio flag; it EXISTS at signet's pinned "
        "SHA d6053703). Required True for an a2v run so the audio latents the A2VStrategy reads "
        "actually get encoded — the SignetConfig cross-field guard fail-fasts an a2v config that "
        "leaves it False (no silent audio-skip). Default False keeps every non-a2v encode "
        "byte-identical (audio out of scope).",
    )
    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Sub-dir under preprocessed_data_root holding the per-sample AUDIO latents "
        "(mirrors the upstream process_dataset ``audio_latents/`` output; the A2VStrategy's "
        "get_data_sources reads this source). Existence is checked Modal-side where the Volume "
        "mounts, NOT locally (Pitfall 1). Only valid when conditioning.mode == 'audio_to_video'.",
    )
    generate_audio: bool = Field(
        default=False,
        description="Whether the SAMPLE path emits generated audio alongside the video. FALSE for "
        "a2v (the proof renders video DRIVEN by input audio; no audio is generated) — threaded into "
        "GenerationConfig.generate_audio=False (inference/sampler.py). Reserved default; a "
        "non-default value is rejected under any non-a2v mode (lean field-split).",
    )


class ModalConfig(_Base):
    """signet Modal block — carries the cost-estimate fields (MODL-03, enochiatron precedent).

    ``est_usd = hourly_rate_usd * est_hours`` is printed before any gated launch (Plan 01-03).
    """

    hourly_rate_usd: float = Field(
        default=1.64,
        ge=0.0,
        description="[ASSUMED 1.64] A100-80GB $/hr guardrail constant; confirm vs live Modal pricing.",
    )
    cost_guardrail_usd: float = Field(default=50.0, ge=0.0, description="enochiatron precedent.")
    # WR-04: CPU-only modes (backup / restore / fuse) run on Modal fns with NO gpu= — the A100
    # hourly_rate_usd is the WRONG basis for their cost print, and with a large training est_hours
    # (e.g. an 18h production round) the A100 estimate could FALSELY block a near-zero-cost CPU job at the
    # guardrail. The entrypoint derives those modes' estimate from THIS CPU rate instead (config-first,
    # D-NOHARDCODE). Default ~near-zero: even at a large est_hours the CPU estimate stays well under
    # the guardrail, while the SAME cost-print + approval gate still runs (no silent free path).
    cpu_hourly_rate_usd: float = Field(
        default=0.05,
        ge=0.0,
        description="[WR-04] ~near-zero $/hr for CPU-only Modal modes (backup/restore/fuse); the "
        "entrypoint uses this instead of the A100 hourly_rate_usd so an A100 est_hours can't falsely "
        "block a CPU-cheap backup at the cost guardrail.",
    )
    # WR-01: single source of truth for the launch-cost estimate. 2.0h matches BOTH shipped example
    # YAMLs (ltx23_lora.example.yaml / bad_frames.example.yaml) and the enochiatron precedent (a
    # sub-$10, ~few-hour LoRA run -> 1.64 * 2.0 = $3.28). Keep schema default and examples in sync.
    est_hours: float = Field(default=2.0, ge=0.0)

    # ---- Phase 09.1 infra-hardening tunables (config-first; AUDIT #5 / #1). D-NOHARDCODE: every
    # new timeout/margin threshold is a documented, defaulted config field — NEVER a code literal
    # (the audit flagged hardcoded rates/timeouts as THE anti-pattern). Consumer logic lands in
    # Wave-2 plans (09.1-04 / 09.1-03); these are the interface-first contracts they READ.
    timeout_margin: float = Field(
        default=1.5,
        gt=0.0,
        description="[AUDIT-#5] Multiplier on est_hours used to DERIVE the sample/preprocess Modal "
        "function timeout at entrypoint dispatch (timeout ≈ est_hours * timeout_margin hours, applied "
        "via .with_options(timeout=...)). Keeps a wedged render from burning to the 24h ceiling; "
        "train() keeps its own 24h timeout. Consumed by 09.1-04 (no consumer logic here).",
    )
    # D-10-DEF-17: the entrypoint dispatches ASYNC (``.spawn()``) so the server does not cancel an
    # in-flight run when the local client disappears. This field is the BOUNDED window the client
    # still watches SYNCHRONOUSLY afterwards, via ``FunctionCall.get(timeout=...)``, so cheap early
    # aborts (arch-gate FAIL, CPU preflight, config refusal, import death) keep surfacing at the
    # console instead of silently becoming a detached failure nobody reads.
    dispatch_watch_seconds: float = Field(
        default=300.0,
        ge=0.0,
        description="[D-10-DEF-17] Seconds the gated entrypoint watches a SPAWNED (async) dispatch "
        "via FunctionCall.get(timeout=...) before disengaging and printing the FunctionCall id. "
        "⛔ EXPIRY NEVER CANCELS ANYTHING — the call is ASYNC, so the client merely stops watching "
        "and the run continues server-side. DERIVED (not chosen) from this repo's OWN measured "
        "fast-abort latencies: D-10-DEF-13 import death = seconds; D-10-DEF-14 pipeline-root failure "
        "at ModularPipeline.from_pretrained = 73 s; arch-gate / two-model-load aborts (preprocess "
        "attempts 1-2) = ~180 s; a server retry-policy refusal fires at app init, BEFORE dispatch, so "
        "this window does not bound it. 300 s is ~1.7x the slowest observed fast abort (180 s). Below "
        "~200 s the arch-gate feedback channel starts being lost; far above it the agent's lifetime "
        "is re-coupled to the run, which is the exact defect .spawn() closes. ge=0.0 so 0 is a legal "
        "'spawn and disengage immediately' (get(timeout=0) is a documented immediate poll).",
    )
    render_stall_minutes: float = Field(
        default=120.0,
        gt=0.0,
        description="[AUDIT-#1] Parallel-watcher artifact-freshness liveness gate: if no NEW committed "
        "mp4 lands on the checkpoints Volume for this many minutes, treat the DETACHED render as "
        "hung/preempted and re-dispatch. 120 ≈ 2x the measured 60-90 min render envelope (time-gate "
        "house rule). Consumed by 09.1-03 (no consumer logic here).",
    )

    # ---- OBS-01 watcher-SPOF hardening tunables (config-first; D-NOHARDCODE). The 2026-07-12
    # incident: a local watcher process died (external kill, clean log end) and silently froze the
    # grid while a healthy DETACHED run kept training — read as "inference hung." These four fields
    # are the interface-first contracts the watchers + the PowerShell supervisor READ (consumer logic
    # lands in 09.1-10, never a code literal). NONE of them re-couple to the detached render dispatch
    # (that client-kill was the AUDIT-#1 root cause) — volume_op_timeout_s is on READ-ONLY volume ops.
    volume_op_timeout_s: float = Field(
        default=120.0,
        gt=0.0,
        description="[OBS-01] Per-call timeout (seconds) for a watcher's READ-ONLY `modal volume "
        "ls/get` op so a hung modal CLI call can never wedge the single-threaded poll loop (D-OBS-4). "
        "A TimeoutExpired is caught and returned as a failed result ('no data this poll'); the loop "
        "continues. NOT applied to the detached/attached render dispatch — re-coupling a client-kill "
        "to the render is the forbidden AUDIT-#1 regression. Consumed by 09.1-10 (watchers).",
    )
    watcher_heartbeat_stall_minutes: float = Field(
        default=15.0,
        gt=0.0,
        description="[OBS-01] The supervisor kills + relaunches the watcher when its heartbeat file is "
        "older than this many minutes (D-OBS-3, WATCHER-stall — distinct from render_stall_minutes / "
        "the training stall). Set well above the poll cadence but low enough that a truly dead loop is "
        "caught fast. Consumed by 09.1-10 (watcher_supervisor.ps1).",
    )
    watcher_relaunch_backoff_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="[OBS-01] Supervisor wait (seconds) between a watcher crash-exit and its relaunch "
        "(fixed/linear backoff — Claude's discretion). ge=0 so a zero-backoff immediate relaunch is a "
        "legal config. Consumed by 09.1-10 (watcher_supervisor.ps1).",
    )
    watcher_relaunch_cap: int = Field(
        default=20,
        ge=0,
        description="[OBS-01] Max supervisor relaunches before it gives up and exits — the fork-bomb / "
        "crash-loop guard so a metered-dispatcher watcher can NEVER be auto-respawned unboundedly "
        "(bounded auto-respawn per D-OBS-1 / the threat model). Consumed by 09.1-10 "
        "(watcher_supervisor.ps1).",
    )

    # Modal secret NAMES (config-driven so each account uses its own secrets). Defaults match the
    # maintainer account's actual secret names; override per-account in YAML. These are carried as
    # DATA only (no `import modal` here — Anti-Pattern 6). The Modal app graph (app.py) reads the same
    # names via env-var overrides at MODULE-IMPORT time (SIGNET_HUGGINGFACE_SECRET_NAME /
    # SIGNET_WANDB_SECRET_NAME). WR-01: because app.py captures the names at import — before the
    # entrypoint's main() runs — a non-default account MUST export those env vars IN THE SHELL before
    # `modal run`; the entrypoint's step-1b guard fail-fasts (pre-approval) if the captured names do
    # not match these config fields, rather than silently ignoring them.
    huggingface_secret_name: str = Field(
        default="my-huggingface-secret",
        description="Name of the Modal secret carrying the HF token (gated weight downloads).",
    )
    wandb_secret_name: str = Field(
        default="my-wandb-secret",
        description="Name of the Modal secret carrying the wandb API key (run logging).",
    )

    # ------------------------------------------------------------------------------------------
    # Modal RESOURCE NAMES — the App name + the three Volume names (D-NOHARDCODE / AUDIT #1).
    #
    # ⛔ WHY THE DEFAULTS STILL SAY "signe-*" AFTER THE SIGNET RENAME — DO NOT "FIX" THESE.
    # These four strings are not branding: they are the NAMES OF PRE-EXISTING MODAL RESOURCES that
    # already hold this project's live data (the weights Volume alone carries ~134 GiB; the
    # checkpoints Volume carries every committed training checkpoint). ``Volume.from_name(...,
    # create_if_missing=True)`` does NOT fail on a name it has never seen — it silently provisions a
    # NEW, EMPTY Volume. So renaming a default here does not break loudly; it makes the trainer
    # quietly look at empty storage while the real data sits untouched under the old name. The
    # signe -> Signet rename is a PUBLIC-IDENTITY change (package, CLI, docs, license) and
    # deliberately stops at this boundary.
    #
    # They are fields (not literals) because the audit found these names hardcoded across ~15
    # scripts, which made the trainer unusable against any other Modal account. A beta user points
    # these at THEIR OWN Volumes/App via YAML (or the SIGNET_* env vars app.py reads at import).
    # See README "Pointing the trainer at your own Modal account".
    # ------------------------------------------------------------------------------------------
    app_name: str = Field(
        default="signe-trainer",
        description="Modal App name (``modal.App(...)``); also what ``modal app list/logs`` "
        "attributes a run to. DEFAULT INTENTIONALLY PRE-RENAME: live/in-flight runs are dispatched "
        "under this exact App name — changing it orphans them from the status tooling. Override "
        "per-account. Read by app.py at MODULE-IMPORT time via SIGNET_APP_NAME.",
    )
    weights_volume_name: str = Field(
        default="signe-trainer-weights",
        description="Modal Volume holding base weights (LTX-2.3 + Gemma + H3 components). DEFAULT "
        "INTENTIONALLY PRE-RENAME: this Volume already holds ~134 GiB of downloaded weights, and "
        "create_if_missing=True would silently mount a new EMPTY Volume under a renamed string. "
        "Override per-account. Read by app.py via SIGNET_WEIGHTS_VOLUME_NAME.",
    )
    dataset_volume_name: str = Field(
        default="signe-trainer-dataset",
        description="Modal Volume holding staged clips + the encoded/pre-computed latent corpus. "
        "DEFAULT INTENTIONALLY PRE-RENAME (see weights_volume_name — silent empty-Volume hazard). "
        "Override per-account. Read by app.py via SIGNET_DATASET_VOLUME_NAME.",
    )
    checkpoints_volume_name: str = Field(
        default="signe-trainer-checkpoints",
        description="Modal Volume holding training checkpoints, samples and grids. DEFAULT "
        "INTENTIONALLY PRE-RENAME (see weights_volume_name — silent empty-Volume hazard); it "
        "currently holds every checkpoint of the completed r1 round. Override per-account. Read by "
        "app.py via SIGNET_CHECKPOINTS_VOLUME_NAME.",
    )

    # D-8-YOLOCAP — cumulative session-spend cap + ledger path. Bounds yolo autonomy by REAL
    # cumulative spend (session_cap.py), not just the per-run cost_guardrail_usd above. Both are
    # config-driven (D-NOHARDCODE) so the harness/skill never hardcodes the cap or the ledger path.
    # 10.0 is the RESEARCH A3 proposed house default the operator confirms at setup.
    session_cap_usd: float = Field(
        default=10.0,
        ge=0.0,
        description="[D-8-YOLOCAP] cumulative session-spend cap (USD) — the HOUSE DEFAULT in the "
        "WR-02 chain. projected + spent must stay <= this or the harness drops to ask-first. The "
        "per-session override is the session_cap_usd the setup gate writes into "
        "session_spend_ledger_path; when present that value is the live cap, else this default. The "
        "training-run skill reads this chain (never a hardcoded cap); confirm the house default at "
        "setup (A3).",
    )
    session_spend_ledger_path: str = Field(
        default=".planning/harness/SESSION-STATE.json",
        description="[D-8-YOLOCAP] path to the SESSION-STATE.json cumulative-spend ledger read by "
        "session_cap.read_ledger / append_spend / consume_blanket. The SINGLE authoritative ledger "
        "path (WR-02): the training-run skill reads it from here, never a hardcoded literal "
        "(D-NOHARDCODE). PROJECT-RELATIVE on purpose: the ledger is LIVE per-project state (one "
        "project's accumulating spend), so unlike the packaged SPEC/TEMPLATE data in "
        "signet_trainer.harness_data it must NOT ship inside the wheel and must not be shared "
        "across projects. A fresh project has no ledger yet — that is a fresh session (spend 0.0), "
        "not an error; seed one by copying the packaged SESSION-STATE.template.json "
        "(signet_trainer.harness_data.spec_path('SESSION-STATE.template.json')) to this path, or "
        "set this field to wherever your project keeps it.",
    )


class MaskCondition(_Base):
    """One ``mask`` validation-sample condition: an input clip + its aligned mask (Phase 9, INPAINT).

    The schema surface for requesting a MASKED test render at sample time (GATE-SPEC-inpaint-a2v
    rev 2: "validation = held-out clip masked + regenerated"). Both paths are carried as DATA —
    Volume-relative, NOT FS-validated locally (Pitfall 1, same convention as ``ConditioningItem``);
    existence is resolved Modal-side where the Volume mounts. The sampler branch that CONSUMES this
    condition (encode input video + per-token denoise mask — an ic_lora-pipeline-class port) is a
    separate build item; the schema is declared first so a masked render is a config-visible,
    fail-fast request rather than a hardcoded path (D-NOHARDCODE).
    """

    type: Literal["mask"] = Field(
        default="mask",
        description="Condition kind discriminator. 'mask' is the Phase-9 masked-render condition; "
        "future condition kinds widen this into a discriminated union on 'type'.",
    )
    video: str = Field(
        ...,
        description="Volume-relative path to the input/held-out test clip to be masked + "
        "regenerated. DATA only — NOT FS-validated locally (Pitfall 1); resolved Modal-side.",
    )
    mask: str = Field(
        ...,
        description="Volume-relative path to the mask aligned 1:1 to the clip (same F/H/W; "
        "precedent polarity: region-to-generate BLACK(0), context WHITE(1)). DATA only — NOT "
        "FS-validated locally (Pitfall 1); resolved Modal-side.",
    )


class AudioCondition(_Base):
    """One ``audio`` validation-sample condition: a driving-audio clip (Phase 9, AUDIO-TO-VIDEO).

    The schema surface for requesting an a2v test render driven by a ``.wav`` (GATE-SPEC-inpaint-a2v
    rev 2 item 7: "encode a .wav via the audio VAE at sample time"). The path is carried as DATA —
    Volume-relative, NOT FS-validated locally (Pitfall 1, same convention as ``MaskCondition``);
    existence is resolved Modal-side where the Volume mounts. The sampler branch that CONSUMES this
    condition (encode the .wav via the audio VAE encoder + frozen-audio conditioning render) is the
    Modal-side a2v render path; the schema is declared first so a driving-audio render is a
    config-visible, fail-fast request rather than a hardcoded path (D-NOHARDCODE).
    """

    type: Literal["audio"] = Field(
        default="audio",
        description="Condition kind discriminator. 'audio' is the Phase-9 a2v driving-audio render "
        "condition; it shares the discriminated-union surface with the inpaint 'mask' kind.",
    )
    audio: str = Field(
        ...,
        description="Volume-relative path to the driving-audio clip (a ``.wav``) for this render. "
        "DATA only — NOT FS-validated locally (Pitfall 1); resolved Modal-side. Only valid when "
        "conditioning.mode == 'audio_to_video'.",
    )


class ValidationSample(_Base):
    """One requested validation render: its prompt + its conditions travel TOGETHER (Phase 9).

    An OBJECT (same D-6-ITEMS rationale as ``ConditioningItem`` — NOT parallel prompt/condition
    lists) so a sample's prompt and its condition can never desync across a config edit.
    A condition-LESS render request belongs in ``validation.prompts`` (the existing surface), so
    ``conditions`` requires at least one entry — an empty list here would silently duplicate the
    prompts path (lean field-split: no ambiguous config block).

    ``conditions`` is a discriminated union on ``type``: ``mask`` (inpaint held-out clip + mask) or
    ``audio`` (a2v driving ``.wav``). The cross-block guard on ``SignetConfig`` fail-fasts a condition
    kind whose ``conditioning.mode`` doesn't match (a 'mask' outside inpaint / an 'audio' outside
    audio_to_video would be silently ignored on a metered render).
    """

    prompt: str = Field(
        ...,
        description="The prompt for this conditioned render (travels WITH its conditions; "
        "plain unconditioned prompts stay in validation.prompts).",
    )
    conditions: list[MaskCondition | AudioCondition] = Field(
        ...,
        min_length=1,
        description="The sample's conditions — a union of 'mask' (inpaint) and 'audio' (a2v) kinds, "
        "disambiguated by their fields under extra='forbid' (a {video, mask} dict is a MaskCondition; "
        "an {audio} dict is an AudioCondition; the optional 'type' discriminator is respected when "
        "present). At least one is required — a condition-less sample belongs in validation.prompts.",
    )


class ValidationConfig(_Base):
    """D-14 — reserve the validation/sampling block from day one (Phase 4.1 extends it)."""

    interval: int = Field(default=250, ge=1)
    prompts: list[str] = Field(default_factory=list)
    frame_count: int = Field(default=49, ge=1)
    guidance_scale: float = Field(default=3.0, ge=0.0)
    num_samples: int = Field(default=1, ge=1)
    seed: int = Field(default=42)

    # Phase-4.1 (INFR-01) LTX canonical sampling fields. Defaults are the RE-VALIDATED
    # LTX-2.3 params (D-PARAMS-1, RESEARCH §Canonical Params @ pinned SHA d6053703) — NOT
    # the Wan-tuned set (no UniPC / shift / frames=33 / guidance=5 / steps=50). There is
    # deliberately NO ``decode_timestep`` field: the ValidationSampler port target has no
    # consumer for it (RESEARCH Pitfall 3). Euler + STG is the clean LTX LoRA path.
    num_inference_steps: int = Field(default=30, ge=1)
    stg_scale: float = Field(default=1.0, ge=0.0)
    stg_blocks: list[int] = Field(default_factory=lambda: [29])
    stg_mode: str = Field(default="stg_v")
    frame_rate: float = Field(default=25.0, gt=0.0)
    width: int = Field(default=768, ge=32)
    height: int = Field(default=352, ge=32)
    # D-DEPTH-1: two-stage spatial upscaler is a separate default-OFF plan/wave; keep it
    # gated here so the single-stage grid never depends on the upscaler weights.
    two_stage_upscale: bool = Field(
        default=False, description="D-DEPTH-1: two-stage spatial upscaler, default OFF"
    )
    # Phase 10 (H3) — waive MiniMax-H3's 5-15 s GENERATION band for this render.
    #
    # WHY A WAIVER EXISTS. A campaign that trains STILLS stages each one as the shortest legal
    # `17n + 5` clip, so its trained length sits BELOW the band by construction and every
    # evaluation it will ever run is off-band. Rendering ~18x longer than trained is not a
    # neutral substitute for rendering at the trained length — it is a different question about
    # the same weights, and telling those two apart is the whole point.
    #
    # WHAT IT WAIVES, AND WHAT IT CANNOT. The band is a POLICY: the pipeline reads it off two
    # `@property` on `MiniMaxH3ModularPipeline`, and the render widens them for itself. It does
    # NOT waive the video VAE's 22-frame DECODE FLOOR (`H3_DECODE_FLOOR_FRAMES`), which is
    # arithmetic — below it the decoder's chunk list is empty and `torch.cat([])` raises AFTER
    # the whole denoise has been paid for. Nor does it waive the `17n + 5` law.
    #
    # Config-first per D-NOHARDCODE, default False so every existing config renders
    # byte-identically and still gets the pre-flight refusal.
    allow_offband_frame_count: bool = Field(
        default=False,
        description="H3: allow validation.frame_count outside MiniMax-H3's 5-15 s generation "
        "band, to evaluate a model AT ITS TRAINED LENGTH when that length is off-band. Widens "
        "the pipeline's min/max duration for this render only. Does NOT waive the 22-frame VAE "
        "decode floor or the 17n+5 law. Default False = historical behaviour.",
    )
    # Phase 10 (H3 / ref2v) — WHICH manifest row supplies the reference slots this render conditions
    # on. Named by the config's own declared subject-id vocabulary
    # (``h3.character_reference_sizes`` / ``h3.environment_reference_sizes``, third tuple element),
    # in D-10-REFORDER order, NOT by a row index and NOT by a clip name:
    #
    #   * an INDEX silently re-points if the manifest is ever re-staged in a different order, and it
    #     names the row rather than the thing the eval actually varies;
    #   * a clip/media name is client property and must never enter a tracked file;
    #   * a subject-id list names the reference CONDITION itself, which is what the render is a probe
    #     of. Re-staging cannot change what ``["C", "018"]`` means.
    #
    # The order is matched EXACTLY against the resolved slots, so this never re-orders anything: it
    # only chooses which row supplies them, and a list written in the wrong order is refused rather
    # than quietly accepted (D-10-REFORDER puts the environment reference LAST).
    #
    # Default [] keeps every existing config byte-identically unaffected — manifest row 0, exactly as
    # before.
    reference_subject_ids: list[str] = Field(
        default_factory=list,
        description="H3 ref2v: the reference slots this render conditions on, as subject_ids in "
        "D-10-REFORDER order (e.g. ['C', '018']). The first manifest row whose seeded rotation "
        "resolves to exactly these slots is used. Empty (default) = manifest row 0, the historical "
        "behaviour. Stable across a re-stage; never an index, never a clip name.",
    )
    # D-9-OFFL02-CLOSE — the OFFL-02 in-loop-sampling / decoder knob. Config-first (D-NOHARDCODE).
    in_loop_sampling: bool = Field(
        default=False,
        description="[D-9-OFFL02-CLOSE] When True the TRAIN path loads with_video_vae_decoder=True "
        "and renders an in-loop validation sample every checkpoint_every steps under active "
        "block-swap (OFFL-02 live proof). Requires validation.prompts to be non-empty. Default False "
        "keeps the training load decoder-off (current behavior). Config-first per D-NOHARDCODE.",
    )
    # Phase 9 (INPAINT) — conditioned validation renders. Each entry is a prompt + its condition
    # objects (today: the 'mask' kind — a held-out clip masked + regenerated, GATE-SPEC rev 2).
    # Default [] keeps every existing config loading unchanged (backward-compat, same convention
    # as prompts). The mask kind is only meaningful when conditioning.mode == 'inpaint' — the
    # cross-block guard lives on SignetConfig (mode is on conditioning, samples on validation).
    samples: list[ValidationSample] = Field(
        default_factory=list,
        description="Conditioned validation renders: {prompt, conditions[]} objects (Phase 9 — "
        "'mask' is the only condition kind today). Plain unconditioned prompts stay in prompts. "
        "Mask conditions require conditioning.mode == 'inpaint' (cross-checked at SignetConfig).",
    )

    @model_validator(mode="after")
    def _check_samples(self) -> "ValidationConfig":
        # T-06-02 / WR-09 shape: every operator-authored sample-condition path obeys the documented
        # Volume-relative contract AT CONFIG LOAD (joined under the Volume prefix Modal-side —
        # pathlib join semantics let an absolute value REPLACE the prefix and '..' escape it).
        # NOTE: deliberately on the ValidationConfig model_validator, NOT a MaskCondition
        # field_validator — same rationale as ConditioningItem.image: the Modal sample branch may
        # rebuild per-render condition views with container-ABSOLUTE resolved paths after load,
        # which must stay legal.
        for i, sample in enumerate(self.samples):
            for j, cond in enumerate(sample.conditions):
                if cond.type == "mask":
                    validate_volume_relative_path(
                        cond.video, field=f"samples[{i}].conditions[{j}].video"
                    )
                    validate_volume_relative_path(
                        cond.mask, field=f"samples[{i}].conditions[{j}].mask"
                    )
                elif cond.type == "audio":
                    validate_volume_relative_path(
                        cond.audio, field=f"samples[{i}].conditions[{j}].audio"
                    )
        return self

    @model_validator(mode="after")
    def _check_in_loop_sampling(self) -> "ValidationConfig":
        # Fail-fast PRE-approval (WR-04): an in-loop sample with no prompt is a misconfiguration, not
        # a silent no-op — the decoder would load (VRAM cost) with nothing to render. Reject at config
        # load, before any metered dispatch.
        if self.in_loop_sampling and not self.prompts:
            raise ValueError(
                "in_loop_sampling is True but validation.prompts is empty: the OFFL-02 in-loop "
                "sample renders one of validation.prompts every checkpoint_every steps — with no "
                "prompt the decoder would load (VRAM cost) with nothing to render. Add at least one "
                "prompt or set in_loop_sampling: false (fail-fast pre-approval, D-9-OFFL02-CLOSE)."
            )
        return self


class ConditioningItem(_Base):
    """One multi-frame keyframe: which pixel frame to anchor, its image, and its strength (D-6-ITEMS).

    An OBJECT (per D-6-ITEMS — NOT parallel image/index/strength lists) so a keyframe's three
    attributes travel together and can never desync across a config edit. ``image`` is carried as
    DATA — NOT FS-validated locally (Pitfall 1, same convention as ``reference_images``); existence
    is resolved Modal-side where the Volume mounts. ``frame_index`` alignment/range is validated by
    ``validate_conditioning_items`` at ``SignetConfig`` construction (it needs ``training_dims``).
    """

    image: str = Field(
        ...,
        description="Volume-relative reference-image path for this keyframe. DATA only — NOT "
        "FS-validated locally (Pitfall 1); resolved Modal-side.",
    )
    frame_index: int = Field(
        ...,
        description="Which pixel frame this keyframe anchors. Must be % TIME_SCALE (8) == 0 and in "
        "[0, F-1]; validated cross-field at SignetConfig load (needs training_dims).",
    )
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Conditioning strength for this keyframe; denoise_mask = 1 - strength.",
    )


class ConditioningConfig(_Base):
    """Reference-control block. Phase 5 adds single-frame, Phase 6 adds multi-frame; ``mode`` is an
    allowlist (D-NOHARDCODE).

    The conditioning probability + strength are TUNABLE config, never hardcoded (SC#1 / D-CONDP /
    D-STRENGTH). ``mode`` is fail-fast validated against the allowlist {none, single_frame,
    multi_frame}; Phase 7 widens it further. CONTRADICTION #2 (must-honor): the strength field is
    ``first_frame_conditioning_strength`` — the phantom noise-scale knob named in CONTEXT
    D-STRENGTH is deliberately absent (it does not exist on the ltx-trainer / ltx-core path
    signet uses).

    Phase 6 (REF-02) adds the multi-frame surface: an object-list ``conditioning_items`` plus the
    ``max_conditioning_items`` / ``conditioning_strength_range`` / ``conditioning_source`` knobs. A
    ``model_validator`` enforces the strength-range invariant and a BIDIRECTIONAL lean field-split
    so neither the Phase-5 nor the Phase-6 conditioning block is ever silently ignored.
    """

    mode: str = Field(default="none", description="Reference-control mode; Phases 5-7 add the rest.")
    first_frame_conditioning_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="P(condition on first frame) during training. Canonical LTX default is 0.1; "
        "signet ships 1.0 (always-condition) for the strongest single-frame anchoring (D-CONDP).",
    )
    first_frame_conditioning_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Conditioning strength; denoise_mask = 1 - strength (ltx_core ConditioningItem "
        "semantics). 1.0 = hard-clean/anchored = the ValidationSampler condition_image path. NOT "
        "yet consumed by ANY training path: the single_frame path is ALWAYS hard-clean "
        "(SingleFrameStrategy stores but never uses it) and multi_frame strengths flow from "
        "conditioning_items / conditioning_strength_range instead — so non-default values are "
        "REJECTED in both modes by the model_validator until a phase actually wires it (lean "
        "field-split, WR-03). NOTE: this is the canonical strength knob, NOT the phantom "
        "noise-scale field named in CONTEXT D-STRENGTH (CONTRADICTION #2).",
    )
    reference_images: list[str] = Field(
        default_factory=list,
        description="Volume-relative reference-image paths, paired positionally with "
        "validation.prompts to build the single-frame sample grid (D-REFIMG / D-GRID). "
        "Carried as DATA — NOT FS-validated locally (Pitfall 1); resolved Modal-side.",
    )
    original_videos: list[str] = Field(
        default_factory=list,
        description="Volume-relative ORIGINAL (photoreal) clip paths for the IC-LoRA re-skin grid "
        "col-1 (07-15 GAP-2 / D-NOHARDCODE), paired positionally with validation.prompts — same "
        "shape as reference_images but the SOURCE footage, not the seg-map. Carried as DATA — NOT "
        "FS-validated locally (Pitfall 1); resolved Modal-side under CHECKPOINTS_DIR. An ic_lora "
        "col-1 feature: rejected non-empty under mode != 'ic_lora' (lean field-split). Default []: "
        "absent -> col-1 falls back cleanly to 'original not staged'.",
    )

    # ---- Phase 6 (REF-02) multi-frame fields ----
    conditioning_items: list[ConditioningItem] = Field(
        default_factory=list,
        description="Multi-frame keyframes as OBJECTS {image, frame_index, strength} (D-6-ITEMS — "
        "NOT parallel lists). Only meaningful when mode == 'multi_frame'; frame_index alignment/range "
        "is validated cross-field at SignetConfig load.",
    )
    max_conditioning_items: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Upper bound on conditioning_items per sample (D-6-MAXITEMS). 3 = real keyframe "
        "training; bounded [1,8] so a runaway value can never reach a metered container.",
    )
    conditioning_strength_range: tuple[float, float] = Field(
        default=(0.3, 1.0),
        description="(lo, hi) strength-sampling range for multi-frame conditioning (D-6-STRENGTH-DIST). "
        "Default (0.3, 1.0); validated 0 <= lo <= hi <= 1 by the model_validator.",
    )
    conditioning_source: Literal["self", "paired"] = Field(
        default="self",
        description="Where conditioning latents come from (D-6-CONDSOURCE). Phase 6 accepted "
        "Literal['self'] ONLY; Phase 7 (REF-03) re-widens to Literal['self', 'paired'] now that the "
        "paired reference_latents/ pre-encode pipeline ships. 'paired' is only meaningful in "
        "mode == 'ic_lora' (the lean field-split rejects it in every other mode).",
    )

    # ---- Phase 7 (REF-03) IC-LoRA in-context video-to-video fields (D-NOHARDCODE) ----
    # Every IC-LoRA tunable is a documented, fail-fast config field BEFORE any GPU is touched. All
    # are only meaningful when mode == 'ic_lora'; the bidirectional lean field-split in the
    # model_validator rejects a non-default value under any other mode (never silently ignored).
    reference_latents_dir: str = Field(
        default="reference_latents",
        description="Sub-dir under preprocessed_data_root holding the paired REFERENCE latents "
        "(mirrors the upstream ltx-trainer video_to_video default). Existence is checked Modal-side "
        "where the Volume mounts, NOT locally (Pitfall 1). Only valid in mode == 'ic_lora'.",
    )
    reference_downscale_factor: int = Field(
        default=1,
        ge=1,
        description="Reference clip is encoded at 1/n the target resolution for token efficiency "
        "(canonical _infer_reference_downscale_factor is an INT; D-7-REF11 keeps 1:1 = 1 for the "
        "first proof — CONTEXT's '1.0' reconciled to int=1, RESEARCH A1). Only valid in "
        "mode == 'ic_lora'.",
    )
    reference_column: str = Field(
        default="reference_path",
        description="metadata.jsonl column naming each sample's reference clip, consumed by the "
        "paired preprocess (07-06/07-11). Only valid in mode == 'ic_lora'.",
    )
    seg_palette_name: str = Field(
        default="compact_driving_v1",
        description="D-7-PALETTE config-visible handle: names WHICH palette table in "
        "data/seg_palette.py (the PALETTES registry) the seg-map remap uses, so the palette is "
        "config-selected rather than an anonymous hardcode (D-NOHARDCODE). The seg staging/extraction "
        "reads this name (07-06/07-11). Only valid in mode == 'ic_lora'.",
    )
    frozen_adapter_path: str | None = Field(
        default=None,
        description="D-7-FREEZE: Volume-relative path to an existing LoRA adapter loaded FROZEN "
        "(official IC-LoRA or a community weight) that a new adapter trains on top of. Routed through "
        "validate_volume_relative_path (T-07-02). None = no frozen stacking. Only valid in "
        "mode == 'ic_lora'.",
    )
    frozen_adapter_format: Literal["peft", "comfy"] = Field(
        default="peft",
        description="D-7-FREEZE: format of frozen_adapter_path. 'peft' = a signet-produced adapter "
        "(no conversion); 'comfy' = a single-file comfy adapter needing comfy->PEFT key conversion. "
        "Only valid in mode == 'ic_lora'.",
    )

    # ---- Phase 9 (INPAINT) masked-region inpaint fields (GATE-SPEC-inpaint-a2v rev 2, D-NOHARDCODE) ----
    # Prior-project flexible/mask semantics on the ENCODED tensor: mask>0.5 tokens = clean latent +
    # timestep 0 + excluded from loss (KEEP); mask<=0.5 = noised + in loss (GENERATE). Both knobs are
    # documented, fail-fast config fields BEFORE any GPU is touched; only meaningful when
    # mode == 'inpaint' (the bidirectional lean field-split below — same shape as the ic_lora block).
    inpaint_mask_dir: str = Field(
        default="video_masks",
        description="Sub-dir under preprocessed_data_root holding the per-sample encoded mask "
        "tensors (one .pt per sample, same rel path as latents/; float32 [F_lat, H_lat, W_lat], "
        "values {0.,1.} thresholded at 0.5 at encode time). [precedent] prior-project flexible/mask "
        "dataset layout. Existence is checked Modal-side where the Volume mounts, NOT locally "
        "(Pitfall 1). Only valid in mode == 'inpaint'.",
    )
    inpaint_mask_probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="P(apply the spatial mask conditioning to a sample during training). "
        "[precedent] the prior project's always-masked inpaint config: mask prob 1.0 is the "
        "validated recipe. Only valid in mode == 'inpaint'.",
    )

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        # Fail-fast allowlist at model_validate time, before any GPU (D-NOHARDCODE / CONF-02).
        return validate_conditioning_mode(v)

    @model_validator(mode="after")
    def _check_multi_frame_fields(self) -> "ConditioningConfig":
        # 1) strength-range invariant: 0 <= lo <= hi <= 1 (fail-fast, names the offending range).
        lo, hi = self.conditioning_strength_range
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"invalid conditioning_strength_range {(lo, hi)!r}: require 0.0 <= lo <= hi <= 1.0 "
                f"(got lo={lo}, hi={hi})."
            )
        # 2) lean field-split (forward): conditioning_items only make sense in multi_frame mode.
        if self.conditioning_items and self.mode != "multi_frame":
            raise ValueError(
                f"conditioning_items is set but mode is {self.mode!r}: the object-list is only valid "
                f"when mode == 'multi_frame' (lean field-split — no silently-ignored config block)."
            )
        # 3) lean field-split (reverse): non-default Phase-5 fields must not linger in multi_frame
            #   mode, or a stale single-frame block would be silently ignored (RESEARCH Seam A pt 7).
        if self.mode == "multi_frame":
            if self.first_frame_conditioning_p != 1.0:
                raise ValueError(
                    "first_frame_conditioning_p is non-default while mode == 'multi_frame': the "
                    "Phase-5 single-frame field would be silently ignored; remove it (lean field-split)."
                )
            if self.first_frame_conditioning_strength != 1.0:
                raise ValueError(
                    "first_frame_conditioning_strength is non-default while mode == 'multi_frame': the "
                    "Phase-5 single-frame field would be silently ignored; remove it (lean field-split)."
                )
            if self.reference_images:
                raise ValueError(
                    "reference_images is non-empty while mode == 'multi_frame': the Phase-5 "
                    "single-frame field would be silently ignored; use conditioning_items instead "
                    "(lean field-split)."
                )
        # 4) lean field-split (WR-03): first_frame_conditioning_strength is consumed by NO
        #    single_frame path either — SingleFrameStrategy is hard-clean (strength ≡ 1.0; it
        #    stores the value without using it) and loop.py threads only the p knob. A config with
        #    mode: single_frame, strength: 0.5 would load clean and silently train at 1.0 — the
        #    exact silently-ignored-knob class this validator exists to prevent. Reject non-default
        #    values until a phase actually wires strength into the single_frame training path.
        if self.mode == "single_frame" and self.first_frame_conditioning_strength != 1.0:
            raise ValueError(
                "first_frame_conditioning_strength is non-default while mode == 'single_frame': "
                "the single-frame path is ALWAYS hard-clean (strength = 1.0; SingleFrameStrategy "
                "does not consume this knob yet), so the value would be silently ignored (lean "
                "field-split). Remove it (or keep the 1.0 default) until a phase wires strength "
                "into the single_frame training path."
            )
        # 5) T-06-02 / WR-09: enforce the documented "Volume-relative" contract on every
        #    operator-authored reference path AT CONFIG LOAD — pathlib join semantics make
        #    ``CHECKPOINTS_DIR / value`` REPLACE the prefix for an absolute value and let ``..``
        #    escape it. NOTE: this deliberately lives on the ConditioningConfig model_validator,
        #    NOT as a ConditioningItem.image field_validator — the Modal sample branch rebuilds
        #    per-column ConditioningItem views with container-ABSOLUTE resolved paths after load
        #    (fns.py: ``image=str(CHECKPOINTS_DIR / item.image)``), which must stay legal.
        for i, item in enumerate(self.conditioning_items):
            validate_volume_relative_path(item.image, field=f"conditioning_items[{i}].image")
        for i, ref in enumerate(self.reference_images):
            validate_volume_relative_path(ref, field=f"reference_images[{i}]")
        # 07-15 GAP-2: the col-1 ORIGINAL clips obey the SAME Volume-relative contract as the
        # reference seg-maps (REUSE the existing validator; no validators.py edit).
        for i, ov in enumerate(self.original_videos):
            validate_volume_relative_path(ov, field=f"original_videos[{i}]")
        # 6) Phase 7 (REF-03) IC-LoRA lean field-split — SAME bidirectional shape as the Phase-6
        #    multi_frame guards. FORWARD: the reference_* / seg_palette_name / frozen_adapter_*
        #    fields (and conditioning_source == 'paired') carry the in-context video-to-video surface,
        #    meaningful ONLY when mode == 'ic_lora'. REVERSE: a non-default value on any of them while
        #    mode != 'ic_lora' would be silently ignored, so it dies at config load (fail-fast).
        ic_lora_field_defaults = {
            "reference_latents_dir": "reference_latents",
            "reference_downscale_factor": 1,
            "reference_column": "reference_path",
            "seg_palette_name": "compact_driving_v1",
            "frozen_adapter_path": None,
            "frozen_adapter_format": "peft",
        }
        if self.mode != "ic_lora":
            nondefault = [
                name
                for name, default in ic_lora_field_defaults.items()
                if getattr(self, name) != default
            ]
            if self.conditioning_source == "paired":
                nondefault.append("conditioning_source='paired'")
            if nondefault:
                raise ValueError(
                    f"IC-LoRA field(s) {nondefault} set while mode is {self.mode!r}: the "
                    f"reference_*, seg_palette_name, frozen_adapter_*, and conditioning_source='paired' "
                    f"fields are only valid when mode == 'ic_lora' (lean field-split — no "
                    f"silently-ignored config block). Remove them or set mode: ic_lora."
                )
        # 6a-2) 07-15 GAP-2 REVERSE guard: original_videos is an ic_lora col-1 (re-skin grid) feature.
        #       A non-empty list under any other mode would be silently dropped on a metered render
        #       (same "no silently-ignored config block" shape as the reference_images-under-multi_frame
        #       reverse guard). Fail-fast at config load.
        if self.original_videos and self.mode != "ic_lora":
            raise ValueError(
                f"original_videos is non-empty while mode is {self.mode!r}: the col-1 ORIGINAL-clip "
                "list is an ic_lora re-skin-grid feature (07-15 GAP-2) and would be silently ignored "
                "in any other mode (lean field-split). Remove it or set mode: ic_lora."
            )
        # 6b) WR-01 (FORWARD, unwired-knob guard): reference_latents_dir is DOCUMENTED as the paired-
        #     reference sub-dir name, but NOTHING reads it — the 3-source dataset dir name is hardcoded
        #     as the literal "reference_latents" in ICLoraStrategy.get_data_sources() +
        #     _PRECOMPUTED_SOURCE_OUTPUT_KEYS (modal/fns.py). A config with reference_latents_dir:
        #     my_refs would validate clean, then either raise a confusing FileNotFoundError naming
        #     reference_latents/ (if only my_refs/ exists) or silently train on a stale reference_latents/
        #     dir if both exist — the exact silently-ignored-knob class this validator kills. Reject a
        #     non-default value in ic_lora mode until the field is actually threaded (mirrors rule 4).
        if self.mode == "ic_lora" and self.reference_latents_dir != "reference_latents":
            raise ValueError(
                f"reference_latents_dir is non-default ({self.reference_latents_dir!r}) while mode == "
                "'ic_lora', but NO code reads it yet — the 3-source dataset dir name is hardcoded as "
                "'reference_latents' (ICLoraStrategy.get_data_sources() + _PRECOMPUTED_SOURCE_OUTPUT_KEYS). "
                "The value would be silently ignored (WR-01). Keep the 'reference_latents' default until "
                "a phase threads the configured dir name through get_data_sources()."
            )
        # 6c) WR-02 (REVERSE, unwired-knob guard): first_frame_conditioning_p /
        #     first_frame_conditioning_strength are consumed by NO ic_lora path. loop.py threads only
        #     reference_downscale_factor for ic_lora (CR-01) and ICLoraStrategy keeps its own 0.0 p
        #     default; strength is consumed nowhere. A config with mode: ic_lora,
        #     first_frame_conditioning_p: 0.5 would load clean and silently train at 0.0 — the same
        #     silently-ignored-knob class rule 4 kills for single_frame. Reject non-default values in
        #     ic_lora mode until a phase wires them into the ic_lora training path (mirrors rule 4).
        if self.mode == "ic_lora":
            if self.first_frame_conditioning_p != 1.0:
                raise ValueError(
                    "first_frame_conditioning_p is non-default while mode == 'ic_lora': the ic_lora "
                    "path trains at p=0.0 (the reference prefix supplies the conditioning; loop.py "
                    "threads only reference_downscale_factor, and ICLoraStrategy keeps its 0.0 "
                    "default), so the value would be silently ignored (WR-02). Remove it (or keep the "
                    "1.0 default) until a phase wires first_frame_conditioning_p into ic_lora training."
                )
            if self.first_frame_conditioning_strength != 1.0:
                raise ValueError(
                    "first_frame_conditioning_strength is non-default while mode == 'ic_lora': it is "
                    "consumed by NO ic_lora training path, so the value would be silently ignored "
                    "(WR-02). Remove it (or keep the 1.0 default) until a phase wires strength in."
                )
        # 6d) Phase 9 (INPAINT) lean field-split — SAME bidirectional shape as the ic_lora rule 6.
        #     FORWARD: inpaint_mask_dir / inpaint_mask_probability carry the masked-region inpaint
        #     surface (prior-project flexible/mask semantics), meaningful ONLY when mode == 'inpaint'.
        #     REVERSE: a non-default value on either while mode != 'inpaint' would be silently
        #     ignored, so it dies at config load (fail-fast, no silently-ignored config block).
        inpaint_field_defaults = {
            "inpaint_mask_dir": "video_masks",
            "inpaint_mask_probability": 1.0,
        }
        if self.mode != "inpaint":
            nondefault = [
                name
                for name, default in inpaint_field_defaults.items()
                if getattr(self, name) != default
            ]
            if nondefault:
                raise ValueError(
                    f"inpaint field(s) {nondefault} set while mode is {self.mode!r}: the "
                    f"inpaint_mask_dir / inpaint_mask_probability fields are only valid when "
                    f"mode == 'inpaint' (lean field-split — no silently-ignored config block). "
                    f"Remove them or set mode: inpaint."
                )
        # 6e) Inpaint REVERSE guard for the single-frame knobs (mirrors rule 6c for ic_lora): the
        #     inpaint path derives its per-token conditioning from the spatial mask (mask>0.5 =
        #     clean + timestep 0), NOT from first-frame conditioning — first_frame_conditioning_p /
        #     _strength and reference_images are consumed by NO inpaint path, so non-default values
        #     would be silently ignored. Reject them at config load (lean field-split).
        if self.mode == "inpaint":
            if self.first_frame_conditioning_p != 1.0:
                raise ValueError(
                    "first_frame_conditioning_p is non-default while mode == 'inpaint': the inpaint "
                    "path derives conditioning from the spatial mask (mask>0.5 = clean + timestep 0), "
                    "not from first-frame conditioning, so the value would be silently ignored (lean "
                    "field-split). Remove it (or keep the 1.0 default)."
                )
            if self.first_frame_conditioning_strength != 1.0:
                raise ValueError(
                    "first_frame_conditioning_strength is non-default while mode == 'inpaint': it is "
                    "consumed by NO inpaint path (the mask is binary keep/generate), so the value "
                    "would be silently ignored (lean field-split). Remove it (or keep the 1.0 default)."
                )
            if self.reference_images:
                raise ValueError(
                    "reference_images is non-empty while mode == 'inpaint': the single-frame "
                    "reference-image list is consumed by NO inpaint path — masked test renders are "
                    "requested via validation.samples[].conditions (type: mask) instead (lean "
                    "field-split). Remove it."
                )
        # 7) T-07-02 / WR-09: the frozen-adapter path obeys the documented Volume-relative contract
        #    (no absolute path, no '..' escape) — it is joined under the checkpoints Volume prefix
        #    Modal-side, same as the reference-image paths.
        if self.frozen_adapter_path is not None:
            validate_volume_relative_path(self.frozen_adapter_path, field="frozen_adapter_path")
        return self


class DryRunConfig(_Base):
    """signet dry-run block — knobs for the CPU hard gate (CONF-03)."""

    n_text_tokens: int = Field(default=8, ge=1)
    ctx_dim: int = Field(default=16, ge=1)
    cond_first_frame: bool = Field(
        default=True,
        description="Exercise the first-frame conditioning region in the synthetic batch.",
    )


# --------------------------------------------------------------------------------------------------
# BK-01 — checkpoint auto-backup config (Phase 09.1, D-BK-1/2/5). Config-first, DEFAULT-OFF so every
# existing YAML loads byte-identically. This is the interface-first CONTRACT the Modal sync/restore
# consumers (09.1-08) read; NO backup activity happens unless enabled=True.
#
# BK-01 re-opens a deliberately-dropped path — enochiatron's HF-upload was cut from signet for
# privacy+cost (checkpoint.py docstring, T-03-44). Re-justified: a PRIVATE HF repo resolves privacy,
# a USER-determined destination resolves both, and off-Volume durability is the new value (a Volume
# is not a backup).
# --------------------------------------------------------------------------------------------------


class BackupConfig(_Base):
    """Checkpoint auto-backup contract (BK-01) — DEFAULT-OFF, mirror-all, private-HF.

    Every field is a documented, defaulted Pydantic ``Field`` (D-NOHARDCODE). ``enabled=False`` is
    the off-default that keeps existing behavior byte-identical (the checkpoint_expected_minutes /
    render_stall precedent). The house scope is ``what='all'`` — mirror EVERY completeness-checked
    checkpoint (D-BK-2): intermediates ARE the research artifacts (the reference campaign's picked likeness was
    step-3000 of a 5000-step run, not the final). ``final`` is a config narrowing for other users,
    NEVER the house default. Destination defaults to a PRIVATE HF repo (D-BK-5) — the durable
    off-Volume default.

    Scope note on destinations: ``hf`` and ``local`` are functionally wired this phase; ``cloud`` is
    schema-ready (enum + creds seam kept for forward-compat) but NOT implemented — an ENABLED cloud
    block FAILS FAST at config load (never deep in a Modal function).
    """

    enabled: bool = Field(
        default=False,
        description="[BK-01] Master gate. False = backup OFF, no behavior change (the None/off-default "
        "precedent; every existing YAML loads byte-identically). ALL backup activity is gated on this.",
    )
    destination: Literal["hf", "local", "cloud"] = Field(
        default="hf",
        description="[D-BK-5] Where completeness-checked checkpoints are mirrored. Default 'hf' = a "
        "PRIVATE HF repo (the durable off-Volume default). 'hf' and 'local' are WIRED this phase; "
        "'cloud' is a RESERVED forward-compat value (enum kept) that FAILS FAST at config load when "
        "enabled=True with a 'not yet implemented' error — never discovered deep in a Modal function. "
        "HF auth reuses the EXISTING modal.huggingface_secret_name (no second HF secret field here).",
    )
    repo_id: str | None = Field(
        default=None,
        description="[D-BK-5] Private HF repo id (owner/name) for destination='hf' (required when "
        "enabled + destination='hf'). None for other destinations. Auth via modal.huggingface_secret_name.",
    )
    path: str | None = Field(
        default=None,
        description="Local backup directory for destination='local' (required when enabled + "
        "destination='local'). None for other destinations.",
    )
    cloud_secret_name: str | None = Field(
        default=None,
        description="RESERVED forward-compat creds handle for the future destination='cloud' — a Modal "
        "secret NAME carried as DATA (mirroring modal.huggingface_secret_name), NEVER a literal bucket "
        "URL or credential value. cloud is NOT implemented this phase; an enabled cloud block fails fast.",
    )
    private: bool = Field(
        default=True,
        description="[D-BK-5] A backup HF repo defaults PRIVATE — privacy is what re-justified the "
        "deliberately-dropped HF-upload path (T-03-44). Only meaningful for destination='hf'.",
    )
    what: Literal["all", "final", "final+intervals"] = Field(
        default="all",
        description="[D-BK-2 HOUSE DEFAULT] Backup scope. 'all' = mirror EVERY completeness-checked "
        "checkpoint (the house default; intermediates ARE the research artifacts). 'final' is a "
        "narrowing for OTHER users and is NEVER the house default. 'final+intervals' = final + every "
        "'interval'-step boundary (requires 'interval').",
    )
    interval: int | None = Field(
        default=None,
        gt=0,
        description="Step boundary consumed ONLY when what='final+intervals' (back up every-N boundary "
        "+ final); must be None otherwise (a value the mode never consumes is rejected at load).",
    )

    @model_validator(mode="after")
    def _check_backup_fields(self) -> "BackupConfig":
        # Off block with defaults (incl. a disabled cloud block) loads clean — the whole block is
        # inert until enabled=True. Only fire the destination/field checks when enabled.
        if self.enabled:
            if self.destination == "hf":
                # hf consumes repo_id; a stray path / cloud_secret_name would be silently ignored.
                if self.repo_id is None:
                    raise ValueError(
                        "backup.destination='hf' requires backup.repo_id (the private HF repo id) "
                        "when enabled=True — a backup with no target repo would silently no-op durability."
                    )
                if self.path is not None:
                    raise ValueError(
                        "backup.path is set while destination='hf': path is only consumed by "
                        "destination='local' and would be silently ignored (symmetric stray-field "
                        "rejection). Remove it or switch destination."
                    )
                if self.cloud_secret_name is not None:
                    raise ValueError(
                        "backup.cloud_secret_name is set while destination='hf': it is a reserved "
                        "forward-compat field for destination='cloud' and would be silently ignored "
                        "(symmetric stray-field rejection). Remove it."
                    )
            elif self.destination == "local":
                # local consumes path; a stray repo_id / cloud_secret_name would be silently ignored.
                if self.path is None:
                    raise ValueError(
                        "backup.destination='local' requires backup.path (the local backup dir) when "
                        "enabled=True — a backup with no target dir would silently no-op durability."
                    )
                if self.repo_id is not None:
                    raise ValueError(
                        "backup.repo_id is set while destination='local': repo_id is only consumed by "
                        "destination='hf' and would be silently ignored (symmetric stray-field "
                        "rejection). Remove it or switch destination."
                    )
                if self.cloud_secret_name is not None:
                    raise ValueError(
                        "backup.cloud_secret_name is set while destination='local': it is a reserved "
                        "forward-compat field for destination='cloud' and would be silently ignored "
                        "(symmetric stray-field rejection). Remove it."
                    )
            elif self.destination == "cloud":
                # cloud is schema-ready but NOT implemented. FIRST reject any stray hf/local field
                # (the symmetric stray-field message), THEN fail fast even for an otherwise-clean
                # cloud block so a cloud misconfig can never reach the CPU Modal function (09.1-08).
                if self.repo_id is not None:
                    raise ValueError(
                        "backup.repo_id is set while destination='cloud': repo_id is only consumed by "
                        "destination='hf' and would be silently ignored (symmetric stray-field "
                        "rejection). Remove it."
                    )
                if self.path is not None:
                    raise ValueError(
                        "backup.path is set while destination='cloud': path is only consumed by "
                        "destination='local' and would be silently ignored (symmetric stray-field "
                        "rejection). Remove it."
                    )
                raise ValueError(
                    "backup.destination='cloud' is schema-ready but not yet implemented — only 'hf' "
                    "and 'local' are wired this phase (the enum value is KEPT for forward-compat; this "
                    "load-time error is the fail-fast so a cloud misconfig never reaches the CPU Modal "
                    "function)."
                )
        # interval is only consumed by what='final+intervals'; a value under any other scope (even
        # when disabled) would be silently ignored — reject it (lean field-split, mode-agnostic).
        if self.interval is not None and self.what != "final+intervals":
            raise ValueError(
                f"backup.interval is set ({self.interval}) while backup.what is {self.what!r}: interval "
                "is ONLY consumed when what='final+intervals' and would otherwise be silently ignored "
                "(lean field-split). Remove it or set what: final+intervals."
            )
        return self


# --------------------------------------------------------------------------------------------------
# The composing top-level config.
# --------------------------------------------------------------------------------------------------


class SignetConfig(_Base):
    """Top-level signet run config — composes LTX data + signet blocks (D-04).

    ``training_dims = [W, H, F]`` is the single source of truth for both the fail-fast
    validators (CONF-02) and the dry-run synthetic batch (CONF-03 / RESEARCH.md Q3/Open-Q2).
    """

    # signet-owned single source of truth for dims.
    training_dims: tuple[int, int, int] = Field(
        ...,
        description="[width, height, frames] — drives both the validators and the dry-run batch.",
    )

    # LTX data blocks (carried as data; native object hydrated Modal-side).
    data: DataConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    # The locked training recipe (required: ``training.max_steps`` is the run-specific step budget).
    training: TrainingConfig
    # Block-swap offloader field; baseline-first (blocks_to_swap=0) per D-OFF-1.
    offload: OffloadConfig = Field(default_factory=OffloadConfig)

    # signet blocks.
    modal: ModalConfig = Field(default_factory=ModalConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    conditioning: ConditioningConfig = Field(default_factory=ConditioningConfig)
    # Audio modality block (Phase 9 a2v) — modality-extensible; all fields reserved-default off, so
    # every existing (video-only) config loads byte-identically.
    audio: AudioConfig = Field(default_factory=AudioConfig)
    dry_run: DryRunConfig = Field(default_factory=DryRunConfig)
    # BK-01 checkpoint auto-backup — DEFAULT-OFF (enabled=False), so every existing YAML loads
    # byte-identically. The Modal sync/restore consumers land in 09.1-08.
    backup: BackupConfig = Field(default_factory=BackupConfig)
    # MiniMax-H3 family block (Phase 10) — every field defaulted, so every LTX config loads
    # byte-identically. Only meaningful when model.family == 'h3'; the REVERSE guard in
    # _cross_field_checks rejects a non-default value under any other family.
    h3: H3Config = Field(default_factory=H3Config)
    # Qwen-Image-Edit-2511 family block (family #3) — every field defaulted, so every LTX and H3
    # config loads byte-identically. Only meaningful when model.family == 'qwen_edit'; the REVERSE
    # guard in _cross_field_checks rejects a non-default value under any other family.
    qwen_edit: QwenEditConfig = Field(default_factory=QwenEditConfig)

    # top-level scalars mirrored from native config.
    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs")

    @field_validator("training_dims")
    @classmethod
    def _check_training_dims(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        # Fires at model_validate time — fail-fast, before any GPU (CONF-02 / D-06).
        #
        # Phase 10: a PRE-SCREEN over BOTH families' frame laws. Pydantic validates fields in
        # definition order and ``training_dims`` is the first field, so ``model.family`` is not yet
        # available here — the FAMILY-EXACT law is therefore re-asserted in ``_cross_field_checks``
        # below, where the family IS known. Dims invalid under BOTH laws still die right here with
        # the unchanged LTX message; only a dims triple that is legal for the OTHER family is passed
        # through, and the cross-field check then rejects it if the family does not match.
        try:
            return validate_training_dims(v)
        except ValueError as ltx_error:
            width, height, frames = v
            try:
                # H3 shares LTX's %32 spatial rules and differs ONLY in the frame law
                # ((F-5) % 17 vs (F-1) % 8), so reuse the spatial validators verbatim.
                validate_width(width)
                validate_height(height)
                validate_h3_frames(frames)
            except ValueError:
                raise ltx_error from None
            return (width, height, frames)

    def resolved_lora_targets(self) -> list[str] | str:
        """The LoRA targets this run will actually inject — explicit override, else family default.

        Resolution order: an explicit ``lora.target_modules`` wins; otherwise the H3 path regex when
        ``model.family == 'h3'``, or the Qwen 14-leaf path regex when ``model.family ==
        'qwen_edit'``; otherwise the ten LTX suffixes. ``_cross_field_checks`` writes the
        result back onto ``lora.target_modules`` at config load, so this accessor and the field
        agree — a consumer may read either.

        ⚠ The qwen_edit default is a REGEX for a different reason than H3's. H3 needed the anchor to
        EXCLUDE collateral (``token_refiner.refiner_blocks.*``, 12 modules). On Qwen the 14 suffixes
        happen to be collateral-free today; the anchor is there because ai-toolkit's real inclusion
        rule is "prefix ``transformer_blocks`` AND module type in {Linear, QLinear}"
        (``lora_special.py:342-434``) and PEFT has no type filter, so the anchored path regex is the
        only faithful expression of it — correct by construction rather than by coincidence of this
        checkpoint's top-level module names.
        """
        explicit = self.lora.target_modules
        if explicit is not None:
            return explicit
        if self.model.family == "h3":
            return H3_LORA_TARGET_REGEX
        if self.model.family == "qwen_edit":
            return QWEN_EDIT_LORA_TARGET_REGEX
        return list(LTX_DEFAULT_LORA_TARGETS)

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "SignetConfig":
        # Single-bucket is enforced on data.batch_size; re-assert here for a clear top-level error
        # if a future block ever carries a duplicate. (Cheap, fail-fast.)
        validate_batch_size(self.data.batch_size)
        # H3-02: resolve the FAMILY-SELECTED LoRA default and write it back, FIRST — before the a2v
        # target guard below reads it, and before any consumer ever sees the config. Writing back
        # (rather than only exposing resolved_lora_targets()) is deliberate: modal/fns.py and
        # train/validate_gate.py read cfg.lora.target_modules directly, and an H3 run that silently
        # injected the LTX-shaped list would NOT fail loud — see the LoraConfig.target_modules note.
        if self.lora.target_modules is None:
            self.lora.target_modules = self.resolved_lora_targets()
        elif not self.lora.target_modules:
            # An EXPLICITLY empty override is never a valid run — it injects zero adapters. It also
            # re-opens the exact silent failure this plan closes: both consumers spell the read
            # ``config.lora.target_modules or P1_FF_LORA_TARGETS`` (modal/fns.py, train/validate_gate.py),
            # so an empty value falls through to the LTX list — which on H3 matches 104 modules and
            # trains a wrong adapter without ever tripping loop.py's trainable-param guard.
            raise ValueError(
                "lora.target_modules is set but EMPTY: an empty target set injects no adapter at "
                "all, and the downstream `target_modules or P1_FF_LORA_TARGETS` fallback would "
                "silently substitute the LTX list (on an H3 run that trains a structurally wrong "
                "adapter without failing loud). Remove the key to take the family default, or list "
                "real targets."
            )
        # Phase 10 (H3-04) REVERSE guard — the bidirectional lean field-split, same shape as the
        # ic_lora / inpaint / a2v guards above. An H3 tunable set while the family is NOT h3 would be
        # silently ignored, so it dies at config load naming the offending field(s) and the family.
        # The defaults are read off a PRISTINE H3Config rather than a hand-written map, so a field
        # added later is covered automatically and the guard cannot drift out of sync (T-10-05-T).
        if self.model.family != "h3":
            pristine_h3 = H3Config()
            nondefault_h3 = [
                name
                for name in H3Config.model_fields
                if getattr(self.h3, name) != getattr(pristine_h3, name)
            ]
            if nondefault_h3:
                raise ValueError(
                    f"H3 field(s) {nondefault_h3} set while model.family is "
                    f"{self.model.family!r}: the h3 block (and the H3-only model IDs) is only valid "
                    f"when model.family == 'h3' (lean field-split — no silently-ignored config "
                    f"block). Remove them or set model.family: h3."
                )
        # Family #3 REVERSE guard — the same bidirectional lean field-split, same shape, same
        # PRISTINE-instance technique so a qwen_edit field added later is covered automatically and
        # the guard cannot drift out of sync with the block (T-10-05-T).
        if self.model.family != "qwen_edit":
            pristine_qe = QwenEditConfig()
            nondefault_qe = [
                name
                for name in QwenEditConfig.model_fields
                if getattr(self.qwen_edit, name) != getattr(pristine_qe, name)
            ]
            if nondefault_qe:
                raise ValueError(
                    f"qwen_edit field(s) {nondefault_qe} set while model.family is "
                    f"{self.model.family!r}: the qwen_edit block is only valid when "
                    f"model.family == 'qwen_edit' (lean field-split — no silently-ignored config "
                    f"block). Remove them or set model.family: qwen_edit."
                )
        # The family-only MODEL IDS (they live on ModelConfig — they are model IDs, not recipe
        # knobs — but are just as silently ignored under a family that reads none of them). This ran
        # as a flat "not h3 -> reject vae_id / audio_vae_id" list until family #3, which BREAKS on
        # qwen_edit: Qwen-Image-Edit ships its VAE as a separate file and legitimately needs vae_id.
        # An allowlist PER FIELD is the shape that survives a fourth family; widening the check to
        # "any non-ltx family" instead would have silently handed LTX a no-op knob back.
        for _id_field, _allowed_families in _FAMILY_ONLY_MODEL_IDS.items():
            if (
                getattr(self.model, _id_field) is not None
                and self.model.family not in _allowed_families
            ):
                raise ValueError(
                    f"model.{_id_field} is set while model.family is {self.model.family!r}: that "
                    f"ID is only read under family "
                    f"{{{', '.join(repr(f) for f in sorted(_allowed_families))}}} and would be "
                    f"silently ignored here (lean field-split — no silently-ignored config block). "
                    f"Remove it or set a family that consumes it."
                )
        # Phase 10 (H3-04) FAMILY-EXACT geometry + budget. The field-level validators above are only
        # a both-families PRE-SCREEN (a sub-model cannot see model.family, and training_dims is
        # validated before model is), so the exact law is decided HERE, where the family is known.
        # Cross-field by nature: geometry lives on training_dims, the reference set and fidelity on
        # the h3 block, the ceiling on the h3 budget triple — the same reason the inpaint dims check
        # lives here.
        if self.model.family == "h3":
            validate_h3_frames(self.training_dims[2])
            validate_h3_resolution_buckets(self.data.resolution_buckets)
            max_rows = max_packed_rows_for_budget(
                self.h3.gpu_usable_gib, self.h3.resident_gib, self.h3.mib_per_packed_row
            )
            if self.h3.character_reference_sizes or self.h3.environment_reference_sizes:
                # ⛔ validate_h3_reference_budget, NEVER h3_packed_seq_len on a nominal pair. Every
                # sample carries exactly references_per_sample slots, but the pairing REGIME varies
                # by design (D-10-ASYM), and the 15 pairs differ in row cost by up to 12%. At
                # reference short edge 1024 the nominal A+B pair prices at 12,362 rows and PASSES
                # while six of the twelve character-by-environment pairs are over the ceiling — so a
                # nominal-pair check passes at config load and then OOMs on the first
                # environment-bearing segment. This prices the WORST of the real pairing domain and
                # names the offending pair in the refusal (T-10-05-D).
                validate_h3_reference_budget(
                    target_frames=self.training_dims[2],
                    aspect=self.h3.target_aspect,
                    character_references=self.h3.character_reference_sizes,
                    environment_references=self.h3.environment_reference_sizes,
                    prompt_tokens=self.h3.prompt_tokens_estimate,
                    ref_short_edge=self.h3.reference_image_short_edge,
                    gpu_usable_gib=self.h3.gpu_usable_gib,
                    resident_gib=self.h3.resident_gib,
                    mib_per_packed_row=self.h3.mib_per_packed_row,
                    references_per_sample=self.h3.references_per_sample,
                )
            else:
                # No reference corpus declared. There is then exactly ONE possible layout, so worst
                # == nominal trivially and this is NOT the nominal-pair hole above. It still must be
                # priced: the 124f NO-REFERENCE t2v baseline is 37,806 rows against a ~13,777-row
                # ceiling, so campaign length busts the budget even with zero references — leaving a
                # reference-free H3 config unchecked would hand that OOM straight to a metered A100.
                layout = h3_packed_seq_len(
                    self.training_dims[2],
                    self.h3.target_aspect,
                    (),
                    self.h3.prompt_tokens_estimate,
                    self.h3.reference_image_short_edge,
                )
                validate_h3_seq_len_budget(layout.total, max_rows)
            # NO-REFERENCE (ALPHA) cross-block guard: reference_subject_ids selects which manifest
            # row supplies a render's reference slots — with zero slots there is nothing to select,
            # and h3_sample refuses no-reference rendering outright (the pinned diffusers ref2va
            # workflow is reference-conditioned), so a value here could only ever be silently
            # ignored. Same lean-field-split shape as the mask/audio condition-kind guard below;
            # cross-field (h3 block vs validation block), so it lives here.
            if self.h3.references_per_sample == 0 and self.validation.reference_subject_ids:
                raise ValueError(
                    f"validation.reference_subject_ids "
                    f"{list(self.validation.reference_subject_ids)} is set while "
                    f"h3.references_per_sample is 0 (NO-REFERENCE, ALPHA): there are no reference "
                    f"slots for it to select, and no-reference `--mode sample` is refused outright "
                    f"(lean field-split — no silently-ignored config block). Remove it."
                )
        elif self.model.family == "qwen_edit":
            # Family #3 FAMILY-EXACT geometry. Note what is NOT here: no pre-screen was widened for
            # qwen_edit, and none needed to be. Its frame law (F == 1 exactly) is a strict SUBSET of
            # LTX's ((F - 1) % 8 == 0, which admits 1), and its spatial rule is LTX's %32 verbatim —
            # so every qwen-legal geometry already passes the field-level pre-screens untouched.
            # The consequence runs the other way and is the whole reason this arm exists: an F of 9
            # sails through the pre-screen WITH LTX'S BLESSING and would reach the packer as a
            # nine-frame video request against an image model. These two lines are the only guard.
            validate_qwen_edit_frames(self.training_dims[2])
            validate_qwen_edit_resolution_buckets(self.data.resolution_buckets)
            # The chained-edit rank/alpha lock (rank == alpha == 42 across every round). CPU, at
            # config load, before any dispatch — a rank change mid-chain re-shapes lora_A/lora_B, so
            # it must be caught here and not by a partial state-dict load in a metered container.
            validate_qwen_edit_rank_alpha_lock(self.lora, self.qwen_edit, self.training)
            # Coverage over the FOURTEEN dual-stream leaves, on the RESOLVED targets — so it guards
            # the explicit override and the family default alike. Reads the written-back value from
            # the top of this method, which is why the write-back has to happen first.
            validate_qwen_edit_lora_coverage(self.resolved_lora_targets())
            # Chain INTERLOCK: covering all fourteen is enough for a fresh round but NOT for a warm
            # start. should_warm_start (train/loop.py:177-189) is a weights-only load at step 0, so
            # the resumed adapter must be shape-identical end to end — a SUPERSET re-shapes nothing
            # yet still changes the module set, and the extra modules would load from nothing.
            if self.training.init_adapter_path is not None:
                if self.lora.target_modules != QWEN_EDIT_LORA_TARGET_REGEX:
                    raise ValueError(
                        f"training.init_adapter_path is set "
                        f"({self.training.init_adapter_path!r}) but lora.target_modules is not the "
                        f"qwen_edit family default: a chained round must be SHAPE-IDENTICAL to the "
                        f"round it resumes, and covering all fourteen leaves is not the same as "
                        f"matching the module set (a superset warm-starts its extra modules from "
                        f"nothing). Remove the lora.target_modules override so the family default "
                        f"is used, or drop init_adapter_path to train a fresh adapter."
                    )
            # LTX reference-control modes are structurally foreign here: qwen_edit conditioning is
            # the multi-SLOT control-image mechanism declared on the qwen_edit block (encoded
            # through BOTH the VL encoder and the VAE), not a ConditioningStrategy mode. A
            # non-'none' mode would select machinery this family has none of.
            if self.conditioning.mode != "none":
                raise ValueError(
                    f"conditioning.mode is {self.conditioning.mode!r} while model.family is "
                    f"'qwen_edit': the LTX reference-control modes (single_frame / multi_frame / "
                    f"ic_lora / inpaint / audio_to_video) have no qwen_edit implementation — this "
                    f"family's conditioning is the multi-slot CONTROL-IMAGE mechanism declared on "
                    f"the qwen_edit block (control_slots / control_area_px / condition_area_px). "
                    f"Set conditioning.mode: none and configure the slots there."
                )
            # validation.frame_count defaults to 49 (an LTX clip length), so a qwen_edit config that
            # simply omits the validation block would request a 49-FRAME render from an image model.
            # Defaulting per-family would be the wrong fix — `validation` is shared, and a
            # family-conditional default is invisible in the YAML. Make the operator state it.
            if self.validation.frame_count != 1:
                raise ValueError(
                    f"validation.frame_count is {self.validation.frame_count} while model.family "
                    f"is 'qwen_edit': Qwen-Image-Edit is an IMAGE model and renders exactly one "
                    f"frame. The shared default is 49 (an LTX clip length), so this fires on a "
                    f"config that merely OMITS the validation block rather than one that gets it "
                    f"wrong. Set validation.frame_count: 1."
                )
            # A cached text embedding is computed ONCE per (caption, control) key, so a per-step
            # caption dropout can never reach it. ai-toolkit resolves the collision by HARD-DISABLING
            # caption dropout under caching (dataloader_mixins.py:402,417) — i.e. it silently trains
            # at rate 0. Refuse at config load instead; a knob that reads as set and behaves as unset
            # is the exact class the lean field-split exists to kill.
            # ⛔ REFUSED OUTRIGHT, not conditionally on caching. The earlier form of this check
            # fired only under `cache_text_embeddings: true` and offered `false` as the remedy —
            # and that remedy DOES NOT WORK. `cache_text_embeddings` has exactly one runtime
            # consumer in this tree (the re-encode SKIP at modal/fns.py:5560, inside
            # qwen_edit_preprocess); it does not switch training to live text encoding. Nothing
            # anywhere supplies `empty_text_conditions`, which is what
            # conditioning/qwen_edit.py:824 needs the moment a dropout draw fires.
            #
            # So an operator who hit the old refusal and followed its instruction got a config that
            # LOADS, dry-runs green, passes the cost gate, boots an A100, pays the ~40.9 GiB
            # transformer load + qfloat8 + LoRA inject, trains to roughly step 20 of 5000, and then
            # raises — eleven times over, because the train stage carries
            # retries=modal.Retries(max_retries=10) with single_use_containers=True. Eleven full
            # container starts, zero usable steps, and the config was following our own advice.
            #
            # A remedy that costs eleven A100 starts is worse than no remedy. Until an
            # empty-caption encode exists, the honest answer is that this family cannot do caption
            # dropout at all.
            if self.qwen_edit.caption_dropout_rate > 0.0:
                raise ValueError(
                    "qwen_edit.caption_dropout_rate > 0 is not supported on this family. Caption "
                    "dropout needs an EMPTY-CAPTION text payload to substitute in, and nothing in "
                    "this tree produces one — QwenEditStrategy.empty_text_conditions defaults to "
                    "None and no caller sets it, so the first dropout draw raises inside the "
                    "training step (conditioning/qwen_edit.py:824). This is refused at config load "
                    "because the failure would otherwise land AFTER the ~40.9 GiB load on a metered "
                    "A100, and repeat ten more times under the stage's retry policy.\n"
                    "        Set caption_dropout_rate: 0.0. Note this REPRODUCES the production "
                    "chain rather than departing from it: every proven house run trained at an "
                    "effective 0 anyway, because ai-toolkit hard-disables caption dropout whenever "
                    "cache_text_embeddings is on (dataloader_mixins.py:402,417) and all three "
                    "production Qwen chains ran with caching enabled.\n"
                    "        WHAT WOULD LAND IT: have qwen_edit_preprocess encode the empty caption "
                    "once into qwen_edit_conditions/ under a reserved key, and thread it as "
                    "QwenEditStrategy(empty_text_conditions=...) at modal/fns.py:5894. Do NOT reach "
                    "for cache_text_embeddings: false — that flag gates a preprocess-side re-encode "
                    "SKIP and has no training-side effect whatsoever."
                )
            # Price the packed sequence ALWAYS; refuse only against a ceiling someone MEASURED.
            # The layout call is not optional even with the ceiling disabled: it is what validates
            # control_slots against the ai-toolkit 1..3 cap and prompt_tokens_estimate positivity,
            # and it is what the dry-run banner reports. See QwenEditConfig.max_packed_rows for why
            # there is no derived ceiling here and why the banner must print
            # "ceiling=DISABLED (unmeasured)" rather than a headroom number.
            layout = qwen_edit_packed_layout(self.training_dims, self.qwen_edit)
            if self.qwen_edit.max_packed_rows:
                validate_qwen_edit_row_budget(layout.total, self.qwen_edit.max_packed_rows)
        else:
            # LTX family: re-assert the EXACT LTX law that the widened pre-screens deliberately let
            # through, so the widening created no hole. Identical messages, identical verdicts — an
            # LTX config sees byte-identical behaviour to before Phase 10.
            validate_training_dims(self.training_dims)
            validate_resolution_buckets(self.data.resolution_buckets)
        # Multi-frame keyframe indices need training_dims (F = training_dims[2]) to range-check, so
        # the per-item frame_index / strength validation lives here on SignetConfig, not on the
        # ConditioningConfig sub-model (REF-02 / D-6-FAILFAST).
        validate_conditioning_items(self.conditioning.conditioning_items, self.training_dims)
        # WR-05: the CONSUMER of conditioning_items is the SAMPLE path, which renders
        # validation.frame_count frames (_generation_kwargs: num_frames = v.frame_count) — NOT
        # training_dims F. A frame_index legal for training_dims but beyond the sample render
        # (e.g. F=49, frame_count=25, frame_index=48) would target a latent frame past the render
        # and only fail (or silently mis-condition) on the metered GPU. Bound by BOTH, fail-fast.
        if self.conditioning.mode == "multi_frame" and self.conditioning.conditioning_items:
            sample_frames = self.validation.frame_count
            for i, item in enumerate(self.conditioning.conditioning_items):
                if not (0 <= item.frame_index <= sample_frames - 1):
                    raise ValueError(
                        f"invalid conditioning_items[{i}].frame_index {item.frame_index}: must be "
                        f"in range 0..{sample_frames - 1} of the SAMPLE render "
                        f"(validation.frame_count = {sample_frames}) — the sample path renders "
                        f"frame_count frames, not training_dims F = {self.training_dims[2]}. "
                        f"Raise validation.frame_count or move the keyframe inside the render."
                    )
        # D-7-BASEVAR (REF-03): the DISTILLED base variant renders through the distilled two-stage
        # inference path (validation.two_stage_upscale -> inference/upscale.run_two_stage, which loads
        # the ltx-2.3-22b-distilled checkpoint + 2x spatial upscaler). Pairing a distilled model_id
        # with the DEV single-stage sample path (two_stage_upscale == False -> the ValidationSampler
        # run_sampler flow) would silently render the wrong substrate on a metered GPU. Fail fast at
        # config load: a distilled base REQUIRES the distilled inference path. (model_id already
        # switches dev/distilled by filename — cross-field, so it lives here, not on ConditioningConfig.)
        if "distilled" in self.model.model_id and not self.validation.two_stage_upscale:
            raise ValueError(
                f"invalid pairing: model.model_id {self.model.model_id!r} is the DISTILLED base "
                f"variant but validation.two_stage_upscale is False (the dev single-stage sample "
                f"path). The distilled variant must be paired with the distilled inference path — "
                f"set validation.two_stage_upscale: true (the two-stage distilled + spatial-upscaler "
                f"render). The dev single-stage path is for the dev base only (D-7-BASEVAR)."
            )
        # Phase 9 (INPAINT) ÷64 dims rule — STRICTER than the video-wide %32 rule, inpaint-mode
        # ONLY (every other mode keeps %32 unchanged). Cross-field: mode lives on conditioning,
        # dims/buckets on training_dims/data — so the check lives here, fail-fast at config load,
        # before any GPU (the GATE-SPEC rev-2 CPU gate). Applied to BOTH the resolution buckets
        # (what preprocess/training actually shape) and training_dims (what the dry-run synthetic
        # batch shapes) so the dry-run always exercises inpaint-legal dims.
        # [precedent] prior-project HANDOFF-2026-06-30-NIGHT '÷64 required for inpaint; validated by the
        # smoke' — all precedent-validated inpaint dims are ÷64 (1280x704 / 512x384 are clean).
        if self.conditioning.mode == "inpaint":
            validate_inpaint_dims(self.training_dims)
            validate_inpaint_resolution_buckets(self.data.resolution_buckets)
            # The VALIDATION render dims must ALSO satisfy ÷64 + 8n+1 (masked-render gap, pre-build
            # audit): run_mask_condition_sampler re-enforces this Modal-side at render time —
            # checking here moves the failure PRE-dispatch (CPU, zero-spend), same fail-fast
            # doctrine as the two lines above.
            validate_inpaint_dims(
                (self.validation.width, self.validation.height, self.validation.frame_count)
            )
        # Phase 9 lean field-split, cross-block: each validation-sample condition KIND is tied to a
        # conditioning.mode ('mask' -> inpaint held-out masked render; 'audio' -> a2v driving-audio
        # render). Under any other mode the sample path has no machinery for that kind, so the
        # condition would be silently ignored on a metered render — reject at config load (no
        # silently-ignored config block; same doctrine as the ic_lora reverse guards).
        _condition_kind_mode = {"mask": "inpaint", "audio": "audio_to_video"}
        for i, sample in enumerate(self.validation.samples):
            for cond in sample.conditions:
                required = _condition_kind_mode.get(cond.type)
                if required is not None and self.conditioning.mode != required:
                    raise ValueError(
                        f"validation.samples[{i}] carries a {cond.type!r} condition while "
                        f"conditioning.mode is {self.conditioning.mode!r}: the {cond.type!r} "
                        f"condition kind is a {required!r}-mode feature and would be silently "
                        f"ignored in any other mode (lean field-split). Remove the sample or set "
                        f"conditioning.mode: {required}."
                    )

        # Phase 9 (AUDIO-TO-VIDEO) cross-field guards (GATE-SPEC rev 2 item 2/3). All are fail-fast at
        # config load, before any GPU (mode lives on conditioning, the audio block + lora on
        # SignetConfig — so the checks live here, cross-field).
        audio_field_defaults = {
            "is_generated": False,
            "with_audio": False,
            "audio_latents_dir": "audio_latents",
            "generate_audio": False,
        }
        if self.conditioning.mode == "audio_to_video":
            # (a) the cross-modal audio->video attention LoRA targets MUST be present, or the adapter
            #     is structurally blind to the audio ("the #1 silent a2v failure", GATE-SPEC item 3).
            validate_a2v_lora_targets(self.lora.target_modules)
            # (b) the audio latents the A2VStrategy reads must actually get encoded: require the
            #     with_audio preprocess flag so a2v can never silently run on a video-only encode.
            if not self.audio.with_audio:
                raise ValueError(
                    "conditioning.mode == 'audio_to_video' but audio.with_audio is False: the a2v "
                    "strategy reads audio_latents/, which the canonical `--mode preprocess` encode "
                    "only writes when with_audio is True. Set audio.with_audio: true (fail-fast — a "
                    "video-only encode would leave the audio source missing at train time)."
                )
            # (c) a2v derives conditioning from the frozen audio branch, NOT from first-frame
            #     conditioning or single-frame reference images — reject non-default values (same
            #     silently-ignored-knob class as the inpaint / ic_lora reverse guards).
            if self.conditioning.first_frame_conditioning_p != 1.0:
                raise ValueError(
                    "first_frame_conditioning_p is non-default while mode == 'audio_to_video': the "
                    "a2v path conditions on the frozen audio branch, not first-frame conditioning, "
                    "so the value would be silently ignored (lean field-split). Remove it."
                )
            if self.conditioning.first_frame_conditioning_strength != 1.0:
                raise ValueError(
                    "first_frame_conditioning_strength is non-default while mode == "
                    "'audio_to_video': it is consumed by NO a2v path, so the value would be "
                    "silently ignored (lean field-split). Remove it."
                )
            if self.conditioning.reference_images:
                raise ValueError(
                    "reference_images is non-empty while mode == 'audio_to_video': the a2v path "
                    "consumes no single-frame reference images — driving-audio test renders are "
                    "requested via validation.samples[].conditions (type: audio) instead (lean "
                    "field-split). Remove it."
                )
        else:
            # (d) REVERSE guard: a non-default audio-block field under any non-a2v mode would be
            #     silently ignored — reject at config load.
            nondefault_audio = [
                name
                for name, default in audio_field_defaults.items()
                if getattr(self.audio, name) != default
            ]
            if nondefault_audio:
                raise ValueError(
                    f"audio field(s) {nondefault_audio} set while conditioning.mode is "
                    f"{self.conditioning.mode!r}: the audio block (is_generated / with_audio / "
                    f"audio_latents_dir / generate_audio) is only valid when "
                    f"mode == 'audio_to_video' (lean field-split — no silently-ignored config "
                    f"block). Remove them or set conditioning.mode: audio_to_video."
                )
        return self
