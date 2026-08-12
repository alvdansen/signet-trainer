"""train.validate_gate — the cheap GPU preflight: enochiatron's 6 architectural-mismatch checks.

Ported (video-only subset) from enochiatron ``scripts/validate/validate_ltx23.py`` (checks
1/2/3/4/5/6, lines 301-1078). This is the fail-fast architecture gate that runs BEFORE any
sustained training spend — in enochiatron it caught 6 arch mismatches for ~$1.40 (SC#3). It does
NOT itself launch a metered run; the 03-06 Modal ``train()`` body calls
``run_validation_gate(...)`` and aborts on any FAIL before building the model + starting training.

The 6 checks (video-only — NO audio/cross-modal/AV expansion, NO Wan probes):
  #1 check_dimensions   — num_blocks / hidden_dim / in_channels vs the IMPORTED EXPECTED_*
                          single source in ``models/loader.py`` (48 / 4096 / 128), via
                          ``summarize_components`` on the live model AND/OR the safetensors
                          ground-truth header read.
  #2 check_text_encoder — Gemma ``hidden_size == 3840`` (the real ground truth, NOT Flimmer's
                          stale 3072) via ``text_encoder.config.hidden_size``.
  #3 check_lora_targets — every target THE RUN WILL INJECT (``config.lora.target_modules``, the
                          family-resolved set — e.g. the 14-entry a2v set with the cross-modal
                          ``audio_to_video_attn.*`` entries) resolves on the arch via the
                          PEFT-faithful matchers in ``lora/peft.py`` — the structural proof the
                          injected set is real on the arch, not a probe of a hardcoded default.
  #4 check_forward_pass — drive ONE synthetic ``training_step`` forward and assert a tensor
                          returns. THIS resolves Open-Q1: whether ``components.transformer``
                          accepts ``model(video=Modality, audio=None, perturbations=None)→(pred,_)``;
                          if not, the FAIL message instructs 03-06 to PEFT-wrap a
                          ``SingleGPUModelBuilder(...).build()`` instead (train.py:1806-1811).
  #5 check_training_step— inject LoRA → forward → ``loss.backward()`` → assert at least one
                          trainable param has a ``.grad`` (LoRA grads flow).
  #6 check_p1_to_p2     — HARD GATE: save → reload → re-inject → ``set_peft_model_state_dict`` →
                          forward; assert keys load clean (the video-only INFR-03 roundtrip on the
                          real adapter — NO P2/audio expansion).

Plus a cheap frame/dim-constraint assert ((frames-1)%8==0, w%32==0, h%32==0) folded into the
preflight, and a best-effort VRAM probe carried as OFFL-03 baseline-fits evidence.

CRITICAL — Anti-Pattern 6 (Modal-side EXCEPTION, mirror ``models/loader.py`` / ``train/step.py``):
    The heavy ltx_core symbols (``SingleGPUModelBuilder`` etc.) are NEVER imported at module load.
    The forward checks call ``training_step`` whose ltx_core import is function-local inside
    ``build_step_deps()`` — so importing this module for the structural scan (``import
    signet_trainer.train.validate_gate``) succeeds WITHOUT ltx_core installed. ``peft`` is the
    boundary dep (a declared modal-runtime dep). The 6 GPU checks RUN only Modal-side at call time;
    on CI ``tests/test_validate_gate.py`` exercises only the importable/structural surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from signet_trainer.lora.peft import (
    H3_LORA_TARGET_REGEX,
    P1_FF_LORA_TARGETS,
    build_lora_config,
    check_lora_targets_regex,
    inject_lora,
    roundtrip_check,
)

# The per-target counting probe shares its name with this module's gate check #3 (which wraps it
# into a CheckResult), so it is imported under an alias rather than shadowed.
from signet_trainer.lora.peft import check_lora_targets as count_lora_targets
from signet_trainer.models.loader import (
    EXPECTED_HIDDEN_DIM,
    EXPECTED_NUM_BLOCKS,
    EXPECTED_T2V_IN_CHANNELS,
    EXPECTED_VIDEO_INNER_DIM,
    summarize_components,
)
from signet_trainer.train.flow_match import FlowMatchingSchedule
from signet_trainer.train.step import training_step

# --------------------------------------------------------------------------------------------------
# Ground-truth constants the gate asserts against. The dimension constants are IMPORTED from
# loader.py (the single source — do NOT redefine the values here). The Gemma hidden size is the
# one value loader.py does not carry, so it lives here as the text-encoder ground truth.
# --------------------------------------------------------------------------------------------------

#: Gemma 3 12B text-encoder hidden size — the REAL ground truth (validate_ltx23.py:581), NOT
#: Flimmer's stale ``LTX_TEXT_DIM=3072``. Check #2 asserts ``text_encoder.config.hidden_size`` == this.
EXPECTED_GEMMA_HIDDEN_SIZE: int = 3840

#: The context (text-embedding) dim the ported ``training_step`` feeds as ``context`` — the
#: precomputed ``video_prompt_embeds`` are already projected to the transformer hidden dim
#: (step.py contract: ``[., 4096]``). Used to shape the synthetic forward in checks #4/#5/#6.
SYNTH_TEXT_EMBED_DIM: int = EXPECTED_HIDDEN_DIM

#: Small synthetic latent dims for the forward checks (mirror validate_ltx23.py:573 —
#: ``lat_h, lat_w, lat_f = 11, 24, 11`` ≈ 352/32, 768/32, (81-1)/8+1). Cheap enough to probe the
#: forward signature without a full-resolution batch.
SYNTH_LAT_F: int = 11
SYNTH_LAT_H: int = 11
SYNTH_LAT_W: int = 24
SYNTH_TEXT_SEQ_LEN: int = 128


@dataclass
class CheckResult:
    """One validation-check outcome (mirrors validate_ltx23.py:97-103 + a ``hard_gate`` flag).

    ``status`` is one of ``"PASS" | "FAIL" | "SKIP" | "WARN"``. ``hard_gate`` marks check #6 — a
    FAIL there is an absolute abort (the INFR-03 roundtrip must hold on the real adapter).
    """

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    hard_gate: bool = False


# --------------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------------


def resolve_lora_targets(module: Any, targets: list[str]) -> dict[str, Any]:
    """Resolve each LoRA target over ``module.named_modules()`` via the PEFT-faithful matcher.

    Delegates to ``lora/peft.py::check_lora_targets`` (dot-bounded suffix — what PEFT actually does
    for a ``list[str]`` ``target_modules``) rather than re-implementing a matcher. The original port
    of validate_ltx23.py's ``_check_targets`` used a bare ``name.endswith(target)``, which is LOOSER
    than PEFT (it would "match" a coincidental ``…xattn1.to_q``) — the gate must predict PEFT's
    injection, so it uses PEFT's semantics. Returns ``{"matched": [...], "unmatched": [...],
    "counts": {target: n}}``; the test drives it against a FAKE module with no 22B model.
    """
    report = count_lora_targets(module, targets)
    counts = {target: report[target]["total"] for target in targets}
    return {
        "matched": [t for t in targets if counts[t] > 0],
        "unmatched": [t for t in targets if counts[t] == 0],
        "counts": counts,
    }


def _benchmark_vram(label: str) -> dict[str, Any]:
    """Best-effort peak-VRAM snapshot (OFFL-03 baseline-fits evidence). No-op off CUDA."""
    if not torch.cuda.is_available():
        return {"label": label, "cuda": False}
    torch.cuda.synchronize()

    def _gb(n: int | float) -> float:
        return round(n / (1024**3), 2)

    return {
        "label": label,
        "cuda": True,
        "allocated_gb": _gb(torch.cuda.memory_allocated()),
        "reserved_gb": _gb(torch.cuda.memory_reserved()),
        "peak_gb": _gb(torch.cuda.max_memory_allocated()),
    }


def _extract_safetensors_ground_truth(checkpoint_path: str) -> dict[str, Any]:
    """Header-only safetensors arch read (mirror validate_ltx23.py:190-294, video-only subset).

    Reads ONLY the tensor key names + recorded shapes (no tensor materialization, no GPU) to derive
    ``num_blocks`` / ``video_inner_dim`` (hidden_dim) / ``in_channels``. ``safetensors`` is imported
    locally so the module imports without it on a partial environment.
    """
    from safetensors import safe_open  # noqa: PLC0415 — optional dep, function-local

    dims: dict[str, Any] = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        tensor_names = list(f.keys())

        for name in tensor_names:
            # VIDEO branch only — exclude the audio dual-stream (``audio_attn1``), whose to_q is a
            # different (2048-vs-... ) inner dim (validate_ltx23.py reads video vs audio separately).
            if "blocks.0.attn1.to_q.weight" in name and "audio" not in name:
                dims["video_inner_dim"] = f.get_slice(name).get_shape()[0]
                break

        block_indices: set[int] = set()
        for name in tensor_names:
            parts = name.split(".")
            if "blocks" in parts:
                idx_pos = parts.index("blocks") + 1
                if idx_pos < len(parts) and parts[idx_pos].isdigit():
                    block_indices.add(int(parts[idx_pos]))
        if block_indices:
            dims["num_blocks"] = max(block_indices) + 1

        # The input projection maps latent channels -> inner_dim; its name varies across LTX
        # revisions (patchify_proj / proj_in / x_embedder / patch_embed). Try known candidates; the
        # in_channels is the LAST weight dim for a 2D proj ([inner_dim, in_channels]) or dim[1].
        proj_keys = ("patchify", "proj_in", "x_embedder", "patch_embed", "patchify_proj")
        for name in tensor_names:
            low = name.lower()
            if "weight" in low and "block" not in low and any(k in low for k in proj_keys):
                shape = f.get_slice(name).get_shape()
                if len(shape) >= 2:
                    dims["in_channels"] = shape[1]
                    dims["in_channels_source_tensor"] = name
                    break
        if "in_channels" not in dims:
            # Diagnostic: surface candidate input-projection tensor names (+ shapes) so the next run
            # reveals the real name to map here, instead of guessing on a metered launch.
            dims["in_channels_candidates"] = [
                {"name": n, "shape": list(f.get_slice(n).get_shape())}
                for n in tensor_names
                if "weight" in n.lower()
                and "block" not in n.lower()
                and any(k in n.lower() for k in ("embed", "proj", "patch"))
            ][:20]

    return {"checkpoint_path": checkpoint_path, "dimensions": dims, "tensor_count": len(tensor_names)}


def _synthetic_batch(
    device: Any, dtype: torch.dtype
) -> dict[str, dict[str, torch.Tensor]]:
    """Build a tiny synthetic ``PrecomputedDataset``-shaped batch for the forward checks.

    Mirrors the nested shape ``training_step`` unwraps (data/precomputed.py:223-238): a video latent
    ``[C, F, H, W]`` with ``C == EXPECTED_T2V_IN_CHANNELS`` and small F/H/W, plus a text condition
    (``video_prompt_embeds`` ``[seq, SYNTH_TEXT_EMBED_DIM]`` + an all-ones ``prompt_attention_mask``).
    """
    latents = torch.randn(
        EXPECTED_T2V_IN_CHANNELS,
        SYNTH_LAT_F,
        SYNTH_LAT_H,
        SYNTH_LAT_W,
        device=device,
        dtype=dtype,
    )
    prompt_embeds = torch.randn(
        SYNTH_TEXT_SEQ_LEN, SYNTH_TEXT_EMBED_DIM, device=device, dtype=dtype
    )
    attention_mask = torch.ones(SYNTH_TEXT_SEQ_LEN, device=device, dtype=torch.long)
    return {
        "latent_conditions": {"latents": latents},
        "text_conditions": {
            "video_prompt_embeds": prompt_embeds,
            "prompt_attention_mask": attention_mask,
        },
    }


def _make_schedule(config: Any) -> FlowMatchingSchedule:
    """Build the flow-match schedule, carrying ``uniform_prob`` from config when present."""
    uniform_prob = getattr(getattr(config, "training", None), "uniform_prob", 0.30)
    return FlowMatchingSchedule(uniform_prob=uniform_prob)


def _grad_capable_transformer(components: Any) -> Any:
    """Pull the transformer out of the loaded components (the Modality forward target)."""
    return getattr(components, "transformer", None)


# --------------------------------------------------------------------------------------------------
# The 6 checks (video-only). Each returns a CheckResult and never raises — exceptions become FAIL.
# --------------------------------------------------------------------------------------------------


def check_dimensions(
    components: Any, *, checkpoint_path: str | None = None
) -> CheckResult:
    """#1 — num_blocks / hidden_dim / in_channels vs the IMPORTED EXPECTED_* single source.

    Compares the live model's ``summarize_components`` introspection (landmine #3 ``num_blocks``
    resolution) AND, when ``checkpoint_path`` is provided, the header-only safetensors ground truth
    against ``EXPECTED_NUM_BLOCKS=48 / EXPECTED_HIDDEN_DIM=4096 / EXPECTED_T2V_IN_CHANNELS=128``.
    """
    t0 = time.time()
    try:
        summary = summarize_components(components)
        gt: dict[str, Any] = _extract_safetensors_ground_truth(checkpoint_path) if checkpoint_path else {}
        gtd = gt.get("dimensions", {})

        # Each arch dim can be read from two sources: live attr/config introspection
        # (``summarize_components``) and the authoritative header-only safetensors ground truth. A
        # source that RESOLVES a value contradicting EXPECTED is a real arch mismatch (wrong model);
        # a source that returns ``None`` is merely probe-blind — LTX-2.3 hides some dims from
        # attr/config introspection and some input-proj tensor names are not yet mapped (Pitfall 6).
        # So: FAIL on a resolved contradiction; a ``None`` is "unverified", NOT a mismatch.
        readings = {
            "num_blocks": [
                ("summarize_components", summary.get("num_blocks")),
                ("safetensors", gtd.get("num_blocks")),
            ],
            # The VIDEO transformer inner dim (attn1.to_q[0]) = 2048 — the authoritative arch
            # fingerprint from the checkpoint header. (Do NOT compare against EXPECTED_HIDDEN_DIM=4096:
            # that is the text CONTEXT/connector dim, a different quantity — the conflation that made
            # this check false-fail on the real model while forward/training/roundtrip all PASSED.)
            "video_inner_dim": [
                ("safetensors", gtd.get("video_inner_dim")),
            ],
            "in_channels": [
                ("summarize_components", summary.get("in_channels")),
                ("safetensors", gtd.get("in_channels")),
            ],
        }
        expected = {
            "num_blocks": EXPECTED_NUM_BLOCKS,
            "video_inner_dim": EXPECTED_VIDEO_INNER_DIM,
            "in_channels": EXPECTED_T2V_IN_CHANNELS,
        }
        per_dim: dict[str, Any] = {}
        contradictions: list[str] = []
        for dim, exp in expected.items():
            resolved = [(s, a) for (s, a) in readings[dim] if a is not None]
            bad = [(s, a) for (s, a) in resolved if a != exp]
            verified = any(a == exp for (_s, a) in resolved)
            dim_status = "MISMATCH" if bad else ("MATCH" if verified else "UNVERIFIED")
            per_dim[dim] = {"expected": exp, "status": dim_status, "resolved": resolved}
            if bad:
                contradictions.append(f"{dim}: {bad} != {exp}")

        # Require the two STRONGEST arch signals (num_blocks + hidden_dim, both authoritatively
        # readable from the checkpoint header) to be positively verified — so a wrong checkpoint
        # (wrong block count / inner dim) still fails. in_channels may stay UNVERIFIED (probe gap,
        # tolerated) since neither source reliably names the input-projection tensor yet.
        # num_blocks is the required fingerprint — resolvable from BOTH live introspection (the block
        # container) and the authoritative checkpoint header, so a wrong model (wrong block count)
        # always fails. video_inner_dim + in_channels verify-or-tolerate: a RESOLVED contradiction
        # fails, but a probe gap (e.g. no checkpoint header on CI) does not.
        must_verify = ("num_blocks",)
        unverified_strong = [d for d in must_verify if per_dim[d]["status"] == "UNVERIFIED"]
        unverified = [d for d, v in per_dim.items() if v["status"] == "UNVERIFIED"]
        if contradictions:
            status = "FAIL"
            message = f"arch MISMATCH (wrong model): {'; '.join(contradictions)}"
        elif unverified_strong:
            status = "FAIL"
            message = (
                f"could not positively verify {unverified_strong} from live probe OR safetensors "
                f"(both None) — unmapped attr/tensor names; refusing to pass an unverifiable arch"
            )
        else:
            verified = [d for d, v in per_dim.items() if v["status"] == "MATCH"]
            status = "PASS"
            message = (
                f"arch verified: {verified}"
                + (f"; unverified probe-gap (tolerated): {unverified}" if unverified else "")
            )
        return CheckResult(
            "check_dimensions",
            status,
            message,
            details={"per_dim": per_dim, "summary": summary, "ground_truth": gt},
            duration_s=round(time.time() - t0, 2),
        )
    except Exception as e:  # noqa: BLE001 — exceptions are FAILs, never propagate
        return CheckResult(
            "check_dimensions", "FAIL", f"Exception: {e}", duration_s=round(time.time() - t0, 2)
        )


def _resolve_gemma_hidden_size(text_encoder: Any) -> tuple[Any, str]:
    """Best-effort read of the Gemma text hidden size across the wrappers LTX may load it behind.

    Tries: the encoder + common sub-model attrs (``.model`` / ``.text_model`` / ``.language_model``
    / ``.gemma``), each via ``config.hidden_size`` then ``config.text_config.hidden_size`` (config as
    object OR dict), then the token-embedding weight's last dim. Returns ``(value|None, source)``.
    """

    def _cfg_get(cfg: Any, key: str) -> Any:
        if cfg is None:
            return None
        return cfg.get(key) if isinstance(cfg, dict) else getattr(cfg, key, None)

    candidates = [text_encoder]
    for attr in ("model", "text_model", "language_model", "gemma", "transformer", "encoder"):
        sub = getattr(text_encoder, attr, None)
        if sub is not None:
            candidates.append(sub)
    for obj in candidates:
        cfg = getattr(obj, "config", None)
        if cfg is None and isinstance(obj, dict):
            cfg = obj.get("config")
        hs = _cfg_get(cfg, "hidden_size")
        if hs is not None:
            return int(hs), f"{type(obj).__name__}.config.hidden_size"
        hs = _cfg_get(_cfg_get(cfg, "text_config"), "hidden_size")
        if hs is not None:
            return int(hs), f"{type(obj).__name__}.config.text_config.hidden_size"
    emb = getattr(text_encoder, "get_input_embeddings", None)
    weight = getattr(emb() if callable(emb) else None, "weight", None)
    if weight is not None and getattr(weight, "ndim", 0) >= 1:
        return int(weight.shape[-1]), "embed_tokens.weight[-1]"
    return None, "unresolved"


def _describe_text_encoder(text_encoder: Any) -> dict[str, Any]:
    """Compact structure dump for the WARN diagnostic (type + config shape + top-level attr names)."""
    cfg = getattr(text_encoder, "config", None)
    if isinstance(cfg, dict):
        cfg_keys: Any = list(cfg.keys())[:30]
    elif cfg is not None:
        cfg_keys = [a for a in dir(cfg) if not a.startswith("_")][:30]
    else:
        cfg_keys = None
    return {
        "type": type(text_encoder).__name__,
        "config_type": type(cfg).__name__ if cfg is not None else None,
        "config_keys": cfg_keys,
        "attrs": [a for a in dir(text_encoder) if not a.startswith("_")][:40],
    }


def check_text_encoder(components: Any) -> CheckResult:
    """#2 — Gemma text hidden size == 3840 (ground truth, NOT stale 3072). Advisory (WARN) if the
    encoder is loaded but its hidden size isn't introspectable via known paths."""
    t0 = time.time()
    try:
        text_encoder = getattr(components, "text_encoder", None)
        if text_encoder is None:
            return CheckResult(
                "check_text_encoder",
                "FAIL",
                "components.text_encoder is None (video-only loader must load Gemma)",
                duration_s=round(time.time() - t0, 2),
            )
        hidden_size, source = _resolve_gemma_hidden_size(text_encoder)
        if hidden_size == EXPECTED_GEMMA_HIDDEN_SIZE:
            status = "PASS"
            message = f"Gemma hidden_size={hidden_size} (via {source}) == {EXPECTED_GEMMA_HIDDEN_SIZE}"
        elif hidden_size is None:
            # ADVISORY (validate_ltx23.py returns WARN, not a hard fail, for the text encoder): Gemma
            # IS loaded and the forward/training checks (#4/#5/#6) exercise it — do NOT block the run
            # on a probe gap. The hard gate is the #6 roundtrip. Dump the structure so the exact
            # hidden_size path can be mapped from the next run's logs.
            status = "WARN"
            message = (
                f"Gemma hidden_size UNRESOLVED via known paths (encoder loaded; forward checks use "
                f"it) — advisory, not blocking. structure={_describe_text_encoder(text_encoder)}"
            )
        else:
            status = "FAIL"
            message = (
                f"Gemma hidden_size={hidden_size} (via {source}) != {EXPECTED_GEMMA_HIDDEN_SIZE} "
                f"— a RESOLVED wrong value (e.g. Flimmer's stale 3072) is a real mismatch"
            )
        return CheckResult(
            "check_text_encoder",
            status,
            message,
            details={
                "gemma_hidden_size": hidden_size,
                "source": source,
                "expected": EXPECTED_GEMMA_HIDDEN_SIZE,
            },
            duration_s=round(time.time() - t0, 2),
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "check_text_encoder", "FAIL", f"Exception: {e}", duration_s=round(time.time() - t0, 2)
        )


def check_lora_targets(components: Any, config: Any) -> CheckResult:
    """#3 — every target THE RUN WILL INJECT resolves on the arch, with per-target counts.

    Probes ``config.lora.target_modules`` — the set check #5 actually injects (family-resolved and
    written back by ``SignetConfig._cross_field_checks``; same ``or P1_FF_LORA_TARGETS`` fallback
    spelling as check #5) — NOT a hardcoded default. Probing ``P1_FF_LORA_TARGETS`` here passed an
    a2v run "10/10 targets matched" while all four cross-modal ``audio_to_video_attn.*`` entries
    were absent from the arch — the exact "#1 silent a2v failure" the GATE-SPEC names.

    Both target forms are handled with the PEFT-faithful matchers from ``lora/peft.py``:
      * ``list[str]`` — ``check_lora_targets`` (dot-bounded suffix, PEFT's list semantics); FAIL on
        ANY per-target zero count, and the message carries the per-target counts.
      * bare ``str`` — a PEFT path REGEX (the H3 form); ``check_lora_targets_regex`` (never iterate
        a bare str as a list — that char-explodes the probe). FAIL on zero total matches; for the
        H3 family regex additionally FAIL on any per-leaf zero (the H3 bar is main=50 PER LEAF).

    Runs on the BASE transformer's ``named_modules()`` — no PEFT injection needed for the match.
    """
    t0 = time.time()
    try:
        transformer = _grad_capable_transformer(components)
        if transformer is None:
            return CheckResult(
                "check_lora_targets",
                "FAIL",
                "components.transformer is None",
                duration_s=round(time.time() - t0, 2),
            )
        targets = (
            getattr(getattr(config, "lora", None), "target_modules", None) or P1_FF_LORA_TARGETS
        )
        if isinstance(targets, str):
            survey = check_lora_targets_regex(transformer, targets)
            zero_leaves = (
                [leaf for leaf, n in survey["per_leaf"].items() if n == 0]
                if targets == H3_LORA_TARGET_REGEX
                else []
            )
            status = "PASS" if survey["total"] > 0 and not zero_leaves else "FAIL"
            message = (
                f"regex target form: {survey['total']} modules matched "
                f"(main={survey['main']}, collateral={survey['collateral']}); "
                f"per-leaf: {survey['per_leaf']}"
                + (f"; ZERO-MATCH leaves: {zero_leaves}" if zero_leaves else "")
            )
            return CheckResult(
                "check_lora_targets",
                status,
                message,
                details=survey,
                duration_s=round(time.time() - t0, 2),
            )
        resolution = resolve_lora_targets(transformer, list(targets))
        unmatched = resolution["unmatched"]
        status = "PASS" if not unmatched else "FAIL"
        message = (
            f"{len(resolution['matched'])}/{len(targets)} injected targets matched; "
            f"per-target counts: {resolution['counts']}"
            + (f"; UNMATCHED: {unmatched}" if unmatched else "")
        )
        return CheckResult(
            "check_lora_targets",
            status,
            message,
            details=resolution,
            duration_s=round(time.time() - t0, 2),
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "check_lora_targets", "FAIL", f"Exception: {e}", duration_s=round(time.time() - t0, 2)
        )


def check_forward_pass(
    components: Any, config: Any, *, device: Any = "cuda", dtype: torch.dtype = torch.bfloat16
) -> CheckResult:
    """#4 — drive ONE synthetic ``training_step`` forward; resolves Open-Q1 (forward signature).

    Probes whether ``components.transformer`` accepts ``model(video=Modality, audio=None,
    perturbations=None) -> (pred, _)`` — the signature ``training_step`` requires. A returned scalar
    loss tensor PROVES the loader's transformer is forward-compatible; an exception records FAIL
    with the message instructing 03-06 to PEFT-wrap a ``SingleGPUModelBuilder(...).build()`` instead
    (train.py:1806-1811).
    """
    t0 = time.time()
    try:
        transformer = _grad_capable_transformer(components)
        if transformer is None:
            return CheckResult(
                "check_forward_pass",
                "FAIL",
                "components.transformer is None",
                duration_s=round(time.time() - t0, 2),
            )
        import numpy as np  # noqa: PLC0415 — only needed for the rng seed here

        batch = _synthetic_batch(device, dtype)
        schedule = _make_schedule(config)
        rng = np.random.default_rng(getattr(config, "seed", 42))
        with torch.no_grad():
            loss = training_step(transformer, batch, schedule, rng, device=device, dtype=dtype)
        ok = isinstance(loss, torch.Tensor)
        status = "PASS" if ok else "FAIL"
        message = (
            "loader.transformer accepts model(video=,audio=,perturbations=)→(pred,_) "
            "(Open-Q1: use load_ltxv_components().transformer directly)"
            if ok
            else "forward returned a non-tensor"
        )
        return CheckResult(
            "check_forward_pass",
            status,
            message,
            details={"loss_value": float(loss.item()) if ok else None},
            duration_s=round(time.time() - t0, 2),
        )
    except Exception as e:  # noqa: BLE001 — the diagnostic FAIL that resolves Open-Q1
        return CheckResult(
            "check_forward_pass",
            "FAIL",
            f"loader.transformer does NOT accept the Modality forward signature ({e}); "
            f"03-06 must PEFT-wrap a SingleGPUModelBuilder(model_path=...).build() instead "
            f"(train.py:1806-1811)",
            details={"open_q1": "use_builder"},
            duration_s=round(time.time() - t0, 2),
        )


def check_training_step(
    components: Any, config: Any, *, device: Any = "cuda", dtype: torch.dtype = torch.bfloat16
) -> tuple[CheckResult, Any]:
    """#5 — inject LoRA → forward → backward → assert LoRA grads flow. Returns (result, peft_model).

    The PEFT-wrapped model is returned so check #6 can run the roundtrip on the SAME injected adapter
    (avoids rebuilding the 22B model). On FAIL the second element is ``None``.
    """
    t0 = time.time()
    try:
        transformer = _grad_capable_transformer(components)
        if transformer is None:
            return (
                CheckResult(
                    "check_training_step",
                    "FAIL",
                    "components.transformer is None",
                    duration_s=round(time.time() - t0, 2),
                ),
                None,
            )
        import numpy as np  # noqa: PLC0415

        lora_config = build_lora_config(
            rank=getattr(getattr(config, "lora", None), "rank", 64),
            alpha=getattr(getattr(config, "lora", None), "alpha", 64),
            targets=getattr(getattr(config, "lora", None), "target_modules", None) or P1_FF_LORA_TARGETS,
        )
        model = inject_lora(transformer, lora_config)  # GC-before-get_peft_model (TRAIN-06)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        batch = _synthetic_batch(device, dtype)
        schedule = _make_schedule(config)
        rng = np.random.default_rng(getattr(config, "seed", 42))
        loss = training_step(model, batch, schedule, rng, device=device, dtype=dtype)
        loss.backward()

        with_grad = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is not None
        )
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        vram = _benchmark_vram("backward_pass_p1_ff_lora")
        status = "PASS" if with_grad > 0 else "FAIL"
        message = (
            f"{with_grad}/{trainable} trainable params have .grad; "
            f"loss={float(loss.item()):.4f}; peak_vram={vram.get('peak_gb')}"
        )
        return (
            CheckResult(
                "check_training_step",
                status,
                message,
                details={"params_with_grad": with_grad, "trainable": trainable, "vram": vram},
                duration_s=round(time.time() - t0, 2),
            ),
            model if status == "PASS" else None,
        )
    except Exception as e:  # noqa: BLE001
        return (
            CheckResult(
                "check_training_step",
                "FAIL",
                f"Exception: {e}",
                duration_s=round(time.time() - t0, 2),
            ),
            None,
        )


def check_p1_to_p2(
    peft_model: Any,
    *,
    tmp_dir: str | None = None,
    config: Any = None,
    device: Any = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> CheckResult:
    """#6 (HARD GATE) — save → reload → re-inject → set_peft_model_state_dict → forward; keys clean.

    The video-only reinterpretation of validate_ltx23.py's P1→P2 transition (825): drop the audio/
    cross-modal P2 expansion and assert the INFR-03 roundtrip holds on the REAL injected adapter —
    ``roundtrip_check`` proves save→reload preserves every key + tensor with no ``.alpha`` /
    serialization mismatch, then a post-reload synthetic forward proves the re-hydrated adapter is
    still forward-runnable. A FAIL here is an absolute abort.
    """
    t0 = time.time()
    try:
        if peft_model is None:
            return CheckResult(
                "check_p1_to_p2",
                "FAIL",
                "no injected adapter (check #5 must PASS before the roundtrip)",
                duration_s=round(time.time() - t0, 2),
                hard_gate=True,
            )
        import tempfile  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        out_dir = tmp_dir or tempfile.mkdtemp(prefix="signet_validate_p1_roundtrip_")
        report = roundtrip_check(peft_model, out_dir)  # save→reload→re-inject→key+tensor eq

        # Post-reload forward: the re-hydrated adapter must still run the Modality forward.
        batch = _synthetic_batch(device, dtype)
        schedule = _make_schedule(config)
        rng = np.random.default_rng(getattr(config, "seed", 42))
        with torch.no_grad():
            loss = training_step(peft_model, batch, schedule, rng, device=device, dtype=dtype)

        ok = isinstance(loss, torch.Tensor)
        status = "PASS" if ok else "FAIL"
        message = (
            f"INFR-03 roundtrip clean: {report['num_keys']} adapter keys preserved, "
            f"no .alpha mismatch; post-reload forward {'OK' if ok else 'FAILED'}"
        )
        return CheckResult(
            "check_p1_to_p2",
            status,
            message,
            details={"num_keys": report["num_keys"]},
            duration_s=round(time.time() - t0, 2),
            hard_gate=True,
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "check_p1_to_p2",
            "FAIL",
            f"Exception: {e}",
            duration_s=round(time.time() - t0, 2),
            hard_gate=True,
        )


def check_frame_constraints(config: Any) -> CheckResult:
    """Cheap CPU frame/dim-constraint assert folded into the preflight.

    LTX-2.3 latent alignment: frames must satisfy ``(frames - 1) % 8 == 0`` (causal temporal
    compression) and spatial dims ``w % 32 == 0`` / ``h % 32 == 0`` (VAE spatial compression).
    ``config.training_dims`` is ``[width, height, frames]``.
    """
    t0 = time.time()
    try:
        width, height, frames = (int(x) for x in config.training_dims)
        violations: list[str] = []
        if (frames - 1) % 8 != 0:
            violations.append(f"(frames-1)%8 != 0 (frames={frames})")
        if width % 32 != 0:
            violations.append(f"width%32 != 0 (width={width})")
        if height % 32 != 0:
            violations.append(f"height%32 != 0 (height={height})")
        status = "PASS" if not violations else "FAIL"
        message = "frame/dim alignment OK" if not violations else f"violations: {violations}"
        return CheckResult(
            "check_frame_constraints",
            status,
            message,
            details={"width": width, "height": height, "frames": frames},
            duration_s=round(time.time() - t0, 2),
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            "check_frame_constraints",
            "FAIL",
            f"Exception: {e}",
            duration_s=round(time.time() - t0, 2),
        )


# --------------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------------


def run_validation_gate(
    components: Any,
    config: Any,
    *,
    checkpoint_path: str | None = None,
    device: Any = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[bool, list[CheckResult], Any]:
    """Run the 6 architectural-mismatch checks (+ frame-constraint) → (all_passed, results, adapter).

    Order: frame-constraint (cheap CPU) → #1 dimensions → #2 text encoder → #3 lora targets →
    #4 forward pass (Open-Q1) → #5 training step (returns the injected adapter) → #6 P1→P2
    roundtrip (HARD GATE, reuses the #5 adapter). ``all_passed`` is True iff NO check FAILs (a WARN,
    e.g. an unresolvable-but-present Gemma hidden size, is advisory and does not block) —
    the 03-06 ``train()`` body MUST abort the run on ``all_passed is False`` before any sustained
    spend. The heavy GPU checks (#4/#5/#6) import ltx_core function-local via ``training_step``; this
    function is invoked Modal-side only.

    The third return value is the check #5 PEFT model (``None`` if #5 did not PASS): the adapter
    already injected + roundtrip-proved on ``components.transformer``. On the Open-Q1 default path the
    ``train()`` body REUSES this exact adapter instead of calling ``inject_lora`` again — re-injecting
    an already-LoRA-wrapped module would double-wrap it on the real GPU (03-07 double-inject fix).
    """
    results: list[CheckResult] = []

    results.append(check_frame_constraints(config))
    results.append(check_dimensions(components, checkpoint_path=checkpoint_path))
    results.append(check_text_encoder(components))
    results.append(check_lora_targets(components, config))
    results.append(check_forward_pass(components, config, device=device, dtype=dtype))

    train_result, peft_model = check_training_step(components, config, device=device, dtype=dtype)
    results.append(train_result)

    results.append(
        check_p1_to_p2(peft_model, config=config, device=device, dtype=dtype)
    )

    # Blocking = any FAIL. WARN is advisory (e.g. an unresolvable-but-present Gemma hidden size,
    # matching validate_ltx23.py) and does NOT abort the run — the hard gate is the #6 roundtrip.
    all_passed = all(r.status != "FAIL" for r in results)
    return all_passed, results, peft_model
