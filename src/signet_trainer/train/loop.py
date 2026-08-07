"""train.loop — the step-driven ``train_loop`` driver + ``build_optimizer`` / ``build_scheduler``.

Ported from enochiatron ``scripts/train/train.py``: the video-only flat ``build_optimizer`` branch
(1431-1447), ``build_scheduler`` (1455-1480), and the step-driven ``train()`` body (1764+). It
consumes ``train/step.py`` (the forward), ``train/flow_match.py`` (03-01, the schedule), and
``train/checkpoint.py`` (the save/resume manager).

Three locked-recipe facts (SignetConfig.training):
  - optimizer = ``bnb.optim.AdamW8bit`` (lr 5e-5, wd 0.01, betas (0.9, 0.999), eps 1e-8). The
    banned adaptive optimizer and the AV per-stream-LR branches are DROPPED (single-stream N/A).
  - scheduler = cosine ``LambdaLR`` parameterized from the flimmer template — ``min_lr_ratio`` 0.01
    (the operator's deviation from the source 0.1) and ``warmup_steps`` 0.
  - the loop is STEP-driven: ``max_steps`` is authoritative (NOT epoch-driven — HANDOFF landmine
    44), and a checkpoint is "done" only once ``checkpoints_vol.commit()`` lands the file on the
    Volume (commit-or-vanish — a "done" LOG LINE is printed early by Modal SIGTERM).

NO Wan carry-over: no MoE / second-transformer expert / per-stream LR / non-flow schedulers / Wan
canonical inference params (CLAUDE.md "What NOT to Use").

CRITICAL — Anti-Pattern 6: ``torch`` is a base dep (fine at module top), but ``bitsandbytes`` and
the ltx_core seam (via ``train/step.py``) are imported FUNCTION-LOCAL so this module imports for a
structural scan / config-load without bnb installed.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Optimizer — adamw8bit (video-only flat-param branch)
# --------------------------------------------------------------------------------------------------


def build_optimizer(model: Any, config: Any) -> Any:
    """Build the locked ``bnb.optim.AdamW8bit`` over the trainable (LoRA) params.

    Collects only ``requires_grad`` params (the injected LoRA tensors; the frozen base is excluded)
    and raises if none are trainable — the canonical "is LoRA applied?" guard (train.py:1432-1434).
    ``bitsandbytes`` is imported function-local (it is a GPU-side dep; the loop must import on CPU).
    """
    import bitsandbytes as bnb  # noqa: PLC0415 — GPU-side dep, function-local (Anti-Pattern 6)

    t = config.training
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Is LoRA applied?")

    return bnb.optim.AdamW8bit(
        trainable_params,
        lr=t.learning_rate,
        weight_decay=t.weight_decay,
        betas=tuple(t.betas),
        eps=t.eps,
    )


# --------------------------------------------------------------------------------------------------
# Scheduler — cosine LambdaLR with a min-LR floor (flimmer template values)
# --------------------------------------------------------------------------------------------------


def build_scheduler(optimizer: Any, config: Any, total_steps: int) -> Any:
    """Build a cosine ``LambdaLR`` with a ``min_lr_ratio`` floor and ``warmup_steps`` warmup.

    Floor = ``min_lr_ratio`` (0.01) of the peak LR; warmup = ``warmup_steps`` (0) linear ramp.
    Mirrors enochiatron ``build_scheduler`` (1466-1475) but parameterized from the flimmer template
    (``cosine_with_min_lr`` / ``min_lr_ratio: 0.01`` / ``warmup_steps: 0``) — the operator's deviation from
    the source's 0.1 floor / ``min(50, total//10)`` warmup.
    """
    t = config.training
    warmup_steps = t.warmup_steps
    floor = t.min_lr_ratio  # min_lr / lr == min_lr_ratio

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(floor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------------------------------
# Conditioning kwargs (Seam D) — the mode-dispatched knobs threaded into training_step
# --------------------------------------------------------------------------------------------------


def build_conditioning_kwargs(conditioning: Any) -> dict[str, Any]:
    """Build the ``training_step`` conditioning kwargs from the conditioning config (Seam D, 06-05).

    Mode-dispatched so each mode threads ONLY the knobs its strategy actually consumes — a config
    knob that reaches ``training_step`` but no strategy is the silently-ignored-knob class this
    project's fail-fast doctrine forbids. ``mode == "none"`` reduces to the current unconditioned
    behavior (byte-equivalent to Phase 3/5).

      * ``multi_frame`` carries the strength-in-training primitives;
      * ``single_frame`` carries the first-frame p;
      * ``inpaint`` carries ``inpaint_mask_probability`` (the per-sample mask-apply Bernoulli gate;
        [precedent] prior-project mask prob 1.0 — the schema default);
      * ``ic_lora`` carries ``reference_downscale_factor`` (the RoPE reference-position scale; CR-01).
        It deliberately does NOT thread ``first_frame_conditioning_p`` — its schema default (1.0)
        would silently flip ic_lora training to always-condition-first-frame, whereas the strategy's
        ic_lora default is 0.0 (the reference prefix already supplies the conditioning). The schema
        rejects a non-default ``first_frame_conditioning_p`` in ic_lora mode (WR-02), so ic_lora
        always keeps the strategy's 0.0.
    """
    c = conditioning
    kwargs: dict[str, Any] = {"conditioning_mode": c.mode}
    if c.mode == "multi_frame":
        kwargs["max_conditioning_items"] = c.max_conditioning_items
        kwargs["conditioning_strength_range"] = tuple(c.conditioning_strength_range)
    elif c.mode == "single_frame":
        kwargs["first_frame_conditioning_p"] = c.first_frame_conditioning_p
    elif c.mode == "ic_lora":
        kwargs["reference_downscale_factor"] = c.reference_downscale_factor
    elif c.mode == "inpaint":
        kwargs["inpaint_mask_probability"] = c.inpaint_mask_probability
    return kwargs


# --------------------------------------------------------------------------------------------------
# OFFL-02 in-loop validation-sample seam — PURE CPU-testable decision helpers (D-9-OFFL02-CLOSE)
# --------------------------------------------------------------------------------------------------


def in_loop_decoder_enabled(validation_cfg: Any) -> bool:
    """Whether the TRAIN path should load the video VAE decoder for in-loop sampling (OFFL-02).

    True iff ``validation_cfg.in_loop_sampling`` is set AND at least one prompt is configured. The
    decoder is loaded ONLY when both hold (config-first, D-NOHARDCODE): the byte-identical default
    (``in_loop_sampling`` False) keeps the decoder off, so every existing training call site is
    unchanged and the ~72GB decoder-on residency is opt-in. The empty-prompts case is already
    rejected at config load (``ValidationConfig._check_in_loop_sampling``); this helper stays
    defensive so the loader knob never turns on with nothing to render.

    Pure (no modal / no ltx_* / no GPU) so the decision is CPU-testable in isolation.
    """
    return bool(getattr(validation_cfg, "in_loop_sampling", False)) and bool(
        getattr(validation_cfg, "prompts", None)
    )


def in_loop_sample_due(
    step: int, checkpoint_every: int, decoder_ready: bool, has_prompts: bool
) -> bool:
    """Whether an in-loop validation sample fires at ``step`` (OFFL-02 cadence gate).

    True iff there is a prompt to render (``has_prompts``), the decoder is present
    (``decoder_ready`` — the loader loaded it because :func:`in_loop_decoder_enabled` was True), AND
    ``step`` lands on the checkpoint cadence (``step % checkpoint_every == 0``). Coupled to
    ``checkpoint_every`` (RESEARCH Pattern 1) — NOT ``validation.interval`` — so the in-loop sample
    rides the SAME boundary as the checkpoint commit. ``checkpoint_every <= 0`` never fires (a
    defensive guard against a modulo-by-zero; the schema pins ``checkpoint_every >= 1``).

    Pure (no modal / no ltx_* / no GPU) so the cadence gate is CPU-testable in isolation.
    """
    if not (has_prompts and decoder_ready):
        return False
    if checkpoint_every <= 0:
        return False
    return step % checkpoint_every == 0


# --------------------------------------------------------------------------------------------------
# Chained warm-restart — PURE CPU-testable cold-start precedence helper (D-9-CHAINED)
# --------------------------------------------------------------------------------------------------


def should_warm_start(has_in_dir_checkpoint: bool, init_adapter_path: Any) -> bool:
    """Whether to cold-start warm-start the trainable adapter from a prior round (D-9-CHAINED).

    True iff ``init_adapter_path`` is set AND there is NO in-dir checkpoint. An in-dir checkpoint
    means a same-dir ``resume()`` is in play (landmine #1) — that path re-injects the saved adapter
    and restores optimizer / scheduler / step, so it takes PRECEDENCE and ``init_adapter_path`` is
    IGNORED (a chained round resumed mid-run must not silently re-warm from the prior round). Only at
    a true cold start (no checkpoint) does the chained warm-restart load the prior adapter with a
    FRESH optimizer at step 0 (prior-project ``--p1-checkpoint`` semantics — no optimizer-state carry).

    Pure (no modal / no ltx_* / no GPU) so the precedence rule is CPU-testable in isolation.
    """
    return init_adapter_path is not None and not has_in_dir_checkpoint


# --------------------------------------------------------------------------------------------------
# In-loop liveness watchdog — PURE CPU-testable stall gate (AUDIT #11, config-gated, OFF by default)
# --------------------------------------------------------------------------------------------------


def _peak_gib(probe: Any) -> float:
    """``torch.cuda.max_memory_*`` in GiB, or ``0.0`` when there is no CUDA context.

    CPU-safe on purpose: ``train_loop`` is exercised by the CPU suite, and a VRAM gauge that raised
    on a CPU box would make the campaign's most useful log line the one nobody could test.
    """
    try:
        if not torch.cuda.is_available():
            return 0.0
        return float(probe()) / 2**30
    except Exception:  # noqa: BLE001 — a gauge must never be able to fail a training run
        return 0.0


def checkpoint_watchdog_exceeded(
    elapsed_minutes: float, expected_minutes: float | None, multiplier: float
) -> bool:
    """Whether the training cadence has stalled past the liveness deadline (AUDIT #11).

    True iff a watchdog is armed (``expected_minutes is not None``) AND the wall-clock minutes since
    the last COMMITTED checkpoint exceed ``expected_minutes * multiplier`` (K). ``expected_minutes
    is None`` → watchdog OFF → always False, so the None-default keeps ``train_loop`` byte-identical
    to today (matches the ``keep_checkpoints`` None-default precedent). Strict ``>``: exactly at the
    threshold is not yet exceeded.

    Scope (audit-scoped): this catches a wedged/slow cadence BETWEEN optimizer steps — a dataloader
    wedge, a degraded-throughput crawl, or a preempted-but-not-crashed container that stops committing
    — NOT a hard C-level hang inside a single forward (no Python frame runs to check the clock there).
    When it trips, ``train_loop`` raises; ``train()`` carries ``modal.Retries(max_retries=3)``, so the
    raise becomes an F9-retried IN-DIR resume from the latest committed checkpoint rather than a
    silent burn to the 24h ceiling.

    Pure (no modal / no ltx_* / no GPU) so the stall gate is CPU-testable in isolation.
    """
    if expected_minutes is None:
        return False
    return elapsed_minutes > expected_minutes * multiplier


# --------------------------------------------------------------------------------------------------
# The step-driven training loop
# --------------------------------------------------------------------------------------------------


def train_loop(
    model: Any,
    dataset: Any,
    optimizer: Any,
    scheduler: Any,
    schedule: Any,
    ckpt_manager: Any,
    config: Any,
    checkpoints_vol: Any = None,
    on_checkpoint: Any = None,
    step_fn: Any = None,
    collate_fn: Any = None,
) -> int:
    """Drive the step-driven loop: resume → forward/backward/step → checkpoint+commit.

    Args:
        model: the PEFT-wrapped LTX-2.3 model (forward via ``train/step.training_step``).
        dataset: a ``PrecomputedDataset`` (wrapped in a bs=1 ``DataLoader`` here).
        optimizer / scheduler: from ``build_optimizer`` / ``build_scheduler``.
        schedule: the 03-01 ``FlowMatchingSchedule`` (carrying ``uniform_prob`` from config).
        ckpt_manager: a ``train/checkpoint.CheckpointManager`` (resume re-injects the adapter —
            landmine #1 — before the loop starts).
        config: the ``SignetConfig`` (the locked ``training`` recipe).
        checkpoints_vol: the Modal checkpoints Volume; ``.commit()`` is called after each save so a
            preemption can't vanish an uncommitted checkpoint. ``None`` off-Modal.
        on_checkpoint: optional ``callable(step: int)`` invoked at every checkpoint boundary,
            AFTER ``ckpt_manager.save`` AND after the checkpoint's FIRST ``checkpoints_vol.commit()``
            (#12) — the just-saved checkpoint is made durable on the Volume BEFORE this callback
            runs, so a preemption during the ~8-13 min render can't vanish it. Anything the callback
            writes (e.g. the OFFL-02 in-loop validation sample, D-9-OFFL02-CLOSE) rides the SECOND
            commit that follows the callback (commit-or-vanish). A callback exception is logged and
            swallowed: a failed render must not kill a metered training round (samples-not-loss —
            the checkpoint itself is already saved AND committed when the callback runs).
        step_fn: optional ``callable(model, batch, schedule, rng, *, device, dtype) -> loss`` — the
            per-step FORWARD. ``None`` (the default) builds the LTX one from ``train/step.py``, so
            every existing call site is unchanged. It is an argument because this loop is otherwise
            model-agnostic and the default is NOT: ``build_step_deps()`` imports ``ltx_core``, which
            ``h3_gpu_image`` deliberately does not carry (10-04), and ``training_step`` assembles LTX
            ``Modality`` inputs the MiniMax-H3 single-stream forward has no analog for. Injecting
            here is what lets ``h3_train`` REUSE the resume / accumulate / clip / save →
            commit → callback → commit cadence instead of forking it — the crown jewels are the
            cadence, not the forward.
        collate_fn: optional ``DataLoader`` collate. ``None`` keeps torch's default. H3 passes
            ``lambda b: b[0]`` because its reference payload is a LIST of per-slot dicts carrying
            strings and differently-shaped tensors, which the default collate would restructure;
            ``batch_size`` is LOCKED at 1 anyway, so taking the sample through untouched loses
            nothing.

    Returns:
        The final ``global_step`` reached (== ``config.training.max_steps`` on a clean finish).
    """
    import numpy as np  # noqa: PLC0415 — function-local to keep the structural-scan path light
    from torch.utils.data import DataLoader  # noqa: PLC0415

    t = config.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if t.mixed_precision == "bf16" else torch.float32
    rng = np.random.default_rng(t.seed)

    if step_fn is None:
        # The LTX default, built here so the ltx_core import NEVER runs when a caller injects its
        # own forward (h3_gpu_image has no ltx_core — the import itself would be the failure).
        from signet_trainer.train.step import build_step_deps, training_step  # noqa: PLC0415

        deps = build_step_deps()  # construct the ltx_core seam ONCE (not per step)

        # Thread the conditioning config into the forward driver ONCE (config is static — 06-05
        # Seam D). mode == "none" reduces to the current unconditioned behavior (byte-equivalent to
        # Phase 3/5); multi_frame carries the strength-in-training primitives, single_frame the
        # first-frame p, and ic_lora the reference_downscale_factor (CR-01 — previously never
        # threaded, so a valid reference_downscale_factor: 2 config silently trained at factor 1).
        # See build_conditioning_kwargs.
        conditioning_kwargs = build_conditioning_kwargs(config.conditioning)

        def step_fn(  # noqa: PLR0913 — mirrors the injected-seam signature exactly
            model_: Any, batch_: Any, schedule_: Any, rng_: Any, *, device: Any, dtype: Any
        ) -> Any:
            return training_step(
                model_, batch_, schedule_, rng_, device=device, dtype=dtype, deps=deps,
                **conditioning_kwargs,
            )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # bs=1 LOCKED (multi-bucket: every sample carries its own shape)
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    total_steps = t.max_steps  # STEP-driven, authoritative (NOT epoch — HANDOFF landmine 44)
    grad_accum = t.gradient_accumulation_steps
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Resume re-injects the adapter AND restores optimizer/scheduler/step/RNG (landmine #1).
    global_step = ckpt_manager.resume(model, optimizer, scheduler)
    if global_step >= total_steps:
        logger.info("Already at step %d >= max_steps %d — nothing to do.", global_step, total_steps)
        return global_step

    micro = 0
    last_loss = torch.tensor(0.0)
    # AUDIT #11 liveness watchdog — wall-clock of the last COMMITTED checkpoint. Armed only when
    # config.training.checkpoint_expected_minutes is set (None → checkpoint_watchdog_exceeded returns
    # False every step → byte-identical to today). Refreshed at EVERY commit() below (D-NOHARDCODE:
    # the deadline is read from config, never a code literal).
    last_commit = time.monotonic()
    while global_step < total_steps:
        for batch in dataloader:
            raw_loss = step_fn(model, batch, schedule, rng, device=device, dtype=dtype)
            last_loss = raw_loss.detach()
            (raw_loss / grad_accum).backward()
            micro += 1
            if micro % grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params, t.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # AUDIT #11 — liveness gate: if no checkpoint has committed for longer than
            # checkpoint_expected_minutes * checkpoint_stall_multiplier, treat the round as
            # hung/preempted and RAISE. train() carries modal.Retries(max_retries=3), so this
            # becomes an F9 in-dir resume from the latest committed checkpoint instead of a silent
            # burn to the 24h ceiling. Inert while checkpoint_expected_minutes is None.
            elapsed_minutes = (time.monotonic() - last_commit) / 60.0
            if checkpoint_watchdog_exceeded(
                elapsed_minutes,
                t.checkpoint_expected_minutes,
                t.checkpoint_stall_multiplier,
            ):
                raise RuntimeError(
                    f"Training liveness watchdog tripped at step {global_step}: no committed "
                    f"checkpoint for {elapsed_minutes:.1f} min > "
                    f"{t.checkpoint_expected_minutes} * {t.checkpoint_stall_multiplier} min "
                    "(checkpoint_expected_minutes * checkpoint_stall_multiplier). Treating the "
                    "round as hung/preempted — the train() retry resumes in-dir from the latest "
                    "committed checkpoint."
                )

            if global_step % t.checkpoint_every == 0:
                # ⛔ THE VRAM GAUGE. The H3 campaign card carries a tripwire — `expected_peak_gib:
                # 76.36  # a peak materially above this means the sequence budget drifted` — and
                # NOTHING printed the live peak, so the first sign of drift would have been an OOM
                # at hour N of a 13.8 h run with 2.89 GiB of headroom. A tripwire needs a gauge.
                # Per checkpoint (not per step) so it costs nothing, and `max_memory_allocated` is
                # the RUNNING maximum, so a spike between checkpoints is still reported.
                logger.info(
                    "step %d — VRAM peak %.2f GiB allocated / %.2f GiB reserved (compare against "
                    "the campaign card's expected_peak_gib; a materially higher peak means the "
                    "packed-sequence budget drifted)",
                    global_step,
                    _peak_gib(torch.cuda.max_memory_allocated),
                    _peak_gib(torch.cuda.max_memory_reserved),
                )
                ckpt_manager.save(model, optimizer, scheduler, global_step, float(last_loss))
                # #12 (AUDIT) — commit BEFORE the render: the just-saved checkpoint is made durable
                # on the Volume IMMEDIATELY, before the ~8-13 min in-loop render, so a preemption
                # DURING the render can't vanish it (commit-or-vanish: "done" == the file on Volume).
                if checkpoints_vol is not None:
                    checkpoints_vol.commit()  # first commit: land the checkpoint before rendering
                    last_commit = time.monotonic()  # #11: refresh the liveness clock
                if on_checkpoint is not None:
                    # In-loop sample (OFFL-02) renders here — mid-run, REAL step number. The
                    # checkpoint above is ALREADY durable; the mp4 this writes rides the SECOND
                    # commit below.
                    try:
                        on_checkpoint(global_step)
                    except Exception:
                        logger.exception(
                            "on_checkpoint callback failed at step %d — training continues "
                            "(checkpoint already saved)", global_step,
                        )
                if checkpoints_vol is not None:
                    checkpoints_vol.commit()  # second commit: land the rendered mp4 artifact
                    last_commit = time.monotonic()  # #11: refresh the liveness clock

            if global_step >= total_steps:
                break

    # Final checkpoint at max_steps (the "done" marker = checkpoint-step-{max_steps} on the Volume).
    ckpt_manager.save(model, optimizer, scheduler, global_step, float(last_loss))
    if checkpoints_vol is not None:
        checkpoints_vol.commit()
        last_commit = time.monotonic()  # #11: refresh the liveness clock (kept consistent)

    return global_step
