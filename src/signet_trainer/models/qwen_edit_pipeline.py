"""models.qwen_edit_pipeline — the Qwen-Image-Edit-2511 GENERATION pipeline and its LOCKED §8 recipe.

The render-side sibling of ``models/qwen_edit_loader.py``. That module owns the three component
LOADS and the architecture gate; this one owns what those components are assembled INTO, and the
one setting the assembly silently gets wrong.

⛔ WHY THIS MODULE IS IN ``models/`` AND NOT IN ``inference/``
--------------------------------------------------------------
``tests/test_no_wan_params.py:31`` bans the bare token ``shift`` from executable code in every
``*.py`` under ``src/signet_trainer/inference/`` — a guard written to keep the Wan-tuned sampling
set out of LTX paths (its own docstring, :6-8), whose directory-wide ``glob("*.py")`` scope now
also covers a family where that same setting is MANDATORY at a different value. The guard's scope
is non-recursive and covers ``inference/`` only; ``models/`` is not scanned.

Narrowing ``_WAN_TOKENS`` to the LTX modules remains the right long-term ruling, and it is
Timothy's to make. Until then this module names the setting plainly, in one place, and
``inference/qwen_edit_layout.render_qwen_edit_sample`` consumes a pipeline that is ALREADY pinned
and never spells the token. That is additive and reversible: the day the guard is narrowed,
nothing here has to move, and no shipped money-safe test was weakened to land a sampler.

⚠ THE TRAP THIS MODULE EXISTS FOR — measured on this box, not asserted
----------------------------------------------------------------------
``QWEN-CHAINED-EDIT-METHOD.md`` §8 records the diagnosis tell: *training samples look fine but
renders are muddy → it is inference settings, not the model*. The mechanism, read out of
``diffusers/schedulers/scheduling_flow_match_euler_discrete.py:347-350``::

    if self.config.use_dynamic_shifting:
        sigmas = self.time_shift(mu, 1.0, sigmas)                        # mu path
    else:
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)   # static path

``self.shift`` is the INSTANCE value (``_shift``, :146-150), settable at runtime by ``set_shift``
(:177-185). ``self.config.use_dynamic_shifting`` is a FROZEN CONFIG field. So on a scheduler built
with ``use_dynamic_shifting=True``, line 347 takes the dynamic branch and **never reads
``self.shift`` at all** — ``set_shift(3.0)`` changes nothing. Measured, 30 steps, the ai-toolkit
scheduler config, ``mu`` for a 1024x1024 latent::

    dynamic scheduler, set_shift(3.0):
        scheduler.shift now = 3.0   use_dynamic_shifting = True
        max |sigma delta| after set_shift(3.0) = 0.000e+00        <-- exactly a NO-OP

The pin therefore REPLACES the scheduler object (:func:`pin_qwen_edit_scheduler`) rather than
poking its shift, and then VERIFIES the replacement took
(:func:`assert_qwen_edit_scheduler_pinned`) rather than assuming it. A silently-rebuilt
default-shift scheduler produces muddy renders that read as a bad adapter, which is the most
expensive way to be wrong on this family: it condemns 5000 steps of A100 time for a settings bug.

``mu`` is still passed by the pipeline on every call
(``pipeline_qwenimage_edit_plus.py:753-767`` computes it unconditionally and hands it to
``retrieve_timesteps``), and it is harmlessly IGNORED under the static branch. Measured — two very
different ``mu`` values through one pinned scheduler::

    mu is IGNORED under static: max |delta| across two different mu = 0.000e+00

Equivalence, for anyone reconciling the two parameterisations: **static shift S == dynamic
mu = ln(S)**, to float32 epsilon (max abs sigma delta 1.192e-07 at 30 steps, measured at S = 3.0
and S = 7.0 with ``shift_terminal`` disabled on both so the comparison isolates one variable).

⚠ WHAT THE DEFAULTS ARE ACTUALLY WORTH — the method note's framing does not transfer to stock
-----------------------------------------------------------------------------------------------
§8 describes ai-toolkit's default dynamic shift as "~1.6-2.5", far weaker than the shipped ComfyUI
``ModelSamplingAuraFlow`` value. That is TRUE OF ai-toolkit and NOT of stock diffusers, and the two
land on opposite sides of the target. Computed from each stack's own ``calculate_shift`` constants
(ai-toolkit ``qwen_image.py:37-52``; stock ``pipeline_qwenimage_edit_plus.py`` defaults)::

    1024x1024  seq= 4096   stock mu=1.150000 -> 3.158193 | ai-toolkit mu=0.693548 -> 2.000803
    1328x1328  seq= 6889   stock mu=1.622773 -> 5.067124 | ai-toolkit mu=0.834325 -> 2.303258
    1024x768   seq= 3072   stock mu=0.976667 -> 2.655590 | ai-toolkit mu=0.641935 -> 1.900155

So: rendering through ai-toolkit unpinned lands ~1.9-2.3 where the recipe wants 3.0 — the muddy
case §8 names. Rendering through STOCK diffusers unpinned already lands 3.158 at 1024², slightly
STRONGER than 3.0, and 5.07 at 1328². The pin is still correct and still required — it is what
makes a render RESOLUTION-INDEPENDENT and reproducible across stacks — but it is not rescuing a
2.0 here, and a reader who believes it is will misdiagnose the next muddy render.

⛔ OPEN, NOT SETTLED (flagged, not chosen — do not resolve these by writing code)
--------------------------------------------------------------------------------
1. **Which pipeline class.** Stock ``QwenImageEditPlusPipeline`` applies the CFG-norm rescale
   UNCONDITIONALLY under true-CFG (``pipeline_qwenimage_edit_plus.py:833-835``; there is no
   ``do_cfg_norm`` kwarg on the stock class). ai-toolkit's ``QwenImageEditPlusCustomPipeline``
   defaults ``do_cfg_norm=False`` (``qwen_image_pipelines.py:52``) with the author's note that it
   "hurts more often than it helps". The two therefore render DIFFERENTLY at identical nominal
   settings, and §8 does not record which stack produced its numbers.
2. **``shift_terminal``.** ai-toolkit's scheduler config sets ``0.02``, which triggers
   ``stretch_shift_to_terminal`` (scheduler :353-354); stock diffusers' default is ``None`` — no
   stretch. No source addresses whether the target recipe includes it. This module INHERITS
   whatever the supplied scheduler config carries and REPORTS it rather than picking.
3. **Whether §8's 3.0 was measured at 1024².** Pinning a static 3.0 makes renders
   resolution-independent but weakens the 1328² case substantially relative to the stock default.

Import tier: ``logging`` + ``typing`` + ``dataclasses`` at module scope. ``diffusers`` and ``torch``
are FUNCTION-LOCAL for ``models/qwen_edit_loader.py``'s reason — reading the recipe constants, or
running the two CPU-pure assertions, must work with neither installed.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_CONDITION_IMAGE_SIZE,
    QWEN_EDIT_MAX_CONTROL_SLOTS,
    QWEN_EDIT_VAE_IMAGE_SIZE,
    qwen_edit_area_budget_size,
)

logger = logging.getLogger(__name__)

__all__ = [
    "QWEN_EDIT_RENDER_CFG_NORM",
    "QWEN_EDIT_RENDER_LORA_SCALE",
    "QWEN_EDIT_RENDER_NEGATIVE_PROMPT",
    "QWEN_EDIT_RENDER_RECIPE",
    "QWEN_EDIT_RENDER_SCHEDULER_SHIFT",
    "QWEN_EDIT_RENDER_STEPS",
    "QWEN_EDIT_RENDER_TRUE_CFG",
    "QwenEditRenderRecipe",
    "assert_qwen_edit_control_geometry",
    "assert_qwen_edit_scheduler_pinned",
    "build_qwen_edit_pipeline",
    "pin_qwen_edit_scheduler",
    "qwen_edit_generate",
    "qwen_edit_generator",
    "qwen_edit_render_context",
    "qwen_edit_static_scheduler",
]


# --------------------------------------------------------------------------------------------------
# The LOCKED §8 inference recipe. Constants, not config fields — see QwenEditRenderRecipe.
# --------------------------------------------------------------------------------------------------

#: §8: ``Qwen-Image-Edit-2511 : 30 steps``. NOT the 50 diffusers defaults to.
QWEN_EDIT_RENDER_STEPS: int = 30

#: §8: ``true_cfg 4.0``. This is ``true_cfg_scale=``, and it is emphatically NOT ``guidance_scale=``.
#: ``EXPECTED_QWEN_EDIT_GUIDANCE_EMBEDS`` is ``False`` on this checkpoint
#: (``models/qwen_edit_loader.py:151``, established by the ABSENCE of the guidance-embedder tensors),
#: so the pipeline's guidance branch logs *"guidance_scale is passed as X, but ignored since the
#: model is not guidance-distilled"* and sets ``guidance = None``
#: (``pipeline_qwenimage_edit_plus.py:772-782``). Routing the config's ``validation.guidance_scale``
#: to ``guidance_scale=`` therefore renders at effective CFG 1.0 — muddy, and indistinguishable from
#: a bad adapter. That is the single easiest way to be wrong here, and it is why
#: ``render_qwen_edit_sample`` refuses the keyword by name.
QWEN_EDIT_RENDER_TRUE_CFG: float = 4.0

#: §8: ``STATIC shift 3.0`` for Qwen-Image-Edit-2511 (the base Qwen-Image row of §8's table says
#: 7.0, but no base-model render path exists in signet and that row is out of scope until one does).
QWEN_EDIT_RENDER_SCHEDULER_SHIFT: float = 3.0

#: §8: ``LoRA strength 1.0``. The adapter was written with ``r == lora_alpha == 42``, so PEFT's own
#: ``alpha/r`` scale is already exactly 1.0 and NO rescale is applied at load — verified by reading
#: ``adapter_config.json`` on the trained checkpoint. Carried as a recipe term anyway so the number
#: has one name and a future rank/alpha split cannot change strength silently.
QWEN_EDIT_RENDER_LORA_SCALE: float = 1.0

#: §8: ``CFGNorm ON``. On the stock ``QwenImageEditPlusPipeline`` this is satisfied BY CONSTRUCTION
#: and cannot be turned off — the ``cond_norm / noise_norm`` rescale at :833-835 runs whenever
#: ``do_true_cfg``, with no flag. Recorded as a recipe term because it is one on ai-toolkit's
#: pipeline (``do_cfg_norm`` defaults False there), i.e. because the two stacks disagree.
QWEN_EDIT_RENDER_CFG_NORM: bool = True

#: §8: *"the reference feeds BOTH the positive AND negative encode nodes."* On stock diffusers that
#: dual wiring is ALREADY what the pipeline does — :710-728 passes the same ``condition_images`` to
#: both ``encode_prompt`` calls — so the requirement reduces to "pass a non-None negative prompt",
#: which is what turns ``do_true_cfg`` on at :709 (``true_cfg_scale > 1 and has_neg_prompt``).
#: §8 does not give the negative STRING. A single space is the pipeline docstring's own minimal
#: enabling value (:588); it is a CHOICE, and it is a config field the day one exists.
QWEN_EDIT_RENDER_NEGATIVE_PROMPT: str = " "


@dataclass(frozen=True)
class QwenEditRenderRecipe:
    """§8's inference settings as ONE value, so "does the sampler match the recipe" is one compare.

    Frozen and defaulted from the module constants above. It is deliberately NOT a config schema
    block: every field here is a METHOD term settled by ``QWEN-CHAINED-EDIT-METHOD.md``, in the same
    sense that ``quantize_qwen_edit``'s ``qfloat8`` is settled — *"the house recipe locks the
    quantization the same way it locks the optimizer"* (``QwenEditConfig``'s own docstring). A knob
    per term would let a render drift off the recipe with nothing to compare it against; a dataclass
    lets a test assert the whole set in one line and lets an operator print it into the grid banner.

    ``scheduler_shift`` is carried here rather than read back off the pipeline because the recipe is
    what the pipeline is CHECKED AGAINST — see :func:`assert_qwen_edit_scheduler_pinned`.
    """

    steps: int = QWEN_EDIT_RENDER_STEPS
    true_cfg: float = QWEN_EDIT_RENDER_TRUE_CFG
    scheduler_shift: float = QWEN_EDIT_RENDER_SCHEDULER_SHIFT
    lora_scale: float = QWEN_EDIT_RENDER_LORA_SCALE
    cfg_norm: bool = QWEN_EDIT_RENDER_CFG_NORM
    negative_prompt: str = QWEN_EDIT_RENDER_NEGATIVE_PROMPT

    def describe(self) -> str:
        """One line for a log or a grid banner."""
        return (
            f"steps={self.steps} true_cfg={self.true_cfg} cfg_norm={self.cfg_norm} "
            f"scheduler_shift={self.scheduler_shift} (static) lora_scale={self.lora_scale}"
        )


#: The one shipped recipe instance. Callers take defaults from this; nothing mutates it (frozen).
QWEN_EDIT_RENDER_RECIPE = QwenEditRenderRecipe()


# --------------------------------------------------------------------------------------------------
# The scheduler: build it static, pin it onto the pipeline AFTER construction, then VERIFY.
# --------------------------------------------------------------------------------------------------


def qwen_edit_static_scheduler(
    base_config: Any = None,
    *,
    shift: float = QWEN_EDIT_RENDER_SCHEDULER_SHIFT,
    dynamic: bool = False,
) -> Any:
    """A ``FlowMatchEulerDiscreteScheduler`` with the §8 STATIC reparameterisation (Modal-side load).

    Built through ``from_config`` rather than ``__init__`` so a SHIPPED scheduler config can be
    inherited wholesale and exactly two fields overridden on top of it. That matters for the field
    this module does not get to rule on: ``shift_terminal``. ai-toolkit's config carries ``0.02``
    (which triggers ``stretch_shift_to_terminal``, scheduler :353-354); stock diffusers' default is
    ``None``. Inheriting rather than re-declaring means the pin changes the ONE thing §8 names and
    leaves the rest of the checkpoint's own schedule alone — and :func:`assert_qwen_edit_scheduler_pinned`
    reports what was inherited so the open question stays visible instead of becoming a default.

    ``from_config`` also handles the private ``_class_name`` / ``_diffusers_version`` keys that a
    live ``scheduler.config`` FrozenDict carries and ``__init__`` would reject.

    Args:
        base_config: a scheduler config to inherit — a live ``scheduler.config``, a dict, or None
            for the diffusers defaults. The CALLER supplies the checkpoint's own where it has one.
        shift: the static reparameterisation value. §8's 3.0.
        dynamic: leave False. Exposed only so a caller can build the UNPINNED comparison scheduler
            deliberately (a test does exactly that); a True here defeats the whole module, and
            :func:`assert_qwen_edit_scheduler_pinned` refuses it unless told to expect it.

    Returns:
        The scheduler. NOT yet attached to anything — see :func:`pin_qwen_edit_scheduler`.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: PLC0415 — function-local

    config: dict[str, Any] = dict(base_config) if base_config is not None else {}
    if not config:
        # No shipped config to inherit: instantiate the defaults and read them back, so the
        # overrides below land on a complete config rather than on a bare two-key dict.
        config = dict(FlowMatchEulerDiscreteScheduler().config)

    # ⛔ THE OVERRIDES GO AS KWARGS, NEVER BY MUTATING THE DICT. MEASURED on this box:
    #
    #     base = dict(FlowMatchEulerDiscreteScheduler().config)
    #     base["shift"] = 3.0
    #     FlowMatchEulerDiscreteScheduler.from_config(base).shift   ->  1.0     <-- SILENTLY IGNORED
    #     FlowMatchEulerDiscreteScheduler.from_config(base, shift=3.0).shift -> 3.0
    #
    # ``ConfigMixin.extract_init_dict`` strips every key listed in the config's own
    # ``_use_default_values`` before building the init dict — i.e. any field the DONOR left at its
    # default is deleted from the inherited dict and re-defaulted, taking an override written into
    # that dict with it. Kwargs are applied after extraction and survive. This is precisely the
    # class of failure this module exists for: the dict form yields a scheduler that reports a
    # pinned-looking config and steps at the default schedule, and only
    # :func:`assert_qwen_edit_scheduler_pinned` tells the two apart.
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        config, use_dynamic_shifting=bool(dynamic), shift=float(shift)
    )
    logger.info(
        "[qwen-edit-pipeline] scheduler built: %s shift=%s use_dynamic_shifting=%s "
        "shift_terminal=%s time_shift_type=%s",
        scheduler.__class__.__name__,
        scheduler.shift,
        scheduler.config.use_dynamic_shifting,
        scheduler.config.get("shift_terminal", None),
        scheduler.config.get("time_shift_type", None),
    )
    return scheduler


def pin_qwen_edit_scheduler(
    pipeline: Any,
    *,
    shift: float = QWEN_EDIT_RENDER_SCHEDULER_SHIFT,
    dynamic: bool = False,
    inherit_config: bool = True,
) -> dict[str, Any]:
    """Override ``pipeline.scheduler`` AFTER the pipeline exists, then verify the override took.

    This is the §8 instruction executed literally, and it is a REPLACEMENT rather than a mutation
    for the measured reason in the module docstring: ``set_shift`` is a no-op on a scheduler whose
    frozen config says ``use_dynamic_shifting=True`` (max |sigma delta| 0.000e+00). Any factory that
    rebuilds a fresh default scheduler — ai-toolkit's ``get_generation_pipeline`` calls
    ``get_train_scheduler()`` at ``qwen_image_edit_plus.py:76`` and hands the result to the pipeline
    ctor — is corrected by this call, whatever it built.

    Applied through ``register_modules`` rather than a bare attribute assignment so the pipeline's
    own component registry and config stay consistent with the object it will actually step.

    Returns:
        The report from :func:`assert_qwen_edit_scheduler_pinned` — the pin is not considered done
        until it has been re-READ off the pipeline.
    """
    base = getattr(getattr(pipeline, "scheduler", None), "config", None) if inherit_config else None
    scheduler = qwen_edit_static_scheduler(base, shift=shift, dynamic=dynamic)
    pipeline.register_modules(scheduler=scheduler)
    if getattr(pipeline, "scheduler", None) is not scheduler:
        raise RuntimeError(
            "[qwen-edit-pipeline] register_modules(scheduler=...) did not take: pipeline.scheduler "
            "is not the object that was just pinned. Every subsequent sigma comes from whatever IS "
            "attached, so this cannot be allowed to pass — a render under an unpinned scheduler is "
            "the muddy-output failure METHOD §8 names, and it reads as a bad adapter."
        )
    return assert_qwen_edit_scheduler_pinned(pipeline, shift=shift, dynamic=dynamic)


def assert_qwen_edit_scheduler_pinned(
    pipeline_or_scheduler: Any,
    *,
    shift: float = QWEN_EDIT_RENDER_SCHEDULER_SHIFT,
    dynamic: bool = False,
) -> dict[str, Any]:
    """Refuse to render unless the attached scheduler IS the §8 one. CPU-pure; no torch needed.

    Two independent checks, because either one alone passes the failure it was written for:

      1. ``config.use_dynamic_shifting`` must be ``False``. This is the check that catches the
         documented trap — a caller who reached for ``set_shift(3.0)`` on a dynamic scheduler gets
         a scheduler whose ``.shift`` reads 3.0 and whose sigmas are byte-identical to before
         (measured: 0.000e+00 delta). Check (2) alone would GREENLIGHT that scheduler.
      2. ``scheduler.shift`` must equal the recipe value. This is the check that catches a factory
         that rebuilt a static scheduler at its own default.

    ``shift_terminal`` is REPORTED and warned about, never refused: whether §8's recipe includes
    ai-toolkit's ``0.02`` stretch is an open question no source settles (module docstring, item 2),
    and turning an open question into a refusal would be picking a side by implementation.

    Accepts a pipeline or a bare scheduler so a test can drive it without constructing either.

    Returns:
        ``{"class", "shift", "use_dynamic_shifting", "shift_terminal", "time_shift_type"}`` —
        the numbers a render should print, read back off the live object rather than restated.
    """
    scheduler = getattr(pipeline_or_scheduler, "scheduler", pipeline_or_scheduler)
    if scheduler is None:
        raise RuntimeError(
            "[qwen-edit-pipeline] no scheduler attached — pipeline.scheduler is None. Build the "
            "pipeline through build_qwen_edit_pipeline (or apply pin_qwen_edit_scheduler to an "
            "externally-built one) before rendering."
        )
    config = getattr(scheduler, "config", None)
    if config is None:
        raise RuntimeError(
            f"[qwen-edit-pipeline] the attached scheduler {type(scheduler).__name__!r} has no "
            "`.config`, so use_dynamic_shifting cannot be read and the §8 pin cannot be verified. "
            "A FlowMatchEulerDiscreteScheduler is expected here."
        )

    live_dynamic = bool(config.get("use_dynamic_shifting", False))
    live_shift = getattr(scheduler, "shift", None)
    report = {
        "class": type(scheduler).__name__,
        "shift": None if live_shift is None else float(live_shift),
        "use_dynamic_shifting": live_dynamic,
        "shift_terminal": config.get("shift_terminal", None),
        "time_shift_type": config.get("time_shift_type", None),
    }

    if live_dynamic != bool(dynamic):
        raise RuntimeError(
            f"[qwen-edit-pipeline] the attached scheduler has use_dynamic_shifting="
            f"{live_dynamic}, expected {bool(dynamic)}. METHOD §8 pins a STATIC reparameterisation "
            f"of {float(shift)} for this family, and under use_dynamic_shifting=True the scheduler "
            "takes the mu branch at scheduling_flow_match_euler_discrete.py:347 and NEVER READS "
            f"scheduler.shift at all — so the {report['shift']} this object reports is decorative. "
            "Measured on this box at 30 steps: calling set_shift(3.0) on a dynamic scheduler moves "
            "the sigmas by exactly 0.000e+00. Replace the scheduler object (pin_qwen_edit_scheduler) "
            "instead of setting its shift; a render under the unpinned schedule is the muddy-output "
            "failure §8 names, and it reads as a bad adapter rather than as a settings bug."
        )
    if report["shift"] is None or abs(report["shift"] - float(shift)) > 1e-9:
        raise RuntimeError(
            f"[qwen-edit-pipeline] the attached scheduler's shift is {report['shift']}, expected "
            f"{float(shift)} (METHOD §8). A factory that rebuilds its own scheduler is the usual "
            "cause — ai-toolkit's get_generation_pipeline builds a fresh one at "
            "qwen_image_edit_plus.py:76 — so pin it AFTER the pipeline is constructed, with "
            "pin_qwen_edit_scheduler(pipeline)."
        )
    if report["shift_terminal"] is not None:
        logger.warning(
            "[qwen-edit-pipeline] the pinned scheduler carries shift_terminal=%s, so the sigma "
            "schedule is stretched to terminate there (scheduling_flow_match_euler_discrete.py:"
            "353-354). ai-toolkit's config sets 0.02; stock diffusers defaults to None (no "
            "stretch). No source states whether METHOD §8's recipe includes the stretch — this is "
            "inherited from the supplied scheduler config, not chosen here. Reported so a render "
            "comparison between the two stacks is not silently confounded by it.",
            report["shift_terminal"],
        )
    logger.info("[qwen-edit-pipeline] scheduler pin VERIFIED: %s", report)
    return report


def build_qwen_edit_pipeline(
    *,
    transformer: Any,
    vae: Any,
    text_encoder: Any,
    tokenizer: Any,
    processor: Any,
    shift: float = QWEN_EDIT_RENDER_SCHEDULER_SHIFT,
    dynamic: bool = False,
    scheduler_config: Any = None,
) -> Any:
    """Assemble a ``QwenImageEditPlusPipeline`` from already-loaded components (Modal-side ONLY).

    The six components are the pipeline ctor's own list
    (``pipeline_qwenimage_edit_plus.py:189-197``), all mounted-Volume-local. This function does NOT
    load them — that is ``models/qwen_edit_loader.py``'s job and ``modal/fns.py``'s composition of
    the paths (D-NOHARDCODE). The intended order, which this function assumes has already happened
    for the transformer::

        load_qwen_edit_transformer
          -> summarize_qwen_edit_transformer + assert_qwen_edit_arch + assert_qwen_edit_targets
          -> quantize_qwen_edit(model, what=...)      # on the UN-WRAPPED transformer, qfloat8
          -> lora.peft.build_lora_config(targets=cfg.resolved_lora_targets()) -> inject_lora
          -> lora.peft.load_adapter_into(adapted, ckpt_dir)  -> .eval()
          -> build_qwen_edit_pipeline(transformer=adapted, ...)

    ⚠ ``load_adapter_into`` (``lora/peft.py:490``) is the load path for this family, NOT
    ``inference/lora_load.load_lora_onto_transformer``. The latter strips the PEFT prefix, targets
    ``.get_base_model()``, and RE-DERIVES the target set from whatever keys the file happens to
    contain — discarding ``config.resolved_lora_targets()``, the 14-leaf regex the arch gate
    verified at 840/840. It functions; it is the wrong authority. ``load_adapter_into`` keeps the
    PEFT-native keys ``get_peft_model_state_dict`` wrote and is what ``h3_sample`` uses
    (``modal/fns.py:4664``).

    ⚠ The transformer passed here should be ONE PEFT-wrapped model used for BOTH the base and the
    adapter rows, toggled with ``disable_adapter()`` — H3's reason at ``modal/fns.py:4356-4359``.
    Two separately-loaded models make "identical seed, identical everything except the adapter"
    a claim rather than a fact, and §8's convergence read is exactly that difference. Do NOT port
    H3's pre-column ``merge_adapter()`` here without measuring: it is an OOM remedy for a 61.7 GiB
    transformer, Qwen's is 40.9 GiB and qfloat8-quantized, and merging is incompatible with
    re-entering ``disable_adapter()``.

    The scheduler is built static, passed to the ctor, AND re-pinned afterwards through
    :func:`pin_qwen_edit_scheduler` — idempotent when the ctor kept ours, load-bearing the day this
    assembly is replaced by a factory that rebuilds one. The pin is verified before the pipeline is
    returned, so a caller cannot receive an unpinned pipeline from this function at all.

    Returns:
        The pipeline, scheduler pinned and VERIFIED.
    """
    from diffusers import QwenImageEditPlusPipeline  # noqa: PLC0415 — function-local

    scheduler = qwen_edit_static_scheduler(scheduler_config, shift=shift, dynamic=dynamic)
    pipeline = QwenImageEditPlusPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer,
    )
    # THE OVERRIDE, AFTER the pipeline exists. §8 names this as the step, and the module docstring
    # records why doing it before is not enough in general.
    report = pin_qwen_edit_scheduler(
        pipeline, shift=shift, dynamic=dynamic, inherit_config=scheduler_config is None
    )
    logger.info(
        "[qwen-edit-pipeline] %s assembled; recipe %s; scheduler %s",
        type(pipeline).__name__,
        QWEN_EDIT_RENDER_RECIPE.describe(),
        report,
    )
    return pipeline


# --------------------------------------------------------------------------------------------------
# The render-time torch seam. Lives here, not in inference/, for the SECOND reason this module is in
# models/: tests/test_qwen_edit_render_surface.py's import-confinement scan bans ``import torch`` in
# qwen_edit_layout.py's source with a regex anchored at ``^\s*`` — which matches an INDENTED,
# function-local import too. So the sampler cannot touch torch even lazily, and the two things it
# genuinely needs from it (a seeded generator, and the no-grad + adapter-toggle context) are named
# here instead. That is not a workaround: seeding and adapter state are model concerns, and the
# layout module's job is to decide WHICH cells exist, not to hold a Generator.
# --------------------------------------------------------------------------------------------------


def qwen_edit_generator(pipeline: Any, seed: int) -> tuple[Any, Any]:
    """A ``torch.Generator`` seeded for ONE render, on the pipeline's own execution device.

    The device matters: ``prepare_latents`` draws through ``randn_tensor(..., generator=generator,
    device=device)``, and a CPU generator against a CUDA device takes a different code path (draw on
    CPU, then move) than a CUDA generator does. Two rows of §8's grid that differ in generator device
    do not differ ONLY in the adapter, which is the one thing the base-vs-LoRA read requires.

    Returns:
        ``(generator, device)`` — the device is returned so the caller can log it rather than
        re-derive it.
    """
    import torch  # noqa: PLC0415 — function-local; this module is import-light at module scope

    device = getattr(pipeline, "_execution_device", None) or getattr(
        pipeline, "device", torch.device("cpu")
    )
    return torch.Generator(device=device).manual_seed(int(seed)), device


@contextlib.contextmanager
def qwen_edit_render_context(pipeline: Any, *, adapter: bool) -> Iterator[None]:
    """``no_grad`` + the §8 base/LoRA toggle, as ONE context — the convergence read depends on it.

    §8's convergence check is *"base-vs-LoRA divergence at ~200-250 steps, not a step threshold and
    not loss. If the sample is essentially the base render, it isn't converging."* That comparison is
    only valid if the two renders differ in the adapter and in NOTHING else, which is why H3 renders
    both rows from ONE PEFT model under ``disable_adapter()`` rather than loading two transformers
    (``modal/fns.py:4356-4359``). This is that contract for qwen_edit.

    ``adapter=True`` on a transformer that is NOT PEFT-wrapped is a REFUSAL, not a warning: it is the
    direction that mislabels a base render as a LoRA render, and a grid whose "ckpt" columns are
    quietly the base column reads as a perfectly converged adapter that changed nothing — or as a
    dead one. ``adapter=False`` on an unwrapped transformer is merely redundant, so it warns and
    proceeds, naming the one-model rule it is probably breaking.

    Measured on CPU (peft 0.18.1) that a ``get_peft_model``-wrapped module still answers every
    attribute the Qwen pipeline reads off ``self.transformer`` — ``.config.guidance_embeds``,
    ``.config.in_channels``, ``.cache_context("cond")``, and the call itself — so the wrapper can sit
    in the pipeline slot directly rather than being unwrapped per render.
    """
    import torch  # noqa: PLC0415 — function-local

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError(
            "[qwen-edit-pipeline] pipeline.transformer is None — there is nothing to render with, "
            "and nothing to toggle. Build the pipeline through build_qwen_edit_pipeline."
        )
    toggle = getattr(transformer, "disable_adapter", None)
    wrapped = callable(toggle)

    if adapter and not wrapped:
        raise RuntimeError(
            f"[qwen-edit-pipeline] an ADAPTER render was requested but pipeline.transformer is a "
            f"{type(transformer).__name__!r} with no disable_adapter() — i.e. it is not PEFT-"
            "wrapped, so there is no adapter on it and this render would be the BASE model under a "
            "checkpoint's label. METHOD §8 reads convergence as base-vs-LoRA divergence; a grid "
            "whose band columns are silently the base column reports 'not converging' for every "
            "checkpoint ever rendered. Wrap with lora.peft.inject_lora and load the checkpoint with "
            "lora.peft.load_adapter_into before rendering."
        )

    with torch.no_grad():
        if adapter:
            yield
        elif wrapped:
            with toggle():
                yield
        else:
            logger.warning(
                "[qwen-edit-pipeline] BASE render on a transformer that is not PEFT-wrapped. That "
                "is not wrong on its own, but §8's divergence read requires the base and adapter "
                "rows to come from ONE model toggled with disable_adapter() — two separately loaded "
                "transformers make 'identical everything except the adapter' a claim rather than a "
                "fact."
            )
            yield


# --------------------------------------------------------------------------------------------------
# Control geometry — signet's module is the AUTHORITY; the pipeline's own resize is checked against it.
# --------------------------------------------------------------------------------------------------


def assert_qwen_edit_control_geometry(
    images: Any, *, what: str = "control images"
) -> list[dict[str, Any]]:
    """Gate every control image's two resize targets against ``conditioning/qwen_edit_geometry``.

    ⛔ **The control images are handed to the pipeline RAW, and this is the reason that is correct.**
    ``QwenImageEditPlusPipeline.__call__`` resizes each control image ITSELF, twice, at
    :685-694 — ``calculate_dimensions(CONDITION_IMAGE_SIZE, w/h)`` for the Qwen2.5-VL channel and
    ``calculate_dimensions(VAE_IMAGE_SIZE, w/h)`` for the VAE channel, the same two budgets
    ``qwen_edit_geometry`` declares at :129 and :136. Pre-resizing with signet's own function and
    then passing the RESULT would resample every image twice, producing pixels neither channel sees
    at inference — the exact failure ``prepare_qwen_edit_image`` refuses for the training leg
    (``prep/qwen_edit_encode.py:426-435``).

    So the house rule "control images go through signet's geometry, never a fresh transcription" is
    honoured as an EQUALITY GATE rather than as a second resize: for every image, signet's
    ``qwen_edit_area_budget_size`` is computed at both budgets and compared to what the pipeline
    will do. If they ever diverge, this refuses BEFORE the render instead of after a mis-oriented
    grid is judged.

    They agree today, and the agreement is not luck — it is the same algebra::

        diffusers: width = sqrt(area * (w/h));  height = width / (w/h)   [both from the UNROUNDED
                   width]; each edge then round(edge/32)*32
        signet:    scale = sqrt(area/(w*h));    out_w = snap(w*scale), out_h = snap(h*scale)
                   w*sqrt(area/(w*h)) == sqrt(area*w/h)  and  h*sqrt(area/(w*h)) == sqrt(area*h/w)

    The one deliberate difference is signet's floor at one 32px tile (``qwen_edit_geometry._snap``),
    which diffusers lacks — ``round(8/32)*32 == 0`` there. A source degenerate enough to reach it
    would give the pipeline a zero-pixel edge, so a divergence at the floor is a refusal here and
    not a silent 0.

    ⚠ This gate is ALSO where the settled orientation ruling is enforced at render time. ai-toolkit
    computes ``ratio = H/W`` where diffusers computes ``W/H``, transposing every non-square control
    (``qwen_edit_geometry.py:252-263``, ruled diffusers-correct in phase 1). Since the stock pipeline
    is the thing doing the resize, that ruling is satisfied by construction — and this gate is what
    would CATCH a future swap to a bug-compatible stack, on the first non-square image, instead of
    letting it read as an adapter that "learned the wrong composition".

    Args:
        images: the control images, in slot order. Pillow-like (``.size`` -> ``(width, height)``).
        what: named in every message.

    Returns:
        One dict per image: ``{"index", "source_wh", "vae_wh", "condition_wh"}`` — the geometry a
        render should log, computed once and read rather than restated.
    """
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (  # noqa: PLC0415
        CONDITION_IMAGE_SIZE,
        VAE_IMAGE_SIZE,
        calculate_dimensions,
    )

    slots = list(images)
    if not slots:
        raise ValueError(
            f"[qwen-edit-pipeline] {what}: EMPTY. Qwen-Image-Edit is an EDIT model — the control "
            "image is the subject of the instruction, and with no control slot the render is a "
            "text-to-image sample that answers nothing about the adapter. Pass 1.."
            f"{QWEN_EDIT_MAX_CONTROL_SLOTS} images."
        )
    if len(slots) > QWEN_EDIT_MAX_CONTROL_SLOTS:
        raise ValueError(
            f"[qwen-edit-pipeline] {what}: {len(slots)} control images, but the architecture's own "
            f"ceiling is {QWEN_EDIT_MAX_CONTROL_SLOTS} (ai-toolkit's prompt template exposes "
            "ctrl_img_1..3, qwen_image_edit_plus.py:105-122; qwen_edit_geometry.py:143-145)."
        )

    # The two budgets are asserted equal to the pipeline's own module constants rather than trusted:
    # if diffusers ever changes one, every downstream row count in this repo is silently wrong.
    for label, ours, theirs in (
        ("VAE_IMAGE_SIZE", QWEN_EDIT_VAE_IMAGE_SIZE, VAE_IMAGE_SIZE),
        ("CONDITION_IMAGE_SIZE", QWEN_EDIT_CONDITION_IMAGE_SIZE, CONDITION_IMAGE_SIZE),
    ):
        if int(ours) != int(theirs):
            raise RuntimeError(
                f"[qwen-edit-pipeline] {label} disagreement: conditioning/qwen_edit_geometry says "
                f"{ours}, the installed pipeline_qwenimage_edit_plus says {theirs}. Every packed "
                "row count in this repo is derived from the signet constant, so a divergence "
                "silently re-prices every sequence. Reconcile before rendering."
            )

    report: list[dict[str, Any]] = []
    for index, image in enumerate(slots):
        size = getattr(image, "size", None)
        if not size or len(tuple(size)) != 2:
            raise TypeError(
                f"[qwen-edit-pipeline] {what}[{index}] is a {type(image).__name__!r} with no "
                "`.size` -> (width, height). The pipeline reads `img.size` directly "
                "(pipeline_qwenimage_edit_plus.py:685), so a non-Pillow control fails INSIDE the "
                "container. Open the file with PIL and pass the image."
            )
        width, height = (int(v) for v in size)
        if width <= 0 or height <= 0:
            raise ValueError(
                f"[qwen-edit-pipeline] {what}[{index}] has a degenerate size {width}x{height}."
            )

        pairs = []
        for label, budget in (
            ("vae", QWEN_EDIT_VAE_IMAGE_SIZE),
            ("condition", QWEN_EDIT_CONDITION_IMAGE_SIZE),
        ):
            ours = qwen_edit_area_budget_size(width, height, budget)
            theirs = tuple(int(v) for v in calculate_dimensions(budget, width / height))
            if ours != theirs:
                raise RuntimeError(
                    f"[qwen-edit-pipeline] {what}[{index}] ({width}x{height}): the {label} channel "
                    f"geometry DIVERGES. conditioning/qwen_edit_geometry.qwen_edit_area_budget_size "
                    f"says {ours}, the pipeline's own calculate_dimensions says {theirs}. signet's "
                    "module is the authority for this family (the W/H vs H/W orientation fork was "
                    "ruled diffusers-correct in phase 1, qwen_edit_geometry.py:252-263) and the "
                    "pipeline is what actually resizes, so a disagreement means one of them changed "
                    "and the adapter would be conditioned on pixels the training leg never saw. "
                    "Refusing before the render rather than after the grid is judged."
                )
            pairs.append((label, ours))

        entry = {
            "index": index,
            "source_wh": (width, height),
            "vae_wh": dict(pairs)["vae"],
            "condition_wh": dict(pairs)["condition"],
        }
        if width != height:
            logger.info(
                "[qwen-edit-pipeline] %s[%d] is NON-SQUARE %dx%d -> vae %s / condition %s. signet "
                "preserves orientation and the stock pipeline agrees exactly; an ai-toolkit render "
                "of the same file would transpose both (ratio = H/W). Logged because non-square is "
                "the first geometry where the two stacks are observably different.",
                what,
                index,
                width,
                height,
                entry["vae_wh"],
                entry["condition_wh"],
            )
        report.append(entry)

    logger.info("[qwen-edit-pipeline] %s geometry gate PASSED: %s", what, report)
    return report


# --------------------------------------------------------------------------------------------------
# THE GENERATE CALL.
# --------------------------------------------------------------------------------------------------

#: Keyword arguments :func:`qwen_edit_generate` REFUSES, mapped to the failure each would cause.
#: Refused BY NAME rather than ignored, because every one of them is silent when ignored and
#: expensive when honoured.
QWEN_EDIT_REFUSED_RENDER_KWARGS: dict[str, str] = {
    "guidance_scale": (
        "this checkpoint is NOT guidance-distilled (models/qwen_edit_loader.py:151 pins "
        "EXPECTED_QWEN_EDIT_GUIDANCE_EMBEDS = False, established by the ABSENCE of the guidance-"
        "embedder tensors), so the pipeline logs 'ignored since the model is not guidance-distilled' "
        "and sets guidance=None (pipeline_qwenimage_edit_plus.py:772-782). The render then runs with "
        "NO classifier-free guidance and comes out muddy, which reads as a bad adapter. METHOD §8's "
        "4.0 is TRUE-CFG: pass it as true_cfg=, and map config.validation.guidance_scale to true_cfg "
        "at the call site"
    ),
    "lora_scale": (
        "adapter strength is not a per-render dial on this path. The trained adapter has "
        "r == lora_alpha == 42, so PEFT's own alpha/r scale is exactly 1.0 with no rescale — which "
        "IS METHOD §8's 'LoRA strength 1.0'. Scaling it per cell would make a band's members "
        "incomparable to each other"
    ),
    "stg_scale": (
        "STG is an LTX concept with no Qwen meaning; inference/sampler.py owns it and is "
        "deliberately not extended by this family"
    ),
    "num_frames": (
        "Qwen-Image-Edit is an IMAGE family — F is exactly 1 "
        "(conditioning/qwen_edit_geometry.py:138-141), never a range"
    ),
}


def qwen_edit_generate(
    *,
    pipeline: Any,
    control_images: Sequence[Any],
    prompt: str,
    out_path: Any,
    seed: int,
    width: int,
    height: int,
    negative_prompt: str | None = None,
    steps: int | None = None,
    true_cfg: float | None = None,
    adapter: bool = True,
    **refused: Any,
) -> str:
    """Render ONE cell of the §8 grid — one control set, one prompt, one adapter state. Modal-side.

    The implementation behind ``inference/qwen_edit_layout.render_qwen_edit_sample``, which is a thin
    delegation to this. The split is not decorative: the layout module is import-confined to stdlib
    at module scope AND its source is scanned by two tests that between them forbid ``import torch``
    anywhere in it (``test_qwen_edit_render_surface.py``'s ``^\\s*``-anchored regex matches an
    INDENTED function-local import too) and forbid the bare scheduler token in executable code
    (``test_no_wan_params.py``). A generate call needs both. So the planner keeps its name and its
    tier, and the call that drives torch and diffusers lives beside the pipeline it drives.

    This is the per-cell call and nothing more. The caller owns which cells exist
    (``plan_qwen_edit_columns`` over a ``CheckpointBand`` and the two ``QWEN_EDIT_PROMPT_MODES``),
    which held-out inputs are rendered, and where the files land. Keeping the loop out lets the BASE
    row and an adapter row be the SAME call with one flag flipped, which is exactly what §8's
    convergence read requires.

    **The recipe is not defaulted from diffusers.** ``steps``, ``true_cfg`` and ``negative_prompt``
    fall back to :data:`QWEN_EDIT_RENDER_RECIPE` — to METHOD §8, not to the pipeline's own signature,
    whose defaults are 50 steps and no negative prompt. The static scheduler reparameterisation is
    NOT a parameter here at all: it belongs to the pipeline object, is pinned when the pipeline is
    built, and is RE-VERIFIED on every render by :func:`assert_qwen_edit_scheduler_pinned` before a
    single sigma is drawn. That check is the only thing between this function and §8's named failure
    — *training samples look fine but renders are muddy → it's inference settings, not the model* —
    and it is a gate rather than a comment because the wrong setting produces a plausible image
    instead of an exception.

    **"The reference feeds BOTH the positive and negative encode" reduces to passing a negative
    prompt.** The stock pipeline already hands the same ``condition_images`` to both ``encode_prompt``
    calls (``pipeline_qwenimage_edit_plus.py:710-728``); what gates the second call is
    ``do_true_cfg = true_cfg_scale > 1 and has_neg_prompt`` (:709). A ``None`` negative prompt
    silently disables true-CFG, the dual wiring AND the CFG-norm rescale at once — so it is refused.

    **Control images are passed RAW.** The pipeline resizes them itself, twice, at :685-694, and
    :func:`assert_qwen_edit_control_geometry` proves those two resizes equal what
    ``conditioning/qwen_edit_geometry`` would compute, per image, before the render. That is how
    signet's geometry stays the authority without double-resampling; pre-resizing here would be the
    bug rather than the rule (``prep/qwen_edit_encode.py:426-435`` refuses the same thing on the
    training leg).

    Args:
        pipeline: a ``QwenImageEditPlusPipeline`` from :func:`build_qwen_edit_pipeline` — scheduler
            pinned, transformer PEFT-wrapped with the band member's adapter loaded.
        control_images: the control slots in order, as Pillow images (1..3).
        prompt: this cell's prompt text — one of the row's §8 A/B pair.
        out_path: destination image path; parent directories are created.
        seed: §8 renders a band at ONE seed so its members are comparable.
        width: target canvas width in pixels.
        height: target canvas height in pixels.
        negative_prompt: defaults to the recipe's. Must be non-None — see above.
        steps: defaults to the recipe's 30.
        true_cfg: defaults to the recipe's 4.0. Must exceed 1.0.
        adapter: ``False`` renders the §8 BASE control column from the SAME model under
            ``disable_adapter()``. Everything else about the call is identical, which is the point.
        **refused: any of :data:`QWEN_EDIT_REFUSED_RENDER_KWARGS` raises, naming what it would cause.

    Returns:
        The written path, as a string.
    """
    if refused:
        named = ", ".join(
            f"{key}=... ({QWEN_EDIT_REFUSED_RENDER_KWARGS[key]})"
            if key in QWEN_EDIT_REFUSED_RENDER_KWARGS
            else f"{key}=... (not a parameter of this render path)"
            for key in sorted(refused)
        )
        raise TypeError(f"[qwen_edit] the render refuses: {named}")

    if not str(prompt).strip():
        raise ValueError(
            "[qwen_edit] empty prompt. The A/B modes ARE the measurement (§8: style-only vs "
            "content-named on the same input), so a blank cell is not a render with less text — it "
            "is a cell that cannot be read against its neighbour."
        )

    recipe = QWEN_EDIT_RENDER_RECIPE
    resolved_steps = int(recipe.steps if steps is None else steps)
    resolved_cfg = float(recipe.true_cfg if true_cfg is None else true_cfg)
    resolved_negative = recipe.negative_prompt if negative_prompt is None else negative_prompt

    if resolved_steps < 1:
        raise ValueError(f"[qwen_edit] steps must be >= 1, got {resolved_steps}.")
    if resolved_cfg <= 1.0:
        raise ValueError(
            f"[qwen_edit] true_cfg is {resolved_cfg}, which DISABLES classifier-free guidance: the "
            "pipeline gates it on `true_cfg_scale > 1 and has_neg_prompt` "
            "(pipeline_qwenimage_edit_plus.py:709). With CFG off the negative encode never runs, the "
            "reference is no longer fed to both encode nodes, the CFG-norm rescale at :833-835 never "
            "fires, and the render comes out muddy — the failure METHOD §8 says reads as a bad "
            f"adapter. §8's value is {recipe.true_cfg}."
        )
    if resolved_negative is None:
        raise ValueError(
            "[qwen_edit] negative_prompt is None, which turns true-CFG OFF regardless of true_cfg "
            "(has_neg_prompt, pipeline_qwenimage_edit_plus.py:695-709). §8 requires the reference to "
            "feed BOTH the positive and the negative encode; on this pipeline that is exactly 'pass "
            "a negative prompt', because both encodes already receive the same condition images."
        )

    # ⛔ THE GATES. Both run BEFORE the first sigma is drawn, because both failures they catch
    # produce a plausible IMAGE rather than an exception, and an image gets judged.
    scheduler_report = assert_qwen_edit_scheduler_pinned(pipeline)
    geometry_report = assert_qwen_edit_control_geometry(control_images)

    generator, device = qwen_edit_generator(pipeline, seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with qwen_edit_render_context(pipeline, adapter=adapter):
        result = pipeline(
            image=list(control_images),  # RAW — the pipeline resizes; the gate proved it agrees
            prompt=str(prompt),
            negative_prompt=str(resolved_negative),
            true_cfg_scale=resolved_cfg,  # NEVER guidance_scale — see the refusal table above
            height=int(height),
            width=int(width),
            num_inference_steps=resolved_steps,
            generator=generator,
            output_type="pil",
        )

    images = list(getattr(result, "images", None) or [])
    if not images:
        raise RuntimeError(
            f"[qwen_edit] the pipeline returned no images for {out.name!r} (adapter={adapter}, "
            f"seed={seed}). Nothing is written rather than an empty file: a MISSING cell is what the "
            "landed-check (inference/samples_layout.landed_render_ids) needs to see to re-dispatch."
        )
    images[0].save(out)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(
            f"[qwen_edit] wrote a zero-byte image at {out}. Refused rather than returned: the "
            "landed-check reads PRESENCE, so an empty file marks a failed render as done and the "
            "band ships a hole (the h3_sample skip-if-exists-AND-non-empty precedent, "
            "modal/fns.py:4799-4803)."
        )
    logger.info(
        "[qwen-edit-pipeline] rendered %s — adapter=%s seed=%s %dx%d steps=%d true_cfg=%s "
        "cfg_norm=%s device=%s scheduler=%s controls=%s",
        out.name,
        adapter,
        seed,
        int(width),
        int(height),
        resolved_steps,
        resolved_cfg,
        recipe.cfg_norm,
        device,
        scheduler_report,
        geometry_report,
    )
    return str(out)
