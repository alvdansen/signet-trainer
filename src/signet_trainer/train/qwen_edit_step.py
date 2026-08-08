"""train.qwen_edit_step — the ``qwen_edit`` forward step, its timestep draw, and THE LOSS-WEIGHT HOOK.

Family #3's answer to ``train/step.py`` (LTX) and ``train/h3_step.py`` (MiniMax-H3). Like
``h3_step`` this is a **HARVEST**, not a derivation: every number, formula and bound below is
transcribed from the ai-toolkit files the proven house chains actually ran under, with the
``path:line`` kept inline so a future reader re-verifies instead of re-deriving.

What lives here, and what deliberately does not
-----------------------------------------------
* **HERE** — the timestep DRAW (§7 of the tensor contract), the per-timestep LOSS WEIGHT (the bsmntw
  bell curve), the device/dtype marshalling, and the closure that ``train/loop.py`` calls once per
  micro-batch.
* **NOT here** — the flow-match noising and the velocity target. Those are §6 of the tensor contract
  and they are already implemented, once, inside ``conditioning/qwen_edit.py``:
  ``noisy = (1 - sigma) * x0 + sigma * eps`` at ``qwen_edit.py:888`` and ``target = eps - x0`` at
  ``:891``, matching ``toolkit/samplers/custom_flowmatch_sampler.py:97-99`` and
  ``extensions_built_in/diffusion_models/qwen_image/qwen_image.py:390-393`` sign for sign. This
  module SUPPLIES the sigma and lets the strategy do the arithmetic — a second transcription of the
  interpolation here would be a second home for the sign, which is the one thing about a
  flow-matching objective that fails silently rather than loudly (an inverted sign trains a
  perfectly stable curve toward the wrong fixed point).
* **NOT here** — the wire format. ``QwenEditModelInputs.transformer_kwargs()``
  (``qwen_edit.py:361-384``) assembles the eight-kwarg call, for the ``h3_ref.py:463-469`` reason:
  if the strategy builds the sequence and something else builds the call, the wire format has two
  homes and drifts in one of them.

⛔ DIVERGE — THE TIMESTEP DRAW IS **NOT** ``FlowMatchingSchedule``. THE ``schedule_`` ARG IS UNUSED.
---------------------------------------------------------------------------------------------------
``train/flow_match.FlowMatchingSchedule.sample_timesteps`` draws a *shifted logit-normal*:
``sigmoid(randn * std + shift)`` with a ``uniform_prob`` fallback and a sequence-length-dependent
shift (``flow_match.py:72-101``). **ai-toolkit draws nothing of the kind.** It draws a discrete
UNIFORM index over a linear grid:

    grid    ``torch.linspace(1000, 1, 1000)``            ``custom_flowmatch_sampler.py:117``
    draw    ``torch.randint(min_idx, max_idx, (B,))``    ``BaseSDTrainProcess.py:1282-1287``
    bounds  ``min_denoising_steps=0`` / ``max_denoising_steps=999``
                                                         ``toolkit/config_modules.py:372-373``
    gather  ``timesteps = noise_scheduler.timesteps[timestep_indices]``
                                                         ``BaseSDTrainProcess.py:1293``
    sigma   ``t_01 = timesteps / 1000``                  ``custom_flowmatch_sampler.py:91``

Per the brief, the family follows **what the proven chains ran**, so the draw below is ai-toolkit's
and ``schedule_`` is accepted-and-ignored exactly the way ``modal/fns.py::_h3_step`` ignores it
(``fns.py:4111-4120``). The loop's ``rng`` still drives it, so the whole schedule stays reproducible
from ``training.seed``.

⚠ ``randint``'s upper bound is EXCLUSIVE, so indices are ``0..998`` and **``t = 1`` is never drawn**
— the reachable timestep set is ``{1000, 999, ..., 2}``, i.e. sigma in ``{1.000, 0.999, ..., 0.002}``.
``sigma = 1.0`` (index 0, pure noise) IS reachable and is not a bug. Both facts are pinned by test.

⛔ THE LOSS-WEIGHT HOOK — signet had none before this module, and the upstream knob is a NO-OP
----------------------------------------------------------------------------------------------
Two different quantities share the word "weight" and the repo already knows it
(``h3_step.py:342-357``): ``FlowMatchingSchedule``'s ``shift`` selects WHICH timesteps are sampled;
this hook scales the LOSS at whatever timestep was sampled. Nothing in signet multiplied a loss by
anything before this file — grep ``timestep_weight|loss_weight|weighing`` over ``src/`` and the only
hits are ``flow_match.py``'s sampler and one prose mention.

**The house recipe's ``timestep_type: weighted`` does not select the table its name implies.**
``custom_flowmatch_sampler.py:65-70`` builds ``weights`` from
``toolkit/timestep_weighing/default_weighing_scheme`` — and then ``:71-74`` is ``if v2: ... else:
weights = self.linear_timesteps_weights[step_indices]``, an ``if``, **not** an ``elif``, which
overwrites it unconditionally. ``set_train_timesteps`` compounds it: ``:116-117`` maps ``'weighted'``
and ``'linear'`` to the identical ``linspace``. So the 1000-entry ``default_weighing_scheme`` table
is **dead code upstream and no house run has ever trained under it.** Every proven chain trained
under ``linear_timesteps_weights`` — the *bsmntw bell curve* built at
``custom_flowmatch_sampler.py:31-42``:

    x = arange(1000);  y = exp(-2 * ((x - 500) / 1000) ** 2)
    y_shifted = y - y.min();  w = y_shifted * (1000 / y_shifted.sum())

which is what :func:`qwen_edit_timestep_weights` reproduces, in four lines, exactly. **Do not "fix"
the upstream ``if``/``elif`` and do not port the table**: doing either would silently move this
family off the distribution every shipped adapter was trained under. The weight is applied the way
ai-toolkit applies it — ``loss = loss * timestep_weight`` at
``extensions_built_in/sd_trainer/SDTrainer.py:801``.

⚠ WHERE the multiply lands, and the one thing that makes it exact
------------------------------------------------------------------
ai-toolkit multiplies the UNREDUCED per-element loss by a ``[B,1,1,1]`` view (``SDTrainer.py:796-801``)
— per-sample weighting BEFORE the reduction. signet's ``QwenEditStrategy.compute_loss`` reduces to a
0-dim scalar internally (``qwen_edit.py:1029-1031``: ``_masked_velocity_loss(...).mean()``), so by the
time this step sees the loss the per-sample axis is gone. At the LOCKED ``batch_size = 1``
(``loop.py:325``, and the house recipe) ``mean([w * l]) == w * mean([l])`` exactly, element for
element, so scaling the scalar HERE is bit-identical to scaling the ``[B,]`` vector inside the
strategy — and it leaves ``conditioning/qwen_edit.py`` untouched. That equivalence is an INVARIANT,
not an assumption: the step refuses ``B > 1`` and names ``_masked_velocity_loss``
(``qwen_edit.py:627-639``) as the exact line the hook must move to when someone unlocks batching.

⛔ NO AUTOCAST — a per-family verdict, not an omission
------------------------------------------------------
``train/step.py:273`` wraps the LTX forward in ``torch.amp.autocast``; ``modal/fns.py:4132-4139``
explicitly refuses it for H3. Family #3's verdict is **refuse**, on evidence: ``grep -rn autocast``
over the whole ai-toolkit tree returns **zero** hits in ``extensions_built_in/sd_trainer/SDTrainer.py``,
``jobs/process/BaseSDTrainProcess.py`` and ``extensions_built_in/diffusion_models/qwen_image/*`` —
the proven chain runs the qwen forward BARE. The house recipe quantizes model and text encoder to
``qfloat8`` (optimum-quanto 0.2.4), whose ``QLinear`` dequantizes to its own compute dtype; an
``autocast`` wrapper layered over that overrides a precision decision the quantizer already made.
Do not add one "for speed".

⛔ THE OFFLOADER IS NOT THIS STEP'S BUSINESS
--------------------------------------------
``offload/block_swap.BlockSwapOffloader`` works entirely through ``register_forward_pre_hook`` /
``register_forward_hook`` / ``register_full_backward_hook`` (``block_swap.py:330-355``) and is inert
by constructor early-return at ``blocks_to_swap = 0`` (``:193-196``). The step is therefore
offloader-aware by CONSTRUCTION: it performs exactly one ``model_(...)`` call and never moves a
block, never reads a block's device, and never wraps the forward in a context the hooks would have
to survive. Nothing here needs changing when swapping turns on.

CRITICAL — Anti-Pattern 6, the CONFINED tier
--------------------------------------------
Module scope is ``torch`` + stdlib + two signet siblings that are themselves confined
(``conditioning/qwen_edit`` is torch + stdlib, ``conditioning/qwen_edit_geometry`` is stdlib +
``dataclasses``). There is **no** module-scope ``modal``, ``diffusers``, ``peft`` or ``bitsandbytes``
import, so the whole timestep/weight surface is CPU-unit-testable on Windows with zero GPU, zero
Modal spend and zero weight downloads.
"""

from __future__ import annotations

from collections.abc import Callable, MutableSequence
from dataclasses import dataclass
from typing import Any

import torch

# The row width the transformer's ``img_in`` consumes — imported, never restated (the h3_ref.py:64
# doctrine). Used only by the batch-shape guard below.
from signet_trainer.conditioning.qwen_edit_geometry import QWEN_EDIT_PATCH_DIM

__all__ = [
    "QWEN_EDIT_AUTOCAST",
    "QWEN_EDIT_MAX_TIMESTEP_INDEX",
    "QWEN_EDIT_MIN_TIMESTEP_INDEX",
    "QWEN_EDIT_NUM_TRAIN_TIMESTEPS",
    "QWEN_EDIT_TIMESTEP_SCALE",
    "QwenEditTimestepDraw",
    "build_qwen_edit_step_fn",
    "qwen_edit_collate_fn",
    "qwen_edit_timestep_draw",
    "qwen_edit_timestep_grid",
    "qwen_edit_timestep_weights",
    "qwen_edit_to_device",
]


# --------------------------------------------------------------------------------------------------
# Constants — every one carries its source citation (D-NOHARDCODE).
# --------------------------------------------------------------------------------------------------

#: ``num_train_timesteps`` — ``toolkit/config_modules.py:421`` (``kwargs.get('num_train_timesteps',
#: 1000)``) and the literal the scheduler builds both the grid and the weight curve from
#: (``custom_flowmatch_sampler.py:31``). It is the SAME 1000 in three places on purpose: the grid,
#: the weight table, and the ``timesteps / 1000`` sigma conversion all index each other.
QWEN_EDIT_NUM_TRAIN_TIMESTEPS: int = 1000

#: ``min_denoising_steps`` — ``toolkit/config_modules.py:372``, clipped to ``>= 0`` at
#: ``BaseSDTrainProcess.py:1222``.
QWEN_EDIT_MIN_TIMESTEP_INDEX: int = 0

#: ``max_denoising_steps`` — ``toolkit/config_modules.py:373``, clipped to
#: ``<= num_train_timesteps - 1`` at ``BaseSDTrainProcess.py:1223``.
#:
#: ⚠ EXCLUSIVE. It is fed straight to ``torch.randint(min_idx, max_idx, ...)``
#: (``BaseSDTrainProcess.py:1282-1287``), whose upper bound does not include itself, so the drawn
#: index set is ``0..998`` and grid index 999 (``t = 1``) is unreachable. Reproduced rather than
#: "corrected": an off-by-one here changes the training distribution's endpoint, and the endpoint of
#: a flow-matching schedule is the least forgiving place to differ from the chain you warm-start from.
QWEN_EDIT_MAX_TIMESTEP_INDEX: int = 999

#: The divisor that turns a grid timestep into the ``[0, 1]`` sigma the transformer consumes.
#: ``custom_flowmatch_sampler.py:91`` (``t_01 = timesteps / 1000``) and ai-toolkit's own call site
#: (``qwen_image_edit_plus.py:336``, ``timestep / 1000``). signet carries the ALREADY-normalized
#: quantity end to end — ``QwenEditModelInputs.transformer_kwargs()`` passes ``self.video.sigma``
#: straight through as ``timestep`` (``qwen_edit.py:369-373``) — so this division happens exactly
#: once, here, and nowhere else.
QWEN_EDIT_TIMESTEP_SCALE: float = 1000.0

#: The autocast verdict for family #3. See the module docstring: ai-toolkit runs the qwen forward
#: bare and the checkpoint is qfloat8-quantized. Named as a constant rather than left implicit so
#: the decision is greppable and a future "why is there no autocast here?" lands on the reasoning
#: instead of on an edit.
QWEN_EDIT_AUTOCAST: bool = False


# --------------------------------------------------------------------------------------------------
# The timestep draw + THE LOSS-WEIGHT HOOK. Pure, torch-only, CPU-testable without a model.
# --------------------------------------------------------------------------------------------------


def qwen_edit_timestep_grid(
    num_timesteps: int = QWEN_EDIT_NUM_TRAIN_TIMESTEPS,
    *,
    device: Any = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """The training timestep grid: ``linspace(1000, 1, num_timesteps)``.

    Transcribed from ``custom_flowmatch_sampler.py:116-117``, which is the branch BOTH ``'linear'``
    and ``'weighted'`` take — they are the same grid, which is half of why the ``'weighted'`` knob is
    a no-op (the other half is the overwritten assignment at ``:65-74``).

    At the default 1000 this is exactly ``t = 1000 - index``, but it is BUILT rather than computed
    from that identity: the identity only holds at ``num_timesteps == 1000`` and a future caller that
    lowers the count would silently get a grid that no longer matches ai-toolkit's.
    """
    if num_timesteps < 1:
        raise ValueError(f"invalid num_timesteps {num_timesteps}: must be >= 1.")
    return torch.linspace(
        QWEN_EDIT_TIMESTEP_SCALE, 1.0, num_timesteps, device=device, dtype=dtype
    )


def qwen_edit_timestep_weights(
    num_timesteps: int = QWEN_EDIT_NUM_TRAIN_TIMESTEPS,
    *,
    device: Any = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """**THE LOSS-WEIGHT HOOK.** The bsmntw bell curve — the ONLY curve any house chain trained under.

    "Bell-Shaped Mean-Normalized Timestep Weighting", transcribed verbatim from
    ``toolkit/samplers/custom_flowmatch_sampler.py:31-42``::

        x = arange(num_timesteps)
        y = exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)
        y_shifted = y - y.min()                          # shift the minimum to 0
        w = y_shifted * (num_timesteps / y_shifted.sum())  # scale the MEAN to 1

    Indexed by GRID INDEX, not by timestep: ``get_weights_for_timesteps`` recovers
    ``step_indices`` by matching each timestep against ``self.timesteps``
    (``custom_flowmatch_sampler.py:61-62``) and then indexes ``linear_timesteps_weights``
    (``:74``). Since the grid is descending, index 0 is ``t = 1000`` (pure noise, weight 0) and
    index 999 is ``t = 1`` (nearly clean, weight ~0.0049): **both ends of the schedule are
    de-weighted and the middle is boosted.** Measured on this box against the formula above:
    ``w[0] = 0.000000``, ``w[250] = 1.107882``, ``w[500] = 1.579605``, ``w[999] = 0.004870``,
    ``mean = 1.000000``.

    ⛔ THIS IS NOT ``toolkit/timestep_weighing/default_weighing_scheme``. That 1000-entry table is
    what ``timestep_type: "weighted"`` *appears* to select at
    ``custom_flowmatch_sampler.py:65-70`` — and it is dead: ``:71-74`` is an ``if v2: / else:``,
    not an ``elif``, so it overwrites the table lookup on every path. **The table has never
    weighted a single house training step.** Porting it would move this family off the
    distribution every proven adapter was trained under, at no visible symptom. Reproduce the bell
    curve; leave the table where it is.

    Returns a ``[num_timesteps]`` tensor. Cheap enough to rebuild per call (1000 floats), so it is
    a function rather than a module-level tensor — a module-level one would pin a device and a
    dtype at import time, which is exactly the kind of hidden placement this house closes by
    construction.
    """
    if num_timesteps < 1:
        raise ValueError(f"invalid num_timesteps {num_timesteps}: must be >= 1.")
    x = torch.arange(num_timesteps, dtype=torch.float32, device=device)
    y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)
    y_shifted = y - y.min()
    return (y_shifted * (num_timesteps / y_shifted.sum())).to(dtype)


@dataclass(frozen=True)
class QwenEditTimestepDraw:
    """One sample's timestep, in all four forms the step needs, from ONE draw.

    Carrying them together is the point: ``sigma`` noises the target and ``loss_weight`` scales the
    resulting loss, and the two MUST come from the same index. Two independent lookups is how a
    trainer ends up weighting step N's loss by step N-1's timestep — arithmetically silent, and
    invisible in a loss curve.

        index         the drawn grid index, ``0..num_timesteps-2`` (upper bound exclusive)
        timestep      ``grid[index]`` — the 0..1000 quantity ai-toolkit calls ``timesteps``
        sigma         ``timestep / 1000`` — the ``[0, 1]`` value the transformer consumes
        loss_weight   ``bsmntw[index]`` — the bell-curve multiplier for THIS step's loss
    """

    index: int
    timestep: float
    sigma: float
    loss_weight: float


def qwen_edit_timestep_draw(
    rng: Any,
    *,
    num_timesteps: int = QWEN_EDIT_NUM_TRAIN_TIMESTEPS,
    min_index: int = QWEN_EDIT_MIN_TIMESTEP_INDEX,
    max_index: int = QWEN_EDIT_MAX_TIMESTEP_INDEX,
    timestep_weighting: bool = True,
) -> QwenEditTimestepDraw:
    """Draw ONE training timestep the way ai-toolkit draws it, and read its weight off the same index.

    ``rng`` is the loop's ``numpy.random.Generator`` (``loop.py:298``, seeded from
    ``training.seed``). ai-toolkit uses ``torch.randint`` on the torch RNG; signet uses the loop's
    numpy generator for the same reason ``h3_draw_timesteps`` does (``h3_step.py:360-399``) — the
    house property is that the whole schedule is reproducible from ``training.seed``, and the
    DISTRIBUTION is what has to match, not the bit generator. Both are discrete-uniform over
    ``[min_index, max_index)``: ``numpy.random.Generator.integers`` and ``torch.randint`` share the
    half-open convention.

    ⚠ ONE draw per STEP, not per batch item. ``BaseSDTrainProcess.py:1282-1287`` draws
    ``(batch_size,)`` indices; signet's loop locks ``batch_size = 1`` (``loop.py:325`` — "bs=1
    LOCKED (multi-bucket: every sample carries its own shape)"), so one draw per step IS one draw
    per batch item here. :func:`build_qwen_edit_step_fn` refuses a realized batch above 1 rather
    than broadcasting this single draw across it.

    Args:
        rng: the loop's numpy generator.
        num_timesteps: grid length. 1000 — the recipe's, not a knob.
        min_index: inclusive lower bound (0).
        max_index: EXCLUSIVE upper bound (999, so 999 itself is unreachable).
        timestep_weighting: ``True`` (the recipe) reads the bell curve; ``False`` pins the weight at
            1.0 and is a DEPARTURE from every proven chain — it exists so an ablation can be stated
            explicitly in a config rather than achieved by deleting a line.
    """
    if not 0 <= min_index < max_index <= num_timesteps:
        raise ValueError(
            f"invalid timestep index bounds [{min_index}, {max_index}) over a {num_timesteps}-entry "
            f"grid: require 0 <= min_index < max_index <= num_timesteps. ai-toolkit's own bounds are "
            f"[0, 999) over 1000 (config_modules.py:372-373 clipped at "
            f"BaseSDTrainProcess.py:1222-1223), which deliberately leaves t = 1 unreachable."
        )
    index = int(rng.integers(min_index, max_index))
    grid = qwen_edit_timestep_grid(num_timesteps)
    timestep = float(grid[index])
    weight = (
        float(qwen_edit_timestep_weights(num_timesteps)[index]) if timestep_weighting else 1.0
    )
    return QwenEditTimestepDraw(
        index=index,
        timestep=timestep,
        sigma=timestep / QWEN_EDIT_TIMESTEP_SCALE,
        loss_weight=weight,
    )


# --------------------------------------------------------------------------------------------------
# Batch marshalling
# --------------------------------------------------------------------------------------------------


def qwen_edit_to_device(value: Any, device: Any, dtype: Any) -> Any:
    """Recursively move a precomputed qwen_edit sample onto ``device``, casting FLOATS to ``dtype``.

    Walks dicts AND lists because the ``qwen_edit_control_latents`` payload carries a ``"controls"``
    LIST of per-slot dicts (``qwen_edit.py:1050-1054``) whose entries mix strings, ints and
    differently-shaped tensors — the same shape as H3's reference payload, and the same reason
    ``modal/fns.py::_h3_to_device`` recurses.

    ⚠ Integer and boolean tensors keep their dtype. ``prompt_embeds_mask`` is int64 BY CONTRACT
    (DIVERGE #2, ``qwen_edit.py:33-40``): ``txt_seq_lens`` is its SUM, and a float cast would make
    that sum a float that ``transformer_qwenimage.py:235`` then slices a buffer with.

    Deliberately a LOCAL implementation rather than an import of H3's: coupling family #3 to family
    #2's private helper would make an H3 refactor a silent Qwen behaviour change
    (``qwen_edit.py:387-390`` states the rule for the strategy's helpers; it holds here too).

    ⚠ This is not the only move: ``QwenEditStrategy`` also casts each payload to its OWN
    ``device``/``dtype`` (``qwen_edit.py:798, 835-836, 869, 876``). The double move is idempotent
    and intentional — doing it HERE first means the 2x2 pack (``_rows_and_grid`` ->
    ``pack_fn``) runs on the compute device instead of packing on CPU and moving 4096x64 afterwards.
    """
    if isinstance(value, torch.Tensor):
        moved = value.to(device)
        return moved.to(dtype) if moved.is_floating_point() else moved
    if isinstance(value, dict):
        return {k: qwen_edit_to_device(v, device, dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [qwen_edit_to_device(v, device, dtype) for v in value]
    return value


def qwen_edit_collate_fn(batch: Any) -> Any:
    """Take the single sample through untouched — the ``collate_fn`` half of the loop seam.

    ``train/loop.py:284-287`` documents why H3 passes ``lambda b: b[0]``, and qwen_edit needs it for
    the identical reason: the control payload is a LIST of per-slot dicts carrying strings (``path``,
    ``stem``, ``fill``) and tensors that legitimately differ in shape between slots, which torch's
    default collate would restructure into batched leaves — turning ``entry["stem"]`` from ``str``
    into ``[str]`` and breaking ``resolve_control_slots``' stem check by TYPE rather than by value.
    ``batch_size`` is LOCKED at 1 (``loop.py:325``), so passing the sample through loses nothing.

    Named (not a lambda) so the loop's ``collate_fn`` argument reads as a contract at the call site
    and so a test can assert the identity rather than the closure.
    """
    if not isinstance(batch, (list, tuple)) or len(batch) != 1:
        raise ValueError(
            f"qwen_edit_collate_fn expected exactly one sample (batch_size is LOCKED at 1, "
            f"loop.py:325), got {type(batch).__name__} of length "
            f"{len(batch) if isinstance(batch, (list, tuple)) else 'n/a'}. A larger batch would "
            f"also break the per-step timestep draw, which is one index per STEP — see "
            f"qwen_edit_timestep_draw."
        )
    return batch[0]


# --------------------------------------------------------------------------------------------------
# Placement reconciliation — the strategy owns a device/dtype and so does the loop
# --------------------------------------------------------------------------------------------------


def _device_conflict(strategy_device: Any, loop_device: Any) -> str | None:
    """Describe a real strategy/loop device split, or ``None`` when the two agree.

    Index-normalized the way ``offload/block_swap._same_device`` is (``block_swap.py:116-138``,
    WR-03): ``"cuda"`` and ``torch.device("cuda:0")`` are the SAME device and must not be reported
    as a split, while ``cpu`` vs ``cuda`` must. A split is not cosmetic here — ``QwenEditStrategy``
    draws the noise with ``torch.randn(..., device=self.device, generator=generator)``
    (``qwen_edit.py:878-880``) and this module builds the generator on the LOOP's device, so a
    mismatch is a ``RuntimeError`` from the generator, thrown from inside the strategy, naming
    neither party.
    """
    try:
        a = torch.device(strategy_device)
        b = torch.device(loop_device)
    except (TypeError, RuntimeError, ValueError):
        return f"strategy device {strategy_device!r} / loop device {loop_device!r} (unparseable)"
    if a.type != b.type:
        return f"strategy is on {a} but the loop resolved {b}"
    if a.index is not None and b.index is not None and a.index != b.index:
        return f"strategy is on {a} but the loop resolved {b}"
    return None


# --------------------------------------------------------------------------------------------------
# The step — the injected seam train/loop.py calls once per micro-batch
# --------------------------------------------------------------------------------------------------


def build_qwen_edit_step_fn(  # noqa: PLR0913 — every argument is a recipe fact with a citation
    strategy: Any,
    *,
    seed: int,
    timestep_weighting: bool = True,
    num_timesteps: int = QWEN_EDIT_NUM_TRAIN_TIMESTEPS,
    min_timestep_index: int = QWEN_EDIT_MIN_TIMESTEP_INDEX,
    max_timestep_index: int = QWEN_EDIT_MAX_TIMESTEP_INDEX,
    record: MutableSequence[QwenEditTimestepDraw] | None = None,
) -> Callable[..., torch.Tensor]:
    """Build the ``qwen_edit`` per-step forward for ``train/loop.py``'s ``step_fn`` seam.

    The returned callable has EXACTLY the signature ``loop.py:274-282`` documents and
    ``loop.py:351`` calls::

        step_fn(model, batch, schedule, rng, *, device, dtype) -> loss

    and returns a 0-dim float32 scalar, never a ``[B,]`` leak to ``loop.py``'s ``backward()``
    (``step.py:195``). The loop's cadence — accumulate, clip, step, save, commit, callback, commit —
    is untouched; this supplies only the forward.

    The body, in order (mirroring ``modal/fns.py:4111-4141``):

      1. move the sample to the loop's device/dtype (:func:`qwen_edit_to_device`);
      2. draw ONE timestep index and read its sigma AND its bell-curve weight off that same index
         (:func:`qwen_edit_timestep_draw`);
      3. inject the sigma into the batch, so the strategy's ``_resolve_sigma``
         (``qwen_edit.py:733-762``) reads it rather than falling through to ``sigma_fn``. That is
         deliberate: the drawn index must be the SAME index the weight came from, and a
         strategy-injected ``sigma_fn`` would draw independently of this step's bookkeeping;
      4. ``strategy.prepare_training_inputs(batch, generator=...)`` — which does the §6 flow-match
         noising and velocity target (``qwen_edit.py:888, 891``) and packs
         ``[target | ctrl_0 | ... ]``;
      5. ``model(**inputs.transformer_kwargs())`` — BARE, no autocast (see the module docstring);
      6. ``strategy.compute_loss(...)`` — the masked velocity loss over the target PREFIX;
      7. **the loss-weight hook**: multiply by this step's bell-curve weight
         (``SDTrainer.py:801``).

    Args:
        strategy: an already-constructed ``QwenEditStrategy``, carrying its own ``pack_fn`` /
            ``blank_latent_fn`` / ``device`` / ``dtype``. Injected rather than built here for the
            ``ic_lora.py`` / ``h3_ref.py`` reason: this module must stay importable and testable
            without the model, the VAE or a config.
        seed: the run seed (``training.seed``). Seeds the ``torch.Generator`` that drives the noise
            draw and the caption-dropout draw inside the strategy. It is a SEPARATE stream from the
            loop's numpy ``rng`` (which drives the timestep index) — see the generator note below.
        timestep_weighting: ``True`` == the locked recipe (``timestep_type: weighted``, which the
            upstream bug resolves to the bsmntw bell curve). ``False`` is an explicit ablation.
        num_timesteps / min_timestep_index / max_timestep_index: the draw's grid and bounds.
        record: optional sink appended with each step's :class:`QwenEditTimestepDraw`. A DIAGNOSTIC
            surface for the campaign banner and the smoke test; the loop never reads it, and
            ``None`` (the default) records nothing.

    ⚠ THE GENERATOR. ``loop.py:298`` hands a ``numpy.random.Generator``; ``QwenEditStrategy``'s
    ``generator`` argument is a ``torch.Generator`` (``qwen_edit.py:844-845``, fed to
    ``torch.randn(..., generator=...)`` at ``:878-880`` and ``torch.rand`` at ``:820``). The two are
    not interchangeable and nothing upstream bridges them, so this closure constructs its own,
    seeded from ``seed``. It is built LAZILY on the first step because a ``torch.Generator`` is bound
    to a device at construction and the device only arrives as a per-step kwarg — a CPU generator
    handed to ``torch.randn(..., device="cuda")`` raises, from inside the strategy, at step 1 of a
    metered run.
    """
    if seed is None:
        raise ValueError(
            "build_qwen_edit_step_fn requires an explicit seed: the strategy's noise and "
            "caption-dropout draws run off a torch.Generator this closure owns, and an unseeded "
            "generator makes a resumed run diverge from the run it resumed — silently, since the "
            "loss curve is identically shaped either way."
        )
    if not 0.0 <= float(getattr(strategy, "caption_dropout_rate", 0.0)) <= 1.0:
        raise ValueError(
            f"strategy.caption_dropout_rate is {strategy.caption_dropout_rate!r}: expected 0.0..1.0."
        )

    # Per-closure state. A dict rather than ``nonlocal`` to mirror ``modal/fns.py:4109``'s
    # ``step_counter = {"n": 0}`` — the established shape for "the closure keeps a counter".
    state: dict[str, Any] = {"steps": 0, "generator": None, "device": None}

    def _generator(device: Any) -> torch.Generator:
        """The seeded torch.Generator, built once on the loop's device. See the docstring note."""
        if state["generator"] is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            state["generator"] = generator
            state["device"] = device
            return generator
        if _device_conflict(state["device"], device) is not None:
            raise RuntimeError(
                f"the training device changed mid-run: the step's generator was built on "
                f"{state['device']} at step 1 and step {state['steps']} was handed {device}. A "
                f"torch.Generator is bound to its device at construction, so continuing would "
                f"either raise inside torch.randn or silently re-seed the noise stream."
            )
        return state["generator"]

    def _qwen_edit_step(
        model_: Any, batch_: Any, schedule_: Any, rng_: Any, *, device: Any, dtype: Any
    ) -> torch.Tensor:
        """One qwen_edit step: draw -> noise+pack -> bare forward -> masked prefix loss -> weight.

        ``schedule_`` (the LTX ``FlowMatchingSchedule``) is deliberately UNUSED — see the module
        docstring's DIVERGE note. Its shifted-logit-normal draw is a different distribution from the
        discrete-uniform-over-a-linear-grid draw every proven qwen chain trained under, and the
        shift parameters are not even the same KIND of quantity (``h3_step.py:342-357`` makes the
        identical point for family #2). The loop's ``rng_`` still drives the draw, so the schedule
        stays reproducible from ``training.seed``.
        """
        state["steps"] += 1
        conflict = _device_conflict(getattr(strategy, "device", device), device)
        if conflict is not None:
            raise RuntimeError(
                f"qwen_edit strategy/loop device split — {conflict}. The strategy draws the target "
                f"noise on ITS device (qwen_edit.py:878-880) while this step builds the generator "
                f"and moves the batch onto the LOOP's, so the split surfaces as a generator error "
                f"thrown from inside the strategy that names neither party. Construct "
                f"QwenEditStrategy(device=...) from the same device train_loop resolves "
                f"(loop.py:296)."
            )

        batch = qwen_edit_to_device(batch_, device, dtype)
        if not isinstance(batch, dict):
            raise TypeError(
                f"qwen_edit step expected a dict sample, got {type(batch).__name__}. Pass "
                f"collate_fn=qwen_edit_collate_fn to train_loop — torch's default collate "
                f"restructures the per-slot control list and turns each entry's 'stem' string into "
                f"a one-element list."
            )

        # ── (2)+(3) ONE draw; the sigma and the loss weight both come off its index ───────────────
        draw = qwen_edit_timestep_draw(
            rng_,
            num_timesteps=num_timesteps,
            min_index=min_timestep_index,
            max_index=max_timestep_index,
            timestep_weighting=timestep_weighting,
        )
        # ``sigma`` is the ONLY key this step injects. ``modal/fns.py:4123-4124`` also stamps
        # ``segment_index`` / ``step`` because H3's strategy CONSUMES them for its reference-dropout
        # draw; QwenEditStrategy consumes neither, and a batch key no strategy reads is the
        # silently-ignored-knob class this house refuses on config fields — it reads as wired-up.
        batch["sigma"] = draw.sigma
        if record is not None:
            record.append(draw)

        # ── (4) noise the target, pack [target | controls], build the loss mask ──────────────────
        inputs = strategy.prepare_training_inputs(batch, generator=_generator(device))

        realized_batch = int(inputs.video.sigma.numel())
        if realized_batch != 1:
            raise RuntimeError(
                f"realized batch size is {realized_batch}, but this step applies ONE per-timestep "
                f"loss weight to the already-reduced scalar loss. That is bit-identical to "
                f"ai-toolkit's per-sample weighting ONLY at batch 1 "
                f"(mean([w*l]) == w*mean([l])), which train/loop.py:325 locks. To batch this "
                f"family, move the multiply into QwenEditStrategy.compute_loss and apply it to the "
                f"[B,] vector _masked_velocity_loss returns (qwen_edit.py:627-639) BEFORE its "
                f".mean() at :1029-1031, and draw one timestep index per batch item the way "
                f"BaseSDTrainProcess.py:1282-1287 does."
            )
        rows = inputs.video.latent
        if rows.ndim != 3 or rows.shape[2] != QWEN_EDIT_PATCH_DIM:
            raise RuntimeError(
                f"packed hidden_states is {tuple(rows.shape)}, expected "
                f"[1, rows, {QWEN_EDIT_PATCH_DIM}]. {QWEN_EDIT_PATCH_DIM} is img_in's input width "
                f"on the live checkpoint (img_in.weight [3072, 64]); a different width is a "
                f"different checkpoint, and the forward below would fail with a matmul shape error "
                f"that names 3072 and never mentions the packer."
            )

        # ── (5) the forward. BARE — no autocast (QWEN_EDIT_AUTOCAST, and the module docstring). ──
        # ``transformer_kwargs()`` already carries ``return_dict=False`` (qwen_edit.py:383), which
        # is why this call has no extra kwarg: the wire format has ONE home.
        output = model_(**inputs.transformer_kwargs())

        # ── (6) masked velocity loss over the TARGET PREFIX (never the control suffix) ───────────
        loss = strategy.compute_loss(inputs, output)

        # ── (7) THE LOSS-WEIGHT HOOK — SDTrainer.py:801, ``loss = loss * timestep_weight`` ───────
        # A python float, so no graph node and nothing to detach (ai-toolkit detaches its weight
        # TENSOR at SDTrainer.py:797/799 for the same reason: the weight is a constant lookup).
        # At timestep_weighting=False this is a multiply by 1.0 and the step is byte-identical to
        # an unweighted one.
        return loss * draw.loss_weight

    return _qwen_edit_step
