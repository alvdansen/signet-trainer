"""AUDIT #34 direction 1 — ``training.seed`` must reach torch, not just numpy.

``train_loop`` (``train/loop.py``) previously constructed its numpy generator from
``config.training.seed`` but never called ``torch.manual_seed`` and never threaded a seeded
``generator=`` into the ``DataLoader`` — so the flow-match noise draw (a real ``torch.randn_like``
in the objective) and the ``RandomSampler`` shuffle order both drew off whatever global torch RNG
state the process happened to be in. A seed-locked A/B ablation was therefore not actually
seed-locked: re-running the "same" config twice produced a different noise trajectory and a
different visit order.

This file drives the REAL ``train_loop`` (only the ``ltx_core``-dependent default ``step_fn`` is
replaced, mirroring ``tests/test_training_step.py``'s injection pattern) twice from a cold start
with the identical ``training.seed`` and asserts the DataLoader visit order and an in-step
``torch.rand`` draw come out byte-identical both times — the behavioural proof that seeding one
process-global torch RNG at loop entry is enough, independent of what the previous test in the same
pytest session happened to draw from it.

CPU-only, zero GPU, zero Modal spend.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from signet_trainer.train import loop as loop_mod
from signet_trainer.train.flow_match import FlowMatchingSchedule


class _FixedSizeDataset:
    """A tiny multi-sample dataset so DataLoader's ``RandomSampler`` has an order to shuffle."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict:
        return {"idx": i, "x": torch.zeros(1)}


class _CkptManager:
    def resume(self, model, optimizer, scheduler):  # noqa: ANN001
        return 0

    def save(self, *a, **k):  # noqa: ANN002, ANN003
        pass


class _Optim:
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass


class _Sched:
    def step(self) -> None:
        pass


def _config(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(
            mixed_precision="fp32", seed=seed, max_steps=4,
            gradient_accumulation_steps=1, max_grad_norm=1.0, checkpoint_every=1000,
            checkpoint_expected_minutes=None, checkpoint_stall_multiplier=2.5,
        ),
    )


def _run_once(seed: int) -> tuple[list[int], list[float]]:
    """One cold-start ``train_loop`` run; returns (dataloader visit order, per-step torch draws)."""
    param = nn.Parameter(torch.zeros(1))

    class _Model:
        def parameters(self):
            return iter([param])

    seen_order: list[int] = []
    seen_draws: list[float] = []

    def _step_fn(model, batch, schedule, rng, *, device, dtype):  # noqa: ANN001
        seen_order.append(int(batch["idx"]))
        # Stands in for a real unrelated torch.* draw inside the forward (the flow-match noise is
        # exactly this shape: torch.randn_like off the global RNG, not the loop's numpy `rng`).
        seen_draws.append(torch.rand(1).item())
        return (param * 2).sum()  # a real graph so .backward() has something to do

    loop_mod.train_loop(
        _Model(), _FixedSizeDataset(4), _Optim(), _Sched(),
        FlowMatchingSchedule(uniform_prob=0.0), _CkptManager(), _config(seed),
        checkpoints_vol=None, step_fn=_step_fn,
    )
    return seen_order, seen_draws


def test_two_runs_of_the_same_seed_draw_the_identical_torch_noise() -> None:
    """The flow-match-noise analog: with torch.manual_seed threaded, an unrelated torch.rand()
    call inside the step draws the SAME sequence on both runs, regardless of what earlier tests in
    this pytest session already pulled from the global torch RNG."""
    _, draws_a = _run_once(seed=42)
    _, draws_b = _run_once(seed=42)
    assert draws_a == draws_b, (
        "training.seed must pin the torch RNG at train_loop entry — two runs of the identical "
        "seed produced different torch draws, so a seed-locked A/B ablation is not seed-locked."
    )


def test_two_runs_of_the_same_seed_visit_the_dataloader_in_the_same_order() -> None:
    """The DataLoader's RandomSampler shuffle order must be pinned by training.seed via the
    generator= threaded into DataLoader, independently of any other torch draw in the process."""
    order_a, _ = _run_once(seed=7)
    order_b, _ = _run_once(seed=7)
    assert order_a == order_b, (
        "the DataLoader shuffle order must be reproducible from training.seed alone; got "
        f"{order_a} then {order_b}."
    )
    # Sanity: it actually shuffled something meaningful (not accidentally a 1-item dataset).
    assert sorted(order_a) == [0, 1, 2, 3]


def test_a_different_seed_can_draw_a_different_torch_sequence() -> None:
    """Negative control: this is not a fluke where torch.rand always returns the same thing
    regardless of the seed — two DIFFERENT seeds are very unlikely to coincide over 4 draws."""
    _, draws_a = _run_once(seed=1)
    _, draws_b = _run_once(seed=2)
    assert draws_a != draws_b


def test_the_numpy_rng_still_seeds_from_training_seed_too() -> None:
    """Direction 1 does not regress the EXISTING numpy seeding (loop.py's ``rng`` argument to
    ``step_fn`` — the timestep draw) — only ADDS the torch half that was missing."""
    seen_numpy_draws: list[float] = []

    def _step_fn(model, batch, schedule, rng, *, device, dtype):  # noqa: ANN001
        seen_numpy_draws.append(float(rng.random()))
        param = next(iter(model.parameters()))
        return (param * 2).sum()

    def _run(seed: int) -> list[float]:
        seen_numpy_draws.clear()
        param = nn.Parameter(torch.zeros(1))

        class _Model:
            def parameters(self):
                return iter([param])

        loop_mod.train_loop(
            _Model(), _FixedSizeDataset(4), _Optim(), _Sched(),
            FlowMatchingSchedule(uniform_prob=0.0), _CkptManager(), _config(seed),
            checkpoints_vol=None, step_fn=_step_fn,
        )
        return list(seen_numpy_draws)

    draws_a = _run(seed=42)
    draws_b = _run(seed=42)
    assert draws_a == draws_b
