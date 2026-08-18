"""train.checkpoint — ``CheckpointManager``: save / find_latest / prune / resume.

Ported from enochiatron ``scripts/train/train.py::CheckpointManager`` (702-925): ``save`` (745-768),
``find_latest`` (813-825), ``prune`` (790-811), ``resume`` (827-885). The source's HF-upload path
(887-922) is DROPPED — signet checkpoints stay on the private Modal Volume (privacy + cost; threat
T-03-44).

⚠ LANDMINE #1 FIX (the highest-priority correctness finding for SC#4):
    The source ``CheckpointManager.resume()`` restored optimizer / scheduler / step / RNG ONLY —
    NOT the adapter weights. In enochiatron the adapter was reloaded by a SEPARATE path
    (``_p1_checkpoint`` → ``set_peft_model_state_dict``, train.py:1864-1881). A naive same-output-dir
    relaunch therefore resumed a ZERO-init LoRA against a restored optimizer pointing at step N =
    silent corruption — SC#4 ("resumes and continues, not restarts") fails with no error.

    signet's ``resume()`` does BOTH: it re-injects the saved adapter via
    ``set_peft_model_state_dict(adapter_model.safetensors)`` AND restores ``training_state.pt``. It
    also asserts the adapter state_dict is non-empty (threat T-03-41 — a zeroed/corrupted checkpoint
    must not silently continue). Warning sign of a regression: loss jumps back up / samples regress
    after a "resume".

Save format mirrors the source: ``checkpoint-step-{step:05d}-loss-{loss:.4f}/`` holding PEFT's
``adapter_model.safetensors`` + ``adapter_config.json`` (``model.save_pretrained``) and a companion
``training_state.pt`` (``{step, loss, optimizer/scheduler state, RNG}``). The Volume ``.commit()``
after each save is the loop's responsibility (commit-or-vanish — train/loop.py).

Import note: the HEAVY peft/safetensors CALLS (``set_peft_model_state_dict`` /
``safetensors.load_file``) are imported FUNCTION-LOCAL inside ``resume`` so the manager's
save/find_latest/prune surface stays usable without invoking peft. ``ADAPTER_FILENAME`` is the
single-sourced constant re-used from ``lora.peft`` (D-9-FILENAME — one definition, no drift); it is
a bare string, and ``lora.peft`` is the canonical home already re-exported by ``inference/lora_load.py``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch

# Single-source the adapter filename from lora.peft (D-9-FILENAME) — do NOT redefine it here.
from signet_trainer.lora.peft import ADAPTER_FILENAME

logger = logging.getLogger(__name__)

#: Checkpoint directory name prefix; the step is zero-padded to 5 digits (``checkpoint-step-00200``).
CHECKPOINT_PREFIX = "checkpoint-step-"
#: The companion training-state file (optimizer / scheduler / step / RNG).
TRAINING_STATE_FILENAME = "training_state.pt"

_STEP_RE = re.compile(rf"^{re.escape(CHECKPOINT_PREFIX)}(\d+)")


class CheckpointManager:
    """Save + resume LoRA training checkpoints to a directory (a Modal Volume mount Modal-side).

    Args:
        output_dir: the checkpoint root (the run's ``output_dir`` — relaunch the SAME dir to resume).
        keep_n: if set, prune to the ``keep_n`` most-recent checkpoints after each save.
    """

    def __init__(self, output_dir: str | Path, keep_n: int | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.keep_n = keep_n

    # ----------------------------------------------------------------------------------------------
    # save
    # ----------------------------------------------------------------------------------------------

    def save(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        step: int,
        loss: float,
    ) -> Path:
        """Write the PEFT adapter + ``training_state.pt`` for ``step``; return the checkpoint dir.

        ATOMIC (audit #3): the adapter + ``training_state.pt`` are written into a STAGING dir whose
        name (leading ``.``) never matches ``_STEP_RE`` — so ``_all_checkpoints()``/``find_latest``
        ignore it while it is being written — then ``os.rename``-d onto the final regex-matching name.
        A directory ``os.rename`` is atomic on one filesystem (the Volume mount), so the final
        ``checkpoint-step-*`` name only ever appears fully-written. A container killed mid-``save``
        can therefore never publish a name-valid-but-half-written dir that poisons resume()/sample().
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_name = f"{CHECKPOINT_PREFIX}{step:05d}-loss-{loss:.4f}"
        ckpt_dir = self.output_dir / final_name
        # Leading '.' => never matches ^checkpoint-step- (invisible to _all_checkpoints/find_latest).
        staging = self.output_dir / f".tmp-{final_name}"

        # Clear any leftover staging dir from a prior crashed save before writing.
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        # PEFT adapter (adapter_model.safetensors + adapter_config.json) — into STAGING.
        model.save_pretrained(str(staging))

        state: dict[str, Any] = {
            "step": step,
            "loss": loss,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng_torch": torch.get_rng_state(),
            "rng_torch_cuda": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
        }
        torch.save(state, staging / TRAINING_STATE_FILENAME)

        # Atomic publish: the final regex-matching name appears only once fully written.
        # If a stale final dir exists (e.g. a re-save of the same step), clear it first —
        # os.rename onto a non-empty dir fails on some platforms.
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        os.rename(staging, ckpt_dir)

        # Prune AFTER the rename so it only ever sees completed dirs.
        if self.keep_n is not None:
            self.prune(self.keep_n)

        logger.info("Saved checkpoint: %s", ckpt_dir.name)
        return ckpt_dir

    # ----------------------------------------------------------------------------------------------
    # find_latest / prune
    # ----------------------------------------------------------------------------------------------

    def _all_checkpoints(self) -> list[Path]:
        """All checkpoint dirs sorted by step ascending (the highest step is last)."""
        if not self.output_dir.exists():
            return []

        def _step_of(path: Path) -> int:
            match = _STEP_RE.match(path.name)
            return int(match.group(1)) if match else -1

        dirs = [
            d
            for d in self.output_dir.iterdir()
            if d.is_dir() and _STEP_RE.match(d.name) is not None
        ]
        return sorted(dirs, key=_step_of)

    @staticmethod
    def _is_complete(path: Path) -> bool:
        """True iff BOTH the adapter and ``training_state.pt`` exist (a fully-written checkpoint).

        Belt-and-braces with atomic save: even though ``save()`` publishes only complete dirs,
        find_latest still verifies completeness so a legacy/externally-created half-written dir
        (or a save from an older non-atomic build) can never poison resume()/sample() (audit #3).
        """
        return (path / ADAPTER_FILENAME).exists() and (path / TRAINING_STATE_FILENAME).exists()

    @staticmethod
    def is_complete(path: Path) -> bool:
        """PUBLIC completeness check — the SAME rule ``find_latest`` filters with.

        Exists so a caller that resolves a checkpoint by NAME instead of by step (``h3_sample``'s
        ``h3.render_checkpoint_name`` pin, D-10-DEF-19) inherits the identical filter rather than
        re-deriving "what counts as a checkpoint" a second time (D-9-DEDUP). A pinned half-written
        dir must be as unusable as a walked-back one.
        """
        return CheckpointManager._is_complete(path)

    def find_latest(self) -> Path | None:
        """The highest-step COMPLETE checkpoint dir, or ``None`` if none exist (cold start).

        Walks ``_all_checkpoints()`` from highest step downward and returns the first COMPLETE dir,
        skipping any half-written newer dir (audit #3 walk-back). resume() and sample()
        (fns.py:1582) both resolve their adapter through this method, so both inherit the filter.
        """
        for candidate in reversed(self._all_checkpoints()):
            if self._is_complete(candidate):
                return candidate
        return None

    def latest_step(self) -> int | None:
        """The step of the highest-step COMPLETE checkpoint, or ``None`` on a cold start.

        A cheap filesystem-only read (no model, no ``torch.load`` of ``training_state.pt``) — for
        callers that only need "how far did this run actually get" without paying to construct a
        model first. Added for the AUDIT #34 direction 7 acceptance-failed marker (h3_train):
        checking whether a checkpoint has advanced past the step a zero-delta verdict was recorded
        at is exactly this question.
        """
        latest = self.find_latest()
        if latest is None:
            return None
        match = _STEP_RE.match(latest.name)
        return int(match.group(1)) if match else None

    def prune(self, keep_n: int) -> None:
        """Delete all but the ``keep_n`` most-recent checkpoints (no-op when ``keep_n <= 0``)."""
        if keep_n <= 0:
            return
        ckpts = self._all_checkpoints()
        for stale in ckpts[:-keep_n]:
            shutil.rmtree(stale, ignore_errors=True)
            logger.info("Pruned old checkpoint: %s", stale.name)

    # ----------------------------------------------------------------------------------------------
    # resume — THE landmine #1 fix
    # ----------------------------------------------------------------------------------------------

    def resume(self, model: Any, optimizer: Any = None, scheduler: Any = None) -> int:
        """Restore from the latest checkpoint; return the step to continue from (0 = cold start).

        Re-injects the saved LoRA adapter (landmine #1) AND restores optimizer / scheduler / RNG.
        On an empty / missing ``output_dir`` returns 0 without error.
        """
        latest = self.find_latest()
        if latest is None:
            # PRINTED, not logger.info'd. "Did this run CONTINUE or RESTART?" is the single most
            # consequential line in a long run's log, and it was invisible: Modal shows stdout at
            # the default WARNING level, so an INFO-level report leaves no trace in a metered run.
            # A silent restart burns the whole budget from step 0 and looks exactly like progress.
            # Same reasoning as the qfloat8 report — a load-bearing event needs an artifact, and the
            # artifact carries NUMBERS so the wrong answer is visible rather than merely absent.
            banner = (
                f"[checkpoint] COLD START at step 0 — no complete checkpoint found in "
                f"{self.output_dir}. If a prior run should have left one, STOP: this run is about "
                f"to retrain from scratch, not continue."
            )
            print(banner)
            logger.info(banner)
            return 0

        # ── LANDMINE #1: re-inject the adapter weights resume() historically did NOT restore ──────
        # ``load_adapter_state_into`` rather than ``set_peft_model_state_dict`` directly — it picks
        # the path this model can survive (see LANDMINE #1b below) and owns the peft import.
        from safetensors.torch import load_file  # noqa: PLC0415

        from signet_trainer.lora.peft import load_adapter_state_into  # noqa: PLC0415

        adapter_weights = load_file(str(latest / ADAPTER_FILENAME))
        if not adapter_weights:
            # A zeroed/corrupted adapter must NOT silently continue (threat T-03-41).
            raise RuntimeError(f"Resume aborted: empty adapter state_dict in {latest}")

        # ── LANDMINE #1b: set_peft_model_state_dict CANNOT be used on a QUANTIZED model ──────────
        #
        # Delegated to ``lora/peft.load_adapter_state_into``, which carries the full reasoning, the
        # direct-assignment path and both refusals. It lives there rather than here because the
        # property belongs to loading an adapter onto a QUANTIZED MODEL, not to resuming — and this
        # file learned that the expensive way twice over. The first time it cost a 5000-step run
        # (phase 1 interrupted at step 500; every one of Modal's ten retries died on
        # ``KeyError: '...timestep_embedder.linear_1.weight._data'``). The second time the fix was
        # here and ONLY here, so the §8 band render walked into the identical failure through
        # ``load_adapter_into``. One implementation, one place to fix it.
        load_adapter_state_into(model, adapter_weights, adapter_name="default", what=str(latest))

        # ── restore optimizer / scheduler / step / RNG from training_state.pt ────────────────────
        state = torch.load(
            latest / TRAINING_STATE_FILENAME, map_location="cpu", weights_only=False
        )
        if optimizer is not None and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("rng_torch") is not None:
            torch.set_rng_state(state["rng_torch"])
        if torch.cuda.is_available() and state.get("rng_torch_cuda") is not None:
            torch.cuda.set_rng_state(state["rng_torch_cuda"])

        step = int(state["step"])
        banner = (
            f"[checkpoint] RESUMED from {latest.name} at step {step} — adapter re-injected, "
            f"optimizer/scheduler/RNG restored. Training CONTINUES from {step}, it does not restart."
        )
        print(banner)
        logger.info(banner)
        return step
