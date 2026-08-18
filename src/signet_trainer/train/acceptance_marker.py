"""train.acceptance_marker — the on-Volume marker AUDIT #34 direction 7 keys a cheap re-raise on.

``h3_train``'s post-loop acceptance signal (``h3_adapter_delta`` on a fixed batch) can return
``0.0`` deterministically: the checkpoint is already at ``max_steps``, so a Modal-driven retry
(``single_use_containers=True`` — a FRESH container, ``max_retries=10``) re-enters, does zero new
training work, and reproduces the identical ``0.0`` verdict. Without a marker, each of those
retries still pays for the full 61.7 GiB weight load before re-deriving a verdict already on
record — eleven container lives for one answer (AUDIT #34, sub-shape 3).

The fix is a marker written to the run's own checkpoint dir on a zero-delta verdict, checked at
``h3_train`` entry BEFORE the load. ``h3_train`` (``modal/fns.py``) owns the Volume plumbing
(``checkpoints_vol.commit()`` / ``.reload()``) and ``CheckpointManager``; this module owns the
KEY — what makes the marker still describe the CURRENT state vs. a stale verdict from a run that
has since moved on.

⛔ **The fix is more dangerous than the problem if the key is wrong** — the same warning
``inference/render_key.py`` carries. A marker that fires on a run it should not block would turn a
genuinely fixed config (different LoRA targets, a bumped ``max_steps``, or just a different
``output_dir``) into a permanently-FAILED run that never gets a real second look. The key here is
scoped in two independent ways:

  * **By RUN** — the caller places the marker INSIDE the run's own checkpoint dir
    (``CHECKPOINTS_DIR / config.output_dir``), so a different ``output_dir`` never sees it. This
    module never chooses that path itself; it is handed ``output_dir`` explicitly so the scoping
    is visible at every call site, not buried in a default.
  * **By STEP** — :func:`acceptance_marker_is_current` compares the marker's recorded step against
    the checkpoint dir's ACTUAL latest step (``CheckpointManager.latest_step()`` — a cheap
    filesystem-only read, no model, no ``torch.load``). The marker is honoured only if nothing has
    trained since it was written. If the latest checkpoint has moved past the recorded step, real
    training happened (a genuinely new attempt — an operator fix followed by a continued run) and
    the marker is STALE: it must be cleared, never honoured.

Import tier: **stdlib only** (``json`` + ``pathlib``), no package side effects — for the same
reason ``inference/render_key.py`` and ``train/loop.checkpoint_watchdog_exceeded`` live where they
do: a test that had to ``import signet_trainer.modal.fns`` to reach this logic would drag ``modal``
into ``sys.modules`` for the whole pytest session and break the dry-run gate's Anti-Pattern-6
assertion elsewhere in the suite (see ``tests/test_h3_sample_resume.py``'s note on the same point).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = [
    "ACCEPTANCE_FAILED_MARKER_NAME",
    "acceptance_marker_is_current",
    "clear_acceptance_failed_marker",
    "marker_path",
    "read_acceptance_failed_marker",
    "write_acceptance_failed_marker",
]

#: The marker's filename inside a run's checkpoint dir. Leading-dot, mirroring the ``.tmp-``
#: staging convention in ``train/checkpoint.py`` — it must NEVER match
#: ``CHECKPOINT_PREFIX = "checkpoint-step-"``, so it stays invisible to
#: ``CheckpointManager._all_checkpoints()`` / ``find_latest()`` / ``prune()``.
ACCEPTANCE_FAILED_MARKER_NAME = ".h3-acceptance-failed.json"


def marker_path(output_dir: Path | str) -> Path:
    """The marker's path inside ``output_dir`` — the run-scoping half of the key.

    The caller supplies the run's OWN checkpoint dir (e.g. ``CHECKPOINTS_DIR / config.output_dir``)
    so the scoping is explicit at the call site rather than a default this module invents.
    """
    return Path(output_dir) / ACCEPTANCE_FAILED_MARKER_NAME


def write_acceptance_failed_marker(output_dir: Path | str, *, step: int, delta: float) -> Path:
    """Write (but do NOT commit) the marker recording a zero-delta verdict at ``step``.

    The caller (``h3_train``) is responsible for ``checkpoints_vol.commit()`` AFTER this returns
    and BEFORE it raises — the marker must be durable before the container that saw the zero-delta
    verdict exits, or a retry's fresh container would never see it.
    """
    path = marker_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"step": step, "delta": delta}))
    return path


def read_acceptance_failed_marker(output_dir: Path | str) -> dict[str, Any] | None:
    """The marker's recorded ``{"step": ..., "delta": ...}``, or ``None`` if no marker exists."""
    path = marker_path(output_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def acceptance_marker_is_current(recorded_step: int, latest_step: int | None) -> bool:
    """True iff the marker still describes the CURRENT state — a real retry, not a new attempt.

    The step-scoping half of the key (see the module docstring). ``latest_step`` is
    ``CheckpointManager(...).latest_step()`` read AT THE CALL SITE, never cached — a marker is
    current only if the checkpoint dir's actual latest step is STILL exactly the step the
    zero-delta verdict was measured against. Any mismatch — the checkpoint has advanced (real
    training happened since) or regressed / vanished (the dir was reset) — means "not current":
    the caller must treat the marker as stale and clear it rather than honour it. A false "stale"
    verdict costs one extra 61.7 GiB load; a false "current" verdict would wrongly block a
    legitimate run, which is the failure mode this function exists to rule out.
    """
    return latest_step == recorded_step


def clear_acceptance_failed_marker(output_dir: Path | str) -> None:
    """Delete the marker (a no-op if absent) — for the stale-marker case."""
    marker_path(output_dir).unlink(missing_ok=True)
