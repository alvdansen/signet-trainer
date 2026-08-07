"""D-9-FILENAME — ``ADAPTER_FILENAME`` has exactly ONE definition (in ``lora.peft``).

Regression guard for the 09-01 dedup: the adapter-weights filename was previously defined TWICE
(``lora/peft.py`` AND ``train/checkpoint.py``), a two-sources-of-one-constant drift risk. This test
proves (a) ``lora.peft`` is the single definition site, (b) every other module that exposes
``ADAPTER_FILENAME`` gets it by IMPORT (object identity, not a re-typed literal), so a change to the
canonical value can never silently diverge across the save (``checkpoint``) and load
(``inference/lora_load``) paths.

CPU-only, no-GPU, no-modal. ``peft`` is a boundary dep of ``lora.peft`` (and ``train.checkpoint``
now imports it transitively); skip cleanly when absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

peft = pytest.importorskip("peft")

from signet_trainer.inference import lora_load as _lora_load  # noqa: E402
from signet_trainer.lora import peft as _peft  # noqa: E402
from signet_trainer.train import checkpoint as _checkpoint  # noqa: E402

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "signet_trainer"
#: A module-level assignment of the constant to a string literal (the "definition" pattern).
_DEFINITION_RE = re.compile(r'^ADAPTER_FILENAME\s*=\s*["\']', re.MULTILINE)


def test_adapter_filename_defined_only_in_lora_peft() -> None:
    """Exactly ONE source file assigns ``ADAPTER_FILENAME`` to a literal — and it is ``lora/peft.py``."""
    definers = []
    for path in _SRC_ROOT.rglob("*.py"):
        if _DEFINITION_RE.search(path.read_text(encoding="utf-8")):
            definers.append(path.relative_to(_SRC_ROOT).as_posix())

    assert definers == ["lora/peft.py"], (
        f"ADAPTER_FILENAME must be defined once (in lora/peft.py); found definitions in: {definers}"
    )


def test_checkpoint_reuses_the_same_object() -> None:
    """``train.checkpoint`` gets the constant by import — object identity, not a re-typed literal."""
    assert _checkpoint.ADAPTER_FILENAME is _peft.ADAPTER_FILENAME


def test_inference_reuses_the_same_object() -> None:
    """``inference.lora_load`` (the load path) shares the identical object with the save path."""
    assert _lora_load.ADAPTER_FILENAME is _peft.ADAPTER_FILENAME


def test_checkpoint_source_has_no_local_definition() -> None:
    """train/checkpoint.py must import the constant, not redefine it."""
    src = (_SRC_ROOT / "train" / "checkpoint.py").read_text(encoding="utf-8")
    assert not _DEFINITION_RE.search(src), "train/checkpoint.py must not redefine ADAPTER_FILENAME"
    assert "from signet_trainer.lora.peft import ADAPTER_FILENAME" in src, (
        "train/checkpoint.py must import ADAPTER_FILENAME from lora.peft"
    )
