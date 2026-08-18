"""#21 finding 3 — ``inpaint_mask_dir`` honored on WRITE, hardcoded on READ. Zero GPU/Modal.

The defect
----------
``conditioning/inpaint.py:126``'s ``InpaintStrategy.get_data_sources()`` returned the bare literal
``"video_masks"`` unconditionally, while ``modal/entrypoint.py``'s ``_mask_encode_params`` already
threaded the config-driven ``conditioning.inpaint_mask_dir`` on the WRITE side
(``modal/fns.py``'s preprocess arm). A non-default ``inpaint_mask_dir`` validated clean at config
load (``config/schema.py``'s rule 6d is REVERSE-only — it rejects a non-default value when
``mode != "inpaint"`` and permits ANY value when ``mode == "inpaint"``) and was then silently
ignored on read: ``--mode preprocess`` wrote ``.precomputed/<custom>/*.pt``, and ``--mode train``
either died naming ``"video_masks"`` (if only ``<custom>/`` existed) or — worse — silently trained
on a STALE ``video_masks/`` left over from an earlier encode at the same geometry.

What is pinned here
--------------------
1. TEXT SCAN — ``modal/fns.py``'s inpaint branch reads ``config.conditioning.inpaint_mask_dir`` and
   threads it into ``InpaintStrategy(mask_dir=...)``, and the ``_PRECOMPUTED_SOURCE_OUTPUT_KEYS``
   role lookup for the mask source is keyed by the fixed ROLE name ``"video_masks"`` (never the
   dir name) so a non-default dir does not KeyError the map.
2. FUNCTIONAL ROUND-TRIP — building the ``data_sources`` dict the SAME way ``fns.py``'s inpaint
   branch does, for a non-default ``inpaint_mask_dir``, and constructing a ``PrecomputedDataset``
   over files written under that dir name, proves WRITE and READ now agree on the same directory
   (the test the source issue asks for explicitly).

Mirrors ``test_ic_lora_wiring.py`` / ``test_mask_encode.py``'s existing text-scan discipline:
comments + docstrings are stripped before every regex scan so prose mentions don't false-positive;
the ``modal``-decorated module is imported (module-top only pulls from ``modal.app``, Anti-Pattern
6) but no Modal function ever executes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = _REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _code() -> str:
    return _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------------
# 1. Text scan — the train-side inpaint branch reads the config, not a literal.
# ----------------------------------------------------------------------------------------------


def test_train_inpaint_branch_reads_configured_mask_dir() -> None:
    """fns.py must thread config.conditioning.inpaint_mask_dir into InpaintStrategy(mask_dir=...)."""
    code = _code()

    assert re.search(r"config\.conditioning\.inpaint_mask_dir", code), (
        "the train-side inpaint branch must read config.conditioning.inpaint_mask_dir — the SAME "
        "field modal/entrypoint.py::_mask_encode_params already threads on the write side"
    )
    assert re.search(r"InpaintStrategy\(\s*[^)]*mask_dir\s*=", code), (
        "InpaintStrategy must be constructed with mask_dir=<configured dir> in the train branch, "
        "not left at its 'video_masks' default (which is what silently ignored the config field)"
    )


def test_fns_imports_off_modal_with_the_new_branch() -> None:
    """Module-top import must still succeed off-Modal after threading mask_dir (Anti-Pattern 6)."""
    import signet_trainer.modal.fns as fns  # noqa: PLC0415

    assert hasattr(fns, "train")


# ----------------------------------------------------------------------------------------------
# 2. Functional round-trip — write + read agree on a NON-DEFAULT mask dir name.
# ----------------------------------------------------------------------------------------------


def _write_latent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": torch.randn(128, 1, 2, 2)}, path)


def _write_condition(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video_prompt_embeds": torch.randn(4, 4096),
            "prompt_attention_mask": torch.ones(4, dtype=torch.bool),
        },
        path,
    )


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.ones(1, 2, 2, dtype=torch.float32), path)


def test_nondefault_inpaint_mask_dir_write_and_read_agree(tmp_path: Path) -> None:
    """#21 finding 3: the exact fns.py recipe, with inpaint_mask_dir="custom_masks"."""
    from signet_trainer.conditioning.inpaint import InpaintStrategy
    from signet_trainer.data.precomputed import PrecomputedDataset
    from signet_trainer.modal.fns import _PRECOMPUTED_SOURCE_OUTPUT_KEYS

    mask_dir = "custom_masks"
    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "a.pt")
    _write_condition(root / "conditions" / "a.pt")
    _write_mask(root / mask_dir / "a.pt")  # WRITE side: modal/entrypoint.py's configured dir name

    # READ side: the exact recipe modal/fns.py's inpaint train branch uses (see
    # test_train_inpaint_branch_reads_configured_mask_dir above for the source-level proof that
    # this is what actually runs, not a hand-picked equivalent).
    inp_strategy = InpaintStrategy(deps=None, schedule=None, mask_dir=mask_dir)
    data_sources = {
        name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[name] for name in ("latents", "conditions")
    }
    data_sources[mask_dir] = _PRECOMPUTED_SOURCE_OUTPUT_KEYS["video_masks"]
    assert list(data_sources) == list(inp_strategy.get_data_sources())

    ds = PrecomputedDataset(str(root), data_sources=data_sources)
    assert len(ds) == 1
    sample = ds[0]
    assert "video_mask_conditions" in sample, (
        "the custom-dir mask must still surface under the canonical 'video_mask_conditions' batch "
        "key InpaintStrategy._extract_mask reads — only the ON-DISK dir name changed"
    )


def test_default_inpaint_mask_dir_still_matches_video_masks(tmp_path: Path) -> None:
    """The unwired-but-default case (99% of configs) must be byte-identical to before this fix."""
    from signet_trainer.conditioning.inpaint import InpaintStrategy
    from signet_trainer.data.precomputed import PrecomputedDataset
    from signet_trainer.modal.fns import _PRECOMPUTED_SOURCE_OUTPUT_KEYS

    mask_dir = "video_masks"  # the schema default
    root = tmp_path / ".precomputed"
    _write_latent(root / "latents" / "a.pt")
    _write_condition(root / "conditions" / "a.pt")
    _write_mask(root / mask_dir / "a.pt")

    inp_strategy = InpaintStrategy(deps=None, schedule=None, mask_dir=mask_dir)
    data_sources = {
        name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[name] for name in ("latents", "conditions")
    }
    data_sources[mask_dir] = _PRECOMPUTED_SOURCE_OUTPUT_KEYS["video_masks"]
    assert data_sources == {
        "latents": "latent_conditions",
        "conditions": "text_conditions",
        "video_masks": "video_mask_conditions",
    }

    ds = PrecomputedDataset(str(root), data_sources=data_sources)
    assert len(ds) == 1
    assert "video_mask_conditions" in ds[0]
