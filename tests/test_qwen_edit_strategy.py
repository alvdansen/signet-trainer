"""Family #3 (``qwen_edit``) conditioning strategy — the packing contract, on CPU.

Written by the VERIFIER pass. Every expected number is re-derived here from the geometry
(1024x1024 pixels -> /8 VAE -> 128x128 latent -> /2 patch -> 64x64 = 4096 rows; row width
16 channels x 2 x 2 = 64) rather than copied from a build report.

The four contract facts that make this family different from LTX and H3, each pinned below:

* **The control block is a SUFFIX**, not a prefix (``qwen_image_edit_plus.py:314-316`` concatenates
  ``[target, control]`` and reads back a PREFIX at ``:346``). ``ref_seq_len`` therefore stays
  ``None`` — every consumer of it slices ``[:, ref_seq_len:, :]``, which here would keep the
  controls and drop the target.
* **The text mask stays int64**, never an additive ``-inf`` float mask, because
  ``txt_seq_lens = mask.sum(dim=1)`` (``:326-329``) and ``-inf`` sums to garbage instead of crashing.
* **A missing control slot is an ERROR, not a shorter list.** ai-toolkit appends only inside
  ``if os.path.exists(...)`` (``dataloader_mixins.py:984-985``), so a stem missing from ``dirB``
  slides ``dirC`` into slot 1 and silently re-addresses every later ``ctrl_img_N``.
* **A blank fill is not free.** It encodes to a real VAE latent and costs its rows.

CPU only, float32, zero GPU, zero downloads.
"""

from __future__ import annotations

import re

import pytest
import torch

from signet_trainer.conditioning.qwen_edit import (
    QWEN_EDIT_BLANK_FILLS,
    QwenEditStrategy,
    resolve_control_slots,
)
from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_MAX_CONTROL_SLOTS,
    QWEN_EDIT_PATCH_DIM,
    qwen_edit_rows_of,
)
from signet_trainer.conditioning.strategy import ModelInputs

# Re-derived, not imported: 16 latent channels x a 2x2 pack.
EXPECTED_ROW_WIDTH = 16 * 2 * 2
# 1024 px /8 = 128 latent /2 patch = 64; 64 x 64 = 4096 rows.
EXPECTED_ROWS_1024 = (1024 // 8 // 2) ** 2
TEXT_TOKENS = 27


def _rows(n: int = EXPECTED_ROWS_1024) -> torch.Tensor:
    return torch.zeros(1, n, EXPECTED_ROW_WIDTH)


def _slot(index: int, stem: str = "img001", **over) -> dict:
    entry = {
        "slot": index,
        "stem": stem,
        "rows": _rows(),
        "latent_hw": (128, 128),
        "path": f"ctrl{index}/{stem}.png",
    }
    entry.update(over)
    return entry


def _batch(*, slots: int = 3, stem: str = "img001", **over) -> dict:
    batch = {
        "stem": stem,
        "sigma": torch.tensor([0.25]),
        "noise": torch.ones(1, EXPECTED_ROWS_1024, EXPECTED_ROW_WIDTH),
        "qwen_edit_latents": {"rows": _rows(), "latent_hw": (128, 128)},
        "qwen_edit_conditions": {
            "prompt_embeds": torch.zeros(1, TEXT_TOKENS, 3584),
            "prompt_embeds_mask": torch.ones(1, TEXT_TOKENS, dtype=torch.int64),
        },
        "qwen_edit_control_latents": {
            "controls": [_slot(i, stem) for i in range(slots)]
        },
    }
    batch.update(over)
    return batch


def _strategy(**over) -> QwenEditStrategy:
    kwargs = {"control_slots": 3, "device": "cpu", "dtype": torch.float32}
    kwargs.update(over)
    return QwenEditStrategy(**kwargs)


# ======================================================================================
# Geometry the packing depends on
# ======================================================================================


def test_row_width_and_row_count_match_the_measured_geometry() -> None:
    assert QWEN_EDIT_PATCH_DIM == EXPECTED_ROW_WIDTH == 64
    assert qwen_edit_rows_of(1024, 1024) == EXPECTED_ROWS_1024 == 4096


def test_ltx_seq_len_would_undercount_this_family_fourfold() -> None:
    """Documents why a family-exact arm is required rather than reuse."""
    from signet_trainer.dryrun.shapes import compute_seq_len

    assert compute_seq_len(1024, 1024, 1) * 4 == qwen_edit_rows_of(1024, 1024)


# ======================================================================================
# Happy path
# ======================================================================================


def test_happy_path_packs_target_then_controls() -> None:
    inputs = _strategy().prepare_training_inputs(_batch())
    assert isinstance(inputs, ModelInputs)
    assert inputs.target_seq_len == EXPECTED_ROWS_1024
    assert inputs.control_seq_len == 3 * EXPECTED_ROWS_1024
    assert tuple(inputs.video.latent.shape) == (
        1,
        4 * EXPECTED_ROWS_1024,
        EXPECTED_ROW_WIDTH,
    )
    assert inputs.control_slot_rows == (EXPECTED_ROWS_1024,) * 3


def test_control_block_is_a_suffix_and_ref_seq_len_is_none() -> None:
    """The single most dangerous inherited habit: ``[:, ref_seq_len:, :]`` on this family."""
    inputs = _strategy().prepare_training_inputs(_batch())
    assert inputs.ref_seq_len is None
    # The target occupies the PREFIX, so the loss mask's True run starts at row 0.
    mask = inputs.video_loss_mask[0]
    assert bool(mask[0]) is True
    assert bool(mask[EXPECTED_ROWS_1024]) is False
    assert int(mask.sum()) == EXPECTED_ROWS_1024


def test_img_shapes_open_with_the_target_then_one_entry_per_slot() -> None:
    inputs = _strategy().prepare_training_inputs(_batch())
    assert len(inputs.img_shapes) == 4
    assert all(shape == (1, 64, 64) for shape in inputs.img_shapes)


def test_img_shapes_ordering_is_provable_on_an_asymmetric_geometry() -> None:
    """At 1024^2 every block is 4096 rows and any permutation looks fine — so use 512 target."""
    batch = _batch()
    batch["qwen_edit_latents"] = {"rows": _rows(1024), "latent_hw": (64, 64)}
    batch["noise"] = torch.ones(1, 1024, EXPECTED_ROW_WIDTH)
    inputs = _strategy().prepare_training_inputs(batch)
    assert inputs.img_shapes[0] == (1, 32, 32)
    assert inputs.img_shapes[1:] == ((1, 64, 64), (1, 64, 64), (1, 64, 64))
    assert inputs.target_seq_len == 1024
    assert inputs.control_seq_len == 3 * EXPECTED_ROWS_1024


def test_text_mask_stays_int64_and_drives_txt_seq_lens() -> None:
    batch = _batch()
    batch["qwen_edit_conditions"]["prompt_embeds_mask"] = torch.tensor(
        [[1] * 12 + [0] * (TEXT_TOKENS - 12)], dtype=torch.int64
    )
    inputs = _strategy().prepare_training_inputs(batch)
    assert inputs.video.context_mask.dtype == torch.int64
    assert inputs.txt_seq_lens == (12,)
    assert inputs.transformer_kwargs()["txt_seq_lens"] == [12]


def test_transformer_kwargs_mirror_the_ai_toolkit_call() -> None:
    kwargs = _strategy().prepare_training_inputs(_batch()).transformer_kwargs()
    assert set(kwargs) == {
        "hidden_states",
        "timestep",
        "guidance",
        "encoder_hidden_states",
        "encoder_hidden_states_mask",
        "img_shapes",
        "txt_seq_lens",
        "return_dict",
    }
    assert kwargs["guidance"] is None
    assert kwargs["return_dict"] is False
    # 0-1 sigma, NOT the 0-1000 scale: ai-toolkit divides at the call site.
    assert float(kwargs["timestep"].max()) <= 1.0


def test_flow_match_algebra_is_noise_ward() -> None:
    """sigma = 0 is CLEAN and the velocity target is ``eps - x0`` (qwen_image.py:389-392)."""
    x0 = torch.rand(1, EXPECTED_ROWS_1024, EXPECTED_ROW_WIDTH)
    eps = torch.rand(1, EXPECTED_ROWS_1024, EXPECTED_ROW_WIDTH)
    batch = _batch()
    batch["qwen_edit_latents"] = {"rows": x0, "latent_hw": (128, 128)}
    batch["noise"] = eps
    batch["sigma"] = torch.tensor([0.0])
    inputs = _strategy().prepare_training_inputs(batch)
    packed = inputs.video.latent[:, :EXPECTED_ROWS_1024, :]
    assert torch.allclose(packed, x0, atol=1e-6), "sigma=0 must leave the target clean"


def test_sigma_is_never_defaulted() -> None:
    batch = _batch()
    batch.pop("sigma")
    with pytest.raises((ValueError, KeyError)):
        _strategy().prepare_training_inputs(batch)


# ======================================================================================
# Loss slicing
# ======================================================================================


def test_loss_slices_the_target_prefix() -> None:
    strategy = _strategy()
    inputs = strategy.prepare_training_inputs(_batch())
    total = inputs.target_seq_len + inputs.control_seq_len
    prediction = torch.zeros(1, total, EXPECTED_ROW_WIDTH)
    loss = strategy.compute_loss(inputs, (prediction,))
    assert loss.ndim == 0
    # Read-back form: exactly the target prefix. Must give the identical loss.
    readback = torch.zeros(1, inputs.target_seq_len, EXPECTED_ROW_WIDTH)
    assert torch.allclose(loss, strategy.compute_loss(inputs, readback))


def test_suffix_sliced_output_is_refused() -> None:
    """A control-length output is >= target length and would silently score the wrong rows."""
    strategy = _strategy()
    inputs = strategy.prepare_training_inputs(_batch())
    suffix_only = torch.zeros(1, inputs.control_seq_len, EXPECTED_ROW_WIDTH)
    with pytest.raises((ValueError, RuntimeError)):
        strategy.compute_loss(inputs, suffix_only)


# ======================================================================================
# Blank-pad behaviour
# ======================================================================================


def test_gap_becomes_a_blank_at_its_own_index_not_a_left_shift() -> None:
    """The ai-toolkit slide: a stem missing from dirB must NOT move dirC into slot 1."""
    slots = resolve_control_slots(
        "img001",
        [{"slot": 0, "path": "a/img001.png"}, {"slot": 2, "path": "c/img001.png"}],
        control_slots=3,
        blank_slot_fill="black",
    )
    assert [s.index for s in slots] == [0, 1, 2]
    assert slots[0].path == "a/img001.png"
    assert slots[1].blank is True and slots[1].fill == "black"
    assert slots[2].path == "c/img001.png"


def test_blank_pad_needs_a_blank_latent_fn_and_refuses_without_one() -> None:
    """A black image's VAE latent is not zeros; a silent zero-fill is a wrong training signal."""
    batch = _batch(slots=3)
    batch["qwen_edit_control_latents"]["controls"] = [_slot(0), _slot(2)]
    with pytest.raises(ValueError) as exc:
        _strategy().prepare_training_inputs(batch)
    assert "blank_latent_fn" in str(exc.value)


def test_blank_fill_costs_rows_against_the_budget() -> None:
    """With the seam injected, a padded slot is present in the sequence and in img_shapes."""
    calls: list[tuple[int, str]] = []

    def blank_latent_fn(*, slot_index: int, fill: str):
        calls.append((slot_index, fill))
        return {"rows": _rows(), "latent_hw": (128, 128)}

    batch = _batch(slots=3)
    batch["qwen_edit_control_latents"]["controls"] = [_slot(0), _slot(2)]
    inputs = _strategy(blank_latent_fn=blank_latent_fn).prepare_training_inputs(batch)
    assert calls == [(1, "black")]
    assert inputs.control_seq_len == 3 * EXPECTED_ROWS_1024
    assert len(inputs.img_shapes) == 4


def test_absent_control_source_is_not_silently_a_zero_slot_sample() -> None:
    batch = _batch()
    batch.pop("qwen_edit_control_latents")
    with pytest.raises(ValueError) as exc:
        _strategy().prepare_training_inputs(batch)
    assert "blank_latent_fn" in str(exc.value)


def test_blank_fills_tuple_matches_the_schema_literal() -> None:
    """Pinned by TEST rather than by cross-import (the config layer must not import torch)."""
    from signet_trainer.config.schema import QwenEditConfig

    annotation = repr(QwenEditConfig.model_fields["blank_slot_fill"].annotation)
    for fill in QWEN_EDIT_BLANK_FILLS:
        assert f"'{fill}'" in annotation
    assert len(QWEN_EDIT_BLANK_FILLS) == len(re.findall(r"'[a-z]+'", annotation))


# ======================================================================================
# Slot-plan refusals
# ======================================================================================


def test_short_payload_without_slot_indices_is_refused() -> None:
    with pytest.raises(ValueError) as exc:
        resolve_control_slots(
            "img001",
            [{"path": "a/img001.png"}, {"path": "c/img001.png"}],
            control_slots=3,
            blank_slot_fill="black",
        )
    assert "slot" in str(exc.value)


def test_complete_positional_payload_is_accepted() -> None:
    slots = resolve_control_slots(
        "img001",
        [{"path": f"{d}/img001.png"} for d in "abc"],
        control_slots=3,
        blank_slot_fill="black",
    )
    assert [s.path for s in slots] == ["a/img001.png", "b/img001.png", "c/img001.png"]


def test_mixed_declared_and_undeclared_indices_are_refused() -> None:
    with pytest.raises(ValueError):
        resolve_control_slots(
            "img001",
            [{"slot": 0, "path": "a.png"}, {"path": "b.png"}],
            control_slots=3,
            blank_slot_fill="black",
        )


def test_duplicate_and_out_of_range_slots_are_refused() -> None:
    with pytest.raises(ValueError):
        resolve_control_slots(
            "img001",
            [{"slot": 1, "path": "a.png"}, {"slot": 1, "path": "b.png"}],
            control_slots=3,
            blank_slot_fill="black",
        )
    with pytest.raises(ValueError):
        resolve_control_slots(
            "img001", [{"slot": 5, "path": "a.png"}], control_slots=3, blank_slot_fill="black"
        )


def test_stem_mismatch_is_refused() -> None:
    with pytest.raises(ValueError) as exc:
        resolve_control_slots(
            "img001",
            [{"slot": 0, "stem": "img002", "path": "a.png"}],
            control_slots=3,
            blank_slot_fill="black",
        )
    assert "stem" in str(exc.value)


@pytest.mark.parametrize("slots", [0, QWEN_EDIT_MAX_CONTROL_SLOTS + 1])
def test_slot_count_bounds_are_enforced_in_the_strategy_too(slots: int) -> None:
    with pytest.raises(ValueError):
        QwenEditStrategy(control_slots=slots, device="cpu")


def test_invalid_blank_fill_is_refused() -> None:
    with pytest.raises(ValueError):
        QwenEditStrategy(control_slots=3, blank_slot_fill="chartreuse", device="cpu")


# ======================================================================================
# Payload-form refusals
# ======================================================================================


def test_rows_form_requires_latent_hw_and_cross_checks_it() -> None:
    batch = _batch()
    batch["qwen_edit_latents"] = {"rows": _rows()}
    with pytest.raises(ValueError) as exc:
        _strategy().prepare_training_inputs(batch)
    assert "latent_hw" in str(exc.value)

    batch["qwen_edit_latents"] = {"rows": _rows(), "latent_hw": (64, 64)}
    with pytest.raises(ValueError):
        _strategy().prepare_training_inputs(batch)


def test_latent_form_without_pack_fn_names_the_element_order_reason() -> None:
    batch = _batch()
    batch["qwen_edit_latents"] = torch.zeros(16, 1, 128, 128)
    with pytest.raises(ValueError) as exc:
        _strategy().prepare_training_inputs(batch)
    assert "pack_fn" in str(exc.value)


def test_missing_target_source_names_every_key_it_looked_for() -> None:
    batch = _batch()
    batch.pop("qwen_edit_latents")
    with pytest.raises(ValueError) as exc:
        _strategy().prepare_training_inputs(batch)
    assert "qwen_edit_latent" in str(exc.value)


def test_missing_stem_is_refused_rather_than_inferred() -> None:
    batch = _batch()
    batch.pop("stem")
    with pytest.raises(ValueError) as exc:
        _strategy().prepare_training_inputs(batch)
    assert "stem" in str(exc.value)


def test_row_ceiling_refuses_an_over_budget_pack() -> None:
    with pytest.raises(ValueError) as exc:
        _strategy(max_packed_rows=8000).prepare_training_inputs(_batch())
    assert "16384" in str(exc.value)


# ======================================================================================
# Import purity — the free-dry-run property
# ======================================================================================


def test_module_imports_without_modal_or_ltx_core() -> None:
    """A fresh interpreter, so a module already imported by an earlier test cannot mask a pull.

    ``PYTHONPATH`` is set explicitly rather than inherited: pytest puts ``src`` on ``sys.path`` via
    ``pyproject.toml``'s ``pythonpath`` setting, which does NOT reach a subprocess, and a probe that
    silently fails to import proves nothing.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    probe = (
        "import sys, json\n"
        "import signet_trainer.conditioning.qwen_edit as m\n"
        "import signet_trainer.conditioning.qwen_edit_geometry as g\n"
        "bad = sorted({k.split('.')[0] for k in sys.modules "
        "if k.split('.')[0] in ('modal', 'ltx_core', 'diffusers', 'peft')})\n"
        "print(json.dumps({'bad': bad, 'ok': m.QwenEditStrategy is not None "
        "and g.QWEN_EDIT_PATCH_DIM == 64}))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(repo / "src"), "PYTHONUTF8": "1"}
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        cwd=repo,
    )
    snapshot = json.loads(out.stdout.strip().splitlines()[-1])
    assert snapshot["ok"] is True, "the purity probe never actually imported the module"
    assert snapshot["bad"] == [], snapshot["bad"]


# ==================================================================================================
# The batch shape PrecomputedDataset ACTUALLY produces (2026-08-08 live-run defect).
# ==================================================================================================


def test_the_stem_is_read_from_the_nested_precomputed_payload() -> None:
    """``_target_stem`` must find the stem where PrecomputedDataset puts it: NESTED.

    ``PrecomputedDataset.__getitem__`` stores each source's whole payload dict under that source's
    key (``result[output_key] = data``, data/precomputed.py:288). prep writes ``payload["stem"]``
    (prep/qwen_edit_encode.py:923), so at training time the stem lives at
    ``batch["qwen_edit_latents"]["stem"]`` — never at the top level. The first live train dispatch
    died here after the arch gate had already passed and the adapter had already been injected
    (1680 tensors, 387,072,000 params), which is the most expensive place this could have surfaced.
    """
    from signet_trainer.conditioning.qwen_edit import _target_stem

    # BOTH spellings must work. fns.py:5828 keys the batch by the OUTPUT KEY
    # (qwen_edit_latent_conditions) while the on-disk directory is qwen_edit_latents; the
    # module's QWEN_EDIT_*_BATCH_KEYS tuples enumerate both, and this reads from those rather
    # than from a hand-written list that would drift from them.
    assert _target_stem({"qwen_edit_latents": {"stem": "img_3404"}}) == "img_3404"
    assert _target_stem({"qwen_edit_latent_conditions": {"stem": "img_3404"}}) == "img_3404"


def test_a_top_level_stem_still_wins() -> None:
    """Back-compat: an explicit top-level stem takes precedence over the nested one."""
    from signet_trainer.conditioning.qwen_edit import _target_stem

    batch = {"stem": "explicit", "qwen_edit_latents": {"stem": "nested"}}
    assert _target_stem(batch) == "explicit"


def test_the_target_source_is_preferred_over_the_control_source() -> None:
    """Read the TARGET's stem, not the control's.

    The stem exists to verify that a sample's controls belong to that sample. Preferring the
    control payload's copy would compare a value against itself and pass unconditionally — the
    check would still run, still be green, and mean nothing.
    """
    from signet_trainer.conditioning.qwen_edit import _target_stem

    batch = {
        "qwen_edit_latent_conditions": {"stem": "the_target"},
        "qwen_edit_control_latent_conditions": {"stem": "a_different_sample"},
    }
    assert _target_stem(batch) == "the_target"


def test_a_batch_with_no_stem_anywhere_is_still_refused() -> None:
    """The refusal must survive: a missing stem makes the 1:1 check unfalsifiable."""
    import pytest as _pytest

    from signet_trainer.conditioning.qwen_edit import _target_stem

    with _pytest.raises(ValueError, match="matched to targets BY STEM"):
        _target_stem({"qwen_edit_latent_conditions": {"latents": None}})


def test_the_stem_lookup_is_single_sourced_from_the_batch_key_tuples() -> None:
    """``_target_stem`` must read the module's key tuples, not a hand-written list.

    The live failure this pins: the batch is keyed by the OUTPUT KEY, not the directory name
    (``fns.py:5828`` maps ``{dir_name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[dir_name]}``). A hardcoded
    ``"qwen_edit_latents"`` looked right, matched the on-disk directory, and found nothing — twice,
    on two separate metered dispatches, each after the arch gate had passed and the adapter had
    been injected. The tuples already enumerated both spellings.
    """
    import inspect

    from signet_trainer.conditioning import qwen_edit

    src = inspect.getsource(qwen_edit._target_stem)
    assert "QWEN_EDIT_TARGET_BATCH_KEYS" in src and "QWEN_EDIT_CONTROL_BATCH_KEYS" in src, (
        "_target_stem must iterate QWEN_EDIT_TARGET_BATCH_KEYS / QWEN_EDIT_CONTROL_BATCH_KEYS so "
        "the accepted batch keys stay single-sourced with the rest of the module"
    )
    # Every spelling the tuples promise actually resolves.
    for key in (*qwen_edit.QWEN_EDIT_TARGET_BATCH_KEYS, *qwen_edit.QWEN_EDIT_CONTROL_BATCH_KEYS):
        assert qwen_edit._target_stem({key: {"stem": "s"}}) == "s", f"{key} not honoured"
