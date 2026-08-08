"""Family #3 (``qwen_edit``) training step — the draw, THE LOSS-WEIGHT HOOK, and a real fwd/bwd.

Four things are proved here, on CPU, with zero GPU / zero downloads / zero Modal:

1. **The timestep draw is ai-toolkit's**, not ``FlowMatchingSchedule``'s: a discrete-uniform index
   over ``linspace(1000, 1, 1000)`` with an EXCLUSIVE upper bound, so ``t = 1`` is unreachable and
   ``sigma = 1.0`` is reachable.
2. **The loss weight is the bsmntw BELL CURVE**, reproduced from
   ``custom_flowmatch_sampler.py:31-42``, and NOT the dead 1000-entry ``default_weighing_scheme``
   table that ``timestep_type: "weighted"`` appears to select (``:65-74`` overwrites it with an
   ``if``, not an ``elif``). Pinned against values recomputed here from the formula.
3. **The hook does not perturb ltx or h3** — structurally (no sibling module references it),
   by import closure (the LTX/H3 path never imports this module), and behaviourally (both
   siblings' draws are bit-identical with and without this module imported).
4. **It trains.** A tiny synthetic dual-stream transformer whose leaves are named exactly like the
   14 real ones, PEFT LoRA at rank 42, driven through the REAL ``step_fn`` for 40 steps, with the
   loss going down.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from signet_trainer.conditioning.qwen_edit import QwenEditStrategy
from signet_trainer.conditioning.qwen_edit_geometry import QWEN_EDIT_PATCH_DIM
from signet_trainer.config.validators import (
    QWEN_EDIT_LORA_LEAVES,
    QWEN_EDIT_LORA_TARGET_REGEX,
)
from signet_trainer.train.family_hooks import LOOP_HOOKS_BY_FAMILY, LoopHooks, build_loop_hooks
from signet_trainer.train.qwen_edit_step import (
    QWEN_EDIT_AUTOCAST,
    QWEN_EDIT_MAX_TIMESTEP_INDEX,
    QWEN_EDIT_MIN_TIMESTEP_INDEX,
    QWEN_EDIT_NUM_TRAIN_TIMESTEPS,
    QwenEditTimestepDraw,
    build_qwen_edit_step_fn,
    qwen_edit_collate_fn,
    qwen_edit_timestep_draw,
    qwen_edit_timestep_grid,
    qwen_edit_timestep_weights,
    qwen_edit_to_device,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tiny-model geometry. The row WIDTH is not a choice — 64 is img_in's input width on the live
# checkpoint — so it is imported. Everything else is small on purpose.
DIM = 64
TEXT_DIM = 48
N_BLOCKS = 2
LATENT_HW = (8, 8)  # -> a 4x4 packed grid -> 16 rows per image block
ROWS = (LATENT_HW[0] // 2) * (LATENT_HW[1] // 2)
TEXT_TOKENS = 7
LORA_RANK = 42


# ======================================================================================
# 1. The grid and the draw — ai-toolkit's, not FlowMatchingSchedule's
# ======================================================================================


def test_grid_is_the_descending_linear_schedule() -> None:
    grid = qwen_edit_timestep_grid()
    assert grid.shape == (QWEN_EDIT_NUM_TRAIN_TIMESTEPS,)
    assert float(grid[0]) == 1000.0
    assert float(grid[-1]) == 1.0
    # At 1000 entries the grid IS ``1000 - index`` exactly; the module builds it rather than
    # asserting the identity, so pin the identity here instead.
    index = torch.arange(QWEN_EDIT_NUM_TRAIN_TIMESTEPS, dtype=torch.float32)
    assert torch.equal(grid, 1000.0 - index)


def test_upper_bound_is_exclusive_so_t_equals_one_is_unreachable() -> None:
    """``torch.randint``'s bound is exclusive (BaseSDTrainProcess.py:1282-1287) — reproduced."""
    rng = np.random.default_rng(0)
    draws = [qwen_edit_timestep_draw(rng) for _ in range(20_000)]
    indices = {d.index for d in draws}
    assert max(indices) == QWEN_EDIT_MAX_TIMESTEP_INDEX - 1 == 998
    assert min(indices) == QWEN_EDIT_MIN_TIMESTEP_INDEX == 0
    assert QWEN_EDIT_MAX_TIMESTEP_INDEX not in indices
    assert 1.0 not in {d.timestep for d in draws}  # t = 1 never drawn
    assert 1000.0 in {d.timestep for d in draws}  # sigma = 1.0 IS reachable


def test_sigma_is_the_timestep_over_one_thousand() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        draw = qwen_edit_timestep_draw(rng)
        assert draw.sigma == pytest.approx(draw.timestep / 1000.0)
        assert 0.0 < draw.sigma <= 1.0


def test_draw_is_uniform_over_the_index_grid_not_logit_normal() -> None:
    """The whole DIVERGE: FlowMatchingSchedule would pile samples around a shifted centre."""
    rng = np.random.default_rng(11)
    indices = np.array([qwen_edit_timestep_draw(rng).index for _ in range(60_000)])
    # A uniform draw over 0..998 has mean 499 and puts ~1/3 of its mass in each outer third.
    assert indices.mean() == pytest.approx(499.0, abs=6.0)
    lo = float((indices < 333).mean())
    hi = float((indices >= 666).mean())
    assert lo == pytest.approx(1 / 3, abs=0.02)
    assert hi == pytest.approx(1 / 3, abs=0.02)

    from signet_trainer.train.flow_match import FlowMatchingSchedule

    logit_normal = FlowMatchingSchedule(uniform_prob=0.0).sample_timesteps(
        60_000, 4096, np.random.default_rng(11)
    )
    # The sibling sampler is emphatically NOT uniform — it is what this family diverges from.
    assert float((logit_normal < 1 / 3).mean()) < 0.10


def test_bad_bounds_are_refused() -> None:
    with pytest.raises(ValueError, match="invalid timestep index bounds"):
        qwen_edit_timestep_draw(np.random.default_rng(0), min_index=5, max_index=5)
    with pytest.raises(ValueError, match="invalid timestep index bounds"):
        qwen_edit_timestep_draw(np.random.default_rng(0), max_index=1001)


# ======================================================================================
# 2. THE LOSS-WEIGHT HOOK — the bell curve, not the dead table
# ======================================================================================


def test_bell_curve_matches_the_upstream_formula_recomputed_here() -> None:
    """Re-derived from ``custom_flowmatch_sampler.py:31-42``, not copied from a report."""
    n = 1000
    x = torch.arange(n, dtype=torch.float32)
    y = torch.exp(-2 * ((x - n / 2) / n) ** 2)
    y_shifted = y - y.min()
    expected = y_shifted * (n / y_shifted.sum())

    weights = qwen_edit_timestep_weights()
    assert torch.allclose(weights, expected, atol=0, rtol=0)
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    assert float(weights[0]) == pytest.approx(0.000000, abs=1e-6)
    assert float(weights[250]) == pytest.approx(1.107882, abs=1e-5)
    assert float(weights[500]) == pytest.approx(1.579605, abs=1e-5)
    assert float(weights[999]) == pytest.approx(0.004870, abs=1e-5)


def test_both_schedule_ends_are_de_weighted_and_the_middle_is_boosted() -> None:
    w = qwen_edit_timestep_weights()
    assert float(w[0]) < float(w[999]) < float(w[250]) < float(w[500])
    assert int(w.argmax()) == 500


def test_weight_and_sigma_come_from_the_same_index() -> None:
    weights = qwen_edit_timestep_weights()
    rng = np.random.default_rng(3)
    for _ in range(500):
        draw = qwen_edit_timestep_draw(rng)
        assert draw.loss_weight == pytest.approx(float(weights[draw.index]))
        assert draw.sigma == pytest.approx((1000.0 - draw.index) / 1000.0)


def test_weighting_off_pins_the_weight_at_one() -> None:
    rng = np.random.default_rng(5)
    assert all(
        qwen_edit_timestep_draw(rng, timestep_weighting=False).loss_weight == 1.0
        for _ in range(100)
    )


def test_the_dead_thousand_entry_table_is_not_shipped() -> None:
    """``default_weighing_scheme`` is dead upstream; nothing here may reproduce it."""
    source = (
        REPO_ROOT / "src" / "signet_trainer" / "train" / "qwen_edit_step.py"
    ).read_text(encoding="utf-8")
    # It is NAMED (the docstring records why it is dead) but never imported or transcribed.
    assert "default_weighing_scheme" in source, "the reason must stay recorded"
    assert "import" not in source.split("default_weighing_scheme")[0].split("\n")[-1]
    # No 1000-entry literal anywhere: the largest float list in the file is nothing like a table.
    numeric_literals = re.findall(r"\d+\.\d+", source)
    assert len(numeric_literals) < 40, f"looks like a transcribed table: {len(numeric_literals)}"


# ======================================================================================
# 3. The hook does not perturb ltx or h3
# ======================================================================================

_SIBLING_MODULES = (
    "train/step.py",
    "train/h3_step.py",
    "train/flow_match.py",
    "train/loop.py",
)


@pytest.mark.parametrize("relpath", _SIBLING_MODULES)
def test_no_sibling_module_references_the_qwen_weight_hook(relpath: str) -> None:
    source = (REPO_ROOT / "src" / "signet_trainer" / relpath).read_text(encoding="utf-8")
    for symbol in (
        "qwen_edit_step",
        "qwen_edit_timestep_weights",
        "qwen_edit_timestep_draw",
        "timestep_weight",
        "loss_weight",
    ):
        assert symbol not in source, f"{relpath} references {symbol}"


def _run(script: str) -> str:
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONIOENCODING": "utf-8"})
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_the_ltx_and_h3_paths_never_import_this_module() -> None:
    """Import closure: the family seam must not drag family #3 into families #1/#2."""
    out = _run(
        "import sys\n"
        "import signet_trainer.train.loop  # noqa\n"
        "import signet_trainer.train.step  # noqa\n"
        "import signet_trainer.train.h3_step  # noqa\n"
        "import signet_trainer.train.flow_match  # noqa\n"
        "assert 'signet_trainer.train.qwen_edit_step' not in sys.modules, 'qwen leaked'\n"
        "assert 'signet_trainer.conditioning.qwen_edit' not in sys.modules, 'strategy leaked'\n"
        "print('clean')\n"
    )
    assert out == "clean"


def test_sibling_draws_are_bit_identical_with_and_without_this_module_imported() -> None:
    """Behavioural non-perturbation: same numbers in a process that never sees family #3."""
    probe = (
        "import numpy as np\n"
        "{maybe_import}"
        "from signet_trainer.train.flow_match import FlowMatchingSchedule\n"
        "from signet_trainer.train.h3_step import h3_draw_timesteps\n"
        "t = FlowMatchingSchedule().sample_timesteps(6, 4096, np.random.default_rng(42))\n"
        "h = h3_draw_timesteps(np.random.default_rng(42))\n"
        "print(repr([float(x) for x in t]), repr(h))\n"
    )
    without = _run(probe.format(maybe_import=""))
    with_qwen = _run(
        probe.format(
            maybe_import="import signet_trainer.train.qwen_edit_step  # noqa\n"
            "import signet_trainer.train.family_hooks  # noqa\n"
        )
    )
    assert without == with_qwen
    assert without  # non-empty, i.e. the probe really ran


def test_this_module_stays_in_the_confined_import_tier() -> None:
    out = _run(
        "import sys\n"
        "import signet_trainer.train.qwen_edit_step as m\n"
        "for heavy in ('modal', 'diffusers', 'peft', 'bitsandbytes', 'ltx_core'):\n"
        "    assert heavy not in sys.modules, heavy + ' leaked'\n"
        "print(m.QWEN_EDIT_NUM_TRAIN_TIMESTEPS, m.QWEN_EDIT_MAX_TIMESTEP_INDEX, m.QWEN_EDIT_AUTOCAST)\n"
    )
    assert out == "1000 999 False"


def test_autocast_verdict_is_recorded_as_false() -> None:
    """A per-family decision (train/step.py:273 wraps LTX; fns.py:4132-4139 refuses for H3)."""
    assert QWEN_EDIT_AUTOCAST is False
    source = (
        REPO_ROOT / "src" / "signet_trainer" / "train" / "qwen_edit_step.py"
    ).read_text(encoding="utf-8")
    # The docstring NAMES autocast (that is where the verdict is recorded); what must be absent is
    # the call form. ``with torch.amp.autocast("cuda", ...)`` is what train/step.py:273 does.
    assert "autocast(" not in source
    assert "with torch." not in source


# ======================================================================================
# 4. A tiny synthetic transformer — leaves named exactly like the real 14
# ======================================================================================


class _TinyAttn(nn.Module):
    """The eight attention leaves, spelled the way the live checkpoint spells them."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])  # -> attn.to_out.0
        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.to_add_out = nn.Linear(dim, dim)


class _Proj(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)


class _TinyMlp(nn.Module):
    """``net.0.proj`` and ``net.2`` — the diffusers ``FeedForward`` spelling."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([_Proj(dim), nn.Identity(), nn.Linear(dim, dim)])


class _TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = _TinyAttn(dim)
        self.img_mlp = _TinyMlp(dim)
        self.txt_mlp = _TinyMlp(dim)
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(1, dim))  # -> img_mod.1
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(1, dim))  # -> txt_mod.1


class TinyQwenEdit(nn.Module):
    """A 2-block dual-stream MMDiT stand-in with the REAL kwarg signature and the REAL leaf names.

    Not a mock: it runs actual joint attention over ``cat([text, image])`` and recovers the image
    half the way ``transformer_qwenimage.py:351`` does, so every one of the 14 LoRA leaves carries
    gradient. It also enforces the tensor contract's hard invariant —
    ``hidden_states.shape[1] == sum(f*h*w for f, h, w in img_shapes[0])`` — which in the real model
    is enforced only implicitly, by RoPE producing that many ``vid_freqs`` rows.
    """

    def __init__(self, dim: int = DIM, blocks: int = N_BLOCKS, text_dim: int = TEXT_DIM) -> None:
        super().__init__()
        self.img_in = nn.Linear(QWEN_EDIT_PATCH_DIM, dim)
        self.txt_in = nn.Linear(text_dim, dim)
        self.transformer_blocks = nn.ModuleList([_TinyBlock(dim) for _ in range(blocks)])
        self.proj_out = nn.Linear(dim, QWEN_EDIT_PATCH_DIM)

    def forward(  # noqa: PLR0913 — this IS the transformer's kwarg set
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        img_shapes: list | None = None,
        txt_seq_lens: list | None = None,
        return_dict: bool = True,
    ):
        assert guidance is None, "config.guidance_embeds is False for this checkpoint"
        assert return_dict is False, "ai-toolkit calls with return_dict=False and indexes [0]"
        assert encoder_hidden_states_mask.dtype == torch.int64, "the mask is INT (DIVERGE #2)"
        priced = sum(f * h * w for f, h, w in img_shapes[0])
        assert priced == hidden_states.shape[1], (priced, hidden_states.shape)
        assert max(txt_seq_lens) <= encoder_hidden_states.shape[1]

        img = self.img_in(hidden_states)
        txt = self.txt_in(encoder_hidden_states) * encoder_hidden_states_mask.unsqueeze(-1).to(
            encoder_hidden_states.dtype
        )
        t = timestep.reshape(-1, 1, 1).to(img.dtype)
        n_txt = txt.shape[1]

        for block in self.transformer_blocks:
            img_scale = block.img_mod(t.expand(-1, img.shape[1], 1))
            txt_scale = block.txt_mod(t.expand(-1, n_txt, 1))
            q = torch.cat([block.attn.add_q_proj(txt), block.attn.to_q(img)], dim=1)
            k = torch.cat([block.attn.add_k_proj(txt), block.attn.to_k(img)], dim=1)
            v = torch.cat([block.attn.add_v_proj(txt), block.attn.to_v(img)], dim=1)
            joint = F.scaled_dot_product_attention(q, k, v)
            txt_attn, img_attn = joint[:, :n_txt], joint[:, n_txt:]
            img = img + block.attn.to_out[0](img_attn) * (1.0 + img_scale)
            txt = txt + block.attn.to_add_out(txt_attn) * (1.0 + txt_scale)
            img = img + block.img_mlp.net[2](F.gelu(block.img_mlp.net[0].proj(img)))
            txt = txt + block.txt_mlp.net[2](F.gelu(block.txt_mlp.net[0].proj(txt)))

        return (self.proj_out(img),)


def _fixed_batch(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(1, ROWS, QWEN_EDIT_PATCH_DIM, generator=g)
    control = torch.randn(1, ROWS, QWEN_EDIT_PATCH_DIM, generator=g)
    return {
        "stem": "smoke001",
        "qwen_edit_latents": {"rows": target, "latent_hw": LATENT_HW},
        "qwen_edit_conditions": {
            "prompt_embeds": torch.randn(1, TEXT_TOKENS, TEXT_DIM, generator=g),
            "prompt_embeds_mask": torch.ones(1, TEXT_TOKENS, dtype=torch.int64),
        },
        "qwen_edit_control_latents": {
            "controls": [
                {
                    "slot": 0,
                    "stem": "smoke001",
                    "rows": control,
                    "latent_hw": LATENT_HW,
                    "path": "ctrl0/smoke001.png",
                }
            ]
        },
    }


def _strategy() -> QwenEditStrategy:
    return QwenEditStrategy(control_slots=1, device="cpu", dtype=torch.float32)


def _lora_model(seed: int = 42) -> nn.Module:
    from signet_trainer.lora.peft import build_lora_config, inject_lora

    torch.manual_seed(seed)
    base = TinyQwenEdit()
    return inject_lora(
        base,
        build_lora_config(
            rank=LORA_RANK, alpha=LORA_RANK, targets=QWEN_EDIT_LORA_TARGET_REGEX
        ),
    )


def test_the_regex_hits_all_fourteen_leaves_in_every_block() -> None:
    from signet_trainer.lora.peft import check_lora_targets_regex

    survey = check_lora_targets_regex(
        TinyQwenEdit(),
        QWEN_EDIT_LORA_TARGET_REGEX,
        collateral_markers=(),
        leaves=list(QWEN_EDIT_LORA_LEAVES),
    )
    assert survey["total"] == len(QWEN_EDIT_LORA_LEAVES) * N_BLOCKS == 28
    assert survey["collateral"] == 0
    assert set(survey["per_leaf"].values()) == {N_BLOCKS}  # never a per-leaf ZERO


def test_lora_injects_at_rank_42_over_the_fourteen_leaves() -> None:
    model = _lora_model()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters — is LoRA applied?"
    assert all("lora_" in n for n, _ in trainable)
    # 28 modules x lora_A + lora_B
    assert len(trainable) == 28 * 2
    a = next(p for n, p in trainable if "lora_A" in n)
    assert a.shape[0] == LORA_RANK


# ======================================================================================
# 5. The step itself
# ======================================================================================


def test_step_returns_a_zero_dim_float32_scalar() -> None:
    step_fn = build_qwen_edit_step_fn(_strategy(), seed=42)
    loss = step_fn(
        _lora_model(),
        _fixed_batch(),
        None,
        np.random.default_rng(42),
        device="cpu",
        dtype=torch.float32,
    )
    assert loss.ndim == 0
    assert loss.dtype == torch.float32
    assert loss.requires_grad


def test_the_weight_is_actually_applied_to_the_loss() -> None:
    """Same seeds -> same draw and same noise, so the two losses differ EXACTLY by the weight."""
    model = _lora_model()
    on_record: list[QwenEditTimestepDraw] = []
    weighted = build_qwen_edit_step_fn(_strategy(), seed=42, record=on_record)(
        model, _fixed_batch(), None, np.random.default_rng(9), device="cpu", dtype=torch.float32
    )
    off_record: list[QwenEditTimestepDraw] = []
    plain = build_qwen_edit_step_fn(
        _strategy(), seed=42, timestep_weighting=False, record=off_record
    )(model, _fixed_batch(), None, np.random.default_rng(9), device="cpu", dtype=torch.float32)

    assert on_record[0].index == off_record[0].index
    assert off_record[0].loss_weight == 1.0
    assert float(weighted) == pytest.approx(
        float(plain) * on_record[0].loss_weight, rel=1e-6
    )


def test_batch_above_one_is_refused_and_names_the_strategy_seam() -> None:
    """The B=1 equivalence of scalar-weighting is an INVARIANT, not an assumption."""
    strategy = _strategy()
    step_fn = build_qwen_edit_step_fn(strategy, seed=42)
    batch = _fixed_batch()
    # Two samples' worth of target rows, which the strategy reads as batch_size 2.
    batch["qwen_edit_latents"] = {
        "rows": torch.randn(2, ROWS, QWEN_EDIT_PATCH_DIM),
        "latent_hw": LATENT_HW,
    }
    with pytest.raises(ValueError, match="leading batch dimension must be 1"):
        step_fn(
            _lora_model(),
            batch,
            None,
            np.random.default_rng(0),
            device="cpu",
            dtype=torch.float32,
        )
    # The step's own refusal message is the one that names where the hook moves to.
    source = (
        REPO_ROOT / "src" / "signet_trainer" / "train" / "qwen_edit_step.py"
    ).read_text(encoding="utf-8")
    assert "realized batch size is" in source
    assert "_masked_velocity_loss" in source
    assert "qwen_edit.py:627-639" in source


def test_a_strategy_loop_device_split_is_refused_by_name() -> None:
    step_fn = build_qwen_edit_step_fn(
        QwenEditStrategy(control_slots=1, device="cuda", dtype=torch.float32), seed=42
    )
    with pytest.raises(RuntimeError, match="strategy/loop device split"):
        step_fn(
            _lora_model(),
            _fixed_batch(),
            None,
            np.random.default_rng(0),
            device="cpu",
            dtype=torch.float32,
        )


def test_a_non_dict_sample_names_the_collate_fn() -> None:
    step_fn = build_qwen_edit_step_fn(_strategy(), seed=42)
    with pytest.raises(TypeError, match="collate_fn=qwen_edit_collate_fn"):
        step_fn(
            _lora_model(),
            [_fixed_batch()],
            None,
            np.random.default_rng(0),
            device="cpu",
            dtype=torch.float32,
        )


def test_missing_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="requires an explicit seed"):
        build_qwen_edit_step_fn(_strategy(), seed=None)


def test_to_device_preserves_integer_masks_and_walks_the_control_list() -> None:
    moved = qwen_edit_to_device(_fixed_batch(), "cpu", torch.bfloat16)
    assert moved["qwen_edit_conditions"]["prompt_embeds"].dtype == torch.bfloat16
    assert moved["qwen_edit_conditions"]["prompt_embeds_mask"].dtype == torch.int64
    assert moved["qwen_edit_control_latents"]["controls"][0]["rows"].dtype == torch.bfloat16
    assert moved["qwen_edit_control_latents"]["controls"][0]["stem"] == "smoke001"


def test_collate_fn_is_the_identity_on_a_single_sample() -> None:
    sample = _fixed_batch()
    assert qwen_edit_collate_fn([sample]) is sample
    with pytest.raises(ValueError, match="expected exactly one sample"):
        qwen_edit_collate_fn([sample, sample])


# ======================================================================================
# 6. The family registry
# ======================================================================================


def test_ltx_resolves_to_the_loops_own_defaults() -> None:
    assert build_loop_hooks("ltx") == LoopHooks(step_fn=None, collate_fn=None)
    with pytest.raises(ValueError, match="takes no builder kwargs"):
        build_loop_hooks("ltx", strategy=object())


def test_h3_is_a_named_stub_pointing_at_its_real_home() -> None:
    with pytest.raises(NotImplementedError, match=r"modal/fns.py::h3_train"):
        build_loop_hooks("h3")


def test_unknown_family_raises_rather_than_falling_back_to_ltx() -> None:
    with pytest.raises(ValueError, match="unknown model family"):
        build_loop_hooks("wan")
    assert sorted(LOOP_HOOKS_BY_FAMILY) == ["h3", "ltx", "qwen_edit"]


def test_qwen_edit_resolves_to_the_real_pair() -> None:
    hooks = build_loop_hooks("qwen_edit", strategy=_strategy(), seed=42)
    assert callable(hooks.step_fn)
    assert hooks.collate_fn is qwen_edit_collate_fn
    with pytest.raises(ValueError, match=r"requires \['seed'\]"):
        build_loop_hooks("qwen_edit", strategy=_strategy())


def test_the_hooks_match_the_loops_own_step_fn_signature() -> None:
    """The seam is positional-then-keyword-only; a drift here fails at loop.py:351, mid-run."""
    import inspect

    hooks = build_loop_hooks("qwen_edit", strategy=_strategy(), seed=42)
    params = inspect.signature(hooks.step_fn).parameters
    positional = [n for n, p in params.items() if p.kind is p.POSITIONAL_OR_KEYWORD]
    kwonly = [n for n, p in params.items() if p.kind is p.KEYWORD_ONLY]
    assert len(positional) == 4  # model, batch, schedule, rng
    assert kwonly == ["device", "dtype"]


# ======================================================================================
# 7. THE PROOF: a real forward + backward that learns
# ======================================================================================


def _train_smoke(steps: int = 200, lr: float = 5e-3, *, verbose: bool = False) -> list[float]:
    """Drive the REAL step_fn exactly as ``loop.py:351-353`` does; return the probe-loss trace."""
    torch.manual_seed(0)
    model = _lora_model()
    strategy = _strategy()
    record: list[QwenEditTimestepDraw] = []
    step_fn = build_qwen_edit_step_fn(strategy, seed=42, record=record)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    rng = np.random.default_rng(42)
    batch = _fixed_batch()

    # A FIXED probe: explicit sigma + explicit noise, so prepare_training_inputs is deterministic
    # and the trace measures learning rather than the timestep lottery.
    probe = _fixed_batch()
    probe["sigma"] = 0.5
    probe["noise"] = torch.randn(1, ROWS, QWEN_EDIT_PATCH_DIM, generator=torch.Generator().manual_seed(1))

    def _probe_loss() -> float:
        with torch.no_grad():
            inputs = strategy.prepare_training_inputs(probe)
            return float(strategy.compute_loss(inputs, model(**inputs.transformer_kwargs())))

    trace = [_probe_loss()]
    if verbose:
        print(
            f"\n  step | idx  |  sigma  | weight  | train loss | probe loss"
            f"\n  -----+------+---------+---------+------------+-----------"
            f"\n     0 |  --  |   --    |   --    |     --     | {trace[0]:10.6f}"
        )
    for step in range(1, steps + 1):
        loss = step_fn(model, batch, None, rng, device="cpu", dtype=torch.float32)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        trace.append(_probe_loss())
        if verbose and (step % 20 == 0 or step == 1):
            d = record[-1]
            print(
                f"  {step:4d} | {d.index:4d} | {d.sigma:7.3f} | {d.loss_weight:7.4f} | "
                f"{float(loss):10.6f} | {trace[-1]:10.6f}"
            )
    return trace


def test_a_real_forward_backward_drives_the_loss_down(capsys) -> None:
    """The end-to-end claim: this seam trains. Thresholds are loose on purpose.

    A tight "loss below X" bound on a 2-block toy would be a pin on the toy, not on the step. What
    is asserted is the shape of a working optimisation — the probe falls materially, the tail sits
    well below the head, and nothing ever climbs above where it started (which is what a wrong
    velocity SIGN or an unweighted-vs-weighted mix-up produces first).
    """
    with capsys.disabled():
        trace = _train_smoke(verbose=True)
        head, tail = sum(trace[:20]) / 20, sum(trace[-20:]) / 20
        print(
            f"  first {trace[0]:.6f} -> last {trace[-1]:.6f} "
            f"({100 * (1 - trace[-1] / trace[0]):.1f}% down); "
            f"head20 {head:.6f} -> tail20 {tail:.6f}"
        )
    assert trace[-1] < trace[0] * 0.85, trace[-1]
    assert sum(trace[-20:]) / 20 < sum(trace[:20]) / 20 * 0.90
    assert max(trace) <= trace[0] * 1.02, max(trace)


def test_gradients_reach_every_one_of_the_fourteen_leaves() -> None:
    """A loss that decreases while a leaf gets no gradient is a partially-injected adapter."""
    model = _lora_model()
    step_fn = build_qwen_edit_step_fn(_strategy(), seed=42)
    loss = step_fn(
        model, _fixed_batch(), None, np.random.default_rng(1), device="cpu", dtype=torch.float32
    )
    loss.backward()
    touched = {
        name.split(".lora_")[0].split("transformer_blocks.")[1].split(".", 1)[1]
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    }
    assert touched == set(QWEN_EDIT_LORA_LEAVES), sorted(set(QWEN_EDIT_LORA_LEAVES) - touched)
