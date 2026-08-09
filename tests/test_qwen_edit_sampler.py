"""SLICE 1 (render) — the qwen_edit GENERATE call, its §8 recipe, and the scheduler trap.

``inference/qwen_edit_layout.render_qwen_edit_sample`` was the last declared stub in the qwen
surface. This file is what makes it landable without a GPU: every property below is proved on CPU,
with a FAKE pipeline that records the kwargs it was called with, plus the real diffusers scheduler
and the real diffusers geometry function.

Zero GPU. Zero Modal dispatch. Zero weight bodies. Nothing here downloads anything.

The one property this file exists for
-------------------------------------
``QWEN-CHAINED-EDIT-METHOD.md`` §8 names a diagnosis tell: *training samples look fine but renders
are muddy → it is inference settings, not the model*. The mechanism is a scheduler whose frozen
config says ``use_dynamic_shifting=True``: ``scheduling_flow_match_euler_discrete.py:347`` then takes
the ``mu`` branch and never reads ``self.shift``, so ``set_shift(3.0)`` moves the sigmas by exactly
zero. :func:`test_the_documented_trap_is_real_set_shift_on_a_dynamic_scheduler_is_a_no_op` MEASURES
that (it does not assert it from the docs), and the two tests after it prove the shipped assertion
refuses precisely that scheduler. A settings bug that produces a plausible image is the most
expensive way to be wrong on this family, because it condemns the adapter instead of the config.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_CONDITION_IMAGE_SIZE,
    QWEN_EDIT_MAX_CONTROL_SLOTS,
    QWEN_EDIT_VAE_IMAGE_SIZE,
    qwen_edit_area_budget_size,
)
from signet_trainer.lora.peft import build_lora_config
from signet_trainer.models.qwen_edit_pipeline import (
    QWEN_EDIT_RENDER_CFG_NORM,
    QWEN_EDIT_RENDER_LORA_SCALE,
    QWEN_EDIT_RENDER_NEGATIVE_PROMPT,
    QWEN_EDIT_RENDER_RECIPE,
    QWEN_EDIT_RENDER_SCHEDULER_SHIFT,
    QWEN_EDIT_RENDER_STEPS,
    QWEN_EDIT_RENDER_TRUE_CFG,
    assert_qwen_edit_control_geometry,
    assert_qwen_edit_scheduler_pinned,
    pin_qwen_edit_scheduler,
    qwen_edit_generate,
    qwen_edit_static_scheduler,
)

REPO = Path(__file__).resolve().parents[1]

#: The generate call under test. ``inference/qwen_edit_layout.render_qwen_edit_sample`` is a thin
#: delegation to it (proved by :func:`test_the_layout_entry_point_delegates_to_the_generate_call`);
#: every behavioural test below drives the implementation directly so a failure names the function
#: that owns the behaviour rather than the seam in front of it.
render_qwen_edit_sample = qwen_edit_generate

#: The 14-leaf path regex, verbatim from the TRAINED adapter's own ``adapter_config.json``
#: (checkpoint-step-00250-loss-0.0283). Used to PEFT-wrap the toy transformer so the base/adapter
#: toggle is exercised through the same target form the real run used.
QWEN_TARGET_REGEX = (
    r"transformer_blocks\.\d+\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|attn\.add_q_proj"
    r"|attn\.add_k_proj|attn\.add_v_proj|attn\.to_add_out|img_mlp\.net\.0\.proj|img_mlp\.net\.2"
    r"|txt_mlp\.net\.0\.proj|txt_mlp\.net\.2|img_mod\.1|txt_mod\.1)"
)


# --------------------------------------------------------------------------------------------------
# Fakes — a pipeline that RECORDS instead of denoising, and a transformer with the pipeline's surface.
# --------------------------------------------------------------------------------------------------


class _Config(dict):
    """Attribute + mapping access, like a diffusers ``FrozenDict`` (peft calls ``.get()`` on it)."""

    def __getattr__(self, name: str):  # noqa: ANN204
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(8, 8)
        self.attn.add_k_proj = nn.Linear(8, 8)


class _ToyTransformer(nn.Module):
    """Mimics ``QwenImageTransformer2DModel``'s SURFACE (not its maths): the attributes the pipeline
    reads off ``self.transformer`` — ``.config.guidance_embeds``, ``.config.in_channels``,
    ``.cache_context(...)`` — plus two of the 14 LoRA leaves so PEFT has something to target."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(), _Block()])
        self.config = _Config(guidance_embeds=False, in_channels=64)


class _FakePipeline:
    """Records the generate kwargs and returns a solid image. No denoising, no weights."""

    def __init__(self, scheduler, transformer) -> None:
        self.scheduler = scheduler
        self.transformer = transformer
        self.calls: list[dict] = []
        self._execution_device = torch.device("cpu")

    def register_modules(self, **modules) -> None:
        for key, value in modules.items():
            setattr(self, key, value)

    def __call__(self, **kwargs):  # noqa: ANN204
        self.calls.append(kwargs)
        img = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), "purple")
        return type("Out", (), {"images": [img]})()


def _peft_transformer():
    from peft import get_peft_model

    model = get_peft_model(
        _ToyTransformer(),
        build_lora_config(rank=42, alpha=42, dropout=0.0, targets=QWEN_TARGET_REGEX),
    )
    model.eval()
    return model


def _pinned_pipeline(*, adapter: bool = True) -> _FakePipeline:
    transformer = _peft_transformer() if adapter else _ToyTransformer()
    return _FakePipeline(qwen_edit_static_scheduler(), transformer)


def _controls(*sizes: tuple[int, int]) -> list[Image.Image]:
    return [Image.new("RGB", size, "white") for size in (sizes or ((1024, 1024),))]


# --------------------------------------------------------------------------------------------------
# (1) The recipe IS §8. One compare, not six scattered literals.
# --------------------------------------------------------------------------------------------------


def test_the_shipped_recipe_is_the_method_section_8_inference_table() -> None:
    """§8, Qwen-Image-Edit-2511 row: 30 steps · true_cfg 4.0 · CFGNorm ON · STATIC shift 3.0 ·
    LoRA strength 1.0 · the reference into both encodes (which on this pipeline == a non-None
    negative prompt)."""
    recipe = QWEN_EDIT_RENDER_RECIPE
    assert (recipe.steps, recipe.true_cfg, recipe.scheduler_shift, recipe.lora_scale) == (
        30,
        4.0,
        3.0,
        1.0,
    )
    assert recipe.cfg_norm is True
    assert recipe.negative_prompt is not None and recipe.negative_prompt != ""
    # The constants and the dataclass must not be able to drift apart.
    assert (
        QWEN_EDIT_RENDER_STEPS,
        QWEN_EDIT_RENDER_TRUE_CFG,
        QWEN_EDIT_RENDER_SCHEDULER_SHIFT,
        QWEN_EDIT_RENDER_LORA_SCALE,
        QWEN_EDIT_RENDER_CFG_NORM,
        QWEN_EDIT_RENDER_NEGATIVE_PROMPT,
    ) == (
        recipe.steps,
        recipe.true_cfg,
        recipe.scheduler_shift,
        recipe.lora_scale,
        recipe.cfg_norm,
        recipe.negative_prompt,
    )
    # NOT the diffusers defaults, which is the whole point of pinning them.
    assert recipe.steps != 50, "50 is the pipeline's default; §8 says 30"


# --------------------------------------------------------------------------------------------------
# (2) THE TRAP — measured here, not quoted.
# --------------------------------------------------------------------------------------------------

_AI_TOOLKIT_SCHEDULER_CONFIG = dict(
    base_image_seq_len=256,
    base_shift=0.5,
    max_image_seq_len=8192,
    max_shift=0.9,
    shift=1.0,
    shift_terminal=0.02,
    use_dynamic_shifting=True,
    time_shift_type="exponential",
)
_STEPS = 30
_SIGMAS = np.linspace(1.0, 1 / _STEPS, _STEPS)


def _sigmas_of(scheduler, mu):
    scheduler.set_timesteps(sigmas=_SIGMAS, device="cpu", mu=mu)
    return scheduler.sigmas.clone()


def test_the_documented_trap_is_real_set_shift_on_a_dynamic_scheduler_is_a_no_op() -> None:
    """``set_shift(3.0)`` on ``use_dynamic_shifting=True`` moves the sigmas by EXACTLY zero.

    This is why :func:`pin_qwen_edit_scheduler` replaces the scheduler OBJECT instead of setting its
    shift, and why the pin is verified rather than assumed. If this test ever goes red, diffusers
    changed the branch at ``scheduling_flow_match_euler_discrete.py:347`` and the pin's rationale
    must be re-read, not the pin deleted.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift

    dynamic = FlowMatchEulerDiscreteScheduler.from_config(_AI_TOOLKIT_SCHEDULER_CONFIG)
    mu = calculate_shift(4096, 256, 8192, 0.5, 0.9)  # a 1024x1024 latent
    before = _sigmas_of(dynamic, mu)
    dynamic.set_shift(3.0)
    after = _sigmas_of(dynamic, mu)

    assert dynamic.shift == 3.0, "the instance value DID change — that is what makes it deceptive"
    assert torch.equal(before, after), (
        "set_shift on a dynamic scheduler is supposed to be a no-op; if it is not, this family's "
        "pin rationale is stale"
    )


def test_static_shift_equals_dynamic_mu_ln_shift_to_float_epsilon() -> None:
    """The two parameterisations are the same curve: static ``S`` == dynamic ``mu = ln(S)``.

    Proved for §8's 3.0 and for the base-Qwen row's 7.0, with ``shift_terminal`` disabled on both so
    the comparison isolates one variable. This is what licenses reading ai-toolkit's and stock
    diffusers' default ``mu`` values as effective static shifts when diagnosing a muddy render.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    clean = dict(_AI_TOOLKIT_SCHEDULER_CONFIG, shift_terminal=None)
    for value in (3.0, 7.0):
        static = FlowMatchEulerDiscreteScheduler.from_config(
            clean, use_dynamic_shifting=False, shift=value
        )
        dynamic = FlowMatchEulerDiscreteScheduler.from_config(clean)
        delta = (_sigmas_of(static, None) - _sigmas_of(dynamic, math.log(value))).abs().max().item()
        assert delta < 1e-6, f"static {value} != dynamic mu=ln({value}): max delta {delta}"


def test_a_pinned_scheduler_ignores_mu_so_renders_are_resolution_independent() -> None:
    """Under the static branch ``mu`` is passed by the pipeline on every call and never read.

    ``pipeline_qwenimage_edit_plus.py:753-767`` computes ``mu`` unconditionally from the latent
    sequence length and hands it to ``retrieve_timesteps``. Two very different latents must therefore
    produce the SAME schedule once pinned — that is the property that makes a band comparable across
    resolutions, and it is the reason the pin is worth having even where the stock default happens
    to land near 3.0 already.
    """
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_shift

    pinned = qwen_edit_static_scheduler()
    small = _sigmas_of(pinned, calculate_shift(3072, 256, 8192, 0.5, 0.9))
    large = _sigmas_of(pinned, calculate_shift(6889, 256, 8192, 0.5, 0.9))
    assert torch.equal(small, large)


# --------------------------------------------------------------------------------------------------
# (3) The pin, and the assertion that proves it took.
# --------------------------------------------------------------------------------------------------


def test_the_static_scheduler_carries_the_recipe_value_and_the_static_branch() -> None:
    scheduler = qwen_edit_static_scheduler()
    report = assert_qwen_edit_scheduler_pinned(scheduler)
    assert report["use_dynamic_shifting"] is False
    assert report["shift"] == QWEN_EDIT_RENDER_SCHEDULER_SHIFT == 3.0


def test_the_assertion_refuses_the_trap_scheduler_and_names_the_mu_branch() -> None:
    """A dynamic scheduler whose ``.shift`` reads 3.0 must be REFUSED.

    This is the exact object a well-intentioned ``pipeline.scheduler.set_shift(3.0)`` produces. A
    check that only compared ``.shift`` would greenlight it and every render would be muddy.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    trap = FlowMatchEulerDiscreteScheduler.from_config(_AI_TOOLKIT_SCHEDULER_CONFIG)
    trap.set_shift(3.0)
    assert trap.shift == 3.0  # the value a naive check would accept

    with pytest.raises(RuntimeError) as excinfo:
        assert_qwen_edit_scheduler_pinned(trap)
    message = str(excinfo.value)
    assert "use_dynamic_shifting" in message
    assert "347" in message, "the message must point at the branch that ignores .shift"
    assert "0.000e+00" in message, "the message must carry the measurement, not just the claim"


def test_a_config_dict_override_is_silently_dropped_which_is_why_the_pin_uses_kwargs() -> None:
    """RED self-check for a live diffusers gotcha the shipped assertion caught during this slice.

    ``ConfigMixin.extract_init_dict`` strips every key named in the donor config's own
    ``_use_default_values`` and re-defaults it. So writing an override INTO an inherited config dict
    is silently dropped for any field the donor left at its default, while the same override passed
    as a kwarg survives. A pin written the first way produces a scheduler that looks configured and
    steps at 1.0.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    base = dict(FlowMatchEulerDiscreteScheduler().config)
    base["shift"] = 3.0
    assert FlowMatchEulerDiscreteScheduler.from_config(base).shift == 1.0, (
        "diffusers no longer drops dict-borne overrides — re-read extract_init_dict before "
        "simplifying qwen_edit_static_scheduler"
    )
    assert FlowMatchEulerDiscreteScheduler.from_config(base, shift=3.0).shift == 3.0
    # ...and the shipped constructor takes the surviving form.
    assert qwen_edit_static_scheduler().shift == 3.0


def test_the_assertion_refuses_a_factory_rebuilt_default_scheduler() -> None:
    """The other half: a STATIC scheduler at somebody else's default value is also refused."""
    wrong = qwen_edit_static_scheduler(shift=1.0)
    with pytest.raises(RuntimeError, match=r"shift is 1\.0, expected 3\.0"):
        assert_qwen_edit_scheduler_pinned(wrong)


def test_pin_replaces_a_rebuilt_scheduler_after_the_pipeline_already_exists() -> None:
    """§8's instruction, executed: override ``pipeline.scheduler`` AFTER the pipeline is built.

    Simulates ai-toolkit's ``get_generation_pipeline``, which calls ``get_train_scheduler()`` at
    ``qwen_image_edit_plus.py:76`` and hands the pipeline a fresh dynamic scheduler whatever the
    caller wanted.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    rebuilt = FlowMatchEulerDiscreteScheduler.from_config(_AI_TOOLKIT_SCHEDULER_CONFIG)
    pipeline = _FakePipeline(rebuilt, _peft_transformer())
    with pytest.raises(RuntimeError):
        assert_qwen_edit_scheduler_pinned(pipeline)  # RED before the pin

    report = pin_qwen_edit_scheduler(pipeline)
    assert pipeline.scheduler is not rebuilt
    assert report["shift"] == 3.0 and report["use_dynamic_shifting"] is False
    assert_qwen_edit_scheduler_pinned(pipeline)  # GREEN after


def test_the_pin_inherits_the_supplied_configs_other_fields_rather_than_re_declaring_them() -> None:
    """``shift_terminal`` is an OPEN question (ai-toolkit 0.02 vs stock None) — inherited, not picked.

    Re-declaring the whole scheduler config here would silently answer a question no source settles.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    donor = FlowMatchEulerDiscreteScheduler.from_config(_AI_TOOLKIT_SCHEDULER_CONFIG)
    pipeline = _FakePipeline(donor, _peft_transformer())
    report = pin_qwen_edit_scheduler(pipeline)
    assert report["shift_terminal"] == 0.02, "the donor's own field must survive the pin"
    assert report["time_shift_type"] == "exponential"


# --------------------------------------------------------------------------------------------------
# (4) Control geometry — signet's module is the authority, checked against the pipeline's own resize.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size",
    [
        (1024, 1024), (512, 512), (1328, 1328), (768, 1024), (1024, 768),
        (2000, 40), (40, 2000), (1920, 1080), (1080, 1920), (333, 777),
    ],
)
def test_signets_geometry_and_the_pipelines_own_resize_agree_exactly(size) -> None:
    """The gate's premise. If these ever diverge, a control image is conditioned on pixels the
    training leg never saw — and on a non-square input that shows up as a transposed composition
    that reads as an adapter that "learned the wrong framing"."""
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions

    width, height = size
    for budget in (QWEN_EDIT_VAE_IMAGE_SIZE, QWEN_EDIT_CONDITION_IMAGE_SIZE):
        ours = qwen_edit_area_budget_size(width, height, budget)
        theirs = tuple(int(v) for v in calculate_dimensions(budget, width / height))
        assert ours == theirs, f"{size} @ {budget}: signet {ours} vs pipeline {theirs}"


def test_the_geometry_gate_reports_both_budgets_per_slot() -> None:
    report = assert_qwen_edit_control_geometry(_controls((1024, 1024), (2048, 1024)))
    assert [entry["index"] for entry in report] == [0, 1]
    assert report[0]["vae_wh"] == (1024, 1024)
    assert report[0]["condition_wh"] == (384, 384)
    # non-square: orientation PRESERVED (diffusers-correct), landscape stays landscape
    assert report[1]["vae_wh"][0] > report[1]["vae_wh"][1]


def test_the_geometry_gate_refuses_a_fabricated_divergence(monkeypatch) -> None:
    """RED self-check: transpose the pipeline's answer and the gate must refuse.

    This is the ai-toolkit ``ratio = H/W`` bug injected deliberately — the failure the gate exists
    to catch if signet is ever pointed at a bug-compatible stack.
    """
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as mod

    original = mod.calculate_dimensions
    monkeypatch.setattr(
        mod, "calculate_dimensions", lambda area, ratio: tuple(reversed(original(area, ratio)))
    )
    with pytest.raises(RuntimeError, match="geometry DIVERGES"):
        assert_qwen_edit_control_geometry(_controls((2048, 1024)))


def test_the_geometry_gate_refuses_empty_and_over_capacity_control_sets() -> None:
    with pytest.raises(ValueError, match="EMPTY"):
        assert_qwen_edit_control_geometry([])
    too_many = _controls(*[(512, 512)] * (QWEN_EDIT_MAX_CONTROL_SLOTS + 1))
    with pytest.raises(ValueError, match="ceiling"):
        assert_qwen_edit_control_geometry(too_many)


# --------------------------------------------------------------------------------------------------
# (5) The generate call itself — run end to end against the fake pipeline.
# --------------------------------------------------------------------------------------------------


def test_the_render_passes_exactly_the_section_8_parameter_set(tmp_path) -> None:
    """The call's kwargs ARE the recipe, and ``guidance_scale`` is absent from them.

    ``guidance_scale`` being absent is not cosmetic: this checkpoint is not guidance-distilled, so
    the pipeline would log 'ignored since the model is not guidance-distilled', set ``guidance=None``
    and render at effective CFG 1.0 — muddy, and indistinguishable from a bad adapter.
    """
    pipeline = _pinned_pipeline()
    out = tmp_path / "lora" / "ckpt" / "a_style" / "train_icon_s42.png"

    written = render_qwen_edit_sample(
        pipeline=pipeline,
        control_images=_controls((1024, 1024)),
        prompt="reimagine the reference icon in the house style",
        out_path=out,
        seed=42,
        width=1024,
        height=1024,
    )

    assert written == str(out) and out.exists() and out.stat().st_size > 0
    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert call["num_inference_steps"] == 30
    assert call["true_cfg_scale"] == 4.0
    assert call["height"] == 1024 and call["width"] == 1024
    assert call["output_type"] == "pil"
    assert call["negative_prompt"] == QWEN_EDIT_RENDER_NEGATIVE_PROMPT
    assert call["negative_prompt"] is not None, "a None negative prompt turns true-CFG off"
    assert "guidance_scale" not in call, "guidance_scale would render at effective CFG 1.0"
    assert call["generator"].initial_seed() == 42
    # RAW control images — the pipeline resizes; pre-resizing here would double-resample.
    assert call["image"][0].size == (1024, 1024)


def test_the_render_refuses_guidance_scale_by_name(tmp_path) -> None:
    with pytest.raises(TypeError) as excinfo:
        render_qwen_edit_sample(
            pipeline=_pinned_pipeline(),
            control_images=_controls(),
            prompt="p",
            out_path=tmp_path / "x.png",
            seed=1,
            width=512,
            height=512,
            guidance_scale=4.0,
        )
    message = str(excinfo.value)
    assert "guidance_scale" in message and "true_cfg" in message
    assert "not guidance-distilled" in message


def test_the_render_refuses_a_true_cfg_that_silently_disables_guidance(tmp_path) -> None:
    with pytest.raises(ValueError, match="DISABLES classifier-free guidance"):
        render_qwen_edit_sample(
            pipeline=_pinned_pipeline(),
            control_images=_controls(),
            prompt="p",
            out_path=tmp_path / "x.png",
            seed=1,
            width=512,
            height=512,
            true_cfg=1.0,
        )


def test_the_render_refuses_an_unpinned_pipeline_before_drawing_a_single_sigma(tmp_path) -> None:
    """The money-safe ordering: the gate runs before generate, so a muddy render never happens."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    unpinned = _FakePipeline(
        FlowMatchEulerDiscreteScheduler.from_config(_AI_TOOLKIT_SCHEDULER_CONFIG),
        _peft_transformer(),
    )
    with pytest.raises(RuntimeError, match="use_dynamic_shifting"):
        render_qwen_edit_sample(
            pipeline=unpinned,
            control_images=_controls(),
            prompt="p",
            out_path=tmp_path / "x.png",
            seed=1,
            width=512,
            height=512,
        )
    assert unpinned.calls == [], "the pipeline must not have been called at all"
    assert not (tmp_path / "x.png").exists()


def test_the_render_refuses_an_empty_prompt(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty prompt"):
        render_qwen_edit_sample(
            pipeline=_pinned_pipeline(),
            control_images=_controls(),
            prompt="   ",
            out_path=tmp_path / "x.png",
            seed=1,
            width=512,
            height=512,
        )


# --------------------------------------------------------------------------------------------------
# (6) The BASE row — same call, adapter off, one model.
# --------------------------------------------------------------------------------------------------


def test_the_base_row_is_the_same_call_with_the_adapter_disabled(tmp_path) -> None:
    """§8 reads convergence as base-vs-LoRA divergence, so the two rows must differ ONLY here."""
    pipeline = _pinned_pipeline()
    common = dict(
        pipeline=pipeline,
        control_images=_controls((1024, 1024)),
        prompt="reimagine the reference icon",
        seed=42,
        width=1024,
        height=1024,
    )
    render_qwen_edit_sample(**common, out_path=tmp_path / "base.png", adapter=False)
    render_qwen_edit_sample(**common, out_path=tmp_path / "lora.png", adapter=True)

    base_call, lora_call = pipeline.calls
    comparable = lambda call: {k: v for k, v in call.items() if k not in ("generator", "image")}  # noqa: E731
    assert comparable(base_call) == comparable(lora_call)
    assert base_call["generator"].initial_seed() == lora_call["generator"].initial_seed() == 42


def test_an_adapter_render_on_an_unwrapped_transformer_is_refused(tmp_path) -> None:
    """The dangerous direction: a BASE render shipped under a checkpoint's label.

    A grid whose band columns are silently the base column reports 'not converging' for every
    checkpoint ever rendered — §8's judgment, inverted, with no visible symptom.
    """
    pipeline = _pinned_pipeline(adapter=False)
    with pytest.raises(RuntimeError, match="not PEFT-"):
        render_qwen_edit_sample(
            pipeline=pipeline,
            control_images=_controls(),
            prompt="p",
            out_path=tmp_path / "x.png",
            seed=1,
            width=512,
            height=512,
            adapter=True,
        )
    assert pipeline.calls == []


def test_a_base_render_on_an_unwrapped_transformer_is_allowed(tmp_path) -> None:
    """The harmless direction warns and proceeds — it is redundant, not a mislabel."""
    pipeline = _pinned_pipeline(adapter=False)
    render_qwen_edit_sample(
        pipeline=pipeline,
        control_images=_controls(),
        prompt="p",
        out_path=tmp_path / "x.png",
        seed=1,
        width=512,
        height=512,
        adapter=False,
    )
    assert len(pipeline.calls) == 1


# --------------------------------------------------------------------------------------------------
# (7) Import tier + the shipped guard. Subprocess, because sys.modules is suite-order dependent.
# --------------------------------------------------------------------------------------------------


def test_the_pipeline_module_is_import_light_in_a_fresh_process() -> None:
    """Reading the recipe constants must not drag in torch, diffusers or the Modal SDK.

    A SUBPROCESS, not an in-process ``sys.modules`` check: in-process, the question answered is
    "what has this suite imported by now", which is order-dependent and has bitten this repo twice.
    """
    probe = (
        "import sys, importlib\n"
        "importlib.import_module('signet_trainer.models.qwen_edit_pipeline')\n"
        "leaked = [m for m in ('torch', 'diffusers', 'modal', 'peft') if m in sys.modules]\n"
        "assert not leaked, leaked\n"
        "print('IMPORT_LIGHT_OK')\n"
    )
    import os

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO),
        env=dict(os.environ, PYTHONPATH="src"),
        capture_output=True,
        text=True,
    )
    assert "IMPORT_LIGHT_OK" in result.stdout, (
        f"stdout={result.stdout}\nstderr={result.stderr[-1500:]}"
    )


def test_the_sampler_never_spells_the_banned_token_in_executable_code() -> None:
    """``tests/test_no_wan_params.py`` bans the bare token from every ``*.py`` under ``inference/``.

    Pinned locally too, so the sampler's own suite fails first if someone inlines the scheduler
    construction back into the layout module rather than importing the pinned pipeline. The guard
    is not weakened to land this — the construction lives in ``models/``, which it does not scan.
    """
    src = REPO / "src" / "signet_trainer" / "inference" / "qwen_edit_layout.py"
    code = re.sub(r'"""(?:.|\n)*?"""', "", src.read_text(encoding="utf-8"))
    code = re.sub(r"#.*", "", code)
    for token in ("UniPC", "shift", "frames=33", "num_inference_steps=50"):
        assert token not in code, f"{token!r} leaked into qwen_edit_layout.py executable code"


# --------------------------------------------------------------------------------------------------
# (8) The declared entry point delegates — one implementation, not two.
# --------------------------------------------------------------------------------------------------


def test_the_layout_entry_point_delegates_to_the_generate_call(tmp_path) -> None:
    """``render_qwen_edit_sample`` is the declared symbol; ``qwen_edit_generate`` is the behaviour.

    Driven end to end through the LAYOUT entry point here (every other test drives the impl), so the
    seam itself is exercised: the forwarding, the function-local import, and the return value.
    """
    from signet_trainer.inference.qwen_edit_layout import (
        render_qwen_edit_sample as entry_point,
    )

    pipeline = _pinned_pipeline()
    out = tmp_path / "cell.png"
    written = entry_point(
        pipeline=pipeline,
        control_images=_controls((1024, 1024)),
        prompt="reimagine the reference icon",
        out_path=out,
        seed=42,
        width=1024,
        height=1024,
    )
    assert written == str(out) and out.exists()
    assert pipeline.calls[0]["num_inference_steps"] == 30
    assert pipeline.calls[0]["true_cfg_scale"] == 4.0


def test_the_delegations_signature_cannot_drift_from_the_implementations() -> None:
    """The entry point restates its parameter list instead of ``**kwargs``-forwarding, so that it is
    self-documenting. This is what makes the restatement safe."""
    import inspect

    from signet_trainer.inference.qwen_edit_layout import (
        render_qwen_edit_sample as entry_point,
    )

    assert inspect.signature(entry_point) == inspect.signature(qwen_edit_generate)


def test_there_is_exactly_one_generate_call_in_the_tree() -> None:
    """No second transcription of the pipeline invocation — the D-9-DEDUP rule, as an AST check.

    The duplicate this guards against is not hypothetical: during this slice the implementation
    briefly existed in both modules at once.
    """
    import ast

    hits: list[str] = []
    for path in sorted((REPO / "src" / "signet_trainer").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and {
                kw.arg for kw in node.keywords if kw.arg
            } >= {"true_cfg_scale", "num_inference_steps", "negative_prompt"}:
                hits.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert len(hits) == 1, f"expected ONE Qwen generate call site, found {hits}"
    assert hits[0].startswith("src\\signet_trainer\\models\\qwen_edit_pipeline.py") or hits[
        0
    ].startswith("src/signet_trainer/models/qwen_edit_pipeline.py"), hits


def test_the_scheduler_construction_lives_outside_the_scanned_directory() -> None:
    """The reversible placement, asserted — so nobody 'tidies' it back into ``inference/``."""
    scanned = REPO / "src" / "signet_trainer" / "inference"
    assert not (scanned / "qwen_edit_pipeline.py").exists()
    assert (REPO / "src" / "signet_trainer" / "models" / "qwen_edit_pipeline.py").exists()
