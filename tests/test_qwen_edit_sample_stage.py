"""The gated ``qwen_edit`` SAMPLE stage: its $0 config gaps, its cost line and its launch order.

Slice 3's surface is the one that spends money — ``modal/fns.py::qwen_edit_sample`` plus the
``--mode sample`` arm of ``modal/entrypoint.py`` — so everything here is a CPU assertion about
ORDER, REFUSALS and ARITHMETIC. Zero GPU, zero Modal dispatch, zero weight bytes: the two
behavioural tests drive the real ``main()`` with a recording stand-in bound over the dispatch verb,
so a regression that reached ``.spawn()`` fails the test instead of booting an A100.

⚠ Imports of ``signet_trainer.modal.*`` are LAZY (inside test bodies), the convention
``tests/test_entrypoint_gate_behavioral.py`` states in its own docstring: pytest COLLECTION must not
drag the Modal SDK into ``sys.modules``, or the dry-run purity assertions in ``test_dryrun_*.py``
(``assert "modal" not in sys.modules``) fail for the whole session.
"""

from __future__ import annotations

import builtins
import re
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_FNS = REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"
_ENTRYPOINT = REPO_ROOT / "src" / "signet_trainer" / "modal" / "entrypoint.py"
_EXAMPLE_CONFIG = REPO_ROOT / "configs" / "qwen_image_edit.example.yaml"


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop triple-quoted blocks + line comments so prose can never satisfy a source-order scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _sample_stage_code() -> str:
    """The EXECUTABLE body of ``qwen_edit_sample`` — decorator excluded, prose stripped."""
    src = _FNS.read_text(encoding="utf-8")
    start = src.index("def qwen_edit_sample(")
    return _strip_comments_and_docstrings(src[start:])


def _held_out(**overrides):
    """A stand-in for one declared ``qwen_edit.render_inputs`` entry (the field has not landed)."""
    item = {
        "id": "train_icon",
        "images": ("heldout/train_a.png", "heldout/train_b.png", "heldout/train_c.png"),
        "prompts": {
            "a_style": "reimagine the reference icon",
            "b_content": "reimagine the train icon",
        },
    }
    item.update(overrides)
    return types.SimpleNamespace(**item)


def _example_cfg():
    from signet_trainer.config.load import load_config

    return load_config(str(_EXAMPLE_CONFIG))


def _declare_render_request(cfg, *, band=("checkpoint-step-4000",), inputs=None) -> None:
    """Attach the two DECLARED-GAP fields to a loaded config.

    ``object.__setattr__`` because ``QwenEditConfig`` is a pydantic model with ``extra="forbid"``
    and these fields do not exist yet — the same idiom
    ``tests/test_qwen_edit_component_paths.py`` already uses to prove the pipeline_root gap. This is
    what makes "the check goes green when the field lands" testable BEFORE it lands.
    """
    from signet_trainer.modal.entrypoint import (
        _QWEN_EDIT_BAND_FIELD,
        _QWEN_EDIT_INPUTS_FIELD,
    )

    object.__setattr__(cfg.qwen_edit, _QWEN_EDIT_BAND_FIELD, tuple(band))
    object.__setattr__(
        cfg.qwen_edit, _QWEN_EDIT_INPUTS_FIELD, tuple(inputs if inputs is not None else (_held_out(),))
    )


# ==================================================================================================
# The $0 gap checks — BOTH directions, because a check that only ever refuses is not a check.
# ==================================================================================================


def test_the_sample_gaps_fire_and_then_clear_when_the_request_is_declared() -> None:
    """Absent band + held-out set -> exactly two named gaps; declared -> none.

    The shipped example config is a legal ``preprocess``/``train`` recipe, so the sample gaps are
    the only difference between the two modes — which is the property that makes them per-MODE
    rather than a blanket refusal.
    """
    from signet_trainer.modal.entrypoint import _qwen_edit_config_gaps

    cfg = _example_cfg()
    gaps = _qwen_edit_config_gaps(cfg, mode="sample")
    assert len(gaps) == 2, gaps
    assert any("render_checkpoint_band" in gap for gap in gaps)
    assert any("render_inputs" in gap for gap in gaps)
    for gap in gaps:
        assert "WHAT LANDS IT" in gap, "a gap that does not name its remedy is a dead end"

    # The same config remains dispatchable for the encode leg — the gaps are sample-only.
    assert _qwen_edit_config_gaps(cfg, mode="preprocess") == []

    _declare_render_request(cfg)
    assert _qwen_edit_config_gaps(cfg, mode="sample") == [], (
        "with the band and the held-out set declared the sample arm must be dispatchable — a check "
        "that can never pass is a refusal wearing a check's clothes"
    )


def test_a_control_list_that_does_not_cover_every_slot_is_named_precisely() -> None:
    """A short/long image list is the POSITIONAL failure, and it is caught per entry, by id."""
    from signet_trainer.modal.entrypoint import _qwen_edit_config_gaps

    cfg = _example_cfg()
    _declare_render_request(cfg, inputs=(_held_out(id="short", images=("only_one.png",)),))
    gaps = _qwen_edit_config_gaps(cfg, mode="sample")
    assert len(gaps) == 1, gaps
    assert "'short'" in gaps[0] and "1 control image" in gaps[0]
    assert str(cfg.qwen_edit.control_slots) in gaps[0]


def test_an_entry_missing_its_prompt_pair_is_refused_by_field_name() -> None:
    from signet_trainer.modal.entrypoint import _qwen_edit_config_gaps

    cfg = _example_cfg()
    _declare_render_request(cfg, inputs=(_held_out(prompts={}),))
    gaps = _qwen_edit_config_gaps(cfg, mode="sample")
    assert len(gaps) == 1 and "prompts" in gaps[0], gaps


def test_a_config_contradicting_the_locked_recipe_is_refused_at_zero_dollars() -> None:
    """``validation.guidance_scale`` / ``num_inference_steps`` are not read; disagreeing is refused.

    The render takes its settings from ``QWEN_EDIT_RENDER_RECIPE``. A config that declares a
    different number would put a figure in the gallery banner that describes a render nobody
    performed — the failure ``h3_sample`` records for its own banner-only width/height.
    """
    from signet_trainer.modal.entrypoint import _qwen_edit_config_gaps
    from signet_trainer.models.qwen_edit_pipeline import QWEN_EDIT_RENDER_RECIPE

    cfg = _example_cfg()
    _declare_render_request(cfg)
    assert _qwen_edit_config_gaps(cfg, mode="sample") == [], (
        "the shipped example must already agree with the locked recipe — it is what every new "
        "qwen_edit config gets copied from"
    )
    assert float(cfg.validation.guidance_scale) == float(QWEN_EDIT_RENDER_RECIPE.true_cfg)

    object.__setattr__(cfg.validation, "guidance_scale", 3.0)
    object.__setattr__(cfg.validation, "num_inference_steps", 25)
    gaps = _qwen_edit_config_gaps(cfg, mode="sample")
    assert len(gaps) == 1, gaps
    assert "guidance_scale=3.0" in gaps[0] and "num_inference_steps=25" in gaps[0]
    assert "true_cfg_scale" in gaps[0], "the gap must also name the keyword the value maps to"


def test_the_sample_arm_inherits_the_pipeline_root_refusal() -> None:
    """The render loads the PROCESSOR from ``<root>/processor`` exactly as the pre-encode does."""
    from signet_trainer.modal.entrypoint import _qwen_edit_config_gaps

    cfg = _example_cfg()
    _declare_render_request(cfg)
    object.__setattr__(cfg.model, "pipeline_root_id", None)
    gaps = _qwen_edit_config_gaps(cfg, mode="sample")
    assert len(gaps) == 1 and "pipeline_root_id" in gaps[0] and "processor" in gaps[0], gaps


def test_the_readiness_table_declares_the_unlanded_pipeline_module_for_sample() -> None:
    """Every module the sample stage imports is declared, so a gap costs $0 rather than a boot."""
    from signet_trainer.modal.entrypoint import _QWEN_EDIT_STAGE_MODULES

    declared = _QWEN_EDIT_STAGE_MODULES["sample"]
    assert "signet_trainer.models.qwen_edit_pipeline" in declared
    stage = _sample_stage_code()
    for module in declared:
        assert module in stage, (
            f"{module} is declared ready-checked for the sample stage but the stage never imports "
            "it — a readiness table that names modules nobody uses stops meaning anything"
        )


def test_the_modal_stage_reads_the_same_field_names_the_entrypoint_gaps_name() -> None:
    """A rename that missed one side = a config the entrypoint accepts and the container refuses."""
    from signet_trainer.modal.entrypoint import (
        _QWEN_EDIT_BAND_FIELD,
        _QWEN_EDIT_INPUTS_FIELD,
    )

    stage = _sample_stage_code()
    for field in (_QWEN_EDIT_BAND_FIELD, _QWEN_EDIT_INPUTS_FIELD):
        assert f'"{field}"' in stage, (
            f"modal/fns.py::qwen_edit_sample must read {field!r} — the entrypoint refuses its "
            "absence at $0 and the stage is the in-container mirror of that refusal"
        )


# ==================================================================================================
# The cost line — real arithmetic on declared inputs.
# ==================================================================================================


def test_the_render_batch_counts_one_base_group_not_one_per_band_member() -> None:
    """``2 x (1 base + 3 members) x 2 inputs = 16``, never ``2 x 3 x 2 x 2``.

    The base render is the whole grid's convergence reference and is keyed on the reserved ``base``
    token, so it is rendered ONCE. Counting it per member would over-state the batch by exactly the
    number of redundant renders a naive layout would also PAY for.
    """
    from signet_trainer.modal.cost import format_render_batch_line, render_batch_estimate

    est = render_batch_estimate(
        band_members=3, prompt_modes=2, held_out_inputs=2, steps_per_image=30, est_hours=4.0
    )
    assert (est.columns, est.images, est.denoise_steps_total) == (8, 16, 480)
    assert est.seconds_per_image_budget == pytest.approx(4.0 * 3600 / 16)
    line = format_render_batch_line(est)
    assert "16 image(s)" in line and "480 total" in line and "900 s per image" in line


def test_the_render_batch_refuses_inputs_it_cannot_size() -> None:
    from signet_trainer.modal.cost import render_batch_estimate

    with pytest.raises(ValueError, match="band_members"):
        render_batch_estimate(
            band_members=-1, prompt_modes=2, held_out_inputs=1, steps_per_image=30, est_hours=1.0
        )
    with pytest.raises(ValueError, match="steps_per_image"):
        render_batch_estimate(
            band_members=1, prompt_modes=2, held_out_inputs=1, steps_per_image=0, est_hours=1.0
        )
    with pytest.raises(ValueError, match="est_hours"):
        render_batch_estimate(
            band_members=1, prompt_modes=2, held_out_inputs=1, steps_per_image=30, est_hours=-1.0
        )
    empty = render_batch_estimate(
        band_members=1, prompt_modes=2, held_out_inputs=0, steps_per_image=30, est_hours=1.0
    )
    assert empty.images == 0 and empty.seconds_per_image_budget is None, (
        "a zero-image batch must not report a budget — 'inf s/image' reads as a generous one"
    )


def test_the_cost_note_says_NOT_SIZEABLE_before_the_request_is_declared() -> None:
    """The note is honest in both states, and names the fields that would make it sizeable."""
    from signet_trainer.modal.entrypoint import _qwen_edit_render_batch_note

    cfg = _example_cfg()
    undeclared = _qwen_edit_render_batch_note(cfg)
    assert "NOT SIZEABLE" in undeclared
    assert "render_checkpoint_band" in undeclared and "render_inputs" in undeclared

    _declare_render_request(cfg, band=("a", "b", "c"))
    sized = _qwen_edit_render_batch_note(cfg)
    assert "NOT SIZEABLE" not in sized
    assert "8 image(s)" in sized, sized  # 2 modes x (1 base + 3 members) x 1 input


def test_the_cost_note_prices_the_steps_the_render_actually_uses() -> None:
    """Sized off the LOCKED recipe, not off ``validation.num_inference_steps``, which is not read."""
    from signet_trainer.modal.entrypoint import _qwen_edit_render_batch_note
    from signet_trainer.models.qwen_edit_pipeline import QWEN_EDIT_RENDER_RECIPE

    cfg = _example_cfg()
    _declare_render_request(cfg)
    object.__setattr__(cfg.validation, "num_inference_steps", 999)
    note = _qwen_edit_render_batch_note(cfg)
    assert f"{QWEN_EDIT_RENDER_RECIPE.steps} denoise step(s)" in note
    assert "999" not in note, "pricing a batch off a field the sampler never reads is a fiction"


# ==================================================================================================
# Launch order (MODL-02) and stage order — source scans, the house convention.
# ==================================================================================================


def test_the_qwen_sample_dispatch_follows_approval_readiness_and_the_gap_refusal() -> None:
    """approval -> readiness -> gap refusal -> ``.spawn``. Nothing may precede the approval pause."""
    code = _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))

    approval = [m.start() for m in re.finditer(r"_require_approval\s*\(", code)][-1]
    dispatch = re.search(
        r"qwen_edit_sample(?:\.with_options\([^)]*\))?\.spawn\s*\(\s*config_text\s*\)", code
    )
    assert dispatch is not None, "the sample arm must dispatch qwen_edit_sample.spawn(config_text)"

    readiness = re.search(r'_qwen_edit_stage_readiness\(\s*"sample"\s*\)', code)
    refusal = re.search(r'_qwen_edit_refuse_on_gaps\(\s*cfg,\s*mode="sample"\s*\)', code)
    assert readiness is not None and refusal is not None

    assert approval < readiness.start() < refusal.start() < dispatch.start(), (
        "MODL-02: the $0 pre-dispatch checks sit strictly between the blocking approval pause and "
        f"the dispatch — approval@{approval} readiness@{readiness.start()} "
        f"refusal@{refusal.start()} spawn@{dispatch.start()}"
    )


def test_the_render_batch_note_prints_before_the_cost_line_and_decides_nothing() -> None:
    """It is a LINE, not a decision: the guardrail's basis stays ``cfg.modal.est_hours``."""
    code = _strip_comments_and_docstrings(_ENTRYPOINT.read_text(encoding="utf-8"))
    note = re.search(r"print\(_qwen_edit_render_batch_note\(cfg\)\)", code)
    cost = re.search(r"print\(format_cost_line\(decision\)\)", code)
    assert note is not None and cost is not None and note.start() < cost.start()
    assert not re.search(r"decision\s*=\s*[^\n]*render_batch", code), (
        "the render-batch estimate must never become the guardrail's basis: deriving hours from it "
        "would need a seconds-per-step figure nobody has measured"
    )


def test_the_stage_gates_the_architecture_before_it_quantizes_injects_or_renders() -> None:
    """Body order is the recipe's, and the arch gate is unconditional and FIRST among the loads."""
    stage = _sample_stage_code()
    order = [
        "run_qwen_edit_arch_gate(",
        "quantize_qwen_edit(",
        "inject_lora(",
        "build_qwen_edit_pipeline(",
        "render_qwen_edit_sample(",
        "write_qwen_edit_gallery(",
    ]
    positions = []
    for token in order:
        index = stage.find(token)
        assert index != -1, f"qwen_edit_sample must call {token}"
        positions.append(index)
    assert positions == sorted(positions), (
        f"the stage's order is wrong: {list(zip(order, positions))}. quantize_qwen_edit runs on the "
        "UN-WRAPPED transformer (assert_qwen_edit_not_peft_wrapped enforces it), and the arch gate "
        "precedes every load because a cost line is only truthful if the function it prices always "
        "does the same work."
    )
    # The adapter is loaded into the ALREADY-BUILT pipeline's transformer, once per band member —
    # never before the wrapper exists, and never a second transformer.
    load = stage.index("load_adapter_into(adapted,")
    assert stage.index("build_qwen_edit_pipeline(") < load < stage.index(
        "for column in (col for col in columns if col.checkpoint == member)"
    )


def test_the_stage_commits_the_volume_and_resumes_what_already_landed() -> None:
    """commit-or-vanish per image + skip-if-non-empty — the h3_sample per-clip precedent."""
    stage = _sample_stage_code()
    assert stage.count("checkpoints_vol.commit()") >= 2, (
        "a single commit at the end loses every rendered cell to a preemption; h3_sample commits "
        "per clip for exactly this reason"
    )
    assert "checkpoints_vol.reload()" in stage, "the band is resolved against a re-read Volume"
    assert "st_size > 0" in stage, (
        "resume must require a NON-EMPTY file: a container killed mid-save leaves a 0-byte image, "
        "and skipping that would put a corrupt cell in the grid"
    )


def test_the_stage_never_passes_guidance_scale_and_never_reads_the_unread_fields() -> None:
    """The single easiest way to be wrong on this family, pinned as source.

    ``guidance_scale=`` is ignored by this checkpoint (no guidance embedder), so routing §8's 4.0
    there renders the whole grid at CFG 1.0 — muddy, and indistinguishable from a bad adapter.
    """
    stage = _sample_stage_code()
    assert "guidance_scale" not in stage, (
        "qwen_edit_sample must not name guidance_scale at all: the recipe's true_cfg lives in "
        "QWEN_EDIT_RENDER_RECIPE and the render takes it from there"
    )
    assert "validation.prompts" not in stage, (
        "§8's A/B pair travels with its held-out input; refusing on validation.prompts would refuse "
        "on a field this grid never reads"
    )
    # ``.find_latest(`` — the CALL. The name still appears inside the band refusal's message, which
    # is where it belongs: the refusal explains why the fallback is not taken.
    assert re.search(r"\.find_latest\s*\(", stage) is None, (
        "the band is DECLARED; find_latest is a moving target while a training run commits "
        "(D-10-DEF-19) and would land every re-dispatch in a fresh render dir"
    )


# ==================================================================================================
# Behavioural — the real main(), with the dispatch verb replaced by a recorder.
# ==================================================================================================


class _RecordingCall:
    object_id = "fc-test"

    def get(self, timeout=None):  # noqa: ANN001, ARG002 — the FunctionCall surface main() touches
        return None


class _RecordingFn:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def with_options(self, **_kwargs):
        return self

    def spawn(self, *args, **kwargs) -> _RecordingCall:
        self.calls.append((args, kwargs))
        return _RecordingCall()


def _sample_harness(monkeypatch):
    """``(raw_main, recorder)`` with the heavy dry-run gate skipped and the dispatch verb recorded."""
    from signet_trainer.modal import entrypoint, fns

    monkeypatch.setattr(entrypoint, "run_dryrun", lambda cfg: 0)
    recorder = _RecordingFn()
    monkeypatch.setattr(fns, "qwen_edit_sample", recorder)
    return entrypoint.main.info.raw_f, recorder


def test_a_sample_run_without_approve_stops_at_the_pause_and_dispatches_nothing(monkeypatch) -> None:
    raw_main, recorder = _sample_harness(monkeypatch)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: (_ for _ in ()).throw(EOFError()))

    with pytest.raises(SystemExit):
        raw_main(config=str(_EXAMPLE_CONFIG), approve=False, mode="sample")

    assert recorder.calls == [], "MODL-02: a declined sample run may never dispatch"


def test_an_approved_sample_run_still_refuses_on_the_gaps_before_dispatching(monkeypatch) -> None:
    """The gap refusal is what stands between an approved-but-undeclared config and a metered A100."""
    raw_main, recorder = _sample_harness(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        raw_main(config=str(_EXAMPLE_CONFIG), approve=True, mode="sample")

    assert recorder.calls == [], "a gap refusal must abort BEFORE the dispatch — nothing spent"
    message = str(excinfo.value)
    assert "render_checkpoint_band" in message and "render_inputs" in message


def test_an_approved_sample_run_dispatches_exactly_once_when_the_gaps_clear(monkeypatch) -> None:
    """With the declared request in place the arm dispatches ONE spawn, carrying the config TEXT."""
    from signet_trainer.modal import entrypoint

    raw_main, recorder = _sample_harness(monkeypatch)
    # The two fields are DECLARED GAPS in the schema, so a YAML file cannot carry them yet; clearing
    # the gap function is the narrowest possible stand-in for the config that will.
    monkeypatch.setattr(entrypoint, "_qwen_edit_config_gaps", lambda cfg, *, mode: [])
    # main() now books the dispatch into the cumulative session-spend ledger (D-8-YOLOCAP, issue
    # #37 finding 1/6) via entrypoint.append_spend. _EXAMPLE_CONFIG is the real shared example
    # config and does not (and must not, per its own scope) declare a tmp session_spend_ledger_path,
    # so neutralize append_spend here rather than write the default project-relative ledger path
    # onto the real filesystem as a side effect of this test.
    monkeypatch.setattr(entrypoint, "append_spend", lambda *a, **k: None)

    raw_main(config=str(_EXAMPLE_CONFIG), approve=True, mode="sample")

    assert len(recorder.calls) == 1, "an approved, gap-free sample run dispatches exactly once"
    args, kwargs = recorder.calls[0]
    assert kwargs == {} and len(args) == 1
    assert "family: qwen_edit" in args[0], (
        "the recipe crosses BY VALUE (YAML text) — the configs/ dir is not in the container image, "
        "and the stage re-parses and re-validates it before any spend"
    )


# ==================================================================================================
# The render floor — read off real PEFT weights, on the CPU.
# ==================================================================================================


def test_the_adapter_floor_separates_a_loaded_adapter_from_an_injected_one() -> None:
    """PEFT initialises ``lora_B`` to ZERO, which is what an adapter that never loaded looks like.

    The refusal this feeds is the degenerate half of §8's convergence read: a band member whose
    weights did not arrive renders pixel-identical to the base column under a checkpoint's label,
    and the grid then reports 'not converging' for a checkpoint that never participated.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")

    from signet_trainer.lora.peft import build_lora_config, inject_lora
    from signet_trainer.modal.fns import _qwen_edit_adapter_is_live

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):  # noqa: ANN001, ANN201 — required by nn.Module, never called here
            return self.proj(x)

    model = inject_lora(_Tiny(), build_lora_config(rank=4, alpha=4, dropout=0.0, targets=["proj"]))

    total, live = _qwen_edit_adapter_is_live(model)
    assert total == 1 and live == 0, (
        "a freshly injected adapter must read as NOT live — every lora_B is zero, so it is the "
        "identity and its render would be the base render"
    )

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(1.0)
    total, live = _qwen_edit_adapter_is_live(model)
    assert (total, live) == (1, 1)


def test_the_stage_refuses_a_band_member_whose_adapter_is_all_zeros() -> None:
    """The floor is wired into the render loop, before the member's own cells are paid for."""
    stage = _sample_stage_code()
    floor = stage.find("_qwen_edit_adapter_is_live(")
    render = stage.find("render_qwen_edit_sample(")
    assert floor != -1 and "live == 0" in stage, "the floor must be checked, not merely computed"
    assert floor < stage.index("for column in (col for col in columns if col.checkpoint == member)"), (
        "the floor must run BEFORE the member's columns render, not after — its whole value is "
        "refusing a dead adapter cheaply"
    )
    assert render != -1
