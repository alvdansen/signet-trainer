"""Phase 10 (10-11) — the ``h3_train`` / ``h3_sample`` stage bodies, pinned by a source scan.

Scans ``src/signet_trainer/modal/fns.py`` as TEXT with comments and docstrings stripped first, and
falls back to ``ast`` wherever a substring would be satisfiable by accident. This is the convention
``tests/test_h3_preprocess_wiring.py`` (10-10) established, applied to the two stages 10-11 adds.

**This file never imports ``fns.py``.** Importing it builds the Modal app graph and eagerly resolves
every ``Secret.from_name`` — a CI failure that has nothing to do with what is being asserted.

Why these properties are asserted structurally rather than behaviourally: every one of them fails
EXPENSIVELY rather than loudly. A gate that runs after the inject spends before it aborts. An unset
``max_packed_rows`` OOMs a metered A100 forty minutes in instead of raising on the first batch. A
stubbed training stage passes every other structural test in the repo. None of these has a cheap
runtime assertion, and the expensive one costs an A100 container.

CPU-only, zero GPU, zero Modal spend.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan.

    Load-bearing here: both new stage docstrings NAME the things they must not do ("no guidance
    scale", "the offloader stays INERT", "do NOT extend inference/sampler.py"), so an un-stripped
    scan would pass on the warnings alone while the code did the opposite.
    """
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"#.*", "", src)


def _code() -> str:
    return _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))


def _function_source(code: str, name: str) -> str:
    """Slice ONE top-level function's stripped source so an assertion can be scoped to it."""
    match = re.search(rf"^def {re.escape(name)}\(", code, re.M)
    assert match, f"{name}() not found in modal/fns.py"
    tail = re.search(r"^(?:def |class |@)", code[match.end() :], re.M)
    end = match.end() + tail.start() if tail else len(code)
    return code[match.start() : end]


def _fn_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found as a top-level def in modal/fns.py")


def _decorator_source(name: str) -> str:
    """The stage's ``@app.function(...)`` block, unstripped (the kwargs are the assertion)."""
    node = _fn_node(name)
    assert node.decorator_list, f"{name} carries no decorator"
    return ast.unparse(node.decorator_list[0])


def _called_names(name: str) -> set[str]:
    """Every function NAME actually invoked inside ``name``'s body, via ``ast``.

    Stripping comments and docstrings is not enough for a call assertion: this file's own log
    messages legitimately mention the helpers they report on, so an f-string literal like
    ``f"max_packed_rows_for_budget({...})"`` satisfies a substring scan while the real call is gone.
    That is the same class of false-green the comment stripping exists to prevent, one layer down —
    so a "this function is CALLED" claim is asserted on the AST, never on text.
    """
    called: set[str] = set()
    for node in ast.walk(_fn_node(name)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    return called


def _call_keywords(name: str, callee: str) -> set[str]:
    """The keyword-argument names passed to ``callee`` anywhere inside ``name``'s body."""
    keywords: set[str] = set()
    for node in ast.walk(_fn_node(name)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        matched = (isinstance(func, ast.Name) and func.id == callee) or (
            isinstance(func, ast.Attribute) and func.attr == callee
        )
        if matched:
            keywords.update(kw.arg for kw in node.keywords if kw.arg)
    return keywords


@pytest.fixture(scope="module")
def code() -> str:
    return _code()


@pytest.fixture(scope="module")
def train_body(code: str) -> str:
    return _function_source(code, "h3_train")


@pytest.fixture(scope="module")
def sample_body(code: str) -> str:
    return _function_source(code, "h3_sample")


# ==================================================================================================
# 1. h3_train — the order that decides whether an abort is cheap or expensive
# ==================================================================================================


def test_the_cold_path_probe_precedes_the_gate_which_precedes_inject_which_precedes_the_loop(
    train_body: str,
) -> None:
    """probe -> run_h3_arch_gate -> inject_lora -> train_loop, in that order.

    enochiatron's equivalent gate caught SIX architecture mismatches in one ~$1.40 run. It only did
    that because it ran before the spend: a gate placed after the inject has already paid for the
    61.7 GiB load and the PEFT wrap it was supposed to prevent.
    """
    probe = train_body.index("import bitsandbytes")
    gate = train_body.index("run_h3_arch_gate(")
    inject = train_body.index("inject_lora(")
    loop = train_body.index("train_loop(")
    assert probe < gate < inject < loop, (
        f"h3_train body order is probe={probe} gate={gate} inject={inject} loop={loop}; the arch "
        "gate must abort BEFORE any load or injection spend."
    )


def test_the_gate_is_the_shared_helper_not_a_second_copy_of_the_assertions(
    train_body: str,
) -> None:
    """The arch assertions live in ONE place. A per-stage copy drifts, and the copy that drifts is
    the one nobody re-reads."""
    assert "run_h3_arch_gate(" in train_body
    for restated in ("assert_h3_arch(", "summarize_h3_transformer(", "check_lora_targets_regex("):
        assert restated not in train_body, (
            f"h3_train re-implements {restated} inline instead of calling the shared "
            "run_h3_arch_gate helper — two sources of truth for the same architecture contract."
        )


def test_the_gate_is_not_asked_to_release_the_model_it_must_hand_back(train_body: str) -> None:
    """``release=True`` alongside a caller that needs the model raises by design (10-10).

    ``h3_train`` trains the very transformer the gate proved; asking the gate to drop it would
    force a SECOND 61.7 GiB load, or train a model the gate never inspected.
    """
    assert "release=True" not in train_body


def test_max_packed_rows_for_budget_is_called_so_the_ceiling_assertion_is_ARMED() -> None:
    """An unset ``max_packed_rows`` defaults to ``None``, which SILENTLY disables the guard.

    ``build_h3_packed_batch``'s realized-``seq_len`` check is the belt to the config-load braces: it
    fires on what the dataloader ACTUALLY served. Without it a dataset that drifted away from the
    validated reference set OOMs a metered A100 forty minutes in rather than raising attributably on
    the first batch. Its absence must therefore be a RED test, not a quiet regression.

    Asserted on the AST: ``h3_train`` PRINTS the budget inputs, and that log message mentions the
    helper by name, so a text scan stays green with the real call deleted (verified — the mutation
    was not caught until this became an AST assertion).
    """
    assert "max_packed_rows_for_budget" in _called_names("h3_train"), (
        "h3_train must CALL max_packed_rows_for_budget — importing it, or naming it in a log "
        "message, leaves the runtime ceiling assertion disabled at its None default."
    )
    assert "max_packed_rows" in _call_keywords("h3_train", "H3RefStrategy"), (
        "the computed ceiling must reach H3RefStrategy as max_packed_rows=; unset it is None and "
        "build_h3_packed_batch's realized-seq_len guard never fires."
    )


def test_the_ceiling_is_computed_in_container_from_config_never_from_a_literal(
    train_body: str,
) -> None:
    """D-NOHARDCODE: the three budget inputs are config fields, so an H200 is a YAML edit."""
    flat = train_body.replace(" ", "").replace("\n", "")
    for field in ("config.h3.gpu_usable_gib", "config.h3.resident_gib", "config.h3.mib_per_packed_row"):
        assert field.replace(" ", "") in flat, f"{field} must feed the ceiling, not a literal"


def test_the_config_arrives_by_value_and_is_revalidated_in_container(train_body: str) -> None:
    """``configs/`` is not in the image, so a PATH would not resolve; and revalidating re-fires the
    frame-law and seq-len-budget checks inside the paid container too (T-03-63)."""
    assert "load_config_from_text(config_yaml)" in train_body


def test_the_h3_lora_targets_come_from_resolved_lora_targets(train_body: str) -> None:
    """``cfg.lora.target_modules`` may be a bare ``str`` on the H3 family (the path regex).

    The LTX idiom ``config.lora.target_modules or P1_FF_LORA_TARGETS`` keeps a regex intact but its
    list-shaped fallback would silently inject the LTX suffix set, matching ~104 wrong modules with
    nothing raising until the loss curve looks odd.
    """
    assert "resolved_lora_targets" in train_body
    assert "P1_FF_LORA_TARGETS" not in train_body


def test_the_target_count_300_is_named_where_an_operator_will_read_it(code: str) -> None:
    """The gate raises naming the MEASURED 300 (50 layers x 6 leaves) — the number that makes
    "4% of the adapter would train the token refiner" concrete at 3am."""
    gate_body = _function_source(code, "run_h3_arch_gate")
    assert "RuntimeError" in gate_body
    assert "300" in gate_body


def test_checkpoints_are_committed_because_done_is_the_file_on_the_volume(train_body: str) -> None:
    """Commit-or-vanish (Pitfall 3): "done" is the checkpoint on the Volume, never a log line —
    Modal prints the log line early on SIGTERM."""
    assert "checkpoints_vol.commit()" in train_body


def test_the_checkpoints_volume_is_threaded_into_the_loop_for_commit_per_save(
    train_body: str,
) -> None:
    """The loop's save -> commit -> callback -> commit cadence only applies if the Volume reaches
    it; a crash then loses at most one interval."""
    loop_call = train_body[train_body.index("train_loop(") :]
    assert "checkpoints_vol" in loop_call[: loop_call.index(")\n")] or "checkpoints_vol," in loop_call


def test_a_zero_adapter_delta_raises(train_body: str) -> None:
    """D-10-SCOPEGUARD's acceptance criterion, automated: the adapter must provably move the model.

    ``lora_B`` is zero-init, so a trained adapter that leaves the velocity untouched is a failed run
    no matter how ordinary the loss curve looked.
    """
    assert "h3_adapter_delta(" in train_body
    assert "delta == 0.0" in train_body, (
        "h3_train must compare the measured delta against exactly 0.0 — a tolerance here would "
        "quietly accept an adapter that moves the model by a rounding error"
    )
    raise_idx = train_body.index("delta == 0.0")
    assert "RuntimeError" in train_body[raise_idx : raise_idx + 400]


def test_the_loop_is_REUSED_not_forked(train_body: str, code: str) -> None:
    """The plan's key link. ``h3_train`` supplies a forward; the cadence is the loop's."""
    assert "from signet_trainer.train.loop import" in code
    assert "train_loop(" in train_body
    assert "step_fn=" in train_body, (
        "h3_train must inject its forward through train_loop's step_fn seam rather than copying the "
        "resume / accumulate / clip / save -> commit -> callback -> commit cadence"
    )
    for forked in ("while global_step", "ckpt_manager.save(", "optimizer.step()"):
        assert forked not in train_body, (
            f"h3_train contains {forked!r} — the loop body is being re-implemented, not reused."
        )


def test_the_strategy_is_constructed_and_the_h3_ref_key_link_holds(train_body: str, code: str) -> None:
    assert "H3RefStrategy" in _called_names("h3_train")
    assert "from signet_trainer.conditioning.h3_ref import" in code
    # Every H3Config field the strategy mirrors must actually be threaded — a missing one silently
    # falls back to the constructor default, i.e. the YAML says one thing and the run does another.
    threaded = _call_keywords("h3_train", "H3RefStrategy")
    for field in (
        "references_per_sample",
        "reference_dropout",
        "reference_pair_seed",
        "environment_ref_last",
        "audio_in_loss",
        "t_visual_cond",
    ):
        assert field in threaded, f"H3RefStrategy is constructed without {field}= (D-NOHARDCODE)"


def test_the_offloader_stays_inert_and_no_in_loop_sampling_is_added(train_body: str) -> None:
    """Two deliberate omissions, asserted so a later "improvement" cannot add them silently.

    blocks_to_swap: the measured Phase-10 geometry (12,362 rows, 76.36 GiB) fits ONE A100 at 0, and
    reaching block-swap at H3 campaign geometry would mean swapping ~38 of 50 blocks.
    in-loop sampling: it doubles the residency risk on a 61.7 GiB model, and Phase 10's acceptance
    signal is the delta plus the separately-gated ``--mode sample``.
    """
    assert "blocks_to_swap" not in train_body
    assert "BlockSwapOffloader" not in train_body
    assert "on_checkpoint=" not in train_body
    assert "run_sampler(" not in train_body


def test_keep_checkpoints_is_read_from_config_and_never_forced_to_a_finite_value(
    train_body: str,
) -> None:
    """HARD house rule: never auto-prune. A finite ``keep_n`` silently deletes intermediates, and in
    a research lab the intermediates ARE the artifacts."""
    assert "config.training.keep_checkpoints" in train_body


# ==================================================================================================
# 2. The anti-stub guard — the failure mode this phase most needs to avoid
# ==================================================================================================


@pytest.mark.parametrize("marker", ["v1", "for now", "placeholder", "TODO", "stub"])
def test_the_train_body_carries_no_provisional_marker(train_body: str, marker: str) -> None:
    """Phase 10 ships a TOOL. A stubbed training stage would pass every other structural test in
    this file — the order scan, the commit scan, the delta scan — while training nothing."""
    assert marker.lower() not in train_body.lower(), (
        f"h3_train's stripped body contains {marker!r}. Phase 10 does not ship a provisional "
        "training stage; a stub here is invisible to every other assertion in this file."
    )


@pytest.mark.parametrize("marker", ["v1", "for now", "placeholder", "TODO", "stub"])
def test_the_sample_body_carries_no_provisional_marker(sample_body: str, marker: str) -> None:
    assert marker.lower() not in sample_body.lower()


def test_neither_body_raises_not_implemented(train_body: str, sample_body: str) -> None:
    for body in (train_body, sample_body):
        assert "NotImplementedError" not in body


# ==================================================================================================
# 3. h3_sample — residency, checkpoint resolution, and the LTX concepts that must NOT appear
# ==================================================================================================


def test_the_sample_body_establishes_the_residency_discipline_before_it_renders(
    sample_body: str,
) -> None:
    """The 62.1 GiB conditioner and the 61.7 GiB transformer cannot coexist on one A100-80GB.

    Either mechanism satisfies this: an explicit ``free_text_encoder`` (the LTX/Gemma pattern) or
    diffusers' own ``enable_auto_cpu_offload``, which moves each component on and off the
    accelerator in turn. What must NOT happen is a render with neither.
    """
    mechanisms = [m for m in ("enable_auto_cpu_offload", "free_text_encoder") if m in sample_body]
    assert mechanisms, (
        "h3_sample renders without any VRAM residency mechanism — the H3 arithmetic is tighter "
        "than LTX's and this is mandatory, not optional."
    )
    first_render = sample_body.index("def _render")
    assert min(sample_body.index(m) for m in mechanisms) < first_render, (
        "the residency mechanism must be established BEFORE the render path is defined/called."
    )


def test_the_sample_body_resolves_the_checkpoint_with_find_latest(sample_body: str) -> None:
    """⚠ signet finals ARE step-numbered — ``CheckpointManager`` writes per-STEP dirs and
    ``find_latest`` is the only correct resolution. A flat ``lora_weights.safetensors`` is the
    CANONICAL ltx-trainer convention, NOT this stack's; looking for one silently renders a
    base-only grid and calls it a comparison."""
    assert "find_latest()" in sample_body
    assert "lora_weights.safetensors" not in sample_body


def test_the_sample_body_reuses_the_existing_gallery_writer_and_commits(sample_body: str) -> None:
    assert "write_comparison_gallery(" in sample_body
    assert "checkpoints_vol.commit()" in sample_body


def test_the_gallery_key_link_holds(code: str) -> None:
    assert "from signet_trainer.inference.grid import" in code


def test_the_sample_body_selects_its_reference_row_by_condition_not_by_index() -> None:
    """``rows[0]`` was hardcoded, so every eval prompt conditioned on ONE reference pair.

    The selector is config-driven and names the reference CONDITION by ``subject_id``. Asserted on
    the AST because "which helper is CALLED" is a structural question and this stage's own log line
    legitimately mentions the field name in an f-string.
    """
    called = _called_names("h3_sample")
    assert "_h3_select_reference_row" in called, (
        "h3_sample must go through the selector — a hardcoded rows[0] gives the whole eval one "
        "reference condition, which is the thinnest possible axis for a ref2v phase"
    )
    assert "_h3_resolve_references" not in called, (
        "h3_sample must not call the resolver directly any more: the selector owns row choice AND "
        "delegates resolution to the same _h3_resolve_references the pre-encode used, so the "
        "rotation and D-10-REFORDER cannot be re-implemented in two places"
    )
    keywords = _call_keywords("h3_sample", "_h3_select_reference_row")
    assert {"subject_ids", "references_per_sample", "reference_pair_seed", "environment_ref_last"} <= (
        keywords
    ), (
        f"the selector must be handed the wanted condition AND the three rotation/order parameters "
        f"the pre-encode used, so a render cannot resolve slots by a different rule than the cache "
        f"was built with; got {sorted(keywords)}"
    )

    body = _function_source(_code(), "h3_sample")
    assert "config.validation.reference_subject_ids" in body, (
        "the wanted condition is a CONFIG value (D-NOHARDCODE) — the render is a probe and which "
        "reference it probes is the operator's decision"
    )
    assert not re.search(r"\brows\[0\]", body), (
        "no hardcoded rows[0] may remain: the default lives in the selector, where its "
        "backwards-compatibility is tested behaviourally"
    )


def test_the_sample_body_never_names_a_reference_by_filename(sample_body: str) -> None:
    """⛔ Client-property hygiene: logs and delta.json carry subject_ids, never file names."""
    assert "subject_id" in sample_body
    assert not re.search(r"\.name\b.*reference|reference.*\.name\b", sample_body), (
        "a reference must never be logged by filename — real reference filenames are client "
        "property and must not reach a log, a paste or an artifact"
    )


def test_the_pipeline_reference_objects_do_not_shadow_the_descriptors(sample_body: str) -> None:
    """[Rule 1] ``delta.json`` reads ``subject_id`` AFTER the pipeline objects are built.

    Rebinding the descriptor list to ``MiniMaxH3ImageReference`` values (as this stage used to) made
    that read a TypeError at the very END of the render — after BOTH metered columns were paid for.
    """
    assert "reference_descriptors" in sample_body
    build_at = sample_body.index("MiniMaxH3ImageReference.from_file")
    assert not re.search(
        r"^\s*references\s*=\s*_h3_", sample_body[:build_at], re.M
    ), "the descriptor list must not be bound to the same name the pipeline objects take"
    tail = sample_body[build_at:]
    assert "for r in references]" not in tail, (
        "nothing after the pipeline objects are built may treat `references` as descriptors"
    )


def test_the_sample_body_does_not_import_the_ltx_sampler(sample_body: str) -> None:
    """``inference/sampler.py`` is ltx-trainer's ValidationSampler + STG + the two-stage upscaler —
    all LTX-only concepts. H3 has no negative branch, no STG and no upscaler; importing it here
    would drag in machinery with no H3 meaning."""
    assert "inference.sampler" not in sample_body
    assert "run_sampler" not in sample_body
    assert "build_generation_config" not in sample_body
    assert "run_two_stage" not in sample_body


def test_the_sample_body_passes_no_guidance_scale_and_no_negative_prompt(sample_body: str) -> None:
    """H3 is guidance-distilled and single-pass: ONE forward per step, no negative branch. A
    ``guidance_scale`` here would either be ignored or would be a different model's parameter."""
    assert "guidance_scale" not in sample_body
    assert "negative_prompt" not in sample_body


def test_the_sample_body_measures_and_prints_the_delta(sample_body: str) -> None:
    """The automated floor stands even if the operator has not looked at the grid yet."""
    assert "h3_adapter_delta(" in sample_body
    assert "delta == 0.0" in sample_body


def test_the_sample_body_renders_both_columns_at_the_same_seed(sample_body: str) -> None:
    """One ``seed`` variable feeds every render; a per-column seed would make the two columns
    incomparable and the grid meaningless."""
    assert "manual_seed(seed)" in sample_body
    assert sample_body.count("manual_seed(") == 1, (
        "both columns must draw from the SAME seed expression — a second one is a second schedule"
    )


# ==================================================================================================
# 4. h3_adapter_delta — the acceptance signal itself
# ==================================================================================================


def test_the_delta_helper_is_a_plain_module_level_function_not_a_stage(code: str) -> None:
    """A decorated function is reachable as ``modal run -m signet_trainer.modal.fns::<name>``, which
    boots a metered A100 with no cost print and no approval pause (Phase 9 AUDIT finding #18 — an
    open defect, not a precedent). The delta helper stays a helper."""
    node = _fn_node("h3_adapter_delta")
    assert node.decorator_list == [], (
        "h3_adapter_delta must carry NO decorator — a second ungated entry point is exactly what "
        "10-10 refused to create for the arch gate."
    )


def test_the_delta_helper_uses_disable_adapter_and_no_grad(code: str) -> None:
    """``disable_adapter()`` means the base pass costs no second 61.7 GiB load (which would not fit
    anyway) and the comparison is exact rather than approximate."""
    body = _function_source(code, "h3_adapter_delta")
    assert "disable_adapter()" in body
    assert "torch.no_grad()" in body


def test_the_delta_helper_slices_off_the_conditioning_prefix(code: str) -> None:
    """The transformer returns conditioning rows UNMASKED by contract
    (``transformer_minimax_h3.py`` L44-50); the prefix is identical on both sides, so including it
    would only dilute the maximum."""
    body = _function_source(code, "h3_adapter_delta")
    assert "n_cond_video:" in body


def test_the_delta_helper_loads_no_second_model(code: str) -> None:
    body = _function_source(code, "h3_adapter_delta")
    assert "load_h3_transformer" not in body
    assert "from_pretrained" not in body


# ==================================================================================================
# 5. The two decorators
# ==================================================================================================


@pytest.mark.parametrize("stage", ["h3_train", "h3_sample"])
def test_both_stages_declare_the_h3_image_and_the_memory_request(stage: str) -> None:
    """A ``gpu=`` with the code-only default image boots an A100 and dies at ``import torch``; the
    ``memory=`` request is what the P10-1 probe needed for the 61.7 GiB load."""
    decorator = _decorator_source(stage)
    assert "image=h3_gpu_image" in decorator
    assert "memory=" in decorator
    # ``ast.unparse`` normalizes string quoting, so match the value rather than the literal.
    assert "gpu=" in decorator
    if stage == "h3_sample":
        # house audit (PR #51, HIGH): a 3-reference render leg is a known Qwen3-VL text-encode OOM
        # on an A100, so h3_sample's gpu reads SIGNET_H3_SAMPLE_GPU instead of a bare literal — the
        # constant's default is still "A100-80GB", pinned by name over in test_no_warm_gpu.py.
        assert "gpu=H3_SAMPLE_GPU" in decorator
    else:
        assert "A100-80GB" in decorator


def test_h3_train_declares_retries_and_h3_sample_does_not() -> None:
    """``h3_train`` resumes in-dir from the latest COMMITTED checkpoint and commits per save, so a
    retry can never see a half-written one. A RENDER is not resumable — a retry would silently
    re-do the whole thing rather than continue it."""
    assert "modal.Retries" in _decorator_source("h3_train")
    assert "modal.Retries" not in _decorator_source("h3_sample")


def test_h3_train_declares_single_use_containers() -> None:
    """A FRESH container per retry — Modal's canonical long-training shape.

    This was documented as PROSE in CLAUDE.md's Modal-patterns table ("fresh container per retry,
    resume from last Volume checkpoint. Critical for >24h or preemption") and set NOWHERE in
    ``src/``, which is exactly why it did not transfer. Pinned here as STRUCTURE so it cannot
    silently erode again.

    The companion half — that ``single_use_containers`` is a REAL kwarg of the INSTALLED
    ``modal.App.function`` rather than a literal that does nothing — is asserted in
    ``tests/test_entrypoint_timeout_and_prints.py``. It lives there because THIS module carries
    ``test_this_file_never_imports_modal`` (importing the app module resolves every
    ``Secret.from_name``), so the SDK-signature probe cannot be written in this file.
    """
    assert "single_use_containers=True" in _decorator_source("h3_train").replace(" ", ""), (
        "h3_train's @app.function must set single_use_containers=True — a preempted retry must get "
        "a FRESH container and resume from the last committed Volume checkpoint"
    )


def _decorator_retry_budget(name: str) -> int:
    """The ``max_retries`` int parsed out of ``name``'s ``modal.Retries(...)``, via ``ast``.

    Parsed rather than substring-matched so the assertion reads the number the decorator will
    actually construct, not a number that merely appears somewhere in the block.
    """
    for node in ast.walk(_fn_node(name).decorator_list[0]):
        if not isinstance(node, ast.Call):
            continue
        if not ast.unparse(node.func).endswith("Retries"):
            continue
        for kw in node.keywords:
            if kw.arg == "max_retries" and isinstance(kw.value, ast.Constant):
                return int(kw.value.value)
    raise AssertionError(f"{name} declares no modal.Retries(max_retries=<int>)")


#: Modal's SERVER-enforced ceiling on ``max_retries``. Measured 2026-08-06 on app
#: ``ap-8Gra2Yka1fs4pwMIh8AgLv``: a dispatch with ``max_retries=60`` is rejected at app init with
#: "Invalid function retries. Must specify number between 0 and 10".
#:
#: ⚠ ``modal/retries.py`` validates only ``max_retries >= 0``, so the CLIENT accepts any
#: non-negative int and this bound is INVISIBLE locally — which is exactly why it is pinned here.
#: A constructor round-trip is not sufficient evidence that a retry policy is dispatchable.
_MODAL_SERVER_MAX_RETRIES = 10


def test_h3_train_retry_budget_sits_at_modals_platform_ceiling() -> None:
    """The budget must be Modal's MAXIMUM — because the cadence-derived requirement is unreachable.

    **The requirement**, measured on ``ap-lEaWCrVX8efqNm9R5EEE1u``: 1 initial attempt + 3 retries =
    4 container lives produced 250 COMMITTED steps in ~1.6 h wall clock => **62.5 steps of net
    progress per container life**. That figure already absorbs both the 61.7 GiB reload and the
    <=50-step re-do since the last commit — it is measured, not modelled. Remaining work
    3000 - 250 = 2750 steps => 2750 / 62.5 = **44 container lives REQUIRED**.

    **The ceiling**: Modal allows at most 10 retries = 11 container lives. So the requirement CANNOT
    be met by this kwarg, and the honest encoding is to take the ceiling AND pin the shortfall.
    """
    observed_steps_per_container_life = 250 / 4  # 62.5 — measured, not modelled
    remaining_steps = 3000 - 250
    required_lives = int(remaining_steps / observed_steps_per_container_life)  # 44
    assert required_lives == 44, "the derivation drifted — re-derive before touching the assertion"

    budget = _decorator_retry_budget("h3_train")
    assert budget == _MODAL_SERVER_MAX_RETRIES, (
        f"h3_train's max_retries={budget} must be exactly {_MODAL_SERVER_MAX_RETRIES} — Modal's "
        "server ceiling. Lower wastes free resilience; higher is REJECTED at app init (the client "
        "does not validate it, so a too-large value fails the dispatch, not the test)."
    )


def test_the_preemption_shortfall_is_pinned_rather_than_forgotten() -> None:
    """⛔ D-10-DEF-16, OPEN: server-side retries CANNOT cover a 2750-step round at this cadence.

    11 available container lives (1 initial + 10 retries) x 62.5 net steps = ~687 steps, against
    2750 remaining. Closing that gap needs a DIFFERENT mechanism — a local re-dispatch supervisor,
    or shorter rounds — which is an operator decision and is deliberately NOT invented in code.

    This test exists so the gap cannot be quietly mistaken for closed by the presence of the
    ``max_retries`` fix. If a future change makes the platform ceiling actually sufficient (Modal
    raises the cap, or the cadence improves), THIS test fails and the shortfall note in
    ``fns.py`` + the DECISION-LOG must be retired deliberately rather than by silence.
    """
    available_lives = _MODAL_SERVER_MAX_RETRIES + 1  # + the initial attempt
    covered_steps = available_lives * (250 / 4)  # 62.5 net steps per life, measured
    remaining_steps = 3000 - 250
    assert covered_steps < remaining_steps, (
        "the preemption shortfall appears CLOSED — retire D-10-DEF-16 deliberately (fns.py comment "
        "+ DECISION-LOG + card) rather than letting this test rot into a false-green"
    )


@pytest.mark.parametrize("stage", ["h3_train", "h3_sample"])
def test_neither_stage_sets_a_warm_gpu_token(stage: str) -> None:
    decorator = _decorator_source(stage)
    assert "keep_warm" not in decorator
    assert "min_containers" not in decorator


@pytest.mark.parametrize("stage", ["h3_train", "h3_sample"])
def test_both_stages_mount_the_checkpoints_volume(stage: str) -> None:
    assert "CHECKPOINTS_MOUNT" in _decorator_source(stage)


# ==================================================================================================
# 6. No second ungated entry point
# ==================================================================================================


def test_the_two_new_stages_are_the_only_h3_app_functions_added(code: str) -> None:
    """Every H3 ``@app.function`` must be a real, entrypoint-reachable stage. A convenience
    ``h3_*_smoke`` would be launchable as ``modal run -m ...::fn`` — a metered A100 with no cost
    print and no approval pause."""
    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    h3_stages = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("h3_")
        and any("app.function" in ast.unparse(d) for d in node.decorator_list)
    }
    assert h3_stages == {"h3_preprocess", "h3_train", "h3_sample"}, (
        f"unexpected H3 stage set {sorted(h3_stages)} — a new @app.function is a new ungated "
        "launch path unless the entrypoint threads it."
    )
    # No smoke ENTRY POINT after h3_train (arch_smoke / h3_*_smoke / smoke_test...). The one
    # sanctioned appearance is the NO-REFERENCE ALPHA banner's "smoke-tested only" wording — a
    # runtime warning string, not a launch path — hence the lookahead.
    assert not re.search(r"smoke(?!-tested)", code[code.index("def h3_train") :], re.IGNORECASE)


def test_this_file_never_imports_modal() -> None:
    """Importing ``fns.py`` builds the app graph and resolves every ``Secret.from_name``."""
    src = Path(__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:import|from)\s+modal\b", src, re.M)
    assert not re.search(r"^\s*from\s+signet_trainer\.modal\.fns\b", src, re.M)
