"""The gated-launch ``@app.local_entrypoint`` (MODL-03). Runs LOCALLY — never auto-spends.

The launch seam the Phase-8 harness drives:

    preflight (load config)
      -> dry-run gate (signet_trainer.dryrun.shapes.run_dryrun)   [CONF-03 hard gate]
      -> cost estimate + guardrail PRINT                          [MODL-03, BEFORE any launch]
      -> BLOCKING approval-pause (_require_approval)              [MODL-02, Phase 3]
      -> train.spawn() ASYNC dispatch                             [Phase 3 — validated on real A100]
      -> bounded synchronous watch (_watch_dispatch)              [D-10-DEF-17]

The full sequence is wired and validated (Phase 3 first metered run): the blocking approval-pause
(MODL-02) sits strictly between the cost print and the dispatch, so a metered run can never
auto-launch. The config is shipped to the container BY VALUE (YAML text), which re-parses +
re-validates it and re-runs the architecture preflight gate before any sustained spend.

Anti-Pattern 6: ``import modal`` is allowed here (this is the modal/ package). The dry-run gate +
cost helpers it calls are Modal-agnostic.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from signet_trainer.conditioning.h3_geometry import max_packed_rows_for_budget
from signet_trainer.config.load import load_config_from_text
from signet_trainer.config.validators import validate_h3_resolution_bucket
from signet_trainer.dryrun.shapes import run_dryrun
from signet_trainer.modal.app import (
    APP_NAME,
    CHECKPOINTS_VOLUME_NAME,
    DATASET_VOLUME_NAME,
    HUGGINGFACE_SECRET_NAME,
    WANDB_SECRET_NAME,
    WEIGHTS_VOLUME_NAME,
    app,
)
from signet_trainer.modal.cost import format_cost_line, guardrail_check
from signet_trainer.modal.retry_policy import ARM_MAX_RETRIES, resolve_arm
from signet_trainer.modal.session_cap import append_spend, read_ledger, session_cap_check

# Register the @app.function stubs (train / preprocess / download_weights / sample / ...) on the app
# graph at MODULE-LOAD time. ``modal run -m signet_trainer.modal.entrypoint`` builds the app from what
# is imported when this module loads; without this side-effecting import the functions are never
# registered and ``train.spawn()`` fails with "Function has not been hydrated". The lazy re-import
# inside main() then resolves to this same registered function. Placement does NOT affect MODL-02:
# the ``.spawn()`` CALL still lives strictly after the approval pause in main().
from signet_trainer.modal import fns as _fns  # noqa: F401 — import side-effect registers app functions

# Issue #33 finding 3: the THIRD Secret.from_name(...) in the app graph (fuse's gated-adapter token),
# captured the SAME way as HUGGINGFACE_SECRET_NAME / WANDB_SECRET_NAME above — at fns.py's
# MODULE-IMPORT time, from the SIGNET_HF_GATED_SECRET_NAME env var. Needed here so step 1b below can
# cover it too (previously the one secret name with NO config field and NO guard coverage).
from signet_trainer.modal.fns import HF_GATED_SECRET_NAME


def _require_approval(approve: bool) -> bool:
    """Blocking explicit-approval gate (D-03 / MODL-02) — the single seam all metered runs pass.

    Returns True ONLY on an explicit go-ahead: either the ``--approve`` flag was passed (harness
    path) or the operator types ``approved`` at the interactive prompt. Anything else (including a
    non-interactive stdin with no flag) returns False -> the caller aborts with NO dispatch, so a
    metered run can never auto-launch (sponsor credits are metered; no surprise spend).
    """
    if approve:
        print("[signet-entrypoint] approval: --approve flag set -> authorized.")
        return True
    try:
        answer = input("[signet-entrypoint] Type 'approved' to authorize this metered run: ").strip()
    except EOFError:
        # Non-interactive (e.g. piped/CI) with no --approve flag -> never auto-spend.
        answer = ""
    if answer.lower() == "approved":
        print("[signet-entrypoint] approval: operator typed 'approved' -> authorized.")
        return True
    print("[signet-entrypoint] approval: DECLINED (no 'approved'/--approve) — aborting, no dispatch.")
    return False


def _detach_requested() -> bool:
    """True if this launch carries ``--detach`` OR modal's documented short alias ``-d`` (issue #33
    finding 2).

    ``_watch_dispatch`` used to string-match ONLY ``"--detach" in sys.argv`` — missing modal's own
    ``-d`` flag (``modal/cli/run.py``: ``@click.option("-d", "--detach", ...)``). A genuinely
    detached run launched as ``modal run -d -m signet_trainer.modal.entrypoint ...`` then printed a
    FALSE "--detach was NOT passed" warning at dispatch and, ``dispatch_watch_seconds`` later, told
    the operator to "treat this run as LOST and re-launch with --detach" — inverting the honesty fix
    (audit 2026-08-11) into a lie in exactly the case it exists to protect, and inviting a duplicate
    metered re-dispatch of an already-healthy run.

    Still an argv scan (KEPT deliberately — ``tests/test_dispatch_is_spawned.py`` asserts both
    ``"sys.argv"`` and ``"--detach"`` appear in this file's source), just one that also accepts a
    short flag CLUSTER carrying ``d`` (e.g. ``-d``), mirroring how getopt-style CLIs bundle single-
    dash flags. A long option (``--anything``) never matches the cluster branch.
    """
    return any(
        arg == "--detach" or (arg.startswith("-") and not arg.startswith("--") and "d" in arg[1:])
        for arg in sys.argv
    )


def _resolve_session_cap_usd(ledger_path: Path, config_default: float) -> float:
    """The WR-02 per-session cap override: SESSION-STATE.json's ``session_cap_usd``, else the
    config default (``cfg.modal.session_cap_usd``).

    Mirrors ``harness_state.read_ledger_figures``' resolution — the setup gate WRITES this key
    into the same ledger file ``training-run/SKILL.md`` §3 reads it from. Deliberately tolerant
    (never raises): a missing/corrupt/legacy ledger falls back to the documented config default
    rather than blocking a dispatch on a formatting problem that ``read_ledger`` (the actual spend
    reader, called separately below) already owns raising on.
    """
    try:
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return config_default
    if not isinstance(data, dict):
        return config_default
    override = data.get("session_cap_usd")
    if override is None:
        return config_default
    try:
        return float(override)
    except (TypeError, ValueError):
        return config_default


def _watch_dispatch(
    fc: object,
    watch_seconds: float,
    label: str,
    *,
    ledger_path: str | None = None,
    est_usd: float | None = None,
) -> None:
    """Post-dispatch handling for a SPAWNED (async) gated run — D-10-DEF-17.

    Every arm in ``main()`` now dispatches with ``.spawn(...)`` rather than ``.remote(...)``. The
    verb is the whole fix: ``_Function.remote`` delegates to ``_call_function``, which carries
    ``FUNCTION_CALL_INVOCATION_TYPE_SYNC``, and the invocation type is sent to the SERVER at call
    creation — Modal cancels in-flight SYNC inputs whose owning client disappears. On 2026-08-07 that
    killed two healthy runs (a training round at step 300 and a measurement render at denoise step
    13) at the moment the dispatching agent returned. ``_Function.spawn`` carries
    ``FUNCTION_CALL_INVOCATION_TYPE_ASYNC``, which is not cancelled that way.

    Five things happen here, in order:

    0. Print the ``FunctionCall`` id — the handle that OUTLIVES this client, and the whole point of
       the change: any later process can re-attach with ``modal.FunctionCall.from_id("<id>")``
       (``poll_function`` uses ``clear_on_success=False``, so a client that did not dispatch the call
       can still collect its output). Printed BEFORE the ledger booking below so a raising
       ``append_spend`` (ledger-lock timeout, corrupt-ledger shape) never costs the re-attach handle
       of an already-spawned run.
    1. Book the dispatch-time estimate into the cumulative session-spend ledger (D-8-YOLOCAP /
       issue #37 finding 1/6) — ``append_spend(ledger_path, est_usd, run_ref=fc.object_id)``, when
       both are given (every real caller in ``main()`` passes them; tests that stub the dispatch
       function may omit them). This is the code-side half of CR-01 airtight accounting: every
       approved dispatch (strict, yolo, blanket) must be recorded so ``read_ledger`` — and the
       cumulative cap check in ``main()`` — reflect REAL cumulative spend, not only the spend an
       agent remembered to append from prose. Booking the ``FunctionCall`` id (rather than no
       ``run_ref`` at all) makes the entry reconcilable against the actual Modal dispatch.
    2. Warn — advisory only, never a hard failure — when ``--detach`` is absent. ``.spawn()`` fixes
       the INPUT; ``--detach`` keeps the ephemeral app SHELL alive. Both are required and neither
       alone is sufficient, so this converts a doc-prose rule into a runtime check.
    3. Watch SYNCHRONOUSLY for a BOUNDED window (``cfg.modal.dispatch_watch_seconds``, config-first
       per D-NOHARDCODE) so cheap early aborts still surface at the console.
    4. On expiry, disengage and return normally. ⛔ Expiry CANCELS NOTHING — the call is async; this
       client simply stops watching. A remote exception raised INSIDE the window is the cheap early
       abort we deliberately keep, and it propagates LOUDLY (non-zero exit).

    ⚠ The ``except TimeoutError`` below is the BUILTIN. ``modal._functions`` imports only
    ``OutputExpiredError`` from ``modal.exception``, so ``FunctionCall.get(timeout=N)`` raises the
    builtin on expiry; ``modal.exception.TimeoutError`` is NOT a builtin-TimeoutError subclass, and
    catching it here would instead swallow ``OutputExpiredError``.
    """
    print(
        f"[signet-entrypoint] {label} dispatched ASYNC (spawn). FunctionCall id: {fc.object_id}\n"
        f"[signet-entrypoint]   re-attach from ANY later process (this id outlives this client):\n"
        f"[signet-entrypoint]   python -c \"import modal; "
        f"print(modal.FunctionCall.from_id('{fc.object_id}').get(timeout=1800))\""
    )
    if ledger_path is not None and est_usd is not None:
        append_spend(ledger_path, est_usd, run_ref=str(fc.object_id))
        print(
            f"[signet-entrypoint] ledger: booked ${est_usd:.2f} for {label} against {ledger_path} "
            f"(run_ref={fc.object_id})."
        )
    detached = _detach_requested()
    if not detached:
        print(
            "[signet-entrypoint] WARNING: --detach was NOT passed. .spawn() makes the INPUT async so "
            "the server will not cancel it when this client dies, but a NON-detached ephemeral app "
            "is still STOPPED when the client exits — which stops the call with it. Every documented "
            "launch form in this repo carries --detach; re-launch with it if this run must survive "
            "this process.",
            file=sys.stderr,
        )
    try:
        fc.get(timeout=watch_seconds)
    except TimeoutError:
        if detached:
            print(
                f"[signet-entrypoint] {label} still RUNNING after {watch_seconds:g}s — this client "
                f"is disengaging, the run is NOT cancelled (async dispatch). Track it via "
                f"FunctionCall id {fc.object_id}, `modal app logs <app-id>`, or the output Volume."
            )
        else:
            # Honesty fix (audit 2026-08-11): without --detach the ephemeral app STOPS when this
            # client exits and the spawned call is stopped with it — the old unconditional "NOT
            # cancelled" print was FALSE in exactly the case the startup warning above describes.
            print(
                f"[signet-entrypoint] {label} still RUNNING after {watch_seconds:g}s — this client "
                f"is disengaging WITHOUT --detach, so the ephemeral app (and this call, "
                f"FunctionCall id {fc.object_id}) will be STOPPED with it. Treat this run as LOST "
                "and re-launch with --detach.",
                file=sys.stderr,
            )
        return
    print(f"[signet-entrypoint] {label} COMPLETED inside the {watch_seconds:g}s watch window.")


def _parse_resolution_buckets(bucket_strings: list[str]) -> list[tuple[int, int, int]]:
    """Convert the config's WxHxF CLI strings to the (F, H, W) tuples fns.preprocess consumes.

    ``cfg.data.resolution_buckets`` carries WxHxF strings (e.g. ``"768x352x25"``); the canonical
    ``process_dataset.py`` encode (via ``fns.preprocess``) takes (F, H, W) tuples — the shape the
    retired ``scripts/_encode_demo*.py`` drivers pre-parsed by hand (``[(25, 352, 768), ...]``).
    This mirrors that pre-parse so the preprocess arm stays config-first (D-NOHARDCODE) instead of
    hardcoding bucket literals. ``"768x352x25"`` -> ``(25, 352, 768)``.
    """
    parsed: list[tuple[int, int, int]] = []
    for bucket in bucket_strings:
        w, h, f = (int(part) for part in bucket.split("x"))
        parsed.append((f, h, w))
    return parsed


def _reference_encode_params(cfg: object) -> tuple[str | None, int]:
    """ic_lora-only reference params for the preprocess arm; (None, 1) for every other mode.

    ``ConditioningConfig.reference_column`` defaults to ``"reference_path"`` but is documented
    "Only valid in mode == 'ic_lora'" — passing it through on a plain (``mode: none``) encode
    makes the canonical ``compute_latents`` raise ``Key 'reference_path' not found in JSONL
    entry`` AFTER the approval gate (a burned gate — first hit by the reference campaign's ``mode: none``
    encode, 09-07 T3). ``fns.preprocess`` treats ``reference_column=None`` as no-reference.
    """
    if cfg.conditioning.mode == "ic_lora":
        return cfg.conditioning.reference_column, cfg.conditioning.reference_downscale_factor
    return None, 1


def _audio_encode_params(cfg: object) -> bool:
    """a2v-only ``with_audio`` flag for the preprocess arm; False for every other mode.

    Mirrors ``_reference_encode_params`` / ``_mask_encode_params``' lean-threading shape:
    ``fns.preprocess`` treats ``with_audio=False`` as no-audio-encode, so every non-a2v encode stays
    byte-identical. For ``mode == "audio_to_video"`` the config's ``audio.with_audio`` drives it
    (config-first, D-NOHARDCODE) — and SignetConfig's cross-field guard already fail-fasts an a2v
    config that leaves it False, so a2v always reaches here with the flag set.
    """
    if cfg.conditioning.mode == "audio_to_video":
        return bool(cfg.audio.with_audio)
    return False


def _mask_encode_params(cfg: object) -> tuple[str | None, str]:
    """inpaint-only mask-encode params for the preprocess arm; (None, "video_masks") otherwise.

    Mirrors ``_reference_encode_params``' lean-threading shape (same burned-gate lesson, 09-07 T3):
    ``fns.preprocess`` treats ``mask_column=None`` as no-mask-encode, so every non-inpaint encode
    stays byte-identical. For ``mode == "inpaint"`` the manifest column is the upstream
    role-convention name ``"video_mask"`` and the output dir comes from the config
    (``conditioning.inpaint_mask_dir``, default ``"video_masks"`` — D-NOHARDCODE; the signet-native
    mask encode is data/mask_encode.py, the pin predates upstream mask support — GATE-SPEC rev 2).
    """
    if cfg.conditioning.mode == "inpaint":
        return "video_mask", cfg.conditioning.inpaint_mask_dir
    return None, "video_masks"


def _h3_encode_params(cfg: object) -> dict[str, object] | None:
    """EVERY required kwarg of ``fns.h3_preprocess`` for a ``family: h3`` config; ``None`` otherwise.

    Same lean-threading shape and the same burned-gate discipline as ``_reference_encode_params`` /
    ``_audio_encode_params`` / ``_mask_encode_params`` (09-07 T3): read ONLY fields that exist on the
    loaded config, and read them HERE — bound to a local, before the dispatch — never inline inside
    the ``.spawn(`` expression. ``SignetConfig`` is ``extra="forbid"``, so a field read that names
    the wrong block raises ``AttributeError`` AFTER the approval pause; a named abort at this point
    still costs $0 (``.spawn()`` never fires) but it tells the operator what to fix.

    ``h3_preprocess`` deliberately declares ALL 18 parameters REQUIRED with no defaults (10-10), so a
    threading gap is a ``TypeError`` at dispatch rather than a silent wrong default. That contract is
    only worth anything if the supply side actually keeps up with it, which is why
    ``tests/test_h3_entrypoint_gate.py`` re-derives BOTH sides from the real signatures and diffs
    them instead of restating either.

    ⛔ ``max_packed_rows`` is COMPUTED, never a literal. It is the value ``build_h3_packed_batch``'s
    runtime ceiling assertion (10-06) compares the REALIZED ``seq_len`` against — the THIRD layer of
    the budget defense, after the config-load worst-pair check (``validate_h3_reference_budget``) and
    the dry-run refusal. Unthreaded it defaults to ``None`` and that whole guard becomes dead code
    while every local gate still goes green; the ONLY remaining discovery channel would be an OOM in
    a metered A100 container. ``conditioning/h3_geometry.max_packed_rows_for_budget`` owns the
    arithmetic (D-NOHARDCODE: an H200 escalation is then a YAML edit to the budget triple PLUS
    ``h3.modal_gpu`` — the TRAIN-tier booking each of ``h3_preprocess`` / ``h3_train`` threads via
    ``.with_options(gpu=...)``; the config-load coherence guards refuse the triple, or the
    booking, edited alone).

    ``with_audio`` reads ``cfg.audio.with_audio`` — the one config field that means "the encode
    writes audio latents". For a ``family: h3`` config today it necessarily reads False: H3's
    reference control lives in the ``h3`` block, so ``conditioning.mode`` is not ``audio_to_video``
    and ``SignetConfig``'s reverse guard rejects a non-default audio block under any other mode. That
    matches D-10-AUDIO (0 of the 44 corpus clips carry an audio stream, measured) — and it is read
    from the config rather than written as ``False`` here so a future H3 audio leg is a schema
    decision, not a literal somebody has to find.
    """
    if cfg.model.family != "h3":
        return None
    for name in ("vae_id", "audio_vae_id"):
        if getattr(cfg.model, name) is None:
            raise SystemExit(
                f"[signet-entrypoint] model.{name} is unset on a family: h3 config. The H3 encode "
                f"loads the video VAE and the audio VAE as DIRECTORIES under WEIGHTS_DIR (e.g. "
                f"'minimax-h3/vae' / 'minimax-h3/audio_vae') and h3_preprocess requires both. "
                f"Aborting before any dispatch — nothing was spent."
            )
    return {
        # cfg.data — where the manifest is read from and where the four h3_* sources are written.
        "metadata_path": cfg.data.metadata_path,
        "output_dir": cfg.data.preprocessed_data_root,
        # geometry: training_dims is [W, H, F], so F is the H3 target frame count (17n+5).
        "target_frames": cfg.training_dims[2],
        # FRAME-COUNT BUCKETING. The declared buckets' F values, so a manifest row may name its
        # own `target_frames` and be refused if it is not one of them. training_dims F is the
        # DEFAULT only in SINGLE-bucket mode (`fns.py::_row_frames`) — under >1 bucket a row that
        # names none is refused outright, no silent default. SignetConfig guarantees training_dims
        # F is a member of this set either way, so it is always a legal explicit choice too.
        "frame_buckets": tuple(
            sorted({validate_h3_resolution_bucket(b)[2] for b in cfg.data.resolution_buckets})
        ),
        "target_aspect": cfg.h3.target_aspect,
        # cfg.h3 — the locked D-10-* recipe values, every one a documented field (D-NOHARDCODE).
        "reference_image_short_edge": cfg.h3.reference_image_short_edge,
        "reference_pair_seed": cfg.h3.reference_pair_seed,
        "references_per_sample": cfg.h3.references_per_sample,
        "environment_ref_last": cfg.h3.environment_ref_last,
        "text_encoder_layer": cfg.h3.text_encoder_layer,
        "with_audio": bool(cfg.audio.with_audio),
        "max_packed_rows": max_packed_rows_for_budget(
            cfg.h3.gpu_usable_gib, cfg.h3.resident_gib, cfg.h3.mib_per_packed_row
        ),
        # cfg.h3 durability knobs (10-14): the periodic-commit window and the resume/overwrite
        # switch — config-first so the loss-window/re-encode trade is a YAML edit, never a literal.
        "preprocess_commit_every": cfg.h3.preprocess_commit_every,
        "preprocess_overwrite": cfg.h3.preprocess_overwrite,
        # cfg.model — the four component DIRECTORIES under WEIGHTS_DIR (H3 IDs are dirs, not files).
        "model_id": cfg.model.model_id,
        "vae_id": cfg.model.vae_id,
        "audio_vae_id": cfg.model.audio_vae_id,
        "text_encoder_id": cfg.model.text_encoder_id,
    }


#: The signet-side modules each ``qwen_edit`` stage imports Modal-side, checked LOCALLY before the
#: dispatch so a gap costs $0 instead of a container boot. Mirrors ``modal/fns._qwen_edit_require_
#: backend``, which re-checks the SAME names in-container for the reason ``modal/app.py``'s
#: ``download_image`` INVARIANT banner states in one sentence: a passing local gate proves nothing
#: about the container's site-packages. The two are cheap and independent, and neither replaces the
#: other.
_QWEN_EDIT_STAGE_MODULES: dict[str, tuple[str, ...]] = {
    "preprocess": (
        "signet_trainer.models.qwen_edit_loader",
        "signet_trainer.prep.qwen_edit_encode",
    ),
    "train": (
        "signet_trainer.models.qwen_edit_loader",
        "signet_trainer.train.qwen_edit_step",
        "signet_trainer.train.family_hooks",
        # The 2x2 pack. The cache stores ``[C, F, H, W]`` latents and ``QwenEditStrategy`` REFUSES a
        # latent-form payload when ``pack_fn`` is None — deliberately, rather than transcribing the
        # pack a second time. Named here so the gap is a pre-dispatch abort, not a strategy raise
        # after the transformer is resident.
        "signet_trainer.conditioning.qwen_edit_packing",
    ),
    "sample": (
        "signet_trainer.inference.qwen_edit_layout",
        "signet_trainer.models.qwen_edit_loader",
        # LANDED (e132b30). Still named here, because the readiness list is a contract about what
        # this stage IMPORTS, not a to-do list — it is what turns a future deletion or rename into
        # a $0 abort instead of a ModuleNotFoundError inside a metered container. The render
        # assembles a ``QwenImageEditPlusPipeline`` whose scheduler carries the §8 STATIC
        # reparameterisation
        # value, and that construction lives OUTSIDE ``inference/`` because
        # ``tests/test_no_wan_params.py``'s ``_WAN_TOKENS`` bans the bare token from every ``*.py``
        # under that directory — a guard written for LTX paths whose directory-wide scope now also
        # covers a family where that setting is mandatory at a different value. Narrowing the guard
        # is a RULING on a shipped money-safe check; putting the builder in ``models/`` is the
        # reversible side of it and costs nothing. Until the module lands, a ``--mode sample``
        # dispatch on this family aborts HERE, at $0, instead of after a 40.9 GiB load.
        "signet_trainer.models.qwen_edit_pipeline",
    ),
}


def _qwen_edit_stage_readiness(mode: str) -> None:
    """Refuse pre-dispatch when a module the ``qwen_edit`` stage imports has not landed. Costs $0.

    Uses ``importlib.util.find_spec`` rather than a real import ON PURPOSE: it answers "does this
    module exist?" WITHOUT executing it, so a module whose body legitimately needs a container-only
    dependency (torch on CUDA, diffusers at the pinned SHA, ``optimum.quanto``) cannot produce a
    FALSE refusal here on a laptop. The signet convention already forbids that shape — every heavy
    import in ``models/qwen_edit_loader.py`` / ``prep/qwen_edit_encode.py`` is function-local for
    exactly this reason — but a readiness check that depends on everybody keeping a convention is a
    check that eventually reports the wrong answer.

    Called AFTER ``_require_approval`` and BEFORE ``.spawn(``, the same position and for the same
    reason as ``_h3_encode_params``: a named abort at this point still costs $0 (``.spawn()`` never
    fires) but it tells the operator exactly what to land.
    """
    import importlib.util  # noqa: PLC0415 — local; nothing else in this module needs importlib

    missing = [
        name
        for name in _QWEN_EDIT_STAGE_MODULES[mode]
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(
            f"[signet-entrypoint] the qwen_edit {mode!r} stage imports {missing}, which do not "
            f"exist in this tree. Family #3 landed as several independent slices and the Modal "
            f"wiring is the one that spends money, so an unlanded module aborts HERE — after the "
            f"approval pause, before any dispatch, at $0 — rather than as a ModuleNotFoundError "
            f"inside a metered container. Land the module(s) named above, then re-run this exact "
            f"command; nothing in the Modal layer needs to change."
        )


#: The two ``qwen_edit`` config fields a RENDER needs and the schema does not carry yet. Named once,
#: here, because three places must spell them identically — the gap message (which tells the operator
#: what to add), the render-batch cost note (which sizes the grid from them) and
#: ``modal/fns.py::qwen_edit_sample`` (which reads them in-container). A test cross-checks that the
#: Modal stage names the same two strings: a rename that missed one would produce a config the
#: entrypoint accepts and the container refuses, which is the burned gate this whole layer avoids.
#:
#: ⚠ There is deliberately NO field here for the §8 inference settings (steps / true_cfg / CFGNorm /
#: the static scheduler reparameterisation / LoRA strength / the negative prompt). Those are LOCKED
#: in ``models/qwen_edit_pipeline.QWEN_EDIT_RENDER_RECIPE`` — recipe terms in the same sense that
#: ``quantize_qwen_edit``'s qfloat8 is one — so the render reads them from there, not from the
#: config. What the config CAN do is contradict them, and gap (5) below refuses exactly that.
_QWEN_EDIT_BAND_FIELD = "render_checkpoint_band"
_QWEN_EDIT_INPUTS_FIELD = "render_inputs"


def _qwen_edit_render_request(cfg: object) -> dict[str, object]:
    """The DECLARED render request, read tolerantly so this works before AND after the fields land.

    ``getattr(..., default)`` rather than attribute access — the idiom the ``control_dirs`` gap
    already uses below. ``SignetConfig`` is ``extra="forbid"`` and these four fields do not exist on
    ``QwenEditConfig`` today, so a plain read would raise ``AttributeError`` instead of producing
    the actionable refusal that names them. The day they land as real fields every caller here
    starts seeing real values and NOTHING in this file changes, which is the property that makes a
    declared gap cheaper than a workaround.
    """
    qwen_edit = cfg.qwen_edit
    return {
        "band": tuple(getattr(qwen_edit, _QWEN_EDIT_BAND_FIELD, ()) or ()),
        "inputs": tuple(getattr(qwen_edit, _QWEN_EDIT_INPUTS_FIELD, ()) or ()),
    }


def _qwen_edit_render_batch_note(cfg: object) -> str:
    """Size the render grid from the config so the cost banner prints WORK, not just an estimate.

    ``cfg.modal.est_hours`` is a declared number — there is no Modal price API and no steps/second
    figure has been measured for Qwen-Image-Edit on any card in this program — and for a training
    run that is genuinely all there is. For a RENDER it is not: the work is fully determined before
    dispatch (``2 prompt modes x (1 base + band size) x held-out inputs`` images at the recipe's
    step count each), so the counts are arithmetic and the per-image budget the declared estimate
    implies is a real quotient. "8 h" cannot be sanity-checked by eye; "8 h / 12 images = 2400 s per
    image" can — and a band that quietly tripled the work still prints the same ``est_hours``.

    The step count comes from ``QWEN_EDIT_RENDER_RECIPE``, not from ``validation.num_inference_steps``,
    because that is what the render actually uses; pricing the batch off a field the sampler does not
    read would be a cost line for a render nobody performs. Gap (5) refuses the case where the two
    disagree, so the operator is never left comparing them by hand.

    The guardrail's BASIS is deliberately unchanged: this line is printed BESIDE
    ``format_cost_line``, never in place of it. Deriving hours from these counts would need a
    seconds-per-denoise-step figure nobody has measured, and inventing one to feed a money gate is
    the opposite of what the gate is for.

    Prints honestly when the grid cannot be sized — the missing declarations are the same ones
    ``_qwen_edit_config_gaps`` refuses on after the approval pause, so the operator sees the shape
    of the problem at the cost line and its remedy at the refusal.
    """
    from signet_trainer.inference.qwen_edit_layout import (  # noqa: PLC0415
        QWEN_EDIT_PROMPT_MODES,
    )
    from signet_trainer.modal.cost import (  # noqa: PLC0415
        format_render_batch_line,
        render_batch_estimate,
    )
    from signet_trainer.models.qwen_edit_pipeline import (  # noqa: PLC0415
        QWEN_EDIT_RENDER_RECIPE,
    )

    request = _qwen_edit_render_request(cfg)
    band, inputs = request["band"], request["inputs"]
    modes = len(QWEN_EDIT_PROMPT_MODES)
    steps = int(QWEN_EDIT_RENDER_RECIPE.steps)
    if not band or not inputs:
        return (
            f"[signet-cost] render batch: NOT SIZEABLE — {modes} prompt mode(s) and {steps} denoise "
            f"step(s) per image are known, but the checkpoint band ({len(band)} member(s) declared) "
            f"and the held-out input set ({len(inputs)} declared) are what multiply them. Both are "
            "DECLARED GAPS; the refusal after the approval pause names the fields "
            f"(qwen_edit.{_QWEN_EDIT_BAND_FIELD} / qwen_edit.{_QWEN_EDIT_INPUTS_FIELD}). The "
            f"guardrail above is priced on the declared cfg.modal.est_hours="
            f"{cfg.modal.est_hours:g}, unchanged."
        )
    return format_render_batch_line(
        render_batch_estimate(
            band_members=len(band),
            prompt_modes=modes,
            held_out_inputs=len(inputs),
            steps_per_image=steps,
            est_hours=float(cfg.modal.est_hours),
        )
    )


def _qwen_edit_config_gaps(cfg: object, *, mode: str) -> list[str]:
    """Every DECLARED gap between a validated ``family: qwen_edit`` config and what the stage needs.

    A config that loads is not the same as a config that can drive a stage, and the difference is
    worth naming rather than discovering inside a paid container. Each entry is one sentence saying
    what is missing, why the stage needs it, and what would land it.

    Five gaps exist today — two shared with the encode legs, three that only ``sample`` can hit —
    and every one is a real finding about the SCHEMA, not about this file. Each names the field that
    would land it, so the check goes green on a schema edit with no change here.

    **(1) ``config_source`` for a single-file ``model_id`` — every mode.** The shipped configs now
    use the DIRECTORY form (``qwen-image-edit-2511/transformer``), which sidesteps this gap
    entirely. It still fires for a genuinely single-file ``model_id``, and
    ``models/qwen_edit_loader.load_qwen_edit_transformer`` documents that ``config_source`` is
    REQUIRED on that path: ``from_single_file`` with no ``config=`` calls ``fetch_diffusers_config``
    -> ``infer_diffusers_model_type``, which has **no Qwen branch at all** on the pinned diffusers and
    falls through to ``model_type = "v1"``, whose default is
    ``stable-diffusion-v1-5/stable-diffusion-v1-5``. diffusers would then reach the HUB for a Stable
    Diffusion v1.5 ``transformer/`` config from inside a metered container. ai-toolkit works around
    it by hard-coding the hub id ``"Qwen/Qwen-Image"``; signet takes the value from the caller so an
    air-gapped Volume-local directory is expressible and ``local_files_only`` can stay on.

    The field that carries it is ``model.pipeline_root_id``. ``config/schema.py``'s
    ``_FAMILY_ONLY_MODEL_IDS`` USED to map it to ``frozenset({"h3"})`` so that setting it under
    ``family: qwen_edit`` was rejected at config load; ``"qwen_edit"`` was added to that entry
    in this branch, because the field now has a consumer on this family. **The remedy is
    therefore a CONFIG edit, not a schema edit** — point ``model.pipeline_root_id`` at the
    diffusers root on the weights Volume, or use a DIRECTORY-form ``model_id`` so the config
    travels with the weights. An operator who reads this text, opens schema.py expecting to make
    the map edit and finds it already made will conclude the refusal is stale rather than that
    their ``model_id`` is the problem — which is why this paragraph is maintained rather than
    left to rot.

    **(2) the control-image directories — ``preprocess`` only.**
    ``prep/qwen_edit_encode.resolve_qwen_edit_control_sources`` takes "the ORDERED config directory
    list", one directory PER SLOT, because the mapping is POSITIONAL: directory *i* fills slot *i*,
    which is what the caption's ``ctrl_img_{i+1}`` refers to. No block in ``SignetConfig`` declares
    it — ``DataConfig`` carries only ``metadata_path`` / ``preprocessed_data_root`` /
    ``resolution_buckets``, and ``QwenEditConfig`` carries the slot COUNT and the blank FILL but no
    paths. Inventing a convention here (``<data_root>/controls/slot_0`` or similar) is the one thing
    that must not happen: a guessed directory order silently re-points every caption's ``ctrl_img_N``
    and the sample trains against a request nobody wrote, at perfectly ordinary shapes and an
    perfectly ordinary loss curve. That is the exact failure ``resolve_qwen_edit_control_sources``
    was written to REFUSE, and it would be defeated by guessing one level up.

    **(3) the checkpoint BAND — ``sample`` only.** §8 of the house method: *"Checkpoint selection =
    a band, not a winner … the shipped deliverable was three checkpoints."* Nothing in
    ``QwenEditConfig`` declares one, and the fallback a reader reaches for —
    ``CheckpointManager.find_latest()`` — is the H3 D-10-DEF-19 failure under a different name: the
    render directory is keyed on the render's IDENTITY and the checkpoint name is part of that key,
    so against a training run that commits every ``checkpoint_every`` steps each re-dispatch
    resolves a DIFFERENT adapter, lands in a FRESH directory, and finds nothing to resume. A
    5000-step run checkpointing every 250 steps moves that target twenty times. WHAT LANDS IT:
    ``qwen_edit.render_checkpoint_band`` — the ordered band-member directory NAMES under
    ``<output_dir>/`` — H3's ``h3.render_checkpoint_name`` pin made plural, because on this family
    the deliverable IS the band.

    **(4) the held-out control inputs AND their A/B prompt pair — ``sample`` only.** §8 requires
    *"A/B prompt modes on every held-out input: (A) style-only … and (B) content-named … Same
    input, two prompts, side by side"*. Two schema surfaces are missing and they are missing
    TOGETHER: ``qwen_edit.control_dirs`` names the TRAINING control directories (rendering those
    asks a different question than a held-out probe does), and ``validation.prompts`` is a flat
    ``list[str]`` that cannot say which entry is A, which is B, or which input either belongs to.
    Deriving the pair positionally from that list is the invention this file must not make: A and B
    differ by whether the SUBJECT IS NAMED, so a mis-paired list silently inverts the
    trace-vs-reinterpret read the grid exists to produce, at a perfectly ordinary-looking grid.
    WHAT LANDS IT: one declared object per held-out input carrying its ``id``, its ordered per-slot
    ``images`` and its prompt PER MODE — ``qwen_edit.render_inputs`` of
    ``{id, images, prompts: {a_style: …, b_content: …}}``, keyed by the ids in
    ``QWEN_EDIT_PROMPT_MODES`` because that tuple is the single place a mode is declared. This is
    the ``ValidationSample`` doctrine this schema already applies (*"an OBJECT … so a sample's
    prompt and its condition can never desync across a config edit"*), and it is the exact shape
    ``inference/qwen_edit_layout.QwenEditHeldOutInput`` consumes, so nothing translates between the
    two.

    **(5) a config that CONTRADICTS the locked §8 recipe — ``sample`` only.** The inference settings
    are NOT config fields and that is deliberate: ``models/qwen_edit_pipeline`` holds them as
    ``QWEN_EDIT_RENDER_RECIPE`` — 30 steps, true_cfg 4.0, CFGNorm on, the static scheduler
    reparameterisation, LoRA strength 1.0, a non-empty negative prompt — recipe terms in the same
    sense that ``quantize_qwen_edit``'s qfloat8 is one. But ``ValidationConfig`` still CARRIES
    ``num_inference_steps`` and ``guidance_scale`` (LTX's, defaulting to 30 and 3.0), and the render
    does not read them. A config that declares 3.0 while the grid renders at 4.0 is a config lying
    to its reader, and the gallery banner would print one of the two — the same
    "banner that describes a render nobody performed" failure ``h3_sample`` records at its
    ``width``/``height`` (``modal/fns.py:4808-4813``). So a divergence is REFUSED at $0 with the
    remedy in the message, rather than silently ignored. WHAT LANDS IT: set the fields to the
    recipe's values (``guidance_scale: 4.0``, ``num_inference_steps: 30``) — or land the ruling that
    makes them real knobs, at which point the render reads them and this gap goes away with it.
    """
    gaps: list[str] = []

    if str(cfg.model.model_id).endswith(".safetensors") and cfg.model.pipeline_root_id is None:
        gaps.append(
            f"model.model_id is the single FILE {cfg.model.model_id!r}, so "
            "load_qwen_edit_transformer needs a config_source — and the field that would carry it "
            "(model.pipeline_root_id) is refused on this family by config/schema.py's "
            "_FAMILY_ONLY_MODEL_IDS, which maps it to frozenset({'h3'}). WHAT LANDS IT: add "
            "'qwen_edit' to that entry (the field now has a consumer, which is the condition its "
            "own comment records for inclusion), or point model.model_id at a diffusers DIRECTORY "
            "on the weights Volume, where the config travels with the weights and no config_source "
            "is needed. Dispatching without it would make diffusers fetch a Stable Diffusion v1.5 "
            "config from the Hub inside a metered container."
        )

    # ``sample`` joins the two encode legs here: the render assembles a QwenImageEditPlusPipeline
    # whose PROCESSOR component lives at <root>/processor, exactly as the pre-encode's does. Both
    # plausible wrong guesses (<root>, <root>/text_encoder) were tried on live hardware and both
    # failed, so the render inherits the refusal instead of re-discovering it on a metered container.
    if mode in ("preprocess", "train", "sample") and cfg.model.pipeline_root_id is None:
        gaps.append(
            "model.pipeline_root_id is unset, and the Qwen2.5-VL PROCESSOR is a PIPELINE-ROOT "
            "component: a Qwen-Image-Edit-2511 snapshot writes preprocessor_config.json into "
            "<root>/processor/, NOT into <root>/text_encoder/. Composing the processor path from "
            "model.text_encoder_id raises \"Can't load image processor for .../text_encoder\" — and "
            "it raises AFTER the arch gate has loaded 38 GiB of transformer, so the discovery is "
            "metered. Declaring the root here makes it free. WHAT LANDS IT: add the root directory "
            "under WEIGHTS_DIR to your `model:` block, e.g.\n"
            "        pipeline_root_id: qwen-image-edit-2511"
        )

    if mode == "preprocess":
        declared = tuple(getattr(cfg.qwen_edit, "control_dirs", ()) or ())
        blanks = tuple(getattr(cfg.qwen_edit, "blank_slots", ()) or ())
        covered = len(declared) + len(blanks)
        if covered != int(cfg.qwen_edit.control_slots):
            gaps.append(
                "the ORDERED control-image directories (one per slot) are not fully declared: "
                f"qwen_edit.control_dirs has {len(declared)} entr(ies) and qwen_edit.blank_slots "
                f"has {len(blanks)}, covering {covered} of {cfg.qwen_edit.control_slots} slot(s). "
                "prep/qwen_edit_encode.resolve_qwen_edit_control_sources requires every slot to be "
                "accounted for because the mapping is POSITIONAL — directory i fills slot i, which "
                "is what the caption's ctrl_img_{i+1} refers to. This is deliberately NOT defaulted "
                "to a convention: a guessed directory order re-points every caption's ctrl_img_N "
                "and trains the sample against a request nobody wrote, silently and at an ordinary "
                "loss. Declare them explicitly, in slot order."
            )

    if mode == "sample":
        request = _qwen_edit_render_request(cfg)

        if not request["band"]:
            gaps.append(
                f"the checkpoint BAND is not declared (qwen_edit.{_QWEN_EDIT_BAND_FIELD} is unset "
                "or empty), so the render has no adapter to load. §8 makes the BAND the deliverable "
                "unit — 'checkpoint selection = a band, not a winner'; the shipped JPM deliverable "
                "was three checkpoints — and inference/qwen_edit_layout.CheckpointBand refuses an "
                "empty band because a grid whose only column is the un-adaptered base control is a "
                "valid-looking artifact that answers nothing. Falling back to "
                "CheckpointManager.find_latest() is NOT the fix: it is a moving target while a run "
                "is live (H3's D-10-DEF-19), and because the render directory is keyed on the "
                "checkpoint NAME, every re-dispatch would resolve a different adapter, land in a "
                "fresh directory and resume nothing. WHAT LANDS IT: "
                f"qwen_edit.{_QWEN_EDIT_BAND_FIELD}, the ordered band-member directory names under "
                "<output_dir>/ — H3's render_checkpoint_name pin made plural."
            )

        if not request["inputs"]:
            gaps.append(
                "the HELD-OUT control inputs and their A/B prompt pair are not declared "
                f"(qwen_edit.{_QWEN_EDIT_INPUTS_FIELD} is unset or empty). §8 renders every held-out "
                "input under BOTH prompt modes — (A) style-only, subject NOT named; (B) "
                "content-named — side by side, because the A-vs-B delta at a fixed checkpoint IS "
                "the trace-vs-reinterpret read. Neither half can be borrowed from what exists: "
                "qwen_edit.control_dirs names the TRAINING controls (rendering those asks a "
                "different question than a held-out probe), and validation.prompts is a flat list "
                "that cannot say which entry is A, which is B, or which input either belongs to — "
                "and a mis-paired list inverts the very read the grid is for, at an "
                f"ordinary-looking grid. WHAT LANDS IT: qwen_edit.{_QWEN_EDIT_INPUTS_FIELD}, one "
                "object per input with {id, images (one per control slot, in slot order), prompts "
                "(keyed by the QWEN_EDIT_PROMPT_MODES ids: a_style, b_content)} — the shape "
                "inference/qwen_edit_layout.QwenEditHeldOutInput already consumes, so nothing "
                "translates between the config and the planner."
            )
        else:
            slots = int(cfg.qwen_edit.control_slots)
            for i, entry in enumerate(request["inputs"]):
                absent = [
                    name
                    for name in ("id", "images", "prompts")
                    if not getattr(entry, name, None)
                ]
                if absent:
                    gaps.append(
                        f"qwen_edit.{_QWEN_EDIT_INPUTS_FIELD}[{i}] is missing {absent} — every "
                        "held-out input needs an id (it is the CONTROL AXIS of the render key AND "
                        "the stem of every file it renders, so two inputs sharing one id overwrite "
                        "each other inside one render dir), its ordered per-slot images, and a "
                        "prompt for BOTH §8 modes."
                    )
                    continue
                images = tuple(getattr(entry, "images", ()) or ())
                if len(images) != slots:
                    gaps.append(
                        f"qwen_edit.{_QWEN_EDIT_INPUTS_FIELD}[{i}] "
                        f"({getattr(entry, 'id', '?')!r}) declares {len(images)} control image(s) "
                        f"but qwen_edit.control_slots is {slots}. The mapping is POSITIONAL — image "
                        "i fills slot i, which is what the prompt's ctrl_img_{i+1} addresses — so a "
                        "short list does not render a smaller grid, it renders the WRONG request "
                        "under the right label."
                    )

        # The recipe is LOCKED in models/qwen_edit_pipeline, so these two config fields are not read
        # by the render. They can still CONTRADICT it, and a banner printing one number while the
        # grid was rendered at another is the failure h3_sample records at its width/height.
        # Imported function-locally: this module must stay importable on the SDK-free interpreter
        # the dry-run contract targets, and the recipe lives in the Modal-side model tier.
        from signet_trainer.models.qwen_edit_pipeline import (  # noqa: PLC0415
            QWEN_EDIT_RENDER_RECIPE,
        )

        contradictions = [
            f"validation.{field}={declared!r} but the locked §8 recipe renders at {expected!r}"
            for field, declared, expected in (
                (
                    "guidance_scale",
                    float(cfg.validation.guidance_scale),
                    float(QWEN_EDIT_RENDER_RECIPE.true_cfg),
                ),
                (
                    "num_inference_steps",
                    int(cfg.validation.num_inference_steps),
                    int(QWEN_EDIT_RENDER_RECIPE.steps),
                ),
            )
            if declared != expected
        ]
        if contradictions:
            gaps.append(
                "the config contradicts the LOCKED §8 inference recipe: "
                + "; ".join(contradictions)
                + ". models/qwen_edit_pipeline.QWEN_EDIT_RENDER_RECIPE is deliberately NOT a config "
                "block — every term in it is settled by the method doc in the same sense that "
                "quantize_qwen_edit's qfloat8 is — so the render uses the recipe and does NOT read "
                "these fields. A config that declares one number while the grid renders at another "
                "puts a figure in the gallery banner that describes a render nobody performed, "
                "which is the failure h3_sample records for its own banner-only width/height "
                "(modal/fns.py:4808-4813). Note validation.guidance_scale would in any case map to "
                "true_cfg_scale and NEVER to the pipeline's guidance_scale=, which this checkpoint "
                "ignores for want of a guidance embedder — passing it there renders the whole grid "
                "at CFG 1.0. WHAT LANDS IT: set the fields to the recipe's values, or land the "
                "ruling that makes them real knobs (at which point the render reads them and this "
                "gap goes with it)."
            )

    return gaps


def _qwen_edit_refuse_on_gaps(cfg: object, *, mode: str) -> None:
    """Abort pre-dispatch, naming every gap at once, when a qwen_edit config cannot drive ``mode``.

    Every gap is reported together rather than the first one — the enochiatron lesson the H3 arch
    gate records (*"caught 6 mismatches in one ~$1.40 run precisely because it did not stop at the
    first"*), applied to configuration instead of architecture. Fixing one gap and re-running to
    discover the next is how a cheap check becomes an expensive loop.
    """
    gaps = _qwen_edit_config_gaps(cfg, mode=mode)
    if not gaps:
        return
    listed = "\n".join(f"  ({i + 1}) {gap}" for i, gap in enumerate(gaps))
    raise SystemExit(
        f"[signet-entrypoint] the qwen_edit {mode!r} stage cannot be dispatched from this config — "
        f"{len(gaps)} DECLARED gap(s):\n{listed}\n"
        "[signet-entrypoint] Aborting AFTER the approval pause and BEFORE any dispatch: nothing was "
        "spent. Every gap above names what lands it."
    )


def _qwen_edit_encode_params(cfg: object) -> dict[str, object] | None:
    """EVERY required kwarg of ``fns.qwen_edit_preprocess`` for a ``family: qwen_edit`` config.

    The ``_h3_encode_params`` shape and the same burned-gate discipline (09-07 T3): read ONLY fields
    that exist on the loaded config, and read them HERE — bound to a local, before the dispatch —
    never inline inside the ``.spawn(`` expression. ``SignetConfig`` is ``extra="forbid"``, so a
    field read that names the wrong block raises ``AttributeError`` AFTER the approval pause; a named
    abort at that point still costs $0 but it tells the operator what to fix.

    ``qwen_edit_preprocess`` declares ALL 17 parameters REQUIRED with no defaults, so a threading gap
    is a ``TypeError`` at dispatch rather than a silent wrong default inside a paid container.

    ``control_dirs`` / ``blank_slots`` are REAL SCHEMA FIELDS on ``QwenEditConfig`` as of this
    branch, so this function no longer always aborts — it aborts only when the declared slots do
    not account for ``control_slots``, which is gap (2)'s per-mode check and a genuine config
    error. The earlier text here said the abort was unconditional; that described the tree
    before the fields landed and would send an operator hunting for a schema gap that is closed.
    """
    if cfg.model.family != "qwen_edit":
        return None
    for name in ("vae_id", "text_encoder_id"):
        if getattr(cfg.model, name) is None:
            raise SystemExit(
                f"[signet-entrypoint] model.{name} is unset on a family: qwen_edit config. The "
                f"pre-encode loads the Qwen-Image VAE and the Qwen2.5-VL text encoder (WITH its "
                f"vision tower) as separate components under WEIGHTS_DIR, and qwen_edit_preprocess "
                f"requires both. Aborting before any dispatch — nothing was spent."
            )
    # Raises on every gap at once. Kept BEFORE the dict build so a missing field cannot surface as a
    # confusing AttributeError from inside the comprehension below.
    _qwen_edit_refuse_on_gaps(cfg, mode="preprocess")
    width, height, _frames = cfg.training_dims
    return {
        # cfg.data — where the manifest is read from and where the three qwen_edit_* sources land.
        "metadata_path": cfg.data.metadata_path,
        "output_dir": cfg.data.preprocessed_data_root,
        # ⛔ UNREACHABLE until gap (2) lands; the refusal above fires first. Spelled out rather than
        #    omitted so the shape of the threading is reviewable now.
        "control_dirs": tuple(getattr(cfg.qwen_edit, "control_dirs", ())),
        "blank_slots": tuple(getattr(cfg.qwen_edit, "blank_slots", ())),
        # cfg.qwen_edit — the locked recipe values, every one a documented field (D-NOHARDCODE).
        "control_slots": cfg.qwen_edit.control_slots,
        "blank_slot_fill": cfg.qwen_edit.blank_slot_fill,
        # ⚠ TWO DIFFERENT BUDGETS for the SAME image, and conflating them is the family's easiest
        #   mistake: control_area_px (1024²) is the VAE channel, condition_area_px (384²) is the
        #   Qwen2.5-VL channel. Threaded as two parameters so neither can stand in for the other.
        "control_area_px": cfg.qwen_edit.control_area_px,
        "condition_area_px": cfg.qwen_edit.condition_area_px,
        "control_cache_key_mode": cfg.qwen_edit.control_cache_key_mode,
        "cache_text_embeddings": cfg.qwen_edit.cache_text_embeddings,
        # Threaded so the arch gate CHECKS the config's [UNVERIFIED] 3584 against the live txt_in
        # rather than believing it — the field's own description says the loading pass must.
        "text_embed_dim": cfg.qwen_edit.text_embed_dim,
        # training_dims is [W, H, F]; F is pinned to exactly 1 on this IMAGE family.
        "target_width": width,
        "target_height": height,
        # cfg.model — the component paths under WEIGHTS_DIR (the CALLER composes them Modal-side).
        "model_id": cfg.model.model_id,
        "vae_id": cfg.model.vae_id,
        "text_encoder_id": cfg.model.text_encoder_id,
        "pipeline_root_id": cfg.model.pipeline_root_id,
    }


# --------------------------------------------------------------------------------------------------
# Family #4 (wan) — the musubi-tuner RUNNER leg. One gated arm inside the existing ``train`` mode.
# --------------------------------------------------------------------------------------------------

#: The ONE mode the wan family serves, named once because three places must agree: the gap check
#: (which refuses the other five), the arm below (which routes on it), and the operator reading the
#: message. There is no wan encode stage because musubi's two cache passes run INSIDE ``wan_train``,
#: and no wan render stage because this repo has no Wan inference path at all.
_WAN_SUPPORTED_MODE = "train"


def _wan_source_views(cfg: object) -> tuple[object, ...]:
    """The declared sources reduced to the flat records ``modal/cost`` prices.

    Read TOLERANTLY through ``getattr`` — the ``_qwen_edit_render_request`` idiom — so this works
    against a config object whether or not ``data.sources`` exists on it, and the extraction is done
    HERE rather than inside ``cost.py`` so that module keeps its pure-arithmetic character and its
    import closure (it never learns what a ``SourceSpec`` is).
    """
    from signet_trainer.modal.cost import WanSourceView  # noqa: PLC0415

    return tuple(
        WanSourceView(
            source_id=source.id,
            kind=source.kind,
            # ``.value`` because ExtractionMode is ``(str, Enum)``: the mixin means ``str(mode)``
            # renders ``ExtractionMode.HEAD`` on 3.10, not ``head``, and the pricing table keys on
            # the wire value. StrEnum would have made these identical — and StrEnum is 3.11+, which
            # is exactly why the enum is a mixin (config/sources.py records the bench-interpreter
            # reason). Reading ``.value`` is the cost of that, paid once, here.
            extraction=source.extraction.value,
            directory=source.directory,
            target_frames=tuple(source.target_frames),
            frame_sample=source.frame_sample,
            num_repeats=source.num_repeats,
        )
        for source in (getattr(cfg.data, "sources", None) or ())
    )


def _wan_batch_note(cfg: object) -> str:
    """Size the musubi round from its declared sources, printed BESIDE the cost line.

    The ``_qwen_edit_render_batch_note`` move, applied to a run whose work is only PARTLY knowable —
    and the partial knowledge is reported as partial rather than rounded up into a total. What can
    be computed at $0 is clip instances PER MEDIA FILE per source; what cannot is how many media
    files each corpus directory holds, because those are container paths on the dataset Volume and
    this process performs no filesystem touch (Pitfall 1). ``chunk`` and ``slide`` additionally
    depend on video LENGTH, so a source using either prices as "corpus-dependent" and the total
    declines to exist rather than silently under-counting.

    The guardrail's BASIS is unchanged: this adds a line, never a decision. Turning clip counts into
    hours would need a seconds-per-step figure nobody in this program has measured for Wan on any
    card, and inventing one to feed a money gate is the opposite of what the gate is for.
    """
    from signet_trainer.modal.cost import format_wan_batch_line, wan_batch_estimate  # noqa: PLC0415
    from signet_trainer.runners.wan_musubi import WAN_MUSUBI_RECIPE  # noqa: PLC0415

    return format_wan_batch_line(
        wan_batch_estimate(
            sources=_wan_source_views(cfg),
            max_train_epochs=WAN_MUSUBI_RECIPE.max_train_epochs,
            declared_max_steps=int(cfg.training.max_steps),
            est_hours=float(cfg.modal.est_hours),
        )
    )


def _wan_components(cfg: object) -> object:
    """Resolve the four Wan component ids, refusing an INHERITED LTX default as firmly as a gap.

    ONE call site for two readers (the gap check and the params helper), so the two can never
    disagree about whether a config names its weights.

    The ``schema_defaults`` mapping is DERIVED from ``ModelConfig.model_fields`` rather than
    restated, and that derivation is the whole point. ``model.text_encoder_id`` defaults to
    ``"gemma-3-12b-it"`` — LTX's Gemma encoder — and ``model.model_id`` to the LTX checkpoint, so a
    truthiness test alone RESOLVES on a wan config that declared neither and the stage would hand
    musubi ``--t5 <weights>/gemma-3-12b-it``. Deriving the defaults means the refusal stays true if
    a default ever changes; restating them here would be a second copy that silently rots.

    ``ModelConfig`` is imported function-locally so this module keeps its import surface — the same
    reason ``_qwen_edit_config_gaps`` imports the render recipe inside its own body.
    """
    from signet_trainer.config.schema import ModelConfig  # noqa: PLC0415
    from signet_trainer.runners.wan_musubi import (  # noqa: PLC0415
        WAN_COMPONENT_CONFIG_FIELDS,
        wan_resolve_component_ids,
    )

    return wan_resolve_component_ids(
        cfg.model,
        schema_defaults={
            field: ModelConfig.model_fields[field].default
            for field in WAN_COMPONENT_CONFIG_FIELDS.values()
            if field in ModelConfig.model_fields
        },
    )


def _wan_config_gaps(cfg: object, *, mode: str) -> list[str]:
    """Every DECLARED gap between a validated ``family: wan`` config and what the stage needs.

    The ``_qwen_edit_config_gaps`` contract: one sentence per gap saying what is missing, why the
    stage needs it, and what would land it — reported ALL AT ONCE (the enochiatron lesson: fixing
    one gap and re-running to find the next is how a cheap check becomes an expensive loop).

    ``cfg`` is typed ``object`` and read by ATTRIBUTE throughout. That is not defensiveness for its
    own sake: two of the checks below are held a SECOND time here, and a second line of defense that
    silently assumes the first one ran is not one.

    **(1) the MODE — every mode but ``train``.** Family #4 has exactly one stage, and the other five
    modes are not merely unimplemented, they are wrong in different ways. ``preprocess`` would be a
    second cost line for work ``wan_train`` already performs, because musubi's two cache passes read
    the dataset TOML directly and run INSIDE the training stage; a signet-side Wan encoder would
    additionally be the "never write a custom encoder" landmine, since a canonical one exists.
    ``sample`` has no Wan inference path anywhere in this repo. ``fuse`` / ``restore`` / ``backup``
    are CPU utilities keyed to signet's own checkpoint layout, and musubi writes its own. Refusing by
    name here is what stops a wan config falling through to an arm built for a different family.

    **(2) the SOURCE LIST — every mode.** A wan run's dataset IS the ``[[datasets]]`` array;
    ``render_from_config`` refuses to synthesise one from ``preprocessed_data_root`` because that
    would invent a resolution, an extraction mode and a cache identity nobody wrote.
    ``SignetConfig``'s wan arm already refuses this at config LOAD, so on the normal path this is a
    second line of defense — and it earns its place: the renderer runs at DISPATCH time, inside this
    seam, and a traceback out of a renderer the operator did not know was running is a worse
    failure than a named refusal beside the other gaps.

    **(3) COLLIDING CACHE ROOTS — every mode.** Likewise held a second time (``DataConfig.
    _check_sources`` delegates to the same ``check_cache_collisions``), and likewise not tidiness:
    musubi deletes cache files that are absent from the current dataset spec, which is CORRECT when
    the spec changes between rounds and is only safe because every source has a unique cache
    directory. Two sources sharing one turn that default into mutual destruction inside a paid
    container. ``config/sources.check_cache_collisions`` is exposed as a pure list-returning check
    precisely so it can be reported HERE, at $0, rather than as a renderer exception.

    **(4) the CAPTION EXTENSION — every mode.** ``[general].caption_extension`` is what musubi
    appends to each media stem to find its caption, and the renderer takes it as a required
    argument. An absent or blank value renders ``caption_extension = ""`` — a perfectly valid TOML
    line that makes every caption path the bare media stem, so every clip trains with an empty
    caption at an entirely ordinary loss curve. Read tolerantly (``getattr``) so this fires on a
    config object that predates the field as well as one that leaves it empty.

    **(5) the FOUR WEIGHT COMPONENTS — every mode.** Wan 2.1 needs a DiT, a VAE, a umT5 text encoder
    AND an open-CLIP encoder (``train_kohya.py:69-92``), and signet resolves all four from the
    weights Volume rather than ``hf_hub_download``-ing them per container. Three distinct kinds of
    gap sit behind that, which is why ``runners/wan_musubi.wan_resolve_component_ids`` reports them
    together and this check simply relays its message: ``model.model_id`` / ``model.text_encoder_id``
    are ordinary UNSET fields; ``model.vae_id`` EXISTS but is fenced away from this family by
    ``config/schema._FAMILY_ONLY_MODEL_IDS``; ``model.clip_id`` does not exist at all, because no
    signet family has ever carried two text-side encoders.

    **(6) the UNPINNED musubi CHECKOUT — every mode.** ``modal/app.MUSUBI_TUNER_COMMIT_SHA`` is
    ``None``. Every other foreign checkout in that file is a literal 40-hex SHA, and this one has no
    value to transcribe (``train_kohya.py:31`` clones ``main`` with no checkout) and none this pass
    could resolve (zero downloads). Writing a plausible hex string would be fabrication of exactly
    the kind those pins exist to prevent, so the gap is refused instead — and it is not cosmetic:
    musubi's dataset-config SCHEMA is what ``runners/musubi_toml.py`` transcribes, so a floating
    ``main`` that renamed a key is discovered as a rejected TOML inside a metered container.
    """
    gaps: list[str] = []

    if mode != _WAN_SUPPORTED_MODE:
        gaps.append(
            f"--mode {mode!r} has no wan arm, and the wan family has exactly one stage "
            f"(--mode {_WAN_SUPPORTED_MODE!r}). musubi owns caching END TO END: "
            "wan_cache_latents.py and wan_cache_text_encoder_outputs.py read the dataset TOML "
            "directly and run INSIDE wan_train, so a 'preprocess' arm would be a second cost line "
            "for work the train stage already performs (and a signet-side Wan encoder would be a "
            "custom re-implementation of a canonical one). 'sample' has no Wan inference path "
            "anywhere in this repo. fuse/restore/backup are keyed to signet's own checkpoint "
            "layout, which musubi does not write. WHAT LANDS a render arm: a Wan inference path "
            "(pipeline + sampler), which is a family of work, not a Modal wiring change."
        )

    sources = tuple(getattr(cfg.data, "sources", None) or ())
    if not sources:
        gaps.append(
            "data.sources is absent or empty. The musubi dataset TOML IS the source list — "
            "runners/musubi_toml.render_from_config refuses to derive a [[datasets]] block from "
            "data.preprocessed_data_root because that would invent a resolution, an extraction mode "
            "and a cache identity nobody wrote. SignetConfig's wan arm refuses this at config load "
            "too; it is repeated here because the renderer RUNS at this seam. WHAT LANDS IT: "
            "declare data.sources (see configs/wan21_kaboom.example.yaml)."
        )
    else:
        from signet_trainer.config.sources import check_cache_collisions  # noqa: PLC0415

        collisions = check_cache_collisions(
            sources, data_root=cfg.data.preprocessed_data_root
        )
        if collisions:
            gaps.append(
                "; ".join(collisions)
                + ". musubi DELETES cache files that are absent from the current dataset spec — "
                "correct when the spec changes between rounds, and safe only because every source "
                "has a unique cache directory. Two sources sharing one do not merely collide, they "
                "mutually destroy each other's latents inside a paid container. WHAT LANDS IT: give "
                "each source its own cache_root, or omit the key and let "
                "SourceSpec.resolve_cache_root derive <preprocessed_data_root>/cache/<id>, which "
                "cannot collide by omission."
            )

    caption_extension = getattr(cfg.data, "caption_extension", None)
    if not caption_extension:
        gaps.append(
            f"data.caption_extension is {caption_extension!r}. It is the [general] key musubi "
            "appends to each media stem to locate that file's caption, and it is a REQUIRED "
            "argument of render_musubi_toml — an empty value renders `caption_extension = \"\"`, "
            "which is valid TOML and makes every caption path the bare media stem, so every clip "
            "trains with an EMPTY caption at a perfectly ordinary loss curve. WHAT LANDS IT: set "
            "data.caption_extension (it must start with '.', e.g. \".txt\"); note "
            "SourceSpec.caption_extension can override it per source, which is what a corpus with "
            "per-clip captions needs."
        )

    try:
        _wan_components(cfg)
    except NotImplementedError as exc:
        gaps.append(str(exc))

    from signet_trainer.modal.app import MUSUBI_TUNER_COMMIT_SHA  # noqa: PLC0415

    if not MUSUBI_TUNER_COMMIT_SHA:
        gaps.append(
            "modal/app.MUSUBI_TUNER_COMMIT_SHA is unpinned (None), so wan_musubi_image would build "
            "against a FLOATING kohya-ss/musubi-tuner main — the failure LTX2_COMMIT_SHA / "
            "DIFFUSERS_SHA / QWEN_DIFFUSERS_SHA all exist to prevent, and worse here because "
            "musubi's dataset-config SCHEMA is what runners/musubi_toml.py transcribes: an upstream "
            "key rename surfaces as a rejected TOML inside a metered container after the weights "
            "are resident. The image build refuses rather than falling back to main, so this is "
            "belt and braces. WHAT LANDS IT: `gh api repos/kohya-ss/musubi-tuner/commits/main "
            "--jq .sha`, then set the literal 40-hex string in modal/app.py."
        )

    return gaps


def _wan_refuse_on_gaps(cfg: object, *, mode: str) -> None:
    """Abort pre-dispatch, naming every gap at once, when a wan config cannot drive ``mode``.

    The ``_qwen_edit_refuse_on_gaps`` shape and position: called AFTER ``_require_approval`` and
    BEFORE ``.spawn(``, so a doomed run aborts without spending and without weakening MODL-02.
    """
    gaps = _wan_config_gaps(cfg, mode=mode)
    if not gaps:
        return
    listed = "\n".join(f"  ({i + 1}) {gap}" for i, gap in enumerate(gaps))
    raise SystemExit(
        f"[signet-entrypoint] the wan {mode!r} stage cannot be dispatched from this config — "
        f"{len(gaps)} DECLARED gap(s):\n{listed}\n"
        "[signet-entrypoint] Aborting AFTER the approval pause and BEFORE any dispatch: nothing was "
        "spent. Every gap above names what lands it."
    )


def _wan_train_params(cfg: object) -> dict[str, object] | None:
    """EVERY required kwarg of ``fns.wan_train`` for a ``family: wan`` config; ``None`` otherwise.

    The ``_h3_encode_params`` shape and the same burned-gate discipline (09-07 T3): read ONLY fields
    that exist on the loaded config, and read them HERE — bound to a local, before the dispatch —
    never inline inside the ``.spawn(`` expression.

    ⛔ THE DATASET TOML IS RENDERED HERE, AT DISPATCH TIME, and shipped BY VALUE. That is the whole
    feature rather than a detail of packaging. ``train_kohya.py:40-43`` bakes the TOML into the
    IMAGE, which makes changing the dataset an image REBUILD and makes "what did round 2 train on?"
    unanswerable from the artifacts. Rendering it from the validated manifest at dispatch — and
    writing it beside the adapter in-container — turns "only the dataset changed between rounds"
    from a memory into a ``diff`` of two committed files.

    It is also the only way this stage can work at all: ``wan_musubi_image`` carries musubi's
    ``pydantic==1.10.13`` and cannot load a ``SignetConfig``, so the manifest must be resolved on
    THIS side of the dispatch. ``wan_train`` declares all eight parameters REQUIRED with no
    defaults, so a threading gap is a ``TypeError`` at dispatch rather than a silent wrong default.

    ``output_name`` is derived from ``cfg.output_dir``'s final segment (``outputs/wan21_kaboom`` ->
    ``wan21_kaboom``) rather than carried as a field: musubi's ``--output_name`` is the adapter's
    FILE stem inside ``--output_dir``, so deriving it keeps one round's artifacts named after that
    round with nothing new to keep in sync. train_kohya.py's literal ``wan21-lora`` is not carried —
    it would name every round's adapter identically.
    """
    if cfg.model.family != "wan":
        return None
    # Raises on every gap at once. Kept BEFORE the render below so a missing component surfaces as
    # the named refusal rather than as a confusing ValueError out of the renderer.
    _wan_refuse_on_gaps(cfg, mode=_WAN_SUPPORTED_MODE)

    from signet_trainer.runners.musubi_toml import render_from_config  # noqa: PLC0415

    components = _wan_components(cfg)
    return {
        "dataset_toml": render_from_config(cfg),
        "dit_id": components.dit,
        "vae_id": components.vae,
        "t5_id": components.t5,
        "clip_id": components.clip,
        "output_dir": cfg.output_dir,
        "output_name": Path(cfg.output_dir).name,
        "seed": cfg.seed,
    }


@app.local_entrypoint()
def main(config: str, approve: bool = False, mode: str = "train") -> None:
    """Preflight + gated launch from a YAML config — load, dry-run gate, cost print, APPROVE, dispatch.

    Runs locally (no GPU). Aborts BEFORE any remote call if the dry-run gate fails, the cost
    guardrail blocks, or approval is declined. The strict launch ordering — load -> dry-run gate ->
    cost print -> APPROVAL PAUSE -> dispatch — is the single gate all metered runs (Phase-2 smoke/
    encode AND Phase-3 train AND Phase-4 sample) pass through, so no metered spend can ever
    auto-launch.

    ``mode`` selects the gated remote fn: ``"train"`` (default, ``train.spawn``), ``"sample"``
    (the Phase-4 base-vs-LoRA grid, ``sample.spawn``), or ``"preprocess"`` (the Phase-8 canonical
    pre-encode, ``preprocess.spawn`` — D-8-PREPROC). ALL THREE dispatch STRICTLY after the same
    ``_require_approval`` pause — each run is metered, so they reuse this identical cost -> approval
    -> dispatch gate rather than a second launch path. The preprocess arm retires the hardcoded
    hourly-rate literal in the throwaway ``scripts/_encode_demo*.py`` by reading rates from
    ``cfg.modal`` through this shared cost line.

    Phase 10 (H3-07) adds the MiniMax-H3 leg WITHOUT adding a mode: ``train`` / ``sample`` /
    ``preprocess`` each route on ``cfg.model.family`` inside their existing arm, dispatching
    ``h3_train`` / ``h3_sample`` / ``h3_preprocess`` for a ``family: h3`` config and the LTX stage
    otherwise. Six dispatches, one gate, one ledger.

    Phase 11 adds the Qwen-Image-Edit leg the SAME way and for the same reason — a THIRD family, and
    still zero new modes. The same three arms gained a ``family == "qwen_edit"`` branch dispatching
    ``qwen_edit_train`` / ``qwen_edit_sample`` / ``qwen_edit_preprocess``. Nine dispatches now, and
    still ONE gate and ONE ledger: that invariant is what makes the count boring rather than
    alarming. Each qwen arm additionally runs two $0 pre-dispatch checks — ``_qwen_edit_stage_
    readiness`` (are the modules the stage imports actually in this tree?) and
    ``_qwen_edit_refuse_on_gaps`` (can this config drive that stage at all?) — both strictly AFTER
    the approval pause and BEFORE the ``.spawn``, so they abort a doomed run without spending and
    without weakening MODL-02.

    Family #4 (``wan``) adds the musubi-tuner RUNNER leg and, again, ZERO new modes — but it adds
    only ONE arm, inside ``train``. That asymmetry is real and is the interesting part: musubi owns
    caching end to end (its two cache passes read the dataset TOML directly and run INSIDE
    ``wan_train``), so there is nothing for a ``preprocess`` arm to do that would not be a second
    cost line for the same work; and no Wan inference path exists in this repo, so there is nothing
    for ``sample`` to dispatch. TEN dispatches now, still ONE gate and ONE ledger. Because the other
    five modes have no wan arm, a family FENCE sits between the approval pause and the routing chain
    — without it a ``family: wan`` config on ``--mode sample`` would fall through to the LTX
    sampler, which is precisely what the dry-run refusal used to prevent for every mode at once.

    Pass ``--approve`` to authorize non-interactively (harness path); otherwise the operator must
    type ``approved`` at the prompt (D-03 / MODL-02).
    """
    # ⛔ Do NOT add an ``h3_*`` value to this tuple. Two structural tests make it a breaking change,
    #    and they are right to: ``tests/test_skill_entrypoint_coverage.py`` pins the AST-parsed set to
    #    exactly these six AND requires every real mode to be documented as ``--mode <value>`` in a
    #    lifecycle skill, so a new mode fails twice over. H3-07 specifies ``--mode
    #    preprocess|train|sample`` anyway — family routing INSIDE the existing arms is both the
    #    required design and the cheaper one (no second launch path, no second cost line, no second
    #    place for the ledger to drift). ``tests/test_h3_entrypoint_gate.py`` asserts the same set a
    #    second time, with the H3-specific reasoning attached.
    if mode not in ("train", "sample", "preprocess", "fuse", "restore", "backup"):
        raise SystemExit(
            f"[signet-entrypoint] unknown --mode {mode!r} "
            "(expected 'train', 'sample', 'preprocess', 'fuse', 'restore', or 'backup')."
        )
    # (1) Load + validate the config ONCE (fail-fast LTX validators fire here; CONF-02). The cost
    #     estimate (step 3) and the dry-run gate (step 2) both read THIS same validated object —
    #     no second independent disk read, so the cost banner and the gated config can never
    #     diverge (WR-03; closes the TOCTOU gap of re-reading the file in dryrun_main).
    config_text = Path(config).read_text(encoding="utf-8")
    cfg = load_config_from_text(config_text)

    # (1b) Secret-name guard (WR-01 — the in-main env export was DEAD CODE). app.py builds the Modal
    #      app graph at MODULE-IMPORT time (this module imports it at line ~28) and resolves EVERY
    #      Secret.from_name(...) EAGERLY, reading the NAMES from the SIGNET_*_SECRET_NAME env vars THEN.
    #      By the time main() runs the app graph's Secret objects have already captured those names, so
    #      an os.environ[...] = cfg.modal.*_secret_name assignment HERE can no longer influence the
    #      graph — it worked before only because the defaults happen to match the operator's account. For a
    #      different account, the config names would be silently ignored and surface as a
    #      post-approval secret-not-found error (a burned gate). The honest contract: export
    #      SIGNET_HUGGINGFACE_SECRET_NAME / SIGNET_WANDB_SECRET_NAME IN THE SHELL *before* `modal run` so
    #      app.py captures them at import. We still mirror the config names into the env (harmless,
    #      keeps any lazy env reader aligned) but the AUTHORITATIVE check is fail-fast BELOW: compare
    #      the names app.py ACTUALLY captured to the config and abort PRE-approval on a mismatch, so a
    #      wrong-account run dies with an actionable message before any cost/approval/dispatch. Carries
    #      only NAMES, never the secret values (T-02-MD1).
    #
    #      The SAME seam and the SAME hazard cover the Modal RESOURCE names (App + the three
    #      Volumes) — see the ⛔ block in app.py. Those are worse than a wrong secret name, which at
    #      least fails loudly: ``Volume.from_name(..., create_if_missing=True)`` silently provisions
    #      a NEW EMPTY Volume for an unrecognised name, so a config/graph mismatch there would run
    #      to completion against empty storage. Guarding them here converts that silent-empty-Volume
    #      failure into a pre-approval abort.
    os.environ["SIGNET_HUGGINGFACE_SECRET_NAME"] = cfg.modal.huggingface_secret_name
    os.environ["SIGNET_WANDB_SECRET_NAME"] = cfg.modal.wandb_secret_name
    os.environ["SIGNET_HF_GATED_SECRET_NAME"] = cfg.modal.hf_gated_secret_name
    os.environ["SIGNET_APP_NAME"] = cfg.modal.app_name
    os.environ["SIGNET_WEIGHTS_VOLUME_NAME"] = cfg.modal.weights_volume_name
    os.environ["SIGNET_DATASET_VOLUME_NAME"] = cfg.modal.dataset_volume_name
    os.environ["SIGNET_CHECKPOINTS_VOLUME_NAME"] = cfg.modal.checkpoints_volume_name
    # ``kind`` keeps each abort message literally true: a wrong SECRET name and a wrong VOLUME name
    # are different failures (the first is loud at resolve time, the second is a silent empty Volume).
    for kind, env_var, captured, wanted in (
        (
            "secret-name",
            "SIGNET_HUGGINGFACE_SECRET_NAME",
            HUGGINGFACE_SECRET_NAME,
            cfg.modal.huggingface_secret_name,
        ),
        ("secret-name", "SIGNET_WANDB_SECRET_NAME", WANDB_SECRET_NAME, cfg.modal.wandb_secret_name),
        (
            "secret-name",
            "SIGNET_HF_GATED_SECRET_NAME",
            HF_GATED_SECRET_NAME,
            cfg.modal.hf_gated_secret_name,
        ),
        ("app-name", "SIGNET_APP_NAME", APP_NAME, cfg.modal.app_name),
        (
            "volume-name",
            "SIGNET_WEIGHTS_VOLUME_NAME",
            WEIGHTS_VOLUME_NAME,
            cfg.modal.weights_volume_name,
        ),
        (
            "volume-name",
            "SIGNET_DATASET_VOLUME_NAME",
            DATASET_VOLUME_NAME,
            cfg.modal.dataset_volume_name,
        ),
        (
            "volume-name",
            "SIGNET_CHECKPOINTS_VOLUME_NAME",
            CHECKPOINTS_VOLUME_NAME,
            cfg.modal.checkpoints_volume_name,
        ),
    ):
        if captured != wanted:
            silent_note = (
                " A wrong Volume name does NOT fail loudly — create_if_missing=True would mount a "
                "NEW EMPTY Volume and the run would train against empty storage."
                if kind == "volume-name"
                else ""
            )
            raise SystemExit(
                f"[signet-entrypoint] {kind} mismatch: the Modal app graph captured "
                f"{captured!r} at import time, but the config wants {wanted!r}. The in-main env "
                f"export cannot re-bind an already-built graph — export {env_var}={wanted} in the "
                f"shell BEFORE `modal run` so app.py captures it at import (WR-01). Aborting "
                f"pre-approval, no dispatch.{silent_note}"
            )

    # (1c) BK-01 master-gate pre-approval refusal (issue #33 finding 1) — pre-approval, zero-spend.
    #      backup.enabled is the DOCUMENTED master gate for ALL backup activity (schema.py's
    #      BackupConfig.enabled), but before this fix no dispatch-side code read it: --mode
    #      backup/restore with the SHIPPED default (enabled=False, destination='hf', repo_id=None)
    #      booted a metered CPU container that then crashed inside api.create_repo(repo_id=None) —
    #      a burned gate on every no-op periodic sync. Refuse here, before the dry-run gate and the
    #      cost print, so a disabled block costs $0: no container boots at all. backup_sync/restore
    #      ALSO carry their own `if not backup.enabled: return` no-op as defense-in-depth, for the
    #      (unlikely) case either fn is ever reached by a path that bypasses this gate.
    if mode in ("backup", "restore") and not cfg.backup.enabled:
        raise SystemExit(
            f"[signet-entrypoint] --mode {mode} requires backup.enabled=True (the BK-01 master "
            f"gate); the shipped default is enabled=False. Nothing dispatches — no CPU container "
            f"boots, no cost is incurred. Set backup.enabled: true (and backup.destination / "
            f"backup.repo_id) in the config, or use a different --mode. Aborting pre-approval, no "
            f"dispatch."
        )

    # (1d) H3 NO-REFERENCE (ALPHA) routing — pre-approval, zero-spend. No new mode and no new entry
    #      point: no-reference rides --mode train/preprocess through THIS same gate. `sample` is
    #      REFUSED here (and again in-container, belt to this brace): h3_sample is transcribed for
    #      the reference-conditioned ref2va workflow only, and the t2va workflow a no-reference
    #      render needs is not transcribed in this repo — faking it as "ref2va with no references"
    #      would render an unvalidated request under a no-reference label.
    if cfg.model.family == "h3" and cfg.h3.references_per_sample == 0:
        if mode == "sample":
            raise SystemExit(
                "[signet-entrypoint] h3.references_per_sample is 0 (NO-REFERENCE, ALPHA) and "
                "--mode sample was requested: no-reference H3 rendering is NOT supported (the "
                "pinned diffusers ref2va workflow is reference-conditioned end to end; the t2va "
                "workflow is not transcribed in this repo). Aborting pre-approval, no dispatch. "
                "Train/preprocess ride --mode train/preprocess; for renders, file an issue."
            )
        print(
            "[signet-entrypoint] H3 NO-REFERENCE TRAINING IS ALPHA - smoke-tested only, no "
            "end-to-end run exists; file issues"
        )

    # (1d) H3 render config-load requirement — pre-approval, zero-spend (#39 finding 2). The ONLY
    #      enforcement of this used to be modal/fns.py:h3_sample's own RuntimeError, INSIDE the
    #      metered container, AFTER the ~61.7 GiB arch gate has loaded — a dispatch paid for a
    #      refusal that costs nothing locally. `model.pipeline_root_id` is documented (schema.py)
    #      as "Required by --mode sample on a family: h3 config", but nothing at load asked for
    #      it and no shipped h3 config declares it. Message text matches fns.py's RuntimeError
    #      verbatim (module tag aside) so the operator sees the identical instruction whichever
    #      gate fires.
    if mode == "sample" and cfg.model.family == "h3" and not cfg.model.pipeline_root_id:
        raise SystemExit(
            "[signet-entrypoint] config.model.pipeline_root_id is unset. The render needs the "
            "pipeline ROOT (the dir holding model_index.json and every component partition); "
            "`model.model_id` names the transformer PARTITION inside it and means something "
            "different to h3_train / h3_loader, so it is NOT reused here (D-10-DEF-14). Add to "
            "the config's `model:` block:\n    pipeline_root_id: minimax-h3\n"
            "[signet-entrypoint] Aborting pre-approval, no dispatch."
        )

    # (2) Dry-run hard gate (CONF-03) on the already-loaded cfg — must pass before ANY remote
    #     dispatch. Non-zero -> abort. The mode is threaded through (gap-dryrun-ltx-0) so the
    #     mode-conditional refusals shared with the container bodies (config/mode_gate.py) fire
    #     HERE, pre-approval — a config the metered container would refuse at load can never
    #     burn the approval on a detached dispatch.
    rc = run_dryrun(cfg, mode=mode)
    if rc != 0:
        print(
            f"[signet-entrypoint] dry-run gate FAILED (rc={rc}) — aborting before any launch.",
            file=sys.stderr,
        )
        raise SystemExit(rc)

    # (3) Cost estimate + guardrail — PRINTED before any gated launch (MODL-03 / T-01-MD2).
    # WR-04: backup / restore / fuse run on CPU-only Modal fns (no gpu=). The A100 rate * est_hours is
    # the wrong basis for their cost, and a large training est_hours could FALSELY block a near-zero
    # CPU backup at the guardrail. For those modes derive the estimate from the CPU rate instead
    # (config-first, D-NOHARDCODE) and print an explicit CPU-only line. The approval gate below is
    # UNCHANGED — CPU modes still print a cost AND require the same approval flow (no silent free path).
    # issue #45 PR-2 (dispatch-arms-0): the estimate must be the quantity the launch actually
    # AUTHORIZES. The dispatched arm's decorator grants server-side retries (fns.py; single source:
    # retry_policy.ARM_MAX_RETRIES, weld-tested against the shipped decorators), and each retry is a
    # FRESH container with its own full timeout — so the metered ceiling is
    # rate * bounded_hours * (max_retries + 1), NOT rate * est_hours. A pre-PR-2 LTX train dispatch
    # printed a single-life estimate while authorizing 11x that behind the decorator's retries=10.
    # The guardrail, the printed line and the harness ledger (training-run SKILL projected_usd) all
    # carry the worst-case ceiling now. CPU arms (fuse/restore/backup_sync) carry no retries today
    # (lives resolves to 1), so their print is unchanged in value — only in provenance.
    lives = ARM_MAX_RETRIES[resolve_arm(mode, cfg.model.family)] + 1
    cpu_only_mode = mode in ("fuse", "restore", "backup")
    if cpu_only_mode:
        decision = guardrail_check(
            hourly_rate_usd=cfg.modal.cpu_hourly_rate_usd,
            est_hours=cfg.modal.est_hours,
            cost_guardrail_usd=cfg.modal.cost_guardrail_usd,
            lives=lives,
        )
        if mode == "fuse":
            # Issue #24: "~near-zero cost" is honest for restore/backup (default Modal resources) but
            # not for fuse, which reserves 128 GiB RAM for up to 4h (fns.py apply_loras — [precedent]
            # prior-project: materializes a ~2x 44GB dict) — Modal bills reserved memory, and the
            # cpu_hourly_rate_usd*est_hours estimate below never sees that reservation. This print
            # names it so the operator isn't told "near-zero" over a 128 GiB / 4h hold; the estimate
            # and guardrail math above are UNCHANGED.
            print(
                f"[signet-entrypoint] CPU-only mode {mode!r} (no A100) — estimated from "
                f"cpu_hourly_rate_usd=${cfg.modal.cpu_hourly_rate_usd:.2f}/hr (NOT the A100 rate); "
                "note this estimate does NOT reflect fuse's 128 GiB RAM reservation (up to 4h), "
                "which Modal bills separately from the hourly rate above."
            )
        else:
            print(
                f"[signet-entrypoint] CPU-only mode {mode!r} (no A100) — ~near-zero cost, estimated "
                f"from cpu_hourly_rate_usd=${cfg.modal.cpu_hourly_rate_usd:.2f}/hr (NOT the A100 rate)."
            )
    else:
        decision = guardrail_check(
            hourly_rate_usd=cfg.modal.hourly_rate_usd,
            est_hours=cfg.modal.est_hours,
            cost_guardrail_usd=cfg.modal.cost_guardrail_usd,
            lives=lives,
            # The SAME est_hours * timeout_margin product every GPU arm dispatches with via
            # .with_options(timeout=...) below (issue #45 PR-2 retired train()'s 24h-decorator
            # exemption so this is now uniform across all six GPU arms) — the guardrail must price
            # the bound that is actually dispatched, not a smaller number that reads more flattering.
            bounded_hours=cfg.modal.est_hours * cfg.modal.timeout_margin,
        )
    # The one mode whose WORK is knowable before dispatch: a render grid is
    # ``modes x (base + band) x held-out inputs`` images at a known step count, all declared. Printed
    # BESIDE the estimate, never in place of it — this adds a line, never a decision, so MODL-03's
    # guardrail arithmetic and its BASIS are byte-identical to every other mode.
    if mode == "sample" and cfg.model.family == "qwen_edit":
        print(_qwen_edit_render_batch_note(cfg))
    elif cfg.model.family == "wan":
        # The wan analogue, and it prints on EVERY mode rather than one: a wan round's work is
        # declared per SOURCE, so the arithmetic is available whatever the operator asked for — and
        # for the five modes this family does not serve, seeing the clip counts beside the refusal
        # is how the refusal reads as "wrong stage", not "broken config". Same discipline as above:
        # a line BESIDE format_cost_line, never in place of it.
        print(_wan_batch_note(cfg))
    print(format_cost_line(decision))
    if not decision.allowed:
        print(
            "[signet-entrypoint] cost guardrail BLOCKED the launch (over budget) — "
            "no remote dispatch.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # (3b) Cumulative session-spend cap (D-8-YOLOCAP; issue #37 finding 1/6) — reads/writes the SAME
    # ledger append_spend below writes to. WR-02 chain, config-first (D-NOHARDCODE): the ledger path
    # and the house-default cap both come from the loaded cfg; a SESSION-STATE.json session_cap_usd
    # override, when present, is the LIVE cap (the setup gate writes it there).
    #
    # ⚠ BINDING SEMANTIC (adversarial audit, D-8-YOLOCAP): this is an ASK-FIRST DOWNGRADE, never a
    # second hard refusal. The per-run cost_guardrail_usd above is the hard per-dispatch ceiling;
    # the cumulative cap only narrows the APPROVAL PATH available for THIS dispatch — going over it
    # DISABLES --approve (even when the flag was passed) and forces the interactive prompt below. A
    # present operator can still type 'approved' and proceed over cap (the human IS the fail-safe);
    # a genuinely non-interactive over-cap request (EOFError, nobody to ask) refuses — that IS the
    # yolo bound working, not a bug in it.
    ledger_path = cfg.modal.session_spend_ledger_path
    session_cap_usd = _resolve_session_cap_usd(Path(ledger_path), cfg.modal.session_cap_usd)
    try:
        spent_so_far = read_ledger(ledger_path)
    except ValueError as exc:
        print(f"[signet-entrypoint] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    cap_decision = session_cap_check(decision.est_usd, spent_so_far, session_cap_usd)
    print(f"[signet-cap] {cap_decision.reason}")
    approve_for_gate = approve
    if not cap_decision.allowed:
        print(
            f"[signet-entrypoint] ⚠ CUMULATIVE SESSION CAP EXCEEDED — projected "
            f"${cap_decision.projected_usd:.2f} + spent ${cap_decision.spent_so_far:.2f} = "
            f"${cap_decision.projected_usd + cap_decision.spent_so_far:.2f} vs cap "
            f"${cap_decision.session_cap_usd:.2f}. Dropping to ASK-FIRST (D-8-YOLOCAP): --approve is "
            f"DISABLED for this dispatch even though it was passed. A present operator may still "
            f"type 'approved' at the prompt below to proceed over cap; a non-interactive request "
            f"(no operator to ask) will be refused there.",
            file=sys.stderr,
        )
        approve_for_gate = False

    # (4) BLOCKING approval pause (D-03 / MODL-02) — AFTER the cost print/guardrail, BEFORE any
    #     ``.spawn()`` dispatch. This is the single gate all metered runs pass through. Declining
    #     aborts with NO dispatch — metered spend can never auto-launch (sponsor credits metered).
    #     ``approve_for_gate`` (not the raw ``approve`` param) is what reaches this call — see the
    #     cumulative-cap ask-first downgrade immediately above.
    if not _require_approval(approve_for_gate):
        raise SystemExit(1)

    # (5) Dispatch the gated remote run — STRICTLY AFTER ``_require_approval`` returned True
    # (MODL-02: the dispatch can NEVER precede the blocking approval pause, so a metered run can
    # never auto-launch). The launch ordering — config -> dry-run gate -> cost print -> APPROVE ->
    # dispatch — is now fully enforced for BOTH modes. We pass the config YAML TEXT BY VALUE (not a
    # local path): the ``configs/`` dir is not shipped into the container image, so a relative path
    # would not resolve there. The remote body re-parses + RE-VALIDATES this same text in-process
    # before any sustained spend (T-03-63). Outputs (checkpoints / samples) land on the Volume
    # (commit-or-vanish).
    #
    # D-10-DEF-17: every arm dispatches ASYNC via ``.spawn(...)``, then hands the returned
    # ``FunctionCall`` to ``_watch_dispatch``. ``.remote()`` was SYNC, and the server cancels
    # in-flight SYNC inputs whose owning client vanishes — it killed a healthy training round and a
    # healthy render the moment the dispatching agent returned. The bounded watch window
    # (``cfg.modal.dispatch_watch_seconds``) is what keeps the CHEAP aborts synchronous: an arch-gate
    # FAIL, a CPU-preflight refusal, a config raise or an import death still surfaces at this console
    # inside the window; anything longer-running outlives this client on purpose.
    # ⛔ THE FAMILY FENCE, and it must sit HERE — after the approval pause, before the routing chain
    # below. Family #4 serves exactly ONE mode. Every other arm in the chain is written for a family
    # that owns its own encode/loop/render, so a ``family: wan`` config reaching ``elif mode ==
    # "sample":`` would dispatch the LTX sampler with an LTX-shaped expectation of the checkpoint
    # layout — the fall-through that ``assert_wan_dryrun_geometry``'s refusal used to block for ALL
    # modes at once. Now that the dry-run gate PASSES for this family (it is a manifest gate, and
    # the manifest is fine), that blanket protection is gone and this fence replaces it. Placed
    # after ``_require_approval`` for consistency with the qwen gap refusals: MODL-02 is untouched
    # either way, and ``_wan_refuse_on_gaps`` always raises on a non-train mode (gap 1).
    if cfg.model.family == "wan" and mode != _WAN_SUPPORTED_MODE:
        _wan_refuse_on_gaps(cfg, mode=mode)

    if mode == "sample" and cfg.model.family == "h3":
        # Phase-10 (H3-07) H3 arm of the SAME mode: base-vs-adapter Ref2VA renders at one seed plus
        # the automated ``max|delta velocity|`` floor (D-10-SCOPEGUARD). Routed by family, NOT by a
        # new mode value. ``h3_sample`` takes the recipe BY VALUE (``config_yaml: str``) and
        # re-parses it in-container, exactly like ``sample``; the ``.spawn`` CALL sits strictly
        # after ``_require_approval`` (MODL-02) and the timeout is computed HERE, after approval.
        from signet_trainer.modal.fns import h3_sample

        h3_sample_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        # RULING (bundle PR-5 rework, config-coherence-0): h3.modal_gpu is the TRAIN-tier booking
        # lever (threaded into h3_preprocess / h3_train below) and is deliberately NOT threaded
        # here. h3_sample keeps its own pre-existing SIGNET_H3_SAMPLE_GPU env override (#55/house
        # audit PR#51), an orthogonal fix for a Qwen3-VL text-encode OOM on a 3-reference render
        # leg — set at fns.py IMPORT time via the decorator's gpu=H3_SAMPLE_GPU. Passing
        # gpu=cfg.h3.modal_gpu here would make .with_options(...) silently override that env var
        # with this field's default ("A100-80GB") on every run that has not ALSO escalated
        # modal_gpu, regressing #55 by composition. See h3_geometry.H3_DEFAULT_MODAL_GPU's
        # docstring for the full rationale.
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching h3_sample.spawn() (gated, family: h3); base+adapter mp4s, delta.json and "
            "index.html commit to signe-trainer-checkpoints under <output_dir>/samples_h3/<ts>/. "
            "The H3 arch gate fires inside the stage — there is no separate smoke step."
        )
        fc = h3_sample.with_options(timeout=h3_sample_timeout_s).spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "h3_sample",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif mode == "sample" and cfg.model.family == "qwen_edit":
        # Phase-11 qwen_edit arm of the SAME mode: the A/B-prompt x checkpoint-BAND render grid.
        # Routed by family, NOT by a new mode value — six dispatches, one gate, one ledger.
        # ``qwen_edit_sample`` takes the recipe BY VALUE (``config_yaml: str``) and re-parses it
        # in-container exactly like ``sample`` and ``h3_sample``; the ``.spawn`` CALL sits strictly
        # after ``_require_approval`` (MODL-02) and the timeout is computed HERE, after approval.
        from signet_trainer.modal.fns import qwen_edit_sample

        _qwen_edit_stage_readiness("sample")
        _qwen_edit_refuse_on_gaps(cfg, mode="sample")
        qwen_edit_sample_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching qwen_edit_sample.spawn() (gated, family: qwen_edit); the band's renders "
            "commit to signe-trainer-checkpoints under <output_dir>/samples_qwen_edit/<render-key>/. "
            "The arch gate fires inside the stage — there is no separate smoke step.\n"
            "[signet-entrypoint] ⚠ THIS IS A FULL, METERED RENDER. The generate call landed with "
            "the sampler, so the stage runs end to end: ~40.9 GiB transformer + Qwen2.5-VL + VAE "
            "loaded once, then every cell of (base + each band member) x (held-out inputs) x (the "
            "§8 A/B modes) at 30 steps and true_cfg 4.0. It does NOT abort in the first seconds — "
            "the earlier banner here said so while the render was a declared stub, and that stub "
            "is gone. Budget the container accordingly; the §8 inference settings are LOCKED in "
            "models/qwen_edit_pipeline.QWEN_EDIT_RENDER_RECIPE (steps 30, true_cfg 4.0 + CFGNorm, "
            "the STATIC scheduler reparameterisation pinned AFTER pipeline construction and "
            "re-verified per render, LoRA strength 1.0, reference into BOTH encodes) rather than "
            "read from this config. Cells already on the Volume are resumed, not re-rendered."
        )
        fc = qwen_edit_sample.with_options(timeout=qwen_edit_sample_timeout_s).spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "qwen_edit_sample",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif mode == "sample":
        # Phase-4 base-vs-LoRA grid (D-RUN-1 part 1) — reuses the EXACT gate above; the
        # ``sample.with_options(...).spawn`` CALL sits strictly after ``_require_approval`` (MODL-02).
        from signet_trainer.modal.fns import sample

        # (#5 AUDIT) DERIVE the Modal function timeout from the config (D-NOHARDCODE) instead of the
        # hardcoded 24h decorator ceiling: a driver-level CUDA hang on a ~1.5h render would otherwise
        # burn to 24h (~$40) silently. est_hours * timeout_margin bounds the metered shell; only
        # sample/preprocess get this config-derived override — train() keeps its 24h decorator (#5).
        # Bound to a local so the dispatch stays config-first, and computed HERE — STRICTLY after
        # _require_approval (MODL-02: the .with_options timeout never precedes the approval pause).
        sample_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching sample.spawn() (gated); base+LoRA mp4s + index.html commit "
            "to signe-trainer-checkpoints under <output_dir>/samples/."
        )
        fc = sample.with_options(timeout=sample_timeout_s).spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "sample",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif mode == "preprocess" and cfg.model.family == "h3":
        # Phase-10 (H3-07) H3 arm of the SAME mode: the signet-native MiniMax-H3 Ref2VA pre-encode
        # (there is no canonical H3 encoder anywhere, so the enochiatron "never write a custom
        # encoder" landmine does not apply — 10-07). Routed by family, NOT by a new mode value.
        #
        # Unlike the two config-text stages, ``h3_preprocess`` takes 17 REQUIRED kwargs with no
        # defaults, so the whole threading burden sits in ``_h3_encode_params`` — bound to a local
        # HERE, before the dispatch, so a field-read failure is a named abort rather than a
        # traceback out of a ``.spawn()`` expression (the 09-07 T3 burned-gate lesson).
        from signet_trainer.modal.fns import h3_preprocess

        h3_params = _h3_encode_params(cfg)
        assert h3_params is not None  # family == "h3" was just checked by this arm's condition
        h3_preprocess_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        # The BOOKED GPU comes from the config (h3.modal_gpu, coherence-checked against the budget
        # triple AND modal.hourly_rate_usd at load) — never the fns.py decorator literal, so an
        # H200 escalation really is a YAML edit. Bound to a local, computed HERE, strictly after
        # approval (MODL-02).
        h3_gpu = cfg.h3.modal_gpu
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching h3_preprocess.spawn() (gated, family: h3); the two-phase encode writes "
            "h3_latents/ + h3_conditions/ + h3_reference_latents/ (+ h3_audio_latents/ when "
            "requested) to the dataset Volume under cfg.data.preprocessed_data_root and commits it. "
            "The H3 arch gate fires inside the stage, before a single frame is decoded."
        )
        fc = h3_preprocess.with_options(timeout=h3_preprocess_timeout_s, gpu=h3_gpu).spawn(
            **h3_params
        )
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "h3_preprocess",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif mode == "preprocess" and cfg.model.family == "qwen_edit":
        # Phase-11 qwen_edit arm of the SAME mode: the signet-native Qwen-Image-Edit pre-encode.
        # There is no canonical Qwen encoder anywhere (not in-repo, not upstream), so the "never
        # write a custom encoder" landmine does not apply — there is nothing canonical to prefer.
        # Routed by family, NOT by a new mode value.
        #
        # Like ``h3_preprocess`` and unlike the two config-text stages, ``qwen_edit_preprocess``
        # takes REQUIRED kwargs with no defaults, so the whole threading burden sits in
        # ``_qwen_edit_encode_params`` — bound to a local HERE, before the dispatch, so a field-read
        # failure is a named abort rather than a traceback out of a ``.spawn()`` expression.
        from signet_trainer.modal.fns import qwen_edit_preprocess

        _qwen_edit_stage_readiness("preprocess")
        qwen_edit_params = _qwen_edit_encode_params(cfg)
        assert qwen_edit_params is not None  # family was just checked by this arm's condition
        qwen_edit_preprocess_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching qwen_edit_preprocess.spawn() (gated, family: qwen_edit); the two-phase "
            "encode writes qwen_edit_conditions/ then qwen_edit_latents/ + "
            "qwen_edit_control_latents/ to the dataset Volume under cfg.data.preprocessed_data_root "
            "and commits it. The arch gate fires inside the stage, before a single image is opened, "
            "and releases the transformer before the Qwen2.5-VL encoder is loaded."
        )
        fc = qwen_edit_preprocess.with_options(timeout=qwen_edit_preprocess_timeout_s).spawn(
            **qwen_edit_params
        )
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "qwen_edit_preprocess",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif mode == "preprocess" and cfg.model.family == "ltx" and cfg.model.ltx_generation == "2.5":
        # LTX-2.5 Stage 1 (issue #53) arm of the SAME mode: the canonical v1.2.0 pre-encode
        # (ltx25_preprocess, running upstream's OWN process_dataset.py from /opt/LTX-25, on
        # ltx25_gpu_image — LTX25_STAGE1_DESIGN.md §5). Routed by ltx_generation, NOT by a new
        # mode value, mirroring the family-routing precedent above (h3/qwen_edit's own preprocess
        # arms) — family stays "ltx" for both generations (§0), so this is the ONE arm keyed on
        # ltx_generation instead of family.
        from signet_trainer.modal.fns import ltx25_preprocess

        if cfg.conditioning.mode not in ("none", "single_frame", "multi_frame"):
            raise SystemExit(
                f"[signet-entrypoint] conditioning.mode={cfg.conditioning.mode!r} is out of "
                "Stage-1 scope for LTX-2.5 (issue #53 §9 — ic_lora/inpaint/audio_to_video-on-2.5 "
                "are not implemented by this PR; ltx25_preprocess takes no reference/mask/audio "
                "encode params at all). Set model.ltx_generation: '2.3' for that conditioning "
                "mode, or file an issue for Stage 2+. Aborting pre-approval, no dispatch."
            )
        ltx25_preprocess_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching ltx25_preprocess.spawn() (gated, ltx_generation: '2.5'); the v1.2.0 "
            "canonical encode writes {latents,conditions}/ + PROVENANCE.json to the dataset "
            "Volume under cfg.data.preprocessed_data_root."
        )
        fc = ltx25_preprocess.with_options(timeout=ltx25_preprocess_timeout_s).spawn(
            metadata_path=cfg.data.metadata_path,
            resolution_buckets=_parse_resolution_buckets(cfg.data.resolution_buckets),
            output_dir=cfg.data.preprocessed_data_root,
            model_id=cfg.model.model_id,
            gemma_root=cfg.model.text_encoder_id,
            video_vae_path=cfg.ltx25.video_vae_path,
            audio_vae_path=cfg.ltx25.audio_vae_path,
            overwrite=cfg.data.preprocess_overwrite,
        )
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "ltx25_preprocess",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif mode == "preprocess":
        # Phase-8 canonical pre-encode as a first-class GATED mode (D-8-PREPROC) — retires the
        # hardcoded hourly-rate drift in the throwaway scripts/_encode_demo*.py by routing the
        # encode through THIS shared gate (the cost line above already read cfg.modal.hourly_rate_usd).
        # The ``preprocess.spawn`` CALL sits strictly after ``_require_approval`` (MODL-02) — no
        # metered encode can auto-launch. ALL encode params come from the loaded cfg (config-first,
        # D-NOHARDCODE): metadata_path + resolution_buckets + output_dir live on cfg.data, but the
        # reference params live on cfg.CONDITIONING (NOT cfg.data — extra="forbid" would raise
        # AttributeError here, AFTER approval, a burned gate). resolution_buckets are WxHxF strings on
        # the config, pre-parsed to the (F, H, W) tuples the canonical encoder wants (mirroring the
        # retired drivers), NOT passed as strings.
        from signet_trainer.modal.fns import preprocess

        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching preprocess.spawn() (gated); the canonical process_dataset.py "
            "encode writes {latents,conditions}/ (+ reference_latents/ for ic_lora) to the dataset "
            "Volume under cfg.data.preprocessed_data_root."
        )
        reference_column, reference_downscale_factor = _reference_encode_params(cfg)
        mask_column, mask_output_dir_name = _mask_encode_params(cfg)
        with_audio = _audio_encode_params(cfg)
        # (#5 AUDIT) same config-derived timeout as the sample arm (D-NOHARDCODE): bound the encode
        # to est_hours * timeout_margin instead of the hardcoded 24h decorator ceiling. Computed
        # STRICTLY after _require_approval (MODL-02).
        preprocess_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        fc = preprocess.with_options(timeout=preprocess_timeout_s).spawn(
            metadata_path=cfg.data.metadata_path,
            resolution_buckets=_parse_resolution_buckets(cfg.data.resolution_buckets),
            output_dir=cfg.data.preprocessed_data_root,
            reference_column=reference_column,
            reference_downscale_factor=reference_downscale_factor,
            mask_column=mask_column,
            mask_output_dir_name=mask_output_dir_name,
            with_audio=with_audio,
            # #35 step 3: the ONE operator-visible re-encode knob, config-driven (D-NOHARDCODE) —
            # default False keeps this dispatch byte-identical to before the field existed.
            overwrite=cfg.data.preprocess_overwrite,
        )
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "preprocess",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif mode == "fuse":
        # Phase-9 (INPAINT, GATE-SPEC rev 2 build-order step 3): the In-Outpainting scaffold fuse —
        # a CPU-ONLY Modal job (no gpu=; big-RAM tensor arithmetic) that writes the fused base to
        # the weights Volume. Reuses THIS exact gate (cost print from cfg.modal reflects the fuse
        # job's CPU-time estimate — use a fuse-specific config copy with honest est_hours). The
        # ``fuse.spawn`` CALL sits strictly after ``_require_approval`` (MODL-02).
        from signet_trainer.modal.fns import fuse

        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching fuse.spawn() (CPU-only In-Outpainting fuse; header keycheck -> fuse -> "
            "verify_fused_metadata -> weights Volume commit)."
        )
        fc = fuse.spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "fuse",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif mode == "restore":
        # BK-01 (09.1-08): the GATED rehydrate arm — a CPU-only Modal job (no gpu=; off the training
        # A100, D-BK-3) that downloads cfg.backup's mirrored checkpoints back onto the checkpoints
        # Volume and commits it (commit-or-vanish, D-BK-4). Reuses THIS exact gate — the cost print
        # (near-zero CPU) already fired above and the ``restore.spawn`` CALL sits strictly after
        # ``_require_approval`` (MODL-02: a Volume-mutating restore can never precede the approval
        # pause). An enabled destination='cloud' config already fail-fasted at the gate's config-load
        # step (09.1-07 validator) — it never reaches here.
        from signet_trainer.modal.fns import restore

        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching restore.spawn() (CPU-only, off the A100); it rehydrates the checkpoints "
            "Volume from cfg.backup (1:1 copy-back) and commits it (additive — no existing checkpoint "
            "deleted)."
        )
        fc = restore.spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "restore",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif mode == "backup":
        # BK-01 (09.1-08): the GATED mirror arm — a CPU-only Modal job (no gpu=; off the training A100,
        # D-BK-3) that mirrors ONLY new complete checkpoints from the Volume to cfg.backup (additive,
        # 1:1 layout, token-safe). CPU cost ≈ near-zero but the cost line STILL printed above and the
        # action IS logged (single-gate law + D-BK-3 accounting). The ``backup_sync.spawn`` CALL sits
        # strictly after ``_require_approval`` (MODL-02). The harness drives PERIODIC sync via
        # ``--mode backup --approve`` under the yolo blanket — the SAME mechanism as
        # ``--mode sample --approve`` (no new gate, no second launch path).
        from signet_trainer.modal.fns import backup_sync

        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching backup_sync.spawn() (CPU-only, off the A100); it mirrors new complete "
            "checkpoints to cfg.backup (additive, 1:1 layout, HF token from the Modal secret only)."
        )
        fc = backup_sync.spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "backup_sync",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif cfg.model.family == "h3":
        # Phase-10 (H3-07) H3 arm of the DEFAULT ``train`` mode. Routed by family, NOT by a new mode
        # value. ``h3_train`` takes the recipe BY VALUE and re-parses it in-container, so there is
        # nothing to thread; the ``.spawn`` CALL sits strictly after ``_require_approval`` (MODL-02).
        #
        # DELIBERATE DEVIATION from the LTX ``train`` arm, which keeps the 24h decorator ceiling: the
        # H3 arm bounds the metered shell at ``est_hours * timeout_margin`` like the sample and
        # preprocess arms (#5 AUDIT, D-NOHARDCODE). A driver-level hang on a 61.7 GiB model is
        # expensive enough that riding the 24h ceiling is the wrong default — the LTX arm's exemption
        # predates that figure. Computed HERE, strictly after approval.
        from signet_trainer.modal.fns import h3_train

        h3_train_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        # Same booking rule as the h3_preprocess arm: the GPU is cfg.h3.modal_gpu (the
        # coherence-checked half of the H200 escalation lever, against BOTH the budget triple and
        # modal.hourly_rate_usd), not the fns.py decorator literal. Bound to a local, computed
        # HERE, strictly after approval (MODL-02).
        h3_gpu = cfg.h3.modal_gpu
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching h3_train.spawn() (gated, family: h3); checkpoints commit to "
            "signe-trainer-checkpoints under <output_dir>/. The H3 arch gate + the CPU preflight "
            "both fire inside the stage, before the 61.7 GiB load."
        )
        fc = h3_train.with_options(timeout=h3_train_timeout_s, gpu=h3_gpu).spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "h3_train",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif cfg.model.family == "qwen_edit":
        # Phase-11 qwen_edit arm of the DEFAULT ``train`` mode. Routed by family, NOT by a new mode
        # value. ``qwen_edit_train`` takes the recipe BY VALUE and re-parses it in-container, so
        # there is nothing to thread; the ``.spawn`` CALL sits strictly after ``_require_approval``.
        #
        # Bounds the metered shell at ``est_hours * timeout_margin`` — the H3 arm's deviation from
        # the LTX 24h decorator ceiling, taken here for the same reason: a driver-level hang on a
        # ~38 GiB model riding the ceiling is the wrong default.
        from signet_trainer.modal.fns import qwen_edit_train

        _qwen_edit_stage_readiness("train")
        _qwen_edit_refuse_on_gaps(cfg, mode="train")
        qwen_edit_train_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching qwen_edit_train.spawn() (gated, family: qwen_edit); checkpoints commit to "
            "signe-trainer-checkpoints under <output_dir>/. The arch gate fires inside the stage "
            "and must see 840 LoRA targets across all fourteen leaves before any training spend.\n"
            f"[signet-entrypoint] ⚠ COST BASIS: est_hours={cfg.modal.est_hours:g} is a DECLARED "
            f"estimate, not a measurement — no steps/hour figure has been recorded for "
            f"Qwen-Image-Edit on any card in this program, and it shows up a second time as "
            f"qwen_edit.max_packed_rows=0 (row ceiling DISABLED because no OOM boundary is "
            f"measured either). The guardrail arithmetic above is real and binding; the INPUT to it "
            f"is a declared number. After this round, set est_hours from the observed "
            f"steps/hour over {cfg.training.max_steps} steps and the estimate becomes measured."
        )
        fc = qwen_edit_train.with_options(timeout=qwen_edit_train_timeout_s).spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "qwen_edit_train",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif cfg.model.family == "wan":
        # Family #4's ONLY arm, inside the DEFAULT ``train`` mode. Routed by family, not by a
        # seventh --mode value — six dispatches, one gate, one ledger, unchanged.
        #
        # ⛔ ``.spawn(**wan_params)``, NOT ``.spawn(config_text)``, and the difference is forced
        # rather than stylistic: ``wan_musubi_image`` carries musubi's ``pydantic==1.10.13``
        # (train_kohya.py:37) while ``config/load.load_config_from_text`` needs pydantic>=2.10.4, so
        # this is the one train stage that CANNOT re-parse its config in-container. The whole
        # threading burden therefore sits in ``_wan_train_params`` — bound to a local HERE, before
        # the dispatch, so a field-read failure is a named abort rather than a traceback out of a
        # ``.spawn()`` expression (the 09-07 T3 burned-gate lesson) — and it is also where the
        # dataset TOML is RENDERED, which is what makes "only the dataset changed between rounds" a
        # property of the artifacts rather than of somebody's memory.
        from signet_trainer.modal.fns import wan_train

        wan_params = _wan_train_params(cfg)
        assert wan_params is not None  # family == "wan" was just checked by this arm's condition
        wan_train_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching wan_train.spawn() (gated, family: wan); ONE stage running musubi's three "
            "passes in order — wan_cache_latents.py, wan_cache_text_encoder_outputs.py, then "
            "accelerate launch wan_train_network.py — each with check=True, unlike the "
            "transcription (train_kohya.py never inspects a return code, so a failed cache pass "
            "there trains on an empty cache). The rendered dataset TOML and the adapters commit to "
            "signe-trainer-checkpoints under <output_dir>/.\n"
            f"[signet-entrypoint] ⚠ COST BASIS: est_hours={cfg.modal.est_hours:g} is a DECLARED "
            "estimate, not a measurement — no steps/hour figure has been recorded for Wan on any "
            "card in this program, and musubi is EPOCH-driven so training.max_steps="
            f"{cfg.training.max_steps} is signet's accounting basis and is not passed to the "
            "runner. The wan batch line above prints the clip-instance arithmetic that IS knowable; "
            "the media-file count that completes it lives on the dataset Volume."
        )
        fc = wan_train.with_options(timeout=wan_train_timeout_s).spawn(**wan_params)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "wan_train",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    elif cfg.model.family == "ltx" and cfg.model.ltx_generation == "2.5":
        # LTX-2.5 Stage 1 (issue #53) arm of the DEFAULT ``train`` mode. Routed by
        # ltx_generation, NOT by family (family stays "ltx" for both generations, §0) — the ONE
        # train-arm branch keyed this way, mirroring the preprocess arm above. ``ltx25_train``
        # takes the recipe BY VALUE and re-parses it in-container (same shape as the plain LTX
        # arm below), so there is nothing to thread; the ``.spawn`` CALL sits strictly after
        # ``_require_approval`` (MODL-02).
        from signet_trainer.modal.fns import ltx25_train

        ltx25_train_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching ltx25_train.spawn() (gated, ltx_generation: '2.5'); checkpoints commit "
            "to signe-trainer-checkpoints under <output_dir>/.\n"
            "[signet-entrypoint] ⚠ HONESTY (D3 open): the metadata-driven arch gate RECORDS "
            "observed values from the checkpoint's own embedded config — it does not assert them "
            "against any EXPECTED_*_25 constant, because none exists yet. This is not live-tested "
            "against real LTX-2.5/Gemma-4 weights (issue #53 D3)."
        )
        fc = ltx25_train.with_options(timeout=ltx25_train_timeout_s).spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "ltx25_train",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
    else:
        from signet_trainer.modal.fns import train

        # issue #45 PR-2: train()'s 24h-decorator exemption is RETIRED — it now dispatches with the
        # SAME config-derived timeout bound (est_hours * timeout_margin) every other GPU arm above
        # uses, computed HERE, strictly after _require_approval (MODL-02), and matching the SAME
        # bounded_hours the guardrail_check call above priced this arm's worst case against. Without
        # this the printed/guardrailed estimate would price a bound the dispatch never actually used.
        train_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching train.spawn() (gated); checkpoints commit to "
            "signe-trainer-checkpoints. Phase 2's download/smoke/encode runs are driven manually "
            "through this same gate."
        )
        fc = train.with_options(timeout=train_timeout_s).spawn(config_text)
        _watch_dispatch(
            fc,
            cfg.modal.dispatch_watch_seconds,
            "train",
            ledger_path=ledger_path,
            est_usd=decision.est_usd,
        )
