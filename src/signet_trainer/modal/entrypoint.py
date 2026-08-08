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

import os
import sys
from pathlib import Path

from signet_trainer.conditioning.h3_geometry import max_packed_rows_for_budget
from signet_trainer.config.load import load_config_from_text
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

# Register the @app.function stubs (train / preprocess / download_weights / sample / ...) on the app
# graph at MODULE-LOAD time. ``modal run -m signet_trainer.modal.entrypoint`` builds the app from what
# is imported when this module loads; without this side-effecting import the functions are never
# registered and ``train.spawn()`` fails with "Function has not been hydrated". The lazy re-import
# inside main() then resolves to this same registered function. Placement does NOT affect MODL-02:
# the ``.spawn()`` CALL still lives strictly after the approval pause in main().
from signet_trainer.modal import fns as _fns  # noqa: F401 — import side-effect registers app functions


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


def _watch_dispatch(fc: object, watch_seconds: float, label: str) -> None:
    """Post-dispatch handling for a SPAWNED (async) gated run — D-10-DEF-17.

    Every arm in ``main()`` now dispatches with ``.spawn(...)`` rather than ``.remote(...)``. The
    verb is the whole fix: ``_Function.remote`` delegates to ``_call_function``, which carries
    ``FUNCTION_CALL_INVOCATION_TYPE_SYNC``, and the invocation type is sent to the SERVER at call
    creation — Modal cancels in-flight SYNC inputs whose owning client disappears. On 2026-08-07 that
    killed two healthy runs (a training round at step 300 and a measurement render at denoise step
    13) at the moment the dispatching agent returned. ``_Function.spawn`` carries
    ``FUNCTION_CALL_INVOCATION_TYPE_ASYNC``, which is not cancelled that way.

    Four things happen here, in order:

    1. Print the ``FunctionCall`` id — the handle that OUTLIVES this client, and the whole point of
       the change: any later process can re-attach with ``modal.FunctionCall.from_id("<id>")``
       (``poll_function`` uses ``clear_on_success=False``, so a client that did not dispatch the call
       can still collect its output).
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
    if "--detach" not in sys.argv:
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
        print(
            f"[signet-entrypoint] {label} still RUNNING after {watch_seconds:g}s — this client is "
            f"disengaging, the run is NOT cancelled (async dispatch). Track it via FunctionCall id "
            f"{fc.object_id}, `modal app logs <app-id>`, or the output Volume."
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

    ``h3_preprocess`` deliberately declares ALL 15 parameters REQUIRED with no defaults (10-10), so a
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
    arithmetic (D-NOHARDCODE: an H200 escalation is then a YAML edit to the budget triple).

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


def _qwen_edit_config_gaps(cfg: object, *, mode: str) -> list[str]:
    """Every DECLARED gap between a validated ``family: qwen_edit`` config and what the stage needs.

    A config that loads is not the same as a config that can drive a stage, and the difference is
    worth naming rather than discovering inside a paid container. Each entry is one sentence saying
    what is missing, why the stage needs it, and what would land it.

    Two gaps exist today and BOTH are real findings about the schema, not about this file:

    **(1) ``config_source`` for a single-file ``model_id`` — every mode.** The shipped example config
    sets ``model_id: qwen_image_edit_2511_bf16.safetensors``, a bare file, and
    ``models/qwen_edit_loader.load_qwen_edit_transformer`` documents that ``config_source`` is
    REQUIRED on that path: ``from_single_file`` with no ``config=`` calls ``fetch_diffusers_config``
    -> ``infer_diffusers_model_type``, which has **no Qwen branch at all** on the pinned diffusers and
    falls through to ``model_type = "v1"``, whose default is
    ``stable-diffusion-v1-5/stable-diffusion-v1-5``. diffusers would then reach the HUB for a Stable
    Diffusion v1.5 ``transformer/`` config from inside a metered container. ai-toolkit works around
    it by hard-coding the hub id ``"Qwen/Qwen-Image"``; signet takes the value from the caller so an
    air-gapped Volume-local directory is expressible and ``local_files_only`` can stay on.

    The field that would carry it is ``model.pipeline_root_id`` — and ``config/schema.py``'s
    ``_FAMILY_ONLY_MODEL_IDS`` maps it to ``frozenset({"h3"})``, so setting it under
    ``family: qwen_edit`` is REJECTED at config load with *"that ID is only read under family
    {'h3'} and would be silently ignored here"*. That is correct behaviour for a field with no
    consumer; it is now a field WITH a consumer, so the fix is one entry in that map, not a
    workaround here. A DIRECTORY-form ``model_id`` sidesteps the gap entirely — the config travels
    with the weights — which is why this gap is CONDITIONAL and not a blanket refusal.

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

    ⛔ ``control_dirs`` / ``blank_slots`` have NO source in the schema today, so this function
    currently always aborts through ``_qwen_edit_refuse_on_gaps`` above. That refusal is the deliberate
    output: see ``_qwen_edit_config_gaps`` gap (2) for why a convention must not be invented here. The
    remaining sixteen parameters are threaded and correct, so landing the field is a one-line change
    on both sides rather than a rewrite of this seam.
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

    # (2) Dry-run hard gate (CONF-03) on the already-loaded cfg — must pass before ANY remote
    #     dispatch. Non-zero -> abort.
    rc = run_dryrun(cfg)
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
    cpu_only_mode = mode in ("fuse", "restore", "backup")
    if cpu_only_mode:
        decision = guardrail_check(
            hourly_rate_usd=cfg.modal.cpu_hourly_rate_usd,
            est_hours=cfg.modal.est_hours,
            cost_guardrail_usd=cfg.modal.cost_guardrail_usd,
        )
        print(
            f"[signet-entrypoint] CPU-only mode {mode!r} (no A100) — ~near-zero cost, estimated from "
            f"cpu_hourly_rate_usd=${cfg.modal.cpu_hourly_rate_usd:.2f}/hr (NOT the A100 rate)."
        )
    else:
        decision = guardrail_check(
            hourly_rate_usd=cfg.modal.hourly_rate_usd,
            est_hours=cfg.modal.est_hours,
            cost_guardrail_usd=cfg.modal.cost_guardrail_usd,
        )
    print(format_cost_line(decision))
    if not decision.allowed:
        print(
            "[signet-entrypoint] cost guardrail BLOCKED the launch (over budget) — "
            "no remote dispatch.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # (4) BLOCKING approval pause (D-03 / MODL-02) — AFTER the cost print/guardrail, BEFORE any
    #     ``.spawn()`` dispatch. This is the single gate all metered runs pass through. Declining
    #     aborts with NO dispatch — metered spend can never auto-launch (sponsor credits metered).
    if not _require_approval(approve):
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
    if mode == "sample" and cfg.model.family == "h3":
        # Phase-10 (H3-07) H3 arm of the SAME mode: base-vs-adapter Ref2VA renders at one seed plus
        # the automated ``max|delta velocity|`` floor (D-10-SCOPEGUARD). Routed by family, NOT by a
        # new mode value. ``h3_sample`` takes the recipe BY VALUE (``config_yaml: str``) and
        # re-parses it in-container, exactly like ``sample``; the ``.spawn`` CALL sits strictly
        # after ``_require_approval`` (MODL-02) and the timeout is computed HERE, after approval.
        from signet_trainer.modal.fns import h3_sample

        h3_sample_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching h3_sample.spawn() (gated, family: h3); base+adapter mp4s, delta.json and "
            "index.html commit to signe-trainer-checkpoints under <output_dir>/samples_h3/<ts>/. "
            "The H3 arch gate fires inside the stage — there is no separate smoke step."
        )
        fc = h3_sample.with_options(timeout=h3_sample_timeout_s).spawn(config_text)
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "h3_sample")
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
            "[signet-entrypoint] ⛔ EXPECT A NAMED ABORT IN THE FIRST SECONDS: the GENERATE call "
            "(inference/qwen_edit_layout.render_qwen_edit_sample) is a DECLARED STUB. Everything "
            "around it is real — render key, landed-check, checkpoint band, column plan, gallery — "
            "and the stage reaches the stub AFTER its probes and BEFORE any weight load, so the "
            "container costs seconds. The stub's message names what lands it, chiefly the §8 "
            "inference settings (steps 30, true_cfg 4.0 + CFGNorm, STATIC shift 3.0 overridden onto "
            "pipeline.scheduler AFTER get_generation_pipeline() rebuilds it, LoRA strength 1.0, "
            "reference into BOTH the positive and negative encode)."
        )
        fc = qwen_edit_sample.with_options(timeout=qwen_edit_sample_timeout_s).spawn(config_text)
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "qwen_edit_sample")
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
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "sample")
    elif mode == "preprocess" and cfg.model.family == "h3":
        # Phase-10 (H3-07) H3 arm of the SAME mode: the signet-native MiniMax-H3 Ref2VA pre-encode
        # (there is no canonical H3 encoder anywhere, so the enochiatron "never write a custom
        # encoder" landmine does not apply — 10-07). Routed by family, NOT by a new mode value.
        #
        # Unlike the two config-text stages, ``h3_preprocess`` takes 15 REQUIRED kwargs with no
        # defaults, so the whole threading burden sits in ``_h3_encode_params`` — bound to a local
        # HERE, before the dispatch, so a field-read failure is a named abort rather than a
        # traceback out of a ``.spawn()`` expression (the 09-07 T3 burned-gate lesson).
        from signet_trainer.modal.fns import h3_preprocess

        h3_params = _h3_encode_params(cfg)
        assert h3_params is not None  # family == "h3" was just checked by this arm's condition
        h3_preprocess_timeout_s = int(cfg.modal.est_hours * cfg.modal.timeout_margin * 3600)
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching h3_preprocess.spawn() (gated, family: h3); the two-phase encode writes "
            "h3_latents/ + h3_conditions/ + h3_reference_latents/ (+ h3_audio_latents/ when "
            "requested) to the dataset Volume under cfg.data.preprocessed_data_root and commits it. "
            "The H3 arch gate fires inside the stage, before a single frame is decoded."
        )
        fc = h3_preprocess.with_options(timeout=h3_preprocess_timeout_s).spawn(**h3_params)
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "h3_preprocess")
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
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "qwen_edit_preprocess")
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
        )
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "preprocess")
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
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "fuse")
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
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "restore")
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
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "backup_sync")
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
        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching h3_train.spawn() (gated, family: h3); checkpoints commit to "
            "signe-trainer-checkpoints under <output_dir>/. The H3 arch gate + the CPU preflight "
            "both fire inside the stage, before the 61.7 GiB load."
        )
        fc = h3_train.with_options(timeout=h3_train_timeout_s).spawn(config_text)
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "h3_train")
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
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "qwen_edit_train")
    else:
        from signet_trainer.modal.fns import train

        print(
            "[signet-entrypoint] APPROVED — config valid, dry-run passed, cost within guardrail. "
            "Dispatching train.spawn() (gated); checkpoints commit to "
            "signe-trainer-checkpoints. Phase 2's download/smoke/encode runs are driven manually "
            "through this same gate."
        )
        fc = train.spawn(config_text)
        _watch_dispatch(fc, cfg.modal.dispatch_watch_seconds, "train")
