"""The local LTX training runner -- BETA / UNTESTED (see package docstring).

Mirrors the Modal ``train`` arm's validated sequence (``modal/fns.py``) with local paths and
``checkpoints_vol=None``:

    load config -> REFUSALS (unsupported surface fails loud, never half-runs)
      -> resolve local paths (weights / data / output -- existence-checked BEFORE any load)
      -> dry-run shape gate (the same ``signet-dryrun`` CPU gate)
      -> BETA banner + plan print -> explicit approve gate (``--approve`` or interactive)
      -> load components -> 6-check architecture validation gate -> free Gemma (two-phase VRAM)
      -> adapter obtain (reuse the gate's roundtrip-proved adapter -- no double-inject)
      -> block-swap offloader (``offload.blocks_to_swap``, the <80 GB enabling knob)
      -> PrecomputedDataset -> optimizer/scheduler/schedule -> CheckpointManager
      -> ``train_loop(..., checkpoints_vol=None)``

Module top stays STDLIB + config-loader light so the refusal/plan logic is unit-testable on
CPU with no torch/ltx_core installed. Heavy imports live inside :func:`run`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from signet_trainer.local import BETA_BANNER, ROADMAP_ISSUE

#: The support matrix the refusal gate enforces. A loud refusal beats a silent half-run.
SUPPORTED_FAMILY = "ltx"
SUPPORTED_CONDITIONING_MODES = ("none", "single_frame", "multi_frame")

#: Exit codes (contract, mirrors the watcher convention of distinct terminal codes).
EXIT_REFUSED = 2
EXIT_NOT_APPROVED = 3


def refusals(config: Any) -> list[str]:
    """Every reason this config cannot run on the local-beta path (empty == runnable).

    Pure + CPU-only on purpose: unit tests assert each refusal without torch installed.
    """
    out: list[str] = []
    fam = config.model.family
    if fam != SUPPORTED_FAMILY:
        out.append(
            f"model.family={fam!r}: local beta supports ONLY the {SUPPORTED_FAMILY!r} family. "
            f"The h3/qwen local paths are roadmap items -- {ROADMAP_ISSUE}"
        )
    mode = config.conditioning.mode
    if mode not in SUPPORTED_CONDITIONING_MODES:
        out.append(
            f"conditioning.mode={mode!r}: local beta supports {SUPPORTED_CONDITIONING_MODES}. "
            f"ic_lora/inpaint/audio_to_video need their 3-source dataset + preflight discipline "
            f"ported (and tested) -- {ROADMAP_ISSUE}"
        )
    if getattr(config.conditioning, "frozen_adapter_path", None):
        out.append(
            "conditioning.frozen_adapter_path is set: frozen-adapter stacking is not wired on "
            f"the local path yet -- {ROADMAP_ISSUE}"
        )
    if config.validation.in_loop_sampling:
        out.append(
            "validation.in_loop_sampling=true: the in-loop sampler needs the two-phase cached-"
            "embeddings dance (Gemma pre-encode before the free) which is untested locally. Turn "
            f"it off for local runs; sample after training instead -- {ROADMAP_ISSUE}"
        )
    if mode == "multi_frame" and getattr(config.conditioning, "conditioning_items", None):
        # WR-04 parity with the Modal arm (fns.py): items are SAMPLE-only -- training
        # self-conditions (MultiFrameStrategy samples its own keyframes) and would silently
        # ignore them, the exact silently-ignored-config-block class the schema doctrine forbids.
        out.append(
            "conditioning_items are sample-only for multi_frame training (WR-04): training "
            "self-conditions and would silently ignore them. Remove conditioning_items from the "
            "training config (keyframe items belong in the sample config)."
        )
    return out


@dataclass
class LocalPaths:
    checkpoint_path: Path       # base model .safetensors
    text_encoder_path: Path     # Gemma dir
    data_root: Path             # pre-encoded dataset root (PrecomputedDataset layout)
    output_dir: Path            # checkpoints land here (CheckpointManager root)


def resolve_paths(config: Any, weights_root: str | None, output_root: str) -> LocalPaths:
    """Local path resolution -- the ONLY place the Modal recipe's Volume mounts are replaced.

    ``model.model_id`` / ``model.text_encoder_id`` resolve exactly as on Modal
    (``WEIGHTS_DIR / id``) but against ``--weights-root``; absolute ids pass through untouched.
    Existence is checked HERE, before any GPU/model work, with actionable messages.
    """
    def _resolve(ident: str, label: str) -> Path:
        p = Path(ident)
        if not p.is_absolute():
            if not weights_root:
                raise FileNotFoundError(
                    f"{label} {ident!r} is relative and no --weights-root was given. Pass "
                    "--weights-root <dir that holds your downloaded weights>, or use absolute "
                    "paths in the config."
                )
            p = Path(weights_root) / ident
        if not p.exists():
            raise FileNotFoundError(
                f"{label} not found at {p} -- download the weights locally first (see the "
                "'Local training (BETA)' README section for what to fetch)."
            )
        return p

    # Weights first (the "what do I download" flow), data root second, output last.
    checkpoint_path = _resolve(config.model.model_id, "model.model_id")
    text_encoder_path = _resolve(config.model.text_encoder_id, "model.text_encoder_id")
    data_root = Path(config.data.preprocessed_data_root)
    if not data_root.exists():
        raise FileNotFoundError(
            f"data.preprocessed_data_root not found at {data_root} -- the local runner consumes a "
            "PRE-ENCODED dataset (the PrecomputedDataset layout). Local pre-encoding is a roadmap "
            f"item ({ROADMAP_ISSUE}); today, encode via the gated Modal preprocess and "
            "`modal volume get` the result, or point at any dir with the same layout."
        )
    return LocalPaths(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        data_root=data_root,
        output_dir=Path(output_root) / config.output_dir,
    )


def plan_text(config: Any, paths: LocalPaths, vram_line: str) -> str:
    """The pre-approval plan print -- every load-bearing number as ``name: value`` (typed-state)."""
    t = config.training
    return (
        "[signet-local] PLAN (nothing has run yet)\n"
        f"  family: {config.model.family}    conditioning.mode: {config.conditioning.mode}\n"
        f"  model: {paths.checkpoint_path}\n"
        f"  text_encoder: {paths.text_encoder_path}\n"
        f"  data_root: {paths.data_root}\n"
        f"  output_dir: {paths.output_dir}\n"
        f"  max_steps: {t.max_steps}    checkpoint_every: {t.checkpoint_every}    "
        f"keep_checkpoints: {t.keep_checkpoints}\n"
        f"  lora: rank {config.lora.rank} / alpha {config.lora.alpha}    "
        f"blocks_to_swap: {config.offload.blocks_to_swap}\n"
        f"  {vram_line}\n"
        "  wall-clock: UNKNOWN on your hardware (no local s/it measurements exist yet -- that is "
        "what the beta label means; please report yours on the issue tracker)."
    )


def run(
    config_path: str,
    *,
    weights_root: str | None = None,
    output_root: str = ".",
    approve: bool = False,
    dry_run_only: bool = False,
) -> int:
    """Drive one local training run. Returns a process exit code (0 ok)."""
    from signet_trainer.config.load import load_config  # local import: keeps module top light

    print(BETA_BANNER)
    config = load_config(config_path)

    blockers = refusals(config)
    if blockers:
        print("[signet-local] REFUSED -- this config is outside the local-beta support matrix:")
        for b in blockers:
            print(f"  x {b}")
        return EXIT_REFUSED

    paths = resolve_paths(config, weights_root, output_root)

    # The same CPU-pure shape gate the Modal entrypoint runs (CONF-03) -- free, catches config
    # errors before any weight touches memory. run_dryrun NEVER raises: it returns non-zero and
    # prints its reason to stderr, so the rc check IS the gate (parity-review blocker, 2026-08-11).
    from signet_trainer.dryrun.shapes import run_dryrun

    if run_dryrun(config) != 0:
        print("[signet-local] REFUSED -- dry-run shape gate FAILED (reason printed above); "
              "nothing was loaded, nothing ran.")
        return EXIT_REFUSED
    print("[signet-local] dry-run shape gate PASSED (CPU, zero spend).")

    import torch  # heavy imports start here, AFTER refusals + path checks + dryrun

    if not torch.cuda.is_available():
        print("[signet-local] REFUSED -- torch.cuda.is_available() is False; local training "
              "requires a CUDA GPU.")
        return EXIT_REFUSED
    # Cold-path dependency probe (parity review): surface a missing training dep NOW, with an
    # install hint -- not after the user approves and the 22B spends 20 minutes loading.
    for dep, hint in (("peft", "pip install 'peft>=0.14'"),
                      ("bitsandbytes", "pip install bitsandbytes"),
                      ("ltx_trainer", "see README 'Model weights' + the pinned LTX-2 install"),):
        try:
            __import__(dep)
        except ImportError:
            print(f"[signet-local] REFUSED -- required training dependency {dep!r} is not "
                  f"importable ({hint}). Nothing was loaded.")
            return EXIT_REFUSED
    free_b, total_b = torch.cuda.mem_get_info()
    free_gib, total_gib = free_b / 2**30, total_b / 2**30
    vram_line = (
        f"vram: {free_gib:.1f} GiB free / {total_gib:.1f} GiB total"
        f"    (Modal reference point: the 22B peaked at 62.8 GiB at blocks_to_swap=16 on an "
        f"A100-80GB; smaller cards need DEEPER swap and are UNMEASURED)"
    )
    print(plan_text(config, paths, vram_line))
    if config.offload.blocks_to_swap == 0 and total_gib < 70:
        print(
            "[signet-local] WARNING: blocks_to_swap is 0 and this card has "
            f"{total_gib:.0f} GiB -- the 22B will almost certainly OOM. Raise offload."
            "blocks_to_swap in the config (a Tier-2 knob: your call, not a silent default)."
        )
    if dry_run_only:
        print("[signet-local] --dry-run-only: stopping before the approval gate. Nothing ran.")
        return 0

    # The approval gate -- same discipline as the Modal entrypoint (the meter here is your
    # wall-clock + electricity + an UNTESTED path, not dollars; the pause is still earned).
    if not approve:
        try:
            answer = input("[signet-local] Type 'approved' to start this UNTESTED local run: ").strip()
        except EOFError:
            answer = ""
        if answer.lower() != "approved":
            print("[signet-local] not approved -- aborting, nothing ran.")
            return EXIT_NOT_APPROVED

    # -- From here: the Modal train arm's sequence, verbatim in structure ----------------------
    import gc

    from signet_trainer.data.precomputed import PrecomputedDataset
    from signet_trainer.models.loader import load_ltxv_components
    from signet_trainer.offload.block_swap import BlockSwapOffloader
    from signet_trainer.train.checkpoint import CheckpointManager
    from signet_trainer.train.flow_match import FlowMatchingSchedule
    from signet_trainer.train.loop import (
        build_optimizer,
        build_scheduler,
        should_warm_start,
        train_loop,
    )
    from signet_trainer.train.validate_gate import run_validation_gate

    device = "cuda"
    components = load_ltxv_components(
        checkpoint_path=str(paths.checkpoint_path),
        text_encoder_path=str(paths.text_encoder_path),
        device=device,
        with_video_vae_decoder=False,  # in-loop sampling is refused above -- decoder never loads
    )

    passed, results, gate_adapter = run_validation_gate(
        components, config, checkpoint_path=str(paths.checkpoint_path), device=device
    )
    for r in results:
        gate = " [HARD GATE]" if r.hard_gate else ""
        print(f"[signet-local][gate] {r.name}: {r.status} -- {r.message} ({r.duration_s}s){gate}")
    if not passed:
        failed = [r.name for r in results if r.status != "PASS"]
        raise RuntimeError(
            f"[signet-local] architecture validation gate FAILED ({failed}) -- aborting before "
            "training (the same 6-check gate the Modal path runs)."
        )

    # Two-phase VRAM discipline (06-09 carry-forward): Gemma is needed only by the gate; the loop
    # reads precomputed conditions. Null the attr AND drop the local so assign=True storage frees.
    _text_encoder = getattr(components, "text_encoder", None)
    if _text_encoder is not None:
        components.text_encoder = None
        del _text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        print(
            "[signet-local] freed Gemma after the gate (two-phase VRAM discipline); cuda "
            f"allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB."
        )

    # Adapter obtain WITHOUT double-injecting (03-07): reuse the gate's roundtrip-proved adapter
    # on the Open-Q1 default; build + inject fresh only on the use_builder branch.
    forward_check = next((r for r in results if r.name == "check_forward_pass"), None)
    use_builder = bool(
        forward_check is not None and forward_check.details.get("open_q1") == "use_builder"
    )
    if use_builder:
        from ltx_trainer.model_builder import LTXModelConfigurator, SingleGPUModelBuilder
        from ltx_trainer.model_loader import LTXV_MODEL_COMFY_RENAMING_MAP

        from signet_trainer.lora.peft import P1_FF_LORA_TARGETS, build_lora_config, inject_lora

        base_transformer = SingleGPUModelBuilder(
            model_path=str(paths.checkpoint_path),
            configurator=LTXModelConfigurator(),
            renaming_map=LTXV_MODEL_COMFY_RENAMING_MAP,
        ).build()
        lora_config = build_lora_config(
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            targets=config.lora.target_modules or P1_FF_LORA_TARGETS,
        )
        model = inject_lora(base_transformer, lora_config)
    else:
        base_transformer = components.transformer
        if gate_adapter is None:
            raise RuntimeError(
                "[signet-local] validation gate passed but returned no injected adapter -- "
                "expected the check #5 PEFT model to reuse (03-07 double-inject fix)."
            )
        model = gate_adapter
        model.zero_grad(set_to_none=True)

    block_list = None
    for attr in ("transformer_blocks", "blocks"):
        block_list = getattr(base_transformer, attr, None)
        if block_list is not None:
            break
    if block_list is None:
        raise RuntimeError(
            "[signet-local] could not locate the transformer block ModuleList "
            "(tried .transformer_blocks / .blocks) -- required for the offloader."
        )
    BlockSwapOffloader(
        block_list,
        blocks_to_swap=config.offload.blocks_to_swap,
        device=torch.device(device),
    )

    dataset = PrecomputedDataset(str(paths.data_root))

    ckpt_manager = CheckpointManager(paths.output_dir, keep_n=config.training.keep_checkpoints)

    if should_warm_start(
        ckpt_manager.find_latest() is not None, config.training.init_adapter_path
    ):
        from signet_trainer.lora.peft import load_adapter_into

        init_dir = Path(output_root) / config.training.init_adapter_path
        load_adapter_into(model, init_dir)
        print(f"[signet-local][chain] warm-started from {init_dir} (fresh optimizer, step 0).")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps=config.training.max_steps)
    schedule = FlowMatchingSchedule(uniform_prob=config.training.uniform_prob)

    final_step = train_loop(
        model,
        dataset,
        optimizer,
        scheduler,
        schedule,
        ckpt_manager,
        config,
        None,  # checkpoints_vol=None -- the documented off-Modal case; saves are plain local dirs
        on_checkpoint=None,
    )
    print(
        f"[signet-local] DONE -- reached step {final_step}; checkpoints under {paths.output_dir}. "
        f"This was a BETA/UNTESTED path: please report how it went (good or bad): "
        "https://github.com/alvdansen/signet-trainer/issues"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    # Beta users run bare `python` on stock Windows consoles (cp1252) -- the WIDER repo assumes
    # PYTHONUTF8=1, so shield this entry point rather than crash on the first unicode print from
    # a shared module (errors="replace": degrade a glyph, never the run).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(
        prog="python -m signet_trainer.local",
        description="Local (off-Modal) LTX LoRA training -- BETA / UNTESTED. See issue #25.",
    )
    ap.add_argument("--config", required=True, help="run config YAML (same schema as Modal runs)")
    ap.add_argument("--weights-root", default=None,
                    help="dir holding the downloaded weights; model.model_id / text_encoder_id "
                         "resolve relative to it (absolute config paths bypass it)")
    ap.add_argument("--output-root", default=".",
                    help="checkpoints land under <output-root>/<config.output_dir> (default: cwd)")
    ap.add_argument("--approve", action="store_true",
                    help="skip the interactive approval prompt (the gate discipline still prints "
                         "the full plan first)")
    ap.add_argument("--dry-run-only", action="store_true",
                    help="stop after the refusal gate + shape gate + plan print; run nothing")
    args = ap.parse_args(argv)
    try:
        return run(
            args.config,
            weights_root=args.weights_root,
            output_root=args.output_root,
            approve=args.approve,
            dry_run_only=args.dry_run_only,
        )
    except FileNotFoundError as exc:
        # Path problems are the beta's most likely first contact -- a clean actionable line, not
        # a traceback. (Everything else propagates loudly on purpose: tracebacks ARE the bug
        # reports this beta asks for.)
        print(f"[signet-local] REFUSED -- {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
