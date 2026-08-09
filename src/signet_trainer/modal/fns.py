"""Function-per-stage Modal boundary (D-08) + the Phase-1 CPU Volume round-trip probe (SC#3).

Declares ``@app.function`` stubs for the three pipeline stages — ``preprocess`` / ``train`` /
``sample`` — each carrying its own ``gpu=`` / ``timeout=`` and Volume mounts + secrets. These
are DECLARED in Phase 1, not run: Phases 2-4 fill the bodies (they currently
``raise NotImplementedError``). This is the seam the Phase-8 harness drives.

The one runnable Phase-1 deliverable is ``volume_roundtrip_probe`` — a CPU function (no ``gpu=``,
zero GPU spend) that writes a sentinel under the weights-Volume mount, calls ``weights_vol.commit()``
(REQUIRED — Pitfall 3 commit-or-vanish), and returns the path so a separate ``modal volume ls``
confirms durability + that weights mount from the Volume, not the image (MODL-01).

CRITICAL (D-10 / Pitfall 5 / T-01-MD3): NO function here sets ``keep_warm`` or ``min_containers``.
``import modal`` lives only in this package (Anti-Pattern 6).
"""

from __future__ import annotations

from typing import Any

import modal

from signet_trainer.modal.app import (
    CHECKPOINTS_DIR,
    CHECKPOINTS_MOUNT,
    DATASET_DIR,
    DATASET_MOUNT,
    WEIGHTS_DIR,
    WEIGHTS_MOUNT,
    app,
    checkpoints_vol,
    dataset_vol,
    download_image,
    gpu_image,
    h3_gpu_image,
    huggingface_secret,
    qwen_gpu_image,
    wandb_secret,
    weights_vol,
)

# A single function-call caps at 24h on Modal (RESEARCH.md Q5 / CLAUDE.md "Timeouts").
TWENTY_FOUR_HOURS = 24 * 60 * 60

# IC-LoRA 3-source dataset map (07-09 / Pitfall 3): the canonical precomputed source dir-name ->
# PrecomputedDataset output-key mapping the ICLoraStrategy consumes. Keyed to the dir names the
# strategy declares via ``get_data_sources()`` (["latents", "conditions", "reference_latents"]); the
# output keys are exactly the batch keys ``ICLoraStrategy.prepare_training_inputs`` reads
# (latent_conditions / text_conditions / ref_latent_conditions). Mirrors the map documented in
# data/precomputed.py (the third ``reference_latents`` source pairs by rel path with NO code change).
_PRECOMPUTED_SOURCE_OUTPUT_KEYS: dict[str, str] = {
    "latents": "latent_conditions",
    "conditions": "text_conditions",
    "reference_latents": "ref_latent_conditions",
    # Inpaint (Phase 9, GATE-SPEC rev 2): the signet-native mask-encode source (data/mask_encode.py).
    # The dir name "video_masks" deliberately contains NO "latent" substring, so PrecomputedDataset's
    # ``"latent" in dir_name`` branch never fires — the bare float32 [F_lat, H_lat, W_lat] {0,1}
    # mask tensors pass through UN-normalized (they are not VAE latents; tests/test_mask_encode.py
    # carries the regression proof). The output key mirrors the reference_latents ->
    # ref_latent_conditions pattern and is the primary key conditioning/inpaint.py::_MASK_BATCH_KEYS
    # reads.
    "video_masks": "video_mask_conditions",
    # Audio-to-video (Phase 9, GATE-SPEC rev 2 item 4): the frozen driving-audio latents the upstream
    # ``process_dataset`` emits under ``audio_latents/`` when ``with_audio=True``. Registered as an
    # extra multi-source exactly like ``reference_latents`` / ``video_masks`` (paired by rel path).
    # ``audio_latents`` CONTAINS the "latent" substring but is deliberately NOT in
    # PrecomputedDataset's ``_VIDEO_LATENT_SOURCE_DIRS`` allowlist, so its AUDIO-VAE latents are NOT
    # normalized through the video ``_normalize_video_latents`` einops path (the a2v loader trap the
    # allowlist fixes). The output key mirrors the reference_latents -> ref_latent_conditions pattern
    # and is the key ``conditioning/a2v.py::_AUDIO_BATCH_KEYS`` reads.
    "audio_latents": "audio_latent_conditions",
    # ── MiniMax-H3 Ref2VA (Phase 10, H3-03) — the four ``h3_``-prefixed sources ``h3_preprocess``
    # writes (dir names committed as convention by plan 10-04, produced by prep/h3_encode.py).
    # The ``h3_`` prefix is the whole point: the five LTX entries above stay byte-identical, and an
    # H3 cache can never be paired into an LTX run (nor the reverse) by a dir-name collision.
    # Exactly two of them — h3_latents + h3_reference_latents — are in PrecomputedDataset's
    # ``_VIDEO_LATENT_SOURCE_DIRS`` allowlist, because only those two carry the
    # ``{"latents": [C=24, F, H, W], "num_frames", "height", "width"}`` video-VAE payload. The other
    # two deliberately are not: ``h3_conditions`` holds Qwen3-VL TEXT embeddings with no "latents"
    # key, and ``h3_audio_latents`` is the substring trap again (it CONTAINS "latent" but carries
    # AUDIO-VAE latents). The output keys mirror the reference_latents -> ref_latent_conditions
    # pattern, ``h3_``-prefixed for the same non-collision reason.
    "h3_latents": "h3_latent_conditions",
    "h3_conditions": "h3_text_conditions",
    "h3_reference_latents": "h3_ref_latent_conditions",
    "h3_audio_latents": "h3_audio_latent_conditions",
    # ── Qwen-Image-Edit-2511 (Phase 11, family #3) — the three ``qwen_edit_``-prefixed sources
    # ``qwen_edit_preprocess`` writes. Dir names are IMPORTED-BY-CONVENTION from the two modules that
    # already declare them and agree at import time (``conditioning/qwen_edit.QWEN_EDIT_DATA_SOURCES``
    # for the reader, ``prep/qwen_edit_encode.QWEN_EDIT_*_DIR`` for the writer); the output keys are
    # exactly ``conditioning/qwen_edit.QWEN_EDIT_{TARGET,TEXT,CONTROL}_BATCH_KEYS[0]``, which is what
    # ``QwenEditStrategy`` looks up FIRST. They are restated here as literals for the same reason the
    # H3 four are: this dict is read by ``tests/test_h3_preprocess_wiring.py`` via AST WITHOUT
    # importing the module, so a computed comprehension would make it unreadable to the guard.
    #
    # ⚠ NONE of the three is in ``data/precomputed.py``'s ``_VIDEO_LATENT_SOURCE_DIRS`` allowlist, and
    # that is not an oversight — ``prep/qwen_edit_encode.qwen_edit_allowlist_gap()`` is the named
    # symbol that documents it. Two of them (``qwen_edit_latents`` / ``qwen_edit_control_latents``)
    # DO carry the ``{"latents": [C, F=1, H, W], num_frames, height, width}`` payload the allowlist
    # is for, so allowlisting them would be correct; it is also currently INERT, because the only
    # thing ``_normalize_video_latents`` does is convert LEGACY patchified ``[seq_len, C]`` caches
    # and this writer never produces one. ``qwen_edit_conditions`` carries Qwen2.5-VL text embeddings
    # with no ``"latents"`` key and must NEVER be allowlisted — the ``h3_conditions`` trap exactly.
    "qwen_edit_latents": "qwen_edit_latent_conditions",
    "qwen_edit_conditions": "qwen_edit_text_conditions",
    "qwen_edit_control_latents": "qwen_edit_control_latent_conditions",
}

# Tier-2 baseline (D-7-LADDER / D-7-BASELINE): the OFFICIAL Lightricks IC-LoRA run through the ported
# distilled two-stage ICLoraPipeline to validate the ported inference SURFACE against a KNOWN-good
# adapter BEFORE we train ours. Union-Control is the closest structural-control match (Canny+Depth+Pose)
# and doubles as a "does the reference steer layout?" yardstick (RESEARCH lines 404-416). It is a
# comfy-format single-file adapter (NOT a PEFT dir) and is ``ref0.5`` (reference_downscale_factor=2 read
# from its metadata) — Pitfall 4. The repo is a CONSTANT here (not a schema field: the baseline config
# names it in a comment; the schema stays untouched this plan). [VERIFIED: HF api author=Lightricks.]
OFFICIAL_IC_LORA_BASELINE_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"
# The repo ships a SINGLE comfy-format safetensors whose name carries the ``-ref0.5`` downscale suffix
# (07-11 live re-validation of the 07-10 MEDIUM-confidence wiring — HF ``list_repo_files`` shows exactly
# ``ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors``; the un-suffixed name 404s at fetch). This is
# Open Question 1's first concrete resolution at the gated GPU exercise (07-11 Rule-1 fix).
OFFICIAL_IC_LORA_BASELINE_FILENAME = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

# --------------------------------------------------------------------------------------------------
# Stage functions (D-08) — DECLARED here in Phase 1, bodies filled in Phases 2-4.
# Each gets its own gpu/timeout + the Volume mounts + secrets it needs. NONE sets keep_warm /
# min_containers (D-10) — warm GPUs stay opt-in and we never opt in.
# --------------------------------------------------------------------------------------------------


@app.function(
    # GPU step (Phase 2): the canonical VAE+Gemma encode needs CUDA (RESEARCH.md A2/A4 — was a CPU
    # stub). It precomputes latents -> caches to the dataset Volume so the expensive train step never
    # re-encodes (RESEARCH.md "RUN GATE" split). Uses the pinned-SHA gpu_image (ltx-core/ltx-trainer).
    gpu="A100-80GB",
    image=gpu_image,
    volumes={**WEIGHTS_MOUNT, **DATASET_MOUNT},
    secrets=[huggingface_secret],
    timeout=TWENTY_FOUR_HOURS,
)
def preprocess(
    metadata_path: str = str(DATASET_DIR / "fresh" / "metadata.jsonl"),
    resolution_buckets: list = [(25, 352, 768), (49, 352, 768), (81, 352, 768)],  # noqa: B006
    output_dir: str = str(DATASET_DIR / ".precomputed"),
    reference_column: str | None = None,
    reference_downscale_factor: int = 1,
    mask_column: str | None = None,
    mask_output_dir_name: str = "video_masks",
    with_audio: bool = False,
) -> str:
    """Stage 1 — run the CANONICAL ``process_dataset.py`` encode, write the precomputed dir to the Volume.

    Runs ltx-trainer's verified ``preprocess_dataset`` (NOT a custom encoder — the enochiatron
    landmine, RESEARCH.md Pitfall 1) video-only, multi-bucket, ``batch_size=1``, ``device="cuda"``,
    then ``dataset_vol.commit()`` (Pitfall 3 commit-or-vanish). Writes ``{conditions, latents}/``
    under ``output_dir`` on the dataset Volume for the loop-pure ``PrecomputedDataset`` to read.

    SC#2 / MODL-04: pre-encoding goes THROUGH the canonical path, proving the literal run-through
    requirement.

    Parameterized (Plan 06-08 / REF-02) so a demo subset can be encoded to a distinct
    ``.precomputed_demo`` output dir WITHOUT changing the Phase-2 default behavior:

        * ``metadata_path``      — the metadata.jsonl to encode (default: the Phase-2 ``fresh`` set).
        * ``resolution_buckets`` — the (F, H, W) bucket tuples (default: 768x352x{25,49,81}, D-05).
        * ``output_dir``         — where the precomputed dir is written (default: ``.precomputed``).

    The defaults reproduce the exact Phase-2 ``fresh`` encode (backward-compat); passing demo values
    (metadata.jsonl for the demo subset, the single ``[832, 480, 49]`` bucket, ``.precomputed_demo``)
    drives the gated multi-frame encode (Plan 06-10). All other args are unchanged.

    IC-LoRA paired encode (Plan 07-09 / REF-03): the two NEW args thread the canonical
    ``video_to_video`` reference-latent pre-encode through the SAME ``preprocess_dataset`` (NOT a
    custom encoder — RESEARCH Pitfall 1):

        * ``reference_column``          — metadata.jsonl column naming each sample's paired reference
          (seg-map) clip (``"reference_path"`` for an ic_lora run; ``None`` = no reference encode,
          the byte-identical fresh/demo default — backward-compat).
        * ``reference_downscale_factor`` — the reference clip is encoded at ``1/n`` the target
          resolution (D-7-REF11 keeps ``1`` = 1:1 for the first proof).

    An ic_lora run passes ``reference_column="reference_path"`` + the ``.precomputed_demo_seg``
    output_dir + the single 25f/832x480 bucket, so the paired ``reference_latents/`` are encoded
    alongside the target ``latents/`` for the 3-source ``PrecomputedDataset`` (train step below).

    Inpaint mask encode (Phase 9, GATE-SPEC rev 2): upstream ``process_dataset`` at signet's pinned
    SHA (d6053703) PREDATES the flexible/mask epoch and CANNOT emit ``video_masks/`` — so when
    ``conditioning.mode == "inpaint"`` the entrypoint threads the two mask args and this fn runs the
    SIGNET-NATIVE mask encode (``data/mask_encode.py``, pure CPU, no VAE) AFTER the canonical
    latent/condition encode (it reads each sample's latent-grid metadata from the just-written
    ``latents/``):

        * ``mask_column``          — manifest column naming each sample's polarity-rendered mask
          source (``"video_mask"``, the upstream role-convention name; ``None`` = no mask encode,
          the byte-identical default for every non-inpaint run).
        * ``mask_output_dir_name`` — the output dir under ``output_dir`` (config-driven from
          ``cfg.conditioning.inpaint_mask_dir``, default ``"video_masks"`` — D-NOHARDCODE; the dir
          name must contain no "latent" substring so the loader never normalizes the masks).

    The encoded masks are bare float32 ``[F_lat, H_lat, W_lat]`` {0,1} tensors mirroring the
    ``latents/`` rel-path layout, riding the SAME ``dataset_vol.commit()`` as the canonical encode.

    Audio-to-video audio encode (Phase 9, GATE-SPEC rev 2 item 1):

        * ``with_audio`` — when True the canonical ``process_dataset`` ALSO extracts + encodes each
          clip's audio to ``audio_latents/`` (the upstream ``with_audio`` flag, which EXISTS at
          signet's pinned SHA d6053703 — verified via gh; it predates the mask epoch). Config-driven
          from ``cfg.audio.with_audio`` (D-NOHARDCODE). Default False keeps every non-a2v encode
          byte-identical. The a2v ``A2VStrategy`` reads that ``audio_latents/`` source; the entrypoint
          fail-fasts an a2v config that leaves ``with_audio`` False (no silent audio-skip).
    """
    import sys

    # Pitfall 2 sibling-import shim: process_dataset.py imports its siblings (process_videos, etc.)
    # by BARE name, so its scripts/ dir must be on sys.path before the import.
    sys.path.insert(0, "/opt/LTX-2/packages/ltx-trainer/scripts")
    from process_dataset import preprocess_dataset

    preprocess_dataset(
        dataset_file=metadata_path,
        # caption/video columns: REQUIRED positional args (no defaults in the fn — only in the typer
        # CLI). Must match write_manifest's {caption, media_path} schema (data/dataset_file.py).
        caption_column="caption",
        video_column="media_path",
        # D-05: (F, H, W) bucket tuples (the internal form; CLI strings are WxHxF). Sourced from
        # SignetConfig.data.resolution_buckets; default is the full 768x352x{25,49,81} multi-bucket set.
        resolution_buckets=resolution_buckets,
        batch_size=1,  # Pitfall 4: multi-bucket REQUIRES batch_size=1 (per-sample shapes differ).
        output_dir=output_dir,
        lora_trigger=None,  # REQUIRED arg; no trigger word for the SC#2 encode-path proof.
        vae_tiling=True,  # REQUIRED arg; tiled VAE = low-VRAM safe (CLAUDE.md LTX decode-OOM note).
        decode=False,  # REQUIRED arg; pre-encode only — do NOT decode latents back to video.
        model_path=str(WEIGHTS_DIR / "ltx-2.3-22b-dev.safetensors"),
        text_encoder_path=str(WEIGHTS_DIR / "gemma-3-12b-it"),
        device="cuda",
        # Audio-to-video (Phase 9, GATE-SPEC rev 2 item 1): config-driven (default False). When True
        # the canonical process_dataset extracts + encodes audio to audio_latents/ (the with_audio
        # flag EXISTS at signet's pinned SHA d6053703 — verified via gh; it predates the mask epoch).
        # False keeps every non-a2v encode byte-identical (audio out of scope).
        with_audio=with_audio,
        overwrite=False,
        # IC-LoRA paired reference-latent encode (07-09 / REF-03) — threaded into the CANONICAL
        # encoder, the only change (RESEARCH line 78). None => no reference encode (fresh/demo
        # default, backward-compat); "reference_path" => encode the paired seg-map clips to
        # reference_latents/ for the 3-source PrecomputedDataset.
        reference_column=reference_column,
        reference_downscale_factor=reference_downscale_factor,
    )

    # a2v LOUD-FAILURE guard (2026-07-15 burned gate): the pin's ``_extract_audio`` swallows EVERY
    # exception (torchaudio backend missing, bad container, ...) into a DEBUG-level skip and the
    # encode "succeeds" with an empty audio_latents/ — the training run would then fail (or worse,
    # silently train without audio). When audio was requested, an empty audio_latents/ is a HARD
    # error at the encode, before any further metered step.
    if with_audio:
        from pathlib import Path  # noqa: PLC0415

        audio_out = Path(output_dir) / "audio_latents"
        n_audio = len(list(audio_out.rglob("*.pt"))) if audio_out.exists() else 0
        if n_audio == 0:
            raise RuntimeError(
                "with_audio=True but the canonical encode produced ZERO audio latents "
                f"under {audio_out} — the pin's _extract_audio swallows backend errors "
                "(torchaudio/ffmpeg) into silent skips. Check the image's ffmpeg libs. "
                "Refusing to report a successful a2v encode without audio."
            )

    # Inpaint (Phase 9, GATE-SPEC rev 2): the SIGNET-NATIVE mask encode — upstream at the pin cannot
    # emit video_masks/, so the latent-grid mask .pt files are produced here, AFTER the canonical
    # encode (the per-sample latent grid is read from the just-written latents/ metadata). Pure CPU
    # (no VAE) but co-located so the masks ride the same Volume commit as the latents they mirror.
    # mask_column=None (the default) keeps every non-inpaint run byte-identical.
    if mask_column is not None:
        from pathlib import Path  # noqa: PLC0415

        from signet_trainer.data.mask_encode import encode_mask_dataset  # noqa: PLC0415

        n_masks = encode_mask_dataset(
            dataset_file=metadata_path,
            mask_column=mask_column,
            latents_dir=str(Path(output_dir) / "latents"),
            output_dir=str(Path(output_dir) / mask_output_dir_name),
            # The main-media column names each sample's outputs so masks mirror the latents/ layout
            # (must match the canonical encoder's video_column above — write_manifest's schema).
            media_column="media_path",
        )
        print(
            f"[preprocess] signet-native mask encode wrote {n_masks} latent-grid mask(s) -> "
            f"{Path(output_dir) / mask_output_dir_name} (bare float32 [F_lat,H_lat,W_lat] {{0,1}}, "
            "mirroring latents/ rel paths; upstream at the pin cannot emit video_masks/ — "
            "GATE-SPEC rev 2)."
        )

    # Pitfall 3 commit-or-vanish: without commit() the encoded output_dir is lost on container
    # exit and `modal volume ls signe-trainer-dataset` would show nothing.
    dataset_vol.commit()

    return output_dir


@app.function(
    gpu="A100-80GB",  # 22b LoRA fits A100-80GB (enochiatron precedent); H100 optional speedup.
    image=gpu_image,  # the heavy image (torch/ltx-core/peft/bnb) — the code-only default has no torch.
    volumes={**WEIGHTS_MOUNT, **DATASET_MOUNT, **CHECKPOINTS_MOUNT},
    secrets=[huggingface_secret, wandb_secret],
    timeout=TWENTY_FOUR_HOURS,  # the long-running training call; do NOT use a short default.
    # F9 fix (operator-approved 2026-07-12): Modal's preemption auto-restart was observed 0/2
    # (r1_mf @2400, r2 @1000) — server-side retries make the long round self-heal without a
    # local re-dispatch. Safe because train resumes in-dir from the latest committed checkpoint
    # (CheckpointManager.resume, landmine-#1 path, proven twice this campaign); commit-per-save
    # means a retry can never see a half-written checkpoint.
    #
    # 2026-08-06 — THE PREEMPTION CONTRACT, mirrored from ``h3_train`` (see the full derivation at
    # that decorator). LTX ``train`` RIDES ALONG deliberately: it is the identical defect class —
    # a long RESUMABLE round that calls ``CheckpointManager.resume`` in-dir and commits per save,
    # so a retry can never observe a half-written checkpoint. Same safety argument, zero metered
    # cost to apply. Leaving it at ``max_retries=3`` while fixing only the H3 twin would repeat
    # verbatim the prose-not-structure failure this change exists to close.
    #   ``max_retries=10`` — MODAL'S SERVER-ENFORCED CEILING (see the h3_train derivation), which
    #   the CLIENT does not validate. 11 container lives vs the 44 the observed cadence needs.
    #   ``single_use_containers=True`` — a FRESH container per retry (Modal's canonical
    #   long-training shape, and this repo's own CLAUDE.md Modal-patterns table). Previously set
    #   NOWHERE in src/ despite being documented in both places.
    # ⛔ LTX ``sample`` (below) is DELIBERATELY EXCLUDED from both: a render is NOT resumable.
    retries=modal.Retries(max_retries=10, initial_delay=60.0, backoff_coefficient=2.0),
    single_use_containers=True,
)
def train(config_yaml: str) -> None:
    """Stage 2 — the gated LTX-2.3 LoRA training run (never auto-launched; gated by entrypoint).

    Drives the full ready-to-train sequence, ALL heavy imports function-local (Anti-Pattern 6,
    mirroring ``preprocess`` / ``models/loader.py``):

        (1) COLD-PATH IMPORT PROBE — confirm peft / bitsandbytes / wandb resolve in gpu_image
            BEFORE any model load (the Phase-2 cold-path lesson: an ImportError on the A100 wastes
            a metered launch; T-03-63). Fails loudly with the exact fix.
        (2) load + revalidate the YAML config in-container (the entrypoint passes the YAML TEXT by
            value — the ``configs/`` dir is not shipped into the image, so a path would not resolve).
        (3) load the LTX-2.3 components (transformer / VAE / Gemma / scheduler) from the Volume.
        (4) run the 6-check architecture preflight gate (``run_validation_gate``) and ABORT on any
            FAIL — the cheap fail-fast that caught 6 arch mismatches for ~$1.40 (SC#3 / T-03-63).
        (4b) resolve Open-Q1 from check #4: use ``components.transformer`` directly (PASS default)
            or PEFT-wrap a ``SingleGPUModelBuilder(...).build()`` if the loader's transformer does
            not accept the Modality forward signature (train.py:1806-1811).
        (5) inject LoRA (GC-before-LoRA inside ``inject_lora``) over the ``ff.net`` target set.
        (6) construct the offloader baseline-first (``blocks_to_swap=0`` → INERT; OFFL-03 no-swap
            baseline, OFFL-01 harvested) + define the ``remove_hooks`` suspend seam (OFFL-02)
            wrapping the DEFERRED Phase-4 sampler call-site (a documented no-op here).
        (7) ``PrecomputedDataset`` over the cached latents; (8) optimizer + scheduler + checkpoint
            manager; (9) the step-driven ``train_loop`` whose "done" marker is the committed
            ``checkpoint-step-{max_steps}`` on the Volume (commit-or-vanish), NOT a log line.

    NO ``keep_warm`` / ``min_containers`` (D-10) — warm GPUs stay opt-in and we never opt in.
    """
    import contextlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    # ── (1) COLD-PATH IMPORT PROBE — fail BEFORE any model load / sustained spend (T-03-63) ─────
    # No installs here (T-03-SC supply-chain): we VERIFY presence; a missing dep means the
    # pinned-SHA gpu_image must add it (re-gated by the Phase-2 supply-chain discipline).
    try:
        import bitsandbytes as bnb  # noqa: PLC0415
        import peft  # noqa: PLC0415
        import wandb  # noqa: PLC0415
    except ImportError as exc:  # the cold-path bug class — name the missing dep + the fix.
        raise RuntimeError(
            f"[train] cold-path dependency missing ({exc.name!r}). The gpu_image must carry "
            "peft / bitsandbytes / wandb before any sustained GPU spend. Fix: add "
            "`uv pip install 'peft>=0.14' bitsandbytes wandb` to the pinned-SHA gpu_image and "
            "rebuild (re-gated by the Phase-2 supply-chain discipline, T-03-SC)."
        ) from exc
    print(
        f"[train] cold-path imports OK — peft={peft.__version__} "
        f"bitsandbytes={getattr(bnb, '__version__', '?')} wandb={wandb.__version__}"
    )

    from signet_trainer.data.precomputed import PrecomputedDataset  # noqa: PLC0415
    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415
    from signet_trainer.lora.peft import (  # noqa: PLC0415
        P1_FF_LORA_TARGETS,
        build_lora_config,
        inject_lora,
    )
    from signet_trainer.models.loader import load_ltxv_components  # noqa: PLC0415
    from signet_trainer.offload.block_swap import BlockSwapOffloader  # noqa: PLC0415
    from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415
    from signet_trainer.train.flow_match import FlowMatchingSchedule  # noqa: PLC0415
    from signet_trainer.train.loop import (  # noqa: PLC0415
        build_optimizer,
        build_scheduler,
        in_loop_decoder_enabled,
        in_loop_sample_due,
        should_warm_start,
        train_loop,
    )
    from signet_trainer.train.validate_gate import run_validation_gate  # noqa: PLC0415

    # ── (2) load + revalidate the config in-container (entrypoint passes only the PATH) ─────────
    config = load_config_from_text(config_yaml)

    # Fail-fast (WR-04): conditioning_items are SAMPLE-grid-only in Phase 6 — training is
    # self-conditioning (D-6-CONDSOURCE 'self': MultiFrameStrategy samples its own keyframe
    # positions/strengths from the clip). A training config carrying items (e.g. the multi-frame
    # SAMPLE example passed to train by mistake) would have them silently ignored — the exact
    # silently-ignored-config-block class the schema's field-split doctrine forbids. Raise BEFORE
    # any model load / GPU spend.
    if config.conditioning.mode == "multi_frame" and config.conditioning.conditioning_items:
        raise RuntimeError(
            "[train] conditioning_items are sample-only in Phase 6 (conditioning_source='self': "
            "MultiFrameStrategy samples keyframes from the clip itself); training would silently "
            "ignore them. Remove conditioning_items from the training config (see "
            "configs/ltx23_multi_frame_overfit.example.yaml) — keyframe items belong in the "
            "sample config only."
        )

    device = "cuda"
    checkpoint_path = str(WEIGHTS_DIR / config.model.model_id)
    text_encoder_path = str(WEIGHTS_DIR / config.model.text_encoder_id)

    # ── (3) load the LTX-2.3 components from the mounted weights Volume (MODL-01) ───────────────
    # OFFL-02 (D-9-OFFL02-CLOSE): drive the video VAE decoder from the config knob (config-first,
    # D-NOHARDCODE). ``in_loop_decoder_enabled`` is True iff ``validation.in_loop_sampling`` is set
    # AND prompts are non-empty; only then is the decoder loaded so the in-loop validation-sample
    # seam body is reachable. The byte-identical default (knob off) keeps the decoder OFF — every
    # existing training call site is unchanged and the decoder-on VRAM is strictly opt-in.
    _in_loop = in_loop_decoder_enabled(config.validation)
    components = load_ltxv_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=device,
        with_video_vae_decoder=_in_loop,
    )

    # ── (4) 6-check architecture preflight gate — ABORT on any FAIL, BEFORE training (SC#3) ─────
    passed, results, gate_adapter = run_validation_gate(
        components, config, checkpoint_path=checkpoint_path, device=device
    )
    for r in results:
        gate = " [HARD GATE]" if r.hard_gate else ""
        print(f"[train][gate] {r.name}: {r.status} — {r.message} ({r.duration_s}s){gate}")
    if not passed:
        failed = [r.name for r in results if r.status != "PASS"]
        raise RuntimeError(
            f"[train] architecture validation gate FAILED ({failed}) — aborting BEFORE any "
            "training spend. enochiatron's gate caught 6 arch mismatches this way for ~$1.40; fix "
            "the reported mismatch and re-run (SC#3 / T-03-63)."
        )

    # ── (4a-PRE) OFFL-02 PHASE A — pre-encode the in-loop validation prompts BEFORE Gemma is freed ─
    # When in-loop sampling is on (decoder loaded above), the in-loop sampler must NOT re-encode via
    # Gemma — Gemma is freed below (4a) so the 22B transformer never coexists with it (~72GB OOM,
    # 06-09 runs 3-5). So encode every ``config.validation.prompts`` entry ONCE here (Gemma is still
    # loaded from the gate) into detached-CPU ``CachedPromptEmbeddings``, exactly as the ``sample`` fn
    # does (:622-692): ``load_embeddings_processor`` on CPU then ``.to(device)``, ``torch.no_grad()``
    # MANDATORY (the ltx GemmaTextEncoder.encode runs a full 12B forward with autograd ON — one
    # retained graph is ~20-25GB), ``.detach().to("cpu")`` each cached tensor, ``del`` intermediates
    # between prompts so no two encodes' hidden-state tuples coexist. The in-loop seam body (6) reads
    # ``cached_by_prompt[prompt]`` via ``run_sampler(..., cached_embeddings=...)`` and never touches
    # the freed Gemma. Inert when the knob is off (``cached_by_prompt`` stays None, decoder off).
    cached_by_prompt: dict | None = None
    if _in_loop:
        import gc  # noqa: PLC0415

        from ltx_trainer.model_loader import load_embeddings_processor  # noqa: PLC0415
        from ltx_trainer.validation_sampler import CachedPromptEmbeddings  # noqa: PLC0415

        # EP MUST load on CPU then move only the built connector to the GPU (sample fn run-6 OOM:
        # device="cuda" drags ~50GB of the 44GB checkpoint onto the GPU alongside Gemma + the 22B).
        emb_processor = load_embeddings_processor(checkpoint_path=checkpoint_path, device="cpu")
        emb_processor.to(device)
        cached_by_prompt = {}
        with torch.no_grad():
            neg_hs, neg_mask = components.text_encoder.encode("")
            neg_out = emb_processor.process_hidden_states(neg_hs, neg_mask)
            neg_video = neg_out.video_encoding.detach().to("cpu")
            neg_audio = neg_out.audio_encoding.detach().to("cpu")
            del neg_hs, neg_mask, neg_out  # drop the hidden-state tuple before the next forward
            for prompt in config.validation.prompts:
                pos_hs, pos_mask = components.text_encoder.encode(prompt)
                pos_out = emb_processor.process_hidden_states(pos_hs, pos_mask)
                cached_by_prompt[prompt] = CachedPromptEmbeddings(
                    video_context_positive=pos_out.video_encoding.detach().to("cpu"),
                    audio_context_positive=pos_out.audio_encoding.detach().to("cpu"),
                    video_context_negative=neg_video,
                    audio_context_negative=neg_audio,
                )
                del pos_hs, pos_mask, pos_out  # never two encodes' intermediates at once
        del emb_processor
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"[train] OFFL-02 PHASE A pre-encoded {len(cached_by_prompt)} validation prompt(s) -> "
            "cached embeddings (two-phase VRAM, before the Gemma free); the in-loop sampler decodes "
            f"with these + no_grad + decoder and never re-encodes via Gemma; cuda "
            f"allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB."
        )

    # ── (4a) FREE Gemma before the training loop (06-09 carry-forward two-phase VRAM discipline) ──
    # The text encoder (Gemma-12B, ~24GB) is needed ONLY by the gate's check #2 (a
    # ``config.hidden_size`` read — NO Gemma forward, so no autograd-graph OOM at gate time). The
    # training loop reads PRECOMPUTED text embeddings from the config's precomputed conditions and
    # NEVER calls Gemma, and the in-loop validation-sampler seam below is inert on this training load
    # (decoder off). So drop Gemma NOW — before the 22B transformer runs real 832x480x49 forwards +
    # backward + optimizer state — so it never coexists with the transformer during heavy compute
    # (the proven pattern: Gemma NEVER coexists with the 22B on the A100-80GB; runs 3-5 proved they
    # cannot). ``assign=True`` loader-owned CUDA storage is freed only by dropping the LAST reference
    # (not Module.to("cpu")), so null the attribute AND del the local before gc (06-09 run-5 finding).
    import gc  # noqa: PLC0415

    _text_encoder = getattr(components, "text_encoder", None)
    if _text_encoder is not None:
        components.text_encoder = None
        del _text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        print(
            "[train] freed Gemma text encoder after the validation gate (two-phase VRAM "
            "discipline, 06-09 carry-forward) — training uses precomputed conditions, never Gemma; "
            f"cuda allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB."
        )

    # ── (4b) Open-Q1 resolution + obtain the training model WITHOUT double-injecting (03-07 fix) ──
    # The gate's check #5/#6 already PEFT-injected + roundtrip-proved an adapter on
    # ``components.transformer`` (returned above as ``gate_adapter``). Two paths:
    #   • Open-Q1 default (check #4 PASS → the loader transformer IS the forward target): REUSE that
    #     exact adapter. Re-injecting ``components.transformer`` would PEFT-wrap an already-LoRA'd
    #     module on the real GPU → error/corruption. This trains the adapter the #6 roundtrip proved.
    #   • use_builder (check #4 FAIL): the trained transformer is a DIFFERENT freshly-built object, so
    #     a clean ``inject_lora`` there is correct (the gate's mutation of the loader transformer is
    #     unused on this branch).
    forward_check = next((r for r in results if r.name == "check_forward_pass"), None)
    use_builder = bool(
        forward_check is not None and forward_check.details.get("open_q1") == "use_builder"
    )
    if use_builder:
        # The loader's transformer does NOT accept model(video=,audio=,perturbations=)→(pred,_);
        # build the forward-compatible model the source trains (enochiatron train.py:1806-1811).
        from ltx_trainer.model_builder import (  # noqa: PLC0415 — Modal-side heavy import
            LTXModelConfigurator,
            SingleGPUModelBuilder,
        )
        from ltx_trainer.model_loader import LTXV_MODEL_COMFY_RENAMING_MAP  # noqa: PLC0415

        base_transformer = SingleGPUModelBuilder(
            model_path=checkpoint_path,
            configurator=LTXModelConfigurator(),
            renaming_map=LTXV_MODEL_COMFY_RENAMING_MAP,
        ).build()
        # (5) inject LoRA (GC-before-LoRA inside inject_lora, TRAIN-06) over the ff.net target set.
        lora_config = build_lora_config(
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=config.lora.dropout,
            targets=config.lora.target_modules or P1_FF_LORA_TARGETS,
        )
        model = inject_lora(base_transformer, lora_config)
    else:
        base_transformer = components.transformer  # Open-Q1 default (check #4 PASS).
        # (5) REUSE the gate's injected + roundtrip-proved adapter — no second injection.
        if gate_adapter is None:
            raise RuntimeError(
                "[train] validation gate passed but returned no injected adapter — expected the "
                "check #5 PEFT model to reuse on the Open-Q1 default path (03-07 double-inject fix)."
            )
        model = gate_adapter
        model.zero_grad(set_to_none=True)  # clear the gate's check #5 backward grads before training

    # Resolve the transformer's block ModuleList for the offloader (landmine #3 container naming).
    block_list = None
    for attr in ("transformer_blocks", "blocks"):
        block_list = getattr(base_transformer, attr, None)
        if block_list is not None:
            break
    if block_list is None:
        raise RuntimeError(
            "[train] could not locate the transformer block ModuleList "
            "(tried .transformer_blocks / .blocks) — required for the OFFL offloader."
        )

    # ── (6) offloader baseline-first (blocks_to_swap=0 → INERT) + the OFFL-02 suspend seam ──────
    # blocks_to_swap=0 makes the harvested UNVALIDATED offloader inert (OFFL-03 no-swap baseline;
    # OFFL-01 harvested). Trusting block-swap is gated behind OFFL-03's A/B — NOT this plan.
    offloader = BlockSwapOffloader(
        block_list,
        blocks_to_swap=config.offload.blocks_to_swap,
        device=torch.device(device),
    )

    @contextlib.contextmanager
    def offloader_suspended(off, blocks):  # OFFL-02 suspend seam (structural).
        """Drop the block-swap hooks around an out-of-band forward, then re-arm.

        Wraps the DEFERRED Phase-4 validation-sampler forward so async block-swap DMA cannot race
        the sampler. Inert at ``blocks_to_swap=0`` (``remove_hooks`` is a safe no-op; nothing to
        re-register), so this is a structural seam in Phase 3 — sampling lands in Phase 4.
        """
        off.remove_hooks()
        try:
            yield
        finally:
            if getattr(off, "blocks_to_swap", 0) > 0:
                off._register_hooks(blocks)  # re-arm (no-op on the inert baseline).

    # In-loop validation-sampler (OFFL-02, D-9-OFFL02-CLOSE) — fires INSIDE train_loop at every
    # checkpoint boundary via the ``on_checkpoint`` callback, with the REAL step number. (The
    # r1 finding, 2026-07-11: the previous pre-loop "structural stand-in" rendered an
    # UNTRAINED-adapter mp4 named with max_steps — mid-run cadence never fired. This callback is
    # the fix; r2 carries the live mid-run proof.) Each render runs inside offloader_suspended so
    # async block-swap DMA cannot race it (inert at blocks_to_swap=0), decodes with the PHASE-A
    # cached embeddings + no_grad + decoder, and NEVER re-encodes via the freed Gemma (~72GB OOM
    # avoided). Knob off => decoder is None => ``in_loop_sample_due`` False => callback is a no-op
    # (byte-identical default). The mp4 rides the SAME Volume commit as its checkpoint
    # (train_loop invokes the callback between ckpt save and vol.commit — commit-or-vanish).
    def _in_loop_sample(step: int) -> None:
        _decoder_ready = getattr(components, "video_vae_decoder", None) is not None
        if not in_loop_sample_due(
            step,
            config.training.checkpoint_every,
            _decoder_ready,
            bool(config.validation.prompts),
        ):
            return
        from ltx_trainer.video_utils import save_video  # noqa: PLC0415

        from signet_trainer.inference.sampler import (  # noqa: PLC0415
            build_generation_config,
            run_sampler,
        )

        prompt = config.validation.prompts[0]
        vcfg = build_generation_config(config, prompt=prompt, seed=config.validation.seed)
        # Feed the PHASE-A cached embeddings (Gemma is freed): run_sampler short-circuits
        # _get_prompt_embeddings and decodes with the decoder under no_grad — no Gemma re-encode.
        with offloader_suspended(offloader, block_list):
            video, _audio = run_sampler(
                components,
                model,
                vcfg,
                device=device,
                cached_embeddings=cached_by_prompt[prompt],
            )
        save_video(
            video,
            str(CHECKPOINTS_DIR / config.output_dir / "samples" / f"step{step}.mp4"),
            fps=config.validation.frame_rate,
        )

    # ── (7) the loop-pure dataset over the precomputed latents (TRAIN-03 / D-08) ────────────────
    # Use the CONFIG's preprocessed_data_root (06-10 fix) — NOT a hardcoded ``.precomputed``. The
    # Phase-3 overfit config carries ``/dataset/.precomputed`` (byte-identical to the old hardcode),
    # while the demo multi-frame overfit (06-10) carries ``/dataset/.precomputed_demo``; hardcoding
    # would silently train the demo run on the wrong (fresh) latents or fail if absent.
    if config.conditioning.mode == "ic_lora":
        # IC-LoRA (07-09 / Pitfall 3): build the 3-source PrecomputedDataset so the paired
        # ``reference_latents/`` load ALONGSIDE the target latents + text conditions. Thread the
        # source list the strategy DECLARES (``get_data_sources()``) rather than hardcoding a second
        # time (single-source-of-truth with train/step.py's ICLoraStrategy) — deps/schedule are
        # unused by get_data_sources, so a bare instance (both None) suffices here on CPU-cheap
        # wiring (the per-step strategy that DOES use them is built inside train/step.py's loop).
        from signet_trainer.conditioning.ic_lora import ICLoraStrategy  # noqa: PLC0415
        from signet_trainer.dryrun.shapes import assert_paired_reference  # noqa: PLC0415

        _ic_strategy = ICLoraStrategy(
            deps=None,
            schedule=None,
            reference_downscale_factor=config.conditioning.reference_downscale_factor,
        )
        data_sources = {
            name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[name]
            for name in _ic_strategy.get_data_sources()
        }
        dataset = PrecomputedDataset(
            config.data.preprocessed_data_root, data_sources=data_sources
        )

        # PREFLIGHT (SC#2 / T-07-09-01): assert the paired (reference, target) latents are
        # dim-compatible BEFORE any sustained training spend — a mismatched bucket (Pitfall 2, the
        # doubled-sequence OOM) is rejected cheaply on a single CPU-cheap sample read rather than
        # crashing a metered run mid-loop. Reference is ALWAYS temporally frame-aligned 1:1 with the
        # target; H/W equal at downscale_factor==1 (D-7-REF11) or exact integer multiples otherwise.
        _probe = dataset[0]
        _tgt_latents = _probe["latent_conditions"]["latents"]
        _ref_latents = _probe["ref_latent_conditions"]["latents"]
        assert_paired_reference(
            _ref_latents.shape,
            _tgt_latents.shape,
            config.conditioning.reference_downscale_factor,
        )
        print(
            "[train][ic_lora] paired-reference preflight PASSED — reference "
            f"{tuple(_ref_latents.shape)} vs target {tuple(_tgt_latents.shape)} "
            f"(downscale_factor={config.conditioning.reference_downscale_factor}); 3-source dataset "
            f"({list(data_sources)}) built before any sustained spend."
        )
    elif config.conditioning.mode == "inpaint":
        # INPAINT (Phase 9, GATE-SPEC rev 2): build the 3-source PrecomputedDataset so the
        # signet-native ``video_masks/`` load ALONGSIDE the target latents + text conditions —
        # the same strategy-declared single-source-of-truth pattern as the ic_lora branch above
        # (InpaintStrategy.get_data_sources() == ["latents", "conditions", "video_masks"]).
        from signet_trainer.conditioning.inpaint import InpaintStrategy  # noqa: PLC0415

        _inp_strategy = InpaintStrategy(deps=None, schedule=None)
        data_sources = {
            name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[name]
            for name in _inp_strategy.get_data_sources()
        }
        dataset = PrecomputedDataset(
            config.data.preprocessed_data_root, data_sources=data_sources
        )

        # PREFLIGHT (mirrors the ic_lora paired-reference probe): the encoded mask's latent grid
        # must match the sample's OWN latent dims — a stale/mis-bucketed mask (wrong downsample,
        # re-encoded latents without re-encoded masks) is rejected on ONE CPU-cheap sample read
        # BEFORE any sustained training spend, never mid-loop on a metered A100. The per-step
        # strategy re-checks every sample (fail-loud), so this probe is the cheap early tripwire.
        _probe = dataset[0]
        _tgt_latents = _probe["latent_conditions"]["latents"]
        _mask_payload = _probe["video_mask_conditions"]
        _mask = _mask_payload["mask"] if isinstance(_mask_payload, dict) else _mask_payload
        _lat_grid = tuple(_tgt_latents.shape[-3:])  # [..., F_lat, H_lat, W_lat]
        if tuple(_mask.shape[-3:]) != _lat_grid:
            raise RuntimeError(
                f"[train][inpaint] mask/latent grid mismatch on sample 0: mask "
                f"{tuple(_mask.shape)} vs latent grid {_lat_grid} — the video_masks/ encode does "
                "not match these latents (stale masks after a re-encode, or a wrong bucket). "
                "Re-run the gated preprocess so masks + latents encode together."
            )
        print(
            "[train][inpaint] mask preflight PASSED — mask grid "
            f"{tuple(_mask.shape[-3:])} matches the latent grid {_lat_grid}; 3-source dataset "
            f"({list(data_sources)}) built before any sustained spend "
            f"(mask_probability={config.conditioning.inpaint_mask_probability})."
        )
    elif config.conditioning.mode == "audio_to_video":
        # AUDIO-TO-VIDEO (Phase 9, GATE-SPEC rev 2 item 4): build the 3-source PrecomputedDataset so
        # the frozen driving-audio latents load ALONGSIDE the target video latents + text conditions
        # — the same strategy-declared single-source-of-truth pattern as the ic_lora/inpaint branches
        # (A2VStrategy.get_data_sources() == ["latents", "conditions", "audio_latents"]). The
        # "audio_latents" source is NOT normalized as a video latent (PrecomputedDataset allowlist —
        # the a2v loader trap the GATE-SPEC item 4 fix closes).
        from signet_trainer.conditioning.a2v import A2VStrategy  # noqa: PLC0415

        _a2v_strategy = A2VStrategy(deps=None, schedule=None)
        data_sources = {
            name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[name]
            for name in _a2v_strategy.get_data_sources()
        }
        dataset = PrecomputedDataset(
            config.data.preprocessed_data_root, data_sources=data_sources
        )

        # PREFLIGHT (mirrors the ic_lora / inpaint probes): the audio-latent source must exist and
        # load on ONE CPU-cheap sample read BEFORE any sustained training spend — a missing/empty
        # audio_latents/ (an encode that ran with_audio=False) is rejected cheaply here rather than
        # crashing a metered A100 mid-loop.
        _probe = dataset[0]
        _audio_payload = _probe["audio_latent_conditions"]
        _audio_lat = (
            _audio_payload["latents"] if isinstance(_audio_payload, dict) else _audio_payload
        )
        print(
            "[train][a2v] audio-latent preflight PASSED — audio latent "
            f"{tuple(_audio_lat.shape)} paired with target latents; 3-source dataset "
            f"({list(data_sources)}) built before any sustained spend (frozen-audio conditioning: "
            "audio tokens at timestep 0, excluded from the loss)."
        )
    else:
        dataset = PrecomputedDataset(config.data.preprocessed_data_root)

    # ── (7b) IC-LoRA frozen-adapter stacking (D-7-FREEZE) — stack + freeze BEFORE the optimizer ──
    # If a frozen adapter is configured, load it alongside the trainable "default" adapter, activate
    # BOTH (their LoRA deltas sum in the forward), and re-freeze the loaded one so ONLY the new
    # adapter trains. ``stack_frozen_adapter`` sets ``requires_grad=False`` on every "frozen" param;
    # ``build_optimizer`` then collects ``[p for p in model.parameters() if p.requires_grad]`` (loop.py:
    # 51) so the frozen adapter is EXCLUDED from the optimizer (T-07-09-02 — no frozen-param leak).
    if config.conditioning.mode == "ic_lora" and config.conditioning.frozen_adapter_path:
        from signet_trainer.lora.peft import stack_frozen_adapter  # noqa: PLC0415

        # Volume base is FORMAT-dependent (D-7-FREEZE): a ``peft`` frozen adapter is a
        # signet-produced training OUTPUT and therefore lives on the CHECKPOINTS Volume (this is the
        # only format ``stack_frozen_adapter`` can load today — official ``comfy`` adapters are
        # single-file comfy-key safetensors that need a comfy->PEFT conversion first and, once
        # downloaded, live under the WEIGHTS Volume). Resolving a signet-PEFT adapter under
        # WEIGHTS_DIR would never find it (it is written under ``outputs/<run>/checkpoint-*`` on the
        # checkpoints Volume), so the base is selected by ``frozen_adapter_format``.
        frozen_base = (
            CHECKPOINTS_DIR
            if config.conditioning.frozen_adapter_format == "peft"
            else WEIGHTS_DIR
        )
        frozen_path = frozen_base / config.conditioning.frozen_adapter_path
        model = stack_frozen_adapter(model, frozen_path)
        _trainable = [p for p in model.parameters() if p.requires_grad]
        _n_trainable = sum(p.numel() for p in _trainable)
        _n_frozen = sum(
            p.numel() for n, p in model.named_parameters() if "frozen" in n and not p.requires_grad
        )
        print(
            f"[train][ic_lora] stacked frozen adapter from {frozen_path} (D-7-FREEZE) — optimizer "
            f"trains only the {len(_trainable)} requires_grad tensors ({_n_trainable} params); "
            f"{_n_frozen} frozen-adapter params EXCLUDED (T-07-09-02)."
        )

    # ── (8) checkpoint manager (bounded retention) + chained warm-restart cold-start branch ─────
    # keep_n threads the D-9-RECIPE gap-fill: prune to the N most-recent checkpoints after each save
    # (config.training.keep_checkpoints; None = unbounded, byte-identical to the prior behavior).
    ckpt_manager = CheckpointManager(
        CHECKPOINTS_DIR / config.output_dir, keep_n=config.training.keep_checkpoints
    )

    # D-9-CHAINED chained warm-restart: at a true COLD START (no in-dir checkpoint) with
    # init_adapter_path set, warm-start the trainable "default" adapter from the PRIOR round's final
    # adapter (prior-project --p1-checkpoint semantics) and let the optimizer below be built FRESH over the
    # warm-started weights (step 0, no optimizer-state carry). An in-dir checkpoint takes PRECEDENCE:
    # train_loop's resume() re-injects that adapter + restores optimizer/scheduler/step, and
    # init_adapter_path is ignored (should_warm_start returns False). init_adapter_path is already
    # validated Volume-relative at config load (09-02, T-07-02) BEFORE it is joined under
    # CHECKPOINTS_DIR here — no absolute/'..' path reaches load_adapter_into (T-09-05-01).
    if should_warm_start(
        ckpt_manager.find_latest() is not None, config.training.init_adapter_path
    ):
        from signet_trainer.lora.peft import load_adapter_into  # noqa: PLC0415

        init_dir = CHECKPOINTS_DIR / config.training.init_adapter_path
        load_adapter_into(model, init_dir)
        print(
            f"[train][chain] warm-started from {init_dir} (fresh optimizer, step 0) — chained round "
            "D-9-CHAINED; the prior round's adapter is loaded, the optimizer is built fresh below "
            "(no optimizer-state carry)."
        )

    # ── (8b) optimizer (adamw8bit) + cosine scheduler — built AFTER the warm-start branch ────────
    # build_optimizer filters to requires_grad params (loop.py:51) — the D-7-FREEZE frozen adapter
    # (requires_grad=False after stack_frozen_adapter) is excluded, so only the new adapter trains.
    # Building it AFTER the warm-restart load means a chained round gets a FRESH optimizer over the
    # warm-started weights (D-9-CHAINED — no optimizer-state carry across rounds).
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps=config.training.max_steps)
    schedule = FlowMatchingSchedule(uniform_prob=config.training.uniform_prob)

    # ── (9) the step-driven loop: "done" == checkpoint-step-{max_steps} committed to the Volume ─
    final_step = train_loop(
        model,
        dataset,
        optimizer,
        scheduler,
        schedule,
        ckpt_manager,
        config,
        checkpoints_vol,  # commit-per-save: a preemption can't vanish an uncommitted checkpoint.
        on_checkpoint=_in_loop_sample,  # OFFL-02 mid-run in-loop sample at every checkpoint boundary.
    )
    print(
        f"[train] done — reached step {final_step}/{config.training.max_steps}; final checkpoint "
        f"committed to signe-trainer-checkpoints under {config.output_dir}/ (commit-or-vanish)."
    )


@app.function(
    gpu="A100-80GB",  # inference/sampling; H100 optional.
    image=gpu_image,  # heavy image (torch/ltx-core) — same as train/load_ltxv_smoke (Phase-4 body).
    # DATASET_MOUNT is added because the 6 clip captions (the base-vs-LoRA prompt set) ride the
    # dataset Volume; WEIGHTS for the 22B checkpoint+Gemma, CHECKPOINTS for the adapter + samples out.
    volumes={**WEIGHTS_MOUNT, **DATASET_MOUNT, **CHECKPOINTS_MOUNT},
    secrets=[huggingface_secret],
    timeout=TWENTY_FOUR_HOURS,  # authoritative timeout now comes from the entrypoint's
    # .with_options(timeout=est_hours*margin) (09.1-04 Task 1); this decorator value is only the
    # ceiling for a direct call. Do NOT tune it here.
    # AUDIT-#1 (d): mirror train()'s server-side retries so a Modal PREEMPTION of a detached render
    # self-heals without a local re-dispatch (a preempted render was otherwise a total loss). Safe
    # because sample writes a FRESH timestamp samples dir per render (no half-written collision on
    # retry) and pairs with 09.1-03's artifact-freshness liveness gate.
    retries=modal.Retries(max_retries=3, initial_delay=60.0, backoff_coefficient=2.0),
)
def sample(config_yaml: str) -> None:
    """Stage 3 — the gated base-vs-LoRA validation grid (never auto-launched; gated by entrypoint).

    Mirrors ``train``'s body structure — ALL heavy imports function-local (Anti-Pattern 6):

        (1) COLD-PATH IMPORT PROBE — confirm peft / av (PyAV) / ltx_trainer.validation_sampler
            resolve in gpu_image BEFORE any model load (the Phase-2 cold-path lesson: an
            ImportError on the A100 wastes a metered launch; T-03-63). Fails loudly with the fix.
        (2) load + revalidate the YAML config in-container (config passed BY VALUE — the
            ``configs/`` dir is not shipped into the image, so a path would not resolve).
        (3) load the LTX-2.3 components with the video VAE DECODER ON (Phase-4 flip) so the
            ValidationSampler can decode latents back to pixels (one 22B load, reused for both
            columns — Open-Q1 (a), sequential VRAM discipline).
        (4) BASE column: render every prompt on the raw transformer at seed 42.
        (5) LoRA column: PEFT-wrap the SAME transformer with the Phase-3 adapter (reloaded from
            the checkpoints Volume) and render the same prompts at the same seed. If no adapter
            is present, fall back to a BASE-only grid.
        (6) write the base-vs-LoRA ``index.html`` montage + mp4s under the checkpoints Volume,
            then ``checkpoints_vol.commit()`` (commit-or-vanish — an uncommitted samples/ dir
            vanishes on container exit).

    Single-stage on A100-80GB (enochiatron-proven). Tiled VAE decode stays at its default-on
    (Pitfall 5 OOM guard). NO ``keep_warm`` / ``min_containers`` (D-10). Does NOT construct an
    offloader (standalone path — the offloader-suspend seam is the in-loop ``train`` path).
    """
    import datetime  # noqa: PLC0415

    # ── (1) COLD-PATH IMPORT PROBE — fail BEFORE any model load / sustained spend (T-03-63) ─────
    # No installs here (T-03-SC supply-chain): we VERIFY presence; a missing dep means the
    # pinned-SHA gpu_image must add it (re-gated by the Phase-2 supply-chain discipline).
    try:
        import av  # noqa: PLC0415, F401  (PyAV — save_video mp4 backend; RESEARCH A7)
        # multi_frame (Plan 06-08): the VideoConditionByLatentIndex path — probe it here so a missing
        # ltx_core surface fails loudly BEFORE the 22B load (T-06-SC cold-path guard, no new install).
        import ltx_core.conditioning.types.latent_cond  # noqa: PLC0415, F401
        import ltx_trainer.validation_sampler  # noqa: PLC0415, F401  (the ported sampler path)
        import peft  # noqa: PLC0415
        import torchvision  # noqa: PLC0415, F401  (read staged reference PNGs -> uint8 CHW; single_frame)
    except ImportError as exc:  # the cold-path bug class — name the missing dep + the fix.
        raise RuntimeError(
            f"[sample] cold-path dependency missing ({exc.name!r}). The gpu_image must carry "
            "peft / av (PyAV) / torchvision / ltx-trainer / ltx-core before any sustained GPU spend. "
            "Fix: ensure the pinned-SHA gpu_image installs 'peft>=0.14', PyAV, torchvision, the "
            "ltx-trainer package and ltx-core (VideoConditionByLatentIndex), then rebuild (re-gated "
            "by the Phase-2 supply-chain discipline, T-03-SC)."
        ) from exc
    print(
        f"[sample] cold-path imports OK — peft={peft.__version__} "
        f"av={getattr(av, '__version__', '?')} torchvision={getattr(torchvision, '__version__', '?')}"
    )

    from ltx_trainer.video_utils import save_video  # noqa: PLC0415

    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415
    from signet_trainer.inference.grid import slug, write_comparison_gallery  # noqa: PLC0415
    from signet_trainer.inference.lora_load import (  # noqa: PLC0415
        ADAPTER_FILENAME,
        load_lora_onto_transformer,
    )
    from signet_trainer.inference.sampler import build_generation_config, run_sampler  # noqa: PLC0415
    from signet_trainer.models.loader import load_ltxv_components  # noqa: PLC0415

    # ── (2) load + revalidate the config in-container (entrypoint passes the YAML TEXT by value) ─
    config = load_config_from_text(config_yaml)
    device = "cuda"
    checkpoint_path = str(WEIGHTS_DIR / config.model.model_id)
    text_encoder_path = str(WEIGHTS_DIR / config.model.text_encoder_id)

    prompts = list(config.validation.prompts)
    # Phase 9 (INPAINT): conditioned validation renders (validation.samples) carry their OWN
    # prompts — they need PHASE-A cached embeddings too (the masked render must never live-load
    # Gemma next to the 22B). Collected here so the pre-encode below covers them; the inpaint
    # branch renders validation.samples ONLY (validation.prompts drive the other modes).
    sample_prompts = [
        s.prompt for s in config.validation.samples if s.prompt not in prompts
    ]
    seed = config.validation.seed
    if not prompts and not sample_prompts:
        raise RuntimeError(
            "[sample] config.validation.prompts AND validation.samples are both empty — nothing "
            "to render (a metered GPU container with no prompts is wasted spend). Provide the "
            "clip captions + canonical prompts (04-04) or conditioned samples (Phase 9 inpaint)."
        )

    # Two-stage upscaler toggle (D-DEPTH-1, default OFF) — read up-front because it decides whether
    # the two-phase pre-encode applies. The two-stage path (a separately-gated H100 run, 04-05) is
    # OUT OF SCOPE for the two-phase port and keeps its legacy Gemma-loaded flow; every single-stage
    # render (single_frame / multi_frame / base-vs-LoRA) takes the two-phase path below.
    two_stage = config.validation.two_stage_upscale

    # ── (2b) PHASE A — prompt pre-encode BEFORE any 22B load (06-09 OOM fix, generalized) ────────
    # Runs 3-5 proved the A100-80GB cannot hold the 22B transformer + Gemma-12B + Gemma's forward
    # activations (77.29-77.87 GiB resident at every Gemma forward), and that in-place CPU-parking
    # cannot free SingleGPUModelBuilder's ``assign=True`` loader-owned CUDA storages (run 5:
    # transformer.to("cpu") + empty_cache freed ~nothing). Two-phase load: Gemma NEVER coexists
    # with the transformer. Load ONLY the text encoder + embeddings processor (~26GB peak), encode
    # every prompt once (+ the shared "" negative — guidance != 1 needs CFG context), cache via the
    # ltx-native ``GenerationConfig.cached_embeddings`` surface ("avoids loading Gemma", pinned
    # SHA), then DELETE both models — dropping the last reference frees the loader-owned storage
    # regardless of Module.to() semantics.
    #
    # 06-09 CARRY-FORWARD (operator-endorsed 06-10 prep): this ran ONLY for multi_frame in 06-09;
    # generalized here to EVERY single-stage mode so the single_frame + base-vs-LoRA branches
    # (which still called run_sampler raw at ~72GB residency, proven to OOM on the Phase-6-rebuilt
    # image, runs 3-4) get the SAME proven two-phase load. Every render reads its cached embeddings
    # via run_sampler(..., cached_embeddings=...) / run_multi_condition_sampler(..., cached_embeddings=...).
    cached_by_prompt: dict | None = None
    if not two_stage:
        import gc  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from ltx_trainer.model_loader import (  # noqa: PLC0415
            load_embeddings_processor,
            load_text_encoder,
        )
        from ltx_trainer.validation_sampler import CachedPromptEmbeddings  # noqa: PLC0415

        text_encoder = load_text_encoder(text_encoder_path, device=device)
        # EP MUST load on CPU (run-6 OOM): load_embeddings_processor builds from the FULL 44GB
        # ltx-2.3-22b-dev checkpoint via SingleGPUModelBuilder, which materializes the load on the
        # TARGET device — device="cuda" dragged ~50GB of checkpoint residency onto the GPU alongside
        # Gemma (78.03/79.25 GiB at the first prompt encode). Load on CPU exactly like signet's own
        # models/loader.py inference path ("enochiatron loads it on CPU for its sequential-VRAM
        # discipline"), then move ONLY the built module (~1-3GB of connector params) to the GPU.
        emb_processor = load_embeddings_processor(checkpoint_path=checkpoint_path, device="cpu")
        emb_processor.to(device)
        # no_grad is MANDATORY here (run-7 OOM): ltx-core GemmaTextEncoder.encode() runs the full
        # 12B 48-layer forward with autograd ON (no no_grad anywhere in the source method) and pads
        # EVERY prompt to 1024 tokens (LTXVGemmaTokenizer(root, 1024)) — one encode's retained
        # autograd graph is ~20-25GB, and two coexisting graphs (retained negative + in-flight
        # positive) hit 78.03/79.25 GiB with only ~26GB of weights loaded (runs 6-7, bit-identical
        # allocation across different load compositions). Under no_grad a forward is ~1-2GB
        # transient. Intermediates are del'd promptly so no two encodes' hidden-state tuples
        # coexist; the cached tensors are detached CPU copies that retain no graph references.
        cached_by_prompt = {}
        with torch.no_grad():
            neg_hs, neg_mask = text_encoder.encode("")
            neg_out = emb_processor.process_hidden_states(neg_hs, neg_mask)
            neg_video = neg_out.video_encoding.detach().to("cpu")
            neg_audio = neg_out.audio_encoding.detach().to("cpu")
            del neg_hs, neg_mask, neg_out  # drop the 49-layer hidden-state tuple before the next forward
            for prompt in (*prompts, *sample_prompts):
                pos_hs, pos_mask = text_encoder.encode(prompt)
                pos_out = emb_processor.process_hidden_states(pos_hs, pos_mask)
                # Stored on CPU — _get_prompt_embeddings .to(device)'s them per render (tiny tensors).
                cached_by_prompt[prompt] = CachedPromptEmbeddings(
                    video_context_positive=pos_out.video_encoding.detach().to("cpu"),
                    audio_context_positive=pos_out.audio_encoding.detach().to("cpu"),
                    video_context_negative=neg_video,
                    audio_context_negative=neg_audio,
                )
                del pos_hs, pos_mask, pos_out  # ditto — never two encodes' intermediates at once
        del text_encoder, emb_processor
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"[sample][{config.conditioning.mode}] PHASE A pre-encoded {len(cached_by_prompt)} "
            f"prompts -> cached embeddings; Gemma deleted (two-phase load, 06-09 OOM fix, "
            f"generalized to all single-stage modes); cuda "
            f"allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB."
        )

    # ── (3) load the LTX-2.3 components with the DECODER ON (inference; Phase-4 flip, MODL-01) ───
    # PHASE B of the two-phase load: with_text_encoder=False whenever PHASE A cached the prompt
    # embeddings (multi_frame) — the render loop never touches Gemma, so it is never loaded
    # alongside the 22B transformer. Every other mode keeps the byte-identical default (True).
    components = load_ltxv_components(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=device,
        with_video_vae_decoder=True,
        with_text_encoder=cached_by_prompt is None,
        # a2v render (2026-07-15 burned dispatch): the driving .wav is VAE-encoded at sample time,
        # so the audio VAE ENCODER must ride the components load for audio_to_video mode — the
        # run_audio_condition_sampler guard hard-fails without it. All other modes: byte-identical
        # default (False; decoder/vocoder stay OFF regardless).
        with_audio_vae_encoder=config.conditioning.mode == "audio_to_video",
    )

    fps = config.validation.frame_rate

    # ── (3b') INPAINT masked-render branch (Phase 9, GATE-SPEC rev 2 item 6) ─────────────────────
    # Render each validation.samples entry — a held-out TEST clip masked + regenerated through the
    # LATEST trained adapter (the operator's parallel-testing venue: the watcher dispatches this per new
    # checkpoint; 3 masked test videos, renders every 600 + final). Writes STABLE step-keyed files
    # (samples_inpaint/<test-stem>/step_<N>.mp4 — the gridwatch per-dir/step_N.mp4 convention, NOT
    # a ts-stamped dir) so columns accumulate across dispatches. validation.prompts are IGNORED in
    # this branch (they drive the other modes); each sample's own prompt was PHASE-A pre-encoded.
    if config.conditioning.mode == "inpaint":
        if two_stage:
            raise RuntimeError(
                "[sample][inpaint] two_stage_upscale is not supported for masked renders — the "
                "two-pass path obliterates likeness unless LoRA rides BOTH stages (precedent D9/F5) "
                "and skips the PHASE-A pre-encode (proven A100 OOM). Single-pass at trained res "
                "only (GATE-SPEC rev 2 inference landmines); set validation.two_stage_upscale: "
                "false."
            )
        v_samples = list(config.validation.samples)
        if not v_samples:
            raise RuntimeError(
                "[sample][inpaint] conditioning.mode == 'inpaint' but validation.samples is empty "
                "— nothing to render. Provide the masked test clips as validation.samples "
                "({prompt, conditions: [{type: mask, video, mask}]})."
            )
        import re  # noqa: PLC0415
        import shutil  # noqa: PLC0415
        from pathlib import Path as _P  # noqa: PLC0415

        from signet_trainer.inference.sampler import (  # noqa: PLC0415
            plan_mask_condition,
            run_mask_condition_sampler,
        )
        from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415

        # Latest committed adapter (D-8-REFLOAD: find_latest, never hand-glob; per-STEP dirs).
        checkpoints_vol.reload()
        ckpt_root = CHECKPOINTS_DIR / config.output_dir
        latest_ckpt = CheckpointManager(ckpt_root).find_latest()
        if latest_ckpt is not None:
            render_model = load_lora_onto_transformer(
                components.transformer, latest_ckpt / ADAPTER_FILENAME, lora_scale=1.0
            )
            _m = re.search(r"step-(\d+)", latest_ckpt.name)
            step = int(_m.group(1)) if _m else -1
            print(
                f"[sample][inpaint] rendering with trained adapter {latest_ckpt.name} "
                f"(step {step}, lora_scale=1.0)."
            )
        else:
            # Honest fallback (CR-02 doctrine): no adapter yet -> the column is the BASE (fused)
            # transformer's masked render, labeled step_0 — never silently presented as trained.
            render_model = components.transformer
            step = 0
            print(
                f"[sample][inpaint] WARNING: no trained adapter under {ckpt_root} — rendering the "
                "BASE transformer (step_0 column); a quality verdict on this output validates the "
                "base, not a trained LoRA."
            )

        out_root = CHECKPOINTS_DIR / config.output_dir / "samples_inpaint"
        n_rendered = 0
        for s in v_samples:
            cached = cached_by_prompt[s.prompt] if cached_by_prompt is not None else None
            for cond in s.conditions:
                video_rel, mask_rel = plan_mask_condition(cond)
                stem_dir = out_root / slug(_P(video_rel).stem)
                stem_dir.mkdir(parents=True, exist_ok=True)
                # Stage the raw input clip once per stem as its own comparison column source
                # (named input.mp4, NEVER step_N — an input masquerading as a render would poison
                # the grid verdict).
                _input_copy = stem_dir / "input.mp4"
                _input_src = CHECKPOINTS_DIR / video_rel
                if not _input_copy.exists() and _input_src.exists():
                    shutil.copyfile(_input_src, _input_copy)
                video, _audio = run_mask_condition_sampler(
                    components,
                    render_model,
                    config,
                    video_rel,
                    mask_rel,
                    device=device,
                    cached_embeddings=cached,
                    prompt=s.prompt,
                )
                fname = f"step_{step}.mp4"
                save_video(video, str(stem_dir / fname), fps=fps)
                n_rendered += 1
                print(f"[sample][inpaint] wrote {stem_dir.name}/{fname}")
        checkpoints_vol.commit()  # commit-or-vanish — uncommitted renders vanish on exit.
        print(
            f"[sample][inpaint] done — {n_rendered} masked render(s) at step {step} committed "
            f"under {out_root} (stable step-keyed layout for the parallel watcher/gridwatch)."
        )
        return

    # ── (3b'') AUDIO-TO-VIDEO driving-audio render branch (Phase 9, GATE-SPEC rev 2 item 7) ───────
    # Render each validation.samples entry — a video DRIVEN by an input .wav through the LATEST
    # trained adapter — mirroring the inpaint branch's step-keyed parallel-watcher layout. ⚠ The
    # actual audio encode + frozen-audio render is the flagged LIVE-GPU integration point
    # (run_audio_condition_sampler): a2v is DATA-BLOCKED (3090ti audio) + gated, so this branch is
    # wired end-to-end for the config→render path but fail-fasts at the audio-encode seam until the
    # gated live validation fills it (correct behaviour — never a silent/fake a2v render).
    if config.conditioning.mode == "audio_to_video":
        if two_stage:
            raise RuntimeError(
                "[sample][a2v] two_stage_upscale is not supported for a2v — the distilled two-stage "
                "path has no input-audio surface (D-7-BASEVAR). Single-pass dev base only; set "
                "validation.two_stage_upscale: false."
            )
        v_samples = list(config.validation.samples)
        if not v_samples:
            raise RuntimeError(
                "[sample][a2v] conditioning.mode == 'audio_to_video' but validation.samples is empty "
                "— nothing to render. Provide the driving-audio test clips as validation.samples "
                "({prompt, conditions: [{type: audio, audio: <.wav>}]})."
            )
        import re  # noqa: PLC0415
        from pathlib import Path as _P  # noqa: PLC0415

        from signet_trainer.inference.sampler import (  # noqa: PLC0415
            plan_audio_condition,
            run_audio_condition_sampler,
        )
        from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415

        checkpoints_vol.reload()
        ckpt_root = CHECKPOINTS_DIR / config.output_dir
        latest_ckpt = CheckpointManager(ckpt_root).find_latest()
        if latest_ckpt is not None:
            render_model = load_lora_onto_transformer(
                components.transformer, latest_ckpt / ADAPTER_FILENAME, lora_scale=1.0
            )
            _m = re.search(r"step-(\d+)", latest_ckpt.name)
            step = int(_m.group(1)) if _m else -1
            print(
                f"[sample][a2v] rendering with trained adapter {latest_ckpt.name} "
                f"(step {step}, lora_scale=1.0)."
            )
        else:
            render_model = components.transformer
            step = 0
            print(
                f"[sample][a2v] WARNING: no trained adapter under {ckpt_root} — rendering the BASE "
                "transformer (step_0 column); a verdict on this output validates the base, not a "
                "trained a2v LoRA."
            )

        out_root = CHECKPOINTS_DIR / config.output_dir / "samples_a2v"
        n_rendered = 0
        for s in v_samples:
            cached = cached_by_prompt[s.prompt] if cached_by_prompt is not None else None
            for cond in s.conditions:
                audio_rel = plan_audio_condition(cond)
                stem_dir = out_root / slug(_P(audio_rel).stem)
                stem_dir.mkdir(parents=True, exist_ok=True)
                video, _audio = run_audio_condition_sampler(
                    components,
                    render_model,
                    config,
                    audio_rel,
                    device=device,
                    cached_embeddings=cached,
                    prompt=s.prompt,
                )
                fname = f"step_{step}.mp4"
                save_video(video, str(stem_dir / fname), fps=fps)
                n_rendered += 1
                print(f"[sample][a2v] wrote {stem_dir.name}/{fname}")
        checkpoints_vol.commit()  # commit-or-vanish — uncommitted renders vanish on exit.
        print(
            f"[sample][a2v] done — {n_rendered} a2v render(s) at step {step} committed under "
            f"{out_root} (stable step-keyed layout for the parallel watcher/gridwatch)."
        )
        return

    # ── (3c) SINGLE-FRAME reference-control branch (D-METER base-model proof; SC#2 / SC#3) ────────
    # For ``conditioning.mode == "single_frame"`` render ref-ON (``condition_image=frame0``) vs
    # ref-OFF (``condition_image=None``) on the BASE ``components.transformer`` at seed 42 — NO LoRA
    # required (D-METER) — and write the reference | ref-ON | ref-OFF montage. This is exactly the
    # plumbing the SEPARATELY-GATED 05-06 run drives; the base-vs-LoRA path below is untouched for
    # every other mode. NO offloader is constructed (standalone path). NO ``keep_warm`` (D-10).
    if config.conditioning.mode == "single_frame":
        # Defensive guard (WR-02, mirrors the multi_frame branch's cached_by_prompt raise): with
        # two_stage set, PHASE A is skipped (cached_by_prompt is None, with_text_encoder=True) and
        # every run_sampler call below would live-encode with Gemma alongside the loaded 22B
        # transformer — the exact 77-79 GiB A100-80GB residency that OOM'd runs 3-4. This branch
        # also renders via run_sampler only (never _render), so two_stage would be silently
        # ignored anyway — fail fast instead of burning a metered container.
        if two_stage:
            raise RuntimeError(
                "[sample][single_frame] two_stage_upscale is not supported for single_frame — "
                "the two-stage path skips the PHASE-A prompt pre-encode, so Gemma would coexist "
                "with the 22B transformer (proven A100-80GB OOM, 06-09 runs 3-4), and this branch "
                "renders single-stage only (the toggle would be silently ignored). Set "
                "validation.two_stage_upscale: false for single_frame sample runs."
            )

        from torchvision.io import read_image  # noqa: PLC0415  (staged PNG -> uint8 [C,H,W])
        from torchvision.utils import save_image  # noqa: PLC0415  (reference thumbnail)

        from signet_trainer.inference.grid import write_reference_gallery  # noqa: PLC0415
        from signet_trainer.inference.reference import to_condition_image  # noqa: PLC0415

        ref_images = list(config.conditioning.reference_images)
        if not ref_images:
            raise RuntimeError(
                "[sample] conditioning.mode == 'single_frame' but conditioning.reference_images is "
                "empty — nothing to condition on. Stage the reference images "
                "(scripts/_stage_reference_images.py, 05-05 Task 1) and list them in the run config."
            )

        # Read the latest committed reference images from the checkpoints Volume.
        checkpoints_vol.reload()

        # REF-LOAD (D-8-REFLOAD, Pitfall 5): LOAD THE LATEST TRAINED ADAPTER before the render loop.
        # Without this, ref-ON and ref-OFF both render on the RAW BASE transformer, so any
        # convergence/quality verdict off this grid would validate the base model, NOT the trained
        # LoRA (a wrong verdict from a metered run). Mirror the ic_lora branch (§3e) + base-vs-LoRA
        # tail (§5): resolve the highest-step checkpoint via find_latest() (the CheckpointManager
        # writes per-STEP dirs, NEVER a flat file — do not glob by hand), then PEFT-wrap the
        # transformer. BOTH ref-ON and ref-OFF run through this SAME trained adapter (the divergence
        # signal is ref-vs-no-ref, both on the adapter). If no adapter exists yet (wiring mode), fall
        # back to the base and say so loudly so the verdict stays honest.
        from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415 — import-light

        ckpt_root = CHECKPOINTS_DIR / config.output_dir
        latest_ckpt = CheckpointManager(ckpt_root).find_latest()
        adapter_path = (latest_ckpt / ADAPTER_FILENAME) if latest_ckpt is not None else (
            ckpt_root / ADAPTER_FILENAME
        )
        if adapter_path.exists():
            render_model = load_lora_onto_transformer(
                components.transformer, adapter_path, lora_scale=1.0
            )
            sf_lora_scale: float | str = 1.0
            print(
                f"[sample][single_frame] loaded trained adapter {adapter_path.name} onto the "
                "transformer — ref-ON/ref-OFF render WITH the adapter."
            )
        else:
            render_model = components.transformer
            sf_lora_scale = "none"
            print(
                f"[sample][single_frame] WARNING: no trained adapter at {adapter_path} — rendering "
                "on the BASE transformer (wiring mode); a quality/convergence verdict would validate "
                "the BASE, not a trained adapter. Run train first."
            )

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sf_root = CHECKPOINTS_DIR / config.output_dir / "samples_single_frame" / ts
        ref_thumb_dir = sf_root / "reference"
        ref_on_dir = sf_root / "ref_on"
        ref_off_dir = sf_root / "ref_off"
        for d in (ref_thumb_dir, ref_on_dir, ref_off_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Pair each reference image POSITIONALLY with its prompt (D-GRID). Prompts beyond the
        # reference list (e.g. the trailing canonical yardstick, D-PROMPT-1) render ref-OFF only.
        rows: list[dict] = []
        for idx, prompt in enumerate(prompts):
            ref_rel = ref_images[idx] if idx < len(ref_images) else None
            row: dict = {"prompt": prompt, "seed": seed}
            # PHASE-A cached embeddings (two-phase load, 06-09 carry-forward) — Gemma is already
            # deleted, so every render below MUST feed its cached embeddings (run_sampler short-
            # circuits _get_prompt_embeddings). None only if two_stage (single_frame never two-stages).
            cached = cached_by_prompt[prompt] if cached_by_prompt is not None else None

            condition_image = None
            if ref_rel is not None:
                ref_path = CHECKPOINTS_DIR / ref_rel
                if not ref_path.exists():
                    raise RuntimeError(
                        f"[sample] reference image {ref_path} not found on the checkpoints Volume — "
                        "run scripts/_stage_reference_images.py first (05-05 Task 1)."
                    )
                # Decode the staged PNG -> uint8 [C,H,W] -> normalized [3,H,W] float in [0,1].
                frame = read_image(str(ref_path))
                condition_image = to_condition_image(frame)
                thumb_name = f"{slug(prompt)}_ref.png"
                save_image(condition_image, str(ref_thumb_dir / thumb_name))
                row["reference_img"] = f"reference/{thumb_name}"

                # ref-ON: condition on frame0 (CONTRADICTION #1 — GenerationConfig.condition_image).
                on_cfg = build_generation_config(
                    config, prompt=prompt, seed=seed, condition_image=condition_image
                )
                on_video, _a = run_sampler(
                    components, render_model, on_cfg, device=device, cached_embeddings=cached
                )
                on_name = f"{slug(prompt)}_s{seed}_on.mp4"
                save_video(on_video, str(ref_on_dir / on_name), fps=fps)
                row["ref_on_mp4"] = f"ref_on/{on_name}"

            # ref-OFF control: condition_image=None, SAME prompt+seed — the SC#3 divergence signal.
            off_cfg = build_generation_config(config, prompt=prompt, seed=seed, condition_image=None)
            off_video, _a = run_sampler(
                components, render_model, off_cfg, device=device, cached_embeddings=cached
            )
            off_name = f"{slug(prompt)}_s{seed}_off.mp4"
            save_video(off_video, str(ref_off_dir / off_name), fps=fps)
            row["ref_off_mp4"] = f"ref_off/{off_name}"

            rows.append(row)

        v = config.validation
        index_path = write_reference_gallery(
            rows,
            sf_root / "index.html",
            params={
                "steps": v.num_inference_steps,
                "guidance": v.guidance_scale,
                "stg_scale": v.stg_scale,
                "width": v.width,
                "height": v.height,
                "frames": v.frame_count,
                "first_frame_conditioning": "on/off",
                # CR-02: name WHICH model rendered — 1.0 when the trained adapter is applied,
                # "none" when the branch fell back to the base transformer (wiring mode). Never
                # leave the single_frame artifact silent about the substrate.
                "lora_scale": sf_lora_scale,
            },
        )

        # Commit-or-vanish: without this the samples_single_frame/ dir vanishes on container exit.
        checkpoints_vol.commit()
        print(
            f"[sample][single_frame] wrote {len(rows)} reference rows -> {index_path} "
            "(committed to signe-trainer-checkpoints, commit-or-vanish)."
        )
        return

    # ── (3d) MULTI-FRAME reference-control branch (D-6-METER tier 2; SC#3) ────────────────────────
    # For ``conditioning.mode == "multi_frame"`` render the N=2 / N=3 / strength-lo / strength-hi /
    # no-reference columns on the BASE ``components.transformer`` at seed 42 — NO LoRA required
    # (D-6-METER tier 2), NO offloader, NO keep_warm — via the multi-condition sampler, and write the
    # write_multi_frame_gallery PASS/FAIL grid. This is exactly the plumbing the SEPARATELY-GATED
    # Plan 06-09 run drives; the single_frame and base-vs-LoRA paths below are untouched. The GPU
    # render is proven by that gated run — this branch wires the seam (column plan + per-column view).
    if config.conditioning.mode == "multi_frame":
        from torchvision.io import read_image  # noqa: PLC0415  (staged keyframe PNG -> uint8 [C,H,W])
        from torchvision.utils import save_image  # noqa: PLC0415  (keyframe reference thumbnails)

        from signet_trainer.config.schema import ConditioningItem  # noqa: PLC0415
        from signet_trainer.inference.grid import write_multi_frame_gallery  # noqa: PLC0415
        from signet_trainer.inference.multi_condition import (  # noqa: PLC0415
            plan_multi_frame_columns,
            run_multi_condition_sampler,
        )
        from signet_trainer.inference.reference import to_condition_image  # noqa: PLC0415

        cond_items = list(config.conditioning.conditioning_items)
        if not cond_items:
            raise RuntimeError(
                "[sample] conditioning.mode == 'multi_frame' but conditioning.conditioning_items is "
                "empty — nothing to condition on. Stage the keyframe images "
                "(scripts/_stage_multi_frame_refs.py, 06-07) and list them as conditioning_items "
                "(image / frame_index / strength) in the run config."
            )

        # Read the latest committed keyframe reference images from the checkpoints Volume.
        checkpoints_vol.reload()

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sf_root = CHECKPOINTS_DIR / config.output_dir / "samples_multi_frame" / ts
        ref_thumb_dir = sf_root / "reference"
        video_dir = sf_root / "videos"
        for d in (ref_thumb_dir, video_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Stage the keyframe reference thumbnails ONCE (shared across every prompt row). The staged
        # PNGs are Volume-relative config paths (T-06-02: reuse the single_frame CHECKPOINTS_DIR /
        # rel-path pattern; existence-checked before any render spend).
        reference_imgs: list[str] = []
        for i, item in enumerate(cond_items):
            ref_path = CHECKPOINTS_DIR / item.image
            if not ref_path.exists():
                raise RuntimeError(
                    f"[sample] keyframe reference image {ref_path} not found on the checkpoints "
                    "Volume — run scripts/_stage_multi_frame_refs.py first (06-07)."
                )
            thumb = to_condition_image(read_image(str(ref_path)))
            thumb_name = f"keyframe{i}_f{item.frame_index}.png"
            save_image(thumb, str(ref_thumb_dir / thumb_name))
            reference_imgs.append(f"reference/{thumb_name}")

        # Cached embeddings come from the PHASE-A pre-encode (two-phase load above, 06-09 OOM
        # fix) — ``components.text_encoder`` is None in this mode and the render loop never
        # touches Gemma. Defensive: the mode gate guarantees PHASE A ran.
        if cached_by_prompt is None:
            raise RuntimeError(
                "[sample][multi_frame] PHASE-A cached embeddings missing — the two-phase load "
                "gate (2b) must run for conditioning.mode == 'multi_frame'."
            )

        # REF-LOAD (D-8-REFLOAD, Pitfall 5): LOAD THE LATEST TRAINED ADAPTER before the render loop.
        # Without this every column (including the no-reference control) renders on the RAW BASE
        # transformer while the grid banner claims lora_scale 1.0 — so a convergence/quality verdict
        # off this grid would validate the base model, NOT the trained LoRA (a wrong verdict from a
        # metered run). Mirror the ic_lora branch (§3e) + base-vs-LoRA tail (§5): resolve the highest-
        # step checkpoint via find_latest() (the CheckpointManager writes per-STEP dirs, NEVER a flat
        # file — do not glob by hand), then PEFT-wrap the transformer. ALL columns render through this
        # SAME trained adapter. If no adapter exists yet (wiring mode), fall back to the base and say
        # so loudly so the verdict stays honest.
        from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415 — import-light

        ckpt_root = CHECKPOINTS_DIR / config.output_dir
        latest_ckpt = CheckpointManager(ckpt_root).find_latest()
        adapter_path = (latest_ckpt / ADAPTER_FILENAME) if latest_ckpt is not None else (
            ckpt_root / ADAPTER_FILENAME
        )
        if adapter_path.exists():
            render_model = load_lora_onto_transformer(
                components.transformer, adapter_path, lora_scale=1.0
            )
            mf_lora_scale: float | str = 1.0
            print(
                f"[sample][multi_frame] loaded trained adapter {adapter_path.name} onto the "
                "transformer — all columns render WITH the adapter."
            )
        else:
            render_model = components.transformer
            mf_lora_scale = "none"
            print(
                f"[sample][multi_frame] WARNING: no trained adapter at {adapter_path} — rendering "
                "on the BASE transformer (wiring mode); a quality/convergence verdict would validate "
                "the BASE, not a trained adapter. Run train first."
            )

        # The ordered SC#3 grid columns (N=2 / N=3 / strength-lo / strength-hi / no-reference).
        # PURE/CPU column plan — all strengths flow from config (conditioning_strength_range).
        columns = plan_multi_frame_columns(config)

        rows: list[dict] = []
        for prompt in prompts:
            row: dict = {"prompt": prompt, "seed": seed, "reference_imgs": reference_imgs}
            for column in columns:
                # Per-column config VIEW: rebuild conditioning_items for this column's keyframe subset
                # and swept strengths (config-driven — the strengths come from plan_multi_frame_columns,
                # never hardcoded), and pin validation.prompts to this single row's prompt. The
                # no-reference control has an empty item list (the base model with no conditioning).
                # ``image`` is a Volume-relative config string (schema: "resolved Modal-side where the
                # Volume mounts") — resolve it against CHECKPOINTS_DIR here, exactly like the
                # single_frame branch's ``CHECKPOINTS_DIR / ref_rel``, so run_multi_condition_sampler's
                # read_image gets a container-absolute path (06-09 GPU run: the raw relative string
                # raised FileNotFoundError inside the container cwd).
                col_items = [
                    ConditioningItem(
                        image=str(CHECKPOINTS_DIR / cond_items[idx].image),
                        frame_index=cond_items[idx].frame_index,
                        strength=strength,
                    )
                    for idx, (_latent_idx, strength) in enumerate(column.items)
                ]
                col_config = config.model_copy(deep=True)
                col_config.conditioning.conditioning_items = col_items
                col_config.validation.prompts = [prompt]

                video, _audio = run_multi_condition_sampler(
                    components,
                    render_model,
                    col_config,
                    device=device,
                    cached_embeddings=cached_by_prompt[prompt],
                )
                # column.row_key is one of n2_mp4/n3_mp4/strength_lo_mp4/strength_hi_mp4/no_ref_mp4;
                # strip the _mp4 suffix for the on-disk filename stem.
                stem = column.row_key[: -len("_mp4")]
                fname = f"{slug(prompt)}_{stem}_s{seed}.mp4"
                save_video(video, str(video_dir / fname), fps=fps)
                row[column.row_key] = f"videos/{fname}"
                # Config-driven column label into the gallery row (WR-01): the planner's label
                # carries the ACTUAL swept strengths (conditioning_strength_range endpoints) —
                # _multi_frame_block reads strength_lo_label/strength_hi_label instead of a
                # hardcoded "mid strength 0.5/1.0" literal (D-6-STRENGTHCOL / D-NOHARDCODE).
                row[f"{stem}_label"] = column.label
            rows.append(row)

        v = config.validation
        index_path = write_multi_frame_gallery(
            rows,
            sf_root / "index.html",
            params={
                "steps": v.num_inference_steps,
                "guidance": v.guidance_scale,
                "stg_scale": v.stg_scale,
                "width": v.width,
                "height": v.height,
                "frames": v.frame_count,
                # CR-02: reflect REALITY — 1.0 when the trained adapter is applied, "none" when the
                # branch fell back to the base transformer (wiring mode). A banner claiming 1.0 while
                # every column rendered on the raw base is the "wrong verdict from a metered run" class.
                "lora_scale": mf_lora_scale,
            },
        )

        # Commit-or-vanish: without this the samples_multi_frame/ dir vanishes on container exit.
        checkpoints_vol.commit()
        print(
            f"[sample][multi_frame] wrote {len(rows)} multi-frame rows x {len(columns)} columns -> "
            f"{index_path} (committed to signe-trainer-checkpoints, commit-or-vanish)."
        )
        return

    # ── (3e-baseline) IC-LoRA official-adapter KNOWN-GOOD baseline (D-7-LADDER tier 2 / D-7-BASELINE) ─
    # For ``conditioning.mode == "ic_lora"`` AND ``validation.two_stage_upscale`` (the distilled base
    # variant, D-7-BASEVAR) run the OFFICIAL Lightricks Union-Control IC-LoRA through the ported distilled
    # two-stage ``ICLoraPipeline`` (inference/ic_lora_pipeline.py) — the tier-2 yardstick that validates
    # the ported inference SURFACE against a KNOWN-good adapter BEFORE we train ours (kills
    # pipeline-vs-adapter ambiguity). This is a DIFFERENT substrate from the dev single-stage PEFT V2V
    # branch below (Pitfall 4 — do NOT cross the streams): comfy-format adapter through ``loras=`` (NEVER
    # signet's PEFT loader), distilled + spatial-upscaler weights, ``ref0.5`` (downscale=2 read from the
    # adapter metadata). ``two_stage`` skipped PHASE A above (cached_by_prompt None) — ICLoraPipeline owns
    # its own Gemma via gemma_root on OffloadMode.CPU (the Pitfall-6 CPU-offload precedent). NO offloader,
    # NO keep_warm (D-10). The metered run is the gated 07-10 Task-2 exercise (D-7-BLANKET).
    if config.conditioning.mode == "ic_lora" and two_stage:
        # Cold-path probe for the baseline-only heavy dep (ltx_pipelines.ic_lora) — fail BEFORE any 22B
        # load if the pinned-SHA gpu_image lacks it, mirroring the (1) probe (T-03-63; no new install).
        try:
            import ltx_pipelines.ic_lora  # noqa: PLC0415, F401
        except ImportError as exc:
            raise RuntimeError(
                f"[sample][ic_lora_baseline] cold-path dependency missing ({exc.name!r}). The "
                "gpu_image must carry ltx_pipelines (ICLoraPipeline) before any sustained GPU spend. "
                "Fix: ensure the pinned-SHA gpu_image installs ltx-pipelines, then rebuild (re-gated "
                "by the supply-chain discipline, T-07-10-SC)."
            ) from exc

        from huggingface_hub import hf_hub_download  # noqa: PLC0415 — small comfy adapter fetch

        from signet_trainer.inference.grid import write_reskin_gallery  # noqa: PLC0415
        from signet_trainer.inference.ic_lora_pipeline import (  # noqa: PLC0415
            read_reference_downscale_factor,
            run_ic_lora_baseline,
        )

        # The seg-map REFERENCE videos the official adapter steers on, paired POSITIONALLY with the
        # prompts (mirrors the dev branch below). Volume-relative strings, resolved Modal-side.
        reference_videos = list(config.conditioning.reference_images)
        if not reference_videos:
            raise RuntimeError(
                "[sample][ic_lora_baseline] conditioning.reference_images is empty — nothing to "
                "condition the known-good baseline on. Stage the seg-map reference clips and list "
                "them Volume-relative under conditioning.reference_images (paired with prompts)."
            )

        # Distilled base + 2x spatial upscaler weights (pre-downloaded via download_weights(two_stage=
        # True) — NOT re-fetched on the metered GPU). Reuse the upscale.py filename references (D-7-BASEVAR).
        distilled_ckpt_path = str(WEIGHTS_DIR / "ltx-2.3-22b-distilled-1.1.safetensors")
        upscaler_ckpt_path = str(WEIGHTS_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        gemma_root = str(WEIGHTS_DIR / config.model.text_encoder_id)

        # Fetch the OFFICIAL comfy-format Union-Control adapter (gated HF token, small single-file
        # safetensors — no pickle). NOT loaded via signet's PEFT loader (Pitfall 4 format-mismatch guard):
        # it is handed to ICLoraPipeline(loras=[...]) ONLY, inside run_ic_lora_baseline.
        official_adapter_path = hf_hub_download(
            repo_id=OFFICIAL_IC_LORA_BASELINE_REPO,
            filename=OFFICIAL_IC_LORA_BASELINE_FILENAME,
            local_dir=str(WEIGHTS_DIR),
        )
        weights_vol.commit()  # commit-or-vanish: the fetched adapter survives the container.

        # Honor the adapter's reference_downscale_factor READ FROM ITS METADATA (Union-Control is ref0.5
        # -> 2, so the baseline supplies a half-res reference; the pipeline reads the same value from
        # metadata internally). Do NOT conflate with signet's own dev downscale=1 training (D-7-REF11).
        ref_downscale = read_reference_downscale_factor(official_adapter_path, default=1)
        print(
            f"[sample][ic_lora_baseline] official adapter {OFFICIAL_IC_LORA_BASELINE_REPO} "
            f"reference_downscale_factor={ref_downscale} (read from safetensors metadata, Pitfall 4b); "
            "loaded through ICLoraPipeline(loras=[...]) — NOT signet's PEFT loader (format-mismatch guard)."
        )

        # Read the latest committed seg-map reference videos from the checkpoints Volume.
        checkpoints_vol.reload()

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sf_root = CHECKPOINTS_DIR / config.output_dir / "samples_ic_lora_baseline" / ts
        seg_dir = sf_root / "reference"
        video_dir = sf_root / "videos"
        for d in (seg_dir, video_dir):
            d.mkdir(parents=True, exist_ok=True)

        import shutil  # noqa: PLC0415 — surface the seg-map reference into the grid folder
        from pathlib import Path  # noqa: PLC0415 — clip-name stem from the Volume-relative seg path

        v = config.validation
        rows: list[dict] = []
        for idx, prompt in enumerate(prompts):
            seg_rel = reference_videos[idx] if idx < len(reference_videos) else None
            if seg_rel is None:
                continue
            seg_src = CHECKPOINTS_DIR / seg_rel
            row: dict = {
                "row": chr(ord("A") + idx),
                "row_type": "known-good baseline (official Union-Control)",
                "clip_name": Path(seg_rel).stem,
                "prompt": prompt,
                "seed": seed,
                "steps": v.num_inference_steps,
                # PASS = the official reference demonstrably steers the output; the ported ICLoraPipeline
                # path renders without OOM (the tier-2 yardstick, before we trust it for OUR adapter).
                "criteria_line": (
                    "known-good yardstick: the official Union-Control reference must visibly steer the "
                    "output (validates the ported ICLoraPipeline BEFORE we train ours)."
                ),
            }

            # The official-adapter baseline output: reference enters via video_conditioning; the pipeline
            # loads its own distilled base + upscaler + Gemma (OffloadMode.CPU). seed 42 (house A/B rule).
            reskin_video, _audio = run_ic_lora_baseline(
                prompt=prompt,
                seed=seed,
                height=v.height,
                width=v.width,
                num_frames=v.frame_count,
                frame_rate=fps,
                distilled_checkpoint_path=distilled_ckpt_path,
                spatial_upsampler_path=upscaler_ckpt_path,
                gemma_root=gemma_root,
                official_adapter_path=official_adapter_path,
                reference_video_path=str(seg_src),
                reference_downscale_factor=ref_downscale,
                device=device,
            )
            reskin_name = f"{slug(prompt)}_baseline_s{seed}.mp4"
            save_video(reskin_video, str(video_dir / reskin_name), fps=fps)
            row["ic_lora_mp4"] = f"videos/{reskin_name}"

            # Surface the seg-map reference the official adapter sees (col: what steers the output).
            if seg_src.exists():
                seg_name = f"{slug(prompt)}_seg.mp4"
                shutil.copyfile(str(seg_src), str(seg_dir / seg_name))
                row["seg_map_mp4"] = f"reference/{seg_name}"

            rows.append(row)

        index_path = write_reskin_gallery(
            rows,
            sf_root / "index.html",
            params={
                "steps": v.num_inference_steps,
                "guidance": v.guidance_scale,
                "stg_scale": v.stg_scale,
                "width": v.width,
                "height": v.height,
                "frames": v.frame_count,
                "lora_scale": 1.0,
                "conditioning_mode": "ic_lora_baseline",
                "reference_downscale_factor": ref_downscale,
            },
        )

        # Commit-or-vanish: without this the samples_ic_lora_baseline/ dir vanishes on container exit.
        checkpoints_vol.commit()
        print(
            f"[sample][ic_lora_baseline] wrote {len(rows)} known-good baseline rows -> {index_path} "
            "(committed to signe-trainer-checkpoints, commit-or-vanish)."
        )
        return

    # ── (3e) IC-LoRA V2V re-skin branch (REF-03 / D-7-RESKIN; SC#3) ───────────────────────────────
    # For ``conditioning.mode == "ic_lora"`` render the D-7-GRIDCOL 4-column re-skin story per grid
    # row on the trained IC-LoRA adapter (or the base transformer at wiring time) at seed 42 — via the
    # V2V ``run_reference_video_sampler`` (full CLEAN seg-map reference prefix, NOT a single first-frame
    # index) — and write the ``write_reskin_gallery`` PASS/FAIL grid. Mirrors the multi_frame branch:
    # the two-phase VRAM PHASE-A cached-embeddings load is reused VERBATIM (``components.text_encoder``
    # is None here; the render loop never touches Gemma), the per-row config VIEW is rebuilt, and the
    # save_video + gallery-write + ``vol.commit()`` tail is reused. The single load-bearing swaps vs
    # multi_frame: ``run_multi_condition_sampler`` -> ``run_reference_video_sampler`` and
    # ``write_multi_frame_gallery`` -> ``write_reskin_gallery``. The GPU render is proven by the
    # SEPARATELY-GATED 07-11 run (which populates the real held-out demo / external rows); this branch
    # wires the seam (row plan + per-row view + the two swapped calls). NO offloader, NO keep_warm (D-10).
    if config.conditioning.mode == "ic_lora":
        import shutil  # noqa: PLC0415 — surface the seg-map reference video into the grid folder
        from pathlib import Path  # noqa: PLC0415 — clip-name stem from the Volume-relative seg path

        from signet_trainer.inference.grid import write_reskin_gallery  # noqa: PLC0415
        from signet_trainer.inference.reference_video import (  # noqa: PLC0415
            run_reference_video_sampler,
        )

        # The paired seg-map REFERENCE videos that condition the re-skin grid, paired POSITIONALLY
        # with the re-skin prompts (mirrors the single_frame ``reference_images`` <-> prompts pairing,
        # D-GRID). Volume-relative strings (schema: carried as DATA, resolved Modal-side, Pitfall 1);
        # ``run_reference_video_sampler`` re-validates each via ``validate_volume_relative_path``.
        reference_videos = list(config.conditioning.reference_images)
        if not reference_videos:
            raise RuntimeError(
                "[sample] conditioning.mode == 'ic_lora' but conditioning.reference_images is empty "
                "— nothing to condition the re-skin grid on. Stage the seg-map reference videos "
                "(07-06/07-11) and list them Volume-relative under conditioning.reference_images "
                "(paired positionally with validation.prompts)."
            )

        # 07-15 GAP-2: the col-1 ORIGINAL (photoreal) clips, paired POSITIONALLY with the prompts
        # (config-driven, D-NOHARDCODE). Volume-relative strings resolved under CHECKPOINTS_DIR at
        # copy time (mirrors the col-2 seg copy). Empty/absent -> col-1 falls back to 'original not
        # staged' (schema default []). Staged by scripts/_stage_original_references.py.
        original_videos = list(config.conditioning.original_videos)

        # Cached embeddings come from the PHASE-A pre-encode (two-phase load above, 06-09 OOM fix,
        # generalized to all single-stage modes) — ``components.text_encoder`` is None in this mode and
        # the render loop never touches Gemma. Defensive: the mode gate guarantees PHASE A ran (a
        # two_stage config would skip it -> cached_by_prompt None -> fail fast before any 22B render).
        if cached_by_prompt is None:
            raise RuntimeError(
                "[sample][ic_lora] PHASE-A cached embeddings missing — the two-phase load gate (2b) "
                "must run for conditioning.mode == 'ic_lora' (set validation.two_stage_upscale: false; "
                "the re-skin path is single-stage — the doubled ref+target sequence + Gemma would OOM "
                "the A100-80GB, 06-09 runs 3-4)."
            )

        # Read the latest committed seg-map reference videos AND the trained adapter from the Volume.
        checkpoints_vol.reload()

        # CR-02: LOAD THE TRAINED IC-LoRA ADAPTER before the row loop. Without this the re-skin
        # column (col 3) and the no-ref control (col 4) both render on the RAW BASE transformer while
        # the grid is labeled "IC-LoRA re-skin" and the banner claims lora_scale 1.0 — so an SC#3
        # "PASS" would validate the base model's reference-following, not the trained adapter (a wrong
        # verdict from a metered run). Mirror the base-vs-LoRA tail (fns.py §5): resolve the highest-
        # step checkpoint via find_latest() (the CheckpointManager writes per-step dirs, NOT a flat
        # file), then PEFT-wrap the transformer. BOTH the reference render and the no-ref control run
        # through this SAME trained adapter — the SC#3 divergence signal is ref-vs-no-ref, both on the
        # adapter (D-7-GRIDCOL). If no adapter exists yet (wiring mode), fall back to the base and say
        # so loudly, and set the banner's lora_scale honestly.
        from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415 — import-light

        ckpt_root = CHECKPOINTS_DIR / config.output_dir
        latest_ckpt = CheckpointManager(ckpt_root).find_latest()
        adapter_path = (latest_ckpt / ADAPTER_FILENAME) if latest_ckpt is not None else (
            ckpt_root / ADAPTER_FILENAME
        )
        if adapter_path.exists():
            reskin_model = load_lora_onto_transformer(
                components.transformer, adapter_path, lora_scale=1.0
            )
            reskin_lora_scale: float | str = 1.0
            print(
                f"[sample][ic_lora] loaded trained adapter {adapter_path.name} onto the transformer "
                "— col-3 (re-skin) AND col-4 (no-ref control) render WITH the adapter."
            )
            # D-7-FREEZE note: the saved checkpoint carries ONLY the trainable "default" adapter
            # (save_adapter excludes the frozen one). Frozen stacking is NOT re-applied at sample
            # time here — the rendered adapter is the trained "default" alone. If a frozen adapter
            # was stacked during training, the sampled deltas will NOT include the frozen base; that
            # re-stacking is deferred (a future wave), so flag it rather than silently diverge.
            if config.conditioning.frozen_adapter_path:
                print(
                    "[sample][ic_lora] WARNING: a frozen adapter was configured for training "
                    f"({config.conditioning.frozen_adapter_path}) but is NOT re-stacked at sample "
                    "time — col-3/col-4 render the trained 'default' adapter alone (D-7-FREEZE)."
                )
        else:
            reskin_model = components.transformer
            reskin_lora_scale = "none"
            print(
                f"[sample][ic_lora] WARNING: no trained adapter at {adapter_path} — col-3/col-4 "
                "render on the BASE transformer (wiring mode). An SC#3 verdict would validate the "
                "BASE model's reference-following, NOT a trained adapter. Run train first."
            )

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sf_root = CHECKPOINTS_DIR / config.output_dir / "samples_ic_lora" / ts
        seg_dir = sf_root / "reference"
        video_dir = sf_root / "videos"
        for d in (seg_dir, video_dir):
            d.mkdir(parents=True, exist_ok=True)

        v = config.validation
        rows: list[dict] = []
        for idx, prompt in enumerate(prompts):
            seg_rel = reference_videos[idx] if idx < len(reference_videos) else None
            cached = cached_by_prompt[prompt]
            # Per-row config VIEW: pin validation.prompts to this single row's prompt (mirrors the
            # multi_frame per-column view rebuild) so ``run_reference_video_sampler`` /
            # ``build_generation_config`` read exactly this prompt. deep copy — never mutate the shared
            # config (D-6-STRENGTHCOL discipline).
            row_config = config.model_copy(deep=True)
            row_config.validation.prompts = [prompt]

            # D-7-GRIDROW: Row A = in-domain held-out demo, Row B = generalization (external). 07-11
            # supplies the real clips; the row/clip labels are structural here.
            row: dict = {
                "row": chr(ord("A") + idx),
                "row_type": "in-domain (held-out demo)" if idx == 0 else "generalization (external)",
                "clip_name": Path(seg_rel).stem if seg_rel else "",
                "prompt": prompt,
                "seed": seed,
                "steps": v.num_inference_steps,
            }

            if seg_rel is not None:
                # col 3 — IC-LoRA re-skin: condition on the FULL clean seg-map reference prefix at
                # seed 42 (the SAME seed as the col-4 control — house A/B rule). The V2V sampler
                # VAE-encodes the whole seg-map clip to a reference latent and prepends it.
                reskin_video, _audio = run_reference_video_sampler(
                    components,
                    reskin_model,  # CR-02: the TRAINED adapter (or base in wiring mode), NOT raw base.
                    row_config,
                    seg_rel,
                    device=device,
                    cached_embeddings=cached,
                )
                reskin_name = f"{slug(prompt)}_reskin_s{seed}.mp4"
                save_video(reskin_video, str(video_dir / reskin_name), fps=fps)
                row["ic_lora_mp4"] = f"videos/{reskin_name}"

                # col 2 — surface the seg-map reference video into the grid folder (what the adapter
                # sees). Volume-relative ``seg_rel`` resolves under CHECKPOINTS_DIR for the copy.
                seg_src = CHECKPOINTS_DIR / seg_rel
                if seg_src.exists():
                    seg_name = f"{slug(prompt)}_seg.mp4"
                    shutil.copyfile(str(seg_src), str(seg_dir / seg_name))
                    row["seg_map_mp4"] = f"reference/{seg_name}"

            # col 1 — original footage: copy the config-driven, positionally-paired ORIGINAL clip into
            # the grid videos/ folder (mirrors the col-2 seg copy). Volume-relative under CHECKPOINTS_DIR
            # (07-15 GAP-2 / D-NOHARDCODE). Empty list or missing entry -> row["original_mp4"] stays
            # unset -> grid col-1 falls back to 'original not staged'.
            orig_rel = original_videos[idx] if idx < len(original_videos) else None
            if orig_rel is not None:
                orig_src = CHECKPOINTS_DIR / orig_rel
                if orig_src.exists():
                    original_name = f"{slug(prompt)}_original.mp4"
                    shutil.copyfile(str(orig_src), str(video_dir / original_name))
                    row["original_mp4"] = f"videos/{original_name}"
                else:
                    print(
                        f"[sample][ic_lora] WARNING: original_videos[{idx}] {orig_rel!r} not found "
                        f"under CHECKPOINTS_DIR — col-1 falls back to 'original not staged'."
                    )

            # col 4 — no-reference control: SAME prompt + seed, NO reference prefix, but STILL through
            # the trained adapter (CR-02): the SC#3 divergence signal is ref-vs-no-ref, BOTH on the
            # adapter — a base-model control would confound "adapter effect" with "reference effect".
            control_cfg = build_generation_config(row_config, prompt=prompt, seed=seed)
            control_video, _audio = run_sampler(
                components,
                reskin_model,  # CR-02: same trained adapter as col-3 (or base in wiring mode).
                control_cfg,
                device=device,
                cached_embeddings=cached,
            )
            control_name = f"{slug(prompt)}_noref_s{seed}.mp4"
            save_video(control_video, str(video_dir / control_name), fps=fps)
            row["base_no_ref_mp4"] = f"videos/{control_name}"

            rows.append(row)

        index_path = write_reskin_gallery(
            rows,
            sf_root / "index.html",
            params={
                "steps": v.num_inference_steps,
                "guidance": v.guidance_scale,
                "stg_scale": v.stg_scale,
                "width": v.width,
                "height": v.height,
                "frames": v.frame_count,
                # CR-02: reflect REALITY — 1.0 when the trained adapter is applied, "none" when the
                # branch fell back to the base transformer (wiring mode). Never claim 1.0 with no LoRA.
                "lora_scale": reskin_lora_scale,
                # D-7 re-skin banner extras (UI-SPEC line 108) — config-driven, never hardcoded.
                "conditioning_mode": config.conditioning.mode,
                "reference_downscale_factor": config.conditioning.reference_downscale_factor,
            },
        )

        # Commit-or-vanish: without this the samples_ic_lora/ dir vanishes on container exit.
        checkpoints_vol.commit()
        print(
            f"[sample][ic_lora] wrote {len(rows)} re-skin rows -> {index_path} "
            "(committed to signe-trainer-checkpoints, commit-or-vanish)."
        )
        return

    # Output layout (RESEARCH A4): /checkpoints/<output_dir>/samples/<ts>/{base,lora}/*.mp4 + index.html.
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    samples_root = CHECKPOINTS_DIR / config.output_dir / "samples" / ts
    base_dir = samples_root / "base"
    lora_dir = samples_root / "lora"
    base_dir.mkdir(parents=True, exist_ok=True)

    # ── (3b) render-path dispatch — D-DEPTH-1 two-stage upscaler toggle (DEFAULT OFF) ────────────
    # Single-stage ``run_sampler`` stays the grid default (correctness-first). When
    # ``config.validation.two_stage_upscale`` is True, route BOTH columns through the distilled-base
    # + 2x spatial-upscaler ``run_two_stage`` path instead (import + weights gated behind the toggle;
    # 04-05 Task 1). ``_render`` keeps the branch structural so the grid emission below is identical
    # whichever path runs. Pitfall 6: the two-stage stack can exceed A100-80GB — a LIVE two-stage run
    # is a SEPARATE gated invocation scoped to H100 + CPU offload (D-RUN-1); this wave only wires it.
    # (``two_stage`` was read up-front at (2b) — it decides whether the two-phase pre-encode ran.)
    if two_stage:
        from signet_trainer.inference.upscale import run_two_stage  # noqa: PLC0415

        distilled_ckpt_path = str(WEIGHTS_DIR / "ltx-2.3-22b-distilled-1.1.safetensors")
        upscaler_ckpt_path = str(WEIGHTS_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        distilled_lora_path = str(WEIGHTS_DIR / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
        print("[sample] two_stage_upscale ON — routing both columns through the 2x spatial upscaler.")

    def _render(model: object, cfg: object, cached: object = None) -> object:
        """Dispatch one render to the single-stage sampler or the two-stage upscaler (toggle).

        ``cached`` is the PHASE-A cached embeddings for this render's prompt (two-phase load,
        06-09 carry-forward) — threaded into the single-stage ``run_sampler`` so the base-vs-LoRA
        columns render with Gemma already deleted. It is ``None`` on the two-stage path (which
        keeps its legacy Gemma-loaded flow, out of scope for the two-phase port).
        """
        if two_stage:
            return run_two_stage(
                components,
                model,
                cfg,
                distilled_ckpt_path=distilled_ckpt_path,
                upscaler_ckpt_path=upscaler_ckpt_path,
                distilled_lora_path=distilled_lora_path,
                device=device,
            )
        return run_sampler(components, model, cfg, device=device, cached_embeddings=cached)

    # ── (4) BASE column — raw transformer, seed 42, all prompts ─────────────────────────────────
    base_mp4s: dict[str, str] = {}
    for prompt in prompts:
        cfg = build_generation_config(config, prompt=prompt, seed=seed)
        cached = cached_by_prompt[prompt] if cached_by_prompt is not None else None
        video, _audio = _render(components.transformer, cfg, cached)
        fname = f"{slug(prompt)}_s{seed}.mp4"
        save_video(video, str(base_dir / fname), fps=fps)
        base_mp4s[prompt] = f"base/{fname}"  # relative to index.html (portable folder).
    print(f"[sample] BASE column: {len(base_mp4s)} clips at seed {seed}.")

    # ── (5) LoRA column — ONE 22B load, sequential (Open-Q1 (a)): PEFT-wrap the SAME transformer ─
    # Reload the checkpoints Volume first so we read the latest committed Phase-3 adapter.
    checkpoints_vol.reload()
    # The CheckpointManager writes adapters into per-step dirs
    # (output_dir/checkpoint-step-NNNNN-loss-X/adapter_model.safetensors), NOT a flat file.
    # Resolve the highest-step checkpoint via the same helper resume() uses; fall back to a flat
    # adapter path for compatibility.
    from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415 — local, import-light

    ckpt_root = CHECKPOINTS_DIR / config.output_dir
    latest_ckpt = CheckpointManager(ckpt_root).find_latest()
    adapter_path = (latest_ckpt / ADAPTER_FILENAME) if latest_ckpt is not None else (
        ckpt_root / ADAPTER_FILENAME
    )
    lora_mp4s: dict[str, str] = {}
    if adapter_path.exists():
        lora_dir.mkdir(parents=True, exist_ok=True)
        lora_model = load_lora_onto_transformer(components.transformer, adapter_path, lora_scale=1.0)
        tail_lora_scale: float | str = 1.0
        for prompt in prompts:
            cfg = build_generation_config(config, prompt=prompt, seed=seed)
            cached = cached_by_prompt[prompt] if cached_by_prompt is not None else None
            video, _audio = _render(lora_model, cfg, cached)
            fname = f"{slug(prompt)}_s{seed}.mp4"
            save_video(video, str(lora_dir / fname), fps=fps)
            lora_mp4s[prompt] = f"lora/{fname}"
        print(f"[sample] LoRA column: {len(lora_mp4s)} clips from adapter {adapter_path.name}.")
    else:
        tail_lora_scale = "none"
        print(
            f"[sample] no adapter at {adapter_path} — rendering a BASE-only grid (fallback). "
            "Run train first to produce the adapter for the LoRA column."
        )

    # ── (6) base-vs-LoRA HTML montage + commit-or-vanish ────────────────────────────────────────
    v = config.validation
    rows = [
        {
            "prompt": p,
            "seed": seed,
            "base_mp4": base_mp4s.get(p, ""),
            "lora_mp4": lora_mp4s.get(p, ""),
        }
        for p in prompts
    ]
    index_path = write_comparison_gallery(
        rows,
        samples_root / "index.html",
        params={
            "steps": v.num_inference_steps,
            "guidance": v.guidance_scale,
            "stg_scale": v.stg_scale,
            "width": v.width,
            "height": v.height,
            "frames": v.frame_count,
            # CR-02: 1.0 only when the trained adapter rendered the LoRA column; "none" on the
            # BASE-only fallback (no adapter) so the banner never claims a LoRA scale with no LoRA.
            "lora_scale": tail_lora_scale,
        },
    )

    # Commit-or-vanish: without this the samples/ dir (mp4s + index.html) is lost on container exit.
    checkpoints_vol.commit()
    print(
        f"[sample] wrote {len(rows)} comparison rows -> {index_path} "
        "(committed to signe-trainer-checkpoints, commit-or-vanish)."
    )


# --------------------------------------------------------------------------------------------------
# Phase-2 one-time gated weight download (D-06-W / D-07-W). CPU — pure HF download, NO gpu= and NO
# ltx deps (uses the default code-only image + huggingface_hub). Downloads the LTX-2.3 22b dev
# checkpoint + Gemma 3 12B by DEFAULT; the three two-stage distilled/upscaler weights only when the
# ``two_stage`` flag is set (D-DEPTH-1, 04-05 — default OFF so single-stage never fetches them).
# Write-then-commit (Pitfall 3) mirrors volume_roundtrip_probe.
# --------------------------------------------------------------------------------------------------

# T-04-13 (04-05): the three extra weights the two-stage spatial-upscaler toggle needs, from the
# OFFICIAL public ``Lightricks/LTX-2.3`` repo (same repo/secret as the dev checkpoint — .safetensors
# only, no pickle). Fetched ONLY when download_weights(two_stage=True); the single-stage default
# never touches them (D-DEPTH-1). [VERIFIED: LTX-2 README @ pinned SHA d6053703 — RESEARCH line 258.]
TWO_STAGE_WEIGHT_FILENAMES = (
    "ltx-2.3-22b-distilled-1.1.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
)


@app.function(
    image=download_image,  # code-only image + huggingface_hub (the default App image lacks it).
    volumes={**WEIGHTS_MOUNT},  # CPU — intentionally NO gpu= (pure HF download, RESEARCH.md line 61).
    secrets=[huggingface_secret],  # gated HF downloads need the token (Modal secret only; never log).
    timeout=TWENTY_FOUR_HOURS,  # the 46GB dev checkpoint download can be slow.
)
def download_weights(two_stage: bool = False) -> str:
    """One-time gated download of LTX-2.3 22b dev + Gemma 3 12B into signe-trainer-weights.

    Downloads by DEFAULT (D-07-W):
        * ``Lightricks/LTX-2.3``  ``ltx-2.3-22b-dev.safetensors``  -> WEIGHTS_DIR
        * ``google/gemma-3-12b-it`` (full snapshot)               -> WEIGHTS_DIR/gemma-3-12b-it

    When ``two_stage=True`` (D-DEPTH-1 upscaler toggle, 04-05 — default OFF) ALSO downloads the three
    distilled/spatial-upscaler weights from the SAME public ``Lightricks/LTX-2.3`` repo (the existing
    HF secret suffices — A5): ``ltx-2.3-22b-distilled-1.1.safetensors``,
    ``ltx-2.3-spatial-upscaler-x2-1.1.safetensors``, ``ltx-2.3-22b-distilled-lora-384-1.1.safetensors``.
    Single-stage runs pass ``two_stage=False`` (the default) and fetch nothing extra.

    Then ``weights_vol.commit()`` (Pitfall 3) so the files survive the container + are visible to a
    separate ``modal volume ls signe-trainer-weights``. Verifies the ~46GB dev-checkpoint size.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # LTX-2.3 22b dev checkpoint (license-accepted via the HF secret token; T-02-WP provenance).
    ckpt_path = hf_hub_download(
        repo_id="Lightricks/LTX-2.3",
        filename="ltx-2.3-22b-dev.safetensors",
        local_dir=str(WEIGHTS_DIR),
    )

    # Gemma 3 12B text encoder (A1: the -it variant, matching enochiatron's validated loader).
    gemma_dir = snapshot_download(
        repo_id="google/gemma-3-12b-it",
        local_dir=str(WEIGHTS_DIR / "gemma-3-12b-it"),
    )

    # D-DEPTH-1 two-stage toggle (default OFF): fetch the distilled/upscaler .safetensors ONLY when
    # asked. Same repo + secret as the dev checkpoint (public, T-04-13 — .safetensors, no pickle).
    two_stage_paths: list[str] = []
    if two_stage:
        for filename in TWO_STAGE_WEIGHT_FILENAMES:
            two_stage_paths.append(
                hf_hub_download(
                    repo_id="Lightricks/LTX-2.3",
                    filename=filename,
                    local_dir=str(WEIGHTS_DIR),
                )
            )

    # Pitfall 3 commit-or-vanish (mirrors volume_roundtrip_probe).
    weights_vol.commit()

    # T-02-WP: verify the dev checkpoint size (~46.1GB) post-download as a cheap provenance check.
    import os

    ckpt_gb = os.path.getsize(ckpt_path) / (1024**3)
    two_stage_note = (
        f"; two-stage weights: {len(two_stage_paths)} fetched ({', '.join(TWO_STAGE_WEIGHT_FILENAMES)})"
        if two_stage
        else " (dev + Gemma only, D-07-W — two-stage OFF)"
    )
    return (
        f"[download_weights] LTX-2.3 dev checkpoint: {ckpt_path} ({ckpt_gb:.1f} GB); "
        f"Gemma 3 12B: {gemma_dir}; committed to signe-trainer-weights{two_stage_note}."
    )


@app.function(
    image=download_image,  # code-only image + huggingface_hub, same as download_weights.
    volumes={**WEIGHTS_MOUNT},  # CPU only — a pure HF download, never a GPU.
    secrets=[huggingface_secret],
    timeout=TWENTY_FOUR_HOURS,  # ~54 GB across three components.
)
def download_qwen_edit_weights(repo_id: str = "Qwen/Qwen-Image-Edit-2511") -> str:
    """One-time download of the Qwen-Image-Edit-2511 diffusers snapshot into the weights Volume.

    The ``qwen_edit`` family sibling of :func:`download_weights`. It fetches the DIFFUSERS
    DIRECTORY layout rather than the single ComfyUI-style ``.safetensors`` files, because that is
    the shape the loader can consume without a hand-supplied config:
    ``load_qwen_edit_transformer`` sniffs the path the way ai-toolkit does (``qwen_image.py:96``),
    and on the single-file branch ``from_single_file`` calls ``fetch_diffusers_config`` ->
    ``infer_diffusers_model_type``, **which has no Qwen branch on the pinned diffusers** — so a
    bare ``.safetensors`` REQUIRES an explicit ``config_source``. A snapshot ships its own
    ``config.json`` beside each component, which removes that failure mode entirely.

    Layout written (the three ``model.*_id`` fields address these as subdirectories of
    ``WEIGHTS_DIR``, since ``WEIGHTS_DIR / model_id`` is how every family resolves its weights)::

        <WEIGHTS_DIR>/qwen-image-edit-2511/transformer     -> model.model_id
        <WEIGHTS_DIR>/qwen-image-edit-2511/vae             -> model.vae_id
        <WEIGHTS_DIR>/qwen-image-edit-2511/text_encoder    -> model.text_encoder_id

    The text encoder is **Qwen2.5-VL, and the vision half is load-bearing**. A text-only encoder of
    the same family loads clean, embeds at the right rank, and dies deep in the forward with an
    unattributed matmul shape error. ``assert_qwen_edit_text_encoder_vision`` is the gate for that;
    this function is only its supplier. Do not substitute a text-only Qwen checkpoint here.

    Allowing ``repo_id`` to be overridden is deliberate: ``Qwen/Qwen-Image`` (the T2I base) is the
    PRIME stage of a chained edit run and is fetched by the same code path.
    """
    from huggingface_hub import snapshot_download

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    local_name = repo_id.rsplit("/", 1)[-1].lower()
    root = WEIGHTS_DIR / local_name

    # allow_patterns keeps the .safetensors + configs and skips the duplicate .bin / .gguf mirrors
    # some Qwen repos carry — those would roughly double the transfer for nothing.
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(root),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.safetensors.index.json", "*.py"],
        ignore_patterns=["*.bin", "*.gguf", "*.onnx", "*.msgpack", "*.h5"],
    )

    weights_vol.commit()  # Pitfall 3 commit-or-vanish, same as download_weights.

    import os

    sizes: dict[str, float] = {}
    for component in ("transformer", "vae", "text_encoder"):
        comp_dir = root / component
        if comp_dir.is_dir():
            total = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, files in os.walk(comp_dir)
                for f in files
            )
            sizes[component] = total / (1024**3)

    summary = ", ".join(f"{k} {v:.1f} GB" for k, v in sizes.items()) or "NO COMPONENT DIRS FOUND"
    return (
        f"[download_qwen_edit_weights] {repo_id} -> {root} ({summary}); "
        f"committed. Set model_id/vae_id/text_encoder_id to "
        f"'{local_name}/transformer', '{local_name}/vae', '{local_name}/text_encoder'."
    )


# --------------------------------------------------------------------------------------------------
# Phase-2 GPU loader smoke (SC#1 / D-04). Loads the LTX-2.3 components via the ported
# load_ltxv_components, prints the transformer/VAE/Gemma/scheduler shapes, ASSERTS them against the
# EXPECTED_* ground-truth constants, then exits. Uses the pinned-SHA gpu_image (ltx-core/ltx-trainer).
# --------------------------------------------------------------------------------------------------


@app.function(
    gpu="A100-80GB",  # the 22B load needs VRAM (RESEARCH.md A2).
    image=gpu_image,
    volumes={**WEIGHTS_MOUNT},
    secrets=[huggingface_secret],
    timeout=30 * 60,
)
def load_ltxv_smoke() -> str:
    """D-04 metered proof: load components, print + assert shapes vs EXPECTED_* constants, exit (SC#1).

    Loads transformer/VAE/Gemma/scheduler via the ported ``load_ltxv_components`` (video-only),
    summarizes the arch facts, ASSERTS block-count/hidden-dim/in-channels against the single-source
    ``EXPECTED_*`` constants in ``models/loader.py``, prints the summary, and returns it. Exits 0 on
    a clean load + match; raises (non-zero) on any mismatch — the cheapest gated proof the loader
    works on the real GPU before any training spend.
    """
    from signet_trainer.models.loader import (
        EXPECTED_HIDDEN_DIM,
        EXPECTED_NUM_BLOCKS,
        EXPECTED_T2V_IN_CHANNELS,
        load_ltxv_components,
        summarize_components,
    )

    components = load_ltxv_components(
        checkpoint_path=str(WEIGHTS_DIR / "ltx-2.3-22b-dev.safetensors"),
        text_encoder_path=str(WEIGHTS_DIR / "gemma-3-12b-it"),
        device="cuda",
    )
    summary = summarize_components(components)

    print(f"[load_ltxv_smoke] component summary: {summary}")

    # Assert only on the facts we have ground truth for (summarize_components tolerates attr drift).
    if summary["num_blocks"] is not None:
        assert summary["num_blocks"] == EXPECTED_NUM_BLOCKS, (
            f"transformer block count {summary['num_blocks']} != expected {EXPECTED_NUM_BLOCKS}"
        )
    if summary["hidden_dim"] is not None:
        assert summary["hidden_dim"] == EXPECTED_HIDDEN_DIM, (
            f"hidden_dim {summary['hidden_dim']} != expected {EXPECTED_HIDDEN_DIM}"
        )
    if summary["in_channels"] is not None:
        assert summary["in_channels"] == EXPECTED_T2V_IN_CHANNELS, (
            f"in_channels {summary['in_channels']} != expected {EXPECTED_T2V_IN_CHANNELS}"
        )
    assert summary["has_video_vae_encoder"], "video_vae_encoder missing (needed for pre-encoding)"
    assert summary["has_text_encoder"], "text_encoder (Gemma) missing"
    assert summary["has_scheduler"], "scheduler (LTX2Scheduler) missing"

    return (
        f"[load_ltxv_smoke] OK — blocks={summary['num_blocks']} hidden_dim={summary['hidden_dim']} "
        f"in_channels={summary['in_channels']} (vs EXPECTED "
        f"{EXPECTED_NUM_BLOCKS}/{EXPECTED_HIDDEN_DIM}/{EXPECTED_T2V_IN_CHANNELS})"
    )


# --------------------------------------------------------------------------------------------------
# Phase-1 deliverable: the CPU Volume round-trip probe (SC#3 / MODL-01).
# NO gpu= -> CPU container -> zero GPU spend (Discretion / Open-Q3). Writes a sentinel under the
# weights mount and commits (Pitfall 3) so `modal volume ls signe-trainer-weights` proves durability.
# --------------------------------------------------------------------------------------------------


@app.function(
    volumes=WEIGHTS_MOUNT,  # CPU — intentionally NO gpu=.
    timeout=10 * 60,
)
def volume_roundtrip_probe() -> str:
    """Write a sentinel under the weights Volume mount, commit, and return its path (SC#3).

    Proves Volume durability + the weights-from-Volume pattern (MODL-01) on a CPU container with
    zero GPU spend. The committed file survives the invocation and is visible to a separate
    ``modal volume ls signe-trainer-weights``.
    """
    import datetime  # stdlib-only; runs inside the Modal container.

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = WEIGHTS_DIR / "signet_roundtrip_sentinel.txt"
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sentinel.write_text(
        f"signet-trainer volume round-trip probe OK @ {stamp}\n"
        "Proves: file written under /weights survives commit (Pitfall 3) + weights mount "
        "from the signe-trainer-weights Volume, not the image (MODL-01).\n",
        encoding="utf-8",
    )

    # REQUIRED (Pitfall 3 commit-or-vanish): without commit() the write is lost when the
    # container exits, and `modal volume ls` would show nothing.
    weights_vol.commit()

    return str(sentinel)


# Phase-2 note (MODL-01): inside the stage functions, base weights are read as
# ``WEIGHTS_DIR / model_id`` from the mounted weights Volume — never bundled into the Image.
# ``DATASET_DIR`` / ``CHECKPOINTS_DIR`` / ``checkpoints_vol`` are re-exported here so the stage
# bodies (Phases 2-4) can read/write + ``commit()`` checkpoints without re-importing from app.py.
_PHASE2_REEXPORTS = (DATASET_DIR, CHECKPOINTS_DIR, checkpoints_vol)


# ── Phase-9 (INPAINT) — the In-Outpainting scaffold fuse job (GATE-SPEC rev 2, build-order 3) ──
# The gated HF repo (Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting) needs a token that ACCEPTED the
# repo license — the prior project's dedicated ``hf-gated-secret`` ([precedent], verified live in this
# workspace, last used 07-02). Name is env-overridable at import (same WR-01 capture semantics as
# app.py's SIGNET_*_SECRET_NAME seams: export BEFORE ``modal run``).
import os as _os  # noqa: E402  (module-tail import: keeps the Phase-1 header block byte-stable)

HF_GATED_SECRET_NAME = _os.environ.get("SIGNET_HF_GATED_SECRET_NAME", "hf-gated-secret")
hf_gated_secret = modal.Secret.from_name(HF_GATED_SECRET_NAME)

# Fuse timeout: the prior project's fuse ran well under an hour on big-RAM CPU; 4h is a generous ceiling
# (never 24h — a hung 44GB rewrite should die, not idle a paid container).
FOUR_HOURS = 4 * 60 * 60


@app.function(
    # CPU-ONLY on purpose (NO gpu=): the fuse is pure tensor arithmetic over safetensors dicts.
    image=gpu_image,  # torch + ltx_core at the pinned SHA (apply_loras/StateDict live there).
    memory=131072,  # [precedent] prior-project: apply_loras materializes a ~2x 44GB dict — 128GiB is load-bearing.
    timeout=FOUR_HOURS,
    volumes={**WEIGHTS_MOUNT},
    secrets=[huggingface_secret, hf_gated_secret],
)
def fuse(config_yaml: str) -> None:
    """Gated CPU fuse: In-Outpainting IC-LoRA -> dev base -> fused base on the weights Volume.

    Body mirrors the stage-fn discipline (heavy imports function-local, Anti-Pattern 6): (1)
    cold-path probe, (2) revalidate the config by value (the gate's convention — the fuse reads
    only cfg for provenance logging; filenames are the fuse module's pipeline constants), (3)
    header-only keycheck (cheap, catches a wrong adapter before the 44GB load), (4) the
    config-preserving fuse ([precedent] prior-project PATCHED _FUSE_SCRIPT — meta['config'] rides into
    the save; embed_config is only the repair tool), (5) verify_fused_metadata as the built-in
    post-save gate (BUGLOG #5 config-loss + #6 audio-strip guards), (6) ``weights_vol.commit()``
    (commit-or-vanish).
    """
    try:
        import safetensors  # noqa: PLC0415, F401
        from ltx_core.loader import apply_loras  # noqa: PLC0415, F401
    except ImportError as exc:
        raise RuntimeError(
            f"[fuse] cold-path dependency missing ({exc.name!r}) — gpu_image must carry "
            "safetensors + ltx-core (apply_loras/StateDict) at the pinned SHA. Fix the image "
            "before re-dispatching (T-03-SC)."
        ) from exc

    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415
    from signet_trainer.modal.fuse import (  # noqa: PLC0415
        DEFAULT_BASE_FILENAME,
        DEFAULT_FUSED_FILENAME,
        fuse_inoutpaint,
        fuse_keycheck,
    )

    config = load_config_from_text(config_yaml)  # revalidate in-container (T-03-63 convention)
    base_path = str(WEIGHTS_DIR / DEFAULT_BASE_FILENAME)
    out_path = str(WEIGHTS_DIR / DEFAULT_FUSED_FILENAME)
    print(
        f"[fuse] base={base_path} -> fused={out_path} "
        f"(run config: {config.output_dir}; strength=1.0 [precedent])"
    )

    kc = fuse_keycheck(base_path)
    print(f"[fuse] keycheck: {kc.get('n_hits', '?')} adapter targets matched in the base header.")

    summary = fuse_inoutpaint(base_path=base_path, out_path=out_path)
    weights_vol.commit()  # commit-or-vanish — an uncommitted 44GB fuse is a wasted job.
    print(
        f"[fuse] DONE + committed — {summary.get('n_changed', '?')} tensors changed; "
        "verify_fused_metadata passed in-save (config metadata + audio weights intact). "
        f"Point the inpaint config's model.model_id at {DEFAULT_FUSED_FILENAME}."
    )


# --------------------------------------------------------------------------------------------------
# BK-01 (09.1-08): CPU-only checkpoint backup + restore — OFF the training A100 (D-BK-3).
#
# Both functions mount the CHECKPOINTS Volume, reuse the EXISTING huggingface_secret, and carry NO
# ``gpu=`` — an upload/download hang runs in its OWN CPU container and can never wedge a running
# ``train()`` (the time-gate isolation rule). ``backup_sync`` mirrors ONLY new complete checkpoints
# (via the pure ``backup/plan.py``) to the configured PRIVATE destination, 1:1 with the Volume dir
# layout so ``restore`` is a straight copy-back (D-BK-4) — with exactly ONE documented subtraction
# from the FILE set, ``backup/plan.py::MIRROR_EXCLUDED_FILES`` (D-10-DEF-18: the PEFT-autogenerated
# ``README.md``, whose front matter the Hub rejects for H3 checkpoints and which no loader reads).
# The dir LAYOUT is untouched, so copy-back restore is unaffected. ``restore`` rehydrates + commits the Volume
# (commit-or-vanish). The HF token is read from the Modal secret ONLY (huggingface_hub picks up
# HF_TOKEN from the env the secret injects) — it is NEVER read from config, NEVER printed, NEVER
# written to the repo. ``destination='cloud'`` is UNREACHABLE here: 09.1-07's config validator
# fail-fasts an enabled cloud block at load (the in-body ``load_config_from_text`` re-runs that same
# validator), so the in-fn ``NotImplementedError`` branch is defense-in-depth only. Both are ADDITIVE
# / mirror-only — neither ever deletes anything (never-auto-delete house rule).
# --------------------------------------------------------------------------------------------------


@app.function(
    image=download_image,  # code-only image + huggingface_hub (same as download_weights).
    volumes={**CHECKPOINTS_MOUNT},  # CPU — mounts the checkpoints Volume; intentionally NO gpu= (D-BK-3).
    secrets=[huggingface_secret],  # HF token from the Modal secret ONLY (never config, never logged).
    timeout=TWENTY_FOUR_HOURS,  # a large mirror can be slow; the generous fixed bound download_weights uses.
)
def backup_sync(config_text: str) -> str:
    """CPU mirror of ONLY-new complete checkpoints to cfg.backup — off the training A100 (D-BK-3).

    Reloads the checkpoints Volume, plans the idempotent upload via the pure ``backup/plan.py`` core
    (``list_complete_checkpoints`` -> ``select_for_backup`` -> ``plan_uploads``), and uploads each new
    ``checkpoint-step-*`` dir 1:1 into the destination (``{output_dir}/{dir_name}``) so ``restore`` is
    a straight copy-back. Idempotent (a second run with nothing new uploads 0 dirs) and ADDITIVE
    (never deletes at the destination). The HF token comes from the Modal secret env ONLY — never
    read from config, never printed. cloud is rejected at config load; the in-fn branch is a backstop.

    The one FILE-set subtraction is ``backup/plan.py::MIRROR_EXCLUDED_FILES`` (D-10-DEF-18) — the
    PEFT-autogenerated ``README.md``. It is applied to BOTH destinations so the contract is
    destination-independent, it is disjoint from ``REQUIRED_BACKUP_FILES`` (test-pinned), and it is an
    OMISSION from the copy, never a delete at the destination.
    """
    import os  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from signet_trainer.backup.plan import (  # noqa: PLC0415
        MIRROR_EXCLUDED_FILES,
        complete_remote_names,
        is_complete,
        list_complete_checkpoints,
        plan_uploads,
        select_for_backup,
    )
    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415

    # Revalidate in-container (T-03-63 convention) — this RE-RUNS 09.1-07's BackupConfig validator, so
    # an enabled destination='cloud' block already fail-fasted at the entrypoint gate before dispatch.
    cfg = load_config_from_text(config_text)
    backup = cfg.backup

    checkpoints_vol.reload()  # reload-before-read: see everything committed before planning.
    local_dir = CHECKPOINTS_DIR / cfg.output_dir
    complete = list_complete_checkpoints(local_dir)
    selected = select_for_backup(complete, backup.what, backup.interval)

    if backup.destination == "hf":
        from huggingface_hub import HfApi  # noqa: PLC0415 — function-local (mirrors download_weights).

        api = HfApi()
        # Create the PRIVATE repo if missing (idempotent); a pre-existing repo is untouched.
        api.create_repo(
            repo_id=backup.repo_id, repo_type="model", private=backup.private, exist_ok=True
        )
        # CR-01 privacy gate (D-BK-5 / T-03-44): ``private=`` is HONORED ONLY when create_repo
        # actually creates the repo. With ``exist_ok=True`` a PRE-EXISTING repo is a no-op — the flag
        # NEVER demotes a public repo to private. These are personal likeness LoRA checkpoints, so
        # before any upload VERIFY the destination is actually private via repo_info() and ABORT
        # (naming the repo) otherwise — never silently publish private artifacts to a public repo.
        if backup.private:
            info = api.repo_info(repo_id=backup.repo_id, repo_type="model")
            if not info.private:
                raise RuntimeError(
                    f"[backup_sync] backup.private=True but repo {backup.repo_id!r} is PUBLIC — "
                    "refusing to upload likeness checkpoints to a public repo (D-BK-5). Make the "
                    "repo private or point backup.repo_id at a fresh name."
                )
        # Already-backed-up dir names under the mirrored prefix — tolerate a not-yet-populated repo.
        # WR-01: key idempotency on the COMPLETENESS of the REMOTE copy, not the bare dir NAME. Walk
        # the tree RECURSIVELY, collect each backed-up dir's file basenames, and treat a dir as
        # backed-up ONLY when it carries BOTH required files (adapter + training_state.pt). A dir whose
        # name is present but whose upload was interrupted mid-write is NOT complete -> it is re-planned
        # and re-uploaded (repaired) rather than trusted forever.
        try:
            tree = api.list_repo_tree(
                repo_id=backup.repo_id,
                repo_type="model",
                path_in_repo=cfg.output_dir,
                recursive=True,
            )
            prefix = f"{cfg.output_dir}/"
            remote_dir_files: dict[str, set[str]] = {}
            for entry in tree:
                path = entry.path
                rel = path[len(prefix):] if path.startswith(prefix) else path
                parts = rel.split("/")
                # A required file sits DIRECTLY under a checkpoint dir: parts == [dir_name, file].
                # Register the dir either way (a bare dir entry -> empty set -> stays "incomplete").
                remote_dir_files.setdefault(parts[0], set())
                if len(parts) == 2 and parts[1]:
                    remote_dir_files[parts[0]].add(parts[1])
            remote_names = complete_remote_names(remote_dir_files)
        except Exception:  # noqa: BLE0001 — an empty/fresh repo has no tree at the prefix yet.
            remote_names = set()
        to_upload = plan_uploads(selected, remote_names)
        # D-10-DEF-18: mirror every file EXCEPT the PEFT-autogenerated model card. huggingface_hub
        # validates README front matter SERVER-SIDE before hashing a byte
        # (_prepare_upload_folder_additions -> _validate_yaml), and an H3 checkpoint's card carries
        # `base_model: /weights/minimax-h3/transformer_ref` — a local Volume path, not a Hub id — so
        # the Hub answers 400 on the FIRST dir and the WHOLE mirror aborts with ZERO uploads. The
        # excluded set is disjoint from REQUIRED_BACKUP_FILES (pinned by a test), nothing reads a
        # checkpoint README, and skipping it is an OMISSION, never a delete. Full rationale + the
        # rejected "upload a rewritten card" alternative live on the constant in backup/plan.py.
        # Checkpoint dirs are FLAT, so a bare basename pattern matches exactly the intended file.
        excluded = sorted(MIRROR_EXCLUDED_FILES)
        for d in to_upload:
            api.upload_folder(
                folder_path=str(d),
                path_in_repo=f"{cfg.output_dir}/{d.name}",  # MIRRORS the Volume dir 1:1 -> copy-back restore.
                repo_id=backup.repo_id,
                repo_type="model",
                # A FRESH list per call, deliberately — NOT the shared `excluded`. huggingface_hub's
                # upload_folder does `ignore_patterns += DEFAULT_IGNORE_PATTERNS`, an IN-PLACE mutation
                # of the CALLER's list, so passing one list across the loop appends the hub's default
                # block once PER checkpoint dir: the list grows without bound and the summary print
                # below stops describing our actual exclusion. Verified live (a 2-dir mirror printed
                # the hub defaults twice). Copying costs nothing and keeps the constant unaliased.
                ignore_patterns=list(excluded),
            )
        uploaded_names = [d.name for d in to_upload]
        print(
            f"[backup_sync][hf] mirrored {len(uploaded_names)} new checkpoint(s) to a private repo "
            f"({uploaded_names}); {len(selected) - len(to_upload)} already present. Additive/mirror-only "
            f"— nothing deleted. Not mirrored (D-10-DEF-18): {excluded}."
        )
        return f"[backup_sync] uploaded {len(uploaded_names)} dir(s) (hf); {len(remote_names)} already backed up."

    if backup.destination == "local":
        import shutil  # noqa: PLC0415

        dest_root = Path(backup.path) / cfg.output_dir
        dest_root.mkdir(parents=True, exist_ok=True)
        # WR-01: a dir is "already backed up" only when its DESTINATION copy is COMPLETE (both required
        # files present) — a half-written dir (name present, a file missing after an interrupted copy)
        # is NOT counted, so it is re-selected and repaired rather than masquerading as a good backup.
        remote_names = {
            name
            for name in os.listdir(dest_root)
            if (dest_root / name).is_dir() and is_complete(dest_root / name)
        }
        to_upload = plan_uploads(selected, remote_names)
        # D-10-DEF-18: the SAME exclusion the hf branch applies. A local filesystem has no YAML
        # validator to offend, so this is not needed HERE — it is applied anyway so the backup
        # contract is destination-INDEPENDENT: a `restore` from 'local' must reproduce the same tree
        # as a `restore` from 'hf'. One constant, one behaviour, no destination-shaped surprises.
        excluded = sorted(MIRROR_EXCLUDED_FILES)
        ignore = shutil.ignore_patterns(*excluded)  # a CALLABLE — nothing here can mutate `excluded`.
        for d in to_upload:
            # dirs_exist_ok=True (WR-01): re-uploading an INCOMPLETE dest dir overwrites/repairs it in
            # place rather than crashing on an existing dir. Still additive/mirror-only — nothing is
            # ever deleted (never-auto-delete house rule); a repair only re-materializes missing files.
            # `ignore` OMITS a file from the copy; it never removes one already at the destination.
            shutil.copytree(d, dest_root / d.name, dirs_exist_ok=True, ignore=ignore)
        uploaded_names = [d.name for d in to_upload]
        print(
            f"[backup_sync][local] mirrored {len(uploaded_names)} new checkpoint(s) to {dest_root} "
            f"({uploaded_names}); {len(selected) - len(to_upload)} already present. Additive/mirror-only "
            f"— nothing deleted. Not mirrored (D-10-DEF-18): {excluded}."
        )
        return f"[backup_sync] copied {len(uploaded_names)} dir(s) (local); {len(remote_names)} already backed up."

    # UNREACHABLE via the gate (defense-in-depth only): 09.1-07's load-time validator fail-fasts an
    # enabled destination='cloud' block, and the in-body load_config_from_text above re-ran it. A
    # future refactor that bypassed load-time validation would hit this backstop with the same message.
    raise NotImplementedError(
        "backup.destination='cloud' is schema-ready but not yet implemented — only 'hf' and 'local' "
        "are wired this phase. This branch is unreachable (the config validator fail-fasts an enabled "
        "cloud block at load); it exists only as a defense-in-depth backstop."
    )


@app.function(
    image=download_image,  # code-only image + huggingface_hub (same as download_weights).
    volumes={**CHECKPOINTS_MOUNT},  # CPU — mounts the checkpoints Volume; intentionally NO gpu= (D-BK-3).
    secrets=[huggingface_secret],  # HF token from the Modal secret ONLY (never config, never logged).
    timeout=TWENTY_FOUR_HOURS,  # a large rehydrate can be slow; the generous fixed bound download_weights uses.
)
def restore(config_text: str) -> str:
    """CPU rehydrate of the checkpoints Volume from cfg.backup, then commit (D-BK-4, commit-or-vanish).

    The 1:1 backup layout means the downloaded tree lands under the real ``checkpoint-step-*`` names
    (``{output_dir}/checkpoint-step-*``) — a straight copy-back. ADDITIVE: an existing Volume
    checkpoint is never deleted (a same-named restore would just re-materialize identical files).
    ``checkpoints_vol.commit()`` after the write, else the rehydrate vanishes on container exit. The HF
    token comes from the Modal secret env ONLY — never read from config, never printed. cloud is
    rejected at config load; the in-fn branch is a defense-in-depth backstop.
    """
    from pathlib import Path  # noqa: PLC0415

    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415

    # Revalidate in-container (T-03-63) — re-runs 09.1-07's validator, so an enabled cloud block never
    # reaches here (it fail-fasted at the entrypoint gate).
    cfg = load_config_from_text(config_text)
    backup = cfg.backup

    checkpoints_vol.reload()  # reload-before-read (align the mount to the committed state first).

    if backup.destination == "hf":
        from huggingface_hub import snapshot_download  # noqa: PLC0415 — function-local.

        # Download ONLY this run's mirrored prefix; local_dir is the Volume ROOT so the repo's
        # ``{output_dir}/checkpoint-step-*`` paths reproduce CHECKPOINTS_DIR/{output_dir}/checkpoint-step-*
        # 1:1 (a straight copy-back — NOT double-nested under a second output_dir).
        snapshot_download(
            repo_id=backup.repo_id,
            repo_type="model",
            local_dir=str(CHECKPOINTS_DIR),
            allow_patterns=[f"{cfg.output_dir}/**"],
        )
    elif backup.destination == "local":
        import shutil  # noqa: PLC0415

        src_root = Path(backup.path) / cfg.output_dir
        dest_root = CHECKPOINTS_DIR / cfg.output_dir
        dest_root.mkdir(parents=True, exist_ok=True)
        for d in sorted(p for p in src_root.iterdir() if p.is_dir()) if src_root.exists() else []:
            shutil.copytree(d, dest_root / d.name, dirs_exist_ok=True)  # additive copy-back.
    else:
        # UNREACHABLE via the gate (defense-in-depth only) — same load-time fail-fast as backup_sync.
        raise NotImplementedError(
            "backup.destination='cloud' is schema-ready but not yet implemented — only 'hf' and 'local' "
            "are wired this phase. This branch is unreachable (the config validator fail-fasts an enabled "
            "cloud block at load); it exists only as a defense-in-depth backstop."
        )

    checkpoints_vol.commit()  # commit-or-vanish — an uncommitted restore vanishes on container exit.
    restored_root = CHECKPOINTS_DIR / cfg.output_dir
    restored_names = (
        sorted(p.name for p in restored_root.iterdir() if p.is_dir())
        if restored_root.exists()
        else []
    )
    print(
        f"[restore] rehydrated {len(restored_names)} checkpoint dir(s) under {restored_root} "
        f"({restored_names}) and committed the Volume. Additive — no existing checkpoint deleted."
    )
    return f"[restore] restored {len(restored_names)} dir(s) and committed signe-trainer-checkpoints."


# ==================================================================================================
# Phase 10 (H3-01 / H3-03) — the MiniMax-H3 Ref2VA leg: the architecture gate + the pre-encode stage.
# ==================================================================================================
#
# ⛔ THE ARCH GATE IS A PLAIN HELPER, NOT A STAGE OF ITS OWN.
#
# ``run_h3_arch_gate`` below deliberately carries NO Modal stage decorator, no ``gpu=`` and no
# timeout. A function that carries one is reachable by ``modal run -m signet_trainer.modal.fns::<name>``
# — and THAT invocation style boots a metered A100 with NO cost print and NO approval pause. It is a
# documented, still-open defect (Phase 9 ``AUDIT-SYNTHESIS.md`` finding #18, the ``load_ltxv_smoke``
# entry point), NOT a precedent to extend. Phase 10 adds no second ungated entry point, and there is
# deliberately no standalone H3 arch smoke stage anywhere in this file.
#
# Instead the gate runs UNCONDITIONALLY at the front of whichever already-gated stage calls it
# (``h3_preprocess`` below; ``h3_train`` in plan 10-11), inheriting that stage's container, its cost
# line and its blocking approval pause. There is likewise no config knob that turns it off or stops
# after it (``config/schema.py::H3Config`` records the same refusal in its docstring): a cost line is
# only truthful if the function it prices always does the same work.
#
# A standalone check would be redundant anyway — P10-1 already proved the architecture on live
# weights (10/10 constants, 300/300 targets against real ``named_modules``, ~$0.40;
# ``P10-1-MEASURED.md`` sections 1-2). What is NOT yet proved is that every real dispatch re-checks
# before spending, and that is exactly what this helper delivers.
# --------------------------------------------------------------------------------------------------


def run_h3_arch_gate(
    checkpoint_dir: str,
    *,
    device: str = "cuda",
    model: Any = None,
    release: bool = False,
) -> tuple[str, Any]:
    """Assert the MiniMax-H3 architecture on LIVE weights before the caller spends (H3-01).

    A plain module-level helper — the banner above says WHY it is not a stage of its own. It is
    called unconditionally and FIRST inside every gated H3 stage, so every real dispatch aborts
    before spend on an arch mismatch without a second, ungated launch path existing.

    What it does, in order:

      1. a COLD-PATH IMPORT PROBE (``diffusers`` / ``peft`` / ``bitsandbytes``) BEFORE any model
         load — an ImportError discovered after the 61.7 GiB load wastes a metered launch (T-03-63);
      2. loads (or accepts) the ``transformer_ref`` partition;
      3. PRINTS every arch field as ``name expected X got Y OK/MISMATCH`` — the print is the artifact
         the operator diffs against ``P10-1-MEASURED.md`` section 1 — then asserts, reporting EVERY
         offending field at once (enochiatron's equivalent gate caught 6 mismatches in one ~$1.40
         run precisely because it did not stop at the first);
      4. prints the ``adaln_proj.linear`` weight shape, which is what distinguishes the FULL
         diffusers projection from the ComfyUI pruned baked-bottleneck form;
      5. surveys the H3 LoRA target REGEX over the live ``named_modules`` and RAISES unless the
         main-stack total equals ``EXPECTED_H3_NUM_LAYERS x len(H3_LORA_LEAVES)`` with zero
         collateral — the assertion that stops 4% of the adapter training the text-stream refiner.

    It deliberately does NOT inject LoRA, does NOT run a forward pass, and does NOT touch the text
    encoder or either VAE. The gate exists to be cheap; its whole job is to fail before the caller
    spends. Every number it checks is IMPORTED from ``models/h3_loader.py`` / ``lora/peft.py`` — the
    single sources of the H3 arch and target contracts — so nothing here re-derives an H3 fact.

    Args:
        checkpoint_dir: the ``transformer_ref/`` directory on the mounted weights Volume. The CALLER
            composes it (``WEIGHTS_DIR / cfg.model.model_id``), so this function holds no path
            literal of its own (D-NOHARDCODE).
        device: torch device for the load; ``"cuda"`` Modal-side (the partition needs VRAM).
        model: an ALREADY-LOADED transformer. Pass it and the gate costs nothing beyond the probe —
            this is how ``h3_train`` (plan 10-11) avoids paying for a second 61.7 GiB load.
        release: drop the gate's OWN reference to the model before returning, so the 61.7 GiB
            partition never coexists with the text encoder loaded next. ``h3_preprocess`` needs
            this; ``h3_train`` does not (it trains the very model the gate just proved).

    Returns:
        ``(summary_line, model)`` — a one-line summary in ``load_ltxv_smoke``'s style, plus the
        proved model, or ``None`` in its place when ``release=True``.
    """
    # ── (1) COLD-PATH IMPORT PROBE — before ANY model load (the fns.py:305-322 shape, T-03-63) ────
    # No installs here (supply-chain discipline): we VERIFY presence. A missing dep means
    # ``h3_gpu_image`` must add it, re-gated by the Phase-2/10-04 supply-chain rules.
    try:
        import bitsandbytes as bnb  # noqa: PLC0415
        import diffusers  # noqa: PLC0415
        import peft  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            f"[h3-arch-gate] cold-path dependency missing ({exc.name!r}). h3_gpu_image must carry "
            "diffusers (at the pinned DIFFUSERS_SHA) / peft / bitsandbytes BEFORE any sustained GPU "
            "spend — an ImportError discovered after the 61.7 GiB load wastes a metered launch. "
            "Fix: add it to h3_gpu_image in modal/app.py and rebuild (T-03-SC / T-10-04-SC)."
        ) from exc

    import gc  # noqa: PLC0415
    import torch  # noqa: PLC0415

    print(
        f"[h3-arch-gate] cold-path imports OK — diffusers={diffusers.__version__} "
        f"peft={peft.__version__} bitsandbytes={getattr(bnb, '__version__', '?')}"
    )

    from signet_trainer.lora.peft import (  # noqa: PLC0415
        H3_LORA_LEAVES,
        H3_LORA_TARGET_REGEX,
        check_lora_targets,
        check_lora_targets_regex,
    )
    from signet_trainer.models.h3_loader import (  # noqa: PLC0415
        EXPECTED_H3_ADALN_PROJ_SHAPE,
        EXPECTED_H3_NUM_LAYERS,
        EXPECTED_H3_NUM_REFINER_LAYERS,
        assert_h3_arch,
        expected_h3_arch,
        load_h3_transformer,
        summarize_h3_transformer,
    )

    if model is not None and release:
        raise ValueError(
            "[h3-arch-gate] release=True is only valid for a model the gate LOADED itself: it can "
            "drop only its OWN reference, and assign=True loader-owned CUDA storage is freed just "
            "when the LAST reference goes away (06-09 run-5). A caller-supplied model must be "
            "released by that caller."
        )

    # ── (2) the load — the caller's model when given, so h3_train never pays for it twice ────────
    transformer = model if model is not None else load_h3_transformer(checkpoint_dir, device=device)
    if model is None:
        allocated = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(
            f"[h3-arch-gate] loaded the MiniMax-H3 transformer_ref partition from {checkpoint_dir} "
            f"— cuda allocated={allocated:.2f} GiB (P10-1 measured 61.7 GiB, bf16, A100-80GB)."
        )

    # ── (3) the ten measured constants + the three live facts: PRINT every field, THEN assert ────
    # The print is the artifact, not a debug aid: it is what the operator diffs against
    # P10-1-MEASURED.md section 1 straight out of the Modal log.
    summary = summarize_h3_transformer(transformer)
    expected: dict[str, Any] = dict(expected_h3_arch())
    expected["live_transformer_blocks"] = EXPECTED_H3_NUM_LAYERS
    expected["live_refiner_blocks"] = EXPECTED_H3_NUM_REFINER_LAYERS
    expected["adaln_proj_shape"] = EXPECTED_H3_ADALN_PROJ_SHAPE
    for field, want in expected.items():
        got = summary.get(field)
        if got is None:
            verdict = "SKIPPED (probe returned None)"
        else:
            comparable = tuple(got) if isinstance(want, tuple) else got
            verdict = "OK" if comparable == want else "*** MISMATCH ***"
        print(f"[h3-arch-gate]   {field:<24} expected {want!s:<16} got {got!s:<16} {verdict}")
    # Raises naming EVERY offending field (and every field the probe could not read).
    assert_h3_arch(summary)

    # ── (4) the adaln projection shape — the "am I on the right weight set at all" tell ──────────
    print(
        f"[h3-arch-gate] transformer_blocks[0].adaln_proj.linear.weight shape = "
        f"{summary.get('adaln_proj_shape')} — {EXPECTED_H3_ADALN_PROJ_SHAPE} is the FULL projection "
        "on the diffusers path; a [96768, 8] read would be the ComfyUI pruned baked-bottleneck "
        "form, i.e. the wrong weight set mounted (P10-0e discrepancy, resolved by P10-1)."
    )

    # ── (5) the LoRA target survey over the LIVE named_modules ────────────────────────────────────
    # Per-leaf, never a grand total alone: a grand total hides a per-leaf ZERO, and a per-leaf zero
    # is exactly the "silently trains the wrong thing" failure the regex target form exists to
    # prevent (P10-1-MEASURED.md section 8.2).
    names = [name for name, _ in transformer.named_modules()]
    regex_survey = check_lora_targets_regex(names, H3_LORA_TARGET_REGEX)
    # The SUFFIX probe on the same leaves shows what a bare list[str] target set WOULD have matched;
    # the difference per leaf is the token_refiner collateral this regex excludes.
    suffix_survey = check_lora_targets(names, H3_LORA_LEAVES)
    for leaf in H3_LORA_LEAVES:
        main = int(regex_survey["per_leaf"][leaf])
        collateral = int(suffix_survey[leaf]["total"]) - main
        print(f"[h3-arch-gate]   {leaf:<16} main={main:<4} collateral={collateral:<4} (excluded)")

    # DERIVED from the two single-source constants — a leaf added to H3_LORA_LEAVES updates this for
    # free. The measured figure it reproduces is named in the failure message below, not restated
    # here as a second source of truth.
    expected_main = EXPECTED_H3_NUM_LAYERS * len(H3_LORA_LEAVES)
    if int(regex_survey["main"]) != expected_main or int(regex_survey["collateral"]) != 0:
        raise RuntimeError(
            f"[h3-arch-gate] LoRA TARGET MISMATCH — the H3 path regex matched "
            f"main={regex_survey['main']} collateral={regex_survey['collateral']} "
            f"(examples: {regex_survey['collateral_names']}), expected main={expected_main} "
            f"collateral=0. P10-1 measured 300 main-stack targets on live weights "
            f"({EXPECTED_H3_NUM_LAYERS} layers x {len(H3_LORA_LEAVES)} leaves — "
            f"P10-1-MEASURED.md section 2). A short main count means the adapter would train fewer "
            f"modules than priced; ANY collateral means it would train the "
            f"{EXPECTED_H3_NUM_REFINER_LAYERS}-block text-stream token_refiner, which is ~4% of the "
            f"adapter learning the wrong thing. Aborting BEFORE any spend."
        )

    excluded = [
        name
        for name in names
        if name.endswith("adaln_proj.linear") or "proj_in" in name or "proj_out" in name
    ]
    print(
        f"[h3-arch-gate] deliberately NOT targeted: {len(excluded)} module(s) "
        "(adaln_proj.linear + the patch/head projections). At the measured adaln shape a rank-64 "
        "delta there would be a genuine low-rank update, so excluding it is a recipe CHOICE, not a "
        "correctness fix (P10-1-MEASURED.md section 2)."
    )

    line = (
        f"[h3-arch-gate] OK — layers={summary.get('live_transformer_blocks')} "
        f"hidden={summary.get('hidden_size')} in_ch={summary.get('in_channels')} "
        f"audio_in_ch={summary.get('audio_in_channels')} text_dim={summary.get('text_dim')} "
        f"refiner={summary.get('live_refiner_blocks')} adaln={summary.get('adaln_proj_shape')} "
        f"lora_main={regex_survey['main']} lora_collateral={regex_survey['collateral']} "
        f"(vs EXPECTED {EXPECTED_H3_NUM_LAYERS}/{EXPECTED_H3_NUM_REFINER_LAYERS}/"
        f"{EXPECTED_H3_ADALN_PROJ_SHAPE}/{expected_main})"
    )
    print(line)

    if release:
        # The gate's own reference is the LAST one only because ``model`` was None (asserted above),
        # so dropping it here is what actually frees the 61.7 GiB — Module.to("cpu") would not
        # (06-09 run-5). ``names`` / ``summary`` / the surveys hold strings and ints only.
        del transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(
                "[h3-arch-gate] released the gated transformer (two-phase VRAM discipline) — cuda "
                f"allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB. Qwen3-VL is loaded "
                "ONLY after this point."
            )
        return line, None

    return line, transformer


# --------------------------------------------------------------------------------------------------
# ``h3_preprocess`` support. The ENCODE lives in ``prep/h3_encode.py`` (plan 10-07) — everything
# below is the manifest walk / decode / pairing plumbing the Modal stage needs and nothing else.
# Every backend import is FUNCTION-LOCAL, matching the stage-fn discipline (Anti-Pattern 6).
# --------------------------------------------------------------------------------------------------


def _h3_manifest_rows(metadata_path: str) -> list[dict]:
    """Read the JSONL manifest (``data/dataset_file.py``'s writer shape) into a list of dict rows."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    path = Path(metadata_path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    if not rows:
        raise RuntimeError(
            f"[h3_preprocess] the manifest {metadata_path} is empty — refusing to report a "
            "successful pre-encode over zero samples."
        )
    return rows


def _h3_output_rel(media_path: str) -> Any:
    """Manifest media path -> the rel path naming this sample's four cached outputs.

    Mirrors ``data/mask_encode.py::_output_relative`` verbatim: signet manifests carry ``media_path``
    RELATIVE to the manifest's parent, and ``PrecomputedDataset`` pairs the four H3 sources by that
    same relative path. A divergence here does not raise — the sample is simply absent from the
    index, which is exactly why this is one shared helper rather than four call sites.
    """
    from pathlib import Path  # noqa: PLC0415

    path = Path(media_path)
    rel = Path(*path.parts[1:]) if path.is_absolute() else path
    return rel.with_suffix(".pt")


def _h3_decode_rgb_frames(path: Any, max_frames: int) -> list:
    """Decode up to ``max_frames`` RGB frames (av -> cv2 -> imageio), function-local backends.

    Same preference ladder as ``data/mask_encode.py::_read_video_gray``, but RGB is load-bearing
    here (the H3 video VAE takes ImageNet-normalized RGB), so the cv2 branch converts BGR rather
    than relying on the channel-mean invariance the mask reader enjoys.
    """
    frames: list = []
    try:
        import av  # noqa: PLC0415

        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if len(frames) >= max_frames:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
    except ImportError:
        try:
            import cv2  # noqa: PLC0415

            capture = cv2.VideoCapture(str(path))
            try:
                while len(frames) < max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                capture.release()
        except ImportError:
            import imageio  # noqa: PLC0415

            import numpy as np  # noqa: PLC0415

            reader = imageio.get_reader(str(path))
            for frame in reader:
                if len(frames) >= max_frames:
                    break
                frames.append(np.asarray(frame))
    if not frames:
        raise ValueError(f"[h3_preprocess] clip decoded to zero frames: {path} (empty/unreadable).")
    return frames


def _h3_read_video_rgb(path: Any, num_frames: int, height: int, width: int) -> Any:
    """Decode + LANCZOS-resize a clip to the target canvas -> uint8 ``[3, F, H, W]`` in ``[0, 255]``.

    ``imagenet_normalize`` divides by 255 FIRST, so uint8 in ``[0, 255]`` is exactly the input the
    H3 encode recipe expects — do NOT pre-scale here.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image as PILImage  # noqa: PLC0415

    frames = _h3_decode_rgb_frames(path, num_frames)
    if len(frames) < num_frames:
        raise RuntimeError(
            f"[h3_preprocess] {path} decoded to {len(frames)} frame(s) but this campaign encodes "
            f"{num_frames} (the frame law lives in conditioning/h3_geometry.py). A short clip would "
            "encode to a latent grid the packed-sequence budget never priced. Re-stage the clip or "
            "fix the manifest — do NOT pad, which would teach the adapter a frozen tail."
        )

    resized: list = []
    for frame in frames[:num_frames]:
        image = PILImage.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), PILImage.Resampling.LANCZOS)
        # np.array (a COPY), not np.asarray: Pillow hands back a READ-ONLY buffer and
        # torch.from_numpy on a non-writable array yields a tensor whose writes are UB (10-07 #3).
        resized.append(np.array(image, dtype=np.uint8))

    stacked = torch.from_numpy(np.stack(resized, axis=0))  # [F, H, W, 3]
    return stacked.permute(3, 0, 1, 2).contiguous()  # [3, F, H, W]


def _h3_read_audio_waveform(path: Any, sampling_rate: int) -> Any:
    """Decode a clip's audio to stereo ``[1, 2, N]`` float32 at ``sampling_rate``, or ``None``.

    ``None`` means the container carries NO audio stream — the caller then writes an EXPLICIT
    absent marker rather than fabricating silence (D-10-AUDIO; fabricated zeros would teach the
    model to be silent, which is not the same as not targeting audio).
    """
    import av  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    chunks: list = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        # Resample to the VAE's OWN declared rate: feeding a container-rate waveform would be
        # silently off-distribution at a perfectly valid shape (the trap class this whole leg
        # guards structurally).
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="stereo", rate=sampling_rate
        )
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                chunks.append(np.asarray(resampled.to_ndarray(), dtype=np.float32))
    if not chunks:
        return None
    waveform = torch.from_numpy(np.concatenate(chunks, axis=-1))  # [channels, samples]
    return waveform.unsqueeze(0)  # [1, channels, samples]


def _h3_open_reference_image(path: Any) -> Any:
    """Open one reference image as RGB. The D-10-CROP crops are applied to the SOURCE files."""
    from PIL import Image as PILImage  # noqa: PLC0415

    with PILImage.open(path) as handle:
        return handle.convert("RGB")


def _h3_reference_entry(entry: Any, index: int, kind: str | None, where: str) -> dict:
    """Normalize ONE manifest reference entry into the descriptor its cached slot must carry.

    ⛔ **``kind`` and ``subject_id`` are READ, never inferred.** ``kind`` decides D-10-REFORDER slot
    ordering and the shared rotary clock makes a reordered reference set a genuinely DIFFERENT
    request, so a guessed kind trains against conditioning nobody asked for — silently, at a
    perfectly valid shape. ``subject_id`` groups images of one subject and is the label a budget
    refusal NAMES (``C+008``). Its vocabulary is already declared in the config
    (``h3.character_reference_sizes`` / ``h3.environment_reference_sizes``, third tuple element) but
    the config carries no path, and joining a slot to a config entry by ``source_wh`` is AMBIGUOUS —
    two environment references can share a size. Only the MANIFEST can join a file to its identity.

    Accepted entry shape, per reference::

        {"path": "refs/<file>", "subject_id": "A"}                        # kind from the key
        {"path": "refs/<file>", "subject_id": "A", "kind": "character"}   # kind explicit

    The caller supplies ``kind`` for the pool keys (``character_references`` /
    ``environment_reference`` say what their members are). The flat ``reference_paths`` key says
    nothing, so entries under it must declare their own.
    """
    if isinstance(entry, str):
        raise ValueError(
            f"[h3_preprocess] {where} entry {index} is a bare path string. An H3 reference entry is "
            'a mapping that carries its DESCRIPTOR too: {"path": "...", "subject_id": "A"'
            + ("}" if kind else ', "kind": "character"}')
            + ". The pre-encode is the only place that can join a reference file to its identity: "
            "the config declares the subject_id vocabulary but carries no path, and a size join is "
            "ambiguous. Guessing either field silently reorders the references, which the shared "
            "rotary clock makes a different request."
        )
    if not isinstance(entry, dict):
        raise TypeError(
            f"[h3_preprocess] {where} entry {index} is a {type(entry).__name__}; expected a mapping "
            'like {"path": "...", "subject_id": "A"}.'
        )
    resolved_kind = entry.get("kind", kind)
    missing = [
        name
        for name, value in (
            ("path", entry.get("path")),
            ("subject_id", entry.get("subject_id")),
            ("kind", resolved_kind),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"[h3_preprocess] {where} entry {index} is missing {missing}. "
            "conditioning/h3_ref._parse_reference_pool requires path / kind / subject_id on every "
            "cached slot: selection needs 'kind' (D-10-REFORDER puts the environment LAST) and "
            "'subject_id' (images of one subject are grouped, and it is the label a budget refusal "
            "names). Neither is derivable from a latent tensor."
        )
    return {
        "path": str(entry["path"]),
        "kind": str(resolved_kind),
        "subject_id": str(entry["subject_id"]),
    }


def _h3_resolve_references(
    row: dict,
    index: int,
    data_root: Any,
    *,
    references_per_sample: int,
    reference_pair_seed: int,
    environment_ref_last: bool,
) -> list:
    """Resolve this sample's reference slots — EXACTLY ``references_per_sample`` of them.

    Two manifest shapes, in precedence order:

      * ``reference_paths`` — an explicit per-sample list, used verbatim (order is load-bearing).
        A flat list says nothing about what its members ARE, so every entry declares its own
        ``kind``;
      * ``character_references`` (a pool) + optional ``environment_reference`` — the D-10-PAIRSEED
        round robin. The rotation is a pure function of ``reference_pair_seed`` and the row index,
        so which refs a sample gets is reproducible and debuggable across runs. Here the manifest
        KEY supplies ``kind``, so an entry need only carry ``path`` + ``subject_id``.

    ⛔ D-10-REFORDER + the operator ruling: an environment reference **SUBSTITUTES** for the last
    character slot — the character count is reduced by one — it is **never appended**. Three
    references per sample was never priced by the packed-sequence VRAM budget, and rotating pairs
    are what make identity the INVARIANT rather than letting the model copy a constant reference.

    Returns:
        One DESCRIPTOR per slot, in D-10-REFORDER order::

            {"path": <manifest-relative str>, "kind": ..., "subject_id": ..., "source": <Path>}

        ``path`` is the manifest-RELATIVE string that travels into the cached payload — it is the
        join key ``H3RefStrategy`` gathers reference rows by, and keeping it relative keeps a
        container mount prefix out of a committed artifact. ``source`` is the resolved on-disk file
        and never leaves this container.
    """
    from pathlib import Path  # noqa: PLC0415

    explicit = row.get("reference_paths")
    if explicit:
        picks = [
            _h3_reference_entry(entry, i, None, f"row {index} 'reference_paths'")
            for i, entry in enumerate(explicit)
        ]
    else:
        pool = [
            _h3_reference_entry(entry, i, "character", f"row {index} 'character_references'")
            for i, entry in enumerate(row.get("character_references", []))
        ]
        if not pool:
            raise KeyError(
                f"[h3_preprocess] manifest row {index} names no references (keys: {sorted(row)}). "
                "Every Ref2VA sample needs either 'reference_paths' (explicit) or "
                "'character_references' (a pool for the seeded round robin), optionally plus "
                "'environment_reference'. Refusing to encode a reference-less ref2v sample."
            )
        environment = row.get("environment_reference")
        n_character = references_per_sample - (1 if environment else 0)
        if n_character < 1:
            raise ValueError(
                f"[h3_preprocess] references_per_sample={references_per_sample} leaves no character "
                f"slot once the environment reference substitutes for the last one (row {index})."
            )
        start = (reference_pair_seed + index) % len(pool)
        picks = [pool[(start + offset) % len(pool)] for offset in range(n_character)]
        if environment:
            env = _h3_reference_entry(
                environment, 0, "environment", f"row {index} 'environment_reference'"
            )
            picks = [*picks, env] if environment_ref_last else [env, *picks]

    if len(picks) != references_per_sample:
        raise RuntimeError(
            f"[h3_preprocess] manifest row {index} resolved to {len(picks)} reference(s), expected "
            f"exactly {references_per_sample}. An environment reference SUBSTITUTES for the last "
            f"character slot and is never appended; a different count was never priced by the "
            f"packed-sequence budget and would OOM this metered container."
        )
    return [{**pick, "source": data_root / Path(pick["path"])} for pick in picks]


def _h3_select_reference_row(
    rows: list,
    data_root: Any,
    *,
    subject_ids: Any,
    references_per_sample: int,
    reference_pair_seed: int,
    environment_ref_last: bool,
) -> tuple[int, list]:
    """Pick WHICH manifest row supplies a render's reference slots — by identity, never by index.

    ``h3_sample`` used to hardcode ``rows[0]``, so every eval prompt at every length conditioned on
    ONE reference pair. For a phase whose headline value is ref2v that is the thinnest possible axis,
    and the fix is a per-config selector — but a bare ``rows[N]`` index is the wrong shape for it
    twice over: it silently re-points if the manifest is ever re-staged in a different order, and it
    names the ROW when the thing an eval varies is the reference CONDITION.

    So the selector names the resolved slots by ``subject_id`` — the config's own declared
    vocabulary (``h3.*_reference_sizes``, third tuple element), the label a budget refusal names, and
    the label ``delta.json`` already records. Re-staging cannot change what ``["C", "018"]`` means.
    A clip / media name would be the other durable candidate and is deliberately NOT used: it is
    client property and must never enter a tracked config.

    ⛔ This chooses WHICH ROW supplies the references. It never re-orders them and never re-rolls the
    rotation: each candidate row is resolved by the REAL ``_h3_resolve_references``, so D-10-REFORDER
    (environment reference LAST) and the D-10-PAIRSEED round robin both stand untouched. The match is
    ORDER-EXACT for the same reason — a list written environment-first is refused rather than
    silently accepted, because a reordered reference set is a genuinely different request on the
    shared rotary clock.

    Args:
        rows: The manifest rows, in file order.
        data_root: Directory the manifest's relative paths resolve against.
        subject_ids: The wanted slots in D-10-REFORDER order. EMPTY selects row 0 — the historical
            behaviour, so every config predating this field is byte-identically unaffected.
        references_per_sample: The fixed slot count.
        reference_pair_seed: D-10-PAIRSEED.
        environment_ref_last: D-10-REFORDER.

    Returns:
        ``(row_index, references)`` — the descriptors ``_h3_resolve_references`` produced for it.
    """
    def resolve(index: int) -> list:
        """The REAL resolver, so the rotation and the slot order are never re-implemented here."""
        return _h3_resolve_references(
            rows[index],
            index,
            data_root,
            references_per_sample=references_per_sample,
            reference_pair_seed=reference_pair_seed,
            environment_ref_last=environment_ref_last,
        )

    if not rows:
        raise RuntimeError(
            "[h3_sample] the manifest has no rows, so no reference slots can be resolved."
        )

    wanted = [str(s) for s in (subject_ids or [])]
    if not wanted:
        return 0, resolve(0)

    if len(wanted) != references_per_sample:
        raise RuntimeError(
            f"[h3_sample] validation.reference_subject_ids names {len(wanted)} slot(s) {wanted} but "
            f"h3.references_per_sample is {references_per_sample}. Every H3 sample carries a FIXED "
            f"slot count — an environment reference SUBSTITUTES for the last character slot and is "
            f"never appended — and a different count was never priced by the packed-sequence budget."
        )

    available: dict[tuple, list[int]] = {}
    for index in range(len(rows)):
        slots = tuple(str(r["subject_id"]) for r in resolve(index))
        available.setdefault(slots, []).append(index)

    matches = available.get(tuple(wanted))
    if not matches:
        choices = "\n".join(
            f"    {list(slots)}  ({len(idxs)} row(s))" for slots, idxs in sorted(available.items())
        )
        reordered = [
            list(slots) for slots in available if sorted(slots) == sorted(wanted) and list(slots) != wanted
        ]
        hint = (
            f"\n  ⚠ {reordered} DOES exist — D-10-REFORDER puts the environment reference LAST, and "
            f"the match is order-exact because a reordered reference set is a different request on "
            f"the shared rotary clock."
            if reordered
            else ""
        )
        raise RuntimeError(
            f"[h3_sample] validation.reference_subject_ids {wanted} matches no manifest row. The "
            f"selector names the reference CONDITION, so an unmatched value means the corpus does "
            f"not pair those subjects — refusing to fall back to row 0, which would render a "
            f"different probe under the label of this one.{hint}\n"
            f"  Available reference conditions in this manifest:\n{choices}"
        )

    if len(matches) > 1:
        print(
            f"[h3_sample] reference condition {wanted} is carried by {len(matches)} manifest row(s); "
            f"taking the first (row {matches[0]}). The slots are identical on all of them — that is "
            f"what the selector selects — so the choice is deterministic and immaterial."
        )
    return matches[0], resolve(matches[0])


def _h3_vae_latent_stats(vae: Any) -> tuple[Any, Any]:
    """Read the per-channel ``latents_mean`` / ``latents_std`` off the video VAE's own config.

    Read, never assumed: the H3 encode recipe's final step is ``(latents - mean) / std``, and
    substituting 0/1 would produce correctly-shaped, silently off-distribution latents.
    """
    config = getattr(vae, "config", vae)
    mean = getattr(config, "latents_mean", None)
    std = getattr(config, "latents_std", None)
    if mean is None or std is None:
        raise RuntimeError(
            "[h3_preprocess] the video VAE config carries no latents_mean/latents_std. The H3 "
            "encode normalizes every latent per channel with them; substituting 0/1 would write a "
            "cache that is off-distribution at a perfectly valid shape. Refusing to encode."
        )
    return mean, std


def _h3_load_component(component_dir: str, *, device: str, dtype: Any) -> Any:
    """Load one diffusers-format H3 component (video VAE / audio VAE) from the weights Volume.

    ``AutoModel`` dispatches on the checkpoint's own ``_class_name``, so the concrete class is read
    from the mounted weights rather than restated here — restating it would be a re-derived H3 arch
    fact, which this phase forbids (``models/h3_loader.py`` is the single source of those).

    ⛔ **``.eval()`` alone was the load-site half of D-10-DEF-10.** It reads like "inference mode" and
    is not — it switches norm/dropout behaviour and leaves ``requires_grad=True`` on every parameter
    ``from_pretrained`` just created. The encode helpers all carry ``@h3_no_grad`` now, which is the
    load-bearing guard; ``freeze_h3_component`` is the second lock, and it VERIFIES the freeze took
    rather than assuming it.
    """
    from diffusers import AutoModel  # noqa: PLC0415

    from signet_trainer.prep.h3_grad_contract import freeze_h3_component  # noqa: PLC0415

    component = AutoModel.from_pretrained(component_dir, torch_dtype=dtype)
    return freeze_h3_component(component.to(device), what=f"the component at {component_dir}")


@app.function(
    # The MiniMax-H3 family image (H3-07): diffusers at the pinned DIFFUSERS_SHA, NOT the LTX
    # ltx-core/ltx-trainer stack. A gpu= with the code-only default image boots an A100 and dies at
    # ``import torch`` (tests/test_modal_gpu_image.py).
    gpu="A100-80GB",
    image=h3_gpu_image,
    volumes={**WEIGHTS_MOUNT, **DATASET_MOUNT},
    secrets=[huggingface_secret],
    # The H3 addition no LTX GPU fn needs: the P10-1 probe REQUIRED this for the 61.7 GiB load
    # (scripts/_h3_probe_modal.py:281). Request 80 GiB of RAM, allow up to 200 GiB.
    memory=(80 * 1024, 200 * 1024),
    timeout=TWENTY_FOUR_HOURS,
)
def h3_preprocess(
    metadata_path: str,
    output_dir: str,
    target_frames: int,
    target_aspect: tuple[int, int],
    reference_image_short_edge: int,
    reference_pair_seed: int,
    references_per_sample: int,
    environment_ref_last: bool,
    text_encoder_layer: int,
    with_audio: bool,
    max_packed_rows: int,
    model_id: str,
    vae_id: str,
    audio_vae_id: str,
    text_encoder_id: str,
) -> str:
    """Stage 1 (H3) — the signet-NATIVE MiniMax-H3 Ref2VA pre-encode, cached to the dataset Volume.

    There is no canonical H3 encoder anywhere (not in-repo, not upstream), so the enochiatron "never
    write a custom encoder" landmine does not apply — there is nothing canonical to prefer. The
    encode LOGIC all lives in ``prep/h3_encode.py`` (plan 10-07, where the four silent-corruption
    traps are guarded structurally); this function is the thin gated wrapper around it.

    EVERY parameter is REQUIRED with no default. A defaulted parameter is how a threading gap goes
    silent: with no defaults a missing kwarg is a ``TypeError`` at dispatch, before a container is
    even allocated. The entrypoint (plan 10-12) supplies all of them from the validated config, and
    never a ``SignetConfig`` object nor a path into ``configs/`` (that dir is not in the image).

    Body order is strict, and ``tests/test_h3_preprocess_wiring.py`` source-scans it:

      0. a decode-backend cold-path probe — cheaper than everything below it, so it runs first;
      1. ``run_h3_arch_gate`` — UNCONDITIONAL, before a single frame is decoded. It carries its own
         cold-path import probe, so a missing dependency or a mismatched architecture aborts at the
         cheapest possible point. No flag skips it and none stops after it: a cost line is only
         truthful if the function it prices always does the same work;
      2. PHASE A — mount the processor, run the D-10-DEF-7 PARITY BACKSTOP against it (real
         ``__call__`` vs our bypass, output-key diff, before the 32B load so a gap costs seconds),
         then load Qwen3-VL, build each sample's presentation, encode at the asserted text-encoder
         layer, write ``h3_conditions/``;
      3. ``free_text_encoder`` + the caller-side reference drop, printing the freed VRAM delta.
         ⛔ Qwen3-VL-32B and the 61.7 GiB transformer must NEVER coexist — a single A100-80GB
         physically cannot hold both. The DISCIPLINE is the proven Gemma pattern (fns.py:445-464);
         the CODE is not (``CachedPromptEmbeddings`` / ``load_embeddings_processor`` are ltx-trainer
         objects with no H3 equivalent);
      4. PHASE B — load the video VAE (and the audio VAE when requested) and write ``h3_latents/``,
         ``h3_reference_latents/`` and ``h3_audio_latents/``;
      5. the LOUD-FAILURE guard — a requested output that produced ZERO files RAISES;
      6. ``dataset_vol.commit()`` — commit-or-vanish (Pitfall 3), non-negotiable.

    Reference slots: EXACTLY ``references_per_sample`` per sample. An environment-bearing sample
    gets one rotating character reference plus the environment reference, which SUBSTITUTES for the
    last character slot — never three. ``reference_image_short_edge`` is 896 for Phase 10 (the
    2-slot pairing domain is what fits one A100); VAE latents cannot be spatially downscaled after
    the fact, so a higher-fidelity campaign needs a full RE-ENCODE.

    Audio: **0 of the 44 corpus clips carry an audio stream (measured)**, so the ``with_audio``
    branch is expected NOT to fire for this campaign — and that is not a defect. D-10-AUDIO:
    not-TARGETING audio is not the same as training silence. The target audio ROWS stay present and
    noised in the packed batch, which is ``train/h3_step.py``'s job, not this encoder's; they are
    merely kept out of the loss.

    Note for plan 10-11: ``build_h3_presentation``'s vision spans are deterministic given the
    reference SOURCE sizes, and those are persisted per slot in the ``h3_reference_latents/``
    payload (``source_wh`` / ``latent_rows``), so the train side recomputes them rather than
    depending on a span sidecar this stage does not write.
    """
    # ── (0) decode-backend cold-path probe. h3_gpu_image (10-04) carries diffusers/peft/torch but
    # deliberately no ffmpeg and no decode package, so this is the FIRST thing that can be missing —
    # and finding out after a 61.7 GiB load plus a Qwen3-VL load would burn most of a container.
    try:
        import av  # noqa: PLC0415, F401
    except ImportError as exc:
        raise RuntimeError(
            f"[h3_preprocess] no video decode backend ({exc.name!r}). This stage decodes clips to "
            "RGB (and, when with_audio, demuxes audio), which h3_gpu_image does not yet provide: "
            "10-04 deliberately shipped it WITHOUT ffmpeg because nothing on the H3 path demuxed "
            "audio at the time. Fix: add `av` (and ffmpeg for the audio leg) to h3_gpu_image in "
            "modal/app.py and rebuild, re-gated by the supply-chain discipline (T-10-04-SC). "
            "Aborting before any model load."
        ) from exc

    import gc  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from signet_trainer.conditioning.h3_geometry import (  # noqa: PLC0415
        h3_audio_rows,
        h3_latent_frames,
        resolve_canvas_size,
        rows_of,
    )
    from signet_trainer.models.h3_loader import EXPECTED_H3_TEXT_ENCODER_LAYER  # noqa: PLC0415
    from signet_trainer.prep.h3_encode import (  # noqa: PLC0415
        H3_AUDIO_LATENTS_DIR,
        H3_CONDITIONS_DIR,
        H3_REFERENCE_LATENTS_DIR,
        H3_VIDEO_LATENTS_DIR,
        build_h3_presentation,
        encode_h3_audio_latents,
        encode_h3_reference_latents,
        encode_h3_text_conditions,
        encode_h3_video_latents,
        free_text_encoder,
        h3_absent_audio_payload,
        h3_vae_smoke_encode,
        prepare_h3_reference_images,
        presentation_refs_from_prepared,
        write_h3_precomputed,
    )
    from signet_trainer.prep.h3_grad_contract import freeze_h3_component  # noqa: PLC0415
    from signet_trainer.prep.h3_parity import (  # noqa: PLC0415
        assert_h3_processor_output_parity,
    )
    from signet_trainer.prep.h3_text_payload import build_h3_text_payload  # noqa: PLC0415

    # ── (1) THE ARCH GATE — unconditional, first, and it releases the 61.7 GiB partition before
    # Qwen3-VL is loaded below (they cannot coexist on one A100-80GB).
    gate_line, _gated = run_h3_arch_gate(
        str(WEIGHTS_DIR / model_id), device="cuda", release=True
    )
    print(f"[h3_preprocess] arch gate passed -> {gate_line}")

    # The config's text-encoder layer must AGREE with the asserted arch constant. The encode reads
    # the constant (never a parameter), so a config naming a different layer would be silently
    # ignored — and a final-layer encode is wrong at the correct (B, L, 5120) shape.
    if int(text_encoder_layer) != EXPECTED_H3_TEXT_ENCODER_LAYER:
        raise RuntimeError(
            f"[h3_preprocess] config text_encoder_layer={text_encoder_layer} disagrees with the "
            f"asserted EXPECTED_H3_TEXT_ENCODER_LAYER={EXPECTED_H3_TEXT_ENCODER_LAYER} "
            "(models/h3_loader.py, the single source of every H3 arch number). MiniMax-H3 reads "
            "that Qwen3-VL hidden state out of its 64; a different one is off-distribution "
            "conditioning at a perfectly valid shape. Aborting before any encode."
        )

    rows = _h3_manifest_rows(metadata_path)
    data_root = Path(metadata_path).parent
    canvas_height, canvas_width = resolve_canvas_size(*target_aspect)
    n_target_video = h3_latent_frames(int(target_frames)) * rows_of(canvas_height, canvas_width)
    n_target_audio = h3_audio_rows(int(target_frames))
    print(
        f"[h3_preprocess] {len(rows)} sample(s); canvas {canvas_width}x{canvas_height}, "
        f"{target_frames} pixel frames -> {h3_latent_frames(int(target_frames))} latent frames, "
        f"{n_target_video} target video rows + {n_target_audio} target audio rows; reference short "
        f"edge {reference_image_short_edge}, {references_per_sample} slot(s)/sample; ceiling "
        f"{max_packed_rows} packed rows."
    )

    # ── (1b) THE BOTH-MODALITIES SMOKE — one clip + one reference through the REAL VAE ────────────
    # ⛔ D-10-DEF-9 and D-10-DEF-12 both live in `_encode`'s `num_frames == 1` branch, which NO
    # video-side success can exercise and NO CPU test can reach (one is a CUDA dispatch decision).
    # Five containers died at the TOP of an 88-sample encode, each after PHASE A had already pushed
    # the whole corpus through Qwen3-VL-32B. This runs BEFORE PHASE A — that placement is the whole
    # value, not the check — so the entire family now costs one small VAE load and two encodes.
    #
    # The VAE is loaded and FREED again here rather than kept: it is seconds to reload (3 shards),
    # and Qwen3-VL-32B has to have the card essentially to itself for PHASE A.
    smoke_row = rows[0]
    smoke_references = _h3_resolve_references(
        smoke_row,
        0,
        data_root,
        references_per_sample=references_per_sample,
        reference_pair_seed=reference_pair_seed,
        environment_ref_last=environment_ref_last,
    )
    smoke_vae = _h3_load_component(str(WEIGHTS_DIR / vae_id), device="cuda", dtype=torch.float32)
    smoke_mean, smoke_std = _h3_vae_latent_stats(smoke_vae)
    print(
        h3_vae_smoke_encode(
            smoke_vae,
            clip_pixels=_h3_read_video_rgb(
                data_root / smoke_row["media_path"],
                int(target_frames),
                canvas_height,
                canvas_width,
            ),
            reference_image=_h3_open_reference_image(smoke_references[0]["source"]),
            reference_short_edge=reference_image_short_edge,
            reference_descriptor=smoke_references[0],
            latents_mean=smoke_mean,
            latents_std=smoke_std,
            clip_pixel_frames=int(target_frames),
        )
    )
    # Drop OUR reference too — assign=True loader-owned CUDA storage survives `Module.to("cpu")` and
    # is released only when the last reference goes away (06-09 run-5), the same discipline PHASE A's
    # own teardown uses below.
    smoke_vae = None
    smoke_mean = smoke_std = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[h3_preprocess] smoke VAE released: "
        f"{torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0:.2f} GiB "
        f"resident before Qwen3-VL loads."
    )

    # ── (2) PHASE A — Qwen3-VL text conditioning, FIRST and alone in VRAM ────────────────────────
    from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

    text_encoder_dir = str(WEIGHTS_DIR / text_encoder_id)
    processor = AutoProcessor.from_pretrained(text_encoder_dir)
    # ⛔ D-10-DEF-7 — the PARITY BACKSTOP, between the processor mount and the Qwen3-VL load.
    # `build_h3_processor_inputs` deliberately bypasses `processor.__call__`, and a bypass inherits
    # EVERY output that call makes. Two of those gaps have already been found one at a time, each
    # costing a metered container (the pre-expanded presentation, then `mm_token_type_ids`). This
    # runs the REAL `__call__` against our builder on a tiny synthetic input and diffs the output
    # KEY SETS, so a third gap is a preflight failure measured in seconds — the processor is
    # tokenizer + image-processor config only, and Qwen3-VL-32B has not been loaded yet.
    print(f"[h3_preprocess] {assert_h3_processor_output_parity(processor)}")
    text_encoder = AutoModel.from_pretrained(text_encoder_dir, torch_dtype=torch.bfloat16)
    # D-10-DEF-10, the load-site half. Qwen3-VL survived on `.eval()` alone only because
    # `encode_h3_text_conditions` was the ONE place in prep/h3_encode.py that carried a no-grad
    # context; the VAEs, which did not, exhausted the GPU. Freeze every encode component the same
    # way, so surviving stops depending on which helper happened to remember.
    text_encoder = freeze_h3_component(text_encoder.to("cuda"), what="the Qwen3-VL text encoder")
    before_free_gib = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"[h3_preprocess] PHASE A: Qwen3-VL loaded from {text_encoder_dir}.")

    per_sample: dict[str, dict] = {}
    for index, row in enumerate(rows):
        rel = _h3_output_rel(row["media_path"])
        references = _h3_resolve_references(
            row,
            index,
            data_root,
            references_per_sample=references_per_sample,
            reference_pair_seed=reference_pair_seed,
            environment_ref_last=environment_ref_last,
        )
        # ⛔ D-10-DEF-4: ONE resize, shared with PHASE B. `prepare_h3_reference_images` is the only
        # place a reference's encode geometry is decided, and it returns the RESIZED image and its
        # vision-token count as one object. PHASE A used to hand the processor the RAW file while
        # counting the resized one — reference `A` gridded 1,014 tokens against the 1,176 the spans
        # and the packed budget assumed. There is deliberately no local geometry call here any more.
        prepared = prepare_h3_reference_images(
            [_h3_open_reference_image(r["source"]) for r in references],
            reference_image_short_edge,
        )
        vision_counts = [int(ref.vision_tokens) for ref in prepared]
        presentation, spans = build_h3_presentation(
            row["caption"],
            presentation_refs_from_prepared(prepared),
            # EXACT spans + the hard "the tokenizer knows the vision sentinels" assertion. A
            # tokenizer that shreds them would shift every downstream modality tag silently.
            tokenizer=getattr(processor, "tokenizer", None),
        )
        # The PREPARED references travel in, not raw images: the processor sees exactly the pixels
        # whose row count the spans bill, and `build_h3_processor_inputs` re-checks the realized
        # grid, the realized pad count and the realized span positions before the encoder runs.
        hidden = encode_h3_text_conditions(
            text_encoder, processor, presentation, prepared, vision_spans=spans
        )
        # ⛔ The SECOND, PROMPT-ONLY state. D-10-REFDROP removes the reference LATENT rows on ~20% of
        # steps, but the cached text state used to keep both references' Qwen vision blocks — so a
        # "dropped" step described references the model could not see, a regime that exists nowhere
        # at inference (a no-reference request's presentation contains no vision blocks at all).
        # `build_h3_presentation(caption, [])` is that presentation, and it is built through the SAME
        # function rather than by string surgery, so the label/prompt layout cannot drift between the
        # two states. Empty references => empty spans, which the payload builder then enforces.
        prompt_only_presentation, prompt_only_spans = build_h3_presentation(
            row["caption"], (), tokenizer=getattr(processor, "tokenizer", None)
        )
        prompt_only_hidden = encode_h3_text_conditions(
            text_encoder, processor, prompt_only_presentation, (), vision_spans=prompt_only_spans
        )
        # ⛔ The SPANS ARE PERSISTED. They used to be computed here and dropped, while `h3_ref` read
        # `batch.get("vision_spans", ())` from a batch nothing ever populated — so every training
        # step tagged >90% of its text stream TEXT instead of VIDEO, silently. The payload is
        # self-describing and versioned precisely so the pre-fix cache cannot be read as valid.
        write_h3_precomputed(
            output_dir,
            rel,
            text=build_h3_text_payload(
                hidden_states=hidden,
                vision_spans=spans,
                prompt_only_hidden_states=prompt_only_hidden,
                has_references=bool(prepared),
            ),
        )
        per_sample[str(rel)] = {
            # The DESCRIPTORS, not bare paths: phase B re-opens the images AND writes path / kind /
            # subject_id into the cached payload, which is what makes the cache readable by
            # conditioning/h3_ref._parse_reference_pool at all (D-10-DEF-2).
            "references": references,
            "text_rows": int(hidden.shape[0]),
            "vision_spans": spans,
            # Carried across the phase boundary so PHASE B can REFUSE a disagreement rather than
            # cache one. The shared resize makes them equal; this makes an inequality loud.
            "vision_counts": vision_counts,
        }
        del hidden, prompt_only_hidden  # never two samples' hidden states resident at once

    # ── (3) FREE Qwen3-VL before ANYTHING else large is loaded ────────────────────────────────────
    free_text_encoder(text_encoder, processor)
    # The CALLER must drop its OWN references too: assign=True loader-owned CUDA storage is released
    # only when the LAST reference goes away — Module.to("cpu") does not do it (06-09 run-5).
    text_encoder = None
    processor = None
    gc.collect()
    after_free_gib = 0.0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        after_free_gib = torch.cuda.memory_allocated() / 2**30
    print(
        f"[h3_preprocess] freed Qwen3-VL: {before_free_gib:.2f} -> {after_free_gib:.2f} GiB "
        f"(delta {before_free_gib - after_free_gib:.2f} GiB). The VAEs load ONLY after this point."
    )

    # ── (4) PHASE B — the VAEs ────────────────────────────────────────────────────────────────────
    # float32, NOT bf16, and this is load-bearing: ``imagenet_normalize`` hands the VAE a float32
    # tensor (``pixels.to(torch.float32).div(255.0)`` is step one of the transcribed recipe), and
    # a diffusers VAE does NOT cast its input — ``encode`` on a bf16 module would raise
    # "expected scalar type BFloat16 but found Float" inside the metered container. The DELIBERATE
    # precision reduction on this path is the float16 round-trip AFTER the encode
    # (prep/h3_encode.py::encode_video_latents), not a low-precision module. The VAEs are small and
    # the 61.7 GiB transformer was released by the arch gate, so fp32 here costs nothing that
    # matters.
    video_vae = _h3_load_component(str(WEIGHTS_DIR / vae_id), device="cuda", dtype=torch.float32)
    latents_mean, latents_std = _h3_vae_latent_stats(video_vae)
    audio_vae = None
    audio_sampling_rate = None
    if with_audio:
        audio_vae = _h3_load_component(
            str(WEIGHTS_DIR / audio_vae_id), device="cuda", dtype=torch.float32
        )
        audio_sampling_rate = getattr(getattr(audio_vae, "config", None), "sampling_rate", None)
        if audio_sampling_rate is None:
            raise RuntimeError(
                "[h3_preprocess] with_audio=True but the audio VAE declares no sampling_rate — "
                "resampling to a guessed rate would be off-distribution at a valid shape."
            )

    audio_present = 0
    for index, row in enumerate(rows):
        rel = _h3_output_rel(row["media_path"])
        meta = per_sample[str(rel)]
        media = data_root / row["media_path"]

        pixels = _h3_read_video_rgb(media, int(target_frames), canvas_height, canvas_width)
        # ⛔ D-10-DEF-12: NO `.to("cuda")` here any more, and its absence is the fix. This site used
        # to carry one and the reference site below did not, so the reference reached a CUDA-resident
        # VAE with CPU pixels — cuDNN declined, the dispatcher chose ConvBackend.Slow3d, and Slow3d
        # has no CUDA kernel. Placement now happens ONCE, inside `encode_video_latents`, off the
        # component's own device, exactly like the ONE resize site (D-10-DEF-4) and the ONE rank
        # refusal (D-10-DEF-9). Re-adding a literal device here would restore the two-opinions state
        # that made the reference path silently different from this one.
        video_payload = encode_h3_video_latents(
            video_vae,
            pixels,
            latents_mean,
            latents_std,
            pixel_frames=int(target_frames),
        )
        del pixels

        images = [_h3_open_reference_image(r["source"]) for r in meta["references"]]
        reference_payload = encode_h3_reference_latents(
            video_vae,
            images,
            reference_image_short_edge,
            latents_mean,
            latents_std,
            # D-10-DEF-2: the descriptors travel INTO the payload. Without them the cache carries
            # sizes but no identity, and H3RefStrategy cannot order or gather the slots — it fails
            # loud naming the three missing fields, after a full metered pre-encode has been paid
            # for. They are propagated from the manifest, never inferred from a tensor.
            descriptors=meta["references"],
            references_per_sample=references_per_sample,
        )

        # ⛔ D-10-DEF-4, the cross-phase assertion. `prepare_h3_reference_images` makes these equal
        # by construction (both phases call it, on the same files, at the same short edge), so this
        # can only fire if a future edit gives one phase its own geometry again — which is exactly
        # what happened the first time. It is checked rather than assumed because the failure is
        # silent: wrong vision spans, wrong modality tags, and a packed budget that no longer
        # describes the batch, all at a perfectly valid shape.
        encoded_rows = [int(slot["latent_rows"]) for slot in reference_payload["references"]]
        if encoded_rows != list(meta["vision_counts"]):
            raise RuntimeError(
                f"[h3_preprocess] sample {rel}: PHASE A presented {meta['vision_counts']} vision "
                f"token(s) per reference slot but PHASE B encoded {encoded_rows} latent row(s). "
                f"The two phases must describe the SAME reference at the SAME geometry — the "
                f"conditioning rows, the Qwen vision spans, the AdaLN modality tags and the packed "
                f"budget are all billed off this one number (D-10-DEF-4). Refusing to write a cache "
                f"whose text conditioning and reference latents disagree."
            )

        # ⛔ The absent marker is written for EVERY sample, ``with_audio`` or not. ``h3_audio_latents``
        # is one of the FOUR sources ``H3RefStrategy.get_data_sources()`` declares, and
        # ``data/precomputed.py`` raises ``FileNotFoundError`` on a configured source whose dir does
        # not exist. Leaving ``audio_payload = None`` on the (correct) ``with_audio=False`` campaign
        # meant ``write_h3_precomputed`` skipped it, the dir was never created, and the FIRST
        # training dispatch died on a missing source — container #4, bought for nothing. The
        # committed 4-source contract in ``prep/h3_encode``'s docstring is what this now honours.
        audio_payload: Any = h3_absent_audio_payload(
            "with_audio=False — this campaign does not target audio (D-10-AUDIO: 0 of 44 corpus "
            "clips carry a stream). The source is written as an EXPLICIT absence so the four-source "
            "layout is complete; fabricating silence would train the model to be silent."
        )
        if with_audio:
            waveform = _h3_read_audio_waveform(media, int(audio_sampling_rate))
            latents = (
                None
                if waveform is None
                # D-10-DEF-12: placement is the encode helper's job, off the audio VAE's own
                # device — not a literal written at a call site that has never executed.
                else encode_h3_audio_latents(audio_vae, waveform, is_reference=False)
            )
            if latents is None:
                audio_payload = h3_absent_audio_payload()
            else:
                audio_payload = latents
                audio_present += 1

        write_h3_precomputed(
            output_dir,
            rel,
            video=video_payload,
            references=reference_payload,
            audio=audio_payload,
        )

        # REALIZED packed rows, from what was actually encoded. The config-load budget check prices
        # the worst-case pair from declared source sizes; this is the belt-and-braces sibling that
        # knows the true tokenized text length and the true encoded reference grids.
        realized = (
            int(meta["text_rows"])
            + sum(int(slot["latent_rows"]) for slot in reference_payload["references"])
            + n_target_video
            + n_target_audio
        )
        if realized > max_packed_rows:
            raise RuntimeError(
                f"[h3_preprocess] sample {rel} realizes {realized} packed rows, over the measured "
                f"ceiling of {max_packed_rows} by {realized - max_packed_rows}. Encoding it would "
                f"cache a sample that OOMs the training container. Lower "
                f"reference_image_short_edge (currently {reference_image_short_edge}) or "
                f"target_frames, or escalate the GPU — aborting at the cheap step."
            )
        print(
            f"[h3_preprocess]   {rel}: {realized}/{max_packed_rows} packed rows "
            f"(text {meta['text_rows']}, refs "
            f"{[int(s['latent_rows']) for s in reference_payload['references']]}, "
            f"target {n_target_video}+{n_target_audio})"
        )
        del video_payload, reference_payload, audio_payload

    # ── (5) LOUD-FAILURE guard — a requested output that produced ZERO files RAISES ───────────────
    # Copied from the a2v guard (fns.py:210-226) and it exists for the same reason: a swallowed
    # exception once produced a "successful" encode with an empty dir, and the run only failed much
    # later (or worse, trained without the conditioning it was supposed to have).
    counts: dict[str, int] = {}
    for name in (
        H3_VIDEO_LATENTS_DIR,
        H3_CONDITIONS_DIR,
        H3_REFERENCE_LATENTS_DIR,
        H3_AUDIO_LATENTS_DIR,
    ):
        source_dir = Path(output_dir) / name
        counts[name] = len(list(source_dir.rglob("*.pt"))) if source_dir.exists() else 0
        print(f"[h3_preprocess] {name}/: {counts[name]} file(s)")

    if counts[H3_VIDEO_LATENTS_DIR] == 0 or counts[H3_CONDITIONS_DIR] == 0:
        raise RuntimeError(
            f"[h3_preprocess] the encode produced {counts[H3_VIDEO_LATENTS_DIR]} target latent(s) "
            f"and {counts[H3_CONDITIONS_DIR]} text condition(s) under {output_dir} — both are "
            "MANDATORY sources. Refusing to report a successful pre-encode."
        )
    if reference_image_short_edge and counts[H3_REFERENCE_LATENTS_DIR] == 0:
        raise RuntimeError(
            f"[h3_preprocess] reference encoding was requested (short edge "
            f"{reference_image_short_edge}) but ZERO .pt files landed under h3_reference_latents/ "
            f"in {output_dir}. Refusing to report a successful ref2v encode without references — "
            "an empty reference dir does not raise downstream, it silently drops every sample from "
            "the PrecomputedDataset index."
        )
    if with_audio and audio_present == 0:
        raise RuntimeError(
            f"[h3_preprocess] with_audio=True but ZERO of {len(rows)} clip(s) yielded an audio "
            "stream, so h3_audio_latents/ holds only absent-markers. Either the corpus genuinely "
            "has no audio (0 of 44 measured — D-10-AUDIO; then run with with_audio=False) or the "
            "demux is silently failing. Refusing to report a successful audio encode without "
            "audio."
        )

    # ── (6) Pitfall 3 commit-or-vanish: without commit() the whole encode is lost on container exit
    # and `modal volume ls signe-trainer-dataset` would show nothing.
    dataset_vol.commit()

    print(
        f"[h3_preprocess] committed {sum(counts.values())} file(s) across "
        f"{len([n for n, c in counts.items() if c])} source dir(s) -> {output_dir}."
    )
    return output_dir


# --------------------------------------------------------------------------------------------------
# ``h3_train`` / ``h3_sample`` support (plan 10-11). Every backend import stays FUNCTION-LOCAL.
# --------------------------------------------------------------------------------------------------


def _h3_to_device(value: Any, device: Any, dtype: Any) -> Any:
    """Recursively move a precomputed H3 sample onto the device, casting FLOATS to ``dtype``.

    Walks dicts and lists because the ``h3_reference_latents`` payload is a LIST of per-slot dicts
    (10-07) — the two slots encode at DIFFERENT sizes, so they cannot be one tensor. Integer and
    boolean tensors keep their dtype: ``token_tags`` / the three index tensors are ``long`` by
    contract and a cast would be a silent type error at the AdaLN gather.
    """
    import torch  # noqa: PLC0415

    if isinstance(value, torch.Tensor):
        moved = value.to(device)
        return moved.to(dtype) if moved.is_floating_point() else moved
    if isinstance(value, dict):
        return {k: _h3_to_device(v, device, dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [_h3_to_device(v, device, dtype) for v in value]
    return value


def h3_adapter_delta(model: Any, batch_kwargs: dict, n_cond_video: int) -> float:
    """``max|delta velocity|`` between the BASE model and the adapter on one fixed batch (H3-06).

    This is **D-10-SCOPEGUARD's acceptance criterion in its automatable form**: "the adapter provably
    moves the model". Nothing here grades adapter QUALITY — that is the operator's judgement on the
    grid. The only claim it makes is the one a machine can actually check.

    Both passes run under ``torch.no_grad()`` on the IDENTICAL ``batch_kwargs``; the base pass uses
    PEFT's ``disable_adapter()`` context manager, so the comparison is exact and **no second 61.7 GiB
    model is loaded** — a second load would not fit on the A100 alongside the first anyway.

    Why a non-zero result is attributable: **``lora_B`` is zero-init**, so a freshly injected adapter
    is an exact identity and the two passes are bit-identical before any optimizer step. Any
    difference afterwards is attributable to the optimizer steps and nothing else. P10-1 measured
    ``8.413e-01`` after ONE step on a real A100 (``scripts/_h3_probe_modal.py:449-458`` /
    ``P10-1-MEASURED.md`` section 3), so a TRAINED adapter returning ``0.0`` is a failed run
    regardless of how the loss curve looked.

    Both outputs are sliced at ``[:, n_cond_video:]`` — the transformer returns conditioning rows
    UNMASKED by contract (``transformer_minimax_h3.py`` L44-50) and the reference prefix is
    identical on both sides anyway, so including it would only dilute the maximum.
    """
    import torch  # noqa: PLC0415

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            with model.disable_adapter():
                base = model(**batch_kwargs, return_dict=False)[0][:, n_cond_video:].float()
            adapted = model(**batch_kwargs, return_dict=False)[0][:, n_cond_video:].float()
        return float((adapted - base).abs().max().item())
    finally:
        if was_training:
            model.train()


@app.function(
    # Same H3 family image + memory request as ``h3_preprocess``: diffusers at the pinned
    # DIFFUSERS_SHA, NOT the LTX ltx-core/ltx-trainer stack, and the 80 GiB RAM the P10-1 probe
    # required for the 61.7 GiB load.
    gpu="A100-80GB",
    image=h3_gpu_image,
    volumes={**WEIGHTS_MOUNT, **DATASET_MOUNT, **CHECKPOINTS_MOUNT},
    secrets=[huggingface_secret, wandb_secret],
    memory=(80 * 1024, 200 * 1024),
    timeout=TWENTY_FOUR_HOURS,
    # THE PREEMPTION CONTRACT (2026-08-06, operator-approved). Safe for the reason it always was:
    # ``h3_train`` resumes in-dir from the latest COMMITTED checkpoint and ``train_loop`` commits
    # per save, so a retry can never observe a half-written checkpoint. No warm-GPU tokens (D-10).
    #
    # ``max_retries=10`` is MODAL'S PLATFORM CEILING, taken because the cadence-derived requirement
    # is UNREACHABLE by this kwarg. Both numbers matter, so both are recorded:
    #   * THE REQUIREMENT, measured on ap-lEaWCrVX8efqNm9R5EEE1u — 1 initial attempt + 3 retries =
    #     4 container lives produced 250 COMMITTED steps in ~1.6 h wall clock => 62.5 steps of NET
    #     progress per container life. That figure already absorbs both the 61.7 GiB reload and the
    #     <=50-step re-do since the last commit; it is measured, not modelled. Remaining work
    #     3000 - 250 = 2750 steps => 2750 / 62.5 = 44 CONTAINER LIVES REQUIRED.
    #   * THE CEILING, measured 2026-08-06 on app ``ap-8Gra2Yka1fs4pwMIh8AgLv``: dispatching with
    #     ``max_retries=60`` is REJECTED BY MODAL'S SERVER at app init —
    #     "Invalid function retries. Must specify number between 0 and 10". ⚠ ``modal/retries.py``
    #     validates only ``max_retries >= 0``, so the CLIENT accepts any non-negative int and the
    #     real bound is invisible locally. It failed before any container booted and before any
    #     spend, which is the one good property of discovering it this way.
    #   * SO: 10 retries = 11 container lives ~= 687 steps of net progress at the observed cadence.
    #     That is 2.75x what r1 got, and it does NOT cover the remaining 2750 steps.
    #     ⛔ THE RESIDUAL GAP IS REAL AND STILL OPEN (D-10-DEF-16): server-side retries CANNOT make
    #     a 2750-step round preemption-proof at this cadence. Closing it needs a DIFFERENT
    #     mechanism (a local re-dispatch supervisor, or shorter rounds), which is an operator
    #     decision — deliberately NOT invented here. Do not "fix" this by raising the number; the
    #     platform will reject it at app init.
    #   * THE DELAY TAIL IS BOUNDED, NOT EXPONENTIAL: modal/retries.py caps ``max_delay`` at 60.0 s
    #     (it REJECTS anything above) and ``initial_delay`` is already 60.0, so
    #     ``backoff_coefficient=2.0`` is clamped from the very first retry. 10 retries add AT MOST
    #     ~10 min of cumulative queue delay — never a doubling tail.
    #
    # ``single_use_containers=True`` — a FRESH container per retry. This is Modal's canonical
    # long-training shape AND is documented in this repo's own CLAUDE.md Modal-patterns table
    # ("fresh container per retry, resume from last Volume checkpoint. Critical for >24h or
    # preemption") — yet it was set NOWHERE in src/. Encoding it HERE, as structure, is the point:
    # the lesson existed only as PROSE in a doc and therefore did not transfer.
    #
    # Raising retries removes NO safety, precisely because these stay put: the est_hours-derived
    # container timeout (15.0 * 1.5 = 22.5 h), the armed checkpoint watchdog
    # (``checkpoint_expected_minutes: 15.0``) and ``cost_guardrail_usd: 40.0`` all still bound the
    # round. ⛔ ``h3_sample`` is DELIBERATELY EXCLUDED from both kwargs (it carries no retries at
    # all): a RENDER is not resumable in-dir, so a retry silently re-does the whole thing rather
    # than continuing it — raising a render's retry budget multiplies a total-loss unit.
    retries=modal.Retries(max_retries=10, initial_delay=60.0, backoff_coefficient=2.0),
    single_use_containers=True,
)
def h3_train(config_yaml: str) -> None:
    """Stage 2 (H3) — the gated MiniMax-H3 Ref2VA LoRA training run (H3-05 / H3-06).

    The cadence is REUSED, not forked: ``train/loop.py``'s resume -> accumulate -> clip ->
    save -> commit -> callback -> commit is model-agnostic and is threaded here through its
    ``step_fn`` seam. Only the FORWARD is H3's, because H3 is a single-stream packed-sequence DiT
    with no LTX ``Modality`` analog.

    Body order — ``tests/test_h3_train_wiring.py`` source-scans it:

      1. COLD-PATH IMPORT PROBE before any load (T-03-63). ``diffusers`` / ``peft`` /
         ``bitsandbytes`` only. ⚠ ``wandb`` is deliberately NOT probed and NOT imported: 10-04's
         ``h3_gpu_image`` does not declare it and ``train/loop.py`` never calls it, so probing it
         would abort EVERY H3 run on a dependency nothing uses. The secret is still injected by
         name so a future logging leg needs no decorator change.
      2. ``load_config_from_text(config_yaml)`` — the recipe crosses BY VALUE as YAML text, never a
         path (``configs/`` is not in the image). Re-validating in-container is the T-03-63
         convention and it re-fires the frame-law and seq-len-budget checks inside the paid
         container too.
      3. the CPU PREFLIGHT — build the strategy and the dataset and run ONE real
         ``prepare_training_inputs`` before the 61.7 GiB load, using the arch constants
         ``models/h3_loader`` measured on live weights. Every payload-contract and geometry
         disagreement (a cache encoded at a different short edge, a reference payload missing its
         descriptors, a drifted canvas) therefore aborts at cents rather than after a full load.
         The arch gate at (4) then proves the live model matches the very constants this preflight
         assumed, so the two are not independent guesses.
      4. ``run_h3_arch_gate`` — the SHARED helper ``h3_preprocess`` calls, not a second copy. It
         aborts on any arch mismatch or on anything but 300 main-stack / 0 collateral LoRA targets,
         BEFORE any training spend (enochiatron's equivalent gate caught 6 mismatches for ~$1.40).
         It is handed no ``model=``, so it LOADS one and RETURNS it — the run pays for exactly one
         61.7 GiB load, never two.
      5. ``build_lora_config`` + ``inject_lora`` (GC before ``get_peft_model``, TRAIN-06).
      6. the H3 step closure + ``build_optimizer`` + ``train_loop``.
      7. ``h3_adapter_delta`` on a FIXED batch; a delta of exactly ``0.0`` RAISES.
      8. ``checkpoints_vol.commit()`` — commit-or-vanish.

    ⛔ The offloader stays INERT. ``cfg.offload`` is not read and no ``blocks_to_swap`` is enabled:
    the measured Phase-10 geometry (12,362 rows, 76.36 GiB peak) fits one A100 at
    ``blocks_to_swap 0``, and block-swap is explicitly NOT the answer at H3 campaign geometry —
    reaching it would mean swapping ~38 of the 50 blocks.

    ⛔ No in-loop validation sampling on this run. It doubles the residency risk on a 61.7 GiB model,
    and Phase 10's acceptance signal is the delta measurement plus the separately-gated
    ``--mode sample``.

    ``training.keep_checkpoints`` stays ``None`` for this campaign: a finite value silently prunes
    intermediates, and in a research lab the intermediates ARE the artifacts.
    """
    # ── (1) COLD-PATH IMPORT PROBE — before ANY model load (T-03-63) ─────────────────────────────
    # No installs (supply-chain discipline): we VERIFY presence. A missing dep means h3_gpu_image
    # must add it, re-gated by the Phase-2 / 10-04 supply-chain rules.
    try:
        import bitsandbytes as bnb  # noqa: PLC0415
        import diffusers  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import peft  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            f"[h3_train] cold-path dependency missing ({exc.name!r}). h3_gpu_image must carry "
            "diffusers (at the pinned DIFFUSERS_SHA) / peft / bitsandbytes / numpy BEFORE any "
            "sustained GPU spend — an ImportError discovered after the 61.7 GiB load wastes a "
            "metered launch. (numpy is a hard torch dependency and so is present in practice, but "
            "h3_gpu_image does not DECLARE it, and 'present in practice' is exactly the assumption "
            "that turns into a paid ModuleNotFoundError one image rebuild later.) Fix: add it to "
            "h3_gpu_image in modal/app.py and rebuild (T-03-SC)."
        ) from exc

    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    print(
        f"[h3_train] cold-path imports OK — diffusers={diffusers.__version__} "
        f"peft={peft.__version__} bitsandbytes={getattr(bnb, '__version__', '?')}. "
        "wandb is deliberately NOT probed: h3_gpu_image does not declare it and train/loop.py "
        "never calls it, so probing would abort every run on an unused dependency."
    )

    from signet_trainer.conditioning.h3_geometry import (  # noqa: PLC0415
        h3_audio_rows,
        max_packed_rows_for_budget,
    )
    from signet_trainer.conditioning.h3_packing import make_h3_position_ids_fn  # noqa: PLC0415
    from signet_trainer.conditioning.h3_ref import H3RefStrategy  # noqa: PLC0415
    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415
    from signet_trainer.data.precomputed import PrecomputedDataset  # noqa: PLC0415
    from signet_trainer.lora.peft import build_lora_config, inject_lora  # noqa: PLC0415
    from signet_trainer.models.h3_loader import (  # noqa: PLC0415
        EXPECTED_H3_AUDIO_IN_CHANNELS,
        EXPECTED_H3_IN_CHANNELS,
        EXPECTED_H3_PATCH_SIZE,
        EXPECTED_H3_TEXT_DIM,
    )
    from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415
    from signet_trainer.train.flow_match import FlowMatchingSchedule  # noqa: PLC0415
    from signet_trainer.train.h3_step import (  # noqa: PLC0415
        H3StepDeps,
        h3_draw_timesteps,
        h3_step_deps_from_model,
    )
    from signet_trainer.train.loop import (  # noqa: PLC0415
        build_optimizer,
        build_scheduler,
        should_warm_start,
        train_loop,
    )

    # ── (2) load + REVALIDATE the config in-container (the recipe crossed by value) ──────────────
    config = load_config_from_text(config_yaml)
    if config.model.family != "h3":
        raise RuntimeError(
            f"[h3_train] model.family is {config.model.family!r}, not 'h3'. This stage drives the "
            "MiniMax-H3 packed-sequence forward and the H3 path-regex adapter; an LTX config here "
            "would inject a target set that matches nothing and only surface at build_optimizer's "
            "'No trainable parameters found' — after the metered A100 is already billing."
        )

    device = "cuda"
    checkpoint_path = str(WEIGHTS_DIR / config.model.model_id)

    # The realized-seq_len ceiling, computed IN-CONTAINER from the re-validated config (never a
    # literal, and never taken from the caller: this function holds the whole config already).
    # Leaving it unset would silently disable build_h3_packed_batch's runtime ceiling assertion,
    # turning a drifted dataset into an OOM 40 minutes into a metered run instead of a loud,
    # attributable error on the first batch.
    max_packed_rows = max_packed_rows_for_budget(
        config.h3.gpu_usable_gib, config.h3.resident_gib, config.h3.mib_per_packed_row
    )

    # ⛔ ``training_dims[2]`` and NOTHING ELSE. This builder re-derives the target latent grid and
    # then PROVES it against the rows measured off the cached tensors, so it must be handed the very
    # frame count the CACHE was encoded at — and that is the value ``entrypoint._h3_encode_params``
    # threads into ``h3_preprocess(target_frames=...)``: ``cfg.training_dims[2]``. Two spellings
    # that look plausible here are both wrong:
    #   * ``config.data.frame_count`` does not EXIST (frame_count is a ValidationConfig field) — a
    #     bare AttributeError, which is what D-10-DEF-1 was;
    #   * ``config.validation.frame_count`` resolves but is a RENDER field, and the training config
    #     (configs/h3_embe_r1.yaml) does not set it — it would silently take the schema default 49
    #     against a 22-frame cache and abort the preflight with a geometry disagreement naming a
    #     number that appears nowhere in the YAML.
    # tests/test_h3_config_reads.py pins BOTH: every config chain read here resolves against the
    # real Pydantic models, and this argument is derived from the same source as the pre-encode.
    position_ids_fn = make_h3_position_ids_fn(
        target_frames=config.training_dims[2],
        target_aspect=tuple(config.h3.target_aspect),
        reference_short_edge=config.h3.reference_image_short_edge,
        patch_size=EXPECTED_H3_PATCH_SIZE,
    )

    def _build_strategy(deps: Any) -> Any:
        """Every constructor argument is a config field — D-NOHARDCODE, field for field."""
        return H3RefStrategy(
            references_per_sample=config.h3.references_per_sample,
            reference_dropout=config.h3.reference_dropout,
            reference_pair_seed=config.h3.reference_pair_seed,
            environment_ref_last=config.h3.environment_ref_last,
            audio_in_loss=config.h3.audio_in_loss,
            t_visual_cond=config.h3.t_visual_cond,
            deps=deps,
            position_ids_fn=position_ids_fn,
            patch_size=tuple(deps.config.patch_size),
            max_packed_rows=max_packed_rows,
            # An audio-less sample still carries its priced target-audio rows, synthesized as noise
            # and out of the loss (D-10-AUDIO). The count is a function of the campaign frame count
            # — derived here from the same `training_dims[2]` the cache was encoded at, never a
            # literal and never defaulted to 0.
            target_audio_rows=h3_audio_rows(int(config.training_dims[2])),
        )

    # The dataset reads the ROOT from config (never a hardcoded ``.precomputed`` — that was a real
    # Phase-6 bug), over exactly the sources the strategy DECLARES, so the dir -> output-key map is
    # single-sourced with ``_PRECOMPUTED_SOURCE_OUTPUT_KEYS``.
    data_sources = {
        name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[name]
        for name in H3RefStrategy().get_data_sources()
    }
    dataset = PrecomputedDataset(config.data.preprocessed_data_root, data_sources=data_sources)

    # ── (3) CPU PREFLIGHT — one real batch build BEFORE the 61.7 GiB load ───────────────────────
    # The arch widths come from models/h3_loader, which MEASURED them on live weights; the gate at
    # (4) then asserts the mounted model reports those exact values, so this is not a second guess.
    preflight_deps = H3StepDeps(
        transformer=None,
        config=type("_Cfg", (), {"patch_size": EXPECTED_H3_PATCH_SIZE})(),
        patch_dim=EXPECTED_H3_IN_CHANNELS
        * EXPECTED_H3_PATCH_SIZE[0]
        * EXPECTED_H3_PATCH_SIZE[1]
        * EXPECTED_H3_PATCH_SIZE[2],
        audio_in_channels=EXPECTED_H3_AUDIO_IN_CHANNELS,
        text_dim=EXPECTED_H3_TEXT_DIM,
    )
    _probe_batch = dict(dataset[0])
    _probe_batch["segment_index"] = int(_probe_batch.get("idx", 0))
    _probe_batch["step"] = 0
    _probe_batch["t_video"], _probe_batch["t_audio"] = h3_draw_timesteps(
        np.random.default_rng(config.training.seed), uniform_prob=config.training.uniform_prob
    )
    _probe_inputs = _build_strategy(preflight_deps).prepare_training_inputs(_probe_batch)
    print(
        f"[h3_train] CPU preflight PASSED on sample 0 — packed {_probe_inputs.packed.seq_len} rows "
        f"(text {_probe_inputs.packed.n_text} + ref video {_probe_inputs.packed.n_cond_video} + "
        f"audio {_probe_inputs.packed.n_cond_audio + _probe_inputs.packed.n_target_audio} + target "
        f"video {_probe_inputs.packed.n_target_video}) vs the {max_packed_rows}-row ceiling from "
        f"max_packed_rows_for_budget({config.h3.gpu_usable_gib}, {config.h3.resident_gib}, "
        f"{config.h3.mib_per_packed_row}); references="
        f"{[r.subject_id for r in _probe_inputs.references]}. Built before any model load."
    )
    del _probe_inputs, _probe_batch
    gc.collect()

    # ── (4) the SHARED arch gate — abort BEFORE any training spend ───────────────────────────────
    # Not a second copy: this is the same helper h3_preprocess calls. release defaults to False and
    # no model= is passed, so the gate LOADS the transformer and hands it back — one 61.7 GiB load
    # for the whole run.
    gate_line, transformer = run_h3_arch_gate(checkpoint_path, device=device)
    print(f"[h3_train] {gate_line}")
    if transformer is None:  # defensive: release=False must always return the proved model
        raise RuntimeError(
            "[h3_train] the arch gate returned no model. h3_train must train the very transformer "
            "the gate just proved — re-loading 61.7 GiB would be an expensive mistake and would "
            "train a model the gate never inspected."
        )

    # ── (5) inject the H3 PATH-REGEX adapter (GC before get_peft_model — TRAIN-06) ───────────────
    # cfg.lora.target_modules may be a bare ``str`` on the H3 family (the path regex), which is why
    # resolved_lora_targets() is used rather than ``or P1_FF_LORA_TARGETS``: that idiom would keep a
    # regex intact but the list-shaped LTX fallback would silently match ~104 wrong modules.
    lora_config = build_lora_config(
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        targets=config.resolved_lora_targets(),
    )
    model = inject_lora(transformer, lora_config)
    del transformer
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"[h3_train] injected the H3 adapter over "
        f"{'a path regex' if isinstance(config.resolved_lora_targets(), str) else 'a suffix list'} "
        f"— {len(trainable)} trainable tensor(s), "
        f"{sum(p.numel() for p in trainable)} params, rank={config.lora.rank}."
    )

    # ── (6) the H3 forward, threaded into the model-agnostic loop ────────────────────────────────
    deps = h3_step_deps_from_model(model.base_model.model if hasattr(model, "base_model") else model)
    strategy = _build_strategy(deps)
    torch_dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float32
    step_counter = {"n": 0}

    def _h3_step(model_: Any, batch_: Any, schedule_: Any, rng_: Any, *, device: Any, dtype: Any):
        """One H3 packed-sequence training step: draw -> pack -> forward -> masked velocity loss.

        ``schedule_`` (the LTX ``FlowMatchingSchedule``) is deliberately UNUSED. Its shift is the
        MEAN of a logit-normal draw and lerps with sequence length; H3's is an exponential sigma
        reparameterization fixed at 12.0 / 3.0. Passing 12.0 through the LTX formulation pins every
        sample at ~0.999994 — see train/h3_step.h3_shifted_sigma. The draw comes from
        ``h3_draw_timesteps`` instead, and the loop's ``rng`` still drives it, so the whole schedule
        stays reproducible from ``training.seed``.
        """
        step_counter["n"] += 1
        batch = _h3_to_device(batch_, device, dtype)
        # D-10-REFDROP is a function of (segment, step). ``idx`` is PrecomputedDataset's own sample
        # index (precomputed.py:292) and is the stable segment identity; the step counter makes the
        # dropout draw vary across the run rather than freezing each segment into one regime.
        batch["segment_index"] = int(batch.get("idx", 0))
        batch["step"] = step_counter["n"]
        batch["t_video"], batch["t_audio"] = h3_draw_timesteps(
            rng_, uniform_prob=config.training.uniform_prob
        )
        inputs = strategy.prepare_training_inputs(batch)
        # ⛔ NO autocast. MiniMaxH3Transformer3DModel declares
        # `_keep_in_fp32_modules = ["proj_in", "audio_proj_in", "time_embedder", "proj_out",
        # "audio_proj_out", "rope"]` — it ships a MIXED-PRECISION checkpoint and does its own
        # per-module dtype alignment, precisely so it runs correctly without one (the reference
        # pipeline calls it bare). An `autocast("cuda", bf16)` wrapper overrides that and runs those
        # fp32 linears in bf16, including the TIME EMBEDDER that drives every AdaLN row — silently,
        # at the correct shape. Do not reintroduce it "for speed": the checkpoint already chose its
        # own precisions.
        output = model_(**inputs.packed.kwargs, return_dict=False)
        return strategy.compute_loss(inputs, output)

    ckpt_manager = CheckpointManager(
        CHECKPOINTS_DIR / config.output_dir, keep_n=config.training.keep_checkpoints
    )
    if should_warm_start(
        ckpt_manager.find_latest() is not None, config.training.init_adapter_path
    ):
        from signet_trainer.lora.peft import load_adapter_into  # noqa: PLC0415

        init_dir = CHECKPOINTS_DIR / config.training.init_adapter_path
        load_adapter_into(model, init_dir)
        print(f"[h3_train][chain] warm-started from {init_dir} (fresh optimizer, step 0).")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps=config.training.max_steps)
    final_step = train_loop(
        model,
        dataset,
        optimizer,
        scheduler,
        FlowMatchingSchedule(uniform_prob=config.training.uniform_prob),
        ckpt_manager,
        config,
        checkpoints_vol,  # commit-per-save: a preemption cannot vanish an uncommitted checkpoint.
        step_fn=_h3_step,
        # bs=1 is LOCKED; take the sample through untouched so the reference payload's list of
        # per-slot dicts (strings + differently-shaped tensors) is not restructured by collation.
        collate_fn=lambda samples: samples[0],
    )
    print(f"[h3_train] loop done — reached step {final_step}/{config.training.max_steps}.")

    # ── (7) THE ACCEPTANCE SIGNAL — a trained adapter must provably move the model (H3-06) ───────
    fixed = dict(dataset[0])
    fixed["segment_index"] = int(fixed.get("idx", 0))
    fixed["step"] = 0
    fixed["t_video"], fixed["t_audio"] = h3_draw_timesteps(
        np.random.default_rng(config.training.seed), uniform_prob=0.0
    )
    fixed_inputs = strategy.prepare_training_inputs(
        _h3_to_device(fixed, device, torch_dtype)
    )
    delta = h3_adapter_delta(model, fixed_inputs.packed.kwargs, fixed_inputs.packed.n_cond_video)
    print(
        f"[h3_train] ADAPTER-MOVES-MODEL: max|delta velocity| base-vs-adapter on a fixed batch = "
        f"{delta:.3e} (P10-1 measured 8.413e-01 after ONE step; lora_B is zero-init so any non-zero "
        f"value is attributable to the optimizer steps and nothing else)."
    )
    if delta == 0.0:
        raise RuntimeError(
            f"[h3_train] the trained adapter did NOT change the model output (max|delta velocity| "
            f"== 0.0 after {final_step} step(s)). D-10-SCOPEGUARD's acceptance criterion is that "
            "the adapter provably moves the model, so this is a FAILED run regardless of what the "
            "loss curve did — check that the LoRA targets matched (the arch gate's 300/0 survey), "
            "that build_optimizer saw trainable params, and that the checkpoint being measured is "
            "the one that trained."
        )

    # ── (8) commit-or-vanish (Pitfall 3) ─────────────────────────────────────────────────────────
    checkpoints_vol.commit()
    print(
        f"[h3_train] done — checkpoints committed to signe-trainer-checkpoints under "
        f"{config.output_dir}/ (commit-or-vanish: 'done' is the file on the Volume, not a log line)."
    )


def _h3_fixed_delta_batch(config: Any, dataset: Any, strategy: Any, device: Any, dtype: Any) -> Any:
    """Build the ONE fixed batch both H3 stages measure ``h3_adapter_delta`` on.

    Fixed in every sense that matters to the comparison: sample 0, step 0, and a timestep pair drawn
    from a generator seeded with ``training.seed`` at ``uniform_prob=0`` — so the base and adapter
    passes differ in the adapter and in nothing else, and two runs of the same config measure the
    same thing.
    """
    import numpy as np  # noqa: PLC0415

    from signet_trainer.train.h3_step import h3_draw_timesteps  # noqa: PLC0415

    batch = dict(dataset[0])
    batch["segment_index"] = int(batch.get("idx", 0))
    batch["step"] = 0
    batch["t_video"], batch["t_audio"] = h3_draw_timesteps(
        np.random.default_rng(config.training.seed), uniform_prob=0.0
    )
    return strategy.prepare_training_inputs(_h3_to_device(batch, device, dtype))


@app.function(
    # Same shape as ``h3_train`` MINUS the retries. ⚠ The original reason — "a render is not
    # resumable in-dir, so a retry would silently re-do the whole thing" — NO LONGER HOLDS: the
    # render dir is keyed on the render's identity and every clip is skipped-if-present and
    # committed as it lands, so a re-dispatch continues instead of restarting. Re-enabling retries
    # is now a safe change, but it is a COST decision (it multiplies the worst-case spend by
    # max_retries) and is left to the operator rather than taken here. The entrypoint applies the
    # config-derived timeout.
    gpu="A100-80GB",
    image=h3_gpu_image,
    volumes={**WEIGHTS_MOUNT, **DATASET_MOUNT, **CHECKPOINTS_MOUNT},
    # ⛔ HF_HUB_OFFLINE — the STRUCTURAL half of the D-10-DEF-14 egress guard, and it has to be set
    # HERE rather than inside the function: `huggingface_hub` freezes the flag into a module
    # constant at IMPORT time, and this function imports diffusers (hence hub) on its first line.
    # As an env var on the container it is in place before the interpreter starts. Every H3
    # component is loaded from a mounted local directory, so nothing legitimate here needs the Hub —
    # and the failure this guards is not an exception, it is ~134 GiB of egress whose only symptom
    # is the bill. `inference/h3_pipeline_source.py`'s two assertions make a Hub id LOUD; this makes
    # it IMPOSSIBLE even if a future edit slips one past them.
    secrets=[
        huggingface_secret,
        wandb_secret,
        modal.Secret.from_dict({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
    ],
    memory=(80 * 1024, 200 * 1024),
    timeout=TWENTY_FOUR_HOURS,
)
def h3_sample(config_yaml: str) -> None:
    """Stage 4 (H3) — base-vs-adapter Ref2VA renders at one seed, plus the automated floor (H3-06).

    ⛔ This does NOT extend ``inference/sampler.py``. That file is ltx-trainer's ``ValidationSampler``
    plus STG plus the two-stage upscaler — all LTX-only concepts. H3 is **guidance-distilled and
    single-pass**: no guidance scale, no negative branch, no STG, no upscaler, ONE forward per step
    (``P10-0e-DIFFUSERS-H3.md`` section 3). Importing that sampler here would drag in machinery with
    no H3 meaning and quietly invite someone to pass it a ``guidance_scale``.

    Base vs adapter is rendered from **one** PEFT-wrapped transformer, switching with PEFT's
    ``disable_adapter()``. That is not a convenience: 61.7 GiB twice does not fit on an A100-80GB, and
    it makes "identical seed, identical everything except the adapter" literally true rather than
    approximately true.

    References come from the SAME manifest and the SAME seeded rotation the pre-encode used
    (``_h3_resolve_references``). D-10-REFORDER is load-bearing twice over — it fixes the
    ``<Picture i>`` labels and advances the shared rotary clock — so a render must ask the question
    the adapter was taught, in the order it was taught it.

    Residency: the documented single-80GB recipe is a ``ComponentsManager`` with auto CPU offload,
    which is diffusers' own expression of the two-phase discipline — Qwen3-VL (62.1 GiB) and the
    transformer (61.7 GiB) are moved on and off the accelerator in turn and never coexist. Hand-
    freeing components underneath a ``ModularPipeline`` would fight the manager that owns them.

    Per D-10-SCOPEGUARD nothing here grades adapter QUALITY: the pass condition is that the render
    completes and the delta is non-zero. The operator-facing grid is assembled with
    **finetune-gridwatch** and served live over a tunnel — this function produces the mp4s and an
    index page, it does not replace gridwatch.
    """
    # ── (1) COLD-PATH IMPORT PROBE — before any load ─────────────────────────────────────────────
    try:
        import diffusers  # noqa: PLC0415
        import numpy  # noqa: PLC0415, F401 — probed for the same declared-or-probed reason as below
        import peft  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            f"[h3_sample] cold-path dependency missing ({exc.name!r}). h3_gpu_image must carry "
            "diffusers (at the pinned DIFFUSERS_SHA) / peft / numpy before any sustained GPU spend "
            "(T-03-SC)."
        ) from exc
    try:
        import av  # noqa: PLC0415, F401 — the mp4 muxer behind diffusers' encode_video
    except ImportError as exc:
        raise RuntimeError(
            f"[h3_sample] video-writer backend missing ({exc.name!r}). Writing the render to an mp4 "
            "goes through diffusers' encode_video, which muxes with PyAV. 10-04 shipped "
            "h3_gpu_image WITHOUT ffmpeg or a decode/encode package, because nothing on the H3 path "
            "demuxed anything at the time; 10-10 proved h3_preprocess needs it too and left the "
            "same probe. Adding a distribution is supply-chain-gated, so it is NOT done from this "
            "plan. Fix: add `av` (and ffmpeg via apt_install for the audio leg) to h3_gpu_image in "
            "modal/app.py and rebuild. Aborting here costs cents; discovering it after two 61.7 GiB "
            "renders does not."
        ) from exc

    import json  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from signet_trainer.conditioning.h3_geometry import (  # noqa: PLC0415
        max_packed_rows_for_budget,
    )
    from signet_trainer.conditioning.h3_packing import make_h3_position_ids_fn  # noqa: PLC0415
    from signet_trainer.conditioning.h3_ref import H3RefStrategy  # noqa: PLC0415
    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415
    from signet_trainer.data.precomputed import PrecomputedDataset  # noqa: PLC0415
    from signet_trainer.inference.grid import slug, write_comparison_gallery  # noqa: PLC0415
    from signet_trainer.lora.peft import (  # noqa: PLC0415
        build_lora_config,
        inject_lora,
        load_adapter_into,
    )
    from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415
    from signet_trainer.train.h3_step import h3_step_deps_from_model  # noqa: PLC0415

    print(f"[h3_sample] cold-path imports OK — diffusers={diffusers.__version__} peft={peft.__version__}")

    # ── (2) load + revalidate the config in-container ────────────────────────────────────────────
    config = load_config_from_text(config_yaml)
    if config.model.family != "h3":
        raise RuntimeError(
            f"[h3_sample] model.family is {config.model.family!r}, not 'h3' — this stage drives the "
            "diffusers MiniMax-H3 ref2va workflow, not the LTX pipelines."
        )
    prompts = list(config.validation.prompts)
    if not prompts:
        raise RuntimeError(
            "[h3_sample] config.validation.prompts is empty — nothing to render. The eval prompt "
            "set is FIRST-CLASS setup (settled at the session gate, never presumed), so this is a "
            "config error rather than something to default."
        )
    seed = int(config.validation.seed)
    device = "cuda"
    dtype = torch.bfloat16

    # ⛔ D-10-DEF-15 — the RENDER frame band, checked FIRST, before the manifest and before any
    # load. MiniMax-H3 generates 5-15 s at 24 fps and refuses anything else inside
    # `MiniMaxH3Ref2VASetupStep`, which runs AFTER both 61.7 GiB loads and the adapter injection —
    # the first dispatch paid exactly that and rendered zero clips. It is a DIFFERENT law from the
    # 17n+5 the training dims satisfy, so a legal training bucket (this campaign's 22) can be
    # unrenderable, and the fix is an eval-design decision rather than a bump.
    from signet_trainer.inference.h3_pipeline_source import (  # noqa: PLC0415
        assert_h3_frame_count_is_renderable,
    )

    print(
        assert_h3_frame_count_is_renderable(
            int(config.validation.frame_count), where="validation.frame_count"
        )
    )

    # ── (3) references: the SAME manifest rows and the SAME seeded rotation the pre-encode used ──
    from pathlib import Path  # noqa: PLC0415

    # The render-directory identity (D-10 resume). It lives in `inference/render_key.py`, stdlib
    # only, for the same reason `train/loop.checkpoint_watchdog_exceeded` lives outside this module:
    # a test that had to import `modal.fns` to reach it would drag `modal` into sys.modules and
    # break the dry-run gate's Anti-Pattern-6 assertion for the whole session.
    from signet_trainer.inference.render_key import h3_render_key  # noqa: PLC0415

    declared = Path(config.data.metadata_path)
    # The manifest path is Volume-relative by convention but the LTX configs carry it absolute;
    # accept both rather than silently resolving an absolute path under the mount and 404-ing.
    metadata_path = declared if declared.is_absolute() else DATASET_DIR / declared
    rows = _h3_manifest_rows(str(metadata_path))
    # WHICH row supplies the slots is a per-config decision (validation.reference_subject_ids);
    # HOW they are resolved is not — the selector calls the same _h3_resolve_references the
    # pre-encode did, so D-10-PAIRSEED and D-10-REFORDER are untouched. Empty = row 0, the
    # historical behaviour, so a config predating the field renders byte-identically.
    reference_row, reference_descriptors = _h3_select_reference_row(
        rows,
        metadata_path.parent,
        subject_ids=config.validation.reference_subject_ids,
        references_per_sample=config.h3.references_per_sample,
        reference_pair_seed=config.h3.reference_pair_seed,
        environment_ref_last=config.h3.environment_ref_last,
    )
    reference_paths = [r["source"] for r in reference_descriptors]
    selector = (
        f"validation.reference_subject_ids {list(config.validation.reference_subject_ids)}"
        if config.validation.reference_subject_ids
        else "the default (manifest row 0)"
    )
    print(
        f"[h3_sample] conditioning on {len(reference_paths)} reference slot(s) "
        f"{[r['subject_id'] for r in reference_descriptors]} from manifest row {reference_row}, "
        f"selected by "
        f"{selector} and resolved by the D-10-PAIRSEED rotation at seed "
        f"{config.h3.reference_pair_seed} — the SAME rule the pre-encode used, in the SAME "
        "D-10-REFORDER order, so the render asks the question the adapter was taught. Slots are "
        "named by subject_id, never by filename (client-property hygiene)."
    )

    # ── (4) the pipeline: diffusers' ref2va workflow, EVERY COMPONENT FROM THE MOUNTED VOLUME ────
    # ⛔ D-10-DEF-14. ``ModularPipeline.from_pretrained(<root>)`` is NOT used, and the reasons are
    # in `inference/h3_pipeline_source.py`'s docstring — in short: this checkpoint's index records
    # `pretrained_model_name_or_path: MiniMaxAI/MiniMax-H3` for every component, so the library's
    # own resolution egresses ~134 GiB from the HUB while standing on the Volume that holds them,
    # and its ONLY symptom is the bill. So the pipeline is built from the BLOCKS (which is where
    # the ref2va component list actually lives) and every component is loaded from a local path,
    # the same "resolve it under WEIGHTS_DIR yourself" idiom `h3_preprocess` already uses.
    from diffusers import ComponentsManager  # noqa: PLC0415
    from diffusers.modular_pipelines import ComponentSpec  # noqa: PLC0415
    from diffusers.modular_pipelines.minimax_h3 import (  # noqa: PLC0415
        MiniMaxH3Blocks,
        MiniMaxH3ImageReference,
        MiniMaxH3ModularPipeline,
    )
    from diffusers.utils.export_utils import encode_video  # noqa: PLC0415

    from signet_trainer.inference.h3_pipeline_source import (  # noqa: PLC0415
        H3_REF2VA_TRANSFORMER,
        assert_h3_components_loaded_locally,
        assert_h3_sources_are_local,
        read_h3_pipeline_index,
        resolve_h3_component_sources,
    )

    if not config.model.pipeline_root_id:
        raise RuntimeError(
            "[h3_sample] config.model.pipeline_root_id is unset. The render needs the pipeline "
            "ROOT (the dir holding model_index.json and every component partition); "
            "`model.model_id` names the transformer PARTITION inside it and means something "
            "different to h3_train / h3_loader, so it is NOT reused here (D-10-DEF-14). Add to the "
            "config's `model:` block:\n    pipeline_root_id: minimax-h3"
        )
    pipeline_root = WEIGHTS_DIR / config.model.pipeline_root_id
    index, index_path = read_h3_pipeline_index(pipeline_root)
    print(f"[h3_sample] pipeline index read from {index_path} (the mounted Volume, not the Hub).")

    manager = ComponentsManager()
    # No `pretrained_model_name_or_path=`: passing the root would re-introduce the Hub-sourced
    # specs. The blocks are the authority for WHICH components `ref2va` needs and WHAT class each
    # one is; the Volume is the authority for where they live.
    pipe = MiniMaxH3ModularPipeline(
        blocks=MiniMaxH3Blocks(), workflow="ref2va", components_manager=manager
    )
    needed = list(pipe.pretrained_component_names)
    sources = resolve_h3_component_sources(index, pipeline_root, needed)
    print(assert_h3_sources_are_local(sources, pipeline_root))
    # Loaded ONE AT A TIME through `ComponentSpec.load`, which RAISES. `pipe.load_components()`
    # catches every per-component exception and downgrades it to a `logger.warning`, leaving the
    # attribute None — so a mis-resolved component would surface as an AttributeError deep inside
    # the denoise loop, after both model loads had been paid for.
    for name in needed:
        source = sources[name]
        spec = pipe.get_component_spec(name)
        pipe.update_components(
            **{
                name: ComponentSpec(
                    name=name,
                    # The class comes from the BLOCKS' own spec, never re-derived here. A None
                    # type_hint would send `load()` through AutoModel, which cannot construct a
                    # tokenizer or a processor at all.
                    type_hint=spec.type_hint,
                    pretrained_model_name_or_path=str(pipeline_root),
                    subfolder=source.subfolder,
                ).load(dtype=dtype)
            }
        )
    # The second half of the egress guard: what the pipeline says it loaded, after loading it.
    print(
        assert_h3_components_loaded_locally(
            {n: pipe.get_component_spec(n).pretrained_model_name_or_path for n in needed},
            pipeline_root,
        )
    )
    # THE residency discipline for one 80 GiB card: the manager moves each component on and off the
    # accelerator in turn, so the 62.1 GiB conditioner is off the device (its text encoding already
    # done) before the 61.7 GiB transformer needs it. This is diffusers' own two-phase mechanism —
    # free_text_encoder's discipline expressed in the idiom of the library that owns the components.
    #
    # ⚠ WHAT `memory_reserve_margin` IS NOT (D-10-DEF-19, read off the pinned source, not inferred).
    # It does NOT reserve VRAM and nothing holds it open during the render. It is a one-shot
    # EVICTION THRESHOLD: `CustomOffloadHook.pre_forward` runs the strategy ONLY when the component
    # being called is not already on the device (components_manager.py:86-119), and
    # `AutoOffloadStrategy` subtracts the margin from `mem_get_info()[0]` purely to choose WHICH
    # OTHER components to push to CPU (:189). Once a component is resident, its activations may
    # consume every remaining byte — so this number cannot rescue an activation-peak OOM, and when
    # the strategy logs `no combination of models to offload to cpu is found, offloading all models`
    # (:233) it has already evicted everything and its effect is maximal. Config-first
    # (D-NOHARDCODE) because it is a per-GPU threshold, not a constant of the code.
    manager.enable_auto_cpu_offload(
        device=device, memory_reserve_margin=config.h3.render_offload_reserve
    )
    print(
        f"[h3_sample] {len(needed)} component(s) loaded with auto CPU offload — the text encoder is "
        "never resident alongside the transformer (two-phase VRAM discipline, the H3 arithmetic is "
        f"tighter than LTX's). Offload eviction threshold {config.h3.render_offload_reserve} — a "
        "swap-time threshold for choosing what to evict, NOT a standing VRAM reservation."
    )

    # ⚠ A DISTINCT name from `reference_descriptors`. These are pipeline objects with no
    # `subject_id`; rebinding the descriptor list here (as this function used to) made the
    # `delta.json` write at the very END of the render — after BOTH metered columns were paid for —
    # a TypeError waiting to happen. [Rule 1]
    references = [MiniMaxH3ImageReference.from_file(str(p)) for p in reference_paths]

    # ── (5) the adapter — find_latest, never a hand-glob or an un-numbered final ─────────────────
    # ⚠ signet finals ARE step-numbered: CheckpointManager writes per-STEP dirs and find_latest is the
    # ONLY correct resolution. A flat ``lora_weights.safetensors`` is the CANONICAL ltx-trainer
    # convention, not this stack's — looking for one here would silently render a BASE-only grid.
    checkpoints_vol.reload()
    ckpt_root = CHECKPOINTS_DIR / config.output_dir
    # ⚠ THE PIN EXISTS BECAUSE find_latest IS A MOVING TARGET WHILE A RUN IS LIVE (D-10-DEF-19).
    # The render directory is keyed on the render's IDENTITY and the checkpoint name is part of that
    # key — so against a training run that commits every `checkpoint_every` steps, each re-dispatch
    # resolves a DIFFERENT adapter, lands in a FRESH directory, and finds nothing to resume. The
    # resume logic is correct; it simply cannot survive an adapter that moves underneath it. Empty
    # (the default) is find_latest, byte-identically as before.
    pinned_name = config.h3.render_checkpoint_name
    if pinned_name:
        latest_ckpt = ckpt_root / pinned_name
        if not CheckpointManager.is_complete(latest_ckpt):
            available = sorted(p.name for p in ckpt_root.glob("checkpoint-step-*") if p.is_dir())
            raise RuntimeError(
                f"[h3_sample] h3.render_checkpoint_name pins {pinned_name!r} but "
                f"{latest_ckpt} is not a COMPLETE checkpoint dir (needs both the adapter and "
                f"training_state.pt — the same completeness filter find_latest applies, so a "
                f"half-written dir can never be pinned into a render either). Available: "
                f"{available}. Clear the pin to fall back to find_latest."
            )
        print(
            f"[h3_sample] checkpoint PINNED by h3.render_checkpoint_name -> {latest_ckpt.name} "
            "(find_latest NOT consulted). This is what lets a re-dispatch resume the same "
            "identity-keyed render dir while a training run keeps committing new checkpoints."
        )
    else:
        latest_ckpt = CheckpointManager(ckpt_root).find_latest()
    if latest_ckpt is None:
        raise RuntimeError(
            f"[h3_sample] no checkpoint under {ckpt_root} — find_latest() found no per-step "
            "adapter dir. Run the gated h3_train first; a base-vs-adapter grid with no adapter is "
            "not a comparison."
        )
    lora_config = build_lora_config(
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        targets=config.resolved_lora_targets(),
    )
    # ⛔ `transformer_ref`, NOT `transformer`. ONE repository holds BOTH checkpoint partitions under
    # DIFFERENT COMPONENT NAMES — `transformer` for t2va/fl2va, `transformer_ref` for ref2va — and
    # the ref2va denoise loop does `getattr(components, "transformer_ref")` (transcribed from the
    # pinned `minimax_h3/denoise.py`, where `MiniMaxH3Ref2VALoopDenoiser` is
    # `MiniMaxH3LoopDenoiser(transformer_name="transformer_ref")`). `pipe.transformer` is not the
    # same object under another name: it is a component this workflow never declares, so it is None
    # — `inject_lora(None, ...)`, one line after 61.7 GiB has been paid for. The name is imported
    # from `h3_pipeline_source` rather than spelled here so the constant has one home.
    base_transformer = getattr(pipe, H3_REF2VA_TRANSFORMER, None)
    if base_transformer is None:
        raise RuntimeError(
            f"[h3_sample] the ref2va pipeline has no {H3_REF2VA_TRANSFORMER!r} component after "
            f"loading (it holds {sorted(pipe.component_names)}). The adapter is injected into the "
            "partition the workflow denoises against; injecting into anything else renders a "
            "base-only grid under an adapter label."
        )
    adapted = inject_lora(base_transformer, lora_config)
    load_adapter_into(adapted, latest_ckpt)
    adapted.eval()  # inject_lora leaves the model in train mode + grad-checkpointing (TRAIN-06)
    pipe.update_components(**{H3_REF2VA_TRANSFORMER: adapted})
    # ⛔ THE OFFLOAD MARGIN IS SILENTLY RESET BY THE LINE ABOVE. Traced in the pinned source, not
    # inferred: `update_components` -> `register_components` -> `ComponentsManager.add()`, and an
    # nn.Module compares by IDENTITY, so the PEFT wrapper is `is_new_component=True` even though it
    # CONTAINS the transformer already registered. `add()` then does
    #     if self._auto_offload_enabled and is_new_component:
    #         self.enable_auto_cpu_offload(self._auto_offload_device)
    # — with NO `memory_reserve_margin`, which defaults to "3GB". The configured margin is thrown
    # away at exactly the moment the model gets bigger. Re-asserting it is safe: the first thing
    # `enable_auto_cpu_offload` does is strip every existing hook and call
    # `disable_auto_cpu_offload()`, so it is idempotent at the pin.
    #
    # ⛔⛔ AND IT DOES NOT SAVE YOU. D-10-DEF-19: re-asserting this margin was the first suspect for
    # the adapter-column OOM and it is NOT the cause. The margin only chooses WHAT TO EVICT at
    # swap time (see the block comment at the first enable_auto_cpu_offload above); the failing run
    # logged `no combination of models to offload to cpu is found, offloading all models` before
    # BOTH columns, i.e. only the transformer was ever resident and there was nothing left for a
    # larger margin to evict. The OOM is an ACTIVATION-peak problem inside a single forward, which
    # this number cannot reach. Keep the re-assert — it is correct and free — but do not reach for
    # it when a render runs out of memory mid-denoise.
    manager.enable_auto_cpu_offload(
        device=device, memory_reserve_margin=config.h3.render_offload_reserve
    )
    # ⚠ And the count is LOGGED because the same trace says the OLD bare transformer stays
    # registered alongside the wrapper containing it — the same 61.7 GiB counted twice by the
    # offload strategy's footprint math, both hooked. There is no removal in `add()`. (It costs no
    # VRAM: `get_peft_model` rewrites the transformer's Linears IN PLACE, so the wrapper and the
    # bare registration share one set of parameter tensors and `.to()` on either moves both.) This
    # number is the only way to see the double registration from a container log.
    print(
        f"[h3_sample] loaded the adapter from {latest_ckpt}; re-asserted the "
        f"{config.h3.render_offload_reserve} offload eviction threshold after update_components (it "
        f"silently reverts to the 3GB default) — components manager now holds "
        f"{len(manager.components)} component(s)."
    )

    # ── (6) THE AUTOMATED FLOOR — measured before any render, so a dead adapter aborts cheap ─────
    strategy = H3RefStrategy(
        references_per_sample=config.h3.references_per_sample,
        reference_dropout=0.0,  # the delta batch must never be the reference-dropped regime
        reference_pair_seed=config.h3.reference_pair_seed,
        environment_ref_last=config.h3.environment_ref_last,
        audio_in_loss=config.h3.audio_in_loss,
        t_visual_cond=config.h3.t_visual_cond,
        deps=h3_step_deps_from_model(
            adapted.base_model.model if hasattr(adapted, "base_model") else adapted
        ),
        # ⛔ ``training_dims[2]``, NOT ``validation.frame_count``. This strategy is built for
        # ``_h3_fixed_delta_batch``, which reads sample 0 out of the PRECOMPUTED TRAINING cache —
        # so the geometry it must agree with is the pre-encode's, not the render's. The render
        # count is a separate decision and is read separately, at ``_render``'s ``num_frames=``
        # below. ``config.data.frame_count`` (D-10-DEF-1) was an AttributeError that fired only
        # AFTER load_components + enable_auto_cpu_offload + inject_lora had been paid for.
        position_ids_fn=make_h3_position_ids_fn(
            target_frames=config.training_dims[2],
            target_aspect=tuple(config.h3.target_aspect),
            reference_short_edge=config.h3.reference_image_short_edge,
            # Read off the PEFT-wrapped ref2va partition (`base_transformer` is the same object
            # unwrapped) — `pipe.transformer` is a component this workflow does not have, and the
            # `hasattr` fallback made that a silent (1, 2, 2) guess rather than an error.
            patch_size=tuple(base_transformer.config.patch_size),
        ),
        patch_size=tuple(
            (adapted.base_model.model if hasattr(adapted, "base_model") else adapted).config.patch_size
        ),
        max_packed_rows=max_packed_rows_for_budget(
            config.h3.gpu_usable_gib, config.h3.resident_gib, config.h3.mib_per_packed_row
        ),
    )
    dataset = PrecomputedDataset(
        config.data.preprocessed_data_root,
        data_sources={
            name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[name]
            for name in strategy.get_data_sources()
        },
    )
    fixed = _h3_fixed_delta_batch(config, dataset, strategy, device, dtype)
    delta = h3_adapter_delta(adapted, fixed.packed.kwargs, fixed.packed.n_cond_video)
    print(
        f"[h3_sample] ADAPTER-MOVES-MODEL: max|delta velocity| base-vs-adapter = {delta:.3e} "
        "(the automated floor — it stands whether or not the operator has looked at the grid yet)."
    )
    if delta == 0.0:
        raise RuntimeError(
            f"[h3_sample] the adapter at {latest_ckpt} does NOT change the model output "
            "(max|delta velocity| == 0.0), so the two grid columns would be identical pixels with "
            "different labels — worse than no grid. D-10-SCOPEGUARD's acceptance criterion is that "
            "the adapter provably moves the model."
        )

    # ── (7) render base and adapter at the SAME seed ─────────────────────────────────────────────
    # ⛔ THE RENDER DIRECTORY IS KEYED ON THE RENDER'S IDENTITY, NOT ON THE WALL CLOCK.
    #
    # This is what makes the run RESUMABLE, and the loss unit is why it has to be. 12 clips render
    # SEQUENTIALLY and the old code committed once, at the END: a ~6 h unit with no retries, no
    # resume and commit-at-the-end is the exact shape KNOWLEDGE.md's `preemption no-resume` landmine
    # says cannot survive — already paid for on LTX embe r1, where a 2.2 h render on a ~24-min
    # preemption cycle could never finish. With a stable directory + `_h3_render_key`'s skip, a lost
    # container costs the clip in flight (~30 min) instead of everything, and both model loads are
    # not re-paid for work already on the Volume.
    #
    # ⚠⚠ THE IDENTITY MUST CARRY EVERY AXIS THE FIVE SAMPLE CONFIGS DIFFER ON, and getting this
    # wrong would be far worse than re-rendering. All five share `output_dir: outputs/h3_embe_r1`
    # (they must — find_latest resolves the adapter under it), share `seed: 42`, and share their
    # prompt set by design, so they all write the IDENTICAL filename `{slug}_s42.mp4`. A directory
    # keyed on checkpoint+seed alone would make the `B+029` run skip every clip because the `A+029`
    # run had already written that name — producing a grid LABELLED "reference only: B" containing
    # A's pixels. That is a silent mislabel on the precise axis this phase exists to measure.
    # So the key carries the checkpoint, the seed, the frame count AND the ordered reference
    # condition. `tests/test_h3_sample_resume.py` pins that the five configs land in five dirs.
    samples_root = (
        CHECKPOINTS_DIR
        / config.output_dir
        / "samples_h3"
        / h3_render_key(
            checkpoint=latest_ckpt.name,
            seed=seed,
            frame_count=config.validation.frame_count,
            subject_ids=[r["subject_id"] for r in reference_descriptors],
        )
    )
    base_dir, lora_dir = samples_root / "base", samples_root / "lora"
    base_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)
    print(f"[h3_sample] render dir {samples_root.name} (identity-keyed, so a re-dispatch RESUMES).")

    def _render(prompt: str, out_path: Any) -> str:
        """One single-pass H3 render. NO guidance scale and NO negative prompt — H3 is distilled.

        RESUME + COMMIT-PER-CLIP. An existing non-empty mp4 in the identity-keyed dir is work this
        exact render already did, so it is skipped; a fresh one is committed IMMEDIATELY, because
        commit-or-vanish means a clip that is not on the Volume did not happen.
        """
        if out_path.exists() and out_path.stat().st_size > 0:
            # Non-empty, not merely present: a container killed mid-`encode_video` leaves a 0-byte
            # file, and skipping THAT would put a corrupt clip in the grid rather than re-render it.
            print(f"[h3_sample] resume — {out_path.parent.name}/{out_path.name} already rendered.")
            return out_path.name
        result = pipe(
            prompt=prompt,
            references=references,
            num_frames=config.validation.frame_count,
            # ⛔ `validation.width` / `height` were BANNER-ONLY on this leg: nothing passed them, so
            # the ref2va workflow fell back to its own 16:9 canvas — 1344x768, which equals the
            # config today BY COINCIDENCE. Change either number and the render would silently stay
            # 1344x768 while the gallery banner (which reads v.width / v.height below) reported the
            # config's. A banner that describes a render nobody performed is the silent-at-a-valid-
            # shape class this phase keeps paying for. The workflow validates the /32 multiple.
            height=config.validation.height,
            width=config.validation.width,
            num_inference_steps=config.validation.num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
            output=["videos", "audio", "sampling_rate"],
        )
        encode_video(
            result["videos"][0],
            fps=config.validation.frame_rate,
            output_path=str(out_path),
            # ⛔ `is not None`, NEVER a bare truth test. The ref2va workflow's "audio" output is a
            # TENSOR `(1, 2, num_samples)`, not a list, and `if <multi-element tensor>` raises
            # "Boolean value of Tensor with more than one element is ambiguous" — on the FIRST
            # _render, i.e. after the base column's entire denoise loop has already been paid for.
            # The `[0]` is correct: it yields the `[2, samples]` encode_video documents.
            audio=result["audio"][0] if result.get("audio") is not None else None,
            audio_sample_rate=result.get("sampling_rate"),
        )
        # COMMIT-OR-VANISH, per clip. The old single commit at the end meant a preemption at clip 11
        # of 12 lost all eleven. This is the whole point of the identity-keyed dir above — a commit
        # into a wall-clock dir survives, but nothing would ever look in it again.
        checkpoints_vol.commit()
        return out_path.name

    def _peak_gib() -> float:
        """Peak CUDA bytes allocated since the last reset, in GiB — then reset the watermark.

        D-10-DEF-19 left the measured base-column peak unknown, so the shortfall had to be
        reconstructed from an OOM message. One number per column turns the next residency question
        into data instead of arithmetic, and it costs nothing (a counter read).
        """
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        torch.cuda.reset_peak_memory_stats()
        return peak

    torch.cuda.reset_peak_memory_stats()
    base_mp4s: dict[str, str] = {}
    with adapted.disable_adapter():
        for prompt in prompts:
            name = _render(prompt, base_dir / f"{slug(prompt)}_s{seed}.mp4")
            base_mp4s[prompt] = f"base/{name}"
    print(
        f"[h3_sample] BASE column: {len(base_mp4s)} clip(s) at seed {seed}; peak allocated "
        f"{_peak_gib():.2f} GiB of {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} "
        "GiB (0.00 means every clip resumed from the Volume and nothing was rendered)."
    )

    # ── (7b) ⛔ MERGE BEFORE THE ADAPTER COLUMN — D-10-DEF-19, THE WHOLE REASON THIS RENDER FITS ──
    #
    # The base column above proves the geometry fits on one A100-80GB. The adapter column is not a
    # capacity problem, it is an ARITHMETIC one, and merging removes it exactly.
    #
    # UNMERGED, every target Linear runs peft 0.20.0 `lora/layer.py:1058`:
    #     result = result + lora_B(lora_A(dropout(x))) * scaling
    # MEASURED (render app ap-ENw0kWTvmlfm4WhoYtc7KT): the run wrote 6/6 base clips, then died in
    # transformer block 0 of denoise step 0 with `Tried to allocate 6.88 GiB / 6.70 GiB free /
    # 72.54 GiB in use / 103.23 MiB reserved-but-unallocated` — the deepest frame being `lora_B`'s
    # own `F.linear`, beneath the SwiGLU up-projection `ff.net.0.proj`
    # (`diffusers/models/activations.py:144`), the widest LoRA target in the H3 block. 103 MiB
    # unallocated means this is NOT fragmentation, and the strategy had already logged `offloading
    # all models`, so it is NOT residency either: nothing but the transformer was on the card.
    # INFERRED from the expression above (not measured): the shortfall is about TWICE the failing
    # allocation, because `lora_B`'s output, the `* scaling` temporary and the `+` result are each
    # the shape of `result` — and the base path allocates NONE of them.
    #
    # MERGED, the very same layer takes peft's other branch:
    #     elif self.merged: result = self.base_layer(x, *args, **kwargs)
    # — byte-for-byte the base column's allocation profile, which just rendered six clips. The delta
    # is folded into `base_layer.weight` once, in WEIGHT space, where the transient is one weight
    # matrix rather than one activation tensor.
    #
    # ORDER IS LOAD-BEARING and this is why the merge lives HERE and not next to `inject_lora`:
    #   * the base column must render UNMERGED under `disable_adapter()` (peft would otherwise
    #     lazily `unmerge()` inside the forward — correct, but it would re-pay the merge twice);
    #   * `h3_adapter_delta` above must measure base-vs-adapter on the UNMERGED model, since
    #     `disable_adapter()` is how it gets its base pass;
    #   * merging mutates weights already loaded in RAM, and the offload hooks only ever `.to()`
    #     them — nothing re-reads the safetensors — so the merge survives every CPU<->GPU cycle for
    #     the rest of the render.
    if config.h3.render_merge_adapter:
        merge_device = next(adapted.parameters()).device
        adapted.merge_adapter()
        # PROVE it took. `merge_adapter` swallows a no-op into a `warnings.warn` (peft
        # tuners_utils.check_adapters_to_merge), and a silently-unmerged model does not fail here —
        # it fails ~20 minutes later, on the metered A100, with the identical OOM this exists to
        # prevent. Cheap assertion, expensive alternative.
        merged_layers = sum(
            1 for m in adapted.modules() if getattr(m, "merged", False) and hasattr(m, "lora_A")
        )
        if merged_layers == 0:
            raise RuntimeError(
                "[h3_sample] merge_adapter() reported no merged LoRA layers. The adapter column "
                "would render at the UNMERGED activation cost that OOMs on one A100-80GB "
                "(D-10-DEF-19), or — worse — render base pixels under an adapter label. Refusing "
                "before the metered column rather than after it."
            )
        print(
            f"[h3_sample] adapter MERGED into the base weights on {merge_device} across "
            f"{merged_layers} LoRA layer(s) (D-10-DEF-19). The adapter column now allocates exactly "
            "what the base column allocated: peft takes its `elif self.merged` branch, so there is "
            "no lora_B activation and no delta add. Unmerged, `ff.net.0.proj` alone wants two extra "
            "6.88 GiB tensors per block and the card has ~6.7 GiB spare."
        )
    else:
        print(
            "[h3_sample] ⚠ h3.render_merge_adapter is False — the adapter column renders UNMERGED. "
            "On one A100-80GB at this geometry that OOMs (D-10-DEF-19); this path is only viable "
            "with >=14 GiB of headroom beyond the base column's measured peak above."
        )

    lora_mp4s: dict[str, str] = {}
    for prompt in prompts:
        name = _render(prompt, lora_dir / f"{slug(prompt)}_s{seed}.mp4")
        lora_mp4s[prompt] = f"lora/{name}"
    print(
        f"[h3_sample] LoRA column: {len(lora_mp4s)} clip(s) from {latest_ckpt.name}; peak allocated "
        f"{_peak_gib():.2f} GiB."
    )

    # ── (8) the montage — the EXISTING writer, reused as-is (pure mp4-path templating) ───────────
    v = config.validation
    index_path = write_comparison_gallery(
        [
            {
                "prompt": p,
                "seed": seed,
                "base_mp4": base_mp4s.get(p, ""),
                "lora_mp4": lora_mp4s.get(p, ""),
            }
            for p in prompts
        ],
        samples_root / "index.html",
        params={
            "steps": v.num_inference_steps,
            # H3 is guidance-distilled and single-pass: there is no guidance scale and no STG to
            # report. Saying "n/a" is the truthful banner; echoing the LTX fields would claim a
            # sampling setting this render never had.
            "guidance": "n/a (H3 is guidance-distilled)",
            "stg_scale": "n/a",
            "width": v.width,
            "height": v.height,
            "frames": v.frame_count,
            "lora_scale": 1.0,
        },
    )
    (samples_root / "delta.json").write_text(
        json.dumps(
            {
                "max_abs_delta_velocity": delta,
                "checkpoint": str(latest_ckpt),
                "seed": seed,
                # subject_ids, NOT filenames: this is what a budget refusal and the training
                # preflight both name, it is the config's declared vocabulary, and it keeps
                # client-property filenames out of an artifact that gets shared.
                "references": [r["subject_id"] for r in reference_descriptors],
                # Which manifest row supplied them, so two sibling renders that differ ONLY in
                # their reference condition can be told apart from the artifact alone.
                "reference_row": reference_row,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ── (9) commit-or-vanish: an uncommitted samples dir is lost on container exit ────────────────
    checkpoints_vol.commit()
    print(
        f"[h3_sample] wrote {len(prompts)} comparison row(s) -> {index_path} and delta.json "
        "(committed to signe-trainer-checkpoints). Assemble the operator grid with "
        "finetune-gridwatch and serve it live over a tunnel — this page is the artifact index, not "
        "a replacement for gridwatch."
    )


# ==================================================================================================
# Phase 11 (family #3) — the Qwen-Image-Edit-2511 chained-edit leg: the architecture gate + the three
# gated stages (``qwen_edit_preprocess`` / ``qwen_edit_train`` / ``qwen_edit_sample``).
# ==================================================================================================
#
# ⛔ THE ARCH GATE IS A PLAIN HELPER, NOT A STAGE OF ITS OWN — the SAME refusal recorded above
# ``run_h3_arch_gate`` (fns.py:2564-2582), for the same reason, and repeated here rather than
# cross-referenced because the temptation ("just add a cheap arch-smoke stage") recurs per family. A
# decorated function is reachable via ``modal run -m signet_trainer.modal.fns::<name>``, and THAT
# invocation style boots a metered A100 with NO cost print and NO approval pause. Phase 11 adds no
# second ungated entry point.
#
# ⛔ AND NO NEW ``--mode``. All three stages route on ``cfg.model.family`` INSIDE the existing six
# ``--mode`` arms, exactly as H3 does — "six dispatches, one gate, one ledger"
# (``entrypoint.py``'s ``main`` docstring). Two structural tests pin the mode set at six
# (``tests/test_skill_entrypoint_coverage.py``, ``tests/test_h3_entrypoint_gate.py``) and they are
# right to: a second launch path is a second cost line and a second place for the ledger to drift.
# --------------------------------------------------------------------------------------------------


def _qwen_edit_require_backend(*modules: str) -> None:
    """Import each signet-side qwen_edit module by NAME, raising an actionable error on the first gap.

    The cheapest possible failure, run FIRST inside every qwen stage — before the third-party probe,
    before any weight load, before a single image is opened. It exists because family #3 landed as
    several independent slices and the Modal wiring is the one that spends money: a module that has
    not landed yet must abort in the first second of a container, naming itself, rather than surface
    as a ``ModuleNotFoundError`` traceback after a 40 GiB load.

    ``entrypoint._qwen_edit_stage_readiness`` runs the IDENTICAL check LOCALLY, before the dispatch,
    so in practice a gap costs $0 and this copy never fires. It is kept anyway for the reason
    ``modal/app.py``'s ``download_image`` INVARIANT banner states in one sentence: *a passing local
    gate proves nothing about the container's site-packages*. The two are cheap and independent.
    """
    import importlib  # noqa: PLC0415

    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            raise RuntimeError(
                f"[qwen-edit] required module {name!r} is not importable in this container "
                f"({exc}). Family #3's Modal wiring calls into it directly; the stage is aborting "
                f"at its first statement rather than after a weight load. If the module exists "
                f"locally but not here, the image is stale — ``qwen_gpu_image`` copies the package "
                f"with ``add_local_python_source`` and must be rebuilt. If it does not exist yet, "
                f"it is a DECLARED GAP: land it before dispatching this stage."
            ) from exc


def _qwen_edit_cold_path_probe(where: str, *, need_quanto: bool = True) -> str:
    """Verify the third-party closure BEFORE any load and return the one-line version banner (T-03-63).

    No installs (supply-chain discipline): this VERIFIES presence. A missing dependency means
    ``qwen_gpu_image`` must declare it in ``modal/app.py`` and be rebuilt, re-gated by the same
    supply-chain rules ``LTX2_COMMIT_SHA`` / ``DIFFUSERS_SHA`` / ``QWEN_DIFFUSERS_SHA`` carry.

    ``optimum.quanto`` is probed by DEFAULT because the house recipe locks ``qfloat8`` on both the
    transformer and the text encoder — it is not optional equipment on this family. It is a flag
    only so a stage that genuinely never quantizes anything is not aborted on a dependency it does
    not use, which is the mistake ``h3_train`` records for ``wandb``.
    """
    try:
        import diffusers  # noqa: PLC0415
        import peft  # noqa: PLC0415
        import transformers  # noqa: PLC0415
        from PIL import Image as _PILImage  # noqa: PLC0415, F401
    except ImportError as exc:
        raise RuntimeError(
            f"[{where}] cold-path dependency missing ({exc.name!r}). ``qwen_gpu_image`` must carry "
            "diffusers (at the pinned QWEN_DIFFUSERS_SHA) / transformers (at "
            "QWEN_TRANSFORMERS_VERSION) / peft / pillow BEFORE any sustained GPU spend — an "
            "ImportError discovered after the 40.9 GiB transformer load wastes a metered launch. "
            "Fix: add it to qwen_gpu_image in modal/app.py and rebuild (T-03-SC)."
        ) from exc

    quanto_version = "not probed"
    if need_quanto:
        try:
            import optimum.quanto as _quanto  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                f"[{where}] optimum.quanto is missing ({exc.name!r}). The house recipe quantizes "
                "BOTH the transformer and the Qwen2.5-VL text encoder to qfloat8 and "
                "``QwenEditConfig``'s docstring records that this is deliberately NOT a config "
                "knob — so there is no un-quantized fallback to degrade to. ``qwen_gpu_image`` pins "
                "``optimum-quanto==QWEN_OPTIMUM_QUANTO_VERSION``; rebuild it (T-03-SC)."
            ) from exc
        quanto_version = getattr(_quanto, "__version__", "?")

    line = (
        f"[{where}] cold-path imports OK — diffusers={diffusers.__version__} "
        f"transformers={transformers.__version__} peft={peft.__version__} "
        f"optimum-quanto={quanto_version}"
    )
    print(line)
    return line


def _qwen_edit_load_processor(processor_path: str, *, local_files_only: bool = True) -> Any:
    """Load the Qwen2.5-VL PROCESSOR from the mounted weights Volume (Modal-side ONLY).

    The one component ``models/qwen_edit_loader.py`` deliberately does not own: it loads MODELS, and
    a processor is a tokenizer + image processor pair with no weights and no arch to assert. It lives
    here for the same reason ``_h3_load_component`` does — it is mount-and-construct plumbing that
    only a Modal stage needs.

    ``AutoProcessor`` dispatches on the checkpoint's own ``processor_class`` / ``model_type``, so the
    concrete class (``Qwen2_5_VLProcessor``) is READ from the mounted files rather than restated
    here. Restating it would be a re-derived family fact, which ``models/qwen_edit_loader.py`` is the
    single source of.

    ⚠ ``local_files_only=True`` by default, matching ``load_qwen_edit_text_encoder``: a load inside a
    metered container must not silently depend on hub reachability, and the failure a hub round-trip
    produces here is not an exception but egress on the bill.

    The CALLER composes the path (``WEIGHTS_DIR / cfg.model.text_encoder_id``) — D-NOHARDCODE.
    """
    from transformers import AutoProcessor  # noqa: PLC0415

    return AutoProcessor.from_pretrained(processor_path, local_files_only=local_files_only)


def run_qwen_edit_arch_gate(
    checkpoint_path: str,
    *,
    device: str = "cuda",
    dtype: Any = None,
    config_source: str | None = None,
    subfolder: str | None = None,
    text_embed_dim: int | None = None,
    model: Any = None,
    release: bool = False,
) -> tuple[str, Any]:
    """Assert the Qwen-Image-Edit architecture on LIVE weights before the caller spends.

    The family-#3 sibling of ``run_h3_arch_gate``, deliberately the same SHAPE — plain helper, called
    unconditionally and FIRST inside every gated qwen stage, no flag that skips it and none that
    stops after it. A cost line is only truthful if the function it prices always does the same work.

    What it does, in order:

      1. the COLD-PATH IMPORT PROBE, BEFORE any model load (an ImportError discovered after a
         40,861,031,560-byte load wastes a metered launch);
      2. loads (or accepts) the transformer;
      3. PRINTS every arch field as ``name expected X got Y OK/MISMATCH`` — the print is the artifact
         an operator diffs against the measured header read, not a debug aid — then asserts,
         reporting EVERY offending field at once. ``assert_qwen_edit_arch`` owns that behaviour;
      4. surveys the 14-leaf path regex over the live ``named_modules`` and RAISES unless it resolves
         EXACTLY ``60 x 14 = 840`` modules, PER LEAF. Per-leaf and never on the grand total alone:
         a grand total hides a per-leaf ZERO, and on a DUAL-STREAM MMDiT a per-leaf zero has a
         specific shape — the four ``txt_*`` attention projections and the two ``*_mod.1``
         modulation projections, i.e. exactly the six a ported six-leaf LTX intuition drops. That
         adapter would train, converge, and be unloadable by every later round of the chain.

    It deliberately does NOT inject LoRA, does NOT quantize, does NOT run a forward pass and does NOT
    touch the text encoder or the VAE. Every number it checks is IMPORTED from
    ``models/qwen_edit_loader.py`` (which measured them off the real safetensors header) and
    ``config/validators.py`` (the single source of the leaf list and the regex), so nothing here
    re-derives a family fact.

    Args:
        checkpoint_path: the ``.safetensors`` FILE or diffusers DIRECTORY on the mounted weights
            Volume. The CALLER composes it (``WEIGHTS_DIR / cfg.model.model_id``), so this function
            holds no path literal of its own (D-NOHARDCODE).
        device: torch device for the load; ``"cuda"`` Modal-side.
        dtype: component dtype; ``None`` lets the loader resolve bf16 (the stored dtype).
        config_source: REQUIRED on the single-file path — see ``load_qwen_edit_transformer``'s
            docstring for the SD-1.5-config-fetch failure it prevents. Threaded from
            ``cfg.model.pipeline_root_id`` by the caller.
        subfolder: passed through to diffusers when ``config_source`` names a pipeline root.
        text_embed_dim: ``cfg.qwen_edit.text_embed_dim``, forwarded to ``assert_qwen_edit_arch`` so
            the config's [UNVERIFIED] 3584 is CHECKED against the live ``txt_in`` rather than
            believed. This is the single most valuable thing the gate does for this family: the
            config field's own description says the number is carried as a declared assumption and
            that the weight-loading pass must assert it.
        model: an ALREADY-LOADED transformer. Pass it and the gate costs nothing beyond the probe —
            how ``qwen_edit_train`` avoids paying for a second 40.9 GiB load.
        release: drop the gate's OWN reference before returning, so the transformer never coexists
            with the Qwen2.5-VL encoder loaded next. ``qwen_edit_preprocess`` needs this;
            ``qwen_edit_train`` does not (it trains the very model the gate just proved).

    Returns:
        ``(summary_line, model)`` — a one-line summary in ``run_h3_arch_gate``'s style, plus the
        proved model, or ``None`` in its place when ``release=True``.
    """
    # ── (0) the signet-side modules this gate calls into, named before anything is loaded ─────────
    _qwen_edit_require_backend("signet_trainer.models.qwen_edit_loader")

    # ── (1) COLD-PATH IMPORT PROBE — before ANY model load ────────────────────────────────────────
    # ``need_quanto`` is True: every caller of this gate goes on to quantize, and finding the qfloat8
    # backend missing AFTER the load is the exact waste the probe exists to prevent.
    _qwen_edit_cold_path_probe("qwen-edit-arch-gate")

    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from signet_trainer.models.qwen_edit_loader import (  # noqa: PLC0415
        EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT,
        assert_qwen_edit_arch,
        assert_qwen_edit_targets,
        expected_qwen_edit_arch,
        load_qwen_edit_transformer,
        summarize_qwen_edit_transformer,
    )

    if model is not None and release:
        raise ValueError(
            "[qwen-edit-arch-gate] release=True is only valid for a model the gate LOADED itself: "
            "it can drop only its OWN reference, and loader-owned CUDA storage is freed just when "
            "the LAST reference goes away. A caller-supplied model must be released by that caller."
        )

    # ── (2) the load — the caller's model when given, so the run pays for exactly one ────────────
    if model is not None:
        transformer = model
    else:
        transformer = load_qwen_edit_transformer(
            checkpoint_path,
            device=device,
            dtype=dtype,
            config_source=config_source,
            subfolder=subfolder,
        )
        allocated = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        print(
            f"[qwen-edit-arch-gate] loaded the Qwen-Image-Edit transformer from {checkpoint_path} "
            f"— cuda allocated={allocated:.2f} GiB (the checkpoint on disk is 40,861,031,560 B / "
            f"~38.1 GiB across 1,934 bf16 tensors)."
        )

    # ── (3) every arch field: PRINT, then assert ─────────────────────────────────────────────────
    summary = summarize_qwen_edit_transformer(transformer)
    printable: dict[str, Any] = dict(expected_qwen_edit_arch())
    # The four LIVE facts summarize_* adds on top of the config fields. Named here only so the print
    # covers them too; ``assert_qwen_edit_arch`` is what actually checks them (and knows their
    # expected values), so no expectation is restated on this side.
    for live_field in ("live_transformer_blocks", "img_in_shape", "txt_in_shape", "proj_out_shape"):
        printable.setdefault(live_field, "(see assert_qwen_edit_arch)")
    for field, want in printable.items():
        got = summary.get(field)
        if got is None:
            verdict = "SKIPPED (probe returned None)"
        elif isinstance(want, str):
            verdict = "(live)"
        else:
            comparable = tuple(got) if isinstance(want, tuple) else got
            verdict = "OK" if comparable == want else "*** MISMATCH ***"
        print(f"[qwen-edit-arch-gate]   {field:<26} expected {want!s:<20} got {got!s:<20} {verdict}")
    # Raises naming EVERY offending field, including the config's declared text_embed_dim when the
    # caller supplied it. Not stopping at the first mismatch is the enochiatron lesson h3 records.
    assert_qwen_edit_arch(summary, config_text_embed_dim=text_embed_dim)

    # ── (4) the LoRA target survey over the LIVE named_modules — per leaf, never a grand total ────
    survey = assert_qwen_edit_targets(transformer)
    for leaf, count in survey["per_leaf"].items():
        print(f"[qwen-edit-arch-gate]   {leaf:<20} matched={int(count):<4}")
    print(
        f"[qwen-edit-arch-gate] LoRA targets OK — total={int(survey['total'])} "
        f"(expected {EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT}), collateral={int(survey['collateral'])}. "
        "The 240 attn.norm_{q,k,added_q,added_k} RMSNorms are [128]-shaped and match none of the "
        "fourteen leaves — they are NOT targets and their absence here is the correct result."
    )

    line = (
        f"[qwen-edit-arch-gate] OK — blocks={summary.get('live_transformer_blocks')} "
        f"in_ch={summary.get('in_channels')} patch={summary.get('patch_size')} "
        f"joint_dim={summary.get('joint_attention_dim')} "
        f"heads={summary.get('num_attention_heads')}x{summary.get('attention_head_dim')} "
        f"guidance_embeds={summary.get('guidance_embeds')} img_in={summary.get('img_in_shape')} "
        f"txt_in={summary.get('txt_in_shape')} proj_out={summary.get('proj_out_shape')} "
        f"lora_targets={int(survey['total'])}/{EXPECTED_QWEN_EDIT_LORA_MODULE_COUNT}"
    )
    print(line)

    if release:
        # The gate's own reference is the LAST one only because ``model`` was None (asserted above),
        # so dropping it here is what actually frees the weights — ``Module.to("cpu")`` would not.
        del transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(
                "[qwen-edit-arch-gate] released the gated transformer (two-phase VRAM discipline) "
                f"— cuda allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB. The Qwen2.5-VL "
                "encoder is loaded ONLY after this point."
            )
        return line, None

    return line, transformer


# --------------------------------------------------------------------------------------------------
# ``qwen_edit_preprocess`` support. The ENCODE all lives in ``prep/qwen_edit_encode.py`` — everything
# here is the manifest walk / pairing plumbing the Modal stage needs and nothing else. Every backend
# import stays FUNCTION-LOCAL (Anti-Pattern 6).
# --------------------------------------------------------------------------------------------------


def _qwen_edit_output_rel(media_path: str) -> Any:
    """Manifest media path -> the rel path naming this sample's three cached outputs.

    Delegates to ``_h3_output_rel``: signet manifests carry ``media_path`` RELATIVE to the manifest's
    parent (``data/dataset_file.py``'s writer shape) and ``PrecomputedDataset`` pairs sources by that
    same relative path, so the derivation is family-AGNOSTIC and re-implementing it per family is how
    the two drift. The divergence does not raise — a mismatched rel simply drops the sample from the
    index — which is exactly why it is one shared helper.
    """
    return _h3_output_rel(media_path)


@app.function(
    # The Qwen family image (Phase 11): diffusers at QWEN_DIFFUSERS_SHA + transformers at
    # QWEN_TRANSFORMERS_VERSION + optimum-quanto, NOT the LTX ltx-core stack and NOT h3's diffusers
    # SHA. A ``gpu=`` with the code-only default image boots an A100 and dies at ``import torch``
    # (``tests/test_modal_gpu_image.py``).
    gpu="A100-80GB",
    image=qwen_gpu_image,
    volumes={**WEIGHTS_MOUNT, **DATASET_MOUNT},
    secrets=[huggingface_secret],
    # The same RAM request the H3 stages carry. The transformer is ~38.1 GiB on disk and this stage
    # holds it only for the gate (released before PHASE A), but the Qwen2.5-VL encoder that follows
    # is a 9,384,670,680-byte checkpoint whose VISION half is load-bearing — asking for 80 GiB and
    # allowing 200 keeps a host-RAM spike from being discovered as an OOM kill mid-encode.
    memory=(80 * 1024, 200 * 1024),
    timeout=TWENTY_FOUR_HOURS,
)
def qwen_edit_preprocess(
    metadata_path: str,
    output_dir: str,
    control_dirs: tuple[str, ...],
    control_slots: int,
    blank_slots: tuple[int, ...],
    blank_slot_fill: str,
    control_area_px: int,
    condition_area_px: int,
    control_cache_key_mode: str,
    cache_text_embeddings: bool,
    text_embed_dim: int,
    target_width: int,
    target_height: int,
    model_id: str,
    vae_id: str,
    text_encoder_id: str,
    pipeline_root_id: str | None,
) -> str:
    """Stage 1 (qwen_edit) — the signet-NATIVE Qwen-Image-Edit pre-encode, cached to the dataset Volume.

    The thin gated wrapper around ``prep/qwen_edit_encode.py``, which owns every encode decision and
    every parity citation. This function owns only what a Modal stage can own: the manifest walk, the
    two-phase residency discipline, the loud-failure guard and the commit.

    EVERY parameter is REQUIRED with no default — the ``h3_preprocess`` contract, for the same
    reason: with no defaults a threading gap is a ``TypeError`` at dispatch, before a container is
    even allocated, instead of a silent wrong default inside a paid one. ``entrypoint.
    _qwen_edit_encode_params`` supplies all of them from the validated config, and never a
    ``SignetConfig`` object nor a path into ``configs/`` (that dir is not in the image).

    Body order is strict:

      0. ``_qwen_edit_require_backend`` — the signet-side modules, named before anything loads;
      1. ``run_qwen_edit_arch_gate`` with ``release=True`` — UNCONDITIONAL, before a single image is
         opened, and it frees the transformer before the text encoder is loaded. It carries its own
         cold-path probe, so a missing dependency or a mismatched architecture aborts at the cheapest
         possible point. No flag skips it and none stops after it;
      2. PHASE A — load the Qwen2.5-VL processor + text encoder (vision tower ASSERTED present),
         quantize to qfloat8, and write ``qwen_edit_conditions/``. The VISION half is not optional:
         the pipeline passes ``pixel_values`` + ``image_grid_thw`` and a plain text LLM in this slot
         fails with ``mat1 and mat2 shapes cannot be multiplied (5376x1280 and 3840x1280)`` — a real
         failure this house already hit and fixed;
      3. ``free_qwen_edit_text_encoder`` + the caller-side reference drop, printing the freed delta;
      4. PHASE B — load the VAE and write ``qwen_edit_latents/`` + ``qwen_edit_control_latents/``;
      5. ``assert_qwen_edit_cache_complete`` — the LOUD-FAILURE guard. A source that produced zero
         files, or a rel path present under one source and absent under another, RAISES. It has to:
         ``PrecomputedDataset`` pairs by rel path and a missing file silently drops the whole sample
         from the index rather than raising;
      6. ``dataset_vol.commit()`` — commit-or-vanish (Pitfall 3), non-negotiable.

    ⚠ THE SAME CONTROL IMAGE IS PREPARED TWICE, FROM THE SOURCE, AT TWO DIFFERENT BUDGETS —
    ``condition_area_px`` (384², the Qwen2.5-VL channel) and ``control_area_px`` (1024², the VAE
    channel). That is not redundancy and it is the easiest thing to get wrong on this family;
    ``prepare_qwen_edit_image`` RAISES on an attempt to re-fit an already-prepared image at the other
    budget, which is what makes the mistake impossible rather than merely documented.

    ⚠ ORIENTATION is already ruled and this stage inherits the ruling: signet is DIFFUSERS-CORRECT
    (``ratio = W/H``), where ai-toolkit computes ``H/W`` and transposes every non-square control. The
    two agree exactly on SQUARE sources, so a non-square corpus is the first place the choice is
    observable — ``prepare_qwen_edit_image`` logs it at WARNING per image for exactly that reason.
    Adapters trained under the two are not numerically interchangeable on non-square controls.
    """
    # ── (0) the signet-side modules, named before anything is loaded ──────────────────────────────
    _qwen_edit_require_backend(
        "signet_trainer.models.qwen_edit_loader",
        "signet_trainer.prep.qwen_edit_encode",
    )

    import gc  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from signet_trainer.conditioning.qwen_edit_geometry import (  # noqa: PLC0415
        qwen_edit_area_budget_size,
        qwen_edit_rows_of,
    )
    from signet_trainer.models.qwen_edit_loader import (  # noqa: PLC0415
        assert_qwen_edit_text_encoder_vision,
        load_qwen_edit_text_encoder,
        load_qwen_edit_vae,
        quantize_qwen_edit,
    )
    from signet_trainer.prep.qwen_edit_encode import (  # noqa: PLC0415
        QWEN_EDIT_CONDITIONS_DIR,
        assert_qwen_edit_cache_complete,
        encode_qwen_edit_control_latents,
        encode_qwen_edit_target_latents,
        encode_qwen_edit_text_conditions,
        free_qwen_edit_text_encoder,
        prepare_qwen_edit_image,
        qwen_edit_control_identity,
        qwen_edit_text_cache_key,
        qwen_edit_vae_latent_stats,
        resolve_qwen_edit_control_sources,
        write_qwen_edit_precomputed,
    )

    # ── (1) THE ARCH GATE — unconditional, first, and it releases the transformer before the
    # Qwen2.5-VL encoder is loaded below (they must not coexist).
    gate_line, _gated = run_qwen_edit_arch_gate(
        str(WEIGHTS_DIR / model_id),
        device="cuda",
        config_source=str(WEIGHTS_DIR / pipeline_root_id) if pipeline_root_id else None,
        text_embed_dim=text_embed_dim,
        release=True,
    )
    print(f"[qwen_edit_preprocess] arch gate passed -> {gate_line}")

    rows = _h3_manifest_rows(metadata_path)
    data_root = Path(metadata_path).parent
    output_root = Path(output_dir)
    # The geometry this stage actually PRODUCES, reported straight out of the Modal log so the cache
    # it writes is auditable without re-deriving anything. Deliberately NOT ``qwen_edit_packed_
    # layout``: that function prices the TRAINING sequence (it needs ``prompt_tokens_estimate``, a
    # number this stage does not receive and would have to fake), and the dry-run gate already
    # printed it. What an ENCODER can honestly report is the row count each of its two channels
    # yields, and the two budgets are different numbers on purpose — conflating them is the family's
    # easiest mistake.
    # ⛔ THE TARGET'S BUDGET IS ITS OWN PIXEL AREA, not ``control_area_px``. The target is the image
    # being generated and its size is the training canvas; fitting it to the CONTROL budget would
    # silently re-size the thing the loss is computed against whenever the two differ (they coincide
    # only at the shipped square 1024x1024). Passing its own area makes ``prepare_qwen_edit_image``
    # a pure /32 snap, which is what the canvas already satisfies.
    target_area_px = int(target_width) * int(target_height)
    target_rows = qwen_edit_rows_of(
        *qwen_edit_area_budget_size(target_width, target_height, target_area_px)
    )
    print(
        f"[qwen_edit_preprocess] {len(rows)} sample(s); target {target_width}x{target_height} "
        f"({target_area_px} px) -> {target_rows} packed row(s); {control_slots} control slot(s) "
        f"from {list(control_dirs)} (blank: {list(blank_slots) or 'none'}, fill "
        f"{blank_slot_fill!r}). Each control image is prepared TWICE, FROM THE SOURCE, at two "
        f"different budgets — {control_area_px} px for the VAE channel and {condition_area_px} px "
        f"for the Qwen2.5-VL channel. Per-slot row counts follow each SOURCE image's own aspect and "
        f"are therefore reported by the encoder, not predicted here."
    )

    # ── PHASE A — text conditions. The processor + encoder have the card essentially to themselves.
    #
    # ⛔ The PROCESSOR comes from the PIPELINE ROOT, the encoder from the text_encoder partition.
    # They are different directories and conflating them is a measured failure, not a theoretical
    # one: a Qwen-Image-Edit-2511 snapshot writes preprocessor_config.json into `processor/`, so
    # AutoProcessor.from_pretrained(WEIGHTS_DIR / text_encoder_id) dies with
    #   OSError: Can't load image processor for '/weights/qwen-image-edit-2511/text_encoder'
    # AFTER the 38 GiB arch-gate load has already been paid for. The root is required rather than
    # guessed-with-a-fallback because a silent fallback is how the wrong processor gets used: the
    # image-preprocessing config decides the control image's pixel budget and grid, and a mismatched
    # one produces a plausible tensor of the wrong shape rather than an error.
    if not pipeline_root_id:
        raise RuntimeError(
            "[qwen_edit_preprocess] model.pipeline_root_id is unset. The Qwen2.5-VL PROCESSOR is a "
            "pipeline-root component — a Qwen-Image-Edit-2511 snapshot puts preprocessor_config.json "
            "in <root>/processor/, not in <root>/text_encoder/ — so it cannot be composed from "
            "model.text_encoder_id. Add the root to your config's `model:` block, e.g.\n"
            "    pipeline_root_id: qwen-image-edit-2511"
        )
    # ...and specifically the root's ``processor/`` SUBFOLDER, not the root itself. Measured, both
    # ways round, on live A100 runs:
    #   <root>/text_encoder  -> OSError: Can't load image processor ... no preprocessor_config.json
    #   <root>               -> ValueError: Unrecognized model ... (transformers then prints all
    #                           ~400 known model types). The root holds diffusers'
    #                           ``model_index.json``, which is a PIPELINE index and carries no
    #                           ``model_type``, so AutoProcessor has nothing to dispatch on.
    #   <root>/processor     -> correct: preprocessor_config.json + tokenizer.json + vocab.json +
    #                           tokenizer_config.json + special_tokens_map.json, i.e. a complete
    #                           processor directory.
    # Composed here rather than stored as its own config field: it is a fixed property of the
    # diffusers snapshot layout, not an operator choice, and a field would invite it to disagree
    # with the root it must live under.
    processor = _qwen_edit_load_processor(str(WEIGHTS_DIR / pipeline_root_id / "processor"))
    text_encoder = load_qwen_edit_text_encoder(str(WEIGHTS_DIR / text_encoder_id), device="cuda")
    print(assert_qwen_edit_text_encoder_vision(text_encoder, what="the Qwen2.5-VL text encoder"))
    quantize_qwen_edit(text_encoder, what="the Qwen2.5-VL text encoder")

    # Resolved ONCE per sample and reused in PHASE B: the slot plan is a pure function of the stem
    # and the configured directories, and resolving it twice invites the two phases to disagree about
    # which file filled slot i — the positional-identity failure this family's resolver exists to
    # refuse.
    # ⛔ control_dirs are MANIFEST-RELATIVE, exactly like ``row["media_path"]``, and are resolved
    # against ``data_root`` here — ``resolve_qwen_edit_control_sources`` takes directories as given
    # and does no joining of its own. Anchoring them to the manifest rather than to the process CWD
    # is what makes a config portable between the local dry-run and the container, where CWD is
    # ``/root`` and a bare ``refs_a`` resolves to ``/root/refs_a`` and is simply absent. An ABSOLUTE
    # entry is honoured unchanged, so a corpus whose controls live outside the manifest's tree stays
    # expressible.
    resolved_control_dirs = tuple(
        str(entry) if Path(str(entry)).is_absolute() else str(data_root / str(entry))
        for entry in control_dirs
    )
    print(
        f"[qwen_edit_preprocess] control dirs resolved against {data_root}: "
        f"{list(control_dirs)} -> {list(resolved_control_dirs)}"
    )

    plans: list[dict] = []
    for index, row in enumerate(rows):
        media_path = str(row["media_path"])
        stem = Path(media_path).stem
        slots = resolve_qwen_edit_control_sources(
            stem,
            resolved_control_dirs,
            control_slots=control_slots,
            blank_slot_fill=blank_slot_fill,
            blank_slots=blank_slots,
        )
        plans.append(
            {
                "row": row,
                "index": index,
                "media_path": media_path,
                "stem": stem,
                "rel": _qwen_edit_output_rel(media_path),
                "slots": slots,
                "caption": str(row.get("caption", "")),
            }
        )

    text_written = 0
    for plan in plans:
        # The VL channel: every slot's image at ``condition_area_px``, prepared FROM THE SOURCE.
        condition_images = [
            prepare_qwen_edit_image(
                _h3_open_reference_image(slot.path)
                if not slot.blank
                else _qwen_edit_blank_source(slot.fill or blank_slot_fill, condition_area_px),
                condition_area_px,
                blank=slot.blank,
                fill=slot.fill,
            ).image
            for slot in plan["slots"]
        ]
        cache_key = qwen_edit_text_cache_key(
            caption=plan["caption"],
            controls=[
                qwen_edit_control_identity(slot.path, mode=control_cache_key_mode)
                if not slot.blank
                else f"blank:{slot.fill or blank_slot_fill}"
                for slot in plan["slots"]
            ],
            condition_area_px=condition_area_px,
            text_encoder_id=text_encoder_id,
        )
        if cache_text_embeddings and _qwen_edit_text_cache_hit(output_root, plan["rel"], cache_key):
            continue
        text_payload = encode_qwen_edit_text_conditions(
            text_encoder,
            processor,
            plan["caption"],
            condition_images,
            cache_key=cache_key,
        )
        write_qwen_edit_precomputed(output_root, plan["rel"], text=text_payload)
        text_written += 1
    print(
        f"[qwen_edit_preprocess] PHASE A done — wrote {text_written} of {len(plans)} "
        f"{QWEN_EDIT_CONDITIONS_DIR}/ payload(s) "
        f"({len(plans) - text_written} already current at their cache key)."
    )

    # ── (3) two-phase VRAM discipline. The helper moves-to-CPU + collects + empties the cache, and
    # the CALLER must ALSO drop its own references: loader-owned CUDA storage frees only when the
    # LAST reference goes away.
    free_qwen_edit_text_encoder(text_encoder, processor)
    del text_encoder, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            "[qwen_edit_preprocess] freed the Qwen2.5-VL encoder — cuda allocated="
            f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB. The VAE is loaded only now."
        )

    # ── PHASE B — target + control latents. float32 for the VAE, matching the H3 stage's choice.
    vae = load_qwen_edit_vae(str(WEIGHTS_DIR / vae_id), device="cuda", dtype=torch.float32)
    latents_mean, latents_std = qwen_edit_vae_latent_stats(vae)

    for plan in plans:
        target_prepared = prepare_qwen_edit_image(
            _h3_open_reference_image(data_root / plan["media_path"]), target_area_px
        )
        target_payload = encode_qwen_edit_target_latents(
            vae, target_prepared, latents_mean, latents_std, stem=plan["stem"]
        )
        # The VAE channel: the SAME slot files, prepared again FROM THE SOURCE at the OTHER budget.
        control_prepared = [
            prepare_qwen_edit_image(
                _h3_open_reference_image(slot.path)
                if not slot.blank
                else _qwen_edit_blank_source(slot.fill or blank_slot_fill, control_area_px),
                control_area_px,
                blank=slot.blank,
                fill=slot.fill,
            )
            for slot in plan["slots"]
        ]
        control_payload = encode_qwen_edit_control_latents(
            vae,
            plan["slots"],
            control_prepared,
            latents_mean,
            latents_std,
            stem=plan["stem"],
            control_slots=control_slots,
        )
        write_qwen_edit_precomputed(
            output_root, plan["rel"], target=target_payload, controls=control_payload
        )
    print(f"[qwen_edit_preprocess] PHASE B done — {len(plans)} target + control payload(s).")

    # ── (5) the LOUD-FAILURE guard: a source that produced zero files, or a rel present under one
    # source and absent under another, RAISES. PrecomputedDataset would otherwise drop the sample.
    census = assert_qwen_edit_cache_complete(
        output_root, [str(plan["rel"]) for plan in plans]
    )

    # ── (6) commit-or-vanish (Pitfall 3) ─────────────────────────────────────────────────────────
    dataset_vol.commit()
    summary = (
        f"[qwen_edit_preprocess] done — {census} committed to the dataset Volume under "
        f"{output_dir} ('done' is the file on the Volume, not a log line)."
    )
    print(summary)
    return summary


def _qwen_edit_blank_source(fill: str, area_px: int) -> Any:
    """A synthetic blank slot image, sized to ``area_px``'s own square canvas.

    ``qwen_edit_blank_image(fill, width, height)`` renders it; the square canvas is derived from the
    area budget so ``prepare_qwen_edit_image`` snaps it to the same grid a real image of that budget
    lands on, and never triggers the non-square orientation warning for a picture that has no
    orientation to preserve.
    """
    import math  # noqa: PLC0415

    from signet_trainer.prep.qwen_edit_encode import qwen_edit_blank_image  # noqa: PLC0415

    edge = int(math.isqrt(int(area_px)))
    return qwen_edit_blank_image(fill, edge, edge)


def _qwen_edit_text_cache_hit(output_root: Any, rel: Any, cache_key: str) -> bool:
    """Is a CURRENT ``qwen_edit_conditions/`` payload already on disk for this rel + key?

    The re-encode skip, and the reason ``write_qwen_edit_precomputed`` tolerates a partial write. It
    is a HIT only when the file loads AND ``qwen_edit_text_cache_is_current`` agrees — a stale
    payload (different caption, different control identity, different budget, different encoder) is a
    MISS, which is exactly the chained-edit workflow's hazard: overwriting a control image in place
    is the normal operation on this family, and ``control_cache_key_mode: content`` is what makes the
    key notice.

    A load failure is a MISS, never an error: a truncated or half-written cache file must be
    re-encoded, not raised on, and PHASE A is the cheapest place to absorb that.
    """
    from pathlib import Path  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from signet_trainer.prep.qwen_edit_encode import (  # noqa: PLC0415
        QWEN_EDIT_CONDITIONS_DIR,
        qwen_edit_text_cache_is_current,
    )

    path = Path(output_root) / QWEN_EDIT_CONDITIONS_DIR / Path(rel)
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 — a corrupt cache file is a MISS, not a failed run
        return False
    return bool(qwen_edit_text_cache_is_current(payload, cache_key))


# --------------------------------------------------------------------------------------------------
# ``qwen_edit_train`` / ``qwen_edit_sample``. Every backend import stays FUNCTION-LOCAL.
# --------------------------------------------------------------------------------------------------


@app.function(
    # Same family image + memory request as ``qwen_edit_preprocess``.
    gpu="A100-80GB",
    image=qwen_gpu_image,
    volumes={**WEIGHTS_MOUNT, **DATASET_MOUNT, **CHECKPOINTS_MOUNT},
    secrets=[huggingface_secret, wandb_secret],
    memory=(80 * 1024, 200 * 1024),
    timeout=TWENTY_FOUR_HOURS,
    # THE PREEMPTION CONTRACT, inherited from ``h3_train`` and safe for the identical reason:
    # ``qwen_edit_train`` resumes in-dir from the latest COMMITTED checkpoint and ``train_loop``
    # commits per save, so a retry can never observe a half-written checkpoint. ``max_retries=10`` is
    # MODAL'S PLATFORM CEILING — the client validates only ``>= 0`` and the SERVER rejects anything
    # above 10 at app init ("Invalid function retries. Must specify number between 0 and 10",
    # measured 2026-08-06). Do not "fix" this by raising the number. The delay tail is bounded, not
    # exponential: ``modal/retries.py`` caps ``max_delay`` at 60.0 s and ``initial_delay`` is already
    # 60.0, so ``backoff_coefficient=2.0`` is clamped from the first retry — at most ~10 min of
    # cumulative queue delay. ``single_use_containers=True`` gives a FRESH container per retry, which
    # is Modal's canonical long-training shape. No warm-GPU tokens (D-10).
    retries=modal.Retries(max_retries=10, initial_delay=60.0, backoff_coefficient=2.0),
    single_use_containers=True,
)
def qwen_edit_train(config_yaml: str) -> None:
    """Stage 2 (qwen_edit) — the gated Qwen-Image-Edit chained-edit LoRA training run.

    The cadence is REUSED, not forked: ``train/loop.py``'s resume -> accumulate -> clip -> save ->
    commit -> callback -> commit is model-agnostic and is threaded here through its ``step_fn`` /
    ``collate_fn`` seam. Only the FORWARD is Qwen's, and this function does not even own that —
    ``train/family_hooks.build_loop_hooks("qwen_edit", ...)`` resolves both arguments from one table,
    so the family seam has ONE name rather than an inline closure per stage.

    Body order:

      0. ``_qwen_edit_require_backend`` — the signet-side modules, named before anything loads;
      1. the cold-path third-party probe (T-03-63);
      2. ``load_config_from_text(config_yaml)`` — the recipe crosses BY VALUE as YAML text, never a
         path (``configs/`` is not in the image). Re-validating in-container is the T-03-63
         convention and it re-fires the frame law, the rank/alpha lock, the LoRA-coverage check and
         the packed-row budget inside the paid container too;
      3. ``run_qwen_edit_arch_gate`` — the SHARED helper ``qwen_edit_preprocess`` calls, not a second
         copy. It aborts on any arch mismatch and on anything but 840 targets across all fourteen
         leaves, BEFORE any training spend. It is handed no ``model=``, so it LOADS one and RETURNS
         it — the run pays for exactly one load, never two;
      4. ``quantize_qwen_edit`` — qfloat8, on the un-wrapped transformer, BEFORE ``inject_lora``. The
         order is load-bearing in both directions: quantizing after the PEFT wrap would walk the
         adapter's own Linears (``quantize_qwen_edit`` refuses a wrapped module for exactly that
         reason), and injecting before quantizing would leave the adapter's ``lora_A``/``lora_B``
         quantized alongside the frozen base — they must stay bf16 and trainable;
      5. ``build_lora_config`` + ``inject_lora`` (GC before ``get_peft_model``, TRAIN-06);
      6. ``build_loop_hooks`` + ``build_optimizer`` + ``train_loop``;
      7. ``checkpoints_vol.commit()`` — commit-or-vanish.

    ⛔ NO autocast anywhere on this path. ``train/qwen_edit_step.QWEN_EDIT_AUTOCAST`` is ``False`` and
    the step calls the transformer BARE. The weights are qfloat8 with their own dequantization
    behaviour; an ``autocast("cuda", bf16)`` wrapper on top of that is a second precision policy
    fighting the first, silently and at the correct shape.

    ⛔ The offloader stays INERT (``offload.blocks_to_swap: 0`` in the shipped config, and
    ``BlockSwapOffloader`` early-returns to zero hooks at 0). Reaching for it here would be reaching
    past an UNMEASURED residency boundary: ``qwen_edit.max_packed_rows`` defaults to 0 —
    ceiling DISABLED — precisely because no OOM boundary has been measured for this model on any card
    in this program, and a swap policy tuned against a number nobody measured is not a safety
    feature.

    ``training.init_adapter_path`` is the CHAIN. When it is set the config layer additionally
    requires the resolved targets to be BYTE-EQUAL to the family default, because a warm start is the
    one operation where "covers all fourteen" is not enough — a superset warm-starts its extra
    modules from nothing. ``should_warm_start`` then applies it at COLD START only (no in-dir
    checkpoint), with a FRESH optimizer at step 0.
    """
    # ── (0) the signet-side modules, named before anything is loaded ──────────────────────────────
    # ``conditioning.qwen_edit_packing`` is named FIRST among the gaps this stage can hit: the
    # cached payload is a ``[C, F, H, W]`` latent, and ``QwenEditStrategy`` REFUSES a latent-form
    # payload when ``pack_fn`` is None rather than transcribing the 2x2 pack a second time. Naming
    # the module here turns that into a one-line abort at second zero instead of a strategy raise
    # after the transformer is resident.
    _qwen_edit_require_backend(
        "signet_trainer.models.qwen_edit_loader",
        "signet_trainer.train.qwen_edit_step",
        "signet_trainer.train.family_hooks",
        "signet_trainer.conditioning.qwen_edit_packing",
    )

    # ── (1) COLD-PATH IMPORT PROBE — before ANY model load (T-03-63) ──────────────────────────────
    # ⚠ ``wandb`` is deliberately NOT probed and NOT imported, the ``h3_train`` reasoning verbatim:
    # ``qwen_gpu_image`` does not declare it and ``train/loop.py`` never calls it, so probing it
    # would abort EVERY qwen run on a dependency nothing uses. The secret is still injected by name
    # so a future logging leg needs no decorator change.
    _qwen_edit_cold_path_probe("qwen_edit_train")

    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from signet_trainer.conditioning.qwen_edit import (  # noqa: PLC0415
        QWEN_EDIT_DATA_SOURCES,
        QwenEditStrategy,
    )
    from signet_trainer.conditioning.qwen_edit_packing import (  # noqa: PLC0415
        qwen_edit_image_rows,
    )
    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415
    from signet_trainer.data.precomputed import PrecomputedDataset  # noqa: PLC0415
    from signet_trainer.lora.peft import build_lora_config, inject_lora  # noqa: PLC0415
    from signet_trainer.models.qwen_edit_loader import quantize_qwen_edit  # noqa: PLC0415
    from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415
    from signet_trainer.train.family_hooks import build_loop_hooks  # noqa: PLC0415
    from signet_trainer.train.flow_match import FlowMatchingSchedule  # noqa: PLC0415
    from signet_trainer.train.loop import (  # noqa: PLC0415
        build_optimizer,
        build_scheduler,
        should_warm_start,
        train_loop,
    )

    # ── (2) load + REVALIDATE the config in-container (the recipe crossed by value) ───────────────
    config = load_config_from_text(config_yaml)
    if config.model.family != "qwen_edit":
        raise RuntimeError(
            f"[qwen_edit_train] model.family is {config.model.family!r}, not 'qwen_edit'. This "
            "stage drives the dual-stream MMDiT forward and injects the 14-leaf path-regex adapter; "
            "an LTX config here would inject a target set that matches ZERO modules (there is no "
            "attn1./attn2./ff. path anywhere in this checkpoint) and would only surface at "
            "build_optimizer's 'No trainable parameters found' — after the metered A100 is billing."
        )

    device = "cuda"
    torch_dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float32

    # The dataset reads its ROOT from config over exactly the sources the STRATEGY declares, so the
    # dir -> output-key map stays single-sourced with ``_PRECOMPUTED_SOURCE_OUTPUT_KEYS``. The names
    # come from the module-level tuple rather than from a constructed strategy: constructing one just
    # to read a constant would make the source list depend on the constructor's own defaults.
    data_sources = {name: _PRECOMPUTED_SOURCE_OUTPUT_KEYS[name] for name in QWEN_EDIT_DATA_SOURCES}
    dataset = PrecomputedDataset(config.data.preprocessed_data_root, data_sources=data_sources)
    print(
        f"[qwen_edit_train] dataset: {len(dataset)} sample(s) from "
        f"{config.data.preprocessed_data_root} over {sorted(data_sources)}."
    )

    # ── (3) the SHARED arch gate — abort BEFORE any training spend ────────────────────────────────
    # release defaults to False and no model= is passed, so the gate LOADS the transformer and hands
    # it back: one load for the whole run. ``text_embed_dim`` is threaded so the config's declared
    # 3584 is CHECKED against the live ``txt_in`` rather than believed.
    gate_line, transformer = run_qwen_edit_arch_gate(
        str(WEIGHTS_DIR / config.model.model_id),
        device=device,
        dtype=torch_dtype,
        config_source=(
            str(WEIGHTS_DIR / config.model.pipeline_root_id)
            if config.model.pipeline_root_id
            else None
        ),
        text_embed_dim=config.qwen_edit.text_embed_dim,
    )
    print(f"[qwen_edit_train] {gate_line}")
    if transformer is None:  # defensive: release=False must always return the proved model
        raise RuntimeError(
            "[qwen_edit_train] the arch gate returned no model. This stage must train the very "
            "transformer the gate just proved — re-loading would be an expensive mistake and would "
            "train a model the gate never inspected."
        )

    # ── (4) qfloat8 — on the UN-WRAPPED transformer, BEFORE inject_lora ───────────────────────────
    quantize_qwen_edit(transformer, what="the Qwen-Image-Edit transformer")

    # ── (5) inject the 14-leaf PATH-REGEX adapter (GC before get_peft_model — TRAIN-06) ───────────
    # ``resolved_lora_targets()`` rather than ``or <fallback>``: on this family the value is a bare
    # ``str`` regex, and the ``x or FALLBACK`` idiom keeps a regex intact but would silently
    # substitute the list-shaped LTX default for an empty one — matching zero modules here.
    gc.collect()
    lora_config = build_lora_config(
        rank=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        targets=config.resolved_lora_targets(),
    )
    model = inject_lora(transformer, lora_config)
    del transformer
    gc.collect()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"[qwen_edit_train] injected the qwen_edit adapter over "
        f"{'a path regex' if isinstance(config.resolved_lora_targets(), str) else 'a suffix list'} "
        f"— {len(trainable)} trainable tensor(s), {sum(p.numel() for p in trainable)} params, "
        f"rank={config.lora.rank} alpha={config.lora.alpha} (PEFT scale "
        f"{config.lora.alpha / config.lora.rank:.1f}; the chain's HARD LOCK is rank == alpha)."
    )

    # ── (6) the family seam: ONE table resolves both of train_loop's model-specific arguments ─────
    # ``pack_fn`` is supplied because the cache stores [C, F, H, W] latents; the strategy REFUSES a
    # latent-form payload with pack_fn=None rather than transcribing the 2x2 pack twice.
    # ``blank_latent_fn`` and ``sigma_fn`` are deliberately NOT supplied and must stay unset:
    #   * every slot — including a declared blank — was ENCODED at pre-encode time, so a training
    #     batch never has a gap to synthesize; a blank_latent_fn here could only mask a cache that
    #     is actually incomplete, which assert_qwen_edit_cache_complete already refuses;
    #   * the sigma is INJECTED into the batch by build_qwen_edit_step_fn from the same index its
    #     bell-curve weight came from. A sigma_fn would draw independently of that bookkeeping and
    #     silently decouple the loss weight from the timestep it weights.
    strategy = QwenEditStrategy(
        control_slots=config.qwen_edit.control_slots,
        blank_slot_fill=config.qwen_edit.blank_slot_fill,
        caption_dropout_rate=config.qwen_edit.caption_dropout_rate,
        pack_fn=qwen_edit_image_rows,
        max_packed_rows=config.qwen_edit.max_packed_rows,
        device=device,
        dtype=torch_dtype,
    )
    hooks = build_loop_hooks("qwen_edit", strategy=strategy, seed=config.training.seed)

    ckpt_manager = CheckpointManager(
        CHECKPOINTS_DIR / config.output_dir, keep_n=config.training.keep_checkpoints
    )
    if should_warm_start(ckpt_manager.find_latest() is not None, config.training.init_adapter_path):
        from signet_trainer.lora.peft import load_adapter_into  # noqa: PLC0415

        init_dir = CHECKPOINTS_DIR / config.training.init_adapter_path
        load_adapter_into(model, init_dir)
        print(
            f"[qwen_edit_train][chain] warm-started from {init_dir} (fresh optimizer, step 0) — "
            "the config layer already proved the target set is BYTE-EQUAL to the family default, "
            "which is what makes a rank-shaped warm start loadable."
        )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps=config.training.max_steps)
    final_step = train_loop(
        model,
        dataset,
        optimizer,
        scheduler,
        # ⚠ Handed to the loop for signature compatibility and DELIBERATELY UNUSED by the qwen step:
        # ``FlowMatchingSchedule`` draws a shifted logit-normal, while every proven qwen chain trained
        # under a discrete uniform draw over a linear 1000..1 grid, weighted by the bsmntw bell curve.
        # ``train/qwen_edit_step.py`` records the divergence; the loop's ``rng`` still drives the
        # draw, so the schedule stays reproducible from ``training.seed``.
        FlowMatchingSchedule(uniform_prob=config.training.uniform_prob),
        ckpt_manager,
        config,
        checkpoints_vol,  # commit-per-save: a preemption cannot vanish an uncommitted checkpoint.
        step_fn=hooks.step_fn,
        collate_fn=hooks.collate_fn,
    )
    print(f"[qwen_edit_train] loop done — reached step {final_step}/{config.training.max_steps}.")

    # ── (7) commit-or-vanish (Pitfall 3) ─────────────────────────────────────────────────────────
    checkpoints_vol.commit()
    print(
        f"[qwen_edit_train] done — checkpoints committed to signe-trainer-checkpoints under "
        f"{config.output_dir}/ (commit-or-vanish: 'done' is the file on the Volume, not a log line)."
    )


def _qwen_edit_adapter_is_live(model: Any) -> tuple[int, int]:
    """Count the adapter's ``lora_B`` tensors and how many are NON-ZERO. The cheap render floor.

    §8's convergence check is a base-vs-LoRA comparison — *"if the sample is essentially the base
    render, it isn't converging"* — and the degenerate case of that is an adapter which cannot move
    the model at all. PEFT initialises every ``lora_B`` to ZERO precisely so an injected-but-untrained
    adapter is the identity, so a band member whose weights never loaded (a wrong directory, a
    silently-empty ``set_peft_model_state_dict``) renders pixel-identical to the base column under a
    checkpoint's label: a grid that reports "not converging" for a checkpoint that was never
    consulted.

    ⚠ WHAT THIS DOES AND DOES NOT PROVE. It reads WEIGHTS, not outputs: a non-zero ``lora_B`` means
    the adapter is not the identity by construction, NOT that it changes this render's pixels
    perceptibly. H3's ``h3_adapter_delta`` measures the stronger property (max|delta velocity| across
    a real forward) and this is deliberately not that — the qwen forward needs a packed batch out of
    the training cache, which a render has no reason to mount. The strong floor stays H3's; this one
    is free, runs on the model already in memory, and catches the failure that actually happens.

    Returns:
        ``(lora_b_tensors, non_zero_tensors)`` — printed by the caller, so the number is in the
        container log whether or not anybody looks at the grid.
    """
    import torch  # noqa: PLC0415

    total = 0
    live = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" not in name:
                continue
            total += 1
            if bool(torch.any(param != 0)):
                live += 1
    return total, live


@app.function(
    # Same shape as ``qwen_edit_train`` MINUS the retries — the ``h3_sample`` precedent, and the same
    # judgement: re-enabling them is a COST decision (it multiplies the worst-case spend by
    # max_retries) and belongs to the operator, not to this file. The entrypoint applies the
    # config-derived timeout.
    gpu="A100-80GB",
    image=qwen_gpu_image,
    volumes={**WEIGHTS_MOUNT, **DATASET_MOUNT, **CHECKPOINTS_MOUNT},
    # ⛔ HF_HUB_OFFLINE — the STRUCTURAL half of the egress guard, and it has to be set HERE rather
    # than inside the function: ``huggingface_hub`` freezes the flag into a module constant at IMPORT
    # time, and this function imports diffusers (hence hub) on its first probe. As an env var on the
    # container it is in place before the interpreter starts. Every component is loaded from a
    # mounted local directory, so nothing legitimate here needs the Hub — and the failure this guards
    # is not an exception, it is tens of GiB of egress whose only symptom is the bill.
    #
    # ⚠ This is ALSO why ``load_qwen_edit_transformer``'s ``config_source`` must name a LOCAL
    # directory on the Volume for a render: ai-toolkit hard-codes the HUB id ``"Qwen/Qwen-Image"``
    # there, and with HF_HUB_OFFLINE set that resolves to a loud failure rather than a silent fetch —
    # which is the correct direction, but it means the config must supply ``pipeline_root_id``.
    secrets=[
        huggingface_secret,
        wandb_secret,
        modal.Secret.from_dict({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}),
    ],
    memory=(80 * 1024, 200 * 1024),
    timeout=TWENTY_FOUR_HOURS,
)
def qwen_edit_sample(config_yaml: str) -> None:
    """Stage 4 (qwen_edit) — the A/B-prompt x checkpoint-BAND render grid.

    ⛔ This does NOT extend ``inference/sampler.py``. That file is ltx-trainer's ``ValidationSampler``
    plus STG plus the two-stage upscaler — all LTX-only concepts with no Qwen meaning, and importing
    it here would quietly invite someone to pass this family an ``stg_scale``.

    **Checkpoint selection on this family is a BAND, not a winner.** The shipped deliverable of a
    chained-edit round is three checkpoints, and the render's unit of work is the whole band: the
    members share ``output_dir``, ``seed``, the held-out control set and their prompt pair, so they
    write identical FILENAMES and differ only in the render-dir identity. ``CheckpointBand`` +
    ``plan_qwen_edit_columns`` own that layout; ``expected_qwen_edit_band_keys`` is what a watcher
    asks "did the whole band land?" with.

    **ONE model renders every column.** The base row is the SAME PEFT-wrapped transformer under
    ``disable_adapter()`` (``qwen_edit_render_context``), and each band member is that same wrapper
    with the next checkpoint loaded into it — never a second transformer and never a second load.
    That is H3's reason (``fns.py:4356-4359``) applied to a family where it is if anything stronger:
    §8 reads convergence as base-vs-LoRA DIVERGENCE, so "identical seed, identical everything except
    the adapter" has to be a fact rather than a claim.

    Body order, and every step is load-bearing:

      0. ``_qwen_edit_require_backend`` — the signet-side modules, named before anything loads;
      1. the cold-path probe (diffusers / transformers / peft / PIL / optimum.quanto);
      2. config re-parse + family assert, in-container, before any spend;
      3. THE RENDER REQUEST — the band and the held-out inputs, both DECLARED. Nothing here is
         inferred: ``entrypoint._qwen_edit_config_gaps`` refuses the same two absences at $0 before
         dispatch, and this is the in-container mirror of that refusal (a passing local gate proves
         nothing about what actually reached the container);
      4. the band members are resolved against the CHECKPOINTS Volume and each one must be a
         COMPLETE checkpoint dir — the same completeness filter ``find_latest`` applies, so a
         half-written directory can never be pinned into a render either;
      5. the arch gate, UNCONDITIONAL and FIRST among the loads. No flag skips it and none stops
         after it; it hands back the very transformer this render uses, so the run pays for one load;
      6. ``quantize_qwen_edit`` on the UN-WRAPPED transformer (the recipe's order — ``assert_qwen_
         edit_not_peft_wrapped`` enforces it) then ``inject_lora`` over the 14-leaf path regex;
      7. the other four pipeline components from the mounted Volume, and ``build_qwen_edit_pipeline``,
         which pins the static scheduler AFTER construction and verifies the pin before returning;
      8. render — base columns first, then each band member in band order, resuming any cell already
         on the Volume and committing per image;
      9. the gallery + the §8 divergence read, then commit-or-vanish.

    ⚠ ``validation.prompts`` is NOT this grid's prompt source and is deliberately not read here. §8's
    A/B pair belongs to its held-out INPUT (same input, two prompts, side by side), so the prompts
    travel with the control images in ``qwen_edit.render_inputs``; a flat list cannot say which entry
    is A, which is B, or which input either belongs to. Refusing on ``validation.prompts`` would be
    refusing on a field the render never reads. ``validation.seed`` / ``width`` / ``height`` ARE read.

    ⚠ ``validation.guidance_scale`` and ``validation.num_inference_steps`` are also NOT read: the §8
    inference settings are locked in ``models/qwen_edit_pipeline.QWEN_EDIT_RENDER_RECIPE`` (30 steps,
    true_cfg 4.0, CFGNorm, the static scheduler reparameterisation, LoRA strength 1.0, a non-empty
    negative prompt) in the same sense that ``quantize_qwen_edit``'s qfloat8 is locked. The
    entrypoint refuses a config that CONTRADICTS the recipe, so nothing is silently overridden — and
    ``guidance_scale`` would in any case map to ``true_cfg_scale`` and never to the pipeline's
    ``guidance_scale=``, which this checkpoint ignores for want of a guidance embedder.
    """
    # ── (0) the signet-side modules, named before anything is loaded ──────────────────────────────
    _qwen_edit_require_backend(
        "signet_trainer.inference.qwen_edit_layout",
        "signet_trainer.models.qwen_edit_loader",
        "signet_trainer.models.qwen_edit_pipeline",
    )

    # ── (1) COLD-PATH IMPORT PROBE ────────────────────────────────────────────────────────────────
    _qwen_edit_cold_path_probe("qwen_edit_sample")

    import hashlib  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415
    from signet_trainer.inference.grid import write_qwen_edit_gallery  # noqa: PLC0415
    from signet_trainer.inference.qwen_edit_layout import (  # noqa: PLC0415
        CheckpointBand,
        QwenEditHeldOutInput,
        plan_qwen_edit_columns,
        plan_qwen_edit_rows,
        qwen_edit_cell_relpath,
        qwen_edit_control_ids,
        render_qwen_edit_sample,
    )
    from signet_trainer.inference.samples_layout import samples_root  # noqa: PLC0415
    from signet_trainer.lora.peft import (  # noqa: PLC0415
        build_lora_config,
        inject_lora,
        load_adapter_into,
    )
    from signet_trainer.models.qwen_edit_loader import (  # noqa: PLC0415
        assert_qwen_edit_text_encoder_vision,
        load_qwen_edit_text_encoder,
        load_qwen_edit_vae,
        quantize_qwen_edit,
    )
    from signet_trainer.models.qwen_edit_pipeline import (  # noqa: PLC0415
        QWEN_EDIT_RENDER_RECIPE,
        build_qwen_edit_pipeline,
    )
    from signet_trainer.train.checkpoint import CheckpointManager  # noqa: PLC0415

    # ── (2) load + revalidate the config in-container ─────────────────────────────────────────────
    config = load_config_from_text(config_yaml)
    if config.model.family != "qwen_edit":
        raise RuntimeError(
            f"[qwen_edit_sample] model.family is {config.model.family!r}, not 'qwen_edit' — this "
            "stage drives the diffusers Qwen-Image-Edit workflow, not the LTX pipelines."
        )
    if not config.model.pipeline_root_id:
        raise RuntimeError(
            "[qwen_edit_sample] config.model.pipeline_root_id is unset. The render needs the "
            "pipeline ROOT because the Qwen2.5-VL PROCESSOR lives at <root>/processor — NOT beside "
            "the text encoder, and NOT at the root itself; both were tried on live hardware and "
            "both failed. Add to the config's `model:` block:\n"
            "    pipeline_root_id: qwen-image-edit-2511"
        )
    seed = int(config.validation.seed)
    width, height = int(config.validation.width), int(config.validation.height)
    device, dtype = "cuda", torch.bfloat16
    recipe = QWEN_EDIT_RENDER_RECIPE

    # ── (3) THE RENDER REQUEST — declared, never inferred ─────────────────────────────────────────
    # The in-container mirror of ``entrypoint._qwen_edit_config_gaps``' sample arm. Both exist for
    # the reason ``modal/app.py``'s download_image banner states in one sentence: a passing local
    # gate proves nothing about what reached the container. ``getattr`` with a default rather than
    # attribute access, exactly as the entrypoint does, because these are DECLARED GAPS in the
    # schema today — a plain read would raise AttributeError instead of naming the missing field.
    declared_band = tuple(getattr(config.qwen_edit, "render_checkpoint_band", ()) or ())
    declared_inputs = tuple(getattr(config.qwen_edit, "render_inputs", ()) or ())
    if not declared_band:
        raise RuntimeError(
            "[qwen_edit_sample] qwen_edit.render_checkpoint_band is unset or empty — there is no "
            "adapter to render. §8 makes the BAND the deliverable unit ('checkpoint selection = a "
            "band, not a winner'), and find_latest() is deliberately NOT the fallback: it is a "
            "moving target while a training run commits, and since the render directory is keyed on "
            "the checkpoint NAME every re-dispatch would land in a fresh directory and resume "
            "nothing (H3's D-10-DEF-19). Declare the ordered band-member directory names."
        )
    if not declared_inputs:
        raise RuntimeError(
            "[qwen_edit_sample] qwen_edit.render_inputs is unset or empty — there is nothing "
            "held-out to render. §8 renders every held-out input under BOTH prompt modes (A "
            "style-only, B content-named) side by side; the pair IS the trace-vs-reinterpret "
            "measurement. Declare one entry per input: {id, images (one per control slot, in slot "
            "order), prompts (keyed by the QWEN_EDIT_PROMPT_MODES ids)}."
        )

    band = CheckpointBand.of(declared_band)
    columns = plan_qwen_edit_columns(band)
    # The manifest-free control resolution: each entry names its own images, in SLOT ORDER, so no
    # directory-scan convention is invented here (the pre-encode's positional rule, one layer up).
    # A path is Volume-relative by convention but an absolute one is accepted, the same way
    # ``h3_sample`` treats ``data.metadata_path`` — silently resolving an absolute path under the
    # mount would 404 for a reason nobody could read.
    control_slots = int(config.qwen_edit.control_slots)
    held_out: list[QwenEditHeldOutInput] = []
    control_images: dict[str, list[Any]] = {}
    staged_thumbs: dict[str, list[tuple[Path, str]]] = {}
    for index, entry in enumerate(declared_inputs):
        input_id = str(getattr(entry, "id", "") or "")
        images = tuple(getattr(entry, "images", ()) or ())
        prompts = dict(getattr(entry, "prompts", {}) or {})
        if not input_id or not images or not prompts:
            raise RuntimeError(
                f"[qwen_edit_sample] qwen_edit.render_inputs[{index}] is incomplete (id="
                f"{input_id!r}, {len(images)} image(s), {len(prompts)} prompt(s)). Every held-out "
                "input needs an id — it is the render key's conditioning slot AND the stem of every "
                "file it renders — its ordered per-slot images, and a prompt for both §8 modes."
            )
        if len(images) != control_slots:
            raise RuntimeError(
                f"[qwen_edit_sample] qwen_edit.render_inputs[{index}] ({input_id!r}) declares "
                f"{len(images)} control image(s) but qwen_edit.control_slots is {control_slots}. "
                "The mapping is POSITIONAL — image i fills slot i, which is what the prompt's "
                "ctrl_img_{i+1} addresses — so a short list does not render a smaller grid, it "
                "renders the WRONG request under the right label."
            )
        opened: list[Any] = []
        thumbs: list[tuple[Path, str]] = []
        for slot, raw in enumerate(images):
            declared_path = Path(str(raw))
            source = declared_path if declared_path.is_absolute() else DATASET_DIR / declared_path
            if not source.exists():
                raise RuntimeError(
                    f"[qwen_edit_sample] control image for {input_id!r} slot {slot} does not exist: "
                    f"{source}. Declared as {raw!r}; a relative path resolves under the dataset "
                    "Volume mount. Refusing before the arch gate rather than after ~40 GiB of "
                    "loads — the missing file is the cheapest thing in this stage to discover."
                )
            # RGB, explicitly: the VAE encodes three channels, and an RGBA or paletted control would
            # either raise deep inside the pipeline's own preprocessing or silently drop its alpha.
            # This is the one image transform performed here — the SIZE stays the source's, because
            # the pipeline resizes it itself at both budgets and
            # ``assert_qwen_edit_control_geometry`` proves those two resizes equal signet's geometry.
            # Pre-resizing here would resample every image twice.
            opened.append(Image.open(source).convert("RGB"))
            thumbs.append((source, f"controls/{input_id}_slot{slot}{source.suffix or '.png'}"))
        held_out.append(
            QwenEditHeldOutInput(
                input_id=input_id,
                prompts=prompts,  # refuses a missing/blank/unknown mode id in __post_init__
                control_imgs=tuple(rel for _src, rel in thumbs),
                label=str(getattr(entry, "label", "") or "") or None,
            )
        )
        control_images[input_id] = opened
        staged_thumbs[input_id] = thumbs

    control_ids = qwen_edit_control_ids(held_out)  # refuses a duplicate id; order PRESERVED
    render_root = CHECKPOINTS_DIR / samples_root(config.output_dir, config.model.family)
    print(
        f"[qwen_edit_sample] {len(held_out)} held-out input(s) x {len(columns)} column(s) "
        f"= {len(held_out) * len(columns)} image(s) at seed {seed}, {width}x{height}; "
        f"{band.describe()}; recipe {recipe.describe()}. Renders land under {render_root} "
        "(A/B prompt modes are SUBDIRS of one render dir, not two dirs — the mode is not an "
        "identity axis, so a row whose A half landed can never read as a complete render)."
    )

    # ── (4) the band, resolved against the checkpoints Volume ─────────────────────────────────────
    checkpoints_vol.reload()
    ckpt_root = CHECKPOINTS_DIR / config.output_dir
    member_dirs: dict[str, Any] = {}
    for member in band.members:
        candidate = ckpt_root / member
        if not CheckpointManager.is_complete(candidate):
            available = sorted(p.name for p in ckpt_root.glob("checkpoint-step-*") if p.is_dir())
            raise RuntimeError(
                f"[qwen_edit_sample] band member {member!r} is not a COMPLETE checkpoint dir at "
                f"{candidate} (it needs both the adapter and training_state.pt — the same "
                f"completeness filter find_latest applies, so a half-written dir can never be "
                f"rendered either). Available: {available}. The whole band is checked BEFORE the "
                "arch gate: discovering member 3 is missing after two members have rendered wastes "
                "the expensive half of this stage."
            )
        member_dirs[member] = candidate
    print(
        f"[qwen_edit_sample] band verified on the Volume: "
        f"{', '.join(str(member_dirs[m].name) for m in band.members)}."
    )

    # ── (5) THE ARCH GATE — unconditional, first, and it hands back the model this render uses ────
    gate_line, transformer = run_qwen_edit_arch_gate(
        str(WEIGHTS_DIR / config.model.model_id),
        device=device,
        dtype=dtype,
        config_source=str(WEIGHTS_DIR / config.model.pipeline_root_id),
        text_embed_dim=config.qwen_edit.text_embed_dim,
    )
    print(f"[qwen_edit_sample] {gate_line}")
    if transformer is None:  # defensive: release=False must always return the proved model
        raise RuntimeError(
            "[qwen_edit_sample] the arch gate returned no model. This stage must render with the "
            "very transformer the gate just proved — re-loading would pay for 40.9 GiB twice and "
            "would render with a model the gate never inspected."
        )

    # ── (6) qfloat8 on the UN-WRAPPED transformer, then the 14-leaf adapter ───────────────────────
    # Order is the recipe's and it is enforced rather than remembered: ``quantize_qwen_edit`` calls
    # ``assert_qwen_edit_not_peft_wrapped``. ``resolved_lora_targets()`` (never ``x or FALLBACK``):
    # on this family the value is a bare regex ``str``, and the idiom would substitute the
    # list-shaped LTX default for an empty one — matching zero modules here.
    quantize_qwen_edit(transformer, what="the Qwen-Image-Edit transformer")
    adapted = inject_lora(
        transformer,
        build_lora_config(
            rank=config.lora.rank,
            alpha=config.lora.alpha,
            dropout=0.0,  # a render is not a training step; dropout would randomise the comparison
            targets=config.resolved_lora_targets(),
        ),
    )
    del transformer
    adapted.eval()  # inject_lora leaves the model in train mode + grad checkpointing (TRAIN-06)
    print(
        f"[qwen_edit_sample] adapter injected over "
        f"{'a path regex' if isinstance(config.resolved_lora_targets(), str) else 'a suffix list'} "
        f"— rank={config.lora.rank} alpha={config.lora.alpha} (PEFT scale "
        f"{config.lora.alpha / config.lora.rank:.1f}, which IS §8's LoRA strength "
        f"{recipe.lora_scale}). ONE wrapper renders every column: the base row is this model under "
        "disable_adapter(), each band member is this model with the next checkpoint loaded in."
    )

    # ── (7) the other four components, and the pipeline ───────────────────────────────────────────
    text_encoder = load_qwen_edit_text_encoder(
        str(WEIGHTS_DIR / config.model.text_encoder_id), device=device, dtype=dtype
    )
    # The VISION half is not optional: the pipeline passes pixel_values + image_grid_thw, and a
    # text-only LLM in this slot fails with "mat1 and mat2 shapes cannot be multiplied
    # (5376x1280 and 3840x1280)" — a real failure this house already hit and fixed.
    print(assert_qwen_edit_text_encoder_vision(text_encoder)["summary"])
    quantize_qwen_edit(text_encoder, what="the Qwen2.5-VL text encoder")
    vae = load_qwen_edit_vae(str(WEIGHTS_DIR / config.model.vae_id), device=device, dtype=dtype)
    processor = _qwen_edit_load_processor(
        str(WEIGHTS_DIR / config.model.pipeline_root_id / "processor")
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "[qwen_edit_sample] the loaded processor exposes no .tokenizer, and "
            "QwenImageEditPlusPipeline takes the tokenizer and the processor as SEPARATE "
            f"components ({type(processor).__name__!r} was loaded from "
            f"{WEIGHTS_DIR / config.model.pipeline_root_id / 'processor'}). AutoProcessor resolves "
            "the concrete class from the checkpoint's own processor_class, so an unexpected class "
            "here means the mounted snapshot is not a Qwen2.5-VL processor."
        )
    pipeline = build_qwen_edit_pipeline(
        transformer=adapted,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
    )
    print(
        f"[qwen_edit_sample] pipeline assembled ({type(pipeline).__name__}); scheduler pinned to "
        "the §8 STATIC reparameterisation AFTER construction and verified — the trap on this "
        "family is that a pipeline factory rebuilds its own default-shift scheduler, and the "
        "symptom is a muddy render that reads as a bad adapter."
    )

    # ── (8) render — base first, then each band member, resuming and committing per image ─────────
    # Stage the control thumbnails INTO the render root so the gallery is self-contained: the
    # sources live on the dataset Volume, and a page committed to the checkpoints Volume that linked
    # back to them would show broken control tiles wherever it is actually read.
    for input_id, thumbs in staged_thumbs.items():
        for source, rel in thumbs:
            destination = render_root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size == 0:
                shutil.copyfile(source, destination)

    rendered, resumed = 0, 0

    def _cell(column: Any, item: QwenEditHeldOutInput, *, adapter: bool) -> Path:
        """Render ONE cell, or resume it. The path comes from the SINGLE column->file join.

        ``qwen_edit_cell_relpath`` is the one transcription of ``<render dir>/<render_subdir>/
        <input>_s<seed>.png``, and the gallery reads the same function's answer back through
        ``row[column.row_key]``. Composing a second spelling here is how a grid comes to reference
        files that exist under different names — every tile falls back to 'generation failed' on a
        render that succeeded, or one column's file is found under another column's key.
        """
        nonlocal rendered, resumed
        out_path = render_root / qwen_edit_cell_relpath(
            column, input_id=item.input_id, seed=seed, control_ids=control_ids
        )
        if out_path.exists() and out_path.stat().st_size > 0:
            # Non-empty, not merely present: a container killed mid-save leaves a 0-byte file, and
            # skipping THAT would put a corrupt cell in the grid rather than re-render it.
            resumed += 1
            print(f"[qwen_edit_sample] resume — {column.row_key} / {item.input_id} already rendered.")
            return out_path
        render_qwen_edit_sample(
            pipeline=pipeline,
            control_images=control_images[item.input_id],
            prompt=item.prompts[column.mode.id],
            out_path=out_path,
            seed=seed,
            width=width,
            height=height,
            adapter=adapter,
        )
        # COMMIT-OR-VANISH, per image (h3_sample's per-clip commit): a render that is not on the
        # Volume did not happen, and a preemption at cell 11 of 12 must not lose the other eleven.
        checkpoints_vol.commit()
        rendered += 1
        return out_path

    # The BASE columns, once for the whole band — ``qwen_edit_render_dir`` keys them on the reserved
    # ``base`` token precisely so they are not re-rendered per member into byte-identical files.
    base_paths: dict[tuple[str, str], Path] = {}
    for column in (col for col in columns if col.is_base):
        for item in held_out:
            base_paths[(item.input_id, column.mode.id)] = _cell(column, item, adapter=False)
    print(f"[qwen_edit_sample] BASE columns done — the §8 divergence reference, at seed {seed}.")

    member_paths: dict[tuple[str, str, str], Path] = {}
    for member in band.members:
        load_adapter_into(adapted, member_dirs[member])
        lora_b, live = _qwen_edit_adapter_is_live(adapted)
        if live == 0:
            raise RuntimeError(
                f"[qwen_edit_sample] band member {member!r} loaded {lora_b} lora_B tensor(s) and "
                "EVERY ONE is zero, so this adapter is the identity by construction: its columns "
                "would be the base column's pixels under a checkpoint's label, and §8's divergence "
                "read would report 'not converging' for a checkpoint that never participated. PEFT "
                "initialises lora_B to zero, so this is what an adapter that failed to load looks "
                f"like — check {member_dirs[member]}. Refusing before the member's renders."
            )
        print(
            f"[qwen_edit_sample] loaded {member} — {live}/{lora_b} lora_B tensor(s) non-zero. This "
            "reads WEIGHTS, not outputs: it proves the adapter is not the identity, not that it "
            "moves this render perceptibly (that is the grid's job, and §8 says read samples)."
        )
        for column in (col for col in columns if col.checkpoint == member):
            for item in held_out:
                member_paths[(member, item.input_id, column.mode.id)] = _cell(
                    column, item, adapter=True
                )

    # ── (9) the gallery, the §8 divergence read, and commit-or-vanish ─────────────────────────────
    # The rows come from the planner, so the render loop assembles no dict of its own and the page
    # can only reference paths the join above produced.
    index_path = write_qwen_edit_gallery(
        plan_qwen_edit_rows(held_out, columns, seed=seed),
        render_root / "index.html",
        {
            # EXACTLY the banner's allowlist, every value from the LOCKED recipe rather than from a
            # config field the render does not read — a banner that describes a render nobody
            # performed is the failure h3_sample records for its own width/height (fns.py:4808).
            "steps": recipe.steps,
            "true_cfg": recipe.true_cfg,
            "cfg_norm": recipe.cfg_norm,
            "width": width,
            "height": height,
            "lora_scale": recipe.lora_scale,
            "checkpoint_band": band.describe(),
        },
        columns=columns,
    )

    # §8: *"convergence is read as base-vs-LoRA divergence … if the sample is essentially the base
    # render, it isn't converging."* Byte-identity is the strongest possible form of "essentially
    # the base render" and the only one measurable without a perceptual metric, so it is REPORTED
    # rather than judged: a non-zero count is a finding for the operator reading the grid, and the
    # hard refusal for the degenerate cause (an adapter that never loaded) already fired above.
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""

    identical = [
        f"{member}/{input_id}/{mode_id}"
        for (member, input_id, mode_id), path in member_paths.items()
        if _digest(path) and _digest(path) == _digest(base_paths.get((input_id, mode_id), path))
    ]
    checkpoints_vol.commit()
    print(
        f"[qwen_edit_sample] done — {rendered} image(s) rendered, {resumed} resumed from the "
        f"Volume, gallery at {index_path} (committed to signe-trainer-checkpoints). §8 divergence "
        f"read: {len(identical)} of {len(member_paths)} adapter cell(s) are BYTE-IDENTICAL to their "
        f"base cell{': ' + ', '.join(identical) if identical else ''}. Zero is the expected result; "
        "anything else means those cells are the base render under a checkpoint's label."
    )
