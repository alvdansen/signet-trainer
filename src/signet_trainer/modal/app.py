"""The single Modal App + Image + three Volumes + two secrets (MODL-01, D-07/D-08/D-09).

This is the thin transport boundary (Anti-Pattern 6): the ONLY package allowed to
``import modal``. Everything load-bearing (config, dry-run, strategies) stays Modal-agnostic.

What this module declares (verified Modal 1.5.0 scaffold, RESEARCH.md Q5):
  * one ``modal.App(APP_NAME)`` — default ``"signe-trainer"``              (D-09 naming)
  * three named Volumes, default ``signe-trainer-{weights,dataset,checkpoints}``  (D-07)
        via ``modal.Volume.from_name(name, create_if_missing=True)``
    Those four names are CONFIG-DRIVEN with PRE-RENAME defaults on purpose — they point at
    existing Modal resources holding live data. See the ⛔ block above their definitions.
  * mount-point path constants (``WEIGHTS_DIR`` / ``DATASET_DIR`` / ``CHECKPOINTS_DIR``)
  * a ``modal.Image`` built from the ``modal-runtime`` extras — CODE ONLY. Base weights
    + Gemma checkpoints are NEVER baked into the image; they load from the weights Volume
    mount at run time (MODL-01 / M-4 / T-01-MD4).
  * two ``modal.Secret.from_name(...)`` references whose NAMES are config-driven via env
    overrides (``SIGNET_HUGGINGFACE_SECRET_NAME`` / ``SIGNET_WANDB_SECRET_NAME``), defaulting to
    the account's ``my-huggingface-secret`` / ``my-wandb-secret`` — declared here, consumed
    Phase 2+ (D-09). Modal eagerly resolves every ``Secret.from_name`` in the app graph, so the
    names MUST match the running account's secrets.

CRITICAL (D-10 / Pitfall 5 / T-01-MD3): no construct in this file sets ``keep_warm`` or
``min_containers`` — warm GPUs are opt-in, and we never opt in. A structural test
(``tests/test_no_warm_gpu.py``) fails CI if either token appears.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# --------------------------------------------------------------------------------------------------
# App + Volume NAMES (D-07 / D-09) — CONFIG-DRIVEN via env seam (D-NOHARDCODE / AUDIT #1).
#
# ⛔ THE FOUR DEFAULTS BELOW DELIBERATELY KEEP THE PRE-RENAME "signe-*" STRINGS. DO NOT "FIX" THEM.
# They are not branding — they NAME PRE-EXISTING MODAL RESOURCES that already hold this project's
# live data (~134 GiB of base weights; the encoded corpus; every checkpoint of the completed r1
# round). ``Volume.from_name(..., create_if_missing=True)`` does NOT fail on a name it has never
# seen — it silently provisions a NEW, EMPTY Volume. So renaming a default here does not break
# loudly: the trainer would quietly read empty storage while the real data sits untouched under the
# old name. The App name is likewise what in-flight runs are dispatched under and what
# ``modal app list`` / ``modal app logs`` attribute them to. The signe -> Signet rename is a
# PUBLIC-IDENTITY change (package / CLI / docs / license) and stops at this boundary.
#
# Same env seam as the secrets below, and for the same reason: this module builds the Modal app
# graph at MODULE-IMPORT time, so it cannot read a runtime SignetConfig. The entrypoint exports
# these from the loaded config BEFORE the graph is built and fail-fasts pre-approval on a mismatch.
# Keep the defaults byte-identical to
# SignetConfig.modal.{app_name,weights_volume_name,dataset_volume_name,checkpoints_volume_name}.
# A beta user points these at THEIR OWN App/Volumes with no code edit — see README
# "Pointing the trainer at your own Modal account".
# --------------------------------------------------------------------------------------------------

APP_NAME = os.environ.get("SIGNET_APP_NAME", "signe-trainer")

# Three SEPARATE named Volumes; weights/dataset/checkpoints never share a Volume.
# ``create_if_missing=True`` so a fresh account provisions them on first use (RESEARCH.md Q5).
WEIGHTS_VOLUME_NAME = os.environ.get("SIGNET_WEIGHTS_VOLUME_NAME", "signe-trainer-weights")
DATASET_VOLUME_NAME = os.environ.get("SIGNET_DATASET_VOLUME_NAME", "signe-trainer-dataset")
CHECKPOINTS_VOLUME_NAME = os.environ.get(
    "SIGNET_CHECKPOINTS_VOLUME_NAME", "signe-trainer-checkpoints"
)

weights_vol = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
dataset_vol = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
checkpoints_vol = modal.Volume.from_name(CHECKPOINTS_VOLUME_NAME, create_if_missing=True)

# Mount-point path constants. Base weights live UNDER ``WEIGHTS_DIR`` on the mounted Volume
# (loaded Modal-side in Phase 2 as ``WEIGHTS_DIR / model_id``), NEVER baked into the Image.
WEIGHTS_DIR = Path("/weights")
DATASET_DIR = Path("/dataset")
CHECKPOINTS_DIR = Path("/checkpoints")

# Convenience mappings used by @app.function decorators in fns.py.
WEIGHTS_MOUNT = {str(WEIGHTS_DIR): weights_vol}
DATASET_MOUNT = {str(DATASET_DIR): dataset_vol}
CHECKPOINTS_MOUNT = {str(CHECKPOINTS_DIR): checkpoints_vol}

# --------------------------------------------------------------------------------------------------
# Secrets (D-09) — declared here, consumed Phase 2+ (gated HF download + wandb logging).
# Referenced by name only; values are injected into the container at run time and MUST NEVER be
# logged or baked into the Image (T-01-MD1).
#
# CONFIG-DRIVEN secret names: Modal eagerly resolves EVERY Secret.from_name(...) in the app graph
# when running ANY function, so the names declared here must match the running account's secrets.
# Because app.py builds the app graph at MODULE-IMPORT time it cannot read a runtime SignetConfig
# object — so the names come from env-var overrides resolved here, defaulting to the same values as
# SignetConfig.modal.{huggingface_secret_name,wandb_secret_name}. Phase-2 entrypoint wiring exports
# these env vars from the loaded config BEFORE importing/invoking remote functions, making the
# scaffold portable across accounts (each account keeps its own names) without code edits. This
# is the env-var seam — keep the defaults aligned with the config defaults.
# --------------------------------------------------------------------------------------------------

HUGGINGFACE_SECRET_NAME = os.environ.get("SIGNET_HUGGINGFACE_SECRET_NAME", "my-huggingface-secret")
WANDB_SECRET_NAME = os.environ.get("SIGNET_WANDB_SECRET_NAME", "my-wandb-secret")

huggingface_secret = modal.Secret.from_name(HUGGINGFACE_SECRET_NAME)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)

# --------------------------------------------------------------------------------------------------
# Image (MODL-01 / T-01-SC) — CODE ONLY. Adds the local ``signet_trainer`` package source into the
# container so the function-per-stage boundary (which imports ``signet_trainer.modal.app``) resolves
# container-side. It does NOT add any ``*.safetensors`` / weight file: base weights + Gemma load
# from the weights Volume mount (T-01-MD4 / "Do NOT bake base weights into the image").
#
# Phase-1 scope: signet-trainer is a clean-room, UN-PUBLISHED package, so it cannot be installed by
# name from an index. ``add_local_python_source("signet_trainer")`` copies the local src-layout
# package into the image (code only) — enough for the CPU ``volume_roundtrip_probe``, whose import
# chain (modal/app -> modal/fns) needs only stdlib + ``modal`` (already present), NOT the heavy
# ``[modal-runtime]`` extras. Phase 2 attaches those extras to the GPU stage functions, e.g.
# ``.uv_pip_install("ltx-core", "ltx-trainer", "peft>=0.14", "accelerate>=1.2", "diffusers>=0.32")``
# (or ``pip install -e .`` once the package is buildable in-image) — see the Phase-2 note in fns.py.
# --------------------------------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    # CODE ONLY: copy the local signet_trainer package into the image (clean-room, not on any index).
    .add_local_python_source("signet_trainer")
    # NOTE (MODL-01/M-4): intentionally NO ``.add_local_file(*.safetensors)`` / no baked weights.
    # Weights mount from ``signe-trainer-weights`` at ``/weights`` and are read as
    # ``WEIGHTS_DIR / model_id`` Modal-side in Phase 2.
)

# --------------------------------------------------------------------------------------------------
# GPU-stage image (Phase 2 — T-02-MD2 / T-02-SC supply-chain) — layers the heavy ltx-core/ltx-trainer
# install onto the GPU stage functions ONLY (preprocess + load_ltxv_smoke). The default code-only
# ``image`` above stays attached to the App for the CPU ``volume_roundtrip_probe`` + ``download_weights``
# (pure HF download — no ltx deps). GPU stage fns set ``image=gpu_image`` per-function in fns.py.
#
# Supply-chain discipline (RESEARCH.md A5 / Package Legitimacy Audit / T-02-MD2):
#   * ltx-core + ltx-trainer are NOT on any PyPI index — they are git-installed from the OFFICIAL
#     ``github.com/Lightricks/LTX-2`` monorepo, pinned to a LITERAL commit SHA (NEVER ``main``) so
#     the image build is reproducible and tamper-evident. Bump ``LTX2_COMMIT_SHA`` deliberately.
#   * Base aligns to torch~=2.7 / CUDA 12.9 (cu129 index, RESEARCH.md line 71): LTX-2 wants recent
#     torch+CUDA; ``uv pip install`` resolves the transitive PyPI deps from ltx-trainer's own pins
#     (no registry slop introduced here — slopcheck N/A for the git path).
#   * NO ``*.safetensors`` baked into the image (MODL-01) — weights mount from the Volume at runtime.
# --------------------------------------------------------------------------------------------------

# Pinned via ``gh api repos/Lightricks/LTX-2/commits/main --jq .sha`` on 2026-06-14 (RESEARCH.md A5).
# This is a LITERAL SHA, not ``main`` — bump deliberately to take upstream changes (T-02-MD2).
LTX2_COMMIT_SHA = "d6053703e00195bc668cbd1d5eda9dc0b2e7b74a"

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    # ffmpeg: torchaudio's load() needs the FFmpeg SHARED LIBS to demux audio out of video
    # containers (mp4/AAC). Without them the pin's ``_extract_audio`` throws, the exception is
    # swallowed at DEBUG level, and an a2v encode silently reports "0 videos with audio" —
    # the burned-gate class (a2v leg, 2026-07-15). Benign for all other stages.
    .apt_install("git", "ffmpeg")
    .pip_install("uv")
    # torch~=2.7 on the CUDA 12.9 wheel index (cu129) — match LTX-2's recent-torch/CUDA requirement
    # before the editable installs so ltx-core/ltx-trainer resolve against the right torch.
    # torch + torchvision MUST be the matched cu129 pair: gemma-3-12b-it is multimodal, so
    # transformers imports AutoImageProcessor (needs torchvision ops); a mismatched/CPU torchvision
    # gives "operator torchvision::nms does not exist". Install both here, before the editable install.
    .run_commands(
        "uv pip install --system --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cu129 'torch~=2.7' 'torchvision~=0.22'"
    )
    # Clone the OFFICIAL monorepo and CHECKOUT THE PINNED SHA (never main), then editable-install
    # ltx-core + ltx-trainer. The pinned SHA makes this a reproducible, tamper-evident build.
    # Carry the cu129 index here too so any torch/torchvision the resolver touches stays on cu129
    # (never falls back to the default-PyPI CPU build that breaks torchvision::nms).
    .run_commands(
        "git clone https://github.com/Lightricks/LTX-2 /opt/LTX-2",
        f"cd /opt/LTX-2 && git checkout {LTX2_COMMIT_SHA}",
        "cd /opt/LTX-2 && uv pip install --system --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cu129 -e packages/ltx-core -e packages/ltx-trainer",
        # D-DEPTH-1 (04-05): the two-stage spatial-upscaler toggle (default OFF) needs
        # ltx-pipelines' ``TI2VidTwoStagesPipeline``. Editable-install it from the SAME
        # /opt/LTX-2 checkout at the SAME literal LTX2_COMMIT_SHA — no ``main``, no new SHA,
        # no PyPI resolution (tamper-evident supply chain, Phase-2 discipline; RESEARCH
        # Package Legitimacy Audit marks ltx-pipelines Approved at this SHA). This only adds
        # the import surface; the toggle default stays OFF so single-stage never depends on it.
        "cd /opt/LTX-2 && uv pip install --system --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cu129 -e packages/ltx-pipelines",
    )
    # VRAM fragmentation mitigation (07-12 / D-7-FREEZE proof-b): the doubled ref+target sequence
    # plus a SECOND stacked (frozen) adapter's forward tips the 22B step just over 80 GiB — the
    # frozen-stack run OOM'd by only ~196 MiB with ~1.01 GiB reserved-but-unallocated (fragmentation).
    # ``expandable_segments:True`` lets the CUDA caching allocator grow/reclaim segments instead of
    # fragmenting fixed blocks, recovering that headroom. Must be a process-env var set BEFORE torch's
    # CUDA allocator initializes, so it lives on the image (not set in-code after CUDA is already up).
    # Benign + generally beneficial for the other GPU stages (train/sample/preprocess) too.
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # CODE ONLY: copy the local signet_trainer package in (same as the code-only image). No weights.
    .add_local_python_source("signet_trainer")
)

# --------------------------------------------------------------------------------------------------
# H3 GPU-stage image (Phase 10 — H3-07) — the MiniMax-H3 Ref2VA image. SEPARATE from ``gpu_image``:
# the H3 path goes through ``diffusers``' ``MiniMaxH3Transformer3DModel``, NOT ltx-core/ltx-trainer, so
# it does not carry (and must not pay for) the LTX-2 monorepo clone + three editable installs.
#
# This recipe is not proposed — it is TRANSCRIBED from ``scripts/_h3_probe_modal.py:77-94``, which RAN
# CLEAN on a real A100-80GB (P10-1 / P10-1b: loaded the 22B ref transformer, injected rank-64 LoRA over
# 300 main-stack targets, and took real forward+backward+AdamW8bit steps across a sequence-length
# ladder). Changing any line here invalidates that measurement.
#
# Supply-chain discipline (mirrors ``LTX2_COMMIT_SHA`` above):
#   * ``diffusers`` is git-installed from the OFFICIAL ``github.com/huggingface/diffusers`` repo pinned
#     to a LITERAL commit SHA, NEVER ``main``. MiniMax-H3 landed in diffusers only days before this was
#     written and the surface is still moving, so an unpinned ``main`` would make the image build
#     silently non-reproducible. Bump ``DIFFUSERS_SHA`` DELIBERATELY.
#   * This is a GIT PIN OF AN EXISTING DEPENDENCY, not a new package: ``diffusers>=0.32`` is already
#     declared in ``[project.optional-dependencies].modal-runtime`` in ``pyproject.toml``, and
#     ``tests/test_pyproject_supply_chain.py`` sanctions exactly this ``git+https`` + 40-hex-SHA route.
#     ``peft``/``accelerate`` are likewise already-declared ``[modal-runtime]`` deps, and
#     ``transformers``/``safetensors``/``bitsandbytes``/``huggingface_hub`` are already present in
#     ``gpu_image`` and imported by shipped code (``fns.py`` imports ``bitsandbytes``; ``lora/peft.py``
#     imports ``safetensors``; ``fns.py`` imports ``huggingface_hub``). NO package new to the project
#     is introduced here.
#   * ``av`` (PyAV) is likewise NOT new to the project — it is imported at seven sites already
#     (``data/mask_encode.py``, ``inference/reference_video.py``, five in ``modal/fns.py``), where it
#     arrives on ``gpu_image`` as an ``ltx-trainer`` TRANSITIVE. ``h3_gpu_image`` deliberately does
#     not install ltx-trainer, so ``av`` was simply absent here while the H3 stages imported it
#     directly. Adding it is DECLARATION PARITY on an already-trusted, already-resident distribution,
#     not a new trust decision — so no package-legitimacy checkpoint applies. Left unpinned to match
#     ``safetensors`` / ``bitsandbytes`` above; inventing a floor this project has never verified
#     would be a number nobody could defend.
#   * NO ``*.safetensors`` baked into the image (MODL-01) — see the weights note below.
# --------------------------------------------------------------------------------------------------

# diffusers ``main`` at the SHA the H3 arch was read + probed at. A LITERAL SHA, never ``main``
# (D-10-PIN, the LTX2_COMMIT_SHA discipline applied to diffusers).
DIFFUSERS_SHA = "9f169d98d0bce392a889c3b6524d0d97734dfc0e"

# ⛔ ``transformers`` is a CONDITIONING-CRITICAL dependency on the H3 leg, so it gets the same
# literal-pin discipline as ``DIFFUSERS_SHA`` (D-10-DEF-4). It owns ``Qwen3VLProcessor`` — the object
# that decides how many vision tokens each reference expands to and therefore what the model is
# conditioned on. The old ``'transformers>=4.51'`` range resolved to 5.14.1 at build time, and the
# range is precisely what made the first metered dispatch's failure a COINCIDENCE: 5.14.1's
# replacement path raises on a pre-expanded presentation, while a 4.x resolution of the SAME range
# would have left the surplus pads un-expanded and conditioned the model at a plausible-but-wrong
# shape with nothing to catch it. The pre-encode no longer depends on that exception (it checks the
# realized expansion arithmetically), but a floating range on a conditioning-critical package is a
# reproducibility hole regardless: the image that trains an adapter must be the image that renders
# it. 5.14.1 is the version the P10-1 arch gate and the Qwen3-VL / H3 component loads were measured
# on. Bump it DELIBERATELY, never by drift.
#
# ⛔⛔ This literal is ALSO the CI gate's version. ``pyproject.toml``'s ``[h3-parity]`` extra declares
# the identical pin and ``tests/test_h3_processor_output_parity.py`` refuses to run against anything
# else — because the two versions differ in exactly the way that matters (``return_mm_token_type_ids``
# defaults True at 5.14.1, False at 5.2.x), so a parity diff at the wrong version passes while the
# real dispatch fails. Bumping this number without re-running that gate re-opens D-10-DEF-7.
#
# ⛔ NEVER set ``TRANSFORMERS_DISABLE_TORCH_CHECK=1`` on this image
# (``transformers/utils/import_utils.py``): it disables the total-pad-count vs vision-feature check
# in ``Qwen3VLModel.get_placeholder_mask``, which is the one loud guard this pipeline still gets for
# free on the conditioning path.
TRANSFORMERS_VERSION = "5.14.1"

# Build fresh (NOT chained off ``gpu_image`` or ``image``): both end with ``add_local_python_source``
# and Modal forbids any build step after ``add_local_*`` (see the same note on ``download_image``
# below). So every install step comes FIRST and ``add_local_python_source`` LAST.
h3_gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    # git: needed for the git+https diffusers install below.
    #
    # STILL NO ``ffmpeg`` apt package, and that is now a CHECKED conclusion rather than the original
    # default. 10-04 shipped this image without it and set the precondition verbatim: *"Add it only
    # if a later plan PROVES a demux is needed; an unused apt package is just image-build time."*
    # 10-10 MET that precondition — ``h3_preprocess`` decodes clips to RGB and demuxes audio, and
    # ``h3_sample`` muxes mp4s through diffusers' ``encode_video`` — so a decode/encode backend was
    # genuinely needed and is now DECLARED below (``av``).
    #
    # But the backend needed is PyAV, not the apt package. ``gpu_image`` installs system ``ffmpeg``
    # because **torchaudio**'s ``load()`` links against the FFmpeg SHARED LIBS; the H3 leg never
    # touches torchaudio — ``_h3_decode_rgb_frames`` and ``_h3_read_audio_waveform`` both go through
    # ``av`` (including ``av.audio.resampler`` for the rate conversion), and diffusers' video writer
    # muxes with PyAV too. PyAV's manylinux wheels carry FFmpeg statically, so the apt package would
    # be exactly the unused-package cost 10-04 warned about. If a torchaudio consumer ever lands on
    # this image, THAT is the change that re-opens the ffmpeg question.
    .apt_install("git")
    .pip_install("uv")
    # torch~=2.7 on the CUDA 12.9 wheel index — DELIBERATELY THE SAME LINE as ``gpu_image``'s above, so
    # nothing measured on the P10-1 probe carries a torch-version confound into the real backend.
    # DIVERGE / do-not-chase: ANALYSIS §1's "H3 wants cu130" claim was about the ComfyUI
    # ``int8_convrot`` backend and does NOT apply to this path — the diffusers path measured clean on
    # torch~=2.7 / cu129. Do not chase cu130 without new evidence.
    .run_commands(
        "uv pip install --system --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cu129 'torch~=2.7' 'torchvision~=0.22'"
    )
    .run_commands(
        # diffusers from git at the PINNED SHA — ``MiniMaxH3Transformer3DModel`` is not in any release.
        f"uv pip install --system 'diffusers @ git+https://github.com/huggingface/diffusers@{DIFFUSERS_SHA}'",
        "uv pip install --system 'peft>=0.14' 'accelerate>=1.2' "
        f"'transformers=={TRANSFORMERS_VERSION}' "
        "'safetensors' 'bitsandbytes' 'huggingface_hub>=0.27' 'av' "
        # ⛔ ``pydantic`` + ``pyyaml`` are the CONFIG-LOADER closure, and their absence is a defect
        # this repo has now paid for TWICE — the same ModuleNotFoundError, on two different images.
        # ``h3_train`` and ``h3_sample`` both call ``load_config_from_text`` in-container (the
        # T-03-63 revalidate-in-container convention), which drags:
        #     pydantic  <- config/schema.py
        #     yaml      <- config/load.py
        #     torch     <- NOT imported by any config module. config/validators.py re-exports
        #                  compute_seq_len from conditioning/strategy.py, which imports torch at
        #                  module scope. It arrives TRANSITIVELY, through a re-export.
        # ``gpu_image`` gets pydantic for free as an ``ltx-trainer`` transitive; this image
        # DELIBERATELY installs no ltx-* package, so nothing supplies it. ``h3_preprocess`` is not
        # affected — every one of its parameters is threaded in by the entrypoint, so it never loads
        # a config — which is exactly why the gap survived a green Stage 1 and died in Stage 2.
        # Versions MIRRORED from pyproject's [project.dependencies] floors, never invented; no
        # package new to the project is introduced.
        "'pydantic>=2.10.4' 'pyyaml>=6'",
    )
    # Same VRAM-fragmentation mitigation as ``gpu_image`` — must be a process env var set BEFORE
    # torch's CUDA allocator initializes, so it lives on the image, not in code.
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # CODE ONLY: copy the local signet_trainer package in. NO WEIGHTS (MODL-01) — the 134.1 GiB of
    # diffusers-format H3 components are ALREADY on the ``signe-trainer-weights`` Volume under
    # ``/minimax-h3/`` (pulled by the P10-1 probe) and mount at ``WEIGHTS_DIR / "minimax-h3"`` at run
    # time. Verified layout there: ``transformer_ref/`` (14 shards + index + config), ``text_encoder/``,
    # ``vae/``, ``audio_vae/``, ``scheduler/``, ``audio_scheduler/``, ``processor/``, ``tokenizer/``,
    # ``model_index.json``. Do NOT bake them in and do NOT add a weights pull here.
    .add_local_python_source("signet_trainer")
)

# --------------------------------------------------------------------------------------------------
# CPU download image (Phase 2) — the light, non-GPU image. Kept SEPARATE from the heavy ``gpu_image``:
# none of its users need ltx-core / ltx-trainer / CUDA, so it stays a fast CPU image. NO weights baked
# in (MODL-01) — files land on the mounted Volumes at runtime. (The bare code-only ``image`` has no
# third-party deps at all, so these fns MUST carry this image, not the App default.)
#
# THIS IMAGE IS SHARED BY THREE FUNCTIONS, AND THAT IS WHY ITS DEPENDENCY LIST IS NOT OBVIOUS:
#   * ``download_weights`` — pure HF I/O; needs ``huggingface_hub`` and nothing else.
#   * ``backup_sync`` / ``restore`` (BK-01) — ALSO call ``load_config_from_text`` to revalidate the
#     run config in-container (the T-03-63 convention). That single call drags in a much larger
#     third-party closure than "a config loader" suggests:
#         - ``pydantic``  <- config/schema.py
#         - ``yaml``      <- config/load.py
#         - ``torch``     <- NOT imported by any config module. config/validators.py re-exports
#                            ``compute_seq_len`` from conditioning/strategy.py, which does a
#                            module-scope ``import torch``. It arrives TRANSITIVELY, through a
#                            re-export, which is exactly why it is easy to miss.
#     Declaring only pydantic + pyyaml does not fix the bug — it moves the crash to ``torch``.
#
# WHAT THIS FIXES (verified live on Modal, 2026-08-05): ``--mode backup`` passed every local gate —
# dry-run OK, cost estimate under the guardrail, approval honored — and then died INSIDE the paid
# container with ``ModuleNotFoundError: No module named 'pydantic'`` at fns.py's
# ``from signet_trainer.config.load import load_config_from_text``. BK-01 backup and restore had never
# once executed. A passing local gate proves nothing about the container's site-packages.
#
# ``torch`` here is the CPU wheel ON PURPOSE (``index_url`` = PyTorch's own CPU index). These are
# CPU-only functions that never touch a GPU; a default-PyPI ``torch`` resolves to the multi-GB CUDA
# build and its nvidia-* transitives for nothing. It is a separate ``pip_install`` call because the
# ``index_url`` must apply to torch alone and NOT to huggingface_hub/pydantic/pyyaml.
#
# INVARIANT — any future ``@app.function`` pointed at this image INHERITS this import closure, and
# owes it a re-derivation first. Do not eyeball it; run:
#     PYTHONPATH=src python -c "import sys; b=set(sys.modules); import signet_trainer.config.load; \
#         print(sorted({m.split('.')[0] for m in set(sys.modules)-b} - sys.stdlib_module_names))"
# ``tests/test_backup_restore_fns.py`` now re-derives this automatically and fails CI if a root
# goes undeclared, so the discovery channel is a red test instead of a burned container.
#
# Versions are MIRRORED from ``pyproject.toml``'s ``[project.dependencies]`` floors, never invented:
# ``pydantic>=2.10.4``, ``pyyaml>=6``, ``torch>=2.6``. No package new to the project is introduced.
# --------------------------------------------------------------------------------------------------

# Build fresh (NOT chained off ``image``): Modal forbids any build step after ``add_local_*``, and
# ``image`` ends with ``add_local_python_source``. So every pip_install comes FIRST and
# ``add_local_python_source`` LAST. The local source is needed container-side because ``fns.py``
# imports ``signet_trainer.modal.app`` at module load.
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.27", "pydantic>=2.10.4", "pyyaml>=6")
    # CPU torch, on PyTorch's own CPU index — that index self-hosts torch's full dependency closure,
    # so no ``extra_index_url`` fallback is needed (and none is given: the resolver cannot drift to
    # another host). Its own call so the index_url scopes to torch only.
    .pip_install("torch>=2.6", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source("signet_trainer")
)

# --------------------------------------------------------------------------------------------------
# App (D-09 naming — load-bearing identifier). Constructed AFTER the Image so it can carry it as the
# default ``image=`` — every ``@app.function`` (incl. the CPU probe) then runs with the signet_trainer
# source available container-side. Phase 2 may override ``image=`` per GPU stage function to layer in
# the heavy [modal-runtime] extras only where needed.
# --------------------------------------------------------------------------------------------------

app = modal.App(APP_NAME, image=image)
