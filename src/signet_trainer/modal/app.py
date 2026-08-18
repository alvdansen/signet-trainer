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
# Qwen-Image-Edit GPU-stage image (Phase 11 — family #3) — the Qwen-Image-Edit-2511 chained-edit
# image. A THIRD image, and the decision to make it a third rather than widen ``h3_gpu_image`` is the
# load-bearing part of this block, so it is recorded rather than assumed.
#
# ⛔ WHY NOT REUSE ``h3_gpu_image``. The task that produced this block asked exactly that question of
# the ``transformers==5.14.1`` pin. The answer is that the transformers pin is not even the binding
# constraint — **``diffusers`` is**, and it is binding by construction:
#
#   * ``h3_gpu_image`` git-pins diffusers at ``DIFFUSERS_SHA`` (9f169d98…), the SHA the MiniMax-H3
#     arch was read and probed at, and its banner says to bump it DELIBERATELY.
#   * EVERY line-numbered fact family #3 is built on was measured against a DIFFERENT SHA —
#     ``072d15ee4289ffdc3aa9d65f8b94bc9271319d21`` (self-reporting ``0.36.0``). That is the SHA
#     ``ai-toolkit/requirements.txt:3`` pins, and it is the install every ``pipeline_qwenimage_edit_
#     plus.py:66-67`` / ``transformer_qwenimage.py:525-535`` / ``single_file_utils.py:160,566``
#     citation in ``conditioning/qwen_edit_geometry.py``, ``models/qwen_edit_loader.py`` and
#     ``prep/qwen_edit_encode.py`` was read from.
#
# One image cannot carry two diffusers SHAs. Re-pinning h3's to serve qwen would silently re-resolve
# ``MiniMaxH3Transformer3DModel`` — a surface its own banner records as "still moving" — and the
# working H3 path is not a thing to risk for image-build convenience. So the fork is forced, and the
# transformers question then resolves for free rather than being traded off.
#
# ⚠ ON THE ``transformers`` PIN, and what is and is not verified here. MEASURED on this box:
# ``Qwen2_5_VLProcessor`` and ``Qwen2_5_VLForConditionalGeneration`` both import at **4.57.3** AND at
# **5.1.0**, and ``Qwen2_5_VLProcessorKwargs._defaults["text_kwargs"]["return_mm_token_type_ids"]``
# is ``False`` in BOTH — so the specific hazard that forced h3's pin (the flag defaulting True at
# 5.14.1 on ``Qwen3VLProcessor``) is not visibly reproduced on the 2.5-VL processor at 5.1.0.
# [UNVERIFIED] It was NOT tested at 5.14.1: that version is installed nowhere on this machine, and a
# compatibility claim about a version nobody ran is the kind of number this repo refuses to write.
# What IS visible across the major boundary is a real surface change — at 4.57.3 the processor
# declares family-SPECIFIC ``Qwen2_5_VLImagesKwargs`` / ``Qwen2_5_VLVideosProcessorKwargs``, at 5.1.0
# those collapse into the generic ``ImagesKwargs`` / ``VideosKwargs``. ``prep/qwen_edit_encode.py``
# transcribes the diffusers-0.36 ``processor(text=[...], images=[...])`` call literally, so the
# defensible pin is the one the MEASURED house chain ran under: ``ai-toolkit/requirements.txt:4``,
# ``transformers==4.57.3``. Bump it DELIBERATELY and only with a parity run, never by drift.
#
# Supply chain (the ``LTX2_COMMIT_SHA`` / ``DIFFUSERS_SHA`` discipline, applied a third time):
#   * ``diffusers`` — git+https at a LITERAL 40-hex SHA, never ``main``. Same route
#     ``tests/test_pyproject_supply_chain.py`` sanctions; ``diffusers>=0.32`` is already a declared
#     ``[modal-runtime]`` dep, so this is a git PIN of an existing dependency, not a new package.
#   * ``optimum-quanto==0.2.4`` is the ONE distribution new to this project, and it is new because
#     the house recipe locks ``qfloat8`` on both the transformer and the text encoder — there is no
#     way to run this family's recipe without it. Legitimacy, read off the installed distribution's
#     own metadata in the ``falco`` env on this box (``importlib.metadata``): Name
#     ``optimum-quanto``, Version ``0.2.4``, License ``Apache-2.0``, Project-URL
#     ``homepage, https://github.com/huggingface/optimum-quanto`` — the same Hugging Face org that
#     publishes ``diffusers`` / ``transformers`` / ``peft`` / ``accelerate``, all four of which this
#     project already trusts. The version is EXACT and matches ``ai-toolkit/requirements.txt:29``,
#     which is not a preference: ``models/qwen_edit_loader.quantize_qwen_edit`` documents a live
#     ``include=``/``exclude=`` swap bug in this exact release and is written to keep it unreachable
#     by passing neither filter. A floating range could resolve past that reasoning.
#   * ``peft`` / ``accelerate`` / ``safetensors`` / ``bitsandbytes`` / ``huggingface_hub`` are
#     mirrored from ``h3_gpu_image`` at their existing floors — DELIBERATELY not re-pinned to
#     ai-toolkit's versions. signet writes its adapters through its OWN ``lora/peft.py``, shared with
#     the LTX and H3 legs; forking the adapter-writing surface per family for no measured reason is
#     how two families' checkpoints stop being comparable.
#   * ``pydantic`` + ``pyyaml`` are the CONFIG-LOADER closure — the defect this repo has now paid for
#     TWICE (see ``download_image``'s INVARIANT banner and ``h3_gpu_image``'s copy of the same note).
#     ``qwen_edit_train`` and ``qwen_edit_sample`` both call ``load_config_from_text`` in-container.
#   * NO ``ffmpeg``, NO ``av``. Family #3 is an IMAGE family — ``QWEN_EDIT_FRAMES`` is pinned to
#     exactly 1 (``conditioning/qwen_edit_geometry.py``) — so nothing on this path demuxes or muxes
#     anything. ``Pillow`` is what opens a control image (``prep/qwen_edit_encode.py`` and
#     ``prepare_qwen_edit_image`` both import ``PIL.Image`` function-locally); it arrives as a
#     ``torchvision`` transitive exactly as it does on ``gpu_image``, and is DECLARED here anyway for
#     the ``av`` reason 10-04 recorded: an import that shipped code performs must not depend on
#     somebody else's dependency graph. The day a Qwen leg needs a video container, THAT is the
#     change that re-opens the ffmpeg question.
#   * NO ``*.safetensors`` baked into the image (MODL-01). The three components live on the
#     ``signe-trainer-weights`` Volume and mount at ``WEIGHTS_DIR`` at run time — the transformer as
#     ``cfg.model.model_id``, the text encoder as ``cfg.model.text_encoder_id``, the VAE as
#     ``cfg.model.vae_id``. The CALLER composes every one of those paths (D-NOHARDCODE).
# --------------------------------------------------------------------------------------------------

# diffusers at the SHA every Qwen-Image-Edit line citation in this repo was measured against. A
# LITERAL SHA, never ``main``. Source: ``ai-toolkit/requirements.txt:3``. Bump DELIBERATELY — a
# different SHA invalidates the transcribed constants, not merely the build.
QWEN_DIFFUSERS_SHA = "072d15ee4289ffdc3aa9d65f8b94bc9271319d21"

# ⛔ DELIBERATELY NOT ``TRANSFORMERS_VERSION`` (h3's 5.14.1). See the ⚠ note above: the two families
# pin different majors, and the reason is recorded rather than resolved by picking the newer one.
# Source: ``ai-toolkit/requirements.txt:4``.
QWEN_TRANSFORMERS_VERSION = "4.57.3"

# The qfloat8 backend. EXACT, not a floor — ``quantize_qwen_edit``'s reasoning is written against
# this release's known ``include=``/``exclude=`` bug. Source: ``ai-toolkit/requirements.txt:29``.
QWEN_OPTIMUM_QUANTO_VERSION = "0.2.4"

# Build fresh (NOT chained off any image above): Modal forbids build steps after ``add_local_*``, and
# every image above ends with ``add_local_python_source``. Installs FIRST, local source LAST.
qwen_gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    # git: needed for the git+https diffusers install below. Nothing else — see the NO ffmpeg note.
    .apt_install("git")
    .pip_install("uv")
    # torch~=2.7 on the CUDA 12.9 wheel index — the SAME line as ``gpu_image`` and ``h3_gpu_image``,
    # so no family carries a torch-version confound relative to the others. ``torchvision`` is the
    # matched pair (and is what drags ``Pillow`` in, which is declared explicitly below regardless).
    .run_commands(
        "uv pip install --system --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cu129 'torch~=2.7' 'torchvision~=0.22'"
    )
    .run_commands(
        f"uv pip install --system 'diffusers @ git+https://github.com/huggingface/diffusers@{QWEN_DIFFUSERS_SHA}'",
        "uv pip install --system "
        f"'transformers=={QWEN_TRANSFORMERS_VERSION}' "
        f"'optimum-quanto=={QWEN_OPTIMUM_QUANTO_VERSION}' "
        "'peft>=0.14' 'accelerate>=1.2' 'safetensors' 'bitsandbytes' 'huggingface_hub>=0.27' "
        "'pillow' "
        # The config-loader closure (pydantic <- config/schema.py, yaml <- config/load.py; torch
        # arrives transitively through config/validators.py's re-export of compute_seq_len). Floors
        # MIRRORED from pyproject's [project.dependencies], never invented.
        "'pydantic>=2.10.4' 'pyyaml>=6'",
    )
    # Same VRAM-fragmentation mitigation as the other two GPU images — a process env var that must be
    # set BEFORE torch's CUDA allocator initializes, so it lives on the image, not in code.
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # CODE ONLY: copy the local signet_trainer package in. NO WEIGHTS (MODL-01).
    .add_local_python_source("signet_trainer")
)

# --------------------------------------------------------------------------------------------------
# Wan 2.1 / musubi-tuner RUNNER image (family #4) — a FOURTH image, and the least signet-shaped of
# the four, because it does not host signet's training loop at all. musubi-tuner owns the optimizer
# step here; this image exists to hold that checkout and its dependency closure. It is TRANSCRIBED
# from ``docs/source-methods/musubi-wan21/train_kohya.py:23-48`` — the one Wan 2.1 build this
# project has a working record of — and every deviation from that transcription is noted below.
#
# ⛔ PYTHON 3.10, NOT 3.11. Every other image in this file is 3.11; train_kohya.py:24 builds this one
# on 3.10 and the recipe was measured there. This is also the reason ``runners/wan_musubi.py`` and
# ``config/sources.py`` avoid 3.11-only syntax (``StrEnum``) — the family's own runtime is 3.10.
#
# ⛔⛔ THE PYDANTIC COLLISION, and why ``wan_train`` is a THREADED-PARAMETER stage. train_kohya.py:37
# pins ``pydantic==1.10.13`` into this image, AFTER musubi's own ``requirements.txt``; signet's
# ``config/load.load_config_from_text`` needs ``pydantic>=2.10.4``. One interpreter cannot carry
# both, so this image DELIBERATELY does NOT declare the config-loader closure that
# ``download_image`` / ``h3_gpu_image`` / ``qwen_gpu_image`` all carry, and ``modal/fns.wan_train``
# never calls ``load_config_from_text``. It takes its parameters threaded in from the entrypoint
# (the ``h3_preprocess`` shape) and imports only ``signet_trainer.runners.wan_musubi``, which is
# stdlib-only for exactly this reason. Adding pydantic v2 here to "fix" a config load would break
# musubi instead — silently, in a metered container.
#
# ⛔⛔⛔ SUPPLY CHAIN — THE ONE THING THAT IS NOT DONE, AND IT IS DELIBERATE, NOT AN OVERSIGHT.
# ``MUSUBI_TUNER_COMMIT_SHA`` is ``None``. Every other foreign checkout in this file
# (``LTX2_COMMIT_SHA``, ``DIFFUSERS_SHA``, ``QWEN_DIFFUSERS_SHA``) is a LITERAL 40-hex SHA and each
# banner says to bump it deliberately, because an unpinned ``main`` makes an image build silently
# non-reproducible — the image that trains an adapter must be the image that renders it.
# train_kohya.py:31 clones ``main`` with no checkout at all, so there is no SHA to transcribe, and
# this pass performs ZERO downloads (CPU-only, no network), so there is no SHA to resolve either.
# Writing a plausible-looking hex string would be fabrication of exactly the kind the pins exist to
# prevent. So the gap is DECLARED and defended twice over:
#
#   1. ``modal/entrypoint._wan_config_gaps`` refuses the dispatch at $0, naming the one command that
#      lands it: ``gh api repos/kohya-ss/musubi-tuner/commits/main --jq .sha``.
#   2. The build step below FAILS LOUDLY if it is ever reached unpinned, rather than falling back to
#      a floating ``main``. A build that dies saying why costs one container start; a build that
#      quietly takes today's ``main`` costs the reproducibility of every adapter it produces.
#
# NO WEIGHTS BAKED IN (MODL-01) — and this is the sharpest divergence from the transcription.
# train_kohya.py:69-92 ``hf_hub_download``s all four components (DiT, VAE, umT5, open-CLIP) from two
# HARDCODED Hub repo ids INSIDE the training function, on every container start. signet resolves
# them from the weights Volume instead (``WEIGHTS_DIR / <id>``, the caller composing every path),
# which is what makes a round reproducible, air-gappable, and not re-pulling 30+ GiB per retry.
# ``runners/wan_musubi.wan_resolve_component_ids`` is the named symbol that lands the config fields.
#
# NO ``.add_local_file("wan21-dataset-config.toml", ...)``. train_kohya.py:40-43 bakes the dataset
# TOML into the IMAGE, which makes changing the dataset an image REBUILD and makes "what did round 2
# train on?" unanswerable from the artifacts. signet renders it from the manifest at DISPATCH time
# (``runners/musubi_toml.render_from_config``), ships it BY VALUE, and writes it beside the adapter
# on the checkpoints Volume — so "only the dataset changed between rounds" becomes a diff of two
# committed files instead of a memory.
# --------------------------------------------------------------------------------------------------

#: Where the musubi-tuner checkout lands in the image. train_kohya.py clones into the build CWD and
#: then ``.workdir("musubi-tuner")``; an absolute path is named here instead so ``wan_train`` can set
#: it as the subprocess CWD explicitly rather than inheriting one.
MUSUBI_TUNER_ROOT = "/opt/musubi-tuner"

#: PINNED 2026-08-12 — resolved with the command the banner above names
#: (``gh api repos/kohya-ss/musubi-tuner/commits/main --jq .sha``) and transcribed literally.
#:
#: train_kohya.py:31 clones ``main`` with no checkout, so there was no SHA to transcribe from
#: the oracle and this sat as a declared gap. It is resolved here rather than left floating
#: because musubi's dataset-config SCHEMA is what ``runners/musubi_toml.py`` transcribes: an
#: upstream key rename would otherwise be discovered by a rejected TOML inside a metered
#: container, and every adapter this image produces would be unattributable to a code state.
#: Bump deliberately, the same rule the three sibling pins carry.
MUSUBI_TUNER_COMMIT_SHA: str | None = "8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6"

#: The checkout step, or a loud refusal standing in for it. Never a fallback to ``main``: a floating
#: clone is the failure the three sibling SHA pins exist to prevent, and it is worse here than
#: elsewhere because musubi's dataset-config SCHEMA is what ``runners/musubi_toml.py`` transcribes —
#: upstream adding or renaming a key would be discovered by a rejected TOML inside a metered
#: container.
_MUSUBI_CHECKOUT_COMMAND = (
    f"cd {MUSUBI_TUNER_ROOT} && git checkout {MUSUBI_TUNER_COMMIT_SHA}"
    if MUSUBI_TUNER_COMMIT_SHA
    else (
        "echo 'signet: MUSUBI_TUNER_COMMIT_SHA is UNPINNED in modal/app.py. Refusing to build "
        "against a floating main — pin the literal 40-hex SHA "
        "(gh api repos/kohya-ss/musubi-tuner/commits/main --jq .sha) and rebuild.' >&2 && exit 1"
    )
)

wan_musubi_image = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"HF_HUB_CACHE": "/cache/cache/"})
    .apt_install("git", "ffmpeg", "python3-opencv")
    # ⚠ DELIBERATE DEVIATION from the transcription, and the gate caught it rather than a reviewer.
    # train_kohya.py:29 pre-installs ``transformers>=4.46.0`` BEFORE musubi's own requirements.txt.
    # That floating range is exactly what ``tests/test_modal_gpu_image.py::
    # test_h3_transformers_pin_is_a_literal_version`` bans repo-wide, and its reason applies here
    # verbatim: a range decides a conditioning-critical version at BUILD time, by whatever the
    # resolver picked that day, so the image that trains an adapter need not be the image that
    # renders it. The other families answer that with a literal ``==`` pin; this one CANNOT, because
    # no transformers version has been measured for musubi in this program and inventing a number
    # to satisfy a gate is worse than the gate firing. So the pre-install is DROPPED and musubi's
    # own ``requirements.txt`` — installed below, from the checkout — is left as the single
    # authority on its own dependency. That makes the build's reproducibility rest entirely on
    # MUSUBI_TUNER_COMMIT_SHA, which is precisely the property the ⛔⛔⛔ banner above demands and
    # ``_wan_config_gaps`` gap (6) refuses to dispatch without.
    .pip_install("huggingface_hub")
    .run_commands(
        f"git clone https://github.com/kohya-ss/musubi-tuner.git {MUSUBI_TUNER_ROOT}",
        _MUSUBI_CHECKOUT_COMMAND,
        f"cd {MUSUBI_TUNER_ROOT} && pip3 install torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu124",
        f"cd {MUSUBI_TUNER_ROOT} && pip3 install -r requirements.txt",
    )
    # ⛔ pydantic 1.10.13 — musubi's pin, and the reason this image cannot load a SignetConfig. See
    # the ⛔⛔ banner. ``albumentations==1.4.3`` and ``hf_transfer`` are likewise transcribed.
    .pip_install("pydantic==1.10.13", "hf_transfer", "albumentations==1.4.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # CODE ONLY: the signet package, for ``signet_trainer.modal.app`` (imported by fns.py at module
    # load) and ``signet_trainer.runners.wan_musubi`` (the recipe + argv builders). NO WEIGHTS, and
    # no dataset TOML — both are run-time inputs here, not build-time ones. Last, because Modal
    # forbids any build step after ``add_local_*``.
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
