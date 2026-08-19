# LTX-2.5 Upstream Diff — Gate Report for signet-trainer

**Scope**: diff of `Lightricks/LTX-2` between signet-trainer's current pin
`LTX2_COMMIT_SHA = d6053703e00195bc668cbd1d5eda9dc0b2e7b74a` (2026-05-28) and the
`v1.2.0` tag (commit `d151147788a9284cca791edc6ce898007e727fe6`, 2026-08-11, "Support
for LTX 2.5"). Verified: `d6053703` **is** an ancestor of `v1.2.0` (8 commits between
them, squashed into 3 "Automated PR" merges dated 2026-06-17 / 2026-07-07 / 2026-08-11).

**Method**: full clone of `Lightricks/LTX-2` into the scratchpad; `git diff`/`git show`
against both revisions for every file that defines a symbol signet-trainer imports;
cross-checked against signet-trainer's read-only clone at
`C:/Users/minta/Projects/signet_trainer_upkeep/signet-trainer`. No upstream code was
executed — this is a static diff + call-site audit, not a live import/smoke test.

**Headline verdict**: this is not a version bump, it is a rewrite of every layer signet
ported by hand. Two upstream signature changes (`load_embeddings_processor`,
`ICLoraPipeline.__init__`) will hard-crash signet's existing call sites as literally
written. The validation/sampling module signet ported line-by-line
(`ltx_trainer.validation_sampler.{GenerationConfig,ValidationSampler}`) no longer
exists — it was replaced by a self-contained `ValidationRunner` with an inverted
control-flow (loads its own models, writes its own files) that is incompatible with
signet's two-phase-VRAM component-injection design. Torch/CUDA itself does not need to
move. Recommendation is in §E.

---

## (A) Consumed-symbol compatibility matrix

Built from every `from ltx_core|ltx_trainer|ltx_pipelines import` / `import ltx_core.*`
etc. in `signet-trainer/src`, resolved against both revisions. "Consumed" = actually
imported and executed at a call site (not a docstring mention of the port source).

| Symbol | signet call site | Pin (d6053703) | v1.2.0 | Verdict | Confidence |
|---|---|---|---|---|---|
| `ltx_trainer.model_loader.load_model` | `models/loader.py:202` | `load_model(checkpoint_path, text_encoder_path, device, dtype, with_video_vae_encoder, with_video_vae_decoder, with_audio_vae_decoder, with_vocoder, with_text_encoder)` | Same kwargs still accepted; gained **optional** `video_vae_path=None, audio_vae_path=None` (split-pack overrides, default to monolith `checkpoint_path`). **Still has no `with_audio_vae_encoder` kwarg and `LtxModelComponents` still has no `audio_vae_encoder` field** — signet's "AT-PIN DIVERGENCE" bolt-on comment (`models/loader.py:154-162`) is confirmed to still describe the truth at v1.2.0. | **UNCHANGED / call-compatible** — signet's existing call is byte-safe verbatim. To ingest a *split* LTX-2.5 pack, signet would need to start passing `video_vae_path=` / `audio_vae_path=` explicitly (resolved from the split files); omitting them defaults to treating `checkpoint_path` as a monolith and will raise inside `resolve_video_vae_path`/`resolve_audio_vae_path` if `checkpoint_path` is actually a split transformer-only file (`is_split_transformer()` check). | HIGH |
| `ltx_trainer.model_loader.load_audio_vae_encoder` | `models/loader.py:231, 241-245` | `load_audio_vae_encoder(checkpoint_path, device, dtype)` | **Byte-identical function body** at v1.2.0 (confirmed via `git diff` — zero diff). | **UNCHANGED**, but this standalone function was never extended with split-pack path resolution the way `load_model`'s internal VAE/decoder loaders were. If `checkpoint_path` passed to it is a split transformer-only file, it silently builds an uninitialized (`meta`-device) encoder — `SingleGPUModelBuilder._return_model` logs a warning and returns the meta model rather than raising. **Silent-failure risk on split checkpoints** unless signet resolves the audio-VAE file itself (via the new `resolve_audio_vae_path()`, see row below) before calling this. | HIGH |
| `ltx_trainer.model_loader.load_embeddings_processor` | `models/loader.py:264-268`; `modal/fns.py:464-469, 986-999` | `load_embeddings_processor(checkpoint_path, device="cpu", dtype=...)` | **SIGNATURE-CHANGED — breaking.** Gained a new **required, no-default** second positional parameter `gemma_model_path: str | Path` between `checkpoint_path` and `device` (used to size the feature-extractor/connector projections from the *specific* Gemma root's `hidden_size`/`num_hidden_layers`). | **BREAKS.** All three signet call sites pass only `checkpoint_path=` and `device=`/`dtype=` as keywords — none pass `gemma_model_path`. At v1.2.0 every one of these calls raises `TypeError: load_embeddings_processor() missing 1 required positional argument: 'gemma_model_path'`. This is the single highest-impact, most mechanical break in the diff — every inference/validation render path in signet goes through this function. | HIGH |
| `ltx_trainer.model_loader.load_text_encoder` | `modal/fns.py:992` | `load_text_encoder(gemma_model_path, device, dtype, load_in_8bit=False)` | Same first-positional/`device=`/`dtype=` shape; internals rewritten to auto-detect Gemma3 vs Gemma4 via `get_gemma_ops()` + `resolve_gemma_weight_paths()` (replacing the old `GEMMA_LLM_KEY_OPS`/`GEMMA_MODEL_OPS`/`find_matching_file` idiom the CHANGELOG marks Removed). Return type renamed `GemmaTextEncoder` → `LTXGemmaTextEncoder` (class itself renamed/removed per CHANGELOG). | **UNCHANGED for signet's call shape** (positional `text_encoder_path` + `device=` kwarg still works). The renamed return class is transparent to signet because signet never imports `GemmaTextEncoder` by name — it only calls `.encode()` on the returned object. | HIGH |
| `ltx_trainer.model_loader.LTXV_MODEL_COMFY_RENAMING_MAP` | `local/runner.py:289`; `modal/fns.py:542` | **Does not exist in `ltx_trainer.model_loader`** at the pin — it is defined in `ltx_core.model.transformer.model_configurator` and re-exported via `ltx_core.model.transformer.__init__`. `ltx_trainer/model_loader.py` never re-exports it at module scope (only under `TYPE_CHECKING`-guarded, unrelated imports). | Identical: still lives only in `ltx_core.model.transformer.model_configurator` / `__init__`. | **PRE-EXISTING BUG, not a 2.5 regression.** `from ltx_trainer.model_loader import LTXV_MODEL_COMFY_RENAMING_MAP` is a wrong import path **at the pin already** — it would raise `ImportError` if this branch ever executed. Verified by grepping the actual symbol definition site at both revisions; `ltx_trainer/model_loader.py`'s own top-level (non-`TYPE_CHECKING`) imports never touch this name. | HIGH |
| `ltx_trainer.model_builder.{LTXModelConfigurator, SingleGPUModelBuilder}` | `local/runner.py:288`; `modal/fns.py:538-541` | **Module `ltx_trainer.model_builder` does not exist** at either revision — confirmed via `git cat-file -e` (fatal: path does not exist) at both the pin and v1.2.0. The real locations are `ltx_core.model.transformer.model_configurator.LTXModelConfigurator` and `ltx_core.loader.single_gpu_model_builder.SingleGPUModelBuilder` (also re-exported at `ltx_core.loader`). | Same non-existent module at v1.2.0. | **PRE-EXISTING BUG, not a 2.5 regression.** This import fails identically at both revisions. Additionally, even with the path fixed, signet's call — `SingleGPUModelBuilder(model_path=..., configurator=LTXModelConfigurator(), renaming_map=LTXV_MODEL_COMFY_RENAMING_MAP)` — uses **wrong keyword names** against the real (frozen) dataclass, whose first field is `model_class_configurator: type[ModelConfigurator[ModelType]]` (a **class**, not an instance) and which has **no** `renaming_map=` field (renaming goes through `model_sd_ops=`). Confirmed unchanged at v1.2.0. This is signet's `use_builder` fallback branch (triggered when the validation-gate's forward-pass check selects `open_q1="use_builder"`) — it is dead-until-triggered code that will crash the moment it is ever exercised, on LTX-2.3 or LTX-2.5 alike. Flagging because any PR that touches these lines for the 2.5 bump should fix this in the same pass. | HIGH |
| `ltx_trainer.validation_sampler.GenerationConfig` | `inference/sampler.py` (`build_generation_config`, TYPE_CHECKING import); `inference/upscale.py` (TYPE_CHECKING import) | `@dataclass class GenerationConfig` in `ltx_trainer/validation_sampler.py` (874-line module). | **REMOVED.** The entire module `ltx_trainer/validation_sampler.py` is deleted (confirmed: 874 → 0 lines across the 2026-06-17 squash) and replaced by `ltx_trainer/validation_runner.py` (1441 lines at v1.2.0). No class named `GenerationConfig` exists anywhere in the new module (checked every `@dataclass`/`class` in `validation_runner.py`: `CachedPromptEmbeddings`, `CachedConditionMedia`, `CachedSampleMedia`, `ValidationRunner`). The per-prompt ad hoc config object signet's `_generation_kwargs()`/`build_generation_config()` builds and passes around no longer has an upstream counterpart — the closest analog is `ltx_trainer.config.ValidationConfig`/`ValidationSample`, which is a **batch YAML schema** consumed by `ValidationRunner.__init__`, not a per-call kwargs dataclass. | **BREAKS — architectural, not mechanical.** `inference/sampler.py::build_generation_config` cannot be patched by an import-path fix; the abstraction it wraps is gone. | HIGH |
| `ltx_trainer.validation_sampler.ValidationSampler` | `inference/sampler.py::build_validation_sampler` (constructs `ValidationSampler(transformer, vae_decoder=, vae_encoder=, text_encoder=, embeddings_processor=, audio_decoder=None, vocoder=None)`, then calls `.generate(config=, device=)`) | `class ValidationSampler` — accepted **pre-loaded component objects**, exposed `.generate(config, device) -> video_tensor` (or `(video, audio)`), and private internals (`_get_prompt_embeddings`, `_create_video_latent_tools`, `_apply_image_conditioning`, `_run_denoising`) that signet's inpaint/multi-condition/a2v ports read directly. | **REMOVED / architecturally replaced by `ValidationRunner`** (`ltx_trainer/validation_runner.py:140`). `ValidationRunner.__init__(config: ValidationConfig, model_path, text_encoder_path, video_vae_path=None, audio_vae_path=None, load_text_encoder_in_8bit=False)` takes **checkpoint paths, not pre-loaded objects** — it loads its own text encoder + embeddings processor, caches prompt embeddings, unloads them, loads the VAE encoder, encodes conditioning media, unloads, then loads the decoder/audio decoder/vocoder and keeps them on CPU — an entirely self-managed lifecycle. `run(transformer, step, output_dir, device, progress: TrainingProgress, wandb_run=None, work_items=None) -> list[tuple[int, Path]]` takes the (PEFT-wrapped) transformer but writes output *files* itself rather than returning a raw tensor, and requires a `TrainingProgress` object from `ltx_trainer.progress`. New imports inside the module (`MultiModalGuider`, `EulerDiffusionStep`, `BatchedPerturbationConfig`, `VideoConditionByReferenceLatent`, `DiffusionVideoDecoder`, `AudioLatentTools`/`VideoLatentTools`) confirm the denoise loop's internal structure changed completely — none of the private method names signet's ported inpaint/multi-condition/frozen-audio code depends on (`_apply_image_conditioning`, `_run_denoising`, `_create_video_latent_tools`) are known to survive under those names. | **BREAKS — full rewrite required, not a patch.** This is the single largest-blast-radius finding in the diff: it invalidates `inference/sampler.py`, `inference/multi_condition.py`, `inference/reference_video.py`'s sampler-adjacent code, and the a2v frozen-audio render path (`_render_video_with_frozen_audio`), all of which were hand-ported against `ValidationSampler`'s *private* surface. | HIGH |
| `ltx_trainer.validation_sampler.CachedPromptEmbeddings` | `modal/fns.py:465, 481-486, 990` | `@dataclass class CachedPromptEmbeddings: video_context_positive, audio_context_positive, video_context_negative=None, audio_context_negative=None` | **MOVED, field-compatible.** Same class name and identical field names/order/defaults, now defined in `ltx_trainer/validation_runner.py:107-114`. | **MOVED (import path only)** — once the module path (`ltx_trainer.validation_sampler` → `ltx_trainer.validation_runner`) is updated, signet's construction of this dataclass is a drop-in match. This is the *only* one of the removed `validation_sampler` symbols that survives as a simple import-path fix. | HIGH |
| `ltx_core.loader.{apply_loras, StateDict, LoraStateDictWithStrength}` | `modal/fuse.py:416-420`; `modal/fuse.py:2358` | `apply_loras(model_sd, lora_sd_and_strengths, fuse_rule=bf16_fuse_rule, destination_sd=None)`; `StateDict`/`LoraStateDictWithStrength` dataclasses; all re-exported at `ltx_core.loader.__init__`. | `apply_loras` gained one new **optional, default-`True`** kwarg `preserve_input_device: bool = True` — every existing positional/keyword call is unaffected. `StateDict`, `LoraStateDictWithStrength` unchanged (no diff in `primitives.py`'s dataclass bodies — only `Protocol`/`TypeVar` typing scaffolding changed). All three names still exported from `ltx_core.loader.__init__` (confirmed via diff: only `StateDictRegistry` was removed from that `__init__`, unrelated). | **UNCHANGED** — `modal/fuse.py`'s canonical fuse math is untouched by the 2.5 bump. | HIGH |
| `ltx_core.loader.LTXV_LORA_COMFY_RENAMING_MAP` | `modal/fuse.py`; `inference/ic_lora_pipeline.py:156` | Exported from `ltx_core.loader.sd_ops` / re-exported at `ltx_core.loader.__init__`. | Unchanged — present in the same unchanged context lines of the `__init__.py` diff. | **UNCHANGED** | HIGH |
| `ltx_pipelines.ic_lora.{ICLoraPipeline, LoraPathStrengthAndSDOps, OffloadMode}` | `inference/ic_lora_pipeline.py:157-179` (constructs `ICLoraPipeline(distilled_checkpoint_path=, spatial_upsampler_path=, gemma_root=, loras=[...])`) | `ICLoraPipeline.__init__(self, distilled_checkpoint_path: str, spatial_upsampler_path: str, gemma_root: str, loras, device=None, quantization=None, registry=None, compilation_config=None, offload_mode=OffloadMode.NONE)` | **SIGNATURE-CHANGED — breaking.** `distilled_checkpoint_path` and `gemma_root` are **collapsed into a single new parameter `model_paths: ModelPaths`** (`ltx_pipelines/utils/model_paths.py`, a frozen dataclass with `mode: "monolith"|"split"`, `transformer_path`, `text_encoder_path`, `video_vae_path`, `audio_vae_path`, `duration_head_path`, built via `ModelPaths.from_monolith(checkpoint_path, gemma_root, video_vae_path=None)` or `.from_split(...)`). Three new kwargs also added (`alloc_trim_strategy`, `prompt_enhancer_gemma_root`, `diffvae_optimization`), all with defaults. | **BREAKS.** Signet's call passes `distilled_checkpoint_path=` and `gemma_root=` as keywords — neither name exists on the v1.2.0 constructor. `TypeError: unexpected keyword arguments`. This is the same "collapse two paths into a `ModelPaths` bundle" pattern used across `ltx_pipelines` for split-layout support, and it is the **second confirmed hard break** (after `load_embeddings_processor`) on a call site signet's own docstring marks as *already gated-GPU verified* ("RE-VALIDATED live at the 07-11 gated GPU exercise... verified against the pinned SHA") — i.e., this is a regression against previously-proven-working code, not merely an unexercised path. | HIGH |
| `ltx_pipelines.ti2vid_two_stages.TI2VidTwoStagesPipeline` | `inference/upscale.py:90-106` (constructs with `transformer=, vae_decoder=, vae_encoder=, text_encoder=, embeddings_processor=, distilled_checkpoint_path=, upscaler_checkpoint_path=, distilled_lora_path=, offload_mode=, device=`) | Real constructor at the pin: `__init__(self, checkpoint_path: str, distilled_lora: list[LoraPathStrengthAndSDOps], spatial_upsampler_path: str, gemma_root: str, loras: list[...], device=None, quantization=None, registry=None, compilation_config=None, offload_mode=OffloadMode.NONE)` — **none** of signet's kwarg names (`transformer`, `vae_decoder`, `vae_encoder`, `text_encoder`, `embeddings_processor`, `distilled_checkpoint_path`, `upscaler_checkpoint_path`, `distilled_lora_path`) match the real signature, even at the pin. | At v1.2.0, `checkpoint_path`+`gemma_root` are further collapsed into `model_paths: ModelPaths` (same pattern as `ICLoraPipeline`), plus the same three new optional kwargs. | **Not a new regression** — signet's own module docstring already flags this call as MEDIUM-confidence/unexercised ("the exact constructor + call signature... re-validated only when the toggle is exercised on a gated GPU run"; `two_stage_upscale` defaults OFF). It was wrong at the pin and is wrong (differently) at v1.2.0; either way this module needs a from-scratch rewrite against the real signature before the toggle is ever turned on, independent of the 2.5 question. | HIGH |
| `ltx_core.components.patchifiers.{VideoLatentPatchifier, AudioPatchifier, get_pixel_coords}` | `train/step.py:78-81`; `conditioning/a2v.py:132` | present, unchanged API | `git diff` on `patchifiers.py` between the two revisions is **empty** — byte-identical. | **UNCHANGED** | HIGH |
| `ltx_core.model.transformer.modality.Modality` | `train/step.py:82` | class present | 8 lines added, purely additive (new optional fields on the surrounding `LatentState`/related types, not on `Modality` itself per the file-level diff). | **UNCHANGED (additive only)** | HIGH |
| `ltx_core.types.{SpatioTemporalScaleFactors, VideoLatentShape, AudioLatentShape, LatentState, Audio}` | `train/step.py:83-86`; `conditioning/a2v.py:136`; `inference/sampler.py:883, 940` | present | `types.py` diff is **89 insertions, 0 deletions** — every existing class/field is untouched; new additions are `SpatioTemporalScaleFactors.from_blocks()`/`.from_model_config()` (checkpoint-metadata-derived factors), a new `GeneratedKeyframeLayout` dataclass, and new optional fields on `LatentState` (`keyframes_mask`, `generated_keyframe_layout`, `generated_keyframes`, `frozen`, all defaulted). `SpatioTemporalScaleFactors.default()` itself (which signet's `train/step.py::build_step_deps` calls) is unchanged and still returns `(time=8, height=32, width=32)`. | **UNCHANGED for signet's call shape**, but see §C — the *mechanism* signet mirrors this constant from (`base_strategy.py`'s module-level `VIDEO_SCALE_FACTORS`) was removed upstream in favor of an instance attribute that callers can override with checkpoint-derived values. Signet's own hardcoded `TIME_SCALE=8/HEIGHT_SCALE=32/WIDTH_SCALE=32` in `conditioning/strategy.py` and `EXPECTED_VAE_TEMPORAL_COMPRESSION=8/EXPECTED_VAE_SPATIAL_COMPRESSION=32` in `models/loader.py` are exactly the class of hardcoded assumption the CHANGELOG's "derive spatial/temporal compression from checkpoint metadata instead of assuming 32x32x8" line is about — they will silently mis-shape RoPE grids/latent bucketing on any LTX-2.5 checkpoint whose VAE does not use the 8/32/32 default. | HIGH |
| `ltx_core.components.noisers.GaussianNoiser` | `inference/sampler.py:435, 554-555, 939, 949-967`; `inference/multi_condition.py:191, 264-265` | `__call__(self, latent_state, noise_scale=1.0)`: `scaled_mask = denoise_mask*noise_scale; latent = noise*scaled_mask + latent_state.latent*(1-scaled_mask)` | **Internals rewritten** (constructor unchanged — still just `generator: torch.Generator`). New body: `step1 = lerp(latent.float(), noise.float(), noise_scale)` (unconditional on the mask), then `final = lerp(clean_latent.float(), step1, denoise_mask)`. Mathematically identical to the old formula **only when `latent_state.latent == latent_state.clean_latent` at call time** (true for signet's every call site, which always noises a just-built `clean_state`/`*_clean_state` — verified: `noiser(latent_state=clean_state, ...)`, `noiser(latent_state=video_clean_state, ...)`, `noiser(latent_state=audio_clean_state, ...)`, never a partially-denoised intermediate state). | **UNCHANGED IN PRACTICE for signet's usage** (all call sites noise a fresh `*_clean_state`, so `latent == clean_latent` holds and the two formulas agree), but this is a materially different internal algorithm and any *future* call that reuses `GaussianNoiser` on a non-fresh state (e.g. a resumed/partially-denoised `LatentState`) would silently diverge from pre-2.5 behavior. Flag for anyone porting the a2v/multi-condition denoise loop forward. | MEDIUM |
| `ltx_trainer.datasets.PrecomputedDataset` | `data/precomputed.py` (explicitly ported, not imported) | reference port source | `git diff` on `datasets.py` is **7 lines removed, 1 added** — pure logging-message cleanup, no signature or behavior change to the class signet ported. | **UNCHANGED** — signet's `PrecomputedDataset` port stays byte-faithful to the current upstream shape. | HIGH |
| `ltx_trainer.training_strategies.{base_strategy.TrainingStrategy, text_to_video.TextToVideoStrategy}` | `conditioning/strategy.py`, `conditioning/single_frame.py` (explicitly ported, not imported at runtime — these modules import stdlib+torch only) | reference port source: `TrainingStrategy.get_data_sources() -> list[str] \| dict[str,str]` (abstract on the strategy instance); module-level `VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()`; `ModelInputs.video: Modality` (required), `.video_targets: Tensor` (required), `.video_loss_mask: Tensor` (required), `.ref_seq_len: int \| None` field present. | `base_strategy.py` diff: **20 insertions / 39 deletions**. `get_data_sources()` **moved** to `TrainingStrategyConfigBase` (a pydantic config class, not the runtime strategy) and its return type **narrowed to `dict[str, str]` only** (list form removed). Module-level `VIDEO_SCALE_FACTORS` constant **removed**, replaced by `self.video_scale_factors = SpatioTemporalScaleFactors.default()` set in `TrainingStrategy.__init__` as an **overridable instance attribute**. `ModelInputs` fields became `Optional` (`video: Modality \| None`, `video_targets: Tensor \| None`, `video_loss_mask: Tensor \| None`) and `ref_seq_len` was **removed**. A new `flexible.py` strategy file was added; `requires_audio` property removed from the base class. | **PORT DRIFT, not a runtime import break** (signet never imports these at runtime, so nothing crashes). But signet's mirrored contract — `get_data_sources()` returning a `Sequence[Any]` list (`conditioning/strategy.py:107-118` returns `["latents","conditions","audio_latents"]`), the module-level scale-factor constants, and the non-Optional `ModelInputs` shape — is now stale relative to upstream's actual v1.2.0 design. Anyone re-syncing signet's hand-rolled strategy layer against the current upstream shape (e.g. to pick up checkpoint-derived scale factors properly) needs to re-port against this new shape, not the pinned one. | MEDIUM |
| `ltx_core.model.transformer.model_configurator.LTXModelConfigurator` (indirect, via the broken `use_builder` fallback) | see above | `.from_config(config: dict, ops=...) -> LTXModel` | **REMOVED**, replaced by `.from_metadata(metadata: dict, ops=...) -> LTXModel`, which does `config = metadata.get("config", {})` internally — i.e. it now wants the **full safetensors metadata dict**, not just the inner `config` sub-dict. Also gained new config-driven fields with 2.5-relevant defaults: `use_prompt_adaln_single` (default `True`; **KV-cacheable / LTX-2.5-style checkpoints set it `False`**), `ff_bias`/`audio_ff_bias` (default `True`; **"LTX 2.5 (gemma4) sets `ff_bias=false`"** per an inline comment — i.e. the 2.5 transformer's FFN blocks drop bias tensors entirely, a real architectural difference, not just a config toggle), `use_keyframes_abs_pos_embedding` (default `False`; DFRPipeline generated-keyframe checkpoints only). A new `LTXAudioOnlyModelConfigurator` class was also added. | **Transparent to signet's *working* call paths** (this method is called internally by `SingleGPUModelBuilder.meta_model()`/`.build()`, never directly by signet) — but is the concrete confirmation that the 2.5 transformer has a real architectural delta (no FFN bias) beyond just "more parameters," relevant to §C. Only touches signet if/when the already-broken `use_builder` fallback branch is fixed and exercised. | HIGH |

**Symbols NOT found to have any breaking change**: `LTX2Scheduler` (byte-identical `schedulers.py`), `VideoLatentPatchifier`/`AudioPatchifier`/`get_pixel_coords` (byte-identical `patchifiers.py`), `apply_loras`/`StateDict`/`LoraStateDictWithStrength`/`LTXV_LORA_COMFY_RENAMING_MAP`/`LTXV_MODEL_COMFY_RENAMING_MAP` (all present, unchanged export surface), `PrecomputedDataset` (near-identical).

---

## (B) Torch/CUDA pin verdict vs the tri-family invariant

Signet's tri-family invariant (`modal/app.py:154-161, 279-286, 418-423`): all three GPU
images (`gpu_image` for LTX-2, `h3_gpu_image` for MiniMax-H3, and the Qwen-Edit image)
install `'torch~=2.7' 'torchvision~=0.22'` from the `cu129` PyTorch wheel index via the
*same literal `uv pip install` line*, deliberately, before any editable/git install.

**Verdict: the tri-family invariant does NOT need to change to bump `LTX2_COMMIT_SHA`
to v1.2.0.** Evidence:

- `packages/ltx-core/pyproject.toml`'s base `dependencies` array still declares
  `"torch~=2.7"` unchanged at v1.2.0 (confirmed via `git diff` — that line is untouched
  context, not a `+`/`-` line).
- `packages/ltx-trainer/pyproject.toml`'s base dependency is `"torch>=2.6.0"`,
  also unchanged.
- The new `cu132`/`torch==2.13.0` pin is **gated behind a brand-new, non-default
  optional extra**: `[project.optional-dependencies] natten = ["natten==0.21.7+torch2130cu132; sys_platform=='linux'...", "torch==2.13.0; sys_platform=='linux'..."]`. This extra is for the new "optional NATTEN acceleration" diffusion-VAE decode path (CHANGELOG). Signet's install command is `uv pip install ... -e packages/ltx-core -e packages/ltx-trainer` — **no `[natten]` extra is requested**, so this constraint is never activated.
- The root `pyproject.toml` gained a separate, also-opt-in `ltx-kernels` workspace member (`kernels` dependency group, `uv sync --group kernels`) for compiled CUDA/Blackwell NVFP4 kernels — excluded from the default workspace (`exclude = ["packages/ltx-kernels"]`), also not something signet's plain `uv pip install -e packages/ltx-core -e packages/ltx-trainer` command touches.

**One real, but non-torch, supply-chain tightening**: `ltx-core`'s `transformers` pin
changed from an **unbounded floor** `transformers>=4.52` (at the pin) to a **bounded
window** `transformers>=5.8.0,<5.15` (at v1.2.0), with an inline comment explaining the
upper bound exists because `transformers>=5.15` breaks Gemma-4 config attribute access.
Signet's `gpu_image` does not pin `transformers` itself (unlike the H3 and Qwen images,
which pin `5.14.1`/`4.57.3` respectively via `TRANSFORMERS_VERSION`/
`QWEN_TRANSFORMERS_VERSION` in `modal/app.py`) — it lets `uv` resolve `transformers`
transitively from whatever `ltx-core`/`ltx-trainer` declare. At the pin, an unbounded
`>=4.52` floor resolves to "whatever the latest transformers on PyPI is at image-build
time" (signet's own H3-image comment at `modal/app.py:234` notes this exact resolve-to-
latest behavior already happened once, non-deterministically, for the H3 leg). **Net
effect of the bump: the LTX-2 `gpu_image`'s `transformers` resolution becomes *more*
deterministic (bounded to 5.8-5.14.x) than it is today, not less** — this is a
supply-chain improvement, not a new risk, though it is worth pinning explicitly (mirroring
the H3/Qwen images) rather than left to float even within the new bound.

**Confidence: HIGH** on "no mandatory torch bump" (directly read from both pyproject
files' literal dependency arrays and extras tables). **MEDIUM** on "transformers 5.8-5.14
is drop-in safe for signet's existing gpu_image" — this is inferred from the version
window, not verified by an actual `uv pip install` resolution or a Gemma-3 load test,
since no upstream code was executed for this report.

---

## (C) Split-layout + Gemma-4 + arch-constant findings

### Split-layout config surface

Two **separate, non-identical** config surfaces exist for split checkpoints, and signet
would need to pick the right one depending which layer it touches:

1. **`ltx_trainer.model_loader.load_model()`** (the function signet's `models/loader.py`
   calls) — gained `video_vae_path: str | Path | None = None` and
   `audio_vae_path: str | Path | None = None` kwargs (both optional, default to treating
   `checkpoint_path` as the monolith). Internally resolved via two new module-level
   functions: `resolve_video_vae_path(model_path, video_vae_path=None)` and
   `resolve_audio_vae_path(model_path, audio_vae_path=None)`, which raise
   `ValueError` if the caller omits the override *and* `model_path` is detected to be a
   split transformer-only file (`is_split_transformer()` — checks whether the
   checkpoint's embedded metadata config declares a `"transformer"` section but none of
   `{"vae","audio_vae","vocoder"}`). `text_encoder_path` continues to double as either a
   Gemma HF directory or (new) a packed text-encoder `.safetensors` file.
2. **`ltx_pipelines.utils.model_paths.ModelPaths`** (consumed by `ICLoraPipeline` and
   `TI2VidTwoStagesPipeline`, i.e. exactly the two pipelines signet's
   `inference/ic_lora_pipeline.py` and `inference/upscale.py` wrap) — a frozen dataclass
   with `mode: Literal["monolith","split"]`, `transformer_path`, `text_encoder_path`,
   `video_vae_path`, `audio_vae_path`, `duration_head_path` (new: for the auto-duration
   feature), plus `embeddings_weight_paths: tuple[str,...]` (1 path for a monolith, 2 —
   transformer + text-encoder file — for a split pack with a packed TE). Built via
   `ModelPaths.from_monolith(checkpoint_path, gemma_root, video_vae_path=None)` for
   backward compat, or `ModelPaths.from_split(transformer_path=, text_encoder_path=,
   video_vae_path=, audio_vae_path=, duration_head_path=)` for a real split pack.

These two surfaces are **not interchangeable** — `load_model()` still takes flat kwargs,
while the pipelines now require a `ModelPaths` object. Any integration touching both the
trainer-side loader and the two `ltx_pipelines` wrappers has to build both shapes.

### Architecture auto-detection from checkpoint metadata

Confirmed mechanism, not just a CHANGELOG claim: `SpatioTemporalScaleFactors.from_model_config(model_config: dict)` reads `model_config["vae"]["encoder_blocks"]`/`["decoder_blocks"]` (a list of `(block_name, params)` tuples) and derives `(time, height, width)` scale factors by counting `compress_space*`/`compress_time*`/`compress_all*` block-name prefixes — falling back to the old default `(8,32,32)` only when the checkpoint's metadata carries no VAE block list at all (e.g. an audio-only checkpoint). Similarly, `LTXModelConfigurator.from_metadata(metadata: dict, ...)` (renamed from `.from_config(config: dict, ...)`, CHANGELOG-confirmed Removed/replaced) now receives the **entire safetensors metadata dict** and pulls `metadata["config"]` itself, rather than being handed the inner config directly — consistent with "Model configurators now receive complete checkpoint metadata through `from_metadata`, enabling architecture and version-dependent construction."

### Gemma-3 vs Gemma-4 and the hidden_size assertion

Signet's `models/loader.py` does not itself assert a hidden_size, but its
`EXPECTED_HIDDEN_DIM=4096` comment states the assumption explicitly: "Gemma 3840 →
connector → 4096" (i.e. Gemma-3-12B's hidden_size of 3840 feeds a learned connector that
projects to the transformer's 4096 cross-attention dim). Upstream code confirms there are
now **three** distinguishable Gemma variants by `model_type` string read from the
Gemma root's own HF config (`base_encoder.py`, `encoder_configurator.py`): `"gemma3"`,
`"gemma4_unified"` (the LTX-2.5 encoder family — the comment at
`encoder_configurator.py:267` names `gemma-4-12B` as an example `gemma4_unified`
checkpoint), and `"gemma4"` (dense instruct variant, e.g. `gemma-4-E2B-it`, used only for
prompt enhancement). Every place that would reveal a numeric hidden_size
(`encoder_configurator.py:181`: `embedding_dim = gemma_text_config.hidden_size`;
`:394`: `embed_scale = torch.tensor(config.hidden_size**0.5, ...)`) reads it **from the
checkpoint's own HF `config.json` / embedded `gemma_config` metadata at load time** — there
is no hardcoded Gemma-4 hidden_size constant anywhere in the `Lightricks/LTX-2` monorepo.

**Gemma-4-12B hidden_size: UNKNOWN from this repository.** It is a property of the
actual released Gemma-4-12B checkpoint (weights live on HF Hub / Lightricks' release,
not in this code repo), and cannot be determined without downloading that checkpoint and
reading its config. **Do not guess a number for it.** If signet's arch gate is extended
to assert a Gemma-4 hidden_size, that assertion must be written *after* inspecting the
real checkpoint's `config.json`, exactly the way signet's own `expected_arch()` doctrine
already insists on for the transformer side. — Confidence: HIGH that it's unknowable from
this repo; the "three model_type variants, config-driven, no hardcoded constant" mechanism
itself is HIGH confidence (directly read from source).

### Transformer block count / hidden dim / heads for LTX-2.5

Same conclusion as the VAE factors and hidden_size: **`num_layers`, attention head
count/dim, and channel counts are read from `metadata["config"]` at load time, not
hardcoded**, for both the pin and v1.2.0 — this was already true before 2.5 (the pin's
`LTXModelConfigurator.from_config` also read `config.get("num_layers", ...)` etc.). The
only upstream default value visible in the diff is on the *new* `LTXAudioOnlyModelConfigurator.from_metadata`, which defaults `num_layers=config.get("num_layers", 48)` — a fallback for the audio-only variant, not proof of the real 2.5 T2V/A2V transformer's actual block count. No `MODELS-LTX-2.5.md` file exists in the repo yet (only `MODELS-LTX-2.3.md` is present) to publish real numbers.

**Per-constant verdict against signet's `models/loader.py::EXPECTED_*`** (all UNKNOWN
= "the mechanism that would let you read the true value from a checkpoint is confirmed
present and unchanged; the actual number for an LTX-2.5 checkpoint requires downloading
one and reading its embedded config — it is not published anywhere in this monorepo"):

| Constant | Signet's LTX-2.3 value | 2.5 verdict | Evidence |
|---|---|---|---|
| `EXPECTED_NUM_BLOCKS` | 48 | UNKNOWN | config-driven (`num_layers`), no repo-level published value for 2.5 |
| `EXPECTED_HIDDEN_DIM` / `EXPECTED_VIDEO_INNER_DIM` | 4096 | UNKNOWN | config-driven, connector dim depends on the Gemma root paired at build time |
| `EXPECTED_NUM_HEADS` | 32 | UNKNOWN | config-driven |
| `EXPECTED_HEAD_DIM` | 128 | UNKNOWN | config-driven |
| `EXPECTED_VAE_TEMPORAL_COMPRESSION` | 8 | **CHANGED-MECHANISM, value UNKNOWN** | now derived via `SpatioTemporalScaleFactors.from_model_config()`; falls back to 8 only absent metadata |
| `EXPECTED_VAE_SPATIAL_COMPRESSION` | 32 | **CHANGED-MECHANISM, value UNKNOWN** | same mechanism as above |
| `EXPECTED_T2V_IN_CHANNELS` | 128 | UNKNOWN (config-driven both before and after; no diff visible on this specific field) | not touched in the diffed configurator code |
| (new, not in signet's list) `ff_bias`/`audio_ff_bias` | implicitly `True` (2.3 has FFN biases) | **CONFIRMED CHANGED-TO-False** for LTX-2.5(gemma4) checkpoints per inline upstream comment | `model_configurator.py` diff, verbatim: "LTX 2.5 (gemma4) sets `ff_bias=false`" |

Confidence: HIGH on the mechanism and on the `ff_bias=false` fact (both directly quoted
from source); the actual 2.5 block/dim/head numbers are explicitly UNKNOWABLE from this
repository — HIGH confidence in that negative claim, precisely because the read was
targeted and exhaustive (grepped `hidden_size`, `3840`, `MODELS-LTX-2.5`, checked every
configurator class).

---

## (D) Preprocess / cache implications

- `packages/ltx-trainer/scripts/process_dataset.py` / `process_videos.py` /
  `process_captions.py` all changed substantially (+903/-330 lines combined) — the
  dataset-column convention table itself (`video→latents/`, `audio→audio_latents/`,
  `reference_video→reference_latents/`, `caption→conditions/`, etc.) is **unchanged**,
  and the CLI still exposes `--model-path`/`--text-encoder-path` flags for the monolith
  case. Signet's own `prep/*.py` scripts do not import any of these upstream scripts at
  runtime (confirmed: none of `prep_*.py`/`scripts/prep_*.py` appear in the consumed-symbol
  grep) — they are hand-ported reimplementations, so this diff is "port drift" context,
  not a live import break.
- The text-embedding cache format signet's `data/precomputed.py` reads
  (`conditions/*.pt`) has, at v1.2.0, the same on-disk key set as at the pin:
  `{"video_prompt_embeds": Tensor, "prompt_attention_mask": Tensor, "audio_prompt_embeds": Tensor (optional)}`
  (`process_captions.py::compute_captions_embeddings`, confirmed at v1.2.0 source). The
  *shape* of the schema is stable; the *content* is not portable across Gemma families.
- **Confirmed Gemma-version-specificity of cached conditions.** `load_embeddings_processor` (and, transitively, `load_text_encoder`) is now explicitly parameterized by the specific Gemma root via `GemmaTextEncoderConfigurator.with_gemma_model_path(...)` / `EmbeddingsProcessorConfigurator.with_gemma_model_path(...)` — the feature-extractor/connector that turns raw Gemma hidden states into `video_prompt_embeds` is *built from, and dimensioned by, that specific checkpoint's Gemma root* (`hidden_size`, `num_hidden_layers` read directly off it). A `conditions/*.pt` cache produced against a Gemma-3 root cannot be fed through a Gemma-4-paired checkpoint's connector: at minimum the raw hidden-state width differs (Gemma-3-12B hidden_size 3840 vs. Gemma-4-12B's — unknown but almost certainly different — hidden_size), which fails at the connector's input projection, not silently. **Any bump to an LTX-2.5/Gemma-4 checkpoint requires a full re-run of the text-condition precompute step; existing `.precomputed/conditions/` caches built under LTX-2.3/Gemma-3 are not reusable.** Video/audio VAE latent caches (`latents/`, `audio_latents/`) are separately at risk if the 2.5 checkpoint's VAE compression factors differ from the (8,32,32) default signet's `conditioning/strategy.py` hardcodes (see §C) — those caches would need re-encoding too if the spatial/temporal compression changed, independent of the Gemma question.

Confidence: HIGH on the cache-key-set stability and on the Gemma-specificity argument
(directly evidenced by the per-root-parameterized configurator); MEDIUM on "VAE latent
caches also need re-encoding" since it depends on a fact (2.5's actual VAE compression
factors) this report could not determine (§C).

---

## (E) Recommended integration approach

**Do not bump `LTX2_COMMIT_SHA` on the existing `gpu_image` in place.** The two hard
breaks in §A (`load_embeddings_processor`'s new required `gemma_model_path` arg;
`ICLoraPipeline`'s `distilled_checkpoint_path`+`gemma_root` → `model_paths` collapse)
would need to be patched merely to keep the *current* LTX-2.3 code path alive after the
bump — before any LTX-2.5-specific capability is even attempted. And the removal of
`ltx_trainer.validation_sampler.{GenerationConfig,ValidationSampler}` in favor of the
self-contained, file-writing `ValidationRunner` invalidates the design of
`inference/sampler.py` and everything downstream of it (multi-condition, reference-video,
inpaint mask-condition, a2v frozen-audio rendering) — these were all hand-ported against
`ValidationSampler`'s *private* internals, which do not survive under known names.
Patching call sites is not sufficient here; the validation/sampling layer needs a design
decision, not a diff.

**Recommendation: build a second, parallel `ltx25_gpu_image` (new `LTX25_COMMIT_SHA`
pin) alongside the existing `gpu_image`, exactly mirroring the tri-family pattern
signet already uses for H3/Qwen (separate image, separate pinned SHA, separate
`transformers==` pin) — do not touch the LTX-2.3 `gpu_image`/`LTX2_COMMIT_SHA` pin at
all.** Reasons:
1. Torch/CUDA do not force this (§B) — both images can share the same `torch~=2.7`/cu129
   base, so there is no infrastructure cost to running them side by side, only a second
   `git clone` + editable-install layer in the image build (identical pattern to how
   `h3_gpu_image` and the Qwen image already coexist with `gpu_image` today).
2. LTX-2.3 currently WORKS (validated on the pin, per signet's own docs) and every
   render/training path currently in production runs through the exact `ValidationSampler`/
   `GenerationConfig`/`ICLoraPipeline(distilled_checkpoint_path=,gemma_root=)` call shapes
   this diff shows are gone at v1.2.0. Bumping the shared pin risks the working LTX-2.3
   surface for a 2.5 capability that has not been built yet.
3. The 2.5 split-layout + `ValidationRunner` design is different enough (self-managed
   load/unload lifecycle, file-writing `run()`, `ModelPaths` bundling) that it is a
   genuinely new integration, not a drop-in replacement for signet's carefully-built
   two-phase-VRAM component-injection discipline (`models/loader.py::load_ltxv_components`
   + `inference/sampler.py::build_validation_sampler`). Writing it as new code against a
   new pin, rather than mutating the existing loader, keeps the two designs from fighting
   each other inside one module.

**What has to be decided by the operator before a PR, not inferred by an agent:**
- Whether to adopt `ValidationRunner`'s self-managed load/unload lifecycle *as-is*
  (simpler, but gives up signet's explicit two-phase-VRAM control that exists specifically
  because the A100-80GB cannot hold Gemma + the 22B transformer simultaneously — runs 3-5
  proved this the hard way) or to keep hand-porting the internals again against the new
  private surface (more control, more maintenance, and this report cannot promise the new
  private methods are stable/documented the way `ValidationSampler`'s were assumed to be).
- Whether signet exposes a `model.checkpoint_layout: monolith|split` (or per-component
  path) config surface at all for 2.5, given the two non-interchangeable path bundles in
  §C (`load_model()`'s flat kwargs vs. `ltx_pipelines`' `ModelPaths`) — this is new
  config-schema surface area signet's Pydantic `SignetConfig` does not have today.
- Whether to fix the two **pre-existing, pin-independent** bugs found incidentally
  (`ltx_trainer.model_builder` nonexistent module + wrong `SingleGPUModelBuilder` kwargs
  in the `use_builder` fallback branch; `LTXV_MODEL_COMFY_RENAMING_MAP` imported from the
  wrong module) in the same PR or a separate one — they are unrelated to 2.5 but sit in
  code this PR will already be touching (`local/runner.py`, `modal/fns.py`).
- Whether to actually download a real LTX-2.5 checkpoint and read its embedded metadata
  before writing any `EXPECTED_*_25` constants — §C establishes that the block count,
  hidden dim, head count, VAE compression, and Gemma-4 hidden_size are all genuinely
  unknowable from the `Lightricks/LTX-2` source alone.

Confidence: HIGH on "do not bump the shared pin in place" (directly supported by the
confirmed breaking changes); MEDIUM on "parallel image is the right shape" (a reasoned
recommendation following signet's own established tri-family pattern, not something
verified against a working build).
