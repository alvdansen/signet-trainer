# LTX-2.5 Stage 1 — Implementation Design (train-only, Modal)

**Status**: REVISED post-refuter, then implemented on `feat/ltx25-stage1`. Two independent refuter
passes read this design against the live repo and issued verdicts of MUST-CHANGE and MUST-FIX
respectively. Every finding from both is incorporated below (marked inline at its landing site);
none was dropped. The binding changes, at a glance:

  * §5/§6 — `write_provenance()` moved to fire BEFORE `dataset_vol.commit()`, not after
    (refuter A, HIGH — the commit-or-vanish pitfall applied to the provenance file itself).
  * §6 — `assert_cache_provenance` now also takes `configured_gemma_root` and compares it, not
    just the generation (refuter B, HIGH — a same-generation Gemma-root swap was invisible to a
    generation-only check).
  * §5/§6 — a NEW `assert_output_dir_ready_for_ltx25_encode` refuses a non-empty `output_dir`
    lacking MATCHING provenance unless `overwrite=True` forces a full re-encode (refuter B, HIGH —
    the "mixed cache" case: an incremental encode into a stale 2.3/Gemma-3 dir would otherwise let
    `write_provenance` certify a mixed cache as clean).
  * §7 — ALL SIX entrypoint `--mode` values are now explicitly dispositioned for `ltx_generation
    == '2.5'` (train/preprocess allowed; **sample AND fuse refused**, not just sample; restore/
    backup explicitly cleared as generation-agnostic), and the refusal lives in
    `config/mode_gate.py::validate_mode_config` — the ONE shared CPU-pure home already used by the
    dry-run CLI, the entrypoint, and the container bodies — not only in an entrypoint-local gap
    function (refuter A MEDIUM + refuter B MEDIUM).
  * §2/§9 — `local/runner.py::refusals()` gets ONE new line refusing `ltx_generation != '2.3'`,
    named loudly rather than left to crash cryptically inside a 2.3-only loader (refuter B,
    MEDIUM). This is NOT the local runner "gaining LTX-2.5 awareness" in the sense §9 still rules
    out — it is the one-line loud pointer to Stage 3 the codebase's own doctrine already asks for.
  * §1 — the `ltx25` reverse guard uses the PRISTINE-INSTANCE comparison (`Ltx25Config()`), not a
    hand-read `.default` off each field — the same technique the qwen_edit guard already uses, so
    a later `default_factory` field on `Ltx25Config` cannot silently escape the guard (refuter B,
    LOW).
  * §1/new — `validation.two_stage_upscale` is refused outright for `ltx_generation == '2.5'`
    (mirrors the `in_loop_sampling` guard exactly), and the pre-existing D-7-BASEVAR
    distilled/two_stage_upscale PAIRING check is scoped to `ltx_generation == '2.3'` — a
    `'2.5'`-and-`'distilled'`-named config now hits only the new 2.5 guard, never the old 2.3
    pairing rule (refuter B, LOW).
  * §4.1 — `load_ltx25_components`'s signature gains `with_audio_vae_encoder: bool = False`,
    raising `NotImplementedError`, matching what §4.4 and the test plan already required of it
    (refuter A, LOW — an internal contradiction in the first draft).
  * §2/§9 — weights staging is named explicitly as a Stage-1 non-goal (manual staging until D3),
    closing a touchpoint the qwen_edit-PR rubric has (`download_qwen_edit_weights`) that this
    design's first draft silently had no row for at all (refuter A, MEDIUM).
  * §7/§11 — the 6-check architecture validation gate's coverage loss for 2.5 (no adapter-
    roundtrip proof, no forward-pass smoke) is now stated as an ACKNOWLEDGED Stage-1 gap, not an
    implicit omission an implementer might "fix" by copying `train()`'s gate call against 2.3
    constants (refuter B, LOW).

**Scope**: issue #53 Stage 1 only —
parallel `ltx25_gpu_image`, split-layout loader, config surface, metadata-driven arch gate,
Gemma-4 preprocess arm with cache provenance, dryrun coverage. **Explicitly excludes**
sampling/rendering (Stage 2 / D1), the local runner (Stage 3 / #25/#30), and any hardcoded
`EXPECTED_*_25` architecture constant (D3 — no real checkpoint has been read yet).

Grounded in three sources, cited inline: `LTX25_UPSTREAM_DIFF.md` (the gate report, §A–§E),
issue #53 (the operator-decision brief, D1–D4), and `Lightricks/LTX-2` at the `v1.2.0` tag
(commit `d151147788a9284cca791edc6ce898007e727fe6`), read live from the scratchpad clone for
every claim marked **[v1.2.0 verified]** below.

---

## 0. The one architectural call this design makes, and why

**`model.family` stays `"ltx"`. LTX-2.5 is a new field, `model.ltx_generation`, not a fifth
family value, and not auto-detected from the checkpoint.**

Justification:

1. **What family already means in this codebase is "which native backend loads and steps
   this checkpoint"** (`config/schema.py:422`: `Literal["ltx", "h3", "qwen_edit", "wan"]` —
   LTX transformer+VAE+Gemma vs. the MiniMax-H3 diffusers pipeline vs. Qwen-Image-Edit's
   dual-stream MMDiT vs. the musubi-tuner Wan 2.1 runner). LTX-2.5 does not change that
   answer — it is still the LTX transformer/VAE/Gemma backend, still trained through
   `train/loop.py`'s LTX default step (`build_step_deps()` + `training_step`,
   `train/family_hooks.py:42`), still governed by the same frame law and conditioning
   strategies (`conditioning/strategy.py`). Making it family `"ltx25"` would force every
   LTX-only field (frame-count law, resolution buckets, `ic_lora`/inpaint/a2v conditioning
   knobs) to either be duplicated onto a new family block or bypass the lean field-split
   guard entirely — both worse than adding one discriminator field to the family that
   already owns this surface.
2. **The gate report's own confirmed break is generation-shaped, not family-shaped**: every
   hard break in §A (`load_embeddings_processor`'s new arg, `ICLoraPipeline`'s `ModelPaths`
   collapse, `ValidationSampler` removal) is "the same LTX backend, a newer checkpoint
   generation," never "a different backend."
3. **Auto-detection from the checkpoint is rejected on a structural, not a taste, ground**:
   Modal's image is selected at **CPU config-load / dispatch time**, before any container
   with GPU access — let alone the checkpoint's bytes — is even started
   (`modal/entrypoint.py` builds the dispatch, `modal/app.py` builds the image graph at
   **module-import** time — see `app.py`'s own banner at lines 47–49 making exactly this
   argument for the App/Volume names). Reading the checkpoint's embedded metadata to decide
   *which image to build* would mean mounting the weights Volume, or downloading the
   checkpoint, before deciding which image mounts it — an inversion this codebase's dispatch
   model does not support anywhere today (h3/qwen_edit/wan all select their image from a
   config field read at zero-GPU-cost time, never from a peek at the weights). Auto-detect is
   also a bad fit for the operator's actual constraint (D3): there is no checkpoint to peek at
   until the gated download happens, so "detect from the checkpoint" cannot even be exercised
   before Stage 1 code exists.
4. **This is the same design already picked for `model.family` and `base_variant_of()`**
   (`models/loader.py:92-100`): an **explicit field**, never a filename/shape sniff, precisely
   because sniffing is a guess and a guess in a discriminator that selects which image
   builds, which loader function runs, and which arch gate fires is the single highest-
   leverage place for a silent-failure this codebase's own doctrine refuses (CLAUDE.md
   `reference_training_doctrine.md`: "read samples not loss," "no silent failures" applies
   equally to "which model did I even load").

```python
# config/schema.py — ModelConfig, new field (mirrors the `family` field's own justification,
# lines 411-430)
ltx_generation: Literal["2.3", "2.5"] = Field(
    default="2.3",
    description="LTX checkpoint GENERATION discriminator, meaningful only when "
    "model.family == 'ltx'. Explicit by design — NOT sniffed from model_id or the "
    "checkpoint's embedded metadata (mirrors the family field's own reasoning): the Modal "
    "image graph is built at module-import time (modal/app.py), before any GPU container "
    "or checkpoint byte exists, so nothing can be auto-detected from the weights before the "
    "image is already chosen. Default '2.3' keeps every existing config byte-identical. Set "
    "'2.5' to route through ltx25_preprocess/ltx25_train (ltx25_gpu_image, LTX25_COMMIT_SHA) "
    "instead of preprocess/train. A non-default value under model.family != 'ltx' is refused "
    "by SignetConfig._cross_field_checks (lean field-split).",
)
```

---

## 1. Config schema sketch

Two additions to `config/schema.py`, following the exact shape of the `H3Config`/
`QwenEditConfig` lean field-split blocks (`schema.py:479-490`, `:974-980`) — an all-default
block, reverse-guarded, so every pre-Stage-1 config loads byte-identically.

```python
class Ltx25Config(_Base):
    """LTX-2.5 split-checkpoint tunables. Every field defaulted, so every ltx_generation=='2.3'
    config (i.e. every config that exists today) loads byte-identically.

    Only meaningful when model.ltx_generation == '2.5' — NOT keyed off model.family (family
    stays 'ltx' for both generations; see §0). SignetConfig._cross_field_checks carries the
    REVERSE guard: a non-default value here while ltx_generation != '2.5' is refused at config
    load, the same lean-field-split discipline as the h3/qwen_edit blocks, but keyed off a
    generation field instead of family — record that distinction at the guard site or a future
    reader will assume every reverse-guard checks model.family and add a new one that doesn't.
    """

    checkpoint_layout: Literal["monolith", "split"] = Field(
        default="monolith",
        description="Whether model.model_id names one monolithic .safetensors (transformer + "
        "VAE + audio VAE packed together, the LTX-2.3 shape) or a split pack (transformer-only "
        "file; VAE/audio-VAE/duration-head live in separate files named below). "
        "[v1.2.0 verified] ltx_trainer.model_loader.load_model() gained optional "
        "video_vae_path=/audio_vae_path= kwargs precisely for this split case; omitting them "
        "on a split transformer-only checkpoint raises inside resolve_video_vae_path/"
        "resolve_audio_vae_path (is_split_transformer() reads the checkpoint's own embedded "
        "metadata to detect this — see LTX25_UPSTREAM_DIFF.md §C).",
    )
    video_vae_path: str | None = Field(
        default=None,
        description="Split layout ONLY: video-VAE .safetensors filename under WEIGHTS_DIR. "
        "REQUIRED (fail-fast at config load, not at GPU runtime) when checkpoint_layout == "
        "'split' — the upstream resolver raises ValueError on a split transformer-only "
        "model_id with no override, and this design moves that failure off a metered "
        "container. Refused (must stay None) when checkpoint_layout == 'monolith'.",
    )
    audio_vae_path: str | None = Field(
        default=None,
        description="Split layout ONLY, and a2v-on-2.5 is OUT OF SCOPE for Stage 1 (see §9) — "
        "this field exists so the config surface is forward-declared for Stage 2/3, not "
        "because Stage 1 reads it. If set, Stage 1's loader still refuses to pass it into "
        "load_audio_vae_encoder without re-resolving it (see §4.3 — the standalone encoder "
        "loader was NOT extended with split-path resolution upstream and would silently "
        "build a meta-device encoder otherwise).",
    )
```

`SignetConfig` additions (mirrors `schema.py:2542-2548`, `:2748-2760` verbatim in shape):

```python
# SignetConfig field:
ltx25: Ltx25Config = Field(default_factory=Ltx25Config)

# SignetConfig._cross_field_checks, new block (reverse guard, keyed on ltx_generation not family).
# [REVISED post-refuter B, LOW] uses the PRISTINE-INSTANCE comparison (Ltx25Config()), the SAME
# technique the qwen_edit/H3 reverse guards already use — NOT a hand-read `.default` off each
# field (the first draft's shape), which would silently stop covering a later default_factory
# field on Ltx25Config. Instantiating a pristine Ltx25Config() does not recurse (it is not
# SignetConfig itself), so this is safe here the way it is for H3Config()/QwenEditConfig().
if self.model.ltx_generation != "2.5":
    pristine_ltx25 = Ltx25Config()
    nondefault_ltx25 = [
        f for f in Ltx25Config.model_fields
        if getattr(self.ltx25, f) != getattr(pristine_ltx25, f)
    ]
    if nondefault_ltx25:
        raise ValueError(
            f"ltx25 field(s) {nondefault_ltx25} set while model.ltx_generation is "
            f"{self.model.ltx_generation!r}: the ltx25 block is only valid when "
            f"model.ltx_generation == '2.5' (lean field-split — no silently-ignored config "
            f"block). Remove them or set model.ltx_generation: '2.5'."
        )

# reverse guard #2 — a non-default ltx_generation makes no sense outside family 'ltx':
if self.model.family != "ltx" and self.model.ltx_generation != "2.3":
    raise ValueError(
        f"model.ltx_generation is {self.model.ltx_generation!r} while model.family is "
        f"{self.model.family!r}: ltx_generation is only meaningful under family 'ltx'. "
        f"Remove it or set model.family: ltx."
    )

# fail-fast: split layout without an explicit video_vae_path is a certain upstream ValueError,
# moved off the metered container (§C's resolve_video_vae_path/is_split_transformer finding):
if self.model.ltx_generation == "2.5" and self.ltx25.checkpoint_layout == "split" \
        and self.ltx25.video_vae_path is None:
    raise ValueError(
        "model.ltx_generation is '2.5' and ltx25.checkpoint_layout is 'split' but "
        "ltx25.video_vae_path is unset: a split transformer-only checkpoint has no VAE "
        "inside it, and ltx_trainer.model_loader.resolve_video_vae_path raises ValueError "
        "with no override (LTX25_UPSTREAM_DIFF.md §C). Set ltx25.video_vae_path or "
        "checkpoint_layout: monolith."
    )

# in-loop sampling refusal (see §7) — a config-load-time refusal, not just an entrypoint gap,
# because validation.in_loop_sampling is read purely from config with no separate CLI mode:
if self.model.ltx_generation == "2.5" and self.validation.in_loop_sampling:
    raise ValueError(
        "model.ltx_generation is '2.5' and validation.in_loop_sampling is True: in-loop "
        "sampling goes through inference/sampler.py's run_sampler, built against "
        "ltx_trainer.validation_sampler.{GenerationConfig,ValidationSampler} — REMOVED "
        "upstream at v1.2.0 (LTX25_UPSTREAM_DIFF.md §A). This is Stage 2 scope (issue #53 "
        "D1). Set validation.in_loop_sampling: false for a Stage-1 LTX-2.5 train run."
    )

# [ADDED post-refuter B, LOW] validation.two_stage_upscale refusal -- mirrors the in_loop_sampling
# guard exactly (same removed-and-rebuilt ValidationSampler/ICLoraPipeline call shapes, Stage-2
# scope). Without this a 2.5 config could set two_stage_upscale: true and nothing would refuse it
# at config load -- it would only ever be caught (if at all) by the sample-mode mode_gate refusal
# in §7, and only at the wrong layer (a validation KNOB, not a dispatch MODE).
if self.model.ltx_generation == "2.5" and self.validation.two_stage_upscale:
    raise ValueError(
        "model.ltx_generation is '2.5' and validation.two_stage_upscale is True: the two-stage "
        "distilled + spatial-upscaler render goes through inference/upscale.py's "
        "TI2VidTwoStagesPipeline wrapper (Stage-2/inference scope, issue #53 D1) -- not touched "
        "by Stage 1. Set validation.two_stage_upscale: false for a Stage-1 LTX-2.5 train run."
    )
```

**[ADDED post-refuter B, LOW]** the pre-existing D-7-BASEVAR pairing check
(`"distilled" in self.model.model_id and not self.validation.two_stage_upscale -> raise`) is now
SCOPED to `self.model.ltx_generation == "2.3"`. That pairing is a fact about the LTX-2.3
dev/distilled inference paths specifically; a 2.5 "distilled"-named checkpoint, if one exists,
would pair with Stage-2 machinery this design does not touch, and forcing `two_stage_upscale:
true` on it to satisfy the OLD check would only immediately trip the NEW guard directly above.
Scoping the old check to 2.3 -- rather than adding a second, silent exemption inside it -- keeps
the 2.3 law exact and lets the new 2.5 guard be the only one that speaks for 2.5.

No new field is needed for `load_embeddings_processor`'s new `gemma_model_path` argument.
**[v1.2.0 verified]** `packages/ltx-trainer/scripts/process_captions.py:309-311` (the module
`ltx_trainer.model_loader.load_model`'s own sibling, invoked transitively by
`preprocess_dataset`) already resolves this internally as
`load_embeddings_processor(embedding_weight_paths(model_path, text_encoder_path),
gemma_model_path=text_encoder_path)` — i.e. it reuses the **already-existing**
`model.text_encoder_id` value for both roles. Signet's own direct call site
(`models/loader.py:264-268`) is only reached on the sampling path (`with_video_vae_decoder=True`),
which Stage 1 refuses outright (§7) — so this breaking signature change never has to be patched
in Stage 1's own code, only in upstream's canonical script, which v1.2.0 has already done.
`model.text_encoder_id` is simply pointed at the Gemma-4 root for a 2.5 config; no schema change.

---

## 2. File-by-file change list

Rubric: the **qwen_edit family-add** (PR #14, merge commit `1622806`) is the highest-fidelity
precedent for "what actually gets touched when a new checkpoint surface lands," so its real
diffstat (30+ files) is regrouped below into the 14 touchpoint categories it exercised, each
now answered for Stage 1's narrower (train-only) scope.

| # | Touchpoint (qwen_edit precedent) | Stage-1 file | Change | Byte-identity for 2.3? |
|---|---|---|---|---|
| 1 | `config/schema.py` (+661 lines in #14) | `config/schema.py` | `ModelConfig.ltx_generation` field; new `Ltx25Config` block; `SignetConfig.ltx25` field | Yes — new field defaults preserve every existing config |
| 2 | `config/validators.py` (+340 lines in #14) | `config/schema.py::SignetConfig._cross_field_checks` | 6 new guard blocks (§1: split-layout fail-fast, in_loop_sampling refusal, two_stage_upscale refusal + the 2.3-scoped distilled-pairing exemption, and the two reverse guards) — **not** a new file; qwen_edit's validators.py additions were mostly geometry math this family doesn't need (no new frame law) | Yes — guards only fire on non-default `ltx_generation`/`ltx25` |
| 3 | `dryrun/shapes.py` (+586 lines in #14) | `dryrun/shapes.py` | New `_assert_ltx25_dryrun_contract` called inside `build_dryrun_inputs` (§8) | Yes — additive, `"2.3"` branch unchanged |
| 4 | `conditioning/<family>.py` + geometry/packing (qwen_edit: 3 new files, 1676 lines) | **none** | LTX-2.5 reuses `conditioning/strategy.py` as-is — same frame law, same `Modality` shape, same VAE-scale constants **with a new runtime cross-check** (§4.2), not a new conditioning module | Yes — no edit to this file at all |
| 5 | `models/<family>_loader.py` (qwen_edit: new 966-line file) | `models/ltx25_loader.py` **[REVISED: a NEW SEPARATE file, not new functions in `models/loader.py`]** | `load_ltx25_components()`, `read_ltx25_checkpoint_metadata()`, `compute_vae_scale_factors_from_metadata()`, `assert_ltx25_vae_compression()` — imports `EXPECTED_VAE_*_COMPRESSION` FROM `models/loader.py` rather than duplicating them | Yes — `models/loader.py` untouched, mirrors the h3_loader.py/qwen_edit_loader.py per-family-file precedent |
| 6 | `modal/app.py` (qwen_edit: +121 lines) | `modal/app.py` | New `ltx25_gpu_image` + `LTX25_COMMIT_SHA` + `LTX25_TRANSFORMERS_VERSION` (§3) | Yes — `gpu_image`/`LTX2_COMMIT_SHA` untouched, mirrors the tri-family precedent exactly |
| 7 | `modal/fns.py` (qwen_edit: +1614 lines) | `modal/fns.py` | New `ltx25_preprocess()` + `ltx25_train()` `@app.function`s (§5, §7); **no** `ltx25_sample` | Yes — `preprocess`/`train` untouched |
| 8 | `modal/entrypoint.py` (qwen_edit: +601 lines) | `modal/entrypoint.py` | New `elif mode == "preprocess" and ltx_generation == "2.5"` / `elif ... family == "ltx" and ltx_generation == "2.5"` (train) dispatch arms, mirroring the h3/qwen_edit family-routing precedent but keyed on generation; the mode REFUSAL itself lives in `config/mode_gate.py` (§7), not here | Yes — dispatch table gets two new arms, existing arms unchanged |
| 9 | `modal/cost.py` (qwen_edit: +103 lines) | `modal/cost.py` | Only if Stage 1 runs on H100/H200 (issue #53's "Modal-first (H100/H200 bf16)") rather than A100-80GB: a new hourly-rate constant, `DEFAULT_HOURLY_RATE_USD` stays A100-only, unverified for H100/H200 — **flagged open, not resolved here** | Yes — no edit unless a new GPU class needs its own rate |
| 10 | cache format / provenance (qwen_edit: `write_qwen_edit_precomputed`, its own module) | `data/cache_provenance.py` **[a NEW file, kept separate from `data/precomputed.py`]** | `write_provenance()` / `assert_output_dir_ready_for_ltx25_encode()` / `assert_cache_provenance()` (§6, THREE functions post-refuter) | Yes — 2.3 preprocess never writes/reads a provenance file; `PrecomputedDataset` itself is untouched (byte-identity #10, §11) |
| 10b | mode-conditional refusal home | `config/mode_gate.py::validate_mode_config` **[REVISED post-refuter A+B — not an entrypoint-local gap fn]** | One new block refusing `mode in ("sample", "fuse")` for `ltx_generation=='2.5'` (§7) — the SAME shared CPU-pure home the dry-run CLI/entrypoint/container bodies already call | Yes — fires only for `ltx_generation=='2.5'` |
| 10c | **[ADDED post-refuter B]** local-runner loud refusal | `local/runner.py::refusals()` | One new line refusing `ltx_generation != '2.3'`, naming Stage 3 (§7/§9) | Yes — fires only for `ltx_generation != '2.3'` |
| 11 | `prep/*_encode.py` (qwen_edit: 1861-line new file) | **none** — LTX preprocessing runs through upstream's *own* canonical `process_dataset.py`/`process_captions.py`, not a signet-native `prep/` port (unlike qwen_edit/h3, which hand-port their encode) | The provenance WRITE (touchpoint 10) is called from `modal/fns.py::ltx25_preprocess`, right after the canonical encode returns, BEFORE `dataset_vol.commit()` — no new file under `prep/` | Yes |
| 12 | `train/family_hooks.py` (qwen_edit: new branch) | **none** | `model.family` stays `"ltx"`, so `LOOP_HOOKS_BY_FAMILY["ltx"]` (`family_hooks.py:42`) is unchanged — LTX-2.5 uses the SAME `train_loop` default (`build_step_deps()` + `training_step`) as 2.3. This is a **deliberate non-touchpoint**, worth stating explicitly so a future reader does not add a redundant `"ltx25"` branch that can never be reached (`family` never becomes `"ltx25"`, §0) | Yes |
| 13 | `tests/` (qwen_edit: 10 new files, ~5700 lines) | `tests/test_ltx25_config.py`, `test_ltx25_loader.py`, `test_ltx25_dryrun.py`, `test_ltx25_cache_provenance.py`, `test_ltx25_local_runner.py`, `test_ltx25_entrypoint.py`, `test_modal_gpu_image.py` (extend), `test_modal_image_config_closure.py` (extend — the new `ltx25_gpu_image`'s config-loader closure needed a registered transitive supplier, the SAME class of defect `h3_gpu_image` hit in 2026-08-06) | See §10 test plan | N/A (test-only) |
| 14 | `configs/*.example.yaml` (qwen_edit: 4 new example configs) | `configs/ltx25_train.example.yaml` | One new example: `model.family: ltx`, `model.ltx_generation: "2.5"`, `model.model_id`/`text_encoder_id` pointed at placeholder 2.5/Gemma-4 filenames, `ltx25.checkpoint_layout` demoed both ways in comments | Yes — new file, no edit to existing examples |
| — | `lora/peft.py` (qwen_edit: +163 lines, new 14-leaf target regex) | **none** | LoRA targets `ff.net.0.proj` / `ff.net.2` (`config/schema.py` LTX default target tuple) are unaffected by `ff_bias=False` — LoRA injects low-rank adapters on the **weight** matrix; a base layer's bias tensor (present or absent) is orthogonal to which `nn.Linear`s get a LoRA wrapper. **Deliberate non-touchpoint**, confirmed rather than assumed — flag if a real 2.5 checkpoint's block introspection (§4.1) ever reveals a target-module name that does not exist under 2.3 (unconfirmed until §4.1 runs on a real checkpoint) | Yes |
| — | **[ADDED post-refuter A MEDIUM]** weights staging | **none — explicit non-goal** (§9) | No `download_ltx25_weights` fn. Manual staging until D3 resolves | N/A — no code |

---

## 3. Modal image: `ltx25_gpu_image`

Mirrors the tri-family pattern (`h3_gpu_image`, `qwen_gpu_image`) already established in
`modal/app.py`, **built fresh** (not chained off `gpu_image`, same "no build step after
`add_local_*`" rule every existing image already documents).

```python
# Pinned to the v1.2.0 tag ("Support for LTX 2.5"), 2026-08-11. A LITERAL SHA, never `main`
# (D-10-PIN discipline, same as LTX2_COMMIT_SHA above). Bump deliberately.
LTX25_COMMIT_SHA = "d151147788a9284cca791edc6ce898007e727fe6"

# [v1.2.0 verified, LTX25_UPSTREAM_DIFF.md §B] packages/ltx-core's transformers pin narrowed from
# an unbounded floor (>=4.52) to a bounded window (>=5.8.0,<5.15) — the upper bound exists because
# transformers>=5.15 breaks Gemma-4 config attribute access (inline upstream comment). Pin a
# LITERAL version inside that window rather than let uv float within it (mirrors the
# TRANSFORMERS_VERSION / QWEN_TRANSFORMERS_VERSION discipline two lines up in this same file).
# 5.14.1 REUSES the already-vetted h3_gpu_image pin (TRANSFORMERS_VERSION above) rather than
# inventing a third number — but this is NOT yet verified against a real Gemma-4 checkpoint load
# (D3 is still open: no 2.5 checkpoint has been read). Re-verify this exact version the day the
# gated Gemma-4 root is downloaded; do not treat "reuses an existing pin" as "verified for 2.5."
LTX25_TRANSFORMERS_VERSION = "5.14.1"

ltx25_gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")  # same torchaudio-demux reason as gpu_image (app.py:148-151)
    .pip_install("uv")
    # [LTX25_UPSTREAM_DIFF.md §B] no mandatory torch bump — ltx-core still declares torch~=2.7,
    # ltx-trainer torch>=2.6.0 at v1.2.0. The new torch==2.13/cu132 pin is gated behind the
    # opt-in `natten` extra; ltx-kernels is excluded from the default workspace. SAME literal
    # torch/torchvision line as gpu_image/h3_gpu_image/qwen_gpu_image — no family carries a
    # torch-version confound relative to the others.
    .run_commands(
        "uv pip install --system --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cu129 'torch~=2.7' 'torchvision~=0.22'"
    )
    .run_commands(
        "git clone https://github.com/Lightricks/LTX-2 /opt/LTX-25",
        f"cd /opt/LTX-25 && git checkout {LTX25_COMMIT_SHA}",
        # Pin transformers EXPLICITLY at the editable-install step (unlike gpu_image, which lets
        # uv resolve it transitively — a gap this bump is a good moment to also close per the gate
        # report's recommendation, §B). ltx-pipelines is DELIBERATELY NOT installed here: Stage 1
        # is train-only, and ICLoraPipeline/TI2VidTwoStagesPipeline are Stage-2/inference-only
        # concerns (§9).
        f"cd /opt/LTX-25 && uv pip install --system --index-strategy unsafe-best-match "
        f"--extra-index-url https://download.pytorch.org/whl/cu129 "
        f"'transformers=={LTX25_TRANSFORMERS_VERSION}' -e packages/ltx-core -e packages/ltx-trainer",
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # same fragmentation mitigation
    .add_local_python_source("signet_trainer")  # CODE ONLY, last (Modal build-order rule)
)
```

Do **not** touch `LTX2_COMMIT_SHA` / `gpu_image` (byte-identity guarantee #1, §11) — this is a
parallel image, not an in-place bump, per the gate report's Recommendation (E).

---

## 4. Loader: `models/ltx25_loader.py`

**[REVISED — implementation note]** this is a NEW, SEPARATE module, not new functions bolted onto
`models/loader.py`. `models/loader.py` already owns the LTX-2.3 loader/EXPECTED_* constants and
stays untouched (byte-identity guarantee #2, §11); the h3/qwen_edit precedent is `models/
h3_loader.py` / `models/qwen_edit_loader.py` — a sibling per-family/per-generation loader file, not
a branch inside the existing one. `models/ltx25_loader.py` imports `EXPECTED_VAE_TEMPORAL_
COMPRESSION`/`EXPECTED_VAE_SPATIAL_COMPRESSION` FROM `models/loader.py` (the confirmation target,
§4.2) rather than re-declaring them.

### 4.1 `load_ltx25_components()` — new function, not a branch inside `load_ltxv_components`

```python
def load_ltx25_components(
    checkpoint_path: str,
    text_encoder_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    video_vae_path: str | None = None,
    audio_vae_path: str | None = None,
    with_video_vae_decoder: bool = False,   # Stage 1 default: training never decodes (§9)
    with_text_encoder: bool = True,
    # [ADDED post-refuter A, LOW] was MISSING from the first draft's signature even though §4.4
    # and the §10 test plan both already required load_ltx25_components to raise on it — an
    # internal contradiction a mirroring implementer would have hit immediately.
    with_audio_vae_encoder: bool = False,
) -> "LtxModelComponents":
    """Load LTX-2.5 components. Train-only (Stage 1): both decoder-adjacent flags must stay False.

    with_video_vae_decoder: the decoder path also needs load_embeddings_processor(
    gemma_model_path=...) which routes through inference/sampler.py's ValidationSampler-shaped
    code, REMOVED upstream at v1.2.0 (LTX25_UPSTREAM_DIFF.md §A). Raise here rather than silently
    degrading if a caller ever flips it True before Stage 2 lands.

    with_audio_vae_encoder: a2v-on-2.5 is out of Stage-1 scope (§9) AND a confirmed silent-failure
    risk on a split checkpoint regardless of scope (§4.4) — raise here too, before the deferred
    import, rather than leaving the guard absent until someone is burned by it.
    """
    if with_video_vae_decoder:
        raise NotImplementedError(
            "load_ltx25_components(with_video_vae_decoder=True) is Stage-2 scope (issue #53 "
            "D1) — the decoder path needs load_embeddings_processor(gemma_model_path=...) "
            "which feeds inference/sampler.py's ValidationSampler-shaped code, and "
            "ltx_trainer.validation_sampler.{GenerationConfig,ValidationSampler} is REMOVED "
            "at v1.2.0, replaced by validation_runner.ValidationRunner's self-managed "
            "load/unload lifecycle (LTX25_UPSTREAM_DIFF.md §A). Do not attempt a partial port."
        )

    if with_audio_vae_encoder:
        raise NotImplementedError(
            "load_ltx25_components(with_audio_vae_encoder=True) is out of Stage-1 scope "
            "(a2v-on-2.5, issue #53 §9) AND a confirmed silent-failure risk on a split checkpoint "
            "(§4.4 — the standalone load_audio_vae_encoder was never extended with split-path "
            "resolution upstream). Refusing outright rather than risking that landmine."
        )

    from ltx_trainer.model_loader import load_model  # deferred: this fn runs ONLY inside
                                                        # ltx25_gpu_image's container.

    components = load_model(
        checkpoint_path=checkpoint_path,
        text_encoder_path=text_encoder_path,
        device=device,
        dtype=dtype,
        with_video_vae_encoder=True,
        with_video_vae_decoder=False,
        with_audio_vae_decoder=False,
        with_vocoder=False,
        with_text_encoder=with_text_encoder,
        # [v1.2.0 verified] the ONLY signature delta on load_model itself vs the 2.3 pin —
        # optional, default None (monolith). None on a split transformer-only checkpoint raises
        # inside resolve_video_vae_path (config-load already fail-fasts this, §1) — that's the
        # intended failure surface, not this call site.
        video_vae_path=video_vae_path,
        audio_vae_path=audio_vae_path,
    )
    return components
```

`load_ltxv_components` (2.3) is **not modified** — byte-identity guarantee #2.

### 4.2 VAE-compression self-consistency check (replaces a hardcoded `EXPECTED_VAE_*_25`)

`conditioning/strategy.py`'s `TIME_SCALE=8` / `HEIGHT_SCALE=32` / `WIDTH_SCALE=32` are read by
`config/validators.py`'s frame-law checks at **config-load time**, before any checkpoint is
even named as a Modal Volume path (Pitfall 1 — nothing is FS-checked locally). Those constants
are not touched (`config/validators.py` stays byte-identical — touchpoint #4/#5 in the table).
Instead, `ltx25_preprocess`/`ltx25_train` add one **runtime** assertion, once the checkpoint is
actually mounted:

```python
# models/loader.py, new function — reads the checkpoint's OWN embedded metadata the same way
# ltx_trainer.model_loader.resolve_video_vae_path / SpatioTemporalScaleFactors.from_model_config
# do upstream [v1.2.0 verified, LTX25_UPSTREAM_DIFF.md §C], rather than trusting the (8,32,32)
# signet's conditioning/strategy.py assumes for every family today.
def assert_ltx25_vae_compression(checkpoint_path: str, video_vae_path: str | None) -> dict[str, int]:
    """Read the real VAE compression off the mounted checkpoint and assert it matches the
    (8, 32, 32) conditioning/strategy.py hardcodes. Raises loudly on a mismatch — does NOT
    silently accept a resolution bucket validated against the wrong assumption, and does NOT
    invent an EXPECTED_*_25 constant (D3): the assertion target is signet's EXISTING (8,32,32),
    read as a fact to CONFIRM, not a number chosen for 2.5.
    """
    ...  # safe_open(...).metadata(); locate the VAE encoder/decoder block list; count
         # compress_space*/compress_time*/compress_all* prefixes (the exact upstream mechanism);
         # compare to (8, 32, 32); raise with BOTH values named on mismatch.
```

Called once, early, inside `ltx25_preprocess` (before the canonical encode) and again inside
`ltx25_train` (before `PrecomputedDataset` construction) — cheap (header-only, no tensor
materialization, mirrors the existing `_extract_safetensors_ground_truth` header-only pattern
at `train/validate_gate.py:156-209`).

### 4.3 Metadata-driven arch gate — record, don't compare (D3)

A new `read_ltx25_checkpoint_metadata(checkpoint_path)` function, called once at
`ltx25_train` startup (before LoRA injection), that:

1. Opens the checkpoint with `safetensors.safe_open` and reads **both** the embedded
   `f.metadata()` dict (the `metadata["config"]` shape `LTXModelConfigurator.from_metadata`
   consumes upstream) **and** the tensor-name/shape introspection signet already has
   (`_extract_safetensors_ground_truth`-style: block indices, `to_q` inner dim).
2. **Asserts self-consistency only** — no comparison to any hardcoded 2.5 number, because
   none is trustworthy yet (D3: "block count / hidden dim / heads / VAE factors / Gemma-4
   hidden_size are unknowable from the monorepo" per the gate report §C):
   - declared `num_layers` (if present in metadata) matches the max block index found by
     tensor-name introspection + 1;
   - `ff_bias` (if present) is readable and its polarity is **logged, not asserted** — the
     gate report confirms LTX-2.5/gemma4 sets it `False`, but this design does not hardcode
     that as a requirement, since a dev/distilled-style variant split (D-7-BASEVAR's own
     precedent) could exist for 2.5 too and nothing here has verified it yet;
   - the checkpoint's embedded metadata is **present at all** — its total absence (no
     `__metadata__` header) is itself a loud failure, since every other fact this gate needs
     comes from that dict.
3. **Prints a structured summary** of every observed value (`num_layers`, inner dim, head
   count if derivable, `ff_bias`, VAE block-prefix counts) — this is the artifact the operator
   reads to eventually write real `EXPECTED_*_25` constants (D3's own precondition:
   "approve downloading the gated checkpoint... to read embedded metadata before any
   `EXPECTED_*_25` constants are written"). Stage 1 produces that read; it does not consume it.

`EXPECTED_NUM_BLOCKS` / `EXPECTED_HIDDEN_DIM` / etc. (`models/loader.py:55-69`) are **not**
touched, extended, or duplicated with a `_25` suffix. `expected_arch()` (`loader.py:103-126`)
stays 2.3-only; nothing routes a 2.5 checkpoint through it.

### 4.4 Split-layout audio-VAE-encoder guard (silent-failure close, out of Stage-1's load path but worth fencing now)

**[v1.2.0 verified, gate report row 2]** `load_audio_vae_encoder(checkpoint_path, device, dtype)`
was never extended with split-path resolution — handed a split transformer-only file, it
silently builds a **meta-device** encoder (a warning-logged, not raised, failure inside
`SingleGPUModelBuilder._return_model`). Stage 1 does not call this function at all (a2v-on-2.5
is out of scope, §9), but `load_ltx25_components` should refuse to accept
`with_audio_vae_encoder=True` outright (raise `NotImplementedError` naming this exact silent-
failure risk) rather than silently omitting the guard and leaving a landmine for whichever
Stage first turns that flag on.

---

## 5. Preprocess: `modal/fns.py::ltx25_preprocess`

Mirrors `preprocess()` (`fns.py:141-315`) almost verbatim — same "run the CANONICAL script,
don't hand-port it" discipline (SC#2/MODL-04) — with three deltas:

1. `sys.path.insert(0, "/opt/LTX-25/packages/ltx-trainer/scripts")` (not `/opt/LTX-2/...`).
2. `model_path`/`text_encoder_path` defaults point at the 2.5 checkpoint / Gemma-4 root under
   `WEIGHTS_DIR` (config-driven, not hardcoded — threaded from `cfg.model.model_id` /
   `cfg.model.text_encoder_id`, same as today); `video_vae_path=`/`audio_vae_path=` threaded
   from `cfg.ltx25.video_vae_path`/`.audio_vae_path` straight into
   **upstream's own** `preprocess_dataset(video_vae_path=, audio_vae_path=)`
   (**[v1.2.0 verified]** the canonical script gained these kwargs itself — signet's wrapper
   only has to pass them through, no reimplementation).
3. **[REVISED post-refuter A, HIGH — commit-or-vanish]** After the canonical encode, call
   `write_provenance()` (§6) **BEFORE** `dataset_vol.commit()`, never after. `preprocess()`'s own
   docstring names Pitfall 3 "commit-or-vanish": `dataset_vol.commit()` is the TERMINAL Volume
   commit, and anything written to the mount after the last commit vanishes with the container —
   a `PROVENANCE.json` written post-commit would vanish, and `assert_cache_provenance` would then
   treat the freshly Gemma-4-encoded cache as "missing provenance == legacy 2.3" and refuse it:
   fail-closed (no silent corruption), but the Stage-1 workflow could never pass its own gate. The
   correct order is: canonical encode → `write_provenance()` → `dataset_vol.commit()`.
4. **[ADDED post-refuter B, HIGH — the mixed-cache case]** BEFORE any of the above — before the
   canonical encode is even called — `ltx25_preprocess` calls
   `assert_output_dir_ready_for_ltx25_encode(output_dir, gemma_root=, overwrite=)` (§6). Without
   this, `preprocess_dataset`'s default INCREMENTAL behavior (`overwrite=False` skips
   already-encoded samples) means a dir left over from a 2.3/Gemma-3 encode — or a DIFFERENT
   Gemma root under the same generation — would silently keep its stale items untouched while
   step 3 stamped the WHOLE root `'2.5'`, certifying a MIXED Gemma-3/Gemma-4 cache as clean: worse
   than no gate at all, because it actively vouches for corrupted data. Passes on a fresh/empty
   dir, on `overwrite=True` (a forced full re-encode — no stale item can hide underneath), or on a
   dir already carrying MATCHING provenance (an incremental ltx25-on-ltx25 re-run against its own
   prior output); refuses otherwise.

This function takes NO `with_audio`/`reference_column` args at all (unlike `preprocess()`) — a2v
and ic_lora encoding on LTX-2.5 are out of Stage-1 scope (§9); the entrypoint refuses those
conditioning modes before ever calling it (§7).

`@app.function(image=ltx25_gpu_image, ...)` — otherwise the same volumes/secrets/timeout shape
as `preprocess`.

---

## 6. Cache provenance

**[REVISED]** home: `data/cache_provenance.py`, a NEW sibling module — kept SEPARATE from
`data/precomputed.py` so `PrecomputedDataset` itself stays byte-identical (untouched, guarantee
#10, §11) and the read side still has one import to reach for. THREE functions now, not two —
the third (`assert_output_dir_ready_for_ltx25_encode`) is the refuter B HIGH fix.

```python
def write_provenance(output_dir: str, *, ltx_generation: str, gemma_root: str) -> None:
    """Stamp <output_dir>/PROVENANCE.json at encode time. One file per precomputed root — a
    full output_dir is produced by ONE preprocess call with one fixed (generation, Gemma root)
    pair, so root-level granularity is correct, not merely convenient.

    ⛔⛔ CALL ORDER [REVISED post-refuter A, HIGH]: the caller (ltx25_preprocess, §5) MUST call
    this BEFORE dataset_vol.commit(), never after — see §5 delta 3 for the commit-or-vanish
    argument in full.
    """
    ...

def assert_output_dir_ready_for_ltx25_encode(
    output_dir: str, *, gemma_root: str, overwrite: bool
) -> None:
    """[ADDED post-refuter B, HIGH] Refuse an ltx25_preprocess encode into a non-empty output_dir
    lacking MATCHING provenance — the mixed-cache case (§5 delta 4). Passes on a fresh/empty dir,
    on overwrite=True (forces a full re-encode of every sample), or on a dir already carrying
    matching (generation, gemma_root) provenance. Refuses otherwise, naming the exact silent
    corruption this closes: an incremental encode leaving stale 2.3/Gemma-3 (or wrong-root) items
    untouched while write_provenance stamps the whole root '2.5'.
    """
    ...

def assert_cache_provenance(
    output_dir: str, *, configured_generation: str, configured_gemma_root: str
) -> None:
    """Refuse a generation OR Gemma-root mismatch. Missing provenance == legacy '2.3' (every
    cache written before this change) — refused ONLY when configured_generation == '2.5', so
    every existing 2.3 workflow (whose .precomputed/ dirs carry no PROVENANCE.json and never
    will, since preprocess() — the 2.3 fn — is not being changed to write one) stays
    byte-identical: it never calls this function with configured_generation != '2.3' (train() —
    unmodified — never calls it at all), so the check never fires for it at all.

    [REVISED post-refuter B, HIGH] ALSO compares configured_gemma_root against the recorded
    gemma_root, not just the generation — a generation-only check cannot catch a Gemma-ROOT swap
    under the SAME generation (e.g. re-pointing model.text_encoder_id between two Gemma-4
    variants, the codebase's own -it vs -qat-q4_0 root-variant precedent). The
    feature-extractor/connector that produces video_prompt_embeds is dimensioned by the SPECIFIC
    Gemma root paired at encode time, not merely by "which generation" (LTX25_UPSTREAM_DIFF.md
    §D) — a same-generation root swap is exactly as real a mismatch as a generation swap.
    """
    ...
```

Called from `ltx25_train` (and, symmetrically, could be called from the unchanged `train()` for
defense-in-depth, but that would require touching `train()` — deferred; the byte-identity
guarantee (§11) takes precedence, and the risk this closes — a 2.3 cache silently feeding a 2.5
run — cannot occur through `train()`'s own dispatch path, since `train()` is never routed to by
`ltx_generation=='2.5'` configs, §1/§7).

**Why this closes the real risk named in the gate report (§D)**: `load_embeddings_processor`'s
new `gemma_model_path` argument means the feature-extractor/connector that produces
`video_prompt_embeds` is **dimensioned by the specific Gemma root** — a Gemma-3-encoded
`conditions/*.pt` cache cannot feed a Gemma-4-paired checkpoint's connector (different
`hidden_size` at minimum). Today nothing stops that mismatch from being *attempted* (it would
fail deep inside the connector's input projection, not at a checked boundary) — provenance
moves that failure to the earliest possible point: `ltx25_train`'s startup, before any GPU
memory is committed to a doomed run.

---

## 7. Train: `modal/fns.py::ltx25_train` + the mode-refusal home

`ltx25_train()` mirrors `train()` (`fns.py:350-...`) structurally: cold-path import probe →
load config → `validate_mode_config()` (defence-in-depth; the real refusal already fired
pre-dispatch, see below) → `assert_cache_provenance()` (§6, now comparing BOTH generation and
Gemma root) → the metadata-driven arch gate (`read_ltx25_checkpoint_metadata` + §4.2's VAE
check) → `load_ltx25_components()` (§4, always `with_video_vae_decoder=False` AND
`with_text_encoder=False` — training never calls Gemma at all under Stage 1, see the deviations
below) → LoRA injected DIRECTLY (`lora/peft.py`, unmodified, §2 row "—"; NOT reused from a
gate-built adapter, since Stage 1 runs no such gate — see deviation 2) → `PrecomputedDataset`
over the Gemma-4-encoded cache → `train_loop` with `on_checkpoint=None` (no in-loop sampler; the
`"ltx"` family default step/collate hooks are unchanged, §2 row 12).

**[STATED post-refuter B, LOW — an acknowledged Stage-1 coverage gap]** `train()`'s 6-check
architecture validation gate (`train/validate_gate.py::run_validation_gate`) is DELIBERATELY NOT
RUN for `ltx_generation=='2.5'`. Its `EXPECTED_*` constants are 2.3-only (§4.3/§11 #3 — no `_25`
suffix constant is ever written), and its check-#4-FAIL fallback branch (`use_builder` →
`ltx_trainer.model_builder` / wrong `SingleGPUModelBuilder` kwargs) is the confirmed
PRE-EXISTING, pin-independent bug named in the gate report and issue #53's own D4 — deferred, not
fixed by this design. Running the 2.3 gate against a 2.5 checkpoint would assert wrong constants
and risk tripping straight into that broken fallback. This means the adapter-roundtrip proof
(checks #5/#6) and the forward-pass smoke (check #4) have **no 2.5 replacement in Stage 1** — a
real, named coverage loss, deferred alongside D3 (the operator reads the metadata gate's
recorded values instead, and decides from there). LoRA is injected directly (`build_lora_config`
+ `inject_lora`) rather than reusing a gate-built adapter, because there is no gate to build one.

**A second deliberate deviation from `train()`: `ltx25_train` never loads Gemma at all**, not
even the "load for the gate's `hidden_size` read, then free" two-phase dance `train()` runs.
Stage 1's metadata gate reads the CHECKPOINT's own embedded config, never Gemma's, and training
reads precomputed text conditions and never calls Gemma either — so
`load_ltx25_components(..., with_text_encoder=False)`, and there is nothing to free.

### The mode-refusal home — `config/mode_gate.py::validate_mode_config`, not an entrypoint-local gap fn

**[REVISED post-refuter A MEDIUM + refuter B MEDIUM]** the first draft put the `--mode sample`
refusal in a new entrypoint-local `_ltx25_config_gaps` function, mirroring `_qwen_edit_config_
gaps`'s SHAPE. Both refuters independently flagged the same gap from different angles: (A) only
`sample` was refused — `--mode fuse` on a 2.5 config would silently ride the 2.3-pinned fuse arm,
whose `EXPECTED_*` integrity gate assumes fixed tensor names/shapes that LTX25_UPSTREAM_DIFF.md
§C's `ff_bias=false` finding proves wrong for 2.5; (B) the refusal belongs in `config/
mode_gate.py::validate_mode_config` — the ONE CPU-pure home the codebase's own WR-04 hoist
already built, shared by the dry-run CLI (`signet-dryrun --mode`), the entrypoint's pre-dispatch
`run_dryrun(cfg, mode=...)` call, AND the container bodies — not a second, entrypoint-only gap
function that would leave the dry-run CLI and the container bodies mode-blind the same way the
WR-04 hoist's own docstring already documents a prior incident of.

**ALL SIX `KNOWN_MODES` are now explicitly dispositioned**, in `validate_mode_config` itself:

1. **`train`** — ALLOWED. Routes to `ltx25_train`.
2. **`preprocess`** — ALLOWED. Routes to `ltx25_preprocess`.
3. **`sample`** — REFUSED. There is deliberately **no** `ltx25_sample` function in `modal/fns.py`
   to route to (§2 row 7) — rendering needs the removed-and-rebuilt `ValidationSampler`/
   `ValidationRunner` surface (issue #53 D1, Stage 2).
4. **`fuse`** — REFUSED **[ADDED post-refuter A MEDIUM]**. `modal/fuse.py`'s `EXPECTED_*`
   integrity gate is built on LTX-2.3 arch assumptions (fixed tensor names/shapes); a 2.5
   checkpoint's `ff_bias=false` (LTX25_UPSTREAM_DIFF.md §C) is a DIFFERENT tensor-name set, so
   "fusing changes values, never names/shapes" does not hold for it.
5. **`restore`** — explicitly CLEARED, not merely un-refused **[ADDED post-refuter A MEDIUM]**:
   checkpoint-file-level Volume mirroring (a directory copy-back), no arch gate, generation-
   agnostic by construction.
6. **`backup`** — explicitly CLEARED, same reasoning as `restore` (a directory mirror-out).

`ltx25_train` also calls `validate_mode_config(config, "train")` itself as defence-in-depth (the
same pattern `train()` already follows), even though the real `--mode sample`/`--mode fuse`
refusal fires earlier, pre-dispatch, via `run_dryrun(cfg, mode=mode)`. `ltx25_preprocess` does
NOT call it — like `preprocess()`, it takes individually-threaded params rather than a full
`SignetConfig`, and its own dispatch mode (`"preprocess"`) is never one of the two refused modes,
so there is nothing for a defence-in-depth call to catch that pre-dispatch did not already.

### The local runner — one loud refusal line, not silence

**[ADDED post-refuter B MEDIUM]** `local/runner.py::refusals()` gains ONE new check: `model.
ltx_generation != '2.3'` is refused, naming Stage 3 (#25/#30) / issue #53. Without it, a
`ltx_generation=='2.5'` config passes `family == SUPPORTED_FAMILY` (family stays `"ltx"` for both
generations, §0) and every other existing refusal, then crashes cryptically deep inside the
UNMODIFIED, 2.3-only `load_ltxv_components`/`run_validation_gate` the local runner still imports
— on the operator's own workstation, after a multi-GB load. This is a one-line, CPU-pure,
unit-testable refusal squarely inside Stage 1's own silent-failure-closing mandate; it is NOT the
local runner "gaining LTX-2.5 support" (§9's non-goal is unchanged and still holds for everything
past this one refusal line).

---

## 8. Dryrun coverage

`dryrun/shapes.py::build_dryrun_inputs` (`shapes.py:1332`) dispatches on `cfg.model.family`
today (`"h3"` → `_build_h3_dryrun_inputs`, `"qwen_edit"` → `build_qwen_edit_dryrun_inputs`,
else the LTX default path via `_assert_contract`). Since `family` stays `"ltx"` for 2.5, the
existing LTX branch already runs for a `ltx_generation=='2.5'` config — the new coverage is an
**addition inside** that branch, not a new top-level dispatch arm:

- `_assert_contract` (renamed conceptually to cover both generations) gains a
  `ltx_generation`-conditional sub-check: when `'2.5'`, additionally assert
  `cfg.ltx25.checkpoint_layout` is a legal value, `video_vae_path` is set when `layout=='split'`
  (redundant with the schema-level guard, §1 — cheap, and dryrun should never rely on a guard
  living only in `SignetConfig` construction, since `build_dryrun_inputs` can in principle be
  called with a config already past that stage — belt-and-braces).
- One new dryrun-only synthetic-batch case exercising the (8,32,32) assumption explicitly, so a
  CPU-only run has SOME coverage of the "if the real 2.5 VAE compression differs, does the
  seq_len arithmetic at least not silently produce a plausible-but-wrong shape" question —
  bounded to what's checkable **without weights** (§10): this cannot prove the real compression
  factor, only that the dryrun's own math is self-consistent under the assumed one.
- `run_dryrun` (`shapes.py:2036`) needs no change — it already threads `cfg` through to
  `build_dryrun_inputs`/`_assert_contract` generically.

---

## 9. Explicit non-goals (Stage 1 does NOT do this)

- **Sampling / rendering of any kind for `ltx_generation=='2.5'`** — refused loudly at two
  points (§7). `inference/sampler.py`, `inference/upscale.py`,
  `inference/ic_lora_pipeline.py` are **not touched**; they stay 2.3-only, unaware 2.5 exists.
  `ltx_pipelines` (the package `ICLoraPipeline`/`TI2VidTwoStagesPipeline` live in) is **not
  installed** in `ltx25_gpu_image` (§3) — there is nothing in Stage 1 that imports it.
- **Any hardcoded `EXPECTED_*_25` constant.** §4.3 produces an observed-values report; it does
  not consume one. Writing those constants is explicitly D3's precondition, unresolved.
- **a2v (audio) training or encoding on LTX-2.5.** `ltx25.audio_vae_path` exists on the config
  surface (forward-declared, §1) but nothing in Stage 1 reads it; `load_audio_vae_encoder`'s
  split-path silent-failure risk (§4.4) is fenced, not exercised.
- **The local runner** (`local/runner.py`) gaining any LTX-2.5 *support* — that is Stage 3
  (#25/#30), and per issue #53's own scoping, likely int8-quanto/deep-swap territory that needs
  measurement, not assumption, before any code is written. **[NARROWED post-refuter B MEDIUM]**
  this non-goal does NOT extend to a REFUSAL: `refusals()` gains exactly one line naming Stage 3
  loudly for `ltx_generation != '2.3'` (§7) — a refusal is not support, and it is CPU-pure/free.
- **[ADDED post-refuter A MEDIUM] Weights staging.** How the gated 2.5 checkpoint + Gemma-4 root
  arrive on `WEIGHTS_DIR` is explicitly OUT of Stage 1's code surface — no `download_ltx25_
  weights` function is added (the qwen_edit-PR rubric's own precedent, `download_qwen_edit_
  weights`, is a real touchpoint this design's first draft had no row for at all). Staging is
  MANUAL until issue #53's D3 (the gated-checkpoint-download approval) resolves; a download fn is
  explicitly deferred to whichever stage D3's approval lands in, not built speculatively here.
- **The two pre-existing, pin-independent bugs** the gate report flags incidentally (D4: the
  nonexistent `ltx_trainer.model_builder` import + wrong `SingleGPUModelBuilder` kwargs in the
  `use_builder` fallback; `LTXV_MODEL_COMFY_RENAMING_MAP` imported from the wrong module,
  `local/runner.py:289` / `modal/fns.py:542`) — these are **2.3-affecting today**, independent
  of this bump, and are an operator ruling (D4) on whether to fix them in this PR or separately.
  This design does not fix them as a side effect of touching nearby lines.
- **Fixing `inference/upscale.py`'s already-wrong `TI2VidTwoStagesPipeline` call** (flagged
  MEDIUM-confidence/unexercised in signet's own docstring even at the 2.3 pin) — unrelated to
  this bump, Stage-2-adjacent at best.
- **Bumping `LTX2_COMMIT_SHA` / touching `gpu_image` in place.** Explicitly rejected by the gate
  report's Recommendation (E) and by this design (§0, §3, §11).

---

## 10. Test plan

Everything below runs **without a real checkpoint or GPU** — the mocking boundary is exactly
where `models/ltx25_loader.py` draws it (`ltx_trainer` imports are function-local), so a test can
exercise the *argument-threading* contract by injecting a FAKE `ltx_trainer.model_loader` module
into `sys.modules` (never installing the real, uninstalled package) with a stub `load_model` that
records its kwargs. `safetensors` IS a real, already-installed dependency, so the metadata-gate
tests build real tiny `.safetensors` fixtures via `safetensors.torch.save_file`.

| Area | Test | What it proves without weights |
|---|---|---|
| Schema | `test_ltx25_config.py::test_default_generation_is_23` | Every existing config (no `ltx_generation` key) parses to `"2.3"`, byte-identical |
| Schema | `test_ltx25_config.py::test_byte_identity_regression_dump_is_unaffected_by_the_new_fields` | A loaded 2.3 config's `model_dump()` carries the new fields ONLY at their all-default values |
| Schema | `test_ltx25_config.py::test_ltx25_block_reverse_guard_fires_naming_the_offending_field` | Setting `ltx25.*` non-default while `ltx_generation=='2.3'` raises, naming the offending field |
| Schema | `test_ltx25_config.py::test_ltx_generation_reverse_guard_fires_outside_family_ltx` | Setting `ltx_generation: '2.5'` while `model.family != 'ltx'` raises |
| Schema | `test_ltx25_config.py::test_split_layout_requires_video_vae_path` | `checkpoint_layout: split` with `video_vae_path` unset raises at config load |
| Schema | `test_ltx25_config.py::test_in_loop_sampling_refused_on_25` | `ltx_generation: '2.5'` + `validation.in_loop_sampling: true` raises |
| Schema | `test_ltx25_config.py::test_two_stage_upscale_refused_on_25` **[ADDED post-refuter B]** | `ltx_generation: '2.5'` + `validation.two_stage_upscale: true` raises |
| Schema | `test_ltx25_config.py::test_distilled_pairing_check_is_scoped_to_generation_23_not_25` **[ADDED post-refuter B]** | A `'2.5'`+`'distilled'`-named config hits ONLY the new 2.5 guard, never the 2.3 D-7-BASEVAR pairing check |
| Schema | `test_ltx25_config.py::test_distilled_still_refused_on_23_without_two_stage_upscale` | Regression: the pre-existing 2.3 D-7-BASEVAR check keeps its exact behavior |
| Schema | `test_ltx25_config.py::test_example_config_loads` | The shipped `configs/ltx25_train.example.yaml` loads clean |
| Loader | `test_ltx25_loader.py::test_load_ltx25_threads_split_paths_unchanged` | Fake `ltx_trainer.model_loader`; `load_ltx25_components(video_vae_path=X, audio_vae_path=Y)` forwards both kwargs unchanged and passes `with_video_vae_decoder=False` |
| Loader | `test_ltx25_loader.py::test_decoder_flag_raises_before_any_ltx_trainer_import` | `with_video_vae_decoder=True` raises `NotImplementedError` naming Stage 2, with `ltx_trainer` deleted from `sys.modules` (proves the raise is BEFORE the deferred import) |
| Loader | `test_ltx25_loader.py::test_audio_encoder_flag_raises_before_any_ltx_trainer_import` **[the §4.1/§4.4 signature fix]** | Same shape for `with_audio_vae_encoder=True` |
| Arch gate | `test_ltx25_loader.py::test_read_metadata_missing_header_raises` | A synthetic `.safetensors` with no `__metadata__` block raises loudly, naming what's missing |
| Arch gate | `test_ltx25_loader.py::test_read_metadata_num_layers_self_consistency_passes` / `test_read_metadata_num_layers_mismatch_raises` | Consistent synthetic fixture (`num_layers: 3` + 3 fake blocks) passes; mismatched (`num_layers: 5`, 3 blocks) raises |
| Arch gate | `test_ltx25_loader.py::test_read_metadata_records_without_asserting_against_any_hardcoded_25_constant` | D3 honesty: an unknown declared `num_layers` still returns a summary rather than raising |
| VAE check | `test_ltx25_loader.py::test_vae_compression_mismatch_raises_naming_both_values` | Synthetic metadata resolving to `(4, 16, 8)` instead of `(8, 32, 32)` raises, naming both |
| VAE check | `test_ltx25_loader.py::test_vae_compression_reads_split_video_vae_path_when_given` | Split layout reads the VAE metadata from the SEPARATE `video_vae_path` file |
| Cache provenance | `test_ltx25_cache_provenance.py::test_write_then_assert_roundtrip_passes` | `write_provenance` then `assert_cache_provenance` with matching `(generation, gemma_root)` passes |
| Cache provenance | `test_ltx25_cache_provenance.py::test_missing_provenance_refused_only_for_25` | Missing provenance passes silently for `configured_generation="2.3"`, raises for `"2.5"` |
| Cache provenance | `test_ltx25_cache_provenance.py::test_generation_mismatch_refused_even_though_both_are_present` | A `'2.3'`-provenanced cache is refused for a `'2.5'`-configured run |
| Cache provenance | `test_ltx25_cache_provenance.py::test_gemma_root_mismatch_refused_under_the_same_generation` **[ADDED post-refuter B, HIGH]** | A SAME-generation Gemma-root swap is caught — the generation-only check in the first draft could not see this |
| Cache provenance | `test_ltx25_cache_provenance.py::test_nonempty_dir_with_no_provenance_is_refused_without_overwrite` **[ADDED post-refuter B, HIGH]** | The mixed-cache case: a non-empty `output_dir` with no matching provenance is refused |
| Cache provenance | `test_ltx25_cache_provenance.py::test_nonempty_dir_with_no_provenance_is_allowed_with_overwrite_true` | `overwrite=True` (a full re-encode) is the escape hatch |
| Cache provenance | `test_ltx25_cache_provenance.py::test_nonempty_dir_with_matching_provenance_is_allowed_without_overwrite` | An incremental ltx25-on-ltx25 re-run against its own prior output is fine |
| Cache provenance | `test_ltx25_cache_provenance.py::test_nonempty_dir_with_provenance_for_a_different_gemma_root_is_refused` | The mixed-cache refusal extends to a root swap, not just a generation swap |
| Dryrun | `test_ltx25_dryrun.py::test_dryrun_25_split_layout_produces_the_same_shape_as_23` | `build_dryrun_inputs` on a `ltx_generation=='2.5', checkpoint_layout='split'` config produces the same `ModelInputs` shape as an equivalent 2.3 config |
| Dryrun | `test_ltx25_dryrun.py::test_dryrun_23_unaffected_byte_for_byte` | A 2.3 config's dry-run output is untouched |
| Dryrun | `test_ltx25_dryrun.py::test_run_dryrun_refuses_sample_mode_for_a_25_config` / `test_run_dryrun_refuses_fuse_mode_for_a_25_config` **[the fuse-refusal fix]** | `run_dryrun(cfg, mode=...)` refuses BOTH sample and fuse |
| Dryrun | `test_ltx25_dryrun.py::test_run_dryrun_allows_restore_and_backup_for_a_25_config` | restore/backup are cleared, not refused |
| Image | `test_modal_gpu_image.py::test_ltx25_commit_sha_is_a_literal_40_hex_sha_never_main` | `LTX25_COMMIT_SHA` is a literal 40-hex SHA matching the v1.2.0 tag, never `main`; `LTX2_COMMIT_SHA`/`gpu_image` byte-identity re-asserted in the same test |
| Image | `test_modal_gpu_image.py::test_ltx25_transformers_pin_is_a_literal_version_in_upstream_window` | `LTX25_TRANSFORMERS_VERSION` parses inside `>=5.8.0,<5.15` |
| Image | `test_modal_gpu_image.py::test_ltx25_gpu_image_shares_the_tri_family_torch_line` | Regex-extract the `torch~=`/`torchvision~=` literal from every GPU image; assert all equal |
| Image | `test_modal_gpu_image.py::test_ltx25_gpu_image_does_not_install_ltx_pipelines` | `ltx-pipelines` is absent from the `ltx25_gpu_image` block |
| Image closure | `test_modal_image_config_closure.py` (extended) **[a REAL gap this test caught during implementation]** | `ltx25_gpu_image` needed a registered transitive supplier for `pydantic`/`pyyaml` (the SAME defect class `h3_gpu_image` hit in 2026-08-06 — `ltx25_train` calls `load_config_from_text` in-container) — proves the closure gate itself, not merely a config-shape claim |
| Entrypoint | `test_ltx25_entrypoint.py::test_ltx_generation_25_train_mode_dispatches_ltx25_train_not_the_23_arm` | A `ltx_generation=='2.5'` config on `--mode train` dispatches `ltx25_train.spawn()`, never `train()` |
| Entrypoint | `test_ltx25_entrypoint.py::test_ltx_generation_23_train_mode_still_dispatches_the_plain_train_arm` | Regression: an ordinary config still routes to the unchanged `train()` |
| Entrypoint | `test_ltx25_entrypoint.py::test_ltx_generation_25_preprocess_mode_dispatches_ltx25_preprocess` | Same for `--mode preprocess`, plus asserts the threaded kwargs carry the CONFIG's `model_id`/`text_encoder_id`/`ltx25.*`, not the fn's own standalone defaults |
| Entrypoint | `test_ltx25_entrypoint.py::test_ltx_generation_25_sample_mode_is_refused_pre_dispatch` | `--mode sample` is refused by the REAL `mode_gate` (not a stub) before any dispatch |
| Local runner | `test_ltx25_local_runner.py::test_refuses_ltx_generation_25_with_issue_53_pointer` | `refusals()` names `ltx_generation`/issue #53 for a `'2.5'` config |
| Local runner | `test_ltx25_local_runner.py::test_refusal_survives_alongside_family_refusal` | Both refusals fire independently — collected, not short-circuited |
| Regression | full CPU suite, sorted-FAILED diff against the pre-existing 40 known-reds | Identical set — proves every new field/guard/function is additive |

Not testable without weights (explicitly out of this plan, deferred to the gated GPU exercise
D3 authorizes): the real observed `num_layers`/`ff_bias`/VAE-compression values on an actual
LTX-2.5 checkpoint; whether `LTX25_TRANSFORMERS_VERSION` actually loads a real Gemma-4 root;
whether `ltx25_gpu_image` actually builds (editable install against the real `v1.2.0` tag);
whether `ltx25_train`/`ltx25_preprocess` actually run end to end on a real GPU.

---

## 11. Byte-identity guarantees for LTX-2.3

Every one of these is a **negative** claim — enumerated because §10's regression suite exists
to hold them, not because any of them requires new code:

1. `LTX2_COMMIT_SHA` and `gpu_image` in `modal/app.py` — untouched.
2. `load_ltxv_components` / `EXPECTED_*` / `expected_arch()` in `models/loader.py` — untouched;
   `load_ltx25_components` lives in the NEW, SEPARATE `models/ltx25_loader.py` (§2 row 5,
   **[REVISED]** — an even stronger guarantee than the first draft's "new function in the same
   file," since nothing in `models/loader.py` is edited at all).
3. `EXPECTED_NUM_BLOCKS` / `EXPECTED_HIDDEN_DIM` / … / `expected_arch()` — untouched, no `_25`
   suffix constants added anywhere near them.
4. `conditioning/strategy.py`'s `TIME_SCALE`/`HEIGHT_SCALE`/`WIDTH_SCALE` — untouched;
   `config/validators.py`'s frame-law checks — untouched.
5. `preprocess()` and `train()` in `modal/fns.py` — untouched; `ltx25_preprocess`/`ltx25_train`
   are new functions. Neither is ever reached by a `ltx_generation != '2.5'` config — the
   entrypoint's dispatch condition and `ltx25_train`'s own internal guard (§7) both assert this.
6. `inference/sampler.py`, `inference/upscale.py`, `inference/ic_lora_pipeline.py` — untouched.
7. `train/family_hooks.py`'s `LOOP_HOOKS_BY_FAMILY` table — untouched (no `"ltx25"` key is ever
   looked up, since `family` never becomes `"ltx25"`, §0/§2 row 12).
8. `lora/peft.py`'s LTX target-module tuple — untouched.
9. Every existing YAML under `configs/` — parses to the identical `SignetConfig` it parses to
   today (`ltx_generation` defaults to `"2.3"`, `ltx25` block is all-default).
10. `data/precomputed.py::PrecomputedDataset` — completely UNMODIFIED (**[REVISED]** not merely
    "unmodified read path" — the provenance functions live in the separate `data/
    cache_provenance.py`, §6, so `precomputed.py` itself has zero new lines). `train()` — also
    unmodified — never calls `assert_cache_provenance` at all; only `ltx25_train` does.
11. **[ADDED]** `local/runner.py::refusals()` gains exactly ONE new `if` block (§7/§9); every
    other line, and every existing refusal's independent firing, is unchanged.
12. **[ADDED]** `config/mode_gate.py::validate_mode_config`'s two pre-existing `multi_frame`
    refusal blocks are unchanged; the new ltx25 block is appended, not interleaved.

---

## 12. Carried-forward operator decisions from #53

Stage 1 as designed above **takes a position** on D2 (§0: explicit `ltx_generation` field,
not a new family, not auto-detect) and **defers** D1 (validation/sampling architecture —
irrelevant until Stage 2, since Stage 1 refuses sampling AND fuse outright, §7) and D4 (the two
pre-existing bugs — untouched, §9). **D3 remains the hard gate on Stage 1 landing anything beyond
the metadata-*reading* machinery (§4.3)**: no `EXPECTED_*_25` constant, no VAE-compression
number, no Gemma-4 `hidden_size` is written into this codebase by this design — only the code
that would read and report them once the gated checkpoint download D3 asks the operator to
approve actually happens.

**Revision provenance.** This document was read by two independent refuter passes against the
live repo (one verdict MUST-CHANGE, one MUST-FIX) after the first draft landed; every finding
from both is incorporated at its landing site above (§1, §4.1, §4.4, §5, §6, §7, §9, §10, §11)
and summarized in the banner at the top of this file. No finding was dropped or downgraded. The
implementation on `feat/ltx25-stage1` was built against this REVISED design, not the first
draft — the two occasionally differ from the first draft in file placement (`models/
ltx25_loader.py` / `data/cache_provenance.py` as NEW, separate modules rather than functions
folded into `models/loader.py` / `data/precomputed.py`) as well as in the substantive fixes
listed above; both kinds of drift are noted inline where they occur.
