---
name: training-prep
description: >-
  Drive dataset staging + the gated `--mode preprocess` canonical pre-encode for a signet-trainer
  LoRA run, and SUGGEST chained / shared-transformer training at data-prep time. Use this when
  preparing a dataset for training: staging clips + captions, encoding latents/conditions (and
  reference_latents for ic_lora) through the entrypoint gate, and deciding whether the run should
  chain. Adapts the caption/preflight discipline to video with honest `[PENDING: operator]` marks.
  Modality-extensible — video is the current configuration, not a hardcoded assumption.
---

# training-prep — stage + encode a dataset for a gated LoRA run

> ### ⓘ Beta note — companion tooling and house memory are NOT bundled
>
> These skills are the working playbooks used to operate this trainer in-house, shipped as-is
> because the *methodology* is the point. Two classes of thing they name are **not part of this
> release** and you will not be able to obtain them:
>
> - **Companion tooling** — `alvdansen/finetune-gridwatch` (the canonical sample-grid builder/server)
>   and **Klippbok** (the captioning tool these skills treat as the caption gold standard) are
>   separate projects, not dependencies of `signet-trainer`. Where a skill says "build the grid with
>   gridwatch" or "caption per Klippbok", read it as *"use your own equivalent, and keep the
>   discipline the step describes"* — a fixed-layout comparison grid, and captions that describe
>   action/setting rather than appearance.
> - **The house memory scaffold** — `.planning/harness/` (`DECISION-LOG.md`, `KNOWLEDGE.md`,
>   `SESSION-STATE.json`, `HOUSE-SPEC.yaml`, `TIER-TAXONOMY.yaml`, `MASK-SPEC*.yaml`, campaign cards)
>   holds project-private run history and is not published. Steps that say "read the scaffold first"
>   still describe the right habit — read *your own* running record before improvising, and append a
>   decision after. If you want the machinery, create `.planning/harness/` yourself; the loaders in
>   `src/signet_trainer/harness_state.py` document the expected shape of each file.
>
> Nothing in the metered-run discipline depends on either: the cost print, the approval gate, the
> cumulative session cap, and the spend ledger are all implemented in shipped code
> (`src/signet_trainer/modal/{cost,session_cap,entrypoint}.py`) and are config-driven, not
> house-specific. Dollar figures shown anywhere in these skills are conservative defaults
> (`DEFAULT_SESSION_CAP_USD`), not anyone's account limits.


This skill owns the **prepare** stage of the harness loop:
`prepare (this skill) → dry-run gate → gated train launch → monitor → sample → grid → surface`.
It stages the dataset, drives the **canonical gated `--mode preprocess` encode**, and — because
chaining is a *preferable* technique the harness must not forget — **suggests chained training at
data-prep time**. Everything runs through the same gate as training; there is **no bespoke encode
path** (the old `scripts/_encode_*.py` are retired).

> **Modality (do NOT hardcode video-only):** every rule below is stated against the *current*
> configuration, which is video. Video is a configuration, not an assumption — keep the skill
> modality-extensible (image support is a future pass; just do not hardcode against it).

> **Routing — inpaint datasets go to the sibling.** If the dataset carries inpaint masks (a
> `video_masks` column / a masked-region generate task), STOP and use the **`training-prep-inpaint`**
> sibling skill instead. It reuses §0 STATE-DETECT and §2 encode gate below verbatim and adds the full
> mask playbook (spec read → hand-draw app *or* teal paintover → SAM3 propagate → D-04 QA hard gate →
> staging → the same gated `--mode preprocess` encode). The human tutorial is `docs/inpaint-data-prep.md`.
> This skill stays focused on the standard (non-masked) flow.

---

## 0. STATE-DETECT preamble (D-8-STATEROUTE) — retrieve before improvising

Do this FIRST, every session, before touching a dataset:

1. **Read the memory scaffold** (`.planning/harness/`): `KNOWLEDGE.md` (distilled house policy) and
   `DECISION-LOG.md` (dated decisions). Search by bare keyword (`rg -i "prodigy|klippbok|chaining|frames"`).
   Retrieve the standing answer before reasoning from priors — an untagged model prior is the
   documented drift failure mode.
2. **Detect real dataset state** — do NOT assume. Check whether the target dataset is already encoded:
   ```
   modal volume ls signe-trainer-dataset <preprocessed_data_root>
   ```
   ⛔ **"Encoded" means EVERY source the strategy declares is present AND their `.pt` counts are
   EQUAL. Never infer a completed encode from one dir.** `PrecomputedDataset` pairs sources by
   relative path, so an unequal tree does not raise — it silently drops the unpaired samples from
   the index. Count, do not glance:
   - **LTX**: `latents/` + `conditions/` (+ `reference_latents/` for an ic_lora run).
   - **H3 (`family: h3`)**: **all FOUR** — `h3_latents/`, `h3_conditions/`,
     `h3_reference_latents/`, `h3_audio_latents/`. The audio one is written even when
     `with_audio: false` (an explicit absent marker), so its absence means an INCOMPLETE encode,
     not "this campaign has no audio".
   - any source missing, or counts unequal → **PARTIAL**. Name exactly what is missing and
     re-encode (below). A re-run rewrites the same relative paths; **never delete the partial tree**
     (house rule: never auto-delete intermediates).
   - config/metadata not staged yet → hand off to staging first.

   > ⚠ **Precedent, 2026-08-06.** A failed H3 pre-encode committed `h3_conditions/` (88 payloads)
   > and nothing else. A dir-presence reading of that tree says "conditions present → encoded". It
   > is not: it is one source of four. That tree is additionally a **stale payload version** and is
   > refused by `prep/h3_text_payload.read_h3_text_state` with the re-encode named — so a skip now
   > costs a CPU preflight failure rather than a silently-wrong run, but the state-detect must not
   > produce the misread in the first place.
3. **Proceed / name-missing / hand-off** — never silently guess. If the config is invalid or a
   referenced path does not exist, say so and stop; do not burn a gate on a broken config.

---

## 1. Philosophy + source-tag preamble (D-8-CANON / D-8-SOURCETAG)

Prep decisions are **philosophy territory** (they change *what the model learns*): house rules first,
ask when uncovered, community only via the research loop (§6). Anchor on the canonical methodology —
reference, **never re-derive**:

- `general-training-methodology/dataset-and-captioning.md` — the caption + dataset doctrine.
- **Klippbok is the caption gold standard.** Caption to **disentangle**; the anchor is a natural
  name, **never a trigger token** (trigger tokens are banned for LTX / flow-match — a myth). Reuse
  `klippbok score` / `klippbok audit` — do **NOT** reimplement a caption linter. If Klippbok tooling
  is absent in this environment, mark the caption-audit steps **`[PENDING: operator]`** rather than
  hand-rolling one.
- Every recommendation you log carries a **source tag**: `[house]` · `[precedent]` · `[canonical]` ·
  `[community]`. Untagged = a violation.

**Pre-flight checklist — adapt image-flavored checks to video/LTX, mark unknowns honestly.** The
methodology's checklist is partly image-flavored (alpha-channel, letterboxing, caption/file orphan).
Carry the ones that transfer 1:1 (caption/file orphan pairing, resolution-bucket sanity), and for the
video equivalents that are **not enumerated** — per-frame vs per-clip alpha, temporal
corruption/warp, frame-count alignment vs caption — mark them **`[PENDING: operator]`** instead of
guessing (D-8-CANON: do not invent philosophy).

---

## 2. ENCODE — drive `--mode preprocess` through the SAME gate (D-8-PREPROC)

The pre-encode is a **first-class gated mode** on the canonical entrypoint. Never write or run a
bespoke encode script — the old `scripts/_encode_*.py` are the **retired anti-pattern** (they hardcoded
`HOURLY_RATE_USD`; the gate reads rates from `cfg.modal` instead, config-first).

```
PYTHONPATH=src PYTHONUTF8=1 modal run --detach -m signet_trainer.modal.entrypoint \
    --config <yaml> --mode preprocess [--approve]
```

**`--detach` is REQUIRED** — a multi-hour encode is a metered dispatch, and without it Modal tears
the app shell down with the local client, killing the encode partway with nothing committed to the
dataset Volume (README "The three things about this command that are not optional").

> **Cap step (D-8-YOLOCAP) — before dispatching `--approve` here:** the cumulative session-spend
> cap is enforced BY THE ENTRYPOINT GATE itself (`session_cap.read_ledger` + `session_cap_check`,
> between the cost print and the approval pause) — going over cap disables `--approve` for that
> dispatch and drops to an interactive ask-first prompt even when the flag was passed. This skill
> does not re-derive that chain; see `training-run/SKILL.md` §3 for the full WR-02 cap/ledger-path
> resolution and the blanket → strict/yolo dispatch decision every metered run here inherits.

- The gate runs **load → dry-run → cost print → BLOCKING approval → dispatch**, identical to train.
  The **cost print + approval precede dispatch** — a metered encode can never auto-launch. Under an
  active blanket the approval stop is removed but the cost line still logs (accounting always runs).
- **The dataset encode params come from `cfg.data`** (config-first, D-NOHARDCODE): `metadata_path`,
  `resolution_buckets` (WxHxF strings, pre-parsed to (F,H,W) tuples by the entrypoint),
  `preprocessed_data_root` (the output dir); the **reference params come from `cfg.conditioning`**
  (the per-mode bullet below), NOT `cfg.data`. It runs the canonical `process_dataset.py`
  (`preprocess_dataset`) — **not** a custom encoder (the enochiatron landmine).
- **Per-mode preflight (D-8-MODECONFIG):** an `ic_lora` run additionally needs the **paired
  reference_latents** encoded. The reference params live on **`cfg.conditioning`** (NOT `cfg.data` —
  the schema is `extra="forbid"`, so reading them off the wrong block raises *after* approval, a
  burned gate): `reference_column` (e.g. `"reference_path"`), `reference_downscale_factor`,
  `reference_latents_dir`. Confirm these are set BEFORE dispatching an ic_lora encode.

---

## 2b. MiniMax-H3 Ref2VA encode — the SAME `--mode preprocess`, routed by FAMILY (Phase 10, H3-03)

There is **no new mode**. `--mode preprocess` on a config carrying `family: h3` dispatches
`h3_preprocess` instead of the LTX `preprocess`; the entrypoint routes on `model.family` inside the
existing arm. Same command, same gate, same cost line, same spend ledger:

```
PYTHONPATH=src PYTHONUTF8=1 modal run --detach -m signet_trainer.modal.entrypoint \
    --config configs/<your-h3-run>.yaml --mode preprocess [--approve]
```

> **Cap step (D-8-YOLOCAP)** — same as §2: the entrypoint gate enforces the cumulative session cap
> on this dispatch too (same single gate, all six `--mode` × family combinations). See
> `training-run/SKILL.md` §3 before relying on `--approve` here.

**What it writes** — FOUR sources under `cfg.data.preprocessed_data_root`, not two:

| dir | contents |
|---|---|
| `h3_latents/` | the target video latents, `[24, F, H, W]` |
| `h3_conditions/` | Qwen3-VL text conditioning at `hidden_states[50]` |
| `h3_reference_latents/` | the per-slot reference list (see the payload note below) |
| `h3_audio_latents/` | only when `with_audio`; **measured 0 of 44 clips carry audio** (D-10-AUDIO) |

State-detect accordingly: `modal volume ls signe-trainer-dataset <preprocessed_data_root>` must show
the `h3_`-prefixed dirs — the LTX `latents/` + `conditions/` names are a DIFFERENT encode and their
presence does not mean an H3 run is ready.

**The 2-slot reference rule (operator ruling, and it is a budget invariant not a style choice).**
Every sample carries EXACTLY `h3.references_per_sample` = **2** reference slots. A non-environment
segment gets two rotating character refs; an environment-bearing segment gets one rotating character
ref **plus the environment ref, which SUBSTITUTES for the second character slot — it is never
appended**. Three slots were never priced by the packed-sequence VRAM budget, so a third would OOM a
metered container. The environment ref goes **LAST** (D-10-REFORDER): order fixes the `<Picture i>`
labels AND advances the shared rotary clock, so a reordered set is a *different request* — training
and inference must apply the same order.

**`reference_image_short_edge: 896` is a decision with a re-encode attached.** 896 is the highest
fidelity that fits one A100-80GB: the worst of the 15 real pairs is 12,394 packed rows of a
~13,777-row ceiling, while at 1024 the worst is 14,026 and six of the twelve
character-by-environment pairs exceed it. ⚠ **VAE latents cannot be spatially downscaled after the
fact** — a higher-fidelity campaign needs a FULL RE-ENCODE, not a config edit. Budget for it before
promising one.

**Two-phase VRAM, tighter than LTX's.** Qwen3-VL (~62.1 GiB) and the H3 transformer (61.7 GiB) can
**NEVER** coexist on one 80 GiB card. `h3_preprocess` does PHASE A (text) -> `free_text_encoder` +
the caller-side reference drop -> PHASE B (VAEs), and prints the freed-VRAM delta. Same discipline as
the LTX Gemma rule; different code, because `CachedPromptEmbeddings` / `load_embeddings_processor`
are ltx-trainer objects with no H3 equivalent.

**Consumer note (do not get this wrong):** the reference payload's real data is
`payload["references"]`, a LIST of per-slot dicts. The top-level `"latents"` key **ALIASES SLOT 0
ONLY** — it exists purely to satisfy the precomputed-source allowlist. A consumer reading only the
top-level tensor silently sees one reference where the sample carries two.

**The reference DESCRIPTORS come from the MANIFEST — this is the one thing to get right when you
stage an H3 corpus.** Each cached slot must carry `path` / `kind` / `subject_id` alongside its sizes,
because the selector needs `kind` (the environment ref goes LAST) and `subject_id` (images of one
subject are grouped, and it is the label a budget refusal names, e.g. `C+008`). **None of the three
is recoverable from a latent tensor, and none is closable from config** — the config's
`character_reference_sizes` / `environment_reference_sizes` declare the pricing DOMAIN and their
third tuple element IS the `subject_id`, but they carry no `path`, and joining a slot to a config
entry by `source_wh` is ambiguous (two environment refs are both 1344x768). So the pre-encode reads
them out of the manifest and writes them into the payload; it **refuses before encoding anything** if
they are absent, because the alternative is paying for a whole pre-encode that produces a cache the
trainer cannot read.

Each H3 manifest row is therefore:

```jsonc
{
  "media_path": "clips/segment_000.mp4",
  "caption": "...",
  // the D-10-PAIRSEED pool. The KEY supplies `kind`, so an entry needs path + subject_id.
  "character_references": [
    {"path": "refs/<file>", "subject_id": "A"},
    {"path": "refs/<file>", "subject_id": "B"},
    {"path": "refs/<file>", "subject_id": "C"}
  ],
  // optional; SUBSTITUTES for the last character slot, never appended.
  "environment_reference": {"path": "refs/<file>", "subject_id": "029"}
}
```

`reference_paths` (an explicit per-sample list, used verbatim) is still accepted, but a flat list
says nothing about what its members ARE, so every entry there must also declare
`"kind": "character" | "environment" | "prop"`.

Rules the refusals enforce, all before any spend: paths are **manifest-relative** (an absolute mount
prefix must never reach a cached payload), the `subject_id` vocabulary should match the labels the
config declares, and two slots may not share a `path` — `path` is the join key the trainer gathers
reference rows by, so a duplicate silently collapses two slots into one reference used twice.

⛔ **Never guess `kind` or `subject_id`.** A guessed kind silently reorders the references, and the
shared rotary clock makes a reordered set a *different request* — the run trains against
conditioning nobody asked for, at a perfectly valid shape.

---

## 3. CHAINING SUGGESTION at data-prep (never-forget)

**Chaining is a preferable technique** — the harness **SUGGESTS** chained / shared-transformer
training **at data-prep time** (not mid-run), whenever it applies. Precedent: the Qwen → Qwen-Edit
lineage `[house] [precedent]`. When the dataset/goal looks like a staged progression (base capability
then a refinement, or a shared transformer feeding multiple adapters), **raise the chaining option
now** — it is far cheaper to structure at prep than to retrofit.

Document the **chain-manifest** as a scaffold (the machinery), and mark concrete field-locking
`[PENDING: operator]` — actually *executing* a chained run is future work:

```
chain-manifest (SCAFFOLD — shape only):
  stages:                       # ordered list; each stage trains/consumes in sequence
    - name: <stage id>
      inputs: <dataset / prior-stage outputs>          [PENDING: operator — exact inputs]
      frozen_weights: <which adapter/base is frozen>   [PENDING: operator — freeze spec]
      outputs: <checkpoint / adapter this stage emits>  [PENDING: operator — output contract]
```

The freeze mechanics already exist (D-7-FREEZE: `stack_frozen_adapter` + optimizer excludes frozen
params) — chaining reuses them. Do not invent the field-locking; surface the scaffold and let the
operator lock the concrete fields.

---

## 4. LANDMINES the prep skill MUST carry

- **`(frames - 1) % 8 == 0`** — LTX temporal-compression alignment; the config gate (CONF-02) fails
  fast on violation. **Wan's `num_frames >= 33` rule does NOT apply to LTX** (`[precedent] [canonical]`).
- **Single A100-80GB.** Never multi-GPU, never a fallback GPU list, never a warm/idle container —
  forbidden even in yolo mode (`[house]`).
- **Config-first / no-hardcoding (D-NOHARDCODE).** Buckets, rates, paths, sampling defaults live in
  the YAML with documented defaults — never hardcoded in code. The retired `scripts/_encode_*`
  hardcoded rate is the anti-example.
- **Retired-encode-script anti-pattern.** If you find yourself about to write a one-off encode
  driver, STOP — route through `--mode preprocess` instead.
- **Stale-VRAM discipline** (only for live GPU debugging): `pkill python3` + verify 0 MiB before a run.

---

## 5. Reference-load / two-phase VRAM note (carry-forward)

Encoding itself does not load Gemma alongside the 22B, but the downstream train/sample steps the prep
feeds do — the **two-phase VRAM discipline** (Gemma never coexists with the 22B; free it after the
gate) is why the precomputed `conditions/` matter: training reads **precomputed** text embeddings and
never calls Gemma. Encoding cleanly here is what makes that possible.

---

## 6. LOG the decision + research uncovered knobs (D-8-RESEARCH-LOOP)

Logging a real prep decision is a **required output**, not optional — this is the write-back forcing
function that keeps the log from going stale:

- Append a dated entry to `.planning/harness/DECISION-LOG.md` (four required fields: `what` / `why` /
  `source-tags` / `run-refs`) for any prep decision, preference, or surprise. Do NOT log narration or
  file lists — only things a future session must not re-derive.
- For any **uncovered knob** (a setting with no stored house answer), run the shared research
  protocol verbatim (`.planning/harness/README.md` §Research protocol, D-8-RESEARCH-LOOP):
  1. Check `KNOWLEDGE.md` by keyword FIRST.
  2. If nothing, research the **canonical pinned source** + community docs, then **STORE the finding
     into `KNOWLEDGE.md`** with source tags.
  3. Present options **with explanations** — a `[community]` default is **never a silent override**:
     surface it as *"the common default is X (underlying reason: …); going with your method."*
  4. Auto-log the decision to `DECISION-LOG.md`.

**Typed-state pointer (`.planning/harness/COMPACTION-PROTOCOL.md`):** prep's compaction artifacts
are the staging manifest + the TRAINING-CARD update — dataset/row counts, trim/bucket numbers, and
encode $ land as `name: value` slots (card typed block / `- state:` in the log entry), never prose-only. `[house]`

## 7. ROUND SIZING + RESOLUTION at prep (house corrections, 2026-07-11 — never re-derive)

- **Round unit = 3000 steps** (prior-campaign precedent). NEVER scale the budget down by corpus size on
  priors — round sizing changes what the model learns; ask the operator explicitly, the precedent default
  named. `[house]` `[precedent]`
- **Buckets at native scale** — the largest 32-aligned WxH the clips support. Propose the
  native-scale bucket FIRST; scaled-down buckets only if the operator asks. VRAM headroom is a
  blocks_to_swap problem (mechanical), not a reason to shrink training resolution. `[house]`
- **`blocks_to_swap` is itself a Tier-2 check-in, not an auto-lever** (D-04, overriding SCOPE §3's
  "soft carve-out"). The routing above is correct — VRAM headroom is a `blocks_to_swap` problem,
  not a resolution problem — but `blocks_to_swap` is a FULL Tier-2 carve-out: `draft` may
  **propose** raising it as the sanctioned VRAM lever, but **the operator confirms**; `pro` prefers
  **`0`**, the proven no-swap path, paid for in throughput. Reason: **OFFL-02** long-run adapter
  QUALITY under swapping is still unproven (07-13 validated *freeze-mechanics* only), so
  auto-raising swap to fit resolution is exactly "silently trading quality for fit". Promote to
  Tier 1 only when a long-run quality A/B lands. `[house]` `[precedent]`
- **Silent-resolution-shrink anti-pattern.** If you find yourself about to lower `training_dims` /
  `data.resolution_buckets` / bucket-F to fit VRAM or cut cost, STOP — surface the native-scale
  default and ask, then route the VRAM problem to `blocks_to_swap` (itself a Tier-2 ask) instead.
  `[house]`
- **The resolution check-in is POSTURE-INDEPENDENT — all four quadrants, including pro+yolo**
  (D-18). It is a **VRAM/OOM safety gate, not a cost-direction lever**; pro+yolo can OOM exactly as
  easily as draft+yolo. Framing it as draft-only would wrongly imply pro may move resolution
  freely. **Draft's only new obligation here is negative:** no silent resolution-shrink path may
  ever be introduced. `[house]`
- **Posture at prep (`optimization_target`)** colours default-NAMING direction at data-prep for
  **Tier-1 knobs only** (e.g. `data.num_dataloader_workers`); round sizing, resolution, bucket-F,
  `rank`, and `blocks_to_swap` are **Tier-2 carve-outs that always check in**, and an
  absent/ambiguous value fails to **`draft`** (D-15). Point at the partition by keyword —
  `rg -i "tier|carve-out" .planning/harness/KNOWLEDGE.md` — rather than forking the list. `[house]`
- **validation.prompts carries a SET** (one prompt is never enough for testing). `[house]`
