---
name: segmentation-prep
description: >-
  Drive the SEGMENTATION-CONTROL dataset prep pipeline — property hygiene (neutral codename +
  gitignore + local-only mapping), SAM 3 TEXT-PROMPT auto-seeding (no hand-drawing), fwd/rev
  propagation, EXACT-boundary masks (dilation off), paired image/mask export, house-format captions,
  and the H.264 QA overlay surface. Use this when turning raw clips of a subject into a paired
  segmentation/control training dataset (masks + source + captions) for an EXTERNAL control model —
  NOT an LTX / inpaint run. Rides signet-trainer's `prep/` package seams; the masks are frames, not a
  hardcoded video assumption — modality-extensible.
---

# segmentation-prep — auto-seed, propagate, export a segmentation-control dataset (no hand-drawing)

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


This skill owns the **segmentation-control prepare** stage: turn raw clips of a subject into a paired
**masks + source + captions** dataset for training a SEGMENTATION / CONTROL model, WITHOUT
hand-drawing a single mask. It is a **sibling of `training-prep` / `training-prep-inpaint`** and rides
the same `prep/` seams (`textseed` → `propagate` → export), but its output is a **control dataset for
an external model** — it is NOT an LTX/inpaint encode and it does NOT go through the metered Modal
gate. Nothing here launches a metered run.

> **Routing.** If the dataset is for an LTX LoRA run (latents/conditions through `--mode preprocess`),
> STOP — use `training-prep` (standard) or `training-prep-inpaint` (masked `video_masks`) instead.
> This skill is only for the **standalone segmentation/control dataset** (image+mask pairs, no Modal
> encode). It reuses `training-prep`'s §0 STATE-DETECT discipline and the source-tag preamble; it adds
> the text-seed → exact-mask → paired-export playbook those siblings do not carry.

> **Modality (do NOT hardcode video-only):** the mask surface is a stack of per-frame binary masks
> (`<masks-root>/<mask_stem>/NNNNN.png`, region = WHITE(255)) plus a text-prompted seed frame. Video is
> the current configuration; keep the skill modality-extensible (a single image is the 1-frame subset).

---

## 0. PROPERTY HYGIENE FIRST — the standing rule, before any clip is touched (hard gate)

**Licensed / client material never enters a tracked file, config, commit message, path, or log.** This
is the first move of every run on this pipeline, ahead of any prep step — a leak cannot be un-committed.

- **Stage source under a NEUTRAL CODENAME in a gitignored dir** — `_<codename>_source/` (pick a throwaway
  handle like `_projA_source/`). The codename is arbitrary and carries no meaning; it never encodes the
  real title, character, or studio.
- **Real names live ONLY in `MAPPING.local.txt`** (gitignored) — the codename↔real-name crosswalk stays
  local. Nothing tracked ever needs it.
- **NEVER let a licensed property's real name, character names, or original filenames enter** a tracked
  file, a YAML, a caption that ships, a commit message, or a printed log. Rename source clips to
  `<codename>_NN.<ext>` at staging; the original filename does not survive into the dataset.
- **The concept PROMPT is campaign material.** Pass it at the CLI (`--prompt`), never bake it into a
  tracked config — `configs/prep_textseed.yaml` ships `prompt: null` for exactly this reason. A prompt
  that describes a real character is client-identifying; treat it like the mapping.
- **`.gitignore` is already hardened** for the codename dir families (`_*_source/`, `_*_prep/`,
  `_staging_*/`, and per-campaign codename globs) plus media extensions repo-wide. Verify the codename
  dir actually matches an ignore rule (`git status` shows it untracked-and-ignored) BEFORE the first
  write. `[house]`

> This section is the POINT of the skill's front door. State and enforce the RULE; never write an
> instance of a real property into any artifact this skill produces.

---

## 1. STATE-DETECT + source-tag preamble (reuse `training-prep` §0 / §1 discipline)

Before improvising, retrieve the standing answer — an untagged model prior is the documented drift
failure mode:

1. **Read the memory scaffold** (`.planning/harness/`): `KNOWLEDGE.md` + `DECISION-LOG.md`. Search by
   bare keyword (`rg -i "textseed|segmentation|sam3|fwd|rev|codename" .planning/harness/`).
2. **Detect real pipeline state by artifact** — do NOT assume; each stage is detectable on disk:
   seeds (`seeds/<stem>__<type>.png` + `textseed_records.json` + `manifest.txt`) → propagated
   (`masks_video/<stem>/NNNNN.png` + `overlays_video/*_overlay.mp4`) → exported dataset
   (`<out>/images/`, `<out>/masks/`, `<out>/manifest.jsonl`, `summary.json`). **Name the stage; never
   re-run a completed stage** (nothing here deletes, so a re-run only wastes GPU time).
3. **Source tags are mandatory.** Every recommendation you log carries `[house]` · `[precedent]` ·
   `[canonical]` · `[community]`. Untagged = a violation.
4. **Captions are philosophy territory** (they change what the model learns) — house rules first, ask
   when uncovered. Klippbok is the caption gold standard; never hand-roll a caption linter. Absent
   tooling → mark caption steps **`[PENDING: operator]`**.

**The GPU env is `.venv-sam/Scripts/python.exe`** (torch 2.11 cu128, transformers 5.14.1, cv2, sam2,
SAM 3 weights cached). Use it for the model steps (§2 seed, §3 propagate). Do **NOT** up/downgrade its
deps — the standalone-sam3 install silently forces a numpy downgrade (the fix was uninstall + restore).
Every invocation carries `PYTHONPATH=src PYTHONUTF8=1` (Windows UTF-8 + src layout).

---

## 2. TEXT-PROMPT AUTO-SEED — SAM 3 concept head replaces hand-drawing (`prep.textseed`)

SAM 3's concept-segmentation head (`Sam3Model` + `Sam3Processor`, `text=`) turns a natural-language
concept into the subject mask at whatever frame it appears. It **probes several evenly-spaced frames
(both endpoints ALWAYS), scores the concept detection on each, picks the best inside the coverage
band, and decides the propagation direction** — then writes a binary WHITE(255) seed PNG
byte-compatible with `propagate.load_seed`, so the entire downstream propagation seam is reused verbatim.

```
PYTHONPATH=src PYTHONUTF8=1 .venv-sam/Scripts/python.exe scripts/prep_textseed.py \
    --clips-dir _<codename>_source/ --seeds-out _<codename>_prep/seeds \
    --prompt "<subject-concept>" --mask-type full_body
```

- **`--prompt "<subject-concept>"` is REQUIRED for a real run** and is campaign material — pass it at
  the CLI, NEVER a tracked config (use a generic toy concept in any example, e.g. `"a red fire
  hydrant"` / `"the person in the blue coat"`; never a phrase that matches a real character).
- **`--dry-run` self-checks with NO model load** (counts frames, prints the probe plan, writes
  nothing) — run it first to confirm every clip decodes and the probe indices look right.
- **Every threshold is config-first** (`configs/prep_textseed.yaml`, fail-loud): `detect.threshold` /
  `detect.mask_threshold`; `seed.n_probes` (default 5), `seed.min_score` (0.30), `seed.weak_score`
  (0.15), `seed.min_coverage` (0.002, rejects specks), `seed.max_coverage` (0.90, rejects a runaway
  whole-frame mask), `seed.start_margin` (0.10). CLI overrides exist (`--n-probes`, `--min-score`,
  `--mask-type`, `--device`) but the YAML is the source of truth — never hardcode a threshold in code.
- **Outputs** (into `--seeds-out` and its parent): `<clip_stem>__<mask_type>.png` binary seeds,
  `textseed_records.json` (per-clip `{seed_frame_idx, direction, score, coverage, probes}` — the
  evidence trail), and `manifest.txt` (propagate's `NN | <clip_stem>.png | <clip>` stem→clip map).
- **A NOT-SEEDABLE clip is reported, never silently emitted as an empty seed** — the run prints
  `SUMMARY … not_found=[…]` and `TEXTSEED_DONE`. `probe_hits` distinguishes "subject absent at both
  endpoints" from "subject absent from the whole clip". `[canonical]`

---

## 3. FRAME-0-ABSENT HANDLING — seed the LAST frame, propagate BACKWARD (fwd/rev, D-08 reuse)

Many clips do not contain the subject at frame 0 (an empty room, a different character). The
propagation seam can only seed the frame it feeds the predictor first — physically **frame 0 (`fwd`)**
or **frame n-1 (`rev`)**. `textseed.decide_seed` handles this automatically (house rule — do NOT
reinvent):

- prefers **`fwd`** whenever frame 0 carries a valid detection (the well-trodden path);
- switches to **`rev`** (seed last frame, propagate backward) when the last frame is the only valid
  endpoint, or beats frame 0 by more than `seed.start_margin`;
- falls back to a **flagged WEAK** endpoint (`seed.weak_score`) when neither endpoint clears
  `min_score` but mid-clip probes prove the subject is present.

The `rev` decision flows straight into propagation: `textseed.rev_stems(records)` yields exactly the
stems to pass as `--rev` in §4. The run prints `rev stems: [...]` for that hand-off. Do **not**
hand-pick directions — read them off the records.

---

## 4. PROPAGATE — reuse the existing seam with the SEGMENTATION spec (dilation OFF)

Propagate each seed across its clip via the committed `prep.propagate` (transformers SAM 3 backend —
the standalone package is triton-broken on Windows; SAM 2.1 is the last-ditch fallback only):

```
PYTHONPATH=src PYTHONUTF8=1 .venv-sam/Scripts/python.exe scripts/prep_inpaint_propagate.py \
    --backend sam3 --device cuda \
    --masks-dir _<codename>_prep/seeds --clips-dir _<codename>_source/ \
    --masks-out _<codename>_prep/masks_video --overlays-out _<codename>_prep/overlays_video \
    --spec src/signet_trainer/harness_data/MASK-SPEC-segmentation.yaml \
    [--rev <mask_stem> ...] [--only <mask_stem>] [--dry-run]
```

- **Use `MASK-SPEC-segmentation.yaml`, NOT the inpaint `MASK-SPEC.yaml`.** For a segmentation dataset
  the masks ARE the training target, so they must be the EXACT object boundary. That spec sets every
  class `dilation.max_margin_px: 0` (with `grow_to_target: false`), so `dilate_to_target` is a
  **verified no-op** — config-first, no code branch needed. The inpaint spec deliberately over-grows
  masks (that helped inpaint likeness); using it here would corrupt the boundary. `target` remains a
  REPORTING band only. `[house]` `[canonical]`
- **Pass the `rev` stems from §3** (`--rev <mask_stem>`, repeatable) so drifted / frame-0-absent clips
  seed on the last frame and propagate backward. `--only <mask_stem>` limits a re-run; `--force`
  re-propagates over existing PNGs (default skips them — nothing is deleted).
- **`--dry-run` resolves stems + seed dims with NO model load.** Real runs offload video+state to CPU
  (`--storage-device cpu --state-device cpu`, the defaults) for big clips, print per-clip `avg_cover`,
  emit the per-frame WHITE PNGs + the green-overlay QA mp4s, and end on `SAM_PROPAGATE_DONE`.
- **Stale-VRAM discipline before a GPU run:** `pkill python3` (targeted PIDs only) + verify 0 MiB.
  Never blanket-kill python.exe.

---

## 5. EXPORT — pair masks with source frames into a verified dataset (`prep_segdataset.py`)

Pair every propagated mask with its source frame and emit the trainer-agnostic layout, verifying each
pair before it lands:

```
PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_segdataset.py \
    --masks-root _<codename>_prep/masks_video --clips-dir _<codename>_source/ \
    --manifest _<codename>_prep/manifest.txt --records _<codename>_prep/textseed_records.json \
    --out _<codename>_prep/dataset [--stride N] [--skip-empty]
```

- **Layout produced:** `<out>/images/<codename>_<frame:05d>.png`,
  `<out>/masks/<codename>_<frame:05d>.png`, `<out>/manifest.jsonl` (one JSON object per pair, carrying
  `image` / `mask` / `clip_stem` / `mask_type` / `frame` / dims / `mask_px` / `coverage` / the seed
  metadata), plus `summary.json` (per-clip pairs/empty/rejected/avg_coverage).
- **Every pair is VERIFIED or REJECTED:** the mask must be strictly two-valued `{0,255}` AND
  dimension-identical to its image — a silently mis-sized or grey-valued mask is exactly the failure a
  segmentation trainer cannot detect. Rejects are counted (`rejected=…`), never silently written; a
  non-zero reject count is the process exit code. `[canonical]`
- **Empty-mask frames are KEPT and flagged (`mask_px: 0`) by default** — a segmentation dataset
  legitimately contains negative frames. `--skip-empty` drops them; `--stride N` keeps every Nth frame.
- **Ends on `SEGDATASET_DONE`.** Pure local decode + write — no model, no Modal, never deletes.
- **ProRes / non-÷64 source is FINE here.** cv2 decodes ProRes; source dims may not be divisible by 64.
  The ÷64 / 8n+1 rules are LTX-ENCODE constraints — **N/A to a pure segmentation dataset**. Do NOT
  force-crop for segmentation (a crop changes content for no gain).
### 5b. ALTERNATE EXPORT — paired control/train `.mov` layout (`--paired-out`, a CURRENT mode)

Some external control trainers instead want a **paired, numbered `.mov`** layout: a `control_data/` of
numbered clips whose frames are a white-silhouette-on-black mask **video**, alongside a `train_data/`
of the numbered **source** clips + a per-clip caption `.txt`. The same `prep_segdataset.py` emits this
directly — pass `--paired-out` **instead of** (or alongside) `--out`:

```
PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_segdataset.py \
    --masks-root _<codename>_prep/masks_video --clips-dir _<codename>_source/ \
    --manifest _<codename>_prep/manifest.txt \
    --paired-out _<codename>_prep/paired \
    --captions-dir _<codename>_captions/ --keep 01,02,03
```

- **Layout produced (one numbered clip per KEPT source clip):**
  `<paired-out>/control_data/NNN.mov` (ProRes mask video), `<paired-out>/train_data/NNN.mov`
  (byte-copy of the source clip), `<paired-out>/train_data/NNN.txt` (the caption), and
  `<paired-out>/MAPPING.local.txt` (the `NNN <- <clip_stem>` crosswalk).
- **Numbering:** kept clips are numbered `001, 002, …` in **sorted order over the kept clips**.
  `--keep` is an optional comma-separated selector accepting clip_stem parts OR their trailing numbers
  (e.g. `--keep 01,02,03,13,14,15,16` keeps those seven, renumbered `001..007`); omit `--keep` to keep
  all. This replaces any by-hand subset copy.
- **Control mask video** = ProRes from the per-frame binary mask PNGs
  (`ffmpeg … -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le`) — the proven profile that preserves the
  binary `{0,255}` edges (0 mid-tone leakage). fps comes from the source clip (`--fps` overrides;
  fallback 24). **Train video** = `shutil.copy2` of the source clip verbatim — source is ProRes already,
  so **no re-encode, no quality loss**. `[canonical]`
- **Frame-count parity is a HARD gate.** The control mask video's frame count (= number of `NNNNN.png`
  in the mask dir) MUST equal the source clip's frame count; a mismatch **FAILS that clip loudly with
  both counts and does NOT encode** — a control/train frame desync silently corrupts training. Never
  pad or trim. Unresolved source, missing masks, and ffmpeg failure are likewise hard per-clip failures;
  the run exits nonzero if any clip failed. Ends on `PAIRED_DONE`. `[house]`
- **Captions** resolve from `--captions-dir/<clip_stem>.txt` first, else a `.txt` beside the source
  clip; if neither exists the `.txt` is skipped with a WARN (not a hard failure). The final house-format
  captions live in the dir you point `--captions-dir` at (§6 authors them).
- **`<paired-out>` MUST be a local / git-ignored path** (e.g. under `_<codename>_prep/`, already ignored
  via `_*_prep/` + `*.mov`). It byte-copies the source clips and writes a real-name crosswalk
  (`MAPPING.local.txt`, `.local` keeps it out of git) — **never point it inside a tracked repo dir.**
  The captions it copies may carry the subject's natural name as an anchor — that is fine: it is OUTPUT
  data going to a local path, never committed. `[house]`

---

## 6. CAPTIONS — house format, strip the subject's APPEARANCE (only if the trainer wants them)

If the downstream control trainer consumes captions, match the house format and obey the Klippbok
appearance rule. Captions are **philosophy** — house-first, ask when uncovered:

- **One line**, sectioned: `Foreground:` / `Midground:` / `Background:` / optional `Prop
  interaction:` / `Emotion:` / `Dynamic Motion:`, ~90 words, **name-anchored** (a natural name, never
  a trigger token — trigger tokens are a myth for flow-match models, banned).
- **CRITICALLY strip the subject's physical appearance** (Klippbok rule: appearance is learned from
  pixels, not text). KEEP scenery, other UNNAMED characters, and camera / FOV. Drop hair/eye/clothing
  descriptors of the anchored subject.
- **Any example you write must be fully INVENTED and generic** — a generic subject in a generic
  setting. Never reproduce a real caption, and never let a real property/character/title appear.
  Example shape only: `Foreground: <Name> steps toward camera, mid-shot. Background: a lamp-lit
  hallway. Emotion: wary. Dynamic Motion: a slow dolly-in.`
- If Klippbok tooling is absent, mark the caption-audit step **`[PENDING: operator]`** — never hand-roll a
  linter. `[house]`

---

## 7. QA SURFACE — H.264 overlays, served live + tunnelled (never just a file path)

**LANDMINE this closes:** `cv2.VideoWriter` writes mp4v (MPEG-4 Part 2) by default, which **no browser
can decode** — a grid of mp4v overlays renders as black rectangles and looks like a pipeline failure.
Every video handed to the operator must be **H.264 / yuv420p / `+faststart`**:

```
PYTHONPATH=src PYTHONUTF8=1 python scripts/qa_overlay_h264.py \
    --in-dir _<codename>_prep/overlays_video --out-dir _<codename>_prep/qa_grid
```

- Transcodes each `<stem>_overlay.mp4` → `<out>/<stem>/step_0.mp4` (H.264, gridwatch's
  `{prompt}/step_{step}.mp4` layout) via `ffmpeg -c:v libx264 -pix_fmt yuv420p -movflags +faststart`,
  probes the output codec, and ends on `QA_H264_DONE`. Requires ffmpeg on PATH; never deletes an input.
- **Serve it LIVE + tunnel** (house rule — never hand the operator a bare file path). finetune-gridwatch is
  the DEFAULT grid tool (never hand-rolled):
  ```
  _tools/finetune-gridwatch/.venv/Scripts/grid.exe watch _<codename>_prep/qa_grid \
      -o _<codename>_prep/qa_out --no-open --port <port>
  ```
  Serve **OS-detached** (`Start-Process`, so it survives the turn), **verify HTTP 200** after launch,
  then relay to the tailnet: `python scripts/_tailnet_relay.py <tailnet-ip> <port>`. Report **both**
  the `127.0.0.1` localhost URL (primary) and the tailnet URL.
- **Many-clip QA layout:** gridwatch lays out steps × prompts, so a one-step-many-clips set makes a
  tall single column. For many-clip segmentation QA prefer a **contact-sheet grid** over that column.

---

## 8. LANDMINES the segmentation-prep skill MUST carry

- **Property hygiene is non-negotiable (§0).** Codename + gitignore + local-only `MAPPING.local.txt`;
  no real name / character / original filename / identifying prompt in ANY tracked file, path, log, or
  commit message. `[house]`
- **`.venv-sam` deps are FROZEN.** `.venv-sam/Scripts/python.exe` is the GPU env (torch 2.11 cu128,
  transformers 5.14.1, cv2, sam2, SAM 3 weights cached). Do NOT up/downgrade — the standalone-sam3
  install silently forces a numpy downgrade (fix was uninstall + restore). `[precedent]`
- **Segmentation spec, dilation OFF.** Propagate with `MASK-SPEC-segmentation.yaml`
  (`max_margin_px: 0`), never the inpaint `MASK-SPEC.yaml` — exact boundary is the whole point. `[house]`
- **The concept prompt is CLI-only campaign material** — never in a tracked config; use a generic toy
  concept in examples. `[house]`
- **mp4v is browser-unplayable — always re-encode overlays to H.264** before serving. `[precedent]`
- **÷64 / 8n+1 are LTX-encode rules, N/A to segmentation** — do NOT force-crop a segmentation dataset;
  ProRes non-÷64 source is fine. `[house]`
- **Never blanket-kill python.exe** (targeted PIDs only). **Never delete intermediates** (seeds,
  masks, overlays, grids — they're the recoverable state). **Stale-VRAM discipline** (`pkill python3`
  + verify 0 MiB) before any GPU run. `[house]`
- **Not a metered path.** No Modal, no `--mode preprocess`, no GPU gate — this dataset is for an
  external model. If the plan drifts toward an LTX encode, route to `training-prep` instead. `[house]`

---

## 9. LOG the decision — required output (D-8-RESEARCH-LOOP)

Logging a real segmentation-prep decision is a **required output**, not optional:

- Append a dated entry to `.planning/harness/DECISION-LOG.md` (four fields: `what` / `why` /
  `source-tags` / `run-refs`) for any prompt/threshold/direction/spec/export decision or surprise. Do
  NOT log a real prompt or property name — log the codename and the abstract decision.
- Any **uncovered knob** (no stored house answer) runs the shared research protocol
  (`.planning/harness/README.md` §Research protocol): KNOWLEDGE.md by keyword FIRST → canonical +
  community → STORE into KNOWLEDGE.md with source tags → present options with explanations (a
  `[community]` default is never a silent override) → auto-log to DECISION-LOG.
- **Typed-state pointer** (`.planning/harness/COMPACTION-PROTOCOL.md`): load-bearing numerics (clip /
  pair / seeded / rejected counts, avg_cover, fwd/rev split) land as `name: value` slots, never
  prose-only. `[house]`
