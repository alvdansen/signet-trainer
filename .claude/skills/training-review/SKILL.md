---
name: training-review
description: >-
  Drive the gated `--mode sample` render of a trained LoRA, fetch the samples with `modal volume
  get`, build a headless `finetune-gridwatch` grid, run the ~200-step base-vs-checkpoint convergence
  probe (samples-not-loss), and surface clips to the operator for judgment. Use this after a training run
  produces a checkpoint, or to convergence-check a run at ~200 steps. Always show CLIPS, never a
  first frame. Modality-extensible — video is the current configuration, not a hardcoded assumption.
---

# training-review — sample → gridwatch grid → convergence → surface

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


This skill owns the **review** stage of the harness loop:
`… → gated train launch → monitor → sample (this skill) → grid → surface for approval → iterate`.
It renders the **trained adapter**, builds the grid with **the first-party `finetune-gridwatch`** (the
canonical sampler harness — not a parallel grid), runs the convergence probe, and surfaces clips.

> **Modality (do NOT hardcode video-only):** rules below are stated against the *current* video
> configuration. `finetune-gridwatch` renders video (.mp4/.webm) **or** images — keep the skill
> modality-agnostic (do not hardcode `.mp4` as if it were the only case).

---

## 0. STATE-DETECT preamble (D-8-STATEROUTE) — retrieve before improvising

Every session, before rendering anything:

1. **Read the memory scaffold** (`.planning/harness/`): `KNOWLEDGE.md` + `DECISION-LOG.md`. Search by
   bare keyword (`rg -i "convergence|two-phase|load-order|OFFL-02"`). Retrieve the standing answer
   before improvising.
2. **Detect the latest checkpoint** — do NOT glob by hand. The harness writes **per-STEP checkpoint
   dirs** and resolves the newest via `CheckpointManager.find_latest()` (single source of truth):
   ```
   modal volume ls signe-trainer-checkpoints <output_dir>
   ```
   Confirm a `checkpoint-step-*` dir exists. If none → hand off to the run skill (nothing trained yet).
3. **Confirm the sample renders the TRAINED adapter (Pitfall 5 / D-8-REFLOAD).** After Plan 08-02, the
   sample branches load the latest trained adapter before rendering. In the `--mode sample` log,
   confirm the line **`loaded trained adapter …`**. If it instead says **`rendering on the BASE
   transformer (wiring mode)`**, STOP and flag it — any verdict off that grid would validate the base
   model, not the LoRA (a wrong verdict from a metered run). Do not surface a base-model grid as a
   convergence result.

---

## 1. SAMPLE — drive `--mode sample` through the gate (always default to samples)

**Always default to generating samples** (never-forget — eval is by samples, §3). Drive the render
through the **same gate** as every metered run:

```
PYTHONPATH=src PYTHONUTF8=1 modal run --detach -m signet_trainer.modal.entrypoint \
    --config <yaml> --mode sample [--approve]
```

- Gate order is **load → dry-run → cost print → BLOCKING approval → dispatch**; the **cost print +
  approval precede dispatch**. Under an active blanket the stop is removed but the cost line still logs.
- **Declare the render's tier before dispatch (§11, D-24).** For a **`verdict`** tier, run
  `assert-verdict-spec --config <yaml> --render-tier verdict` FIRST and **do not dispatch on a
  non-zero exit** — a trimmed verdict render wastes a metered A100 AND produces a misleading verdict,
  so the local assertion runs before spend. A `quicklook` tier skips the assertion. `[house]`
- The render loads the LTX-2.3 components with the **video VAE decoder ON**, renders base-vs-LoRA (+
  the reference grids for single_frame / multi_frame / ic_lora), and commits the samples to
  `signe-trainer-checkpoints` under `<output_dir>/samples*/` (commit-or-vanish).

---

## 2. GRID — build the `finetune-gridwatch` grid headless (do NOT build a parallel grid)

`finetune-gridwatch` is **the canonical house sampler-grid harness** (installed via
`scripts/setup_gridwatch.sh` — pinned SHA, isolated venv, never in the Modal image). Use it; do not
hand-roll a competing grid. The in-Modal `grid.py` montage stays **complementary**, not replaced.

1. **Fetch** the committed samples to a local dir. §1 commits under the per-mode
   `<output_dir>/samples*/` glob, so fetch the mode-specific dir — e.g. for an
   `ic_lora` run:
   ```
   modal volume get signe-trainer-checkpoints <output_dir>/samples_ic_lora ./_samples
   ```
   The exact dir names `fns.py` writes, by mode (WR-08 — do NOT guess
   `samples_text_to_video`; it is never written):
   - **default base-vs-LoRA grid** → `samples/` (NOT `samples_text_to_video`)
   - **single_frame** → `samples_single_frame/`
   - **multi_frame** → `samples_multi_frame/`
   - **ic_lora** (dev re-skin) → `samples_ic_lora/`
   - **ic_lora known-good baseline** (distilled two-stage) → `samples_ic_lora_baseline/`

   `modal volume ls signe-trainer-checkpoints <output_dir>` shows which `samples*`
   dir the run actually committed.
2. **Build headless** (no browser auto-open — the agent runs this, not the operator):
   ```
   grid build ./_samples --no-open --template "{prompt}/step_{step}_seed{seed}.mp4"
   # -> ./_samples/grid-output/index.html
   ```
   Adjust the template extension for the modality (`.mp4`/`.webm`/image) — do not assume `.mp4`.
3. **Freeze** for a shareable, server-free bundle when handing back to the operator:
   ```
   grid freeze ./_samples
   ```
4. Invoke the grid binary from the isolated venv the setup script created
   (`_tools/finetune-gridwatch/.venv/bin/grid`), never the trainer env.

---

## 2b. MiniMax-H3 renders — the SAME `--mode sample`, routed by FAMILY (Phase 10, H3-06)

A config carrying `family: h3` makes `--mode sample` dispatch `h3_sample` instead of the LTX
`sample`. No new mode, no new command, same gate:

```
PYTHONPATH=src PYTHONUTF8=1 modal run --detach -m signet_trainer.modal.entrypoint \
    --config configs/<your-h3-run>_sample.yaml --mode sample [--approve]
```

**The automated acceptance floor: `max|delta velocity|`.** Before rendering anything, `h3_sample`
runs the base and adapted forwards on ONE fixed batch and reports
`max|delta velocity| base-vs-adapter`. **A delta of exactly `0.0` RAISES** — two identical columns
with different labels are worse than no grid at all. This floor stands whether or not anyone has
looked at the grid yet, and it is measured BEFORE the renders so a dead adapter aborts cheap.

**D-10-SCOPEGUARD, stated plainly when surfacing:** *Phase 10 does not grade adapter quality.* A run
whose loop completes and whose adapter demonstrably moves the model **PASSES even if the video looks
bad.** Do not surface an H3 Phase-10 grid as a quality verdict, and do not let a bad-looking render
be read as a failed phase — the deliverable is the working tool.

**What the render is and is not:** base-vs-adapter comes from ONE PEFT-wrapped transformer switched
with `disable_adapter()` (61.7 GiB twice does not fit), so "identical seed, identical everything
except the adapter" is literally true. H3 is **guidance-distilled and single-pass**: no guidance
scale, no negative branch, no STG, no two-stage upscaler. The gallery banner prints
`n/a (H3 is guidance-distilled)` for those fields and that is the truthful report — do not paste an
LTX guidance value into an H3 render config expecting it to do something.

**Artifacts** land under `<output_dir>/samples_h3/<timestamp>/` — `base/`, `lora/`, `index.html`, and
`delta.json` (the measured delta, the `find_latest`-resolved checkpoint path, the seed, the reference
filenames). Fetch with `modal volume get signe-trainer-checkpoints <output_dir>/samples_h3 ./_samples`.

**The standing review rules apply unchanged, and they are not optional here:**

- **Build the grid with `finetune-gridwatch`** (§2) — never hand-rolled. `index.html` is the
  artifact index this stage writes; it is **not** a replacement for the gridwatch grid.
- **Serve it live and TUNNEL it** (§10) — `grid watch` + a tunnel URL. Never hand back a file path.
- **Show the CLIP, never a first frame** (§4).

⚠ **References come from manifest row 0.** Every prompt is conditioned on the reference slots the
seeded rotation assigns to row 0, in the same order the pre-encode used — so the render asks the
question the adapter was taught. There is no per-eval-prompt reference surface in the schema; wanting
one is a schema addition and a first-class setup question, not something to default.

---

## 3. CONVERGENCE PROBE (D-8-CONVERGENCE) — samples, not loss

The convergence check is a **~200-step base-vs-checkpoint** comparison on a **matched, seed-locked
prompt** — not a step threshold, not a loss threshold:

- **REUSE the `sample` fn** at the ~200-step checkpoint rather than building a dedicated probe — it
  already renders base-vs-LoRA and already does the two-phase VRAM load. Point it at the
  `checkpoint-step-200`-ish checkpoint (whatever `find_latest` resolves at that point) with a
  seed-locked matched prompt.
- **SAMPLES not loss.** Loss is **sanity-only** — watch it only for NaN or no-descent. A descending
  loss is **never** a success verdict. The evidence hierarchy is the rendered clips.
- **Default = render + surface; the judgment is the operator's.** Do not auto-declare pass/fail. Only if
  yolo mode is implemented **with a judge** may yolo act on the probe verdict — the default (1) is
  render + surface.
- **The ~200-step probe IS a verdict (D-25) — declared `--render-tier verdict` and locked at the
  house spec.** A real go/no-go call rides on it (continue vs restart the recipe) and that decision
  costs rounds; a trimmed probe that misleads at step 200 is the **most expensive possible false
  economy**. **Draft may trim it in NO way — not its params, and not its prompt count** (a thinner
  prompt set is the eval-composition trap; cross-ref §8 PROMPT SET: one prompt is never a valid
  verdict basis). Run `assert-verdict-spec` before this dispatch like any other verdict (§11). `[house]`

---

## 4. SURFACE (D-8-SURFACE) — show clips, never a first frame

- **Surface on milestones + blockers only** — needs-approval, unhandleable failure, the
  convergence-probe result, completion with the grid. Everything else logs **quietly** ("do the work,
  don't narrate").
- **Show the CLIP / hand back the grid path** — **NEVER review a video by its first frame** (Pitfall
  6). A first-frame thumbnail hides temporal corruption/warp; surface the playable clip or the frozen
  grid bundle.
- **Honor the operator's configured quiet hours** — no nudging, dry/direct tone, no guilt language.
  Batch non-urgent surfacing outside quiet hours.
- **Easy success ≠ validated (KNOWLEDGE `test-difficulty`).** When judging performance, a good result
  on easy subjects (realism/3D/full-face portraits) is the FLOOR, not the finish line — do not surface
  "looks great / done" off the low-hanging fruit; check the HARD cases (angle-variant clothing,
  linework, difficult motion) before calling settings/quality validated. A failure on an EASY subject
  is a strong TRUE-negative signal worth flagging.
- **Carry BOTH honest scope limits with every verdict surface** — the offloader's
  freeze-mechanics-only validation (**§5 SCOPE LIMIT**) AND the single-stage render disclosure
  (**§12 SINGLE-STAGE DISCLOSURE**). State them when surfacing a verdict grid, not just at the gate.

---

## 5. SCOPE LIMIT (D-8-OFFL02-DEFER) — record honestly

**No run has ever sampled under active block-swap.** The offloader is PEFT-aware but validated for
**freeze-mechanics only** (07-13); a validation sample under *active* block-swap (OFFL-02) was never
behaviourally run, and long-run adapter-**quality** under swapping is **UNPROVEN**. When surfacing a
verdict, state this scope limit — do not imply the offloader is blanket-trusted.

---

## 6. LANDMINES the review skill MUST carry

- **Two-phase VRAM load** — Gemma (12B) must **NEVER coexist** with the 22B transformer on the
  A100-80GB. The sample path pre-encodes prompts in PHASE A, deletes Gemma, then loads the 22B with
  `with_text_encoder=False` and feeds cached `CachedPromptEmbeddings` under `torch.no_grad()`. Proven
  profile: gate peak ~62.91 GiB → ~38.43 GiB floor once Gemma is freed. Getting this wrong OOMs
  (runs 3–7).
- **Sample-inference load order** — `load_embeddings_processor(checkpoint_path=…, device="cpu")` must
  receive the checkpoint path and load on **CPU** (device="cuda" drags ~50 GB of checkpoint residency
  onto the GPU alongside Gemma). The adapter lives in **per-STEP checkpoint dirs** resolved via
  `CheckpointManager.find_latest()`, not a flat file — getting this wrong renders the BASE, not the
  adapter (§0.3).
- **Single A100-80GB**, no fallback GPU lists, no warm containers — even in yolo mode.

---

## 7. LOG the verdict + research uncovered knobs (D-8-RESEARCH-LOOP)

Logging the review verdict is a **required output**, not optional (the write-back forcing function):

- Append a dated entry to `.planning/harness/DECISION-LOG.md` (four required fields: `what` / `why` /
  `source-tags` / `run-refs`) recording the convergence/review verdict — with the checkpoint path /
  run ref it points at. Do NOT log narration or transcripts.
- For any **uncovered knob**, run the shared research protocol verbatim
  (`.planning/harness/README.md` §Research protocol, D-8-RESEARCH-LOOP):
  1. Check `KNOWLEDGE.md` by keyword FIRST.
  2. If nothing, research the **canonical pinned source** + community docs, then **STORE** the finding
     into `KNOWLEDGE.md` with source tags.
  3. Present options **with explanations** — a `[community]` default is **never a silent override**
     (surface as *"the common default is X (underlying reason: …); going with your method"*).
  4. Auto-log the decision to `DECISION-LOG.md`.

**Typed-state pointer (`.planning/harness/COMPACTION-PROTOCOL.md`):** verdicts are compaction
artifacts — the DECISION-LOG verdict entry carries a `- state:` block (checkpoint path, steps,
loss, $) and the TRAINING-CARD round/verdict rows update the card's typed slots, never prose-only. `[house]`

**Emitter step (D-TS-4 — explicit skill step, no hook).** After recording the verdict, update the
round's `verdict` / `checkpoint_judged` slots in `<card>.state.yaml`, then regenerate the emitted
TYPED-STATE blocks — a REQUIRED step:

```
PYTHONPATH=src PYTHONUTF8=1 python -m signet_trainer.harness_state emit
```

A `verdict`-class DECISION-LOG entry declares `class: verdict` in its `- state:` block and carries
the `{round, checkpoint_judged, verdict}` core (D-TS-2). Before committing, run the backstops:
`python -m signet_trainer.harness_state check` (block-vs-source drift) and
`python -m signet_trainer.harness_lint` (log lint). `[house]`

## 8. PROMPT SET rule (house rule, 2026-07-11)

One prompt is never enough for testing. Review renders (`--mode sample`) use the config's full
prompt SET (varied framing/setting, Klippbok grammar); a single-prompt grid is not a valid
verdict basis. `[house]`

## 9. PARALLEL INFERENCE venue (OPT-IN, not default — house rule, 2026-07-11)

When the session-setup gate picked **parallel** as the sampling venue (prior-campaign precedent),
run `python scripts/watch_parallel_inference.py <render_config.yaml>` locally: it polls the run's
checkpoints and dispatches ONE gated `--mode sample` render per NEW checkpoint on a SECOND
single-A100 container while training continues (single-A100 holds PER JOB). Rules:
- The render config is a COPY of the training config with `est_hours` set to an honest
  per-render figure (~0.5h) so the ledger doesn't inherit the multi-hour train estimate.
- The dispatch ALSO declares its tier via `--render-tier` (§11, D-24). In-loop watcher renders
  are **exactly the ones the operator judges on the grid** — so they are **verdicts when a decision rides
  on them**, NOT quicklooks by virtue of being in-loop. This is D-24's rejection of by-venue tiering
  made concrete at the venue: assert the house spec before a verdict dispatch here too. `[house]`
- Every dispatch still goes through the entrypoint gate (`--approve` under the session
  pre-auth) and appends to the SESSION-STATE spend ledger; the gridwatch grid rebuilds per
  render (`grid-output/index.html` updates live).
- NOT the default venue — it doubles concurrent GPU spend; it is chosen explicitly at the
  training-session-setup gate. `[house]` `[precedent]` (prior parallel campaigns)

## 10. ALWAYS serve + tunnel the grid (house rule, 2026-07-11)

After the first grid build of a session, ALWAYS start the live view — `grid watch <grid_dir>`
from the isolated venv (localhost live-reload) — and open a tunnel (cloudflared/ngrok if in
PATH) so the operator can view from any device. Post the URL; never hand back only a file path. `[house]`

§10 tunnel mechanics: use `scripts/serve_gridwatch.ps1` — serves `grid watch` and auto-tunnels
via ngrok (preferred, if in PATH + authtoken) or cloudflared quick tunnel; falls back to
localhost with an install hint. ngrok is scaffolded-not-prioritized (house note, 2026-07-11).

## 11. RENDER TIERS (D-24/D-31) — declared at the call site

Every render dispatch declares its tier — **`quicklook`** or **`verdict`**. The tier is declared at
the CALL SITE, by PURPOSE.

- **Tiers are defined by PURPOSE, not by venue.** A render is a **verdict because a decision rides on
  it** — not because of where or how it was launched. By-venue tiering was REJECTED: it would
  mislabel the §9 parallel-watcher in-loop renders, which are exactly the ones the operator judges
  on the grid, as non-verdicts. (Separate config blocks per tier were also rejected — param duplication +
  migration cost.) `[house]`
- **`quicklook`** — a glance; **free for the mode to cheapen** (Tier 1: quicklook render fidelity). No
  decision rides on it.
- **`verdict`** — **locked at the house spec and NOT a mode lever.** A verdict dispatch **asserts the
  house spec and REJECTS trimmed params** before spending. Neither `draft` nor `pro` may move a
  verdict render. `[house]`
- **The tier is a DISPATCH ARGUMENT, never a config key.** `schema.py::_Base` sets
  `model_config = ConfigDict(extra="forbid")`, so a `render_tier:` key written into a render-config
  YAML would be **rejected at config load** — and `schema.py` is zero-edit. This is why the tier does
  **NOT** follow the `est_hours` override convention (§9): `est_hours` works only because it is a real
  schema field (`modal.est_hours`); `render_tier` is not, so it stays on the CLI. `[canonical]`
- **The assertion command**, run BEFORE a `verdict` dispatch and never dispatched-past on a non-zero
  exit:
  ```
  PYTHONPATH=src PYTHONUTF8=1 python -m signet_trainer.harness_state \
      assert-verdict-spec --config <render.yaml> --render-tier verdict
  ```
  It reads `src/signet_trainer/harness_data/HOUSE-SPEC.yaml` (`verdict_spec`: 81f / 40 steps / guidance 4.0 /
  seed 42 / single-stage / a prompt-SET floor) and **exits non-zero listing every trimmed param**. It
  runs locally, before any metered spend. **`--render-tier` defaults to `verdict`** — the fail-safe
  direction: an undeclared tier gets the strict check, never a wave-through. `[house]`
- **No verdict path reads `optimization_target`** (D-24). The posture has no say in eval judgment; the
  verdict tier is locked. A guardrail test (Plan 07) proves `assert_verdict_spec` never reads the
  posture, and text-scans these verdict sections. `[house]`

## 12. SINGLE-STAGE DISCLOSURE (D-26/D-27) — record honestly

The verdict half of D-27's two-place disclosure. D-27 names it at the setup gate as the house default
(Plan 04, landed) **and restates it on every verdict-grid surface** — the moment it actually matters,
and the moment a compacted session may have lost the setup context.

- **When surfacing a verdict grid, state that the render is single-stage** — and that **two-stage can
  misrepresent LoRA convergence**: the upscale stage **may not fully respect the LoRA's convergence**
  and can show detail the adapter did not actually learn (house rule, 2026-07-16). Do NOT imply the grid
  understates the adapter. Point at the canonical wording in
  `src/signet_trainer/harness_data/HOUSE-SPEC.yaml` (`disclosures.single_stage`) rather than forking the sentence.
  `[house]`
- **`two_stage_upscale` is OFF by default, POSTURE-INDEPENDENT, and is NOT a draft/pro lever** (D-26).
  It is an **eval-honesty rule** on the learns/eval side of D-8-BOUNDARY, not a cost knob. Neither
  draft nor pro may move it; it sits under `not_a_mode_lever` in `TIER-TAXONOMY.yaml`. Default-OFF is
  already true in code — `validation.two_stage_upscale` on `ValidationConfig` (`schema.py:524-528`),
  NOT a top-level field — so D-26 adds only this disclosure obligation, not a default change. `[house]`
  `[canonical]`
- **If a user asks for two-stage it IS allowed** — but the convergence caveat is explained first.
  `[house]`
- **You render on the model you TRAIN on; the render substrate is never a mode lever** (D-28).
  `model.model_id` (`schema.py:134-137`) is a single field used for BOTH training and inference, and
  the cross-field validator at `schema.py:1202-1216` **slaves** `two_stage_upscale` to it (a distilled
  `model_id` with `two_stage_upscale: false` is rejected at config load — D-7-BASEVAR/REF-03).
  Consequence: **"distilled for quicklook / dev for final" is incoherent** — it would render a
  dev-trained LoRA on a distilled substrate. On the house dev path `two_stage_upscale: false` is the
  **only valid setting**, which is also the honest one. Anyone who *does* train distilled inherits the
  forced two-stage path with its caveat — correctly so. The fence already exists in code; **no new
  machinery**. `[house]` `[canonical]`

Reach the depth by keyword rather than restating it:
`rg -i "two-stage|eval-honesty|distilled" .planning/harness/KNOWLEDGE.md`.
