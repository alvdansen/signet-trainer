---
name: training-run
description: >-
  The run-stage playbook for the signet-trainer agentic harness (HARN-01,
  HARN-03). Drives the single gated launch loop —
  dryrun -> cost print -> approval -> launch -> monitor — through
  `modal run -m signet_trainer.modal.entrypoint` ONLY (never a function handle
  directly). Handles strict vs yolo dispatch on the same gate, the cumulative
  session-cap check, D-8-BLANKET blanket grants, the D-8-STATUS read-only Modal
  CLI status snapshot, the training landmines (two-phase VRAM, stale-VRAM,
  single-A100, frame alignment), and the D-8-RESEARCH-LOOP for uncovered knobs.
  Trigger when a config is ready to train/sample/preprocess and you are about to
  launch or monitor a metered Modal run.
---

# training-run — dryrun → gate → launch → monitor

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


> **Read before improvising, log after deciding.** First action: read the
> memory scaffold (`.planning/harness/DECISION-LOG.md` + `KNOWLEDGE.md`) — the
> D-8-BOUNDARY/SOURCETAG/EVIDENCE philosophy and the tone rules from
> `training-session-setup` apply here too. This skill is the gate-respecting
> playbook for launching and monitoring a metered run. It never bypasses the
> run-gate and it always writes back a decision.

---

## 1. STATE-DETECT PREAMBLE (D-8-STATEROUTE)

Retrieval before improvising. Before doing anything metered:

1. **Read the memory scaffold FIRST.** `.planning/harness/DECISION-LOG.md`
   (episodic — what was decided/when) and `KNOWLEDGE.md` (semantic — standing
   policy, `rg -i "<keyword>"`). Do not reason from model priors before
   searching.
2. **Detect real state, read-only (no spend):**
   - dataset encoded? `modal volume ls signe-trainer-dataset`
   - checkpoints present? `modal volume ls signe-trainer-checkpoints <output_dir>`
   - config valid? load it via `signet_trainer.config.load.load_config_from_text`
     (fail-fast LTX validators fire locally — free). Copy-pasteable zero-spend
     one-liner (validates the schema + prints mode/output_dir, no Modal, no GPU):

     ```bash
     PYTHONPATH=src PYTHONUTF8=1 python -c "from pathlib import Path; from signet_trainer.config.load import load_config_from_text; c=load_config_from_text(Path('configs/<your>.yaml').read_text()); print('CONFIG VALID:', type(c).__name__, '| mode:', c.conditioning.mode, '| output_dir:', c.output_dir)"
     ```

     This runs the fail-fast Pydantic/LTX validators only — it does **NOT** run the
     full dry-run stage. The full dryrun (synthetic-batch forward) runs **only inside
     the gate**: `modal run -m signet_trainer.modal.entrypoint --config <yaml> --mode
     <...>` **without `--approve`** is the way to exercise it — the entrypoint's
     app init/teardown + dry-run is free; the metered `.spawn()` sits after the
     approval pause and never fires without `--approve`.
3. **Then route:** proceed if ready · **name what's missing** if not · hand off
   to **training-prep** if the dataset isn't encoded. **NEVER auto-chain
   backwards into metered work** — surface the gap and let the prep stage run
   under its own gate.

Mode is **reference-mode-agnostic** (D-8-MODECONFIG): the mode comes from
`conditioning.mode` in the YAML, not from this skill. Note per-mode preflight:
**`ic_lora` needs `reference_latents/` encoded first** (a prep-stage output).

---

## 2. THE SINGLE-GATE LAW (D-8-PREPROC / the one launch seam)

Every metered run is driven through the entrypoint, never around it:

```bash
PYTHONPATH=src PYTHONUTF8=1 \
  modal run --detach -m signet_trainer.modal.entrypoint \
  --config configs/<x>.yaml \
  --mode <train|sample|preprocess|fuse> \
  [--approve]
```

**`--detach` is REQUIRED here, not optional** — this IS the canonical launch block CLAUDE.md
points at. Without it, Modal tears the ephemeral app shell down with the local client that
dispatched it; `.spawn()` alone only fixes the INPUT type, not the app lifetime (see the
`--detach` advisory in `entrypoint.py`, and README "The three things about this command that
are not optional").

The entrypoint enforces the ordering:
`load config -> dry-run hard gate -> cost print + guardrail -> BLOCKING approval
pause (_require_approval) -> .spawn() ASYNC dispatch -> bounded watch window`. The
`.spawn()` call sits **strictly after** the approval pause, so a metered run can
never auto-launch. **The verb is `.spawn()`, not `.remote()`, since `b9098dd`** —
`.remote()` is `INVOCATION_TYPE_SYNC` and the server cancels a SYNC input whose
client dies (see the launch-env landmine in §5). After dispatch the client prints
the `FunctionCall` id, watches for a BOUNDED `cfg.modal.dispatch_watch_seconds`
window so cheap early aborts still surface, then **disengages without cancelling
anything**. A later process can re-attach with `modal.FunctionCall.from_id(<id>)`.

**Forbidden:**
- **NEVER call `train.spawn()` / `sample.spawn()` / `preprocess.spawn()` — or their
  `.remote()` forms — directly.** Always drive the entrypoint. (Threat T-08-04-EOP.)
- **NEVER re-print a cost line with a hardcoded rate.** The entrypoint prints the
  cost line, reading rates from `cfg.modal` (D-NOHARDCODE). Surface *that* line.
- Every `modal run` needs `PYTHONPATH=src PYTHONUTF8=1` (Windows UTF-8 + src
  layout). Secret **names** flow from `cfg.modal.{huggingface,wandb}_secret_name`
  — carried, never valued.

The four modes reuse the **same** gate: `train` (`train.spawn`), `sample` (the
base-vs-LoRA grid, `sample.spawn`), `preprocess` (the canonical
`process_dataset.py` pre-encode, `preprocess.spawn` — D-8-PREPROC, which retires
the hardcoded `HOURLY_RATE_USD` in the throwaway `scripts/_encode_*` by routing
through this shared cost line), and `fuse` (`fuse.spawn` — Phase 9 inpaint gate:
the **CPU-only** In-Outpainting scaffold fuse, `--mode fuse`, which fuses the
gated `LTX-2.3-22b-IC-LoRA-In-Outpainting` adapter into the dev base with the
config metadata PRESERVED and commits the fused base to the weights Volume; needs
the `hf-gated-secret` Modal secret; run ONCE before the first inpaint train —
the inpaint config's `model.model_id` then points at the fused filename. An
inpaint `sample` run renders `validation.samples` (masked held-out test clips)
into stable `samples_inpaint/<stem>/step_<N>.mp4` columns for the parallel
watcher; an inpaint `preprocess` additionally runs the signet-native mask encode
(`video_masks/`, CPU — upstream at the pin cannot emit it).

---

## 2b. MiniMax-H3 runs — the SAME `--mode train`, routed by FAMILY (Phase 10, H3-05/H3-06/H3-07)

**There is no H3 mode.** A config carrying `family: h3` makes `--mode train` dispatch `h3_train`,
`--mode sample` dispatch `h3_sample`, and `--mode preprocess` dispatch `h3_preprocess` — the
entrypoint routes on `model.family` INSIDE the existing arms. Six dispatches, one gate, one ledger:

```bash
PYTHONPATH=src PYTHONUTF8=1 \
  modal run -m signet_trainer.modal.entrypoint \
  --config configs/<your-h3-run>.yaml \
  --mode train [--approve] --detach
```

**The H3 arch gate runs AUTOMATICALLY, at the front of every H3 stage.** `run_h3_arch_gate` asserts
the ten measured architecture constants and the 300-target / 0-collateral LoRA survey against the
LIVE weights before any sustained spend. Three things follow, and all three are load-bearing:

- **There is no separate smoke step to run first** — every real dispatch already pays for
  abort-before-spend, and P10-1 already proved the architecture on live weights.
- **There is no flag that skips it or stops after it**, deliberately: a cost line is only truthful if
  the function it prices always does the same work. Do not add one; do not look for one.
- **The gate is a plain helper, NOT a Modal function, and that is the point.** ⛔ **Never invoke an
  H3 stage as `modal run -m signet_trainer.modal.fns::<name>`.** That invocation style boots a
  metered A100 with **no cost print and no approval pause** — it is a documented, still-open defect
  on the LTX side (Phase 9 audit finding #18), never a pattern to copy. The single-gate law is
  absolute for H3 too.

`h3_train` additionally runs a **CPU preflight** — one real packed batch built from sample 0 — before
the 61.7 GiB load, so a cache/geometry/payload disagreement aborts at cents.

**The geometry, and why it is not negotiable:**

| | |
|---|---|
| frame law | **`17n + 5`** — valid counts 5, 22, 39, 56, 73, 90, 107, 124 |
| ⛔ | **LTX's `{25, 49, 81}` buckets are NOT valid H3 counts.** Different modulus AND different offset from LTX's `(frames - 1) % 8`. A carried-over LTX bucket list dies at config load under `family: h3` — which is the good outcome; the bad one is editing the law instead of the config. |
| Phase-10 round | 22f target @ 1344x768 + **exactly 2** references at short edge **896** |
| worst pair | **12,394 packed rows** (`C+008`) of a **~13,777**-row ceiling |
| measured anchor | 12,362 rows PASSED on a real A100-80GB at **76.36 GiB peak / 16.6 s/it** |
| at short edge 1024 | worst pair 14,026 — and **6 of the 12** character-by-environment pairs are over. The nominal pair still prices at 12,362 and PASSES, which is exactly how an over-budget pairing reaches a metered run. |

**Campaign geometry needs H200-class VRAM — and trimming references does not rescue it.** A 124f
NO-REFERENCE t2v baseline is 37,806 rows against the same ceiling, so full campaign length is a
GPU-CLASS decision (P10-4), not a geometry or reference-count one. Do not try to reach it by cutting
reference fidelity.

**Cost gate + blanket handling are UNCHANGED** — §3 below applies verbatim. H3 spends against the
same `SESSION-STATE.json` ledger and the same session cap; there is no H3-specific authority.

`[house]` `[precedent: P10-1 / P10-1b measured on a real A100]`

---

## 3. DISPATCH: BLANKET → STRICT / YOLO (on the same gate)

The dispatch decision reads the cumulative-spend ledger written by
`training-session-setup`. Evaluate in this order:

**Cap + ledger-path source of truth (WR-02 — ONE authoritative chain, config-first):**
Never hardcode the ledger path or the cap in this skill. The chain is:

1. **House default (config):** `cfg.modal.session_spend_ledger_path` is the ledger
   path; `cfg.modal.session_cap_usd` is the house cap. Both come from the loaded
   config (`load_config_from_text`), never a literal in this skill (D-NOHARDCODE).
2. **Per-session override (SESSION-STATE.json):** the setup gate writes
   `session_cap_usd` into the ledger file at `cfg.modal.session_spend_ledger_path`;
   that value, when present, is the live cap for the session. Absent → fall back to
   `cfg.modal.session_cap_usd`.

So in every snippet below, `ledger_path = cfg.modal.session_spend_ledger_path` and
`session_cap_usd = <SESSION-STATE session_cap_usd if set, else cfg.modal.session_cap_usd>`.
The old literal `".planning/harness/SESSION-STATE.json"` was that config default's
current value — read it from `cfg.modal`, do not retype it.

**Missing SESSION-STATE.json fails SAFE (D-8-FAILSAFE):** if the file is absent
(fresh session, setup gate not yet run), treat it as **strict mode with no blankets
and no yolo** — never assume yolo or an implicit blanket. Do **not** dispatch with
`--approve` on an absent state. Run the **training-session-setup** gate first to
create the file (it writes the template shape) before any yolo/blanket dispatch;
until then, every metered run goes through the strict ask-first flow (§3b).
(`session_cap.read_ledger` already treats a missing file as `0.0` spent — the same
fail-safe posture: absence never grants spend authority.)

### 3a. BLANKET CHECK FIRST (D-8-BLANKET — consumed BEFORE asking)

In **either** triage mode, before pausing for approval, evaluate the
SESSION-STATE blankets:

```python
from signet_trainer.modal.session_cap import blanket_authorizes, consume_blanket
decision = blanket_authorizes(blankets, run_type, projected_usd)
```

- A matching **in-scope, under-cap, unexpired** blanket → dispatch **with
  `--approve`** without pausing. The **cost line is STILL printed and STILL
  logged** to `DECISION-LOG.md`. **After the dispatch, record the spend with
  `consume_blanket`** (CR-01 — the single blanket-spend path). It does BOTH
  halves in one call: depletes the matched blanket's `spent_usd` (so the
  blanket's `cap_usd` ceiling actually shrinks) **and** appends a `spend` entry
  (so blanket spend is visible to `read_ledger` / the cumulative cap). Use the
  `blanket_index` the decision names — never hand-attribute:

  ```python
  # after the approved blanket dispatch (ledger path from cfg — see §3c):
  consume_blanket(ledger_path, decision.blanket_index, projected_usd, run_ref)
  ```

- Out-of-scope / over-cap / expired (`allowed=False`) → **falls through** to the
  normal strict/yolo flow below.

### 3b. STRICT (default)

- Invoke the gate **WITHOUT** `--approve`. The entrypoint prints the cost line
  and DECLINES (EOFError on non-interactive stdin) → aborts with **no dispatch**.
- **Surface the printed cost line** + what will launch, then **STOP**. The
  operator's explicit go-message is the gate (D-8-STRICTGATE).
- On the go-message: **the agent re-invokes the identical command WITH
  `--approve`** — the operator never types the command.
- **After the approved dispatch, append the spend** (CR-01 — strict dollars must
  NOT be invisible to the cumulative cap): `append_spend(ledger_path,
  projected_usd, run_ref)`. Every approved dispatch — strict, yolo, and blanket
  — records its spend; only yolo appended before, which let a mixed strict+yolo
  session undercount and over-authorize a later yolo run.

### 3c. YOLO

- Pass `--approve` **iff** the cumulative cap allows it:

  ```python
  from signet_trainer.modal.session_cap import session_cap_check, read_ledger, append_spend
  # ledger_path + session_cap_usd come from the config chain above (WR-02) — never a literal here.
  spent = read_ledger(ledger_path)               # ledger_path = cfg.modal.session_spend_ledger_path
  cap_decision = session_cap_check(projected_usd, spent, session_cap_usd)
  ```

- `cap_decision.allowed` → dispatch with `--approve`; then
  `append_spend(ledger_path, projected_usd, run_ref)` to grow the ledger.
- **Flag the operator only when** the projected cost would exceed the remaining cap
  (`allowed=False`) — then drop to ask-first (the strict flow).
- **D-8-YOLOLIMITS — forbidden regardless of cap:** GPU-tier escalation (single
  A100-80GB house rule) and dataset changes. Recipe tuning within house bounds is
  allowed; anything that changes what the model learns still routes through the
  philosophy gate.

---

### 3d. OPTIMIZATION TARGET — direction only, never the arithmetic

The `optimization_target` posture (`draft` | `pro`) colours **which option the
agent names or defaults toward** — venue and cadence direction only. It **never**
touches the cost gate, the guardrail, or the cap. The money calls in §3c take
only `projected_usd` / `spent` / `cap` — **never a mode**. `pro` does NOT raise
`session_cap_usd` or `cost_guardrail_usd`; **pro hitting the cap is CORRECT
behavior** — it drops to ask-first (the §3b strict flow), which is the backstop
working, not a signal to raise the ceiling. Draft's spend effect is **indirect**
(cheaper knobs → lower projected → more runs fit the same cap), never a cap
change. `[house]`

- **Tier-1 authority — yolo only.** Under **yolo**, the mode may auto-select the
  cheaper (draft) or higher-quality (pro) option on **Tier-1 knobs only** —
  venue/cadence direction: `checkpoint_every` cadence,
  `validation.in_loop_sampling`, `validation.num_samples`, sampling venue,
  `data.num_dataloader_workers`, `training.gradient_checkpointing`, quicklook
  render fidelity, `gradient_accumulation_steps`. Do not fork the list — the
  authoritative table is EMITTED from `src/signet_trainer/harness_data/TIER-TAXONOMY.yaml`; find
  it by keyword: `rg -i "tier|carve-out" .planning/harness/KNOWLEDGE.md`. A knob
  ABSENT from it is Tier 3 → ask (D-06). `[house]`
- **Both strict cells are recommend-then-STOP.** Only the two **yolo** cells act
  autonomously on the optimization target. Under strict, draft/pro changes only
  which option is *recommended* — the cost print + STOP (§3b) is unchanged. `[house]`
- **Tier 2 always checks in — in all four quadrants, including yolo+draft**
  (D-03/D-04): resolution (`training_dims` + `data.resolution_buckets`), bucket-F
  (`resolution_buckets.frame_count`), `training.max_steps`, `lora.rank`,
  `blocks_to_swap`. A Tier-2 pick is surfaced and waits — never auto-taken. **Tier
  3 is house-first** (D-05); the mode may ONLY reword the recommendation, never
  auto-pick. `target_modules` is **hard-fenced** — never-touch, not merely ask
  (D-07): dropping `ff.net` is the documented same-lineage scaffold bug that
  corrupted a prior project's likeness. `[house]` `[precedent]`
- **D-23, stated plainly:** draft/yolo does **NOT** repeal the standing rule
  *"Never silently optimize for compute/cost — resource tradeoffs are the
  operator's call"* (§1 tone law, `training-session-setup`). It pre-authorizes a
  **bounded, logged, quality-neutral subset** of it — the three-tier partition IS
  that enforcement. `[house]`
- **D-21 — the one Tier-1 case worth a sentence before it happens:** a cheaper
  `[community]`-sourced Tier-1 option must be **NAMED before use, never applied
  silently**, even though it is Tier 1. House-philosophy-over-community is
  standing; community solutions are for *how it runs* and are verified by testing.
  Surface it in the D-8-COMMUNITY shape: *"the common default is X (underlying
  reason: …); going with your method."* `[house]`

**Absent/ambiguous `optimization_target` fails to `draft` (D-15).** An absent
SESSION-STATE, or a present-but-ambiguous `optimization_target`, is treated as
**`draft`** — never assume `pro`. This mirrors D-8-FAILSAFE (absent state →
strict) and is structurally safe because draft can only trim Tier-1 runs-side
knobs; it cannot cheapen what the model learns. The read is `optimization_target`
from `.planning/harness/SESSION-STATE.json`, one place — the setup gate mirrors
the campaign card's `campaign.optimization_target` (D-13) into it. `[house]`

---

## 3b. VENUE ENFORCEMENT — the declared sampling venue DRIVES config + dispatch (D-VENUE)

`sampling_plan.venue` in `SESSION-STATE.json` is a **declaration that must be enforced**, never a
note the agent reads and then hand-derives from. A declared venue with no acting step is the
divergence failure class the typed-state discipline exists to prevent: SESSION-STATE says
`parallel` while the run quietly does something else, and nobody notices until the grid is empty.

**Read the venue, then apply its row — every train dispatch, no exceptions:**

| `sampling_plan.venue` | `validation.in_loop_sampling` | What ELSE the agent MUST do |
|---|---|---|
| `parallel` *(template default)* | **`false`** — the training container must never load the decoder | **START the watcher** as soon as the train dispatch prints its `FunctionCall id` — i.e. immediately, since `.spawn()` returns at dispatch and the client then disengages when the bounded `dispatch_watch_seconds` window expires. Point `scripts/watch_parallel_inference.py` at the run's `output_dir`; it dispatches a gated `--mode sample` per NEW checkpoint on a SECOND single-A100 container. **The watcher must be OS-detached** (`Start-Process`) — see the client-ownership rule below. Sampling spend lands on the `sample`-scoped blanket if one exists, else the session cap. |
| `pause-to-sample` | **`true`** — renders happen inside the training loop at `checkpoint_every` | No watcher. Expect the wall-clock hit; the operator accepted it when choosing this venue. |
| `end-of-training` | **`false`** | No watcher during the run. Queue a `--mode sample` dispatch AFTER the round completes, before handing to `training-review`. |

**Two hard rules:**

1. **`in_loop_sampling` is DERIVED, not authored.** Setting it by hand against the declared venue is
   a config bug. Under `parallel` it must be `false` — in-loop decoder VRAM would stack on the
   measured training peak at every checkpoint and can eat the headroom a VRAM probe just proved
   (2026-08-04 precedent: peak 62.05 GiB of 80 with sampling OFF; in-loop would have added to it).
2. **A `parallel` venue with no running watcher is a FAILED venue, not a quiet degrade.** If the
   watcher cannot start, say so and surface it — do not let checkpoints pile up unrendered while
   SESSION-STATE claims parallel sampling is happening.
3. **A watcher pointed at the WRONG FAMILY's paths is the same failure, and it is silent.** The
   watcher must be verified against the family actually being trained before it is trusted: LTX
   writes `{output_dir}/samples`, H3 writes `{output_dir}/samples_h3/<key>/` (377a800). A
   family-hardcoded watcher reports healthy and renders nothing. **Check the paths, and check
   whose campaign the supervisor defaults to** — `scripts/watcher_supervisor.ps1` still defaults to
   the LTX `campaign_r5` and no H3 invocation exists in the repo (Wave 2 follow-up).

`[house]` `[precedent: 2026-08-04]` `[precedent: 2026-08-06 — 377a800]`

---

## 4. MONITOR (D-8-STATUS — read-only, no spend)

Poll run state with the read-only Modal CLI snapshot (no live log-attach by
default; poll at sensible intervals):

```bash
modal app list                         # run state: queued/running/done/failed
modal app logs <app-id>                # latest loss lines (snapshot, not attach)
modal app history <app-id>             # run history
modal container list                   # active containers
modal volume ls signe-trainer-checkpoints <output_dir>          # checkpoint progress
modal volume get signe-trainer-checkpoints <output_dir>/samples ./_samples  # pull samples
```

**TIME-GATE LIVENESS (house rule, 2026-07-12 — `rg time-gate` in KNOWLEDGE.md):**
processes lie, artifacts don't. `tasks:1` can be hung; a log tail can lag reality;
no "done" line ≠ still running. **Liveness = artifact freshness** (new checkpoint
dir on the Volume, committed samples), checked against time gates:
- **inference/render: ~60 min** (2× the observed render envelope) without committed
  samples → hung.
- **training: ~60 min** without a NEW checkpoint dir → stalled/preempted.
Gates scale with the eval/cadence spec. On trip: `modal app logs` → stop the
stale app → re-dispatch (in-dir resume is safe — F4/F9). The parallel watcher
carries both gates as config-first constants and prints `[watcher][STALL]`;
verify status against the Volume/grid BEFORE reporting it.

**D-8-SURFACE — surface on blockers + milestones ONLY:** needs-approval, an
unhandleable failure, a convergence-probe result, completion with a grid.
Everything else logs quietly. "Do the work and report once" — no nudging, no
proactive status reports, honor the operator's configured quiet hours.

---

## 5. LANDMINES THIS SKILL CARRIES

Search the memory scaffold for full detail (`rg -i "<keyword>" .planning/harness/KNOWLEDGE.md`):

- **Two-phase VRAM load** — the Gemma text encoder must **NEVER coexist** with
  the 22B transformer on GPU. Free Gemma after the gate, before the loop;
  `load_embeddings_processor(device="cpu")`; wrap every LTX forward in
  `torch.no_grad()`; feed cached `CachedPromptEmbeddings`. Real-A100 profile:
  62.91 GiB gate peak → 38.43 GiB floor once Gemma is released. `[precedent]`
- **Block-swap = freeze-mechanics scope only** — PEFT-aware fix landed 07-13
  (base weights swapped, LoRA sub-Linears kept GPU-resident); long-run adapter
  **quality** under swapping is **UNPROVEN** and **OFFL-02** (a sample under
  active block-swap) is **deferred**. Document the scope limit; do not claim the
  offloader is blanket-trusted. `[precedent]`
- **Stale-VRAM discipline** — for any live GPU debugging: `pkill python3` **and
  verify 0 MiB** before every run. `[house]`
- **Single A100-80GB / no warm GPU** — never multi-GPU, never fallback GPU lists,
  never a warm/idle container (forbidden even in yolo). `[house]`
- **Frame alignment** — **`(frames - 1) % 8 == 0`** (LTX temporal-compression;
  CONF-02 fails fast). Wan's `num_frames >= 33` does **NOT** apply. `[precedent]`
- **Launch env** — every `modal run` uses `PYTHONPATH=src PYTHONUTF8=1`, and every
  metered dispatch adds **`--detach`**. ⚠ **The rationale this landmine carried
  until 2026-08-07 — "`--detach` so the run survives local client/session death" —
  is FALSE.** It was falsified by artifact: apps listed `(detached)` were killed
  with their client anyway (observed twice — once on an LTX inpaint run, once on
  an H3 train + render). **Survival needs TWO independent things and `--detach` is only one of
  them:**
  - **`--detach` keeps the ephemeral APP SHELL alive** past the client
    (`modal/runner.py`) — **REQUIRED, NOT SUFFICIENT.** On an
    `@app.local_entrypoint()` it does not change the dispatch verb at all: Modal's
    CLI auto-swaps `.remote()` → `.spawn().get()` under `--detach` **only for a
    direct function ref** (`modal/cli/run.py`, verified in installed modal 1.5.0);
    the local-entrypoint path merely prints a warning. This repo's entire launch
    surface is a local entrypoint, so it never received Modal's own fix.
  - **The ASYNC dispatch verb keeps the IN-FLIGHT INPUT alive.** `.remote()` sends
    `FUNCTION_CALL_INVOCATION_TYPE_SYNC` and the server cancels a SYNC input whose
    owning client disappears; `.spawn()` sends `ASYNC` and it is not cancelled that
    way. The entrypoint dispatches via `.spawn()` at **all nine gated arms** since
    `b9098dd`, proven by artifact in `42e8e5f`.

  Keep `--detach` on every dispatch — the entrypoint prints a runtime WARNING when
  it is missing — but **never restate it as the survival mechanism**. Separately,
  Modal preemption auto-restart is NOT reliable (verify tasks>0 after any
  preemption notice): that is D-10-DEF-16, a *different* failure from client-death
  cancellation. Do not conflate them. `[canonical]` `[precedent]`
- **Who holds the Modal client, and for how long** — ask it before every long
  metered run; "an agent turn" is never an acceptable answer. Full rule:
  `rg client-ownership .planning/harness/KNOWLEDGE.md`. `[house]`
- **Final-checkpoint naming is PER-STACK** — signet-trainer's final IS
  step-numbered (`checkpoint-step-{max_steps:05d}-loss-*`; resolve via
  `find_latest()` only); canonical ltx-trainer's final is
  `lora_weights.safetensors` (NO step). Read the REAL dir name off the Volume
  before pointing anything (init_adapter_path, sample fetch) at a "final".
  `[canonical]` `[precedent]`
- **Chained-round handoff** — follow the 6-step checklist in KNOWLEDGE.md
  (`rg round-handoff`): real-dir round final → re-point init_adapter_path →
  watcher OUTPUT_DIR/SAMPLE_CONFIG/**truthful cumulative STEP_OFFSET** →
  round-scoped staging hygiene → relaunch watcher + detached gated dispatch →
  **warm-start probe** (log line + keyset + first-sample continuation)
  before trusting the round. `[precedent]`
- **Leaner-YAML anti-pattern.** If you find yourself about to omit a config block
  because draft "does not need it", STOP — emit the explicit value instead. **Both
  modes emit FULLY-SPECIFIED effective configs.** Draft cheapening is explicit
  values, never omitted blocks — an omitted block silently inherits a schema
  default, a documented prior-round trap (`uniform_prob` 0.30 → 0.1: the config looked
  lean and the model learned something different). `[house]` `[precedent]`

---

## 6. LOGGING FORCING FUNCTION (D-8-RESEARCH-LOOP step 4)

**After any run decision, append a `DECISION-LOG.md` entry — required output, not
optional.** This is the write-back that prevents the previous agent's core
failure (stale logs). Each entry has all four fields:

```
## YYYY-MM-DD - <short title>
- what: <the decision / preference / surprise>
- why:  <evidence, precedent, or house rule>
- state:                      # TYPED SLOTS (typed-state-beats-prose, 2026-07-14): every
    <name>: <value>           # load-bearing numeric/path as a `name: value` line — steps,
    <name>: <value>           # loss, $, checkpoint dirs, app-ids. NEVER only-in-prose.
- source-tags: <[house] [precedent] [canonical] [community]>
- run-refs: <run ids / checkpoint paths / commit SHAs / phase-plan ids, or "none">
```

**Mode-pick obligation (D-20).** Every **mode-driven pick** — any option the
`optimization_target` posture selected — is logged to `DECISION-LOG.md` as a
`class: mode-pick` entry carrying the mandatory core in its typed `- state:`
block: `knob`, `tier`, `chosen`, `source_tag`, `optimization_target` (the exact
core `COMPACTION-PROTOCOL.md` declares), plus the reasoning in `why`. The
`source_tag` names the **OPTION'S OWN source class** (`[house]` / `[precedent]` /
`[canonical]` / `[community]`). Per D-8-SOURCETAG the mode is a **selection policy
over already-tagged options — never itself a tag**; **"because draft/pro" is a
banned untagged `[prior]`.** This is **hard-fail enforced, not advisory**:
`harness_lint.py` resolves the entry's `knob:` against
`src/signet_trainer/harness_data/TIER-TAXONOMY.yaml` and fails a mode-driven pick on any
Tier-2/3 knob — and on any knob absent from the taxonomy (D-06 fail-safe to Tier
3). Run it: `PYTHONPATH=src PYTHONUTF8=1 python -m signet_trainer.harness_lint`.
`optimization_target` is a SESSION-STATE / card field (not a skill literal) and
the concrete values still land in the YAML — **D-NOHARDCODE is preserved**
precisely because the posture selects values rather than hiding them. `[house]`

**Numeric-drift law (`rg typed-state` in KNOWLEDGE.md):** `SESSION-STATE.json` is the single
source of truth for spend/cap — quote it fresh (`read_ledger`), never hand-copy a remembered
figure into prose (the remembered-figure-vs-ledger drift class, e.g. a recap quoting a stale
spend total dollars off the real ledger). Prose narrates; the `state:` block carries the numbers.

**Emitter step (D-TS-4 — explicit skill step, no hook).** After appending the DECISION-LOG entry,
regenerate the TYPED-STATE blocks in the prose artifacts from the typed sources — a REQUIRED step,
not optional:

```
PYTHONPATH=src PYTHONUTF8=1 python -m signet_trainer.harness_state emit
```

Edit the SOURCES (`<card>.state.yaml`, `SESSION-STATE.json`) — never the emitted blocks. Before
committing, run the two hard-fail backstops: `python -m signet_trainer.harness_state check` (emitted
blocks match their sources) and `python -m signet_trainer.harness_lint` (this log's post-cutoff
entries carry their `state:` slots + class cores). Both are wired into the test suite, so a normal
`pytest` run also catches drift. `[house]`

Log **decisions, preferences, surprises** — not narration, file lists, or
command transcripts (git already tracks those). Test: *"will this be true and
useful next month?"*

---

## 7. UNCOVERED-KNOB RESEARCH PROTOCOL (D-8-RESEARCH-LOOP)

The house's "most important part of this build." When you hit an **uncovered knob** (a
setting/decision with no stored house answer), run the shared procedure verbatim
— it lives in `.planning/harness/README.md` §"Research protocol":

1. **Check `KNOWLEDGE.md` by keyword FIRST** (retrieval-before-improvising).
   Search the bare token; if a house rule / landmine / precedent covers it, use
   it and skip to step 4. Do not reason from model priors before searching.
2. **If nothing found, research then STORE.** Consult the **canonical pinned
   source** (the trainer / ltx-core code or official docs) and **community
   docs**, then **write the findings into `KNOWLEDGE.md` with source tags**
   (`[canonical]` / `[community]`). Storing is part of the loop — the next
   session must not re-research it.
3. **Present options WITH explanations.** Per **D-8-COMMUNITY, community
   knowledge is NEVER a silent override.** Surface it as:
   **"the common default is X (underlying reason: …); going with your method"** —
   the community default is named with its reasoning, and the house method
   is the pick. Never present a `[community]` default as the recommendation.
4. **Log the decision automatically to `DECISION-LOG.md`** (§6). Required output.

**Boundary routing (D-8-BOUNDARY):** if the knob changes **what the model
learns** (data, captions, recipe, eval/convergence judgment) it is philosophy
territory — house rules first, ask when uncovered. If it changes **how
fast/cheap/reliably it runs** (VRAM, throughput, infra, API) it is mechanical
territory — community solutions welcome, verified by testing. **Precedent
conflict (D-8-LOGHOME): the project log wins, and the divergence is flagged to
the operator** — never silently reconciled.
