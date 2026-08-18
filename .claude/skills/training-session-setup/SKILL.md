---
name: training-session-setup
description: >-
  The ONE upfront session-setup gate for the signet-trainer agentic harness
  (D-8-SETUPGATE). Invoke this FIRST, at session start, before any metered work
  (any modal run, any preprocess/train/sample dispatch). It carries the house
  training-philosophy preamble (D-8-BOUNDARY / D-8-SOURCETAG / D-8-EVIDENCE) and
  collects the session's triage mode (strict/yolo), yolo spend cap,
  optimization target (draft/pro), sampling plan, and blanket grants — then
  writes them to
  .planning/harness/SESSION-STATE.json. Trigger whenever a fresh session begins
  operating the trainer, or before the first training-prep / training-run /
  training-review step.
---

# training-session-setup — the one upfront gate

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


> **Read before improvising, log after deciding.** This is the first skill a
> fresh session invokes. It does two things: (1) loads the house **philosophy
> layer** into the session so every later decision is sourced correctly, and
> (2) runs the **one setup dialog** that fixes triage mode, cap, sampling plan,
> and blankets — persisted to `.planning/harness/SESSION-STATE.json`. Everything
> after this honors that state ("honor pre-authorization, self-sequence").

---

## 0. PHILOSOPHY PREAMBLE (D-8-CANON — reference, never re-derive)

### ⭐ D-8-USECASE — PRIME DIRECTIVE: work backwards from the use case (governs everything)

Before philosophy, before recipe, before any knob: **start from the END GOAL, never the tech.**
Working backwards from the use case is the tried-and-true recipe for success (how the consulting
business is run) and matters **doubly in AI** — the field is insulated from the domains it's trying
to master, so the "obvious" default is often **half-baked**. So, at the very start of every
engagement:
1. Ask the user the **EXACT goal** — *what is the final PIXEL they want to achieve*, and *what do
   they hope to DRIVE it with* (text, image, video, control, audio, …). Name the goal + the
   driver-modality up front; never presume the driver (this is why the harness is modality-extensible).
2. Pressure-test the **building blocks FROM THE START** — think critically about whether the
   primitives can ACTUALLY execute the proposed architecture *before* committing, not after building.
   (The validation gate + "validate on a real GPU first" are this instinct made concrete.)

Goal-first + primitives-pressure-tested + driver-named → success honestly every time. Global memory:
`work-backwards-from-usecase` (vault); KNOWLEDGE.md prime-directive banner.

### D-8-CANON sources

The most important part of this build (house intent, verbatim): the agent must
apply **house training philosophy over community knowledge AND over its own
model priors**, and can identify which territory it is in. A previous training
agent failed on exactly this. These sources are canonical — **reference them,
never re-derive**:

- the house training-methodology docs (external to this repo):
  `general-training-methodology/{README,harness-requirements,agent-behavior,dataset-and-captioning,eval-and-config}.md`
- the training-agent behavioral handoff spec (`TRAINING-AGENT-HANDOFF.md`, external — §1 behavioral
  spec + §6/§7 log discipline)
- `.planning/harness/{README,KNOWLEDGE,DECISION-LOG}.md` (this project's memory scaffold)
- *Forgetting on Purpose* — **the published house methodology paper (Alvdansen Labs,
  2026): generalization-within-concept as the quality criterion, the five-tell overfitting
  diagnostic, chained subset-rotation → combined-consolidation training. EVERY agent operating
  the trainer reads/knows this** (house directive, 2026-07-11).
- `.planning/harness/LEARNINGS-CROSS-PROJECT.md` — scrubbed cross-project audit (prior campaigns
  + Wan-era): ~48 sourced findings incl. silent zero-load warm-starts,
  effective-config diffing, five-tell review, verify-the-artifact.

### D-8-BOUNDARY — the learns-vs-runs test (routes every uncovered decision)

- **Philosophy territory** = anything that changes **WHAT the model learns**:
  data, captions, recipe, eval judgment, convergence calls. → House rules /
  precedent first; **ask when uncovered**; community only via the research loop
  (never a silent override).
- **Mechanical territory** = anything that changes **how fast / cheap / reliably
  it runs**: VRAM, throughput, infra, API mechanics. → Community solutions are
  welcome, **verified by testing**.

When unsure which side a decision sits on, treat it as philosophy territory and
ask.

### D-8-SOURCETAG — every recommendation carries a source class

`[house]` (methodology docs / house rule) · `[precedent]` (a logged decision or
executed-phase finding) · `[canonical]` (pinned source code / official docs) ·
`[community]` (external docs) · `[prior]` (the model's own training).
**Untagged model priors are the documented drift failure mode** — an untagged
claim is a violation. Search the memory scaffold with a bare keyword
(`rg -i "<token>" .planning/harness/KNOWLEDGE.md`) before stating a rule.

### D-8-EVIDENCE — house wins unless same-project A/B evidence

The bar for the agent to even **raise a challenge** to a house rule is
**same-project A/B sample evidence** (one variable changed, same data/seed).
Priors and community posts never qualify. The operator arbitrates; the challenge
and the verdict are logged either way (to `DECISION-LOG.md`).

### D-8-COMMUNITY — never a silent override

When community knowledge is relevant, surface it as: **"the common default is X
(underlying reason: …); going with your method."** The community default is
named with its reasoning; the house method is the pick. Never present a
`[community]` default as the recommendation.

---

## 1. TONE (house behavioral spec §1 — non-negotiable)

The operator is the domain authority; the agent adapts to the operator's
communication needs, never the reverse.

- **Dry, direct.** Lead with the answer. No walls of text; scannable structure.
- **No guilt/shame language**, no cheerleading, no apology loops, no flattery.
- **Do the work and report once.** No nudging, no proactive/unprompted status
  reports. **Honor the operator's configured quiet hours — no task recaps during
  them.**
- **Every turn is self-contained** — restate any earlier question you reference;
  never "as I asked above."
- **Never silently optimize for compute/cost.** Resource tradeoffs are the
  operator's call — surface them and wait.
- **One question at a time** in this setup dialog (and in teach mode).
- **A question timeout = the operator is away.** Pause and wait. Never text-dump
  the questions, never auto-complete the setup, never proceed to metered work on
  a timeout. A terse reply still gets its follow-up; it is not permission to skip.

---

## 2. THE SETUP DIALOG (collect upfront, one question at a time)

Ask these **before any metered work**, one at a time, and record the answers.
Do not batch them into a wall of text.

1. **Triage mode** — `strict` (default) or `yolo`?
   - **strict** — the agent diagnoses + recommends, **prints the cost line, then
     STOPS**; the operator's explicit go-message is the gate; the agent then
     dispatches with `--approve` itself (the agent runs all commands — the
     operator never types them).
   - **yolo** — the agent decides and acts **within bounds**: dispatches flow on
     the cost gate + cumulative cap alone; flag the operator only when a projected
     cost would exceed the remaining cap. (Bounds in §4.)

2. **Yolo cap (USD)** — only if yolo. Propose the house default from
   `session_cap.py` (`DEFAULT_SESSION_CAP_USD`, currently the proposed house
   figure) and let the operator confirm or change it. **Do NOT hardcode the number
   in this skill** — read it from the module / SESSION-STATE, state it, let the
   operator set the live value. This is a cumulative session cap, not a per-run figure.
   For a docs-only operator: the `session_cap_usd` shipped in
   `src/signet_trainer/harness_data/SESSION-STATE.template.json` **IS** the house default and
   mirrors `DEFAULT_SESSION_CAP_USD` in `src/signet_trainer/modal/session_cap.py` —
   that module constant is the single source of the number; read it from
   `session_cap.py` (or mirror it from the template) rather than restating a literal here.

3. **Optimization target** — `draft` (default) or `pro`?
   - **draft** — cost/speed. On quality-neutral **runs-side** knobs (Tier 1) the agent
     names/defaults toward the cheaper option. It can ONLY trim Tier-1 knobs — it can never
     cheapen WHAT the model learns.
   - **pro** — professional-client, quality-max. The agent names/defaults toward the
     highest-quality runs-side option (parallel venue, denser cadence, the proven no-swap path).

   Three properties keep this axis safe:
   - This is the **optimization-target** axis, orthogonal to `triage_mode`'s **approval-friction**
     axis: strict/yolo decides *does the dispatch pause?*; draft/pro decides *which option does the
     agent name/default toward?* They compose into a 2×2; all four quadrants keep the
     cost-print-always-logs invariant and the run-gate.
   - **Tier 2 always checks in, even under yolo+draft** — resolution, bucket-F, `max_steps`, `rank`,
     `blocks_to_swap` (D-03/D-04). **Tier 3 is house-first**; the mode may only reword the
     recommendation, never auto-pick (D-05). Reach the partition by keyword:
     `rg -i "carve-out|tier" .planning/harness/KNOWLEDGE.md` (that table is EMITTED from
     `src/signet_trainer/harness_data/TIER-TAXONOMY.yaml`, the single source).
   - **`pro` does NOT raise the cap** (D-19) — not `session_cap_usd`, not `cost_guardrail_usd`.
     Accounting stays entirely on the triage axis; `pro` biases values only. `pro` hitting the cap is
     CORRECT behavior — it drops to ask-first (the backstop working), never a signal to raise the ceiling.

   **House default — name it, do not hardcode it (D-NOHARDCODE).** Unlike the yolo cap, there is
   **no module constant** for this field — no Python reads the posture (PATTERNS F-1). The
   `optimization_target` shipped in `src/signet_trainer/harness_data/SESSION-STATE.template.json` **IS** the house
   default; read it from the template and state it rather than restating a literal here.

   **Fail-safe (D-15).** An absent or ambiguous `optimization_target` fails to **`draft`** (least
   spend) — structurally safe because draft can only trim Tier-1 runs-side knobs, exactly as an
   absent SESSION-STATE fails to `strict` (D-8-FAILSAFE).

   **Conflict rule (D-16).** The campaign's TRAINING-CARD `.state.yaml` carries the durable posture
   (`campaign.optimization_target`) and the gate shows THAT value as the named default. If the session
   answer differs, **the session wins for the session only and the card is untouched unless the
   operator says so** — the override applies to this session only, is logged to `DECISION-LOG.md` as a deliberate
   deviation, and a follow-up asks whether to make it durable (write back to the card). A one-off
   cheap experiment must never silently convert a `pro` campaign to `draft`. `[house]`

4. **Sampling plan (D-8-GRIDWATCH)** — **ALWAYS ask** whether sampling should
   occur this session. If yes, pick the **venue**:
   - **parallel** (if GPUs allow),
   - **pause-to-sample** mid-run (slower; the operator accepts the tradeoff),
   - **end-of-training**.
   Samples are produced with `alvdansen/finetune-gridwatch` (the canonical
   sampler harness — do not build a parallel one). Default posture:
   **always default to generating samples.**

5. **Blanket grants (D-8-BLANKET)** — any phase/session-scoped pre-authorization?
   Each blanket is a named, capped, expiring grant:
   `scope` (run types: any of `train`/`sample`/`preprocess`), `cap_usd`,
   `expires` (`"session"` or an ISO-8601 timestamp), `spent_usd` (starts 0.0).
   A matching in-scope, under-cap, unexpired blanket authorizes dispatch **with
   `--approve`** without pausing — but the **cost line is still printed and still
   logged** to `DECISION-LOG.md`. Out-of-scope / over-cap / expired falls through
   to the normal strict/yolo flow.

---

## 3. WRITE STEP — persist to SESSION-STATE.json

Write the collected answers to `.planning/harness/SESSION-STATE.json` in the
template shape (`src/signet_trainer/harness_data/SESSION-STATE.template.json`):

```json
{
  "triage_mode": "strict",
  "session_cap_usd": 10.0,
  "optimization_target": "draft",
  "sampling_plan": {
    "generate_samples": true,
    "venue": "parallel",
    "note": ""
  },
  "blankets": [],
  "spend": []
}
```

- `triage_mode`: `"strict"` | `"yolo"`.
- `session_cap_usd`: the confirmed yolo cap — defaults to `DEFAULT_SESSION_CAP_USD`
  (`session_cap.py`, the single source) unless the operator changes it; the literal shown in
  the template block above must equal that constant. Present-but-unused in strict mode.
- `optimization_target`: `"draft"` | `"pro"`. The optimization-target axis; absent/ambiguous fails to
  `"draft"` (D-15). Mirrored at setup from the campaign card's `campaign.optimization_target` so the
  run/review skills read one place (D-13).
- `sampling_plan`: `{generate_samples, venue, note}` — whether to sample this session
  (`generate_samples`), the `venue` (`parallel` / `pause-to-sample` / `end-of-training`), and a
  free-text `note`. Default posture: always generate samples.
- `blankets`: list of `{scope, cap_usd, expires, spent_usd}` entries (empty if
  none). `scope` is a list of run types; `expires` is `"session"` or ISO-8601;
  `spent_usd` starts at `0.0`.
- `spend`: append-only ledger of `{ts, est_usd, run_ref}` dispatch entries —
  **leave `[]` at setup**; the run skill appends to it via
  `session_cap.append_spend`.

This is the **session-scoped** file — NOT part of the durable memory layers
(`KNOWLEDGE.md` semantic / `DECISION-LOG.md` episodic). It is rewritten per
session.

**Typed-state pointer (`.planning/harness/COMPACTION-PROTOCOL.md`):** SESSION-STATE.json IS this
stage's typed compaction artifact — the single numeric source of truth for spend/cap (quote via
`session_cap.read_ledger`, never hand-copy its figures into prose recaps of setup). `[house]`

---

## 4. SESSION-LONG INVARIANTS (honor pre-authorization, self-sequence)

Once setup is written, everything after honors it. Hard limits that hold in
**every** mode:

- **D-8-YOLOLIMITS — forbidden even in yolo:**
  - **GPU-tier escalation.** Single **A100-80GB** house rule — never multi-GPU,
    never fallback GPU lists, never a warm/idle container. `[house]`
  - **Dataset changes.** Dataset composition is methodology (learns-what)
    territory — off-limits to autonomous action. `[house]`
  - Recipe tuning *within house bounds* is allowed in yolo; anything that changes
    what the model learns still routes through the philosophy gate.
- **Cost prints ALWAYS still log.** A blanket or a yolo cap removes the per-run
  *stop*, never the *accounting*: the cost line is printed and a
  `DECISION-LOG.md` entry is appended on every dispatch decision.
- **The run-gate is law.** No metered run auto-launches — the entrypoint's
  `_require_approval` pause sits strictly between the cost print and the
  `.spawn()` dispatch (ASYNC since `b9098dd`; it was `.remote()` before, and that
  SYNC verb is what let a dying client cancel healthy runs). Skills drive that
  gate; they never bypass it.
- **Modality:** the current configuration is **video**. Video is a
  configuration, not an assumption — do **NOT** hardcode video-only. The setup
  gate and SESSION-STATE shape are modality-extensible.

---

## 5. LANDMINE POINTERS (skills carry the depth)

These are carried in full by the stage skills; named here so setup surfaces
them. Search the memory scaffold for detail:

- **Two-phase VRAM load** — Gemma never coexists with the 22B on GPU
  (`rg -i "two-phase" .planning/harness/KNOWLEDGE.md`). `[precedent]`
- **Single A100-80GB / no warm GPU** — `rg -i "A100" ...`. `[house]`
- **Stale-VRAM discipline** — `pkill python3` + verify 0 MiB before any live GPU
  run. `[house]`
- **Config-first / D-NOHARDCODE** — caps, rates, sampling defaults, log paths
  live in config/SESSION-STATE, never in code or in this skill. `[house]`
- **Block-swap offloader is freeze-mechanics-scoped only** (OFFL-02 deferred) —
  `rg -i "block-swap" ...`. `[precedent]`

---

## 6. AFTER SETUP — hand off, don't chain into spend

Setup is written. Route to the right stage skill by detected state — do **not**
auto-chain into metered work:

- data not encoded → **training-prep** (drives the gated `preprocess` mode);
- encoded + config valid, ready to train → **training-run**;
- checkpoints exist, sampling/convergence wanted → **training-review**.

Each `SKILL.md` carries its stage's full playbook. Log the mode/cap/sampling/
blanket choices as a dated `DECISION-LOG.md` entry (with source tags) — the
write-back is a **required output** of setup, not an afterthought.

**Emitter step (D-TS-4 — explicit skill step, no hook).** At any session pause / handoff /
`.continue-here` write, regenerate the TYPED-STATE blocks from the typed sources so the compaction
artifacts (card / `.continue-here` / STATE.md continuity) carry live figures — a REQUIRED step:

```
PYTHONPATH=src PYTHONUTF8=1 python -m signet_trainer.harness_state emit
```

Edit the SOURCES (`<card>.state.yaml`, `SESSION-STATE.json`), never the emitted blocks. Before
committing a handoff, run `python -m signet_trainer.harness_state check` (emitted blocks match their
sources) — the staleness backstop that catches a hand-edit inside the markers or a source changed
without a re-emit. `[house]`

6. **Sample clip length (house rule, 2026-07-11 — never default silently).** Ask the user the eval
   clip length as part of setup. House standing default: **5 s = 81 frames** at the 16 fps ref
   cadence ((F-1)%8==0). The locked eval spec: 81f / 40 inference steps / guidance 4.0 / seed 42 /
   res-matched / no two-stage upscale (house `ltx-training-defaults`, 2026-07-10). Inference
   length is independent of the training window; longer inference time is accepted. `[house]`

   **Single-stage render disclosure (D-27 setup half — name it, don't leave it silent).** State it
   explicitly at the gate as the house default: **renders are single-stage — two-stage can
   misrepresent LoRA convergence.** House reasoning (2026-07-16): the upscale stage **may not fully
   respect the LoRA's convergence** — it can show detail the adapter did not actually learn. Three
   properties keep this OFF the axis:
   - `two_stage_upscale` is **OFF by default, POSTURE-INDEPENDENT, and NOT a draft/pro lever** (D-26).
     It is an **eval-honesty rule** on the learns/eval side of D-8-BOUNDARY, not a cost knob — neither
     draft nor pro may move it. It sits under `not_a_mode_lever` in `TIER-TAXONOMY.yaml`.
   - If a user asks for two-stage it **IS allowed** — but the convergence caveat is explained first.
   - **You render on the model you TRAIN on (D-28).** `model.model_id` is ONE field used for BOTH
     training and inference, and a cross-field validator already SLAVES `two_stage_upscale` to it (a
     distilled `model_id` with `two_stage_upscale: false` is rejected at config load). On the house dev
     path `two_stage_upscale: false` is the ONLY valid setting — which is also the honest one. The
     render substrate is never a mode lever.

   Point at the depth by keyword rather than restating it —
   `rg -i "two-stage|eval-honesty|distilled" .planning/harness/KNOWLEDGE.md` — and note that
   `src/signet_trainer/harness_data/HOUSE-SPEC.yaml` (`disclosures.single_stage`, `verdict_spec`) is the typed home
   of the locked eval spec this item recites. Do NOT add a "two-stage?" question — D-26 makes this a
   disclosure of a posture-independent house default, not a setup choice, and D-11 caps the question
   count. `[house]`

7. **THE TRAINING RECIPE SURFACE IS FIRST-CLASS SETUP — never presume (house rule, 2026-07-11).**
   Beyond triage/cap/sampling/blankets/clip-length, setup MUST sort these WITH the user, one at
   a time, house default named per question (user confirms or overrides — a Claude-derived value
   is never a default):
   - **Round sizing**: default 3000 steps/round (prior-campaign unit; final round may run longer). Never
     corpus-scaled.
   - **Training resolution**: default = native-scale (largest 32-aligned W×H the data supports).
     Never offer only pre-shrunk options.
   - **Bucket lengths (multi-F)**: default = the locked {25,49,81} pattern — clips train at their
     REAL lengths. Never flatten to single-F.
   - **Eval prompt set**: a SET (Klippbok grammar, in-dist echoes + generalization probes), never
     one prompt. **Dataset captions must NOT dominate** (KNOWLEDGE `sample-prompts`) — a verbatim
     caption or two is fine (in-dist recall check), but weight the set toward RANGING probes: a
     different prompt FORMAT (respecting anchor words) + a NOVEL action/scene; a caption-dominated
     set only tests memorization, not generalization. **Difficulty follows the GOAL**
     (KNOWLEDGE `test-difficulty`): proving a NEW
     capability → weight EASY subjects (realism/3D/full-face portraits) so a failure is a trusted
     TRUE negative; refining/validating settings → weight HARD subjects (angle-variant clothing,
     linework, difficult motion); measuring performance → BOTH, and never call it validated on the
     easy subjects alone (don't stop at low-hanging fruit).
   - **Methodology application (ALWAYS ASKED — house rule, 2026-07-11)**: forgetting-on-purpose is the
     DEFAULT methodology; ask whether this campaign applies it **strictly or loosely** and what
     the round plan is (e.g. the reference campaign, strict: 3 rounds x 3000 + final full-dataset round at 8000).
     Small corpora with little to rotate are fine — that's what chaining is designed for. The
     chained approach itself CAN BE OVERRIDDEN at this question (single-round / other schedule —
     the user's call; house rule, 2026-07-11): default != mandatory.
   - **Learning rate (`lr`) — first-class, never buried (D-09)**: default **5e-5** (`[house]`,
     KNOWLEDGE `lr`/`rank`/`alpha`). Asked every session — house position: lr and alpha are normal
     things to ask about and must never be buried, though a default exists for users who aren't
     sure. `lr` **stays Tier 3 for mode purposes** — the mode may NEVER auto-pick it; the promotion
     is only that the setup gate always SURFACES it, not mode authority. `[house]`
   - **Rank + alpha, ONE coupled question (D-10)**: house default presented as a **pair**,
     `rank=64 / alpha=64`, with an explicit option to decouple. PEFT scale = alpha/rank, so
     `rank == alpha == 64 → scale 1.0` is a clean-convert property that should be intentional rather
     than implied. Coupling heuristics (KNOWLEDGE `lr`/`rank`/`alpha`): raise rank → drop LR; rank ~16
     favours likeness, rank 64 for style — pick rank by intent, then set LR accordingly. `rank` remains
     a **Tier-2 carve-out** (always checks in, even under yolo+draft); `alpha` rides the same question
     as **Tier 3**. `[house]`
   - **`keep_checkpoints` — save-everything default, ask BOTH halves (D-08)**: default = **save
     everything automatically, never prune** (`keep_checkpoints: null`). Setup ASKS whether the operator
     wants to override it or set a rate — house rule: automatically save, but ask for both halves
     (override / rate) during setup. Two hard facts: **draft can never prune** — this is
     NOT a mode lever at all; it is excluded from the taxonomy (see `not_a_mode_lever` in
     `TIER-TAXONOMY.yaml`); and `keep_checkpoints` **silently prunes** to the N-most-recent via
     `shutil.rmtree` after each save, so a finite value on a long run deletes early checkpoints. In a
     research lab the intermediates ARE the artifacts — the reference campaign's picked likeness was a mid-run step, not
     the final. `[house]`
   - **No other promotions (D-11)**: `optimizer`, scheduler/warmup/`min_lr`, and `uniform_prob` stay
     **house-first / ask-when-uncovered** and are deliberately NOT promoted — this keeps the gate from
     becoming a 15-question interrogation. `[house]`
   The 2026-07-11 reference campaign burned three restarts on presumed values for exactly these —
   the failure class is "pre-shrunk options presented as the menu." `[house]`

8. **Checkpoint backup destination (BK-01 — never presumed, D-BK-5).** Off-Volume durability is a
   first-class setup question, asked like the recipe surface above — a Modal Volume alone is not a
   backup, so where checkpoints mirror is the USER's call, never a silent default. Ask it, one at a
   time, and write the answer onto the run config's `backup` block (`config/schema.py::BackupConfig`):
   - **House default (name it, let the user confirm/override):** a PRIVATE HF repo —
     `enabled: true`, `destination: hf`, `private: true`, `what: all` (mirror EVERY complete
     checkpoint; intermediates ARE the research artifacts, D-BK-2 — the reference campaign's picked likeness was a
     mid-run step, not the final). `repo_id: owner/name` is required for `hf`. HF auth reuses the
     EXISTING `huggingface-secret` Modal secret (`modal.huggingface_secret_name`) — no token in the
     config, ever.
   - **NOT a working option this phase — disclose honestly:** `destination: local` and
     `destination: cloud` are both schema-ready (the enum + a reserved `cloud_secret_name` seam for
     `cloud` exist for forward-compat) but neither is durable/implemented here. Never present either
     as a choice: an enabled `destination: local` config **fails fast at load** — `backup_sync` runs
     in a Modal container with an EPHEMERAL filesystem, so a "local" copy is never committed to a
     Volume and would report success and then vanish (issue #23 finding 1) — and an enabled
     `destination: cloud` config fails fast as "not yet implemented" (`BackupConfig` validator).
     `hf` is the only wired, durable destination this phase — offer only `hf`.
   - **Default OFF is honest too:** if the user declines backup, leave `enabled: false` — every
     existing YAML loads byte-identically and no backup runs.
   - **How it runs (off the metered GPU):** backup is a CPU-only Modal job driven by
     `--mode backup` (mirrors only NEW complete checkpoints, additive — never deletes); restore is
     `--mode restore` (a 1:1 copy-back that rehydrates + commits the checkpoints Volume). Both thread
     the SAME entrypoint gate (cost print + approval) as every other run — the cost is near-zero but
     still printed and logged. Neither ever touches the A100 (D-BK-3) — an upload hang can never wedge
     a running train. The harness can drive periodic sync via `--mode backup --approve` under a yolo
     blanket, exactly like `--mode sample --approve`. `[house]`

9. **TRAINING CARD (house rule, 2026-07-11).** Setup CREATES (or resumes) the campaign's card at
   `.planning/harness/cards/TRAINING-CARD-<campaign-slug>.md` — the per-campaign durable record
   (goal, corpus, final recipe, rounds table, corrections, spend, verdicts). Run/review skills
   UPDATE it at every round, verdict, and closeout. It is the recall unit after memory clears —
   deeper than DECISION-LOG entries, scoped tighter than KNOWLEDGE. The card carries a
   **Findings (F-numbered)** section, precedent-style — DOCUMENT FINDINGS AS THEY LAND (per leg,
   not at closeout; house rule, 2026-07-11). `[house]`
