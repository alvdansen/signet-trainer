---
name: training-prep-inpaint
description: >-
  Drive the INPAINT dataset prep pipeline — mask creation (hand-draw app OR external teal paintover),
  SAM3 mask propagation, the D-04 QA hard gate, staging, and the gated `--mode preprocess` encode of
  `video_masks`. Use this instead of `training-prep` when the dataset carries inpaint masks (a
  `video_masks` column / a masked-region generate task). It is the sibling of `training-prep`: it
  REUSES `training-prep`'s §0 STATE-DETECT and §2 single-encode-gate verbatim and adds the full mask
  playbook on top. The masks are frames, not a hardcoded video assumption — modality-extensible.
---

# training-prep-inpaint — mask a dataset, propagate, QA-gate, encode for a gated LoRA run

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


This skill owns the **inpaint-specific prepare** stage of the harness loop:
`mask → SAM propagate → D-04 QA hard gate → stage → gated encode → dry-run gate → gated train`.
It is the **sibling** of `training-prep` — same harness discipline (STATE-DETECT, source tags, the ONE
encode gate, the DECISION-LOG write-back), plus the mask playbook `training-prep` deliberately does not
carry. "No other trainer takes responsibility for inpaint data prep" — this harness IS that value-add.

> **Modality (do NOT hardcode video-only):** the mask surface is a stack of per-frame binary masks
> (`masks_video/<stem>/NNNNN.png`) plus a frame-0 seed (`masks_frame0/<stem>__<type>.png`). Video is
> the current configuration; keep the skill modality-extensible (image/first-frame is a subset).

> **The doc EXPLAINS, this skill OPERATES.** `docs/inpaint-data-prep.md` is the human tutorial that
> walks the flow and inlines the MASK-SPEC values with the *why*. Do NOT restate that prose here — this
> file is operate-order steps for an agent driving the pipeline. No duplication (D-15).

---

## 0. STATE-DETECT preamble (D-8-STATEROUTE) — REUSED VERBATIM from `training-prep` §0

**Do not fork this.** Run `training-prep` §0 exactly as written before touching an inpaint dataset:

1. **Read the memory scaffold** (`.planning/harness/`): `KNOWLEDGE.md` + `DECISION-LOG.md`. Search by
   bare keyword (`rg -i "mask|inpaint|sam3|polarity|coverage|chroma" .planning/harness/`). Retrieve the
   standing answer before reasoning from priors — an untagged model prior is the documented drift
   failure mode.
2. **Detect real dataset state** — do NOT assume. Check what already exists on disk:
   `masks_frame0/` seeds? `masks_video/<stem>/` per-frame stacks? a staged `metadata.jsonl`? an encoded
   `latents/`+`conditions/` under `<preprocessed_data_root>` (`modal volume ls signe-trainer-dataset
   <root>`)? Encoded → skip to hand-off to the run skill. Partial → **name exactly what is missing**.
3. **Proceed / name-missing / hand-off** — never silently guess; a broken/invalid config stops here,
   do not burn a gate on it.

**Inpaint addendum to STATE-DETECT:** the mask pipeline is stage-detectable by artifact:
`masks_frame0/` (seeds exist) → `masks_video/<stem>/` (propagated) → `overlays_video/*_overlay.mp4`
(QA-ready, gate NOT yet passed) → staged `metadata.jsonl` (gate passed, ready to encode). Name the
stage; do not re-run a completed stage.

---

## 1. Philosophy + source-tag preamble (D-8-CANON / D-8-SOURCETAG) — same discipline as `training-prep` §1

Inpaint prep decisions are **philosophy territory** (they change *what the model learns*): house rules
first, ask when uncovered, community only via the research loop. Carry `training-prep` §1 verbatim:

- Every recommendation you log carries a **source tag**: `[house]` · `[precedent]` · `[canonical]` ·
  `[community]`. Untagged = a violation (the drift failure mode).
- **Klippbok is the caption gold standard**; the anchor is a natural name, never a trigger token. If
  Klippbok tooling is absent, mark caption-audit steps **`[PENDING: operator]`** — never hand-roll a linter.
- Inpaint-specific: captions default to the honest **`[PENDING: operator]`** placeholder at staging
  (`prep.stage` does this) — never invent inpaint captions.

**The mask conventions are DATA, not prose.** They live in `src/signet_trainer/harness_data/MASK-SPEC.yaml` (Plan 01)
and are drift-fenced into `KNOWLEDGE.md` (the `MASK-SPEC:BEGIN/END` block, distinct from `TYPED-STATE`).
Never hardcode hex / thresholds / coverage / tolerance / dims — read them via
`signet_trainer.prep.spec.load_mask_spec(path)`. Never hand-edit the fenced `MASK-SPEC` block in
KNOWLEDGE.md (a hand-edit fails `python -m signet_trainer.harness_state check`; re-emit instead).

---

## 2. READ THE SPEC first — `MASK-SPEC.yaml` is the single source of truth (Plan 01)

Before any mask step, load the spec and let it drive every constant:

```
PYTHONPATH=src PYTHONUTF8=1 python -c "from signet_trainer.prep.spec import load_mask_spec; \
  import json; print(json.dumps(load_mask_spec('src/signet_trainer/harness_data/MASK-SPEC.yaml'), indent=2))"
```

What it carries (all tunable-per-campaign, all read from the file — the *why* is in the doc):

- **`hex: "#458070"` / `rgb: [69,128,112]`** — the canonical teal brush (the colour actually painted in round 1).
- **`chroma`** — `channel_relationship` default (BGR channel-rule constants) + `distance` alternate + `tolerance`.
- **`thresholds`** — `seed_load: 127` (`>127` binarize a loaded seed PNG) / `encode_binarize: 0.5`.
- **`polarity.chain`** — paint=WHITE(255) → stage `negate` → mask mp4 region=BLACK(0) → encode `>0.5`
  → tensor `1.0=KEEP/context`, `0.0=GENERATE`. **Inverting any link is catastrophic and silent until render.**
- **`coverage_bands`** — per class (`face` / `face_hair` / `full_body`) `target` + `warn_below` + a
  `dilation` block (`max_margin_px`, `grow_to_target`) — the D-02 auto-extend knob propagate reads.
- **`dims`** — `spatial_divisor: 64` + `frame_rule: "8n+1"` (STRICTER than the general ÷32; see §7).

---

## 3. MASK CREATION — two paths, both land binary WHITE(255) seeds

Masks are **binary by construction** (region=WHITE(255), D-05 polarity), never trusted from the browser.

**(a) Hand-draw app (D-06, proven)** — launch the stdlib web-canvas masking app (127.0.0.1 ONLY):

```
PYTHONPATH=src PYTHONUTF8=1 python scripts/serve_mask_app.py <clips-dir> \
    --output <prep-out-dir> --spec src/signet_trainer/harness_data/MASK-SPEC.yaml [--port 8030]
```

- Binds `127.0.0.1` ALWAYS (house rule — never `0.0.0.0`). iPad / Apple-Pencil access is opt-in via
  `scripts/_tailnet_relay.py` ONLY, never a direct non-loopback bind.
- The **binary/WHITE export guarantee is server-side** (`alpha>127 → 255`) — browser anti-aliasing can
  never leak grey edges. Exports land as `masks_frame0/<stem>.png` seed + `masks_video/<stem>/NNNNN.png`.
- The app also carries the review/QA-overlay mode (§4) with an `Approve & stage` gate.

**(b) External paintover (D-10, iPad/Procreate teal)** — extract the teal `#458070` paintover to a seed:

```
PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_chroma.py \
    --paintovers-dir <dir> --masks-out masks_frame0 --record-out img_match.json \
    --spec src/signet_trainer/harness_data/MASK-SPEC.yaml [--dry-run]
```

- `--dry-run` self-checks resolution/naming with no cv2/spec load, no writes. Real runs print
  `CHROMA_EXTRACT_DONE: <n> seeds` and write WHITE(255) seeds + an `img_match.json`-shaped record.
  (`resid` in that record is a documented approximation of the r1 halo, not bit-exact — the mask IoU is
  what matters; round-trips a surviving r1 seed at IoU 0.9962.) `[precedent]`

---

## 4. SAM PROPAGATION — sam3 is the DEFAULT (D-14), with fix-it-first on unavailability

Propagate the frame-0 seed across all frames. The D-02 dilation is applied **inside** this step (post-SAM,
pre-QA) so the gate judges the EXTENDED mask (exact-SAM ~4.8% HURT likeness in r1; extended ~14% won).

```
PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_propagate.py \
    --backend sam3 --masks-dir masks_frame0 --clips-dir <clips> \
    --masks-out masks_video --overlays-out overlays_video \
    --spec src/signet_trainer/harness_data/MASK-SPEC.yaml [--rev <stem>] [--only <stem>] [--dry-run]
```

- **`--backend sam3` is the DEFAULT** — it auto-resolves the Plan-02 install: standalone
  `facebookresearch/sam3` on Linux, else the **transformers>=5.5 native tracker-video** path (the
  resolved backend on this Windows box; `SAM3_BACKEND_RESOLVED: transformers mask_seed=yes`). Weights are
  a ~7 GB gated `facebook/sam3` HF pull into the local cache.
- **`--rev <stem>`** re-seeds a drifted clip backward (D-08); **`--only <stem>`** limits the run.
- **`--dry-run`** resolves seeds with NO model load (torch/sam3/transformers stay unimported). Real runs
  print `SAM_PROPAGATE_DONE` + per-clip post-dilation avg_cover and emit the QA overlay mp4s.
- **D-14 fix-it-first on SAM3 unavailability** — the CLI surfaces the fix path FIRST: **HF gated access →
  install transformers → weights auto-pull**, NOT an immediate SAM2.1 fallback. `--backend sam21` (needs
  `--checkpoint sam2.1_hiera_large.pt`) is the **last-ditch** fallback only.
- **VRAM (real production-res clips):** transformers session offloads video+state to CPU
  (`--storage-device cpu --state-device cpu`, the defaults) per the Plan-02 discipline.
- **Status honesty:** the transformers propagation has NOT yet run on real GPU production clips — that is
  Plan 08's stress leg. Do not report it as proven on real footage.
- Held-out **test videos** (for eval) prep via `scripts/prep_inpaint_testvideos.py`
  (mediapipe→haar face-seed + the same ÷64/8n+1 preflight).

---

## 5. D-04 QA HARD GATE — serve + tunnel the overlay grid, STOP for operator approval

**This is a hard gate, mirroring the metered-run gate — no staging/encode happens before approval.**

- **Always serve the green-overlay QA grid LIVE + tunnel it** for the operator (house rule — never just
  a file path). Use the app's review/QA-overlay mode (the `Approve & stage` surface) or serve the
  `overlays_video/*_overlay.mp4` set; surface per-clip avg_cover against the spec `coverage_bands`
  (`warn_below` flags a thin mask).
- **STOP for the operator's explicit approval** before §6 staging or §7 encode. Reject → re-draw (§3) /
  re-propagate with `--rev` (§4) / re-dilate. Do not proceed on a thin or drifted mask.
- Test-subject difficulty should span EASY→HARD (angle-variant / linework / motion) — a mask set that
  only QAs the easy clips only proves the easy case. `[house]`

---

## 6. STAGE — mux + polarity mask + 8n+1 trim → `metadata.jsonl` (Plan 04, D-13)

```
PYTHONPATH=src PYTHONUTF8=1 python scripts/prep_inpaint_stage.py \
    --mask-manifest <prep>/mask_manifest.json --stem-manifest <prep>/manifest.txt \
    --masks-video <prep>/masks_video --seeds-dir <prep>/masks_frame0 \
    --captions <prep>/inpaint_captions.json --clips-dir <prep> \
    --spec src/signet_trainer/harness_data/MASK-SPEC.yaml --dest <staging-dir> [--dry-run]
```

- One row per mask; the polarity **negate goes through the committed `mask_encode.render_mask_video`
  seam** (never re-inline a `-vf negate` filter — D-13).
- **Frames auto-TRIM to the largest 8n+1 (never pad)**; only the **÷64 spatial rule hard-raises** (a crop
  changes content, never silent). Both rules are surfaced by the preflight report.
- Captions default to **`[PENDING: operator]`**. `--dry-run` resolves every pairing with no ffmpeg/writes;
  real runs print `STAGE_DONE: <n> rows -> <manifest>`.

---

## 7. ENCODE — drive `--mode preprocess` through the SAME gate (D-8-PREPROC) — REUSED VERBATIM from `training-prep` §2

**Do not fork the gate.** The inpaint encode is the SAME first-class gated mode on the canonical
entrypoint — there is NO bespoke encode path (the retired `scripts/_encode_*.py` are the anti-pattern):

```
PYTHONPATH=src PYTHONUTF8=1 modal run -m signet_trainer.modal.entrypoint \
    --config <yaml> --mode preprocess [--approve]
```

- The gate runs **load → dry-run → cost print → BLOCKING approval → dispatch**, identical to train. The
  **cost print + approval precede dispatch** — a metered encode can never auto-launch. Under an active
  blanket the approval stop is removed but the cost line still logs (accounting always runs).
- **Config-first (D-NOHARDCODE):** the dataset encode params come from `cfg.data` (`metadata_path`,
  `resolution_buckets`, `preprocessed_data_root`); the inpaint mask column (`video_masks`) is the
  additional per-mode surface — confirm it is set BEFORE dispatch (schema is `extra="forbid"`; reading it
  off the wrong block raises *after* approval, a burned gate). It runs the canonical `process_dataset.py`,
  never a custom encoder (the enochiatron landmine).

**Recipe rules bind here too (`training-prep` §7):** buckets at native scale, no silent resolution-shrink;
VRAM headroom is a `blocks_to_swap` problem (a Tier-2 always-check-in), not a reason to shrink resolution.
Inpaint adds the **STRICTER ÷64 spatial** rule (vs the general ÷32) — surfaced at prep by
`prep.preflight.check_inpaint_rules` / `prep.stage.assert_stage_dims`, both spec-sourced.

---

## 8. LOG the decision — required output (D-8-RESEARCH-LOOP)

Logging the real inpaint-prep decision is a **required output**, not optional — the write-back forcing
function that keeps the log from going stale:

- Append a dated entry to `.planning/harness/DECISION-LOG.md` (four fields: `what` / `why` /
  `source-tags` / `run-refs`) for any mask/coverage/backend/staging decision, preference, or surprise.
- Any **uncovered knob** (no stored house answer) runs the shared research protocol
  (`.planning/harness/README.md` §Research protocol): KNOWLEDGE.md by keyword FIRST → canonical + community
  → STORE into KNOWLEDGE.md with source tags → present options with explanations (a `[community]` default
  is never a silent override) → auto-log to DECISION-LOG.
- **Typed-state pointer:** load-bearing numerics (row counts, avg_cover per class, trim/bucket numbers,
  encode $) land as `name: value` slots in the card typed block / `- state:` log entry, never prose-only
  (`.planning/harness/COMPACTION-PROTOCOL.md`). `[house]`

---

## 9. LANDMINES the inpaint-prep skill MUST carry

- **Polarity is silent-until-render.** paint=WHITE(255) → negate → mask mp4 BLACK(0) → `1.0=KEEP`,
  `0.0=GENERATE`. Never re-inline the negate; go through `render_mask_video` (D-13). `[canonical]`
- **sam3 is the DEFAULT; SAM 2.1 is last-ditch only.** On unavailability, surface fix-it-first (HF access
  → install transformers → weights), never an immediate 2.1 fallback (D-14). `[house]`
- **Never hardcode mask constants** — hex / thresholds / coverage / tolerance / dims come from
  `MASK-SPEC.yaml` via `load_mask_spec` (D-03). Never hand-edit the fenced `MASK-SPEC` KNOWLEDGE.md block.
- **÷64 spatial hard-raises; 8n+1 auto-trims (never pads).** Cropping changes content → never silent. `[house]`
- **127.0.0.1 ONLY for the app.** Never `0.0.0.0`; tailnet exposure via `_tailnet_relay.py` only. `[house]`
- **No secrets in commands.** The HF-gated SAM3 pull uses the local cached token out-of-band (Plan 02) —
  never an inline token in a command example.
- **Single A100-80GB** for any downstream metered run; no multi-GPU / fallback GPU list (`[house]`).
- **Stale-VRAM discipline** for live GPU propagation debugging: `pkill python3` + verify 0 MiB first.
