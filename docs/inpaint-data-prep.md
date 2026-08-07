# Inpaint data prep — from source clips to an encoded masked dataset

This is the human tutorial for the signet-trainer **inpaint** data-prep pipeline. It explains *what
happens and why* end to end. It is the companion to the agent-facing `training-prep-inpaint` skill:
**this document explains, the skill operates.** When you want the exact commands and flags, drive the
skill; when you want to understand the pipeline, read here.

Why this exists: **no other trainer takes responsibility for inpaint data prep.** Turning a pile of
source clips into a correct, QA'd, encoded masked dataset — with the mask polarity, coverage, and
dimension rules all handled — is the value-add. This page is the map of that work.

> **A note on modality.** Throughout, "the mask" is really a *stack of per-frame binary masks* plus a
> single frame-0 seed. The pipeline is written against frames, not against "video" as a hardcoded
> assumption — a single-frame / first-frame inpaint is just the one-frame subset. Keep that framing in
> mind: nothing below is video-only by nature.

---

## The flow at a glance

```
source clips
   │  extract frame 0
   ▼
frame-0 seed  ── (a) hand-draw in the masking app
   │           └─ (b) external teal paintover → chroma-extract
   ▼
SAM3 propagation  (seed → every frame, then auto-extend coverage)
   │
   ▼
QA hard gate  (eyeball the green overlays — STOP for approval)
   │
   ▼
staging  (mux clip + mask, negate polarity, trim to 8n+1, write metadata.jsonl)
   │
   ▼
gated `--mode preprocess` encode  →  latents + conditions + video_masks
```

Each arrow is a step below. The conventions that make the masks *correct* — the brush colour, the
binarize thresholds, the polarity chain, the coverage targets, the dimension rules — all live in one
typed file, `src/signet_trainer/harness_data/MASK-SPEC.yaml`, and are inlined in the relevant step here so you can see
the *why*. Nothing is hardcoded in code; the scripts read that spec.

---

## Step 1 — Source clips → frame-0 seed

Start from your source clips. The masking pipeline is **seeded on frame 0**: you mark the region to be
regenerated (the inpaint region) on the first frame, and SAM propagates that region across the rest.

You produce the frame-0 seed one of two ways.

### (a) Hand-draw in the masking app

A small local web-canvas tool (stdlib `http.server`, no framework, no build step) lets you paint the
mask directly over the frame with a brush/eraser, adjustable size, zoom, and undo/redo. It binds
`127.0.0.1` only — loopback, always; if you want to draw on an iPad with an Apple Pencil, the exposure
goes through a tailnet relay, never a raw `0.0.0.0` bind.

The important guarantee: **the exported mask is binary by construction, enforced on the server.** The
browser canvas anti-aliases brush edges (soft grey pixels), but on export the server thresholds
`alpha > 127 → 255`, so the saved mask is strictly `{0, 255}` with the painted region = **WHITE (255)**.
Grey edges can never leak into the mask. (This draw path is human-verified — a real export landed a
strictly-binary seed at ~25% coverage matching the source clip's dimensions.)

### (b) External teal paintover → chroma extraction

If you'd rather paint in iPad Procreate (or any external editor), you paint the region in a specific
teal and the pipeline extracts it. The canonical brush colour is:

- **`#458070`** (RGB `[69, 128, 112]`) — teal.

Why *this* teal and not white or red? It's what was actually painted in the first real round (auto-
detected uniformly across all 13 paintovers), so it canonizes proven reality rather than inventing a
convention. White (`#FFFFFF`) is invisible over light hair and confusable with real white pixels; red
(`#F01C00`) collides with lips and clothing far more often than teal does. Teal reads cleanly over
skin, hair, and most content, and the encoder is hue-agnostic so the exact shade is a workflow choice.

Extraction separates the teal paint from content with a **channel-relationship** rule (in BGR the teal
sits at roughly `112, 128, 69` — green and blue dominate red), which is robust to the compression an
iPad export applies. A `distance` method (L2 distance to teal within a tolerance) is the documented
alternate. Either way the output is the same binary WHITE(255) seed the app produces, so the two paths
converge. (The extraction round-trips a surviving real seed at IoU 0.9962; the QA `resid` figure it
records is a documented approximation of the anti-aliased halo, not a bit-exact reproduction — the mask
itself is what's load-bearing.)

Either path lands the seed at `masks_frame0/<stem>__<type>.png`, where `<type>` is the subject class
(`face`, `face_hair`, or `full_body`) — that class selects the coverage target in Step 3.

---

## Step 2 — SAM3 propagation (seed → every frame)

One painted frame is not a dataset. **SAM3** takes the frame-0 seed and tracks the masked region across
every frame of the clip, producing the per-frame mask stack `masks_video/<stem>/NNNNN.png`.

SAM3 is the **default** segmentation backend. On this environment it resolves to the transformers
tracker-video backend (the standalone SAM3 package is non-functional on Windows because its kernel
requires triton, which has no authoritative Windows wheel); on Linux it can use the standalone package.
The weights are a gated download (~7 GB) into the local Hugging Face cache. If SAM3 is ever
unavailable, the pipeline surfaces the **fix-it-first** path — get Hugging Face gated access, install
the backend, pull the weights — rather than silently dropping to the older SAM 2.1, which is a
last-ditch fallback only.

> **Honest status:** the transformers propagation has been validated on synthetic seeds and unit
> coverage, but has **not yet been run on real GPU footage** at production resolution — that stress run is a
> later step. Treat propagation as wired-and-tested, not yet proven on real clips.

For a clip where the mask drifts, you can re-seed it **backward** from a later frame. Backward re-seed
and per-clip selection are both options on the propagation step.

---

## Step 3 — Auto-extend the coverage (D-02), then QA

A subtle but decisive finding: an **exact** SAM mask (tight to the segmented region, ~4.8% of frame)
actually *hurt* likeness in the first round, while an **extended** mask (~14%) won on the hair-heavy
classes. So propagation doesn't stop at the exact mask — it **dilates** each mask toward a per-class
coverage target *before* anything reaches the QA gate, so you always judge the mask you'll actually
train on.

The targets are per subject class (all tunable per campaign, none hardcoded):

| Class       | Target coverage | Warn below | Max dilation margin |
|-------------|-----------------|------------|---------------------|
| `face`      | 8%              | 4%         | 12 px               |
| `face_hair` | 14%             | 8%         | 28 px               |
| `full_body` | 14%             | 8%         | 32 px               |

Each mask grows one step at a time until it reaches its class `target` *or* hits the `max_margin_px`
ceiling, whichever comes first. `warn_below` is the soft band that flags a suspiciously thin mask at QA.

The binarize thresholds along the way come from the spec too: a loaded seed PNG binarizes at `> 127`
(grayscale), and the final encode binarizes at `> 0.5`.

---

## Step 4 — The QA hard gate (eyeball before you commit)

**Nothing is staged or encoded until a human approves the masks.** This mirrors the metered-run gate:
propagation renders a **green-overlay QA video** per clip (the mask drawn over the footage), and those
overlays are **served live and tunnelled** for review — never handed over as a bare file path.

You look for: does the overlay cover the right region on every frame, does it drift, is coverage in
band (the per-class `warn_below` flags thin masks)? A good QA set spans **easy and hard** clips — if you
only check the easy, straight-on clips you've only proven the easy case; the angle-variant, linework,
and motion clips are where masks fail. Reject → redraw, re-propagate (backward re-seed), or re-dilate,
then re-QA. Approve → proceed to staging.

---

## Step 5 — Staging (mux, negate, trim)

Staging turns the approved clip + mask stack into training rows and a `metadata.jsonl` manifest. Three
things happen here that matter:

**The polarity chain.** This is the single most catastrophic thing to get wrong, because inverting any
link is silent until you see a render come out backward. The chain, pinned as spec law:

```
paint region = WHITE(255)          ← what you painted / SAM produced
        │  staging applies -vf negate
        ▼
mask mp4 region = BLACK(0)
        │  encode binarizes > 0.5
        ▼
tensor 1.0 = KEEP (context)   ·   tensor 0.0 = GENERATE (the masked region)
```

So the region you *painted* becomes the region the model *generates*, and everything you left unpainted
is kept as context. The negate is applied through a single committed, tested rendering seam — never a
re-inlined filter — precisely so the chain can't drift.

**Dimension rules — stricter than usual.** Inpaint uses `÷64` spatial alignment (stricter than the
general `÷32`) and the `8n+1` frame rule (`(frames − 1) % 8 == 0`). Frame counts that don't fit
**auto-trim to the largest valid `8n+1` — never pad**. A non-`÷64` spatial size **hard-raises** with
crop guidance rather than silently cropping, because a crop changes content and must never happen
without you knowing. Both rules are surfaced by the prep preflight, reading `spatial_divisor: 64` and
`frame_rule: "8n+1"` straight from the spec.

**Captions.** Inpaint captions default to an honest `[PENDING: operator]` placeholder — they are never
invented. (Captioning follows the house Klippbok methodology when you fill them in: caption the action
and setting, anchor on a natural name, not a trigger token.)

Staging writes one row per mask into `metadata.jsonl` — the manifest the encode consumes.

---

## Step 6 — The gated encode (`--mode preprocess`)

Finally the staged dataset is pre-encoded to latents through the **same single gate every metered run
uses** — there is no bespoke inpaint encoder:

```
PYTHONPATH=src PYTHONUTF8=1 modal run -m signet_trainer.modal.entrypoint \
    --config <yaml> --mode preprocess [--approve]
```

The gate runs load → dry-run → **cost print → blocking approval** → dispatch, so a metered encode can
never auto-launch. Everything the encode needs is config-first: the dataset params (`metadata_path`,
`resolution_buckets`, `preprocessed_data_root`) come from `cfg.data`, and the inpaint mask column
(`video_masks`) is the additional surface to confirm before dispatch. It runs the canonical dataset
encoder, not a custom one.

The output — `latents/`, `conditions/`, and the `video_masks` — is a fully-encoded inpaint dataset,
ready to hand off to a gated training run.

---

## Where the rules live

Every convention above — the teal hex, the thresholds, the polarity chain, the coverage bands and
dilation, the `÷64`/`8n+1` dims — is stored as **typed data** in `src/signet_trainer/harness_data/MASK-SPEC.yaml`, and
mirrored into a drift-fenced block in `KNOWLEDGE.md`. The scripts read the spec; nothing is hardcoded.
If you need to change a convention (a different brush colour, a wider coverage target for a campaign),
change it there — never in code, and never by hand-editing the fenced KNOWLEDGE.md block (that block is
emitted from the spec and a hand-edit fails the drift check).

To *operate* the pipeline — the exact commands, dry-run flags, and completion markers for each step —
use the `training-prep-inpaint` skill. This document is the explanation; the skill is the driver.
