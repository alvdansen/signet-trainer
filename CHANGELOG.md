# Changelog

All notable changes to signet-trainer are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project does **not** follow semantic
versioning during beta — the config schema and CLI surface may change in any release.

## [0.1.0b1] — 2026-08-07

First private beta cut. Published with fresh history; the full development record is retained
privately.

### Model families

- **LTX-2.3 (22B)** — loader, canonical pre-encode, flow-matching training loop, checkpoint/resume,
  base-vs-adapter sampling.
- **MiniMax-H3 (Ref2VA)** — signet-native pre-encode, packed-sequence training step, reference
  geometry and RoPE position construction, base-vs-adapter renders. Routed by `model.family` inside
  the existing modes rather than as a separate mode.

### Reference control

`conditioning.mode` selects the strategy; all modes share one training loop and one gate.

- `single_frame` — first-frame / single-reference conditioning.
- `multi_frame` — multi-keyframe conditioning.
- `ic_lora` — in-context video-to-video conditioning (IC-LoRA).
- `inpaint` — masked-region spatial inpainting, with signet-native mask encoding.
- `audio_to_video` — frozen-audio-driven video generation.
- `none` — plain text-to-video LoRA.

### Modal integration

- One gated entrypoint for all six modes (`train`, `sample`, `preprocess`, `fuse`, `restore`,
  `backup`). Cost print and blocking approval precede every dispatch; `.spawn()` sits strictly after
  the pause, so no metered run can auto-launch.
- Asynchronous dispatch with a bounded watch window, then clean disengage — re-attachable via
  `modal.FunctionCall.from_id`.
- Three separate persisted Volumes (weights / dataset / checkpoints). Base weights are never baked
  into the image.
- Config-driven secret **names** with env-var overrides, so the app is portable across accounts.
- Per-run cost guardrail, cumulative session cap with an append-only spend ledger, and blanket-grant
  support.
- Pinned supply chain: `ltx-core` / `ltx-trainer` from the Lightricks git repo at a fixed commit,
  `diffusers` from git at a fixed SHA, `transformers` at an exact version.

### Data preparation

- Dataset staging, chroma-keyed mask seeding, SAM-based mask propagation, dilation, QA overlay
  export, text-seeded segmentation, and a browser mask-review app.

### Training mechanics

- PEFT LoRA inject / freeze / save / load, including frozen-adapter exclusion from committed
  checkpoints.
- **PEFT-LoRA-aware block-swap offloader.** The harvested offloader tracked only the PEFT wrapper's
  proxied base weight and skipped every adapter's `lora_A` / `lora_B`, so a late-loaded frozen
  adapter's sub-linear ran on CPU while its block ran on GPU (`mat2 is on cpu`). Weight enumeration
  now descends into PEFT wrappers; only large base weights are swapped and LoRA sub-linears stay
  GPU-resident. Re-validated for **freeze mechanics** on LTX-2.3, single A100-80GB,
  `blocks_to_swap: 16`. Long-run adapter *quality* under swapping remains unvalidated.
- CPU dry-run shape gate (`signet-dryrun`) that runs GPU-free and Modal-free.
- Checkpoint backup and restore to HF, local, or cloud destinations.

### Known limitations

- Single-GPU only. Multi-GPU and GPU fallback lists are not supported.
- `inpaint` and `audio_to_video` are the least-exercised reference-control modes.
- Parts of the test suite require setup or private files — see `CONTRIBUTING.md`.

[0.1.0b1]: https://promptcrafted.com
