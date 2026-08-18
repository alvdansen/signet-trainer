# signet-trainer

> **Public beta.** Interfaces, config schema, and CLI surface may change without notice.
> See [Beta status](#beta-status) before you build anything on this.

A video LoRA trainer built around **reference control as a first-class feature** rather than an
afterthought — single-frame conditioning, multi-frame / keyframe conditioning, and IC-LoRA-style
in-context (video-to-video) conditioning are configuration modes, not forks. It runs on
[Modal](https://modal.com) serverless GPU, and supports two model families: **LTX-2.3 (22B)** and
**MiniMax-H3** (Ref2VA).

Every metered run passes through exactly one gate that prints a cost estimate and blocks for
approval before it dispatches. There is no code path that starts a paid GPU without that print.

---

## Quickstart

Everything — training, sampling, pre-encoding, adapter fusion, checkpoint backup and restore —
goes through a single entrypoint:

```bash
PYTHONPATH=src PYTHONUTF8=1 \
  modal run --detach -m signet_trainer.modal.entrypoint \
  --config configs/ltx23_single_frame.example.yaml \
  --mode train \
  --approve
```

**Drop `--approve` to do a free dry run.** Without it the entrypoint loads and validates the
config, runs the CPU shape gate, prints the cost line, and then stops at the approval pause. No GPU
is provisioned and nothing is billed. That is the intended way to check a config before spending.

### The three things about this command that are not optional

1. **`--detach` is REQUIRED for any metered dispatch.** Without it, Modal tears the app shell down
   with your local client — close the laptop, lose the run. `--detach` keeps the *app* alive past
   the client.
2. **Dispatch is asynchronous (`.spawn()`, not `.remote()`).** A synchronous input is cancelled by
   the server when its client dies, so the entrypoint spawns, prints the `FunctionCall` id, watches
   for a short bounded window so cheap early failures still surface, and then disengages *without
   cancelling anything*. Re-attach later with `modal.FunctionCall.from_id("<id>")`.
   `--detach` alone is necessary but **not** sufficient; both pieces are needed.
3. **Never call a Modal function directly.** `modal run -m signet_trainer.modal.fns::train` (or any
   `.spawn()` / `.remote()` on a function handle) boots a paid GPU with **no cost print and no
   approval pause**. Always drive `signet_trainer.modal.entrypoint`.

### Modes

| `--mode` | GPU? | What it does |
| --- | --- | --- |
| `preprocess` | yes | Pre-encode the dataset to latents + text embeddings on the dataset Volume, so training never re-encodes |
| `train` | yes | The LoRA training loop; checkpoints committed to the checkpoints Volume |
| `sample` | yes | Base-vs-adapter inference at a fixed seed, for side-by-side comparison |
| `fuse` | no (CPU) | Fuse a gated upstream adapter into a base checkpoint (needed once before inpaint training) |
| `backup` | no (CPU) | Copy checkpoints off the Volume to a backup destination |
| `restore` | no (CPU) | Restore checkpoints back onto the Volume |

The mode set does **not** change per model family or per reference-control mode. A config carrying
`model.family: h3` makes `train` / `sample` / `preprocess` dispatch the H3 arm of that same mode;
the reference-control mode comes from `conditioning.mode` in the YAML
(`none`, `single_frame`, `multi_frame`, `ic_lora`, `inpaint`, `audio_to_video`).

### Local, free checks

```bash
pip install -e ".[dev]"          # or: uv pip install -e ".[dev]"
python -m pytest                 # CPU-only; no GPU, no Modal, no spend
signet-dryrun configs/ltx23_single_frame.example.yaml
# equivalently, without installing:
PYTHONPATH=src python -m signet_trainer.dryrun configs/ltx23_single_frame.example.yaml
```

`signet-dryrun` loads a real run config, builds a synthetic batch through the configured strategy,
asserts shapes and masks, and exits non-zero on any violation — before a GPU is touched. The
`config/`, `conditioning/`, and `dryrun/` packages import neither `modal` nor `ltx_core`, so this
runs on a laptop (including Windows) with zero cost.

Pass `--mode <train|sample|preprocess|fuse|restore|backup>` to also fire the mode-conditional
refusals the metered container would raise at load (e.g. a multi_frame sample config dispatched as
train) — the same shared check the entrypoint runs pre-approval. For the LTX family the gate
builds and asserts the batch at **every** `data.resolution_buckets` geometry (what training
actually shapes) and reports the worst bucket's sequence length alongside `training_dims`.

> See [Known beta gaps](#known-beta-gaps) for the parts of the test suite that need extra setup.

---

## Compute venues

Two venues are implemented today: **Modal** (serverless GPU — the primary, validated path) and
**local workstation GPUs** (the BETA runner above). We plan to support others, but we prioritize
the venues we actually have access to and can validate on real hardware — that's the "ship what
works" rule applied to infrastructure. (If, say, RunPod wanted to send a compute grant our way, we
would be *delighted* to implement RunPod support properly.)

That said, the venue layer is deliberately thin and porting it is largely mechanical: every run
flows through one config-driven entrypoint, the training core never imports the venue, weights and
datasets are plain paths, and the dispatch arms are isolated in `src/signet_trainer/modal/`. An
agent — or an engineer with one — pointed at this repo should be able to convert the scripts to
another provider without touching the training machinery.

---

## The agentic harness

This repo doesn't just ship a trainer — it ships **the way we operate it**. `.claude/skills/`
contains the six lifecycle playbooks we drive every real campaign with, built for (and with)
Claude Code:

| Skill | Stage |
| --- | --- |
| `training-session-setup` | The one upfront gate: triage mode, spend cap, sampling plan, recipe surface — collected before any metered work |
| `training-prep` / `training-prep-inpaint` | Dataset staging + the gated pre-encode (inpaint adds the mask pipeline + QA gate) |
| `training-run` | Dry-run → cost print → approval → launch → monitor, through the single entrypoint only |
| `training-review` | Sample renders, side-by-side grids, the ~200-step convergence probe — judged on clips, not loss curves |
| `segmentation-prep` | SAM3 text-prompt seeding → propagation → paired control datasets |

Open this repo in Claude Code and start with `training-session-setup`; the skills carry the full
playbooks (commands, artifact paths, VRAM and cost landmines) and enforce the house rules the rest
of this README describes — the approval gate, the spend ledger, save-everything checkpoints, and
verdicts rendered on visual output. They are our in-house methodology, shipped as-is; a few
reference private companion tooling and say so where they do. You can run everything in this repo
without them — but they are how it's *meant* to be run, and they're half the reason a two-person
domain team can operate a trainer like this at all.

---

## Requirements

| | |
| --- | --- |
| **GPU** | A single **A100-80GB**. That is the target, not a floor to beat — the 22B LoRA path fits one A100-80GB using sequential model loading plus the block-swap offloader. H100/H200 is an optional speedup, never a requirement. Multi-GPU is not supported. |
| **Python** | **3.11+** (see `requires-python` in `pyproject.toml`) |
| **Modal** | Your own [Modal](https://modal.com) account, with the CLI authenticated (`modal setup`). Confirm the active profile is yours before running anything — `modal profile current`. |
| **Hugging Face** | An HF account with **accepted licenses** for every gated checkpoint you intend to load. Access is per-account and is not conveyed by this repo. |

### Modal secrets — create these on YOUR account

The Modal app graph resolves secrets **by name at import time**, so these must exist on the account
you are running under or every command fails immediately:

| Secret name | Contents | Needed for |
| --- | --- | --- |
| `my-huggingface-secret` | `HF_TOKEN` | Downloading base weights and the text encoder |
| `my-wandb-secret` | `WANDB_API_KEY` | Run logging (only if you enable `wandb`) |
| `hf-gated-secret` | `HF_TOKEN` for an account that accepted the *gated* adapter's terms | Consumed by `--mode fuse` only, but **required for every mode's dispatch** — Modal resolves the whole app graph's secrets at import time (see below), so a missing/wrong name fails `train`/`sample`/`preprocess`/`backup`/`restore` too, not just `fuse` |

The names are configurable — set `modal.huggingface_secret_name` / `modal.wandb_secret_name` /
`modal.hf_gated_secret_name` in your YAML, or export `SIGNET_HUGGINGFACE_SECRET_NAME` /
`SIGNET_WANDB_SECRET_NAME` / `SIGNET_HF_GATED_SECRET_NAME`. Configs carry secret **names** only;
token values are injected into the container at run time and are never logged or baked into the
image.

### Modal Volumes

Three separate Volumes are provisioned on first use (`create_if_missing=True`) — weights, dataset,
and checkpoints never share one:

```
signe-trainer-weights       ->  /weights       base checkpoints + text encoder, downloaded once
signe-trainer-dataset       ->  /dataset       clips + pre-encoded latents/embeddings
signe-trainer-checkpoints   ->  /checkpoints   LoRA checkpoints + rendered samples
```

> **Why do the defaults say `signe-` and not `signet-`?**
> The project was renamed to **Signet**, but these four strings (the three Volumes plus the Modal
> **App** name, default `signe-trainer`) are *not* branding — they name Modal resources that already
> hold the maintainer's live data. They are kept at their pre-rename spelling on purpose. Note that
> `Volume.from_name(..., create_if_missing=True)` does **not** error on an unknown name: it silently
> provisions a new, *empty* Volume. A "tidy-up" rename would therefore fail **silently** — training
> against empty storage — rather than loudly. Leave the defaults alone; override them instead.

#### Pointing the trainer at your own Modal account

The App name and all three Volume names are config fields, not literals. Set them in your run YAML:

```yaml
modal:
  app_name: my-trainer
  weights_volume_name: my-trainer-weights
  dataset_volume_name: my-trainer-dataset
  checkpoints_volume_name: my-trainer-checkpoints
```

⚠ `signet_trainer/modal/app.py` builds the Modal app graph at **module-import time**, before the
entrypoint's `main()` runs, so a YAML-only override cannot re-bind an already-built graph. Export
the matching env vars **in the shell before `modal run`**:

```bash
export SIGNET_APP_NAME=my-trainer
export SIGNET_WEIGHTS_VOLUME_NAME=my-trainer-weights
export SIGNET_DATASET_VOLUME_NAME=my-trainer-dataset
export SIGNET_CHECKPOINTS_VOLUME_NAME=my-trainer-checkpoints
```

The entrypoint compares what the graph captured against your config and **aborts pre-approval** on
any mismatch, so a half-applied override can never reach a metered dispatch.

Base weights are deliberately **not** baked into the container image. Download them once into the
weights Volume and every subsequent run reuses them.

---

## Layout

`src/`-layout, importable as `signet_trainer.*`:

```
src/signet_trainer/
├── config/          # Pydantic config schema, fail-fast validators, YAML load
├── conditioning/    # reference-control strategies: single-frame, multi-frame,
│                    #   IC-LoRA, inpaint, audio-to-video, + H3 packing/geometry
├── data/            # dataset files, pre-encoded latent reads, mask encoding
├── models/          # LTX-2.3 and MiniMax-H3 loaders
├── lora/            # PEFT LoRA mechanics (inject / freeze / save / load)
├── offload/         # PEFT-aware block-swap CPU<->GPU offloader
├── train/           # flow-matching training loop, checkpointing, validation gate
├── prep/            # dataset staging, pre-encode, masking, propagation, QA
├── inference/       # sampling, reference handling, grids, upscale
├── backup/          # checkpoint backup/restore planning
├── modal/           # the ONLY package that imports `modal` — app, image,
│                    #   Volumes, secrets, cost gate, session cap, entrypoint
└── dryrun/          # CPU-only shape gate
```

`config/`, `conditioning/`, and `dryrun/` are transport-agnostic on purpose: they import nothing
from `modal` or `ltx_core`, which is what makes the local dry run an honest, GPU-free, zero-spend
check.

---

## Cost discipline

Three independent mechanisms, all implemented in shipped code and all driven from your config —
none of them are house-specific and none carry anyone else's account limits:

- **Per-run guardrail** (`modal/cost.py`) — estimates `hourly_rate * est_hours` and prints it before
  the approval pause. Rates come from `modal.hourly_rate_usd` in your config, never from a literal
  in the code.
- **Cumulative session cap** (`modal/session_cap.py`) — an append-only spend ledger with a
  cumulative ceiling. `DEFAULT_SESSION_CAP_USD` is a conservative default you are expected to set
  for yourself.
- **The approval pause** (`modal/entrypoint.py`) — `.spawn()` sits strictly after it. A metered run
  cannot auto-launch.

`.claude/skills/` ships the operating playbooks that wrap these (session setup, prep, run, review).
They are the in-house methodology, shipped as-is; each one carries a beta note about the companion
tooling and private run history that are **not** part of this release.

---

## Local training (BETA — **UNTESTED**, issues wanted)

For users with a workstation GPU rather than Modal credits, there is a local runner that drives
the **same** training loop, architecture gate, and block-swap offloader as the Modal path — with
local paths and no Volume/commit machinery:

```bash
python -m signet_trainer.local --config configs/ltx23_lora.example.yaml \
    --weights-root /path/to/weights --dry-run-only    # free: refusals + shape gate + plan only
python -m signet_trainer.local --config <your.yaml> --weights-root <dir> --approve
```

**⚠ Read before relying on it:**

- **No end-to-end local run on real weights has been performed.** The mechanics are unit-tested
  on CPU; the full path is exactly what this label disclaims. **You are the test — please file
  everything you hit**, good or bad: local support is being tackled next and tickets filed now
  directly shape it. Roadmap: [#25](https://github.com/alvdansen/signet-trainer/issues/25).
- Scope today: **LTX family**, conditioning modes `none` / `single_frame` / `multi_frame`.
  Everything else (h3/qwen families, `ic_lora`/`inpaint`/`audio_to_video`, frozen-adapter
  stacking, in-loop sampling) is **refused loudly at startup** with a pointer — never half-run.
- It consumes a **pre-encoded** dataset (the `PrecomputedDataset` layout). Local pre-encoding is
  a roadmap item; today, encode via the gated Modal preprocess and `modal volume get` the result.
- VRAM reality: the 22B peaked at **62.8 GiB at `blocks_to_swap: 16` on an A100-80GB**. Smaller
  cards need deeper swapping and are **unmeasured**; a 24 GB card does not fit the 22B at all.
  The offloader's long-run adapter *quality* is itself an open item (OFFL-02) — one more reason
  this is a beta.
- The runner keeps the gate discipline (dry-run gate → plan print → explicit `--approve`) even
  though nothing is metered — the meter here is your wall-clock and an untested path.

## How this is built

We are **domain experts, not career software engineers** — this trainer comes out of a working
video-LoRA practice, and we build it in heavy collaboration with Claude Code, with close oversight
on our part at every step: every training decision, recipe, and merge is reviewed against real
runs on real GPUs. **We ship what works and label what's untested** (you'll see BETA and ALPHA
tags on exactly the surfaces that haven't earned trust yet), and we always welcome feedback from
established engineers — issues and PRs pointing out sharper ways to do things are genuinely
appreciated, not just tolerated.

---

## Beta status

This is a **public beta** cut with fresh history.

- The config schema, YAML field names, and CLI flags **will** change. Pin a commit if you need
  stability.
- The LTX-2.3 and MiniMax-H3 legs have both been exercised on real A100 hardware, but coverage is
  uneven across reference-control modes. `single_frame`, `multi_frame`, and `ic_lora` are the
  best-trodden; `inpaint` and `audio_to_video` are newer.
- The block-swap offloader was made PEFT-LoRA-aware and re-validated for **freeze mechanics**
  (two-adapter forward/backward/optimizer-step, frozen-adapter exclusion) on LTX-2.3 at
  `blocks_to_swap: 16`. Long-run adapter **quality** under swapping has not been A/B'd. Treat it as
  a scoped result, not a blanket guarantee.
- Please report what breaks. Reproduction beats description: the config, the mode, and the failing
  output.

### Known beta gaps

Two operational caveats from the most recent deep audit (fixes are staged for the next round —
tracked in issue #45):

- **The parallel-render watcher's unattended completion-detection has known gaps** (it can declare
  a long render stalled or complete at the wrong moment). For now, prefer attended renders and
  re-run the grid script manually to refresh a live grid; the artifacts themselves always commit
  to the Volume regardless.
- **The cost line prices ONE container life.** Arms that carry server-side retries can multiply
  the worst-case metered spend well past the printed estimate if a run is repeatedly preempted or
  times out. Watch long metered runs; do not fire-and-forget under a tight budget.

`python -m pytest` is **not** green out of the box, for reasons that are known rather than
mysterious. Expect roughly **40 failed, ~2766 passed, 51 skipped**, in three groups:

- **~33 — the H3 parity suites.** They deliberately refuse to skip: they must execute against
  `transformers`/`diffusers` at the exact pinned version, in a separate overlay interpreter, or the
  check would pass while the real dispatch failed. Bootstrap it once with the two commands the
  failure message prints, or leave them red. A few in this group are regression checks pinned to
  in-house run configs that are not published.
- **~6 — the campaign-fork watcher tests.** `tests/test_watcher_hardening.py`'s `test_fork_*` tests
  (and one `test_both_final_done_guarded_by_success` half) read a long-running campaign watcher
  script that is operational tooling rather than library code and is not published. The suite's
  `test_both_*` money-safety assertions (session-cap gate, append-spend-on-every-dispatch, the
  render-dispatch timeout exemption, the `CAP_STOP_EXIT` contract, the heartbeat touch) run against
  the SHIPPED `scripts/watch_parallel_inference.py` and are green — issue #19 item 1 stopped the
  unpublished fork's absence from masking that result.
- **51 skipped (not failed) — mostly the house-memory scaffold lints.** Some tests lint a project-private
  memory scaffold under `.planning/harness/` (`DECISION-LOG.md`, `KNOWLEDGE.md`, `SESSION-STATE.json`,
  campaign cards). That directory holds live per-project run history and is deliberately not
  published, so those tests skip with a reason instead of failing.

**Runtime defaults are self-contained.** The five SPEC/TEMPLATE files the runtime needs
(`MASK-SPEC.yaml`, `MASK-SPEC-segmentation.yaml`, `TIER-TAXONOMY.yaml`, `HOUSE-SPEC.yaml`,
`SESSION-STATE.template.json`) ship **inside the package** at `signet_trainer/harness_data/` and are
resolved with `importlib.resources`, so they work from an installed wheel with no `.planning/`
present. Point them somewhere else with `SIGNET_HARNESS_DATA_DIR=/path/to/dir`, or per call with
`--spec`. The only project-relative defaults left are genuinely per-project **live state** — the
spend ledger (`session_spend_ledger_path`) and the decision log — and each fails with an actionable
message naming `SIGNET_PROJECT_ROOT` rather than a stack trace.

`CONTRIBUTING.md` lists these precisely.

---

## License

Source-available under the **Signet Trainer License 1.0.0** — see [`LICENSE`](LICENSE) for the
binding text and [`NOTICE`](NOTICE) for third-party attribution. Both files must travel with any
copy you distribute. Licensing questions: **minta@promptcrafted.com**.

---

## Model weights

**This project conveys no rights in any model's weights.** You must obtain and comply with the
license of every checkpoint you load — LTX-2.3, Gemma 3, MiniMax-H3, SAM, and anything else. Some of
those terms survive training and attach to the outputs you generate.

**MiniMax-H3, specifically**, because it is easy to get wrong:

- **Excluded Territories.** The H3 license carves out Excluded Territories, which include the
  **European Union, the United Kingdom, the Republic of Korea, and the United States**. If you or
  your use falls within one, the standard grant does not reach you.
- **You need your own grant.** Any access or approval this project's authors hold is personal to
  them and **does not transfer**. Obtain your own before touching H3 weights.
- **No cross-model training (§V.3).** H3 outputs must not be used to train or improve another AI
  model — including as synthetic training data for an LTX or Wan LoRA.
- **Attribution on commercial products (§IV.2).** Commercial products or services built on H3 must
  prominently display "MiniMax H3".
- **Revenue-triggered authorization (§IV.1).** Above USD 20M/yr you need separate authorization from
  MiniMax. This is independent of the Signet Trainer License threshold; you may be subject to both.

Section numbers are pointers to help you find the clauses, not restatements. The current MiniMax-H3
license governs. See [`NOTICE`](NOTICE) for the full inventory.

---

## Credits

Training and inference for this project run on **Modal**, whose sponsored GPU credits made its
development and validation possible.

signet-trainer harvests and ports a validated foundation from `enochiatron` and `flimmer-trainer`,
whose block-swap offloader descends from **musubi-tuner** (kohya-ss). It contains ported code from
**Lightricks/LTX-2** and transcribed code from **huggingface/diffusers**. Full attribution, change
notices, and license texts are in [`NOTICE`](NOTICE) and [`LICENSES/`](LICENSES).

A particular thank-you to **ostris** and [ai-toolkit](https://github.com/ostris/ai-toolkit) — for
generous advice along the way, and for a trainer whose public example (not least its MiniMax-H3
work) repeatedly sharpened our own thinking about how these models should be trained.
