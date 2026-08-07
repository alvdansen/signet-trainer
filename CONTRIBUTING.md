# Contributing

signet-trainer is in **private beta**. The most valuable contribution right now is a precise bug
report; code contributions are welcome but the schema is still moving, so open an issue before
writing anything large.

## Reporting a problem

Include, in this order:

1. The **config** you ran (redact dataset paths and captions — see [Privacy](#privacy) below).
2. The exact **command**, including `--mode` and whether `--approve` was passed.
3. The **model family** (`model.family`) and **reference-control mode** (`conditioning.mode`).
4. The failing output. If it failed on Modal, the `FunctionCall` id and the container log tail.
5. Whether `signet-dryrun <your-config.yaml>` passes. That is free and it isolates config errors
   from runtime errors — please run it first.

## Development setup

```bash
pip install -e ".[dev]"     # or: uv pip install -e ".[dev]"
python -m pytest            # CPU-only, no GPU, no Modal, no spend
```

The test suite is deliberately runnable on a laptop, including Windows. `config/`, `conditioning/`,
and `dryrun/` import neither `modal` nor `ltx_core`; keep it that way. If you add a module that
needs a heavy backend, import it **function-locally**, not at module scope.

## Running the tests — what is expected to fail

`python -m pytest` is **not green on a fresh clone** of this beta: expect roughly **48 failed,
~1890 passed, 27 skipped**. Every failure falls into one of the known buckets below. If you hit
something outside them, that is a real bug.

### 1. Tests that lint the private house-memory scaffold — these SKIP, they do not fail

**Runtime defaults are self-contained; there is nothing to work around.** The five SPEC/TEMPLATE
files the runtime needs ship **inside the package** and are resolved with `importlib.resources`, so
they work from an installed wheel with no `.planning/` anywhere:

| Packaged file (`signet_trainer/harness_data/`) | Read by |
| --- | --- |
| `MASK-SPEC.yaml` | `harness_state.load_mask_spec`, `prep/propagate.py`, the `prep_inpaint_*` scripts |
| `MASK-SPEC-segmentation.yaml` | `prep/textseed.py` and its tests |
| `TIER-TAXONOMY.yaml` | `harness_state.load_tier_taxonomy`, `harness_lint.py` |
| `HOUSE-SPEC.yaml` | `harness_state.load_house_spec` |
| `SESSION-STATE.template.json` | the session-ledger seed template |

Override all five at once with `SIGNET_HARNESS_DATA_DIR=/path/to/dir` (a directory that is missing a
spec it overrides fails loudly rather than silently mixing two spec generations), or pass an
explicit path per call — every loader still takes one.

What stays **project-relative** is genuinely per-project **live state**, which is one project's
accumulating record and must never be baked into a wheel:

| Path | Read by |
| --- | --- |
| `.planning/harness/SESSION-STATE.json` | `config/schema.py::session_spend_ledger_path` (absent = fresh session, spend 0.0) |
| `.planning/harness/DECISION-LOG.md` | `harness_lint.py` CLI default |
| `.planning/harness/KNOWLEDGE.md` | the memory-scaffold tests |
| `.planning/harness/cards/TRAINING-CARD-*.state.yaml` | `harness_state.load_campaign_state` |

Each resolves against `project_root()` — `SIGNET_PROJECT_ROOT`, else the nearest ancestor of the CWD
carrying `.planning/harness/` — and reports an actionable message naming that env var when absent,
never a stack trace.

That directory is project-private run history and is **not published**, so the tests that lint it
are marked `requires_live_harness` and **skip** (with a reason) instead of failing. Affected:
`test_memory_scaffold.py`, `test_optimization_target.py`, `test_harness_state.py`. The shipped
SPEC/TEMPLATE data is covered instead by `test_harness_data.py`, which runs everywhere.

### 2. Tests pinned to in-house run configs

A handful of regression tests assert properties of specific configs from in-house training
campaigns, which are not published. They assert nothing about the library's behaviour and can be
deselected. Affected: `test_h3_sample_resume.py`, `test_h3_sample_render_residency.py`,
`test_h3_reference_selector.py`, `test_h3_reference_geometry_agreement.py`,
`test_h3_pipeline_local_source.py`, plus one test each in `test_watcher_hardening.py` and
`test_session_cap_ledger_runtime.py` that reference campaign watcher scripts not included here.

### 3. The H3 parity suites (these need a one-time bootstrap)

`test_h3_real_class_parity.py` and `test_h3_processor_output_parity.py` check this repo's
transcribed H3 constants against the **real** `diffusers` classes at the pinned SHA. They
deliberately **refuse to skip** — the defect class they close is "a self-authored surrogate agreed
with us and the real class did not", and a skip is that failure with extra steps.

Bootstrap the overlay interpreter once:

```bash
uv venv --system-site-packages .venv-h3parity
uv pip install --python .venv-h3parity/Scripts/python.exe --no-deps \
  'diffusers @ git+https://github.com/huggingface/diffusers@9f169d98d0bce392a889c3b6524d0d97734dfc0e'
```

`--system-site-packages` reuses your existing torch; `--no-deps` keeps the install from moving the
`transformers` pin the sibling parity gate asserts. Point `SIGNET_H3_REALCLASS_PYTHON` elsewhere if
you prefer a different interpreter. The overlay is gitignored by the existing `.venv-*/` rule.

## House rules that are not negotiable

These are invariants the tests enforce; a PR that breaks one will be rejected on principle, not
style:

- **The single gate.** Every metered run goes through `signet_trainer.modal.entrypoint`. Never add a
  path that calls `.spawn()` or `.remote()` on a function handle — that boots a paid GPU with no
  cost print and no approval pause.
- **The cost print always logs.** No mode, flag, or config may suppress it.
- **Config-first, no hardcoding.** Tunable values live in the config schema with a documented
  default. A literal hourly rate, cap, resolution, or path in code is a bug.
- **Secret names, never secret values.** Configs and logs carry the *name* of a Modal secret. Tokens
  are injected at run time and must never be printed or baked into an image.
- **No unpinned supply chain.** `ltx-core` / `ltx-trainer` are installed from the Lightricks git
  repo at `LTX2_COMMIT_SHA`, and `diffusers` from git at `DIFFUSERS_SHA`. `ltx-core` is
  name-squatted on PyPI by an unverified publisher; it must never appear as a bare dependency name.
  `tests/test_pyproject_supply_chain.py` enforces this.
- **Never delete checkpoints or intermediate artifacts** automatically. `keep_checkpoints: null` on
  long runs — a finite value silently prunes.

## Privacy

Do not put private, licensed, or client material into anything tracked: no real subject/property
names, no client brands, no dataset captions, no source filenames — not in configs, not in test
fixtures, not in commit messages, not in issue reports. Use a neutral codename. The `.gitignore`
covers the local scratch directories (`_*_source/`, `_*_stage/`, `_*_refs/`, `_*.log`, all media
extensions), but it cannot cover a name typed into a YAML field or a commit subject.

## License of contributions

By submitting a contribution you agree it is licensed under the **Signet Trainer License 1.0.0**
(see [`LICENSE`](LICENSE)), and that you have the right to license it that way. If your change
derives from third-party code, say so in the PR and add the attribution to [`NOTICE`](NOTICE) —
including an Apache-2.0 §4(b) change notice where the upstream is Apache-2.0.
