# tests/fixtures — committed ground-truth artifacts

Small, hand-checkable files that the CPU contract tests assert against. Nothing here is
generated at test time: each file is a **transcription of something that was read off real
weights or real upstream source**, so a silent edit shows up in `git diff` rather than
quietly retiring a gate.

| File | Provenance |
|---|---|
| `h3_transformer_ref_config.json` | `MiniMaxAI/MiniMax-H3`, diffusers-format **`transformer_ref/config.json`** (the Ref2VA partition), read 2026-08-05 and recorded verbatim in `.planning/phases/10-.../P10-0d-REF2VA-PARTITION.md` (§2). Cross-confirmed field-for-field on live weights by `scripts/_h3_probe_modal.py` (`h3_arch_probe`, 10/10 constants vs `model.config` + `named_modules`, single A100-80GB, diffusers pinned at `9f169d98d0bce392a889c3b6524d0d97734dfc0e`). It is the file `src/signet_trainer/models/h3_loader.py`'s `EXPECTED_H3_*` constants are diffed against by `tests/test_h3_arch_constants.py`. |

**Do not "correct" a value here.** These are measurements, not preferences — a hand-tweaked
number silently retires the arch gate it exists to enforce (T-10-01-T). If a value looks
wrong, re-read the weights and update the provenance row in the same commit.
