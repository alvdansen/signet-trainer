"""Local (off-Modal) training -- **BETA, UNTESTED on real weights**.

This package runs the SAME training machinery the Modal path runs (``train/loop.py``'s
``train_loop`` with ``checkpoints_vol=None`` -- the off-Modal case it was designed for;
``models/loader.py``; ``train/validate_gate.py``; ``offload/block_swap.py``) on a local GPU,
for users who have a workstation card rather than Modal credits.

WARNING: **Read this before trusting it:**

* **No end-to-end run has been performed on real weights.** The mechanics are unit-tested on
  CPU with toy modules; the full 22B path is exactly what the beta label means. Issue reports
  are the point -- file them: https://github.com/alvdansen/signet-trainer/issues
* The staged local-training roadmap (what is supported when, and why) is issue #25.
* Scope: **LTX family only**, conditioning modes ``none`` / ``single_frame`` / ``multi_frame``.
  Everything else (h3, qwen families; ``ic_lora`` / ``inpaint`` / ``audio_to_video`` modes;
  frozen-adapter stacking; in-loop sampling) is refused at startup with a pointer -- a loud
  refusal beats a silent half-configured run.
* The offloader (``blocks_to_swap``) is the enabling knob for <80 GB cards, and its long-run
  adapter QUALITY is an open validation item (OFFL-02) even on the Modal path. On a 48 GB card
  expect to need deep swapping; on a 24 GB card the 22B does not fit at all today.

This package never imports ``modal`` and never dispatches metered work. It keeps the gate
DISCIPLINE anyway: config -> dry-run shape gate -> plan print -> explicit ``--approve``.
"""

from __future__ import annotations

ISSUES_URL = "https://github.com/alvdansen/signet-trainer/issues"
ROADMAP_ISSUE = "https://github.com/alvdansen/signet-trainer/issues/25"

BETA_BANNER = f"""
==============================================================================================
  !!  LOCAL TRAINING -- BETA / UNTESTED  !!

  This path shares its training loop with the validated Modal path, but NO end-to-end LOCAL
  run on real weights has been performed yet. You are the test. Expect rough edges around
  VRAM fit, weight paths and speed -- and please file EVERYTHING you hit:

      {ISSUES_URL}

  Roadmap + what is deliberately not supported yet: {ROADMAP_ISSUE}
  Local support is being tackled next; tickets filed now directly shape it.
==============================================================================================
"""
