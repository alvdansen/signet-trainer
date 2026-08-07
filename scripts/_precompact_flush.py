"""PreCompact hook: flush typed state to disk + journal the compaction (typed-state derisk).

Anthropic's context autocompaction is a PROSE summarizer we don't control — exactly the
failure mode of "Typed State Beats Prose" (36.4% numeric corruption under cascaded prose
compaction). This hook fires BEFORE every compaction (auto AND manual) and derisks it:

  1. FLUSH — run the typed-state emitter (``signet_trainer.harness_state emit``) so every
     TYPED STATE block on disk is regenerated from its typed sources at the moment of
     compaction. Whatever the summarizer garbles, the disk is fresh and authoritative.
  2. JOURNAL — append a timestamped line to ``.planning/harness/COMPACTION-JOURNAL.log``
     so compactions stop being invisible (an audit trail of when context was rewritten).
  3. STEER — print a pointer the post-compact context will carry: numerics come from the
     typed artifacts on disk, never from the summary.

Defensive by design: the emitter may not exist yet (it lands with the TS-01 build) and a
hook must NEVER break the session — every path exits 0.
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    # (2) JOURNAL — compactions leave evidence.
    try:
        journal = REPO / ".planning" / "harness" / "COMPACTION-JOURNAL.log"
        with journal.open("a", encoding="utf-8") as f:
            f.write(f"{stamp} compaction fired (PreCompact hook; trigger via stdin payload)\n")
    except OSError:
        pass

    # (1) FLUSH — regenerate every TYPED STATE block from its typed sources.
    emitter = REPO / "src" / "signet_trainer" / "harness_state.py"
    if emitter.exists():
        env = dict(os.environ, PYTHONPATH="src", PYTHONUTF8="1")
        try:
            r = subprocess.run(
                [sys.executable, "-m", "signet_trainer.harness_state", "emit"],
                cwd=str(REPO), env=env, capture_output=True, text=True, timeout=60,
            )
            status = "flushed" if r.returncode == 0 else f"emitter rc={r.returncode}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            status = f"emitter skipped ({type(exc).__name__})"
    else:
        status = "emitter not yet installed (TS-01 pending) — journal-only"

    # (3) STEER — this line rides into the post-compact context.
    print(
        f"[precompact] typed-state {status}. COMPACTION LAW (typed-state-beats-prose): "
        "after this summary, NUMERICS COME FROM DISK, not from the summary — ledger/cap from "
        ".planning/harness/SESSION-STATE.json (read_ledger); campaign state from "
        ".planning/harness/cards/*.state.yaml + the TYPED STATE blocks; decisions from "
        "DECISION-LOG.md 'state:' slots. Verify any remembered number against these before use."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
