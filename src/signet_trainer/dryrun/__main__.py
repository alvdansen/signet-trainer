"""``python -m signet_trainer.dryrun <config.yaml>`` — the D-13 single-command gate, second form.

Delegates to ``shapes.main`` so both invocation forms (console script + ``-m``) behave identically.
"""

from __future__ import annotations

import sys

from signet_trainer.dryrun.shapes import main

if __name__ == "__main__":
    sys.exit(main())
