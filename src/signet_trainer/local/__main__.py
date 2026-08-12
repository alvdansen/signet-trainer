"""``python -m signet_trainer.local`` -- the local-training CLI (BETA / UNTESTED)."""

from __future__ import annotations

import sys

from signet_trainer.local.runner import main

if __name__ == "__main__":
    sys.exit(main())
