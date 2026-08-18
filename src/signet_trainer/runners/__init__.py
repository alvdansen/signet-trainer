"""runners — translations of the signet MANIFEST into a foreign trainer's own dialect.

"One driver, N runners, one manifest." signet owns the manifest, the gates, the cost ledger and
reproducibility; a RUNNER executes the optimizer step. ``musubi-tuner`` is a runner for Wan
exactly as ``ai-toolkit`` is a runner for Qwen. Multi-source composition therefore belongs in the
MANIFEST layer (``config/sources.py``), and each runner translates it: a TOML for musubi, dataset
blocks for ai-toolkit, precomputed roots for the signet-native families.

Every module in this package is PURE — no ``modal``, no ``torch``, no filesystem, no network — so
a translation can be asserted byte-for-byte against a real, known-good runner config on CPU/CI
before anything is metered.
"""
