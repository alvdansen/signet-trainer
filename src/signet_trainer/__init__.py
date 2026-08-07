"""signet_trainer — best-of-breed video LoRA trainer (LTX-2.3, Modal).

Top-level package. Kept import-light on purpose: importing ``signet_trainer`` must NOT pull in
``modal``, ``ltx_core``, ``ltx_trainer``, or CUDA so the local dry-run runs free on Windows/CI.
Only ``signet_trainer.modal`` imports Modal (Anti-Pattern 6).
"""

__version__ = "0.1.0"
