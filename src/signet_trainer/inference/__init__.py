"""inference — validation sampling / inference (ValidationSampler), Phase 4 (INFR-01).

Modules keep the repo's Anti-Pattern-6 import-confinement discipline: heavy
``ltx_trainer`` / ``ltx_core`` / ``modal`` imports go function-local so the CPU/Windows/CI
Wave-0 gates (param mapping, frame alignment, no-Wan-params scan) import these modules
without a GPU or the ltx stack installed.
"""
