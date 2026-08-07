"""conditioning — the isolated Strategy seam (TrainingStrategy ABC + batch contract).

Locked in Plan 01-01; Phases 5-7 add reference-control subclasses here without touching the
loop/loader/offloader. Transport-agnostic: imports stdlib + torch only, never modal/ltx_core.
"""
