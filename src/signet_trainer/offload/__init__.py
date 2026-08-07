"""offload — block-swap offloader (harvested, treated as UNVALIDATED), filled in Phase 3.

``BlockSwapOffloader`` is a VERBATIM harvest from ``flimmer-trainer-dev`` (OFFL-01),
musubi-derived and Wan-validated NOT LTX-validated. It carries the unsolved
BSWP-TRAINING-01 training-corruption bug and MUST be A/B re-validated (OFFL-03)
before it is trusted in training. The baseline-first first run uses
``blocks_to_swap=0`` (constructor early-return → inert, zero hooks).
"""

from signet_trainer.offload.block_swap import BlockSwapOffloader

__all__ = ["BlockSwapOffloader"]
