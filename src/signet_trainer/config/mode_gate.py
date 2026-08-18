"""Mode-conditional config refusals -- ONE CPU-pure home (WR-04 hoist, audit gap-dryrun-ltx-0).

The two multi_frame refusals used to live ONLY inside the metered container bodies
(``modal/fns.py`` ``train()`` / ``sample()``), so the free gates -- ``signet-dryrun`` and the
entrypoint's pre-dispatch ``run_dryrun`` call -- were mode-blind: they PASSED the shipped
multi_frame sample config that ``train()`` then refused at load, POST-approval, on a metered A100.
Worse, the dispatch is detached (``.spawn`` + a bounded ``dispatch_watch_seconds`` window), so with
train's retry policy pending the raise never surfaced at the operator's console -- the burned
approval played out invisibly over hours of retry backoff. Hoisting the predicates here gives them
one home shared by the dry-run CLI (``--mode``), the entrypoint (``run_dryrun(cfg, mode=...)``)
and the container bodies -- the refusal fires PRE-dispatch, on CPU, for free.

No ``modal`` / torch import -- pure config predicates (Pitfall 4 / Windows).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # typing only -- a runtime import would be a schema<->gate cycle risk for nothing
    from signet_trainer.config.schema import SignetConfig

#: The entrypoint's ``--mode`` vocabulary. The entrypoint keeps its own LITERAL tuple (two
#: structural tests AST-pin it there); this mirror serves the dry-run CLI's optional ``--mode``
#: flag and the unknown-mode refusal below.
KNOWN_MODES: tuple[str, ...] = ("train", "sample", "preprocess", "fuse", "restore", "backup")


def validate_mode_config(cfg: SignetConfig, mode: str | None) -> None:
    """Raise ``ValueError`` on a config that is illegal FOR THE GIVEN MODE (CPU-pure, zero spend).

    ``mode=None`` (a bare ``signet-dryrun <config.yaml>``) keeps the historical mode-agnostic
    behaviour: no mode-conditional refusal can fire without a mode to condition on. Callers that
    KNOW the mode -- the entrypoint and the container bodies -- must pass it, so both multi_frame
    refusals fire pre-dispatch instead of on a metered container.
    """
    if mode is None:
        return
    if mode not in KNOWN_MODES:
        raise ValueError(f"unknown mode {mode!r} (expected one of: {', '.join(KNOWN_MODES)}).")

    # WR-04 (train): conditioning_items are SAMPLE-only in Phase 6 -- training is self-conditioning
    # (D-6-CONDSOURCE 'self': MultiFrameStrategy samples its own keyframe positions/strengths from
    # the clip), so a train config carrying items (e.g. the multi-frame SAMPLE example passed to
    # train by mistake) would have them silently ignored -- the exact silently-ignored-config-block
    # class the schema's field-split doctrine forbids.
    if (
        mode == "train"
        and cfg.conditioning.mode == "multi_frame"
        and cfg.conditioning.conditioning_items
    ):
        raise ValueError(
            "[train] conditioning_items are sample-only in Phase 6 (conditioning_source='self': "
            "MultiFrameStrategy samples keyframes from the clip itself); training would silently "
            "ignore them. Remove conditioning_items from the training config (see "
            "configs/ltx23_multi_frame_overfit.example.yaml) -- keyframe items belong in the "
            "sample config only."
        )

    # The mirror image (sample): a multi_frame RENDER with no keyframe items has nothing to
    # condition on -- of the two shipped multi_frame examples exactly one is train-legal and the
    # other is sample-legal, and each direction of the mix-up must be refused for free.
    if (
        mode == "sample"
        and cfg.conditioning.mode == "multi_frame"
        and not cfg.conditioning.conditioning_items
    ):
        raise ValueError(
            "[sample] conditioning.mode == 'multi_frame' but conditioning.conditioning_items is "
            "empty -- nothing to condition on. Stage the keyframe images "
            "(scripts/_stage_multi_frame_refs.py, 06-07) and list them as conditioning_items "
            "(image / frame_index / strength) in the run config."
        )
