"""Multi-source datasets on signet's NATIVE training loop. NOT LANDED — this is the declared gap.

``DataConfig.sources`` (slice A) is honoured by exactly one family: ``wan``, whose runner
(musubi-tuner) consumes the list as a rendered ``[[datasets]]`` array and owns caching end to end.
Every signet-NATIVE family — ``ltx``, ``h3``, ``qwen_edit`` — REFUSES the key at config load, and
this module is the symbol that refusal names.

WHY THE REFUSAL, in the terms the feature itself sets out. ``config/sources.py`` describes a source
as a ``(corpus, geometry, sampling policy, repeat weight, cache identity)`` tuple, and notes that
keeping "sampling policy" separate from its video specialization is what would let an IMAGE family
use the list for corpus balancing — 500 studio stills at ``num_repeats: 1`` beside 40 hero shots at
``num_repeats: 3``. That reasoning is sound and it is still the right first native application. It
is also not the blocker. The blocker is one layer down:

  * **There is no native consumer of a source LIST.** ``data/precomputed.py`` reads ONE
    ``preprocessed_data_root`` and enumerates the precomputed latent/condition directories beneath
    it. Nothing in ``train/loop.py`` or the strategies takes a list of corpora, and nothing carries
    a per-sample provenance tag. A ``sources:`` key accepted today would be read by no one.
  * **``num_repeats`` does not mean on this loop what it means on musubi**, which ``SourceSpec``
    records on the field itself: musubi is EPOCH-driven, so ``num_repeats: 3`` lengthens the run
    threefold; signet's native loop is STEP-driven, so the same 3 must change MIXTURE PROPORTIONS
    at a fixed step budget. That is a real sampler, not a field read — a weighted draw across
    per-source index ranges, with the weights reported in the census so a silently-uniform draw is
    visible.
  * **The video extraction vocabulary is meaningless on two of the three.** ``qwen_edit`` pins F to
    exactly 1 (``validators.validate_qwen_edit_frames``), so "the first N frames" names nothing
    there; only ``extraction: image`` could ever be legal. ``ltx`` and ``h3`` consume PRECOMPUTED
    latents, so clip windowing already happened in the encode pass — an ``extraction`` on a native
    source would describe work the preprocess stage did, not work the loader will do.

Accepting the key under a native family would therefore be "a config nothing honours": it would
load, print, diff cleanly against the previous round, and change nothing about the run. That is the
defect class this schema spends most of its guards on, so the key is refused instead.

WHAT LANDS IT — the three items above, in that order, plus one decision that is not ours: whether
native multi-source belongs in the DATALOADER (a weighted ``ConcatDataset``) or in the PREPROCESS
pass (one precomputed root per source, mixed at encode time). The second is closer to how
``preprocessed_data_root`` already works and costs no loop change; the first is what makes
``num_repeats`` re-weightable without a re-encode. Neither is implied by slice A.

PURE: stdlib only. No ``torch``, no ``modal`` — this module is imported by nothing yet and must
stay free to be imported by the config layer's refusal message without dragging a backend in.
"""

from __future__ import annotations

from typing import Sequence

#: The families whose training path is signet's own loop rather than an external runner. Kept here
#: rather than re-derived from ``ModelConfig.family`` so the refusal in ``config/schema.py`` and the
#: gap named by this module cannot drift apart: a fifth family is native unless it says otherwise.
NATIVE_FAMILIES: tuple[str, ...] = ("ltx", "h3", "qwen_edit")


def build_native_source_datasets(sources: Sequence[object], *, family: str) -> object:
    """Build a weighted multi-corpus dataset for signet's native loop. NOT LANDED.

    Raises ``NotImplementedError`` naming what would land it. This is the symbol
    ``config/schema.py``'s native-family refusal points at, so the message an operator reads at
    config load and the work that closes the gap have one name between them.
    """
    raise NotImplementedError(
        f"build_native_source_datasets: data.sources has no consumer on family {family!r}. "
        f"signet's native loop reads ONE data.preprocessed_data_root through "
        f"signet_trainer.data.precomputed and takes no source list, and num_repeats means "
        f"something different on a STEP-driven loop than it does on musubi's EPOCH-driven one "
        f"(SourceSpec.num_repeats). Landing native support needs, in order: a multi-corpus "
        f"dataset with per-source provenance, a weighted sampler that reads num_repeats as a "
        f"MIXTURE PROPORTION, and a per-source census line so a silently-uniform draw is visible. "
        f"Until then data.sources is legal only on model.family 'wan' (the musubi-tuner runner), "
        f"which consumes it via signet_trainer.runners.musubi_toml.render_from_config; "
        f"{len(sources)} source(s) were offered."
    )
