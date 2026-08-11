"""PEFT/LoRA mechanics — inject, the locked target list, and the PEFT-aware weight enumerators.

Ported from enochiatron ``scripts/train/train.py`` (PEFT inject 1814-1847, ``P1_*_LORA_TARGETS``
137-162) and flimmer-trainer-dev ``flimmer/training/wan/block_swap.py::_get_linear_weights``
(50-76). The PEFT-aware weight enumerators (single source of truth for the offloader, D-9-DEDUP)
and the save/load roundtrip (INFR-03) live alongside.

Import-confinement (Anti-Pattern 6): this module imports ``torch`` + ``peft`` + ``safetensors``
+ stdlib ONLY — NO ``modal`` / ``ltx_core`` / ``ltx_trainer`` — so the skip-filter and config
builders stay CPU-unit-testable on Windows/CI. ``peft`` is the boundary (a declared
modal-runtime dep).

Family surfaces: the LTX contract is a SUFFIX list (``P1_*_LORA_TARGETS`` +
:func:`check_lora_targets`); the MiniMax-H3 contract is a PATH REGEX
(:data:`H3_LORA_TARGET_REGEX` + :func:`check_lora_targets_regex`). The two live side by side —
the regex probe is an additive sibling, never a replacement.

Key recipe facts encoded here:
  - the target set is ``attn1+attn2+ff.net`` (the ff.net FIX — attn-only silently under-fits
    identity, HANDOFF.md:19);
  - ``r == lora_alpha == 64`` so the PEFT scale ``alpha/rank == 1.0`` (clean convert, no rescale);
  - gradient checkpointing is enabled BEFORE ``get_peft_model`` (order matters, TRAIN-06).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file

# --------------------------------------------------------------------------------------------------
# The locked LoRA target list (the ff.net FIX)
# --------------------------------------------------------------------------------------------------

#: The 8 attention projections (both self- and cross-attention). The enochiatron default
#: video-only set was attn-only — which silently under-fit identity capacity.
P1_LORA_TARGETS: list[str] = [
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
]

#: The locked video-only set: attn + the two feed-forward projections. ``ff.net`` is ~2/3 of
#: each block's params and is where likeness/identity is encoded (HANDOFF.md:19) — adding it
#: is the single highest-value correctness fix in the port.
P1_FF_LORA_TARGETS: list[str] = P1_LORA_TARGETS + ["ff.net.0.proj", "ff.net.2"]

#: Audio-to-video (a2v) cross-modal attention targets (GATE-SPEC rev 2 item 3). The audio→video
#: cross-attention block is where the driving audio steers the generated video; a LoRA that does
#: NOT target it is structurally blind to the audio ("the #1 silent a2v failure"). Kept byte-for-byte
#: in sync with ``config/validators.py::A2V_CROSS_MODAL_ATTN_TARGETS`` (the fail-fast guard's
#: canonical copy) — the two live in different modules to keep the config layer free of the heavy
#: ``peft`` import; ``tests/test_a2v_lora_targets.py`` asserts they are equal (single source of truth
#: enforced by test, not a cross-import).
A2V_CROSS_MODAL_LORA_TARGETS: list[str] = [
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_k",
    "audio_to_video_attn.to_v",
    "audio_to_video_attn.to_out.0",
]

#: The a2v LoRA target set: the locked video set PLUS the cross-modal audio→video attention targets.
#: An a2v run injects over this so the adapter can BOTH carry likeness (video ff/attn) AND learn the
#: audio→video steering (cross-modal attn).
P1_A2V_LORA_TARGETS: list[str] = P1_FF_LORA_TARGETS + A2V_CROSS_MODAL_LORA_TARGETS


# --------------------------------------------------------------------------------------------------
# The MiniMax-H3 family target contract (H3-02) — a PATH REGEX, not a suffix list
# --------------------------------------------------------------------------------------------------
# MEASURED on live ``MiniMaxAI/MiniMax-H3`` diffusers weights, single A100-80GB
# (``P10-1-MEASURED.md`` §2, driver ``scripts/_h3_probe_modal.py``). The house leaf names carry over
# from LTX with ZERO renaming — only the module-path PREFIX differs (H3 is SINGLE-stream
# ``transformer_blocks.N.attn.*``; LTX is dual ``attn1.*``/``attn2.*``). The fused-QKV trap that the
# original-format ``FL2VA/``/``Ref2VA/`` folders carry does NOT apply on the diffusers path.

#: The four H3 attention projections. MEASURED: each matches EXACTLY 50 modules in the main
#: ``transformer_blocks.*`` stack (one per layer; ``num_layers 50``).
H3_ATTN_LORA_LEAVES: list[str] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
]

#: The H3 leaf set: attn + the two feed-forward projections. DERIVED by concatenation exactly the way
#: ``P1_FF_LORA_TARGETS`` is, so the house ff-inclusion invariant (``ff.net`` is ~2/3 of a block's
#: params and is where likeness/identity is encoded) cannot be dropped by editing the attn list alone.
#: MEASURED: 50 matches per leaf, **300 total** (50 x 6); at rank 64 that is 332.6 M trainable params
#: across 600 tensors (300 modules x ``lora_A``/``lora_B``).
H3_LORA_LEAVES: list[str] = H3_ATTN_LORA_LEAVES + ["ff.net.0.proj", "ff.net.2"]

#: Substrings marking a module as living OFF the generative stack for the H3 family. H3's collateral
#: bucket is the TEXT-STREAM token refiner (``token_refiner.refiner_blocks.{0,1}``, 12 modules) — it
#: is NOT an audio branch: H3 is single-stream and has no audio transformer stack at all. Deliberately
#: a separate constant from ``_AUDIO_BRANCH_MARKERS`` (LTX dual-stream) rather than an overload of it;
#: :func:`check_lora_targets_regex` takes the marker tuple as an argument so each family brings its own.
H3_COLLATERAL_MARKERS: tuple[str, ...] = ("token_refiner",)


def _leaves_to_path_regex(leaves: list[str], stack_prefix: str) -> str:
    """Compose ``<stack_prefix>(<leaf>|<leaf>|…)`` with every leaf regex-escaped.

    Deriving the pattern from the leaf list (rather than pinning a second, hand-written literal)
    makes drift between the two structurally impossible: adding a leaf updates the regex for free.
    """
    return stack_prefix + "(" + "|".join(re.escape(leaf) for leaf in leaves) + ")"


#: The main block stack only — ``token_refiner.refiner_blocks.*`` is deliberately unreachable.
_H3_MAIN_STACK_PREFIX: str = r"transformer_blocks\.\d+\."

#: The H3 target form PEFT actually receives: a path REGEX over module names, NOT a suffix list.
#:
#: PEFT accepts either a ``list[str]`` (dot-bounded SUFFIX matching) or a bare ``str`` (a regex
#: matched with ``re.fullmatch``). H3 requires the latter: a plain suffix list over ``H3_LORA_LEAVES``
#: ALSO hits ``token_refiner.refiner_blocks.{0,1}`` — 12 text-stream refiner modules, i.e. **4% of the
#: adapter training the wrong thing** (``P10-1-MEASURED.md`` §2). This constant is semantically
#: identical to the measured probe literal at ``scripts/_h3_probe_modal.py:260``
#: (``transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))``); the flat
#: alternation is the derived form, proven equivalent in ``tests/test_h3_lora_targets.py``.
#:
#: Deliberately NOT targeted (54 modules): ``adaln_proj.linear`` + the patch/head projections. At the
#: measured ``[96768, 2688]`` shape a rank-64 delta on ``adaln_proj`` would be a genuine low-rank
#: update, so excluding it is a recipe CHOICE, not a correctness fix (P10-1 §2).
H3_LORA_TARGET_REGEX: str = _leaves_to_path_regex(H3_LORA_LEAVES, _H3_MAIN_STACK_PREFIX)


def _assert_h3_regex_tracks_its_leaves() -> None:
    """Import-time drift guard: the regex MUST admit every leaf and refuse the refiner stack.

    An explicit ``raise`` (not ``assert``) so ``python -O`` cannot strip the one check standing
    between a silent target-set regression and a metered A100 training the token refiner.
    """
    compiled = re.compile(H3_LORA_TARGET_REGEX)
    for leaf in H3_LORA_LEAVES:
        if not compiled.fullmatch(f"transformer_blocks.0.{leaf}"):
            raise RuntimeError(
                f"H3_LORA_TARGET_REGEX no longer matches leaf {leaf!r} — the target contract drifted"
            )
        if compiled.fullmatch(f"token_refiner.refiner_blocks.0.{leaf}"):
            raise RuntimeError(
                f"H3_LORA_TARGET_REGEX now over-matches the token refiner at {leaf!r} "
                "(P10-1-MEASURED §2: 12 modules of text-stream refiner)"
            )
    if compiled.fullmatch("transformer_blocks.0.adaln_proj.linear"):
        raise RuntimeError("H3_LORA_TARGET_REGEX now matches adaln_proj.linear (excluded by recipe)")


_assert_h3_regex_tracks_its_leaves()


# --------------------------------------------------------------------------------------------------
# Config builder
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# LoRA-target collateral-matching probe (GATE-SPEC rev 2 item 3)
# --------------------------------------------------------------------------------------------------

#: Substrings that mark a module as living on the AUDIO dual-stream branch of the LTX-2.3 transformer
#: (loader.py:62 confirmed ``audio_transformer_blocks…attn1.to_q`` = 2048 vs the video 4096). PEFT's
#: SHORT-suffix target matching (e.g. ``attn1.to_q``) matches these audio modules COLLATERALLY —
#: which affects VIDEO-ONLY runs too, so the probe makes it VISIBLE rather than silent.
_AUDIO_BRANCH_MARKERS: tuple[str, ...] = ("audio_transformer_blocks", "audio_to_video_attn")


def _peft_suffix_match(module_name: str, target: str) -> bool:
    """True iff PEFT would inject ``target`` into ``module_name`` (exact or dot-bounded suffix).

    Mirrors PEFT's ``target_modules`` string matching: a target matches a module whose name equals
    it OR ends with ``"." + target`` (so ``attn1.to_q`` matches ``…blocks.0.attn1.to_q`` but NOT a
    coincidental ``…xattn1.to_q``). Pure string logic — no torch, no peft internals.
    """
    return module_name == target or module_name.endswith("." + target)


def check_lora_targets(
    model_or_names: Any, targets: list[str]
) -> dict[str, dict[str, Any]]:
    """Per-target instance-count probe: how many modules EACH LoRA target would match, video vs audio.

    GATE-SPEC rev 2 item 3 verification probe. PEFT injects a LoRA into every module whose name
    suffix-matches an entry of ``target_modules``. A SHORT suffix like ``attn1.to_q`` therefore
    collaterally matches BOTH the video ``transformer_blocks.*.attn1.to_q`` AND the audio
    ``audio_transformer_blocks.*.attn1.to_q`` — this affects VIDEO-ONLY runs too (they inject into
    the audio branch whether they mean to or not), so the probe surfaces it instead of leaving it
    silent. It ALSO confirms the a2v cross-modal targets (``audio_to_video_attn.*``) actually hit
    modules on a real model (a zero count => "the #1 silent a2v failure": the adapter is blind to
    audio).

    Args:
        model_or_names: either an ``nn.Module`` (its ``named_modules()`` names are used) OR a plain
            iterable of module-name strings (so the CPU test can drive it without a real transformer).
        targets: the LoRA ``target_modules`` list to probe.

    Returns:
        ``{target: {"total": n, "video": nv, "audio": na, "audio_names": [...up to 3...]}}`` — per
        target: total matched modules, the video/audio split (audio = a name carrying an
        ``_AUDIO_BRANCH_MARKERS`` substring), and a few example audio matches for the log.
    """
    if hasattr(model_or_names, "named_modules"):
        names = [name for name, _ in model_or_names.named_modules()]
    else:
        names = list(model_or_names)

    report: dict[str, dict[str, Any]] = {}
    for target in targets:
        matched = [n for n in names if _peft_suffix_match(n, target)]
        audio_names = [n for n in matched if any(m in n for m in _AUDIO_BRANCH_MARKERS)]
        report[target] = {
            "total": len(matched),
            "video": len(matched) - len(audio_names),
            "audio": len(audio_names),
            "audio_names": audio_names[:3],
        }
    return report


def _peft_regex_match(module_name: str, pattern: str) -> bool:
    """True iff PEFT would inject a bare-``str`` ``target_modules`` regex into ``module_name``.

    Mirrors PEFT EXACTLY: ``peft.tuners.tuners_utils.match_target_against_key`` is
    ``re.fullmatch(target_pattern, key)``. NOT ``re.search`` — a search-based probe would
    over-report (e.g. it would "match" ``transformer_blocks.0.attn.to_q.base_layer``, which PEFT
    does not inject), and the probe's entire job is to predict PEFT's injection count exactly.
    """
    return re.fullmatch(pattern, module_name) is not None


def check_lora_targets_regex(
    model_or_names: Any,
    pattern: str,
    collateral_markers: tuple[str, ...] = H3_COLLATERAL_MARKERS,
    leaves: list[str] | None = None,
) -> dict[str, Any]:
    """Regex-aware SIBLING of :func:`check_lora_targets` — the H3 arch-gate counting probe.

    :func:`check_lora_targets` matches by dot-bounded SUFFIX, which is what PEFT does for a
    ``list[str]`` ``target_modules``. It structurally cannot express H3's target form, which is a
    bare ``str`` regex (see :data:`H3_LORA_TARGET_REGEX`). This is an ADDITIVE sibling rather than a
    change to ``_peft_suffix_match`` — ``tests/test_a2v_lora_targets.py`` pins the suffix semantics.

    The gate this feeds (``P10-1-MEASURED.md`` §8.2) must print ``main=50`` PER LEAF before any
    metered dispatch: a grand total alone would hide a per-leaf zero, and a per-leaf zero is exactly
    the "silently trains the wrong thing" failure this whole contract exists to prevent.

    Args:
        model_or_names: an ``nn.Module`` (its ``named_modules()`` names are used) OR a plain iterable
            of module-name strings — same duck-typing as :func:`check_lora_targets`, so the CPU test
            drives it without a real transformer.
        pattern: the regex handed to ``LoraConfig(target_modules=...)`` as a bare ``str``.
        collateral_markers: substrings marking a matched module as living OFF the generative stack.
            Defaults to the H3 text-refiner marker; pass another family's tuple to reuse the probe.
        leaves: the leaf names to break ``per_leaf`` down by (dot-bounded suffix among the MATCHED
            names). Defaults to :data:`H3_LORA_LEAVES`.

    Returns:
        ``{"pattern": …, "total": n, "main": n_main, "collateral": n_collateral,
        "collateral_names": [...up to 3...], "per_leaf": {leaf: count}}``. ``main`` is
        ``total - collateral``; a non-zero ``collateral`` means the pattern is too loose.
    """
    if hasattr(model_or_names, "named_modules"):
        names = [name for name, _ in model_or_names.named_modules()]
    else:
        names = list(model_or_names)
    if leaves is None:
        leaves = H3_LORA_LEAVES

    matched = [n for n in names if _peft_regex_match(n, pattern)]
    collateral = [n for n in matched if any(m in n for m in collateral_markers)]
    return {
        "pattern": pattern,
        "total": len(matched),
        "main": len(matched) - len(collateral),
        "collateral": len(collateral),
        "collateral_names": collateral[:3],
        "per_leaf": {
            leaf: sum(1 for n in matched if _peft_suffix_match(n, leaf)) for leaf in leaves
        },
    }


def build_lora_config(
    rank: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    targets: list[str] | str | None = None,
) -> LoraConfig:
    """Build the locked LoRA config.

    Defaults: ``r == alpha == 64`` (PEFT scale ``alpha/rank == 1.0`` → clean convert with no
    rescale), ``lora_dropout == 0.0``, ``bias == "none"``, ``target_modules == P1_FF_LORA_TARGETS``.

    ``targets`` accepts BOTH PEFT target forms:

    * ``list[str]`` — dot-bounded suffix matching (the LTX family; PEFT itself normalises the list
      to a ``set`` in ``LoraConfig.__post_init__``, so the stored value is a set either way);
    * ``str`` — a path REGEX matched with ``re.fullmatch`` (the H3 family,
      :data:`H3_LORA_TARGET_REGEX`).

    A ``str`` is passed through UNCHANGED. This is a live-bug fix: the previous unconditional
    ``list(targets)`` char-exploded a regex (``list("abc") == ["a", "b", "c"]``), which PEFT would
    then treat as a suffix list of single characters, match NOTHING, and only surface at
    ``train/loop.py``'s ``"No trainable parameters found. Is LoRA applied?"`` — i.e. after the
    metered A100 is already billing.
    """
    if targets is None:
        targets = P1_FF_LORA_TARGETS
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets if isinstance(targets, str) else list(targets),
        lora_dropout=dropout,
        bias="none",
    )


# --------------------------------------------------------------------------------------------------
# Inject — GC BEFORE LoRA (TRAIN-06, order matters)
# --------------------------------------------------------------------------------------------------


def inject_lora(model: nn.Module, lora_config: LoraConfig) -> nn.Module:
    """Enable gradient checkpointing, THEN wrap the model with ``get_peft_model``.

    Order is load-bearing (train.py:1814-1847): grad-checkpointing must be enabled on the BASE
    transformer before PEFT wraps it, otherwise the enable can land on the PEFT wrapper and
    silently no-op. Returns the PEFT-wrapped model in train mode.

    Three method names, tried in order. The first two are the pre-existing LTX branches and stay
    FIRST so LTX behaviour is byte-identical; ``enable_gradient_checkpointing`` is APPENDED for the
    MiniMax-H3 diffusers module, which exposes neither of the other two (confirmed by the P10-1
    probe, ``scripts/_h3_probe_modal.py:360``). A silent no-op here is not cosmetic: without GC the
    H3 activation budget collapses and the run OOMs on the A100 (P10-1-MEASURED §4).
    """
    # 1. Gradient checkpointing FIRST (prefer the ltx-core method, fall back to the HF method,
    #    then the diffusers method used by H3).
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    # 2. THEN inject LoRA and switch to train mode.
    model = get_peft_model(model, lora_config)
    model.train()
    return model


# --------------------------------------------------------------------------------------------------
# PEFT-aware weight enumerators (promoted from offload, OFFL-03 BSWP-TRAINING-01 fix)
# --------------------------------------------------------------------------------------------------
# Promoted VERBATIM from ``offload/block_swap.py`` (07-13 BSWP-TRAINING-01 fix) so there is a SINGLE
# source of truth for "which leaf Linears own real weights under a PEFT-wrapped block". The offloader
# imports these; do NOT re-derive a second copy (D-9-DEDUP). Any semantic change here re-opens the
# ``mat2 is on cpu`` device mismatch — these are a MOVE, not a rewrite. Pure duck-typing on
# ``base_layer`` + class-name (no new imports; ``torch.nn as nn`` already present above).


def _is_lora_weight_name(name: str) -> bool:
    """True for a PEFT adapter sub-linear (lora_A/lora_B/lora_embedding/…).

    A PEFT ``lora.Linear`` wrapper attaches its adapter sub-Linears under names like
    ``…lora_A.default`` / ``…lora_B.frozen``; the frozen base sits at ``…base_layer``.
    Only the ``lora_*`` leaves are adapter weights.
    """
    return "lora_" in name


def get_linear_weights(block: nn.Module) -> list[tuple[nn.Module, str]]:
    """Walk a block and yield (module, 'weight') for every REAL weight-owning leaf Linear.

    PEFT-LoRA-aware (OFFL-03 / BSWP-TRAINING-01 fix): a targeted Linear becomes a PEFT
    ``lora.Linear`` wrapper whose class name still ends in "Linear" and whose ``.weight``
    PROXIES ``base_layer.weight``. The harvested Wan code matched that wrapper and then
    SKIPPED all of its descendants — so ``base_layer`` and every adapter's ``lora_A``/
    ``lora_B`` sub-Linear were never tracked. With TWO active adapters (frozen + default)
    a freshly ``load_adapter``-ed frozen LoRA sub-linear then executed with its weight still
    on CPU while its block ran on GPU → the ``mat2 is on cpu`` crash (07-12 run 4).

    The fix: detect a PEFT wrapper by its ``base_layer`` attribute; do NOT yield the wrapper
    (its ``.weight`` is a proxy — yielding it would double-count the base), but DESCEND into
    it and yield the real leaves (``base_layer`` + each active adapter's ``lora_A``/``lora_B``).
    A plain nn.Linear (no ``base_layer``) still yields exactly once. Order is deterministic
    (``named_modules`` pre-order), so callers that ``zip`` against this enumerator stay aligned.

    This is the "ALL tracked weights" set — used by the forward-pre-hook device guard so it
    notices an off-device lora sub-linear (not just ``weights[0]``). The narrower "swappable"
    subset (what actually moves to CPU) is :func:`get_swappable_weights`.

    Coverage (P7-IN-02): the ``endswith("Linear")`` class-name predicate matches ``nn.Linear``
    and the PEFT ``lora.Linear`` wrapper (which is then skipped via ``base_layer``). It does NOT
    match bitsandbytes ``Linear8bitLt`` / ``Linear4bit`` (their class names end in ``Lt`` / ``4bit``,
    not ``Linear``) — the LTX-2.3 base transformer is plain bf16 ``nn.Linear``, so this is
    plain-Linear-only by design. Class-name matching (rather than ``isinstance``) keeps the
    enumerator import-light: it needs no optional-dep imports to duck-type a Linear leaf.
    """
    results = []
    for name, module in block.named_modules():
        # PEFT wrapper (lora.Linear): its .weight proxies base_layer.weight. Do NOT yield the
        # proxy — descend into it so we reach base_layer + the adapter sub-Linears as real leaves.
        if hasattr(module, "base_layer"):
            continue
        if module.__class__.__name__.endswith("Linear") and getattr(module, "weight", None) is not None:
            results.append((module, "weight"))
    return results


def get_swappable_weights(block: nn.Module) -> list[tuple[nn.Module, str]]:
    """The subset of leaf Linears that actually get parked to CPU: base_layer + plain Linears.

    LoRA sub-Linears (``lora_A``/``lora_B``, both frozen and trainable) are EXCLUDED — they are
    kept GPU-RESIDENT. Rationale (kohya-proven, and the second half of the BSWP-TRAINING-01 fix):
      * the trainable ("default") adapter's params MUST be on the GPU at ``optimizer.step()`` —
        adamw8bit/bitsandbytes require CUDA params + GPU-resident optimizer state; parking them
        to CPU after the backward hook would break the step;
      * lora sub-Linears are tiny (rank 64) so keeping them resident costs almost no VRAM;
      * the frozen adapter is ``load_adapter``-ed AFTER the offloader is constructed — excluding
        lora from the swappable/pinned set keeps the pinned-buffer count STABLE across that late
        load (only base_layer weights are pinned, and their count/order never changes), so the
        positional ``zip`` in the movers stays aligned. The device guard re-homes any late-loaded
        lora that landed off-device.

    Only large frozen base weights (``base_layer`` + non-targeted plain Linears) are swapped.
    Deterministic ``named_modules`` order → positional pinned buffers stay aligned.
    """
    results = []
    for name, module in block.named_modules():
        if hasattr(module, "base_layer"):
            continue  # PEFT wrapper proxy — descend
        if module.__class__.__name__.endswith("Linear") and getattr(module, "weight", None) is not None:
            if _is_lora_weight_name(name):
                continue  # resident — never parked to CPU
            results.append((module, "weight"))
    return results


# --------------------------------------------------------------------------------------------------
# Save / load roundtrip (INFR-03)
# --------------------------------------------------------------------------------------------------

#: The PEFT-native module prefix on every saved adapter key.
_PEFT_PREFIX = "base_model.model."

#: The adapter weights filename PEFT ``save_pretrained`` writes.
ADAPTER_FILENAME = "adapter_model.safetensors"


def save_adapter(
    model: nn.Module, out_dir: str | Path, adapter_name: str = "default"
) -> Path:
    """Persist ONE PEFT adapter (writes ``adapter_model.safetensors`` + ``adapter_config.json``).

    ``adapter_name`` selects which adapter to serialize. In the single-adapter path this is the
    lone ``"default"`` adapter (unchanged behaviour). Under frozen-LoRA stacking (D-7-FREEZE) the
    model carries a second ``"frozen"`` adapter; restricting the save to ``adapter_name="default"``
    routes through ``get_peft_model_state_dict(model, adapter_name="default")`` internally, so the
    frozen adapter's weights are EXCLUDED — only our trainable adapter is written. ``save_pretrained``
    (not a raw ``safetensors`` dump) is used deliberately: it also writes ``adapter_config.json``,
    which ``load_adapter`` / ``stack_frozen_adapter`` require to re-hydrate the adapter later.

    LAYOUT (P7-IN-04): ``save_pretrained`` writes the ``"default"`` adapter to the root of ``out_dir``
    (``out/adapter_model.safetensors``) but every NON-default adapter into a ``<name>/`` subdirectory
    (``out/<name>/adapter_model.safetensors``). :func:`load_adapter_into` mirrors this resolution, so
    save→load roundtrips correctly for any ``adapter_name``. Returns the ``out_dir`` root either way.
    """
    out = Path(out_dir)
    model.save_pretrained(str(out), selected_adapters=[adapter_name])
    return out


def load_adapter_into(
    model: nn.Module, ckpt_dir: str | Path, adapter_name: str = "default"
) -> Any:
    """Load ``adapter_model.safetensors`` from ``ckpt_dir`` into ``model`` via PEFT.

    Keeps PEFT-native keys (the saved file already carries the ``base_model.model.`` prefix that
    ``set_peft_model_state_dict`` expects). An adapter-only file legitimately leaves the frozen
    ``base_layer`` weights as PEFT "missing keys" — that is expected, not an error; what matters
    is that no UNEXPECTED keys appear. This is also the resume adapter-reload seam (landmine #1,
    03-04): ``set_peft_model_state_dict`` is what re-hydrates the LoRA weights that
    ``CheckpointManager.resume()`` does NOT restore.

    P7-IN-04: :func:`save_adapter` routes through ``save_pretrained(selected_adapters=[name])``, which
    writes the ``"default"`` adapter to ``ckpt_dir`` root but every NON-default adapter into a
    ``<name>/`` subdirectory. Resolve the matching path here so a non-default ``adapter_name`` reads
    ``ckpt_dir/<name>/adapter_model.safetensors`` (not the flat root, which would raise FileNotFound).
    """
    adapter_dir = Path(ckpt_dir)
    if adapter_name != "default":
        adapter_dir = adapter_dir / adapter_name
    weights = load_file(str(adapter_dir / ADAPTER_FILENAME))
    return load_adapter_state_into(model, weights, adapter_name=adapter_name, what=str(adapter_dir))


def is_quanto_quantized(model: nn.Module) -> bool:
    """Does this model carry quanto ``QModuleMixin`` layers? Decides which load path is usable."""
    return any(
        type(module).__name__.startswith("Q")
        and hasattr(module, "weight")
        and hasattr(type(module), "_load_from_state_dict")
        and "quanto" in type(module).__module__
        for module in model.modules()
    )


def load_adapter_state_into(
    model: nn.Module,
    adapter_weights: dict[str, Any],
    *,
    adapter_name: str = "default",
    what: str = "adapter",
) -> Any:
    """Assign a PEFT adapter state dict into ``model``, by whichever path the model can survive.

    ⛔ ``set_peft_model_state_dict`` IS UNUSABLE ON A QUANTIZED MODEL, and this is the fourth
    instance of the house pattern *the flag is not the behaviour*. It ends in
    ``model.load_state_dict(peft_state_dict, strict=False)``, and ``load_state_dict`` walks EVERY
    submodule — not only the ones the incoming dict names. On a quanto-quantized transformer the
    base Linears are ``QModuleMixin``, whose ``_load_from_state_dict`` does an UNCONDITIONAL
    ``state_dict.pop(prefix + name)`` for its own ``_data`` internals. A LoRA adapter dict carries no
    base weights, so the pop raises — and ``strict=False`` does not save you, because quanto's
    override never consults it::

        KeyError: 'base_model.model.time_text_embed.timestep_embedder.linear_1.weight._data'

    That exact line has now killed a run TWICE: once as the crash-resume that cost 4500 steps of
    phase 1 (fixed in ``train/checkpoint.py`` at f8f2815), and once as the §8 band render, which got
    its base column out and died on the first band member. The second time is why this lives here
    rather than inline in ``resume()``: it is a property of loading an adapter onto a quantized
    model, not a property of resuming, and every caller of :func:`load_adapter_into` needs it.

    ``CheckpointManager.resume`` now DELEGATES here rather than carrying its own copy. It briefly
    did carry one — the resume fix landed first and the render walked into the identical failure
    through :func:`load_adapter_into` — which is the whole argument for one implementation: a
    duplicated money-critical routine is fixed in one place and stays broken in the other.

    ⚠ THIS FUNCTION IS NOW SOURCE-TEXT ASSERTED. ``tests/test_checkpoint_resume.py`` greps THIS FILE
    for "HARD LOCK across a chain" and "PARTIAL adapter is worse than refusing", because a refusal
    that stops explaining WHY has lost the part that makes it actionable. Reword them and that test
    fails on purpose; it is not being brittle about formatting, it is holding the reasoning in place.

    THE SAFE PATH. The adapter parameters are NOT quantized — quantization runs on the un-wrapped
    transformer BEFORE ``inject_lora``, so quanto converts the base Linears and the LoRA tensors are
    added afterwards as ordinary parameters. They can therefore be assigned DIRECTLY, which never
    visits a quantized module. The only transform PEFT applies to a saved key is inserting the
    adapter name before the final component; verified against the real 1680-tensor checkpoint, all
    1680 keys map. Non-quantized models keep the original ``set_peft_model_state_dict`` path
    untouched.

    Two refusals rather than a best-effort load:

      * a SHAPE MISMATCH names the rank lock — ``rank == alpha`` is a HARD LOCK across a chain
        precisely because changing either re-shapes ``lora_A`` / ``lora_B``;
      * a PARTIAL match refuses outright. Continuing from a half-restored adapter at a plausible
        loss, or rendering a band column from one, is the worst available outcome: nothing looks
        wrong.
    """
    if not is_quanto_quantized(model):
        return set_peft_model_state_dict(model, adapter_weights, adapter_name=adapter_name)

    live = dict(model.named_parameters())
    missing: list[str] = []
    loaded = 0
    with torch.no_grad():
        for key, tensor in adapter_weights.items():
            target = live.get(key)
            if target is None:
                head, sep, tail = key.rpartition(".")
                target = live.get(f"{head}.{adapter_name}{sep}{tail}") if sep else None
            if target is None:
                missing.append(key)
                continue
            if tuple(target.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"Adapter load aborted ({what}): {key} is {tuple(tensor.shape)} on disk but "
                    f"{tuple(target.shape)} in the live model. A rank/alpha change between rounds "
                    f"re-shapes lora_A/lora_B — that is why rank == alpha is a HARD LOCK across a "
                    f"chain."
                )
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded += 1
    if missing or loaded != len(adapter_weights):
        raise RuntimeError(
            f"Adapter load aborted ({what}): {loaded}/{len(adapter_weights)} adapter tensor(s) "
            f"matched a live parameter; {len(missing)} unmatched, first few: {missing[:4]}. Loading "
            f"a PARTIAL adapter is worse than refusing — the caller would continue from a "
            f"half-restored state at a plausible loss."
        )
    banner = (
        f"[lora] quantized adapter load ({what}): {loaded}/{len(adapter_weights)} tensors assigned "
        f"directly — set_peft_model_state_dict is unusable on a quanto model (LANDMINE #1b)."
    )
    print(banner)
    return banner


# --------------------------------------------------------------------------------------------------
# Frozen-LoRA stacking (D-7-FREEZE) — load a second adapter, activate both, re-freeze the loaded one
# --------------------------------------------------------------------------------------------------


def stack_frozen_adapter(
    model: nn.Module, frozen_path: str | Path, adapter_name: str = "frozen"
) -> nn.Module:
    """Load an existing PEFT adapter alongside the trainable ``"default"`` and freeze it out.

    Standard PEFT multi-adapter (RESEARCH §Pattern 4, ``[VERIFIED: Context7 /huggingface/peft]``):

    1. ``model.load_adapter(frozen_path, adapter_name="frozen")`` — load the second adapter.
    2. ``model.base_model.set_adapter(["frozen", "default"])`` — activate BOTH so their LoRA deltas
       SUM in the forward. (PEFT requires an adapter be active *before* optimizer init or its params
       never update — hence we activate first, then re-freeze.)
    3. ``set_adapter`` sets ``requires_grad=True`` on BOTH adapters, so we immediately RE-FREEZE the
       frozen one: every param whose key carries the adapter as a dot-bounded name COMPONENT
       (``f".{adapter_name}."``, e.g. ``…lora_A.frozen.weight``) gets ``requires_grad_(False)``. The
       optimizer (built over ``[p for p in model.parameters() if p.requires_grad]``) then trains ONLY
       our new ``"default"`` adapter. Save with ``save_adapter(model, out, adapter_name="default")``
       to exclude the frozen weights.

    P7-IN-01: the freeze match is a dot-bounded component test, NOT a bare ``adapter_name in name``
    substring. A substring match would over-freeze any param whose name merely CONTAINS the adapter
    string (e.g. a module coincidentally named ``…frozen_norm…``), silently stranding weights the
    optimizer expected to train. Matching ``f".{adapter_name}."`` requires the adapter to be a full
    PEFT name segment.

    CONSTRAINT (RESEARCH A2 / Anti-Pattern line 219): the ``frozen_path`` MUST be a signet-produced
    **PEFT-format** adapter (``base_model.model.`` keys + ``adapter_config.json``). The official
    2.3-22b IC-LoRAs are single-file *comfy*-key safetensors (``diffusion_model.`` prefix) — they do
    NOT load through this PEFT loader; use the ``ltx_core`` ``loras=`` path (ICLoraPipeline) to feed
    them. Do NOT cross the streams.

    Runtime form (the default ``adapter_name="frozen"``) is exactly::

        model.base_model.set_adapter(["frozen", "default"])
        marker = ".frozen."
        for n, p in model.named_parameters():
            if marker in n:
                p.requires_grad_(False)
    """
    # 1. Load the second adapter alongside "default".
    model.load_adapter(str(frozen_path), adapter_name=adapter_name)
    # 2. Activate BOTH — their LoRA deltas sum in the forward.
    model.base_model.set_adapter([adapter_name, "default"])
    # 3. set_adapter set requires_grad=True on BOTH — re-freeze the loaded adapter so the optimizer
    #    trains ONLY "default". P7-IN-01: match the adapter as a dot-bounded PEFT name COMPONENT
    #    (f".{adapter_name}.", e.g. "…lora_A.frozen.weight"), not a bare substring — a substring
    #    match would over-freeze any param whose name merely contained the adapter string.
    marker = f".{adapter_name}."
    for name, param in model.named_parameters():
        if marker in name:
            param.requires_grad_(False)
    return model


def strip_peft_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip the ``base_model.model.`` prefix (generate.py:984) for non-PEFT consumers."""
    return {k.replace(_PEFT_PREFIX, "", 1): v for k, v in state_dict.items()}


def roundtrip_check(model: nn.Module, tmp_dir: str | Path) -> dict[str, int]:
    """The cheapest sufficient INFR-03 proof: save -> reload -> assert key-count + tensor eq.

    Captures the in-memory adapter state, saves it, reloads the serialized file back into the
    SAME model, and asserts the adapter state is byte-faithful before vs after (same keys, all
    tensors ``allclose``) with no ``.alpha`` / serialization mismatch. Returns ``{"num_keys": N}``.
    """
    before = get_peft_model_state_dict(model)

    save_adapter(model, tmp_dir)
    loaded = load_file(str(Path(tmp_dir) / ADAPTER_FILENAME))

    if set(loaded) != set(before):
        raise AssertionError(
            "INFR-03 roundtrip key mismatch: "
            f"missing={set(before) - set(loaded)} extra={set(loaded) - set(before)}"
        )
    alpha_keys = [k for k in loaded if k.endswith(".alpha")]
    if alpha_keys:
        raise AssertionError(f"unexpected serialized scale keys (rank==alpha): {alpha_keys}")

    set_peft_model_state_dict(model, loaded, adapter_name="default")
    after = get_peft_model_state_dict(model)

    if len(after) != len(before):
        raise AssertionError(
            f"INFR-03 roundtrip count mismatch: before={len(before)} after={len(after)}"
        )
    for k in before:
        if not torch.allclose(before[k], after[k]):
            raise AssertionError(f"INFR-03 roundtrip tensor mismatch at {k!r}")

    return {"num_keys": len(before)}
