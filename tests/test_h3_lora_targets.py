"""H3-02 — the MiniMax-H3 LoRA target contract is a PATH REGEX, and it pins 300 / 0.

CPU unit tests for the H3 family surface of ``lora/peft.py``. P10-1 measured on live
``MiniMaxAI/MiniMax-H3`` diffusers weights that the house leaf names carry over from LTX with
ZERO renaming (only the module-path PREFIX differs), but that a plain SUFFIX list over-matches
into ``token_refiner.refiner_blocks.{0,1}`` — 12 modules of text-stream refiner, i.e. **4% of the
adapter training the wrong thing** (``P10-1-MEASURED.md`` §2). The target form therefore has to be
a path regex, and this file is the cheapest correctness gate in the phase.

Proven here, on a synthetic H3-shaped module-name tree (50 main blocks + 2 refiner blocks):

* ``H3_LORA_TARGET_REGEX`` matches **300** main-stack modules (50 x 6) and **0** collateral;
* the bare-suffix-list form over-matches by exactly **12** — asserted explicitly so a future
  "simplify it back to a suffix list" is caught here rather than on a metered A100;
* ``adaln_proj.linear`` and the patch/head projections are never matched;
* the probe mirrors PEFT's REAL matching semantics (``re.fullmatch`` for a bare-``str``
  ``target_modules``), cross-checked against PEFT's own ``check_target_module_exists``;
* PEFT genuinely injects 300-shaped (here 2-block-scaled) — an end-to-end ``inject_lora`` proof
  that also covers H3's ``enable_gradient_checkpointing`` GC-chain entry (TRAIN-06 order);
* ``build_lora_config`` passes a regex STRING through un-exploded (the live bug: ``list("abc")``
  is ``["a","b","c"]``, which would make PEFT match nothing and raise "No trainable parameters
  found. Is LoRA applied?" only after the A100 is already billing);
* the LTX schema default's eight ``attn1``/``attn2`` suffixes match ZERO H3 modules — and the
  ff pair matches 104, which is the load-bearing correction to "an H3 config fails LOUD at
  inject": it does NOT. It fails SILENTLY (see ``test_ltx_default_on_h3_fails_silently_not_loud``).

Mirrors the shape of ``tests/test_a2v_lora_targets.py`` (synthetic module-name list + explicit
per-target count assertions). CPU-only; no weights, no Modal, no GPU.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")

from signet_trainer.lora.peft import (  # noqa: E402  — after skip-guard
    H3_ATTN_LORA_LEAVES,
    H3_COLLATERAL_MARKERS,
    H3_LORA_LEAVES,
    H3_LORA_TARGET_REGEX,
    P1_FF_LORA_TARGETS,
    build_lora_config,
    check_lora_targets,
    check_lora_targets_regex,
    inject_lora,
)

# --------------------------------------------------------------------------------------------------
# The synthetic H3 module-name tree (mirrors the live named_modules survey, P10-1-MEASURED §1-2)
# --------------------------------------------------------------------------------------------------

#: MEASURED live: ``transformer_blocks=50``, ``token_refiner.refiner_blocks=2``.
H3_MAIN_BLOCKS = 50
H3_REFINER_BLOCKS = 2

#: MEASURED: 50 x 6 = 300 main-stack targets; 2 x 6 = 12 token-refiner modules the suffix list
#: would collaterally hit.
EXPECTED_MAIN_MATCHES = H3_MAIN_BLOCKS * 6
EXPECTED_COLLATERAL_MATCHES = H3_REFINER_BLOCKS * 6

#: The patch/head projections + embedders that live outside both block stacks. Together with the
#: 50 ``adaln_proj.linear`` modules these are the "deliberately NOT targeted" set (P10-1 §2).
_H3_HEAD_AND_PATCH_NAMES = [
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "proj_out",
    "audio_proj_out",
    "time_embedder",
    "norm_out.linear",
]


def _build_h3_module_names() -> list[str]:
    """The synthetic H3 ``named_modules()`` name list — built, not transcribed."""
    names: list[str] = []
    for n in range(H3_MAIN_BLOCKS):
        for leaf in H3_LORA_LEAVES:
            names.append(f"transformer_blocks.{n}.{leaf}")
        # Deliberately excluded neighbours that live INSIDE a targeted block: a bad regex that
        # anchors too loosely would sweep these up.
        names.append(f"transformer_blocks.{n}.adaln_proj.linear")
        names.append(f"transformer_blocks.{n}.attn.norm_q")
        names.append(f"transformer_blocks.{n}.norm1")
    for n in range(H3_REFINER_BLOCKS):
        # The text-stream refiner: byte-identical leaf names, WRONG stack.
        for leaf in H3_LORA_LEAVES:
            names.append(f"token_refiner.refiner_blocks.{n}.{leaf}")
    names.extend(_H3_HEAD_AND_PATCH_NAMES)
    return names


_H3_NAMES = _build_h3_module_names()


# --------------------------------------------------------------------------------------------------
# The leaf contract (the ff-inclusion invariant must survive a one-line edit)
# --------------------------------------------------------------------------------------------------


def test_h3_leaves_are_the_measured_six() -> None:
    assert H3_LORA_LEAVES == [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ]


def test_h3_leaves_derive_the_ff_inclusion_invariant() -> None:
    # Same constant-with-invariant pattern as ``P1_FF_LORA_TARGETS = P1_LORA_TARGETS + [ff...]``:
    # the ff pair is CONCATENATED on, so it cannot be dropped by editing the attn list alone.
    assert H3_LORA_LEAVES == H3_ATTN_LORA_LEAVES + ["ff.net.0.proj", "ff.net.2"]
    assert all(leaf in H3_LORA_LEAVES for leaf in H3_ATTN_LORA_LEAVES)


def test_h3_collateral_markers_are_the_text_refiner_not_the_audio_branch() -> None:
    # H3 is SINGLE-stream: it has no audio branch, so the LTX ``_AUDIO_BRANCH_MARKERS`` name must
    # not be overloaded. H3's collateral bucket is the text-stream token refiner.
    assert H3_COLLATERAL_MARKERS == ("token_refiner",)


# --------------------------------------------------------------------------------------------------
# The 300 / 0 proof — the reason this plan exists
# --------------------------------------------------------------------------------------------------


def test_h3_regex_matches_300_main_stack_modules_and_zero_collateral() -> None:
    report = check_lora_targets_regex(_H3_NAMES, H3_LORA_TARGET_REGEX)
    assert report["total"] == EXPECTED_MAIN_MATCHES, report["total"]
    assert report["main"] == EXPECTED_MAIN_MATCHES, report["main"]
    assert report["collateral"] == 0, report["collateral_names"]
    assert report["collateral_names"] == []
    assert report["pattern"] == H3_LORA_TARGET_REGEX


def test_h3_regex_reports_fifty_per_leaf() -> None:
    # The gate prints ``main=50`` per leaf (P10-1-MEASURED §8.2) — a per-leaf zero is the silent
    # failure mode a grand total alone would hide.
    report = check_lora_targets_regex(_H3_NAMES, H3_LORA_TARGET_REGEX)
    assert set(report["per_leaf"]) == set(H3_LORA_LEAVES)
    for leaf in H3_LORA_LEAVES:
        assert report["per_leaf"][leaf] == H3_MAIN_BLOCKS, f"{leaf}: {report['per_leaf'][leaf]}"


def test_h3_regex_never_matches_adaln_or_patch_head_projections() -> None:
    # ``adaln_proj.linear`` is [96768, 2688] — deliberately NOT targeted (P10-1 §2).
    adaln = [f"transformer_blocks.{n}.adaln_proj.linear" for n in range(H3_MAIN_BLOCKS)]
    assert check_lora_targets_regex(adaln, H3_LORA_TARGET_REGEX)["total"] == 0
    assert check_lora_targets_regex(_H3_HEAD_AND_PATCH_NAMES, H3_LORA_TARGET_REGEX)["total"] == 0


def test_h3_regex_does_not_match_the_refiner_stack_alone() -> None:
    refiner = [
        f"token_refiner.refiner_blocks.{n}.{leaf}"
        for n in range(H3_REFINER_BLOCKS)
        for leaf in H3_LORA_LEAVES
    ]
    assert len(refiner) == EXPECTED_COLLATERAL_MATCHES
    assert check_lora_targets_regex(refiner, H3_LORA_TARGET_REGEX)["total"] == 0


def test_bare_suffix_list_over_matches_the_token_refiner_by_twelve() -> None:
    """THE over-match proof — asserted so 'simplify it to a suffix list' dies here, not on an A100."""
    suffix_report = check_lora_targets(_H3_NAMES, H3_LORA_LEAVES)
    suffix_total = sum(r["total"] for r in suffix_report.values())
    regex_report = check_lora_targets_regex(_H3_NAMES, H3_LORA_TARGET_REGEX)

    # Every leaf suffix-matches 52 modules: 50 main + 2 refiner.
    for leaf in H3_LORA_LEAVES:
        assert suffix_report[leaf]["total"] == H3_MAIN_BLOCKS + H3_REFINER_BLOCKS
    assert suffix_total == EXPECTED_MAIN_MATCHES + EXPECTED_COLLATERAL_MATCHES == 312
    assert regex_report["main"] == EXPECTED_MAIN_MATCHES
    assert suffix_total - regex_report["main"] == EXPECTED_COLLATERAL_MATCHES == 12


def test_regex_probe_flags_collateral_when_the_pattern_is_too_loose() -> None:
    # A deliberately loose pattern that DOES sweep the refiner in — proves the collateral counter
    # is live (a hard-coded ``collateral: 0`` would pass every assertion above).
    loose = r".*\.(" + "|".join(re.escape(leaf) for leaf in H3_LORA_LEAVES) + ")"
    report = check_lora_targets_regex(_H3_NAMES, loose)
    assert report["total"] == 312
    assert report["collateral"] == EXPECTED_COLLATERAL_MATCHES
    assert report["main"] == EXPECTED_MAIN_MATCHES
    assert all("token_refiner" in n for n in report["collateral_names"])
    assert len(report["collateral_names"]) <= 3


# --------------------------------------------------------------------------------------------------
# The probe must mirror PEFT, not approximate it
# --------------------------------------------------------------------------------------------------


def test_regex_probe_uses_fullmatch_not_search() -> None:
    # PEFT matches a bare-``str`` ``target_modules`` with ``re.fullmatch``. ``re.search`` would
    # over-report (and the probe's whole job is to predict PEFT's injection count exactly).
    trailing = ["transformer_blocks.0.attn.to_q.base_layer"]
    assert re.search(H3_LORA_TARGET_REGEX, trailing[0]) is not None  # search WOULD match
    assert check_lora_targets_regex(trailing, H3_LORA_TARGET_REGEX)["total"] == 0


def test_regex_probe_agrees_with_peft_target_matching() -> None:
    """Cross-check every synthetic name against PEFT's own matcher — probe == reality."""
    tuners_utils = pytest.importorskip("peft.tuners.tuners_utils")
    check_target_module_exists = getattr(
        tuners_utils, "check_target_module_exists", None
    )
    if check_target_module_exists is None:  # pragma: no cover — very old peft
        pytest.skip("peft.tuners.tuners_utils.check_target_module_exists unavailable")

    cfg = build_lora_config(rank=8, alpha=8, targets=H3_LORA_TARGET_REGEX)
    peft_matched = [n for n in _H3_NAMES if bool(check_target_module_exists(cfg, n))]
    assert len(peft_matched) == EXPECTED_MAIN_MATCHES, len(peft_matched)
    assert all(n.startswith("transformer_blocks.") for n in peft_matched)


# --------------------------------------------------------------------------------------------------
# build_lora_config — the live char-explosion bug
# --------------------------------------------------------------------------------------------------


def test_build_lora_config_passes_a_regex_string_through_unexploded() -> None:
    cfg = build_lora_config(targets=H3_LORA_TARGET_REGEX)
    assert isinstance(cfg.target_modules, str)
    assert cfg.target_modules == H3_LORA_TARGET_REGEX
    # The bug this pins: ``list("ab")`` == ``["a", "b"]`` — PEFT would then match nothing.
    assert cfg.target_modules != list(H3_LORA_TARGET_REGEX)


def test_build_lora_config_ltx_list_path_is_unchanged() -> None:
    # LTX semantics byte-identical: PEFT itself normalises a list to a set in ``__post_init__``,
    # so the invariant is set-equality (exactly what ``tests/test_lora.py`` already pins).
    cfg = build_lora_config(targets=P1_FF_LORA_TARGETS)
    assert not isinstance(cfg.target_modules, str)
    assert set(cfg.target_modules) == set(P1_FF_LORA_TARGETS)
    assert build_lora_config().target_modules == cfg.target_modules  # default is the LTX list


def test_build_lora_config_keeps_rank_alpha_and_bias_for_the_h3_path() -> None:
    cfg = build_lora_config(rank=64, alpha=64, targets=H3_LORA_TARGET_REGEX)
    assert cfg.r == 64
    assert cfg.lora_alpha == 64  # scale 1.0 — the house clean-convert rule
    assert cfg.lora_dropout == 0.0
    assert cfg.bias == "none"


# --------------------------------------------------------------------------------------------------
# The LTX schema default can never silently apply to H3
# --------------------------------------------------------------------------------------------------


def test_ltx_attn_defaults_match_zero_h3_modules() -> None:
    # H3 is single-stream: ``attn.*``. LTX's ``attn1.*``/``attn2.*`` suffixes cannot match it.
    report = check_lora_targets(_H3_NAMES, P1_FF_LORA_TARGETS)
    for target in P1_FF_LORA_TARGETS:
        if target.startswith("attn"):
            assert report[target]["total"] == 0, f"{target} matched {report[target]['total']}"


def test_ltx_default_on_h3_fails_silently_not_loud() -> None:
    """The load-bearing correction: an H3 run on the LTX default does NOT raise — it under-trains.

    ``train/loop.py`` raises "No trainable parameters found. Is LoRA applied?" only when the
    matched set is EMPTY. On H3 the LTX default still matches the two ``ff.net`` suffixes (they are
    byte-identical across families) — 100 main + 4 refiner modules. So the guard does NOT fire:
    the run proceeds with an attn-BLIND, refiner-polluted, 1/3-capacity adapter. The family-selected
    default (Plan 10-03) is therefore a correctness requirement, not an ergonomic one.
    """
    report = check_lora_targets(_H3_NAMES, P1_FF_LORA_TARGETS)
    total = sum(r["total"] for r in report.values())
    assert total > 0, "if this ever becomes 0 the loop's guard fires and the failure IS loud"
    assert total == 2 * (H3_MAIN_BLOCKS + H3_REFINER_BLOCKS) == 104
    # ...and none of the attention capacity is reached.
    attn_total = sum(report[t]["total"] for t in P1_FF_LORA_TARGETS if t.startswith("attn"))
    assert attn_total == 0


def test_check_lora_targets_keeps_its_existing_return_keys() -> None:
    # The LTX probe's contract is untouched by the H3 sibling.
    report = check_lora_targets(_H3_NAMES, ["ff.net.2"])
    assert set(report["ff.net.2"]) == {"total", "video", "audio", "audio_names"}


# --------------------------------------------------------------------------------------------------
# End-to-end: PEFT really injects the main stack only, and the H3 GC-chain entry fires
# --------------------------------------------------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.to_out = nn.ModuleList([nn.Linear(d, d)])
        self.norm_q = nn.LayerNorm(d)


class _GEGLU(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d, d)


class _FF(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([_GEGLU(d), nn.Identity(), nn.Linear(d, d)])


class _AdaLnProj(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d)


class _Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.attn = _Attn(d)
        self.ff = _FF(d)
        self.adaln_proj = _AdaLnProj(d)
        self.norm1 = nn.LayerNorm(d)


class _TokenRefiner(nn.Module):
    def __init__(self, d: int, n: int) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList([_Block(d) for _ in range(n)])


class _FakeH3(nn.Module):
    """H3-shaped module tree exposing ONLY ``enable_gradient_checkpointing`` (diffusers' name)."""

    def __init__(self, d: int = 8, blocks: int = 2, refiner_blocks: int = 1) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(d) for _ in range(blocks)])
        self.token_refiner = _TokenRefiner(d, refiner_blocks)
        self.proj_in = nn.Linear(d, d)
        self.proj_out = nn.Linear(d, d)
        self.gc_enabled = False

    def enable_gradient_checkpointing(self) -> None:
        self.gc_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover — not exercised
        return x


def test_inject_lora_targets_only_the_main_stack_on_a_fake_h3() -> None:
    model = _FakeH3(blocks=2, refiner_blocks=1)
    wrapped = inject_lora(model, build_lora_config(rank=2, alpha=2, targets=H3_LORA_TARGET_REGEX))

    injected = sorted(
        n.replace("base_model.model.", "", 1).rsplit(".lora_A", 1)[0]
        for n, _ in wrapped.named_modules()
        if n.endswith("lora_A")
    )
    assert len(injected) == 2 * 6, injected  # 2 blocks x 6 leaves
    assert all(n.startswith("transformer_blocks.") for n in injected), injected
    assert not any("token_refiner" in n for n in injected), injected
    assert not any("adaln_proj" in n for n in injected), injected


def test_inject_lora_enables_gradient_checkpointing_via_the_h3_method() -> None:
    # TRAIN-06: GC is enabled on the BASE module BEFORE ``get_peft_model`` wraps it. H3's diffusers
    # module exposes neither ``set_gradient_checkpointing`` nor ``gradient_checkpointing_enable``.
    model = _FakeH3()
    wrapped = inject_lora(model, build_lora_config(rank=2, alpha=2, targets=H3_LORA_TARGET_REGEX))
    assert model.gc_enabled is True
    assert wrapped.training is True


def test_inject_lora_prefers_the_ltx_gc_methods_when_present() -> None:
    """LTX order is byte-identical: the pre-existing branches still win the hasattr chain."""

    class _LtxShaped(_FakeH3):
        def __init__(self) -> None:
            super().__init__()
            self.ltx_gc = False

        def set_gradient_checkpointing(self, value: bool) -> None:
            self.ltx_gc = value

    model = _LtxShaped()
    inject_lora(model, build_lora_config(rank=2, alpha=2, targets=H3_LORA_TARGET_REGEX))
    assert model.ltx_gc is True
    assert model.gc_enabled is False  # the H3 branch did NOT fire
