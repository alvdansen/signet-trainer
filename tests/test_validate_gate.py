"""Structural CPU test for the validation gate (the GPU checks run in the 03-07 preflight).

This proves the gate's STRUCTURAL surface on CI — WITHOUT the 22B model, a checkpoint, ltx_core,
or CUDA. The six GPU checks (#1 dimensions on a live model, #2 Gemma config, #4/#5/#6 forward/
backward/roundtrip) run only Modal-side in the metered preflight; here we assert:

  (1) ``signet_trainer.train.validate_gate`` imports cleanly with NO ltx_core / ltx_trainer / modal
      pulled in (heavy imports are function-local — mirrors ``tests/test_loop_purity.py``);
  (2) the dimension constants the gate asserts against are the SAME single-source values imported
      from ``signet_trainer.models.loader`` (no duplicated literals);
  (3) check #3 probes the set THE RUN WILL INJECT (``config.lora.target_modules``) with the
      PEFT-faithful matchers from ``lora/peft.py`` — including the a2v regression (a video-only
      arch must FAIL the gate when the run injects the 14-entry a2v set), the dot-bounded matcher
      semantics (no ``xattn1.to_q`` coincidence match), and the bare-``str`` regex (H3) form —
      against FAKE ``nn.Module``s whose ``named_modules()`` mimic the real key layouts;
  (4) ``CheckResult`` is a structured record with ``name`` / ``status`` (+ ``hard_gate``) fields and
      check #6 is flagged as the hard gate.

It does NOT validate the GPU checks — that is the 03-07 metered preflight, deliberately out of scope.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

# The heavy modules the gate must NOT pull in at import time (function-local only).
_FORBIDDEN_MODULES = ("modal", "ltx_core", "ltx_trainer")

_VALIDATE_GATE_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "signet_trainer"
    / "train"
    / "validate_gate.py"
)


# --------------------------------------------------------------------------------------------------
# (1) import-confinement — the gate imports without ltx_core (heavy imports are function-local)
# --------------------------------------------------------------------------------------------------


def test_validate_gate_imports_without_ltx_core() -> None:
    """Importing the gate must NOT transitively pull a heavy ltx_core / ltx_trainer / modal dep."""
    for mod in _FORBIDDEN_MODULES:
        sys.modules.pop(mod, None)

    import signet_trainer.train.validate_gate as gate  # noqa: F401

    for mod in _FORBIDDEN_MODULES:
        assert mod not in sys.modules, (
            f"validate_gate import pulled in {mod!r} — heavy ltx imports must be function-local "
            f"so the structural scan runs on CI without ltx_core installed"
        )

    assert hasattr(gate, "run_validation_gate")
    assert callable(gate.run_validation_gate)


# --------------------------------------------------------------------------------------------------
# (2) EXPECTED_* single source — the gate uses loader.py's constants, not duplicated literals
# --------------------------------------------------------------------------------------------------


def test_gate_dimension_constants_are_loader_single_source() -> None:
    """The constants the gate asserts against ARE the loader.py EXPECTED_* values (not re-typed)."""
    from signet_trainer.models import loader
    from signet_trainer.train import validate_gate

    # Identity of value: the gate must import these names from loader (same object/value).
    assert validate_gate.EXPECTED_NUM_BLOCKS is loader.EXPECTED_NUM_BLOCKS
    assert validate_gate.EXPECTED_HIDDEN_DIM is loader.EXPECTED_HIDDEN_DIM
    assert validate_gate.EXPECTED_T2V_IN_CHANNELS is loader.EXPECTED_T2V_IN_CHANNELS

    # The known ground-truth values (48 / 4096 / 128) — asserted here so a silent drift in loader.py
    # is caught at the gate's contract boundary too.
    assert loader.EXPECTED_NUM_BLOCKS == 48
    assert loader.EXPECTED_HIDDEN_DIM == 4096
    assert loader.EXPECTED_T2V_IN_CHANNELS == 128


def test_gate_source_imports_constants_not_redefines() -> None:
    """The gate SOURCE imports EXPECTED_* from loader and does not assign them as new literals."""
    src = _VALIDATE_GATE_SRC.read_text(encoding="utf-8")
    assert "from signet_trainer.models.loader import" in src
    for const in ("EXPECTED_NUM_BLOCKS", "EXPECTED_HIDDEN_DIM", "EXPECTED_T2V_IN_CHANNELS"):
        # No re-definition like ``EXPECTED_NUM_BLOCKS = 48`` in the gate (single source = loader.py).
        assert f"{const} = " not in src, (
            f"{const} must be IMPORTED from loader.py, not redefined in validate_gate.py"
        )
        assert f"{const}:" not in src.replace("EXPECTED_GEMMA", ""), (
            f"{const} must not be annotated/redefined in validate_gate.py"
        )


# --------------------------------------------------------------------------------------------------
# (3) check #3 matcher — resolve the locked target list (incl. ff.net) against a FAKE module
# --------------------------------------------------------------------------------------------------


class _FakeBlock(nn.Module):
    """A fake transformer block exposing the real attn1/attn2/ff.net submodule names."""

    def __init__(self) -> None:
        super().__init__()
        self.attn1 = nn.Module()
        self.attn1.to_q = nn.Linear(4, 4)
        self.attn1.to_k = nn.Linear(4, 4)
        self.attn1.to_v = nn.Linear(4, 4)
        self.attn1.to_out = nn.ModuleList([nn.Linear(4, 4)])  # to_out.0
        self.attn2 = nn.Module()
        self.attn2.to_q = nn.Linear(4, 4)
        self.attn2.to_k = nn.Linear(4, 4)
        self.attn2.to_v = nn.Linear(4, 4)
        self.attn2.to_out = nn.ModuleList([nn.Linear(4, 4)])  # to_out.0
        # ff.net.0.proj and ff.net.2 — the ff.net FIX entries.
        self.ff = nn.Module()
        proj = nn.Module()
        proj.proj = nn.Linear(4, 8)
        self.ff.net = nn.ModuleList([proj, nn.GELU(), nn.Linear(8, 4)])  # net.0.proj, net.2


class _FakeTransformer(nn.Module):
    """A fake transformer with a small ``transformer_blocks`` stack (mimics the real key layout)."""

    def __init__(self, n_blocks: int = 2) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_FakeBlock() for _ in range(n_blocks))


def test_check3_matcher_resolves_full_target_list_on_fake_module() -> None:
    """``resolve_lora_targets`` matches all P1_FF_LORA_TARGETS (incl. ff.net) on a fake module."""
    from signet_trainer.lora.peft import P1_FF_LORA_TARGETS
    from signet_trainer.train.validate_gate import resolve_lora_targets

    resolution = resolve_lora_targets(_FakeTransformer(n_blocks=2), P1_FF_LORA_TARGETS)

    assert resolution["unmatched"] == [], (
        f"all locked targets must resolve on the fake arch; unmatched={resolution['unmatched']}"
    )
    assert set(resolution["matched"]) == set(P1_FF_LORA_TARGETS)

    # The ff.net entries (the fix) MUST be present and counted (2 blocks -> 2 instances each).
    assert resolution["counts"]["ff.net.0.proj"] == 2
    assert resolution["counts"]["ff.net.2"] == 2
    assert resolution["counts"]["attn1.to_q"] == 2
    assert resolution["counts"]["attn2.to_out.0"] == 2


def test_check3_matcher_reports_unmatched_target() -> None:
    """A target with no matching module is reported as unmatched (the FAIL signal on a bad arch)."""
    from signet_trainer.train.validate_gate import resolve_lora_targets

    resolution = resolve_lora_targets(
        _FakeTransformer(n_blocks=1), ["attn1.to_q", "nonexistent.module"]
    )
    assert "attn1.to_q" in resolution["matched"]
    assert resolution["unmatched"] == ["nonexistent.module"]


def test_check3_matcher_is_dot_bounded_like_peft() -> None:
    """The matcher mirrors PEFT: ``attn1.to_q`` must NOT match a coincidental ``xattn1.to_q``.

    The original port used a bare ``name.endswith(target)``, which is looser than PEFT's
    dot-bounded suffix matching — the gate would count modules PEFT never injects into.
    """
    from signet_trainer.train.validate_gate import resolve_lora_targets

    decoy = nn.Module()
    decoy.xattn1 = nn.Module()
    decoy.xattn1.to_q = nn.Linear(4, 4)
    resolution = resolve_lora_targets(decoy, ["attn1.to_q"])
    assert resolution["unmatched"] == ["attn1.to_q"], (
        "a coincidental 'xattn1.to_q' must not satisfy target 'attn1.to_q' (PEFT is dot-bounded)"
    )


# --------------------------------------------------------------------------------------------------
# (3b) check #3 probes the CONFIG's injected set — the a2v-blind-adapter regression
# --------------------------------------------------------------------------------------------------


def _lora_config_ns(target_modules):
    """A minimal config namespace shaped like SignetConfig for the gate's getattr chain."""
    from types import SimpleNamespace

    return SimpleNamespace(lora=SimpleNamespace(target_modules=target_modules))


def test_check3_fails_when_a2v_targets_missing_from_arch() -> None:
    """An a2v run (14-entry injected set) on a video-only arch must FAIL check #3.

    Regression for the gate-honesty blocker: probing the hardcoded P1_FF_LORA_TARGETS reported
    'PASS 10/10' while all four cross-modal audio_to_video_attn.* targets were unmatched — the
    metered run would train an adapter structurally blind to the driving audio.
    """
    from types import SimpleNamespace

    from signet_trainer.lora.peft import A2V_CROSS_MODAL_LORA_TARGETS, P1_A2V_LORA_TARGETS
    from signet_trainer.train.validate_gate import check_lora_targets

    components = SimpleNamespace(transformer=_FakeTransformer(n_blocks=2))
    result = check_lora_targets(components, _lora_config_ns(list(P1_A2V_LORA_TARGETS)))

    assert result.status == "FAIL", result.message
    assert set(result.details["unmatched"]) == set(A2V_CROSS_MODAL_LORA_TARGETS)
    for target in A2V_CROSS_MODAL_LORA_TARGETS:
        assert target in result.message  # the refusal names the missing cross-modal targets


def test_check3_passes_when_config_set_resolves_and_counts_are_reported() -> None:
    """The locked LTX set (config-carried) passes on the fake LTX arch, with per-target counts."""
    from types import SimpleNamespace

    from signet_trainer.lora.peft import P1_FF_LORA_TARGETS
    from signet_trainer.train.validate_gate import check_lora_targets

    components = SimpleNamespace(transformer=_FakeTransformer(n_blocks=2))
    result = check_lora_targets(components, _lora_config_ns(list(P1_FF_LORA_TARGETS)))

    assert result.status == "PASS", result.message
    assert result.details["counts"]["ff.net.2"] == 2
    assert "per-target counts" in result.message


def test_check3_falls_back_to_locked_list_when_config_carries_no_targets() -> None:
    """target_modules None -> the same ``or P1_FF_LORA_TARGETS`` fallback check #5 uses."""
    from types import SimpleNamespace

    from signet_trainer.lora.peft import P1_FF_LORA_TARGETS
    from signet_trainer.train.validate_gate import check_lora_targets

    components = SimpleNamespace(transformer=_FakeTransformer(n_blocks=2))
    result = check_lora_targets(components, _lora_config_ns(None))

    assert result.status == "PASS", result.message
    assert set(result.details["matched"]) == set(P1_FF_LORA_TARGETS)


class _FakeH3Block(nn.Module):
    """A fake H3 single-stream block (``attn.*`` + ``ff.net.*`` leaves)."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(4, 4)
        self.attn.to_k = nn.Linear(4, 4)
        self.attn.to_v = nn.Linear(4, 4)
        self.attn.to_out = nn.ModuleList([nn.Linear(4, 4)])
        self.ff = nn.Module()
        proj = nn.Module()
        proj.proj = nn.Linear(4, 8)
        self.ff.net = nn.ModuleList([proj, nn.GELU(), nn.Linear(8, 4)])


class _FakeH3Transformer(nn.Module):
    def __init__(self, n_blocks: int = 2) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_FakeH3Block() for _ in range(n_blocks))


def test_check3_bare_str_target_is_probed_as_regex_not_char_exploded() -> None:
    """A bare-``str`` target (the H3 PEFT-regex form) routes to the regex probe, never list(str).

    Iterating the string as a list would probe single characters ('t', 'r', ...) — the
    char-explosion hazard the schema's ``list[str] | str`` union carries.
    """
    from types import SimpleNamespace

    from signet_trainer.lora.peft import H3_LORA_TARGET_REGEX
    from signet_trainer.train.validate_gate import check_lora_targets

    components = SimpleNamespace(transformer=_FakeH3Transformer(n_blocks=2))
    result = check_lora_targets(components, _lora_config_ns(H3_LORA_TARGET_REGEX))

    assert result.status == "PASS", result.message
    assert result.details["total"] == 12  # 2 blocks x 6 leaves
    assert all(n == 2 for n in result.details["per_leaf"].values())


def test_check3_bare_str_h3_regex_fails_on_per_leaf_zero() -> None:
    """The H3 family regex on an attn-less arch FAILs (per-leaf zero), naming the cold leaves."""
    from types import SimpleNamespace

    from signet_trainer.lora.peft import H3_LORA_TARGET_REGEX
    from signet_trainer.train.validate_gate import check_lora_targets

    # An arch carrying ONLY the ff leaves: the four attn.* leaves match zero modules.
    class _FfOnlyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ff = nn.Module()
            proj = nn.Module()
            proj.proj = nn.Linear(4, 8)
            self.ff.net = nn.ModuleList([proj, nn.GELU(), nn.Linear(8, 4)])

    class _FfOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList([_FfOnlyBlock()])

    components = SimpleNamespace(transformer=_FfOnly())
    result = check_lora_targets(components, _lora_config_ns(H3_LORA_TARGET_REGEX))

    assert result.status == "FAIL", result.message
    assert "ZERO-MATCH leaves" in result.message


# --------------------------------------------------------------------------------------------------
# (4) CheckResult is a structured record + check #6 is the hard gate
# --------------------------------------------------------------------------------------------------


def test_checkresult_is_structured_record() -> None:
    """``CheckResult`` carries name/status (+ details/duration/hard_gate) fields."""
    from signet_trainer.train.validate_gate import CheckResult

    r = CheckResult(name="check_x", status="PASS", message="ok")
    assert r.name == "check_x"
    assert r.status == "PASS"
    assert r.details == {}
    assert r.duration_s == 0.0
    assert r.hard_gate is False


def test_frame_constraints_check_is_cpu_runnable() -> None:
    """The frame-constraint check runs on CPU (no GPU/ltx_core) and accepts the locked dims."""
    from types import SimpleNamespace

    from signet_trainer.train.validate_gate import check_frame_constraints

    # (768, 512, 49): (49-1)%8==0, 768%32==0, 512%32==0 -> PASS.
    ok = check_frame_constraints(SimpleNamespace(training_dims=(768, 512, 49)))
    assert ok.status == "PASS", ok.message

    # A misaligned frame count fails fast.
    bad = check_frame_constraints(SimpleNamespace(training_dims=(768, 512, 50)))
    assert bad.status == "FAIL"


def test_check6_is_flagged_hard_gate() -> None:
    """Check #6 (no injected adapter path) returns a hard-gate-flagged FAIL result."""
    from signet_trainer.train.validate_gate import check_p1_to_p2

    result = check_p1_to_p2(None)
    assert result.name == "check_p1_to_p2"
    assert result.hard_gate is True
    assert result.status == "FAIL"  # no adapter -> the roundtrip cannot run


# --------------------------------------------------------------------------------------------------
# (5) check_dimensions semantics — fail on a RESOLVED contradiction, tolerate probe-blind None
# --------------------------------------------------------------------------------------------------


def _dim_components(*, n_blocks=48, hidden=4096):
    from types import SimpleNamespace

    transformer = SimpleNamespace(
        transformer_blocks=list(range(n_blocks)),  # len -> num_blocks
        config=SimpleNamespace(hidden_size=hidden),  # -> hidden_dim via _probe
    )
    return SimpleNamespace(
        transformer=transformer, text_encoder=None, video_vae_encoder=None, scheduler=None
    )


def test_check_dimensions_passes_when_strong_dims_verified_and_in_channels_unverified() -> None:
    """num_blocks + hidden_dim resolve to EXPECTED; in_channels stays UNVERIFIED (probe gap) -> PASS.

    This is the real-model case that WAS wrongly failing: LTX-2.3 hides in_channels from live
    introspection, but blocks (48) + inner_dim (4096) are authoritative -> the arch is verified.
    """
    from signet_trainer.train.validate_gate import check_dimensions

    r = check_dimensions(_dim_components(), checkpoint_path=None)
    assert r.status == "PASS", r.message
    assert r.details["per_dim"]["in_channels"]["status"] == "UNVERIFIED"


def test_check_dimensions_fails_on_resolved_contradiction() -> None:
    """A source that RESOLVES a wrong value (num_blocks=99, not 48) is a real arch mismatch -> FAIL."""
    from signet_trainer.train.validate_gate import check_dimensions

    r = check_dimensions(_dim_components(n_blocks=99), checkpoint_path=None)
    assert r.status == "FAIL"
    assert "MISMATCH" in r.message


def test_check_dimensions_fails_when_strong_dim_unverifiable() -> None:
    """If neither num_blocks nor hidden_dim can be resolved from ANY source -> FAIL (no free pass)."""
    from types import SimpleNamespace

    from signet_trainer.train.validate_gate import check_dimensions

    blind = SimpleNamespace(
        transformer=SimpleNamespace(config=SimpleNamespace()),  # no blocks, no hidden_size
        text_encoder=None,
        video_vae_encoder=None,
        scheduler=None,
    )
    r = check_dimensions(blind, checkpoint_path=None)
    assert r.status == "FAIL"


# --------------------------------------------------------------------------------------------------
# (6) check_text_encoder — resolve Gemma 3's nested config.text_config.hidden_size
# --------------------------------------------------------------------------------------------------


def test_check_text_encoder_resolves_nested_text_config() -> None:
    """Gemma 3 nests hidden_size under config.text_config; the check must find 3840 there -> PASS."""
    from types import SimpleNamespace

    from signet_trainer.train.validate_gate import check_text_encoder

    te = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=3840)))
    components = SimpleNamespace(text_encoder=te)
    r = check_text_encoder(components)
    assert r.status == "PASS", r.message


def test_check_text_encoder_fails_on_stale_3072() -> None:
    """A stale Flimmer hidden_size (3072) still fails — the fix must not paper over a real mismatch."""
    from types import SimpleNamespace

    from signet_trainer.train.validate_gate import check_text_encoder

    te = SimpleNamespace(config=SimpleNamespace(hidden_size=3072))
    r = check_text_encoder(SimpleNamespace(text_encoder=te))
    assert r.status == "FAIL"


def test_check_text_encoder_warns_when_present_but_unresolvable() -> None:
    """A loaded-but-unintrospectable Gemma is advisory (WARN), NOT a hard FAIL that blocks the run.

    The encoder is present (forward checks exercise it) but exposes no known hidden_size path — the
    gate must not abort a real run over a probe gap (validate_ltx23.py treats this as WARN).
    """
    from types import SimpleNamespace

    from signet_trainer.train.validate_gate import check_text_encoder

    te = SimpleNamespace()  # present, but no config / embeddings to read a hidden_size from
    r = check_text_encoder(SimpleNamespace(text_encoder=te))
    assert r.status == "WARN"


def test_gate_video_inner_dim_constant_is_loader_single_source() -> None:
    """The 2048 video inner-dim fingerprint is imported from loader.py (single source), == 2048."""
    from signet_trainer.models import loader
    from signet_trainer.train import validate_gate

    assert validate_gate.EXPECTED_VIDEO_INNER_DIM is loader.EXPECTED_VIDEO_INNER_DIM
    assert loader.EXPECTED_VIDEO_INNER_DIM == 4096  # video branch (audio_attn1 is the 2048 misread)
