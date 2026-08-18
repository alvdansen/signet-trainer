"""D-10-DEF-19 — the H3 adapter column must render from MERGED weights, and resume must be pinnable.

WHAT THIS GUARDS. ``h3_sample`` renders base-vs-adapter from ONE transformer. The base column runs
under ``disable_adapter()`` and allocates exactly what a plain forward allocates. The adapter column,
if the adapter is left UNMERGED, runs peft's

    result = result + lora_B(lora_A(dropout(x))) * scaling        # peft 0.20.0 lora/layer.py:1058

at every target Linear — and on the H3 block the widest target is the SwiGLU up-projection
``ff.net.0.proj``, so ``lora_B``'s output alone is another tensor the size of the base output.
MEASURED on render app ``ap-ENw0kWTvmlfm4WhoYtc7KT``: six base clips written, then
``Tried to allocate 6.88 GiB / 6.70 GiB free / 72.54 GiB in use`` in transformer block 0 of denoise
step 0, with only 103 MiB reserved-but-unallocated (not fragmentation) and the offload strategy
already logging ``offloading all models`` (not residency). Merging folds the delta into
``base_layer.weight`` once, in weight space, and peft then takes its ``elif self.merged`` branch —
the base column's allocation profile exactly.

WHY A SOURCE SCAN. The failure mode is a ~2 h metered A100 render that completes its first column
and dies on its second, so there is no cheap runtime assertion: reproducing it costs the render. And
the three properties that matter are STRUCTURAL — that the merge exists, that it sits BETWEEN the two
columns (before it, the base column would render from merged weights and the comparison would be a
lie; after it, nothing is fixed), and that its "did it take" check is not dropped by a tidy-up. Same
convention as ``tests/test_h3_preprocess_wiring.py``: comments and docstrings are stripped FIRST,
because this module's own prose names everything it guards and an un-stripped scan would pass on the
warnings alone.

⚠ NOT ``from signet_trainer.modal.fns import ...``. Importing that module pulls ``modal`` into
``sys.modules`` for the whole pytest session and breaks the dry-run gate's Anti-Pattern-6 assertion
in an unrelated file — a failure that only appears when the suite runs in full. The scan reads the
file as TEXT.

CPU-only, zero GPU, zero Modal spend.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _function_source(path: Path, name: str) -> str:
    """The source of one top-level function, comments and docstrings stripped."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return _strip_comments_and_docstrings(ast.get_source_segment(path.read_text(encoding="utf-8"), node) or "")
    raise AssertionError(f"{path.name} has no top-level function {name!r}")


@pytest.fixture(scope="module")
def h3_sample_src() -> str:
    return _function_source(_FNS, "h3_sample")


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# The merge — existence, placement, and the proof that it took
# ══════════════════════════════════════════════════════════════════════════════════════════════════


def test_the_adapter_column_renders_from_merged_weights(h3_sample_src: str) -> None:
    """``merge_adapter()`` is called at all. Without it the adapter column OOMs on one A100-80GB."""
    assert "merge_adapter()" in h3_sample_src, (
        "h3_sample no longer merges the adapter. Unmerged, every LoRA target allocates lora_B's "
        "output on top of the base output it is added to; on ff.net.0.proj at the campaign geometry "
        "that was a 6.88 GiB allocation against 6.70 GiB free (D-10-DEF-19, render app "
        "ap-ENw0kWTvmlfm4WhoYtc7KT). The base column is not evidence that the adapter column fits."
    )


def test_the_merge_sits_between_the_two_columns(h3_sample_src: str) -> None:
    """Placement is the whole fix — before the base column it would corrupt the comparison.

    The base column MUST render unmerged under ``disable_adapter()`` (that is what makes it the
    base), and the adapter column MUST render merged (that is what makes it fit). A merge hoisted up
    next to ``inject_lora`` would make peft lazily ``unmerge()`` inside the base forward and re-merge
    afterwards — correct output, twice the merge cost — while a merge after the adapter loop fixes
    nothing at all.
    """
    base_col = h3_sample_src.index("with adapted.disable_adapter():")
    merge = h3_sample_src.index("merge_adapter()")
    lora_col = h3_sample_src.index("lora_mp4s: dict[str, str] = {}")

    assert base_col < merge, (
        "the merge happens BEFORE the base column. The base column would then render from weights "
        "that already contain the adapter — two identically-adapted columns under 'base' and 'lora' "
        "labels, which is worse than no grid (the same class of silent mislabel D-10-SCOPEGUARD and "
        "the render-key design exist to prevent)."
    )
    assert merge < lora_col, (
        "the merge happens AFTER the adapter column starts rendering. The OOM it fixes is in the "
        "first forward of the first adapter clip, so a merge placed here never runs."
    )


def test_the_merge_is_config_gated_not_hardcoded(h3_sample_src: str) -> None:
    """D-NOHARDCODE: the residency strategy is a documented config field, and it guards the call."""
    guard = h3_sample_src.index("config.h3.render_merge_adapter")
    merge = h3_sample_src.index("merge_adapter()")
    assert guard < merge, (
        "h3.render_merge_adapter must GATE the merge, not merely be mentioned after it. A knob that "
        "does not reach the behaviour it names is worse than no knob."
    )


def test_the_merge_is_verified_before_the_metered_column(h3_sample_src: str) -> None:
    """A silent no-op merge does not fail here — it fails 20 minutes later, on a billing A100.

    ``merge_adapter`` downgrades "nothing to merge" to a ``warnings.warn``
    (peft ``tuners_utils.check_adapters_to_merge``), and a warning on a Modal container is a line in
    a log nobody reads until the OOM. The count-and-raise turns that into an abort that costs cents.
    """
    tail = h3_sample_src[h3_sample_src.index("merge_adapter()") :]
    assert 'getattr(m, "merged", False)' in tail, (
        "the post-merge verification is gone. Restore a check that COUNTS merged LoRA layers off "
        "the live module tree — not a flag, not a return value, the modules themselves."
    )
    assert "merged_layers == 0" in tail and "raise RuntimeError" in tail, (
        "merging without proving it took is the whole D-10-DEF-19 failure re-armed: the adapter "
        "column renders at the unmerged cost and OOMs, or renders base pixels under a lora label."
    )


def test_both_columns_report_a_measured_peak(h3_sample_src: str) -> None:
    """D-10-DEF-19 had to reconstruct the base peak from an OOM string. One counter read per column
    means the next residency question is answered with data rather than arithmetic."""
    assert h3_sample_src.count("max_memory_allocated()") >= 1, (
        "the per-column peak-VRAM report is gone. It is free (a counter read) and it is the only "
        "thing that made the next OOM diagnosable without paying for another render."
    )
    assert "reset_peak_memory_stats()" in h3_sample_src, (
        "the peak watermark is never reset, so the second column's number is really the max of both "
        "columns — which is exactly the ambiguity the report exists to remove."
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# The offload margin — the FIRST suspect, and NOT the cause
# ══════════════════════════════════════════════════════════════════════════════════════════════════


def test_the_offload_margin_is_config_first(h3_sample_src: str) -> None:
    """D-NOHARDCODE. It was a ``"12GB"`` literal in two places; a per-GPU threshold is a config field.

    Both call sites must read the same field: ``ComponentsManager.add()`` silently re-enables auto
    offload with the ``"3GB"`` default when a new component is registered, so the re-assert after
    ``update_components`` is load-bearing — and two sites that could drift apart is a threshold that
    is really two thresholds.
    """
    assert '"12GB"' not in h3_sample_src and "'12GB'" not in h3_sample_src, (
        "the offload margin is a literal again. It is a per-GPU threshold (D-NOHARDCODE) and an "
        "H200 escalation must be a YAML edit."
    )
    assert h3_sample_src.count("memory_reserve_margin=config.h3.render_offload_reserve") == 2, (
        "expected exactly two enable_auto_cpu_offload call sites, both fed from "
        "h3.render_offload_reserve: the initial one and the re-assert after update_components "
        "(ComponentsManager.add() reverts it to the 3GB default when the PEFT wrapper registers)."
    )


def test_the_log_line_no_longer_calls_the_margin_a_reserve(h3_sample_src: str) -> None:
    """The old log said a 12GB reserve had been "re-asserted", which reads as VRAM held open.

    It is not. ``AutoOffloadStrategy`` consults the margin only from ``CustomOffloadHook.pre_forward``
    and only for a component that is not already on the device, purely to decide WHICH OTHER
    components to evict. Once a component is resident its activations may take every remaining byte.
    That misreading is what sent the first D-10-DEF-19 hypothesis at the margin instead of at the
    activation arithmetic, so the string itself is worth pinning.
    """
    assert "offload reserve after update_components" not in h3_sample_src, (
        "the log line calls the margin a 'reserve' again. Say eviction THRESHOLD — an operator "
        "reading 'reserve' will conclude, wrongly, that free VRAM at render time is a bug."
    )
    assert "eviction threshold" in h3_sample_src.lower(), (
        "neither log line states what the margin actually is. The verified semantics belong in the "
        "container log, where the next person debugging a render will be looking."
    )


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# The checkpoint pin — what makes the identity-keyed resume survive a live training run
# ══════════════════════════════════════════════════════════════════════════════════════════════════


def test_the_pin_is_optional_and_find_latest_is_the_default(h3_sample_src: str) -> None:
    """Empty pin must still go through find_latest, byte-identically to before."""
    assert "config.h3.render_checkpoint_name" in h3_sample_src
    assert "CheckpointManager(ckpt_root).find_latest()" in h3_sample_src, (
        "the find_latest fallback is gone. signet finals ARE step-numbered and find_latest is the "
        "only correct resolution when no pin is declared; a config predating the field must render "
        "exactly as it did."
    )


def test_a_pinned_checkpoint_inherits_the_completeness_filter(h3_sample_src: str) -> None:
    """A pinned half-written dir must be as unusable as a walked-back one (D-9-DEDUP: one rule)."""
    assert "CheckpointManager.is_complete(" in h3_sample_src, (
        "the pin path does not apply the completeness filter. find_latest walks BACK past a "
        "half-written newer dir; resolving by NAME must refuse the same dir rather than load a "
        "checkpoint that is missing its adapter or its training state."
    )


def test_is_complete_is_the_same_rule_find_latest_filters_with(tmp_path: Path) -> None:
    """Behavioural, not structural: the public helper and find_latest agree on real directories."""
    pytest.importorskip("torch")
    from signet_trainer.train.checkpoint import (  # noqa: PLC0415
        ADAPTER_FILENAME,
        TRAINING_STATE_FILENAME,
        CheckpointManager,
    )

    root = tmp_path / "outputs" / "run"
    complete = root / "checkpoint-step-00100-loss-0.1000"
    half = root / "checkpoint-step-00200-loss-0.0900"
    for d in (complete, half):
        d.mkdir(parents=True)
    (complete / ADAPTER_FILENAME).write_bytes(b"x")
    (complete / TRAINING_STATE_FILENAME).write_bytes(b"x")
    (half / ADAPTER_FILENAME).write_bytes(b"x")  # adapter but no training state — half written

    assert CheckpointManager.is_complete(complete) is True
    assert CheckpointManager.is_complete(half) is False
    assert CheckpointManager.is_complete(root / "checkpoint-step-00300-loss-0.0") is False
    # find_latest walks BACK past the newer half-written dir — the same verdict, reached the other
    # way round. If these ever disagree, a pin can load something a walk-back would refuse.
    assert CheckpointManager(root).find_latest() == complete


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# The config surface
# ══════════════════════════════════════════════════════════════════════════════════════════════════


def test_render_knob_defaults_are_the_decided_values() -> None:
    pytest.importorskip("torch")
    from signet_trainer.config.schema import H3Config  # noqa: PLC0415

    pristine = H3Config()
    assert pristine.render_merge_adapter is True, (
        "merging must be the DEFAULT, not an opt-in. The A100-80GB single-card path is the "
        "documented one, and unmerged it does not fit."
    )
    assert pristine.render_offload_reserve == "12GB"
    assert pristine.render_checkpoint_name == "", (
        "the pin must default to empty so every config authored before D-10-DEF-19 keeps resolving "
        "find_latest."
    )


@pytest.mark.parametrize("bad", ["12 gigs", "GB", "12GBB", "twelve", "12 GB extra"])
def test_a_bad_offload_reserve_dies_at_config_load(bad: str) -> None:
    """Not Modal-side: convert_file_size_to_int raises inside enable_auto_cpu_offload, i.e. AFTER
    eight components have loaded off the Volume. A typo should cost nothing."""
    pytest.importorskip("torch")
    from signet_trainer.config.schema import H3Config  # noqa: PLC0415

    with pytest.raises(ValueError):
        H3Config(render_offload_reserve=bad)


@pytest.mark.parametrize("bad", ["/checkpoints/x", "../escape", "a/b", ".."])
def test_a_pin_that_is_a_path_is_refused(bad: str) -> None:
    """WR-09: ``ckpt_root / <value>`` lets an absolute value REPLACE the Volume prefix and '..'
    escape it. The pin names a directory, never a path."""
    pytest.importorskip("torch")
    from signet_trainer.config.schema import H3Config  # noqa: PLC0415

    with pytest.raises(ValueError):
        H3Config(render_checkpoint_name=bad)


def test_the_render_knobs_are_refused_under_a_non_h3_family() -> None:
    """The lean field-split REVERSE guard covers them automatically (it diffs a pristine H3Config),
    so an LTX config that sets one gets a config-load error instead of a silent no-op.

    Sources the SHIPPED ``configs/ltx23_lora.example.yaml`` rather than an unpublished in-house
    fixture (issue #41 finding 3): this is a pure library-behaviour assertion —
    ``load_config_from_text`` must refuse an ``h3:`` knob under a non-h3 family — and needs no
    private carrier to demonstrate that. Any non-h3 config would do; this one ships in every clone.
    """
    pytest.importorskip("torch")
    import yaml  # noqa: PLC0415

    from signet_trainer.config.load import load_config_from_text  # noqa: PLC0415

    raw = yaml.safe_load(
        (REPO / "configs" / "ltx23_lora.example.yaml").read_text(encoding="utf-8")
    )
    # No `model:` block at all — the LTX family is the schema default, which is precisely the
    # config shape a stray h3 knob would hide in.
    assert raw.get("model", {}).get("family", "ltx") != "h3", (
        "fixture drifted — this must be a NON-h3 config for the reverse guard to have anything to say"
    )
    raw.setdefault("h3", {})["render_merge_adapter"] = False

    with pytest.raises(ValueError, match="render_merge_adapter"):
        load_config_from_text(yaml.safe_dump(raw))


def test_the_campaign_config_pins_the_checkpoint_the_base_clips_belong_to() -> None:
    """The 124f render's six base clips are on the Volume under the step-00300 identity key.

    While r1 is still training, an unpinned re-dispatch resolves a newer checkpoint, lands in a fresh
    directory and re-renders all twelve clips. This is the one config that carries the pin; if it is
    cleared (which it SHOULD be once the fix is proven) this test goes with it.
    """
    pytest.importorskip("torch")
    from signet_trainer.config.load import load_config  # noqa: PLC0415

    config = load_config(str(REPO / "configs" / "h3_embe_r1_sample.yaml"))
    pin = config.h3.render_checkpoint_name
    if not pin:
        pytest.skip("pin cleared — the D-10-DEF-19 fix render is done, which is the intended end state")
    assert pin.startswith("checkpoint-step-"), (
        f"render_checkpoint_name={pin!r} is not a signet per-step checkpoint dir name. A flat "
        "'lora_weights.safetensors' is the CANONICAL ltx-trainer convention, not this stack's."
    )
