"""Audit PR-7 regression gates — dryrun honesty on the LTX side (gap-dryrun-ltx-0 / -1).

Two blockers, one bundle:

  (0) The free dry-run gate was MODE-BLIND: ``signet-dryrun`` PASSED the shipped multi_frame
      SAMPLE example that the metered ``train()`` container then refused at load — post-approval,
      detached, invisible inside the entrypoint's bounded watch window. The fix threads the
      entrypoint's ``--mode`` into ``run_dryrun(cfg, mode=...)`` and hoists the two
      mode-conditional refusals into ONE CPU-pure home (``config/mode_gate.py``) shared by the
      dry-run CLI, the entrypoint and the container bodies.

  (1) The LTX arm priced ``training_dims``, which NO LTX container path consumes — the trained
      sequence comes from ``data.resolution_buckets``. The fix prices the WORST bucket the way the
      H3 arm prices the worst reference pair: build + assert the synthetic batch at EVERY bucket,
      report the max.

CPU only — no ``modal`` / ltx_core / CUDA import (Pitfall 4 / Windows). Container/entrypoint call
sites are checked at SOURCE level for the same reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from signet_trainer.config.load import load_config
from signet_trainer.config.mode_gate import KNOWN_MODES, validate_mode_config
from signet_trainer.dryrun.shapes import main, run_dryrun

REPO_ROOT = Path(__file__).resolve().parents[1]
MULTI_FRAME_SAMPLE_YAML = str(REPO_ROOT / "configs" / "ltx23_multi_frame.example.yaml")
MULTI_FRAME_TRAIN_YAML = str(REPO_ROOT / "configs" / "ltx23_multi_frame_overfit.example.yaml")
IC_LORA_YAML = str(REPO_ROOT / "configs" / "ltx23_ic_lora.example.yaml")
ENTRYPOINT_PY = REPO_ROOT / "src" / "signet_trainer" / "modal" / "entrypoint.py"
FNS_PY = REPO_ROOT / "src" / "signet_trainer" / "modal" / "fns.py"


# ==================================================================================================
# gap-dryrun-ltx-0 — the mode-conditional refusals fire on the FREE gate
# ==================================================================================================


def test_train_mode_refuses_the_sample_only_items_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """THE burned-approval repro: the shipped multi_frame SAMPLE example dispatched as train.

    Pre-fix the gate returned 0 for this exact pair and the refusal fired in-container,
    post-approval, on a metered A100 (then repeated across the retry policy, detached).
    """
    cfg = load_config(MULTI_FRAME_SAMPLE_YAML)
    assert run_dryrun(cfg, mode="train") != 0
    err = capsys.readouterr().err
    assert "sample-only" in err, "the refusal must carry the WR-04 rationale"


def test_sample_mode_refuses_the_empty_items_config(capsys: pytest.CaptureFixture[str]) -> None:
    """The mirror image: the train-legal overfit example has no items to render from."""
    cfg = load_config(MULTI_FRAME_TRAIN_YAML)
    assert run_dryrun(cfg, mode="sample") != 0
    err = capsys.readouterr().err
    assert "nothing to condition on" in err


def test_each_multi_frame_example_passes_its_own_mode() -> None:
    """Of the two shipped multi_frame examples, exactly one is train-legal, the other sample-legal."""
    assert run_dryrun(load_config(MULTI_FRAME_SAMPLE_YAML), mode="sample") == 0
    assert run_dryrun(load_config(MULTI_FRAME_TRAIN_YAML), mode="train") == 0


def test_bare_dryrun_stays_mode_agnostic() -> None:
    """``mode=None`` keeps the historical behaviour — both examples still pass a bare dryrun."""
    assert run_dryrun(load_config(MULTI_FRAME_SAMPLE_YAML)) == 0
    assert run_dryrun(load_config(MULTI_FRAME_TRAIN_YAML)) == 0


def test_cli_mode_flag_reaches_the_gate() -> None:
    """``signet-dryrun <cfg> --mode train`` models the destination the operator will dispatch."""
    assert main([MULTI_FRAME_SAMPLE_YAML, "--mode", "train"]) == 1
    assert main([MULTI_FRAME_SAMPLE_YAML, "--mode", "sample"]) == 0
    # flag-before-path order works too (no positional coupling)
    assert main(["--mode", "train", MULTI_FRAME_TRAIN_YAML]) == 0


def test_cli_rejects_unknown_or_dangling_mode() -> None:
    assert main([MULTI_FRAME_SAMPLE_YAML, "--mode", "bogus"]) == 2
    assert main([MULTI_FRAME_SAMPLE_YAML, "--mode"]) == 2
    assert main([]) == 2


def test_validate_mode_config_vocabulary_matches_the_entrypoint() -> None:
    """One vocabulary: the gate's KNOWN_MODES mirror the entrypoint's literal --mode tuple."""
    source = ENTRYPOINT_PY.read_text(encoding="utf-8")
    for mode in KNOWN_MODES:
        assert f'"{mode}"' in source
    with pytest.raises(ValueError, match="unknown mode"):
        validate_mode_config(load_config(MULTI_FRAME_TRAIN_YAML), "h3_train")


def test_entrypoint_threads_the_mode_into_the_gate() -> None:
    """The pre-dispatch gate call must carry the mode — a mode-blind call is the audited defect."""
    source = ENTRYPOINT_PY.read_text(encoding="utf-8")
    assert re.search(r"run_dryrun\(\s*cfg\s*,\s*mode=mode\s*\)", source), (
        "entrypoint must call run_dryrun(cfg, mode=mode) so the mode-conditional refusals fire "
        "pre-approval (gap-dryrun-ltx-0)"
    )


def test_container_bodies_share_the_one_home() -> None:
    """train() and sample() call the SHARED validator — no private predicate copies drift apart."""
    source = FNS_PY.read_text(encoding="utf-8")
    assert 'validate_mode_config(config, "train")' in source
    assert 'validate_mode_config(config, "sample")' in source


# ==================================================================================================
# gap-dryrun-ltx-1 — the gate prices the WORST resolution bucket, not training_dims
# ==================================================================================================


def test_gate_prices_the_worst_bucket_not_training_dims(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The audited 9.2x under-pricing: dims [768,352,25] vs bucket 1280x704x81 under ic_lora.

    Pre-fix the banner certified seq_len=1056 while the container's ic_lora path patchified the
    real 1280x704x81 latents into a 19,360-token doubled sequence — the Pitfall-2 OOM class this
    arm exists to prevent.
    """
    cfg = load_config(IC_LORA_YAML)
    cfg = cfg.model_copy(
        update={
            "training_dims": (768, 352, 25),
            "data": cfg.data.model_copy(update={"resolution_buckets": ["1280x704x81"]}),
        },
        deep=True,
    )
    assert run_dryrun(cfg) == 0
    out = capsys.readouterr().out
    # 1280x704x81 -> (704//32)*(1280//32)*((81-1)//8+1) = 22*40*11 = 9680; ic_lora doubles it.
    assert "worst bucket [W=1280, H=704, F=81] -> seq_len=9680" in out
    assert "(1, 19360, 128)" in out, "the ic_lora banner must carry the DOUBLED combined latent"


def test_gate_builds_and_asserts_every_bucket(capsys: pytest.CaptureFixture[str]) -> None:
    """The shipped divergence: ltx23_lora certifies 768x512 while no bucket carries it."""
    cfg = load_config(str(REPO_ROOT / "configs" / "ltx23_lora.example.yaml"))
    assert list(cfg.data.resolution_buckets) == ["768x352x25", "768x352x49", "768x352x81"]
    assert run_dryrun(cfg) == 0
    out = capsys.readouterr().out
    # worst of the defaults: 768x352x81 -> (352//32)*(768//32)*11 = 11*24*11 = 2904
    assert "worst bucket [W=768, H=352, F=81] -> seq_len=2904" in out
    assert "(3 bucket(s) priced)" in out


def test_multi_frame_items_config_prices_buckets_in_the_train_posture() -> None:
    """Items are sample-only and range-checked against training_dims, not the bucket grid.

    The shipped multi_frame example carries frame_index 48 while the default buckets go down to
    F=25 — per-bucket pricing must run the SELF-conditioning posture (what training actually
    does), not index item keyframes past a shorter bucket.
    """
    assert run_dryrun(load_config(MULTI_FRAME_SAMPLE_YAML)) == 0


def test_bucket_pricing_matches_training_dims_when_they_agree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ic_lora example: single bucket == training_dims — the two numbers must coincide."""
    assert run_dryrun(load_config(IC_LORA_YAML)) == 0
    out = capsys.readouterr().out
    assert "seq_len=1560" in out
    assert "worst bucket [W=832, H=480, F=25] -> seq_len=1560" in out
    assert "(1 bucket(s) priced)" in out
