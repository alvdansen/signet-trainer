"""prep.textseed unit coverage. Pure CPU — NO modal, NO GPU, NO network, NO model load.

Import-confined by contract (mirrors tests/test_prep_propagate.py): the module must PARSE and every
leg below must run on a box with no torch / transformers / cv2 installed, because the heavy deps are
function-local. Asserts, unconditionally:

1. ``import signet_trainer.prep.textseed`` succeeds without any heavy dep loaded (subprocess-checked).
2. The probe-index plan ALWAYS contains both endpoints (the only two frames the propagation seam can
   seed) and is evenly spaced in between.
3. The fwd/rev direction decision: frame-0 preference, the ``start_margin`` flip to ``rev``, the
   end-only case (the clip that opens without the subject — the operator's directive), the flagged weak
   fallback, and the honest NOT-SEEDABLE result (never a silent empty seed).
4. Instance selection respects the coverage band and picks the top score.
5. The config loader fails LOUD on a half-written config.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from signet_trainer import harness_data

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"


def _cfg(**seed_over) -> dict:
    seed = {
        "prompt": None,
        "n_probes": 5,
        "min_score": 0.30,
        "weak_score": 0.15,
        "min_coverage": 0.002,
        "max_coverage": 0.90,
        "start_margin": 0.10,
        "mask_type": "full_body",
    }
    seed.update(seed_over)
    return {
        "version": 1,
        "model": {"id": "facebook/sam3", "dtype": "bfloat16", "device": "cuda"},
        "detect": {"threshold": 0.4, "mask_threshold": 0.5},
        "seed": seed,
    }


def _os_environ() -> dict:
    import os  # noqa: PLC0415

    return dict(os.environ)


# --------------------------------------------------------------------------------------------------
# 1. import-confinement
# --------------------------------------------------------------------------------------------------

def test_module_exports_present():
    import signet_trainer.prep.textseed as ts  # noqa: PLC0415

    for name in ("seed_from_text", "decide_seed", "probe_indices", "select_instance",
                 "load_textseed_config", "rev_stems", "summarize", "run"):
        assert hasattr(ts, name), name


def test_import_pulls_no_heavy_backend():
    """A FRESH interpreter importing textseed must not drag in torch/transformers/cv2."""
    code = (
        "import sys; import signet_trainer.prep.textseed as t; "
        "assert hasattr(t, 'seed_from_text'); "
        "bad = [m for m in ('torch', 'transformers', 'cv2') if m in sys.modules]; "
        "print('LOADED:' + ','.join(bad)); "
        "sys.exit(1 if bad else 0)"
    )
    env = {**_os_environ(), "PYTHONPATH": str(_SRC), "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"textseed import pulled heavy deps: {r.stdout}{r.stderr}"


# --------------------------------------------------------------------------------------------------
# 2. probe plan
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(54, 5), (54, 2), (10, 4), (2, 5), (121, 3)])
def test_probe_indices_always_include_both_endpoints(n, k):
    from signet_trainer.prep.textseed import probe_indices  # noqa: PLC0415

    idx = probe_indices(n, k)
    assert idx[0] == 0
    assert idx[-1] == n - 1, "the last frame is the only rev-seedable frame — it must always be probed"
    assert idx == sorted(set(idx))
    assert all(0 <= i < n for i in idx)


def test_probe_indices_evenly_spaced():
    from signet_trainer.prep.textseed import probe_indices  # noqa: PLC0415

    assert probe_indices(54, 5) == [0, 13, 26, 40, 53]  # banker's rounding at the 26.5 midpoint
    assert probe_indices(1, 5) == [0]
    assert probe_indices(0, 5) == []


# --------------------------------------------------------------------------------------------------
# 3. direction decision (D-08 fwd/rev)
# --------------------------------------------------------------------------------------------------

def test_frame0_detection_wins_forward():
    from signet_trainer.prep.textseed import FWD, REASON_START, decide_seed  # noqa: PLC0415

    probes = {0: {"score": 0.84, "coverage": 0.29}, 53: {"score": 0.90, "coverage": 0.30}}
    d = decide_seed(probes, 54, _cfg())
    assert d["found"] and d["direction"] == FWD
    assert d["seed_frame_idx"] == 0 and d["reason"] == REASON_START


def test_last_frame_flips_to_rev_only_beyond_the_margin():
    from signet_trainer.prep.textseed import FWD, REASON_END_BEATS_START, REV, decide_seed  # noqa: PLC0415

    cfg = _cfg(start_margin=0.10)
    # +0.10 exactly -> NOT beyond the margin -> stays forward
    d = decide_seed({0: {"score": 0.50, "coverage": 0.1}, 9: {"score": 0.60, "coverage": 0.1}}, 10, cfg)
    assert d["direction"] == FWD
    # +0.11 -> beyond the margin -> rev, seeded on the LAST frame
    d = decide_seed({0: {"score": 0.50, "coverage": 0.1}, 9: {"score": 0.61, "coverage": 0.1}}, 10, cfg)
    assert d["direction"] == REV
    assert d["seed_frame_idx"] == 9 and d["reason"] == REASON_END_BEATS_START


def test_clip_opening_without_subject_seeds_backward():
    """The operator's directive: a clip that opens on an empty room seeds from the LAST frame (rev)."""
    from signet_trainer.prep.textseed import REASON_END, REV, decide_seed  # noqa: PLC0415

    probes = {0: None, 13: None, 27: {"score": 0.55, "coverage": 0.05}, 53: {"score": 0.965, "coverage": 0.018}}
    d = decide_seed(probes, 54, _cfg())
    assert d["direction"] == REV and d["seed_frame_idx"] == 53
    assert d["reason"] == REASON_END and d["weak"] is False


def test_weak_endpoint_fallback_is_flagged_not_silent():
    from signet_trainer.prep.textseed import REASON_WEAK, REV, decide_seed  # noqa: PLC0415

    probes = {0: {"score": 0.16, "coverage": 0.01}, 53: {"score": 0.22, "coverage": 0.01}}
    d = decide_seed(probes, 54, _cfg())
    assert d["found"] and d["weak"] is True
    assert d["direction"] == REV and d["reason"] == REASON_WEAK


def test_subject_never_found_is_reported_not_silently_empty():
    from signet_trainer.prep.textseed import REASON_NOT_FOUND, decide_seed  # noqa: PLC0415

    d = decide_seed({0: None, 27: None, 53: None}, 54, _cfg())
    assert d["found"] is False
    assert d["direction"] is None and d["seed_frame_idx"] is None
    assert d["reason"] == REASON_NOT_FOUND


def test_below_min_score_at_both_ends_and_below_weak_is_not_found():
    from signet_trainer.prep.textseed import decide_seed  # noqa: PLC0415

    d = decide_seed({0: {"score": 0.05, "coverage": 0.01}, 9: {"score": 0.10, "coverage": 0.01}}, 10, _cfg())
    assert d["found"] is False


# --------------------------------------------------------------------------------------------------
# 4. helpers feeding the propagation seam
# --------------------------------------------------------------------------------------------------

def test_rev_stems_feed_propagate_rev_set():
    from signet_trainer.prep.textseed import FWD, REV, rev_stems, summarize  # noqa: PLC0415

    records = [
        {"clip_stem": "c1", "mask_stem": "c1__full_body", "found": True, "direction": FWD},
        {"clip_stem": "c2", "mask_stem": "c2__full_body", "found": True, "direction": REV},
        {"clip_stem": "c3", "mask_stem": "c3__full_body", "found": True, "direction": REV, "weak": True},
        {"clip_stem": "c4", "mask_stem": "c4__full_body", "found": False, "direction": None},
    ]
    assert rev_stems(records) == ["c2__full_body", "c3__full_body"]
    s = summarize(records)
    assert s == {"clips": 4, "seeded": 3, "fwd": 1, "rev": 2, "weak": 1, "not_found": ["c4"]}


def test_mask_stem_matches_propagate_split_contract():
    from signet_trainer.prep.resolve import split_mask_stem  # noqa: PLC0415
    from signet_trainer.prep.textseed import mask_stem  # noqa: PLC0415

    stem = mask_stem("segtest_a_01", "full_body")
    assert stem == "segtest_a_01__full_body"
    assert split_mask_stem(stem) == ("segtest_a_01", "full_body")


def test_select_instance_respects_coverage_band_and_picks_top_score():
    from signet_trainer.prep.textseed import select_instance  # noqa: PLC0415

    cfg = _cfg(min_coverage=0.01, max_coverage=0.50)
    # a speck (below band) scores highest but must lose; the runaway full-frame mask is rejected too
    scores = [0.99, 0.80, 0.95]
    covs = [0.0001, 0.20, 0.95]
    assert select_instance(scores, covs, cfg) == 1
    assert select_instance([0.9], [0.999], cfg) is None
    assert select_instance([], [], cfg) is None


# --------------------------------------------------------------------------------------------------
# 5. config-first: fail LOUD
# --------------------------------------------------------------------------------------------------

def test_shipped_config_loads_and_carries_every_threshold():
    from signet_trainer.prep.textseed import DEFAULT_CONFIG, load_textseed_config  # noqa: PLC0415

    cfg = load_textseed_config(DEFAULT_CONFIG)
    assert cfg["model"]["id"] == "facebook/sam3"
    for key in ("min_score", "weak_score", "min_coverage", "max_coverage", "start_margin", "n_probes"):
        assert key in cfg["seed"], key


def test_missing_config_raises(tmp_path):
    from signet_trainer.prep.textseed import load_textseed_config  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        load_textseed_config(tmp_path / "nope.yaml")


def test_half_written_config_raises(tmp_path):
    from signet_trainer.prep.textseed import load_textseed_config  # noqa: PLC0415

    p = tmp_path / "half.yaml"
    p.write_text("version: 1\nmodel: {id: facebook/sam3}\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_textseed_config(p)
    assert "missing required blocks" in str(exc.value)

    p.write_text(
        "version: 1\nmodel: {id: x}\ndetect: {threshold: 0.4, mask_threshold: 0.5}\nseed: {n_probes: 5}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_textseed_config(p)
    assert "seed block missing" in str(exc.value)


def test_segmentation_mask_spec_disables_dilation():
    """The segmentation export must keep EXACT SAM boundaries — dilation off via config, not code."""
    from signet_trainer.prep.spec import load_mask_spec  # noqa: PLC0415

    spec = load_mask_spec(harness_data.spec_path(harness_data.MASK_SPEC_SEGMENTATION))
    for cls, band in spec["coverage_bands"].items():
        assert band["dilation"]["max_margin_px"] == 0, cls
        assert band["dilation"]["grow_to_target"] is False, cls


def test_segmentation_spec_dilation_is_a_noop():
    """dilate_to_target with the segmentation spec returns the mask byte-identical (no growth)."""
    np = pytest.importorskip("numpy")
    from signet_trainer.prep.dilate import dilate_to_target  # noqa: PLC0415
    from signet_trainer.prep.spec import load_mask_spec  # noqa: PLC0415

    spec = load_mask_spec(harness_data.spec_path(harness_data.MASK_SPEC_SEGMENTATION))
    m = np.zeros((16, 16), dtype=bool)
    m[4:8, 4:8] = True
    out = dilate_to_target({0: m}, "full_body", spec)
    assert np.array_equal(out[0], m)
