"""prep.dilate D-02 auto-extend unit coverage. Pure CPU — NO modal, NO GPU, NO network.

D-02 is LOCKED: exact-SAM masks (~4.8% coverage) HURT likeness; DILATED masks (~14%) won. This
module's job is to GROW the masks toward the per-class coverage target read FROM the spec. These
tests lock, unconditionally (pure-numpy growth path — cv2 not required):

1. ``coverage_fraction`` is exact on a known mask.
2. ``dilate_to_target`` GROWS a small mask toward ``target`` (coverage post >= pre) and stops
   AT/NEAR the target when ``grow_to_target`` (does not blow far past it).
3. Growth respects ``max_margin_px`` as a hard ceiling (a tiny margin caps the growth).
4. A mask already >= ``target`` is returned effectively unchanged (no over-growth).
5. The target/margin come from the passed spec dict (config-first, D-03) — a class absent from the
   spec's coverage_bands raises loudly.

House test rules honored: no metered dispatch, no modal, no network — synthetic in-memory fixtures.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from signet_trainer.prep.dilate import coverage_fraction, dilate_to_target  # noqa: E402


def _spec(target: float, max_margin_px: int, grow_to_target: bool = True) -> dict:
    """A minimal MASK-SPEC-shaped dict carrying one coverage band with a D-02 dilation sub-block."""
    return {
        "coverage_bands": {
            "face_hair": {
                "target": target,
                "warn_below": target / 2,
                "dilation": {"max_margin_px": max_margin_px, "grow_to_target": grow_to_target},
            }
        }
    }


def _center_dot(hw=(64, 64), r=2):
    """A tiny filled square in the middle of an HxW frame (low coverage, room to grow)."""
    h, w = hw
    m = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    m[cy - r : cy + r, cx - r : cx + r] = True
    return m


def test_coverage_fraction_exact():
    m = np.zeros((10, 10), dtype=bool)
    m[:2, :] = True  # 20 / 100
    assert coverage_fraction(m) == pytest.approx(0.20)
    assert coverage_fraction(np.zeros((4, 4), dtype=bool)) == 0.0
    assert coverage_fraction(np.ones((4, 4), dtype=bool)) == 1.0


def test_dilate_grows_toward_target():
    """Pure-numpy path (use_cv2=False): a tiny mask grows until coverage reaches the target band."""
    m = _center_dot()
    pre = coverage_fraction(m)
    target = 0.10
    grown = dilate_to_target(m, "face_hair", _spec(target, max_margin_px=64), use_cv2=False)
    post = coverage_fraction(grown)
    assert post >= pre                      # it GREW
    assert post >= target                   # reached the per-class target
    assert post < target + 0.10             # stopped near target — did not blow far past it


def test_dilate_respects_max_margin_ceiling():
    """A tiny max_margin caps growth even when the target is not yet reached."""
    m = _center_dot()
    pre = coverage_fraction(m)
    # target unreachable within 1 px of growth -> the margin ceiling binds
    grown_capped = dilate_to_target(m, "face_hair", _spec(0.99, max_margin_px=1), use_cv2=False)
    grown_more = dilate_to_target(m, "face_hair", _spec(0.99, max_margin_px=8), use_cv2=False)
    cap = coverage_fraction(grown_capped)
    more = coverage_fraction(grown_more)
    assert cap > pre            # grew a little
    assert more > cap           # a larger ceiling grew more -> the ceiling is the binding limit
    assert cap < 0.99           # never reached the impossible target


def test_mask_already_at_target_not_overgrown():
    """A mask already >= target is returned effectively unchanged (grow_to_target stops immediately)."""
    m = np.zeros((20, 20), dtype=bool)
    m[:, :] = True  # coverage 1.0, well above target
    grown = dilate_to_target(m, "face_hair", _spec(0.14, max_margin_px=28), use_cv2=False)
    assert coverage_fraction(grown) == pytest.approx(1.0)
    assert np.array_equal(np.asarray(grown, dtype=bool), m)


def test_dict_stack_grows_every_member():
    """propagate.py passes a {idx: bool HxW} dict — every member is grown, keys preserved."""
    stack = {0: _center_dot(), 5: _center_dot()}
    out = dilate_to_target(stack, "face_hair", _spec(0.10, max_margin_px=64), use_cv2=False)
    assert set(out.keys()) == {0, 5}
    for k in out:
        assert coverage_fraction(out[k]) >= 0.10


def test_unknown_class_raises_from_spec():
    """Config-first: a class not in the spec's coverage_bands fails loud (never a hardcoded default)."""
    with pytest.raises(ValueError, match="coverage_bands"):
        dilate_to_target(_center_dot(), "not_a_class", _spec(0.10, max_margin_px=8), use_cv2=False)


def test_target_is_spec_sourced_not_hardcoded():
    """Different spec targets produce different growth — proving the target comes from the spec dict."""
    m = _center_dot()
    low = dilate_to_target(m, "face_hair", _spec(0.05, max_margin_px=64), use_cv2=False)
    high = dilate_to_target(m, "face_hair", _spec(0.30, max_margin_px=64), use_cv2=False)
    assert coverage_fraction(high) > coverage_fraction(low)
