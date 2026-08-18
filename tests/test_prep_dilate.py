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
6. #36 finding 4: the cv2 path (``MORPH_RECT``) and the pure-numpy path grow a mask IDENTICALLY —
   guarded skip only if cv2 is genuinely absent (a prior ``MORPH_ELLIPSE`` kernel was 4-connected
   and grew a measurably SMALLER mask than the numpy path's 8-connected full 3x3 from the same
   seed, while production always took the untested cv2 branch).

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


def test_cv2_and_numpy_dilation_paths_agree_bit_for_bit():
    """#36 finding 4: the shipped cv2 path (``MORPH_RECT``, full 3x3, 8-connected) must grow a mask
    IDENTICALLY to the pure-numpy path it claims to be a "faster equivalent" of.

    Before the fix, the cv2 branch used ``MORPH_ELLIPSE`` (4-connected diamond) — a DIFFERENT,
    SMALLER kernel than the numpy path's full 3x3 — so ``_grow_one``'s stop-at-target loop
    terminated at a different margin per backend (measured on the issue's repro: numpy 256 px vs
    cv2 172 px for an identical seed). Multi-step growth (``max_margin_px`` > 1) is what makes a
    per-step kernel mismatch compound into a visible coverage difference.
    """
    cv2 = pytest.importorskip("cv2")  # noqa: F841  (import used only to trigger the skip)
    m = _center_dot()
    spec = _spec(target=0.99, max_margin_px=6)  # unreachable target -> margin ceiling always binds

    grown_cv2 = dilate_to_target(m, "face_hair", spec, use_cv2=True)
    grown_np = dilate_to_target(m, "face_hair", spec, use_cv2=False)

    assert np.array_equal(np.asarray(grown_cv2, dtype=bool), np.asarray(grown_np, dtype=bool))
    # both actually grew past the seed — an accidental early-return can't pass this trivially
    assert coverage_fraction(grown_cv2) > coverage_fraction(m)


def test_cv2_dilation_uses_morph_rect_not_morph_ellipse():
    """Pins the exact kernel choice (#36 finding 4) — a regression back to ``MORPH_ELLIPSE`` would
    still coincidentally pass a coverage-only equivalence check on some shapes, so assert the
    kernel itself is the numpy-equivalent full 3x3 all-ones.
    """
    cv2 = pytest.importorskip("cv2")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    assert np.array_equal(kernel, np.ones((3, 3), dtype=kernel.dtype))

    from signet_trainer.prep.dilate import _dilate_1px  # noqa: PLC0415

    seed = np.zeros((9, 9), dtype=bool)
    seed[4, 4] = True
    grown_cv2 = _dilate_1px(seed, use_cv2=True)
    grown_np = _dilate_1px(seed, use_cv2=False)
    assert np.array_equal(np.asarray(grown_cv2, dtype=bool), np.asarray(grown_np, dtype=bool))
    assert int(np.asarray(grown_cv2, dtype=bool).sum()) == 9  # full 3x3 neighborhood, not a 5-px diamond
