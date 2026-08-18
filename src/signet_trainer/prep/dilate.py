"""D-02 auto-extend — grow exact-SAM masks toward the per-class coverage target (spec-driven).

D-02 is a LOCKED decision (operator chose "Auto-extend + QA gate" over "Guidance only"): in r1 the
raw exact-SAM masks (~4.8% coverage) HURT likeness on the hair-heavy classes while the DILATED
masks (~14%) won. So the inpaint pipeline must actually GROW the masks, not merely measure them.

``dilate_to_target(masks, subject_class, spec)`` reads ``spec['coverage_bands'][subject_class]``
for its ``target`` coverage fraction and its ``dilation`` sub-block (``max_margin_px`` +
``grow_to_target``) and grows each exact-SAM bool mask outward — 1 px at a time — until the mask's
coverage fraction reaches ``target`` (when ``grow_to_target``) OR the accumulated growth reaches
``max_margin_px``, whichever comes FIRST. Every knob (target + margin) comes ONLY from the passed
spec dict — NEVER hardcoded here (D-03 / config-first: the numbers live in
the packaged ``MASK-SPEC.yaml``, shipped as package data under ``signet_trainer.harness_data``). A
mask already at/above ``target`` is returned unchanged (no
over-growth past the ceiling).

Import-confined (Anti-Pattern 6): ``numpy`` and ``cv2`` are imported function-local so the module
PARSES on CI without a numeric/GPU env. The coverage/target-stop math has a PURE-NUMPY dilation
path (``_dilate_once_numpy``) so the growth/target/ceiling logic is fully testable WITHOUT cv2;
cv2's structuring-element dilation (``MORPH_RECT``, full 3x3 all-ones — matching the numpy path's
8-connectivity bit-for-bit) is used as a faster equivalent when cv2 is present (#36 finding 4: a
prior ``MORPH_ELLIPSE`` kernel was 4-connected and produced a DIFFERENT grown mask than the numpy
path from the same seed — production always took that untested cv2 branch while every test in
``tests/test_prep_dilate.py`` pinned ``use_cv2=False``).
"""

from __future__ import annotations


def _need_numpy():
    """Import numpy lazily with a clear hint (module must PARSE without it)."""
    try:
        import numpy  # noqa: PLC0415  (deliberate function-local import — import-confined)

        return numpy
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "[prep.dilate] missing dependency 'numpy': "
            f"{exc}\n    install hint: pip install numpy"
        ) from exc


def _try_cv2():
    """Return the cv2 module if importable, else None (the pure-numpy path is used instead)."""
    try:
        import cv2  # noqa: PLC0415  (deliberate function-local import — import-confined)

        return cv2
    except ImportError:  # pragma: no cover - env-dependent
        return None


def coverage_fraction(mask) -> float:
    """Fraction of True pixels in a boolean mask (0.0 for an empty array)."""
    np = _need_numpy()
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(arr.sum()) / float(arr.size)


def _dilate_once_numpy(mask):
    """Pure-numpy 8-connectivity 1 px binary dilation (no cv2 — keeps the growth math testable)."""
    np = _need_numpy()
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"dilation expects a 2D mask, got ndim={m.ndim}")
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    out = np.zeros_like(m)
    h, w = m.shape
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= padded[dy : dy + h, dx : dx + w]
    return out


def _dilate_1px(mask, use_cv2: bool):
    """Grow a bool mask by ~1 px. Uses cv2's 3x3 full-square kernel when available; pure-numpy otherwise.

    #36 finding 4: this MUST be ``MORPH_RECT`` (all-ones 3x3, 8-connected), not ``MORPH_ELLIPSE``
    ([[0,1,0],[1,1,1],[0,1,0]], 4-connected) — the ellipse kernel grows a DIFFERENT (smaller) region
    per step than :func:`_dilate_once_numpy`'s full 3x3, so ``max_margin_px`` silently meant a
    different radius depending on whether cv2 was installed. ``MORPH_RECT`` (3, 3) is bit-identical
    to the numpy path.
    """
    if use_cv2:
        cv2 = _try_cv2()
        if cv2 is not None:
            np = _need_numpy()
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            grown = cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel, iterations=1)
            return grown.astype(bool)
    return _dilate_once_numpy(mask)


def _grow_one(mask, target: float, max_margin: int, grow_to_target: bool, use_cv2: bool):
    """Grow ONE bool mask toward ``target`` coverage, capped at ``max_margin`` px of growth."""
    np = _need_numpy()
    cur = np.asarray(mask, dtype=bool)
    if max_margin <= 0:
        return cur
    margin = 0
    while margin < max_margin:
        if grow_to_target and coverage_fraction(cur) >= target:
            break  # reached the per-class target — do not over-grow past it
        cur = _dilate_1px(cur, use_cv2)
        margin += 1
    return cur


def dilate_to_target(masks, subject_class: str, spec: dict, *, use_cv2: bool | None = None):
    """D-02 auto-extend: grow exact-SAM masks toward the per-class coverage target from the spec.

    Args:
        masks: one bool mask (2D array), a 3D stack ``(T, H, W)``, a list/tuple of 2D masks, or a
            ``{idx: 2D bool}`` dict (the shape :mod:`prep.propagate` produces). The same container
            type is returned, every member grown.
        subject_class: the mask type ``{face, face_hair, full_body}`` (from ``split_mask_stem``);
            selects ``spec['coverage_bands'][subject_class]``.
        spec: a loaded ``MASK-SPEC.yaml`` dict. ``target`` and the ``dilation`` sub-block
            (``max_margin_px`` + ``grow_to_target``) are read FROM here — never hardcoded (D-03).
        use_cv2: force cv2 (True) or the pure-numpy path (False); ``None`` auto-detects cv2.

    Growth stops at the FIRST of: coverage reaches ``target`` (when ``grow_to_target``) OR the
    accumulated growth reaches ``max_margin_px``. A mask already >= ``target`` is returned unchanged.
    """
    bands = spec.get("coverage_bands") if isinstance(spec, dict) else None
    if not isinstance(bands, dict) or subject_class not in bands:
        available = sorted(bands) if isinstance(bands, dict) else bands
        raise ValueError(
            f"subject class {subject_class!r} not in spec coverage_bands ({available}); the "
            f"per-class dilation target/margin is spec-sourced (D-03), never hardcoded in code"
        )
    band = bands[subject_class]
    target = float(band["target"])
    dcfg = band.get("dilation") or {}
    max_margin = int(dcfg.get("max_margin_px", 0))
    grow_to_target = bool(dcfg.get("grow_to_target", True))
    if use_cv2 is None:
        use_cv2 = _try_cv2() is not None

    def _one(m):
        return _grow_one(m, target, max_margin, grow_to_target, use_cv2)

    np = _need_numpy()
    if isinstance(masks, dict):
        return {k: _one(v) for k, v in masks.items()}
    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        return np.stack([_one(masks[i]) for i in range(masks.shape[0])])
    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        return _one(masks)
    if isinstance(masks, (list, tuple)):
        grown = [_one(m) for m in masks]
        return tuple(grown) if isinstance(masks, tuple) else grown
    return _one(masks)  # single 2D array-like fallback
