"""Paintover -> binary teal-seed chroma extraction (D-10 rebuild). Spec-driven, config-first.

The r1 extraction script is gone; only its outputs survive (``_inpaint_prep/img_match.json`` +
``masks_frame0/``). This module re-implements the reverse-engineered contract: an external
iPad/Procreate paintover — teal ``#458070`` painted over the region to regenerate on frame 0 —
becomes a binary WHITE(255) seed the SAM propagator (``_sam_propagate_masks.load_seed``, ``>127``)
consumes. App-drawn masks (Plan 06) are binary-by-construction and skip this path.

EVERY chroma constant (the teal hex/rgb, the channel-relationship thresholds, the distance
tolerance) is read from the loaded ``MASK-SPEC.yaml`` dict passed in — NEVER hardcoded here
(D-03/config-first: the numbers below live in the packaged ``MASK-SPEC.yaml``, shipped as package
data under ``signet_trainer.harness_data``). The default
``channel_relationship`` method is the prior project's ``teal_mask`` precedent shape (a BGR channel rule,
robust to iPad-export compression); the ``distance`` alternate is L2-to-teal within tolerance.

Import-confined (Anti-Pattern 6): ``numpy`` is imported function-local so the module PARSES on CI
without a numeric/GPU env. No cv2, no torch, no modal — the caller (the CLI) owns image I/O.
"""

from __future__ import annotations


def _need_numpy():
    """Import numpy lazily with a clear hint (module must PARSE without it)."""
    try:
        import numpy  # noqa: PLC0415  (deliberate function-local import — import-confined)

        return numpy
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "[prep.chroma] missing dependency 'numpy': "
            f"{exc}\n    install hint: pip install numpy"
        ) from exc


def extract_chroma_mask(paintover, spec: dict, match: str | None = None):
    """Extract the hand-painted teal region from a paintover to a binary WHITE(255) seed.

    Args:
        paintover: an HxWx3 **BGR** uint8 array (as ``cv2.imread`` returns). ``None`` / a
            non-HxWx3 array raises ``ValueError`` (T-09.3-03-I: a corrupt decode must fail loud,
            never produce a silent empty mask).
        spec: the loaded ``MASK-SPEC.yaml`` dict. Read: ``spec['rgb']`` (canonical teal, R,G,B),
            ``spec['chroma']['method']`` (``channel_relationship`` | ``distance``),
            ``spec['chroma']['channel_rules']`` (b_min/g_min/g_minus_r_min/b_minus_r_min),
            ``spec['chroma']['tolerance']`` (L2 band for ``distance`` + the residual band).
        match: optional paintover->clip name to stamp on the record (the CLI resolves it).

    Returns:
        ``(mask_uint8, record)`` where ``mask_uint8`` is HxW ``{0, 255}`` (region = WHITE(255),
        D-05 polarity) and ``record`` is the ``img_match.json``-shaped dict
        ``{match, paint_pct, resid, bbox, paint_rgb}`` — an entry is emitted even when no paint is
        found (``paint_rgb``/``bbox`` = ``None``, ``paint_pct`` = 0.0), as the r1 extractor did.
    """
    np = _need_numpy()

    if paintover is None:
        raise ValueError(
            "[prep.chroma] paintover image is None — the decode failed (corrupt/unreadable PNG); "
            "refusing to emit a silent empty mask"
        )
    arr = np.asarray(paintover)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(
            f"[prep.chroma] paintover must be an HxWx3 BGR array, got shape {arr.shape} — "
            "pass a colour image (cv2.imread default), not a grayscale/None decode"
        )

    bgr = arr[:, :, :3].astype(np.int32)
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    chroma = spec["chroma"]
    method = chroma["method"]

    # canonical teal as a BGR pixel, derived from the spec rgb (never a code literal)
    teal_r, teal_g, teal_b = spec["rgb"]
    teal_bgr = np.array([teal_b, teal_g, teal_r], dtype=np.int32)

    if method == "channel_relationship":
        rules = chroma["channel_rules"]
        mask_bool = (
            (b > rules["b_min"])
            & (g > rules["g_min"])
            & ((g - r) > rules["g_minus_r_min"])
            & ((b - r) > rules["b_minus_r_min"])
        )
    elif method == "distance":
        dist_all = np.sqrt(((bgr - teal_bgr) ** 2).sum(axis=2))
        mask_bool = dist_all <= chroma["tolerance"]
    else:
        raise ValueError(
            f"[prep.chroma] unknown chroma.method {method!r} in mask spec — expected "
            "'channel_relationship' or 'distance'"
        )

    # region = WHITE(255) (D-05 seed polarity); belt-and-braces {0, 255}, no intermediate greys
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255

    paint_pct = round(float(mask_bool.mean()) * 100.0, 2)

    # residual quality signal: the anti-aliased teal halo (within the spec L2 band) unioned with the
    # strict match. resid >= paint_pct always — it inflates on face_hair/full_body where soft hair
    # edges leave a partial-teal fringe (matches the r1 resid > paint_pct ordering; the exact r1
    # value is unrecoverable — the script is gone — so this is a documented approximation, not a
    # bit-for-bit reproduction). Distance is measured from the spec-derived teal, never a literal.
    dist = np.sqrt(((bgr - teal_bgr) ** 2).sum(axis=2))
    resid_bool = mask_bool | (dist <= chroma["tolerance"])
    resid = round(float(resid_bool.mean()) * 100.0, 2)

    ys, xs = np.where(mask_bool)
    if ys.size:
        med_bgr = np.median(bgr[ys, xs], axis=0)
        # report back in RGB to mirror img_match.json paint_rgb
        paint_rgb = [int(round(med_bgr[2])), int(round(med_bgr[1])), int(round(med_bgr[0]))]
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    else:
        paint_rgb = None
        bbox = None
        paint_pct = 0.0

    record = {
        "match": match,
        "paint_pct": paint_pct,
        "resid": resid,
        "bbox": bbox,
        "paint_rgb": paint_rgb,
    }
    return mask_uint8, record
