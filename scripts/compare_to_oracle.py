"""D-02 oracle diff — compare a fresh-encode .precomputed tree against enochiatron-encoded.

Validates the eventual Plan-03 encode output (``.precomputed/{latents,conditions}/<rel>.pt``)
against enochiatron's shipped correctness oracle (the private ``enochiatron-encoded`` HF
dataset, same flat layout). For each ``.pt`` paired by relative path:

    * assert shape + dtype equality (hard failure -> non-zero exit);
    * report value statistics (mean/std/min/max) and their diffs — a NUMERICAL diff where
      the same clip+bucket overlaps enochiatron's set, a shape/dtype/stats sanity check
      elsewhere (D-02 caveat).

It also explicitly RECORDS and prints the observed ``video_prompt_embeds`` last dim
(expected 4096 per enochiatron-encoded) so Phase 3's training config uses the right text
dim (RESEARCH.md Pitfall 7 carry-forward).

This script is RUN MANUALLY after the gated encode lands (Plan 03) — it is NOT part of the
automated test suite. It mirrors src/signet_trainer/dryrun/shapes.py's
``main(argv) -> int`` + ``sys.exit(main())`` CLI pattern (try/except -> stderr -> non-zero).

Imports: ``torch`` only (heavy). NO modal / ltx_core / ltx_trainer. ``torch.load`` uses
``weights_only=True`` so an untrusted pickle in a ``.pt`` cannot execute code (threat
T-02-MD2). EXPECTED_* constants come from signet_trainer.models.loader (single source).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

# EXPECTED_HIDDEN_DIM is the Gemma/text hidden dim (4096) — the value Pitfall 7 wants recorded.
try:
    from signet_trainer.models.loader import EXPECTED_HIDDEN_DIM
except Exception:  # noqa: BLE001 — keep the script runnable standalone (e.g. without PYTHONPATH=src)
    EXPECTED_HIDDEN_DIM = 4096

LATENTS_SUBDIR = "latents"
CONDITIONS_SUBDIR = "conditions"
TEXT_EMBED_KEY = "video_prompt_embeds"


def _load_pt(path: Path) -> Any:
    """Safe-load a .pt (weights_only=True so an untrusted pickle can't execute — T-02-MD2)."""
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _iter_tensors(obj: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Flatten a loaded .pt (tensor | dict[str, tensor] | nested) into (name, tensor) pairs."""
    out: list[tuple[str, torch.Tensor]] = []
    if isinstance(obj, torch.Tensor):
        out.append((prefix or "<tensor>", obj))
    elif isinstance(obj, dict):
        for key, val in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_iter_tensors(val, child))
    return out


def _stats(t: torch.Tensor) -> dict[str, float]:
    tf = t.detach().to(torch.float32)
    return {
        "mean": float(tf.mean()),
        "std": float(tf.std()),
        "min": float(tf.min()),
        "max": float(tf.max()),
    }


def _pairs(root: Path) -> dict[str, Path]:
    """Map relative-path key -> .pt file under latents/ and conditions/ of a .precomputed root."""
    found: dict[str, Path] = {}
    for sub in (LATENTS_SUBDIR, CONDITIONS_SUBDIR):
        base = root / sub
        if not base.is_dir():
            continue
        for pt in sorted(base.rglob("*.pt")):
            rel = pt.relative_to(root).as_posix()  # e.g. latents/muxed/clip.pt
            found[rel] = pt
    return found


def _compare_one(rel: str, fresh_pt: Path, oracle_pt: Path) -> tuple[int, int]:
    """Compare one paired .pt. Returns (mismatches, prompt_embed_dim_or_-1)."""
    fresh = _load_pt(fresh_pt)
    oracle = _load_pt(oracle_pt)

    fresh_t = dict(_iter_tensors(fresh))
    oracle_t = dict(_iter_tensors(oracle))

    mismatches = 0
    embed_dim = -1

    print(f"\n[{rel}]")
    common = sorted(set(fresh_t) & set(oracle_t))
    only_fresh = sorted(set(fresh_t) - set(oracle_t))
    only_oracle = sorted(set(oracle_t) - set(fresh_t))
    if only_fresh:
        print(f"  WARN keys only in fresh:  {only_fresh}")
    if only_oracle:
        print(f"  WARN keys only in oracle: {only_oracle}")

    for name in common:
        ft, ot = fresh_t[name], oracle_t[name]
        if name.endswith(TEXT_EMBED_KEY) or name == TEXT_EMBED_KEY:
            embed_dim = int(ft.shape[-1]) if ft.ndim >= 1 else -1
        if tuple(ft.shape) != tuple(ot.shape):
            print(f"  MISMATCH {name}: shape {tuple(ft.shape)} != oracle {tuple(ot.shape)}")
            mismatches += 1
            continue
        if ft.dtype != ot.dtype:
            print(f"  MISMATCH {name}: dtype {ft.dtype} != oracle {ot.dtype}")
            mismatches += 1
            continue
        fs, os_ = _stats(ft), _stats(ot)
        d_mean = fs["mean"] - os_["mean"]
        d_std = fs["std"] - os_["std"]
        print(
            f"  OK {name}: shape {tuple(ft.shape)} dtype {ft.dtype} | "
            f"mean {fs['mean']:.4g} (d{d_mean:+.3g}) std {fs['std']:.4g} (d{d_std:+.3g}) "
            f"min {fs['min']:.4g} max {fs['max']:.4g}"
        )

    return mismatches, embed_dim


def main(argv: list[str] | None = None) -> int:
    """Compare a fresh-encode .precomputed root against the enochiatron-encoded oracle.

    Returns 0 when every paired .pt matches by shape + dtype, non-zero on any mismatch
    (or on a usage/IO error). Stats diffs are reported but do NOT fail the run (D-02: a
    numerical diff is only meaningful where clips/buckets overlap).
    """
    parser = argparse.ArgumentParser(
        prog="compare_to_oracle",
        description="D-02 oracle diff: fresh-encode .precomputed vs enochiatron-encoded.",
    )
    parser.add_argument("fresh_dir", help="fresh-encode .precomputed/ root (latents/ + conditions/)")
    parser.add_argument("oracle_dir", help="enochiatron-encoded .precomputed/ root")

    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

        fresh_root = Path(args.fresh_dir)
        oracle_root = Path(args.oracle_dir)
        if not fresh_root.is_dir():
            print(f"[compare_to_oracle] fresh_dir not a directory: {fresh_root}", file=sys.stderr)
            return 1
        if not oracle_root.is_dir():
            print(f"[compare_to_oracle] oracle_dir not a directory: {oracle_root}", file=sys.stderr)
            return 1

        fresh = _pairs(fresh_root)
        oracle = _pairs(oracle_root)
        overlap = sorted(set(fresh) & set(oracle))
        if not overlap:
            print(
                "[compare_to_oracle] no overlapping .pt files between fresh and oracle "
                "(nothing to numerically diff; check directory layout)",
                file=sys.stderr,
            )
            return 1

        total_mismatches = 0
        observed_embed_dim = -1
        for rel in overlap:
            mism, embed_dim = _compare_one(rel, fresh[rel], oracle[rel])
            total_mismatches += mism
            if embed_dim > 0:
                observed_embed_dim = embed_dim

        print("\n" + "=" * 60)
        print(f"[compare_to_oracle] compared {len(overlap)} paired .pt file(s)")
        if observed_embed_dim > 0:
            flag = "OK" if observed_embed_dim == EXPECTED_HIDDEN_DIM else "WARN"
            print(
                f"[compare_to_oracle] {flag} {TEXT_EMBED_KEY} last-dim = {observed_embed_dim} "
                f"(EXPECTED_HIDDEN_DIM={EXPECTED_HIDDEN_DIM}) -- Phase-3 text-dim carry-forward "
                f"(Pitfall 7)"
            )
        else:
            print(
                f"[compare_to_oracle] WARN {TEXT_EMBED_KEY} not found -- could not record "
                f"text last-dim (Pitfall 7)"
            )

        if total_mismatches:
            print(f"[compare_to_oracle] FAIL -- {total_mismatches} shape/dtype mismatch(es)")
            return 1
        print("[compare_to_oracle] PASS -- all paired tensors match shape + dtype")
        return 0

    except SystemExit:
        # argparse exits non-zero on bad/missing args; surface that as our non-zero return.
        return 2
    except Exception as exc:  # noqa: BLE001 — any failure becomes a clean non-zero exit.
        print(f"[compare_to_oracle] FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
