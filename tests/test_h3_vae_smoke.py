"""The BOTH-MODALITIES smoke — the family closer for "a path only the reference case takes".

Why a smoke and not another contract
------------------------------------
``AutoencoderKLMiniMaxH3._encode`` short-circuits ``num_frames == 1`` straight to ``_encode_clip``.
That branch is **structurally unreachable from any video-side success**, and two defects have now
been bought inside it, each at the top of an 88-sample encode, each after PHASE A had already pushed
the whole corpus through Qwen3-VL-32B:

* **D-10-DEF-9** — 4-D pixels. Caught by a contract (the named rank refusal), because it is a shape.
* **D-10-DEF-12** — CPU pixels into a CUDA VAE. **No contract could catch it**: shapes, values, keys,
  counts, ranks and grad mode were all correct. It is a CUDA *dispatch* decision
  (``cudnn_is_acceptable`` -> ``ConvBackend.Slow3d`` -> no CUDA kernel), and a CPU box has no such
  dispatcher to disagree with. Five containers died there.

So the closer is not another diff — it is **executing the branch, in-container, on one sample,
before the loop**. What the tests below pin is that it (a) drives the REAL production entry points,
(b) covers BOTH modalities, and (c) **runs before PHASE A**, which is the half that turns the whole
family into cents rather than into a slightly earlier container death.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from signet_trainer.models.h3_loader import EXPECTED_H3_IN_CHANNELS
from signet_trainer.prep.h3_encode import h3_vae_smoke_encode
from signet_trainer.prep.h3_vae_contract import H3VideoVaeContractStub

FNS_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal" / "fns.py"
).read_text(encoding="utf-8")

_LATENT_STATS = (torch.zeros(EXPECTED_H3_IN_CHANNELS), torch.ones(EXPECTED_H3_IN_CHANNELS))
_DESCRIPTOR = {"path": "refs/char_a.png", "kind": "character", "subject_id": "A"}


class _StubImage:
    """The three things ``_reference_pixels`` uses: ``.size``, ``.convert``, ``.resize``."""

    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)

    def convert(self, _mode: str) -> _StubImage:
        return self

    def resize(self, size: tuple[int, int], _resample: object = None) -> _StubImage:
        return _StubImage(*size)

    def __array__(self, dtype: object = None, copy: object = None) -> object:  # noqa: ARG002
        width, height = self.size
        array = np.full((height, width, 3), 128, dtype=np.uint8)
        return array if dtype is None else array.astype(dtype)


def _h3_preprocess_node() -> ast.FunctionDef:
    for node in ast.walk(ast.parse(FNS_SOURCE)):
        if isinstance(node, ast.FunctionDef) and node.name == "h3_preprocess":
            return node
    raise AssertionError("h3_preprocess not found in modal/fns.py")


def _body_text() -> str:
    return ast.unparse(_h3_preprocess_node())


def _smoke(vae: H3VideoVaeContractStub, *, frames: int = 22) -> str:
    return h3_vae_smoke_encode(
        vae,
        clip_pixels=torch.zeros(3, frames, 32, 48, dtype=torch.uint8),
        reference_image=_StubImage(1024, 1536),
        reference_short_edge=896,
        reference_descriptor=_DESCRIPTOR,
        latents_mean=_LATENT_STATS[0],
        latents_std=_LATENT_STATS[1],
        clip_pixel_frames=frames,
    )


# --------------------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------------------


def test_both_modalities_are_encoded_and_the_report_names_both() -> None:
    """A passing smoke must leave EVIDENCE in the container log, not just an absence of failure."""
    report = _smoke(H3VideoVaeContractStub())
    assert "clip 22f" in report
    assert "reference" in report
    assert "D-10-DEF-9" in report and "D-10-DEF-12" in report, (
        "the report names the two defects it exists to catch, so a reader who sees it in a "
        "container log knows what was proven and what was not."
    )


def test_the_smoke_survives_the_D_10_DEF_12_shape_end_to_end() -> None:
    """CPU-built pixels, CPU-opened image, a component that lives somewhere else.

    This is the reference path exactly as production runs it: ``_reference_pixels`` makes its tensor
    with ``torch.from_numpy`` (CPU) and the VAE is elsewhere. On a GPU-less box ``meta`` stands in
    for "elsewhere", which is what makes the whole family testable at all here.
    """
    assert _smoke(H3VideoVaeContractStub(device="meta"))


def test_the_frame_law_is_cross_checked_by_the_real_entry_point() -> None:
    """22 -> 7 comes from ``encode_h3_video_latents``'s own guard, not from a re-derivation here.

    The smoke owns no arithmetic of its own; it inherits every check the loop's entry points carry,
    which is the point of driving them rather than reimplementing them. A non-conforming count is
    refused by ``h3_latent_frames`` under the ``17n+5`` law, from inside that entry point.
    """
    assert "7f x" in _smoke(H3VideoVaeContractStub()), "22 pixel frames must encode to 7 latent"
    with pytest.raises(ValueError, match=r"17n\+5"):
        _smoke(H3VideoVaeContractStub(), frames=23)


def test_a_reference_that_does_not_encode_to_one_latent_frame_is_refused() -> None:
    """The single-frame branch is the whole subject; a multi-frame result means it was not taken."""
    source = inspect.getsource(h3_vae_smoke_encode)
    assert "expected 1" in source and "num_frames == 1 short-circuit" in source, (
        "the smoke must assert the reference took _encode's single-frame branch — that branch IS "
        "the family, and a reference that chunked temporally has silently encoded something else."
    )


# --------------------------------------------------------------------------------------------------
# It drives the REAL entry points — a smoke with its own encode proves only its own encode
# --------------------------------------------------------------------------------------------------


def test_the_smoke_calls_the_production_entry_points_and_never_the_vae_directly() -> None:
    """⛔ The D-10-DEF-9 lesson, applied to the guard itself.

    A smoke that re-implemented the encode would go green on its own logic while the loop kept
    doing something else — which is exactly how hand-rolled stubs kept the suite green for the whole
    life of D-10-DEF-9. Driving the real functions is what makes their channel checks, their frame
    law and the chokepoint's rank + device refusals part of the preflight.
    """
    tree = ast.parse(inspect.getsource(h3_vae_smoke_encode))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "encode_h3_video_latents" in called
    assert "encode_h3_reference_latents" in called
    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "encode" not in attribute_calls, (
        "the smoke must not call vae.encode itself — the video VAE is reached from exactly one "
        "function (tests/test_h3_preprocess_wiring), and a second path around it would bypass the "
        "rank and device refusals the chokepoint carries."
    )


# --------------------------------------------------------------------------------------------------
# Placement — the half that makes it worth having
# --------------------------------------------------------------------------------------------------


def test_the_smoke_runs_BEFORE_phase_a_not_at_the_top_of_phase_b() -> None:
    """⛔ Placement is the value, not the check.

    PHASE A is 88 samples through a 32B model and is most of what every dead container paid for. A
    smoke at the top of PHASE B would fail a few seconds earlier than the loop does and save
    nothing. Run before PHASE A it costs one small VAE load and two encodes.
    """
    body = _body_text()
    smoke_at = body.index("h3_vae_smoke_encode(")
    assert body.index("run_h3_arch_gate(") < smoke_at, (
        "the arch gate stays FIRST and unconditional — it also releases the 61.7 GiB partition the "
        "smoke would otherwise be sharing the card with."
    )
    assert smoke_at < body.index("AutoProcessor.from_pretrained"), (
        "the smoke must precede the processor mount, i.e. all of PHASE A."
    )
    assert smoke_at < body.index("AutoModel.from_pretrained"), (
        "the smoke must precede the Qwen3-VL-32B load — that load plus 88 text encodes is the cost "
        "this preflight exists to stop paying."
    )
    assert smoke_at < body.index("for index, row in enumerate(rows)"), (
        "the smoke must precede the per-sample loop entirely."
    )


def test_the_smoke_vae_is_released_before_qwen3_vl_loads() -> None:
    """Qwen3-VL-32B needs the card essentially to itself; a 3-shard reload costs seconds.

    The caller must drop its OWN reference too — ``assign=True`` loader-owned CUDA storage is freed
    only when the last reference goes away, and ``Module.to("cpu")`` does not do it (06-09 run-5).
    """
    body = _body_text()
    release_at = body.index("smoke_vae = None")
    assert body.index("h3_vae_smoke_encode(") < release_at < body.index("AutoModel.from_pretrained")
    assert "gc.collect()" in body[release_at : body.index("AutoModel.from_pretrained")], (
        "dropping the name is not enough on its own — the existing PHASE A teardown pairs it with "
        "gc.collect() + empty_cache(), and so must this one."
    )


def test_the_smoke_uses_the_REAL_row_zero_and_the_REAL_reference_resolver() -> None:
    """A synthetic sample would preflight synthetic geometry.

    Row 0 through ``_h3_resolve_references`` is the same manifest read, the same D-10-PAIRSEED
    rotation and the same descriptor mapping the loop will use — so what the smoke encodes is a
    sample the campaign actually contains.
    """
    body = _body_text()
    smoke_at = body.index("h3_vae_smoke_encode(")
    assert "smoke_row = rows[0]" in body[:smoke_at]
    assert "_h3_resolve_references(" in body[:smoke_at]
    assert "_h3_read_video_rgb(" in body
    assert "_h3_open_reference_image(" in body


def test_the_smoke_supplies_BOTH_modalities_at_its_call_site() -> None:
    """One clip AND one reference. Either alone re-opens exactly half of the family."""
    for call in ast.walk(_h3_preprocess_node()):
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "h3_vae_smoke_encode":
            supplied = {kw.arg for kw in call.keywords}
            assert "clip_pixels" in supplied, "the target-video modality is missing"
            assert "reference_image" in supplied, "the single-frame modality is missing"
            return
    raise AssertionError(
        "h3_preprocess does not call h3_vae_smoke_encode at all. Five containers died at the top "
        "of an 88-sample encode without it."
    )
