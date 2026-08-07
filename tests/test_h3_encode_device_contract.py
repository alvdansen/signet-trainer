"""D-10-DEF-12 — placement is decided ONCE, off the component, and refused BY NAME if it is not.

The defect
----------
``h3_preprocess`` wrote ``pixels.to("cuda")`` at the target-video call site and NOTHING at the
reference one. ``_reference_pixels`` builds its tensor with ``torch.from_numpy`` — on CPU — so the
reference reached a CUDA-resident VAE with CPU pixels. What PyTorch then does is the reason this
cost a container instead of a minute::

    cudnn_is_acceptable(cpu_tensor)  -> False
    _select_conv_backend(...)        -> ConvBackend.Slow3d
    aten::slow_conv3d_forward        -> has no CUDA kernel

    NotImplementedError: Could not run 'aten::slow_conv3d_forward' with arguments from the
                         'CUDA' backend.

That message names a kernel. It never says "device". The reporting agent's read — *"the only
differing input property is the temporal extent, 1 vs 17"* — was the honest conclusion from the two
``encode_video_latents`` calls, and it was wrong: the difference was one ``.to("cuda")`` two frames
up the stack, at a site the traceback does not show.

Measured, not reasoned (``scripts/_h3_probe_modal.py::h3_conv3d_dispatch_probe``, three passes on a
real A100):

===========================================  ========  ======================  ==============
case                                         pixels    backend                 result
===========================================  ========  ======================  ==============
target clip, 22 frames, 768x1344              cuda      ``Cudnn`` x1904         OK
reference A, 1 frame, 1344x896                cuda      ``Cudnn`` x1190         OK
reference A, 1 frame, 1344x896                **cpu**   ``Slow3d``              the exact error
**target clip, 22 frames** (the "known good")  **cpu**   ``Slow3d``              **the same error**
===========================================  ========  ======================  ==============

The fourth row is why this file exists. The 22-frame clip fails IDENTICALLY on CPU pixels, so
temporal extent was never the variable and no amount of reasoning about ``num_frames == 1`` would
have found it. The video path survived on one ad-hoc literal.

What is pinned here
-------------------
1. ``h3_component_device`` reads placement off the component, from three sources in order.
2. ``assert_h3_encode_device`` refuses a foreign tensor BY NAME, and is a no-op when the component
   has no opinion.
3. ``encode_video_latents`` MOVES, at the chokepoint, before ``imagenet_normalize`` — behaviourally
   (via a sanctioned stub on the ``meta`` device, so a GPU-less box can still prove it) and
   structurally (AST order).
4. **A mutation probe that lives in the suite**: with the move neutralized, the named refusal fires.
5. ``modal/fns.py::h3_preprocess`` carries NO device literal on any encode call site — the
   two-opinions state IS the defect, so its absence is asserted rather than assumed.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from signet_trainer.prep import h3_encode
from signet_trainer.prep.h3_encode import (
    encode_h3_audio_latents,
    encode_video_latents,
)
from signet_trainer.prep.h3_vae_contract import (
    H3_AUDIO_VAE_HOP_LENGTH,
    H3AudioVaeContractStub,
    H3VideoVaeContractStub,
    assert_h3_encode_device,
    h3_component_device,
)

FNS_SOURCE = (
    Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal" / "fns.py"
).read_text(encoding="utf-8")

#: The video VAE's real per-channel stats are 24 long; 1-element vectors broadcast identically and
#: keep these tests about placement rather than about numbers that live in a checkpoint config.
MEAN = 0.0
STD = 1.0


def _clip(frames: int = 22) -> torch.Tensor:
    """One tiny CPU clip in the rank BOTH production producers emit — ``[C, F, H, W]``."""
    return torch.zeros(3, frames, 32, 48, dtype=torch.uint8)


def _h3_preprocess_node() -> ast.FunctionDef:
    """``h3_preprocess`` as an AST node — the scan is structural, never a text sweep.

    A text sweep cannot work here: ``text_encoder.to("cuda")`` is a legitimate MODEL LOAD placement
    (a module has to be put somewhere, and that is the load site's job), while ``pixels.to("cuda")``
    handed to an encode helper is the defect. They are indistinguishable as strings and trivially
    distinguishable as syntax.
    """
    for node in ast.walk(ast.parse(FNS_SOURCE)):
        if isinstance(node, ast.FunctionDef) and node.name == "h3_preprocess":
            return node
    raise AssertionError("h3_preprocess not found in modal/fns.py")


def _callee_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


# --------------------------------------------------------------------------------------------------
# 1. h3_component_device — three sources, in order, and an honest None.
# --------------------------------------------------------------------------------------------------


def test_the_device_comes_from_parameters_when_the_component_is_a_module() -> None:
    """An ``nn.Module`` answers from ``parameters()`` — the diffusers/Qwen3-VL case."""
    assert h3_component_device(torch.nn.Linear(2, 2)) == torch.device("cpu")
    assert h3_component_device(torch.nn.Linear(2, 2).to("meta")) == torch.device("meta")


def test_the_device_comes_from_a_device_attribute_when_there_are_no_parameters() -> None:
    """diffusers' ``ModelMixin`` exposes ``.device``; ``prep/h3_encode._to_device`` already uses it."""

    class _Declared:
        device = torch.device("meta")

    assert h3_component_device(_Declared()) == torch.device("meta")

    class _DeclaredAsString:
        device = "meta"

    assert h3_component_device(_DeclaredAsString()) == torch.device("meta")


def test_the_sanctioned_stubs_answer_too_via_the_instance_scan() -> None:
    """The scan is what makes the whole discipline testable without a GPU.

    ``H3VideoVaeContractStub`` is not an ``nn.Module`` and declares no ``device``; it holds a real
    ``nn.Parameter``. If ``h3_component_device`` could not see that, every behavioural test below
    would be vacuous on a CPU box — which is the D-10-DEF-9 stub failure, one dimension over.
    """
    assert h3_component_device(H3VideoVaeContractStub()) == torch.device("cpu")
    assert h3_component_device(H3VideoVaeContractStub(device="meta")) == torch.device("meta")
    assert h3_component_device(H3AudioVaeContractStub(device="meta")) == torch.device("meta")


def test_a_component_with_no_tensors_has_no_opinion() -> None:
    """``None`` means "no placement opinion", and callers must NOT invent one.

    Moving a tensor to a guessed device would be worse than leaving it: it would make a genuine
    mismatch un-observable at the one place designed to observe it.
    """

    # ⚠ Deliberately NOT given an `encode` method: `tests/test_h3_real_class_parity` forbids a
    # hand-rolled VAE stub by AST (the `return_dict` keyword is the discriminator), and it is right
    # to — a second stub is how D-10-DEF-9 stayed green. The question here is only "does this object
    # declare a device", which needs no encode at all.
    class _Weightless:
        pass

    assert h3_component_device(_Weightless()) is None


# --------------------------------------------------------------------------------------------------
# 2. assert_h3_encode_device — the named refusal.
# --------------------------------------------------------------------------------------------------


def test_a_matching_device_passes() -> None:
    assert_h3_encode_device(torch.zeros(2), torch.nn.Linear(2, 2), what="probe")


def test_a_foreign_tensor_is_refused_and_the_message_names_BOTH_devices() -> None:
    """The whole point: a message that says *device*, where the library's says *slow_conv3d*."""
    with pytest.raises(RuntimeError) as excinfo:
        assert_h3_encode_device(
            torch.zeros(2), torch.nn.Linear(2, 2).to("meta"), what="encode_video_latents"
        )
    message = str(excinfo.value)
    assert "meta" in message and "cpu" in message, message
    assert "encode_video_latents" in message, "the refusal must name the site that produced it"
    assert "D-10-DEF-12" in message, "a refusal with no defect id is a dead end for the next reader"
    assert "slow_conv3d_forward" in message, (
        "the message must carry the string the NEXT person will paste into a search box — that "
        "text is what a container prints, and connecting it to 'device' is the entire value here."
    )


def test_a_component_with_no_device_opinion_is_never_refused() -> None:
    class _Weightless:
        pass

    assert_h3_encode_device(torch.zeros(2), _Weightless(), what="probe")


def test_the_refusal_is_by_device_TYPE_not_by_index() -> None:
    """cuda:0 vs cuda:1 is not this defect, and refusing it would be a false alarm.

    The failure guarded against is CPU-vs-CUDA — a *type* difference. A sharded component answers
    with one of its parameters' devices, and the house is single-GPU regardless
    (memory ``single-gpu-preference``).
    """
    a = torch.zeros(2, device="cpu")
    assert_h3_encode_device(a, torch.nn.Linear(2, 2), what="probe")


# --------------------------------------------------------------------------------------------------
# 3. The chokepoint MOVES — behaviourally, on a box with no GPU.
# --------------------------------------------------------------------------------------------------


def test_cpu_pixels_reach_a_non_cpu_vae_because_the_chokepoint_moves_them() -> None:
    """⛔ THE DEFECT, verbatim — a CPU tensor and a component that lives somewhere else.

    Before the fix this is the reference path exactly: ``_reference_pixels`` -> CPU tensor ->
    ``encode_video_latents`` -> a VAE on another device. It now succeeds, and the latents come back
    on the VAE's device rather than the producer's.
    """
    vae = H3VideoVaeContractStub(device="meta")
    latents = encode_video_latents(vae, _clip(), MEAN, STD)
    assert latents.device == torch.device("meta")


def test_the_single_frame_REFERENCE_shape_is_covered_by_the_same_move() -> None:
    """``[3, 1, H, W]`` — the producer that had no ``.to(...)`` at its call site at all."""
    vae = H3VideoVaeContractStub(device="meta")
    latents = encode_video_latents(vae, _clip(frames=1), MEAN, STD)
    assert latents.device == torch.device("meta")
    assert latents.dim() == 4, "the documented same-rank-as-pixels promise still holds"


def test_a_component_with_no_device_opinion_leaves_the_pixels_alone() -> None:
    """No opinion means no move — and no fabricated one."""
    vae = H3VideoVaeContractStub()
    assert encode_video_latents(vae, _clip(), MEAN, STD).device == torch.device("cpu")


def test_the_audio_helper_moves_too_even_though_that_path_has_never_run() -> None:
    """0 of 44 corpus clips carry a stream, which is exactly why this is pinned.

    Its ``waveform.to("cuda")`` lived at a call site that has never executed, so the gap would have
    shipped and bought its own container on the first audio campaign — the same argument that put
    ``@h3_no_grad`` on this helper (D-10-DEF-10).
    """
    audio_vae = H3AudioVaeContractStub(device="meta")
    waveform = torch.zeros(2, 1, H3_AUDIO_VAE_HOP_LENGTH * 3, dtype=torch.float32)
    latents = encode_h3_audio_latents(audio_vae, waveform, is_reference=True)
    assert latents is not None
    assert latents.device == torch.device("meta")


# --------------------------------------------------------------------------------------------------
# 4. The MUTATION PROBE — it lives in the suite, not in a one-off session.
# --------------------------------------------------------------------------------------------------


def test_neutralizing_the_move_makes_the_named_refusal_fire(monkeypatch) -> None:  # noqa: ANN001
    """D-10-DEF-12 reintroduced: the move is skipped, and the guard must catch it BY NAME.

    Patching ``h3_encode.h3_component_device`` to have no opinion is the precise mutation — the
    chokepoint's ``if device is not None`` branch goes dead while ``assert_h3_encode_device``
    (which resolves the device through its OWN module) still sees the truth. If the assertion were
    ever softened to reuse the same lookup, this test goes green for the wrong reason and the next
    test below catches that.
    """
    monkeypatch.setattr(h3_encode, "h3_component_device", lambda component: None)  # noqa: ARG005
    with pytest.raises(RuntimeError, match="D-10-DEF-12"):
        encode_video_latents(H3VideoVaeContractStub(device="meta"), _clip(), MEAN, STD)


def test_the_refusal_does_not_route_through_the_callers_lookup() -> None:
    """The guard must not be satisfiable by neutralizing the mover — they are separate reads."""
    source = inspect.getsource(h3_encode.encode_video_latents)
    assert "assert_h3_encode_device(normalized, vae" in source, (
        "the refusal takes the COMPONENT, not a device value the caller already computed. Passing "
        "a precomputed device would let one edit disable the move and the check together."
    )


# --------------------------------------------------------------------------------------------------
# 5. Structure — the order inside the chokepoint, and NO device literal at the call sites.
# --------------------------------------------------------------------------------------------------


def test_the_move_precedes_the_normalization_and_the_refusal_precedes_the_encode() -> None:
    """Order is load-bearing twice over.

    * move BEFORE ``imagenet_normalize``: uint8 crosses the bus instead of float32 (4x less), and
      it is bit-identically what the target-video path already did and attempt 5 proved on an A100.
    * refusal BEFORE ``vae.encode``: after it, the only available error is the one about a kernel.
    """
    body = inspect.getsource(h3_encode.encode_video_latents)
    body = body[body.index('"""', body.index('"""') + 3) :]  # drop the docstring
    move_at = body.index("pixels.to(device)")
    normalize_at = body.index("imagenet_normalize(pixels)")
    assert_at = body.index("assert_h3_encode_device(")
    encode_at = body.index("vae.encode(")
    assert move_at < normalize_at < assert_at < encode_at, (
        f"expected move < normalize < assert < encode, got "
        f"{move_at} / {normalize_at} / {assert_at} / {encode_at}"
    )


def test_no_encode_call_site_in_h3_preprocess_places_its_own_tensor() -> None:
    """⛔ The two-opinions state IS the defect — its absence is asserted, never assumed.

    One site wrote ``pixels.to("cuda")`` and the other wrote nothing, so the two producers reached
    the same VAE from different devices. Re-adding a literal at ANY encode call site restores that
    exact shape, and the one that gets forgotten next is the one that fails.

    ⚠ MODEL LOAD placement is deliberately NOT flagged. ``text_encoder.to("cuda")`` and
    ``component.to(device)`` inside ``_h3_load_component`` are where a module is *supposed* to be
    put; the rule is that a TENSOR ARGUMENT to an encode helper must not carry its own opinion,
    because the helper reads placement off the component it was handed.
    """
    offenders: list[str] = []
    for call in ast.walk(_h3_preprocess_node()):
        if not isinstance(call, ast.Call):
            continue
        callee = _callee_name(call)
        if not (callee.startswith("encode_") or callee == "h3_vae_smoke_encode"):
            continue
        for argument in [*call.args, *(kw.value for kw in call.keywords)]:
            for inner in ast.walk(argument):
                if isinstance(inner, ast.Call) and _callee_name(inner) == "to":
                    offenders.append(f"{callee}(... {ast.unparse(inner)} ...)")
    assert not offenders, (
        f"an encode call site places its own tensor: {offenders}. Placement belongs to "
        f"encode_video_latents / encode_h3_audio_latents, which read the device off the component "
        f"(D-10-DEF-12). A literal here is a SECOND opinion, and this pre-encode has now been "
        f"broken twice by exactly that shape — once for the resize (D-10-DEF-4) and once for the "
        f"device."
    )


def test_the_sanctioned_stubs_refuse_a_foreign_tensor_like_the_real_class_does() -> None:
    """A stub must not be more permissive than the component it stands in for.

    The real ``AutoencoderKLMiniMaxH3`` does not raise a friendly error here — it gets into
    ``F.conv3d`` and dies on a missing kernel. Either way it is a hard failure, and a stub that
    accepted the tensor would make every device test in this file vacuous.
    """
    with pytest.raises(RuntimeError, match="D-10-DEF-12"):
        H3VideoVaeContractStub(device="meta").encode(
            torch.zeros(1, 3, 22, 32, 48, dtype=torch.float32), return_dict=False
        )
    with pytest.raises(RuntimeError, match="D-10-DEF-12"):
        H3AudioVaeContractStub(device="meta").encode(
            torch.zeros(1, 1, H3_AUDIO_VAE_HOP_LENGTH, dtype=torch.float32), return_dict=False
        )
