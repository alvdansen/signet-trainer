"""prep.h3_vae_contract — what the H3 VAEs ACTUALLY accept, and the ONLY sanctioned stubs.

⛔ **D-10-DEF-9, and the class it belongs to.**

The immediate defect was one rank: ``AutoencoderKLMiniMaxH3._encode`` requires 5-D
``[B, C, F, H, W]`` and ``encode_video_latents`` handed it 4-D. It cost a metered container, and it
reached one for a reason worth naming:

    plan 10-07 validated the encode on CPU with **hand-rolled stub VAEs that accepted 4-D**. The
    stub disagreed with the real component and manufactured confidence.

That is the same shape as D-10-DEF-4 and D-10-DEF-7 (a hand-reconstruction of
``Qwen3VLProcessor.__call__``), and it was closed the same way: by making the REAL component the
oracle. ``prep/h3_parity.py`` diffs our processor outputs against the real ``__call__``; this module
does the equivalent for the VAE, and adds the half the processor case did not need — **a stub that
cannot be more permissive than the thing it stands in for.**

How the three pieces fit
------------------------

1. ``assert_h3_vae_encode_input`` — the NAMED rank refusal, called by ``encode_video_latents``
   immediately before ``vae.encode``. Without it the next rank slip surfaces as a ``torch.cat``
   error 480 lines inside a vendored file, saying "got 4 and 5" about a padding tensor rather than
   "expected [B, C, F, H, W]" about the input.
2. ``H3VideoVaeContractStub`` — the **only** video-VAE stub the test suite may use. It calls that
   same refusal on its own input, so a CPU test that feeds it 4-D fails exactly where the real VAE
   would. ``tests/test_h3_vae_input_contract.py`` forbids a hand-rolled replacement by AST.
3. ``h3_vae_contract_report`` — one probe matrix, asked of ANY VAE-shaped object. Run against the
   REAL ``AutoencoderKLMiniMaxH3`` (built from config alone, no weights, ~0.1 s on CPU) and against
   the stub, the two reports must be EQUAL. A stub that accepts what the real class refuses fails
   by probe name; so does one that refuses what the real class accepts.

The AUDIO half, and why it is here now
--------------------------------------
There was no sanctioned audio-VAE stub at all — the audio path is measured unexercised (0 of 44
corpus clips carry a stream), so nothing had ever needed one. D-10-DEF-10 changed that: the runtime
idiom is a property of EVERY encode helper, and proving ``encode_h3_audio_latents`` obeys it needs an
audio component to drive. Writing that stub freehand would have been the D-10-DEF-9 mistake with a
different component, so ``H3AudioVaeContractStub`` is transcribed from
``AutoencoderKLMiniMaxH3Audio`` at the pinned SHA, derives its hop length from constants
``h3_geometry`` already owns, and is diffed against the real class by the same probe machinery.

⚠ That diff immediately found something (**D-10-DEF-11**, logged, NOT fixed here): the real class
refuses anything but ``[B, 1, samples]`` — stereo goes in as ``batch_size = 2`` — while
``modal/fns.py::_h3_read_audio_waveform`` hands over ``[1, 2, samples]``. It is the D-10-DEF-9 shape
on the audio path, and it has never run.

The autograd dimension (D-10-DEF-10)
------------------------------------
Both stubs hold a real ``torch.nn.Parameter``, so their outputs carry a ``grad_fn`` under an enabled
autograd context exactly as the real classes' do, and both record the grad mode they were called in.
A stub whose output never required grad would be **more permissive in the autograd dimension** — the
same failure as accepting a rank the real class refuses, and precisely the property a missing
``no_grad`` context is invisible against. The parameter is initialized to 1 and applied as a scale,
so every latent VALUE and every latent SHAPE is unchanged.

Why the error message lied, and why the fix is not "read it literally"
----------------------------------------------------------------------
At the pinned ``DIFFUSERS_SHA``::

    clip_length = self.config.clip_length          # 17
    num_frames = x.shape[2]
    if num_frames == 1:
        return self._encode_clip(x)
    if num_frames % clip_length != 0:
        pad_frames = x[:, :, -1:].repeat(1, 1, (-num_frames) % clip_length, 1, 1)
        x = torch.cat([x, pad_frames], dim=2)

On a 4-D ``[C, F, H, W]`` input ``x.shape[2]`` is the canvas **HEIGHT**, not the frame count, and
``Tensor.repeat`` with five arguments PREPENDS a dimension to the 4-D slice. So the observable
symptom is a rank mismatch inside ``torch.cat`` — ``got 4 and 5`` — and the real problem (an axis
read as something it is not) is invisible in it. The 768-tall canvas crashed only because
``768 % 17 == 3``; a canvas height that happened to be a multiple of 17 would have encoded the
HEIGHT axis as time and produced a plausible tensor with nothing raising.

⚠ **Every number below is DERIVED, never restated.** ``H3_VAE_TOKEN_DROP`` comes out of the frame
law itself, ``H3_VAE_SPATIAL_COMPRESSION`` out of the canvas multiple and the patch size, and
``tests/test_h3_vae_input_contract.py`` asserts each one against the real class's own
``config`` — so a diffusers bump that moves one of them fails here rather than in a container.

Import tier — **Modal-side EXCEPTION** (10-PATTERNS "Anti-Pattern 6", tier 3): ``torch`` is the only
module-scope third-party import, so a CPU box with no ``diffusers`` can import this module to read
the contract. The real class is only ever touched from the out-of-process probe runner.
"""

from __future__ import annotations

from typing import Any

import torch

from signet_trainer.conditioning.h3_geometry import (
    H3_AUDIO_LATENTS_PER_SECOND,
    H3_CANVAS_MULTIPLE,
    H3_FRAMES_PER_CHUNK,
    H3_LATENTS_PER_CHUNK,
    h3_latent_frames,
)
from signet_trainer.models.h3_loader import (
    EXPECTED_H3_AUDIO_IN_CHANNELS,
    EXPECTED_H3_IN_CHANNELS,
    EXPECTED_H3_PATCH_SIZE,
)

__all__ = [
    "H3_AUDIO_VAE_CONTRACT_PROBES",
    "H3_AUDIO_VAE_ENCODE_AXES",
    "H3_AUDIO_VAE_ENCODE_RANK",
    "H3_AUDIO_VAE_HOP_LENGTH",
    "H3_AUDIO_VAE_LATENT_CHANNELS",
    "H3_AUDIO_VAE_SAMPLING_RATE",
    "H3_AUDIO_VAE_WAVEFORM_CHANNEL_AXIS",
    "H3_VAE_CONTRACT_PROBES",
    "H3_VAE_ENCODE_AXES",
    "H3_VAE_ENCODE_RANK",
    "H3_VAE_FRAME_AXIS",
    "H3_VAE_PIXEL_CHANNELS",
    "H3_VAE_SPATIAL_COMPRESSION",
    "H3_VAE_TOKEN_DROP",
    "H3_VAE_UNBATCHED_RANK",
    "H3AudioVaeContractStub",
    "H3VideoVaeContractStub",
    "assert_h3_audio_vae_encode_input",
    "assert_h3_encode_device",
    "assert_h3_vae_encode_input",
    "h3_audio_vae_contract_report",
    "h3_audio_vae_latent_frames",
    "h3_component_device",
    "h3_vae_contract_report",
    "h3_vae_latent_frames",
]

# --------------------------------------------------------------------------------------------------
# The contract. Every value is derived from a single source that already exists.
# --------------------------------------------------------------------------------------------------

#: ⛔ ``AutoencoderKLMiniMaxH3._encode`` indexes ``x.shape[2]`` as the frame count and pads with a
#: five-argument ``Tensor.repeat``. Both are only meaningful on a 5-D tensor. This is the whole of
#: D-10-DEF-9.
H3_VAE_ENCODE_RANK: int = 5
H3_VAE_ENCODE_AXES: tuple[str, ...] = ("B", "C", "F", "H", "W")

#: The rank BOTH pixel producers on this path emit — ``[C, F, H, W]``, i.e. the 5-D contract with the
#: batch axis missing. ``encode_video_latents`` adds it and takes it back off, and it is derived
#: rather than written as ``4`` so the two ranks cannot be edited apart from each other.
H3_VAE_UNBATCHED_RANK: int = H3_VAE_ENCODE_RANK - 1

#: The axis ``_encode`` reads as time. On a 4-D ``[C, F, H, W]`` tensor this same index is the canvas
#: HEIGHT — which is why the failure was a concat-rank error rather than an honest frame-count one.
H3_VAE_FRAME_AXIS: int = H3_VAE_ENCODE_AXES.index("F")

#: RGB. Asserted against the real class's ``config.in_channels`` by the contract test, so this is a
#: checked derivation rather than a third opinion about what a pixel is.
H3_VAE_PIXEL_CHANNELS: int = 3

#: The trailing latent frames ``_encode`` drops (``config.token_drop``), DERIVED from the frame law
#: rather than restated: one 17-frame chunk yields ``H3_LATENTS_PER_CHUNK`` latent frames before the
#: drop, and ``h3_latent_frames`` says what survives it. 5 - 2 = 3.
H3_VAE_TOKEN_DROP: int = H3_LATENTS_PER_CHUNK - h3_latent_frames(H3_LATENTS_PER_CHUNK)

#: The VAE's own spatial factor. ``H3_CANVAS_MULTIPLE`` is ``vae_spatial_compression * patch_w``, so
#: the VAE's half is 32 // 2 = 16 — the same derivation ``h3_latent_grid_of_reference`` uses.
H3_VAE_SPATIAL_COMPRESSION: int = H3_CANVAS_MULTIPLE // EXPECTED_H3_PATCH_SIZE[2]


def assert_h3_vae_encode_input(pixels: Any, *, what: str) -> None:
    """Refuse anything but 5-D ``[B, C, F, H, W]``, BY NAME, before it reaches ``vae.encode``.

    ⛔ This exists because the library's own failure is unreadable. ``_encode`` treats
    ``x.shape[2]`` as the frame count; on a 4-D ``[C, F, H, W]`` that index is the canvas HEIGHT,
    and the five-argument ``Tensor.repeat`` used to build the padding PREPENDS an axis — so the slip
    surfaces as ``RuntimeError: Tensors must have same number of dimensions: got 4 and 5`` from a
    ``torch.cat`` deep inside a vendored file, describing the padding tensor rather than the input.

    ⚠ And it was LOUD only by luck. The failing canvas was 768 tall and ``768 % 17 == 3``, so the
    padding branch ran. A height that happened to be a multiple of the clip length would have
    skipped it, chunked the HEIGHT axis as time, and returned a plausibly-shaped latent nobody could
    have told from a correct one.

    Args:
        pixels: The tensor about to be handed to ``vae.encode``.
        what: The caller, named in the message — a rank slip should say which site produced it.
    """
    rank = int(getattr(pixels, "dim", lambda: -1)())
    if rank == H3_VAE_ENCODE_RANK:
        return
    shape = tuple(int(v) for v in getattr(pixels, "shape", ()))
    axes = ", ".join(H3_VAE_ENCODE_AXES)
    raise TypeError(
        f"[h3-vae-contract] {what} would hand the H3 video VAE a {rank}-D tensor {shape}; "
        f"AutoencoderKLMiniMaxH3._encode requires {H3_VAE_ENCODE_RANK}-D [{axes}] (D-10-DEF-9). "
        f"It reads x.shape[{H3_VAE_FRAME_AXIS}] as the FRAME COUNT — on a 4-D [C, F, H, W] that "
        f"index is the canvas HEIGHT — and pads with a five-argument Tensor.repeat that PREPENDS a "
        f"dimension, so the library's own error is a torch.cat rank mismatch about the padding "
        f"tensor ('got 4 and 5') and says nothing about the axis it actually misread. "
        f"encode_video_latents adds and removes the batch axis itself; any other path to "
        f"vae.encode must do the same."
    )


def h3_component_device(component: Any) -> torch.device | None:
    """The device an H3 component's OWN weights live on — read, never assumed. ``None`` if weightless.

    ⛔ **D-10-DEF-12.** Every encode helper in ``prep/h3_encode`` is handed a component it did not
    construct ("Passed in, never constructed here"), so it cannot know where that component lives —
    and a hardcoded ``"cuda"`` would be a second opinion about placement, which is the shape of every
    defect this phase has paid for. The component itself is the only honest source.

    Three sources, in decreasing authority:

    1. ``parameters()`` — the ``nn.Module`` truth. ``AutoencoderKLMiniMaxH3`` and Qwen3-VL both
       answer here.
    2. a ``device`` attribute — diffusers' ``ModelMixin`` exposes one, and ``_to_device`` in
       ``prep/h3_encode`` already reads it for the text encoder.
    3. a scan of the instance ``__dict__`` for a tensor — this is what makes the SANCTIONED CPU
       stubs answer too (``H3VideoVaeContractStub.scale`` is a real ``nn.Parameter``), so the device
       discipline is exercised by the ordinary suite rather than only on an A100.

    Returns ``None`` for a component with no tensors at all, and callers must treat that as "no
    placement opinion" — moving a tensor to a guessed device would be worse than leaving it.
    """
    parameters = getattr(component, "parameters", None)
    if callable(parameters):
        try:
            first = next(iter(parameters()), None)
        except TypeError:  # a non-Module object whose `parameters` is not iterable
            first = None
        if first is not None and hasattr(first, "device"):
            return first.device
    device = getattr(component, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    for value in vars(component).values():
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def assert_h3_encode_device(tensor: Any, component: Any, *, what: str) -> None:
    """Refuse a tensor that is not on its component's device, BY NAME, before the forward.

    ⛔ **This is D-10-DEF-12, and the library's failure is worse than unreadable — it is about the
    wrong subject entirely.** ``h3_preprocess`` wrote ``pixels.to("cuda")`` at the target-video call
    site and NOTHING at the reference one (``_reference_pixels`` builds its tensor with
    ``torch.from_numpy``, on CPU), so a CPU tensor reached a CUDA-resident VAE. PyTorch then::

        cudnn_is_acceptable(cpu_tensor) -> False
        _select_conv_backend(...)       -> ConvBackend.Slow3d
        aten::slow_conv3d_forward       -> no CUDA kernel exists

    and the message is ``NotImplementedError: Could not run 'aten::slow_conv3d_forward' with
    arguments from the 'CUDA' backend`` — which names a kernel, invites a hunt through cuDNN
    versions and tensor layouts, and never once says "device". **Measured, not reasoned**
    (``scripts/_h3_probe_modal.py::h3_conv3d_dispatch_probe``, three passes on a real A100): the
    same reference on CUDA takes ``ConvBackend.Cudnn`` for all 1,190 convs and succeeds, and the
    22-frame TARGET CLIP on CPU fails IDENTICALLY — so temporal extent was never the variable.

    ⚠ It is a ``RuntimeError``, matching the class PyTorch itself raises for placement, and it names
    both devices plus the chokepoint that is supposed to have already moved the tensor.

    Args:
        tensor: The tensor about to be handed to the component's forward.
        component: The loaded component (video VAE / audio VAE / text encoder).
        what: The caller, named in the message.
    """
    device = h3_component_device(component)
    if device is None:
        return
    actual = getattr(tensor, "device", None)
    if actual is None or actual.type == device.type:
        # Type equality, not index equality: a component sharded across cuda:0/cuda:1 answers with
        # one of its parameters' devices, and refusing cuda:1 pixels against a cuda:0 answer would
        # be a false alarm. The dispatch failure this guards is CPU-vs-CUDA, which is a type
        # difference — the house is single-GPU anyway (memory `single-gpu-preference`).
        return
    raise RuntimeError(
        f"[h3-vae-contract] {what} would run a component whose weights are on {device} against a "
        f"tensor on {actual} (D-10-DEF-12). PyTorch does NOT say that: cuDNN declines a non-CUDA "
        f"input, the dispatcher falls through to ConvBackend.Slow3d, and Slow3d has no CUDA kernel "
        f"— so the observable failure is \"Could not run 'aten::slow_conv3d_forward' with arguments "
        f"from the 'CUDA' backend\", which names a kernel and never mentions placement. "
        f"encode_video_latents moves the pixels onto the component's own device at ONE chokepoint; "
        f"any other path to a component forward must do the same."
    )


def h3_vae_latent_frames(pixel_frames: int) -> int:
    """Latent frames ``_encode`` returns for ``pixel_frames`` — the real class's algebra, derived.

    Two branches, both transcribed from the pinned ``_encode`` and both proven against the real
    class by ``tests/test_h3_vae_input_contract.py``:

    * ``F == 1`` short-circuits to ``_encode_clip`` — a single frame has no temporal extent to
      chunk, so it encodes to exactly ONE latent frame. **This is the reference-image path**, and it
      is why a reference must arrive as ``[1, 3, 1, H, W]``: at 4-D the same tensor would have its
      HEIGHT read as the frame count and take the chunking branch instead.
    * otherwise the clip is padded up to a whole number of ``H3_FRAMES_PER_CHUNK`` chunks, each
      chunk yields ``H3_LATENTS_PER_CHUNK`` latent frames, and ``H3_VAE_TOKEN_DROP`` trailing latent
      frames are dropped.

    For a conforming ``17n+5`` count this is identically ``h3_geometry.h3_latent_frames`` — 22 pads
    to 34, two chunks give 10, minus 3 is **7**. It is written generally here because the real class
    accepts non-conforming counts too, and a stub that refused them would be more RESTRICTIVE than
    the component it stands in for, which the contract report would (correctly) fail on.
    """
    frames = int(pixel_frames)
    if frames < 1:
        raise ValueError(f"a video needs at least one frame, got {frames}.")
    if frames == 1:
        return 1
    chunks = -(-frames // H3_FRAMES_PER_CHUNK)  # ceil, without importing math
    return H3_LATENTS_PER_CHUNK * chunks - H3_VAE_TOKEN_DROP


# --------------------------------------------------------------------------------------------------
# The ONE sanctioned stub.
# --------------------------------------------------------------------------------------------------


class _H3StubPosterior:
    """``DiagonalGaussianDistribution``'s two consumed methods, with the real signatures.

    ``sample`` keeps its ``generator`` keyword even though the value is fixed: dropping it would
    make ``encode_video_latents``'s fixed-seed draw (trap 3) an unexercised line in every CPU test.
    """

    def __init__(self, moments: torch.Tensor) -> None:
        self._moments = moments

    def sample(self, generator: Any = None) -> torch.Tensor:  # noqa: ARG002 — signature parity
        return self._moments

    def mode(self) -> torch.Tensor:
        return self._moments


class H3VideoVaeContractStub:
    """The ONLY video-VAE stub the test suite may use — it refuses what the real class refuses.

    ⛔ **A hand-rolled stub is what let D-10-DEF-9 reach a metered container.** Plan 10-07's CPU
    round trip used stubs whose ``encode`` did ``_, frames, height, width = pixels.shape`` — so they
    REQUIRED the 4-D input the real ``AutoencoderKLMiniMaxH3`` rejects, and every green run of that
    suite was evidence for the opposite of the truth. A stub is only worth anything when it is no
    more permissive than the component it replaces, and nothing about writing one by hand enforces
    that.

    So this one:

    * calls ``assert_h3_vae_encode_input`` — the SAME refusal production calls — on its input;
    * derives its latent frame count from ``h3_vae_latent_frames`` and its spatial factor from
      ``H3_VAE_SPATIAL_COMPRESSION``, both of which are checked against the real class's ``config``
      by ``tests/test_h3_real_class_parity.py``;
    * is diffed against the real class probe-for-probe by ``h3_vae_contract_report``, so "the stub
      agrees with reality" is a test result rather than an intention;
    * holds a real ``torch.nn.Parameter``, so it reproduces the real class's AUTOGRAD behaviour too
      (D-10-DEF-10). A stub whose output never requires grad cannot fail when a caller forgets the
      no-grad context — it is more permissive in the dimension that costs VRAM rather than the one
      that costs a shape, and that dimension is invisible to every contract diff.

    Cheap by construction: a constant fill scaled by one parameter, no convolutions, no ``diffusers``
    import. ``scale`` is initialized to 1, so every emitted VALUE and SHAPE is what it always was.
    """

    def __init__(
        self,
        *,
        fill: float = 0.25,
        latent_channels: int = EXPECTED_H3_IN_CHANNELS,
        device: Any = None,
    ) -> None:
        self.fill = float(fill)
        self.latent_channels = int(latent_channels)
        # ⛔ requires_grad=True is the POINT (D-10-DEF-10). A diffusers component arrives from
        # `from_pretrained` exactly like this, and `.eval()` does not change it. Multiplying the
        # moments through it is what gives the stub's output a `grad_fn` under an enabled autograd
        # context — i.e. what makes a missing no-grad context observable on CPU instead of on a
        # metered A100.
        #
        # ⛔ `device` is the D-10-DEF-12 dimension, added for the same reason `scale` carries a
        # gradient: a stub that lives nowhere cannot fail when a caller hands it a tensor from
        # somewhere else. Defaults to None (= CPU), so every pre-existing use is byte-identical;
        # tests pass `device="meta"` to get a component that is provably NOT where CPU pixels are,
        # on a box with no GPU.
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float32, device=device))
        #: Grad mode observed at each `encode` call, oldest first. The load-bearing observation:
        #: `requires_grad == False` on an output is ALSO satisfied by a trailing `.detach()`, which
        #: does nothing about the graph that was built and retained during the forward.
        self.grad_enabled_calls: list[bool] = []

    @property
    def last_grad_enabled(self) -> bool | None:
        """``torch.is_grad_enabled()`` as observed inside the most recent ``encode``, or ``None``."""
        return self.grad_enabled_calls[-1] if self.grad_enabled_calls else None

    def encode(  # noqa: FBT001, FBT002 — signature parity with diffusers' `encode`
        self, pixels: Any, return_dict: bool = True
    ) -> tuple[_H3StubPosterior, ...]:
        self.grad_enabled_calls.append(bool(torch.is_grad_enabled()))
        assert_h3_vae_encode_input(pixels, what="H3VideoVaeContractStub.encode")
        # ⛔ D-10-DEF-12: the SAME refusal production calls. The real class does not raise here — it
        # gets all the way into `F.conv3d`, where cuDNN declines the foreign tensor and the
        # dispatcher lands on a kernel that does not exist for CUDA. A stub that silently accepted
        # pixels from another device would be more permissive in exactly the dimension that cost
        # container #5.
        assert_h3_encode_device(pixels, self, what="H3VideoVaeContractStub.encode")
        batch, channels, frames, height, width = (int(v) for v in pixels.shape)
        if channels != H3_VAE_PIXEL_CHANNELS:
            raise ValueError(
                f"[h3-vae-contract] the H3 video VAE takes {H3_VAE_PIXEL_CHANNELS}-channel RGB "
                f"(config.in_channels); got {channels}. The real class fails in its first "
                f"convolution, so the stub must refuse here rather than invent a latent for pixels "
                f"the VAE could not have encoded."
            )
        if return_dict:
            raise NotImplementedError(
                "[h3-vae-contract] H3VideoVaeContractStub only implements the return_dict=False "
                "form, which is the ONLY form prep/h3_encode uses (`vae.encode(x, "
                "return_dict=False)[0]`). Building diffusers' AutoencoderKLOutput would need "
                "diffusers installed, which is exactly what this stub exists to avoid."
            )
        moments = torch.full(
            (
                batch,
                self.latent_channels,
                h3_vae_latent_frames(frames),
                height // H3_VAE_SPATIAL_COMPRESSION,
                width // H3_VAE_SPATIAL_COMPRESSION,
            ),
            self.fill,
            dtype=torch.float32,
            # A real module emits on its OWN device. Emitting on CPU regardless would make the
            # stub's output a poorer stand-in than its input check (D-10-DEF-12).
            device=self.scale.device,
        )
        # The parameter multiply: identity on the values (scale == 1), but it is what routes the
        # output through autograd when autograd is on — the real VAE's behaviour, reproduced.
        return (_H3StubPosterior(moments * self.scale),)


# --------------------------------------------------------------------------------------------------
# The probe matrix — one set of questions, asked of the stub AND of the real class.
# --------------------------------------------------------------------------------------------------

#: ``(name, shape)`` pairs covering every rank the pipeline can produce plus the neighbours a slip
#: would land on. The two **4-D** entries are the D-10-DEF-9 defect verbatim: ``_h3_read_video_rgb``
#: returns ``[3, F, H, W]`` and ``_reference_pixels`` returns ``[3, 1, H, W]``, and both reached
#: ``vae.encode`` through ``encode_video_latents``. Spatial dims are deliberately tiny and unequal
#: (32x48) — the report is about rank and shape ALGEBRA, and unequal axes catch a transposed grid.
H3_VAE_CONTRACT_PROBES: tuple[tuple[str, tuple[int, ...]], ...] = (
    # ⛔ the two production 4-D shapes — the defect itself
    ("target_video_4d", (3, 22, 32, 48)),
    ("reference_image_4d", (3, 1, 32, 48)),
    # the same two, correctly batched
    ("target_video_5d", (1, 3, 22, 32, 48)),
    ("reference_image_5d", (1, 3, 1, 32, 48)),
    # the frame law across chunk counts: n = 0, 1, 2
    ("one_chunk_5d", (1, 3, 5, 32, 48)),
    ("three_chunk_5d", (1, 3, 39, 32, 48)),
    # a NON-conforming count: the real class pads it, so a stub that refused would be too strict
    ("non_conforming_5d", (1, 3, 10, 32, 48)),
    # a real batch axis, so "batch 1" is not the only thing ever proven
    ("batch_of_two_5d", (2, 3, 22, 32, 48)),
    # neighbouring ranks a future slip could land on
    ("still_image_3d", (3, 32, 48)),
    ("nchw_image_4d", (1, 3, 32, 48)),
    # wrong pixel channels, at the right rank
    ("wrong_channels_5d", (1, 5, 22, 32, 48)),
)


def h3_vae_contract_report(vae: Any) -> dict[str, Any]:
    """Ask every probe of ``vae`` and report accept/reject + the realized latent shape.

    JSON-serializable on purpose: the real-class half runs in a subprocess under the interpreter
    that has ``diffusers`` at the pinned SHA, and the verdicts have to cross that boundary intact.

    ⚠ The probe asks the question production asks — ``vae.encode(x, return_dict=False)[0]`` then
    ``.sample()`` — so it exercises the same two-step the recipe does. A verdict is ``"accept"``
    only if BOTH steps complete; the error TYPE is reported (never the message, which for the real
    class is the misleading ``got 4 and 5``).
    """
    return _contract_report(vae, H3_VAE_CONTRACT_PROBES, posterior="sample")


def _contract_report(
    vae: Any, probes: tuple[tuple[str, tuple[int, ...]], ...], *, posterior: str
) -> dict[str, Any]:
    """The shared probe loop for both VAEs — shape verdicts, plus the D-10-DEF-10 grad verdict.

    ⛔ **Two passes, and the second one is the new half.** The first runs under ``torch.no_grad()``,
    which is how production calls the component, and yields the accept/reject verdict and the latent
    shape. The second re-runs the ACCEPTED probes with autograd **enabled** and records whether the
    output requires grad.

    That second pass is what makes the stub's autograd behaviour a diffed FACT rather than an
    intention. A stub that quietly returned a constant would report ``requires_grad: false`` where
    the real class reports ``true``, and every CPU test asserting "the encode ran under no_grad"
    would be vacuous — green whether or not the context existed. It is the exact shape of the
    D-10-DEF-9 stub failure, moved one dimension over.
    """
    report: dict[str, Any] = {}
    for name, shape in probes:
        pixels = torch.zeros(*shape, dtype=torch.float32)
        try:
            with torch.no_grad():
                distribution = vae.encode(pixels, return_dict=False)[0]
                latents = _draw(distribution, posterior)
            report[name] = {
                "shape": list(shape),
                "verdict": "accept",
                "latent_shape": [int(v) for v in latents.shape],
                "error": None,
                # Under no_grad NOTHING may require grad, real class or stub. Reported rather than
                # assumed: it is the production call form, and a `True` here would mean the context
                # is not doing what the whole fix depends on it doing.
                "requires_grad_under_no_grad": bool(latents.requires_grad),
                "requires_grad_when_enabled": _requires_grad_when_enabled(
                    vae, pixels, posterior
                ),
            }
        except Exception as exc:  # noqa: BLE001 — the VERDICT is the datum, not the traceback
            report[name] = {
                "shape": list(shape),
                "verdict": "reject",
                "latent_shape": None,
                "error": type(exc).__name__,
                "requires_grad_under_no_grad": None,
                "requires_grad_when_enabled": None,
            }
    return report


def _draw(distribution: Any, posterior: str) -> torch.Tensor:
    """``.sample()`` or ``.mode()`` — the two policies the pipeline runs (trap 4)."""
    if posterior == "mode":
        return distribution.mode()
    return distribution.sample(generator=torch.Generator().manual_seed(0))


def _requires_grad_when_enabled(vae: Any, pixels: torch.Tensor, posterior: str) -> bool | None:
    """Does an ACCEPTED probe's output carry a graph when autograd is on? ``None`` if it re-raised.

    ⛔ This is the D-10-DEF-10 dimension. The real VAEs' parameters require grad (``.eval()`` does
    not change that), so the honest answer for them is ``True`` — which is precisely why an encode
    without a no-grad context retained ~78 GiB of activations. A stub that answered ``False`` would
    be more permissive than the component it stands in for, in the one dimension no shape diff sees.
    """
    try:
        with torch.enable_grad():
            distribution = vae.encode(pixels, return_dict=False)[0]
            latents = _draw(distribution, posterior)
        return bool(latents.requires_grad)
    except Exception:  # noqa: BLE001 — an accepted probe that re-raises is reported, not swallowed
        return None


# --------------------------------------------------------------------------------------------------
# The AUDIO VAE — same doctrine, transcribed from AutoencoderKLMiniMaxH3Audio at the pinned SHA.
# --------------------------------------------------------------------------------------------------

#: ⛔ ``AutoencoderKLMiniMaxH3Audio.encode`` opens with
#: ``if sample.ndim != 3 or sample.shape[1] != 1: raise ValueError``. Stereo is **NOT** two channels
#: on axis 1 — the class's own docstring says *"MiniMax-H3 passes the two stereo channels of a
#: reference clip as ``batch_size = 2``"*. So the contract is ``[B, 1, samples]``, and a stereo clip
#: is ``[2, 1, samples]``.
H3_AUDIO_VAE_ENCODE_RANK: int = 3
H3_AUDIO_VAE_ENCODE_AXES: tuple[str, ...] = ("B", "1", "samples")

#: The axis the real class requires to be exactly 1. Named rather than written as ``1`` twice.
H3_AUDIO_VAE_WAVEFORM_CHANNEL_AXIS: int = H3_AUDIO_VAE_ENCODE_AXES.index("1")

#: The audio VAE's declared rate. Asserted against the real class's ``config.sampling_rate`` by
#: ``tests/test_h3_real_class_parity.py`` — a checked restatement, not a third opinion. It is the
#: rate ``modal/fns.py`` resamples every clip to, read off the mounted config at runtime.
H3_AUDIO_VAE_SAMPLING_RATE: int = 32_000

#: ``self.hop_length = math.prod(encoder_rates)`` = ``prod((2, 4, 4, 5, 5))`` = 800. DERIVED from the
#: latent rate ``h3_geometry`` already owns rather than restated: ``H3_AUDIO_LATENTS_PER_SECOND`` is
#: how the packed-row budget prices audio, so if the two ever disagree the budget is wrong too.
H3_AUDIO_VAE_HOP_LENGTH: int = int(round(H3_AUDIO_VAE_SAMPLING_RATE / H3_AUDIO_LATENTS_PER_SECOND))

#: The posterior's channel count. Same integer ``encode_h3_audio_latents`` checks its result against.
H3_AUDIO_VAE_LATENT_CHANNELS: int = EXPECTED_H3_AUDIO_IN_CHANNELS


def assert_h3_audio_vae_encode_input(waveform: Any, *, what: str) -> None:
    """Refuse anything but ``[B, 1, samples]``, BY NAME, before it reaches ``audio_vae.encode``.

    The real class's own message is already honest (it prints the shape it wanted), unlike the video
    VAE's misleading ``got 4 and 5``. This exists so the STUB can refuse identically without
    reimplementing the condition twice, and so the refusal names the layout trap explicitly: stereo
    is ``batch_size = 2``, **not** two channels on axis 1.
    """
    rank = int(getattr(waveform, "dim", lambda: -1)())
    shape = tuple(int(v) for v in getattr(waveform, "shape", ()))
    channel_ok = (
        rank == H3_AUDIO_VAE_ENCODE_RANK
        and shape[H3_AUDIO_VAE_WAVEFORM_CHANNEL_AXIS] == 1
    )
    if channel_ok:
        return
    axes = ", ".join(H3_AUDIO_VAE_ENCODE_AXES)
    raise ValueError(
        f"[h3-vae-contract] {what} would hand the H3 audio VAE a {rank}-D tensor {shape}; "
        f"AutoencoderKLMiniMaxH3Audio.encode requires {H3_AUDIO_VAE_ENCODE_RANK}-D [{axes}] and "
        f"raises on anything else. ⚠ STEREO IS batch_size=2, NOT two channels on axis "
        f"{H3_AUDIO_VAE_WAVEFORM_CHANNEL_AXIS}: a stereo clip is [2, 1, samples]. A [1, 2, samples] "
        f"waveform is the shape modal/fns.py::_h3_read_audio_waveform emits today (D-10-DEF-11)."
    )


def h3_audio_vae_latent_frames(samples: int) -> int:
    """Latent frames for ``samples`` waveform samples — the real class's algebra, derived.

    ``encode`` right-pads the waveform up to a whole number of ``hop_length`` (800) windows and the
    encoder stride is that same hop, so the count is ``ceil(samples / hop_length)``.
    """
    count = int(samples)
    if count < 1:
        raise ValueError(f"a waveform needs at least one sample, got {count}.")
    return -(-count // H3_AUDIO_VAE_HOP_LENGTH)  # ceil, without importing math


class H3AudioVaeContractStub:
    """The ONLY audio-VAE stub the test suite may use. Same doctrine as its video sibling.

    It calls ``assert_h3_audio_vae_encode_input`` (the SAME refusal it would face for real), derives
    its latent length from ``h3_audio_vae_latent_frames``, and holds a grad-requiring parameter so a
    missing no-grad context is observable (D-10-DEF-10). ``h3_audio_vae_contract_report`` diffs it
    against the real ``AutoencoderKLMiniMaxH3Audio``, built from config alone with no weights.
    """

    def __init__(
        self,
        *,
        fill: float = 0.125,
        latent_channels: int = H3_AUDIO_VAE_LATENT_CHANNELS,
        device: Any = None,
    ) -> None:
        self.fill = float(fill)
        self.latent_channels = int(latent_channels)
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float32, device=device))
        self.grad_enabled_calls: list[bool] = []

    @property
    def last_grad_enabled(self) -> bool | None:
        """``torch.is_grad_enabled()`` as observed inside the most recent ``encode``, or ``None``."""
        return self.grad_enabled_calls[-1] if self.grad_enabled_calls else None

    def encode(  # noqa: FBT001, FBT002 — signature parity with diffusers' `encode`
        self, sample: Any, return_dict: bool = True
    ) -> tuple[_H3StubPosterior, ...]:
        self.grad_enabled_calls.append(bool(torch.is_grad_enabled()))
        assert_h3_audio_vae_encode_input(sample, what="H3AudioVaeContractStub.encode")
        assert_h3_encode_device(sample, self, what="H3AudioVaeContractStub.encode")  # D-10-DEF-12
        batch, _, samples = (int(v) for v in sample.shape)
        if return_dict:
            raise NotImplementedError(
                "[h3-vae-contract] H3AudioVaeContractStub only implements the return_dict=False "
                "form, which is the ONLY form prep/h3_encode uses (`audio_vae.encode(x, "
                "return_dict=False)[0]`)."
            )
        moments = torch.full(
            (batch, self.latent_channels, h3_audio_vae_latent_frames(samples)),
            self.fill,
            dtype=torch.float32,
            device=self.scale.device,  # a real module emits on its own device (D-10-DEF-12)
        )
        return (_H3StubPosterior(moments * self.scale),)


#: ``(name, shape)`` pairs for the audio VAE. Sample counts are tiny multiples/non-multiples of the
#: hop so the ceil is exercised without building a real waveform.
H3_AUDIO_VAE_CONTRACT_PROBES: tuple[tuple[str, tuple[int, ...]], ...] = (
    # the accepted layout: mono, and stereo as TWO BATCH ELEMENTS
    ("mono_1x1xN", (1, 1, H3_AUDIO_VAE_HOP_LENGTH * 3)),
    ("stereo_as_batch_2x1xN", (2, 1, H3_AUDIO_VAE_HOP_LENGTH * 3)),
    # a NON-multiple of the hop: the real class right-pads, so a stub that refused would be stricter
    ("ragged_1x1xN", (1, 1, H3_AUDIO_VAE_HOP_LENGTH * 2 + 1)),
    # ⛔ D-10-DEF-11 verbatim — what `_h3_read_audio_waveform` emits today
    ("stereo_on_axis1_1x2xN", (1, 2, H3_AUDIO_VAE_HOP_LENGTH * 3)),
    # neighbouring ranks a slip could land on
    ("bare_stereo_2d", (2, H3_AUDIO_VAE_HOP_LENGTH * 3)),
    ("bare_mono_1d", (H3_AUDIO_VAE_HOP_LENGTH * 3,)),
    ("batched_stereo_4d", (1, 2, 1, H3_AUDIO_VAE_HOP_LENGTH * 3)),
)


def h3_audio_vae_contract_report(audio_vae: Any) -> dict[str, Any]:
    """Ask every audio probe of ``audio_vae``, using the posterior policy production uses.

    ``.mode()`` — the reference-soundtrack policy, and the one the real class's own docstring says
    MiniMax-H3 *always* consumes. The target-audio ``.sample()`` policy draws noise through a
    generator and would make the report non-deterministic for no gain: the questions here are rank,
    refusal, latent shape and autograd, none of which the draw changes.
    """
    return _contract_report(audio_vae, H3_AUDIO_VAE_CONTRACT_PROBES, posterior="mode")
