"""Phase 10 (H3-03) — the H3 pre-encode's four silent-corruption traps, pinned by a source scan.

Scans ``src/signet_trainer/prep/h3_encode.py`` as TEXT, with comments and docstrings stripped first,
mirroring ``tests/test_preprocess_wiring.py`` / ``test_no_warm_gpu.py`` / ``test_h3_arch_constants.py``.
The stripping is the whole point of the convention: this module's own WARNING prose names every trap
it guards against, so an un-stripped scan would pass on the warnings alone while the code did the
wrong thing.

Why a source scan and not a behavioral test. All four traps degrade **silently** — they produce
correctly-shaped, plausible tensors that train against off-distribution conditioning. There is no
cheap assertion that catches them at runtime, and the expensive one costs a metered A100 container.
A structural guard is what makes them un-droppable by a later refactor:

  1. the text-encoder layer (a final-layer encode is wrong at the correct ``(B, L, 5120)`` shape)
  2. the fixed encode seed (a run-seeded posterior is off-distribution on every sample)
  3. ImageNet pixel normalization over a ``[0, 1]`` base (most VAEs in this stack are ``[-1, 1]``)
  4. the sample-vs-mode posterior split (reference soundtracks take the MEAN, everything else is
     sampled)

plus the ``float16`` parity rounding, which looks exactly like sloppiness a tidy-up would delete.

CPU-only, zero GPU, zero Modal spend. T-10-07-T (tampering / silent encode divergence) and
T-10-07-D (Qwen3-VL coexisting with the 61.7 GiB transformer) are discharged here.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_H3_ENCODE = REPO / "src" / "signet_trainer" / "prep" / "h3_encode.py"

#: Roots that must NEVER appear in this module's import closure. ``prep/h3_encode`` is the Modal-side
#: EXCEPTION tier: it MAY import these, but only function-local, so importing it to read constants
#: (this file, the CPU dry-run gate, CI on Windows) never needs a backend installed.
_BACKEND_ROOTS = frozenset({"modal", "diffusers", "transformers", "torchvision", "PIL"})

#: The only third-party root ``prep/h3_encode`` may pull at MODULE scope. Widening this is a
#: deliberate decision, not a drive-by: every entry becomes a hard dependency of the CPU gate.
_ALLOWED_MODULE_SCOPE_ROOTS = frozenset({"torch"})


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove ``# ...`` comments + triple-quoted strings so prose doesn't trip the scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    src = re.sub(r"#.*", "", src)
    return src


def _code() -> str:
    return _strip_comments_and_docstrings(_H3_ENCODE.read_text(encoding="utf-8"))


def _function_source(code: str, name: str) -> str:
    """Slice one top-level function's stripped source, so an assertion can be scoped to it."""
    match = re.search(rf"^def {re.escape(name)}\(", code, re.M)
    assert match, f"{name}() not found in prep/h3_encode.py"
    tail = re.search(r"^(?:def |class |@)", code[match.end() :], re.M)
    end = match.end() + tail.start() if tail else len(code)
    return code[match.start() : end]


def _ast_node(path: Path, name: str):
    """The ``ast`` node of a top-level function/class in an arbitrary module."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"no top-level {name!r} in {path.name}")


def _called_names(node: object) -> set[str]:
    """Every callee name reachable inside an AST node — ``f(...)`` and ``x.f(...)`` alike.

    ⚠ AST, not text. "Is this function CALLED here" is a structural question, and the
    comment-stripper cannot reach an f-string literal — so a message that merely NAMES
    ``reference_image_size`` would satisfy a substring scan while the call was gone (or vice versa).
    Three prior plans in this phase produced a false green exactly this way.
    """
    import ast

    names: set[str] = set()
    for inner in ast.walk(node):  # type: ignore[arg-type]
        if not isinstance(inner, ast.Call):
            continue
        func = inner.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


# ==================================================================================================
# Trap 1 — the text-encoder layer index
# ==================================================================================================


def test_text_encoder_layer_comes_from_the_imported_constant() -> None:
    """A FINAL-layer encode is silently wrong at the correct (B, L, 5120) shape.

    MiniMax-H3 conditions on a specific Qwen3-VL hidden state out of its 64; the last layer is
    post-norm and is not what the released weights were trained against. Nothing about the tensor
    reveals the mistake, so the index must come from ``models/h3_loader.py`` — the single source of
    every H3 architecture number — and never from a literal that can drift.
    """
    code = _code()
    assert "EXPECTED_H3_TEXT_ENCODER_LAYER" in code, (
        "the text-encoder layer index must be the IMPORTED constant: a final-layer encode is "
        "silently wrong at the correct (B, L, 5120) shape, with no error to catch it"
    )
    assert re.search(
        r"from\s+signet_trainer\.models\.h3_loader\s+import\s*\(", code
    ), "the index must be imported from models/h3_loader.py, the single source of H3 arch numbers"
    assert not re.search(r"hidden_states\s*\[\s*50\s*\]", code), (
        "no bare hidden_states[50] — the layer index is a re-derivable literal exactly where "
        "re-deriving it is fatal; import EXPECTED_H3_TEXT_ENCODER_LAYER instead"
    )
    assert not re.search(r"\[\s*50\s*\]", code), (
        "no literal 50 subscript anywhere: its equality with the DiT depth is a NUMERICAL "
        "COINCIDENCE, so a literal here invites the two being conflated"
    )


def test_a_short_hidden_state_tuple_is_a_hard_error() -> None:
    """Falling back to the last available layer would be the silent failure, dressed as robustness."""
    body = _function_source(_code(), "encode_h3_text_conditions")
    assert "output_hidden_states=True" in body, (
        "the encoder must be called with output_hidden_states=True — without it there is no layer "
        "to select and the pre-encode would silently condition on the final output"
    )
    assert "raise RuntimeError" in body and "EXPECTED_H3_TEXT_ENCODER_LAYER" in body, (
        "a hidden-state tuple too short to reach the constant must RAISE and NAME the constant; "
        "clamping to the last available layer is the exact silent-corruption this guards"
    )


# ==================================================================================================
# Trap 2 — the fixed encode seed
# ==================================================================================================


def test_the_posterior_seed_is_fixed_not_the_run_seed() -> None:
    """A run-seeded posterior is off-distribution on every sample, invisibly.

    The reference implementation draws the conditioning posterior under a seed that is deliberately
    independent of the request generator. Re-using the run seed produces valid latents drawn from
    the wrong point of the distribution — no shape error, no crash, just a worse adapter.
    """
    code = _code()
    assert re.search(r"H3_KEYFRAME_ENCODE_SEED\s*:\s*int\s*=\s*42", code), (
        "the keyframe encode seed must be a NAMED constant fixed at 42, independent of the run "
        "seed (encoders.py:102-136 / P10-0d section 5.4 trap 2)"
    )
    assert "manual_seed" in code, (
        "the fixed seed must actually drive a generator — a named constant nothing consumes is "
        "documentation, not a guard"
    )
    body = _function_source(code, "encode_video_latents")
    assert re.search(r"manual_seed\(\s*seed\s*\)", body), (
        "encode_video_latents must seed its posterior generator from the seed argument, which "
        "defaults to the fixed H3_KEYFRAME_ENCODE_SEED"
    )
    assert re.search(r"seed\s*:\s*int\s*=\s*H3_KEYFRAME_ENCODE_SEED", body), (
        "the seed parameter must DEFAULT to the fixed constant, so a caller that forgets it still "
        "gets parity rather than the run seed"
    )


# ==================================================================================================
# Trap 3 — ImageNet pixel normalization over a [0, 1] base
# ==================================================================================================


def test_pixel_normalization_is_imagenet_over_a_zero_to_one_base() -> None:
    """Most video VAEs in this stack are [-1, 1]; H3's is not, and the difference degrades silently."""
    code = _code()
    assert "div(255" in code, (
        "pixels must be divided by 255 FIRST (uint8 -> [0, 1]) before the mean/std — reordering "
        "these is the silent trap, not a style choice"
    )
    assert "(0.485, 0.456, 0.406)" in code, "the ImageNet mean must be present verbatim"
    assert "(0.229, 0.224, 0.225)" in code, "the ImageNet std must be present verbatim"
    assert not re.search(r"\*\s*2(?:\.0)?\s*-\s*1", code), (
        "no [-1, 1]-style `* 2 - 1` normalization: H3's video VAE takes ImageNet-normalized RGB "
        "over a [0, 1] base, and the wrong convention DEGRADES rather than crashes"
    )
    assert not re.search(r"127\.5", code), (
        "no 127.5 rescale — that is the [-1, 1] convention, which is not H3's"
    )

    body = _function_source(code, "imagenet_normalize")
    div_at = body.index("div(255")
    mean_at = body.index("H3_PIXEL_MEAN")
    assert div_at < mean_at, (
        "the div(255.0) must come BEFORE the mean/std subtraction — applying ImageNet statistics "
        "to a [0, 255] range is the reordering failure this check exists for"
    )


# ==================================================================================================
# Trap 4 — two posterior policies in one pipeline
# ==================================================================================================


def test_both_posterior_policies_are_present() -> None:
    """Reference soundtracks take the posterior MEAN; everything else is SAMPLED. Both, in one file."""
    code = _code()
    assert ".sample(" in code, (
        "images, video references and target video take the posterior .sample() — a .mode()-only "
        "pre-encode is off-distribution"
    )
    assert ".mode(" in code, (
        "reference SOUNDTRACKS take the posterior .mode() and are NEVER sampled — two different "
        "posterior policies live in one pipeline (encoders.py:544-548)"
    )

    audio = _function_source(code, "encode_h3_audio_latents")
    assert re.search(r"is_reference\s*:\s*bool", audio), (
        "the posterior policy must be selected by an EXPLICIT is_reference flag, not inferred"
    )
    assert ".mode()" in audio and ".sample(" in audio, (
        "encode_h3_audio_latents must implement BOTH policies: .mode() for a reference soundtrack, "
        ".sample() at the fixed seed for target audio"
    )
    assert re.search(r"return\s+None", audio), (
        "a clip with no audio stream must return None so the caller records an explicit absent "
        "marker — fabricating silence trains on zeros the model never saw (0 of 44 corpus clips "
        "carry audio, D-10-AUDIO)"
    )


# ==================================================================================================
# The fifth trap — the deliberate float16 parity rounding
# ==================================================================================================


def test_float16_rounding_precedes_the_per_channel_normalization() -> None:
    """~11 bits of every conditioning latent are a DELIBERATE parity step, not sloppiness.

    The released model's conditioning cannot be reproduced without it, and it happens BEFORE the
    per-channel latent normalization — moving it after changes the result. It looks exactly like
    something a tidy-up would delete, which is why it is pinned structurally.
    """
    body = _function_source(_code(), "encode_video_latents")
    assert "torch.float16" in body, (
        "the sampled latent must be rounded through float16: ~11 bits of every conditioning latent "
        "are a deliberate parity step without which the released model's conditioning cannot be "
        "reproduced"
    )
    fp16_at = body.index("torch.float16")
    norm_at = body.index("_broadcast_channel_vector(latents_mean")
    assert fp16_at < norm_at, (
        "the float16 round-trip must happen BEFORE the per-channel (latents - mean) / std — the "
        "order is load-bearing, not incidental"
    )


# ==================================================================================================
# D-10-DEF-9 — the VAE takes 5-D and nothing else, and the refusal says so BY NAME
# ==================================================================================================


def test_the_vae_encode_input_rank_is_asserted_by_name_before_the_call() -> None:
    """A rank slip must fail saying what it wanted, not 'got 4 and 5' from inside a dependency.

    ``AutoencoderKLMiniMaxH3._encode`` reads ``x.shape[2]`` as the frame count — on a 4-D
    ``[C, F, H, W]`` that index is the canvas HEIGHT — and pads with a five-argument
    ``Tensor.repeat`` that PREPENDS an axis. So the library's own error describes the padding tensor
    and says nothing about the axis it misread, 480 lines into a vendored file.
    """
    import ast

    called = _called_names(_ast_node(_H3_ENCODE, "encode_video_latents"))
    assert "assert_h3_vae_encode_input" in called, (
        "encode_video_latents must assert the rank it is about to hand the VAE, by name "
        "(prep/h3_vae_contract.assert_h3_vae_encode_input)"
    )
    assert "unsqueeze" in called, (
        "a 4-D [C, F, H, W] input must get the batch axis added HERE — both producers on this path "
        "are 4-D (_h3_read_video_rgb and _reference_pixels) and the VAE takes 5-D only"
    )

    body = _function_source(_code(), "encode_video_latents")
    assert body.index("assert_h3_vae_encode_input") < body.index("vae.encode"), (
        "the rank refusal must PRECEDE vae.encode, or the library's misleading error wins the race"
    )
    assert "H3_VAE_UNBATCHED_RANK" in body, (
        "the unbatched rank must be the DERIVED constant (H3_VAE_ENCODE_RANK - 1), never a literal "
        "4 that can be edited apart from the 5 it is one less than"
    )

    # The refusal must be UNCONDITIONAL — a guard inside an `if` is a guard some call skips, and
    # `if False and ...` keeps every substring a text scan looks for.
    node = _ast_node(_H3_ENCODE, "encode_video_latents")
    guarded = [
        stmt
        for stmt in ast.walk(node)
        if isinstance(stmt, (ast.If, ast.Try))
        and "assert_h3_vae_encode_input" in _called_names(stmt)
    ]
    assert not guarded, "the rank refusal must be an unconditional top-level statement"


def test_the_video_vae_is_reached_from_exactly_one_function() -> None:
    """One call site is what makes one guard sufficient — a second is a second unguarded rank."""
    import ast

    tree = ast.parse(_H3_ENCODE.read_text(encoding="utf-8"))
    callers = sorted(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "encode"
            and getattr(inner.func.value, "id", None) == "vae"
            for inner in ast.walk(node)
        )
    )
    assert callers == ["encode_video_latents"], (
        f"`vae.encode` is called from {callers}. It must be reachable from exactly ONE function — "
        f"the one that owns the batch axis and the rank refusal. A second call site is a second "
        f"opinion about what rank the VAE takes, which is D-10-DEF-9."
    )


def test_the_rank_promise_is_kept_in_BOTH_directions() -> None:
    """"Same rank as pixels" was advertised and false; the squeeze is what makes it true."""
    pytest.importorskip("torch")
    import torch

    from signet_trainer.prep.h3_encode import encode_video_latents
    from signet_trainer.prep.h3_vae_contract import H3VideoVaeContractStub

    stub = H3VideoVaeContractStub()
    mean, std = torch.zeros(24), torch.ones(24)
    assert encode_video_latents(stub, torch.zeros(3, 22, 32, 48), mean, std).dim() == 4
    assert encode_video_latents(stub, torch.zeros(1, 3, 22, 32, 48), mean, std).dim() == 5
    # The reference path: a batched single frame takes `_encode`'s num_frames == 1 short-circuit.
    assert tuple(
        encode_video_latents(stub, torch.zeros(3, 1, 32, 48), mean, std).shape
    ) == (24, 1, 2, 3)


# ==================================================================================================
# The output contract — the four h3_-prefixed precomputed sources
# ==================================================================================================


def test_the_four_output_dirs_match_the_registered_precomputed_sources() -> None:
    """The dir names are a COMMITTED convention (plan 10-04); a typo here silently drops samples.

    ``PrecomputedDataset`` pairs sources by relative path, so a source written under a name nothing
    reads does not raise — the sample is simply absent from the index.
    """
    pytest.importorskip("torch")
    from signet_trainer.data.precomputed import _VIDEO_LATENT_SOURCE_DIRS
    from signet_trainer.prep import h3_encode

    code = _code()
    written = {
        h3_encode.H3_VIDEO_LATENTS_DIR,
        h3_encode.H3_CONDITIONS_DIR,
        h3_encode.H3_REFERENCE_LATENTS_DIR,
        h3_encode.H3_AUDIO_LATENTS_DIR,
    }
    assert written == {
        "h3_latents",
        "h3_conditions",
        "h3_reference_latents",
        "h3_audio_latents",
    }, "the four h3_-prefixed source names are a committed convention (plan 10-04)"

    for name in sorted(written):
        assert f'"{name}"' in code, (
            f"{name} must appear as a literal source name in prep/h3_encode.py — the writer is "
            f"what makes the registration real"
        )

    # Derived, not hardcoded: exactly the h3_ sources the ALLOWLIST carries must be the ones the
    # encoder writes as video-VAE latent dicts. The allowlist entry is only CORRECT because of that
    # payload shape, so the two sets agreeing is the actual invariant.
    allowlisted_h3 = {n for n in _VIDEO_LATENT_SOURCE_DIRS if n.startswith("h3_")}
    video_latent_written = {h3_encode.H3_VIDEO_LATENTS_DIR, h3_encode.H3_REFERENCE_LATENTS_DIR}
    assert allowlisted_h3 == video_latent_written, (
        "the H3 sources written as {'latents': [C,F,H,W], ...} dicts must be exactly the H3 sources "
        "in _VIDEO_LATENT_SOURCE_DIRS — an allowlisted source without that shape raises inside "
        "_normalize_video_latents, and a video latent left OUT of it skips the normalization"
    )
    assert h3_encode.H3_CONDITIONS_DIR not in _VIDEO_LATENT_SOURCE_DIRS, (
        "h3_conditions holds TEXT embeddings with no 'latents' key — allowlisting it would raise"
    )
    assert h3_encode.H3_AUDIO_LATENTS_DIR not in _VIDEO_LATENT_SOURCE_DIRS, (
        "h3_audio_latents is the substring trap again: it CONTAINS 'latent' but carries AUDIO-VAE "
        "latents, which is exactly why the allowlist replaced the old substring test"
    )


def test_video_latent_payloads_are_asserted_before_they_are_written() -> None:
    """The 10-04 dict shape is a hard contract — writing a divergent one corrupts training silently."""
    code = _code()
    writer = _function_source(code, "write_h3_precomputed")
    assert "_assert_video_latent_payload" in writer, (
        "the two video-latent sources must be shape-checked before torch.save — the allowlist "
        "entry committed in plan 10-04 is only correct because of that payload shape"
    )
    guard = _function_source(code, "_assert_video_latent_payload")
    for key in ("latents", "num_frames", "height", "width"):
        assert f'"{key}"' in guard, f"the payload guard must require the committed '{key}' key"
    assert "EXPECTED_H3_IN_CHANNELS" in guard, (
        "the channel count must be checked against the IMPORTED arch constant — H3's video VAE is "
        "not LTX's, and a mis-channelled cache trains silently wrong"
    )


# ==================================================================================================
# The reference contract — exactly 2 slots, config-driven short edge, geometry imported
# ==================================================================================================


def test_exactly_two_references_per_sample_is_enforced() -> None:
    """The environment ref SUBSTITUTES for the last character slot — it is never appended.

    Every sample carries a fixed slot count (operator ruling). A sample with a different count was
    never priced by the packed-sequence VRAM budget, so it must be refused at the cheap step rather
    than OOM a metered container.
    """
    pytest.importorskip("torch")
    from signet_trainer.conditioning.h3_geometry import H3_PHASE10_REFERENCES_PER_SAMPLE
    from signet_trainer.prep.h3_encode import encode_h3_reference_latents

    signature = inspect.signature(encode_h3_reference_latents)
    assert (
        signature.parameters["references_per_sample"].default == H3_PHASE10_REFERENCES_PER_SAMPLE
    ), "the slot count must default to the geometry module's Phase-10 value, not a local literal"

    # The count check runs BEFORE anything is encoded, so this needs no VAE and costs nothing.
    with pytest.raises(ValueError, match="expected exactly"):
        encode_h3_reference_latents(object(), [], 896, None, None, descriptors=[])
    with pytest.raises(ValueError, match="never appended"):
        encode_h3_reference_latents(object(), [object()], 896, None, None, descriptors=[])


def test_reference_short_edge_is_required_never_a_hardcoded_default() -> None:
    """D-NOHARDCODE: the reference short edge is a CONFIG value (2048 spec, 896 for Phase 10)."""
    pytest.importorskip("torch")
    from signet_trainer.prep.h3_encode import encode_h3_reference_latents

    short_edge = inspect.signature(encode_h3_reference_latents).parameters["short_edge"]
    assert short_edge.default is inspect.Parameter.empty, (
        "short_edge must be REQUIRED: it is a config field, and baking 896 or 2048 in would make a "
        "later higher-fidelity campaign silently encode at the wrong resolution (VAE latents "
        "cannot be spatially downscaled after the fact — that campaign needs a re-encode)"
    )


# ==================================================================================================
# D-10-DEF-4 — the reference resize has exactly ONE site, and both phases consume its output
# ==================================================================================================


def test_the_reference_resize_has_exactly_one_site() -> None:
    """Two independent opinions about a reference's geometry is what D-10-DEF-4 WAS.

    PHASE A counted vision tokens off ``reference_image_size(...)`` while handing the processor the
    RAW file; PHASE B resized before the VAE. Reference ``A`` therefore gridded 1,014 tokens against
    the 1,176 the spans and the packed budget assumed. The fix is not "call it correctly in both
    places" — it is that there is only ONE place, and it returns the resized pixels and their count
    as one object.
    """
    import ast

    tree = ast.parse(_H3_ENCODE.read_text(encoding="utf-8"))
    callers = sorted(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and "reference_image_size" in _called_names(node)
    )
    assert callers == ["prepare_h3_reference_images"], (
        f"reference_image_size is called from {callers}. It must be reachable from exactly ONE "
        f"function — prepare_h3_reference_images, the single resize site. A second caller is a "
        f"second source of truth for what geometry a reference is encoded at, and the two drifting "
        f"apart IS D-10-DEF-4."
    )

    encode = _called_names(_ast_node(_H3_ENCODE, "encode_h3_reference_latents"))
    assert "prepare_h3_reference_images" in encode, (
        "encode_h3_reference_latents (PHASE B) must obtain its geometry from the shared resize site, "
        "not from its own call to reference_image_size"
    )

    prepare = _function_source(_code(), "prepare_h3_reference_images")
    assert "Resampling.LANCZOS" in prepare, (
        "the LANCZOS resample must live at the single site — a consumer that resizes again is the "
        "second source of truth this test exists to forbid"
    )
    pixels = _called_names(_ast_node(_H3_ENCODE, "_reference_pixels"))
    assert "resize" not in pixels, (
        "_reference_pixels must NOT resize: it converts an ALREADY-PREPARED image to a tensor. A "
        "resize here would silently re-decide the geometry PHASE A already presented"
    )


def test_the_processors_own_text_path_is_never_used() -> None:
    """``processor(text=..., images=...)`` re-expands vision blocks this presentation already expanded.

    ``build_h3_presentation`` emits ``<|vision_start|>`` + pad*N + ``<|vision_end|>`` because its
    half-open spans are only meaningful over the expanded stream. ``Qwen3VLProcessor.__call__``
    wants exactly ONE pad per image and expands it itself, so it exhausts its per-image replacement
    after 2 of 2,576 pads.
    """
    import ast

    tree = ast.parse(_H3_ENCODE.read_text(encoding="utf-8"))
    offenders = sorted(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "processor"
            for inner in ast.walk(node)
        )
    )
    assert not offenders, (
        f"{offenders} call the processor's own text path. It double-expands the vision blocks "
        f"(D-10-DEF-4 defect 1) — build the inputs from processor.image_processor + "
        f"processor.tokenizer instead, which is literally what __call__ merges anyway."
    )

    builder = _called_names(_ast_node(_H3_ENCODE, "build_h3_processor_inputs"))
    assert "image_processor" in builder and "tokenizer" in builder, (
        "the replacement must call BOTH halves — image_processor for pixel_values/image_grid_thw "
        "and tokenizer for input_ids — or it is not equivalent to what __call__ returns"
    )
    assert "_as_prepared_references" in builder, (
        "the builder must refuse a raw image by TYPE: a raw image satisfies every duck-typed thing "
        "a consumer asks of it, so nothing downstream would complain about the wrong geometry"
    )

    body = _function_source(_code(), "build_h3_processor_inputs")
    assert "image_grid_thw" in body and "merge_length" in body, (
        "the realized expansion must be derived from image_grid_thw and the processor's own "
        "merge_size — a hardcoded divisor would agree with the processor only by coincidence"
    )
    assert "add_special_tokens=False" in body, (
        "the presentation is documented as emitted verbatim with no chat template and no special "
        "tokens, and build_h3_presentation measures its spans the same way — an injected BOS would "
        "shift every vision span by one"
    )
    assert body.count("raise RuntimeError") >= 2, (
        "the grid-vs-declared check and the pad-count check must BOTH raise. The transformers "
        "version that happened to raise on the live failure is a coincidence, not the guard: a "
        "version whose loop merely stopped would have conditioned the model at a plausible-but-"
        "wrong shape with nothing to catch it"
    )

    encoder = _called_names(_ast_node(_H3_ENCODE, "encode_h3_text_conditions"))
    assert "build_h3_processor_inputs" in encoder, (
        "encode_h3_text_conditions must build its inputs through the checked builder"
    )


# ==================================================================================================
# D-10-DEF-7 — mm_token_type_ids, and the M-RoPE invariants nothing downstream raises about
# ==================================================================================================


def test_mm_token_type_ids_comes_from_the_processors_own_public_method() -> None:
    """A reimplemented modality mask re-derives a pad-id vocabulary the processor already owns.

    ``ProcessorMixin.create_mm_token_type_ids`` reads ``image_token_ids`` / ``video_token_ids`` /
    ``audio_token_ids`` off the MOUNTED processor. Hand-rolling the mask would agree with it only by
    coincidence — exactly the failure class that produced D-10-DEF-4 (a hardcoded ``merge_size``)
    and then D-10-DEF-7.
    """
    builder = _called_names(_ast_node(_H3_ENCODE, "_mm_token_type_ids"))
    assert "create_mm_token_type_ids" in builder, (
        "the M-RoPE modality mask must be built by the processor's own PUBLIC "
        "create_mm_token_type_ids, never reimplemented from the pad ids"
    )
    assert "tensor" in builder, (
        "create_mm_token_type_ids returns a LIST OF LISTS (it accepts ragged batches), so it needs "
        "a torch.tensor conversion before joining a return_tensors='pt' dict and flowing through "
        "_to_device"
    )
    source = _function_source(_code(), "_mm_token_type_ids")
    assert "raise RuntimeError" in source, (
        "a processor without create_mm_token_type_ids means a transformers version other than the "
        "pin is mounted — that must RAISE, not fall back to a hand-built mask"
    )


def test_mm_token_type_ids_is_emitted_unconditionally() -> None:
    """``__call__`` adds it when TEXT is given, NOT when images are — so both branches carry it.

    ``ProcessorMixin.__call__`` (processing_utils.py:696 at the pin) populates ``mm_token_type_ids``
    inside ``if getattr(self, "tokenizer", None) is not None and text is not None``. Gating our
    emission on "has references" would make the TEXT-ONLY branch's key set differ from
    ``__call__``'s — and the key-set parity gate would then need a special case, which is how a
    real gap gets hidden. The assignment must therefore sit at the function's top level and BEFORE
    the ``if not prepared: return`` early exit.
    """
    import ast

    node = _ast_node(_H3_ENCODE, "build_h3_processor_inputs")
    assignments = [
        stmt
        for stmt in node.body  # TOP LEVEL only — an assignment nested in an `if` is conditional
        if isinstance(stmt, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and getattr(target.value, "id", None) == "text_inputs"
            and getattr(target.slice, "id", None) == "H3_MM_TOKEN_TYPE_IDS_KEY"
            for target in stmt.targets
        )
    ]
    assert len(assignments) == 1, (
        "build_h3_processor_inputs must assign text_inputs[H3_MM_TOKEN_TYPE_IDS_KEY] exactly once, "
        "as a TOP-LEVEL statement. Inside an `if` it is conditional, and __call__ emits the key "
        "unconditionally whenever text is given (D-10-DEF-7 / GAP-3)"
    )

    early_returns = [
        index
        for index, stmt in enumerate(node.body)
        if isinstance(stmt, ast.If)
        and any(isinstance(inner, ast.Return) for inner in ast.walk(stmt))
    ]
    assign_at = node.body.index(assignments[0])
    assert early_returns and assign_at < min(early_returns), (
        "the mm_token_type_ids assignment must precede the `if not prepared: return text_inputs` "
        "early exit, or the text-only branch returns a dict __call__ would have carried the key in"
    )


def test_the_modality_mask_is_checked_per_run_and_in_order() -> None:
    """A PRESENT-but-wrong mask raises NOWHERE — the library only guards the MISSING case.

    ``compute_3d_position_ids`` raises when ``mm_token_type_ids`` is absent and multimodal grids are
    present; it says nothing about a mask whose contents are wrong, and that guard evaporates
    entirely on the ``inputs_embeds`` path. ``get_rope_index`` then pulls ONE grid per CONTIGUOUS run
    and builds that run's positions from the GRID, checking only the TOTAL length at the end — so a
    swapped or merged run cancels in the total and silently gives a reference the other one's
    lattice. Hence: per run, in order, not a count.
    """
    guard = _function_source(_code(), "_assert_mm_token_type_ids")
    assert "H3_MM_TYPE_IMAGE" in guard, (
        "the mask's modality values must come from the named constants, not from bare literals"
    )
    assert "H3_MM_TYPE_VIDEO" in guard and "H3_MM_TYPE_AUDIO" in guard, (
        "a video/audio-typed token with no matching grid makes get_rope_index call next() on a None "
        "iterator — that must be refused here, because a VIDEO reference is a D-10-DEF-4-sized "
        "re-derivation (pixel_values_videos, video_grid_thw, per-merged-frame-group grid splitting)"
    )
    assert guard.count("raise RuntimeError") >= 3, (
        "three distinct refusals are needed: the all-zeros mask (silent), the per-run mismatch "
        "(silent), and a non-image modality with no grid"
    )
    assert re.search(r"image_runs\s*!=\s*expected_runs", guard), (
        "the comparison must be per-run and ORDER-SENSITIVE (list != list). A count-only check "
        "passes on a swap of two differently-shaped references, because the per-run mismatches "
        "cancel in the total that get_rope_index actually verifies"
    )

    builder = _called_names(_ast_node(_H3_ENCODE, "build_h3_processor_inputs"))
    assert "_assert_mm_token_type_ids" in builder, (
        "the mask check must actually run on every encode, not merely exist"
    )
    assert "_mm_token_type_ids" in builder, "the mask must actually be built"


# ==================================================================================================
# D-10-DEF-7 — the CLASS closer: key-set parity against the REAL processor.__call__
# ==================================================================================================

_H3_PARITY = REPO / "src" / "signet_trainer" / "prep" / "h3_parity.py"


def test_the_parity_oracle_exists_and_is_the_only_sanctioned_call_site() -> None:
    """Every bypass of ``__call__`` inherits its outputs; the gaps cost one container each.

    ``processor(text=..., images=...)`` is forbidden in ``prep/h3_encode.py`` (it double-expands the
    vision blocks) and MANDATORY in ``prep/h3_parity.py``, where it is the ORACLE the key diff is
    taken against. A stub oracle cannot close this class — it can only reproduce what we already
    believe ``__call__`` does, so it can never reveal an output nobody has heard of.
    """
    import ast

    tree = ast.parse(_H3_PARITY.read_text(encoding="utf-8"))
    callers = sorted(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "processor"
            for inner in ast.walk(node)
        )
    )
    assert callers == ["h3_processor_output_key_diff"], (
        f"the REAL processor.__call__ must be invoked from exactly one function in prep/h3_parity — "
        f"h3_processor_output_key_diff, the oracle — got {callers}. Without that call there is no "
        f"oracle, and the key diff degenerates into comparing our builder against itself."
    )

    diff = _called_names(_ast_node(_H3_PARITY, "h3_processor_output_key_diff"))
    assert "build_h3_processor_inputs" in diff, (
        "the oracle must be diffed against the REAL builder, not against a restatement of it"
    )
    assert "build_h3_parity_probe" in diff, (
        "the probe must come from the shared builder so the two presentations differ only in the "
        "vision-token counts — a hand-written __call__ string would be a second source of truth "
        "for the label/sentinel layout"
    )

    parity_code = _strip_comments_and_docstrings(_H3_PARITY.read_text(encoding="utf-8"))
    assert re.search(r"H3_PROCESSOR_PARITY_ALLOWLIST\s*:\s*dict\[str,\s*str\]\s*=\s*\{\}", parity_code), (
        "the allowlist must be declared EMPTY. An entry is a documented exception, never a way to "
        "make a red diff green — an over-broad allowlist would have swallowed mm_token_type_ids"
    )
    assert "assert_h3_processor_output_parity" in parity_code, (
        "the raising wrapper the container preflight calls must exist"
    )


def test_prep_h3_parity_keeps_the_modal_side_exception_tier() -> None:
    """It may import transformers/PIL, but FUNCTION-LOCAL — the CPU gate imports it for constants."""
    raw = _H3_PARITY.read_text(encoding="utf-8")
    offenders = re.findall(
        r"^(?:import|from)\s+(modal|diffusers|transformers|torchvision|PIL)\b", raw, re.M
    )
    assert not offenders, (
        f"module-scope backend import(s) {sorted(set(offenders))} in prep/h3_parity.py — indent "
        f"them inside the function that needs them (Modal-side EXCEPTION tier), or reading "
        f"H3_PROCESSOR_PARITY_ALLOWLIST on a CPU box starts requiring transformers"
    )
    assert not re.search(r"(?:^|\s)(?:import|from)\s+modal\b", raw), (
        "prep/ is NOT the transport tier — the Modal SDK belongs only under src/signet_trainer/modal/"
    )


def test_the_vision_spans_are_verified_against_the_realized_token_stream() -> None:
    """A span that is off by one mis-tags a conditioning row at a perfectly valid shape."""
    guard = _function_source(_code(), "_assert_vision_spans_land_on_sentinels")
    assert "H3_VISION_START_TOKEN" in guard and "H3_VISION_END_TOKEN" in guard, (
        "the spans must be checked against the REALIZED sentinel ids, not trusted from the string "
        "they were measured on — train/h3_step.h3_token_tags consumes them to tag video rows"
    )
    assert "raise RuntimeError" in guard, "a mis-landed span must RAISE, not warn"
    assert "_assert_vision_spans_land_on_sentinels" in _called_names(
        _ast_node(_H3_ENCODE, "build_h3_processor_inputs")
    ), "the span check must actually run on every encode, not merely exist"


def test_a_prepared_reference_carries_its_pixels_and_its_count_together() -> None:
    """The type is the fix: no call form can pair one reference's pixels with another's row count."""
    pytest.importorskip("torch")
    from signet_trainer.prep.h3_encode import H3PreparedReference

    fields = set(H3PreparedReference.__dataclass_fields__)
    assert {"image", "vision_tokens", "encoded_wh", "source_wh", "short_edge"} <= fields, (
        f"H3PreparedReference must carry the pixels AND the count AND the geometry that produced "
        f"them; got {sorted(fields)}"
    )
    assert H3PreparedReference.__dataclass_params__.frozen, (
        "it must be frozen — a mutable count could be edited apart from the pixels it describes, "
        "which is the very divergence this type exists to prevent"
    )


def test_geometry_math_is_imported_never_re_derived() -> None:
    """``conditioning/h3_geometry.py`` owns the frame law and reference sizing — this module imports."""
    code = _code()
    assert re.search(r"from\s+signet_trainer\.conditioning\.h3_geometry\s+import", code), (
        "frame and reference-geometry math must come from h3_geometry, the single home for it"
    )
    for symbol in ("reference_image_size", "h3_latent_frames", "rows_of"):
        assert symbol in code, f"{symbol} must be imported from h3_geometry, not re-implemented"
    assert not re.search(r"\bround\(\s*\w+\s*/\s*32\s*\)", code), (
        "the round-to-32 snap belongs to h3_geometry._round_to_multiple; a local copy is how two "
        "sources of truth start disagreeing"
    )
    assert not re.search(r"17\s*\*\s*|%\s*17\b", code), (
        "the 17n+5 frame law must be expressed through h3_geometry's constants, never re-derived "
        "with local literals (LTX's (F-1)//8+1 does NOT transfer)"
    )


# ==================================================================================================
# The presentation — prompt LAST, verbatim, no chat template
# ==================================================================================================


def test_the_prompt_is_emitted_last_and_verbatim() -> None:
    """Labels and vision blocks come FIRST; the prompt is last, with no chat template."""
    code = _code()
    segments = _function_source(code, "_presentation_segments")
    for label in ("<Audio ", "<Picture ", "<Video "):
        assert label in segments, f"the per-modality label {label!r} must be emitted, one-based"
    assert "H3_VISION_START_TOKEN" in code and "H3_VISION_END_TOKEN" in code, (
        "a vision block is wrapped <|vision_start|> pad*N <|vision_end|>; the sentinels are part of "
        "the block and are tagged 0 (VIDEO) along with the pads"
    )
    prompt_at = segments.rindex("prompt")
    label_at = segments.index("<Picture ")
    assert label_at < prompt_at, (
        "the prompt must be appended LAST — the labels and vision blocks precede it, and the "
        "prompt is used verbatim with no chat template and no special tokens"
    )
    assert "apply_chat_template" not in code, (
        "no chat template: the prompt is presented verbatim (P10-0d section 3.4)"
    )


# ==================================================================================================
# T-10-07-D — the two-phase VRAM discipline
# ==================================================================================================


def test_two_phase_free_is_explicit_and_complete() -> None:
    """Qwen3-VL-32B and the 61.7 GiB transformer must NEVER coexist — a single A100 cannot hold both.

    The discipline is copied from the proven Gemma free (``modal/fns.py:445-464``): move to CPU,
    drop the reference, collect, empty the cache. It is MORE mandatory on H3 than it was on LTX.
    """
    body = _function_source(_code(), "free_text_encoder")
    assert "empty_cache" in body, (
        "torch.cuda.empty_cache() must run — without it the freed blocks stay in the caching "
        "allocator and the transformer load still OOMs"
    )
    assert re.search(r"\bdel\b", body), (
        "the helper must del its references: assign=True loader-owned CUDA storage is freed only "
        "by dropping the LAST reference, not by Module.to('cpu') (06-09 run-5 finding)"
    )
    assert '"cpu"' in body or "'cpu'" in body, (
        "the encoder must be moved off the GPU before it is dropped"
    )
    assert "gc.collect()" in body, "gc.collect() must run before empty_cache(), as in the Gemma free"


# ==================================================================================================
# Import confinement — derived in a subprocess, never hardcoded
# ==================================================================================================


def test_import_closure_pulls_no_backend_root() -> None:
    """Importing ``prep/h3_encode`` to read its constants must not require a backend to be installed.

    ``prep/h3_encode`` is the Modal-side EXCEPTION tier: it MAY import diffusers/transformers/
    torchvision/Pillow, but only FUNCTION-LOCAL. If one of those ever migrated to module scope, this
    whole test file (and the CPU dry-run gate) would stop being runnable on Windows/CI — and the
    failure would surface as a ModuleNotFoundError in a PAID container, after every local gate had
    already passed.

    The answer is DERIVED in a subprocess, never hardcoded: an in-process ``sys.modules`` diff
    returns nothing useful, because another test in the same session has almost certainly already
    imported the module.
    """
    probe = (
        "import json,sys\n"
        "before=set(sys.modules)\n"
        "import signet_trainer.prep.h3_encode\n"
        "new=set(sys.modules)-before\n"
        "roots=sorted({m.split('.')[0] for m in new}-set(sys.stdlib_module_names))\n"
        "roots=[r for r in roots if not r.startswith('_') and r!='signet_trainer']\n"
        "files=[getattr(sys.modules[n],'__file__',None) "
        "for n in new if n.split('.')[0]=='signet_trainer']\n"
        "print(json.dumps({'roots':roots,'files':[f for f in files if f]}))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONUTF8": "1"}
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True, cwd=REPO
    )
    snapshot = json.loads(out.stdout.strip().splitlines()[-1])

    leaked = _BACKEND_ROOTS & set(snapshot["roots"])
    assert not leaked, (
        f"importing prep/h3_encode pulled {sorted(leaked)} into sys.modules. Those imports must be "
        f"FUNCTION-LOCAL (Modal-side EXCEPTION tier) so reading the parity constants on a CPU box "
        f"needs no backend installed."
    )

    # Narrow the full closure to the roots signet_trainer imports at MODULE SCOPE; everything else
    # (numpy, tqdm, typing_extensions, ...) is a torch transitive that arrives for free.
    #
    # ⚠ DIVERGE from tests/test_backup_restore_fns.py:255-259, deliberately. That test anchors the
    # regex with ``^\s*`` because its question is "what must the IMAGE declare", and a FUNCTION-LOCAL
    # import needs declaring just as much as a module-scope one. This test asks the opposite
    # question — "what is pulled merely by IMPORTING the module" — and a function-local import is
    # precisely what the Modal-side EXCEPTION tier ALLOWS. So the regex is anchored at column 0:
    # with ``^\s*`` this file's own function-local ``import numpy as np`` would be reported as a
    # module-scope dependency, failing the tier rule it is actually obeying.
    sources = [Path(f).read_text(encoding="utf-8") for f in snapshot["files"]]
    direct = {
        root
        for root in snapshot["roots"]
        if any(
            re.search(rf"^(?:import|from)\s+{re.escape(root)}\b", s, re.MULTILINE) for s in sources
        )
    }
    assert direct, "closure derivation produced no direct roots — the probe is broken, not the module"
    assert direct <= _ALLOWED_MODULE_SCOPE_ROOTS, (
        f"prep/h3_encode's module-scope third-party imports are {sorted(direct)}, outside the "
        f"allowed {sorted(_ALLOWED_MODULE_SCOPE_ROOTS)}. Widening this is a deliberate decision: "
        f"every entry becomes a hard dependency of the CPU-only gate."
    )


def test_no_module_scope_backend_import_in_the_raw_source() -> None:
    """A belt-and-braces text check: the closure probe cannot see an import that is never reached."""
    raw = _H3_ENCODE.read_text(encoding="utf-8")
    offenders = re.findall(
        r"^(?:import|from)\s+(modal|diffusers|transformers|torchvision|PIL)\b", raw, re.M
    )
    assert not offenders, (
        f"module-scope backend import(s) {sorted(set(offenders))} in prep/h3_encode.py — indent "
        f"them inside the function that needs them (Modal-side EXCEPTION tier)"
    )
    # Built as a regex rather than a literal substring on purpose: this file must not itself
    # contain an `import`-`modal` line for a grep-based gate to trip over.
    assert not re.search(r"(?:^|\s)(?:import|from)\s+modal\b", raw), (
        "prep/ is NOT the transport tier — the Modal SDK belongs only under "
        "src/signet_trainer/modal/. The thin @app.function wrapper lives in modal/fns.py (10-10)."
    )


# ==================================================================================================
# Plan 10-10 — the `modal/fns.py` half of the wiring scan (RESERVED section, filled).
#
# 10-07 (above) owns `prep/h3_encode.py`; 10-10 owns the thin `@app.function` wrapper, the
# `run_h3_arch_gate` helper it calls first, the LOUD-FAILURE guard for an optional-but-requested
# output, and the commit-or-vanish `dataset_vol.commit()`. These scan `modal/fns.py` as TEXT and via
# `ast` — they never IMPORT it, because importing `modal/fns.py` pulls the Modal SDK and builds the
# app graph (which eagerly resolves every `Secret.from_name`). Zero Modal spend, same as above.
#
# The scans reuse `_strip_comments_and_docstrings` + `_function_source` from the top of this file:
# `modal/fns.py` DOCUMENTS every invariant it obeys in prose, so an un-stripped scan would pass on
# the comments alone while the code did the wrong thing.
# ==================================================================================================

_FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"


def _fns_code() -> str:
    return _strip_comments_and_docstrings(_FNS.read_text(encoding="utf-8"))


def _fns_node(name: str):
    """The `ast` node of a top-level function in ``modal/fns.py`` (raw source — decorators intact)."""
    import ast

    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no top-level function {name!r} in modal/fns.py")


def _raise_segments(name: str) -> list[str]:
    """Source of every ``raise`` statement inside a named ``modal/fns.py`` function."""
    import ast

    src = _FNS.read_text(encoding="utf-8")
    node = _fns_node(name)
    return [
        ast.get_source_segment(src, stmt) or ""
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Raise)
    ]


def _decorator_source(name: str) -> str:
    """Only the decorator block of a named ``modal/fns.py`` function, comments stripped."""
    src = _FNS.read_text(encoding="utf-8")
    node = _fns_node(name)
    if not node.decorator_list:
        return ""
    lines = src.splitlines()
    start = min(d.lineno for d in node.decorator_list) - 1
    return _strip_comments_and_docstrings("\n".join(lines[start : node.lineno - 1]))


# --------------------------------------------------------------------------------------------------
# The two stages exist, and the arch gate is NOT a second ungated entry point
# --------------------------------------------------------------------------------------------------


def test_the_gate_helper_and_the_preprocess_stage_both_exist() -> None:
    """Both halves must be present: the gate is useless unless a real dispatch calls it."""
    code = _fns_code()
    assert "def run_h3_arch_gate" in code, (
        "run_h3_arch_gate must exist — it is the abort-before-spend gate every H3 stage fires first"
    )
    assert "def h3_preprocess" in code, (
        "h3_preprocess must exist — H3-03's 'caches to a Volume' half is what makes the encode real"
    )


def test_no_bare_ungated_arch_smoke_entry_point_exists() -> None:
    """A bare ``modal run ...::fn`` arch check would skip the cost print AND the approval pause.

    ``load_ltxv_smoke`` is reachable that way and it is a documented, still-open defect (Phase 9
    AUDIT-SYNTHESIS finding #18) — not a precedent to extend. Phase 10 adds no second launch path,
    so no ``h3_*smoke`` stage may exist, and no operational-mode switch may change what a metered H3
    function does (a cost line is only truthful if the function it prices always does the same work).
    """
    code = _fns_code()
    offenders = re.findall(r"^def\s+(h3_\w*smoke\w*)\s*\(", code, re.M)
    assert not offenders, (
        f"H3 smoke entry point(s) {offenders} in modal/fns.py — that invocation style boots a "
        f"metered A100 with no cost print and no approval pause (Phase 9 AUDIT #18). The arch gate "
        f"belongs INSIDE an already-gated stage, as a plain helper."
    )
    assert "arch_smoke_only" not in code, (
        "no operational-mode switch: a flag that changes what a metered function DOES makes its "
        "printed cost estimate a lie"
    )


def test_run_h3_arch_gate_is_a_plain_helper_not_a_modal_stage() -> None:
    """No decorator at all — that is precisely what makes it unreachable as an entry point."""
    node = _fns_node("run_h3_arch_gate")
    assert not node.decorator_list, (
        "run_h3_arch_gate must carry NO decorator. A Modal stage decorator would make it reachable "
        "by `modal run -m signet_trainer.modal.fns::run_h3_arch_gate`, which bypasses the "
        "entrypoint's cost print and blocking approval (Phase 9 AUDIT #18). It runs inside the "
        "stage that calls it and inherits THAT stage's gate."
    )


# --------------------------------------------------------------------------------------------------
# What the gate actually asserts
# --------------------------------------------------------------------------------------------------


def test_the_arch_gate_asserts_the_constants_and_the_lora_target_count() -> None:
    """A gate that loads 61.7 GiB and checks nothing is spend with extra steps."""
    body = _function_source(_fns_code(), "run_h3_arch_gate")

    assert "assert_h3_arch" in body, (
        "the ten measured constants must be checked through models/h3_loader.assert_h3_arch — it "
        "reports EVERY offending field at once (enochiatron's gate caught 6 mismatches in one "
        "~$1.40 run precisely because it did not stop at the first)"
    )
    assert "check_lora_targets_regex" in body, (
        "the target survey must use the REGEX-aware probe: H3's target form is a bare-str regex, "
        "and the suffix probe structurally cannot express it"
    )
    assert "EXPECTED_H3_NUM_LAYERS" in body and "H3_LORA_LEAVES" in body, (
        "the expected main-stack count must be DERIVED from the two single-source constants "
        "(layers x leaves), never pinned as a second literal that can drift"
    )

    raises = _raise_segments("run_h3_arch_gate")
    target_raise = [r for r in raises if "main" in r and "collateral" in r]
    assert target_raise, "the gate must RAISE on a target-count mismatch, not merely print it"
    assert any("300" in r for r in target_raise), (
        "the target-mismatch raise must name the MEASURED figure (300 main-stack targets, P10-1 "
        "against real named_modules) — the condition is derived, but the message has to tell the "
        "operator which measurement it just contradicted"
    )
    assert any("token_refiner" in r for r in target_raise), (
        "the raise must name WHY collateral matters: it is the text-stream token refiner, i.e. ~4% "
        "of the adapter training the wrong thing"
    )

    assert "adaln_proj" in body, (
        "the adaln_proj.linear shape must be printed — it is what distinguishes the FULL diffusers "
        "projection from the ComfyUI pruned baked-bottleneck form (the wrong weight set)"
    )
    assert "import diffusers" in body, (
        "a cold-path import probe must run FIRST: an ImportError discovered AFTER the 61.7 GiB "
        "load wastes a metered launch"
    )
    assert body.index("import diffusers") < body.index("load_h3_transformer"), (
        "the cold-path probe must PRECEDE the model load, or it is not a cold-path probe"
    )


# --------------------------------------------------------------------------------------------------
# h3_preprocess — the decorator, the strict body order, and the guards
# --------------------------------------------------------------------------------------------------


def test_h3_preprocess_declares_the_h3_image_and_the_measured_memory_request() -> None:
    """A gpu= on the code-only default image boots an A100 and dies at ``import torch``."""
    decorator = _decorator_source("h3_preprocess")
    assert "image=h3_gpu_image" in decorator, (
        "h3_preprocess must run on the MiniMax-H3 family image (diffusers at the pinned "
        "DIFFUSERS_SHA), not the LTX gpu_image and not the code-only default"
    )
    assert re.search(r"\bmemory\s*=", decorator), (
        "memory= is the H3 addition no LTX GPU fn needs: the P10-1 probe REQUIRED it for the "
        "61.7 GiB load (scripts/_h3_probe_modal.py:281)"
    )
    assert 'gpu="A100-80GB"' in decorator, "single A100-80GB — never a fallback GPU list (house rule)"
    assert not re.search(r"\b(keep_warm|min_containers)\s*=", decorator), (
        "warm GPUs are opt-in on Modal and we never opt in (SC#4 / D-10)"
    )


def test_every_h3_preprocess_parameter_is_required() -> None:
    """A defaulted parameter is how a threading gap goes SILENT instead of raising at dispatch."""
    node = _fns_node("h3_preprocess")
    assert not node.args.defaults and not node.args.kw_defaults, (
        "every h3_preprocess parameter must be REQUIRED with no default: with no defaults a "
        "missing kwarg is a TypeError before a container is even allocated, whereas a default "
        "silently encodes at the wrong resolution / frame count / reference budget"
    )
    assert node.args.args, "h3_preprocess must actually take the threaded parameters"


def test_the_arch_gate_fires_first_and_unconditionally_inside_h3_preprocess() -> None:
    """No flag skips it and none stops after it — that is what makes the printed cost truthful."""
    import ast

    node = _fns_node("h3_preprocess")

    def _calls_gate(stmt: object) -> bool:
        return any(
            isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == "run_h3_arch_gate"
            for inner in ast.walk(stmt)  # type: ignore[arg-type]
        )

    branching = (ast.If, ast.For, ast.While, ast.Try, ast.With)
    assert not [s for s in node.body if isinstance(s, branching) and _calls_gate(s)], (
        "run_h3_arch_gate must NOT sit inside an if/for/while/try/with — a conditional gate is a "
        "gate that some dispatch skips, and abort-before-spend is delivered by EVERY dispatch"
    )
    assert [s for s in node.body if not isinstance(s, branching) and _calls_gate(s)], (
        "h3_preprocess must call run_h3_arch_gate as a top-level statement of its body"
    )

    body = _function_source(_fns_code(), "h3_preprocess")
    assert body.index("run_h3_arch_gate") < body.index("transformers"), (
        "the arch gate must fire BEFORE the Qwen3-VL load: it releases the 61.7 GiB transformer, "
        "and the two cannot coexist on one A100-80GB"
    )


def test_the_processor_parity_backstop_runs_before_the_qwen_load() -> None:
    """D-10-DEF-7 — the point of a preflight is that it is cheaper than what it precedes.

    ``AutoProcessor.from_pretrained`` mounts a tokenizer and an image-processor CONFIG — megabytes.
    ``AutoModel.from_pretrained`` mounts Qwen3-VL-32B. Running the key-set diff between the two
    turns the next gap in this class into a failure measured in seconds instead of a weights load
    plus the start of an encode, which is what attempts 1 and 2 each paid for.
    """
    import ast

    node = _fns_node("h3_preprocess")
    calls = _called_names(node)
    assert "assert_h3_processor_output_parity" in calls, (
        "h3_preprocess must run the processor-output parity backstop. CI green is not proof the "
        "IMAGE resolved to the pinned transformers, and this bypass inherits every output "
        "processor.__call__ makes"
    )

    # The CALL, not the import: `from ... import assert_h3_processor_output_parity` sits at the top
    # of the function body and would satisfy a bare name search while the call sat anywhere at all.
    body = _function_source(_fns_code(), "h3_preprocess")
    call_at = body.index("assert_h3_processor_output_parity(processor")
    assert call_at < body.index("AutoModel.from_pretrained"), (
        "the parity backstop must run BEFORE AutoModel.from_pretrained — after it, the gap it "
        "catches has already cost the Qwen3-VL-32B load it was supposed to precede"
    )
    assert body.index("AutoProcessor.from_pretrained") < call_at, (
        "it needs the mounted processor, so it comes after AutoProcessor.from_pretrained"
    )

    # Reachable, not merely present: a guard tucked inside `if False and ...` keeps every substring
    # a text scan looks for while never running.
    guarded = [
        stmt
        for stmt in ast.walk(node)
        if isinstance(stmt, (ast.If, ast.Try))
        and "assert_h3_processor_output_parity" in _called_names(stmt)
    ]
    assert not guarded, (
        "the parity backstop must be an UNCONDITIONAL statement — a preflight some dispatch skips "
        "is not a preflight, and a swallowed one is worse than none"
    )


def test_the_two_phase_free_precedes_the_video_vae_encode() -> None:
    """Qwen3-VL-32B and the 61.7 GiB transformer must never coexist; nor Qwen3-VL and the VAEs."""
    body = _function_source(_fns_code(), "h3_preprocess")
    assert body.index("free_text_encoder(") < body.index("encode_h3_video_latents("), (
        "free_text_encoder must be CALLED before the first video-VAE encode — the discipline is "
        "the proven Gemma free (fns.py:445-464), and it is MORE mandatory on H3 than it was on LTX"
    )
    # The helper can only reach its own locals; the caller owns the last reference (06-09 run-5).
    assert re.search(r"text_encoder\s*=\s*None", body), (
        "the CALLER must also null its own reference: assign=True loader-owned CUDA storage is "
        "freed only when the LAST reference goes away — Module.to('cpu') does not release it"
    )


def test_h3_preprocess_commits_the_dataset_volume_last() -> None:
    """Commit-or-vanish (Pitfall 3): without commit() the whole encode dies with the container."""
    body = _function_source(_fns_code(), "h3_preprocess")
    assert "dataset_vol.commit()" in body, (
        "h3_preprocess must commit the dataset Volume — an uncommitted encode leaves "
        "`modal volume ls signe-trainer-dataset` showing nothing, and the metered container is gone"
    )
    assert body.index("write_h3_precomputed(") < body.index("dataset_vol.commit()"), (
        "the commit must come AFTER the writes it is durably persisting"
    )


def test_a_requested_output_that_produced_zero_files_raises() -> None:
    """A swallowed exception once produced a 'successful' encode with an empty dir (a2v, 2026-07-15)."""
    raises = _raise_segments("h3_preprocess")

    reference_raise = [r for r in raises if "h3_reference_latents" in r]
    assert reference_raise, (
        "a requested reference encode that produced ZERO .pt files under h3_reference_latents/ "
        "must RAISE. An empty source dir does not fail downstream — PrecomputedDataset pairs by "
        "rel path, so it silently drops every sample from the index instead"
    )
    assert any("RuntimeError" in r for r in reference_raise), (
        "the loud failure must be a RuntimeError, matching the a2v guard it is copied from"
    )
    assert any("with_audio" in r for r in raises), (
        "the same guard must cover the audio leg: with_audio=True that yields no audio is a demux "
        "failure, not a success (0 of 44 corpus clips carry audio — D-10-AUDIO)"
    )


def test_exactly_the_configured_reference_slot_count_is_enforced() -> None:
    """Three references per sample was never priced by the packed-sequence budget."""
    code = _fns_code()
    resolver = _function_source(code, "_h3_resolve_references")
    assert "references_per_sample" in resolver, "the slot count must be threaded, never assumed"
    assert any(
        "references_per_sample" in segment
        for segment in _raise_segments("_h3_resolve_references")
    ), (
        "a sample that resolves to a different reference count must RAISE at the cheap step — an "
        "environment reference SUBSTITUTES for the last character slot and is never appended"
    )
    assert "reference_pair_seed" in resolver, (
        "the round-robin rotation must be seeded (D-10-PAIRSEED) so which refs a sample gets is "
        "reproducible and debuggable across runs"
    )

    body = _function_source(code, "h3_preprocess")
    assert "max_packed_rows" in body, (
        "the REALIZED packed-row count must be checked against the measured ceiling: the "
        "config-load check prices declared worst-case sizes, this one knows the true tokenized "
        "text length and the true encoded reference grids"
    )
    assert any("max_packed_rows" in r for r in _raise_segments("h3_preprocess")), (
        "over-budget must RAISE at the cheap step rather than caching a sample that OOMs training"
    )


def test_phase_a_and_phase_b_share_one_reference_geometry() -> None:
    """D-10-DEF-4 on the wrapper side: PHASE A owns no geometry of its own, and B refuses a drift.

    The wrapper is where the two phases meet, and it is where the divergence lived: PHASE A computed
    ``rows_of(reference_image_size(raw_w, raw_h, ...))`` and then handed the processor the RAW image.
    So the assertions are (a) it calls the shared preparer, (b) it has no local geometry call left,
    and (c) PHASE B raises rather than caching a disagreement.
    """
    calls = _called_names(_fns_node("h3_preprocess"))
    assert "prepare_h3_reference_images" in calls, (
        "h3_preprocess must prepare its references through the single shared resize site — that is "
        "what makes PHASE A's presented count and PHASE B's encoded rows the same integer"
    )
    assert "reference_image_size" not in calls, (
        "h3_preprocess must NOT compute reference geometry itself. Its own copy of that arithmetic "
        "is exactly how it ended up counting the 896-resized reference while presenting the raw one "
        "(162 rows per reference of silent disagreement)"
    )
    assert "presentation_refs_from_prepared" in calls, (
        "the presentation's vision_token_counts must come off the prepared references, never from a "
        "locally computed number"
    )

    body = _function_source(_fns_code(), "h3_preprocess")
    assert re.search(r"encode_h3_text_conditions\(\s*\n?\s*text_encoder,\s*processor,", body), (
        "PHASE A must still call encode_h3_text_conditions"
    )
    assert "vision_spans=spans" in body, (
        "the spans build_h3_presentation returned must be handed to the encode so they can be "
        "checked against the realized token stream"
    )
    assert "images=images" not in body, (
        "raw images must no longer travel into the text encode — the PREPARED references do, so the "
        "processor sees exactly the pixels whose row count the spans bill"
    )

    # ⚠ The refusal is checked as a REACHABLE branch, on the AST — not as a `raise` that merely
    # appears in the text. `if False and encoded_rows != ...:` keeps every substring a text scan
    # looks for while disabling the guard entirely, and that mutation is the one a hurried "just
    # silence it for a second" edit actually produces.
    import ast

    node = _fns_node("h3_preprocess")
    branches = [
        stmt
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.If)
        and {"encoded_rows", "vision_counts"} <= set(re.findall(r"\w+", ast.unparse(stmt.test)))
    ]
    assert branches, (
        "PHASE B must compare the latent rows it encoded against the vision counts PHASE A "
        "presented. The shared preparer makes them equal by construction; this comparison makes an "
        "inequality LOUD instead of caching a sample whose text conditioning and reference latents "
        "describe different images"
    )
    for branch in branches:
        test = ast.unparse(branch.test)
        assert "!=" in test, f"the cross-phase branch must fire on INEQUALITY, got `{test}`"
        assert not re.search(r"\b(False|None)\b", test), (
            f"the cross-phase branch test `{test}` is short-circuited by a constant — the guard is "
            f"present in the source and dead at runtime, which is worse than absent"
        )
        raises = [s for s in ast.walk(branch) if isinstance(s, ast.Raise)]
        assert raises, "the cross-phase disagreement must RAISE, not print"

    cross_check = [
        r for r in _raise_segments("h3_preprocess") if "PHASE A" in r and "PHASE B" in r
    ]
    assert cross_check and any(
        "latent_rows" in r or "vision_counts" in r or "vision token" in r for r in cross_check
    ), "the cross-phase refusal must NAME the two quantities it compared"


def test_phase_a_persists_its_vision_spans_and_a_prompt_only_state() -> None:
    """The spans were computed here and DROPPED, while the train side read a key nothing set.

    `h3_ref` read `batch.get("vision_spans", ())` and `modal/fns.py` promised in a comment that
    "the train side recomputes them" — no train-side code did. So every step ran with empty spans and
    `h3_token_tags` tagged >90% of the text stream TEXT instead of VIDEO. The prompt-only state is
    the sibling fix: a D-10-REFDROP step dropped the reference LATENTS while the cached text kept
    describing them, a regime that occurs nowhere at inference.
    """
    calls = _called_names(_fns_node("h3_preprocess"))
    assert "build_h3_text_payload" in calls, (
        "PHASE A must write the VERSIONED text payload — a bare tensor carries no spans and no "
        "prompt-only state, which is the (v1) format sitting on the Volume right now"
    )

    body = _function_source(_fns_code(), "h3_preprocess")
    assert "prompt_only_hidden_states=prompt_only_hidden" in body, (
        "the second, reference-free hidden state must be encoded and stored: a dropped step has to "
        "condition on a presentation that genuinely has no vision blocks"
    )
    assert body.count("build_h3_presentation(") >= 2, (
        "the prompt-only presentation must come from the SAME builder as the reference-bearing one "
        "— string surgery would be a second source of truth for the label/prompt layout"
    )
    assert "vision_spans=spans" in body and "vision_spans=prompt_only_spans" in body, (
        "both encodes must have their spans checked against the realized token stream"
    )


def test_the_absent_audio_marker_is_written_unconditionally() -> None:
    """A configured source whose dir never exists is a FileNotFoundError on the first train batch.

    ``with_audio=False`` (correct for this corpus) left ``audio_payload = None``,
    ``write_h3_precomputed`` skipped it, and ``h3_audio_latents/`` was never created — while
    ``H3RefStrategy.get_data_sources()`` declares all four and ``data/precomputed.py`` raises on a
    missing one. The committed four-source contract is what this restores.
    """
    import ast

    node = _fns_node("h3_preprocess")

    def _binds_absent_marker(stmt: object) -> bool:
        """`audio_payload = h3_absent_audio_payload(...)`, in either assign form."""
        targets = list(getattr(stmt, "targets", []))
        annotated = getattr(stmt, "target", None)
        if annotated is not None:
            targets.append(annotated)
        return (
            isinstance(stmt, (ast.Assign, ast.AnnAssign))
            and any(getattr(t, "id", None) == "audio_payload" for t in targets)
            and "h3_absent_audio_payload" in _called_names(stmt)
        )

    # The per-sample loop, and the binding must sit at ITS top level — not inside the
    # `if with_audio:` branch, where it only fires on the campaign that IS targeting audio.
    loops = [stmt for stmt in ast.walk(node) if isinstance(stmt, ast.For)]
    unconditional = [
        loop
        for loop in loops
        if any(_binds_absent_marker(stmt) for stmt in loop.body)  # TOP LEVEL of the loop body
    ]
    assert unconditional, (
        "the absent marker must be bound UNCONDITIONALLY for every sample, at the top level of the "
        "per-sample loop. Bound only inside `if with_audio:` it never fires on a with_audio=False "
        "campaign — h3_audio_latents/ is then never created, and data/precomputed.py raises "
        "FileNotFoundError on the first training batch for a source the strategy declares."
    )
    conditional_only = [
        stmt
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.If)
        and "with_audio" in ast.unparse(stmt.test)
        and any(_binds_absent_marker(inner) for inner in stmt.body)
    ]
    assert not (conditional_only and not unconditional)


def test_the_text_encoder_layer_is_cross_checked_against_the_arch_constant() -> None:
    """A final-layer encode is silently wrong at the correct (B, L, 5120) shape."""
    body = _function_source(_fns_code(), "h3_preprocess")
    assert "EXPECTED_H3_TEXT_ENCODER_LAYER" in body, (
        "the config's text_encoder_layer must be compared against the ASSERTED arch constant — the "
        "encode reads the constant, so a config naming a different layer would be silently ignored"
    )
    assert any(
        "EXPECTED_H3_TEXT_ENCODER_LAYER" in r for r in _raise_segments("h3_preprocess")
    ), "a disagreement must RAISE before any encode, not be logged and ignored"


# --------------------------------------------------------------------------------------------------
# The source -> output-key registry
# --------------------------------------------------------------------------------------------------


def _precomputed_source_output_keys() -> dict:
    """Parse ``_PRECOMPUTED_SOURCE_OUTPUT_KEYS`` out of ``modal/fns.py`` WITHOUT importing it."""
    import ast

    tree = ast.parse(_FNS.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "target", None)
        name = getattr(targets, "id", None) if targets is not None else None
        if name == "_PRECOMPUTED_SOURCE_OUTPUT_KEYS":
            return ast.literal_eval(node.value)  # type: ignore[attr-defined]
    raise AssertionError("_PRECOMPUTED_SOURCE_OUTPUT_KEYS not found in modal/fns.py")


def test_the_h3_forward_is_never_wrapped_in_autocast() -> None:
    """⛔ The checkpoint is MIXED-PRECISION and aligns its own dtypes; autocast overrides that.

    ``MiniMaxH3Transformer3DModel`` declares ``_keep_in_fp32_modules = ["proj_in", "audio_proj_in",
    "time_embedder", "proj_out", "audio_proj_out", "rope"]`` and the reference pipeline calls it
    BARE. An ``autocast("cuda", bf16)`` wrapper runs those pinned fp32 linears in bf16 — including
    the time embedder that drives every AdaLN row — silently and at the correct shape.
    """
    for stage in ("h3_train", "h3_sample"):
        body = _function_source(_fns_code(), stage)
        assert "autocast" not in body, (
            f"{stage} wraps the H3 forward in autocast. The transformer ships a mixed-precision "
            f"checkpoint and does its own per-module dtype alignment precisely so it runs without "
            f"one; overriding it demotes the fp32-pinned modules."
        )


def test_the_sample_path_tests_the_audio_output_for_None_not_for_truth() -> None:
    """The ref2va workflow's "audio" output is a TENSOR, so a bare truth test RAISES.

    ``if <multi-element tensor>`` is "Boolean value of Tensor with more than one element is
    ambiguous" — thrown on the FIRST ``_render``, i.e. after the base column's entire denoise loop
    has already been paid for.
    """
    body = _function_source(_fns_code(), "h3_sample")
    assert 'result.get("audio") is not None' in body, (
        'h3_sample must test `result.get("audio") is not None`. The workflow returns a '
        "(1, 2, num_samples) tensor, not a list, and a bare truth test on it raises."
    )
    assert not re.search(r'if\s+result\.get\("audio"\)\s*(?:else|\))', body), (
        "a bare truth test on the audio output survives in h3_sample"
    )


def test_position_ids_are_moved_onto_the_batch_device() -> None:
    """CPU-built geometry meets CUDA buffers in the rope — and CPU tests cannot see a device split.

    ``build_h3_ref2va_position_ids`` builds ``[seq, 3]`` float64 on the CPU because the coordinates
    are pure geometry; every OTHER structural tensor is built on ``video.device``.
    ``MiniMaxH3RotaryPosEmbed.forward`` does ``position_ids.to(torch.float32)`` — a DTYPE cast with
    no device move — then multiplies by a module buffer. The reference pipeline moves it in
    ``before_denoise``; the training path has no equivalent, so the split reaches the first forward,
    after the arch gate, the 61.7 GiB load and the LoRA inject.
    """
    step_src = _strip_comments_and_docstrings(
        (REPO / "src" / "signet_trainer" / "train" / "h3_step.py").read_text(encoding="utf-8")
    )
    body = _function_source(step_src, "build_h3_packed_batch")
    assert re.search(r"position_ids\s*=\s*position_ids\.to\(device\)", body), (
        "build_h3_packed_batch must move position_ids onto the batch device. This is closed BY "
        "CONSTRUCTION rather than by a behavioural test — a CPU-only suite structurally cannot "
        "observe a CPU/CUDA split."
    )
    assert body.index("position_ids = position_ids.to(device)") < body.index("h3_indices("), (
        "the move must happen alongside the other device-aware construction, before the tensors "
        "that are already built on `device`"
    )
    assert not re.search(r"position_ids\.to\(\s*torch\.float32", body), (
        "position_ids must stay float64 — the rope casts to fp32 itself, and the area-normalized "
        "h/w grids carry meaningful low bits until it does"
    )


def test_the_four_h3_sources_are_registered_and_the_five_ltx_ones_are_untouched() -> None:
    """The H3 leg is ADDITIVE: an H3 cache must never be pairable into an LTX run, or the reverse."""
    pytest.importorskip("torch")
    from signet_trainer.prep import h3_encode

    keys = _precomputed_source_output_keys()

    for source in (
        h3_encode.H3_VIDEO_LATENTS_DIR,
        h3_encode.H3_CONDITIONS_DIR,
        h3_encode.H3_REFERENCE_LATENTS_DIR,
        h3_encode.H3_AUDIO_LATENTS_DIR,
    ):
        assert source in keys, (
            f"{source} is written by prep/h3_encode.py but has no output-key registration — the "
            f"training batch would simply never carry it"
        )
        assert keys[source].startswith("h3_"), (
            f"{source} -> {keys[source]}: the output key must be h3_-prefixed for the same "
            f"non-collision reason the dir name is"
        )

    for source, output_key in (
        ("latents", "latent_conditions"),
        ("conditions", "text_conditions"),
        ("reference_latents", "ref_latent_conditions"),
        ("video_masks", "video_mask_conditions"),
        ("audio_latents", "audio_latent_conditions"),
    ):
        assert keys.get(source) == output_key, (
            f"the LTX entry {source!r} -> {output_key!r} must stay byte-identical: Phase 10 is "
            f"additive, and an LTX run reading a renamed key is a silent regression"
        )

    assert len(set(keys.values())) == len(keys), (
        "two sources mapping to the same output key would have one silently overwrite the other"
    )
