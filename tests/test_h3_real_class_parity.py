"""The CLASS closer for "verified against a self-authored surrogate" — REAL classes as the oracle.

Three metered containers found three integration defects, and a fourth read of the pinned tree found
eight more. The shape they share is not "a contract was reconstructed": it is

    **every seam was verified against a self-authored surrogate — a stub, a transcription, or a
    sibling's docstring — and never against the thing actually on the other side.**

D-10-DEF-9 is the cleanest instance. Plan 10-07 validated the encode on CPU with hand-rolled stub
VAEs whose ``encode`` did ``_, frames, height, width = pixels.shape`` — a 4-D unpack accepting
**precisely the rank the real ``AutoencoderKLMiniMaxH3`` refuses** — and mapping F pixel frames to F
latent frames, ignoring the 17→5 chunk law entirely. The suite was green for the whole life of the
defect, and the defect surfaced on a paid container.

So this file makes the REAL classes the oracle, on CPU, for free. Three mechanisms:

1. **Exact parity against the reference's OWN static builders.**
   ``MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence`` and
   ``MiniMaxH3SetTimestepsStep.build_row_timesteps`` are ``@staticmethod``s importable with no
   weights, no components and no GPU. One synthetic geometry goes through them AND through our
   ``h3_packing`` / ``h3_step`` transcriptions, and the results are diffed **bit-exactly**.
2. **Tiny-config REAL classes instead of stubs.** ``AutoencoderKLMiniMaxH3`` and
   ``MiniMaxH3Transformer3DModel`` both instantiate from config alone with random weights. Widths
   shrink; the geometry knobs keep their real defaults. The rank check, the frame law, the
   single-frame path and the ten-kwarg forward contract all live in the forward CODE, so a 0.5M-param
   instance answers the same questions the 22B one would — for cents of CPU rather than a container.
3. **One end-to-end CPU rehearsal across the writer→reader seam.** The real encode writes a real
   cache into a temp dir; the real ``PrecomputedDataset`` and the real ``H3RefStrategy`` read it back.
   Writer and reader were each tested against the same DOCUMENT; joining them is what makes an
   unkept intra-repo promise a red test instead of a container.

⛔ **Never skips.** A missing overlay, a wrong SHA, a crashed probe or unparseable output is a
FAILURE with the exact bootstrap command. The whole failure mode this file addresses is
manufactured confidence, and a skip is the purest form of it.

Zero GPU, zero Modal spend, zero model weights, zero network (after the one-time install).
"""

from __future__ import annotations

import ast
import functools
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_APP = REPO / "src" / "signet_trainer" / "modal" / "app.py"
_RUNNER = REPO / "tests" / "_h3_real_class_runner.py"
_OVERLAY = REPO / ".venv-h3parity"


def pinned_diffusers_sha() -> str:
    """``DIFFUSERS_SHA`` read off ``modal/app.py`` by AST — importing it would pull the Modal SDK."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets = getattr(node, "targets", None) or (
            [node.target] if getattr(node, "target", None) is not None else []
        )
        for target in targets:
            if getattr(target, "id", None) == "DIFFUSERS_SHA":
                return str(ast.literal_eval(node.value))  # type: ignore[attr-defined]
    raise AssertionError("DIFFUSERS_SHA not found in modal/app.py")


def _overlay_python() -> Path | None:
    for relative in ("Scripts/python.exe", "bin/python", "bin/python3"):
        candidate = _OVERLAY / relative
        if candidate.exists():
            return candidate
    return None


def _bootstrap_instructions() -> str:
    python = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return (
        f"    uv venv --system-site-packages .venv-h3parity\n"
        f"    uv pip install --python .venv-h3parity/{python} --no-deps \\\n"
        f"        'diffusers @ git+https://github.com/huggingface/diffusers"
        f"@{pinned_diffusers_sha()}'\n"
        f"  (--system-site-packages reuses this box's torch; --no-deps keeps the install from "
        f"moving the transformers pin the sibling parity gate asserts. The overlay is gitignored "
        f"by the existing .venv-*/ rule. Set SIGNET_H3_REALCLASS_PYTHON to point at a different "
        f"interpreter.)"
    )


@functools.lru_cache(maxsize=1)
def report() -> dict:
    """Run the probe ONCE per session under the pinned interpreter and return its JSON."""
    override = os.environ.get("SIGNET_H3_REALCLASS_PYTHON")
    interpreter = Path(override) if override else _overlay_python()
    if interpreter is None or not Path(interpreter).exists():
        return {
            "ok": False,
            "error": (
                "no interpreter with diffusers at the pinned SHA is available. This check is NOT "
                "skippable — the defect class it closes is 'a self-authored surrogate agreed with "
                "us and the real class did not', and a skip is that failure with extra steps.\n"
                f"Bootstrap the overlay once:\n{_bootstrap_instructions()}"
            ),
        }
    completed = subprocess.run(  # noqa: S603
        [str(interpreter), str(_RUNNER)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip().startswith("{")]
    if not lines:
        return {
            "ok": False,
            "error": (
                f"the real-class probe produced no JSON (exit {completed.returncode}). "
                f"stdout tail:\n{completed.stdout[-2000:]}\nstderr:\n{completed.stderr[-3000:]}"
            ),
        }
    try:
        parsed = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"unparseable probe output ({exc}): {lines[-1][-2000:]}"}
    return parsed


def _ok() -> dict:
    result = report()
    assert result.get("ok"), (
        f"the real-class probe did not complete.\n{result.get('error')}\n"
        f"{result.get('traceback', '')}"
    )
    return result


# ==================================================================================================
# Version identity — without it everything below is theatre
# ==================================================================================================


def test_the_probe_runs_at_the_pinned_diffusers_sha() -> None:
    """``diffusers`` reports ``0.40.0.dev0`` for every commit on main, so the VERSION proves nothing.

    The PEP 610 ``direct_url.json`` records the exact commit a git install resolved to, which is a
    strictly stronger identity than a declared version string — it is what the interpreter is
    actually running, not what something asked for.
    """
    result = _ok()
    pinned = pinned_diffusers_sha()
    assert result["diffusers_commit"] == pinned, (
        f"the probe ran diffusers at commit {result['diffusers_commit']}, not the "
        f"{pinned} that modal/app.py installs into the H3 image. An oracle taken from a different "
        f"commit is a different oracle.\n{_bootstrap_instructions()}"
    )
    assert result["vae"]["class"] == "AutoencoderKLMiniMaxH3"
    assert result["transformer"]["class"] == "MiniMaxH3Transformer3DModel"


# ==================================================================================================
# Mechanism 2 — the REAL VAE, and the stub that may not be more permissive than it
# ==================================================================================================


def test_every_constant_the_stub_derives_is_the_real_vaes_own_config() -> None:
    """The tiny instance shrinks WIDTHS only; every geometry knob is the class's own default."""
    from signet_trainer.conditioning.h3_geometry import H3_FRAMES_PER_CHUNK
    from signet_trainer.models.h3_loader import EXPECTED_H3_IN_CHANNELS
    from signet_trainer.prep.h3_vae_contract import (
        H3_VAE_PIXEL_CHANNELS,
        H3_VAE_SPATIAL_COMPRESSION,
        H3_VAE_TOKEN_DROP,
    )

    vae = _ok()["vae"]
    config = vae["config"]
    assert config["clip_length"] == H3_FRAMES_PER_CHUNK, (
        f"the real VAE chunks at {config['clip_length']} frames but h3_geometry says "
        f"{H3_FRAMES_PER_CHUNK}. The 17n+5 frame law is derived from that chunk length."
    )
    assert config["token_drop"] == H3_VAE_TOKEN_DROP, (
        f"the real VAE drops {config['token_drop']} trailing latent frames; the value DERIVED from "
        f"the frame law is {H3_VAE_TOKEN_DROP}. These agreeing is what makes the derivation legal."
    )
    assert config["latent_channels"] == EXPECTED_H3_IN_CHANNELS
    assert config["in_channels"] == H3_VAE_PIXEL_CHANNELS
    assert vae["spatial_compression_ratio"] == H3_VAE_SPATIAL_COMPRESSION, (
        f"the real VAE compresses space by {vae['spatial_compression_ratio']}; "
        f"H3_CANVAS_MULTIPLE // patch_w derives {H3_VAE_SPATIAL_COMPRESSION}. "
        f"tests/test_h3_reference_geometry_agreement.py once hardcoded 32 here — the CANVAS "
        f"multiple, which is this ratio TIMES the patch width."
    )


def test_the_real_vae_agrees_with_the_h3_frame_law() -> None:
    """Measured on the real forward, not read off a config field. 22 -> 7 is the campaign's."""
    from signet_trainer.conditioning.h3_geometry import h3_latent_frames

    frame_law = _ok()["vae"]["frame_law"]
    assert frame_law, "the frame-law probe produced nothing"
    for frames, latents in frame_law.items():
        pixel_frames = int(frames)
        if pixel_frames == 1:
            # The single-frame REFERENCE path: `_encode` short-circuits to `_encode_clip` because a
            # single frame has no temporal extent to chunk. `h3_latent_frames` has no opinion here —
            # its floor is 5 — which is exactly why a 4-D reference silently took the OTHER branch.
            assert latents == 1
            continue
        assert latents == h3_latent_frames(pixel_frames), (
            f"the real VAE turns {pixel_frames} pixel frames into {latents} latent frames; "
            f"h3_geometry.h3_latent_frames says {h3_latent_frames(pixel_frames)}"
        )
    assert frame_law["22"] == 7, (
        "22 frames must encode to 7 latent frames — pad to 34, two 17-frame chunks give 10, minus "
        "token_drop 3. That is the arithmetic the campaign's pixel_frames guard asserts."
    )


def test_the_stub_is_never_more_permissive_than_the_real_vae() -> None:
    """⛔ THE CLASS CLOSER. A stub that accepts what the real class refuses fails by probe NAME.

    This is the assertion plan 10-07 did not have. Its stub accepted 4-D — the rank
    ``AutoencoderKLMiniMaxH3._encode`` refuses — so the CPU suite was green while the encode was
    broken, and the disagreement surfaced on a metered container instead.
    """
    result = _ok()
    real, stub = result["vae"]["real_contract"], result["vae"]["stub_contract"]
    assert set(real) == set(stub), f"probe sets differ: {sorted(set(real) ^ set(stub))}"

    # Non-vacuity FIRST: a matrix that was all-accept or all-reject on both sides would satisfy
    # every comparison below while proving nothing at all.
    verdicts = {entry["verdict"] for entry in real.values()}
    assert verdicts == {"accept", "reject"}, (
        f"the probe matrix produced only {verdicts} from the REAL class — it must contain both, or "
        f"the stub/real comparison is vacuous"
    )

    more_permissive = sorted(
        name
        for name, entry in real.items()
        if entry["verdict"] == "reject" and stub[name]["verdict"] == "accept"
    )
    assert not more_permissive, (
        f"the stub ACCEPTS {more_permissive}, which the real AutoencoderKLMiniMaxH3 REFUSES. That "
        f"is D-10-DEF-9's mechanism exactly: a CPU suite green on inputs a metered container "
        f"rejects. Fix prep/h3_vae_contract.H3VideoVaeContractStub — never this assertion."
    )
    more_restrictive = sorted(
        name
        for name, entry in real.items()
        if entry["verdict"] == "accept" and stub[name]["verdict"] == "reject"
    )
    assert not more_restrictive, (
        f"the stub REFUSES {more_restrictive}, which the real class accepts — the opposite error, "
        f"and it makes a valid production shape fail a test instead of a container"
    )


def test_the_stub_reproduces_the_real_vaes_latent_shapes() -> None:
    """Agreeing on accept/reject is not enough — a stub may not invent a different latent grid."""
    result = _ok()
    real, stub = result["vae"]["real_contract"], result["vae"]["stub_contract"]
    accepted = {n for n, e in real.items() if e["verdict"] == "accept"}
    assert accepted, "no probe was accepted — the matrix is vacuous"
    for name in sorted(accepted):
        assert stub[name]["latent_shape"] == real[name]["latent_shape"], (
            f"probe {name}: the stub emits {stub[name]['latent_shape']} but the real VAE emits "
            f"{real[name]['latent_shape']}. A stub with the wrong shape algebra is how "
            f"test_h3_reference_geometry_agreement came to assert `latent_w * latent_h == rows` — "
            f"true only because its stub divided by 32 instead of the VAE's own 16."
        )


def test_the_stub_reproduces_the_real_vaes_AUTOGRAD_behaviour() -> None:
    """⛔ D-10-DEF-10's dimension. A stub that never requires grad cannot fail on a missing context.

    Agreeing on rank and on latent shape is not enough. ``.eval()`` does **not** freeze anything, so
    the real ``AutoencoderKLMiniMaxH3``'s output carries a graph whenever autograd is on — which is
    why an encode outside ``torch.no_grad()`` retained ~78 GiB of tiled activations. A stub that
    quietly returned a constant would be more permissive in exactly that dimension, and every CPU
    test asserting "the encode ran grad-free" would pass whether or not the context existed.
    """
    result = _ok()
    assert result["vae"]["eval_leaves_parameters_requiring_grad"], (
        "the real VAE came back with every parameter frozen after .eval(), which contradicts the "
        "premise of D-10-DEF-10. If diffusers ever starts freezing on eval() the fix is still "
        "correct, but this oracle would stop proving anything — re-read before relaxing."
    )
    _assert_autograd_parity(result["vae"], "AutoencoderKLMiniMaxH3")


def _assert_autograd_parity(section: dict, class_name: str) -> None:
    """Shared by the video and audio halves: real and stub must agree on the grad properties."""
    real, stub = section["real_contract"], section["stub_contract"]
    accepted = sorted(n for n, e in real.items() if e["verdict"] == "accept")
    assert accepted, "no probe was accepted — the autograd comparison is vacuous"

    # Non-vacuity: the real class MUST carry a graph under an enabled context, or the stub agreeing
    # with it proves nothing about the property that costs VRAM.
    assert any(real[name]["requires_grad_when_enabled"] for name in accepted), (
        f"no accepted probe made {class_name} produce a grad-bearing output with autograd enabled. "
        f"That is the property a missing no-grad context turns into an OOM; if it is absent the "
        f"comparison below is empty."
    )
    for name in accepted:
        assert stub[name]["requires_grad_when_enabled"] == real[name]["requires_grad_when_enabled"], (
            f"probe {name}: with autograd ENABLED the stub reports "
            f"requires_grad={stub[name]['requires_grad_when_enabled']} but {class_name} reports "
            f"{real[name]['requires_grad_when_enabled']}. A stub that cannot carry a graph cannot "
            f"fail when a caller forgets torch.no_grad() (D-10-DEF-10)."
        )
        for side, entry in (("the real class", real[name]), ("the stub", stub[name])):
            assert entry["requires_grad_under_no_grad"] is False, (
                f"probe {name}: {side} produced a grad-bearing output INSIDE torch.no_grad(). The "
                f"whole fix depends on that context doing what it says."
            )


def test_D_10_DEF_9_is_fixed_against_the_REAL_class_not_against_a_stub() -> None:
    """The 4-D producers now reach the real ``_encode`` correctly, and the rank promise holds."""
    fix = _ok()["vae"]["encode_video_latents"]
    assert fix["target_video"] == {"in": [3, 22, 32, 48], "out": [24, 7, 2, 3]}, (
        f"encode_video_latents on the real VAE gave {fix['target_video']}; a 4-D [3, 22, H, W] "
        f"clip must come back 4-D [24, 7, h, w] — 22 frames -> 7 latent frames, same rank in and out"
    )
    assert fix["reference_image"] == {"in": [3, 1, 32, 48], "out": [24, 1, 2, 3]}, (
        f"encode_video_latents on a reference gave {fix['reference_image']}; a single frame must "
        f"take `_encode`'s num_frames == 1 short-circuit and encode to exactly ONE latent frame. "
        f"The reference path was never reached on the failing run — the target video encodes first."
    )


# ==================================================================================================
# The AUDIO VAE — same doctrine, added for D-10-DEF-10, and it immediately found D-10-DEF-11
# ==================================================================================================


def test_every_constant_the_audio_stub_derives_is_the_real_audio_vaes_own_config() -> None:
    """``hop_length`` is DERIVED from the latent rate the packed budget already prices audio with.

    If the two ever disagree, it is not only the stub that is wrong — ``h3_audio_rows`` would be
    pricing a different number of rows than the VAE emits, and that lands in the packed sequence.
    """
    from signet_trainer.prep.h3_vae_contract import (
        H3_AUDIO_VAE_HOP_LENGTH,
        H3_AUDIO_VAE_LATENT_CHANNELS,
        H3_AUDIO_VAE_SAMPLING_RATE,
    )

    audio = _ok()["audio_vae"]
    assert audio["class"] == "AutoencoderKLMiniMaxH3Audio"
    assert audio["hop_length"] == H3_AUDIO_VAE_HOP_LENGTH, (
        f"the real audio VAE hops {audio['hop_length']} samples (math.prod(encoder_rates)); the "
        f"value derived from H3_AUDIO_LATENTS_PER_SECOND is {H3_AUDIO_VAE_HOP_LENGTH}. These "
        f"agreeing is what makes the derivation legal — and h3_audio_rows prices the packed "
        f"sequence off the same rate."
    )
    assert audio["config"]["sampling_rate"] == H3_AUDIO_VAE_SAMPLING_RATE
    assert audio["config"]["latent_channels"] == H3_AUDIO_VAE_LATENT_CHANNELS


def test_the_audio_stub_is_never_more_permissive_than_the_real_audio_vae() -> None:
    """The D-10-DEF-9 assertion, applied to the component that had no sanctioned stub at all."""
    audio = _ok()["audio_vae"]
    real, stub = audio["real_contract"], audio["stub_contract"]
    assert set(real) == set(stub), f"probe sets differ: {sorted(set(real) ^ set(stub))}"

    verdicts = {entry["verdict"] for entry in real.values()}
    assert verdicts == {"accept", "reject"}, (
        f"the audio probe matrix produced only {verdicts} from the REAL class — it must contain "
        f"both, or the stub/real comparison is vacuous"
    )
    disagreements = sorted(
        name for name in real if real[name]["verdict"] != stub[name]["verdict"]
    )
    assert not disagreements, (
        f"the audio stub and AutoencoderKLMiniMaxH3Audio disagree on {disagreements}. Fix "
        f"prep/h3_vae_contract.H3AudioVaeContractStub — never this assertion."
    )
    for name in sorted(n for n, e in real.items() if e["verdict"] == "accept"):
        assert stub[name]["latent_shape"] == real[name]["latent_shape"], (
            f"probe {name}: the stub emits {stub[name]['latent_shape']}, the real audio VAE emits "
            f"{real[name]['latent_shape']}"
        )


def test_the_audio_stub_reproduces_the_real_audio_vaes_AUTOGRAD_behaviour() -> None:
    """The audio path is UNEXERCISED, which is exactly why its guard must be proven, not assumed."""
    audio = _ok()["audio_vae"]
    assert audio["eval_leaves_parameters_requiring_grad"], (
        "the real audio VAE came back frozen after .eval() — see the video sibling's note"
    )
    _assert_autograd_parity(audio, "AutoencoderKLMiniMaxH3Audio")


def test_D_10_DEF_11_the_real_audio_vae_REFUSES_the_waveform_layout_we_produce() -> None:
    """⛔ EVIDENCE, not a fix. Logged in deferred-items.md; the fix is NOT one line.

    ``AutoencoderKLMiniMaxH3Audio.encode`` opens with
    ``if sample.ndim != 3 or sample.shape[1] != 1: raise ValueError`` and its own docstring says
    *"MiniMax-H3 passes the two stereo channels of a reference clip as ``batch_size = 2``"*. So the
    contract is ``[B, 1, samples]`` and a stereo clip is ``[2, 1, samples]``.

    ``modal/fns.py::_h3_read_audio_waveform`` documents and emits ``[1, 2, samples]``. It has never
    run (0 of 44 corpus clips carry a stream, and the campaign is ``with_audio: false``), which is
    why nothing has surfaced it — the same reason the missing no-grad context on
    ``encode_h3_audio_latents`` would have shipped unnoticed.

    This test pins the CONSTRAINT so the eventual fix is checked against the real class rather than
    against a reading of it. It deliberately asserts nothing about our producer: transposing the
    axes also changes what the cached payload means to ``H3RefStrategy``, and that layout is
    unverified — a "one-line fix" here would be a new self-authored surrogate.
    """
    real = _ok()["audio_vae"]["real_contract"]
    assert real["stereo_as_batch_2x1xN"]["verdict"] == "accept", (
        "stereo-as-batch is the layout the real class documents; if it is refused, the whole "
        "reading below is wrong and D-10-DEF-11 must be re-derived"
    )
    assert real["stereo_on_axis1_1x2xN"]["verdict"] == "reject", (
        "the real audio VAE ACCEPTED [1, 2, samples]. If that is now true, D-10-DEF-11 is not a "
        "defect and the deferred-items entry must be closed — but check WHY it changed first."
    )
    assert real["stereo_as_batch_2x1xN"]["latent_shape"][0] == 2, (
        "stereo must come back as two BATCH elements, which is what makes the [1, 2, N] layout a "
        "genuine defect rather than a stylistic difference"
    )


# ==================================================================================================
# Mechanism 1 — bit-exact parity against the reference's OWN static builders
# ==================================================================================================


def test_our_packed_layout_is_BIT_EXACT_against_build_ref2va_packed_sequence() -> None:
    """``position_ids`` is float64 and the area-normalized grids carry meaningful low bits.

    ``torch.equal``, not ``allclose``: the reference implementation keeps TWO different summation
    orders for the same rotary series precisely because they differ in the last ulp, so "close" is
    not the contract here.
    """
    layout = _ok()["layout"]
    # Non-vacuity: an all-zero position_ids on both sides would satisfy an equality diff.
    assert layout["position_ids_distinct_rows"] == layout["seq_len"], (
        f"only {layout['position_ids_distinct_rows']} of {layout['seq_len']} position rows are "
        f"distinct — the probe geometry is degenerate, so the exactness claim below is empty"
    )
    assert layout["position_ids_dtype"] == {"ours": "torch.float64", "reference": "torch.float64"}
    assert layout["position_ids_exact"], (
        f"our RoPE coordinates differ from build_ref2va_packed_sequence's by up to "
        f"{layout['position_ids_max_abs_diff']}. H3's RoPE is 3-axis float64 with h/w "
        f"area-normalized to a fixed 32-unit extent and t absolute from text_len — a drift here "
        f"trains the adapter against a geometry the latents do not have, silently."
    )
    assert layout["n_cond_video"]["ours"] == layout["n_cond_video"]["reference"]
    assert layout["n_cond_audio"]["ours"] == layout["n_cond_audio"]["reference"]


def test_our_token_tags_and_index_tensors_match_the_reference_exactly() -> None:
    """Modality tags are a CHECKPOINT contract (``timestep_indices * 3 + token_tags``)."""
    layout = _ok()["layout"]
    assert layout["token_tag_values"] == [0, 1, 2], (
        f"the reference produced tag values {layout['token_tag_values']} — all three modalities "
        f"must appear or the tag diff is vacuous"
    )
    assert layout["token_tags_exact"], (
        "our token tags differ from the reference's. Qwen vision rows live in the TEXT stream but "
        "modulate as VIDEO, and a wrong tag addresses the wrong AdaLN row with no shape error."
    )
    for key in ("video_indices_exact", "audio_indices_exact", "text_indices_exact"):
        assert layout[key], f"{key} is False — our packed ORDER differs from the reference's"


def test_our_row_timestep_plan_matches_build_row_timesteps_exactly() -> None:
    """One forward serves rows at different noise levels; the AdaLN table is indexed off this."""
    timesteps = _ok()["timesteps"]
    assert timesteps["distinct_timesteps"] >= 3, (
        f"only {timesteps['distinct_timesteps']} distinct timestep(s) — the probe must exercise "
        f"target video, target audio and the pinned conditioning rows, or the diff is vacuous"
    )
    assert timesteps["timestep_exact"], (
        f"our timesteps {timesteps['ours']} vs the reference's {timesteps['reference']}"
    )
    assert timesteps["timestep_indices_exact"], (
        "the per-row index into the timestep table differs — the transformer gathers its AdaLN "
        "modulation off exactly this tensor"
    )


# ==================================================================================================
# Mechanism 2 (cont.) — the REAL transformer eats our packed dict
# ==================================================================================================


def test_the_real_transformer_accepts_our_ten_kwarg_packed_batch() -> None:
    """Kwarg drift, index/rank/dtype contracts and output shapes, for cents of CPU."""
    from signet_trainer.train.h3_step import H3_PACKED_BATCH_KEYS

    transformer = _ok()["transformer"]
    assert transformer["kwargs"] == sorted(H3_PACKED_BATCH_KEYS), (
        f"the packed kwarg set is {transformer['kwargs']}, not H3_PACKED_BATCH_KEYS. The set is "
        f"CLOSED — an extra key is a TypeError inside a metered container."
    )
    assert transformer["video_out"][1] + transformer["audio_out"][1] < transformer["seq_len"], (
        "the model returned as many rows as the packed sequence — the outputs must be the VIDEO "
        "and AUDIO slices, not the whole thing"
    )
    assert transformer["loss_is_scalar"] and transformer["loss_is_finite"], (
        "compute_loss on a real forward must return a finite 0-dim scalar"
    )
    assert transformer["position_ids_dtype"] == "torch.float64", (
        "position_ids must stay float64 into the model — MiniMaxH3RotaryPosEmbed casts to fp32 "
        "itself, and casting early throws away the low bits the area-normalized grids carry"
    )


# ==================================================================================================
# Mechanism 3 — the writer -> reader rehearsal
# ==================================================================================================


def test_the_cache_the_encoder_writes_is_the_cache_the_strategy_reads() -> None:
    """⛔ Writer and reader were each tested against the same DOCUMENT, never against each other.

    That is how the vision spans came to be computed in PHASE A, written down nowhere, and read from
    a batch key nothing populated — with a comment in ``modal/fns.py`` asserting the train side
    recomputed them and no train-side code doing so. Joining the halves makes an unkept intra-repo
    promise a red test.
    """
    rehearsal = _ok()["writer_reader"]
    assert rehearsal["sources_present"] == [
        "h3_audio_latent_conditions",
        "h3_latent_conditions",
        "h3_ref_latent_conditions",
        "h3_text_conditions",
    ], (
        f"the round trip surfaced {rehearsal['sources_present']} — all FOUR sources must pair by "
        f"relative path, including the absent-audio marker (a missing source dir raises)"
    )
    assert rehearsal["target_latent_frames"] == 7, (
        "the real VAE encoded the 22-frame clip to "
        f"{rehearsal['target_latent_frames']} latent frames through the real writer"
    )
    assert rehearsal["vision_span_tags"] == [0, 0, 0], (
        f"the vision rows came back tagged {rehearsal['vision_span_tags']}, not VIDEO. The spans "
        f"PHASE A wrote must survive torch.save -> PrecomputedDataset -> H3RefStrategy and land on "
        f"the modality tags — that whole path was broken and nothing raised."
    )
    assert rehearsal["text_tag_outside_spans"] == 1, (
        "rows outside the vision spans must stay TEXT, or the assertion above passes on a batch "
        "where everything happens to be tagged video"
    )
    assert rehearsal["n_target_audio"] == 8, (
        f"an absent-audio sample carried {rehearsal['n_target_audio']} target-audio rows; the "
        f"priced count must be materialized as noise, not dropped to zero"
    )


# ==================================================================================================
# The stub REGISTRY — no test module may hand-roll a VAE stub again
# ==================================================================================================

_SANCTIONED_STUB = "H3VideoVaeContractStub"
#: Both sanctioned stubs. The audio one was added for D-10-DEF-10 (proving the runtime idiom needs a
#: component to drive) and is held to the identical standard — diffed against the real class above.
_SANCTIONED_STUBS = (_SANCTIONED_STUB, "H3AudioVaeContractStub")


def _vae_shaped_stub_classes(source: str) -> list[str]:
    """Class names defining an ``encode(self, x, return_dict=...)`` — the diffusers VAE duck.

    ``return_dict`` is the discriminator on purpose: a tokenizer also has ``encode``, and a scan
    that flagged those would be noise nobody would keep.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name != "encode":
                continue
            args = [a.arg for a in item.args.args + item.args.kwonlyargs]
            if "return_dict" in args:
                found.append(node.name)
    return found


def test_the_detector_itself_finds_a_hand_rolled_stub() -> None:
    """Non-vacuity for the scan below: prove it fires on the exact class that was retired."""
    retired = (
        "class _StubVideoVae:\n"
        "    def encode(self, pixels, return_dict=True):\n"
        "        _, frames, height, width = pixels.shape\n"
        "        return (None,)\n"
    )
    assert _vae_shaped_stub_classes(retired) == ["_StubVideoVae"], (
        "the detector does not recognize the very stub that let D-10-DEF-9 through, so the scan "
        "below would pass on a reintroduction of it"
    )
    assert _vae_shaped_stub_classes("class T:\n    def encode(self, text):\n        return []\n") == []


def test_no_test_module_hand_rolls_its_own_video_vae_stub() -> None:
    """A stub is only worth something when it cannot be more permissive than the real class.

    Nothing about writing one by hand enforces that, and two independently-written ones disagreed
    with the real class AND with each other (one divided space by 16, the other by 32). So there is
    exactly ONE, it lives in ``prep/h3_vae_contract``, and it is diffed against the real class above.
    """
    offenders: dict[str, list[str]] = {}
    for path in sorted((REPO / "tests").glob("*.py")):
        names = _vae_shaped_stub_classes(path.read_text(encoding="utf-8"))
        if names:
            offenders[path.name] = names
    assert not offenders, (
        f"hand-rolled VAE stub(s) {offenders}. Use one of prep.h3_vae_contract.{_SANCTIONED_STUBS} "
        f"— they call the same refusals production calls and are diffed probe-for-probe against the "
        f"real AutoencoderKLMiniMaxH3 / AutoencoderKLMiniMaxH3Audio, on rank, on latent shape AND "
        f"on autograd behaviour, so they cannot quietly become more permissive than the things they "
        f"stand in for."
    )


def test_the_sanctioned_stub_is_actually_the_one_in_use() -> None:
    """An empty scan proves nothing if nobody is stubbing a VAE at all any more."""
    users = sorted(
        path.name
        for path in (REPO / "tests").glob("*.py")
        if _SANCTIONED_STUB in path.read_text(encoding="utf-8") and path.name != Path(__file__).name
    )
    assert len(users) >= 2, (
        f"only {users} use {_SANCTIONED_STUB}. The registry scan above is only meaningful while "
        f"the sanctioned stub is genuinely carrying the CPU encode tests."
    )
