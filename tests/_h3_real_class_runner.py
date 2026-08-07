"""Drives the REAL MiniMax-H3 classes at the pinned diffusers SHA and prints ONE line of JSON.

⛔ **Executed by a DIFFERENT interpreter than the one running pytest** — the one that has
``diffusers`` installed from git at ``modal/app.py::DIFFUSERS_SHA``. See
``tests/test_h3_real_class_parity.py`` for how it is located and why a missing overlay is a FAILURE
rather than a skip. This file reports FACTS; the parent process makes every assertion.

Why real classes and not stubs
------------------------------
Three metered containers found three integration defects, and the shape they share is that **every
seam was verified against a self-authored surrogate — a stub, a transcription, or a sibling's
docstring — and never against the thing actually on the other side.** D-10-DEF-9 is the cleanest
example: plan 10-07 validated the encode on CPU with hand-rolled stub VAEs whose ``encode`` unpacked
``_, frames, height, width = pixels.shape``, i.e. they REQUIRED precisely the 4-D rank the real
``AutoencoderKLMiniMaxH3`` refuses. The suite was green for the entire life of the defect.

The answer is that all three heavy classes instantiate **from config alone, with random weights and
no network**:

* ``MiniMaxH3Scheduler()`` is free;
* ``AutoencoderKLMiniMaxH3`` shrinks its WIDTHS (``block_out_channels``, ``layers_per_block``, the
  decoder) while every GEOMETRY knob keeps its real default — ``latent_channels=24``,
  ``clip_length=17``, ``token_drop=3``, the spatial/temporal downsample factors. The rank check, the
  frame law, the single-frame short-circuit and the tiling stitch all live in the forward CODE, not
  in the weights, so a 0.5M-parameter instance answers the same questions the 22B one would;
* ``MiniMaxH3Transformer3DModel`` at ``hidden_size=64, num_layers=2`` accepts the real ten-kwarg
  packed dict, so index/rank/dtype contracts and output shapes are exercised for free.

And two of the reference's builders need no instance at all: ``build_ref2va_packed_sequence`` and
``build_row_timesteps`` are ``@staticmethod``s, so our ``h3_packing`` / ``h3_step`` transcriptions
can be diffed against them BIT-EXACTLY on one synthetic geometry.

Prints ``{"ok": true, ...}`` or ``{"ok": false, "error": ..., "traceback": ...}`` and exits 0 either
way — a non-zero exit is indistinguishable from "the interpreter could not start", and the parent
needs to tell those apart.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# --------------------------------------------------------------------------------------------------
# The tiny real VAE. WIDTHS shrink; GEOMETRY does not.
# --------------------------------------------------------------------------------------------------

#: Shrunk purely for construction cost. ``block_out_channels`` must keep its LENGTH (6) because the
#: spatial/temporal downsample factor tuples are parallel to it — shortening it would change the
#: compression ratios, i.e. the very algebra under test.
_TINY_VAE_KWARGS = {
    "block_out_channels": (32,) * 6,
    "layers_per_block": 1,
    "decoder_num_layers": 1,
    "decoder_num_attention_heads": 1,
    "decoder_attention_head_dim": 8,
}

#: Everything NOT overridden above, asserted by the parent against our own derived constants. These
#: are the knobs the encode geometry actually depends on, so leaving them at the class's own
#: defaults is what makes the tiny instance a valid oracle rather than a differently-wrong one.
_VAE_GEOMETRY_KEYS = (
    "latent_channels",
    "clip_length",
    "token_drop",
    "in_channels",
    "spatial_downsample_factors",
    "temporal_downsample_factors",
)

#: Frame counts probed against ``h3_geometry.h3_latent_frames``. Conforming ``17n+5`` counts for
#: n = 0, 1, 2 plus the single-frame reference path, which takes a DIFFERENT branch (``_encode_clip``
#: directly) and is the one a 4-D reference silently missed.
_FRAME_LAW_PROBES = (1, 5, 22, 39)

#: The tiny AUDIO VAE. Same rule: WIDTHS shrink, GEOMETRY does not.
#:
#: * ``encoder_rates`` / ``decoder_rates`` are NOT overridden — their product IS ``hop_length`` (800),
#:   which is the whole latent-rate algebra, and the class refuses a decoder that does not undo the
#:   encoder's hop.
#: * ``decoder_dim`` cannot go below ``2 ** len(decoder_rates)`` = 128: the BigVGAN decoder halves
#:   its channel count per upsample stage and a 0-channel conv is a construction error on the
#:   FIXTURE rather than a finding.
#: * ``latent_dim`` must stay a multiple of ``latent_channels`` (the class checks) and divisible by
#:   the attention head count.
_TINY_AUDIO_VAE_KWARGS = {
    "encoder_dim": 8,
    "latent_dim": 64,
    "num_attention_heads": 1,
    "decoder_dim": 128,
}

#: Asserted by the parent against the values ``prep/h3_vae_contract`` derives.
_AUDIO_VAE_GEOMETRY_KEYS = ("latent_channels", "sampling_rate", "encoder_rates", "decoder_rates")


def _tiny_vae():
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

    return AutoencoderKLMiniMaxH3(**_TINY_VAE_KWARGS).eval()


def _tiny_audio_vae():
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
        AutoencoderKLMiniMaxH3Audio,
    )

    return AutoencoderKLMiniMaxH3Audio(**_TINY_AUDIO_VAE_KWARGS).eval()


def _vae_section() -> dict:
    """The D-10-DEF-9 section: real-vs-stub contract, the frame law, and the fix itself."""
    import torch

    from signet_trainer.prep.h3_encode import encode_video_latents
    from signet_trainer.prep.h3_vae_contract import (
        H3VideoVaeContractStub,
        h3_vae_contract_report,
    )

    vae = _tiny_vae()
    config = {key: getattr(vae.config, key, None) for key in _VAE_GEOMETRY_KEYS}
    config = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in config.items()}

    # The frame LAW, measured on the real forward rather than read off a config field.
    frame_law = {}
    for frames in _FRAME_LAW_PROBES:
        with torch.no_grad():
            latents = vae.encode(
                torch.zeros(1, 3, frames, 32, 48), return_dict=False
            )[0].sample(generator=torch.Generator().manual_seed(0))
        frame_law[str(frames)] = int(latents.shape[2])

    # ⛔ THE FIX, against the REAL class. Both production producers are 4-D; `encode_video_latents`
    # is the one place that owns the batch axis. Driving the real function through the real VAE is
    # what makes "D-10-DEF-9 is fixed" a measurement instead of a claim.
    mean, std = torch.zeros(24), torch.ones(24)
    fix = {}
    for name, shape in (("target_video", (3, 22, 32, 48)), ("reference_image", (3, 1, 32, 48))):
        with torch.no_grad():
            out = encode_video_latents(vae, torch.zeros(*shape), mean, std)
        fix[name] = {"in": list(shape), "out": [int(v) for v in out.shape]}

    return {
        "class": type(vae).__name__,
        "config": config,
        "spatial_compression_ratio": int(vae.spatial_compression_ratio),
        "frame_law": frame_law,
        "real_contract": h3_vae_contract_report(vae),
        "stub_contract": h3_vae_contract_report(H3VideoVaeContractStub()),
        "encode_video_latents": fix,
        # ⛔ D-10-DEF-10. `.eval()` does NOT freeze anything, which is the fact the whole defect
        # rests on: a component loaded by `from_pretrained` and `.eval()`d still has every parameter
        # requiring grad, so an encode outside a no-grad context retains a graph. Measured on the
        # real class rather than reasoned about — it is the premise, and premises are what this file
        # exists to stop taking on trust.
        "eval_leaves_parameters_requiring_grad": bool(
            any(p.requires_grad for p in vae.parameters())
        ),
    }


def _audio_vae_section() -> dict:
    """The AUDIO half of the VAE contract — added for D-10-DEF-10, and it found D-10-DEF-11.

    There was no sanctioned audio stub because the audio path is measured unexercised. Proving
    ``encode_h3_audio_latents`` obeys the runtime idiom needs one, and a freehand stub would have
    been the D-10-DEF-9 mistake with a different component — so it is diffed here, same as its
    video sibling.
    """
    from signet_trainer.prep.h3_vae_contract import (
        H3AudioVaeContractStub,
        h3_audio_vae_contract_report,
    )

    vae = _tiny_audio_vae()
    config = {key: getattr(vae.config, key, None) for key in _AUDIO_VAE_GEOMETRY_KEYS}
    config = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in config.items()}
    return {
        "class": type(vae).__name__,
        "config": config,
        # `math.prod(encoder_rates)`, computed by the class itself. The parent diffs it against the
        # value derived from `H3_AUDIO_LATENTS_PER_SECOND`, which is what prices audio rows in the
        # packed budget — so a disagreement is a budget bug as well as a stub bug.
        "hop_length": int(vae.hop_length),
        "real_contract": h3_audio_vae_contract_report(vae),
        "stub_contract": h3_audio_vae_contract_report(H3AudioVaeContractStub()),
        "eval_leaves_parameters_requiring_grad": bool(
            any(p.requires_grad for p in vae.parameters())
        ),
    }


# --------------------------------------------------------------------------------------------------
# The layout + timestep parity, against the reference's OWN static builders.
# --------------------------------------------------------------------------------------------------

#: One synthetic ref2va geometry. Deliberately ASYMMETRIC everywhere it can be: two references at
#: DIFFERENT sizes (a swap would cancel in the totals both sides check), a non-square target, and a
#: latent frame count > 1 so the temporal grid is exercised rather than collapsing to a constant.
_PARITY = {
    "n_text": 11,
    "reference_grids": [(1, 8, 6), (1, 4, 10)],
    "target_grid": (3, 6, 8),
    "num_audio_latents": 5,
    "patch_size": (1, 2, 2),
    "audio_channels": 2,
    "vision_spans": [(2, 5), (6, 10)],
    "t_video": 0.42,
    "t_audio": 0.31,
    "t_visual_cond": 0.999,
    "t_audio_cond": 1.0,
}


def _layout_section() -> dict:
    """Bit-exact diff of our packed-sequence layout against ``build_ref2va_packed_sequence``."""
    import numpy as np
    import torch
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3Ref2VAPrepareLayoutStep,
    )

    from signet_trainer.conditioning.h3_packing import build_h3_ref2va_position_ids
    from signet_trainer.train.h3_step import (
        H3_AUDIO_TAG,
        H3_TEXT_TAG,
        H3_VIDEO_TAG,
        h3_indices,
        h3_token_tags,
    )

    p = _PARITY
    _, patch_h, patch_w = p["patch_size"]
    rows_of = lambda f, h, w: f * (h // patch_h) * (w // patch_w)  # noqa: E731
    n_cond_video = sum(rows_of(*g) for g in p["reference_grids"])
    n_target_video = rows_of(*p["target_grid"])
    n_target_audio = p["num_audio_latents"] * p["audio_channels"]
    seq = p["n_text"] + n_cond_video + n_target_audio + n_target_video

    # ⚠ The reference takes `text_token_tags` as an INPUT, so it is stated here independently of
    # `h3_token_tags`: all text, with the vision spans overridden to VIDEO. That is the *definition*
    # of what a vision span means, and handing the reference our own function's output instead would
    # make the token-tag half of this diff tautological.
    text_token_tags = torch.full((p["n_text"],), H3_TEXT_TAG, dtype=torch.long)
    for start, stop in p["vision_spans"]:
        text_token_tags[start:stop] = H3_VIDEO_TAG

    condition_latents = [torch.zeros(1, 24, f, h, w) for f, h, w in p["reference_grids"]]
    references = [
        MiniMaxH3ImageReference(image=np.zeros((8, 8, 3), dtype="uint8"))
        for _ in p["reference_grids"]
    ]
    (
        ref_position_ids,
        ref_token_tags,
        ref_video_indices,
        ref_audio_indices,
        ref_text_indices,
        ref_n_cond_video,
        ref_n_cond_audio,
    ) = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        text_token_tags=text_token_tags,
        references=references,
        condition_latents=condition_latents,
        audio_condition_latents=[],
        num_latent_frames=p["target_grid"][0],
        latent_height=p["target_grid"][1],
        latent_width=p["target_grid"][2],
        num_audio_latents=p["num_audio_latents"],
        patch_size=tuple(p["patch_size"]),
        audio_channels=p["audio_channels"],
        audio_tag=H3_AUDIO_TAG,
        video_tag=H3_VIDEO_TAG,
    )

    ours_position_ids = build_h3_ref2va_position_ids(
        p["n_text"],
        [tuple(g) for g in p["reference_grids"]],
        tuple(p["target_grid"]),
        p["num_audio_latents"],
        tuple(p["patch_size"]),
        audio_channels=p["audio_channels"],
    )
    our_video_indices, our_audio_indices, our_text_indices = h3_indices(
        p["n_text"], n_cond_video, n_target_audio, n_target_video
    )
    our_token_tags = h3_token_tags(
        seq, our_video_indices, our_audio_indices, [tuple(s) for s in p["vision_spans"]]
    )

    def _same(a, b) -> bool:
        return bool(a.shape == b.shape and torch.equal(a, b))

    return {
        "seq_len": seq,
        "n_cond_video": {"ours": n_cond_video, "reference": int(ref_n_cond_video)},
        "n_cond_audio": {"ours": 0, "reference": int(ref_n_cond_audio)},
        # BIT-exact on float64. `torch.equal` and not `allclose`: H3's RoPE is float64 throughout and
        # the area-normalized h/w grids carry meaningful low bits, so "close" is not the contract.
        "position_ids_exact": _same(ours_position_ids, ref_position_ids),
        "position_ids_dtype": {
            "ours": str(ours_position_ids.dtype),
            "reference": str(ref_position_ids.dtype),
        },
        "position_ids_max_abs_diff": float(
            (ours_position_ids - ref_position_ids).abs().max()
            if ours_position_ids.shape == ref_position_ids.shape
            else float("nan")
        ),
        "token_tags_exact": _same(our_token_tags, ref_token_tags),
        "video_indices_exact": _same(our_video_indices, ref_video_indices),
        "audio_indices_exact": _same(our_audio_indices, ref_audio_indices),
        "text_indices_exact": _same(our_text_indices, ref_text_indices),
        # Non-vacuity floor: an all-zero position_ids on BOTH sides would satisfy every diff above.
        "position_ids_distinct_rows": int(
            torch.unique(ref_position_ids, dim=0).shape[0]
        ),
        "token_tag_values": sorted({int(v) for v in ref_token_tags.tolist()}),
    }


def _timestep_section() -> dict:
    """Diff ``h3_row_timesteps`` against ``MiniMaxH3SetTimestepsStep.build_row_timesteps``."""
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

    from signet_trainer.train.h3_step import h3_indices, h3_row_timesteps

    p = _PARITY
    _, patch_h, patch_w = p["patch_size"]
    n_cond_video = sum(f * (h // patch_h) * (w // patch_w) for f, h, w in p["reference_grids"])
    n_target_video = (
        p["target_grid"][0]
        * (p["target_grid"][1] // patch_h)
        * (p["target_grid"][2] // patch_w)
    )
    n_target_audio = p["num_audio_latents"] * p["audio_channels"]
    seq = p["n_text"] + n_cond_video + n_target_audio + n_target_video

    video_indices, audio_indices, _ = h3_indices(
        p["n_text"], n_cond_video, n_target_audio, n_target_video
    )
    _, our_timestep, our_indices = h3_row_timesteps(
        seq,
        video_indices,
        audio_indices,
        n_cond_video,
        0,
        p["t_video"],
        p["t_audio"],
        t_visual_cond=p["t_visual_cond"],
        t_audio_cond=p["t_audio_cond"],
    )
    # ⚠ The reference takes `condition_video_timestep` as a value; OUR builder applies the
    # D-10-REFPIN `max(t_video, t_visual_cond)` itself, exactly as the reference PIPELINE does one
    # level up. So the oracle is handed the same post-max number — the diff is about the row PLAN,
    # not about who computes the max.
    ref_timestep, ref_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices=video_indices,
        audio_indices=audio_indices,
        num_condition_video_rows=n_cond_video,
        num_condition_audio_rows=0,
        num_text_tokens=p["n_text"],
        video_timestep=p["t_video"],
        audio_timestep=p["t_audio"],
        condition_video_timestep=max(p["t_video"], p["t_visual_cond"]),
        condition_audio_timestep=p["t_audio_cond"],
    )
    return {
        "timestep_exact": bool(torch.equal(our_timestep, ref_timestep)),
        "timestep_indices_exact": bool(torch.equal(our_indices, ref_indices)),
        "distinct_timesteps": int(ref_timestep.numel()),
        "ours": [float(v) for v in our_timestep.tolist()],
        "reference": [float(v) for v in ref_timestep.tolist()],
    }


def _transformer_section() -> dict:
    """Feed the REAL transformer our packed ten-kwarg dict. Tiny widths, real forward code."""
    import torch
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    from signet_trainer.conditioning.h3_ref import H3RefStrategy
    from signet_trainer.prep.h3_text_payload import build_h3_text_payload
    from signet_trainer.train.h3_step import h3_step_deps_from_model

    torch.manual_seed(0)
    # ⚠ `attention_head_dim` and `rope_freq_dim` keep their REAL defaults (128 / 16). H3's RoPE is
    # applied partially, over `3 axes * rope_freq_dim * 2` = 96 of the 128 head dims, so shrinking
    # the head dim below 96 makes the rotary broadcast fail on the FIXTURE rather than on a defect.
    # Everything that is genuinely a width — head COUNT, hidden size, depth, ffn, the embed dims —
    # shrinks.
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=1,
        num_layers=2,
        num_refiner_layers=1,
        hidden_size=64,
        ffn_dim=128,
        text_dim=48,
        time_embed_hidden_dim=64,
        time_embed_dim=32,
    ).eval()
    deps = h3_step_deps_from_model(model)

    p = _PARITY
    n_text, ref_rows, target_rows, audio_rows = 11, 6, 9, 4
    strategy = H3RefStrategy(
        references_per_sample=1,
        reference_dropout=0.0,
        deps=deps,
        position_ids_fn=lambda *, seq_len, **_: torch.zeros(seq_len, 3, dtype=torch.float64),
        target_audio_rows=audio_rows,
    )
    batch = {
        "segment_index": 0,
        "step": 0,
        "t_video": p["t_video"],
        "t_audio": p["t_audio"],
        "h3_latent_conditions": {"latents": torch.randn(1, target_rows, deps.patch_dim)},
        "h3_text_conditions": build_h3_text_payload(
            hidden_states=torch.randn(1, n_text, deps.text_dim),
            vision_spans=[(2, 5)],
            prompt_only_hidden_states=torch.randn(1, 4, deps.text_dim),
            has_references=True,
        ),
        "h3_ref_latent_conditions": {
            "references": [
                {
                    "path": "refs/a.png",
                    "kind": "character",
                    "subject_id": "A",
                    "width": 64,
                    "height": 64,
                    "latents": torch.randn(1, ref_rows, deps.patch_dim),
                }
            ]
        },
        "h3_audio_latent_conditions": {"latents": torch.randn(1, audio_rows, deps.audio_in_channels)},
    }
    inputs = strategy.prepare_training_inputs(batch)
    with torch.no_grad():
        video_out, audio_out = model(**inputs.packed.kwargs, return_dict=False)
    loss = strategy.compute_loss(inputs, (video_out, audio_out))
    return {
        "class": type(model).__name__,
        "kwargs": sorted(inputs.packed.kwargs),
        "video_out": [int(v) for v in video_out.shape],
        "audio_out": [int(v) for v in audio_out.shape],
        "seq_len": int(inputs.packed.seq_len),
        "loss_is_finite": bool(torch.isfinite(loss)),
        "loss_is_scalar": bool(loss.ndim == 0),
        "position_ids_dtype": str(inputs.packed.kwargs["position_ids"].dtype),
    }


def _writer_reader_section() -> dict:
    """The writer -> reader rehearsal: real VAE -> write_h3_precomputed -> PrecomputedDataset."""
    import tempfile

    import torch

    from signet_trainer.conditioning.h3_ref import H3RefStrategy
    from signet_trainer.data.precomputed import PrecomputedDataset
    from signet_trainer.models.h3_loader import EXPECTED_H3_PATCH_SIZE
    from signet_trainer.prep.h3_encode import (
        encode_h3_reference_latents,
        encode_h3_video_latents,
        h3_absent_audio_payload,
        prepare_h3_reference_images,
        write_h3_precomputed,
    )
    from signet_trainer.prep.h3_text_payload import build_h3_text_payload
    from signet_trainer.train.h3_step import H3StepDeps

    from PIL import Image as PILImage

    vae = _tiny_vae()
    mean, std = torch.zeros(24), torch.ones(24)
    frames, height, width = 22, 64, 96
    short_edge = 64

    video = encode_h3_video_latents(
        vae,
        torch.randint(0, 255, (3, frames, height, width), dtype=torch.uint8),
        mean,
        std,
        pixel_frames=frames,
    )
    prepared = prepare_h3_reference_images(
        [PILImage.new("RGB", (64, 96)), PILImage.new("RGB", (96, 64))], short_edge
    )
    references = encode_h3_reference_latents(
        vae,
        prepared,
        short_edge,
        mean,
        std,
        descriptors=[
            {"path": "refs/a.png", "kind": "character", "subject_id": "A"},
            {"path": "refs/e.png", "kind": "environment", "subject_id": "029"},
        ],
        references_per_sample=2,
    )
    text = build_h3_text_payload(
        hidden_states=torch.randn(9, 48),
        vision_spans=[(1, 4), (5, 8)],
        prompt_only_hidden_states=torch.randn(3, 48),
        has_references=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        write_h3_precomputed(
            tmp,
            "clips/sample_000.pt",
            video=video,
            text=text,
            references=references,
            audio=h3_absent_audio_payload("no stream"),
        )
        sources = {
            "h3_latents": "h3_latent_conditions",
            "h3_conditions": "h3_text_conditions",
            "h3_reference_latents": "h3_ref_latent_conditions",
            "h3_audio_latents": "h3_audio_latent_conditions",
        }
        dataset = PrecomputedDataset(tmp, data_sources=sources)
        sample = dict(dataset[0])

    patch_dim = 24 * EXPECTED_H3_PATCH_SIZE[0] * EXPECTED_H3_PATCH_SIZE[1] * EXPECTED_H3_PATCH_SIZE[2]
    deps = H3StepDeps(
        transformer=None,
        config=type("_Cfg", (), {"patch_size": EXPECTED_H3_PATCH_SIZE})(),
        patch_dim=patch_dim,
        audio_in_channels=32,
        text_dim=48,
    )
    strategy = H3RefStrategy(
        references_per_sample=2,
        reference_dropout=0.0,
        deps=deps,
        patch_size=tuple(EXPECTED_H3_PATCH_SIZE),
        position_ids_fn=lambda *, seq_len, **_: torch.zeros(seq_len, 3, dtype=torch.float64),
        target_audio_rows=8,
    )
    sample.update({"segment_index": 0, "step": 0, "t_video": 0.4, "t_audio": 0.3})
    inputs = strategy.prepare_training_inputs(sample)
    tags = inputs.packed.kwargs["token_tags"]
    return {
        "dataset_len": 1,
        "sources_present": sorted(sources.values()),
        "n_text": int(inputs.packed.n_text),
        "n_cond_video": int(inputs.packed.n_cond_video),
        "n_target_video": int(inputs.packed.n_target_video),
        "n_target_audio": int(inputs.packed.n_target_audio),
        # ⛔ The whole point of joining the two halves: the spans PHASE A wrote must survive
        # torch.save -> PrecomputedDataset -> the strategy and land on the tags.
        "vision_span_tags": [int(v) for v in tags[1:4].tolist()],
        "text_tag_outside_spans": int(tags[0]),
        "target_latent_frames": int(video["num_frames"]),
    }


def main() -> int:
    payload: dict = {"ok": False}
    try:
        import diffusers

        payload["diffusers_version"] = diffusers.__version__
        payload["python"] = sys.executable
        payload["diffusers_commit"] = _installed_commit("diffusers")
        payload["vae"] = _vae_section()
        payload["audio_vae"] = _audio_vae_section()
        payload["layout"] = _layout_section()
        payload["timesteps"] = _timestep_section()
        payload["transformer"] = _transformer_section()
        payload["writer_reader"] = _writer_reader_section()
        payload["ok"] = True
    except BaseException as exc:  # noqa: BLE001 — the parent turns ANY failure into a red test
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    print(json.dumps(payload))
    return 0


def _installed_commit(package: str) -> str | None:
    """The PEP 610 ``direct_url.json`` commit of a git-installed distribution.

    Stronger than a version string: ``diffusers`` reports ``0.40.0.dev0`` for every commit on main,
    so only the recorded ``vcs_info.commit_id`` can say whether the interpreter is running the SHA
    the image installs.
    """
    from importlib.metadata import distribution

    try:
        raw = distribution(package).read_text("direct_url.json")
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    return json.loads(raw).get("vcs_info", {}).get("commit_id")


if __name__ == "__main__":
    raise SystemExit(main())
