"""Family #3 (``qwen_edit``) PRE-ENCODE contract — the on-disk shape, the ordering, the cache key.

Written against ``prep/qwen_edit_encode.py``. Everything here runs on CPU with tiny synthetic
arrays, a stub VAE and a stub Qwen2.5-VL: **zero GPU, zero Modal spend, zero weights, zero network.**

What is pinned, and why each one is a defect somebody paid for
-------------------------------------------------------------
1. **The on-disk dict shape.** ``height`` / ``width`` are the **LATENT** grid dims (the
   ``h3_latents`` convention that makes ``data/precomputed.py::_normalize_video_latents`` correct),
   and the PIXEL size travels separately as ``source_wh`` / ``encoded_wh``. Writing pixels into
   those keys reshapes latents wrongly, silently.
2. **Control-slot ordering is CONFIG ORDER, and a missing file is an ERROR.** ai-toolkit appends
   only inside ``if os.path.exists(...)`` (``dataloader_mixins.py:984-985``), so a stem missing from
   ``dirB`` slides ``dirC`` into slot 1 — every later ``ctrl_img_N`` re-points and nothing raises.
   The slide is reproduced here as the thing we refuse, not as the thing we do.
3. **The text-embedding cache key includes the CONTROL IMAGES.** ``encode_control_in_text_embeddings
   = True`` (``qwen_image_edit_plus.py:66``) puts the control images' visual tokens INSIDE
   ``prompt_embeds``, so a caption-only key hands round 2 of a chain round 1's controls — while the
   VAE channel correctly carries round 2's. Both halves are proven: the collision a caption-only key
   produces, and the in-place-overwrite staleness ``control_cache_key_mode='path'`` reproduces
   deliberately.
4. **D-10-DEF-10 (autograd) and D-10-DEF-12 (placement)**, using the SHARED structural scanners from
   ``prep/h3_grad_contract`` rather than a second copy — including the mutation probe, so the
   mechanism cannot rot into a tautology.
5. **The partial-cache guard.** ``PrecomputedDataset`` keeps only samples present in EVERY source,
   so a sample missing one file is silently dropped rather than refused.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from signet_trainer.conditioning.qwen_edit import (
    QWEN_EDIT_DATA_SOURCES,
    QwenEditStrategy,
    resolve_control_slots,
)
from signet_trainer.conditioning.qwen_edit_geometry import (
    QWEN_EDIT_LATENT_CHANNELS,
    QWEN_EDIT_VAE_SCALE_FACTOR,
)
from signet_trainer.data.precomputed import _VIDEO_LATENT_SOURCE_DIRS
from signet_trainer.prep import qwen_edit_encode as qe
from signet_trainer.prep.h3_grad_contract import (
    H3_NO_GRAD_DECORATOR,
    H3_NO_GRAD_MARKER_ATTR,
    h3_encode_entry_points,
    model_forward_sites,
)

MODULE_SOURCE = Path(inspect.getfile(qe)).read_text(encoding="utf-8")

# Re-derived, never imported: 512 px / 8 (VAE) = 64 latent / 2 (pack) = 32; 32 x 32 = 1024 rows.
CANVAS = 512
CANVAS_AREA = CANVAS * CANVAS
LATENT_EDGE = CANVAS // 8
ROWS = (LATENT_EDGE // 2) ** 2
TEXT_DIM = 3584


# --------------------------------------------------------------------------------------------------
# Stubs. Each is NO MORE PERMISSIVE than the component it stands in for — the D-10-DEF-9 rule: a
# stub that accepts what the real class refuses manufactures confidence, which is how a 4-D tensor
# reached a metered container on the H3 leg.
# --------------------------------------------------------------------------------------------------


class _StubPosterior:
    """``DiagonalGaussianDistribution``'s two consumed methods, with the real signatures.

    ``sample`` keeps its ``generator`` keyword and RECORDS the seed, because the fixed-seed draw
    (``QWEN_EDIT_ENCODE_SEED``, independent of the run seed) is otherwise an unexercised line.
    """

    def __init__(self, moments: torch.Tensor, log: list[tuple[str, int | None]]) -> None:
        self._moments = moments
        self._log = log

    def sample(self, generator: Any = None) -> torch.Tensor:
        self._log.append(("sample", None if generator is None else generator.initial_seed()))
        return self._moments

    def mode(self) -> torch.Tensor:
        self._log.append(("mode", None))
        return self._moments


class _StubEncoderOutput:
    def __init__(self, latent_dist: _StubPosterior) -> None:
        self.latent_dist = latent_dist


class _QwenVaeStub:
    """``AutoencoderKLQwenImage``'s encode surface, refusing exactly what the real class refuses.

    * calls ``assert_qwen_edit_vae_encode_input`` — the SAME refusal production calls — so a 4-D
      tensor fails here exactly where ``_encode``'s five-name unpack fails
      (``autoencoder_kl_qwenimage.py:783``);
    * calls ``assert_qwen_edit_encode_device``, so a foreign tensor is refused rather than silently
      accepted (D-10-DEF-12);
    * holds a real ``nn.Parameter`` with ``requires_grad=True``, so its output carries a ``grad_fn``
      under an enabled autograd context — which is what makes a missing ``no_grad`` observable on
      CPU rather than on a metered A100 (D-10-DEF-10);
    * declares the 16-channel ``latents_mean`` / ``latents_std`` vectors the real config ships
      (``autoencoder_kl_qwenimage.py:687-688``) — this VAE has no ``scaling_factor``.
    """

    class _Config:
        latents_mean = [0.0] * QWEN_EDIT_LATENT_CHANNELS
        latents_std = [1.0] * QWEN_EDIT_LATENT_CHANNELS
        z_dim = QWEN_EDIT_LATENT_CHANNELS

    def __init__(self, *, fill: float = 0.25, device: Any = None) -> None:
        self.fill = float(fill)
        self.config = self._Config()
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float32, device=device))
        self.grad_enabled_calls: list[bool] = []
        self.posterior_calls: list[tuple[str, int | None]] = []
        self.encoded_devices: list[torch.device] = []

    def parameters(self):  # noqa: ANN201 — nn.Module-shaped duck type
        return iter([self.scale])

    def encode(self, pixels: Any) -> _StubEncoderOutput:
        self.grad_enabled_calls.append(bool(torch.is_grad_enabled()))
        qe.assert_qwen_edit_vae_encode_input(pixels, what="_QwenVaeStub.encode")
        qe.assert_qwen_edit_encode_device(pixels, self, what="_QwenVaeStub.encode")
        self.encoded_devices.append(pixels.device)
        batch, channels, frames, height, width = (int(v) for v in pixels.shape)
        if channels != 3:
            raise ValueError(
                f"[stub] the Qwen VAE takes 3-channel RGB; got {channels}. The real class fails in "
                f"its first QwenImageCausalConv3d, so the stub refuses rather than inventing a "
                f"latent for pixels the VAE could not have encoded."
            )
        moments = (
            torch.full(
                (
                    batch,
                    QWEN_EDIT_LATENT_CHANNELS,
                    frames,
                    height // QWEN_EDIT_VAE_SCALE_FACTOR,
                    width // QWEN_EDIT_VAE_SCALE_FACTOR,
                ),
                self.fill,
                device=pixels.device,
            )
            * self.scale
        )
        return _StubEncoderOutput(_StubPosterior(moments, self.posterior_calls))


class _ProcessorStub:
    """``Qwen2_5_VLProcessor.__call__``'s output keys, including the VISION half when images arrive.

    ``pixel_values`` / ``image_grid_thw`` are emitted ONLY when images are supplied, which is what
    lets the "a text-only encoder is in the slot" refusal be tested by simply turning them off.
    """

    def __init__(self, *, tokens: int = 96, emit_vision: bool = True) -> None:
        self.tokens = int(tokens)
        self.emit_vision = bool(emit_vision)
        self.texts: list[str] = []
        self.image_counts: list[int] = []

    def __call__(self, *, text, images=None, padding=True, return_tensors="pt") -> dict[str, Any]:  # noqa: ARG002
        self.texts.append(text[0])
        self.image_counts.append(0 if images is None else len(images))
        out: dict[str, Any] = {
            "input_ids": torch.arange(self.tokens).unsqueeze(0),
            "attention_mask": torch.ones(1, self.tokens, dtype=torch.int64),
        }
        if images and self.emit_vision:
            out["pixel_values"] = torch.zeros(len(images), 4, 8)
            out["image_grid_thw"] = torch.tensor([[1, 2, 2]] * len(images))
        return out


class _TextEncoderOutput:
    def __init__(self, hidden_states) -> None:  # noqa: ANN001
        self.hidden_states = hidden_states


class _TextEncoderStub:
    """Qwen2.5-VL's forward surface. Records grad mode and the kwargs it was actually handed."""

    def __init__(self, *, tokens: int = 96, layers: int = 4) -> None:
        self.tokens = int(tokens)
        self.layers = int(layers)
        self.scale = torch.nn.Parameter(torch.ones((), dtype=torch.float32))
        self.grad_enabled_calls: list[bool] = []
        self.seen_kwargs: list[set[str]] = []

    def parameters(self):  # noqa: ANN201
        return iter([self.scale])

    def __call__(self, **kwargs: Any) -> _TextEncoderOutput:
        self.grad_enabled_calls.append(bool(torch.is_grad_enabled()))
        self.seen_kwargs.append(set(kwargs))
        if not kwargs.get("output_hidden_states"):
            return _TextEncoderOutput(None)
        # The LAST entry is the one the edit pipeline takes (:270). Each layer is distinguishable so
        # a test can prove which was selected.
        states = tuple(
            torch.full((1, self.tokens, TEXT_DIM), float(i)) * self.scale
            for i in range(self.layers)
        )
        return _TextEncoderOutput(states)


def _image(width: int = CANVAS, height: int = CANVAS, colour=(10, 20, 30)):  # noqa: ANN001, ANN202
    from PIL import Image

    return Image.new("RGB", (width, height), colour)


def _prepared(width: int = CANVAS, height: int = CANVAS, **kw: Any) -> qe.QwenEditPreparedImage:
    return qe.prepare_qwen_edit_image(_image(width, height), CANVAS_AREA, **kw)


def _stats(vae: _QwenVaeStub):  # noqa: ANN202
    return qe.qwen_edit_vae_latent_stats(vae)


# --------------------------------------------------------------------------------------------------
# 1. The source-name contract: the writer and the reader must name the same three dirs.
# --------------------------------------------------------------------------------------------------


def test_writer_and_reader_declare_the_same_three_sources() -> None:
    assert set(qe.QWEN_EDIT_PRECOMPUTED_DIRS) == set(QWEN_EDIT_DATA_SOURCES)
    assert set(QwenEditStrategy().get_data_sources()) == set(qe.QWEN_EDIT_PRECOMPUTED_DIRS)


def test_conditions_source_is_not_allowlisted_as_a_latent_source() -> None:
    # The h3_conditions trap: text embeddings have no "latents" key, so routing them through
    # _normalize_video_latents raises. The two LATENT sources are also (still) absent — the open
    # wiring gap this module announces rather than silently assuming.
    assert qe.QWEN_EDIT_CONDITIONS_DIR not in _VIDEO_LATENT_SOURCE_DIRS
    assert qe.QWEN_EDIT_LATENTS_DIR not in _VIDEO_LATENT_SOURCE_DIRS
    assert "NOT in" in qe.qwen_edit_allowlist_gap()


# --------------------------------------------------------------------------------------------------
# 2. Pixels: [-1, 1], and provably NOT the H3 recipe.
# --------------------------------------------------------------------------------------------------


def test_pixel_normalization_is_minus_one_to_one_not_imagenet() -> None:
    from signet_trainer.prep.h3_encode import imagenet_normalize

    pixels = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(1, 1, 1, 3).expand(3, 1, 1, 3)
    out = qe.qwen_edit_normalize_pixels(pixels)
    assert out.min().item() == pytest.approx(-1.0)
    assert out.max().item() == pytest.approx(1.0)
    assert out[0, 0, 0, 1].item() == pytest.approx(128 / 255 * 2 - 1, abs=1e-6)
    assert qe.QWEN_EDIT_PIXEL_RANGE == (-1.0, 1.0)
    # The copy-paste this guards: H3's ImageNet-over-[0,1] gives a different answer at every pixel.
    assert not torch.allclose(out, imagenet_normalize(pixels))


def test_vae_rank_refusal_matches_the_real_class_unpack() -> None:
    with pytest.raises(TypeError, match=r"4-D.*autoencoder_kl_qwenimage.py:783"):
        qe.assert_qwen_edit_vae_encode_input(torch.zeros(3, 1, 8, 8), what="probe")
    qe.assert_qwen_edit_vae_encode_input(torch.zeros(1, 3, 1, 8, 8), what="probe")

    # Real-class oracle, READ rather than constructed: the claim is that _encode unpacks FIVE names.
    source = _diffusers_source("models", "autoencoders", "autoencoder_kl_qwenimage.py")
    assert "_, _, num_frame, height, width = x.shape" in source


# --------------------------------------------------------------------------------------------------
# 3. THE ON-DISK DICT SHAPE.
# --------------------------------------------------------------------------------------------------


def test_target_payload_records_LATENT_dims_and_carries_pixels_separately(tmp_path: Path) -> None:
    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    payload = qe.encode_qwen_edit_target_latents(
        vae, _prepared(), mean, std, stem="img001"
    )

    assert tuple(payload["latents"].shape) == (QWEN_EDIT_LATENT_CHANNELS, 1, LATENT_EDGE, LATENT_EDGE)
    assert payload["num_frames"] == 1
    # ⛔ LATENT grid, not pixels. 64, never 512.
    assert (payload["height"], payload["width"]) == (LATENT_EDGE, LATENT_EDGE)
    assert payload["source_wh"] == (CANVAS, CANVAS)
    assert payload["encoded_wh"] == (CANVAS, CANVAS)
    assert payload["latent_rows"] == ROWS
    assert payload["stem"] == "img001"

    written = qe.write_qwen_edit_precomputed(tmp_path, "img001", target=payload)
    assert set(written) == {qe.QWEN_EDIT_LATENTS_DIR}
    reloaded = torch.load(written[qe.QWEN_EDIT_LATENTS_DIR], weights_only=False)
    assert (reloaded["height"], reloaded["width"]) == (LATENT_EDGE, LATENT_EDGE)
    assert tuple(reloaded["latents"].shape[2:]) == (reloaded["height"], reloaded["width"])


def test_control_payload_is_a_slot_list_with_a_slot_zero_alias(tmp_path: Path) -> None:
    from signet_trainer.conditioning.qwen_edit import _control_entries

    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    slots = resolve_control_slots(
        "img001",
        [
            {"slot": 0, "stem": "img001", "path": "a/img001.png"},
            {"slot": 1, "stem": "img001", "path": "b/img001.png"},
            {"slot": 2, "stem": "img001", "blank": True, "fill": "black"},
        ],
        control_slots=3,
        blank_slot_fill="black",
    )
    images = [
        _prepared(),
        _prepared(),
        qe.prepare_qwen_edit_image(
            qe.qwen_edit_blank_image("black", CANVAS, CANVAS), CANVAS_AREA, blank=True, fill="black"
        ),
    ]
    payload = qe.encode_qwen_edit_control_latents(
        vae, slots, images, mean, std, stem="img001", control_slots=3
    )

    assert payload["num_controls"] == 3
    assert [c["slot"] for c in payload["controls"]] == [0, 1, 2]
    assert [c["blank"] for c in payload["controls"]] == [False, False, True]
    assert payload["controls"][2]["fill"] == "black"
    assert payload["controls"][0]["path"] == "a/img001.png"
    for entry in payload["controls"]:
        assert tuple(entry["latents"].shape) == (
            QWEN_EDIT_LATENT_CHANNELS,
            1,
            LATENT_EDGE,
            LATENT_EDGE,
        )
        assert entry["latent_rows"] == ROWS
        assert entry["latent_hw"] == (LATENT_EDGE, LATENT_EDGE)
    # The allowlist ALIAS is slot 0 and only slot 0.
    assert payload["latents"] is payload["controls"][0]["latents"]
    # ...and the reader refuses to mistake it for the whole set.
    assert len(_control_entries(payload)) == 3

    written = qe.write_qwen_edit_precomputed(tmp_path, "img001", controls=payload)
    assert set(written) == {qe.QWEN_EDIT_CONTROL_LATENTS_DIR}


def test_latent_payload_refusals() -> None:
    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    good = qe.encode_qwen_edit_target_latents(vae, _prepared(), mean, std, stem="s")

    with pytest.raises(ValueError, match="is missing"):
        qe._assert_latent_payload(qe.QWEN_EDIT_LATENTS_DIR, {"latents": good["latents"]})
    with pytest.raises(ValueError, match="channel"):
        qe._assert_latent_payload(
            qe.QWEN_EDIT_LATENTS_DIR,
            {**good, "latents": torch.zeros(24, 1, LATENT_EDGE, LATENT_EDGE)},
        )
    with pytest.raises(ValueError, match="IMAGE family"):
        qe._assert_latent_payload(qe.QWEN_EDIT_LATENTS_DIR, {**good, "num_frames": 5})
    with pytest.raises(ValueError, match="LATENT grid"):
        # The exact mistake the task brief's "H_px" phrasing invites: pixels in the LATENT keys.
        qe._assert_latent_payload(
            qe.QWEN_EDIT_LATENTS_DIR, {**good, "height": CANVAS, "width": CANVAS}
        )
    with pytest.raises(ValueError, match="aliases slot 0"):
        qe._assert_latent_payload(qe.QWEN_EDIT_CONTROL_LATENTS_DIR, {**good, "controls": []})


def test_channel_count_refusal_names_the_wrong_vae() -> None:
    vae = _QwenVaeStub()
    vae.config.z_dim = 24
    mean, std = _stats(vae)  # still 16-wide vectors -> the encode is what disagrees
    with pytest.raises(ValueError, match="16 channel"):
        qe._broadcast_channel_vector([0.0] * 24, torch.zeros(1, 16, 1, 4, 4), "latents_mean")


# --------------------------------------------------------------------------------------------------
# 4. CONTROL-SLOT ORDERING: config order, and a missing file is an ERROR.
# --------------------------------------------------------------------------------------------------


def _control_tree(tmp_path: Path, present: tuple[bool, ...]) -> list[Path]:
    dirs = []
    for index, exists in enumerate(present):
        directory = tmp_path / f"ctrl{index}"
        directory.mkdir()
        if exists:
            (directory / "img001.png").write_bytes(b"slot-%d" % index)
        dirs.append(directory)
    return dirs


def test_slot_order_is_config_order(tmp_path: Path) -> None:
    dirs = _control_tree(tmp_path, (True, True, True))
    slots = qe.resolve_qwen_edit_control_sources(
        "img001", dirs, control_slots=3, blank_slot_fill="black"
    )
    assert [s.index for s in slots] == [0, 1, 2]
    assert [Path(s.path).parent.name for s in slots] == ["ctrl0", "ctrl1", "ctrl2"]
    assert all(s.stem == "img001" for s in slots)


def test_missing_control_file_raises_and_never_slides(tmp_path: Path) -> None:
    dirs = _control_tree(tmp_path, (True, False, True))
    with pytest.raises(FileNotFoundError) as excinfo:
        qe.resolve_qwen_edit_control_sources(
            "img001", dirs, control_slots=3, blank_slot_fill="black"
        )
    message = str(excinfo.value)
    assert "slot 1" in message
    assert "ctrl1" in message
    assert "dataloader_mixins.py:984-985" in message
    # The refusal is the whole point: ctrl2's image must NEVER become slot 1's.
    assert "slide" in message


def test_declared_blank_occupies_its_own_index(tmp_path: Path) -> None:
    dirs = _control_tree(tmp_path, (True, False, True))
    slots = qe.resolve_qwen_edit_control_sources(
        "img001", dirs, control_slots=3, blank_slot_fill="gray", blank_slots=(1,)
    )
    assert [s.blank for s in slots] == [False, True, False]
    assert slots[1].fill == "gray"
    assert slots[1].path is None
    # ctrl2 stayed in slot 2 — the property ai-toolkit loses.
    assert Path(slots[2].path).parent.name == "ctrl2"


def test_two_files_for_one_slot_is_ambiguity_not_a_preference(tmp_path: Path) -> None:
    dirs = _control_tree(tmp_path, (True, True, True))
    (dirs[1] / "img001.jpg").write_bytes(b"second")
    with pytest.raises(ValueError, match="Two files claim one slot"):
        qe.resolve_qwen_edit_control_sources(
            "img001", dirs, control_slots=3, blank_slot_fill="black"
        )


def test_directory_count_must_equal_slot_count(tmp_path: Path) -> None:
    dirs = _control_tree(tmp_path, (True, True))
    with pytest.raises(ValueError, match="POSITIONAL"):
        qe.resolve_qwen_edit_control_sources(
            "img001", dirs, control_slots=3, blank_slot_fill="black"
        )


def test_resolution_is_provable_without_a_filesystem() -> None:
    # The injected `exists` is what makes the refusal a unit-testable claim rather than an
    # integration accident.
    seen: list[str] = []

    def _never(path: Path) -> bool:
        seen.append(path.name)
        return False

    with pytest.raises(FileNotFoundError):
        qe.resolve_qwen_edit_control_sources(
            "img001",
            ["a", "b", "c"],
            control_slots=3,
            blank_slot_fill="black",
            exists=_never,
        )
    assert seen[0] == "img001.png"


def test_blank_real_mismatch_is_refused() -> None:
    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    slots = resolve_control_slots(
        "s", [{"slot": 0, "blank": True, "fill": "black"}], control_slots=1, blank_slot_fill="black"
    )
    with pytest.raises(ValueError, match="prepared image is blank=False"):
        qe.encode_qwen_edit_control_latents(
            vae, slots, [_prepared()], mean, std, stem="s", control_slots=1
        )


# --------------------------------------------------------------------------------------------------
# 5. THE CACHE KEY: it MUST include the control images.
# --------------------------------------------------------------------------------------------------


def _key(controls, caption: str = "make it red", area: int = 384 * 384) -> str:
    return qe.qwen_edit_text_cache_key(
        caption=caption,
        controls=controls,
        condition_area_px=area,
        text_encoder_id="qwen2.5-vl-fp8-scaled",
    )


def test_cache_key_changes_when_a_control_image_changes() -> None:
    round1 = [{"slot": 0, "identity": "aaa", "encoded_wh": (384, 384)}]
    round2 = [{"slot": 0, "identity": "bbb", "encoded_wh": (384, 384)}]
    assert _key(round1) != _key(round2)
    # ...and the caption is identical in both, which is exactly the chained-edit workflow: round 2
    # overwrites the control images and keeps the caption. A caption-only key collides here.
    caption_only = qe.qwen_edit_text_cache_key(
        caption="make it red",
        controls=[],
        condition_area_px=384 * 384,
        text_encoder_id="qwen2.5-vl-fp8-scaled",
    )
    assert caption_only != _key(round1) and caption_only != _key(round2)


def test_cache_key_covers_slot_position_size_and_budget() -> None:
    base = [
        {"slot": 0, "identity": "aaa", "encoded_wh": (384, 384)},
        {"slot": 1, "identity": "bbb", "encoded_wh": (384, 384)},
    ]
    swapped = [
        {"slot": 0, "identity": "bbb", "encoded_wh": (384, 384)},
        {"slot": 1, "identity": "aaa", "encoded_wh": (384, 384)},
    ]
    assert _key(base) != _key(swapped), "slot ORDER is part of the request"
    resized = [{**base[0], "encoded_wh": (512, 288)}, base[1]]
    assert _key(base) != _key(resized), "the same file at a different size is different pixels"
    assert _key(base) != _key(base, area=1024 * 1024), "the budget is part of the request"
    assert _key(base) == _key(list(reversed(base))), "input order is normalized; SLOT index is not"


def test_the_flattened_string_form_the_modal_stage_builds_is_accepted() -> None:
    """``modal/fns.py::qwen_edit_preprocess`` passes strings, not mappings. Both must key correctly.

    The first integration run of slice 2 against slice 3 crashed here with
    ``'str' object has no attribute 'get'``. The contract widened rather than the stage changing,
    because the load-bearing property — the key changes when a control changes — holds identically
    in both forms.
    """
    assert _key(["aaa", "blank:black"]) != _key(["bbb", "blank:black"])
    assert _key(["aaa", "blank:black"]) != _key(["aaa", "blank:white"])
    assert _key(["aaa", "bbb"]) != _key(["bbb", "aaa"]), "list position IS the slot index"
    # A blank is never mistaken for a file whose identity happens to start with 'blank'.
    assert _key(["blank:black"]) != _key([{"slot": 0, "identity": "blank:black"}])


def test_blank_slots_contribute_their_fill_and_a_real_slot_must_carry_an_identity() -> None:
    black = [{"slot": 0, "blank": True, "fill": "black", "encoded_wh": (384, 384)}]
    white = [{"slot": 0, "blank": True, "fill": "white", "encoded_wh": (384, 384)}]
    assert _key(black) != _key(white)
    with pytest.raises(ValueError, match="contributes no identity"):
        _key([{"slot": 0, "identity": None, "encoded_wh": (384, 384)}])


def test_path_mode_reproduces_the_ai_toolkit_staleness_and_content_mode_does_not(
    tmp_path: Path,
) -> None:
    control = tmp_path / "ctrl0" / "img001.png"
    control.parent.mkdir()
    control.write_bytes(b"round-1 pixels")
    path_r1 = qe.qwen_edit_control_identity(control, mode="path")
    content_r1 = qe.qwen_edit_control_identity(control, mode="content")

    # The chained-edit workflow: overwrite the control IN PLACE and keep the caption.
    control.write_bytes(b"round-2 pixels")
    path_r2 = qe.qwen_edit_control_identity(control, mode="path")
    content_r2 = qe.qwen_edit_control_identity(control, mode="content")

    assert path_r1 == path_r2, "'path' hashes the string — this IS ai-toolkit's staleness bug"
    assert content_r1 != content_r2, "'content' hashes the bytes — the default, and the correct one"

    # Bytes in hand, no filesystem: the chunked read and the direct hash must agree.
    assert qe.qwen_edit_control_identity(control, mode="content", data=b"round-2 pixels") == content_r2

    with pytest.raises(ValueError, match="control_cache_key_mode"):
        qe.qwen_edit_control_identity(control, mode="mtime")


def test_cache_gate_answers_false_for_every_re_encode_case() -> None:
    key = _key([{"slot": 0, "identity": "aaa", "encoded_wh": (384, 384)}])
    fresh = {
        qe.QWEN_EDIT_TEXT_PAYLOAD_VERSION_KEY: qe.QWEN_EDIT_TEXT_PAYLOAD_VERSION,
        "cache_key": key,
    }
    assert qe.qwen_edit_text_cache_is_current(fresh, key) is True
    assert qe.qwen_edit_text_cache_is_current({**fresh, "cache_key": "other"}, key) is False
    assert qe.qwen_edit_text_cache_is_current({"cache_key": key}, key) is False
    assert (
        qe.qwen_edit_text_cache_is_current(
            {**fresh, qe.QWEN_EDIT_TEXT_PAYLOAD_VERSION_KEY: 999}, key
        )
        is False
    )
    # The version-1 shape of every other family's text cache: a bare tensor.
    assert qe.qwen_edit_text_cache_is_current(torch.zeros(4, 8), key) is False


# --------------------------------------------------------------------------------------------------
# 6. The Qwen2.5-VL channel.
# --------------------------------------------------------------------------------------------------


def _diffusers_source(*parts: str) -> str:
    """Read a diffusers source file WITHOUT importing diffusers.

    ``importorskip("diffusers")`` was the obvious spelling and it is wrong here twice over. First,
    reading a file to check a constant against it should never execute 30 modules of somebody else's
    package. Second — measured, not theorized — it made these tests suite-ORDER dependent: in the
    full run diffusers' ``utils/export_utils.py:28`` raised ``module 'PIL' has no attribute
    'Image'``, so the oracle failed for a reason that had nothing to do with the constants it
    checks. ``PathFinder`` locates the package on ``sys.path`` and touches neither ``sys.modules``
    nor any ``__init__``.
    """
    from importlib.machinery import PathFinder

    spec = PathFinder().find_spec("diffusers", sys.path)
    locations = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    if not locations:
        pytest.skip("diffusers is not installed — the real-file oracle needs its sources")
    return Path(locations[0]).joinpath(*parts).read_text(encoding="utf-8")


def _pipeline_source() -> str:
    return _diffusers_source("pipelines", "qwenimage", "pipeline_qwenimage_edit_plus.py")


def _pipeline_literals() -> dict[str, Any]:
    """``{attribute: literal}`` for every ``self.<name> = <constant>`` in the pipeline's ``__init__``."""
    found: dict[str, Any] = {}
    for node in ast.walk(ast.parse(_pipeline_source())):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                found[target.attr] = node.value.value
    return found


def _pipeline_string_constants() -> set[str]:
    return {
        node.value
        for node in ast.walk(ast.parse(_pipeline_source()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_presentation_matches_the_pipeline_verbatim() -> None:
    presentation = qe.build_qwen_edit_presentation("make it red", 2)
    assert "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" in presentation
    assert "Picture 2: <|vision_start|><|image_pad|><|vision_end|>" in presentation
    assert presentation.startswith("<|im_start|>system\n")
    assert presentation.endswith("<|im_start|>assistant\n")
    assert "Picture 3" not in presentation

    # REAL-CLASS ORACLE. The four constants are compared against the installed pipeline's own
    # assignments, parsed out of its AST — not searched for as substrings, because the file stores
    # "\n" as two characters while our constant holds real newlines, and a substring test would go
    # green for the wrong reason (or, as it did on the first run, red for the wrong reason).
    literals = _pipeline_literals()
    assert literals["prompt_template_encode"] == qe.QWEN_EDIT_PROMPT_TEMPLATE
    assert literals["prompt_template_encode_start_idx"] == qe.QWEN_EDIT_PROMPT_TEMPLATE_START_IDX
    assert literals["tokenizer_max_length"] == qe.QWEN_EDIT_TOKENIZER_MAX_LENGTH
    # The per-image block is a local, not a self-assignment; it is compared as a literal node.
    assert qe.QWEN_EDIT_IMG_PROMPT_TEMPLATE in _pipeline_string_constants()


def test_text_payload_shape_and_prefix_drop(tmp_path: Path) -> None:
    processor, encoder = _ProcessorStub(tokens=96), _TextEncoderStub(tokens=96)
    payload = qe.encode_qwen_edit_text_conditions(
        encoder, processor, "make it red", [_image(384, 384)], cache_key="k"
    )
    kept = 96 - qe.QWEN_EDIT_PROMPT_TEMPLATE_START_IDX
    assert tuple(payload["prompt_embeds"].shape) == (1, kept, TEXT_DIM)
    assert tuple(payload["prompt_embeds_mask"].shape) == (1, kept)
    assert payload["prompt_embeds_mask"].dtype == torch.int64
    assert payload["text_length"] == kept
    assert payload["n_condition_images"] == 1
    assert payload["cache_key"] == "k"
    # hidden_states[-1] — the LAST layer, which is what the edit pipeline takes (:270). The stub
    # fills layer i with the value i, so this is a positive identification, not a shape check.
    assert payload["prompt_embeds"].unique().tolist() == [float(encoder.layers - 1)]
    # The processor saw the presentation, and the VL half was exercised.
    assert processor.image_counts == [1]
    assert {"pixel_values", "image_grid_thw"} <= encoder.seen_kwargs[0]

    written = qe.write_qwen_edit_precomputed(tmp_path, "img001", text=payload)
    assert set(written) == {qe.QWEN_EDIT_CONDITIONS_DIR}
    # The strategy's own reader must accept what this writer produced.
    from signet_trainer.conditioning.qwen_edit import _text_payload

    embeds, mask = _text_payload(
        torch.load(written[qe.QWEN_EDIT_CONDITIONS_DIR], weights_only=False), "text conditions"
    )
    assert embeds.shape[1] == mask.shape[1] == kept
    assert mask.dtype == torch.int64


def test_short_presentation_does_not_survive_the_prefix_drop() -> None:
    processor, encoder = _ProcessorStub(tokens=32), _TextEncoderStub(tokens=32)
    with pytest.raises(RuntimeError, match="prefix drop"):
        qe.encode_qwen_edit_text_conditions(encoder, processor, "hi", [], cache_key="k")


def test_missing_vision_half_is_named() -> None:
    processor = _ProcessorStub(tokens=96, emit_vision=False)
    encoder = _TextEncoderStub(tokens=96)
    with pytest.raises(RuntimeError, match="5376x1280"):
        qe.encode_qwen_edit_text_conditions(
            encoder, processor, "make it red", [_image(384, 384)], cache_key="k"
        )


def test_over_long_presentation_is_refused_before_the_encoder_runs() -> None:
    processor = _ProcessorStub(tokens=qe.QWEN_EDIT_TOKENIZER_MAX_LENGTH + 1)
    encoder = _TextEncoderStub(tokens=qe.QWEN_EDIT_TOKENIZER_MAX_LENGTH + 1)
    with pytest.raises(ValueError, match="tokenizer_max_length"):
        qe.encode_qwen_edit_text_conditions(encoder, processor, "x", [], cache_key="k")
    assert encoder.grad_enabled_calls == [], "the refusal must land before the forward"


def test_text_payload_write_refusals() -> None:
    good = {
        qe.QWEN_EDIT_TEXT_PAYLOAD_VERSION_KEY: qe.QWEN_EDIT_TEXT_PAYLOAD_VERSION,
        "cache_key": "k",
        "prompt_embeds": torch.zeros(1, 4, TEXT_DIM),
        "prompt_embeds_mask": torch.ones(1, 4, dtype=torch.int64),
    }
    qe._assert_text_payload(qe.QWEN_EDIT_CONDITIONS_DIR, good)
    with pytest.raises(ValueError, match="int64"):
        qe._assert_text_payload(
            qe.QWEN_EDIT_CONDITIONS_DIR,
            {**good, "prompt_embeds_mask": torch.zeros(1, 4).masked_fill(torch.ones(1, 4).bool(), float("-inf"))},
        )
    with pytest.raises(ValueError, match="cache_key"):
        qe._assert_text_payload(
            qe.QWEN_EDIT_CONDITIONS_DIR, {k: v for k, v in good.items() if k != "cache_key"}
        )


# --------------------------------------------------------------------------------------------------
# 7. THE PARTIAL-CACHE GUARD.
# --------------------------------------------------------------------------------------------------


def _write_sample(root: Path, rel: str, *, sources: tuple[str, ...]) -> None:
    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    target = qe.encode_qwen_edit_target_latents(vae, _prepared(), mean, std, stem=rel)
    slots = resolve_control_slots(
        rel, [{"slot": 0, "stem": rel, "path": "a.png"}], control_slots=1, blank_slot_fill="black"
    )
    controls = qe.encode_qwen_edit_control_latents(
        vae, slots, [_prepared()], mean, std, stem=rel, control_slots=1
    )
    text = {
        qe.QWEN_EDIT_TEXT_PAYLOAD_VERSION_KEY: qe.QWEN_EDIT_TEXT_PAYLOAD_VERSION,
        "cache_key": "k",
        "prompt_embeds": torch.zeros(1, 4, TEXT_DIM),
        "prompt_embeds_mask": torch.ones(1, 4, dtype=torch.int64),
    }
    qe.write_qwen_edit_precomputed(
        root,
        rel,
        target=target if qe.QWEN_EDIT_LATENTS_DIR in sources else None,
        controls=controls if qe.QWEN_EDIT_CONTROL_LATENTS_DIR in sources else None,
        text=text if qe.QWEN_EDIT_CONDITIONS_DIR in sources else None,
    )


def test_complete_cache_passes_and_reports_counts(tmp_path: Path) -> None:
    for rel in ("a", "b"):
        _write_sample(tmp_path, rel, sources=qe.QWEN_EDIT_PRECOMPUTED_DIRS)
    counts = qe.assert_qwen_edit_cache_complete(tmp_path)
    assert counts == {name: 2 for name in qe.QWEN_EDIT_PRECOMPUTED_DIRS}


def test_partial_cache_is_refused_by_name(tmp_path: Path) -> None:
    _write_sample(tmp_path, "a", sources=qe.QWEN_EDIT_PRECOMPUTED_DIRS)
    _write_sample(
        tmp_path,
        "b",
        sources=(qe.QWEN_EDIT_LATENTS_DIR, qe.QWEN_EDIT_CONTROL_LATENTS_DIR),
    )
    with pytest.raises(RuntimeError) as excinfo:
        qe.assert_qwen_edit_cache_complete(tmp_path)
    message = str(excinfo.value)
    assert "PARTIAL CACHE" in message
    assert "b.pt" in message
    assert qe.QWEN_EDIT_CONDITIONS_DIR in message
    # Nothing is deleted — the h3_text_payload rule.
    assert (tmp_path / qe.QWEN_EDIT_LATENTS_DIR / "b.pt").exists()


def test_empty_source_is_refused(tmp_path: Path) -> None:
    _write_sample(tmp_path, "a", sources=(qe.QWEN_EDIT_LATENTS_DIR,))
    with pytest.raises(RuntimeError, match="ZERO .pt files"):
        qe.assert_qwen_edit_cache_complete(tmp_path)


def test_a_sample_skipped_by_every_branch_is_still_caught(tmp_path: Path) -> None:
    # Self-consistent across sources, but smaller than the corpus. The pairing check cannot see it.
    for rel in ("a", "b"):
        _write_sample(tmp_path, rel, sources=qe.QWEN_EDIT_PRECOMPUTED_DIRS)
    with pytest.raises(RuntimeError, match="believes it processed"):
        qe.assert_qwen_edit_cache_complete(tmp_path, expected_rels=["a", "b", "c"])


# --------------------------------------------------------------------------------------------------
# 8. D-10-DEF-10 — autograd, via the SHARED scanners (no second copy of the mechanism).
# --------------------------------------------------------------------------------------------------


def test_every_encode_entry_point_carries_the_shared_no_grad_marker() -> None:
    entry_points = h3_encode_entry_points(MODULE_SOURCE)
    assert entry_points, "the scan found no encode_* entry points — it is pointed at the wrong file"
    missing = [name for name, decorators in entry_points.items() if H3_NO_GRAD_DECORATOR not in decorators]
    assert not missing, f"encode entry points without @{H3_NO_GRAD_DECORATOR}: {missing}"
    # ...and the marker survives on the LIVE object, not only in the source.
    for name in entry_points:
        assert getattr(getattr(qe, name), H3_NO_GRAD_MARKER_ATTR, False), name


def test_every_model_forward_site_in_this_module_is_grad_free() -> None:
    sites = model_forward_sites(MODULE_SOURCE)
    assert sites, "the scan found no model-forward sites — it is pointed at the wrong file"
    leaky = [repr(site) for site in sites if not site.grad_free]
    assert not leaky, f"model forwards outside a no-grad context: {leaky}"


@pytest.mark.parametrize("entry", ["target", "control", "text"])
def test_components_observe_grad_DISABLED(entry: str) -> None:
    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    if entry == "target":
        qe.encode_qwen_edit_target_latents(vae, _prepared(), mean, std, stem="s")
        assert vae.grad_enabled_calls == [False]
    elif entry == "control":
        slots = resolve_control_slots(
            "s", [{"slot": 0, "path": "a.png"}], control_slots=1, blank_slot_fill="black"
        )
        qe.encode_qwen_edit_control_latents(
            vae, slots, [_prepared()], mean, std, stem="s", control_slots=1
        )
        assert vae.grad_enabled_calls == [False]
    else:
        processor, encoder = _ProcessorStub(), _TextEncoderStub()
        qe.encode_qwen_edit_text_conditions(encoder, processor, "c", [], cache_key="k")
        assert encoder.grad_enabled_calls == [False]


def test_mutation_probe_the_guard_is_not_a_tautology(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip the decorators and the SAME probe must come back grad-ENABLED. Otherwise it proves nothing."""
    for name in h3_encode_entry_points(MODULE_SOURCE):
        function = getattr(qe, name)
        monkeypatch.setattr(qe, name, function.__wrapped__)
    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    qe.encode_qwen_edit_target_latents(vae, _prepared(), mean, std, stem="s")
    assert vae.grad_enabled_calls == [True], "the no-grad guard is vacuous — it would never go red"


# --------------------------------------------------------------------------------------------------
# 9. D-10-DEF-12 — placement, decided ONCE, off the component.
# --------------------------------------------------------------------------------------------------


def test_foreign_device_is_refused_by_name() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        qe.assert_qwen_edit_encode_device(
            torch.zeros(2), torch.nn.Linear(2, 2).to("meta"), what="encode_qwen_edit_latents"
        )
    message = str(excinfo.value)
    assert "meta" in message and "cpu" in message
    assert "slow_conv3d_forward" in message
    # A component with no opinion is a no-op, never a guess.

    class _Weightless:
        pass

    qe.assert_qwen_edit_encode_device(torch.zeros(2), _Weightless(), what="probe")


def test_pixels_are_moved_onto_the_components_own_device() -> None:
    vae = _QwenVaeStub(device="meta")
    mean, std = _stats(vae)
    latents = qe.encode_qwen_edit_latents(
        vae, qe.qwen_edit_pixels_from_image(_image(64, 64)), mean, std, posterior="mode"
    )
    # The pixels were built on CPU by torch.from_numpy and arrived on the VAE's device anyway.
    assert vae.encoded_devices == [torch.device("meta")]
    assert latents.device == torch.device("meta")


def test_the_move_precedes_the_normalization_structurally() -> None:
    """AST order, because the reason for the order is not observable from the result.

    ``uint8`` crossing the bus instead of ``float32`` is 4x less traffic, and the normalization then
    runs where the weights are. A refactor that normalizes first is correct-looking and slower on
    every sample of every run.
    """
    tree = ast.parse(MODULE_SOURCE)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "encode_qwen_edit_latents"
    )
    move = min(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
    )
    normalize = min(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "qwen_edit_normalize_pixels"
    )
    assert move < normalize, "the device move must precede the normalization (uint8 over the bus)"


# --------------------------------------------------------------------------------------------------
# 10. Posterior policy and the two-budget resize.
# --------------------------------------------------------------------------------------------------


def test_target_samples_at_a_fixed_seed_and_control_takes_the_mode() -> None:
    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    qe.encode_qwen_edit_target_latents(vae, _prepared(), mean, std, stem="s")
    assert vae.posterior_calls == [("sample", qe.QWEN_EDIT_ENCODE_SEED)]

    vae.posterior_calls.clear()
    slots = resolve_control_slots(
        "s", [{"slot": 0, "path": "a.png"}], control_slots=1, blank_slot_fill="black"
    )
    qe.encode_qwen_edit_control_latents(
        vae, slots, [_prepared()], mean, std, stem="s", control_slots=1
    )
    assert vae.posterior_calls == [("mode", None)]


def test_posterior_is_required_never_defaulted() -> None:
    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    pixels = qe.qwen_edit_pixels_from_image(_image(64, 64))
    with pytest.raises(TypeError):
        qe.encode_qwen_edit_latents(vae, pixels, mean, std)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="invalid posterior policy"):
        qe.encode_qwen_edit_latents(vae, pixels, mean, std, posterior="argmax")


def test_the_same_source_is_prepared_twice_at_two_budgets_never_re_fitted() -> None:
    source = _image(1024, 1024)
    vae_channel = qe.prepare_qwen_edit_image(source, qe.QWEN_EDIT_VAE_IMAGE_SIZE)
    vl_channel = qe.prepare_qwen_edit_image(source, qe.QWEN_EDIT_CONDITION_IMAGE_SIZE)
    assert vae_channel.encoded_wh == (1024, 1024)
    assert vl_channel.encoded_wh == (384, 384)
    # Idempotent at the SAME budget...
    assert qe.prepare_qwen_edit_image(vae_channel, qe.QWEN_EDIT_VAE_IMAGE_SIZE) is vae_channel
    # ...and a refusal at a different one: re-fitting a resized copy resamples twice.
    with pytest.raises(ValueError, match="twice FROM THE SOURCE"):
        qe.prepare_qwen_edit_image(vae_channel, qe.QWEN_EDIT_CONDITION_IMAGE_SIZE)


def test_orientation_is_preserved_the_phase_1_ruling(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        prepared = qe.prepare_qwen_edit_image(_image(1024, 512), qe.QWEN_EDIT_VAE_IMAGE_SIZE)
    width, height = prepared.encoded_wh
    assert width > height, "landscape in, landscape out (diffusers pipeline:678-680)"
    # ai-toolkit would emit the transpose; the operator is told so rather than left to discover it.
    assert "ai-toolkit" in caplog.text and f"{height}x{width}" in caplog.text


def test_blank_fills_render_the_declared_pixels() -> None:
    for fill, rgb in qe.QWEN_EDIT_BLANK_FILL_RGB.items():
        image = qe.qwen_edit_blank_image(fill, 32, 32)
        assert image.getpixel((0, 0)) == rgb
    with pytest.raises(ValueError, match="unknown blank fill"):
        qe.qwen_edit_blank_image("puce", 32, 32)
    with pytest.raises(ValueError, match="degenerate"):
        qe.qwen_edit_blank_image("black", 0, 32)


def test_realized_rows_expose_the_non_square_pricing_gap() -> None:
    """A non-square control realizes MORE rows than ``qwen_edit_packed_layout`` prices. Measured.

    The geometry module states that slot cost "does not vary with source resolution" because every
    control is fitted to one area budget. The area is fitted; the per-edge 32-snap then moves the
    PRODUCT. This test is the standing evidence, so the claim cannot quietly stay true-looking.
    """
    from signet_trainer.conditioning.qwen_edit_geometry import (
        qwen_edit_area_budget_size,
        qwen_edit_rows_of,
    )

    budget = 1024 * 1024
    square = qwen_edit_rows_of(*qwen_edit_area_budget_size(1024, 1024, budget))
    landscape = qwen_edit_rows_of(*qwen_edit_area_budget_size(1024, 512, budget))
    assert square == 4096
    assert landscape == 4140, "the 32-snap does not preserve the area, so it does not preserve rows"

    vae = _QwenVaeStub()
    mean, std = _stats(vae)
    target = qe.encode_qwen_edit_target_latents(vae, _prepared(), mean, std, stem="s")
    slots = resolve_control_slots(
        "s", [{"slot": 0, "path": "a.png"}], control_slots=1, blank_slot_fill="black"
    )
    controls = qe.encode_qwen_edit_control_latents(
        vae, slots, [_prepared()], mean, std, stem="s", control_slots=1
    )
    realized = qe.qwen_edit_realized_image_rows(target, controls)
    assert realized == {
        "n_target": ROWS,
        "n_control": ROWS,
        "per_slot": [ROWS],
        "n_image_stream": 2 * ROWS,
    }


def test_vae_latent_stats_refuse_a_config_without_them() -> None:
    class _NoStats:
        class config:  # noqa: N801
            scaling_factor = 0.13025

    with pytest.raises(RuntimeError, match="no scaling_factor to fall back to"):
        qe.qwen_edit_vae_latent_stats(_NoStats())
