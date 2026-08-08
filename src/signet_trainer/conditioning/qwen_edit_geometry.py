"""Qwen-Image-Edit-2511 arch math — the SINGLE home for ``qwen_edit`` geometry (family #3).

This is the ``qwen_edit`` analog of ``conditioning/h3_geometry.py``, and it obeys the same
"constants + helpers, imported never duplicated" contract: ``config/validators.py`` RE-EXPORTS
from here and never redefines a law, ``config/schema.py`` calls it to price a config, and
``dryrun/shapes.py`` calls it to size a synthetic batch. Three callers, one arithmetic.

CRITICAL — Anti-Pattern 6, STRICTEST import tier (identical to ``h3_geometry``):
    stdlib + ``dataclasses`` ONLY. **No ``torch``. No ``modal``. No ``diffusers``. No ``peft``.**
    ``config/validators.py`` re-exports this module and ``validators`` sits inside
    ``modal/app.py::download_image``'s import closure (``modal/app.py:176-208``); a ``torch``
    import here would widen that closure — the BK-01 landmine, whose only symptom is a
    ``ModuleNotFoundError`` in a PAID container after every local gate has passed.

Where the numbers come from
---------------------------
Every constant below was TRANSCRIBED from a file that was read, with the citation kept inline so a
future reader can re-verify without re-deriving:

    control area cap  ``VAE_IMAGE_SIZE = 1024 * 1024``
                      diffusers ``pipelines/qwenimage/pipeline_qwenimage_edit_plus.py:67``
    VL condition cap  ``CONDITION_IMAGE_SIZE = 384 * 384`` — same file, ``:66``
    VAE scale factor  ``2 ** len(vae.temperal_downsample)``, else 8 — same file, ``:209``
    latent channels   ``vae.config.z_dim``, else 16 — same file, ``:210``
    2x2 patch pack    ai-toolkit ``extensions_built_in/diffusion_models/qwen_image/
                      qwen_image_edit_plus.py:222-228``::

                          latent.view(B, C, H//2, 2, W//2, 2)
                               .permute(0, 2, 4, 1, 3, 5)
                               .reshape(B, (H//2)*(W//2), C*4)

                      so ONE packed row covers a 2x2 latent cell = a 16x16 PIXEL cell, and the row
                      feature width is ``C * 4 = 64``.
    control APPEND    ai-toolkit ``qwen_image_edit_plus.py:315-317``::

                          torch.cat([packed_latents_list[b], control], dim=1)

                      The control block is a **SUFFIX**, not a prefix. See the warning below.
    edge rounding     ai-toolkit ``qwen_image_edit_plus.py:270-271`` (control) and ``:181-182``
                      (VL condition): ``round(edge / 32) * 32``.
    transformer       measured on the live checkpoint ``qwen_image_edit_2511_bf16.safetensors``
                      (40,861,031,560 B, 1934 tensors): 60 ``transformer_blocks``, hidden dim 3072,
                      dual-stream MMDiT (``img_*`` and ``txt_*`` are separate parameter sets joined
                      only in attention).

⚠ THE CONTROL BLOCK IS A SUFFIX, NOT A PREFIX
---------------------------------------------
Every other conditioning arm in this repo puts its clean conditioning rows FIRST — ``ic_lora``'s
reference prefix (``dryrun/shapes.py`` ic_lora branch) and H3's ``n_cond_video`` prefix both live at
``[:ref_seq_len]``, and both are excluded from the loss by slicing the HEAD. Qwen-Image-Edit does
the opposite: ``torch.cat([image, control], dim=1)``. A ``[:, ref_seq_len:]`` slice copied from
either sibling arm therefore selects exactly the wrong half — it would drop every target row and
train the adapter on nothing but its own control conditioning. ``ModelInputs.ref_seq_len`` is
deliberately left ``None`` on this family so the name cannot be borrowed by accident; the row split
travels as :class:`QwenEditPackedLayout` instead.

Why this is NOT ``compute_seq_len``
-----------------------------------
``conditioning/strategy.py::compute_seq_len(width, height, frames)`` divides by 32 (LTX's
``HEIGHT_SCALE``/``WIDTH_SCALE``) and applies the ``(F-1)//8 + 1`` frame law. Qwen packs at
**16 pixels per row edge** (VAE 8x then a 2x2 patch), so ``compute_seq_len`` under-counts a Qwen
image by exactly 4x while returning a perfectly plausible integer — probed on this tree:

    compute_seq_len(1024, 1024, 1) == 1024      (LTX law)
    qwen_edit_rows_of(1024, 1024) == 4096       (this module)

A silently-4x-wrong sequence length is not a shape crash; it is a budget that passes and an
adapter that trains against the wrong geometry. Do NOT reuse ``compute_seq_len`` here, and do NOT
add Qwen constants to ``strategy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "QWEN_EDIT_BLOCK_COUNT",
    "QWEN_EDIT_CANVAS_MULTIPLE",
    "QWEN_EDIT_CONDITION_IMAGE_SIZE",
    "QWEN_EDIT_FRAMES",
    "QWEN_EDIT_HIDDEN_DIM",
    "QWEN_EDIT_LATENT_CHANNELS",
    "QWEN_EDIT_MAX_CONTROL_SLOTS",
    "QWEN_EDIT_PATCH",
    "QWEN_EDIT_PATCH_DIM",
    "QWEN_EDIT_PIXELS_PER_ROW_EDGE",
    "QWEN_EDIT_VAE_IMAGE_SIZE",
    "QWEN_EDIT_VAE_SCALE_FACTOR",
    "QwenEditPackedLayout",
    "qwen_edit_area_budget_size",
    "qwen_edit_latent_size",
    "qwen_edit_packed_layout",
    "qwen_edit_rows_of",
    "qwen_edit_vae_latent_shape",
]

# --------------------------------------------------------------------------------------------------
# Transcribed constants. NONE of these may be re-derived at a call site (D-NOHARDCODE).
# --------------------------------------------------------------------------------------------------

#: ``2 ** len(vae.temperal_downsample)``, else 8 — ``pipeline_qwenimage_edit_plus.py:209``. The
#: qwen_image VAE on this box (``qwen_image_vae.safetensors``, 253,806,246 B) is the 8x spatial one.
QWEN_EDIT_VAE_SCALE_FACTOR: int = 8

#: ``vae.config.z_dim``, else 16 — ``pipeline_qwenimage_edit_plus.py:210``.
QWEN_EDIT_LATENT_CHANNELS: int = 16

#: The 2x2 latent patch the packer folds into ONE row (ai-toolkit ``qwen_image_edit_plus.py:222``).
QWEN_EDIT_PATCH: int = 2

#: Feature width of ONE packed row: ``C * patch**2`` == ``num_channels_latents * 4``, the literal
#: reshape argument at ai-toolkit ``qwen_image_edit_plus.py:227``. 16 * 4 = 64.
QWEN_EDIT_PATCH_DIM: int = QWEN_EDIT_LATENT_CHANNELS * QWEN_EDIT_PATCH * QWEN_EDIT_PATCH

#: PIXELS per packed-row edge: VAE 8x then the 2x2 pack = 16. Independently corroborated by
#: ``VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)`` at
#: ``pipeline_qwenimage_edit_plus.py:213`` — the processor's own alignment unit is this same 16.
QWEN_EDIT_PIXELS_PER_ROW_EDGE: int = QWEN_EDIT_VAE_SCALE_FACTOR * QWEN_EDIT_PATCH

#: Pixel-dimension multiple every canvas and every control image is snapped to. 32, not 16: it is
#: what ai-toolkit actually rounds to (``round(edge / 32) * 32``, ``qwen_image_edit_plus.py:270-271``
#: and ``:181-182``), and it is also LTX's ``HEIGHT_SCALE``/``WIDTH_SCALE``, so a bucket string that
#: is legal for LTX spatially is legal here spatially. Being 2x the strict 16 requirement, it
#: guarantees an EVEN latent edge, which is what makes the 2x2 pack exact rather than lossy.
QWEN_EDIT_CANVAS_MULTIPLE: int = 32

#: ``VAE_IMAGE_SIZE`` — the AREA budget (in pixels) each CONTROL image is resized to before the VAE.
#: diffusers ``pipeline_qwenimage_edit_plus.py:67``: ``1024 * 1024``.
QWEN_EDIT_VAE_IMAGE_SIZE: int = 1024 * 1024

#: ``CONDITION_IMAGE_SIZE`` — the (much smaller) AREA budget each control image is resized to for
#: the Qwen2.5-VL text-encoder pass. diffusers ``pipeline_qwenimage_edit_plus.py:66``: ``384 * 384``.
#: A control image is therefore resized TWICE, to two different budgets, for two different consumers
#: (``qwen_image_edit_plus.py:179`` VL channel, ``:217`` VAE channel) — conflating them silently
#: mis-sizes one of the two streams.
QWEN_EDIT_CONDITION_IMAGE_SIZE: int = 384 * 384

#: Qwen-Image-Edit is an IMAGE family. F is EXACTLY 1 — never a law with a modulus, never a range.
#: ``(1 - 1) % 8 == 0`` so LTX's frame law already ADMITS 1, which is why the family-exact arm must
#: pin it: a ``9`` typo rides LTX's law straight through the pre-screen otherwise.
QWEN_EDIT_FRAMES: int = 1

#: Control-slot cap. ai-toolkit's prompt template exposes ``ctrl_img_1..3``
#: (``qwen_image_edit_plus.py:105-122``), so 3 is the architecture's own ceiling, not a house choice.
QWEN_EDIT_MAX_CONTROL_SLOTS: int = 3

#: Measured on the live checkpoint: 60 ``transformer_blocks`` (0..59), hidden dim 3072. Kept here
#: (not only in ``lora/peft.py``) because the dry-run banner reports the module count and must not
#: import the heavy LoRA layer to do it.
QWEN_EDIT_BLOCK_COUNT: int = 60
QWEN_EDIT_HIDDEN_DIM: int = 3072


@dataclass(frozen=True)
class QwenEditPackedLayout:
    """The packed-sequence breakdown for ONE ``qwen_edit`` sample, in the model's own order.

    Packed order (ai-toolkit ``qwen_image_edit_plus.py:315-317``)::

        image stream:  [ target image rows | control slot 1 | control slot 2 | ... ]
        text stream:   [ prompt rows ]                       (SEPARATE — dual-stream MMDiT)

    ``n_text`` is the Qwen2.5-VL prompt row count. It is NOT part of the image ``hidden_states``
    tensor — ``img_*`` and ``txt_*`` are separate parameter sets joined only inside attention — but
    it IS part of the attention sequence, so it counts toward ``total`` (the budget number) while
    ``n_image_stream`` is the number a ``hidden_states`` tensor is actually shaped by. Reporting one
    and shaping with the other is the mistake this split exists to prevent.

    ``per_slot`` is the per-control-slot row count, surfaced separately because the control block is
    by far the biggest lever an operator has (3 slots at 1024x1024 are 75% of the sequence).
    """

    n_target_image: int
    n_control: int
    control_slots: int
    per_slot: int
    n_text: int
    total: int

    @property
    def n_image_stream(self) -> int:
        """Rows in the ``hidden_states`` tensor: target rows FIRST, control rows appended after."""
        return self.n_target_image + self.n_control

    def describe(self) -> str:
        """One-liner for the dry-run banner."""
        return (
            f"packed {self.total} rows = target image {self.n_target_image} "
            f"+ control {self.n_control} ({self.control_slots} slot(s) x {self.per_slot}) "
            f"+ text {self.n_text} [image stream {self.n_image_stream} rows x "
            f"{QWEN_EDIT_PATCH_DIM} feat]"
        )


def qwen_edit_latent_size(width: int, height: int) -> tuple[int, int]:
    """VAE latent ``(H_lat, W_lat)`` for a ``(width, height)`` PIXEL canvas — ``//8`` each axis.

    Returns ``(height, width)`` order to match torch's trailing-dims convention (and
    ``h3_geometry.rows_of``'s ``(height, width)`` argument order), NOT the ``WxH`` order of a bucket
    string. Raises on a non-positive or mis-aligned edge rather than returning a plausible integer:
    an odd latent edge makes the 2x2 pack lossy, and the loss is silent.
    """
    for name, edge in (("width", width), ("height", height)):
        if edge <= 0:
            raise ValueError(
                f"invalid {name} {edge}: must be a positive multiple of "
                f"{QWEN_EDIT_CANVAS_MULTIPLE} (Qwen-Image-Edit canvas rule)."
            )
        if edge % QWEN_EDIT_PIXELS_PER_ROW_EDGE:
            raise ValueError(
                f"invalid {name} {edge}: Qwen-Image-Edit packs {QWEN_EDIT_PIXELS_PER_ROW_EDGE} "
                f"pixels per packed-row edge (VAE {QWEN_EDIT_VAE_SCALE_FACTOR}x then a "
                f"{QWEN_EDIT_PATCH}x{QWEN_EDIT_PATCH} patch), so every edge must be a multiple of "
                f"{QWEN_EDIT_PIXELS_PER_ROW_EDGE}; got remainder "
                f"{edge % QWEN_EDIT_PIXELS_PER_ROW_EDGE}. The house rule is the stricter "
                f"{QWEN_EDIT_CANVAS_MULTIPLE} (what ai-toolkit rounds to)."
            )
    return (height // QWEN_EDIT_VAE_SCALE_FACTOR, width // QWEN_EDIT_VAE_SCALE_FACTOR)


def qwen_edit_vae_latent_shape(width: int, height: int) -> tuple[int, int, int, int]:
    """The STORED VAE-latent shape for one image: ``(C, F, H_lat, W_lat)`` with ``F == 1``.

    The F axis is carried EXPLICITLY at rank 4 even though Qwen is an image family and the
    transformer boundary is rank 3 (``qwen_image_edit_plus.py:213`` unpacks
    ``batch_size, num_channels_latents, height, width``). That is not redundancy — it is the
    precomputed-latents contract: ``data/precomputed.py``'s ``_normalize_video_latents`` einops path
    consumes ``{"latents": [C, F, H, W], "num_frames": ..., "height": ..., "width": ...}``, and
    storing rank-3 latents would put ``qwen_edit_latents/`` outside the allowlist that makes that
    path correct. F is squeezed at the model boundary, never at rest.
    """
    lat_h, lat_w = qwen_edit_latent_size(width, height)
    return (QWEN_EDIT_LATENT_CHANNELS, QWEN_EDIT_FRAMES, lat_h, lat_w)


def qwen_edit_rows_of(width: int, height: int) -> int:
    """Packed rows a ``(width, height)`` PIXEL image occupies: ``(H//16) * (W//16)``.

    ``(H_lat // 2) * (W_lat // 2)`` after the VAE, which is the literal reshape at ai-toolkit
    ``qwen_image_edit_plus.py:226-228``. ``qwen_edit_rows_of(1024, 1024) == 4096``.
    """
    lat_h, lat_w = qwen_edit_latent_size(width, height)
    return (lat_h // QWEN_EDIT_PATCH) * (lat_w // QWEN_EDIT_PATCH)


def qwen_edit_area_budget_size(width: int, height: int, area_px: int) -> tuple[int, int]:
    """Resize ``(width, height)`` to an AREA budget, aspect-preserving, each edge snapped to 32.

    Transcribes ai-toolkit ``qwen_image_edit_plus.py:265-271`` (control channel) and ``:178-182``
    (VL channel), which are the same four lines twice. Returns ``(out_width, out_height)``.

    ⚠ DOCUMENTED DIVERGENCE — orientation. ai-toolkit computes ``ratio = shape[2] / shape[3]``,
    i.e. ``H / W`` (its own comment at ``:259`` names the layout ``(1, ch, height, width)``), then
    assigns ``width = sqrt(area * ratio)`` and ``height = width / ratio`` and finally calls
    ``F.interpolate(size=(c_height, c_width))``. Since ``interpolate``'s ``size`` is ``(H, W)``,
    the two names are swapped relative to the axes they land on, so a landscape control image is
    resized to a PORTRAIT canvas. This function preserves orientation instead.

    The divergence is inert for the BUDGET this module exists to compute — a transposition leaves
    ``H * W`` unchanged, so ``qwen_edit_rows_of`` returns the same count either way — and it is
    exactly inert for square inputs, which is what the shipped example config uses. It is recorded
    here rather than silently reproduced or silently "fixed": whoever lands the real preprocessing
    pass must decide deliberately, and must not discover this from a mis-oriented sample grid.
    """
    if area_px < QWEN_EDIT_CANVAS_MULTIPLE * QWEN_EDIT_CANVAS_MULTIPLE:
        raise ValueError(
            f"invalid area budget {area_px}: must be at least one "
            f"{QWEN_EDIT_CANVAS_MULTIPLE}x{QWEN_EDIT_CANVAS_MULTIPLE} tile "
            f"({QWEN_EDIT_CANVAS_MULTIPLE**2} px). The two real budgets are "
            f"VAE_IMAGE_SIZE={QWEN_EDIT_VAE_IMAGE_SIZE} and "
            f"CONDITION_IMAGE_SIZE={QWEN_EDIT_CONDITION_IMAGE_SIZE}."
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            f"invalid source size (width={width}, height={height}): both edges must be positive."
        )

    # aspect-preserving area fit: out_w * out_h == area_px, out_w / out_h == width / height.
    scale = (area_px / (width * height)) ** 0.5
    out_width = _snap(width * scale)
    out_height = _snap(height * scale)
    return (out_width, out_height)


def _snap(edge: float) -> int:
    """``round(edge / 32) * 32``, floored at one tile (ai-toolkit ``qwen_image_edit_plus.py:270``).

    The floor is this module's addition, not ai-toolkit's: ``round(8 / 32) * 32 == 0`` there, which
    would yield a zero-pixel edge, a zero-row control slot, and a control block that silently
    vanishes from the sequence. A degenerate slot must be impossible, not merely unlikely.
    """
    snapped = round(edge / QWEN_EDIT_CANVAS_MULTIPLE) * QWEN_EDIT_CANVAS_MULTIPLE
    return max(snapped, QWEN_EDIT_CANVAS_MULTIPLE)


def qwen_edit_packed_layout(
    training_dims: tuple[int, int, int], qwen_edit: object
) -> QwenEditPackedLayout:
    """Price ONE ``qwen_edit`` sample's packed sequence from the config's own numbers.

    ``training_dims`` is the config triple ``(W, H, F)``. ``qwen_edit`` is DUCK-TYPED — it is read
    for ``control_slots``, ``control_area_px`` and ``prompt_tokens_estimate`` and nothing else. It
    is not annotated as ``QwenEditConfig`` on purpose: importing ``config.schema`` from
    ``conditioning/`` would invert the dependency direction (schema imports geometry, never the
    reverse) and would drag pydantic into the strictest import tier in the repo.

    The control slot count is FIXED, not variable: a sample with fewer real control images is
    blank-padded to ``control_slots`` (``qwen_edit.blank_slot_fill``), so the sequence length is a
    constant of the config rather than a property of the row. A variable-length packed sequence
    would make the row budget a distribution instead of a number, and the whole point of pricing at
    config load is that the answer is one integer.

    Every slot is priced at the SAME ``control_area_px`` budget because that is what the resize
    does — an arbitrarily-sized control image is fitted to the budget before the VAE ever sees it,
    so slot cost does not vary with source resolution.
    """
    width, height, frames = (int(d) for d in training_dims)
    if frames != QWEN_EDIT_FRAMES:
        raise ValueError(
            f"invalid frame count {frames} for the qwen_edit family: Qwen-Image-Edit is an IMAGE "
            f"model and F must be EXACTLY {QWEN_EDIT_FRAMES}. (LTX's law admits 1 as well — "
            f"(1 - 1) % 8 == 0 — so a wrong F rides the shared pre-screen through and only this "
            f"exact pin stops it.)"
        )

    control_slots = int(getattr(qwen_edit, "control_slots"))
    control_area_px = int(getattr(qwen_edit, "control_area_px"))
    prompt_tokens = int(getattr(qwen_edit, "prompt_tokens_estimate"))
    if control_slots < 1 or control_slots > QWEN_EDIT_MAX_CONTROL_SLOTS:
        raise ValueError(
            f"invalid control_slots {control_slots}: must be 1..{QWEN_EDIT_MAX_CONTROL_SLOTS} "
            f"(ai-toolkit's prompt template exposes ctrl_img_1..{QWEN_EDIT_MAX_CONTROL_SLOTS}, "
            f"qwen_image_edit_plus.py:105-122)."
        )
    if prompt_tokens < 1:
        raise ValueError(
            f"invalid prompt_tokens_estimate {prompt_tokens}: must be a positive integer."
        )

    n_target_image = qwen_edit_rows_of(width, height)
    slot_width, slot_height = qwen_edit_area_budget_size(width, height, control_area_px)
    per_slot = qwen_edit_rows_of(slot_width, slot_height)
    n_control = per_slot * control_slots

    return QwenEditPackedLayout(
        n_target_image=n_target_image,
        n_control=n_control,
        control_slots=control_slots,
        per_slot=per_slot,
        n_text=prompt_tokens,
        total=n_target_image + n_control + prompt_tokens,
    )
