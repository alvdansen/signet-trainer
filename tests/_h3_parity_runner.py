"""Runs the D-10-DEF-7 output-key parity probe and prints ONE line of JSON. Not a test module.

⛔ **This file is executed by a DIFFERENT interpreter than the one running pytest** — the one that
has the PINNED ``transformers`` (``modal/app.py::TRANSFORMERS_VERSION``) installed. See
``tests/test_h3_processor_output_parity.py`` for why that separation exists and how the interpreter
is located. It reports facts; the parent process makes every assertion.

It builds a **REAL** ``Qwen3VLProcessor`` — the real class, the real ``ProcessorMixin.__call__``,
the real Qwen2-VL image processor configured from the H3 checkpoint's own
``preprocessor_config.json`` numbers — with **no model weights and no network**. Only the tokenizer
is synthetic, and that is sound for this question: BOTH sides of the diff are handed the SAME
tokenizer object, so whatever keys it contributes cancel. What the diff isolates is exactly what
``ProcessorMixin.__call__`` adds or removes AROUND the tokenizer and image processor — which is the
D-10-DEF-7 class.

A stub processor (``tests/test_h3_reference_geometry_agreement.py`` has one, correctly, for the
geometry question) can NEVER close this class: a stub reproduces what we already believe ``__call__``
does, so it cannot reveal an output nobody has heard of. Hence the real class.

Prints ``{"ok": true, ...}`` on success or ``{"ok": false, "error": ..., "traceback": ...}`` on any
failure, then exits 0 either way — a non-zero exit would be indistinguishable from "the interpreter
could not start", and the parent needs to tell those apart.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# The H3 checkpoint's own ``processor/preprocessor_config.json``, read off the mounted weights during
# the D-10-DEF-4 diagnosis and recorded in `deferred-items.md`. Restated here (not imported from
# `tests/test_h3_reference_geometry_agreement.py`) because that module's copies feed a hand-written
# `smart_resize` transcription, whereas these feed the REAL image processor — two different uses of
# the same measured numbers, and cross-importing would make one file's refactor silently retune the
# other's oracle.
_PATCH_SIZE = 16
_MERGE_SIZE = 2
_TEMPORAL_PATCH_SIZE = 2
_MIN_PIXELS = 65536
_MAX_PIXELS = 16777216


def _tiny_tokenizer():
    """A real ``PreTrainedTokenizerFast`` with a 1-word vocabulary and the four vision specials.

    The vocabulary is irrelevant to an output-KEY question and deliberately tiny (no download, no
    checkpoint). What must be real is the ADDED-TOKEN handling: ``<|image_pad|>`` has to survive as
    a single id, or ``__call__``'s own ``_check_special_mm_tokens`` and our pad-count assertion
    would both be measuring a shredded stream.

    ``model_input_names`` is pinned to Qwen2's ``["input_ids", "attention_mask"]``. Qwen3-VL's
    ``_defaults`` set ``return_token_type_ids: False``, so a tokenizer that emitted
    ``token_type_ids`` would have it suppressed on ``__call__``'s side and NOT on ours — a real
    divergence, and one this fixture must not accidentally hide by being quieter than the real
    tokenizer.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from signet_trainer.prep.h3_encode import (
        H3_IMAGE_PAD_TOKEN,
        H3_VIDEO_PAD_TOKEN,
        H3_VISION_END_TOKEN,
        H3_VISION_START_TOKEN,
    )

    backend = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                H3_VISION_START_TOKEN,
                H3_VISION_END_TOKEN,
                H3_IMAGE_PAD_TOKEN,
                H3_VIDEO_PAD_TOKEN,
            ]
        }
    )
    return tokenizer


def build_real_h3_processor():
    """A genuine ``Qwen3VLProcessor``, weightless. Every part except the tokenizer vocab is real.

    ``Qwen2VLImageProcessor`` and ``Qwen3VLVideoProcessor`` are both torchvision-gated, and the H3
    image installs ``torchvision~=0.22`` — so requiring torchvision here makes the fixture's
    sub-processors the SAME classes ``AutoProcessor.from_pretrained`` builds in the container,
    rather than the PIL fallback. ``ProcessorMixin.__init__`` type-checks all three declared
    sub-processors against their modality base class, so none of them may be ``None``; the video
    one is never exercised (no probe input carries a video) but must exist to construct.
    """
    from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    image_processor = Qwen2VLImageProcessor(
        patch_size=_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
        temporal_patch_size=_TEMPORAL_PATCH_SIZE,
        size={"shortest_edge": _MIN_PIXELS, "longest_edge": _MAX_PIXELS},
    )
    return Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=_tiny_tokenizer(),
        video_processor=Qwen3VLVideoProcessor(),
    )


def main() -> int:
    payload: dict = {"ok": False}
    try:
        import transformers

        payload["transformers_version"] = transformers.__version__
        payload["python"] = sys.executable

        from signet_trainer.prep.h3_parity import (
            H3_PROCESSOR_PARITY_ALLOWLIST,
            h3_processor_output_key_diff,
        )

        processor = build_real_h3_processor()
        payload["processor_class"] = type(processor).__name__
        payload["allowlist"] = dict(H3_PROCESSOR_PARITY_ALLOWLIST)
        payload.update(h3_processor_output_key_diff(processor))

        # The TEXT-ONLY branch is diffed too. ``ProcessorMixin.__call__`` adds
        # ``mm_token_type_ids`` whenever TEXT is given — not when images are — so a builder that
        # gated it on "has references" would diverge here and nowhere else.
        from signet_trainer.prep.h3_encode import build_h3_processor_inputs

        text_only_call = processor(text=["a parity probe caption"], return_tensors="pt")
        text_only_ours = build_h3_processor_inputs(
            processor, "a parity probe caption", (), vision_spans=()
        )
        payload["text_only_call_keys"] = sorted(text_only_call.keys())
        payload["text_only_our_keys"] = sorted(text_only_ours.keys())
        payload["ok"] = True
    except BaseException as exc:  # noqa: BLE001 — the parent turns ANY failure into a red test
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
