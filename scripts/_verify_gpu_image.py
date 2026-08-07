"""Throwaway CPU check (Phase 2 Task 4) — validate gpu_image's torch/torchvision/transformers chain.

Runs the heavy gpu_image on a CPU container (no GPU spend) and exercises exactly the imports that
broke load_ltxv_smoke: torchvision::nms op registration + transformers AutoImageProcessor (needed
to load the multimodal gemma-3-12b-it). Cheap pre-check before the real A100 smoke.

Run: PYTHONPATH=src PYTHONIOENCODING=utf-8 python -m modal run scripts/_verify_gpu_image.py
Delete after Phase 2.
"""

import modal

from signet_trainer.modal.app import gpu_image

app = modal.App("signet-verify-gpu-image")


@app.function(image=gpu_image, timeout=600)  # CPU — intentionally NO gpu=.
def verify() -> None:
    import torch

    print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
    import torchvision

    print("torchvision:", torchvision.__version__)
    from torchvision.ops import nms  # noqa: F401

    print("torchvision.ops.nms import: OK")
    # The op that was 'does not exist' — touch it to force C++ op resolution.
    import torch as _t

    boxes = _t.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    scores = _t.tensor([0.9, 0.8])
    _ = nms(boxes, scores, 0.5)
    print("torchvision::nms call: OK")
    from transformers import AutoImageProcessor  # noqa: F401

    print("transformers AutoImageProcessor import: OK")
    print("ALL GOOD — gpu_image import chain is healthy")


@app.local_entrypoint()
def main() -> None:
    verify.remote()
