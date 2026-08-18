"""Issue #40 finding 3 — the "shared-GPU polite" gate must judge the SAME card `_GPU_PROG` probes.

**The defect.** `stage_gpu` computed ``free = max(g["free_mib"] for g in inv["gpus"])`` — the most
free VRAM across ALL cards — then ran `_GPU_PROG`, which hardcodes ``torch.device("cuda")``, i.e.
whichever card the child process sees as device 0 (never selected). On a two-GPU box with GPU0
saturated and GPU1 idle, the gate passed on GPU1's headroom while the probe allocated on GPU0 — the
exact "polite" property inverted.

The fix: pick the GPU with the most free VRAM, gate on THAT card's free vs. `need`, and on PASS pin
the child to it via `CUDA_VISIBLE_DEVICES` so the child's device 0 really is the card the gate
cleared. `gpu_inventory()` is loaded via importlib (standalone script, not a package — per the
`test_qa_overlay_h264.py` precedent) and `nvidia-smi` is monkeypatched via `sh()`, so this needs no
GPU and no real nvidia-smi.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    modname = f"{name}_under_test"
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # a6000_smoke.py's @dataclass needs its own module registered in sys.modules BEFORE exec
    # (dataclasses resolves annotations via sys.modules[cls.__module__] on 3.12+).
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


smoke = _load("a6000_smoke")


class _Args:
    min_free_vram_gib = 6.0


def _fake_inventory(gpus, holders=()):
    """Build the dict shape `gpu_inventory()` returns, without touching nvidia-smi."""
    return {"gpus": gpus, "holders": list(holders)}


def test_gate_judges_the_gpu_with_most_free_vram_not_a_hardcoded_one(monkeypatch):
    """Two cards, GPU0 saturated / GPU1 idle: the gate must choose GPU1 and pass on ITS headroom."""
    inv = _fake_inventory([
        {"index": 0, "uuid": "GPU-aaaaaaaa-0000-0000-0000-000000000000", "name": "RTX 6000 Ada",
         "total_mib": 49140, "used_mib": 48740, "free_mib": 400},
        {"index": 1, "uuid": "GPU-bbbbbbbb-1111-1111-1111-111111111111", "name": "RTX 6000 Ada",
         "total_mib": 49140, "used_mib": 2140, "free_mib": 47000},
    ])
    monkeypatch.setattr(smoke, "gpu_inventory", lambda: inv)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    captured_env = {}

    def fake_sh(cmd, cwd=None, env=None, timeout=3600):
        captured_env.update(env or {})
        return 0, "OK — bf16 matmul, h3 row/mask ops, toy backward on fake; peak 100 MiB"

    monkeypatch.setattr(smoke, "sh", fake_sh)
    monkeypatch.setattr(smoke, "_stage_python", lambda args, repo: Path("python"))

    result = smoke.stage_gpu(_Args(), REPO_ROOT)

    assert result.status == "PASS", f"expected PASS on GPU1's 47000 MiB free, got: {result}"
    assert "gpu1" in result.detail, f"PASS detail must name the chosen card: {result.detail!r}"
    # Pin by UUID (not the nvidia-smi PCI-order integer index) — CUDA's default enumeration is
    # FASTEST_FIRST, so an integer index can silently resolve to a different physical card.
    assert captured_env.get("CUDA_VISIBLE_DEVICES") == "GPU-bbbbbbbb-1111-1111-1111-111111111111", (
        f"the probe subprocess must be pinned by UUID to the GPU the gate actually cleared "
        f"(CUDA_VISIBLE_DEVICES=GPU-<uuid>, the exact string nvidia-smi's own uuid column "
        f"reports), got env={captured_env!r}. A bare integer index is interpreted in "
        f"CUDA_DEVICE_ORDER=FASTEST_FIRST order by default and can name a different physical "
        f"card than nvidia-smi's PCI-order index."
    )
    assert captured_env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", (
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID must be set alongside the UUID pin, got env={captured_env!r}"
    )


def test_gate_respects_an_operator_exported_cuda_visible_devices(monkeypatch):
    """Operator has already exported CUDA_VISIBLE_DEVICES=1 on a shared box (masking GPU0 out).

    nvidia-smi is CVD-blind and still reports both cards, with GPU0 showing the most free VRAM.
    The gate must NOT choose GPU0 — that would silently override the operator's own export and
    probe a card they deliberately excluded (`sh()` merges {**os.environ, **env}, so a hardcoded
    CVD in the env dict would otherwise trample the operator's value). It must intersect the
    candidate set with what the operator made visible and choose only among those.
    """
    inv = _fake_inventory([
        {"index": 0, "uuid": "GPU-aaaaaaaa-0000-0000-0000-000000000000", "name": "RTX 6000 Ada",
         "total_mib": 49140, "used_mib": 2140, "free_mib": 47000},
        {"index": 1, "uuid": "GPU-bbbbbbbb-1111-1111-1111-111111111111", "name": "RTX 6000 Ada",
         "total_mib": 49140, "used_mib": 20140, "free_mib": 29000},
    ])
    monkeypatch.setattr(smoke, "gpu_inventory", lambda: inv)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")  # operator masked GPU0 out

    captured_env = {}

    def fake_sh(cmd, cwd=None, env=None, timeout=3600):
        captured_env.update(env or {})
        return 0, "OK — bf16 matmul, h3 row/mask ops, toy backward on fake; peak 100 MiB"

    monkeypatch.setattr(smoke, "sh", fake_sh)
    monkeypatch.setattr(smoke, "_stage_python", lambda args, repo: Path("python"))

    result = smoke.stage_gpu(_Args(), REPO_ROOT)

    assert result.status == "PASS", f"expected PASS on the operator-visible GPU1, got: {result}"
    assert "gpu1" in result.detail, (
        f"the gate must choose GPU1 (the only operator-visible card), not GPU0's larger free "
        f"headroom: {result.detail!r}"
    )
    assert captured_env.get("CUDA_VISIBLE_DEVICES") == "GPU-bbbbbbbb-1111-1111-1111-111111111111", (
        f"must pin the child to the operator-visible card (GPU1's uuid), not override the "
        f"operator's export with GPU0: got env={captured_env!r}"
    )


def test_gate_skips_when_operator_cvd_excludes_every_reported_card(monkeypatch):
    """Operator's CVD names a device nvidia-smi never reports (e.g. stale export) — SKIP, don't
    fall back to probing an unlisted card."""
    inv = _fake_inventory([
        {"index": 0, "uuid": "GPU-aaaaaaaa-0000-0000-0000-000000000000", "name": "RTX 6000 Ada",
         "total_mib": 49140, "used_mib": 2140, "free_mib": 47000},
    ])
    monkeypatch.setattr(smoke, "gpu_inventory", lambda: inv)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")

    called = {"sh": False}
    monkeypatch.setattr(smoke, "sh", lambda *a, **k: called.update(sh=True) or (0, ""))

    result = smoke.stage_gpu(_Args(), REPO_ROOT)

    assert result.status == "SKIP"
    assert "CUDA_VISIBLE_DEVICES" in result.detail
    assert not called["sh"], "must never probe when the operator's CVD excludes every card"


def test_gate_skips_when_the_best_card_alone_is_still_under_the_floor(monkeypatch):
    """Both cards busy: SKIP, name the chosen (best) card, and attribute holders to their GPU."""
    inv = _fake_inventory(
        [
            {"index": 0, "name": "A", "total_mib": 49140, "used_mib": 48740, "free_mib": 400},
            {"index": 1, "name": "B", "total_mib": 49140, "used_mib": 47140, "free_mib": 2000},
        ],
        holders=[{"pid": 4242, "name": "python", "used_mib": 47140, "gpu_index": 1}],
    )
    monkeypatch.setattr(smoke, "gpu_inventory", lambda: inv)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    called = {"sh": False}
    monkeypatch.setattr(smoke, "sh", lambda *a, **k: called.update(sh=True) or (0, ""))

    result = smoke.stage_gpu(_Args(), REPO_ROOT)

    assert result.status == "SKIP"
    assert "gpu1" in result.detail, f"SKIP detail must name the best (still-insufficient) card: {result.detail!r}"
    assert "4242" in result.detail and "gpu1" in result.detail, (
        f"holder must be attributed to its actual GPU index: {result.detail!r}"
    )
    assert not called["sh"], "a SKIP must never invoke the GPU probe subprocess"


def test_gpu_inventory_attributes_holders_to_their_gpu_index(monkeypatch):
    """`gpu_inventory()` must join compute-app holders to a GPU index via gpu_uuid, not guess."""
    gpu_csv = "0, GPU-aaa, RTX 6000 Ada, 49140, 400\n1, GPU-bbb, RTX 6000 Ada, 49140, 47140\n"
    holder_csv = "GPU-bbb, 4242, python, 46700\n"

    calls = []

    def fake_sh(cmd, cwd=None, env=None, timeout=3600):
        calls.append(cmd)
        if "--query-gpu=index,uuid,name,memory.total,memory.used" in cmd:
            return 0, gpu_csv
        if "--query-compute-apps=gpu_uuid,pid,process_name,used_memory" in cmd:
            return 0, holder_csv
        raise AssertionError(f"unexpected nvidia-smi invocation: {cmd}")

    monkeypatch.setattr(smoke, "shutil", smoke.shutil)  # keep real shutil.which
    monkeypatch.setattr(smoke.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(smoke, "sh", fake_sh)

    inv = smoke.gpu_inventory()

    assert inv["holders"] == [{"pid": 4242, "name": "python", "used_mib": 46700, "gpu_index": 1}], (
        f"holder was not attributed to gpu index 1 via its gpu_uuid: {inv['holders']!r}"
    )
