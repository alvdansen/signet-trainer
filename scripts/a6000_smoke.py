#!/usr/bin/env python3
"""a6000_smoke — the on-metal smoke suite for signet-trainer (shared-GPU polite).

Runs the repo's own CPU-runnable gates end to end on a workstation GPU box (the house
RTX 6000 Ada hosts), plus a σ-schedule instrumentation stage and an optional
GPU sanity stage that only runs when the GPU is actually free.

    python3 scripts/a6000_smoke.py                 # all stages, defaults
    python3 scripts/a6000_smoke.py --stages env,configs,sampler
    python3 scripts/a6000_smoke.py --json-out smoke.json

Stages (in order; each independently PASS / FAIL / SKIP):

  env      python / disk / GPU inventory. Detects a busy GPU and reports the holder
           PIDs — this script NEVER kills anything (shared box; stale-VRAM discipline
           applies only to processes we own, and we own none here).
  venv     create/reuse a smoke venv (``--venv``, default ``.venv-smoke`` in the repo
           root) with ``--system-site-packages`` so an existing CUDA torch is reused,
           then ``pip install -e .[dev]``. Skipped with ``--skip-install``.
  configs  schema-load every ``configs/*.yaml`` (load-only; per-file verdicts).
  dryrun   ``signet-dryrun`` over every ``configs/*.example.yaml`` — the repo's
           CPU-pure shape gate, zero GPU, zero Modal.
  unit     full ``pytest`` run, with failures CLASSIFIED against the documented
           red-by-design buckets from CONTRIBUTING.md ("Running the tests — what is
           expected to fail"). Expected-red failures are reported and tolerated;
           ANY failure outside the buckets fails the stage — that is the
           "outside them is a real bug" contract, executable.
  sampler  CPU-pure σ-visit instrumentation for BOTH families: the LTX logit-normal
           schedule and the H3 exponential-shift draw, tabulated against the H3
           25-step deployment grid (the issue-#13 step-1 instrumentation, landed as
           a reusable tool). Informational bands + drift warnings; fails only on
           mechanical error (NaN / out-of-domain), never on distribution shape.
  gpu      tiny CUDA sanity — bf16 matmul, H3 row-timestep/loss-mask ops on device,
           a toy backward — ONLY when free VRAM >= ``--min-free-vram-gib`` (default
           6.0) AND torch sees CUDA. A busy GPU yields SKIP (with holders listed),
           never a kill, never a wait.

Config-first: every tunable is a flag with the default documented here. Exit code 0
iff no stage FAILED (SKIPs are honest and do not fail the run).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# CONTRIBUTING.md "expected to fail on a fresh clone" buckets, encoded.
# Bucket 1 (requires_live_harness) SKIPS on its own — no entry needed here.
# Bucket 2: tests pinned to unpublished in-house campaign configs / watcher scripts.
# Bucket 3: H3 parity suites that REFUSE to skip until the overlay venv is bootstrapped;
#           they are expected-red only while the overlay is absent.
# --------------------------------------------------------------------------------------
EXPECTED_RED_FILES: dict[str, str] = {
    "test_h3_sample_resume.py": "bucket-2: in-house campaign config",
    "test_h3_sample_render_residency.py": "bucket-2: in-house campaign config",
    "test_h3_reference_selector.py": "bucket-2: in-house campaign config",
    "test_h3_reference_geometry_agreement.py": "bucket-2: in-house campaign config",
    "test_h3_pipeline_local_source.py": "bucket-2: in-house campaign config",
    "test_watcher_hardening.py": "bucket-2: campaign watcher script not shipped",
    "test_session_cap_ledger_runtime.py": "bucket-2: campaign watcher script not shipped",
}
PARITY_RED_FILES: dict[str, str] = {
    "test_h3_real_class_parity.py": "bucket-3: parity overlay venv not bootstrapped",
    "test_h3_processor_output_parity.py": "bucket-3: parity overlay venv not bootstrapped",
}

FAILED_LINE = re.compile(r"^(?:FAILED|ERROR) (tests[/\\][\w./\\-]+\.py)(?:::|\s|$)")


@dataclass
class StageResult:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str = ""
    data: dict = field(default_factory=dict)


def sh(cmd: list[str] | str, cwd: Path | None = None, env: dict | None = None,
       timeout: int = 3600) -> tuple[int, str]:
    """Run a command, capture combined output."""
    merged = {**os.environ, **(env or {})}
    p = subprocess.run(
        cmd, cwd=cwd, env=merged, shell=isinstance(cmd, str),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
    )
    return p.returncode, p.stdout


def gpu_inventory() -> dict:
    """nvidia-smi probe: totals, free, and compute-app holders (attributed to a GPU index where
    nvidia-smi can tell us). Empty dict if no smi."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    rc, out = sh([smi, "--query-gpu=index,uuid,name,memory.total,memory.used",
                  "--format=csv,noheader,nounits"])
    if rc != 0:
        return {}
    gpus = []
    uuid_to_index = {}
    for line in out.strip().splitlines():
        idx, uuid, name, total, used = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "uuid": uuid, "name": name, "total_mib": int(total),
                     "used_mib": int(used), "free_mib": int(total) - int(used)})
        uuid_to_index[uuid] = int(idx)
    # gpu_uuid lets a holder be attributed to the card it is actually sitting on (issue #40
    # finding 3) — without it, a SKIP on a busy card cannot tell the operator which one is busy.
    rc, out = sh([smi, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                  "--format=csv,noheader,nounits"])
    holders = []
    if rc == 0:
        for line in out.strip().splitlines():
            if line.strip():
                uuid, pid, pname, mem = [x.strip() for x in line.split(",")]
                holders.append({"pid": int(pid), "name": pname, "used_mib": int(mem),
                                "gpu_index": uuid_to_index.get(uuid)})
    return {"gpus": gpus, "holders": holders}


def stage_env(args, repo: Path) -> StageResult:
    info: dict = {"python": sys.version.split()[0], "platform": sys.platform}
    du = shutil.disk_usage(repo)
    info["disk_free_gib"] = round(du.free / 2**30, 1)
    info["gpu"] = gpu_inventory()
    lines = [f"python {info['python']} on {info['platform']}, "
             f"{info['disk_free_gib']} GiB disk free"]
    for g in info["gpu"].get("gpus", []):
        lines.append(f"gpu{g['index']} {g['name']}: {g['free_mib']} MiB free"
                     f" / {g['total_mib']} MiB")
    for h in info["gpu"].get("holders", []):
        lines.append(f"  holder pid {h['pid']} ({h['name']}) {h['used_mib']} MiB"
                     " — shared box, NOT ours to touch")
    if info["disk_free_gib"] < args.min_disk_gib:
        return StageResult("env", "FAIL",
                           f"disk {info['disk_free_gib']} GiB < --min-disk-gib "
                           f"{args.min_disk_gib}; " + "; ".join(lines), info)
    return StageResult("env", "PASS", "; ".join(lines), info)


def venv_python(repo: Path, venv: str) -> Path:
    d = repo / venv
    return d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def stage_venv(args, repo: Path) -> StageResult:
    if args.skip_install:
        py = Path(args.python) if args.python else Path(sys.executable)
        return StageResult("venv", "SKIP", f"--skip-install; later stages use {py}")
    vp = venv_python(repo, args.venv)
    if not vp.exists():
        rc, out = sh([sys.executable, "-m", "venv", "--system-site-packages",
                      str(repo / args.venv)])
        if rc != 0:
            return StageResult("venv", "FAIL", f"venv create failed:\n{out[-800:]}")
    rc, out = sh([str(vp), "-m", "pip", "install", "-q", "-e", ".[dev]"], cwd=repo,
                 timeout=1800)
    if rc != 0:
        return StageResult("venv", "FAIL", f"pip install -e .[dev] failed:\n{out[-1200:]}")
    rc, out = sh([str(vp), "-c",
                  "import torch, pydantic, yaml, signet_trainer; "
                  "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"])
    if rc != 0:
        return StageResult("venv", "FAIL", f"import sanity failed:\n{out[-800:]}")
    return StageResult("venv", "PASS", out.strip().splitlines()[-1])


def _stage_python(args, repo: Path) -> Path:
    if args.skip_install:
        return Path(args.python) if args.python else Path(sys.executable)
    return venv_python(repo, args.venv)


def stage_configs(args, repo: Path) -> StageResult:
    py = _stage_python(args, repo)
    yamls = sorted((repo / "configs").glob("*.yaml"))
    if not yamls:
        return StageResult("configs", "FAIL", "no configs/*.yaml found")
    expect_invalid = {s.strip() for s in args.expect_invalid.split(",") if s.strip()}
    # Tool configs (the repo's own namespace convention: prep_*.yaml feeds
    # signet_trainer.prep.* fail-loud loaders, not the run-config schema) are checked
    # as parseable YAML only and reported separately, never against load_config.
    import fnmatch
    tool_globs = [g.strip() for g in args.tool_config_glob.split(",") if g.strip()]
    prog = (
        "import sys\n"
        "from signet_trainer.config.load import load_config\n"
        "try:\n"
        "    load_config(sys.argv[1]); print('OK')\n"
        "except Exception as e:\n"
        "    print(f'LOAD-FAIL: {type(e).__name__}: {e}'); sys.exit(1)\n"
    )
    yaml_prog = (
        "import sys, yaml\n"
        "yaml.safe_load(open(sys.argv[1], encoding='utf-8')); print('YAML-OK (tool config)')\n"
    )
    results, failed, skipped, guarded = {}, [], [], []
    for y in yamls:
        is_tool = any(fnmatch.fnmatch(y.name, g) for g in tool_globs)
        rc, out = sh([str(py), "-c", yaml_prog if is_tool else prog, str(y)], cwd=repo,
                     env={"PYTHONPATH": "src", "PYTHONUTF8": "1"})
        tail = out.strip().splitlines()[-1] if out.strip() else f"rc={rc}"
        results[y.name] = tail
        if is_tool:
            (skipped if rc == 0 else failed).append(y.name)
        elif y.name in expect_invalid:
            # INVERTED assertion: this file exists to prove the validator fires.
            if rc == 0:
                failed.append(f"{y.name} (expected-INVALID but LOADED — validator regression)")
            else:
                guarded.append(y.name)
        elif rc != 0:
            failed.append(y.name)
    n_ok = len(yamls) - len(failed) - len(skipped) - len(guarded)
    detail = (f"{n_ok} load OK, {len(guarded)} expected-invalid correctly rejected, "
              f"{len(skipped)} tool-config (YAML-parse only), {len(failed)} FAILED")
    if failed:
        detail += ": " + ", ".join(failed)
    return StageResult("configs", "FAIL" if failed else "PASS", detail, results)


def stage_dryrun(args, repo: Path) -> StageResult:
    py = _stage_python(args, repo)
    expect_invalid = {s.strip() for s in args.expect_invalid.split(",") if s.strip()}
    yamls = [y for y in sorted((repo / "configs").glob(args.dryrun_glob))
             if y.name not in expect_invalid]  # covered by configs' inverted assertion
    if not yamls:
        return StageResult("dryrun", "SKIP", f"nothing matches --dryrun-glob {args.dryrun_glob}")
    results, failed = {}, []
    for y in yamls:
        rc, out = sh([str(py), "-m", "signet_trainer.dryrun", str(y)], cwd=repo,
                     env={"PYTHONPATH": "src", "PYTHONUTF8": "1"}, timeout=600)
        tail = out.strip().splitlines()[-1] if out.strip() else f"rc={rc}"
        results[y.name] = tail
        if rc != 0:
            failed.append(f"{y.name} ({tail[:120]})")
    detail = f"{len(yamls) - len(failed)}/{len(yamls)} dryruns pass"
    if failed:
        detail += "; FAILED: " + "; ".join(failed)
    return StageResult("dryrun", "FAIL" if failed else "PASS", detail, results)


def stage_unit(args, repo: Path) -> StageResult:
    py = _stage_python(args, repo)
    parity_bootstrapped = (repo / ".venv-h3parity").exists() or bool(
        os.environ.get("SIGNET_H3_REALCLASS_PYTHON"))
    rc, out = sh([str(py), "-m", "pytest", "-q", "--tb=no", *args.pytest_args.split()],
                 cwd=repo, env={"PYTHONPATH": "src", "PYTHONUTF8": "1"},
                 timeout=args.pytest_timeout)
    failures: list[str] = []
    for line in out.splitlines():
        m = FAILED_LINE.match(line.strip())
        if m:
            failures.append(m.group(1).replace("\\", "/"))
    expected, unexpected = [], []
    for f in failures:
        base = f.split("/")[-1]
        if base in EXPECTED_RED_FILES:
            expected.append(f"{base} [{EXPECTED_RED_FILES[base]}]")
        elif base in PARITY_RED_FILES and not parity_bootstrapped:
            expected.append(f"{base} [{PARITY_RED_FILES[base]}]")
        else:
            unexpected.append(f)
    tally = re.compile(r"\d+ (?:passed|failed|error|skipped|deselected)")
    summary = next((ln.strip() for ln in reversed(out.strip().splitlines())
                    if tally.search(ln)), out.strip().splitlines()[-1] if out.strip() else "no output")
    data = {"summary": summary, "expected_red": sorted(set(expected)),
            "unexpected_red": sorted(set(unexpected)),
            "parity_bootstrapped": parity_bootstrapped}
    if unexpected:
        return StageResult(
            "unit", "FAIL",
            f"{summary} — {len(set(unexpected))} failure file(s) OUTSIDE the documented "
            f"red-by-design buckets (CONTRIBUTING: 'that is a real bug'): "
            + ", ".join(sorted(set(unexpected))), data)
    if rc != 0 and not failures:
        return StageResult("unit", "FAIL", f"pytest rc={rc} with no parsable failures: {summary}", data)
    return StageResult("unit", "PASS",
                       f"{summary}; expected-red files: {len(set(expected))}"
                       + ("" if parity_bootstrapped else " (parity overlay not bootstrapped)"),
                       data)


_SAMPLER_PROG = r"""
import json, math, sys
import numpy as np
from signet_trainer.train.h3_step import h3_draw_timesteps, h3_shifted_sigma
from signet_trainer.train.flow_match import FlowMatchingSchedule

N, SEED = int(sys.argv[1]), int(sys.argv[2])
bands = [0.5, 0.7, 0.9, 0.95]

def band_table(sigmas):
    s = np.asarray(sigmas)
    if np.isnan(s).any() or (s < 0).any() or (s > 1).any():
        raise SystemExit("MECHANICAL-FAIL: sampler produced NaN or out-of-domain sigma")
    return {f"P(sig<{b})": round(float((s < b).mean()), 4) for b in bands} | {
        "median": round(float(np.median(s)), 4)}

out = {}
# H3 deployment reference: linspace(1,0,25) through the exponential shift-12 map
grid = [h3_shifted_sigma(1.0 - i / 24.0, 12.0) for i in range(25)]
out["h3_deploy_grid_25step"] = {f"P(sig<{b})": round(sum(g < b for g in grid) / 25, 4)
                                for b in bands}
for up in (0.0, 0.30):
    rng = np.random.default_rng(SEED)
    draws = [h3_draw_timesteps(rng, uniform_prob=up) for _ in range(N)]
    out[f"h3_video_sigma_up{up}"] = band_table([1.0 - t for t, _ in draws])
    out[f"h3_audio_sigma_up{up}"] = band_table([1.0 - t for _, t in draws])
sched = FlowMatchingSchedule()
rng = np.random.default_rng(SEED)
for seq_len in (1024, 4096):
    t = sched.sample_timesteps(N, seq_len, rng)
    out[f"ltx_sigma_seq{seq_len}"] = band_table(np.asarray(t))
print(json.dumps(out))
"""


def stage_sampler(args, repo: Path) -> StageResult:
    py = _stage_python(args, repo)
    rc, out = sh([str(py), "-c", _SAMPLER_PROG, str(args.sampler_draws), "42"],
                 cwd=repo, env={"PYTHONPATH": "src", "PYTHONUTF8": "1"}, timeout=600)
    if rc != 0:
        return StageResult("sampler", "FAIL", out.strip()[-800:])
    data = json.loads(out.strip().splitlines()[-1])
    ref = data["h3_deploy_grid_25step"]
    ours = data["h3_video_sigma_up0.3"]
    drift = {k: round(ours[k] - ref[k], 4) for k in ref}
    data["h3_video_vs_deploy_drift_up0.3"] = drift
    warn = [f"{k} drifts {v:+.3f} vs deployment" for k, v in drift.items()
            if abs(v) >= args.sampler_warn_drift]
    detail = (f"{args.sampler_draws} draws/family instrumented; "
              f"H3-video-vs-deploy drift (uniform_prob=0.30): "
              + ", ".join(f"{k} {v:+.3f}" for k, v in drift.items()))
    if warn:
        detail += " | WARN (informational, issue #13): " + "; ".join(warn)
    return StageResult("sampler", "PASS", detail, data)


_GPU_PROG = r"""
import torch
from signet_trainer.train.h3_step import h3_row_timesteps, h3_loss_mask
assert torch.cuda.is_available(), "cuda not available to torch"
dev = torch.device("cuda")
a = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
b = (a @ a).float().sum()
assert torch.isfinite(b), "bf16 matmul produced non-finite"
seq = 128
vid = torch.arange(0, 96, device=dev)
aud = torch.arange(96, 128, device=dev)
row_t, ts, ti = h3_row_timesteps(seq, vid, aud, 16, 8, 0.42, 0.77, device=dev)
assert row_t.shape == (seq,) and ts[ti].allclose(row_t), "row-timestep reconstruction failed on cuda"
vm, am = h3_loss_mask(16, 80, 8, 24, device=dev)
assert vm.sum().item() == 80 and am.sum().item() == 0, "loss-mask counts wrong on cuda"
lin = torch.nn.Linear(256, 256, device=dev, dtype=torch.bfloat16)
loss = lin(torch.randn(8, 256, device=dev, dtype=torch.bfloat16)).float().pow(2).mean()
loss.backward()
assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad.float()).all()
peak = torch.cuda.max_memory_allocated() / 2**20
print(f"OK — bf16 matmul, h3 row/mask ops, toy backward on {torch.cuda.get_device_name(0)}; "
      f"peak {peak:.0f} MiB")
"""


def _operator_visible_gpus(gpus: list[dict], cvd_value: str) -> list[dict]:
    """Intersect `gpus` with an operator-exported CUDA_VISIBLE_DEVICES.

    nvidia-smi is CVD-blind (it always enumerates every physical card), but the CUDA runtime
    inside any child process enforces the operator's export — so if we ignore it here, the gate
    can choose and pin a card the operator deliberately excluded on a shared box. `cvd_value` may
    list either integer indices or UUIDs (with or without the "GPU-" prefix); match on either.
    """
    tokens = [t.strip() for t in cvd_value.split(",") if t.strip()]
    if not tokens:
        return gpus
    ids = set(tokens)

    def _visible(g: dict) -> bool:
        uuid = g.get("uuid") or ""
        bare_uuid = uuid[len("GPU-"):] if uuid.startswith("GPU-") else uuid
        return str(g["index"]) in ids or uuid in ids or bare_uuid in ids

    return [g for g in gpus if _visible(g)]


def stage_gpu(args, repo: Path) -> StageResult:
    inv = gpu_inventory()
    if not inv.get("gpus"):
        return StageResult("gpu", "SKIP", "no nvidia-smi / no GPU")
    candidates = inv["gpus"]
    operator_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if operator_cvd is not None:
        candidates = _operator_visible_gpus(candidates, operator_cvd)
        if not candidates:
            return StageResult("gpu", "SKIP",
                               f"CUDA_VISIBLE_DEVICES={operator_cvd!r} (operator-exported) "
                               f"excludes every card nvidia-smi reports; shared box, respecting "
                               f"the operator's pin rather than overriding it")
    # Issue #40 finding 3: the "shared-GPU polite" gate must judge the SAME card the probe runs
    # on. `_GPU_PROG` always opens `torch.device("cuda")`, i.e. whichever index the child process
    # sees as device 0 — so the gate picks the card with the MOST free VRAM, compares THAT card's
    # free against `need`, and (on PASS) pins the child to it with CUDA_VISIBLE_DEVICES so the
    # child's "cuda:0" really is the card the gate just cleared. Taking max() across all cards
    # while probing an unrelated, hardcoded device let the stage PASS on an idle card's headroom
    # while allocating on a saturated one. Candidates are first restricted to what the operator
    # already made visible (see _operator_visible_gpus), so the gate never picks a masked card.
    chosen = max(candidates, key=lambda g: g["free_mib"])
    free = chosen["free_mib"]
    need = int(args.min_free_vram_gib * 1024)
    if free < need:
        holders = ", ".join(
            f"pid {h['pid']} ({h['used_mib']} MiB, gpu{h['gpu_index']})"
            if h.get("gpu_index") is not None else f"pid {h['pid']} ({h['used_mib']} MiB)"
            for h in inv.get("holders", [])
        )
        return StageResult("gpu", "SKIP",
                           f"gpu{chosen['index']} (most free): only {free} MiB free < {need} MiB "
                           f"required — GPU busy ({holders or 'holder unknown'}); shared box, "
                           f"not killing anything")
    py = _stage_python(args, repo)
    # Pin by UUID, not the nvidia-smi PCI-order index: CUDA's default enumeration order is
    # FASTEST_FIRST, so an integer index can resolve to a different physical card than the one
    # nvidia-smi (and this gate) numbered — recreating the finding-3 bug on a heterogeneous box.
    # nvidia-smi's own uuid column already comes back "GPU-<hex-uuid>" (that IS the accepted
    # CUDA_VISIBLE_DEVICES form) — do not re-prefix it. CUDA_DEVICE_ORDER=PCI_BUS_ID is set too,
    # belt-and-suspenders, in case anything downstream ever falls back to an integer form.
    rc, out = sh([str(py), "-c", _GPU_PROG], cwd=repo,
                 env={"PYTHONPATH": "src", "PYTHONUTF8": "1",
                      "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                      "CUDA_VISIBLE_DEVICES": chosen["uuid"]}, timeout=600)
    if rc != 0:
        return StageResult("gpu", "FAIL", out.strip()[-800:])
    return StageResult("gpu", "PASS", f"gpu{chosen['index']}: " + out.strip().splitlines()[-1])


STAGES = {
    "env": stage_env, "venv": stage_venv, "configs": stage_configs,
    "dryrun": stage_dryrun, "unit": stage_unit, "sampler": stage_sampler,
    "gpu": stage_gpu,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="repo root (default: this script's grandparent)")
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma list from: {','.join(STAGES)} (default: all)")
    ap.add_argument("--venv", default=".venv-smoke",
                    help="smoke venv dir name in the repo root (default .venv-smoke; "
                         "created with --system-site-packages to reuse system torch)")
    ap.add_argument("--skip-install", action="store_true",
                    help="skip the venv stage; run later stages with --python/current python")
    ap.add_argument("--python", default=None,
                    help="interpreter for stages when --skip-install (default: current)")
    ap.add_argument("--dryrun-glob", default="*.example.yaml",
                    help="configs glob for the dryrun stage (default *.example.yaml)")
    ap.add_argument("--tool-config-glob", default="prep_*.yaml",
                    help="comma list of filename globs for TOOL configs on their own schema "
                         "(the repo convention: prep_*.yaml feeds signet_trainer.prep.* "
                         "loaders); YAML-parse-checked only (default prep_*.yaml)")
    ap.add_argument("--expect-invalid", default="bad_frames.example.yaml",
                    help="comma list of configs that MUST fail validation (shipped invalid "
                         "examples, e.g. the D-13 hard-gate driver); loading successfully "
                         "is the failure (default: bad_frames.example.yaml)")
    ap.add_argument("--pytest-args", default="",
                    help="extra args appended to the pytest invocation")
    ap.add_argument("--pytest-timeout", type=int, default=2400,
                    help="unit-stage timeout seconds (default 2400)")
    ap.add_argument("--sampler-draws", type=int, default=100_000,
                    help="draws per family for the sampler stage (default 100000)")
    ap.add_argument("--sampler-warn-drift", type=float, default=0.03,
                    help="absolute band-probability drift that triggers the informational "
                         "issue-#13 warning (default 0.03)")
    ap.add_argument("--min-free-vram-gib", type=float, default=6.0,
                    help="free VRAM required to run the gpu stage (default 6.0)")
    ap.add_argument("--min-disk-gib", type=float, default=5.0,
                    help="free disk below which env FAILS (default 5.0)")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="write the full results as JSON to this path")
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        ap.error(f"unknown stage(s): {unknown}")

    print(f"[a6000_smoke] repo: {repo}")
    results: list[StageResult] = []
    for name in wanted:
        print(f"[a6000_smoke] --- stage: {name} ---", flush=True)
        try:
            r = STAGES[name](args, repo)
        except Exception as exc:  # a stage crashing is a FAIL, not a crash of the suite
            r = StageResult(name, "FAIL", f"stage raised {type(exc).__name__}: {exc}")
        results.append(r)
        print(f"[a6000_smoke] {name}: {r.status} — {r.detail}", flush=True)

    print("\n[a6000_smoke] ================= SUMMARY =================")
    for r in results:
        print(f"  {r.name:8s} {r.status:5s} {r.detail[:160]}")
    failed = [r for r in results if r.status == "FAIL"]
    print(f"[a6000_smoke] {'FAIL' if failed else 'PASS'} — "
          f"{sum(r.status == 'PASS' for r in results)} pass / "
          f"{len(failed)} fail / {sum(r.status == 'SKIP' for r in results)} skip")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            [{"stage": r.name, "status": r.status, "detail": r.detail, "data": r.data}
             for r in results], indent=2), encoding="utf-8")
        print(f"[a6000_smoke] json written: {args.json_out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
