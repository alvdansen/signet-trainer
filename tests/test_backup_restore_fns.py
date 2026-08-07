"""Phase 09.1 — BK-01 backup_sync/restore CPU-fn + entrypoint-gate structure (static source-scan).

Zero GPU / zero modal import / zero spend: these tests READ ``modal/fns.py`` and ``modal/entrypoint.py``
source and assert the money-safety + durability invariants WITHOUT importing them (importing pulls
``modal`` and would build the app graph). The backup fns drive real (CPU-cheap) Modal jobs, so their
correctness is verified by static structure — the same discipline as ``tests/test_watcher_hardening.py``.

Asserted invariants:
  * backup_sync + restore EXIST and are CPU-only (no ``gpu=`` in either decorator) — D-BK-3, off the A100.
  * Both carry ``secrets=[huggingface_secret]`` — the HF token comes from the Modal secret ONLY.
  * restore commits the Volume (``checkpoints_vol.commit()``); backup_sync calls ``plan_uploads`` (idempotent).
  * Neither fn body names an HF-token env var (the token is never read from config nor printed).
  * Both fns keep a defense-in-depth ``cloud`` NotImplementedError with the "only 'hf' and 'local'" message.
  * entrypoint: the mode allowlist carries "restore" + "backup"; both ``.spawn()`` dispatches occur
    STRICTLY after ``_require_approval`` in main() (single-gate law — no pre-approval dispatch).
  * Both fns still declare ``image=download_image`` — the link the import-closure gate below depends on.
  * ``download_image`` DECLARES every third-party distribution that ``signet_trainer.config.load``
    imports directly. Re-derived at test time, never hardcoded: both fns call ``load_config_from_text``
    in-container, so an undeclared root is a ModuleNotFoundError inside a PAID container that every
    local gate happily passes (this is not hypothetical — it happened live 2026-08-05 on ``pydantic``).
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"
APP = REPO / "src" / "signet_trainer" / "modal" / "app.py"
ENTRYPOINT = REPO / "src" / "signet_trainer" / "modal" / "entrypoint.py"


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    # Line-wise ``#`` strip so a token named only in a comment can't be mistaken for real code.
    return "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())


def _function_segment(path: Path, name: str) -> str:
    """The FULL source of a top-level function INCLUDING its decorator list (decorator -> end)."""
    src = _src(path)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
            lines = src.splitlines()
            return "\n".join(lines[start : node.end_lineno])
    raise AssertionError(f"no top-level function {name!r} found in {path}")


def _decorator_segment(path: Path, name: str) -> str:
    """Only the ``@app.function(...)`` decorator block of a named function."""
    src = _src(path)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            if not node.decorator_list:
                return ""
            lines = src.splitlines()
            dec_start = min(d.lineno for d in node.decorator_list) - 1
            dec_end = node.lineno - 1  # up to (not incl.) the `def` line
            return "\n".join(lines[dec_start:dec_end])
    raise AssertionError(f"no top-level function {name!r} found in {path}")


def _main_body(path: Path) -> str:
    src = _src(path)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return _strip_comments(ast.get_source_segment(src, node) or "")
    raise AssertionError(f"no main() function found in {path}")


# ---- Task 1: the two CPU fns exist and are money-safe / durable ----


def test_backup_and_restore_functions_exist():
    s = _src(FNS)
    assert "def backup_sync" in s
    assert "def restore" in s


def test_both_fns_are_cpu_only_no_gpu():
    # D-BK-3: neither decorator carries `gpu=` — an upload/restore hang runs in its OWN CPU container
    # and can never wedge a running train().
    for name in ("backup_sync", "restore"):
        dec = _strip_comments(_decorator_segment(FNS, name))
        assert "gpu=" not in dec, f"{name} decorator must be CPU-only (no gpu=)"


def test_both_fns_use_the_huggingface_secret():
    for name in ("backup_sync", "restore"):
        dec = _decorator_segment(FNS, name)
        assert "secrets=[huggingface_secret]" in dec, f"{name} must use the Modal huggingface_secret"


def test_restore_commits_the_volume():
    # commit-or-vanish (D-BK-4): a restore that never commits vanishes on container exit.
    body = _function_segment(FNS, "restore")
    assert "checkpoints_vol.commit()" in body


def test_backup_sync_uses_the_idempotent_plan():
    body = _function_segment(FNS, "backup_sync")
    assert "plan_uploads(" in body


# ---- D-10-DEF-18: the PEFT model card must be excluded at BOTH destinations ----


def test_backup_sync_excludes_the_peft_model_card_at_both_destinations():
    """Both mirror call sites must carry the exclusion, or the H3 mirror dies on the FIRST dir.

    ``huggingface_hub`` validates README front matter SERVER-SIDE before hashing any bytes, and an
    H3 checkpoint's PEFT-written card carries ``base_model: /weights/minimax-h3/transformer_ref``
    (a local Volume path) -> HTTP 400 -> the whole mirror aborts with ZERO uploads. The local branch
    carries the same exclusion so the backup contract stays destination-independent.
    """
    body = _strip_comments(_function_segment(FNS, "backup_sync"))
    assert "MIRROR_EXCLUDED_FILES" in body, (
        "backup_sync must import the excluded-file set from backup.plan (D-10-DEF-18)"
    )
    assert "ignore_patterns=list(excluded)" in body, (
        "the hf branch's upload_folder must pass ignore_patterns (else the H3 mirror 400s), and it "
        "must pass a FRESH list — huggingface_hub does `ignore_patterns += DEFAULT_IGNORE_PATTERNS`, "
        "mutating the CALLER's list, so a shared list grows by the hub defaults once per dir"
    )
    assert "shutil.ignore_patterns(" in body and "ignore=ignore" in body, (
        "the local branch's copytree must apply the same exclusion"
    )


def test_backup_sync_does_not_hardcode_the_excluded_filename():
    """The exclusion is THREADED from the pure constant, never re-typed as a literal here.

    A second literal would drift from ``backup/plan.py`` and slip past the disjointness guard in
    ``tests/test_backup_plan.py``, which only pins the constant.
    """
    body = _strip_comments(_function_segment(FNS, "backup_sync"))
    assert '"README.md"' not in body and "'README.md'" not in body, (
        "backup_sync must derive the exclusion from MIRROR_EXCLUDED_FILES, not a local literal"
    )


def test_upload_folder_still_mutates_its_ignore_patterns_argument():
    """Pins the UPSTREAM behaviour the fresh-copy above defends against, so a hub change is visible.

    If huggingface_hub ever stops mutating the caller's list, this test fails and tells us the
    ``list(excluded)`` copy is no longer load-bearing — better than the copy quietly becoming cargo
    cult. Skipped when huggingface_hub is not installed (it is not a test-time hard dep).
    """
    hf_api = pytest.importorskip("huggingface_hub.hf_api")
    src = inspect.getsource(hf_api.HfApi.upload_folder)
    assert "ignore_patterns += DEFAULT_IGNORE_PATTERNS" in src, (
        "huggingface_hub no longer mutates the caller's ignore_patterns in place — re-evaluate the "
        "list(excluded) defensive copy in backup_sync"
    )


def test_restore_is_untouched_by_the_exclusion():
    """AP-7 / D-BK-4: ``restore`` stays a STRAIGHT copy-back — the fix is upload-side only."""
    body = _strip_comments(_function_segment(FNS, "restore"))
    assert "MIRROR_EXCLUDED_FILES" not in body and "ignore_patterns" not in body, (
        "restore must remain an unfiltered straight copy-back of whatever the mirror holds"
    )


def test_neither_fn_names_an_hf_token_env_var():
    # Info-disclosure guard (T-09.1-08-I): the token comes from the Modal secret env (huggingface_hub
    # reads HF_TOKEN implicitly) — the fns NEVER name a token env var, so none can be read-from-config
    # or printed. Scanned comment-stripped so an explanatory comment can't false-fail.
    forbidden = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN")
    for name in ("backup_sync", "restore"):
        body = _strip_comments(_function_segment(FNS, name))
        for tok in forbidden:
            assert tok not in body, f"{name} must never name the {tok} env var"


def test_both_fns_reject_cloud_as_not_implemented_backstop():
    # Defense-in-depth (T-09.1-08-D2): the primary rejection is 09.1-07's load-time validator; each fn
    # ALSO carries a NotImplementedError backstop with the honest "only 'hf' and 'local'" message.
    for name in ("backup_sync", "restore"):
        body = _function_segment(FNS, name)
        assert "NotImplementedError" in body
        assert "only 'hf' and 'local'" in body


# ---- CR-01: HF backup verifies the destination repo is PRIVATE before any upload ----


def test_backup_sync_verifies_repo_is_private_before_upload():
    # CR-01 (D-BK-5): ``create_repo(private=..., exist_ok=True)`` only honors ``private=`` on CREATE —
    # a pre-existing PUBLIC repo is a no-op and would silently publish likeness checkpoints. The fn
    # MUST call ``repo_info`` and refuse (RuntimeError) when ``backup.private`` is True but the repo is
    # public, and this privacy gate MUST precede the first ``upload_folder`` call.
    body = _strip_comments(_function_segment(FNS, "backup_sync"))
    assert "repo_info(" in body, "backup_sync must verify repo visibility via repo_info()"
    assert ".private" in body, "backup_sync must inspect the repo's .private flag"
    assert "RuntimeError" in body, "backup_sync must ABORT (RuntimeError) on a public repo"
    check_idx = body.find("repo_info(")
    upload_idx = body.find("upload_folder(")
    assert check_idx != -1 and upload_idx != -1
    assert check_idx < upload_idx, "the privacy gate must run BEFORE any upload_folder()"


def test_backup_sync_keys_idempotency_on_remote_completeness():
    # WR-01: idempotency keys on the COMPLETENESS of the remote copy, not the bare dir name. HF walks
    # the tree recursively + uses complete_remote_names; local re-runs is_complete on each dest dir.
    body = _strip_comments(_function_segment(FNS, "backup_sync"))
    assert "recursive=True" in body, "HF listing must be recursive to inspect per-dir file sets"
    assert "complete_remote_names(" in body, "HF branch must gate on complete_remote_names"
    assert "is_complete(" in body, "local branch must re-check destination completeness"
    assert "dirs_exist_ok=True" in body, "local re-upload must repair an incomplete dir in place"


# ---- Task 2: the entrypoint arms join the single gate ----


def test_mode_allowlist_carries_restore_and_backup():
    src = _src(ENTRYPOINT)
    allowlist_line = next(ln for ln in src.splitlines() if "mode not in (" in ln)
    assert '"restore"' in allowlist_line
    assert '"backup"' in allowlist_line


def test_restore_and_backup_dispatch_after_approval():
    # Single-gate law (MODL-02): both `.spawn()` dispatches occur STRICTLY after `_require_approval`
    # in main() — a Volume-mutating restore / a metered backup can never precede the approval pause.
    # (D-10-DEF-17 swapped the verb `.remote` -> `.spawn`; the ORDERING claim is unchanged.)
    body = _main_body(ENTRYPOINT)
    approval_idx = body.find("_require_approval(approve)")
    assert approval_idx != -1, "expected the _require_approval(approve) gate in main()"
    for dispatch in ("restore.spawn(config_text)", "backup_sync.spawn(config_text)"):
        idx = body.find(dispatch)
        assert idx != -1, f"expected {dispatch} in main()"
        assert approval_idx < idx, f"{dispatch} must dispatch AFTER _require_approval"


# ---- BK-01 durability: the shared image must carry both fns' import closure ----


def _normalize_dist(name: str) -> str:
    """PEP-503-ish normalization so ``PyYAML`` / ``huggingface_hub`` / ``pyyaml`` all compare equal."""
    return name.strip().lower().replace("_", "-").replace(".", "-")


def _declared_distributions() -> set[str]:
    """The distributions ``download_image`` pip_installs, parsed out of app.py source.

    Static parse (no ``modal`` import, no image build, no network). The index_url string is dropped
    by the ``/``/``:`` filter, and the ``add_local_python_source`` argument by the name filter.
    """
    block = _strip_comments(_src(APP)).split("download_image = (")[1].split("\n)")[0]
    declared = set()
    for raw in re.findall(r'"([^"]*)"', block):
        if "/" in raw or ":" in raw or raw == "signet_trainer":
            continue
        declared.add(_normalize_dist(re.split(r"[<>=!~ ]", raw)[0]))
    return declared


def test_both_fns_declare_the_download_image():
    # The link the closure gate below rests on: if a fn is ever moved to another image, the
    # closure assertion would silently stop describing what these fns actually run on.
    for name in ("backup_sync", "restore"):
        dec = _strip_comments(_decorator_segment(FNS, name))
        assert "image=download_image" in dec, f"{name} must run on download_image"


def test_download_image_declares_config_load_third_party_roots():
    """``download_image`` must declare every third-party root ``config.load`` imports DIRECTLY.

    The answer is DERIVED here, never hardcoded — a hardcoded list would rot the moment a config
    module grew an import, which is precisely the failure this guards. Reproduce by hand with:

        PYTHONPATH=src python -c "import sys; b=set(sys.modules); import signet_trainer.config.load; \
            print(sorted({m.split('.')[0] for m in set(sys.modules)-b} - sys.stdlib_module_names))"

    Today that yields pydantic, yaml and torch (torch transitively: config/validators.py re-exports
    ``compute_seq_len`` from conditioning/strategy.py, which imports torch at module scope).

    A SUBPROCESS is required: an in-process sys.modules diff returns empty, because another test in
    the same session has almost certainly already imported config.load. Costs a few seconds of torch
    import; zero network, zero Modal, zero spend.
    """
    probe = (
        "import json,sys\n"
        "before=set(sys.modules)\n"
        "import signet_trainer.config.load\n"
        "new=set(sys.modules)-before\n"
        "roots=sorted({m.split('.')[0] for m in new}-set(sys.stdlib_module_names))\n"
        "roots=[r for r in roots if not r.startswith('_') and r!='signet_trainer']\n"
        "files=[getattr(sys.modules[n],'__file__',None) for n in new if n.split('.')[0]=='signet_trainer']\n"
        "print(json.dumps({'roots':roots,'files':[f for f in files if f]}))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONUTF8": "1"}
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True, cwd=REPO
    )
    snapshot = json.loads(out.stdout.strip().splitlines()[-1])

    # Narrow the full closure to the roots signet_trainer imports DIRECTLY. Everything else
    # (numpy, tqdm, pydantic_core, annotated_types, ...) is a pip transitive that arrives for free
    # and must NOT be declared — pinning transitives would be noise that rots.
    sources = [Path(f).read_text(encoding="utf-8") for f in snapshot["files"]]
    direct = [
        root
        for root in snapshot["roots"]
        if any(re.search(rf"^\s*(?:import|from)\s+{re.escape(root)}\b", s, re.MULTILINE) for s in sources)
    ]
    assert direct, "closure derivation produced no direct roots — the probe is broken, not the image"

    # Import name -> distribution name (yaml -> PyYAML). Fall back to the import name when unmapped.
    import importlib.metadata as importlib_metadata

    mapping = importlib_metadata.packages_distributions()
    declared = _declared_distributions()

    missing = []
    for root in direct:
        candidates = {_normalize_dist(d) for d in mapping.get(root, [])} or {_normalize_dist(root)}
        if not (candidates & declared):
            missing.append(f"{root} (distribution: {'/'.join(sorted(candidates))})")

    assert not missing, (
        "download_image does not declare: "
        + ", ".join(missing)
        + f"\ndeclared today: {sorted(declared)}"
        + "\nAdd it to download_image's pip_install in src/signet_trainer/modal/app.py."
        + " backup_sync and restore import config.load INSIDE the container, so the only other way"
        + " you will learn about this is a ModuleNotFoundError in a paid container after every local"
        + " gate has already passed (BK-01, live failure 2026-08-05)."
    )
