"""Plan 10-10 — the SELF-DERIVING import-closure gate for the Phase-10 MiniMax-H3 Modal functions.

Why this file exists, in one sentence: **a passing local gate proves nothing about the container's
site-packages.** ``--mode backup`` passed every local gate — dry-run OK, cost estimate under the
guardrail, approval honored — and then died INSIDE the paid container with
``ModuleNotFoundError: No module named 'pydantic'`` (BK-01, live failure 2026-08-05). The only other
discovery channel for a missing dependency is exactly that: a ``ModuleNotFoundError`` in a container
you are already paying for, after every local check has already gone green.

So this test re-derives the answer instead of asserting a remembered one. **No closure assertion in
this file is driven by a hardcoded root list** — such a list rots the moment a module grows an
import, which is precisely the failure it would be pretending to guard. The handful of literal
package names that do appear below belong to ``test_the_derivation_machinery_is_real...`` only, and
they are of two deliberately different kinds: DECLARED-side fixtures (what ``h3_gpu_image``'s
recipe installs, used to prove the parser reads ``run_commands`` strings at all) and the single
BK-01 canary (``torch``, which reaches ``config.load`` transitively through a re-export). Neither
is a roster of what any function's closure is expected to contain.

What it covers for the H3 leg:

  * every H3 ``@app.function`` that declares ``gpu=`` must declare ``image=h3_gpu_image`` (the
    family image carrying diffusers at the pinned ``DIFFUSERS_SHA``);
  * every H3 ``@app.function`` pointed at a CPU image must have its import closure DECLARED by that
    image. There are none today — the test asserts that fact explicitly and stays green, so it
    starts describing reality the moment one is added rather than being quietly forgotten;
  * the derivation machinery itself is exercised against the known BK-01 pair
    (``download_image`` + ``signet_trainer.config.load``), so a broken probe goes red TODAY rather
    than on the day someone adds the first CPU H3 function;
  * no ``h3_*smoke`` function exists — a bare ``modal run ...::fn`` arch check would skip the cost
    print and the approval pause (Phase 9 ``AUDIT-SYNTHESIS.md`` finding #18).

CPU-only, zero GPU, zero Modal spend. The Modal modules are read as TEXT and parsed with ``ast``;
they are never imported (importing ``modal/fns.py`` builds the app graph and eagerly resolves every
``Secret.from_name``).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"
APP = REPO / "src" / "signet_trainer" / "modal" / "app.py"

#: The GPU image family every H3 metered container must run (plan 10-04, H3-07).
H3_GPU_IMAGE = "h3_gpu_image"

#: Shell verbs and flags that appear inside ``run_commands`` strings alongside the real package
#: specs. Filtering them keeps the "declared" set from becoming a rubber stamp.
_COMMAND_TOKENS = frozenset(
    {
        "uv",
        "pip",
        "install",
        "cd",
        "git",
        "clone",
        "checkout",
        "apt",
        "apt-get",
        "python",
        "python3",
        "bash",
        "sh",
        "echo",
        "mkdir",
        "system",
        "packages",
        "true",
        "false",
    }
)


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_comments_and_docstrings(src: str) -> str:
    """Drop ``# ...`` comments and triple-quoted blocks so prose cannot satisfy a scan."""
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"#.*", "", src)


def _normalize_dist(name: str) -> str:
    """PEP-503-ish normalization so ``PyYAML`` / ``huggingface_hub`` / ``pyyaml`` compare equal."""
    return name.strip().lower().replace("_", "-").replace(".", "-")


# --------------------------------------------------------------------------------------------------
# Discovery — which H3 @app.functions exist, and what image does each declare
# --------------------------------------------------------------------------------------------------


def _app_function_nodes() -> list[ast.FunctionDef]:
    """Every top-level ``@app.function``-decorated function in ``modal/fns.py``."""
    tree = ast.parse(_src(FNS))
    nodes: list[ast.FunctionDef] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute) and target.attr == "function":
                nodes.append(node)
                break
    return nodes


def _decorator_args(node: ast.FunctionDef) -> str:
    """The decorator block's source with comments stripped (prose must not count as a kwarg)."""
    lines = _src(FNS).splitlines()
    start = min(d.lineno for d in node.decorator_list) - 1
    return _strip_comments_and_docstrings("\n".join(lines[start : node.lineno - 1]))


def _h3_app_functions() -> list[tuple[ast.FunctionDef, str]]:
    """The Phase-10 H3 stage functions, discovered by name prefix — never a hardcoded roster."""
    return [(n, _decorator_args(n)) for n in _app_function_nodes() if n.name.startswith("h3_")]


def _raise_segments(node: ast.FunctionDef) -> list[str]:
    """Source of every ``raise`` statement inside a function — where cold-path probes live."""
    src = _src(FNS)
    return [
        ast.get_source_segment(src, stmt) or ""
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Raise)
    ]


def _declared_image(decorator_args: str) -> str | None:
    match = re.search(r"image=(\w+)", decorator_args)
    return match.group(1) if match else None


# --------------------------------------------------------------------------------------------------
# Derivation — the closure a function actually pulls, computed in a subprocess
# --------------------------------------------------------------------------------------------------


def _imported_by(node: ast.FunctionDef) -> tuple[list[str], set[str]]:
    """``(signet_trainer modules, direct third-party roots)`` a function imports IN ITS BODY.

    Both channels matter for a container: the third-party roots are needed directly, and each
    ``signet_trainer`` module drags its own (much larger, much less obvious) closure — which is
    exactly how BK-01 hid: ``config.load`` pulls ``torch`` transitively through a re-export.
    """
    signet: set[str] = set()
    third_party: set[str] = set()
    stdlib = set(sys.stdlib_module_names)
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.ImportFrom) and stmt.module:
            root = stmt.module.split(".")[0]
            if root == "signet_trainer":
                signet.add(stmt.module)
            elif root not in stdlib:
                third_party.add(root)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                root = alias.name.split(".")[0]
                if root == "signet_trainer":
                    signet.add(alias.name)
                elif root not in stdlib:
                    third_party.add(root)
    return sorted(signet), third_party


def _derive_direct_roots(modules: list[str]) -> list[str]:
    """Third-party roots ``modules`` import DIRECTLY, derived in a subprocess. Never hardcoded.

    A SUBPROCESS is required: an in-process ``sys.modules`` diff returns nothing useful, because
    another test in the same session has almost certainly already imported these modules.

    ⚠ The regex here is anchored ``^\\s*`` — the SAME as ``tests/test_backup_restore_fns.py:255-259``
    and deliberately NOT the column-0 anchoring that ``tests/test_h3_preprocess_wiring.py`` uses.
    The two files ask opposite questions. That one asks "what is pulled merely by IMPORTING the
    module", where a function-local import is precisely what the Modal-side EXCEPTION tier ALLOWS.
    THIS one asks "what must the IMAGE declare", and a function-local import needs declaring just as
    much as a module-scope one — it still runs inside the paid container.
    """
    if not modules:
        return []
    probe = (
        "import json,sys\n"
        "before=set(sys.modules)\n"
        f"for name in {modules!r}:\n"
        "    __import__(name)\n"
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

    # Narrow the full closure to the roots signet_trainer imports DIRECTLY. Everything else (numpy,
    # tqdm, annotated_types, ...) is a pip transitive that arrives for free; pinning transitives
    # would be noise that rots.
    sources = [Path(f).read_text(encoding="utf-8") for f in snapshot["files"]]
    return [
        root
        for root in snapshot["roots"]
        if any(
            re.search(rf"^\s*(?:import|from)\s+{re.escape(root)}\b", s, re.MULTILINE)
            for s in sources
        )
    ]


# --------------------------------------------------------------------------------------------------
# Declaration — what an image actually installs, parsed out of app.py
# --------------------------------------------------------------------------------------------------


def _image_block(image_name: str) -> str:
    src = _src(APP)
    marker = f"{image_name} = ("
    assert marker in src, f"no image named {image_name!r} is defined in modal/app.py"
    start = src.index(marker)
    return _strip_comments_and_docstrings(src[start : src.index("\n)", start)])


def _declared_distributions(image_name: str) -> set[str]:
    """Distributions an image installs, parsed from BOTH ``pip_install`` args and command strings.

    Deliberately generous: its job is to catch an UNDECLARED root, so a stray command word landing
    in the set is a far smaller risk than a real package being missed because the parser only knew
    one of the two install shapes (``gpu_image`` and ``h3_gpu_image`` install through
    ``run_commands`` strings; ``download_image`` through ``pip_install`` args).

    Only DOUBLE-quoted literals are scanned, and the single-quoted package specs nested inside them
    are recovered per token. Treating the two quote styles interchangeably mis-pairs them — the
    opening ``"`` closes against the first inner ``'`` — which silently reduces
    ``"uv pip install --system 'peft>=0.14' 'accelerate>=1.2'"`` to the words before ``peft``. That
    is a rubber stamp wearing a parser's clothes; ``test_the_derivation_machinery_is_real...``
    exists precisely to catch it.
    """
    declared: set[str] = set()
    for literal in re.findall(r'"([^"\n]*)"', _image_block(image_name)):
        for raw in literal.split():
            token = raw.strip("'\"")
            if not token or token.startswith("-") or "/" in token or ":" in token:
                continue
            base = re.split(r"[<>=!~@\[]", token)[0].strip()
            if not base or not re.fullmatch(r"[A-Za-z0-9._-]+", base):
                continue
            if base.lower() in _COMMAND_TOKENS:
                continue
            declared.add(_normalize_dist(base))
    return declared


def _undeclared(roots: list[str], image_name: str) -> list[str]:
    """Roots the image does not declare, mapped import-name -> distribution name where possible."""
    import importlib.metadata as importlib_metadata

    mapping = importlib_metadata.packages_distributions()
    declared = _declared_distributions(image_name)
    missing: list[str] = []
    for root in roots:
        candidates = {_normalize_dist(d) for d in mapping.get(root, [])} or {_normalize_dist(root)}
        if not (candidates & declared):
            missing.append(f"{root} (distribution: {'/'.join(sorted(candidates))})")
    return missing


# --------------------------------------------------------------------------------------------------
# The gates
# --------------------------------------------------------------------------------------------------


def test_h3_gpu_functions_declare_the_h3_family_image() -> None:
    """A ``gpu=`` on the code-only default image boots an A100 and then dies at ``import torch``."""
    h3_functions = _h3_app_functions()
    assert h3_functions, (
        "no H3 @app.function found in modal/fns.py — this gate would be describing nothing. If the "
        "H3 leg was renamed, update the discovery prefix rather than deleting the gate."
    )
    offenders = [
        node.name
        for node, args in h3_functions
        if "gpu=" in args and _declared_image(args) != H3_GPU_IMAGE
    ]
    assert not offenders, (
        f"H3 GPU function(s) {offenders} do not declare image={H3_GPU_IMAGE}. The H3 path goes "
        f"through diffusers' MiniMaxH3Transformer3DModel at the pinned DIFFUSERS_SHA, which the LTX "
        f"gpu_image does not carry — a wrong-family image is a metered container guaranteed to fail "
        f"at import."
    )


def test_h3_gpu_stage_roots_are_either_declared_or_cold_path_probed() -> None:
    """A third-party root a GPU stage imports must be DECLARED by its image, or PROBED cold-path.

    The CPU-image gate below is the plan's headline, but the same money lesson applies to a GPU
    stage — more expensively, because an A100 is already ticking. There are two acceptable states
    for a root a stage imports directly:

      * the image installs it (the derived closure is covered); or
      * the stage's FIRST act is a cold-path probe that raises NAMING that root, so the container
        dies in milliseconds with an actionable message instead of after a 61.7 GiB load.

    Today ``h3_preprocess`` imports ``av`` (video decode) which ``h3_gpu_image`` deliberately does
    NOT declare: plan 10-04 shipped that image without ffmpeg or a decode package because nothing
    on the H3 path demuxed anything at the time. The real fix — adding ``av`` (+ ffmpeg for the
    audio leg) to ``h3_gpu_image`` — is a supply-chain-gated change to ``modal/app.py`` owned by
    whoever wires the first real H3 preprocess dispatch. Until then the cold-path probe is what
    keeps the gap a ~$0.02 abort instead of a burned container, and this test is what keeps the gap
    VISIBLE rather than remembered.

    Scope note: only the stage function's OWN body imports are checked. A probe can only guard what
    the stage itself reaches first, and roots pulled by module-level helpers are covered by the
    ``signet_trainer`` closure derivation the CPU gate performs.
    """
    failures: list[str] = []
    for node, args in _h3_app_functions():
        if "gpu=" not in args:
            continue
        image_name = _declared_image(args)
        assert image_name, f"{node.name} declares gpu= with no image="
        _, direct_roots = _imported_by(node)
        raises = _raise_segments(node)
        for entry in _undeclared(sorted(direct_roots), image_name):
            root = entry.split(" ")[0]
            if not any(re.search(rf"\b{re.escape(root)}\b", segment) for segment in raises):
                failures.append(
                    f"{node.name} imports {root!r}, {image_name} does not declare it, and no raise "
                    f"in the function names it"
                )
    assert not failures, (
        "\n".join(failures)
        + f"\nEither add the distribution to that image in {APP}, or add a cold-path import probe"
        + " that raises naming exc.name BEFORE any model load. Silence is the one option that is"
        + " not available: it spends A100 minutes to reach a guaranteed ImportError."
    )


def test_cpu_h3_functions_have_their_import_closure_declared() -> None:
    """Every H3 ``@app.function`` on a CPU image must have its closure DECLARED by that image.

    Today there are none, and this test says so out loud instead of silently passing over an empty
    loop — so the moment a CPU H3 function is added (a manifest builder, a Volume lister, a stager)
    it is covered without anyone having to remember this file exists.
    """
    h3_functions = _h3_app_functions()
    cpu_functions = [(node, args) for node, args in h3_functions if "gpu=" not in args]

    if not cpu_functions:
        gpu_names = sorted(node.name for node, args in h3_functions if "gpu=" in args)
        assert gpu_names, (
            "every H3 @app.function is neither GPU nor CPU — the discovery is broken, not the code"
        )
        assert all(_declared_image(args) == H3_GPU_IMAGE for _, args in h3_functions), (
            "the 'no CPU H3 functions' claim only holds while every H3 function is on the H3 GPU "
            "image; one pointed at a CPU image would need its closure derived below"
        )
        return

    failures: list[str] = []
    for node, args in cpu_functions:
        image_name = _declared_image(args)
        assert image_name, f"{node.name} declares no image= at all — it would run on the code-only default"
        signet_modules, direct_third_party = _imported_by(node)
        roots = sorted(set(_derive_direct_roots(signet_modules)) | direct_third_party)
        missing = _undeclared(roots, image_name)
        if missing:
            failures.append(
                f"{node.name} on {image_name} does not declare: {', '.join(missing)} "
                f"(derived closure: {roots}; declared: {sorted(_declared_distributions(image_name))})"
            )

    assert not failures, (
        "\n".join(failures)
        + "\nAdd the missing distribution(s) to that image in src/signet_trainer/modal/app.py."
        + " These functions import in-container, so the ONLY other way you will learn about this is"
        + " a ModuleNotFoundError inside a PAID container after every local gate has already passed"
        + " (BK-01, live failure 2026-08-05)."
    )


def test_the_derivation_machinery_is_real_not_a_rubber_stamp() -> None:
    """Exercise the probe + the parser on the KNOWN BK-01 pair, so a broken gate goes red today.

    Without this, the loop above would pass vacuously forever (there are no CPU H3 functions yet)
    and a silently broken derivation would only surface on the day it was first needed — in a paid
    container, which is the exact failure mode this whole file exists to move earlier.
    """
    roots = _derive_direct_roots(["signet_trainer.config.load"])
    assert roots, "the subprocess closure probe returned nothing — the probe is broken"
    assert "torch" in roots, (
        "config.load's closure must include torch: config/validators.py re-exports compute_seq_len "
        "from conditioning/strategy.py, which imports torch at module scope. It arrives "
        "TRANSITIVELY through a re-export, which is exactly why it is easy to miss by eye"
    )
    assert not _undeclared(roots, "download_image"), (
        "download_image must declare config.load's whole direct closure — this is the live BK-01 "
        "case, and if this assertion ever fails the parser (not the image) is the first suspect"
    )

    h3_declared = _declared_distributions(H3_GPU_IMAGE)
    for package in ("diffusers", "peft", "torch", "transformers"):
        assert _normalize_dist(package) in h3_declared, (
            f"the image-package parser did not find {package!r} in {H3_GPU_IMAGE} — it installs "
            f"through run_commands strings, so a parser that only understands pip_install args "
            f"would silently report an empty declared set and rubber-stamp everything"
        )


def test_no_h3_smoke_entry_point_bypasses_the_single_gate() -> None:
    """Structural: no Phase-10 function may be reachable by a bare ``modal run ...::fn`` arch check.

    ``load_ltxv_smoke`` is reachable that way, and that invocation style boots a metered A100 with
    NO cost print and NO approval pause — a documented, still-open defect (Phase 9
    ``AUDIT-SYNTHESIS.md`` finding #18), not a precedent to extend. The H3 arch gate is a plain
    module-level helper called from inside stages that already passed the cost print and approval.
    """
    code = _strip_comments_and_docstrings(_src(FNS))
    offenders = re.findall(r"^def\s+(h3_\w*smoke\w*)\s*\(", code, re.M)
    assert not offenders, (
        f"modal/fns.py defines {offenders}, which `modal run -m signet_trainer.modal.fns::<name>` "
        f"can launch directly — skipping the cost print and the blocking approval pause. Phase 10 "
        f"adds no second ungated entry point (Phase 9 AUDIT #18)."
    )
    assert "def run_h3_arch_gate" in code, (
        "the arch gate must still exist as a plain helper — deleting it would satisfy the negative "
        "assertion above while removing abort-before-spend entirely"
    )
