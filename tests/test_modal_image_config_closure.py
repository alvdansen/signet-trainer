"""EVERY Modal image must carry the import closure of the functions pointed at it.

The class, and it has now been paid for TWICE with the identical error message
-----------------------------------------------------------------------------

::

    ModuleNotFoundError: No module named 'pydantic'

* **2026-08-05, ``download_image``** — ``--mode backup`` passed every local gate (dry-run OK, cost
  under the guardrail, approval honoured) and died inside the PAID container. BK-01 backup and
  restore had never once executed.
* **2026-08-06, ``h3_gpu_image``** — ``--mode train`` on the H3 leg, immediately after a *successful*
  88-sample ``--mode preprocess`` on the same image. `h3_preprocess` takes every parameter threaded
  in by the entrypoint and so never loads a config; ``h3_train`` and ``h3_sample`` both call
  ``load_config_from_text`` in-container (the T-03-63 revalidate-in-container convention). **A green
  Stage 1 is therefore not evidence about Stage 2, even on the same image.**

The 2026-08-05 fix came with a derivation-based guard — and it was scoped to ``download_image``, so
it could not have caught the second one. This file is that guard generalized: it asks the question of
**every** image, driven by which functions actually load a config, so image number three fails on the
day it is written rather than in a metered container.

Why "it works today" is not the standard
----------------------------------------
``gpu_image`` never declared ``pydantic`` either — it receives it as a transitive of ``ltx-trainer``,
editable-installed from a cloned upstream repo at a pinned SHA. That is the same defect one step
removed: a first-party import satisfied by a third party's dependency list, which an upstream SHA
bump may drop without warning. It is registered below **with a written reason**, not silently
accepted, and the registry is enforced in BOTH directions so the entry cannot outlive its cause.

Zero network, zero Modal, zero spend. The closure is DERIVED in a subprocess, never hardcoded — a
hardcoded list would rot the moment a config module grew an import, which is precisely the failure
this guards.
"""

from __future__ import annotations

import ast
import importlib.metadata as importlib_metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FNS = REPO / "src" / "signet_trainer" / "modal" / "fns.py"
APP = REPO / "src" / "signet_trainer" / "modal" / "app.py"

#: The in-container call that drags the config closure. A function that makes it owes its image
#: every root that import pulls; a function that does not is unaffected — which is exactly why
#: ``h3_preprocess`` succeeded on an image ``h3_train`` then died on.
CONFIG_LOADER = "load_config_from_text"

#: Shell verbs and flags that appear inside ``uv pip install ...`` command strings and are not
#: distributions. A stray extra name here is harmless (it can only ADD to the declared set); a
#: MISSING real declaration makes the test fail loudly, which is the safe direction.
_NOT_A_DISTRIBUTION = {
    "uv", "pip", "install", "cd", "git", "clone", "checkout", "&&", "-e", "python", "-m",
    "apt", "apt-get", "-y", "mkdir", "-p", "echo", "true",
}

#: ⛔ Images that satisfy part of the closure TRANSITIVELY, each with a written reason and its
#: supplier named. This is a REGISTRY, not an allowlist: an entry whose image now declares the root
#: itself is a FAILURE (stale exemptions are how a guard quietly stops guarding).
_TRANSITIVE_SUPPLIER_REGISTRY: dict[tuple[str, str], str] = {
    ("gpu_image", "pydantic"): (
        "supplied by ltx-trainer, editable-installed from the cloned Lightricks/LTX-2 checkout at "
        "LTX2_COMMIT_SHA. NOT re-declared here on purpose: an explicit pip_install would invalidate "
        "that image's layer cache and force a full LTX image rebuild, which is a change to a "
        "working leg made for a defect that has not occurred on it. ⚠ It is nonetheless FRAGILE — a "
        "deliberate SHA bump that drops pydantic from ltx-trainer's dependencies breaks `train` and "
        "`sample` inside a paid container, exactly as h3_gpu_image broke. Logged as D-10-IMGCLOSURE."
    ),
    ("gpu_image", "yaml"): (
        "same supplier and the same reasoning as the pydantic entry above (PyYAML arrives through "
        "the ltx-trainer dependency tree)."
    ),
}


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Line-wise ``#`` strip, so a name appearing only in a comment is never read as a declaration.

    Load-bearing here: ``app.py``'s comments SPELL OUT the closure (``pydantic  <- config/schema.py``),
    so a text scan that kept comments would have reported the roots as declared while the image did
    not install them — a guard that passes on its own documentation.
    """
    return "\n".join(re.sub(r"#.*$", "", line) for line in src.splitlines())


def _normalize_dist(name: str) -> str:
    """PEP-503-ish normalization so ``PyYAML`` / ``pyyaml`` / ``huggingface_hub`` compare equal."""
    return name.strip().lower().replace("_", "-").replace(".", "-")


def image_blocks() -> dict[str, str]:
    """``{image_name: source of its assignment block}`` for every ``modal.Image`` built in app.py."""
    source = _strip_comments(_src(APP))
    blocks: dict[str, str] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        text = ast.get_source_segment(source, node.value) or ""
        if "modal.Image" in text:
            blocks[target.id] = text
    return blocks


def declared_distributions(block: str) -> set[str]:
    """Distribution names an image block installs — from ``pip_install`` args AND shell commands.

    Both forms have to be read: ``download_image`` uses ``.pip_install("pydantic>=2.10.4", ...)``
    while the GPU images shell out to ``uv pip install --system 'peft>=0.14' ...`` inside
    ``run_commands``. A parser that only understood the first form would report every GPU image as
    declaring nothing.
    """
    declared: set[str] = set()
    for literal in re.findall(r'"([^"]*)"', block) + re.findall(r"'([^']*)'", block):
        for token in literal.split():
            token = token.strip("'\"")
            if not token or token in _NOT_A_DISTRIBUTION:
                continue
            if token.startswith("-") or "/" in token or ":" in token or "@" in token:
                continue
            name = re.split(r"[<>=!~\[]", token)[0]
            if name and re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", name):
                declared.add(_normalize_dist(name))
    return declared


def _calls_config_loader(node: ast.FunctionDef) -> bool:
    """Does this function actually CALL the loader — as opposed to merely NAMING it?

    ⚠ SHARPENED 2026-08-09, and the imprecision it replaces was live rather than theoretical. The
    predecessor was ``CONFIG_LOADER in ast.unparse(node)`` — a substring test over the whole
    unparsed body, DOCSTRINGS INCLUDED. ``modal/fns.wan_train`` is the first stage in this repo that
    deliberately does NOT load a config (``wan_musubi_image`` carries musubi's ``pydantic==1.10.13``
    while ``config/load`` needs pydantic>=2.10.4, so the stage takes threaded parameters instead),
    and its docstring says exactly that — naming ``load_config_from_text`` in order to explain its
    absence. The substring test read that sentence as a call and demanded the config closure be
    installed on an image whose whole design is that it cannot carry it.

    An AST CALL-node scan answers the question the incidents were actually about. Verified
    equivalent before the change: over the nine pre-existing config-loading stages
    (``backup_sync``/``restore`` on ``download_image``, ``fuse``/``sample``/``train`` on
    ``gpu_image``, ``h3_sample``/``h3_train``, ``qwen_edit_sample``/``qwen_edit_train``) the two
    scans return an IDENTICAL mapping, so both historical incidents remain detected. The only
    difference is ``wan_train``, which does not call it.

    Deliberately does NOT chase indirect calls (a helper that loads a config on the function's
    behalf). The substring form did not catch those either — a helper's body is not inside this
    function's AST — so nothing is lost, and inventing a call-graph walk here would be a bigger,
    less checkable thing than the guard it protects.
    """
    return any(
        isinstance(child, ast.Call)
        and (
            (isinstance(child.func, ast.Name) and child.func.id == CONFIG_LOADER)
            or (isinstance(child.func, ast.Attribute) and child.func.attr == CONFIG_LOADER)
        )
        for child in ast.walk(node)
    )


def functions_by_image() -> dict[str, list[str]]:
    """``{image_name: [fn, ...]}`` for every ``@app.function(image=...)`` that loads a config."""
    source = _src(FNS)
    tree = ast.parse(source)
    mapping: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        image = None
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                for keyword in decorator.keywords:
                    if keyword.arg == "image" and isinstance(keyword.value, ast.Name):
                        image = keyword.value.id
        if image is None:
            continue
        if _calls_config_loader(node):
            mapping.setdefault(image, []).append(node.name)
    return mapping


def config_load_roots() -> list[str]:
    """The third-party roots ``signet_trainer.config.load`` imports DIRECTLY. Derived, never listed.

    A SUBPROCESS is required: an in-process ``sys.modules`` diff returns empty because another test
    in the same session has almost certainly already imported ``config.load``.
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
    # Narrow to the roots signet_trainer imports DIRECTLY. Everything else (numpy, pydantic_core,
    # annotated_types, ...) is a pip transitive that arrives for free and must NOT be declared —
    # pinning transitives would be noise that rots.
    sources = [Path(f).read_text(encoding="utf-8") for f in snapshot["files"]]
    return [
        root
        for root in snapshot["roots"]
        if any(
            re.search(rf"^\s*(?:import|from)\s+{re.escape(root)}\b", s, re.MULTILINE)
            for s in sources
        )
    ]


def _candidates(root: str) -> set[str]:
    """Import name -> distribution name(s) (``yaml`` -> ``PyYAML``), falling back to the import name."""
    mapping = importlib_metadata.packages_distributions()
    return {_normalize_dist(d) for d in mapping.get(root, [])} or {_normalize_dist(root)}


# --------------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------------


def test_the_derivation_itself_is_not_broken() -> None:
    """A guard whose probe silently returns nothing passes forever. Non-vacuity floor."""
    roots = config_load_roots()
    assert roots, "the closure probe produced no direct roots — the probe is broken, not the images"
    assert "pydantic" in roots, (
        "config/schema.py imports pydantic at module scope; a derivation that misses it would have "
        "missed BOTH real incidents."
    )
    assert functions_by_image(), "no config-loading Modal function found — the AST scan is broken"


def test_the_call_detector_reads_calls_and_not_prose() -> None:
    """RED self-check on the sharpened detector: a mention is not a call, and a call is not missed.

    Both directions, because getting either wrong is expensive in a different way. A detector that
    flags prose demands dependencies an image cannot carry (the ``wan_train`` case that prompted
    this). A detector that misses a real call is the ModuleNotFoundError in a paid container this
    whole file exists to prevent, and it would fail SILENTLY — an image simply drops out of the
    mapping and the gate reports clean.
    """
    mentions_only = ast.parse(
        f'def f():\n    """This stage does NOT call {CONFIG_LOADER} — see the image note."""\n'
        f"    x = 1\n"
    ).body[0]
    really_calls = ast.parse(f"def g():\n    cfg = {CONFIG_LOADER}(text)\n").body[0]
    attribute_call = ast.parse(f"def h():\n    cfg = mod.{CONFIG_LOADER}(text)\n").body[0]
    assert not _calls_config_loader(mentions_only), "a docstring mention was read as a call"
    assert _calls_config_loader(really_calls), "a plain call was missed — the gate would go silent"
    assert _calls_config_loader(attribute_call), "a module-qualified call was missed"


def test_the_real_config_loading_stages_are_all_still_detected() -> None:
    """Non-vacuity for the sharpening: every stage that genuinely loads a config is still in scope.

    Named explicitly — not derived — precisely BECAUSE the detector changed. The point of this
    assertion is to be an independent statement of the expected result that a change to the
    detection logic must survive; deriving it from the detector would make it circular. Both
    documented incidents are represented: ``backup_sync``/``restore`` (download_image, 2026-08-05)
    and ``h3_train``/``h3_sample`` (h3_gpu_image, 2026-08-06).

    ``wan_train`` is deliberately ABSENT: it is the one stage that cannot load a config, because
    ``wan_musubi_image`` carries musubi's pydantic 1.x.
    """
    detected = {image: sorted(fns) for image, fns in functions_by_image().items()}
    assert detected == {
        "download_image": ["backup_sync", "restore"],
        "gpu_image": ["fuse", "sample", "train"],
        "h3_gpu_image": ["h3_sample", "h3_train"],
        "qwen_gpu_image": ["qwen_edit_sample", "qwen_edit_train"],
    }, detected


def test_every_image_carries_the_closure_of_the_functions_pointed_at_it() -> None:
    """⛔ THE GATE, asked of every image rather than of one.

    A green Stage 1 on an image says nothing about Stage 2 on the same image: the two stages have
    different import closures, and it is the difference that kills the paid container.
    """
    roots = config_load_roots()
    blocks = image_blocks()
    missing: list[str] = []
    for image, fns in sorted(functions_by_image().items()):
        assert image in blocks, f"{image} is used in fns.py but not built in app.py"
        declared = declared_distributions(blocks[image])
        for root in roots:
            if _candidates(root) & declared:
                continue
            if (image, root) in _TRANSITIVE_SUPPLIER_REGISTRY:
                continue
            missing.append(
                f"{image} does not declare {root!r} "
                f"(distribution: {'/'.join(sorted(_candidates(root)))}), but {', '.join(sorted(fns))} "
                f"call {CONFIG_LOADER} in-container"
            )
    assert not missing, (
        "an image is missing part of its functions' import closure:\n  "
        + "\n  ".join(missing)
        + "\n\nThis is a ModuleNotFoundError inside a PAID container that every local gate passes "
        "(dry-run OK, cost under the guardrail, approval honoured). It has happened twice — "
        "download_image 2026-08-05, h3_gpu_image 2026-08-06 — with the identical message. Declare "
        "the distribution on the image, mirroring the pyproject [project.dependencies] floor, or "
        "register a NAMED transitive supplier with a written reason."
    )


def test_the_transitive_registry_has_no_stale_entries() -> None:
    """Enforced in BOTH directions — an exemption that outlives its cause is worse than none.

    If an image later declares the root itself, its registry entry must go. Otherwise the registry
    slowly becomes an allowlist that describes nothing, and the next reader cannot tell which
    entries are still load-bearing.
    """
    blocks = image_blocks()
    stale: list[str] = []
    for (image, root), reason in _TRANSITIVE_SUPPLIER_REGISTRY.items():
        assert reason.strip(), f"registry entry ({image}, {root}) has no written reason"
        if image not in blocks:
            stale.append(f"({image}, {root}): {image} no longer exists in app.py")
            continue
        if _candidates(root) & declared_distributions(blocks[image]):
            stale.append(f"({image}, {root}): {image} now declares it directly — drop the entry")
    assert not stale, "stale transitive-supplier registry entries:\n  " + "\n  ".join(stale)


def test_the_h3_image_declares_the_closure_directly_after_the_2026_08_06_incident() -> None:
    """The specific regression, pinned by name so a rebuild-avoiding revert is loud.

    ``h3_gpu_image`` deliberately installs no ltx-* package, so nothing supplies pydantic
    transitively — it must declare it, and it may never be moved into the registry.
    """
    declared = declared_distributions(image_blocks()["h3_gpu_image"])
    assert "pydantic" in declared, "h3_gpu_image must declare pydantic (the 2026-08-06 train failure)"
    assert "pyyaml" in declared, "h3_gpu_image must declare pyyaml (config/load.py imports yaml)"
    for root in ("pydantic", "yaml"):
        assert ("h3_gpu_image", root) not in _TRANSITIVE_SUPPLIER_REGISTRY, (
            "h3_gpu_image installs no ltx-* package, so there is no transitive supplier to name. "
            "Registering one would be a fiction that reopens the exact failure."
        )


def test_the_versions_mirror_the_pyproject_floors_and_introduce_no_new_package() -> None:
    """No package NEW to the project may be introduced by an image fix.

    The house rule on package installs: a fix for a missing import declares something the project
    already depends on. If a root is not in pyproject's dependencies, the correct move is a human
    legitimacy check, not a pip line added under time pressure inside a debugging session.
    """
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    block = pyproject.split("dependencies", 1)[1]
    for root in config_load_roots():
        names = _candidates(root)
        assert any(
            re.search(rf'"{re.escape(name)}[<>=!~\[\]a-z0-9.,\s-]*"', block, re.IGNORECASE)
            for name in names
        ), (
            f"{root} (distribution {'/'.join(sorted(names))}) is in the config-load closure but is "
            f"NOT a declared project dependency in pyproject.toml. Adding it to a Modal image would "
            f"introduce a package new to the project — stop and verify its legitimacy first."
        )
