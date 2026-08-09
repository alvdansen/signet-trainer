"""VERIFIER-authored coverage for the cross-slice gaps the five per-slice suites cannot see.

Family #3 landed as five independent slices, each with its own test file, each green **in
isolation**. Every gap below is a seam BETWEEN slices, or a property of the working tree as a
whole — which is precisely the class of defect a per-slice suite is structurally blind to. Two of
these tests were written because the full-suite run is redder than ``ea48def``, and the per-slice
runs are not.

Nothing here is a fix. Each test states a property the shipped code either has or does not have,
and fails loudly when it does not, so the gap has a name and a line number instead of a paragraph
in a report.

Zero GPU, zero Modal dispatch, zero weight bodies. Safetensors HEADERS (8-byte length + JSON) are
read where the real checkpoint is present, and skipped where it is not.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

#: The modules family #3 added. Every one is new in this working tree; none exists at ea48def.
#: ``models/qwen_edit_pipeline.py`` joined 2026-08-09 with the render slice — it holds the generation
#: pipeline assembly, the §8 inference recipe and the scheduler pin, and it is registered here so the
#: silent-no-op scan, the actionable-refusal scan and the co-import probe all cover it too.
QWEN_NEW_MODULES = (
    "src/signet_trainer/models/qwen_edit_loader.py",
    "src/signet_trainer/models/qwen_edit_pipeline.py",
    "src/signet_trainer/prep/qwen_edit_encode.py",
    "src/signet_trainer/train/qwen_edit_step.py",
    "src/signet_trainer/train/family_hooks.py",
    "src/signet_trainer/inference/qwen_edit_layout.py",
)

#: The modules family #3 edited in place. ADDITIVE-ONLY is asserted elsewhere (byte-identical
#: dry-run output across all 17 configs); these are scanned here only for silent no-ops.
QWEN_TOUCHED_MODULES = (
    "src/signet_trainer/inference/grid.py",
    "src/signet_trainer/inference/render_key.py",
    "src/signet_trainer/inference/samples_layout.py",
    "src/signet_trainer/modal/app.py",
    "src/signet_trainer/modal/entrypoint.py",
    "src/signet_trainer/modal/fns.py",
)

QWEN_MODULE_DOTTED = (
    "signet_trainer.models.qwen_edit_loader",
    "signet_trainer.models.qwen_edit_pipeline",
    "signet_trainer.prep.qwen_edit_encode",
    "signet_trainer.train.qwen_edit_step",
    "signet_trainer.train.family_hooks",
    "signet_trainer.inference.qwen_edit_layout",
    "signet_trainer.conditioning.qwen_edit",
    "signet_trainer.conditioning.qwen_edit_geometry",
)


def _run_probe(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a FRESH interpreter with ``src`` importable. Fresh = no suite state."""
    import os

    env = dict(os.environ, PYTHONPATH="src")
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


# ==================================================================================================
# GAP 1 — RETIRED 2026-08-08. Optional-dependency probes are now suite-order robust.
#
# The two tests that lived here were written to be self-retiring: "It is RED until the loader's
# two probes stop being order-dependent (or test_mask_encode restores what it pops)... if the
# fragility was fixed, delete this test." BOTH fixes landed, so both were deleted:
#
#   1. tests/test_mask_encode.py now snapshots sys.modules before popping and restores in a
#      finally, matching tests/test_grid_html.py:80-92. That kills the poisoning at source —
#      popping "PIL" alone had left sys.modules["PIL.Image"] cached, so a later import rebuilt
#      the parent without rebinding Image, and diffusers/utils/export_utils.py:27 then raised
#      AttributeError on the annotation list[PIL.Image.Image].
#   2. tests/test_qwen_edit_loader.py replaced both pytest.importorskip call sites with a
#      PathFinder-based _require(), because importorskip decides availability BY IMPORTING and
#      catches only ImportError — so any exception raised during a present package's import
#      becomes a FAILURE instead of a skip. Second line of defence against a future evictor.
# ==================================================================================================
# ==================================================================================================
# GAP 2 — cross-slice symbol resolution. Slice N's test file imports only slice N.
# ==================================================================================================


def _cross_slice_imports() -> list[tuple[str, str, str, int]]:
    """``(consumer_path, module, name, lineno)`` for every qwen import in the Modal layer."""
    found: list[tuple[str, str, str, int]] = []
    for rel in ("src/signet_trainer/modal/fns.py", "src/signet_trainer/modal/entrypoint.py"):
        path = REPO_ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "qwen_edit" in node.module:
                for alias in node.names:
                    found.append((rel, node.module, alias.name, node.lineno))
            if isinstance(node, ast.ImportFrom) and node.module == "signet_trainer.train.family_hooks":
                for alias in node.names:
                    found.append((rel, node.module, alias.name, node.lineno))
    return found


def test_every_symbol_the_modal_qwen_arms_import_actually_resolves() -> None:
    """The Modal layer names symbols across FOUR sibling slices; nothing before this checked them.

    The Modal wiring is the layer that spends money, and it reaches into
    ``models/qwen_edit_loader``, ``prep/qwen_edit_encode``, ``train/qwen_edit_step``,
    ``train/family_hooks`` and ``inference/qwen_edit_layout`` — each written by a different slice,
    each tested only against itself. An ``ImportError`` here surfaces INSIDE a metered container,
    after the arch gate, after the weight load. Slice 2 already caught one live break of exactly
    this shape (``modal/fns.py`` passing flat strings to ``qwen_edit_text_cache_key``, which
    expected mappings), which is the argument for pinning the whole set.

    An import whose MODULE has not landed is not automatically a break — it is acceptable exactly
    when the pre-dispatch readiness table declares it, because then the operator meets a $0 named
    abort instead of a container traceback. So the property asserted is two-sided: every symbol of
    an existing module must resolve, and every missing module must be DECLARED in
    ``entrypoint._QWEN_EDIT_STAGE_MODULES``.
    """
    import importlib

    # ``modal/entrypoint.py`` imports the Modal SDK at module scope (it builds the app graph), so
    # the readiness table is only readable where the SDK is installed. Rather than skip the whole
    # test on the SDK-free interpreter — which is the one the dry-run contract targets — the table
    # is read from the AST, which needs nothing but the file. Same data, no import.
    entrypoint_src = (REPO_ROOT / "src/signet_trainer/modal/entrypoint.py").read_text(
        encoding="utf-8"
    )
    declared_gaps: set[str] = set()
    for node in ast.walk(ast.parse(entrypoint_src)):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_QWEN_EDIT_STAGE_MODULES":
            for value in ast.literal_eval(node.value).values():  # type: ignore[arg-type]
                declared_gaps.update(value)
    assert declared_gaps, "could not read _QWEN_EDIT_STAGE_MODULES out of modal/entrypoint.py"

    imports = _cross_slice_imports()
    assert imports, "no qwen imports found in the Modal layer — the AST scan is broken, not clean"

    unresolved: list[str] = []
    undeclared: list[str] = []
    for consumer, module, name, lineno in imports:
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError:
            if module not in declared_gaps:
                undeclared.append(
                    f"{consumer}:{lineno} -> {module} does not exist AND is absent from "
                    "_QWEN_EDIT_STAGE_MODULES, so a run would die inside a metered container"
                )
            continue
        except Exception as exc:  # noqa: BLE001 — an unimportable module IS the finding
            unresolved.append(f"{consumer}:{lineno} -> module {module} not importable: {exc!r}")
            continue
        if not hasattr(mod, name):
            unresolved.append(f"{consumer}:{lineno} -> {module}.{name} MISSING")
    assert not unresolved, "cross-slice symbol breaks:\n  " + "\n  ".join(unresolved)
    assert not undeclared, "UNDECLARED unlanded module(s):\n  " + "\n  ".join(undeclared)


def test_the_whole_family_imports_together_in_one_fresh_process() -> None:
    """All seven qwen modules co-resident, in a fresh interpreter, with no GPU and no Modal SDK.

    Per-slice suites import one module each. This is the first thing that imports all of them at
    once — the cheapest possible proof that no two slices disagree at module scope (duplicate
    constant names with different values, a circular import, an eager heavy dependency).
    """
    probe = (
        "import importlib, sys\n"
        f"mods = {list(QWEN_MODULE_DOTTED)!r}\n"
        "for m in mods:\n"
        "    importlib.import_module(m)\n"
        "assert 'modal' not in sys.modules, 'a qwen module dragged in the Modal SDK'\n"
        "print('IMPORTED_ALL', len(mods))\n"
    )
    result = _run_probe(probe)
    assert "IMPORTED_ALL" in result.stdout, (
        f"the qwen family does not import as a set:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-1500:]}"
    )


# ==================================================================================================
# GAP 3 — the loss weight. Independently recomputed here, from the upstream four lines.
# ==================================================================================================


def test_the_loss_weight_is_the_bell_curve_recomputed_from_upstream() -> None:
    """Bit-exact against ``custom_flowmatch_sampler.py:35-42``, recomputed in this file.

    Written independently of ``tests/test_qwen_edit_step.py``'s own version so the property is
    asserted twice from two transcriptions. The four upstream lines, verbatim::

        x = torch.arange(num_timesteps, dtype=torch.float32)
        y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())
    """
    import torch

    from signet_trainer.train.qwen_edit_step import qwen_edit_timestep_weights

    n = 1000
    x = torch.arange(n, dtype=torch.float32)
    y = torch.exp(-2 * ((x - n / 2) / n) ** 2)
    y_shifted = y - y.min()
    reference = y_shifted * (n / y_shifted.sum())

    shipped = qwen_edit_timestep_weights(n)
    assert torch.equal(shipped, reference), (
        f"the shipped curve is not the bsmntw curve: max abs delta "
        f"{(shipped - reference).abs().max().item()}"
    )


def test_the_curve_peaks_mid_schedule_with_an_exact_zero_minimum() -> None:
    """Shape assertions, not values: min is EXACTLY 0, the peak is mid-grid, the mean is 1.

    Both ends de-weighted, the middle boosted. ``min == 0`` exactly is load-bearing — it is what
    the ``y - y.min()`` shift buys, and it is what distinguishes the curve from a table.
    """
    import torch

    from signet_trainer.train.qwen_edit_step import qwen_edit_timestep_weights

    w = qwen_edit_timestep_weights(1000)
    assert w.min().item() == 0.0, f"minimum is {w.min().item()!r}, not an exact 0.0"
    assert int(w.argmin()) == 0, "the zero is not at grid index 0 (t = 1000, pure noise)"
    assert int(w.argmax()) == 500, f"peak at index {int(w.argmax())}, expected mid-grid 500"
    assert w.max().item() > w[0].item() and w.max().item() > w[-1].item()
    assert abs(w.mean().item() - 1.0) < 1e-6, f"mean {w.mean().item()} is not normalised to 1"
    assert bool(torch.all(w[:501].diff() >= 0)), "not monotonically increasing up to the peak"
    assert bool(torch.all(w[500:].diff() <= 0)), "not monotonically decreasing after the peak"


def test_no_thousand_entry_weight_table_is_shipped_anywhere_in_src() -> None:
    """The dead ``default_weighing_scheme`` table must not appear as data anywhere under ``src``.

    ``custom_flowmatch_sampler.py:65-70`` reads that table, and ``:71-74`` is an ``if v2: / else:``
    — an ``if``, not an ``elif`` — which overwrites it on every path. It has never weighted a
    house step. Porting it would move family #3 off the distribution every proven adapter trained
    under, at no visible symptom. ``tests/test_qwen_edit_step.py`` asserts this for its own module;
    this widens the scan to the whole package, including the Modal layer.
    """
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) > 64:
                numeric = sum(
                    1
                    for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, (int, float))
                )
                if numeric > 64:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} ({numeric} numbers)")
    assert not offenders, "a large numeric literal table is shipped: " + ", ".join(offenders)


def test_the_hook_leaves_the_ltx_and_h3_timestep_draws_bit_identical() -> None:
    """Behavioural non-perturbation, in subprocesses: sibling draws unchanged by loading the hook.

    Not a grep. One process imports the qwen weight hook and the family registry and CALLS them;
    the other never sees them. Both then draw the LTX and H3 timesteps from a seeded generator and
    hash the results. Equal hashes is the only acceptable answer — the hook must be inert for
    families that do not ask for it.
    """
    body = (
        "import sys, hashlib, numpy as np\n"
        "if sys.argv[-1] == 'load':\n"
        "    from signet_trainer.train.qwen_edit_step import qwen_edit_timestep_weights\n"
        "    from signet_trainer.train.family_hooks import build_loop_hooks\n"
        "    qwen_edit_timestep_weights()\n"
        "    build_loop_hooks('ltx')\n"
        "from signet_trainer.train.flow_match import FlowMatchingSchedule\n"
        "from signet_trainer.train.h3_step import h3_draw_timesteps\n"
        "rng = np.random.default_rng(42)\n"
        "ltx = [FlowMatchingSchedule().sample_timesteps(4, 4096, rng) for _ in range(8)]\n"
        "rng2 = np.random.default_rng(42)\n"
        "h3 = [h3_draw_timesteps(rng2) for _ in range(8)]\n"
        "blob = b''.join(a.tobytes() for a in ltx) + repr(h3).encode()\n"
        "print('DIGEST', hashlib.sha256(blob).hexdigest())\n"
    )
    digests = {tag: _run_probe_argv(body, tag) for tag in ("noload", "load")}
    assert digests["noload"] and digests["noload"] == digests["load"], (
        f"loading the qwen loss-weight hook perturbed a sibling family's draw: {digests}"
    )


def _run_probe_argv(code: str, argv: str) -> str:
    import os

    env = dict(os.environ, PYTHONPATH="src")
    result = subprocess.run(
        [sys.executable, "-c", code, argv],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("DIGEST "):
            return line.split(" ", 1)[1]
    raise AssertionError(f"probe produced no digest ({argv}):\n{result.stderr[-1500:]}")


# ==================================================================================================
# GAP 4 — no silent no-ops, and every remaining gap is a NAMED, ACTIONABLE refusal.
# ==================================================================================================


@pytest.mark.parametrize("rel", QWEN_NEW_MODULES + QWEN_TOUCHED_MODULES)
def test_no_qwen_module_contains_a_silent_no_op(rel: str) -> None:
    """Bare ``pass``, lone ``return None``, lone ``...``, and ``except: pass`` are all refusals.

    The binding rule for this phase is that every remaining gap is a NAMED symbol raising
    ``NotImplementedError`` with an actionable message — never a silent no-op. This is that rule as
    an AST check, over the new modules AND the six the slices edited in place.
    """
    src = (REPO_ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = list(node.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if not body:
                offenders.append(f"{node.lineno}: {node.name}() has a docstring and nothing else")
            elif len(body) == 1:
                stmt = body[0]
                if isinstance(stmt, ast.Pass):
                    offenders.append(f"{node.lineno}: {node.name}() is a bare pass")
                elif isinstance(stmt, ast.Expr) and stmt.value.__class__ is ast.Constant:
                    if getattr(stmt.value, "value", None) is Ellipsis:
                        offenders.append(f"{node.lineno}: {node.name}() is a bare ...")
                elif isinstance(stmt, ast.Return) and (
                    stmt.value is None
                    or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)
                ):
                    offenders.append(f"{node.lineno}: {node.name}() only returns None")
        if isinstance(node, ast.ExceptHandler) and all(
            isinstance(s, ast.Pass) for s in node.body
        ):
            offenders.append(f"{node.lineno}: except-handler swallows silently")
    assert not offenders, f"{rel} contains silent no-ops:\n  " + "\n  ".join(offenders)


def test_every_remaining_qwen_gap_names_what_lands_it() -> None:
    """Each surviving ``NotImplementedError`` in the qwen surface must be ACTIONABLE.

    Actionable is defined operationally, not stylistically: the message names at least one module
    path, file, or symbol the reader can go to. A refusal that says only "not implemented" costs a
    reader the same as a silent no-op, one traceback later.
    """
    stubs: list[tuple[str, int, str]] = []
    for rel in QWEN_NEW_MODULES:
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", "") == "NotImplementedError"
            ):
                text = " ".join((ast.get_source_segment(src, node.exc) or "").split())
                stubs.append((rel, node.lineno, text))

    weak = [
        f"{rel}:{lineno}"
        for rel, lineno, text in stubs
        if not any(tok in text for tok in (".py", "signet_trainer", "_fn", "build_", "load_", "()"))
        or len(text) < 120
    ]
    assert not weak, f"non-actionable NotImplementedError message(s): {weak}"
    # The count is reported, not asserted — a gap closing is progress, not a regression.
    print(f"[verifier] surviving declared gaps in the qwen surface: {len(stubs)}")


def test_no_qwen_entry_point_the_brief_named_still_raises_not_implemented() -> None:
    """RETIRED-AND-REPLACED 2026-08-09 — the fourth and last of the brief's stubs landed.

    The predecessor asserted ``render_qwen_edit_sample`` still raised ``NotImplementedError``, and
    said in its own docstring that it was "pinned here so the remaining gap has a test that goes
    green the day it lands". It landed (``models/qwen_edit_pipeline.qwen_edit_generate``, with the
    layout symbol delegating), so this is the inverted form: none of the four entry points the brief
    named may raise ``NotImplementedError`` any more, and the check is by AST over the modules that
    own them rather than by calling them, so it cannot be satisfied by an exception that merely
    changed type.

    Note ``train/family_hooks.py``'s H3 arm and ``prep/h3_vae_contract.py``'s two test stubs are NOT
    qwen entry points and are deliberately out of scope here; ``modal/fns.py``'s two are unreachable
    defence-in-depth for ``backup.destination='cloud'``.
    """
    owners = {
        "src/signet_trainer/models/qwen_edit_loader.py": (
            "load_qwen_edit_transformer",
            "assert_qwen_edit_arch",
            "quantize_qwen_edit",
        ),
        "src/signet_trainer/prep/qwen_edit_encode.py": ("prepare_qwen_edit_image",),
        "src/signet_trainer/train/qwen_edit_step.py": ("build_qwen_edit_step_fn",),
        "src/signet_trainer/inference/qwen_edit_layout.py": ("render_qwen_edit_sample",),
    }
    unlanded: list[str] = []
    for rel, names in owners.items():
        path = REPO_ROOT / rel
        assert path.exists(), f"{rel} is missing entirely"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        by_name = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in names:
            node = by_name.get(name)
            if node is None:
                unlanded.append(f"{rel}::{name} does not exist")
                continue
            raises = [
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Raise)
                and isinstance(child.exc, ast.Call)
                and getattr(child.exc.func, "id", "") == "NotImplementedError"
            ]
            if raises:
                unlanded.append(f"{rel}::{name} still raises at line(s) {[r.lineno for r in raises]}")
    assert not unlanded, "declared qwen entry point(s) still unlanded:\n  " + "\n  ".join(unlanded)

    # The generate call must be a real delegation, not a renamed placeholder.
    from signet_trainer.inference import qwen_edit_layout

    import inspect

    assert "pipeline" in inspect.signature(qwen_edit_layout.render_qwen_edit_sample).parameters


def test_the_packing_module_is_landed_and_is_the_single_transcription() -> None:
    """RETIRED-AND-REPLACED 2026-08-08. The predecessor asserted ``qwen_edit_packing`` did NOT
    exist and told its reader to delete it once the module landed; it has landed, so this is the
    inverted form that keeps the property it was really protecting.

    What mattered was never the absence — it was that the 2x2 pack has exactly ONE transcription.
    Every wrong ordering of the six-axis permute yields the right shape and the wrong values, so a
    second copy is a silent fork. ``tests/test_qwen_edit_packing.py`` proves the ordering by
    round-tripping to bit-equality; this asserts nobody has added a rival.
    """
    packing = SRC / "signet_trainer" / "conditioning" / "qwen_edit_packing.py"
    assert packing.exists(), "conditioning/qwen_edit_packing.py is the strategy's declared pack_fn"

    strategy_src = (SRC / "signet_trainer" / "conditioning" / "qwen_edit.py").read_text(
        encoding="utf-8"
    )
    assert "qwen_edit_packing" in strategy_src, (
        "the strategy no longer names its pack_fn producer"
    )

    # Exactly one EXECUTABLE ``.permute(0, 2, 4, 1, 3, 5)`` under src/ — the pack itself.
    #
    # Parsed with ast rather than grepped, because the transform is QUOTED in four docstrings
    # (this module's, qwen_edit.py's refusal message, qwen_edit_geometry.py's header) to explain
    # what it is and why it must not be duplicated. Quoting it is documentation; calling it twice
    # is the fork. A text scan cannot tell those apart and would punish the documentation.
    import ast

    hits: list[str] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "permute":
                continue
            args = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if args == [0, 2, 4, 1, 3, 5]:
                hits.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")

    assert len(hits) == 1, (
        f"the 2x2 pack permute has {len(hits)} executable transcription(s) under src/: {hits}. It "
        f"must have exactly one, in conditioning/qwen_edit_packing.py. Every wrong ordering of "
        f"those six axes yields the RIGHT shape and the WRONG values, so a second copy is a silent "
        f"fork of a checkpoint contract — nothing raises, the loss descends, the adapter is wrong."
    )
    assert "qwen_edit_packing.py" in hits[0], (
        f"the single transcription moved out of qwen_edit_packing.py to {hits[0]}"
    )


# ==================================================================================================
# GAP 5 — the dry-run contract, as a test rather than a manual command.
# ==================================================================================================


@pytest.mark.parametrize(
    "config_rel",
    [
        "configs/qwen_image_edit.example.yaml",
        "configs/ltx23_lora.example.yaml",
        "configs/ltx23_ic_lora.example.yaml",
        "configs/campaign_a2v.example.yaml",
    ],
)
def test_the_dry_run_still_exits_zero_for_every_family(config_rel: str) -> None:
    """One qwen config and three pre-existing LTX configs, each through the real CLI, rc == 0.

    The additive-only contract's cheapest observable: family #3 landing must not change what the
    dry-run says about families #1 and #2, and the qwen arm must still reach a clean synthetic
    packed batch on CPU.
    """
    import os

    env = dict(os.environ, PYTHONPATH="src")
    result = subprocess.run(
        [sys.executable, "-m", "signet_trainer.dryrun", config_rel],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{config_rel} dry-run exited {result.returncode}:\n{result.stdout}\n{result.stderr}"
    )
    assert "[signet-dryrun] OK" in result.stdout


def test_the_bad_frames_config_still_refuses() -> None:
    """The known-bad config must still be REFUSED — a gate that stops refusing is a broken gate."""
    import os

    env = dict(os.environ, PYTHONPATH="src")
    result = subprocess.run(
        [sys.executable, "-m", "signet_trainer.dryrun", "configs/bad_frames.example.yaml"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert "config validation FAILED" in combined, combined[-800:]
    assert "frames % 8 == 1" in combined or "(frames - 1) % 8 == 0" in combined


# ==================================================================================================
# GAP 6 — the arch gate, driven against the REAL headers when they are on this box.
# ==================================================================================================

_TRANSFORMER = Path("F:/AI-Models/ComfyUI-models/diffusion_models/qwen_image_edit_2511_bf16.safetensors")
_TE_VL = Path("F:/AI-Models/ComfyUI-models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors")
_TE_TEXT_ONLY = Path(
    "F:/AI-Models/ComfyUI-models/text_encoders/_MISLABELED_was-qwen2.5-7b-instruct-not-VL.safetensors"
)


def _header_keys(path: Path) -> list[str]:
    """Read ONLY the safetensors header: an 8-byte little-endian length, then that many JSON bytes."""
    import json
    import struct

    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    return list(header)


@pytest.mark.skipif(not _TE_VL.is_file(), reason="the VL text encoder is not on this box")
def test_the_real_vl_header_passes_the_vision_gate() -> None:
    """The correct encoder, read as a header, is ACCEPTED — the positive half of the day-lost gate."""
    from signet_trainer.models.qwen_edit_loader import assert_qwen_edit_text_encoder_vision

    census = assert_qwen_edit_text_encoder_vision(_header_keys(_TE_VL))
    assert census["vision"] > 0
    assert census["total"] == 1446, f"the measured VL tensor count moved: {census}"
    assert census["vision"] == 714, f"the measured VL vision count moved: {census}"


@pytest.mark.skipif(
    not _TE_TEXT_ONLY.is_file(), reason="the mislabeled text-only checkpoint is not on this box"
)
def test_the_real_text_only_header_is_refused_by_name() -> None:
    """The mislabeled checkpoint is REFUSED, and the message quotes the symptom it replaces.

    The failure it prevents names neither file nor component::

        mat1 and mat2 shapes cannot be multiplied (5376x1280 and 3840x1280)
    """
    from signet_trainer.models.qwen_edit_loader import assert_qwen_edit_text_encoder_vision

    with pytest.raises(RuntimeError) as excinfo:
        assert_qwen_edit_text_encoder_vision(_header_keys(_TE_TEXT_ONLY))
    message = str(excinfo.value)
    assert "NO VISION TENSORS" in message
    assert "5376x1280" in message, "the refusal no longer quotes the symptom it exists to replace"


@pytest.mark.skipif(not _TRANSFORMER.is_file(), reason="the transformer checkpoint is not on this box")
def test_the_real_transformer_header_still_resolves_exactly_840_lora_targets() -> None:
    """840 modules = 14 distinct leaves x 60 blocks, and zero RMSNorms, off the LIVE header.

    The measured ground truth, re-derived here from the file rather than from the fixture, so a
    fixture that drifts from the checkpoint cannot keep the suite green.
    """
    from signet_trainer.models.qwen_edit_loader import assert_qwen_edit_targets

    module_names = sorted({key.rsplit(".", 1)[0] for key in _header_keys(_TRANSFORMER)})
    survey = assert_qwen_edit_targets(module_names)
    assert int(survey["total"]) == 840, survey
    assert len(survey["per_leaf"]) == 14, survey
    assert set(survey["per_leaf"].values()) == {60}, survey["per_leaf"]
    assert int(survey["collateral"]) == 0, survey


@pytest.mark.parametrize("blocks", [59, 48, 61])
def test_the_arch_gate_refuses_a_wrong_block_count(blocks: int) -> None:
    """A transformer with the wrong number of blocks aborts BEFORE spend, naming both fields.

    Both ``num_layers`` (what it was configured with) and ``live_transformer_blocks`` (what it was
    loaded with) must be named — the two disagree exactly in the single-file-wrong-config case the
    gate exists for.
    """
    import torch.nn as nn

    from signet_trainer.models.qwen_edit_loader import (
        assert_qwen_edit_arch,
        expected_qwen_edit_arch,
        summarize_qwen_edit_transformer,
    )

    class Toy(nn.Module):
        def __init__(self, n_blocks: int) -> None:
            super().__init__()
            fields: dict[str, Any] = dict(expected_qwen_edit_arch())
            fields["num_layers"] = n_blocks
            for key, value in fields.items():
                setattr(self, key, value)
            self.transformer_blocks = nn.ModuleList([nn.Identity() for _ in range(n_blocks)])
            self.img_in = nn.Linear(64, 3072, bias=False)
            self.txt_in = nn.Linear(3584, 3072, bias=False)
            self.proj_out = nn.Linear(3072, 64, bias=False)

    with pytest.raises(RuntimeError) as excinfo:
        assert_qwen_edit_arch(summarize_qwen_edit_transformer(Toy(blocks)))
    message = str(excinfo.value)
    assert "ARCH MISMATCH" in message
    assert f"live_transformer_blocks: expected 60, got {blocks}" in message, message
    assert f"num_layers: expected 60, got {blocks}" in message, message
