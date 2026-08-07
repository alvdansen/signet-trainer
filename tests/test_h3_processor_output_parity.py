"""D-10-DEF-7 — the class-closing gate: our bypass must reproduce EVERY ``__call__`` output key.

``prep/h3_encode.build_h3_processor_inputs`` bypasses ``Qwen3VLProcessor.__call__`` on purpose, and
that bypass inherits **all** of that call's outputs. The gaps surfaced ONE AT A TIME, each costing a
metered container:

  * attempt 1 — the pre-expanded presentation reaching ``__call__`` at all (D-10-DEF-4 defect 1);
  * attempt 2 — ``mm_token_type_ids``, which ``__call__`` adds and ``compute_3d_position_ids``
    REQUIRES (D-10-DEF-7).

Patching the second alone would have bought the third at the same price. So this file diffs the
OUTPUT KEY SETS of the **REAL** ``processor.__call__`` and our builder over one tiny synthetic
input — a key we have never heard of still trips it.

⛔ **THE VERSION TRAP IS THE CRUX OF THIS FILE.** ``return_mm_token_type_ids`` defaults to **True**
at the pinned ``transformers==5.14.1`` and to **False** at the 5.2.0 that happens to be installed on
the authoring box (which does not even define ``ProcessorMixin.create_mm_token_type_ids``). A
key-diff executed against 5.2.0 would come out GREEN while the real dispatch failed — strictly worse
than no test, because it manufactures confidence. So:

  * the probe runs in a **subprocess** driven by an interpreter that has the PINNED version;
  * the pin is read from ``modal/app.py::TRANSFORMERS_VERSION`` — the same literal the image
    installs — never restated here;
  * a version mismatch, a missing interpreter or a crashed probe is a **FAILURE with instructions**,
    never a skip. A skip here is the false green this whole file exists to prevent.

Why an isolated overlay venv rather than bumping the box's ``transformers``: ``pip install
transformers==5.14.1`` into the global interpreter also drags ``huggingface_hub`` 1.4.1 -> 1.26.1,
``safetensors`` 0.7 -> 0.8 and ``click`` 8.2.1 -> 8.4.2, and ``scenedetect`` 0.6.7.1 pins
``click<8.3.0`` — i.e. the honest-looking option silently breaks a tool the operator segments corpora
with, in every other project on the machine. ``.venv-h3parity`` follows the ``.venv-sam`` precedent
already in this repo: a dedicated venv for one heavy, version-sensitive dependency, gitignored by
the existing ``.venv-*/`` rule.

Zero GPU, zero Modal spend, zero model weights, zero network. See ``tests/_h3_parity_runner.py`` for
the fixture and ``src/signet_trainer/prep/h3_parity.py`` for the diff itself (shared, verbatim, with
the in-container backstop in ``modal/fns.py::h3_preprocess``).
"""

from __future__ import annotations

import ast
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_APP = REPO / "src" / "signet_trainer" / "modal" / "app.py"
_RUNNER = REPO / "tests" / "_h3_parity_runner.py"
_OVERLAY = REPO / ".venv-h3parity"

#: The keys ``ProcessorMixin.__call__`` returns for THIS configuration (text + image references, no
#: video, no audio), enumerated from the 5.14.1 source rather than from a run. Asserted as a
#: NON-VACUITY floor: ``set() - set()`` is empty, so a probe that silently produced nothing on both
#: sides would satisfy the diff and prove nothing. A future ``transformers`` bump that adds a sixth
#: key must fail here and be reviewed deliberately — same discipline as the literal pin itself.
_EXPECTED_CALL_KEYS = frozenset(
    {"input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw"}
)

#: ``__call__`` adds ``mm_token_type_ids`` whenever TEXT is given — NOT when images are — so the
#: text-only branch carries it too (all zeros). Gating our emission on "has references" would
#: diverge here and nowhere else.
_EXPECTED_TEXT_ONLY_KEYS = frozenset({"input_ids", "attention_mask", "mm_token_type_ids"})


def pinned_transformers_version() -> str:
    """``TRANSFORMERS_VERSION`` read off ``modal/app.py`` by AST — importing it would pull Modal."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        targets = getattr(node, "targets", None) or (
            [node.target] if getattr(node, "target", None) is not None else []
        )
        for target in targets:
            if getattr(target, "id", None) == "TRANSFORMERS_VERSION":
                return str(ast.literal_eval(node.value))  # type: ignore[attr-defined]
    raise AssertionError("TRANSFORMERS_VERSION not found in modal/app.py")


def _installed_version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _overlay_python() -> Path | None:
    for relative in ("Scripts/python.exe", "bin/python", "bin/python3"):
        candidate = _OVERLAY / relative
        if candidate.exists():
            return candidate
    return None


def _bootstrap_instructions() -> str:
    """The exact two commands, with ``torchvision`` resolved against THIS box's torch.

    torchvision's ABI is tied to torch: ``torch 2.N`` pairs with ``torchvision 0.(N+15)`` (2.7 ->
    0.22, 2.10 -> 0.25), and the H3 image installs the pair the same way. Printing a computed
    version beats printing a literal that goes stale the next time torch moves.
    """
    torch_version = _installed_version("torch") or "?"
    torchvision_pin = "torchvision"
    parts = torch_version.split("+")[0].split(".")
    if len(parts) >= 2 and parts[0] == "2" and parts[1].isdigit():
        torchvision_pin = f"torchvision==0.{int(parts[1]) + 15}.0"
    python = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return (
        f"    uv venv --system-site-packages .venv-h3parity\n"
        f"    uv pip install --python .venv-h3parity/{python} "
        f"'transformers=={pinned_transformers_version()}'\n"
        f"    uv pip install --python .venv-h3parity/{python} --no-deps '{torchvision_pin}'\n"
        f"  (--system-site-packages reuses this box's torch {torch_version}; --no-deps keeps the "
        f"matching torchvision from dragging a second torch in. Alternatively install the pin into "
        f"the interpreter running pytest: pip install -e \".[h3-parity]\" — but read this module's "
        f"docstring first, that path upgrades huggingface_hub/safetensors/click globally.)"
    )


@functools.lru_cache(maxsize=1)
def parity_report() -> dict:
    """Run the probe ONCE per session under the pinned interpreter and return its JSON.

    Never skips. Every failure mode — no overlay, wrong version, crashed probe, unparseable output
    — returns a report the assertions below turn red, because the alternative is the false green
    this file exists to prevent.
    """
    interpreter = _overlay_python()
    if interpreter is None:
        env_override = os.environ.get("SIGNET_H3_PARITY_PYTHON")
        if env_override:
            interpreter = Path(env_override)
        elif _installed_version("transformers") == pinned_transformers_version():
            # CI that installed `.[h3-parity]` into the interpreter running pytest: use it directly.
            interpreter = Path(sys.executable)
    if interpreter is None or not Path(interpreter).exists():
        return {
            "ok": False,
            "error": (
                f"no interpreter with transformers=={pinned_transformers_version()} is available "
                f"(this one has {_installed_version('transformers')}). This check is NOT skippable "
                f"— a parity diff run at the wrong version passes while the real dispatch fails.\n"
                f"Bootstrap the pinned overlay once (~145 MB, gitignored):\n"
                f"{_bootstrap_instructions()}"
            ),
        }

    completed = subprocess.run(  # noqa: S603
        [str(interpreter), str(_RUNNER)],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        return {
            "ok": False,
            "error": (
                f"the parity probe produced no output (exit {completed.returncode}). "
                f"stderr:\n{completed.stderr[-4000:]}"
            ),
        }
    try:
        report = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"unparseable probe output ({exc}): {stdout[-4000:]}"}
    report.setdefault("stderr", completed.stderr[-4000:])
    return report


def _require_ok(report: dict) -> dict:
    assert report.get("ok"), (
        f"the D-10-DEF-7 parity probe did not complete.\n{report.get('error')}\n"
        f"{report.get('traceback', '')}"
    )
    return report


# ==================================================================================================
# The version identity — without this, everything below is theatre
# ==================================================================================================


def test_the_probe_runs_at_the_pinned_transformers_version() -> None:
    """A 5.2.0 reading describes a bug that does not exist at the version that actually runs."""
    report = _require_ok(parity_report())
    pinned = pinned_transformers_version()
    assert report["transformers_version"] == pinned, (
        f"the parity probe ran under transformers {report['transformers_version']}, not the pinned "
        f"{pinned} that modal/app.py installs into the image. The two versions DIFFER in exactly "
        f"the way that matters: return_mm_token_type_ids defaults True at {pinned} and False at "
        f"5.2.x, where the mask is built inline in the Qwen3-VL processor instead of in the shared "
        f"ProcessorMixin. A diff taken at the wrong version is a false green.\n"
        f"{_bootstrap_instructions()}"
    )
    assert report["processor_class"] == "Qwen3VLProcessor", (
        f"the probe must exercise the REAL Qwen3VLProcessor (the class the container mounts), got "
        f"{report['processor_class']!r}. A stub can only reproduce what we already believe __call__ "
        f"does, so it can never reveal an output nobody has heard of — which is the whole class "
        f"D-10-DEF-7 belongs to."
    )


def test_the_pinned_version_is_identical_here_and_on_the_image() -> None:
    """Local == image, or the gate is measuring a different library than the dispatch runs."""
    pinned = pinned_transformers_version()
    declared = [
        dep
        for dep in _parse_optional_dependencies().get("h3-parity", [])
        if dep.replace(" ", "").startswith("transformers")
    ]
    assert declared == [f"transformers=={pinned}"], (
        f"pyproject's [h3-parity] extra declares {declared}, but modal/app.py pins "
        f"transformers=={pinned}. They must be byte-identical and both must be LITERAL versions "
        f"(the DIFFUSERS_SHA discipline): a divergence between what CI checks and what the image "
        f"installs is precisely what made D-10-DEF-7 diagnosable only from a downloaded wheel."
    )


def _parse_optional_dependencies() -> dict[str, list[str]]:
    """Read ``[project.optional-dependencies]`` without requiring a TOML lib beyond the stdlib."""
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


# ==================================================================================================
# The class-closing diff
# ==================================================================================================


def test_our_builder_reproduces_every_processor_call_output_key() -> None:
    """The D-10-DEF-7 direction: a key ``__call__`` emits and we do not, by NAME."""
    report = _require_ok(parity_report())

    # Non-vacuity FIRST. `set() - set()` is empty, so a probe that produced nothing on both sides
    # would satisfy the diff below while proving nothing at all.
    call_keys = set(report["call_keys"])
    assert call_keys == set(_EXPECTED_CALL_KEYS), (
        f"the ORACLE returned {sorted(call_keys)}, not the {sorted(_EXPECTED_CALL_KEYS)} enumerated "
        f"from the pinned ProcessorMixin.__call__. Either the probe is broken (proving nothing), or "
        f"transformers changed its output surface — which is a deliberate review, not a number to "
        f"edit until this passes."
    )

    assert not report["missing"], (
        f"build_h3_processor_inputs does NOT produce {report['missing']}, which "
        f"processor.__call__ does. This is D-10-DEF-7's class: the bypass of __call__ is correct "
        f"and must stay, but it inherits ALL of that call's outputs. Add the key using the "
        f"processor's OWN public method for it (as mm_token_type_ids uses "
        f"create_mm_token_type_ids) — never a reimplementation, which re-derives a vocabulary the "
        f"processor already owns."
    )
    assert not report["extra"], (
        f"build_h3_processor_inputs produces {report['extra']}, which processor.__call__ does not. "
        f"Every key in the returned dict becomes a kwarg to text_encoder(**inputs) — an unexpected "
        f"one is a TypeError inside a metered container, on the LAST line of PHASE A."
    )
    assert set(report["our_keys"]) == call_keys, (
        f"key sets differ: ours {report['our_keys']} vs __call__'s {report['call_keys']}"
    )


def test_the_text_only_branch_matches_too() -> None:
    """``__call__`` adds ``mm_token_type_ids`` when TEXT is given, not when IMAGES are."""
    report = _require_ok(parity_report())
    call_keys = set(report["text_only_call_keys"])
    assert call_keys == set(_EXPECTED_TEXT_ONLY_KEYS), (
        f"the text-only oracle returned {sorted(call_keys)}, expected "
        f"{sorted(_EXPECTED_TEXT_ONLY_KEYS)} — the probe or the library surface changed"
    )
    assert set(report["text_only_our_keys"]) == call_keys, (
        f"the TEXT-ONLY branch diverges: ours {report['text_only_our_keys']} vs __call__'s "
        f"{report['text_only_call_keys']}. ProcessorMixin.__call__ (processing_utils.py:696) adds "
        f"mm_token_type_ids whenever text is given, so a builder that gated it on 'has references' "
        f"would be right on one branch and wrong on the other."
    )


def test_our_pre_expansion_equals_the_processors_own_expansion() -> None:
    """Free, and it is D-10-DEF-4 defect (1) in its SILENT form."""
    report = _require_ok(parity_report())
    assert report["input_ids_match"], (
        "our PRE-EXPANDED presentation and __call__'s own expansion of the same "
        "one-pad-per-image presentation produced DIFFERENT token streams. The version that RAISED "
        "on the live failure was a coincidence; a version whose replacement loop merely stopped "
        "would have conditioned the model at a plausible-but-wrong shape with nothing to catch it."
    )


def test_the_parity_allowlist_is_empty_and_every_entry_carries_a_reason() -> None:
    """An over-broad allowlist swallows exactly the gaps this file exists to surface."""
    report = _require_ok(parity_report())
    allowlist = report["allowlist"]
    assert allowlist == {}, (
        f"H3_PROCESSOR_PARITY_ALLOWLIST is {allowlist}, not empty. build_h3_processor_inputs is "
        f"documented as reproducing BatchFeature(data={{**text_inputs, **image_inputs}}) exactly, "
        f"so every key __call__ emits is a key the model may require. An entry added to make this "
        f"green is how the third metered container gets bought — it would have swallowed "
        f"mm_token_type_ids."
    )
    assert all(str(reason).strip() for reason in allowlist.values()), (
        "every allowlist entry must map to a WRITTEN reason, never an empty string"
    )
    assert not report["allowlisted"], (
        f"the diff was suppressed for {report['allowlisted']} by the allowlist — with an empty "
        f"allowlist that is impossible, so the probe is reporting inconsistently"
    )
