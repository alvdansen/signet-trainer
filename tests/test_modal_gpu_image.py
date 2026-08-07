"""03-07 regression: every GPU Modal function must declare a KNOWN GPU image.

The app's DEFAULT image is code-only (no torch/ltx-core/peft/diffusers) — the heavy deps live on the
per-family GPU images. A ``@app.function(gpu=...)`` that forgets ``image=`` spins up an A100 and then
dies at ``import torch`` (metered spend for a guaranteed ImportError). ``train`` and ``sample`` both
shipped without it (03-06 was structurally tested, never launched). CPU source scan — zero spend.

Phase 10 (H3-07) widened this from the single hardcoded ``image=gpu_image`` substring to a KNOWN-IMAGE
SET, because the guard now covers TWO GPU image families:

  * ``gpu_image``    — the LTX-2.3 family (ltx-core / ltx-trainer / ltx-pipelines at LTX2_COMMIT_SHA)
  * ``h3_gpu_image`` — the MiniMax-H3 family (diffusers at DIFFUSERS_SHA)

It is a SET, deliberately NOT a per-function exemption list. The whole point of the test is that a
``gpu=`` with no image boots a GPU and dies at import, and that must stay true for H3 too — so a new
family widens the set by one name, and never earns a function-level opt-out. A ``gpu=`` paired with a
non-GPU image (e.g. the CPU ``download_image``) is just as fatal and is flagged the same way.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "signet_trainer" / "modal"
_FNS = _SRC / "fns.py"
_APP = _SRC / "app.py"

#: The GPU image families a ``gpu=`` function may declare. Widen by ONE NAME when a new model family
#: gets its own image; never add a per-function exemption.
_KNOWN_GPU_IMAGES = ("gpu_image", "h3_gpu_image")

# Each @app.function(...) decorator paired with its function name. The decorators contain no nested
# parens (only {}/[] for volumes/secrets), so a non-greedy match to the first ')' is safe.
_DECORATOR_RE = re.compile(r"@app\.function\((.*?)\)\s*\ndef\s+(\w+)", re.DOTALL)


def _blocks(src: str) -> list[tuple[str, str]]:
    """Every ``(decorator_args, function_name)`` pair in ``src``."""
    return _DECORATOR_RE.findall(src)


def _offenders(src: str) -> list[str]:
    """Names of ``gpu=`` functions that do not declare an image from ``_KNOWN_GPU_IMAGES``."""
    offenders = []
    for args, name in _blocks(src):
        # Strip line comments so prose like "intentionally NO gpu=" doesn't count as a real kwarg.
        args_nc = re.sub(r"#.*", "", args)
        if "gpu=" not in args_nc:
            continue
        declared = set(re.findall(r"image=(\w+)", args_nc))
        if not declared & set(_KNOWN_GPU_IMAGES):
            offenders.append(name)
    return offenders


def test_all_gpu_functions_set_a_known_gpu_image() -> None:
    src = _FNS.read_text(encoding="utf-8")
    assert _blocks(src), "expected @app.function decorators in fns.py"
    offenders = _offenders(src)
    assert not offenders, (
        f"GPU function(s) missing a known GPU image {_KNOWN_GPU_IMAGES} — they would boot a GPU then "
        f"fail at import torch on the code-only default image: {offenders}"
    )


def test_known_gpu_images_are_actually_defined_in_app() -> None:
    """The set must name REAL images — otherwise it silently widens into a rubber stamp."""
    app_src = _APP.read_text(encoding="utf-8")
    missing = [
        name for name in _KNOWN_GPU_IMAGES if not re.search(rf"^{name} = \(", app_src, re.MULTILINE)
    ]
    assert not missing, f"_KNOWN_GPU_IMAGES names not defined in app.py: {missing}"


def test_h3_diffusers_pin_is_a_literal_40_hex_sha() -> None:
    """H3-07 supply chain: ``diffusers`` is git-installed at a LITERAL pinned SHA, never ``main``.

    The install line INTERPOLATES the ``DIFFUSERS_SHA`` constant (the same single-source-of-truth
    shape as ``LTX2_COMMIT_SHA``'s ``git checkout {LTX2_COMMIT_SHA}``), so this asserts the two halves
    that together make the effective URL tamper-evident: the constant is a literal 40-hex SHA, and the
    git+https line is pinned to that constant rather than to a moving ref.
    """
    src = _APP.read_text(encoding="utf-8")

    sha = re.search(r'^DIFFUSERS_SHA = "([0-9a-f]{40})"$', src, re.MULTILINE)
    assert sha, "DIFFUSERS_SHA must be a literal 40-hex commit SHA assigned at module scope"

    pin = re.search(
        r"diffusers @ git\+https://github\.com/huggingface/diffusers@"
        r"(?:\{DIFFUSERS_SHA\}|[0-9a-f]{40})",
        src,
    )
    assert pin, "the diffusers git install must be pinned to DIFFUSERS_SHA (or a literal 40-hex SHA)"

    assert "huggingface/diffusers@main" not in src, (
        "diffusers must never be installed from `main` — MiniMax-H3's surface is still moving, so an "
        "unpinned ref makes the image build silently non-reproducible"
    )


def test_h3_transformers_pin_is_a_literal_version() -> None:
    """D-10-DEF-4: ``transformers`` is conditioning-critical on the H3 leg, so it gets a literal pin.

    ``transformers`` owns ``Qwen3VLProcessor`` — the object that decides how many vision tokens each
    reference expands to, i.e. what the model is conditioned on. The old ``'transformers>=4.51'``
    range is what made the first metered dispatch's failure a COINCIDENCE rather than a guard: it
    resolved to a version whose replacement path raises on a pre-expanded presentation, while a 4.x
    resolution of the SAME range would have left the surplus pads un-expanded and conditioned the
    model at a plausible-but-wrong shape with nothing to catch it.

    The pre-encode no longer depends on that exception (it checks the realized expansion
    arithmetically — ``tests/test_h3_reference_geometry_agreement.py``), but a floating range on a
    conditioning-critical package is a reproducibility hole regardless: the image that TRAINS an
    adapter must be the image that RENDERS it.
    """
    src = _APP.read_text(encoding="utf-8")

    version = re.search(r'^TRANSFORMERS_VERSION = "([0-9][0-9A-Za-z.\-]*)"$', src, re.MULTILINE)
    assert version, (
        "TRANSFORMERS_VERSION must be a literal version assigned at module scope, the same "
        "single-source-of-truth shape as DIFFUSERS_SHA"
    )

    assert re.search(r"transformers==\{TRANSFORMERS_VERSION\}", src), (
        "the h3_gpu_image install must pin transformers to that constant with `==`, interpolated "
        "rather than restated (a second literal is a second thing that can drift)"
    )
    # Comments stripped first: this file DOCUMENTS the range it replaced, and an un-stripped scan
    # would fail on the explanation rather than on an actual install line.
    installs = [
        line
        for line in re.sub(r"#.*", "", src).splitlines()
        if "transformers" in line
    ]
    assert installs, "the transformers install line vanished — the scan is broken, not the image"
    assert not [line for line in installs if "transformers>=" in line], (
        f"a `transformers>=` range survives in an install line ({installs}): a range on the package "
        f"that owns Qwen3VLProcessor means the vision-token expansion the adapter was conditioned "
        f"on is decided at build time, by whatever the resolver picked that day"
    )


def test_negative_control_bare_gpu_is_flagged() -> None:
    """The real logic (not a copy of it) must still catch a ``gpu=`` with no image at all."""
    synthetic = '''
@app.function(
    gpu="A100-80GB",
    volumes={"/weights": weights_vol},
)
def forgot_the_image() -> str:
    ...
'''
    assert _offenders(synthetic) == ["forgot_the_image"]


def test_negative_control_wrong_image_family_is_flagged() -> None:
    """A ``gpu=`` paired with a NON-GPU image is just as fatal as no image — set, not "any image"."""
    synthetic = '''
@app.function(gpu="A100-80GB", image=download_image)
def cpu_image_on_a_gpu() -> str:
    ...
'''
    assert _offenders(synthetic) == ["cpu_image_on_a_gpu"]


def test_positive_control_h3_image_is_accepted() -> None:
    """A GPU function declaring the H3 family image must NOT be flagged (H3-07)."""
    synthetic = '''
@app.function(
    gpu="A100-80GB",
    image=h3_gpu_image,
    volumes={"/weights": weights_vol},
)
def h3_stage() -> str:
    ...
'''
    assert _offenders(synthetic) == []
