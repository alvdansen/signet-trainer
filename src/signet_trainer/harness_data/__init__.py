"""Packaged harness SPEC/TEMPLATE data — the shipped defaults, resolvable from an installed wheel.

WHY THIS PACKAGE EXISTS. The harness modules (``harness_state``, ``harness_lint``, ``prep.spec``
and its callers) read five typed SPEC/TEMPLATE files. Those files used to live under
``.planning/harness/`` and were addressed as ``Path(__file__).parents[2] / ".planning" / ...`` —
a path that only resolves inside the maintainer's checkout. ``.planning/`` is a working-repo
planning surface that is deliberately NOT shipped, so a ``pip install signet-trainer`` produced
DANGLING DEFAULTS: every default path pointed at a directory the installed package does not have.
Shipping the specs AS PACKAGE DATA makes the defaults resolve for a fresh install with no
``.planning/`` at all.

WHAT LIVES HERE (SPEC / TEMPLATE — versioned house law, identical for every install)::

    MASK-SPEC.yaml                the inpaint mask conventions (D-03)
    MASK-SPEC-segmentation.yaml   the segmentation-prep variant of the same schema
    TIER-TAXONOMY.yaml            the knob->tier partition (D-30)
    HOUSE-SPEC.yaml               the house VERDICT spec (D-29)
    SESSION-STATE.template.json   the EMPTY session-ledger template (a template, not a ledger)

WHAT DELIBERATELY DOES **NOT** LIVE HERE (LIVE STATE — per-project, operator-written, never
shipped; it stays under ``.planning/harness/`` in the working repo)::

    SESSION-STATE.json            the LIVE spend ledger (note: no ``.template``)
    DECISION-LOG.md               the append-only episodic log
    KNOWLEDGE.md                  the distilled semantic memory
    cards/*.state.yaml            the live campaign cards

The distinction is the whole point: package data is the same for everyone and is read-only at
runtime; live state is one project's accumulating record and must never be baked into a wheel
(nor overwritten by an upgrade). Modules that default to LIVE state keep a PROJECT-RELATIVE
default and fail with an actionable message when it is absent — see
``harness_state.project_root``.

RESOLUTION MECHANISM. ``importlib.resources``, never ``Path(__file__)``. A path derived from
``__file__`` is correct only for a plain directory install; it breaks the moment the package is
imported from a zip (zipapp / pex / a frozen bundle). ``resources.files()`` returns a Traversable
that is correct in both cases, and ``spec_path`` materialises it to a real filesystem path
(extracting to a temp file only when the package is NOT a real directory) so the existing
path-taking loaders — ``prep.spec.load_mask_spec``, ``harness_state.load_house_spec``, etc. —
keep their ``str | Path`` signature unchanged.

OVERRIDE SEAM. Set ``SIGNET_HARNESS_DATA_DIR=/path/to/dir`` to point every packaged-spec lookup at
your own directory (per-house or per-campaign spec tuning without a fork — MASK-SPEC.yaml's own
comments call its coverage bands "tunable per campaign"). The override is EXPLICIT on purpose:
there is no implicit "a file in the CWD shadows the packaged one" rule, because silent shadowing
is exactly the two-sources-of-truth drift this package was created to remove. A named override
directory that is missing the requested file fails LOUD rather than falling back to the packaged
copy — a half-populated override would otherwise mix two spec generations silently.

Import-confined (Anti-Pattern 6): stdlib only — no yaml, no torch, no modal, no network. Parsing
is the CALLER's job (each spec has its own fail-loud, shape-checking loader).
"""

from __future__ import annotations

import atexit
import os
from contextlib import ExitStack
from importlib import resources
from pathlib import Path

# The importable name of this data package. Used as the ``resources.files()`` anchor; kept as a
# module constant so ``harness_lint`` (which must not import any ``signet_trainer`` module) can
# hardcode the same string and be cross-checked against it by a test.
PACKAGE = "signet_trainer.harness_data"

MASK_SPEC = "MASK-SPEC.yaml"
MASK_SPEC_SEGMENTATION = "MASK-SPEC-segmentation.yaml"
TIER_TAXONOMY = "TIER-TAXONOMY.yaml"
HOUSE_SPEC = "HOUSE-SPEC.yaml"
SESSION_STATE_TEMPLATE = "SESSION-STATE.template.json"

#: Every file this package ships. ``spec_path``/``spec_text`` refuse anything not listed here, so a
#: typo'd name is a loud ``KeyError`` instead of a confusing missing-file error at load time.
PACKAGED_SPECS: tuple[str, ...] = (
    MASK_SPEC,
    MASK_SPEC_SEGMENTATION,
    TIER_TAXONOMY,
    HOUSE_SPEC,
    SESSION_STATE_TEMPLATE,
)

#: Explicit override: a directory that supplies these files instead of the packaged copies.
OVERRIDE_ENV_VAR = "SIGNET_HARNESS_DATA_DIR"

# Materialised-resource lifetime. ``resources.as_file`` may create a temp file (zip import); the
# ExitStack keeps it alive for the process and cleans up at exit. For a normal directory install
# nothing is extracted and this stack stays empty.
_MATERIALIZED: dict[str, Path] = {}
_FILE_MANAGER = ExitStack()
atexit.register(_FILE_MANAGER.close)


def override_dir() -> Path | None:
    """The ``SIGNET_HARNESS_DATA_DIR`` override directory, or ``None`` when unset/blank."""
    raw = os.environ.get(OVERRIDE_ENV_VAR, "").strip()
    return Path(raw) if raw else None


def _check_name(name: str) -> str:
    if name not in PACKAGED_SPECS:
        raise KeyError(
            f"{name!r} is not packaged harness data; shipped specs are {list(PACKAGED_SPECS)}. "
            f"LIVE per-project state (SESSION-STATE.json, DECISION-LOG.md, KNOWLEDGE.md, "
            f"cards/*.state.yaml) is deliberately NOT packaged — address it project-relative."
        )
    return name


def spec_path(name: str) -> Path:
    """Real filesystem path of a packaged spec. Fails LOUD; never returns a dangling path.

    Resolution order: the ``SIGNET_HARNESS_DATA_DIR`` override directory (explicit, fail-loud when
    it lacks the file) -> the copy shipped inside this package. The result is a genuine
    ``Path``, materialised via ``importlib.resources.as_file`` so it is valid even when the
    package was imported from a zip.
    """
    _check_name(name)
    override = override_dir()
    if override is not None:
        candidate = override / name
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{OVERRIDE_ENV_VAR}={override} does not contain {name!r} ({candidate}). An "
                f"override directory must supply every spec it overrides — falling back to the "
                f"packaged copy would silently mix two spec generations. Add the file, or unset "
                f"{OVERRIDE_ENV_VAR} to use the packaged defaults."
            )
        return candidate

    cached = _MATERIALIZED.get(name)
    if cached is not None and cached.is_file():
        return cached
    ref = resources.files(PACKAGE).joinpath(name)
    if not ref.is_file():
        raise FileNotFoundError(
            f"packaged harness spec {name!r} is missing from {PACKAGE} — the install is "
            f"incomplete (the wheel must carry {list(PACKAGED_SPECS)}). Reinstall signet-trainer, "
            f"or set {OVERRIDE_ENV_VAR} to a directory holding the specs."
        )
    path = Path(_FILE_MANAGER.enter_context(resources.as_file(ref)))
    _MATERIALIZED[name] = path
    return path


def spec_text(name: str) -> str:
    """UTF-8 text of a packaged spec (honours the override). Zip-safe, no temp file needed."""
    _check_name(name)
    override = override_dir()
    if override is not None:
        return spec_path(name).read_text(encoding="utf-8")
    return resources.files(PACKAGE).joinpath(name).read_text(encoding="utf-8")
