#!/usr/bin/env bash
#
# setup_gridwatch.sh — one-time LOCAL install of finetune-gridwatch (the operator's canonical
# sampler-grid harness) for the training-review skill.
#
# WHO RUNS THIS: Claude, once per machine (Harvest Sources & Testing Protocol — Claude runs
# all scripts; the operator never SSHes or runs setup themselves). The operator only does visual approval of the
# grids this tool builds.
#
# SUPPLY-CHAIN RATIONALE (08-RESEARCH Package Legitimacy Audit — APPROVED):
#   * FIRST-PARTY: github.com/alvdansen/finetune-gridwatch is a first-party public repo. It is NOT a
#     random PyPI package, so it needs no slopcheck / package-legitimacy gate — but it is still
#     PINNED to a literal commit SHA for reproducibility + tamper-evidence, exactly like the repo's
#     LTX2_COMMIT_SHA precedent (never a moving ref like `master`).
#   * ISOLATED VENV: gridwatch is installed into its OWN venv (_tools/finetune-gridwatch/.venv) so
#     its fastapi / uvicorn / jinja2 / watchfiles / sse-starlette pins can NEVER perturb the trainer
#     environment (torch / ltx-core / peft / bitsandbytes). The two dependency worlds stay disjoint.
#   * NEVER IN THE MODAL IMAGE: gridwatch is a LOCAL, agent-side rendering tool. It is deliberately
#     NOT added to gpu_image / download_image or any Modal image build — the trainer image carries
#     none of gridwatch's web-server deps. Grids are built on this machine from samples fetched via
#     `modal volume get`, never inside a metered container.
#
# IDEMPOTENT: safe to re-run. Skips the clone if the checkout already exists, but ALWAYS re-pins to
# GRIDWATCH_COMMIT and re-verifies the `grid` entry point.
#
# Requires: git, gh (GitHub CLI, authenticated), and either uv (preferred) or python>=3.11 (gridwatch
# is requires-python >=3.11). Deps it pulls: jinja2 typer pillow fastapi uvicorn watchfiles sse-starlette.

set -euo pipefail

# ── PINNED commit SHA (literal — NEVER `master`; repo precedent is literal-SHA pinning like
#    LTX2_COMMIT_SHA=d6053703). Resolved via `gh api repos/alvdansen/finetune-gridwatch/commits/master`
#    at setup on 2026-07-10. To bump: re-resolve that SHA, replace the literal below, re-run this script.
GRIDWATCH_COMMIT="e82808979d9cfc6c60d861f7dc38b382c5745086"

GRIDWATCH_REPO="alvdansen/finetune-gridwatch"

# Resolve the repo root so the tool lands in a stable sibling dir regardless of cwd.
REPO_ROOT="$(git rev-parse --show-toplevel)"
TOOLS_DIR="${REPO_ROOT}/_tools"
GRIDWATCH_DIR="${TOOLS_DIR}/finetune-gridwatch"

echo "[setup_gridwatch] target: ${GRIDWATCH_DIR}"
echo "[setup_gridwatch] pinned commit: ${GRIDWATCH_COMMIT}"

mkdir -p "${TOOLS_DIR}"

# ── (1) Clone (idempotent — skip if the checkout already exists) ───────────────────────────────
if [ -d "${GRIDWATCH_DIR}/.git" ]; then
  echo "[setup_gridwatch] checkout exists — skipping clone, will re-pin."
  git -C "${GRIDWATCH_DIR}" fetch --quiet origin || true
else
  echo "[setup_gridwatch] cloning ${GRIDWATCH_REPO} via gh ..."
  gh repo clone "${GRIDWATCH_REPO}" "${GRIDWATCH_DIR}"
fi

# ── (2) Check out the PINNED SHA (always — even on an existing checkout) ────────────────────────
git -C "${GRIDWATCH_DIR}" checkout --quiet "${GRIDWATCH_COMMIT}"
RESOLVED_SHA="$(git -C "${GRIDWATCH_DIR}" rev-parse HEAD)"
if [ "${RESOLVED_SHA}" != "${GRIDWATCH_COMMIT}" ]; then
  echo "[setup_gridwatch] FATAL: resolved HEAD ${RESOLVED_SHA} != pinned ${GRIDWATCH_COMMIT}" >&2
  exit 1
fi
echo "[setup_gridwatch] checked out pinned SHA: ${RESOLVED_SHA}"

# ── (3) Create the ISOLATED venv and install gridwatch editable into IT (never the trainer env) ─
cd "${GRIDWATCH_DIR}"
if command -v uv >/dev/null 2>&1; then
  echo "[setup_gridwatch] using uv for the isolated venv ..."
  uv venv --python 3.11 .venv
  # shellcheck disable=SC1091
  uv pip install --python .venv -e .
  GRID_BIN="${GRIDWATCH_DIR}/.venv/bin/grid"
  [ -x "${GRID_BIN}" ] || GRID_BIN="${GRIDWATCH_DIR}/.venv/Scripts/grid.exe"  # Windows layout
else
  echo "[setup_gridwatch] uv not found — falling back to python -m venv + pip ..."
  python -m venv .venv
  if [ -x ".venv/bin/pip" ]; then
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -e .
    GRID_BIN="${GRIDWATCH_DIR}/.venv/bin/grid"
  else  # Windows venv layout
    .venv/Scripts/pip.exe install --upgrade pip
    .venv/Scripts/pip.exe install -e .
    GRID_BIN="${GRIDWATCH_DIR}/.venv/Scripts/grid.exe"
  fi
fi

# ── (4) Verify the entry point + print the resolved SHA and grid version ────────────────────────
echo "[setup_gridwatch] verifying '${GRID_BIN} --help' ..."
"${GRID_BIN}" --help >/dev/null
GRID_VERSION="$("${GRID_BIN}" --version 2>/dev/null || echo 'version-flag-unavailable')"

echo "[setup_gridwatch] OK — finetune-gridwatch installed (ISOLATED venv, first-party, pinned)."
echo "[setup_gridwatch]   commit : ${RESOLVED_SHA}"
echo "[setup_gridwatch]   version: ${GRID_VERSION}"
echo "[setup_gridwatch]   grid   : ${GRID_BIN}"
echo "[setup_gridwatch] The training-review skill invokes this grid binary; it is NEVER baked into"
echo "[setup_gridwatch] the Modal image — grids are built locally from 'modal volume get' samples."
