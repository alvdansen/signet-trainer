"""Issue #40 finding 5 (minor) — `_tailnet_relay.py` must validate its host argument before binding.

**The defect.** `sys.argv[1]` reached ``asyncio.start_server(host=...)`` with zero validation: no
``100.`` prefix check, no rejection of ``0.0.0.0`` / ``::`` / empty. This is the repo's ONE sanctioned
non-loopback bind — every skill and sibling script (``serve_mask_app.py``,
``training-prep-inpaint/SKILL.md``, ``segmentation-prep/SKILL.md``) defers to it as "the safe path" —
yet a plausible typo (``0.0.0.0`` when ``tailscale ip -4`` isn't to hand) bound silently and exposed
the reviewed surface (decoded frames of licensed footage) to the whole LAN, guest wifi included,
despite the module's own docstring promising "NEVER 0.0.0.0".

The fix, per the issue's proposed direction: reject anything outside Tailscale's CGNAT range
(``100.64.0.0/10``) via ``ipaddress``, exiting non-zero with the ``tailscale ip -4`` recipe in the
message. Loaded via importlib (standalone script, not a package — the ``test_qa_overlay_h264.py``
precedent); no asyncio server is ever started here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    modname = f"{name}_under_test"
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


relay = _load("_tailnet_relay")


@pytest.mark.parametrize("bad_host", ["0.0.0.0", "::", "", "192.168.1.5", "10.0.0.1", "not-an-ip"])
def test_rejects_every_non_tailnet_host(bad_host: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        relay._validate_tailnet_host(bad_host)
    msg = str(exc_info.value)
    assert "tailscale ip -4" in msg, (
        f"the refusal for {bad_host!r} must name the command that produces a correct value, got: "
        f"{msg!r}"
    )


@pytest.mark.parametrize("good_host", ["100.64.0.1", "100.100.100.100", "100.127.255.254"])
def test_accepts_addresses_inside_the_tailscale_cgnat_range(good_host: str) -> None:
    assert relay._validate_tailnet_host(good_host) == good_host


def test_rejects_addresses_just_outside_the_cgnat_range() -> None:
    """100.64.0.0/10 spans 100.64.0.0-100.127.255.255 — one address on each side must be refused."""
    with pytest.raises(SystemExit):
        relay._validate_tailnet_host("100.63.255.255")
    with pytest.raises(SystemExit):
        relay._validate_tailnet_host("100.128.0.0")


def test_main_validates_argv_before_binding(monkeypatch) -> None:
    """`main()` must run the host through the validator before touching asyncio.start_server."""
    monkeypatch.setattr(sys, "argv", ["_tailnet_relay.py", "0.0.0.0", "8030"])

    async def _boom(*a, **k):
        raise AssertionError("asyncio.start_server must not be reached for a rejected host")

    monkeypatch.setattr(relay.asyncio, "start_server", _boom)

    import asyncio as real_asyncio

    with pytest.raises(SystemExit):
        real_asyncio.run(relay.main())
