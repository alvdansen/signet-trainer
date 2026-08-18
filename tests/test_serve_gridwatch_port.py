"""Issue #40 finding 2 — `-Port` must reach `grid watch`, not just the printed URL/tunnel.

**The defect.** `serve_gridwatch.ps1` built its printed URL (`http://127.0.0.1:$Port`) and its
ngrok/cloudflared tunnel target from `$Port`, but the `ArgumentList` handed to `grid watch` was
`@("watch", $GridDir)` — `$Port` never reached the child process. `grid watch` defaults its own
`--port` to 8000, so the script only worked by coincidence when the caller left `-Port` at its
default AND 8000 happened to be free; any explicit non-default `-Port` (which the script's own
usage line advertises) broke deterministically.

Static source-scan only (never executed) — matches the convention in
`test_watcher_hardening.py`'s `watcher_supervisor.ps1` checks: a `.ps1` is Windows-only and this
suite must stay runnable on any CI box.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "serve_gridwatch.ps1"


def _source() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def test_script_exists() -> None:
    assert _SCRIPT.exists(), "serve_gridwatch.ps1 must exist"


def test_grid_watch_argument_list_forwards_port() -> None:
    """The `grid watch` `-ArgumentList` must carry `--port` and `$Port`, not just `watch`/`$GridDir`."""
    src = _source()
    m = re.search(r"-ArgumentList\s+@\(([^)]*)\)", src)
    assert m is not None, "no -ArgumentList @(...) call found — grid watch is no longer invoked this way"
    arg_list = m.group(1)
    assert "watch" in arg_list and "$GridDir" in arg_list, (
        f"-ArgumentList changed shape unexpectedly: {arg_list!r}"
    )
    assert "--port" in arg_list and "$Port" in arg_list, (
        f"-ArgumentList does not forward the port to `grid watch`: {arg_list!r}. The printed URL "
        f"and the tunnel target are both built from $Port, so the child must see the same value or "
        f"the two silently diverge (issue #40 finding 2)."
    )


def test_printed_url_and_tunnel_still_use_the_same_port_variable() -> None:
    """Regression guard: the fix must not stop the URL/tunnel from tracking $Port either."""
    src = _source()
    assert "http://127.0.0.1:$Port" in src, "printed URL no longer built from $Port"
    assert 'ngrok http $Port' in src, "ngrok tunnel target no longer built from $Port"
    assert 'http://127.0.0.1:$Port' in src.split("cloudflared tunnel --url")[-1], (
        "cloudflared tunnel target no longer built from $Port"
    )
