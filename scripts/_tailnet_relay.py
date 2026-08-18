"""Tailnet-only TCP relay: exposes localhost-bound review servers to the operator's tailnet devices.

Binds EXCLUSIVELY to this machine's Tailscale IP (100.x — WireGuard-encrypted, tailnet-only;
NEVER 0.0.0.0, honoring gridwatch's localhost-only posture: the private folder is reachable
only from the operator's own signed-in devices). One process, N port forwards. Stopgap until
`tailscale serve` is enabled on the tailnet (then HTTPS URLs replace this).

Usage: python scripts/_tailnet_relay.py <tailscale-ip> <port> [<port> ...]
"""
import asyncio
import ipaddress
import sys

#: Tailscale's CGNAT range (100.64.0.0/10) — the ONLY range this relay may bind to. Issue #40
#: finding 5: sys.argv[1] used to reach asyncio.start_server(host=...) with zero validation, so
#: a plausible typo (0.0.0.0 when `tailscale ip -4` isn't to hand) bound silently and exposed the
#: reviewed surface to the whole LAN despite the module docstring's "NEVER 0.0.0.0" promise.
_TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _validate_tailnet_host(raw: str) -> str:
    """Reject anything that is not a literal address inside Tailscale's CGNAT range.

    Exits (via SystemExit, caught by ``main``'s caller as a non-zero process exit) rather than
    binding, and names the command that produces a correct value.
    """
    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        raise SystemExit(
            f"[relay] refusing to bind: {raw!r} is not a valid IP address. Run `tailscale ip -4` "
            f"and pass ITS output as the host argument."
        )
    if addr not in _TAILSCALE_CGNAT:
        raise SystemExit(
            f"[relay] refusing to bind: {raw!r} is not in Tailscale's CGNAT range "
            f"({_TAILSCALE_CGNAT}) — binding outside it (0.0.0.0, ::, a LAN IP, ...) would expose "
            f"the reviewed surface beyond the tailnet, which this relay exists to prevent. Run "
            f"`tailscale ip -4` and pass ITS output as the host argument."
        )
    return raw


async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await r.read(65536)
            if not data:
                break
            w.write(data)
            await w.drain()
    except (ConnectionResetError, OSError):
        pass
    finally:
        try:
            w.close()
        except OSError:
            pass


async def handle(port: int, cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
    try:
        ur, uw = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        cw.close()
        return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))


async def main() -> None:
    ip, ports = _validate_tailnet_host(sys.argv[1]), [int(p) for p in sys.argv[2:]]
    servers = []
    for port in ports:
        servers.append(await asyncio.start_server(
            lambda r, w, p=port: handle(p, r, w), host=ip, port=port))
        print(f"[relay] {ip}:{port} -> 127.0.0.1:{port}", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
