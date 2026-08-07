"""Tailnet-only TCP relay: exposes localhost-bound review servers to the operator's tailnet devices.

Binds EXCLUSIVELY to this machine's Tailscale IP (100.x — WireGuard-encrypted, tailnet-only;
NEVER 0.0.0.0, honoring gridwatch's localhost-only posture: the private folder is reachable
only from the operator's own signed-in devices). One process, N port forwards. Stopgap until
`tailscale serve` is enabled on the tailnet (then HTTPS URLs replace this).

Usage: python scripts/_tailnet_relay.py <tailscale-ip> <port> [<port> ...]
"""
import asyncio
import sys


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
    ip, ports = sys.argv[1], [int(p) for p in sys.argv[2:]]
    servers = []
    for port in ports:
        servers.append(await asyncio.start_server(
            lambda r, w, p=port: handle(p, r, w), host=ip, port=port))
        print(f"[relay] {ip}:{port} -> 127.0.0.1:{port}", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main())
