"""prep.app — stdlib ``http.server`` masking app (D-06..D-11). The phase's primary UI surface.

A zero-build web-canvas tool (extends the ``scripts/_serve_clip_picker.py`` house shell) where users
draw inpaint masks BY HAND over video frames — D-06 makes "users drawing the mask" a FIRST-CLASS
requirement. The binary-by-construction export guarantee (D-09) lives SERVER-SIDE: on export the
server thresholds every pixel (``alpha > 127 -> 255``) before writing the PNG, so the mask is pure
binary ``{0, 255}`` regardless of the browser's anti-aliasing. Region = WHITE(255) per the D-05
polarity chain; staging (``prep.stage``) negates downstream — the app NEVER negates.

Routes (see :func:`make_handler`):
  * ``GET /``                    — the four-zone canvas-painter client (Task 2; inline style/script).
  * ``GET /api/spec``            — the spec brush hex + coverage bands (client reads teal from HERE,
                                   never a client literal — D-01/D-03).
  * ``GET /api/clips``           — the clips list + per-clip frame counts (the frame-strip source).
  * ``GET /frame/<clip>/<n>``    — a clip's decoded frame N as PNG (traversal-guarded on the clip).
  * ``GET /api/review``          — the propagation-review/QA grid data (masks_video coverage).
  * ``GET /masks_video/...``     — serve a written per-frame mask PNG (traversal-guarded).
  * ``POST /export``             — decode the posted RGBA overlay, threshold SERVER-SIDE, write the
                                   ``masks_frame0/<stem>.png`` seed and/or ``masks_video/<stem>/NNNNN.png``.
  * ``POST /paintover``          — route an uploaded teal PNG through :func:`prep.chroma.extract_chroma_mask`
                                   (D-10) and write the same seed layout.

Security (the export POST is the ONLY new network surface):
  * bind ``127.0.0.1`` ALWAYS (:func:`serve` refuses a non-loopback host); tailnet exposure is opt-in
    via ``scripts/_tailnet_relay.py`` only — the app NEVER binds a wildcard/public interface (T-09.3-06-03).
  * every write target is resolved and asserted UNDER the prep dir (:func:`resolve_under_dir`,
    mirroring the shell ``f.parent == DIR`` guard) — ``../`` traversal is rejected (T-09.3-06-01).
  * POST bodies are size-capped BEFORE buffering (:func:`check_payload_size`) — T-09.3-06-02.

Import confinement (Anti-Pattern 6): module top imports stdlib only. ``numpy`` (the export threshold)
and ``cv2`` (PNG/frame decode) are FUNCTION-LOCAL, so ``import signet_trainer.prep.app`` never pulls a
numeric/decode stack and the export helpers unit-test free on Windows/CI. NO modal, NO torch, NO GPU,
NO metered anything — pure local file prep.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from signet_trainer.prep.chroma import extract_chroma_mask
from signet_trainer.prep.spec import load_mask_spec

# The client (Task 2) is assembled from these constants so the whole tool ships zero-build. Task 1
# ships a functional-but-minimal placeholder; Task 2 replaces the body/style/script with the full
# four-zone UI-SPEC contract. render_page() injects the spec brush hex so the teal is never a literal.
from signet_trainer.prep._app_client import CLIENT_BODY, CLIENT_SCRIPT, CLIENT_STYLE

__all__ = ["AppConfig", "make_handler", "serve", "render_page"]

#: House rule — the masking app binds loopback ONLY. Tailnet exposure goes through
#: ``scripts/_tailnet_relay.py`` (opt-in, WireGuard-encrypted); the app never binds a wildcard interface.
LOOPBACK = "127.0.0.1"

#: T-09.3-06-02 — reject POST bodies larger than this before buffering (DoS guard). An overlay PNG
#: for a 4K frame is well under this; a config knob keeps it tunable (config-first).
DEFAULT_MAX_POST_BYTES = 48 * 1024 * 1024

#: A safe export stem: alnum plus ``_ - .`` only, no path separators, no ``..`` — the write layout is
#: ``masks_frame0/<stem>.png`` / ``masks_video/<stem>/NNNNN.png`` (belt-and-braces with resolve_under_dir).
_SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class AppConfig:
    """Everything the handler closes over. ``spec`` is a loaded MASK-SPEC dict (never re-read per request)."""

    clips_dir: Path
    prep_dir: Path
    spec: dict
    spec_path: Path
    max_post_bytes: int = DEFAULT_MAX_POST_BYTES


# --------------------------------------------------------------------------------------------------
# Pure helpers — directly unit-testable, no live socket. numpy is function-local (import-confined).
# --------------------------------------------------------------------------------------------------


def _need_numpy():
    """Import numpy lazily with a clear hint (module must PARSE without it)."""
    try:
        import numpy  # noqa: PLC0415 — deliberate function-local import (import-confined)

        return numpy
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "[prep.app] missing dependency 'numpy': "
            f"{exc}\n    install hint: pip install numpy"
        ) from exc


def threshold_alpha_to_binary(rgba):
    """RGBA overlay (anti-aliased grey edges) -> pure binary ``{0, 255}`` HxW mask, region = WHITE(255).

    The D-09 guarantee, SERVER-SIDE: any pixel with ``alpha > 127`` becomes 255, else 0 — the
    browser's anti-aliasing is discarded, so no intermediate greys survive. ``rgba`` is an HxWx4
    uint8 array (as decoded from the posted overlay PNG); a non-RGBA array raises ``ValueError``
    (a missing alpha channel means the client posted the wrong buffer — fail loud, never a silent
    empty mask).
    """
    np = _need_numpy()
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] < 4:
        raise ValueError(
            f"[prep.app] export overlay must be an HxWx4 RGBA array, got shape {arr.shape} — "
            "the client must post the transparent paint overlay (with an alpha channel)."
        )
    # region = WHITE(255) per D-05; threshold on the alpha channel regardless of browser AA.
    return (arr[..., 3] > 127).astype(np.uint8) * 255


def _assert_safe_stem(stem: str) -> None:
    """Reject a stem carrying path separators / ``..`` before it ever touches the filesystem."""
    if not isinstance(stem, str) or not _SAFE_STEM.match(stem) or ".." in stem:
        raise ValueError(
            f"[prep.app] unsafe export stem {stem!r} — stems are alnum + [._-] only, no path "
            "separators, no '..' (path-traversal guard T-09.3-06-01)."
        )


def export_target_rel(stem: str, kind: str, frame: int | None) -> str:
    """Map ``(stem, kind, frame)`` to the D-05 write layout rel path. Validates the stem is safe.

    ``kind='frame0'`` -> ``masks_frame0/<stem>.png`` (the seed); ``kind='video'`` ->
    ``masks_video/<stem>/NNNNN.png`` (per-frame, 5-digit zero-padded). Any other kind raises.
    """
    _assert_safe_stem(stem)
    if kind == "frame0":
        return f"masks_frame0/{stem}.png"
    if kind == "video":
        if frame is None:
            raise ValueError("[prep.app] kind='video' export requires a frame index")
        return f"masks_video/{stem}/{int(frame):05d}.png"
    raise ValueError(f"[prep.app] unknown export kind {kind!r} (expected 'frame0' or 'video')")


def resolve_under_dir(base: Path | str, rel: str) -> Path:
    """Resolve ``base/rel`` and assert it stays UNDER ``base`` — reject ``../`` traversal (T-09.3-06-01).

    Mirrors the ``_serve_clip_picker.py`` ``f.parent == CLIPS_DIR`` shape but allows one nested
    subdir (``masks_video/<stem>/``). Raises ``ValueError`` when the resolved target escapes ``base``.
    """
    base_r = Path(base).resolve()
    target = (base_r / rel).resolve()
    if target != base_r and base_r not in target.parents:
        raise ValueError(
            f"[prep.app] export target {rel!r} escapes the prep dir {base_r} — refusing the write "
            "(path-traversal guard T-09.3-06-01)."
        )
    return target


def check_payload_size(content_length: int, cap: int) -> None:
    """Reject an over-cap POST body BEFORE buffering (T-09.3-06-02). ``content_length == cap`` is allowed."""
    if content_length > cap:
        raise ValueError(
            f"[prep.app] POST body {content_length}B exceeds the {cap}B cap — refusing to buffer "
            "(payload-cap DoS guard T-09.3-06-02)."
        )


# --------------------------------------------------------------------------------------------------
# Image I/O — cv2 is function-local (import-confined). PNG decode/encode + clip-frame decode.
# --------------------------------------------------------------------------------------------------


def _need_cv2():
    try:
        import cv2  # noqa: PLC0415 — function-local (import-confined; the operator-box decode dep)

        return cv2
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            "[prep.app] missing dependency 'cv2' (opencv-python): "
            f"{exc}\n    install hint: pip install opencv-python"
        ) from exc


def _decode_png_rgba(png_bytes: bytes):
    """Decode a posted overlay PNG to an HxWx4 array (cv2 gives BGRA; alpha stays channel 3)."""
    np = _need_numpy()
    cv2 = _need_cv2()
    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("[prep.app] could not decode the posted overlay PNG (corrupt upload).")
    if img.ndim != 3 or img.shape[2] < 4:
        raise ValueError(
            "[prep.app] posted overlay has no alpha channel — the paint overlay must be RGBA."
        )
    return img  # BGRA — threshold_alpha_to_binary only reads channel 3 (alpha), order-agnostic


def _decode_png_bgr(png_bytes: bytes):
    """Decode an uploaded paintover PNG to HxWx3 BGR (the shape prep.chroma expects). Fails loud."""
    np = _need_numpy()
    cv2 = _need_cv2()
    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, drops alpha
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("[prep.app] could not decode the uploaded paintover PNG (corrupt/greyscale).")
    return img


def _encode_gray_png(mask_u8) -> bytes:
    """Encode an HxW uint8 mask to a grayscale PNG (binary {0,255} by construction)."""
    cv2 = _need_cv2()
    ok, buf = cv2.imencode(".png", mask_u8)
    if not ok:
        raise RuntimeError("[prep.app] failed to PNG-encode the exported mask.")
    return buf.tobytes()


def write_binary_mask(prep_dir: Path | str, stem: str, rgba, *, kind: str = "frame0", frame: int | None = None):
    """Threshold an RGBA overlay SERVER-SIDE and write the binary WHITE mask in the D-05 layout.

    Returns ``(target_path, mask_u8)``. The write target is resolved and asserted under ``prep_dir``
    (traversal guard) before any bytes hit disk. The written PNG is binary ``{0, 255}`` by
    construction — the browser's anti-aliasing never reaches disk.
    """
    mask = threshold_alpha_to_binary(rgba)
    rel = export_target_rel(stem, kind, frame)
    target = resolve_under_dir(prep_dir, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_encode_gray_png(mask))
    return target, mask


def import_paintover(prep_dir: Path | str, stem: str, png_bytes: bytes, spec: dict):
    """Route an uploaded teal paintover through the rebuilt chroma extractor (D-10) -> WHITE seed.

    Decodes the PNG to BGR, validates it is HxWx3 (a corrupt/greyscale decode raises — ASVS V5 /
    T-09.3-06-05), runs :func:`prep.chroma.extract_chroma_mask` with the loaded spec (never a
    hardcoded teal), and writes the ``masks_frame0/<stem>.png`` seed. Both the canvas-draw path and
    this import path converge on the SAME binary WHITE seed layout. Returns ``(target, mask, record)``.
    """
    bgr = _decode_png_bgr(png_bytes)
    mask, record = extract_chroma_mask(bgr, spec, match=stem)
    rel = export_target_rel(stem, "frame0", None)
    target = resolve_under_dir(prep_dir, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_encode_gray_png(mask))
    return target, mask, record


def _clip_frame_count(clip: Path) -> int:
    cv2 = _need_cv2()
    cap = cv2.VideoCapture(str(clip))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _decode_clip_frame(clip: Path, n: int) -> bytes:
    """Decode frame ``n`` of a clip to PNG bytes (cv2, function-local)."""
    cv2 = _need_cv2()
    cap = cv2.VideoCapture(str(clip))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n)))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"[prep.app] could not decode frame {n} of {clip.name}")
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError(f"[prep.app] failed to PNG-encode frame {n} of {clip.name}")
        return buf.tobytes()
    finally:
        cap.release()


def _coverage_target_map(spec: dict) -> dict:
    """The per-subject-class coverage targets + warn bands from the spec (client color-bands against these)."""
    out = {}
    for name, band in spec.get("coverage_bands", {}).items():
        if isinstance(band, dict) and "target" in band:
            out[name] = {"target": band["target"], "warn_below": band.get("warn_below")}
    return out


def render_page(config: AppConfig) -> str:
    """Assemble the zero-build client HTML, injecting the spec brush hex (never a client literal)."""
    hexcol = config.spec["hex"]
    body = CLIENT_BODY.replace("__BRUSH_HEX__", hexcol)
    script = CLIENT_SCRIPT.replace("__BRUSH_HEX__", hexcol)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>signet mask app</title>{CLIENT_STYLE}</head><body>{body}{script}</body></html>"


# --------------------------------------------------------------------------------------------------
# The handler — closes over an AppConfig. Routing is explicit; every write is traversal-guarded.
# --------------------------------------------------------------------------------------------------


def make_handler(config: AppConfig):
    """Produce a ``BaseHTTPRequestHandler`` subclass bound to ``config`` (clips/prep/spec)."""

    prep_dir = Path(config.prep_dir).resolve()
    clips_dir = Path(config.clips_dir).resolve()

    class MaskAppHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet (house shell convention)
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        # ---- GET ---------------------------------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, render_page(config).encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/spec":
                self._send_json(200, {
                    "hex": config.spec["hex"],
                    "rgb": config.spec["rgb"],
                    "coverage_bands": _coverage_target_map(config.spec),
                })
                return
            if path == "/api/clips":
                self._send_json(200, {"clips": self._clip_index()})
                return
            if path == "/api/review":
                self._send_json(200, {"clips": self._review_index()})
                return
            if path.startswith("/frame/"):
                self._serve_frame(path)
                return
            if path.startswith("/masks_video/"):
                self._serve_written_mask(path)
                return
            self._send(404, b"not found", "text/plain")

        def _clip_index(self) -> list[dict]:
            clips = []
            for c in sorted(clips_dir.glob("*.mp4")):
                try:
                    n = _clip_frame_count(c)
                except SystemExit:  # cv2 missing — degrade to an unknown count, still list the clip
                    n = None
                clips.append({"name": c.name, "stem": c.stem, "frames": n})
            return clips

        def _review_index(self) -> list[dict]:
            base = prep_dir / "masks_video"
            out = []
            if base.is_dir():
                for d in sorted(p for p in base.iterdir() if p.is_dir()):
                    pngs = sorted(d.glob("*.png"))
                    out.append({"stem": d.name, "frames": len(pngs)})
            return out

        def _serve_frame(self, path: str) -> None:
            # /frame/<clip>/<n>
            rest = path[len("/frame/"):]
            clip_name, _, n_str = rest.rpartition("/")
            clip_name = unquote(clip_name)
            f = (clips_dir / clip_name).resolve()
            if f.parent != clips_dir or not f.is_file():  # traversal guard (shell shape)
                self._send(404, b"clip not found", "text/plain")
                return
            try:
                png = _decode_clip_frame(f, int(n_str))
            except (ValueError, RuntimeError):
                self._send(404, b"frame not found", "text/plain")
                return
            self._send(200, png, "image/png")

        def _serve_written_mask(self, path: str) -> None:
            rel = unquote(path[len("/"):])  # 'masks_video/<stem>/NNNNN.png'
            try:
                f = resolve_under_dir(prep_dir, rel)
            except ValueError:
                self._send(404, b"not found", "text/plain")
                return
            if f.is_file():
                self._send(200, f.read_bytes(), "image/png")
                return
            self._send(404, b"not found", "text/plain")

        # ---- POST --------------------------------------------------------------------------------
        def _read_capped_body(self) -> bytes | None:
            length = int(self.headers.get("Content-Length", 0))
            try:
                check_payload_size(length, config.max_post_bytes)
            except ValueError as exc:
                self._send(413, str(exc).encode("utf-8"), "text/plain; charset=utf-8")
                return None
            return self.rfile.read(length)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            body = self._read_capped_body()
            if body is None:
                return  # already 413'd
            try:
                payload = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send(400, b"bad json", "text/plain")
                return
            if path == "/export":
                self._handle_export(payload)
                return
            if path == "/paintover":
                self._handle_paintover(payload)
                return
            self._send(404, b"not found", "text/plain")

        @staticmethod
        def _decode_dataurl(png_field: str) -> bytes:
            # accepts 'data:image/png;base64,....' or a bare base64 string
            b64 = png_field.split(",", 1)[1] if png_field.startswith("data:") else png_field
            return base64.b64decode(b64)

        def _handle_export(self, payload: dict) -> None:
            try:
                stem = payload["stem"]
                kind = payload.get("kind", "frame0")
                frame = payload.get("frame")
                rgba = _decode_png_rgba(self._decode_dataurl(payload["png"]))
                target, _ = write_binary_mask(prep_dir, stem, rgba, kind=kind, frame=frame)
            except (KeyError, ValueError, base64.binascii.Error) as exc:
                self._send(400, f"export rejected: {exc}".encode("utf-8"), "text/plain; charset=utf-8")
                return
            self._send_json(200, {"written": str(target.relative_to(prep_dir))})

        def _handle_paintover(self, payload: dict) -> None:
            try:
                stem = payload["stem"]
                png = self._decode_dataurl(payload["png"])
                target, _, record = import_paintover(prep_dir, stem, png, config.spec)
            except (KeyError, ValueError, base64.binascii.Error) as exc:
                self._send(400, f"paintover rejected: {exc}".encode("utf-8"), "text/plain; charset=utf-8")
                return
            self._send_json(200, {"written": str(target.relative_to(prep_dir)), "record": record})

    return MaskAppHandler


def serve(host: str = LOOPBACK, port: int = 8030, *, clips_dir, prep_dir, spec_path, max_post_bytes: int = DEFAULT_MAX_POST_BYTES) -> None:
    """Bind ``127.0.0.1`` ALWAYS and serve the masking app. Refuses any non-loopback host (house rule).

    Tailnet exposure is opt-in via ``scripts/_tailnet_relay.py`` — the app NEVER binds a wildcard/public interface.
    """
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(
            f"[prep.app] refusing to bind {host!r} — the masking app is loopback-only (127.0.0.1); "
            "tailnet exposure goes through scripts/_tailnet_relay.py, never a wildcard bind."
        )
    spec = load_mask_spec(spec_path)
    config = AppConfig(
        clips_dir=Path(clips_dir),
        prep_dir=Path(prep_dir),
        spec=spec,
        spec_path=Path(spec_path),
        max_post_bytes=max_post_bytes,
    )
    Path(prep_dir).mkdir(parents=True, exist_ok=True)
    handler = make_handler(config)
    httpd = ThreadingHTTPServer((LOOPBACK, port), handler)
    print(f"[mask-app] serving {config.clips_dir} -> {config.prep_dir} on http://{LOOPBACK}:{port}", flush=True)
    print(f"[mask-app] spec {config.spec_path} · brush {config.spec['hex']} · loopback-only (tailnet via _tailnet_relay.py)", flush=True)
    httpd.serve_forever()
