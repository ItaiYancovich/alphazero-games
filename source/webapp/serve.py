#!/usr/bin/env python3
"""Serve ``webapp/site`` locally, the way a real host would, for testing.

    python webapp/serve.py              # http://localhost:8765
    python webapp/serve.py --port 9000

Sends the two cross-origin isolation headers itself, so the page works here
without its service worker (which is what makes it work on GitHub Pages).
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path

# The site: ``webapp/site`` in the main project; in the copy of the source
# kept inside the site's own repo (``source/``), the repo root itself.
SITE = Path(__file__).resolve().parent / "site"
if not SITE.is_dir():
    SITE = Path(__file__).resolve().parents[2]


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript",
                      ".wasm": "application/wasm", ".onnx": "application/octet-stream",
                      ".whl": "application/zip", ".svg": "image/svg+xml"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    handler = functools.partial(Handler, directory=str(SITE))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"serving {SITE} on http://localhost:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
