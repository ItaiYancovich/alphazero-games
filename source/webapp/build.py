#!/usr/bin/env python3
"""Build the self-contained web version of the game GUI into ``webapp/site``.

    python webapp/build.py                  # engine + page (networks must be exported)
    python webapp/build.py --models         # ...and re-export the ONNX networks first

What ends up in ``site/``:

* ``engine.zip`` -- the project's Python (``game_gui.py`` and the game packages,
  source only), the web layer from ``webapp/py`` and the bot rating tables.
  The browser unpacks it into Pyodide's file system and imports it.
* ``models/`` -- the ONNX networks and ``models.json`` (``export_onnx.py``).
* ``engine.json`` -- which checkpoint path each network stands in for.
* the page, from ``web/index.html`` with the web bridge wired in, and the
  static files from ``webapp/static``.

Nothing here touches the desktop GUI: ``web/index.html`` is read, never written.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / "webapp"
# The site: ``webapp/site`` in the main project; in the copy of the source
# kept inside the site's own repo (``source/``), the repo root itself.
SITE = WEBAPP / "site" if (WEBAPP / "site").is_dir() else ROOT.parent

PACKAGES = ["alphazero_core", "alphazero_hex", "alphazero_c4", "alphazero_uttt",
            "alphazero_uxx", "alphazero_rps2", "alphazero_bg", "alphazero_splendor",
            "alphazero_splendor2", "alphazero_splendor3"]
# Training, tournaments and GPU plumbing: never imported by the GUI.
SKIP = re.compile(r"(^|/)(train[^/]*|tournament|gpu_server|ov_evaluator|bootstrap|"
                  r"selfplay[^/]*|utttai_[^/]*|v3data|v3net|v3selfplay)\.py$")

# The checkpoint each network stands in for, relative to the project root --
# exactly where the adapters look, so their own discovery finds it.
PLACEMENTS = {
    "hex": "runs/az_hex/{ckpt}",
    "c4": "runs/az_c4/{ckpt}",
    "uttt": "runs/az_uttt_v2/{ckpt}",
    "rps2": "runs/az_rps2/{ckpt}",
    "bg": "runs/az_bg/{ckpt}",
    "splendor3": "runs/az_splendor3/{ckpt}",
    "uttt_v3": "runs/uttt_v3/{ckpt}",
}
# Data files the engine reads besides the rating tables.
EXTRA_DATA = ["runs/uttt_v3/book.json"]


def build_engine() -> None:
    out = SITE / "engine.zip"
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(ROOT / "game_gui.py", "game_gui.py")
        for pkg in PACKAGES:
            for path in sorted((ROOT / pkg).rglob("*.py")):
                rel = path.relative_to(ROOT).as_posix()
                if "__pycache__" in rel or SKIP.search(rel):
                    continue
                z.write(path, rel)
                count += 1
        for path in sorted((WEBAPP / "py").rglob("*.py")):
            z.write(path, path.relative_to(WEBAPP / "py").as_posix())
            count += 1
        for path in sorted((ROOT / "runs").glob("ratings*.json")):
            z.write(path, f"runs/{path.name}")
            count += 1
        for rel in EXTRA_DATA:
            if (ROOT / rel).is_file():
                z.write(ROOT / rel, rel)
                count += 1
    print(f"engine.zip  {out.stat().st_size / 1e6:.2f} MB, {count} files")


def build_placements() -> None:
    models = json.loads((SITE / "models" / "models.json").read_text())
    placements = {name: PLACEMENTS[name].format(ckpt=meta["ckpt"])
                  for name, meta in models.items()}
    (SITE / "engine.json").write_text(json.dumps(placements, indent=1))


def build_page() -> None:
    """The desktop page, with its one server call routed to the web bridge."""
    src = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    bridge = '<script src="pool.js"></script>\n<script src="bridge.js"></script>\n<script>'
    if src.count("<script>") != 1:
        raise SystemExit("web/index.html: expected exactly one inline <script>")
    src = src.replace("<script>", bridge, 1)
    # ``api()`` is the single door to the server; the bridge supplies a
    # replacement with the same contract, and the page keeps its own for the
    # desktop build.
    head = "async function api(path, body) {"
    if src.count(head) != 1:
        raise SystemExit("web/index.html: could not find api()")
    src = src.replace(head, "async function api(path, body) {\n"
                            "  if (window.WEB_API) return window.WEB_API(path, body);", 1)
    extra = (WEBAPP / "static" / "head.html").read_text(encoding="utf-8")
    src = src.replace("</head>", extra + "</head>", 1)
    body_extra = (WEBAPP / "static" / "body.html").read_text(encoding="utf-8")
    src = src.replace("</body>", body_extra + "</body>", 1)
    (SITE / "index.html").write_text(src, encoding="utf-8")
    for path in (WEBAPP / "static").iterdir():
        if path.suffix in (".js", ".css", ".svg", ".png", ".webmanifest", ".json") \
                or path.name == "bench.html":
            shutil.copy2(path, SITE / path.name)
    print("index.html + static files")


DEV = WEBAPP / "dev"
VENDOR = {
    # Pyodide: the core, the standard library, and numpy -- nothing else is used.
    "pyodide": [DEV / "node_modules/pyodide" / name for name in
                ("pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm",
                 "python_stdlib.zip", "pyodide-lock.json")]
               + [DEV / "vendor/numpy-2.0.2-cp312-cp312-pyodide_2024_0_wasm32.whl"],
    # onnxruntime-web, CPU (WASM) backend only.
    "ort": [DEV / "node_modules/onnxruntime-web/dist" / name for name in
            ("ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs",
             "ort-wasm-simd-threaded.wasm")],
    # Signing for the relay messages of online play (Unlicense).
    "": [DEV / "node_modules/nostr-tools/lib/nostr.bundle.js"],
}


def build_vendor() -> None:
    """Third-party runtimes, served from the site itself rather than a CDN.

    Same-origin files need no CORS or CORP headers, which is what keeps the
    page cross-origin isolated on a host that cannot set headers.
    """
    total = 0
    for folder, files in VENDOR.items():
        dest = SITE / folder
        dest.mkdir(parents=True, exist_ok=True)
        for src in files:
            src = Path(src)
            if not src.exists() and (dest / src.name).exists():
                total += (dest / src.name).stat().st_size     # already vendored
                continue
            if not src.exists():
                raise SystemExit(f"missing {src} -- run `npm install` in webapp/dev")
            shutil.copy2(src, dest / src.name)
            total += src.stat().st_size
    print(f"vendor      {total / 1e6:.1f} MB")


WASM_CRATE = WEBAPP / "uttt_wasm"


def build_wasm() -> None:
    """``uttt_rs``, the native UTTT core, compiled for the browser (WASI)."""
    wasm = WASM_CRATE / "target" / "wasm32-wasip1" / "release" / "uttt_wasm.wasm"
    cargo = shutil.which("cargo")
    if cargo:
        subprocess.run([cargo, "build", "--release", "--target", "wasm32-wasip1"],
                       cwd=WASM_CRATE, check=True, capture_output=True)
    if not wasm.exists():
        print("uttt_wasm   not built (needs cargo + `rustup target add wasm32-wasip1`)")
        return
    shutil.copy2(wasm, SITE / "uttt_wasm.wasm")
    print(f"uttt_wasm   {wasm.stat().st_size / 1e3:.0f} KB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="re-export the ONNX networks")
    args = ap.parse_args()
    SITE.mkdir(parents=True, exist_ok=True)
    if args.models or not (SITE / "models" / "models.json").exists():
        subprocess.run([sys.executable, str(WEBAPP / "export_onnx.py")], check=True)
    build_engine()
    build_placements()
    build_vendor()
    build_wasm()
    if (WEBAPP / "static" / "head.html").exists():
        build_page()
    # GitHub Pages: serve the files as they are, no Jekyll pass.
    (SITE / ".nojekyll").write_text("")


if __name__ == "__main__":
    main()
