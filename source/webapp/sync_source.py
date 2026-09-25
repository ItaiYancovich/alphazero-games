#!/usr/bin/env python3
"""Keep a copy of the web build's source inside the site's repo, and bring
changes made there back.

The site's repo (``webapp/site``, published to GitHub Pages) holds only what
the browser loads.  A session that works on that repo alone -- a cloud
session, say -- also needs what the site is built *from*: the engine's Python,
the Rust core, the page, the build scripts.  So a copy of exactly that lives
in the repo under ``source/``, laid out like this project, and
``source/webapp/build.py`` there builds straight into the repo root.

    python webapp/sync_source.py push           # this project -> site/source/
    python webapp/sync_source.py pull           # list what differs the other way
    python webapp/sync_source.py pull --apply   # site/source/ -> this project

``push`` makes ``source/`` an exact copy of the list below (and removes
anything else there).  ``pull`` never deletes: it copies files that are new or
changed in ``source/`` back over this project's.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / "webapp"
DEST = WEBAPP / "site" / "source"

sys.path.insert(0, str(WEBAPP))
from build import PACKAGES, SKIP, EXTRA_DATA  # noqa: E402

JUNK = ("__pycache__", "target", "node_modules")


def _tree(folder: Path, patterns=("*",)) -> list[Path]:
    out = []
    for pattern in patterns:
        for path in folder.rglob(pattern):
            if path.is_file() and not any(part in JUNK for part in path.parts):
                out.append(path)
    return out


def files() -> list[Path]:
    """Everything the site is built from, as paths in this project."""
    out = [ROOT / "game_gui.py", ROOT / "web" / "index.html"]
    for pkg in PACKAGES:
        out += [p for p in _tree(ROOT / pkg, ("*.py",))
                if not SKIP.search(p.relative_to(ROOT).as_posix())]
    out += sorted((ROOT / "runs").glob("ratings*.json"))
    out += [ROOT / rel for rel in EXTRA_DATA if (ROOT / rel).is_file()]
    for crate in (ROOT / "uttt_rs", WEBAPP / "uttt_wasm"):
        out += [crate / "Cargo.toml", crate / "Cargo.lock"] + _tree(crate / "src")
    out += [WEBAPP / name for name in ("build.py", "export_onnx.py", "serve.py",
                                       "sync_source.py")]
    out += _tree(WEBAPP / "py") + _tree(WEBAPP / "static")
    dev = WEBAPP / "dev"
    out += sorted(dev.glob("*.mjs")) + [dev / "package.json", dev / "package-lock.json",
                                        dev / "parity_positions.json"]
    return sorted(set(p for p in out if p.is_file()))


def push() -> None:
    wanted = {p.relative_to(ROOT) for p in files()}
    for rel in sorted(wanted):
        dest = DEST / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or not filecmp.cmp(ROOT / rel, dest, shallow=False):
            shutil.copy2(ROOT / rel, dest)
    keep = wanted | {Path("README.md")}
    for p in _tree(DEST):
        if p.relative_to(DEST) not in keep:
            p.unlink()
    for d in sorted((p for p in DEST.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    print(f"source/: {len(wanted)} files")


def pull(apply: bool) -> None:
    changed = []
    for path in _tree(DEST):
        rel = path.relative_to(DEST)
        if rel == Path("README.md"):
            continue
        mine = ROOT / rel
        if not mine.exists() or not filecmp.cmp(path, mine, shallow=False):
            changed.append((rel, "new" if not mine.exists() else "changed"))
    for rel, what in changed:
        print(f"  {what:8} {rel.as_posix()}")
        if apply:
            (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(DEST / rel, ROOT / rel)
    if not changed:
        print("nothing differs")
    elif not apply:
        print("(run with --apply to copy these into the project)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("direction", choices=["push", "pull"])
    ap.add_argument("--apply", action="store_true", help="pull: actually copy")
    args = ap.parse_args()
    if not (WEBAPP / "site").is_dir():
        raise SystemExit("run this in the main project, not in the site's source/ copy")
    push() if args.direction == "push" else pull(args.apply)


if __name__ == "__main__":
    main()
