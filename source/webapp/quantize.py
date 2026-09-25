#!/usr/bin/env python3
"""8-bit versions of the networks, for the browser.

    python source/webapp/quantize.py c4 hex uttt

In the browser the networks run on the CPU (onnxruntime-web, WebAssembly), and
there an 8-bit convolution does the same work as a 32-bit one in about 60% of
the time.  This writes ``models/<name>_int8.onnx`` next to each network and
points its entry in ``models/models.json`` at it (the float file stays, as
``float_file``, for comparing -- bench.html's "before" uses it).

Only the residual tower is quantized.  The stem and the heads stay float: they
are a small part of the work, and the value head in particular loses more
than it is worth -- quantized whole, Hex's evaluations moved by 0.1 on average;
tower only, by 0.02.

Calibration uses real positions: the script plays a few games between the
bots, natively (CPython + onnxruntime, with the same engine the site runs),
records every batch the networks are asked about, and calibrates on a sample
of them.  It then reports how often the 8-bit network's favourite move is the
float network's, and how far its evaluations are, on positions it was not
calibrated on.  Check the bots' strength before shipping a new one (a match
of the 8-bit bot against the float one at 400 simulations is what the
current files passed).

Needs ``pip install onnxruntime onnx numpy``, and the site built
(``build.py``), since it runs the engine from ``engine.zip``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / "webapp"
SITE = WEBAPP / "site" if (WEBAPP / "site").is_dir() else ROOT.parent
GAMES = {"hex": "11", "c4": "", "uttt": "", "rps2": ""}


def record_positions(names: list[str], games: int, sims: int) -> dict[str, np.ndarray]:
    """Input planes the bots ask each network about, over a few games."""
    import onnxruntime as ort

    models = json.loads((SITE / "models" / "models.json").read_text())
    work = Path(tempfile.mkdtemp(prefix="az-quantize-"))
    with zipfile.ZipFile(SITE / "engine.zip") as z:
        z.extractall(work)
    sys.path.insert(0, str(work))
    import webgui as W

    sessions: dict = {}
    seen: dict[str, list[np.ndarray]] = {}

    def infer(name: str, planes: np.ndarray) -> list[np.ndarray]:
        meta = models[name]
        if name not in sessions:
            sessions[name] = ort.InferenceSession(str(SITE / meta.get("float_file", meta["file"])),
                                                  providers=["CPUExecutionProvider"])
        planes = np.ascontiguousarray(planes, dtype=np.float32)
        seen.setdefault(name, []).append(planes)
        return [np.asarray(o, dtype=np.float32) for o in sessions[name].run(meta["outputs"], {"x": planes})]

    W._infer = infer
    placements = json.loads((SITE / "engine.json").read_text())
    placements.pop("uttt_v3", None)          # needs the native core; not quantized
    W.boot(str(work), json.dumps(models), json.dumps(placements))
    session = W.SESSION
    session.set_pace(0.0)
    for name in names:
        for seed in range(games):
            session.new_game(game=name, black="az", white="az", size=GAMES[name],
                             black_sims=sims, white_sims=sims,
                             black_vary=True, white_vary=True, seed=100 + seed)
            while session.bot_pending():
                session.bot_think()
                session.bot_commit()
        print(f"{name:5} {sum(len(p) for p in seen[name]):6d} positions from {games} games")
    rng = np.random.default_rng(0)
    return {n: np.concatenate(seen[n])[rng.permutation(sum(len(p) for p in seen[n]))]
            for n in names}


def quantize(name: str, positions: np.ndarray, calibrate: int) -> Path:
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    models = json.loads((SITE / "models" / "models.json").read_text())
    meta = models[name]
    source = SITE / meta.get("float_file", meta["file"])
    target = SITE / "models" / f"{name}_int8.onnx"
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(source), str(pre), skip_symbolic_shape=True)
        tower = [n.name for n in onnx.load(str(pre)).graph.node
                 if n.op_type == "Conv" and "/tower/" in n.name]

        class Reader(CalibrationDataReader):
            def __init__(self):
                x = positions[:calibrate]
                self.batches = iter([{"x": x[i:i + 16]} for i in range(0, len(x), 16)])

            def get_next(self):
                return next(self.batches, None)

        quantize_static(str(pre), str(target), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=True, calibrate_method=CalibrationMethod.MinMax,
                        nodes_to_quantize=tower)
    return target


def agreement(name: str, positions: np.ndarray) -> dict:
    """How closely the 8-bit network follows the float one on ``positions``."""
    import onnxruntime as ort

    meta = json.loads((SITE / "models" / "models.json").read_text())[name]
    float_net = ort.InferenceSession(str(SITE / meta.get("float_file", meta["file"])),
                                     providers=["CPUExecutionProvider"])
    int8_net = ort.InferenceSession(str(SITE / "models" / f"{name}_int8.onnx"),
                                    providers=["CPUExecutionProvider"])
    a = float_net.run(None, {"x": positions})
    b = int8_net.run(None, {"x": positions})
    n = len(positions)
    same = float(np.mean(a[0].reshape(n, -1).argmax(1) == b[0].reshape(n, -1).argmax(1)))
    va, vb = a[1].reshape(n, -1)[:, 0], b[1].reshape(n, -1)[:, 0]
    return dict(top_move_same=round(same, 4), value_diff_mean=round(float(np.abs(va - vb).mean()), 4),
                value_diff_max=round(float(np.abs(va - vb).max()), 4), positions=n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(GAMES))
    ap.add_argument("--games", type=int, default=2, help="games per network to record")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--calibrate", type=int, default=800, help="positions to calibrate on")
    args = ap.parse_args()
    positions = record_positions(args.names, args.games, args.sims)
    path = SITE / "models" / "models.json"
    for name in args.names:
        target = quantize(name, positions[name], args.calibrate)
        check = agreement(name, positions[name][args.calibrate:args.calibrate + 1000])
        models = json.loads(path.read_text())
        meta = models[name]
        meta.setdefault("float_file", meta["file"])
        meta["file"] = f"models/{target.name}"
        meta["int8"] = check
        path.write_text(json.dumps(models, indent=1), newline="\r\n")   # as exported
        print(f"{name:5} {target.stat().st_size // 1024} KB  {check}")


if __name__ == "__main__":
    main()
