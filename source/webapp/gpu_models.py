#!/usr/bin/env python3
"""GPU versions of the v3 network, for the browser's WebGPU worker.

    python source/webapp/gpu_models.py

Two files next to ``models/uttt_v3.onnx``, both listed in ``models.json``:

* ``uttt_v3_gpu.onnx`` (``gpu_file``): the same network with its Softplus
  rewritten as ``Relu(x) + Log(1 + Exp(-|x|))``.  onnxruntime-web has no WebGPU
  Softplus, so the original runs those four steps on the CPU -- a round trip
  from the GPU and back for each, every call -- and, with any step off the
  GPU, cannot use graph capture (recording a run once and replaying it, which
  saves the GPU the per-step overhead a network of many small steps pays).
  The rewrite is the same function, stable for large |x|; outputs agree with
  the original to about 4e-6.
* ``uttt_v3_gpu16.onnx`` (``gpu16_file``): that, in half precision, inputs and
  outputs still float32.  The desktop runs this network in half precision on
  the GPU too.  Used only where the GPU has ``shader-f16``.

The CPU keeps the original file: its moves stay exactly the desktop's.

Needs ``pip install onnx onnxruntime sympy`` (onnxruntime's float16 converter).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / "webapp"
SITE = WEBAPP / "site" if (WEBAPP / "site").is_dir() else ROOT.parent
MODELS = SITE / "models"


def without_softplus(model: onnx.ModelProto) -> onnx.ModelProto:
    """Softplus(x) as Relu(x) + Log(1 + Exp(-|x|)), from ops WebGPU has."""
    graph = model.graph
    graph.initializer.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32),
                                                     "softplus_one"))
    nodes = []
    for node in graph.node:
        if node.op_type != "Softplus":
            nodes.append(node)
            continue
        x, y, p = node.input[0], node.output[0], node.name or node.output[0]
        nodes += [
            helper.make_node("Abs", [x], [p + "_abs"], name=p + "_abs"),
            helper.make_node("Neg", [p + "_abs"], [p + "_neg"], name=p + "_neg"),
            helper.make_node("Exp", [p + "_neg"], [p + "_exp"], name=p + "_exp"),
            helper.make_node("Add", [p + "_exp", "softplus_one"], [p + "_1p"], name=p + "_1p"),
            helper.make_node("Log", [p + "_1p"], [p + "_log"], name=p + "_log"),
            helper.make_node("Relu", [x], [p + "_relu"], name=p + "_relu"),
            helper.make_node("Add", [p + "_relu", p + "_log"], [y], name=p + "_out"),
        ]
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def half(model: onnx.ModelProto) -> onnx.ModelProto:
    """Half precision inside, float32 inputs and outputs."""
    from onnxruntime.transformers.onnx_model import OnnxModel

    wrapped = OnnxModel(model)
    wrapped.convert_float_to_float16(keep_io_types=True)
    wrapped.topological_sort()
    return wrapped.model


def compare(reference: Path, other: Path, x: np.ndarray) -> dict:
    import onnxruntime as ort

    a = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"]).run(None, {"x": x})
    b = ort.InferenceSession(str(other), providers=["CPUExecutionProvider"]).run(None, {"x": x})
    n = len(x)
    return dict(top_move_same=round(float(np.mean(a[0].reshape(n, -1).argmax(1)
                                                   == b[0].reshape(n, -1).argmax(1))), 4),
                value_diff_max=round(float(np.abs(a[1] - b[1]).max()), 6))


def main() -> None:
    source = MODELS / "uttt_v3.onnx"
    gpu = MODELS / "uttt_v3_gpu.onnx"
    gpu16 = MODELS / "uttt_v3_gpu16.onnx"
    onnx.save(without_softplus(onnx.load(str(source))), str(gpu))
    onnx.save(half(onnx.load(str(gpu))), str(gpu16))
    # Positions with the planes' own statistics: the empty-and-playable cells
    # of a real game are what the values depend on, but for a check that the
    # two files compute the same function, random 0/1 planes do.
    x = (np.random.default_rng(0).random((256, 23, 9, 9)) < 0.3).astype(np.float32)
    path = MODELS / "models.json"
    models = json.loads(path.read_text())
    meta = models["uttt_v3"]
    meta["gpu_file"] = f"models/{gpu.name}"
    meta["gpu16_file"] = f"models/{gpu16.name}"
    meta["gpu_check"] = {"float": compare(source, gpu, x), "half": compare(source, gpu16, x)}
    path.write_text(json.dumps(models, indent=1), newline="\r\n")   # as exported
    print(json.dumps(meta["gpu_check"]))


if __name__ == "__main__":
    main()
