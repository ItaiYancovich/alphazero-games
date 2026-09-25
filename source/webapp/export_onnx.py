#!/usr/bin/env python3
"""Export the networks the web build plays with to ONNX, and check them.

    python webapp/export_onnx.py            # -> webapp/site/models/*.onnx + models.json

One network per game: the one the desktop GUI judges positions with and rates
its AlphaZero opponent by (``analysis_checkpoint``), and for Splendor the v3
network, the strongest there is.  Each is traced on the CPU, exported with a
dynamic batch axis, and then run once through onnxruntime next to PyTorch on
random input -- a model that exports but disagrees is worse than none.

``models.json`` records, per model, the checkpoint's own config (so the browser
side can answer ``net.cfg.<field>`` exactly as the real network would) and the
output names, which the web evaluator hands back in the order ``forward`` did.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The site: ``webapp/site`` in the main project; in the copy of the source
# kept inside the site's own repo (``source/``), the repo root itself.
OUT = (ROOT / "webapp" / "site" if (ROOT / "webapp" / "site").is_dir()
       else ROOT.parent) / "models"


class _Forward(torch.nn.Module):
    """``forward`` as the evaluators call it, with every output a tensor."""

    def __init__(self, net, extra: str | None = None):
        super().__init__()
        self.net = net
        self.extra = extra

    def forward(self, x):
        out = self.net(x)
        out = list(out) if isinstance(out, (tuple, list)) else [out]
        if self.extra:
            out.append(getattr(self.net, self.extra)(x))
        return tuple(out)


def _cfg_dict(net) -> dict:
    cfg = getattr(net, "cfg", None)
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        raw = cfg.to_dict()
    else:
        raw = dict(vars(cfg))
    return {k: v for k, v in raw.items()
            if isinstance(v, (int, float, str, bool, type(None), list))}


def export(name: str, net, example: np.ndarray, extra: str | None = None,
           outputs: list[str] | None = None) -> dict:
    net = net.eval()
    wrapped = _Forward(net, extra).eval()
    x = torch.from_numpy(example)
    with torch.no_grad():
        ref = wrapped(x)
    names = outputs or [f"out{i}" for i in range(len(ref))]
    dyn = {"x": {0: "batch"}, **{n: {0: "batch"} for n in names}}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    torch.onnx.export(wrapped, (x,), str(path), input_names=["x"], output_names=names,
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # A different batch size from the trace, so the dynamic axis is exercised.
    probe = np.repeat(example, 3, axis=0)
    got = sess.run(None, {"x": probe})
    with torch.no_grad():
        want = wrapped(torch.from_numpy(probe))
    worst = max(float(np.max(np.abs(g - w.numpy()))) for g, w in zip(got, want))
    print(f"{name:10s} {path.stat().st_size / 1e6:5.2f} MB  in {list(example.shape[1:])}"
          f"  outs {[list(o.shape[1:]) for o in ref]}  max|diff| {worst:.2e}")
    if worst > 1e-3:
        raise SystemExit(f"{name}: onnxruntime disagrees with torch by {worst}")
    return dict(file=f"models/{name}.onnx", input_shape=list(example.shape[1:]),
                outputs=names, cfg=_cfg_dict(net))


def main() -> None:
    import game_gui as g

    rng = np.random.default_rng(0)
    models: dict[str, dict] = {}

    def planes_for(adapter, board, encode, net):
        boards = np.stack([board.canonical_board()])
        return encode(boards, net.cfg.in_planes).astype(np.float32)

    # ---- the four board games on the shared PVNet -------------------------
    from alphazero_core.net import load_checkpoint as core_load

    for key, feat_mod in (("hex", "alphazero_hex.features"),
                          ("c4", "alphazero_c4.features"),
                          ("uttt", "alphazero_uttt.features"),
                          ("rps2", "alphazero_rps2.features")):
        adapter = g.GAMES[key]
        ckpt = adapter.analysis_checkpoint()
        path = adapter.ckpt_path(ckpt)
        if key == "hex":
            from alphazero_hex.net_v3 import load_any
            net, _ = load_any(path)
        else:
            net, _ = core_load(path)
        encode = __import__(feat_mod, fromlist=["planes_from_boards"]).planes_from_boards
        board = adapter.new_board(adapter.default_size, 1)
        # A few random legal moves in, so the planes are not all zero.
        for _ in range(6):
            moves = board.legal_moves()
            if board.is_terminal() or not len(moves):
                break
            board.play(int(rng.choice(moves)))
        if key == "rps2":
            x = encode(np.stack([board.canonical_board()]), net.cfg.in_planes,
                       reps=np.zeros(1, dtype=np.int64)).astype(np.float32)
        else:
            x = planes_for(adapter, board, encode, net)
        models[key] = dict(export(key, net, x, outputs=["policy", "value"]),
                           ckpt=ckpt)

    # ---- backgammon: afterstate values, two heads the evaluator reads -----
    from alphazero_bg.features import encode_batch
    from alphazero_bg.net import load_checkpoint as bg_load

    adapter = g.GAMES["bg"]
    ckpt = adapter.analysis_checkpoint()
    net, _ = bg_load(str(adapter.ckpt_dir / ckpt))
    board = adapter.new_board(adapter.default_size, 1)
    x = encode_batch([board], getattr(net.cfg, "features", "basic")).astype(np.float32)
    models["bg"] = dict(export("bg", net, x, extra="equities",
                               outputs=["forward", "equities"]), ckpt=ckpt)

    # ---- Ultimate Tic-Tac-Toe v3: the match bot's network ------------------
    # Exported as V3Export -- policy logits, value and a Q per move, which is
    # exactly what the native search (uttt_rs, compiled to WASM) consumes.
    from alphazero_uttt.features import planes_from_boards as uttt_planes
    from alphazero_uttt.v3net import V3Export, load_v3

    v3_path = g.ROOT / "runs" / "uttt_v3" / "best.pt"
    if v3_path.is_file():
        net, _ = load_v3(str(v3_path))
        adapter = g.GAMES["uttt"]
        board = adapter.new_board(adapter.default_size, 1)
        for _ in range(6):
            board.play(int(rng.choice(board.legal_moves())))
        boards = np.stack([board.canonical_board()])
        x = uttt_planes(boards, net.cfg.in_planes).astype(np.float32)
        wrapped = V3Export(net).eval()
        wrapped.cfg = net.cfg
        models["uttt_v3"] = dict(export("uttt_v3", wrapped, x, outputs=["policy", "value", "q"]),
                                 ckpt="best.pt")

    # ---- Splendor v3: token network, Gumbel search ------------------------
    from alphazero_splendor3.bridge import from_v1
    from alphazero_splendor3.features import encode as sp_encode
    from alphazero_splendor3.net import load_checkpoint as sp_load

    adapter = g.GAMES["splendor"]
    path = adapter.v3_checkpoint()
    net, _ = sp_load(path)
    board = adapter.new_board("2", 1)
    x = sp_encode([from_v1(board)]).astype(np.float32)
    models["splendor3"] = dict(export("splendor3", net, x, outputs=["policy", "value"]),
                               ckpt=Path(path).name)

    (OUT / "models.json").write_text(json.dumps(models, indent=1))
    print("wrote", OUT / "models.json")


if __name__ == "__main__":
    main()
