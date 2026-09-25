"""Batched evaluation for the v3 network: 17 planes in, cells-and-swap out.

Differs from :class:`alphazero_hex.evaluator.BatchEvaluator` in the two places
v3 differs from v2:

* **The input** carries the swap-available flag, which is not a property of the
  board, so it is read from each position's ``swap_available()``.
* **The output** has ``n*n + 1`` entries -- every cell, then the swap -- and is
  masked to what is legal *here*: empty cells, plus the swap only where the pie
  rule offers it.  ``HexNetV3.forward`` emits log-probabilities over all of
  them, so masking and renormalising gives exactly the right distribution in
  either case.

The priors are therefore always ``n*n + 1`` wide, which is what the search
needs whenever a swap-rule game reaches its second move.
"""

from __future__ import annotations

import numpy as np
import torch

from .features_v3 import PLANES_V3, planes_v3
from .net_v3 import HexNetV3


class V3Evaluator:
    def __init__(self, net: HexNetV3, board_size: int = 11, jit: bool = True):
        if net.cfg.in_planes != PLANES_V3:
            raise ValueError(f"v3 evaluator expects {PLANES_V3} input planes, "
                             f"the network has {net.cfg.in_planes}")
        self.net = net.eval()
        self.in_planes = PLANES_V3
        self.module = self.net
        if jit:
            try:
                example = torch.zeros(8, PLANES_V3, board_size, board_size)
                traced = torch.jit.trace(self.net, example)
                self.module = torch.jit.optimize_for_inference(traced)
            except Exception:
                self.module = self.net

    @torch.inference_mode()
    def evaluate(self, states) -> tuple[np.ndarray, np.ndarray]:
        boards = np.stack([s.canonical_board() for s in states])
        k, n, _ = boards.shape
        swap = np.array([bool(getattr(s, "swap_available", lambda: False)()) for s in states])
        x = torch.from_numpy(planes_v3(boards, swap))
        logp, values = self.module(x)
        if logp.shape[1] != n * n + 1:
            raise ValueError("v3 evaluator needs a network with a swap head")
        mask = np.zeros((k, n * n + 1), dtype=bool)
        mask[:, :n * n] = boards.reshape(k, -1) == 0
        mask[:, n * n] = swap
        logp = logp.masked_fill(torch.from_numpy(~mask), -1e9)
        priors = torch.softmax(logp, dim=1).numpy().astype(np.float32)
        return priors, values.numpy().astype(np.float32).reshape(-1)
