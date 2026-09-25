"""One batched forward pass, shared by everything that needs the network.

Returns **log-probabilities over the legal moves**, not probabilities.  The
Gumbel search in :mod:`alphazero_splendor3.search` adds Gumbel noise to logits
and compares sums, so it wants a log scale; a masked log-softmax differs from
the network's raw logits only by a constant, which cancels in both the softmax
and the argmax it is used in.
"""

from __future__ import annotations

import numpy as np
import torch

from .features import encode
from .game import NACTIONS

NEG = -1e9


class BatchEvaluator:
    """The network, called on a list of positions at once."""

    def __init__(self, net, batch_size: int = 256):
        self.net = net
        self.net.eval()
        self.batch_size = int(batch_size)

    @torch.no_grad()
    def evaluate(self, games) -> tuple[np.ndarray, np.ndarray]:
        """``(log-priors (n, 72), values (n, seats))``, seat-relative."""
        n = len(games)
        if n == 0:
            return (np.zeros((0, NACTIONS), dtype=np.float32),
                    np.zeros((0, self.net.cfg.seats), dtype=np.float32))
        x = torch.from_numpy(encode(games))
        logits, values = self.net(x)
        mask = np.zeros((n, NACTIONS), dtype=bool)
        for i, g in enumerate(games):
            mask[i] = g.legal_mask()
        logits = logits.numpy()
        logits = np.where(mask, logits, NEG)
        logits -= logits.max(axis=1, keepdims=True)
        np.exp(logits, out=logits)
        total = logits.sum(axis=1, keepdims=True)
        np.divide(logits, np.maximum(total, 1e-30), out=logits)
        np.log(np.maximum(logits, 1e-30), out=logits)
        return logits, values.numpy()
