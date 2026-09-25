"""Batched network evaluation for Hex.

The batching, the JIT trace and the legality mask are game-independent and live
in :mod:`alphazero_core.evaluator`.  What is Hex-specific is the feature
encoder and the fact that a Hex board is square, so ``board_size`` is enough to
describe the trace shape -- which is the signature every caller here already
uses.
"""

from __future__ import annotations

import numpy as np

from alphazero_core.evaluator import BatchEvaluator as _CoreEvaluator

from .features import planes_from_boards
from .net import HexNet


class BatchEvaluator(_CoreEvaluator):
    """Runs the net on a list of Hex positions and returns (priors, values)."""

    def __init__(self, net: HexNet, device: str = "cpu", jit: bool = True,
                 board_size: int = 11):
        super().__init__(net, planes_from_boards, device=device, jit=jit,
                         example_shape=(board_size, board_size))


class SwapAwareEvaluator:
    """Lets a network with no swap output play pie-rule games.

    The pie-rule swap is move index ``n*n``, one past the last cell, so the
    search needs priors of width ``n*n + 1``.  A network trained without the
    rule has no opinion about swapping, so it is handed a fixed prior of
    ``swap_prior`` wherever the swap is legal and nothing anywhere else.

    That is enough to play the rule properly, not just legally: the swapped
    position is an ordinary position, which the network's value head *can*
    judge, so the search visits the swap and keeps it exactly when it is worth
    more than the best placement.  v3's learned swap head replaces the fixed
    prior; the value judgement was never the fixed part.
    """

    def __init__(self, inner, swap_prior: float = 0.1):
        self.inner = inner
        self.swap_prior = float(swap_prior)

    def evaluate(self, states) -> tuple[np.ndarray, np.ndarray]:
        priors, values = self.inner.evaluate(states)
        b, cells = priors.shape
        out = np.zeros((b, cells + 1), dtype=np.float32)
        out[:, :cells] = priors
        for i, state in enumerate(states):
            available = getattr(state, "swap_available", None)
            if available is not None and available():
                out[i, :cells] *= 1.0 - self.swap_prior
                out[i, cells] = self.swap_prior
        return out, values

    def __getattr__(self, name):
        # Anything else (``net``, ``in_planes``...) is the wrapped evaluator's.
        return getattr(self.inner, name)
