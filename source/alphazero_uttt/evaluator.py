"""Batched network evaluation for Ultimate Tic-Tac-Toe.

Everything but the encoder is shared with Hex and Connect Four.  The shared
evaluator also asks the position for its policy mask, which is what restricts
the priors to the small board the last move sent us to instead of every empty
cell on the grid -- though here the canonical board already says the same thing
(a reachable empty reads 0, an unreachable one reads 3), so the two agree by
construction.
"""

from __future__ import annotations

from alphazero_core.evaluator import BatchEvaluator as _CoreEvaluator

from .features import planes_from_boards
from .net import UtttNet
from .uttt_game import N


class BatchEvaluator(_CoreEvaluator):
    """Runs the net on a list of Ultimate positions -> (priors, values)."""

    def __init__(self, net: UtttNet, device: str = "cpu", jit: bool = True,
                 rows: int = N, cols: int = N):
        super().__init__(net, planes_from_boards, device=device, jit=jit,
                         example_shape=(rows, cols))
