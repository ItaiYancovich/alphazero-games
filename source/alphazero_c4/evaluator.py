"""Batched network evaluation for Connect Four.

Everything but the encoder and the board shape is shared with Hex; the shared
evaluator also asks the position for its policy mask, which is what restricts
the priors to the seven landing squares instead of every empty cell.
"""

from __future__ import annotations

from alphazero_core.evaluator import BatchEvaluator as _CoreEvaluator

from .c4_game import COLS, ROWS
from .features import planes_from_boards
from .net import C4Net


class BatchEvaluator(_CoreEvaluator):
    """Runs the net on a list of Connect Four positions -> (priors, values)."""

    def __init__(self, net: C4Net, device: str = "cpu", jit: bool = True,
                 rows: int = ROWS, cols: int = COLS):
        super().__init__(net, planes_from_boards, device=device, jit=jit,
                         example_shape=(rows, cols))
