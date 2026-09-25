"""Batched network evaluation for Intransitive.

Everything but two things is shared with the other games.  Those two are both
consequences of a move being a piece and a direction rather than a square:

* **The legality mask is not the board.**  The shared evaluator masks the
  policy with "the cells that read empty", intersected with whatever the
  position itself reports.  Here the policy has 648 entries and the board 81,
  so the board-shaped half of that is not merely redundant, it does not fit --
  the position's own ``canonical_policy_mask`` is the whole answer.
* **The JIT trace is over an 8-channel head.**  Nothing to configure: the
  checkpoint carries ``policy_planes``, so a traced module has the right shape
  by construction.
"""

from __future__ import annotations

import numpy as np
import torch

from alphazero_core.evaluator import BatchEvaluator as _CoreEvaluator

from .features import planes_from_boards
from .net import RPS2Net
from .rps2_game import N


class BatchEvaluator(_CoreEvaluator):
    """Runs the net on a list of Intransitive positions -> (priors, values)."""

    def __init__(self, net: RPS2Net, device: str = "cpu", jit: bool = True,
                 rows: int = N, cols: int = N):
        super().__init__(net, planes_from_boards, device=device, jit=jit,
                         example_shape=(rows, cols))

    def _inputs(self, states):
        boards = np.stack([s.canonical_board() for s in states])
        # The search reaches every node by copying the position and playing on
        # the copy, and a copy shares the move history -- so a node deep in the
        # tree knows how often its position has already occurred, and the
        # network gets to see it.  Without this the repetition planes would
        # read zero everywhere in search, which is the blindness they fix.
        reps = np.fromiter((s.repetitions() - 1 for s in states),
                           dtype=np.int64, count=len(states))
        return self.encode(boards, self.in_planes, reps=reps), states

    def _outputs(self, states, logits, values) -> tuple[np.ndarray, np.ndarray]:
        mask = np.stack([s.canonical_policy_mask() for s in states])
        logits = logits.masked_fill(torch.from_numpy(~mask).to(self.device), -1e9)
        priors = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        return priors, values.cpu().numpy().astype(np.float32)
