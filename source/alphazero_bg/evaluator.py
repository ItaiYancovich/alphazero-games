"""Evaluating backgammon positions and choosing moves.

The search here is not MCTS.  A backgammon turn begins with a roll nobody
chooses, so the tree alternates between decision nodes and chance nodes with a
branching factor of 21 -- and PUCT, which assumes the player controls which
child it descends into, has nothing to say about the chance layer.  What works
instead, and what every strong program is built on, is **afterstate
evaluation**:

* **One ply** -- generate the positions each legal move leads to, ask the network
  what each is worth to the *opponent*, and take the move with the worst answer.
  The roll is already known when the choice is made, so no averaging is needed.
* **Two ply** -- for each candidate, average the opponent's best one-ply reply
  over all 21 rolls, weighted 1/36 or 2/36.  That is one full chance layer, and
  it costs about 21x more network calls, which is why it is worth batching.

Both return *equity for the side to move*: 1/3 a win, 2/3 a gammon, 1 a
backgammon, negative for the other side.
"""

from __future__ import annotations

import numpy as np
import torch

from .bg_game import ROLLS, Backgammon
from .features import encode_batch
from .net import BGNet
from .outcomes import OUTPUTS
from .outcomes import equity as equity_of
from .outcomes import terminal_vector


class Evaluator:
    """Batched network evaluation, plus the move choice built on top of it."""

    def __init__(self, net: BGNet, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.device = device
        # The encoding and the head are properties of the weights, not of the
        # caller: a checkpoint knows what it was trained as.
        self.features = getattr(net.cfg, "features", "basic")
        self.outputs = getattr(net.cfg, "outputs", 1)

    @torch.inference_mode()
    def values(self, boards: list[Backgammon]) -> np.ndarray:
        """Equity for the side to move at each position, as a batch.

        Finished positions are not shown to the network at all: their value is
        a fact, and a network guess would only add noise to it.
        """
        out = np.empty(len(boards), dtype=np.float32)
        pending, index = [], []
        for i, board in enumerate(boards):
            if board.is_terminal():
                out[i] = board.terminal_value()
            else:
                pending.append(board)
                index.append(i)
        if pending:
            x = torch.from_numpy(encode_batch(pending, self.features)).to(self.device)
            values = self.net.equities(x).cpu().numpy().astype(np.float32)
            for slot, value in zip(index, values):
                out[slot] = value
        return out

    @torch.inference_mode()
    def probabilities(self, boards: list[Backgammon]) -> np.ndarray:
        """The five outcome probabilities per position, ``(N, 5) float32``.

        Only a five-output network has these to give.  A finished game is not
        shown to the network at all -- its outcome is a fact, and the exact
        indicator vector is a better label than any estimate of it.
        """
        if self.outputs != OUTPUTS:
            raise ValueError("this network predicts a single equity, not outcomes")
        out = np.zeros((len(boards), OUTPUTS), dtype=np.float32)
        pending, index = [], []
        for i, board in enumerate(boards):
            if board.is_terminal():
                out[i] = terminal_vector(board, board.to_move)
            else:
                pending.append(board)
                index.append(i)
        if pending:
            x = torch.from_numpy(encode_batch(pending, self.features)).to(self.device)
            probs = self.net(x).cpu().numpy().astype(np.float32)
            for slot, row in zip(index, probs):
                out[slot] = row
        return out

    # ------------------------------------------------------------- one ply
    def afterstates(self, board: Backgammon) -> tuple[list, list[Backgammon]]:
        """Every legal move and the position it leads to, turn already passed."""
        moves = board.legal_moves()
        return moves, [board.after_turn(move) for move in moves]

    def move_equities(self, board: Backgammon) -> tuple[list, np.ndarray]:
        """Each legal move, and what it is worth *to the mover*.

        The afterstate belongs to the opponent -- the turn has passed -- so the
        value that comes back is theirs, and the mover's equity is its negation.
        A move that ends the game is scored from the finished position instead.
        """
        moves, states = self.afterstates(board)
        if not moves:
            return [], np.empty(0, dtype=np.float32)
        mover = board.to_move
        equities = np.empty(len(states), dtype=np.float32)
        ask, index = [], []
        for i, state in enumerate(states):
            if state.is_terminal():
                equities[i] = state.result_for(mover)
            else:
                ask.append(state)
                index.append(i)
        if ask:
            values = self.values(ask)
            for slot, value in zip(index, values):
                equities[slot] = -value
        return moves, equities

    # ------------------------------------------------------------- two ply
    def move_equities_two_ply(self, board: Backgammon,
                              candidates: int = 6) -> tuple[list, np.ndarray]:
        """One-ply equities, refined by searching the opponent's reply.

        Only the best ``candidates`` moves by one-ply equity are examined: the
        cost is 21 rolls x a full move generation each, and a move the network
        already dislikes almost never survives a deeper look.
        """
        moves, shallow = self.move_equities(board)
        if len(moves) <= 1:
            return moves, shallow
        order = np.argsort(-shallow)[:max(2, candidates)]
        mover = board.to_move
        # Only the deepened moves stay in the running.  A one-ply score and a
        # two-ply score are not on the same scale -- the two-ply one averages
        # over the opponent's reply and is systematically less extreme -- so
        # leaving the rest at their one-ply values would let an unexamined move
        # win on an inflated number.  This cost two-ply the match against
        # one-ply in the first ladder run.
        deep = np.full_like(shallow, -np.inf)
        for slot in order:
            state = board.after_turn(moves[int(slot)])
            if state.is_terminal():
                deep[slot] = state.result_for(mover)
                continue
            deep[slot] = -self._expected_best_reply(state)
        return moves, deep

    def _expected_best_reply(self, state: Backgammon) -> float:
        """Equity for the side to move at ``state``, averaged over its roll.

        Every reply to every roll is gathered first and evaluated in one batch:
        21 rolls of a dozen replies each is 250-odd positions, and asking for
        them one at a time is where a two-ply search would otherwise go.
        """
        buckets: list[tuple[float, list[Backgammon], list[float]]] = []
        gather: list[Backgammon] = []
        for die_a, die_b, weight in ROLLS:
            probe = state.copy()
            probe.set_dice(die_a, die_b)
            replies = probe.legal_moves()
            states, known = [], []
            for reply in replies:
                child = probe.after_turn(reply)
                if child.is_terminal():
                    known.append(child.result_for(probe.to_move))
                else:
                    states.append(child)
            gather.extend(states)
            buckets.append((weight, states, known))

        values = self.values(gather) if gather else np.empty(0, dtype=np.float32)
        total, cursor = 0.0, 0
        for weight, states, known in buckets:
            take = len(states)
            # The replier picks its own best: our value is the negation of the
            # afterstate, which belongs to us.
            options = list(known) + [-float(v) for v in values[cursor:cursor + take]]
            cursor += take
            total += weight * (max(options) if options else 0.0)
        return total
