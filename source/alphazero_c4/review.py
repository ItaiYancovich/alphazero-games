"""Evaluating every position of a finished Connect Four game.

Positions are independent -- each one is just "run a search here" -- so a review
parallelises across processes almost perfectly.  The catch is that the workers
have to be *processes*: the search is Python-bound, so threads would serialise
on the GIL, and each worker pins torch to a single thread so N workers actually
occupy N cores rather than fighting over all of them.

The per-position work lives here rather than in the GUI so that a spawned worker
can import it by name, which is the only way this works on Windows.

The Hex version of this file is its twin; the difference is that a finished
position here can be a draw, which reads as a value of exactly 0 rather than a
win for somebody.
"""

from __future__ import annotations

import numpy as np

from alphazero_core.mcts import BATCH, run_search  # noqa: F401  (shared with Hex)

from .c4_game import COLS, RED, ROWS, Connect4Board, move_to_str
from .mcts import MCTSConfig, Search

C_PUCT = 1.6


def readout(search: Search, board: Connect4Board) -> dict:
    """What the review keeps for one position."""
    # Everything is reported from Red's point of view, so neither the graph nor
    # the numbers flip meaning every ply.
    red = 1.0 if board.to_move == RED else -1.0
    best = best_value = None
    if search.is_expanded(0):
        visits = search.N[0]
        proven = search.root_proven_moves()
        pick = int(proven[int(np.argmax(visits[proven]))]) if len(proven) \
            else int(np.argmax(visits))
        best = int(search.moves[0][pick])
        # What the position would be worth had the search's choice been played.
        best_value = float(search.root_child_scores()[pick]) * red
    return dict(
        value_black=search.root_score() * red,
        best=best,
        best_label=move_to_str(best, board.cols) if best is not None else None,
        best_value_black=best_value,
        proven=search.solved[0] != 0,
        to_move=board.to_move,
        sims=int(search.sims_done),
    )


def terminal_readout(board: Connect4Board) -> dict:
    """A finished position is a certainty, not a network estimate."""
    if board.winner == 0:
        value = 0.0  # a full board with no four: drawn, and worth exactly zero
    else:
        value = 1.0 if board.winner == RED else -1.0
    return dict(value_black=value, best=None, best_label=None,
                best_value_black=None, proven=True, to_move=board.to_move, sims=0)


def evaluate_position(task: tuple) -> dict:
    """Worker entry point: evaluate the position after ``moves``.

    Takes the move prefix rather than a board because a board is far more
    awkward to ship between processes than a list of ints.
    """
    moves, shape, ckpt, sims, index, *rest = task
    device = rest[0] if rest else "cpu"
    rows, cols = shape if isinstance(shape, (tuple, list)) else (ROWS, COLS)
    import torch

    # One thread per worker: the parallelism is across positions, and letting
    # every worker also spin up its own thread pool only causes contention.
    torch.set_num_threads(1)
    from .registry import evaluator  # cached per process, so loaded once

    engine = evaluator(ckpt, rows, cols, device=device)
    board = Connect4Board(rows, cols)
    for move in moves:
        board.play(move)

    if board.is_terminal():
        entry = terminal_readout(board)
    else:
        cfg = MCTSConfig(simulations=sims, c_puct=C_PUCT, add_noise=False)
        search = Search(board.copy(), cfg, np.random.default_rng(0))
        run_search(search, engine, sims)
        entry = readout(search, board)
    entry["ply"] = index
    return entry
