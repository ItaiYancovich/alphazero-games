"""Evaluating every position of a finished game.

Positions are independent -- each one is just "run a search here" -- so a review
parallelises across processes almost perfectly.  The catch is that the workers
have to be *processes*: the search is Python-bound, so threads would serialise
on the GIL, and each worker pins torch to a single thread so N workers actually
occupy N cores rather than fighting over all of them.

The per-position work lives here rather than in ``hex_gui`` so that a spawned
worker can import it by name, which is the only way this works on Windows.
"""

from __future__ import annotations

import numpy as np

from .hex_game import BLACK, HexBoard, move_to_str
from alphazero_core.mcts import BATCH, run_search  # noqa: F401  (shared with C4)

from .mcts import MCTSConfig, Search

C_PUCT = 1.6


def readout(search: Search, board: HexBoard) -> dict:
    """What the review keeps for one position."""
    # Everything is reported from Red's point of view, so neither the graph nor
    # the numbers flip meaning every ply.
    red = 1.0 if board.to_move == BLACK else -1.0
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
        best_label=move_to_str(best, board.n) if best is not None else None,
        best_value_black=best_value,
        proven=search.solved[0] != 0,
        to_move=board.to_move,
        sims=int(search.sims_done),
    )


def terminal_readout(board: HexBoard) -> dict:
    """A finished position is a certainty, not a network estimate."""
    return dict(value_black=1.0 if board.winner == BLACK else -1.0,
                best=None, best_label=None, best_value_black=None, proven=True,
                to_move=board.to_move, sims=0)


def evaluate_position(task: tuple) -> dict:
    """Worker entry point: evaluate the position after ``moves``.

    Takes the move prefix rather than a board because a board is far more
    awkward to ship between processes than a list of ints.
    """
    moves, board_size, ckpt, sims, index, *rest = task
    device = rest[0] if rest else "cpu"
    import torch

    # One thread per worker: the parallelism is across positions, and letting
    # every worker also spin up its own thread pool only causes contention.
    torch.set_num_threads(1)
    from .registry import evaluator  # cached per process, so loaded once

    engine = evaluator(ckpt, board_size, device=device)
    board = HexBoard(board_size)
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
