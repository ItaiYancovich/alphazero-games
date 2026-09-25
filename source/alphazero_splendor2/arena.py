"""Playing agents against each other, and counting it honestly.

Seats are swapped every game so that whatever first move is worth is shared
equally, and a tie counts as half.  ``play_match`` is the two-player case the
ratings use; ``play_table`` is the general one, for three and four seats.
"""

from __future__ import annotations

import numpy as np

from .agents import Agent
from .game import Game, as_py_random


def play_game(agents: list[Agent], rng, seed=None) -> Game:
    """One game.  ``agents[k]`` sits in seat ``k + 1``."""
    board = Game(len(agents), as_py_random(seed if seed is not None else rng))
    for agent in agents:
        agent.reset()
    last: int | None = None
    while not board.is_terminal():
        move = int(agents[board.to_move - 1].select_move(board, last))
        if move not in board.legal_actions():
            raise ValueError(f"{agents[board.to_move - 1].name} played an "
                             f"illegal move {move}")
        board.play(move)
        last = move
    return board


def play_match(a: Agent, b: Agent, games: int, rng=None) -> dict:
    """``games`` games at a two-player table, seats swapped every other game."""
    rng = as_py_random(rng)
    wins = losses = ties = 0
    score = 0.0
    plies = 0
    for i in range(games):
        a_first = (i % 2 == 0)
        seats = [a, b] if a_first else [b, a]
        board = play_game(seats, rng)
        a_seat = 1 if a_first else 2
        s = board.score_for(a_seat)
        score += s
        plies += board.move_count
        if s > 0.5:
            wins += 1
        elif s < 0.5:
            losses += 1
        else:
            ties += 1
    return {"games": games, "a_wins": wins, "b_wins": losses, "ties": ties,
            "a_win_rate": score / max(games, 1),
            "avg_plies": plies / max(games, 1)}


def play_table(agents: list[Agent], games: int, rng=None) -> dict:
    """``games`` games at a table of ``len(agents)``, rotating the seating.

    Rotating rather than shuffling, so every agent sits in every seat the same
    number of times whenever ``games`` is a multiple of the table size -- which
    is what makes the scores comparable when the first seat has an edge.
    """
    rng = as_py_random(rng)
    n = len(agents)
    score = np.zeros(n, dtype=np.float64)
    plies = 0
    for i in range(games):
        order = [agents[(j + i) % n] for j in range(n)]
        board = play_game(order, rng)
        plies += board.move_count
        for seat in range(1, n + 1):
            score[(seat - 1 + i) % n] += board.score_for(seat)
    return {"games": games, "scores": (score / max(games, 1)).tolist(),
            "avg_plies": plies / max(games, 1)}
