"""Head-to-head evaluation for Ultimate XX: one game, or a balanced match.

The first player's advantage here is a real question rather than a formality --
in a game where making the last safe move is the whole point, parity matters --
so, as everywhere else in this project, every opening is played twice, once with
each agent moving first, and whatever the edge is cancels.

Draws are handled exactly as in Connect Four: half a point to each side in the
reported score rate and in the first-player advantage estimate.  They are less
common than in Ultimate Tic-Tac-Toe -- the board runs out sooner and somebody
usually has to take a third small board before it does -- but two players who
both avoid the traps reach one often enough that counting them as anything else
would distort the numbers.

There is no round-robin or Elo fit here, because there is nothing to rate: this
game ships with its classical agents only and no trained network, so a match is
run to check the agents play the game rather than to place them on a ladder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .agents.base import Agent
from .heuristics import static_order
from .uxx_game import FIRST, N, SECOND, UltimateXXBoard

DRAW = "draw"


@dataclass
class GameRecord:
    black: str          # the first player, named as in the Hex arena
    white: str          # the second player
    winner: str         # a name, or ``DRAW``
    plies: int
    opening: int | None
    moves: list[int] = field(default_factory=list)
    seconds: float = 0.0
    black_seconds: float = 0.0
    white_seconds: float = 0.0
    black_moves: int = 0
    white_moves: int = 0

    def score_for(self, name: str) -> float:
        if self.winner == DRAW:
            return 0.5
        return 1.0 if self.winner == name else 0.0


def play_game(
    black: Agent,
    white: Agent,
    rows: int = N,
    cols: int = N,
    opening: int | None = None,
    max_plies: int | None = None,
    record_moves: bool = False,
) -> GameRecord:
    """Play one game. ``opening`` forces the first player's opening cell."""
    board = UltimateXXBoard(rows, cols)
    black.reset()
    white.reset()
    moves: list[int] = []
    last: int | None = None
    t0 = time.time()
    if opening is not None:
        board.play(opening)
        moves.append(opening)
        last = opening
    limit = max_plies or board.ncells
    clock = {FIRST: 0.0, SECOND: 0.0}
    counts = {FIRST: 0, SECOND: 0}
    while not board.is_terminal() and board.move_count < limit:
        side = board.to_move
        agent = black if side == FIRST else white
        tm = time.time()
        move = agent.select_move(board, last)
        clock[side] += time.time() - tm
        counts[side] += 1
        if not board.is_legal(move):
            raise ValueError(f"{agent.name} played illegal move {move}")
        board.play(move)
        moves.append(move)
        last = move
    # ``winner`` is whoever did *not* complete a line of small boards.
    if board.winner == FIRST:
        winner = black.name
    elif board.winner == SECOND:
        winner = white.name
    else:
        winner = DRAW
    return GameRecord(
        black=black.name,
        white=white.name,
        winner=winner,
        plies=board.move_count,
        opening=opening,
        moves=moves if record_moves else [],
        seconds=time.time() - t0,
        black_seconds=clock[FIRST],
        white_seconds=clock[SECOND],
        black_moves=counts[FIRST],
        white_moves=counts[SECOND],
    )


def balanced_openings(cols: int, count: int, rng: np.random.Generator,
                      rows: int = N) -> list[int | None]:
    """A spread of forced opening cells, so matches do not repeat one game.

    Deterministic agents would otherwise replay the same game every time.  The
    cells are taken in the order the heuristics rank them -- quietest squares
    first, since in this game the busy ones are the liabilities -- so a short
    match is played from openings a real player might choose, and a long one
    eventually reaches the odd ones.
    """
    if count <= 1:
        return [None]
    cells = static_order()[: max(count, 1)]
    rng.shuffle(cells)
    return cells[:count]


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    rows: int = N,
    cols: int = N,
    games: int = 20,
    rng: np.random.Generator | None = None,
    openings: list[int | None] | None = None,
    verbose: bool = False,
) -> dict:
    """Play ``games`` games, sides alternating and openings mirrored."""
    rng = rng or np.random.default_rng(0)
    pairs = max(1, games // 2)
    if openings is None:
        openings = balanced_openings(cols, pairs, rng, rows)
    records: list[GameRecord] = []
    a_wins = 0
    draws = 0
    a_score_as_first = 0.0
    a_games_as_first = 0
    for i in range(pairs):
        op = openings[i % len(openings)]
        # Same opening played from both sides: the first-move edge cancels out.
        r1 = play_game(agent_a, agent_b, rows, cols, opening=op)
        records.append(r1)
        a_games_as_first += 1
        a_score_as_first += r1.score_for(agent_a.name)
        r2 = play_game(agent_b, agent_a, rows, cols, opening=op)
        records.append(r2)
        for rec in (r1, r2):
            if rec.winner == DRAW:
                draws += 1
            elif rec.winner == agent_a.name:
                a_wins += 1
        if verbose:
            print(f"    opening {op}: {r1.winner} (A first), {r2.winner} (B first)",
                  flush=True)
    total = len(records)
    a_score = a_wins + 0.5 * draws
    first_score = sum(1.0 if r.winner == r.black else (0.5 if r.winner == DRAW else 0.0)
                      for r in records)
    return {
        "a": agent_a.name,
        "b": agent_b.name,
        "games": total,
        "a_wins": a_wins,
        "b_wins": total - a_wins - draws,
        "draws": draws,
        "a_score": a_score,
        # ``a_win_rate`` counts a draw as half a game, so it reads as a score
        # rate; the Hex arena's key name is kept so shared callers do not care.
        "a_win_rate": a_score / total,
        "a_win_rate_as_black": a_score_as_first / max(a_games_as_first, 1),
        "black_win_rate": first_score / total,
        "draw_rate": draws / total,
        "avg_plies": float(np.mean([r.plies for r in records])),
        "avg_seconds": float(np.mean([r.seconds for r in records])),
        "records": records,
    }

