"""Head-to-head evaluation for Intransitive: matches and round-robin.

Blue moves first and the edge is real, so every opening is played twice, once
with each agent moving first, and it cancels.

Draws are handled as in Connect Four -- half a point each in the reported score
rate, in the Bradley-Terry fit and in the first-player advantage estimate -- but
they arrive by a different route.  Here a draw is the 200-half-move no-capture
rule, so it is not a property of a filled board but of two players who have run
out of ways to make progress, and it is much commoner between two weak players
than between two strong ones.  A hard ply cap sits above it as a guard, so an
agent that somehow keeps capturing forever still returns a game.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Fitting ratings is game-independent and shared with the other games.
from alphazero_core.elo import elo_standard_errors, fit_elo  # noqa: F401

from .agents.base import Agent
from .heuristics import static_order
from .rps2_game import BLUE, MAX_PLIES, N, RED, RPS2Board

DRAW = "draw"


@dataclass
class GameRecord:
    black: str          # the first player (BLUE), named as in the Hex arena
    white: str          # the second player (RED)
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
    """Play one game. ``opening`` forces blue's first move."""
    board = RPS2Board(rows, cols)
    black.reset()
    white.reset()
    moves: list[int] = []
    last: int | None = None
    t0 = time.time()
    if opening is not None:
        board.play(opening)
        moves.append(opening)
        last = opening
    limit = max_plies or MAX_PLIES
    clock = {BLUE: 0.0, RED: 0.0}
    counts = {BLUE: 0, RED: 0}
    while not board.is_terminal() and board.move_count < limit:
        side = board.to_move
        agent = black if side == BLUE else white
        tm = time.time()
        move = agent.select_move(board, last)
        clock[side] += time.time() - tm
        counts[side] += 1
        if not board.is_legal(move):
            raise ValueError(f"{agent.name} played illegal move {move}")
        board.play(move)
        moves.append(move)
        last = move
    if board.winner == BLUE:
        winner = black.name
    elif board.winner == RED:
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
        black_seconds=clock[BLUE],
        white_seconds=clock[RED],
        black_moves=counts[BLUE],
        white_moves=counts[RED],
    )


def balanced_openings(cols: int, count: int, rng: np.random.Generator,
                      rows: int = N) -> list[int | None]:
    """A spread of forced opening moves, so matches do not repeat one game.

    Deterministic agents would otherwise replay the same game every time.  The
    moves are taken in the order the static evaluation ranks them, so a short
    match is played from openings a real player might choose and a long one
    eventually reaches the odd ones.  There are 36 legal first moves, which is
    far more opening variety than any other game here offers.
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


def round_robin(
    agents: list[Agent],
    rows: int = N,
    cols: int = N,
    games_per_pair: int = 20,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> dict:
    """Every agent against every other, sides balanced."""
    rng = rng or np.random.default_rng(0)
    matches = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            t0 = time.time()
            res = play_match(agents[i], agents[j], rows, cols, games_per_pair, rng)
            res.pop("records")
            res["seconds"] = time.time() - t0
            matches.append(res)
            if verbose:
                print(f"  {res['a']:<28} vs {res['b']:<28} "
                      f"{res['a_wins']:>3}-{res['b_wins']:<3} "
                      f"({res['draws']} drawn, {res['a_win_rate'] * 100:5.1f}%)  "
                      f"[{res['seconds']:.0f}s]", flush=True)
    # Half a point each for a draw, which is what the Bradley-Terry fit expects.
    elo = fit_elo([(m["a"], m["b"], m["a_score"], m["games"] - m["a_score"])
                   for m in matches])
    return {"matches": matches, "elo": elo}
