"""Head-to-head evaluation for Connect Four: balanced matches and round-robin.

Connect Four's first player has a large advantage -- 6x7 is a first-player win
with perfect play -- so, exactly as in Hex, every match plays each opening twice,
once with each agent moving first, and the edge cancels.

The difference from the Hex arena is draws.  A drawn game is half a point to
each side, everywhere: in the reported score rate, in the Bradley-Terry fit that
produces the ratings, and in the first-player advantage estimate.  Counting them
as anything else (or dropping them) would flatter whichever agent draws more.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Fitting ratings is game-independent and shared with Hex.
from alphazero_core.elo import elo_standard_errors, fit_elo  # noqa: F401

from .agents.base import Agent
from .c4_game import COLS, RED, ROWS, YELLOW, Connect4Board
from .heuristics import centre_order

DRAW = "draw"


@dataclass
class GameRecord:
    black: str          # the first player (Red), named as in the Hex arena
    white: str          # the second player (Yellow)
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
    rows: int = ROWS,
    cols: int = COLS,
    opening: int | None = None,
    max_plies: int | None = None,
    record_moves: bool = False,
) -> GameRecord:
    """Play one game. ``opening`` forces the first player's opening drop."""
    board = Connect4Board(rows, cols)
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
    clock = {RED: 0.0, YELLOW: 0.0}
    counts = {RED: 0, YELLOW: 0}
    while not board.is_terminal() and board.move_count < limit:
        side = board.to_move
        agent = black if side == RED else white
        tm = time.time()
        move = agent.select_move(board, last)
        clock[side] += time.time() - tm
        counts[side] += 1
        if not board.is_legal(move):
            raise ValueError(f"{agent.name} played illegal move {move}")
        board.play(move)
        moves.append(move)
        last = move
    if board.winner == RED:
        winner = black.name
    elif board.winner == YELLOW:
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
        black_seconds=clock[RED],
        white_seconds=clock[YELLOW],
        black_moves=counts[RED],
        white_moves=counts[YELLOW],
    )


def balanced_openings(cols: int, count: int, rng: np.random.Generator,
                      rows: int = ROWS) -> list[int | None]:
    """A spread of forced opening drops, so matches do not repeat one game.

    Deterministic agents would otherwise replay the same game every time.  The
    columns are taken from the middle outwards -- Connect Four's sane openings
    are central, and there are only seven of them, so with more pairs than
    columns the list simply repeats.
    """
    if count <= 1:
        return [None]
    bottom = (rows - 1) * cols
    cells = [bottom + c for c in centre_order(cols)][: max(count, 1)]
    rng.shuffle(cells)
    return cells[:count]


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    rows: int = ROWS,
    cols: int = COLS,
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
    rows: int = ROWS,
    cols: int = COLS,
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
