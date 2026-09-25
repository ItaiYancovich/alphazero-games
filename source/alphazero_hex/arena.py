"""Head-to-head evaluation: balanced matches, round-robin, and Elo.

Hex has a large first-player advantage on 11x11, so any honest comparison has
to control for colour.  Every match here plays each opening position twice --
once with each agent as Black -- so the advantage cancels exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Fitting ratings is game-independent, so it lives in the core.  Re-exported
# here because the tournament, the ladder and the tests all import it from
# ``arena``, which is where it used to be defined.
from alphazero_core.elo import elo_standard_errors, fit_elo  # noqa: F401

from .hex_game import BLACK, WHITE, HexBoard
from .agents.base import Agent


@dataclass
class GameRecord:
    black: str
    white: str
    winner: str
    plies: int
    opening: int | None
    moves: list[int] = field(default_factory=list)
    seconds: float = 0.0
    black_seconds: float = 0.0
    white_seconds: float = 0.0
    black_moves: int = 0
    white_moves: int = 0


def play_game(
    black: Agent,
    white: Agent,
    board_size: int = 11,
    opening: int | None = None,
    max_plies: int | None = None,
    record_moves: bool = False,
) -> GameRecord:
    """Play one game. ``opening`` forces Black's first move if given."""
    board = HexBoard(board_size)
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
    clock = {BLACK: 0.0, WHITE: 0.0}
    counts = {BLACK: 0, WHITE: 0}
    while not board.is_terminal() and board.move_count < limit:
        side = board.to_move
        agent = black if side == BLACK else white
        tm = time.time()
        move = agent.select_move(board, last)
        clock[side] += time.time() - tm
        counts[side] += 1
        if not board.is_legal(move):
            raise ValueError(f"{agent.name} played illegal move {move}")
        board.play(move)
        moves.append(move)
        last = move
    winner = black.name if board.winner == BLACK else white.name
    return GameRecord(
        black=black.name,
        white=white.name,
        winner=winner,
        plies=board.move_count,
        opening=opening,
        moves=moves if record_moves else [],
        seconds=time.time() - t0,
        black_seconds=clock[BLACK],
        white_seconds=clock[WHITE],
        black_moves=counts[BLACK],
        white_moves=counts[WHITE],
    )


def balanced_openings(board_size: int, count: int, rng: np.random.Generator) -> list[int | None]:
    """A spread of forced first moves, to stop matches repeating one game.

    Deterministic agents would otherwise replay the same game every time, so
    the opening is what supplies variety; sampling across the board (rather
    than only strong cells) also probes positions the agents did not choose.
    """
    if count <= 1:
        return [None]
    n = board_size
    cells = []
    # Prefer central/short-diagonal cells, which are the sane Hex openings,
    # then fall back to anything for extra variety.
    ranked = sorted(
        range(n * n),
        key=lambda m: (abs(m // n - (n - 1) / 2) + abs(m % n - (n - 1) / 2)),
    )
    cells = ranked[: max(count, 1)]
    rng.shuffle(cells)
    return cells[:count]


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    board_size: int = 11,
    games: int = 20,
    rng: np.random.Generator | None = None,
    openings: list[int | None] | None = None,
    verbose: bool = False,
) -> dict:
    """Play ``games`` games, colours alternating and openings mirrored."""
    rng = rng or np.random.default_rng(0)
    pairs = max(1, games // 2)
    if openings is None:
        openings = balanced_openings(board_size, pairs, rng)
    records: list[GameRecord] = []
    a_wins = 0
    a_wins_as_black = 0
    a_games_as_black = 0
    for i in range(pairs):
        op = openings[i % len(openings)]
        # Same opening played from both sides: the colour edge cancels out.
        r1 = play_game(agent_a, agent_b, board_size, opening=op)
        records.append(r1)
        a_games_as_black += 1
        if r1.winner == agent_a.name:
            a_wins += 1
            a_wins_as_black += 1
        r2 = play_game(agent_b, agent_a, board_size, opening=op)
        records.append(r2)
        if r2.winner == agent_a.name:
            a_wins += 1
        if verbose:
            print(f"    opening {op}: {r1.winner} (A black), {r2.winner} (B black)", flush=True)
    total = len(records)
    return {
        "a": agent_a.name,
        "b": agent_b.name,
        "games": total,
        "a_wins": a_wins,
        "b_wins": total - a_wins,
        "a_win_rate": a_wins / total,
        "a_win_rate_as_black": a_wins_as_black / max(a_games_as_black, 1),
        "black_win_rate": sum(1 for r in records if r.winner == r.black) / total,
        "avg_plies": float(np.mean([r.plies for r in records])),
        "avg_seconds": float(np.mean([r.seconds for r in records])),
        "records": records,
    }


# --------------------------------------------------------------------- Elo
def round_robin(
    agents: list[Agent],
    board_size: int = 11,
    games_per_pair: int = 20,
    rng: np.random.Generator | None = None,
    verbose: bool = True,
) -> dict:
    """Every agent against every other, colours balanced."""
    rng = rng or np.random.default_rng(0)
    matches = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            t0 = time.time()
            res = play_match(agents[i], agents[j], board_size, games_per_pair, rng)
            res.pop("records")
            res["seconds"] = time.time() - t0
            matches.append(res)
            if verbose:
                print(f"  {res['a']:<28} vs {res['b']:<28} "
                      f"{res['a_wins']:>3}-{res['b_wins']:<3} "
                      f"({res['a_win_rate'] * 100:5.1f}%)  [{res['seconds']:.0f}s]", flush=True)
    elo = fit_elo([(m["a"], m["b"], m["a_wins"], m["b_wins"]) for m in matches])
    return {"matches": matches, "elo": elo}
