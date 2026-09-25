"""Head-to-head evaluation for backgammon.

Two things differ from the other two arenas, and both come from the dice.

**A game is worth points, not a win.**  A gammon is two and a backgammon three,
so the natural measure is *equity per game* -- average points won -- and a
player can win more games while losing on points.  Both numbers are reported;
the rating fit uses the win rate, because Bradley-Terry is a model of who beats
whom, and equity is reported alongside as the thing that actually matters.

**Colour balancing is not enough.**  Half the variance in a match is the dice,
so a pair of games is played from the *same seed*, with the agents swapped: both
sides get the same rolls, and a result then reflects the players rather than who
was luckier.  This is standard practice for comparing bots, and it cuts the
number of games needed for a given error bar by roughly a factor of four.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from alphazero_core.elo import elo_standard_errors, fit_elo  # noqa: F401

from .agents.base import Agent
from .bg_game import BLACK, WHITE, Backgammon

MAX_TURNS = 600  # a stuck game is a bug, but it must not hang the tournament


@dataclass
class GameRecord:
    black: str
    white: str
    winner: str
    points: int          # 1, 2 or 3
    turns: int
    seed: int
    seconds: float = 0.0
    black_seconds: float = 0.0
    white_seconds: float = 0.0
    moves: list = field(default_factory=list)

    def equity_for(self, name: str) -> float:
        """Points won on the -1..+1 scale, from ``name``'s point of view."""
        magnitude = self.points / 3.0
        return magnitude if self.winner == name else -magnitude


def play_game(black: Agent, white: Agent, seed: int = 0,
              record_moves: bool = False) -> GameRecord:
    """One game.  ``seed`` fixes the dice, so a rematch can replay them."""
    board = Backgammon()
    rng = np.random.default_rng(seed)
    black.reset()
    white.reset()
    clock = {BLACK: 0.0, WHITE: 0.0}
    moves = []
    t0 = time.time()
    while not board.is_terminal() and board.move_count < MAX_TURNS:
        board.roll_dice(rng)
        side = board.to_move
        agent = black if side == BLACK else white
        tm = time.time()
        move = agent.select_move(board)
        clock[side] += time.time() - tm
        if record_moves:
            moves.append({"dice": list(board.dice), "move": [list(h) for h in move],
                          "player": side})
        board.play(move)
    winner = black.name if board.winner == BLACK else white.name
    return GameRecord(
        black=black.name, white=white.name, winner=winner,
        points=board.points_won(), turns=board.move_count, seed=seed,
        seconds=time.time() - t0,
        black_seconds=clock[BLACK], white_seconds=clock[WHITE],
        moves=moves,
    )


def play_match(agent_a: Agent, agent_b: Agent, games: int = 20,
               rng: np.random.Generator | None = None,
               verbose: bool = False) -> dict:
    """Play ``games`` games in duplicate pairs: same dice, sides swapped."""
    rng = rng or np.random.default_rng(0)
    pairs = max(1, games // 2)
    records: list[GameRecord] = []
    a_wins = 0
    a_points = 0.0
    for _ in range(pairs):
        seed = int(rng.integers(1 << 30))
        for first, second in ((agent_a, agent_b), (agent_b, agent_a)):
            rec = play_game(first, second, seed=seed)
            records.append(rec)
            if rec.winner == agent_a.name:
                a_wins += 1
                a_points += rec.points
            else:
                a_points -= rec.points
        if verbose:
            print(f"    seed {seed}: {records[-2].winner} / {records[-1].winner}",
                  flush=True)
    total = len(records)
    return {
        "a": agent_a.name,
        "b": agent_b.name,
        "games": total,
        "a_wins": a_wins,
        "b_wins": total - a_wins,
        "draws": 0,                       # backgammon has none
        "a_win_rate": a_wins / total,
        "a_points_per_game": a_points / total,
        "a_equity": a_points / total / 3.0,
        "gammon_rate": sum(1 for r in records if r.points >= 2) / total,
        "avg_turns": float(np.mean([r.turns for r in records])),
        "avg_seconds": float(np.mean([r.seconds for r in records])),
        "records": records,
    }


def round_robin(agents: list[Agent], games_per_pair: int = 20,
                rng: np.random.Generator | None = None,
                verbose: bool = True) -> dict:
    rng = rng or np.random.default_rng(0)
    matches = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            t0 = time.time()
            res = play_match(agents[i], agents[j], games_per_pair, rng)
            res.pop("records")
            res["seconds"] = time.time() - t0
            matches.append(res)
            if verbose:
                print(f"  {res['a']:<28} vs {res['b']:<28} "
                      f"{res['a_wins']:>3}-{res['b_wins']:<3} "
                      f"({res['a_equity']:+.3f} equity)  [{res['seconds']:.0f}s]",
                      flush=True)
    elo = fit_elo([(m["a"], m["b"], m["a_wins"], m["b_wins"]) for m in matches])
    return {"matches": matches, "elo": elo}
