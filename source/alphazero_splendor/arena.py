"""Head-to-head evaluation for Splendor: duplicate deals and rotated seats.

Two things differ from the Hex and Connect Four arenas, and both come from the
shuffle.

**Duplicate deals.**  The deck is most of the variance in a Splendor game, so a
comparison plays the *same shuffle* several times with the players rotated
through the seats.  Both sides therefore see the same cards come up, and a
result reflects the players rather than who was dealt the cheap tier-3 card.
This is the same trick the backgammon arena uses on the dice, and it is worth
roughly a factor of four in games needed for a given error bar.

**A result is a standing, not a win.**  With three or four seats a game produces
an order of finish.  A record therefore carries a score per seat -- 1 for an
outright win, 0 for outright last, evenly spaced between -- which is
:meth:`~alphazero_splendor.splendor_game.SplendorGame.score_for` and reduces to
the ordinary win/loss at a two-player table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# Fitting ratings is game-independent and shared with the other three games.
from alphazero_core.elo import elo_standard_errors, fit_elo  # noqa: F401

from .agents.base import Agent
from .splendor_game import MAX_PLIES, SplendorGame

DRAW = "draw"


@dataclass
class GameRecord:
    """One finished game, from the point of view of the seating it was played in."""

    names: list[str]          # by seat, first player first
    winner: str               # a name, or ``DRAW`` if the top place was shared
    scores: list[float]       # 0..1 per seat, in seat order
    points: list[int]         # prestige per seat
    cards: list[int]          # development cards per seat, the tie-break
    plies: int
    seed: int
    seconds: float = 0.0
    seat_seconds: list[float] = field(default_factory=list)
    moves: list[int] = field(default_factory=list)

    @property
    def players(self) -> int:
        return len(self.names)

    def score_for(self, name: str) -> float:
        """This agent's score.  Averaged if it somehow occupied two seats."""
        got = [s for n, s in zip(self.names, self.scores) if n == name]
        return float(np.mean(got)) if got else 0.0


def play_game(agents: list[Agent], seed: int = 0,
              record_moves: bool = False) -> GameRecord:
    """One game.  ``seed`` fixes the shuffle, so a rematch can replay the deal."""
    players = len(agents)
    board = SplendorGame(players, np.random.default_rng(seed))
    for agent in agents:
        agent.reset()
    clock = [0.0] * (players + 1)
    counts = [0] * (players + 1)
    moves: list[int] = []
    last: int | None = None
    t0 = time.time()
    while not board.is_terminal():
        seat = board.to_move
        agent = agents[seat - 1]
        tm = time.time()
        move = agent.select_move(board, last)
        clock[seat] += time.time() - tm
        counts[seat] += 1
        if not board.is_legal(move):
            raise ValueError(f"{agent.name} played illegal move {move}")
        board.play(move)
        if record_moves:
            moves.append(int(move))
        last = int(move)

    scores = [board.score_for(s) for s in range(1, players + 1)]
    winner = agents[board.winner - 1].name if board.winner else DRAW
    return GameRecord(
        names=[a.name for a in agents],
        winner=winner,
        scores=scores,
        points=[int(board.points[s]) for s in range(1, players + 1)],
        cards=[int(board.ncards[s]) for s in range(1, players + 1)],
        plies=board.move_count,
        seed=seed,
        seconds=time.time() - t0,
        seat_seconds=clock[1:],
        moves=moves,
    )


def play_match(agent_a: Agent, agent_b: Agent, games: int = 20,
               rng: np.random.Generator | None = None,
               verbose: bool = False) -> dict:
    """Two agents over ``games`` two-player games, seats and deals balanced.

    Each deal is played twice, once from each seat, so the first-player edge
    cancels exactly.
    """
    rng = rng or np.random.default_rng(0)
    pairs = max(1, games // 2)
    records: list[GameRecord] = []
    a_score = 0.0
    a_wins = draws = 0
    for _ in range(pairs):
        seed = int(rng.integers(1 << 30))
        for seating in ((agent_a, agent_b), (agent_b, agent_a)):
            rec = play_game(list(seating), seed=seed)
            records.append(rec)
            score = rec.score_for(agent_a.name)
            a_score += score
            a_wins += 1 if score > 0.75 else 0
            draws += 1 if rec.winner == DRAW else 0
        if verbose:
            print(f"    deal {seed}: {records[-2].winner} / {records[-1].winner}",
                  flush=True)
    total = len(records)
    return {
        "a": agent_a.name,
        "b": agent_b.name,
        "games": total,
        "a_wins": a_wins,
        "b_wins": total - a_wins - draws,
        "draws": draws,
        # Ties count half, as in Connect Four; the name matches the other arenas
        # so the training loop can read any of them.
        "a_win_rate": a_score / total,
        "avg_plies": float(np.mean([r.plies for r in records])),
        "avg_seconds": float(np.mean([r.seconds for r in records])),
        "timeouts": sum(1 for r in records if r.plies >= MAX_PLIES),
        "records": records,
    }
