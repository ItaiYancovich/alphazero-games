"""Sequential probability ratio test for Hex matches.

A fixed-length match has to be sized for the worst case: 100 games at 200
simulations took 44 minutes and still could not separate two networks 35 Elo
apart.  A sequential test asks after every result whether the evidence is
already decisive and stops as soon as it is, so a clear result costs a few dozen
games and only a close one pays for hundreds.

The test is the one chess engine testing (fishtest) uses, adapted to Hex's two
facts:

* **Games come in colour-balanced pairs.**  Hex's first-player advantage is
  large, so each opening is played twice with the colours swapped and the *pair*
  is the sample.  A pair scores 0, 1/2 or 1.  Treating the two games as
  independent would overstate the evidence, because a lopsided opening makes
  both of them predictable.
* **There are no draws**, so a pair's score is simply its number of wins over
  two.

For hypotheses ``H0: elo = elo0`` and ``H1: elo = elo1`` the log-likelihood
ratio is the generalised SPRT's normal approximation::

    LLR = n (s1 - s0) (2 m - s0 - s1) / (2 v)

where ``m`` and ``v`` are the mean and variance of the pair scores and ``s0``,
``s1`` the expected scores at the two Elo values.  The test accepts H1 once
``LLR >= log((1 - beta) / alpha)`` and H0 once ``LLR <= log(beta / (1 - alpha))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hex_game import HexBoard


def expected_score(elo: float) -> float:
    """Expected score of a player ``elo`` points stronger than its opponent."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


@dataclass
class SprtStatus:
    pairs: int
    games: int
    wins: int
    score: float
    elo: float
    elo_lo: float
    elo_hi: float
    llr: float
    lower: float
    upper: float
    decision: str | None  # "H1" (stronger by elo1), "H0" (not by elo0), or None


class Sprt:
    """Accumulates pair results and decides between two Elo hypotheses."""

    def __init__(self, elo0: float = 0.0, elo1: float = 30.0,
                 alpha: float = 0.05, beta: float = 0.05):
        if elo1 <= elo0:
            raise ValueError("elo1 must be greater than elo0")
        self.elo0, self.elo1 = elo0, elo1
        self.lower = math.log(beta / (1.0 - alpha))
        self.upper = math.log((1.0 - beta) / alpha)
        self.counts = [0, 0, 0]  # pairs won 0, 1 and 2 times

    def add(self, wins_in_pair: int) -> None:
        if wins_in_pair not in (0, 1, 2):
            raise ValueError(f"a pair has 0, 1 or 2 wins, not {wins_in_pair}")
        self.counts[wins_in_pair] += 1

    @property
    def pairs(self) -> int:
        return sum(self.counts)

    def _moments(self) -> tuple[float, float]:
        n = self.pairs
        m = (0.5 * self.counts[1] + self.counts[2]) / n
        # Variance with half a pseudo-pair in each outcome.  Without it a run of
        # identical pairs has zero variance and the ratio jumps to infinity on
        # the strength of three or four games.
        c = [x + 0.5 for x in self.counts]
        nc = sum(c)
        mc = (0.5 * c[1] + c[2]) / nc
        vc = (0.25 * c[1] + c[2]) / nc - mc * mc
        return m, vc

    def llr(self) -> float:
        if self.pairs == 0:
            return 0.0
        m, v = self._moments()
        s0, s1 = expected_score(self.elo0), expected_score(self.elo1)
        return self.pairs * (s1 - s0) * (2.0 * m - s0 - s1) / (2.0 * v)

    def status(self) -> SprtStatus:
        n = self.pairs
        wins = self.counts[1] + 2 * self.counts[2]
        if n == 0:
            return SprtStatus(0, 0, 0, 0.5, 0.0, -math.inf, math.inf, 0.0,
                              self.lower, self.upper, None)
        m, v = self._moments()
        half = 1.96 * math.sqrt(v / n)
        llr = self.llr()
        decision = "H1" if llr >= self.upper else "H0" if llr <= self.lower else None
        return SprtStatus(n, 2 * n, wins, m, elo_from_score(m),
                          elo_from_score(m - half), elo_from_score(m + half),
                          llr, self.lower, self.upper, decision)


def random_openings(board_size: int, pairs: int, plies: int,
                    rng: np.random.Generator) -> list[list[int]]:
    """``pairs`` distinct random openings of ``plies`` moves each.

    Uniform over the whole board rather than the central cells: an edge opening
    is a worse position for Black, which is exactly what makes it useful -- the
    pair plays it from both sides, so its imbalance cancels, and deterministic
    agents cannot replay one game hundreds of times.
    """
    n2 = board_size * board_size
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    limit = min(pairs, math.perm(n2, plies)) if plies else 1
    while len(out) < limit:
        seq = tuple(int(x) for x in rng.choice(n2, size=plies, replace=False))
        if seq not in seen:
            seen.add(seq)
            out.append(list(seq))
    return out


def play_from(black, white, board_size: int, opening: list[int],
              swap_rule: bool = False) -> int:
    """Play one game after ``opening``; returns 1 if Black wins, 2 if White."""
    board = HexBoard(board_size, swap_rule=swap_rule)
    black.reset()
    white.reset()
    last = None
    for mv in opening:
        board.play(mv)
        last = mv
    while not board.is_terminal():
        agent = black if board.to_move == 1 else white
        mv = agent.select_move(board, last)
        if not board.is_legal(mv):
            raise ValueError(f"{agent.name} played illegal move {mv}")
        board.play(mv)
        last = mv
    return board.winner


def play_pair(agent_a, agent_b, board_size: int, opening: list[int],
              swap_rule: bool = False) -> int:
    """Both colours from one opening; returns how many of the two A won."""
    wins = 0
    if play_from(agent_a, agent_b, board_size, opening, swap_rule) == 1:
        wins += 1
    if play_from(agent_b, agent_a, board_size, opening, swap_rule) == 2:
        wins += 1
    return wins
