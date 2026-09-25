"""Playing agents against each other, and counting it honestly.

This is the file the v2 run needed and did not have.  Its evaluation dealt a
fresh shuffle every game and played thirty of them, which in a game this
deal-driven is a 95% band of about +/-0.18 -- so its reported score against the
planner wandering between 0.43 and 0.50 was one number, not a trend, and its
gate (fifty games, promote above 0.53) was testing for an effect several times
smaller than its own error bar.

Two changes fix that:

**Paired deals.**  Every deal is played twice, once with each agent in each
seat, and the *pair* is one sample.  Whatever the shuffle was worth is then in
both halves and cancels.

**A sequential test.**  Instead of a fixed number of games and a fixed
threshold, the gate runs an SPRT between "no better than the incumbent" and
"better by ``elo1``", and stops as soon as either is established.  Easy
decisions cost a handful of pairs; close ones get the games they need.

``play_table`` is the general case for three and four seats, where a pair is a
rotation through the seats rather than a swap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .agents import Agent
from .game import Game, as_py_random


def play_game(agents: list[Agent], deal_seed) -> Game:
    """One game on a named deal.  ``agents[k]`` sits in seat ``k + 1``."""
    board = Game(len(agents), as_py_random(deal_seed))
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


def play_pair(a: Agent, b: Agent, deal_seed) -> tuple[float, float]:
    """One deal played both ways.  Returns A's score in each orientation."""
    first = play_game([a, b], deal_seed).score_for(1)
    second = play_game([b, a], deal_seed).score_for(2)
    return first, second


def play_match(a: Agent, b: Agent, pairs: int, seed0: int = 0) -> dict:
    """``pairs`` paired deals -- ``2 * pairs`` games -- with an honest interval."""
    res = [play_pair(a, b, seed0 + 1013 * i) for i in range(pairs)]
    return summarise(res)


def summarise(res: list[tuple[float, float]]) -> dict:
    """Aggregate paired results, reporting the paired standard error."""
    n = max(len(res), 1)
    paired = np.asarray([(x + y) / 2.0 for x, y in res], dtype=np.float64)
    singles = np.asarray([v for pair in res for v in pair], dtype=np.float64)
    se = float(paired.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    wins = int((singles > 0.5).sum())
    losses = int((singles < 0.5).sum())
    ties = int((singles == 0.5).sum())
    return {"pairs": len(res), "games": len(singles),
            "score": float(paired.mean()) if len(res) else 0.5,
            "se": se, "ci95": 1.96 * se if se == se else float("nan"),
            "a_wins": wins, "b_wins": losses, "ties": ties}


# ------------------------------------------------------------------- the test
def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


@dataclass
class SPRT:
    """A sequential test between "no better" and "better by ``elo1``".

    The generalised SPRT on the paired scores: normal-approximate the sample and
    compare the two hypotheses' likelihoods.  It stops when the evidence is
    conclusive rather than after a number of games picked in advance, which is
    the only way a gate can be both cheap on obvious cases and trustworthy on
    close ones.
    """

    elo0: float = 0.0
    elo1: float = 35.0
    alpha: float = 0.05
    beta: float = 0.05
    min_pairs: int = 12
    max_pairs: int = 200

    @property
    def upper(self) -> float:
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def lower(self) -> float:
        return math.log(self.beta / (1.0 - self.alpha))

    def llr(self, res: list[tuple[float, float]]) -> float:
        n = len(res)
        if n < 2:
            return 0.0
        x = np.asarray([(a + b) / 2.0 for a, b in res], dtype=np.float64)
        mean = float(x.mean())
        var = float(x.var(ddof=1))
        if var < 1e-9:
            var = 1e-9
        p0, p1 = elo_to_score(self.elo0), elo_to_score(self.elo1)
        return n * ((mean - p0) ** 2 - (mean - p1) ** 2) / (2.0 * var)

    def verdict(self, res: list[tuple[float, float]]) -> str:
        n = len(res)
        if n < self.min_pairs:
            return "continue"
        llr = self.llr(res)
        if llr >= self.upper:
            return "accept"
        if llr <= self.lower:
            return "reject"
        return "continue" if n < self.max_pairs else "inconclusive"


def sprt_match(a: Agent, b: Agent, test: SPRT, seed0: int = 0) -> dict:
    """Play paired deals until the test decides, or until ``max_pairs``."""
    res: list[tuple[float, float]] = []
    verdict = "continue"
    while verdict == "continue":
        res.append(play_pair(a, b, seed0 + 1013 * len(res)))
        verdict = test.verdict(res)
    out = summarise(res)
    out["verdict"] = verdict
    out["llr"] = test.llr(res)
    return out


# ------------------------------------------------------------ larger tables
def play_table(agents: list[Agent], rounds: int, seed0: int = 0) -> dict:
    """Each deal played ``n`` times, rotating the seating: the paired form of a table."""
    n = len(agents)
    score = np.zeros(n, dtype=np.float64)
    per_round = np.zeros((rounds, n), dtype=np.float64)
    plies = 0
    for r in range(rounds):
        deal = seed0 + 1013 * r
        for shift in range(n):
            order = [agents[(j + shift) % n] for j in range(n)]
            board = play_game(order, deal)
            plies += board.move_count
            for seat in range(1, n + 1):
                per_round[r, (seat - 1 + shift) % n] += board.score_for(seat) / n
        score += per_round[r]
    mean = score / max(rounds, 1)
    se = (per_round.std(axis=0, ddof=1) / math.sqrt(rounds)
          if rounds > 1 else np.full(n, float("nan")))
    return {"rounds": rounds, "games": rounds * n, "scores": mean.tolist(),
            "se": se.tolist(), "avg_plies": plies / max(rounds * n, 1)}
