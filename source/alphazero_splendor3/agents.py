"""What plays a game, and what the ladder rates.

Four rungs, each a claim about where its strength comes from: the floor, pure
game knowledge one ply deep, the network with no search at all, and the network
inside the Gumbel search.  Every agent sees a *determinization* rather than the
server's own position, so a card it has not been shown cannot influence it.
"""

from __future__ import annotations

import random

import numpy as np

from .evaluator import BatchEvaluator
from .game import Game, as_py_random
from .heuristics import plan_value
from .search import Search, SearchConfig, run_search


class Agent:
    name = "agent"
    last_value = 0.0

    def select_move(self, board: Game, last_move: int | None = None) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None, name: str = "random"):
        self.rng = as_py_random(seed)
        self.name = name

    def select_move(self, board: Game, last_move: int | None = None) -> int:
        return self.rng.choice(board.legal_actions())


class PlannerAgent(Agent):
    """Greedy over the reach evaluation, averaged across guesses at the deck.

    The reference opponent: it is the strongest thing in this project that has
    no network in it, so "level with the planner" is the line a run has to
    cross before any of its other numbers mean anything.
    """

    def __init__(self, samples: int = 2, noise: float = 0.05,
                 seed: int | None = None, name: str = "planner"):
        self.samples = max(1, int(samples))
        self.noise = float(noise)
        self.rng = as_py_random(seed)
        self.name = name
        self.last_value = 0.0

    def move_scores(self, board: Game) -> tuple[list[int], list[float]]:
        seat = board.to_move
        legal = board.legal_actions()
        totals = [0.0] * len(legal)
        for _ in range(self.samples):
            view = board.determinize(seat, self.rng)
            for i, move in enumerate(legal):
                after = view.copy()
                after.play(move)
                totals[i] += plan_value(after, seat)
        inv = 1.0 / self.samples
        return legal, [t * inv for t in totals]

    def select_move(self, board: Game, last_move: int | None = None) -> int:
        legal, scores = self.move_scores(board)
        if len(legal) == 1:
            self.last_value = scores[0]
            return legal[0]
        if self.noise > 0:
            gauss = self.rng.gauss
            scores = [s + gauss(0.0, self.noise) for s in scores]
        best = max(range(len(legal)), key=scores.__getitem__)
        self.last_value = scores[best]
        return legal[best]


class PolicyAgent(Agent):
    """The policy head alone -- one forward pass a move, no tree.

    What the network knows on its own, which is the number that says whether
    the search is carrying it.
    """

    def __init__(self, engine: BatchEvaluator, temperature: float = 0.0,
                 seed: int | None = None, name: str = "policy"):
        self.engine = engine
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.pyrng = as_py_random(seed)
        self.name = name
        self.last_value = 0.0

    def select_move(self, board: Game, last_move: int | None = None) -> int:
        view = board.determinize(board.to_move, self.pyrng)
        logp, values = self.engine.evaluate([view])
        self.last_value = float(values[0][0])
        p = logp[0]
        legal = view.legal_actions()
        if self.temperature <= 1e-3:
            return int(max(legal, key=lambda m: p[m]))
        w = np.exp((p[legal] - p[legal].max()) / self.temperature)
        w /= w.sum()
        return int(self.rng.choice(legal, p=w))


class GumbelAgent(Agent):
    """The network inside the max^n Gumbel search.

    ``gumbel_scale`` is the exploration: 1.0 samples root actions the way
    self-play does, 0.0 makes the search deterministic, which is what a rated
    game wants.
    """

    def __init__(self, engine: BatchEvaluator, simulations: int = 128,
                 max_considered: int = 16, batch_size: int = 16,
                 gumbel_scale: float = 0.0, seed: int | None = None,
                 name: str = "az3"):
        self.engine = engine
        self.cfg = SearchConfig(simulations=int(simulations),
                                max_considered=int(max_considered),
                                gumbel_scale=float(gumbel_scale))
        self.batch_size = int(batch_size)
        self.rng = as_py_random(seed)
        self.name = name
        self.last_value = 0.0
        self.last_search: Search | None = None

    def select_move(self, board: Game, last_move: int | None = None) -> int:
        legal = board.legal_actions()
        if len(legal) == 1:
            return legal[0]
        search = Search(board, self.cfg, self.rng)
        run_search(search, self.engine, self.batch_size)
        self.last_search = search
        self.last_value = search.root_score()
        return int(search.best_action())
