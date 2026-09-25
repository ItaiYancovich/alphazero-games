"""The v2 agents: what plays a game, and what the ladder rates.

Four rungs, each one a claim about where its strength comes from:

``RandomAgent``
    The floor.

``PlannerAgent``
    Pure game knowledge, one ply deep: play every legal move, score the position
    it reaches with :func:`alphazero_splendor2.heuristics.plan_value`, take the
    best.  This is the run's *teacher* as well as an opponent -- the network is
    fitted to a few thousand of its games before self-play starts, which is what
    a v1 run spent its first forty iterations discovering for itself.

``PolicyAgent``
    The trained policy head with no search at all.  What the network knows on
    its own, which is the number that says whether search is carrying it.

``MCTSAgent``
    The network inside the max^n open-loop search.

Every agent sees a *determinization* rather than the server's own position, so a
card it has not been shown cannot influence its choice.
"""

from __future__ import annotations

import random

import numpy as np

from .evaluator import BatchEvaluator
from .game import Game, as_py_random
from .heuristics import plan_value
from .search import Search, SearchConfig, run_search


class Agent:
    """Something that picks a move.  ``name`` is what the ratings key on."""

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

    ``samples`` independent determinizations, because a search over one guess at
    the shuffle will happily plan around a card that is not really there.
    ``noise`` is a standard deviation in points-equivalent added to each
    candidate, so the agent varies its play between games instead of replaying
    one line for ever -- and the ladder rates it with the noise it plays with.
    """

    def __init__(self, samples: int = 2, noise: float = 0.05,
                 seed: int | None = None, name: str = "planner"):
        self.samples = max(1, int(samples))
        self.noise = float(noise)
        self.rng = as_py_random(seed)
        self.name = name
        self.last_value = 0.0

    def move_scores(self, board: Game) -> tuple[list[int], list[float]]:
        """Every legal move and what it is worth, in the order ``legal_actions``."""
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
    """The policy head alone -- no tree, one forward pass a move."""

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
        priors, values = self.engine.evaluate([view])
        self.last_value = float(values[0][0])
        p = priors[0]
        legal = view.legal_actions()
        if self.temperature <= 1e-3:
            return int(max(legal, key=lambda m: p[m]))
        w = np.power(np.maximum(p[legal], 1e-12), 1.0 / self.temperature)
        w /= w.sum()
        return int(self.rng.choice(legal, p=w))


class MCTSAgent(Agent):
    """The network inside the max^n open-loop search."""

    def __init__(self, engine: BatchEvaluator, simulations: int = 200,
                 c_puct: float = 1.5, batch_size: int = 16,
                 explore_moves: int = 0, explore_temperature: float = 0.6,
                 add_noise: bool = False, dirichlet_eps: float = 0.25,
                 seed: int | None = None, name: str = "az"):
        self.engine = engine
        self.cfg = SearchConfig(simulations=int(simulations), c_puct=float(c_puct),
                                add_noise=bool(add_noise),
                                dirichlet_eps=float(dirichlet_eps))
        self.batch_size = int(batch_size)
        self.explore_moves = int(explore_moves)
        self.explore_temperature = float(explore_temperature)
        self.rng = as_py_random(seed)
        self.nprng = np.random.default_rng(seed)
        self.name = name
        self.last_value = 0.0
        self.last_search: Search | None = None

    def select_move(self, board: Game, last_move: int | None = None) -> int:
        legal = board.legal_actions()
        if len(legal) == 1:
            return legal[0]
        search = Search(board, self.cfg, self.rng)
        run_search(search, self.engine, self.cfg.simulations, self.batch_size)
        self.last_search = search
        self.last_value = search.root_score()
        counts = search.root_visit_counts()
        if board.move_count < self.explore_moves and self.explore_temperature > 0:
            w = np.power(np.maximum(counts[legal], 0.0),
                         1.0 / self.explore_temperature)
            total = w.sum()
            if total > 0:
                return int(self.nprng.choice(legal, p=w / total))
        return int(max(legal, key=lambda m: counts[m]))
