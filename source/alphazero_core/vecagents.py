"""Game-independent agents for the vector-shaped, many-player games.

The counterpart of :mod:`alphazero_core.agents`.  Everything here talks to a
position only through :class:`~alphazero_core.vecstate.VecState`, so the same
code plays any game that can describe itself as a feature vector and a fixed
action list.  Knowledge-driven players are the opposite -- they are nothing but
game knowledge -- and live in each game's own package.

``torch`` is deliberately not imported: an evaluator arrives ready-made, so the
random agent and the GUI's whole non-network half need nothing but numpy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .vecmcts import VecMCTSConfig, VecSearch
from .vecstate import VecState

if TYPE_CHECKING:  # only for the annotations
    from .vecevaluator import BatchVecEvaluator


class VecAgent:
    """Anything that can pick a move in a vector-shaped game."""

    name: str = "agent"

    def reset(self) -> None:
        """Called at the start of every game."""

    def select_move(self, board: VecState, last_move: int | None = None) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class RandomVecAgent(VecAgent):
    """Uniform random legal action -- the floor of the rating scale."""

    def __init__(self, seed: int | None = None, name: str = "random"):
        self.rng = np.random.default_rng(seed)
        self.name = name

    def select_move(self, board: VecState, last_move: int | None = None) -> int:
        return int(self.rng.choice(board.legal_moves()))


class VecPolicyAgent(VecAgent):
    """The network's policy head with no search -- what search is measured against.

    Deterministic at ``temperature == 0``, the default; a positive temperature
    samples from the policy instead, which is what makes a casual opponent stop
    replying identically to the same position.
    """

    def __init__(self, evaluator: "BatchVecEvaluator", temperature: float = 0.0,
                 seed: int | None = None, name: str = "policy-only(no search)"):
        self.evaluator = evaluator
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.last_value = 0.0

    def select_move(self, board: VecState, last_move: int | None = None) -> int:
        # The policy head reads the position, and a position it is not entitled
        # to see would make it a cheat as surely as a search would.
        view = board.determinize(board.to_move, self.rng)
        priors, values = self.evaluator.evaluate([view])
        self.last_value = float(values[0][0])  # seat-relative: 0 is the mover
        legal = board.legal_moves()
        weights = priors[0][legal]
        if self.temperature <= 1e-3:
            return int(legal[int(np.argmax(weights))])
        w = np.power(weights.astype(np.float64), 1.0 / self.temperature)
        total = w.sum()
        if total <= 0:
            return int(self.rng.choice(legal))
        return int(self.rng.choice(legal, p=w / total))


class VecMCTSAgent(VecAgent):
    """AlphaZero at play time, generalised: max^n PUCT guided by the network.

    With ``explore_moves == 0`` (the default, and what the ladder and rated
    games use) this is as deterministic as a game with shuffled decks can be:
    no root noise, and the most-visited action wins.  Set ``explore_moves`` > 0
    to reproduce self-play's opening exploration -- root Dirichlet noise and
    temperature sampling for that many plies -- so the same opening does not
    draw the identical reply every time.
    """

    def __init__(
        self,
        evaluator: "BatchVecEvaluator",
        nactions: int,
        simulations: int = 200,
        c_puct: float = 1.6,
        temperature: float = 0.0,
        explore_moves: int = 0,
        explore_temperature: float = 0.6,
        batch_size: int = 16,
        seed: int | None = None,
        name: str | None = None,
    ):
        self.evaluator = evaluator
        self.nactions = int(nactions)
        self.cfg = VecMCTSConfig(simulations=simulations, c_puct=c_puct,
                                 add_noise=False)
        self.temperature = temperature
        self.explore_moves = explore_moves
        self.explore_temperature = explore_temperature
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.name = name or f"mcts({simulations}sims)"
        self.last_value = 0.0
        self.nodes = 0

    def select_move(self, board: VecState, last_move: int | None = None) -> int:
        exploring = board.move_count < self.explore_moves
        self.cfg.add_noise = exploring
        search = VecSearch(board, self.cfg, self.rng, self.nactions)
        while search.sims_done < self.cfg.simulations:
            states = search.next_leaf_batch(self.batch_size)
            if not states:
                break
            priors, values = self.evaluator.evaluate(states)
            search.expand_batch(priors, values)

        self.last_value = search.root_score()
        self.nodes = search.n_nodes
        temperature = self.explore_temperature if exploring else self.temperature
        probs = search.policy_target(temperature)
        # A search that never got a single simulation in (a zero-simulation
        # budget) leaves every count at zero; fall back to the legal set rather
        # than to action 0.
        legal = board.legal_moves()
        weights = probs[legal]
        if weights.sum() <= 0:
            return int(self.rng.choice(legal))
        if temperature <= 1e-3:
            return int(legal[int(np.argmax(weights))])
        return int(self.rng.choice(legal, p=weights / weights.sum()))
