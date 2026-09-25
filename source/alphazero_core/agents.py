"""Game-independent agents: the interface, a random player, and AlphaZero.

Everything here talks to a position only through the
:class:`~alphazero_core.state.GameState` protocol, so the same trained-agent
code plays Hex and Connect Four.  The knowledge-driven players (rule-based,
alpha-beta, rollout MCTS) are the opposite -- they are nothing *but* game
knowledge -- and live in each game's own package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mcts import MCTSConfig, Search, run_search
from .state import GameState

if TYPE_CHECKING:  # only for the annotations below
    from .evaluator import BatchEvaluator

# Deliberately *not* a runtime import: an evaluator is handed in ready-made, and
# importing the module would pull in torch.  The classical agents and the GUI's
# whole non-AlphaZero half need nothing but numpy, and that stays true only if
# this file keeps its hands off torch.


class Agent:
    """Anything that can pick a move.

    ``select_move`` receives the position and the opponent's last move (some
    agents use pattern responses that depend on it) and returns a legal move.
    """

    name: str = "agent"

    def reset(self) -> None:
        """Called at the start of every game."""

    def select_move(self, board: GameState, last_move: int | None = None) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class RandomAgent(Agent):
    """Uniform random legal move -- the floor of the rating scale."""

    def __init__(self, seed: int | None = None, name: str = "random"):
        self.rng = np.random.default_rng(seed)
        self.name = name

    def select_move(self, board: GameState, last_move: int | None = None) -> int:
        return int(self.rng.choice(board.legal_moves()))


class AlphaZeroAgent(Agent):
    """AlphaZero at play time: PUCT search guided by the learned net.

    With ``explore_moves == 0`` (the default, and what the tournament and rated
    games use) this is deterministic: no root noise, ``argmax`` over visit
    counts.  Set ``explore_moves`` > 0 to reproduce self-play's opening
    exploration instead -- root Dirichlet noise plus temperature sampling for
    the first ``explore_moves`` plies of the *game*, full strength after --
    so the same human opening does not draw the identical reply every time.
    """

    def __init__(
        self,
        evaluator: "BatchEvaluator",
        simulations: int = 400,
        c_puct: float = 1.6,
        temperature: float = 0.0,
        explore_moves: int = 0,
        explore_temperature: float = 0.6,
        batch_size: int = 16,
        seed: int | None = None,
        name: str | None = None,
        resistance: bool = True,
        value_discount: float = 0.999,
    ):
        self.evaluator = evaluator
        self.cfg = MCTSConfig(simulations=simulations, c_puct=c_puct, add_noise=False,
                              value_discount=value_discount)
        self.resistance = resistance
        self.temperature = temperature
        self.explore_moves = explore_moves
        self.explore_temperature = explore_temperature
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.name = name or f"alphazero({simulations}sims)"
        self.last_value = 0.0

    def select_move(self, board: GameState, last_move: int | None = None) -> int:
        exploring = board.move_count < self.explore_moves
        self.cfg.add_noise = exploring  # read once, at Search construction below
        search = Search(board.copy(), self.cfg, self.rng)
        # Gather several leaves per network call using virtual loss; a batch of
        # one leaves most of the CPU idle.
        run_search(search, self.evaluator, self.cfg.simulations, self.batch_size)
        # Report the proven result when there is one, so a forced win reads as
        # +1 rather than as an average diluted by unrefuted branches.
        self.last_value = search.root_score()
        moves, _ = search.root_visit_distribution()

        # A proven win ends the discussion: take it, the soonest one there is.
        # A proven loss does not end it -- play the line that lasts longest and
        # make the opponent find every move.
        if self.resistance and not exploring:
            scores = search.root_selection_scores()
            if (scores > 1.0).any() or (scores < 0.0).all():
                return int(moves[int(np.argmax(scores))])
        else:
            proven = search.root_proven_moves()
            if len(proven):
                visits = search.N[0][proven]
                return int(moves[int(proven[int(np.argmax(visits))])])

        temperature = self.explore_temperature if exploring else self.temperature
        moves, probs = search.policy_target(temperature)
        if temperature <= 1e-3:
            return int(moves[int(np.argmax(probs))])
        p = probs.astype(np.float64)
        p /= p.sum()
        return int(self.rng.choice(moves, p=p))


class PolicyOnlyAgent(Agent):
    """Network policy head with no search -- isolates what search contributes.

    Deterministic (``argmax``) at ``temperature == 0``, the default; a positive
    temperature samples from the policy instead, for casual play variety.
    """

    def __init__(
        self,
        evaluator: "BatchEvaluator",
        temperature: float = 0.0,
        seed: int | None = None,
        name: str = "policy-only(no search)",
    ):
        self.evaluator = evaluator
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.name = name

    def select_move(self, board: GameState, last_move: int | None = None) -> int:
        priors, _ = self.evaluator.evaluate([board])
        p = priors[0]
        legal = board.legal_moves()
        canon = np.array([board.to_canonical_move(int(m)) for m in legal])
        vals = p[canon]
        if self.temperature <= 1e-3:
            return int(legal[int(np.argmax(vals))])
        w = np.power(vals.astype(np.float64), 1.0 / self.temperature)
        w /= w.sum()
        return int(self.rng.choice(legal, p=w))
