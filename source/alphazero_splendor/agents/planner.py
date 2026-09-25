"""The strong classical Splendor player: one that knows what it is saving for.

The greedy :class:`~alphazero_splendor.agents.heuristic.HeuristicAgent` prices a
token at a flat rate whatever colour it is, so it takes essentially arbitrary
gems -- and in Splendor taking gems is most of what a turn is.  What is missing
from its account of the game is **reach**, and that is what this one adds:
:func:`~alphazero_splendor.heuristics.plan_value` asks, for every card worth
buying, how many turns of collecting stand between here and owning it.  Taking
two blue is then worth something specific -- it is a turn off that card --
rather than worth 0.12 like any other two gems.

There is also a beam-limited max^n lookahead in here, switched off by default;
the measurement that decided that is at the bottom of this docstring, and it is
the more interesting half of the story.

Why a *beam* and not alpha-beta.  Alpha-beta prunes on the promise that a branch
cannot affect the result, which needs two players and a zero-sum score; at three
or four seats max^n admits no such cut -- every seat maximises its own component
and there is nothing to be provably worse than.  So the pruning here is a plain
beam: evaluate every legal move one ply, keep the best ``beam`` of them, and
recurse only on those.  That is not sound, and a beam can throw away the best
move; it is what there is at this branching factor, and it is honest about what
it is.

Why several determinizations.  The deck is shuffled, so a search over one guess
at it will happily plan around a card that is not really there.  The agent draws
``samples`` independent guesses consistent with what it can see, searches each,
and averages a move's value over them -- perfect-information Monte Carlo, the
standard treatment for a game like this, and the same information barrier every
other agent here plays under.

**And it ships at depth 1, which is to say with the lookahead switched off.**
That is a measurement, not an oversight.  Depth 1 and depth 2 played each other
84 games and finished exactly level, 42-42, having disagreed about which was
better in each half -- and that is the comparison that settles it, being between
the two agents themselves rather than against a third party that has since
changed.  So the ~120 Elo this agent gains over the greedy heuristic is the
*evaluation*, and the tree adds nothing measurable for eight times the time.
``depth`` is kept because it is the control that establishes that, and because a
better evaluation might one day give the search something to work with; the
default is 1 because there is no evidence for paying more.

Why the tree does so little is not mysterious: the beam is ordered by the same
evaluation that scores the leaves, so a second ply mostly re-ranks an ordering
that was already good, and Splendor at one reply deep has very little forcing
tactics in it -- the opponent has thirty replies and only occasionally the one
that matters to you.
"""

from __future__ import annotations

import numpy as np

from alphazero_core.vecagents import VecAgent

from ..heuristics import plan_value, plan_values
from ..splendor_game import SplendorGame


class PlannerAgent(VecAgent):
    """Max^n over a beam, averaged over several guesses at the deck.

    ``depth`` counts *plies*, not rounds: at a two-player table depth 3 is your
    move, the reply and your answer to it.  ``depth=1`` -- the default, for the
    reason given in the module docstring -- is greedy over the evaluation, with
    no tree at all, and is what any deeper setting has to be measured against.
    """

    def __init__(self, depth: int = 1, beam: int = 8, samples: int = 2,
                 noise: float = 0.03, seed: int | None = None,
                 name: str = "planner"):
        self.depth = max(1, int(depth))
        self.beam = max(1, int(beam))
        self.samples = max(1, int(samples))
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.last_value = 0.0
        self.nodes = 0

    # ------------------------------------------------------------------ search
    def _beam(self, board: SplendorGame):
        """The most promising moves, played, best first.

        Ordered by what the position is worth to the seat *on turn* -- which is
        the whole of max^n: each node prefers what is best for whoever is moving
        there, not for the searcher.  Only the survivors are recursed into.
        """
        seat = board.to_move
        scored = []
        for move in board.legal_moves():
            after = board.copy()
            after.play(int(move))
            self.nodes += 1
            scored.append((plan_value(after, seat), int(move), after))
        scored.sort(key=lambda row: -row[0])
        return scored[:self.beam]

    def _search(self, board: SplendorGame, depth: int) -> np.ndarray:
        """Values of ``board`` to every seat, as a ``(players + 1,)`` vector."""
        if board.is_terminal():
            return 100.0 * board.result_vector().astype(np.float64)
        if depth <= 0:
            return plan_values(board)

        seat = board.to_move
        best: np.ndarray | None = None
        for _score, _move, after in self._beam(board):
            values = self._search(after, depth - 1)
            # The seat on turn takes whatever is best for itself, and the whole
            # vector comes with it -- that is what "max^n" means, and why there
            # is no sign flip anywhere in this file.
            if best is None or values[seat] > best[seat]:
                best = values
        return best if best is not None else plan_values(board)

    # ------------------------------------------------------------------ moving
    def select_move(self, board: SplendorGame, last_move: int | None = None) -> int:
        seat = board.to_move
        legal = board.legal_moves()
        if len(legal) == 1:
            return int(legal[0])

        index = {int(m): i for i, m in enumerate(legal)}
        total = np.zeros(len(legal), dtype=np.float64)
        seen = np.zeros(len(legal), dtype=np.float64)
        self.nodes = 0
        for _ in range(self.samples):
            view = board.determinize(seat, self.rng)
            # The beam is chosen per determinization: a different deck makes a
            # different set of moves look promising, which is the point of
            # drawing more than one.
            for _score, move, after in self._beam(view):
                values = self._search(after, self.depth - 1)
                total[index[move]] += values[seat]
                seen[index[move]] += 1.0

        # A move no sample ever put in its beam has no score, which is not the
        # same as a score of zero -- zero would beat a move that is merely bad.
        mean = np.where(seen > 0, total / np.maximum(seen, 1.0), -np.inf)
        if self.noise > 0:
            mean = mean + self.rng.normal(0.0, self.noise, size=len(mean))
        best = int(np.argmax(mean))
        self.last_value = float(mean[best])
        return int(legal[best])
