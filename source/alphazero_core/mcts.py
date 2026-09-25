"""PUCT Monte-Carlo tree search (AlphaZero style), for any game.

The tree is stored as flat per-node numpy arrays so that the selection step is
a handful of vector ops rather than a Python loop over children.

The search is written in a *coroutine* style: :meth:`Search.next_leaf` walks the
tree until it hits a position that needs a network evaluation and then returns,
handing control back to the caller.  The caller can therefore drive many
searches at once and evaluate all their leaves in a single batched forward pass
-- which is the difference between a useless and a usable amount of self-play
on a CPU.

The tree knows nothing about the game beyond the
:class:`~alphazero_core.state.GameState` protocol, and in particular it handles
*drawn* terminal positions: Hex has none, Connect Four does, and a solver that
cannot say "proven draw" would report a dead-drawn position as unresolved
forever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import GameState


# Values of ``Search.solved``.  DRAW is a separate state rather than a zero
# because 0 already means "not proven": a position can be a proven draw, which
# is a fact about the game, or simply unexamined, which is not.
UNKNOWN = 0
WIN = 1
LOSS = -1
DRAW = 2


def solved_value(flag: int) -> float:
    """Game-theoretic value of a solved flag, from the mover's point of view."""
    return 0.0 if flag == DRAW else float(flag)


@dataclass
class MCTSConfig:
    simulations: int = 128
    c_puct: float = 1.6
    dirichlet_alpha: float = 0.0  # 0 -> derived from board size
    dirichlet_eps: float = 0.25
    # Per-ply decay applied to backed-up values.  1.0 keeps search a pure
    # estimate of win probability, which is what self-play wants; a shade below
    # 1.0 makes an equal-value choice prefer the shorter win and the longer
    # loss.  Keep it very close to 1.0: at 0.99 a 60-ply certain win scores
    # 0.55 and an 80%-likely 10-ply win scores 0.54, so the search would trade
    # away a won game for a quicker one.
    value_discount: float = 1.0
    fpu_reduction: float = 0.25  # first-play-urgency: discount unvisited actions
    add_noise: bool = True
    # KataGo's forced playouts, and the pruning that pays for them.  ``k`` in
    # ``n_forced = sqrt(k * P * N)``; 0 turns the whole thing off.
    #
    # Dirichlet noise buys exploration by corrupting the *policy target*: the
    # visits the noise caused are counted in the distribution the network is
    # then asked to fit, so a quarter of what it learns early on is noise
    # nothing could predict.  Forced playouts separate the two.  Each root move
    # is guaranteed its share of visits during the search, and
    # ``policy_target(prune=True)`` subtracts exactly those guaranteed visits
    # back out before building the target -- so the exploration happens and
    # leaves no trace in what is learned.
    forced_playouts: float = 0.0

    def alpha_for(self, ncells: int) -> float:
        if self.dirichlet_alpha > 0:
            return self.dirichlet_alpha
        return max(0.03, 10.0 / ncells)


BATCH = 24  # leaves per network call, as in self-play
BACKGROUND_SPLIT = 2  # see run_search


def run_search(search: "Search", engine, target: int, batch: int = BATCH) -> None:
    """Grow a tree until it has ``target`` simulations behind it.

    ``engine`` is anything with the evaluator's ``evaluate(states)`` signature.

    An engine whose network can answer in the background (``engine.background``
    -- the web build's, where the network runs in other workers) is kept busy:
    the next batch of leaves is collected while the network works on the last
    one, and the last one's answers are backed up while it works on the next,
    so the search's own time hides behind the network's.  The price is a
    second batch in flight, collected without the first one's answers -- as
    engines that search on several threads do.  Every other engine runs one
    batch at a time, exactly as before.
    """
    if not getattr(engine, "background", False):
        while search.sims_done < target:
            states = search.next_leaf_batch(batch)
            if not states:
                break
            priors, values = engine.evaluate(states)
            search.expand_batch(priors, values)
        return
    # Two batches of half the size, so the leaves in flight -- and the virtual
    # loss steering the search away from them -- are as many as before.
    half = max(1, batch // BACKGROUND_SPLIT)
    ahead = None  # (leaves, wait): the batch the network is working on
    while True:
        busy = ahead[0] if ahead is not None else ()
        leaves = search.collect_leaves(half, busy) if search.sims_done < target else []
        answer = ahead[1]() if ahead is not None else None
        wait = engine.evaluate_start([leaf[2] for leaf in leaves]) if leaves else None
        if ahead is not None:
            search.apply_leaves(ahead[0], *answer)
        if leaves:
            ahead = (leaves, wait)
        elif ahead is not None:
            ahead = None  # nothing new (a collision, or the budget): look again
        else:
            break


class Search:
    """One MCTS tree rooted at a position."""

    __slots__ = (
        "root_state", "cfg", "rng", "n_nodes",
        "moves", "P", "N", "W", "child", "term_value", "solved", "solved_depth",
        "_pending_path", "_pending_state", "_pending_node",
        "sims_done", "root_noise_applied", "_batch",
        "sim_budget", "noise_off",
        "depth_reached", "child_depth",
    )

    def __init__(self, root_state: GameState, cfg: MCTSConfig, rng: np.random.Generator):
        self.root_state = root_state
        self.cfg = cfg
        self.rng = rng
        self.n_nodes = 0
        self.moves: list[np.ndarray] = []
        self.P: list[np.ndarray] = []
        self.N: list[np.ndarray] = []
        self.W: list[np.ndarray] = []
        self.child: list[np.ndarray] = []
        self.term_value: list[float | None] = []
        # Proven game-theoretic result, from the point of view of the side to
        # move at that node: WIN, LOSS, DRAW, or UNKNOWN.  Terminal positions
        # seed it and it propagates upwards (an MCTS-solver), which is what lets
        # a forced win read as exactly +1 instead of an average diluted by the
        # visits PUCT still spends on losing branches.
        self.solved: list[int] = []
        # Plies from a proven node until the game actually ends, under fastest
        # win / slowest loss.  Without it every lost move scores exactly -1 and
        # the agent picks among them arbitrarily -- which is what makes a
        # beaten engine look like it has stopped trying.
        self.solved_depth: list[int] = []
        # How far each root branch has actually been read, in plies below the
        # root, and the deepest any one descent has gone.  Recorded as the
        # descents happen: walking the tree to measure it afterwards would cost
        # more than the simulations it is measuring.  ``child_depth`` is one
        # entry per root child and exists only once the root is expanded.
        self.depth_reached = 0
        self.child_depth: np.ndarray | None = None
        self._pending_path: list[tuple[int, int]] = []
        self._pending_state: GameState | None = None
        self._pending_node: int = -1
        self.sims_done = 0
        self.root_noise_applied = False
        # Per-search overrides of the shared config.  Playout-cap
        # randomisation gives most moves a small budget and no root noise, and
        # the full budget to the few whose result is actually stored -- so the
        # budget cannot live in the config, which every game in flight shares.
        # ``None`` means "whatever the config says", i.e. the old behaviour.
        self.sim_budget: int | None = None
        self.noise_off = False
        self._batch: list[tuple[list, int, GameState]] = []
        self._new_node()  # node 0 = root, unexpanded

    @property
    def budget(self) -> int:
        """Simulations this particular search should run."""
        return self.cfg.simulations if self.sim_budget is None else self.sim_budget

    @property
    def wants_noise(self) -> bool:
        return self.cfg.add_noise and not self.noise_off

    # ----------------------------------------------------------------- nodes
    def _new_node(self) -> int:
        idx = self.n_nodes
        self.n_nodes += 1
        self.moves.append(None)
        self.P.append(None)
        self.N.append(None)
        self.W.append(None)
        self.child.append(None)
        self.term_value.append(None)
        self.solved.append(0)
        self.solved_depth.append(0)
        return idx

    def _note_depth(self, path: list[tuple[int, int]]) -> None:
        """Record how deep one descent went, under the root move it began with."""
        depth = len(path)
        if depth == 0:
            return
        if depth > self.depth_reached:
            self.depth_reached = depth
        if self.child_depth is not None:
            a = path[0][1]
            if self.child_depth[a] < depth:
                self.child_depth[a] = depth

    def is_expanded(self, node: int) -> bool:
        return self.moves[node] is not None

    # ------------------------------------------------------------ traversal
    def next_leaf(self) -> tuple[GameState, int] | None:
        """Descend to a leaf needing evaluation.

        Returns ``(state, node)``; terminal leaves are backed up internally and
        the descent restarts.  Returns ``None`` once ``simulations`` sims are
        done.
        """
        while self.sims_done < self.budget:
            state = self.root_state.copy()
            path: list[tuple[int, int]] = []
            node = 0
            while True:
                if self.term_value[node] is not None:
                    self._note_depth(path)
                    self._backup(path, self.term_value[node])
                    self._propagate_solved(path)
                    self.sims_done += 1
                    break
                if not self.is_expanded(node):
                    self._note_depth(path)
                    self._pending_path = path
                    self._pending_state = state
                    self._pending_node = node
                    return state, node
                a = self._select(node)
                move = int(self.moves[node][a])
                path.append((node, a))
                state.play(move)
                nxt = int(self.child[node][a])
                if nxt < 0:
                    nxt = self._new_node()
                    self.child[node][a] = nxt
                    if state.is_terminal():
                        # Value for the side to move at `state`: -1 when the
                        # move just played won, 0 when it filled the board.
                        tv = state.terminal_value()
                        self.term_value[nxt] = tv
                        self.solved[nxt] = DRAW if tv == 0.0 else LOSS
                        self.solved_depth[nxt] = 0
                node = nxt
        return None

    # --------------------------------------------------- batched traversal
    def next_leaf_batch(self, max_leaves: int) -> list[GameState]:
        """Collect up to ``max_leaves`` distinct leaves in one go.

        Each collected path carries a *virtual loss*: the edges along it are
        temporarily credited with a lost simulation, so the next descent is
        pushed towards a different branch instead of piling onto the same leaf.
        The loss is undone in :meth:`expand_batch`.  This is what lets a single
        game batch its network calls -- a batch of one wastes most of a CPU.
        """
        self._batch = self.collect_leaves(max_leaves)
        return [b[2] for b in self._batch]

    def collect_leaves(self, max_leaves: int, busy=()) -> list:
        """:meth:`next_leaf_batch` for a caller that keeps the batch itself.

        Returns ``(path, node, state)`` for each leaf, to hand back to
        :meth:`apply_leaves`.  ``busy`` is a batch still waiting for the
        network: its leaves count against the budget, and reaching one of them
        again is a collision, as reaching a leaf of this batch twice is.
        """
        batch: list[tuple[list, int, GameState]] = []
        pending_nodes: set[int] = {leaf[1] for leaf in busy}
        in_flight = len(busy)
        vl = 1.0
        while (
            len(batch) < max_leaves
            and self.sims_done + in_flight + len(batch) < self.budget
        ):
            state = self.root_state.copy()
            path: list[tuple[int, int]] = []
            node = 0
            hit_terminal = False
            while True:
                if self.term_value[node] is not None:
                    self._note_depth(path)
                    self._undo_virtual_loss(path, vl)
                    self._backup(path, self.term_value[node])
                    self._propagate_solved(path)
                    self.sims_done += 1
                    hit_terminal = True
                    break
                if not self.is_expanded(node):
                    break
                a = self._select(node)
                move = int(self.moves[node][a])
                path.append((node, a))
                self.N[node][a] += vl
                self.W[node][a] -= vl
                state.play(move)
                nxt = int(self.child[node][a])
                if nxt < 0:
                    nxt = self._new_node()
                    self.child[node][a] = nxt
                    if state.is_terminal():
                        tv = state.terminal_value()
                        self.term_value[nxt] = tv
                        self.solved[nxt] = DRAW if tv == 0.0 else LOSS
                        self.solved_depth[nxt] = 0
                node = nxt
            if hit_terminal:
                continue
            if node in pending_nodes:
                # Same leaf twice: stop here and evaluate what we have.
                self._undo_virtual_loss(path, vl)
                break
            pending_nodes.add(node)
            self._note_depth(path)
            batch.append((path, node, state))
        return batch

    def expand_batch(self, priors: np.ndarray, values: np.ndarray) -> None:
        self.apply_leaves(self._batch, priors, values)
        self._batch = []

    def apply_leaves(self, batch: list, priors: np.ndarray, values: np.ndarray) -> None:
        """Expand a batch from :meth:`collect_leaves` with the network's answers."""
        for i, (path, node, state) in enumerate(batch):
            self._undo_virtual_loss(path, 1.0)
            self._pending_path = path
            self._pending_state = state
            self._pending_node = node
            self.expand(node, priors[i], float(values[i]))

    def _undo_virtual_loss(self, path, vl: float) -> None:
        for node, a in path:
            self.N[node][a] -= vl
            self.W[node][a] += vl

    def _select(self, node: int) -> int:
        N = self.N[node]
        W = self.W[node]
        P = self.P[node]
        total = N.sum()
        sqrt_total = np.sqrt(total) if total > 0 else 1.0
        visited = N > 0
        # FPU: unvisited actions inherit the node's own value, reduced.
        parent_q = (W.sum() / total) if total > 0 else 0.0
        q = np.where(visited, W / np.maximum(N, 1.0), parent_q - self.cfg.fpu_reduction)
        u = self.cfg.c_puct * P * sqrt_total / (1.0 + N)
        score = q + u
        if node == 0 and self.cfg.forced_playouts > 0.0 and self.wants_noise and total > 0:
            # A root move below its guaranteed visit count outranks every move
            # that is not, whatever PUCT thinks; among those still short, PUCT
            # picks as usual rather than the choice falling to array order.
            short = (N < self._forced_visits(total)) & (P > 0)
            if short.any():
                score = np.where(short, score, -np.inf)
        return int(np.argmax(score))

    def _forced_visits(self, total: float) -> np.ndarray:
        """``sqrt(k * P * N)`` -- the visits each root move is guaranteed."""
        return np.sqrt(self.cfg.forced_playouts * self.P[0] * max(total, 0.0))

    def _pruned_visits(self, N: np.ndarray) -> np.ndarray:
        """Root visits with the forced ones taken back out.

        Every move keeps whatever it earned beyond its guarantee; a move that
        never got past the guarantee is dropped entirely, since one or two
        visits it was *made* to spend say nothing.  The most-visited move is
        left alone -- it is the search's answer, and pruning the thing the
        target is mostly about would be self-defeating.
        """
        counts = np.asarray(N, dtype=np.float64)
        best = int(np.argmax(counts))
        forced = self._forced_visits(float(counts.sum()))
        out = counts - np.minimum(forced, counts)
        out[out <= 1.0] = 0.0
        out[best] = counts[best]
        return out if out.sum() > 0 else counts

    def expand(self, node: int, priors: np.ndarray, value: float) -> None:
        """Attach network output to the pending leaf and back the value up."""
        state = self._pending_state
        legal = state.legal_moves()
        if len(legal) == 0:
            # A position with no moves that the rules did not call terminal.
            # Unreachable in either game; scored as a draw rather than trusting
            # the rules to be perfect.
            self.term_value[node] = 0.0
            self.solved[node] = DRAW
            self._backup(self._pending_path, 0.0)
            self.sims_done += 1
            self._pending_state = None
            return
        # `priors` are in the canonical frame; map them onto real board moves.
        canon_idx = np.array([state.to_canonical_move(int(m)) for m in legal], dtype=np.int64)
        p = priors[canon_idx].astype(np.float64)
        s = p.sum()
        p = p / s if s > 1e-12 else np.full(len(legal), 1.0 / max(len(legal), 1))

        self.moves[node] = legal
        self.P[node] = p.astype(np.float32)
        self.N[node] = np.zeros(len(legal), dtype=np.float32)
        self.W[node] = np.zeros(len(legal), dtype=np.float32)
        self.child[node] = np.full(len(legal), -1, dtype=np.int32)
        if node == 0:
            self.child_depth = np.zeros(len(legal), dtype=np.int32)

        if node == 0 and self.wants_noise and not self.root_noise_applied:
            self._apply_root_noise()

        self._backup(self._pending_path, value)
        self.sims_done += 1
        self._pending_state = None
        self._pending_node = -1

    def _apply_root_noise(self) -> None:
        k = len(self.P[0])
        if k <= 1:
            self.root_noise_applied = True
            return
        alpha = self.cfg.alpha_for(self.root_state.ncells)
        noise = self.rng.dirichlet(np.full(k, alpha)).astype(np.float32)
        eps = self.cfg.dirichlet_eps
        self.P[0] = (1 - eps) * self.P[0] + eps * noise
        self.root_noise_applied = True

    def _backup(self, path, value: float) -> None:
        # `value` is from the point of view of the side to move at the leaf.
        v = value
        g = self.cfg.value_discount
        for node, a in reversed(path):
            v = -v * g  # flip going up one ply, and decay with distance
            self.N[node][a] += 1.0
            self.W[node][a] += v

    # ------------------------------------------------------------- MCTS solver
    def _update_solved(self, node: int) -> bool:
        """Recompute one node's proven result from its children.

        A node is a proven win as soon as *one* child is a proven loss for the
        player to move there.  Anything else needs the full set of children to
        be known, and then the node takes the best available: a draw if any
        child is a proven draw, otherwise a loss.  Returns whether the flag
        changed.

        Also records the distance to the end of the game: the *shortest* proven
        win and the *longest* proven loss, so a decided position still has a
        best move.  The loss distance is exact -- a node is only proven lost
        once every child is known, so the maximum is over the full set.  The win
        distance is the shortest among the children proven *so far*, which a
        later, shorter proof will not revise; winning quickly matters much less
        than resisting for as long as possible, so that is left alone.
        """
        if self.solved[node] != UNKNOWN:
            return False
        kids = self.child[node]
        if kids is None:
            return False
        all_known = True
        soonest_win = -1
        latest_loss = -1
        soonest_draw = -1
        for j in range(len(kids)):
            c = int(kids[j])
            if c < 0 or self.solved[c] == UNKNOWN:
                all_known = False  # an unknown child can still hold anything
                continue
            depth = self.solved_depth[c]
            flag = self.solved[c]
            if flag == LOSS:  # opponent is lost after this move
                if soonest_win < 0 or depth < soonest_win:
                    soonest_win = depth
            elif flag == DRAW:
                if soonest_draw < 0 or depth < soonest_draw:
                    soonest_draw = depth
            elif depth > latest_loss:
                latest_loss = depth
        if soonest_win >= 0:
            self.solved[node] = WIN
            self.solved_depth[node] = soonest_win + 1
            return True
        if not all_known:
            return False
        if soonest_draw >= 0:
            # Every reply is known and none of them loses for the opponent, but
            # one of them is drawn: a draw is then the best this node can force.
            self.solved[node] = DRAW
            self.solved_depth[node] = soonest_draw + 1
            return True
        self.solved[node] = LOSS
        self.solved_depth[node] = latest_loss + 1
        return True

    def _propagate_solved(self, path) -> None:
        """Carry a freshly proven result up the path it was found on.

        If a node's flag does not change, no ancestor's can either -- the only
        thing that changed for the parent is this child -- so the walk stops.
        """
        for node, _ in reversed(path):
            if not self._update_solved(node):
                break

    # -------------------------------------------------------------- readout
    def root_visit_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        return self.moves[0], self.N[0]

    def root_value(self) -> float:
        N = self.N[0]
        total = N.sum()
        return float(self.W[0].sum() / total) if total > 0 else 0.0

    def root_score(self) -> float:
        """The value to *report*: exactly +-1 once the result is proven.

        ``root_value`` is a mean over every visit, so even a forced win reads
        below 1.0 -- PUCT keeps spending a slice of its budget on branches it
        has not refuted.  For a readout that is misleading, so a proven result
        overrides it.  Search itself still uses the average.
        """
        if self.solved[0] != UNKNOWN:
            return solved_value(self.solved[0])
        return self.root_value()

    def root_child_scores(self) -> np.ndarray:
        """Per-move value from the root player's view, exact where proven."""
        N, W = self.N[0], self.W[0]
        scores = np.divide(W, N, out=np.zeros_like(W, dtype=np.float64), where=N > 0)
        for j, c in enumerate(self.child[0]):
            c = int(c)
            if c >= 0 and self.solved[c] != UNKNOWN:
                # the child's view is the opponent's, so negate it
                scores[j] = -solved_value(self.solved[c])
        return scores

    def root_selection_scores(self) -> np.ndarray:
        """Ranking key for *choosing* a move, worst to best in three bands.

        Proven losses (-1, 0) < unproven [0, 1] < proven wins (2, 3].  Inside
        each band the tie is broken by distance: a proven win is better the
        sooner it lands, a proven loss is better the longer it takes.

        A proven *draw* is deliberately left in the unproven band, scored by its
        visit share.  It needs no special case to beat the proven losses (they
        are all negative, and a proven draw has been visited at least once), and
        forcing it above every unrefuted move would throw away won positions.

        The bands matter as much as the ordering inside them.  An unproven move
        might not lose; a proven loss certainly does.  And once every move is a
        proven loss, visit counts carry no information at all -- every branch
        was backed up as exactly -1 -- so without this the engine picks one at
        random, which is what "it stops trying when it's losing" actually is.
        """
        N = self.N[0].astype(np.float64)
        scores = N / max(N.sum(), 1.0)
        kids = self.child[0]
        if kids is None:
            return scores
        for j in range(len(kids)):
            c = int(kids[j])
            if c < 0 or self.solved[c] in (UNKNOWN, DRAW):
                continue
            depth = self.solved_depth[c]
            if self.solved[c] == LOSS:  # playing here wins
                scores[j] = 2.0 + 1.0 / (1.0 + depth)
            else:  # playing here loses; sell it as dearly as possible
                scores[j] = -1.0 + depth / (1e4 + depth)
        return scores

    def root_proven_moves(self) -> np.ndarray:
        """Indices of root moves that are a proven win for the side to move."""
        kids = self.child[0]
        if kids is None:
            return np.empty(0, dtype=np.int64)
        return np.array([j for j, c in enumerate(kids)
                         if int(c) >= 0 and self.solved[int(c)] == LOSS], dtype=np.int64)

    def policy_target(self, temperature: float,
                      prune: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(moves, probabilities) from root visit counts.

        ``prune`` undoes the forced playouts before reading the distribution
        off, which is the half of the technique that makes it worth having: the
        visits that were *guaranteed* rather than *earned* say nothing about how
        good the move is, so they are taken back out.  A move left with no
        earned visits at all is dropped.  Pass it for the training target and
        not for choosing a move -- the search should still play what it
        actually explored.
        """
        moves, N = self.root_visit_distribution()
        if prune and self.cfg.forced_playouts > 0.0 and self.wants_noise \
                and self.is_expanded(0) and len(N) > 1:
            N = self._pruned_visits(N)
        if temperature <= 1e-3:
            probs = np.zeros_like(N)
            probs[int(np.argmax(N))] = 1.0
            return moves, probs
        counts = np.power(N, 1.0 / temperature)
        s = counts.sum()
        if s <= 0:
            return moves, np.full(len(moves), 1.0 / len(moves), dtype=np.float32)
        return moves, (counts / s).astype(np.float32)

    def child_search(self, move: int) -> "Search | None":
        """Reuse the subtree after playing ``move`` (tree reuse between plies)."""
        if not self.is_expanded(0):
            return None
        idx = np.flatnonzero(self.moves[0] == move)
        if len(idx) == 0:
            return None
        node = int(self.child[0][idx[0]])
        if node < 0:
            return None
        new_state = self.root_state.copy()
        new_state.play(move)
        sub = Search.__new__(Search)
        sub.root_state = new_state
        sub.cfg = self.cfg
        sub.rng = self.rng
        sub._pending_path = []
        sub._pending_state = None
        sub._pending_node = -1
        sub.sims_done = 0
        sub.root_noise_applied = False
        sub.sim_budget = self.sim_budget
        sub.noise_off = self.noise_off
        sub._batch = []  # a reused subtree may still be driven in batched mode

        # Walk the reachable subtree.  Only *created* children are visited --
        # there are exactly (nodes - 1) of those in total, whereas every node
        # carries one slot per empty cell, so touching every slot here would
        # make re-rooting cost more than the search it saves.
        # The same walk also re-measures the depths, which the new root
        # inherits one ply shallower than it held them: `depths` is plies below
        # the new root, `branches` the slot in the new root's child array that
        # each node hangs under.
        mapping = {node: 0}
        order = [node]
        depths = [0]
        branches = [-1]
        i = 0
        while i < len(order):
            kids = self.child[order[i]]
            depth, branch = depths[i] + 1, branches[i]
            i += 1
            if kids is None:
                continue
            for slot in np.flatnonzero(kids >= 0).tolist():
                ch = int(kids[slot])
                if ch not in mapping:
                    mapping[ch] = len(order)
                    order.append(ch)
                    depths.append(depth)
                    branches.append(slot if branch < 0 else branch)

        # Renumber every child array at once, old index -> new index.
        remap = np.full(self.n_nodes, -1, dtype=np.int32)
        remap[np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))] = \
            np.fromiter(mapping.values(), dtype=np.int32, count=len(mapping))
        sub.moves, sub.P, sub.N, sub.W, sub.child, sub.term_value = [], [], [], [], [], []
        sub.solved = []
        sub.solved_depth = []
        for old in order:
            sub.moves.append(self.moves[old])
            sub.P.append(self.P[old])
            sub.N.append(self.N[old])
            sub.W.append(self.W[old])
            sub.term_value.append(self.term_value[old])
            sub.solved.append(self.solved[old])
            sub.solved_depth.append(self.solved_depth[old])
            kids = self.child[old]
            sub.child.append(None if kids is None else
                             np.where(kids >= 0, remap[np.maximum(kids, 0)], -1
                                      ).astype(np.int32))
        sub.n_nodes = len(sub.moves)
        sub.depth_reached = max(depths)
        sub.child_depth = None
        if sub.is_expanded(0):
            sub.child_depth = np.zeros(len(sub.moves[0]), dtype=np.int32)
            if len(order) > 1:
                np.maximum.at(sub.child_depth,
                              np.asarray(branches[1:], dtype=np.int64),
                              np.asarray(depths[1:], dtype=np.int32))
            # Inherited visits count towards this move's simulation budget.
            sub.sims_done = int(sub.N[0].sum())
            if sub.wants_noise:
                sub.P[0] = sub.P[0].copy()
                sub._apply_root_noise()
        return sub
