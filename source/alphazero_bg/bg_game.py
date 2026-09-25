"""Backgammon rules engine.

Board geometry
--------------
Twenty-four points, indexed 0..23 in a fixed absolute frame, plus a bar and a
tray for each player::

    WHITE moves 0 -> 23, home is 18..23, bears off past 24, enters on die-1
    BLACK moves 23 -> 0, home is 0..5,   bears off past -1, enters on 24-die

BLACK moves first, so that "the first player" means the same thing here as in
the other two games and the shared arena code can stay as it is.

What makes this game different
------------------------------
Hex and Connect Four are deterministic games where a move is a cell.  Neither is
true here, and both differences reach all the way up into the GUI:

* **Dice.**  A position is only half the state; the roll is the other half.  A
  turn therefore has two phases -- the roll, which nobody chooses, and the move,
  which is chosen knowing the roll.  ``Backgammon`` stores the pending roll, and
  ``roll_dice`` advances from the first phase to the second.
* **A move is a sequence.**  Two hops, or four on doubles, each of which may hit
  a blot or bear a checker off.  The rules force you to play as many dice as you
  can, and to play the larger one if only one will go, so legality is a property
  of the whole sequence rather than of its parts.

:meth:`legal_moves` returns whole sequences, deduplicated by the position they
produce: two orderings of the same two hops are one choice, not two, and an
agent that treated them as two would spread its probability mass over a
distinction that does not exist.

Scoring
-------
A game is worth 1 point normally, 2 for a gammon (the loser has borne off
nothing) and 3 for a backgammon (the loser has borne off nothing *and* still has
a checker on the bar or in the winner's home board).  Values are reported on the
usual -1..+1 scale, so a plain win is 1/3.

There is no doubling cube.  It is a large addition -- a second decision problem
with its own equity theory -- and everything here works without it.
"""

from __future__ import annotations

import itertools

import numpy as np

BLACK = 1   # moves 23 -> 0, home 0..5, moves first
WHITE = 2   # moves 0 -> 23, home 18..23

POINTS = 24
CHECKERS = 15
BAR = -1     # the "from" of an entering move, and the "to" of a hit
OFF = -2     # the "to" of a bearing-off move

# Every distinct unordered pair of dice, with the 1/36 weight of each.  Used by
# the two-ply evaluation and by anything that has to average over the roll.
ROLLS: tuple[tuple[int, int, float], ...] = tuple(
    (a, b, (1.0 if a == b else 2.0) / 36.0)
    for a in range(1, 7) for b in range(a, 7)
)


def other(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def direction(player: int) -> int:
    """+1 if the player's points increase as it moves, -1 if they decrease."""
    return 1 if player == WHITE else -1


class Backgammon:
    """A mutable backgammon position, including whose turn it is and the roll.

    ``points`` is a length-24 int array: positive counts are BLACK checkers,
    negative are WHITE.  One array rather than two is worth it here because
    almost every rule question is "who owns this point and how many", and the
    sign answers both at once.
    """

    __slots__ = ("points", "bar", "off", "to_move", "dice", "winner", "move_count",
                 "_legal_cache")

    def __init__(self, setup: bool = True):
        self.points = np.zeros(POINTS, dtype=np.int8)
        self.bar = {BLACK: 0, WHITE: 0}
        self.off = {BLACK: 0, WHITE: 0}
        self.to_move = BLACK
        self.dice: tuple[int, ...] = ()   # the roll waiting to be played
        self.winner = 0
        self.move_count = 0               # completed turns, not hops
        self._legal_cache: list | None = None
        if setup:
            self.reset()

    def reset(self) -> None:
        """The standard opening position."""
        self.points[:] = 0
        # BLACK (positive) runs 23 -> 0; WHITE (negative) is its mirror image.
        for index, count in ((23, 2), (12, 5), (7, 3), (5, 5)):
            self.points[index] = count
            self.points[POINTS - 1 - index] = -count
        self.bar = {BLACK: 0, WHITE: 0}
        self.off = {BLACK: 0, WHITE: 0}
        self.to_move = BLACK
        self.dice = ()
        self.winner = 0
        self.move_count = 0
        self._legal_cache = None

    # ---------------------------------------------------------------- basics
    def copy(self) -> "Backgammon":
        new = Backgammon.__new__(Backgammon)
        new.points = self.points.copy()
        new.bar = dict(self.bar)
        new.off = dict(self.off)
        new.to_move = self.to_move
        new.dice = self.dice
        new.winner = self.winner
        new.move_count = self.move_count
        new._legal_cache = None
        return new

    def count_on(self, point: int, player: int) -> int:
        """How many of ``player``'s checkers sit on ``point`` (0 if none)."""
        value = int(self.points[point])
        if player == BLACK:
            return value if value > 0 else 0
        return -value if value < 0 else 0

    def blocked(self, point: int, player: int) -> bool:
        """Is ``point`` closed to ``player`` -- two or more enemy checkers?"""
        return self.count_on(point, other(player)) >= 2

    def home_range(self, player: int) -> range:
        return range(0, 6) if player == BLACK else range(18, 24)

    def all_home(self, player: int) -> bool:
        """Can ``player`` start bearing off?  Every checker home, none on bar."""
        if self.bar[player]:
            return False
        inside = sum(self.count_on(p, player) for p in self.home_range(player))
        return inside + self.off[player] == CHECKERS

    def pip_count(self, player: int) -> int:
        """Pips still to travel.  The oldest and bluntest measure of who leads."""
        total = 0
        for point in range(POINTS):
            n = self.count_on(point, player)
            if n:
                total += n * (point + 1 if player == BLACK else POINTS - point)
        total += self.bar[player] * 25
        return total

    # ------------------------------------------------------------------ dice
    def roll_dice(self, rng: np.random.Generator) -> tuple[int, ...]:
        """Roll for the side to move.  Doubles are played four times."""
        a, b = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        return self.set_dice(a, b)

    def set_dice(self, a: int, b: int) -> tuple[int, ...]:
        self.dice = (a,) * 4 if a == b else (max(a, b), min(a, b))
        self._legal_cache = None
        return self.dice

    def needs_roll(self) -> bool:
        return not self.dice and not self.winner

    # ------------------------------------------------------------ move rules
    def _entry_point(self, player: int, die: int) -> int:
        """Where a checker coming off the bar lands for this die."""
        return 24 - die if player == BLACK else die - 1

    def _bear_off_distance(self, point: int, player: int) -> int:
        """The die that bears a checker off ``point`` exactly."""
        return point + 1 if player == BLACK else POINTS - point

    def _hops_for(self, die: int, player: int) -> list[tuple[int, int]]:
        """Every single legal hop with one die, as ``(from, to)``.

        ``from`` is :data:`BAR` for an entering checker and ``to`` is
        :data:`OFF` for one being borne off.
        """
        hops: list[tuple[int, int]] = []
        if self.bar[player]:
            # Nothing else may move while a checker waits on the bar.
            entry = self._entry_point(player, die)
            if not self.blocked(entry, player):
                hops.append((BAR, entry))
            return hops

        step = direction(player) * die
        for point in range(POINTS):
            if not self.count_on(point, player):
                continue
            target = point + step
            if 0 <= target < POINTS:
                if not self.blocked(target, player):
                    hops.append((point, target))
                continue
            # Off the end of the board: only legal while bearing off.
            if not self.all_home(player):
                continue
            exact = self._bear_off_distance(point, player)
            if die == exact:
                hops.append((point, OFF))
            elif die > exact and not self._checker_further_back(point, player):
                # A bigger die bears off only from the furthest checker back.
                hops.append((point, OFF))
        return hops

    def _checker_further_back(self, point: int, player: int) -> bool:
        """Is any of ``player``'s checkers further from home than ``point``?"""
        if player == BLACK:
            return any(self.count_on(p, BLACK) for p in range(point + 1, 6))
        return any(self.count_on(p, WHITE) for p in range(18, point))

    def apply_hop(self, hop: tuple[int, int]) -> None:
        """Play one hop for the side to move, hitting a blot if there is one."""
        player = self.to_move
        source, target = hop
        if source == BAR:
            self.bar[player] -= 1
        else:
            # One checker off the source point, in whichever sign owns it.
            self.points[source] += -1 if player == BLACK else 1
        if target == OFF:
            self.off[player] += 1
            if self.off[player] == CHECKERS:
                self.winner = player
        else:
            if self.count_on(target, other(player)) == 1:
                # A lone enemy checker is hit and goes back to the bar.
                self.points[target] = 0
                self.bar[other(player)] += 1
            self.points[target] += 1 if player == BLACK else -1
        self._legal_cache = None

    # ------------------------------------------------------- move generation
    def legal_moves(self) -> list[tuple[tuple[int, int], ...]]:
        """Every legal full move for the current roll, as a tuple of hops.

        Deduplicated by the resulting position: the same two hops in either
        order are one move.  The empty tuple is returned -- as the only option --
        when the roll cannot be played at all, which is a legal and fairly
        common outcome.
        """
        if self._legal_cache is not None:
            return self._legal_cache
        if self.winner or not self.dice:
            self._legal_cache = []
            return self._legal_cache

        sequences: dict[bytes, tuple[tuple[int, int], ...]] = {}
        best = 0
        if len(self.dice) == 4:  # doubles: one die, played up to four times
            best = self._search_sequences(self.dice, (), sequences, best)
        else:
            # Both orders, because one may allow a hop the other forbids.
            for order in {self.dice, tuple(reversed(self.dice))}:
                best = self._search_sequences(order, (), sequences, best)

        moves = [seq for seq in sequences.values() if len(seq) == best]
        if best == 1 and len(self.dice) == 2:
            # Only one die can be played: the rules say it must be the larger
            # one if that is possible at all.
            larger = max(self.dice)
            with_larger = [m for m in moves if self._hop_die(m[0]) == larger]
            if with_larger:
                moves = with_larger
        # ``sequences`` is already keyed by the position each move reaches, so
        # what survives the length filter is distinct by construction.  Replaying
        # the moves to re-derive those keys was the single most expensive thing
        # this engine did.
        self._legal_cache = moves if moves else [()]
        return self._legal_cache

    def _search_sequences(self, dice: tuple[int, ...], played: tuple,
                          out: dict, best: int) -> int:
        """Depth-first over the dice, recording every reachable sequence."""
        if not dice:
            return best
        die, rest = dice[0], dice[1:]
        moved = False
        for hop in self._hops_for(die, self.to_move):
            moved = True
            child = self.copy()
            child.apply_hop(hop)
            sequence = played + (hop,)
            key = child.signature()
            if len(sequence) > len(out.get(key, ())):
                out[key] = sequence
            best = max(best, len(sequence))
            if not child.winner:
                best = child._search_sequences(rest, sequence, out, best)
        if not moved and rest:
            # This die is unplayable; try the next one on its own.
            best = self._search_sequences(rest, played, out, best)
        return best

    def hop_die(self, hop: tuple[int, int]) -> int:
        """Which die a hop consumes -- public, because the GUI validates with it."""
        return self._hop_die(hop)

    def _hop_die(self, hop: tuple[int, int]) -> int:
        """Which die a hop consumed, so the larger-die rule can be applied."""
        source, target = hop
        player = self.to_move
        if source == BAR:
            return 24 - target if player == BLACK else target + 1
        if target == OFF:
            return self._bear_off_distance(source, player)
        return abs(target - source)

    def after(self, move: tuple[tuple[int, int], ...]) -> "Backgammon":
        """The position this move leads to, with the turn *not* yet passed."""
        state = self.copy()
        for hop in move:
            state.apply_hop(hop)
        return state

    def after_turn(self, move: tuple[tuple[int, int], ...]) -> "Backgammon":
        """The position after ``move``, with the turn passed and no validation.

        The validating :meth:`play` regenerates the legal move list to check its
        argument, which is fine once per turn but quadratic when an evaluator is
        walking every candidate -- and the evaluator got those candidates from
        that very list, so there is nothing left to check.
        """
        state = self.copy()
        for hop in move:
            state.apply_hop(hop)
        state.move_count += 1
        if not state.winner:
            state.to_move = other(state.to_move)
        state.dice = ()
        state._legal_cache = None
        return state

    def play(self, move: tuple[tuple[int, int], ...]) -> None:
        """Play a full move and hand the turn over."""
        if self.winner:
            raise ValueError("game already decided")
        legal = self.legal_moves()
        if tuple(move) not in {tuple(m) for m in legal}:
            raise ValueError(f"illegal move {move!r} for dice {self.dice}")
        for hop in move:
            self.apply_hop(hop)
        self.move_count += 1
        if not self.winner:
            self.to_move = other(self.to_move)
        self.dice = ()
        self._legal_cache = None

    # ---------------------------------------------------------------- result
    def is_terminal(self) -> bool:
        return self.winner != 0

    def points_won(self) -> int:
        """1 for a win, 2 for a gammon, 3 for a backgammon; 0 while running."""
        if not self.winner:
            return 0
        loser = other(self.winner)
        if self.off[loser] > 0:
            return 1
        # Nothing borne off: a gammon at least.  A checker still on the bar or
        # in the winner's home board makes it a backgammon.
        if self.bar[loser] or any(self.count_on(p, loser)
                                  for p in self.home_range(self.winner)):
            return 3
        return 2

    def result_for(self, player: int) -> float:
        """Signed points on a -1..+1 scale: a plain win is 1/3, a gammon 2/3."""
        if not self.winner:
            return 0.0
        magnitude = self.points_won() / 3.0
        return magnitude if self.winner == player else -magnitude

    def terminal_value(self) -> float:
        """Value of a finished position for the side to move.

        Note the sign, which differs from Hex and Connect Four: those pass the
        turn even on the move that wins, so their terminal positions belong to
        the loser and are worth -1.  Here the turn stays with whoever bore off
        the last checker -- the game is over and nobody moves next -- so this is
        *positive* for a win.  Both conventions are self-consistent; what would
        not be is assuming one while the engine implements the other.
        """
        return self.result_for(self.to_move)

    # ------------------------------------------------------- representations
    def signature(self) -> bytes:
        """A hashable identity for the position (ignoring the pending roll)."""
        return (self.points.tobytes()
                + bytes((self.bar[BLACK], self.bar[WHITE],
                         self.off[BLACK], self.off[WHITE], self.to_move)))

    def canonical(self) -> "Backgammon":
        """The position as the side to move sees it, always as BLACK.

        Mirroring the board costs one array reversal and halves what the
        network has to learn -- the same trick the other two games use, and the
        reason the encoder never sees a WHITE-to-move position.
        """
        if self.to_move == BLACK:
            return self
        flipped = Backgammon.__new__(Backgammon)
        flipped.points = -self.points[::-1].copy()
        flipped.bar = {BLACK: self.bar[WHITE], WHITE: self.bar[BLACK]}
        flipped.off = {BLACK: self.off[WHITE], WHITE: self.off[BLACK]}
        flipped.to_move = BLACK
        flipped.dice = self.dice
        flipped.winner = 0 if not self.winner else (
            BLACK if self.winner == WHITE else WHITE)
        flipped.move_count = self.move_count
        flipped._legal_cache = None
        return flipped

    def __str__(self) -> str:
        def cell(point: int) -> str:
            value = int(self.points[point])
            if value == 0:
                return " . "
            return f"{abs(value):2d}{'B' if value > 0 else 'W'}"

        top = " ".join(cell(p) for p in range(12, 24))
        bottom = " ".join(cell(p) for p in range(11, -1, -1))
        return (f"13..24: {top}\n"
                f"12.. 1: {bottom}\n"
                f"bar B{self.bar[BLACK]} W{self.bar[WHITE]}   "
                f"off B{self.off[BLACK]} W{self.off[WHITE]}   "
                f"to move: {'BLACK' if self.to_move == BLACK else 'WHITE'}"
                + (f"   dice {self.dice}" if self.dice else ""))


def move_to_str(move: tuple[tuple[int, int], ...], player: int = BLACK) -> str:
    """Standard-ish notation: ``8/5 6/5``, ``bar/22``, ``3/off``."""
    if not move:
        return "(no play)"

    def name(index: int) -> str:
        if index == BAR:
            return "bar"
        if index == OFF:
            return "off"
        # Points are always named from the mover's own side, 24 down to 1.
        return str(index + 1 if player == BLACK else POINTS - index)

    return " ".join(f"{name(a)}/{name(b)}" for a, b in move)
