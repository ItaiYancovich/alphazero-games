"""Input planes for the Intransitive network.

Twenty-eight planes, in the *canonical* frame -- the side to move is always
"own", with its base in the bottom-left corner and its advance up and to the
right::

     0  own rock
     1  own paper
     2  own scissors
     3  opponent rock
     4  opponent paper
     5  opponent scissors
     6  constant 1
     7  our base square           (constant: always a1)
     8  their base square         (constant: always i9)
     9  empty squares

    what is hanging right now:
    10  our pieces an adjacent enemy beats
    11  their pieces an adjacent piece of ours beats

    where anybody may go:
    12  squares one of our pieces may legally move to this turn
    13  squares one of theirs may legally move to, were it their turn

    what a square would do to a piece standing on it:
    14  our rock here would be capturable       (an enemy paper is adjacent)
    15  our paper here would be capturable      (an enemy scissors is adjacent)
    16  our scissors here would be capturable   (an enemy rock is adjacent)
    17  their rock here we could capture        (our paper is adjacent)
    18  their paper here we could capture       (our scissors is adjacent)
    19  their scissors here we could capture    (our rock is adjacent)

    geometry (constant; the same for every position):
    20  king-move distance to their base, over 8
    21  king-move distance to our base, over 8

    the two armies, each count broadcast over the whole board:
    22  how many rock we have, over 4
    23  how many paper we have, over 4
    24  how many scissors we have, over 4
    25  how many rock they have, over 4
    26  how many paper they have, over 4
    27  how many scissors they have, over 4

Planes 0-6 and 9 are the plain encoding, split by type because the type *is*
the piece: a rock and a scissors on the same square are not two values of one
quantity, they are opposite ends of a cycle, and a single "my piece here" plane
would ask the trunk to recover from a channel mixture the one fact every
capture depends on.

* **Planes 10-13 are the tactics a prior has to know.**  Search finds all of
  them one ply later, but the prior is what decides where a few hundred
  simulations go, and a prior that cannot see a hanging piece pays to
  rediscover it at every node.  Plane 12 is also where "we can reach their base
  *this move*" lives -- the single most important bit on the board, and one
  that a 3x3 convolution reads straight off the corner.

* **Planes 14-19 are the ones this game cannot do without, and they are why
  the piece planes are split by type.**  Every piece moves like a king, so
  choosing a move is choosing a *square to stand on*, and whether a square is
  safe depends entirely on which of the three things is standing there: a
  square flanked by an enemy paper is death for a rock, free for a scissors and
  irrelevant to a paper.  Each of the six is one dilation of one piece plane by
  the eight-neighbourhood -- so they cost almost nothing -- and together they
  turn "is this square safe" from a question about the neighbourhood's contents
  into a value the trunk can read at the square itself.

* **Planes 20-21 are geometry, and they are free.**  The whole game is a race
  towards a corner, and "how far is this square from the corner I am running
  at" is a constant the trunk would otherwise have to count out through the
  zero padding, spending depth on something a precomputed mask gives away.

* **Planes 22-27 are the matchup, said out loud.**  Who is winning is, more
  than anything else, a question about the two type mixtures -- three rocks
  are decisive against an army of scissors and dead weight against an army of
  paper -- and that is a *global count*, which is the one kind of thing a stack
  of 3x3 convolutions is worst at.  The global-pooling blocks can recover it,
  but only by spending channels and depth on arithmetic that six constants give
  away.  They are also what carries the plainest fact in the game, the one that
  needs no explanation at all: more pieces is better.

  They are inputs, not instructions.  Nothing here tells the network what to
  conclude from a count; it is free to learn that a rock deficit does not
  matter in a position where the race is already won, which is exactly the kind
  of judgement a hand-written evaluation gets wrong and self-play does not.

All twenty-eight are functions of the canonical board array alone, which is what
keeps self-play and training honest: a stored example is a single ``uint8``
grid, and a live position encodes identically.  They are also *equivariant*
under the game's one symmetry -- reflection in the a1-i9 diagonal, which fixes
both bases -- because every one of them is either a constant that the
reflection preserves or a dilation by the eight-neighbourhood, which the
reflection permutes among itself.

Repetition
----------
Planes 28 and 29 say "this exact position has occurred once before" and
"...twice before, so one more repeat is a draw".  They are constants over the
board, because a repetition is a property of the whole position rather than of
any square, and a constant is invariant under the reflection -- so the
augmentation carries them through untouched.

They are here because leaving them out was measured, and it was expensive.
Forty games of the iteration-66 network against itself at 128 simulations,
noise off:

    repetition   26  (65% of games)   -- every single draw
    base         14  (35% of games)
    stagnation    0

Two games in three ended by a rule the network had no way to perceive.  A
position on its third occurrence encoded byte-for-byte identically to its
first, so the value head read a win at a drawn position, the policy prior
pointed straight back into the loop, and only the search could save it -- and
only when the third occurrence fell inside its horizon.  It also meant
self-play handed the trainer one input carrying two different value targets, a
win in one game and a draw in another, on the majority of games.

What is deliberately *not* here
-------------------------------
The stagnation counter.  The game is drawn after 200 half-moves with no
capture, so two identical arrangements a hundred quiet plies apart are not the
same position -- and no plane says which one this is.  The same forty games
that indicted repetition exonerated this one: *zero* of them ended that way,
because the repetition rule fires long before the stagnation counter can.
Carrying it would mean a second field for a rule that no longer decides games.
If stagnation draws ever stop being nonexistent, this is the thing to revisit.
"""

from __future__ import annotations

import numpy as np

from .rps2_game import (BLUE_BASE, DIST_TO, FLIP_DIR, N, NCELLS, NDIRS, RED,
                        RED_BASE, BLUE)

PLANES = 30

# The planes from before repetition was representable.  A checkpoint trained
# against 28 of them still loads and still plays: ``planes_from_boards`` encodes
# either width, and the evaluator asks the network which one it wants.  That is
# what lets a running model be widened in place rather than retrained.
LEGACY_PLANES = 28

# The board has exactly one non-trivial symmetry.  Reflection in the a1-i9
# diagonal maps each base to itself, so a reflected position is a position that
# could really have arisen -- unlike a rotation, which swaps the two bases and
# is therefore the *player* swap the canonical frame already performs.  Two
# orientations rather than Ultimate Tic-Tac-Toe's eight; the price of a board
# with a distinguished pair of corners.
SYMMETRIES = 2

# Which type captures which: index by the attacker's type, get the victim's.
# rock (0) takes scissors (2), paper (1) takes rock (0), scissors (2) takes
# paper (1).
PREY = (2, 0, 1)
# ...and the reverse: index by a type, get the type that captures it.
PREDATOR = (1, 2, 0)

_r, _c = np.divmod(np.arange(NCELLS), N)
DIST_THEIRS = (np.array([DIST_TO[RED][s] for s in range(NCELLS)], dtype=np.float32)
               / (N - 1)).reshape(N, N)
DIST_OURS = (np.array([DIST_TO[BLUE][s] for s in range(NCELLS)], dtype=np.float32)
             / (N - 1)).reshape(N, N)
OUR_BASE_MASK = np.zeros((N, N), dtype=np.float32)
OUR_BASE_MASK.reshape(-1)[BLUE_BASE] = 1.0
THEIR_BASE_MASK = np.zeros((N, N), dtype=np.float32)
THEIR_BASE_MASK.reshape(-1)[RED_BASE] = 1.0


def _dilate(mask: np.ndarray) -> np.ndarray:
    """``(n, 9, 9)`` bool -> "one of the eight neighbours of this square is set".

    Eight shifts of a zero-padded copy.  The centre square itself is *not*
    included: the question every caller asks is about what stands next to a
    square, not on it.
    """
    n = mask.shape[0]
    pad = np.zeros((n, N + 2, N + 2), dtype=bool)
    pad[:, 1:-1, 1:-1] = mask
    out = np.zeros_like(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out |= pad[:, 1 + dr:1 + dr + N, 1 + dc:1 + dc + N]
    return out


def planes_from_boards(boards: np.ndarray, in_planes: int = PLANES,
                       reps: np.ndarray | None = None) -> np.ndarray:
    """``(n, 9, 9)`` canonical boards -> ``(n, planes, 9, 9)`` float32.

    ``boards`` uses the canonical encoding: 0 empty, 1-3 own rock/paper/
    scissors, 4-6 the opponent's.

    ``reps`` is how many times each position has occurred *before* this one --
    0, 1 or 2 -- and is the one thing here that the board itself cannot say.
    ``None`` means "no prior occurrence", which is both the common case and the
    only honest answer for a position restored from a buffer written before
    this existed.
    """
    if in_planes not in (PLANES, LEGACY_PLANES):
        raise ValueError(
            f"Intransitive uses {PLANES} input planes ({LEGACY_PLANES} for a "
            f"checkpoint from before the repetition planes), not {in_planes}")
    boards = np.asarray(boards, dtype=np.uint8)
    if boards.ndim != 3 or boards.shape[1:] != (N, N):
        raise ValueError(f"expected (n, {N}, {N}), got {boards.shape}")
    n = boards.shape[0]

    own = [boards == 1 + t for t in range(3)]
    opp = [boards == 4 + t for t in range(3)]
    empty = boards == 0

    # "An enemy of type t stands next to this square", and its mirror.  Six
    # dilations, and every tactical plane below is built out of them.
    opp_near = [_dilate(m) for m in opp]
    own_near = [_dilate(m) for m in own]

    out = np.zeros((n, in_planes, N, N), dtype=np.float32)
    for t in range(3):
        out[:, t] = own[t]
        out[:, 3 + t] = opp[t]
    out[:, 6] = 1.0
    out[:, 7] = OUR_BASE_MASK
    out[:, 8] = THEIR_BASE_MASK
    out[:, 9] = empty

    hanging = np.zeros((n, N, N), dtype=bool)
    winnable = np.zeros((n, N, N), dtype=bool)
    own_dest = np.zeros((n, N, N), dtype=bool)
    opp_dest = np.zeros((n, N, N), dtype=bool)
    for t in range(3):
        # Our type-t piece is attacked when the type that beats it is adjacent.
        hanging |= own[t] & opp_near[PREDATOR[t]]
        winnable |= opp[t] & own_near[PREDATOR[t]]
        # A type-t piece of ours may step onto any square it can reach that is
        # empty or holds the type it beats.
        own_dest |= own_near[t] & (empty | opp[PREY[t]])
        opp_dest |= opp_near[t] & (empty | own[PREY[t]])
    out[:, 10] = hanging
    out[:, 11] = winnable
    out[:, 12] = own_dest
    out[:, 13] = opp_dest

    for t in range(3):
        # Standing our type-t piece here loses it to whatever beats it.
        out[:, 14 + t] = opp_near[PREDATOR[t]]
        # ...and their type-t piece here is ours for the taking.
        out[:, 17 + t] = own_near[PREDATOR[t]]

    out[:, 20] = DIST_THEIRS
    out[:, 21] = DIST_OURS

    # The two armies as six counts, broadcast.  Divided by four -- the largest
    # any one type starts at -- so every plane lands in roughly 0..1 like the
    # masks around it.
    for t in range(3):
        out[:, 22 + t] = (own[t].sum(axis=(1, 2)) / 4.0)[:, None, None]
        out[:, 25 + t] = (opp[t].sum(axis=(1, 2)) / 4.0)[:, None, None]

    if in_planes > LEGACY_PLANES:
        # Two thresholds rather than one count, so "one more repeat ends the
        # game" is a plane of its own rather than a value the trunk has to
        # compare against a constant it would also have to learn.
        r = (np.zeros(n, dtype=np.int64) if reps is None
             else np.asarray(reps, dtype=np.int64).reshape(n))
        out[:, 28] = (r >= 1).astype(np.float32)[:, None, None]
        out[:, 29] = (r >= 2).astype(np.float32)[:, None, None]
    return out


def transform_boards(a: np.ndarray, sym: int) -> np.ndarray:
    """The game's symmetry, applied to the last two axes of a 9x9 grid.

    ``sym`` is 0 (identity) or 1 (reflection in the a1-i9 diagonal).  The
    reflection sends square ``(r, c)`` to ``(8 - c, 8 - r)``, which fixes both
    base corners -- so a reflected position is one the game could really have
    reached, and both players' geometry is preserved rather than swapped.
    """
    a = np.asarray(a)
    if not sym:
        return np.ascontiguousarray(a)
    return np.ascontiguousarray(np.swapaxes(a, -2, -1)[..., ::-1, ::-1])


def transform_policy(pi: np.ndarray, sym: int) -> np.ndarray:
    """The same symmetry, applied to a ``(648,)`` policy vector.

    A policy here is not a picture of the board: it is eight boards, one per
    direction, and a reflection moves the *directions* as well as the squares.
    So the direction axis is permuted by ``FLIP_DIR`` -- which is its own
    inverse -- and each of the eight planes is reflected like any other grid.
    """
    pi = np.asarray(pi)
    if not sym:
        return np.ascontiguousarray(pi)
    grid = pi.reshape(NDIRS, N, N)
    moved = transform_boards(grid[list(FLIP_DIR)], 1)
    return np.ascontiguousarray(moved.reshape(-1))
