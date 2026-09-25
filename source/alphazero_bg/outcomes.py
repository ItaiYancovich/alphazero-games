"""The five outcome probabilities, and the equity they imply.

A backgammon game does not end in "win" or "lose".  It ends in one of six
results, and the three magnitudes on each side are decided by different things:
whether you win at all is mostly the race, whether you win a *gammon* is whether
the loser got any checker home, and a *backgammon* needs one of theirs still
trapped.  Asking a single number to carry all of that is asking it to average
three different questions -- and the network then cannot tell you that a position
is a 55% win but a 30% gammon, which is exactly the distinction that decides how
to play it.

So the network predicts five probabilities, from the point of view of the side
to move, in the representation every strong backgammon program uses::

    0  P(win at all)
    1  P(win a gammon or better)
    2  P(win a backgammon)
    3  P(lose a gammon or better)
    4  P(lose a backgammon)

They are *nested*, not exclusive: winning a backgammon is also winning a gammon
is also winning.  That is why they are five independent sigmoids rather than a
softmax over six classes -- each one answers a yes/no question about the same
game, and the nesting falls out of the data rather than being imposed.

Losing at all needs no output of its own: it is ``1 - P(win)``.

Equity in points is then the sum of what each outcome is worth::

    E = P(w) - (1 - P(w)) + P(wg) + P(wbg) - P(lg) - P(lbg)
      = 2*P(w) - 1 + P(wg) + P(wbg) - P(lg) - P(lbg)

which this module reports divided by three, so a plain win is 1/3 and a
backgammon 1, matching the scale the rest of the package uses.
"""

from __future__ import annotations

import numpy as np

OUTPUTS = 5
WIN, WIN_G, WIN_BG, LOSE_G, LOSE_BG = range(OUTPUTS)


def outcome_vector(points: int, won: bool) -> np.ndarray:
    """What actually happened, as the five indicators, for one player.

    ``points`` is 1, 2 or 3; ``won`` says which side of the result this is.
    """
    v = np.zeros(OUTPUTS, dtype=np.float32)
    if won:
        v[WIN] = 1.0
        if points >= 2:
            v[WIN_G] = 1.0
        if points >= 3:
            v[WIN_BG] = 1.0
    else:
        if points >= 2:
            v[LOSE_G] = 1.0
        if points >= 3:
            v[LOSE_BG] = 1.0
    return v


def flip(v: np.ndarray) -> np.ndarray:
    """The same position seen from the other side of the board.

    Not a negation -- a *permutation*.  My chance of winning a gammon is your
    chance of losing one, and my chance of winning at all is one minus yours.
    Getting this backwards would train the network on its own mirror image,
    which is the kind of bug that produces a player that is merely mediocre
    rather than one that is obviously broken.
    """
    out = np.empty_like(v)
    out[..., WIN] = 1.0 - v[..., WIN]
    out[..., WIN_G] = v[..., LOSE_G]
    out[..., WIN_BG] = v[..., LOSE_BG]
    out[..., LOSE_G] = v[..., WIN_G]
    out[..., LOSE_BG] = v[..., WIN_BG]
    return out


def equity(v: np.ndarray) -> np.ndarray | float:
    """Points per game on the -1..+1 scale, from the five probabilities."""
    points = (2.0 * v[..., WIN] - 1.0
              + v[..., WIN_G] + v[..., WIN_BG]
              - v[..., LOSE_G] - v[..., LOSE_BG])
    return points / 3.0


def terminal_vector(board, player: int) -> np.ndarray:
    """The exact outcome of a finished game, for ``player``."""
    return outcome_vector(board.points_won(), board.winner == player)
