"""Input encoding for the backgammon network.

Two encodings live here, and a checkpoint records which one it was trained with.

``basic`` (196 inputs) is the classic TD-Gammon representation: four units per
point per player::

    unit 0   at least one checker
    unit 1   at least two          (the point is made)
    unit 2   at least three
    unit 3   (n - 3) / 2           however many beyond three

The odd shape is deliberate.  The first three checkers on a point mean three
different things -- a blot, a made point, a spare -- while the fourth and fifth
mean much the same as each other, so they share one graded unit.  Plus the bar
and the tray for each side: 24 x 4 x 2 + 4 = 196.

``rich`` (220 inputs) adds twelve summary numbers per player on top.  This is the
step that took TD-Gammon from a respectable intermediate player to a strong one,
and the reason is worth stating: everything in ``basic`` is *local*.  Whether a
blot can be hit depends on where every enemy checker is; whether the game is
still a contact game depends on whether the two sides have passed each other.
A dense net can in principle derive those from the raw board, but it has to
learn them from scratch, from a signal that is one noisy number per game -- and
the first run's evidence is that it does not: twelve hours of training past
iteration 500 moved the reference match by less than its own error bar.

Both encodings canonicalise first, so the network only ever sees positions with
the side to move as BLACK.  Everything is scaled into roughly 0..1, because a
plain MLP with no normalisation layers is sensitive to input scale in a way a
residual conv tower is not.
"""

from __future__ import annotations

import numpy as np

from .bg_game import BLACK, CHECKERS, POINTS, WHITE, Backgammon

UNITS_PER_POINT = 4
BOARD_INPUTS = POINTS * UNITS_PER_POINT * 2 + 4
SUMMARY_PER_PLAYER = 12
RICH_INPUTS = BOARD_INPUTS + 2 * SUMMARY_PER_PLAYER

INPUTS = BOARD_INPUTS          # what "basic" means, and the default
KINDS = {"basic": BOARD_INPUTS, "rich": RICH_INPUTS}

# How often a blot this many pips away is hit on an open board.  Direct shots
# (1-6) are common and the odds fall away sharply past them; the exact numbers
# depend on what is in between, but the shape is what matters to the network.
_HIT_CHANCE = {1: 11 / 36, 2: 12 / 36, 3: 14 / 36, 4: 15 / 36, 5: 15 / 36,
               6: 17 / 36, 7: 6 / 36, 8: 6 / 36, 9: 5 / 36, 10: 3 / 36,
               11: 2 / 36, 12: 3 / 36}


def size_of(kind: str) -> int:
    try:
        return KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown feature set {kind!r}; expected one of {sorted(KINDS)}")


def _write_point(out: np.ndarray, base: int, count: int) -> None:
    if count <= 0:
        return
    out[base] = 1.0
    if count >= 2:
        out[base + 1] = 1.0
    if count >= 3:
        out[base + 2] = 1.0
    if count > 3:
        out[base + 3] = (count - 3) / 2.0


def encode(board: Backgammon, kind: str = "basic") -> np.ndarray:
    """One position as float32, from the point of view of the side to move."""
    view = board.canonical()
    out = np.zeros(size_of(kind), dtype=np.float32)
    points = view.points
    for point in range(POINTS):
        value = int(points[point])
        if value > 0:
            _write_point(out, point * UNITS_PER_POINT, value)
        elif value < 0:
            _write_point(out, (POINTS + point) * UNITS_PER_POINT, -value)
    tail = POINTS * UNITS_PER_POINT * 2
    out[tail + 0] = view.bar[BLACK] / 2.0
    out[tail + 1] = view.bar[WHITE] / 2.0
    out[tail + 2] = view.off[BLACK] / CHECKERS
    out[tail + 3] = view.off[WHITE] / CHECKERS
    if kind == "rich":
        _write_summary(out, BOARD_INPUTS, view, BLACK)
        _write_summary(out, BOARD_INPUTS + SUMMARY_PER_PLAYER, view, WHITE)
    return out


def _write_summary(out: np.ndarray, base: int, view: Backgammon, player: int) -> None:
    """Twelve numbers about ``player``'s position as a whole.

    All of them are things a backgammon player reads off the board in a glance
    and none of them is local to one point, which is exactly why they are worth
    handing over rather than making the network rediscover them.
    """
    opponent = WHITE if player == BLACK else BLACK
    mine_black = player == BLACK

    blots = made = home_points = outside = back_in_enemy = 0
    exposure = 0.0
    furthest_back = -1          # in pips from home: bigger is further away
    for point in range(POINTS):
        raw = int(view.points[point])
        count = raw if mine_black else -raw
        if count <= 0:
            continue
        distance = point + 1 if mine_black else POINTS - point
        if distance > furthest_back:
            furthest_back = distance
        if count == 1:
            blots += 1
            exposure += _blot_exposure(view, point, player)
        else:
            made += 1
            if distance <= 6:
                home_points += 1
        if distance > 6:
            outside += count
        if distance >= 19:      # still in the opponent's home board
            back_in_enemy += count
    if view.bar[player]:
        furthest_back = 25
        outside += view.bar[player]

    out[base + 0] = view.pip_count(player) / 167.0
    out[base + 1] = view.off[player] / CHECKERS
    out[base + 2] = min(view.bar[player], 3) / 3.0
    out[base + 3] = min(blots, 5) / 5.0
    out[base + 4] = min(exposure, 2.0) / 2.0
    out[base + 5] = made / 8.0
    out[base + 6] = home_points / 6.0
    out[base + 7] = max(furthest_back, 0) / 25.0
    out[base + 8] = min(back_in_enemy, 5) / 5.0
    out[base + 9] = 1.0 if view.all_home(player) else 0.0
    out[base + 10] = outside / CHECKERS
    # Contact: can the two sides still reach each other at all?  Once they
    # cannot, the position is a pure race and nothing about blots matters.
    out[base + 11] = 0.0 if _contact(view, player, opponent) else 1.0


def _blot_exposure(view: Backgammon, point: int, player: int) -> float:
    """Roughly how likely the blot on ``point`` is to be hit next roll."""
    opponent = WHITE if player == BLACK else BLACK
    if view.bar[opponent]:
        entry = 24 if opponent == BLACK else -1
        return _HIT_CHANCE.get(abs(entry - point), 0.0)
    # The opponent moves towards this point from behind it, in its own sense.
    candidates = (range(point + 1, POINTS) if opponent == BLACK
                  else range(point - 1, -1, -1))
    for other_point in candidates:
        raw = int(view.points[other_point])
        if (raw > 0) if opponent == BLACK else (raw < 0):
            return _HIT_CHANCE.get(abs(other_point - point), 1 / 36)
    return 0.0


def _contact(view: Backgammon, player: int, opponent: int) -> bool:
    """True while either side can still be hit by the other."""
    if view.bar[player] or view.bar[opponent]:
        return True
    mine_black = player == BLACK
    my_back = None
    their_back = None
    for point in range(POINTS):
        raw = int(view.points[point])
        if raw == 0:
            continue
        if (raw > 0) == mine_black:
            distance = point + 1 if mine_black else POINTS - point
            if my_back is None or distance > my_back:
                my_back = distance
                my_back_point = point
        else:
            distance = POINTS - point if mine_black else point + 1
            if their_back is None or distance > their_back:
                their_back = distance
                their_back_point = point
    if my_back is None or their_back is None:
        return False
    # Contact remains while my furthest-back checker has not passed theirs.
    return (my_back_point > their_back_point) if mine_black \
        else (my_back_point < their_back_point)


def encode_batch(boards, kind: str = "basic") -> np.ndarray:
    """``(N, inputs) float32`` for a list of positions."""
    out = np.empty((len(boards), size_of(kind)), dtype=np.float32)
    for i, board in enumerate(boards):
        out[i] = encode(board, kind)
    return out
