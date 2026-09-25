"""Hand-written Splendor knowledge: what a position is worth, without search.

This is the game's counterpart of Hex's two-distance and Connect Four's
four-in-a-row windows -- the "rule-based expert" rung of the ladder, and the
thing the trained network has to beat before any of its numbers mean anything.

The evaluation is what a person is taught in their first few games:

* **Prestige is the point**, and nothing else scores.
* **A bonus is a permanent discount**, so it is worth more the more of the
  visible board it makes cheaper, and worth more early than late -- a bonus
  bought on the last turn discounts nothing.
* **Nobles are three points for bonuses you wanted anyway**, but only if you
  actually finish one, so partial progress is worth much less than the last
  bonus that completes it.  Hence the square.
* **Tokens are potential**, worth something, worth less as they pile up, and
  worth nothing at all above the ten you are allowed to keep.
* **Gold is flexible** and worth a little more than a coloured token.

The weights are not tuned against anything -- they are the obvious ones, and
the ladder prices what they are worth.
"""

from __future__ import annotations

import numpy as np

from .cards import BONUS, COST, GEMS, GOLD, NOBLE_NEED, POINTS
from .splendor_game import TOKEN_LIMIT, WIN_POINTS, SplendorGame

# What each term is worth, in units of "one prestige point".
W_PRESTIGE = 1.0
W_BONUS = 0.42          # a development card is worth roughly half a point of tempo
W_DISCOUNT = 0.05       # ...plus a little per visible card it makes cheaper
W_NOBLE = 3.0           # a noble really is three points
W_TOKEN = 0.06
W_GOLD = 0.10
W_OVERFLOW = 0.30       # tokens you will have to hand back
# A reserved card is an *option*, and the two terms below are what that means.
# It is worth a little, and more when the card is worth more -- but nothing like
# a share of its prestige, because you have not bought it and may never.  And it
# costs one of three hand slots, which is why the penalty grows with the square
# of how many are gone: the third slot is far dearer than the first, since it is
# the one that will not be there for the tier-3 card you actually wanted.
#
# These used to read ``W_RESERVED * (1 + points)`` with no slot cost at all,
# which priced a reserved two-point card at 0.30 against 0.18 for taking three
# gems -- so the agent opened every single game by reserving three times, which
# is not naive play but a degenerate reading of its own weights.
W_RESERVED = 0.10
W_SLOT = 0.06


def bonus_usefulness(game: SplendorGame) -> np.ndarray:
    """Per colour, how much of the visible board a bonus of it would discount."""
    visible = [int(c) for c in game.board if c >= 0]
    if not visible:
        return np.zeros(GEMS, dtype=np.float64)
    return (COST[visible] > 0).sum(axis=0).astype(np.float64) / len(visible)


def noble_progress(game: SplendorGame, seat: int) -> float:
    """Value of how close ``seat`` is to each remaining noble.

    Squared, because three quarters of a noble is worth nothing: the tile goes
    to whoever finishes it.  A noble already out of reach for want of a colour
    still counts what has been earned towards it, since those bonuses were not
    wasted -- they are just not this tile.
    """
    bonus = game.bonus[seat]
    total = 0.0
    for noble in game.nobles:
        need = NOBLE_NEED[noble]
        wanted = int(need.sum())
        if wanted == 0:
            continue
        have = int(np.minimum(bonus, need).sum())
        total += W_NOBLE * (have / wanted) ** 2
    return total


def position_value(game: SplendorGame, seat: int) -> float:
    """What the position is worth to ``seat``, in points-equivalent.

    Absolute, not relative: the agent below compares afterstates of *its own*
    moves, where the opponents are unchanged, so subtracting them would cancel.
    """
    if game.is_terminal():
        # Once it is over the only thing that matters is the result.
        return 100.0 * float(game.result_vector()[seat])

    points = float(game.points[seat])
    bonus = game.bonus[seat]
    useful = bonus_usefulness(game)

    # A bonus is worth less once the game is nearly over: it has fewer turns
    # left in which to discount anything.
    leader = float(game.points.max())
    remaining = max(0.0, 1.0 - leader / WIN_POINTS)
    tempo = 0.35 + 0.65 * remaining

    value = W_PRESTIGE * points
    value += tempo * float((W_BONUS + W_DISCOUNT * useful * 12.0) @ bonus)
    value += noble_progress(game, seat)

    held = game.hand[seat]
    gems = float(held[:GEMS].sum())
    value += W_TOKEN * gems + W_GOLD * float(held[GOLD])
    over = float(held.sum()) - TOKEN_LIMIT
    if over > 0:
        value -= W_OVERFLOW * over

    slots = len(game.reserved[seat])
    for card in game.reserved[seat]:
        value += W_RESERVED * (0.3 + 0.2 * float(POINTS[card]))
    value -= W_SLOT * slots * slots / 2.0
    return value


# ---------------------------------------------------------------------------
# A second, stronger evaluation, for the planner.
#
# ``position_value`` above is the beginner's account of the game and it has one
# large blind spot: it prices a token at 0.06 whatever colour it is.  An agent
# maximising it therefore takes essentially arbitrary gems, which in Splendor is
# most of what a turn is.  What is missing is **reach** -- how close the tokens
# in hand bring a card actually worth buying -- and that is what the terms below
# add.  ``position_value`` is left exactly as it was: it is a rated opponent,
# and quietly changing what it does would invalidate the ladder it sits on.
# ---------------------------------------------------------------------------

W_REACH = 0.62          # what a card within reach is worth, before distance
W_RACE = 0.30           # falling behind, once somebody is close to winning
W_HOARD = 0.05          # tokens doing nothing in particular
REACH_SHARE = (1.0, 0.40, 0.18)   # the best three targets, and how much each counts
REACH_DECAY = 0.85      # how fast a card's worth falls off with turns to afford it


def colour_demand(game: SplendorGame) -> np.ndarray:
    """Per colour, how much the cards worth buying actually cost in it.

    Weighted by what each card is worth, so the colours the *good* cards want
    count for more than the colours the cheap ones do.  This is what makes one
    bonus better than another: it is a discount, and a discount is worth what it
    discounts.
    """
    visible = [int(c) for c in game.board if c >= 0]
    if not visible:
        return np.zeros(GEMS, dtype=np.float64)
    weight = (POINTS[visible] + 1.0).astype(np.float64)
    return (COST[visible].astype(np.float64) * weight[:, None]).sum(0) / weight.sum()


def turns_to_afford(game: SplendorGame, seat: int, card_id: int) -> int:
    """Roughly how many turns of taking gems until ``seat`` can buy ``card_id``.

    Two limits bind and the slower one wins: a turn yields at most three tokens
    in all, and at most two of any one colour.  So needing five of one colour is
    three turns however many other colours are already covered.  Gold in hand
    counts against the total, since it pays for anything.
    """
    need = np.maximum(COST[card_id] - game.bonus[seat], 0)
    missing = np.maximum(need - game.hand[seat, :GEMS], 0)
    total = int(missing.sum()) - int(game.hand[seat, GOLD])
    if total <= 0:
        return 0
    worst = int(missing.max())
    return max((total + 2) // 3, (worst + 1) // 2)


def noble_gain(game: SplendorGame, seat: int, gem: int) -> float:
    """What one more bonus of ``gem`` is worth towards the nobles still out."""
    bonus = game.bonus[seat]
    gain = 0.0
    for noble in game.nobles:
        need = NOBLE_NEED[noble]
        if not need[gem] or bonus[gem] >= need[gem]:
            continue          # this noble does not want it, or has enough
        short = int(np.maximum(need - bonus, 0).sum())
        if short <= 0:
            continue
        # The last bonus a noble needs is worth the whole tile; the first is
        # worth very little, because the tile may well go to somebody else.
        gain += W_NOBLE * (1.0 / short) ** 1.5
    return gain


def card_worth(game: SplendorGame, seat: int, card_id: int,
               demand: np.ndarray, tempo: float) -> float:
    """What owning this card would be worth to ``seat``, prestige and engine."""
    gem = int(BONUS[card_id])
    return (W_PRESTIGE * float(POINTS[card_id])
            + tempo * (W_BONUS + W_DISCOUNT * float(demand[gem]) * 4.0)
            + noble_gain(game, seat, gem))


def reach_value(game: SplendorGame, seat: int, demand: np.ndarray,
                tempo: float) -> float:
    """What ``seat`` is positioned to buy, discounted by how far off it is.

    Only the best few targets count, and steeply: a player buys one card a turn,
    so being one turn from three good cards is not three times being one turn
    from one.  This is the term ``position_value`` has no analogue for, and the
    reason it cannot tell a useful gem from a useless one.
    """
    scores = []
    for card in list(game.board) + game.reserved[seat]:
        card = int(card)
        if card < 0:
            continue
        gap = turns_to_afford(game, seat, card)
        scores.append(card_worth(game, seat, card, demand, tempo)
                      * REACH_DECAY ** gap)
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    return W_REACH * sum(s * w for s, w in zip(scores, REACH_SHARE))


def _seat_value(game: SplendorGame, seat: int, demand: np.ndarray, tempo: float,
                leader: float) -> float:
    points = float(game.points[seat])
    hand = game.hand[seat]

    value = W_PRESTIGE * points
    value += tempo * float((W_BONUS + W_DISCOUNT * demand * 4.0) @ game.bonus[seat])
    value += reach_value(game, seat, demand, tempo)

    # Tokens are worth having, but only faintly on their own: what they are
    # actually for is already counted by the reach term above.
    value += W_HOARD * float(hand[:GEMS].sum()) + W_GOLD * float(hand[GOLD])
    over = float(hand.sum()) - TOKEN_LIMIT
    if over > 0:
        value -= W_OVERFLOW * over

    # The race.  Being two points down matters little at 4-4 and a great deal
    # at 13-13, so the penalty scales with how far the leader has got.
    others = [float(game.points[s]) for s in range(1, game.players + 1) if s != seat]
    if others:
        urgency = min(1.0, leader / WIN_POINTS)
        value -= W_RACE * urgency * max(0.0, max(others) - points)
    return value


def plan_values(game: SplendorGame) -> np.ndarray:
    """Every seat's value at once, as ``(players + 1,)`` with index 0 unused.

    A max^n search needs the whole vector at a leaf, and the expensive half of
    the evaluation -- what the visible cards cost, and how near the game is to
    over -- is the same for every seat.  Computing it once is most of the reason
    the search is affordable at all.
    """
    out = np.zeros(game.players + 1, dtype=np.float64)
    if game.is_terminal():
        return 100.0 * game.result_vector().astype(np.float64)
    demand = colour_demand(game)
    leader = float(game.points.max())
    # A bonus bought on the last turn discounts nothing, so the engine is worth
    # less the closer the game is to over.
    tempo = 0.30 + 0.70 * max(0.0, 1.0 - leader / WIN_POINTS)
    for seat in range(1, game.players + 1):
        out[seat] = _seat_value(game, seat, demand, tempo, leader)
    return out


def plan_value(game: SplendorGame, seat: int) -> float:
    """What the position is worth to ``seat`` -- the planner's evaluation.

    Unlike :func:`position_value` this is meant to be compared across positions
    in which the *opponents* have moved too, because a search visits those.  So
    it carries a race term: what matters near the end is not how good your
    engine is but whether you get to fifteen first.
    """
    if game.is_terminal():
        return 100.0 * float(game.result_vector()[seat])
    demand = colour_demand(game)
    leader = float(game.points.max())
    tempo = 0.30 + 0.70 * max(0.0, 1.0 - leader / WIN_POINTS)
    return _seat_value(game, seat, demand, tempo, leader)
