"""What a position is worth without a network -- the teacher the run starts from.

This is :func:`alphazero_splendor.heuristics.plan_value` -- the "reach"
evaluation the planner agent is built on, which prices a gem by how many turns
of collecting it saves on a card actually worth buying -- rewritten against the
v2 game.  Two things changed and both are about what it is for here.

**It is fast.**  The v1 version calls numpy once per card per seat per candidate
move; on this game's lists it is arithmetic on Python ints, which at five
elements is several times quicker.  That matters because it is no longer only an
opponent: it plays a few thousand games to give the network something to imitate
before self-play starts, and it is called once per legal move of every one of
them.

**It knows about denial and about the clock.**  Two terms are new: a card an
opponent can buy *this turn* is worth less to plan around than one they cannot,
and the race term is sharper near fifteen.

Neither of those turned out to be worth anything measurable, and that is worth
writing down rather than quietly leaving in.  Against the v1 planner over 80
duplicated games this agent scored 0.431 +/- 0.109 as it ships; with the denial
term switched off, 0.425; with the race term softened as well, 0.425; with v1's
bonus and discount weights restored on top of that, 0.456.  Every one of those
is inside every other one's interval, and inside 0.5.  So: the v2 evaluation is
**level with the v1 planner**, the new terms neither help nor hurt at a
measurable size, and the honest reading is that what this file changed is its
*speed* -- which is what it was rewritten for, since it has to play a few
thousand games to seed a network rather than merely be an opponent.

The weights are the obvious ones, not tuned against anything; the ladder is what
prices them.
"""

from __future__ import annotations

from .game import (CGEM, CPAIRS, CPTS, GEMS, GOLD, Game, NPAIRS, TOKEN_LIMIT,
                   WIN_POINTS)

W_PRESTIGE = 1.0
W_BONUS = 0.45          # a development card is worth about half a point of tempo
W_DISCOUNT = 0.22       # ...plus what its colour actually discounts
W_NOBLE = 3.0
W_GOLD = 0.11
W_HOARD = 0.05
W_OVERFLOW = 0.35       # tokens you will have to hand back
W_REACH = 0.62          # what a card within reach is worth, before distance
W_RACE = 0.34           # falling behind, once somebody is close to winning
W_DENIED = 0.55         # how much of a target survives an opponent taking it
REACH_SHARE = (1.0, 0.40, 0.18)
REACH_DECAY = 0.85


def colour_demand(game: Game) -> list[float]:
    """Per colour, what the cards worth buying actually cost in it.

    Weighted by what each card is worth, so the colours the *good* cards want
    count for more than the colours the cheap ones do.  This is what makes one
    bonus better than another: a bonus is a discount, and a discount is worth
    what it discounts.
    """
    total = [0.0] * GEMS
    weight = 0.0
    for card in game.board:
        if card < 0:
            continue
        w = CPTS[card] + 1.0
        weight += w
        for colour, amount in CPAIRS[card]:
            total[colour] += amount * w
    if weight <= 0:
        return total
    return [t / weight for t in total]


def turns_to_afford(game: Game, seat: int, card: int) -> int:
    """Roughly how many turns of collecting until ``seat`` can buy ``card``.

    Two limits bind on a turn and the slower one wins: at most three tokens in
    all, and at most two of any one colour.  Gold in hand counts against the
    total, since it pays for anything.
    """
    bonus = game.bonus[seat]
    hand = game.hand[seat]
    total = 0
    worst = 0
    for colour, amount in CPAIRS[card]:
        gap = amount - bonus[colour] - hand[colour]
        if gap > 0:
            total += gap
            if gap > worst:
                worst = gap
    total -= hand[GOLD]
    if total <= 0:
        return 0
    a = (total + 2) // 3
    b = (worst + 1) // 2
    return a if a > b else b


def noble_gain(game: Game, seat: int, gem: int) -> float:
    """What one more bonus of ``gem`` is worth towards the nobles still out.

    The last bonus a noble needs is worth the whole tile; the first is worth
    very little, because the tile may well go to somebody else.
    """
    bonus = game.bonus[seat]
    gain = 0.0
    for noble in game.nobles:
        short = 0
        wants = 0
        for colour, amount in NPAIRS[noble]:
            gap = amount - bonus[colour]
            if gap > 0:
                short += gap
                if colour == gem:
                    wants = 1
        if not wants or short <= 0:
            continue
        gain += W_NOBLE * (1.0 / short) ** 1.5
    return gain


def card_worth(game: Game, seat: int, card: int, demand: list[float],
               tempo: float) -> float:
    """What owning this card would be worth to ``seat``: prestige and engine."""
    gem = CGEM[card]
    return (W_PRESTIGE * CPTS[card]
            + tempo * (W_BONUS + W_DISCOUNT * demand[gem])
            + noble_gain(game, seat, gem))


def _seat_value(game: Game, seat: int, demand: list[float], tempo: float,
                leader: int, threatened: frozenset[int]) -> float:
    points = game.points[seat]
    hand = game.hand[seat]
    bonus = game.bonus[seat]

    value = float(W_PRESTIGE * points)
    for gem in range(GEMS):
        if bonus[gem]:
            value += tempo * (W_BONUS + W_DISCOUNT * demand[gem]) * bonus[gem]

    # What this seat is positioned to buy, discounted by how far off it is.
    # Only the best few targets count, and steeply: you buy one card a turn, so
    # being one turn from three good cards is not three times as good as one.
    scores = []
    for card in game.board:
        if card < 0:
            continue
        worth = card_worth(game, seat, card, demand, tempo)
        if card in threatened:
            # A card somebody else can take this turn is a target you may not
            # get; a reserved one, below, is a target nobody can take.
            worth *= W_DENIED
        scores.append(worth * REACH_DECAY ** turns_to_afford(game, seat, card))
    for card in game.reserved[seat]:
        scores.append(card_worth(game, seat, card, demand, tempo)
                      * REACH_DECAY ** turns_to_afford(game, seat, card))
    if scores:
        scores.sort(reverse=True)
        value += W_REACH * sum(s * w for s, w in zip(scores, REACH_SHARE))

    # Tokens are worth having but only faintly on their own: what they are for
    # is already counted by the reach term above.
    gems = 0
    for g in range(GEMS):
        gems += hand[g]
    value += W_HOARD * gems + W_GOLD * hand[GOLD]
    over = gems + hand[GOLD] - TOKEN_LIMIT
    if over > 0:
        value -= W_OVERFLOW * over

    # The race.  Being two points down matters little at 4-4 and a great deal
    # at 13-13, so the penalty scales with how far the leader has got.
    best_other = -99
    for s in range(1, game.players + 1):
        if s != seat and game.points[s] > best_other:
            best_other = game.points[s]
    if best_other > points:
        urgency = min(1.0, leader / WIN_POINTS)
        value -= W_RACE * urgency * (best_other - points)
    return value


def _threatened(game: Game, by_seat: int) -> frozenset[int]:
    """Face-up cards some seat other than ``by_seat`` could buy this turn."""
    out = set()
    for seat in range(1, game.players + 1):
        if seat == by_seat:
            continue
        gold = game.hand[seat][GOLD]
        for card in game.board:
            if card < 0 or card in out:
                continue
            if game.shortfall(seat, card) <= gold:
                out.add(card)
    return frozenset(out)


def plan_values(game: Game) -> list[float]:
    """Every seat's value at once, as a list indexed by seat (0 unused).

    A max^n search wants the whole vector at a leaf, and the expensive half of
    the evaluation -- what the visible cards cost, and how near the game is to
    over -- is the same for every seat.
    """
    out = [0.0] * (game.players + 1)
    if game.to_move == 0:
        result = game.result_vector()
        return [0.0] + [100.0 * float(result[s]) for s in range(1, game.players + 1)]
    demand = colour_demand(game)
    leader = max(game.points)
    # A bonus bought on the last turn discounts nothing, so the engine is worth
    # less the closer the game is to over.
    tempo = 0.30 + 0.70 * max(0.0, 1.0 - leader / WIN_POINTS)
    for seat in range(1, game.players + 1):
        out[seat] = _seat_value(game, seat, demand, tempo, leader,
                                _threatened(game, seat))
    return out


def plan_value(game: Game, seat: int) -> float:
    """What the position is worth to ``seat``, in points-equivalent."""
    if game.to_move == 0:
        return 100.0 * float(game.result_vector()[seat])
    demand = colour_demand(game)
    leader = max(game.points)
    tempo = 0.30 + 0.70 * max(0.0, 1.0 - leader / WIN_POINTS)
    return _seat_value(game, seat, demand, tempo, leader, _threatened(game, seat))
