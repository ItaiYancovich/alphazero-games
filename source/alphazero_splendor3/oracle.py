"""How many turns a seat needs to reach fifteen, if nobody stopped it.

Splendor's value function is close to one quantity: *whose engine gets to
fifteen first*.  Everything else -- gems, bonuses, reserves, denial -- is
instrumental to it.  So this file computes that quantity directly, by playing
the position out as a **solitaire**: no opponent, no interference, buy the cards
that are actually on the table, collect exactly the tokens they cost, and count
the turns until the fifteenth point.

It is the Splendor analogue of a Go score, and it is used the way KataGo uses
one: as an **auxiliary training target**, not as an input feature.

That distinction is the whole design and it is worth stating plainly, because
the obvious thing to do is the wrong one.  A greedy solitaire costs on the order
of a hundred microseconds; a search evaluates a couple of hundred leaves a move,
so feeding it to the encoder would cost tens of milliseconds a move against a
game that currently takes about a second end to end.  It cannot be a feature.

As a *target* it is nearly free -- a self-play game stores about thirty-five
positions, not thirty-five thousand -- and it is strictly better placed there.
Fitting it forces the trunk to compute the planning quantity internally, from
the raw board, which is the thing you actually wanted the network to know.  A
feature would have let it read the answer off the input and learn nothing.

The estimate is a *relaxation*, not an optimum: greedy card choice, and one turn
of collecting buys three tokens (or two of one colour).  It does not need to be
exact -- it needs to be right on average about which seat is closer, and cheap
enough to run on every stored position.  It is not even monotone, which is worth
saying rather than assuming; see :func:`turns_to_win` for what is done about
that and ``tests/test_splendor3.py`` for what is actually claimed.
"""

from __future__ import annotations

from .game import (CGEM, CPAIRS, CPTS, GEMS, GOLD, Game, NPAIRS, TOKENS,
                   WIN_POINTS)

# What a solitaire that can never finish is worth.  Long enough that the network
# reads it as "not happening", short enough not to dominate the loss.
TURN_CAP = 30.0
# A bonus is not prestige, but it is what buys prestige later; without some
# credit for it the greedy choice never takes a cheap zero-point card, which is
# most of a real opening.
BONUS_CREDIT = 0.55
# The greedy is run once at each of these and the best answer kept; they
# disagree about how much a bare bonus is worth, which is where one pass errs.
CREDITS = (0.20, 0.55, 1.10)
NOBLE_POINTS = 3


def _cost_after(card: int, bonus: list[int], hand: list[int]) -> tuple[int, int]:
    """``(tokens still to collect, worst single colour)`` for one card."""
    short = 0
    worst = 0
    for colour, amount in CPAIRS[card]:
        gap = amount - bonus[colour] - hand[colour]
        if gap > 0:
            short += gap
            if gap > worst:
                worst = gap
    return short, worst


def _turns_for(short: int, worst: int, gold: int) -> int:
    """Turns of collecting before a card with this shortfall can be paid for.

    Two limits bind on a turn and the slower one wins: three tokens in all, and
    two of any one colour.  Gold already in hand covers part of the shortfall.
    """
    after = short - gold
    if after <= 0:
        return 0
    worst = min(worst, after)
    return max((after + 2) // 3, (worst + 1) // 2)


def turns_to_win(game: Game, seat: int, cap: float = TURN_CAP) -> float:
    """The best of a few greedy solitaires -- ``seat``'s turns to fifteen.

    A single greedy pass is not monotone: give a seat three more of every token
    and the ratio it maximises can point at a different, slower plan, so the
    estimate occasionally *rises* when the position has plainly improved.
    Running the greedy at several values of :data:`BONUS_CREDIT` and taking the
    minimum is a cheap partial fix -- the plans disagree about how much a bare
    bonus is worth, which is exactly where the single pass goes wrong -- and the
    minimum of several upper bounds is a better upper bound than any of them.

    It is still a relaxation, not an optimum.  ``tests/test_splendor3.py``
    asserts what is actually true of it: that it improves on average when the
    position improves, and never gets worse by more than a turn.
    """
    best = cap
    for credit in CREDITS:
        t = _greedy(game, seat, credit, cap)
        if t < best:
            best = t
            if best == 0.0:
                break
    return best


def _greedy(game: Game, seat: int, credit: float, cap: float) -> float:
    """One greedy solitaire: buy the best ratio of value to turns, repeat.

    Only cards the seat can actually reach are considered -- the twelve face up
    and its own reserves -- because a solitaire that may draw the whole deck is
    not measuring this position.
    """
    points = game.points[seat]
    if points >= WIN_POINTS:
        return 0.0

    bonus = game.bonus[seat][:]
    hand = game.hand[seat][:]
    nobles = list(game.nobles)

    # Candidates: the face-up board, plus this seat's own reserves.
    pool = [c for c in game.board if c >= 0]
    pool.extend(game.reserved[seat])
    if not pool:
        return cap

    turns = 0.0
    # A game cannot need more buys than points, and each buy is at least a turn.
    for _ in range(WIN_POINTS + 1):
        best = -1
        best_score = -1e30
        best_cost = 0
        for i, card in enumerate(pool):
            if card < 0:
                continue
            short, worst = _cost_after(card, bonus, hand)
            cost = _turns_for(short, worst, hand[GOLD]) + 1
            gain = CPTS[card] + credit
            gem = CGEM[card]
            # A bonus that completes a noble is worth three more points, and
            # the greedy choice has to see that or it never builds toward one.
            for noble in nobles:
                need = 0
                for colour, amount in NPAIRS[noble]:
                    have = bonus[colour] + (1 if colour == gem else 0)
                    if have < amount:
                        need += amount - have
                if need == 0:
                    gain += NOBLE_POINTS
                    break
            score = gain / cost
            if score > best_score:
                best_score = score
                best = i
                best_cost = cost
        if best < 0:
            return cap

        card = pool[best]
        pool[best] = -1
        short, worst = _cost_after(card, bonus, hand)
        collect = best_cost - 1
        turns += best_cost
        if turns >= cap:
            return cap

        # Pay for it: bonuses first, then tokens in hand, then the gold we hold,
        # then whatever the collecting turns brought in.
        gained = short
        for colour, amount in CPAIRS[card]:
            gap = amount - bonus[colour]
            if gap <= 0:
                continue
            have = hand[colour]
            if have >= gap:
                hand[colour] = have - gap
            else:
                hand[colour] = 0
                owed = gap - have
                take = min(owed, gained)
                gained -= take
                owed -= take
                if owed:
                    hand[GOLD] = max(0, hand[GOLD] - owed)
        # Collecting three tokens a turn usually overshoots what this card
        # needed; the surplus is real and the next card gets to use it.
        surplus = collect * 3 - short
        if surplus > 0:
            _spread(hand, surplus)

        bonus[gem] += 1
        points += CPTS[card]
        for noble in list(nobles):
            for colour, amount in NPAIRS[noble]:
                if bonus[colour] < amount:
                    break
            else:
                nobles.remove(noble)
                points += NOBLE_POINTS
        if points >= WIN_POINTS:
            return min(turns, cap)
    return cap


def _spread(hand: list[int], surplus: int) -> None:
    """Credit leftover collected tokens, capped at the ten a hand may hold."""
    room = 10 - sum(hand)
    if room <= 0:
        return
    surplus = min(surplus, room)
    i = 0
    while surplus > 0:
        hand[i % GEMS] += 1
        surplus -= 1
        i += 1


def race_margin(game: Game, seat: int) -> float:
    """``opponent turns - my turns``: positive when this seat is winning the race."""
    mine = turns_to_win(game, seat)
    best = TURN_CAP
    for other in range(1, game.players + 1):
        if other == seat:
            continue
        t = turns_to_win(game, other)
        if t < best:
            best = t
    return best - mine
