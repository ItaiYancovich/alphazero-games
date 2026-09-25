"""The base-game Splendor components: 90 development cards and 10 nobles.

Card statistics -- cost, bonus colour, prestige -- are the game's data rather
than anything this project invented, so the table below was transcribed from two
independent published datasets and cross-checked: they agree card for card, and
``tests/test_splendor.py`` re-asserts the structural invariants the printed game
has (40/30/20 cards per tier, eight/six/four per bonus colour, and the prestige
distribution of each tier).  Getting one cost wrong would not raise anything --
it would quietly train the network on a slightly different game -- so the
invariants are checked rather than trusted.

Gems are indexed in a fixed order everywhere in this package::

    0 white (diamond)   1 blue (sapphire)   2 green (emerald)
    3 red (ruby)        4 black (onyx)      5 gold (joker)

A cost is always a 5-tuple in that order; gold is never part of a cost, only of
a payment.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# ------------------------------------------------------------------ the gems
WHITE, BLUE, GREEN, RED, BLACK, GOLD = range(6)
GEMS = 5           # colours a card can cost or grant
TOKENS = 6         # ...plus gold, which only ever pays
GEM_NAMES = ("white", "blue", "green", "red", "black", "gold")
GEM_LETTERS = ("W", "U", "G", "R", "K", "*")
# What the cards are actually called, for anything a person reads.
GEM_GEMS = ("diamond", "sapphire", "emerald", "ruby", "onyx", "gold")

TIERS = 3


class Card(NamedTuple):
    """One development card.  ``cost`` is a 5-tuple in gem order."""

    tier: int
    gem: int            # the bonus it grants, a permanent discount of one
    points: int         # prestige
    cost: tuple[int, ...]


class Noble(NamedTuple):
    """One noble tile.  ``need`` counts *bonuses*, never tokens."""

    points: int
    need: tuple[int, ...]


# ``(tier, bonus gem, prestige, cost)``, tier 1 first, grouped by bonus colour.
_CARD_TABLE: tuple[tuple[int, int, int, tuple[int, ...]], ...] = (
    # ------------------------------------------- tier 1 (40 cards)
    # white bonus
    (0, 0, 0, (0, 0, 0, 2, 1)),
    (0, 0, 0, (0, 1, 1, 1, 1)),
    (0, 0, 0, (0, 1, 2, 1, 1)),
    (0, 0, 0, (0, 2, 0, 0, 2)),
    (0, 0, 0, (0, 2, 2, 0, 1)),
    (0, 0, 0, (0, 3, 0, 0, 0)),
    (0, 0, 0, (3, 1, 0, 0, 1)),
    (0, 0, 1, (0, 0, 4, 0, 0)),
    # blue bonus
    (0, 1, 0, (0, 0, 0, 0, 3)),
    (0, 1, 0, (0, 0, 2, 0, 2)),
    (0, 1, 0, (0, 1, 3, 1, 0)),
    (0, 1, 0, (1, 0, 0, 0, 2)),
    (0, 1, 0, (1, 0, 1, 1, 1)),
    (0, 1, 0, (1, 0, 1, 2, 1)),
    (0, 1, 0, (1, 0, 2, 2, 0)),
    (0, 1, 1, (0, 0, 0, 4, 0)),
    # green bonus
    (0, 2, 0, (0, 0, 0, 3, 0)),
    (0, 2, 0, (0, 1, 0, 2, 2)),
    (0, 2, 0, (0, 2, 0, 2, 0)),
    (0, 2, 0, (1, 1, 0, 1, 1)),
    (0, 2, 0, (1, 1, 0, 1, 2)),
    (0, 2, 0, (1, 3, 1, 0, 0)),
    (0, 2, 0, (2, 1, 0, 0, 0)),
    (0, 2, 1, (0, 0, 0, 0, 4)),
    # red bonus
    (0, 3, 0, (0, 2, 1, 0, 0)),
    (0, 3, 0, (1, 0, 0, 1, 3)),
    (0, 3, 0, (1, 1, 1, 0, 1)),
    (0, 3, 0, (2, 0, 0, 2, 0)),
    (0, 3, 0, (2, 0, 1, 0, 2)),
    (0, 3, 0, (2, 1, 1, 0, 1)),
    (0, 3, 0, (3, 0, 0, 0, 0)),
    (0, 3, 1, (4, 0, 0, 0, 0)),
    # black bonus
    (0, 4, 0, (0, 0, 1, 3, 1)),
    (0, 4, 0, (0, 0, 2, 1, 0)),
    (0, 4, 0, (0, 0, 3, 0, 0)),
    (0, 4, 0, (1, 1, 1, 1, 0)),
    (0, 4, 0, (1, 2, 1, 1, 0)),
    (0, 4, 0, (2, 0, 2, 0, 0)),
    (0, 4, 0, (2, 2, 0, 1, 0)),
    (0, 4, 1, (0, 4, 0, 0, 0)),
    # ------------------------------------------- tier 2 (30 cards)
    # white bonus
    (1, 0, 1, (0, 0, 3, 2, 2)),
    (1, 0, 1, (2, 3, 0, 3, 0)),
    (1, 0, 2, (0, 0, 0, 5, 0)),
    (1, 0, 2, (0, 0, 0, 5, 3)),
    (1, 0, 2, (0, 0, 1, 4, 2)),
    (1, 0, 3, (6, 0, 0, 0, 0)),
    # blue bonus
    (1, 1, 1, (0, 2, 2, 3, 0)),
    (1, 1, 1, (0, 2, 3, 0, 3)),
    (1, 1, 2, (0, 5, 0, 0, 0)),
    (1, 1, 2, (2, 0, 0, 1, 4)),
    (1, 1, 2, (5, 3, 0, 0, 0)),
    (1, 1, 3, (0, 6, 0, 0, 0)),
    # green bonus
    (1, 2, 1, (2, 3, 0, 0, 2)),
    (1, 2, 1, (3, 0, 2, 3, 0)),
    (1, 2, 2, (0, 0, 5, 0, 0)),
    (1, 2, 2, (0, 5, 3, 0, 0)),
    (1, 2, 2, (4, 2, 0, 0, 1)),
    (1, 2, 3, (0, 0, 6, 0, 0)),
    # red bonus
    (1, 3, 1, (0, 3, 0, 2, 3)),
    (1, 3, 1, (2, 0, 0, 2, 3)),
    (1, 3, 2, (0, 0, 0, 0, 5)),
    (1, 3, 2, (1, 4, 2, 0, 0)),
    (1, 3, 2, (3, 0, 0, 0, 5)),
    (1, 3, 3, (0, 0, 0, 6, 0)),
    # black bonus
    (1, 4, 1, (3, 0, 3, 0, 2)),
    (1, 4, 1, (3, 2, 2, 0, 0)),
    (1, 4, 2, (0, 0, 5, 3, 0)),
    (1, 4, 2, (0, 1, 4, 2, 0)),
    (1, 4, 2, (5, 0, 0, 0, 0)),
    (1, 4, 3, (0, 0, 0, 0, 6)),
    # ------------------------------------------- tier 3 (20 cards)
    # white bonus
    (2, 0, 3, (0, 3, 3, 5, 3)),
    (2, 0, 4, (0, 0, 0, 0, 7)),
    (2, 0, 4, (3, 0, 0, 3, 6)),
    (2, 0, 5, (3, 0, 0, 0, 7)),
    # blue bonus
    (2, 1, 3, (3, 0, 3, 3, 5)),
    (2, 1, 4, (6, 3, 0, 0, 3)),
    (2, 1, 4, (7, 0, 0, 0, 0)),
    (2, 1, 5, (7, 3, 0, 0, 0)),
    # green bonus
    (2, 2, 3, (5, 3, 0, 3, 3)),
    (2, 2, 4, (0, 7, 0, 0, 0)),
    (2, 2, 4, (3, 6, 3, 0, 0)),
    (2, 2, 5, (0, 7, 3, 0, 0)),
    # red bonus
    (2, 3, 3, (3, 5, 3, 0, 3)),
    (2, 3, 4, (0, 0, 7, 0, 0)),
    (2, 3, 4, (0, 3, 6, 3, 0)),
    (2, 3, 5, (0, 0, 7, 3, 0)),
    # black bonus
    (2, 4, 3, (3, 3, 5, 3, 0)),
    (2, 4, 4, (0, 0, 0, 7, 0)),
    (2, 4, 4, (0, 0, 3, 6, 3)),
    (2, 4, 5, (0, 0, 0, 7, 3)),
)

CARDS: tuple[Card, ...] = tuple(Card(t, g, p, c) for t, g, p, c in _CARD_TABLE)
NCARDS = len(CARDS)

# Card ids that make up each tier's deck, in table order.  A game shuffles a
# copy of these; the ids themselves are stable, which is what lets a position be
# stored, hashed and sent to the browser as small integers.
DECKS: tuple[tuple[int, ...], ...] = tuple(
    tuple(i for i, c in enumerate(CARDS) if c.tier == t) for t in range(TIERS)
)

# Cost, bonus and prestige as arrays, so scoring a whole row of cards is one
# numpy expression rather than ninety attribute lookups.
COST = np.array([c.cost for c in CARDS], dtype=np.int16)          # (90, 5)
BONUS = np.array([c.gem for c in CARDS], dtype=np.int8)           # (90,)
POINTS = np.array([c.points for c in CARDS], dtype=np.int16)      # (90,)
TIER_OF = np.array([c.tier for c in CARDS], dtype=np.int8)        # (90,)

# The ten nobles, every one worth 3 prestige: five want four bonuses of each of
# two colours, five want three of each of three.  Each colour is wanted by
# exactly two of the first kind and three of the second, so no colour is a
# better route to a noble than any other.
_NOBLE_TABLE: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 4, 4),  # red + black
    (0, 0, 4, 4, 0),  # green + red
    (0, 4, 4, 0, 0),  # blue + green
    (4, 4, 0, 0, 0),  # white + blue
    (4, 0, 0, 0, 4),  # white + black
    (0, 3, 3, 3, 0),  # blue + green + red
    (3, 3, 3, 0, 0),  # white + blue + green
    (3, 3, 0, 0, 3),  # white + blue + black
    (3, 0, 0, 3, 3),  # white + red + black
    (0, 0, 3, 3, 3),  # green + red + black
)

NOBLES: tuple[Noble, ...] = tuple(Noble(3, need) for need in _NOBLE_TABLE)
NNOBLES = len(NOBLES)
NOBLE_NEED = np.array([n.need for n in NOBLES], dtype=np.int16)   # (10, 5)


def card_label(card_id: int) -> str:
    """``"T2 green 2pt (3g,5b)"`` -- short enough for a move list."""
    card = CARDS[int(card_id)]
    parts = [f"{n}{GEM_LETTERS[g]}" for g, n in enumerate(card.cost) if n]
    price = "+".join(parts) if parts else "free"
    return (f"T{card.tier + 1} {GEM_NAMES[card.gem]} "
            f"{card.points}pt [{price}]")


def noble_label(noble_id: int) -> str:
    need = NOBLES[int(noble_id)].need
    parts = [f"{n}{GEM_LETTERS[g]}" for g, n in enumerate(need) if n]
    return f"Noble 3pt [{'+'.join(parts)}]"
