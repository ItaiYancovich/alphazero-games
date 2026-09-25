"""The input encoding: one Splendor position as a fixed-length vector.

Everything is **seat-relative**.  The player to move is always described first
and the others follow in turn order, so the network only ever learns one point
of view -- the same trick as Hex's canonical board, which is what halves the
self-play a given strength needs.  Seat slots past the table's size are zeroed
and flagged inactive, so a two-player and a four-player position are the same
length and one set of weights serves both.

**Nothing here reads a card the mover has not been shown.**  That is a property
of the encoding rather than a check somewhere: deck *order* never appears (only
each deck's size), and another seat's reserved card is written out in full only
when it was reserved off the face-up board, where everyone saw it.  A card taken
off the top of a deck is encoded as "a card of this tier, contents unknown" --
which is exactly what the other players know.  So a network fed this vector
cannot cheat even if the position handed to it is the server's own.

The encoding is a little wider than the bare state, and deliberately.  Three
groups of features are *derived* rather than raw:

* **Affordability.**  Whether a card can be bought right now, and how many
  tokens short of it its buyer is.  This is a comparison between a card's cost,
  a player's bonuses and their tokens -- three places in the vector -- and the
  first thing any human reads off the board.  A plain MLP can learn it, but it
  is the input to nearly every decision in the game, so it is spelled out.  The
  same reasoning gave the Hex network its connectivity planes.  Board cards are
  scored for the player to move; a reserved card is scored for whoever holds it,
  because there the interesting question is what *they* are about to do.
* **Nobles as a distance.**  How many bonuses short of each noble the mover is,
  rather than only the requirement.
* **The endgame clock.**  Whether the fifteen-point trigger has fired, which
  changes what a turn is worth and is not visible in the point totals alone.

Counts are scaled into roughly ``[0, 1]`` by the game's own maxima rather than
standardised, so the meaning of a feature does not drift with the data.
"""

from __future__ import annotations

import numpy as np

from .cards import (BONUS, CARDS, COST, GEMS, GOLD, NOBLE_NEED, POINTS, TIERS,
                    TOKENS, card_label)
from .splendor_game import (MAX_RESERVED, MAX_SEATS, PHASE_MAIN, ROW,
                            SplendorGame, TOKEN_LIMIT, WIN_POINTS)

NAME = "v1"

# --- one card: present, unknown, tier, prestige, bonus, cost, and two derived
CARD_WIDTH = 1 + 1 + TIERS + 1 + GEMS + GEMS + 1 + 1          # 18
# --- one seat: active, tokens, bonuses, prestige, reserved count, cards owned,
#     and its three reserved slots
SEAT_BASE = 1 + TOKENS + GEMS + 1 + 1 + 1                     # 15
SEAT_WIDTH = SEAT_BASE + MAX_RESERVED * CARD_WIDTH            # 69
NOBLE_WIDTH = 1 + GEMS + 1                                    # present, need, distance
BOARD_SLOTS = TIERS * ROW
NOBLE_SLOTS = MAX_SEATS + 1                 # a four-player table lays out five

# Scales.  Chosen from the game's own maxima so a feature means the same thing
# in every position, rather than from the data, which would drift as the network
# changes what positions occur.
TOKEN_SCALE = 7.0        # the largest pile, at four players
BONUS_SCALE = 8.0        # nobody needs more than this many of one colour
POINT_SCALE = float(WIN_POINTS)
CARD_POINT_SCALE = 5.0
COST_SCALE = 7.0
DECK_SCALE = 40.0
CARDS_SCALE = 20.0
SHORTFALL_CAP = 10.0
PLY_SCALE = 100.0

# --- the layout, as offsets into the vector -------------------------------
O_BANK = 0
O_SEATS = O_BANK + TOKENS
O_BOARD = O_SEATS + MAX_SEATS * SEAT_WIDTH
O_NOBLES = O_BOARD + BOARD_SLOTS * CARD_WIDTH
O_DECKS = O_NOBLES + NOBLE_SLOTS * NOBLE_WIDTH
O_PHASE = O_DECKS + TIERS
O_FINAL = O_PHASE + 3
O_PLAYERS = O_FINAL + 1
O_PLY = O_PLAYERS + 3
INPUTS = O_PLY + 1


def size_of(name: str = NAME) -> int:
    if name != NAME:
        raise ValueError(f"unknown Splendor feature set {name!r}")
    return INPUTS


def _card_block(out: np.ndarray, at: int, card_id: int, known: bool,
                tier: int, bonus: np.ndarray, hand: np.ndarray) -> None:
    """Write one card slot.

    ``known`` is False for a card reserved off a deck by somebody else: its tier
    is public, its identity is not, and the block says exactly that much.
    ``bonus`` and ``hand`` belong to the player this card's affordability is
    being judged for.
    """
    if card_id < 0 and tier < 0:
        return                      # an empty slot: everything stays zero
    out[at] = 1.0
    out[at + 2 + max(tier, 0)] = 1.0
    if not known:
        out[at + 1] = 1.0           # present, tier known, contents not
        return
    out[at + 2 + TIERS] = POINTS[card_id] / CARD_POINT_SCALE
    out[at + 3 + TIERS + int(BONUS[card_id])] = 1.0
    cost = COST[card_id]
    base = at + 3 + TIERS + GEMS
    out[base: base + GEMS] = cost / COST_SCALE
    need = np.maximum(cost - bonus, 0)
    short = int(np.maximum(need - hand[:GEMS], 0).sum())
    gold = int(hand[GOLD])
    out[base + GEMS] = 1.0 if short <= gold else 0.0
    # How far out of reach, in tokens still to find.  Capped, so a hopeless
    # tier-3 card and a slightly hopeless one do not look equally far away.
    out[base + GEMS + 1] = min(max(short - gold, 0), SHORTFALL_CAP) / SHORTFALL_CAP


def encode_one(game: SplendorGame, out: np.ndarray) -> None:
    """Write ``game`` into ``out`` (length :data:`INPUTS`), seat-relative."""
    out[:] = 0.0
    players = game.players
    # A finished position still has to encode -- the GUI asks for an evaluation
    # of it -- and has no player to move, so seat 1 stands in.
    mover = game.to_move if game.to_move else 1

    out[O_BANK:O_BANK + TOKENS] = game.bank / TOKEN_SCALE

    mover_bonus = game.bonus[mover]
    mover_hand = game.hand[mover]

    for k in range(MAX_SEATS):
        if k >= players:
            continue
        base = O_SEATS + k * SEAT_WIDTH
        seat = (mover - 1 + k) % players + 1
        out[base] = 1.0
        out[base + 1: base + 1 + TOKENS] = game.hand[seat] / TOKEN_SCALE
        out[base + 1 + TOKENS: base + 1 + TOKENS + GEMS] = game.bonus[seat] / BONUS_SCALE
        out[base + 1 + TOKENS + GEMS] = game.points[seat] / POINT_SCALE
        out[base + 2 + TOKENS + GEMS] = len(game.reserved[seat]) / MAX_RESERVED
        out[base + 3 + TOKENS + GEMS] = game.ncards[seat] / CARDS_SCALE
        # A reserved card is judged for the seat holding it: what matters about
        # an opponent's reserve is whether *they* can pay for it.
        held = game.reserved[seat]
        secret = game.hidden[seat]
        for i in range(MAX_RESERVED):
            at = base + SEAT_BASE + i * CARD_WIDTH
            if i >= len(held):
                continue
            card = int(held[i])
            known = (seat == mover) or not secret[i]
            _card_block(out, at, card if known else -1, known, CARDS[card].tier,
                        game.bonus[seat], game.hand[seat])

    for slot in range(BOARD_SLOTS):
        card = int(game.board[slot])
        _card_block(out, O_BOARD + slot * CARD_WIDTH, card, True,
                    CARDS[card].tier if card >= 0 else -1,
                    mover_bonus, mover_hand)

    for i in range(NOBLE_SLOTS):
        if i >= len(game.nobles):
            continue
        base = O_NOBLES + i * NOBLE_WIDTH
        need = NOBLE_NEED[game.nobles[i]]
        out[base] = 1.0
        out[base + 1: base + 1 + GEMS] = need / 4.0
        out[base + 1 + GEMS] = float(np.maximum(need - mover_bonus, 0).sum()) / 12.0

    out[O_DECKS:O_DECKS + TIERS] = [len(d) / DECK_SCALE for d in game.decks]
    out[O_PHASE + game.phase] = 1.0
    out[O_FINAL] = 1.0 if game.triggered else 0.0
    out[O_PLAYERS + players - 2] = 1.0
    out[O_PLY] = min(game.move_count, 400) / PLY_SCALE


def encode(games) -> np.ndarray:
    """A batch of positions as ``(N, INPUTS) float32`` -- the evaluator's encoder."""
    out = np.zeros((len(games), INPUTS), dtype=np.float32)
    for i, game in enumerate(games):
        encode_one(game, out[i])
    return out


def describe(game: SplendorGame) -> str:
    """A one-line summary of a position, for logs and for reading a game back."""
    parts = []
    for seat in range(1, game.players + 1):
        mark = "*" if seat == game.to_move else " "
        parts.append(f"{mark}P{seat} {int(game.points[seat])}pt "
                     f"{int(game.ncards[seat])}c "
                     f"t{int(game.hand[seat].sum())}")
    board = " ".join(card_label(int(c)) for c in game.board if c >= 0)
    phase = "" if game.phase == PHASE_MAIN else f" phase={game.phase}"
    limit = f" (limit {TOKEN_LIMIT})" if game.phase != PHASE_MAIN else ""
    return f"ply {game.move_count}{phase}{limit} | " + " | ".join(parts) + f"\n  {board}"
