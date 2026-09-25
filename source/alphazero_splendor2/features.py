"""The v2 encoding: what a Splendor player actually looks at, and fast.

Two things are wrong with the v1 encoding, and they are the reason the network
trained on it never got past the hand-written expert.

**It cannot see which gem to take.**  v1 gives each card its cost, and each
player their bonuses and their tokens, and one derived number per card: how many
tokens short of it you are *in total*.  A total is exactly the wrong summary for
the decision the game is mostly made of.  "Take blue, blue" is a good move
because a card you want is two blue short; a network fed only the total has to
re-derive that from three widely separated blocks of the vector, for twelve
cards, every position.  So v2 spells out the **missing tokens per colour** for
every card in play, plus a board-wide *demand* vector -- what the good cards on
the table cost, weighted by how good they are.

**It cannot see the clock.**  Splendor is a race, and v1's only concession to
that is a bit saying the fifteen-point trigger has fired.  v2 carries, per card,
how many *turns of collecting* stand between the mover and owning it (the reach
number the planner agent is built on), and per seat, how far off fifteen they
are and how far ahead or behind.

Two smaller additions come from what the game is like to play rather than from
theory: whether an **opponent can buy** each face-up card right now, which is
the whole of why anybody ever reserves defensively, and whether a card would
**complete a noble**, which is three points that the raw bonus counts only imply.

Everything is seat-relative -- the mover is described first and the rest follow
in turn order -- so one set of weights serves every seat and every table size,
and seat slots past the table are zeroed and flagged inactive.

**Nothing here reads a card the mover has not been shown.**  Deck *order* does
not exist in v2 at all (a deck is a bag), and a card another seat reserved off
the top of a deck is written as "a card of this tier, contents unknown", which
is what everyone else knows.

On speed: this is written to encode a *batch*.  Every derived quantity is one
numpy operation over all the games and all their cards at once, because the
alternative -- a five-element numpy call per card per position -- is what made
v1's encoder a fifth of self-play while the network itself was two per cent.
"""

from __future__ import annotations

import numpy as np

from .game import (BOARD_SLOTS, CCOST, CGEM, CPTS, CTIER, GEMS, Game, GOLD,
                   MAX_RESERVED, MAX_SEATS, NCARDS, NNEED, TIERS, TOKENS,
                   WIN_POINTS)

NAME = "s2"

# ------------------------------------------------------------------- layout
O_BANK = 0                    # 6   tokens in the bank, gold last
O_TAKE2 = O_BANK + TOKENS     # 5   is a "take two of these" legal
O_DEMAND = O_TAKE2 + GEMS     # 5   what the good face-up cards cost, by colour
O_DECKS = O_DEMAND + GEMS     # 3   cards left in each tier
O_PHASE = O_DECKS + TIERS     # 3   main / discarding / choosing a noble
O_TRIG = O_PHASE + 3          # 1   somebody has reached fifteen
O_NPLAYERS = O_TRIG + 1       # 3   two, three or four seats
O_PLY = O_NPLAYERS + 3        # 1
O_NNOBLES = O_PLY + 1         # 1   nobles still unclaimed
GLOBAL_W = O_NNOBLES + 1      # 28

# One card.  The first CARD_STATIC_W are a property of the card and are copied
# from a precomputed table; the rest depend on who is looking at it.
CS_PRESENT = 0
CS_UNKNOWN = 1
CS_TIER = 2                   # 3
CS_POINTS = CS_TIER + TIERS   # 1
CS_BONUS = CS_POINTS + 1      # 5
CS_COST = CS_BONUS + GEMS     # 5
CARD_STATIC_W = CS_COST + GEMS                    # 16

CD_AFFORD = CARD_STATIC_W     # 1   buyable right now, gold included
CD_AFTER = CD_AFFORD + 1      # 1   tokens still to find once gold is spent
CD_TURNS = CD_AFTER + 1       # 1   turns of collecting until it is affordable
CD_MISSING = CD_TURNS + 1     # 5   ...and of which colours.  The important one.
CD_SHORT = CD_MISSING + GEMS  # 1   raw shortfall, before gold
CARD_W = CD_SHORT + 1                             # 25

# A face-up card carries two more things a reserved one does not.
BD_THREAT = CARD_W            # 1   some opponent could buy it this turn
BD_NOBLE = BD_THREAT + 1      # 1   its bonus would complete a noble for me
BOARD_W = BD_NOBLE + 1                            # 27

SB_ACTIVE = 0
SB_HAND = 1                   # 6
SB_TOKENS = SB_HAND + TOKENS  # 1   how full the ten-token hand is
SB_BONUS = SB_TOKENS + 1      # 5
SB_POINTS = SB_BONUS + GEMS   # 1
SB_TOWIN = SB_POINTS + 1      # 1   points still needed for fifteen
SB_NCARDS = SB_TOWIN + 1      # 1
SB_NRESERVED = SB_NCARDS + 1  # 1
SB_LEAD = SB_NRESERVED + 1    # 1   prestige over the best other seat, signed
SB_BUYABLE = SB_LEAD + 1      # 1   how much of the board this seat can afford
SEAT_BASE_W = SB_BUYABLE + 1                      # 19
SEAT_W = SEAT_BASE_W + MAX_RESERVED * CARD_W      # 94

NB_PRESENT = 0
NB_NEED = 1                   # 5
NB_DIST = NB_NEED + GEMS      # 1   bonuses the mover still needs
NB_OPP = NB_DIST + 1          # 1   ...and the nearest opponent
NB_MINE = NB_OPP + 1          # 1   the mover qualifies for it now
NOBLE_W = NB_MINE + 1                             # 9
NOBLE_SLOTS = MAX_SEATS + 1                       # a four-player table lays out 5

O_SEATS = GLOBAL_W
O_BOARD = O_SEATS + MAX_SEATS * SEAT_W
O_NOBLES = O_BOARD + BOARD_SLOTS * BOARD_W
INPUTS = O_NOBLES + NOBLE_SLOTS * NOBLE_W         # 773

# ------------------------------------------------------------------ scales
TOKEN_SCALE = 7.0
BONUS_SCALE = 8.0
COST_SCALE = 7.0
CARD_POINT_SCALE = 5.0
DECK_SCALE = 40.0
CARDS_SCALE = 20.0
SHORT_CAP = 12.0
AFTER_CAP = 10.0
TURN_CAP = 6.0
MISS_SCALE = 5.0
NOBLE_DIST_SCALE = 12.0
PLY_SCALE = 100.0

# --------------------------------------------------------- precomputed tables
# Row NCARDS is the empty slot, so ``table[card_ids]`` works with -1 mapped to
# it and no branching anywhere in the encoder.
_EMPTY = NCARDS
COST_PAD = np.zeros((NCARDS + 1, GEMS), dtype=np.int16)
GEM_PAD = np.zeros(NCARDS + 1, dtype=np.int8)
PTS_PAD = np.zeros(NCARDS + 1, dtype=np.int16)
CARD_STATIC = np.zeros((NCARDS + 1, CARD_STATIC_W), dtype=np.float32)
# The same, for a card whose tier is public and whose face is not.
TIER_STATIC = np.zeros((TIERS + 1, CARD_STATIC_W), dtype=np.float32)

for _c in range(NCARDS):
    COST_PAD[_c] = CCOST[_c]
    GEM_PAD[_c] = CGEM[_c]
    PTS_PAD[_c] = CPTS[_c]
    row = CARD_STATIC[_c]
    row[CS_PRESENT] = 1.0
    row[CS_TIER + CTIER[_c]] = 1.0
    row[CS_POINTS] = CPTS[_c] / CARD_POINT_SCALE
    row[CS_BONUS + CGEM[_c]] = 1.0
    row[CS_COST:CS_COST + GEMS] = np.asarray(CCOST[_c], dtype=np.float32) / COST_SCALE
for _t in range(TIERS):
    TIER_STATIC[_t, CS_PRESENT] = 1.0
    TIER_STATIC[_t, CS_UNKNOWN] = 1.0
    TIER_STATIC[_t, CS_TIER + _t] = 1.0

NOBLE_NEED_PAD = np.zeros((len(NNEED) + 1, GEMS), dtype=np.int16)
for _n, _need in enumerate(NNEED):
    NOBLE_NEED_PAD[_n] = _need
_NO_NOBLE = len(NNEED)

RES_SLOTS = MAX_SEATS * MAX_RESERVED


def _views_are_views() -> bool:
    """The encoder writes through reshaped slices of its output; check it can.

    ``ndarray.reshape`` returns a *copy* when the shape it is asked for cannot
    be expressed in the array's strides, and a copy would be written into and
    silently thrown away.  Every reshape here splits a contiguous trailing axis,
    which is always expressible -- but that is the kind of thing worth asserting
    once at import rather than discovering as a network that will not learn.
    """
    probe = np.zeros((2, INPUTS), dtype=np.float32)
    seats = probe[:, O_SEATS:O_SEATS + MAX_SEATS * SEAT_W].reshape(
        2, MAX_SEATS, SEAT_W)
    reserved = seats[:, :, SEAT_BASE_W:].reshape(
        2, MAX_SEATS, MAX_RESERVED, CARD_W)
    board = probe[:, O_BOARD:O_BOARD + BOARD_SLOTS * BOARD_W].reshape(
        2, BOARD_SLOTS, BOARD_W)
    return bool(np.shares_memory(seats, probe)
                and np.shares_memory(reserved, probe)
                and np.shares_memory(board, probe))


assert _views_are_views(), "the feature layout does not reshape into views"


def size_of(name: str = NAME) -> int:
    if name != NAME:
        raise ValueError(f"unknown Splendor feature set {name!r}")
    return INPUTS


# ---------------------------------------------------------------- gathering
def _gather(games) -> dict:
    """Pull every game's raw numbers into rectangular Python lists.

    One pass of plain Python, then a handful of ``np.array`` calls -- rather
    than a numpy call per card, which is what made the v1 encoder slow.
    """
    n = len(games)
    bonus = np.zeros((n, MAX_SEATS, GEMS), dtype=np.int16)
    hand = np.zeros((n, MAX_SEATS, TOKENS), dtype=np.int16)
    active = np.zeros((n, MAX_SEATS), dtype=bool)
    points = np.zeros((n, MAX_SEATS), dtype=np.int16)
    ncards = np.zeros((n, MAX_SEATS), dtype=np.int16)
    nres = np.zeros((n, MAX_SEATS), dtype=np.int16)
    board = np.full((n, BOARD_SLOTS), _EMPTY, dtype=np.int16)
    res = np.full((n, MAX_SEATS, MAX_RESERVED), _EMPTY, dtype=np.int16)
    res_tier = np.full((n, MAX_SEATS, MAX_RESERVED), TIERS, dtype=np.int8)
    res_known = np.zeros((n, MAX_SEATS, MAX_RESERVED), dtype=bool)
    nobles = np.full((n, NOBLE_SLOTS), _NO_NOBLE, dtype=np.int16)
    scalars = np.zeros((n, 4 + TOKENS + TIERS), dtype=np.float32)

    bonus_rows, hand_rows = [], []
    for i, g in enumerate(games):
        players = g.players
        mover = g.to_move or 1
        active[i, :players] = True
        brow, hrow = [], []
        for k in range(MAX_SEATS):
            if k >= players:
                brow.append((0,) * GEMS)
                hrow.append((0,) * TOKENS)
                continue
            seat = (mover - 1 + k) % players + 1
            brow.append(g.bonus[seat])
            hrow.append(g.hand[seat])
            points[i, k] = g.points[seat]
            ncards[i, k] = g.ncards[seat]
            held = g.reserved[seat]
            secret = g.hidden[seat]
            nres[i, k] = len(held)
            for j, card in enumerate(held):
                res[i, k, j] = card
                res_tier[i, k, j] = CTIER[card]
                res_known[i, k, j] = (seat == mover) or not secret[j]
        bonus_rows.append(brow)
        hand_rows.append(hrow)
        for s, card in enumerate(g.board):
            if card >= 0:
                board[i, s] = card
        for j, noble in enumerate(g.nobles[:NOBLE_SLOTS]):
            nobles[i, j] = noble
        row = scalars[i]
        row[0] = g.phase
        row[1] = 1.0 if g.triggered else 0.0
        row[2] = players
        row[3] = min(g.move_count, 400) / PLY_SCALE
        row[4:4 + TOKENS] = g.bank
        row[4 + TOKENS:] = [len(d) for d in g.decks]

    bonus[:] = np.asarray(bonus_rows, dtype=np.int16)
    hand[:] = np.asarray(hand_rows, dtype=np.int16)
    # A hidden card must not leak its identity into anything derived, so it is
    # replaced by the empty row and reconstructed from its tier alone.
    res_shown = np.where(res_known, res, _EMPTY)
    return dict(bonus=bonus, hand=hand, active=active, points=points,
                ncards=ncards, nres=nres, board=board, res=res,
                res_shown=res_shown, res_tier=res_tier, res_known=res_known,
                nobles=nobles, scalars=scalars)


def _derive(cards: np.ndarray, bonus: np.ndarray, hand: np.ndarray) -> dict:
    """Affordability of every card in ``cards`` for the matching bonus and hand.

    ``cards`` is any shape ending in a card axis; ``bonus`` and ``hand`` carry
    the same leading shape with a trailing gem axis, already broadcast to line
    up.  Returns the five derived numbers the encoder writes per card.
    """
    cost = COST_PAD[cards]
    need = cost - bonus
    np.maximum(need, 0, out=need)
    missing = need - hand[..., :GEMS]
    np.maximum(missing, 0, out=missing)
    short = missing.sum(-1)
    gold = hand[..., GOLD]
    after = short - gold
    afford = after <= 0
    np.maximum(after, 0, out=after)
    worst = missing.max(-1)
    # Two limits bind on a turn of collecting and the slower one wins: three
    # tokens in all, and two of any one colour.
    turns = np.maximum((after + 2) // 3, (worst + 1) // 2)
    return dict(missing=missing, short=short, after=after, afford=afford,
                turns=turns)


def encode(games) -> np.ndarray:
    """A batch of positions as ``(N, INPUTS) float32``."""
    n = len(games)
    out = np.zeros((n, INPUTS), dtype=np.float32)
    if n == 0:
        return out
    d = _gather(games)
    bonus, hand = d["bonus"], d["hand"]
    active = d["active"]
    board = d["board"]
    mover_bonus = bonus[:, 0, :]
    mover_hand = hand[:, 0, :]

    # ----------------------------------------------------------- the globals
    scal = d["scalars"]
    bank = scal[:, 4:4 + TOKENS]
    out[:, O_BANK:O_BANK + TOKENS] = bank / TOKEN_SCALE
    out[:, O_TAKE2:O_TAKE2 + GEMS] = bank[:, :GEMS] >= 4
    out[:, O_DECKS:O_DECKS + TIERS] = scal[:, 4 + TOKENS:] / DECK_SCALE
    out[np.arange(n), O_PHASE + scal[:, 0].astype(np.int64)] = 1.0
    out[:, O_TRIG] = scal[:, 1]
    out[np.arange(n), O_NPLAYERS + scal[:, 2].astype(np.int64) - 2] = 1.0
    out[:, O_PLY] = scal[:, 3]

    present = board != _EMPTY
    # What the face-up cards cost, weighted by how good they are: the colours
    # the *valuable* cards want count for more than the cheap ones'.
    weight = (PTS_PAD[board] + 1) * present
    wsum = weight.sum(1, keepdims=True).clip(min=1)
    demand = (COST_PAD[board] * weight[:, :, None]).sum(1) / wsum
    out[:, O_DEMAND:O_DEMAND + GEMS] = demand / COST_SCALE

    # ------------------------------------------------------------- the board
    bd = _derive(board, mover_bonus[:, None, :], mover_hand[:, None, :])
    bv = out[:, O_BOARD:O_BOARD + BOARD_SLOTS * BOARD_W].reshape(n, BOARD_SLOTS,
                                                                BOARD_W)
    bv[:, :, :CARD_STATIC_W] = CARD_STATIC[board]
    _write_card(bv, bd, present)

    # Could an opponent take it off me this turn?  This is the whole reason a
    # defensive reserve exists, and nothing in the raw state says it.
    opp = _derive(board[:, None, :], bonus[:, 1:, None, :], hand[:, 1:, None, :])
    threat = (opp["afford"] & active[:, 1:, None] & present[:, None, :]).any(1)
    bv[:, :, BD_THREAT] = threat

    # ------------------------------------------------------------ the nobles
    nobles = d["nobles"]
    npresent = nobles != _NO_NOBLE
    nneed = NOBLE_NEED_PAD[nobles]                        # (n, slots, gems)
    deficit = np.maximum(nneed - mover_bonus[:, None, :], 0)
    ndist = deficit.sum(-1)
    opp_def = np.maximum(nneed[:, None, :, :] - bonus[:, 1:, None, :], 0).sum(-1)
    opp_def = np.where(active[:, 1:, None], opp_def, 99)
    opp_dist = opp_def.min(1) if opp_def.shape[1] else np.full_like(ndist, 99)

    nv = out[:, O_NOBLES:O_NOBLES + NOBLE_SLOTS * NOBLE_W].reshape(
        n, NOBLE_SLOTS, NOBLE_W)
    nv[:, :, NB_PRESENT] = npresent
    nv[:, :, NB_NEED:NB_NEED + GEMS] = nneed / 4.0 * npresent[:, :, None]
    nv[:, :, NB_DIST] = np.minimum(ndist, NOBLE_DIST_SCALE) / NOBLE_DIST_SCALE * npresent
    nv[:, :, NB_OPP] = np.minimum(opp_dist, NOBLE_DIST_SCALE) / NOBLE_DIST_SCALE * npresent
    nv[:, :, NB_MINE] = (ndist == 0) & npresent

    # A card is worth three more points than it says when its bonus is the one
    # a noble is waiting on.
    one_away = (ndist == 1) & npresent                    # (n, slots)
    completes = (one_away[:, :, None] & (deficit == 1)).any(1)   # (n, gems)
    bv[:, :, BD_NOBLE] = np.take_along_axis(
        completes, GEM_PAD[board].astype(np.int64), axis=1) & present

    # ------------------------------------------------------------- the seats
    sv = out[:, O_SEATS:O_SEATS + MAX_SEATS * SEAT_W].reshape(n, MAX_SEATS, SEAT_W)
    points = d["points"]
    sv[:, :, SB_ACTIVE] = active
    sv[:, :, SB_HAND:SB_HAND + TOKENS] = hand / TOKEN_SCALE
    sv[:, :, SB_TOKENS] = hand.sum(-1) / 10.0
    sv[:, :, SB_BONUS:SB_BONUS + GEMS] = bonus / BONUS_SCALE
    sv[:, :, SB_POINTS] = points / WIN_POINTS
    sv[:, :, SB_TOWIN] = np.maximum(WIN_POINTS - points, 0) / WIN_POINTS * active
    sv[:, :, SB_NCARDS] = d["ncards"] / CARDS_SCALE
    sv[:, :, SB_NRESERVED] = d["nres"] / MAX_RESERVED
    # Prestige over the best *other* seat: what the race actually turns on.
    masked = np.where(active, points, -99)
    best_other = np.empty_like(points)
    for k in range(MAX_SEATS):
        others = np.delete(masked, k, axis=1)
        best_other[:, k] = others.max(1)
    lead = np.clip((points - best_other) / WIN_POINTS, -1.0, 1.0)
    sv[:, :, SB_LEAD] = lead * active
    seat_buy = _derive(board[:, None, :], bonus[:, :, None, :], hand[:, :, None, :])
    sv[:, :, SB_BUYABLE] = (seat_buy["afford"] & present[:, None, :]).sum(-1) / BOARD_SLOTS

    # ------------------------------------------------- the reserved cards
    res_shown, res_known = d["res_shown"], d["res_known"]
    rv = sv[:, :, SEAT_BASE_W:].reshape(n, MAX_SEATS, MAX_RESERVED, CARD_W)
    rv[:, :, :, :CARD_STATIC_W] = np.where(
        res_known[:, :, :, None],
        CARD_STATIC[res_shown],
        TIER_STATIC[d["res_tier"].astype(np.int64)],
    )
    # A reserved card is judged for whoever holds it: what matters about an
    # opponent's reserve is whether *they* can pay for it.
    rd = _derive(res_shown, bonus[:, :, None, :], hand[:, :, None, :])
    _write_card(rv, rd, res_known & (res_shown != _EMPTY))
    return out


def _write_card(view: np.ndarray, d: dict, present: np.ndarray) -> None:
    """Write the derived half of a card block, zeroed where there is no card."""
    view[..., CD_AFFORD] = d["afford"] & present
    view[..., CD_AFTER] = np.minimum(d["after"], AFTER_CAP) / AFTER_CAP * present
    view[..., CD_TURNS] = np.minimum(d["turns"], TURN_CAP) / TURN_CAP * present
    view[..., CD_MISSING:CD_MISSING + GEMS] = (
        np.minimum(d["missing"], MISS_SCALE) / MISS_SCALE * present[..., None])
    view[..., CD_SHORT] = np.minimum(d["short"], SHORT_CAP) / SHORT_CAP * present


def encode_one(game: Game, out: np.ndarray) -> None:
    """Write one position into ``out``.  Once per ply, not once per leaf."""
    out[:] = encode([game])[0]


def describe(game: Game) -> str:
    """A one-line summary of a position, for logs and for reading a game back."""
    from alphazero_splendor.cards import card_label

    parts = []
    for seat in range(1, game.players + 1):
        mark = "*" if seat == game.to_move else " "
        parts.append(f"{mark}P{seat} {game.points[seat]}pt "
                     f"{game.ncards[seat]}c t{sum(game.hand[seat])}")
    board = " ".join(card_label(c) for c in game.board if c >= 0)
    phase = "" if game.phase == 0 else f" phase={game.phase}"
    return f"ply {game.move_count}{phase} | " + " | ".join(parts) + f"\n  {board}"
