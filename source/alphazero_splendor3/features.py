"""What the v3 network sees, and the two symmetries that makes it worth 100x.

v2 encodes a position as one flat vector and hands it to an MLP.  That works,
but it throws away everything the game says about which parts of the vector are
interchangeable, and Splendor says a great deal:

**The four face-up cards of a tier are a set, not a list.**  Swapping two of
them is the same position.  v2's encoder writes them into fixed slots and v2's
policy head reads twelve fixed logits, so the network has to learn twelve times
that a two-point card costing three blue is worth buying.  Here a card is a
**token**, all twenty-four of them go through the same weights, and the policy
logit for "buy this card" is read off *that card's* token.  The symmetry group
is 4!^3 = 13,824 relabellings of one position.

**The five colours are interchangeable** -- but only if you permute the whole
state, and the interesting part is that the obvious version of this is wrong.
The printed deck is *not* colour-symmetric: 24 to 26 of its 90 cards change
identity under a cyclic colour shift, and its exact symmetry group has order
one.  Permuting the visible board alone would train on a game that does not
exist.  It becomes exact once the *remaining decks* are permuted too, which is
why :class:`alphazero_splendor3.game.Game` carries a running summary of them.
With that, a position is worth up to 120 relabellings, and the five colours get
their own tokens so the policy logits for "take two blue" and "take two red"
come out of the same weights.

Neither symmetry costs a forward pass: they are applied to training batches, not
at inference.

The layout is four blocks -- globals, six colour tokens, twenty-four card
tokens, five noble tokens -- and every colour-indexed run inside them is listed
in :data:`GEM_SLICES` so the permutation is a table lookup rather than a hand
edit that quietly misses one.
"""

from __future__ import annotations

import numpy as np

from .game import (BOARD_SLOTS, CCOST, CGEM, CPTS, CTIER, DA_BONUS, DA_COST,
                   DA_POINTS, DECK_AGG_W, GEMS, GOLD, Game, MAX_RESERVED,
                   MAX_SEATS, NACTIONS, NCARDS, NNEED, ROW, TIERS, TOKENS,
                   WIN_POINTS, A_BUY, A_BUY_RESERVED, A_DISCARD, A_RESERVE,
                   A_TAKE1, A_TAKE2D, A_TAKE2S, A_TAKE3, TAKE2D, TAKE3)

NAME = "s3"

# ------------------------------------------------------------ global block
G_PHASE = 0                          # 3
G_TRIG = G_PHASE + 3                 # 1
G_NPLAYERS = G_TRIG + 1              # 3
G_PLY = G_NPLAYERS + 3               # 1
G_NNOBLES = G_PLY + 1                # 1
G_DECKS = G_NNOBLES + 1              # 3   cards left per tier
G_DAGG = G_DECKS + TIERS             # 33  what is left in them, by colour
G_SEATS = G_DAGG + TIERS * DECK_AGG_W

SB_ACTIVE = 0
SB_HAND = 1                          # 6   (colour-indexed, gold last)
SB_NTOK = SB_HAND + TOKENS           # 1
SB_BONUS = SB_NTOK + 1               # 5   (colour-indexed)
SB_POINTS = SB_BONUS + GEMS          # 1
SB_TOWIN = SB_POINTS + 1             # 1
SB_NCARDS = SB_TOWIN + 1             # 1
SB_NRES = SB_NCARDS + 1              # 1
SB_LEAD = SB_NRES + 1                # 1
SB_BUYABLE = SB_LEAD + 1             # 1
SEAT_W = SB_BUYABLE + 1              # 19

GLOBAL_W = G_SEATS + MAX_SEATS * SEAT_W          # 121

# ------------------------------------------------------------ colour tokens
COLOUR_TOKENS = TOKENS               # five gems and gold
L_ISGOLD = 0
L_BANK = 1
L_HAND = 2
L_BONUS = 3
L_DEMAND = 4                         # what the good face-up cards cost in it
L_TAKE1 = 5
L_TAKE2S = 6
L_OPPBONUS = 7                       # the best opponent's holding of it
L_DECKCOST = 8                       # what is left in the decks, in it
COLOUR_W = L_DECKCOST + 1            # 9

# ------------------------------------------------------------- card tokens
CARD_TOKENS = BOARD_SLOTS + MAX_SEATS * MAX_RESERVED     # 12 + 12
C_PRESENT = 0
C_UNKNOWN = 1
C_ISBOARD = 2
C_OWNER = 3                          # 4   seat-relative, all zero on the board
C_TIER = C_OWNER + MAX_SEATS         # 3
C_POINTS = C_TIER + TIERS            # 1
C_BONUS = C_POINTS + 1               # 5   (colour-indexed)
C_COST = C_BONUS + GEMS              # 5   (colour-indexed)
C_AFFORD = C_COST + GEMS             # 1
C_AFTER = C_AFFORD + 1               # 1
C_TURNS = C_AFTER + 1                # 1
C_MISSING = C_TURNS + 1              # 5   (colour-indexed) -- the important one
C_SHORT = C_MISSING + GEMS           # 1
C_THREAT = C_SHORT + 1               # 1   an opponent could buy it this turn
C_NOBLE = C_THREAT + 1               # 1   its bonus would complete a noble
CARD_W = C_NOBLE + 1                 # 32

# ------------------------------------------------------------ noble tokens
NOBLE_TOKENS = MAX_SEATS + 1         # a four-player table lays out five
N_PRESENT = 0
N_NEED = 1                           # 5   (colour-indexed)
N_DIST = N_NEED + GEMS               # 1
N_OPP = N_DIST + 1                   # 1
N_MINE = N_OPP + 1                   # 1
NOBLE_W = N_MINE + 1                 # 9

O_GLOBAL = 0
O_COLOUR = O_GLOBAL + GLOBAL_W
O_CARDS = O_COLOUR + COLOUR_TOKENS * COLOUR_W
O_NOBLES = O_CARDS + CARD_TOKENS * CARD_W
INPUTS = O_NOBLES + NOBLE_TOKENS * NOBLE_W

# ------------------------------------------------------------------ scales
TOKEN_SCALE = 7.0
BONUS_SCALE = 8.0
COST_SCALE = 7.0
CARD_POINT_SCALE = 5.0
DECK_SCALE = 40.0
DECK_COST_SCALE = 80.0
DECK_BONUS_SCALE = 10.0
DECK_POINT_SCALE = 40.0
CARDS_SCALE = 20.0
SHORT_CAP = 12.0
AFTER_CAP = 10.0
TURN_CAP = 6.0
MISS_SCALE = 5.0
NOBLE_DIST_SCALE = 12.0
PLY_SCALE = 100.0

# --------------------------------------------------------- precomputed tables
_EMPTY = NCARDS
COST_PAD = np.zeros((NCARDS + 1, GEMS), dtype=np.int16)
GEM_PAD = np.zeros(NCARDS + 1, dtype=np.int8)
PTS_PAD = np.zeros(NCARDS + 1, dtype=np.int16)
CARD_STATIC = np.zeros((NCARDS + 1, C_AFFORD), dtype=np.float32)
TIER_STATIC = np.zeros((TIERS + 1, C_AFFORD), dtype=np.float32)

for _c in range(NCARDS):
    COST_PAD[_c] = CCOST[_c]
    GEM_PAD[_c] = CGEM[_c]
    PTS_PAD[_c] = CPTS[_c]
    row = CARD_STATIC[_c]
    row[C_PRESENT] = 1.0
    row[C_TIER + CTIER[_c]] = 1.0
    row[C_POINTS] = CPTS[_c] / CARD_POINT_SCALE
    row[C_BONUS + CGEM[_c]] = 1.0
    row[C_COST:C_COST + GEMS] = np.asarray(CCOST[_c], dtype=np.float32) / COST_SCALE
for _t in range(TIERS):
    TIER_STATIC[_t, C_PRESENT] = 1.0
    TIER_STATIC[_t, C_UNKNOWN] = 1.0
    TIER_STATIC[_t, C_TIER + _t] = 1.0

NOBLE_NEED_PAD = np.zeros((len(NNEED) + 1, GEMS), dtype=np.int16)
for _n, _need in enumerate(NNEED):
    NOBLE_NEED_PAD[_n] = _need
_NO_NOBLE = len(NNEED)


def size_of(name: str = NAME) -> int:
    if name != NAME:
        raise ValueError(f"unknown feature set {name!r}")
    return INPUTS


# ==================================================================== views
# The network wants tokens, the replay buffer wants one flat row.  These are
# views, not copies: reshaping a contiguous slice costs nothing.
def views(x: np.ndarray) -> tuple[np.ndarray, ...]:
    """``(globals, colours, cards, nobles)`` over a ``(N, INPUTS)`` batch."""
    n = x.shape[0]
    g = x[:, O_GLOBAL:O_GLOBAL + GLOBAL_W]
    c = x[:, O_COLOUR:O_CARDS].reshape(n, COLOUR_TOKENS, COLOUR_W)
    k = x[:, O_CARDS:O_NOBLES].reshape(n, CARD_TOKENS, CARD_W)
    b = x[:, O_NOBLES:].reshape(n, NOBLE_TOKENS, NOBLE_W)
    return g, c, k, b


# =============================================================== symmetries
def _gem_slices() -> list[tuple[str, int, int]]:
    """Every five-wide colour run, as ``(block, token or -1, offset)``.

    Listed once, used by the permutation, so adding a colour-indexed field
    cannot silently escape being permuted: the round-trip test in
    ``tests/test_splendor3.py`` re-derives a permuted encoding from a permuted
    *game* and asserts it matches.
    """
    out: list[tuple[str, int, int]] = []
    for t in range(TIERS):
        out.append(("global", -1, G_DAGG + t * DECK_AGG_W + DA_BONUS))
        out.append(("global", -1, G_DAGG + t * DECK_AGG_W + DA_COST))
    for k in range(MAX_SEATS):
        base = G_SEATS + k * SEAT_W
        out.append(("global", -1, base + SB_HAND))     # six wide, gold last
        out.append(("global", -1, base + SB_BONUS))
    out.append(("card", -1, C_BONUS))
    out.append(("card", -1, C_COST))
    out.append(("card", -1, C_MISSING))
    out.append(("noble", -1, N_NEED))
    return out


GEM_SLICES = _gem_slices()


def permute_features(x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Relabel the colours of a ``(N, INPUTS)`` batch.  ``perm[old] = new``."""
    out = x.copy()
    g, c, k, b = views(out)
    src_g, src_c, src_k, src_b = views(x)
    inv = np.argsort(perm)                      # new position <- old position

    for block, _tok, off in GEM_SLICES:
        if block == "global":
            g[:, off:off + GEMS] = src_g[:, off + inv]
        elif block == "card":
            k[:, :, off:off + GEMS] = src_k[:, :, off + inv]
        else:
            b[:, :, off:off + GEMS] = src_b[:, :, off + inv]
    # The colour tokens themselves; gold is token five and does not move.
    c[:, :GEMS, :] = src_c[:, inv, :]
    return out


def permute_actions(perm: np.ndarray) -> np.ndarray:
    """``idx[old_action] = new_action`` for a colour relabelling."""
    idx = np.arange(NACTIONS, dtype=np.int64)
    trio = {frozenset(t): i for i, t in enumerate(TAKE3)}
    for i, t in enumerate(TAKE3):
        idx[A_TAKE3 + i] = A_TAKE3 + trio[frozenset(int(perm[g]) for g in t)]
    pair = {frozenset(t): i for i, t in enumerate(TAKE2D)}
    for i, t in enumerate(TAKE2D):
        idx[A_TAKE2D + i] = A_TAKE2D + pair[frozenset(int(perm[g]) for g in t)]
    for g in range(GEMS):
        idx[A_TAKE1 + g] = A_TAKE1 + int(perm[g])
        idx[A_TAKE2S + g] = A_TAKE2S + int(perm[g])
        idx[A_DISCARD + g] = A_DISCARD + int(perm[g])
    return idx


def permute_slots(order: np.ndarray) -> np.ndarray:
    """``idx[old_action] = new_action`` for a relabelling of the board slots.

    ``order`` is read the way the token shuffle uses it: the card that was in
    slot ``order[j]`` is now in slot ``j``.
    """
    idx = np.arange(NACTIONS, dtype=np.int64)
    for j in range(BOARD_SLOTS):
        s = int(order[j])
        idx[A_BUY + s] = A_BUY + j
        idx[A_RESERVE + s] = A_RESERVE + j
    return idx


def random_slot_order(rng: np.random.Generator) -> np.ndarray:
    """A permutation of the twelve slots that keeps each tier in its own row."""
    order = np.empty(BOARD_SLOTS, dtype=np.int64)
    for t in range(TIERS):
        order[t * ROW:(t + 1) * ROW] = t * ROW + rng.permutation(ROW)
    return order


def augment(x: np.ndarray, pi: np.ndarray, rng: np.random.Generator,
            colours: bool = True, slots: bool = True):
    """One relabelled copy of a training batch: features and policy together.

    Both symmetries are exact given what the state carries, so this is more
    training data rather than noise added to it.
    """
    if colours:
        perm = rng.permutation(GEMS)
        if not np.array_equal(perm, np.arange(GEMS)):
            x = permute_features(x, perm)
            pi = pi[:, np.argsort(permute_actions(perm))]
    if slots:
        order = random_slot_order(rng)
        if not np.array_equal(order, np.arange(BOARD_SLOTS)):
            x = x.copy()
            _, _, k, _ = views(x)
            k[:, :BOARD_SLOTS, :] = k[:, order, :]
            pi = pi[:, np.argsort(permute_slots(order))]
    return x, pi


# ==================================================================== encode
def _gather(games) -> dict:
    """Every game's raw numbers as rectangular arrays, in one Python pass."""
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
    nobles = np.full((n, NOBLE_TOKENS), _NO_NOBLE, dtype=np.int16)
    scalars = np.zeros((n, 4 + TOKENS + TIERS), dtype=np.float32)
    dagg = np.zeros((n, TIERS, DECK_AGG_W), dtype=np.float32)

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
        for j, noble in enumerate(g.nobles[:NOBLE_TOKENS]):
            nobles[i, j] = noble
        row = scalars[i]
        row[0] = g.phase
        row[1] = 1.0 if g.triggered else 0.0
        row[2] = players
        row[3] = min(g.move_count, 400) / PLY_SCALE
        row[4:4 + TOKENS] = g.bank
        row[4 + TOKENS:] = [len(d) for d in g.decks]
        dagg[i] = g.dagg

    bonus[:] = np.asarray(bonus_rows, dtype=np.int16)
    hand[:] = np.asarray(hand_rows, dtype=np.int16)
    res_shown = np.where(res_known, res, _EMPTY)
    return dict(bonus=bonus, hand=hand, active=active, points=points,
                ncards=ncards, nres=nres, board=board, res_shown=res_shown,
                res_tier=res_tier, res_known=res_known, nobles=nobles,
                scalars=scalars, dagg=dagg)


def _derive(cards: np.ndarray, bonus: np.ndarray, hand: np.ndarray) -> dict:
    """Affordability of every card, for the matching bonus and hand."""
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
    # Three tokens a turn, or two of one colour: the slower limit wins.
    turns = np.maximum((after + 2) // 3, (worst + 1) // 2)
    return dict(missing=missing, short=short, after=after, afford=afford,
                turns=turns)


def _write_card(view: np.ndarray, d: dict, present: np.ndarray) -> None:
    view[..., C_AFFORD] = d["afford"] & present
    view[..., C_AFTER] = np.minimum(d["after"], AFTER_CAP) / AFTER_CAP * present
    view[..., C_TURNS] = np.minimum(d["turns"], TURN_CAP) / TURN_CAP * present
    view[..., C_MISSING:C_MISSING + GEMS] = (
        np.minimum(d["missing"], MISS_SCALE) / MISS_SCALE * present[..., None])
    view[..., C_SHORT] = np.minimum(d["short"], SHORT_CAP) / SHORT_CAP * present


def encode(games) -> np.ndarray:
    """A batch of positions as ``(N, INPUTS) float32``, mover-relative."""
    n = len(games)
    out = np.zeros((n, INPUTS), dtype=np.float32)
    if n == 0:
        return out
    d = _gather(games)
    gv, cv, kv, bv = views(out)

    bonus, hand, active = d["bonus"], d["hand"], d["active"]
    board, points = d["board"], d["points"]
    mover_bonus = bonus[:, 0, :]
    mover_hand = hand[:, 0, :]
    scal = d["scalars"]
    bank = scal[:, 4:4 + TOKENS]
    rows = np.arange(n)

    # ----------------------------------------------------------- the globals
    gv[rows, G_PHASE + scal[:, 0].astype(np.int64)] = 1.0
    gv[:, G_TRIG] = scal[:, 1]
    gv[rows, G_NPLAYERS + scal[:, 2].astype(np.int64) - 2] = 1.0
    gv[:, G_PLY] = scal[:, 3]
    gv[:, G_NNOBLES] = (d["nobles"] != _NO_NOBLE).sum(1) / NOBLE_TOKENS
    gv[:, G_DECKS:G_DECKS + TIERS] = scal[:, 4 + TOKENS:] / DECK_SCALE

    dagg = d["dagg"].copy()
    dagg[:, :, DA_BONUS:DA_BONUS + GEMS] /= DECK_BONUS_SCALE
    dagg[:, :, DA_COST:DA_COST + GEMS] /= DECK_COST_SCALE
    dagg[:, :, DA_POINTS] /= DECK_POINT_SCALE
    gv[:, G_DAGG:G_DAGG + TIERS * DECK_AGG_W] = dagg.reshape(n, -1)

    sv = gv[:, G_SEATS:].reshape(n, MAX_SEATS, SEAT_W)
    sv[:, :, SB_ACTIVE] = active
    sv[:, :, SB_HAND:SB_HAND + TOKENS] = hand / TOKEN_SCALE
    sv[:, :, SB_NTOK] = hand.sum(-1) / 10.0
    sv[:, :, SB_BONUS:SB_BONUS + GEMS] = bonus / BONUS_SCALE
    sv[:, :, SB_POINTS] = points / WIN_POINTS
    sv[:, :, SB_TOWIN] = np.maximum(WIN_POINTS - points, 0) / WIN_POINTS * active
    sv[:, :, SB_NCARDS] = d["ncards"] / CARDS_SCALE
    sv[:, :, SB_NRES] = d["nres"] / MAX_RESERVED
    masked = np.where(active, points, -99)
    best_other = np.empty_like(points)
    for k in range(MAX_SEATS):
        best_other[:, k] = np.delete(masked, k, axis=1).max(1)
    sv[:, :, SB_LEAD] = np.clip((points - best_other) / WIN_POINTS, -1.0, 1.0) * active

    present = board != _EMPTY
    seat_buy = _derive(board[:, None, :], bonus[:, :, None, :], hand[:, :, None, :])
    sv[:, :, SB_BUYABLE] = ((seat_buy["afford"] & present[:, None, :]).sum(-1)
                            / BOARD_SLOTS)

    # ------------------------------------------------------ the colour tokens
    # Weighted by how good the card is, so the colours the *valuable* cards
    # want count for more than the cheap ones'.
    weight = (PTS_PAD[board] + 1) * present
    wsum = weight.sum(1, keepdims=True).clip(min=1)
    demand = (COST_PAD[board] * weight[:, :, None]).sum(1) / wsum

    cv[:, GOLD, L_ISGOLD] = 1.0
    cv[:, :, L_BANK] = bank / TOKEN_SCALE
    cv[:, :, L_HAND] = mover_hand / TOKEN_SCALE
    cv[:, :GEMS, L_BONUS] = mover_bonus / BONUS_SCALE
    cv[:, :GEMS, L_DEMAND] = demand / COST_SCALE
    cv[:, :GEMS, L_TAKE1] = bank[:, :GEMS] > 0
    cv[:, :GEMS, L_TAKE2S] = bank[:, :GEMS] >= 4
    opp_bonus = np.where(active[:, 1:, None], bonus[:, 1:, :], 0)
    cv[:, :GEMS, L_OPPBONUS] = (opp_bonus.max(1) if opp_bonus.shape[1]
                                else 0) / BONUS_SCALE
    cv[:, :GEMS, L_DECKCOST] = (d["dagg"][:, :, DA_COST:DA_COST + GEMS].sum(1)
                                / DECK_COST_SCALE)

    # -------------------------------------------------------- the card tokens
    bd = _derive(board, mover_bonus[:, None, :], mover_hand[:, None, :])
    face = kv[:, :BOARD_SLOTS, :]
    face[:, :, :C_AFFORD] = CARD_STATIC[board]
    face[:, :, C_ISBOARD] = present
    _write_card(face, bd, present)

    opp = _derive(board[:, None, :], bonus[:, 1:, None, :], hand[:, 1:, None, :])
    face[:, :, C_THREAT] = (opp["afford"] & active[:, 1:, None]
                            & present[:, None, :]).any(1)

    res_shown, res_known = d["res_shown"], d["res_known"]
    rv = kv[:, BOARD_SLOTS:, :].reshape(n, MAX_SEATS, MAX_RESERVED, CARD_W)
    rv[:, :, :, :C_AFFORD] = np.where(
        res_known[:, :, :, None], CARD_STATIC[res_shown],
        TIER_STATIC[d["res_tier"].astype(np.int64)])
    held = np.arange(MAX_RESERVED)[None, None, :] < d["nres"][:, :, None]
    for k in range(MAX_SEATS):
        rv[:, k, :, C_OWNER + k] = held[:, k, :]
    # An opponent's reserve is judged by whether *they* can pay for it.
    rd = _derive(res_shown, bonus[:, :, None, :], hand[:, :, None, :])
    _write_card(rv, rd, res_known & (res_shown != _EMPTY))

    # ------------------------------------------------------- the noble tokens
    nobles = d["nobles"]
    npresent = nobles != _NO_NOBLE
    nneed = NOBLE_NEED_PAD[nobles]
    deficit = np.maximum(nneed - mover_bonus[:, None, :], 0)
    ndist = deficit.sum(-1)
    opp_def = np.maximum(nneed[:, None, :, :] - bonus[:, 1:, None, :], 0).sum(-1)
    opp_def = np.where(active[:, 1:, None], opp_def, 99)
    opp_dist = opp_def.min(1) if opp_def.shape[1] else np.full_like(ndist, 99)

    bv[:, :, N_PRESENT] = npresent
    bv[:, :, N_NEED:N_NEED + GEMS] = nneed / 4.0 * npresent[:, :, None]
    bv[:, :, N_DIST] = (np.minimum(ndist, NOBLE_DIST_SCALE)
                        / NOBLE_DIST_SCALE * npresent)
    bv[:, :, N_OPP] = (np.minimum(opp_dist, NOBLE_DIST_SCALE)
                       / NOBLE_DIST_SCALE * npresent)
    bv[:, :, N_MINE] = (ndist == 0) & npresent

    one_away = (ndist == 1) & npresent
    completes = (one_away[:, :, None] & (deficit == 1)).any(1)     # (n, gems)
    face[:, :, C_NOBLE] = np.take_along_axis(
        completes, GEM_PAD[board].astype(np.int64), axis=1) & present
    return out


def encode_one(game: Game, out: np.ndarray) -> None:
    out[:] = encode([game])[0]


def describe(game: Game) -> str:
    from alphazero_splendor.cards import card_label

    parts = []
    for seat in range(1, game.players + 1):
        mark = "*" if seat == game.to_move else " "
        parts.append(f"{mark}P{seat} {game.points[seat]}pt "
                     f"{game.ncards[seat]}c t{sum(game.hand[seat])}")
    board = " ".join(card_label(c) for c in game.board if c >= 0)
    phase = "" if game.phase == 0 else f" phase={game.phase}"
    return f"ply {game.move_count}{phase} | " + " | ".join(parts) + f"\n  {board}"
