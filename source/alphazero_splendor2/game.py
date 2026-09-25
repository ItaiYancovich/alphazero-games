"""Splendor, written for the search rather than for generality.

:mod:`alphazero_splendor.splendor_game` is the readable reference implementation
and this is the same game with the same 72 actions, rewritten around what a
profile of self-play actually showed.  Two thirds of a self-play second went
into ``token_shortfall`` and the mask that calls it -- a dozen numpy operations
on five-element arrays, twelve times a node -- while the *network* took two per
cent.  Numpy is the wrong tool at that size: the array bookkeeping costs more
than the five subtractions it is hiding.  So every hot quantity here is a plain
Python ``int`` in a plain Python ``list``, and a card's cost is stored as the
two or three non-zero ``(colour, amount)`` pairs a real card has rather than as
a five-vector that is mostly zero.

The other change is *where chance is resolved*.  The reference shuffles each
deck once at setup, and a search that must not read that shuffle re-permutes
every deck at the root of every simulation.  Here a deck is an unordered **bag**
and a card is drawn from it at the moment a slot is refilled -- which is where
the rules put the chance, costs nothing, and makes the root determinization
almost free: all that is left to resample is the handful of cards an *opponent*
reserved off the top of a deck, which is usually none.

Everything else -- the action numbering, the sub-turn phases, the tie-break, the
result vector -- is deliberately identical to the reference, so a move index
means the same thing in both and :mod:`alphazero_splendor2.bridge` can hand a
position from one to the other.
"""

from __future__ import annotations

import random
from itertools import combinations

import numpy as np

from alphazero_splendor.cards import (CARDS, DECKS, GEM_NAMES, GEMS, GOLD,
                                      NNOBLES, NOBLES, TIERS, TOKENS,
                                      card_label, noble_label)

# --------------------------------------------------------------- card tables
# Plain tuples of ints.  ``CPAIRS`` is the one that matters: a Splendor card
# costs two or three colours, so iterating its non-zero entries is half the work
# of iterating five, and none of it allocates.
NCARDS = len(CARDS)
CCOST: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(x) for x in c.cost) for c in CARDS)
CPAIRS: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple((i, int(a)) for i, a in enumerate(c.cost) if a) for c in CARDS)
CGEM: tuple[int, ...] = tuple(int(c.gem) for c in CARDS)
CPTS: tuple[int, ...] = tuple(int(c.points) for c in CARDS)
CTIER: tuple[int, ...] = tuple(int(c.tier) for c in CARDS)

NNEED: tuple[tuple[int, ...], ...] = tuple(
    tuple(int(x) for x in n.need) for n in NOBLES)
NPAIRS: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple((i, int(a)) for i, a in enumerate(n.need) if a) for n in NOBLES)

# ------------------------------------------------------------------- setup
ROW = 4
MAX_RESERVED = 3
TOKEN_LIMIT = 10
GOLD_SUPPLY = 5
WIN_POINTS = 15
TAKE_SAME_MIN = 4
SUPPLY = {2: 4, 3: 5, 4: 7}
MIN_PLAYERS, MAX_PLAYERS = 2, 4
MAX_SEATS = MAX_PLAYERS
MAX_PLIES = 400

PHASE_MAIN = 0
PHASE_DISCARD = 1
PHASE_NOBLE = 2

# --------------------------------------------------- the action space (72)
TAKE3 = tuple(combinations(range(GEMS), 3))
TAKE2D = tuple(combinations(range(GEMS), 2))

A_TAKE3 = 0
A_TAKE2D = A_TAKE3 + len(TAKE3)          # 10
A_TAKE1 = A_TAKE2D + len(TAKE2D)         # 20
A_TAKE2S = A_TAKE1 + GEMS                # 25
A_BUY = A_TAKE2S + GEMS                  # 30
A_BUY_RESERVED = A_BUY + TIERS * ROW     # 42
A_RESERVE = A_BUY_RESERVED + MAX_RESERVED        # 45
A_RESERVE_DECK = A_RESERVE + TIERS * ROW         # 57
A_DISCARD = A_RESERVE_DECK + TIERS               # 60
A_NOBLE = A_DISCARD + TOKENS                     # 66
A_PASS = A_NOBLE + 5                             # 71
NACTIONS = A_PASS + 1                            # 72

BOARD_SLOTS = TIERS * ROW
SLOT_TIER = tuple(s // ROW for s in range(BOARD_SLOTS))


def slot_of(tier: int, index: int) -> int:
    return tier * ROW + index


class Game:
    """One Splendor position for two to four seats.

    Seats are ``1..players`` and every seat-indexed list has a dead entry 0, so
    a seat number indexes directly -- the convention the rest of the project
    uses.  ``draw`` is what makes a position a *search* position rather than the
    server's: when it is set, a refill draws uniformly from the tier's bag
    instead of taking a fixed card off it.
    """

    __slots__ = ("players", "to_move", "move_count", "phase", "winner",
                 "triggered", "pending", "bank", "hand", "bonus", "points",
                 "ncards", "reserved", "hidden", "board", "decks", "nobles",
                 "draw", "_legal", "_mask", "_legal_np")

    # ------------------------------------------------------------------ setup
    def __init__(self, players: int = 2, rng=None, _blank: bool = False):
        if not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"Splendor is for {MIN_PLAYERS}-{MAX_PLAYERS} players")
        self.players = int(players)
        self.draw: random.Random | None = None
        self._legal: list[int] | None = None
        self._mask: np.ndarray | None = None
        self._legal_np: np.ndarray | None = None
        if _blank:
            return
        py = as_py_random(rng)

        self.to_move = 1
        self.move_count = 0
        self.phase = PHASE_MAIN
        self.winner = 0
        self.triggered = False
        self.pending: list[int] = []

        n = self.players + 1
        supply = SUPPLY[self.players]
        self.bank = [supply] * GEMS + [GOLD_SUPPLY]
        self.hand = [[0] * TOKENS for _ in range(n)]
        self.bonus = [[0] * GEMS for _ in range(n)]
        self.points = [0] * n
        self.ncards = [0] * n
        self.reserved: list[list[int]] = [[] for _ in range(n)]
        self.hidden: list[list[bool]] = [[] for _ in range(n)]

        # A deck is a bag *to the search*, which draws from it with
        # ``pop_random`` and so cannot depend on its order.  The server is a
        # different matter: with ``draw`` unset a refill takes ``deck.pop()``,
        # the last element, so the order is exactly what it deals.  This shuffle
        # is therefore not optional and not a formality.
        #
        # It was missing once, and the failure is worth recording because
        # nothing raised.  ``DECKS`` is grouped by bonus colour, so popping the
        # end dealt the first tier-1 refill as a black card in 60 games out of
        # 60.  Self-play then ran on a deck it could memorise, the network duly
        # memorised it, and every metric measured against that same biased deal
        # looked excellent -- 0.87 against the planner -- while the same network
        # scored 0.45 on properly shuffled positions.  A network that reads the
        # shuffle is the one thing this whole design exists to prevent.
        self.decks = [list(DECKS[t]) for t in range(TIERS)]
        for deck in self.decks:
            py.shuffle(deck)
        self.board = [-1] * BOARD_SLOTS
        for tier in range(TIERS):
            deck = self.decks[tier]
            for i in range(ROW):
                self.board[slot_of(tier, i)] = pop_random(deck, py)
        pick = list(range(NNOBLES))
        py.shuffle(pick)
        self.nobles = pick[:self.players + 1]

    def copy(self) -> "Game":
        o = Game(self.players, _blank=True)
        o.to_move = self.to_move
        o.move_count = self.move_count
        o.phase = self.phase
        o.winner = self.winner
        o.triggered = self.triggered
        o.pending = self.pending
        o.bank = self.bank[:]
        o.hand = [h[:] for h in self.hand]
        o.bonus = [b[:] for b in self.bonus]
        o.points = self.points[:]
        o.ncards = self.ncards[:]
        o.reserved = [r[:] for r in self.reserved]
        o.hidden = [h[:] for h in self.hidden]
        o.board = self.board[:]
        o.decks = [d[:] for d in self.decks]
        o.nobles = self.nobles[:]
        o.draw = self.draw
        o._legal = self._legal
        o._mask = self._mask
        o._legal_np = self._legal_np
        return o

    # ------------------------------------------------------------- internals
    def _refill(self, slot: int) -> None:
        deck = self.decks[SLOT_TIER[slot]]
        if not deck:
            self.board[slot] = -1
        elif self.draw is None:
            self.board[slot] = deck.pop()
        else:
            self.board[slot] = pop_random(deck, self.draw)

    def shortfall(self, seat: int, card: int) -> int:
        """Coloured tokens ``seat`` is still missing for ``card``, gold aside."""
        bonus = self.bonus[seat]
        hand = self.hand[seat]
        short = 0
        for colour, amount in CPAIRS[card]:
            gap = amount - bonus[colour] - hand[colour]
            if gap > 0:
                short += gap
        return short

    def can_afford(self, seat: int, card: int) -> bool:
        return self.shortfall(seat, card) <= self.hand[seat][GOLD]

    def _pay(self, seat: int, card: int) -> None:
        bonus = self.bonus[seat]
        hand = self.hand[seat]
        bank = self.bank
        gold = 0
        for colour, amount in CPAIRS[card]:
            need = amount - bonus[colour]
            if need <= 0:
                continue
            have = hand[colour]
            if have >= need:
                hand[colour] = have - need
                bank[colour] += need
            else:
                hand[colour] = 0
                bank[colour] += have
                gold += need - have
        if gold:
            hand[GOLD] -= gold
            bank[GOLD] += gold

    def _gain_card(self, seat: int, card: int) -> None:
        self.bonus[seat][CGEM[card]] += 1
        self.points[seat] += CPTS[card]
        self.ncards[seat] += 1

    def _qualifying_nobles(self, seat: int) -> list[int]:
        bonus = self.bonus[seat]
        out = []
        for noble in self.nobles:
            for colour, amount in NPAIRS[noble]:
                if bonus[colour] < amount:
                    break
            else:
                out.append(noble)
        return out

    def held_tokens(self, seat: int) -> int:
        return sum(self.hand[seat])

    # ------------------------------------------------------------- legality
    def legal_actions(self) -> list[int]:
        """The legal action indices, as a Python list -- what the search wants."""
        legal = self._legal
        if legal is None:
            legal = self._legal = self._compute_legal()
        return legal

    def legal_mask(self) -> np.ndarray:
        """The same, as a ``(72,) bool`` array -- what the encoder wants."""
        mask = self._mask
        if mask is None:
            mask = np.zeros(NACTIONS, dtype=bool)
            legal = self.legal_actions()
            if legal:
                mask[legal] = True
            self._mask = mask
        return mask

    def legal_moves(self) -> np.ndarray:
        """The legal actions as an index array, cached: the search asks per visit."""
        idx = self._legal_np
        if idx is None:
            idx = self._legal_np = np.asarray(self.legal_actions(), dtype=np.int64)
        return idx

    def is_legal(self, move) -> bool:
        return int(move) in self.legal_actions()

    def _compute_legal(self) -> list[int]:
        if self.to_move == 0:
            return []
        seat = self.to_move
        out: list[int] = []

        if self.phase == PHASE_DISCARD:
            hand = self.hand[seat]
            for colour in range(TOKENS):
                if hand[colour]:
                    out.append(A_DISCARD + colour)
            return out
        if self.phase == PHASE_NOBLE:
            pending = self.pending
            for i, noble in enumerate(self.nobles):
                if noble in pending:
                    out.append(A_NOBLE + i)
            return out

        bank = self.bank
        available = 0
        for g in range(GEMS):
            if bank[g]:
                available += 1
        if available >= 3:
            for i, trio in enumerate(TAKE3):
                if bank[trio[0]] and bank[trio[1]] and bank[trio[2]]:
                    out.append(A_TAKE3 + i)
        elif available == 2:
            for i, pair in enumerate(TAKE2D):
                if bank[pair[0]] and bank[pair[1]]:
                    out.append(A_TAKE2D + i)
        elif available == 1:
            for g in range(GEMS):
                if bank[g]:
                    out.append(A_TAKE1 + g)
        for g in range(GEMS):
            if bank[g] >= TAKE_SAME_MIN:
                out.append(A_TAKE2S + g)

        bonus = self.bonus[seat]
        hand = self.hand[seat]
        gold = hand[GOLD]
        board = self.board
        for slot in range(BOARD_SLOTS):
            card = board[slot]
            if card < 0:
                continue
            short = 0
            for colour, amount in CPAIRS[card]:
                gap = amount - bonus[colour] - hand[colour]
                if gap > 0:
                    short += gap
            if short <= gold:
                out.append(A_BUY + slot)
        held = self.reserved[seat]
        for i, card in enumerate(held):
            if self.shortfall(seat, card) <= gold:
                out.append(A_BUY_RESERVED + i)

        if len(held) < MAX_RESERVED:
            for slot in range(BOARD_SLOTS):
                if board[slot] >= 0:
                    out.append(A_RESERVE + slot)
            for tier in range(TIERS):
                if self.decks[tier]:
                    out.append(A_RESERVE_DECK + tier)

        if not out:
            out.append(A_PASS)
        return out

    # ------------------------------------------------------------------ play
    def play(self, move) -> None:
        move = int(move)
        seat = self.to_move
        self._legal = None
        self._mask = None
        self._legal_np = None
        self.move_count += 1

        if self.phase == PHASE_DISCARD:
            colour = move - A_DISCARD
            self.hand[seat][colour] -= 1
            self.bank[colour] += 1
            if sum(self.hand[seat]) <= TOKEN_LIMIT:
                self.phase = PHASE_MAIN
                self._end_turn()
            return

        if self.phase == PHASE_NOBLE:
            self._take_noble(seat, self.nobles[move - A_NOBLE])
            self.phase = PHASE_MAIN
            self._end_turn()
            return

        if move >= A_BUY:
            if move < A_BUY_RESERVED:
                slot = move - A_BUY
                card = self.board[slot]
                self._pay(seat, card)
                self._gain_card(seat, card)
                self._refill(slot)
                self._after_card(seat)
                return
            if move < A_RESERVE:
                index = move - A_BUY_RESERVED
                card = self.reserved[seat].pop(index)
                self.hidden[seat].pop(index)
                self._pay(seat, card)
                self._gain_card(seat, card)
                self._after_card(seat)
                return
            if move < A_RESERVE_DECK:
                slot = move - A_RESERVE
                self._reserve(seat, self.board[slot], False)
                self._refill(slot)
            elif move < A_DISCARD:
                tier = move - A_RESERVE_DECK
                deck = self.decks[tier]
                card = deck.pop() if self.draw is None else pop_random(deck, self.draw)
                self._reserve(seat, card, True)
            else:  # A_PASS
                self._end_turn()
                return
        elif move < A_TAKE2D:
            self._take(seat, TAKE3[move])
        elif move < A_TAKE1:
            self._take(seat, TAKE2D[move - A_TAKE2D])
        elif move < A_TAKE2S:
            colour = move - A_TAKE1
            self.bank[colour] -= 1
            self.hand[seat][colour] += 1
        else:
            colour = move - A_TAKE2S
            self.bank[colour] -= 2
            self.hand[seat][colour] += 2

        if sum(self.hand[seat]) > TOKEN_LIMIT:
            self.phase = PHASE_DISCARD
        else:
            self._end_turn()

    def _take(self, seat: int, colours) -> None:
        hand = self.hand[seat]
        bank = self.bank
        for colour in colours:
            bank[colour] -= 1
            hand[colour] += 1

    def _reserve(self, seat: int, card: int, hidden: bool) -> None:
        self.reserved[seat].append(int(card))
        self.hidden[seat].append(hidden)
        if self.bank[GOLD] > 0:
            self.bank[GOLD] -= 1
            self.hand[seat][GOLD] += 1

    def _take_noble(self, seat: int, noble: int) -> None:
        self.nobles.remove(noble)
        self.points[seat] += 3

    def _after_card(self, seat: int) -> None:
        qualifying = self._qualifying_nobles(seat)
        if len(qualifying) == 1:
            self._take_noble(seat, qualifying[0])
        elif len(qualifying) > 1:
            self.phase = PHASE_NOBLE
            self.pending = qualifying
            return
        self._end_turn()

    def _end_turn(self) -> None:
        self.phase = PHASE_MAIN
        self.pending = []
        if max(self.points) >= WIN_POINTS:
            self.triggered = True
        self.to_move = self.to_move % self.players + 1
        if (self.triggered and self.to_move == 1) or self.move_count >= MAX_PLIES:
            self._finish()

    def _finish(self) -> None:
        best = self.standings()[0]
        key = self.rank_key(best)
        tied = sum(1 for s in range(1, self.players + 1)
                   if self.rank_key(s) == key)
        self.winner = int(best) if tied == 1 else 0
        self.to_move = 0
        self._legal = None
        self._mask = None
        self._legal_np = None

    def is_terminal(self) -> bool:
        return self.to_move == 0

    # ---------------------------------------------------------------- result
    def standings(self) -> list[int]:
        return sorted(range(1, self.players + 1), key=self.rank_key)

    def rank_key(self, seat: int) -> tuple[int, int]:
        return (-self.points[seat], self.ncards[seat])

    def result_vector(self) -> np.ndarray:
        """Per-seat score in ``[-1, +1]``, index 0 unused.  As v1's, exactly."""
        out = np.zeros(self.players + 1, dtype=np.float32)
        if self.to_move != 0:
            return out
        keys = {s: self.rank_key(s) for s in range(1, self.players + 1)}
        denom = self.players - 1
        for seat, key in keys.items():
            beaten = sum(1 for o, k in keys.items() if o != seat and k > key)
            tied = sum(1 for o, k in keys.items() if o != seat and k == key)
            out[seat] = 2.0 * ((beaten + 0.5 * tied) / denom) - 1.0
        return out

    def score_for(self, seat: int) -> float:
        if self.to_move != 0:
            return 0.5
        return float(self.result_vector()[seat] + 1.0) / 2.0

    # ------------------------------------------------- hidden information
    def determinize(self, seat: int, rng: random.Random) -> "Game":
        """A copy ``seat`` cannot tell from this one, with chance left unrolled.

        The decks are bags already, so there is nothing to shuffle: the only
        thing another seat knows that this one does not is the identity of a
        card it reserved off the top of a deck, and each of those is swapped for
        a uniform draw from the pool it came out of.  Usually there are none and
        this is a copy that has been told which RNG to draw refills from.
        """
        out = self.copy()
        out.draw = rng
        for other in range(1, self.players + 1):
            if other == seat:
                continue
            secret = out.hidden[other]
            if not any(secret):
                continue
            held = out.reserved[other]
            for i, is_hidden in enumerate(secret):
                if not is_hidden:
                    continue
                deck = out.decks[CTIER[held[i]]]
                j = rng.randrange(len(deck) + 1)
                if j < len(deck):
                    held[i], deck[j] = deck[j], held[i]
        return out

    # ---------------------------------------------------------------- naming
    def move_label(self, move) -> str:
        move = int(move)

        def names(gems) -> str:
            return "+".join(GEM_NAMES[g] for g in gems)

        if move < A_TAKE2D:
            return f"take {names(TAKE3[move])}"
        if move < A_TAKE1:
            return f"take {names(TAKE2D[move - A_TAKE2D])}"
        if move < A_TAKE2S:
            return f"take {GEM_NAMES[move - A_TAKE1]}"
        if move < A_BUY:
            return f"take 2 {GEM_NAMES[move - A_TAKE2S]}"
        if move < A_BUY_RESERVED:
            slot = move - A_BUY
            card = self.board[slot]
            return f"buy {card_label(card)}" if card >= 0 else f"buy slot {slot}"
        if move < A_RESERVE:
            index = move - A_BUY_RESERVED
            held = self.reserved[self.to_move] if self.to_move else []
            if index < len(held):
                return f"buy reserved {card_label(held[index])}"
            return f"buy reserved #{index + 1}"
        if move < A_RESERVE_DECK:
            slot = move - A_RESERVE
            card = self.board[slot]
            return f"reserve {card_label(card)}" if card >= 0 else f"reserve slot {slot}"
        if move < A_DISCARD:
            return f"reserve from tier {move - A_RESERVE_DECK + 1} deck"
        if move < A_NOBLE:
            return f"return {GEM_NAMES[move - A_DISCARD]}"
        if move < A_PASS:
            index = move - A_NOBLE
            if index < len(self.nobles):
                return f"take {noble_label(self.nobles[index])}"
            return f"take noble #{index + 1}"
        return "pass"


# ------------------------------------------------------------------ helpers
def pop_random(deck: list[int], rng: random.Random) -> int:
    """Remove and return a uniform element.  O(1): the last one fills the hole."""
    j = rng.randrange(len(deck))
    last = deck.pop()
    if j == len(deck):
        return last
    card = deck[j]
    deck[j] = last
    return card


def as_py_random(rng) -> random.Random:
    """Accept either RNG the project uses, and hand back the fast one.

    ``numpy.random.Generator`` costs about 1.5 microseconds a draw, which is
    more than a whole legality mask here; :class:`random.Random` costs a fifth
    of that, and this file draws once per refill.
    """
    if rng is None:
        return random.Random()
    if isinstance(rng, random.Random):
        return rng
    if isinstance(rng, (int, np.integer)):
        return random.Random(int(rng))
    return random.Random(int(rng.integers(1 << 62)))
