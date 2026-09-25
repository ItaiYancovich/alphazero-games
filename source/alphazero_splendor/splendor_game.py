"""Splendor: rules, the fixed action space, and belief sampling.

Three things make this game a different shape from the other three here, and
each of them shows up below.

**More than two players.**  Seats are numbered ``1..players`` -- the same
one-based convention Hex and Connect Four use for their two colours -- and a
result is a *vector*, one score per seat, not a single number that flips sign
going up the tree.  Two-player Splendor is the special case where that vector is
``(+1, -1)``.

**A turn is not always one decision.**  Taking tokens can put a player over the
ten-token limit, and buying a card can qualify them for more than one noble.
Both are choices the rules give the *same* player before their turn ends, so the
position carries a :data:`PHASE_DISCARD` / :data:`PHASE_NOBLE` sub-turn in which
``to_move`` does not advance.  Folding them into the main move instead would
multiply the action space by every combination of tokens returned.

**Chance and hidden information.**  The decks are shuffled, so the card that
replaces a bought one is unknown, and a card reserved off the top of a deck is
unknown to everyone but its owner.  The shuffle is realised *once*, at setup:
``play`` is then a pure function, which is what lets the search copy positions
cheaply.  A searcher must not read the shuffle it cannot see, so
:meth:`SplendorGame.determinize` re-shuffles everything one seat does not know
into a position consistent with what it does -- and the search samples a fresh
determinization per simulation.

The action space is deliberately *fixed*: a move names a board **slot**, not a
card.  Which card sits in a slot changes with the shuffle; that there are twelve
slots does not.  So the policy head has a stable 72 outputs whatever the deck
does, and only the legality mask moves.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .cards import (BONUS, CARDS, COST, DECKS, GEM_NAMES, GEMS, GOLD, NNOBLES,
                    NOBLE_NEED, POINTS, TIERS, TOKENS, card_label, noble_label)

# ------------------------------------------------------------------- setup
ROW = 4                 # face-up cards per tier
MAX_RESERVED = 3
TOKEN_LIMIT = 10        # tokens in hand at the end of a turn
GOLD_SUPPLY = 5
WIN_POINTS = 15
TAKE_SAME_MIN = 4       # a pile must hold this many before you may take two

# Tokens of each colour in the bank, by player count.  Gold is always 5.
SUPPLY = {2: 4, 3: 5, 4: 7}
MIN_PLAYERS, MAX_PLAYERS = 2, 4
MAX_SEATS = MAX_PLAYERS  # what the network's value head is sized for

# A game that has gone this long is not going to finish.  Unreachable in
# practice (a real game is 25-35 turns each) but self-play must terminate even
# when an early network learns to pass forever.
MAX_PLIES = 600

# ------------------------------------------------------------- the sub-turns
PHASE_MAIN = 0
PHASE_DISCARD = 1   # over the token limit; return one token per move
PHASE_NOBLE = 2     # more than one noble qualifies; pick one

# --------------------------------------------------------- the action space
TAKE3 = tuple(combinations(range(GEMS), 3))    # 10
TAKE2D = tuple(combinations(range(GEMS), 2))   # 10

A_TAKE3 = 0
A_TAKE2D = A_TAKE3 + len(TAKE3)          # 10
A_TAKE1 = A_TAKE2D + len(TAKE2D)         # 20
A_TAKE2S = A_TAKE1 + GEMS                # 25
A_BUY = A_TAKE2S + GEMS                  # 30  .. 41  board slot
A_BUY_RESERVED = A_BUY + TIERS * ROW     # 42  .. 44  reserved index
A_RESERVE = A_BUY_RESERVED + MAX_RESERVED        # 45 .. 56  board slot
A_RESERVE_DECK = A_RESERVE + TIERS * ROW         # 57 .. 59  deck top
A_DISCARD = A_RESERVE_DECK + TIERS               # 60 .. 65  token colour
A_NOBLE = A_DISCARD + TOKENS                     # 66 .. 70  noble index
A_PASS = A_NOBLE + 5                             # 71
NACTIONS = A_PASS + 1                            # 72


def slot_of(tier: int, index: int) -> int:
    return tier * ROW + index


def tier_slot(slot: int) -> tuple[int, int]:
    return divmod(slot, ROW)


class SplendorGame:
    """One Splendor position, for two to four players.

    Everything a seat may legally know is in plain attributes; the two things it
    may not -- deck order, and the identity of another seat's deck-reserved
    cards -- are here too, because the *server* holds the real game.  A search
    is handed a :meth:`determinize` of the position instead.
    """

    __slots__ = ("players", "to_move", "move_count", "phase", "bank", "hand",
                 "bonus", "points", "ncards", "reserved", "hidden", "board",
                 "decks", "nobles", "pending", "triggered", "winner",
                 "_legal_cache")

    # ------------------------------------------------------------------ setup
    def __init__(self, players: int = 2, rng: np.random.Generator | None = None,
                 _blank: bool = False):
        if not MIN_PLAYERS <= players <= MAX_PLAYERS:
            raise ValueError(f"Splendor is for {MIN_PLAYERS}-{MAX_PLAYERS} players")
        self.players = int(players)
        if _blank:
            return
        rng = rng if rng is not None else np.random.default_rng()

        self.to_move = 1
        self.move_count = 0
        self.phase = PHASE_MAIN
        self.winner = 0
        self.triggered = False
        self.pending: list[int] = []
        self._legal_cache: np.ndarray | None = None

        # Seat-indexed arrays are one longer than the table and row 0 is never
        # used, so a seat number indexes directly and no code has to remember
        # whether it is holding a seat or an offset.
        n = self.players + 1
        self.bank = np.zeros(TOKENS, dtype=np.int16)
        self.bank[:GEMS] = SUPPLY[self.players]
        self.bank[GOLD] = GOLD_SUPPLY
        self.hand = np.zeros((n, TOKENS), dtype=np.int16)
        self.bonus = np.zeros((n, GEMS), dtype=np.int16)
        self.points = np.zeros(n, dtype=np.int16)
        self.ncards = np.zeros(n, dtype=np.int16)
        self.reserved: list[list[int]] = [[] for _ in range(n)]
        # Parallel to ``reserved``: was this card taken off a deck, and so
        # unknown to everyone else?  A card reserved off the board is public.
        self.hidden: list[list[bool]] = [[] for _ in range(n)]

        self.decks = [[int(c) for c in rng.permutation(DECKS[t])] for t in range(TIERS)]
        self.board = np.full(TIERS * ROW, -1, dtype=np.int16)
        for tier in range(TIERS):
            for i in range(ROW):
                self._refill(slot_of(tier, i))
        self.nobles = [int(i) for i in rng.permutation(NNOBLES)[:self.players + 1]]

    def copy(self) -> "SplendorGame":
        other = SplendorGame(self.players, _blank=True)
        other.to_move = self.to_move
        other.move_count = self.move_count
        other.phase = self.phase
        other.winner = self.winner
        other.triggered = self.triggered
        other.pending = list(self.pending)
        other._legal_cache = self._legal_cache
        other.bank = self.bank.copy()
        other.hand = self.hand.copy()
        other.bonus = self.bonus.copy()
        other.points = self.points.copy()
        other.ncards = self.ncards.copy()
        other.reserved = [list(r) for r in self.reserved]
        other.hidden = [list(h) for h in self.hidden]
        other.decks = [list(d) for d in self.decks]
        other.board = self.board.copy()
        other.nobles = list(self.nobles)
        return other

    # ------------------------------------------------------------- internals
    def _refill(self, slot: int) -> None:
        """Put the top of the slot's deck face up, or leave the slot empty."""
        tier, _ = tier_slot(slot)
        deck = self.decks[tier]
        self.board[slot] = deck.pop(0) if deck else -1

    def token_shortfall(self, seat: int, card_id: int) -> int:
        """Coloured tokens ``seat`` is missing for ``card_id``, before spending gold.

        Bonuses are a discount and are already taken off.  Gold is *not*: a
        caller that wants "can this be bought" compares this against the gold in
        hand, and one that wants "how far off is it" subtracts the gold instead.
        """
        need = np.maximum(COST[card_id] - self.bonus[seat], 0)
        return int(np.maximum(need - self.hand[seat, :GEMS], 0).sum())

    def can_afford(self, seat: int, card_id: int) -> bool:
        return self.token_shortfall(seat, card_id) <= int(self.hand[seat, GOLD])

    def _shortfall(self, seat: int, card_id: int) -> int:
        """Gold needed to buy ``card_id``, or -1 if even gold cannot cover it."""
        short = self.token_shortfall(seat, card_id)
        return short if short <= int(self.hand[seat, GOLD]) else -1

    def _pay(self, seat: int, card_id: int) -> None:
        need = np.maximum(COST[card_id] - self.bonus[seat], 0)
        spend = np.minimum(need, self.hand[seat, :GEMS])
        gold = int((need - spend).sum())
        self.hand[seat, :GEMS] -= spend
        self.hand[seat, GOLD] -= gold
        self.bank[:GEMS] += spend
        self.bank[GOLD] += gold

    def _gain_card(self, seat: int, card_id: int) -> None:
        self.bonus[seat, BONUS[card_id]] += 1
        self.points[seat] += POINTS[card_id]
        self.ncards[seat] += 1

    def _qualifying_nobles(self, seat: int) -> list[int]:
        return [n for n in self.nobles
                if bool((self.bonus[seat] >= NOBLE_NEED[n]).all())]

    def _held_tokens(self, seat: int) -> int:
        return int(self.hand[seat].sum())

    # ------------------------------------------------------------- legality
    def legal_mask(self) -> np.ndarray:
        """Boolean mask over the whole action space, cached per position."""
        if self._legal_cache is None:
            self._legal_cache = self._compute_mask()
        return self._legal_cache

    def _compute_mask(self) -> np.ndarray:
        mask = np.zeros(NACTIONS, dtype=bool)
        if self.is_terminal():
            return mask
        seat = self.to_move

        if self.phase == PHASE_DISCARD:
            mask[A_DISCARD:A_DISCARD + TOKENS] = self.hand[seat] > 0
            return mask
        if self.phase == PHASE_NOBLE:
            for i, noble in enumerate(self.nobles):
                if noble in self.pending:
                    mask[A_NOBLE + i] = True
            return mask

        # --- taking tokens.  The rules say "three of different colours"; with
        # fewer than three colours left in the bank you take what there is, so
        # the two- and one-token takes are legal only when three is impossible.
        available = int((self.bank[:GEMS] > 0).sum())
        if available >= 3:
            for i, trio in enumerate(TAKE3):
                if all(self.bank[g] > 0 for g in trio):
                    mask[A_TAKE3 + i] = True
        elif available == 2:
            for i, pair in enumerate(TAKE2D):
                if all(self.bank[g] > 0 for g in pair):
                    mask[A_TAKE2D + i] = True
        elif available == 1:
            for g in range(GEMS):
                if self.bank[g] > 0:
                    mask[A_TAKE1 + g] = True
        for g in range(GEMS):
            if self.bank[g] >= TAKE_SAME_MIN:
                mask[A_TAKE2S + g] = True

        # --- buying, from the board and from the hand
        for slot in range(TIERS * ROW):
            card = int(self.board[slot])
            if card >= 0 and self._shortfall(seat, card) >= 0:
                mask[A_BUY + slot] = True
        for i, card in enumerate(self.reserved[seat]):
            if self._shortfall(seat, card) >= 0:
                mask[A_BUY_RESERVED + i] = True

        # --- reserving.  Costs nothing but a hand slot; the gold is a bonus
        # when there is any left, not a requirement.
        if len(self.reserved[seat]) < MAX_RESERVED:
            for slot in range(TIERS * ROW):
                if self.board[slot] >= 0:
                    mask[A_RESERVE + slot] = True
            for tier in range(TIERS):
                if self.decks[tier]:
                    mask[A_RESERVE_DECK + tier] = True

        if not mask.any():
            mask[A_PASS] = True  # nothing to take, nothing to buy, hand full
        return mask

    def legal_moves(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask()).astype(np.int64)

    def is_legal(self, move) -> bool:
        move = int(move)
        return 0 <= move < NACTIONS and bool(self.legal_mask()[move])

    # ------------------------------------------------------------------ play
    def play(self, move) -> None:
        move = int(move)
        if not self.is_legal(move):
            raise ValueError(f"illegal move {move} ({self.move_label(move)})")
        seat = self.to_move
        self._legal_cache = None
        self.move_count += 1

        if self.phase == PHASE_DISCARD:
            colour = move - A_DISCARD
            self.hand[seat, colour] -= 1
            self.bank[colour] += 1
            if self._held_tokens(seat) <= TOKEN_LIMIT:
                self.phase = PHASE_MAIN
                self._end_turn()
            return

        if self.phase == PHASE_NOBLE:
            noble = self.nobles[move - A_NOBLE]
            self._take_noble(seat, noble)
            self.phase = PHASE_MAIN
            self._end_turn()
            return

        if move == A_PASS:
            self._end_turn()
            return

        if A_TAKE3 <= move < A_TAKE2D:
            self._take(seat, TAKE3[move - A_TAKE3])
        elif A_TAKE2D <= move < A_TAKE1:
            self._take(seat, TAKE2D[move - A_TAKE2D])
        elif A_TAKE1 <= move < A_TAKE2S:
            self._take(seat, (move - A_TAKE1,))
        elif A_TAKE2S <= move < A_BUY:
            colour = move - A_TAKE2S
            self._take(seat, (colour, colour))
        elif A_BUY <= move < A_BUY_RESERVED:
            slot = move - A_BUY
            card = int(self.board[slot])
            self._pay(seat, card)
            self._gain_card(seat, card)
            self._refill(slot)
            return self._after_card(seat)
        elif A_BUY_RESERVED <= move < A_RESERVE:
            index = move - A_BUY_RESERVED
            card = self.reserved[seat].pop(index)
            self.hidden[seat].pop(index)
            self._pay(seat, card)
            self._gain_card(seat, card)
            return self._after_card(seat)
        elif A_RESERVE <= move < A_RESERVE_DECK:
            slot = move - A_RESERVE
            self._reserve(seat, int(self.board[slot]), hidden=False)
            self._refill(slot)
        elif A_RESERVE_DECK <= move < A_DISCARD:
            tier = move - A_RESERVE_DECK
            self._reserve(seat, self.decks[tier].pop(0), hidden=True)
        else:  # pragma: no cover - the mask above admits nothing else
            raise ValueError(f"unhandled move {move}")

        self._check_limit(seat)

    def _take(self, seat: int, colours) -> None:
        for colour in colours:
            self.bank[colour] -= 1
            self.hand[seat, colour] += 1

    def _reserve(self, seat: int, card_id: int, hidden: bool) -> None:
        self.reserved[seat].append(int(card_id))
        self.hidden[seat].append(bool(hidden))
        if self.bank[GOLD] > 0:
            self.bank[GOLD] -= 1
            self.hand[seat, GOLD] += 1

    def _take_noble(self, seat: int, noble: int) -> None:
        self.nobles.remove(noble)
        self.points[seat] += 3

    def _after_card(self, seat: int) -> None:
        """A card was gained: award a noble, then finish the turn.

        One qualifying noble is not a decision, so it is taken outright; two or
        more is, and the same seat chooses before the turn ends.  Buying never
        gains tokens, so the limit cannot be exceeded here.
        """
        qualifying = self._qualifying_nobles(seat)
        if len(qualifying) == 1:
            self._take_noble(seat, qualifying[0])
        elif len(qualifying) > 1:
            self.phase = PHASE_NOBLE
            self.pending = qualifying
            return
        self._end_turn()

    def _check_limit(self, seat: int) -> None:
        if self._held_tokens(seat) > TOKEN_LIMIT:
            self.phase = PHASE_DISCARD
            return
        self._end_turn()

    def _end_turn(self) -> None:
        self.phase = PHASE_MAIN
        self.pending = []
        if int(self.points.max()) >= WIN_POINTS:
            # The round is played out so that every seat has had the same
            # number of turns; the game ends when the turn comes back round.
            self.triggered = True
        self.to_move = self.to_move % self.players + 1
        if (self.triggered and self.to_move == 1) or self.move_count >= MAX_PLIES:
            self._finish()

    def _finish(self) -> None:
        best = self.standings()[0]
        # A tie on prestige is broken by the fewest development cards; if that
        # ties too the game really is drawn, and ``winner`` stays 0.
        tied = [s for s in range(1, self.players + 1)
                if self.rank_key(s) == self.rank_key(best)]
        self.winner = int(best) if len(tied) == 1 else 0
        self.to_move = 0
        self._legal_cache = None

    def is_terminal(self) -> bool:
        return self.to_move == 0

    # ---------------------------------------------------------------- result
    def standings(self) -> list[int]:
        """Seats best first: most prestige, then fewest development cards."""
        return sorted(range(1, self.players + 1), key=self.rank_key)

    def rank_key(self, seat: int) -> tuple[int, int]:
        """Sort key, smallest is best: most prestige, then fewest cards."""
        return (-int(self.points[seat]), int(self.ncards[seat]))

    def result_vector(self) -> np.ndarray:
        """Per-seat score in ``[-1, +1]``; index 0 is unused, as everywhere here.

        A seat scores by how many opponents it finished ahead of, rescaled so
        the outright winner of any table gets ``+1`` and the outright last
        ``-1``.  With two seats that is exactly the win/loss the other games
        use; with four it keeps second place worth more than fourth, which is
        what stops a losing network from playing as if every non-win were the
        same.  The vector sums to zero, so nothing is created by finishing.
        """
        out = np.zeros(self.players + 1, dtype=np.float32)
        if not self.is_terminal():
            return out
        keys = {s: self.rank_key(s) for s in range(1, self.players + 1)}
        for seat, key in keys.items():
            beaten = sum(1 for other, k in keys.items() if other != seat and k > key)
            tied = sum(1 for other, k in keys.items() if other != seat and k == key)
            share = (beaten + 0.5 * tied) / (self.players - 1)
            out[seat] = 2.0 * share - 1.0
        return out

    def score_for(self, seat: int) -> float:
        """The same, as a 0..1 score -- what the arena and the ratings want."""
        if not self.is_terminal():
            return 0.5
        return float(self.result_vector()[seat] + 1.0) / 2.0

    # ------------------------------------------------- hidden information
    def unseen_by(self, seat: int) -> list[list[int]]:
        """Every card ``seat`` cannot see, split by tier.

        A tier's deck, plus the cards other seats reserved off *that* deck --
        drawn off the top without anyone else looking.  Split by tier because
        which deck a card came off is public even when the card is not: the
        pool a hidden card is resampled from is only ever its own tier.
        """
        pool: list[list[int]] = [list(self.decks[t]) for t in range(TIERS)]
        for other in range(1, self.players + 1):
            if other == seat:
                continue
            for card, is_hidden in zip(self.reserved[other], self.hidden[other]):
                if is_hidden:
                    pool[CARDS[card].tier].append(int(card))
        return pool

    def determinize(self, seat: int, rng: np.random.Generator) -> "SplendorGame":
        """A position consistent with what ``seat`` knows, with the rest resampled.

        Deck order is unknown to everyone, and a card another seat reserved off
        a deck is unknown to all but its owner.  Both are drawn from the unseen
        cards *of that tier*, which is what keeps a search from reading a
        shuffle it has no right to while still respecting what it did see.
        Everything else -- tokens, bonuses, board, nobles, and this seat's
        *own* reserved cards -- is public or its own, and carries over
        untouched.
        """
        out = self.copy()
        pool = [[int(c) for c in rng.permutation(tier_pool)] if tier_pool else []
                for tier_pool in self.unseen_by(seat)]
        for other in range(1, self.players + 1):
            if other == seat:
                continue
            for i, is_hidden in enumerate(out.hidden[other]):
                if is_hidden:
                    tier = CARDS[out.reserved[other][i]].tier
                    out.reserved[other][i] = pool[tier].pop()
        # What is left of each tier's pool is that tier's deck, in a fresh order.
        out.decks = [list(pool[t]) for t in range(TIERS)]
        out._legal_cache = None
        return out

    # ------------------------------------------------------------- identity
    def _common(self) -> list[bytes]:
        """The parts of the position every seat can see."""
        return [np.int16([self.to_move, self.phase, self.move_count,
                          int(self.triggered)]).tobytes(),
                self.bank.tobytes(), self.hand.tobytes(), self.bonus.tobytes(),
                self.points.tobytes(), self.ncards.tobytes(),
                self.board.tobytes(),
                np.int16(sorted(self.nobles)).tobytes(),
                np.int16([len(d) for d in self.decks]).tobytes()]

    def signature(self) -> bytes:
        """The whole position, deck order aside -- the server's own identity for it.

        Includes every seat's reserved cards, hidden ones and all, because this
        is what the *server* holds and what "has the position changed" has to
        mean there.  A searcher wants :meth:`public_signature` instead.
        """
        parts = self._common()
        for seat in range(1, self.players + 1):
            parts.append(np.int16(self.reserved[seat]).tobytes())
        return b"|".join(parts)

    def public_signature(self, seat: int) -> bytes:
        """The position as ``seat`` knows it: its own hand, and nobody else's secrets.

        Two positions with the same public signature are indistinguishable to
        ``seat``, which is exactly what :meth:`determinize` must preserve.
        """
        parts = self._common()
        for other in range(1, self.players + 1):
            if other == seat:
                parts.append(np.int16(self.reserved[other]).tobytes())
            else:
                # A face-up reserve is public.  A deck reserve is not, but
                # which deck it came off is -- so its tier survives and its
                # identity does not.
                parts.append(np.int16([c if not h else -1 - CARDS[c].tier
                                       for c, h in zip(self.reserved[other],
                                                       self.hidden[other])]).tobytes())
        return b"|".join(parts)

    # ---------------------------------------------------------------- naming
    def move_label(self, move) -> str:
        """What a move is called in the move list.  Never raises."""
        move = int(move)

        def names(gems) -> str:
            return "+".join(GEM_NAMES[g] for g in gems)

        if A_TAKE3 <= move < A_TAKE2D:
            return f"take {names(TAKE3[move - A_TAKE3])}"
        if A_TAKE2D <= move < A_TAKE1:
            return f"take {names(TAKE2D[move - A_TAKE2D])}"
        if A_TAKE1 <= move < A_TAKE2S:
            return f"take {GEM_NAMES[move - A_TAKE1]}"
        if A_TAKE2S <= move < A_BUY:
            return f"take 2 {GEM_NAMES[move - A_TAKE2S]}"
        if A_BUY <= move < A_BUY_RESERVED:
            slot = move - A_BUY
            card = int(self.board[slot])
            return f"buy {card_label(card)}" if card >= 0 else f"buy slot {slot}"
        if A_BUY_RESERVED <= move < A_RESERVE:
            index = move - A_BUY_RESERVED
            held = self.reserved[self.to_move] if self.to_move else []
            if index < len(held):
                return f"buy reserved {card_label(held[index])}"
            return f"buy reserved #{index + 1}"
        if A_RESERVE <= move < A_RESERVE_DECK:
            slot = move - A_RESERVE
            card = int(self.board[slot])
            return f"reserve {card_label(card)}" if card >= 0 else f"reserve slot {slot}"
        if A_RESERVE_DECK <= move < A_DISCARD:
            return f"reserve from tier {move - A_RESERVE_DECK + 1} deck"
        if A_DISCARD <= move < A_NOBLE:
            return f"return {GEM_NAMES[move - A_DISCARD]}"
        if A_NOBLE <= move < A_PASS:
            index = move - A_NOBLE
            if index < len(self.nobles):
                return f"take {noble_label(self.nobles[index])}"
            return f"take noble #{index + 1}"
        return "pass"


def move_to_str(move, board: SplendorGame) -> str:
    return board.move_label(move)
