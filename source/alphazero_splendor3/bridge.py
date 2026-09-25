"""Converting the reference game into this one.

The GUI, the ladder and the tournament all hold an
:class:`alphazero_splendor.splendor_game.SplendorGame`, and there is no reason to
rewrite them: every implementation here agrees on the rules, on the 72-action
numbering and on what a result vector means, so a v3 agent can take a v1
position, convert it, search, and hand back a move index that means the same
thing on either side.

One thing v3 needs that v2 did not: the running **deck summary**.  It is
maintained incrementally during play, so a position built from the outside has
to have it computed once, here, from the decks it was handed.  Getting that
wrong would not raise -- it would quietly feed the network a description of a
deck that is not on the table -- so ``tests/test_splendor3.py`` recounts it.
"""

from __future__ import annotations

from alphazero_splendor.splendor_game import SplendorGame

from .game import DECK_AGG_W, TIERS, Game, _agg_add


def from_v1(src: SplendorGame) -> Game:
    """A v3 position holding exactly what ``src`` holds.

    Deck *order* is dropped, because v3 has none: a deck is a bag and a card is
    drawn when a slot refills.  For a search that is the correct treatment; for
    a server it means a converted position will not deal the same next card as
    the original would have, which is why the GUI keeps its own v1 game and
    converts a copy for the agent rather than the other way round.
    """
    out = Game(int(src.players), _blank=True)
    out.to_move = int(src.to_move)
    out.move_count = int(src.move_count)
    out.phase = int(src.phase)
    out.winner = int(src.winner)
    out.triggered = bool(src.triggered)
    out.pending = [int(p) for p in src.pending]
    out.bank = [int(x) for x in src.bank]
    out.hand = [[int(x) for x in row] for row in src.hand]
    out.bonus = [[int(x) for x in row] for row in src.bonus]
    out.points = [int(x) for x in src.points]
    out.ncards = [int(x) for x in src.ncards]
    out.reserved = [[int(c) for c in row] for row in src.reserved]
    out.hidden = [[bool(h) for h in row] for row in src.hidden]
    out.board = [int(c) for c in src.board]
    out.decks = [[int(c) for c in deck] for deck in src.decks]
    out.nobles = [int(n) for n in src.nobles]
    out.dagg = [[0] * DECK_AGG_W for _ in range(TIERS)]
    for tier in range(TIERS):
        agg = out.dagg[tier]
        for card in out.decks[tier]:
            _agg_add(agg, card, 1)
    return out
