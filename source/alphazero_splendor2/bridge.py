"""Converting between the reference game and this one.

The GUI, the ladder and the tournament code all hold an
:class:`alphazero_splendor.splendor_game.SplendorGame`, and there is no reason
to rewrite them: the two implementations agree on the rules, on the 72-action
numbering, and on what a result vector means, so a v2 agent can take a v1
position, convert it, search, and hand back a move index that means the same
thing on either side.

The conversion is also how the rules are *tested*.  ``tests/test_splendor2.py``
walks a v1 game and a converted v2 game in step and asserts that the legal sets
and every visible counter agree at every ply -- which is a much sharper check
than any invariant either implementation could assert about itself.
"""

from __future__ import annotations

from alphazero_splendor.splendor_game import SplendorGame

from .game import Game


def from_v1(src: SplendorGame) -> Game:
    """A v2 position holding exactly what ``src`` holds.

    Deck *order* is dropped, because v2 has none: a deck is a bag and a card is
    drawn from it when a slot refills.  For a search that is the correct
    treatment; for a server it means a converted position will not deal the same
    next card as the original would have, which is why the GUI keeps its own v1
    game and converts a copy for the agent rather than the other way round.
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
    return out


def to_v1(src: Game) -> SplendorGame:
    """The inverse, for tests and for anything that wants the reference type."""
    import numpy as np

    out = SplendorGame(src.players, _blank=True)
    out.to_move = int(src.to_move)
    out.move_count = int(src.move_count)
    out.phase = int(src.phase)
    out.winner = int(src.winner)
    out.triggered = bool(src.triggered)
    out.pending = list(src.pending)
    out._legal_cache = None
    out.bank = np.asarray(src.bank, dtype=np.int16)
    out.hand = np.asarray(src.hand, dtype=np.int16)
    out.bonus = np.asarray(src.bonus, dtype=np.int16)
    out.points = np.asarray(src.points, dtype=np.int16)
    out.ncards = np.asarray(src.ncards, dtype=np.int16)
    out.reserved = [list(r) for r in src.reserved]
    out.hidden = [list(h) for h in src.hidden]
    out.board = np.asarray(src.board, dtype=np.int16)
    out.decks = [list(d) for d in src.decks]
    out.nobles = list(src.nobles)
    return out
