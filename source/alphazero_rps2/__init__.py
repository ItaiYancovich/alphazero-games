"""AlphaZero for Intransitive (RPS2).

Chess-like movement on a 9x9 board, with capture decided by rock-paper-scissors
rather than by piece value: rock takes scissors, scissors takes paper, paper
takes rock, and equal types are mutually immovable walls.  Reach the enemy's
base corner to win.

The package mirrors :mod:`alphazero_uttt` -- an engine, feature planes, a ladder
of agents from random up to the trained network, self-play training, a
tournament and ratings -- and shares the search, the network, the evaluator and
the rating machinery with the other games through :mod:`alphazero_core`.

Three differences from every other board game here are load-bearing:

* **A move is a pair, not a square.**  Which piece, and which of eight
  directions, so the action space is ``81 x 8 = 648`` rather than 81.  The
  shared network grows one configuration field for it -- a policy head with
  eight output channels instead of one -- and nothing else changes: channel
  ``d`` over square ``s`` *is* the move "the piece on ``s`` goes direction
  ``d``", which is exactly the locality a convolution is good at.
* **No piece dominates.**  The capture relation is a 3-cycle, so material is
  not a scalar: three rocks are worth a great deal against scissors and
  nothing at all against paper.  Every heuristic here counts *matchups*, not
  pieces.
* **Pieces move.**  Every other game here only ever adds marks, so a position
  is monotone and a game is bounded by the board.  Here a position can repeat
  forever, which is why the game has a no-capture draw at 200 half-moves and
  why the stagnation counter is part of the position key.
"""
