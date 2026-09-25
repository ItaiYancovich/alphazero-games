"""AlphaZero for Connect Four.

Mirrors :mod:`alphazero_hex`: an engine, feature planes, a ladder of agents from
random up to the trained network, self-play training, a tournament and ratings.
The search, the network, the evaluator and the rating machinery are shared with
Hex through :mod:`alphazero_core`; what lives here is the parts that are
actually about Connect Four.

Two differences from Hex are load-bearing rather than cosmetic:

* **Draws exist.**  A full board with no line of four is a real result, so the
  search has a proven-draw state, the value target has a 0, and the arena and
  ratings count half points.
* **Moves are columns.**  A move is stored as the cell the disc lands in, which
  keeps the plain per-cell policy head, but only one cell per column is ever
  legal -- so the evaluator masks to the landing squares, not to empty ones.
"""
