"""Splendor: the fourth game, and the first with more than two players.

What is different here, and why it needed new machinery in
:mod:`alphazero_core` rather than a fourth copy of the Hex package:

* **Two to four seats.**  The shared PUCT search negates the value going up the
  tree, which is only meaningful between two players.  :mod:`alphazero_core.vecmcts`
  backs up a *vector* of values instead, one per seat, and each node maximises
  the component belonging to whoever is on turn there.
* **Chance and hidden information.**  Shuffled decks mean the search must not
  read the card that has not been turned over yet, so it re-samples a
  determinization of everything the searching seat cannot see on every
  simulation.
* **A position is not a board.**  There is no grid and no translation
  invariance, so the network is a residual MLP over a feature *vector* with a
  fixed 72-action policy head -- :mod:`alphazero_core.vecnet`, which is written
  for any game that can describe itself that way rather than for this one.
"""
