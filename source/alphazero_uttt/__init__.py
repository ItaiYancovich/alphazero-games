"""AlphaZero for Ultimate Tic-Tac-Toe.

Mirrors :mod:`alphazero_c4`: an engine, feature planes, a ladder of agents from
random up to the trained network, self-play training, a tournament and ratings.
The search, the network, the evaluator and the rating machinery are shared with
Hex and Connect Four through :mod:`alphazero_core`; what lives here is the parts
that are actually about this game.

Three differences from Connect Four are load-bearing rather than cosmetic:

* **The legal set is not a function of the board.**  A move's slot sends the
  opponent to the small board of the same index, so a position is the marks
  *plus* where the last move pointed.  The canonical board therefore carries a
  fourth value for "empty but out of reach", and a feature plane spells the
  same thing out for the network.
* **There are two boards to win.**  Nine games of noughts and crosses decide a
  tenth, and three feature planes hand the trunk that upper board directly
  rather than making it rediscover a line of three from 81 cells every layer.
* **The full dihedral group.**  Nothing here picks a direction the way gravity
  does in Connect Four, so all eight symmetries of the square are available for
  augmentation -- four times Connect Four's mirror.

Draws are as real as they are in Connect Four -- every small board decided with
no line among them -- so the search's proven-draw state, the half point in the
arena, and the 0 in the value target all apply unchanged.
"""
