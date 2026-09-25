"""Ultimate XX -- Ultimate Tic-Tac-Toe with one mark, played misere.

An engine, the knowledge the classical players need, and a ladder of agents from
random up to alpha-beta.  There is deliberately **no network here**: unlike the
other five games in this project, Ultimate XX ships with its classical agents
only, so nothing in this package imports torch and nothing needs a trained
checkpoint to be playable.

Three differences from Ultimate Tic-Tac-Toe are load-bearing rather than
cosmetic:

* **One mark.**  Both players place X, so a line of three inside a small board
  belongs to whoever *closed* it and not to whoever built it.  A cell that would
  close a line is therefore hot for both sides at once, and a small board whose
  every empty cell is hot cannot be entered safely by anybody.  There is no
  race to win a small board -- only a question of who is standing there when it
  ends.
* **Winning loses.**  Complete a line of three small boards and you have lost.
  Everything the other engines call a threat is a threat against yourself, and
  the terminal value is ``+1`` for the side to move rather than ``-1``: you can
  only lose on your own move, so a finished position is one the player on turn
  has just won without touching the board.
* **A small board cannot be drawn.**  Six marks is the most a 3x3 grid can hold
  without a line, so the seventh always closes one and every small board is
  claimed before it fills.  *Claimed* and *closed* are the same thing here, and
  the meta board fills up steadily whether anybody wants it to or not, which is
  what makes the game finite and reasonably short -- forty-odd plies.

The *game* can still be drawn: nine claimed boards with no line of three among
them is a real result, so the half point in the arena and in the ratings applies
exactly as it does in Connect Four.
"""
