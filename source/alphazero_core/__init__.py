"""Game-independent AlphaZero machinery.

Everything in here is written against the :class:`~alphazero_core.state.GameState`
protocol rather than against a particular board, so the same search, network,
evaluator and rating code serves both games in this project:

* ``alphazero_hex``  -- Hex on an n x n rhombus, no draws.
* ``alphazero_c4``   -- Connect Four on a 6 x 7 grid, draws possible.

The split happened when Connect Four was added.  ``alphazero_hex.mcts``,
``.net`` and ``.evaluator`` still exist and still export the same names -- they
are thin re-exports now -- so existing checkpoints, scripts and imports keep
working unchanged.
"""
