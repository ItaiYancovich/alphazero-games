"""Hex game engine.

Board geometry
--------------
An ``n x n`` rhombus indexed by ``(row, col)``.  Each cell has (up to) six
neighbours::

        (r-1, c)   (r-1, c+1)
    (r, c-1)   [r, c]   (r, c+1)
        (r+1, c-1) (r+1, c)

BLACK owns the top and bottom edges (connects row 0 to row n-1).
WHITE owns the left and right edges (connects col 0 to col n-1).

Hex can never be drawn: once the board is full exactly one player owns a
connection.  Win detection is incremental via a union-find over the cells plus
four virtual nodes (TOP, BOTTOM, LEFT, RIGHT).
"""

from __future__ import annotations

import numpy as np

EMPTY = 0
BLACK = 1  # top <-> bottom
WHITE = 2  # left <-> right

_NEIGHBOUR_CACHE: dict[int, tuple[tuple[int, ...], ...]] = {}


def neighbour_table(n: int) -> tuple[tuple[int, ...], ...]:
    """Flat-index neighbour lists for an ``n x n`` hex board (cached)."""
    cached = _NEIGHBOUR_CACHE.get(n)
    if cached is not None:
        return cached
    offsets = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
    table = []
    for r in range(n):
        for c in range(n):
            nbs = []
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    nbs.append(rr * n + cc)
            table.append(tuple(nbs))
    out = tuple(table)
    _NEIGHBOUR_CACHE[n] = out
    return out


def other(player: int) -> int:
    return WHITE if player == BLACK else BLACK


class HexBoard:
    """Mutable Hex position with incremental (union-find) win detection."""

    __slots__ = ("n", "ncells", "board", "_parent", "to_move", "winner", "move_count", "_nei",
                 "swap_rule")

    def __init__(self, n: int = 11, swap_rule: bool = False):
        self.n = n
        self.ncells = n * n
        self.board = bytearray(self.ncells)
        # cells 0..N-1, then TOP, BOTTOM, LEFT, RIGHT
        self._parent = list(range(self.ncells + 4))
        self.to_move = BLACK
        self.winner = 0
        self.move_count = 0
        self._nei = neighbour_table(n)
        # The pie rule: after Black's first stone, White may take it over
        # instead of placing one.  Off by default, which is the game every
        # checkpoint before v3 was trained on.  The swap is move index
        # ``ncells`` -- one past the last cell -- see :meth:`play`.
        self.swap_rule = swap_rule

    # ---------------------------------------------------------------- basics
    @property
    def TOP(self) -> int:
        return self.ncells

    @property
    def BOTTOM(self) -> int:
        return self.ncells + 1

    @property
    def LEFT(self) -> int:
        return self.ncells + 2

    @property
    def RIGHT(self) -> int:
        return self.ncells + 3

    def copy(self) -> "HexBoard":
        new = HexBoard.__new__(HexBoard)
        new.n = self.n
        new.ncells = self.ncells
        new.board = bytearray(self.board)
        new._parent = self._parent[:]
        new.to_move = self.to_move
        new.winner = self.winner
        new.move_count = self.move_count
        new._nei = self._nei
        new.swap_rule = self.swap_rule
        return new

    @property
    def SWAP(self) -> int:
        """Move index of the pie-rule swap: one past the last cell."""
        return self.ncells

    def swap_available(self) -> bool:
        return self.swap_rule and self.move_count == 1 and self.winner == 0

    # ------------------------------------------------------------ union-find
    def _find(self, x: int) -> int:
        p = self._parent
        while p[x] != x:
            p[x] = p[p[x]]  # path halving
            x = p[x]
        return x

    def _union(self, a: int, b: int) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            # Bias virtual nodes (high indices) towards being roots; keeps the
            # trees shallow without carrying a separate rank array.
            if ra < rb:
                self._parent[ra] = rb
            else:
                self._parent[rb] = ra

    # ---------------------------------------------------------------- moves
    def legal_moves(self) -> np.ndarray:
        arr = np.frombuffer(bytes(self.board), dtype=np.uint8)
        moves = np.flatnonzero(arr == EMPTY).astype(np.int32)
        if self.swap_rule and self.move_count == 1 and self.winner == 0:
            moves = np.append(moves, np.int32(self.ncells))
        return moves

    def is_legal(self, move: int) -> bool:
        if move == self.ncells:
            return self.swap_available()
        return 0 <= move < self.ncells and self.board[move] == EMPTY and self.winner == 0

    def play(self, move: int) -> None:
        """Place a stone for the side to move and update win state."""
        if self.winner:
            raise ValueError("game already decided")
        if move == self.ncells:
            self._swap()
            return
        if self.board[move] != EMPTY:
            raise ValueError(f"cell {move} is occupied")
        player = self.to_move
        self.board[move] = player
        n = self.n
        r, c = divmod(move, n)
        for nb in self._nei[move]:
            if self.board[nb] == player:
                self._union(move, nb)
        if player == BLACK:
            if r == 0:
                self._union(move, self.TOP)
            if r == n - 1:
                self._union(move, self.BOTTOM)
            if self._find(self.TOP) == self._find(self.BOTTOM):
                self.winner = BLACK
        else:
            if c == 0:
                self._union(move, self.LEFT)
            if c == n - 1:
                self._union(move, self.RIGHT)
            if self._find(self.LEFT) == self._find(self.RIGHT):
                self.winner = WHITE
        self.move_count += 1
        self.to_move = other(player)

    def _swap(self) -> None:
        """White takes over Black's first stone.

        The stone at ``(r, c)`` becomes a White stone at ``(c, r)`` -- the same
        cell seen from the other player's edges -- and Black moves again.  So
        the position is exactly as if White had opened there, which is what
        makes Black's first move a real decision: an opening strong enough to
        win will simply be taken.
        """
        if not self.swap_available():
            raise ValueError("swap is only legal as the second move of a swap-rule game")
        first = self.board.index(BLACK)
        r, c = divmod(first, self.n)
        target = c * self.n + r
        self.board[first] = EMPTY
        self._parent = list(range(self.ncells + 4))
        self.board[target] = WHITE
        if r == 0:  # the new stone's column is the old stone's row
            self._union(target, self.LEFT)
        if r == self.n - 1:
            self._union(target, self.RIGHT)
        self.move_count += 1
        self.to_move = BLACK

    def is_terminal(self) -> bool:
        return self.winner != 0

    def terminal_value(self) -> float:
        """Value of a finished position for the side to move.

        Always -1: Hex cannot be drawn, so a position is only terminal because
        the move just played won, and whoever is on turn has lost.  The shared
        search asks for this because Connect Four can also end in a draw.
        """
        return -1.0

    def result_for(self, player: int) -> float:
        """+1 if ``player`` won, -1 if it lost, 0 while the game is running."""
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    # ------------------------------------------------------- representations
    def array(self) -> np.ndarray:
        return np.frombuffer(bytes(self.board), dtype=np.uint8).reshape(self.n, self.n)

    def canonical_board(self) -> np.ndarray:
        """n x n uint8 board seen from the side to move: 0 empty, 1 own, 2 opp.

        The side to move is *always* rendered as the player connecting top to
        bottom: when White is to move the board is transposed and the colours
        swapped.  A network therefore only ever has to learn one point of view,
        which halves the amount of self-play needed.
        """
        return canonicalise(self.array(), self.to_move)

    def canonical_planes(self, in_planes: int | None = None) -> np.ndarray:
        """``in_planes x n x n`` float32 input tensor for the network."""
        from .features import PLANES, planes_from_boards

        return planes_from_boards(
            self.canonical_board()[None], PLANES if in_planes is None else in_planes
        )[0]

    def to_canonical_move(self, move: int) -> int:
        """Real board move -> index in the canonical (side-to-move) frame."""
        if self.to_move == BLACK or move == self.ncells:
            return move
        r, c = divmod(move, self.n)
        return c * self.n + r

    def from_canonical_move(self, move: int) -> int:
        """Canonical-frame index -> real board move."""
        if self.to_move == BLACK or move == self.ncells:
            return move
        r, c = divmod(move, self.n)
        return c * self.n + r

    def __str__(self) -> str:
        n = self.n
        symbols = {EMPTY: ".", BLACK: "B", WHITE: "W"}
        header = "    " + " ".join(chr(ord("a") + c) for c in range(n))
        lines = [header]
        for r in range(n):
            row = " ".join(symbols[self.board[r * n + c]] for c in range(n))
            lines.append(f"{r + 1:>2}  " + " " * r + row)
        lines.append(f"    to move: {'BLACK' if self.to_move == BLACK else 'WHITE'}")
        return "\n".join(lines)


def canonicalise(board: np.ndarray, to_move: int) -> np.ndarray:
    """Board array -> ``to_move``'s point of view (0 empty, 1 own, 2 opp).

    Free function so a *finished* game can be re-rendered from the point of
    view of whoever was to move at some earlier position -- which is what the
    auxiliary final-occupancy target needs.
    """
    if to_move == BLACK:
        out = np.where(board == BLACK, 1, np.where(board == WHITE, 2, 0))
    else:
        bt = board.T
        out = np.where(bt == WHITE, 1, np.where(bt == BLACK, 2, 0))
    return out.astype(np.uint8)


def move_to_str(move: int, n: int) -> str:
    if move == n * n:
        return "swap"
    r, c = divmod(move, n)
    return f"{chr(ord('a') + c)}{r + 1}"


def str_to_move(text: str, n: int) -> int:
    text = text.strip().lower()
    if text in ("swap", "swap-pieces"):
        return n * n
    c = ord(text[0]) - ord("a")
    r = int(text[1:]) - 1
    return r * n + c


def rotate180_move(move: int, n: int) -> int:
    if move == n * n:
        return move
    r, c = divmod(move, n)
    return (n - 1 - r) * n + (n - 1 - c)


def winner_of_full_board(board: np.ndarray) -> int:
    """Winner of a *completely filled* board, via a flood fill for BLACK.

    Used by rollout policies, where the cheapest correct check is to fill the
    board and ask who owns a connection (in Hex a full board always has exactly
    one winner, so 'not BLACK' implies WHITE).
    """
    n = board.shape[0]
    nei = neighbour_table(n)
    flat = board.reshape(-1)
    stack = [c for c in range(n) if flat[c] == BLACK]
    seen = bytearray(n * n)
    for s in stack:
        seen[s] = 1
    while stack:
        cur = stack.pop()
        if cur >= (n - 1) * n:
            return BLACK
        for nb in nei[cur]:
            if not seen[nb] and flat[nb] == BLACK:
                seen[nb] = 1
                stack.append(nb)
    return WHITE
