"""Classical (non-neural) Hex knowledge: two-distance and bridges.

These power the hand-written opponents that the trained agent is measured
against.  Nothing here touches the network.

Two-distance (Anshelevich) is the standard connection-strength measure for
Hex.  Ordinary shortest path badly overrates a connection that hangs on a
single cell; two-distance charges an empty cell the *second* smallest
neighbour distance plus one, so a cell only becomes cheap when it has two
independent ways through -- exactly the redundancy that makes a Hex
connection hard to cut.
"""

from __future__ import annotations

import numpy as np

from .hex_game import BLACK, EMPTY, WHITE, HexBoard, neighbour_table

INF = 1e6

_PAD_CACHE: dict[int, np.ndarray] = {}
_BRIDGE_CACHE: dict[int, tuple] = {}

# (dr, dc) of the six bridge partners, each with the two carrier offsets that
# must stay empty for the bridge to be a real (second-order) connection.
BRIDGE_OFFSETS = (
    ((-1, -1), ((-1, 0), (0, -1))),
    ((1, 1), ((1, 0), (0, 1))),
    ((-2, 1), ((-1, 0), (-1, 1))),
    ((2, -1), ((1, 0), (1, -1))),
    ((-1, 2), ((-1, 1), (0, 1))),
    ((1, -2), ((1, -1), (0, -1))),
)


def padded_neighbours(n: int) -> np.ndarray:
    """(n*n, 6) int array; off-board entries point at a sentinel slot n*n."""
    cached = _PAD_CACHE.get(n)
    if cached is not None:
        return cached
    nei = neighbour_table(n)
    out = np.full((n * n, 6), n * n, dtype=np.int32)
    for i, nbs in enumerate(nei):
        out[i, : len(nbs)] = nbs
    _PAD_CACHE[n] = out
    return out


def bridge_table(n: int) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """For each cell, tuples of (partner, carrier_a, carrier_b) flat indices."""
    cached = _BRIDGE_CACHE.get(n)
    if cached is not None:
        return cached
    table = []
    for r in range(n):
        for c in range(n):
            entries = []
            for (dr, dc), carriers in BRIDGE_OFFSETS:
                pr, pc = r + dr, c + dc
                if not (0 <= pr < n and 0 <= pc < n):
                    continue
                ok = True
                cells = []
                for cr, cc in carriers:
                    ar, ac = r + cr, c + cc
                    if not (0 <= ar < n and 0 <= ac < n):
                        ok = False
                        break
                    cells.append(ar * n + ac)
                if ok:
                    entries.append((pr * n + pc, cells[0], cells[1]))
            table.append(tuple(entries))
    out = tuple(table)
    _BRIDGE_CACHE[n] = out
    return out


def two_distance(board_flat: np.ndarray, n: int, player: int, source: str) -> np.ndarray:
    """Two-distance from one of ``player``'s edges to every cell.

    ``source`` is 'top'/'bottom' for BLACK and 'left'/'right' for WHITE.
    """
    ncells = n * n
    nei = padded_neighbours(n)
    opp = WHITE if player == BLACK else BLACK

    is_own = board_flat == player
    is_opp = board_flat == opp

    rows = np.arange(ncells) // n
    cols = np.arange(ncells) % n
    src = np.full(ncells, INF, dtype=np.float64)
    if source == "top":
        src[rows == 0] = 0.0
    elif source == "bottom":
        src[rows == n - 1] = 0.0
    elif source == "left":
        src[cols == 0] = 0.0
    else:
        src[cols == n - 1] = 0.0
    # An edge cell already owned by `player` sits *on* the source group.
    src[is_opp] = INF

    d = np.full(ncells + 1, INF, dtype=np.float64)
    d[ncells] = INF
    # The source edge occupies *two* of the candidate slots: a board edge
    # cannot be cut, so touching it already counts as a redundant connection.
    # With only one slot the relaxation deadlocks at infinity, because no empty
    # cell can ever find a second finite neighbour to get started from.
    cand = np.empty((ncells, 8), dtype=np.float64)
    for _ in range(2 * n + 4):
        cand[:, :6] = d[nei]
        cand[:, 6] = src
        cand[:, 7] = src
        cand.sort(axis=1)
        new = np.where(is_own, cand[:, 0], cand[:, 1] + 1.0)
        new[is_opp] = INF
        new = np.minimum(new, d[:ncells])
        if np.array_equal(new, d[:ncells]):
            break
        d[:ncells] = new
    return np.minimum(d[:ncells], INF)


def _edges(player: int) -> tuple[str, str]:
    return ("top", "bottom") if player == BLACK else ("left", "right")


def connection_cost(board_flat: np.ndarray, n: int, player: int) -> tuple[float, np.ndarray]:
    """(potential, per-cell cost) for ``player``.

    ``potential`` is the cheapest total two-distance through any single cell --
    a proxy for "how many more stones do I need to connect my two edges".
    """
    e1, e2 = _edges(player)
    d1 = two_distance(board_flat, n, player, e1)
    d2 = two_distance(board_flat, n, player, e2)
    total = d1 + d2
    # A cell is counted by both halves, so an empty one is charged twice.
    total = np.where(board_flat == EMPTY, total - 1.0, total)
    total = np.minimum(total, INF)
    return float(total.min()), total


def evaluate(board: HexBoard, player: int) -> float:
    """Static evaluation in [-1, 1] from ``player``'s point of view."""
    if board.winner:
        return 1.0 if board.winner == player else -1.0
    flat = board.array().reshape(-1)
    opp = WHITE if player == BLACK else BLACK
    my_pot, _ = connection_cost(flat, board.n, player)
    op_pot, _ = connection_cost(flat, board.n, opp)
    if my_pot >= INF:
        return -1.0
    if op_pot >= INF:
        return 1.0
    diff = op_pot - my_pot
    return float(np.tanh(diff / 3.0)) * 0.99  # keep clear of true terminal scores


def move_scores(board: HexBoard, player: int, defence_weight: float = 1.0) -> np.ndarray:
    """Per-cell desirability for ``player`` (higher is better, -inf if illegal).

    A good Hex move usually does two jobs at once: it shortens my own
    connection and lengthens my opponent's.  Scoring both and adding them is
    the classic "attack + defence" move ordering.
    """
    flat = board.array().reshape(-1)
    n = board.n
    opp = WHITE if player == BLACK else BLACK
    _, my_cost = connection_cost(flat, n, player)
    _, op_cost = connection_cost(flat, n, opp)
    score = -np.minimum(my_cost, 1e5) - defence_weight * np.minimum(op_cost, 1e5)
    # Two-distance is flat on an empty board (every cell costs the same n
    # stones to connect through), so break ties towards the centre -- the
    # standard Hex convention, and correct: central cells keep more options.
    score = score + 0.15 * _centrality(n)
    score[flat != EMPTY] = -np.inf
    return score


_CENTRALITY_CACHE: dict[int, np.ndarray] = {}


def _centrality(n: int) -> np.ndarray:
    cached = _CENTRALITY_CACHE.get(n)
    if cached is not None:
        return cached
    idx = np.arange(n * n)
    r, c = idx // n, idx % n
    mid = (n - 1) / 2.0
    # Chebyshev-ish distance from the centre, normalised to [0, 1] then negated.
    dist = np.maximum(np.abs(r - mid), np.abs(c - mid)) / max(mid, 1e-9)
    out = (1.0 - dist).astype(np.float64)
    _CENTRALITY_CACHE[n] = out
    return out


def immediate_wins(board: HexBoard, player: int) -> list[int]:
    """Cells where ``player`` moving now would complete a connection.

    Read straight off the union-find: a cell wins iff it simultaneously touches
    a group joined to one of ``player``'s edges and a group joined to the
    other.  O(6) per empty cell instead of copying the board 121 times.
    """
    if board.winner:
        return []
    n = board.n
    flat = board.board
    if player == BLACK:
        a_root, b_root = board._find(board.TOP), board._find(board.BOTTOM)
    else:
        a_root, b_root = board._find(board.LEFT), board._find(board.RIGHT)
    if a_root == b_root:
        return []
    out = []
    nei = board._nei
    for cell in range(n * n):
        if flat[cell]:
            continue
        r, c = divmod(cell, n)
        if player == BLACK:
            hits_a, hits_b = r == 0, r == n - 1
        else:
            hits_a, hits_b = c == 0, c == n - 1
        for nb in nei[cell]:
            if flat[nb] != player:
                continue
            root = board._find(nb)
            if root == a_root:
                hits_a = True
            elif root == b_root:
                hits_b = True
        if hits_a and hits_b:
            out.append(cell)
    return out


def broken_bridge_response(board: HexBoard, player: int, last_move: int | None) -> int | None:
    """If the opponent just intruded into one of my bridges, plug the other cell.

    This single pattern is most of what separates a naive Hex bot from a
    respectable one: it makes second-order connections effectively unbreakable.
    """
    if last_move is None:
        return None
    n = board.n
    flat = board.array().reshape(-1)
    opp = WHITE if player == BLACK else BLACK
    if flat[last_move] != opp:
        return None
    bridges = bridge_table(n)
    for cell, entries in enumerate(bridges):
        if flat[cell] != player:
            continue
        for partner, ca, cb in entries:
            if flat[partner] != player:
                continue
            if last_move == ca and flat[cb] == EMPTY:
                return cb
            if last_move == cb and flat[ca] == EMPTY:
                return ca
    return None
