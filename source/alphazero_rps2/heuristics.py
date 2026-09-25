"""What an Intransitive position is worth to somebody who is not searching.

Shared by the rule-based agent, the alpha-beta agent's leaf evaluation and its
move ordering, and the rollout player's playout policy.  Nothing here looks
more than one ply ahead; everything that does is in ``agents/``.

The three things worth knowing about this game, and where each one lives below
-------------------------------------------------------------------------------

* **Material is not a scalar.**  A rock is worth a great deal against an army of
  scissors and nothing at all against an army of paper, so counting pieces --
  the first thing every other evaluation here does -- is close to meaningless.
  :func:`material` prices each piece by the *matchup*: what it can eat, minus
  what can eat it, over what the opponent actually has on the board.  Ten
  pieces against ten is dead level only when the two armies' type mixtures are.

* **The game is a race to a corner, and races are asymmetric.**  Both sides run
  at once, so what matters is not how far your fastest piece is from their base
  but how far it is *compared to* theirs, and whether anything of theirs can
  stand in the way.  :func:`race` scores the nearest few runners on each side,
  and charges heavily for an enemy already standing next to the base.

* **A square is only as safe as the piece standing on it.**  Every piece moves
  like a king, so choosing a move is choosing a square, and the same square is
  suicide for one type and free for another.  :func:`hanging` counts what is
  currently attacked-and-undefended, and :func:`gives_opponent_win` refuses the
  moves that hand over the game outright -- the one blunder that ends a game on
  the spot rather than costing a piece.

Every function takes a plain board and a player, so the same code answers about
either side and about a position it is not that side's turn in -- which is what
"can they do this to me next move" needs.
"""

from __future__ import annotations

import numpy as np

from .rps2_game import (BASE_OF, NACTIONS, NCELLS, NDIRS, STEP_LIST,
                        TARGET_BASE, DIST_TO, RPS2Board, beats, kind_of,
                        other, owner_of, split_action)

# Type -> the type it eats, and the type that eats it.
PREY = (2, 0, 1)
PREDATOR = (1, 2, 0)


# ------------------------------------------------------------------- moves
def moves_for(board: RPS2Board, player: int) -> list[int]:
    """Every move ``player`` could make, whether or not it is their turn.

    "Whether or not" is the point: half the questions a static evaluation asks
    are about what the *opponent* would do next, and the opponent is by
    definition not on turn.  The rules of movement do not mention the clock, so
    the same generator answers both.
    """
    cells = board.cells
    out: list[int] = []
    for src in board.pieces[player]:
        mine = kind_of(cells[src])
        steps = STEP_LIST[src]
        for d in range(NDIRS):
            dst = steps[d]
            if dst < 0:
                continue
            v = cells[dst]
            if v and (owner_of(v) == player or not beats(mine, kind_of(v))):
                continue
            out.append(d * NCELLS + src)
    return out


def immediate_wins(board: RPS2Board, player: int) -> list[int]:
    """``player``'s moves that reach the enemy base, i.e. win on the spot.

    When ``player`` is the side to move these are legal moves and playing one
    ends the game; when it is not, they are the threats that have to be dealt
    with this turn.
    """
    goal = TARGET_BASE[player]
    return [m for m in moves_for(board, player)
            if STEP_LIST[m % NCELLS][m // NCELLS] == goal]


def captures(board: RPS2Board, player: int) -> list[int]:
    """``player``'s capturing moves."""
    cells = board.cells
    return [m for m in moves_for(board, player)
            if cells[STEP_LIST[m % NCELLS][m // NCELLS]]]


def gives_opponent_win(board: RPS2Board, move: int, player: int) -> bool:
    """Would playing ``move`` let the opponent walk into our base next turn?

    The defining blunder: unlike losing a piece, it is not a matter of degree.
    A board copy per call, which is why the alpha-beta agent only asks it near
    the root where the ordering is worth paying for.
    """
    child = board.copy()
    child.play(move)
    if child.is_terminal():
        return child.winner == other(player)
    return bool(immediate_wins(child, other(player)))


# ------------------------------------------------------------- the pieces
def _counts(board: RPS2Board, player: int) -> tuple[int, int, int]:
    out = [0, 0, 0]
    for s in board.pieces[player]:
        out[kind_of(board.cells[s])] += 1
    return out[0], out[1], out[2]


def piece_value(kind: int, enemy_counts: tuple[int, int, int]) -> float:
    """What one piece is worth against a particular enemy army.

    A base of 1, plus what fraction of the enemy it eats, minus what fraction
    eats it.  The two extremes are a piece that dominates every enemy on the
    board (1.5) and one that every enemy dominates (0.5), which is about the
    right spread: a losing matchup is still a body in the way, and blocking is
    a real resource in a game where equal types cannot pass each other.
    """
    total = sum(enemy_counts)
    if total == 0:
        return 1.0
    return 1.0 + 0.5 * (enemy_counts[PREY[kind]] - enemy_counts[PREDATOR[kind]]) / total


def material(board: RPS2Board, player: int) -> float:
    """Matchup-weighted material, from ``player``'s point of view."""
    opp = other(player)
    mine = _counts(board, player)
    theirs = _counts(board, opp)
    ours = sum(mine[t] * piece_value(t, theirs) for t in range(3))
    hers = sum(theirs[t] * piece_value(t, mine) for t in range(3))
    return ours - hers


# --------------------------------------------------------------- the race
def _nearest(board: RPS2Board, player: int, take: int = 3) -> list[int]:
    """Distances from ``player``'s nearest pieces to the enemy base."""
    goal = other(player)
    d = sorted(DIST_TO[goal][s] for s in board.pieces[player])
    return d[:take]


def base_attackers(board: RPS2Board, attacker: int) -> list[int]:
    """``attacker``'s pieces that could step into the defender's base *now*.

    A base is defended only by standing on it with something the intruder
    cannot take -- there is no blocking a king move otherwise -- so this is
    exactly "adjacent, and the square is enterable".
    """
    base = BASE_OF[other(attacker)]
    guard = board.cells[base]
    # Walk the eight neighbours of the base rather than the board: a king move
    # is the only way in, so nothing further away can be a threat this turn.
    out = []
    for d in range(NDIRS):
        src = STEP_LIST[base][d]
        if src < 0:
            continue
        v = board.cells[src]
        if not v or owner_of(v) != attacker:
            continue
        if guard and not beats(kind_of(v), kind_of(guard)):
            continue
        out.append(src)
    return out


def race(board: RPS2Board, player: int) -> float:
    """How the two runs to the far corner compare, from ``player``'s side.

    Only the nearest few pieces count.  A piece eight squares from the enemy
    base is not in the race at all, and summing over every piece would let a
    solid but static army outscore one runner who is two moves from winning --
    which is the wrong way round in a game decided by arriving first.
    """
    opp = other(player)
    mine = _nearest(board, player)
    theirs = _nearest(board, opp)
    if not mine:
        return -8.0
    if not theirs:
        return 8.0
    score = 0.0
    for i in range(max(len(mine), len(theirs))):
        weight = 1.0 / (i + 1)
        a = mine[i] if i < len(mine) else 8
        b = theirs[i] if i < len(theirs) else 8
        score += weight * (b - a)
    # An enemy standing next to our base is a move from winning and has to be
    # answered this turn; the reverse is just as urgent in our favour.
    score += 3.0 * len(base_attackers(board, player))
    score -= 3.0 * len(base_attackers(board, opp))
    return score


# ------------------------------------------------------------- what hangs
def hanging(board: RPS2Board, player: int) -> float:
    """``player``'s attacked pieces, discounted by whether they can be avenged.

    "Defended" here means the square can be retaken: some piece of ours could
    move onto it and beat whatever would be standing there.  In a game where
    capture is decided by type rather than by value, an exchange is never a
    trade -- the piece that arrives is simply better than the one it took -- so
    a defended piece is not safe, only paid for.
    """
    opp = other(player)
    cells = board.cells
    loss = 0.0
    for square in board.pieces[player]:
        mine = kind_of(cells[square])
        attackers = [s for s in _neighbours(square)
                     if cells[s] and owner_of(cells[s]) == opp
                     and beats(kind_of(cells[s]), mine)]
        if not attackers:
            continue
        # Whoever takes us ends up standing there; can we take *them*?
        avenged = any(
            any(cells[s] and owner_of(cells[s]) == player
                and beats(kind_of(cells[s]), kind_of(cells[a]))
                for s in _neighbours(square) if s != a)
            for a in attackers
        )
        loss += 0.35 if avenged else 1.0
    return loss


def _neighbours(square: int) -> list[int]:
    return [s for s in STEP_LIST[square] if s >= 0]


def mobility(board: RPS2Board, player: int) -> int:
    return len(moves_for(board, player))


# ------------------------------------------------------------- evaluation
# The weights.  Material and the race are the two things that decide games and
# they are deliberately close in size: an army that is winning on matchups but
# a move behind in the race is losing, and vice versa.
W_MATERIAL = 6.0
W_RACE = 4.0
W_HANGING = 3.0
W_MOBILITY = 0.06


def evaluate(board: RPS2Board, player: int) -> float:
    """Static value of a position for ``player``, on roughly a -1..+1 scale.

    Terminal positions answer exactly; everything else is the weighted sum, put
    through a ``tanh`` so that a search comparing it with a proven win never
    prefers a very good position to a won one.
    """
    if board.winner:
        return 1.0 if board.winner == player else -1.0
    if board.drawn:
        return 0.0
    opp = other(player)
    raw = (W_MATERIAL * material(board, player)
           + W_RACE * race(board, player)
           - W_HANGING * (hanging(board, player) - hanging(board, opp))
           + W_MOBILITY * (mobility(board, player) - mobility(board, opp)))
    return float(np.tanh(raw / 30.0))


# ------------------------------------------------------------ move scoring
def move_scores(board: RPS2Board, player: int) -> np.ndarray:
    """A score for every legal move, ``-inf`` elsewhere.

    One ply of lookahead, done cheaply and without copying the board: what the
    move takes, where it lands relative to the two bases, whether the square it
    lands on is attacked, and whether leaving the square it came from exposes
    anything.  The one thing it does copy the board for is the outright
    blunder check, and only on moves that survive everything else.
    """
    scores = np.full(NACTIONS, -np.inf, dtype=np.float64)
    legal = board.legal_moves()
    if not len(legal):
        return scores
    opp = other(player)
    cells = board.cells
    goal = TARGET_BASE[player]
    home = BASE_OF[player]
    theirs = _counts(board, opp)
    threatened = bool(base_attackers(board, opp))

    for m in legal:
        m = int(m)
        src, d = split_action(m)
        dst = STEP_LIST[src][d]
        mine = kind_of(cells[src])
        s = 0.0
        if dst == goal:
            scores[m] = 1e6
            continue
        victim = cells[dst]
        if victim:
            s += 12.0 * piece_value(kind_of(victim), _counts(board, player))
            # Taking the piece that is about to walk into our base is worth
            # more than the piece is.
            if dst in _neighbours(home):
                s += 18.0
        # Progress towards their corner, and away from ours.
        s += 2.0 * (DIST_TO[opp][src] - DIST_TO[opp][dst])
        # Sitting on our own base with something the intruder cannot take is a
        # real defence, and the only one this game has.
        if threatened and dst == home:
            s += 10.0
        # Is the square we are moving to attacked, and could we retake?
        attackers = [s2 for s2 in _neighbours(dst)
                     if s2 != src and cells[s2] and owner_of(cells[s2]) == opp
                     and beats(kind_of(cells[s2]), mine)]
        if attackers:
            defended = any(
                cells[s2] and owner_of(cells[s2]) == player and s2 != src
                and beats(kind_of(cells[s2]), kind_of(cells[a]))
                for a in attackers for s2 in _neighbours(dst)
            )
            s -= 6.0 * piece_value(mine, theirs) * (0.4 if defended else 1.0)
        # ...and does the square we are leaving stop protecting something?
        for n in _neighbours(src):
            v = cells[n]
            if v and owner_of(v) == opp and beats(mine, kind_of(v)):
                s -= 1.0
        scores[m] = s

    # Only moves that are otherwise attractive are worth a board copy.
    order = np.argsort(scores)[::-1]
    for m in order[:8]:
        m = int(m)
        if not np.isfinite(scores[m]) or scores[m] >= 1e5:
            continue
        if gives_opponent_win(board, m, player):
            scores[m] -= 500.0
    return scores


# --------------------------------------------------------- opening variety
def static_order() -> list[int]:
    """Opening moves for the arena, best-looking first.

    Deterministic agents would otherwise replay one game every time.  Ranked by
    :func:`move_scores` on the starting position, so a short match is played
    from openings a real player might choose and a long one eventually reaches
    the odd ones.
    """
    board = RPS2Board()
    scores = move_scores(board, board.to_move)
    order = [int(m) for m in np.argsort(scores)[::-1] if np.isfinite(scores[int(m)])]
    return order
