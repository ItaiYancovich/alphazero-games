"""Splendor's rating table and player profiles.

The machinery is :mod:`alphazero_core.ratings`, as for the other three games.
What is here is this game's pair of file paths, and one label the shared table
does not know about (``heuristic``, which Hex and Connect Four call ``rule``).

One thing genuinely differs, and it is worth saying plainly: **rated games are
two-player only.**  Incremental Elo is a statement about one player against one
opponent of known strength, and finishing third of four is not that -- it is a
joint result against a field.  The bots are rated at every table size in the
tournament (which decomposes each table into its pairwise finishes), but a human
rating is only updated from a two-player game.  The GUI says so rather than
quietly scoring a four-player game as if it were a duel.
"""

from __future__ import annotations

from pathlib import Path

from alphazero_core.ratings import (ANCHOR, ANCHOR_RATING, K_PROVISIONAL,  # noqa: F401
                                    K_SETTLED, PROVISIONAL_GAMES, START_RATING,
                                    RatingStore, bot_key, expected_score,
                                    first_player_advantage, k_factor)
from alphazero_core.ratings import bot_label as _core_bot_label

ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT / "runs" / "ratings_splendor.json"
USERS_PATH = ROOT / "runs" / "users_splendor.json"

STORE = RatingStore(RATINGS_PATH, USERS_PATH, game="splendor")


def bot_label(key: str) -> str:
    """Human-readable name for a rating key, including this game's own."""
    if key.startswith("az2@"):
        sims = key.split("@")[1]
        return ("AlphaZero v2 policy (no search)" if sims == "0"
                else f"AlphaZero v2 ({sims} sims)")
    if key == "heuristic":
        return "Heuristic (greedy expert)"
    if key == "planner":
        return "Planner (plans its purchases)"
    if key.startswith("planner:"):
        return f"Planner (depth {key.split(':')[1]})"
    return _core_bot_label(key)


def _store() -> RatingStore:
    """The store, re-pointed at the module-level paths.

    ``RATINGS_PATH`` and ``USERS_PATH`` are module attributes and the tests
    rebind them to a temporary directory, so the paths are read here on every
    call rather than captured once when the store was built.
    """
    STORE.ratings_path = Path(RATINGS_PATH)
    STORE.users_path = Path(USERS_PATH)
    return STORE


def load_ratings() -> dict:
    return _store().load_ratings()


def save_ratings(bots: dict, first_player_advantage: float, board_size,
                 games_per_pair: int, n_games: int) -> None:
    _store().save_ratings(bots, first_player_advantage, board_size,
                          games_per_pair, n_games)


def load_users() -> dict:
    return _store().load_users()


def create_user(name: str) -> tuple[dict | None, str | None]:
    return _store().create_user(name)


def record_result(name: str, opponent_key: str, opponent_rating: float,
                  score: float, played_first: bool, advantage: float,
                  plies: int, board_size) -> dict | None:
    """Apply one rated two-player Splendor game to a player's Elo.

    ``score`` rather than ``won``: a Splendor game can be tied on both prestige
    and card count, which is half a point, so this takes the same signature as
    Connect Four's.
    """
    return _store().record_result(name, opponent_key, opponent_rating, score,
                                  played_first, advantage, plies, board_size)
