"""Hex's rating table and player profiles.

The logic lives in :mod:`alphazero_core.ratings` (Connect Four keeps its own
files with the same machinery).  What stays here is Hex's pair of paths and the
module-level function names the GUI, the ladder and the tests already use.
"""

from __future__ import annotations

from pathlib import Path

from alphazero_core.ratings import (ANCHOR, ANCHOR_RATING, K_PROVISIONAL,  # noqa: F401
                                    K_SETTLED, PROVISIONAL_GAMES, START_RATING,
                                    RatingStore, bot_key, bot_label,
                                    expected_score, first_player_advantage,
                                    k_factor)

ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT / "runs" / "ratings.json"
USERS_PATH = ROOT / "runs" / "users.json"

STORE = RatingStore(RATINGS_PATH, USERS_PATH, game="hex")


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


def save_ratings(bots: dict, first_player_advantage: float, board_size: int,
                 games_per_pair: int, n_games: int) -> None:
    _store().save_ratings(bots, first_player_advantage, board_size,
                          games_per_pair, n_games)


def load_users() -> dict:
    return _store().load_users()


def create_user(name: str) -> tuple[dict | None, str | None]:
    return _store().create_user(name)


def record_result(name: str, opponent_key: str, opponent_rating: float,
                  won: bool, played_first: bool, advantage: float,
                  plies: int, board_size: int) -> dict | None:
    """Apply one rated Hex game to a player's Elo.

    Hex cannot be drawn, so the result is a bool here rather than a score.
    """
    return _store().record_result(name, opponent_key, opponent_rating,
                                  1.0 if won else 0.0, played_first, advantage,
                                  plies, board_size)
