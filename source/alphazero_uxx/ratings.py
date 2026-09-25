"""Ultimate XX's rating table and player profiles.

Same machinery as the other games (:mod:`alphazero_core.ratings`), separate
files: a player is a different strength at each game and the bot fields have
nothing in common beyond both being anchored at ``random`` = 1000.

Draws count, as they do in Connect Four and Ultimate Tic-Tac-Toe: a drawn rated
game scores half a point, which is what :func:`record_result` takes here.  They
are rarer in this game -- roughly one random game in fourteen ends with all nine
boards claimed and no line among them -- but a strong pair finds them far more
often than a random one does, so the half point still has to be there.
"""

from __future__ import annotations

from pathlib import Path

from alphazero_core.ratings import (ANCHOR, ANCHOR_RATING, K_PROVISIONAL,  # noqa: F401
                                    K_SETTLED, PROVISIONAL_GAMES, START_RATING,
                                    RatingStore, bot_key, bot_label,
                                    expected_score, first_player_advantage,
                                    k_factor)

ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT / "runs" / "ratings_uxx.json"
USERS_PATH = ROOT / "runs" / "users_uxx.json"

STORE = RatingStore(RATINGS_PATH, USERS_PATH, game="uxx")


def _store() -> RatingStore:
    """The store, re-pointed at the module-level paths (the tests rebind them)."""
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
    """Apply one rated Ultimate XX game to a player's Elo.

    ``score`` is 1 for a win, 0.5 for a draw, 0 for a loss.
    """
    return _store().record_result(name, opponent_key, opponent_rating, score,
                                  played_first, advantage, plies, board_size)
