"""Backgammon's rating table and player profiles.

Same machinery as the other two games (:mod:`alphazero_core.ratings`), its own
pair of files.  Two things are named differently because the game is:

* the strength dial in a bot key is *plies*, not simulations, so the keys read
  ``net@1`` and ``net@2``;
* backgammon has no draws, so a result is a win or a loss like Hex -- but a
  win is worth 1, 2 or 3 points, and the ladder reports equity alongside the
  rating because Elo can only model who beat whom.
"""

from __future__ import annotations

from pathlib import Path

from alphazero_core.ratings import (ANCHOR, ANCHOR_RATING, K_PROVISIONAL,  # noqa: F401
                                    K_SETTLED, PROVISIONAL_GAMES, START_RATING,
                                    RatingStore, expected_score,
                                    first_player_advantage, k_factor)

ROOT = Path(__file__).resolve().parents[1]
RATINGS_PATH = ROOT / "runs" / "ratings_bg.json"
USERS_PATH = ROOT / "runs" / "users_bg.json"

STORE = RatingStore(RATINGS_PATH, USERS_PATH, game="bg")


def bot_key(choice: str, plies: int = 0, net: str | None = None) -> str:
    """Stable identity for one opponent configuration.

    ``plies`` stands in for the simulation count the other games use, so the
    GUI, the ladder and the rating file agree on one name per opponent.
    """
    if choice != "net":
        return choice
    if net:
        stem = net[:-3] if net.endswith(".pt") else net
        return f"net:{stem}@{int(plies)}"
    return f"net@{int(plies)}"


def bot_label(key: str) -> str:
    if key.startswith("net:"):
        net, _, plies = key[4:].partition("@")
        return f"Network[{net}] {plies}-ply"
    if key.startswith("net@"):
        return f"Network {key[4:]}-ply"
    return {"random": "Random", "heuristic": "Heuristic"}.get(key, key)


def _store() -> RatingStore:
    """The store, re-pointed at the module-level paths (tests rebind them)."""
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
                  won: bool, played_first: bool, advantage: float,
                  plies: int, board_size) -> dict | None:
    """Apply one rated backgammon game to a player's Elo.

    A win is a win for rating purposes, whether it was a gammon or not; the
    points are reported separately.  ``plies`` here is the game length in turns,
    matching the other two games' signature.
    """
    return _store().record_result(name, opponent_key, opponent_rating,
                                  1.0 if won else 0.0, played_first, advantage,
                                  plies, board_size)
