"""Bot rating table and human player profiles, for one game.

The tournament rates the bots against each other with a Bradley-Terry fit
(:mod:`alphazero_core.elo`); that fit is a *joint* estimate over a fixed field
and is re-run offline.  A human cannot be folded into it -- they arrive with no
games and play a handful at a time -- so human ratings use ordinary incremental
Elo against bots whose ratings are held fixed.  Both live on the same scale,
anchored so ``random`` sits at :data:`ANCHOR_RATING`.

Both games have a first-player advantage large enough to distort a human rating
(always taking the first seat would inflate it), so the expected score is
corrected by the advantage measured in the very tournament that produced the bot
ratings.

A :class:`RatingStore` is one game's pair of files.  Hex and Connect Four keep
separate tables and separate player profiles: the same person is a different
strength at the two games, and the bot fields have nothing to do with each
other beyond sharing an anchor.
"""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path

ANCHOR = "random"
ANCHOR_RATING = 1000.0
START_RATING = 1000.0
PROVISIONAL_GAMES = 15      # while a rating is still finding its level
K_PROVISIONAL = 40.0
K_SETTLED = 20.0


def bot_key(choice: str, sims: int = 0, net: str | None = None) -> str:
    """Stable identity for one opponent configuration.

    The GUI and the tournament must agree on this exactly, or a rated game
    cannot find its opponent's rating.

    ``net`` names a checkpoint when the entry is a *comparison* network rather
    than the rated one.  The rated network deliberately keeps the bare
    ``az@N`` key: rated games, stored player histories and every previously
    written ratings file all use it, and renaming it would orphan them.
    """
    if choice != "az":
        return choice
    if net:
        stem = net[:-3] if net.endswith(".pt") else net
        return f"az:{stem}@{int(sims)}"
    return f"az@{int(sims)}"


def bot_label(key: str) -> str:
    """Human-readable name for a rating key."""
    if key.startswith("az:"):
        net, _, sims_text = key[3:].partition("@")
        sims = int(sims_text)
        what = "policy (no search)" if sims == 0 else f"{sims} sims"
        return f"AlphaZero[{net}] {what}"
    if key.startswith("az@"):
        sims = int(key[3:])
        return "AlphaZero policy (no search)" if sims == 0 else f"AlphaZero {sims} sims"
    if key.startswith("minimax:"):
        return f"Alpha-beta {float(key.split(':')[1]):g}s"
    if key.startswith("rollout:"):
        return f"Rollout MCTS {float(key.split(':')[1]):g}s"
    return {"random": "Random", "rule": "Rule-based"}.get(key, key)


def first_player_advantage(first_score_rate: float) -> float:
    """The first move's worth in Elo, from the score the first player took.

    ``first_score_rate`` counts a draw as half a game, so this is meaningful in
    Connect Four as well as in Hex, where draws do not exist.
    """
    p = min(max(first_score_rate, 1e-3), 1 - 1e-3)
    return 400.0 * math.log10(p / (1 - p))


def expected_score(rating: float, opponent: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def k_factor(games: int) -> float:
    return K_PROVISIONAL if games < PROVISIONAL_GAMES else K_SETTLED


class RatingStore:
    """One game's ratings file plus its player profiles."""

    def __init__(self, ratings_path: Path, users_path: Path, game: str):
        self.ratings_path = Path(ratings_path)
        self.users_path = Path(users_path)
        self.game = game
        self._lock = threading.Lock()

    # ----------------------------------------------------------- bot ratings
    def load_ratings(self) -> dict:
        """The rating table, or an empty one if the tournament has not run."""
        try:
            with open(self.ratings_path) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {"bots": {}, "first_player_advantage": 0.0, "anchor": ANCHOR,
                    "anchor_rating": ANCHOR_RATING, "board_size": None,
                    "game": self.game, "generated": None}
        data.setdefault("bots", {})
        data.setdefault("first_player_advantage", 0.0)
        data.setdefault("game", self.game)
        return data

    def save_ratings(self, bots: dict, first_player_advantage: float, board_size,
                     games_per_pair: int, n_games: int) -> None:
        self.ratings_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "game": self.game,
            "anchor": ANCHOR,
            "anchor_rating": ANCHOR_RATING,
            "board_size": board_size,
            "games_per_pair": games_per_pair,
            "n_games": n_games,
            "first_player_advantage": first_player_advantage,
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bots": bots,
        }
        with open(self.ratings_path, "w") as fh:
            json.dump(payload, fh, indent=2)

    # --------------------------------------------------------- human players
    def load_users(self) -> dict:
        try:
            with open(self.users_path) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_users(self, users: dict) -> None:
        self.users_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.users_path, "w") as fh:
            json.dump(users, fh, indent=2)

    def create_user(self, name: str) -> tuple[dict | None, str | None]:
        """Add a player.  Returns ``(profile, problem)``."""
        name = " ".join(name.split())[:32]
        if not name:
            return None, "a name is required"
        with self._lock:
            users = self.load_users()
            if any(u.lower() == name.lower() for u in users):
                return None, f"{name} already exists"
            users[name] = {"name": name, "rating": START_RATING, "games": 0,
                           "wins": 0, "losses": 0, "draws": 0,
                           "peak": START_RATING, "history": []}
            self._save_users(users)
            return users[name], None

    def record_result(self, name: str, opponent_key: str, opponent_rating: float,
                      score: float, played_first: bool, advantage: float,
                      plies: int, board_size) -> dict | None:
        """Apply one rated game to a player's Elo. Returns the updated profile.

        ``score`` is 1 for a win, 0.5 for a draw and 0 for a loss.
        ``played_first`` corrects for the first-player edge: the mover holds
        ``advantage`` Elo of it, so beating a bot from the second seat is worth
        more than beating it from the first.
        """
        with self._lock:
            users = self.load_users()
            user = users.get(name)
            if user is None:
                return None
            rating = float(user["rating"])
            # Put the advantage on whichever side actually moved first.
            mine = rating + (advantage if played_first else 0.0)
            theirs = opponent_rating + (0.0 if played_first else advantage)
            expected = expected_score(mine, theirs)
            k = k_factor(int(user["games"]))
            new_rating = rating + k * (score - expected)

            user["rating"] = round(new_rating, 1)
            user["games"] = int(user["games"]) + 1
            user["wins"] = int(user["wins"]) + (1 if score > 0.75 else 0)
            user["losses"] = int(user["losses"]) + (1 if score < 0.25 else 0)
            if 0.25 <= score <= 0.75:
                user["draws"] = int(user.get("draws", 0)) + 1
            user["peak"] = round(max(float(user.get("peak", START_RATING)), new_rating), 1)
            user["history"].append({
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "opponent": opponent_key,
                "opponent_rating": round(opponent_rating, 1),
                "colour": "red" if played_first else "blue",
                "result": "win" if score > 0.75 else ("draw" if score >= 0.25 else "loss"),
                "expected": round(expected, 3),
                "delta": round(new_rating - rating, 1),
                "rating": round(new_rating, 1),
                "plies": plies,
                "board_size": board_size,
            })
            user["history"] = user["history"][-200:]
            users[name] = user
            self._save_users(users)
            return user
