"""Bradley-Terry ratings on the Elo scale.

A round-robin gives every pairing's win counts; what a table wants is one
number per agent.  The maximum-likelihood fit below is *joint* -- beating a
strong opponent is worth more than beating a weak one, and the error bars come
from the likelihood's own information matrix -- which is why the ladder re-runs
it over the whole field rather than updating ratings game by game.

Game-independent: it only ever sees names and win counts.
"""

from __future__ import annotations

import math

import numpy as np


def fit_elo(
    results: list[tuple[str, str, int, int]],
    anchor: str | None = None,
    anchor_rating: float = 0.0,
    iterations: int = 3000,
    prior_games: float = 1.0,
) -> dict[str, float]:
    """Maximum-likelihood Bradley-Terry ratings on the Elo scale.

    ``results`` is a list of ``(player_a, player_b, a_wins, b_wins)``.  A small
    symmetric prior (``prior_games`` virtual drawn games against a phantom
    average opponent) keeps ratings finite when someone wins or loses every
    game -- which happens a lot when the field spans random to AlphaZero.
    """
    players = sorted({p for r in results for p in (r[0], r[1])})
    index = {p: i for i, p in enumerate(players)}
    rating = np.zeros(len(players))  # natural (logistic) scale

    wins = np.zeros((len(players), len(players)))
    for a, b, aw, bw in results:
        ia, ib = index[a], index[b]
        wins[ia, ib] += aw
        wins[ib, ia] += bw

    games = wins + wins.T
    total_wins = wins.sum(axis=1) + prior_games * 0.5
    for _ in range(iterations):
        # Minorization-maximization update for Bradley-Terry.
        gamma = np.exp(rating)
        denom = np.zeros(len(players))
        for i in range(len(players)):
            mask = games[i] > 0
            if mask.any():
                denom[i] = (games[i][mask] / (gamma[i] + gamma[mask])).sum()
        denom += prior_games / (gamma + np.exp(np.mean(rating)))
        new_gamma = total_wins / np.maximum(denom, 1e-12)
        new_rating = np.log(np.maximum(new_gamma, 1e-12))
        new_rating -= new_rating.mean()
        if np.abs(new_rating - rating).max() < 1e-10:
            rating = new_rating
            break
        rating = new_rating

    scale = 400.0 / math.log(10.0)
    elo = {p: float(rating[index[p]] * scale) for p in players}
    if anchor is not None and anchor in elo:
        shift = anchor_rating - elo[anchor]
        elo = {p: v + shift for p, v in elo.items()}
    return elo


def elo_standard_errors(
    results: list[tuple[str, str, int, int]], elo: dict[str, float]
) -> dict[str, float]:
    """+/- 1 sigma on each rating, from the Bradley-Terry Fisher information.

    A per-player binomial error bar is wrong here (and diverges on a clean
    sweep).  The likelihood's information matrix uses who each player actually
    faced, so beating a strong opponent tightens the estimate more than
    beating a weak one.  The matrix is singular by one dimension -- ratings are
    only defined up to a shift -- so the pseudo-inverse gives errors relative
    to the field mean.
    """
    players = sorted({p for r in results for p in (r[0], r[1])})
    index = {p: i for i, p in enumerate(players)}
    k = len(players)
    scale = 400.0 / math.log(10.0)
    r = np.array([elo[p] / scale for p in players])

    info = np.zeros((k, k))
    for a, b, aw, bw in results:
        n = aw + bw
        if n == 0:
            continue
        ia, ib = index[a], index[b]
        p = 1.0 / (1.0 + np.exp(r[ib] - r[ia]))
        w = n * p * (1 - p)
        info[ia, ia] += w
        info[ib, ib] += w
        info[ia, ib] -= w
        info[ib, ia] -= w
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None)) * scale
    return {p: float(se[index[p]]) for p in players}
