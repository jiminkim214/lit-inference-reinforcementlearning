"""Tiny linear value network for the RL stretch goal.

Features come from one player's observation (hand sizes, scores, how many
cards of each half-suit the observer holds). The target is the eventual
set-difference for the observer's team. This is not DouZero; it is a
working, trainable baseline that ISMCTS can use as a cutoff heuristic
once trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from litbot.cards import popcount
from litbot.game import Observation


def features(obs: Observation) -> np.ndarray:
    spec = obs.spec
    xs: list[float] = []
    xs.extend(h / 8.0 for h in obs.hand_sizes)
    xs.append(obs.scores[0] / spec.win_threshold)
    xs.append(obs.scores[1] / spec.win_threshold)
    xs.append(1.0 if spec.team(obs.player) == 0 else 0.0)
    for hs in range(spec.n_halfsuits):
        owned = 1.0 if obs.claimed[hs] == spec.team(obs.player) else 0.0
        lost = 1.0 if obs.claimed[hs] == 1 - spec.team(obs.player) else 0.0
        mine = popcount(obs.hand & spec.halfsuit_mask(hs)) / spec.ranks_per_half
        xs.extend([owned, lost, mine])
    return np.asarray(xs, dtype=np.float64)


@dataclass
class LinearValue:
    w: np.ndarray
    b: float = 0.0

    def predict(self, obs: Observation) -> float:
        x = features(obs)
        return float(self.w @ x + self.b)

    def save(self, path) -> None:
        np.savez(path, w=self.w, b=np.array([self.b]))

    @classmethod
    def load(cls, path) -> "LinearValue":
        data = np.load(path)
        return cls(w=data["w"], b=float(data["b"][0]))


def train_from_games(
    games: Iterable,
    l2: float = 1e-3,
) -> LinearValue:
    """Ordinary least squares: features -> team set difference.

    `games` yields (list of Observation from one seat, final set-diff for that team).
    """
    xs = []
    ys = []
    for traj, y in games:
        for obs in traj:
            xs.append(features(obs))
            ys.append(y)
    if not xs:
        raise ValueError("no training data")
    X = np.stack(xs)
    y = np.asarray(ys, dtype=np.float64)
    ones = np.ones((X.shape[0], 1))
    A = np.concatenate([X, ones], axis=1)
    n_feat = A.shape[1]
    ridge = A.T @ A + l2 * np.eye(n_feat)
    theta = np.linalg.solve(ridge, A.T @ y)
    return LinearValue(w=theta[:-1], b=float(theta[-1]))
