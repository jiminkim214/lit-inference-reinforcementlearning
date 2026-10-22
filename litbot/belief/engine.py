"""Belief engine facade: marginals, MAP claims, sampled worlds."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from litbot.belief.constraints import (
    Constraints,
    constraints_from_obs,
    hands_from_assignment,
)
from litbot.belief.counting import dp_marginals
from litbot.belief.enumerator import exact_marginals
from litbot.belief.opponent import weight_fn
from litbot.belief.sampler import sample_worlds
from litbot.cards import Spec
from litbot.game import Observation


@dataclass
class Belief:
    obs: Observation
    cons: Constraints
    samples: list[tuple[dict[int, int], float]]
    marginals: list[list[float]]  # [card][player]

    def p_has(self, player: int, card: int) -> float:
        return self.marginals[card][player]

    def p_success(self, target: int, card: int) -> float:
        return self.p_has(target, card)

    def map_holders(self, halfsuit: int) -> tuple[int, ...]:
        spec = self.obs.spec
        named = []
        for card in spec.cards_of_halfsuit(halfsuit):
            probs = self.marginals[card]
            named.append(max(range(spec.n_players), key=lambda p: probs[p]))
        return tuple(named)

    def claim_confidence(self, halfsuit: int, named: tuple[int, ...]) -> float:
        """Fraction of sample weight in which `named` is exactly correct."""
        spec = self.obs.spec
        cards = spec.cards_of_halfsuit(halfsuit)
        total = 0.0
        hit = 0.0
        for asn, w in self.samples:
            total += w
            holders = []
            ok = True
            for card in cards:
                loc = self.cons.known_loc[card]
                if loc == -2:
                    ok = False
                    break
                if loc >= 0:
                    holders.append(loc)
                else:
                    holders.append(asn[card])
            if ok and tuple(holders) == named:
                hit += w
        if total == 0:
            return 0.0
        return hit / total

    def team_owns_prob(self, halfsuit: int, team: int) -> float:
        spec = self.obs.spec
        cards = spec.cards_of_halfsuit(halfsuit)
        total = 0.0
        hit = 0.0
        for asn, w in self.samples:
            total += w
            owns = True
            for card in cards:
                loc = self.cons.known_loc[card]
                if loc == -2:
                    owns = False
                    break
                holder = loc if loc >= 0 else asn[card]
                if spec.team(holder) != team:
                    owns = False
                    break
            if owns:
                hit += w
        if total == 0:
            return 0.0
        return hit / total

    def sampled_state_hands(self) -> list[tuple[list[int], float]]:
        return [
            (hands_from_assignment(self.obs, asn, self.cons), w)
            for asn, w in self.samples
        ]


class BeliefEngine:
    def __init__(
        self,
        n_samples: int = 64,
        use_opponent_model: bool = False,
        opponent_alpha: float = 1.5,
        seed: int = 0,
        method: str = "sample",
    ):
        self.n_samples = n_samples
        self.use_opponent_model = use_opponent_model
        self.opponent_alpha = opponent_alpha
        self.rng = random.Random(seed)
        self.method = method  # "sample" | "enumerate" | "dp"

    def infer(self, obs: Observation) -> Belief:
        cons = constraints_from_obs(obs)
        weight = (
            weight_fn(obs, cons, alpha=self.opponent_alpha)
            if self.use_opponent_model
            else None
        )
        if self.method == "enumerate":
            from litbot.belief.enumerator import enumerate_consistent

            worlds = enumerate_consistent(obs, cons)
            samples = [(asn, 1.0) for asn in worlds[: self.n_samples]]
            if not samples:
                samples = sample_worlds(obs, self.n_samples, self.rng, cons, opponent_weight=weight)
            marg, _n = exact_marginals(obs, cons)
        elif self.method == "dp":
            samples = sample_worlds(
                obs, self.n_samples, self.rng, cons, opponent_weight=weight
            )
            marg, _n = dp_marginals(obs, cons)
        else:
            samples = sample_worlds(
                obs, self.n_samples, self.rng, cons, opponent_weight=weight
            )
            marg = _marginals_from_samples(obs.spec, cons, samples)
        if self.use_opponent_model and samples:
            marg = _marginals_from_samples(obs.spec, cons, samples)
        return Belief(obs=obs, cons=cons, samples=samples, marginals=marg)


def _marginals_from_samples(
    spec: Spec, cons: Constraints, samples: list[tuple[dict[int, int], float]]
) -> list[list[float]]:
    marg = [[0.0] * spec.n_players for _ in range(spec.n_cards)]
    for c, loc in enumerate(cons.known_loc):
        if loc >= 0:
            marg[c][loc] = 1.0
    total = sum(w for _a, w in samples)
    if total <= 0:
        return marg
    acc = [[0.0] * spec.n_players for _ in range(spec.n_cards)]
    for asn, w in samples:
        for card, player in asn.items():
            acc[card][player] += w
    for card in cons.unknown_cards():
        for p in range(spec.n_players):
            marg[card][p] = acc[card][p] / total
    return marg
