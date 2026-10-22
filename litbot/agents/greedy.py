"""Greedy agent: ask for the card with the highest estimated chance of success.

Claims when the MAP assignment of a half-suit to teammates is confident
enough, and always claims a half-suit held entirely in its own hand.
"""

from __future__ import annotations

import random
from typing import Optional

from litbot.agents.base import Agent
from litbot.agents.random_agent import _solo_claims, _transfers, _forced_claims
from litbot.belief.engine import Belief, BeliefEngine
from litbot.game import Move, Observation, legal_asks_from, iter_unclaimed_halfsuits


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(
        self,
        n_samples: int = 48,
        claim_threshold: float = 0.92,
        use_opponent_model: bool = False,
        seed: Optional[int] = None,
    ):
        self.engine = BeliefEngine(
            n_samples=n_samples,
            use_opponent_model=use_opponent_model,
            seed=0 if seed is None else seed,
        )
        self.claim_threshold = claim_threshold
        self.rng = random.Random(seed)
        if use_opponent_model:
            self.name = "greedy-om"

    def select(self, obs: Observation) -> Move:
        solo = _solo_claims(obs)
        if solo:
            return solo[0]
        transfers = _transfers(obs)
        if transfers:
            return transfers[self.rng.randrange(len(transfers))]
        belief = self.engine.infer(obs)
        claim = _best_claim(obs, belief, self.claim_threshold)
        if claim is not None:
            return claim
        asks = legal_asks_from(
            obs.spec, obs.player, obs.hand, obs.hand_sizes, obs.claimed
        )
        if not asks:
            forced = _forced_claims(obs, self.rng)
            if forced:
                return forced
            raise RuntimeError("greedy agent has no legal move")
        best_p = -1.0
        best: list[Move] = []
        for move in asks:
            p = belief.p_success(move[1], move[2])
            if p > best_p + 1e-9:
                best_p = p
                best = [move]
            elif abs(p - best_p) <= 1e-9:
                best.append(move)
        return best[self.rng.randrange(len(best))]


def _best_claim(
    obs: Observation, belief: Belief, threshold: float
) -> Optional[Move]:
    spec = obs.spec
    team = spec.team(obs.player)
    mates = set(spec.team_players(team))
    best_move = None
    best_conf = threshold
    for hs in iter_unclaimed_halfsuits(obs.claimed):
        named = belief.map_holders(hs)
        if any(n not in mates for n in named):
            continue
        # Own cards must be assigned to self.
        ok = True
        for i, card in enumerate(spec.cards_of_halfsuit(hs)):
            if obs.hand & (1 << card) and named[i] != obs.player:
                ok = False
                break
        if not ok:
            continue
        conf = belief.claim_confidence(hs, named)
        if conf > best_conf:
            best_conf = conf
            best_move = ("claim", hs, named)
    return best_move
