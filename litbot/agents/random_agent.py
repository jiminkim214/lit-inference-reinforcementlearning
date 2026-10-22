"""Uniform random legal asks; always claims a half-suit held entirely in hand.

When no ask is legal (endgame / remaining cards on our team), claims a
half-suit using a random assignment of unknown cards to teammates.
"""

from __future__ import annotations

import random
from typing import Optional

from litbot.agents.base import Agent
from litbot.game import (
    Observation,
    Move,
    legal_asks_from,
    iter_unclaimed_halfsuits,
)


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None, claim_prob: float = 0.06):
        self.rng = random.Random(seed)
        self.claim_prob = claim_prob

    def select(self, obs: Observation) -> Move:
        solo = _solo_claims(obs)
        if solo:
            return solo[self.rng.randrange(len(solo))]
        transfers = _transfers(obs)
        if transfers:
            return transfers[self.rng.randrange(len(transfers))]
        asks = legal_asks_from(
            obs.spec, obs.player, obs.hand, obs.hand_sizes, obs.claimed
        )
        if asks:
            if self.claim_prob > 0 and self.rng.random() < self.claim_prob:
                guess = _forced_claims(obs, self.rng)
                if guess is not None:
                    return guess
            return asks[self.rng.randrange(len(asks))]
        forced = _forced_claims(obs, self.rng)
        if forced:
            return forced
        raise RuntimeError("random agent has no legal move")


def _transfers(obs: Observation) -> list[Move]:
    if obs.hand_sizes[obs.player] > 0:
        return []
    spec = obs.spec
    mates = [t for t in spec.teammates(obs.player) if obs.hand_sizes[t] > 0]
    if mates:
        return [("pass", t) for t in mates]
    opps = [o for o in spec.opponents(obs.player) if obs.hand_sizes[o] > 0]
    return [("designate", o) for o in opps]


def _solo_claims(obs: Observation) -> list[Move]:
    spec = obs.spec
    p = obs.player
    out: list[Move] = []
    for hs in iter_unclaimed_halfsuits(obs.claimed):
        hmask = spec.halfsuit_mask(hs)
        if (obs.hand & hmask) == hmask:
            out.append(("claim", hs, tuple(p for _ in range(spec.ranks_per_half))))
    return out


def _forced_claims(obs: Observation, rng: random.Random) -> Optional[Move]:
    spec = obs.spec
    p = obs.player
    team = spec.team_players(spec.team(p))
    options: list[Move] = []
    for hs in iter_unclaimed_halfsuits(obs.claimed):
        named = []
        for card in spec.cards_of_halfsuit(hs):
            if obs.hand & (1 << card):
                named.append(p)
            else:
                named.append(rng.choice(team))
        options.append(("claim", hs, tuple(named)))
    if not options:
        return None
    return options[rng.randrange(len(options))]
