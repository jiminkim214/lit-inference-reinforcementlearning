"""Perfect-Information Monte Carlo.

Sample worlds consistent with the current information set, play the
clairvoyant greedy policy in each world, and majority-vote among the
actions that are legal from the information set.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Optional

from litbot.agents.base import Agent
from litbot.agents.determinize import state_from_sample
from litbot.agents.greedy import GreedyAgent
from litbot.agents.random_agent import _solo_claims, _transfers
from litbot.belief.engine import BeliefEngine
from litbot.game import (
    Move,
    Observation,
    apply_move,
    legal_asks_from,
    perfect_info_policy,
)


class PIMCAgent(Agent):
    name = "pimc"

    def __init__(
        self,
        n_samples: int = 24,
        inner_depth: int = 0,
        use_opponent_model: bool = False,
        seed: Optional[int] = None,
        claim_threshold: float = 0.9,
    ):
        self.n_samples = n_samples
        self.inner_depth = inner_depth
        self.engine = BeliefEngine(
            n_samples=n_samples,
            use_opponent_model=use_opponent_model,
            seed=0 if seed is None else seed,
        )
        self.rng = random.Random(seed)
        self.claim_threshold = claim_threshold
        self._greedy = GreedyAgent(
            n_samples=n_samples,
            claim_threshold=claim_threshold,
            use_opponent_model=use_opponent_model,
            seed=seed,
        )
        if use_opponent_model:
            self.name = "pimc-om"

    def select(self, obs: Observation) -> Move:
        solo = _solo_claims(obs)
        if solo:
            return solo[0]
        transfers = _transfers(obs)
        if transfers:
            return transfers[self.rng.randrange(len(transfers))]

        belief = self.engine.infer(obs)
        legal_asks = legal_asks_from(
            obs.spec, obs.player, obs.hand, obs.hand_sizes, obs.claimed
        )
        legal_set = set(legal_asks)
        if not belief.samples:
            return self._greedy.select(obs)

        ask_votes: Counter[Move] = Counter()
        claim_votes: Counter[Move] = Counter()
        total_w = 0.0
        for hands, w in belief.sampled_state_hands():
            world = state_from_sample(obs, hands)
            try:
                move = _clairvoyant_action(world, self.rng, self.inner_depth)
            except Exception:
                continue
            total_w += w
            if move[0] == "ask" and move in legal_set:
                ask_votes[move] += w
            elif move[0] == "claim":
                team = obs.spec.team(obs.player)
                if all(obs.spec.team(p) == team for p in move[2]):
                    claim_votes[move] += w

        if total_w > 0 and claim_votes:
            claim, cw = claim_votes.most_common(1)[0]
            if cw / total_w >= 0.5:
                return claim
        if ask_votes:
            top = max(ask_votes.values())
            best = [m for m, v in ask_votes.items() if v == top]
            return best[self.rng.randrange(len(best))]
        return self._greedy.select(obs)


def _clairvoyant_action(state, rng: random.Random, inner_depth: int) -> Move:
    if inner_depth <= 0:
        return perfect_info_policy(state, rng)
    # One-ply expectimax over perfect-info replies: try every hit-ask and
    # every correct claim, take the one whose greedy continuation looks best.
    from litbot.game import legal_asks, perfect_info_claims, legal_transfers

    claims = perfect_info_claims(state)
    if claims:
        return claims[0]
    transfers = legal_transfers(state)
    if transfers:
        return transfers[rng.randrange(len(transfers))]
    asks = legal_asks(state)
    hits = [m for m in asks if state.hands[m[1]] & (1 << m[2])]
    pool = hits if hits else asks
    team = state.spec.team(state.current_player)
    best_move = pool[0]
    best_val = -1e9
    for move in pool[:12]:
        nxt = apply_move(state, move, record=False)
        val = _rollout_value(nxt, team, rng, plies=inner_depth)
        if val > best_val:
            best_val = val
            best_move = move
    return best_move


def _rollout_value(state, team: int, rng: random.Random, plies: int) -> float:
    s = state.clone(record_history=False)
    for _ in range(plies):
        if s.is_terminal():
            break
        try:
            mv = perfect_info_policy(s, rng)
        except Exception:
            break
        s = apply_move(s, mv, record=False)
    if s.is_terminal():
        w = s.winner()
        if w is None:
            return 0.0
        return 1.0 if w == team else -1.0
    return (s.scores[team] - s.scores[1 - team]) / max(s.spec.win_threshold, 1)
