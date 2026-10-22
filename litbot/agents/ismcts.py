"""Single-observer Information Set MCTS (Cowling, Powley, Whitehouse 2012).

Each iteration samples a determinization consistent with the root information
set, then runs one UCT traversal restricted to actions legal in that world.
Availability counts replace parent visit counts in UCB.

Claims are tree actions: in a determinization, the correct team-owned
half-suit claims are added alongside asks. A heuristic rollout cutoff stops
playouts after a fixed number of perfect-info greedy moves.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from litbot.agents.base import Agent
from litbot.agents.determinize import state_from_sample
from litbot.agents.greedy import GreedyAgent
from litbot.agents.random_agent import _solo_claims, _transfers
from litbot.belief.engine import BeliefEngine
from litbot.game import (
    GameState,
    Move,
    Observation,
    apply_move,
    legal_asks,
    legal_transfers,
    perfect_info_claims,
    perfect_info_policy,
)


class _Node:
    __slots__ = ("children", "n", "w", "avail")

    def __init__(self) -> None:
        self.children: dict[Move, _Node] = {}
        self.n: dict[Move, int] = {}
        self.w: dict[Move, float] = {}
        self.avail: dict[Move, int] = {}

    def ucb(self, move: Move, expl: float) -> float:
        visits = self.n.get(move, 0)
        avail = self.avail.get(move, 1)
        if visits == 0:
            return float("inf")
        q = self.w[move] / visits
        return q + expl * math.sqrt(math.log(avail) / visits)


class ISMCTSAgent(Agent):
    name = "ismcts"

    def __init__(
        self,
        iterations: int = 200,
        time_budget: Optional[float] = None,
        n_samples: int = 32,
        exploration: float = 1.4,
        rollout_cutoff: int = 24,
        use_opponent_model: bool = False,
        seed: Optional[int] = None,
    ):
        self.iterations = iterations
        self.time_budget = time_budget
        self.exploration = exploration
        self.rollout_cutoff = rollout_cutoff
        self.engine = BeliefEngine(
            n_samples=max(n_samples, 8),
            use_opponent_model=use_opponent_model,
            seed=0 if seed is None else seed,
        )
        self.rng = random.Random(seed)
        self._greedy = GreedyAgent(
            n_samples=n_samples,
            use_opponent_model=use_opponent_model,
            seed=seed,
        )
        if use_opponent_model:
            self.name = "ismcts-om"

    def select(self, obs: Observation) -> Move:
        solo = _solo_claims(obs)
        if solo:
            return solo[0]
        transfers = _transfers(obs)
        if transfers:
            return transfers[self.rng.randrange(len(transfers))]

        belief = self.engine.infer(obs)
        worlds = belief.sampled_state_hands()
        if not worlds:
            return self._greedy.select(obs)

        root = _Node()
        team = obs.spec.team(obs.player)
        deadline = (
            time.perf_counter() + self.time_budget if self.time_budget else None
        )
        n_iter = 0
        target = self.iterations
        while n_iter < target:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            hands, _w = worlds[n_iter % len(worlds)]
            world = state_from_sample(obs, hands)
            self._iterate(root, world, team, belief)
            n_iter += 1
            if deadline is None and n_iter >= target:
                break

        if not root.n:
            return self._greedy.select(obs)
        best = max(root.n.items(), key=lambda kv: kv[1])[0]
        return best

    def _iterate(self, root: _Node, state: GameState, root_team: int, belief) -> None:
        path: list[tuple[_Node, Move]] = []
        node = root
        s = state.clone(record_history=False)
        at_root = True

        # Selection / expansion
        while not s.is_terminal():
            if at_root:
                moves = _root_moves(s, belief, root_team)
                at_root = False
            else:
                moves = _world_moves(s)
            if not moves:
                break
            for m in moves:
                node.avail[m] = node.avail.get(m, 0) + 1
            untried = [m for m in moves if m not in node.children]
            if untried:
                move = untried[self.rng.randrange(len(untried))]
                child = _Node()
                node.children[move] = child
                path.append((node, move))
                s = apply_move(s, move, record=False)
                node = child
                break
            move = max(moves, key=lambda m: node.ucb(m, self.exploration))
            path.append((node, move))
            s = apply_move(s, move, record=False)
            node = node.children[move]

        reward = self._rollout(s, root_team)
        for parent, move in path:
            parent.n[move] = parent.n.get(move, 0) + 1
            parent.w[move] = parent.w.get(move, 0.0) + reward

    def _rollout(self, state: GameState, root_team: int) -> float:
        s = state
        rng = self.rng
        for _ in range(self.rollout_cutoff):
            if s.is_terminal():
                break
            try:
                mv = perfect_info_policy(s, rng)
            except Exception:
                break
            s = apply_move(s, mv, record=False)
        return _evaluate(s, root_team)


def _root_moves(state: GameState, belief, root_team: int) -> list[Move]:
    """Info-set asks plus only those claims the belief thinks the team owns."""
    transfers = legal_transfers(state)
    if transfers:
        return transfers
    moves = legal_asks(state)
    for claim in perfect_info_claims(state):
        hs = claim[1]
        if belief.team_owns_prob(hs, root_team) >= 0.45:
            moves.append(claim)
    return moves or _world_moves(state)


def _world_moves(state: GameState) -> list[Move]:
    transfers = legal_transfers(state)
    if transfers:
        return transfers
    return perfect_info_claims(state) + legal_asks(state)


def _evaluate(state: GameState, team: int) -> float:
    if state.is_terminal():
        w = state.winner()
        if w is None:
            return 0.0
        return 1.0 if w == team else -1.0
    diff = state.scores[team] - state.scores[1 - team]
    # Heuristic: owned remaining half-suits in this determinization.
    bonus = 0.0
    spec = state.spec
    remaining = 0
    for hs, owner in enumerate(state.claimed):
        if owner >= 0:
            continue
        remaining += 1
        holders = set()
        ok = True
        for card in spec.cards_of_halfsuit(hs):
            h = state.holder_of(card)
            if h is None:
                ok = False
                break
            holders.add(spec.team(h))
        if ok and holders == {team}:
            bonus += 1.0
        elif ok and holders == {1 - team}:
            bonus -= 1.0
    scale = max(spec.win_threshold, 1)
    return diff / scale + 0.15 * bonus / max(remaining, 1)
