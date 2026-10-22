from __future__ import annotations

from typing import Optional

from litbot.agents.base import Agent
from litbot.agents.greedy import GreedyAgent
from litbot.agents.ismcts import ISMCTSAgent
from litbot.agents.pimc import PIMCAgent
from litbot.agents.random_agent import RandomAgent


def make_agent(kind: str, seed: Optional[int] = None, **kwargs) -> Agent:
    kind = kind.lower().replace("_", "-")
    if kind in ("random", "rand"):
        return RandomAgent(seed=seed)
    if kind in ("greedy",):
        return GreedyAgent(seed=seed, **kwargs)
    if kind in ("greedy-om", "greedy_om"):
        return GreedyAgent(seed=seed, use_opponent_model=True, **kwargs)
    if kind in ("pimc",):
        return PIMCAgent(seed=seed, **kwargs)
    if kind in ("pimc-om",):
        return PIMCAgent(seed=seed, use_opponent_model=True, **kwargs)
    if kind in ("ismcts",):
        return ISMCTSAgent(seed=seed, **kwargs)
    if kind in ("ismcts-om",):
        return ISMCTSAgent(seed=seed, use_opponent_model=True, **kwargs)
    raise ValueError(f"unknown agent {kind}")
