from __future__ import annotations

from litbot.game import Move, Observation


class Agent:
    name: str = "agent"

    def select(self, obs: Observation) -> Move:
        raise NotImplementedError

    def reset(self) -> None:
        return None
