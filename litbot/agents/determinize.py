"""Build a perfect-information GameState from an observation + sampled hands."""

from __future__ import annotations

from litbot.game import GameState, Observation


def state_from_sample(obs: Observation, hands: list[int]) -> GameState:
    return GameState(
        spec=obs.spec,
        hands=hands[:],
        current_player=obs.current_player,
        scores=list(obs.scores),
        claimed=list(obs.claimed),
        history=[],
        plies=obs.plies,
    )


def info_set_actions(obs: Observation) -> list:
    from litbot.agents.random_agent import _solo_claims, _transfers
    from litbot.game import legal_asks_from

    transfers = _transfers(obs)
    if transfers:
        return transfers
    return _solo_claims(obs) + legal_asks_from(
        obs.spec, obs.player, obs.hand, obs.hand_sizes, obs.claimed
    )
