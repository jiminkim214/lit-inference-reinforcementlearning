"""JSON helpers for the web UI and game logs."""

from __future__ import annotations

from litbot.cards import card_name, halfsuit_name, iter_bits
from litbot.game import Observation


def obs_to_dict(obs: Observation) -> dict:
    spec = obs.spec
    hand = [
        {
            "id": c,
            "name": card_name(c, spec),
            "halfsuit": spec.halfsuit_of(c),
            "halfsuit_name": halfsuit_name(spec.halfsuit_of(c), spec),
        }
        for c in iter_bits(obs.hand)
    ]
    history = [e.describe(spec) for e in obs.history[-40:]]
    events = [e.to_dict(spec) for e in obs.history[-40:]]
    unclaimed = [
        {"id": hs, "name": halfsuit_name(hs, spec)}
        for hs, owner in enumerate(obs.claimed)
        if owner < 0
    ]
    return {
        "player": obs.player,
        "current_player": obs.current_player,
        "hand": hand,
        "hand_sizes": list(obs.hand_sizes),
        "scores": list(obs.scores),
        "claimed": list(obs.claimed),
        "claimed_names": [
            None if owner < 0 else halfsuit_name(hs, spec)
            for hs, owner in enumerate(obs.claimed)
        ],
        "history": history,
        "events": events,
        "plies": obs.plies,
        "terminal": obs.is_terminal(),
        "unclaimed": unclaimed,
        "halfsuits": [
            {
                "id": hs,
                "name": halfsuit_name(hs, spec),
                "cards": [
                    {"id": c, "name": card_name(c, spec)}
                    for c in spec.cards_of_halfsuit(hs)
                ],
                "owner": obs.claimed[hs],
            }
            for hs in range(spec.n_halfsuits)
        ],
        "teammates": list(spec.teammates(obs.player)),
        "opponents": list(spec.opponents(obs.player)),
        "win_threshold": spec.win_threshold,
    }
