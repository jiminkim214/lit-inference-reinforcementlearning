"""Simple opponent model: asking behavior as extra evidence.

A player who asks about a half-suit must hold at least one card of it at
that time (already a hard constraint). Soft evidence: they are more likely
to ask from half-suits they hold more of. Repeated asks about the same
half-suit therefore up-weight worlds where they hold several of those cards.

Likelihood of an observed ask given a reconstructed initial deal is
proportional to (cards of that half-suit in the asker's hand at that time)^alpha.
"""

from __future__ import annotations

from litbot.belief.constraints import reconstruct_initial
from litbot.cards import popcount
from litbot.game import AskEvent, ClaimEvent, Observation


def likelihood_of_history(
    obs: Observation,
    current_hands: list[int],
    alpha: float = 1.5,
) -> float:
    spec = obs.spec
    hands = reconstruct_initial(current_hands, obs)
    like = 1.0
    for event in obs.history:
        if isinstance(event, AskEvent):
            hs = spec.halfsuit_of(event.card)
            n = popcount(hands[event.asker] & spec.halfsuit_mask(hs))
            # Must be >= 1 if the deal is consistent; still guard.
            like *= (max(n, 0) ** alpha)
            bit = 1 << event.card
            if event.success:
                hands[event.target] &= ~bit
                hands[event.asker] |= bit
        elif isinstance(event, ClaimEvent):
            start = event.halfsuit * spec.ranks_per_half
            hmask = spec.halfsuit_mask(event.halfsuit)
            for p in range(spec.n_players):
                hands[p] &= ~hmask
    return like


def weight_fn(obs: Observation, cons, alpha: float = 1.5):
    from litbot.belief.constraints import hands_from_assignment

    def _w(assignment: dict[int, int]) -> float:
        hands = hands_from_assignment(obs, assignment, cons)
        return likelihood_of_history(obs, hands, alpha=alpha)

    return _w
