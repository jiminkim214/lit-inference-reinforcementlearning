"""Public constraints implied by a player's observation.

Successful asks reveal a card's current holder (until it is claimed or
asked away). Failed asks forbid that target from holding an unmoved card.
Claims reveal and remove a whole half-suit.

Historical "must hold a card of that half-suit to have asked" constraints
are checked by reversing a candidate current assignment to an initial deal
and replaying the history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from litbot.cards import Spec, iter_bits, popcount
from litbot.game import AskEvent, ClaimEvent, Observation, PassEvent


@dataclass
class Constraints:
    spec: Spec
    observer: int
    observer_hand: int
    hand_sizes: tuple[int, ...]
    claimed: tuple[int, ...]
    # known_loc[card] = player id, or -1 unknown, or -2 out of play (claimed)
    known_loc: list[int]
    # forbidden[card] = bitmask of players who cannot currently hold it
    forbidden: list[int]
    # Cards whose original holder never changed in public (never successfully asked)
    unmoved: list[bool]

    def unknown_cards(self) -> list[int]:
        return [
            c
            for c in range(self.spec.n_cards)
            if self.known_loc[c] == -1
        ]

    def remaining_capacity(self) -> list[int]:
        """Slots still to fill per player after placing known cards."""
        cap = list(self.hand_sizes)
        for c, loc in enumerate(self.known_loc):
            if loc >= 0:
                cap[loc] -= 1
        return cap

    def is_forbidden(self, player: int, card: int) -> bool:
        if self.known_loc[card] == -2:
            return True
        if self.known_loc[card] >= 0:
            return player != self.known_loc[card]
        if player == self.observer:
            return True  # observer's unknown cards are already in observer_hand
        return (self.forbidden[card] >> player) & 1 == 1


def constraints_from_obs(obs: Observation) -> Constraints:
    spec = obs.spec
    known = [-1] * spec.n_cards
    forbidden = [0] * spec.n_cards
    unmoved = [True] * spec.n_cards

    for card in iter_bits(obs.hand):
        known[card] = obs.player

    for event in obs.history:
        if isinstance(event, AskEvent):
            if event.success:
                known[event.card] = event.asker
                unmoved[event.card] = False
                # The target no longer has it; the asker does (public).
                forbidden[event.card] = 0
            else:
                # Target did not have the card at this moment. If it has not
                # moved since (and never moved before), they still don't.
                if unmoved[event.card]:
                    forbidden[event.card] |= 1 << event.target
                elif known[event.card] != event.target:
                    forbidden[event.card] |= 1 << event.target
        elif isinstance(event, ClaimEvent):
            for i, _owner in enumerate(event.actual):
                card = event.halfsuit * spec.ranks_per_half + i
                known[card] = -2
                unmoved[card] = False
                forbidden[card] = (1 << spec.n_players) - 1

    # Observer cannot hold unknown cards; known observer cards already placed.
    for c in range(spec.n_cards):
        if known[c] == -1:
            forbidden[c] |= 1 << obs.player

    return Constraints(
        spec=spec,
        observer=obs.player,
        observer_hand=obs.hand,
        hand_sizes=obs.hand_sizes,
        claimed=obs.claimed,
        known_loc=known,
        forbidden=forbidden,
        unmoved=unmoved,
    )


def hands_from_assignment(
    obs: Observation,
    assignment: dict[int, int],
    cons: Optional[Constraints] = None,
) -> list[int]:
    """Build current unclaimed hands from observer hand + unknown-card map."""
    spec = obs.spec
    cons = cons or constraints_from_obs(obs)
    hands = [0] * spec.n_players
    for c, loc in enumerate(cons.known_loc):
        if loc >= 0:
            hands[loc] |= 1 << c
    for card, player in assignment.items():
        hands[player] |= 1 << card
    return hands


def reconstruct_initial(hands: list[int], obs: Observation) -> list[int]:
    """Reverse public history to recover the initial deal from current hands."""
    spec = obs.spec
    h = hands[:]
    for event in reversed(obs.history):
        if isinstance(event, ClaimEvent):
            start = event.halfsuit * spec.ranks_per_half
            for i, owner in enumerate(event.actual):
                h[owner] |= 1 << (start + i)
        elif isinstance(event, AskEvent) and event.success:
            bit = 1 << event.card
            h[event.asker] &= ~bit
            h[event.target] |= bit
    return h


def replay_consistent(initial: list[int], obs: Observation) -> bool:
    """True iff `initial` produces the observed history with no rule break."""
    spec = obs.spec
    hands = initial[:]
    for event in obs.history:
        if isinstance(event, AskEvent):
            asker, target, card = event.asker, event.target, event.card
            hs = spec.halfsuit_of(card)
            if hands[asker] & spec.halfsuit_mask(hs) == 0:
                return False
            bit = 1 << card
            if hands[asker] & bit:
                return False
            if hands[target] == 0:
                return False
            has = (hands[target] & bit) != 0
            if has != event.success:
                return False
            if has:
                hands[target] &= ~bit
                hands[asker] |= bit
        elif isinstance(event, ClaimEvent):
            start = event.halfsuit * spec.ranks_per_half
            for i, owner in enumerate(event.actual):
                bit = 1 << (start + i)
                if hands[owner] & bit == 0:
                    return False
                hands[owner] &= ~bit
        elif isinstance(event, PassEvent):
            if hands[event.src] != 0:
                return False
            if hands[event.dest] == 0:
                return False
    # Current hands of the observer must match, and hand sizes must match.
    if hands[obs.player] != obs.hand:
        return False
    for p in range(spec.n_players):
        if popcount(hands[p]) != obs.hand_sizes[p]:
            return False
    return True


def assignment_consistent(
    obs: Observation, assignment: dict[int, int], cons: Optional[Constraints] = None
) -> bool:
    cons = cons or constraints_from_obs(obs)
    # Cheap filters.
    cap = cons.remaining_capacity()
    used = [0] * obs.spec.n_players
    for card, player in assignment.items():
        if cons.is_forbidden(player, card):
            return False
        used[player] += 1
    if used != cap:
        return False
    hands = hands_from_assignment(obs, assignment, cons)
    initial = reconstruct_initial(hands, obs)
    return replay_consistent(initial, obs)
