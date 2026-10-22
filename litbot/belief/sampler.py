"""Markov-chain sampler over current card locations.

Unknown cards are assigned to players under capacity and forbidden-location
constraints. A random swap chain mixes the assignment; samples that fail
the history-replay check are discarded. When few cards are unknown we fall
back to exact enumeration.
"""

from __future__ import annotations

import random
from typing import Optional

from litbot.belief.constraints import (
    Constraints,
    assignment_consistent,
    constraints_from_obs,
)
from litbot.belief.enumerator import enumerate_consistent
from litbot.game import Observation

ENUMERATE_MAX_UNKNOWN = 7


def _random_assignment(
    cons: Constraints, rng: random.Random, attempts: int = 4000
) -> Optional[dict[int, int]]:
    unknown = cons.unknown_cards()
    if not unknown:
        return {}
    n_p = cons.spec.n_players
    for _ in range(attempts):
        cap = cons.remaining_capacity()
        if any(c < 0 for c in cap):
            return None
        order = unknown[:]
        rng.shuffle(order)
        asn: dict[int, int] = {}
        ok = True
        for card in order:
            choices = [p for p in range(n_p) if cap[p] > 0 and not cons.is_forbidden(p, card)]
            if not choices:
                ok = False
                break
            p = rng.choice(choices)
            asn[card] = p
            cap[p] -= 1
        if ok and all(c == 0 for c in cap):
            return asn
    return None


def _swap(asn: dict[int, int], cons: Constraints, rng: random.Random) -> bool:
    """Try one random swap of two cards; mutate asn in place. Return whether swapped."""
    cards = list(asn.keys())
    if len(cards) < 2:
        return False
    a, b = rng.sample(cards, 2)
    pa, pb = asn[a], asn[b]
    if pa == pb:
        return False
    if cons.is_forbidden(pb, a) or cons.is_forbidden(pa, b):
        return False
    asn[a], asn[b] = pb, pa
    return True


def sample_worlds(
    obs: Observation,
    n_samples: int,
    rng: random.Random,
    cons: Optional[Constraints] = None,
    burn: int = 80,
    sweep: int = 24,
    opponent_weight=None,
) -> list[tuple[dict[int, int], float]]:
    """Return a list of (assignment, weight).

    `opponent_weight(assignment) -> float` is an optional likelihood.
    """
    cons = cons or constraints_from_obs(obs)
    unknown = cons.unknown_cards()
    if len(unknown) <= ENUMERATE_MAX_UNKNOWN:
        worlds = enumerate_consistent(obs, cons)
        if not worlds:
            return []
        out = []
        if n_samples >= len(worlds):
            chosen = worlds
        else:
            chosen = [worlds[i] for i in rng.sample(range(len(worlds)), n_samples)]
        for asn in chosen:
            w = 1.0 if opponent_weight is None else max(opponent_weight(asn), 1e-12)
            out.append((asn, w))
        return out

    asn = _random_assignment(cons, rng)
    if asn is None:
        return []

    # Mix until we land on a history-consistent state.
    consistent = None
    for _ in range(max(burn * 8, 400)):
        for _s in range(sweep):
            _swap(asn, cons, rng)
        if assignment_consistent(obs, asn, cons):
            consistent = dict(asn)
            break
    if consistent is None:
        # Last resort: return whatever cheap-constraint samples we can, unfiltered.
        consistent = dict(asn)

    asn = consistent
    samples: list[tuple[dict[int, int], float]] = []
    guard = 0
    max_tries = max(n_samples * 20, 200)
    while len(samples) < n_samples and guard < max_tries:
        guard += 1
        for _s in range(sweep):
            _swap(asn, cons, rng)
        if not assignment_consistent(obs, asn, cons):
            continue
        w = 1.0 if opponent_weight is None else max(opponent_weight(asn), 1e-12)
        samples.append((dict(asn), w))
    return samples


def sample_hands(
    obs: Observation,
    n_samples: int,
    rng: random.Random,
    cons: Optional[Constraints] = None,
    opponent_weight=None,
) -> list[tuple[list[int], float]]:
    from litbot.belief.constraints import hands_from_assignment

    cons = cons or constraints_from_obs(obs)
    worlds = sample_worlds(obs, n_samples, rng, cons, opponent_weight=opponent_weight)
    return [(hands_from_assignment(obs, asn, cons), w) for asn, w in worlds]
