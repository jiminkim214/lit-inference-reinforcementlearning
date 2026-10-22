"""Brute-force enumeration of deals consistent with an observation.

Used as ground truth on toy decks (and on full-game states that have few
unknown cards). Assignments of unknown cards are generated under capacity
and forbidden-location constraints, then filtered by history replay.
"""

from __future__ import annotations

from typing import Iterator

from litbot.belief.constraints import (
    Constraints,
    assignment_consistent,
    constraints_from_obs,
)
from litbot.game import Observation


def iter_assignments(
    unknown: list[int],
    capacities: list[int],
    forbidden_fn,
) -> Iterator[dict[int, int]]:
    """Yield maps unknown-card -> player that fill `capacities` exactly."""
    n = len(unknown)
    cap = capacities[:]
    assign: dict[int, int] = {}

    def rec(i: int) -> Iterator[dict[int, int]]:
        if i == n:
            yield dict(assign)
            return
        card = unknown[i]
        for p, left in enumerate(cap):
            if left <= 0:
                continue
            if forbidden_fn(p, card):
                continue
            cap[p] -= 1
            assign[card] = p
            yield from rec(i + 1)
            del assign[card]
            cap[p] += 1

    yield from rec(0)


def enumerate_consistent(
    obs: Observation, cons: Constraints | None = None, limit: int | None = None
) -> list[dict[int, int]]:
    cons = cons or constraints_from_obs(obs)
    unknown = cons.unknown_cards()
    cap = cons.remaining_capacity()
    if any(c < 0 for c in cap):
        return []
    out: list[dict[int, int]] = []
    for asn in iter_assignments(unknown, cap, cons.is_forbidden):
        if assignment_consistent(obs, asn, cons):
            out.append(asn)
            if limit is not None and len(out) >= limit:
                break
    return out


def exact_marginals(
    obs: Observation, cons: Constraints | None = None
) -> tuple[list[list[float]], int]:
    """Return (marginal[card][player], n_worlds). Unseen cards only; known cards are 0/1."""
    cons = cons or constraints_from_obs(obs)
    spec = obs.spec
    worlds = enumerate_consistent(obs, cons)
    n = len(worlds)
    marg = [[0.0] * spec.n_players for _ in range(spec.n_cards)]
    if n == 0:
        return marg, 0
    for c, loc in enumerate(cons.known_loc):
        if loc >= 0:
            marg[c][loc] = 1.0
    for asn in worlds:
        for card, player in asn.items():
            marg[card][player] += 1.0
    if n:
        for card in cons.unknown_cards():
            for p in range(spec.n_players):
                marg[card][p] /= n
    return marg, n
