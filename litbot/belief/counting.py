"""Exact location counting by dynamic programming on remaining capacities.

This ignores historical must-hold constraints (those need a replay filter)
and only uses current capacities plus forbidden locations. On toy decks the
enumerator in `enumerator.py` is the ground truth; this module is the
"exact counting by dynamic programming" comparison for the location
marginals of unseen cards, and is profiled against the Markov sampler.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from litbot.belief.constraints import Constraints, constraints_from_obs
from litbot.game import Observation


def count_locations(obs: Observation, cons: Constraints | None = None):
    """Return (counts[i][player], total) for unknown cards (index i)."""
    cons = cons or constraints_from_obs(obs)
    unknown = cons.unknown_cards()
    cap0 = tuple(cons.remaining_capacity())
    n_p = obs.spec.n_players
    n = len(unknown)
    if n == 0:
        return [], 1
    if any(c < 0 for c in cap0):
        return [[0] * n_p for _ in range(n)], 0

    legal = []
    for card in unknown:
        legal.append(tuple(p for p in range(n_p) if not cons.is_forbidden(p, card)))

    @lru_cache(maxsize=None)
    def ways(i: int, cap: tuple[int, ...]) -> int:
        if i == n:
            return 1 if all(c == 0 for c in cap) else 0
        total = 0
        cap_list = list(cap)
        for p in legal[i]:
            if cap_list[p] <= 0:
                continue
            cap_list[p] -= 1
            total += ways(i + 1, tuple(cap_list))
            cap_list[p] += 1
        return total

    total = ways(0, cap0)
    counts = [[0] * n_p for _ in range(n)]
    if total == 0:
        ways.cache_clear()
        return counts, 0

    reach: dict[tuple[int, ...], int] = {cap0: 1}
    for i in range(n):
        nxt: dict[tuple[int, ...], int] = defaultdict(int)
        for cap, r in reach.items():
            cap_list = list(cap)
            for p in legal[i]:
                if cap_list[p] <= 0:
                    continue
                cap_list[p] -= 1
                newcap = tuple(cap_list)
                w = ways(i + 1, newcap)
                if w:
                    counts[i][p] += r * w
                    nxt[newcap] += r
                cap_list[p] += 1
        reach = nxt

    ways.cache_clear()
    return counts, total


def dp_marginals(obs: Observation, cons: Constraints | None = None):
    cons = cons or constraints_from_obs(obs)
    spec = obs.spec
    counts, total = count_locations(obs, cons)
    marg = [[0.0] * spec.n_players for _ in range(spec.n_cards)]
    for c, loc in enumerate(cons.known_loc):
        if loc >= 0:
            marg[c][loc] = 1.0
    if total == 0:
        return marg, 0
    for i, card in enumerate(cons.unknown_cards()):
        for p in range(spec.n_players):
            marg[card][p] = counts[i][p] / total
    return marg, total
