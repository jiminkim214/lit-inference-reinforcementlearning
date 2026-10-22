#!/usr/bin/env python3
"""Profile exact DP counting against the Markov sampler on a live observation."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litbot.agents.random_agent import RandomAgent
from litbot.belief.constraints import constraints_from_obs
from litbot.belief.counting import dp_marginals
from litbot.belief.enumerator import exact_marginals
from litbot.belief.sampler import sample_worlds
from litbot.cards import TOY_12, STANDARD
from litbot.game import apply_move, new_game


def l1(a, b) -> float:
    total = 0.0
    n = 0
    for i in range(len(a)):
        for j in range(len(a[i])):
            total += abs(a[i][j] - b[i][j])
            n += 1
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plies", type=int, default=12)
    p.add_argument("--toy", action="store_true", default=True)
    p.add_argument("--full", action="store_true")
    args = p.parse_args()
    spec = STANDARD if args.full else TOY_12
    state = new_game(spec, seed=0)
    agents = [RandomAgent(seed=i) for i in range(6)]
    for _ in range(args.plies):
        if state.is_terminal():
            break
        mv = agents[state.current_player].select(state.observe(state.current_player))
        state = apply_move(state, mv)
    obs = state.observe(0)
    cons = constraints_from_obs(obs)
    print("unknown cards", len(cons.unknown_cards()), "history", len(obs.history))

    t0 = time.perf_counter()
    dp, n_dp = dp_marginals(obs, cons)
    t_dp = time.perf_counter() - t0
    print(f"dp     n={n_dp}  {t_dp:.3f}s")

    if len(cons.unknown_cards()) <= 8:
        t0 = time.perf_counter()
        ex, n_ex = exact_marginals(obs, cons)
        t_ex = time.perf_counter() - t0
        print(f"enum   n={n_ex}  {t_ex:.3f}s  L1 vs dp {l1(dp, ex):.4f}")

    t0 = time.perf_counter()
    samples = sample_worlds(obs, n_samples=64, rng=random.Random(0), cons=cons)
    t_s = time.perf_counter() - t0
    print(f"sample k={len(samples)}  {t_s:.3f}s")


if __name__ == "__main__":
    main()
