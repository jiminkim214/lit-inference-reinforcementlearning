#!/usr/bin/env python3
"""Stretch goal: fit a linear value function from greedy self-play."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litbot.agents.greedy import GreedyAgent
from litbot.cards import TOY_12, STANDARD
from litbot.game import apply_move, new_game
from litbot.rl.value import train_from_games


def collect(spec, games: int, seed0: int):
    data = []
    for g in range(games):
        agents = [GreedyAgent(n_samples=16, seed=seed0 + g * 6 + i) for i in range(6)]
        state = new_game(spec, seed=seed0 + g)
        traj = {p: [] for p in range(6)}
        while not state.is_terminal() and state.plies < 800:
            p = state.current_player
            obs = state.observe(p)
            traj[p].append(obs)
            state = apply_move(state, agents[p].select(obs))
        for p, obs_list in traj.items():
            team = spec.team(p)
            y = float(state.scores[team] - state.scores[1 - team])
            data.append((obs_list, y))
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=12)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "value.npz")
    args = p.parse_args()
    spec = TOY_12 if args.toy else STANDARD
    data = collect(spec, args.games, 0)
    model = train_from_games(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out} weights={model.w.size} bias={model.b:.3f}")


if __name__ == "__main__":
    main()
