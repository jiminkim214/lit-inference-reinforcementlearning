#!/usr/bin/env python3
"""Play a seeded game of random agents and print the public log."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litbot.agents.random_agent import RandomAgent
from litbot.cards import STANDARD, TOY_12
from litbot.game import describe_move, play_game


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--toy", action="store_true")
    args = p.parse_args()
    spec = TOY_12 if args.toy else STANDARD
    agents = [RandomAgent(seed=args.seed + i) for i in range(spec.n_players)]

    def on_move(state, move):
        actor = state.history[-1].describe(spec) if state.history else describe_move(move, spec)
        print(f"{state.plies:4d}  {actor}")
        print(f"      scores={state.scores} hands={state.hand_sizes()}")

    state = play_game(agents, spec=spec, seed=args.seed, on_move=on_move)
    print("terminal", state.is_terminal(), "winner", state.winner(), "scores", state.scores)


if __name__ == "__main__":
    main()
