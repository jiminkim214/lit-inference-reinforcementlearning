"""Terminal play: `python -m litbot.play`."""

from __future__ import annotations

import argparse
import sys

from litbot.agents.factory import make_agent
from litbot.cards import STANDARD, card_name, halfsuit_name, iter_bits
from litbot.game import apply_move, describe_move, legal_asks_from, new_game
from litbot.agents.random_agent import _transfers


def _print_obs(obs) -> None:
    spec = obs.spec
    print(f"\n--- ply {obs.plies}  turn P{obs.current_player}  score {obs.scores} ---")
    print("hand:", " ".join(card_name(c, spec) for c in iter_bits(obs.hand)) or "(empty)")
    print("sizes:", obs.hand_sizes)
    claimed = [
        f"{halfsuit_name(hs, spec)}:T{owner}"
        for hs, owner in enumerate(obs.claimed)
        if owner >= 0
    ]
    print("claimed:", ", ".join(claimed) or "(none)")
    if obs.history:
        print("last:", obs.history[-1].describe(spec))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Play Literature in the terminal as P0.")
    p.add_argument("--teammate", default="greedy")
    p.add_argument("--opponent", default="random")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    spec = STANDARD
    state = new_game(spec, seed=args.seed)
    lineup = ["human", args.opponent, args.teammate, args.opponent, args.teammate, args.opponent]
    agents = [None]
    for i, name in enumerate(lineup[1:], start=1):
        agents.append(make_agent(name, seed=args.seed + i))

    while not state.is_terminal():
        if state.current_player != 0:
            obs = state.observe(state.current_player)
            mv = agents[state.current_player].select(obs)
            print(f"P{state.current_player} ({lineup[state.current_player]}): {describe_move(mv, spec)}")
            state = apply_move(state, mv)
            continue
        obs = state.observe(0)
        _print_obs(obs)
        transfers = _transfers(obs)
        asks = legal_asks_from(obs.spec, 0, obs.hand, obs.hand_sizes, obs.claimed)
        if transfers:
            print("pass/designate:", ", ".join(str(m) for m in transfers))
        for i, mv in enumerate(asks[:40]):
            print(f"  {i:3d}  {describe_move(mv, spec)}")
        if len(asks) > 40:
            print(f"  ... {len(asks)} legal asks")
        raw = input("index, or 'claim HS p p p p p p', or 'pass N': ").strip()
        if raw.startswith("claim"):
            parts = raw.split()
            hs = int(parts[1])
            named = tuple(int(x) for x in parts[2:])
            mv = ("claim", hs, named)
        elif raw.startswith("pass"):
            dest = int(raw.split()[1])
            mv = transfers[0][0], dest
        else:
            mv = asks[int(raw)]
        state = apply_move(state, mv)
    print("final", state.scores, "winner", state.winner())


if __name__ == "__main__":
    main(sys.argv[1:])
