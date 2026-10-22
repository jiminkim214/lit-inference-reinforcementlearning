#!/usr/bin/env python3
"""Run duplicate tournaments between LitBot agents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litbot.cards import STANDARD, TOY_12
from litbot.eval.tournament import run_tournament, summarize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=10, help="deals per matchup (x2 if duplicate)")
    p.add_argument("--quick", action="store_true", help="toy deck, random vs greedy only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "tournament.jsonl")
    p.add_argument(
        "--matchup",
        action="append",
        default=[],
        help="repeatable pair like greedy,random or ismcts,greedy",
    )
    p.add_argument(
        "--mixed",
        action="store_true",
        help="one ISMCTS with two greedy teammates vs three greedy opponents",
    )
    args = p.parse_args()
    spec = TOY_12 if args.quick else STANDARD
    if args.mixed:
        from litbot.eval.tournament import play_match, summarize as _sum

        kwargs = {
            "greedy": {"n_samples": 32},
            "ismcts": {"n_samples": 16, "iterations": 60, "time_budget": 1.0, "rollout_cutoff": 12},
        }
        lineup = ["ismcts", "greedy", "greedy", "greedy", "greedy", "greedy"]
        results = []
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for g in range(args.games):
            r = play_match(lineup, spec, seed=args.seed + g, agent_kwargs=kwargs)
            results.append(r)
            print(
                f"seed={r.seed} mixed {r.names} -> {r.scores} winner={r.winner} "
                f"plies={r.plies} {r.seconds:.1f}s"
            )
        summary = _sum(results)
        print(json.dumps(summary, indent=2))
        return
    if args.quick:
        matchups = [("greedy", "random")]
        kwargs = {"greedy": {"n_samples": 24}}
    elif args.matchup:
        matchups = [tuple(m.split(",")) for m in args.matchup]
        kwargs = {
            "greedy": {"n_samples": 32},
            "pimc": {"n_samples": 16},
            "ismcts": {"n_samples": 16, "iterations": 60, "time_budget": 1.0, "rollout_cutoff": 12},
            "ismcts-om": {
                "n_samples": 16,
                "iterations": 60,
                "time_budget": 1.0,
                "rollout_cutoff": 12,
            },
        }
    else:
        matchups = [
            ("greedy", "random"),
            ("pimc", "greedy"),
            ("ismcts", "greedy"),
            ("ismcts-om", "ismcts"),
        ]
        kwargs = {
            "greedy": {"n_samples": 32},
            "pimc": {"n_samples": 16},
            "ismcts": {"n_samples": 16, "iterations": 60, "time_budget": 1.0, "rollout_cutoff": 12},
            "ismcts-om": {
                "n_samples": 16,
                "iterations": 60,
                "time_budget": 1.0,
                "rollout_cutoff": 12,
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def on_result(r):
        print(
            f"seed={r.seed} swapped={r.swapped} {r.names[0]}/{r.names[1]} "
            f"-> {r.scores} winner={r.winner} plies={r.plies} {r.seconds:.1f}s"
        )

    results = run_tournament(
        matchups,
        games=args.games,
        spec=spec,
        seed0=args.seed,
        agent_kwargs=kwargs,
        log_path=args.out,
        on_result=on_result,
    )
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
