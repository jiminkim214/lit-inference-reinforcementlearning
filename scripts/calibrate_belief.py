#!/usr/bin/env python3
"""Compare belief-engine calibration with and without the opponent model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litbot.belief.engine import BeliefEngine
from litbot.cards import STANDARD, TOY_12
from litbot.eval.calibration import calibrate, plot_reliability


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "calibration.json")
    args = p.parse_args()
    spec = TOY_12 if args.toy else STANDARD
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, engine in [
        ("constraint", BeliefEngine(n_samples=48, use_opponent_model=False, seed=1)),
        ("opponent-model", BeliefEngine(n_samples=48, use_opponent_model=True, seed=1)),
    ]:
        print(f"calibrating {name}...")
        rows[name] = calibrate(
            spec=spec, games=args.games, seed0=0, agent_name="greedy", engine=engine, every=5
        )
        print(f"  n={rows[name]['n']} brier={rows[name]['brier']}")
        png = args.out.with_name(f"calibration-{name}.png")
        plot_reliability(rows[name]["bins"], png, title=name)
    args.out.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
