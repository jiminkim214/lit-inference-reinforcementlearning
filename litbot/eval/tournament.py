"""Duplicate tournament harness for Literature agents."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

from litbot.agents.factory import make_agent
from litbot.cards import Spec, STANDARD
from litbot.game import play_game


@dataclass
class GameResult:
    seed: int
    names: list[str]
    scores: list[int]
    winner: Optional[int]
    plies: int
    seconds: float
    swapped: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def play_match(
    names: Sequence[str],
    spec: Spec = STANDARD,
    seed: int = 0,
    swapped: bool = False,
    agent_kwargs: Optional[dict] = None,
) -> GameResult:
    agent_kwargs = agent_kwargs or {}
    agents = []
    for i, name in enumerate(names):
        kw = dict(agent_kwargs.get(name, {}))
        agents.append(make_agent(name, seed=seed * 17 + i, **kw))
    t0 = time.perf_counter()
    state = play_game(agents, spec=spec, seed=seed)
    dt = time.perf_counter() - t0
    return GameResult(
        seed=seed,
        names=list(names),
        scores=list(state.scores),
        winner=state.winner(),
        plies=state.plies,
        seconds=dt,
        swapped=swapped,
    )


def duplicate_pair(team_a: str, team_b: str) -> tuple[list[str], list[str]]:
    """Seats 0,2,4 vs 1,3,5. Return (lineup, swapped_lineup)."""
    a = [team_a, team_b, team_a, team_b, team_a, team_b]
    b = [team_b, team_a, team_b, team_a, team_b, team_a]
    return a, b


def run_tournament(
    matchups: Sequence[tuple[str, str]],
    games: int = 20,
    spec: Spec = STANDARD,
    seed0: int = 0,
    duplicate: bool = True,
    agent_kwargs: Optional[dict] = None,
    log_path: Optional[Path] = None,
    on_result=None,
) -> list[GameResult]:
    results: list[GameResult] = []
    handle = open(log_path, "a") if log_path else None
    try:
        for a, b in matchups:
            lineup, swapped = duplicate_pair(a, b)
            for g in range(games):
                seed = seed0 + g
                r1 = play_match(lineup, spec, seed, swapped=False, agent_kwargs=agent_kwargs)
                results.append(r1)
                if on_result:
                    on_result(r1)
                if handle:
                    handle.write(json.dumps(r1.to_dict()) + "\n")
                    handle.flush()
                if duplicate:
                    r2 = play_match(
                        swapped, spec, seed, swapped=True, agent_kwargs=agent_kwargs
                    )
                    results.append(r2)
                    if on_result:
                        on_result(r2)
                    if handle:
                        handle.write(json.dumps(r2.to_dict()) + "\n")
                        handle.flush()
    finally:
        if handle:
            handle.close()
    return results


def summarize(results: Sequence[GameResult]) -> dict:
    """Win rate and mean sets for each team-style matchup, plus per-agent stats."""
    from collections import defaultdict

    stats = defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0, "sets": 0.0, "plies": 0.0, "seconds": 0.0})
    for r in results:
        # Team 0 is even seats
        name0 = r.names[0]
        name1 = r.names[1]
        key = f"{name0} vs {name1}"
        s = stats[key]
        s["games"] += 1
        s["sets"] += r.scores[0]
        s["plies"] += r.plies
        s["seconds"] += r.seconds
        if r.winner == 0:
            s["wins"] += 1
        elif r.winner is None:
            s["draws"] += 1
    out = {}
    for k, s in stats.items():
        n = max(s["games"], 1)
        out[k] = {
            "games": s["games"],
            "win_rate": s["wins"] / n,
            "draw_rate": s["draws"] / n,
            "avg_sets": s["sets"] / n,
            "avg_plies": s["plies"] / n,
            "avg_seconds": s["seconds"] / n,
        }
    return out
