"""Belief calibration: predicted locations vs the true deal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from litbot.agents.factory import make_agent
from litbot.belief.engine import BeliefEngine
from litbot.cards import Spec, STANDARD
from litbot.game import GameState, Observation, apply_move, new_game


@dataclass
class CalibrationPoint:
    p: float
    hit: int
    player: int
    card: int
    ply: int


def _true_locations(state: GameState) -> list[int]:
    loc = [-1] * state.spec.n_cards
    for p, h in enumerate(state.hands):
        m = h
        while m:
            bit = m & -m
            loc[bit.bit_length() - 1] = p
            m ^= bit
    return loc


def collect_points(
    spec: Spec = STANDARD,
    games: int = 8,
    seed0: int = 0,
    agent_name: str = "greedy",
    engine: Optional[BeliefEngine] = None,
    every: int = 4,
    observer: int = 0,
) -> list[CalibrationPoint]:
    engine = engine or BeliefEngine(n_samples=48, seed=1)
    points: list[CalibrationPoint] = []
    for g in range(games):
        agents = [make_agent(agent_name, seed=seed0 + g * 6 + i) for i in range(spec.n_players)]
        state = new_game(spec, seed=seed0 + g)
        while not state.is_terminal() and state.plies < 800:
            if state.plies % every == 0:
                obs = state.observe(observer)
                belief = engine.infer(obs)
                truth = _true_locations(state)
                for card in range(spec.n_cards):
                    if obs.hand & (1 << card):
                        continue
                    if belief.cons.known_loc[card] == -2:
                        continue
                    holder = truth[card]
                    if holder < 0:
                        continue
                    for p in range(spec.n_players):
                        if p == observer:
                            continue
                        points.append(
                            CalibrationPoint(
                                p=belief.p_has(p, card),
                                hit=int(holder == p),
                                player=p,
                                card=card,
                                ply=state.plies,
                            )
                        )
            obs_act = state.observe(state.current_player)
            move = agents[state.current_player].select(obs_act)
            state = apply_move(state, move)
    return points


def binned_reliability(points: Sequence[CalibrationPoint], bins: int = 10) -> list[dict]:
    buckets = [[] for _ in range(bins)]
    for pt in points:
        i = min(int(pt.p * bins), bins - 1)
        buckets[i].append(pt)
    rows = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_p = sum(pt.p for pt in bucket) / len(bucket)
        mean_hit = sum(pt.hit for pt in bucket) / len(bucket)
        rows.append(
            {
                "bin": i,
                "lo": i / bins,
                "hi": (i + 1) / bins,
                "n": len(bucket),
                "predicted": mean_p,
                "observed": mean_hit,
            }
        )
    return rows


def calibrate(**kwargs) -> dict:
    points = collect_points(**kwargs)
    rows = binned_reliability(points)
    if not points:
        return {"n": 0, "brier": None, "bins": []}
    brier = sum((pt.p - pt.hit) ** 2 for pt in points) / len(points)
    return {"n": len(points), "brier": brier, "bins": rows}


def plot_reliability(rows: Sequence[dict], path, title: str = "Belief calibration") -> None:
    import matplotlib.pyplot as plt

    xs = [r["predicted"] for r in rows]
    ys = [r["observed"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    if xs:
        ax.plot(xs, ys, "o-", color="#0f3d2e")
    ax.set_xlabel("predicted P(player has card)")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
