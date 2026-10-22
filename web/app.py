"""Simple Flask UI for human evaluation of LitBot teammates / opponents."""

from __future__ import annotations

import json
import secrets
import sys
import time
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litbot.agents.factory import make_agent
from litbot.agents.random_agent import _transfers
from litbot.cards import STANDARD, card_name
from litbot.game import apply_move, legal_asks_from, new_game
from litbot.serialize import obs_to_dict

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

GAMES: dict[str, dict] = {}
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _make_agents(teammate: str, opponent: str, seed: int):
    kwargs = {
        "greedy": {"n_samples": 32},
        "pimc": {"n_samples": 16},
        "ismcts": {"n_samples": 16, "iterations": 80, "time_budget": 1.0, "rollout_cutoff": 16},
        "ismcts-om": {
            "n_samples": 16,
            "iterations": 80,
            "time_budget": 1.0,
            "rollout_cutoff": 16,
        },
    }
    lineup = ["human", opponent, teammate, opponent, teammate, opponent]
    agents = [None]
    for i, name in enumerate(lineup[1:], start=1):
        kw = dict(kwargs.get(name, {}))
        agents.append(make_agent(name, seed=seed + i, **kw))
    return lineup, agents


def _advance_ai(rec: dict) -> None:
    state = rec["state"]
    agents = rec["agents"]
    while not state.is_terminal() and state.current_player != 0:
        agent = agents[state.current_player]
        obs = state.observe(state.current_player)
        move = agent.select(obs)
        rec["moves"].append({"player": state.current_player, "move": str(move)})
        state = apply_move(state, move)
        rec["state"] = state


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/new", methods=["POST"])
def new():
    teammate = request.form.get("teammate", "greedy")
    opponent = request.form.get("opponent", "greedy")
    seed = int(request.form.get("seed") or int(time.time()) % 10_000_000)
    gid = secrets.token_hex(8)
    state = new_game(STANDARD, seed=seed)
    lineup, agents = _make_agents(teammate, opponent, seed)
    rec = {
        "id": gid,
        "state": state,
        "agents": agents,
        "lineup": lineup,
        "seed": seed,
        "human": 0,
        "moves": [],
        "created": time.time(),
        "error": None,
        "feedback": None,
    }
    GAMES[gid] = rec
    _advance_ai(rec)
    return redirect(url_for("game", gid=gid))


@app.route("/game/<gid>")
def game(gid: str):
    rec = GAMES.get(gid)
    if rec is None:
        return redirect(url_for("index"))
    state = rec["state"]
    obs = state.observe(0)
    payload = obs_to_dict(obs)
    payload["lineup"] = rec["lineup"]
    payload["seed"] = rec["seed"]
    payload["error"] = rec.get("error")
    payload["winner"] = state.winner() if state.is_terminal() else None
    asks = []
    if not obs.is_terminal() and obs.current_player == 0:
        asks = legal_asks_from(obs.spec, 0, obs.hand, obs.hand_sizes, obs.claimed)
    payload["legal_asks"] = [
        {"target": m[1], "card": m[2], "card_name": card_name(m[2], obs.spec)}
        for m in asks
    ]
    payload["legal_transfers"] = _transfers(obs) if obs.current_player == 0 else []
    return render_template("game.html", view=payload, gid=gid)


@app.route("/game/<gid>/act", methods=["POST"])
def act(gid: str):
    rec = GAMES.get(gid)
    if rec is None:
        return redirect(url_for("index"))
    state = rec["state"]
    rec["error"] = None
    if state.is_terminal() or state.current_player != 0:
        return redirect(url_for("game", gid=gid))
    kind = request.form.get("kind")
    try:
        if kind == "ask":
            target_s, card_s = request.form["ask"].split(",")
            move = ("ask", int(target_s), int(card_s))
        elif kind == "claim":
            hs = int(request.form["halfsuit"])
            named = []
            spec = state.spec
            for i, _c in enumerate(spec.cards_of_halfsuit(hs)):
                named.append(int(request.form[f"owner_{i}"]))
            move = ("claim", hs, tuple(named))
        elif kind == "pass":
            dest = int(request.form["dest"])
            empty_team = all(
                state.hand_sizes()[p] == 0
                for p in state.spec.teammates(0)
            ) and state.hands[0] == 0
            move = ("designate", dest) if empty_team else ("pass", dest)
        else:
            raise ValueError("unknown action")
        rec["moves"].append({"player": 0, "move": str(move)})
        rec["state"] = apply_move(state, move)
        _advance_ai(rec)
    except Exception as exc:
        rec["error"] = str(exc)
    return redirect(url_for("game", gid=gid))


@app.route("/game/<gid>/feedback", methods=["POST"])
def feedback(gid: str):
    rec = GAMES.get(gid)
    if rec is None:
        return redirect(url_for("index"))
    note = {
        "game_id": gid,
        "seed": rec["seed"],
        "lineup": rec["lineup"],
        "scores": rec["state"].scores,
        "winner": rec["state"].winner(),
        "plies": rec["state"].plies,
        "reasonable": request.form.get("reasonable"),
        "comments": request.form.get("comments", "").strip(),
        "t": time.time(),
    }
    rec["feedback"] = note
    path = DATA_DIR / "feedback.jsonl"
    with path.open("a") as fh:
        fh.write(json.dumps(note) + "\n")
    return redirect(url_for("game", gid=gid))


def main():
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
