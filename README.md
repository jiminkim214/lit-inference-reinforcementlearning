# LitBot

An agent for the six-player team card game **Literature**. Almost all of the skill in this game is inference: tracking where unseen cards can be, asking the questions most likely to succeed, and claiming half-suits at the right moment.

The stack is:

1. A precise rules engine with legal-move generation and a public history log
2. A belief engine (constraint tracking, exact enumeration on small decks, a Markov sampler, optional opponent-model weighting)
3. Agents: random, greedy, PIMC, ISMCTS, and ISMCTS with the opponent model
4. Duplicate tournaments and belief calibration
5. A Flask table for human evaluation
6. A stretch-goal linear value function trained from self-play outcomes

House rule (stricter than Pagat): **any incorrect claim gives the half-suit to the opponents.** First team to five half-suits wins. See [RULES.md](RULES.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest
```

Checkpoint covered by the tests: the engine plays full random games with no rule violations; the toy-deck enumerator is ground truth for the sampler; greedy beats random on the 12-card deck.

## Agents

| Name | Behaviour |
| --- | --- |
| `random` | Uniform legal ask; claims a half-suit held entirely in hand |
| `greedy` | Ask for the card with highest `P(target has it)` |
| `pimc` | Sample worlds, play a clairvoyant greedy policy in each, vote |
| `ismcts` | Single-observer information-set MCTS, claims as tree actions |
| `ismcts-om` / `greedy-om` / `pimc-om` | Same, with asking-behaviour likelihood weights |

```bash
# Toy-deck duplicate: greedy vs random
python scripts/run_tournament.py --quick --games 8

# Mixed team: one ISMCTS + two greedy teammates vs three greedy
python scripts/run_tournament.py --mixed --games 8 --quick

# Belief calibration (constraint vs opponent model)
python scripts/calibrate_belief.py --toy --games 4

# Stretch-goal value fit
python scripts/train_value.py --toy --games 8
```

Full-deck ISMCTS is aimed at a **one second per move** budget (`time_budget=1.0`).

## Human evaluation

```bash
python web/app.py
```

Open `http://127.0.0.1:5000`, pick teammate and opponent agents, and play as seat 0. After the game a short feedback form is appended to `web/data/feedback.jsonl`.

Terminal play:

```bash
python -m litbot.play --teammate greedy --opponent random
```

## Layout

```
litbot/          engine, belief, agents, eval, rl
tests/           unit + checkpoint tests
web/             Flask table
scripts/         tournaments, calibration, profiling
RULES.md         edge-case rules spec
```

The reinforcement-learning stretch goal is optional and sits in `litbot/rl`. Train it only after the belief and search stack is working.
