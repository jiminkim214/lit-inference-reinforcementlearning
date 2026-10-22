from litbot.agents.factory import make_agent
from litbot.agents.greedy import GreedyAgent
from litbot.agents.random_agent import RandomAgent
from litbot.cards import TOY_12
from litbot.game import play_game, new_game, apply_move, perfect_info_policy
import random


def test_greedy_beats_random_on_toy():
    spec = TOY_12
    greedy_team_wins = 0
    random_team_wins = 0
    games = 16
    for seed in range(games):
        names = ["greedy", "random", "greedy", "random", "greedy", "random"]
        agents = [
            GreedyAgent(n_samples=24, seed=seed * 6 + i)
            if n == "greedy"
            else RandomAgent(seed=seed * 6 + i)
            for i, n in enumerate(names)
        ]
        state = play_game(agents, spec=spec, seed=seed, max_plies=1500)
        if state.winner() == 0:
            greedy_team_wins += 1
        elif state.winner() == 1:
            random_team_wins += 1
    assert greedy_team_wins + random_team_wins > 0
    assert greedy_team_wins > random_team_wins


def test_agents_return_legal_asks_or_claims():
    spec = TOY_12
    state = new_game(spec, seed=4)
    for kind in ("random", "greedy"):
        agent = make_agent(kind, seed=0, **({"n_samples": 16} if kind == "greedy" else {}))
        obs = state.observe(state.current_player)
        move = agent.select(obs)
        nxt = apply_move(state, move)
        assert nxt.invariant_errors() == []


def test_pimc_plays_a_move():
    spec = TOY_12
    state = new_game(spec, seed=5)
    agent = make_agent("pimc", seed=1, n_samples=8)
    obs = state.observe(state.current_player)
    move = agent.select(obs)
    nxt = apply_move(state, move)
    assert nxt.plies == 1


def test_ismcts_plays_a_move():
    spec = TOY_12
    state = new_game(spec, seed=6)
    agent = make_agent("ismcts", seed=1, n_samples=8, iterations=20, rollout_cutoff=8)
    obs = state.observe(state.current_player)
    move = agent.select(obs)
    nxt = apply_move(state, move)
    assert nxt.invariant_errors() == []


def test_perfect_info_never_asks_missing_card_when_hit_exists():
    spec = TOY_12
    rng = random.Random(0)
    state = new_game(spec, seed=9)
    for _ in range(30):
        if state.is_terminal():
            break
        mv = perfect_info_policy(state, rng)
        if mv[0] == "ask":
            target, card = mv[1], mv[2]
            # If any hitting ask existed, the chosen one should hit.
            from litbot.game import legal_asks

            asks = legal_asks(state)
            hits = [m for m in asks if state.hands[m[1]] & (1 << m[2])]
            if hits:
                assert state.hands[target] & (1 << card)
        state = apply_move(state, mv)
