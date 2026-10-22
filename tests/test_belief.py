import random

from litbot.belief.constraints import constraints_from_obs, assignment_consistent
from litbot.belief.counting import dp_marginals
from litbot.belief.engine import BeliefEngine
from litbot.belief.enumerator import enumerate_consistent, exact_marginals
from litbot.belief.sampler import sample_worlds
from litbot.cards import TOY_6
from litbot.game import GameState, Observation, apply_move, new_game


def test_open_deal_enumerator_uniform():
    state = new_game(TOY_6, seed=0)
    obs = state.observe(0)
    worlds = enumerate_consistent(obs)
    # 5 remaining cards, one per other player: 5! = 120
    assert len(worlds) == 120
    marg, n = exact_marginals(obs)
    assert n == 120
    my = next(iter(_bits(obs.hand)))
    assert marg[my][0] == 1.0
    for card in range(TOY_6.n_cards):
        if card == my:
            continue
        for p in range(6):
            if p == 0:
                assert marg[card][p] == 0.0
            else:
                assert abs(marg[card][p] - 0.2) < 1e-9


def test_failed_ask_eliminates_target():
    spec = TOY_6
    hands = [0] * 6
    for i in range(6):
        hands[i] = 1 << i
    state = GameState(spec, hands, current_player=0)
    # P0 holds card 0 (low). Ask P1 for card 1; P1 holds card 1 so this succeeds.
    # Use a fail: ask P3 for card 1, P3 holds 3.
    state = apply_move(state, ("ask", 3, 1))
    assert state.history[-1].success is False
    obs = state.observe(0)
    worlds = enumerate_consistent(obs)
    for asn in worlds:
        assert asn.get(1, constraints_from_obs(obs).known_loc[1]) != 3
    # Card 1 cannot be with P3
    marg, n = exact_marginals(obs)
    assert n > 0
    assert marg[1][3] == 0.0


def test_sampler_matches_enumerator_on_toy():
    spec = TOY_6
    hands = [0] * 6
    for i in range(6):
        hands[i] = 1 << i
    state = GameState(spec, hands, current_player=0)
    state = apply_move(state, ("ask", 1, 1))  # success, P0 now has 0 and 1
    obs = state.observe(0)
    exact, n = exact_marginals(obs)
    assert n > 0
    rng = random.Random(0)
    samples = sample_worlds(obs, n_samples=n, rng=rng)
    assert samples
    # With unknown <= 10 the sampler enumerates exactly
    cons = constraints_from_obs(obs)
    engine = BeliefEngine(n_samples=64, seed=1, method="sample")
    belief = engine.infer(obs)
    for card in cons.unknown_cards():
        for p in range(6):
            assert abs(belief.marginals[card][p] - exact[card][p]) < 0.08


def test_dp_matches_capacity_count_on_open_deal():
    state = new_game(TOY_6, seed=2)
    obs = state.observe(0)
    dp, n_dp = dp_marginals(obs)
    exact, n_ex = exact_marginals(obs)
    assert n_dp == n_ex
    for card in range(TOY_6.n_cards):
        for p in range(6):
            assert abs(dp[card][p] - exact[card][p]) < 1e-9


def test_successful_ask_is_known_location():
    spec = TOY_6
    hands = [0] * 6
    for i in range(6):
        hands[i] = 1 << i
    state = GameState(spec, hands, current_player=0)
    state = apply_move(state, ("ask", 1, 1))
    obs = state.observe(0)
    cons = constraints_from_obs(obs)
    assert cons.known_loc[1] == 0
    marg, _ = exact_marginals(obs)
    assert marg[1][0] == 1.0


def test_assignment_rejects_inconsistent():
    spec = TOY_6
    hands = [0] * 6
    for i in range(6):
        hands[i] = 1 << i
    state = GameState(spec, hands, current_player=0)
    state = apply_move(state, ("ask", 3, 1))  # fail, P3 does not have 1
    obs = state.observe(0)
    cons = constraints_from_obs(obs)
    # Force card 1 onto P3
    bad = {c: ((p + 1) % 6 or 1) for p, c in enumerate(range(1, 6))}
    bad[1] = 3
    # Build a capacity-matching assignment that puts 1 on 3
    others = [c for c in range(1, 6) if c != 1]
    players = [p for p in range(1, 6) if p != 3]
    asn = {1: 3}
    for c, p in zip(others, players):
        asn[c] = p
    assert not assignment_consistent(obs, asn, cons)


def _bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit
