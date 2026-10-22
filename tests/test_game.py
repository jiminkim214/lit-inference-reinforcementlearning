import pytest

from litbot.agents.random_agent import RandomAgent
from litbot.cards import STANDARD, TOY_6, TOY_12, popcount
from litbot.game import (
    GameState,
    IllegalMoveError,
    apply_move,
    legal_asks,
    new_game,
    play_game,
    perfect_info_claims,
    perfect_info_policy,
)


def test_deal_conserves_cards():
    state = new_game(STANDARD, seed=7)
    occupied = 0
    for h in state.hands:
        assert popcount(h) == 8
        assert occupied & h == 0
        occupied |= h
    assert occupied == (1 << 48) - 1
    assert state.invariant_errors() == []


def test_ask_success_keeps_turn_and_transfers():
    spec = TOY_6
    # Give P0 low card 0, P1 the other low card 1.
    hands = [0] * 6
    hands[0] = 1 << 0
    hands[1] = 1 << 1
    for i, c in enumerate(range(2, 6)):
        hands[i + 2] = 1 << c
    state = GameState(spec, hands, current_player=0)
    nxt = apply_move(state, ("ask", 1, 1))
    assert nxt.hands[0] & (1 << 1)
    assert nxt.hands[1] == 0
    assert nxt.current_player == 0
    assert nxt.history[-1].success is True


def test_ask_fail_passes_turn():
    spec = TOY_6
    hands = [0] * 6
    hands[0] = 1 << 0
    hands[1] = 1 << 3  # high half, not card 1
    for i, c in enumerate([1, 2, 4, 5]):
        hands[i + 2] = 1 << c
    state = GameState(spec, hands, current_player=0)
    nxt = apply_move(state, ("ask", 1, 1))
    assert nxt.current_player == 1
    assert nxt.history[-1].success is False
    assert nxt.hands[0] == 1 << 0


def test_cannot_ask_teammate_or_own_card():
    state = new_game(TOY_6, seed=1)
    p = state.current_player
    my = next(iter(_bits(state.hands[p])))
    with pytest.raises(IllegalMoveError):
        apply_move(state, ("ask", 2, my))  # teammate 2
    # Asking for a card we hold, targeting an opponent
    opp = 1
    with pytest.raises(IllegalMoveError):
        apply_move(state, ("ask", opp, my))


def test_correct_claim_awards_claimer():
    spec = TOY_6
    hands = [0] * 6
    # Low half-suit cards 0,1,2 all on team 0
    hands[0] = 1 << 0
    hands[2] = 1 << 1
    hands[4] = 1 << 2
    hands[1] = 1 << 3
    hands[3] = 1 << 4
    hands[5] = 1 << 5
    state = GameState(spec, hands, current_player=0)
    named = (0, 2, 4)
    nxt = apply_move(state, ("claim", 0, named))
    assert nxt.claimed[0] == 0
    assert nxt.scores == [1, 0]
    assert nxt.hands[0] == 0
    assert nxt.current_player == 0  # turn continues


def test_wrong_teammate_claim_awards_opponents():
    spec = TOY_6
    hands = [0] * 6
    hands[0] = 1 << 0
    hands[2] = 1 << 1
    hands[4] = 1 << 2
    hands[1] = 1 << 3
    hands[3] = 1 << 4
    hands[5] = 1 << 5
    state = GameState(spec, hands, current_player=0)
    # Swap P2 and P4
    named = (0, 4, 2)
    nxt = apply_move(state, ("claim", 0, named))
    assert nxt.claimed[0] == 1
    assert nxt.scores == [0, 1]


def test_opponent_holds_card_awards_opponents():
    spec = TOY_6
    hands = [0] * 6
    hands[0] = 1 << 0
    hands[1] = 1 << 1  # opponent has a low card
    hands[2] = 1 << 2
    hands[3] = 1 << 3
    hands[4] = 1 << 4
    hands[5] = 1 << 5
    state = GameState(spec, hands, current_player=0)
    named = (0, 2, 4)
    nxt = apply_move(state, ("claim", 0, named))
    assert nxt.claimed[0] == 1
    assert nxt.scores == [0, 1]


def test_empty_hand_pass_to_teammate():
    spec = TOY_6
    hands = [0] * 6
    hands[2] = 1 << 0
    hands[4] = 1 << 1
    hands[1] = 1 << 2
    hands[3] = 1 << 3
    hands[5] = (1 << 4) | (1 << 5)
    state = GameState(spec, hands, current_player=0)  # P0 empty
    nxt = apply_move(state, ("pass", 2))
    assert nxt.current_player == 2


def test_first_to_win_threshold_ends_game():
    spec = TOY_6  # win_threshold 2
    hands = [0] * 6
    hands[0] = spec.halfsuit_mask(0)  # all 3 low cards
    hands[1] = spec.halfsuit_mask(1)
    state = GameState(spec, hands, current_player=0, scores=[1, 0])
    named = (0, 0, 0)
    nxt = apply_move(state, ("claim", 0, named))
    assert nxt.is_terminal()
    assert nxt.winner() == 0


def test_legal_asks_require_held_halfsuit():
    spec = TOY_6
    hands = [0] * 6
    hands[0] = 1 << 0  # low
    for i, c in enumerate(range(1, 6)):
        hands[i + 1] = 1 << c
    state = GameState(spec, hands, current_player=0)
    asks = legal_asks(state)
    assert asks
    for mv in asks:
        assert mv[0] == "ask"
        card = mv[2]
        assert spec.halfsuit_of(card) == 0
        assert card != 0


def test_perfect_info_claims_finds_team_set():
    spec = TOY_6
    hands = [0] * 6
    hands[0] = 1 << 0
    hands[2] = 1 << 1
    hands[4] = 1 << 2
    hands[1] = 1 << 3
    hands[3] = 1 << 4
    hands[5] = 1 << 5
    state = GameState(spec, hands, current_player=0)
    claims = perfect_info_claims(state)
    assert ("claim", 0, (0, 2, 4)) in claims


def test_random_games_no_violations():
    spec = TOY_12
    for seed in range(12):
        agents = [RandomAgent(seed=seed * 6 + i) for i in range(6)]
        state = play_game(agents, spec=spec, seed=seed, max_plies=2000)
        assert state.invariant_errors() == []
        assert state.plies <= 2000
        assert sum(state.scores) <= spec.n_halfsuits
        for event in state.history:
            if getattr(event, "success", None) is not None:
                assert event.asker != event.target


def test_perfect_info_policy_finishes_toy_game():
    spec = TOY_12
    rng = __import__("random").Random(0)
    state = new_game(spec, seed=3)

    class PI:
        def select(self, obs):
            # Cheat with the outer `state` captured by play loop — use a wrapper.
            raise AssertionError("use manual loop")

    s = state
    for _ in range(500):
        if s.is_terminal():
            break
        mv = perfect_info_policy(s, rng)
        s = apply_move(s, mv)
        assert s.invariant_errors() == []
    assert s.is_terminal()


def _bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit
