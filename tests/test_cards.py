from litbot.cards import STANDARD, TOY_6, card_name, parse_card, popcount, iter_bits


def test_standard_counts():
    s = STANDARD
    assert s.n_cards == 48
    assert s.n_halfsuits == 8
    assert s.deal_sizes() == (8, 8, 8, 8, 8, 8)
    assert s.team(0) == 0 and s.team(1) == 1 and s.team(4) == 0
    assert s.opponents(0) == (1, 3, 5)
    assert s.teammates(0) == (2, 4)


def test_card_roundtrip():
    for c in range(STANDARD.n_cards):
        assert parse_card(card_name(c)) == c


def test_toy_deck():
    assert TOY_6.n_cards == 6
    assert TOY_6.deal_sizes() == (1, 1, 1, 1, 1, 1)


def test_iter_bits():
    mask = (1 << 3) | (1 << 7) | (1 << 0)
    assert list(iter_bits(mask)) == [0, 3, 7]
    assert popcount(mask) == 3
