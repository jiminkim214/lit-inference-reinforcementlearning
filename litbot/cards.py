"""Cards, half-suits, and the configurable rules spec.

A standard Literature deck is a 52-card pack minus the four 8s (48 cards).
Each suit splits into a low half-suit (2–7) and a high half-suit (9–A), giving
eight half-suits of six cards. Card ids are 0 .. n_cards-1, laid out as
``halfsuit * ranks_per_half + rank_index``.
"""

from __future__ import annotations

from dataclasses import dataclass

SUIT_NAMES = ("C", "D", "H", "S")
SUIT_SYMBOLS = ("♣", "♦", "♥", "♠")
LOW_RANKS = (2, 3, 4, 5, 6, 7)
HIGH_RANKS = (9, 10, 11, 12, 13, 14)
RANK_LABEL = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    9: "9",
    10: "10",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}


@dataclass(frozen=True, slots=True)
class Spec:
    """Rules parameters. The standard game is 6 players, 4 suits, 6 ranks/half."""

    n_players: int = 6
    n_suits: int = 4
    ranks_per_half: int = 6
    win_threshold: int = 5
    n_teams: int = 2

    def __post_init__(self) -> None:
        if self.n_players < 2 or self.n_players % self.n_teams != 0:
            raise ValueError("n_players must be a positive multiple of n_teams")
        if self.n_suits < 1:
            raise ValueError("n_suits must be >= 1")
        if self.ranks_per_half < 1:
            raise ValueError("ranks_per_half must be >= 1")

    @property
    def n_halfsuits(self) -> int:
        return self.n_suits * 2

    @property
    def n_cards(self) -> int:
        return self.n_halfsuits * self.ranks_per_half

    @property
    def team_size(self) -> int:
        return self.n_players // self.n_teams

    def team(self, player: int) -> int:
        return player % self.n_teams

    def teammates(self, player: int) -> tuple[int, ...]:
        t = self.team(player)
        return tuple(p for p in range(self.n_players) if self.team(p) == t and p != player)

    def team_players(self, team: int) -> tuple[int, ...]:
        return tuple(p for p in range(self.n_players) if self.team(p) == team)

    def opponents(self, player: int) -> tuple[int, ...]:
        t = self.team(player)
        return tuple(p for p in range(self.n_players) if self.team(p) != t)

    def deal_sizes(self) -> tuple[int, ...]:
        q, r = divmod(self.n_cards, self.n_players)
        return tuple(q + (1 if i < r else 0) for i in range(self.n_players))

    def halfsuit_of(self, card: int) -> int:
        return card // self.ranks_per_half

    def rank_index(self, card: int) -> int:
        return card % self.ranks_per_half

    def suit_of(self, card: int) -> int:
        return self.halfsuit_of(card) // 2

    def is_high(self, card: int) -> bool:
        return self.halfsuit_of(card) % 2 == 1

    def rank_value(self, card: int) -> int:
        ranks = HIGH_RANKS if self.is_high(card) else LOW_RANKS
        return ranks[self.rank_index(card)]

    def halfsuit_mask(self, hs: int) -> int:
        start = hs * self.ranks_per_half
        mask = 0
        for i in range(self.ranks_per_half):
            mask |= 1 << (start + i)
        return mask

    def all_cards_mask(self) -> int:
        return (1 << self.n_cards) - 1

    def cards_of_halfsuit(self, hs: int) -> tuple[int, ...]:
        start = hs * self.ranks_per_half
        return tuple(start + i for i in range(self.ranks_per_half))


STANDARD = Spec()

# Smaller decks used as ground truth for the belief enumerator.
TOY_6 = Spec(n_players=6, n_suits=1, ranks_per_half=3, win_threshold=2)  # 6 cards, 1 each
TOY_12 = Spec(n_players=6, n_suits=2, ranks_per_half=3, win_threshold=2)  # 12 cards, 2 each


def iter_bits(mask: int):
    """Yield bit indices set in mask."""
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def popcount(mask: int) -> int:
    return mask.bit_count()


def card_name(card: int, spec: Spec = STANDARD) -> str:
    suit = spec.suit_of(card)
    rank = spec.rank_value(card)
    return f"{RANK_LABEL[rank]}{SUIT_SYMBOLS[suit]}"


def card_ascii(card: int, spec: Spec = STANDARD) -> str:
    suit = spec.suit_of(card)
    rank = spec.rank_value(card)
    return f"{RANK_LABEL[rank]}{SUIT_NAMES[suit]}"


def halfsuit_name(hs: int, spec: Spec = STANDARD) -> str:
    suit = hs // 2
    high = hs % 2 == 1
    kind = "high" if high else "low"
    return f"{kind} {SUIT_SYMBOLS[suit]}"


def parse_card(text: str, spec: Spec = STANDARD) -> int:
    """Parse 'QS', 'Q♠', '10H', 'AH' etc. into a card id."""
    text = text.strip().replace(" ", "").upper()
    replacements = {"♣": "C", "♦": "D", "♥": "H", "♠": "S", "T": "10"}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    if len(text) < 2:
        raise ValueError(f"bad card: {text}")
    suit_letter = text[-1]
    rank_text = text[:-1]
    suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
    if suit_letter not in suit_map:
        raise ValueError(f"bad suit: {text}")
    suit = suit_map[suit_letter]
    rank_map = {label: val for val, label in RANK_LABEL.items()}
    rank_map["T"] = 10
    if rank_text not in rank_map:
        raise ValueError(f"bad rank: {text}")
    rank = rank_map[rank_text]
    if rank in LOW_RANKS[: spec.ranks_per_half]:
        hs = suit * 2
        idx = LOW_RANKS.index(rank)
    elif rank in HIGH_RANKS[: spec.ranks_per_half]:
        hs = suit * 2 + 1
        idx = HIGH_RANKS.index(rank)
    else:
        raise ValueError(f"rank not in this spec: {text}")
    return hs * spec.ranks_per_half + idx
