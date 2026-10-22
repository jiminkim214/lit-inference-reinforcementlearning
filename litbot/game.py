"""Literature game engine: state, legal moves, apply, play loop.

Hands are bitmasks. A `GameState` is the simulator's perfect-information
state. Agents only see `Observation`, built by `GameState.observe`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Union

from litbot.cards import Spec, STANDARD, card_name, halfsuit_name, iter_bits, popcount

# Moves are hashable tuples so they can key ISMCTS dictionaries.
#   ("ask", target, card)
#   ("claim", halfsuit, named_owners_tuple)
#   ("pass", teammate)
#   ("designate", opponent)
Move = tuple


class IllegalMoveError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AskEvent:
    asker: int
    target: int
    card: int
    success: bool

    def to_dict(self, spec: Spec) -> dict:
        return {
            "type": "ask",
            "asker": self.asker,
            "target": self.target,
            "card": self.card,
            "card_name": card_name(self.card, spec),
            "success": self.success,
        }

    def describe(self, spec: Spec) -> str:
        verb = "took" if self.success else "was denied"
        return (
            f"P{self.asker} asked P{self.target} for {card_name(self.card, spec)} "
            f"and {verb}"
        )


@dataclass(frozen=True, slots=True)
class ClaimEvent:
    claimer: int
    halfsuit: int
    named: tuple[int, ...]
    actual: tuple[int, ...]
    winner: int

    def to_dict(self, spec: Spec) -> dict:
        return {
            "type": "claim",
            "claimer": self.claimer,
            "halfsuit": self.halfsuit,
            "halfsuit_name": halfsuit_name(self.halfsuit, spec),
            "named": list(self.named),
            "actual": list(self.actual),
            "winner": self.winner,
            "correct": self.named == self.actual,
        }

    def describe(self, spec: Spec) -> str:
        ok = "correct" if self.named == self.actual else "incorrect"
        return (
            f"P{self.claimer} claimed {halfsuit_name(self.halfsuit, spec)} "
            f"({ok}) → team {self.winner}"
        )


@dataclass(frozen=True, slots=True)
class PassEvent:
    src: int
    dest: int
    kind: str  # "pass" or "designate"

    def to_dict(self, spec: Spec) -> dict:
        return {"type": self.kind, "src": self.src, "dest": self.dest}

    def describe(self, spec: Spec) -> str:
        verb = "designated" if self.kind == "designate" else "passed to"
        return f"P{self.src} {verb} P{self.dest}"


Event = Union[AskEvent, ClaimEvent, PassEvent]


class GameState:
    """Perfect-information state. Clone before mutating in search."""

    __slots__ = (
        "spec",
        "hands",
        "current_player",
        "scores",
        "claimed",
        "history",
        "plies",
    )

    def __init__(
        self,
        spec: Spec,
        hands: list[int],
        current_player: int = 0,
        scores: Optional[list[int]] = None,
        claimed: Optional[list[int]] = None,
        history: Optional[list[Event]] = None,
        plies: int = 0,
    ):
        self.spec = spec
        self.hands = hands
        self.current_player = current_player
        self.scores = scores if scores is not None else [0, 0]
        self.claimed = claimed if claimed is not None else [-1] * spec.n_halfsuits
        self.history = history if history is not None else []
        self.plies = plies

    def clone(self, record_history: bool = True) -> "GameState":
        return GameState(
            spec=self.spec,
            hands=self.hands[:],
            current_player=self.current_player,
            scores=self.scores[:],
            claimed=self.claimed[:],
            history=self.history[:] if record_history else [],
            plies=self.plies,
        )

    def hand_sizes(self) -> tuple[int, ...]:
        return tuple(popcount(h) for h in self.hands)

    def unclaimed_mask(self) -> int:
        mask = 0
        for hs, owner in enumerate(self.claimed):
            if owner < 0:
                mask |= self.spec.halfsuit_mask(hs)
        return mask

    def is_terminal(self) -> bool:
        if self.scores[0] >= self.spec.win_threshold:
            return True
        if self.scores[1] >= self.spec.win_threshold:
            return True
        return all(c >= 0 for c in self.claimed)

    def winner(self) -> Optional[int]:
        """Winning team, or None if unfinished / draw."""
        if not self.is_terminal():
            return None
        if self.scores[0] > self.scores[1]:
            return 0
        if self.scores[1] > self.scores[0]:
            return 1
        return None

    def team_has_cards(self, team: int) -> bool:
        return any(
            self.hands[p] != 0
            for p in range(self.spec.n_players)
            if self.spec.team(p) == team
        )

    def holder_of(self, card: int) -> Optional[int]:
        bit = 1 << card
        for p, h in enumerate(self.hands):
            if h & bit:
                return p
        return None

    def observe(self, player: int) -> "Observation":
        return Observation(
            spec=self.spec,
            player=player,
            hand=self.hands[player],
            hand_sizes=self.hand_sizes(),
            current_player=self.current_player,
            scores=tuple(self.scores),
            claimed=tuple(self.claimed),
            history=tuple(self.history),
            plies=self.plies,
        )

    def invariant_errors(self) -> list[str]:
        """Return a list of broken invariants (empty means OK)."""
        errors: list[str] = []
        spec = self.spec
        occupied = 0
        for p, h in enumerate(self.hands):
            if h & occupied:
                errors.append(f"duplicate cards in hand of P{p}")
            occupied |= h
            extra = h & ~spec.all_cards_mask()
            if extra:
                errors.append(f"P{p} holds out-of-range cards")
        claimed_mask = spec.all_cards_mask() & ~self.unclaimed_mask()
        if occupied & claimed_mask:
            errors.append("claimed cards still in hands")
        if popcount(occupied | claimed_mask) != spec.n_cards:
            errors.append("cards not conserved")
        if not (0 <= self.current_player < spec.n_players):
            errors.append("bad current player")
        if any(s < 0 for s in self.scores):
            errors.append("negative score")
        return errors


@dataclass(frozen=True, slots=True)
class Observation:
    """Information set of one player."""

    spec: Spec
    player: int
    hand: int
    hand_sizes: tuple[int, ...]
    current_player: int
    scores: tuple[int, int]
    claimed: tuple[int, ...]
    history: tuple[Event, ...]
    plies: int = 0

    def is_terminal(self) -> bool:
        if self.scores[0] >= self.spec.win_threshold:
            return True
        if self.scores[1] >= self.spec.win_threshold:
            return True
        return all(c >= 0 for c in self.claimed)

    def team(self) -> int:
        return self.spec.team(self.player)

    def my_cards(self) -> list[int]:
        return list(iter_bits(self.hand))


def new_game(spec: Spec = STANDARD, seed: Optional[int] = None) -> GameState:
    rng = random.Random(seed)
    cards = list(range(spec.n_cards))
    rng.shuffle(cards)
    hands = [0] * spec.n_players
    i = 0
    for p, sz in enumerate(spec.deal_sizes()):
        for _ in range(sz):
            hands[p] |= 1 << cards[i]
            i += 1
    return GameState(spec=spec, hands=hands, current_player=0)


def legal_asks_from(
    spec: Spec,
    player: int,
    hand: int,
    hand_sizes: Sequence[int],
    claimed: Sequence[int],
) -> list[Move]:
    if hand == 0:
        return []
    asks: list[Move] = []
    opponents = spec.opponents(player)
    live_opponents = [o for o in opponents if hand_sizes[o] > 0]
    if not live_opponents:
        return []
    for hs in range(spec.n_halfsuits):
        if claimed[hs] >= 0:
            continue
        hmask = spec.halfsuit_mask(hs)
        if hand & hmask == 0:
            continue
        missing = hmask & ~hand
        for card in iter_bits(missing):
            for opp in live_opponents:
                asks.append(("ask", opp, card))
    return asks


def legal_asks(state: GameState) -> list[Move]:
    return legal_asks_from(
        state.spec,
        state.current_player,
        state.hands[state.current_player],
        state.hand_sizes(),
        state.claimed,
    )


def legal_transfers(state: GameState) -> list[Move]:
    """Pass / designate when the current player has no cards."""
    p = state.current_player
    if state.hands[p] != 0:
        return []
    spec = state.spec
    mates = [
        t
        for t in spec.teammates(p)
        if state.hands[t] != 0
    ]
    if mates:
        return [("pass", t) for t in mates]
    opps = [o for o in spec.opponents(p) if state.hands[o] != 0]
    return [("designate", o) for o in opps]


def singleton_claims(state: GameState) -> list[Move]:
    """Claims where the current player holds every remaining card of a half-suit."""
    p = state.current_player
    spec = state.spec
    moves: list[Move] = []
    for hs in range(spec.n_halfsuits):
        if state.claimed[hs] >= 0:
            continue
        hmask = spec.halfsuit_mask(hs)
        if (state.hands[p] & hmask) == hmask:
            named = tuple(p for _ in range(spec.ranks_per_half))
            moves.append(("claim", hs, named))
    return moves


def perfect_info_claims(state: GameState) -> list[Move]:
    """Correct claims for every half-suit fully owned by the current player's team."""
    p = state.current_player
    spec = state.spec
    team = spec.team(p)
    moves: list[Move] = []
    for hs in range(spec.n_halfsuits):
        if state.claimed[hs] >= 0:
            continue
        named = []
        ok = True
        for card in spec.cards_of_halfsuit(hs):
            holder = state.holder_of(card)
            if holder is None or spec.team(holder) != team:
                ok = False
                break
            named.append(holder)
        if ok:
            moves.append(("claim", hs, tuple(named)))
    return moves


def legal_info_moves(state: GameState) -> list[Move]:
    """Moves determined by the information set (asks, pass, designate).

    Claims are not enumerated combinatorially; callers add their own.
    """
    if state.is_terminal():
        return []
    transfers = legal_transfers(state)
    if transfers:
        return transfers
    return legal_asks(state)


def actual_holders(state: GameState, hs: int) -> tuple[int, ...]:
    spec = state.spec
    holders = []
    for card in spec.cards_of_halfsuit(hs):
        holder = state.holder_of(card)
        if holder is None:
            raise IllegalMoveError(f"card {card} of half-suit {hs} is not in play")
        holders.append(holder)
    return tuple(holders)


def apply_move(state: GameState, move: Move, record: bool = True) -> GameState:
    """Apply `move` to a clone of `state` and return it."""
    if state.is_terminal():
        raise IllegalMoveError("game is over")
    kind = move[0]
    if kind == "ask":
        return _apply_ask(state, move[1], move[2], record)
    if kind == "claim":
        return _apply_claim(state, move[1], move[2], record)
    if kind == "pass":
        return _apply_pass(state, move[1], "pass", record)
    if kind == "designate":
        return _apply_pass(state, move[1], "designate", record)
    raise IllegalMoveError(f"unknown move {move}")


def _apply_ask(state: GameState, target: int, card: int, record: bool) -> GameState:
    s = state.clone(record_history=record)
    spec = s.spec
    p = s.current_player
    if not (0 <= target < spec.n_players):
        raise IllegalMoveError("bad target")
    if spec.team(target) == spec.team(p):
        raise IllegalMoveError("cannot ask a teammate")
    if s.hands[p] == 0:
        raise IllegalMoveError("empty-handed player cannot ask")
    if s.hands[target] == 0:
        raise IllegalMoveError("cannot ask a player with no cards")
    if not (0 <= card < spec.n_cards):
        raise IllegalMoveError("bad card")
    hs = spec.halfsuit_of(card)
    if s.claimed[hs] >= 0:
        raise IllegalMoveError("half-suit already claimed")
    bit = 1 << card
    if s.hands[p] & bit:
        raise IllegalMoveError("cannot ask for a card you hold")
    if s.hands[p] & spec.halfsuit_mask(hs) == 0:
        raise IllegalMoveError("must hold a card of that half-suit")
    success = (s.hands[target] & bit) != 0
    if success:
        s.hands[target] &= ~bit
        s.hands[p] |= bit
    else:
        s.current_player = target
    if record:
        s.history.append(AskEvent(p, target, card, success))
    s.plies += 1
    return s


def _apply_claim(
    state: GameState, hs: int, named: tuple[int, ...], record: bool
) -> GameState:
    s = state.clone(record_history=record)
    spec = s.spec
    p = s.current_player
    if not (0 <= hs < spec.n_halfsuits) or s.claimed[hs] >= 0:
        raise IllegalMoveError("cannot claim that half-suit")
    if len(named) != spec.ranks_per_half:
        raise IllegalMoveError("claim must name every card of the half-suit")
    claimer_team = spec.team(p)
    for owner in named:
        if spec.team(owner) != claimer_team:
            raise IllegalMoveError("claim may only name teammates")
    actual = actual_holders(s, hs)
    correct = named == actual
    opponent_has = any(spec.team(o) != claimer_team for o in actual)
    # Any error awards the set to the opponents.
    winner = claimer_team if correct and not opponent_has else 1 - claimer_team
    hmask = spec.halfsuit_mask(hs)
    for owner in range(spec.n_players):
        s.hands[owner] &= ~hmask
    s.claimed[hs] = winner
    s.scores[winner] += 1
    if record:
        s.history.append(ClaimEvent(p, hs, tuple(named), actual, winner))
    s.plies += 1
    return s


def _apply_pass(state: GameState, dest: int, kind: str, record: bool) -> GameState:
    s = state.clone(record_history=record)
    spec = s.spec
    p = s.current_player
    if s.hands[p] != 0:
        raise IllegalMoveError("can only pass with an empty hand")
    if not (0 <= dest < spec.n_players):
        raise IllegalMoveError("bad destination")
    if s.hands[dest] == 0:
        raise IllegalMoveError("destination has no cards")
    if kind == "pass":
        if spec.team(dest) != spec.team(p) or dest == p:
            raise IllegalMoveError("must pass to a teammate")
    else:
        if spec.team(dest) == spec.team(p):
            raise IllegalMoveError("must designate an opponent")
    s.current_player = dest
    if record:
        s.history.append(PassEvent(p, dest, kind))
    s.plies += 1
    return s


def perfect_info_policy(state: GameState, rng: random.Random) -> Move:
    """Greedy cheater used inside determinized worlds (PIMC / rollouts).

    1. Claim any half-suit the team fully owns (correct assignment).
    2. If empty-handed, pass / designate.
    3. Ask for a card the target actually holds; prefer that.
    4. Otherwise a random legal ask (should be rare).
    """
    if state.is_terminal():
        raise IllegalMoveError("no moves in a terminal state")
    claims = perfect_info_claims(state)
    if claims:
        # Prefer a claim if the team owns a set; first-to-five makes this free EV.
        return claims[0]
    transfers = legal_transfers(state)
    if transfers:
        return transfers[rng.randrange(len(transfers))]
    asks = legal_asks(state)
    if not asks:
        # No ask and no correct team-owned set: forced incorrect-or-unknown claim.
        # Under perfect info this means remaining cards of our half-suits sit with
        # teammates but we already claimed those... so this is an endgame hole.
        # Claim nothing illegally; if we truly have no move, pass is impossible.
        raise IllegalMoveError("perfect-info policy has no move")
    hits = [m for m in asks if state.hands[m[1]] & (1 << m[2])]
    pool = hits if hits else asks
    return pool[rng.randrange(len(pool))]


def describe_move(move: Move, spec: Spec = STANDARD) -> str:
    kind = move[0]
    if kind == "ask":
        return f"ask P{move[1]} for {card_name(move[2], spec)}"
    if kind == "claim":
        hs, named = move[1], move[2]
        parts = [
            f"{card_name(c, spec)}:P{named[i]}"
            for i, c in enumerate(spec.cards_of_halfsuit(hs))
        ]
        return f"claim {halfsuit_name(hs, spec)} ({', '.join(parts)})"
    if kind == "pass":
        return f"pass to P{move[1]}"
    if kind == "designate":
        return f"designate P{move[1]}"
    return str(move)


def play_game(
    agents: Sequence,
    spec: Spec = STANDARD,
    seed: Optional[int] = None,
    max_plies: int = 4000,
    on_move=None,
) -> GameState:
    """Play a full game. Each agent exposes `select(obs) -> Move`."""
    state = new_game(spec, seed)
    while not state.is_terminal() and state.plies < max_plies:
        agent = agents[state.current_player]
        obs = state.observe(state.current_player)
        move = agent.select(obs)
        state = apply_move(state, move)
        if on_move is not None:
            on_move(state, move)
    return state


def iter_unclaimed_halfsuits(claimed: Sequence[int]) -> Iterator[int]:
    for hs, owner in enumerate(claimed):
        if owner < 0:
            yield hs
