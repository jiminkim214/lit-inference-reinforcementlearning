# Literature rules spec (LitBot)

This is the ruleset LitBot plays. Pagat-style procedures cover empty hands, turn passing, and endgame claims. Where Pagat disagrees with the claim penalty below, this spec wins.

## Players, teams, cards

- Six players, two teams of three, sitting alternately.
- Team 0 is players `{0, 2, 4}`. Team 1 is players `{1, 3, 5}`.
- Standard 52-card deck with the four 8s removed (48 cards).
- Eight half-suits of six cards:
  - Low: 2–7 of each suit
  - High: 9–A of each suit
- Each player is dealt eight cards. Player 0 (dealer) asks first.
- No table talk. The only shared information is the public history of asks, transfers, claims, hand sizes, and cards shown during claims.

## Asking

On your turn, if you have at least one card and some opponent still has cards, you must ask an opponent for a specific card, or claim.

A question is legal iff:

1. You name a specific card (rank and suit).
2. You hold at least one other card of that half-suit.
3. You do not hold the named card.
4. The target is an opponent.
5. The target holds at least one card.
6. The named card’s half-suit is still unclaimed.

If the target has the card, they pass it face-up and you keep the turn (you must ask or claim again; you may change targets).

If the target does not have the card, the turn passes to that target.

You may never ask a teammate.

## Claiming

Claims are actions on your turn (before, instead of, or after asks). You name a half-suit and, for each of its six cards, exactly one teammate (including yourself) who you say holds it.

Resolution:

- If every named location is correct, your team wins the half-suit.
- If any card is assigned to the wrong teammate, **or** any card is actually held by an opponent, the **opponents** win the half-suit.

This is stricter than Pagat (Pagat cancels a half-suit when your team holds it all but you mis-name teammates). We always award an incorrect claim to the opponents.

After a claim:

- Every card of that half-suit is shown, then removed from the game.
- The actual holders become public knowledge.
- If the game is not over, **the same player keeps the turn**.

A player may claim a half-suit they do not hold any cards in.

## Empty hands and turn passing

- A player with no cards cannot be asked.
- If it is your turn and you have no cards, but a teammate still has cards, you **pass** the turn to a teammate of your choice who has cards.
- You can run out of cards on your own turn because of a claim. Then you pass as above.
- If your whole team has no cards, no more asks are possible. You **designate** an opponent who still has cards; that opponent must claim the remaining half-suits (without consulting teammates).
- If the team that still has cards also has the turn, that player must claim remaining half-suits.

## End of the game

The first team to claim **five** half-suits wins. If all eight half-suits are claimed without either team reaching five (4–4), the game is a draw.

Remaining unclaimed half-suits are only forced out in the endgame case above; there is no silent scoring of unclaimed sets.

## Illegal actions

The engine never generates illegal asks. If an agent returns an illegal move, `apply_move` raises `IllegalMoveError`. Invalid questions in a human game are rejected and the player must choose again.

## Public information

Always public:

- Who asked whom for which card, and whether it succeeded
- Current hand sizes
- Which half-suits have been claimed, by whom
- The true pre-claim location of every card in a claimed half-suit
- The current location of any card that was transferred face-up, until it is claimed or transferred again

Not public: anyone else’s remaining unrevealed cards, and any private convention between teammates.
