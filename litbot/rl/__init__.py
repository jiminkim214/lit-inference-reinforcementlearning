"""Stretch goal: a linear value function trained from self-play outcomes."""

from litbot.rl.value import LinearValue, features, train_from_games

__all__ = ["LinearValue", "features", "train_from_games"]
