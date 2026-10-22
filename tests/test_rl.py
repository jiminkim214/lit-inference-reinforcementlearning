import numpy as np

from litbot.cards import TOY_6
from litbot.game import new_game
from litbot.rl.value import LinearValue, features, train_from_games


def test_features_finite():
    state = new_game(TOY_6, seed=0)
    x = features(state.observe(0))
    assert x.ndim == 1
    assert np.isfinite(x).all()


def test_ols_fit():
    state = new_game(TOY_6, seed=1)
    obs = [state.observe(0), state.observe(1)]
    model = train_from_games([(obs, 1.0), (obs, -1.0)])
    assert isinstance(model, LinearValue)
    v = model.predict(state.observe(0))
    assert np.isfinite(v)
