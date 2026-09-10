import pytest

from dnn_forward_lab.config import Config
from dnn_forward_lab.data import synthetic_ohlc


@pytest.fixture
def prices():
    return synthetic_ohlc(210, 7)


@pytest.fixture
def fast_config():
    return Config(
        min_train=150, horizon=10, folds=2, epochs=3, batch_size=64, validation_size=10, patience=2
    )
