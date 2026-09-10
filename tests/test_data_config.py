from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dnn_forward_lab.config import Config
from dnn_forward_lab.data import OHLC, project_candle, read_csv, synthetic_ohlc, validate_ohlc


def test_csv_roundtrip(prices, tmp_path):
    path = tmp_path / "input.csv"
    prices.to_csv(path)
    pd.testing.assert_frame_equal(read_csv(path), prices, check_freq=False)


@pytest.mark.parametrize(
    "problem",
    [
        "nan",
        "inf",
        "negative",
        "candle",
        "duplicate",
        "reverse",
        "time",
        "timezone",
        "volume",
        "text",
        "symbol",
    ],
)
def test_invalid_data_rejected(prices, problem):
    bad = prices.copy()
    if problem in ["nan", "inf", "negative"]:
        bad.iloc[4, 0] = {"nan": np.nan, "inf": np.inf, "negative": -1}[problem]
    elif problem == "candle":
        bad.iloc[4, 1] = 0.1
    elif problem == "duplicate":
        bad = pd.concat([bad.iloc[:1], bad])
    elif problem == "reverse":
        bad = bad.iloc[::-1]
    elif problem == "time":
        bad.index += pd.Timedelta(hours=1)
    elif problem == "timezone":
        bad.index = bad.index.tz_localize("UTC")
    elif problem == "volume":
        bad.iloc[4, 4] = -1
    elif problem == "text":
        bad["open"] = "invalid"
    else:
        bad["symbol"] = "A"
    with pytest.raises(ValueError):
        validate_ohlc(bad)


def test_missing_date_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("time,open\n2020-01-01,1\n")
    with pytest.raises(ValueError, match="date"):
        read_csv(path)


def test_bad_date_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,open,high,low,close\ninvalid,1,1,1,1\n")
    with pytest.raises(ValueError, match="ISO"):
        read_csv(path)


@pytest.mark.parametrize("raw", [[10, 7, 15, 11], [-1, 2, -4, 3], [1, 1, 1, 1]])
def test_forecast_projection_is_valid(raw):
    candle, repaired = project_candle(np.array(raw))
    o, h, lo, c = candle
    assert 0 < lo <= min(o, c) <= max(o, c) <= h
    assert repaired == (raw != candle.tolist())


def test_synthetic_is_reproducible_and_coherent():
    pd.testing.assert_frame_equal(synthetic_ohlc(seed=42), synthetic_ohlc(seed=42))
    assert not synthetic_ohlc(seed=42)[OHLC].equals(synthetic_ohlc(seed=43)[OHLC])


@pytest.mark.parametrize(
    "setting,value",
    [
        ("horizon", 0),
        ("seed", -1),
        ("epochs", 1.5),
        ("dropout", 1),
        ("dropout", float("nan")),
        ("commission_bps", -1),
        ("slippage_bps", 10000),
        ("learning_rate", 0),
        ("initial_cash", 0),
        ("min_train", 5),
        ("folds", True),
        ("validation_size", 1),
    ],
)
def test_invalid_settings_fail(setting, value):
    with pytest.raises(ValueError):
        replace(Config(), **{setting: value})


def test_toml_unknown_key_is_error(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("epohcs = 5\n")
    with pytest.raises(ValueError, match="Unknown"):
        Config.from_toml(path)
