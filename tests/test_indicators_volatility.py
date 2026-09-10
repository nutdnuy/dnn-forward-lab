import numpy as np
import pandas as pd
import pytest

from dnn_forward_lab.data import OHLC, synthetic_ohlc
from dnn_forward_lab.indicators import adx, ema, rsi, tema, wilder
from dnn_forward_lab.strategies import CANDIDATES, targets
from dnn_forward_lab.volatility import cluster_assets, historical_volatility


def test_wilder_seed_and_recurrence():
    series = pd.Series([1.0, 2.0, 3.0, 7.0, 8.0])
    smoothed = wilder(series, 3)
    assert smoothed.iloc[:2].isna().all()
    np.testing.assert_allclose(smoothed.iloc[2:], [2, 11 / 3, 46 / 9])


def test_rsi_endpoints_and_flat_market():
    assert rsi(pd.Series(np.arange(1.0, 30.0))).dropna().eq(100).all()
    assert rsi(pd.Series(np.arange(30.0, 1.0, -1))).dropna().eq(0).all()
    assert rsi(pd.Series(np.ones(30))).dropna().eq(50).all()


def test_flat_ema_tema_and_adx(prices):
    flat = prices.copy()
    flat[OHLC] = 100.0
    assert ema(flat.close, 5).dropna().eq(100).all()
    assert tema(flat.close, 5).dropna().eq(100).all()
    assert adx(flat).dropna().eq(0).all().all()


def test_all_strategies_and_volatility_are_causal(prices):
    cutoff = 150
    altered = prices.copy()
    altered.iloc[cutoff:, :4] *= 10
    before, after = targets(prices), targets(altered)
    assert tuple(before) == CANDIDATES
    assert len(before) == 15
    for name in before:
        assert before[name].isin([0, 1]).all()
        pd.testing.assert_series_equal(before[name].iloc[:cutoff], after[name].iloc[:cutoff])
    pd.testing.assert_frame_equal(
        historical_volatility(prices).iloc[:cutoff],
        historical_volatility(altered).iloc[:cutoff],
    )


def test_volatility_closed_form_constant_candle():
    n, window = 50, 20
    frame = pd.DataFrame(
        {"open": [100.0] * n, "high": [110.0] * n, "low": [90.0] * n, "close": [100.0] * n},
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    result = historical_volatility(frame, window, 1).iloc[-1]
    hl = np.log(110 / 90)
    rs = np.log(1.1) ** 2 + np.log(0.9) ** 2
    k = 0.34 / (1.34 + 21 / 19)
    np.testing.assert_allclose(
        result,
        [
            np.sqrt(hl**2 / (4 * np.log(2))),
            np.sqrt(0.5 * hl**2),
            np.sqrt(rs),
            np.sqrt((1 - k) * rs),
        ],
    )


def test_yang_zhang_captures_overnight_variance():
    prices = 100 * np.exp(np.cumsum(np.tile([0.02, -0.02], 30)))
    frame = pd.DataFrame(
        {name: prices for name in OHLC}, index=pd.bdate_range("2020-01-01", periods=60)
    )
    result = historical_volatility(frame).dropna()
    assert result[["parkinson", "garman_klass", "rogers_satchell"]].eq(0).all().all()
    assert result.yang_zhang.gt(0).all()


def test_clustering_ignores_future_observations():
    assets = {name: synthetic_ohlc(120, seed) for name, seed in [("A", 1), ("B", 2), ("C", 3)]}
    cutoff = str(assets["A"].index[89].date())
    original = cluster_assets(assets, cutoff, 2)
    assets["A"].iloc[90:, :4] *= 100
    assert original == cluster_assets(assets, cutoff, 2)
    assert {row["cluster"] for row in original["assets"]} == {0, 1}


def test_clustering_rejects_excess_clusters(prices):
    with pytest.raises(ValueError, match="asset count"):
        cluster_assets({"A": prices}, "2025-01-01", 2)
