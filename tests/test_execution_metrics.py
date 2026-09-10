from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from dnn_forward_lab.backtest import simulate
from dnn_forward_lab.config import Config
from dnn_forward_lab.metrics import forecast_metrics, performance


def candles(opens, closes):
    return pd.DataFrame(
        {
            "open": opens,
            "close": closes,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
        },
        index=pd.bdate_range("2020-01-01", periods=len(opens)),
    )


def test_next_open_and_exact_fee_slippage_accounting():
    frame = candles([50.0, 100.0, 120.0], [50.0, 110.0, 130.0])
    target = pd.Series([1, 0, 0], index=frame.index)
    config = Config(initial_cash=1000, commission_bps=10, slippage_bps=20)
    result = simulate(frame, target, 1, config)
    shares = 1000 / (100 * 1.002 * 1.001)
    expected = shares * 120 * 0.998 * 0.999
    assert result.orders.fill_price.tolist() == [100 * 1.002, 120 * 0.998]
    assert result.daily.equity.iloc[-1] == pytest.approx(expected)
    assert result.trades.pnl.iloc[0] == pytest.approx(expected - 1000)
    assert result.orders.iloc[0].signal_date < result.orders.iloc[0].date
    assert result.daily.shares.iloc[-1] == 0


def test_signal_cannot_execute_at_same_close():
    frame = candles([100.0, 100.0, 500.0], [100.0, 500.0, 500.0])
    target = pd.Series([0, 1, 0], index=frame.index)
    result = simulate(frame, target, 1, Config(commission_bps=0, slippage_bps=0))
    assert result.daily.equity.iloc[-1] == 10000
    assert result.orders.iloc[0].date == frame.index[2]
    assert result.orders.iloc[-1].reason == "scheduled_terminal_close"


def test_terminal_liquidation_charged():
    frame = candles([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    config = Config(commission_bps=10, slippage_bps=0)
    result = simulate(frame, pd.Series(1, index=frame.index), 1, config)
    assert result.daily.equity.iloc[-1] == pytest.approx(10000 / 1.001 * 0.999)
    assert len(result.trades) == 1
    assert result.orders.iloc[-1].reason == "scheduled_terminal_close"


def test_cash_is_constant_and_aligned(prices):
    result = simulate(prices, pd.Series(0, index=prices.index), 100, Config())
    assert result.daily.equity.eq(10000).all()
    assert len(result.orders) == 0 and len(result.trades) == 0
    assert result.daily.index.equals(prices.index[100:])


def test_more_costs_reduce_identical_strategy_returns(prices):
    config = Config()
    target = pd.Series(np.arange(len(prices)) % 2, index=prices.index)
    low = simulate(prices, target, 100, replace(config, commission_bps=0, slippage_bps=0))
    high = simulate(prices, target, 100, config)
    assert high.daily.equity.iloc[-1] < low.daily.equity.iloc[-1]


def test_shifted_target_index_rejected(prices):
    target = pd.Series(1, index=prices.index + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="align"):
        simulate(prices, target, 1, Config())


def test_metrics_include_initial_drawdown_and_compounding():
    result = performance([-0.1, 0.2, -0.25], trade_returns=[0.2, -0.1])
    assert result["total_return"] == pytest.approx(0.9 * 1.2 * 0.75 - 1)
    assert result["max_drawdown"] == pytest.approx(0.25)
    assert performance([-0.1])["max_drawdown"] == pytest.approx(0.1)
    assert result["expectancy_ratio"] == pytest.approx(0.5)
    assert result["win_rate"] == 0.5


def test_undefined_ratios_are_null():
    result = performance([0.0, 0.0, 0.0])
    for key in ["sharpe", "sortino", "calmar", "mean_trade_return", "expectancy_ratio"]:
        assert result[key] is None
    assert performance([0.1, 0.2])["sortino"] is None


def test_forecast_errors_hand_calculated():
    result = forecast_metrics([1, 2, 3], [2, 2, 5])
    assert result["mse"] == pytest.approx(5 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert result["mae"] == 1
    assert result["mape_pct"] == pytest.approx((1 + 2 / 3) / 3 * 100)
    assert forecast_metrics([1, 1], [2, 2])["explained_variance"] is None
    assert forecast_metrics([0, 1], [0, 1])["mape_pct"] is None


@pytest.mark.parametrize("values", [[], [float("nan")], [-1], [float("inf")]])
def test_invalid_returns_fail(values):
    with pytest.raises(ValueError):
        performance(values)
