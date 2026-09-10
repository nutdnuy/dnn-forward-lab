"""Metric definitions are independent of the paper's inconsistent displayed equations."""

import numpy as np


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 1e-12 else None


def forecast_metrics(actual, forecast) -> dict:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    if actual.shape != forecast.shape or actual.size < 2:
        raise ValueError("Forecast and actual must have equal shapes and at least two values")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("Metrics require finite inputs")
    errors = forecast - actual
    variance = float(np.var(actual))
    return {
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(abs(errors))),
        "mape_pct": float(np.mean(abs(errors / actual)) * 100) if (actual != 0).all() else None,
        "explained_variance": 1 - float(np.var(errors)) / variance if variance > 1e-12 else None,
    }


def performance(returns, periods_per_year: int = 252, trade_returns=None) -> dict:
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or len(r) < 1 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Returns must be a nonempty finite vector strictly above -1")
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    equity = np.r_[1.0, np.cumprod(1 + r)]
    drawdown = equity / np.maximum.accumulate(equity) - 1
    annual_log = float(np.log1p(r).sum() * periods_per_year / len(r))
    annualized = float(np.expm1(annual_log)) if annual_log < 700 else None
    volatility = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)))
    mdd = max(0.0, float(-drawdown.min()))
    trades = np.asarray([] if trade_returns is None else trade_returns, dtype=float)
    if not np.isfinite(trades).all():
        raise ValueError("Trade returns must be finite")
    losses = trades[trades < 0]
    return {
        "observations": len(r),
        "total_return": float(equity[-1] - 1),
        "annualized_return": annualized,
        "max_drawdown": mdd,
        "annualized_volatility": volatility * np.sqrt(periods_per_year),
        "sharpe": _ratio(float(r.mean()) * np.sqrt(periods_per_year), volatility),
        "sortino": _ratio(float(r.mean()) * np.sqrt(periods_per_year), downside),
        "calmar": _ratio(annualized, mdd) if annualized is not None else None,
        "trades": len(trades),
        "win_rate": float(np.mean(trades > 0)) if len(trades) else None,
        "mean_trade_return": float(trades.mean()) if len(trades) else None,
        "expectancy_ratio": _ratio(float(trades.mean()), float(-losses.mean()))
        if len(losses)
        else None,
    }
