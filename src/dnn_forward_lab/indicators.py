"""Causal indicators with explicit warm-up and Wilder smoothing."""

import numpy as np
import pandas as pd


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Seed with the first complete arithmetic mean, then use alpha=1/period."""
    if period < 1:
        raise ValueError("period must be positive")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    seed = series.rolling(period, min_periods=period).mean().first_valid_index()
    if seed is None:
        return out
    start = series.index.get_loc(seed)
    out.iloc[start] = series.iloc[start - period + 1 : start + 1].mean()
    for i in range(start + 1, len(series)):
        out.iloc[i] = (out.iloc[i - 1] * (period - 1) + series.iloc[i]) / period
    return out


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def tema(series: pd.Series, period: int = 10) -> pd.Series:
    first = ema(series, period)
    second = ema(first, period)
    return 3 * first - 3 * second + ema(second, period)


def rsi(close: pd.Series, period: int = 5) -> pd.Series:
    change = close.diff()
    gains = wilder(change.clip(lower=0), period)
    losses = wilder(-change.clip(upper=0), period)
    ratio = gains / losses.replace(0, np.nan)
    result = 100 - 100 / (1 + ratio)
    result = result.mask((losses == 0) & (gains > 0), 100)
    return result.mask((losses == 0) & (gains == 0), 50)


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous = frame.close.shift(1)
    return pd.concat(
        [
            frame.high - frame.low,
            (frame.high - previous).abs(),
            (frame.low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)


def adx(frame: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    up, down = frame.high.diff(), -frame.low.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    plus.iloc[0] = minus.iloc[0] = np.nan
    tr = true_range(frame)
    tr.iloc[0] = np.nan
    atr = wilder(tr, period)
    pos, neg = 100 * wilder(plus, period) / atr, 100 * wilder(minus, period) / atr
    pos, neg = pos.mask(atr == 0, 0), neg.mask(atr == 0, 0)
    denominator = pos + neg
    dx = (100 * (pos - neg).abs() / denominator).mask(denominator == 0, 0)
    return pd.DataFrame({"plus_di": pos, "minus_di": neg, "adx": wilder(dx, period)})
