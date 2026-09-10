"""A fixed, documented candidate universe. A target is known only after that day's close."""

import pandas as pd

from .data import OHLC
from .indicators import adx, ema, rsi, tema, true_range, wilder

CANDIDATES = (
    "sma",
    "ema",
    "macd",
    "bollinger",
    "stochastic",
    "williams_r",
    "momentum",
    "rsi",
    "atr",
    "price_oscillator",
    "tema",
    "adx",
    "st_mo_macd",
    "po_wr",
    "po_rsi",
)


def _state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Hold between events; simultaneous entry and exit resolves to cash."""
    state = 0
    values = []
    for buy, sell in zip(entry.fillna(False), exit_.fillna(False), strict=True):
        if sell:
            state = 0
        elif buy:
            state = 1
        values.append(state)
    return pd.Series(values, index=entry.index, dtype=int)


def targets(frame: pd.DataFrame) -> dict[str, pd.Series]:
    close = frame.close
    sma20, ema20 = close.rolling(20).mean(), ema(close, 20)
    macd = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd, 9)
    sd = close.rolling(20).std(ddof=0)
    lo, hi = frame.low.rolling(14).min(), frame.high.rolling(14).max()
    span = (hi - lo).replace(0, float("nan"))
    stochastic = 100 * (close - lo) / span
    wr = -100 * (hi - close) / span
    momentum = close / close.shift(10) - 1
    strength = rsi(close, 5)
    atr = wilder(true_range(frame), 14)
    po = 100 * (ema(close, 12) / ema(close, 26) - 1)
    direction = adx(frame, 14)
    t = {col: tema(frame[col], 10) for col in OHLC}
    tema_entry = ((frame.low < t["low"]) | (frame.high < t["high"])) & (
        (close < t["close"]) | (frame.open < t["open"])
    )
    tema_exit = ((frame.low > t["low"]) | (frame.high > t["high"])) & (
        (close > t["close"]) | (frame.open > t["open"])
    )
    rules = {
        "sma": (close > sma20, close < sma20),
        "ema": (close > ema20, close < ema20),
        "macd": (macd > macd_signal, macd < macd_signal),
        "bollinger": (close < sma20 - 2 * sd, close >= sma20),
        "stochastic": (stochastic < 20, stochastic > 80),
        "williams_r": (wr < -80, wr > -20),
        "momentum": (momentum > 0, momentum < 0),
        "rsi": (strength < 30, strength > 70),
        "atr": (close > ema20 + 2 * atr, close < ema20),
        "price_oscillator": (po > 0, po < 0),
        "tema": (tema_entry, tema_exit),
        "adx": (
            (direction.plus_di > direction.minus_di) & (direction.adx > 25),
            (direction.minus_di > direction.plus_di) & (direction.adx > 25),
        ),
        "st_mo_macd": (
            (stochastic < 20) & (momentum > 0) & (macd > macd_signal),
            (stochastic > 80) | (momentum < 0) | (macd < macd_signal),
        ),
        "po_wr": ((po > 0) & (wr < -80), (po < 0) | (wr > -20)),
        "po_rsi": ((po > 0) & (strength < 30), (po < 0) | (strength > 70)),
    }
    return {name: _state(*rules[name]) for name in CANDIDATES}
