"""Strict single-asset OHLC data ingestion and an explicitly synthetic fixture."""

from pathlib import Path

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


def validate_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject ambiguity rather than silently sorting, filling, or repairing real data."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("OHLC data must use a DatetimeIndex")
    if frame.index.tz is not None:
        raise ValueError("Use timezone-free daily session dates")
    if frame.index.hasnans or not frame.index.equals(frame.index.normalize()):
        raise ValueError("Dates must be valid daily session dates without intraday times")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and strictly increasing; no automatic sorting")
    if len(frame) < 2 or not set(OHLC).issubset(frame.columns):
        raise ValueError("At least two rows and open, high, low, close columns are required")
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate columns are not supported")
    allowed = set(OHLC + ["volume"])
    if set(frame.columns) - allowed:
        raise ValueError("Only OHLC and optional volume are allowed; supply one asset per CSV")
    result = frame.copy()
    try:
        result = result.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Prices and volume must be numeric") from exc
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Missing or infinite prices/volume are not allowed")
    if (result[OHLC] <= 0).any().any():
        raise ValueError("All OHLC prices must be positive")
    if ((result.low > result[OHLC].min(axis=1)) | (result.high < result[OHLC].max(axis=1))).any():
        raise ValueError("Invalid candle: low <= open, close <= high is required")
    if "volume" in result and (result.volume < 0).any():
        raise ValueError("Volume must be non-negative")
    result.index.name = "date"
    return result


def read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower()
    if frame.columns.has_duplicates or "date" not in frame:
        raise ValueError("CSV must have a single date column and unique column names")
    try:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("date must contain ISO YYYY-MM-DD session dates") from exc
    return validate_ohlc(frame.set_index("date"))


def synthetic_ohlc(rows: int = 720, seed: int = 42) -> pd.DataFrame:
    """A seeded regime-changing stochastic path. It is not market data or a trading calendar."""
    if type(rows) is not int or rows < 2:
        raise ValueError("rows must be an integer >= 2")
    rng = np.random.default_rng(seed)
    t = np.arange(rows)
    drift = np.where((t // 90) % 2 == 0, 0.0008, -0.0004)
    vol = np.where((t // 120) % 2 == 0, 0.012, 0.025)
    overnight = rng.normal(0, vol * 0.25)
    intraday = drift + 0.001 * np.sin(t / 9) + rng.normal(0, vol)
    close = 100 * np.exp(np.cumsum(overnight + intraday))
    previous = np.r_[100, close[:-1]]
    opening = previous * np.exp(overnight)
    spread = abs(rng.normal(0, vol * 0.6)) + 0.001
    return validate_ohlc(
        pd.DataFrame(
            {
                "open": opening,
                "high": np.maximum(opening, close) * np.exp(spread),
                "low": np.minimum(opening, close) * np.exp(-spread),
                "close": close,
                "volume": rng.integers(100000, 2000000, rows).astype(float),
            },
            index=pd.bdate_range("2018-01-02", periods=rows, name="date"),
        )
    )


def project_candle(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """Repair forecast geometry only. Preserve open/close and expand high/low envelope."""
    values = np.asarray(values, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Forecast must contain four finite OHLC values")
    repaired = np.maximum(values, 1e-8)
    repaired[1] = max(repaired)
    repaired[2] = min(repaired[0], repaired[2], repaired[3])
    return repaired, not np.array_equal(values, repaired)
