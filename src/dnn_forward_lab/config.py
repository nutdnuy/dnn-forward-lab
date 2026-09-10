"""Validated, immutable experiment settings. Unknown settings are errors."""

import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Config:
    lookback: int = 5
    horizon: int = 30
    min_train: int = 504
    folds: int = 3
    epochs: int = 100
    batch_size: int = 5
    learning_rate: float = 0.001
    dropout: float = 0.002
    validation_size: int = 30
    patience: int = 15
    seed: int = 42
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    initial_cash: float = 10000.0
    periods_per_year: int = 252

    def __post_init__(self):
        integers = (
            "lookback",
            "horizon",
            "min_train",
            "folds",
            "epochs",
            "batch_size",
            "validation_size",
            "patience",
            "periods_per_year",
        )
        for name in integers:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("learning_rate", "dropout", "commission_bps", "slippage_bps", "initial_cash"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.dropout < 1 or not 0 < self.learning_rate <= 1:
            raise ValueError("dropout must be in [0, 1); learning_rate in (0, 1]")
        if any(not 0 <= getattr(self, k) < 10000 for k in ("commission_bps", "slippage_bps")):
            raise ValueError("costs must be in [0, 10000) basis points per side")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.min_train < max(100 + self.horizon, self.validation_size + self.lookback + 20):
            raise ValueError(
                "min_train needs 100 warm-up bars plus horizon and enough fitting data"
            )
        if self.validation_size < 2:
            raise ValueError("validation_size must be at least 2")
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2 for forecast error evaluation")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_toml(cls, path: str | Path) -> "Config":
        with Path(path).open("rb") as handle:
            values = tomllib.load(handle)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)
