"""CPU DNNs with chronological validation and recursive, observation-free future paths."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import Config
from .data import OHLC, project_candle, validate_ohlc


@dataclass(frozen=True)
class Scale:
    minimum: float
    width: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "Scale":
        return cls(float(values.min()), max(float(np.ptp(values)), 1e-8))

    def transform(self, values):
        return (values - self.minimum) / self.width

    def inverse(self, values):
        return values * self.width + self.minimum


def lagged(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    if lookback < 1 or len(values) <= lookback:
        raise ValueError("Need more observations than lookback")
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback + 1)
    return windows[:, :-1].copy(), windows[:, -1:].copy()


class PriceNet(nn.Module):
    def __init__(self, lookback: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(lookback, 10 * lookback),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(10 * lookback, 5 * lookback),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(5 * lookback, 1),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class DNNForecaster:
    """One independent network per OHLC field and per asset. Never pools assets implicitly."""

    def __init__(self, config: Config):
        self.config = config
        self.networks: dict[str, PriceNet] = {}
        self.scales: dict[str, Scale] = {}
        self.diagnostics: dict = {}
        self.history: pd.DataFrame | None = None
        self.repaired_bars = 0

    def fit(self, history: pd.DataFrame) -> "DNNForecaster":
        history = validate_ohlc(history)[OHLC]
        c = self.config
        fit_end = len(history) - c.validation_size
        if fit_end <= c.lookback + 1:
            raise ValueError("Insufficient history for fitting and chronological validation")
        self.history = history.copy()
        self.diagnostics = {
            "fit_end": str(history.index[fit_end - 1].date()),
            "validation_start": str(history.index[fit_end].date()),
            "validation_end": str(history.index[-1].date()),
            "validation_objective": "recursive MAE in scaled price units; no teacher forcing",
            "refit_on_validation": False,
            "device": "cpu",
            "channels": {},
        }
        # Scope RNG and threading changes so using the library does not reseed caller code.
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                for channel, name in enumerate(OHLC):
                    torch.manual_seed((c.seed + channel) % 2**32)
                    values = history[name].to_numpy()
                    scale = Scale.fit(values[:fit_end])
                    x, y = lagged(scale.transform(values[:fit_end]), c.lookback)
                    x, y = (
                        torch.tensor(x, dtype=torch.float32),
                        torch.tensor(y, dtype=torch.float32),
                    )
                    net = PriceNet(c.lookback, c.dropout)
                    optimizer = torch.optim.Adam(net.parameters(), lr=c.learning_rate)
                    loss_fn = nn.L1Loss()
                    best, stale, best_epoch, best_state = float("inf"), 0, 0, None
                    curve = []
                    for epoch in range(c.epochs):
                        net.train()
                        losses = []
                        # Chronological batches. No randomized split or cross-validation.
                        for start in range(0, len(x), c.batch_size):
                            optimizer.zero_grad(set_to_none=True)
                            loss = loss_fn(
                                net(x[start : start + c.batch_size]),
                                y[start : start + c.batch_size],
                            )
                            loss.backward()
                            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                            optimizer.step()
                            losses.append(
                                float(loss.detach()) * len(x[start : start + c.batch_size])
                            )
                        net.eval()
                        predicted = self._recursive_channel(
                            net,
                            scale.transform(values[fit_end - c.lookback : fit_end]),
                            c.validation_size,
                        )
                        validation = float(
                            np.mean(abs(predicted - scale.transform(values[fit_end:])))
                        )
                        if not np.isfinite(validation):
                            raise ValueError(f"Training diverged for {name}")
                        curve.append(
                            {
                                "epoch": epoch + 1,
                                "train_mae": sum(losses) / len(x),
                                "val_mae": validation,
                            }
                        )
                        if validation < best - 1e-8:
                            best, stale, best_epoch = validation, 0, epoch + 1
                            best_state = copy.deepcopy(net.state_dict())
                        else:
                            stale += 1
                        if stale >= c.patience:
                            break
                    net.load_state_dict(best_state)
                    net.eval()
                    self.networks[name], self.scales[name] = net, scale
                    self.diagnostics["channels"][name] = {
                        "best_epoch": best_epoch,
                        "epochs_run": len(curve),
                        "validation_mae_scaled": best,
                        "scale_min": scale.minimum,
                        "scale_width": scale.width,
                        "loss_curve": curve,
                    }
        finally:
            torch.set_num_threads(threads)
        return self

    @staticmethod
    def _recursive_channel(net: PriceNet, seed: np.ndarray, steps: int) -> np.ndarray:
        buffer = list(seed)
        predictions = []
        with torch.inference_mode():
            for _ in range(steps):
                x = torch.tensor([buffer[-len(seed) :]], dtype=torch.float32)
                value = float(net(x).item())
                predictions.append(value)
                buffer.append(value)
        return np.array(predictions)

    def predict(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        if self.history is None:
            raise ValueError("Fit the forecaster before prediction")
        if len(dates) < 1 or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Prediction dates must be nonempty, unique and ordered")
        if dates.tz is not None or dates.hasnans or not dates.equals(dates.normalize()):
            raise ValueError("Prediction dates must be timezone-free daily session dates")
        if dates[0] <= self.history.index[-1]:
            raise ValueError("Prediction dates must all follow observed history")
        buffers = {name: list(self.history[name].to_numpy()) for name in OHLC}
        predictions = []
        self.repaired_bars = 0
        with torch.inference_mode():
            for _ in dates:
                raw = []
                for name in OHLC:
                    scale = self.scales[name]
                    x = scale.transform(np.array(buffers[name][-self.config.lookback :]))
                    value = float(
                        self.networks[name](torch.tensor(x[None], dtype=torch.float32)).item()
                    )
                    raw.append(scale.inverse(value))
                candle, changed = project_candle(np.array(raw))
                self.repaired_bars += changed
                predictions.append(candle)
                # Repaired forecast values, never future observations, feed subsequent steps.
                for name, value in zip(OHLC, candle, strict=True):
                    buffers[name].append(value)
        return (
            validate_ohlc(pd.DataFrame(predictions, columns=OHLC, index=dates))
            if len(dates) > 1
            else pd.DataFrame(predictions, columns=OHLC, index=dates)
        )

    def save(self, directory: str | Path) -> None:
        """Save non-executable tensor arrays; no pickle checkpoint loading is exposed."""
        if not self.networks:
            raise ValueError("Cannot save an unfitted model")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1,
            "config": self.config.to_dict(),
            "scales": {
                name: {"minimum": scale.minimum, "width": scale.width}
                for name, scale in self.scales.items()
            },
            "diagnostics": self.diagnostics,
        }
        (directory / "snapshot.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        self.history.tail(max(2, self.config.lookback)).to_csv(
            directory / "context.csv", float_format="%.17g"
        )
        for name, net in self.networks.items():
            np.savez_compressed(
                directory / f"{name}.npz",
                **{key: value.detach().numpy() for key, value in net.state_dict().items()},
            )

    @classmethod
    def load(cls, directory: str | Path) -> "DNNForecaster":
        """Load JSON/CSV/NPZ format. Never deserialize executable pickle."""
        from .data import read_csv

        directory = Path(directory)
        metadata = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("Unsupported model snapshot format")
        model = cls(Config(**metadata["config"]))
        model.history = read_csv(directory / "context.csv")
        if len(model.history) < model.config.lookback:
            raise ValueError("Snapshot has insufficient price context")
        model.diagnostics = metadata["diagnostics"]
        with torch.random.fork_rng(devices=[]):
            for name in OHLC:
                scale = Scale(**metadata["scales"][name])
                if not np.isfinite([scale.minimum, scale.width]).all() or scale.width <= 0:
                    raise ValueError("Invalid model scale")
                net = PriceNet(model.config.lookback, model.config.dropout)
                with np.load(directory / f"{name}.npz", allow_pickle=False) as archive:
                    expected = net.state_dict()
                    if set(archive.files) != set(expected):
                        raise ValueError("Model tensor names do not match the architecture")
                    state = {}
                    for key, tensor in expected.items():
                        array = archive[key]
                        if array.shape != tuple(tensor.shape) or not np.isfinite(array).all():
                            raise ValueError("Model tensor shape or values are invalid")
                        state[key] = torch.tensor(array, dtype=tensor.dtype)
                net.load_state_dict(state)
                net.eval()
                model.networks[name], model.scales[name] = net, scale
        return model


def baseline_forecast(history: pd.DataFrame, dates: pd.DatetimeIndex, method: str) -> pd.DataFrame:
    history = validate_ohlc(history)
    if method == "persistence":
        raw = np.tile(history[OHLC].iloc[-1].to_numpy(), (len(dates), 1))
    elif method == "drift":
        logs = np.log(history[OHLC].to_numpy())
        drift = (logs[-1] - logs[0]) / (len(logs) - 1)
        raw = np.exp(logs[-1] + np.arange(1, len(dates) + 1)[:, None] * drift)
    elif method == "arima":
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise ValueError("Install dnn-forward-lab[statistics] to use ARIMA") from exc
        # Fixed ARIMA(1,1,1) on log prices, deliberately not a claim of paper auto-ARIMA.
        channels = []
        for name in OHLC:
            fitted = ARIMA(np.log(history[name].to_numpy()), order=(1, 1, 1)).fit(
                method_kwargs={"maxiter": 200}
            )
            if not fitted.mle_retvals.get("converged", False):
                raise ValueError(f"ARIMA did not converge for {name}; inspect this baseline")
            channels.append(np.exp(fitted.forecast(len(dates))))
        raw = np.column_stack(channels)
    else:
        raise ValueError(f"Unknown forecast baseline: {method}")
    return pd.DataFrame([project_candle(row)[0] for row in raw], columns=OHLC, index=dates)
