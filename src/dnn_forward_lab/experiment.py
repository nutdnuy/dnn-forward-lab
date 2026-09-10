"""Orchestration keeps strategy selection physically separated from observed future data."""

import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import __version__
from .backtest import simulate
from .config import Config
from .data import OHLC, validate_ohlc
from .metrics import forecast_metrics, performance
from .models import DNNForecaster, baseline_forecast
from .strategies import targets


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fold_boundaries(rows: int, config: Config) -> list[tuple[int, int]]:
    required = config.min_train + config.folds * config.horizon
    if rows < required:
        raise ValueError(f"Need at least {required} rows; received {rows}")
    return [
        (config.min_train + i * config.horizon, config.min_train + (i + 1) * config.horizon)
        for i in range(config.folds)
    ]


def rank_strategies(
    history: pd.DataFrame, future: pd.DataFrame | None, config: Config
) -> list[dict]:
    """Only history and the generated path may enter forward selection."""
    if future is None:
        frame, start = history, len(history) - config.horizon
    else:
        frame, start = pd.concat([history[OHLC], future[OHLC]]), len(history)
    ranked = []
    for name, target in targets(frame).items():
        result = simulate(frame, target, start, config)
        ranked.append(
            {
                "strategy": name,
                "net_return": float(result.daily.equity.iloc[-1] / config.initial_cash - 1),
                "trades": len(result.trades),
            }
        )
    # Stable lexicographic tie break, declared before seeing any results.
    return sorted(ranked, key=lambda row: (-row["net_return"], row["strategy"]))


def run_experiment(
    frame: pd.DataFrame,
    config: Config,
    output: str | Path,
    *,
    asset: str,
    source: str,
    adjustment: str,
    synthetic: bool = False,
    include_arima: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    frame = validate_ohlc(frame)
    boundaries = fold_boundaries(len(frame), config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(
            "Output directory must be empty to prevent mixing or overwriting experiments"
        )
    for name, value in (("asset", asset), ("source", source), ("adjustment", adjustment)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be an explicit nonempty provenance label")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "torch", "scikit-learn", "matplotlib")
    }
    if include_arima:
        versions["statsmodels"] = importlib.metadata.version("statsmodels")
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode() + b"\0" + path.read_bytes())
    csv = frame.to_csv(float_format="%.17g", date_format="%Y-%m-%d")
    frame.to_csv(output / "input.csv", float_format="%.17g", date_format="%Y-%m-%d")
    metadata = {
        "package_version": __version__,
        "source_sha256": source_hash.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "asset": asset,
        "source": source,
        "adjustment": adjustment,
        "data_status": "synthetic demonstration"
        if synthetic
        else "user-supplied, unverified market data",
        "data_sha256": hashlib.sha256(csv.encode()).hexdigest(),
        "rows": len(frame),
        "unused_tail_rows": len(frame) - boundaries[-1][1],
        "config": config.to_dict(),
        "forecast_protocol": "recursive fixed-origin; known dates only, no future prices",
        "forecast_baselines": ["persistence", "drift"] + (["arima"] if include_arima else []),
        "selection_objective": "net compounded return; alphabetical strategy tie break",
        "execution": (
            "previous close signal -> next open; scheduled terminal close liquidation each fold"
        ),
        "visual_route": "no-image-generator",
        "complete": False,
    }
    write_json(output / "manifest.json", metadata)
    all_returns = {
        name: [] for name in ("forward_selected", "backward_selected", "buy_hold", "cash")
    }
    all_trade_returns = {name: [] for name in all_returns}
    folds = []
    methods = ["persistence", "drift"] + (["arima"] if include_arima else [])
    for number, (start, end) in enumerate(boundaries, 1):
        if progress:
            progress(
                f"Fold {number}/{len(boundaries)}: training through {frame.index[start - 1].date()}"
            )
        history = frame.iloc[:start].copy()
        # Dates are scheduling information. Actual future prices are accessed only after selection.
        dates = frame.index[start:end]
        fold_config = replace(config, seed=(config.seed + number - 1) % 2**32)
        model = DNNForecaster(fold_config).fit(history)
        predicted = model.predict(dates)
        forward_ranking = rank_strategies(history, predicted, config)
        backward_ranking = rank_strategies(history, None, config)
        chosen = {
            "forward_selected": forward_ranking[0]["strategy"],
            "backward_selected": backward_ranking[0]["strategy"],
        }
        directory = output / f"fold_{number:02d}"
        directory.mkdir()
        predicted.to_csv(directory / "forecast.csv", float_format="%.17g")
        write_json(
            directory / "selection.json", {"forward": forward_ranking, "backward": backward_ranking}
        )
        write_json(directory / "training.json", model.diagnostics)
        model.save(directory / "weights")
        # Evaluation boundary: observations become visible after the decision has been frozen.
        actual = frame.iloc[start:end]
        observed = frame.iloc[:end]
        observed_targets = targets(observed)
        evaluations = {}
        for label in all_returns:
            if label in chosen:
                target = observed_targets[chosen[label]]
            else:
                target = pd.Series(int(label == "buy_hold"), index=observed.index)
            result = simulate(observed, target, start, config)
            result.daily.to_csv(directory / f"{label}_daily.csv", float_format="%.17g")
            result.orders.to_csv(directory / f"{label}_orders.csv", index=False)
            result.trades.to_csv(directory / f"{label}_trades.csv", index=False)
            all_returns[label].append(result.daily["return"])
            all_trade_returns[label].extend(result.trades["return"].tolist())
            evaluations[label] = performance(
                result.daily["return"], config.periods_per_year, result.trades["return"]
            )
        forecast_scores = {
            "dnn": {name: forecast_metrics(actual[name], predicted[name]) for name in OHLC}
        }
        for method in methods:
            baseline = baseline_forecast(history, dates, method)
            baseline.to_csv(directory / f"forecast_{method}.csv", float_format="%.17g")
            forecast_scores[method] = {
                name: forecast_metrics(actual[name], baseline[name]) for name in OHLC
            }
        fold = {
            "fold": number,
            "seed": fold_config.seed,
            "train_rows": start,
            "cutoff": str(history.index[-1].date()),
            "test_start": str(dates[0].date()),
            "test_end": str(dates[-1].date()),
            "selected": chosen,
            "repaired_forecast_bars": model.repaired_bars,
            "performance": evaluations,
            "forecast_errors": forecast_scores,
        }
        write_json(directory / "result.json", fold)
        folds.append(fold)
    returns = pd.DataFrame({name: pd.concat(series) for name, series in all_returns.items()})
    returns.index.name = "date"
    returns.to_csv(output / "daily_returns.csv", float_format="%.17g")
    aggregate = {
        name: performance(returns[name], config.periods_per_year, all_trade_returns[name])
        for name in returns
    }
    summary = {"manifest": metadata, "aggregate": aggregate, "folds": folds}
    write_json(output / "summary.json", summary)
    from .report import render_report

    render_report(summary, returns, output)
    metadata["complete"] = True
    write_json(output / "manifest.json", metadata)
    write_json(output / "summary.json", summary)
    return summary
