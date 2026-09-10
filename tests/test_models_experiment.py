import json

import numpy as np
import pandas as pd
import pytest
import torch

from dnn_forward_lab.cli import main
from dnn_forward_lab.data import OHLC
from dnn_forward_lab.experiment import fold_boundaries, rank_strategies, run_experiment
from dnn_forward_lab.models import DNNForecaster, baseline_forecast, lagged


def test_lag_windows_never_contain_target():
    x, y = lagged(np.arange(8), 3)
    np.testing.assert_array_equal(x[0], [0, 1, 2])
    np.testing.assert_array_equal(y[:, 0], [3, 4, 5, 6, 7])


def test_fit_scaler_excludes_validation(prices, fast_config):
    history = prices.iloc[:150].copy()
    fit_end = len(history) - fast_config.validation_size
    history.iloc[fit_end:, :4] *= 10
    model = DNNForecaster(fast_config).fit(history)
    assert model.scales["close"].minimum == history.close.iloc[:fit_end].min()
    assert model.scales["close"].width == np.ptp(history.close.iloc[:fit_end].to_numpy())
    assert model.diagnostics["fit_end"] < model.diagnostics["validation_start"]


def test_repeatability_and_rng_isolation(prices, fast_config):
    torch.manual_seed(456)
    rng_before = torch.get_rng_state().clone()
    first = DNNForecaster(fast_config).fit(prices.iloc[:150])
    assert torch.equal(rng_before, torch.get_rng_state())
    second = DNNForecaster(fast_config).fit(prices.iloc[:150])
    predicted = first.predict(prices.index[150:160])
    pd.testing.assert_frame_equal(predicted, second.predict(prices.index[150:160]))
    assert (predicted.low <= predicted[["open", "close"]].min(axis=1)).all()
    assert (predicted.high >= predicted[["open", "close"]].max(axis=1)).all()


def test_future_mutation_cannot_change_prediction_or_selection(prices, fast_config):
    altered = prices.copy()
    altered.iloc[150:, :4] *= 20
    outputs = []
    for frame in [prices, altered]:
        history = frame.iloc[:150]
        model = DNNForecaster(fast_config).fit(history)
        forecast = model.predict(frame.index[150:160])
        outputs.append((forecast, rank_strategies(history, forecast, fast_config)))
    pd.testing.assert_frame_equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1] == outputs[1][1]


def test_recursive_forecast_prefix_is_horizon_independent(prices, fast_config):
    model = DNNForecaster(fast_config).fit(prices.iloc[:150])
    short = model.predict(prices.index[150:155])
    long = model.predict(prices.index[150:170])
    pd.testing.assert_frame_equal(short, long.iloc[:5])


def test_forecaster_rejects_observed_dates(prices, fast_config):
    model = DNNForecaster(fast_config)
    with pytest.raises(ValueError, match="Fit"):
        model.predict(prices.index[:5])
    model.fit(prices.iloc[:150])
    with pytest.raises(ValueError, match="follow"):
        model.predict(prices.index[140:150])


def test_baselines_preserve_last_prices_or_log_drift(prices):
    history, dates = prices.iloc[:150], prices.index[150:160]
    fixed = baseline_forecast(history, dates, "persistence")
    np.testing.assert_allclose(fixed.iloc[0], history[OHLC].iloc[-1])
    drift = baseline_forecast(history, dates, "drift")
    expected = history.close.iloc[-1] * np.exp(
        np.log(history.close.iloc[-1] / history.close.iloc[0]) / 149
    )
    assert drift.close.iloc[0] == pytest.approx(expected)


def test_fold_boundaries_are_disjoint(fast_config):
    assert fold_boundaries(200, fast_config) == [(150, 160), (160, 170)]
    with pytest.raises(ValueError, match="170"):
        fold_boundaries(160, fast_config)


def test_end_to_end_report_and_audit_files(prices, fast_config, tmp_path):
    output = tmp_path / "run"
    summary = run_experiment(
        prices,
        fast_config,
        output,
        asset="<script>bad()</script>",
        source="synthetic test fixture",
        adjustment="none",
        synthetic=True,
    )
    assert summary["manifest"]["complete"]
    assert summary["manifest"]["unused_tail_rows"] == 40
    assert len(summary["folds"]) == 2
    returns = pd.read_csv(output / "daily_returns.csv")
    assert len(returns) == 20 and returns.date.is_unique
    for name in summary["aggregate"]:
        assert summary["aggregate"][name]["total_return"] == pytest.approx(
            np.prod(1 + returns[name]) - 1
        )
    assert (output / "fold_01" / "weights" / "close.npz").exists()
    assert (output / "performance.svg").stat().st_size > 1000
    report = (output / "report.html").read_text()
    assert "<script>bad()" not in report and "&lt;script&gt;" in report
    assert "SYNTHETIC DEMONSTRATION" in report and "data:font/woff2" in report
    assert json.loads((output / "manifest.json").read_text())["complete"]
    with pytest.raises(ValueError, match="empty"):
        run_experiment(prices, fast_config, output, asset="X", source="Y", adjustment="Z")


def test_cli_rejects_invalid_file(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "absent.csv")]) == 2
    assert "Error:" in capsys.readouterr().err


def test_arima_optional_baseline_smoke(prices):
    pytest.importorskip("statsmodels")
    # Convergence warnings are evidence; keep this smoke on a stable integrated path.
    rng = np.random.default_rng(10)
    history = prices.iloc[:150][OHLC].copy()
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 150)))
    for name in OHLC:
        history[name] = close
    forecast = baseline_forecast(history, prices.index[150:160], "arima")
    assert forecast.shape == (10, 4) and np.isfinite(forecast).all().all()


def test_model_snapshot_roundtrip_and_corrupt_tensor(prices, fast_config, tmp_path):
    model = DNNForecaster(fast_config).fit(prices.iloc[:150])
    model.save(tmp_path)
    restored = DNNForecaster.load(tmp_path)
    np.testing.assert_allclose(
        model.predict(prices.index[150:160]),
        restored.predict(prices.index[150:160]),
        rtol=1e-6,
    )
    np.savez(tmp_path / "open.npz", wrong_tensor=np.zeros(2))
    with pytest.raises(ValueError, match="tensor names"):
        DNNForecaster.load(tmp_path)


def test_cli_validate_and_cluster_success(prices, tmp_path, capsys):
    path = tmp_path / "input.csv"
    prices.to_csv(path)
    assert main(["validate", str(path)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    cluster = tmp_path / "cluster.json"
    assert (
        main(
            [
                "cluster",
                "--asset",
                f"A={path}",
                "--cutoff",
                "2020-01-01",
                "--clusters",
                "1",
                "--output",
                str(cluster),
            ]
        )
        == 0
    )
    assert json.loads(cluster.read_text())["assets"][0]["cluster"] == 0
    assert (
        main(
            [
                "cluster",
                "--asset",
                "invalid",
                "--cutoff",
                "2020-01-01",
                "--output",
                str(tmp_path / "bad.json"),
            ]
        )
        == 2
    )


def test_cli_demo_uses_config_and_writes_complete_report(tmp_path, capsys):
    config = tmp_path / "small.toml"
    config.write_text("min_train=130\nhorizon=2\nfolds=1\nepochs=1\nbatch_size=128\n")
    output = tmp_path / "demo"
    assert main(["demo", "--config", str(config), "--output", str(output)]) == 0
    assert "SYNTHETIC demonstration" in capsys.readouterr().out
    assert json.loads((output / "manifest.json").read_text())["complete"]


def test_cli_run_requires_explicit_provenance_and_handles_valid_csv(prices, tmp_path):
    config = tmp_path / "small.toml"
    config.write_text("min_train=130\nhorizon=2\nfolds=1\nepochs=1\nbatch_size=128\n")
    path = tmp_path / "input.csv"
    prices.to_csv(path)
    output = tmp_path / "run"
    assert (
        main(
            [
                "run",
                "--csv",
                str(path),
                "--asset",
                "TEST",
                "--source",
                "fixture",
                "--adjustment",
                "none",
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads((output / "manifest.json").read_text())["data_status"].startswith(
        "user-supplied"
    )
