"""Run from the repository root: uv run python examples/research_api.py."""

from dnn_forward_lab.config import Config
from dnn_forward_lab.data import synthetic_ohlc
from dnn_forward_lab.experiment import run_experiment

config = Config.from_toml("configs/demo.toml")
prices = synthetic_ohlc(config.min_train + config.folds * config.horizon, config.seed)

if __name__ == "__main__":
    summary = run_experiment(
        prices,
        config,
        "runs/api-demo",
        asset="SYNTHETIC",
        source="Seeded synthetic generator v1; not market data",
        adjustment="None; synthetic OHLC",
        synthetic=True,
        progress=print,
    )
    print("Report: runs/api-demo/report.html")
    print("Synthetic demonstrations do not establish market profitability.")
