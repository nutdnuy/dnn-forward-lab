"""Command-line entry points with no implicit network access or credential use."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .data import read_csv, synthetic_ohlc


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="DNN Forward Lab: auditable model-based strategy selection"
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("demo", "Run a seeded synthetic demonstration"),
        ("run", "Evaluate a single-asset daily CSV"),
    ):
        command = commands.add_parser(name, help=help_)
        command.add_argument(
            "--output", type=Path, required=True, help="New or empty output directory"
        )
        command.add_argument("--config", type=Path, help="TOML settings; unknown keys fail")
        command.add_argument(
            "--arima",
            action="store_true",
            help="Include optional fixed ARIMA(1,1,1) forecast baseline",
        )
        if name == "run":
            command.add_argument("--csv", type=Path, required=True)
            command.add_argument("--asset", required=True)
            command.add_argument(
                "--source", required=True, help="Data provider and retrieval/snapshot date"
            )
            command.add_argument(
                "--adjustment", required=True, help="How all four OHLC columns were adjusted"
            )
    inspect = commands.add_parser("validate", help="Validate a local OHLC CSV without training")
    inspect.add_argument("csv", type=Path)
    cluster = commands.add_parser(
        "cluster", help="Cluster cross-asset historical volatility before a cutoff"
    )
    cluster.add_argument("--asset", action="append", required=True, metavar="NAME=CSV")
    cluster.add_argument("--cutoff", required=True)
    cluster.add_argument("--clusters", type=int, default=3)
    cluster.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None) -> int:
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "validate":
            frame = read_csv(args.csv)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "rows": len(frame),
                        "start": str(frame.index[0].date()),
                        "end": str(frame.index[-1].date()),
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "cluster":
            from .experiment import write_json
            from .volatility import cluster_assets

            assets = {}
            for item in args.asset:
                if "=" not in item:
                    raise ValueError("--asset must use NAME=CSV")
                name, path = item.split("=", 1)
                if not name.strip() or name in assets:
                    raise ValueError("Asset names must be nonempty and unique")
                assets[name] = read_csv(path)
            if args.output.exists():
                raise ValueError("Output file already exists")
            result = cluster_assets(assets, args.cutoff, args.clusters)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
            print(f"Wrote {args.output}")
            return 0
        from .experiment import run_experiment

        if args.config:
            config = Config.from_toml(args.config)
        elif args.command == "demo":
            config = Config(min_train=360, folds=3, epochs=35, batch_size=32, patience=8)
        else:
            config = Config()
        synthetic = args.command == "demo"
        if synthetic:
            frame = synthetic_ohlc(config.min_train + config.folds * config.horizon, config.seed)
            provenance = {
                "asset": "SYNTHETIC",
                "source": "Seeded synthetic generator v1; not market data",
                "adjustment": "None; synthetic internally consistent OHLC",
            }
        else:
            frame = read_csv(args.csv)
            provenance = {"asset": args.asset, "source": args.source, "adjustment": args.adjustment}
        result = run_experiment(
            frame,
            config,
            args.output,
            synthetic=synthetic,
            include_arima=args.arima,
            progress=print,
            **provenance,
        )
        print(f"Completed {len(result['folds'])} folds. Report: {args.output / 'report.html'}")
        if synthetic:
            print("SYNTHETIC demonstration: results are not evidence of market profitability.")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
