# Contributing

Create a focused branch, explain the scientific or software problem, and include evidence.

```bash
uv sync --frozen --all-extras
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen --all-extras pytest --cov=dnn_forward_lab --cov-fail-under=85
uv build
```

For restricted containers that cannot inspect physical cores, set `LOKY_MAX_CPU_COUNT=2`. For read-only home directories, point `MPLCONFIGDIR` and `UV_CACHE_DIR` at writable scratch directories. NumPy is capped below 2.5 because the supported pandas/statsmodels versions still emit incompatible deprecations under the strict test gate.

Changes to data, splitting, forecasts, indicators, selection, execution or metrics require meaningful tests. Protect chronological separation, next-open execution, commission and slippage accounting, starting-capital drawdown, and reproducibility. Prefer mutation tests that modify future observations and verify unchanged earlier decisions.

Do not tune defaults to improve the checked-in synthetic sample. Any new strategy parameter or candidate changes the research search space; document it and report adverse as well as favorable results. Do not label a favorable backtest as statistically significant without a declared inference method and suitable data.

Generated run directories should be new/empty. A failed run keeps an incomplete manifest for debugging. Never commit user data, credentials, machine-specific paths, executable pickle checkpoints or copyrighted paper copies. The provided sample run may be refreshed only from the documented synthetic generator and config; state environment and source changes.

Font notices must remain in distributions. Keep all chart values tied to data, retain textual units and disclosure labels, and test reports at desktop and mobile sizes.
