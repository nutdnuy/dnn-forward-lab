# Data contract

One CSV contains **one asset**, in strictly increasing daily session order:

```csv
date,open,high,low,close,volume
2021-01-04,100.0,103.0,99.0,102.0,1500000
2021-01-05,102.0,104.0,101.0,103.0,1300000
```

This two-row illustration is too short to train a model. `volume` is optional; all other columns are required. Column names are case-insensitive. The accepted columns are `date`, `open`, `high`, `low`, `close` and optional `volume` only. Extra ticker, adjusted-close or unnamed-index columns must be resolved explicitly upstream.

The loader rejects duplicate/unsorted dates, intraday timestamps, timezones, nonnumeric/missing/infinite values, nonpositive prices, invalid candles and negative volume. It does not sort, forward-fill, backfill or silently adjust real data. It cannot tell whether a missing row represents a holiday, a suspension or a provider omission. Validate trading calendars and corporate actions before importing.

Use ISO `YYYY-MM-DD` **session dates**. The engine counts rows as trading bars, preserving the provider's calendar. The synthetic generator uses weekdays, including some exchange holidays, and is explicitly not a real trading calendar.

## Minimum observations

`min_train + folds × horizon` rows are required. Research configuration requires `504 + 12 × 30 = 864` rows. A longer input is allowed; unused trailing rows are disclosed. Start at your intended historical boundary by creating a correctly bounded CSV before running.

## Provenance

`run` requires an asset label, a source label including snapshot/retrieval context, and a price-adjustment explanation. These values are stored in reports and manifests. Do not include credentials or confidential text in those labels. The package is offline and does not validate provider licenses or data accuracy.

The run output includes a copy of the exact normalized input, canonical CSV SHA-256, configuration, runtime versions and per-fold cutoffs. Treat a run directory as having the same sensitivity and redistribution restrictions as its input data. `data/` and `runs/` are Git-ignored. The checked-in example contains synthetic data only.

The MIT software license does not grant rights to the source paper, third-party market data or external trained models. Fetch data through a provider you are authorized to use; retain its terms and provenance. No downloading or credential access occurs in CLI commands.
