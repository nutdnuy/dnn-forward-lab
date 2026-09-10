# Methodology and research contract

This project implements the central hypothesis of **DNN-ForwardTesting**: select an indicator-based strategy on a model-generated future path and compare that decision with a strategy selected on the recent past. This is a research implementation, not a reproduction of the published numerical results.

## Time boundaries

For each fold, `cutoff` is the last observed session. The next `horizon` rows are the observed test window. Folds expand from `min_train` and advance by exactly `horizon`, so test windows do not overlap. Any trailing rows beyond the configured number of folds are excluded and counted in the manifest.

1. Split the observed history into a fitting prefix and its last `validation_size` bars.
2. Fit each price scaler on the fitting prefix only. No clipping to a training range is applied.
3. Train on lag windows whose **targets** fall in the fitting prefix. Input windows contain only earlier observations. There are no multi-period overlapping labels requiring a separate purge gap.
4. At each epoch, recursively forecast the historical validation tail from the fitting prefix's final lag window. Use the validation tail only to calculate MAE and choose the epoch. No validation observation enters the recursive validation path or gradient updates. Keep the best weights; do not refit on validation data.
5. Seed the final forecast from the last observed prices at the cutoff, which may include the now-observed validation tail. Generate the entire future horizon recursively, feeding forecasts back into the network. The model cannot receive actual test prices.
6. Rank the fixed strategy universe twice: on the predicted future and on the last `horizon` historical bars. Historical indicator warm-up is included in both cases. Each ranking simulation starts in cash and uses the same execution and cost rules.
7. Freeze both selected strategy names, then simulate each on observed test prices. Future observations legitimately update each fixed strategy's indicators as time passes; they never change the already selected strategy within that fold.

The first trade in a predicted or observed window can only use the last **observed** close signal. A signal from predicted day 1 could influence a predicted day 2 fill, but cannot justify an earlier trade. This deliberately conservative alignment matters at a 30-bar horizon.

## Neural architecture

Each asset has four separate networks: open, high, low and close. Each has `t -> 10t -> 5t -> 1` neurons, ReLU hidden activations, hidden-layer dropout, linear output, Adam and L1 loss. With `t=5`, this is `5 -> 50 -> 25 -> 1`. Volume is validated and retained but is not a model feature.

Defaults use `dropout=0.002`, a literal interpretation of the paper's **0.2%**. The common machine-learning value `0.2` would mean **20%** and is not silently substituted. Batch size, epochs, patience, learning rate, validation size, seed and costs are explicit configuration values. Demo settings prioritize runtime and are not an optimized model.

Networks train on CPU with a scoped seed and one training thread. Independent channel seeds derive from the fold seed. Full reproducibility across different PyTorch versions/platforms is not promised; see [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html). The lockfile and run manifest record the environment. This package does not search hyperparameters on the test windows.

## Forecast candle constraints

Independent price forecasts may violate OHLC geometry. Before simulation, clamp forecast prices to a strictly positive floor (`1e-8`), preserve the positive open and close predictions, raise high to the envelope maximum, and lower low to the envelope minimum. Feed these repaired values back into subsequent forecast steps. Every fold reports how many forecast bars were repaired.

This projection is an explicit engineering extension. A high repair count or implausible price level is a model failure diagnostic, not a reason to trust a profitable simulated path. Historical input candles are **rejected**, never repaired. Recursive epoch validation operates separately per channel without the joint candle projection; final forecast metrics include projection.

## Candidate strategy universe

All strategies are long/cash, with no short positions or leverage. Rules produce an end-of-day target state. Entry sets target to 1; exit sets target to 0; otherwise the previous state persists. If entry and exit occur together, exit wins. Undefined warm-up values trigger neither event. Parameters are fixed before evaluation.

| Candidate | Entry | Exit |
|---|---|---|
| SMA | Close > SMA(20) | Close < SMA(20) |
| EMA | Close > EMA(20) | Close < EMA(20) |
| MACD | EMA(12)-EMA(26) > EMA(9) of MACD | Below that signal |
| Bollinger | Close < SMA(20)-2 population standard deviations | Close >= SMA(20) |
| Stochastic | %K(14) < 20 | %K(14) > 80 |
| Williams %R | %R(14) < -80 | %R(14) > -20 |
| Momentum | Close / Close[-10] - 1 > 0 | Below 0 |
| RSI | Wilder RSI(5) < 30 | Wilder RSI(5) > 70 |
| ATR breakout | Close > EMA(20)+2 Wilder ATR(14) | Close < EMA(20) |
| Price oscillator | 100(EMA(12)/EMA(26)-1) > 0 | Below 0 |
| TEMA | Figure 22 entry, TEMA(10) for each OHLC field | Figure 22 exit |
| ADX | +DI > -DI and ADX(14) > 25 | -DI > +DI and ADX(14) > 25 |
| ST+MO+MACD | Stochastic entry AND positive momentum AND MACD above signal | Any constituent exit |
| PO+W%R | Positive PO AND Williams entry | Negative PO OR Williams exit |
| PO+RSI | Positive PO AND RSI entry | Negative PO OR RSI exit |

Figure 22 TEMA rules, in project notation:

```text
entry = ((low < TEMA(low)) OR (high < TEMA(high)))
        AND ((close < TEMA(close)) OR (open < TEMA(open)))
exit  = ((low > TEMA(low)) OR (high > TEMA(high)))
        AND ((close > TEMA(close)) OR (open > TEMA(open)))
```

These TEMA rules are intentionally mean-reversion-like, following the paper's displayed logic. The paper's ADX prose says `<25` for one sell condition while Figure 22 says `>25`; this implementation follows **Figure 22**. TEMA period 10 and most other rules/parameters are explicit project assumptions. The paper names 12 indicators while describing a set of ten. We implement all 12 names plus the three named combinations; naming does not imply their original undisclosed code has been recovered.

EMA uses pandas `ewm(adjust=False, min_periods=period)`, seeded from its first observation. Wilder averages start at the first complete arithmetic mean, then use alpha `1/period`. Flat RSI is 50, zero-loss positive-gain RSI is 100, flat ADX is 0, and flat-range stochastic/Williams values remain undefined.

## Execution and accounting

- All-in fractional-share buys execute at the **next session's open** after the signal. Cash never becomes negative.
- Buy fill = open × (1 + slippage); shares = cash / [fill × (1 + commission)].
- Sell fill = open × (1 - slippage); cash = shares × fill × (1 - commission).
- Commission and slippage are specified in basis points **per side**; one basis point is 0.0001.
- Daily equity marks shares at the close. Existing positions capture overnight gaps until an open exit; no position captures a gap before its open entry.
- A predetermined market-on-close liquidation ends each fold, with the same fees/slippage. This scheduled action does not depend on that final close's indicator. A buy at the last open may therefore be liquidated at the same day's close.
- Buy-and-hold resets and liquidates each fold, matching the experimental trading windows. It is not an uninterrupted full-sample buy-and-hold benchmark.
- Each fold's raw cash ledger starts at `initial_cash`; aggregate returns compound fold returns. Absolute per-fold trade P&L is local to that fold, while trade-return averages are dimensionless.
- Taxes, volume constraints, spread beyond configured slippage, market impact, borrow fees, financing, cash interest, lot sizes, dividends paid separately and settlement rules are not modeled.

For adjusted OHLC, state exactly what the provider adjusted. Do not mix adjusted close with unadjusted open/high/low. Split adjustments and total-return adjustments have different interpretations; use consistent series and validate corporate actions outside the package.

## Metrics

All strategy metrics use daily **net** returns, `r`. Return units are fractions in JSON/CSV; report percentages are explicitly labeled.

| Metric | Definition |
|---|---|
| Total return | product(1+r) - 1 |
| Annualized return | exp(sum(log(1+r)) × P/N) - 1 |
| Max drawdown | Largest relative decline from running peak, including initial capital; never annualized |
| Sharpe | mean(r) / sample_std(r) × sqrt(P), zero risk-free assumption |
| Sortino | mean(r) / sqrt(mean(min(r,0)^2)) × sqrt(P), zero minimum acceptable return |
| Calmar | Annualized return / max drawdown |
| Expectancy ratio | Mean net round-trip return / absolute mean losing round-trip return |
| MAE | mean(abs(actual - forecast)) in input price units |
| MSE / RMSE | mean(error²) / sqrt(mean(error²)) |
| MAPE | 100 × mean(abs(error / actual)) |
| Explained variance | 1 - Var(error) / Var(actual), population variances |

`P` defaults to 252. Undefined ratios are JSON `null` and displayed as **Undefined**, never fabricated as zero or infinity. Calmar is null if the annualized return overflows. Annualized ratios on short samples are unstable and are not statistical significance tests. No confidence interval, p-value, or superiority claim is inferred from this demonstration.

## Volatility and clustering

Rolling Parkinson, Garman–Klass, Rogers–Satchell and Yang–Zhang estimators use conventional log-ratio formulas, a 20-observation window and annualization by sqrt(252). Yang–Zhang includes sample overnight/open-close variances, Rogers–Satchell variance and `k = 0.34 / [1.34 + (n+1)/(n-1)]`. It therefore needs a previous close in addition to the current rolling window. Tiny negative numerical variances are clipped to zero before square root.

The separate `cluster` command intersects assets' dates at or before a required cutoff, averages each of the four estimators per asset, standardizes those four features and runs k-means++ with 20 initializations. Labels are reordered from lowest to highest mean observed volatility. The number of clusters cannot exceed the asset count or the distinct feature-vector count. The command does not silently feed full-sample clusters into trading evaluation.

This **cross-asset summary clustering is an extension**, distinct from the paper's clustering of volatility observations with k=6. It does not prove two assets are independent or establish a universal definition of medium volatility. Selection across asset universes can create survivorship and selection bias; choose the cutoff and universe prospectively.

## Baselines and omissions

Persistence and historical log drift use only the observed history. Optional ARIMA is a **fixed (1,1,1) model on log prices**, not the paper's Auto-ARIMA search. Prophet, DTW/Pearson analysis, the original ten-asset dataset, original trained weights, tuning grid and real-time execution are not included. The unavailable data/model artifacts and ambiguous original specifications prevent exact reproduction.

See [SOURCE_MAP.md](SOURCE_MAP.md) for a page-by-page evidence map and [DATA.md](DATA.md) for input/provenance rules.
