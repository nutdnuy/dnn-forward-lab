"""Explicit cash/share ledger: close signal, next open execution, terminal close liquidation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data import validate_ohlc


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame


def simulate(frame: pd.DataFrame, target: pd.Series, start: int, config: Config) -> BacktestResult:
    frame = validate_ohlc(frame)
    if not 1 <= start < len(frame):
        raise ValueError("start must leave a preceding signal bar and at least one evaluation bar")
    if not target.index.equals(frame.index) or not target.isin([0, 1]).all():
        raise ValueError("Targets must align exactly with OHLC and contain only 0/1")
    cash, shares = config.initial_cash, 0.0
    fee, slip = config.commission_bps / 10000, config.slippage_bps / 10000
    previous_equity = config.initial_cash
    orders, trades, daily = [], [], []
    entry_date, invested = None, None

    def order(side, raw_price, date, reason, signal_date):
        nonlocal cash, shares, entry_date, invested
        fill = raw_price * (1 + slip if side == "buy" else 1 - slip)
        if side == "buy":
            invested, entry_date = cash, date
            shares = cash / (fill * (1 + fee))
            quantity = shares
            commission = shares * fill * fee
            cash = 0.0
        else:
            quantity = shares
            commission = shares * fill * fee
            cash = shares * fill - commission
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_equity": invested,
                    "exit_equity": cash,
                    "pnl": cash - invested,
                    "return": cash / invested - 1,
                }
            )
            shares = 0.0
        orders.append(
            {
                "date": date,
                "signal_date": signal_date,
                "side": side,
                "reason": reason,
                "reference_price": raw_price,
                "fill_price": fill,
                "shares": quantity,
                "commission": commission,
                "slippage_cost": abs(fill - raw_price) * quantity,
            }
        )

    for i in range(start, len(frame)):
        row, date = frame.iloc[i], frame.index[i]
        desired = int(target.iloc[i - 1])
        if desired and shares == 0:
            order("buy", row.open, date, "previous_close_signal", frame.index[i - 1])
        elif not desired and shares > 0:
            order("sell", row.open, date, "previous_close_signal", frame.index[i - 1])
        exposed = shares > 0
        if i == len(frame) - 1 and shares > 0:
            order("sell", row.close, date, "scheduled_terminal_close", None)
        equity = cash + shares * row.close
        daily.append(
            {
                "date": date,
                "equity": equity,
                "return": equity / previous_equity - 1,
                "cash": cash,
                "shares": shares,
                "exposed": exposed,
            }
        )
        previous_equity = equity
    daily_frame = pd.DataFrame(daily).set_index("date")
    if not np.isfinite(daily_frame.equity).all() or (daily_frame.equity <= 0).any():
        raise ValueError("Non-finite or non-positive equity; inspect input prices")
    return BacktestResult(
        daily_frame,
        pd.DataFrame(
            orders,
            columns=[
                "date",
                "signal_date",
                "side",
                "reason",
                "reference_price",
                "fill_price",
                "shares",
                "commission",
                "slippage_cost",
            ],
        ),
        pd.DataFrame(
            trades,
            columns=[
                "entry_date",
                "exit_date",
                "entry_equity",
                "exit_equity",
                "pnl",
                "return",
            ],
        ),
    )
