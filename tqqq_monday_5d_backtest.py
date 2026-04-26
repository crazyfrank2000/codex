"""
TQQQ Monday 5D Tactical Backtest
=================================
Account: $25,000  |  Tactical Capital: $10,000  |  R: $250/trade
Entry  : Monday Close
Exit   : Friday Close  OR  Stop Loss ($250 = 2.5% of tactical capital)
Modes  : long_only  |  long_short

WARNING: TQQQ is a 3x daily-reset leveraged ETF.
Multi-day holding incurs volatility decay and path-dependence risk.
This is NOT the original intraday ORB strategy.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL           = "TQQQ"
ACCOUNT_CAPITAL  = 25_000
TACTICAL_CAPITAL = 10_000
R_DOLLAR         = 250          # max loss per trade in USD
COST_BPS         = 0.0          # round-trip cost per notional (0 = no friction)

START_DATE = (date.today() - timedelta(days=5*365 + 30)).strftime("%Y-%m-%d")
END_DATE   = date.today().strftime("%Y-%m-%d")

# ── Download & clean data ─────────────────────────────────────────────────────
print(f"Downloading {SYMBOL} from {START_DATE} to {END_DATE} …")
raw = yf.download(SYMBOL, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

if raw.empty:
    sys.exit("ERROR: No data downloaded. Check ticker or internet connection.")

# Flatten MultiIndex columns if present
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[["Open", "High", "Low", "Close"]].copy()
df.columns = ["open", "high", "low", "close"]
df.index = pd.to_datetime(df.index)
df["weekday"] = df.index.dayofweek          # 0=Mon … 4=Fri
df["iso_week"] = df.index.isocalendar().week.astype(int)
df["iso_year"] = df.index.isocalendar().year.astype(int)
df["year_week"] = df["iso_year"].astype(str) + "-" + df["iso_week"].astype(str).str.zfill(2)

print(f"  Loaded {len(df)} trading days  ({df.index[0].date()} → {df.index[-1].date()})")

# ── Backtest engine ───────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, mode: str = "long_only") -> pd.DataFrame:
    """
    For each Monday:
      - Long Only : Monday close > open  → LONG TQQQ; else skip
      - Long/Short: Monday close > open  → LONG;  close < open → SHORT
    Hold until Friday close or $250 stop loss (checked on subsequent days' intraday).
    """
    assert mode in ("long_only", "long_short"), "mode must be 'long_only' or 'long_short'"

    monday_mask = df["weekday"] == 0
    monday_idx  = df.index[monday_mask]

    trades = []

    for mon_date in monday_idx:
        mon_loc = df.index.get_loc(mon_date)

        # ── Direction signal ────────────────────────────────────────────────
        entry_open  = df["open"].iloc[mon_loc]
        entry_close = df["close"].iloc[mon_loc]

        monday_up   = entry_close > entry_open
        monday_down = entry_close < entry_open

        if mode == "long_only":
            if not monday_up:
                continue
            direction = 1
        else:  # long_short
            if monday_up:
                direction = 1
            elif monday_down:
                direction = -1
            else:
                continue   # doji / exactly flat

        # ── Identify same-week rows (Mon–Fri, max 5 days) ──────────────────
        week_key  = df["year_week"].iloc[mon_loc]
        week_rows = df.index[df["year_week"] == week_key]
        hold_rows = week_rows[week_rows >= mon_date][:5]

        if len(hold_rows) < 2:   # need at least one day after Monday
            continue

        entry_price    = entry_close
        shares         = int(TACTICAL_CAPITAL // entry_price)
        if shares <= 0:
            continue

        actual_notional = shares * entry_price

        # ── Stop price (fixed-dollar R) ────────────────────────────────────
        r_per_share = R_DOLLAR / shares
        if direction == 1:
            stop_price = entry_price - r_per_share   # long stop
        else:
            stop_price = entry_price + r_per_share   # short stop

        # ── Check for stop hit on Tue–Fri of the week ─────────────────────
        exit_price  = df["close"].iloc[df.index.get_loc(hold_rows[-1])]
        exit_date   = hold_rows[-1]
        exit_reason = "time_exit"

        for day in hold_rows[1:]:          # skip Monday itself
            loc = df.index.get_loc(day)
            lo  = df["low"].iloc[loc]
            hi  = df["high"].iloc[loc]

            if direction == 1 and lo <= stop_price:
                exit_price  = stop_price
                exit_date   = day
                exit_reason = "stop_loss"
                break
            if direction == -1 and hi >= stop_price:
                exit_price  = stop_price
                exit_date   = day
                exit_reason = "stop_loss"
                break

        # ── P&L ────────────────────────────────────────────────────────────
        gross_pnl    = direction * shares * (exit_price - entry_price)
        trading_cost = actual_notional * COST_BPS * 2
        net_pnl      = gross_pnl - trading_cost

        trades.append(dict(
            entry_date   = mon_date.date(),
            exit_date    = exit_date.date(),
            direction    = "LONG" if direction == 1 else "SHORT",
            entry_price  = round(entry_price, 4),
            exit_price   = round(exit_price, 4),
            stop_price   = round(stop_price, 4),
            shares       = shares,
            notional     = round(actual_notional, 2),
            gross_pnl    = round(gross_pnl, 2),
            trading_cost = round(trading_cost, 2),
            net_pnl      = round(net_pnl, 2),
            R_multiple   = round(net_pnl / R_DOLLAR, 3),
            exit_reason  = exit_reason,
        ))

    result = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    result["cum_pnl"]         = result["net_pnl"].cumsum()
    result["account_equity"]  = ACCOUNT_CAPITAL  + result["cum_pnl"]
    result["tactical_equity"] = TACTICAL_CAPITAL + result["cum_pnl"]
    return result

# ── Metrics ───────────────────────────────────────────────────────────────────
def max_drawdown(equity_series: pd.Series) -> float:
    roll_max = equity_series.cummax()
    dd       = (equity_series - roll_max) / roll_max
    return float(dd.min())

def calc_metrics(trades: pd.DataFrame, label: str) -> dict:
    pnl = trades["net_pnl"]
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total_pnl       = pnl.sum()
    final_account   = trades["account_equity"].iloc[-1]
    final_tactical  = trades["tactical_equity"].iloc[-1]

    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.nan

    return dict(
        strategy              = label,
        trades                = len(trades),
        total_pnl             = round(total_pnl, 2),
        final_account         = round(final_account, 2),
        final_tactical        = round(final_tactical, 2),
        account_total_return  = round(final_account / ACCOUNT_CAPITAL - 1, 4),
        tactical_total_return = round(final_tactical / TACTICAL_CAPITAL - 1, 4),
        win_rate              = round((pnl > 0).mean(), 4),
        avg_win               = round(wins.mean(), 2)  if len(wins) else np.nan,
        avg_loss              = round(losses.mean(), 2) if len(losses) else np.nan,
        profit_factor         = round(profit_factor, 3),
        max_drawdown_account  = round(max_drawdown(trades["account_equity"]), 4),
        avg_R                 = round(trades["R_multiple"].mean(), 3),
        median_R              = round(trades["R_multiple"].median(), 3),
        stop_rate             = round((trades["exit_reason"] == "stop_loss").mean(), 4),
    )

# ── Run both modes ────────────────────────────────────────────────────────────
print("\nRunning backtest …")
trades_lo = run_backtest(df, mode="long_only")
trades_ls = run_backtest(df, mode="long_short")

metrics_lo = calc_metrics(trades_lo, "Monday-Up → Long TQQQ (Long Only)")
metrics_ls = calc_metrics(trades_ls, "Monday-Up Long / Monday-Down Short TQQQ")

summary = pd.DataFrame([metrics_lo, metrics_ls])

# ── Display ───────────────────────────────────────────────────────────────────
DIVIDER = "═" * 72

print(f"\n{DIVIDER}")
print("  TQQQ MONDAY 5D MOMENTUM BACKTEST  ─  SUMMARY")
print(f"  Account: ${ACCOUNT_CAPITAL:,}  |  Tactical: ${TACTICAL_CAPITAL:,}  |  R: ${R_DOLLAR}")
print(f"  Period : {df.index[0].date()} → {df.index[-1].date()}")
print(DIVIDER)

for _, row in summary.iterrows():
    print(f"\n  Strategy : {row['strategy']}")
    print(f"  {'Trades':<28} {row['trades']}")
    print(f"  {'Total Net PnL':<28} ${row['total_pnl']:>10,.2f}")
    print(f"  {'Final Account Equity':<28} ${row['final_account']:>10,.2f}")
    print(f"  {'Account Total Return':<28} {row['account_total_return']*100:>9.2f}%")
    print(f"  {'Tactical Total Return':<28} {row['tactical_total_return']*100:>9.2f}%")
    print(f"  {'Win Rate':<28} {row['win_rate']*100:>9.2f}%")
    print(f"  {'Avg Win / Avg Loss':<28} ${row['avg_win']:>7,.2f}  /  ${row['avg_loss']:>7,.2f}")
    print(f"  {'Profit Factor':<28} {row['profit_factor']:>9.3f}")
    print(f"  {'Max Drawdown (account)':<28} {row['max_drawdown_account']*100:>9.2f}%")
    print(f"  {'Avg R-multiple':<28} {row['avg_R']:>9.3f}R")
    print(f"  {'Median R-multiple':<28} {row['median_R']:>9.3f}R")
    print(f"  {'Stop-Loss Rate':<28} {row['stop_rate']*100:>9.2f}%")

# ── Benchmark: Buy-and-Hold TQQQ ─────────────────────────────────────────────
first_close = df["close"].iloc[0]
last_close  = df["close"].iloc[-1]
bnh_return  = last_close / first_close - 1

print(f"\n{DIVIDER}")
print(f"  Buy-and-Hold TQQQ benchmark")
print(f"  {'Period start price':<28} ${first_close:>10.2f}")
print(f"  {'Period end price':<28} ${last_close:>10.2f}")
print(f"  {'Total Return':<28} {bnh_return*100:>9.2f}%")
print(f"  (Barchart 5yr summary reference: +132.44%)")
print(DIVIDER)

# ── Stop-loss breakdown ───────────────────────────────────────────────────────
print("\n  Exit Reason Breakdown (Long Only):")
print(trades_lo["exit_reason"].value_counts().to_string())
print("\n  Exit Reason Breakdown (Long/Short):")
print(trades_ls["exit_reason"].value_counts().to_string())

# ── Recent 10 trades ──────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  LAST 10 TRADES  ─  Long Only")
print(DIVIDER)
cols = ["entry_date","exit_date","direction","entry_price","exit_price",
        "shares","net_pnl","R_multiple","exit_reason"]
print(trades_lo[cols].tail(10).to_string(index=False))

print(f"\n{DIVIDER}")
print("  LAST 10 TRADES  ─  Long / Short")
print(DIVIDER)
print(trades_ls[cols].tail(10).to_string(index=False))

# ── Save CSVs ─────────────────────────────────────────────────────────────────
trades_lo.to_csv("TQQQ_monday_5D_long_only_trades.csv",   index=False)
trades_ls.to_csv("TQQQ_monday_5D_long_short_trades.csv",  index=False)
summary.to_csv(  "TQQQ_monday_5D_backtest_summary.csv",   index=False)
print(f"\n  CSVs saved: long_only_trades, long_short_trades, backtest_summary")
print(DIVIDER + "\n")
