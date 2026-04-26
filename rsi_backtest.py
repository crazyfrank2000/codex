"""
Multi-ticker RSI Overbought Backtest
Tickers : SOXX, QQQ, TQQQ
Period  : 5 years
Output  : per-ticker chart + combined summary CSV
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

# ── global params ──────────────────────────────────────────────────────────────
TICKERS       = ["SOXX", "QQQ", "TQQQ"]
RSI_PERIOD    = 14
RSI_THRESH    = 75   # primary overbought level
RSI_THRESH_EX = 80   # extreme overbought level
PULLBACK_DAYS = 126  # max look-forward window (~6 months)
END_DATE      = datetime.today()
START_DATE    = END_DATE - timedelta(days=5 * 365)

DIVIDER = "=" * 118


# ── helpers ────────────────────────────────────────────────────────────────────
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def find_segments(series: pd.Series, threshold: float):
    above = series > threshold
    segs, in_s, s_dt, s_i = [], False, None, None
    for i, (dt, val) in enumerate(above.items()):
        if val and not in_s:
            in_s, s_dt, s_i = True, dt, i
        elif not val and in_s:
            in_s = False
            segs.append((s_dt, series.index[i - 1], s_i, i - 1))
    if in_s:
        segs.append((s_dt, series.index[-1], s_i, len(series) - 1))
    return segs


def calc_pullback_stats(segs, df_in, pb_days):
    records = []
    for seg_start, seg_end, si, ei in segs:
        seg_dur   = ei - si + 1
        seg_close = df_in["Close"].iloc[si:ei + 1]
        peak_val  = float(seg_close.max())
        peak_date = seg_close.idxmax()

        post_start = ei + 1
        post_end   = min(ei + pb_days, len(df_in) - 1)

        if post_start > len(df_in) - 1:
            records.append(dict(
                seg_start=seg_start.date(), seg_end=seg_end.date(),
                seg_days=seg_dur, peak=round(peak_val, 2), peak_date=peak_date.date(),
                trough=None, trough_date=None, drawdown_pct=None, pullback_days=None,
            ))
            continue

        post_close  = df_in["Close"].iloc[post_start:post_end + 1]
        trough_val  = float(post_close.min())
        trough_date = post_close.idxmin()
        trough_idx  = df_in.index.get_loc(trough_date)

        records.append(dict(
            seg_start=seg_start.date(), seg_end=seg_end.date(),
            seg_days=seg_dur, peak=round(peak_val, 2), peak_date=peak_date.date(),
            trough=round(trough_val, 2), trough_date=trough_date.date(),
            drawdown_pct=round((trough_val - peak_val) / peak_val * 100, 2),
            pullback_days=int(trough_idx - ei),
        ))
    return pd.DataFrame(records)


def print_table(ticker, rdf, thresh):
    print(DIVIDER)
    print(f"  {ticker}  |  RSI > {thresh}  Overbought Segments + Pullback Statistics  (5Y)")
    print(DIVIDER)
    rename = {
        "seg_start": "OB Start", "seg_end": "OB End",
        "seg_days": "OB Days", "peak": "Peak($)",
        "peak_date": "Peak Date", "trough": "Trough($)",
        "trough_date": "Trough Date", "drawdown_pct": "Drawdown%",
        "pullback_days": "Pullback Days",
    }
    print(rdf.rename(columns=rename).to_string(index=False))
    valid = rdf.dropna(subset=["drawdown_pct"])
    if not valid.empty:
        print(f"\n  [Summary]  n={len(valid)}")
        print(f"    Avg OB duration  : {valid['seg_days'].mean():.1f} trading days")
        print(f"    Avg drawdown     : {valid['drawdown_pct'].mean():.2f}%")
        print(f"    Max drawdown     : {valid['drawdown_pct'].min():.2f}%"
              f"  (ended {valid.loc[valid['drawdown_pct'].idxmin(), 'seg_end']})")
        print(f"    Min drawdown     : {valid['drawdown_pct'].max():.2f}%"
              f"  (ended {valid.loc[valid['drawdown_pct'].idxmax(), 'seg_end']})")
        print(f"    Avg pullback len : {valid['pullback_days'].mean():.1f} trading days")
    print()


def plot_ticker(ticker, df, segs, segs_ex, result_df, out_path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(17, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]}
    )
    fig.suptitle(
        f"{ticker} — RSI>{RSI_THRESH}/{RSI_THRESH_EX} Overbought & Pullback (5Y)",
        fontsize=13, fontweight="bold"
    )

    colors = {"SOXX": "steelblue", "QQQ": "darkorange", "TQQQ": "green"}
    lc = colors.get(ticker, "steelblue")

    ax1.plot(df.index, df["Close"], color=lc, linewidth=1.2, label="Close")
    for s, e, si, ei in segs:
        ax1.axvspan(s, e, alpha=0.18, color="orange")
    for s, e, si, ei in segs_ex:
        ax1.axvspan(s, e, alpha=0.38, color="red")

    valid_rows = result_df.dropna(subset=["drawdown_pct"])
    for _, row in valid_rows.iterrows():
        ax1.annotate(
            f"{row['drawdown_pct']:.1f}%",
            xy=(pd.Timestamp(row["trough_date"]), row["trough"]),
            xytext=(0, -30), textcoords="offset points",
            fontsize=7, color="darkred", ha="center",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
        )

    ax1.set_ylabel("Close Price (USD)", fontsize=10)
    ax1.grid(alpha=0.3)
    legend_elems = [
        plt.Line2D([0], [0], color=lc, lw=1.2, label="Close"),
        mpatches.Patch(facecolor="orange", alpha=0.4, label=f"RSI>{RSI_THRESH}"),
        mpatches.Patch(facecolor="red",    alpha=0.5, label=f"RSI>{RSI_THRESH_EX}"),
    ]
    ax1.legend(handles=legend_elems, loc="upper left", fontsize=9)

    ax2.plot(df.index, df["RSI"], color="purple", linewidth=1, label="RSI(14)")
    for thresh, col, ls in [
        (RSI_THRESH,    "orange", "--"),
        (RSI_THRESH_EX, "red",    "--"),
        (70,            "gold",   ":"),
        (30,            "green",  ":"),
    ]:
        ax2.axhline(thresh, color=col, linestyle=ls, linewidth=0.9, label=f"RSI={thresh}")

    ax2.fill_between(df.index, df["RSI"], RSI_THRESH,
                     where=(df["RSI"] > RSI_THRESH), alpha=0.2, color="orange")
    ax2.fill_between(df.index, df["RSI"], RSI_THRESH_EX,
                     where=(df["RSI"] > RSI_THRESH_EX), alpha=0.3, color="red")

    ax2.set_ylabel("RSI(14)", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart -> {out_path}")


# ── main loop ──────────────────────────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.2f}".format)

all_summaries = []

for ticker in TICKERS:
    print(f"\n{'─'*60}")
    print(f"  Downloading {ticker}  ({START_DATE.date()} -> {END_DATE.date()})")
    print(f"{'─'*60}")

    raw = yf.download(
        ticker,
        start=START_DATE.strftime("%Y-%m-%d"),
        end=END_DATE.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
    )
    if raw.empty:
        print(f"  [WARN] No data for {ticker}, skipping.")
        continue

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df["RSI"] = calc_rsi(df["Close"], RSI_PERIOD)

    print(f"  Rows: {len(df)}  |  {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"  RSI  max={df['RSI'].max():.2f}  mean={df['RSI'].mean():.2f}  "
          f"[>70:{(df['RSI']>70).sum()}  >75:{(df['RSI']>75).sum()}  "
          f">80:{(df['RSI']>80).sum()}  >85:{(df['RSI']>85).sum()}]")

    segs    = find_segments(df["RSI"], RSI_THRESH)
    segs_ex = find_segments(df["RSI"], RSI_THRESH_EX)
    print(f"  Segments  RSI>{RSI_THRESH}: {len(segs)}   RSI>{RSI_THRESH_EX}: {len(segs_ex)}")

    res    = calc_pullback_stats(segs,    df, PULLBACK_DAYS)
    res_ex = calc_pullback_stats(segs_ex, df, PULLBACK_DAYS)

    print()
    print_table(ticker, res,    RSI_THRESH)
    print_table(ticker, res_ex, RSI_THRESH_EX)

    # chart
    out_png = f"/home/user/codex/{ticker.lower()}_rsi_backtest.png"
    plot_ticker(ticker, df, segs, segs_ex, res, out_png)

    # CSV
    res["ticker"]    = ticker
    res_ex["ticker"] = ticker
    res["thresh"]    = RSI_THRESH
    res_ex["thresh"] = RSI_THRESH_EX
    all_summaries.extend([res, res_ex])

    # per-ticker CSV
    res.drop(columns=["ticker","thresh"]).to_csv(
        f"/home/user/codex/{ticker.lower()}_rsi{RSI_THRESH}_backtest.csv",
        index=False, encoding="utf-8-sig"
    )
    res_ex.drop(columns=["ticker","thresh"]).to_csv(
        f"/home/user/codex/{ticker.lower()}_rsi{RSI_THRESH_EX}_backtest.csv",
        index=False, encoding="utf-8-sig"
    )

# combined CSV
combined = pd.concat(all_summaries, ignore_index=True)
combined_path = "/home/user/codex/all_tickers_rsi_backtest.csv"
combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
print(f"\nCombined CSV -> {combined_path}")


# ── cross-ticker comparison chart ─────────────────────────────────────────────
print("\nGenerating comparison chart ...")
fig, axes = plt.subplots(len(TICKERS), 1, figsize=(17, 5 * len(TICKERS)), sharex=False)
if len(TICKERS) == 1:
    axes = [axes]

colors = {"SOXX": "steelblue", "QQQ": "darkorange", "TQQQ": "forestgreen"}

for ax, ticker in zip(axes, TICKERS):
    raw = yf.download(
        ticker,
        start=START_DATE.strftime("%Y-%m-%d"),
        end=END_DATE.strftime("%Y-%m-%d"),
        auto_adjust=True, progress=False,
    )
    if raw.empty:
        continue
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df["RSI"] = calc_rsi(df["Close"], RSI_PERIOD)

    lc = colors.get(ticker, "steelblue")
    ax.plot(df.index, df["RSI"], color=lc, linewidth=1, label=f"{ticker} RSI(14)")
    ax.axhline(RSI_THRESH,    color="orange", linestyle="--", linewidth=0.9, label=f"RSI={RSI_THRESH}")
    ax.axhline(RSI_THRESH_EX, color="red",    linestyle="--", linewidth=0.9, label=f"RSI={RSI_THRESH_EX}")
    ax.axhline(70,            color="gold",   linestyle=":",  linewidth=0.8)
    ax.fill_between(df.index, df["RSI"], RSI_THRESH,
                    where=(df["RSI"] > RSI_THRESH), alpha=0.2, color="orange")
    ax.fill_between(df.index, df["RSI"], RSI_THRESH_EX,
                    where=(df["RSI"] > RSI_THRESH_EX), alpha=0.35, color="red")
    ax.set_ylabel(f"{ticker} RSI", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

fig.suptitle(
    f"RSI(14) Comparison: SOXX / QQQ / TQQQ  (RSI>{RSI_THRESH} & >{RSI_THRESH_EX} highlighted)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
cmp_path = "/home/user/codex/rsi_comparison.png"
plt.savefig(cmp_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Comparison chart -> {cmp_path}")
print("\nDone.")
