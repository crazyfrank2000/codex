"""
SOXX RSI Backtest
- 回测区间: 近5年
- 分析 RSI 超买区间(可配置阈值), 统计每次超买持续时长、随后回调幅度和交易日数
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
from matplotlib import rcParams
from datetime import datetime, timedelta


# ── 参数 ──────────────────────────────────────────────────────────────────────
TICKER         = "SOXX"
RSI_PERIOD     = 14
RSI_THRESH     = 75        # 主分析阈值 (近5年 RSI>85 仅1天, 用75更有统计意义)
RSI_THRESH_EX  = 80        # 极端超买辅助阈值
PULLBACK_DAYS  = 126       # 回调搜索窗口(约6个月交易日)
END_DATE       = datetime.today()
START_DATE     = END_DATE - timedelta(days=5 * 365)

# 用英文标签避免中文字体问题
LABEL = {
    "title":      f"{TICKER} RSI>{RSI_THRESH} Overbought Periods & Pullback Analysis (5Y)",
    "price":      "Close Price (USD)",
    "rsi":        "RSI(14)",
    "ob_zone":    f"RSI>{RSI_THRESH} Zone",
    "ex_zone":    f"RSI>{RSI_THRESH_EX} Zone",
    "close":      "Close Price",
}


# ── RSI 计算 ──────────────────────────────────────────────────────────────────
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── 数据下载 ──────────────────────────────────────────────────────────────────
print(f"Downloading {TICKER}  ({START_DATE.date()} -> {END_DATE.date()}) ...")
df = yf.download(
    TICKER,
    start=START_DATE.strftime("%Y-%m-%d"),
    end=END_DATE.strftime("%Y-%m-%d"),
    auto_adjust=True,
    progress=False,
)

if df.empty:
    raise RuntimeError("Download failed. Check ticker or network.")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

df["RSI"] = calc_rsi(df["Close"], RSI_PERIOD)
print(f"Rows: {len(df)}  |  Range: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"RSI max={df['RSI'].max():.2f}  mean={df['RSI'].mean():.2f}")
print(f"  Days RSI>70: {(df['RSI']>70).sum()}")
print(f"  Days RSI>75: {(df['RSI']>75).sum()}")
print(f"  Days RSI>80: {(df['RSI']>80).sum()}")
print(f"  Days RSI>85: {(df['RSI']>85).sum()}")
print()


# ── 识别连续超买区间 ──────────────────────────────────────────────────────────
def find_segments(series: pd.Series, threshold: float):
    """Return list of (start_dt, end_dt, start_idx, end_idx) for consecutive runs above threshold."""
    above = series > threshold
    segs  = []
    in_s  = False
    s_dt  = None
    s_i   = None
    for i, (dt, val) in enumerate(above.items()):
        if val and not in_s:
            in_s, s_dt, s_i = True, dt, i
        elif not val and in_s:
            in_s = False
            segs.append((s_dt, series.index[i - 1], s_i, i - 1))
    if in_s:
        segs.append((s_dt, series.index[-1], s_i, len(series) - 1))
    return segs


segments    = find_segments(df["RSI"], RSI_THRESH)
segments_ex = find_segments(df["RSI"], RSI_THRESH_EX)

print(f"RSI > {RSI_THRESH}  segments: {len(segments)}")
print(f"RSI > {RSI_THRESH_EX} segments: {len(segments_ex)}")


# ── 统计回调 ──────────────────────────────────────────────────────────────────
def calc_pullback_stats(segs, df_in, pullback_days):
    records = []
    for seg_start, seg_end, si, ei in segs:
        seg_dur     = ei - si + 1
        seg_slice   = df_in["Close"].iloc[si:ei + 1]
        peak_val    = float(seg_slice.max())
        peak_date   = seg_slice.idxmax()

        post_start = ei + 1
        post_end   = min(ei + pullback_days, len(df_in) - 1)

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

        drawdown  = (trough_val - peak_val) / peak_val * 100
        pb_days   = trough_idx - ei

        records.append(dict(
            seg_start=seg_start.date(), seg_end=seg_end.date(),
            seg_days=seg_dur, peak=round(peak_val, 2), peak_date=peak_date.date(),
            trough=round(trough_val, 2), trough_date=trough_date.date(),
            drawdown_pct=round(drawdown, 2), pullback_days=pb_days,
        ))
    return pd.DataFrame(records)


result_df    = calc_pullback_stats(segments,    df, PULLBACK_DAYS)
result_ex_df = calc_pullback_stats(segments_ex, df, PULLBACK_DAYS)


# ── 打印结果 ──────────────────────────────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.2f}".format)

DIVIDER = "=" * 115

def print_table(rdf, thresh):
    print(DIVIDER)
    print(f"  {TICKER}  RSI > {thresh}  Overbought Segments + Pullback Statistics  (5Y Backtest)")
    print(DIVIDER)
    rename = {
        "seg_start": "OB Start", "seg_end": "OB End",
        "seg_days": "OB Days", "peak": "Peak ($)",
        "peak_date": "Peak Date", "trough": "Trough ($)",
        "trough_date": "Trough Date", "drawdown_pct": "Drawdown%",
        "pullback_days": "Pullback Days",
    }
    print(rdf.rename(columns=rename).to_string(index=False))
    print(DIVIDER)
    valid = rdf.dropna(subset=["drawdown_pct"])
    if not valid.empty:
        print(f"\n  [Summary]")
        print(f"    Avg OB duration  : {valid['seg_days'].mean():.1f} trading days")
        print(f"    Avg drawdown     : {valid['drawdown_pct'].mean():.2f}%")
        print(f"    Max drawdown     : {valid['drawdown_pct'].min():.2f}%  (ended {valid.loc[valid['drawdown_pct'].idxmin(),'seg_end']})")
        print(f"    Min drawdown     : {valid['drawdown_pct'].max():.2f}%  (ended {valid.loc[valid['drawdown_pct'].idxmax(),'seg_end']})")
        print(f"    Avg pullback len : {valid['pullback_days'].mean():.1f} trading days")
    print()

print_table(result_df,    RSI_THRESH)
print_table(result_ex_df, RSI_THRESH_EX)


# ── 可视化 ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(17, 10), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.2]}
)
fig.suptitle(LABEL["title"], fontsize=13, fontweight="bold")

# 上图: 价格
ax1.plot(df.index, df["Close"], color="steelblue", linewidth=1.2, label=LABEL["close"])

for seg_start, seg_end, si, ei in segments:
    ax1.axvspan(seg_start, seg_end, alpha=0.18, color="orange")

for seg_start, seg_end, si, ei in segments_ex:
    ax1.axvspan(seg_start, seg_end, alpha=0.35, color="red")

# 标记回调低点
valid_rows = result_df.dropna(subset=["drawdown_pct"])
for _, row in valid_rows.iterrows():
    ax1.annotate(
        f"{row['drawdown_pct']:.1f}%",
        xy=(pd.Timestamp(row["trough_date"]), row["trough"]),
        xytext=(0, -32), textcoords="offset points",
        fontsize=7.5, color="darkred", ha="center",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
    )

ax1.set_ylabel(LABEL["price"], fontsize=10)
ax1.grid(alpha=0.3)
legend_elems = [
    plt.Line2D([0], [0], color="steelblue", lw=1.2, label=LABEL["close"]),
    mpatches.Patch(facecolor="orange", alpha=0.4, label=LABEL["ob_zone"]),
    mpatches.Patch(facecolor="red",    alpha=0.5, label=LABEL["ex_zone"]),
]
ax1.legend(handles=legend_elems, loc="upper left", fontsize=9)

# 下图: RSI
ax2.plot(df.index, df["RSI"], color="purple", linewidth=1, label="RSI(14)")
for thresh, col, ls in [
    (RSI_THRESH,    "orange", "--"),
    (RSI_THRESH_EX, "red",    "--"),
    (70,            "gold",   ":"),
    (30,            "green",  ":"),
]:
    ax2.axhline(thresh, color=col, linestyle=ls, linewidth=0.9,
                label=f"RSI={thresh}")

ax2.fill_between(df.index, df["RSI"], RSI_THRESH,
                 where=(df["RSI"] > RSI_THRESH), alpha=0.2, color="orange")
ax2.fill_between(df.index, df["RSI"], RSI_THRESH_EX,
                 where=(df["RSI"] > RSI_THRESH_EX), alpha=0.3, color="red")

ax2.set_ylabel(LABEL["rsi"], fontsize=10)
ax2.set_ylim(0, 100)
ax2.legend(loc="upper left", fontsize=8, ncol=2)
ax2.grid(alpha=0.3)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
out_png = "/home/user/codex/soxx_rsi_backtest.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Chart saved: {out_png}")

# CSV
csv_path = "/home/user/codex/soxx_rsi_backtest.csv"
result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
csv_ex_path = "/home/user/codex/soxx_rsi80_backtest.csv"
result_ex_df.to_csv(csv_ex_path, index=False, encoding="utf-8-sig")
print(f"CSV  saved: {csv_path}")
print(f"CSV  saved: {csv_ex_path}")
