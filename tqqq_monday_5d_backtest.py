"""
TQQQ Monday 5D Tactical Backtest  ─  Long Only，三版过滤条件对比
================================================================
账户: $25,000  |  战术资金: $10,000  |  R: $250/笔
入场: 周一收盘  |  出场: 周五收盘 或 止损 $250（战术资金的2.5%）

版本对比：
  V0  周一 TQQQ 涨                              → 买入 TQQQ（基线）
  V1  周一 TQQQ 涨  ∧  QQQ > MA20             → 买入 TQQQ
  V2  周一 TQQQ 涨  ∧  QQQ > MA20  ∧  QQQ > MA65  → 买入 TQQQ

注：MA 均以 QQQ 当日收盘前的历史数据计算，无未来数据泄漏。
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── 参数 ────────────────────────────────────────────────────────────────────
ACCOUNT_CAPITAL  = 25_000
TACTICAL_CAPITAL = 10_000
R_DOLLAR         = 250
COST_BPS         = 0.0        # 单边手续费（0 = 不计摩擦成本）
MA_SHORT         = 20         # QQQ 短期均线
MA_LONG          = 65         # QQQ 长期均线

# 多取60个交易日预热数据，确保 MA65 第一天就能计算
START_DATA  = (date.today() - timedelta(days=5*365 + 120)).strftime("%Y-%m-%d")
START_BT    = (date.today() - timedelta(days=5*365 + 30)).strftime("%Y-%m-%d")
END_DATE    = date.today().strftime("%Y-%m-%d")

# ── 下载数据 ─────────────────────────────────────────────────────────────────
def download(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=START_DATA, end=END_DATE,
                      auto_adjust=True, progress=False)
    if raw.empty:
        sys.exit(f"ERROR: 无法下载 {ticker} 数据，请检查网络或代码。")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].copy()
    df.columns = ["open", "high", "low", "close"]
    df.index = pd.to_datetime(df.index)
    return df

print(f"下载数据中… ({START_DATA} → {END_DATE})")
tqqq = download("TQQQ")
qqq  = download("QQQ")
print(f"  TQQQ: {len(tqqq)} 交易日  QQQ: {len(qqq)} 交易日")

# ── 计算 QQQ 均线（基于收盘价，当日收盘即可用） ───────────────────────────
qqq["ma20"] = qqq["close"].rolling(MA_SHORT).mean()
qqq["ma65"] = qqq["close"].rolling(MA_LONG).mean()

# ── 合并 TQQQ + QQQ 均线 ────────────────────────────────────────────────────
df = tqqq.copy()
df = df.join(qqq[["close", "ma20", "ma65"]].rename(columns={
    "close": "qqq_close",
    "ma20":  "qqq_ma20",
    "ma65":  "qqq_ma65",
}), how="inner")

# 截掉预热期，从回测起始日开始
df = df[df.index >= START_BT].copy()

df["weekday"]  = df.index.dayofweek
df["iso_week"] = df.index.isocalendar().week.astype(int)
df["iso_year"] = df.index.isocalendar().year.astype(int)
df["year_week"] = (df["iso_year"].astype(str) + "-"
                   + df["iso_week"].astype(str).str.zfill(2))

print(f"  回测区间: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} 交易日)")

# ── 回测引擎 ─────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame, filter_mode: str = "none") -> pd.DataFrame:
    """
    filter_mode:
        'none'   → 仅周一TQQQ上涨
        'ma20'   → 周一TQQQ涨 ∧ QQQ > MA20
        'ma20_65'→ 周一TQQQ涨 ∧ QQQ > MA20 ∧ QQQ > MA65
    """
    monday_dates = df.index[df["weekday"] == 0]
    trades = []

    for mon_date in monday_dates:
        loc = df.index.get_loc(mon_date)
        row = df.iloc[loc]

        # ── 信号条件 ───────────────────────────────────────────────────────
        tqqq_up = row["close"] > row["open"]
        if not tqqq_up:
            continue

        if filter_mode == "ma20":
            if pd.isna(row["qqq_ma20"]) or row["qqq_close"] <= row["qqq_ma20"]:
                continue

        if filter_mode == "ma20_65":
            if (pd.isna(row["qqq_ma20"]) or pd.isna(row["qqq_ma65"])
                    or row["qqq_close"] <= row["qqq_ma20"]
                    or row["qqq_close"] <= row["qqq_ma65"]):
                continue

        # ── 本周持仓区间（最多5个交易日，周一到周五） ────────────────────
        week_key  = row["year_week"]
        week_idx  = df.index[df["year_week"] == week_key]
        hold_days = week_idx[week_idx >= mon_date][:5]
        if len(hold_days) < 2:
            continue

        entry_price     = row["close"]
        shares          = int(TACTICAL_CAPITAL // entry_price)
        if shares <= 0:
            continue
        notional        = shares * entry_price
        r_per_share     = R_DOLLAR / shares
        stop_price      = entry_price - r_per_share   # 多头止损

        # ── 止损检查：周二起逐日检查当日最低价 ───────────────────────────
        exit_price  = df["close"].iloc[df.index.get_loc(hold_days[-1])]
        exit_date   = hold_days[-1]
        exit_reason = "time_exit"

        for day in hold_days[1:]:
            d_loc = df.index.get_loc(day)
            if df["low"].iloc[d_loc] <= stop_price:
                exit_price  = stop_price
                exit_date   = day
                exit_reason = "stop_loss"
                break

        # ── 盈亏 ─────────────────────────────────────────────────────────
        gross_pnl    = shares * (exit_price - entry_price)
        trading_cost = notional * COST_BPS * 2
        net_pnl      = gross_pnl - trading_cost

        trades.append(dict(
            entry_date    = mon_date.date(),
            exit_date     = exit_date.date(),
            entry_price   = round(entry_price, 4),
            exit_price    = round(exit_price, 4),
            stop_price    = round(stop_price, 4),
            qqq_close     = round(row["qqq_close"], 4),
            qqq_ma20      = round(row["qqq_ma20"], 4) if not pd.isna(row["qqq_ma20"]) else np.nan,
            qqq_ma65      = round(row["qqq_ma65"], 4) if not pd.isna(row["qqq_ma65"]) else np.nan,
            shares        = shares,
            notional      = round(notional, 2),
            gross_pnl     = round(gross_pnl, 2),
            net_pnl       = round(net_pnl, 2),
            R_multiple    = round(net_pnl / R_DOLLAR, 3),
            exit_reason   = exit_reason,
        ))

    if not trades:
        raise RuntimeError(f"filter_mode='{filter_mode}' 未产生任何交易，请检查数据或条件。")

    result = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    result["cum_pnl"]         = result["net_pnl"].cumsum()
    result["account_equity"]  = ACCOUNT_CAPITAL  + result["cum_pnl"]
    result["tactical_equity"] = TACTICAL_CAPITAL + result["cum_pnl"]
    return result

# ── 指标计算 ─────────────────────────────────────────────────────────────────
def max_drawdown(eq: pd.Series) -> float:
    return float(((eq - eq.cummax()) / eq.cummax()).min())

def calc_metrics(trades: pd.DataFrame, label: str) -> dict:
    pnl    = trades["net_pnl"]
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf     = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.nan
    return dict(
        版本                  = label,
        交易笔数              = len(trades),
        总净盈亏              = round(pnl.sum(), 2),
        账户最终权益          = round(trades["account_equity"].iloc[-1], 2),
        账户总收益率          = round(trades["account_equity"].iloc[-1] / ACCOUNT_CAPITAL - 1, 4),
        战术层总收益率        = round(trades["tactical_equity"].iloc[-1] / TACTICAL_CAPITAL - 1, 4),
        胜率                  = round((pnl > 0).mean(), 4),
        平均盈利              = round(wins.mean(), 2)   if len(wins)   else np.nan,
        平均亏损              = round(losses.mean(), 2) if len(losses) else np.nan,
        盈亏比PF              = round(pf, 3),
        最大回撤_账户         = round(max_drawdown(trades["account_equity"]), 4),
        平均R                 = round(trades["R_multiple"].mean(), 3),
        中位R                 = round(trades["R_multiple"].median(), 3),
        止损触发率            = round((trades["exit_reason"] == "stop_loss").mean(), 4),
    )

# ── 运行三个版本 ──────────────────────────────────────────────────────────────
print("\n运行回测…")
versions = [
    ("none",    "V0  周一TQQQ涨（无过滤，基线）"),
    ("ma20",    "V1  周一TQQQ涨 ∧ QQQ>MA20"),
    ("ma20_65", "V2  周一TQQQ涨 ∧ QQQ>MA20 ∧ QQQ>MA65"),
]

all_trades  = {}
all_metrics = []

for fmode, label in versions:
    t = run_backtest(df, filter_mode=fmode)
    all_trades[fmode] = t
    all_metrics.append(calc_metrics(t, label))

summary = pd.DataFrame(all_metrics)

# ── 买入持有基准 ─────────────────────────────────────────────────────────────
bnh_start = df["close"].iloc[0]
bnh_end   = df["close"].iloc[-1]
bnh_ret   = bnh_end / bnh_start - 1

# ── 输出 ─────────────────────────────────────────────────────────────────────
D = "═" * 76

print(f"\n{D}")
print("  TQQQ 周一5日动量回测  ─  仅做多，QQQ均线过滤条件对比")
print(f"  账户: ${ACCOUNT_CAPITAL:,}  |  战术资金: ${TACTICAL_CAPITAL:,}  |  R: ${R_DOLLAR}")
print(f"  回测区间: {df.index[0].date()} → {df.index[-1].date()}")
print(f"  均线参数: MA{MA_SHORT}（短期）  MA{MA_LONG}（长期）  ─  基于 QQQ 收盘价")
print(D)

# 核心指标横向对比表
print(f"\n{'指标':<22}", end="")
for m in all_metrics:
    print(f"  {m['版本'][:28]:<28}", end="")
print()
print("─" * 76)

row_keys = [
    ("交易笔数",       "交易笔数",       "{}",      ""),
    ("总净盈亏",       "总净盈亏",       "${:,.2f}", ""),
    ("账户总收益率",   "账户总收益率",   "{:.2%}",  ""),
    ("战术层总收益率", "战术层总收益率", "{:.2%}",  ""),
    ("胜率",           "胜率",           "{:.2%}",  ""),
    ("平均R",          "平均R",          "{:+.3f}R", ""),
    ("中位R",          "中位R",          "{:+.3f}R", ""),
    ("盈亏比PF",       "盈亏比PF",       "{:.3f}",  ""),
    ("最大回撤_账户",  "最大回撤_账户",  "{:.2%}",  ""),
    ("止损触发率",     "止损触发率",     "{:.2%}",  ""),
    ("平均盈利",       "平均盈利",       "${:,.2f}", ""),
    ("平均亏损",       "平均亏损",       "${:,.2f}", ""),
]

for label, key, fmt, _ in row_keys:
    print(f"  {label:<20}", end="")
    for m in all_metrics:
        val = m[key]
        try:
            cell = fmt.format(val)
        except (ValueError, TypeError):
            cell = "N/A"
        print(f"  {cell:<28}", end="")
    print()

print(f"\n{'─'*76}")
print(f"  TQQQ 买入持有（同期基准）: {bnh_ret:.2%}   "
      f"({df.index[0].date()} 以 ${bnh_start:.2f} 买入 → ${bnh_end:.2f})")
print(f"  Barchart 5年摘要参考: +132.44%")
print(D)

# ── 各版本最近10笔交易 ────────────────────────────────────────────────────────
cols_show = ["entry_date", "exit_date", "entry_price", "exit_price",
             "shares", "net_pnl", "R_multiple", "exit_reason"]

for fmode, label in versions:
    print(f"\n{D}")
    print(f"  最近10笔交易  ─  {label}")
    print(D)
    t = all_trades[fmode]
    print(t[cols_show].tail(10).to_string(index=False))

# ── 各版本出场原因统计 ────────────────────────────────────────────────────────
print(f"\n{D}")
print("  出场原因统计")
print(D)
for fmode, label in versions:
    t = all_trades[fmode]
    vc = t["exit_reason"].value_counts()
    parts = "  |  ".join(f"{k}: {v}笔" for k, v in vc.items())
    print(f"  {label[:38]:<38}  {parts}")

# ── 保存 CSV ─────────────────────────────────────────────────────────────────
for fmode, label in versions:
    fname = f"TQQQ_5D_{fmode}_trades.csv"
    all_trades[fmode].to_csv(fname, index=False)

summary_cn = summary.copy()
summary_cn.to_csv("TQQQ_5D_backtest_summary.csv", index=False)
print(f"\n  已保存: TQQQ_5D_none_trades.csv / ma20_trades.csv / ma20_65_trades.csv / summary.csv")
print(D + "\n")
