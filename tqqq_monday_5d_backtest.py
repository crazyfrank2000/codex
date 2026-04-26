"""
TQQQ Monday Momentum Backtest  ─  Long Only，持仓时长 + 均线过滤对比
=====================================================================
账户: $25,000  |  战术资金: $10,000  |  R: $250/笔
入场: 周一收盘  |  出场: N 个交易日后收盘 或 止损 $250（战术资金的2.5%）

版本对比：
  V0-5D   周一TQQQ涨（无过滤，持有5天）
  V0-10D  周一TQQQ涨（无过滤，持有10天）← 新增
  V1-5D   周一TQQQ涨 ∧ QQQ>MA20（持有5天）
  V2-5D   周一TQQQ涨 ∧ QQQ>MA20 ∧ QQQ>MA65（持有5天）

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
def run_backtest(df: pd.DataFrame, filter_mode: str = "none",
                 max_hold_days: int = 5) -> pd.DataFrame:
    """
    filter_mode:
        'none'   → 仅周一TQQQ上涨
        'ma20'   → 周一TQQQ涨 ∧ QQQ > MA20
        'ma20_65'→ 周一TQQQ涨 ∧ QQQ > MA20 ∧ QQQ > MA65
    max_hold_days:
        入场当天算第1天，最多持有 N 个交易日（跨周无限制）
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

        # ── 持仓区间：入场日起最多 max_hold_days 个交易日（跨周） ──────────
        hold_days = df.index[loc : loc + max_hold_days]
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

# ── 运行四个版本 ──────────────────────────────────────────────────────────────
print("\n运行回测…")
# (filter_mode, max_hold_days, key, label)
versions = [
    ("none",     5, "none_5d",  "V0-5D   无过滤·5天"),
    ("none",    10, "none_10d", "V0-10D  无过滤·10天"),
    ("ma20",     5, "ma20_5d",  "V1-5D   QQQ>MA20·5天"),
    ("ma20_65",  5, "ma65_5d",  "V2-5D   QQQ>MA20∧MA65·5天"),
]

all_trades  = {}
all_metrics = []

for fmode, hold, key, label in versions:
    t = run_backtest(df, filter_mode=fmode, max_hold_days=hold)
    all_trades[key] = t
    all_metrics.append(calc_metrics(t, label))

summary = pd.DataFrame(all_metrics)

# ── 买入持有基准（TQQQ + QQQ） ───────────────────────────────────────────────
def bnh_metrics(ticker: str, price_series: pd.Series, capital: float) -> dict:
    start_p = float(price_series.iloc[0])
    end_p   = float(price_series.iloc[-1])
    shares  = int(capital // start_p)
    equity  = capital + shares * (price_series - start_p)
    dd      = float(((equity - equity.cummax()) / equity.cummax()).min())
    return dict(
        总收益率   = round(end_p / start_p - 1, 4),
        账户最终权益 = round(capital + shares * (end_p - start_p), 2),
        最大回撤   = round(dd, 4),
    )

qqq_bnh  = bnh_metrics("QQQ",  qqq.loc[df.index[0]:, "close"],  ACCOUNT_CAPITAL)
tqqq_bnh = bnh_metrics("TQQQ", df["close"],                      ACCOUNT_CAPITAL)

# ── 输出 ─────────────────────────────────────────────────────────────────────
D  = "═" * 84
D2 = "─" * 84

print(f"\n{D}")
print("  TQQQ 周一动量回测  ─  仅做多，持仓时长 + 均线过滤对比")
print(f"  账户: ${ACCOUNT_CAPITAL:,}  |  战术资金: ${TACTICAL_CAPITAL:,}  |  R: ${R_DOLLAR}")
print(f"  回测区间: {df.index[0].date()} → {df.index[-1].date()}")
print(f"  均线参数: MA{MA_SHORT}（短期）  MA{MA_LONG}（长期）  ─  基于 QQQ 收盘价")
print(D)

# ── 策略对比表 ────────────────────────────────────────────────────────────────
COL = 18
HDR = f"  {'指标':<20}"
for m in all_metrics:
    HDR += f"  {m['版本'][:COL]:<{COL}}"
print(f"\n{HDR}")
print(D2)

row_keys = [
    ("交易笔数",       "交易笔数",       "{}"),
    ("总净盈亏",       "总净盈亏",       "${:,.0f}"),
    ("账户总收益率",   "账户总收益率",   "{:.2%}"),
    ("战术层总收益率", "战术层总收益率", "{:.2%}"),
    ("胜率",           "胜率",           "{:.2%}"),
    ("平均R",          "平均R",          "{:+.3f}R"),
    ("中位R",          "中位R",          "{:+.3f}R"),
    ("盈亏比PF",       "盈亏比PF",       "{:.3f}"),
    ("最大回撤",       "最大回撤_账户",  "{:.2%}"),
    ("止损触发率",     "止损触发率",     "{:.2%}"),
    ("平均盈利",       "平均盈利",       "${:,.0f}"),
    ("平均亏损",       "平均亏损",       "${:,.0f}"),
]

for disp, key, fmt in row_keys:
    line = f"  {disp:<20}"
    for m in all_metrics:
        val = m[key]
        try:
            cell = fmt.format(val)
        except (ValueError, TypeError):
            cell = "N/A"
        line += f"  {cell:<{COL}}"
    print(line)

# ── 买入持有基准行 ────────────────────────────────────────────────────────────
print(D2)
bnh_rows = [
    ("总收益率",     "总收益率",   "{:.2%}"),
    ("账户最终权益", "账户最终权益", "${:,.0f}"),
    ("最大回撤",     "最大回撤",   "{:.2%}"),
]
print(f"\n  {'基准':<20}  {'TQQQ B&H':>{COL}}  {'QQQ B&H':>{COL}}")
print(D2)
for disp, key, fmt in bnh_rows:
    t_str = fmt.format(tqqq_bnh[key])
    q_str = fmt.format(qqq_bnh[key])
    print(f"  {disp:<20}  {t_str:>{COL}}  {q_str:>{COL}}")
print(D)

# ── 各版本最近10笔交易 ────────────────────────────────────────────────────────
cols_show = ["entry_date", "exit_date", "entry_price", "exit_price",
             "shares", "net_pnl", "R_multiple", "exit_reason"]

for _, _, key, label in versions:
    print(f"\n{D}")
    print(f"  最近10笔交易  ─  {label}")
    print(D)
    print(all_trades[key][cols_show].tail(10).to_string(index=False))

# ── 出场原因统计 ──────────────────────────────────────────────────────────────
print(f"\n{D}")
print("  出场原因统计")
print(D)
for _, _, key, label in versions:
    vc    = all_trades[key]["exit_reason"].value_counts()
    parts = "  |  ".join(f"{k}: {v}笔" for k, v in vc.items())
    print(f"  {label:<30}  {parts}")

# ── 保存 CSV ─────────────────────────────────────────────────────────────────
for _, _, key, label in versions:
    all_trades[key].to_csv(f"TQQQ_{key}_trades.csv", index=False)
summary.to_csv("TQQQ_backtest_summary.csv", index=False)
print(f"\n  已保存: TQQQ_none_5d / none_10d / ma20_5d / ma65_5d _trades.csv + summary.csv")
print(D + "\n")
