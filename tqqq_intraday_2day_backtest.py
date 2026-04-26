"""
TQQQ 日内开盘区间·当日收盘离场 回测
======================================
策略规则：
  信号  : 周一首根 N 分钟 K 线收盘 > 开盘  → LONG
  入场  : 周一首根 K 线收盘价（约 10:00 ET）
  出场  : ① 当日收盘（周一最后一根 K 线，约 16:00 ET）
          ② 持有期内任意 K 线最低价触及止损线即离场
  止损  : 固定亏损 $250（战术资金 $10,000 的 2.5%）
  标的  : TQQQ
  账户  : 总账户 $25,000  |  战术资金 $10,000  |  R = $250

数据限制（yfinance）：
  30 分钟 K 线：最多 60 天 ≈ 11 个周一（仅供结构验证）
  1 小时  K 线：最多 730 天 ≈ 137 个周一（统计意义较强）
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date

# ── 参数 ────────────────────────────────────────────────────────────────────
ACCOUNT_CAPITAL  = 25_000
TACTICAL_CAPITAL = 10_000
R_DOLLAR         = 250
MAX_HOLD_DAYS    = 1      # 当日收盘离场（持有至周一最后一根 K 线）
COST_BPS         = 0.0

# 要对比的两组参数：(interval, period, label)
CONFIGS = [
    ("30m", "60d",  "30分钟K线 (~60天/11周一) · 当日收盘离场"),
    ("1h",  "730d", "1小时K线  (~730天/137周一) · 当日收盘离场"),
]

# ── 下载并预处理日内数据 ─────────────────────────────────────────────────────
def load_intraday(ticker: str, interval: str, period: str) -> pd.DataFrame:
    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        sys.exit(f"ERROR: 无法下载 {ticker} {interval} 数据。")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close"]].copy()
    df.columns = ["open", "high", "low", "close"]
    df.index = pd.to_datetime(df.index)

    # 统一转换为纽约时区
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert("America/New_York")
    else:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")

    df["date"]    = pd.to_datetime(df.index.date)
    df["weekday"] = df.index.dayofweek   # 0=Mon 4=Fri
    return df

# ── 回测引擎 ─────────────────────────────────────────────────────────────────
def run_intraday_backtest(df: pd.DataFrame, max_hold_days: int = 2) -> pd.DataFrame:
    """
    信号：周一首根 K 线（市场开盘后第一根）close > open → LONG
    入场：该首根 K 线收盘价
    持仓：从入场 K 线之后开始，最多持有至当日（周一）收盘
          max_hold_days=1  →  周一入场后剩余 K 线（当日 EOD 离场）
    止损：每根 K 线最低价触及止损线即按止损价离场
    """
    # 找所有交易日（按日期去重）
    all_days = df["date"].drop_duplicates().sort_values().tolist()
    day_to_idx = {d: i for i, d in enumerate(all_days)}

    monday_days = sorted(df.loc[df["weekday"] == 0, "date"].drop_duplicates().tolist())
    trades = []

    for mon_date in monday_days:
        mon_bars = df[df["date"] == mon_date]
        if len(mon_bars) < 2:
            continue

        # 信号 K 线 = 周一第一根
        sig_bar = mon_bars.iloc[0]
        if sig_bar["close"] <= sig_bar["open"]:   # 首根非阳线，跳过
            continue

        entry_price = sig_bar["close"]
        entry_ts    = mon_bars.index[0]

        shares = int(TACTICAL_CAPITAL // entry_price)
        if shares <= 0:
            continue

        notional    = shares * entry_price
        r_per_share = R_DOLLAR / shares
        stop_price  = entry_price - r_per_share

        # ── 确定持仓期：入场 K 线之后的 K 线，最多到第 max_hold_days 个交易日结束 ──
        mon_day_idx = day_to_idx[mon_date]
        end_day_idx = min(mon_day_idx + max_hold_days - 1, len(all_days) - 1)
        end_day     = all_days[end_day_idx]

        # 入场 K 线之后（不含入场 K 线本身）到 end_day 收盘
        hold_bars = df[(df.index > entry_ts) & (df["date"] <= end_day)]

        if hold_bars.empty:
            continue

        exit_price  = hold_bars["close"].iloc[-1]
        exit_ts     = hold_bars.index[-1]
        exit_reason = "time_exit"

        # ── 逐根检查止损 ────────────────────────────────────────────────────
        for ts, bar in hold_bars.iterrows():
            if bar["low"] <= stop_price:
                exit_price  = stop_price
                exit_ts     = ts
                exit_reason = "stop_loss"
                break

        gross_pnl    = shares * (exit_price - entry_price)
        trading_cost = notional * COST_BPS * 2
        net_pnl      = gross_pnl - trading_cost

        trades.append(dict(
            entry_date   = mon_date.date(),
            entry_time   = entry_ts.strftime("%H:%M"),
            exit_date    = exit_ts.date(),
            exit_time    = exit_ts.strftime("%H:%M"),
            entry_price  = round(entry_price, 4),
            exit_price   = round(exit_price, 4),
            stop_price   = round(stop_price, 4),
            shares       = shares,
            notional     = round(notional, 2),
            gross_pnl    = round(gross_pnl, 2),
            net_pnl      = round(net_pnl, 2),
            R_multiple   = round(net_pnl / R_DOLLAR, 3),
            exit_reason  = exit_reason,
        ))

    if not trades:
        return pd.DataFrame()

    result = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    result["cum_pnl"]         = result["net_pnl"].cumsum()
    result["account_equity"]  = ACCOUNT_CAPITAL  + result["cum_pnl"]
    result["tactical_equity"] = TACTICAL_CAPITAL + result["cum_pnl"]
    return result

# ── 指标计算 ─────────────────────────────────────────────────────────────────
def max_drawdown(eq: pd.Series) -> float:
    return float(((eq - eq.cummax()) / eq.cummax()).min())

def calc_metrics(trades: pd.DataFrame, label: str, data_start, data_end) -> dict:
    if trades.empty:
        return {"版本": label, "备注": "无交易"}
    pnl    = trades["net_pnl"]
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf     = wins.sum() / abs(losses.sum()) if len(losses) > 0 else np.nan
    return dict(
        版本          = label,
        数据区间      = f"{data_start} → {data_end}",
        交易笔数      = len(trades),
        总净盈亏      = round(pnl.sum(), 2),
        账户最终权益  = round(trades["account_equity"].iloc[-1], 2),
        账户总收益率  = round(trades["account_equity"].iloc[-1] / ACCOUNT_CAPITAL - 1, 4),
        战术层收益率  = round(trades["tactical_equity"].iloc[-1] / TACTICAL_CAPITAL - 1, 4),
        胜率          = round((pnl > 0).mean(), 4),
        平均盈利      = round(wins.mean(), 2)   if len(wins)   else np.nan,
        平均亏损      = round(losses.mean(), 2) if len(losses) else np.nan,
        盈亏比PF      = round(pf, 3),
        最大回撤      = round(max_drawdown(trades["account_equity"]), 4),
        平均R         = round(trades["R_multiple"].mean(), 3),
        中位R         = round(trades["R_multiple"].median(), 3),
        止损触发率    = round((trades["exit_reason"] == "stop_loss").mean(), 4),
    )

# ── 主流程 ───────────────────────────────────────────────────────────────────
D = "═" * 78
all_results = []
all_trades  = {}

print(D)
print("  TQQQ 周一首根K线信号  |  当日收盘离场  |  仅做多")
print(f"  账户 ${ACCOUNT_CAPITAL:,}  |  战术资金 ${TACTICAL_CAPITAL:,}  |  R ${R_DOLLAR}")
print(D)

for interval, period, label in CONFIGS:
    print(f"\n下载 TQQQ {interval} 数据（period={period}）…")
    df = load_intraday("TQQQ", interval, period)
    data_start = df.index[0].date()
    data_end   = df.index[-1].date()
    monday_cnt = df.loc[df["weekday"] == 0, "date"].drop_duplicates().shape[0]
    print(f"  区间: {data_start} → {data_end}  |  K线数: {len(df)}  |  周一天数: {monday_cnt}")

    trades = run_intraday_backtest(df, max_hold_days=MAX_HOLD_DAYS)
    all_trades[interval] = trades

    if trades.empty:
        print("  ⚠ 该区间内无有效交易（信号不足）")
        continue

    m = calc_metrics(trades, label, data_start, data_end)
    all_results.append(m)

    # 输出该版本明细
    print(f"\n{'─'*78}")
    print(f"  {label}")
    print(f"{'─'*78}")
    for k, v in m.items():
        if k == "版本":
            continue
        if isinstance(v, float):
            if k in ("账户总收益率", "战术层收益率", "胜率", "最大回撤", "止损触发率"):
                print(f"  {k:<18}  {v*100:>8.2f}%")
            elif k in ("平均R", "中位R"):
                print(f"  {k:<18}  {v:>+8.3f}R")
            else:
                print(f"  {k:<18}  {v:>10.3f}")
        elif isinstance(v, (int, np.integer)):
            print(f"  {k:<18}  {v:>10,}")
        else:
            print(f"  {k:<18}  {v}")

    # 止损统计
    vc = trades["exit_reason"].value_counts()
    parts = "  |  ".join(f"{k}: {v}笔" for k, v in vc.items())
    print(f"\n  出场原因: {parts}")

    # 全部交易明细
    cols = ["entry_date", "entry_time", "exit_date", "exit_time",
            "entry_price", "exit_price", "shares", "net_pnl", "R_multiple", "exit_reason"]
    print(f"\n  全部交易明细:")
    print(trades[cols].to_string(index=False))

# ── 与日线 V0（同期）对比：仅对 1h 版本对应区间跑一次日线 V0 ─────────────────
print(f"\n{D}")
print("  对比参照：日线 V0（同期，周一日线涨 → 持有至周五，最大5天）  vs  日内EOD版")
print(D)

# 取 1h 对应的时间区间跑日线版本
h1_start = all_trades.get("1h")
if h1_start is not None and not h1_start.empty:
    ref_start = str(all_trades["1h"]["entry_date"].min())
    ref_end   = str(all_trades["1h"]["exit_date"].max())

    raw_daily = yf.download("TQQQ", start=ref_start, end=ref_end,
                            auto_adjust=True, progress=False)
    if isinstance(raw_daily.columns, pd.MultiIndex):
        raw_daily.columns = raw_daily.columns.get_level_values(0)
    dd = raw_daily[["Open","High","Low","Close"]].copy()
    dd.columns = ["open","high","low","close"]
    dd.index   = pd.to_datetime(dd.index)
    dd["weekday"]  = dd.index.dayofweek
    dd["iso_week"] = dd.index.isocalendar().week.astype(int)
    dd["iso_year"] = dd.index.isocalendar().year.astype(int)
    dd["year_week"]= dd["iso_year"].astype(str)+"-"+dd["iso_week"].astype(str).str.zfill(2)

    daily_trades = []
    for mon_date in dd.index[dd["weekday"]==0]:
        loc = dd.index.get_loc(mon_date)
        row = dd.iloc[loc]
        if row["close"] <= row["open"]:
            continue
        wk   = row["year_week"]
        wrows= dd.index[dd["year_week"]==wk]
        hrows= wrows[wrows>=mon_date][:5]
        if len(hrows) < 2:
            continue
        ep = row["close"]
        sh = int(TACTICAL_CAPITAL // ep)
        if sh <= 0: continue
        stp= ep - R_DOLLAR/sh
        xp = dd["close"].iloc[dd.index.get_loc(hrows[-1])]
        xd = hrows[-1]; xr = "time_exit"
        for dy in hrows[1:]:
            dloc = dd.index.get_loc(dy)
            if dd["low"].iloc[dloc] <= stp:
                xp=stp; xd=dy; xr="stop_loss"; break
        pnl = sh*(xp-ep)
        daily_trades.append(dict(
            entry_date=mon_date.date(), exit_date=xd.date(),
            net_pnl=round(pnl,2), R_multiple=round(pnl/R_DOLLAR,3),
            exit_reason=xr))

    if daily_trades:
        dt = pd.DataFrame(daily_trades)
        dt["cum_pnl"]        = dt["net_pnl"].cumsum()
        dt["account_equity"] = ACCOUNT_CAPITAL + dt["cum_pnl"]
        dt["tactical_equity"]= TACTICAL_CAPITAL+ dt["cum_pnl"]
        dm = calc_metrics(dt, "日线V0(同期对比)", ref_start, ref_end)
        print(f"\n  {'交易笔数':<18}  {dm['交易笔数']:>10,}")
        print(f"  {'总净盈亏':<18}  ${dm['总净盈亏']:>10,.2f}")
        print(f"  {'账户总收益率':<18}  {dm['账户总收益率']*100:>8.2f}%")
        print(f"  {'战术层收益率':<18}  {dm['战术层收益率']*100:>8.2f}%")
        print(f"  {'胜率':<18}  {dm['胜率']*100:>8.2f}%")
        print(f"  {'平均R':<18}  {dm['平均R']:>+8.3f}R")
        print(f"  {'盈亏比PF':<18}  {dm['盈亏比PF']:>10.3f}")
        print(f"  {'最大回撤':<18}  {dm['最大回撤']*100:>8.2f}%")
        print(f"  {'止损触发率':<18}  {dm['止损触发率']*100:>8.2f}%")
        vc2 = dt["exit_reason"].value_counts()
        print(f"\n  出场原因: {'  |  '.join(f'{k}: {v}笔' for k,v in vc2.items())}")

# ── 保存 CSV ─────────────────────────────────────────────────────────────────
for interval, _, _ in CONFIGS:
    t = all_trades.get(interval)
    if t is not None and not t.empty:
        fname = f"TQQQ_intraday_{interval.replace('m','min').replace('h','h')}_eod_trades.csv"
        t.to_csv(fname, index=False)
        print(f"\n  已保存: {fname}")

print(f"\n{D}\n")
