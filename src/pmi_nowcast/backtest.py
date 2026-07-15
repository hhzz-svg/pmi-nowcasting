"""backtest.py — ★信号→择时净值曲线（设计文档"升级 3"：经济价值评估）。

把二分类信号翻译成"钱"：
  预测扩张 → 下月持有沪深300；预测收缩 → 持币（收益 0）。
对比买入持有，算年化收益、夏普比率、最大回撤。

诚实声明（README 也会写）：仅示意信号有无经济价值，未计交易成本 / 滑点，
非真实可交易策略。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ingest


def load_hs300_monthly_returns() -> pd.Series:
    """沪深300 月度收益率，index 归一到月初，与预测面板对齐。"""
    import akshare as ak

    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    monthly_close = df["close"].resample("MS").last()
    return monthly_close.pct_change().rename("hs300_ret")


def run_backtest(signal: pd.Series, market_ret: pd.Series) -> pd.DataFrame:
    """按信号择时，返回逐月策略/基准收益与净值。

    signal[t] = 对 t+1 月的预测（1=扩张持股，0=收缩持币）。
    该信号赚取的是 t+1 月的市场收益，故对齐时把信号前移一期。
    """
    df = pd.DataFrame({"signal": signal}).join(market_ret, how="inner")
    # 信号预测下月 → 用信号获取下月收益
    df["strategy_ret"] = df["signal"].shift(1).fillna(0) * df["hs300_ret"]
    df["buyhold_ret"] = df["hs300_ret"]
    df = df.dropna(subset=["hs300_ret"])
    df["strategy_nav"] = (1 + df["strategy_ret"]).cumprod()
    df["buyhold_nav"] = (1 + df["buyhold_ret"]).cumprod()
    return df


def performance_stats(ret: pd.Series, periods_per_year: int = 12) -> dict:
    """年化收益、年化波动、夏普、最大回撤。"""
    ret = ret.dropna()
    if len(ret) == 0:
        return {"annual_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    ann_ret = (1 + ret).prod() ** (periods_per_year / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    nav = (1 + ret).cumprod()
    max_dd = (nav / nav.cummax() - 1).min()
    return {
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
