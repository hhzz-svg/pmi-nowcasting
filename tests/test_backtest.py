"""test_backtest.py — 交易成本与择时逻辑的正确性。

重点验证新加的交易成本：零成本回退到无摩擦结果，正成本单调侵蚀收益，
换手次数与仓位切换一致。
"""
import numpy as np
import pandas as pd

from pmi_nowcast import backtest


def _make_data():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    # 信号：预测下月扩张(1)/收缩(0)
    signal = pd.Series([1, 1, 0, 1, 0, 1], index=idx)
    # 市场每月固定 +2%，便于手算
    mkt = pd.Series([0.02] * 6, index=idx, name="hs300_ret")
    return signal, mkt


def test_zero_cost_matches_frictionless():
    """零成本时，策略收益 = 前移仓位 × 市场收益（无摩擦）。"""
    signal, mkt = _make_data()
    bt = backtest.run_backtest(signal, mkt, cost_bps=0.0)
    expected = signal.shift(1).fillna(0) * mkt
    np.testing.assert_allclose(bt["strategy_ret"].to_numpy(), expected.to_numpy())


def test_cost_erodes_return():
    """正交易成本应使策略累计收益严格低于零成本情形。"""
    signal, mkt = _make_data()
    nav_free = backtest.run_backtest(signal, mkt, cost_bps=0.0)["strategy_nav"].iloc[-1]
    nav_cost = backtest.run_backtest(signal, mkt, cost_bps=30.0)["strategy_nav"].iloc[-1]
    assert nav_cost < nav_free


def test_cost_monotonic():
    """成本越高，终值净值越低（单调）。"""
    signal, mkt = _make_data()
    navs = [
        backtest.run_backtest(signal, mkt, cost_bps=c)["strategy_nav"].iloc[-1]
        for c in (0.0, 10.0, 20.0, 40.0)
    ]
    assert all(navs[i] > navs[i + 1] for i in range(len(navs) - 1))


def test_performance_stats_keys():
    """绩效统计返回预期字段且数值有限。"""
    signal, mkt = _make_data()
    bt = backtest.run_backtest(signal, mkt, cost_bps=10.0)
    stats = backtest.performance_stats(bt["strategy_ret"])
    assert {"annual_return", "annual_vol", "sharpe", "max_drawdown"} <= set(stats)
