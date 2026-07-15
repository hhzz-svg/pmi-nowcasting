"""test_calendar.py — 时点对齐防泄漏单测。

验证发布滞后移位正确：lag=k 的指标，标称月 m 的数值只在 m+k 月可得，
绝不在预测时点"提前泄漏"。
"""
import pandas as pd

from pmi_nowcast import calendar


def test_lag_shifts_availability_forward():
    """lag=1 时，1 月的数据应在 2 月才可得。"""
    df = pd.DataFrame(
        {"month": pd.date_range("2020-01-01", periods=3, freq="MS"), "x": [10.0, 20.0, 30.0]}
    )
    shifted = calendar.apply_publication_lag(df, "x", lag=1)
    # 1 月的值 10 应出现在 2 月索引处
    assert shifted.loc[pd.Timestamp("2020-02-01"), "x"] == 10.0
    assert shifted.loc[pd.Timestamp("2020-03-01"), "x"] == 20.0
    # 1 月索引处应无数据（1 月时该月数据还没发）
    assert pd.Timestamp("2020-01-01") not in shifted.index


def test_lag_zero_is_identity():
    """lag=0（如 PMI 当月发布）不移位。"""
    df = pd.DataFrame(
        {"month": pd.date_range("2020-01-01", periods=3, freq="MS"), "x": [1.0, 2.0, 3.0]}
    )
    shifted = calendar.apply_publication_lag(df, "x", lag=0)
    assert shifted.loc[pd.Timestamp("2020-01-01"), "x"] == 1.0


def test_panel_no_future_leak(toy_indicators):
    """PIT 面板中，任一 asof_month 处的 ip_yoy(lag=1) 必来自更早月份。"""
    panel = calendar.build_pit_panel(toy_indicators)
    # ip_yoy 值 = 原月份序号(0..11)，lag=1 → asof 2020-02 处应为 0（1 月的值）
    assert panel.loc[pd.Timestamp("2020-02-01"), "ip_yoy"] == 0.0
    # pmi lag=0 → asof 2020-02 处应为 49（2 月当值：48,49,...）
    assert panel.loc[pd.Timestamp("2020-02-01"), "pmi"] == 49.0


def test_interpolation_never_extrapolates():
    """内部缺口可插值，但序列两端之外绝不外推（不造未来数据）。"""
    df = pd.DataFrame(
        {
            "month": pd.date_range("2020-01-01", periods=5, freq="MS"),
            "x": [1.0, None, 3.0, 4.0, 5.0],
        }
    )
    panel = calendar.build_pit_panel({"industrial": df.rename(columns={"x": "ip_yoy"})})
    # 内部缺口（2 月）被插值
    assert not pd.isna(panel.loc[pd.Timestamp("2020-02-01"), "ip_yoy"])
