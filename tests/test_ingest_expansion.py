"""test_ingest_expansion.py — 数据范围扩充相关的单测。

验证：
- _shape 去重修复：同月多条报告行（预告行值为空）时，保留有值行而非空行；
- 新增特征的发布滞后表：市场特征零滞后、出口滞后 1 个月；
- 零滞后列经时点对齐后月份不移位（月末收盘即知）。
"""
import pandas as pd

from pmi_nowcast import calendar, config, ingest


def test_shape_keeps_valued_row_over_empty_forecast_row():
    """2024 年出口源的真实病灶：同月先出有值行、后出空预告行，不能被空行覆盖。"""
    raw = pd.DataFrame(
        {
            "日期": ["2024-09-07", "2024-09-10", "2024-09-11", "2024-10-14"],
            "今值": [None, 8.7, None, 2.4],
        }
    )
    spec = {"date_col": "日期", "value_cols": {"今值": "export_yoy"}}
    out = ingest._shape(raw, spec)
    assert len(out) == 2
    sep = out[out["month"] == "2024-09-01"]["export_yoy"].iloc[0]
    assert sep == 8.7  # 有值行胜出，而非 NaN


def test_publication_lags_for_new_features():
    assert config.PUBLICATION_LAG["hs300_ret"] == 0
    assert config.PUBLICATION_LAG["hs300_vol"] == 0
    assert config.PUBLICATION_LAG["export_yoy"] == 1


def test_zero_lag_market_column_not_shifted():
    """lag=0 的市场列时点对齐后月份原地不动。"""
    months = pd.date_range("2020-01-01", periods=3, freq="MS")
    df = pd.DataFrame({"month": months, "hs300_ret": [0.01, -0.02, 0.03]})
    shifted = calendar.apply_publication_lag(df, "hs300_ret", lag=0)
    assert list(shifted.index) == list(months)
    assert shifted["hs300_ret"].tolist() == [0.01, -0.02, 0.03]


def test_market_indicator_spec_is_custom():
    spec = config.INDICATORS["market"]
    assert spec.get("custom") == "load_hs300_monthly"
    assert hasattr(ingest, "load_hs300_monthly")
