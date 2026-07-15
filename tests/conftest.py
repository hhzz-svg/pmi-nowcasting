"""tests/conftest.py — 共享测试夹具。"""
import pandas as pd
import pytest


@pytest.fixture
def toy_indicators():
    """构造可控的假指标数据，用于验证时点对齐逻辑。"""
    months = pd.date_range("2020-01-01", periods=12, freq="MS")
    pmi = pd.DataFrame({"month": months, "pmi": range(48, 60)})  # 48..59 递增
    # 一个 lag=1 的指标：值等于月份序号，便于验证移位
    other = pd.DataFrame({"month": months, "ip_yoy": [float(i) for i in range(12)]})
    return {"pmi": pmi, "industrial": other}
