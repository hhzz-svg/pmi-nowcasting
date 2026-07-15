"""test_dataset.py — 标签构造防泄漏单测。

验证标签是"下月 PMI"（唯一允许的前视，即预测目标本身），
且特征行不含下月信息。
"""
import pandas as pd

from pmi_nowcast import dataset


def test_label_is_next_month_pmi(toy_indicators):
    """y_expansion[t] 应基于 PMI(t+1)，而非 PMI(t)。"""
    labels = dataset.build_labels(toy_indicators)
    # pmi = 48..59，asof 2020-01 的下月(2月)PMI=49 <50 → y=0
    assert labels.loc[pd.Timestamp("2020-01-01"), "pmi_next"] == 49
    assert labels.loc[pd.Timestamp("2020-01-01"), "y_expansion"] == 0
    # asof 2020-03 下月(4月)PMI=51 >50 → y=1
    assert labels.loc[pd.Timestamp("2020-03-01"), "pmi_next"] == 51
    assert labels.loc[pd.Timestamp("2020-03-01"), "y_expansion"] == 1


def test_direction_label(toy_indicators):
    """PMI 单调递增 → 方向标签恒为 1（上行）。"""
    labels = dataset.build_labels(toy_indicators)
    valid = labels["y_direction"].dropna()
    assert (valid == 1).all()


def test_last_month_label_is_na(toy_indicators):
    """最后一个月没有'下月'，pmi_next 应为 NaN（不可造标签）。"""
    labels = dataset.build_labels(toy_indicators)
    last = labels.index.max()
    assert pd.isna(labels.loc[last, "pmi_next"])


def test_feature_columns_excludes_labels():
    """特征列绝不包含任何标签/前视列。"""
    df = pd.DataFrame(
        columns=["pmi", "m2_yoy", "pmi_current", "pmi_next", "y_expansion", "y_direction"]
    )
    feats = dataset.feature_columns(df)
    for leak in ["pmi_next", "y_expansion", "y_direction", "pmi_current"]:
        assert leak not in feats
    assert "pmi" in feats and "m2_yoy" in feats
