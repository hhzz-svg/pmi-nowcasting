"""test_shap_analysis.py — SHAP 分析的轻量单测（构造数据，不拉网络）。

只验证：SHAP 值矩阵形状正确、平均 |SHAP| 与特征列对齐、"信息特征"确实
比"纯噪声特征"重要。避免依赖 akshare 真实数据，保证 CI 可复现。
"""
import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")  # 缺 shap 则整文件跳过（进阶可选依赖）

from pmi_nowcast import dataset, shap_analysis


def _toy_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """构造：pmi 是强信号（决定标签），其余为噪声，模拟真实数据结构。"""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2005-01", periods=n, freq="M").to_timestamp()
    pmi = rng.normal(50, 1.5, n)
    df = pd.DataFrame(
        {
            "pmi": pmi,
            "m2_yoy": rng.normal(10, 2, n),  # 噪声
            "cpi_yoy": rng.normal(2, 1, n),  # 噪声
            # 标签：下月扩张与否，主要由本月 pmi 决定（强信号）
            "y_expansion": (pmi + rng.normal(0, 0.3, n) > 50).astype(int),
            "y_direction": rng.integers(0, 2, n),
            "pmi_current": pmi,
            "pmi_next": pmi,
        },
        index=idx,
    )
    return df


def test_compute_shap_shape_matches_features():
    df = _toy_df()
    sv1, X, feat_cols = shap_analysis.compute_shap(df, target="y_expansion")
    assert sv1.shape[0] == len(df)
    assert sv1.shape[1] == len(feat_cols)
    assert X.shape == (len(df), len(feat_cols))


def test_signal_feature_ranks_above_noise():
    """pmi 是构造的强信号，其平均 |SHAP| 应高于纯噪声特征。"""
    df = _toy_df()
    sv1, _, feat_cols = shap_analysis.compute_shap(df, target="y_expansion")
    mean_abs = pd.Series(np.abs(sv1).mean(axis=0), index=feat_cols)
    assert mean_abs["pmi"] > mean_abs["cpi_yoy"]
    assert mean_abs["pmi"] > mean_abs["m2_yoy"]


def test_feature_columns_excludes_labels():
    """回归护栏：SHAP 用的特征列绝不含标签，防泄漏。"""
    df = _toy_df()
    feat_cols = dataset.feature_columns(df)
    for leak in ("y_expansion", "y_direction", "pmi_next", "pmi_current"):
        assert leak not in feat_cols
