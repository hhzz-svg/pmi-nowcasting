"""test_ensemble_calibration.py — 软融合集成 + 概率校准的单测。"""
import numpy as np

from pmi_nowcast import evaluate, models


class _ConstModel:
    """输出恒定概率 p 的假模型，用于验证融合是取平均。"""

    def __init__(self, p):
        self.p = p

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.p)
        return np.column_stack([1 - p1, p1])


def test_soft_ensemble_averages_probabilities():
    """等权软融合：两成员概率 0.2 与 0.8 → 融合应为 0.5。"""
    ens = models.SoftEnsemble([_ConstModel(0.2), _ConstModel(0.8)])
    X = np.zeros((5, 3))
    ens.fit(X, np.array([0, 1, 0, 1, 0]))
    proba = ens.predict_proba(X)[:, 1]
    assert np.allclose(proba, 0.5)


def test_ensemble_registered_with_pmi_index():
    """给定 pmi_col_index 时，模型字典应包含 ensemble 与 naive_persistence。"""
    md = models.make_models(pmi_col_index=0)
    assert "ensemble" in md
    assert "naive_persistence" in md


def test_brier_score_present_and_bounded():
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.6])
    m = evaluate.classification_metrics(y_true, (y_proba >= 0.5).astype(int), y_proba)
    assert "brier" in m
    assert 0.0 <= m["brier"] <= 1.0


def test_reliability_curve_perfect_calibration():
    """完美校准数据：某箱内预测均值应约等于实际频率。"""
    # 构造：概率 0.0 的样本全负，概率 1.0 的样本全正
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    rc = evaluate.reliability_curve(y_true, y_proba, n_bins=5)
    # 每个非空箱，mean_pred 与 frac_pos 应一致
    assert np.allclose(rc["mean_pred"].to_numpy(), rc["frac_pos"].to_numpy())
