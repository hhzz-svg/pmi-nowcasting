"""test_significance.py — 统计显著性检验的单测。

用构造数据验证：
- bootstrap CI 覆盖点估计、区间有序；
- 配对差检验对"两个完全相同的模型"给出差≈0、p 大；对"一强一弱"给出差>0；
- McNemar 对"两模型逐点预测完全一致"给 n_ab=n_ba=0、p=1。
"""
import numpy as np

from pmi_nowcast import significance


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # 强模型：概率与标签相关；弱模型：几乎随机
    strong = np.clip(y * 0.6 + rng.normal(0.2, 0.15, n), 0, 1)
    weak = np.clip(rng.normal(0.5, 0.2, n), 0, 1)
    return y, strong, weak


def test_bootstrap_ci_brackets_point_and_ordered():
    y, strong, _ = _make_data()
    ci = significance.bootstrap_auc_ci(y, strong, n_boot=500)
    assert ci["lo"] <= ci["auc"] <= ci["hi"]
    assert 0.0 <= ci["lo"] < ci["hi"] <= 1.0


def test_auc_diff_strong_beats_weak():
    """强模型对弱模型：AUC 差应显著为正，p 应较小。"""
    y, strong, weak = _make_data()
    d = significance.bootstrap_auc_diff(y, strong, weak, n_boot=500)
    assert d["diff"] > 0
    assert d["p_value"] < 0.05


def test_auc_diff_identical_models_not_significant():
    """两个完全相同的模型：差=0、p=1（不显著）。"""
    y, strong, _ = _make_data()
    d = significance.bootstrap_auc_diff(y, strong, strong, n_boot=500)
    assert abs(d["diff"]) < 1e-9
    assert d["p_value"] == 1.0


def test_mcnemar_identical_predictions():
    """逐点预测完全一致：无分歧样本，p=1。"""
    y = np.array([0, 1, 0, 1, 1, 0])
    pred = np.array([0, 1, 1, 1, 0, 0])
    mc = significance.mcnemar_test(y, pred, pred)
    assert mc["n_ab"] == 0 and mc["n_ba"] == 0
    assert mc["p_value"] == 1.0


def test_mcnemar_detects_systematic_difference():
    """A 全对、B 全错：分歧全在一侧，p 应很小。"""
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    pred_a = y.copy()          # 全对
    pred_b = 1 - y             # 全错
    mc = significance.mcnemar_test(y, pred_a, pred_b)
    assert mc["n_ab"] == len(y) and mc["n_ba"] == 0
    assert mc["p_value"] < 0.05
