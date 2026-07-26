"""significance.py — 样本外指标的统计显著性检验。

项目的核心诚实发现是"朴素持续基准打平甚至略胜 ML 模型"。但 ~80 个样本外点上，
零点零几的 AUC 差距，究竟是**真实差异**还是**抽样噪声**？不回答这个问题，
"打平"的结论就站不住脚。本模块给出三种互补的证据：

  1) bootstrap_auc_ci     每个模型 AUC 的自助置信区间（单模型不确定性）。
  2) bootstrap_auc_diff   两模型 AUC 差的配对自助分布 + 双侧 p 值（差异是否跨 0）。
  3) mcnemar_test         两模型逐点预测对错的配对检验（精确二项，不依赖大样本近似）。

小样本宏观预测里，"我报告了差异的置信区间、且承认它不显著"远比"我的模型 AUC 高
0.03"更有说服力——这正是本模块存在的意义。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


def bootstrap_auc_ci(
    y_true, y_proba, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> dict:
    """单模型 AUC 的自助置信区间。

    对 (y_true, y_proba) 有放回重采样 n_boot 次，每次重算 AUC，取分位数为区间。
    返回 {auc, lo, hi}，其中 lo/hi 为 (alpha/2, 1-alpha/2) 分位。
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    point = roc_auc_score(y_true, y_proba)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:  # 重采样偶尔只剩单类，AUC 无定义，跳过
            continue
        aucs.append(roc_auc_score(yt, y_proba[idx]))
    aucs = np.asarray(aucs)
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": float(point), "lo": float(lo), "hi": float(hi)}


def bootstrap_auc_diff(
    y_true, proba_a, proba_b, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> dict:
    """两模型 AUC 差 (A - B) 的配对自助检验。

    关键是**配对**重采样：同一组重采样下标同时作用于两模型，消除样本本身难易带来的
    共同波动，只留下模型差异。返回 {diff, lo, hi, p_value}。

    p 值为双侧：差的自助分布中跨过 0 的比例（2×较小尾），衡量"差异方向是否稳健"。
    """
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    n = len(y_true)
    point = roc_auc_score(y_true, proba_a) - roc_auc_score(y_true, proba_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, proba_a[idx]) - roc_auc_score(yt, proba_b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # 双侧 p：分布落在 0 另一侧的比例的两倍，截断到 [0,1]
    frac_le0 = float(np.mean(diffs <= 0))
    frac_ge0 = float(np.mean(diffs >= 0))
    p_value = min(1.0, 2 * min(frac_le0, frac_ge0))
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "p_value": p_value}


def bootstrap_loss_diff(
    loss_a, loss_b, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> dict:
    """两模型逐点损失差 (B - A) 的配对自助检验——Diebold-Mariano 检验的自助版。

    用于回归任务：loss_a / loss_b 为两模型在同一批样本外点上的逐点损失
    （如绝对误差 |e_t|）。检验 mean(loss_b - loss_a) 是否显著异于 0，
    即"A 是否真的比 B 准"。配对（同下标重采样）消除样本难易的共同波动。

    返回 {diff, lo, hi, p_value}；diff>0 表示 A 的平均损失更小（A 更准）。
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    d = loss_b - loss_a  # 逐点损失差
    n = len(d)
    point = float(d.mean())
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    frac_le0 = float(np.mean(means <= 0))
    frac_ge0 = float(np.mean(means >= 0))
    p_value = min(1.0, 2 * min(frac_le0, frac_ge0))
    return {"diff": point, "lo": float(lo), "hi": float(hi), "p_value": p_value}


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar 配对检验：比较两模型逐点预测的对错模式。

    只看"分歧样本"——A 对 B 错 (n_ab) 与 A 错 B 对 (n_ba)。零假设下两者应各半，
    用精确二项检验（不依赖卡方大样本近似，适合本项目小样本）。

    返回 {n_ab, n_ba, p_value}。p 小 → 两模型准确率有系统性差异。
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n_ab = int(np.sum(a_correct & ~b_correct))  # A 对、B 错
    n_ba = int(np.sum(~a_correct & b_correct))  # A 错、B 对
    n_disc = n_ab + n_ba
    if n_disc == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(n_ab, n_ba), n_disc, 0.5).pvalue)
    return {"n_ab": n_ab, "n_ba": n_ba, "p_value": p_value}
