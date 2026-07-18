"""evaluate.py — 样本外分类评估 + 特征重要性。

所有指标都在 walk-forward 样本外预测上计算，绝非训练集。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    """标准分类指标。AUC 对不平衡数据比 accuracy 更可靠；Brier 衡量概率校准。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC 需两类都出现且概率有区分度
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            out["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            out["auc"] = np.nan
    else:
        out["auc"] = np.nan
    # Brier 分数：概率预测的均方误差，越小越准（区分度 + 校准的联合度量）
    if y_proba is not None:
        out["brier"] = brier_score_loss(y_true, np.asarray(y_proba, dtype=float))
    else:
        out["brier"] = np.nan
    return out


def reliability_curve(y_true, y_proba, n_bins: int = 5) -> pd.DataFrame:
    """可靠性曲线（校准）数据：按预测概率分箱，比较平均预测 vs 实际频率。

    完美校准时，"预测扩张概率 0.7 的那些月份，约 70% 真的扩张"，即点落在对角线上。
    小样本用较少箱（默认 5），否则每箱点太少、噪声淹没信号。
    返回列：bin_mid, mean_pred, frac_pos, count。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_proba, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_mid": (edges[b] + edges[b + 1]) / 2,
                "mean_pred": float(y_proba[mask].mean()),
                "frac_pos": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def confusion(y_true, y_pred) -> pd.DataFrame:
    """混淆矩阵。行=真实，列=预测；标注收缩(0)/扩张(1)。"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["真实:收缩", "真实:扩张"],
        columns=["预测:收缩", "预测:扩张"],
    )


def logreg_coefficients(model, feature_names: list[str]) -> pd.Series:
    """逻辑回归系数（在全样本上重训以做可解释性展示）。"""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    coefs = pd.Series(clf.coef_.ravel(), index=feature_names)
    return coefs.sort_values(key=abs, ascending=False)


def tree_importances(model, feature_names: list[str]) -> pd.Series:
    """树模型特征重要性。"""
    imp = pd.Series(model.feature_importances_, index=feature_names)
    return imp.sort_values(ascending=False)
