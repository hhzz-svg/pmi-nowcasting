"""evaluate.py — 样本外分类评估 + 特征重要性。

所有指标都在 walk-forward 样本外预测上计算，绝非训练集。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    """标准分类指标。AUC 对不平衡数据比 accuracy 更可靠。"""
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
    return out


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
