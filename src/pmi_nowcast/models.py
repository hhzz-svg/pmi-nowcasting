"""models.py — 统一模型接口（LogReg / RF / XGBoost / naive baseline）。

所有模型走同一 (.fit / .predict_proba) 协议，train.py 只需循环调用，
"加新模型只需几行"。含关键的朴素基准——ML 模型若打不过它就是没学到东西。

标准化说明：LogReg 对量纲敏感，用 Pipeline 内嵌 StandardScaler，scaler 只在
每折训练集上 fit（sklearn Pipeline 保证），故不泄漏测试期统计量。
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:  # xgboost 为进阶可选依赖
    _HAS_XGB = False


class NaiveMajority:
    """朴素基准：永远预测训练集多数类（对 PMI 即"永远预测扩张"）。"""

    def fit(self, X, y):
        y = np.asarray(y)
        self.majority_ = int(round(y.mean()))
        self.p_ = float(y.mean())
        return self

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.p_)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return np.full(len(X), self.majority_)


class NaivePersistence:
    """朴素基准：预测"下月与本月同向"，即下月仍扩张 iff 本月 PMI>50。

    需要在预测时拿到当期 pmi 列，故 fit 时记录列位置。
    """

    def __init__(self, pmi_col_index: int):
        self.pmi_col_index = pmi_col_index

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        pmi_now = X[:, self.pmi_col_index]
        p1 = (pmi_now > config.EXPANSION_THRESHOLD).astype(float)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SoftEnsemble:
    """概率软融合：等权平均若干子模型的 predict_proba[:,1]。

    动机：naive_persistence 抓住了 PMI 强惯性（AUC 高），ML 模型抓非线性/多变量
    交互但单独打不过它。把两者概率等权平均，看能否兼收并蓄、稳定超过单一模型。
    这是一个诚实的"能否做得更好"的尝试，而非调参凑指标。

    子模型在 fit 时各自独立训练，故不引入额外泄漏（每折仍只见训练集）。
    """

    def __init__(self, members: list):
        self.members = members

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        ps = np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)
        return np.column_stack([1 - ps, ps])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_models(pmi_col_index: int | None = None) -> dict:
    """构造模型字典。小样本下所有模型都开较强正则 / 限制复杂度。"""
    models: dict = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=0.5,  # 较强正则（L2 为默认），小样本防过拟合
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,  # 浅树，小样本防过拟合
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "naive_majority": NaiveMajority(),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
        )
    if pmi_col_index is not None:
        models["naive_persistence"] = NaivePersistence(pmi_col_index)
        # 混合集成：随机森林（多变量非线性）+ 持续基准（PMI 强惯性）等权融合。
        rf_member = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )
        models["ensemble"] = SoftEnsemble(
            [rf_member, NaivePersistence(pmi_col_index)]
        )
    return models


def has_xgboost() -> bool:
    return _HAS_XGB
