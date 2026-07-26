"""ablation.py — 特征组消融实验：量化"哪一类经济信号真正带来样本外增益"。

做法：把特征按经济逻辑分组（PMI 惯性 / 货币金融 / 价格 / 实体活动 / 信用），
每次**剔除一整组**特征，用同一模型跑完整 walk-forward，比较样本外 AUC 相对
"全特征"的变化 ΔAUC。

- ΔAUC 显著为负 → 该组信息有用（拿掉就变差）。
- ΔAUC ≈ 0 或为正 → 该组在本样本上贡献不大甚至是噪声（小样本常见）。

这比"看一次特征重要性"更可信：重要性是全样本、单模型的内部量；消融是
样本外、端到端的因果式验证，直接回答"少了它预测会不会真的变差"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config, dataset, evaluate, validation

# 特征前缀 → 经济组。按列名前缀归组（lag/mom 派生列一并纳入其母指标组）。
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "PMI惯性": ("pmi",),
    "货币金融": ("m1_yoy", "m2_yoy", "m1_m2_gap"),
    "价格": ("cpi_yoy", "ppi_yoy"),
    "实体活动": ("ip_yoy",),
    "信用": ("credit_yoy",),
    "外需": ("export_yoy",),
    "市场": ("hs300_ret", "hs300_vol"),
}


def _cols_in_group(feat_cols: list[str], prefixes: tuple[str, ...]) -> list[str]:
    """返回属于该组的列：列名等于前缀，或以 '<前缀>_' 开头（lag/mom 派生）。"""
    hit = []
    for c in feat_cols:
        for p in prefixes:
            if c == p or c.startswith(p + "_"):
                hit.append(c)
                break
    return hit


def _oos_auc(df: pd.DataFrame, feat_cols: list[str], target: str) -> float:
    """在给定特征子集上跑 walk-forward，返回随机森林样本外 AUC。

    与 train.run_walk_forward 同构，但只跑单个（随机森林）模型、可换特征列，
    专供消融循环高效复用。
    """
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)
    folds = validation.walk_forward_splits(len(df))
    y_true, y_proba = [], []
    for fold in folds:
        ytr = y[fold.train_idx]
        if len(np.unique(ytr)) < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        )
        model.fit(X[fold.train_idx], ytr)
        proba = model.predict_proba(X[fold.test_idx])[:, 1]
        y_proba.extend(proba.tolist())
        y_true.extend(y[fold.test_idx].tolist())
    m = evaluate.classification_metrics(
        y_true, (np.array(y_proba) >= 0.5).astype(int), y_proba
    )
    return m["auc"]


def run_ablation(df: pd.DataFrame, target: str = "y_expansion") -> pd.DataFrame:
    """逐组剔除做消融，返回 DataFrame：组名、剩余特征数、AUC、ΔAUC。"""
    feat_cols = dataset.feature_columns(df)
    base_auc = _oos_auc(df, feat_cols, target)

    rows = [
        {
            "剔除组": "（全特征基准）",
            "剩余特征数": len(feat_cols),
            "oos_auc": base_auc,
            "delta_auc": 0.0,
        }
    ]
    for group, prefixes in FEATURE_GROUPS.items():
        drop = set(_cols_in_group(feat_cols, prefixes))
        remain = [c for c in feat_cols if c not in drop]
        if not remain:
            continue
        auc = _oos_auc(df, remain, target)
        rows.append(
            {
                "剔除组": group,
                "剩余特征数": len(remain),
                "oos_auc": auc,
                "delta_auc": auc - base_auc,
            }
        )
    out = pd.DataFrame(rows)
    # 按 ΔAUC 升序：最负（最不可或缺）在前
    body = out.iloc[1:].sort_values("delta_auc")
    return pd.concat([out.iloc[:1], body], ignore_index=True)


if __name__ == "__main__":
    df = dataset.build_dataset()
    print(run_ablation(df).round(4).to_string(index=False))
