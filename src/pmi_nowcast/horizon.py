"""horizon.py — ★多步时距实验：预测 t+1 / t+2 / t+3 月荣枯，可预测性如何随距离衰减？

单一 h=1 的结论可能是"惯性红利"的特例：PMI 一个月内很少跨越荣枯线，
持续基准因此近乎免费地拿高分。把预测距离拉长到 2、3 个月，惯性红利衰减，
两个问题随之浮现：

  1) 整体可预测性衰减多快？（AUC-vs-h 曲线的斜率）
  2) 持续基准的相对优势是否随 h 缩小？——若 ML 在长时距上反超，
     说明宏观特征携带的是"超越惯性"的慢变量信息。

方法论细节：h 步标签 y_t 依赖 PMI(t+h)，训练集末尾与测试点的信息重叠
随 h 变长，故 purge 取 max(PURGE_MONTHS, h)，保证 gap ≥ h，防泄漏口径
与主实验同样严格。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, dataset, evaluate, models, validation

HORIZONS = (1, 2, 3)


def target_for_horizon(h: int) -> str:
    """h 步时距对应的标签列名。h=1 即主任务标签。"""
    return "y_expansion" if h == 1 else f"y_expansion_h{h}"


def run_horizon_walk_forward(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """对时距 h 跑全模型 walk-forward，返回逐期样本外预测长表。

    与 train.run_walk_forward 同构，两点不同：
    - 标签换成 y_expansion_h{h}，并剔除该标签缺失的尾部行；
    - purge 随 h 扩大（标签前视 h 个月，训练/测试 gap 必须 ≥ h）。
    """
    target = target_for_horizon(h)
    sub = df.dropna(subset=[target])
    feat_cols = dataset.feature_columns(sub)
    X = sub[feat_cols].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=int)
    months = sub.index
    pmi_idx = feat_cols.index("pmi")

    purge = max(config.PURGE_MONTHS, h)
    folds = validation.walk_forward_splits(len(sub), purge=purge)
    validation.assert_no_leakage(folds, purge, config.EMBARGO_MONTHS)

    records = []
    for fold in folds:
        ytr = y[fold.train_idx]
        if len(np.unique(ytr)) < 2:
            continue
        model_dict = models.make_models(pmi_col_index=pmi_idx)
        for name, model in model_dict.items():
            model.fit(X[fold.train_idx], ytr)
            proba = model.predict_proba(X[fold.test_idx])[:, 1]
            for j, pos in enumerate(fold.test_idx):
                records.append(
                    {
                        "asof_month": months[pos],
                        "horizon": h,
                        "model": name,
                        "y_true": int(y[pos]),
                        "y_pred": int(proba[j] >= 0.5),
                        "y_proba": float(proba[j]),
                    }
                )
    return pd.DataFrame(records)


def run_horizon_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """对每个时距跑 walk-forward，汇总成 模型 × 时距 的样本外 AUC 宽表。"""
    rows = []
    for h in HORIZONS:
        preds = run_horizon_walk_forward(df, h)
        if preds.empty:
            continue
        for name, g in preds.groupby("model"):
            m = evaluate.classification_metrics(g["y_true"], g["y_pred"], g["y_proba"])
            rows.append(
                {
                    "model": name,
                    "horizon": h,
                    "n": len(g),
                    "auc": m["auc"],
                    "accuracy": m["accuracy"],
                }
            )
    long = pd.DataFrame(rows)
    wide = long.pivot(index="model", columns="horizon", values="auc")
    wide.columns = [f"h={c}" for c in wide.columns]
    wide = wide.sort_values("h=1", ascending=False)
    return wide, long


def plot_horizon(wide: pd.DataFrame, path) -> None:
    """AUC-vs-时距 衰减曲线：每模型一条线，直观展示惯性红利如何随 h 消退。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    hs = [int(c.split("=")[1]) for c in wide.columns]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, row in wide.iterrows():
        style = "o--" if name.startswith("naive") else "o-"
        ax.plot(hs, row.values, style, label=name)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=0.8, label="无信息(0.5)")
    ax.set_xticks(hs)
    ax.set_xlabel("预测时距 h（月）")
    ax.set_ylabel("样本外 AUC")
    ax.set_title("多步时距：可预测性随预测距离的衰减")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    df = dataset.build_dataset()
    wide, long = run_horizon_experiment(df)
    print(wide.round(4).to_string())
