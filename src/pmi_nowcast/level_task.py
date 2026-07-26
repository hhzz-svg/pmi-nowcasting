"""level_task.py — ★第三任务：PMI 点位回归（预测下月 PMI 的具体数值）。

荣枯（>50）与方向（较上月升降）都是二分类问法；点位回归回答更细的问题：
"下月 PMI 是 49.8 还是 50.2？"——对荣枯线附近的择时决策，这 0.4 的差别
就是全部。同时它给了持续基准第三次上场机会：ŷ = 本月 PMI（预测零变化）。
PMI 月度变化的标准差本就只有约 1 个点，打败"预测不变"远比听上去难。

评估：
  - MAE / RMSE / 偏差（平均符号误差）——点位精度；
  - 隐含荣枯准确率（ŷ>50 vs 真值>50）——回归输出翻译回分类的一致性检查；
  - 配对自助 Diebold-Mariano 检验（significance.bootstrap_loss_diff）：
    ML 的 MAE 优势是否显著异于持续基准。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, dataset, significance, validation

try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


class PersistenceRegressor:
    """持续基准：预测下月 PMI = 本月 PMI（零变化）。"""

    def __init__(self, pmi_col_index: int):
        self.pmi_col_index = pmi_col_index

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float)[:, self.pmi_col_index]


def make_regressors(pmi_col_index: int) -> dict:
    """回归模型字典。与分类任务同一克制哲学：小样本、强正则、浅树。"""
    regs: dict = {
        "naive_persistence": PersistenceRegressor(pmi_col_index),
        "ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=10.0, random_state=config.RANDOM_STATE)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            random_state=config.RANDOM_STATE,
        ),
    }
    if _HAS_XGB:
        regs["xgboost"] = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=config.RANDOM_STATE,
        )
    return regs


def run_level_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """点位回归 walk-forward：目标 = pmi_next（连续值）。

    折与分类任务完全同参（同 purge/embargo/最小训练窗），结果可逐月对照。
    返回长表：asof_month, model, y_true, y_pred。
    """
    feat_cols = dataset.feature_columns(df)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["pmi_next"].to_numpy(dtype=float)
    months = df.index
    pmi_idx = feat_cols.index("pmi")

    folds = validation.walk_forward_splits(len(df))
    validation.assert_no_leakage(folds, config.PURGE_MONTHS, config.EMBARGO_MONTHS)

    records = []
    for fold in folds:
        regs = make_regressors(pmi_col_index=pmi_idx)
        for name, reg in regs.items():
            reg.fit(X[fold.train_idx], y[fold.train_idx])
            pred = np.asarray(reg.predict(X[fold.test_idx]), dtype=float)
            for j, pos in enumerate(fold.test_idx):
                records.append(
                    {
                        "asof_month": months[pos],
                        "model": name,
                        "y_true": float(y[pos]),
                        "y_pred": float(pred[j]),
                    }
                )
    return pd.DataFrame(records)


def summarize_level(preds: pd.DataFrame) -> pd.DataFrame:
    """按模型汇总点位回归指标 + 隐含荣枯准确率。"""
    thr = config.EXPANSION_THRESHOLD
    rows = []
    for name, g in preds.groupby("model"):
        err = g["y_pred"] - g["y_true"]
        rows.append(
            {
                "model": name,
                "n": len(g),
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "bias": float(err.mean()),
                "隐含荣枯acc": float(((g["y_pred"] > thr) == (g["y_true"] > thr)).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("mae")


def run_level_significance(preds: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """每个 ML 回归器 vs 持续基准的 MAE 差配对自助检验（DM 风格）。

    diff = mean(|e_naive| - |e_model|)：正 → 模型比基准准。
    """
    pv = preds.pivot_table(index="asof_month", columns="model", values=["y_true", "y_pred"])
    y = pv["y_true"]["naive_persistence"].to_numpy()
    loss_naive = np.abs(pv["y_pred"]["naive_persistence"].to_numpy() - y)
    rows = []
    for name in summary.index:
        if name == "naive_persistence":
            continue
        loss_m = np.abs(pv["y_pred"][name].to_numpy() - y)
        r = significance.bootstrap_loss_diff(loss_m, loss_naive)
        rows.append(
            {
                "model": name,
                "mae_gain_vs_naive": r["diff"],
                "ci_lo": r["lo"],
                "ci_hi": r["hi"],
                "p_value": r["p_value"],
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("mae_gain_vs_naive", ascending=False)


def plot_level_forecast(preds: pd.DataFrame, best: str, path) -> None:
    """真实 PMI vs 最佳回归器 vs 持续基准的逐月对照 + 荣枯线。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    pv = preds.pivot_table(index="asof_month", columns="model", values="y_pred")
    truth = preds[preds["model"] == best].set_index("asof_month")["y_true"].sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(truth.index, truth, "k-", linewidth=1.8, label="真实 PMI")
    ax.plot(pv.index, pv[best], "-", linewidth=1.2, label=f"{best} 预测")
    ax.plot(pv.index, pv["naive_persistence"], "--", linewidth=1.0, alpha=0.7, label="持续基准（=本月值）")
    ax.axhline(config.EXPANSION_THRESHOLD, color="#c0392b", linestyle=":", linewidth=1, label="荣枯线 50")
    ax.set_ylabel("下月 PMI")
    ax.set_title("第三任务：PMI 点位回归（walk-forward 样本外）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    df = dataset.build_dataset()
    preds = run_level_walk_forward(df)
    summary = summarize_level(preds)
    print(summary.round(3).to_string())
    print(run_level_significance(preds, summary).round(4).to_string())
