"""train.py — 训练 + 滚动回测总驱动。

流程：build_dataset → walk-forward 逐期重训所有模型 → 收集样本外预测
→ 分类评估 + 混淆矩阵 + 特征重要性 → 经济价值回测 → 存图表与指标。

一条命令跑通整个研究管线：`python -m pmi_nowcast.train`
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import backtest, config, dataset, evaluate, models, validation

warnings.filterwarnings("ignore")


def run_walk_forward(df: pd.DataFrame, target: str = "y_expansion") -> pd.DataFrame:
    """对所有模型做扩展窗口 walk-forward，返回逐期样本外预测长表。

    列：asof_month, model, y_true, y_pred, y_proba
    """
    feat_cols = dataset.feature_columns(df)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)
    months = df.index

    pmi_idx = feat_cols.index("pmi")
    folds = validation.walk_forward_splits(len(df))
    validation.assert_no_leakage(folds, config.PURGE_MONTHS, config.EMBARGO_MONTHS)
    print(f"[train] {len(folds)} 折 walk-forward，防泄漏自检通过")

    records = []
    for fold in folds:
        model_dict = models.make_models(pmi_col_index=pmi_idx)
        Xtr, ytr = X[fold.train_idx], y[fold.train_idx]
        Xte, yte = X[fold.test_idx], y[fold.test_idx]
        # 训练期若只有单一类别，跳过（AUC/拟合无意义）
        if len(np.unique(ytr)) < 2:
            continue
        for name, model in model_dict.items():
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]
            pred = (proba >= 0.5).astype(int)
            for j, pos in enumerate(fold.test_idx):
                records.append(
                    {
                        "asof_month": months[pos],
                        "model": name,
                        "y_true": int(yte[j]),
                        "y_pred": int(pred[j]),
                        "y_proba": float(proba[j]),
                    }
                )
    return pd.DataFrame(records)


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    """按模型汇总样本外分类指标。"""
    rows = []
    for name, g in preds.groupby("model"):
        m = evaluate.classification_metrics(g["y_true"], g["y_pred"], g["y_proba"])
        m["model"] = name
        m["n"] = len(g)
        rows.append(m)
    out = pd.DataFrame(rows).set_index("model")
    cols = ["n", "accuracy", "precision", "recall", "f1", "auc"]
    return out[cols].sort_values("auc", ascending=False)


def fit_full_for_interpretability(df: pd.DataFrame, target: str = "y_expansion"):
    """在全样本上重训 logreg / rf，仅用于展示系数与特征重要性。"""
    feat_cols = dataset.feature_columns(df)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)
    mdict = models.make_models()
    out = {}
    lr = mdict["logreg"].fit(X, y)
    out["logreg_coef"] = evaluate.logreg_coefficients(lr, feat_cols)
    rf = mdict["random_forest"].fit(X, y)
    out["rf_importance"] = evaluate.tree_importances(rf, feat_cols)
    return out


def main():
    df = dataset.build_dataset()
    dataset.save_processed(df)
    print(f"[train] dataset shape={df.shape}, span={df.index.min().date()}..{df.index.max().date()}")

    preds = run_walk_forward(df, target="y_expansion")
    summary = summarize(preds)
    print("\n=== 样本外分类指标（walk-forward）===")
    print(summary.round(3).to_string())

    # 保存
    preds.to_csv(config.OUTPUT_DIR / "oos_predictions.csv", index=False, encoding="utf-8-sig")
    summary.round(4).to_csv(config.OUTPUT_DIR / "metrics_summary.csv", encoding="utf-8-sig")

    # 可解释性
    interp = fit_full_for_interpretability(df)
    print("\n=== 逻辑回归系数（Top-8，全样本）===")
    print(interp["logreg_coef"].head(8).round(3).to_string())
    interp["logreg_coef"].round(4).to_csv(config.OUTPUT_DIR / "logreg_coef.csv", encoding="utf-8-sig")
    interp["rf_importance"].round(4).to_csv(config.OUTPUT_DIR / "rf_importance.csv", encoding="utf-8-sig")

    # 选最佳非朴素模型做混淆矩阵与回测
    ml_models = [m for m in summary.index if not m.startswith("naive")]
    best = summary.loc[ml_models, "auc"].idxmax() if ml_models else summary.index[0]
    best_preds = preds[preds["model"] == best].set_index("asof_month").sort_index()
    print(f"\n=== 最佳模型 '{best}' 混淆矩阵 ===")
    print(evaluate.confusion(best_preds["y_true"], best_preds["y_pred"]).to_string())

    # 经济价值回测
    try:
        mkt = backtest.load_hs300_monthly_returns()
        bt = backtest.run_backtest(best_preds["y_pred"], mkt, cost_bps=10.0)
        strat = backtest.performance_stats(bt["strategy_ret"])
        bh = backtest.performance_stats(bt["buyhold_ret"])
        print(f"\n=== 经济价值回测（{best} 信号 vs 买入持有，含 10bp 单边成本）===")
        stat_df = pd.DataFrame({"策略": strat, "买入持有": bh}).T
        print(stat_df.round(3).to_string())

        # 交易成本敏感性：策略优势随成本侵蚀（正面回应"未计成本"局限）
        print("\n=== 交易成本敏感性（策略年化收益）===")
        sweep = []
        for c in (0.0, 5.0, 10.0, 20.0, 30.0):
            s = backtest.run_backtest(best_preds["y_pred"], mkt, cost_bps=c)
            st = backtest.performance_stats(s["strategy_ret"])
            n_trades = int(s["position"].diff().abs().fillna(s["position"].abs()).sum())
            sweep.append(
                {"cost_bps": c, "annual_return": st["annual_return"], "sharpe": st["sharpe"], "换手次数": n_trades}
            )
        sweep_df = pd.DataFrame(sweep).set_index("cost_bps")
        print(sweep_df.round(4).to_string())
        sweep_df.round(6).to_csv(config.OUTPUT_DIR / "cost_sensitivity.csv", encoding="utf-8-sig")

        bt.to_csv(config.OUTPUT_DIR / "backtest.csv", encoding="utf-8-sig")
        _plot_results(summary, bt, best)
    except Exception as e:  # noqa: BLE001 - 行情接口不稳不应中断主流程
        print(f"[train] 回测跳过（行情数据获取失败）: {type(e).__name__}: {str(e)[:80]}")

    print(f"\n[train] 全部产出已存至 {config.OUTPUT_DIR}")


def _plot_results(summary: pd.DataFrame, bt: pd.DataFrame, best: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt.index, bt["strategy_nav"], label=f"PMI信号择时 ({best})", linewidth=2)
    ax.plot(bt.index, bt["buyhold_nav"], label="买入持有 沪深300", linewidth=2, alpha=0.7)
    ax.set_title("经济价值回测：净值曲线对比")
    ax.set_ylabel("净值（初始=1）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "nav_curve.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
