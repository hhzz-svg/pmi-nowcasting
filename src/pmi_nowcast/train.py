"""train.py — 训练 + 滚动回测总驱动。

流程：build_dataset → walk-forward 逐期重训所有模型 → 收集样本外预测
→ 分类评估 + 混淆矩阵 + 特征重要性 → 经济价值回测 → 存图表与指标。

一条命令跑通整个研究管线：`python -m pmi_nowcast.train`
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import (
    ablation,
    backtest,
    config,
    dataset,
    evaluate,
    horizon,
    level_task,
    models,
    shap_analysis,
    significance,
    validation,
)

warnings.filterwarnings("ignore")


def run_walk_forward(
    df: pd.DataFrame, target: str = "y_expansion", scheme: str = "expanding"
) -> pd.DataFrame:
    """对所有模型做 walk-forward，返回逐期样本外预测长表。

    scheme: "expanding"（扩展窗口，默认）或 "rolling"（固定长度滚动窗口）。
    列：asof_month, model, y_true, y_pred, y_proba
    """
    feat_cols = dataset.feature_columns(df)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)
    months = df.index

    pmi_idx = feat_cols.index("pmi")
    if scheme == "rolling":
        folds = validation.rolling_window_splits(len(df))
    else:
        folds = validation.walk_forward_splits(len(df))
    validation.assert_no_leakage(folds, config.PURGE_MONTHS, config.EMBARGO_MONTHS)
    print(f"[train] {len(folds)} 折 walk-forward（{scheme}），防泄漏自检通过")

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
    cols = ["n", "accuracy", "precision", "recall", "f1", "auc", "brier"]
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

    # 统计显著性：ML 与朴素持续基准的差异是真实的还是噪声？
    _run_significance(preds, summary)

    # 概率校准：可靠性曲线（nowcast 输出概率，校准很重要）
    _plot_calibration(preds, summary)

    # 特征组消融：哪一类经济信号真正带来样本外增益
    try:
        abl = ablation.run_ablation(df, target="y_expansion")
        print("\n=== 特征组消融（剔除一组，看样本外 AUC 变化）===")
        print(abl.round(4).to_string(index=False))
        abl.round(6).to_csv(config.OUTPUT_DIR / "ablation.csv", index=False, encoding="utf-8-sig")
        _plot_ablation(abl)
    except Exception as e:  # noqa: BLE001
        print(f"[train] 消融跳过: {type(e).__name__}: {str(e)[:80]}")

    # 分时期表现拆解：不同宏观阶段（2015 放缓 / 2020 疫情 / 2022 冲击等）是否稳健
    _run_regime_breakdown(preds, best)

    # 第二任务：方向预测（下月 PMI 是否较本月上行）
    _run_direction_task(df)

    # 第三任务：PMI 点位回归（预测具体数值 + DM 显著性）
    _run_level_task(df)

    # 多步时距：h=1/2/3 月，可预测性衰减曲线
    _run_horizon_experiment(df)

    # 验证稳健性：滚动窗口 vs 扩展窗口
    _run_window_comparison(df, preds)

    # SHAP 特征贡献（进阶可解释性；shap 未装则自动跳过）
    try:
        shap_imp = shap_analysis.run_shap(df, target="y_expansion")
        print("\n=== SHAP 平均绝对贡献（Top-8）===")
        print(shap_imp.head(8).round(4).to_string())
    except Exception as e:  # noqa: BLE001
        print(f"[train] SHAP 跳过: {type(e).__name__}: {str(e)[:80]}")

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


def _run_significance(preds: pd.DataFrame, summary: pd.DataFrame):
    """检验 AUC 最高的 ML 模型与 naive_persistence 的差异是否统计显著。

    朴素持续基准是本项目的"擂主"。若某 ML 模型 AUC 名义更高，必须回答：
    这点优势跨得过抽样噪声吗？给出 bootstrap 置信区间 + 配对差检验 + McNemar。
    """
    if "naive_persistence" not in summary.index:
        return
    ml = [m for m in summary.index if not m.startswith("naive")]
    if not ml:
        return
    challenger = summary.loc[ml, "auc"].idxmax()  # 最强 ML 挑战者

    pv = preds.pivot_table(
        index="asof_month", columns="model", values=["y_true", "y_pred", "y_proba"]
    )
    y_true = pv["y_true"]["naive_persistence"].to_numpy()
    pa = pv["y_proba"][challenger].to_numpy()
    pb = pv["y_proba"]["naive_persistence"].to_numpy()

    ci_a = significance.bootstrap_auc_ci(y_true, pa)
    ci_b = significance.bootstrap_auc_ci(y_true, pb)
    diff = significance.bootstrap_auc_diff(y_true, pa, pb)
    mc = significance.mcnemar_test(
        y_true, pv["y_pred"][challenger].to_numpy(), pv["y_pred"]["naive_persistence"].to_numpy()
    )

    print(f"\n=== 统计显著性：{challenger} vs naive_persistence ===")
    print(f"  {challenger:18s} AUC={ci_a['auc']:.3f}  95%CI[{ci_a['lo']:.3f}, {ci_a['hi']:.3f}]")
    print(f"  naive_persistence  AUC={ci_b['auc']:.3f}  95%CI[{ci_b['lo']:.3f}, {ci_b['hi']:.3f}]")
    print(f"  ΔAUC(ML-naive)={diff['diff']:+.3f}  95%CI[{diff['lo']:+.3f}, {diff['hi']:+.3f}]  p={diff['p_value']:.3f}")
    print(f"  McNemar: ML对/naive错={mc['n_ab']}, ML错/naive对={mc['n_ba']}, p={mc['p_value']:.3f}")
    verdict = "差异不显著（CI 跨 0）——'打平'结论成立" if diff["lo"] <= 0 <= diff["hi"] else "差异显著"
    print(f"  结论：{verdict}")

    rows = [
        {"模型": challenger, "auc": ci_a["auc"], "ci_lo": ci_a["lo"], "ci_hi": ci_a["hi"]},
        {"模型": "naive_persistence", "auc": ci_b["auc"], "ci_lo": ci_b["lo"], "ci_hi": ci_b["hi"]},
    ]
    sig_df = pd.DataFrame(rows)
    sig_df["delta_vs_naive"] = [diff["diff"], 0.0]
    sig_df["diff_p_value"] = [diff["p_value"], np.nan]
    sig_df["mcnemar_p"] = [mc["p_value"], np.nan]
    sig_df.round(4).to_csv(config.OUTPUT_DIR / "significance.csv", index=False, encoding="utf-8-sig")


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


def _plot_calibration(preds: pd.DataFrame, summary: pd.DataFrame):
    """画连续概率模型的可靠性曲线（校准图）。朴素多数类概率恒定，不含信息，略过。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 只画有概率区分度的模型（naive_majority 概率恒定，naive_persistence 概率仅取 0/1）
    show = [m for m in summary.index if m not in ("naive_majority", "naive_persistence")]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="完美校准")
    for name in show:
        g = preds[preds["model"] == name]
        rc = evaluate.reliability_curve(g["y_true"], g["y_proba"], n_bins=5)
        if rc.empty:
            continue
        brier = summary.loc[name, "brier"]
        ax.plot(rc["mean_pred"], rc["frac_pos"], "o-", label=f"{name} (Brier={brier:.3f})")
    ax.set_xlabel("平均预测扩张概率")
    ax.set_ylabel("实际扩张频率")
    ax.set_title("概率校准（可靠性曲线，walk-forward 样本外）")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "calibration_curve.png", dpi=120)
    plt.close(fig)


def _plot_ablation(abl: pd.DataFrame):
    """画特征组消融的 ΔAUC 条形图（相对全特征基准）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    body = abl[abl["剔除组"] != "（全特征基准）"].copy()
    colors = ["#c0392b" if d < 0 else "#27ae60" for d in body["delta_auc"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(body["剔除组"], body["delta_auc"], color=colors)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("ΔAUC（剔除该组后 - 全特征基准）")
    ax.set_title("特征组消融：负值=该组不可或缺，正值=拿掉反而更好（噪声）")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "ablation.png", dpi=120)
    plt.close(fig)


def _run_window_comparison(df: pd.DataFrame, expanding_preds: pd.DataFrame):
    """对比扩展窗口 vs 固定滚动窗口的样本外表现。

    扩展窗口用尽全部历史（假设关系稳定）；滚动窗口只用近 N 年（假设关系随时代漂移，
    更早的样本反而是噪声）。宏观关系是否有"结构性漂移"，用这两条窗口的差异来体察。
    对每个模型分别在两种窗口下算 AUC，并列展示。

    扩展窗口结果直接复用主流程已算好的 expanding_preds（二者完全一致），
    只补跑滚动窗口一遍，避免重复一次昂贵的全模型 walk-forward。
    """
    try:
        exp = summarize(expanding_preds)
        roll = summarize(run_walk_forward(df, target="y_expansion", scheme="rolling"))
    except Exception as e:  # noqa: BLE001
        print(f"[train] 窗口对比跳过: {type(e).__name__}: {str(e)[:80]}")
        return
    cmp = pd.DataFrame(
        {
            "扩展窗口_auc": exp["auc"],
            "滚动窗口_auc": roll["auc"],
        }
    )
    cmp["Δ(滚动-扩展)"] = cmp["滚动窗口_auc"] - cmp["扩展窗口_auc"]
    cmp = cmp.sort_values("扩展窗口_auc", ascending=False)
    print("\n=== 扩展窗口 vs 固定滚动窗口（样本外 AUC）===")
    print(cmp.round(4).to_string())
    cmp.round(4).to_csv(config.OUTPUT_DIR / "window_comparison.csv", encoding="utf-8-sig")


def _run_regime_breakdown(preds: pd.DataFrame, best: str):
    """分时期拆解：同一模型在不同宏观阶段的样本外表现是否稳定？

    整体一个 AUC 会掩盖"某些时期特别好、某些时期彻底失灵"。按中国宏观史
    划分几个有经济含义的阶段，分别算准确率与 AUC，暴露模型的稳健性/脆弱性。

    时段（按预测基准月 asof_month）：
      危机后复苏(08-11) / 增速换挡(12-14) / 股灾供改(15-16) /
      贸易战(18-19) / 疫情冲击(20) / 后疫情反复(21-22) / 弱复苏(23-)
    """
    regimes = [
        ("危机后复苏(08-11)", "2008-01", "2011-12"),
        ("增速换挡(12-14)", "2012-01", "2014-12"),
        ("股灾供改(15-16)", "2015-01", "2016-12"),
        ("平稳(17)", "2017-01", "2017-12"),
        ("贸易战(18-19)", "2018-01", "2019-12"),
        ("疫情冲击(20)", "2020-01", "2020-12"),
        ("后疫情反复(21-22)", "2021-01", "2022-12"),
        ("弱复苏(23-)", "2023-01", "2025-12"),
    ]
    g = preds[preds["model"] == best].copy()
    g["asof_month"] = pd.to_datetime(g["asof_month"])
    rows = []
    for name, lo, hi in regimes:
        seg = g[(g["asof_month"] >= lo) & (g["asof_month"] <= hi)]
        if len(seg) == 0:
            continue
        m = evaluate.classification_metrics(seg["y_true"], seg["y_pred"], seg["y_proba"])
        rows.append(
            {
                "时期": name,
                "n": len(seg),
                "accuracy": m["accuracy"],
                "auc": m["auc"],
                "扩张占比": float(seg["y_true"].mean()),
            }
        )
    if not rows:
        return
    reg_df = pd.DataFrame(rows)
    print(f"\n=== 分时期表现拆解（{best}，样本外）===")
    print(reg_df.round(3).to_string(index=False))
    reg_df.round(4).to_csv(config.OUTPUT_DIR / "regime_breakdown.csv", index=False, encoding="utf-8-sig")
    _plot_regime(reg_df, best)


def _plot_regime(reg_df: pd.DataFrame, best: str):
    """分时期准确率条形图，叠加整体基准线，直观看哪些阶段模型失灵。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    overall = reg_df["accuracy"].mean()
    colors = ["#27ae60" if a >= 0.5 else "#c0392b" for a in reg_df["accuracy"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(reg_df["时期"], reg_df["accuracy"], color=colors)
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="抛硬币(0.5)")
    ax.axhline(overall, color="#2980b9", linestyle=":", linewidth=1.2, label=f"各期均值({overall:.2f})")
    ax.set_ylabel("样本外准确率")
    ax.set_title(f"分时期表现拆解（{best}）：绿=胜过抛硬币，红=失灵")
    ax.set_ylim(0, 1)
    ax.legend()
    for i, (a, n) in enumerate(zip(reg_df["accuracy"], reg_df["n"])):
        ax.text(i, a + 0.02, f"n={n}", ha="center", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "regime_breakdown.png", dpi=120)
    plt.close(fig)


def _run_direction_task(df: pd.DataFrame):
    """第二任务：预测下月 PMI 是否较本月上行（y_direction）。

    扩张/收缩（相对荣枯线 50）与上行/下行（相对上月）是两种不同的景气问法。
    复用同一 walk-forward 管线，几乎零额外代码即把实验内容翻倍。
    """
    try:
        preds = run_walk_forward(df, target="y_direction")
        if preds.empty:
            print("\n[train] 方向任务样本不足，跳过")
            return
        summary = summarize(preds)
        print("\n=== 第二任务：方向预测（下月 PMI 上行 vs 下行，walk-forward）===")
        print(summary.round(3).to_string())
        preds.to_csv(config.OUTPUT_DIR / "oos_predictions_direction.csv", index=False, encoding="utf-8-sig")
        summary.round(4).to_csv(config.OUTPUT_DIR / "metrics_direction.csv", encoding="utf-8-sig")
    except Exception as e:  # noqa: BLE001
        print(f"[train] 方向任务跳过: {type(e).__name__}: {str(e)[:80]}")


def _run_level_task(df: pd.DataFrame):
    """第三任务：点位回归。持续基准（预测零变化）第三次上场当擂主。"""
    try:
        preds = level_task.run_level_walk_forward(df)
        if preds.empty:
            print("\n[train] 点位回归样本不足，跳过")
            return
        summary = level_task.summarize_level(preds)
        print("\n=== 第三任务：PMI 点位回归（walk-forward 样本外）===")
        print(summary.round(3).to_string())
        sig = level_task.run_level_significance(preds, summary)
        print("\n--- MAE 优势 vs 持续基准（配对自助 DM 检验，正=模型更准）---")
        print(sig.round(4).to_string())
        preds.to_csv(config.OUTPUT_DIR / "oos_predictions_level.csv", index=False, encoding="utf-8-sig")
        summary.round(4).to_csv(config.OUTPUT_DIR / "level_metrics.csv", encoding="utf-8-sig")
        sig.round(4).to_csv(config.OUTPUT_DIR / "level_significance.csv", encoding="utf-8-sig")
        best = summary.index[0]
        level_task.plot_level_forecast(preds, best, config.OUTPUT_DIR / "level_forecast.png")
    except Exception as e:  # noqa: BLE001
        print(f"[train] 点位回归跳过: {type(e).__name__}: {str(e)[:80]}")


def _run_horizon_experiment(df: pd.DataFrame):
    """多步时距：AUC 随预测距离 h 的衰减曲线。"""
    try:
        wide, long = horizon.run_horizon_experiment(df)
        print("\n=== 多步时距：样本外 AUC 随预测距离衰减（h=1/2/3 月）===")
        print(wide.round(4).to_string())
        wide.round(4).to_csv(config.OUTPUT_DIR / "horizon_comparison.csv", encoding="utf-8-sig")
        long.round(4).to_csv(config.OUTPUT_DIR / "horizon_detail.csv", index=False, encoding="utf-8-sig")
        horizon.plot_horizon(wide, config.OUTPUT_DIR / "horizon_curve.png")
    except Exception as e:  # noqa: BLE001
        print(f"[train] 多步时距跳过: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
