"""shap_analysis.py — SHAP 特征贡献分析（可选进阶：行业标准可解释性）。

特征重要性只告诉你"哪个特征重要"，SHAP 进一步告诉你"某特征在哪个方向、
多大程度上推动了每一次预测"——把黑箱树模型的每个预测拆成各特征的加性贡献。

用随机森林在全样本上重训（与 evaluate.tree_importances 的口径一致，仅供
展示，不参与样本外评估），产出：
  - shap_summary.png    beeswarm 蜂群图：每点一个样本，颜色=特征值高低，
                        横轴=对"扩张"预测的推动方向与力度。
  - shap_bar.png        平均绝对 SHAP 条形图：全局特征重要性排序。
  - shap_importance.csv  平均 |SHAP| 数值表。

shap 是进阶可选依赖（pip install -e ".[advanced]"）；缺失时本模块整体跳过，
绝不阻断主管线。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config, dataset


def _fit_rf(df: pd.DataFrame, target: str):
    feat_cols = dataset.feature_columns(df)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=int)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )
    rf.fit(X, y)
    return rf, X, feat_cols


def compute_shap(df: pd.DataFrame, target: str = "y_expansion"):
    """计算随机森林的 SHAP 值矩阵，返回 (shap_for_class1, X, feat_cols)。

    TreeExplainer 对二分类返回每类一组 SHAP；取"扩张"类（索引 1）。
    不同 shap 版本形状略异（(n,f) 或 (n,f,2)），统一处理。
    """
    import shap

    rf, X, feat_cols = _fit_rf(df, target)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X)
    # 兼容不同 shap 版本的输出形状
    if isinstance(sv, list):  # 旧版：[class0_array, class1_array]
        sv1 = sv[1]
    else:
        sv = np.asarray(sv)
        sv1 = sv[:, :, 1] if sv.ndim == 3 else sv  # 新版可能是 (n,f,2)
    return np.asarray(sv1), X, feat_cols


def run_shap(df: pd.DataFrame, target: str = "y_expansion") -> pd.Series:
    """算 SHAP 并存 beeswarm / bar 图与平均 |SHAP| 表，返回重要性 Series。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    sv1, X, feat_cols = compute_shap(df, target)

    mean_abs = pd.Series(np.abs(sv1).mean(axis=0), index=feat_cols).sort_values(ascending=False)
    mean_abs.round(5).to_csv(config.OUTPUT_DIR / "shap_importance.csv", encoding="utf-8-sig")

    # beeswarm 蜂群图（对"扩张"类的贡献）
    shap.summary_plot(sv1, X, feature_names=feat_cols, show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle("SHAP 蜂群图：各特征对'下月扩张'预测的贡献", y=1.02)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 平均 |SHAP| 条形图
    shap.summary_plot(sv1, X, feature_names=feat_cols, plot_type="bar", show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle("平均 |SHAP|：全局特征重要性", y=1.02)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "shap_bar.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return mean_abs


if __name__ == "__main__":
    df = dataset.build_dataset()
    imp = run_shap(df)
    print(imp.head(10).round(4).to_string())
