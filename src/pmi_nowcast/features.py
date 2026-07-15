"""features.py — 时序特征工程。

每一步都对应设计文档第 2.2 节的一个"面试点"：
- 滞后特征（lag1/2/3）：预测 t+1 只用 t 及之前的信息。
- 环比变换（MoM diff）：宏观序列多有趋势/非平稳，用变化率转平稳。
- 滚动统计（3 月均值/标准差）：捕捉趋势与波动。
- 衍生信号：M1-M2 剪刀差——经济学家真在看的信用扩张/收缩信号。

★关键：本模块只做"横截面上安全"的变换。滚动 z-score 标准化因涉及
均值/方差估计，必须只用训练窗口统计量，故放到 validation/train 阶段做，
不在此处泄漏全样本信息。
"""
from __future__ import annotations

import pandas as pd

from . import config


def add_derived_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """加衍生特征：M1-M2 剪刀差。"""
    out = panel.copy()
    if {"m1_yoy", "m2_yoy"}.issubset(out.columns):
        # 剪刀差收窄/转负常领先景气回落，是经典信用周期信号
        out["m1_m2_gap"] = out["m1_yoy"] - out["m2_yoy"]
    return out


def add_mom_changes(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """环比变化（一阶差分），转成变化率以增强平稳性。"""
    out = panel.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_mom"] = out[c].diff()
    return out


def add_rolling_stats(panel: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    """滚动均值/标准差。仅用历史窗口（closed='left' 不含当期），避免当期泄漏。

    注意：这是描述性滚动统计（趋势/波动），非标准化。标准化在训练阶段用
    训练窗口统计量单独做。
    """
    out = panel.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}_roll{window}_mean"] = out[c].rolling(window).mean()
            out[f"{c}_roll{window}_std"] = out[c].rolling(window).std()
    return out


def add_lags(panel: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    """造滞后特征。所有特征来自 t 及之前，服务于预测 t+1。"""
    out = panel.copy()
    for c in cols:
        if c in out.columns:
            for lag in lags:
                out[f"{c}_lag{lag}"] = out[c].shift(lag)
    return out


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """精简特征管线，克制地控制特征数（小样本防过拟合）。

    设计文档 2.4 节明确：有效样本仅约 200 行，特征数必须克制，
    否则再严谨的验证也救不了过拟合。因此不做组合式特征爆炸，
    只保留每条经济逻辑最有信息量的少数特征：

      - 每个基础指标：当期水平 + lag1（一阶自相关/惯性）+ 环比变化。
      - 衍生信号：M1-M2 剪刀差及其 lag1。
      - PMI 自身：额外给 lag2/lag3（预期/情绪组自相关性最强）。

    输入：calendar.build_pit_panel 输出的时点对齐面板（index=asof_month）。
    输出：加好特征的宽表，仍以 asof_month 为 index。
    """
    out = add_derived_signals(panel)
    base_cols = [c for c in out.columns]

    # 环比变化（增强平稳性）
    out = add_mom_changes(out, base_cols)

    # 每个基础指标造 lag1
    out = add_lags(out, base_cols, [1])
    # PMI 惯性最强，额外给 lag2/lag3
    if "pmi" in out.columns:
        out = add_lags(out, ["pmi"], [2, 3])

    return out


if __name__ == "__main__":
    from . import calendar, ingest

    data = ingest.load_all()
    panel = calendar.build_pit_panel(data)
    feats = build_features(panel)
    print(f"features shape={feats.shape}")
    print(feats.columns.tolist())
