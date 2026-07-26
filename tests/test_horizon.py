"""test_horizon.py — 多步时距实验的单测。

验证：
- 多步标签 y_expansion_h2/h3 在构造数据上逐点正确，且尾部无未来数据时为 NA；
- PMI 序列有漏发月份时，标签按日历月位移（而非按行位移）不错位；
- 多步任务的 purge 随 h 扩大，训练/测试 gap ≥ h（防标签前视泄漏）；
- feature_columns 不把多步标签当特征。
"""
import numpy as np
import pandas as pd

from pmi_nowcast import config, dataset, horizon, validation


def _pmi_frame(values, start="2020-01-01"):
    months = pd.date_range(start, periods=len(values), freq="MS")
    return {"pmi": pd.DataFrame({"month": months, "pmi": values})}


def test_horizon_labels_pointwise_correct():
    values = [49, 51, 52, 48, 50.5, 49.5]
    labels = dataset.build_labels(_pmi_frame(values))
    # y_expansion_h2[t] = 1{PMI(t+2) > 50}
    assert labels["y_expansion_h2"].iloc[0] == 1  # PMI(3月)=52
    assert labels["y_expansion_h2"].iloc[1] == 0  # PMI(4月)=48
    assert labels["y_expansion_h3"].iloc[0] == 0  # PMI(4月)=48
    assert labels["y_expansion_h3"].iloc[1] == 1  # PMI(5月)=50.5
    # 尾部无 t+h 数据 → NA 而非 0
    assert labels["y_expansion_h2"].iloc[-2:].isna().all()
    assert labels["y_expansion_h3"].iloc[-3:].isna().all()


def test_horizon_labels_calendar_shift_across_gap():
    """PMI 漏发一个月时，h 步标签必须按日历月对齐，不能按行错位。"""
    months = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01", "2020-05-01"])
    ind = {"pmi": pd.DataFrame({"month": months, "pmi": [49.0, 52.0, 48.0, 55.0]})}
    labels = dataset.build_labels(ind)
    # 2020-01 的 h2 标签对应 2020-03（漏发）→ 必须是 NA，不能取到 04 月的 48
    assert pd.isna(labels.loc["2020-01-01", "y_expansion_h2"])
    # 2020-02 的 h2 标签对应 2020-04 = 48 → 0
    assert labels.loc["2020-02-01", "y_expansion_h2"] == 0
    # 2020-02 的 h3 标签对应 2020-05 = 55 → 1
    assert labels.loc["2020-02-01", "y_expansion_h3"] == 1


def test_horizon_target_names_and_not_features():
    assert horizon.target_for_horizon(1) == "y_expansion"
    assert horizon.target_for_horizon(3) == "y_expansion_h3"
    df = pd.DataFrame(
        {
            "pmi": [50.0],
            "y_expansion": [1],
            "y_expansion_h2": [1],
            "y_expansion_h3": [0],
            "y_direction": [1],
            "pmi_current": [50.0],
            "pmi_next": [50.3],
        }
    )
    feats = dataset.feature_columns(df)
    assert feats == ["pmi"]


def test_horizon_purge_scales_with_h():
    """h=3 时训练/测试 gap 必须 ≥ 3：标签前视 3 个月，购者自防。"""
    for h in horizon.HORIZONS:
        purge = max(config.PURGE_MONTHS, h)
        folds = validation.walk_forward_splits(200, purge=purge)
        for fold in folds:
            gap = fold.test_idx.min() - fold.train_idx.max() - 1
            assert gap >= h + config.EMBARGO_MONTHS - 1
            assert gap >= h  # 标签依赖 PMI(t+h)，gap 必须盖住前视


def test_horizon_walk_forward_runs_on_synthetic():
    """构造 160 行小数据端到端跑通 h=2，预测覆盖尾部且无 NaN。"""
    rng = np.random.default_rng(0)
    n = 160
    months = pd.date_range("2010-01-01", periods=n, freq="MS")
    pmi = 50 + np.cumsum(rng.normal(0, 0.6, n)).round(1)
    df = pd.DataFrame(
        {
            "pmi": pmi,
            "x1": rng.normal(size=n),
            "pmi_current": pmi,
            "pmi_next": np.roll(pmi, -1),
            "y_expansion": (np.roll(pmi, -1) > 50).astype(int),
            "y_direction": rng.integers(0, 2, n),
        },
        index=months,
    )
    df["y_expansion_h2"] = pd.array((np.roll(pmi, -2) > 50).astype("int64"), dtype="Int64")
    df.loc[df.index[-2:], "y_expansion_h2"] = pd.NA
    df["y_expansion_h3"] = df["y_expansion_h2"]
    preds = horizon.run_horizon_walk_forward(df, h=2)
    assert not preds.empty
    assert preds["y_proba"].notna().all()
    assert set(preds["horizon"]) == {2}
