"""test_level_task.py — 点位回归任务的单测。

验证：
- 持续基准回归器恰好输出当期 PMI 列（预测零变化）；
- summarize_level 的 MAE/RMSE/隐含荣枯准确率在构造数据上逐点可算对；
- bootstrap_loss_diff（DM 检验自助版）：强弱分明时显著、完全相同时不显著；
- walk-forward 端到端在合成数据上跑通，持续基准的 MAE 等于 |ΔPMI| 均值。
"""
import numpy as np
import pandas as pd

from pmi_nowcast import level_task, significance


def test_persistence_regressor_returns_pmi_column():
    X = np.array([[1.0, 49.5], [2.0, 51.2]])
    reg = level_task.PersistenceRegressor(pmi_col_index=1)
    reg.fit(X, np.zeros(2))
    assert np.allclose(reg.predict(X), [49.5, 51.2])


def test_summarize_level_metrics_exact():
    preds = pd.DataFrame(
        {
            "asof_month": pd.date_range("2020-01-01", periods=4, freq="MS").tolist() * 1,
            "model": ["m"] * 4,
            "y_true": [50.0, 51.0, 49.0, 50.0],
            "y_pred": [50.5, 50.0, 49.0, 52.0],
        }
    )
    s = level_task.summarize_level(preds)
    assert np.isclose(s.loc["m", "mae"], (0.5 + 1.0 + 0.0 + 2.0) / 4)
    assert np.isclose(s.loc["m", "rmse"], np.sqrt((0.25 + 1.0 + 0.0 + 4.0) / 4))
    assert np.isclose(s.loc["m", "bias"], (0.5 - 1.0 + 0.0 + 2.0) / 4)
    # 隐含荣枯：真值>50 的只有 51；预测>50 的有 50.5 和 52 → 逐点比较
    # (50.5>50)!=(50>50)Х, (50>50)==(51>50)Х, (49>50)==(49>50)✓, (52>50)!=(50>50)Х
    assert np.isclose(s.loc["m", "隐含荣枯acc"], 0.25)


def test_loss_diff_significant_when_model_clearly_better():
    rng = np.random.default_rng(1)
    loss_good = np.abs(rng.normal(0.3, 0.1, 150))
    loss_bad = loss_good + np.abs(rng.normal(0.5, 0.1, 150))
    r = significance.bootstrap_loss_diff(loss_good, loss_bad, n_boot=500)
    assert r["diff"] > 0  # 正 = A（good）平均损失更小
    assert r["p_value"] < 0.05


def test_loss_diff_identical_not_significant():
    rng = np.random.default_rng(2)
    loss = np.abs(rng.normal(0.4, 0.1, 100))
    r = significance.bootstrap_loss_diff(loss, loss, n_boot=300)
    assert np.isclose(r["diff"], 0.0)
    assert r["p_value"] > 0.9


def test_level_walk_forward_persistence_mae_equals_abs_change():
    """合成数据端到端：持续基准 MAE = 测试期 |PMI(t+1)-PMI(t)| 的均值。"""
    rng = np.random.default_rng(3)
    n = 140
    months = pd.date_range("2010-01-01", periods=n, freq="MS")
    pmi = 50 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame(
        {
            "pmi": pmi,
            "x1": rng.normal(size=n),
            "pmi_current": pmi,
            "pmi_next": np.append(pmi[1:], np.nan),
            "y_expansion": (np.append(pmi[1:], 50) > 50).astype(int),
            "y_direction": rng.integers(0, 2, n),
        },
        index=months,
    ).dropna(subset=["pmi_next"])
    preds = level_task.run_level_walk_forward(df)
    assert not preds.empty
    g = preds[preds["model"] == "naive_persistence"]
    # 持续基准预测 = 当期 pmi，故其误差 = pmi_next - pmi
    expect = (df.loc[g["asof_month"], "pmi_next"] - df.loc[g["asof_month"], "pmi"]).to_numpy()
    got = (g["y_true"] - g["y_pred"]).to_numpy()
    assert np.allclose(expect, got)
    s = level_task.summarize_level(preds)
    assert np.isclose(s.loc["naive_persistence", "mae"], np.abs(expect).mean())
