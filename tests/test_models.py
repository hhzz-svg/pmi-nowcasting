"""test_models.py — 模型接口与朴素基准单测。"""
import numpy as np

from pmi_nowcast import models


def test_naive_majority_predicts_majority():
    m = models.NaiveMajority()
    y = np.array([1, 1, 1, 0])  # 多数=1
    m.fit(np.zeros((4, 2)), y)
    assert (m.predict(np.zeros((3, 2))) == 1).all()
    assert np.isclose(m.predict_proba(np.zeros((1, 2)))[0, 1], 0.75)


def test_naive_persistence_uses_pmi():
    """持续基准：当期 PMI>50 → 预测扩张。"""
    m = models.NaivePersistence(pmi_col_index=0)
    X = np.array([[51.0], [49.0], [50.5]])
    m.fit(X, np.array([1, 0, 1]))
    preds = m.predict(X)
    assert list(preds) == [1, 0, 1]


def test_all_models_share_interface():
    """每个模型都能 fit / predict_proba，输出形状一致。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] > 0).astype(int)
    for name, model in models.make_models(pmi_col_index=0).items():
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (60, 2), f"{name} 概率形状异常"
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), f"{name} 概率未归一"
