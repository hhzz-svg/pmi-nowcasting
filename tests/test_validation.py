"""test_validation.py — walk-forward 防泄漏的核心单测（项目面试亮点）。

用构造数据验证："测试期样本绝不进训练集"，且训练/测试间隔满足 purge+embargo。
"""
import numpy as np

from pmi_nowcast import validation


def test_expanding_window_grows():
    """训练窗口应随折数递增（扩展窗口特性）。"""
    folds = validation.walk_forward_splits(100, min_train=20, purge=3, embargo=1)
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes), "训练窗口未单调扩展"
    assert sizes[-1] > sizes[0]


def test_no_train_test_overlap():
    """训练集与测试集下标绝不相交。"""
    folds = validation.walk_forward_splits(100, min_train=20, purge=3, embargo=1)
    for f in folds:
        assert not (set(f.train_idx) & set(f.test_idx))


def test_purge_embargo_gap_enforced():
    """训练集末尾与测试集之间至少隔 purge+embargo 个样本。"""
    purge, embargo = 3, 2
    folds = validation.walk_forward_splits(120, min_train=30, purge=purge, embargo=embargo)
    for f in folds:
        gap = f.test_idx.min() - f.train_idx.max() - 1
        assert gap >= purge + embargo, f"gap={gap} 小于 purge+embargo={purge+embargo}"


def test_test_never_before_train():
    """测试期一定在训练期之后（不能用未来预测过去）。"""
    folds = validation.walk_forward_splits(100, min_train=20, purge=3, embargo=1)
    for f in folds:
        assert f.test_idx.min() > f.train_idx.max()


def test_assert_no_leakage_passes_on_valid_folds():
    folds = validation.walk_forward_splits(100, min_train=20, purge=3, embargo=1)
    # 不抛异常即通过
    validation.assert_no_leakage(folds, purge=3, embargo=1)


def test_assert_no_leakage_catches_leak():
    """人为构造泄漏折，自检必须抛异常。"""
    import pytest

    bad = [validation.Fold(train_idx=np.arange(0, 10), test_idx=np.array([10]))]
    # gap=0 < purge+embargo=4 → 应报错
    with pytest.raises(AssertionError):
        validation.assert_no_leakage(bad, purge=3, embargo=1)


def test_min_train_respected():
    """首折训练样本数不少于 min_train。"""
    folds = validation.walk_forward_splits(100, min_train=30, purge=2, embargo=1)
    assert len(folds[0].train_idx) >= 30
