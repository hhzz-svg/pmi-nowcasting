"""test_rolling_window.py — 固定滚动窗口切分单测（对照扩展窗口）。

验证：训练窗口长度恒定、随折右移、仍满足 purge+embargo 防泄漏、不与测试集相交。
"""
from pmi_nowcast import validation


def test_rolling_window_size_is_fixed():
    """滚动窗口训练集长度应恒定（除去被 max(0,...) 截断的极早期）。"""
    folds = validation.rolling_window_splits(200, window=60, purge=3, embargo=1)
    sizes = [len(f.train_idx) for f in folds]
    # 首折 train_end = 64, train_start = max(0,4)=4 → size 60；此后恒 60
    assert all(s == 60 for s in sizes), f"窗口长度不恒定: {set(sizes)}"


def test_rolling_window_slides_forward():
    """训练窗口起点应随折递增（右移）。"""
    folds = validation.rolling_window_splits(200, window=60, purge=3, embargo=1)
    starts = [f.train_idx.min() for f in folds]
    assert starts == sorted(starts)
    assert starts[-1] > starts[0]


def test_rolling_no_leakage():
    """滚动窗口同样满足 purge+embargo 间隔、训练/测试不相交。"""
    purge, embargo = 3, 1
    folds = validation.rolling_window_splits(200, window=60, purge=purge, embargo=embargo)
    validation.assert_no_leakage(folds, purge=purge, embargo=embargo)
    for f in folds:
        assert not (set(f.train_idx) & set(f.test_idx))


def test_rolling_vs_expanding_same_test_points():
    """滚动与扩展窗口在相同参数下应覆盖相同的测试期（只是训练集不同）。"""
    exp = validation.walk_forward_splits(200, min_train=60, purge=3, embargo=1)
    roll = validation.rolling_window_splits(200, window=60, purge=3, embargo=1)
    exp_tests = [f.test_idx.tolist() for f in exp]
    roll_tests = [f.test_idx.tolist() for f in roll]
    assert exp_tests == roll_tests
    # 但扩展窗口训练集更大（末折）
    assert len(exp[-1].train_idx) > len(roll[-1].train_idx)
