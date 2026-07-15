"""validation.py — ★Walk-Forward 验证 + Purging / Embargo（设计文档"升级 2"核心）。

绝不使用随机 K 折——那会用未来预测过去。正确做法是扩展窗口滚动前推：
每一期用"截至该时点的全部历史"训练，预测下一期，再逐月扩窗重训。

再叠加 López de Prado《Advances in Financial Machine Learning》的两道防线，
消除训练/测试在时间上的"沾边泄漏"：

  Purging（净化）：因为标签 y_t 依赖 PMI(t+1)，且特征含 lag，训练集末尾若
    与测试期时间重叠，信息会渗漏。剔除训练窗口末尾 purge 个月的样本。
  Embargo（禁运）：在（净化后的）训练集与测试集之间再留 embargo 个月缓冲，
    进一步隔离自相关带来的泄漏。

本模块只产出"每一期的训练/测试行索引"，与具体模型解耦，便于单测：
可用构造数据验证"测试期样本绝不出现在训练集里"。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Fold:
    """一期 walk-forward：训练行位置 + 测试行位置（基于 0..n-1 的整数下标）。"""

    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n_samples: int,
    min_train: int = config.MIN_TRAIN_MONTHS,
    purge: int = config.PURGE_MONTHS,
    embargo: int = config.EMBARGO_MONTHS,
    test_size: int = 1,
) -> list[Fold]:
    """生成扩展窗口 walk-forward 折。

    参数
    ----
    n_samples : 样本总数（按时间升序排列）。
    min_train : 首个训练窗口的最小长度（月）。
    purge     : 训练集末尾剔除的月数（净化）。
    embargo   : 训练与测试间的缓冲月数（禁运）。
    test_size : 每期测试样本数（默认 1，逐月前推）。

    对第 k 期：
      test  = [test_start, test_start + test_size)
      gap   = embargo 个月（紧邻测试之前，不进训练）
      train = [0, test_start - embargo - purge)
    训练窗口随 k 增大而扩展。
    """
    folds: list[Fold] = []
    # 首个测试起点：保证训练至少有 min_train 个样本（含 purge+embargo 缓冲）
    test_start = min_train + purge + embargo
    while test_start + test_size <= n_samples:
        train_end = test_start - embargo - purge  # 独占上界
        if train_end <= 0:
            test_start += test_size
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_start + test_size)
        folds.append(Fold(train_idx=train_idx, test_idx=test_idx))
        test_start += test_size
    return folds


def assert_no_leakage(folds: list[Fold], purge: int, embargo: int) -> None:
    """自检：每一折训练集最大下标与测试集最小下标之间，至少隔 purge+embargo。

    供单元测试与运行时双重使用——这是本项目"防泄漏写进代码"的直接证据。
    """
    for i, fold in enumerate(folds):
        if len(fold.train_idx) == 0:
            continue
        gap = fold.test_idx.min() - fold.train_idx.max() - 1
        if gap < purge + embargo:
            raise AssertionError(
                f"fold {i}: gap={gap} < purge+embargo={purge + embargo}（存在泄漏）"
            )
        # 训练/测试绝不相交
        if set(fold.train_idx) & set(fold.test_idx):
            raise AssertionError(f"fold {i}: 训练集与测试集下标相交")


if __name__ == "__main__":
    folds = walk_forward_splits(210)
    print(f"生成 {len(folds)} 折")
    assert_no_leakage(folds, config.PURGE_MONTHS, config.EMBARGO_MONTHS)
    print("防泄漏自检通过")
    f0, fl = folds[0], folds[-1]
    print(f"首折 train={f0.train_idx.min()}..{f0.train_idx.max()} test={f0.test_idx}")
    print(f"末折 train={fl.train_idx.min()}..{fl.train_idx.max()} test={fl.test_idx}")
