"""dataset.py — 组装训练表（严格时点对齐 + 标签构造）。

核心时点逻辑：
  站在 asof_month = t 月末，用当时可得的信息（特征面板第 t 行），
  预测 t+1 月的 PMI 是否 > 50。
  → 特征 = features 面板第 t 行；标签 y_t = 1{PMI(t+1) > 50}。

因此标签是 PMI 相对特征行"向前看一个月"。这是唯一允许的"看未来"——
它就是预测目标本身，不是特征泄漏。
"""
from __future__ import annotations

import pandas as pd

from . import calendar, config, features, ingest


def build_labels(indicators: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """从原始 PMI 构造标签，index=预测基准月 asof_month。

    y_expansion:  1{PMI(t+1) > 50}   下月是否扩张（主标签）
    y_direction:  1{PMI(t+1) > PMI(t)}  下月是否上行（副标签）
    还保留 pmi_next 原值用于回测。
    """
    pmi = indicators["pmi"].set_index("month")["pmi"].sort_index()
    # 补齐连续月度索引再 shift：若历史上有漏发月份，按"行"位移会跨月错位，
    # 按连续日历索引位移才保证 shift(-h) 恰是 h 个日历月之后。
    full_idx = pd.period_range(pmi.index.min(), pmi.index.max(), freq="M").to_timestamp()
    pmi = pmi.reindex(full_idx)
    pmi_next = pmi.shift(-1)  # 把 t+1 的 PMI 对到 t 行
    has_next = pmi_next.notna()  # 无下月数据时标签应为 NA，而非 False→0
    labels = pd.DataFrame(
        {
            "pmi_current": pmi,
            "pmi_next": pmi_next,
            "y_expansion": (pmi_next > config.EXPANSION_THRESHOLD).where(has_next).astype("Int64"),
            "y_direction": (pmi_next > pmi).where(has_next).astype("Int64"),
        }
    )
    # 多步时距标签：t+2 / t+3 月是否扩张（horizon.py 的多步实验用）
    for h in (2, 3):
        pmi_h = pmi.shift(-h)
        labels[f"y_expansion_h{h}"] = (
            (pmi_h > config.EXPANSION_THRESHOLD).where(pmi_h.notna()).astype("Int64")
        )
    labels.index.name = "asof_month"
    return labels


def build_dataset(use_cache: bool = True) -> pd.DataFrame:
    """端到端组装训练表：ingest → PIT 对齐 → 特征 → 拼标签。

    返回：index=asof_month，含所有特征列 + 标签列，已 dropna。
    """
    data = ingest.load_all(use_cache=use_cache)
    if "pmi" not in data:
        raise RuntimeError("PMI 数据拉取失败，无法构造标签")

    panel = calendar.build_pit_panel(data)
    feats = features.build_features(panel)
    labels = build_labels(data)

    df = feats.join(labels, how="inner")

    # 丢弃标签缺失行（最后一行 pmi_next 为空、以及早期 lag 造成的 NaN）
    df = df.dropna(subset=["y_expansion", "pmi_next"])

    # 特征列的 NaN（lag/rolling 预热期）整行剔除，保证模型输入完整
    feature_cols = [c for c in feats.columns]
    df = df.dropna(subset=feature_cols)

    df = df.sort_index()
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """训练用特征列 = 全部列去掉标签/辅助列。"""
    label_like = {
        "pmi_current",
        "pmi_next",
        "y_expansion",
        "y_direction",
        "y_expansion_h2",
        "y_expansion_h3",
    }
    return [c for c in df.columns if c not in label_like]


def save_processed(df: pd.DataFrame) -> None:
    path = config.PROCESSED_DIR / "training_table.csv"
    df.to_csv(path, encoding="utf-8-sig")
    print(f"[dataset] saved {path}  shape={df.shape}")


if __name__ == "__main__":
    df = build_dataset()
    print(f"dataset shape={df.shape}, span={df.index.min()}..{df.index.max()}")
    print(f"features: {len(feature_columns(df))}")
    save_processed(df)
