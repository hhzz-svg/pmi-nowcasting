"""calendar.py — ★发布日历 / 时点对齐（设计文档"升级 1"核心）。

宏观数据的致命陷阱：发布滞后（publication lag）。
预测 t+1 月 PMI 时，我们站在 t 月末。此刻 t 月的工业增加值可能还没发布
（要到 t+1 月中旬才出）。若直接用 t 月工业数据，就是"未来函数"——偷看了
预测时点根本拿不到的信息，回测会虚高。

本模块把每个指标按其发布滞后"向后移位"：一个 lag=k 的指标，其标称月份 m
的数值，直到 m+k 月才可用。对齐后，某一行代表"该时点真正可得的信息集"。
"""
from __future__ import annotations

import pandas as pd

from . import config


def apply_publication_lag(df: pd.DataFrame, col: str, lag: int) -> pd.DataFrame:
    """把某列按发布滞后向后移 lag 个月。

    返回单列 DataFrame，index=可得月份（availability month）。
    lag=0 表示当月即可得（如 PMI）；lag=1 表示次月才可得。
    """
    s = df.set_index("month")[col]
    # 标称月 m 的数据在 m+lag 月才可得 → index 整体后移 lag 个月
    shifted = s.copy()
    shifted.index = shifted.index + pd.DateOffset(months=lag)
    # 归一回月初，避免 offset 产生的非月初时间戳
    shifted.index = shifted.index.to_period("M").to_timestamp()
    return shifted.to_frame(name=col)


def build_pit_panel(indicators: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """组装 Point-in-Time 面板。

    对每个数值列施加其发布滞后，再按可得月份 outer-join。
    结果的每一行 = "站在该月末，历史上真正已发布的所有指标最新值"。

    注意：PMI 有两个角色——
      1) 作为特征时按 lag=0 参与（当月可得，用于造 lag 特征）；
      2) 作为标签（下月 PMI）由 dataset.py 单独处理，不在此对齐。
    """
    aligned: list[pd.DataFrame] = []
    for df in indicators.values():
        value_cols = [c for c in df.columns if c != "month"]
        for col in value_cols:
            lag = config.PUBLICATION_LAG.get(col, 1)
            aligned.append(apply_publication_lag(df, col, lag))

    if not aligned:
        raise ValueError("没有可对齐的指标数据")

    panel = pd.concat(aligned, axis=1).sort_index()

    # 补齐连续月度索引，并只对"内部"短缺口做线性插值。
    # 背景：工业增加值等偶有个别月份漏发（如疫情期间），造成序列内部散点缺失。
    # limit_area='inside' 保证绝不外推（不向序列两端之外造数），不引入未来函数。
    full_idx = pd.period_range(panel.index.min(), panel.index.max(), freq="M").to_timestamp()
    panel = panel.reindex(full_idx)
    panel = panel.interpolate(method="linear", limit=2, limit_area="inside")

    panel.index.name = "asof_month"
    return panel


if __name__ == "__main__":
    from . import ingest

    data = ingest.load_all()
    panel = build_pit_panel(data)
    print(panel.tail(6).to_string())
