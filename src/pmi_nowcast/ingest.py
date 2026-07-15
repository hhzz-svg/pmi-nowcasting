"""ingest.py — 从 akshare 拉取宏观指标，带时间戳缓存。

akshare 接口偶尔不稳、字段可能变动，因此：
- 每次拉取存 CSV 到 data/raw/，文件名带拉取日期（记录数据"版本"）。
- 若当天已有缓存，直接复用，不重复请求（接口友好 + 可复现）。
- 每个指标独立容错：一个失败不影响其余。
"""
from __future__ import annotations

import datetime as dt

import akshare as ak
import pandas as pd

from . import config


def _parse_month(series: pd.Series) -> pd.Series:
    """把各种中文/英文日期格式统一成月初 Timestamp。

    akshare 里出现过 '2008年02月份'、'1996-02-01' 两类格式。
    统一归到月初（Period[M] → Timestamp），便于跨指标对齐。
    """
    s = series.astype(str).str.strip()
    # 中文格式 '2008年02月份'
    cn = s.str.extract(r"(\d{4})年(\d{1,2})月")
    parsed = pd.to_datetime(
        cn[0] + "-" + cn[1] + "-01", errors="coerce", format="%Y-%m-%d"
    )
    # 其余走通用解析
    fallback = pd.to_datetime(s, errors="coerce")
    parsed = parsed.fillna(fallback)
    # 归一到月初
    return parsed.dt.to_period("M").dt.to_timestamp()


def _cache_path(name: str, stamp: str) -> "config.Path":
    return config.RAW_DIR / f"{name}_{stamp}.csv"


def _latest_cache(name: str):
    files = sorted(config.RAW_DIR.glob(f"{name}_*.csv"))
    return files[-1] if files else None


def fetch_indicator(name: str, spec: dict, use_cache: bool = True) -> pd.DataFrame:
    """拉取单个指标，返回列 = ['month', <renamed value cols>]。"""
    today = dt.date.today().isoformat()
    path = _cache_path(name, today)

    if use_cache:
        cached = _latest_cache(name)
        if cached is not None:
            raw = pd.read_csv(cached)
            return _shape(raw, spec)

    func = getattr(ak, spec["func"])
    raw = func()
    raw.to_csv(path, index=False, encoding="utf-8-sig")
    return _shape(raw, spec)


def _shape(raw: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """裁剪到关心的列，重命名，解析月份，按月排序去重。"""
    df = raw.copy()
    df["month"] = _parse_month(df[spec["date_col"]])
    keep = {src: dst for src, dst in spec["value_cols"].items()}
    for src in keep:
        df[src] = pd.to_numeric(df[src], errors="coerce")
    out = df[["month", *keep.keys()]].rename(columns=keep)
    out = out.dropna(subset=["month"]).sort_values("month")
    out = out.drop_duplicates(subset="month", keep="last").reset_index(drop=True)
    return out


def load_all(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """拉取全部指标，返回 {name: DataFrame}。单个失败仅告警。"""
    out: dict[str, pd.DataFrame] = {}
    for name, spec in config.INDICATORS.items():
        try:
            out[name] = fetch_indicator(name, spec, use_cache=use_cache)
            print(f"[ingest] {name:14s} OK  rows={len(out[name])}")
        except Exception as e:  # noqa: BLE001 - 容错：接口不稳不能中断整条管线
            print(f"[ingest] {name:14s} FAIL {type(e).__name__}: {str(e)[:80]}")
    return out


if __name__ == "__main__":
    load_all(use_cache=False)
