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


def _cache_path(name: str, stamp: str) -> config.Path:
    return config.RAW_DIR / f"{name}_{stamp}.csv"


def _latest_cache(name: str):
    files = sorted(config.RAW_DIR.glob(f"{name}_*.csv"))
    return files[-1] if files else None


def load_hs300_monthly() -> pd.DataFrame:
    """沪深300 日频行情聚合成月频市场特征：月收益率 + 月内已实现波动。

    价格在月末收盘即完全可得（PUBLICATION_LAG=0），是全部特征里最"即时"的
    信号——这正是 nowcasting 用金融市场数据抢跑宏观发布的经典思路。
    返回已成形的 ['month', 'hs300_ret', 'hs300_vol']，无需再过 _shape。
    """
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    close = df.set_index("date").sort_index()["close"]
    daily_ret = close.pct_change()
    monthly = pd.DataFrame(
        {
            "hs300_ret": close.resample("MS").last().pct_change(),
            "hs300_vol": daily_ret.resample("MS").std(),
        }
    )
    monthly = monthly.dropna().reset_index().rename(columns={"date": "month"})
    return monthly


def fetch_indicator(name: str, spec: dict, use_cache: bool = True) -> pd.DataFrame:
    """拉取单个指标，返回列 = ['month', <renamed value cols>]。

    标准指标：akshare 函数 → 缓存原始表 → _shape 裁剪。
    自定义指标（spec 含 'custom'）：loader 直接产出成形表，缓存亦为成形表。
    """
    today = dt.datetime.now(tz=dt.timezone.utc).date().isoformat()
    path = _cache_path(name, today)

    if "custom" in spec:
        if use_cache:
            cached = _latest_cache(name)
            if cached is not None:
                out = pd.read_csv(cached)
                out["month"] = pd.to_datetime(out["month"])
                return out
        out = globals()[spec["custom"]]()
        out.to_csv(path, index=False, encoding="utf-8-sig")
        return out

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
    # 同一月可能有多条报告行（初值/终值/预告），预告行的值为空。
    # 先剔除全空值行，再按月去重保留最后一条，避免空的预告行覆盖有值行。
    out = out.dropna(subset=list(keep.values()), how="all")
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
