"""config.py — 指标清单、发布滞后表、路径与建模参数。

集中管理项目的所有"事实"配置，让其他模块保持无魔法数字。
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 指标清单
# ---------------------------------------------------------------------------
# 每个指标：akshare 拉取函数名 + 我们关心的列 + 重命名。
# ingest.py 按此表拉数并缓存。
INDICATORS: dict[str, dict] = {
    "pmi": {
        "func": "macro_china_pmi",
        "date_col": "月份",
        "value_cols": {"制造业-指数": "pmi"},
        "desc": "官方制造业 PMI 指数（预测标的）",
    },
    "money_supply": {
        "func": "macro_china_money_supply",
        "date_col": "月份",
        "value_cols": {
            "货币和准货币(M2)-同比增长": "m2_yoy",
            "货币(M1)-同比增长": "m1_yoy",
        },
        "desc": "M1/M2 货币供应同比（金融领先信号）",
    },
    "cpi": {
        "func": "macro_china_cpi_monthly",
        "date_col": "日期",
        "value_cols": {"今值": "cpi_yoy"},
        "desc": "CPI 同比（需求/价格）",
    },
    "ppi": {
        "func": "macro_china_ppi",
        "date_col": "月份",
        "value_cols": {"当月同比增长": "ppi_yoy"},
        "desc": "PPI 当月同比（工业价格）",
    },
    "industrial": {
        "func": "macro_china_industrial_production_yoy",
        "date_col": "日期",
        "value_cols": {"今值": "ip_yoy"},
        "desc": "规模以上工业增加值同比（实体活动）",
    },
    "credit": {
        "func": "macro_china_new_financial_credit",
        "date_col": "月份",
        "value_cols": {"当月-同比增长": "credit_yoy"},
        "desc": "新增信贷当月同比（信用扩张领先信号，替代社融存量）",
    },
}

# ---------------------------------------------------------------------------
# 发布滞后表（升级 1 核心：Point-in-Time 意识）
# ---------------------------------------------------------------------------
# 单位：月。表示"某月的数据，要到该月结束后第几个月才真正发布"。
# 例：工业增加值 lag=1 → 6 月的工业数据大约 7 月中旬发布。
# 预测 t+1 月 PMI 时，站在 t 月末的时点，只能用"已发布"的历史数据。
PUBLICATION_LAG: dict[str, int] = {
    "pmi": 0,          # PMI 当月月末发布，最抢先
    "m2_yoy": 1,       # 金融数据次月发布
    "m1_yoy": 1,
    "cpi_yoy": 1,      # CPI 次月上旬
    "ppi_yoy": 1,      # PPI 次月上旬
    "ip_yoy": 1,       # 工业增加值次月中旬
    "credit_yoy": 1,   # 新增信贷次月上旬（央行金融统计）
}

# ---------------------------------------------------------------------------
# 特征与建模参数
# ---------------------------------------------------------------------------
EXPANSION_THRESHOLD = 50.0     # PMI 荣枯线
LAGS = [1, 2, 3]               # 造 lag1/lag2/lag3
ROLL_WINDOW = 3                # 滚动统计窗口

# walk-forward 参数
MIN_TRAIN_MONTHS = 120         # 首个训练窗口至少 10 年
EMBARGO_MONTHS = 1             # 训练/测试间隔（embargo）
PURGE_MONTHS = 3              # purge：剔除与测试期重叠的训练样本（=最大 lag）

RANDOM_STATE = 42
