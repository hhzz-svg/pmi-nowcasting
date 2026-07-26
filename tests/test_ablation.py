"""test_ablation.py — 特征分组逻辑单测（不跑完整 walk-forward，只验证归组正确）。"""
from pmi_nowcast import ablation


def test_group_matches_exact_and_derived():
    """归组应同时命中：列名等于前缀、以及 '<前缀>_lag/mom' 派生列。"""
    feat_cols = [
        "pmi", "pmi_lag1", "pmi_mom",  # PMI 组
        "m1_yoy", "m2_yoy", "m1_m2_gap", "m1_yoy_lag1",  # 货币金融组
        "ppi_yoy", "cpi_yoy_mom",  # 价格组
    ]
    pmi_cols = ablation._cols_in_group(feat_cols, ("pmi",))
    assert set(pmi_cols) == {"pmi", "pmi_lag1", "pmi_mom"}

    money_cols = ablation._cols_in_group(feat_cols, ("m1_yoy", "m2_yoy", "m1_m2_gap"))
    assert set(money_cols) == {"m1_yoy", "m2_yoy", "m1_m2_gap", "m1_yoy_lag1"}


def test_group_no_prefix_collision():
    """前缀归组不应误伤：'ppi_yoy' 不能被 'pi' 之类误匹配（用等号或 '_' 边界）。"""
    feat_cols = ["ppi_yoy", "cpi_yoy", "ip_yoy"]
    # 'ip_yoy' 组不应吞掉 'ppi_yoy'（后者不以 'ip_yoy_' 开头也不等于 'ip_yoy'）
    ip_cols = ablation._cols_in_group(feat_cols, ("ip_yoy",))
    assert ip_cols == ["ip_yoy"]


def test_all_groups_defined():
    """七个经济组齐全：四条传导链 + 信用 + 外需 + 市场（数据范围扩充后）。"""
    assert set(ablation.FEATURE_GROUPS) == {
        "PMI惯性",
        "货币金融",
        "价格",
        "实体活动",
        "信用",
        "外需",
        "市场",
    }
