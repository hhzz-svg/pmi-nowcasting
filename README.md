# PMI Nowcasting — 制造业景气度即时预测

> 用央行真在用的 **Nowcasting** 方法，预测下月制造业 PMI 是否站上荣枯线（>50）。
> 卖点不是技术栈，而是**方法论的严谨性**：时点意识、walk-forward 验证、系统性防数据泄漏、经济价值评估。

![净值曲线](outputs/nav_curve.png)

---

## 一句话摘要

在 2008–2025 共 210 个月的样本上做**扩展窗口 walk-forward** 样本外预测，
最诚实的发现是:**朴素持续基准（"下月延续本月荣枯状态"）AUC 0.73，打平甚至略胜随机森林/XGBoost**。
这正是小样本宏观预测的真实图景——报告这个对比，比报一个虚高的准确率更有价值。

## 1. 背景:什么是 Nowcasting,为什么预测 PMI 有价值

PMI(采购经理指数)是最抢先发布的景气指标之一,月末即出。但它本身在荣枯线附近反复横跳,
"下月会否扩张"对宏观择时、行业配置都有直接意义。**Nowcasting** 是美联储、欧央行、IMF
在用的技术:用高频、抢先发布的数据"抢跑"预测发布滞后的低频指标。本项目即用截至当月
真正可得的金融/实体/价格数据,预测下月 PMI 的荣枯方向。

## 2. 数据与方法

### 2.1 数据源
全部经 [akshare](https://akshare.akfamily.xyz/) 程序化拉取并带时间戳缓存(`data/raw/`),可复现。

| 组别 | 指标 | 经济含义 |
|------|------|---------|
| ① 领先/金融 | M1、M2 同比,**M1-M2 剪刀差**,**新增信贷同比** | 信用扩张领先景气 |
| ② 实体活动 | 工业增加值同比、PPI | 当下生产热度 |
| ③ 需求/价格 | CPI、PPI 同比 | 需求与价格冷热 |
| ④ 预期/情绪 | 上月 PMI 及其 lag | 自相关性最强 |

> **信用信号数据说明**:原计划用社融存量(`macro_china_shrzgm`),但其数据源
> `data.mofcom.gov.cn` 只支持旧 TLS 配置,与本机 Python 的 OpenSSL 3.0.18 无法完成
> 握手(`SSLV3_ALERT_HANDSHAKE_FAILURE`)——系统 openssl 命令行可连通,证明是 Python
> 运行时的加密库兼容问题,非网络不可达。改用 **新增信贷当月同比**
> (`macro_china_new_financial_credit`,东财源,2008–今 222 月完整),它是社融的核心分项、
> 同样刻画信用扩张,历史更长且无 SSL 障碍。另叠加 **M1-M2 剪刀差**作为经典信用周期信号。

### 2.2 三大方法论升级(项目的真正支点)

**① Point-in-Time 时点对齐(`calendar.py`)** — 避免"未来函数"。
每个指标按其**发布滞后**向后移位:预测 6 月 PMI 时,站在 5 月末,5 月工业增加值可能还没发。
`PUBLICATION_LAG` 表显式编码"某月数据到第几个月才可得",对齐后每一行只含"该时点真正拿得到"的信息。

**② Walk-Forward + Purging/Embargo(`validation.py`)** — 绝不用随机 K 折。
扩展窗口滚动前推:每期用截至当时的全部历史重训,预测下一期。再叠加 López de Prado
《Advances in Financial Machine Learning》的两道防线:
- **Purging**:剔除训练集末尾与测试期时间重叠的样本(因标签依赖 PMI(t+1)、特征含 lag)。
- **Embargo**:训练/测试间再留缓冲月,隔离自相关泄漏。

该模块独立、可单测——`tests/` 里用构造数据验证"测试期样本绝不进训练集"。

**③ 经济价值回测(`backtest.py`)** — 从"准确率"到"钱"。
把信号翻译成简单择时(预测扩张→持沪深300,收缩→持币),画净值曲线 vs 买入持有,算夏普与最大回撤。

### 2.3 防泄漏总原则(黄金句:"我系统性地防范了数据泄漏")
- 时点对齐:特征按发布滞后可得性移位。
- 滚动标准化只用训练窗口统计量(`logreg` 的 `StandardScaler` 内嵌 Pipeline,每折仅在训练集 fit)。
- Walk-forward + purge/embargo。
- 标签是唯一允许的前视(它就是预测目标),特征列严格排除任何 `pmi_next` 派生量。

## 3. 结果(全部为 walk-forward 样本外)

| 模型 | Accuracy | Precision | Recall | F1 | AUC |
|------|---------|-----------|--------|-----|-----|
| naive_persistence | 0.733 | 0.714 | 0.732 | 0.723 | **0.733** |
| random_forest | 0.709 | 0.767 | 0.561 | 0.648 | 0.723 |
| xgboost | 0.651 | 0.608 | 0.756 | 0.674 | 0.712 |
| logreg | 0.651 | 0.720 | 0.439 | 0.545 | 0.616 |
| naive_majority | 0.477 | 0.477 | 1.000 | 0.646 | 0.561 |

> 数值由 `python -m pmi_nowcast.train` 在最新数据上生成,会随 akshare 数据更新而略变。

**最强领先信号**(逻辑回归系数 + 树重要性一致):当期及滞后 **PMI 自身**惯性最强,
其次是 **PPI 环比**、**工业增加值**、**M2**——与第 2 节的经济逻辑吻合。

## 4. 结论与局限

- **模型打过 naive_majority,但没稳定打过 naive_persistence。** 这不是失败,而是荣枯线附近
  预测的固有难度:PMI 长期在 49–51 横跳,拐点样本天然稀少。诚实报告 > 虚高指标。
- **小样本(~210 行)**:坚决不用深度学习;靠强正则、浅树、严格样本外验证控制过拟合。
- **无完整 vintage**:国内难拿修订前数据,本项目只显式处理**发布滞后**,数据修订(vintage)
  问题在此讨论而非假装解决。
- **回测非真实策略**:未计交易成本/滑点,仅示意信号有无经济价值。

## 5. 如何复现

```bash
pip install -e .            # 安装(核心依赖)
pip install -e ".[advanced]"  # 可选:xgboost / shap / streamlit
python -m pmi_nowcast.train   # 一条命令跑通:拉数→对齐→训练→评估→回测
pytest                        # 跑防泄漏等关键单测
```

产出在 `outputs/`:样本外预测、指标汇总、系数/重要性、净值曲线图。

## 6. 项目结构

```
src/pmi_nowcast/
├── config.py       指标清单、发布滞后表、参数
├── ingest.py       akshare 拉数 + 时间戳缓存
├── calendar.py     ★时点对齐 / 发布滞后(升级1)
├── features.py     lag / 环比 / 剪刀差等精简特征
├── dataset.py      组装训练表 + 标签(严格时点对齐)
├── validation.py   ★walk-forward + purge/embargo(升级2)
├── models.py       LogReg / RF / XGBoost / 朴素基准 统一接口
├── evaluate.py     样本外指标 + 可解释性
├── backtest.py     ★信号→净值曲线(升级3)
└── train.py        总驱动
tests/              防泄漏单测(对齐 / walk-forward / 标签)
```

---

*方法论参考:López de Prado, "Advances in Financial Machine Learning" (2018) — purging & embargo。*
