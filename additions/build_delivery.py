"""Generate reports ONLY from local measured artifacts. No placeholder performance numbers."""

from pathlib import Path
import json, shutil, hashlib, zipfile, difflib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
OUT = BASE / "outputs/01_零售经营分析"
PN = {
    "global_matched": "全局按风险优先",
    "per_store_top_n": "每店最多 K 个",
    "one_per_store_then_global": "每店至少 1 个，其余全局分配",
}


def pct(x):
    return f"{100 * x:.2f}%"


def pp(x):
    return f"{100 * x:+.2f} 个百分点"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = ROOT / "analysis/reports/tables"
    alloc = ROOT / "results/allocation"
    biz = ROOT / "results/business"
    meta = json.loads((ROOT / "analysis/models/best_model.json").read_text())
    entry = meta["entry"]
    models = pd.read_csv(tabs / "model_comparison_test.csv").set_index("entry")
    best = models.loc[entry]
    base = models.loc["baseline_today"]
    bt = pd.read_csv(tabs / "backtest_summary.csv").set_index("entry")
    ci = pd.read_csv(tabs / "test_metrics_ci.csv").set_index("entry")
    policies = pd.read_csv(alloc / "policy_summary.csv")
    primary = policies[(policies.split == "test") & (policies.k == 5)].set_index(
        "policy"
    )
    g, l, f = (primary.loc[p] for p in PN)
    daily = pd.read_csv(biz / "global_daily.csv")
    test = daily[daily.dt >= "2024-06-26"]
    persistence = pd.read_csv(biz / "mart_persistence.csv")
    pt = persistence[persistence.label_period == "test"].set_index("current_severe")
    aci = pd.read_csv(alloc / "test_primary_paired_diff_ci.csv")
    floor_ci = aci[
        (aci.policy == "one_per_store_then_global") & (aci.metric == "precision")
    ].iloc[0]
    rawstatus = pd.read_csv(tabs / "data_checks_status_vs_sales.csv").set_index(
        "stock_status"
    )
    original = pd.read_csv(
        ROOT / "upstream/reports/tables/model_comparison_test.csv"
    ).set_index("entry")
    comparison = models[["pr_auc", "precision_at_10", "recall_at_10", "roc_auc"]].join(
        original[["pr_auc", "precision_at_10", "recall_at_10", "roc_auc"]],
        rsuffix="_author",
    )
    for c in ["pr_auc", "precision_at_10", "recall_at_10", "roc_auc"]:
        comparison[c + "_difference"] = comparison[c] - comparison[c + "_author"]
    comparison.to_csv(ROOT / "results/reproduction_vs_author.csv")
    metric_table = table(
        ["方法", "本次 PR-AUC（AP）", "本次 P@10%", "本次 R@10%", "5 轮平均 PR-AUC"],
        [
            [
                n,
                f"{models.loc[n, 'pr_auc']:.4f}",
                pct(models.loc[n, "precision_at_10"]),
                pct(models.loc[n, "recall_at_10"]),
                f"{bt.loc[n, 'pr_auc_mean']:.4f}",
            ]
            for n in ["baseline_today", "baseline_mean7", "lightgbm_daily", entry]
        ],
    )
    policy_table = table(
        [
            "K=5 分配策略",
            "每日总名额",
            "命中率",
            "召回率",
            "每日命中数",
            "门店覆盖率",
            "日任务量 P95",
            "每日超过 K 的门店数",
        ],
        [
            [
                PN[n],
                f"{r.alerts:.0f}",
                pct(r.precision),
                pct(r.recall),
                f"{r.tp:.1f}",
                pct(r.store_coverage),
                f"{r.p95_workload:.1f}",
                f"{r.stores_over_k:.1f}",
            ]
            for n, r in primary.iterrows()
        ],
    )
    report = f"""# 即时零售经营诊断与缺货排查分配

完成日期：2026-11-20。定位：数据分析实习作品集。数据为 2024 年历史记录，数据集于 2025 年公开；不是当前企业经营情况，也不是企业实习经历。

## 先看结论

已在本机完成全部 50,000 条门店商品序列的 SQL 建模、原始 5 轮时间回测、完整模型网格、独立测试与解释性分析，并新增经营诊断与同总名额排查实验。

在原始每日 5,000 个名额下，回测选中的 `{entry}` 测试命中率为 **{pct(best.precision_at_10)}**，当天缺货小时规则为 **{pct(base.precision_at_10)}**，差值 **{pp(best.precision_at_10 - base.precision_at_10)}**。这属于**原开源方案的本次复现成绩**，不是本次发明新算法。模型只能捕获约 **{pct(best.recall_at_10)}** 的次日缺货商品，仍有大量漏报。

本次新增的主场景为 K=5、每日 {g.alerts:.0f} 个名额：全局风险优先覆盖 **{pct(g.store_coverage)}** 的门店；每店至少 1 个可覆盖 **{pct(f.store_coverage)}**，其命中率相对全局策略变化 **{pp(f.precision - g.precision)}**。该结果说明覆盖与效率存在取舍，不等于减少了实际缺货。

## 原项目本来做什么

原项目：[Jacob Willson / freshretail-stockout-prediction](https://github.com/JacobWillson13/freshretail-stockout-prediction)，MIT，固定提交 `a1419c6aabea568a1e6e6ee9faec454791c73283`。它不是叮咚买菜官方缺货预测系统。

原作者已经实现：14 层 SQL 特征管道、3 条风险规则、两特征逻辑回归、5 组特征的逻辑回归与 LightGBM、4 组 LightGBM 消融、5 轮滚动时间回测、每日前 5%/10%/20% 排查评估、门店聚类置信区间、校准、切片、SHAP 与本地推断命令。这些不能写成我们新增的部分。

原任务用 D 日结束后的信息，预测 D+1 在 06:00–22:00（不含 22 点）是否缺货超过 2 小时，然后按风险排序。特征含 D 日 22、23 点记录，因此名单必须等全日数据到齐；不能用于更早的采购下单。

## 我们完成的复现与新增

| 部分 | 原项目已有 | 本次实际完成 |
|---|---|---|
| 数据与模型 | 原 SQL、特征与完整评估框架 | 固定代码与数据版本，下载全量数据，本机重新建库、回测、训练、评估；保存模型与预测 |
| 经营指标 | 预测研究中的探索分析 | 新增门店日、品类日、城市时段与跨日持续性汇总；明确分母并完成小时/日对账 |
| 交互展示 | 原静态图与 notebook | 新增中文离线看板：日期/城市筛选、门店下钻、时段/品类诊断、策略对比 |
| 名额分配 | 每天全局 top-k% | 新增同总名额的全局、每店上限、覆盖保底三策略，4 个事先确定的 K 场景 |
| 分配策略评估 | 无上述三策略实验 | 新增逐日命中、召回、覆盖、P95 任务量、超上限门店统计与 1,000 次配对门店 bootstrap |
| 可核验性 | 上游测试 | 本次运行上游 43 项测试；新增 11 项约束/标签独立性测试；保留来源哈希、实验方案、日志与作者成绩差异 |

`upstream/` 保留未修改原代码；工作副本仅对 DuckDB 加资源限制。训练使用 6 线程，其余原模型参数与完整网格保留。未使用付费 API、云端训练或真实企业账号。

## 数据与质量核验

来源：[Dingdong-Inc / FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)，CC BY 4.0，固定数据版本见代码包 `provenance.json`。全量 485 万条日记录、1.164 亿条小时记录，898 家门店、865 个商品、18 个城市。营业窗口共 7,760 万个商品小时；所有 5 万条序列均连续 97 天。

- 训练原始记录 450 万，评估原始记录 35 万；主键重复、字段缺失、数组长度异常均为 0。
- 原字段缺货小时数与 `[6..21]` 的缺货状态求和完全匹配；日、时段、门店、品类汇总分母与缺货总数逐层对齐。
- 测试周大于 2 小时的缺货商品日占比 **{pct(test.severe_days.sum() / test.product_days.sum())}**；营业小时缺货占比 **{pct(test.oos_hours.sum() / (16 * test.product_days.sum()))}**；21 点缺货商品日占比 **{pct(test.close_oos_days.sum() / test.product_days.sum())}**。这些分母不同，不能混称“缺货率”。
- 测试周，前一日已超过 2 小时缺货的商品，次日同样缺货概率 **{pct(pt.loc[True, "next_severe_rate"])}**；其余为 **{pct(pt.loc[False, "next_severe_rate"])}**。这只是关联，不是原因证明。
- **标签局限：**标记为缺货的营业小时仍有 **{pct(rawstatus.loc[1, "share_hours_with_sale"])}** 记录存在销量。数据是整点/小时粒度状态，不能把状态简单理解为整小时完全无法交易。
- 销量经统一系数缩放。没有原始货币、进价、毛利与补货记录，因此本作品不输出人民币收益、GMV 或实际损失。

## 模型复现结果

暖启动后建模 415 万条记录：训练 310 万、验证 70 万、测试 35 万。按 **标签日期 D+1** 分割：训练 4 月 11 日–6 月 11 日，验证 6 月 12–25 日，测试 6 月 26 日–7 月 2 日；特征日期均早一天。5 轮验证周都在测试周之前，另留 7 天早停集，模型选择依据平均回测 PR-AUC。预测为每天更新历史的一步预测，不是一次提前预测完整未来一周。

{metric_table}

P@10% 是每天风险最高的 10% 商品中真实缺货的比例，先按天计算再平均；R@10% 是这批名额捕获的实际缺货比例。PR-AUC 此处按 sklearn `average_precision_score` 定义。

所选模型 P@10% 的门店聚类 95% 区间为 **{pct(ci.loc[entry, "precision_at_10_lo"])}–{pct(ci.loc[entry, "precision_at_10_hi"])}**。作者原公布 full LightGBM P@10% 约 65.1%、PR-AUC 约 0.567；本次分别为 {pct(models.loc["lightgbm_full", "precision_at_10"])}、{models.loc["lightgbm_full", "pr_auc"]:.4f}。逐项差异见 `证据/reproduction_vs_author.csv`，没有直接拿作者分数充作本次结果。依赖版本、平台和并行数可能产生数值差异；未逐一做环境差异的因果定位。

## 新增：同总名额分配实验

对 K=1、3、5、10，先令每日总名额等于每店 `min(K, 当日纳入观察商品数)` 的总和，再比较三种分配。这样不会因某策略多发任务而虚增命中数量。K=5 在实验方案中事先固定；没有根据测试结果挑 K。

{policy_table}

表内均为 7 天均值，P95 为每日单店任务量 P95 后按天平均。全局与覆盖保底策略可能超过每店 K 上限，“保证每店至少 1 个”不等于“保证每店最多 K 个”。

保底策略相对全局策略的命中率差为 **{pp(f.precision - g.precision)}**，配对门店 bootstrap 95% 区间 **{pp(floor_ci.lo)} 至 {pp(floor_ci.hi)}**。差值区间跨 0，不能证明准确率提升或统计等效。逐日结果、全部 K、验证周结果、主场景单店任务量和第一日事后核验名单均已导出。

**如何解释：**如果目标只强调命中，可以参考全局风险排序；如果每个门店都需要覆盖，可参考保底方案并评估损失；如果每店确有严格日上限，应采用满足上限的分配。当前人员能力、动作成本与覆盖要求都没有实际调研，所以本项目只量化取舍，不宣布某方案可以直接上线。

## 完成验证

- 上游 43 项测试、新增 11 项测试通过，合计 54 项；新增测试覆盖同日总名额一致、每店上限、覆盖保底、标签不参与选择、并列分数与输入顺序、非法输入、重采样可重复与小样本零分母处理。
- 原报告包含 1,000 次门店聚类置信区间、校准与切片；SHAP 使用 2 万测试样本。置换重要性使用 10 万验证样本、3 次重复，全部是本次实算。
- 另有最后观察日 2024-07-02 的本地推断名单。其目标日 7 月 3 日没有标签，**未计算或编造该日预测效果**。
- 看板按日期、城市筛选并可点击门店查看；策略部分固定测试周，不随经营筛选改变，页面已明确标注。

## 仍存在、无法靠本数据补出的缺陷

1. **缺少实际干预。**没有真实排查、补货或对照实验，不能说“缺货降低 X%”“收入增长 X%”。
2. **可外推时间短。**独立测试仅一周，全部数据约三个月，无法证明换季、春节或大促效果。5 轮回测与门店区间也不能弥补未来时间样本不足。
3. **容量只是情景。**没有人员工时、路线、商品检查成本或真实每店上限。K=5 不能写成实际业务调研结论。
4. **门店与商品已匿名。**不能向具体门店下发建议；缺货相关性也不能区分上游供给、陈列、补货延迟与标签误差。
5. **样本覆盖与目标存在局限。**公开的 5 万条序列不代表门店完整商品盘，名额与覆盖统计只针对样本。大于 2 小时是原预测任务阈值，未按真实损失校准；所有商品在名额计数上等权。
6. **召回仍有限。**前 10% 排查仅捕获约 {pct(best.recall_at_10)} 的缺货商品。增加名额也增加工作量，不能仅强调命中率。
7. **解释与因果不同。**SHAP、分组差异与前后关联不能证明调整促销、人员或库存会带来何种增量收益。
8. **未生产部署、无真实用户验证。**当前有可运行的本地作品与评估，不具备线上监控、实时入库、权限、告警闭环或实际使用反馈；跨店特征能否及时到齐也未经生产数据验证。

## 与目标岗位的关系

本项目主要用于**数据分析／经营分析／业务分析实习**，展示 SQL、口径、诊断、策略比较、回测与表达。它不单独证明 AI 产品的用户需求、PRD、交互与 Agent 评测能力；这些将在后续 AI 项目中另补。

简历不要按原作者工作量冒写“从零设计所有特征与算法”。可写“基于公开数据复现缺货预测流程，新增经营诊断与同总名额排查分配实验”。文件由 AI 辅助完成，投递前仍需要你自己复跑、修改并能解释关键步骤。

## 文件索引

- `经营分析看板.html`：离线交互演示。
- `简历写法与面试准备.md`：可用描述及需要掌握的问题。
- `证据/`：实际运行指标、分配结果、校准、切片与原作者差异。
- `项目代码与复现说明.zip`：源代码、模型、结果、日志、版本与复现入口；不包含可重新下载的原始大文件和数据库。
"""
    (OUT / "项目报告.md").write_text(report)
    resume = f"""# 简历写法与面试准备

以下数字来自本次实际运行。项目性质应标明“公开数据个人项目／开源复现与扩展”；不能当作企业实习成果。你自行复跑、修改、核验并能解释后，再据实际负责范围使用。

## 建议项目名

即时零售经营诊断与缺货排查策略分析｜公开数据复现与扩展

技术：SQL、DuckDB、Python/Pandas、LightGBM、时间回测、配对 bootstrap、Plotly。当前看板为自建 HTML，不应写熟练使用 Power BI/Tableau；Excel、A/B 与因果实验能力也应另有证据。

## 项目经历参考

- 基于 FreshRetailNet-50K 的 **485 万条商品日、898 家门店**，复现 SQL 数据管道，构建门店/品类/时段缺货指标；核验主键、连续日期、24 小时数组与统计分母，完成 **7,760 万个营业商品小时与日汇总对账**。
- 复现缺货排序模型与 **5 轮时间回测**，按次日标签日期隔离测试；全量测试下每日前 10% 名额命中率 **{pct(best.precision_at_10)}**，较当天缺货规则变化 **{pp(best.precision_at_10 - base.precision_at_10)}**，召回率 **{pct(best.recall_at_10)}**。此条应明确是开源模型复现，不宣称自创模型提升。
- 新增 **3 种同总名额分配策略、4 个容量场景**，同时评估命中、召回、门店覆盖与单店负荷；K=5、每日 {g.alerts:.0f} 名额时，覆盖保底使覆盖率由 **{pct(g.store_coverage)}** 变为 **{pct(f.store_coverage)}**，命中率变化 **{pp(f.precision - g.precision)}**，并用 **1,000 次配对门店重采样**报告不确定性。
- 制作中文离线交互看板，支持日期/城市筛选、门店趋势下钻及分配方案比较，保留逐日数据与核验明细；通过 **54 项原有及新增测试**验证数据时序和策略约束。

简历空间有限时保留三条：指标与数据、策略与结果、看板与核验。模型细节可以放项目链接和面试说明。以上条目可按个人实际完成部分合并，不能全照搬为亲手独立完成。

## 不应出现的表述

- “叮咚买菜实习／内部经营项目”“负责 898 家门店运营”：没有该经历。
- “缺货降低 X%、节省 X 元、ROI 提升”：没有实际干预、金额与成本。
- “上线服务 XX 用户”“与业务团队推进落地”：没有对应用户与协作证据。
- “完成 A/B 测试”：本项目是离线预测与分配比较，没有在线随机干预。
- “独立设计全部 SQL 特征、算法和 SHAP”：主体继承原作者，应标明复现。

## 投递前至少自己完成的练习

1. 从原始 parquet 重建一个门店一周的缺货指标，手算商品日率与小时率，解释为什么不同。
2. 修改 K=5 为其他容量，独立复跑并解释名额、覆盖与命中为何改变；不要只记结果。
3. 自己新增一个有依据的指标或限制（例如只分析闭店仍缺货、按商品价值权重的局限讨论），保留修改与复测记录。
4. 随机抽查 5 条预测：确认特征日 D、标签日 D+1、分配依据与结果，找出一个预测失败的例子。
5. 用 5 分钟讲清业务问题、方法、取舍与无法验证的内容；现场能改 SQL 或解释代码。

## 高频追问与回答要点

| 追问 | 应解释的内容 |
|---|---|
| 为什么定义大于 2 小时？ | 继承原任务阈值，代表较长缺货；没有实际损失阈值验证，需做敏感性分析才能改任务 |
| 什么时间生成次日名单？ | D 日全日数据到齐后，含 23 点；不能用它支持 D 日白天更早的采购 |
| 怎样排除时间泄漏？ | 按标签日期切分，滚动训练；早停与验证分开，上游含修改未来数据的泄漏测试 |
| 为什么三种策略可以比较？ | 每日总名额相同；同一模型、同一候选集与日期，只改变名额在门店之间的分配 |
| 哪种最好？ | 取决于覆盖与门店上限，报告效率代价；没有真实工时调研，不能直接下业务结论 |
| 为什么不用准确率？ | 实际关心有限名额中命中多少、漏掉多少；准确率不能描述排查工作负担 |
| bootstrap 为什么按门店？ | 同店商品与日期相关；保留整店记录，配对同一抽样，仍不能代表未来周 |
| 为什么还不能写业务提升？ | 预测识别不等于可避免；还需真实动作、执行率、对照与成本数据 |
| 项目哪些是你新增的？ | 新增业务汇总、看板、同总名额分配、负荷与覆盖评估；原预测框架来自开源 |
| 做过什么产品工作？ | 本项目主要证明分析能力，不能替代真实需求、原型、用户试用与 Agent 评测 |

## 如何继续补强

优先补真实使用反馈或一次个人修改验证，而不是继续堆特征名称。若目标 JD 要求 Power BI/Excel，可在同一已核验汇总表上亲手制作对应版本；当前交付未冒称这些工具经验。若有门店从业者可交流，再访谈检查容量、动作和提前期，区分访谈发现与个人假设。
"""
    (OUT / "简历写法与面试准备.md").write_text(resume)
    evidence = OUT / "证据"
    evidence.mkdir(exist_ok=True)
    for directory in [tabs, alloc, biz]:
        dest = evidence / directory.name
        shutil.copytree(directory, dest, dirs_exist_ok=True)
    shutil.copy2(
        ROOT / "results/reproduction_vs_author.csv",
        evidence / "reproduction_vs_author.csv",
    )
    changes = []
    for path in (ROOT / "analysis/src").glob("*.py"):
        before = ROOT / "upstream/src" / path.name
        if before.exists() and path.read_bytes() != before.read_bytes():
            changes.extend(
                difflib.unified_diff(
                    before.read_text().splitlines(keepends=True),
                    path.read_text().splitlines(keepends=True),
                    fromfile="upstream/src/" + path.name,
                    tofile="analysis/src/" + path.name,
                )
            )
    (ROOT / "upstream_changes.patch").write_text("".join(changes))
    manifest = {}
    include_roots = [
        "analysis/src",
        "analysis/sql",
        "analysis/tests",
        "analysis/scripts",
        "analysis/docs",
        "analysis/reports",
        "analysis/models",
        "upstream/src",
        "upstream/sql",
        "additions",
        "results",
        "logs",
    ]
    files = []
    for directory in include_roots:
        files.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not p.name.endswith(".pyc")
        )
    files.extend(
        ROOT / p
        for p in [
            "README.md",
            "STATUS.md",
            "run.py",
            "setup_macos_runtime.py",
            "provenance.json",
            "experiment_plan.json",
            "requirements-lock.txt",
            "libomp_provenance.json",
            "upstream_changes.patch",
            "analysis/LICENSE",
            "analysis/README.md",
            "analysis/pyproject.toml",
            "upstream/LICENSE",
            "upstream/README.md",
            "upstream/reports/tables/model_comparison_test.csv",
        ]
    )
    files.extend(
        ROOT / "analysis/data" / name
        for name in ["predictions.parquet", "predictions_valid.parquet"]
    )
    for p in sorted(set(files)):
        manifest[str(p.relative_to(ROOT))] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    (ROOT / "delivery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    files.append(ROOT / "delivery_manifest.json")
    with zipfile.ZipFile(
        OUT / "项目代码与复现说明.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(set(files)):
            z.write(p, arcname="retail_analysis/" + str(p.relative_to(ROOT)))
    print("report, resume guide, evidence and code archive complete")


if __name__ == "__main__":
    main()
