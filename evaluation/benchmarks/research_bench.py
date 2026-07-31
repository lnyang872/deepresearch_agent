#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/benchmarks/research_bench.py
================================================================================
自建深度研究评测集 (ResearchBench)。

包含 50 道跨领域深度研究题目。
每道题附带 expected_topics、ground_truth、知识截止日期和审计状态。
================================================================================
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any


class ResearchBench:
    """
    自建深度研究评测集。
    """

    BENCHMARK_VERSION = "1.1"
    DEFAULT_AS_OF_DATE = "2024-12-31"

    # 这 17 道旧题包含动态措辞、口径含糊或未经来源约束的数值，正式实验前需重写 ground truth。
    LEGACY_REWRITE_IDS = {
        "tech_002", "tech_003", "med_002", "med_003", "fin_002", "fin_003",
        "tech_008", "edu_001", "law_001", "energy_001", "energy_002",
        "retail_001", "auto_001", "auto_002", "media_001", "cross_001", "cross_002",
    }

    # 10 题 demo 覆盖 10 个主题域，且优先选择事实边界清晰、可由官方来源复核的题。
    DEMO_IDS = (
        "tech_001", "med_004", "fin_001", "edu_002", "law_002",
        "fin_006", "cyber_001", "climate_002", "science_001", "ai_safety_001",
    )

    LEGACY_REVIEWED_METADATA: dict[str, dict[str, Any]] = {
        "tech_001": {
            "reference_urls": [
                "https://openai.com/index/hello-gpt-4o/",
                "https://www.anthropic.com/news/claude-3-5-sonnet",
                "https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/",
                "https://qwenlm.github.io/blog/qwen2.5/",
            ],
            "audit_status": "reviewed",
        },
        "med_004": {
            "reference_urls": [
                "https://www.fda.gov/news-events/press-announcements/fda-approves-first-gene-therapies-treat-patients-sickle-cell-disease"
            ],
            "audit_status": "reviewed",
        },
        "fin_001": {
            "reference_urls": [
                "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm"
            ],
            "audit_status": "reviewed",
        },
        "edu_002": {
            "reference_urls": [
                "https://www.gov.cn/zhengce/2021-07/24/content_5627132.htm",
                "https://www.nextgenscience.org/",
            ],
            "audit_status": "reviewed",
        },
        "law_002": {
            "reference_urls": [
                "https://eur-lex.europa.eu/eli/reg/2016/679/oj",
                "http://www.npc.gov.cn/englishnpc/c2759/c23934/202112/t20211209_385109.html",
            ],
            "audit_status": "reviewed",
        },
        "fin_006": {
            "reference_urls": [
                "https://www.icmagroup.org/sustainable-finance/the-principles-guidelines-and-handbooks/green-bond-principles-gbp/",
                "https://www.icmagroup.org/sustainable-finance/the-principles-guidelines-and-handbooks/sustainability-linked-bond-principles-slbp/",
            ],
            "audit_status": "reviewed",
        },
    }

    # 原始 35 题。保留 ID，避免已有结果失效。
    DEFAULT_QUESTIONS: list[dict[str, Any]] = [
        {
            "id": "tech_001",
            "domain": "科技",
            "query": "对比分析 2024 年主流大语言模型（GPT-4o、Claude 3.5、Gemini 1.5、Qwen2.5）在中文推理、代码生成和长上下文任务上的表现差异，并分析其技术路线差异。",
            "expected_topics": ["中文推理", "代码生成", "长上下文", "技术路线", "GPT-4o", "Claude 3.5", "Gemini 1.5", "Qwen2.5"],
            "ground_truth": {
                "GPT-4o": "OpenAI 发布于 2024 年 5 月，原生多模态",
                "Claude 3.5 Sonnet": "Anthropic 发布于 2024 年 6 月，Artifacts 功能",
                "Gemini 1.5 Pro": "Google 发布，1M+ token 上下文窗口",
                "Qwen2.5": "阿里巴巴发布，开源并支持 128K 上下文",
            },
        },
        {
            "id": "tech_002",
            "domain": "科技",
            "query": "分析当前 AI 芯片市场格局，比较 NVIDIA、AMD、Intel 及中国厂商（华为昇腾、寒武纪）在训练与推理场景下的竞争力。",
            "expected_topics": ["NVIDIA", "AMD", "Intel", "华为昇腾", "寒武纪", "训练", "推理", "AI芯片"],
            "ground_truth": {
                "NVIDIA": "H100/H200 占据训练市场主导地位",
                "AMD": "MI300 系列作为追赶者",
                "昇腾": "华为 AI 芯片，受限于先进制程",
            },
        },
        {
            "id": "med_001",
            "domain": "医疗",
            "query": "综述 2023-2024 年 mRNA 癌症疫苗临床试验进展，分析其技术原理、关键试验数据及面临的挑战。",
            "expected_topics": ["mRNA", "癌症疫苗", "临床试验", "技术原理", "关键数据", "挑战"],
            "ground_truth": {
                "Moderna": "mRNA-4157 联合 Keytruda 的 2b 期临床结果",
                "BioNTech": "BNT122 在结直肠癌中的研究",
                "技术原理": "个性化新抗原疫苗",
            },
        },
        {
            "id": "med_002",
            "domain": "医疗",
            "query": "评估 GLP-1 受体激动剂（司美格鲁肽、替尔泊肽）在减重以外的潜在医疗应用，包括心血管保护、认知功能和成瘾治疗。",
            "expected_topics": ["GLP-1", "司美格鲁肽", "替尔泊肽", "心血管", "认知功能", "成瘾治疗"],
            "ground_truth": {
                "SELECT 试验": "司美格鲁肽降低心血管风险 20%",
                "SURMOUNT": "替尔泊肽减重效果",
            },
        },
        {
            "id": "fin_001",
            "domain": "金融",
            "query": "分析美联储 2024 年降息周期对全球资本流动、新兴市场汇率和中国货币政策的影响路径。",
            "expected_topics": ["美联储", "降息", "资本流动", "新兴市场", "汇率", "中国货币政策"],
            "ground_truth": {
                "降息": "2024 年 9 月首次降息 50bp",
                "新兴市场": "资本回流与汇率波动",
            },
        },
        {
            "id": "fin_002",
            "domain": "金融",
            "query": "比较比特币现货 ETF 与黄金 ETF 在机构资产配置中的异同，分析其风险收益特征和监管差异。",
            "expected_topics": ["比特币ETF", "黄金ETF", "机构配置", "风险收益", "监管"],
            "ground_truth": {
                "现货比特币 ETF": "2024 年 1 月美国 SEC 批准",
                "IBIT": "BlackRock 比特币 ETF 规模",
            },
        },
        {
            "id": "tech_003",
            "domain": "科技",
            "query": "探讨具身智能（Embodied AI）在机器人领域的最新进展，分析世界模型、触觉感知和任务规划三大技术瓶颈。",
            "expected_topics": ["具身智能", "机器人", "世界模型", "触觉感知", "任务规划"],
            "ground_truth": {
                "Figure AI": "人形机器人 Figure 02",
                "Tesla Optimus": "特斯拉人形机器人进展",
            },
        },
        {
            "id": "tech_004",
            "domain": "科技",
            "query": "分析 RISC-V 架构在服务器和 AI 加速器领域的生态发展现状，对比 x86 和 ARM 的优劣势。",
            "expected_topics": ["RISC-V", "服务器", "AI加速器", "x86", "ARM"],
            "ground_truth": {
                "RISC-V": "开源指令集架构",
                "SiFive": "高性能 RISC-V 处理器",
            },
        },
        {
            "id": "med_003",
            "domain": "医疗",
            "query": "综述阿尔茨海默病早期诊断生物标志物（血液 p-tau217、Aβ42/40 比值）的最新临床验证进展。",
            "expected_topics": ["阿尔茨海默病", "早期诊断", "p-tau217", "Aβ42/40", "生物标志物"],
            "ground_truth": {
                "p-tau217": "血液检测灵敏度超过 90%",
                "Aβ PET": "与血液标志物的一致性",
            },
        },
        {
            "id": "med_004",
            "domain": "医疗",
            "query": "评估 CRISPR 基因编辑疗法在镰状细胞病和 β 地中海贫血中的临床疗效与长期安全性数据。",
            "expected_topics": ["CRISPR", "基因编辑", "镰状细胞病", "地中海贫血", "安全性"],
            "ground_truth": {
                "Casgevy": "首个获批的 CRISPR 基因编辑疗法",
                "Vertex": "与 CRISPR Therapeutics 合作",
            },
        },
        {
            "id": "fin_003",
            "domain": "金融",
            "query": "分析中国地方政府债务化解的最新政策工具（特殊再融资债券、债务置换）及其对银行体系的影响。",
            "expected_topics": ["地方政府债务", "再融资债券", "债务置换", "银行体系"],
            "ground_truth": {
                "特殊再融资债券": "2023-2024 年大规模发行",
                "化债": "一揽子化债方案",
            },
        },
        {
            "id": "fin_004",
            "domain": "金融",
            "query": "比较被动指数基金（ETF）与主动管理基金在 2020-2024 年间的风险调整后收益表现，分析其背后的市场结构变化。",
            "expected_topics": ["ETF", "主动管理基金", "风险调整后收益", "市场结构"],
            "ground_truth": {
                "SPIVA": "标普指数 vs 主动基金长期业绩对比",
                "费率": "ETF 低费率优势",
            },
        },
        {
            "id": "tech_005",
            "domain": "科技",
            "query": "探讨空间计算（Spatial Computing）和 Apple Vision Pro 对混合现实产业生态的影响，分析内容创作、企业应用和消费级落地的关键障碍。",
            "expected_topics": ["空间计算", "Apple Vision Pro", "混合现实", "内容创作", "企业应用"],
            "ground_truth": {
                "Vision Pro": "苹果首款空间计算设备，2024 年发售",
                "passthrough": "VST 视频透视技术",
            },
        },
        {
            "id": "tech_006",
            "domain": "科技",
            "query": "分析自动驾驶 L3/L4 级别在 2024 年的商业化落地进展，比较 Waymo、特斯拉 FSD 和中国厂商（百度、小鹏、华为）的技术路线差异。",
            "expected_topics": ["自动驾驶", "L3", "L4", "Waymo", "特斯拉FSD", "百度", "小鹏", "华为"],
            "ground_truth": {
                "Waymo": "纯视觉+激光雷达融合方案，Robotaxi 运营",
                "特斯拉 FSD": "端到端神经网络，纯视觉方案",
            },
        },
        {
            "id": "med_005",
            "domain": "医疗",
            "query": "综述 2024 年 WHO 关注的 X 疾病（Disease X）大流行防范准备框架，分析疫苗平台技术、监测网络和全球治理机制。",
            "expected_topics": ["Disease X", "WHO", "大流行防范", "疫苗平台", "监测网络", "全球治理"],
            "ground_truth": {
                "Disease X": "WHO 定义的下一次未知大流行病原体",
                "100 Days Mission": "100 天内开发疫苗的目标",
            },
        },
        {
            "id": "med_006",
            "domain": "医疗",
            "query": "评估数字疗法（Digital Therapeutics）在慢病管理（糖尿病、高血压、抑郁症）中的临床证据、监管路径和商业化挑战。",
            "expected_topics": ["数字疗法", "慢病管理", "糖尿病", "高血压", "抑郁症", "监管"],
            "ground_truth": {
                "DTx": "经临床验证的软件干预手段",
                "Pear Therapeutics": "破产案例与商业化困境",
            },
        },
        {
            "id": "fin_005",
            "domain": "金融",
            "query": "分析人工智能对保险行业精算、承保和理赔环节的影响，评估保险科技（InsurTech）初创企业的竞争格局。",
            "expected_topics": ["人工智能", "保险", "精算", "承保", "理赔", "InsurTech"],
            "ground_truth": {
                "Lemonade": "AI 驱动的保险理赔",
                "telematics": "UBI 基于使用的保险",
            },
        },
        {
            "id": "fin_006",
            "domain": "金融",
            "query": "比较绿色债券（Green Bond）与可持续发展挂钩债券（SLB）在募集资金用途、信息披露和投资者保护方面的差异。",
            "expected_topics": ["绿色债券", "可持续发展挂钩债券", "募集资金", "信息披露", "投资者保护"],
            "ground_truth": {
                "ICMA": "绿色债券原则 GBP",
                "SLB": "票面利率与可持续发展 KPI 挂钩",
            },
        },
        {
            "id": "tech_007",
            "domain": "科技",
            "query": "探讨量子计算在密码学（后量子密码 PQC）和药物发现领域的应用前景，分析 NIST 标准化进程和当前技术瓶颈。",
            "expected_topics": ["量子计算", "后量子密码", "PQC", "药物发现", "NIST"],
            "ground_truth": {
                "NIST PQC": "2024 年发布首批标准化算法",
                "CRYSTALS-Kyber": "密钥封装机制标准",
            },
        },
        {
            "id": "tech_008",
            "domain": "科技",
            "query": "分析生成式 AI 在软件开发领域的应用现状（GitHub Copilot、Devin 等），评估其对开发者生产力、代码质量和软件工程教育的影响。",
            "expected_topics": ["生成式AI", "软件开发", "GitHub Copilot", "Devin", "开发者生产力", "代码质量"],
            "ground_truth": {
                "GitHub Copilot": "基于 OpenAI Codex 的代码补全工具",
                "Devin": "Cognition AI 发布的全自主 AI 软件工程师",
            },
        },
        # ------------------------------------------------------------------
        # 教育 (2题)
        # ------------------------------------------------------------------
        {
            "id": "edu_001",
            "domain": "教育",
            "query": "评估自适应学习系统（如 Khan Academy、松鼠 AI）在 K-12 数学教育中的效果，分析其个性化推荐算法、学习效果量化指标和师生接受度。",
            "expected_topics": ["自适应学习", "K-12", "数学教育", "个性化推荐", "学习效果", "Khan Academy", "松鼠AI"],
            "ground_truth": {
                "Khan Academy": "非营利性教育平台，提供免费个性化练习",
                "松鼠 AI": "中国自适应学习公司，智适应教育系统",
                "效果": "自适应学习平均提升成绩 10-20%",
            },
        },
        {
            "id": "edu_002",
            "domain": "教育",
            "query": "对比分析中美两国 STEM 教育的政策差异、课程设计和师资培养模式，评估中国「双减」政策对 STEM 课外培训的影响。",
            "expected_topics": ["STEM教育", "中美对比", "双减政策", "课程设计", "师资培养", "课外培训"],
            "ground_truth": {
                "双减": "2021 年中国减轻义务教育阶段学生作业和校外培训负担",
                "STEM": "科学、技术、工程、数学跨学科教育",
                "Next Generation Science Standards": "美国 K-12 科学教育标准",
            },
        },
        # ------------------------------------------------------------------
        # 法律 (2题)
        # ------------------------------------------------------------------
        {
            "id": "law_001",
            "domain": "法律",
            "query": "分析生成式 AI 训练数据中的版权问题，比较美国合理使用原则（Fair Use）与欧盟《AI 法案》在训练数据授权方面的法律冲突。",
            "expected_topics": ["AI版权", "训练数据", "Fair Use", "AI法案", "欧盟", "纽约时报诉OpenAI"],
            "ground_truth": {
                "纽约时报诉OpenAI": "2023 年纽约时报起诉 OpenAI 和微软侵犯版权",
                "欧盟AI法案": "2024 年生效的全球首部全面 AI 监管法规",
                "Fair Use": "美国版权法中的合理使用抗辩",
            },
        },
        {
            "id": "law_002",
            "domain": "法律",
            "query": "评估中国《个人信息保护法》和欧盟 GDPR 在数据跨境传输规则上的差异，分析对企业出海合规成本的影响。",
            "expected_topics": ["个人信息保护法", "GDPR", "数据跨境", "合规成本", "企业出海", "标准合同条款"],
            "ground_truth": {
                "GDPR": "欧盟通用数据保护条例，2018 年生效",
                "个人信息保护法": "中国 2021 年生效的数据隐私法规",
                "标准合同条款": "SCCs，数据跨境传输的主要合规工具",
            },
        },
        # ------------------------------------------------------------------
        # 能源 (2题)
        # ------------------------------------------------------------------
        {
            "id": "energy_001",
            "domain": "能源",
            "query": "对比固态电池、钠离子电池和磷酸铁锂电池在能量密度、安全性和成本上的技术路线差异，评估其对电动车产业的影响。",
            "expected_topics": ["固态电池", "钠离子电池", "磷酸铁锂", "能量密度", "电动车", "宁德时代", "QuantumScape"],
            "ground_truth": {
                "固态电池": "能量密度可达 500 Wh/kg，预计 2027-2030 量产",
                "宁德时代": "全球动力电池龙头，发布凝聚态电池",
                "磷酸铁锂": "成本低、安全性高，但能量密度约 160 Wh/kg",
            },
        },
        {
            "id": "energy_002",
            "domain": "能源",
            "query": "分析中国光伏产业从硅料到组件的全产业链竞争力，评估美国《通胀削减法案》(IRA) 对中国光伏出口的影响。",
            "expected_topics": ["光伏", "硅料", "组件", "IRA", "通胀削减法案", "隆基绿能", "通威股份"],
            "ground_truth": {
                "隆基绿能": "全球最大单晶硅片制造商",
                "IRA": "美国 2022 年通胀削减法案，提供光伏税收抵免",
                "中国光伏": "占全球组件产能 80% 以上",
            },
        },
        # ------------------------------------------------------------------
        # 消费零售 (2题)
        # ------------------------------------------------------------------
        {
            "id": "retail_001",
            "domain": "消费",
            "query": "评估直播电商（抖音电商、淘宝直播）对传统货架电商的替代效应，分析其供应链模式、主播佣金结构和退货率问题。",
            "expected_topics": ["直播电商", "抖音电商", "淘宝直播", "货架电商", "主播佣金", "退货率"],
            "ground_truth": {
                "直播电商": "2024 年中国直播电商规模预计超 5 万亿元",
                "退货率": "直播电商退货率 30-50%，远高于传统电商",
                "东方甄选": "新东方旗下直播带货品牌",
            },
        },
        {
            "id": "retail_002",
            "domain": "消费",
            "query": "分析中国新消费品牌（喜茶、完美日记、泡泡玛特）的崛起路径、供应链策略和海外扩张挑战。",
            "expected_topics": ["新消费", "喜茶", "完美日记", "泡泡玛特", "DTC", "海外扩张"],
            "ground_truth": {
                "泡泡玛特": "盲盒潮玩龙头，海外收入占比持续提升",
                "喜茶": "新茶饮代表品牌，开放加盟加速下沉",
                "完美日记": "逸仙电商旗下美妆品牌，面临盈利压力",
            },
        },
        # ------------------------------------------------------------------
        # 汽车 (2题)
        # ------------------------------------------------------------------
        {
            "id": "auto_001",
            "domain": "汽车",
            "query": "对比比亚迪、特斯拉和理想汽车在增程式/纯电技术路线、智能驾驶能力和全球化策略上的差异。",
            "expected_topics": ["比亚迪", "特斯拉", "理想汽车", "增程式", "智能驾驶", "全球化"],
            "ground_truth": {
                "比亚迪": "2024 年全球新能源车销量第一，垂直整合模式",
                "理想汽车": "增程式 SUV 路线，家庭用户定位",
                "FSD": "特斯拉完全自动驾驶能力，端到端神经网络",
            },
        },
        {
            "id": "auto_002",
            "domain": "汽车",
            "query": "评估 2024-2025 年中国新能源车渗透率超过 50% 后对燃油车产业链（经销商、加油站、零部件）的冲击和转型路径。",
            "expected_topics": ["新能源车渗透率", "燃油车", "经销商", "加油站", "零部件", "转型"],
            "ground_truth": {
                "渗透率": "2024 年中国新能源车零售渗透率突破 50%",
                "经销商": "传统 4S 店大面积关闭或转型新能源",
                "充电桩": "公共充电桩数量快速增长",
            },
        },
        # ------------------------------------------------------------------
        # 游戏 (2题)
        # ------------------------------------------------------------------
        {
            "id": "game_001",
            "domain": "游戏",
            "query": "分析 AI NPC、程序化内容生成（PCG）和动态难度调整（DDA）在游戏开发中的应用现状，评估其对游戏体验和开发成本的影响。",
            "expected_topics": ["AI NPC", "PCG", "程序化生成", "动态难度", "游戏体验", "开发成本"],
            "ground_truth": {
                "AI NPC": "英伟达 ACE、网易伏羲等 AI 角色技术",
                "PCG": "《无人深空》《我的世界》为代表的程序化生成",
                "DDA": "动态难度调整，根据玩家能力实时调节",
            },
        },
        {
            "id": "game_002",
            "domain": "游戏",
            "query": "评估中国游戏出海（《原神》《PUBG Mobile》《黑神话：悟空》）的全球化策略、文化本地化挑战和各国监管差异。",
            "expected_topics": ["游戏出海", "原神", "PUBG Mobile", "黑神话悟空", "本地化", "监管"],
            "ground_truth": {
                "原神": "米哈游开发，全球收入最高的国产游戏之一",
                "黑神话悟空": "游戏科学开发，2024 年 3A 动作游戏",
                "版号": "中国游戏出版审批制度",
            },
        },
        # ------------------------------------------------------------------
        # 传媒 (1题)
        # ------------------------------------------------------------------
        {
            "id": "media_001",
            "domain": "传媒",
            "query": "分析短视频平台（TikTok / 抖音）推荐算法的核心机制，评估其对用户注意力、内容创作生态和信息茧房的影响。",
            "expected_topics": ["TikTok", "抖音", "推荐算法", "注意力经济", "内容生态", "信息茧房"],
            "ground_truth": {
                "算法": "协同过滤 + 深度学习排序，多目标优化",
                "信息茧房": "算法推荐导致用户视野窄化",
                "TikTok": "全球月活超 15 亿，字节跳动旗下",
            },
        },
        {
            "id": "cross_001",
            "domain": "交叉",
            "query": "分析全球半导体供应链的地缘政治风险，评估台积电、三星和 Intel 在先进制程上的产能分布及各国的「芯片法案」补贴效果。",
            "expected_topics": ["半导体供应链", "地缘政治", "台积电", "三星", "Intel", "芯片法案", "先进制程"],
            "ground_truth": {
                "台积电": "全球最先进制程（3nm/2nm）主要制造商",
                "CHIPS Act": "美国 2022 年芯片法案，补贴 520 亿美元",
                "产能分布": "台湾占全球先进芯片代工 90% 以上",
            },
        },
        {
            "id": "cross_002",
            "domain": "交叉",
            "query": "评估 AI 制药（AlphaFold、Atomwise 等）在靶点发现和临床试验设计中的进展，分析其对传统 Pharma 研发投入回报率的潜在改变。",
            "expected_topics": ["AI制药", "AlphaFold", "靶点发现", "临床试验", "Pharma", "研发回报率"],
            "ground_truth": {
                "AlphaFold": "DeepMind 开发的蛋白质结构预测系统",
                "AI制药": "可缩短药物发现周期 30-50%",
                "Atomwise": "AI 驱动的虚拟筛选平台",
            },
        },
    ]

    # v1.1 新增 15 题。ground truth 仅使用截至 2024-12-31 已发生且有官方来源的事实。
    EXTENDED_QUESTIONS: list[dict[str, Any]] = [
        {
            "id": "cyber_001",
            "domain": "网络安全",
            "query": "比较零信任架构与传统边界防御的信任模型、实施路径和迁移成本，并分析身份、设备、网络、应用和数据治理之间的关系。",
            "expected_topics": ["零信任", "边界防御", "身份", "设备", "网络", "应用", "数据治理"],
            "ground_truth": {
                "NIST SP 800-207": "将零信任描述为面向资源保护的架构方法，不基于网络位置给予隐式信任",
                "CISA 成熟度模型": "围绕身份、设备、网络、应用与工作负载、数据五个支柱组织能力",
            },
            "reference_urls": [
                "https://csrc.nist.gov/pubs/sp/800/207/final",
                "https://www.cisa.gov/resources-tools/resources/zero-trust-maturity-model",
            ],
            "audit_status": "reviewed",
        },
        {
            "id": "cyber_002",
            "domain": "网络安全",
            "query": "分析软件供应链安全中的 SBOM、依赖签名、构建可复现性和漏洞响应机制，比较其对开源项目与企业软件的实施难点。",
            "expected_topics": ["软件供应链", "SBOM", "依赖签名", "可复现构建", "漏洞响应", "开源软件"],
            "ground_truth": {
                "美国行政令 14028": "2021 年行政令要求改进联邦政府网络安全并推动软件供应链安全要求",
                "SBOM": "软件物料清单记录软件组件及其供应链关系，NTIA 发布了最低要素说明",
            },
            "reference_urls": [
                "https://www.whitehouse.gov/briefing-room/presidential-actions/2021/05/12/executive-order-on-improving-the-nations-cybersecurity/",
                "https://www.ntia.gov/report/2021/minimum-elements-software-bill-materials-sbom",
            ],
            "audit_status": "reviewed",
        },
        {
            "id": "climate_001",
            "domain": "气候",
            "query": "比较欧盟碳边境调节机制、欧盟碳市场与中国全国碳市场的覆盖范围、价格信号和碳泄漏治理逻辑。",
            "expected_topics": ["CBAM", "EU ETS", "中国碳市场", "碳价格", "碳泄漏", "排放核算"],
            "ground_truth": {
                "CBAM 过渡期": "欧盟 CBAM 过渡期于 2023 年 10 月开始",
                "中国全国碳市场": "全国碳排放权交易市场于 2021 年启动上线交易，初期覆盖发电行业",
            },
            "reference_urls": [
                "https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en",
                "https://www.mee.gov.cn/ywdt/hjywnews/202107/t20210716_848093.shtml",
            ],
            "audit_status": "reviewed",
        },
        {
            "id": "climate_002",
            "domain": "气候",
            "query": "基于 IPCC 第六次评估报告，解释 1.5 摄氏度路径中的剩余碳预算、净零二氧化碳、甲烷减排与负排放技术之间的关系。",
            "expected_topics": ["IPCC AR6", "1.5摄氏度", "碳预算", "净零", "甲烷", "负排放"],
            "ground_truth": {
                "人为变暖": "IPCC AR6 认为人类活动已经明确导致全球变暖",
                "净零二氧化碳": "限制二氧化碳导致的升温需要实现净零二氧化碳排放",
            },
            "reference_urls": ["https://www.ipcc.ch/report/ar6/syr/"],
            "audit_status": "reviewed",
        },
        {
            "id": "public_001",
            "domain": "公共卫生",
            "query": "分析抗微生物药物耐药性监测、抗生素合理使用、医院感染控制与 One Health 治理之间的协同关系。",
            "expected_topics": ["AMR", "抗生素", "监测", "医院感染", "One Health", "GLASS"],
            "ground_truth": {
                "GLASS": "WHO 的全球抗微生物药物耐药性和使用监测系统用于标准化收集相关数据",
                "One Health": "AMR 治理需要统筹人类、动物、食品与环境健康",
            },
            "reference_urls": [
                "https://www.who.int/initiatives/glass",
                "https://www.who.int/health-topics/antimicrobial-resistance",
            ],
            "audit_status": "reviewed",
        },
        {
            "id": "science_001",
            "domain": "科学",
            "query": "解释惯性约束聚变实现点火的物理含义，比较靶丸能量增益、设施总能耗和商业聚变发电可行性三个不同口径。",
            "expected_topics": ["惯性约束聚变", "NIF", "点火", "能量增益", "设施能耗", "商业化"],
            "ground_truth": {
                "首次点火": "美国国家点火装置在 2022 年 12 月实验中实现聚变靶增益大于 1",
                "能量口径": "靶增益只比较到达靶丸的激光能量与聚变输出，不等同于整套设施净发电",
            },
            "reference_urls": ["https://www.energy.gov/articles/doe-national-laboratory-makes-history-achieving-fusion-ignition"],
            "audit_status": "reviewed",
        },
        {
            "id": "science_002",
            "domain": "科学",
            "query": "评估 DART 任务对行星防御的验证价值，说明动量转移、碎屑喷射、轨道周期变化与真实近地小行星风险之间的关系。",
            "expected_topics": ["DART", "行星防御", "动量转移", "Dimorphos", "轨道周期", "近地小行星"],
            "ground_truth": {
                "撞击时间": "DART 于 2022 年 9 月撞击小行星卫星 Dimorphos",
                "轨道变化": "撞击使 Dimorphos 绕 Didymos 的轨道周期缩短约 32 分钟",
            },
            "reference_urls": ["https://www.nasa.gov/planetarydefense/dart/"],
            "audit_status": "reviewed",
        },
        {
            "id": "labor_001",
            "domain": "劳动经济",
            "query": "分析生成式 AI 对文职、专业技术和服务岗位的任务暴露差异，区分岗位替代、任务自动化与人机增强三种效应。",
            "expected_topics": ["生成式AI", "任务暴露", "文职岗位", "自动化", "岗位替代", "人机增强"],
            "ground_truth": {
                "ILO 结论": "ILO 研究认为生成式 AI 更可能增强而非完全替代多数岗位",
                "高暴露岗位": "文职支持类工作在 ILO 分析中具有较高的生成式 AI 暴露度",
            },
            "reference_urls": ["https://www.ilo.org/publications/generative-ai-and-jobs-global-analysis-potential-effects-job-quantity-and"],
            "audit_status": "reviewed",
        },
        {
            "id": "gov_001",
            "domain": "数字治理",
            "query": "比较零售型央行数字货币与商业银行存款、电子支付工具和稳定币的负债主体、隐私设计与金融稳定影响。",
            "expected_topics": ["CBDC", "央行负债", "商业银行存款", "电子支付", "稳定币", "隐私", "金融稳定"],
            "ground_truth": {
                "CBDC": "央行数字货币是中央银行的数字负债，与商业银行存款的发行主体不同",
                "数字欧元": "欧洲央行于 2023 年 11 月进入数字欧元准备阶段",
            },
            "reference_urls": [
                "https://www.bis.org/publ/othp33.htm",
                "https://www.ecb.europa.eu/euro/digital_euro/html/index.en.html",
            ],
            "audit_status": "reviewed",
        },
        {
            "id": "ai_safety_001",
            "domain": "AI治理",
            "query": "比较 NIST AI 风险管理框架与欧盟 AI 法案的治理逻辑，分析自愿风险管理与强制风险分级监管如何衔接。",
            "expected_topics": ["NIST AI RMF", "欧盟AI法案", "风险管理", "风险分级", "治理", "合规"],
            "ground_truth": {
                "NIST AI RMF": "NIST AI RMF 1.0 于 2023 年发布，核心函数为 Govern、Map、Measure、Manage",
                "欧盟 AI 法案": "欧盟 AI 法案采用基于风险的分级监管方法，并于 2024 年生效后分阶段适用",
            },
            "reference_urls": [
                "https://www.nist.gov/itl/ai-risk-management-framework",
                "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai",
            ],
            "audit_status": "reviewed",
        },
        {
            "id": "space_001",
            "domain": "航天",
            "query": "分析 Artemis I 无人试飞对深空载人任务的验证意义，区分运载火箭、猎户座飞船、热防护和地面系统的风险。",
            "expected_topics": ["Artemis I", "SLS", "猎户座", "热防护", "载人航天", "月球"],
            "ground_truth": {
                "任务性质": "Artemis I 是一次无人绕月试飞",
                "任务时间": "Artemis I 于 2022 年发射并完成猎户座飞船返回地球",
            },
            "reference_urls": ["https://www.nasa.gov/mission/artemis-i/"],
            "audit_status": "reviewed",
        },
        {
            "id": "ocean_001",
            "domain": "航运",
            "query": "分析国际航运脱碳中的船舶能效、替代燃料、碳定价和港口基础设施约束，比较甲醇、氨和液化天然气路线。",
            "expected_topics": ["IMO", "航运脱碳", "能效", "甲醇", "氨", "LNG", "港口"],
            "ground_truth": {
                "IMO 2023 战略": "IMO 2023 温室气体战略提出国际航运在 2050 年前后实现净零排放",
                "全生命周期": "替代燃料比较需要考虑从生产到船上使用的全生命周期排放",
            },
            "reference_urls": ["https://www.imo.org/en/MediaCentre/HotTopics/Pages/Cutting-GHG-emissions.aspx"],
            "audit_status": "reviewed",
        },
        {
            "id": "data_001",
            "domain": "数据治理",
            "query": "比较差分隐私、联邦学习、安全多方计算和可信执行环境在数据协作中的威胁模型、效用损失与部署成本。",
            "expected_topics": ["差分隐私", "联邦学习", "安全多方计算", "可信执行环境", "威胁模型", "效用"],
            "ground_truth": {
                "差分隐私": "差分隐私通过限制单条记录对输出分布的影响提供可量化的隐私保证",
                "联邦学习": "联邦学习让参与方在不集中原始训练数据的情况下协同训练，但本身不消除更新泄露风险",
            },
            "reference_urls": ["https://www.nist.gov/blogs/cybersecurity-insights/differential-privacy-privacy-enhancing-technology"],
            "audit_status": "reviewed",
        },
        {
            "id": "bio_001",
            "domain": "生命科学",
            "query": "评估 AlphaFold 类蛋白质结构预测对结构生物学和药物发现的影响，区分静态结构预测、动力学、配体结合与实验验证。",
            "expected_topics": ["AlphaFold", "蛋白质结构", "结构生物学", "动力学", "配体", "实验验证"],
            "ground_truth": {
                "AlphaFold2": "AlphaFold2 使用深度学习从氨基酸序列预测蛋白质三维结构",
                "能力边界": "高质量静态结构预测不能替代对蛋白质动力学、相互作用和实验功能的验证",
            },
            "reference_urls": ["https://www.ebi.ac.uk/training/online/courses/alphafold/"],
            "audit_status": "reviewed",
        },
        {
            "id": "infra_001",
            "domain": "软件工程",
            "query": "分析软件交付性能中的部署频率、变更前置时间、变更失败率和恢复时间，讨论这些指标如何避免被团队机械优化。",
            "expected_topics": ["DORA", "部署频率", "前置时间", "变更失败率", "恢复时间", "指标治理"],
            "ground_truth": {
                "DORA 指标": "DORA 研究长期使用交付吞吐与稳定性指标评估软件交付表现",
                "指标边界": "指标应在系统和团队上下文中联合解释，单独优化一个指标可能产生错误激励",
            },
            "reference_urls": ["https://dora.dev/guides/dora-metrics-four-keys/"],
            "audit_status": "reviewed",
        },
    ]

    def __init__(self, data_path: str | None = None) -> None:
        """
        初始化评测集。

        Args:
            data_path: 外部 JSON 文件路径。若为 None 则使用内置题库。
        """
        if data_path and os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                questions = json.load(f)
        else:
            questions = self.DEFAULT_QUESTIONS + self.EXTENDED_QUESTIONS

        self.questions = [self._prepare_question(q) for q in questions]
        self.validate()

    @classmethod
    def _prepare_question(cls, question: dict[str, Any]) -> dict[str, Any]:
        prepared = copy.deepcopy(question)
        prepared.update(copy.deepcopy(cls.LEGACY_REVIEWED_METADATA.get(prepared.get("id", ""), {})))
        prepared.setdefault("as_of_date", cls.DEFAULT_AS_OF_DATE)
        prepared.setdefault("benchmark_version", cls.BENCHMARK_VERSION)
        prepared.setdefault("reference_urls", [])
        if "audit_status" not in prepared:
            prepared["audit_status"] = (
                "needs_rewrite" if prepared.get("id") in cls.LEGACY_REWRITE_IDS else "needs_source_review"
            )
        cutoff = prepared["as_of_date"]
        if "知识截止日期" not in prepared.get("query", ""):
            prepared["query"] = f"{prepared['query']}（知识截止日期：{cutoff}。）"
        return prepared

    def validate(self) -> None:
        ids = [q.get("id") for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("ResearchBench 存在重复 question id")
        required = {"id", "domain", "query", "expected_topics", "ground_truth", "as_of_date", "audit_status"}
        for question in self.questions:
            missing = required - question.keys()
            if missing:
                raise ValueError(f"题目 {question.get('id', '<unknown>')} 缺少字段: {sorted(missing)}")
            if not question["expected_topics"] or not question["ground_truth"]:
                raise ValueError(f"题目 {question['id']} 缺少 expected_topics 或 ground_truth")

    def get_questions(
        self,
        domain: str | None = None,
        n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        获取评测题目。

        Args:
            domain: 按领域过滤（科技/医疗/金融）。
            n: 返回前 n 道题。

        Returns:
            题目列表。
        """
        result = list(self.questions)
        if domain:
            result = [q for q in result if q.get("domain") == domain]
        if n is not None:
            result = result[:n]
        return result

    def get_demo_questions(self) -> list[dict[str, Any]]:
        by_id = {q["id"]: q for q in self.questions}
        missing = [qid for qid in self.DEMO_IDS if qid not in by_id]
        if missing:
            raise ValueError(f"Demo 题目不存在: {missing}")
        return [copy.deepcopy(by_id[qid]) for qid in self.DEMO_IDS]

    def get_audit_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for question in self.questions:
            status = question["audit_status"]
            summary[status] = summary.get(status, 0) + 1
        return summary

    def evaluate_report(
        self,
        report: str,
        question_id: str,
        metrics_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        对单篇研究报告进行评测。

        Args:
            report: 生成的研究报告文本。
            question_id: 对应题目的 ID。
            metrics_weights: 自定义指标权重。

        Returns:
            包含各维度得分和综合得分的字典。
        """
        from evaluation.metrics.rule_based import RuleBasedMetrics

        q = next((x for x in self.questions if x["id"] == question_id), None)
        if q is None:
            raise ValueError(f"未找到题目 ID: {question_id}")

        expected_topics = q.get("expected_topics", [])
        ground_truth = q.get("ground_truth", {})

        factual_str = RuleBasedMetrics.fact_accuracy(report, ground_truth)
        factual_sem = RuleBasedMetrics.semantic_fact_accuracy(report, ground_truth, threshold=0.65)
        hallucination = RuleBasedMetrics.hallucination_rate(report)
        citation = RuleBasedMetrics.citation_coverage(report)
        logic = RuleBasedMetrics.logical_consistency(report)
        comprehensive = RuleBasedMetrics.comprehensiveness(report, expected_topics)

        # bias 维度用 (1 - hallucination_rate) 作为代理
        bias_score = max(0.0, 1.0 - hallucination)

        metrics = {
            # canonical key consumed by composite_score; semantic matching is the
            # less gameable primary signal, while string matching stays diagnostic.
            "factual_accuracy": factual_sem,
            "factual_accuracy_str": factual_str,
            "factual_accuracy_sem": factual_sem,
            "logical_consistency": logic,
            "citation_coverage": citation,
            "bias": bias_score,
            "comprehensiveness": comprehensive,
        }

        composite = RuleBasedMetrics.composite_score(metrics, metrics_weights)

        return {
            "question_id": question_id,
            "domain": q.get("domain", ""),
            "metrics": metrics,
            "composite_score": composite,
            "hallucination_rate": hallucination,
        }

    def batch_evaluate(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        批量评估多篇报告。

        Args:
            results: 每条包含 {"question_id": ..., "report": ...} 的列表。

        Returns:
            聚合评测结果，含平均分、按领域统计。
        """
        all_scores = []
        by_domain: dict[str, list[float]] = {}

        for item in results:
            qid = item["question_id"]
            report = item["report"]
            eval_result = self.evaluate_report(report, qid)
            all_scores.append(eval_result)

            domain = eval_result["domain"]
            by_domain.setdefault(domain, []).append(eval_result["composite_score"])

        if not all_scores:
            return {"average_composite": 0.0, "by_domain": {}, "details": []}

        avg_composite = sum(s["composite_score"] for s in all_scores) / len(all_scores)
        domain_avg = {
            d: sum(scores) / len(scores) for d, scores in by_domain.items()
        }

        return {
            "average_composite": avg_composite,
            "by_domain": domain_avg,
            "details": all_scores,
        }


# =============================================================================
# 简单自测
# =============================================================================
if __name__ == "__main__":
    bench = ResearchBench()
    print(f"内置题目数: {len(bench.questions)}")

    sample_report = """
    GPT-4o 是 OpenAI 于 2024 年 5 月发布的原生多模态大模型[1]。
    Claude 3.5 Sonnet 由 Anthropic 于 2024 年 6 月发布，引入了 Artifacts 功能[2]。
    Gemini 1.5 Pro 支持超过 100 万 token 的上下文窗口[3]。
    Qwen2.5 是阿里巴巴的开源模型，支持 128K 上下文[4]。
    在中文推理方面，各模型表现接近；代码生成和长上下文处理各有优势。
    """

    result = bench.evaluate_report(sample_report, "tech_001")
    print("评测结果:", json.dumps(result, ensure_ascii=False, indent=2))
