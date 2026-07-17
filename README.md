# 🚀 DeepResearch Agent

面向复杂深度研究任务的推理系统——将一个开放式研究请求转化为完整的工作流：拆题、发现来源、阅读原文、存证据、汇总、对抗修订，最终输出结构化 Markdown 研究报告。

本项目聚焦 **inference-time system design**，核心是运行时多智能体编排与检索增强，不涉及后训练或模型微调。

---

## 🎯 解决了什么问题

单轮 LLM 在复杂研究任务中通常会遇到：

- 问题过大，一次对话难以组织清楚
- 多源证据混杂，引用链不清晰
- 多步推理的中间结果无法复用
- 初版答案常有遗漏、幻觉或引用不准确

本项目的思路不是把 prompt 写得更长，而是将任务拆成明确的阶段——先规划、再执行、存证据、合成报告、攻防修订——每个阶段有专门的模块负责。

---

## 🔄 整体流程

```
用户问题
  → Planner 拆解为 DAG 子任务图
  → Orchestrator 按依赖关系并发调度
  → Worker Agent：发现候选来源 → 自动打开原文验证
  → Memory Store 保存并复用中间证据
  → GraphRAG + Reranker 二阶段检索优化
  → Summarizer 合成研究报告初稿
  → Red-Blue Adversarial Loop 多轮攻防修订
  → 输出带行内引用、研究标题和参考来源的 Markdown 报告
```

### 外部证据的两阶段检索

这里的“两阶段”指外部研究证据，不要与 Memory Store 的 GraphRAG/Reranker 重排混淆。

1. **发现阶段**：每个子任务最多执行 2 次 `web_search` 或 `arxiv_reader`，结果经相关性、去重、域名质量门禁筛选。
2. **原文验证阶段**：系统从已准入候选中按来源质量和相关性选择最多 2 条，自动使用 `browser` 打开原文；成功读取的文本以 `full_text` 证据写入轨迹和证据账本。
3. **报告阶段**：摘要器优先使用 `full_text` 证据；无法打开的候选保留为 `discovery` 摘要，避免少量网页或 PDF 读取失败导致报告整体缺失。

搜索摘要用于发现，不应被当作原文事实的等价替代。最终报告要求实质性段落包含 `[S1]` 形式的行内引用，参考来源只输出正文实际使用的来源。

### 报告交付

- 报告标题取自正文的首个实质性 Markdown 标题，而不是直接复用用户问题。
- 正文目标通常不少于 3000 个中文字，包含问题界定、研究进展、技术路线比较、实验计划、风险与下一步。
- 标题、正文、引用和参考来源会在最终渲染阶段再次校验；保存文件名也使用报告标题。

---

## 🏗️ 系统架构

### M1 Orchestrator — 编排引擎

自研 9 状态状态机驱动的异步编排器，基于 `asyncio + Semaphore` 实现 DAG 拓扑并发执行。支持动态增量重规划与三级降级策略（单任务超时 → 批量失败重规划 → 全局超时强制合成）。

### M2 Planner — 自适应规划器

将研究问题分解为 3-8 个子任务的 DAG，支持 JSON Schema 约束解码与多层 JSON 解析 fallback。运行中可根据失败任务触发增量重规划。

### M3 Compressor — 上下文压缩器

三级渐进式压缩：L1 Embedding 粗过滤 → L2 TextRank 关键句提取 → L3 LLM 层级摘要。基于 token 预算使用率自动触发，避免长链路任务的信息损失。

### M4 Memory Store — 共享记忆与检索增强

SQLite + numpy 向量索引实现跨 Agent 共享记忆，写前自动去重（cosine > 0.92）与矛盾检测（启发式反义词 + 语义对立）。检索链路为三段式：

1. **向量召回** — Embedding cosine similarity top-K
2. **GraphRAG 扩展** — 知识图谱 1-hop 邻居遍历，双通道加权融合
3. **Reranker 重排** — 融合检索分、语义相似度、词面重合、topic 匹配四维特征二次排序，可选 cross-encoder 精排

这条链路用于**已保存记忆的召回**；外部网页和论文的“发现 → 原文验证”流程由 `ResearcherAgent` 和工具层负责。

### M5 Adversarial Loop — 对抗降噪

Red Agent 从事实性 / 幻觉 / 逻辑一致性 / 引用质量 / 覆盖完整性五个维度攻击报告，Blue Agent 执行 IN_PLACE / SUPPLEMENTARY / REMOVAL 三种修复操作。内置评分收敛判据与震荡检测，防止无限循环。

---

## 🗂️ 代码结构

```
deepresearch-agent/
├── configs/              YAML 配置中心
│   ├── default.yaml      全局配置（模型、编排、压缩、记忆、对抗）
│   └── agents/           Agent 行为配置
├── src/
│   ├── core/             系统装配层（配置加载、模块初始化、研究流程入口）
│   ├── planner/          任务规划（DAG 分解、增量重规划、预算追踪）
│   ├── orchestrator/     编排调度（状态机、Agent 池、并发执行）
│   ├── agents/           Agent 实现（Researcher、Summarizer）
│   ├── tools/            工具层（搜索、网页、论文、计算、文件、代码、笔记）
│   ├── memory/           记忆与检索（SQLite 持久化、向量索引、GraphRAG、Reranker）
│   ├── compressor/       长上下文压缩（三级渐进式）
│   ├── adversarial/      红蓝对抗降噪（Red Agent、Blue Agent、Loop）
│   ├── models/           多后端 LLM 路由（DeepSeek / MiMo / vLLM / OpenAI）
│   └── utils/            公共工具（环境配置、追踪、结构化输出）
├── evaluation/           评测体系
│   ├── benchmarks/       ResearchBench + HotpotQA
│   └── metrics/          规则指标 + 统计显著性 + LLM-as-Judge
├── scripts/              可执行脚本（单条、REPL、评测、消融、对比）
└── tests/                测试
```

---

## 📖 阅读路线

建议按以下顺序理解整个系统：

1. `configs/default.yaml` — 了解各模块有哪些可配置项
2. `src/core/runner.py` — 系统装配入口，看模块如何初始化和串联
3. `src/orchestrator/schemas.py` — 核心数据结构（状态机状态、SubTask、RunConfig）
4. `src/orchestrator/orchestrator.py` — 编排器主循环
5. `src/planner/planner.py` — 如何将 query 拆成 DAG
6. `src/agents/researcher.py` — Worker Agent 如何做工具调用
7. `src/tools/` — 各工具的实现
8. `src/memory/memory_store.py` — 共享记忆核心接口
9. `src/memory/graph_retriever.py` — GraphRAG 图谱扩展检索
10. `src/memory/reranker.py` — 二阶段重排序
11. `src/adversarial/loop.py` — 红蓝对抗循环
12. `evaluation/` 和 `scripts/` — 评测与实验入口

---

## ⚙️ 关键配置

主配置文件：`configs/default.yaml`

最值得关注的段落：

| 配置段 | 控制内容 |
|--------|---------|
| `model` | 后端选择、模块级采样参数、后端分工映射 |
| `orchestrator` | 并发度、全局超时、重规划上限 |
| `compressor` | 三级压缩阈值、上下文长度限制 |
| `memory` | 数据库路径、去重/冲突阈值、GraphRAG、Reranker |
| `adversarial` | 红蓝对抗开关、最大轮数、收敛阈值 |
| `tools` | 搜索、论文和网页工具行为、mock 模式；每个子任务固定最多 2 次发现 + 2 次原文验证 |

---

## 🛠️ 安装

**环境要求：**

- Python 3.10+
- 至少一个可用的 LLM API（DeepSeek / OpenAI / vLLM / MiMo）
- （可选）`vllm` 用于本地模型推理服务

```bash
git clone https://github.com/qiqihezh/deepresearch-agent.git
cd deepresearch-agent

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

**配置环境变量：**

```bash
cp .env.template .env
# 编辑 .env，至少填写一个 LLM 后端的 API Key + 一个搜索后端的 API Key
```

默认配置将 `solver`、`planner`、`summarizer` 路由到 DeepSeek，将 `judge`、`red_agent`、`blue_agent`、`compressor` 路由到 MiMo。若未配置 MiMo Key，请在 `configs/default.yaml` 中将这些模块改为已配置后端，或关闭对应模块；否则运行日志会出现认证失败，相关能力会降级。

**验证环境：**

```bash
python tests/validate_env.py
```

---

## ▶️ 运行方式

### 单条研究

```bash
python scripts/run_single.py --query "比较 2024-2025 年开源推理模型的发展路线"
```

### 交互式 REPL

```bash
python scripts/run_repl.py
```

同一 session 内的记忆跨问题共享，适合连续追问。

### 标准评测

```bash
# ResearchBench（自建 35 题）
python scripts/run_eval.py --benchmark research_bench --num_questions 20

# HotpotQA 深度研究变体
python scripts/run_eval.py --benchmark hotpotqa --use_mock
```

### 消融实验

```bash
# 模块消融（量化每个模块的边际贡献）
python scripts/run_ablation.py --mode module --questions 12

# 对抗轮数消融（0 → 1 → 2 → 3 轮）
python scripts/run_ablation.py --mode rounds --questions 12
```

### Agent vs 单轮 LLM

```bash
python scripts/run_benchmark.py --queries "你的研究问题1" "你的研究问题2"
```

### LLM-as-Judge 深度评分

```bash
python scripts/run_judge.py --report_file outputs/reports/report_xxx.md --query "原始研究问题"
```

### RAG vs GraphRAG 对比

```bash
# 检索级对比（无需 LLM，秒级）
python scripts/run_rag_vs_graphrag.py --mode retrieval

# 端到端对比（完整 Agent 流程）
python scripts/run_rag_vs_graphrag.py --mode e2e --num_questions 5
```

### 8 维度定量评测

```bash
# 只跑不需要 LLM 的维度（秒级）
python scripts/run_quantitative_bench.py --dim D3,D4,D5

# 全部 8 个维度
python scripts/run_quantitative_bench.py --all

# 快速验证模式
python scripts/run_quantitative_bench.py --all --quick
```

### 一键批量实验

```bash
python scripts/run_all_experiments.py
```

---

## 🧭 适用与不适用

**适合的问题：**

- 趋势研究与技术路线比较
- 多来源资料综合分析
- 需要明确引用链的开放题
- 学术、产业、政策类长答案任务

**不适合的问题：**

- 一跳事实问答（如"某人的生日是哪天"）
- 强实时查询
- 必须外部严格验真的高风险结论

---

## 📌 项目定位

本仓库不打算做：

- 后训练框架或 RLHF / GRPO 训练管线
- 通用工作流平台
- 多模态训练基础设施

它的目标始终是：**复杂研究任务的运行时推理系统**——通过更好的编排、检索、记忆和对抗质量控制，让 LLM 在深度研究任务上表现得更好。

---

## 🙏 参考说明

本项目在系统结构设计、部分模块拆分方式以及若干实现细节上，参考了开源项目 [qiqihezh/deepresearch-agent](https://github.com/qiqihezh/deepresearch-agent) 的代码。

如果你希望了解该方向的原始实现背景，建议同时阅读该项目仓库。

---

## ⚖️ License

MIT
