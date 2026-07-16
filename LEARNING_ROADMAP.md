# DeepResearch Agent — 技术栈学习清单

> 按层次递进，每层标注了在项目中的具体位置和推荐学习资源。

---

## 第一层：Python 基础（必须先行）

| 序号 | 知识点 | 在项目中的位置 | 学习资源 |
|------|--------|---------------|---------|
| 1.1 | `async/await` 异步编程 | 整个编排引擎都是 asyncio 驱动的 | [Python asyncio 官方文档](https://docs.python.org/3/library/asyncio.html) |
| 1.2 | `asyncio.gather` / `Semaphore` | `src/orchestrator/orchestrator.py` 并发调度 | 搜 "Python asyncio gather semaphore pattern" |
| 1.3 | `dataclass` 数据类 | `src/orchestrator/schemas.py` 所有数据结构 | [dataclasses 官方文档](https://docs.python.org/3/library/dataclasses.html) |
| 1.4 | `from __future__ import annotations` | 每个文件第一行 | 搜 "PEP 563 postponed evaluation" |
| 1.5 | Type Hints（`list[dict]`, `Optional`, `Any`, `Callable`） | 全项目类型标注 | [mypy cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) |
| 1.6 | `logging` 日志模块 | `src/core/runner.py` `setup_logging()` | [Python logging 教程](https://docs.python.org/3/howto/logging.html) |

---

## 第二层：LLM 基础概念

| 序号 | 知识点 | 在项目中的位置 | 学习资源 |
|------|--------|---------------|---------|
| 2.1 | LLM 工作原理（Token / Context Window / Temperature） | `src/models/vllm_policy.py` 中的 `max_tokens`、`temperature`、`top_p` | [Andrej Karpathy - Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (1h) |
| 2.2 | Prompt Engineering（System Prompt / User Prompt / Few-shot） | `src/planner/planner.py` 的 `INITIAL_PLAN_PROMPT`（第34-76行） | [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) |
| 2.3 | Function Calling / Tool Use | `src/agents/researcher.py` 工具系统（第272-301行） | [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling) |
| 2.4 | OpenAI API 消息格式（messages 结构 + role/content/tool_calls） | `src/models/vllm_policy.py` `__call__` 方法（第140-262行） | [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) |
| 2.5 | Structured Output / JSON Mode | 新增 `src/utils/structured_output.py` | [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) |

---

## 第三层：Agent 核心架构

| 序号 | 知识点 | 在项目中的位置 | 学习资源 |
|------|--------|---------------|---------|
| 3.1 | Agent 是什么（感知 → 规划 → 执行 → 反思 循环） | `src/orchestrator/orchestrator.py` 9 状态状态机 | [Lilian Weng - LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) |
| 3.2 | Multi-Agent 协作模式 | M5 Red-Blue 对抗（`src/adversarial/`）+ Orchestrator/AgentPool 协作（`src/orchestrator/`） | [AutoGen 论文](https://arxiv.org/abs/2308.08155) |
| 3.3 | DAG 有向无环图 + 拓扑排序 | `src/planner/dag.py` + `src/planner/planner.py` | 搜 "Kahn's algorithm topological sort" |
| 3.4 | 状态机（State Machine）模式 | `src/orchestrator/schemas.py` OrchestratorState（第29-44行） | 搜 "finite state machine Python pattern" |
| 3.5 | ReAct Agent 范式（Reasoning + Acting） | `src/agents/researcher.py` 多轮工具调用循环（第55-270行） | [ReAct 论文](https://arxiv.org/abs/2210.03629) |
| 3.6 | Agent 对象池模式 | `src/orchestrator/agent_pool.py` | 搜 "object pool pattern Python" |

---

## 第四层：RAG 与记忆系统

| 序号 | 知识点 | 在项目中的位置 | 学习资源 |
|------|--------|---------------|---------|
| 4.1 | Embedding（文本向量化）原理 | `src/memory/embedder.py` — 使用 `all-MiniLM-L6-v2` | [Sentence Transformers 文档](https://www.sbert.net/) |
| 4.2 | Cosine Similarity（余弦相似度） | `src/memory/memory_store.py` `_cosine_similarity()`（第53-59行） | 搜 "cosine similarity explained visually" |
| 4.3 | RAG 基础（Retrieval-Augmented Generation） | M4 Memory Store 整个模块（`src/memory/`） | [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/) |
| 4.4 | GraphRAG（知识图谱增强检索） | 新增 `src/memory/graph_extractor.py` + `src/memory/graph_retriever.py` | [Microsoft GraphRAG 论文](https://arxiv.org/abs/2404.16130) |
| 4.5 | TextRank 关键句提取 | `src/compressor/extractive.py` | [TextRank 原始论文](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) |
| 4.6 | 上下文压缩策略（L1→L2→L3 三级） | `src/compressor/compressor.py` | 搜 "LLM context compression techniques" |

---

## 第五层：基础设施与工程

| 序号 | 知识点 | 在项目中的位置 | 学习资源 |
|------|--------|---------------|---------|
| 5.1 | SQLite 数据库（CRUD + 索引 + 外键） | `src/memory/long_term.py` entries/conflicts/entities/relations 四表 | [SQLite Tutorial](https://www.sqlitetutorial.net/) |
| 5.2 | NumPy 向量操作（dot / norm / argmax） | `src/memory/memory_store.py` 向量索引 | [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) |
| 5.3 | 线程安全（`threading.RLock`） | `src/memory/memory_store.py` + `src/memory/long_term.py` | 搜 "Python threading Lock vs RLock when to use" |
| 5.4 | YAML 配置文件管理 | `configs/default.yaml` + 各子模块配置 | [PyYAML 文档](https://pyyaml.org/) |
| 5.5 | JSON Schema（结构化输出约束） | 新增 `src/utils/structured_output.py` 预定义 schema | [JSON Schema 规范](https://json-schema.org/learn/getting-started-step-by-step) |
| 5.6 | 正则表达式（`re.search` / `re.compile` / `re.DOTALL`） | 全项目大量使用（JSON 解析、置信度提取等） | [RegexOne 交互式教程](https://regexone.com/) |
| 5.7 | Pydantic 或 dataclass 数据验证 | `src/orchestrator/schemas.py` 所有数据结构 | [Python dataclasses 最佳实践](https://realpython.com/python-data-classes/) |

---

## 第六层：评测体系与统计学

| 序号 | 知识点 | 在项目中的位置 | 学习资源 |
|------|--------|---------------|---------|
| 6.1 | Bootstrap 重采样置信区间 | `evaluation/metrics/stats.py` | 搜 "bootstrap confidence interval explained Python" |
| 6.2 | Cohen's d 效应量（Effect Size） | `evaluation/metrics/stats.py` | 搜 "Cohen's d effect size interpretation" |
| 6.3 | LLM-as-Judge 评估范式 | `src/core/judge.py` 统一评分接口 | [Judging LLM-as-a-Judge with MT-Bench](https://arxiv.org/abs/2306.05685) |
| 6.4 | HotpotQA 多跳问答评测 | `evaluation/benchmarks/hotpotqa.py` | [HotpotQA 官网](https://hotpotqa.github.io/) |
| 6.5 | 消融实验（Ablation Study）设计 | `src/core/ablation.py` + `scripts/run_ablation.py` | 搜 "ablation study machine learning explained" |
| 6.6 | 成对 t 检验（Paired t-test） | `evaluation/metrics/stats.py` | 搜 "paired t-test explained" |

---

## 推荐学习顺序（8 周计划）

```
第1周 │ Python async/await + dataclass + type hints + logging
      │ → 能读懂 src/orchestrator/ 和 src/core/ 的所有代码
      │
第2周 │ LLM 基本原理 + OpenAI API 格式 + Function Calling
      │ → 能读懂 src/models/vllm_policy.py 和 src/agents/researcher.py
      │
第3周 │ Prompt Engineering + ReAct Agent 范式 + 状态机/DAG
      │ → 能读懂 src/planner/ 和 src/orchestrator/orchestrator.py 主循环
      │
第4周 │ Embedding + Cosine Similarity + RAG 基础 + SQLite
      │ → 能读懂 src/memory/ 和 src/compressor/ 全部代码
      │
第5周 │ Multi-Agent 协作 + Red-Blue 对抗 + 对象池模式
      │ → 能读懂 src/adversarial/ 全部代码
      │
第6周 │ GraphRAG + 评测统计学 + 消融实验设计
      │ → 能理解新增的 graph_extractor/retriever 和 evaluation/ 全部代码
```

---

## 速成路径（时间有限）

如果只有 **2 周时间**，按以下优先级学习：

| 优先级 | 知识点 | 覆盖项目代码比例 |
|--------|--------|-----------------|
| 🔴 P0 | Python async/await + asyncio.gather | ~30% |
| 🔴 P0 | LLM 基本原理 + OpenAI API | ~25% |
| 🔴 P0 | Agent/ReAct 范式 | ~15% |
| 🟡 P1 | Embedding + Cosine Similarity + RAG | ~15% |
| 🟡 P1 | Prompt Engineering | ~10% |
| 🟢 P2 | GraphRAG + 统计评测 | ~5% |
