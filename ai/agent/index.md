# Agent 系统

Agent 系统以模型交互为基础，把外部资料、工具操作和多步控制组织成能够完成任务的程序。学习主线从模型的一次输入输出开始，逐步进入多轮交互、信息管理、能力执行、流程方法、运行时和评价。

---

## 新版文档

| 模块 | 主要内容 |
| --- | --- |
| [模型交互](notes/model/llm-api.md) | 单次调用、Prompt、图片、输出解析、多轮与工具回填、生成机制 |
| [上下文管理与优化](notes/context/context-engineering.md) | 资料组织、对话存取、选择、RAG、记忆与压缩 |
| [工具系统与能力扩展](notes/tools/tool-calling.md) | 契约、执行、结果、环境、MCP、Skills 与扩展 |
| [流程控制与 Agent 方法](notes/control/workflows.md) | Workflow、Agent Loop、规划、多 Agent 与动态编排 |
| [Agent 运行时与工程化](notes/runtime/harness-architecture.md) | 程序组成、生命周期、人工控制、恢复、调度与安全 |
| [评测、可观测性与优化](notes/quality/evaluation.md) | 标准、实验、Trace、性能成本、回归与发布 |

[学习总览与完整阅读目录](notes/index.md)给出前置关系；[练习与实验](notes/labs.md)提供消息构造、过程推演和数据比较。

---

## 历史内容

重写前的文件原样保留在 [旧版归档](archive/catalog.md)。归档用于历史查阅，不作为新版默认学习路径，也不混入站内搜索结果。
