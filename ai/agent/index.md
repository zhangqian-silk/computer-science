# Agent 系统

Agent 系统以模型交互为基础，把外部资料、工具操作和多步控制组织成能够完成任务的程序。学习主线从模型的一次输入输出开始，逐步进入多轮交互与信息管理。

---

## 当前文档

| 模块 | 主要内容 |
| --- | --- |
| [学习总览](notes/index.md) | 从单次生成到多 Agent 的各种执行范式及其判别标准 |
| [模型交互](notes/model/llm-api.md) | 单次调用、Prompt、图片、输出解析、多轮与工具回填、生成机制 |
| [Prompt and Context](notes/context/context-engineering.md) | 注意力预算、失效模式、上下文操作与长程策略 |
| [上下文压缩](notes/context/compaction.md) | 三层数据模型、处置阶梯、触发时机、摘要与续跑、缓存代价 |
| [Agent 记忆](notes/context/memory.md) | 三层分层与固化路径、类型学、召回打分、维护与有效性、风险与实现对照 |
| [memory / context / knowledge 边界](notes/context/memory-boundaries.md) | 从无状态推出三套机制、两条判据、易混对象辨析、归属判断与排障 |
| [知识如何进入一次请求](notes/knowledge-supply.md) | 五条供给路径、检索增强全链路、免索引路线与两者之争、四层副本一致性 |
| [工具调用机制](notes/tools/tool-calling-mechanics.md) | 七道执行关卡与失败去向、契约与标注边界、结果整形、三种通知形态、超时预算与重试、工具规模治理、协议的无状态化主线、权限与人在环 |
| [Agent 工具族](notes/tools/agent-tool-families.md) | 文件、命令、进程、网络、浏览器五族的核心约束与接口取舍，以及代码语义、编辑格式、任务状态与时间、子 Agent、对人的动作 |
| [Agent Skills](notes/tools/skills.md) | 渐进披露的三级加载、描述预算与发现链、与相邻机制的分工判据、打包分发与信任边界 |
| [Agent 扩展机制](notes/tools/extensibility.md) | 四类开洞位置、契约三要件与返回值语义、组合范式、能力集与打包分发、护栏与强制边界 |
| [Agent 运行时接入](notes/runtime/integration.md) | 五层归属与方向性、协议语义与审批的双向前提、六种接入面的取舍、隔离粒度与逃逸面、前缀缓存成本、暂停与恢复 |
| [Agent 运行时的工程问题](notes/runtime/engineering-problems.md) | 按判定依据分层的故障域、两层重试与空闲看门狗、隔离强度阶梯与平台差异、审批链路上的缝隙、恢复语义与落盘粒度、预算旋钮 |

题目化的复习材料统一收在[面试题库](notes/interview.md)，按主题分章，与各篇正文一一对应。来源与核验记录统一放在[参考资料](notes/references.md)。

---

## 历史内容

工具系统、流程控制、运行时、评测等主题的此前版本按代保留在[旧版归档](archive/catalog.md)。归档用于历史查阅，不作为当前默认学习路径，也不混入站内搜索结果。
