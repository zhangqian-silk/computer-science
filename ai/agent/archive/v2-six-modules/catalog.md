---
title: "Agent 六模块版归档目录"
search: false
---

# Agent 六模块版归档

这里保留「按六模块组织学习文档」阶段的工作区快照，正文、代码、数据和图片均不改写。页面上的归档提示与链接适配由站点渲染层提供，不写回历史文件。

本代归档收录 2026-09-13 结构调整前的 32 篇文档，另附当时的参考文献快照。当前保留在正文的[学习总览](../../notes/index.md)、[模型交互](../../notes/model/llm-api.md)与[Prompt and Context](../../notes/context/context-engineering.md)不在此处重复。

[返回归档总目录](../catalog.md)。

---

## 上下文与记忆

- [提示工程](notes/context/prompt-engineering.md)
- [会话状态与长期记忆](notes/context/conversation-memory.md)
- [RAG 与代码检索](notes/context/rag-retrieval.md)
- [上下文压缩](notes/context/compaction.md)

---

## 模型交互的拆分页

以下三页在本代已并入模型交互主文，仅保留迁移入口。

- [KV Cache 与请求缓存](notes/model/kv-cache.md)
- [模型输出与结构化结果](notes/model/outputs.md)
- [流式响应与模型适配](notes/model/streaming.md)

---

## 工具与能力扩展

- [工具契约、执行与结果](notes/tools/tool-calling.md)
- [文件、进程、浏览器与服务](notes/tools/execution-environments.md)
- [MCP 与外部连接器](notes/tools/mcp-connectors.md)
- [Skills 与操作知识加载](notes/tools/skills.md)
- [Hooks、Plugins 与扩展管理](notes/tools/hooks-plugins.md)

---

## 流程控制与 Agent 方法

- [Workflow 与流程控制](notes/control/workflows.md)
- [ReAct 与 Agent Loop](notes/control/react-loop.md)
- [规划与计划修订](notes/control/reasoning-planning.md)
- [Subagent 与多 Agent 协作](notes/control/subagents-multi-agent.md)
- [Dynamic Workflows 与代码编排](notes/control/dynamic-workflows.md)

---

## 运行时与工程化

- [运行时组成与任务生命周期](notes/runtime/harness-architecture.md)
- [人工交互与执行控制](notes/runtime/human-control.md)
- [任务恢复与副作用](notes/runtime/persistence-recovery.md)
- [任务调度与 Agent 接入](notes/runtime/scheduling-protocols.md)
- [安全与隔离](notes/runtime/security.md)

以下四页在本代已迁至 `quality/`，仅保留迁移入口。

- [评测：新版阅读入口](notes/runtime/evaluation.md)
- [可观测性：新版阅读入口](notes/runtime/observability.md)
- [性能与成本：新版阅读入口](notes/runtime/performance-cost.md)
- [版本与发布：新版阅读入口](notes/runtime/versioning-release.md)

---

## 评测、可观测性与优化

- [评测：对象、标准与实验设计](notes/quality/evaluation.md)
- [可观测性：事件、Trace 与故障诊断](notes/quality/observability.md)
- [性能与成本：测量、权衡与优化](notes/quality/performance-cost.md)
- [版本、回归与发布](notes/quality/versioning-release.md)

---

## 资料与实验

- [实现案例阅读索引](notes/implementation-index.md)
- [学习练习与实验](notes/labs.md)
- [参考资料与来源记录](notes/references.md)
