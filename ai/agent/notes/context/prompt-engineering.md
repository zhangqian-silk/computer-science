---
search: false
---

# 提示工程：新版阅读入口

提示词工程解决「一次指令怎么表达」，是上下文工程的一个子集；上下文工程决定「每一步推理该让模型看见哪组 token」。写好 prompt 仍是基本功，但长程任务的上限由信息环境的治理决定——注意力预算、失效模式与长程策略见[上下文管理主文](context-engineering.md)。

基础提示设计已与模型调用连续组织，不再作为上下文模块的独立入门：

- [任务、资料与输出要求](../model/llm-api.md#input-design)
- [Zero-shot 与 few-shot](../model/llm-api.md#few-shot)
- [System prompt 与多轮消息](../model/llm-api.md#system-prompt)
- [提示改动的对照实验](../quality/evaluation.md#experiments)

此前版本原样保留在[旧版提示工程](../../archive/notes/context/prompt-engineering.md)。
