# Agent 层：从模型能力到系统能力

`agent/` 目录讨论的是：当模型不再只回答一个输入，而是需要规划、调用工具、访问外部记忆、与工作流协作时，系统应该如何组织。

这一层的重点，不再是单个语言模型本身，而是模型如何与外部组件组成更完整的执行系统。

---

## 本目录与相邻目录的边界

- 本目录负责：Agent 架构、工具调用、RAG、规划、记忆、工作流。
- 本目录不负责：基础模型内部数学机制。
- 若重点是 GPT、Transformer 等模型本体，应进入 [model](../model/index.md)。
- 若重点是 embedding、检索表示，应进入 [representation](../representation/index.md)。

---

## 当前文档

| 文档 | 角色 | 建议用途 |
| --- | --- | --- |
| [Agent Arch](./arch.md) | 本目录入口 | 理解 Agent 系统的模块与编排方式 |

---

## 推荐阅读顺序

1. [GPT](../model/gpt.md)
2. [Embedding](../representation/embedding.md)
3. [Agent Arch](./arch.md)

如果重点关注工具与系统编排，也可以先读：

1. [Transformer](../model/transformer.md)
2. [GPT](../model/gpt.md)
3. [Agent Arch](./arch.md)

---

## 阅读提示

理解 Agent 系统时，建议区分 3 层能力：

- 模型能力：模型本身会什么；
- 机制能力：模型如何接入记忆、检索、工具；
- 系统能力：多个组件如何被编排成稳定流程。

当前本目录先从架构总览切入，后续可以继续扩展为更细的 Agent 专题层。
