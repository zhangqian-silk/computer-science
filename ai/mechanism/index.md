# 机制层：模型内部的通用计算模块

`mechanism/` 目录讨论的不是完整模型，而是可被多个模型复用的通用机制，例如 attention、self-attention、位置编码、LoRA、MoE 与状态空间模型。

这一层最核心的任务，是回答：

- 某个机制是什么；
- 它在数学上如何计算；
- 它解决了什么问题；
- 它通常在模型中的哪个位置发挥作用。

---

## 本目录与相邻目录的边界

- 本目录负责：通用模块、计算机制、结构部件、参数化技巧。
- 本目录不负责：完整模型的总体任务接口、训练闭环和家族演化。
- 若重点是 Transformer、BERT、GPT 等整机结构，应进入 [model](../model/index.md)。
- 若重点是 embedding 等表示对象，应进入 [representation](../representation/index.md)。

---

## 推荐阅读顺序

### 基础路径

1. [Attention](./attention.md)
2. [Self-Attention](./self-attention.md)
3. [Positional Encoding](./positional-encoding.md)

### 进阶路径

4. [Sparse Attention](./sparse-attention.md)
5. [LoRA](./lora.md)
6. [MoE](./moe.md)
7. [State Space Model](./state-space-model.md)

---

## 文档角色划分

| 文档 | 角色 | 建议用途 |
| --- | --- | --- |
| [Attention](./attention.md) | 一般 attention 主文档 | 理解 Q/K/V、打分、softmax、mask |
| [Self-Attention](./self-attention.md) | attention 在序列内部的特化 | 理解 Transformer 主干的核心交互方式 |
| [Positional Encoding](./positional-encoding.md) | 位置机制主文档 | 理解顺序信息如何注入 |
| [RoPE](./rope.md) | 位置机制专题 | 理解旋转位置编码的几何与工程实现 |
| [Long Context Position](./long-context-position.md) | 长上下文位置专题 | 理解长度外推、插值、ALiBi 与扩窗路线 |
| [Sparse Attention](./sparse-attention.md) | 长序列高效机制专题 | 理解稀疏连接与复杂度权衡 |
| [LoRA](./lora.md) | 参数高效微调专题 | 理解低秩适配 |
| [MoE](./moe.md) | 条件计算专题 | 理解专家路由与容量扩展 |
| [State Space Model](./state-space-model.md) | 非 attention 序列主线专题 | 理解 SSM / Mamba 路线 |

---

## 阅读提示

阅读本目录时，建议始终区分两类问题：

- “机制本身怎么算”；
- “这个机制在模型里如何被使用”。

前者主要由本目录承担；后者通常由 [model](../model/index.md) 目录继续展开。
