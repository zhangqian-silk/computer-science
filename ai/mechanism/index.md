# 机制层：模型内部的通用计算模块

`mechanism/` 目录讨论的不是完整模型，而是可被多个模型复用的通用机制，例如 attention、self-attention、位置编码、LoRA、MoE 与状态空间模型。

这一层最核心的任务，是回答：

- 某个机制是什么；
- 它在数学上如何计算；
- 它解决了什么问题；
- 它通常在模型中的哪个位置发挥作用。

---

## 相关主题

- 若重点是 Transformer、BERT、GPT 等整机结构，可继续阅读 [model](../model/index.md)。
- 若重点是 embedding 等表示对象，可继续阅读 [representation](../representation/index.md)。

---

## 推荐阅读顺序

### 基础路径

1. [神经网络训练基础](./neural-network-basics.md)
2. [Attention](./attention.md)
3. [Self-Attention](./self-attention.md)
4. [Positional Encoding](./positional-encoding.md)

### 进阶路径

5. [Sparse Attention](./sparse-attention.md)
6. [LoRA](./lora.md)
7. [MoE](./moe.md)
8. [State Space Model](./state-space-model.md)

---

## 文档角色划分

| 文档 | 角色 | 建议用途 |
| --- | --- | --- |
| [神经网络训练基础](./neural-network-basics.md) | 数学与训练基础补充页 | 理解矩阵运算、激活、损失与反向传播基础 |
| [Attention](./attention.md) | 一般 attention 核心篇 | 理解 Q/K/V、打分、softmax、mask |
| [Self-Attention](./self-attention.md) | attention 在序列内部的特化 | 理解 Transformer 主干的核心交互方式 |
| [Positional Encoding](./positional-encoding.md) | 位置机制总览 | 理解顺序信息如何注入 |
| [RoPE](./rope.md) | 位置机制专题 | 理解旋转位置编码的几何与工程实现 |
| [Long Context Position](./long-context-position.md) | 长上下文位置专题 | 理解长度外推、插值、ALiBi 与扩窗路线 |
| [Sparse Attention](./sparse-attention.md) | 长序列高效机制专题 | 理解稀疏连接与复杂度权衡 |
| [LoRA](./lora.md) | 参数高效微调专题 | 理解低秩适配 |
| [MoE](./moe.md) | 条件计算专题 | 理解专家路由与容量扩展 |
| [State Space Model](./state-space-model.md) | 非 attention 序列主线专题 | 理解 SSM / Mamba 路线 |

---

## 阅读提示

阅读本目录时，建议始终区分两类问题：

- “这些机制依赖哪些矩阵与训练基础”；
- “机制本身怎么算”；
- “这个机制在模型里如何被使用”。

第一类可先回到 [neural-network-basics.md](./neural-network-basics.md)；第二类主要由本目录承担；第三类通常由 [model](../model/index.md) 目录继续展开。
