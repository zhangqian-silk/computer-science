# NLP 层：从历史主线进入现代方法地图

`nlp/` 目录关注的不是某一个单独模型，而是自然语言处理方法如何沿着历史主线不断演化。

这一层主要回答：

- NLP 的主干问题是如何一步步改变的；
- 统计方法、表示学习、序列模型、注意力与预训练之间是什么关系；
- 为什么现代方法会组织成今天这样的知识地图。

---

## 本目录与相邻目录的边界

- 本目录负责：历史主线、阶段转折、方法演化关系。
- 本目录不负责：单个机制或单个模型的完整数学展开。
- 若重点是具体机制，应进入 [mechanism](../mechanism/index.md)。
- 若重点是完整模型，应进入 [model](../model/index.md)。

---

## 当前文档

| 文档 | 角色 | 建议用途 |
| --- | --- | --- |
| [NLP 历史](./history.md) | 本目录主文档 | 建立从统计建模到预训练 Transformer 的主线理解 |

---

## 推荐阅读顺序

1. [NLP 历史](./history.md)
2. [Embedding](../representation/embedding.md)
3. [Attention](../mechanism/attention.md)
4. [Transformer](../model/transformer.md)
5. [BERT](../model/bert.md)
6. [GPT](../model/gpt.md)

如果希望先抓住完整历史脉络，再进入专题层，当前目录就是整个 `ai/` 子树里最合适的起点之一。
