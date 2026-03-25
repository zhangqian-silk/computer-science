# 评估层：如何判断模型与表示是否有效

`evaluation/` 目录关注的不是“如何建模”，而是“如何判断一个模型或表示学得好不好”。

这一层主要回答：

- 语言模型的概率分配质量如何衡量；
- 词向量或句向量的几何结构如何分析；
- 检索系统的召回、排序与最终命中效果如何拆开衡量；
- 文本生成结果的质量、忠实性与任务完成度如何评估；
- 长上下文模型到底是真的“用到了长上下文”，还是只是在名义上扩窗；
- 某类方法的优势是否真的来自更好的泛化，而不只是训练集拟合。

---

## 相关主题

- 若重点是模型本身，可进入 [model](../model/index.md)。
- 若重点是表示机制，可进入 [representation](../representation/index.md)。

---

## 当前文档

| 文档 | 角色 | 建议用途 |
| --- | --- | --- |
| [语言模型评估](./language-model-evaluation.md) | 语言模型评估入口 | 理解困惑度、覆盖率与泛化 |
| [向量表示分析](./embedding-geometry.md) | 表示评估入口 | 理解邻域、聚类与线性关系 |
| [检索评估](./retrieval-evaluation.md) | 检索系统评估入口 | 理解 Recall、MRR、NDCG 与多阶段评估 |
| [生成评估](./generation-evaluation.md) | 文本生成评估入口 | 理解重合度指标、人工评估与忠实性边界 |
| [长上下文评估](./long-context-evaluation.md) | 长上下文能力评估入口 | 理解 Needle、Passkey 与位置敏感性测试 |

---

## 推荐阅读顺序

1. [语言模型评估](./language-model-evaluation.md)
2. [向量表示分析](./embedding-geometry.md)
3. [检索评估](./retrieval-evaluation.md)
4. [生成评估](./generation-evaluation.md)
5. [长上下文评估](./long-context-evaluation.md)

如果是配合模型学习，建议边读边回看：

- [N-Gram](../model/n-gram.md)
- [Embedding](../representation/embedding.md)
- [Retrieval Embedding](../representation/retrieval-embedding.md)
- [Seq2Seq](../model/seq2seq.md)
- [GPT](../model/gpt.md)
- [Transformer Extensions](../model/transformer-extensions.md)
- [Long Context Position](../mechanism/long-context-position.md)

---

## 本目录内部边界

目前 5 篇文档大致分工如下：

- `language-model-evaluation.md`：关注概率分配质量与泛化；
- `embedding-geometry.md`：关注表示空间的局部与全局结构；
- `retrieval-evaluation.md`：关注召回、排序与最终命中效果；
- `generation-evaluation.md`：关注生成文本的质量、忠实性与任务适配；
- `long-context-evaluation.md`：关注模型对远距离信息的真实利用能力。

如果后续继续扩展，本目录仍可逐步补入多模态评估、Agent 系统评估与在线实验评估专题。
