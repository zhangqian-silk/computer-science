# 表示层：离散对象如何进入连续空间

`representation/` 目录讨论的核心问题是：模型如何表示词、子词、句子、模态片段，以及这些表示如何被学习、比较和使用。

这一层关注的是“表示本身”，而不是完整模型架构。更具体地说，它主要回答：

- 离散对象为什么需要 embedding；
- 静态表示与上下文化表示有什么差异；
- 词向量、句向量、多模态表示如何形成；
- 表示如何进入检索、匹配与对齐系统。

---

## 相关主题

- 若重点是 attention、LoRA、MoE 等模块机制，可进入 [mechanism](../mechanism/index.md)。
- 若重点是 Transformer、BERT、GPT 等完整模型，可进入 [model](../model/index.md)。

---

## 推荐阅读顺序

1. [Embedding](./embedding.md)
2. [Text Embedding Training](./text-embedding-training.md)
3. [word2vec](./word2vec.md)
4. [Multimodal Alignment](./multimodal-alignment.md)
5. [Retrieval Embedding](./retrieval-embedding.md)
6. [向量表示分析](../evaluation/embedding-geometry.md)

如果想先建立历史位置，也可以先读 [NLP 历史](../nlp/history.md) 再回到本目录。

---

## 文档角色划分

| 文档 | 角色 | 建议用途 |
| --- | --- | --- |
| [Embedding](./embedding.md) | 本目录总入口 | 先理解连续表示的统一定义 |
| [word2vec](./word2vec.md) | 经典方法专题 | 理解高效词向量训练 |
| [Text Embedding Training](./text-embedding-training.md) | 训练路线专题 | 理解静态词向量到上下文化表示的演化 |
| [Multimodal Alignment](./multimodal-alignment.md) | 跨模态专题 | 理解图文等不同模态如何共享表示空间 |
| [Retrieval Embedding](./retrieval-embedding.md) | 系统落地专题 | 理解向量索引、混合检索与重排流程 |

---

## 阅读提示

当前表示层最适合先抓住 3 个主问题：

- 表示是什么；
- 表示如何学出来；
- 学出来的表示如何被用于相似度、检索与迁移。

若只想先读一篇，应优先从 [Embedding](./embedding.md) 开始。
