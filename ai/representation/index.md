# 表示：从离散标识到可用向量

表示层研究的不是某个完整模型，而是对象进入计算系统后的数值接口。一个向量是否「好」，取决于训练信号、比较方式和下游任务，不能只凭降维图或少数类比例子判断。

---

## 主题地图

| 页面 | 核心内容 | 前置知识 |
| --- | --- | --- |
| [Embedding](./embedding.md) | 定义查表表示、上下文化表示与向量几何 | 线性代数基础 |
| [word2vec](./word2vec.md) | 推导 CBOW、Skip-gram 与负采样 | Embedding |
| [文本嵌入](./text-embedding.md) | 解释句子/文档向量的对比学习、池化与负样本 | Embedding |
| [多模态对齐](./multimodal-alignment.md) | 解释不同模态如何进入共享空间 | Embedding、对比学习 |
| [向量检索](./vector-retrieval.md) | 解释向量如何进入 ANN、混合检索与重排系统 | Embedding |

表示质量的诊断方法放在[向量表示分析](../evaluation/embedding-geometry.md)，召回与排序指标放在[检索评估](../evaluation/retrieval-evaluation.md)。这样可以把「训练表示」与「验证系统」分开。

| 表示类型 | 主要训练信号 | 使用时的读出方式 |
| --- | --- | --- |
| 查表 Embedding | 下游任务梯度 | 按离散 ID 取参数行 |
| word2vec | 局部共现预测或负采样 | 词向量近邻、聚类或下游特征 |
| 文本嵌入 | 句子/文档正负对比 | 双编码后做向量检索 |
| 多模态对齐 | 跨模态配对或生成目标 | 跨模态相似度或条件 token |

向量的维度相同不代表语义空间兼容。只有编码器版本、训练目标、归一化和相似度约定一致时，两组向量才可以直接比较。

---

## 建议路线

- 理解静态表示：Embedding → word2vec → 向量表示分析；
- 构建语义检索：Embedding → 文本嵌入 → 向量检索 → 检索评估；
- 理解多模态模型：Embedding → 多模态对齐 → [Transformer](../model/transformer.md)。
