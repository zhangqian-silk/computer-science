# 表示：从离散标识到可用向量

表示层沿「输入怎样切分 → 如何变成向量 → 向量怎样被训练 → 如何比较和检索」组织。Tokenization 的 ID、Embedding 的参数行、上下文化 token 和文档级向量是不同对象；先确定读出用途，再讨论表示质量。

---

## 主题地图

| 页面 | 核心内容 | 前置知识 |
| --- | --- | --- |
| [Tokenization](./tokenization.md) | 文本怎样变成稳定的 token ID 接口 | 字符串与 Unicode 基础 |
| [Embedding](./embedding.md) | 定义查表表示、上下文化表示与向量几何 | 线性代数基础 |
| [word2vec](./word2vec.md) | 推导 CBOW、Skip-gram 与负采样 | Embedding |
| [矩阵分解视角](./embedding-matrix-factorization.md) | 推导 SGNS 理想内积，区分 SPPMI 与 GloVe 的目标 | word2vec |
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

相同维度不代表空间兼容。两侧需通过联合对齐、兼容训练或验证建立可比较接口，并记录版本、归一化和度量；双塔编码器可以不同，版本相同也不能单独证明语义兼容。

---

## 建议路线

贯穿例子可以用「猫追狗」与「狗追猫」：分词保留哪些单位，查表是否看见上下文，池化是否丢失角色，训练正例是否要求区分事件。读完本分区应能解释错误来自表示粒度、训练信号、相似度还是近似索引，而不是统称为「向量不好」。

- 理解静态表示：Embedding → word2vec → 矩阵分解视角 → 向量表示分析；
- 构建语义检索：Embedding → 文本嵌入 → 向量检索 → 检索评估；
- 理解多模态模型：Embedding → 多模态对齐 → [Transformer](../model/transformer.md)。
