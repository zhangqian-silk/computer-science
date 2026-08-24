# 模型：上下文建模方法地图

模型层按「条件信息如何到达当前预测」组织。固定窗口、递归状态、全局注意力与状态空间对应四种不同的信息传播约束，与新旧无关。

| 路线 | 条件信息的载体 | 代表页面 | 主要约束 |
| --- | --- | --- | --- |
| 固定窗口 | 最近 $n-1$ 个 token | [N-gram](./n-gram.md)、[NPLM](./nplm.md) | 窗口有限，组合稀疏 |
| 递归状态 | 逐步更新的隐藏状态 | [RNN](./rnn.md)、[LSTM](./lstm.md) | 串行路径长 |
| 条件生成 | 编码器表示与解码前缀 | [Seq2Seq](./seq2seq.md) | 固定摘要或对齐质量 |
| 全局交互 | token 间的注意力连接 | [Transformer](./transformer.md)、[BERT](./bert.md)、[GPT](./gpt.md) | 长度带来计算与缓存压力 |
| 状态空间 | 可并行训练的线性状态递推 | [状态空间模型](./state-space-model.md) | 选择机制与硬件实现决定效果 |

---

## 训练与推理总览

| 模型 | 训练信号怎样产生 | 推理时依赖什么 |
| --- | --- | --- |
| [N-gram](./n-gram.md) | 对滑窗事件计数并估计平滑参数 | 最近 $n-1$ 个 token 与多阶概率表 |
| [NPLM](./nplm.md) | 固定窗口预测下一 token，交叉熵端到端更新 | 最近固定窗口与一次神经网络前向 |
| [RNN](./rnn.md) / [LSTM](./lstm.md) | 时间展开后通过 BPTT 更新共享参数 | 固定大小递归状态，生成时还依赖上一步 token |
| [Seq2Seq](./seq2seq.md) | Encoder 条件下使用真实目标前缀做教师强制 | Encoder 表示、Decoder 状态与解码搜索 |
| [BERT](./bert.md) | 破坏部分输入并恢复被选 token，再按任务微调 | 完整输入的一次双向编码与任务头 |
| [GPT](./gpt.md) | 目标右移一位，因果 mask 下预测所有下一 token | Prompt、逐步生成结果与各层 KV cache |
| [状态空间模型](./state-space-model.md) | 整段 Selective Scan 后计算序列任务损失 | 每层固定大小状态与当前 token |

同一种 Transformer 主干可以使用不同训练目标和推理接口。BERT 与 GPT 的主要差异不止 Attention mask，还包括样本构造、损失位置和输出过程；Seq2Seq 则把源端编码与目标端生成明确分开。

---

## 先读哪一篇

- 想理解语言模型的概率定义：从 [N-gram](./n-gram.md) 开始；
- 想理解序列状态与梯度：读 [RNN](./rnn.md)，再读 [LSTM](./lstm.md)；
- 想理解翻译、摘要等条件生成：读 [Seq2Seq](./seq2seq.md)；
- 想理解现代预训练模型：先读 [Transformer](./transformer.md)，再按任务选择 [BERT](./bert.md) 或 [GPT](./gpt.md)；
- 想比较 Attention 之外的长序列主干：读 [状态空间模型](./state-space-model.md)。

[BERT](./bert.md)同时覆盖 RoBERTa、ALBERT、ELECTRA、DeBERTa 与 Sentence-BERT 的关键改造轴；不同模型的效果需要结合[评估层](../evaluation/index.md)中的任务、数据和指标进行比较。
