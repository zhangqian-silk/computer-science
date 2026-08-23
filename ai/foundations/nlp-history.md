# NLP 方法演化：上下文如何被表示和读取

自然语言处理的发展围绕两个持续变化的问题展开：语言单位怎样表示，当前预测怎样获得上下文。表示从离散符号走向上下文化向量，上下文则从固定窗口、递归状态逐步演化为动态读取与全局交互。

::: info 符号与约定
沿用[数学与符号约定](./math-notation.md)。$w_t$ 是第 $t$ 个 token，$T$ 是序列长度，$n$ 是 N-gram 阶数；$x_t,h_t$ 表示输入与递归隐藏状态；$X,Y$ 表示条件生成的输入、输出序列；$P(A\mid B)$ 表示条件概率。
:::

| 阶段 | 表示 | 上下文载体 | 主要突破 | 未解决问题 |
| --- | --- | --- | --- | --- |
| 统计语言模型 | 离散 token 与计数 | 固定 $n$-gram 窗口 | 语言概率可估计、可比较 | 稀疏、窗口有限 |
| 分布式表示 | 稠密词向量 | 固定窗口或局部共现 | 相似对象共享统计强度 | 一词一向量、缺少语境 |
| 递归网络 | 动态隐藏状态 | 时间递推 | 支持变长历史 | 串行、长依赖困难 |
| Seq2Seq + Attention | 编码器状态 | 解码步按需读取 | 条件生成与软对齐 | 主干仍可能串行 |
| Transformer | 上下文化 token | 全局或受限注意力 | 并行训练、短依赖路径 | 长序列成本高 |
| 预训练基础模型 | 大规模上下文表示 | 统一预训练目标 | 迁移、生成与少样本接口统一 | 可靠性、成本与评估更复杂 |

架构演化之外，训练信号与推理接口也在同步变化：

| 阶段 | 主要训练信号 | 推理时的核心操作 |
| --- | --- | --- |
| N-gram | 语料计数与平滑参数估计 | 查询概率表并逐级回退 |
| NPLM / word2vec | 固定窗口预测或对比目标 | 神经网络前向或向量相似度 |
| RNN / LSTM | 序列损失通过 BPTT 回传 | 持续更新递归状态 |
| Seq2Seq + Attention | 教师强制下的条件序列损失 | 编码输入并搜索输出序列 |
| BERT | 恢复被破坏 token，再微调任务头 | 完整输入一次双向编码 |
| GPT | 因果下一 token 预测及后续对齐 | Prefill 后逐 token Decode |

因此，判断两种方法的差别时，需要同时比较「信息怎样传播」「误差从哪里产生」和「使用时保存什么状态」。只比较参数量或论文年份，无法解释模型行为。

---

## 从计数到概率模型

早期语言模型用链式法则分解句子概率，再用有限阶马尔可夫假设截断历史：

$$
P(w_1,\ldots,w_T)
\approx
\prod_{t=1}^{T}P(w_t\mid w_{t-n+1:t-1})
$$

[N-gram](../model/n-gram.md) 的价值在于建模对象明确、估计与误差可解释；它的根本限制也来自同一假设。窗口扩大时组合数量急剧增长，未见事件需要平滑与回退，语义相近的词仍被视为毫无关系的符号。

---

## 连续表示让统计能够共享

[NPLM](../model/nplm.md) 把上下文词映射成稠密向量，再用神经网络预测下一词。相似词的参数和隐藏特征可以共享，从而缓解离散计数的稀疏性。

[word2vec](../representation/word2vec.md) 进一步把重点从完整句子概率转向高效学习词表示。CBOW、Skip-gram 与负采样证明了局部共现可以塑造有用的向量几何，但每个词仍只有一个静态向量。「银行」在金融与河岸语境中的表示不会自动改变。

这一步建立了现代 NLP 的通用接口：[Embedding](../representation/embedding.md) 将离散 ID 转换成可微分的连续表示。

---

## 递归状态取代固定窗口

[RNN](../model/rnn.md) 用状态递推把可变长度历史压缩到 $h_t$：

$$
h_t=f(x_t,h_{t-1})
$$

它不再显式限制最近几个 token，但早期信息必须经过连续的状态变换才能影响远处输出。长链条上的梯度消失或爆炸，使理论上的无限历史很难转化为稳定的长期记忆。

[LSTM](../model/lstm.md) 增加可加性细胞状态和门控，允许模型学习保留、写入与遗忘。它显著改善了长依赖训练，但时间步之间仍存在串行依赖。

---

## 条件生成与动态读取

[Seq2Seq](../model/seq2seq.md) 将「读输入」与「生成输出」拆成 Encoder 和 Decoder，使翻译、摘要等任务可以统一写为条件概率：

$$
P(Y\mid X)=\prod_{t=1}^{m}P(y_t\mid y_{<t},X)
$$

早期实现把整个输入压缩成一个固定向量，长句容易形成信息瓶颈。[Attention](../mechanism/attention.md) 让每个解码步根据当前 query 重新加权编码器状态，将一次性压缩改为按需读取。它最初是递归主干的辅助模块，随后成为新架构的核心。

---

## Self-Attention 重写序列主干

[Self-Attention](../mechanism/self-attention.md) 让同一序列中的每个位置直接读取其他位置。[Transformer](../model/transformer.md) 再把这种交互与位置表示、残差连接、归一化和前馈网络组合成可堆叠 block。

这次变化同时缩短了任意位置间的信息路径，并允许训练时对整个序列并行计算。代价是标准注意力的连接数随长度近似二次增长，而且顺序必须通过[位置表示](../mechanism/positional-encoding.md)显式注入。

Transformer 随后分化出三种常见结构接口：

- Encoder-only 读取双向上下文，适合表示与判别任务，[BERT](../model/bert.md)是代表；
- Decoder-only 在因果掩码下预测下一 token，[GPT](../model/gpt.md)沿这条路线扩展生成能力；
- Encoder-decoder 保留条件生成接口，继续服务翻译、摘要和转换任务。

---

## 预训练改变任务组织方式

统计模型与早期神经模型常围绕单一任务训练。预训练范式先在大规模语料上学习通用参数，再通过微调、提示或参数高效适配转向具体任务。[LoRA](../mechanism/lora.md)等方法改变的是适配成本，而不是基座模型的基本预测接口。

当模型进入检索、工具调用和长任务执行，研究对象又从单个网络扩展为系统：[文本嵌入](../representation/text-embedding.md)与检索提供外部证据，[评估层](../evaluation/index.md)验证能力主张，[Agent 系统](../agent/index.md)管理动作、状态与权限。

这并不意味着旧方法全部失效。N-gram 仍可作为可解释基线，RNN 适合某些流式与资源受限任务，word2vec 仍是轻量离散对象表示。历史演化改变的是默认折中，而不是给所有场景规定唯一答案。

---

## 参考文献

- Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model*.
- Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space*.
- Hochreiter, S., and Schmidhuber, J. (1997). *Long Short-Term Memory*.
- Bahdanau, D., Cho, K., and Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*.
- Vaswani, A. et al. (2017). *Attention Is All You Need*.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
