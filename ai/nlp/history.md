# NLP 历史：从统计语言模型到预训练 Transformer

## 概览

自然语言处理的发展主线，可以概括为两个持续推进的问题：

- 如何表示语言；
- 如何建模上下文。

如果把近几十年的核心变化压缩成一条主线，那么大致可以写成：

- **统计时代**：用离散计数和有限上下文近似语言；
- **神经网络时代**：用连续向量和可学习参数缓解稀疏性；
- **序列建模时代**：用 RNN、LSTM、Seq2Seq 处理变长输入输出；
- **注意力时代**：用 attention 与 self-attention 改写上下文读取方式；
- **预训练时代**：用 Transformer 作为统一基座，分别走向 BERT 与 GPT 两条主线。

从关键节点看，一条更细的时间线如下：

| 时期 | 代表节点 | 核心变化 |
| --- | --- | --- |
| 1980s-2000s | N-Gram | 用有限上下文与计数建模语言 |
| 1990-1997 | RNN、LSTM | 用递归状态处理序列，开始显式建模长程依赖 |
| 2003-2014 | NPLM、Embedding、word2vec | 从离散符号转向连续表示 |
| 2014-2015 | Seq2Seq、Attention | 建立条件序列生成与动态对齐框架 |
| 2017 | Transformer、Self-Attention、Positional Encoding | 以自注意力重构序列建模主干 |
| 2018 以后 | BERT、GPT | 预训练成为主流，理解与生成两条路线分化又汇合 |

---

## 一、统计语言模型时代：N-Gram

早期 NLP 的主流思路，是把语言建模为离散符号序列上的概率估计问题。N-Gram 的核心假设是：当前词只依赖有限长度的历史，而不是完整前文。

\[
P(w_t\mid w_1,\dots,w_{t-1})\approx P(w_t\mid w_{t-n+1},\dots,w_{t-1})
\]

这一路线的优点是：

- 训练方式直接，通常基于计数与平滑；
- 模型可解释性强；
- 在语音识别、输入法、早期机器翻译中都很实用。

但它的局限也非常明显：

- 词表一大就会严重稀疏；
- 高阶组合难以可靠估计；
- 很难处理长距离依赖；
- 不同词之间无法共享语义结构。

这些问题直接推动了后续神经语言模型与连续表示方法的发展。

- 相关专题：[N-Gram](../model/n-gram.md)

---

## 二、连续表示与神经语言模型：NPLM、Embedding、word2vec

### 1. NPLM：从离散计数到连续参数化

2003 年，Bengio 等人提出 NPLM（Neural Probabilistic Language Model）。它保留了“预测下一个词”的语言模型目标，但把上下文词映射到低维连续空间，再用前馈神经网络估计条件概率。

这一步的历史意义非常大：

- 词不再只是离散符号，而变成可学习向量；
- 相似上下文开始能够共享统计强度；
- 语言模型第一次系统进入“embedding + 神经网络”的范式。

它没有彻底摆脱固定窗口，但已经明确指出：**语言建模的关键，不只是更好的计数，也包括更好的表示。**

- 相关专题：[NPLM](../model/nplm.md)

### 2. Embedding：离散对象进入连续空间

随着神经语言模型的发展，Embedding 逐渐成为几乎所有 NLP 模型的标准入口。它解决的是一个更基础的问题：如何让词、子词、位置、类别等离散对象进入可优化的连续向量空间。

Embedding 的作用不只是“降维”，更重要的是：

- 让相似对象在空间中靠近；
- 让模型可以复用参数；
- 让后续网络在连续空间中学习语义与结构。

如果说 N-Gram 的世界是“每个词彼此独立”，那么 embedding 的世界就是“词与词之间存在可学习的几何关系”。

- 相关专题：[Embedding](../model/embedding.md)

### 3. word2vec：词向量走向大规模实用化

2013 年前后，word2vec 通过 CBOW、Skip-gram、Negative Sampling 等设计，把词向量训练做得足够高效，从而推动 embedding 在大规模语料上的广泛应用。

它的历史价值主要体现在两点：

- 从工程上证明高质量词向量可以被大规模高效学习；
- 从观念上推动 NLP 社区普遍接受“分布式表示”这一核心思想。

这使得“先学表示，再做任务”逐渐成为主流路线。

- 相关专题：[word2vec](../model/word2vec.md)

---

## 三、循环序列建模：RNN 与 LSTM

### 1. RNN：让模型显式拥有时间状态

NPLM 虽然引入了连续表示，但仍然依赖固定窗口。RNN 的关键改进在于：模型不再只看固定长度上下文，而是通过递归隐藏状态在时间维度上传递历史信息。

\[
h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
\]

RNN 的历史意义在于：

- 序列第一次被统一地建模为“状态递推过程”；
- 模型可以处理变长输入；
- 语言建模、序列标注、语音等任务开始有了统一神经结构。

但 RNN 很快暴露出核心缺陷：长序列训练中梯度消失与爆炸严重，远距离依赖难以稳定学习。

- 相关专题：[RNN](../model/rnn.md)

### 2. LSTM：用门控缓解长期依赖问题

LSTM 通过细胞状态和门控机制，对 RNN 的长程依赖问题做出系统修正。它的关键思想不是单纯“更深”，而是显式控制哪些信息保留、哪些信息遗忘、哪些信息输出。

因此，LSTM 在 2010 年代中期成为 NLP 的核心架构之一，尤其在：

- 语言模型；
- 机器翻译；
- 序列标注；
- 语音识别

等任务中广泛使用。

从历史上看，LSTM 并不是最终答案，但它是 Seq2Seq 与注意力机制普及前最重要的序列主干。

- 相关专题：[LSTM](../model/lstm.md)

---

## 四、条件生成框架：Seq2Seq

2014 年前后，Seq2Seq 建立了“输入一段序列，输出另一段序列”的统一建模框架。它的核心结构是编码器-解码器：

- 编码器读取输入序列；
- 解码器逐步生成输出序列。

\[
P(Y\mid X)=\prod_{t=1}^{m}P(y_t\mid y_{<t},X)
\]

Seq2Seq 的历史意义在于，它让许多原本分散的任务被统一起来，例如：

- 机器翻译；
- 文本摘要；
- 对话生成；
- 改写与纠错。

不过，早期 Seq2Seq 有一个著名缺陷：编码器常常必须用单个固定长度向量压缩整段输入。这在长句场景下会形成明显的信息瓶颈。

- 相关专题：[Seq2Seq](../model/seq2seq.md)

---

## 五、动态对齐机制：Attention

Attention 的出现，本质上是对 Seq2Seq 固定向量瓶颈的修正。它让解码器在每一步都可以回看输入序列的不同部分，而不是只能依赖单一压缩向量。

\[
c_t=\sum_{i=1}^{n}\alpha_{t,i}h_i
\]

其中，\(\alpha_{t,i}\) 表示当前输出步对输入位置 \(i\) 的关注程度。

这一机制的历史价值体现在：

- 输入输出之间的对齐关系开始被显式建模；
- 长句翻译和长序列生成效果显著提升；
- “根据当前需求动态读取信息”成为后续架构的重要思想。

从这里开始，NLP 的重心不再只是“如何压缩历史”，而逐渐转向“如何按需读取上下文”。

- 相关专题：[Attention](../model/attention.md)

---

## 六、自注意力与位置信息：Self-Attention、Positional Encoding

### 1. Self-Attention：序列内部的全局交互

Attention 最初用于“解码器读取编码器输出”，而 self-attention 进一步把这一思想推广到序列内部：每个位置都可以直接查看整段序列中的其他位置。

这带来两个决定性变化：

- 任意两个位置之间的依赖路径大幅缩短；
- 整段序列的关系可以并行计算。

这使得模型不再必须依赖 RNN 那样的时间递归主干。

- 相关专题：[Self-Attention](../model/self-attention.md)

### 2. Positional Encoding：为注意力补上顺序坐标

Self-attention 的代价是：它本身不天然携带顺序信息。因此，Transformer 时代必须显式加入位置编码，让模型知道“谁在前、谁在后、相距多远”。

从历史上看，位置编码不是附属技巧，而是 self-attention 成为序列模型后所必须补上的结构性组件。

- 相关专题：[Positional Encoding](../model/positional-encoding.md)

---

## 七、Transformer：序列建模主干的重构

2017 年，Transformer 提出“Attention Is All You Need”，将 self-attention 从辅助机制提升为主干架构。它的核心变化是：

- 用 multi-head self-attention 替代递归状态传递；
- 用 position-wise FFN 补充非线性变换；
- 用残差连接、层归一化和位置编码形成稳定深层结构。

Transformer 的历史地位之所以关键，是因为它同时解决了两个长期问题：

- **建模能力**：远距离依赖可以直接建立；
- **训练效率**：整段序列可以高度并行。

从此以后，NLP 的基础设施发生了实质性变化：模型设计不再围绕“怎样改进 RNN”，而是围绕“怎样扩展 Transformer”展开。

- 相关专题：[Transformer](../model/transformer.md)

---

## 八、预训练范式：BERT 与 GPT

### 1. BERT：双向编码预训练路线

BERT 把 Transformer 编码器与遮蔽语言模型结合起来，建立了“预训练 + 微调”的统一理解框架。它的核心不在于生成文本，而在于学习可迁移的上下文化表示。

其影响主要体现在：

- 双向上下文表示成为主流；
- 大量理解任务可以共享同一个预训练主干；
- NLP 从“任务定制模型”明显转向“通用预训练表示”。

BERT 代表的是**编码器预训练路线**，更偏语言理解、分类、匹配、抽取与检索。

- 相关专题：[BERT](../model/bert.md)

### 2. GPT：自回归生成预训练路线

GPT 则把 Transformer 解码器与下一词预测结合起来，建立了生成式预训练路线。它的核心是：

\[
P(X)=\prod_{t=1}^{n}P(x_t\mid x_{<t})
\]

这一路线的关键优势在于：

- 训练目标统一；
- 文本生成天然成立；
- 很多任务都可以通过 prompt 改写为“给定前缀继续生成”。

随着模型规模、数据规模和对齐技术不断扩展，GPT 逐渐从“语言模型”演化为“通用生成式基础模型”。

- 相关专题：[GPT](../model/gpt.md)

---

## 九、主线关系总结

把整条历史主线压缩成几个核心转变，可以概括为：

### 1. 从离散计数到连续表示

N-Gram 的本质是离散统计；NPLM、Embedding、word2vec 的本质是让词进入连续空间，从而共享统计强度并缓解稀疏性。

### 2. 从固定窗口到可学习历史状态

RNN 与 LSTM 让模型不再依赖固定上下文窗口，而是通过递归状态处理变长序列。

### 3. 从固定压缩到动态读取

Seq2Seq 建立了条件生成框架，但早期版本受限于固定向量瓶颈；Attention 则把输入读取变成动态对齐过程。

### 4. 从递归主干到自注意力主干

Self-attention 与 Transformer 让序列模型不再以时间递归为主，而以全局交互为主。

### 5. 从任务专用模型到预训练基础模型

BERT 和 GPT 共同推动了预训练范式成为主流，只是路线不同：

- BERT 偏表示学习与理解；
- GPT 偏生成建模与统一提示接口。

---

## 十、建议阅读顺序

如果希望按知识依赖顺序阅读本仓库中的相关文档，可以参考以下路径：

1. [N-Gram](../model/n-gram.md)
2. [NPLM](../model/nplm.md)
3. [Embedding](../model/embedding.md)
4. [word2vec](../model/word2vec.md)
5. [RNN](../model/rnn.md)
6. [LSTM](../model/lstm.md)
7. [Seq2Seq](../model/seq2seq.md)
8. [Attention](../model/attention.md)
9. [Self-Attention](../model/self-attention.md)
10. [Positional Encoding](../model/positional-encoding.md)
11. [Transformer](../model/transformer.md)
12. [BERT](../model/bert.md)
13. [GPT](../model/gpt.md)

---

## Ref

- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- Katz, S. M. (1987). Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer.
- Elman, J. L. (1990). Finding Structure in Time.
- Werbos, P. J. (1990). Backpropagation Through Time: What It Does and How to Do It.
- Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning Long-Term Dependencies with Gradient Descent is Difficult.
- Kneser, R., and Ney, H. (1995). Improved Backing-Off for M-gram Language Modeling.
- Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory.
- Gers, F. A., Schmidhuber, J., and Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM.
- Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. (2003). A Neural Probabilistic Language Model.
- Morin, F., and Bengio, Y. (2005). Hierarchical Probabilistic Neural Network Language Model.
- Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.
- Mikolov, T. et al. (2013). Distributed Representations of Words and Phrases and their Compositionality.
- Sutskever, I., Vinyals, O., and Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks.
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.
- Bahdanau, D., Cho, K., and Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.
- Luong, M.-T., Pham, H., and Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation.
- Vaswani, A. et al. (2017). Attention Is All You Need.
- Peters, M. E. et al. (2018). Deep Contextualized Word Representations.
- Radford, A. et al. (2018). Improving Language Understanding by Generative Pre-Training.
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners.
- Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
- Brown, T. B. et al. (2020). Language Models are Few-Shot Learners.
