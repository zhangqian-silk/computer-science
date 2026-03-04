# NLP 历史：从 N-Gram 到 Transformer

## 概览

自然语言处理（NLP）的主线演进，可以概括为「统计方法 -> 神经网络方法 -> 预训练范式」。如果按关键技术节点来看，一条常见时间线如下：

- 1980s-2000s：N-Gram 统计语言模型成为主流
- 1990：Elman 提出 RNN（序列递归建模起点）
- 1997：LSTM 提出门控记忆机制（长期依赖建模关键改进）
- 2003：NPLM 提出分布式词表示与神经概率语言模型
- 2013：word2vec 推动 embedding 的大规模应用
- 2013-2014：在 GPU 与大规模数据推动下，RNN/LSTM 在 NLP 任务中进入应用高峰
- 2014：seq2seq 建立「编码器-解码器」生成框架并带动 LSTM 机器翻译落地
- 2015：attention 缓解长距离依赖与信息瓶颈
- 2017：Transformer 以自注意力为核心重构 NLP 基础架构
- 2018：GPT 与 BERT 分别确立生成式与判别式预训练路线

## N-Gram：统计语言模型时代

> 相关论文：
> - Shannon (1948)：提出信息论与有限上下文近似思想，为 n-gram 语言建模奠定理论基础。
> - Katz (1987)：提出 Katz Backoff，核心是「高频组合用高阶模型，低频组合回退到低阶模型」。
> - Kneser-Ney (1995)：提出绝对折扣与改进回退分布，显著提升稀疏场景泛化能力。

N-gram 是统计语言模型的代表方法，核心是假设当前词仅依赖有限长度历史（马尔可夫假设）：

$$
P(w_t|w_1,\dots,w_{t-1}) \approx P(w_t|w_{t-n+1},\dots,w_{t-1})
$$

其训练方式通常是基于计数的最大似然估计：

$$
P(w_t|w_{t-n+1}^{t-1})=\frac{C(w_{t-n+1}^{t})}{C(w_{t-n+1}^{t-1})}
$$

在工程上，它以实现简单和可解释性强著称，但也受限于数据稀疏、长距离依赖缺失与高阶组合爆炸，这些局限直接推动了后续神经语言模型与预训练范式的发展。

- 详细说明见：[N-Gram 专题文档](./model/n-gram.md)

## NPLM：神经语言模型的起点

> 相关论文：
> - Bengio et al. (2003)：首次系统提出基于词向量与前馈网络的神经概率语言模型，证明连续表示可缓解数据稀疏问题，并指出大词表 softmax 的训练瓶颈。
> - Morin and Bengio (2005)：提出分层 softmax，缓解大词表训练开销问题。

2003 年，Bengio 等人提出 NPLM（Neural Probabilistic Language Model）。该工作用神经网络学习词的低维连续向量，并在此基础上估计语言模型概率。

### 原理拆解

- 输入表示：将上下文词映射为 embedding 并拼接为向量。

$$
x_t=[e_{t-n+1};e_{t-n+2};\dots;e_{t-1}]
$$

- 前馈建模：通过 MLP 生成隐藏状态，再映射到词表分布。

$$
h_t=\tanh(W_1x_t+b_1),\quad z_t=W_2h_t+b_2
$$

$$
P(w_t|w_{t-n+1}^{t-1})=\text{softmax}(z_t)
$$

- 优化目标：最大化语料对数似然（等价最小化交叉熵）。

$$
\mathcal{L}=-\sum_{t}\log P(w_t|w_{t-n+1}^{t-1})
$$

- 关键思想：将离散词投影到连续空间，使相似上下文共享统计强度。
- 主要瓶颈：softmax 需要遍历大词表，训练代价较高。

- 贡献：首次系统展示神经语言模型优于传统统计方法的潜力
- 局限：在当时算力条件下训练成本高

## RNN：循环序列建模范式

> 相关论文：
> - Elman (1990)：提出简单循环网络（SRN），将前一时刻隐状态反馈到当前计算，实现序列上下文建模。
> - Werbos (1990)：系统化阐述 BPTT（Through Time 反向传播）训练路径。
> - Bengio et al. (1994)：分析长期依赖下梯度消失/爆炸问题，为后续门控结构提供理论动机。

RNN（Recurrent Neural Network）通过循环连接在时间维度上传递隐藏状态，使模型能够处理变长输入并建模时序依赖。
其核心思想提出较早（1990），但在 NLP 中的大规模应用高峰主要出现在 2013-2014 年之后。

### 原理拆解

- 状态递推：

$$
h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$

- 输出映射：

$$
y_t=g(W_{hy}h_t+b_y)
$$

- 序列条件概率分解（语言建模）：

$$
P(w_{1:T})=\prod_{t=1}^{T}P(w_t|w_{<t})
$$

- 训练方式：通过 BPTT 在展开时间轴后反向传播梯度。
- 主要问题：在长序列上容易出现梯度消失/爆炸，难以稳定学习远距离依赖。

## LSTM：门控记忆机制

> 相关论文：
> - Hochreiter and Schmidhuber (1997)：提出 LSTM，通过记忆单元与门控机制显著缓解长期依赖训练困难。
> - Gers et al. (2000)：引入 Forget Gate，形成现代 LSTM 常用结构。

LSTM（Long Short-Term Memory）是 RNN 的门控变体，通过显式细胞状态（cell state）与输入/遗忘/输出门控制信息流。
其结构提出于 1997 年，但真正“火爆”主要发生在 2014 年前后 seq2seq 机器翻译兴起阶段。

### 原理拆解

- 门控与状态更新：

$$
f_t=\sigma(W_f[x_t,h_{t-1}]+b_f),\quad
i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)
$$

$$
\tilde{c}_t=\tanh(W_c[x_t,h_{t-1}]+b_c),\quad
c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t
$$

$$
o_t=\sigma(W_o[x_t,h_{t-1}]+b_o),\quad
h_t=o_t\odot\tanh(c_t)
$$

- 核心作用：在时间维度上建立更稳定的误差信号通路，提升长期依赖建模能力。
- 工程影响：在 Transformer 出现前，LSTM 是机器翻译、语音识别、语言建模等任务的主力序列主干。

## Embedding

> 相关论文：
> - Collobert et al. (2011)：将词向量作为统一神经 NLP 框架的基础输入表示。
> - Pennington et al. (2014, GloVe)：用全局共现统计学习词向量，提升语义线性结构质量。
> - Peters et al. (2018, ELMo)：引入上下文相关词表示，推动动态 embedding 发展。
> - Devlin et al. (2018, BERT)：将上下文化表示扩展到大规模双向预训练范式。

Embedding 指将词或子词映射到稠密向量空间，使模型可以通过向量距离表达语义相似性。

### 原理拆解

- 矩阵映射：词表大小为 \(|V|\)，维度为 \(d\)，参数矩阵 \(E\in\mathbb{R}^{|V|\times d}\)。

$$
e_w = E^\top x_w
$$

- 相似度度量：语义相近词通常具有更高余弦相似度。

$$
\cos(e_i,e_j)=\frac{e_i^\top e_j}{\|e_i\|\|e_j\|}
$$

- 学习方式：embedding 参数在下游任务损失反向传播中联合更新。
- 表示演进：
- 静态 embedding：同一词在任意上下文对应同一个向量。
- 上下文 embedding：同一词在不同上下文有不同向量（如 ELMo、BERT）。
- 分布语义关系：在 SGNS 等目标下，词向量内积与 PMI 存在近似关系。

$$
u_w^\top v_c \approx PMI(w,c)-\log k
$$

## word2vec：高效词向量学习

> 相关论文：
> - Mikolov et al. (2013, ICLR Workshop)：提出 CBOW 与 Skip-gram 两类高效词向量训练目标。
> - Mikolov et al. (2013, NIPS)：提出负采样与子采样，显著提升大规模训练效率。
> - Mikolov et al. (2013)：展示词向量线性结构（类比关系），推动 embedding 在工业系统中落地。

2013 年，Mikolov 等人提出 word2vec，主要包含 CBOW 与 Skip-gram 两种训练目标，并结合负采样等技巧，大幅提升训练效率。

### 原理拆解

- CBOW：用上下文预测中心词。

$$
\max \sum_{t}\log P(w_t|w_{t-m},\dots,w_{t-1},w_{t+1},\dots,w_{t+m})
$$

- Skip-gram：用中心词预测上下文词。

$$
\max \sum_{t}\sum_{-m\le j\le m,j\ne 0}\log P(w_{t+j}|w_t)
$$

- softmax 形式：

$$
P(w_o|w_c)=\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}
$$

- Negative Sampling：将多分类近似为二分类，降低训练成本。

$$
\log \sigma(u_o^\top v_c)+\sum_{i=1}^{k}\log \sigma(-u_i^\top v_c)
$$

- 高频词子采样：降低高频噪声词影响。

$$
P_{discard}(w)=1-\sqrt{\frac{t}{f(w)}}
$$

- 训练流程：滑动窗口采样正样本 -> 构造负样本 -> SGD 更新输入/输出词向量。

## seq2seq：端到端生成范式

> 相关论文：
> - Sutskever et al. (2014)：提出 Encoder-Decoder LSTM 框架，统一建模变长输入输出序列；并以端到端学习替代传统 SMT 多模块流水线，同时指出源序列反转可改善优化路径并提升训练效果。
> - Cho et al. (2014)：提出 RNN Encoder-Decoder，为端到端序列转换提供早期实证基础。

2014 年，seq2seq（Sequence-to-Sequence）架构提出后，机器翻译等任务从「流水线模块拼接」转向「端到端建模」。

### 原理拆解

- 概率分解：目标序列按自回归方式生成。

$$
P(y|x)=\prod_{t=1}^{T_y}P(y_t|y_{<t},x)
$$

- 编码器递推（RNN/LSTM 抽象）：

$$
h_t=f_{enc}(h_{t-1},x_t)
$$

- 解码器递推（无 attention 时）：

$$
s_t=f_{dec}(s_{t-1},y_{t-1},c),\quad c=h_{T_x}
$$

- 输出分布：

$$
P(y_t|y_{<t},x)=\text{softmax}(W_os_t+b_o)
$$

- 训练目标（Teacher Forcing）：

$$
\mathcal{L}=-\sum_{t=1}^{T_y}\log P(y_t^\*|y_{<t}^\*,x)
$$

- 推理：贪心搜索或束搜索（Beam Search）；束搜索常用长度归一化分数：

$$
score(y)=\frac{1}{|y|^\alpha}\sum_{t}\log P(y_t|y_{<t},x)
$$

## Attention：解决长依赖与信息瓶颈

> 相关论文：
> - Bahdanau et al. (2015)：提出 Additive Attention，实现对齐与翻译的联合学习。
> - Luong et al. (2015)：提出多种打分函数与全局/局部注意力机制，推进工程应用。
> - Bahdanau et al. (2015) 与 Luong et al. (2015)：注意力权重提供了源词与目标词软对齐的可解释视角。

2015 年，Bahdanau Attention 被引入 seq2seq。在解码每一步时，模型不再只依赖单一全局向量，而是对输入序列进行动态加权。

### 原理拆解

- Additive Attention 打分：

$$
e_{t,i}=v^\top\tanh(W_s s_{t-1}+W_h h_i)
$$

- 权重归一化：

$$
\alpha_{t,i}=\frac{\exp(e_{t,i})}{\sum_j\exp(e_{t,j})}
$$

- 上下文向量：

$$
c_t=\sum_i\alpha_{t,i}h_i
$$

- 输出预测：

$$
P(y_t|y_{<t},x)=\text{softmax}(W_o[s_t;c_t]+b_o)
$$

- Dot-Product（Luong）变体：

$$
e_{t,i}=s_t^\top W h_i
$$

- 本质变化：从「单向量压缩」转为「每步按需读取输入序列」，显著改善长句翻译。

## Transformer：自注意力成为主干

> 相关论文：
> - Vaswani et al. (2017)：提出 Transformer，以多头自注意力替代 RNN/CNN 主干并显著提升并行性，并在机器翻译任务上以更低训练成本取得更优效果。
> - Shaw et al. (2018)：提出相对位置表示，改进长序列位置建模方式。

2017 年，《Attention Is All You Need》提出 Transformer，核心改动是用多头自注意力替代循环结构，并通过位置编码注入顺序信息。

### 原理拆解

- Scaled Dot-Product Attention：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- Multi-Head Attention：

$$
\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\text{MHA}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O
$$

- 前馈网络（Position-wise FFN）：

$$
\text{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2
$$

- 位置编码（正余弦）：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d}}\right),\quad
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

- 解码器因果 Mask：确保位置 \(t\) 只能访问 \(\le t\) 的 token。
- 复杂度特征：单层自注意力时间复杂度约为 \(O(n^2d)\)，并行性显著优于 RNN。

## BERT：双向编码预训练范式

> 相关论文：
> - Devlin et al. (2018)：提出「预训练 + 微调」统一范式，通过 MLM+NSP 的双向预训练提升语言理解能力，并在 GLUE、SQuAD 等基准上刷新当时 SOTA。
> - Liu et al. (2019, RoBERTa)：验证并改进 BERT 训练策略，强化预训练规模与稳定性。

BERT（Bidirectional Encoder Representations from Transformers）由 Devlin 等人在 2018 年提出，核心是通过 Transformer Encoder 双向建模大规模预训练，再迁移到下游任务。

### 原理拆解

- 输入表示：token、segment、position 三种 embedding 相加。

$$
h_i^{(0)}=E_{token}(x_i)+E_{segment}(x_i)+E_{position}(i)
$$

- MLM 目标：随机 mask 部分 token，预测被遮蔽词。

$$
\mathcal{L}_{MLM}=-\sum_{i\in M}\log P(x_i|x_{\backslash M})
$$

- NSP 目标：判断句子 B 是否为句子 A 的下一句。

$$
\mathcal{L}_{NSP}=-\big[y\log p+(1-y)\log(1-p)\big]
$$

- 总损失：

$$
\mathcal{L}=\mathcal{L}_{MLM}+\mathcal{L}_{NSP}
$$

- 微调机制：在预训练参数上接轻量任务头，端到端微调。
- 能力侧重：更偏语言理解任务（分类、序列标注、抽取式问答）。

## GPT：自回归生成预训练范式

> 相关论文：
> - Radford et al. (2018)：提出 GPT-1，验证生成式预训练对多类 NLP 任务的迁移能力。
> - Radford et al. (2019)：提出 GPT-2，展示规模扩展带来的强零样本生成能力。
> - Brown et al. (2020)：提出 GPT-3，系统展示 in-context learning 与 few-shot 能力。

GPT（Generative Pre-trained Transformer）由 OpenAI 在 2018 年提出，核心是基于 Decoder-only Transformer 的单向自回归建模。

### 原理拆解

- 自回归语言建模：

$$
P(x)=\prod_{t=1}^{T}P(x_t|x_{<t})
$$

- 训练损失：

$$
\mathcal{L}=-\sum_{t=1}^{T}\log P(x_t|x_{<t})
$$

- 因果 Mask 注意力：

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top+M}{\sqrt{d_k}}\right)V
$$

- 其中 \(M_{ij}=-\infty\)（当 \(j>i\)）用于屏蔽未来位置。
- 迁移路径：
- GPT-1：预训练后微调。
- GPT-2/3：通过规模扩展与 prompt 完成 zero-shot/few-shot 迁移。
- 能力侧重：更偏自然语言生成、续写、对话与通用指令跟随。

## 模型关系与对比

1. 从离散统计到连续表示：
N-Gram 依赖离散计数与局部上下文；NPLM、Embedding、word2vec 将词映射到连续向量空间，使语义相近样本可共享统计强度，显著缓解稀疏问题。

2. 从固定压缩到动态对齐：
RNN/LSTM 为 seq2seq 提供了可学习的序列编码与解码骨架；seq2seq 先以单一向量压缩源序列，再逐步生成目标序列；Attention 通过每步动态读取输入，直接缓解长句场景下的信息瓶颈。

3. 从循环结构到自注意力主干：
Transformer 去除了 RNN 的时间递归路径，以并行自注意力作为统一基础模块，为大规模预训练提供了更高效的基础设施。

4. 从通用预训练到任务分化：
BERT 与 GPT 均建立在 Transformer 之上，但建模方向不同。BERT 偏双向理解（判别式任务），GPT 偏单向生成（生成式任务），两者共同推动了“预训练 + 迁移”成为主流范式。

## 小结

从历史脉络看，NLP 的核心转变是「如何表示语言」与「如何建模上下文」：

- N-Gram：基于离散计数的局部上下文建模
- NPLM 与 embedding：进入连续表示空间
- RNN 与 LSTM：建立循环序列建模与门控记忆机制
- word2vec：高效学习通用词向量
- seq2seq 与 attention：建立可学习的序列生成与对齐机制
- Transformer：统一并放大上下文建模能力，成为现代 NLP 基座
- BERT：以双向编码预训练强化语言理解
- GPT：以自回归预训练强化语言生成与通用能力

## Ref

- Shannon, C. E. (1948). A Mathematical Theory of Communication.
- Katz, S. M. (1987). Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer.
- Kneser, R., and Ney, H. (1995). Improved Backing-Off for M-gram Language Modeling.
- Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. (2003). A Neural Probabilistic Language Model.
- Elman, J. L. (1990). Finding Structure in Time.
- Werbos, P. J. (1990). Backpropagation Through Time: What It Does and How to Do It.
- Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning Long-Term Dependencies with Gradient Descent is Difficult.
- Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory.
- Gers, F. A., Schmidhuber, J., and Cummins, F. (2000). Learning to Forget: Continual Prediction with LSTM.
- Morin, F., and Bengio, Y. (2005). Hierarchical Probabilistic Neural Network Language Model.
- Collobert, R. et al. (2011). Natural Language Processing (Almost) from Scratch.
- Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space.
- Mikolov, T. et al. (2013). Distributed Representations of Words and Phrases and their Compositionality.
- Pennington, J., Socher, R., and Manning, C. D. (2014). GloVe: Global Vectors for Word Representation.
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.
- Sutskever, I., Vinyals, O., and Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks.
- Bahdanau, D., Cho, K., and Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.
- Luong, M.-T., Pham, H., and Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation.
- Vaswani, A. et al. (2017). Attention Is All You Need.
- Shaw, P., Uszkoreit, J., and Vaswani, A. (2018). Self-Attention with Relative Position Representations.
- Peters, M. E. et al. (2018). Deep Contextualized Word Representations.
- Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
- Radford, A. et al. (2018). Improving Language Understanding by Generative Pre-Training.
- Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners.
- Brown, T. B. et al. (2020). Language Models are Few-Shot Learners.
