# NPLM：用连续表示平滑固定窗口语言模型

神经概率语言模型（Neural Probabilistic Language Model, NPLM）仍使用最近若干 token 预测下一 token，但不再为每个离散 N-gram 独立保存概率。上下文词先映射为向量，再由神经网络共享参数完成预测。

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$\mathcal{V}$ 是词表，$m$ 是上下文窗口长度，$w_t$ 是目标 token；$E$ 是 Embedding 矩阵，$x_t,h_t,z_t$ 分别是拼接输入、隐藏状态和 logits；$W,b$ 表示层参数，$\theta$ 表示全部可训练参数，$\mathcal{B}$ 是 mini-batch 样本集合，$\mathcal{L}$ 表示 batch 平均损失。
:::

---

## 模型结构

设上下文长度为 $m$，固定词表为 $\mathcal{V}$：

$$
(w_{t-m},\ldots,w_{t-1})\longrightarrow w_t
$$

用共享 embedding 矩阵 $E$ 查出词向量并拼接：

$$
x_t=
[E[w_{t-m}];\ldots;E[w_{t-1}]]
\in\mathbb{R}^{md}
$$

一个基本隐藏层为：

$$
h_t=\tanh(W_hx_t+b_h)
$$

输出 logits 与概率：

$$
z_t=W_oh_t+b_o,\qquad
P(w_t=v\mid w_{t-m:t-1})
=
\frac{\exp(z_{t,v})}{\sum_{u\in\mathcal{V}}\exp(z_{t,u})}
$$

原始 NPLM 还可从输入到输出加入直接线性连接。无论具体层数如何，它都是固定窗口、共享参数的条件概率模型。

```mermaid
flowchart LR
	W["最近 m 个 token"] --> E["共享 Embedding 查表"]
	E --> C["拼接为 md 维向量"]
	C --> H["隐藏层提取上下文特征"]
	H --> Z["投影到词表 logits"]
	Z --> P["Softmax：下一 token 分布"]
```

这里有两种不同的参数共享：同一个词无论出现在窗口哪个样本中，都访问同一组 embedding 参数；所有可能的离散上下文也都经过同一组隐藏层和输出层参数。前者让词获得可复用表示，后者让相似上下文不必分别维护概率表。

---

## 连续表示怎样缓解稀疏

在 N-gram 中，「猫 在 睡觉」与「狗 在 睡觉」是两个独立计数事件。NPLM 中，若「猫」和「狗」在训练中承担相似预测角色，其 embedding 与下游隐藏特征会相近；一个上下文的梯度因此可改善附近的上下文。

这种泛化不是把未见组合直接当成已见组合，而是让条件概率成为连续参数的函数：

$$
P(w_t\mid h)=f_\theta(E[h])
$$

相似输入在参数空间中共享计算路径，从离散回退变成学习到的平滑。

---

## 语料怎样变成训练样本

假设窗口长度 $m=2$，句子为：

```text
<BOS> 我 爱 学习 <EOS>
```

滑动窗口产生的监督样本为：

| 上下文输入 | 预测目标 |
| --- | --- |
| `<BOS> <BOS>` | `我` |
| `<BOS> 我` | `爱` |
| `我 爱` | `学习` |
| `爱 学习` | `<EOS>` |

每个样本都只包含目标左侧最近的 $m$ 个 token。窗口左侧缺少的部分可用边界符补齐；padding 和边界符的具体约定必须在训练与推理时保持一致。

语料通过滑动窗口构造监督样本。对 mini-batch $\mathcal{B}$，模型最小化交叉熵：

$$
\mathcal{L}
=
-\frac{1}{|\mathcal{B}|}
\sum_{(h,y)\in\mathcal{B}}
\log P_\theta(y\mid h)
$$

梯度同时更新：

- 上下文中被访问的 embedding 行；
- 隐藏层参数；
- 输出层参数。

Embedding 因此不是独立预处理，而是由下一 token 预测目标端到端塑造。完整 softmax 的输出层参数与计算仍随词表 $|\mathcal{V}|$ 增长；分层 softmax、采样 softmax 等方法可减少训练开销，但会改变优化近似。

对于单个目标 $y$，softmax 与交叉熵对 logit 的梯度为：

$$
\frac{\partial\ell}{\partial z_v}
=
P(v\mid h)-\mathbb{1}[v=y]
$$

正确 token 的 logit 被向上推动，其他 token 的 logit 被向下推动。梯度再经输出层、隐藏层传到窗口内的 embedding 行。一个词在大量上下文中反复收到这些信号，其向量逐渐编码对预测有用的共性。

完整训练循环可以概括为：

1. 对语料分词并加入边界 token；
2. 滑动固定窗口，构造「上下文—下一 token」样本；
3. 查 embedding、拼接窗口并执行前向计算；
4. 计算词表交叉熵；
5. 反向传播，同时更新 embedding、隐藏层和输出层；
6. 在验证集上选择窗口长度、隐藏维度与停止时机。

---

## 一个泛化例子

假设训练语料频繁出现：

- 「猫 在 沙发 上」；
- 「狗 在 地毯 上」；
- 「猫 正在 睡觉」。

测试时出现未见过的「狗 正在 睡觉」。离散 Trigram 可能退回低阶分布；NPLM 则可利用「猫」与「狗」的向量邻近性，使两个上下文在隐藏空间产生相似激活，从而提高「睡觉」的概率。

这种能力依赖语料是否真的把两个词训练到相近区域。连续表示提供了共享机制，不保证语义关系必然正确。

---

## 推理时怎样预测和生成

对给定文本预测下一个 token 时，NPLM 不维护递归状态，也不回看完整前缀：

1. 按训练时相同的规则分词；
2. 只截取最近 $m$ 个 token，不足时补边界符；
3. 经过 embedding、隐藏层和 softmax 得到下一 token 分布；
4. 根据任务选择最大概率 token 或从分布采样；
5. 若继续生成，将新 token 追加到序列并向右滑动窗口。

例如窗口为 2 时，生成序列已经是「我 爱 学习」，下一步实际使用的上下文只有「爱 学习」。「我」已离开窗口，不可能再直接影响预测。这是 NPLM 与 N-gram 共同保留的有限上下文边界。

训练时，每个滑窗样本彼此独立，可在 batch 中并行处理；自回归生成时，下一轮窗口包含刚生成的 token，时间步之间仍然串行。这里的串行来自生成任务，而不是模型内部存在循环状态。

---

## 与相邻模型的区别

| 模型 | 上下文 | 参数共享方式 | 主要瓶颈 |
| --- | --- | --- | --- |
| N-gram | 固定窗口 | 回退与平滑 | 离散组合稀疏 |
| NPLM | 固定窗口 | embedding 与神经网络 | 窗口仍有限、softmax 昂贵 |
| RNN | 递归状态 | 时间步共享转移函数 | 串行与长梯度路径 |
| Transformer LM | Attention 可见范围 | token 间全局交互 | 长度成本与显存 |

NPLM 的历史意义在于把语言概率、分布式表示和反向传播放进同一个可训练模型。它没有解决长距离上下文，却为 [word2vec](../representation/word2vec.md) 和后续神经语言模型提供了表示学习基础。

---

## 评估边界

测试时应报告平均负对数似然或困惑度，并固定词表、分词和测试域。还可用词替换、低频上下文与未见组合分析连续表示是否真的带来泛化。几何近邻只能作为诊断，不能替代语言模型概率评估。

相关方法见[语言模型评估](../evaluation/language-model-evaluation.md)与[向量表示分析](../evaluation/embedding-geometry.md)。

---

## 参考文献

- Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model*.
- Mnih, A., and Hinton, G. E. (2009). *A Scalable Hierarchical Distributed Language Model*.
