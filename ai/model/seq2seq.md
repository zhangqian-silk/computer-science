# Seq2Seq：把条件生成拆成编码与解码

Seq2Seq 定义条件生成接口：编码输入 $X$，在它和目标前缀的条件下生成 $Y$。输入输出可以不同长度，主干可以是循环网络或 Transformer；固定摘要、Attention 与搜索策略是进一步的设计选择，不应混成一个模型名称。

<Seq2SeqDecodeExplorer />

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$X=(x_1,\ldots,x_n)$ 是源序列，$Y=(y_1,\ldots,y_m)$ 是目标序列；$h_i^{\text{enc}}$ 是 Encoder 状态，$s_t$ 是 Decoder 状态，$c$ 或 $c_t$ 是源端条件表示；$e_{t,i},\alpha_{t,i}$ 是对齐分数与归一化权重，$y_t^*$ 是真实目标 token。
:::

$$
P(Y\mid X)
=
\prod_{t=1}^{m}
P(y_t\mid y_{<t},X)
$$

---

## 经典 RNN Encoder-Decoder

Encoder 顺序读取输入：

$$
h_i^{\text{enc}}
=
f_{\text{enc}}(x_i,h_{i-1}^{\text{enc}})
$$

最早的结构用最后状态作为固定上下文：

$$
c=h_n^{\text{enc}}
$$

Decoder 以 $c$ 初始化或在每一步使用它：

$$
s_t=f_{\text{dec}}(E_y[y_{t-1}],s_{t-1},c)
$$

$$
P(y_t\mid y_{<t},X)
=
\left[\operatorname{softmax}(W_os_t+b_o)\right]_{y_t}
$$

Encoder 与 Decoder 不要求共享参数、词表或时间长度。输入通过 $c$ 提供条件，输出长度由结束 token 决定。

~~~mermaid
flowchart LR
	X["输入序列 X"] --> E["Encoder"]
	E --> C["条件表示"]
	B["&lt;BOS&gt;"] --> D["Decoder"]
	C --> D
	D --> Y1["y₁"]
	Y1 --> D
	D --> Y2["y₂"]
	Y2 --> D
	D --> EOS["&lt;EOS&gt;"]
~~~

---

## 为什么输入输出长度可以不同

Decoder 从 &lt;BOS&gt; 开始，每一步产生词表分布。选中的 token 作为下一步输入，直到生成 &lt;EOS&gt; 或达到长度上限。模型学习的是条件概率，不是输入位置到输出位置的固定一一映射。

这使同一框架可处理：

- 翻译：两种语言的长度和词序不同；
- 摘要：输出通常更短；
- 语音识别：输入帧远多于输出 token；
- 改写与结构化生成：输出接口由目标序列定义。

---

## 固定向量为什么会失效

$c=h_n^{\text{enc}}$ 要同时保存所有源端细节。输入越长，早期信息到最后状态的路径越长，单个向量也更容易形成容量瓶颈。双向 Encoder 能改善输入表示，却不能取消一次性压缩。

Attention 让 Decoder 在每个时间步重新读取全部 Encoder 状态：

$$
e_{t,i}=a(s_{t-1},h_i^{\text{enc}})
$$

$$
\alpha_{t,i}
=
\frac{\exp(e_{t,i})}{\sum_j\exp(e_{t,j})}
$$

$$
c_t=\sum_i\alpha_{t,i}h_i^{\text{enc}}
$$

Decoder 从固定 $c$ 改为随输出步变化的 $c_t$，一次性摘要变成动态软对齐。打分函数与矩阵化形式见 [Attention](../mechanism/attention.md)。

---

## 教师强制

训练样本的目标序列已知，因此第 $t$ 步通常输入真实的 $y_{t-1}$，并对所有位置求交叉熵：

$$
\mathcal{L}
=
-\sum_{t=1}^{m}
\log P(y_t^*\mid y_{<t}^*,X)
$$

教师强制允许预先构造所有真实前缀，但不消除 RNN Decoder 的状态串行依赖；Transformer Decoder 才能在因果 mask 下并行计算这些位置。推理前缀来自模型预测，错误会改变后续输入分布，这种训练与使用差异通常称为曝光偏差。

一条翻译样本可以拆成如下数据流：

| 阶段 | 输入 | 作用或目标 |
| --- | --- | --- |
| Encoder | `I love you` | 产生源端条件表示 |
| Decoder 第 1 步 | `<BOS>` | 预测 `我` |
| Decoder 第 2 步 | `我` | 预测 `爱` |
| Decoder 第 3 步 | `爱` | 预测 `你` |
| Decoder 第 4 步 | `你` | 预测 `<EOS>` |

训练时 Decoder 第 2 步读取的「我」来自真实目标，而不是第 1 步的预测。假设四个正确目标的条件概率依次为 $0.7,0.6,0.8,0.9$，则整段目标在当前模型下的概率为：

$$
P(Y\mid X)=0.7\times0.6\times0.8\times0.9=0.3024
$$

对应序列负对数似然为：

$$
-\log P(Y\mid X)
=
-\sum_{t=1}^{4}\log P(y_t\mid y_{<t},X)
\approx1.196
$$

乘法概率转为对数和后，每个位置都能提供训练信号；某一步给正确 token 的概率很低，会直接增大总损失。

Scheduled sampling 尝试在训练中混入模型预测，但会改变优化问题且不保证更好。更根本的做法是用与任务一致的序列级评估、改善数据和解码，并明确训练目标与最终指标的差异。

---

## Padding 与 mask

batch 中的序列长度不同，通常补齐到共同长度。Padding token 不应：

- 更新 RNN 的有效状态；
- 参与 Attention 权重；
- 计入目标交叉熵；
- 被当作可生成的正常内容。

因此，源端需要有效长度或 padding mask，目标端损失也需 mask。Transformer Decoder 还要加入因果 mask，防止训练时读取未来目标。Padding 是张量整理手段，mask 才定义哪些计算有效。

---

## 解码是搜索问题

页面交互的推理模式使用显式玩具概率树实际展开候选，不把参考文本传给搜索器。比较 Greedy 与 Beam 时要固定同一分布，查看累计路径概率而不是每步单独最大值；训练模式展示的真实前缀是另一个接口。

每一步取最大概率 token 的贪心解码不保证得到概率最高的整段序列。Beam Search 保留 $B$ 条部分路径，并累积对数概率：

$$
\log P(Y\mid X)
=
\sum_{t=1}^{m}
\log P(y_t\mid y_{<t},X)
$$

直接相加常偏向短序列，因为每项对数概率通常为负。长度归一化或长度惩罚可调整比较口径。Beam 越大也不保证任务质量越高；开放生成还可能需要 temperature、top-k 或 nucleus sampling。

一个两步例子可以说明贪心为什么不保证最优。第 1 步有两个候选：

$$
P(A\mid X)=0.6,\qquad P(B\mid X)=0.4
$$

贪心会选择 $A$。若后续结束概率为：

$$
P(\text{<EOS>}\mid A,X)=0.4,\qquad
P(\text{<EOS>}\mid B,X)=0.9
$$

再设 A 后剩余 0.6 分给两个概率各 0.3 的候选，B 后其余候选总和 0.1；这样 Greedy 在 A 后确实会选择 EOS，而非未说明的更高概率候选。

两条完整路径的概率分别是：

$$
P(A,\text{<EOS>}\mid X)=0.6\times0.4=0.24
$$

$$
P(B,\text{<EOS>}\mid X)=0.4\times0.9=0.36
$$

局部概率更高的 $A$ 反而形成较差的完整序列。Beam Width 为 2 时，两条前缀都能保留到下一步，从而找出 $B$ 路径；真实词表和序列更大时，Beam Search 仍只是受宽度限制的近似搜索。

解码参数属于模型输出接口的一部分，评估时必须固定。

推理过程与训练存在三个关键差异：

1. Encoder 表示通常只计算一次，Decoder 在整个生成过程中反复读取；
2. Decoder 下一步输入来自自己的选择，而不是真实目标；
3. 系统需要维护每条候选路径的状态、累计得分与停止标记。

RNN Decoder 为每条路径保存递归状态；Transformer Decoder 为每条路径保存各层 KV cache。扩展或重排 beam 时，这些状态也必须同步复制与重排，否则 token 路径与模型状态会错位。

---

## Transformer 如何继承 Seq2Seq

Sutskever 等人的序列到序列工作展示深层 LSTM 编码与解码；Cho 等人的 Encoder-Decoder 工作也在翻译中学习句对表示；Bahdanau 的动态对齐松开固定摘要瓶颈；Transformer 替换循环主干但保留条件生成分解。它们的关联是接口继承与信息读取改造，不是翻译任务只能选一个历史版本。

Transformer Encoder 并行产生源端上下文化表示，Decoder 使用两类 Attention：

- 带因果 mask 的 Self-Attention 读取目标前缀；
- Cross-Attention 读取 Encoder 输出。

它改变了底层信息传播方式，却保留相同的条件分解、教师强制和自回归解码。因此「Seq2Seq」应理解为任务与概率接口，而不是「RNN 翻译模型」的别名。

完整 block 结构见 [Transformer](./transformer.md)。生成结果的 BLEU、ROUGE、忠实性与人工评估见[生成评估](../evaluation/generation-evaluation.md)。

---

## 参考文献

- Sutskever, I., Vinyals, O., and Le, Q. V. (2014). [*Sequence to Sequence Learning with Neural Networks*](https://arxiv.org/abs/1409.3215).
- Cho, K. et al. (2014). [*Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*](https://aclanthology.org/D14-1179/).
- Bahdanau, D., Cho, K., and Bengio, Y. (2015). [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473).
