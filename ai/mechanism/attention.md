# Attention：按相关性动态聚合信息

> 相关文献：
> - Bahdanau, Cho, and Bengio (2015)：在神经机器翻译中提出可学习对齐机制，开启现代 attention 方法。
> - Luong, Pham, and Manning (2015)：系统比较多种 attention 打分形式，推动其在 Seq2Seq 中普及。
> - Vaswani et al. (2017)：以 self-attention 为核心构建 Transformer，使 attention 从辅助模块上升为主干机制。
> - Shaw, Uszkoreit, and Vaswani (2018)：把相对位置信息引入 attention，增强序列关系建模。

本文讨论一般 attention 的统一抽象、数学骨架与典型变体，重点回答“attention 是什么、如何计算、为什么重要”。若关注序列内部的特化形式，可进一步阅读 [Self-Attention](./self-attention.md)；若关注 attention 如何被组织成完整主干架构，可进一步阅读 [Transformer](../model/transformer.md)。

## 问题背景与机制转向

Attention 的核心贡献，不只是“让模型看得更远”，而是把序列建模中的一个基本问题彻底改写了。

在早期 RNN / LSTM 或原始 Seq2Seq 中，模型通常需要把大量历史信息压缩进一个或少数几个隐藏状态里，再依赖这些状态继续向后传递。这样做的困难在于：

- 信息必须沿时间链条逐步传播；
- 越远位置之间的依赖路径越长；
- 输入一长，固定长度状态就容易成为瓶颈。

Attention 的思想是：**与其被动继承过去，不如在当前步骤主动查询整个信息集合，并按相关性动态聚合最有用的部分。**

这一变化带来了三层深远影响：

- 从建模方式看，它把“压缩全部历史”改写成“按需读取相关信息”；
- 从数学形式看，它把序列依赖改写成一组可微的矩阵权重运算；
- 从架构演化看，它最终促成了 Transformer，使 attention 从辅助机制变成主干计算单元。

从今天回看，Attention 之所以重要，不只是因为它提升了长程依赖建模能力，更因为它提出了一种新的统一范式：**先计算相关性，再依据相关性重组表示。**

---

## 路线图、符号约定与公式索引

本文统一使用教学记号组织叙述，避免不同论文之间原始符号互相冲突。主要符号如下：

| 符号 | 含义 |
| --- | --- |
| $L$ | 序列长度 |
| $d_{\mathrm{model}}$ | 模型隐藏维度 |
| $d_k$ | query / key 的维度 |
| $d_v$ | value 的维度 |
| $d_{\mathrm{ff}}$ | 前馈网络隐藏层维度 |
| $x_i$ | 第 $i$ 个位置的输入向量 |
| $X$ | 整段输入组成的矩阵 |
| $q_i,k_i,v_i$ | 第 $i$ 个位置的 query、key、value 向量 |
| $Q,K,V$ | 矩阵形式的 query、key、value |
| $s(q_i,k_j)$ | 位置 $i$ 对位置 $j$ 的匹配分数 |
| $A$ | 注意力权重矩阵 |
| $M$ | 掩码矩阵 |
| $W^Q,W^K,W^V,W^O$ | attention 相关投影矩阵 |

本文核心公式索引如下：

1. 线性投影：
$$
Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V
$$

2. 缩放点积 attention：
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

3. 带因果掩码的 attention：
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

4. 多头注意力：
$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O
$$

下文依次讨论问题动机、基本定义、数学展开、掩码机制、典型扩展与常见误区。位置编码、自注意力特化、Transformer block 组织与系统级训练推理问题，分别回到 [positional-encoding.md](./positional-encoding.md)、[self-attention.md](./self-attention.md) 与 [transformer.md](../model/transformer.md)。

下面这张交互式路线图概括了全文主线：

<AttentionRoadmap />

---

## 问题动机：为什么需要 Attention

### 固定长度状态为什么会成为瓶颈

在原始 Seq2Seq 中，编码器常把整段输入压缩成一个固定长度向量，再交给解码器使用。这种方案在短句上尚可，但输入一长，就会出现两个典型困难：

- 所有信息都要经过同一个固定容量通道；
- 远距离信息必须沿递归链条逐步传递，容易衰减、覆盖或扭曲。

从本质上说，问题不在于模型“记忆力差”，而在于它被迫用过于局促的方式组织上下文。

### 从“压缩全部历史”到“按需读取信息”

Attention 的关键转向在于：模型不再要求自己事先把所有信息压缩成一个全局摘要，而是在当前步骤到来时，再去判断“我现在最需要哪部分信息”。

因此，Attention 可以被概括为一种**内容寻址（content-based addressing）**机制：

- 当前状态产生一个查询；
- 候选位置提供可被匹配的索引；
- 模型按相关性分配权重，并动态读取内容。

这种思路与人类认知直觉也高度一致。无论是阅读长文、观察图像，还是回忆一段复杂事件，我们通常都不会平均使用全部信息，而是把注意力集中在当前最关键的词、区域或线索上。

### Attention 的一般性思想

随着模型发展，Attention 的含义已不再局限于“Seq2Seq 解码器回看编码器状态”。更一般地说，它是一种统一的计算模式：

- 给定一个 query；
- 对一组 key 计算匹配分数；
- 对 value 做加权聚合。

因此，它既可以用于文本序列，也可以用于图像 patch、音频帧、多模态条件输入、检索记忆库，甚至图结构或视频表示。

---

## Attention 的基本定义与三种典型形态

### 三步过程

Attention 可以抽象为一个三步计算过程：

1. 用 query 表示“当前到底要找什么”；
2. 计算 query 与各个 key 的匹配分数；
3. 用归一化后的权重对 value 做加权求和，得到新的上下文表示。

如果记当前 query 为 $q$，候选记忆单元为 $\{(k_i,v_i)\}_{i=1}^{L}$，则可写为：

$$
\alpha_i=\frac{\exp(s(q,k_i))}{\sum_{j=1}^{L}\exp(s(q,k_j))}
$$

$$
c=\sum_{i=1}^{L}\alpha_i v_i
$$

这里：

- $s(q,k_i)$ 决定“当前位置应该看谁更多”；
- $\alpha_i$ 决定“不同候选位置占多少权重”；
- $c$ 是最终聚合得到的上下文向量。

### Q、K、V 分别扮演什么角色

很多初学者最大的困惑，是为什么要把表示拆成 query、key、value 三部分。更准确的理解是：

- **query**：当前任务想找什么；
- **key**：每个候选位置如何被检索；
- **value**：每个候选位置真正提供什么内容。

若借用检索系统类比：

- query 像用户当前输入的搜索词；
- key 像候选文档的标题、标签、索引信息；
- value 像候选文档真正承载的正文内容。

因此，attention 并不是先找到“最匹配的 key”就直接把 key 输出，而是先用 key 完成匹配，再依据权重去汇聚对应的 value。

### Self-Attention、Masked Self-Attention 与 Cross-Attention

三者的核心公式非常接近，但输入来源与可见性约束不同。

| 机制 | $Q$ 来源 | $K,V$ 来源 | 是否看未来 | 典型位置 |
| --- | --- | --- | --- | --- |
| Self-Attention | 当前序列 | 当前序列 | 视任务而定 | 编码器、双向理解模型 |
| Masked Self-Attention | 当前序列 | 当前序列 | 不能看未来 | 自回归解码器 |
| Cross-Attention | 当前状态 | 外部条件序列 | 通常不受因果掩码约束 | Seq2Seq 解码器、多模态模型 |

这说明：attention 不同变体之间最根本的差别，不在公式骨架，而在 query、key、value 的来源与信息可见范围。

### 一个直观例子：词义消歧与指代解析

Self-Attention 的价值，并不只是“让每个词看见其他词”，而是让每个词能够依据上下文动态重写自己的表示。

例如，在句子「我昨天去银行存钱」中，词「银行」可以借助「存钱」获得“金融机构”意义；在句子「河边的银行已经被冲垮」中，它又会与其他上下文一起指向不同语义。  

再如句子：

> 这只动物没有穿过那条街道，因为它太累了。

当模型处理「它」时，query 会与整句中其他词的 key 计算匹配分数。若训练充分，模型通常会更关注「动物」与「累」等相关区域，而不会主要依赖「街道」。这正说明 attention 的强项在于：它能借助句内依赖关系完成上下文化语义建模，而不是孤立地理解单个词。

---

## 缩放点积 Attention 的数学展开

Transformer 中最核心的 attention 形式是**缩放点积注意力（Scaled Dot-Product Attention）**：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这一公式看起来简洁，但实际上浓缩了完整的“匹配、归一化、聚合”过程。下面按矩阵运算顺序展开。

从计算流程上看，其骨架可画为：

<AttentionMathFlow />

### 线性投影：从输入表示生成 $Q,K,V$

设输入序列长度为 $L$，模型维度为 $d_{\mathrm{model}}$，整段输入矩阵为：

$$
X\in\mathbb{R}^{L\times d_{\mathrm{model}}}
$$

Attention 不直接对 $X$ 做匹配，而是先通过三组线性投影生成：

$$
Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V
$$

其中：

$$
W^Q\in\mathbb{R}^{d_{\mathrm{model}}\times d_k},\qquad
W^K\in\mathbb{R}^{d_{\mathrm{model}}\times d_k},\qquad
W^V\in\mathbb{R}^{d_{\mathrm{model}}\times d_v}
$$

因此：

$$
Q\in\mathbb{R}^{L\times d_k},\qquad
K\in\mathbb{R}^{L\times d_k},\qquad
V\in\mathbb{R}^{L\times d_v}
$$

从线性代数角度看，这一步等价于把同一份输入表示映射到三个不同子空间，使“检索索引”和“实际内容”能够分开表达。

### 点积打分：为什么是 $QK^\top$

第二步计算匹配分数矩阵：

$$
E=QK^\top
$$

因为 $Q\in\mathbb{R}^{L\times d_k}$，$K^\top\in\mathbb{R}^{d_k\times L}$，所以：

$$
E\in\mathbb{R}^{L\times L}
$$

矩阵中第 $(i,j)$ 个元素为：

$$
e_{ij}=q_i^\top k_j
$$

其含义是：第 $i$ 个位置当前对第 $j$ 个位置分配的原始相关性得分。

之所以采用点积，有两个核心原因：

- 它是向量相似度的一种自然代数量化；
- 它允许我们用一次矩阵乘法并行得到所有位置对之间的分数。

### 为什么要除以 $\sqrt{d_k}$

若直接把 $QK^\top$ 送入 softmax，训练往往会不稳定。原因可以用随机变量的简单近似说明。

设 $q_i,k_j\in\mathbb{R}^{d_k}$ 的各维分量近似独立，且满足：

$$
\mathbb{E}[q_{ir}]=\mathbb{E}[k_{jr}]=0,\qquad
\mathrm{Var}(q_{ir})=\mathrm{Var}(k_{jr})=1
$$

则点积

$$
q_i^\top k_j=\sum_{r=1}^{d_k}q_{ir}k_{jr}
$$

的期望为 0，而方差近似满足：

$$
\mathrm{Var}(q_i^\top k_j)\approx d_k
$$

这意味着：维度越大，点积的典型幅度越大。若这些分数直接进入 softmax，则会出现：

- 最大值对应的概率迅速接近 1；
- 其余位置的概率迅速接近 0；
- softmax 落入饱和区；
- 梯度变小，训练不稳定。

因此引入缩放：

$$
\frac{QK^\top}{\sqrt{d_k}}
$$

其作用就是把点积分数的尺度重新拉回更适合 softmax 的范围，从而改善数值稳定性。

### softmax：从实数打分到概率权重

令：

$$
S=\frac{QK^\top}{\sqrt{d_k}}
$$

再逐行施加 softmax，得到注意力权重矩阵：

$$
A=\mathrm{softmax}(S)
$$

其中对任意第 $i$ 行有：

$$
a_{ij}=\frac{\exp(s_{ij})}{\sum_{t=1}^{L}\exp(s_{it})}
$$

并满足：

$$
\sum_{j=1}^{L} a_{ij}=1,\qquad a_{ij}\ge 0
$$

因此，第 $i$ 行就是一个离散概率分布，表示第 $i$ 个位置对整段序列所有位置分配的注意力权重。

### 与 $V$ 相乘：为什么最终是加权求和

最后一步计算：

$$
O=AV
$$

因为 $A\in\mathbb{R}^{L\times L}$，$V\in\mathbb{R}^{L\times d_v}$，所以：

$$
O\in\mathbb{R}^{L\times d_v}
$$

其中第 $i$ 行可写为：

$$
o_i=\sum_{j=1}^{L} a_{ij}v_j
$$

这说明 attention 的输出，本质上是对所有 value 向量做加权平均。权重不是固定的，而是由当前位置与各候选位置之间的相关性动态决定的。

如果想把这个过程看得更“像矩阵在流动”，可以直接观察下方交互视图。它把 $QK^\top$、加掩码、softmax 和最终的 $AV$ 输出放在同一组界面里，并允许逐行查看某个 query 是如何完成信息聚合的：

<AttentionScoreMatrixExplorer />

### 一个最小例子

设序列长度为 3，当前第 2 个位置对三个位置的打分分别为：

$$
s_{21}=0.2,\qquad s_{22}=1.0,\qquad s_{23}=2.4
$$

经 softmax 后可能得到：

$$
a_{21}=0.08,\qquad a_{22}=0.18,\qquad a_{23}=0.74
$$

于是第 2 个位置的输出就是：

$$
o_2=0.08v_1+0.18v_2+0.74v_3
$$

这意味着：第 2 个位置的最终表示主要吸收了第 3 个位置的信息，但仍保留了少量来自其余位置的内容。这就是“软选择”机制的数学体现。

---

## 掩码机制：如何在并行计算中维持时序约束

### 因果掩码的数学形式

在自回归生成中，第 $t$ 个位置的预测不能依赖未来位置 $t+1,t+2,\dots$ 的信息。若不加限制，模型在训练时会因为整段序列被并行送入而“偷看答案”。

因此，带因果掩码的 attention 写为：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

其中 $M\in\mathbb{R}^{L\times L}$ 为掩码矩阵。在因果掩码场景下：

$$
m_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

若 $L=4$，则：

$$
M=
\begin{bmatrix}
0 & -\infty & -\infty & -\infty\\
0 & 0 & -\infty & -\infty\\
0 & 0 & 0 & -\infty\\
0 & 0 & 0 & 0
\end{bmatrix}
$$

它对应的可见性结构，可以直接在上面的交互视图中切换到「因果掩码」模式来观察。此时右上角未来区域会被置为 `-inf`，随后在 softmax 阶段对应权重归零。

### 为什么加上 $M$ 就能“屏蔽未来”

令原始分数矩阵为：

$$
S=\frac{QK^\top}{\sqrt{d_k}}
$$

加入掩码后：

$$
\tilde{S}=S+M
$$

若 $j>i$，则 $\tilde{s}_{ij}=-\infty$，于是：

$$
\exp(\tilde{s}_{ij})=\exp(-\infty)=0
$$

再经过 softmax 可得：

$$
a_{ij}=0\qquad (j>i)
$$

因此，非法位置并不是在 softmax 之后“再手工清零”，而是在 softmax 之前就被排除出概率分布之外了。

### Padding Mask 与 Causal Mask 的区别

除了因果掩码，还常见 padding mask。两者作用不同：

- **causal mask**：禁止当前位置访问未来 token；
- **padding mask**：禁止模型把注意力分配给补齐出来的 `<PAD>` 位置。

前者解决的是时序约束问题，后者解决的是批量训练时无效位置污染问题。在实际系统中，两者常常需要同时使用。

### 为什么双向理解任务通常不需要 look-ahead mask

在情感分类、自然语言推断、编码型阅读理解等任务中，输入序列通常完整给出，模型目标是理解整句，而不是预测“下一个词”。此时我们反而希望当前位置能够同时参考左右两侧上下文。

因此：

- 在自回归解码器中，attention 通常是下三角可见的；
- 在双向编码器中，attention 通常是全可见的。

这也是为什么 BERT 一类模型在标准 encoder self-attention 中不使用 look-ahead mask，而更多依赖 padding mask 或其他任务相关掩码。

<MaskPositionExplorer />

---

## 与位置机制、Self-Attention 与 Transformer 的边界

从抽象层看，attention 只回答一个问题：**如何按相关性动态聚合信息。**  
它本身并不自动回答另外三个问题：

- 序列顺序如何进入模型；
- query、key、value 是否来自同一序列；
- attention 如何被组织成可深度堆叠的 block。

因此：

- 位置机制与顺序建模，应回到 [positional-encoding.md](./positional-encoding.md)；
- 序列内部的同源特化，应回到 [self-attention.md](./self-attention.md)；
- 多头、残差、LayerNorm、FFN 如何共同组成完整主干，应回到 [transformer.md](../model/transformer.md)。

---

## 多头注意力与典型扩展

### 多头注意力为什么不是简单重复

单头 attention 只在一个表示子空间中做一次匹配。多头注意力的想法是：把表示投影到多个子空间，并行执行多次 attention，再把结果拼回统一空间。

设共有 $h$ 个头，则第 $m$ 个头为：

$$
\mathrm{head}_m=\mathrm{Attention}(QW_m^Q,KW_m^K,VW_m^V)
$$

最终多头输出为：

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O
$$

其中：

- $W_m^Q,W_m^K,W_m^V$ 为第 $m$ 个头的独立投影矩阵；
- $W^O$ 用于把多头拼接结果映射回模型维度。

因此，多头的价值不在于“同样的事情做很多次”，而在于“在不同子空间中并行学习不同类型的关系模式”。

如果把这一过程画成静态图，往往只会看到「分叉再拼接」的骨架，却不容易看出每个头为什么要独立存在。下面这个交互视图会把“同源输入、独立投影、各头关注点不同、最后再拼回统一空间”的逻辑展开：

<MultiHeadAttentionExplorer />

多头属于 attention 在表达能力上的自然扩展，但它本身仍不等于完整架构。  
当 attention 进一步与位置机制、残差连接、LayerNorm、FFN 以及编码器/解码器信息流边界结合时，讨论重点就会转到 Transformer 的整体结构。

---

## 训练、推理与复杂度

Attention 的一个重要优点，是匹配分数可以通过矩阵乘法并行计算；它的典型代价，则来自相关性矩阵随长度增长而迅速膨胀。

若 query 长度为 $L_q$、key 长度为 $L_k$，则分数矩阵规模为：

$$
L_q\times L_k
$$

因此，当二者都与序列长度同阶时，标准全局 attention 往往会面临近似二次的计算与显存压力。

但需要注意：

- KV cache 是自回归 self-attention 在推理期的系统级优化，不是一般 attention 抽象本身的一部分；
- 长上下文优化、稀疏模式与高效 kernel，已经进入 [self-attention.md](./self-attention.md) 与 [transformer-extensions.md](../model/transformer-extensions.md) 的职责边界。

这些方法的共同本质，是在保持 attention 长程建模优势的同时，减少全连接匹配带来的二次代价。

如果想更直观地理解这组取舍关系，可以把它看成一个连续的权衡链条：

<AttentionTradeoffExplorer />

---

## 优势、局限与可解释性边界

### 主要优势

Attention 之所以影响深远，主要因为它同时具备以下优点：

- **长程依赖路径短**：任意两个位置可直接交互；
- **并行性强**：矩阵运算适合现代硬件；
- **表示灵活**：模型可动态组合上下文，而非被迫压缩进单一状态；
- **适用范围广**：可用于文本、语音、图像 patch、多模态输入等。

### 主要局限

但 attention 也并非没有边界：

- **长序列成本高**：全局 self-attention 常有 $O(n^2)$ 的计算与显存压力；
- **对位置机制有依赖**：纯 attention 本身缺少顺序感；
- **多头可能冗余**：不同头未必都学到真正互补的模式；
- **表现依赖数据与训练**：强大的机制不自动保证稳定泛化。

### 可解释性边界

attention 常被说成“天然可解释”，因为我们能直接观察注意力权重。但需要谨慎的是，**高权重并不自动等于严格因果贡献最大**。

更稳妥的理解应是：

- 高权重说明某位置在当前层、当前头中被更多读取；
- 但最终输出还会受到多层、多头、残差、FFN 与输出层共同影响；
- 因此，attention weight 更像信息流向线索，而不是完整的因果归因结论。

---

## 常见误区与易混概念

### Self-Attention、Masked Self-Attention 与 Cross-Attention 不是三种完全不同的机制

三者的公式骨架相同，真正差别在于：

- query 从哪里来；
- key / value 从哪里来；
- 是否受因果约束。

因此，与其把它们理解为三套不同算法，不如把它们理解为同一 attention 公式在不同输入来源与约束条件下的三种使用方式。

### 位置编码与词向量不是一回事

词向量回答“这个 token 是什么”；位置编码回答“这个 token 在哪里”。  

两者都以向量形式存在，也常直接相加，因此极易混淆。但没有词向量，模型难以表达语义；没有位置编码，模型又难以区分顺序。两者职责不同，却同样必要。

### 多头 attention 不是简单重复

多头看起来像“把 attention 做很多遍”，但真正关键的是每个头有自己独立的投影矩阵，因此它们工作在不同子空间中。  

多头的价值不在“次数更多”，而在“视角更多”。

### mask 不是推理阶段的临时补丁

因果掩码不仅在推理时防止偷看未来，在训练自回归模型时同样是定义本身的一部分。  

训练虽然可以整段并行送入，但每个位置的目标仍必须满足自回归条件分解，因此 mask 不是附属技巧，而是生成式 attention 的核心约束。

### Attention 很强，但不单独构成完整模型

Attention 是中心算子，但不是全部。一个稳定可用的 Transformer 模型通常还依赖：

- embedding 与位置机制提供输入表示；
- 残差与 LayerNorm 保证深层稳定；
- FFN 提供逐位置非线性表达；
- 输出层与训练目标把隐藏表示映射到具体任务。

因此，更准确的说法不是“Transformer = Attention”，而是“Transformer 以 Attention 为中心，围绕它组织起一整套可训练的深层结构”。

---

## 小结

从全文主线回看，Attention 的本质是一种**按相关性动态读取并重组信息**的数学机制。它把原本依赖固定长度状态压缩的序列建模方式，改写为一组可学习的矩阵运算：

1. 输入表示先投影为 $Q,K,V$；
2. 通过 $QK^\top$ 计算位置间的相关性；
3. 通过缩放与 softmax 得到可导的注意力权重；
4. 通过与 $V$ 相乘，把相关性矩阵转化为上下文化表示；
5. 再配合 mask、位置编码、多头机制、残差连接、LayerNorm 与 FFN，形成可深度堆叠的 Transformer block。

也正因为这条链路严密协作，attention 才不只是一个“补充上下文”的技巧，而是演化为现代 Transformer 与大语言模型的中心计算范式。无论是文本生成、机器翻译、信息检索，还是多模态理解，这套机制的共同本质都没有改变：**模型不是被动记忆全部历史，而是在当前任务需要下，主动、连续、可学习地选择信息并重组表示。**

## Ref

- Bahdanau, D., Cho, K., and Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.
- Luong, M.-T., Pham, H., and Manning, C. D. (2015). Effective Approaches to Attention-based Neural Machine Translation.
- Vaswani, A. et al. (2017). Attention Is All You Need.
- Shaw, P., Uszkoreit, J., and Vaswani, A. (2018). Self-Attention with Relative Position Representations.
