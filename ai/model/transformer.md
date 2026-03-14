# Transformer：以自注意力为核心的序列建模架构

> 相关文献：
> - Vaswani et al. (2017)：提出 Transformer，使用 self-attention 取代 RNN 成为序列建模主干。
> - Devlin et al. (2019)：以 BERT 展示 Transformer 编码器在表示学习上的强大能力。
> - Radford et al. (2018, 2019) 与 Brown et al. (2020)：以 GPT 系列表明 Transformer 解码器可扩展为大规模生成模型。
> - Touvron et al. (2023) 等：展示 Transformer 在大模型训练、推理与部署中的持续扩展路径。

## 符号约定与核心公式

本文统一使用以下记号：

| 符号 | 含义 |
| --- | --- |
| $X\in\mathbb{R}^{n\times d}$ | 输入序列表示矩阵，长度为 $n$，维度为 $d$ |
| $x_t\in\mathbb{R}^d$ | 第 $t$ 个位置的输入表示 |
| $p_t\in\mathbb{R}^d$ | 第 $t$ 个位置的位置编码 |
| $Q,K,V$ | query、key、value 矩阵 |
| $d_k$ | attention 中 key / query 的维度 |
| $h$ | 多头 attention 的头数 |
| $W_O$ | 多头拼接后的输出投影矩阵 |
| $\mathrm{FFN}(\cdot)$ | 位置前馈网络 |
| $\mathrm{LN}(\cdot)$ | Layer Normalization |
| $M$ | 掩码矩阵，如 padding mask 或 causal mask |
| $Y$ | Transformer 层的输出表示 |

本文核心公式索引：

1. 输入与位置编码结合：
$$
z_t = x_t + p_t
$$

2. Transformer 原始缩放点积 attention：
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

3. 多头 attention：
$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W_O
$$

4. 前馈网络：
$$
\mathrm{FFN}(x)=W_2\,\sigma(W_1 x+b_1)+b_2
$$

5. 解码器中的带掩码 attention：
$$
\mathrm{MaskedAttention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

6. 一个 Transformer block 的典型形式：
$$
Y=\mathrm{LN}\bigl(X+\mathrm{MultiHead}(XW_Q,XW_K,XW_V)\bigr),\quad
Z=\mathrm{LN}\bigl(Y+\mathrm{FFN}(Y)\bigr)
$$

---

## 模型定义与全局交互假设

Transformer 是一种以 **self-attention 为核心计算单元** 的神经网络架构，用于处理序列数据。它最初为机器翻译提出，但很快扩展到语言建模、信息抽取、语音、视觉、多模态等广泛任务中。

它相对 RNN、LSTM 的根本变化在于：模型不再依赖递归状态沿时间逐步传递信息，而是让序列中每个位置通过 attention 与其他位置直接交互。这样做的结果是：

- 长距离依赖路径显著缩短；
- 训练可以高度并行；
- 表示更新从“时间递推”转向“全局交互”。

Transformer 通常建立在以下建模假设之上：

- **全局依赖可直接建模假设**：序列中任意两个位置之间的关系，都应允许在同一层内直接建立；
- **内容相关选择假设**：模型应根据内容动态决定关注哪些位置，而不是固定窗口或固定递归路径；
- **位置需显式注入假设**：由于 self-attention 本身不包含顺序结构，必须额外加入位置编码；
- **层次组合假设**：通过多层 attention 与前馈网络叠加，模型可以逐层构造更高阶的语义表示。

例如，在句子「The animal didn't cross the street because it was too tired」中，理解「it」指代谁，需要同时结合远处实体、局部否定结构与谓词语义。Transformer 的设计目标，就是让这种跨位置关系能被高效、并行地建模。

---

## Self-Attention 主干与结构组织

Transformer 的核心由以下几个部分组成：

- token embedding；
- positional encoding；
- multi-head self-attention；
- position-wise feed-forward network；
- residual connection 与 layer normalization。

### 输入表示：内容与位置的结合

输入 token 先被映射到 embedding 向量 $x_t$，再与位置编码 $p_t$ 相加：

$$
z_t = x_t + p_t
$$

这一步的作用是同时保留：

- token 本身是什么；
- 它位于序列中的什么位置。

如果没有位置编码，Transformer 很难区分相同 token 集合的不同排列。

### 多头 self-attention

设输入矩阵为 $X$，先做线性投影得到：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

然后计算缩放点积 attention。若直接对照 Vaswani et al. (2017)，原始公式不含掩码项：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中：

- $QK^\top$ 给出任意两个位置之间的匹配分数；
- softmax 把分数归一化为注意力权重；
- 与 $V$ 相乘后，每个位置得到对整段序列信息的加权聚合。

多头机制则把这一过程并行做 $h$ 次：

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W_O
$$

在解码器中，为保持自回归约束，还会额外加入因果掩码：

$$
\mathrm{MaskedAttention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

不同头可以在不同表示子空间中学习不同关系，例如：

- 局部搭配；
- 远距离依赖；
- 句法结构；
- 指代或对齐关系。

### 前馈网络

Attention 负责跨位置交互，而前馈网络负责对每个位置的表示做进一步非线性变换：

$$
\mathrm{FFN}(x)=W_2\,\sigma(W_1 x+b_1)+b_2
$$

它的特点是：

- 对每个位置独立应用；
- 参数在所有位置共享；
- 提升表示变换能力。

因此，一个 Transformer block 可以理解为：

- attention：在位置之间交换信息；
- FFN：在单个位置内重组特征。

### 残差连接与归一化

Transformer 每个子层通常都配有残差连接与层归一化：

$$
Y=\mathrm{LN}\bigl(X+\mathrm{MultiHead}(X)\bigr),\quad
Z=\mathrm{LN}\bigl(Y+\mathrm{FFN}(Y)\bigr)
$$

这两部分的作用分别是：

- **残差连接**：缓解深层训练困难，保留原始信息通路；
- **层归一化**：稳定数值分布，改善优化过程。

这类设计使 Transformer 能够堆叠很多层，而不至于训练过早失稳。

### 编码器、解码器与两种主流用法

原始 Transformer 包含编码器与解码器两部分：

| 组件 | 核心结构 | 作用 |
| --- | --- | --- |
| **Encoder** | self-attention + FFN | 将输入序列编码为上下文化表示 |
| **Decoder** | masked self-attention + cross-attention + FFN | 在生成时读取历史输出并结合编码器信息 |

后来，Transformer 逐渐演化出三种主流变体：

| 变体 | 结构特征 | 代表用途 |
| --- | --- | --- |
| **Encoder-only** | 仅保留编码器 | BERT、分类、检索、理解任务 |
| **Decoder-only** | 仅保留解码器 | GPT、自回归生成 |
| **Encoder-Decoder** | 保留完整结构 | 翻译、摘要、条件生成 |

---

## 训练过程与推理方式

Transformer 的训练与推理方式，与具体变体密切相关，但总体流程具有一致框架。

### 训练阶段

典型训练过程如下：

1. 将输入 token 转换为 embedding，并叠加位置编码；
2. 多层堆叠 attention 与 FFN，对输入表示逐层更新；
3. 根据任务类型连接输出头，如分类头或语言模型头；
4. 计算损失函数；
5. 通过反向传播更新全部参数。

若是语言模型训练，目标通常是预测下一个 token；若是编码器预训练，目标可能包括掩码语言建模、句子关系判断或对比学习。

### 推理阶段

不同 Transformer 变体在推理阶段差异明显：

| 变体 | 推理方式 | 典型特点 |
| --- | --- | --- |
| **Encoder-only** | 一次性编码整段输入 | 并行性强，适合理解任务 |
| **Decoder-only** | 自回归逐 token 生成 | 依赖 causal mask 与 KV cache |
| **Encoder-Decoder** | 先编码源序列，再逐步解码 | 适合条件生成 |

在生成任务中，推理时通常采用：

- Greedy decoding；
- Beam search；
- Top-k / Top-p sampling；
- 温度调节等策略。

这些策略不改变 Transformer 主干结构，但决定模型如何从预测分布中产生最终输出。

### 工程上的关键问题

Transformer 在训练和推理中都高度依赖矩阵并行，但也带来若干工程挑战：

- self-attention 的时间与空间代价通常随 $n^2$ 增长；
- 长上下文训练对显存与带宽要求高；
- 自回归推理的延迟会随生成长度累积；
- 大模型下参数规模、缓存大小和通信成本都成为瓶颈。

因此，现代 Transformer 系统通常需要结合：

- 混合精度训练；
- 张量并行、流水并行；
- KV cache；
- 高效 attention 实现；
- 更长上下文的 RoPE 或相对位置机制。

---

## 最小推演：信息如何跨位置流动

考虑一个极简翻译片段：

$$
[\text{I},\ \text{love},\ \text{you}]
$$

目标是生成中文：

$$
[\text{我},\ \text{爱},\ \text{你}]
$$

在 encoder-decoder Transformer 中，其最小推演过程可以概括如下：

| 步骤 | 当前模块 | 内部在做什么 | 结果 |
| --- | --- | --- | --- |
| 1 | Encoder 输入层 | 将 `I / love / you` 映射为 embedding，并加位置编码 | 每个源词有了内容与位置信息 |
| 2 | Encoder self-attention | 每个源词与整句其他词交互 | 得到上下文化源端表示 |
| 3 | Decoder 输入层 | 输入已生成前缀，如「我」 | 作为当前解码条件 |
| 4 | Decoder masked self-attention | 只看历史输出，不能看未来 | 维持自回归约束 |
| 5 | Decoder cross-attention | 当前解码状态读取 encoder 输出 | 重点对齐 `love` 或 `you` 等源词 |
| 6 | 输出层 | 生成下一个词分布 | 更容易预测「爱」或「你」 |

这个例子说明 Transformer 的关键不是“只有 attention”，而是几类机制的组合：

- encoder self-attention 负责理解源句内部关系；
- decoder self-attention 负责维护目标端历史；
- cross-attention 负责把源端信息条件性注入目标端；
- FFN、残差、归一化负责稳定而高效地堆叠深层表示。

---

## 优势、局限与工程要点

Transformer 之所以成为主流架构，核心原因在于它同时兼顾了表达能力、并行性与可扩展性。

它的主要优势包括：

- **长程依赖建模能力强**：任意两个位置可直接交互；
- **训练并行性强**：相比 RNN，更适合现代加速硬件；
- **模块化程度高**：embedding、attention、FFN、位置机制可灵活替换；
- **扩展性好**：从中小模型到超大模型都能沿同一路线放大；
- **跨模态适配能力强**：文本、图像、音频、蛋白质序列都能映射为 token 序列处理。

但它也存在清晰局限：

- **self-attention 通常为二次复杂度**：长序列时代价高；
- **归纳偏置较弱**：相较 CNN 和 RNN，更依赖数据规模学到结构；
- **对训练资源要求高**：性能很大程度依赖大数据、大算力与精细工程；
- **推理成本不低**：特别是在长上下文和大模型场景下。

训练难点主要包括：

- 深层网络训练稳定性；
- 长序列显存与速度瓶颈；
- 大词表或大输出空间的 softmax 成本；
- 自回归生成中的延迟与缓存膨胀问题。

常见缓解手段包括：

- 改进归一化与初始化策略；
- 使用高效 attention、稀疏 attention、线性 attention；
- 采用参数共享、蒸馏、量化、低秩适配等手段；
- 在推理时结合 KV cache、speculative decoding 等优化。

---

## 历史位置与相关模型

Transformer 的历史位置极其关键：它不是在已有序列模型上做局部修补，而是重新定义了序列建模的主干范式。

在它之前：

- N-gram 主要依赖固定窗口；
- RNN / LSTM 依赖递归状态传递；
- Attention 多数仍作为 Seq2Seq 中的辅助对齐模块存在。

Transformer 的突破在于：直接把 self-attention 放到中心位置，使“全局内容相关交互”成为主干，而不是附属机制。这带来了明显代际差异：

| 模型 | 上下文建模方式 | 长依赖路径 | 并行性 | 核心问题 |
| --- | --- | --- | --- | --- |
| **N-gram** | 固定窗口 | 短 | 强 | 稀疏，视野有限 |
| **RNN** | 递归状态传递 | 长 | 弱 | 长依赖困难 |
| **LSTM / GRU** | 门控递归状态 | 较长 | 弱 | 仍受串行限制 |
| **Seq2Seq + Attention** | 递归主干 + 动态读取 | 中 | 中 | 仍以 RNN 为主干 |
| **Transformer** | self-attention 全局交互 | 极短 | 强 | 长序列二次复杂度 |

在它之后，模型演化主要沿两条路线展开：

- **能力扩展**：更大参数规模、更长上下文、更强多模态融合；
- **效率改进**：稀疏化、线性化、缓存优化、硬件友好实现。

因此，Transformer 的历史意义不仅是“效果好”，更在于它把深度学习中的序列建模，从递归时代推进到了 attention 时代。

---

## 总结

Transformer 的最关键思想，是用 self-attention 替代递归状态传递，让序列中每个位置都能直接、并行地与其他位置交互，再配合前馈网络逐层构造上下文化表示。

在模型演化史上，它继承了 attention 的动态信息读取思想，但把这一机制从辅助模块提升为架构核心。后续的 BERT、GPT、T5 以及大量视觉、多模态与科学计算模型，基本都可以看作在 Transformer 框架上的延伸、裁剪或扩展。
