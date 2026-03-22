# RoPE：把位置信息写入 Query / Key 旋转相位

> 相关文献：
> - Su et al. (2021)：提出 RoPE，将位置编码写入 query / key 的旋转结构。
> - Press, Smith, and Lewis (2022)：ALiBi 提供了与 RoPE 明显不同的长上下文位置建模路线。
> - Chen et al. (2023)、Peng et al. (2023)、Ding et al. (2024)：围绕 RoPE 的长度扩展、插值与缩放方法持续发展。

本文主要负责说明 RoPE（Rotary Positional Embedding）的数学机制、几何直觉、工程实现与 KV cache 友好性。若希望先建立位置机制的总览地图，应先阅读 [Positional Encoding](./positional-encoding.md)；若希望进一步理解基于 RoPE 的长度外推与长上下文扩展，应继续阅读 [Long Context Position](./long-context-position.md)。

---

## 问题背景

经典绝对位置编码通常把位置向量直接加到输入表示中：

$$
z_t = x_t + p_t
$$

这种方式能够告诉模型“你在第几个位置”，但 attention 在真正计算 query / key 内积时，仍需自己从输入表示中再提取距离信息。RoPE 的关键改写在于：**不再先把位置写到输入上，而是直接把位置写到 query / key 的几何结构里。**

从历史上看，RoPE 可以理解为对正弦余弦位置编码的一次“内化”：

- 绝对位置编码把旋转结构放在输入层；
- RoPE 把旋转结构推进到 attention 内部；
- 于是相对位移会在 query / key 的点积中自然出现。

---

## 符号约定与核心公式

| 符号 | 含义 |
| --- | --- |
| $q_t,k_t \in \mathbb{R}^{d}$ | 位置 $t$ 的 query / key 向量 |
| $q_t^{(i)}, k_t^{(i)}$ | 第 $i$ 个二维子空间中的 query / key 子向量 |
| $\theta_i$ | 第 $i$ 个二维子空间对应的基础频率 |
| $R_{\theta_i,t}$ | 位置 $t$ 在第 $i$ 个子空间中的旋转矩阵 |
| $\tilde{q}_t,\tilde{k}_t$ | 施加 RoPE 后的 query / key |
| $m,n$ | 两个位置索引 |

RoPE 的核心公式可以压缩为 4 步：

1. 第 $i$ 个二维子空间的旋转矩阵：
$$
R_{\theta_i,t}=
\begin{bmatrix}
\cos(t\theta_i) & -\sin(t\theta_i)\\
\sin(t\theta_i) & \cos(t\theta_i)
\end{bmatrix}
$$

2. 对 query / key 做位置相关旋转：
$$
\tilde{q}_t^{(i)} = R_{\theta_i,t} q_t^{(i)},\qquad
\tilde{k}_t^{(i)} = R_{\theta_i,t} k_t^{(i)}
$$

3. 旋转后的 attention 打分：
$$
\mathrm{Score}(m,n)=\frac{\tilde{q}_m^\top \tilde{k}_n}{\sqrt{d_k}}
$$

4. 相对位移在内积中自动出现：
$$
\langle R_{\theta_i,m} q,\ R_{\theta_i,n} k\rangle
=
\langle q,\ R_{\theta_i,n-m} k\rangle
$$

最后这条式子是 RoPE 的关键。它说明 RoPE 虽然在编码阶段使用绝对位置 $m,n$，但在打分阶段真正起作用的是位置差 $n-m$。

---

## 核心直觉：为什么旋转能够表达位置

RoPE 的一个二维子空间可以被理解为平面上的一根箭头。若位置为 $t$，则这个箭头会被旋转 $t\theta_i$ 的角度。

于是：

- 同一个内容向量，位于不同位置时会指向不同方向；
- 两个位置做内积时，真正重要的是它们之间的夹角差；
- 这个夹角差恰好对应相对位移。

这意味着 RoPE 并不是“额外加一个位置向量”，而是让向量本身因为所处位置不同而具有不同相位。

可以把它和几类位置机制做一个压缩对比：

| 机制 | 位置如何进入模型 | 相对距离如何体现 |
| --- | --- | --- |
| 绝对位置编码 | 位置向量加到输入上 | 由后续层间接学习 |
| 相对位置偏置 | 距离项直接加到 score 上 | 显式写入 |
| RoPE | 对 $Q,K$ 做旋转 | 在内积中自动出现 |

---

## 从二维旋转到多维向量

真实模型中的 query / key 维度通常远大于 2。RoPE 的做法是把最后一维切成若干二维块，每个二维块各自应用一个不同频率的旋转。

设：

$$
q_t = [q_t^{(1)}, q_t^{(2)}, \dots, q_t^{(d/2)}]
$$

那么 RoPE 实际上是在不同频率的二维平面中并行旋转。这样做有两个好处：

- 高频维度更擅长区分近邻 token 的细粒度顺序；
- 低频维度更擅长表达更大跨度的全局位置趋势。

这和正弦余弦位置编码中的“多频率设计”是一脉相承的，只不过 RoPE 把这种结构直接写进了 attention 的内部表示。

---

## 最小例子

设在某个二维子空间中：

- 位置 $m$ 的 query 被旋转了 $m\theta$；
- 位置 $n$ 的 key 被旋转了 $n\theta$。

若两者相差角度较小，则内积往往较大；若相差角度较大，则内积往往较小。于是：

- 位置越接近，某些频段上越容易保持较强匹配；
- 位置越远，角度差越大，部分频段的匹配就会衰减。

因此，RoPE 的位置感不是通过单独查表获得的，而是通过向量间的相位关系自然产生的。

---

## 复数实现

RoPE 在工程上常被写成复数乘法。设第 $i$ 个二维子空间中的实数对写为：

$$
q_i = q_{2i} + j q_{2i+1},\qquad
k_i = k_{2i} + j k_{2i+1}
$$

则位置 $m$ 上的旋转可写成：

$$
\tilde{q}_i^{(m)} = q_i^{(m)} e^{j m \omega_i},\qquad
\tilde{k}_i^{(m)} = k_i^{(m)} e^{j m \omega_i}
$$

这里的好处是：

- 二维旋转可以统一写成逐元素复数乘法；
- 实现上通常只需一次形状重排和逐元素运算；
- 对现有 attention 框架的侵入性较低。

因此，RoPE 很适合在现代张量框架中高效实现。

---

## KV Cache 友好性

RoPE 在大语言模型中广泛使用的一个关键原因，是它与自回归推理和 KV cache 非常兼容。

设第 $m$ 个 token 的线性投影结果为：

$$
q^{(m)} = x^{(m)}W_Q,\qquad
k^{(m)} = x^{(m)}W_K,\qquad
v^{(m)} = x^{(m)}W_V
$$

RoPE 只作用在 query 和 key 上：

$$
\tilde{q}^{(m)} = \mathrm{RoPE}(q^{(m)}, m),\qquad
\tilde{k}^{(m)} = \mathrm{RoPE}(k^{(m)}, m)
$$

此时历史 key 一旦旋转并写入 cache，就不必因为未来新 token 到来而重写。原因在于，相对位移已经在后续点积中自动体现出来。

这带来两个工程优势：

- 历史 key 可直接缓存复用；
- 不需要在每一步重新构造显式距离偏置或重编码旧位置。

需要注意的是，RoPE 并没有把全局 attention 的整体复杂度从 $O(n^2)$ 降为更低量级；它改善的是位置编码部分的实现负担与缓存兼容性，而不是取消两两交互本身。

---

## 为什么现代开源 LLM 偏好 RoPE

RoPE 成为现代开源 LLM 默认起点，通常来自以下几方面折中：

- 与 attention 内积结构耦合紧密；
- 能同时保留绝对位置信息与相对位移效应；
- 与 KV cache 兼容性好；
- 工程实现简洁；
- 后续还能通过插值、缩放等方式继续做长上下文扩展。

更准确地说，RoPE 并不是在每个维度上都绝对最优，而是在表达能力、长上下文潜力和工程复杂度之间取得了很均衡的平衡。

---

## 与其他位置机制的关系

可以把 RoPE 放在一条更清晰的演化链里理解：

- 绝对位置编码：先给每个位置分配坐标；
- 相对位置编码：直接在 attention 中写入距离项；
- RoPE：编码时使用绝对位置旋转，打分时自动呈现相对位移。

这说明 RoPE 既不是传统绝对位置编码的简单重复，也不是一般相对位置偏置的直接替代，而是一种把两者连接起来的结构化方法。

---

## 小结

RoPE 的最关键思想，是把位置信息从输入层推进到 query / key 的几何关系中。这样一来：

- 单个向量携带绝对位置相位；
- 两两打分自然体现相对位移；
- 工程实现上又保持了较好的轻量性与缓存友好性。

因此，RoPE 才会成为现代大语言模型中最常见的位置机制之一。

---

## Ref

- Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- Press, O., Smith, N. A., and Lewis, M. (2022). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.
- Chen, S., Wong, S., Chen, L., and Tian, Y. (2023). Extending Context Window of Large Language Models via Positional Interpolation.
- Peng, B., Quesnelle, J., Fan, H., and Shippole, E. (2023). YaRN: Efficient Context Window Extension of Large Language Models.
- Ding, Y. et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.
