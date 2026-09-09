# RoPE：用旋转相位编码相对位置

RoPE 用位置决定 Query/Key 的旋转，使点积中的位置因子只依赖相对位移。点积仍依赖内容，不是只看距离的函数；这个限定是理解几何图与真实模型区别的关键。

<PositionEncodingExplorer initial-method="rope" />

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$t,m,n$ 是位置，$r$ 是二维频率对索引，$\theta_r$ 是对应角频率，$R_r(t)$ 是位置 $t$ 的旋转矩阵；$q,k$ 是未旋转 Query/Key，$\tilde{q},\tilde{k}$ 是旋转结果，$d_{\text{head}}$ 是单个 Attention head 的维度。
:::

---

## 二维旋转

对第 $r$ 个二维子空间，位置 $t$ 的旋转矩阵为：

$$
R_r(t)=
\begin{bmatrix}
\cos(t\theta_r) & -\sin(t\theta_r)\\
\sin(t\theta_r) & \cos(t\theta_r)
\end{bmatrix}
$$

query 与 key 分别变为：

$$
\tilde{q}_t^{(r)}=R_r(t)q_t^{(r)},\qquad
\tilde{k}_t^{(r)}=R_r(t)k_t^{(r)}
$$

旋转矩阵满足 $R_r(m)^\top R_r(n)=R_r(n-m)$，所以：

$$
\langle R_r(m)q,R_r(n)k\rangle
=
\langle q,R_r(n-m)k\rangle
$$

这条等式是 RoPE 的核心：Attention 分数通过相位差获得相对位移，而不需要额外的距离查表。

### 一个二维相位例子

为了直观看旋转，取频率 $\theta=\pi/2$，并令未旋转的 $q=k=[1,0]$：

- 两者都在位置 0 时，点积为 $1$；
- query 在位置 0、key 在位置 1 时，$R(1)k=[0,1]$，点积为 $0$；
- query 在位置 0、key 在位置 2 时，$R(2)k=[-1,0]$，点积为 $-1$；
- 两者都在位置 1 时，它们同时旋转为 $[0,1]$，点积仍为 $1$。

共同平移不改变点积，位置差变化才改变匹配结果。真实 RoPE 同时使用许多频率，不会只靠一个二维周期区分所有距离。

---

## 多频率如何覆盖不同尺度

一般 $q=(q_1,q_2)$、$k=(k_1,k_2)$ 时，令 $\delta=(n-m)\theta$：

下式 $R_{\text{angle}}$ 直接以角度为参数，区别于前文以位置为参数的 $R_r(t)$。

$$
q^\top R_{\text{angle}}(\delta)k
=(q_1k_1+q_2k_2)\cos\delta
+(q_2k_1-q_1k_2)\sin\delta
$$

同方向单位向量才简化为 $\cos\delta$。真实得分不保证随距离单调下降；共同平移只保持位置因子，内容若因上下文改变，得分仍可能变化。

真实 head 维度被分成多个二维对，每对使用不同频率，例如：

$$
\theta_r=10000^{-2r/d_{\text{head}}}
$$

高频维度对短距离变化敏感，低频维度在更长范围内缓慢旋转。实现可把每个二维对视为复数，乘以 $e^{\mathrm{i}t\theta_r}$；也可用实数的 rotate-half 操作完成相同变换。

本文采用通常用于语言模型的 Q/K-only RoPE，不旋转 V；其他变体需单独定义。它改变匹配几何，不直接旋转被读取的内容。

---

## 与 KV cache 的配合

二维配对布局也属于权重解释。相邻成对与前后半区配对可通过相应参数排列对应，不能只换 rotate-half 实现却保持权重排列不变。相位正确、shape 正确仍不能排除配对错误。

自回归推理时，位置 $t$ 的 key 一旦完成旋转即可写入 cache。未来 query 位于位置 $m$ 时：

$$
\tilde{q}_m^\top\tilde{k}_t
=
q_m^\top R(t-m)k_t
$$

相对位移在点积中自动产生，不需要重写历史 key。实现必须确保训练、prefill 和逐 token decode 使用同一位置索引与缩放规则，否则缓存中的相位不兼容。

RoPE 不降低全局 Attention 的 $O(n^2)$ 连接代价；它只提供位置结构和缓存友好的实现接口。

---

## 长度边界

当推理位置远超训练范围时，模型会遇到未训练过的相位组合。不同频率的周期还会造成分辨率与外推稳定性不一致。常见扩展方法包括位置插值、频率分段缩放与额外长序列微调，详见[长上下文位置扩展](./long-context.md)。

判断扩展是否有效必须测量不同长度和证据位置上的任务表现，而不能只检查位置索引是否还能计算。

改变 base 或缩放规则通常改变得分；纯 RoPE 下所有 Q/K 共同平移相同 offset 是相对位移不变的例外。若只改新 Query 的 offset 而旧 K 保持原相位，则会失配；需重算或按已证明等价的规则转换缓存，保持两阶段一致。

---

## 参考文献

- Su, J. et al. (2021). [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864).
- Chen, S. et al. (2023). [*Extending Context Window of Large Language Models via Positional Interpolation*](https://arxiv.org/abs/2306.15595).
