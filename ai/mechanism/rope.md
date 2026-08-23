# RoPE：用旋转相位编码相对位置

RoPE（Rotary Positional Embedding）不把位置向量加到输入上，而是在计算 Attention 前旋转 query 和 key。它使用绝对位置决定旋转角，却使点积只依赖两个位置的相对差。

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

真实 head 维度被分成多个二维对，每对使用不同频率，例如：

$$
\theta_r=10000^{-2r/d_{\text{head}}}
$$

高频维度对短距离变化敏感，低频维度在更长范围内缓慢旋转。实现可把每个二维对视为复数，乘以 $e^{\mathrm{i}t\theta_r}$；也可用实数的 rotate-half 操作完成相同变换。

RoPE 只作用于 $Q,K$，不旋转 $V$。它改变「怎样匹配」，不直接改变被读取的内容。

---

## 与 KV cache 的配合

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

RoPE 配置属于模型权重的解释方式。改变 base、缩放函数或位置 offset 后，即使权重张量完全相同，Attention 分数也会变化。服务端切换配置时必须让旧 KV cache 失效，并保证 Prefill 与 Decode 使用相同规则。

---

## 参考文献

- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
- Chen, S. et al. (2023). *Extending Context Window of Large Language Models via Positional Interpolation*.
