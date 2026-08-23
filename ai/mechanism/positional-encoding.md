# 位置表示：让 Attention 区分顺序与距离

Self-Attention 根据内容建立连接，但不天然区分输入行的顺序。位置表示要提供三类信息：token 在哪里、两个 token 相距多远，以及顺序变化是否应改变交互结果。

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$t$ 是绝对位置，$r$ 是正弦余弦的二维频率对索引，$d_{\text{model}}$ 是位置向量维度；$x_t,p_t,z_t$ 是 token、位置与组合后的表示；$i,j$ 是 Attention 中的两个位置，$q_i,k_j$ 是对应 Query/Key，$b(i-j)$ 是相对位置偏置。
:::

---

## 为什么内容向量不够

若没有位置项，对输入 $X$ 做任意置换 $\Pi X$，Self-Attention 输出只会按相同置换重排：

$$
\operatorname{Attn}(\Pi X)=\Pi\operatorname{Attn}(X)
$$

因此，「猫追狗」与「狗追猫」包含同样的 token 集合，纯内容交互无法确定主客体顺序。位置机制必须在输入表示、Attention 分数或 query/key 几何中打破这种对称性。

---

## 绝对位置：给每个位置一个坐标

最直接的做法把 token embedding 与位置向量相加：

$$
z_t=x_t+p_t
$$

$p_t$ 可以是可学习参数，也可以由固定函数生成。原始 Transformer 使用正弦余弦：

$$
\operatorname{PE}(t,2r)=
\sin\left(t/10000^{2r/d_{\text{model}}}\right)
$$

$$
\operatorname{PE}(t,2r+1)=
\cos\left(t/10000^{2r/d_{\text{model}}}\right)
$$

不同维度对应不同频率，使近距离和长距离变化同时可分辨。固定函数能为训练长度之外的位置生成值，但「能计算」不等于模型在这些位置上仍能正确工作。

对第 $r$ 对维度，记频率为 $\omega_r=10000^{-2r/d_{\text{model}}}$：

$$
p_t^{(r)}=[\sin(t\omega_r),\cos(t\omega_r)]
$$

位置平移 $k$ 后：

$$
p_{t+k}^{(r)}=
\begin{bmatrix}
\cos(k\omega_r) & \sin(k\omega_r)\\
-\sin(k\omega_r) & \cos(k\omega_r)
\end{bmatrix}
p_t^{(r)}
$$

因此，同一频率上的相对位移可以由一个只依赖 $k$ 的线性变换表示。这解释了正弦余弦编码为何不仅提供绝对坐标，也让线性投影有机会读取相对位移。

可学习绝对位置更自由，却通常只为预设范围保存参数。扩展长度需要初始化新位置或改变已有坐标，且 Attention 仍需从混合后的内容与位置中学习相对距离。

位置向量与 token 向量通常相加，而不是拼接。相加保持隐藏维度和残差接口不变，让后续投影共同读取内容与位置；拼接则会扩大维度，必须再投影回模型宽度。相加并不意味着两类信息不可区分，训练可以让它们分布在不同方向，但也不能保证模型一定形成完全解耦的子空间。

---

## 相对位置：直接修改两两交互

如果任务主要关心距离 $i-j$，可以把相对位置写进 Attention 分数：

$$
S_{ij}=
\frac{q_i^\top k_j}{\sqrt{d_k}}
+
b(i-j)
$$

$b$ 可以查表、按距离分桶或由函数计算。相同距离可在不同绝对位置共享参数，因此这种表示更贴近「相邻」「向前若干步」等关系。

相对偏置直接改变权重，却不一定改变被聚合的 value。某些方案还会给 key 或 value 加相对表示，以增加表达能力和计算代价。

---

## RoPE：把位置写入 Query 与 Key

[RoPE](./rope.md) 将 query、key 的成对维度按位置旋转。对位置 $m,n$：

$$
\langle R_m q,R_n k\rangle
=
\langle q,R_{n-m}k\rangle
$$

单个向量使用绝对位置角度，内积却自然依赖相对位移。它不额外添加 score 查表，并与自回归 KV cache 配合良好，因此常用于 decoder-only 模型。

RoPE 仍有训练长度边界。超出训练范围后，新的相位组合可能导致性能退化；插值与频率缩放见[长上下文位置扩展](./long-context.md)。

---

## ALiBi：用线性距离偏置表达近距先验

ALiBi 在第 $h$ 个头的分数上加入随距离增大的惩罚：

$$
S_{ij}^{(h)}
=
\frac{{q_i^{(h)}}^\top k_j^{(h)}}{\sqrt{d_k}}
-
m_h(i-j),\qquad j\le i
$$

不同头使用不同斜率 $m_h$。它不保存位置 embedding，也不会遇到绝对位置表越界；但线性近距先验会持续压低远距离连接，是否合适取决于任务与训练。

---

## 如何比较位置方案

| 方案 | 注入位置 | 主要关系 | 扩长时的直接问题 |
| --- | --- | --- | --- |
| 学习式绝对位置 | 输入 | 绝对索引 | 新位置没有训练参数 |
| 正弦余弦 | 输入 | 多频绝对坐标 | 新频率组合未被训练使用 |
| 相对位置偏置 | score | 距离或距离桶 | 分桶/函数范围与计算开销 |
| RoPE | Q、K | 旋转相位中的相对位移 | 相位分布超出训练范围 |
| ALiBi | score | 线性距离惩罚 | 远距偏置可能过强 |

选择不能只看名义最大长度。还要检查：

- 模型是双向编码还是因果生成；
- 是否依赖 KV cache；
- 任务更需要绝对位置还是相对距离；
- 训练长度与部署长度差多少；
- 在不同长度和关键证据位置上的实际准确率。

最后一项由[长上下文评估](../evaluation/long-context-evaluation.md)验证。位置机制只提供坐标或偏置，不保证模型会检索、组合并利用远处信息。

还需要确认位置索引的实际生成规则。Padding 后是否从 0 重新编号、拼接多个文档时是否连续计数、KV cache 续写从哪个 offset 开始，都会改变位置输入。训练、Prefill 和 Decode 的编号不一致，会直接破坏已缓存 key 与新 query 的相对关系。

---

## 参考文献

- Vaswani, A. et al. (2017). *Attention Is All You Need*.
- Shaw, P., Uszkoreit, J., and Vaswani, A. (2018). *Self-Attention with Relative Position Representations*.
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*.
- Press, O., Smith, N. A., and Lewis, M. (2022). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*.
