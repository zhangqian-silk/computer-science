# Self-Attention：让序列位置直接交换信息

Self-Attention 是 Attention 的一个输入约束：query、key、value 都由同一序列产生。每个位置因此可以按内容读取其他位置，并把静态输入改写为上下文化表示。

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$X\in\mathbb{R}^{n\times d_{\text{model}}}$ 是输入序列表示，$Q,K,V$ 是同一 $X$ 的三组投影，$M$ 是可见性 mask，$H$ 是输出；$i,j$ 是 Query 与 Key 位置，$t$ 是自回归步，$K_{\leq t},V_{\leq t}$ 表示缓存到当前步的 Key/Value。
:::

---

## 从输入序列到交互矩阵

给定：

$$
X\in\mathbb{R}^{n\times d_{\text{model}}}
$$

三组投影产生：

$$
Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V
$$

输出为：

$$
H=\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
$$

$QK^\top\in\mathbb{R}^{n\times n}$ 可以看成一张有向加权图：第 $i$ 行的边描述位置 $i$ 从每个位置读取多少信息。同一个词在不同句子中会形成不同的边，因此输出 $h_i$ 依赖整段上下文。

同一输入 $X$ 产生三种角色，并不表示 $Q=K=V$。三组投影分别学习：当前位置需要检索什么、每个位置以什么特征接受匹配、以及被选中后提供什么内容。某个 token 可以在 key 空间与 query 高度匹配，却通过 value 投影提供完全不同的特征。

Self-Attention 本身对位置排列具有等变性：若输入行以同一置换重排，输出也只会随之重排。模型必须结合[位置表示](./positional-encoding.md)才能区分「狗追猫」与「猫追狗」。

---

## 可见性由 mask 定义

双向 Encoder 通常只屏蔽 padding；自回归 Decoder 还需要因果掩码：

$$
M_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

这样第 $i$ 个位置只能读取自己和历史前缀。训练时整段序列仍可一次送入矩阵运算，但因果 mask 保证每个位置没有使用未来 token。

| 使用位置 | Q / K / V 来源 | 可见范围 |
| --- | --- | --- |
| Encoder Self-Attention | 同一输入序列 | 通常双向 |
| Decoder Self-Attention | 同一目标序列 | 当前与过去 |
| Cross-Attention | Q 来自 Decoder，K/V 来自 Encoder | 全部条件位置 |

最后一行不是 Self-Attention，只是共享同一个 Attention 公式。

长度为 4 的因果 mask 对应如下可见性：

| Query 位置 | 可读取的 Key 位置 |
| ---: | --- |
| 1 | 1 |
| 2 | 1、2 |
| 3 | 1、2、3 |
| 4 | 1、2、3、4 |

训练位置 3 的输出时，位置 4 虽然位于同一个输入张量中，仍会因 $-\infty$ mask 获得 0 权重。Padding mask 则按样本屏蔽补齐位置；因果 mask 与 padding mask 通常需要合并应用。

---

## 多层之后信息能传播多远

全局 Self-Attention 在一层内连接任意两个位置，依赖路径短。局部或稀疏模式只连接部分位置，但多层叠加会扩大感受野。例如窗口半径为 $w$ 时，忽略边界后，$L$ 层可传播约 $Lw$ 的局部距离。

这说明 mask 不只控制计算量，也定义模型可表达的信息路径。具体图结构与连通性见[稀疏注意力](./sparse-attention.md)。

---

## 训练并行与自回归推理

训练时所有位置的 $Q,K,V$ 可以并行计算。自回归推理却必须逐 token 生成；若每一步重算整个前缀，会重复计算历史 key 和 value。

KV cache 保存每层已经旋转并投影后的历史 $K,V$。新 token 到来时只计算新增项，再让新 query 读取缓存：

$$
K_{\le t}=[K_{<t};k_t],\qquad
V_{\le t}=[V_{<t};v_t]
$$

缓存降低重复计算，但其容量随层数、序列长度、KV 头数和头维度线性增长。Multi-Query Attention 与 Grouped-Query Attention 通过让多个 query 头共享较少的 KV 头来减少缓存；它们改变头的参数组织，不改变因果读取语义。

Encoder 推理通常不使用这种自回归 KV cache：完整输入已知，一次前向需要同时更新所有 token 的双向上下文。Decoder Decode 阶段只有新 token 的 query 需要输出，历史位置的 key/value 不再变化，缓存才成立。修改历史 token、位置缩放规则或模型权重后，已有 cache 也必须失效。

---

## 二次连接成本

全局 Self-Attention 的分数矩阵有 $n^2$ 个元素。常用复杂度写为：

$$
O(n^2d)
$$

这个表达强调随长度增长的主导连接代价。实际速度还取决于 head 维度、batch、kernel、显存带宽以及是否需要保存注意力矩阵。FlashAttention 通过改变分块与访存方式计算相同结果，并没有把全局连接数改成线性。

长序列优化因此分成三类：

- 保持数学结果，优化 kernel 和内存访问；
- 限制连接图，使用局部或[稀疏注意力](./sparse-attention.md)；
- 改用不同序列主干，例如[状态空间模型](../model/state-space-model.md)。

---

## 参考文献

- Vaswani, A. et al. (2017). *Attention Is All You Need*.
- Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
- Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need*.
