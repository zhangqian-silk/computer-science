# 推理 Attention Kernel

Attention kernel 必须实现相同的可见性和加权读取，但不必按公式书写顺序物化所有矩阵。FlashAttention 重排计算以减少 HBM 往返，PagedAttention 配合非连续 KV 寻址，Decode kernel 则面向少 Query、长历史的形状。三者不应被当成同层互斥算法。

---

## 标准实现的问题

标准 Attention 若显式物化 $S_q\times S_k$ 分数矩阵，会把大中间量写入并再次读取 HBM。分块算法在片上计算局部分数，并通过在线 Softmax 合并块，从而避免保存完整矩阵。

这类算法可以保持与精确 Attention 等价到数值容差，而不是稀疏或低秩近似。实际内存和速度仍受 shape、mask、dtype、硬件和 kernel 版本影响。

---

## 三类 kernel

| Kernel 形态 | 主要输入特征 | 主要挑战 |
| --- | --- | --- |
| Prefill/Flash Attention | $S_q$ 较大、QKV 连续 | tile、在线 Softmax、前后向 |
| Decode Attention | 每序列少量 query、KV 很长 | KV 读带宽、跨序列并行 |
| Paged Attention | KV 通过 block table 间接寻址 | 不连续访问、尾 block、共享前缀 |

GQA/MLA、滑动窗口、ALiBi/RoPE、KV 量化和多种 mask 都会扩大 kernel 变体。框架可能在不同 shape 间 dispatch 到多个后端，不能用一个 microbenchmark 代表全部请求。

---

## 在线 Softmax 合并

::: info 符号与约定
对单个 Query，$s_j$ 为已包含 mask 的有限 score，$v_j$ 为 Value。对一个非空有效分块维护最大值 $m$、重标定指数和 $l$，以及未归一化向量和 $u=\sum_j e^{s_j-m}v_j$。
:::

对每个分块维护当前最大值 $m$ 和指数和 $l$。新块最大值为 $m_b$、指数和为 $l_b$ 时：

$$
m'=\max(m,m_b)
$$

$$
l'=e^{m-m'}l+e^{m_b-m'}l_b
$$

输出分子必须用同样尺度合并，不能直接平均各块已经归一化的输出：

$$
u'=e^{m-m'}u+e^{m_b-m'}u_b,\qquad o=u'/l'
$$

例如两个单元素块的 score 分别为 0、$\log3$，Value 为 0、4，正确全局输出为 $(1\times0+3\times4)/(1+3)=3$。两个块各自 softmax 后输出是 0、4，直接平均得 2，丢失了块之间的概率质量。

当所有块处理完成，得到与完整 softmax 相同的实数公式；浮点重排仍可能产生小误差。被 mask 后全无效的块应跳过，而整行无有效 Key 时须按接口定义处理，不能让 $-\infty-(-\infty)$ 混入普通计算。

原始 FlashAttention 的关键是 IO-aware 分块与反向重算，FlashAttention-2 继续改进工作划分与并行效率；两者没有把标准 Attention 的数学连接数从二次改成线性。这里减少的首先是中间状态搬运与保存。

---

## 验收矩阵

Attention kernel 至少验证：

- prefill 与 decode；
- MHA、GQA/必要时 MLA；
- causal、padding、sliding window；
- 连续与 paged KV；
- 不同 head dim、长度和 block 尾部；
- FP16/BF16/目标 KV dtype；
- 与参考实现的输出和必要时梯度容差；
- 性能 profile 与端到端收益。

---

## CPU 路线

CPU 可以实现朴素 Attention、分块在线 Softmax 和 paged gather，验证数值等价、mask 与 block table。可用 cache blocking 观察数据复用趋势。GPU fused kernel、Shared Memory 和 HBM 收益必须在目标加速器上测量。

---

## 参考文献

- Dao, T. et al. (2022). [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135).
- Dao, T. (2023). [*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*](https://arxiv.org/abs/2307.08691).
- Kwon, W. et al. (2023). [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180).
