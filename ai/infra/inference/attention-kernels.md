# 推理 Attention Kernel

Attention 的数学结果由 $Q,K,V$、mask、位置机制和 Softmax 定义。FlashAttention、PagedAttention 和 Decode Attention 是不同执行问题：前者减少中间数据往返，Paged 版本处理非连续 KV，Decode 版本针对少量 query 读取长历史。

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

对每个分块维护当前最大值 $m$ 和指数和 $l$。新块最大值为 $m_b$、指数和为 $l_b$ 时：

$$
m'=\max(m,m_b)
$$

$$
l'=e^{m-m'}l+e^{m_b-m'}l_b
$$

输出累加也按相同 scale 重标定。这样可以任意顺序处理 K/V 块而不保存全部分数。实现仍需处理 mask 后全无效行、低精度累加和极端 logits。

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

## 参考文献

- Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.
- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*.
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*.
