# MHA、MQA、GQA 与 MLA

Attention 头共享方案改变 Query 头如何复用 Key/Value 表示。它们的主要系统差异集中在 KV 投影、KV cache 容量、Decode 带宽和并行切分，而不是 Attention 的概率读取接口本身。

<HeadSharingExplorer />

::: info 符号与约定
$N_q$ 是 Query 头数，$N_{kv}$ 是 Key/Value 头数，$D$ 是单头维度，$S$ 是缓存序列长度。
:::

---

## 四种结构

| 结构 | 头关系 | KV cache 规模特征 | 主要约束 |
| --- | --- | --- | --- |
| MHA | $N_{kv}=N_q$ | 最大 | 每个 Query 头有独立 KV 头 |
| MQA | $N_{kv}=1$ | 最小 | 所有 Query 头共享一组 KV |
| GQA | $1<N_{kv}<N_q$ | 介于二者之间 | 每组 Query 头共享 KV |
| MLA | KV 被投影到低维潜表示 | 由潜表示与位置部分决定 | 需要匹配具体模型分解与 kernel |

MHA、MQA、GQA 的单层 KV 元素数量为：

$$
2\times S\times N_{kv}\times D
$$

从 MHA 改成 GQA 可以按 $N_{kv}/N_q$ 的比例降低这一主项，但实际缓存还包含 block 元数据、对齐、量化 scale 和可能的并行副本。

---

## GQA 的映射

若 $N_q=32$、$N_{kv}=8$，每 4 个 Query 头共享一个 KV 头。实现不必真的复制 KV 4 次；kernel 可通过头索引映射读取对应 KV：

$$
h_{kv}=\left\lfloor\frac{h_qN_{kv}}{N_q}\right\rfloor
$$

当 $N_q$ 不能被 $N_{kv}$ 整除时，映射和模型定义需要明确，不能自行假设均匀分组。

---

## MLA 不能只看成「更小的 GQA」

MLA 把历史表示压缩为潜向量，并在计算 Attention 时恢复或吸收部分投影。RoPE 相关部分可能单独保存，因为旋转位置结构与矩阵吸收存在约束。不同模型的 MLA 细节并不完全相同，应以模型论文和当前实现为准。

系统上需要回答：

- cache 实际保存哪些分量；
- 投影是否被权重吸收；
- Prefill 与 Decode 分别增加哪些计算；
- TP 如何切分潜表示和头；
- FlashAttention 或 PagedAttention kernel 是否原生支持。

---

## 质量与性能边界

更少的 KV 头通常减少缓存与 Decode 读流量，但不保证在所有模型、硬件和 workload 上按比例提速。瓶颈还可能位于权重读取、调度、采样、CPU 前端或通信。

质量比较必须控制参数量、训练 token、训练配方和评测集。一个已有 MHA 权重也不能通过部署时简单合并 KV 头而无损转换为 GQA。

---

## CPU 实验设计

CPU 可以构造同一组小 Tensor，分别执行显式 KV 复制与索引共享，验证输出一致性和缓存元素数量。还可以测量不同 $N_{kv}$ 下的内存流量趋势。该实验不能代表 GPU fused attention 的吞吐，但能验证头映射与容量公式。

## 参考文献

- Vaswani, A. et al. (2017). *Attention Is All You Need*.
- Shazeer, N. (2019). *Fast Transformer Decoding: One Write-Head is All You Need*.
- Ainslie, J. et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*.
- DeepSeek-AI. (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*.
