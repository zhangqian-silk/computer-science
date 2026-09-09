# MHA、MQA、GQA 与 MLA

Attention 头共享方案改变 Query 头如何复用 Key/Value 表示。它们的主要系统差异集中在 KV 投影、KV cache 容量、Decode 带宽和并行切分，而不是 Attention 的概率读取接口本身。

<HeadSharingExplorer />

::: info 符号与约定
$n_{\text{head}}$ 是 Query 头数，$n_{\text{kv}}$ 是 Key/Value 头数，$d_{\text{head}}$ 是单头维度，$T$ 是缓存序列长度；$B,L,b_{\text{elem}}$ 分别是 batch、层数和每元素字节数。MHA/MQA/GQA 的容量比较假设 K 与 V 维度相同，MLA 在其小节单独定义。
:::

---

## 四种结构

| 结构 | 头关系 | KV cache 规模特征 | 主要约束 |
| --- | --- | --- | --- |
| MHA | $n_{\text{kv}}=n_{\text{head}}$ | 同头维度下，前三者中最大 | 每个 Query 头有独立 KV 头 |
| MQA | $n_{\text{kv}}=1$ | 同头维度下，前三者中最小 | 所有 Query 头共享一组 KV |
| GQA | $1<n_{\text{kv}}<n_{\text{head}}$ | 介于二者之间 | 每组 Query 头共享 KV |
| MLA | KV 被投影到低维潜表示 | 由潜表示与位置部分决定 | 需要匹配具体模型分解与 kernel |

MHA、MQA、GQA 对单条序列的单层 KV 元素数量为：

$$
2Tn_{\text{kv}}d_{\text{head}}
$$

从 MHA 改成 GQA 可以按 $n_{\text{kv}}/n_{\text{head}}$ 的比例降低这一主项，但实际缓存还包含 block 元数据、对齐、量化 scale 和可能的并行副本。

Query 通常只用于当前步读取历史，历史 token 的 Q 不必缓存；每层 K/V 则会被后续生成反复读取。因此减少 KV 头数改变的是需要保存和搬运的历史表示，而不是「少计算几个当前 Query」。

---

## GQA 的映射

若 $n_{\text{head}}=32$、$n_{\text{kv}}=8$，每 4 个 Query 头共享一个 KV 头。实现不必真的复制 KV 4 次；kernel 可通过从 0 开始的头索引映射读取对应 KV：

$$
h_{\text{kv}}=\left\lfloor\frac{h_qn_{\text{kv}}}{n_{\text{head}}}\right\rfloor
$$

这里要求 $n_{\text{head}}$ 能被 $n_{\text{kv}}$ 整除，且 $0\le h_q<n_{\text{head}}$。不能整除时，映射和模型定义需要明确，不能自行假设均匀分组。

### 共享 KV 不等于共享注意力权重

记 $g(h)$ 为 Query 头 $h$ 所属的 KV 组。在单步位置 $t$，省略位置变换的读取公式为：

$$
\alpha_{t,j}^{(h)}
=\operatorname{softmax}_{j\le t}
\left(\frac{(q_t^{(h)})^\top k_j^{(g(h))}}{\sqrt{d_{\text{head}}}}\right),
\qquad
o_t^{(h)}=\sum_{j\le t}\alpha_{t,j}^{(h)}v_j^{(g(h))}
$$

同组头用相同的 $k_j,v_j$，但 $q_t^{(h)}$ 来自不同投影，通常得到不同权重和输出。上方交互图应按「多条 Query 读取同一组 KV」理解，不能把整组输出直接合并成一条。

GQA 不改变因果 mask；每个 Query 头仍读取全部允许的历史。对于长度 $T$ 的训练序列，标准全连接注意力仍有 $O(n_{\text{head}}T^2d_{\text{head}})$ 主导计算，而非自动变成线性注意力。

### 从元素数算到真实容量单位

假设 $B=1$、$L=32$、$T=8\,192$、$n_{\text{head}}=32$、$d_{\text{head}}=128$，K/V 均为每元素 2 字节，不含元数据：

$$
M_{\text{KV}}=2BLTn_{\text{kv}}d_{\text{head}}b_{\text{elem}}
$$

| 方案 | KV 头数 | 全部 32 层的理论 KV 字节数 | 二进制单位 |
| --- | --- | --- | --- |
| MHA | 32 | 4,294,967,296 | 4 GiB |
| GQA | 8 | 1,073,741,824 | 1 GiB |
| MQA | 1 | 134,217,728 | 128 MiB |

这是容量算例，不是吞吐测量。若批量增加到 4，且每条请求长度相同、无前缀共享，以上结果乘 4；分布式复制、分页和低精度元数据另算，详见 [KV Cache 管理](../infra/inference/kv-cache-management.md)。

---

## MLA 不能只看成「更小的 GQA」

MLA 把历史表示压缩为潜向量，并在计算 Attention 时恢复或吸收部分投影。RoPE 相关部分可能单独保存，因为旋转位置结构与矩阵吸收存在约束。不同模型的 MLA 细节并不完全相同，应以模型论文和当前实现为准。

以 DeepSeek-V2 的低秩 KV 内容分支为例，$x_j\in\mathbb{R}^{d_{\text{model}}}$ 经降维得到 $c_j\in\mathbb{R}^{d_c}$；对一个头，设内容 K、V 的宽度均为 $d_{\text{head}}$：

$$
c_j=W^{DKV}x_j,\qquad
k_j^C=U_Kc_j,\qquad v_j^C=U_Vc_j
$$

$W^{DKV}\in\mathbb{R}^{d_c\times d_{\text{model}}}$，$U_K,U_V\in\mathbb{R}^{d_{\text{head}}\times d_c}$。内容分数可以改写为：

$$
(q_t^C)^\top k_j^C
=(q_t^C)^\top U_Kc_j
=(U_K^\top q_t^C)^\top c_j
$$

因此不必把每个历史 $c_j$ 全部恢复成 K 再打分。Value 的线性映射也可移出加权求和：

$$
\sum_j\alpha_{t,j}U_Vc_j
=U_V\left(\sum_j\alpha_{t,j}c_j\right)
$$

以上是给定参数化内部的代数恒等式，不表示任意已训练 MHA 都可以无损低秩压缩。它与 GQA 的区别是「多个头从共享潜表示恢复不同内容」，而不是仅共享相同的显式 KV 头。

若在完整 K 上加入位置旋转 $R_j$，分数中会出现 $(q_t)^\top R_t^\top R_jU_Kc_j$，历史位置 $j$ 相关的矩阵妨碍一次性把 $U_K$ 吸收到当前 Query 投影。因此 DeepSeek-V2 另设携带 RoPE 的分支，令位置 K 为所有头共享的 $k_j^R\in\mathbb{R}^{d_R}$：

$$
s_{t,j}
=\frac{(q_t^C)^\top k_j^C+(q_t^R)^\top k_j^R}
{\sqrt{d_{\text{head}}+d_R}}
$$

每层每个历史 token 缓存 $c_j$ 和 $k_j^R$，共 $d_c+d_R$ 个元素；不是两份 $d_c$，也不是只存潜向量而完全忽略位置。位置分支如何旋转见 [RoPE](./rope.md)。

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

GQA 原论文为已有 MHA checkpoint 提供了组内 K/V 投影均值初始化与继续训练的配方。论文使用原预训练计算的一小部分进行 uptraining，并在其模型和任务上展示质量与解码速度的折中；这不是无需训练的等价变换，也不是任何模型只追加固定比例训练就能恢复质量的定律。

从演进轴看，MQA 先把 KV 共享推到极端，GQA 增加可调的组数恢复部分容量，MLA 则改变保存表示的参数化。这些方法都针对解码缓存问题，但不是「年份越晚、质量和速度越高」的统一排序。

---

## CPU 实验设计

CPU 可以构造同一组小 Tensor，分别执行显式 KV 复制与索引共享，验证输出一致性和缓存元素数量。还可以测量不同 $n_{\text{kv}}$ 下的内存流量趋势。该实验不能代表 GPU fused attention 的吞吐，但能验证头映射与容量公式。

---

## 参考文献

- Vaswani, A. et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
- Shazeer, N. (2019). [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150). MQA 与增量解码的内存带宽动机。
- Ainslie, J. et al. (2023). [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245). 分组、均值初始化与 uptraining；质量结论需连同训练与推理实验读取。
- DeepSeek-AI. (2024). [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434). MLA 部分定义联合压缩与解耦 RoPE。本文讨论 DeepSeek-V2 的设计，不把它当作所有 MLA 变体的规范。
