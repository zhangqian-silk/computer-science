# 前馈网络与门控激活

Attention 在 token 之间搬运信息，前馈网络在每个 token 内重组通道。对大多数 Decoder-only 模型，FFN 包含大规模矩阵乘，是参数量和训练 FLOPs 的重要来源；MoE 则把其中一部分替换成条件激活的专家。

---

## 逐位置 MLP

基础形式为：

$$
\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2
$$

若 $x\in\mathbb{R}^{H}$，则 $W_1\in\mathbb{R}^{H_{ff}\times H}$，$W_2\in\mathbb{R}^{H\times H_{ff}}$。忽略 bias，一层 FFN 参数量约为 $2HH_{ff}$。

对 $T=B\times S$ 个 token，两次投影可以视为矩阵乘：

$$
[T,H]\times[H,H_{ff}],
\qquad
[T,H_{ff}]\times[H_{ff},H]
$$

训练或 Prefill 的 $T$ 较大，容易形成高效 GEMM；单请求 Decode 的 $T$ 很小，更可能受到权重读带宽和 kernel launch 影响。

---

## 激活与门控

ReLU、GELU 和 SiLU 提供不同非线性。门控 FFN 常写为：

$$
\operatorname{GatedFFN}(x)
=
W_o\left(\phi(W_gx)\odot W_ux\right)
$$

当 $\phi$ 使用 SiLU 时通常称为 SwiGLU。门控形式有三个投影，因此不能在相同 $H_{ff}$ 下直接与两投影 FFN 比较参数量；实际模型常调整中间维度，使总参数或计算接近目标预算。

激活函数名称也不足以确定实现。需要检查 bias、维度取整、并行切分、量化粒度，以及门和值分支是否融合计算。

---

## Tensor Parallel 切分

第一层投影可以按输出通道切分，每个 rank 得到部分中间维；第二层按输入通道读取局部结果，随后用 AllReduce 或 ReduceScatter 合并输出。这样中间大激活不必完整复制到每个设备。

门控 FFN 要让 gate 与 up 投影采用一致切分，否则逐元素乘无法在本地完成。MoE 还会按 token 路由到专家，详见 [MoE](./moe.md) 与[训练并行策略](../infra/training/parallelism.md)。

---

## 融合与量化

常见优化包括：

- 合并 gate/up 投影；
- 把 bias、激活和逐元素乘融合；
- 选择适合形状的 GEMM tile；
- 量化权重并在 kernel 中融合反量化；
- 对 Decode 合并多个请求，增大 token batch。

优化必须保留形状与数值语义。对低精度门控网络，还应检查激活离群值和 scale 是否导致某一分支饱和。

CPU 可以用 BLAS 或 oneDNN 验证 FFN 形状、门控公式、量化误差和 batch 对吞吐的影响；GPU 才能验证 Tensor Core、HBM 和特定融合 kernel 的收益。

## 参考文献

- Hendrycks, D. and Gimpel, K. (2016). *Gaussian Error Linear Units*.
- Shazeer, N. (2020). *GLU Variants Improve Transformer*.
