# 推理量化

量化将实数近似为有限网格上的数，首先是一种带误差的表示变换，其次才可能节省容量与带宽。权重、激活和 KV 的统计分布及复用方式不同，不能用一个「INT4 模型」名称概括所有数值路径。

<QuantizationTradeoff />

---

## 量化对象

| 对象 | 主要收益 | 主要风险 |
| --- | --- | --- |
| Weight-only | 减少权重容量和 Decode 读带宽 | 反量化开销、离群通道 |
| Weight + Activation | 命中整数/FP8 矩阵路径 | 校准与激活动态范围 |
| KV cache | 提高上下文容量、减少 KV 带宽 | 长上下文质量与 Attention kernel 支持 |
| Logits/通信 | 减少特定 buffer 或传输 | 对排序与归约敏感 |

每种对象可以使用不同 dtype。不能用「模型是 INT4」概括整个执行图。

---

## Scale 粒度

对量化整数 $q$，重建值一般为 $\hat{x}=s(q-z)$。对称形式令 $z=0$，通常还需裁剪到允许整数范围。下式适用于组内最大绝对值非零；全零组必须另设正的合法 scale，并令输出全零，避免除零。

对称 group-wise 量化把一组 $G$ 个值共享 scale：

$$
s_g=\frac{\max_{i\in g}|x_i|}{q_{max}},
\qquad
q_i=\operatorname{round}(x_i/s_g)
$$

group 越小通常更能适应局部范围，但 scale 元数据更多、kernel 解码更复杂。还需记录是否非对称、是否有 zero point、布局如何 pack、计算时使用何种累加精度。

以对称整数范围 $[-7,7]$ 和组内值 $(0.1,0.2,0.3,7)$ 为例，scale 为 1，前三个值都可能舍入成 0。单个离群值决定了整组网格。若拆组，小值可以用更细 scale，但每组都要额外保存元数据；这解释了 group size 的质量—容量折中。

在未被裁剪且普通舍入的区间内，标量重建误差最多约 $s/2$，但网络输出误差还受输入、权重相关性及多层传播影响。逐权重误差小不必然等于任务误差小，所以校准方法会利用激活或曲率信息。

---

## 算法与制品格式

GPTQ 近似研究量化某些权重后，如何补偿同一层其余权重以减小输出重建误差；AWQ 关注由激活反映的重要通道并通过缩放改善量化；SmoothQuant 用等价缩放把激活中的量化困难迁移给权重。这是不同误差建模策略，不是三种文件后缀。比较论文结果时需控制校准集、位宽、group 和 kernel。

- GPTQ 是基于近似二阶信息的 post-training weight quantization 方法；
- AWQ 依据激活观察保护重要权重通道，并配合 weight-only kernel；
- SmoothQuant 通过缩放在权重和激活之间迁移量化难度；
- GGUF 是模型制品与量化数据的文件格式，不是与 GPTQ/AWQ 同层的量化算法；
- FP8 可能用于权重、激活与 KV，具体格式和 scale 规则依框架与硬件。

框架支持列表会变化，部署前必须核对目标模型、算子、GPU/CPU 后端和并行模式。

---

## 质量验收

依次验证：

1. 单层或固定输入 logits 与高精度参考的误差；
2. perplexity 或领域 loss；
3. 目标任务准确性、生成质量和结构化输出；
4. 长上下文与高困惑输入；
5. 不同采样随机种子和停止条件；
6. 吞吐、TTFT、TPOT、KV 容量和峰值内存。

生成文本逐 token 完全一致不是所有量化的合理目标，因为很小 logits 差异即可改变采样路径；应同时保留确定性 logits 测试与任务级统计。

---

## 性能边界

Weight-only 量化在小 batch Decode 中常有机会减少权重带宽；大 batch Prefill 可能更依赖计算 kernel。若每轮先把全部权重反量化到高精度临时 Tensor，内存和转换开销可能抵消收益。

还应检查量化模型是否禁用了 CUDA Graph、FlashAttention、TP 或 speculative decoding 等其他能力。单项节省不能代表完整服务配置更优。

---

## CPU 路线

CPU 是量化的重要部署目标。llama.cpp、GGML/BLAS 或 PyTorch quantization 可以验证 INT8/INT4 制品、SIMD kernel、内存和质量。报告 CPU 型号、ISA、线程数、NUMA 和量化格式。CPU 结果与 GPU kernel 不直接可比，但可以形成一条完整的低成本推理路线。

---

## 参考文献

- Frantar, E. et al. (2022). [*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*](https://arxiv.org/abs/2210.17323).
- Lin, J. et al. (2023). [*AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*](https://arxiv.org/abs/2306.00978).
- Xiao, G. et al. (2022). [*SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*](https://arxiv.org/abs/2211.10438).
- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
