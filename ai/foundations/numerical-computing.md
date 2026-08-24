# 数值计算：精度、稳定性与可复现性

模型公式定义的是实数运算，硬件执行的是有限位宽浮点或整数运算。AI Infra 必须同时维护两类正确性：算法语义正确，以及给定 dtype、kernel 和并行顺序下误差可接受。

---

## 浮点数表达什么

浮点数用符号、指数和尾数近似实数。指数位决定动态范围，尾数位决定局部精度。FP16 的尾数精度较高但指数范围较窄；BF16 复用 FP32 的指数宽度，训练时通常更不易溢出；FP8 存在不同编码，必须连同 scale 和累加类型描述。

| 类型 | 典型字节 | 主要特征 | 常见用途 |
| --- | ---: | --- | --- |
| FP32 | 4 | 范围和精度较高 | 参考结果、部分累加与状态 |
| TF32 | 4 | FP32 范围、较低乘法精度 | NVIDIA Tensor Core 矩阵乘 |
| FP16 | 2 | 范围较窄 | 训练与推理，需要注意缩放 |
| BF16 | 2 | 范围接近 FP32 | 现代训练和推理 |
| FP8 | 1 | 范围与精度依格式而异 | 受支持硬件上的低精度计算 |
| INT8/INT4 | 1/半字节 | 离散整数与 scale | 量化推理或训练 |

表中的字节数只描述数据本身，量化还要保存 scale、zero point、分组元数据和可能的 padding。

---

## 非结合性改变并行结果

浮点加法通常不满足严格结合律：

$$
(a+b)+c\ne a+(b+c)
$$

并行归约、不同 kernel、不同 GPU 数量可能改变求和顺序，因此结果不一定逐位相同。可复现性应分层描述：

- bitwise reproducibility：每一位完全相同；
- numerical reproducibility：误差在给定容差内；
- statistical reproducibility：训练指标或分布在统计范围内。

分布式训练通常更适合后两种标准。声明「确定性」时应写明随机种子、算子、硬件、线程和拓扑边界。

---

## Softmax 与归一化的稳定形式

直接计算 $\exp(x_i)$ 可能上溢。Softmax 利用平移不变性：

$$
\operatorname{softmax}(x_i)
=
\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)},
\qquad m=\max_j x_j
$$

在线 Softmax 可以分块更新局部最大值和归一化因子，是 FlashAttention 避免物化完整分数矩阵的数值基础。归一化 kernel 也常在低精度输入上使用更高精度累加，以控制均值与方差误差。

---

## 量化误差来自哪里

对称量化可写为：

$$
q=\operatorname{clip}\left(\operatorname{round}(x/s),q_{\min},q_{\max}\right),
\qquad \hat{x}=sq
$$

误差受位宽、scale 粒度、离群值和校准数据影响。per-tensor scale 元数据少但难以适应通道差异；per-channel 或 group-wise scale 更细，却增加元数据与 kernel 复杂度。

量化验收不能只比较 perplexity，还应覆盖真实任务、长上下文、结构化输出和极端输入，并验证 kernel 是否真正使用目标低精度路径。

---

## 混合精度的职责分离

「混合精度」不是把所有 Tensor 改成相同低精度。常见策略分别选择：

- 参数存储 dtype；
- GEMM 输入与输出 dtype；
- 累加 dtype；
- loss 和归约 dtype；
- optimizer state dtype；
- collective 传输 dtype。

训练中的 loss scaling 通过放大 loss，避免 FP16 的小梯度下溢；若检测到 inf 或 nan，则跳过更新并调整 scale。BF16 动态范围更大，通常不依赖同样的缩放策略，但仍可能产生非有限值。

---

## CPU 路线

CPU 足以验证 dtype 转换、Softmax 稳定性、量化误差、归约顺序和容差测试。可使用 NumPy、PyTorch CPU 或支持 BF16/INT8 的 CPU 后端。CPU 结果不能证明 Tensor Core 或特定 GPU kernel 的吞吐，但很适合作为高精度参考实现。

后续阅读：[精度与 Tensor Core](../infra/accelerator/precision-and-tensor-cores.md)、[混合精度训练](../infra/training/mixed-precision.md)和[推理量化](../infra/inference/quantization.md)。

## 参考文献

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*.
- Micikevicius, P. et al. (2018). *Mixed Precision Training*.
