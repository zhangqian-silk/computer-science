# 精度与 Tensor Core

低精度可以减少权重、激活和通信字节，并在支持的硬件上使用更高吞吐的矩阵单元。真正的执行路径由输入 dtype、累加 dtype、矩阵形状、对齐、kernel 和硬件共同决定；把权重文件改小并不等于计算一定使用低精度加速。

---

## 存储、计算与累加

一次 GEMM 至少涉及三种精度角色：

- 输入 A、B 的存储与乘法精度；
- 乘加累加精度；
- 输出存储精度。

例如输入可以是 FP16/BF16，内部使用 FP32 累加，输出再写回低精度。TF32 则常把 FP32 输入映射到 Tensor Core 支持的乘法精度，同时保留更宽的指数范围。

Tensor Core 要求受硬件和库版本约束的形状与对齐。若维度太小、不对齐或算子不受支持，框架可能 padding、选择其他 kernel 或退回普通执行单元。

---

## 精度选择表

| 目标 | 首先检查 | 风险 |
| --- | --- | --- |
| 训练稳定 | 梯度、归约、optimizer state、loss scaling | overflow、underflow、非有限梯度 |
| 推理容量 | 权重和 KV dtype、scale 元数据 | 质量损失、kernel 不支持 |
| 推理吞吐 | 是否命中硬件低精度路径 | 转换开销抵消收益 |
| 分布式通信 | collective 传输 dtype 与归约 dtype | 累积误差、协议不匹配 |

FP8 和 INT4 等路径尤其需要记录格式、scale 粒度、校准方法和目标硬件。不同名称相同位宽也可能拥有不同动态范围。

---

## 验证顺序

1. 用 FP32 或可信高精度实现建立参考输出；
2. 对单层和完整模型设置绝对、相对误差或任务指标；
3. 检查实际 kernel 与硬件计数器，确认命中目标路径；
4. 测量转换、反量化和 padding 开销；
5. 在真实长度和并发分布上验证端到端收益。

只通过单次输出接近不能证明长序列、自回归误差累积或训练收敛稳定。

---

## CPU 路线

现代 CPU 可能支持 BF16、INT8 或更宽向量指令，但能力依型号而异。无专用支持时仍可模拟转换和误差，或使用量化 BLAS 验证模型容量。CPU 实验应报告 ISA、线程数和后端库，不能使用 GPU 的 Tensor Core 吞吐作为比较基线。

前置知识见[数值计算](../../foundations/numerical-computing.md)，训练策略见[混合精度](../training/mixed-precision.md)，部署量化见[推理量化](../inference/quantization.md)。

## 参考资料

- NVIDIA. *Tensor Core Programming*.
- Micikevicius, P. et al. (2022). *FP8 Formats for Deep Learning*.
