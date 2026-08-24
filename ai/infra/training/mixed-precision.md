# 混合精度训练

混合精度训练为不同状态选择不同 dtype，以降低显存和数据移动，并使用加速器的低精度矩阵能力。它的正确性边界是训练收敛与目标质量，而不只是单次前向误差。

---

## 状态精度分工

一套训练配置需要分别记录：

| 状态或操作 | 可选精度角色 |
| --- | --- |
| 模型参数 | 计算副本、存储副本、FP32 master weight |
| 激活 | forward 输出与 backward 保存 dtype |
| 梯度 | 产生、累积、归约和存储 dtype |
| Optimizer | 一阶、二阶矩与更新计算 dtype |
| Loss/Norm | 归约与稳定计算 dtype |
| Collective | 传输与归约 dtype |

同样写着「BF16 训练」的两个框架，optimizer state、梯度归约和 master weight 仍可能不同。显存估算必须读取实际配置和实现。

---

## Loss Scaling

FP16 小梯度可能下溢。Loss scaling 先把 loss 乘以 $s$，反向得到放大梯度，再在更新前除以 $s$。动态策略在发现 inf/nan 时跳过更新并降低 scale，在稳定一段时间后尝试增大。

需要跨 rank 一致处理非有限梯度：如果一个 rank 跳过而其他 rank 更新，模型状态会立即分叉。检测、unscale、梯度裁剪与 AllReduce 的顺序必须由框架契约确定。

BF16 具有更宽指数范围，通常不需要相同的 loss scaling，但仍需监测溢出、异常 loss 和梯度范数。

---

## FP8 训练

FP8 常依赖每 Tensor 或更细粒度 scale、amax 历史和硬件支持。Scale 本身成为训练状态，更新时序和跨 rank 一致性会影响结果。启用 FP8 前需要验证：

- 模型与算子覆盖范围；
- fallback 到高精度的比例；
- scale 更新与 Checkpoint；
- 目标硬件是否真正加速；
- 收敛曲线和最终质量。

---

## 数值排障

非有限值出现时，从第一个异常位置开始：

1. 记录 loss、梯度范数、scale 和学习率；
2. 检查输入、mask 和标签是否有效；
3. 在关键层插入有限值断言；
4. 对比 FP32 小规模参考；
5. 检查 fused kernel、collective 和重算是否改变顺序；
6. 缩小到最早出现差异的算子。

只降低学习率可能暂时掩盖 kernel 或数据错误，不能替代定位。

---

## CPU 路线

CPU 可在支持的 ISA 和 PyTorch 后端上测试 BF16/AMP；不支持时可显式转换 dtype，验证 loss scaling、overflow 检测和状态机。CPU 无法证明 GPU Tensor Core 的性能，但可以建立 FP32 收敛参考和混合精度正确性测试。

前置知识见[数值计算](../../foundations/numerical-computing.md)和[精度与 Tensor Core](../accelerator/precision-and-tensor-cores.md)。

## 参考文献

- Micikevicius, P. et al. (2018). *Mixed Precision Training*.
- NVIDIA. *Transformer Engine Documentation*.
