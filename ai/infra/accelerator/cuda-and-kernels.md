# CUDA、Triton 与 Kernel

Kernel 是 Tensor 公式的执行实现，不是另一种模型机制。同一个 softmax 可以产生许多不同的线程和访存组织，只要输入输出语义一致。CUDA C++ 侧重线程与硬件原语，Triton 侧重一组元素的张量化程序；二者最终都需要面对地址、归约、资源占用和同步。

---

## 用一行 Softmax 连接公式与实现

对有限输入行 $x_1,\ldots,x_n$，先求 $m=\max_i x_i$，再求 $s=\sum_i e^{x_i-m}$，最后输出 $e^{x_i-m}/s$。减最大值避免指数上溢；并行化的困难在于两个统计量依赖整行，而不是逐元素独立。

若一个 program 负责一行，可以先把该行加载到寄存器中的张量，完成最大值、指数和求和，再写回。长度不是块尺寸整数倍时，加载的额外位置对最大值应等价于 $-\infty$，对指数和应贡献 0；不能默认填 0 后仍得到相同最大值。

但「全被 mask 的行」不是这个推导的合法普通输入：最大值为 $-\infty$ 时，减法会产生非有限值。模型或 kernel 必须明确这种行是否可能出现、若出现应该输出什么；不能用一个随意的 epsilon 掩盖可见性错误。

融合的收益可以按流量估算。若分成多个逐元素 kernel，中间指数向量会写回再读出；保留在片上可以消除这些往返。反例是行很长导致寄存器 spill，或单个 block 过大降低并行度。此时多阶段归约可能优于强行单 kernel。

没有 GPU 时，先写一个高精度 CPU 参考，验证每行和接近 1、输出非负以及整体平移不变。GPU 实验再测真实布局、边界长度和累加精度；不能把参考实现速度当成 CUDA 优化基线。

---

## 从公式到 Kernel

一个 kernel 设计至少包含：

1. 输出空间怎样分块；
2. 每个 program 或 block 负责哪些元素；
3. 输入怎样加载、复用和写回；
4. 哪些线程需要通信或同步；
5. 边界、mask、非连续 stride 和 dtype 怎样处理。

以按行 Softmax 为例，每一行需要最大值归约、指数、求和归约和归一化。朴素实现若拆成多个 kernel，会多次读写整行；融合实现可在片上保留中间量，但必须处理长行、数值稳定和资源上限。

---

## CUDA 与 Triton 的分工

| 维度 | CUDA C++ | Triton |
| --- | --- | --- |
| 控制层级 | thread/warp/block 与硬件原语 | block program 与张量化操作 |
| 开发成本 | 较高 | 适合快速表达规则算子 |
| 可移植性 | 与 CUDA 生态紧密 | 依编译器和后端支持 |
| 极限控制 | 更强 | 某些特化路径受编译器能力约束 |

选择依据应是目标算子、团队维护能力、框架集成和支持硬件，而不是把一种工具视为普遍替代另一种。

---

## Kernel Fusion 的边界

融合可以减少 launch 和 HBM 往返，但可能增加寄存器、降低 occupancy、扩大编译时间或产生过多特化版本。适合融合的通常是生产者—消费者之间中间量大、计算简单且形状兼容的链路。

验证融合 kernel 时至少覆盖：

- 多组形状和非整除边界；
- 连续与允许的非连续 stride；
- 所有支持 dtype；
- 与参考实现的 forward/backward 误差；
- 空输入、极值、nan/inf 传播；
- warm-up 后的延迟分布和端到端收益。

---

## CPU 替代路线

CPU 可以用 Go、C++、OpenMP、SIMD intrinsics 或 BLAS 实现同一算子，学习 blocking、融合和数据布局。它适合建立 reference kernel，并可验证减少内存往返的方向。Triton 与 CUDA kernel 本身要求支持的加速器，CPU 版本不是相同性能实验。

仓库的编译前后端知识见[经典编译流程](../../../fundamentals/compiler/classic-pipeline.md)与[中间表示和后端](../../../fundamentals/compiler/intermediate-and-backend.md)。

---

## 参考资料

- NVIDIA. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- OpenAI. [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html).
