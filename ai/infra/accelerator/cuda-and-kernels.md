# CUDA、Triton 与 Kernel

Kernel 把 Tensor 级公式映射为线程、内存访问和同步。CUDA C++ 提供直接控制，Triton 用 block program 和编译器抽象表达相同数据并行；二者都需要先证明正确，再通过 profile 证明性能。

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

## 参考资料

- NVIDIA. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- OpenAI. [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html).
