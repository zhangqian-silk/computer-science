# 加速器与性能工程

单设备性能是一条从数学到时间的链：张量形状决定工作量，布局决定地址，线程组织决定复用，硬件资源决定上界，运行时决定额外开销。按下表学习时，应使用同一个矩阵乘或 softmax 案例贯穿各页，而不是分别记忆术语。

| 页面 | 核心问题 |
| --- | --- |
| [GPU 执行与存储层级](./gpu-architecture.md) | Thread、Warp、SM 与不同存储层级如何协作 |
| [性能模型](./performance-model.md) | 算子受 FLOPs、带宽、并行度还是 launch 限制 |
| [精度与 Tensor Core](./precision-and-tensor-cores.md) | dtype 如何影响正确性、容量和硬件路径 |
| [CUDA、Triton 与 Kernel](./cuda-and-kernels.md) | 如何把 Tensor 公式映射成高效 kernel |
| [编译图与执行计划](./compiler-and-graphs.md) | 图捕获、融合和代码生成怎样减少调度开销 |
| [Profiling](./profiling.md) | 如何从端到端时间线定位到具体 kernel |

CPU 不是本分区的附属版本。它同样需要理解 SIMD、cache、NUMA、线程池与 BLAS；但 GPU 特有的 Warp、Shared Memory、Tensor Core 和 CUDA Graph 只能在相应硬件上验证。

---

## 分区学习闭环

先对一个 $[m,k][k,n]$ 矩阵乘写出 FLOPs 与理想读写字节，再解释增大 $m$ 为什么改变权重复用。随后比较 FP32 与低精度误差，最后用时间线确认瓶颈是否真的在该矩阵乘。读完应能区分「理论上可能获益」「局部实测获益」与「请求级实测获益」。

完成后进入[分布式通信](../distributed/index.md)，把单设备上的数据复用问题扩展为跨设备的数据归属；实践安排见[加速器与分布式实验](../labs/accelerator-distributed-labs.md)。
