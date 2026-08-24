# 加速器与性能工程

本分区从单设备视角解释模型算子如何执行。阅读顺序应从执行和存储层级开始，再学习性能模型、数值精度、kernel、编译图和 profiling。

| 页面 | 核心问题 |
| --- | --- |
| [GPU 执行与存储层级](./gpu-architecture.md) | Thread、Warp、SM 与不同存储层级如何协作 |
| [性能模型](./performance-model.md) | 算子受 FLOPs、带宽、并行度还是 launch 限制 |
| [精度与 Tensor Core](./precision-and-tensor-cores.md) | dtype 如何影响正确性、容量和硬件路径 |
| [CUDA、Triton 与 Kernel](./cuda-and-kernels.md) | 如何把 Tensor 公式映射成高效 kernel |
| [编译图与执行计划](./compiler-and-graphs.md) | 图捕获、融合和代码生成怎样减少调度开销 |
| [Profiling](./profiling.md) | 如何从端到端时间线定位到具体 kernel |

CPU 不是本分区的附属版本。它同样需要理解 SIMD、cache、NUMA、线程池与 BLAS；但 GPU 特有的 Warp、Shared Memory、Tensor Core 和 CUDA Graph 只能在相应硬件上验证。
