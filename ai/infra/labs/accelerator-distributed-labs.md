# 加速器与分布式实验设计

本页的 CPU 路线用于学习性能方法和通信语义；GPU 路线用于验证 Tensor Core、HBM、CUDA/Triton 与 NCCL。

---

## 实验 5：GEMM、GEMV 与 Roofline

**假设**：矩阵形状与 batch 改变算术强度，Decode 风格的小 $M$ 更容易受权重带宽或 launch 影响。

**变量**：$M,N,K$、dtype、线程/stream、布局、warm/cold cache。

**CPU 版**：使用 BLAS，记录 CPU ISA、线程和 NUMA；通过 STREAM 或等价工具测内存带宽。**GPU 版**：使用 cuBLAS/Triton，记录 Tensor Core 路径和 HBM 带宽。

**输出**：FLOPs、估算 Bytes、算术强度、实测性能、Roofline 位置和 profile 证据。

---

## 实验 6：Fused Softmax/Norm/Attention

**假设**：融合和分块减少中间数据读写，但资源压力可能在特定 shape 抵消收益。

**CPU 版**：实现朴素与 cache-blocked/fused 参考，验证稳定 Softmax。**GPU 版**：使用 Triton 或 CUDA 完成 Softmax、RMSNorm，再对照官方 Attention tutorial。

**正确性**：多 shape、尾部、极值、FP32 参考、绝对/相对误差。**性能**：warm-up 后分位数、Bytes、kernel 数、端到端层耗时。

---

## 实验 7：Collective 与拓扑

**假设**：小消息主要受延迟影响，大消息主要受带宽和拓扑影响。

**CPU 版**：Gloo/MPI 多进程执行 Broadcast、AllReduce、AllGather、ReduceScatter、AllToAll，可在单机和多机比较。**GPU 版**：NCCL tests，并记录 GPU/NIC affinity、NVLink 与网络拓扑。

**故障注入**：rank 延迟、顺序不一致、进程退出和 timeout。验证全组失败、日志能够定位最早异常。

---

## 实验 8：DDP 到混合并行

**假设**：每增加一维并行都降低某类本地内存或增加容量，同时引入特定 Collective 与调度开销。

**阶段**：单进程 → DDP → FSDP/ZeRO → 小型 TP → PP/EP 模拟。每阶段保持模型、global batch 与有效 token 语义一致。

**CPU 版**：小模型、多进程、Gloo，重点验证 state ownership 和 checkpoint。**GPU 版**：比较显存、tokens/s、通信暴露与 scaling efficiency。

**结论边界**：CPU 的正确性结论可迁移，性能排名不可直接迁移。
