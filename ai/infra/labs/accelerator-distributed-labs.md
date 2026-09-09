# 加速器与分布式实验设计

本组实验从同一数学结果出发改变布局、精度和设备划分。CPU 路线检验公式与测量方法，GPU 路线检验真实执行路径；以下均为实验设计，不含实测性能承诺。

---

## 实验 5：GEMM、GEMV 与 Roofline

固定 $k,n$ 后扫描 $m=1,8,64,256$，分别记录 $2mkn$ FLOPs 与输入各读一次、输出写一次的字节下界。每组先预热，再重复计时；GPU 使用设备事件或可靠同步边界，不能只测异步提交时间。

预期较大 $m$ 提高权重复用，但小形状、cache 命中、线程调度或布局转换可能破坏简单趋势。结果应同时给矩阵尺寸、实际后端和误差；若发生隐式 dtype/布局转换，要么纳入端到端，要么明确排除。

**假设**：矩阵形状与 batch 改变算术强度，Decode 风格的小 $M$ 更容易受权重带宽或 launch 影响。

**变量**：$M,N,K$、dtype、线程/stream、布局、warm/cold cache。

**CPU 版**：使用 BLAS，记录 CPU ISA、线程和 NUMA；通过 STREAM 或等价工具测内存带宽。**GPU 版**：使用 cuBLAS/Triton，记录 Tensor Core 路径和 HBM 带宽。

**输出**：FLOPs、估算 Bytes、算术强度、实测性能、Roofline 位置和 profile 证据。

---

## 实验 6：Fused Softmax/Norm/Attention

从 1 行长度 7 开始，扩到跨 tile 边界的长度 31、33、127、129。验证 softmax 和为 1、有限输入输出有限、整体平移不改变结果；Attention 分块与完整实现比较相同 mask 和相同值。

至少构造一个「直接平均两个分块输出得到错误值」的反例，再使用在线最大值、分母和分子合并修复。全无效行按明确接口处理，不以 NaN 恰好消失为正确证据。

**假设**：融合和分块减少中间数据读写，但资源压力可能在特定 shape 抵消收益。

**CPU 版**：实现朴素与 cache-blocked/fused 参考，验证稳定 Softmax。**GPU 版**：使用 Triton 或 CUDA 完成 Softmax、RMSNorm，再对照官方 Attention tutorial。

**正确性**：多 shape、尾部、极值、FP32 参考、绝对/相对误差。**性能**：warm-up 后分位数、Bytes、kernel 数、端到端层耗时。

---

## 实验 7：Collective 与拓扑

先在四 rank 上使用可手算整数输入，逐个验证输入输出归属；再换成浮点并比较容差。消息大小从少量元素逐步放大，分别画启动主导区与带宽主导区。

单机多进程得到的延迟包含主机竞争，不是网络设备的独立性能。故障注入应设置有限测试期限，保留最早异常 rank 和操作序号，避免实验自身无限挂起。

**假设**：小消息主要受延迟影响，大消息主要受带宽和拓扑影响。

**CPU 版**：Gloo/MPI 多进程执行 Broadcast、AllReduce、AllGather、ReduceScatter、AllToAll，可在单机和多机比较。**GPU 版**：NCCL tests，并记录 GPU/NIC affinity、NVLink 与网络拓扑。

**故障注入**：rank 延迟、顺序不一致、进程退出和 timeout。验证全组失败、日志能够定位最早异常。

---

## 实验 8：DDP 到混合并行

以同一全局 batch 和一次更新后的参数作为正确性目标。先比较单进程与数据并行，再比较不均等 token 数下的归一化。随后明确画出每种 ZeRO 状态分片表，测量前向前 AllGather 导致的瞬时峰值。

PP 模拟记录每个 microbatch 的 stage 和时隙；EP 模拟记录原 token ID、目标专家和逆路由。最终报告不能只写「多卡更快」，还应指出单步工作量、最慢 rank 和暴露通信。

**假设**：每增加一维并行都降低某类本地内存或增加容量，同时引入特定 Collective 与调度开销。

**阶段**：单进程 → DDP → FSDP/ZeRO → 小型 TP → PP/EP 模拟。每阶段保持模型、global batch 与有效 token 语义一致。

**CPU 版**：小模型、多进程、Gloo，重点验证 state ownership 和 checkpoint。**GPU 版**：比较显存、tokens/s、通信暴露与 scaling efficiency。

**结论边界**：CPU 的正确性结论可迁移，性能排名不可直接迁移。
