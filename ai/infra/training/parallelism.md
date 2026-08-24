# 训练并行策略

训练并行的本质是决定参数、梯度、优化器状态、激活、序列和 token 分别由哪些 rank 持有。每种策略都以通信、调度或重算换取容量与吞吐，不能只用缩写判断优劣。

<ParallelismMesh />

---

## 并行维度

| 策略 | 主要切分对象 | 典型通信 | 主要目的 |
| --- | --- | --- | --- |
| DP/DDP | 样本 batch | 梯度 AllReduce | 扩大数据吞吐 |
| ZeRO/FSDP | 参数、梯度、optimizer state | ReduceScatter、AllGather | 降低每 rank 状态内存 |
| TP | 层内矩阵通道/头 | AllReduce、ReduceScatter、AllGather | 单层跨设备 |
| PP | 层或 block | stage 间 send/recv | 模型深度跨设备 |
| SP | 激活的序列维 | 与 TP 配套的 gather/scatter | 降低重复激活 |
| CP | 上下文/序列 | Attention 相关 P2P/Collective | 扩展长上下文 |
| EP | MoE 专家 | AllToAll | 专家参数和 token 分布 |

一个大型训练作业通常把多个维度组成 DeviceMesh。global world size 是各独立并行维度大小的乘积，但具体框架可能让某些维度嵌套或共享 group。

---

## Data Parallel 与状态分片

DDP 在每个 rank 保留完整模型和 optimizer，按数据切分 batch，backward 时同步梯度。它最简单，但模型状态必须能放入单 rank。

ZeRO 按阶段分片 optimizer state、gradient 和 parameter。FSDP 同样围绕参数分片，在 forward/backward 前后 AllGather 和 ReduceScatter。不同框架的参数生命周期、预取和扁平化不同，不能只依据「stage 3」与「full shard」名称判断完全等价。

---

## Tensor Parallel

对线性层 $Y=XW$，可以按 $W$ 的列切分输出，再在后续行切分层完成局部计算并归约。合理配对能避免在每个线性层之间完整 AllGather。

Attention 的 head 和 FFN 中间维很适合切分，但 LayerNorm、残差和词表输出需要明确复制或切分方式。小矩阵或慢网络上，频繁 Collective 会抵消并行收益。

---

## Pipeline Parallel

PP 把层分为 stage，并把一个 batch 拆成 microbatch 形成流水线。流水线填充和排空产生 bubble；调度方式还影响 activation memory、权重版本与通信重叠。

阶段划分应按实测时间和内存平衡，而不是只按层数平均。Embedding、输出头、MoE 和长序列 Attention 都可能让各层成本不同。

---

## Context 与 Expert Parallel

CP 把长序列分到多个 rank，同时保持 Attention 看到所需上下文；通信量和算法取决于 ring、all-gather 或其他实现。EP 把专家分片，router 决定 token 经 AllToAll 发往何处。负载不均会形成 straggler，capacity 与 token drop 又可能影响训练语义。

---

## 选择顺序

一种稳妥的选择逻辑是：

1. 单设备能放下时先建立单设备基线；
2. 模型能放下但需要吞吐时使用 DDP；
3. 状态放不下时引入 FSDP/ZeRO；
4. 单层或长上下文仍放不下时引入 TP/CP；
5. 模型层数与节点数需要扩展时考虑 PP；
6. MoE 模型按专家布局引入 EP；
7. 根据真实拓扑重新映射各维度。

这是诊断顺序，不是固定配置。每增加一维并行都会扩大通信、故障和调试空间。

---

## CPU 路线

多进程 CPU 可以验证 DDP、FSDP、TP 的 shape 与 collective 语义，以及 PP microbatch 调度和 EP token 路由。可使用小模型和 Gloo/MPI。CPU 结果不用于预测 GPU/NVLink/NCCL 的 scaling efficiency。

## 参考资料

- PyTorch. [Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html).
- NVIDIA. [Megatron Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).
- Rajbhandari, S. et al. (2020). *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*.
