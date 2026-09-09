# 训练并行策略

并行策略首先是一张所有权表：同一参数有几份、一次激活在哪个 rank、谁产生梯度、谁负责更新。DP 划分样本，TP 划分层内算子，PP 划分层，状态分片改变副本数量；先分清对象，才可能正确计算设备数、显存与通信。

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

设参数、梯度、优化器状态分别占 $M_p,M_g,M_o$ 字节，数据并行度为 $P$，忽略临时 AllGather 和其他 buffer，则每 rank 主体近似为：

| 方案 | 单 rank 模型状态 |
| --- | --- |
| DDP | $M_p+M_g+M_o$ |
| ZeRO-1 | $M_p+M_g+M_o/P$ |
| ZeRO-2 | $M_p+(M_g+M_o)/P$ |
| ZeRO-3 | $(M_p+M_g+M_o)/P$ |

这张表是 ZeRO 论文的分片对象视角，不是任意框架实际峰值。全分片执行某层前可能临时收集参数；预取层数越多，通信更可能被隐藏，但同时存活的完整参数也更多。FSDP 的参数组粒度与重分片策略正是在调整这条时空折中。

DDP 在每个 rank 保留完整模型和 optimizer，按数据切分 batch，backward 时同步梯度。它最简单，但模型状态必须能放入单 rank。

ZeRO 按阶段分片 optimizer state、gradient 和 parameter。FSDP 同样围绕参数分片，在 forward/backward 前后 AllGather 和 ReduceScatter。不同框架的参数生命周期、预取和扁平化不同，不能只依据「stage 3」与「full shard」名称判断完全等价。

---

## Tensor Parallel

列并行把 $W=[W_0,W_1]$，各 rank 得到 $XW_0,XW_1$；若下一层权重相应按行分片为 $U_0,U_1$，可在本地计算 $(XW_0)U_0$、$(XW_1)U_1$，最后求和。这解释了为何成对线性层能减少中间聚合，但非线性必须能够在局部分片上正确执行。

对线性层 $Y=XW$，可以按 $W$ 的列切分输出，再在后续行切分层完成局部计算并归约。合理配对能避免在每个线性层之间完整 AllGather。

Attention 的 head 和 FFN 中间维很适合切分，但 LayerNorm、残差和词表输出需要明确复制或切分方式。小矩阵或慢网络上，频繁 Collective 会抵消并行收益。

---

## Pipeline Parallel

在均匀 stage、理想流水且忽略通信的简化前向模型中，$p$ 个 stage 处理 $m$ 个 microbatch 需要约 $m+p-1$ 个时隙，填充/排空比例为 $(p-1)/(m+p-1)$。更多 microbatch 可摊薄 bubble，却改变内存与调度；实际 1F1B 等训练日程不能只套这个前向式子。

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

CPU 可用小模型和 Gloo/MPI 验证通信及显式分片语义，并模拟 PP/EP；是否能直接运行某版本 FSDP 入口需要另查加速器要求。概念模拟不等于框架整条 CPU 路径受支持，也不能预测 GPU 扩展效率。

---

## 参考资料

- PyTorch. [Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html).
- NVIDIA. [Megatron Core Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html).
- Rajbhandari, S. et al. (2020). [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054).
