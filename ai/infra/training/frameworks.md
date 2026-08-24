# 训练框架地图

::: info 实现核验
本页描述稳定能力边界；具体 API 以使用时的官方文档和源码为准。框架选型记录必须包含版本、commit、硬件和启用的并行维度。
:::

训练框架不是互斥的完整产品。PyTorch Distributed 提供基础抽象，Megatron-Core 提供大模型并行组件，DeepSpeed 提供 ZeRO 与训练系统能力，TorchTitan 等项目展示原生 PyTorch 组合方式。实际栈可能同时使用多个层次。

---

## 能力对照

| 框架/组件 | 主要职责 | 阅读入口 |
| --- | --- | --- |
| PyTorch DDP | 复制模型的数据并行 | process group、bucket、autograd hook |
| PyTorch FSDP2/DTensor | 参数分片与设备网格 | placement、reshard、state dict |
| Megatron-Core | TP、PP、CP、EP 与高性能模型组件 | parallel state、transformer、optimizer |
| DeepSpeed | ZeRO、offload、engine 与配置系统 | parameter partition、optimizer、checkpoint |
| TorchTitan | PyTorch 原生大模型训练参考 | 训练 loop、parallelize、checkpoint |
| JAX/XLA 生态 | 编译式数组程序与 sharding | mesh、partition spec、collective |

框架名称不代替配置。例如「使用 Megatron」仍需说明 TP/PP/CP/EP 大小、sequence parallel、distributed optimizer、microbatch 和 pipeline schedule。

---

## 源码阅读接口

选择一个框架时，沿以下路径阅读：

1. 配置如何建立 DeviceMesh/process group；
2. 模型构造时参数如何切分或包装；
3. forward 前何时 AllGather；
4. backward hook 何时触发 ReduceScatter/AllReduce；
5. optimizer 持有什么 shard；
6. checkpoint 如何保存逻辑 Tensor；
7. 日志如何归并 step、rank 和吞吐；
8. 错误如何跨 rank 传播。

这条路径比逐文件阅读更容易把框架实现映射到状态生命周期。

---

## 选型问题

- 单层是否需要 TP，还是 FSDP 已能满足容量；
- 目标模型组件和自定义 kernel 是否受支持；
- Checkpoint 是否允许改变 world size；
- MoE、长上下文和后训练算法需要哪些并行维度；
- 团队能否调试框架的图编译、通信与版本兼容；
- 目标云或集群的 launcher、存储和网络是否匹配；
- 性能收益是否由可复现 profile 证明。

---

## CPU 路线

PyTorch DDP/FSDP、DeepSpeed 的部分功能和 JAX CPU 后端可用于小规模语义实验。Megatron 的某些 fused kernel 或训练路径要求 GPU。CPU 路线的目标是理解初始化、分片、collective、state dict 和失败，不比较大模型吞吐。

## 官方资料

- PyTorch. [Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html).
- NVIDIA. [Megatron Core Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/).
- Microsoft. [DeepSpeed Documentation](https://www.deepspeed.ai/).
- PyTorch. [TorchTitan](https://github.com/pytorch/torchtitan).
