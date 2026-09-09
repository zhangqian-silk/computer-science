# 训练框架地图

::: info 实现核验
本页描述稳定能力边界；具体 API 以使用时的官方文档和源码为准。框架选型记录必须包含版本、commit、硬件和启用的并行维度。
:::

训练框架应按层理解：张量与自动微分、进程组与分片、模型并行组件、训练循环和作业管理。不同项目覆盖范围可以重叠，因此本页不是产品排行榜，而是把实现映射到已学状态生命周期的阅读方法。以下为概念基线；未在目标环境运行的功能不视为已验证支持。

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

先选择一层普通线性层和一次 optimizer step，建立四条证据：参数何时完整、梯度在哪里归约、优化器更新哪一片、快照如何恢复逻辑 Tensor。然后才扩展到 MoE、长上下文和混合并行。若第一条简单路径尚不能解释，直接阅读整个仓库会被配置与优化分支淹没。

阅读记录至少包含仓库 commit、入口命令、配置、关键调用链、张量 shape/placement 与一次运行 trace。类名或目录路径可能变更，稳定的比较对象是上述状态契约；不要把某版本内部函数名当作长期公共 API。

CPU 路线需要逐功能确认：多进程 Collective、自动微分和部分分片语义可在 CPU 验证，但某些 FSDP 或融合训练路径要求加速器。若入口拒绝 CPU，应改做显式小张量分片实验，而不是宣称整个框架支持无 GPU 训练。

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

CPU 路线优先使用 DDP、普通自动微分与显式小张量分片模拟。FSDP、DeepSpeed、JAX 或 Megatron 的具体入口是否支持 CPU，需要逐版本、逐功能核验；不能把可模拟其语义写成整套训练路径已支持 CPU。实验目标是理解初始化、分片、Collective、state dict 和失败，不比较大模型吞吐。

---

## 官方资料

- PyTorch. [Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html).
- NVIDIA. [Megatron Core Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/).
- Microsoft. [DeepSpeed Documentation](https://www.deepspeed.ai/).
- PyTorch. [TorchTitan](https://github.com/pytorch/torchtitan).
