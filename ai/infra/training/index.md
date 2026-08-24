# 训练 Runtime

训练 Runtime 把数据批次、模型前向、损失、反向传播和优化器更新组织为可恢复、可扩展的状态机。分布式框架之间的名称不同，但都必须回答：每个状态属于哪个 rank、何时需要通信、怎样保存，以及失败后从哪里继续。

```mermaid
flowchart LR
	D["Data Batch"] --> F["Forward"]
	F --> L["Loss"]
	L --> B["Backward"]
	B --> C["Gradient Communication"]
	C --> O["Optimizer Step"]
	O --> S["Scheduler / Step State"]
	S --> D
	S -.-> K["Checkpoint"]
```

---

## 主题地图

| 页面 | 核心问题 |
| --- | --- |
| [训练循环与内存](./memory-and-loop.md) | 参数、激活、梯度和优化器状态何时存在 |
| [混合精度](./mixed-precision.md) | 不同状态使用何种 dtype，怎样检测数值失败 |
| [数据管线](./data-pipeline.md) | 样本怎样被切分、打包、加载和恢复 |
| [并行策略](./parallelism.md) | DP、TP、PP、CP、EP、FSDP、ZeRO 怎样组合 |
| [Checkpoint 与恢复](./checkpoint-and-recovery.md) | 哪些状态必须一致保存，失败后保证什么语义 |
| [训练框架](./frameworks.md) | PyTorch、Megatron-Core、DeepSpeed 等如何映射到统一概念 |
| [后训练 Runtime](./post-training-runtime.md) | SFT、偏好优化和在线生成训练增加哪些系统组件 |
| [Scaling 与资源预算](./scaling-and-compute.md) | 参数、训练 token、FLOPs、时间和成本如何连接 |
| [训练性能](./performance.md) | 怎样测量 MFU、吞吐、通信与 Straggler |

推荐先在单进程 CPU 或单 GPU 上理解状态生命周期，再使用多进程 CPU 验证 Collective 和分片语义，最后进入多 GPU 性能测试。
