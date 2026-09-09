# 分布式通信

本分区把分布式问题拆成三层：Collective 定义数学结果，拓扑定义搬运路径，运行时定义何时执行与何时可消费结果。先用四个 CPU 进程验证结果，再把同一个数据流映射到 GPU 集群；不要从性能参数反推正确性。

| 页面 | 核心问题 |
| --- | --- |
| [Collective](./collectives.md) | AllReduce、AllGather、ReduceScatter、AllToAll 分别改变什么状态 |
| [拓扑与网络](./topology-and-network.md) | PCIe、NVLink、InfiniBand、RDMA 如何形成分层带宽 |
| [重叠、启动与故障](./overlap-launch-and-failures.md) | 如何组织异步通信、进程生命周期与失败语义 |

学习时应先画出每个 rank 持有什么数据、操作后需要什么数据，再选择 Collective。直接从框架参数名称反推通信，容易忽略实际分片布局。

---

## 接到训练与推理

读完应能解释：为什么 AllGather 不能替代求和，为什么均匀 rank 平均不一定等于 token 平均，以及为什么一个 rank 的权限或进程故障会让其他 rank 等待。随后进入[训练并行](../training/parallelism.md)研究状态归属，或进入[分布式推理](../inference/distributed-inference.md)研究逐 token 延迟。
