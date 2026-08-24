# 分布式通信

分布式训练和推理把一个逻辑 Tensor 切到多个进程与设备。正确性依赖所有 rank 对 shape、dtype、顺序和 process group 达成一致；性能则取决于消息大小、拓扑、并发通信和计算重叠。

| 页面 | 核心问题 |
| --- | --- |
| [Collective](./collectives.md) | AllReduce、AllGather、ReduceScatter、AllToAll 分别改变什么状态 |
| [拓扑与网络](./topology-and-network.md) | PCIe、NVLink、InfiniBand、RDMA 如何形成分层带宽 |
| [重叠、启动与故障](./overlap-launch-and-failures.md) | 如何组织异步通信、进程生命周期与失败语义 |

学习时应先画出每个 rank 持有什么数据、操作后需要什么数据，再选择 Collective。直接从框架参数名称反推通信，容易忽略实际分片布局。
