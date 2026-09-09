# 拓扑与网络：通信经过哪些链路

拓扑是数据实际经过的资源图。同一逻辑 AllReduce 可以经过 GPU 互联、PCIe、CPU NUMA 和网络交换机；任一共享链路都可能成为瓶颈。设备数说明容量，链路与竞争关系才解释通信代价。

---

## 分层拓扑

```mermaid
flowchart TB
	G0["GPU 0"] <-->|"NVLink/NVSwitch"| G1["GPU 1"]
	G0 <-->|"PCIe"| C0["CPU / NUMA 0"]
	G1 <-->|"PCIe"| C1["CPU / NUMA 1"]
	C0 --> N0["NIC 0"]
	C1 --> N1["NIC 1"]
	N0 <-->|"InfiniBand / Ethernet Fabric"| RN["Remote Node"]
	N1 <-->|"InfiniBand / Ethernet Fabric"| RN
```

数据路径可能经过 GPU peer link、PCIe switch、CPU interconnect、NIC 和网络交换机。错误的 GPU/NIC affinity 会让流量绕经更慢路径或跨 NUMA 节点。

### 带宽不是端口速率的简单相加

假设两张 GPU 各自拥有高带宽本地连接，但共同通过一条较慢的上联访问远端。两个传输并发时，上联容量仍需共享，不能把每条本地链路速率相加当作远端带宽。分析应标出方向、全双工能力和共享段。

延迟也不同于带宽。大张量可把启动成本摊薄；逐层的小 TP 归约即使字节少，也会反复支付网络往返和同步成本。将高频组内通信放到快互联，是减少暴露延迟，而非认为跨节点通信永远不可用。

GPUDirect RDMA 的「直接」是减少主机 staging，不是绕过权限、内存注册、设备同步和网络拥塞。实验需同时记录 GPU/NIC 亲和性、消息大小、参与进程组与背景流量；只报告网卡标称速率无法复核。

---

## 关键技术

- GPUDirect P2P 允许 GPU 之间更直接地访问或复制；
- GPUDirect RDMA 让 NIC 与 GPU memory 直接传输，减少主机 staging；
- RDMA 降低 CPU 参与和复制，但仍需正确注册内存与管理队列；
- NVLink/NVSwitch 提供节点内高带宽互联，具体带宽随硬件代际和拓扑变化；
- InfiniBand 与 RoCE 都可支持 RDMA，但拥塞、无损网络和运维模型不同。

技术名称不能代替实测。应分别测量单向带宽、双向带宽、Collective 带宽、延迟和并发竞争。

---

## 并行策略如何使用拓扑

高频、细粒度的 Tensor Parallel 通信通常优先放在最快的节点内互联；Pipeline Parallel 跨 stage 传输的频率与消息形状不同，可以延伸到节点间；Data Parallel 的梯度同步可通过 bucket 与反向计算重叠。

MoE 的 AllToAll 对全局带宽和负载均衡敏感。上下文并行还会按序列切分 Attention 状态。最优映射取决于模型、序列长度、并发和拓扑，不存在固定的「先 TP 后 PP」万能配置。

---

## CPU 路线

CPU 集群可以研究 NUMA、NIC affinity、TCP/RDMA、MPI 和层次化 Collective。在只有单机时，可通过多个进程和限制带宽/延迟验证调度与通信模型。GPU Direct 和 NVLink 行为必须在相应硬件上补测。

通用网络前置知识见 [IP](../../../fundamentals/network/ip.md)、[TCP](../../../fundamentals/network/tcp.md)和[网络基础](../../../fundamentals/network/network.md)。

---

## 参考资料

- NVIDIA. [*GPUDirect RDMA Documentation*](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html).
- NVIDIA. *NVLink and NVSwitch Technical Overview*.
