# Collective：分布式 Tensor 的数据变换

Collective 是一组 rank 共同参与的数据操作。它们不是普通 RPC：所有参与者必须以兼容的顺序、shape 和 dtype 进入操作，否则可能错误或长期等待。

<CollectiveDataFlowExplorer />

---

## 常用操作

设有 $P$ 个 rank，每个 rank 持有一段数据。

| 操作 | 输入与输出关系 | 典型场景 |
| --- | --- | --- |
| Broadcast | 一个 rank 的数据复制到全部 rank | 初始状态同步 |
| AllReduce | 全部输入归约，每个 rank 得到完整结果 | DDP 梯度同步、TP 局部和 |
| ReduceScatter | 归约后按 rank 分片输出 | 分片梯度或输出 |
| AllGather | 收集各 rank 分片，每个 rank 得到完整数据 | 参数/激活重建 |
| AllToAll | 每个 rank 向每个 rank 发送不同分片 | MoE token dispatch |

AllReduce 可以逻辑分解为 ReduceScatter 加 AllGather，但实际库会按消息大小和拓扑选择 ring、tree 或其他算法。

---

## 成本模型

简化的通信时间可写为：

$$
T\approx \alpha\times N_{steps}+\frac{\text{bytes}}{BW_{effective}}
$$

$\alpha$ 表示每步启动或网络延迟，$BW_{effective}$ 是考虑协议、拓扑和竞争后的有效带宽。小消息更受延迟影响，大消息更受带宽影响。模型不包含排队、拓扑不均、软件调度等全部因素，只用于形成初始判断。

---

## 正确性不变量

- 参与 rank 集合一致；
- collective 顺序一致；
- Tensor 元素数和 dtype 满足接口约束；
- 归约操作符合数值语义；
- 异步 handle 在数据被消费前已经完成；
- stream 依赖被正确建立。

浮点归约顺序会随算法和拓扑变化，结果可能不逐位相同。测试应使用数值容差和训练级统计，而不是把 bitwise 一致作为默认要求。

---

## CPU 路线

PyTorch Gloo 或 MPI 可以在多进程 CPU 上验证 Collective 语义、分片布局、死锁和小规模成本模型。单机可使用 loopback 或多进程模拟 rank。CPU 后端无法代表 NCCL、NVLink 或 GPUDirect RDMA 的带宽，但很适合先调通协议。

## 参考资料

- NVIDIA. [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).
- MPI Forum. *MPI Standard*.
