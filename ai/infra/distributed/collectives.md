# Collective：分布式 Tensor 的数据变换

Collective 定义一组进程共同执行的张量变换。Rank 是组内编号，process group 决定谁参与；通信库再选择算法与路径。学习时先写「操作前各自有什么，操作后各自需要什么」，再考虑 Ring、Tree 或网络带宽。

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

以四个 rank 各有完整梯度向量为例，sum AllReduce 得到四份相同的逐元素和；ReduceScatter 则让每个 rank 只得到和向量的一段。接着 AllGather 这些段，逻辑结果才等于 AllReduce。AllGather 自身不执行求和，AllToAll 也不是广播同一份数据。

若每个 rank 的完整张量为 $M$ 字节，理想 Ring AllReduce 有两段，各含 $P-1$ 步，每步发送 $M/P$ 字节。每 rank 发出字节为 $2(P-1)M/P$；每步延迟 $\alpha$、有效单向带宽 $BW$ 时：

$$
t_{\text{ring}}\approx2(P-1)\alpha+\frac{2(P-1)M}{P\,BW}
$$

这里统计发送，不把接收再算一遍。小张量可能受启动步数限制，大张量可能受带宽限制；真实算法还会按拓扑分层和流水化。

梯度平均是另一层约定。每 rank 等量样本的平均 loss，梯度和通常除以 $P$；若有效 token 数不同，简单按 rank 平均会偏重较短的 batch，应按全局有效 token 数归一化。通信结果正确并不自动意味着训练目标正确。

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

---

## 参考资料

- NVIDIA. [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html).
- MPI Forum. [*MPI Standard*](https://www.mpi-forum.org/docs/).
