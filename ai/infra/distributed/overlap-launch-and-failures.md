# 计算通信重叠、进程启动与故障

分布式执行需要同时满足依赖正确和参与者一致。异步 API 允许主机继续工作，但不保证结果已经可读；一个 rank 失败也不是其他 rank 能局部重试的普通网络错误。本页把重叠、启动与失败放在同一个运行时生命周期中理解。

---

## 计算与通信重叠

以三层网络为例，反向先产生第 3 层梯度，再产生第 2、1 层梯度。第 3 层梯度已就绪时，可以通信它，同时计算较早层；不能在梯度产生之前「提前通信」，也不能在通信完成之前更新该参数。

Bucket 过大会等待更多梯度才能启动；过小则启动次数增多。理想时间线是各 bucket 尽早就绪，并让最后一个通信尾巴尽可能短。测量 exposed communication 应看计算结束后仍需等待多久，而不是把所有通信事件时长求和。

两条 GPU stream 仍可能争用 SM、HBM 或互联。事件在时间线上重叠但各自变慢时，端到端可能没有收益。实验应比较不重叠基线、同一模型的重叠运行和正确性，不把 `async=True` 当成加速证据。

DDP 常把梯度按 bucket 组织：某个 bucket 的梯度计算完成后立即开始 AllReduce，反向传播继续计算更早层。理想时间不是计算与通信之和，而接近两者较大者：

$$
T_{step}\gtrsim\max(T_{compute},T_{communication})
$$

真正实现重叠需要：

- 依赖关系允许通信提前开始；
- 使用独立 stream 或异步 API；
- 通信不与计算争抢到完全相同的资源瓶颈；
- bucket 不过大也不过小；
- 消费结果前正确同步。

只看到 timeline 上两段重叠不代表零成本，HBM、NVLink 或 SM 资源竞争仍可能拉长两者。

---

## 启动与映射

Global rank 在整个作业内唯一，local rank 通常用于节点内设备选择；process-group rank 则依组而定。World size 改变后，旧 shard 编号不能直接解释成新组的设备归属。最少日志上下文是作业、节点、global rank、step、操作序号与首个错误时间。

分布式作业需要确定 world size、global/local rank、master endpoint、device mapping 和 process group。集群调度器负责分配节点与设备，框架 launcher 再创建进程。

常见故障包括：

- 两个进程绑定同一设备；
- 节点间环境或模型制品不一致；
- 某个 rank 没有进入相同 Collective；
- timeout 只报告表面等待，没有保留最早失败；
- 日志缺少 rank、host、step 和 collective sequence。

---

## 失败语义

Collective 中一个 rank 丢失时，其他 rank 通常不能假设本轮操作已经完成。默认的安全语义是让整个一致性组失败，保留可诊断信息，再由更高层从已确认的 Checkpoint 重新启动。

弹性成员变更、局部重试或冗余状态会引入新的协议和状态成本。只有明确的恢复时间目标、故障频率和大规模作业损失证据，才应把它们设为必需能力；否则先采用超时、快速失败与作业级重启。

---

## CPU 路线

多进程 CPU 后端足以模拟 rank 退出、顺序不一致、timeout、日志关联和作业级重启。可先在 CPU 上建立故障注入和诊断规范，再在 GPU/NCCL 环境检查异步错误与设备状态。

---

## 参考资料

- PyTorch. [*Distributed Communication Package*](https://docs.pytorch.org/docs/stable/distributed.html).
- NVIDIA. [*NCCL Troubleshooting*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).
