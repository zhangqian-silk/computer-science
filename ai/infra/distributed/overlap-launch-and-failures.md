# 计算通信重叠、进程启动与故障

分布式 Runtime 除了执行 Collective，还要建立 rank、设备和网络关系，协调异步 stream，并定义任一参与者失败时其他进程如何退出或恢复。

---

## 计算与通信重叠

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

## 参考资料

- PyTorch. *Distributed Communication Package*.
- NVIDIA. *NCCL Troubleshooting*.
