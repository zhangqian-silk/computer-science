# Profiling：从请求延迟定位到 Kernel

Profiling 是一条可证伪的诊断链：先确定慢的是哪种请求，再定位它在等待什么，最后才进入具体 kernel。CPU 采样、GPU 时间线、硬件计数器和服务日志各有观测盲区，不能用其中一个百分比替代全链路解释。

---

## 从「GPU 利用率低」恢复因果链

假设服务的 TTFT 增大，同时 GPU 利用率下降，可以提出至少三个互斥程度不同的假设：

| 假设 | 应看到的证据 | 可以排除它的证据 |
| --- | --- | --- |
| CPU 未及时提交 | GPU kernel 间有空洞，主机线程忙于 tokenize 或调度 | GPU 队列持续非空 |
| 数据/网络等待 | 前向前出现传输或等待，输入队列供应不足 | 已准备 batch 充足且无传输等待 |
| 小 kernel 难以饱和 | kernel 连续但规模小，设备执行单元利用不充分 | 热点主要是大 GEMM 且算力接近上界 |

随后只改变一个变量，例如预先 tokenize 以隔离 CPU 前端，或固定 batch 以隔离调度。目标不是让 GPU utilization 变成 100%，而是让相关请求的延迟或 Goodput 改善。

主机 API 返回也不等于 GPU 已执行完毕。测 GPU 时间应使用相应设备事件或明确同步边界；每个算子后都同步又会破坏原本的重叠，所以 microbenchmark 和端到端测量须分开。

一份可学习的诊断记录应写出：原始症状、候选原因、用于排除的观测、唯一改动、正确性检查和重跑结果。没有重跑结果时称为假设或实验设计，不应写成性能优化已成功。

---

## 分层方法

| 层级 | 主要问题 | 证据 |
| --- | --- | --- |
| 服务 | 慢在排队、Prefill、Decode 还是网络 | TTFT、TPOT、queue time、trace |
| Runtime | CPU 是否及时提交工作，是否有同步 | CPU/GPU timeline、NVTX |
| 算子 | 哪些 shape 和算子占时间 | framework profiler |
| Kernel | 受计算、内存、occupancy 还是依赖限制 | hardware counters |
| 分布式 | Collective 是否进入关键路径 | NCCL trace、网络计数器 |

采样器、timeline 和计数器回答的问题不同。计数器采集可能显著扰动程序，应在可控范围内缩小采集目标。

---

## 基本流程

1. 固定模型、输入输出长度、并发、精度和版本；
2. 先 warm-up，分离编译、权重加载与稳定执行；
3. 记录端到端指标和资源利用率；
4. 用 NVTX 或等价标记切分 Tokenize、Queue、Prefill、Decode、Sample；
5. 检查 CPU 空洞、同步和数据复制；
6. 选择关键 kernel 采集深入计数器；
7. 修改一个变量并重新执行相同 workload；
8. 同时比较正确性、平均值与尾延迟。

若只优化 microbenchmark，而端到端时间不变，应重新检查 Amdahl 定律和测量层级。

---

## 常见误判

- GPU utilization 是时间窗口内忙碌比例，不等于计算单元满载；
- allocated memory、reserved memory 和设备总占用口径不同；
- 单 kernel 加速可能被额外转换、同步或通信抵消；
- 首次请求包含加载、JIT 和图捕获，不能与稳态混合；
- 平均延迟掩盖排队、长请求和错误重试造成的尾部；
- profiler 本身会改变时序。

---

## CPU 路线

CPU 可使用 PyTorch Profiler、Linux perf、火焰图和 eBPF，观察线程、cache miss、分支、系统调用和 NUMA。无 GPU 时仍可完成服务分段、调度和 CPU kernel 分析；GPU 时间线与硬件计数器需在目标设备补测。

---

## 参考资料

- NVIDIA. [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
- NVIDIA. [*Nsight Compute Documentation*](https://docs.nvidia.com/nsight-compute/).
- PyTorch. [*Profiler Recipes*](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
