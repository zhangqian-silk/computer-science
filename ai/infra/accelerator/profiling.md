# Profiling：从请求延迟定位到 Kernel

Profiling 的目标是用时间线、计数器和调用关系排除错误假设。一个可靠分析从端到端 workload 开始，逐层缩小到服务阶段、CPU/GPU 时间线、算子和 kernel，而不是先盯住最耗时的一行代码。

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

## 参考资料

- NVIDIA. [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
- NVIDIA. *Nsight Compute Documentation*.
- PyTorch. *Profiler Recipes*.
