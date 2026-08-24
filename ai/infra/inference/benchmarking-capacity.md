# 推理基准测试与容量规划

推理 Benchmark 必须模拟目标流量并同时测量正确性、延迟、吞吐和资源。离线最大 batch token/s、单请求 latency 与在线 open-loop Goodput 回答不同问题，不能互相替代。

<InferenceMetricExplorer />

---

## 指标定义

| 指标 | 定义 | 主要影响因素 |
| --- | --- | --- |
| TTFT | 请求到首 token 的时间 | queue、tokenize、Prefill、stream |
| TPOT | 首 token 后每个输出 token 平均时间 | Decode iteration、调度、通信 |
| ITL | 相邻输出 token 间隔 | iteration 波动与输出背压 |
| E2E | 请求到结束 | 输入、输出长度和排队 |
| Token throughput | 单位时间处理/生成 token | batch、硬件、模型 |
| Request throughput | 单位时间完成请求 | 长度分布 |
| Goodput | 满足指定 SLO 的有效吞吐 | 延迟目标与过载策略 |

TPOT 常按单请求输出阶段平均，ITL 保留逐 token 分布；报告时应说明是否包含首 token、网络和 detokenization。

---

## Workload 模型

必须记录：

- 输入/输出长度联合分布，而不是只有平均值；
- 到达过程：closed-loop 还是 open-loop；
- 并发、QPS、burst 与租户比例；
- sampling、stop、beam/speculation；
- prefix 重复率、adapter 和 multimodal 比例；
- warm/cold、编译、模型加载是否计入。

Closed-loop 客户端等待前一请求完成再发送，会在服务变慢时自动降低压力，可能掩盖排队崩溃。容量测试通常还需要 open-loop 到达过程。

---

## 容量估算

首先计算静态内存：权重、runtime buffer 和安全余量。剩余显存转换为 KV block 容量。若平均在途序列 KV 为 $\bar{M}_{KV}$，粗略并发上界为：

$$
C_{memory}\lesssim\frac{M_{available}}{\bar{M}_{KV}}
$$

它只是内存上界。延迟约束可能更早限制并发。通过压力曲线找到 QPS 增大时 TTFT/TPOT/Goodput 的拐点，再结合故障余量和副本维护确定可承诺容量。

---

## 对比模板

每次结果至少保存：

| 类别 | 字段 |
| --- | --- |
| Model | repo、revision、参数、context、tokenizer/template |
| Runtime | framework、commit、启动参数、kernel 后端 |
| Hardware | CPU、GPU、数量、内存、互联、NUMA |
| Software | OS、driver、CUDA、NCCL、compiler/container |
| Precision | weight、activation、KV、quantization |
| Traffic | 数据集、长度、到达、并发、sampling |
| Result | TTFT/TPOT/ITL/E2E 分位数、throughput、Goodput、errors、memory |
| Quality | logits/任务指标/输出有效性 |

---

## CPU 路线

CPU 可完成端到端 Benchmark：选择适合 CPU 的小模型或 llama.cpp 量化模型，使用相同请求生成器记录全部指标。也可用 mock engine 验证 open-loop 调度器。CPU 容量结论只适用于记录的 CPU/ISA/线程与后端，不与 GPU 峰值直接比较。

## 参考资料

- MLCommons. *MLPerf Inference Benchmark*.
- vLLM. *Benchmarking Suite*.
