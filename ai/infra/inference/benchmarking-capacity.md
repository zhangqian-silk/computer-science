# 推理基准测试与容量规划

Benchmark 是一个定义清楚的实验，不是运行命令后挑最大的数字。离线批处理问设备能处理多少工作，单请求问交互延迟，开放到达压力问过载下有多少请求仍满足 SLO；三者必须分别报告。

<InferenceMetricExplorer />

---

## 指标定义

令 $t_0$ 为客户端发起时刻，$t_1$ 为首 token 可见时刻，$t_n$ 为最后 token 可见时刻，输出数为 $n$。则 TTFT 为 $t_1-t_0$；当 $n>1$ 时，平均后续间隔为 $(t_n-t_1)/(n-1)$。只有一个输出时 TPOT 无样本，不能除以零或当作测得 0 ms。

每请求平均间隔再跨请求平均，与将全部间隔合并平均不同：前者每请求等权，后者长输出权重更大。报告 p99 也要说明对象是请求、迭代还是 token 间隔。

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

稳定状态下，Little 定律连接平均在途数 $L$、到达速率 $\lambda$ 与平均驻留时间 $W$：$L=\lambda W$。它是长期平均关系，不是尾延迟保证；过载下不稳定队列不能套一个固定 $W$ 推容量。

例如同为每秒 10 个请求，平均驻留 2 秒约需 20 个在途请求容量，驻留 20 秒则约需 200 个。输出长度分布会改变驻留与 KV 生命周期，因此只按 QPS 配置显存不够。

Goodput 应把满足约定 TTFT、TPOT 和正确性条件的完成请求计入分子，观测时长计入分母；失败、拒绝和超时仍需单列，不可从原到达流量中消失。

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

---

## 参考资料

- MLCommons. *MLPerf Inference Benchmark*.
- vLLM. *Benchmarking Suite*.
