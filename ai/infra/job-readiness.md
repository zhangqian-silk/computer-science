# AI Infra 岗位能力路线

本页把学习深度分成解释、实现和验证三层。能够解释状态与公式是基础，能够修改一条执行路径是实践，能够用目标 workload 证明质量与性能才是岗位证据。这里是课程能力地图，不是对所有团队岗位要求的统一声明。

---

## 方向与能力深度

| 能力 | 所有方向 | 推理 Engine | Serving 平台 | 训练 Runtime |
| --- | --- | --- | --- | --- |
| Transformer/MoE/量化 | 必须掌握 | 深入 | 工作理解 | 深入 |
| Tensor shape/数值/内存 | 必须掌握 | 深入 | 工作理解 | 深入 |
| GPU/CUDA/Triton | 能读 profile | 深入 | 工作理解 | 深入 |
| Collective/拓扑 | 必须掌握 | TP/EP 深入 | 部署诊断 | 全面深入 |
| KV/Batch/Decode | 必须掌握 | 全面深入 | 容量与路由 | 工作理解 |
| PyTorch Distributed | 工作理解 | 模型集成 | 基本理解 | 全面深入 |
| API/Kubernetes/SLO | 工作理解 | Engine 接口 | 全面深入 | 作业平台 |
| Profiling/Benchmark | 必须掌握 | Kernel 到端到端 | 服务与容量 | Step 到集群 |

「工作理解」表示能解释接口、读日志并与该方向协作；「深入」表示能阅读源码、修改实现并设计验证。

---

## 四个里程碑

### 里程碑 1：模型可计算

完成 Tokenization、Transformer、Norm/FFN、MHA/GQA、数值和 Tensor 内存。能够手算参数、shape、KV 和训练状态。

**证据**：最小 Decoder 与 KV Cache CPU 实验设计可按规格实现；能解释 cached 与 uncached logits 的正确性不变量。

### 里程碑 2：性能可解释

完成 GPU 执行、Roofline、精度、kernel、编译图、profiling 与 Collective。

**证据**：能从一条 timeline 区分 CPU launch、计算、带宽和通信瓶颈；完成 GEMM/Softmax/Collective 实验报告模板。

### 里程碑 3：训推 Runtime 可追踪

推理方向完成请求、Prefill/Decode、KV、Batch、量化、分布式与框架；训练方向完成状态、混合精度、并行、Checkpoint 与框架。

**证据**：锁定一个框架 commit，画出请求或训练 step 的对象、rank、collective 与状态生命周期。

### 里程碑 4：系统可上线和排障

完成 API、路由、Kubernetes、容量、安全、故障和综合项目。

**证据**：提供 SLO、benchmark、容量、发布、故障注入、回退和成本报告；结论绑定硬件、版本与 workload。

---

## 推理岗位快速路径

若目标是尽快进入推理框架岗位，按以下顺序：

1. [推理请求生命周期](./inference/request-lifecycle.md)与[Prefill/Decode](./inference/prefill-decode.md)；
2. [KV Cache](./inference/kv-cache-management.md)与[Batch 调度](./inference/batching-scheduling.md)；
3. [GPU 性能模型](./accelerator/performance-model.md)、[Kernel](./accelerator/cuda-and-kernels.md)与[Profiling](./accelerator/profiling.md)；
4. [Collective](./distributed/collectives.md)与[分布式推理](./inference/distributed-inference.md)；
5. [框架案例](./inference/frameworks.md)、[排障](./inference/troubleshooting.md)与[Benchmark](./inference/benchmarking-capacity.md)；
6. [模型与 Runtime 实验](./labs/model-runtime-labs.md)和[综合项目 A/C](./labs/projects.md)。

模型基础不熟时先走首页完整路线；不要跳过 Tensor shape、数值和 GQA/MLA。

---

## 语言与源码能力

- Python/PyTorch：建立参考模型、训练和框架集成；
- C++：Runtime、算子接口、内存和并发；
- CUDA/Triton：GPU kernel 与性能验证；
- Go：Gateway、调度模拟器、压测和平台控制面；
- Shell/YAML：复现环境、容器和 Kubernetes；
- SQL/查询语言：分析 benchmark、trace 与服务指标。

推理 Engine 候选人至少应能在 Python 与 C++/CUDA 源码之间追踪一次 Tensor；Serving 候选人至少应能把 token/KV 成本映射到 API、队列与容量；训练 Runtime 候选人至少应能把每种 shard 映射到 Collective 和 Checkpoint。

---

## 面试与复盘问题

- 为什么 Prefill 和 Decode 的瓶颈可能不同？
- GQA 如何改变 KV 容量，为什么不保证同比提速？
- Paged KV 解决哪类碎片，block size 如何权衡？
- Continuous Batching 为什么提高吞吐，又如何伤害 TPOT？
- FSDP/ZeRO 的参数何时 AllGather，梯度何时 ReduceScatter？
- TP 为什么通常偏好节点内高速互联？
- 怎样证明量化既正确又真正加速？
- GPU utilization 高或低分别不能说明什么？
- 已经流式输出后 worker 失败，为什么不应默认透明重试？
- 如何建立一个可比较、可复现的推理 benchmark？

能回答问题只是起点。高质量答案应包含 shape、状态、成本模型、边界和验证证据。

---

## 岗位要求参考

下列岗位页面仅作外部阅读线索，未在本轮核验是否仍开放；不作为当前招聘结论。选择方向时应读目标团队的实际职责，并把每条要求映射为可以展示的代码、运行记录或诊断报告。推理、训练与平台共享基础，但深入的接口不同。

- OpenAI. [Software Engineer, Model Inference](https://openai.com/careers/software-engineer-model-inference-san-francisco/).
- OpenAI. [Training Performance Engineer](https://openai.com/careers/training-performance-engineer-san-francisco/).
- OpenAI. [Software Engineer, Workload Enablement](https://openai.com/careers/software-engineer-workload-enablement-san-francisco/).
- OpenAI. [Software Engineer, Model Deployment](https://openai.com/careers/software-engineer-model-deployment-chatgpt-engineering-london-uk/).
