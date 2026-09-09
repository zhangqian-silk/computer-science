# 综合项目规格

综合项目按三种证据组织：自己实现一条运行路径、把既有引擎部署成服务、从源码与 profile 推导一项改进。以下是项目规格而非已实现项目；选择其一深入，比同时搭三个没有验收的框架更能检验理解。

---

## 项目 A：Mini LLM Inference Runtime

### 目标

实现一个单机 Runtime，重点展示状态与调度，不要求自研高性能模型 kernel。

### 必需模块

- 请求/序列状态机；
- tokenizer 与 mock/小模型 runner adapter；
- prefill/decode 阶段；
- block-based KV allocator；
- continuous batching 与 chunked prefill；
- greedy/sampling/stop；
- streaming、取消和有界队列；
- TTFT、TPOT、KV 与调度 metrics。

### CPU 版本

使用 Go 实现控制面和调度器，模型 runner 可调用 Python/PyTorch CPU、llama.cpp 或确定性 mock。CPU 版本是完整项目，不是占位符。

### GPU 扩展

替换为 CUDA/Triton 或现有 GPU model runner，增加 CUDA Graph、PagedAttention 和 profile。GPU 扩展不能破坏 CPU 测试中的状态不变量。

### 验收

交付目录应让陌生读者找到启动说明、固定测试输入、模型或 mock 契约、调度/内存 trace 和预期结果。先跑单请求，再跑不同长度请求，最后取消其中一条；剩余请求的状态与输出不得改变。

Mock runner 的结果只证明控制面行为，不能证明真实 Transformer logits 正确；接入真实小模型后应补共同前缀的数值比较。把两类证据标开，避免演示看起来完整但数据面从未验证。

随机事件 property test 无 KV 泄漏；输出与参考 greedy decode 一致；取消幂等；不同流量下能解释吞吐—延迟曲线；提供架构图、benchmark 和故障报告。

---

## 项目 B：生产级推理服务方案

### 目标

选择 vLLM 或 SGLang，设计从不可变模型制品到 Kubernetes 服务、压测和发布回退的完整方案。

### 交付物

- 模型/runtime 版本清单；
- 单副本和多副本容量报告；
- gateway、路由、admission 与租户配额；
- readiness、监控、trace 和告警；
- 灰度发布、排空与回退；
- 取消、OOM、worker 失败演练；
- 成本、风险与不采用复杂 KV 恢复的理由。

### CPU 版本

使用 llama.cpp server 或 mock engine 部署同一控制面，完成协议、Kubernetes、路由、监控与故障演练。GPU 版再补真实 engine、KV、NCCL 和容量数据。

---

## 项目 C：主流框架源码追踪与改进提案

### 目标

锁定一个 vLLM 或 SGLang release/commit，追踪单请求从 API 到 GPU worker 和返回的完整路径。

### 交付物

- 进程/线程、对象所有权与状态图；
- scheduler、KV manager、model runner 的关键接口；
- 一次真实或 CPU mock trace；
- 选定瓶颈的 profile 证据；
- 一个范围明确的改进提案，包含正确性测试、性能实验和回退。

CPU 环境无法执行 GPU worker 时，可以完成 frontend、scheduler、KV 逻辑和单元测试阅读，并对 GPU 路径做静态追踪；性能结论必须标为待 GPU 验证。

---

## 项目完成标准

为每个项目保留一项失败案例及其收敛过程。例如先重现跨块缓存读取错误，再给出最小输入、根因与修正后对照；或者证明某次分片降低稳态内存却提高峰值，最终选择更小的预取。能说明被证据否定的方案，也是完整工程判断。

三个项目均需让陌生读者在不依赖当前对话的情况下回答：系统保证什么、不保证什么；数据和状态由谁拥有；哪里可能失败；怎样复现；结论适用于哪些硬件、版本和 workload。
