# 推理框架案例地图

::: info 实现核验
本页给出源码阅读地图，不固化易变的类名和调用细节。实际学习时应锁定 release 或 commit，并在笔记中记录核验日期。历史 PagedAttention 设计文档不等同于 vLLM 当前全部实现。
:::

推理框架是已学机制的可运行组合。阅读时追踪同一请求的 token、KV、调度状态与输出，不按项目名判断「先进程度」。本页描述概念接口，不提供未经目标 release 核验的功能保证；任何可执行命令都应来自锁定版本。

---

## 框架定位

| 框架 | 主要定位 | 建议观察路径 |
| --- | --- | --- |
| vLLM | 通用 LLM engine 与 OpenAI-compatible API 服务 | API、scheduler、KV manager、worker 内的 model runner |
| SGLang | 结构化语言程序与高性能 serving runtime | frontend → scheduler → Radix cache → model worker |
| TensorRT-LLM | NVIDIA GPU 优化 kernel、engine 与 executor | model convert/build → runtime/executor → kernels |
| llama.cpp | C/C++ 本地与 CPU/多后端推理 | GGUF load → graph build → backend scheduler → decode |
| NVIDIA Dynamo | 分布式推理编排与数据面 | frontend/router → planner → engine worker → KV transport |

它们不是简单的同类替代。例如 Dynamo 可以编排其他 engine；TensorRT-LLM 更强调 NVIDIA 执行栈；llama.cpp 对 CPU、边缘和 GGUF 生态尤其重要。

表中路径是阅读导航，不是精确调用顺序；vLLM 的 worker 通常持有 model runner，scheduler 使用 KV manager，不能把它们误画成反向包含关系。

---

## vLLM 源码阅读任务

一条最小追踪应建立以下对应关系：

| 逻辑对象 | 需要找到的真实职责 | 不能混淆 |
| --- | --- | --- |
| 请求 | 输入、输出上限与停止状态 | HTTP 连接不等于独立模型序列 |
| 调度结果 | 本轮 token、序列与块表 | 请求数不等于 token 数 |
| 缓存管理器 | 分配、引用、释放 | 物理 block 不等于逻辑 token |
| 模型执行器 | 张量准备、图/算子调用 | 数学机制与 kernel 后端不同 |
| 输出处理 | 采样、解码、流式事件 | token 已生成不等于字节已发送 |

先用单请求、关闭非必要优化建立调用链，再增加两个不同长度请求，观察短请求退出后长请求的状态是否保持。随后才加入 prefix cache、量化或推测执行；每项变化应能落回上表的一项所有权或计算变化。

1. 从 offline/online API 找到一次请求进入 engine 的入口；
2. 观察输入处理与 sequence/request 对象；
3. 跟踪 scheduler 如何形成一次 model execution；
4. 找到 KV cache manager 的 allocate、append、free；
5. 跟踪 worker/model runner 如何准备 Tensor 和执行模型；
6. 观察 sampling、输出处理与流式返回；
7. 对照一次 profile，关联 CPU span 和 GPU kernel。

完成标准是能画出对象所有权和线程/进程边界，而不是只知道 PagedAttention 名称。

---

## SGLang 阅读任务

重点观察 RadixAttention/prefix cache、请求调度、chunked prefill、speculative decoding、quantized KV、并行与 PD/EPD disaggregation。对每项能力分别记录：缓存 key、调度预算、失败边界与指标。

---

## TensorRT-LLM 阅读任务

区分模型转换/构建、kernel/plugin、runtime executor 和 serving 集成。关注 in-flight batching、paged KV、quantization、CUDA Graph、speculative decoding、LoRA、guided decoding 与多 GPU。任何支持声明都要绑定具体 release、GPU 架构和模型。

---

## llama.cpp 阅读任务

从 GGUF metadata 与 tensor load 开始，观察 computation graph、backend buffer、quantized kernel、KV cache、batch/decode 与 server。它提供完整 CPU 路线，可用于理解权重布局、线程、SIMD、NUMA 和本地服务。

---

## 跨框架对比方法

不要将「特性存在于文档」写成「本模型本硬件组合已支持」。功能验证矩阵应交叉记录模型架构、权重格式、KV 格式、并行方式、约束解码与 adapter，并标注已运行、仅文档声明或未知。历史论文说明设计动机，当前源码说明实现，实际运行说明组合可用，三者不能互相替代。

同一模型和 tokenizer 下固定：输入/输出 token、sampling、精度、并发、硬件和服务协议。先验证输出/质量，再比较 TTFT、TPOT、Goodput、内存和运维复杂度。某框架在一个模型上的最优配置不能推导为普遍排名。

---

## 官方资料

- vLLM. [Architecture Overview](https://docs.vllm.ai/en/latest/design/arch_overview/).
- SGLang. [Documentation](https://docs.sglang.io/).
- NVIDIA. [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html).
- ggml-org. [llama.cpp](https://github.com/ggml-org/llama.cpp).
- NVIDIA. [Dynamo Overall Architecture](https://docs.nvidia.com/dynamo/dev/knowledge-base/overview).
