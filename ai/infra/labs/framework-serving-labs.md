# 框架与在线服务实验设计

本组实验检验「组合后的服务是否仍正确且满足目标」。模型、框架、量化、调度与网络都参与结果；先固定比较协议，再分别观察正确性、容量、延迟与失败。未在目标框架运行的矩阵单元标记为未验证，而不是补一个推测值。

---

## 实验 9：跨框架推理

为每个框架保存启动命令和不可变 revision，先比较 tokenizer ID、模板与固定输入 logits，再启用 greedy 生成。若制品转换引入量化，就不再是纯框架比较，必须作为独立实验组。

分别测首次加载、首次请求与稳态请求，不混合平均。先用单请求，再用同一开放到达轨迹压测；任何一个组合不支持目标模型时，如实记录失败原因，不通过更换模型偷偷补齐表格。

**目标**：在 Hugging Face Transformers、vLLM、SGLang、llama.cpp，以及硬件允许时 TensorRT-LLM 上运行同一模型家族。

**控制变量**：model revision、tokenizer/template、sampling、输入/输出 token、精度、硬件。若框架使用不同制品格式，记录转换过程和校验。

**CPU 版**：Transformers CPU 与 llama.cpp/GGUF；可用小模型完成 OpenAI-compatible server。**GPU 版**：vLLM/SGLang/TensorRT-LLM，启用能力以具体版本为准。

**结果**：正确性、加载时间、内存、TTFT、TPOT、Goodput、错误率和可运维性说明。

---

## 实验 10：量化质量—性能矩阵

校准数据与测试数据分离；固定输入不仅覆盖普通句子，还包含数字、罕见符号、长上下文和结构输出。记录每种方法的位宽、group、scale、校准量及实际 kernel。

如果只有权重文件变小而峰值内存未降低，检查是否在运行时完整反量化；如果 logits 误差小但生成分叉，先在共同前缀上定位差异，再用任务统计判断是否可接受。

选择 FP32/BF16 基线和至少两种目标量化。比较固定 logits、perplexity/任务、结构输出、长上下文、模型大小、峰值内存和服务指标。

**CPU 版**：llama.cpp 的 GGUF 多位宽是完整路线。**GPU 版**：按框架/硬件支持选择 AWQ、GPTQ、FP8 或 KV quantization。不要为了填满表格使用未优化 fallback。

---

## 实验 11：端到端 Profiling

每类 workload 固定到达过程与长度分布，保存请求 trace 和设备时间线的时间对应关系。只选择一个可验证瓶颈改动，例如去掉重复 tokenize；重跑后比较原主指标及可能恶化的指标。

将「profile 观察到什么」「推测为什么」「改动后是否支持推测」分成三栏。一次看见 GPU 空洞不能直接确认 CPU 是唯一原因。

构造短 prompt/短输出、长 prefill、长 decode 和高并发四类 workload。用 span 标出 tokenize、queue、prefill、decode、sample、stream。

**CPU 工具**：PyTorch Profiler、perf、火焰图。**GPU 工具**：PyTorch Profiler、NVTX、Nsight Systems，必要时 Nsight Compute。

输出一份瓶颈报告：证据、假设、改动、前后对比、Amdahl 边界和未解决问题。

---

## 实验 12：故障、取消与过载

注入点应覆盖排队前、已分配 KV、kernel 已提交、部分流已发送和正常完成同时收到取消。每次记录终止原因、可见事件、剩余配额与所有块的引用。

通过条件不仅是进程没崩溃，还包括无重复输出、无越权执行、资源最终回到基线、客户端能区分完整与部分结果。对于不承诺续传的协议，明确失败是正确行为。

注入慢客户端、排队超时、执行中取消、allocator OOM、worker 退出、模型版本不匹配和网络中断。

**CPU 版**：mock engine 能完整验证 API 事件、幂等释放、限流和 trace。**GPU 版**：验证 kernel 已提交时的取消边界、worker group 失败和设备 OOM 诊断。

通过标准是没有资源泄漏、错误分类稳定、部分流不透明重试、过载时队列有界。
