# 框架与在线服务实验设计

本页把相同模型和 workload 放入不同 Runtime，训练的是框架核验、压测、Profiling 和故障分析能力，而不是制造一张无条件排行榜。

---

## 实验 9：跨框架推理

**目标**：在 Hugging Face Transformers、vLLM、SGLang、llama.cpp，以及硬件允许时 TensorRT-LLM 上运行同一模型家族。

**控制变量**：model revision、tokenizer/template、sampling、输入/输出 token、精度、硬件。若框架使用不同制品格式，记录转换过程和校验。

**CPU 版**：Transformers CPU 与 llama.cpp/GGUF；可用小模型完成 OpenAI-compatible server。**GPU 版**：vLLM/SGLang/TensorRT-LLM，启用能力以具体版本为准。

**结果**：正确性、加载时间、内存、TTFT、TPOT、Goodput、错误率和可运维性说明。

---

## 实验 10：量化质量—性能矩阵

选择 FP32/BF16 基线和至少两种目标量化。比较固定 logits、perplexity/任务、结构输出、长上下文、模型大小、峰值内存和服务指标。

**CPU 版**：llama.cpp 的 GGUF 多位宽是完整路线。**GPU 版**：按框架/硬件支持选择 AWQ、GPTQ、FP8 或 KV quantization。不要为了填满表格使用未优化 fallback。

---

## 实验 11：端到端 Profiling

构造短 prompt/短输出、长 prefill、长 decode 和高并发四类 workload。用 span 标出 tokenize、queue、prefill、decode、sample、stream。

**CPU 工具**：PyTorch Profiler、perf、火焰图。**GPU 工具**：PyTorch Profiler、NVTX、Nsight Systems，必要时 Nsight Compute。

输出一份瓶颈报告：证据、假设、改动、前后对比、Amdahl 边界和未解决问题。

---

## 实验 12：故障、取消与过载

注入慢客户端、排队超时、执行中取消、allocator OOM、worker 退出、模型版本不匹配和网络中断。

**CPU 版**：mock engine 能完整验证 API 事件、幂等释放、限流和 trace。**GPU 版**：验证 kernel 已提交时的取消边界、worker group 失败和设备 OOM 诊断。

通过标准是没有资源泄漏、错误分类稳定、部分流不透明重试、过载时队列有界。
