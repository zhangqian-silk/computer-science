# AI Infra 实验与项目路线

本分区提供实验规格，不提交具体实现。每个实验都先定义假设、正确性不变量和测量方法；支持 CPU 时提供 CPU 版本。GPU 版本用于验证 CUDA、Tensor Core、HBM、NCCL 和真实框架性能，不把 CPU 趋势直接外推。

---

## 路线总览

| 阶段 | 实验 | CPU 支持 | GPU 价值 |
| --- | --- | --- | --- |
| 模型正确性 | 最小 Decoder、KV Cache | 完整支持 | 验证低精度与 fused kernel |
| 性能基础 | GEMM/GEMV、Norm/Softmax、Roofline | 完整支持 | Tensor Core、HBM、Triton/CUDA |
| 分布式 | Collective、分片训练、故障 | 多进程支持 | NCCL、NVLink、RDMA |
| 推理 Runtime | Paged KV、连续批处理 | 模拟器完整支持 | 校准真实 iteration 成本 |
| 框架 Serving | 多框架基准、量化、Profiling | 小模型/llama.cpp 支持 | vLLM/SGLang/TRT-LLM 全能力 |
| 综合项目 | Mini Runtime、生产方案、源码追踪 | 核心控制面支持 | 完整数据面与性能证据 |

---

## 统一实验记录

每个实验报告包含：

- 假设与非目标；
- 环境：CPU/GPU、内存、互联、OS、driver、CUDA、框架 commit；
- 模型、tokenizer、shape、dtype；
- 自变量、控制变量、warm-up 和重复次数；
- 正确性不变量与误差阈值；
- 延迟分位数、吞吐、内存和 profile；
- 预期现象、实际结果、反例与结论边界；
- 完整命令和原始结果位置。

当前只补充设计，未执行实验，不应在文档中填写虚构数值。

---

## 推荐顺序

[模型与 Runtime 实验](./model-runtime-labs.md) → [加速器与分布式实验](./accelerator-distributed-labs.md) → [框架与服务实验](./framework-serving-labs.md) → [综合项目](./projects.md)

前四个 CPU 实验完成后即可开始读推理引擎源码；有 GPU 时再将相同正确性基线迁移到加速 kernel 和多卡执行。
