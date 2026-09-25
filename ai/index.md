# 人工智能

本知识库沿一个完整问题展开：原始对象怎样表示，训练怎样得到参数，机制怎样组合成模型，模型怎样运行与参与任务，以及哪些证据足以支持能力结论。基础、表示、机制、模型、Infra、评估与 Agent 是相互关联的视角，不是人工智能领域的互斥分类，也不是单线替代的技术年代。

<LearningRoadmap />

第一次学习先从分类与训练基础进入；已有基础则沿目标任务选择路径，并在遇到前置缺口时回到主讲页面。交互实验隔离一个机制或成本关系，正文负责补齐它省略的假设、版本和应用边界。

---

## 知识分区

| 分区 | 核心问题 | 起点 |
| --- | --- | --- |
| [基础](./foundations/index.md) | 数学符号如何阅读，神经网络如何学习，NLP 方法为何演化到 Transformer | [数学与符号约定](./foundations/math-notation.md) |
| [表示](./representation/index.md) | 离散对象如何进入连续空间并形成可比较的几何结构 | [Embedding](./representation/embedding.md) |
| [机制](./mechanism/index.md) | Attention、位置、稀疏路由和低秩适配如何计算 | [Attention](./mechanism/attention.md) |
| [模型](./model/index.md) | 从 N-gram 到 Transformer 与状态空间模型，完整建模对象如何变化 | [模型地图](./model/index.md) |
| [AI Infra](./infra/index.md) | GPU、通信、训练和推理 Runtime 如何把模型变成可运行系统 | [AI Infra 总览](./infra/index.md) |
| [评估](./evaluation/index.md) | 概率、检索、生成和长上下文能力如何被可靠测量 | [评估设计](./evaluation/index.md) |
| [Agent](./agent/index.md) | 如何把模型、工具、状态和控制流组成可执行系统 | [Agent 系统](./agent/index.md) |

---

## 六条阅读路线

尚未建立整体概念时，先读[学习问题与分类](./foundations/learning-problem-map.md)，分清任务、表示、训练信号、架构和系统，再选择下面的路线。路线是遍历知识的入口，不意味着其他分区可以省略。

### 语言模型主线

[NLP 方法演化](./foundations/nlp-history.md) → [N-gram](./model/n-gram.md) → [NPLM](./model/nplm.md) → [RNN](./model/rnn.md) → [LSTM](./model/lstm.md) → [Seq2Seq](./model/seq2seq.md) → [Transformer](./model/transformer.md) → [BERT](./model/bert.md) / [GPT](./model/gpt.md)

这条路线关注上下文如何从固定窗口、递归状态演化为全局交互，以及训练目标如何从任务专用转向预训练。

### 表示与检索主线

[Embedding](./representation/embedding.md) → [word2vec](./representation/word2vec.md) → [文本嵌入](./representation/text-embedding.md) → [向量检索](./representation/vector-retrieval.md) → [检索评估](./evaluation/retrieval-evaluation.md)

这条路线把「向量是什么」「向量如何学到」「向量怎样部署」「系统怎样验收」分开处理。

### 长上下文与系统主线

[Self-Attention](./mechanism/self-attention.md) → [位置表示](./mechanism/positional-encoding.md) → [长上下文](./mechanism/long-context.md) → [稀疏注意力](./mechanism/sparse-attention.md) / [状态空间模型](./model/state-space-model.md) → [长上下文评估](./evaluation/long-context-evaluation.md) → [Agent 系统](./agent/index.md)

这条路线同时检查能力与代价，避免把名义窗口长度、理论复杂度或工具数量直接等同于系统效果。

### 推理框架岗位主线

[Tokenization](./representation/tokenization.md) → [Transformer](./model/transformer.md) → [Tensor 与内存](./foundations/tensor-and-memory.md) → [GPU 执行](./infra/accelerator/gpu-architecture.md) → [性能模型](./infra/accelerator/performance-model.md) → [Prefill 与 Decode](./infra/inference/prefill-decode.md) → [KV Cache](./infra/inference/kv-cache-management.md) → [Batch 调度](./infra/inference/batching-scheduling.md) → [推理框架](./infra/inference/frameworks.md) → [实验与项目](./infra/labs/index.md)

这条路线优先覆盖推理引擎岗位的完整数据面。读者应同步完成 CPU 模拟实验；具备 GPU 后再补 CUDA/Triton、NCCL 和目标框架性能证据。

### 训练框架主线

[神经网络训练基础](./foundations/neural-network-basics.md) → [数值计算](./foundations/numerical-computing.md) → [Collective](./infra/distributed/collectives.md) → [训练循环与内存](./infra/training/memory-and-loop.md) → [混合精度](./infra/training/mixed-precision.md) → [训练并行](./infra/training/parallelism.md) → [Checkpoint](./infra/training/checkpoint-and-recovery.md) → [训练框架](./infra/training/frameworks.md) → [训练性能](./infra/training/performance.md)

这条路线用状态归属与通信解释 DDP、FSDP、ZeRO、TP、PP、CP 和 EP，避免只记配置项。

### 检索增强与 Agent 主线

[文本嵌入](./representation/text-embedding.md) → [向量检索](./representation/vector-retrieval.md) → [检索评估](./evaluation/retrieval-evaluation.md) → [GPT](./model/gpt.md) → [生成评估](./evaluation/generation-evaluation.md) → [Agent 系统](./agent/index.md)

这条路线把外部证据、语言生成和真实动作分成独立接口。检索到文档不代表回答忠实，模型输出工具调用也不代表动作已经授权或成功执行。

---

## 理解框架

- 完整模型由表示、机制、训练目标和推理接口共同定义；
- 方法演化描述各类模型之间的因果关系，公式与推演解释具体计算过程；
- 指标必须结合数据集、切分方式和任务边界理解，单个分数不是模型能力的完整结论；
- 复杂度通常描述主导项，实际性能仍受批量大小、硬件、kernel 与内存访问影响。

---

## 如何使用正文、交互与论文

先读主题的输入输出与核心问题，再操作实验中的一个变量，预测结果后对照数值与解释。最后回到推导，检查图中省略了哪些维度、参数或系统成本。实验的可计算结果不是论文实测，能拖动滑块也不等于已经理解；应能够离开组件手算一个小例子。

论文按「提出的问题、关键改造、比较条件和结论边界」融入专题。历史页连接技术关系，机制页保留主要推导，模型页组合完整训练和推理接口。原始版本与后续变体分开标记，不把某个模型、框架或评测分数当作整个领域的最终结论。

当前主题重心是语言表示、序列模型及其运行系统。它不是对所有人工智能分支的完整覆盖；未设专题的视觉生成、强化学习基础等不能从本目录标题推断已经讲全。
