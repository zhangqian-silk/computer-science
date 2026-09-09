# AI 文档全文核验记录

核验对象是当前工作树中的完整正文，而非仅检查 diff。核验期间暂停 PR 创建与合并，核验完成后由用户决定发布。此记录区分「全文已读」「公式和示例已核」「来源证据」「运行验证」，不以构建通过替代内容正确。

当前状态：93 篇正文均已完整通读，已完成问题修正；最终运行验收结果见文末。文件未修改不等于未读，文件被修改也不等于核验通过。

正文快照 SHA-256：`d6c3669a371234dcb0ff3607337c138378422b8caf88b51e385bcdfe1dec82ee`。算法为按相对路径排序，依次散列「路径、NUL、文件字节、NUL」，仅含 93 篇内容 Markdown，不含 AGENTS.md 或本记录；后续改动不自动继承此核验结论。

---

## 已确认问题

| 编号 | 问题 | 处理状态 |
| --- | --- | --- |
| F01 | SGNS 与 SPPMI/SVD 被等同，内积/对数损失界与零项处理错误，GloVe 数量级错误 | 已修正；驻点、极限及舍入数值复算 |
| F02 | 相邻表示页无条件传播平移 PMI 等式；CBOW 集合丢失重复项 | 已修正；补噪声条件与按位置计数 |
| F03 | 显式位置被说成唯一顺序来源，固定对称局部 mask 的排列条件遗漏 | 已修正；全可见/联合置换与反例检验 |
| F04 | 位移方向与因果索引混用，GPT 停止图分支错误 | 已修正；逐位置对照并删除绕过停止检查的边 |
| F05 | N-gram 折扣范围、零历史及句首计数策略遗漏 | 已修正；归一化与各阶目标集合复算 |
| F06 | RNN Jacobian 次序、LSTM 未来隐藏路径和导数形状不严谨 | 已修正；12 参数标量 LSTM 有限差分检验 |
| F07 | SSM 漏直通项/短卷积缓冲，BERT 生成边界过强 | 已修正；含 D 的递推—卷积比较 |
| F08 | log-sum-exp 最大值及行/列向量约定遗漏 | 已修正；形状与稳定形式核对 |
| F09 | 向量兼容性、ANN 偶然返回被过度概括 | 已修正；区分训练兼容与排序近似 |
| F10 | RoPE 统一平移被说成必改分数，插值闭区间越界、门控梯度可能为零未说明 | 已修正；旋转恒等与输入边界检查 |
| F11 | 污染导致 PPL 的方向写反，Precision 分母和长上下文归因过强 | 已修正；概率和指标口径核对 |
| F12 | Agent 伪代码将模型 final 直接当完成，预算/授权失败语义不完整 | 已修正；补完成检查与失败边界 |
| F13 | 激活二次项、MFU/HFU 重算、HBM/Global 概念、缓存所有者及公平性前提不完整 | 已修正；分层成本与模型假设明确 |
| F14 | HTTP 成功误作流完成、量化 scale 清单遗漏、框架包含关系和 Schema 能力过强 | 已修正；接口/制品/部署界限明确 |
| F15 | KV fork/COW 实验取消时没有明确递减全部持有引用 | 已修正；逐块所有者推演 |
| F16 | 长度缩小时窗口标签仍显示旧值，相对偏置符号与正文相反，PMI 热图白字对比不足 | 已复现旧版失败并修正，回归通过；热图改用主题文字色 |

---

## 逐篇记录

以下记录「已完整阅读并完成当前内容检查」，不表示相关外部软件已经在 GPU 上执行。修正后的复核与来源限制在最终验收时汇总。

| 文档 | 已检查的核心内容 | 当前结论 |
| --- | --- | --- |
| `ai/foundations/index.md` | 阅读依赖、入口职责 | 全文已读，未发现实质错误 |
| `ai/foundations/learning-problem-map.md` | 六维分类、目标/架构/系统边界 | 全文已读，需收紧兼容性表述的跨页关联 |
| `ai/foundations/math-notation.md` | 形状、归约、梯度记法 | 全文已读，F08 |
| `ai/foundations/numerical-computing.md` | dtype、log-sum-exp、量化和误差 | 全文已读，F08 |
| `ai/foundations/tensor-and-memory.md` | stride、别名、字节与形状 | 全文已读，PyTorch Tensor Views 核对 |
| `ai/foundations/neural-network-basics.md` | CE 导数、链式法则、SGD/Adam、数据切分 | 全文已读，补统计结论前提 |
| `ai/foundations/nlp-history.md` | 发表版本、架构演进与因果约束 | 全文已读，F03 |
| `ai/representation/index.md` | 分区关系和向量兼容性 | 全文已读，F02/F09 |
| `ai/representation/tokenization.md` | BPE/Unigram/WordPiece、分词与词表代价 | 全文已读，补往返/覆盖条件 |
| `ai/representation/embedding.md` | 查表与梯度、几何、上下文化 | 全文已读，F02/F08 |
| `ai/representation/word2vec.md` | 样本、负采样导数、噪声与层次 softmax | 全文已读，F02 |
| `ai/representation/embedding-matrix-factorization.md` | 完整目标重组、驻点、截断、GloVe 数值与零项 | 全文已读，F01 |
| `ai/representation/text-embedding.md` | InfoNCE、池化、MaxSim、论文改造轴 | 全文已读，F08/F09 |
| `ai/representation/multimodal-alignment.md` | 对称损失、ViT patch、连接器与冻结梯度 | 全文已读，补 patch/负例条件 |
| `ai/representation/vector-retrieval.md` | 索引与编码误差、归一化等价、过滤 | 全文已读，F09 |
| `ai/model/n-gram.md` | 全部平滑、回退、KN 推导及计数伪代码 | 全文已读，F05 |
| `ai/model/nplm.md` | 有序窗口、共享参数、训练与生成 | 全文已读，区分层次模型和采样近似 |
| `ai/model/rnn.md` | 三步递推、BPTT、截断、生成索引 | 全文已读，F06 |
| `ai/model/lstm.md` | 门控前向、全部反向路径、参数与状态 | 全文已读，F06 |
| `ai/model/seq2seq.md` | 概率读出、教师强制、搜索反例与缓存 | 全文已读，补标量读出与反例条件 |
| `ai/model/transformer.md` | 输入/位置、Block、mask、样本位移和 KV | 全文已读，F03/F04 |
| `ai/model/gpt.md` | CLM、归一化口径、采样与终止图 | 全文已读，F04 |
| `ai/model/bert.md` | MLM/NSP、任务头和家族变体 | 全文已读，F07 |
| `ai/model/state-space-model.md` | ZOH、卷积展开、scan、固定状态 | 全文已读，F07 |
| `ai/model/index.md` | 模型载体、训练/推理接口对照 | 全文已读，F04 |
| `ai/mechanism/index.md` | 机制分类与模型接口 | 全文已读，未发现实质矛盾 |
| `ai/mechanism/attention.md` | 权重/Value、缩放假设、梯度、多头投影 | 全文已读，收紧独立性与投影类型 |
| `ai/mechanism/self-attention.md` | mask、排列等变、缓存及复杂度 | 全文已读，F03 |
| `ai/mechanism/attention-head-sharing.md` | 映射、字节算例、MLA 投影吸收 | 全文已读，代数与字节复算 |
| `ai/mechanism/positional-encoding.md` | 正弦平移、相对偏置与因果边界 | 全文已读，F03/F16 |
| `ai/mechanism/rope.md` | 旋转展开、共同平移与缓存相位 | 全文已读，F10 |
| `ai/mechanism/long-context.md` | 插值端点、频率扩展、ALiBi 与评估 | 全文已读，F10 |
| `ai/mechanism/normalization-and-residual.md` | LN/RMS、Pre/Post Jacobian、Dropout 期望 | 全文已读，解析式核对 |
| `ai/mechanism/feed-forward-and-gating.md` | 形状、SiLU、局部导数、等参数 FLOPs | 全文已读，F10 |
| `ai/mechanism/moe.md` | Top-k、容量、分配归一化、通信 | 全文已读，算例与条件核对 |
| `ai/mechanism/lora.md` | 低秩参数、初始梯度、合并与 QLoRA | 全文已读，参数及导数核对 |
| `ai/mechanism/sparse-attention.md` | 边数、两跳传播、因果全局 Key | 全文已读，图连通性及近似边界核对 |
| `ai/evaluation/index.md` | 能力主张、冻结协议、分层证据 | 全文已读，未发现实质矛盾 |
| `ai/evaluation/embedding-geometry.md` | 近邻、降维、各向异性和探针 | 全文已读，几何与任务边界核对 |
| `ai/evaluation/language-model-evaluation.md` | NLL/PPL、滑窗、校准与污染 | 全文已读，F11 |
| `ai/evaluation/retrieval-evaluation.md` | Recall/Precision/RR/NDCG、空标注 | 全文已读，F11 |
| `ai/evaluation/generation-evaluation.md` | BLEU/ROUGE、来源支持、配对评审 | 全文已读，数值与聚合口径核对 |
| `ai/evaluation/long-context-evaluation.md` | 长度/位置矩阵、对照、可用长度集合 | 全文已读，F11 |
| `ai/agent/index.md` | 全文、静态图片、状态机、伪代码和工具边界 | 全文已读，F12 |
| `ai/index.md` | 分区与路线、全站覆盖范围、来源使用 | 全文已读，导航与正文范围核对 |
| `ai/infra/index.md` | 系统层次、路线、历史来源 | 全文已读，理论与实测分离 |
| `ai/infra/job-readiness.md` | 能力矩阵、里程碑、岗位引用 | 全文已读；岗位是否开放不作当前结论 |
| `ai/infra/accelerator/index.md` | 单设备阅读链与 CPU/GPU 边界 | 全文已读，未发现实质矛盾 |
| `ai/infra/accelerator/gpu-architecture.md` | GEMM tile、存储层级、occupancy | 全文已读，F13 |
| `ai/infra/accelerator/performance-model.md` | Roofline、字节下界、Amdahl、时间叠加 | 全文已读，F13 |
| `ai/infra/accelerator/precision-and-tensor-cores.md` | 指数/尾数、存储/乘法/累加 | 全文已读，格式角色核对 |
| `ai/infra/accelerator/cuda-and-kernels.md` | Softmax、全 mask 行、融合与 spill | 全文已读，边界和测试方案核对 |
| `ai/infra/accelerator/compiler-and-graphs.md` | 捕获、编译、重放、别名和动态性 | 全文已读；具体版本执行未做 |
| `ai/infra/accelerator/profiling.md` | 时序、同步、计数器和单变量对照 | 全文已读，未将现象直接当根因 |
| `ai/infra/distributed/index.md` | 语义/拓扑/运行时三层 | 全文已读，未发现实质矛盾 |
| `ai/infra/distributed/collectives.md` | 各操作、Ring 字节与梯度平均 | 全文已读，成本式与归一化核对 |
| `ai/infra/distributed/topology-and-network.md` | 共享上联、GPU/NIC 亲和性与 RDMA | 全文已读；真实链路性能未测 |
| `ai/infra/distributed/overlap-launch-and-failures.md` | bucket 依赖、异步、rank 与组失败 | 全文已读，依赖及失败语义核对 |
| `ai/infra/training/index.md` | 一次更新闭环与分区链接 | 全文已读，未发现实质矛盾 |
| `ai/infra/training/memory-and-loop.md` | 累积、状态生命周期、重算与 OOM | 全文已读，F13 |
| `ai/infra/training/mixed-precision.md` | loss scale、unscale/clip、跨 rank 更新 | 全文已读；真实收敛未运行 |
| `ai/infra/training/data-pipeline.md` | packing、Attention/loss mask、预取游标 | 全文已读，恢复与目标语义核对 |
| `ai/infra/training/parallelism.md` | ZeRO 状态表、TP 乘法、PP bubble、CP/EP | 全文已读，F13 |
| `ai/infra/training/checkpoint-and-recovery.md` | 一致 step、异步快照、重分片与制品状态 | 全文已读，F14 |
| `ai/infra/training/frameworks.md` | 框架分层、参数生命周期、CPU 限制 | 全文已读；概念图不是功能组合验证 |
| `ai/infra/training/post-training-runtime.md` | DPO mask/参考概率、rollout 版本与供需 | 全文已读；未运行后训练作业 |
| `ai/infra/training/scaling-and-compute.md` | 6ND、幂律驻点、时间/GPU-hour | 全文已读，F13 |
| `ai/infra/training/performance.md` | 强弱扩展、最慢 rank、MFU/HFU | 全文已读，F13 |
| `ai/infra/inference/index.md` | 请求、KV、执行和释放接口 | 全文已读，分区链路核对 |
| `ai/infra/inference/request-lifecycle.md` | 三游标、取消、首 token 终止与释放 | 全文已读，F04 |
| `ai/infra/inference/prefill-decode.md` | o−1 轮、连接数、缓存长度、chunk | 全文已读，公式与交互数值核对 |
| `ai/infra/inference/kv-cache-management.md` | 页表寻址、引用、COW、MLA 范围 | 全文已读，F13 |
| `ai/infra/inference/batching-scheduling.md` | 两类预算、饥饿、不可拆分工作 | 全文已读，F13 |
| `ai/infra/inference/attention-kernels.md` | 在线 Softmax 分子/分母、全无效块 | 全文已读，分块反例复算 |
| `ai/infra/inference/quantization.md` | scale/zero、零组、误差及方法/格式 | 全文已读；未宣称实测加速 |
| `ai/infra/inference/decoding-and-speculation.md` | 接受质量、残差分布、p=q、状态回滚 | 全文已读，F13 |
| `ai/infra/inference/distributed-inference.md` | 副本/TP/PP、通信与 KV 粘性 | 全文已读，逻辑接口核对 |
| `ai/infra/inference/cache-and-disaggregation.md` | cache key、最长前缀、TTFT 交付边界 | 全文已读，缓存与传输假设核对 |
| `ai/infra/inference/model-serving-features.md` | adapter、路由逆映射、媒体及约束 | 全文已读，F14 |
| `ai/infra/inference/api-observability-reliability.md` | HTTP/流、配额、日志、错误类型 | 全文已读，F14 |
| `ai/infra/inference/benchmarking-capacity.md` | TPOT 分母、Little、Goodput 与容量 | 全文已读，均值与尾部口径核对 |
| `ai/infra/inference/troubleshooting.md` | 同前缀 logits、OOM 分类与归因 | 全文已读，假设和证据分离 |
| `ai/infra/inference/frameworks.md` | V1 组件归属、阅读顺序与兼容接口 | 全文已读，F14；未部署目标 release |
| `ai/infra/platform/index.md` | 平台与 engine 权责、发布案例 | 全文已读，未发现实质矛盾 |
| `ai/infra/platform/gpu-containers-kubernetes.md` | 驱动/容器、资源、探针与 gang | 全文已读；未部署真实集群 |
| `ai/infra/platform/deployment-and-routing.md` | 不可变制品、影子流、排空与回退 | 全文已读，旧 KV 不切新权重 |
| `ai/infra/platform/fleet-and-capacity.md` | group 容量、余量、启动时标与降级 | 全文已读；容量式只是估计 |
| `ai/infra/platform/security-and-multitenancy.md` | 输入、缓存、日志、内存与安全域 | 全文已读，F14 |
| `ai/infra/labs/index.md` | 设计/结果区分与统一实验记录 | 全文已读；实验未执行不算通过 |
| `ai/infra/labs/model-runtime-labs.md` | 四组实验的输入、不变量、指标与反例 | 全文已读，F04/F15 |
| `ai/infra/labs/accelerator-distributed-labs.md` | 形状扫描、kernel、collective、并行对照 | 全文已读；CPU 语义不替代 GPU 性能 |
| `ai/infra/labs/framework-serving-labs.md` | 同模型协议、量化、profile、故障注入 | 全文已读，控制变量和验收核对 |
| `ai/infra/labs/projects.md` | 项目边界、mock 与真实 runner 的证据 | 全文已读，交付/验收范围核对 |

---

## 来源证据与边界

- PyTorch 官方 Tensor Views 页面实际跳转到 `docs/2.14/tensor_view.html`，已读取关于共享存储、转置、reshape 和 contiguous 的正文；这只核验视图语义，不代表已执行该版本 PyTorch。
- GloVe 原论文已在作者站点 PDF 阅读器中实际核对：第 3 页 Table 1 的四列概率/比值，第 4 页式 (7)–(9) 的偏置、加权目标、零项与超参数。报告保留原表舍入值，不把它们伪装成重新实测的数据。
- SGNS 的驻点、零共现极限及非等价反例另做独立代数检查；不能将题名链接存在当成已验证每一项实验结论。
- ACL 的 PDF 下载曾超时；该失败不计证据。随后在 [Stanford 作者站点 PDF](https://nlp.stanford.edu/pubs/glove.pdf) 成功加载页面并逐页截图核对，未声称下载失败已经通过。
- vLLM 文档站点受到访问检查限制，改为读取官方仓库的 [Architecture Overview](https://github.com/vllm-project/vllm/blob/main/docs/design/arch_overview.md) 第 1–230 行。内容 blob SHA 为 `711e9a2668eb44ae6999cce30aae6472698c734f`，不是 commit SHA。V1 Process Architecture 与 Model Runner 段落确认 API、EngineCore、worker 和 runner 的关系；这不等于已部署某个 release。
- 本次不是重新运行所有论文实验，也不是对所有外链 HTTP 可达性的承诺。不能获取或未实际运行的版本/性能结论保留为条件或阅读任务；没有把它们升级为已验证事实。

---

## 交互组件核验

已完整检查 30 个实验组件、LearningLab、LearningRoadmap、实验说明注册表、重置助手及 BPE/Beam/调度算法源文件。重点核对：

- 概率与读取：AttentionRead、AttentionPattern、HeadSharing、PositionEncoding、NGramSmoothing；
- 表示与模型：Tokenization、VectorSimilarity、SPPMI、MultimodalAlignment、LanguageModelObjective、RecurrentState、StateSpaceScan、Seq2Seq、TransformerBlock；
- 参数与训练：Normalization、FeedForward、MoE、LoRA、TrainingMemory、ParallelismMesh、Collective；
- 运行与评估：BatchScheduler、PagedKV、PrefillDecode、SpeculativeDecoding、Quantization、Roofline、InferenceMetric、RetrievalMetrics、AgentRuntime。

数值输出分别按解析公式检查；手动示意流程不当作真实训练或网络执行。F16 的旧状态已在浏览器复现：长度缩到 4 后 input 值为 3、标签仍为「窗口：9」，而相对偏置原符号与正文相反。

---

## 最终验收

12 组独立数值/导数检查已通过：SGNS 驻点和零项极限、GloVe 舍入一致性、SGD/PPL、平滑/句首计数、Attention 有限差分、置换反例、RNN 连乘、LSTM 全部 12 个标量参数的有限差分、含 D 的 SSM、RoPE/FFN/LoRA、KV/Scaling/排序指标、推测采样概率质量。

最终执行结果：

- `npm run docs:build`：通过，仅有 JavaScript 分块超过 500 kB 的体积警告。
- `node /tmp/audit-ai-docs.mjs`：93 页、399 个本地链接/锚点、35 处组件嵌入核验通过；没有 KaTeX 错误或未渲染公式。
- `node /tmp/ai-math-review.mjs`：12 组数值/导数检查通过，包括 LSTM 全部 12 参数的有限差分。
- 浏览器：30 个实验组件及 2 个重复实例，共 32 实例；键盘调整、重置、解释展开、标签绑定与唯一 ID 检查通过。375/1280 宽度无页面级横向溢出，七分区导航与深色模式检查通过。
- 窗口回归：先确认旧版失败，再确认长度 4 时 Query/窗口/滑块均为 3；相对偏置显示 Q−K=-4。PMI 主题配色在 375/1280 截图中复查。
- `git diff --check`：通过；核验记录与实际目录为 93/93，无遗漏、无重复。

临时集成页已删除，不混入正式站点；临时脚本命令记录的是本轮执行证据，不加入日常 CI。真实训练、GPU 性能、部署和论文全部实验没有运行，不计为通过。用户已在核验完成后要求创建并合并 PR；实际发布状态以 GitHub PR 和部署记录为准。
