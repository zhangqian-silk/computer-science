# 模型与推理 Runtime 实验设计

这四个实验依次建立计算、缓存、存储所有权和调度的证据。它们是可实施的实验设计，尚不代表仓库提供了完整 Runtime；CPU 即可完成语义验证，GPU 只负责补充目标硬件上的成本结论。

---

## 实验 1：最小 Decoder-only Transformer

先固定小配置：词表 32、隐藏宽度 16、2 层、4 个 Query 头、每头 4 维、长度 8，采用明确的 Pre-Norm 与无 Dropout 配方。随机初始化即可验证结构，不需要训练到会说话。

按以下顺序执行：打印逐矩阵形状；对固定 ID 输入保存每层输出；只修改未来位置的 ID；比较此前位置 logits；最后将 batch 中两条输入交换，检查输出是否按同样顺序交换。相同数字输出无法证明因果性，修改不可见输入仍不变才是在检验依赖。

若结果失败，先检查目标位移、mask 行列方向与位置索引，不要用放宽容差掩盖明显结构差异。低精度迁移时另定误差阈值，不能要求所有 dtype 逐位相同。

**假设**：完整推理行为可以分解为 Tokenization、Embedding、Causal Attention、FFN、Norm、LM Head 和 Sampling。

**设计**：使用 PyTorch 构造小词表、2～4 层、可配置 MHA/GQA 的模型；不下载大模型。分别输出每层 shape、参数量和中间统计。

**正确性不变量**：

- causal mask 后未来 token 不影响当前位置 logits；
- Decoder 输入为目标序列右移并补 BOS，输入位置预测后一 token；
- eval 模式关闭 Dropout；
- 参数量与逐矩阵求和一致；
- greedy decode 在固定权重下可复现。

**CPU 版**：完整实现和测试。**GPU 版**：复用同一测试，增加 FP16/BF16 与 kernel profile。

---

## 实验 2：完整重算与 KV Cache

使用实验 1 的同一权重和同一预先固定生成前缀，分别逐步重算和缓存执行。每一步比较的是相同输入下的 logits，不是两条已采样分叉的文本。

记录位置、缓存长度、理论 KV 字节、实际张量字节与最大绝对误差。首输出由 Prefill 产生；测试输出长度为 1、跨块边界和多请求长度不同的情形。预测缓存路径更少重算，但小形状可能被框架开销淹没，这种反例是结果而不是实验失败。

**假设**：缓存解码与每步重算整个前缀在数值容差内产生相同 logits，同时减少重复工作。

**变量**：prompt 长度、输出长度、MHA/GQA、dtype、batch。

**指标**：每 token 延迟、累计时间、KV 字节、峰值内存和 logits 最大误差。

**预期现象**：缓存容量随历史长度线性增长；完整重算的重复计算随生成长度快速增加。小 CPU 模型上框架开销可能掩盖理论收益，应如实记录。

---

## 实验 3：Paged KV Allocator

先用固定轨迹：容量 4 块、每块 2 token；A 的 3 个 token 占两块，B fork 共享这两块；A 写第 4 个 token 前复制共享尾块。取消 B 时递减它持有的每个块引用：首块仍被 A 持有，旧尾块最后引用归零才回收。再加入固定种子的随机事件。

每次事件后输出 free list、所有者集合和逻辑块表。OOM 操作应保持之前状态不变；「失败后回收一部分原有块」不满足原子准入。生成最小失败轨迹比单独报告随机测试失败更有诊断价值。

**假设**：固定 block 分配可以降低为每请求预留最大连续区的浪费，并支持 fork/share/cancel。

**设计**：使用 Go 或 Python 实现纯 allocator 模拟器，不执行模型。输入随机事件流：admit、append、fork、finish、cancel、OOM。

**检查**：

- free + owned block 总数守恒；
- 无 block 同时处于 free 与 owned；
- refcount 与所有者一致；
- copy-on-write 后共享前缀不被修改；
- 重复 release 不破坏 free list；
- OOM 不产生部分提交。

**CPU 版**：完整支持。**GPU 版**：可选地把 block 映射到真实 KV Tensor，验证 paged attention 读取。

---

## 实验 4：Continuous Batching 模拟器

先复现交互组件的 A/B/C/D 轨迹，再引入非等时成本：让 Prefill 成本随 chunk token 数变化、Decode 成本随活跃序列与历史长度变化。保持同一到达轨迹，比较策略；不要让两个策略使用不同随机流量。

至少输出三类曲线：队列与在途请求、已用/分配 KV、请求级 TTFT/ITL。若平均吞吐提升却长请求长期等待，应把饥饿计入失败，不把未完成请求从统计分母中删除。

**假设**：iteration-level 调度能提高变长流量的资源利用率，但策略会改变 TTFT、TPOT 与公平性。

**设计**：使用离散事件模拟请求到达、prefill token、decode token、KV block 和每轮成本。比较 static batch、FIFO continuous batch、decode-priority、chunked prefill。

**输入**：短/长混合长度分布、Poisson 与 burst 到达、不同 KV 容量。

**输出**：TTFT/TPOT/E2E 分位数、Goodput、队列、KV 时间线、饥饿请求和拒绝率。

**CPU 版**：完整支持。**GPU 版**：用真实 engine profile 校准每类 iteration 成本，不改变调度不变量。
