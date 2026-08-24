# 模型与推理 Runtime 实验设计

本页定义四个从模型语义走向推理调度的实验。它们均可先在 CPU 完成，不要求训练大模型。

---

## 实验 1：最小 Decoder-only Transformer

**假设**：完整推理行为可以分解为 Tokenization、Embedding、Causal Attention、FFN、Norm、LM Head 和 Sampling。

**设计**：使用 PyTorch 构造小词表、2～4 层、可配置 MHA/GQA 的模型；不下载大模型。分别输出每层 shape、参数量和中间统计。

**正确性不变量**：

- causal mask 后未来 token 不影响当前位置 logits；
- 训练序列右移一位；
- eval 模式关闭 Dropout；
- 参数量与逐矩阵求和一致；
- greedy decode 在固定权重下可复现。

**CPU 版**：完整实现和测试。**GPU 版**：复用同一测试，增加 FP16/BF16 与 kernel profile。

---

## 实验 2：完整重算与 KV Cache

**假设**：缓存解码与每步重算整个前缀在数值容差内产生相同 logits，同时减少重复工作。

**变量**：prompt 长度、输出长度、MHA/GQA、dtype、batch。

**指标**：每 token 延迟、累计时间、KV 字节、峰值内存和 logits 最大误差。

**预期现象**：缓存容量随历史长度线性增长；完整重算的重复计算随生成长度快速增加。小 CPU 模型上框架开销可能掩盖理论收益，应如实记录。

---

## 实验 3：Paged KV Allocator

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

**假设**：iteration-level 调度能提高变长流量的资源利用率，但策略会改变 TTFT、TPOT 与公平性。

**设计**：使用离散事件模拟请求到达、prefill token、decode token、KV block 和每轮成本。比较 static batch、FIFO continuous batch、decode-priority、chunked prefill。

**输入**：短/长混合长度分布、Poisson 与 burst 到达、不同 KV 容量。

**输出**：TTFT/TPOT/E2E 分位数、Goodput、队列、KV 时间线、饥饿请求和拒绝率。

**CPU 版**：完整支持。**GPU 版**：用真实 engine profile 校准每类 iteration 成本，不改变调度不变量。
