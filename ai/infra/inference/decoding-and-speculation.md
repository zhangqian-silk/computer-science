# 解码、约束与 Speculative Decoding

模型输出 logits 后，解码器应用温度、惩罚、截断、约束和随机数，选择 token 并判断是否停止。Speculative Decoding 不改变目标分布，而是让便宜的草稿过程提出多个候选，再由目标模型批量验证，以减少串行目标模型 iteration。

<SpeculativeDecodingExplorer />

---

## Logits 处理顺序

常见流水线包含：

1. 取得当前序列最后位置 logits；
2. 应用允许/禁止 token 的结构约束；
3. 应用 repetition、presence、frequency 等 penalty；
4. 调整 temperature；
5. 应用 top-k、top-p 或 min-p 等过滤；
6. 按概率采样或选择最大值；
7. 更新 RNG、约束状态与停止匹配器。

处理顺序会改变分布，API 必须固定语义。分布式词表切分时，可以选择 gather 完整 logits，或在 shard 上完成部分 top-k/归约；后者要证明与单卡语义一致。

---

## 停止与结构约束

停止条件包括 EOS、最大 token、字符串/字节序列、deadline 和外部取消。停止字符串可能跨 token 边界，流式输出需要保留足够尾部，避免先发送后才发现应截断。

JSON grammar、正则或有限状态机约束通常通过 mask 非法 token 实现。约束状态属于序列运行时状态；beam、回滚和 speculative reject 都必须同步更新。

---

## Speculative Decoding

草稿模型提出 $k$ 个 token，目标模型一次前向验证这些位置。若多个候选被接受，一次昂贵 iteration 可以推进多个 token；拒绝时从目标分布修正采样，保持目标分布正确。

收益受以下因素共同决定：

- 草稿生成成本；
- 候选接受率；
- 验证 batch 的目标模型效率；
- KV 写入、回滚与调度开销；
- 当前并发是否已经让目标模型充分利用硬件。

高并发吞吐场景下，额外草稿计算可能不如低并发延迟场景划算。

---

## 其他草稿来源

- 独立小模型；
- 同一模型的早退层或多 token head；
- n-gram/前缀匹配；
- 多候选树与并行验证。

它们共享「提案—验证—接受/回退」接口，却有不同状态和质量保证。必须说明是否严格保持目标模型分布，还是近似解码。

---

## CPU 路线

CPU 可用小模型或合成概率分布验证 sampling、RNG、约束状态和 speculative acceptance 的统计正确性。还可用成本模型扫描草稿成本与接受率的盈亏点。真实加速需在目标硬件和并发下测量。

## 参考文献

- Leviathan, Y. et al. (2023). *Fast Inference from Transformers via Speculative Decoding*.
- Chen, C. et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*.
