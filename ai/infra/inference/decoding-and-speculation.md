# 解码、约束与 Speculative Decoding

解码器把 logits 变成实际输出序列。温度与截断定义要采样的分布，约束定义合法集合，停止规则定义何时结束；Speculative Decoding 则尝试以更少目标模型迭代实现这一分布。只有满足正确接受和修正公式的版本才保持目标分布，不是所有草稿验证策略都无偏。

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

::: info 符号与约定
$p(v)$ 是给定同一前缀后目标模型的采样分布，$q(v)$ 是草稿分布，均已经应用相应解码处理。草稿采到 $v$ 后，以 $\min(1,p(v)/q(v))$ 接受；只有 $q(v)>0$ 的 token 才可能由草稿提出。
:::

拒绝时，不能直接重新从 $p$ 采样，而要从剩余质量归一化得到的分布采样：

$$
r(v)=\frac{\max(p(v)-q(v),0)}
{\sum_u\max(p(u)-q(u),0)}
$$

若 p=q，拒绝概率为零，无需计算这个分母为零的分支；实现仍应避免因舍入进入无效残差归一化。

接受分支给 token $v$ 的概率质量为 $q(v)\min(1,p(v)/q(v))=\min(p(v),q(v))$；拒绝分支补上 $\max(p(v)-q(v),0)$，两者之和恰为 $p(v)$。这是无偏性的核心，不是「多数时候两个模型意见一致」的经验解释。

例如两 token 分布 $p=(0.8,0.2)$、$q=(0.5,0.5)$，接受贡献为 $(0.5,0.2)$，拒绝概率为 0.3，剩余分布只能选第一个 token，最终恢复 $(0.8,0.2)$。若拒绝后直接采 $p$，最终反而变成 $(0.74,0.26)$。

多个草稿按前缀顺序验证；首次拒绝后，后续草稿依赖了错误前缀，不能直接保留。全部接受时可以利用额外目标分布再取一个 token。Leviathan 与 Chen 等人的 2023 年工作分别给出相关推测验证算法；速度仍要扣除草稿、验证、回滚和调度成本。

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

---

## 参考文献

- Leviathan, Y. et al. (2023). [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192).
- Chen, C. et al. (2023). [*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318).
