# 后训练 Runtime

后训练改变了监督与数据生成方式，因此也改变执行系统。SFT 消费给定回答，DPO 消费偏好对和参考概率，在线策略优化还需要生成、评分与版本同步。不能把所有过程统称为「再跑一遍微调」，也不能仅因有奖励就断言需要同一套 RL 架构。

---

## 工作负载类型

对一个偏好对 $(x,y^+,y^-)$，DPO 会比较策略与参考模型的序列对数概率差。序列概率是回答 token 的对数概率求和，padding 和 prompt 的 loss mask 必须一致；否则比较可能反映模板和长度处理差异，而非偏好本身。参考分数可在固定参考权重和固定预处理下预计算，但换任一版本后旧分数不再兼容。

在线 rollout 还应保存生成时策略版本、必要的旧 log-prob、终止原因和有效 token mask。训练器不能仅凭回答文本重建当时策略概率；拿新权重重算得到的是另一分布。

Ouyang 等人的工作展示监督微调、奖励建模和策略优化的组合；Rafailov 等人的 DPO 将特定偏好优化问题改写为直接训练目标；DeepSeekMath 的 GRPO 用同题样本组构造相对信号。这些贡献分别改变监督流程、目标推导和优势估计，不能仅按「需要几个模型」判断算法等价。

| 阶段 | 主要计算 | 额外状态或服务 |
| --- | --- | --- |
| SFT | teacher-forced forward/backward | 指令数据、loss mask |
| DPO/偏好优化 | chosen/rejected 多次前向与反向 | reference log-prob 或参考模型 |
| Reward Model | 序列评分训练 | 偏好对、标量 head |
| 在线 RL | 生成、奖励、优势估计、策略更新 | rollout worker、旧策略、经验数据 |
| Distillation | 教师生成或 logits 与学生训练 | 教师服务、缓存或离线数据 |

SFT 的系统形态最接近预训练。DPO 增加成对序列和参考概率。在线 RL 在生成与训练之间建立闭环：推理引擎产生 rollout，训练器消费经验并更新权重，二者的吞吐必须匹配。

---

## 生成—训练闭环

若生成端每秒产出 $r_g$ 个可用回答，训练端每秒消费 $r_t$ 个，长期 $r_g>r_t$ 会积累队列与版本陈旧度，反之训练器等待。增加 worker 前先测验收过滤后的可用率；生成 token 多不代表有效训练样本多。有限批次同步可以用明确的停顿换取容易解释的版本边界。

```mermaid
flowchart LR
	P["Prompt"] --> R["Rollout Engine"]
	R --> X["Responses + LogProbs"]
	X --> W["Reward / Verifier"]
	W --> A["Advantage / Targets"]
	A --> T["Policy Training"]
	T --> U["Weight Version"]
	U --> R
```

需要定义权重版本语义：rollout 使用哪个策略版本，训练 batch 是否允许混合版本，更新如何传给推理 worker。频繁同步降低陈旧度却消耗网络和暂停时间；宽松同步提高吞吐，但算法必须容忍 off-policy 数据。

在没有明确算法契约和质量证据时，不应默认引入复杂的增量权重协议或持久 rollout cache。可以先采用离线批次、显式版本和失败即重跑的简单边界。

---

## 推理模型的系统影响

长思维链和采样多条候选会增加输出 token、KV cache 和 rollout 方差；验证器可能增加 CPU 或 GPU 服务；按结果过滤会让有效训练 token 比例下降。规划容量时应记录：

- 每 prompt 的候选数量；
- 输出长度分布和提前终止率；
- reward/verification 延迟；
- 生成 token 到训练 token 的转化率；
- 权重更新与 rollout 的版本差。

---

## CPU 路线

小模型可以在 CPU 上验证 SFT、DPO loss、rollout 数据结构、版本字段和队列背压。在线 RL 的系统拓扑也可用 mock model 模拟。真实生成与训练吞吐、低精度 kernel 和多 GPU 权重同步仍需 GPU 环境。

---

## 参考文献

- Ouyang, L. et al. (2022). [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155).
- Rafailov, R. et al. (2023). [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290).
- Shao, Z. et al. (2024). [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300).
