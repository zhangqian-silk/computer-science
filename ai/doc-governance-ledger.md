# AI 文档治理台账

本文是 [refactor-plan.md](./refactor-plan.md) 的配套执行台账，用于把高优先级文档的治理状态固定为可追踪事项。它不重复展开整体改造原则，而是聚焦回答 3 个问题：

- 哪些文档当前承担主文档职责；
- 哪些文档仍存在越界、重复或过长问题；
- 下一轮重构应优先改什么。

---

## 一、台账字段说明

下表中的字段含义如下：

| 字段 | 含义 |
| --- | --- |
| 当前角色 | 该文档目前在知识体系中的职责定位 |
| 主文档 | 是否承担该主题的完整展开责任 |
| 当前行数 | 仅作为治理信号；超过 700 行默认进入复核区 |
| 主要问题 | 当前正文中最需要治理的边界或重复问题 |
| 是否需拆分 | 是否需要新增专题页来承接外溢内容 |
| 下一步动作 | 下一轮重构时应直接执行的动作 |
| 优先级 | `P0` 表示当前最高优先级，`P1` 表示次高优先级，`P2` 表示边界维护项 |
| 状态 | 当前治理进度 |

说明：

- 当前行数为 2026-03-22 的快照，后续可随重构推进更新；
- 本台账先覆盖首批高优先级 8 篇文档，后续再逐步扩展到其他主题。

---

## 二、高优先级治理清单

| 文档 | 当前角色 | 主文档 | 当前行数 | 主要问题 | 是否需拆分 | 下一步动作 | 优先级 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ai/mechanism/positional-encoding.md](./mechanism/positional-encoding.md) | 位置机制总览 | 是 | 1593 | 首轮拆分已完成，但正文体量仍大，后续仍需继续压缩摘要与交叉引用 | 否 | 维持总览角色，后续配合 `transformer.md`、`gpt.md` 回收重复解释 | P1 | 已完成首轮拆分 |
| [ai/representation/embedding.md](./representation/embedding.md) | Embedding 统一定义与表示入口 | 是 | 323 | 首轮拆分已完成，后续仍需继续压缩与 `word2vec.md`、模型文档的重复背景说明 | 是 | 已拆出 `text-embedding-training.md`、`retrieval-embedding.md` 与 `multimodal-alignment.md`，下一轮转入边界对齐与摘要压缩 | P0 | 已完成首轮拆分 |
| [ai/model/transformer.md](./model/transformer.md) | Transformer 主干与架构主文档 | 是 | 1172 | 同时覆盖 attention、position、模型比较、ViT、多模态与高效变体，中心问题被稀释 | 是 | 拆出 `transformer-extensions.md`，并把公共机制细节回收到 mechanism 层文档 | P0 | 未开始 |
| [ai/model/bert.md](./model/bert.md) | BERT 本体与训练范式主文档 | 是 | 1329 | 模型本体、家族比较、任务适配与扩展路线耦合过重 | 是 | 拆出 `bert-family.md`，正文回到 BERT 架构、MLM/NSP 与微调范式 | P0 | 未开始 |
| [ai/model/gpt.md](./model/gpt.md) | Decoder-only Transformer 与自回归语言模型主文档 | 是 | 1182 | 对 Transformer block、位置机制与表示层公共内容重复较多 | 否 | 压缩公共机制复述，聚焦自回归目标、推理流程与能力演化 | P1 | 待回收重复 |
| [ai/model/seq2seq.md](./model/seq2seq.md) | Encoder-Decoder 条件生成框架主文档 | 是 | 950 | attention 数学展开过深，削弱了 Seq2Seq 框架本体的辨识度 | 否 | 收缩 attention 细节，保留“为什么需要 + 如何接入”的最小闭环 | P1 | 待回收重复 |
| [ai/mechanism/attention.md](./mechanism/attention.md) | 一般 attention 机制主文档 | 是 | 962 | 与 `self-attention.md`、`transformer.md` 在 Q/K/V、mask、多头层面仍有明显重复 | 否 | 保留抽象与数学骨架，后续回收其他文档中可替代的大段说明 | P1 | 待对齐边界 |
| [ai/mechanism/self-attention.md](./mechanism/self-attention.md) | 序列内部 attention 的特化机制文档 | 是 | 311 | 篇幅已可控，但仍需进一步固化与 `attention.md`、`transformer.md` 的边界 | 否 | 强化“序列内部特化”定位，避免继续吸收 block 级与整机级内容 | P2 | 待对齐边界 |

---

## 三、建议执行顺序

按当前优先级，下一轮正文改造建议按以下顺序推进：

1. 拆分 [ai/representation/embedding.md](./representation/embedding.md)；
2. 收缩 [ai/model/transformer.md](./model/transformer.md) 并新增扩展页；
3. 收缩 [ai/model/bert.md](./model/bert.md) 并新增家族页；
4. 回收 [ai/model/gpt.md](./model/gpt.md) 与 [ai/model/seq2seq.md](./model/seq2seq.md) 的公共机制重复；
5. 对齐 [ai/mechanism/attention.md](./mechanism/attention.md)、[ai/mechanism/self-attention.md](./mechanism/self-attention.md)、[ai/model/transformer.md](./model/transformer.md) 的边界。

这一定序与 [refactor-plan.md](./refactor-plan.md) 中第 7-12 步保持一致。当前第 7 步已完成，因此下一轮最应立即进入的重构任务是第 8 步，也就是收缩 `transformer.md`。

---

## 四、更新规则

后续维护本台账时，统一遵守以下规则：

- 只有当文档职责发生变化、状态推进或优先级调整时，才更新对应条目；
- 若新增专题页，应同时回写“来源主文档”和“承接内容”；
- 若某篇文档完成首轮拆分，不代表后续不再治理，应继续记录二轮压缩与重复回收任务；
- 台账只记录治理决策，不承载长篇正文说明，详细原因统一留在 [refactor-plan.md](./refactor-plan.md)。
