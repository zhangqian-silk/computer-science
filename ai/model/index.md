# 模型层：从统计模型到预训练基础模型

`model/` 目录讨论的是可以作为独立对象理解和比较的完整模型，例如 N-Gram、RNN、LSTM、Seq2Seq、Transformer、BERT 与 GPT。

这一层的重点不是某个局部模块，而是完整回答：

- 这个模型解决什么问题；
- 它如何组织输入、状态、机制与输出；
- 它怎样训练、怎样推理；
- 它相对于前代方法修正了什么限制。

---

## 相关主题

- 若重点是 attention、位置机制、LoRA 等机制问题，可先看 [mechanism](../mechanism/index.md)。
- 若重点是表示学习与 embedding，可先看 [representation](../representation/index.md)。

---

## 推荐阅读顺序

### 序列模型主线

1. [N-Gram](./n-gram.md)
2. [NPLM](./nplm.md)
3. [RNN](./rnn.md)
4. [LSTM](./lstm.md)
5. [Seq2Seq](./seq2seq.md)
6. [Transformer](./transformer.md)
7. [BERT](./bert.md)
8. [GPT](./gpt.md)

### 快速现代路径

1. [Transformer](./transformer.md)
2. [Transformer Extensions](./transformer-extensions.md)
3. [BERT](./bert.md)
4. [BERT Family](./bert-family.md)
5. [GPT](./gpt.md)

---

## 文档角色划分

| 文档 | 角色 | 建议用途 |
| --- | --- | --- |
| [N-Gram](./n-gram.md) | 统计语言模型入口 | 理解早期有限上下文假设 |
| [NPLM](./nplm.md) | 神经语言模型起点 | 理解离散到连续的过渡 |
| [RNN](./rnn.md) | 递归序列模型核心篇 | 理解状态递推与 BPTT |
| [LSTM](./lstm.md) | 门控序列模型核心篇 | 理解长期依赖修正 |
| [Seq2Seq](./seq2seq.md) | 条件生成框架核心篇 | 理解 encoder-decoder |
| [Transformer](./transformer.md) | 现代序列主干总览 | 理解 self-attention 架构 |
| [Transformer Extensions](./transformer-extensions.md) | 扩展路线专题 | 理解长上下文、ViT 与多模态扩展 |
| [BERT](./bert.md) | 编码器预训练核心篇 | 理解双向表示学习 |
| [BERT Family](./bert-family.md) | 后续谱系专题 | 理解变体、句向量与检索路线 |
| [GPT](./gpt.md) | 解码器预训练核心篇 | 理解自回归生成路线 |

---

## 阅读提示

若你是第一次进入本目录，建议不要直接跳到单篇长文细节，而是先把模型分成 4 代：

- 统计模型；
- 递归模型；
- 条件生成模型；
- 预训练 Transformer 模型。

如果想先建立主线，可以先读 [NLP 历史](../nlp/history.md) 再回到本目录。
