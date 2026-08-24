# GPT：以因果语言建模统一理解与生成

GPT 家族使用 Decoder-only Transformer，在因果 mask 下反复预测下一 token。原始 GPT 使用较早的 Transformer Decoder 变体；现代 decoder-only LLM 常采用 pre-LN/RMSNorm、RoPE、门控 FFN 与共享 KV 头等工程改造。不同实现共享因果语言建模这一概率接口，具体 block 配置则随模型而异。

<LanguageModelObjectiveExplorer initial-mode="causal" />

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$x_t$ 是第 $t$ 个 token，$T$ 是序列长度，$\theta$ 是模型参数，$\mathcal{L}$ 是因果语言模型损失；$z_w$ 是候选 token $w$ 的 logit，$\tau$ 是采样温度；$K_{\leq t},V_{\leq t}$ 是截至位置 $t$ 的 KV cache。
:::

---

## 因果语言模型

对 token 序列 $x_1,\ldots,x_T$：

$$
P(x_1,\ldots,x_T)
=
\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

训练时把目标向右平移一位，对每个非 padding 位置计算交叉熵：

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log P_\theta(x_t\mid x_{<t})
$$

因果 mask 保证位置 $t$ 只能读取自己之前的 token。整段训练样本仍可并行计算；「因果」限制的是信息可见性，不是要求训练代码逐 token 循环。

### 一条序列怎样产生训练信号

序列：

```text
<BOS> 天空 是 蓝色 <EOS>
```

可以整体右移一位构造输入与目标：

| 位置 | 模型看到的前缀末尾 | 监督目标 |
| ---: | --- | --- |
| 1 | `<BOS>` | `天空` |
| 2 | `<BOS> 天空` | `是` |
| 3 | `<BOS> 天空 是` | `蓝色` |
| 4 | `<BOS> 天空 是 蓝色` | `<EOS>` |

一次 Transformer 前向会并行产生四个位置的 logits。假设模型分别为四个正确目标分配 $0.8,0.6,0.7,0.9$ 的概率，则：

$$
P(x)=0.8\times0.6\times0.7\times0.9=0.3024
$$

平均负对数似然为：

$$
\mathcal{L}
=-\frac{1}{4}\log0.3024
\approx0.299
$$

对应 token 级困惑度约为 $\exp(0.299)\approx1.35$。训练只使用真实 token 右移后的目标，不需要先让模型生成完整句子再比较文本。

预训练循环由分词、分段或拼接样本、构造因果 mask、前向计算、有效 token 交叉熵和参数更新组成。文档边界通常使用结束 token 标记；padding 或被明确屏蔽的位置不参与损失。数据怎样切块会决定模型在训练中见到哪些跨段上下文。

---

## Decoder-only 主干

~~~mermaid
flowchart LR
	X["token + position"] --> B1["Causal Transformer Block"]
	B1 --> B2["… × L"]
	B2 --> N["Final Norm"]
	N --> H["Vocabulary Projection"]
	H --> P["Next-token Distribution"]
~~~

每个 block 包含带因果 mask 的 Self-Attention、FFN、残差与归一化。与原始 Transformer Decoder 不同，GPT 不需要 Encoder，也没有必需的 Cross-Attention；提示、问题、示例和待生成内容被序列化到同一个前缀。

输出层把隐藏状态投影到词表 logits。它常与输入 embedding 共享权重，但是否共享属于具体实现选择。

---

## 为什么同一目标能覆盖多种任务

分类、翻译、问答和代码生成都可以被序列化为「给定前缀，继续写出目标文本」。模型在大规模语料上学习的不是任务标签表，而是条件分布；提示中的指令、示例和上下文会改变后续 token 的概率。

In-context learning 指模型仅通过当前前缀中的说明或示例改变输出，不更新参数。它依赖预训练学到的模式和上下文窗口，不等同于梯度学习，也不保证对提示改写稳定。

对话系统会先把 system、user、assistant 等消息按照模型约定的模板序列化成单一 token 流。角色标记本身也是 token；模型仍然只执行「根据前缀预测下一个 token」。如果使用了错误的聊天模板，即使文字相同，模型接收到的 token 结构也可能与指令微调阶段不同。

---

## 从预训练到指令模型

现代 GPT 类系统常经历多个目标不同的阶段：

1. 因果预训练学习通用语言与代码分布；
2. 监督微调学习指令—回答格式和目标行为；
3. 偏好优化使用比较信号调整回答取舍；
4. 任务适配可使用全量微调或 [LoRA](../mechanism/lora.md)。

后续阶段不会修改因果解码接口，却会显著改变模型如何解释指令、拒绝请求和组织回答。因而「架构相同」不表示行为相同。

---

## 自回归生成

给定前缀，模型每一步产生 logits $z_t$。温度调整：

$$
P_\tau(w)
=
\frac{\exp(z_w/\tau)}
{\sum_v\exp(z_v/\tau)}
$$

$\tau<1$ 使分布更尖锐，$\tau>1$ 增加随机性。Top-k 只保留概率最高的 $k$ 个 token；top-p 保留累计概率达到阈值的最小集合。贪心、beam search 与采样对应不同任务假设，没有普遍最优设置。

例如三个候选 token 的 logits 为 $(2,1,0)$。不同温度下的概率约为：

| 温度 | 第 1 个候选 | 第 2 个候选 | 第 3 个候选 |
| ---: | ---: | ---: | ---: |
| $\tau=0.5$ | $0.867$ | $0.117$ | $0.016$ |
| $\tau=1$ | $0.665$ | $0.245$ | $0.090$ |
| $\tau=2$ | $0.506$ | $0.307$ | $0.186$ |

温度改变的是同一组 logits 的相对尖锐程度，不会增加模型没有学到的候选。Top-k 或 top-p 随后还会截断并重新归一化分布。

停止条件包括结束 token、结构约束、最大长度或外部控制器终止。解码参数会直接改变生成评估结果，必须与模型版本一起记录。

---

## KV cache 与长上下文

第 $t$ 步只需要为新 token 计算 query/key/value，历史 key/value 可复用：

$$
K_{\le t}=[K_{<t};k_t],\qquad
V_{\le t}=[V_{<t};v_t]
$$

一次请求的推理生命周期为：

```mermaid
flowchart LR
	P["Prompt token"] --> F["Prefill：并行处理完整提示"]
	F --> K["建立各层 KV cache"]
	K --> L["输出末位置 logits"]
	L --> S["采样或选择新 token"]
	S --> D["Decode：只计算新 token"]
	D --> K
	S --> Q{"停止条件满足？"}
	Q -->|否| D
	Q -->|是| O["返回生成序列"]
```

Prefill 的输出只有最后位置用于选择第一个新 token，但所有提示位置的 key/value 都会写入缓存。Decode 每轮把新 token 的 key/value 追加到缓存，并让它的 query 读取全部允许的历史。模型不能一次算出尚未生成的后续 token，因为下一轮输入取决于本轮选择。

KV cache 减少重复投影和历史 block 计算，却随上下文线性占用内存。Grouped-Query Attention 减少 KV 头数，量化或淘汰策略减少缓存体积；这些优化可能影响质量或实现复杂度。

位置扩长见[长上下文位置扩展](../mechanism/long-context.md)，真实利用能力见[长上下文评估](../evaluation/long-context-evaluation.md)。

---

## 概率预测不等于事实检索

因果目标奖励对训练分布中下一 token 的预测，并不显式验证陈述真假、推理过程或工具执行结果。因此 GPT 类模型可能：

- 生成流畅但无依据的内容；
- 对提示措辞和上下文位置敏感；
- 复现训练数据中的偏差；
- 在长输出中累积早期错误；
- 对训练截止后的事实没有可靠更新机制。

检索可以提供外部证据，工具可以获得可验证结果，Agent 运行时可以控制动作；它们都不能被模型参数本身的语言概率替代。评估应按任务拆分准确性、忠实性、鲁棒性与成本，见[生成评估](../evaluation/generation-evaluation.md)。

---

## 与 BERT 的结构差异

| 维度 | BERT | GPT |
| --- | --- | --- |
| 主干 | Encoder-only | Decoder-only |
| 可见性 | 双向 | 因果 |
| 预训练目标 | 恢复被遮蔽 token | 预测下一 token |
| 原生输出 | 上下文化表示 | 自回归序列 |
| 典型接口 | 分类、标注、抽取 | 补全、对话、生成 |

差异来自 mask、目标和输出接口，而不是「理解」与「生成」的绝对能力边界。

---

## 参考文献

- Radford, A. et al. (2018). *Improving Language Understanding by Generative Pre-Training*.
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*.
- Brown, T. B. et al. (2020). *Language Models are Few-Shot Learners*.
- Ouyang, L. et al. (2022). *Training Language Models to Follow Instructions with Human Feedback*.
