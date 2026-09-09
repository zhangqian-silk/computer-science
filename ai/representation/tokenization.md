# Tokenization：从文本到模型序列

Tokenizer 把原始输入转换为有限词表中的 token ID，并负责把生成 ID 还原为文本。它不是无损的「按词切分」步骤，而是模型接口的一部分：词表、正规化、特殊 token 和聊天模板只要有一项不匹配，权重就可能接收到完全不同的输入。

<TokenizationMergeExplorer />

---

## 接口与不变量

设词表为 $\mathcal{V}$，编码和解码为：

$$
\operatorname{encode}:\text{text}\rightarrow\mathcal{V}^{*},
\qquad
\operatorname{decode}:\mathcal{V}^{*}\rightarrow\text{text}
$$

并非所有 tokenizer 都满足逐字符串严格往返。Unicode 正规化、非法字节、空白处理和特殊 token 可能改变结果。系统至少要验证：

- 模型权重与词表 ID 一致；
- BOS、EOS、PAD、UNK 和角色标记语义一致；
- encode/decode 的正规化策略可追踪；
- 流式解码不会在多字节字符中间输出损坏文本；
- 统计 token 数使用实际部署 tokenizer，而不是字符数估算。

---

## 子词方法

整词词表把「低频词」变成很少被更新的独立参数，未登录词还可能全部落到 `<UNK>`；字符词表覆盖面较好，却需要更多位置才能表达一个词。子词在两者之间折中：常见片段用一个 ID，罕见组合拆成已有片段。它学到的是编码单元，不是语言学上唯一正确的词根。

| 方法 | 核心思想 | 主要取舍 |
| --- | --- | --- |
| BPE | 反复合并高频相邻符号 | 实现直观，合并表决定分词 |
| WordPiece | 依据词表构造规则学习子词；常见编码器采用最长匹配 | 训练规则依实现而异，不能与 BPE 合并排序混用 |
| Unigram | 从大候选词表中删减并选择高概率切分 | 可以保留多种候选切分 |
| byte-level | 以字节为基础符号，可与 BPE 等算法组合 | 完整字节覆盖避免普通文本的字符 OOV，但序列可能变长 |

BPE 的训练过程从基础符号开始，统计相邻 pair，并把选中的 pair 合并为新 token。推理时必须按固定合并规则执行；不能重新依据当前文本频率训练。

中文、代码、数字和多语言文本的 token 长度差异很大。相同字符数可能对应完全不同的序列长度，进而改变 Attention 计算、KV cache、计费和限流。

### BPE：训练合并表与应用合并表

下面是手算示例，不是某个已发布 tokenizer 的实际词表。假设语料只有 `low` 出现 5 次、`lower` 出现 2 次；初始按字符切分，暂时省略词尾标记，也不跨词合并：

| 步骤 | 当前表示 | 按词频加权的相邻对计数 | 本步选择 |
| --- | --- | --- | --- |
| 初始 | `l o w`；`l o w e r` | `l o`: 7，`o w`: 7，`w e`: 2，`e r`: 2 | 平局规则选 `l o` |
| 第一次合并后 | `lo w`；`lo w e r` | `lo w`: 7，`w e`: 2，`e r`: 2 | 合并 `lo w` |
| 第二次合并后 | `low`；`low e r` | `low e`: 2，`e r`: 2 | 若预算用完，停止 |

得到有顺序的规则 `l + o → lo`、`lo + w → low`。编码 `lower` 时重放适用的合并优先级，得到 `low | e | r`，而不是重新统计这个输入的高频对。`new` 中若某字符不属于基础字母表，单靠这两条规则也不能解决 OOV；是否有字节回退是另一个配置问题。

上方交互图适合观察「一次合并会改变哪些相邻对」。需要注意，合并次数是训练预算，token ID 是词表中的索引，二者都不是语义相似度。两个 ID 相邻不代表两个 token 语义相近；连续表示要到 [Embedding](./embedding.md) 查表后才出现。

Sennrich 等人的 2016 年论文把 BPE 用于神经机器翻译的罕见词问题，比较对象包括固定词表与其他切分策略；关键贡献是用可组合子词扩大可表达范围，而不是证明任意语料上的高频合并都具有最优语义。

### Unigram：对候选切分评分

::: info 符号与约定
$s$ 是待编码字符串，$\mathcal{S}(s)$ 是能拼回 $s$ 的候选 token 序列集合，$u_i$ 是其中第 $i$ 个 token，$p(u)$ 是词表 token 的概率。这里的概率属于分词模型，不是后续语言模型的条件概率。
:::

Unigram 假设一条切分的得分可以分解为 token 概率的乘积：

$$
P(u_1,\ldots,u_m)=\prod_{i=1}^{m}p(u_i),\qquad
\sum_{u\in\mathcal{V}}p(u)=1
$$

确定性编码选择负对数代价最小的有效切分：

$$
\hat{u}
=\arg\min_{(u_1,\ldots,u_m)\in\mathcal{S}(s)}
\sum_{i=1}^{m}-\log p(u_i)
$$

若 `ab` 有 `ab` 和 `a | b` 两种切分，设 $p(\text{ab})=0.2$、$p(\text{a})=p(\text{b})=0.3$，剩余 0.2 分给其他 token，则单块得分 0.2 高于两块的 $0.3\times0.3=0.09$。因此选单块不是因为「永远选最长」，而是因为本例的模型得分更高。

可以把字符边界看成节点，合法 token 看成跨越若干字符的边，边代价是 $-\log p(u)$；最优切分就是从起点到终点的最短路径，可用动态规划求解。训练则从较大的候选词表出发，估计概率并按删词造成的目标损失逐步裁剪，保留必要的基础单元。

Kudo 的 2018 年论文进一步在训练中采样多种切分，让同一句话以不同 token 序列出现，减少对唯一切分的依赖。这是数据表示层的正则化，不是推理时不断修改词表，也不保证每一种采样切分都更短。

### WordPiece：最长匹配不等于 BPE 重放

以 BERT 常见编码约定为例，假设词表包含 `play`、`##ing`、`p`、`##lay`。编码 `playing` 时，从词首选择可匹配的最长前缀 `play`，再从剩余部分选择带续接标记的 `##ing`。`##` 表达这是词内续接片段，并不表示原文真的出现了两个井号。

该过程依赖最终词表而不是 BPE 的合并优先级。若剩余字符没有合法切分，常见 BERT WordPiece 实现会将整个预切分词标为 `[UNK]`，而非自动回退字节；实际还要检查最长词长度等配置。训练词表的方法与应用词表的算法应分开理解，不能由最长匹配反推训练时一定选择最高频 pair。

### byte-level 和 SentencePiece 不构成另一条互斥分类

需要把三个问题分开：

- 基础符号是字符还是字节；
- 词表通过 BPE、Unigram 还是其他规则学习；
- 实现是否要求预先按空格分词，怎样保存空白、正规化和特殊标记。

SentencePiece 是可直接处理原始句子的分词工具，支持 BPE 和 Unigram；它不是与两者并列的单一算法。UTF-8 字节序列也不是 Unicode 字符序列，一个汉字或 Emoji 可以由多个字节构成。具备完整字节覆盖不代表每个 token 单独解码后都是完整字符。

---

## 词表大小为何影响模型，而不只是文本长度

对隐藏维度 $d$、词表 $\mathcal{V}$，输入 Embedding 有 $|\mathcal{V}|d$ 个参数；未共享权重的输出投影还需要同量级参数。扩大词表可能缩短输入，但会增大词表投影和完整 softmax 的工作量。缩小词表则可能增加序列长度，进一步增加 Attention 与 KV 成本。

因此不能只按「压缩成最少 token」选择 tokenizer。还要考察低频 token 获得多少训练、不同语言的长度差异，以及下游质量。两个模型对同一段文本的 token 平均 NLL 或 PPL 不能直接比较，因为计数单位已经改变，详见[语言模型评估](../evaluation/language-model-evaluation.md)。

---

## 特殊 token 与聊天模板

现代服务通常先把消息转换为模型约定的模板：

~~~text
<system>...</system><user>...</user><assistant>
~~~

模板中的角色、分隔符和终止标志最终都是 token。错误模板可能导致模型续写用户文本、无法停止或表现明显下降。服务端应把 tokenizer、模板与模型制品作为同一版本单元发布。

Padding 方向也会影响 batch。训练常把多个序列拼接或 padding 后用 mask 排除无效位置；推理框架更倾向直接维护不同长度请求，避免对 padding token 做无效计算。

---

## 性能与系统影响

Tokenization 常在 CPU 上执行，可能成为短请求、高并发服务的前端瓶颈。需要观察：

- 每秒编码和解码 token 数；
- 长文本内存分配与复制；
- Python 调用、线程池和 GIL 开销；
- 输入 token 数估算与实际值偏差；
- 流式 detokenization 的增量状态；
- 超长输入在进入 GPU 前是否已经被拒绝。

限流应同时考虑请求数、输入 token、最大输出 token 和在途 KV 容量。只按 HTTP QPS 限流，无法约束不同长度请求的真实资源占用。

---

## CPU 实验设计

Tokenizer 本身适合完全在 CPU 上实验：

1. 选择中文、英文、代码、数字、Emoji 和混合文本语料；
2. 使用两个 tokenizer 记录 token 数、往返结果和吞吐；
3. 分别测试单条、批量和多线程编码；
4. 检查 Unicode 边界和流式解码；
5. 以真实 token 数代入 KV cache 与请求成本估算。

正确性不变量是模型使用的 token ID 与官方 tokenizer 一致。吞吐比较需要固定语料、线程数、CPU 型号、库版本和是否启用批量接口。

---

## 参考文献

- Sennrich, R. et al. (2016). [*Neural Machine Translation of Rare Words with Subword Units*](https://aclanthology.org/P16-1162/). §3 的子词方法与实验中的罕见词分析对应本文 BPE 动机。
- Kudo, T. (2018). [*Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*](https://aclanthology.org/P18-1007/). Unigram 词表学习、切分采样与翻译实验。
- Kudo, T. and Richardson, J. (2018). [*SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing*](https://aclanthology.org/D18-2012/). 区分原始文本处理工具与内部算法。
- Hugging Face. [*WordPiece*](https://huggingface.co/learn/llm-course/chapter6/6). 常见最长匹配编码及训练实现的说明，不作为原始 WordPiece 训练代码的公开规范。
