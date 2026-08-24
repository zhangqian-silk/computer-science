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

| 方法 | 核心思想 | 主要取舍 |
| --- | --- | --- |
| BPE | 反复合并高频相邻符号 | 实现直观，合并表决定分词 |
| WordPiece | 按语言模型式目标选择子词 | 常与 BERT 家族关联 |
| Unigram | 从大候选词表中删减并选择高概率切分 | 可以保留多种候选切分 |
| byte-level | 以字节为基础符号再做合并 | 几乎没有未知字符，但序列可能变长 |

BPE 的训练过程从基础符号开始，统计相邻 pair，并把选中的 pair 合并为新 token。推理时必须按固定合并规则执行；不能重新依据当前文本频率训练。

中文、代码、数字和多语言文本的 token 长度差异很大。相同字符数可能对应完全不同的序列长度，进而改变 Attention 计算、KV cache、计费和限流。

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

## 参考文献

- Sennrich, R. et al. (2016). *Neural Machine Translation of Rare Words with Subword Units*.
- Kudo, T. (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*.
- Hugging Face. *Tokenizers Documentation*.
