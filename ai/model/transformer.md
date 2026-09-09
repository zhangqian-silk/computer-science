# Transformer：用全局交互构建可堆叠序列主干

Transformer 将序列交互与逐位置通道变换交替堆叠，用残差和归一化维持统一层间接口。它不是「只有 Attention」：位置、FFN、mask、读出头和训练目标共同决定行为。以下主推导采用 Pre-LN，交互图同时对照原论文的 Post-LN，避免把现代实现倒写成原始设计。

<TransformerBlockFlow />

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$B,n,d_{\text{model}}$ 是 batch、序列长度与隐藏维度；$n_{\text{head}},n_{\text{kv}},d_{\text{head}}$ 是 Attention 头数、KV 头数与单头维度；$H^{(\ell)}$ 是第 $\ell$ 层隐藏状态，$Q,K,V,M$ 是 Query、Key、Value 与 mask；$L$ 表示层数，$b$ 表示缓存元素字节数。
:::

---

## 输入与位置

对加性绝对位置变体，token ID 查表后与位置向量相加：

$$
H^{(0)}=E[x_1,\ldots,x_n]+P
$$

这里 P 只表示加性位置向量。相对偏置加入 score，RoPE 旋转 Q/K，不能代入上式；只用非加性位置机制时，输入可为 Embedding 本身。因果 mask 也能提供顺序约束。

---

## 一个 pre-LN Block

主残差流保持 $[B,n,d_{\text{model}}]$，子层暂时改变内部布局再投影回来。头拆分不是序列切分，FFN 升维不是增加 token。

一次 Attention 直接连接所有可见位置，不意味着一层已经完成任意复杂推理。后续层读取的是已更新表示，可以逐步组合关系；深度、宽度和上下文长度增加的是不同能力与成本，不应混为「更大模型」。

对第 $\ell$ 层：

$$
\tilde{H}^{(\ell)}
=
H^{(\ell-1)}
+
\operatorname{MHA}\left(
\operatorname{LN}(H^{(\ell-1)})
\right)
$$

$$
H^{(\ell)}
=
\tilde{H}^{(\ell)}
+
\operatorname{FFN}\left(
\operatorname{LN}(\tilde{H}^{(\ell)})
\right)
$$

FFN 独立作用于每个位置：

$$
\operatorname{FFN}(x)
=
W_2\phi(W_1x+b_1)+b_2
$$

各部件职责不同：

| 部件 | 交换位置间信息 | 主要作用 |
| --- | --- | --- |
| Multi-Head Attention | 是 | 按内容读取其他位置 |
| FFN | 否 | 在通道维扩展、非线性变换 |
| Residual | - | 保留恒等路径并叠加更新 |
| LayerNorm | - | 稳定每个 token 的特征尺度 |

原始 post-LN 把 LayerNorm 放在残差加法之后。现代实现还可能使用 RMSNorm、SwiGLU、Grouped-Query Attention 等替换部件；这些不改变 block 的「跨位置读取 + 逐位置变换」骨架。

### 一批 token 如何穿过 Block

设 batch 大小为 $B$、序列长度为 $n$、隐藏维度为 $d_{\text{model}}$、Attention 头数为 $n_{\text{head}}$，则一层中的主要张量流为：

| 步骤 | 典型形状 | 含义 |
| --- | --- | --- |
| 输入隐藏状态 | $B\times n\times d_{\text{model}}$ | 每个 token 的当前表示 |
| 投影并拆分 $Q,K,V$ | $B\times n_{\text{head}}\times n\times d_{\text{head}}$ | 每个头的查询、索引和内容 |
| Attention 分数 | $B\times n_{\text{head}}\times n\times n$ | 每个位置对其他位置的读取强度 |
| 各头聚合并拼接 | $B\times n\times d_{\text{model}}$ | 融合跨位置上下文 |
| FFN 中间层 | $B\times n\times d_{\mathrm{ff}}$ | 每个位置独立扩展通道 |
| Block 输出 | $B\times n\times d_{\text{model}}$ | 供下一层继续处理 |

Attention 改变的是 token 之间的信息来源，FFN 改变的是每个 token 内部的特征组合；残差保证两次更新都在原表示上增量叠加。层数增加后，一个 token 会经历多轮「读取上下文—重组特征」。

对因果 Decoder，长度为 4 的可见性矩阵为：

$$
M=
\begin{bmatrix}
0 & -\infty & -\infty & -\infty\\
0 & 0 & -\infty & -\infty\\
0 & 0 & 0 & -\infty\\
0 & 0 & 0 & 0
\end{bmatrix}
$$

它在 softmax 前加到 Attention 分数上。第 3 行可以读取位置 1 至 3，却无法读取位置 4；因此整段张量可以一次前向计算，同时仍满足下一 token 预测的因果约束。

---

## Encoder 与 Decoder

原始 Decoder 包含 masked self-attention、读取源端的 cross-attention 和 FFN。Decoder-only 语言模型通常移除源端与 cross-attention，不能仅因为都叫 Decoder 就认为结构完全一样。Encoder-only 也不自动规定 MLM；训练目标是另一个选择轴。

~~~mermaid
flowchart LR
	S["源序列"] --> E["Encoder Stack"]
	E --> M["源端表示"]
	T["目标前缀"] --> D["Masked Decoder Stack"]
	M --> C["Cross-Attention"]
	C --> D
	D --> O["下一 token 分布"]
~~~

Encoder 的 Self-Attention 通常双向可见，用于整段理解。Decoder 的 Self-Attention 使用因果 mask，只读取当前与过去；在 Encoder-Decoder 结构中，Decoder 另有 Cross-Attention 读取源端表示。

由此形成三类常见接口：

| 结构 | 可见性 | 典型目标 | 代表 |
| --- | --- | --- | --- |
| Encoder-only | 输入内双向 | MLM、分类、标注 | BERT |
| Decoder-only | 因果前缀 | 下一 token 预测 | GPT |
| Encoder-Decoder | 源端双向、目标端因果 | 条件生成 | 原始 Transformer、T5 |

它们共享 block 家族，但输入输出、mask 与目标函数不同，不能只按层数或参数量视为同一种模型。

---

## 训练目标定义能力接口

Encoder-Decoder 的条件生成目标：

$$
P(Y\mid X)=\prod_{t=1}^{m}P(y_t\mid y_{<t},X)
$$

Decoder-only 的因果语言模型目标：

$$
P(x_1,\ldots,x_n)=\prod_{t=1}^{n}P(x_t\mid x_{<t})
$$

Encoder-only 常用掩码语言建模，根据双向上下文恢复被遮蔽 token。结构决定信息可见性，目标决定哪些预测误差产生训练信号；两者共同塑造模型，而不是只靠 Attention 公式。

以 Decoder-only 训练样本为例：

| 模型输入 | `<BOS>` | `天气` | `很` | `好` |
| --- | --- | --- | --- | --- |
| 监督目标 | `天气` | `很` | `好` | `<EOS>` |

模型输出每个输入位置的词表 logits，与其后一 token 的标签计算交叉熵；也可说 Decoder 输入是目标序列右移并补 BOS。预测「很」的位置只读 `<BOS> 天气`，不会偷看「好」。

Encoder-Decoder 训练时，Encoder 先对源序列执行一次双向前向；Decoder 读取完整的真实目标前缀，并通过 Cross-Attention 使用源端表示。Encoder-only 则通常一次处理被遮蔽的完整输入，只在被选位置或任务头位置计算损失。三种结构共享 Block 组件，但训练样本的构造方式不同。

---

## 推理形态与训练有何不同

Encoder-only 模型的输入在推理开始前已经完整可用，通常一次前向即可得到分类、标注或向量结果。Encoder-Decoder 模型先计算一次源端表示，再逐步运行 Decoder。因果 Decoder 则分成两个阶段：

1. **Prefill**：并行处理完整提示，建立各层历史 key/value；
2. **Decode**：每次只输入一个新 token，读取已有缓存并生成下一 token。

训练时可一次计算整段因果序列，因为 mask 已阻止未来信息；Decode 时下一 token 依赖刚生成的结果，时间步无法提前确定，必须逐步推进。

KV cache 保存各层历史 key/value，避免每步重算前缀。其大小随层数、上下文长度、KV 头数和 head 维度增长。长上下文系统因此同时面临：

- prefill 的全局 Attention 计算；
- decode 的串行步数；
- KV cache 的容量和读带宽；
- 位置超出训练范围后的稳定性。

忽略对齐和元数据时，单个样本的 KV cache 近似占用：

$$
2\times L\times n\times n_{\text{kv}}\times d_{\text{head}}\times b
$$

$2$ 对应 key 与 value，$L$ 为层数，$n_{\text{kv}}$ 为 KV 头数，$b$ 为每个元素的字节数。例如 $L=32$、$n_{\text{kv}}=8$、$d_{\text{head}}=128$、FP16 时，每新增一个 token 约增加：

$$
2\times32\times8\times128\times2
=131\,072\text{ bytes}
$$

即约 $128\text{ KiB}$。长度达到 8192 时，单样本缓存约为 $1\text{ GiB}$。Grouped-Query Attention 减少 $n_{\text{kv}}$，正是降低 Decode 缓存和带宽成本的一种方式。

[Self-Attention](../mechanism/self-attention.md)解释缓存接口，[长上下文位置扩展](../mechanism/long-context.md)解释位置外推，[稀疏注意力](../mechanism/sparse-attention.md)解释受限连接图。

---

## 复杂度应怎样理解

原始论文在机器翻译上对比循环/卷积主干，强调并行性、依赖路径与结果；它不证明每种序列长度和硬件上 Attention 都比循环网络更便宜。短序列时投影与 FFN 可能主导，长序列时两两交互增长更快，最终应结合[性能模型](../infra/accelerator/performance-model.md)。

对序列长度 $n$、隐藏维度 $d_{\text{model}}$，全局 Attention 的主要连接代价近似 $O(n^2d_{\text{model}})$，FFN 常含 $O(nd_{\text{model}}^2)$ 的矩阵计算。短序列、大模型时 FFN 可能占更多 FLOPs；超长序列时 $n^2$ 项与中间矩阵成为瓶颈。

理论复杂度不能替代硬件测量。FlashAttention 改善访存并保持精确结果；稀疏 Attention 减少连接；MoE 改变 FFN 的激活参数；状态空间模型换用另一种序列主干。这些方案处理不同成本来源。

---

## 能力边界

Transformer 的优势来自并行训练、短信息路径与统一 token 接口。它并不自动拥有：

- 完整位置关系：因果 mask 只规定可见顺序，坐标与距离还取决于具体位置机制；
- 事实可靠性：语言建模目标只要求预测分布；
- 无限上下文：计算、缓存和训练分布都有边界；
- 严格可解释性：Attention 权重不是因果证明；
- 多模态对齐：图像 token 还需编码器、连接器和跨模态目标。

视觉 token、跨模态连接器与对齐目标共同决定 Transformer 如何处理图像等输入，具体结构见[多模态对齐](../representation/multimodal-alignment.md)。

---

## 参考文献

- Vaswani, A. et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
- Xiong, R. et al. (2020). *On Layer Normalization in the Transformer Architecture*.
- Raffel, C. et al. (2020). [*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*](https://jmlr.org/papers/v21/20-074.html).
