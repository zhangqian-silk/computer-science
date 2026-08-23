# N-gram：用有限上下文估计语言概率

N-gram 语言模型把「下一个 token 是什么」转化为条件概率估计问题：先统计局部序列出现了多少次，再用平滑、回退或插值处理稀疏事件。它的价值不仅在于构成传统语言模型，也在于提供了一套可以逐项核对的概率建模范式。

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$\mathcal{V}$ 是固定词表，$w_t$ 是第 $t$ 个 token，$T$ 是序列长度，$n$ 是 N-gram 阶数，$h$ 是局部历史；$C(\cdot)$ 表示计数，$P(\cdot)$ 表示概率；$k$ 是加法平滑强度，$D$ 是绝对折扣，$\alpha,\lambda$ 是回退或插值权重。
:::

---

## 从链式法则到有限窗口

任意序列 $w_1,w_2,\ldots,w_T$ 的联合概率都可以按链式法则分解：

$$
P(w_1,\ldots,w_T)
=
\prod_{t=1}^{T}P(w_t\mid w_{<t})
$$

直接估计 $P(w_t\mid w_{<t})$ 需要区分几乎所有可能的历史。N-gram 使用 $n-1$ 阶马尔可夫近似，只保留最近的 $n-1$ 个 token：

$$
P(w_t\mid w_{<t})
\approx
P(w_t\mid w_{t-n+1:t-1})
$$

| 名称 | $n$ | 预测时使用的历史 | 例子 |
| --- | ---: | --- | --- |
| Unigram | 1 | 不使用上下文 | $P(\text{学习})$ |
| Bigram | 2 | 前 1 个 token | $P(\text{学习}\mid\text{爱})$ |
| Trigram | 3 | 前 2 个 token | $P(\text{学习}\mid\text{我，爱})$ |

句首和句尾通常加入 `<BOS>` 与 `<EOS>`：

$$
P(\text{<BOS> 我 爱 学习 <EOS>})
=
P(\text{我}\mid\text{<BOS>})
P(\text{爱}\mid\text{我})
P(\text{学习}\mid\text{爱})
P(\text{<EOS>}\mid\text{学习})
$$

边界符使模型能够学习句子怎样开始和停止。分词方式、大小写、标点以及低频词如何映射到 `<UNK>`，都会改变计数事件，因此属于模型定义的一部分。

---

## 最大似然估计

记 $\mathcal{V}$ 为固定词表，$h$ 为长度不超过 $n-1$ 的历史，$C(h,w)$ 为序列 $(h,w)$ 的计数，则最大似然估计为：

$$
P_{\mathrm{MLE}}(w\mid h)
=
\frac{C(h,w)}{C(h)}
$$

其中：

$$
C(h)=\sum_{w'\in\mathcal{V}}C(h,w')
$$

### 最小计数推演

语料包含三句话：

- `<BOS> 我 爱 学习 <EOS>`；
- `<BOS> 我 爱 编程 <EOS>`；
- `<BOS> 你 爱 学习 <EOS>`。

历史「爱」一共出现 3 次，后继词计数为：

| 后继词 $w$ | $C(\text{爱},w)$ | $P_{\mathrm{MLE}}(w\mid\text{爱})$ |
| --- | ---: | ---: |
| 学习 | 2 | $2/3$ |
| 编程 | 1 | $1/3$ |
| 数学 | 0 | $0$ |

因此：

$$
P(\text{学习}\mid\text{爱})=\frac{2}{3},\qquad
P(\text{编程}\mid\text{爱})=\frac{1}{3}
$$

「爱数学」即使语言上合理，只要没有在训练语料中出现，MLE 仍会给出 0。由于句子概率是各条件概率的乘积，一个未见 N-gram 就会使整句概率变为 0。

未见 N-gram 与未登录词是两个问题：前者由平滑处理；后者通常先通过固定词表、子词切分或 `<UNK>` 映射转化为词表内事件。

---

## 概率质量为何需要重新分配

平滑不是简单地「给零概率补一个小数」，而是从已见事件中释放一部分概率质量，再分配给低频或未见事件。任何合法方法都必须满足：

$$
P(w\mid h)\geq 0,\qquad
\sum_{w\in\mathcal{V}}P(w\mid h)=1
$$

```mermaid
flowchart LR
	C["N-gram 计数"] --> M["MLE 分布"]
	M --> D["折扣已见事件"]
	D --> R["释放概率质量"]
	R --> L["依据低阶统计重新分配"]
	L --> P["归一化条件分布"]
```

不同平滑方法的核心差异，是「释放多少」以及「按照什么规则重新分配」。

---

## 加法平滑：最直观的概率重分配

加 $k$ 平滑为每个候选后继增加相同的伪计数：

$$
P_{\text{add-}k}(w\mid h)
=
\frac{C(h,w)+k}{C(h)+k|\mathcal{V}|},\qquad k>0
$$

$k=1$ 时称为 Laplace 平滑。分母增加 $k|\mathcal{V}|$，保证所有候选概率之和仍然为 1。

### 数值推演

假设历史「喝」出现 3 次，其后「水」出现 2 次、「茶」出现 1 次。固定后继词表为：

$$
\mathcal{V}=\{\text{水},\text{茶},\text{咖啡},\text{<EOS>}\},\qquad |\mathcal{V}|=4
$$

MLE 给出：

$$
P_{\mathrm{MLE}}(\text{水}\mid\text{喝})=\frac{2}{3},\qquad
P_{\mathrm{MLE}}(\text{咖啡}\mid\text{喝})=0
$$

应用 Laplace 平滑后：

$$
P_{\mathrm{Laplace}}(\text{水}\mid\text{喝})
=\frac{2+1}{3+4}=\frac{3}{7}
$$

$$
P_{\mathrm{Laplace}}(\text{咖啡}\mid\text{喝})
=\frac{0+1}{3+4}=\frac{1}{7}
$$

完整分布为：

| 后继词 | 平滑后计数 | 条件概率 |
| --- | ---: | ---: |
| 水 | 3 | $3/7$ |
| 茶 | 2 | $2/7$ |
| 咖啡 | 1 | $1/7$ |
| `<EOS>` | 1 | $1/7$ |

概率和为 $1$，零概率消失；但「水」从 $2/3\approx0.667$ 降到 $3/7\approx0.429$，说明加一操作明显稀释了已见事件。

### 为什么大词表会过平滑

当 $|\mathcal{V}|=100\,000$、$C(h)=10$ 时，即使某个后继出现了 10 次，Laplace 估计也只有：

$$
\frac{10+1}{10+100\,000}\approx 1.10\times10^{-4}
$$

未见后继的概率则为：

$$
\frac{1}{10+100\,000}\approx 1.00\times10^{-5}
$$

大量未见事件共同占用了过多概率质量，见过 10 次与从未见过的事件只相差约 11 倍。较小的 $k$ 可以减弱这种现象，却仍然把所有未见事件视为同等合理。

---

## 回退与插值：利用较短的历史

高阶 N-gram 更具体，但计数稀疏；低阶 N-gram 覆盖更广，但丢失了上下文。回退与插值都组合多阶统计，使用方式并不相同。

### 回退

回退仅在高阶事件不可用时调用低阶模型。一个简化的折扣回退形式为：

$$
P_{\mathrm{bo}}(w\mid h)
=
\begin{cases}
P_{\mathrm{disc}}(w\mid h), & C(h,w)>0\\
\alpha(h)P_{\mathrm{bo}}(w\mid h'), & C(h,w)=0
\end{cases}
$$

$h'$ 是删除最左侧 token 后的较短历史。$\alpha(h)$ 不能随意指定，它必须把高阶模型释放的剩余概率质量，按照低阶分布在未见事件上的相对比例进行归一化。

假设候选词为「水、茶、咖啡、牛奶」，高阶历史 $h$ 的计数为：

$$
C(h,\cdot)=(3,1,0,0),\qquad C(h)=4
$$

使用绝对折扣 $D=0.5$，已见事件获得：

$$
P_{\mathrm{disc}}(\text{水}\mid h)=\frac{3-0.5}{4}=0.625
$$

$$
P_{\mathrm{disc}}(\text{茶}\mid h)=\frac{1-0.5}{4}=0.125
$$

两者合计 $0.75$，因此有 $0.25$ 的概率质量留给未见事件。若低阶分布为：

| 候选词 | $P_{\mathrm{lower}}(w)$ |
| --- | ---: |
| 水 | $0.45$ |
| 茶 | $0.20$ |
| 咖啡 | $0.20$ |
| 牛奶 | $0.15$ |

未见事件在低阶模型中的总质量为 $0.20+0.15=0.35$，所以回退分配为：

$$
P_{\mathrm{bo}}(\text{咖啡}\mid h)
=0.25\times\frac{0.20}{0.35}\approx0.143
$$

$$
P_{\mathrm{bo}}(\text{牛奶}\mid h)
=0.25\times\frac{0.15}{0.35}\approx0.107
$$

最终分布 $(0.625,0.125,0.143,0.107)$ 的和为 1。这个例子说明：回退并不是发现计数为 0 后直接照搬低阶概率，而是要对剩余概率质量重新归一化。

Katz Backoff 使用 Good–Turing 等方法折扣已见事件，再把释放的质量交给低阶模型；高计数事件保留高阶统计，低计数和未见事件更多依赖低阶分布。

### 插值

插值始终同时使用不同阶的模型，即使高阶事件已经出现：

$$
P_{\mathrm{interp}}(w_t\mid w_{t-2},w_{t-1})
=
\lambda_3P_3(w_t\mid w_{t-2},w_{t-1})
+\lambda_2P_2(w_t\mid w_{t-1})
+\lambda_1P_1(w_t)
$$

其中 $\lambda_i\geq0$ 且 $\sum_i\lambda_i=1$。权重通常在验证集上估计，也可以依赖上下文计数动态变化。

继续使用前面的高阶分布，令高阶 MLE 为 $(0.75,0.25,0,0)$、低阶分布为 $(0.45,0.20,0.20,0.15)$，并取高阶权重 $0.7$：

$$
P_{\mathrm{interp}}=0.7P_{\mathrm{high}}+0.3P_{\mathrm{lower}}
$$

得到：

| 候选词 | 计算 | 插值概率 |
| --- | --- | ---: |
| 水 | $0.7\times0.75+0.3\times0.45$ | $0.660$ |
| 茶 | $0.7\times0.25+0.3\times0.20$ | $0.235$ |
| 咖啡 | $0.7\times0+0.3\times0.20$ | $0.060$ |
| 牛奶 | $0.7\times0+0.3\times0.15$ | $0.045$ |

插值中，低阶信息会影响所有候选；回退中，低阶信息主要接管高阶模型无法可靠估计的部分。

需要注意，若某个词在所有阶都没有出现，单纯插值仍会得到 0。此时仍需要 `<UNK>`、子词建模或额外平滑。

---

## Kneser–Ney：低阶概率应表达续接能力

普通低阶模型使用词频 $C(w)$。这会产生一个问题：某个词可能总频率很高，却只出现在一种固定搭配中，因此未必适合作为新上下文的后继。

例如：

- 「金山」出现 1000 次，但几乎都接在「旧」之后；
- 「苹果」出现 500 次，却分别接在「吃、买、红、切」等多种词之后。

面对一个未见上下文，词频模型可能偏向「金山」；续接能力则会偏向能出现在更多不同前缀后的「苹果」。

### 续接概率

定义：

$$
N_{1+}(\ast,w)=|\{v:C(v,w)>0\}|
$$

它表示 $w$ 曾经出现在多少种不同前缀之后。所有不同 Bigram 类型的数量为：

$$
N_{1+}(\ast,\ast)=|\{(v,w):C(v,w)>0\}|
$$

续接概率为：

$$
P_{\mathrm{cont}}(w)
=
\frac{N_{1+}(\ast,w)}{N_{1+}(\ast,\ast)}
$$

若语料共有 20 种不同 Bigram，「金山」只接在 1 种前缀后，而「苹果」接在 4 种前缀后，则：

$$
P_{\mathrm{cont}}(\text{金山})=\frac{1}{20}=0.05
$$

$$
P_{\mathrm{cont}}(\text{苹果})=\frac{4}{20}=0.20
$$

即使「金山」的绝对词频更高，它在未知上下文中的低阶概率仍低于「苹果」。

### 二元插值式 Kneser–Ney

使用绝对折扣 $D$ 时：

$$
P_{\mathrm{KN}}(w\mid h)
=
\frac{\max(C(h,w)-D,0)}{C(h)}
+\lambda(h)P_{\mathrm{cont}}(w)
$$

第一项保留折扣后的已见计数，第二项按照续接概率分配释放的质量。记：

$$
N_{1+}(h,\ast)=|\{w:C(h,w)>0\}|
$$

已见事件折扣后的总质量为：

$$
\sum_{w:C(h,w)>0}\frac{C(h,w)-D}{C(h)}
=
\frac{C(h)-D\,N_{1+}(h,\ast)}{C(h)}
$$

因此释放的质量，也就是低阶项权重，为：

$$
\lambda(h)
=
1-\frac{C(h)-D\,N_{1+}(h,\ast)}{C(h)}
=
\frac{D\,N_{1+}(h,\ast)}{C(h)}
$$

若 $C(h)=10$、历史 $h$ 有 2 种不同后继、$D=0.75$，则：

$$
\lambda(h)=\frac{0.75\times2}{10}=0.15
$$

对于未见 Bigram，折扣项为 0。「金山」与「苹果」从低阶项获得的概率分别为：

$$
0.15\times0.05=0.0075,\qquad
0.15\times0.20=0.03
$$

### 三元递归

Trigram 的插值式 Kneser–Ney 继续沿较短历史递归：

$$
\begin{aligned}
P_{\mathrm{KN}}(w_i\mid w_{i-2},w_{i-1})
={}&
\frac{\max(C(w_{i-2},w_{i-1},w_i)-D,0)}{C(w_{i-2},w_{i-1})}\\
&+\lambda(w_{i-2},w_{i-1})P_{\mathrm{KN}}(w_i\mid w_{i-1})
\end{aligned}
$$

递归中的低阶计数不是简单的原始 Bigram 频次，而是由更高阶上下文诱导出的续接计数。这一点使 Kneser–Ney 与「普通绝对折扣后再接一个 Unigram」有本质区别。

实际系统常使用 Modified Kneser–Ney：根据计数为 1、2、3 及以上的事件使用不同折扣值，以更细致地匹配频率分布。

---

## 方法对比

| 方法 | 已见事件如何处理 | 未见事件依据 | 主要特点 |
| --- | --- | --- | --- |
| MLE | 保留原始相对频率 | 概率为 0 | 简单，但无法泛化到未见组合 |
| 加 $k$ | 所有事件增加相同伪计数 | 均匀先验 | 易理解，大词表下容易过平滑 |
| Katz Backoff | 对已见事件折扣 | 较低阶分布 | 高阶可用时优先使用高阶统计 |
| 线性插值 | 混合各阶概率 | 各阶共同提供 | 已见事件也会吸收低阶信息 |
| Kneser–Ney | 绝对折扣 | 续接概率 | 区分总词频与上下文多样性 |

---

## 训练、生成与评估

训练不依赖梯度下降，而是完成计数、折扣参数估计和概率表构建：

```
TRAIN-N-GRAM(corpus, n)
	counts ← empty count tables for orders 1..n
	for each sentence in corpus do
		tokens ← [<BOS>] + TOKENIZE(sentence) + [<EOS>]
		for order ← 1 to n do
			for each window in SLIDING-WINDOW(tokens, order) do
				counts[order][window] ← counts[order][window] + 1
			end for
		end for
	end for

	parameters ← ESTIMATE-DISCOUNTS(counts)
	return BUILD-PROBABILITY-TABLES(counts, parameters)
```

生成时，模型截取最近 $n-1$ 个 token，查询条件分布并采样或选择概率最大的后继；若使用回退模型，则逐级缩短历史，直到获得可用分布。生成 `<EOS>` 后停止。

测试集平均负对数似然为：

$$
\mathcal{L}
=
-\frac{1}{T}\sum_{t=1}^{T}\log P(w_t\mid w_{t-n+1:t-1})
$$

困惑度为：

$$
\operatorname{PPL}=\exp(\mathcal{L})
$$

不同分词方案和词表会改变 token 数与概率空间，因此 PPL 不能脱离这些条件直接比较，详见[语言模型评估](../evaluation/language-model-evaluation.md)。平滑强度与插值权重也应在验证集上选择，不能用测试集调参。

---

## 模型边界

N-gram 的优势是训练快、概率可解释、错误容易定位，适合作为轻量补全、拼写纠错、语音识别融合与神经模型的对照基线。其限制来自有限离散上下文：

- $n$ 增大时，可能组合数按 $|\mathcal{V}|^n$ 增长；
- 语义相近的词不能自然共享统计强度；
- 超出窗口的依赖被直接丢弃；
- 领域变化会使计数和折扣参数迅速失效；
- 平滑只能重新分配已有概率质量，不能凭空获得语义知识。

[NPLM](./nplm.md)保留固定窗口，却用连续词向量和神经网络让相似上下文共享参数；[RNN](./rnn.md)进一步用递归状态承载变长历史；[Transformer](./transformer.md)则通过 Attention 缩短长距离 token 之间的信息路径。

---

## 参考文献

- Katz, S. M. (1987). *Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer*.
- Kneser, R., and Ney, H. (1995). *Improved Backing-Off for M-gram Language Modeling*.
- Chen, S. F., and Goodman, J. (1999). *An Empirical Study of Smoothing Techniques for Language Modeling*.
- Jurafsky, D., and Martin, J. H. *Speech and Language Processing*.
