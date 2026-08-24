# Attention：按当前查询读取一组信息

Attention 把固定摘要改成按需读取。给定一个 query 和若干 key-value 对，模型先计算 query 与各 key 的匹配程度，再用归一化权重聚合对应 value。它既可以连接 Encoder 与 Decoder，也可以连接文本与图像、当前状态与外部记忆。

<AttentionReadExplorer />

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$q,k_i,v_i$ 是单个 Query、Key、Value，$m$ 是键值对数量，$e_i,\alpha_i,c$ 是匹配分数、归一化权重和读取结果；$Q,K,V$ 是矩阵形式，$n_q,n_k$ 是两侧长度，$d_k,d_v$ 是特征维度，$M$ 是 mask，$n_{\text{head}}$ 是头数。
:::

~~~mermaid
flowchart LR
	Q["Query：现在需要什么"] --> S["匹配分数"]
	K["Keys：每项可按什么检索"] --> S
	S --> N["softmax 权重"]
	N --> A["加权聚合"]
	V["Values：实际读取的内容"] --> A
	A --> O["上下文向量"]
~~~

---

## 一般形式

对 query $q$ 与键值集合 $\{(k_i,v_i)\}_{i=1}^{m}$，先计算：

$$
e_i=s(q,k_i)
$$

再归一化并聚合：

$$
\alpha_i=\frac{\exp(e_i)}{\sum_{j=1}^{m}\exp(e_j)},\qquad
c=\sum_{i=1}^{m}\alpha_i v_i
$$

key 只用于匹配，value 才是最终被读取的内容。二者可以来自同一表示的不同线性投影，也可以承担完全不同的接口职责。

常见打分函数包括：

| 打分 | 形式 | 特点 |
| --- | --- | --- |
| 点积 | $q^\top k$ | 可直接用矩阵乘法 |
| 缩放点积 | $q^\top k/\sqrt{d_k}$ | 控制高维点积的尺度 |
| 加性 | $v_a^\top\tanh(W_q q+W_k k)$ | 额外引入小型网络 |

缩放点积并非任意常数。若 $q$、$k$ 各维方差近似为 1，点积方差会随 $d_k$ 增长；除以 $\sqrt{d_k}$ 可避免 softmax 输入过大而过早饱和。

### 一个 Query 的读取过程

设 $d_k=2$：

$$
q=[1,0],\quad
k_1=[1,0],\quad k_2=[0,1]
$$

对应 value 为：

$$
v_1=[2,0],\qquad v_2=[0,4]
$$

缩放点积分数为：

$$
e=\left[\frac{1}{\sqrt{2}},0\right]\approx[0.707,0]
$$

softmax 权重约为：

$$
\alpha\approx[0.670,0.330]
$$

最终读取结果是 value 的加权和：

$$
c=0.670[2,0]+0.330[0,4]\approx[1.340,1.320]
$$

输出既不是最相似的 key，也不是简单复制某个 value，而是根据 key 匹配得到的权重混合 value。若第二项被 mask，softmax 前分数变成 $(0.707,-\infty)$，权重变成 $(1,0)$，输出就是 $v_1=[2,0]$。

---

## 矩阵化计算

批量 query、key、value 分别组成：

$$
Q\in\mathbb{R}^{n_q\times d_k},\qquad
K\in\mathbb{R}^{n_k\times d_k},\qquad
V\in\mathbb{R}^{n_k\times d_v}
$$

缩放点积 Attention 为：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

softmax 按行计算。第 $i$ 行表示第 $i$ 个 query 对所有 key 的读取分布。掩码 $M$ 在归一化前把非法连接设为极小值；它可以屏蔽 padding、未来位置或不允许访问的模态。

输出形状为 $n_q\times d_v$。Attention 不要求 query 与 key 的数量相同，因此很适合 Cross-Attention。

训练时，任务损失会同时通过两条路径回传：权重路径更新 query/key 投影，使模型改变「读哪里」；value 与输出路径更新 value 投影，使模型改变「读到什么」。Attention 权重没有独立的正确答案，除非任务额外提供对齐监督。

---

## Cross-Attention 怎样解除固定摘要瓶颈

早期 Seq2Seq 把整个输入压缩成单个向量 $c$。加入 Attention 后，Decoder 在第 $t$ 步用状态 $s_{t-1}$ 查询全部 Encoder 状态：

$$
e_{t,i}=s(s_{t-1},h_i),\qquad
c_t=\sum_i\alpha_{t,i}h_i
$$

每个输出步拥有不同的 $c_t$。生成译文某个词时，模型可集中读取对应源词，而不必让一个固定向量同时保存所有细节。这种权重是软对齐，不要求位置一一对应。

Cross-Attention 还用于：

- 文本 Decoder 读取视觉 token；
- 查询读取检索候选或外部记忆；
- 条件生成模型读取控制信号。

这些场景共享同一个接口：query 来自当前计算流，key-value 来自条件信息。

---

## 多头读取

单个打分空间必须同时承载多种关系。多头 Attention 为每个头使用独立投影：

$$
\operatorname{head}_r
=
\operatorname{Attention}(QW_r^Q,KW_r^K,VW_r^V)
$$

$$
\operatorname{MHA}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_{n_{\text{head}}})W^O
$$

多头允许不同子空间学习不同匹配模式，但不保证每个头都具有稳定、可命名的功能。注意力权重可以用于诊断，不应直接当作严格因果解释。

若 $d_{\text{model}}=512$、头数为 8，常见设置让每头 $d_k=d_v=64$。每个头先独立产生 $n_q\times64$ 输出，拼接后恢复为 $n_q\times512$，再由 $W^O$ 混合各头通道。拆成多头并不会自动减少标准 Attention 的总投影维度或连接数。

---

## 代价与职责边界

分数矩阵大小为 $n_qn_k$。当两侧都很长时，计算和中间存储成为主要瓶颈；稀疏连接、分块 kernel 或低秩近似分别从连接数量、实现方式和数学近似入手降低代价。

Attention 只定义「怎样读取」。它不自带顺序、不定义完整 block，也不决定训练目标：

- 同一序列内部的特化见 [Self-Attention](./self-attention.md)；
- 顺序如何进入打分见[位置表示](./positional-encoding.md)；
- block、残差和三种架构路线见 [Transformer](../model/transformer.md)；
- 受限连接图见[稀疏注意力](./sparse-attention.md)。

---

## 参考文献

- Bahdanau, D., Cho, K., and Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*.
- Luong, M.-T., Pham, H., and Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation*.
- Vaswani, A. et al. (2017). *Attention Is All You Need*.
