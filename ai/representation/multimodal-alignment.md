# 多模态对齐：让不同输入共享可比较的表示

文本、图像和音频的原始数据结构不同，编码器输出的维度与分布也不同。多模态对齐通过成对数据与共享目标，使相关内容在同一表示空间中接近。它可以服务跨模态检索，也可以为生成模型提供条件 token。

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$I_i,T_i$ 是第 $i$ 对图像与文本，$v_i,t_i$ 是归一化后的模态向量，$S$ 是 batch 内相似度矩阵，$B$ 是 batch 大小，$\tau$ 是温度；$H,W$ 是图像高与宽，$P$ 是 patch 边长，$N$ 是 patch 数量；$Z_v$ 是视觉 token，$g$ 是连接器，$\widetilde{Z}_v$ 是映射到语言模型维度后的视觉 token。
:::

---

## 双塔对齐

<MultimodalAlignmentExplorer />

以图文对为例，图像编码器和文本编码器分别产生向量：

$$
v_i=\operatorname{norm}(f_{\text{img}}(I_i)),\qquad
t_i=\operatorname{norm}(f_{\text{text}}(T_i))
$$

一个 batch 的相似度矩阵为：

$$
S_{ij}=\frac{v_i^\top t_j}{\tau}
$$

$\tau>0$ 是对比学习温度，控制 softmax 分布的尖锐程度。

匹配图文位于对角线。对称对比损失同时要求图像找对文本、文本找对图像：

$$
\mathcal{L}
=
\frac{1}{2}
\left(
\mathcal{L}_{\text{image}\rightarrow\text{text}}
+
\mathcal{L}_{\text{text}\rightarrow\text{image}}
\right)
$$

训练完成后，两侧可以独立编码和缓存，因此双塔适合图搜文、文搜图与跨模态聚类。

负样本质量仍是关键。困难负例可以增强属性和关系的区分信号，但不是学习颜色、数量等能力的唯一途径；仍要避免把同一事件的其他相关图像误作负例。

一个 batch 的图文编码会形成 $B\times B$ 相似度矩阵。图像到文本方向对每一行做 softmax，要求第 $i$ 张图找到第 $i$ 段文本；文本到图像方向对每一列做 softmax，要求第 $j$ 段文本找回第 $j$ 张图。对称损失避免只优化其中一个检索方向。

若 batch 中包含「一只红色杯子」和「一只蓝色杯子」两组图文，它们既是彼此的困难负例，也能直接检查模型是否学习颜色属性。若文字只写「杯子」，则两张图都可能与同一句描述相关，把其中一张强制当作负例会制造错误监督。

---

## 图像怎样进入 Transformer

若 H、W 均可被 P 整除，且使用边长与步长均为 P 的不重叠 patch，数量为：

$$
N=\frac{HW}{P^2}
$$

每个 patch 展平并线性投影为 token，再加位置表示送入 Transformer Encoder。这个步骤把二维信号改写为序列，但不自动完成图文对齐；对齐仍来自配对数据和跨模态目标。

例如 $224\times224$ 图像使用 $16\times16$ patch，会得到：

$$
N=\frac{224\times224}{16^2}=196
$$

如果加入一个聚合 token，Encoder 输入长度为 197。patch 变小会保留更多局部细节，同时按面积快速增加 token 数与 Attention 成本。

音频可以按时间帧或谱图片段编码，视频还要保留时间关系。不同模态的 token 化方式决定了局部结构与计算成本。

---

## 检索空间与生成接口不是一回事

双塔通常把每个模态压缩成单个全局向量，适合快速相似度检索。多模态生成模型则需要保留更多局部信息，让语言模型读取一组视觉或音频 token。

设视觉编码器输出：

$$
Z_v\in\mathbb{R}^{N_v\times d_v}
$$

连接器 $g$ 将其映射到语言模型维度：

$$
\tilde{Z}_v=g(Z_v)\in\mathbb{R}^{M\times d_{\text{lm}}}
$$

$g$ 可以是线性投影、带查询的压缩模块或交叉注意力层。它需要解决维度、分布和 token 数量的差异。把向量投到同一维度只是接口兼容，不保证细粒度语义已经对齐。

### CLIP 学到的不是图像描述生成器

CLIP 的训练把一个 batch 中的配对关系当作分类标签。将上文两个方向展开为：

$$
\mathcal{L}_{\text{image}\rightarrow\text{text}}
=-\frac{1}{B}\sum_{i=1}^{B}
\log\frac{\exp(S_{ii})}{\sum_{j=1}^{B}\exp(S_{ij})}
$$

$$
\mathcal{L}_{\text{text}\rightarrow\text{image}}
=-\frac{1}{B}\sum_{i=1}^{B}
\log\frac{\exp(S_{ii})}{\sum_{j=1}^{B}\exp(S_{ji})}
$$

例如两对图文的温度缩放后得分为 $S=\left[\begin{smallmatrix}2&0\\0&2\end{smallmatrix}\right]$，两个方向的正确匹配概率都是 $e^2/(e^2+1)\approx0.881$，平均损失约为 0.127。如果四个得分完全相等，概率为 $1/2$，损失为 $\log2$；仅让所有向量完全重合无法解决配对任务。

这里的 softmax 只在当前候选集合内归一化。概率 0.881 不是「图里有杯子的客观概率」；改变候选文字，就会改变分母和得分解释。

零样本分类时，把类别名写进提示文本，例如「一张猫的照片」和「一张狗的照片」，分别编码，再用图像与候选文本的相似度选类别。训练时没有为每个下游数据集单独训练固定分类头，这是 CLIP 迁移接口的关键；但类别描述、提示模板、数据域和类别集合仍会影响结果。

CLIP 原论文比较不同视觉主干、规模与多种下游数据集的迁移，展示自然语言监督的可迁移性。它没有通过下一词损失训练描述解码器，因此仅有 CLIP 编码器不能直接逐词生成图像说明。其零样本分类结果也不能直接当作视觉问答、计数或空间推理的证据。

### 从共享空间到语言模型条件：不同连接器改了什么

| 路线 | 输入到语言模型的接口 | 学习问题 | 代价与边界 |
| --- | --- | --- | --- |
| CLIP 双塔 | 不进入生成器，比较全局向量 | 匹配图文对 | 便于缓存和检索，丢失部分局部细节 |
| 线性或 MLP 投影 | 视觉 token 映射后与文本 token 组合 | 使视觉特征可被语言模型使用 | 接口简单，但视觉 token 数仍占上下文与计算 |
| BLIP-2 Q-Former | 可学习 query 从视觉特征提取一组输出 | 在冻结视觉编码器与语言模型间建立桥接 | 控制接口 token 数，同时引入信息压缩瓶颈 |
| Flamingo | Perceiver Resampler 后经插入的 Cross-Attention 读取 | 让冻结语言主干按条件使用视觉信息 | 需专门的交叉注意力结构，不是简单拼接前缀 |

这些路线并非所有步骤依次发生的单线演进。CLIP 主要定义检索空间，BLIP-2 和 Flamingo 主要解决如何接入生成器，线性/MLP 路线则展示较直接的视觉指令适配接口。选择应围绕保留多少空间细节、固定哪些预训练组件、允许多少视觉 token 和训练数据。

常见训练顺序是先固定大部分单模态编码器、训练连接器建立基本对应，再按任务需要联合微调部分或全部组件。冻结策略取决于数据规模与保持原能力的要求，不是结构定义的一部分。

BLIP-2 的两阶段不能仅概括为「训练一个投影层」：第一阶段让 Q-Former 与冻结视觉编码器配合，结合图文对比、匹配和图像条件文本生成目标学习视觉—语言表示；第二阶段把 query 输出接到冻结语言模型，用语言生成信号学习接口。冻结语言模型参数不代表切断输入梯度；梯度仍需要穿过语言模型回到连接器。

跨模态生成的训练和推理可以沿同一数据流理解：

```mermaid
flowchart LR
	I["图像或音频"] --> VE["模态编码器"]
	VE --> Z["局部模态 token"]
	Z --> C["连接器 / Cross-Attention"]
	T["文本前缀"] --> LM["语言模型"]
	C --> LM
	LM --> O["下一文本 token"]
```

对自回归条件生成，训练时由真实文本前缀预测下一 token，梯度是否更新模态编码器、连接器和语言模型取决于冻结策略。前缀拼接接口在推理时先编码模态输入，把条件 token 纳入语言模型 Prefill，再逐 token Decode；Cross-Attention 接口则把视觉状态作为独立可读取的 memory。模态特征通常在一次回答中复用，不必每生成一个词都重新运行视觉编码器。

跨模态检索则不同：图像向量与文本向量分别离线或在线编码，直接比较相似度，不进入语言模型生成循环。二者可能共享视觉主干，但输出粒度和训练目标不能混用。

---

## 对齐失败怎样表现

多模态系统常见的失败是只学到捷径，而非完全不相关：

- 依赖图像中的文字或水印，而非视觉内容；
- 能判断主体类别，却混淆数量、空间关系和属性；
- 训练数据中的共同偏差被映射成错误关联；
- 长尾模态或低资源语言被主流数据分布淹没；
- 检索向量表现良好，但生成连接器丢失局部细节。

评估应按方向分别报告 image-to-text 与 text-to-image 检索，并加入组合属性、关系、否定和分布外样本。生成系统还需检查视觉依据是否真正支持回答，不能只看语言流畅度。

对「图中有几个红杯子」答错，可以沿三层定位：图像缩放是否让小物体不可见；编码器与连接器是否在压缩中丢掉了数量信息；语言模型是否忽略视觉条件而套用了常见答案。比较原图、遮挡图和替换图的输出是诊断思路，但遮挡也会制造分布偏移，不能仅凭一次输出变化断言某个 patch 就是原因。

---

## 与相邻主题的关系

[Embedding](./embedding.md)提供连续表示与相似度的基础；[文本嵌入](./text-embedding.md)解释单模态对比学习与双编码器；[向量检索](./vector-retrieval.md)负责对齐向量的部署。Transformer 只提供编码主干，跨模态训练目标和连接器才定义模态间如何交换信息。

---

## 参考文献

- Radford, A. et al. (2021). [*Learning Transferable Visual Models From Natural Language Supervision*](https://proceedings.mlr.press/v139/radford21a.html). 对称对比训练与零样本分类接口；迁移结论需结合其任务与提示配置。
- Dosovitskiy, A. et al. (2021). [*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929). Patch 序列和视觉 Transformer；该表示步骤本身不是跨模态监督。
- Alayrac, J.-B. et al. (2022). [*Flamingo: a Visual Language Model for Few-Shot Learning*](https://arxiv.org/abs/2204.14198). Resampler、Gated Cross-Attention 与交错图文条件。
- Li, J. et al. (2023). [*BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*](https://proceedings.mlr.press/v202/li23q.html). Q-Former、两阶段目标与冻结组件的桥接。
- Liu, H. et al. (2023). [*Visual Instruction Tuning*](https://arxiv.org/abs/2304.08485). 原始 LLaVA 的视觉投影与视觉指令微调，不代表所有后续连接器版本。
