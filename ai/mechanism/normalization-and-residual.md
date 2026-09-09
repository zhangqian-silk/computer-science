# 归一化、残差与 Dropout

Transformer Block 通过残差路径保存原表示，通过归一化控制每个 token 的特征尺度。二者共同影响深层网络的梯度传播、数值稳定性和 kernel 组织方式。

<NormalizationExplorer />

::: info 符号与约定
$d$ 是单个 token 的隐藏维度，$\gamma,\beta\in\mathbb{R}^d$ 是可训练缩放与偏置，$\epsilon>0$ 防止分母接近零；$F$ 是保持输入输出形状的子层，$J_F$ 是其 Jacobian。这里的归一化沿特征维进行，不是对 token 概率做 softmax。
:::

---

## LayerNorm 与 RMSNorm

对单个 token 的隐藏向量 $x\in\mathbb{R}^{d}$，LayerNorm 为：

$$
\mu=\frac{1}{d}\sum_{i=1}^{d} x_i,
\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\operatorname{LN}(x)
=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

RMSNorm 不减均值：

$$
\operatorname{RMSNorm}(x)
=
\gamma\odot
\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2+\epsilon}}
$$

两者都沿隐藏维归约，不在 batch 或序列维之间混合样本。实现通常用 FP32 或更高精度的累加路径计算统计量，再转换回目标 dtype。

### 归一化到底保留了什么

为看清差别，暂设 $\gamma=(1,1)$、$\beta=(0,0)$，忽略 $\epsilon$，取 $x=(1,3)$：

- LN：均值为 2，方差为 1，得到 $(-1,1)$；
- RMSNorm：均方为 5，得到 $(1/\sqrt{5},3/\sqrt{5})\approx(0.447,1.342)$。

两者控制的是不同统计量。LN 在仿射变换前去掉均值和尺度；RMSNorm 只控制均方根，因此保留非零均值。在交互图里给所有通道加同一常数，可以观察前者的平移不变性与后者的变化。

忽略 $\epsilon$ 时，正比例缩放输入不改变两种归一化的结果；加入 $\epsilon$ 后这只是近似性质。训练得到的 $\gamma$ 与 $\beta$ 又允许网络重新调节各通道，所以最终输出不保证均值为 0、方差为 1。Norm 也不是把向量变成 $L_2$ 单位向量：未仿射的 RMSNorm 输出，其平方和约为 $d$，不是 1。

与 BatchNorm 的关键区别是统计维度。这里每条序列、每个 token 独立求统计量，不需要汇总 batch 中其他样本，也不依赖训练期保存的 running mean。因而小 batch 或变长序列不会像批统计那样直接改变归一化的参照群体。

---

## Pre-Norm 与 Post-Norm

Pre-Norm 子层写作：

$$
y=x+F(\operatorname{Norm}(x))
$$

Post-Norm 写作：

$$
y=\operatorname{Norm}(x+F(x))
$$

原始 Transformer 使用 Post-Norm，现代大语言模型常见 Pre-Norm 或其变体。两者不能只靠移动一行代码互换：初始化、学习率、深度稳定性和已训练权重都与选择绑定。

残差加法要求子层输出与输入具有相同形状。它提供恒等路径，但不表示每一层都只是微小变化；残差分支的尺度仍受初始化、归一化和训练动态影响。

### 为什么顺序会影响梯度

对 Pre-Norm 子层应用链式法则：

$$
\frac{\partial y}{\partial x}
=I+J_F(\operatorname{Norm}(x))J_{\operatorname{Norm}}(x)
$$

梯度除了经过归一化与子层，还存在直接的恒等分量 $I$。对 Post-Norm：

$$
\frac{\partial y}{\partial x}
=J_{\operatorname{Norm}}(x+F(x))(I+J_F(x))
$$

即便沿残差路径，梯度也必须经过外侧 Norm 的 Jacobian。多层连乘后，两者的尺度行为不相同。这解释了为什么残差与 Norm 位置需要一起讨论，但不能推出「Pre-Norm 永远不会梯度爆炸」：子层 Jacobian、残差增长以及最终输出 Norm 仍然有影响。

Xiong 等人的 2020 年论文使用初始化时的平均场分析及翻译、语言建模实验，研究 Post-LN 的大梯度与 warm-up 需求，以及 Pre-LN 的改进。结论依赖其初始化与实验配置，不是把任何现成权重挪成 Pre-LN 都能省掉 warm-up 的保证。RMSNorm 则是在另一个维度上简化归一化统计，两项设计可以组合，并不存在必须先 LN 再 RMSNorm 的单线替代关系。

---

## Dropout 的阶段差异

训练时 Dropout 对激活乘随机 mask，并按保留概率缩放；推理时关闭随机丢弃。由此产生两个工程要求：

- 训练和 eval 模式必须显式切换；
- 分布式重算、Checkpoint 恢复和确定性测试要管理随机数状态。

若丢弃概率为 $p_{\text{drop}}<1$，$m_i\sim\operatorname{Bernoulli}(1-p_{\text{drop}})$，常用 inverted dropout 为：

$$
\widetilde{x}_i=\frac{m_ix_i}{1-p_{\text{drop}}},\qquad
\mathbb{E}[\widetilde{x}_i\mid x_i]=x_i
$$

缩放保持的是这一层激活的条件期望，不是整张非线性网络输出的期望严格不变。它也不是推理时要重复运行多次并投票的要求。对同一训练 batch 做 activation checkpointing 重算时，若随机 mask 改变，重算就不再是原前向路径，所得梯度也随之改变。

许多现代 LLM 预训练配置可能把部分 Dropout 设为零，但这不是 Transformer 的结构定律。阅读具体模型时应检查配置，而不是从模型家族推断。

---

## Kernel 与内存

Norm 包含读输入、归约、缩放和写输出，算术强度通常不高。融合可以减少中间 Tensor 和 HBM 往返，例如把 residual add、dropout 与 norm 组合为一个 kernel。融合前后应验证：

- 数值容差；
- 随机 mask 语义；
- 非连续输入和尾部维度；
- forward 与 backward；
- 不同 dtype 的累加规则。

CPU 可以用向量化和线程内归约验证同样语义；GPU 性能还取决于 block 划分、Shared Memory、warp reduction 和融合后的寄存器压力。

---

## 读者检查

分析一个 Block 时，应先标出每个 Norm 所在位置、归约维度、累加 dtype、残差来源和 Dropout 是否启用。训练不稳定或推理结果不一致时，这些信息比笼统地说「用了 LayerNorm」更有诊断价值。

---

## 参考文献

- Ba, J. L. et al. (2016). [*Layer Normalization*](https://arxiv.org/abs/1607.06450). 每样本归一化的定义与循环网络等场景中的实验。
- Zhang, B. and Sennrich, R. (2019). [*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467). 研究去掉重中心化、保留重缩放的效果，不是所有任务上与 LN 数值等价的证明。
- Xiong, R. et al. (2020). [*On Layer Normalization in the Transformer Architecture*](https://proceedings.mlr.press/v119/xiong20b.html). 归一化位置、初始化梯度与 warm-up 的关联。
- Srivastava, N. et al. (2014). [*Dropout: A Simple Way to Prevent Neural Networks from Overfitting*](https://jmlr.org/papers/v15/srivastava14a.html). 随机丢弃正则化；本文采用训练时缩放的等价常用约定。
