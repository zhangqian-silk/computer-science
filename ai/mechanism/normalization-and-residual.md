# 归一化、残差与 Dropout

Transformer Block 通过残差路径保存原表示，通过归一化控制每个 token 的特征尺度。二者共同影响深层网络的梯度传播、数值稳定性和 kernel 组织方式。

<NormalizationExplorer />

---

## LayerNorm 与 RMSNorm

对单个 token 的隐藏向量 $x\in\mathbb{R}^{H}$，LayerNorm 为：

$$
\mu=\frac{1}{H}\sum_i x_i,
\qquad
\sigma^2=\frac{1}{H}\sum_i(x_i-\mu)^2
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
\frac{x}{\sqrt{\frac{1}{H}\sum_i x_i^2+\epsilon}}
$$

两者都沿隐藏维归约，不在 batch 或序列维之间混合样本。实现通常用 FP32 或更高精度的累加路径计算统计量，再转换回目标 dtype。

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

---

## Dropout 的阶段差异

训练时 Dropout 对激活乘随机 mask，并按保留概率缩放；推理时关闭随机丢弃。由此产生两个工程要求：

- 训练和 eval 模式必须显式切换；
- 分布式重算、Checkpoint 恢复和确定性测试要管理随机数状态。

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

## 参考文献

- Ba, J. L. et al. (2016). *Layer Normalization*.
- Zhang, B. and Sennrich, R. (2019). *Root Mean Square Layer Normalization*.
- Xiong, R. et al. (2020). *On Layer Normalization in the Transformer Architecture*.
