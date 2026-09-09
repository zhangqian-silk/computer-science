# Scaling、训练计算量与资源预算

Scaling 分析把模型参数、训练 token、FLOPs、设备效率、时间和成本连接起来。它适合做数量级规划，不替代真实 kernel profile、收敛实验或数据质量评估。

::: info 符号与约定
$N$ 是参与近似的非 Embedding 参数量，$D$ 是训练 token 数，$C$ 是训练 FLOPs；$\mathcal{V}$ 为词表，$d_{\text{model}}$ 为隐藏维度。设备峰值以 FLOP/s 为单位，时间公式得到秒。本文使用稠密自回归 Transformer 的概念近似，不给出现行硬件的性能承诺。
:::

---

## 参数与训练 token

设非 embedding 模型参数量为 $N$，训练 token 数为 $D$。Decoder-only Transformer 的参数主项来自 Attention 投影和 FFN 矩阵；词表 embedding 还约包含 $|\mathcal{V}|d_{\text{model}}$ 个参数。Gated FFN、GQA、MoE、tied embedding 和 bias 会改变系数。

读取模型配置时应逐矩阵求和建立精确值，再用近似式检查数量级。参数量不随 batch 或序列长度变化，激活与 Attention 工作量则随 token shape 变化。

---

## 训练 FLOPs 近似

稠密 Transformer 常用：

$$
C_{train}\approx kND
$$

$k$ 常被近似为 6，用于表示 forward、backward 对权重和激活的矩阵计算。但它不是通用常数：Attention 的序列二次项、embedding、MoE 激活参数、activation recompute、稀疏/状态空间主干和 optimizer kernel 都会改变实际工作。

计算预算文档必须写明公式包含哪些算子，以及 $D$ 是原始 token、有效 loss token 还是包含 padding 的执行 token。

### 系数 6 来自哪一步

考虑一个稠密矩阵 $W\in\mathbb{R}^{a\times b}$，有 $ab$ 个参数。对一个输入向量，前向矩阵乘约需 $2ab$ FLOPs；反向对输入计算梯度约需 $2ab$，对权重形成梯度约需 $2ab$。把大量主导线性层相加，就得到每 token 约 $6N$ FLOPs，再乘训练 token 数 $D$。

这是运算计数，不是固定次数的完整模型运行。注意力中 $QK^\top$ 和 $AV$ 不直接对应新的模型参数，但仍需计算；序列很长时不能继续忽略。重计算则会额外执行部分前向，所以 $6ND$ 不是实际设备执行指令的精确总数。小词表查表和大型词表输出投影也不能仅因同属「embedding 相关」就用同一种成本处理。

---

## 从 FLOPs 到时间

若有 $G$ 个设备，单设备目标精度峰值为 $P_{peak}$，端到端有效利用率为 $u$，理想训练时间近似：

$$
T\approx\frac{C_{train}}{G\times P_{peak}\times u}
$$

$u$ 应来自相近模型、shape、并行和拓扑的实测 MFU，而不是任意选取。随着设备数增加，Collective、Pipeline bubble、Checkpoint 和 Straggler 会让 $u$ 改变。

分子和利用率须同口径：C 若是不含重算的模型工作量，可配 MFU；若 C 已计重算等实际硬件工作，则应配相应 HFU。把含重算的 C 再除以 MFU 会重复计入该代价。

若 $T$ 以上式得到的秒计，GPU-hour 为 $G\times T/3\,600$；费用还应包含 CPU、存储、网络、失败重算、空闲预留和数据处理。只按 GPU 标价乘理想时间会低估总成本。

---

## Compute-optimal 不是固定配方

Scaling Law 研究 loss 随模型、数据和计算预算的经验关系。Chinchilla 类结论指出，在其模型家族、数据和训练设定下，给定计算预算时模型与训练 token 应共同扩展。它不是所有领域、数据质量、后训练或部署目标的永久比例。

规划时区分：

- 经验 scaling fit 的数据范围；
- 目标是最低预训练 loss、最小训练成本还是最低推理成本；
- 数据质量、重复和领域覆盖；
- 大模型推理成本是否限制最终部署；
- 后训练和 test-time compute 是否改变总系统预算。

### 固定计算量下，为什么模型与数据要一起考虑

一种经验拟合形式为：

$$
\mathcal{L}(N,D)\approx\mathcal{L}_\infty+\frac{a}{N^\alpha}+\frac{b}{D^\beta}
$$

$a,b,\alpha,\beta>0$ 由实验拟合，$\mathcal{L}_\infty$ 是拟合中的剩余损失项；两项分别近似描述模型容量有限和数据有限的影响。这不是数学上对任意语言模型成立的定理。

用简化预算 $C=6ND$ 消去 $D$，得到：

$$
\mathcal{L}(N,C/6N)
\approx\mathcal{L}_\infty+aN^{-\alpha}
+b(6N/C)^\beta
$$

模型增大时第一项下降，但同预算下训练 token 减少，第二项上升。对 $N$ 求导并令零：

$$
\alpha aN^{-\alpha}
=\beta b(6N/C)^\beta
$$

由此得到幂次关系：

$$
N_{\text{opt}}\propto C^{\frac{\beta}{\alpha+\beta}},
\qquad
D_{\text{opt}}\propto C^{\frac{\alpha}{\alpha+\beta}}
$$

当两个指数接近时，参数和 token 随计算预算近似各按平方根增长；并不能仅从 $C=6ND$ 推出固定的 token/参数比例，比例还取决于拟合系数和目标。

Kaplan 等人的 2020 年工作系统拟合了神经语言模型的幂律关系，其计算最优建议更偏向扩大模型；Hoffmann 等人的 2022 年工作重新研究不同模型规模与训练 token 分配，指出当时许多大模型相对训练不足，支持模型和数据更均衡地扩展。Chinchilla 的代表对比是在与 Gopher 相近训练预算下使用更小模型和更多 token；改变的是资源分配，而不是简单证明「小模型优于大模型」。

两篇工作的拟合范围、训练调度与估计方法不同，不应把差异写成同一公式的代数矛盾。更不能把某组拟合系数跨数据域、架构和训练阶段直接外推。

### 同一预算不代表同一效果

仅作数量级示例：方案 A 为 $N=7\times10^9,D=140\times10^9$，方案 B 为 $N=14\times10^9,D=70\times10^9$。两者的近似训练计算都是：

$$
C=6ND=5.88\times10^{21}\text{ FLOPs}
$$

这只能说明两方案在该简化模型下计算相同，不能给出谁 loss 更低。若假设集群端到端有效算力为 $10^{17}$ FLOP/s，则时间为 $58\,800$ 秒，约 16.3 小时；这是给定假设下的除法结果，不是任何实际集群 benchmark。

若模型将服务大量推理请求，较小模型接受更多训练可能在部署总成本上更合适，即便不是纯预训练 loss 的最优点。训练最优、延迟最优与生命周期成本最优是不同目标；评估时应同时检查目标任务质量和资源约束。

---

## 存储与网络预算

除计算外，还要估算：

- 原始/Tokenized 数据与 shuffle read bandwidth；
- Checkpoint 大小、保存频率、保留数量和写入带宽；
- 参数/梯度/激活 Collective 字节；
- 故障导致的平均丢失 step；
- 权重转换、评测和推理制品存储。

这些预算与训练时间共同决定集群能否持续运行。计算资源充足但 Checkpoint 写入阻塞，仍会降低有效利用率。

---

## CPU 路线

所有数量级模型都可在 CPU 上完成，并用小模型实测参数、FLOPs、step time 与 scaling fit 的方法。CPU 可验证公式和规划表，但得到的利用率只能用于同一 CPU workload，不能代替 GPU MFU。

---

## 参考文献

- Kaplan, J. et al. (2020). [*Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361). 经验幂律、训练计算估计与资源分配。
- Hoffmann, J. et al. (2022). [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556). 不同估计方法、参数化损失拟合与 Chinchilla/Gopher 的预算对比；本文推导只采用其中一种简化形式。
