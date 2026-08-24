# Scaling、训练计算量与资源预算

Scaling 分析把模型参数、训练 token、FLOPs、设备效率、时间和成本连接起来。它适合做数量级规划，不替代真实 kernel profile、收敛实验或数据质量评估。

---

## 参数与训练 token

设非 embedding 模型参数量为 $N$，训练 token 数为 $D$。Decoder-only Transformer 的参数主项来自 Attention 投影和 FFN 矩阵；词表 embedding 还约包含 $VH$ 个参数。Gated FFN、GQA、MoE、tied embedding 和 bias 会改变系数。

读取模型配置时应逐矩阵求和建立精确值，再用近似式检查数量级。参数量不随 batch 或序列长度变化，激活与 Attention 工作量则随 token shape 变化。

---

## 训练 FLOPs 近似

稠密 Transformer 常用：

$$
C_{train}\approx kND
$$

$k$ 常被近似为 6，用于表示 forward、backward 对权重和激活的矩阵计算。但它不是通用常数：Attention 的序列二次项、embedding、MoE 激活参数、activation recompute、稀疏/状态空间主干和 optimizer kernel 都会改变实际工作。

计算预算文档必须写明公式包含哪些算子，以及 $D$ 是原始 token、有效 loss token 还是包含 padding 的执行 token。

---

## 从 FLOPs 到时间

若有 $G$ 个设备，单设备目标精度峰值为 $P_{peak}$，端到端有效利用率为 $u$，理想训练时间近似：

$$
T\approx\frac{C_{train}}{G\times P_{peak}\times u}
$$

$u$ 应来自相近模型、shape、并行和拓扑的实测 MFU，而不是任意选取。随着设备数增加，Collective、Pipeline bubble、Checkpoint 和 Straggler 会让 $u$ 改变。

GPU-hour 为 $G\times T$；费用还应包含 CPU、存储、网络、失败重算、空闲预留和数据处理。只按 GPU 标价乘理想时间会低估总成本。

---

## Compute-optimal 不是固定配方

Scaling Law 研究 loss 随模型、数据和计算预算的经验关系。Chinchilla 类结论指出，在其模型家族、数据和训练设定下，给定计算预算时模型与训练 token 应共同扩展。它不是所有领域、数据质量、后训练或部署目标的永久比例。

规划时区分：

- 经验 scaling fit 的数据范围；
- 目标是最低预训练 loss、最小训练成本还是最低推理成本；
- 数据质量、重复和领域覆盖；
- 大模型推理成本是否限制最终部署；
- 后训练和 test-time compute 是否改变总系统预算。

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

## 参考文献

- Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models*.
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models*.
