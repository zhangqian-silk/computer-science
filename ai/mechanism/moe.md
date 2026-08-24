# MoE：用稀疏路由扩大参数容量

Mixture of Experts（MoE）把 Transformer 中的某些稠密 FFN 替换为多个专家。每个 token 只激活少量专家，因此总参数量可以大幅增加，而单 token 计算量不必按专家数量同比增长。

<MoERouterExplorer />

::: info 符号与约定
沿用[数学与符号约定](../foundations/math-notation.md)。$E$ 是专家数量，$f_e(x)$ 是专家 $e$ 的 FFN，$p_e(x)$ 是 Router 概率，$\mathcal{T}(x)$ 是 Top-$k$ 专家集合；$N$ 是 batch 内 token 数，$C$ 是单专家容量，$c_{\text{factor}}$ 是容量因子；$\rho_e$ 是实际分配比例，$P_e$ 是平均路由概率。
:::

~~~mermaid
flowchart LR
	X["token 状态 x"] --> R["Router"]
	R -->|Top-k 权重| E1["Expert 1"]
	R -->|Top-k 权重| E2["Expert 2"]
	R -.未选中.-> E3["Expert 3"]
	E1 --> S["加权合并"]
	E2 --> S
	S --> Y["输出 y"]
~~~

---

## 路由与专家计算

设有 $E$ 个专家，每个专家是独立 FFN：

$$
f_e(x)=W_{2,e}\,\phi(W_{1,e}x)
$$

Router 产生 logits 并转成概率：

$$
p(x)=\operatorname{softmax}(W_rx)
$$

取概率最大的 $k$ 个专家集合 $\mathcal{T}(x)$，输出为：

$$
y=
\sum_{e\in\mathcal{T}(x)}
\tilde{p}_e(x)f_e(x)
$$

$\tilde{p}$ 可以是选中专家上的重新归一化权重。Top-1 路由计算更少，Top-2 提供更多组合与冗余；$k$ 是质量、计算和通信的直接折中。

例如 Router 对 4 个专家给出：

$$
p(x)=[0.1,0.6,0.2,0.1]
$$

Top-2 选择专家 2 和 3。若在选中集合上重新归一化，权重为：

$$
\tilde{p}_2=\frac{0.6}{0.8}=0.75,\qquad
\tilde{p}_3=\frac{0.2}{0.8}=0.25
$$

输出为 $y=0.75f_2(x)+0.25f_3(x)$。未选中的专家不执行这个 token 的 FFN，也不会从该 token 的主任务损失获得直接专家梯度。

MoE 通常只替换部分 FFN，Attention、残差和归一化仍保持稠密。它是容量扩展机制，不是一套独立语言建模目标。

---

## 容量约束与 token 丢弃

一个 batch 中的路由可能把大量 token 送到同一专家。设备缓冲区常按容量因子限制每个专家接收的 token 数：

$$
C=
\left\lceil
\frac{N k}{E}\cdot c_{\text{factor}}
\right\rceil
$$

$N$ 是本批 token 数。容量太小会溢出；具体实现可能丢弃 token、走残差旁路或重新路由。容量太大则预留空缓冲，降低硬件利用率。

若 $N=1024$、$k=2$、$E=8$、容量因子为 $1.25$：

$$
C=\left\lceil\frac{1024\times2}{8}\times1.25\right\rceil=320
$$

8 个专家共预留 2560 个 token 槽位，而实际路由分配数为 2048，额外空间用于吸收不均衡。如果某个专家收到 600 个分配，它仍会溢出 280 个；增加总容量只能缓解波动，不能替代负载均衡。

因此，「每个 token 只计算 $k$ 个专家」并不完整。实际吞吐还取决于负载均衡、padding、路由排序和设备间通信。

---

## 为什么需要负载均衡

Router 若只优化主任务损失，容易让少数专家持续获胜，形成专家塌缩。常见辅助目标鼓励路由概率和实际分配在专家间更均匀，例如让每个专家的 token 比例 $\rho_e$ 与平均路由概率 $P_e$ 的乘积不过度集中：

$$
\mathcal{L}_{\text{balance}}
\propto
E\sum_{e=1}^{E}\rho_eP_e
$$

均衡不是要求每个 batch 完全平均。数据本身可能具有专业化结构；辅助损失过强会阻止有意义的专家分工。还可加入 router z-loss 约束 logits 尺度，减少数值不稳定。

理想均匀情况下 $\rho_e=P_e=1/E$，上式未经额外系数缩放的量为：

$$
E\sum_{e=1}^{E}\frac{1}{E}\frac{1}{E}=1
$$

若实际 token 全部集中到少数专家，$\rho_eP_e$ 会增大。Router 同时接收主任务梯度与辅助梯度：前者鼓励选择能降低任务损失的专家，后者抑制长期拥塞。两者权重需要通过质量、溢出和吞吐共同验证。

---

## 分布式执行

专家通常分布在多个设备上。一次 MoE 层包含：

1. 本地计算路由；
2. 按专家重新排列 token；
3. All-to-All 把 token 发送到专家所在设备；
4. 专家 FFN 计算；
5. All-to-All 把结果送回原 token；
6. 按路由权重合并。

通信可能超过专家矩阵乘法成为瓶颈。专家并行需要与数据并行、张量并行和流水线并行共同设计；单看 FLOPs 无法判断训练速度。

推理时虽然只激活少数专家，所有专家权重仍需放在设备内存或通过分层加载获得。MoE 降低的是每 token 激活计算，不自动降低总参数存储和服务复杂度。

训练时可在较大 token batch 上摊薄 All-to-All；自回归 Decode 每步 token 数较少，通信和专家负载更容易不均衡。服务端还可能因不同请求路由到不同专家而出现尾延迟，动态 batching 需要同时考虑序列阶段和专家目的地。

如果专家没有常驻设备而是按需加载，权重传输延迟可能远高于单次 FFN 计算。这属于另一种存储—延迟折中，不能从稀疏激活公式推出必然收益。

---

## 专家是否真的学会分工

专家可能按语言、领域、词性或频率形成偏好，也可能只是为了优化负载而出现不稳定分配。判断专业化不能只看少量 token 示例，应检查：

- 路由在不同 batch 和训练阶段是否稳定；
- 专家使用率与溢出率；
- 屏蔽某个专家对哪些样本造成影响；
- 路由偏好是否来自数据捷径；
- 训练和服务中的通信与尾延迟。

MoE 的收益应在相近训练计算或推理预算下比较稠密模型，而不是只比较总参数量。

---

## 参考文献

- Shazeer, N. et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*.
- Fedus, W., Zoph, B., and Shazeer, N. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*.
- Lepikhin, D. et al. (2021). *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*.
