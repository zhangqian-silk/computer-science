# 机制：可复用的计算部件

机制分成信息读取、位置关系、稳定训练、容量分配与参数适配五条轴。它们可以组合但不互相替代：GQA 减少 KV 头，稀疏 Attention 减少连接，MoE 选择专家，LoRA 约束更新。沿下面的主页面学习完整推导，再回到模型页观察组合。

---

## 信息交互

- [Attention](./attention.md)：query 如何从一组 key-value 中按相关性读取信息；
- [Self-Attention](./self-attention.md)：同一序列如何建立全局交互，以及 mask、KV cache 带来什么约束；
- [MHA、MQA、GQA 与 MLA](./attention-head-sharing.md)：Query 头如何共享或压缩 KV 表示；
- [稀疏注意力](./sparse-attention.md)：怎样用受限连接图降低长序列代价。

---

## 位置与长度

- [位置表示](./positional-encoding.md)：绝对位置、相对偏置与 RoPE 的统一比较；
- [RoPE](./rope.md)：旋转位置编码的二维几何和相对位移性质；
- [长上下文](./long-context.md)：插值、频率缩放与 ALiBi 如何处理训练长度之外的位置。

---

## 容量与适配

读完任一机制，应能画出输入输出、说明参数是否训练、指出影响哪个成本项，再给一个不成立的边界。例如容量减少不等于加速，位置可计算不等于可外推，少量训练参数不等于不需要基座计算。

- [归一化、残差与 Dropout](./normalization-and-residual.md)：深层 Block 如何保持尺度与梯度路径；
- [前馈网络与门控激活](./feed-forward-and-gating.md)：逐 token 通道变换如何贡献参数与计算；
- [MoE](./moe.md)：稀疏路由怎样在近似固定计算量下扩大参数容量；
- [LoRA](./lora.md)：低秩增量怎样降低任务适配所需的可训练参数。

[模型地图](../model/index.md)进一步组合这些部件，形成具有完整输入输出和训练目标的架构。[状态空间模型](../model/state-space-model.md)则以独立的状态递推与序列扫描构成完整模型主干。

阅读机制页时应区分三个层次：数学规则定义了输出怎样计算，训练目标决定规则中的参数怎样获得梯度，部署实现决定同一规则是否真正节省时间或内存。只看复杂度公式，无法判断机制在目标硬件和任务上的实际收益。
