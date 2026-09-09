# 训练循环与内存生命周期

训练循环的核心是把一次 loss 变成一次受控参数更新。显存取决于哪些状态在同一时刻存活，而不只是模型有多少参数；梯度累积、重算和状态分片分别改变不同的生命周期。前置计算见[神经网络训练基础](../../foundations/neural-network-basics.md)。

<TrainingMemoryEstimator />

---

## 一次优化步

设单 rank 每个 microbatch 有 $b$ 个样本，累积 $a$ 次，数据并行度为 $P$，且样本不重叠，则一次更新的 global batch 为 $abP$。若各 microbatch 大小相等，累积前将每个平均 loss 除以 $a$，即可得到同一大 batch 的平均梯度；变长文本还需按有效 loss token 数加权。

累积不等于每次反向都更新参数。若每个 microbatch 都调用 optimizer，后面的梯度是在不同权重上计算，已不再是同一个大 batch 的梯度。Dropout、BatchNorm 或其他随机/批统计操作也可能使两条路径不能逐位等价。

参数、梯度、优化器一阶矩与二阶矩常驻或跨步存在，激活则主要服务本步反向。以 7B 参数为例，两字节参数主体是 14 GB，约 13.04 GiB；即使参数放得下，梯度、矩状态与临时工作区仍可能导致 OOM。交互实验的数值因此不能当作峰值承诺。

假设使用梯度累积，一次 optimizer step 可以包含多个 microbatch：

1. 读取并准备 microbatch；
2. 前向计算并保存反向所需状态；
3. 计算 loss；
4. 反向传播，把梯度累积到参数；
5. 达到累积步数后执行梯度通信、裁剪和 optimizer update；
6. 更新学习率、step、随机数与日志状态；
7. 按策略保存 Checkpoint。

梯度累积增加有效 global batch，却不会消除每个 microbatch 的激活峰值。若框架在 `no_sync` 类阶段延迟梯度通信，最后一个 microbatch 的时序也会改变。

---

## 状态分类

| 状态 | 规模主项 | 生命周期 | 可否分片 |
| --- | --- | --- | --- |
| 参数 | $O(p)$ | 整个训练 | 可以 |
| 梯度 | $O(p)$ | backward 到 update | 可以 |
| Optimizer state | $O(p)$ | 整个训练 | 可以 |
| 激活 | 与 batch、序列、层和实现有关 | forward 到对应 backward | 可重算或切分 |
| 临时 workspace | 与 kernel/算法有关 | 单算子或单阶段 | 通常本地 |
| 通信 buffer | 与 bucket/分片有关 | Collective 前后 | 随策略变化 |

逐位置激活主项常随 $BTd_{\text{model}}L$ 增长；若显式保存注意力矩阵，还存在 $Bn_{\text{head}}T^2L$ 项。FlashAttention 可避免保存这一二次矩阵，不只是改变线性项系数；重算与分片也改变实际峰值。

---

## Activation Checkpointing

若 $L$ 层各保存大小近似相同的激活，完全保存占 $O(L)$ 份。选择边界后只保存边界输入，在反向进入区间时重算内部状态；具体节省量由区间长度和算子保存需求决定，不是每种网络都固定省一半。Chen 等人的 2016 年工作研究了这种计算换内存的调度；持久快照恢复是另一个问题。

Activation Checkpointing 只保留选定边界的激活，在 backward 时重算区间内 forward。它以额外计算换内存：

- 粒度过细会产生更多调度与重算开销；
- 粒度过粗仍可能保留大中间量；
- Dropout 等随机算子需要保存或重建 RNG 状态；
- 有副作用或非确定性通信的区域不能随意重算。

该机制与持久 Checkpoint 无关：前者在一次训练 step 内重算激活，后者用于进程退出后的作业恢复。

---

## 峰值定位

分析内存时应画出时间轴，而不是只比较稳态占用：

- forward 末尾通常保存了最多激活；
- backward 逐层释放激活并产生梯度；
- optimizer update 可能创建临时转换或融合 buffer；
- checkpoint save 可能产生 staging 与序列化副本；
- 第一次迭代还包含 lazy initialization 和 autotune。

allocated、reserved 与设备总占用口径不同。OOM 表示所需分配无法满足，原因可包括总容量、碎片、临时峰值或分配器限制；不应一概解释为物理连续块不足。

---

## CPU 路线

PyTorch CPU 可以完整验证训练循环、自动微分、梯度累积、activation checkpointing 和状态生命周期。可用进程 RSS、allocator 或 profiler 观察内存。CPU 不具有 GPU HBM 和 CUDA allocator 行为，但适合作为正确性基线。

---

## 参考资料

- PyTorch. [*Autograd Mechanics*](https://docs.pytorch.org/docs/stable/notes/autograd.html).
- Chen, T. et al. (2016). [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174).
