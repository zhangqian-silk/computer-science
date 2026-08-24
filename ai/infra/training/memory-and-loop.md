# 训练循环与内存生命周期

训练显存不是「参数量乘某个固定常数」。峰值由参数、梯度、优化器状态、保存的激活、临时 workspace、通信 buffer 和 allocator 行为共同决定，而且这些对象的生命周期并不相同。

<TrainingMemoryEstimator />

---

## 一次优化步

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

激活显存往往随 $B\times S\times H\times L$ 增长，但 FlashAttention、序列并行、activation checkpointing 和不同自动微分保存策略会改变系数。

---

## Activation Checkpointing

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

框架的 allocated、reserved 与设备进程占用口径不同。OOM 发生在连续 block 申请失败时，空闲总量看似足够也可能受碎片或 workspace 峰值影响。

---

## CPU 路线

PyTorch CPU 可以完整验证训练循环、自动微分、梯度累积、activation checkpointing 和状态生命周期。可用进程 RSS、allocator 或 profiler 观察内存。CPU 不具有 GPU HBM 和 CUDA allocator 行为，但适合作为正确性基线。

## 参考资料

- PyTorch. *Autograd Mechanics*.
- Chen, T. et al. (2016). *Training Deep Nets with Sublinear Memory Cost*.
