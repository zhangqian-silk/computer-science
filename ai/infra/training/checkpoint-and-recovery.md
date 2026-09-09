# Checkpoint、状态一致性与恢复

Checkpoint 的含义由恢复承诺决定：是只恢复预测函数、继续近似相同训练，还是精确续接样本与随机过程。只有权重的文件足以部署某些模型，却不足以重现 Adam 下一步更新；不同承诺需要不同状态，不应一概要求最复杂的恢复协议。

---

## 状态清单

| 状态 | 继续训练 | 仅部署推理 |
| --- | --- | --- |
| 模型参数 | 必需 | 必需 |
| Optimizer state | 必需 | 不需要 |
| LR scheduler / global step | 必需 | 不需要 |
| Loss scale、训练 amax 历史 | 精确续训按配方保存 | 通常不直接用于推理 |
| 推理量化制品的 scale/zero point | 按制品格式 | 量化格式需要时必须保留 |
| RNG state | 精确重现时需要 | 通常不需要 |
| Data sampler/packing state | 按恢复语义 | 不需要 |
| 并行与分片元数据 | 分片加载需要 | 转换后可不同 |
| Tokenizer/config/template | 需要追踪 | 必需 |

推理制品应从训练 Checkpoint 转换和验证，而不是让在线服务直接依赖训练进程的内部布局。

---

## 一致性边界

假设有两个 shard，rank 0 保存 step 100，rank 1 保存 step 101。即使二者形状正确、都能加载，组合后也不是任一训练时刻的模型。异步保存必须冻结该逻辑 step 的内容或创建受控副本，不能让后台线程读取不断更新的原参数。

因此「写入成功」与「快照发布完整」是两个事件。只有所有必要 shard 和元数据完成后，才能将版本标成可恢复；损坏或缺 shard 时应失败，而不是默默初始化缺失参数继续训练。

若每隔 $\Delta$ 时间保存一次，故障时刻近似均匀，丢失的未保存工作平均约为 $\Delta/2$；保存本身又消耗时间与带宽。这个粗略模型用于解释频率折中，不考虑相关故障、异步复制与恢复耗时，不能直接当作最优间隔公式。

所有 shard 必须属于同一个逻辑 step。若部分 rank 写入成功、部分失败，不能把目录存在当作完整 Checkpoint。可使用临时命名加完成 manifest，或由协调者在确认所有 shard 后发布可见标记。

这会引入持久状态和清理成本，因此需要明确：

- 保护的是多大训练损失与恢复时间目标；
- 保存频率、写入时间和存储成本；
- 不完整文件如何识别与清理；
- manifest 与 shard 的兼容版本；
- 加载失败时是回退上一快照还是终止。

在没有自动恢复承诺时，简单而清晰的语义可以是「只加载最后一个已验证 manifest，失败则人工选择更早版本」。

---

## 分片与重分片

例如全局矩阵的行区间被两 rank 保存，恢复到四 rank 时应按全局 shape 与 shard offset 重新取交集，而不是按旧文件名一一分配给新 rank。恢复测试因此应检验逻辑参数值，而非仅检验文件数相同。

分布式 Checkpoint 可以按保存时 rank 布局存储，也可以保存可重分片的逻辑 Tensor 元数据。后者便于用不同 world size 恢复，但要求框架理解全局 shape、placement 和 shard offset。

Checkpoint 格式升级应保留版本与迁移策略。参数重命名、融合 QKV、不同 TP 切分和量化转换都可能改变制品结构。

---

## 恢复验证

至少执行：

1. 保存前后比较 global step 与参数摘要；
2. 新进程加载并执行一个固定 batch；
3. 检查 loss、输出和 optimizer state；
4. 验证缺 shard、损坏文件和版本不匹配能够快速失败；
5. 若承诺确定性恢复，对比后续若干 step 的数值或统计容差；
6. 记录恢复耗时与所需存储带宽。

---

## CPU 路线

Checkpoint 语义完全可以用多进程 CPU 小模型验证，包括中断、缺文件、跨 world size 加载与数据游标。GPU 环境再补充大规模吞吐、设备内存 staging 和真实分片成本。

---

## 参考资料

- PyTorch. [*Distributed Checkpoint Documentation*](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html).
- NVIDIA Megatron Core. *Distributed Checkpointing*.
