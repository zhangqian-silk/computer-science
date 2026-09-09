# 训练性能与扩展效率

训练效率应围绕一次有效更新计量：相同目标、相同 token 与相同质量下用了多少资源。增加设备、减短样本或跳过部分 loss 都可能提高表面速度，却不一定更快完成原来的训练任务。

---

## 指标

| 指标 | 含义 | 注意事项 |
| --- | --- | --- |
| tokens/s | 单位时间处理的训练 token | 区分有效 token 与 padding |
| step time | 一次 optimizer step 的时间 | 同时记录 global batch |
| MFU | 每秒有效模型 FLOPs 相对设备总峰值 FLOP/s | 不计额外重算 |
| HFU | 包含重算等实际硬件工作后的利用率 | 可能高于 MFU，含义不同 |
| scaling efficiency | 多设备加速相对理想线性加速 | 固定 global 或 local workload 要说明 |
| data stall | 等待数据的时间或比例 | 需与 GPU timeline 对齐 |
| communication exposed | 未被计算隐藏的通信时间 | 不是 Collective 总时间 |

MFU 的模型工作量应说明是否包含 embedding、Attention、MoE 与反向；额外重算计入 HFU 而非有效模型工作。时间范围、精度与稠密/稀疏峰值也须一致。

---

## Step 分解

强扩展固定总工作量，若单设备时间为 $t_1$，$P$ 设备时间为 $t_P$，效率为 $t_1/(Pt_P)$。弱扩展固定每设备工作量，不能再用同一个总任务时间解释结果。报告效率时应写明固定的是样本数、token 数还是 optimizer step。

MFU 通常用「每秒有效模型 FLOPs / 所有设备峰值」估计；重算增加硬件活动，却不增加有效模型工作，所以 HFU 可能上升而 MFU 降低。用更多重算解决 OOM 是合理取舍，但不能把更高 HFU 宣传成更高训练效率。

假设四 rank 中三者各用 100 ms，另一者用 180 ms，同步 step 不会是均值 120 ms，而受慢 rank 限制。需要把最慢者拆为数据、计算和通信，再看慢的是固定设备还是随样本漂移，才能区分硬件与负载问题。

一次 step 可分为：data、forward、backward、gradient communication、optimizer、checkpoint/logging。Pipeline 还要观察 bubble，MoE 要观察 token imbalance，FSDP 要观察参数 AllGather 与 prefetch。

性能排查顺序：

1. 验证 loss、global batch 和有效 token 语义一致；
2. 记录稳态 step 分段；
3. 检查最慢 rank，而不只看 rank 0；
4. 定位通信是否暴露在关键路径；
5. 分析关键 kernel 和数据等待；
6. 改一个变量并重复；
7. 扩到更多节点，观察拐点和 Straggler。

---

## Straggler

同步训练的 step 由最慢 rank 决定。Straggler 可能来自坏 GPU、网络重传、NUMA、数据倾斜、MoE 路由不均、后台任务或温度降频。平均资源利用率会掩盖单 rank 异常，应保存 rank 级时间分布和拓扑信息。

自动隔离或重调度会增加集群状态与恢复成本。先用 profile、监控和故障注入证明频率与影响，再决定是否需要自动恢复；否则以快速失败、保留诊断和作业级重启为默认。

---

## CPU 路线

CPU 可以验证指标计算、step 分段、数据 stall、进程间不均衡和 strong/weak scaling 方法。使用多进程与固定小模型即可构造 Straggler。绝对 MFU 和 GPU kernel 结论必须在目标 GPU 上测量。

---

## 参考资料

- Narayanan, D. et al. (2021). [*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*](https://arxiv.org/abs/2104.04473).
- Chowdhery, A. et al. (2022). [*PaLM: Scaling Language Modeling with Pathways*](https://arxiv.org/abs/2204.02311).
