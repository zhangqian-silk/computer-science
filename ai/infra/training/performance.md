# 训练性能与扩展效率

训练性能分析关注每个有效 token 需要多少时间、计算、通信和成本。单卡 kernel 快不代表多机训练快；每秒样本数也不能比较不同序列长度、packing 或梯度累积配置。

---

## 指标

| 指标 | 含义 | 注意事项 |
| --- | --- | --- |
| tokens/s | 单位时间处理的训练 token | 区分有效 token 与 padding |
| step time | 一次 optimizer step 的时间 | 同时记录 global batch |
| MFU | 实际模型 FLOPs 相对硬件峰值 | FLOPs 估算与峰值口径需一致 |
| HFU | 包含重算等实际硬件工作后的利用率 | 可能高于 MFU，含义不同 |
| scaling efficiency | 多设备加速相对理想线性加速 | 固定 global 或 local workload 要说明 |
| data stall | 等待数据的时间或比例 | 需与 GPU timeline 对齐 |
| communication exposed | 未被计算隐藏的通信时间 | 不是 Collective 总时间 |

MFU 的 FLOPs 模型应说明是否包含 embedding、Attention、MoE、反向和 activation recompute。不同计算口径不能直接横向比较。

---

## Step 分解

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

## 参考资料

- Narayanan, D. et al. (2021). *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*.
- Chowdhery, A. et al. (2022). *PaLM: Scaling Language Modeling with Pathways*.
