# 分布式推理

多设备可以容纳一个更大的模型，也可以复制模型服务更多请求；这是不同目标。单请求需要跨设备通信时，计算减少与通信增加同时发生；增加独立副本时，单请求计算不变，但排队可能减少。选择并行方式应先区分这两类收益。

---

## 并行方式

| 方式 | 请求如何分布 | 主要通信 | 适用目标 |
| --- | --- | --- | --- |
| Replica/Data Parallel | 每个副本处理不同请求 | 通常无逐层通信 | 提高吞吐与隔离 |
| Tensor Parallel | 一个请求跨设备执行每层 | 高频 Collective | 单设备放不下或需要更大计算能力 |
| Pipeline Parallel | 一个请求依次经过多个 stage | stage 间激活传输 | 按层容纳模型 |
| Expert Parallel | token 发往专家所在设备 | AllToAll | MoE 专家容量 |
| Context/Sequence Parallel | 长上下文分片 | Attention 相关通信 | 超长上下文容量 |

Replica 最容易扩吞吐，但每个副本都要容纳权重。TP 减少每卡权重，却让每个 decode step 进行多次通信。PP 对单请求增加 stage 路径，需依靠 microbatch 或多请求填充流水线。

---

## TP 的延迟边界

简化一轮 Decode：若模型有 $L$ 层，每层存在 $c$ 次不可隐藏通信，单次暴露延迟为 $\alpha$，仅启动延迟就约为 $Lc\alpha$，尚未计字节传输。把矩阵切到更多设备可能减少每卡 FLOPs，却不一定减少这些串行边界。

例如原有 8 张设备，可以组成一个 TP=8 group，也可以组成两个 TP=4 group。前者对单个请求使用更多设备，后者可并发处理更多独立请求。比较应固定总设备预算，并同时测单请求延迟、排队、KV 容量和质量，不只选一组最有利的数字。

层内切分常在 Attention output 和 FFN output 附近执行 AllReduce 或 ReduceScatter。Decode 每 token 都重复，因此通信延迟与同步抖动会直接累积到 TPOT。

高 TP degree 不是免费的容量手段。选择时需比较：

- 单卡权重和 KV 是否可容纳；
- 节点内/节点间拓扑；
- Prefill 与 Decode shape；
- kernel 变小后的效率；
- Collective 暴露时间；
- 副本数量和总体排队。

---

## 路由与 KV 局部性

粘性应绑定「持有这一序列状态的 worker group」，不是某个偶然接收 HTTP 的进程。入口可以重新连接，但继续生成必须找到同一状态或按明确协议重建。多副本负载均衡因此与无状态 Web 请求有实质区别。

有状态生成请求必须在后续 iteration 找到其 KV 所在 worker group。入口路由到 replica 后，内部通常保持粘性。Prefix cache 或 session cache 进一步增加局部性；KV-aware routing 可能降低重复 Prefill，却需维护缓存目录和失效边界。

没有强缓存命中收益证据时，优先采用简单的 least-loaded/queue-aware 路由与请求级粘性。跨副本迁移 KV 会增加协议、带宽和部分失败处理。

---

## 多机失败语义

一个 TP/PP group 中任一 rank 失败，当前 iteration 通常无法安全继续。若尚未输出，可由上层决定是否重试整个请求；若已经流式输出，透明重试可能重复或分叉。默认应向客户端暴露失败并释放整个 group 的请求状态。

自动重建 group 和迁移请求只有在明确 SLO、故障率和恢复协议下才值得引入。

---

## CPU 路线

多进程 CPU 可验证 TP linear、PP stage、replica router、EP token dispatch 和 group failure。可使用 Gloo/MPI 和小模型。CPU 不能代表 NCCL/NVLink 的逐 token延迟，但能发现 shape、顺序、粘性和状态释放错误。

---

## 参考资料

- vLLM. [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/).
- NVIDIA TensorRT-LLM. *Multi-GPU and Multi-Node Support*.
