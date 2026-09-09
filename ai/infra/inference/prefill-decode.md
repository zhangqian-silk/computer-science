# Prefill 与 Decode

Prefill 与 Decode 是同一个自回归模型的两种执行形状，而不是两个模型。前者读取整个已有前缀并建立状态，后者把刚生成的 token 再送入模型继续推进。理解首 token 从哪里产生，才能正确统计轮数、KV 长度和时间。

<PrefillDecodeExplorer />

---

## 阶段定义

Prefill 一次处理提示中的多个 token，为每层写入历史 KV，并用最后有效位置的 logits 产生第一个输出 token。Decode 每轮只处理每条活跃序列的新 token，读取全部历史 KV，再追加新 KV。

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 当前输入 token | 每请求多个 | 每序列通常一个 |
| 矩阵形状 | 较大的 GEMM | 小 M 的 GEMM/GEMV |
| Attention | prompt 内多 query | 少量 query 读取长 KV |
| 常见瓶颈 | 计算、Attention 中间量 | 权重/KV 带宽、launch、同步 |
| 用户指标 | TTFT | TPOT/ITL、输出吞吐 |

这只是典型趋势。大 batch Decode 可能提高算术强度，极长 Prefill 也可能主要受 Attention 内存与计算限制，必须实测。

---

## 时间与工作量

设 prompt 长度为 $p$，要求输出 $o\ge1$ 个 token，不计额外验证。Prefill 的末位置 logits 产生第一个输出，之后只需 $o-1$ 轮 Decode。第 $r$ 轮 Decode 处理第 $r$ 个输出，读取到长度 $p+r$ 的 KV，并产生下一个输出。

因此单头、单请求的因果可见连接数为：

$$
C_{\text{prefill}}=\frac{p(p+1)}{2},\qquad
C_{\text{decode}}=\sum_{r=1}^{o-1}(p+r)
=(o-1)p+\frac{o(o-1)}{2}
$$

最后一个输出若不再进入模型，就没有对应的新 KV；终止前缓存长度为 $p+o-1$。分配器可能提前预留更多槽，因此逻辑已写长度和物理容量仍是两回事。

交互组件按这一口径计算。稠密 kernel 可能计算部分被 mask 的块，FlashAttention 也有块边界工作，所以上式不是实际 FLOPs 或字节计数。要预测耗时，还需加线性投影、FFN、采样与调度。

一条请求的首 token 时间近似包含：

$$
TTFT=T_{queue}+T_{tokenize}+T_{prefill}+T_{sample}+T_{stream}
$$

后续输出时间受每轮 Decode、调度和排队影响。若生成 $N$ 个输出 token，不能简单用一次单请求 TPOT 乘 $N$ 推断高并发 E2E，因为 active batch 会随请求完成不断变化。

Prefill 的线性层 FLOPs 随输入 token 数近似线性增长，标准 Attention 连接项随序列长度呈二次增长；Decode 每步只新增一个 query，但要读取随历史长度增长的 KV。

---

## Chunked Prefill

后一个 chunk 必须能读取前面 chunk 已写的 KV，因果位置不能从零重新编号。拆分的是执行时间片，不是将一条长输入变成互不相关的短序列。Sarathi-Serve 等系统研究这种分块与解码调度的配合；结论需结合其 workload，而不是一个固定 chunk size 适合所有模型。

把长 Prefill 拆成多个 chunk，可以控制单 iteration token 数，让 Decode 请求在 chunk 之间获得执行机会，从而降低被长 prompt 阻塞的尾延迟。代价包括更多调度轮次、边界元数据和可能的 kernel 效率下降。

chunk 大小应由目标 TTFT/TPOT、kernel 形状、KV 容量和请求长度分布共同决定。仅在合成长 prompt benchmark 上选出的固定值不一定适合线上短长混合流量。

---

## 阶段分离的动机

Prefill 更偏计算密集，Decode 更偏内存带宽与延迟。把两者放到不同 worker 可以分别调优和扩缩容，但需要传输 KV 或共享存储、进行跨池路由，并处理 worker 失败和版本一致性。只有阶段资源冲突或独立扩展带来的收益超过传输与运维成本时，PD 分离才成立。

---

## CPU 路线

CPU 小模型可以真实执行 Prefill/Decode，验证输出与 KV cache 一致性，并观察提示长度和 batch 对耗时的趋势。也可用矩阵与 sleep 模型模拟两阶段调度。CPU 的 cache/带宽比例不同，不能据此推断 GPU 的具体转折点。

---

## 参考资料

- Pope, R. et al. (2022). [*Efficiently Scaling Transformer Inference*](https://arxiv.org/abs/2211.05102).
- Agrawal, A. et al. (2024). [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://arxiv.org/abs/2403.02310).
