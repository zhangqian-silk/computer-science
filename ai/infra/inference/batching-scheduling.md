# Continuous Batching 与请求调度

调度器决定本轮让哪些 token 使用模型，而不只是凑齐多少 HTTP 请求。静态批次固定一组序列，连续批次在迭代边界重新准入；二者的区别是控制流与资源回收时机，不是更换模型公式。

<BatchSchedulerTimeline />

---

## 调度对象

在交互实验中，A/B/C 先占三个槽，B 提前结束。静态批次必须等 A/C 结束后才接纳 D，连续批次则可在 B 结束的下一轮接纳 D。列表示调度轮次：一次长 Prefill 与一次短 Decode 不应被误认为同样的毫秒长度。

真实约束至少有两个维度。Token budget 限制本轮算子形状，KV budget 限制存活历史。一个请求即便本轮只处理一个 token，仍可能持有数万 token 的 KV；反过来短 prompt 可以占较多本轮 token，但缓存仍很小。用单一请求数量无法表达两者。

一次调度要在预算内选择 token 工作，而非简单选「多少请求」：

- Decode 序列通常每轮需要 1 个新 token；
- Prefill 请求可能需要多个 prompt token；
- chunked prefill 只消费当前 chunk；
- speculative decoding 可能在一轮验证多个候选；
- beam/multi-sample 让一个请求拥有多个序列。

预算至少包括 scheduled tokens、KV blocks、active sequences、adapter/encoder cache 与模型支持的最大 batch shape。

---

## 基础调度算法

```text
SCHEDULE(waiting, running, token_budget, kv_budget)
    batch ← empty

    for each sequence in PRIORITIZE(running, waiting) do
        work ← NEXT-TOKEN-WORK(sequence)

        if work.tokens > token_budget then
            continue
        end if

        if REQUIRED-KV(work) > kv_budget then
            continue
        end if

        APPEND(batch, work)
        token_budget ← token_budget - work.tokens
        kv_budget ← kv_budget - REQUIRED-KV(work)
    end for

    return batch
```

真实调度还要处理不可拆分工作、优先级老化、deadline、抢占、模型/adapter affinity 和输出背压。算法必须保证等待请求不会在持续 Decode 流量下永久饥饿。

以上只演示预算选择，不构成公平性证明。单个工作若永久大于预算，应分块或显式拒绝，不能每轮 continue；等待保证还需限定可接纳负载和优先级规则。

---

## 吞吐与延迟权衡

若 Prefill 占满当前 token budget，已在流式生成的请求可能等待，ITL 变长；若永远优先 Decode，新到长 prompt 可能饥饿。需要通过 chunk 和有界等待规则平衡，而不是把某一类永远排在首位。

Orca 的重要思想是 iteration-level 调度与选择性 batching；PagedAttention 提供更灵活的缓存管理。二者解决相互配合但不同的问题：能在迭代边界准入，不代表一定有足够连续或分页 KV 容量。

更大 batch 通常提高设备利用率，却可能：

- 增加 queue time；
- 拉长单次 iteration，使所有序列更久才得到下一个 token；
- 占用更多 KV；
- 让长 Prefill 阻塞 Decode；
- 放大不同长度请求的尾部。

因此目标应定义为 SLO 内 Goodput，例如满足 TTFT 和 TPOT 阈值的每秒完成请求或 token，而不是无约束峰值吞吐。

---

## 公平、优先级与取消

FIFO 简单但无法表达 deadline 或租户等级；最短作业优先可降低平均延迟，却可能饿死长请求；deadline 调度需要可靠的成本估计。生产设计应写清：

- 排队顺序与老化规则；
- Prefill/Decode 优先级；
- 单租户最大在途 token/KV；
- 超时发生在排队还是执行阶段；
- 取消何时释放配额；
- OOM 前的 admission 行为。

---

## CPU 路线

CPU 调度模拟器可以用离散事件或 mock iteration 验证算法。输入真实或合成的到达时间、prompt/output 长度，输出队列时间、TTFT、TPOT、Goodput、公平性和 KV 占用时间线。它不依赖 GPU，非常适合比较策略；在 GPU 上只需替换成本模型并校准执行时间。

---

## 参考文献

- Yu, G.-I. et al. (2022). [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/conference/osdi22/presentation/yu).
- Kwon, W. et al. (2023). [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180).
