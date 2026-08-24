# Continuous Batching 与请求调度

静态 batching 等待一组请求组成 batch，并让整个 batch 运行到结束；短请求完成后，其槽位可能空闲。Continuous Batching 在每个 iteration 边界重新选择活跃序列，让新请求进入并移除已完成请求。

<BatchSchedulerTimeline />

---

## 调度对象

一次调度不是简单选「多少请求」，而是在预算内选择 token 工作：

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

---

## 吞吐与延迟权衡

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

## 参考文献

- Yu, G.-I. et al. (2022). *Orca: A Distributed Serving System for Transformer-Based Generative Models*.
- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*.
