# KV Cache 管理

KV cache 保存每层历史 token 的 Key/Value，使 Decode 不必重算整个前缀。推理引擎的关键问题不是「有没有 KV cache」，而是如何在变长、取消、共享和高并发下分配固定容量，并保证 block 的所有权正确。

<PagedKVSimulator />

::: info 符号与约定
$L,S,N_{kv},D,b$ 分别表示层数、序列长度、KV 头数、head 维度与单元素字节数。
:::

---

## 容量公式

忽略对齐与元数据，单序列 KV 主体为：

$$
M_{KV}=2LSN_{kv}Db
$$

每新增一个 token 的 KV 字节为 $2LN_{kv}Db$。量化后还要加 scale，Paged KV 还要加 block table 与 allocator 元数据，多并行 rank 则按实际切分或副本重新计算。

引擎容量不是用总显存除以单请求最大 KV。还要扣除权重、workspace、CUDA Graph buffer、通信 buffer 和安全余量，并考虑请求长度分布与碎片。

---

## 从连续区到 Paged KV

为每个请求预留最大连续空间会产生内部浪费，并要求大连续区。Paged KV 把逻辑 token 位置映射到固定 token 容量的物理 block：

$$
\text{physical block}=\text{block table}[\lfloor position/B_s\rfloor]
$$

$B_s$ 是每 block 的 token 数。末 block 仍有内部碎片，但其他未使用 block 可分配给别的请求。Attention kernel 通过 block table 读取非连续物理页。

block 越小，内部碎片少但元数据和寻址多；block 越大，寻址简单但短请求浪费更多。

---

## 分配器不变量

```text
ALLOCATE-BLOCKS(sequence, count)
    if FREE-BLOCK-COUNT() < count then
        return OUT-OF-CAPACITY
    end if

    blocks ← TAKE-FREE-BLOCKS(count)
    for each block in blocks do
        block.refcount ← 1
        APPEND(sequence.block_table, block.id)
    end for

    return blocks
```

必须持续成立：

- 一个 free block 不属于任何活跃序列；
- refcount 等于共享该 block 的活跃所有者数；
- refcount 降到零时恰好回收到 free list 一次；
- block table 的逻辑顺序与 token position 一致；
- 写共享前缀末 block 前执行 copy-on-write 或禁止共享可变区域；
- 请求结束、取消和失败都进入同一幂等释放路径。

---

## 抢占与交换

容量不足时可以拒绝新请求、暂停低优先级序列、释放并在恢复时重算，或把 KV 换出到 CPU/远端。换出引入带宽、状态一致性和恢复延迟；重算消耗计算；拒绝最简单但降低接纳率。

选择需要明确 SLO 与流量证据。若没有恢复协议，优先使用 admission control 和快速失败，避免在请求已输出部分 token 后静默迁移导致不可控停顿。

---

## CPU 路线

CPU 非常适合实现 KV allocator 模拟器：用普通 byte array 或 Tensor 代表 block，随机生成 allocate、append、fork、cancel 和 release 操作，并用 property test 检查引用计数与泄漏。还可以真实运行小模型缓存 K/V，验证 cached decode 与完整重算 logits 一致。

## 参考文献

- Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*.
- vLLM. *KV Cache Manager Documentation*.
