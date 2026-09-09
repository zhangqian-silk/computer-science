# KV Cache 管理

KV cache 用保存中间状态换取不重算前缀。它只适用于历史表示在未来输入到来后仍保持不变的执行语义：因果模型符合这一点，普通双向 Encoder 则会因新输入改变旧位置表示。机制前置见 [Self-Attention](../../mechanism/self-attention.md)，本页研究状态如何存放与回收。

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

该式适用于 K/V 同宽的 MHA/MQA/GQA，不适用于 MLA 的单一联合潜缓存；MLA 应按实际潜向量与位置分支计数。

每新增一个 token 的 KV 字节为 $2LN_{kv}Db$。量化后还要加 scale，Paged KV 还要加 block table 与 allocator 元数据，多并行 rank 则按实际切分或副本重新计算。

引擎容量不是用总显存除以单请求最大 KV。还要扣除权重、workspace、CUDA Graph buffer、通信 buffer 和安全余量，并考虑请求长度分布与碎片。

---

## 从连续区到 Paged KV

例如块大小为 16，位置从 0 开始，逻辑位置 35 位于第 2 个逻辑块、块内偏移 3。若块表为 `[7, 2, 9]`，实际读取物理块 9 的偏移 3，而不是地址连续的第 35 个槽。页表改变寻址，不改变注意力数学结果。

长度 32 恰好用两块，增加一个 token 到 33 就需要三块，末块只有一个有效位置。交互图应观察这种阶梯变化，而不是把分配曲线当成连续线性。

Kwon 等人的 PagedAttention 将分页式 KV 管理与注意力读取结合，目的在于控制变长请求的浪费与共享成本；它不是把历史 token 压缩成更少语义向量。GQA/MLA 改变每 token 状态大小，分页改变这些状态的物理安排，可以组合使用。

为每个请求预留最大连续空间会产生内部浪费，并要求大连续区。Paged KV 把逻辑 token 位置映射到固定 token 容量的物理 block：

$$
\text{physical block}=\text{block table}[\lfloor position/B_s\rfloor]
$$

$B_s$ 是每 block 的 token 数。末 block 仍有内部碎片，但其他未使用 block 可分配给别的请求。Attention kernel 通过 block table 读取非连续物理页。

block 越小，内部碎片少但元数据和寻址多；block 越大，寻址简单但短请求浪费更多。

---

## 分配器不变量

下列伪代码是假设分配器串行执行的纯活跃序列模型，检查容量和取块须原子化。真实系统还要计入缓存保留、共享持有者与在途 GPU 读取；不能只因请求离开队列就立即复用块。

共享前缀若是只读整块，可以让多个序列引用同一物理块；一旦要修改共享的尾块，就必须复制或禁止这种共享。只对 refcount 加一而不处理可变写入，会让一个请求的后续 token 改变另一个请求看到的历史。

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

---

## 参考文献

- Kwon, W. et al. (2023). [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180).
- vLLM. *KV Cache Manager Documentation*.
