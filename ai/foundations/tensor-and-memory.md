# Tensor、形状与内存

AI Infra 的许多问题表面上属于模型、GPU 或分布式系统，最终都要落到同一个接口：一个具有形状、数据类型、步长和设备位置的 Tensor。只写出矩阵公式不足以判断它如何占用内存、能否连续访问，以及怎样跨设备传输。

::: info 符号与约定
$B,S,H,L,V$ 分别表示 batch、序列长度、隐藏维度、层数和词表大小；$p$ 是参数量，$b$ 是单个元素的字节数。
:::

---

## Tensor 的四个基本属性

设隐藏状态 $X$ 的逻辑形状为 $[B,S,H]$。运行时还需要知道：

| 属性 | 回答的问题 | 常见错误 |
| --- | --- | --- |
| shape | 每一维有多少元素 | 把逻辑维度顺序与物理布局混为一谈 |
| dtype | 每个元素如何编码 | 只按名称估算精度，不看累加类型 |
| stride | 相邻逻辑元素跨过多少存储单元 | transpose 后误以为数据仍连续 |
| device | 存储位于 CPU、GPU 还是其他设备 | 忽略隐式复制和同步 |

连续的 row-major $[B,S,H]$ 张量通常有 stride $[S H,H,1]$。转置最后两维可以只修改元数据而不搬运数据，此时新视图的 stride 改变；若后续 kernel 要求连续输入，运行时可能额外执行一次复制。

元素数量与逻辑存储量为：

$$
N=B\times S\times H,\qquad M=N\times b
$$

实际分配还可能包含对齐、allocator block、workspace 和碎片，不能把公式值直接当作进程峰值显存。

---

## Transformer 中常见形状

| 对象 | 典型形状 | 主要用途 |
| --- | --- | --- |
| token ID | $[B,S]$ | 查 embedding |
| hidden state | $[B,S,H]$ | 层间接口 |
| attention query | $[B,N_q,S,D]$ | 发起读取 |
| attention key/value | $[B,N_{kv},S,D]$ | 提供索引和内容 |
| logits | $[B,S,V]$ | 词表预测 |
| FFN 中间激活 | $[B,S,H_{ff}]$ | 通道扩展与门控 |

若 $H=N_qD$，MHA 通常有 $N_{kv}=N_q$；GQA 和 MQA 让 $N_{kv}<N_q$。这一差异既改变权重形状，也改变 KV cache 的容量与 Decode 带宽。

形状推导应从语义维度开始，再决定 flatten、transpose 或 shard。直接从一串整数猜语义，很容易在多头拆分和并行切分后失去方向。

---

## 模型状态与运行时状态

推理至少包含以下内存：

- 参数权重；
- 临时激活与算子 workspace；
- KV cache；
- 通信 buffer；
- allocator 保留但当前未使用的 block；
- 框架、CUDA context 与图捕获状态。

训练还需要保存反向传播所需激活、梯度和优化器状态。以不考虑分片的 AdamW 为例，每个参数可能同时对应参数、梯度、一阶矩和二阶矩；若使用混合精度，还可能保留 FP32 master weight。因实现与 dtype 不同，不能把「每参数固定若干字节」当作通用常数。

---

## View、复制与别名

reshape 只有在新形状与原 stride 兼容时才能成为零复制 view。contiguous、跨设备移动、dtype 转换通常会产生新存储。两个 Tensor 还可能共享同一底层 storage：修改一个 view 会影响另一个 view。

阅读性能 profile 时，应主动查找：

- 不必要的 contiguous 或 clone；
- CPU 与 GPU 之间的隐式同步；
- 小 Tensor 的大量分配；
- 生命周期过长导致的峰值；
- 通信前后的 pack/unpack。

---

## CPU 与 GPU 的共同点和差异

CPU 和 GPU 都受缓存局部性、向量化、内存带宽和线程调度影响，因此 shape、stride、blocking 与 NUMA 在 CPU 上同样重要。差异在于 GPU 拥有更大规模的并行线程、显式 Shared Memory、独立设备内存和更高的 kernel launch 成本。

CPU 可以验证 Tensor 语义、布局、allocator、量化格式和分布式切分，但不能代表 GPU 上的 Warp 分歧、Tensor Core、HBM 带宽和 CUDA Graph 行为。

---

## 读者检查

给定一个 Tensor，应能回答：它的逻辑维度分别代表什么、是否连续、一次算子会读写多少字节、是否产生新存储、在哪个设备上，以及它的生命周期何时结束。后续的[性能模型](../infra/accelerator/performance-model.md)会把这些字节数与 FLOPs 放在同一张图中分析。

## 参考资料

- PyTorch. *Tensor Views*.
- NumPy. *Internal organization of NumPy arrays*.
