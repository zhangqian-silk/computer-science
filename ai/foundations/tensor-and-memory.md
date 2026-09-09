# Tensor、形状与内存

Tensor 同时有逻辑视图与物理存储。形状说明元素组织，stride 决定地址，dtype 决定每个元素的编码，device 决定访问位置；一个转置可以只改视图，但后续算子可能要求复制。本页从地址推演连接模型形状与真实内存成本。

::: info 符号与约定
$B,T,d_{\text{model}},L,|\mathcal{V}|$ 分别表示 batch、序列长度、隐藏维度、层数和词表大小；$p$ 是参数量，$b$ 是单个元素的字节数。形状中的每个数字都是元素数，stride 默认也以元素为单位。
:::

---

## Tensor 的四个基本属性

设隐藏状态 $X$ 的逻辑形状为 $[B,T,d_{\text{model}}]$。运行时还需要知道：

| 属性 | 回答的问题 | 常见错误 |
| --- | --- | --- |
| shape | 每一维有多少元素 | 把逻辑维度顺序与物理布局混为一谈 |
| dtype | 每个元素如何编码 | 只按名称估算精度，不看累加类型 |
| stride | 相邻逻辑元素跨过多少存储单元 | transpose 后误以为数据仍连续 |
| device | 存储位于 CPU、GPU 还是其他设备 | 忽略隐式复制和同步 |

连续 row-major 张量通常有 stride $[Td_{\text{model}},d_{\text{model}},1]$。转置最后两维可只改元数据，但后续 kernel 若要求连续输入，仍可能产生复制。

元素数量与逻辑存储量为：

$$
N=B T d_{\text{model}},\qquad M=Nb
$$

实际分配还可能包含对齐、allocator block、workspace 和碎片，不能把公式值直接当作进程峰值显存。

---

## Transformer 中常见形状

| 对象 | 典型形状 | 主要用途 |
| --- | --- | --- |
| token ID | $[B,T]$ | 查 embedding |
| hidden state | $[B,T,d_{\text{model}}]$ | 层间接口 |
| attention query | $[B,n_{\text{head}},T,d_{\text{head}}]$ | 发起读取 |
| attention key/value | $[B,n_{\text{kv}},T,d_{\text{head}}]$ | 提供索引和内容 |
| logits | $[B,T,\lvert\mathcal{V}\rvert]$ | 词表预测 |
| FFN 中间激活 | $[B,T,d_{\text{ff}}]$ | 通道扩展与门控 |

常见配置满足 $d_{\text{model}}=n_{\text{head}}d_{\text{head}}$。MHA 的 KV 头数等于 Query 头数，GQA/MQA 则更少；这一差异改变投影形状与缓存容量，但不改变 token 序列的含义。

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

连续 $[2,3]$ 数组存为 `[0,1,2,3,4,5]`，stride 为 `[3,1]`，逻辑元素 $(1,2)$ 的偏移为 $1\times3+2=5$。转置后的 shape 为 $[3,2]$、stride 为 `[1,3]`，元素 $(2,1)$ 仍指偏移 5，没有搬运。

但是把转置视图按新逻辑行序展开，顺序应为 `[0,3,1,4,2,5]`，与底层存储不同；若下游要求连续的一维向量，就不能仅改 shape 解决。别名、contiguous 和 reshape 的开销由这个差别产生。

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

---

## 参考资料

- PyTorch. [*Tensor Views*](https://docs.pytorch.org/docs/stable/tensor_view.html).
- NumPy. [*Internal organization of NumPy arrays*](https://numpy.org/doc/stable/dev/internals.html).
