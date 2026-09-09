# 编译图与执行计划

执行图解决重复程序的组织成本，但「捕获 Python 计算」「生成优化 kernel」「重放 GPU launch」是三个不同层次。本页以这三个接口建立概念基线；具体 `torch.compile` 或 CUDA Graph 限制随版本变化，不提供未经核验的万能配置。

---

## 从一次前向到可复用计划

考虑 `y = activation(xW + b)`。Eager 执行依次调度矩阵乘、加法和激活；图捕获看到生产者—消费者关系后，可以决定将后两步融合，或在后端支持时并入 GEMM epilogue。

复用计划需要条件：输入 dtype、设备、shape 与别名关系是否仍满足捕获时假设。例如一个输出 buffer 被下一步原地修改，编译器就不能在还有使用者时提前复用它。内存规划依据的是生命周期，而不是只看两个 Tensor 元素数相同。

遇到数据相关分支时，也不能假定图会包含所有 Python 路径。系统可能生成带 guard 的特化、产生 graph break，或保留动态控制流；究竟是哪一种，需要看该框架版本的图和编译日志。

CUDA Graph 重放则发生在更低层：若捕获的是同一 launch 序列，可以减少主机重复提交开销，但不会仅因「用了 Graph」就改变矩阵乘算法。输入内容可以更新，地址和执行拓扑等捕获条件仍须满足。图编译与图重放因此可以叠加，而非二选一。

训练第一次运行、首次遇到新 shape 和稳定重放应分别计时。若一次请求只执行一次，而编译成本很高，总时延可能变差；长时间重复同类 shape 才可能摊薄准备成本。

---

## 从计算图到执行

```mermaid
flowchart LR
	E["Eager Program"] --> C["Graph Capture"]
	C --> IR["IR + Shape/Type"]
	IR --> O["Fusion / Layout / Scheduling"]
	O --> K["Generated Kernels"]
	K --> R["Runtime Execution"]
```

编译器需要知道控制流、shape、dtype、设备和别名。输入 shape 变化可能触发新的特化版本或 graph break。动态 batch、变长序列和自定义算子因此是推理编译的重要边界。

---

## 三类能力不要混淆

| 能力 | 主要作用 | 典型限制 |
| --- | --- | --- |
| 图捕获/编译 | 看见跨算子数据流并生成代码 | 动态控制流、未知算子 |
| 内存规划 | 复用生命周期不重叠的 buffer | 动态 shape、别名、外部引用 |
| CUDA Graph | 重放固定的 GPU launch 序列 | 地址、shape、控制流需满足捕获约束 |

CUDA Graph 本身不自动生成更快的算子；它主要降低反复 launch 的 CPU 开销。编译器融合则可能改变 kernel 数量和数据移动，两者可以组合。

---

## 推理中的动态性

在线推理的 active batch、序列长度和 KV block 每轮都可能变化。常见处理方式包括：

- 为若干 batch bucket 捕获图；
- 用 padding 或固定地址表稳定形状；
- 把动态调度留在图外，只捕获模型执行；
- 未命中 bucket 时回退 Eager 路径。

这些设计增加图缓存、显存和测试矩阵。只有 profile 证明 launch 开销位于关键路径时，才应扩展复杂的捕获策略。

---

## CPU 路线

CPU 可使用 `torch.compile`、XLA、ONNX Runtime、oneDNN 或编译器 IR 学习图捕获、融合、shape 特化和内存规划。CUDA Graph 没有 CPU 等价行为，但固定执行计划与线程调度仍可在 CPU Runtime 中研究。

阅读本页前可先看[编译前端](../../../fundamentals/compiler/frontend.md)、[中间表示和后端](../../../fundamentals/compiler/intermediate-and-backend.md)与[VM/JIT Runtime](../../../fundamentals/compiler/vm-jit-runtime.md)。

---

## 参考资料

- PyTorch. [*torch.compile Programming Model*](https://docs.pytorch.org/docs/stable/compile/programming_model.html).
- NVIDIA. *CUDA Graphs*.
