# 编译图与执行计划

Eager Runtime 每遇到一个算子便由主机调度一次。图编译、AOT 代码生成与 CUDA Graph 试图把稳定的算子序列转成更大的执行单元，减少 Python、dispatcher 和 kernel launch 开销，并为融合和内存规划提供全局视野。

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

## 参考资料

- PyTorch. *torch.compile Programming Model*.
- NVIDIA. *CUDA Graphs*.
