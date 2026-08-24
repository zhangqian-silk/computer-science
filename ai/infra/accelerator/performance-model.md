# 性能模型：计算、带宽与延迟

性能优化的第一步不是选择某个 kernel，而是建立成本模型。给定 workload，需要区分理论工作量、硬件上界、框架开销与实测结果。

<RooflineExplorer />

---

## 四类主要限制

| 限制 | 典型信号 | 常见方向 |
| --- | --- | --- |
| 计算受限 | 计算单元利用率高，数据复用充分 | 更低精度、减少 FLOPs、使用 Tensor Core |
| 内存带宽受限 | 大量 HBM 读写，计算单元等待数据 | 融合、量化、增加 batch、减少中间 Tensor |
| Launch/CPU 受限 | 大量短 kernel、GPU 间隙明显 | 融合、CUDA Graph、异步调度 |
| 通信受限 | Collective 或传输占关键路径 | 拓扑优化、切分调整、计算通信重叠 |

端到端系统还可能受 Tokenization、队列、网络、存储和锁竞争限制。GPU 利用率低并不自动等于需要优化 GPU kernel。

---

## Roofline

算术强度定义为：

$$
I=\frac{\text{FLOPs}}{\text{Bytes transferred}}
$$

若设备峰值计算为 $P_{peak}$，内存带宽为 $BW$，理想性能上界为：

$$
P\le\min(P_{peak},I\times BW)
$$

交点 $P_{peak}/BW$ 称为 ridge point。低于交点的 workload 更可能受带宽限制，高于交点更可能受计算限制。实际 Bytes 必须说明统计的是 HBM、L2 还是主机传输；缓存命中会改变观察层级。

---

## GEMM 与 Decode

矩阵乘 $[M,K]\times[K,N]$ 约执行 $2MKN$ FLOPs。大 $M,N,K$ 可以充分复用权重与激活；Decode 时 $M$ 可能只是当前批次的 token 数，权重被读取后只服务少量 token，算术强度下降。

Continuous Batching 的一个重要作用是把不同请求的当前 token 合并，增大有效 $M$。但 batch 增大也会增加排队、KV 容量和单轮执行时间，不能只追求峰值吞吐。

---

## 延迟不能简单相加平均

一条请求的端到端时间可分为：

$$
T_{e2e}=T_{queue}+T_{tokenize}+T_{prefill}+T_{decode}+T_{network}
$$

Decode 又包含多个串行 iteration。平均 kernel 时间无法直接给出 p99 请求延迟，因为队列、长度分布、抢占和同步会放大尾部。

服务评测至少同时记录：TTFT、TPOT 或 ITL、E2E latency、token throughput、request throughput 和 Goodput。

---

## CPU 路线

CPU 同样可构建 Roofline：使用对应 CPU 的峰值向量计算能力和内存带宽，配合 BLAS、perf、线程数与 NUMA 绑定测量。CPU 实验可以验证 batch、blocking、量化和内存带宽趋势，但不能把绝对吞吐外推到 GPU。

## 参考文献

- Williams, S. et al. (2009). *Roofline: An Insightful Visual Performance Model for Multicore Architectures*.
- Patterson, D. and Hennessy, J. *Computer Organization and Design*.
